//! Hot prefix cache commit, SSD-first flush, write-through, and media-chain tests.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
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
const SlotRingCps = pc.SlotRingCps;
const MediaSpan = pc.MediaSpan;
const chainKey = pc.chainKey;
const truncatedMediaKey = pc.truncatedMediaKey;
const MIN_CANCELLED_COMMIT_TOKENS = pc.MIN_CANCELLED_COMMIT_TOKENS;
const RING_CHECKPOINT_MAX = pc.RING_CHECKPOINT_MAX;

const testFillCache = util.testFillCache;
const ringFill = util.ringFill;
const ringRowBytes = util.ringRowBytes;
const ringTurn = util.ringTurn;
const expectRingContinuesCold = util.expectRingContinuesCold;
const pcSsmVal = util.pcSsmVal;
const pcBuildHybrid = util.pcBuildHybrid;
const pcFreeHybrid = util.pcFreeHybrid;
const pcEmptySsm = util.pcEmptySsm;
const pcExpectRowsEqual = util.pcExpectRowsEqual;
const commitUnflushed = util.commitUnflushed;
const pcAppendCheckedOut = util.pcAppendCheckedOut;
const mediaTurnTokens = util.mediaTurnTokens;

// A new image changes the KV only when its dynamic placeholder rows are
// forwarded. The text prefix before that boundary remains valid even though
// the media hash changes. Hybrid models must restore the last checkpoint at
// or before the boundary, never a later checkpoint whose SSM state has seen
// the old pixels.
test "HotPrefixCache: hybrid lookup reuses only the prefix before changed media" {
    const s = mlx.gpuStream();
    const media_start: usize = 8;
    const cached_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 900, 900, 20, 21 };
    const lookup_tokens = cached_tokens;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var source_cache = try KVCache.init(testing.allocator, 3);
    defer source_cache.deinit();
    try testFillCache(&source_cache, s, 3, cached_tokens.len);
    var source_ssm = pcBuildHybrid(s, 123.0, 456.0);
    defer pcFreeHybrid(&source_ssm);
    const checkpoints = try testing.allocator.alloc(SSMCheckpoint, 2);
    checkpoints[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start, s);
    checkpoints[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start + 2, s);
    _ = try hc.commitWithMediaState(
        &source_cache,
        &cached_tokens,
        false,
        0x1111,
        0,
        media_start,
        checkpoints,
        null,
        null,
        cached_tokens.len,
    );

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestoreWithMedia(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        0x2222,
        media_start,
        &.{},
        null,
        null,
        null,
        false,
    );

    try testing.expectEqual(media_start, result.matched);
    try testing.expectEqual(media_start, target_cache.step);
    try testing.expectEqual(media_start, moe_off);
    try testing.expectEqual(@as(f32, 123.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

// A vision turn can move the current image span when the same image becomes
// conversation history. The newest entry then has the longest raw token match
// (ending exactly at the old image boundary), but its first SSM checkpoint can
// sit just AFTER that boundary. An older entry for the same pixels may have a
// slightly shorter token match with a usable checkpoint. Picking by raw token
// match alone turns that recoverable case into a full cold prefill.
test "HotPrefixCache: hybrid lookup falls back to the best restorable RAM entry" {
    const s = mlx.gpuStream();
    const vision_key: u64 = 0xdecaf;
    const older_tokens = [_]u32{ 1, 2, 3, 4, 5, 90, 91, 92, 93, 94 };
    const newer_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 80, 81, 82 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 70, 71, 72 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var older_cache = try KVCache.init(testing.allocator, 3);
    defer older_cache.deinit();
    try testFillCache(&older_cache, s, 3, older_tokens.len);
    var older_ssm = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&older_ssm);
    const older_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    older_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &older_ssm, 4, s);
    _ = try hc.commitWithState(&older_cache, &older_tokens, false, vision_key, older_cps, null, null);

    var newer_cache = try KVCache.init(testing.allocator, 3);
    defer newer_cache.deinit();
    try testFillCache(&newer_cache, s, 3, newer_tokens.len);
    var newer_ssm = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&newer_ssm);
    const newer_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    // The raw match with this entry is 7, so this checkpoint cannot restore it.
    newer_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &newer_ssm, 8, s);
    _ = try hc.commitWithState(&newer_cache, &newer_tokens, false, vision_key, newer_cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestore(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        vision_key,
        null,
        null,
    );

    // The newer 7-token raw match is unusable; the older checkpoint at 4 is
    // still vastly better than a cold prefill and must win the hybrid lookup.
    try testing.expectEqual(@as(usize, 4), result.matched);
    try testing.expectEqual(@as(usize, 4), target_cache.step);
    try testing.expectEqual(@as(usize, 4), moe_off);
    try testing.expectEqual(@as(f32, 100.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

// Bar: an image turn commits under the PIXEL key, so the pre-media checkpoints
// it can only get from the text entry must survive that entry's eviction.
test "HotPrefixCache: a text turn after image turns restores the pre-media prefix" {
    const s = mlx.gpuStream();
    const media_start: usize = 12;
    const image_key: u64 = 0xF00D;

    // Turn N: the text conversation, with checkpoints spanning the prefix.
    const text_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    // Turn N+1: the same conversation plus an image at `media_start`.
    const image_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 900, 900, 900, 900, 61, 62, 63, 64 };
    // Turn N+2: a text-only turn. The image message is history now, so the
    // token stream still agrees far past the boundary; only the pixel key
    // caps the reusable prefix at `media_start`.
    const later_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 900, 900, 900, 900, 61, 62, 71, 72 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
    defer hc.deinit();

    var text_cache = try KVCache.init(testing.allocator, 3);
    defer text_cache.deinit();
    try testFillCache(&text_cache, s, 3, text_tokens.len);
    var text_ssm = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&text_ssm);
    const text_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    text_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 4, s);
    text_cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 8, s);
    text_cps[2] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 10, s);
    _ = try hc.commitWithMediaState(&text_cache, &text_tokens, false, 0, 0, null, text_cps, null, null, text_tokens.len);

    // The image turn restores the pre-media text prefix (cross-key, capped
    // at the boundary) and prefills the rest.
    var image_cache = try KVCache.init(testing.allocator, 3);
    defer image_cache.deinit();
    var image_ssm = pcEmptySsm();
    defer pcFreeHybrid(&image_ssm);
    var image_off: usize = 0;
    const restored = try hc.lookupAndRestoreWithMedia(
        &image_cache,
        &image_off,
        &image_ssm,
        s,
        &image_tokens,
        false,
        image_key,
        media_start,
        &.{},
        null,
        null,
        null,
        false,
    );
    try testing.expectEqual(@as(usize, 10), restored.matched);

    // ... and commits under the pixel key. Its OWN prefill only reached
    // positions past the image, so its own checkpoints all sit above the
    // boundary; the count cap evicts the text entry.
    try testFillCache(&image_cache, s, 3, image_tokens.len);
    var image_state = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&image_state);
    const image_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    image_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &image_state, 16, s);
    image_cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &image_state, 18, s);
    _ = try hc.commitWithMediaState(&image_cache, &image_tokens, false, image_key, 0, media_start, image_cps, null, null, image_tokens.len);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());

    // The next text-only turn must still restore the pre-media prefix: the
    // rows below `media_start` are pure text and pixel-independent.
    var later_cache = try KVCache.init(testing.allocator, 3);
    defer later_cache.deinit();
    var later_ssm = pcEmptySsm();
    defer pcFreeHybrid(&later_ssm);
    var later_off: usize = 0;
    const result = try hc.lookupAndRestoreWithMedia(
        &later_cache,
        &later_off,
        &later_ssm,
        s,
        &later_tokens,
        false,
        0,
        null,
        &.{},
        null,
        null,
        null,
        false,
    );
    try testing.expectEqual(@as(usize, 10), result.matched);
    try testing.expectEqual(@as(usize, 10), later_cache.step);
    try testing.expectEqual(@as(usize, 10), later_off);
    try testing.expectEqual(@as(f32, 100.0), pcSsmVal(later_ssm[0].conv_state, 0, s));
}

// Class guard: no retention policy may thin away the highest checkpoint at or
// below an entry's media boundary.
test "HotPrefixCache: thinning spreads checkpoints across the whole record, media or not" {
    // The thinning used to protect the highest checkpoint at or below the first media row,
    // because the SSD record ended there. A record now persists whole, so that rule only sacrificed
    // the checkpoints ABOVE the image -- the very ones the next image-bearing turn restores from.
    const s = mlx.gpuStream();
    const media_start: usize = 8;

    var tokens: [24]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);

    var kept_positions: [2][3]usize = undefined;
    for ([_]bool{ true, false }, 0..) |media, arm| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
        hc.ssm_checkpoint_max = 3;
        defer hc.deinit();

        var ssm = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&ssm);

        var c1 = try KVCache.init(testing.allocator, 3);
        defer c1.deinit();
        try testFillCache(&c1, s, 3, 16);
        const first = try testing.allocator.alloc(SSMCheckpoint, 3);
        for (first, 0..) |*c, i| c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &ssm, 2 + i * 2, s);
        _ = try hc.commitWithMediaState(&c1, tokens[0..16], false, if (media) 0xF00D else 0, 0, if (media) media_start else null, first, null, null, 16);

        // The next turn extends the same conversation, so the commit replaces the
        // entry and merges: seven checkpoints thinned down to the cap of three.
        var c2 = try KVCache.init(testing.allocator, 3);
        defer c2.deinit();
        try testFillCache(&c2, s, 3, 24);
        const second = try testing.allocator.alloc(SSMCheckpoint, 4);
        for (second, 0..) |*c, i| c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &ssm, 12 + i * 2, s);
        _ = try hc.commitWithMediaState(&c2, &tokens, false, if (media) 0xF00D else 0, 0, if (media) media_start else null, second, null, null, tokens.len);

        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        const kept = hc.entries.items[0].ssm_checkpoints.?;
        try testing.expectEqual(@as(usize, 3), kept.len);
        for (kept, &kept_positions[arm]) |*cp, *out| out.* = cp.pos;
        // Restore points above the image survive, which is what the SSD record needs.
        try testing.expect(kept[kept.len - 1].pos > media_start);
    }
    // The media boundary changes nothing about which checkpoints are kept.
    try testing.expectEqualSlices(usize, &kept_positions[1], &kept_positions[0]);
}

test "HotPrefixCache: hybrid SSM state restores from the SSD tier across a restart" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit (KV + two SSM checkpoints) through the RAM
    // path, then the post-markFinished flush the scheduler makes.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
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
        // commitWithSsm takes ownership of `cps`.
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("restart"): fresh RAM cache + fresh tier over the same root.
    // A hybrid lookup must restore BOTH KV and SSM state from disk at the
    // highest checkpoint ≤ the match.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        // Highest checkpoint ≤ 600 is 512 — never a full match on hybrid.
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
        try testing.expectEqual(@as(usize, 512), moe_off);
        // SSM state at pos 512 installed (conv base 300 / ssm base 700).
        try testing.expect(ssm2[0].initialized);
        try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 700.0), pcSsmVal(ssm2[0].ssm_state, 0, s));
        try testing.expect(ssm2[1].ssm_state.ctx == null);
        try testing.expect(!ssm2[2].initialized);

        // Diverged tail: shares the first 400 tokens → clamps to the largest
        // checkpoint ≤ 400, which is 256.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 3);
        defer cache3.deinit();
        var ssm3 = pcEmptySsm();
        defer pcFreeHybrid(&ssm3);
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, &ssm3, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 256), res3.matched);
        try testing.expectEqual(@as(usize, 256), cache3.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm3[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 500.0), pcSsmVal(ssm3[0].ssm_state, 0, s));
    }
}

test "HotPrefixCache: chunk-heavy hybrid flush still lands its SSM checkpoints" {
    // Live 2026-09-07: checkpoints were written AFTER the chunks from the
    // REMAINING per-flush budget — a turn appending ≥ the cap in chunks
    // (cancel-salvage retries: +16 chunks ≈ 544 MB vs a 512 MB cap) left
    // ZERO checkpoint budget every time, and the same-token catch-up path
    // never fires for a growing conversation. Every entry of the Sep-4
    // disk wave landed KV-only: unrestorable, gigabytes of dead chunks.
    // Checkpoints must come off the TOP of the budget, not the bottom.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit + a flush budget tight enough that the KV
    // chunks alone would consume it (5 chunks ≈ 24 KB each vs a 2-chunk
    // cap) — the exact starvation shape, scaled down.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc.deinit();
        hc.disk.?.max_flush_bytes = 2 * 128 * 3 * 2 * 8 * 4; // two KV chunks

        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        hc.flushPendingDisk(s);
    }

    // Session 2 (RAM empty): the hybrid lookup must restore from the disk
    // entry's highest checkpoint ≤ its flushed kv_len (256 of 5 chunks) —
    // not cold-miss on a checkpoint-less entry.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 256), res.matched);
        try testing.expectEqual(@as(usize, 256), cache2.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm2[0].conv_state, 0, s));
    }
}

test "HotPrefixCache: under SSD-first a decline rides pending_disk, never a second synchronous spill" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill-ssdfirst", 0, 128);
    defer hc.deinit();
    const cap_before = hc.disk.?.max_flush_bytes;

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    // The capture at commit time is the one record; nothing lands until the flush.
    try testing.expect(hc.pending_disk != null);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expectEqual(cap_before, hc.disk.?.max_flush_bytes);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "HotPrefixCache: with an SSD tier the RAM entry keeps only its newest two checkpoints" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    for ([_]bool{ true, false }) |with_disk| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        if (with_disk) {
            hc.ssd_first = true;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ram-keep", 0, 128);
            hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
        }

        var c1 = try KVCache.init(testing.allocator, 3);
        defer c1.deinit();
        try testFillCache(&c1, s, 3, 700);
        const cps1 = try testing.allocator.alloc(SSMCheckpoint, 5);
        for (cps1, 0..) |*c, i| c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], 512 + 32 * i, s);
        _ = try hc.commitWithSsm(&c1, tokens[0..700], false, cps1, null, null);
        const kept1 = hc.entries.items[0].ssm_checkpoints.?;
        // Without a tier there is nowhere else to restore from: today's behaviour.
        try testing.expectEqual(@as(usize, if (with_disk) 2 else 5), kept1.len);
        try testing.expectEqual(@as(usize, 640), kept1[kept1.len - 1].pos);
        try testing.expectEqual(@as(usize, if (with_disk) 608 else 512), kept1[0].pos);

        // The next turn extends the entry and brings three more: still the newest two.
        var c2 = try KVCache.init(testing.allocator, 3);
        defer c2.deinit();
        try testFillCache(&c2, s, 3, 900);
        const cps2 = try testing.allocator.alloc(SSMCheckpoint, 3);
        for (cps2, 0..) |*c, i| c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[5 + i], 704 + 32 * i, s);
        _ = try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        const kept2 = hc.entries.items[0].ssm_checkpoints.?;
        try testing.expectEqual(@as(usize, if (with_disk) 2 else 8), kept2.len);
        try testing.expectEqual(@as(usize, 768), kept2[kept2.len - 1].pos);
        var bytes: u64 = 0;
        for (kept2) |*cp| bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(bytes, hc.entries.items[0].ssm_bytes);

        if (with_disk) {
            // Everything RAM let go is on the SSD record, to restore from there.
            hc.flushPendingDisk(s);
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
            try testing.expectEqual(@as(usize, 8), hc.disk.?.entries.items[0].ssm_positions.len);
        }
    }
}

test "HotPrefixCache: a byte-budget trim never lands on a released checkpoint, which this entry cannot restore from" {
    const cps = [_]SSMCheckpoint{
        .{ .pos = 1024, .layers = &.{}, .released_bytes = 100 },
        .{ .pos = 2048, .layers = &.{} },
    };
    const positions = [_]usize{ 1024, 2048 };
    const bytes = [_]u64{ 0, 100 };
    // 1024 is the highest position under the limit, and a placeholder: the entry declines to trim there.
    try testing.expectEqual(@as(?usize, null), HotPrefixCache.trimLenForBudgetPure(10_000_000, 1500, 1000, &positions, &bytes, .min_span_recency, &cps));
    const held = [_]SSMCheckpoint{ .{ .pos = 1024, .layers = &.{} }, .{ .pos = 2048, .layers = &.{} } };
    try testing.expectEqual(@as(?usize, 1024), HotPrefixCache.trimLenForBudgetPure(10_000_000, 1500, 1000, &positions, &bytes, .min_span_recency, &held));
}

test "HotPrefixCache: a budget-declined candidate spills to the SSD tier" {
    // Live 2026-09-07: an oversized decline discarded the cancelled
    // prefill's work entirely — under a 1 GB hot budget a 122k session's
    // 76-82k-token forwarded prefixes were thrown away on every retry
    // while a 40 GB SSD tier sat idle. The decline must OFFER the
    // candidate to the disk tier instead of discarding it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A RAM budget too small for even the commit floor forces the decline.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 0), hc.entryCount()); // RAM keeps nothing
        // The candidate's work must live on the SSD tier.
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The next request (fresh RAM, generous budget) restores from SSD
    // instead of cold-prefilling the whole prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: with RAM retention on, the SSD tier gets the whole entry the RAM budget trims, after the response" {
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
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 40 * 1024);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ram-trim", 0, 128);
        hc.disk.?.enableBackgroundWriter();
        hc.disk.?.max_flush_bytes = 1; // one chunk per piece

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .ok and st.ok < 600);
        // Nothing is written on the response path.
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
    }
    // A fresh process restores the whole prompt from the SSD tier.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ram-trim", 0, 128);
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "HotPrefixCache: a commit that checked-out residents decline still reaches the SSD tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const big: u64 = 1 << 30;
    // The count cap, then the byte cap, held by an entry a live slot checked out.
    const Cap = struct { entries: u32, bytes: u64, fp: []const u8 };
    for ([_]Cap{ .{ .entries = 1, .bytes = 0, .fp = "fp-held-count" }, .{ .entries = 8, .bytes = big, .fp = "fp-held-bytes" } }) |cap| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, cap.entries, cap.bytes);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, cap.fp, 0, 128);
        hc.disk.?.enableBackgroundWriter();
        try pcAppendCheckedOut(&hc, &.{ 1, 2 }, cap.bytes, 1);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        try testing.expect(try hc.commit(&cache, &tokens, false) == .declined);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
    }
}

test "HotPrefixCache: a fork's SSD entry keeps the checkpoints its donor holds in RAM" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    const a_tokens = prompt ++ [_]u32{ 200, 201 };
    const b_tokens = prompt ++ [_]u32{ 210, 211 };
    var srcs: [4][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-fork-cps", 0, 128);
    hc.disk.?.enableBackgroundWriter();

    // A: checkpoints at 256 and 512 inside the shared prompt, 601 inside its own reply.
    var a_cache = try KVCache.init(testing.allocator, 3);
    defer a_cache.deinit();
    try testFillCache(&a_cache, s, 3, a_tokens.len);
    const a_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    a_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[0], 256, s);
    a_cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[1], 512, s);
    a_cps[2] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[2], 601, s);
    _ = try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);
    hc.flushPendingDisk(s);

    // B restored A's prompt and checkpointed only its own prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 600, s);
    _ = try hc.commitWithState(&b_cache, &b_tokens, false, 0, b_cps, null, null);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();

    const d = &hc.disk.?;
    const b_disk = for (d.entries.items) |*e| {
        if (e.tokens.len > 600 and e.tokens[600] == 210) break e;
    } else return error.TestExpectedForkEntry;
    // A's RAM entry kept its newest two (512, 601): 256 stays only in A's own SSD entry.
    try testing.expectEqualSlices(u32, &.{ 512, 600 }, b_disk.ssm_positions);
}

test "a fork's SSD entry keeps its donor's ring checkpoint, so a restart restores there with the donor gone" {
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
    const tail: u32 = 100;
    const reminder: u32 = 330;
    const fork_prompt = prompt + tail + reminder;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fork-disk", 0, 128);
        hc.disk.?.enableBackgroundWriter();
        var donor = try KVCache.init(testing.allocator, n_layers);
        defer donor.deinit();
        const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
        _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });
        hc.flushPendingDisk(s);

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
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    }
    // The donor's SSD entry goes; the fork's linked chunks outlive it.
    kv_disk_fs.testDeleteTreeAt(io, base, "fp-ring-fork-disk/e1");

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc2.deinit();
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fork-disk", 0, 128);
    try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
    var next: [prompt + tail + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + tail], toks[0 .. prompt + tail]);
    for (next[prompt + tail ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc2.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

/// A ringed-trunk turn whose slot took a ring checkpoint at each of `at` (ascending, the last the
/// prompt end, the rest marks), then `reply` rows.
fn ringTurnAt(live: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, at: []const u32, reply: u32) !SlotRingCps {
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    errdefer cps.deinit();
    var from: u32 = 0;
    for (at, 0..) |p, i| {
        try ringFill(live, s, n_layers, window, from, p, 64);
        from = p;
        const cp = (try live.ringCheckpoint(p, s)).?;
        if (i + 1 == at.len) {
            cps.prompt_end = cp;
        } else {
            cps.marks[cps.n_marks] = cp;
            cps.n_marks += 1;
        }
    }
    try ringFill(live, s, n_layers, window, from, from + reply, 16);
    return cps;
}

fn ringCpSteps(hc: *const HotPrefixCache, entry: usize, out: *[RING_CHECKPOINT_MAX]usize) []const usize {
    const cps = hc.entries.items[entry].ring_cps orelse return out[0..0];
    for (cps, 0..) |cp, i| out[i] = cp.step;
    return out[0..cps.len];
}

test "RAM and SSD: a flushed ringed entry keeps its newest two ring checkpoints, and an older point restores its exact rows from SSD" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const reply: u32 = 600;

    var toks: [700 + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-keep", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, reply);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);

    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
    const before = hc.current_kv_bytes;
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqualSlices(usize, &.{ 600, 700 }, ringCpSteps(&hc, 0, &steps));

    // The bill follows the storage: two checkpoints' bytes left it, and the cache total agrees.
    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys.array) + ringRowBytes(e.values.array));
    }
    try testing.expectEqual(before - 2 * cp_bytes, hc.current_kv_bytes);
    try testing.expectEqual(hc.entries.items[0].kv_bytes, hc.current_kv_bytes);

    // Between two points no RAM checkpoint reaches: the SSD tier's file at 450 serves it.
    var next: [520]u32 = undefined;
    @memcpy(next[0..460], toks[0..460]);
    for (next[460..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, 450), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, 450, next.len);

    // A newest point still restores from RAM.
    var newest: [660]u32 = undefined;
    @memcpy(newest[0..610], toks[0..610]);
    for (newest[610..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var from_ram = try KVCache.init(testing.allocator, n_layers);
    defer from_ram.deinit();
    from_ram.setSwaRing(window);
    const ram_hit = try hc.lookupAndRestore(&from_ram, &moe_off, null, s, &newest, false, 0, null, null);
    try testing.expectEqual(@as(usize, 600), ram_hit.matched);
    try expectRingContinuesCold(&from_ram, s, n_layers, window, 600, newest.len);
}

test "RAM tier without an SSD tier keeps every ring checkpoint" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
}

test "RAM and SSD: a ring checkpoint whose write failed stays in RAM" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fail", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    hc.disk.?.writer.?.injectFailure("r0000450", .submit);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    // The entry the failure invalidated holds no restore point, so RAM gives none up.
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
}

test "a ring file past the rows its entry persisted is not a held restore point" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-kvlen", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();

    // A flush the byte cap stopped leaves ring files above the rows the entry holds.
    hc.disk.?.entries.items[0].kv_len = 500;
    var held: [2]bool = undefined;
    hc.disk.?.ringPointsHeld(&toks, false, live.config, &.{ 450, 600 }, &held, .{});
    try testing.expectEqualSlices(bool, &.{ true, false }, &held);
}

test "RAM and SSD: a fork inherits the donor's newest checkpoints, and the donor's older points stay on its SSD entry" {
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
    const fork_prompt: u32 = prompt + 100 + 330;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-donor", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cps = try ringTurnAt(&donor, s, n_layers, window, &.{ 300, 450, 600, prompt }, reply);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, donor_cps);
    hc.flushPendingDisk(s);
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 600, prompt }, ringCpSteps(&hc, 0, &steps));

    // The side request restores off the donor and appends its own tail.
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
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    const fork_idx: usize = for (hc.entries.items, 0..) |*e, i| {
        if (e.tokens.len == toks.len) break i;
    } else return error.TestUnexpectedResult;
    // It inherited the donor's 600 and prompt end, then released the older of its three.
    try testing.expectEqualSlices(usize, &.{ prompt, fork_prompt }, ringCpSteps(&hc, fork_idx, &steps));

    // An unrelated request under the count cap evicts the donor from RAM.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    for (hc.entries.items) |*e| try testing.expect(e.tokens.len != donor_toks.len);

    // A turn diverging at the inherited 600 restores there from the fork's SSD entry.
    var at_600: [660]u32 = undefined;
    @memcpy(at_600[0..610], toks[0..610]);
    for (at_600[610..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var a = try KVCache.init(testing.allocator, n_layers);
    defer a.deinit();
    a.setSwaRing(window);
    const hit_a = try hc.lookupAndRestore(&a, &moe_off, null, s, &at_600, false, 0, null, null);
    try testing.expectEqual(@as(usize, 600), hit_a.matched);
    try expectRingContinuesCold(&a, s, n_layers, window, 600, at_600.len);

    // One diverging at the donor's released 450 restores from the donor's SSD entry.
    var at_450: [520]u32 = undefined;
    @memcpy(at_450[0..460], toks[0..460]);
    for (at_450[460..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var b = try KVCache.init(testing.allocator, n_layers);
    defer b.deinit();
    b.setSwaRing(window);
    const hit_b = try hc.lookupAndRestore(&b, &moe_off, null, s, &at_450, false, 0, null, null);
    try testing.expectEqual(@as(usize, 450), hit_b.matched);
    try expectRingContinuesCold(&b, s, n_layers, window, 450, at_450.len);
}

test "HotPrefixCache: a decline-spill is not bounded by the per-flush byte cap" {
    // The per-flush cap bounds the stall a LIVE next request pays after a
    // response. A decline-spill runs on a request whose client is already
    // gone — capping it (live 2026-09-07: 512 MB ≈ 13k tokens banked per
    // retry) stranded most of each retry's work while ~35k tokens were
    // recomputed every time. The spill owes no latency; the tier's byte
    // budget + LRU is the real bound.
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
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc.deinit();
        // A cap of two KV chunks — the exact starvation shape, scaled down.
        hc.disk.?.max_flush_bytes = 2 * 128 * 2 * 2 * 8 * 4;

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The WHOLE candidate must be restorable, not just the cap's worth.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc2.deinit();
        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: hybrid disk restore ranks entries by restorable checkpoint, not raw length" {
    // The RAM tier learned this the hard way (#312): a longer raw match
    // whose checkpoints sit past the divergence restores nothing. The disk
    // tier's hybrid arm used bestMatch (ranked by usable length) and then
    // took that ONE entry's checkpoint — an entry with a high usable length
    // but low checkpoints shadowed a shorter entry with a higher
    // restorable position (live 2026-09-07: the growing conversation entry
    // would have shadowed the old entry's cp@51200 with cp@49152).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Entry A: shares 550 tokens with the prompt (longer) but its only
    // checkpoint is at 256. Diverges from the prompt at 550.
    var a_tokens: [600]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = if (i < 550) @intCast(i + 7) else @intCast(i + 700);
    // Entry B: shares only 512 tokens (shorter) but carries a checkpoint
    // at 512. Diverges from the prompt at 512 — so neither entry is a
    // prefix of the other and the flush cannot merge them.
    var b_tokens: [520]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_a[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        _ = try hc.commitWithSsm(&cache, &a_tokens, false, cps_a, null, null);
        hc.flushPendingDisk(s);
        var cache_b = try KVCache.init(testing.allocator, 3);
        defer cache_b.deinit();
        try testFillCache(&cache_b, s, 3, 600);
        const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_b[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache_b, &b_tokens, false, cps_b, null, null);
        hc.flushPendingDisk(s);
    }

    // RAM empty; the hybrid lookup must restore from B's cp@512, not A's
    // cp@256 — even though A's raw usable length (600) beats B's (512).
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 2), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
    }
}

test "HotPrefixCache: a hybrid SSD restore hands the slot rows it owns, a RAM share does not" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var entry_tokens: [520]u32 = undefined;
    for (&entry_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    var src = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&src);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-own", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 520);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache, &entry_tokens, false, cps, null, null);
        hc.flushPendingDisk(s);

        var slot = try KVCache.init(testing.allocator, 3);
        defer slot.deinit();
        var ssm = pcEmptySsm();
        defer pcFreeHybrid(&ssm);
        var moe_off: usize = 0;
        const ram = try hc.lookupAndRestore(&slot, &moe_off, &ssm, s, &tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, 512), ram.matched);
        // The entry keeps the buffers: the first append copies them, so nothing is credited.
        try testing.expect(!ram.ownsRestoredRows());
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-own", 0, 128);
    defer hc2.deinit();
    var slot2 = try KVCache.init(testing.allocator, 3);
    defer slot2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off2: usize = 0;
    const disk = try hc2.lookupAndRestore(&slot2, &moe_off2, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), disk.matched);
    try testing.expect(disk.ownsRestoredRows());
}

test "HotPrefixCache: a hybrid disk restore adopts the spec sidecar of the entry it restored" {
    // The trunk verifies, so no wrong token ships — but an MTP history describing another
    // entry's tokens collapses acceptance.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // A wins the raw match (550) but checkpoints at 256; B matches 512 and checkpoints there.
    var a_tokens: [600]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = if (i < 550) @intCast(i + 7) else @intCast(i + 700);
    var b_tokens: [520]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-spec", 0, 128);
        defer hc.deinit();
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);

        var cache_a = try KVCache.init(testing.allocator, 3);
        defer cache_a.deinit();
        try testFillCache(&cache_a, s, 3, 600);
        var hist_a = try KVCache.init(testing.allocator, 1);
        defer hist_a.deinit();
        try testFillCache(&hist_a, s, 1, 600);
        const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_a[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        _ = try hc.commitWithSsm(&cache_a, &a_tokens, false, cps_a, null, .{ .cache = &hist_a, .base_pos = 100 });
        hc.flushPendingDisk(s);

        var cache_b = try KVCache.init(testing.allocator, 3);
        defer cache_b.deinit();
        try testFillCache(&cache_b, s, 3, 600);
        var hist_b = try KVCache.init(testing.allocator, 1);
        defer hist_b.deinit();
        try testFillCache(&hist_b, s, 1, 520);
        const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_b[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache_b, &b_tokens, false, cps_b, null, .{ .cache = &hist_b, .base_pos = 0 });
        hc.flushPendingDisk(s);
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-spec", 0, 128);
    defer hc2.deinit();
    try testing.expectEqual(@as(usize, 2), hc2.disk.?.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var mtp_dst = try KVCache.init(testing.allocator, 1);
    defer mtp_dst.deinit();
    var moe_off: usize = 0;
    var mbase: usize = 99;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, .{ .cache = &mtp_dst, .base_pos = &mbase });
    try testing.expectEqual(@as(usize, 512), res.matched);
    // B's history, not A's (which would adopt at base 100 for 412 rows).
    try testing.expectEqual(@as(?usize, 0), res.mtp_base);
    try testing.expectEqual(@as(usize, 0), mbase);
    try testing.expectEqual(@as(usize, 512), mtp_dst.step);
}

test "HotPrefixCache: hybrid RAM match at least as good as disk skips the SSD read" {
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
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hybskip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry's checkpoint at 512 ties the disk's, so
    // the disk advantage gate fails and the SSD read is skipped (counter
    // unchanged) — the RAM path serves the restore.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), res.matched);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
    // RAM restore installed the SSM state just the same.
    try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
}

// Regression: an entry that is EXTENDED in place must not accumulate SSM
// checkpoints without bound, and bounding them must not collapse the survivors
// onto the end of the prompt. `generate.zig` caps what a single prefill
// captures, but the replace path merges the previous entry's checkpoints with
// this turn's, and nothing re-applied a cap to the merged list — so an agent
// conversation gained one checkpoint per turn forever. Observed on
// A 36-GDN-layer trunk: 31237 MB of SSM state in ONE entry under
// `--ssm-checkpoint-max 8`, which starved the prompt-admission check.
//
// Capping oldest-first fixes the size but leaves every survivor near the end,
// and a request diverging earlier then pays a full cold prefill
// ("hybrid miss (no checkpoint <= 16382 of 178509)", 415 s). So the cap thins
// the interior and keeps a spread.
test "HotPrefixCache: replace path bounds SSM checkpoints and keeps them spread" {
    const s = mlx.gpuStream();

    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 3);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssm_checkpoint_max = 4;
    defer hc.deinit();

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    // Turn 1: four checkpoints at 100..400 over a 450-token prefix.
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 450);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps1, 0..) |*c, i| {
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }
    _ = try hc.commitWithSsm(&c1, tokens[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 4), hc.entries.items[0].ssm_checkpoints.?.len);

    // Turn 2 extends that exact prefix and brings four more at 500..800.
    // Merged that is eight against a cap of four.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps2, 0..) |*c, i| {
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
    }
    _ = try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);

    // Extended, not appended: still one entry, and the cap holds.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const kept = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 4), kept.len);

    // The first and the newest always survive, the interior is thinned to keep
    // coverage. Oldest-first would have left 500/600/700/800, and a request
    // matching at 150 would then have nothing at or below it to restore from.
    try testing.expectEqual(@as(usize, 100), kept[0].pos);
    try testing.expectEqual(@as(usize, 300), kept[1].pos);
    try testing.expectEqual(@as(usize, 500), kept[2].pos);
    try testing.expectEqual(@as(usize, 800), kept[3].pos);
}

test "SSD-first: the disk flush carries the full prefix while RAM keeps a trim" {
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    try testing.expect(row_bytes > 0);
    const budget: u64 = row_bytes * 768;

    // Arm A (ssd_first on): RAM trims, the disk entry covers the full prompt.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-on", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        try testing.expect(hc.entries.items[0].tokens.len < tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
        try testing.expectEqual(@as(usize, tokens.len), hc.disk.?.entries.items[0].tokens.len);
        try testing.expect(hc.pending_disk == null);
    }

    // Arm B (ssd_first off): the disk copy is exactly what RAM retained.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-off", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expect(hc.disk.?.entries.items[0].kv_len < tokens.len);
    }
}

test "SSD-first companion: a restore adopts the entry's buffer when its capacity suffices" {
    // A grow is not in place, so a restore must land in the donor's buffer (which already
    // carries the previous turn's reservation) rather than allocate the entry's whole size.
    const s = mlx.gpuStream();
    const Grows = &trunk_mod.KVCache.kv_cap_buf_grows;

    // Turn 1: a reserved cache grows once, to the reservation.
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(4096);
    const g0 = Grows.*;
    try testFillCache(&donor, s, 1, 600);
    try testing.expectEqual(@as(usize, 1), Grows.* - g0);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try hc.commit(&donor, &tokens, false);

    // Turn 2: restore into a fresh slot cache that reserves the same length: nothing allocates.
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&slot, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    slot.reserve(4096);
    const g1 = Grows.*;
    try testFillCache(&slot, s, 1, 8); // the diverged tail
    try testing.expectEqual(@as(usize, 0), Grows.* - g1);

    // A reservation is not retroactive: reserving more than the donor holds still allocates nothing.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    _ = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &tokens, false, 0, null, null);
    slot2.reserve(65536);
    const g2 = Grows.*;
    try testFillCache(&slot2, s, 1, 8);
    try testing.expectEqual(@as(usize, 0), Grows.* - g2);

    // Negative arm: writing past the donor's capacity does grow, exactly once.
    const g3 = Grows.*;
    try testFillCache(&slot2, s, 1, 4096);
    try testing.expect(Grows.* - g3 >= 1);
}

test "SSD-first: an idle entry spills to disk and leaves RAM; the active session stays" {
    // RAM floors at one entry under SSD-first: everything but the active session goes to the
    // SSD, and only once its copy is complete.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Arm A: SSD-first spills the idle session.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());

        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
        // ...and A is still served, from disk.
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
        hc.disk.?.drainWriter();
        var back = try KVCache.init(testing.allocator, 1);
        defer back.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tokens_a, false, 0, null, null);
        // A restore always leaves the last token to forward.
        try testing.expectEqual(@as(usize, 599), res.matched);
    }

    // Arm B: every other arch keeps both entries resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-nospill", 0, 128);
        defer hc.deinit();
        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: an in-flight write does not stall the tick — the entry is re-checked next pass" {
    // The durability check must not drain the writer on the inference thread: an entry whose
    // files are still staged is not evictable on this pass, and the next pass evicts.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-inflight", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    // Generous allowance: this test is about the write state, not the cap.
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    // Held before the commits: the second commit flushes the first entry, which a running writer would finish.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    hc.spillIdleEntries(s);
    // It returned, the index knows the entry, and RAM still holds it.
    try testing.expect(hc.disk.?.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Let the writer run and drop the allowance: the next pass evicts.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);
}

test "SSD-first: the idle ALLOWANCE bounds eviction, not the fact of being idle" {
    // The spill used to evict every non-newest entry on every finish, ignoring
    // `--prefix-cache-mem`; writing stays unconditional, evicting is what the allowance bounds.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Two sessions, an allowance that covers the idle one: both stay.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for one idle entry
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // ...and the write still happened.
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // A third session past the allowance: the oldest idle entry goes, and only that one.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow3", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for ONE of the two idle
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
        var saw_b = false;
        var saw_c = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_b)) saw_b = true;
            if (std.mem.eql(u32, e.tokens, &tok_c)) saw_c = true;
        }
        try testing.expect(saw_b and saw_c);
    }

    // Allowance 0 means what it says: nothing idle stays resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow0", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }
}

test "SSD-first: the allowance is a HARD cap, shed in two tiers (durable first)" {
    // Shed the entries that have a durable copy first, then the rest: an unpersistable entry
    // survives while the cache is under the cap and is dropped only past it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // A is the oldest and unpersistable (a layer offset short of the persist target); B is newer and persists; C is active.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) e.snapshot.entries[0].offset = 300;
        }
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        const before = cache_stats.process.snapshot();
        hc.spillIdleEntries(s);

        // Tier 1 shed B, the durable one, even though A is older.
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        var saw_a = false;
        var a_bytes: u64 = 0;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) {
                saw_a = true;
                a_bytes = e.kv_bytes;
            }
            try testing.expect(!std.mem.eql(u32, e.tokens, &tok_b));
        }
        try testing.expect(saw_a);
        // B's copy is on the SSD: a spill, not an eviction.
        try testing.expectEqual(before.evictions[0], cache_stats.process.snapshot().evictions[0]);

        // Allowance zero: A has nowhere to go and the cap is hard, so tier 2 drops it.
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqual(before.evictions[0] + 1, cache_stats.process.snapshot().evictions[0]);
        try testing.expectEqual(before.evicted_bytes[0] + a_bytes, cache_stats.process.snapshot().evicted_bytes[0]);
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }

    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2b", 0, 128);
        defer hc.deinit();
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
    }
}

test "SSD-first: a silent SKIP is not a durable copy — the idle entry stays resident" {
    // Every silent skip used to read as "the SSD holds this session" and evicted the RAM copy.
    // Four skip reasons, each asserting both halves: the tier holds nothing, RAM still holds the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    // Arm 1: the volume declined the store.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-declined", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try commitUnflushed(&hc, &cache, &tokens_a);
        _ = try commitUnflushed(&hc, &cache, &tokens_b);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 2: under `MIN_PERSIST_TOKENS`.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 400);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-short", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try commitUnflushed(&hc, &cache, tokens_a[0..400]);
        _ = try commitUnflushed(&hc, &cache, tokens_b[0..400]);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 3: a layer offset short of the persist target.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-offset", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try commitUnflushed(&hc, &cache, &tokens_a);
        _ = try commitUnflushed(&hc, &cache, &tokens_b);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.entries[1].offset = 300;
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: a PARTIAL copy is not a copy — the idle entry stays resident" {
    // A byte-capped flush lands real bytes and stops on a chunk boundary: the entry on disk is short and RAM keeps it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-partial", 0, 128);
    defer hc.deinit();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
    // One byte: the loop writes chunk 0 and stops.
    hc.disk.?.max_flush_bytes = 1;

    _ = try commitUnflushed(&hc, &cache, &tokens_a);
    _ = try commitUnflushed(&hc, &cache, &tokens_b);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].kv_len < 600);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Lift the cap and the allowance: the next pass completes the copy and the durable entry goes.
    hc.disk.?.max_flush_bytes = 512 * 1024 * 1024;
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

test "DiskTier.holdsFullPrefix: the INDEX must agree before a RAM copy is discarded" {
    // `.persisted` is the write path's claim; this is the manifest's.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-holds", 0, 128);
    defer tier.deinit();

    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, true, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, .{ .scheme = .affine, .bits = 4, .group_size = 64 }));

    // A truncated tail chunk, the shape `scan` records after a kill -9.
    const cb = tier.entries.items[0].chunk_bytes;
    const keep = cb[cb.len - 1];
    cb[cb.len - 1] = 0;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    cb[cb.len - 1] = keep;
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    tier.entries.items[0].kv_len = 400;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
}

test "SSD-first: one resident session makes reclaimableBytes truthfully ZERO" {
    // Under SSD-first RAM holds exactly the active session at rest, so the largest entry is the
    // only entry and the provable discount is 0 (its buffers are the live KV).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-reclaim", 0, 128);
    defer hc.deinit();

    _ = try hc.commit(&cache, &tokens_a, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() > 0);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // Mid-switch, two sessions are briefly resident and the non-active one is reclaimable.
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.reclaimableBytes() > 0);

    // ...and the idle spill returns it to 0 without evicting the session being served.
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
}

test "reclaimableBytesFor: only an entry the PROMPT could restore from is unevictable" {
    // The guard's credit with the prompt in hand: the prompt-blind rule always subtracts the
    // whole cache under one-session-resident.
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const resident = hc.residentBytes();
    try testing.expect(resident > 0);

    // (1) The prompt extends the resident session: nothing to reclaim.
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&tokens_a));
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // (2) A different session's prompt: the bytes are reclaimable.
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&tokens_b));
    try testing.expect(hc.reclaimableBytesFor(&tokens_b) > hc.reclaimableBytes());

    // A prefix too short to restore from does not pin the entry either.
    var barely: [600]u32 = undefined;
    for (&barely, 0..) |*t, i| t.* = @intCast(i + 7);
    for (barely[MIN_CANCELLED_COMMIT_TOKENS - 8 ..]) |*t| t.* = 424_242;
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&barely));

    // (3) Two entries, prompt matches one: only that one is withheld.
    _ = try hc.commit(&cache, &tokens_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const both = hc.residentBytes();
    const credit_a = hc.reclaimableBytesFor(&tokens_a);
    try testing.expect(credit_a > 0 and credit_a < both);
    try testing.expect(credit_a >= hc.reclaimableBytes());
}

test "EntryDigest: the published snapshot answers the reclaimable question without the cache" {
    // The guard reads a published snapshot of these digests instead of the cache.
    const s = mlx.gpuStream();
    const A = testing.allocator;

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var short: [64]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(A, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(A, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    const resident = hc.residentBytes();

    // Publish, then replace: the superseded slice is the caller's to free.
    var d1 = try hc.digestsAlloc(A);
    try testing.expectEqual(@as(usize, 1), d1.len);
    try testing.expectEqual(resident, d1[0].kv_bytes);
    _ = try hc.commit(&cache, &tokens_b, false);
    const d2 = try hc.digestsAlloc(A);
    A.free(d1);
    d1 = d2;
    defer A.free(d1);
    try testing.expectEqual(@as(usize, 2), d1.len);

    const fp_a = HotPrefixCache.prefixFingerprint(&tokens_a).?;
    const both = hc.residentBytes();
    const credit_a = HotPrefixCache.reclaimableFromDigests(d1, both, fp_a);
    try testing.expect(credit_a > 0 and credit_a < both);

    // A prompt matching neither session credits the whole residency.
    var tokens_c: [600]u32 = undefined;
    for (&tokens_c, 0..) |*t, i| t.* = @intCast(i + 500_000);
    const fp_c = HotPrefixCache.prefixFingerprint(&tokens_c);
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, fp_c));

    // A prompt under the restore floor pins nothing, and hashes to null.
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.prefixFingerprint(&short));
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, null));

    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_a), credit_a);
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_c), both);
}

test "HotPrefixCache: a bounded disk flush leaves disk_dirty set and later flushes complete the entry — never a whole-entry claim in between" {
    // A long entry reaches the tier in pieces; each piece is a valid shorter entry and the next
    // `flushPendingDisk` extends it.
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
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-partial", 0, 128);
    defer hc.deinit();
    hc.disk.?.max_flush_bytes = 1; // one chunk per flush: the bounded shape

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    _ = try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.disk_dirty);

    // 600 tokens at 128/chunk = 5 chunks; each flush lands one.
    var flushes: usize = 0;
    var last_kv: u32 = 0;
    while (hc.disk_dirty and flushes < 10) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const d = &hc.disk.?;
        try testing.expectEqual(@as(usize, 1), d.entryCount());
        const kv = d.entries.items[0].kv_len;
        try testing.expect(kv >= last_kv);
        if (hc.disk_dirty) try testing.expectEqual(@as(u32, 0), kv % 128);
        const m = d.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
        try testing.expectEqual(kv, m.usable);
        last_kv = kv;
    }
    try testing.expect(!hc.disk_dirty);
    try testing.expectEqual(@as(usize, 5), flushes);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

/// The resident entry whose token record is exactly `toks`.
fn testEntryFor(hc: *HotPrefixCache, toks: []const u32) !*Entry {
    for (hc.entries.items) |*e| {
        if (std.mem.eql(u32, e.tokens, toks)) return e;
    }
    return error.EntryGone;
}

test "SSD-first: a chunk write that fails AFTER the pass invalidates the entry — RAM is never dropped against it" {
    // Pass N stages A and skips it (pending). The writer then loses `c000003` but, FIFO, still
    // lands meta.json, so the index keeps a non-zero `chunk_bytes[3]`. Pass N+1 must not call
    // A durable. The bar is the verdict (`spill_durable`), not the eviction it licenses.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-poison", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try commitUnflushed(&hc, &cache, &tok_a);
    _ = try commitUnflushed(&hc, &cache, &tok_b);

    // Pass N: the writer is held, so A's files stay staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.writer.?.injectFailure("c000003", .write);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const dead_id = hc.disk.?.entries.items[0].id;

    // Between the passes: chunk 3 dies, meta.json lands.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expectEqual(@as(usize, 5), hc.disk.?.entries.items[0].chunk_bytes.len);

    // The failure names its entry, so a restore from it misses.
    try testing.expectEqual(@as(usize, 1), hc.disk.?.harvestWriteFailures());
    try testing.expect(hc.disk.?.entries.items[0].poisoned);
    try testing.expect(hc.disk.?.bestMatch(&tok_a, false, cache.config, .{}) == null);
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));

    // Pass N+1: A is not durable; the dead directory is reclaimed and the rebuild is staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].id != dead_id);

    // Control: with the writer healthy the same entry is durable and restores whole.
    hc.disk.?.writer.?.injectFailure(null, .write);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);

    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tok_a, false, 0, null, null);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "SSD-first: a write failure inside the SAME pass still keeps the entry resident" {
    // The same-pass interleaving (`writeErrors() != errs_before`), via a submit-time injection.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-samepass", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try commitUnflushed(&hc, &cache, &tok_a);
    _ = try commitUnflushed(&hc, &cache, &tok_b);
    hc.disk.?.writer.?.injectFailure("c000002", .submit);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));
}

test "SSD-first: the durability check STATS the chunks — a truncated file is never durable" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-stat", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    // Pass 1 stages A with the writer held; pass 2, after the files land, is the healthy control.
    // Held before the commits: the second commit flushes the first entry, which a running writer would finish.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    // One chunk loses its bytes behind the tier's back; every in-memory bar still passes.
    const id = hc.disk.?.entries.items[0].id;
    var sub: [64]u8 = undefined;
    const rel = try std.fmt.bufPrint(&sub, "fp-stat/e{d}/c000002.safetensors", .{id});
    try kv_disk_fs.testWriteAt(io, buf[0..root_len], rel, "short");

    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
}

test "HotPrefixCache: disk-only commit persists the full prefix and retains no RAM entry" {
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
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, tokens.len);
        const status = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
        try testing.expectEqual(@as(usize, tokens.len), status.disk_only);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
        try testing.expect(hc.pending_disk != null);

        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        var moe_offset: usize = 0;
        const restored = try hc.lookupAndRestore(&cache, &moe_offset, null, s, &tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, tokens.len - 1), restored.matched);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

test "SSD-only ring checkpoints survive restart without idle RAM" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.gpuStream();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const base = path[0..try tmp.dir.realPath(io, &path)];
    const prompt: u32 = 700;
    const total: u32 = 1300;
    const window: u32 = 8;
    const layers: u32 = 4;
    var tokens: [total]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    for (0..2) |boot| {
        var hc = HotPrefixCache.initWithMem(a, 0, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(a, io, base, "ssd-only-ring", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        defer hc.deinit();
        var cache = try KVCache.init(a, layers);
        defer cache.deinit();
        cache.setSwaRing(window);
        if (boot == 0) {
            try ringFill(&cache, s, layers, window, 0, prompt, 16);
            const cp = try cache.ringCheckpoint(prompt, s);
            try ringFill(&cache, s, layers, window, prompt, total, 16);
            const status = try hc.commitWithRing(&cache, &tokens, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
            try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
            hc.flushPendingDisk(s);
            hc.disk.?.drainWriter();
            try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
            // Once the writer has landed them, no restore point is held in RAM.
            try testing.expect(hc.pending_disk == null);
            try testing.expectEqual(@as(u64, 0), hc.disk.?.stagedHostBytes());
        } else {
            var next: [800]u32 = undefined;
            @memcpy(next[0..prompt], tokens[0..prompt]);
            for (next[prompt..], 0..) |*t, i| t.* = @intCast(90_000 + i);
            var offset: usize = 0;
            const hit = try hc.lookupAndRestore(&cache, &offset, null, s, &next, false, 0, null, null);
            try testing.expectEqual(@as(usize, prompt), hit.matched);
            try testing.expect(!hit.ownsRestoredRows());
            try testing.expectEqual(@as(usize, prompt), offset);
        }
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

test "an image turn persists its WHOLE turn to the SSD tier, image rows included" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(3072);
    // The record carries the media identity (manifest v10), so it holds the whole turn whatever
    // the media boundary is; a hybrid still restores at its highest checkpoint, now above the item.
    // The record always holds the whole turn (3072 rows). `want` = what the SAME image restores:
    // a full token match leaves its last token to forward (3071), and a hybrid stops at its
    // highest checkpoint instead. `cross` = what another key restores, which is the text before
    // the item, again clamped to a hybrid's highest checkpoint at or below it.
    const Arm = struct { ssd_first: bool = false, mem: u64 = 0, cps: ?[]const usize = null, ms: usize = 2600, want: u32, cross: u32 };
    const arms = [_]Arm{
        .{ .ssd_first = true, .want = 3071, .cross = 2600 },
        .{ .want = 3071, .cross = 2600 },
        .{ .mem = 16 * 1024, .want = 3071, .cross = 2600 },
        .{ .ssd_first = true, .mem = 16 * 1024, .want = 3071, .cross = 2600 },
        .{ .cps = &.{ 1024, 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .ssd_first = true, .cps = &.{ 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .mem = 16 * 1024, .cps = &.{ 1024, 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .cps = &.{ 1024, 2048, 3000 }, .ms = 2048, .want = 3000, .cross = 2048 },
        .{ .ssd_first = true, .cps = &.{ 1024, 2048, 3000 }, .ms = 2047, .want = 3000, .cross = 1024 },
        .{ .cps = &.{3000}, .want = 3000, .cross = 0 },
        .{ .ssd_first = true, .cps = &.{3000}, .want = 3000, .cross = 0 },
    };
    for (arms) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        var cold = try KVCache.init(testing.allocator, 3);
        defer cold.deinit();
        try testFillCache(&cold, s, 3, tokens.len);
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, arm.mem);
            hc.ssd_first = arm.ssd_first;
            hc.hybrid = arm.cps != null;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc.deinit();
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try testFillCache(&cache, s, 3, tokens.len);
            var src = pcBuildHybrid(s, 100.0, 500.0);
            defer pcFreeHybrid(&src);
            var cps: ?[]SSMCheckpoint = null;
            if (arm.cps) |positions| {
                cps = try testing.allocator.alloc(SSMCheckpoint, positions.len);
                for (positions, cps.?) |p, *cp| cp.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, p, s);
            }
            _ = try hc.commitWithMediaState(&cache, &tokens, false, 0xABCD, 0, arm.ms, cps, null, null, tokens.len);
            hc.flushPendingDisk(s);
        }
        // A cold RAM tier (restart, eviction) restores, under the SAME image, everything the
        // record holds — image rows included. Another image whose placeholders carry the same
        // ids, a later first item, and no image at all get the text before the item and no more.
        const Ask = struct { key: u64, ms: ?usize, want: u32 };
        for ([_]Ask{
            .{ .key = 0xABCD, .ms = arm.ms, .want = arm.want },
            .{ .key = 0x1234, .ms = arm.ms, .want = arm.cross },
            .{ .key = 0x1234, .ms = 2900, .want = arm.cross },
            .{ .key = 0, .ms = null, .want = arm.cross },
        }) |ask| {
            var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc2.ssd_first = arm.ssd_first;
            hc2.hybrid = arm.cps != null;
            hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc2.deinit();
            try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
            // The whole turn, not a prefix cut at the item.
            try testing.expectEqual(@as(u32, tokens.len), hc2.disk.?.entries.items[0].kv_len);
            // The record names the media it was computed under, which is what bounds a foreign key.
            try testing.expectEqual(@as(u64, 0xABCD), hc2.disk.?.entries.items[0].vision_key);
            try testing.expectEqual(@as(?u32, @intCast(arm.ms)), hc2.disk.?.entries.items[0].media_start);
            var dst = try KVCache.init(testing.allocator, 3);
            defer dst.deinit();
            var ssm = pcEmptySsm();
            defer pcFreeHybrid(&ssm);
            var moe: usize = 0;
            const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, if (arm.cps != null) &ssm else null, s, &tokens, false, ask.key, ask.ms, &.{}, null, null, null, false);
            try testing.expectEqual(@as(usize, ask.want), res.matched);
            if (ask.want == 0) continue;
            for (0..3) |li| try pcExpectRowsEqual(s, dst.entries[li].keys.array, cold.entries[li].keys.array, ask.want);
            if (arm.cps != null) try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm[0].conv_state, 0, s));
        }
    }
}

test "truncatedMediaKey: a record claims exactly the media blocks its rows cover" {
    const chain = [_]MediaSpan{ .{ .start = 100, .key = 0xA1 }, .{ .start = 2000, .key = 0xB2 } };
    const key = chainKey(&chain);

    // Whole prompt: the request's own key.
    try testing.expectEqual(@as(u64, 0xB2), truncatedMediaKey(&chain, key, 100, 3000).key);
    try testing.expectEqual(@as(?u32, 100), truncatedMediaKey(&chain, key, 100, 3000).start);
    // Cut between the two blocks: the first block's key, which is what a later request's chain
    // can recognise as its own prefix. Claiming 0xB2 here would be safe but would cost the reuse
    // of the first image's rows.
    try testing.expectEqual(@as(u64, 0xA1), truncatedMediaKey(&chain, key, 100, 1500).key);
    // Exactly at the second block's first row: that block's rows are NOT covered.
    try testing.expectEqual(@as(u64, 0xA1), truncatedMediaKey(&chain, key, 100, 2000).key);
    // Cut below every block: no media row, so a text prompt may share the record whole.
    try testing.expectEqual(@as(u64, 0), truncatedMediaKey(&chain, key, 100, 100).key);
    try testing.expectEqual(@as(?u32, null), truncatedMediaKey(&chain, key, 100, 50).start);
    // No chain (a caller that does not track blocks) keeps the caller's key and boundary.
    try testing.expectEqual(key, truncatedMediaKey(&.{}, key, 100, 3000).key);
    // A text request claims nothing, whatever it covers.
    try testing.expectEqual(@as(u64, 0), truncatedMediaKey(&.{}, 0, null, 3000).key);
}

test "the prefill write-through persists an image turn, keyed by the media it has forwarded" {
    // Step 4: a media turn's chunks were skipped outright, so a long image conversation banked
    // nothing until its end-of-request commit. Each chunk record now claims the blocks inside it.
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(2048);
    const first_at: usize = 600;
    const k1: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-wt", 0, 1024);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);

    // The first chunk stops below the item: it holds no media row, so it claims none.
    hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..512], false, truncatedMediaKey(&.{}, k1, first_at, 512), null, &.{}, &.{}, std.math.maxInt(u64), s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.disk.?.entries.items[0].vision_key);

    // A later chunk covers the item, so the record names the pixels its rows were computed under.
    hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..2048], false, truncatedMediaKey(&.{}, k1, first_at, 2048), null, &.{}, &.{}, std.math.maxInt(u64), s);
    const e = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(u64, k1), e.vision_key);
    try testing.expectEqual(@as(?u32, @intCast(first_at)), e.media_start);
    try testing.expectEqual(@as(u32, 2048), e.kv_len);
}

test "SSD: an image turn's record links the chunks its prefix was restored from, through the commit" {
    // An agent appends a screenshot per turn: the new record's key is another chain, so the donor rule lends only the
    // text before the first image, and every turn rewrote the rows it had just restored.
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(1200);
    const first_at: usize = 200;
    const k1: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    const chain = [_]MediaSpan{ .{ .start = first_at, .key = k1 }, .{ .start = 900, .key = 0xB2B2_B2B2_B2B2_B2B2 } };
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-restored", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    _ = try hc.commitWithMediaState(&cache, tokens[0..800], false, k1, 0, first_at, null, null, null, 800);
    hc.flushPendingDisk(s);

    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreWithMedia(&dst, &moe, null, s, tokens[0..1100], false, chainKey(&chain), first_at, &chain, null, null, null, false);
    try testing.expectEqual(@as(usize, 800), res.matched);
    const src = res.disk_restore orelse return error.TestExpectedDiskRestore;
    try testing.expectEqual(hc.disk.?.entries.items[0].id, src.id);
    try testing.expectEqual(@as(u32, 800), src.rows);

    hc.commit_restored = src;
    _ = try hc.commitWithMediaState(&cache, tokens[0..1100], false, chainKey(&chain), 0, first_at, null, null, null, 1100);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    try testing.expectEqual(@as(u32, 800 / 128), hc.disk.?.entries.items[1].inherited_chunks);
    try testing.expectEqual(@as(?kv_disk_cache.RestoredRows, null), hc.commit_restored);
}

test "SSD: a record holding the request's first image restores up to the new one" {
    // The disk half of the chain rule (`crossKeyBoundary`), which only the RAM tier could serve
    // before: an agent turn that appends a second screenshot to a conversation already holding
    // one reuses the persisted record through the FIRST image's rows, and stops at the new block.
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(3072);
    const first_at: usize = 600;
    const second_at: usize = 2000;
    const k1: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    const k2: u64 = 0xB2B2_B2B2_B2B2_B2B2;

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var cold = try KVCache.init(testing.allocator, 3);
    defer cold.deinit();
    try testFillCache(&cold, s, 3, tokens.len);
    {
        // The earlier turn: one image at 100, keyed by its pixels.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-chain", 0, 1024);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        _ = try hc.commitWithMediaState(&cache, &tokens, false, k1, 0, first_at, null, null, null, tokens.len);
        hc.flushPendingDisk(s);
    }

    // The next turn appends a second image at 2000. Its chain's first block IS the record's.
    const chain = [_]MediaSpan{ .{ .start = first_at, .key = k1 }, .{ .start = second_at, .key = k2 } };
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-chain", 0, 1024);
    defer hc2.deinit();
    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var moe: usize = 0;
    const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, null, s, &tokens, false, chainKey(&chain), first_at, &chain, null, null, null, false);
    try testing.expectEqual(second_at, res.matched);
    for (0..3) |li| try pcExpectRowsEqual(s, dst.entries[li].keys.array, cold.entries[li].keys.array, second_at);

    // Without the chain, the same request can only be held to the text before its first image:
    // nothing then proves the record's block is this request's.
    var bare = try KVCache.init(testing.allocator, 3);
    defer bare.deinit();
    var moe2: usize = 0;
    const res2 = try hc2.lookupAndRestoreWithMedia(&bare, &moe2, null, s, &tokens, false, chainKey(&chain), first_at, &.{}, null, null, null, false);
    try testing.expectEqual(first_at, res2.matched);
}

test "an image turn's speculative snapshots ride its whole SSD record" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(1200);
    for ([_]bool{ true, false }) |media| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-spec", 0, 128);
        defer hc.deinit();
        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, tokens.len);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, tokens.len);
        var hist = try KVCache.init(testing.allocator, 1);
        defer hist.deinit();
        try testFillCache(&hist, s, 1, tokens.len);
        _ = try hc.commitWithMediaState(&trunk, &tokens, false, if (media) 0xABCD else 0, 0, if (media) 800 else null, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 }, tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const e = &hc.disk.?.entries.items[0];
        // The record is no longer cut at the media boundary, so the dflash context and the MTP
        // history cover exactly its rows and persist with it either way.
        try testing.expectEqual(@as(u32, 1200), e.kv_len);
        try testing.expect(e.spec_dflash != null);
        try testing.expect(e.spec_mtp != null);
    }
}

test "an image turn on a ringed cache persists whole and restores at its ring point under its own key" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    // The only ring point sits at the prompt end (700). Under the record's OWN key it serves
    // whatever the item's position; a foreign key is held to the text before the item, and the
    // point's rows cannot reach back that far when the item sits at 500.
    // Under its own key the record's END point serves the whole turn less the token a restore
    // leaves to forward; a foreign key is held to the text before the item, which only the
    // prompt-end point can reach when the item sits at 700.
    for ([_]struct { ms: usize, want: u32, cross: u32 }{
        .{ .ms = 700, .want = prompt + reply - 1, .cross = 700 },
        .{ .ms = 500, .want = prompt + reply - 1, .cross = 0 },
    }) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-ring", 0, 128);
            defer hc.deinit();
            var live = try KVCache.init(testing.allocator, n_layers);
            defer live.deinit();
            const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
            _ = try hc.commitWithRing(&live, &toks, false, 0xABCD, 0, arm.ms, null, null, null, prompt, .{ .prompt_end = cp });
            hc.flushPendingDisk(s);
        }
        for ([_]struct { key: u64, want: u32 }{ .{ .key = 0xABCD, .want = arm.want }, .{ .key = 0x1234, .want = arm.cross } }) |ask| {
            var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-ring", 0, 128);
            defer hc2.deinit();
            // The record exists either way now: it is keyed by its pixels, not cut at them.
            try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
            var dst = try KVCache.init(testing.allocator, n_layers);
            defer dst.deinit();
            dst.setSwaRing(window);
            var moe: usize = 0;
            const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, null, s, &toks, false, ask.key, arm.ms, &.{}, null, null, null, false);
            try testing.expectEqual(@as(usize, ask.want), res.matched);
            // The restored ring keeps serving from WHEREVER the restore landed, which under the
            // record's own key is now past the prompt end.
            if (ask.want > 0) try expectRingContinuesCold(&dst, s, n_layers, window, ask.want, ask.want + 40);
        }
    }
}

test "HotPrefixCache: a checkpoint taken inside the generated reply is where a prompt that repeats the thought restores" {
    const s = mlx.gpuStream();
    const vision_key: u64 = 0xcafe;
    // 12 prompt tokens, then the reply: its thought, the closer (99) and the first answer token. The prompt-end checkpoint
    // sits at 10; the thought's, taken behind the closer, at 19.
    const entry = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 99, 27 };
    // The next turn re-renders the thought and the closer, then carries the tool result.
    const repeats = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 99, 70, 71, 72, 73 };
    // A client that did not send the thought back diverges at the reply's first token.
    const drops = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 80, 81, 82, 83, 84, 85, 86, 87, 88 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, entry.len);
    var prompt_end = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&prompt_end);
    var thought = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&thought);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &prompt_end, 10, s);
    cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &thought, 19, s);
    _ = try hc.commitWithMediaState(&cache, &entry, true, vision_key, 0, null, cps, null, null, 12);

    for ([_]struct { prompt: []const u32, at: usize, conv: f32 }{
        .{ .prompt = &repeats, .at = 19, .conv = 300.0 },
        .{ .prompt = &drops, .at = 10, .conv = 100.0 },
    }) |c| {
        var target = try KVCache.init(testing.allocator, 3);
        defer target.deinit();
        var ssm = pcEmptySsm();
        defer pcFreeHybrid(&ssm);
        var moe_off: usize = 0;
        const result = try hc.lookupAndRestore(&target, &moe_off, &ssm, s, c.prompt, true, vision_key, null, null);
        try testing.expectEqual(c.at, result.matched);
        try testing.expectEqual(c.at, moe_off);
        try testing.expectEqual(c.conv, pcSsmVal(ssm[0].conv_state, 0, s));
    }
}

test "a write-through under other pixels never extends the record that holds the first image's rows" {
    // Two images of one size render the same placeholder ids, so the token match alone cannot tell them apart.
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(2048);
    const at: usize = 600;
    const ka: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    const kb: u64 = 0xB2B2_B2B2_B2B2_B2B2;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-swap", 0, 1024);
    defer hc.deinit();
    var rows_a = try KVCache.init(testing.allocator, 3);
    defer rows_a.deinit();
    try kv_disk_cache.fillCache(&rows_a, s, 3, tokens.len, 8, 0.0, .float32);
    var rows_b = try KVCache.init(testing.allocator, 3);
    defer rows_b.deinit();
    try kv_disk_cache.fillCache(&rows_b, s, 3, tokens.len, 8, 5_000_000.0, .float32);

    // Image A's prefill stops after its first chunk (a disconnect); image B's request has the same ids.
    hc.writeThroughHybrid(rows_a.entries, rows_a.config, tokens[0..1024], false, truncatedMediaKey(&.{}, ka, at, 1024), null, &.{}, &.{}, std.math.maxInt(u64), s);
    hc.writeThroughHybrid(rows_b.entries, rows_b.config, tokens[0..2048], false, truncatedMediaKey(&.{}, kb, at, 2048), null, &.{}, &.{}, std.math.maxInt(u64), s);
    hc.disk.?.drainWriter();

    const chain = [_]MediaSpan{.{ .start = at, .key = kb }};
    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreWithMedia(&dst, &moe, null, s, &tokens, false, kb, at, &chain, null, null, null, false);
    try testing.expect(res.matched > at);
    for (0..3) |li| try pcExpectRowsEqual(s, dst.entries[li].keys.array, rows_b.entries[li].keys.array, res.matched);
}

test "HotPrefixCache: commit_media_chain inherits RAM checkpoints past image 1 when a follow-up appends image 2" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    const k1: u64 = 0x1111_AAAA;
    const k2: u64 = 0x2222_BBBB;
    const img1_at: usize = 8;
    const img2_at: usize = 40;
    const chain = [_]MediaSpan{
        .{ .start = img1_at, .key = k1 },
        .{ .start = img2_at, .key = k2 },
    };
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);

    // Turn 1: image 1 at 8, checkpoint at 32 (past image 1, before image 2).
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try kv_disk_cache.fillCache(&c1, s, 3, 36, 4, 0.0, .float32);
    var src1 = pcBuildHybrid(s, 10.0, 20.0);
    defer pcFreeHybrid(&src1);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps1[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src1, 32, s);
    hc.commit_media_chain = chain[0..1];
    _ = try hc.commitWithMediaState(&c1, tokens[0..36], false, k1, 0, img1_at, cps1, null, null, 36);

    // Turn 2: appends image 2 at 40, checkpoint at 60 -> must inherit checkpoint at 32 from turn 1.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try kv_disk_cache.fillCache(&c2, s, 3, 64, 4, 0.0, .float32);
    var src2 = pcBuildHybrid(s, 30.0, 40.0);
    defer pcFreeHybrid(&src2);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps2[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src2, 60, s);
    hc.commit_media_chain = &chain;
    _ = try hc.commitWithMediaState(&c2, &tokens, false, k2, 0, img1_at, cps2, null, null, 64);

    try testing.expectEqual(@as(usize, 2), hc.entries.items.len);
    const kept = hc.entries.items[1].ssm_checkpoints orelse return error.TestExpectedCheckpoints;
    try testing.expectEqual(@as(usize, 2), kept.len);
    try testing.expectEqual(@as(usize, 32), kept[0].pos);
    try testing.expectEqual(@as(usize, 60), kept[1].pos);
}

test "HotPrefixCache: a write-through that returns before writing leaves no media chain for the next consumer" {
    var hc = HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    const chain = [_]MediaSpan{.{ .start = 8, .key = 0x1111_AAAA }};
    hc.commit_media_chain = &chain;
    hc.writeThroughHybrid(&.{}, kv_quant.KVQuantConfig.dense, &.{}, false, .{}, null, &.{}, &.{}, std.math.maxInt(u64), mlx.gpuStream());
    try testing.expectEqual(@as(usize, 0), hc.commit_media_chain.len);
}
