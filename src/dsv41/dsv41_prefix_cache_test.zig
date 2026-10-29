const transformer_mod = @import("../transformer.zig");
const KVCache = transformer_mod.KVCache;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const dsv41_model = @import("dsv41_model.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const glm5_forward = @import("../glm5/glm5_forward.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const disk_placement = @import("../disk_placement.zig");
const kv_quant = @import("../kv_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;
const std = @import("std");
const testing = std.testing;
const pc = @import("../prefix_cache.zig");

const Dsv41Net = struct {
    m: *dsv41_model.Model,
    ids: []u32,
    fx: model_mod.Weights,

    fn load(self: *Dsv41Net) !void {
        const dir = try dsv41_model.fixtureDir();
        self.fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
        errdefer self.fx.deinit();
        var sc = @import("dsv41_scope.zig").Scope.init(testing.allocator, mlx.gpuStream());
        defer sc.deinit();
        self.ids = try dsv41_model.tokens(&sc, &self.fx);
        errdefer testing.allocator.free(self.ids);
        self.m = try dsv41_model.tinyModel(dir, .{ .act = .bfloat16, .dspark = true }, mlx.gpuStream());
    }

    fn deinit(self: *Dsv41Net) void {
        self.m.deinit();
        testing.allocator.free(self.ids);
        self.fx.deinit();
    }

    /// Prefills `tokens` in `chunks` from the request's current position, a checkpoint after each listed chunk.
    fn prefill(self: *Dsv41Net, tokens: []const u32, chunks: []const usize, at_start: usize, checkpoints: []const usize, cps: *std.ArrayList(SSMCheckpoint)) !void {
        var at = at_start;
        for (chunks, 0..) |n, i| {
            _ = mlx.mlx_array_free(try dsv41_model.extend(self.m, tokens[at .. at + n], .last));
            at += n;
            if (std.mem.indexOfScalar(usize, checkpoints, i) != null) try cps.append(testing.allocator, try dsv41_prefix.capture(testing.allocator, self.m, at));
        }
    }
};

fn dsv41Lookup(hc: *pc.HotPrefixCache, net: *Dsv41Net, prompt: []const u32) !pc.LookupResult {
    return dsv41LookupMedia(hc, net, prompt, 0, null);
}

fn dsv41LookupMedia(hc: *pc.HotPrefixCache, net: *Dsv41Net, prompt: []const u32, vision_key: u64, media_start: ?usize) !pc.LookupResult {
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer target.deinit();
    var off: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &off, net.m, mlx.gpuStream(), prompt, false, vision_key, media_start, &.{}, null, false);
    try testing.expectEqual(res.matched, target.step);
    try testing.expectEqual(res.matched, (try dsv41_model.ensureState(net.m)).n);
    return res;
}

fn dsv41SsdOnlyCache(io: std.Io, base: []const u8, fp: []const u8) !pc.HotPrefixCache {
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    errdefer hc.deinit();
    hc.ssd_first = true;
    hc.hybrid = true;
    hc.cp_thin = .min_span_recency;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.cp_thin = .min_span_recency;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    return hc;
}

fn cloneCheckpointForTest(src: *const SSMCheckpoint) !SSMCheckpoint {
    const layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, src.layers.len);
    for (src.layers, layers) |l, *out| out.* = .{
        .conv_state = try transformer_mod.materializedOwnedCopy(mlx.gpuStream(), l.conv_state),
        .ssm_state = try transformer_mod.materializedOwnedCopy(mlx.gpuStream(), l.ssm_state),
        .initialized = true,
    };
    return .{ .pos = src.pos, .layers = layers };
}

fn expectCheckpointBits(want: *const SSMCheckpoint, got: *const SSMCheckpoint) !void {
    try testing.expectEqual(want.pos, got.pos);
    try testing.expectEqual(want.layers.len, got.layers.len);
    for (want.layers, got.layers) |a, b| {
        for ([_][2]mlx.mlx_array{ .{ a.conv_state, b.conv_state }, .{ a.ssm_state, b.ssm_state } }) |pair| {
            try testing.expectEqualSlices(c_int, mlx.getShape(pair[0]), mlx.getShape(pair[1]));
            try testing.expectEqual(mlx.mlx_array_dtype(pair[0]), mlx.mlx_array_dtype(pair[1]));
            try glm5_forward.expectArrayBits(pair[0], pair[1]);
        }
    }
}

fn dsv41LongPrompt(net: *const Dsv41Net, buf: []u32, shared_with: ?[]const u32, shared: usize) []const u32 {
    for (buf, 0..) |*t, i| t.* = if (shared_with != null and i < shared) shared_with.?[i] else net.ids[(i * 5 + 3) % net.ids.len];
    return buf;
}

/// `dsv41SsdOnlyCache` as the scheduler arms it: restore points split into a row store per entry, on a grid of `span` tokens.
fn dsv41SplitCache(io: std.Io, base: []const u8, fp: []const u8, net: *const Dsv41Net, span: u32) !pc.HotPrefixCache {
    var hc = try dsv41SsdOnlyCache(io, base, fp);
    errdefer hc.deinit();
    const divs = try dsv41_prefix.rowDivisors(testing.allocator, &net.m.cfg, true);
    defer testing.allocator.free(divs);
    hc.disk.?.ssm_max_per_entry = dsv41_prefix.disk_checkpoint_cap;
    try hc.disk.?.setGrownSpec(span, divs, net.m.cfg.stateBytesPerToken());
    return hc;
}

test "HotPrefixCache: shouldUse admits deepseek_v41 only with checkpoints (its state restores from them)" {
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "deepseek_v41";
    try testing.expect(!pc.HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(pc.HotPrefixCache.shouldUse(&cfg, true));
}

test "DeepSeek hot cache: an image prompt restores whole for its own images and only up to the image for others (DSV41_TINY)" {
    try mlx.requireGpu();
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    const prompt = net.ids[0..30];
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();

    // The prompt holds an image from position 15; checkpoints sit at 12 (before it) and 20 (inside its span).
    dsv41_model.resetState(net.m);
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    try net.prefill(prompt, &.{ 12, 8, 10 }, 0, &.{ 0, 1 }, &cps);
    var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    cache.step = prompt.len;
    defer cache.deinit();
    const owned = try cps.toOwnedSlice(testing.allocator);
    const own_key: u64 = 0xdee5eec;
    try testing.expectEqual(pc.CommitStatus{ .ok = prompt.len }, try hc.commitWithMediaState(&cache, prompt, false, own_key, 0, 15, owned, null, null, prompt.len));

    // Another image under the same ids: equal ids do not make equal rows, so only the text before it is shared.
    const other = try dsv41LookupMedia(&hc, &net, prompt, own_key ^ 1, 15);
    try testing.expectEqual(@as(usize, 12), other.matched);
    // The same image shares the checkpoint inside its span.
    const same = try dsv41LookupMedia(&hc, &net, prompt, own_key, 15);
    try testing.expectEqual(@as(usize, 20), same.matched);
    // A text prompt with the same ids is no image prompt either.
    const text = try dsv41LookupMedia(&hc, &net, prompt, 0, null);
    try testing.expectEqual(@as(usize, 12), text.matched);
}

test "DeepSeek hot cache restores the highest checkpoint under the match and continues like a cold prefill (DSV41_TINY)" {
    try mlx.requireGpu();
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    const first = net.ids[0..30];
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();

    // Turn one: chunks of 12, 8 and 10 with a checkpoint after the first two (positions 12 and 20).
    dsv41_model.resetState(net.m);
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    try net.prefill(first, &.{ 12, 8, 10 }, 0, &.{ 0, 1 }, &cps);
    var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    cache.step = first.len;
    defer cache.deinit();
    const owned = try cps.toOwnedSlice(testing.allocator);
    try testing.expectEqual(pc.CommitStatus{ .ok = first.len }, try hc.commitWithState(&cache, first, false, 0, owned, null, null));
    const e = &hc.entries.items[0];
    var billed: u64 = 0;
    for (e.ssm_checkpoints.?) |*cp| billed += ssmCheckpointBytes(cp);
    try testing.expect(billed > 0);
    try testing.expectEqual(billed, e.kv_bytes);
    try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);

    // Turn two shares the first 25 tokens and goes its own way: it restores at 20 and runs the same chunks as cold.
    var next: [40]u32 = undefined;
    @memcpy(next[0..25], first[0..25]);
    for (next[25..], 25..) |*t, i| t.* = (net.ids[i] + 1) % 90;
    const res = try dsv41Lookup(&hc, &net, &next);
    try testing.expectEqual(@as(usize, 20), res.matched);
    const warm = try dsv41_model.extend(net.m, next[20..], .all);
    defer _ = mlx.mlx_array_free(warm);
    var warm_cp = try dsv41_prefix.capture(testing.allocator, net.m, next.len);
    defer warm_cp.deinit(testing.allocator);

    dsv41_model.resetState(net.m);
    var cold_cps: std.ArrayList(SSMCheckpoint) = .empty;
    defer {
        for (cold_cps.items) |*cp| cp.deinit(testing.allocator);
        cold_cps.deinit(testing.allocator);
    }
    try net.prefill(&next, &.{ 12, 8 }, 0, &.{}, &cold_cps);
    const cold = try dsv41_model.extend(net.m, next[20..], .all);
    defer _ = mlx.mlx_array_free(cold);
    var cold_cp = try dsv41_prefix.capture(testing.allocator, net.m, next.len);
    defer cold_cp.deinit(testing.allocator);
    try glm5_forward.expectArrayBits(cold, warm);
    for (cold_cp.layers, warm_cp.layers) |a, b| {
        for ([_][2]mlx.mlx_array{ .{ a.conv_state, b.conv_state }, .{ a.ssm_state, b.ssm_state }, .{ a.aux_state, b.aux_state }, .{ a.qsa_pooled, b.qsa_pooled } }) |pair| {
            try testing.expectEqual(pair[0].ctx == null, pair[1].ctx == null);
            if (pair[0].ctx != null and mlx.mlx_array_size(pair[0]) > 0) try glm5_forward.expectArrayBits(pair[0], pair[1]);
        }
    }

    // A prompt that shares nothing restores nothing and leaves the model's state empty.
    var other: [20]u32 = undefined;
    for (&other) |*t| t.* = 90;
    try testing.expectEqual(@as(usize, 0), (try dsv41Lookup(&hc, &net, &other)).matched);
    // A prompt that diverges before the first checkpoint cannot use any.
    var early: [20]u32 = undefined;
    @memcpy(early[0..5], first[0..5]);
    for (early[5..]) |*t| t.* = 91;
    try testing.expectEqual(@as(usize, 0), (try dsv41Lookup(&hc, &net, &early)).matched);
}

test "DeepSeek hot cache bills the held checkpoints in its residency and gives them back on eviction (DSV41_TINY)" {
    try mlx.requireGpu();
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var hc = pc.HotPrefixCache.init(testing.allocator, 1);
    defer hc.deinit();
    var billed: [2]u64 = undefined;
    for (&billed, 0..) |*b, turn| {
        var prompt: [24]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = (net.ids[i] + @as(u32, @intCast(turn)) * 7) % 90;
        dsv41_model.resetState(net.m);
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        try net.prefill(&prompt, &.{ 16, 8 }, 0, &.{ 0, 1 }, &cps);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        cache.step = prompt.len;
        _ = try hc.commitWithState(&cache, &prompt, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
        var held: u64 = 0;
        for (hc.entries.items) |*e| for (e.ssm_checkpoints.?) |*cp| {
            held += ssmCheckpointBytes(cp);
        };
        b.* = held;
        try testing.expect(held > 0);
        try testing.expectEqual(held, hc.residentBytes());
    }
    // The second turn shares nothing with the first, so the one-entry cache evicted it: only the second is billed.
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(billed[1], hc.residentBytes());
}

test "DeepSeek hot cache over its byte budget keeps the entry with fewer checkpoints, billed exactly (DSV41_TINY)" {
    try mlx.requireGpu();
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    dsv41_model.resetState(net.m);
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    try net.prefill(net.ids[0..24], &.{ 8, 8, 8 }, 0, &.{ 0, 1, 2 }, &cps);
    // The tiny state is far shorter than the commit floor (256 tokens): relabel the positions, as the budget only reads bytes.
    for (cps.items, 0..) |*cp, i| cp.pos = 800 * (i + 1);
    const one = ssmCheckpointBytes(&cps.items[0]);
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, one * 2 + one / 2);
    defer hc.deinit();
    hc.cp_thin = .min_span_recency;
    var tokens: [2400]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = net.ids[i % net.ids.len];
    var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer cache.deinit();
    cache.step = tokens.len;
    _ = try hc.commitWithState(&cache, &tokens, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() <= one * 2 + one / 2);
    const e = &hc.entries.items[0];
    var sum: u64 = 0;
    for (e.ssm_checkpoints.?) |*cp| sum += ssmCheckpointBytes(cp);
    try testing.expectEqual(sum, e.kv_bytes);
    try testing.expectEqual(e.kv_bytes, hc.residentBytes());
}

test "DeepSeek SSD tier: a checkpoint written to disk reads back as the same bytes, and a restart restores it like a cold prefill (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [700]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var next_buf: [760]u32 = undefined;
    for (&next_buf, 0..) |*t, i| t.* = if (i < 650) first[i] else (net.ids[i % net.ids.len] + 1) % 90;
    const next: []const u32 = &next_buf;

    var want_512: SSMCheckpoint = undefined;
    {
        var hc = try dsv41SsdOnlyCache(io, base, "dsv41-ssd");
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        try net.prefill(first, &.{ 256, 256, 188 }, 0, &.{ 0, 1 }, &cps);
        // the in-RAM state at 512, kept for the byte comparison
        want_512 = try cloneCheckpointForTest(&cps.items[1]);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        cache.step = first.len;
        try dsv41_prefix.markRows(&cache);
        const status = try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
        try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, status);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    defer want_512.deinit(testing.allocator);

    // A fresh tier on the same directory: nothing in RAM, the entry only on disk.
    var hc = try dsv41SsdOnlyCache(io, base, "dsv41-ssd");
    defer hc.deinit();
    dsv41_model.resetState(net.m);
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(@as(usize, 512), res.matched);
    try testing.expect(hc.last_restored_disk_id != null);
    try testing.expectEqual(@as(usize, 512), target.step);
    try testing.expectEqual(@as(usize, 512), (try dsv41_model.ensureState(net.m)).n);
    var got_512 = try dsv41_prefix.capture(testing.allocator, net.m, 512);
    defer got_512.deinit(testing.allocator);
    try expectCheckpointBits(&want_512, &got_512);

    const warm = try dsv41_model.extend(net.m, next[512..], .all);
    defer _ = mlx.mlx_array_free(warm);
    dsv41_model.resetState(net.m);
    var cold_cps: std.ArrayList(SSMCheckpoint) = .empty;
    defer {
        for (cold_cps.items) |*cp| cp.deinit(testing.allocator);
        cold_cps.deinit(testing.allocator);
    }
    try net.prefill(next, &.{ 256, 256 }, 0, &.{}, &cold_cps);
    const cold = try dsv41_model.extend(net.m, next[512..], .all);
    defer _ = mlx.mlx_array_free(cold);
    try glm5_forward.expectArrayBits(cold, warm);
}

test "DeepSeek SSD tier: restore points a prefill writes through leave RAM as they are taken and restore the same bytes (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [700]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var next_buf: [760]u32 = undefined;
    for (&next_buf, 0..) |*t, i| t.* = if (i < 650) first[i] else (net.ids[i % net.ids.len] + 1) % 90;
    const next: []const u32 = &next_buf;

    var want_512: SSMCheckpoint = undefined;
    {
        var hc = try dsv41SsdOnlyCache(io, base, "dsv41-wt");
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        var at: usize = 0;
        for ([_]usize{ 512, 188 }, 0..) |n, i| {
            _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[at .. at + n], .last));
            at += n;
            if (i != 0) continue;
            // The state at 512 as an owned copy, kept for the byte comparison.
            want_512 = try dsv41_prefix.capture(testing.allocator, net.m, at);
            _ = mlx.mlx_synchronize(s);
            var before: usize = 0;
            _ = mlx.mlx_get_active_memory(&before);
            try cps.append(testing.allocator, try dsv41_prefix.captureShared(testing.allocator, net.m, at));
            cache.step = at;
            hc.writeThroughDsv41(net.m, &cache, first[0..at], false, .{}, cps.items, s, "prefill");
            try testing.expect(cps.items[0].isReleased());
            _ = mlx.mlx_synchronize(s);
            var after: usize = 0;
            _ = mlx.mlx_get_active_memory(&after);
            // What the prefill still holds is the tier's row markers (a byte a token for keys and for values) and the point's
            // fixed arrays kept for the next turn's rollback, not the state.
            try testing.expect(after <= before + 2 * at + hc.live_dsv41.bytes());
        }
        cache.step = first.len;
        try dsv41_prefix.markRows(&cache);
        try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    defer want_512.deinit(testing.allocator);

    var hc = try dsv41SsdOnlyCache(io, base, "dsv41-wt");
    defer hc.deinit();
    dsv41_model.resetState(net.m);
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(@as(usize, 512), res.matched);
    var got_512 = try dsv41_prefix.capture(testing.allocator, net.m, 512);
    defer got_512.deinit(testing.allocator);
    try expectCheckpointBits(&want_512, &got_512);

    const warm = try dsv41_model.extend(net.m, next[512..], .all);
    defer _ = mlx.mlx_array_free(warm);
    dsv41_model.resetState(net.m);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, next[0..512], .last));
    const cold = try dsv41_model.extend(net.m, next[512..], .all);
    defer _ = mlx.mlx_array_free(cold);
    try glm5_forward.expectArrayBits(cold, warm);
}

test "DeepSeek SSD tier split: the layers the tier treats as rows are the arrays a capture holds as rows (DSV41_TINY)" {
    try mlx.requireGpu();
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    dsv41_model.resetState(net.m);
    var first_buf: [512]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first, .last));
    var cp = try dsv41_prefix.capture(testing.allocator, net.m, first.len);
    defer cp.deinit(testing.allocator);
    const divs = try dsv41_prefix.rowDivisors(testing.allocator, &net.m.cfg, true);
    defer testing.allocator.free(divs);
    try testing.expectEqual(cp.layers.len, divs.len);
    var rows_layers: usize = 0;
    for (cp.layers, divs) |l, d| for ([_]mlx.mlx_array{ l.conv_state, l.ssm_state }) |a| {
        const sh = mlx.getShape(a);
        if (d == 0) {
            // A fixed layer's arrays are never rows of a compressed history: either a ring or a placeholder.
            continue;
        }
        try testing.expectEqual(@as(usize, 2), sh.len);
        try testing.expectEqual(@as(c_int, @intCast(first.len / d)), sh[0]);
        rows_layers += 1;
    };
    try testing.expect(rows_layers > 0);
    // The window rings and DSpark windows are not rows.
    for (0..net.m.cfg.layers) |li| try testing.expectEqual(@as(u32, 0), divs[4 * li]);
    for (divs[4 * net.m.cfg.layers ..]) |d| try testing.expectEqual(@as(u32, 0), d);
}

test "DeepSeek SSD tier split: every restore point on the grid reads back the bytes a full capture holds, and a restore between two points prefills like cold (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [700]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    // The next turn repeats 400 tokens: between the points at 384 and 512.
    var next_buf: [760]u32 = undefined;
    for (&next_buf, 0..) |*t, i| t.* = if (i < 400) first[i] else (net.ids[i % net.ids.len] + 1) % 90;
    const next: []const u32 = &next_buf;

    const points = [_]usize{ 128, 256, 384, 512, 700 };
    var want: [points.len]SSMCheckpoint = undefined;
    var made: usize = 0;
    defer for (want[0..made]) |*cp| cp.deinit(testing.allocator);
    {
        var hc = try dsv41SplitCache(io, base, "dsv41-split", &net, 128);
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        try net.prefill(first, &.{ 128, 128, 128, 128, 188 }, 0, &.{ 0, 1, 2, 3, 4 }, &cps);
        for (cps.items, 0..) |*cp, i| {
            want[i] = try cloneCheckpointForTest(cp);
            made += 1;
        }
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        cache.step = first.len;
        try dsv41_prefix.markRows(&cache);
        try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        const e = &hc.disk.?.entries.items[0];
        try testing.expectEqualSlices(u32, &.{ 128, 256, 384, 512, 700 }, e.ssm_positions);
        // 700 / 128 whole spans, stored once; the points past them keep the rest.
        try testing.expectEqual(@as(usize, 5), e.grown.len);
        // A point on the grid keeps its fixed arrays only: no bigger than the one before it plus the rings' share of a prompt end.
        try testing.expect(e.ssm_bytes[3] <= e.ssm_bytes[0]);
    }

    var hc = try dsv41SplitCache(io, base, "dsv41-split", &net, 128);
    defer hc.deinit();
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer target.deinit();
    for (points, 0..) |p, i| {
        var one = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer one.deinit();
        var got = try hc.disk.?.restoreCheckpointInto(&one, 0, @intCast(p), dsv41_prefix.snapshotLayers(net.m), s);
        defer got.deinit(testing.allocator);
        try expectCheckpointBits(&want[i], &got);
    }

    dsv41_model.resetState(net.m);
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(@as(usize, 384), res.matched);
    const warm = try dsv41_model.extend(net.m, next[384..], .all);
    defer _ = mlx.mlx_array_free(warm);
    dsv41_model.resetState(net.m);
    // Cold on the grid the points were taken on.
    for (0..3) |c| _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, next[c * 128 ..][0..128], .last));
    const cold = try dsv41_model.extend(net.m, next[384..], .all);
    defer _ = mlx.mlx_array_free(cold);
    try glm5_forward.expectArrayBits(cold, warm);
}

test "DeepSeek SSD tier split: a write-through prefill stages each point with only the rows past the store, and reports the bytes it staged (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [700]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);

    var hc = try dsv41SplitCache(io, base, "dsv41-split-wt", &net, 128);
    defer hc.deinit();
    dsv41_model.resetState(net.m);
    var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer cache.deinit();
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    defer {
        for (cps.items) |*cp| cp.deinit(testing.allocator);
        cps.deinit(testing.allocator);
    }
    var at: usize = 0;
    var seen_ssm: usize = 0;
    var seen_grown: usize = 0;
    // Under the tier's floor nothing is written, so the first point is at 512.
    for ([_]usize{ 512, 188 }, 0..) |n, step| {
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[at .. at + n], .last));
        at += n;
        try cps.append(testing.allocator, try dsv41_prefix.captureShared(testing.allocator, net.m, at));
        cache.step = at;
        const before = hc.disk.?.point_bytes_written;
        hc.writeThroughDsv41(net.m, &cache, first[0..at], false, .{}, cps.items, s, "prefill");
        hc.disk.?.drainWriter();
        const e = &hc.disk.?.entries.items[0];
        // What the call reports is what the entry's files gained: the new point and the new segments.
        var gained: u64 = 0;
        for (e.ssm_bytes[seen_ssm..]) |b| gained += b;
        for (e.grown[seen_grown..]) |b| gained += b;
        try testing.expectEqual(gained, hc.disk.?.point_bytes_written - before);
        // The first point stores four spans; the second only the one past them, not the rows from 0.
        try testing.expectEqual(@as(usize, if (step == 0) 4 else 1), e.grown.len - seen_grown);
        seen_ssm = e.ssm_bytes.len;
        seen_grown = e.grown.len;
        try testing.expect(cps.items[cps.items.len - 1].isReleased());
    }
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entries.items[0].ssm_positions.len);
}

test "DeepSeek decode point: a restore point taken mid-decode serves the next turn from RAM and from an SSD file, like cold on the same grid (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    const first_len = 600;
    var first_buf: [first_len]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    // The turn generates 40 tokens one forward at a time; the thought ends after 24 and the reply goes on to 40.
    var gen: [40]u32 = undefined;
    for (&gen, 0..) |*t, i| t.* = (net.ids[(i * 7 + 1) % net.ids.len] + 3) % 90;
    const point = first_len + 24;
    // The next turn re-renders the thought, then goes its own way.
    var next_buf: [point + 60]u32 = undefined;
    @memcpy(next_buf[0..first_len], first);
    @memcpy(next_buf[first_len..point], gen[0..24]);
    for (next_buf[point..], point..) |*t, i| t.* = (net.ids[i % net.ids.len] + 1) % 90;
    const next: []const u32 = &next_buf;

    var want_point: SSMCheckpoint = undefined;
    var ram = pc.HotPrefixCache.init(testing.allocator, 4);
    defer ram.deinit();
    {
        var hc = try dsv41SsdOnlyCache(io, base, "dsv41-decode");
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[0..512], .last));
        try cps.append(testing.allocator, try dsv41_prefix.captureShared(testing.allocator, net.m, 512));
        cache.step = 512;
        hc.writeThroughDsv41(net.m, &cache, first[0..512], false, .{}, cps.items, s, "prefill");
        try testing.expect(cps.items[0].isReleased());
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[512..], .last));
        for (gen[0..24]) |t| _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, &.{t}, .last));

        cache.step = point;
        var tokens: [point]u32 = undefined;
        @memcpy(tokens[0..first_len], first);
        @memcpy(tokens[first_len..], gen[0..24]);
        want_point = try dsv41_prefix.capture(testing.allocator, net.m, point);
        // The RAM tier gets the same point as an owned copy.
        var ram_cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer ram_cache.deinit();
        ram_cache.step = point;
        const ram_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        ram_cps[0] = try dsv41_prefix.capture(testing.allocator, net.m, point);
        try testing.expectEqual(pc.CommitStatus{ .ok = point }, try ram.commitWithState(&ram_cache, &tokens, false, 0, ram_cps, null, null));

        try hc.writeDecodePointDsv41(&cache, net.m, &tokens, false, .{}, s);
        // The writer has the state; the request goes on decoding from where it stood.
        for (gen[24..]) |t| _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, &.{t}, .last));

        cache.step = first_len + gen.len;
        var all: [first_len + 40]u32 = undefined;
        @memcpy(all[0..first_len], first);
        @memcpy(all[first_len..], &gen);
        try dsv41_prefix.markRows(&cache);
        // The end-of-request commit hands over the prefill's released point and adds none of its own.
        try testing.expectEqual(pc.CommitStatus{ .disk_only = all.len }, try hc.commitWithState(&cache, &all, false, 0, try cps.toOwnedSlice(testing.allocator), null, null));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const e = &hc.disk.?.entries.items[0];
        try testing.expectEqualSlices(u32, &.{ 512, point }, e.ssm_positions);
    }
    defer want_point.deinit(testing.allocator);

    // The cold run replays turn one's forwards up to the point, then the suffix.
    dsv41_model.resetState(net.m);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[0..512], .last));
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[512..], .last));
    for (gen[0..24]) |t| _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, &.{t}, .last));
    const cold = try dsv41_model.extend(net.m, next[point..], .all);
    defer _ = mlx.mlx_array_free(cold);

    // From RAM: the point is the highest under the match, and the suffix runs like cold.
    const from_ram = try dsv41Lookup(&ram, &net, next);
    try testing.expectEqual(point, from_ram.matched);
    const ram_warm = try dsv41_model.extend(net.m, next[point..], .all);
    defer _ = mlx.mlx_array_free(ram_warm);
    try glm5_forward.expectArrayBits(cold, ram_warm);

    // From the file, by a tier that never saw it written.
    var hc = try dsv41SsdOnlyCache(io, base, "dsv41-decode");
    defer hc.deinit();
    dsv41_model.resetState(net.m);
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(point, res.matched);
    try testing.expect(hc.last_restored_disk_id != null);
    var got_point = try dsv41_prefix.capture(testing.allocator, net.m, point);
    defer got_point.deinit(testing.allocator);
    try expectCheckpointBits(&want_point, &got_point);
    const warm = try dsv41_model.extend(net.m, next[point..], .all);
    defer _ = mlx.mlx_array_free(warm);
    try glm5_forward.expectArrayBits(cold, warm);
}

test "DeepSeek live rollback: the state rolled back to each kept point is a full capture of that point, byte for byte (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var turn_buf: [60]u32 = undefined;
    const turn = dsv41LongPrompt(&net, &turn_buf, null, 0);
    const marks = [_]usize{ 12, 20, 30 };

    var want: [marks.len]SSMCheckpoint = undefined;
    dsv41_model.resetState(net.m);
    var at: usize = 0;
    for (marks, &want) |mark, *w| {
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn[at..mark], .last));
        at = mark;
        w.* = try dsv41_prefix.capture(a, net.m, mark);
    }
    defer for (&want) |*w| w.deinit(a);

    for (marks, 0..) |target, t| {
        dsv41_model.resetState(net.m);
        var live: dsv41_prefix.Live = .{};
        defer live.deinit(a);
        at = 0;
        for (marks) |mark| {
            _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn[at..mark], .last));
            at = mark;
            var cp = try dsv41_prefix.capture(a, net.m, mark);
            defer cp.deinit(a);
            try live.keep(a, net.m, &cp);
        }
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn[at..45], .last));
        live.publish(a, net.m, turn[0..45], false, 0, null);

        // The next prompt repeats the turn through the target and three tokens more, then goes its own way.
        var prompt: [50]u32 = undefined;
        @memcpy(prompt[0 .. target + 3], turn[0 .. target + 3]);
        for (prompt[target + 3 ..]) |*tok| tok.* = 91;
        const idx = live.pick(net.m, &prompt, false, std.math.maxInt(u32)) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(t, idx);
        try live.rollback(a, net.m, idx, &prompt);
        try testing.expectEqual(t + 1, live.points.items.len);
        try testing.expect(!live.ready);

        var got = try dsv41_prefix.capture(a, net.m, target);
        defer got.deinit(a);
        try expectCheckpointBits(&want[t], &got);
    }
}

test "DeepSeek live rollback: only a state still standing for the same history serves, at a point the prompt shares (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var turn_buf: [40]u32 = undefined;
    const turn = dsv41LongPrompt(&net, &turn_buf, null, 0);
    var live: dsv41_prefix.Live = .{};
    defer live.deinit(a);
    dsv41_model.resetState(net.m);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn[0..20], .last));
    var cp = try dsv41_prefix.capture(a, net.m, 20);
    defer cp.deinit(a);
    try live.keep(a, net.m, &cp);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn[20..], .last));
    const max: usize = std.math.maxInt(u32);

    // A request is still running over the points: nothing to roll back to.
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn, false, max));
    live.publish(a, net.m, turn, false, 0, null);
    try testing.expectEqual(@as(?usize, 0), live.pick(net.m, turn[0..30], false, max));
    // The point is 20 tokens in: a prompt that leaves the turn earlier, or ends at it, or crosses an image, cannot use it.
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn[0..19], false, max));
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn[0..20], false, max));
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn[0..30], false, 19));
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn[0..30], true, max));
    // Rows past what the prompt's own state would hold are not billed.
    try testing.expect(dsv41_model.liveRowsFit(net.m, 40));
    // Another state took the model's place.
    dsv41_model.resetState(net.m);
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, turn[0..30], false, max));
}

test "DeepSeek live rollback: a turn that diverges inside the last turn's generation rolls back to its deepest kept point and continues like an SSD restore (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    const first_len = 600;
    var first_buf: [first_len]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var gen: [24]u32 = undefined;
    for (&gen, 0..) |*t, i| t.* = (net.ids[(i * 7 + 1) % net.ids.len] + 3) % 90;
    var all: [first_len + gen.len]u32 = undefined;
    @memcpy(all[0..first_len], first);
    @memcpy(all[first_len..], &gen);
    // The template re-renders the generation from its 10th token on.
    var next_buf: [first_len + 40]u32 = undefined;
    @memcpy(next_buf[0 .. first_len + 10], all[0 .. first_len + 10]);
    for (next_buf[first_len + 10 ..], first_len + 10..) |*t, i| t.* = (net.ids[i % net.ids.len] + 1) % 90;
    const next: []const u32 = &next_buf;
    const prompt_end = 570;

    var live_point: SSMCheckpoint = undefined;
    var live_end: SSMCheckpoint = undefined;
    var via_live: mlx.mlx_array = undefined;
    {
        var hc = try dsv41SsdOnlyCache(io, base, "dsv41-live");
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        var at: usize = 0;
        for ([_]usize{ 512, prompt_end }) |point| {
            _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[at..point], .last));
            at = point;
            try cps.append(testing.allocator, try dsv41_prefix.captureShared(testing.allocator, net.m, point));
            cache.step = point;
            hc.writeThroughDsv41(net.m, &cache, first[0..point], false, .{}, cps.items, s, "prefill");
        }
        _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, first[at..], .last));
        for (gen) |t| _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, &.{t}, .last));
        cache.step = all.len;
        try dsv41_prefix.markRows(&cache);
        _ = try hc.commitWithState(&cache, &all, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
        hc.publishLiveDsv41(net.m, &all, false, .{});
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        // The two points' fixed arrays and the history's ids ride beside the model's state.
        const sets = hc.live_dsv41.points.items.len;
        try testing.expectEqual(@as(usize, 2), sets);
        try testing.expect(hc.live_dsv41.bytes() > all.len * @sizeOf(u32));
        try testing.expect(hc.live_dsv41.bytes() <= sets * net.m.cfg.stateFixedBytes(true) + all.len * @sizeOf(u32));

        // The live state serves: the deepest point under the 610 shared tokens, with no stored copy read.
        var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, next, false, 0, null, &.{}, 9, false);
        try testing.expectEqual(@as(usize, prompt_end), res.matched);
        try testing.expectEqual(@as(usize, prompt_end), target.step);
        try testing.expectEqual(@as(usize, prompt_end), moe);
        try testing.expect(hc.last_restored_live);
        try testing.expectEqual(@as(?u64, null), hc.last_restored_disk_id);
        live_point = try dsv41_prefix.capture(testing.allocator, net.m, prompt_end);
        via_live = try dsv41_model.extend(net.m, next[prompt_end..], .all);
        live_end = try dsv41_prefix.capture(testing.allocator, net.m, next.len);
    }
    defer live_point.deinit(testing.allocator);
    defer _ = mlx.mlx_array_free(via_live);
    defer live_end.deinit(testing.allocator);

    // A tier that never saw the write restores the same point from its file.
    var fresh = try dsv41SsdOnlyCache(io, base, "dsv41-live");
    defer fresh.deinit();
    dsv41_model.resetState(net.m);
    var ssd_target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer ssd_target.deinit();
    var ssd_moe: usize = 0;
    const from_ssd = try fresh.lookupAndRestoreDsv41(&ssd_target, &ssd_moe, net.m, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(@as(usize, prompt_end), from_ssd.matched);
    try testing.expect(!fresh.last_restored_live);
    try testing.expect(fresh.last_restored_disk_id != null);
    var ssd_point = try dsv41_prefix.capture(testing.allocator, net.m, prompt_end);
    defer ssd_point.deinit(testing.allocator);
    try expectCheckpointBits(&ssd_point, &live_point);
    const via_ssd = try dsv41_model.extend(net.m, next[prompt_end..], .all);
    defer _ = mlx.mlx_array_free(via_ssd);
    try glm5_forward.expectArrayBits(via_ssd, via_live);
    var ssd_end = try dsv41_prefix.capture(testing.allocator, net.m, next.len);
    defer ssd_end.deinit(testing.allocator);
    try expectCheckpointBits(&ssd_end, &live_end);
}

test "DeepSeek live rollback: another conversation, a replaced state or a point behind a stored one takes the normal path (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [600]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var hc = try dsv41SsdOnlyCache(io, base, "dsv41-live-miss");
    defer hc.deinit();

    const Turn = struct {
        fn run(h: *pc.HotPrefixCache, n: *Dsv41Net, tokens: []const u32) !void {
            dsv41_model.resetState(n.m);
            var cache = try KVCache.initWithConfig(testing.allocator, n.m.cfg.layers, kv_quant.KVQuantConfig.dense);
            defer cache.deinit();
            var cps: std.ArrayList(SSMCheckpoint) = .empty;
            _ = mlx.mlx_array_free(try dsv41_model.extend(n.m, tokens[0..512], .last));
            try cps.append(testing.allocator, try dsv41_prefix.captureShared(testing.allocator, n.m, 512));
            cache.step = 512;
            h.writeThroughDsv41(n.m, &cache, tokens[0..512], false, .{}, cps.items, mlx.gpuStream(), "prefill");
            _ = mlx.mlx_array_free(try dsv41_model.extend(n.m, tokens[512..], .last));
            cache.step = tokens.len;
            try dsv41_prefix.markRows(&cache);
            _ = try h.commitWithState(&cache, tokens, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
            h.publishLiveDsv41(n.m, tokens, false, .{});
            h.flushPendingDisk(mlx.gpuStream());
            h.disk.?.drainWriter();
        }
    };
    var next_buf: [620]u32 = undefined;
    @memcpy(next_buf[0..580], first[0..580]);
    for (next_buf[580..], 580..) |*t, i| t.* = (net.ids[i % net.ids.len] + 1) % 90;

    // Another conversation shares 100 tokens, short of every point: nothing is restored and the points are dropped.
    try Turn.run(&hc, &net, first);
    var other: [620]u32 = undefined;
    @memcpy(other[0..100], first[0..100]);
    for (other[100..]) |*t| t.* = 92;
    try testing.expectEqual(@as(usize, 0), (try dsv41Lookup(&hc, &net, &other)).matched);
    try testing.expect(!hc.last_restored_live);
    try testing.expectEqual(@as(usize, 0), hc.live_dsv41.points.items.len);
    try testing.expectEqual(@as(u64, 0), hc.live_dsv41.bytes());

    // The model's state was replaced after the turn: the stored point serves.
    try Turn.run(&hc, &net, first);
    dsv41_model.resetState(net.m);
    const stored = try dsv41Lookup(&hc, &net, &next_buf);
    try testing.expectEqual(@as(usize, 512), stored.matched);
    try testing.expect(!hc.last_restored_live);
    try testing.expect(hc.last_restored_disk_id != null);
    // The restored point is the first of this request's own.
    try testing.expectEqual(@as(usize, 1), hc.live_dsv41.points.items.len);
    try testing.expectEqual(@as(usize, 512), hc.live_dsv41.points.items[0].pos);

    // The live state serves where it can: the same prompt after a finished turn.
    try Turn.run(&hc, &net, first);
    const again = try dsv41Lookup(&hc, &net, &next_buf);
    try testing.expectEqual(@as(usize, 512), again.matched);
    try testing.expect(hc.last_restored_live);
}

// A cancelled request publishes its live record when the scheduler culls it, before the next exclusive request is
// admitted (`cullDecodingWith`); its tokens are never paired with that request's state, whatever happens to it.
test "DeepSeek live rollback: a cancelled request's record, published at the cull, never serves a request that continues it from the next one's state (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var a_buf: [60]u32 = undefined;
    const turn_a = dsv41LongPrompt(&net, &a_buf, null, 0);
    var b_buf: [60]u32 = undefined;
    for (&b_buf, 0..) |*t, i| t.* = if (i < 10) turn_a[i] else (net.ids[(i * 3 + 1) % net.ids.len] + 7) % 90;
    var live: dsv41_prefix.Live = .{};
    defer live.deinit(a);

    // A: a prefill point at 20, decodes to 40, is cancelled, and publishes at the cull.
    dsv41_model.resetState(net.m);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn_a[0..20], .last));
    {
        var cp = try dsv41_prefix.capture(a, net.m, 20);
        defer cp.deinit(a);
        try live.keep(a, net.m, &cp);
    }
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, turn_a[20..40], .last));
    live.publish(a, net.m, turn_a[0..40], false, 0, null);

    // B shares 10 tokens, short of every point: its lookup drops the record, it prefills cold to a point at 30 and
    // is cancelled mid-prefill, which publishes nothing.
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, &b_buf, false, std.math.maxInt(u32)));
    live.clear(a);
    dsv41_model.resetState(net.m);
    _ = mlx.mlx_array_free(try dsv41_model.extend(net.m, b_buf[0..30], .last));
    {
        var cp = try dsv41_prefix.capture(a, net.m, 30);
        defer cp.deinit(a);
        try live.keep(a, net.m, &cp);
    }

    // C continues A's conversation: B's state is no state for it.
    var c_buf: [50]u32 = undefined;
    @memcpy(c_buf[0..40], turn_a[0..40]);
    for (c_buf[40..]) |*t| t.* = 93;
    try testing.expectEqual(@as(?usize, null), live.pick(net.m, &c_buf, false, std.math.maxInt(u32)));
}

test "DeepSeek SSD tier: the namespace names the layout, and another layout's entries are never read (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const dir = try dsv41_model.fixtureDir();
    var cfg = try model_mod.parseConfig(io, testing.allocator, dir);
    defer cfg.deinit(testing.allocator);
    cfg.stream_mtp_head = true;
    const tag = cfg.cacheLayoutNamespace() orelse return error.TestUnexpectedResult;
    try testing.expectEqualStrings(dsv41_prefix.namespaceTag(cfg.dsparkOn()), tag);
    try testing.expect(!std.mem.eql(u8, dsv41_prefix.namespaceTag(true), dsv41_prefix.namespaceTag(false)));
    const a = try disk_placement.modelFingerprintWithLayout(testing.allocator, io, dir, dsv41_prefix.namespaceTag(true));
    defer testing.allocator.free(a);
    const b = try disk_placement.modelFingerprintWithLayout(testing.allocator, io, dir, dsv41_prefix.namespaceTag(false));
    defer testing.allocator.free(b);
    try testing.expect(!std.mem.eql(u8, a, b));

    // An entry stored under one fingerprint is a miss under the other.
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [600]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    {
        var hc = try dsv41SsdOnlyCache(io, base, a);
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        try net.prefill(first, &.{ 256, 256, 88 }, 0, &.{ 0, 1 }, &cps);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer cache.deinit();
        cache.step = first.len;
        try dsv41_prefix.markRows(&cache);
        _ = try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
    }
    var next_buf: [620]u32 = undefined;
    @memcpy(next_buf[0..600], first);
    for (next_buf[600..]) |*t| t.* = 91;
    for ([_]struct { fp: []const u8, want: usize }{ .{ .fp = a, .want = 512 }, .{ .fp = b, .want = 0 } }) |arm| {
        var hc = try dsv41SsdOnlyCache(io, base, arm.fp);
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, &next_buf, false, 0, null, &.{}, 9, false);
        try testing.expectEqual(arm.want, res.matched);
    }
}

test "DeepSeek with both tiers: the RAM entry serves the active session and, once it is gone, the SSD entry does (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [600]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var next_buf: [620]u32 = undefined;
    @memcpy(next_buf[0..600], first);
    for (next_buf[600..]) |*t| t.* = 91;

    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.hybrid = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "dsv41-both", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    dsv41_model.resetState(net.m);
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    try net.prefill(first, &.{ 256, 256, 88 }, 0, &.{ 0, 1 }, &cps);
    var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv_quant.KVQuantConfig.dense);
    defer cache.deinit();
    cache.step = first.len;
    try dsv41_prefix.markRows(&cache);
    try testing.expectEqual(pc.CommitStatus{ .ok = first.len }, try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null));
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());

    const ram = try dsv41Lookup(&hc, &net, &next_buf);
    try testing.expectEqual(@as(usize, 512), ram.matched);
    try testing.expect(hc.last_restored_disk_id == null);
    // Dropping the RAM entry leaves the disk copy, which restores the same point.
    hc.evictAt(0, "test");
    try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    const disk = try dsv41Lookup(&hc, &net, &next_buf);
    try testing.expectEqual(@as(usize, 512), disk.matched);
    try testing.expect(hc.last_restored_disk_id != null);
}

test "DeepSeek SSD tier under the engine's default KV scheme, which every slot cache carries: commit, restart, restore (DSV41_TINY)" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var net: Dsv41Net = undefined;
    try net.load();
    defer net.deinit();
    var first_buf: [600]u32 = undefined;
    const first = dsv41LongPrompt(&net, &first_buf, null, 0);
    var next_buf: [620]u32 = undefined;
    @memcpy(next_buf[0..600], first);
    for (next_buf[600..]) |*t| t.* = 91;
    const kv = kv_quant.KVQuantConfig.engine_default;
    try testing.expect(kv.scheme != .off);
    {
        var hc = try dsv41SsdOnlyCache(io, base, "dsv41-kv8");
        defer hc.deinit();
        dsv41_model.resetState(net.m);
        var cps: std.ArrayList(SSMCheckpoint) = .empty;
        try net.prefill(first, &.{ 256, 256, 88 }, 0, &.{ 0, 1 }, &cps);
        var cache = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv);
        defer cache.deinit();
        cache.step = first.len;
        try dsv41_prefix.markRows(&cache);
        try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, try hc.commitWithState(&cache, first, false, 0, try cps.toOwnedSlice(testing.allocator), null, null));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    var hc = try dsv41SsdOnlyCache(io, base, "dsv41-kv8");
    defer hc.deinit();
    dsv41_model.resetState(net.m);
    var target = try KVCache.initWithConfig(testing.allocator, net.m.cfg.layers, kv);
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, net.m, s, &next_buf, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(@as(usize, 512), res.matched);
    try testing.expectEqual(@as(usize, 512), (try dsv41_model.ensureState(net.m)).n);
}
