const transformer_mod = @import("../transformer.zig");
const KVCache = transformer_mod.KVCache;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const glm5_forward = @import("glm5_forward.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const kv_quant = @import("../kv_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;
const std = @import("std");
const testing = std.testing;
const pc = @import("../prefix_cache.zig");

/// A GLM fixture turn: `tokens` prefilled in `chunks`, a KDA checkpoint after each chunk listed in
/// `checkpoints` (chunk indices), MLA rows to the request's last pool boundary.
const GlmTurn = struct {
    cps: []SSMCheckpoint,
    rows: glm5_prefix.MlaRows,
    cache: KVCache,

    fn run(net: *const glm5_forward.Model, bits: u8, tokens: []const u32, chunks: []const usize, checkpoints: []const usize) !GlmTurn {
        var req = try glm5_prefix.servedRequest(bits);
        defer req.deinit();
        var cps = try testing.allocator.alloc(SSMCheckpoint, checkpoints.len);
        var built: usize = 0;
        errdefer {
            for (cps[0..built]) |*cp| cp.deinit(testing.allocator);
            testing.allocator.free(cps);
        }
        var at: usize = 0;
        for (chunks, 0..) |n, i| {
            _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &req, tokens[at .. at + n], &.{n}));
            at += n;
            if (std.mem.indexOfScalar(usize, checkpoints, i) != null) {
                cps[built] = try glm5_prefix.captureKda(testing.allocator, &req, mlx.gpuStream());
                built += 1;
            }
        }
        var rows = try glm5_prefix.MlaRows.capture(testing.allocator, &req, req.offset / 4 * 4, mlx.gpuStream());
        errdefer rows.deinit();
        var cache = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        cache.step = req.offset;
        return .{ .cps = cps, .rows = rows, .cache = cache };
    }

    /// Hands `cps` and `rows` to the cache.
    fn commit(self: *GlmTurn, hc: *pc.HotPrefixCache, tokens: []const u32, vision_key: u64, media_start: ?usize) !pc.CommitStatus {
        defer self.cache.deinit();
        return hc.commitGlm(&self.cache, tokens, false, vision_key, 0, media_start, self.cps, self.rows, null, tokens.len);
    }
};

fn glmKvConfig(bits: u8) kv_quant.KVQuantConfig {
    return if (bits == 0) kv_quant.KVQuantConfig.dense else kv_quant.KVQuantConfig.affine(bits);
}

const glm_first = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0, 0 };

const glm_next = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 2, 1, 3, 1, 0, 2 };

const glm_other = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };

fn glmLookup(hc: *pc.HotPrefixCache, req: *glm5_forward.Request, bits: u8, prompt: []const u32, vision_key: u64, media_start: ?usize, chain: []const pc.MediaSpan) !pc.LookupResult {
    var target = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreGlm(&target, &moe, req, mlx.gpuStream(), prompt, false, vision_key, media_start, chain, null, null, false);
    try testing.expectEqual(res.matched, target.step);
    try testing.expectEqual(res.matched, req.offset);
    return res;
}

const GlmNet = struct {
    weights: model_mod.Weights,
    net: glm5_forward.Model,

    fn load(self: *GlmNet) !void {
        self.weights = model_mod.Weights.init(testing.allocator);
        errdefer self.weights.deinit();
        var cfg = try glm5_forward.nonzeroDecodeFixture(&self.weights);
        cfg.max_position_embeddings = 4096;
        self.net = try glm5_forward.Model.load(testing.allocator, cfg, &self.weights, mlx.gpuStream());
    }
    fn deinit(self: *GlmNet) void {
        self.net.deinit();
        self.weights.deinit();
    }
};

fn glmSsdOnlyCache(io: std.Io, base: []const u8, fp: []const u8, max_bytes: u64, budget_from_volume: bool) !pc.HotPrefixCache {
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    errdefer hc.deinit();
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, max_bytes, 128);
    hc.disk.?.ssd_first = budget_from_volume;
    if (budget_from_volume) hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    return hc;
}

/// A GLM prefill driven chunk by chunk as the scheduler drives it: a KDA checkpoint after each chunk, then the write-through.
const GlmPrefill = struct {
    req: glm5_forward.Request,
    cps: std.ArrayList(SSMCheckpoint) = .empty,
    entry_id: ?u64 = null,
    /// Most checkpoint states in RAM at once, read right after a capture.
    peak_held: usize = 0,

    /// The last chunk is the loop's end: its checkpoint is taken after the loop, past the hook, and the
    /// rest of `tokens` is the final span.
    fn run(hc: *pc.HotPrefixCache, net: *const glm5_forward.Model, bits: u8, tokens: []const u32, chunks: []const usize) !GlmPrefill {
        var self = GlmPrefill{ .req = try glm5_prefix.servedRequest(bits) };
        errdefer self.deinit();
        var at: usize = 0;
        for (chunks, 0..) |n, i| {
            _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &self.req, tokens[at .. at + n], &.{n}));
            at += n;
            if (i + 1 < chunks.len) try self.capture();
            hc.writeThroughGlm(&self.entry_id, &self.req, tokens[0..at], false, .{}, self.cps.items, mlx.gpuStream());
        }
        try self.capture();
        if (at < tokens.len) _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &self.req, tokens[at..], &.{tokens.len - at}));
        return self;
    }

    fn capture(self: *GlmPrefill) !void {
        try self.cps.append(testing.allocator, try glm5_prefix.captureKda(testing.allocator, &self.req, mlx.gpuStream()));
        self.peak_held = @max(self.peak_held, self.held());
    }

    fn held(self: *const GlmPrefill) usize {
        var n: usize = 0;
        for (self.cps.items) |*cp| n += @intFromBool(!cp.isReleased());
        return n;
    }

    /// The request's end-of-request commit for the SSD tier alone, as `commitGlmSlot` builds it.
    fn commitSsdOnly(self: *GlmPrefill, hc: *pc.HotPrefixCache, bits: u8, tokens: []const u32) !pc.CommitStatus {
        var cache = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        defer cache.deinit();
        cache.step = self.req.offset;
        const disk_rows = try glm5_prefix.MlaRows.shareLive(testing.allocator, &self.req, glm5_prefix.commitRows(self.cps.items, self.req.offset), mlx.gpuStream());
        const owned = try self.cps.toOwnedSlice(testing.allocator);
        return hc.commitGlmTiers(&cache, tokens, false, 0, 0, null, owned, null, disk_rows, null, tokens.len);
    }

    fn deinit(self: *GlmPrefill) void {
        for (self.cps.items) |*cp| cp.deinit(testing.allocator);
        self.cps.deinit(testing.allocator);
        self.req.deinit();
    }
};

/// A prompt that shares `shared` tokens with `first` and then differs.
fn glmDiverging(buf: []u32, first: []const u32, shared: usize) []const u32 {
    @memcpy(buf[0..shared], first[0..shared]);
    for (buf[shared..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    return buf;
}

/// Restores `prompt` from `hc` at `at` and checks the continuation against a cold prefill of it, bit for bit.
fn expectGlmRestoreLikeCold(hc: *pc.HotPrefixCache, net: *const glm5_forward.Model, bits: u8, prompt: []const u32, at: usize) !void {
    var warm = try glm5_prefix.servedRequest(bits);
    defer warm.deinit();
    const res = try glmLookup(hc, &warm, bits, prompt, 0, null, &.{});
    try testing.expectEqual(at, res.matched);
    const got = try glm5_prefix.prefill(net, &warm, prompt[at..], &.{prompt.len - at});
    defer _ = mlx.mlx_array_free(got);
    var cold = try glm5_prefix.servedRequest(bits);
    defer cold.deinit();
    const want = try glm5_prefix.prefill(net, &cold, prompt, &.{ at, prompt.len - at });
    defer _ = mlx.mlx_array_free(want);
    try glm5_forward.expectArrayBits(want, got);
    try glm5_prefix.expectSameLogicalState(&cold, &warm);
}

test "GLM hot cache restores the highest checkpoint under the match and continues like a cold prefill" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    for ([_]u8{ 0, 8 }) |bits| {
        var hc = pc.HotPrefixCache.init(testing.allocator, 4);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, bits, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
        try testing.expectEqual(pc.CommitStatus{ .ok = glm_first.len }, try turn.commit(&hc, &glm_first, 0, null));
        const e = &hc.entries.items[0];
        var cps_bytes: u64 = 0;
        for (e.ssm_checkpoints.?) |*cp| cps_bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(e.glm.?.bytes() + cps_bytes, e.kv_bytes);
        try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);

        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &glm_next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 16), res.matched);
        const got = try glm5_prefix.prefill(&fx.net, &warm, glm_next[16..], &.{8});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &glm_next, &.{ 8, 8, 8 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);

        // The other latent width never restores this entry, and its miss leaves the request cold.
        const other_bits: u8 = if (bits == 0) 8 else 0;
        var other = try glm5_prefix.servedRequest(other_bits);
        defer other.deinit();
        try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &other, other_bits, &glm_next, 0, null, &.{})).matched);
    }
}

test "GLM hot cache trims an entry over the byte budget to a checkpoint and bills what it keeps" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var long: [300]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var probe = try GlmTurn.run(&fx.net, 8, &long, &.{ 264, 32, 4 }, &.{ 0, 1 });
    const row = probe.rows.rowBytes();
    const per_cp = ssmCheckpointBytes(&probe.cps[0]);
    probe.cache.deinit();
    probe.rows.deinit();
    for (probe.cps) |*cp| cp.deinit(testing.allocator);
    testing.allocator.free(probe.cps);

    // The assistant window survives the trim and is priced inside the budget.
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try pc.testFillCache(&window, mlx.gpuStream(), 2, 64);
    var window_snap = try window.snapshot();
    defer window_snap.deinit();
    const window_bytes = pc.HotPrefixCache.snapshotBytes(&window_snap);

    // Room for the window, the rows to 264 and one checkpoint, not for 300 rows and two.
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, window_bytes + 264 * row + per_cp);
    defer hc.deinit();
    var turn = try GlmTurn.run(&fx.net, 8, &long, &.{ 264, 32, 4 }, &.{ 0, 1 });
    defer turn.cache.deinit();
    try testing.expectEqual(pc.CommitStatus{ .ok = 264 }, try hc.commitGlm(&turn.cache, &long, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 236 }, long.len));
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 264), e.glm.?.len);
    try testing.expect(e.dflash != null);
    try testing.expectEqual(window_bytes + 264 * row + per_cp, e.kv_bytes);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 264), (try glmLookup(&hc, &warm, 8, &long, 0, null, &.{})).matched);
}

test "GLM hot cache evicts by count, invalidates, and a miss leaves the request cold" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = pc.HotPrefixCache.init(testing.allocator, 1);
    defer hc.deinit();
    var a = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try a.commit(&hc, &glm_first, 0, null);
    var b = try GlmTurn.run(&fx.net, 0, &glm_other, &.{ 8, 4 }, &.{0});
    _ = try b.commit(&hc, &glm_other, 0, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 0, &glm_next, 0, null, &.{})).matched);
    const again = glm_other ++ [_]u32{1};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    hc.invalidateAll("test");
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 0), warm.layers[3].attention.processed);
}

test "GLM hot cache gives entries back LRU first to admit a long prefill, sparing the one it restored from" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    var a = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try a.commit(&hc, &glm_first, 0, null);
    var b = try GlmTurn.run(&fx.net, 0, &glm_other, &.{ 8, 4 }, &.{0});
    _ = try b.commit(&hc, &glm_other, 0, null);
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    const again = glm_other ++ [_]u32{1};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    const Room = struct {
        fn oneEntry(ctx: ?*anyopaque) bool {
            const cache: *pc.HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() <= 1;
        }
        fn never(_: ?*anyopaque) bool {
            return false;
        }
    };
    const report = hc.evictLruToAdmit(again.len, &hc, Room.oneEntry, true);
    try testing.expect(report.admitted);
    try testing.expectEqual(@as(usize, 1), report.entries);
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    try testing.expect(!hc.evictLruToAdmit(again.len, &hc, Room.never, true).admitted);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
}

test "GLM hot cache shares across media keys only below the first block the request lacks" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    // One media block at 10 (key 0xA); the next turn appends a second block at 20 (key 0xB).
    var turn = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try turn.commit(&hc, &glm_first, 0xA, 10);
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    const chain = [_]pc.MediaSpan{ .{ .start = 10, .key = 0xA }, .{ .start = 20, .key = 0xB } };
    try testing.expectEqual(@as(usize, 16), (try glmLookup(&hc, &warm, 0, &glm_next, 0xB, 10, &chain)).matched);
    const stranger = [_]pc.MediaSpan{.{ .start = 10, .key = 0xC }};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &glm_next, 0xC, 10, &stranger)).matched);
}

test "GLM hot cache checkout hands the MLA rows to the slot so its first append donates" {
    try mlx.requireGpu();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
    var hc = pc.HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    hc.ssd_first = true;
    var turn = try GlmTurn.run(&fx.net, 8, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try turn.commit(&hc, &glm_first, 0, null);
    const billed = hc.entries.items[0].glm_bytes;
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    var target = try KVCache.initWithConfig(testing.allocator, 4, kv_quant.KVQuantConfig.affine(8));
    defer target.deinit();
    var moe: usize = 0;
    const longer = glm_first ++ [_]u32{ 2, 2, 2, 2 };
    const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, mlx.gpuStream(), &longer, false, 0, null, &.{}, null, 7, false);
    try testing.expect(res.checked_out);
    try testing.expectEqual(@as(usize, 16), res.matched);
    hc.donateCheckout(7);
    try testing.expectEqual(@as(u64, 0), hc.entries.items[0].glm.?.bytes());
    try testing.expectEqual(billed, hc.donatedBytes(7));
    hc.releaseCheckout(7, "test");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "GLM SSD tier restores a checkpoint after the RAM copy is gone and continues like a cold prefill" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    for ([_]u8{ 0, 8, 6 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "fp-glm-{d}", .{bits});
        {
            var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
            defer hc.deinit();
            var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
            _ = try turn.commit(&hc, &first, 0, null);
            hc.flushPendingDisk(s);
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        }
        // A restart: nothing in RAM.
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
        defer hc.deinit();
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
        for ([_]u8{ 0, 8, 6 }) |other_bits| {
            if (other_bits == bits) continue;
            var other = try glm5_prefix.servedRequest(other_bits);
            defer other.deinit();
            try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &other, other_bits, &next, 0, null, &.{})).matched);
        }
    }
}

test "GLM SSD tier: one fingerprint dir serves each latent width its own entry, and a lookup at another width misses" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    for ([_]u8{ 6, 8, 0 }) |written| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = buf[0..try tmp.dir.realPath(io, &buf)];
        {
            var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-shared", 0, 128);
            defer hc.deinit();
            var turn = try GlmTurn.run(&fx.net, written, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
            _ = try turn.commit(&hc, &first, 0, null);
            hc.flushPendingDisk(s);
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        }
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-shared", 0, 128);
        defer hc.deinit();
        for ([_]u8{ 0, 8, 6 }) |asked| {
            var req = try glm5_prefix.servedRequest(asked);
            defer req.deinit();
            const want: usize = if (asked == written) 512 else 0;
            try testing.expectEqual(want, (try glmLookup(&hc, &req, asked, &next, 0, null, &.{})).matched);
        }
    }
}

test "GLM entries carry the assistant window, and a restore hands back its rows below the checkpoint, RAM and SSD" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    // The window as prefill left it: 300 rows ending at the prompt end.
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try pc.testFillCache(&window, s, 2, 300);
    var committing = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    committing.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
    defer turn.cache.deinit();
    _ = try committing.commitGlm(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
    committing.flushPendingDisk(s);
    var restarted = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer restarted.deinit();
    for ([_]*pc.HotPrefixCache{ &committing, &restarted }) |hc| {
        if (hc == &restarted) {
            committing.deinit();
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
        }
        var drafter = try KVCache.init(testing.allocator, 2);
        defer drafter.deinit();
        var drafter_base: usize = 0;
        var warm = try glm5_prefix.servedRequest(8);
        defer warm.deinit();
        var target = try KVCache.initWithConfig(testing.allocator, 4, kv_quant.KVQuantConfig.affine(8));
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, s, &next, false, 0, null, &.{}, .{ .cache = &drafter, .base_pos = &drafter_base }, null, false);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(?usize, 400), res.dflash_base);
        try testing.expectEqual(@as(usize, 112), drafter.step);
    }
}

test "GLM SSD-only: a commit reaches disk with no idle RAM entry, and a restart restores it like a cold prefill" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try pc.testFillCache(&window, s, 2, 300);
    for ([_]u8{ 0, 8, 6 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "glm-ssd-only-{d}", .{bits});
        {
            var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
            defer hc.deinit();
            var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
            defer turn.cache.deinit();
            const status = try hc.commitGlm(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
            try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, status);
            try testing.expectEqual(@as(usize, 0), hc.entryCount());
            try testing.expectEqual(@as(u64, 0), hc.residentBytes());
            hc.flushPendingDisk(s);
            hc.disk.?.drainWriter();
            try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
            try testing.expect(hc.pending_disk == null);
        }
        var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
        defer hc.deinit();
        var drafter = try KVCache.init(testing.allocator, 2);
        defer drafter.deinit();
        var drafter_base: usize = 0;
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        var target = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, s, &next, false, 0, null, &.{}, .{ .cache = &drafter, .base_pos = &drafter_base }, 9, false);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        try testing.expectEqual(@as(?usize, 400), res.dflash_base);
        try testing.expectEqual(@as(usize, 112), drafter.step);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
    }
}

test "GLM: the SSD tier gets every row in one flush whatever the RAM tier keeps, and the longer tier restores" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const Arm = struct { ram: bool, fp: []const u8 };
    for ([_]u8{ 0, 8 }) |bits| for ([_]Arm{ .{ .ram = true, .fp = "glm-tiers-both" }, .{ .ram = false, .fp = "glm-tiers-ssd" } }) |arm| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "{s}-{d}", .{ arm.fp, bits });
        var hc = if (arm.ram) pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0) else try glmSsdOnlyCache(io, base, fp, 0, false);
        defer hc.deinit();
        if (arm.ram) {
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
            hc.disk.?.enableBackgroundWriter();
        }
        hc.disk.?.max_flush_bytes = 1; // one chunk per piece
        var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
        defer turn.cache.deinit();
        // The RAM tier keeps the rows through the first checkpoint; the SSD tier gets them all.
        const ram_rows: ?glm5_prefix.MlaRows = if (arm.ram) try turn.rows.trimmedCopy(256, s) else null;
        const status = try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, ram_rows, turn.rows, null, first.len);
        if (arm.ram) {
            try testing.expectEqual(pc.CommitStatus{ .ok = first.len }, status);
            try testing.expectEqual(@as(usize, 256), hc.entries.items[0].glm.?.len);
            try testing.expectEqual(@as(usize, 1), hc.entries.items[0].ssm_checkpoints.?.len);
        } else {
            try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, status);
            try testing.expectEqual(@as(usize, 0), hc.entryCount());
        }
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const on_disk = &hc.disk.?.entries.items[0];
        try testing.expectEqual(@as(u32, 700), on_disk.kv_len);
        try testing.expectEqualSlices(u32, &.{ 256, 512 }, on_disk.ssm_positions);

        // The SSD tier's checkpoint at 512 beats the RAM entry's at 256.
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
    };
}

test "GLM SSD-only: a prefill keeps at most two checkpoint states in RAM, and a restore at a released one reads SSD like a cold prefill" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    for ([_]u8{ 0, 8 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "glm-wt-{d}", .{bits});
        var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
        defer hc.deinit();
        var turn = try GlmPrefill.run(&hc, &fx.net, bits, &first, &.{ 512, 64, 64, 48 });
        defer turn.deinit();
        try testing.expect(turn.peak_held <= 2);
        // The checkpoint of the last hooked chunk and the loop-end one beside it.
        try testing.expectEqual(@as(usize, 2), turn.held());
        for (turn.cps.items[0 .. turn.cps.items.len - 2]) |cp| try testing.expect(cp.isReleased());
        try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, try turn.commitSsdOnly(&hc, bits, &first));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const on_disk = &hc.disk.?.entries.items[0];
        try testing.expectEqual(@as(u32, 688), on_disk.kv_len);
        try testing.expectEqualSlices(u32, &.{ 512, 576, 640, 688 }, on_disk.ssm_positions);

        // The oldest, a middle one and the newest: each state comes off the disk.
        var other: [800]u32 = undefined;
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..600], &first, 530), 512);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..680], &first, 650), 640);
        @memcpy(other[0..700], &first);
        for (other[700..750], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, other[0..750], 688);
    }
}

test "GLM RAM tier beside an SSD tier judges the byte budget by the two checkpoints that stay" {
    try mlx.requireGpu();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var probe = try GlmTurn.run(&fx.net, 8, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
    // Room for every row and two checkpoints, not for four.
    const budget = probe.rows.bytes() + 2 * ssmCheckpointBytes(&probe.cps[0]) + 1;
    probe.rows.deinit();
    for (probe.cps) |*cp| cp.deinit(testing.allocator);
    testing.allocator.free(probe.cps);
    probe.cache.deinit();

    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-price", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    // The same session twice: the second commit replaces the first instead of being judged oversized and dropped.
    for (0..2) |_| {
        var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
        defer turn.cache.deinit();
        const disk_rows = try turn.rows.share();
        try testing.expectEqual(pc.CommitStatus{ .ok = first.len }, try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, disk_rows, null, first.len));
        try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ssm_checkpoints.?.len);
        try testing.expect(hc.current_kv_bytes <= budget);
    }
}

test "GLM SSD-only: past the checkpoint cap the loop-end checkpoint still reaches disk" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [1400]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var hc = try glmSsdOnlyCache(io, base, "glm-wt-cap", 0, true);
    defer hc.deinit();
    hc.disk.?.ssm_max_per_entry = glm5_prefix.checkpoint_cap;
    var chunks: [15]usize = @splat(64);
    chunks[0] = 512;
    chunks[14] = 40;
    // 512 + 13 x 64 + 40 = 1384: thirteen grid checkpoints, the loop-end one after them, a 16-token final span.
    var turn = try GlmPrefill.run(&hc, &fx.net, 8, &first, &chunks);
    defer turn.deinit();
    try testing.expect(turn.peak_held <= 2);
    _ = try turn.commitSsdOnly(&hc, 8, first[0..1400]);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const on_disk = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(usize, glm5_prefix.checkpoint_cap), on_disk.ssm_positions.len);
    try testing.expectEqual(@as(u32, 1384), on_disk.ssm_positions[on_disk.ssm_positions.len - 1]);
    var other: [1500]u32 = undefined;
    @memcpy(other[0..1400], &first);
    for (other[1400..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    try expectGlmRestoreLikeCold(&hc, &fx.net, 8, &other, 1384);
}

test "GLM SSD-only: a restore at an 8192 grid point reads SSD like a cold prefill" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    fx.net.cfg.max_position_embeddings = 2 * kv_disk_cache.POINT_GRID;
    var first: [8300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var hc = try glmSsdOnlyCache(io, base, "glm-wt-grid", 0, true);
    defer hc.deinit();
    hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
    // The grid point closes the first chunk; the loop's end follows it before the final span.
    var turn = try GlmPrefill.run(&hc, &fx.net, 8, &first, &.{ kv_disk_cache.POINT_GRID, 40 });
    defer turn.deinit();
    try testing.expect(turn.peak_held <= 2);
    _ = try turn.commitSsdOnly(&hc, 8, &first);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    const on_disk = &hc.disk.?.entries.items[0];
    try testing.expectEqualSlices(u32, &.{ kv_disk_cache.POINT_GRID, kv_disk_cache.POINT_GRID + 40 }, on_disk.ssm_positions);
    var other: [8400]u32 = undefined;
    try expectGlmRestoreLikeCold(&hc, &fx.net, 8, glmDiverging(&other, &first, kv_disk_cache.POINT_GRID + 8), kv_disk_cache.POINT_GRID);
}

test "GLM SSD-only: a checkpoint whose write failed is not a restore point" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var hc = try glmSsdOnlyCache(io, base, "glm-wt-fail", 0, true);
    defer hc.deinit();
    hc.disk.?.writer.?.injectFailure("s0000512", .write);
    var turn = try GlmPrefill.run(&hc, &fx.net, 8, &first, &.{ 512, 64, 112 });
    defer turn.deinit();
    try testing.expect(turn.peak_held <= 2);
    _ = try turn.commitSsdOnly(&hc, 8, &first);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const on_disk = &hc.disk.?.entries.items[0];
    try testing.expect(!on_disk.poisoned);
    try testing.expectEqualSlices(u32, &.{ 576, 688 }, on_disk.ssm_positions);
    try testing.expectEqual(@as(u32, 688), on_disk.kv_len);

    var other: [800]u32 = undefined;
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 8, glmDiverging(other[0..600], &first, 530), 0, null, &.{})).matched);
    try expectGlmRestoreLikeCold(&hc, &fx.net, 8, glmDiverging(other[0..680], &first, 650), 576);
}

test "GLM RAM tier beside an SSD tier: the hot entry keeps its two newest checkpoints, the SSD tier all of them" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    for ([_]u8{ 0, 8 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "glm-ram-newest-{d}", .{bits});
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
        hc.disk.?.enableBackgroundWriter();
        var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
        defer turn.cache.deinit();
        const disk_rows = try turn.rows.share();
        try testing.expectEqual(pc.CommitStatus{ .ok = first.len }, try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, disk_rows, null, first.len));
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(usize, 2), e.ssm_checkpoints.?.len);
        try testing.expectEqualSlices(usize, &.{ 640, 700 }, &.{ e.ssm_checkpoints.?[0].pos, e.ssm_checkpoints.?[1].pos });
        var bytes: u64 = 0;
        for (e.ssm_checkpoints.?) |*cp| bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(bytes, e.ssm_bytes);
        try testing.expectEqual(e.glm.?.bytes() + bytes, e.kv_bytes);
        try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqualSlices(u32, &.{ 512, 576, 640, 700 }, hc.disk.?.entries.items[0].ssm_positions);

        // Under the RAM entry's oldest checkpoint the SSD tier restores.
        var other: [800]u32 = undefined;
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..600], &first, 530), 512);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..680], &first, 600), 576);
    }
}

test "GLM SSD-only: two commits before one flush (two cancels culled in one pass) both reach disk" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var prompts: [2][600]u32 = undefined;
    for (&prompts, 0..) |*p, k| for (p, 0..) |*t, i| {
        t.* = @intCast((i * (k + 3) + k + i / 7) % 4);
    };
    var hc = try glmSsdOnlyCache(io, base, "glm-two-cancels", 0, false);
    defer hc.deinit();
    for (&prompts) |*p| {
        var turn = try GlmTurn.run(&fx.net, 8, p, &.{ 512, 88 }, &.{0});
        try testing.expectEqual(pc.CommitStatus{ .disk_only = p.len }, try turn.commit(&hc, p, 0, null));
    }
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.pending_disk == null);
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    for (&prompts) |*p| {
        const again = p.* ++ [_]u32{1};
        try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again, 0, null, &.{})).matched);
    }
}

test "GLM SSD-only without the writer: one flush finishes the whole entry before its source goes" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);

    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-no-writer", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.max_flush_bytes = 1; // one chunk per synchronous piece
    try testing.expect(hc.disk.?.writer == null);
    var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
    try testing.expectEqual(pc.CommitStatus{ .disk_only = first.len }, try turn.commit(&hc, &first, 0, null));
    hc.flushPendingDisk(s);
    try testing.expect(hc.pending_disk == null);
    const e = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(u32, 700), e.kv_len);
    try testing.expectEqualSlices(u32, &.{ 256, 512 }, e.ssm_positions);
    try testing.expect(hc.disk.?.entryWholeOnDisk(e.id));
}

test "a GLM commit hands the SSD tier every row through its newest checkpoint, past any one flush" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const cp = struct {
        fn at(pos: usize) SSMCheckpoint {
            return .{ .pos = pos, .layers = &.{} };
        }
    }.at;
    const cps = [_]SSMCheckpoint{ cp(2048), cp(524_288), cp(1_040_000) };
    const row: u64 = 6688;
    {
        var hc = try glmSsdOnlyCache(io, base, "glm-len-ssd", 20 << 30, false);
        defer hc.deinit();
        try testing.expectEqual(@as(?usize, null), hc.glmCommitLen(&cps, 1_040_003, row, 0));
        try testing.expectEqual(@as(?usize, 1_040_000), hc.glmDiskLen(&cps, 1_040_003, row, 0));
    }
    {
        // Both tiers: the RAM tier keeps what its 1 GiB holds, the SSD tier every row.
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 1 << 30);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-len-both", 20 << 30, 128);
        try testing.expectEqual(@as(?usize, 2048), hc.glmCommitLen(&cps, 1_040_003, row, 0));
        try testing.expectEqual(@as(?usize, 1_040_000), hc.glmDiskLen(&cps, 1_040_003, row, 0));
    }
}

test "GLM SSD-only: the disk budget evicts the least recently used conversation" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var prompts: [3][600]u32 = undefined;
    for (&prompts, 0..) |*p, k| for (p, 0..) |*t, i| {
        t.* = @intCast((i * (k + 3) + k + i / 7) % 4);
    };
    // Measure one entry, then allow two.
    var one: u64 = 0;
    {
        var hc = try glmSsdOnlyCache(io, base, "glm-ssd-probe", 0, false);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, 8, &prompts[0], &.{ 512, 88 }, &.{0});
        _ = try turn.commit(&hc, &prompts[0], 0, null);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        one = hc.disk.?.total_bytes;
    }
    var hc = try glmSsdOnlyCache(io, base, "glm-ssd-budget", one * 5 / 2, false);
    defer hc.deinit();
    for (&prompts) |*p| {
        var turn = try GlmTurn.run(&fx.net, 8, p, &.{ 512, 88 }, &.{0});
        _ = try turn.commit(&hc, p, 0, null);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
    }
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.total_bytes <= one * 5 / 2);
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    const again: [3][601]u32 = .{ prompts[0] ++ [_]u32{1}, prompts[1] ++ [_]u32{1}, prompts[2] ++ [_]u32{1} };
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 8, &again[0], 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again[1], 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again[2], 0, null, &.{})).matched);
}

test "GLM's 1 GiB RAM tier keeps a 30K or 60K session at its prompt end and sheds checkpoints, a longer one trims" {
    const cp: u64 = 147_619_840;
    const row: u64 = 6688;
    const window: u64 = 2047 * 5 * 8 * 128 * 2 * 2;
    const budget: u64 = (1 << 30) - window;
    const Case = struct { positions: [8]usize, kept: usize, cps: u64 };
    const cases = [_]Case{
        .{ .positions = .{ 2048, 8192, 14336, 20480, 24576, 26624, 28672, 30688 }, .kept = 30688, .cps = 5 },
        .{ .positions = .{ 2048, 14336, 26624, 38912, 51200, 55296, 59392, 61408 }, .kept = 61408, .cps = 4 },
        .{ .positions = .{ 2048, 34816, 67584, 100352, 120832, 133120, 137216, 140256 }, .kept = 120832, .cps = 1 },
    };
    const bytes: [8]u64 = @splat(cp);
    for (cases) |c| {
        const tl = pc.HotPrefixCache.trimLenForBudgetPure(budget, std.math.maxInt(usize), row, &c.positions, &bytes, .min_span_recency, null);
        try testing.expectEqual(@as(?usize, c.kept), tl);
        const k = std.mem.indexOfScalar(usize, &c.positions, c.kept).? + 1;
        const survivors = pc.HotPrefixCache.shedSurvivorBytes(c.positions[0..k], bytes[0..k], budget - c.kept * row, .min_span_recency).?;
        try testing.expectEqual(c.cps * cp, survivors);
    }
}

test "a GLM commit copies only the rows its destination tier keeps, chosen before the copy" {
    const t = testing;
    const row: u64 = 6688;
    const cp: u64 = 147_619_840;
    const gib: u64 = 1 << 30;
    const positions = [_]usize{ 2048, 131072, 262144, 524288, 1_040_000 };
    const bytes: [5]u64 = @splat(cp);
    const policy: transformer_mod.ThinPolicy = .min_span_recency;
    // Everything fits: every row to the newest checkpoint.
    try t.expectEqual(@as(?usize, 64), pc.HotPrefixCache.glmCommitLenFor(policy, gib, false, 64, row, 0, &.{64}, &.{cp}));
    try t.expectEqual(@as(?usize, 1_040_000), pc.HotPrefixCache.glmCommitLenFor(policy, 0, false, 1_040_000, row, 0, &positions, &bytes));
    // A 1 GiB RAM tier keeps the newest checkpoint whose rows and a surviving checkpoint fit, never the full 7 GB.
    const ram = pc.HotPrefixCache.glmCommitLenFor(policy, gib, false, 1_040_000, row, 0, &positions, &bytes).?;
    try t.expectEqual(@as(usize, 131072), ram);
    try t.expect(@as(u64, ram) * row + cp <= gib);
    // The window comes out of the budget first.
    try t.expectEqual(@as(?usize, 2048), pc.HotPrefixCache.glmCommitLenFor(policy, gib, false, 1_040_000, row, 200 << 20, &positions, &bytes));
    // The SSD tier keeps every row through the newest checkpoint; only its own byte budget bounds
    // them, priced with every checkpoint it keeps at or below.
    try t.expectEqual(@as(?usize, 1_040_000), pc.HotPrefixCache.glmCommitLenFor(policy, 0, true, 1_040_000, row, 0, &positions, &bytes));
    try t.expectEqual(@as(?usize, 1_040_000), pc.HotPrefixCache.glmCommitLenFor(policy, 20 * gib, true, 1_040_000, row, 0, &positions, &bytes));
    // 262,144 rows (1.75 GB) fit 2 GiB alone; with their three checkpoints (0.44 GB) they do not.
    try t.expectEqual(@as(?usize, 131072), pc.HotPrefixCache.glmCommitLenFor(policy, 2 * gib, true, 1_040_000, row, 0, &positions, &bytes));
    // A 1 GiB tier and ~131K rows with eight checkpoints: 0.82 GiB of rows, 1.92 GiB in all.
    const eight = [_]usize{ 2048, 18432, 43008, 69632, 92160, 116736, 126976, 131040 };
    const eight_bytes: [8]u64 = @splat(cp);
    const kept = pc.HotPrefixCache.glmCommitLenFor(policy, gib, true, 131040, row, 0, &eight, &eight_bytes).?;
    var priced: u64 = @as(u64, kept) * row;
    for (eight) |p| priced += if (p <= kept) cp else 0;
    try t.expect(priced <= gib);
    try t.expect(kept < 131040);
    try t.expectEqual(@as(?usize, null), pc.HotPrefixCache.glmCommitLenFor(policy, gib, false, 0, row, 0, &.{}, &.{}));
}

test "a GLM commit whose retention snapshot fails still destroys every transferred checkpoint once" {
    try mlx.requireGpu();
    const cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    for (cps, 0..) |*cp, i| {
        const layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 2);
        for (layers) |*l| l.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        cp.* = .{ .pos = (i + 1) * 8, .layers = layers };
    }
    const rows: glm5_prefix.MlaRows = .{
        .allocator = testing.allocator,
        .layers = try testing.allocator.alloc(glm5_prefix.MlaRows.Layer, 2),
        .len = 8,
        .latent_bits = 8,
    };
    @memset(rows.layers, .{});

    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
    var source = try KVCache.initWithConfig(failing.allocator(), 2, glmKvConfig(8));
    // The next allocation is the retention snapshot's.
    failing.fail_index = failing.alloc_index;
    defer source.deinit();

    var hc = pc.HotPrefixCache.init(testing.allocator, 2);
    defer hc.deinit();
    const tokens: [24]u32 = @splat(1);
    try testing.expectError(error.OutOfMemory, hc.commitGlm(&source, &tokens, false, 0, 0, null, cps, rows, null, tokens.len));
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "a GLM replace commit that fails at any allocation leaks nothing and leaves the cache consistent" {
    try mlx.requireGpu();
    const Stub = struct {
        fn cps(pos: usize) ![]SSMCheckpoint {
            const out = try testing.allocator.alloc(SSMCheckpoint, 1);
            const layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 1);
            layers[0] = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            out[0] = .{ .pos = pos, .layers = layers };
            return out;
        }
        fn rows() !glm5_prefix.MlaRows {
            const layers = try testing.allocator.alloc(glm5_prefix.MlaRows.Layer, 1);
            @memset(layers, .{});
            return .{ .allocator = testing.allocator, .layers = layers, .len = 8, .latent_bits = 8 };
        }
    };
    const tokens: [24]u32 = @splat(1);
    var k: usize = 0;
    while (k < 64) : (k += 1) {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
        var hc = pc.HotPrefixCache.init(failing.allocator(), 2);
        defer hc.deinit();
        var source = try KVCache.initWithConfig(testing.allocator, 1, glmKvConfig(8));
        defer source.deinit();
        _ = try hc.commitGlm(&source, tokens[0..16], false, 0, 0, null, try Stub.cps(8), try Stub.rows(), null, 16);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());

        failing.fail_index = failing.alloc_index + k;
        const res = hc.commitGlm(&source, &tokens, false, 0, 0, null, try Stub.cps(16), try Stub.rows(), null, 24);
        var sum: u64 = 0;
        for (hc.entries.items) |e| sum += e.kv_bytes;
        try testing.expectEqual(sum, hc.current_kv_bytes);
        if (res) |_| break else |err| try testing.expectEqual(error.OutOfMemory, err);
    }
    try testing.expect(k < 64);
}

test "GLM image turn persists its whole turn, and a foreign key restarts like a cold prefill below its item" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try pc.testFillCache(&window, s, 2, 300);
    {
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.hybrid = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-media", 0, 128);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
        defer turn.cache.deinit();
        _ = try hc.commitGlm(&turn.cache, &first, false, 0xABCD, 0, 600, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, first.len), hc.disk.?.entries.items[0].kv_len);
        // The record is not cut at the item any more, so the dflash window covers its rows.
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
    }
    var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.hybrid = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-media", 0, 128);
    defer hc.deinit();
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    const res = try glmLookup(&hc, &warm, 8, &next, 0x1234, 600, &.{});
    try testing.expectEqual(@as(usize, 512), res.matched);
    const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
    defer _ = mlx.mlx_array_free(got);
    var cold = try glm5_prefix.servedRequest(8);
    defer cold.deinit();
    const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
    defer _ = mlx.mlx_array_free(want);
    try glm5_forward.expectArrayBits(want, got);
    try glm5_prefix.expectSameLogicalState(&cold, &warm);
}
