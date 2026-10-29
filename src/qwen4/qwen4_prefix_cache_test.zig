const pc = @import("../prefix_cache.zig");
const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const log = @import("../log.zig");
const restore_dump = @import("../restore_dump.zig");
const prefill_plan = @import("../prefill_plan.zig");
const KVCache = transformer_mod.KVCache;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const checkpointHasQsaPooled = transformer_mod.checkpointHasQsaPooled;
const sliceQsaHistoryOntoCheckpoint = transformer_mod.sliceQsaHistoryOntoCheckpoint;
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;
const testing = std.testing;
const CommitStatus = pc.CommitStatus;
const DflashSnap = pc.DflashSnap;
const HotPrefixCache = pc.HotPrefixCache;
const LookupResult = pc.LookupResult;
const SpecAdopt = pc.SpecAdopt;
const conv_shape_pc = pc.conv_shape_pc;
const mediaTurnTokens = pc.mediaTurnTokens;
const pcArange = pc.pcArange;
const pcArrEqual = pc.pcArrEqual;
const pcBuildHybrid = pc.pcBuildHybrid;
const pcEmptySsm = pc.pcEmptySsm;
const pcFreeHybrid = pc.pcFreeHybrid;
const pcFreeQsaHybrid = pc.pcFreeQsaHybrid;
const pcQsaExpectViewEqual = pc.pcQsaExpectViewEqual;
const pcQsaFeed = pc.pcQsaFeed;
const pcQsaSliceAxis1 = pc.pcQsaSliceAxis1;
const pcRowBytes = pc.pcRowBytes;
const pcSsmVal = pc.pcSsmVal;
const specAdoptPlan = pc.specAdoptPlan;
const testFillCache = pc.testFillCache;
const testWriteCacheLayer = pc.testWriteCacheLayer;

/// Fill a head-shaped cache: one layer at `layer` (the qwen4_exp MTP head's layer is never 0),
/// driven through `Transformer.qwen4MtpAdvance` as `qwen4MtpForward` drives it.
fn testFillHeadCache(cache: *KVCache, s: mlx.mlx_stream, layer: u32, tokens: u32, seq_offset: *usize) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        try testWriteCacheLayer(cache, s, layer, written, step);
        transformer_mod.Transformer.qwen4MtpAdvance(cache, seq_offset, @intCast(step));
        written += step;
    }
}

fn pcHeld(cps: []const SSMCheckpoint) usize {
    var n: usize = 0;
    for (cps) |*cp| n += @intFromBool(!cp.isReleased());
    return n;
}

/// A two-layer QSA hybrid: the layer counts differ from `pcBuildQsaHybrid`'s, which is how a
/// slice between two snaps of this list fails deterministically.
fn pcBuildQsaPair(s: mlx.mlx_stream, rows: c_int) [2]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, 100.0), .ssm_state = mlx.mlx_array_new(), .initialized = true },
    };
}

fn pcFreeSsmSlice(entries: []SSMCacheEntry) void {
    for (entries) |*x| {
        if (x.conv_state.ctx != null) _ = mlx.mlx_array_free(x.conv_state);
        if (x.ssm_state.ctx != null) _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

/// qwen4-shaped hybrid: layer 0 is a QSA full-attention layer (no conv/ssm,
/// `aux_state` = `[1, rows, 8]` indexer key history), layer 1 GDN, layer 2 idle.
fn pcBuildQsaHybrid(s: mlx.mlx_stream, rows: c_int, conv_base: f64) [3]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

/// Two QSA layers, the second holding a SHORTER pooled bank than the first.
fn pcBuildQsaHybridUneven(s: mlx.mlx_stream, rows: c_int, short_blocks: c_int) [3]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    const short_shape = [_]c_int{ 1, short_blocks, 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 2000.0), .qsa_pooled = pcArange(s, &short_shape, 3000.0), .qsa_ratio = 4 },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

fn pcQsaArangeKeys(s: mlx.mlx_stream, rows: c_int, hd: c_int) !mlx.mlx_array {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(rows * hd), 1.0, .float32, s)) catch return error.TestUnexpectedResult;
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    mlx.check(mlx.mlx_reshape(&shaped, flat, &[_]c_int{ 1, rows, hd }, 3, s)) catch return error.TestUnexpectedResult;
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_astype(&out, shaped, .bfloat16, s)) catch return error.TestUnexpectedResult;
    _ = mlx.mlx_array_eval(out);
    return out;
}

fn pcQsaSweep(hc: *HotPrefixCache, toks: []const u32, keys: mlx.mlx_array, lo: usize, hi: usize, s: mlx.mlx_stream) !void {
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    var r = lo;
    while (r <= hi) : (r += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const got = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, toks[0..r], false, 0, null, null) catch |err| {
            log.warn("qsa value sweep lookup at r={d}: {s}\n", .{ r, @errorName(err) });
            return err;
        };
        try testing.expect(got.matched > 0);
        const pos: c_int = @intCast(got.matched);
        var cold = pcEmptySsm();
        defer pcFreeQsaHybrid(&cold);
        cold[0].qsa_ratio = 4;
        try pcQsaFeed(&xfm, &cold[0], keys, pos, s);
        pcQsaExpectViewEqual(s, &target[0], &cold[0], pos) catch {
            log.warn("qsa value sweep mismatch r={d} restored={d} full_match={}\n", .{ r, got.matched, got.full_match });
            return error.TestExpectedEqual;
        };
    }
}

fn pcSsmDiffField(s: mlx.mlx_stream, a: []const SSMCacheEntry, b: []const SSMCacheEntry) ![]const u8 {
    if (a.len != b.len) return "layer_count";
    for (a, b, 0..) |*la, *lb, i| {
        _ = i;
        if (la.initialized != lb.initialized) return "initialized";
        if (la.ple_prev_valid != lb.ple_prev_valid) return "ple_prev_valid";
        if (!std.mem.eql(u32, &la.ple_prev, &lb.ple_prev)) return "ple_prev";
        if (la.qsa_ratio != lb.qsa_ratio) return "qsa_ratio";
        if (la.qsa_hist_rows != lb.qsa_hist_rows) return "qsa_hist_rows";
        if (la.qsa_key_rows != lb.qsa_key_rows) return "qsa_key_rows";
        if (la.qsa_pooled_blocks != lb.qsa_pooled_blocks) return "qsa_pooled_blocks";
        pcArrEqual(s, la.conv_state, lb.conv_state) catch return "conv_state";
        pcArrEqual(s, la.ssm_state, lb.ssm_state) catch return "ssm_state";
        pcArrEqual(s, la.qsa_pooled, lb.qsa_pooled) catch return "qsa_pooled";
        pcArrEqual(s, la.aux_state, lb.aux_state) catch return "aux_state";
    }
    return "";
}

fn pcCaptureFed(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    xfm: *transformer_mod.Transformer,
    keys: mlx.mlx_array,
    pos: c_int,
    conv_base: f64,
    ple0: u32,
) !SSMCheckpoint {
    var live: [3]SSMCacheEntry = .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = mlx.mlx_array_new(), .initialized = true, .ple_prev = .{ ple0, 0, 0, 0, 0, 0, 0, 0 }, .ple_prev_valid = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer pcFreeQsaHybrid(&live);
    try pcQsaFeed(xfm, &live[0], keys, pos, s);
    return transformer_mod.captureSsmCheckpoint(allocator, &live, @intCast(pos), s);
}

test "Qwen SSD-only: a prefill holds at most two checkpoint states, lists every position, and any of them restores from SSD exactly" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tokens: [1500]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 1500);
    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-qwen-wt", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();

    var cps = std.ArrayList(SSMCheckpoint).empty;
    errdefer {
        for (cps.items) |*cp| cp.deinit(testing.allocator);
        cps.deinit(testing.allocator);
    }
    var peak: usize = 0;
    // The prefill loop: a checkpoint after each chunk, then the write-through.
    for (0..7) |i| {
        const pos = 512 + 128 * i;
        try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], pos, s));
        peak = @max(peak, pcHeld(cps.items));
        hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..pos], false, .{}, cps.items, std.math.maxInt(u64), s);
        try testing.expectEqual(@as(usize, 1), pcHeld(cps.items));
    }
    // The prompt-end snapshot is taken after the loop, past the hook.
    try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[7], 1400, s));
    try testing.expectEqual(@as(usize, 2), pcHeld(cps.items));
    try testing.expect(peak <= 2);
    try testing.expectEqual(@as(usize, 8), cps.items.len);

    // The commit lists every position and the SSD record writes only the states still held.
    const owned = try cps.toOwnedSlice(testing.allocator);
    try testing.expectEqual(CommitStatus{ .disk_only = 1500 }, try hc.commitWithSsm(&cache, &tokens, false, owned, null, null));
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqualSlices(u32, &.{ 512, 640, 768, 896, 1024, 1152, 1280, 1400 }, hc.disk.?.entries.items[0].ssm_positions);

    // Released and held alike: each position restores its own state from the file.
    for ([_]struct { pos: u32, conv: f32, ssm: f32 }{
        .{ .pos = 512, .conv = 100, .ssm = 500 },
        .{ .pos = 896, .conv = 400, .ssm = 2000 },
        .{ .pos = 1280, .conv = 700, .ssm = 3500 },
        .{ .pos = 1400, .conv = 800, .ssm = 4000 },
    }) |w| {
        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst = pcEmptySsm();
        defer pcFreeHybrid(&dst);
        try testing.expectEqual(w.pos, try hc.disk.?.restoreIntoHybrid(&cache2, &dst, 0, w.pos, s));
        try testing.expectEqual(w.conv, pcSsmVal(dst[0].conv_state, 0, s));
        try testing.expectEqual(w.ssm, pcSsmVal(dst[0].ssm_state, 0, s));
    }
}

test "HotPrefixCache: a trim that cannot carry the QSA history over declines instead of committing" {
    // The bank lives only on the latest snap; a trim that drops it without slicing it onto the
    // last kept snap leaves an entry whose every restore is a silent miss.
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var wide = pcBuildQsaHybrid(s, 600, 100.0);
    defer pcFreeQsaHybrid(&wide);
    var narrow = pcBuildQsaPair(s, 600);
    defer pcFreeSsmSlice(&narrow);

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &wide, 300, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &narrow, 600, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &narrow, s);
    try testing.expect(checkpointHasQsaPooled(&cps[1]));

    var probe = try c1.snapshot();
    defer probe.deinit();
    const budget = pcRowBytes(&probe) * 300 + transformer_mod.ssmCheckpointBytes(&cps[0]);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    const status = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
    try testing.expect(status == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "HotPrefixCache: restore dump writes named tensors and is a no-op when unset" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 8);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);

    restore_dump.dump_restore_override = null;
    try testing.expectEqual(@as(?u64, null), restore_dump.dumpRestoreIfEnabled(&cache, &live, s, .{
        .kind = "restore",
        .pos = 8,
    }));

    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const dir = buf[0..root_len];
    restore_dump.dump_restore_override = dir;
    defer restore_dump.dump_restore_override = null;
    const n = restore_dump.dumpRestoreIfEnabled(&cache, &live, s, .{
        .kind = "restore",
        .pos = 8,
        .cp = 8,
        .bank_from = 8,
        .source = "own",
        .entry_idx = 1,
        .entry_count = 1,
    }) orelse return error.TestExpectedEqual;

    const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/restore-{d}-pos8.safetensors", .{ dir, n }, 0);
    defer testing.allocator.free(st_path);
    const json_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/restore-{d}-pos8.json", .{ dir, n }, 0);
    defer testing.allocator.free(json_path);
    {
        const sf = std.c.fopen(st_path, "r") orelse return error.TestExpectedEqual;
        _ = std.c.fclose(sf);
        const jf = std.c.fopen(json_path, "r") orelse return error.TestExpectedEqual;
        _ = std.c.fclose(jf);
    }
    var tmap = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tmap);
    var mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    try mlx.check(mlx.mlx_load_safetensors(&tmap, &mmap, st_path, s));
    const names = [_][:0]const u8{
        "pos",
        "layers.0.aux_state",
        "layers.0.qsa_pooled",
        "layers.0.ple_prev",
        "layers.0.qsa_rows",
        "layers.0.qsa_key_rows",
        "layers.0.ring_start",
        "layers.1.conv_state",
        "kv.0.k_tail",
        "kv.0.v_tail",
    };
    for (names) |name| {
        var got = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_map_string_to_array_get(&got, tmap, name));
        try testing.expect(got.ctx != null);
    }
}

test "HotPrefixCache: a QSA arch restore with no indexer history is a miss, never a poisoned entry" {
    // A snap without QSA history (a cancel handoff whose attach failed, an
    // old on-disk entry) used to restore aux-less; the next prefill then died
    // in qsaMaskFromQk with QsaHistoryGap on EVERY turn on that prefix. A
    // QSA arch treats "no history after restore" like "no checkpoint".
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    for ([_]bool{ false, true }) |with_history| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        defer pcFreeQsaHybrid(&live);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (with_history) try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        if (with_history) {
            try testing.expectEqual(@as(usize, 4), r.matched);
            try testing.expect(target[0].aux_state.ctx == null);
            try testing.expectEqual(@as(c_int, 4), target[0].qsa_hist_rows);
        } else {
            try testing.expectEqual(@as(usize, 0), r.matched);
            try testing.expectEqual(@as(usize, 0), moe_off);
            try testing.expect(target[0].aux_state.ctx == null);
        }
    }
}

test "HotPrefixCache: a history tensor shorter than the checkpoint is a miss, not a short hit" {
    const s = mlx.gpuStream();
    var tokens: [70]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    var lookup_latest: [70]u32 = tokens;
    lookup_latest[64] = 999;
    var lookup_interior: [70]u32 = tokens;
    lookup_interior[16] = 999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 16, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 64, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    try testing.expect(cps[1].layers[0].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    for ([_][]const u32{ &lookup_latest, &lookup_interior }) |lookup| {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        try testing.expectError(error.QsaHistoryGap, hc.lookupAndRestore(&target_cache, &moe_off, &target, s, lookup, false, 0, null, null));
        try testing.expectEqual(@as(usize, 0), target_cache.step);
        try testing.expect(target[0].aux_state.ctx == null);
    }
}

test "HotPrefixCache: a lookup that fails after binding the KV leaves nothing behind" {
    // The sole caller reads any lookup error as "no match" and cold-prefills the WHOLE prompt:
    // a half-restored cache would hold the prefix twice, at the wrong RoPE positions.
    const s = mlx.gpuStream();
    var tokens: [16]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    var lookup: [16]u32 = tokens;
    lookup[10] = 999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    // The coverage pre-check maxes the pooled bank over layers; the apply is per layer.
    var live = pcBuildQsaHybridUneven(s, 16, 1);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 12, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 7;
    try testing.expectError(error.QsaHistoryGap, hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null));
    try testing.expectEqual(@as(usize, 0), target_cache.step);
    try testing.expectEqual(@as(usize, 0), moe_off);
    for (&target) |*e| {
        try testing.expect(!e.initialized);
        try testing.expect(e.aux_state.ctx == null);
        try testing.expect(e.qsa_pooled.ctx == null);
    }
}

test "HotPrefixCache: prefix-extend keeps ONE QSA history across turns" {
    // The replace path inherits the old entry's checkpoints. Its latest snap
    // carried the full history and the new latest gets another one: one copy
    // per committed turn, the leak the stride fix closed by another door.
    const s = mlx.gpuStream();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, t1.len);
    var l1 = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&l1);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps1[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l1, 4, s);
    try transformer_mod.attachQsaHistoryToLatest(cps1, &l1, s);
    _ = try hc.commitWithState(&c1, &t1, false, 0, cps1, null, null);

    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, t2.len);
    var l2 = pcBuildQsaHybrid(s, 14, 200.0);
    defer pcFreeQsaHybrid(&l2);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps2[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l2, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps2, &l2, s);
    _ = try hc.commitWithState(&c2, &t2, false, 0, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const merged = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 2), merged.len);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&merged[0]));
    try testing.expect(transformer_mod.checkpointHasQsaHistory(&merged[1]));
    try testing.expect(merged[1].layers[0].aux_state.ctx == null);
    try testing.expect(merged[1].layers[0].qsa_pooled.ctx != null);
}

test "HotPrefixCache: a handed-off QSA history commits, restores and bills exactly like the prefill-end copy" {
    // The commit handoff gives the newest snap a view of the slot's live history; the entry must
    // be indistinguishable from the copy arm and outlive the slot.
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    var restored_val: [2]f32 = .{ -1.0, -2.0 };
    var billed: [2]u64 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |handoff, arm| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (handoff) {
            try transformer_mod.handoffQsaHistoryToLatest(cps, &live, s);
        } else {
            try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        }
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
        pcFreeQsaHybrid(&live);
        billed[arm] = hc.entries.items[0].ssm_bytes;

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, 4), r.matched);
        try testing.expect(target[0].aux_state.ctx == null);
        try testing.expectEqual(@as(c_int, 4), target[0].qsa_hist_rows);
        try testing.expectEqual(@as(c_int, 1), target[0].qsa_pooled_blocks);
        restored_val[arm] = 0;
    }
    try testing.expectEqual(restored_val[0], restored_val[1]);
    try testing.expect(billed[0] > 0);
    try testing.expectEqual(billed[0], billed[1]);
}

test "HotPrefixCache: inheriting a 1-token-tail commit cannot restore a bank-less leftover" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [19]u32 = undefined;
    @memcpy(heir_toks[0..17], donor_toks[0..17]);
    heir_toks[17] = 900;
    heir_toks[18] = 901;

    var live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    donor_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var lookup = heir_toks;
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null);
    const pos: c_int = @intCast(r.matched);
    const ratio: c_int = if (target[0].qsa_ratio > 0) target[0].qsa_ratio else 4;
    const blocks: c_int = if (target[0].qsa_pooled.ctx != null) mlx.getShape(target[0].qsa_pooled)[1] else 0;
    const held: c_int = if (target[0].aux_state.ctx != null) mlx.getShape(target[0].aux_state)[1] else 0;
    const backed = target[0].qsa_hist_rows == pos and blocks * ratio + held >= pos;
    try testing.expect(r.matched == 0 or backed);
}

test "HotPrefixCache: a 1-token-tail heir restores a bank that sits at prompt_end-1" {
    const s = mlx.gpuStream();
    var prefix: [18]u32 = undefined;
    for (&prefix, 0..) |*t, i| t.* = @intCast(i + 1);
    const donor_toks = prefix ++ [_]u32{ 90, 91 };
    const heir_toks = prefix ++ [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207 };
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 18, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_live = pcBuildQsaHybrid(s, 21, 100.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 21, s);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var lookup = heir_toks ++ [_]u32{ 300, 301 };
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null);
    try testing.expectEqual(@as(usize, 18), r.matched);
    try testing.expectEqual(@as(c_int, 18), target[0].qsa_hist_rows);
    try testing.expect(target[0].qsa_pooled.ctx != null);
    const ratio: c_int = if (target[0].qsa_ratio > 0) target[0].qsa_ratio else 4;
    const blocks: c_int = mlx.getShape(target[0].qsa_pooled)[1];
    const held: c_int = if (target[0].aux_state.ctx != null) mlx.getShape(target[0].aux_state)[1] else 0;
    try testing.expect(blocks * ratio + held >= 18);
}

test "HotPrefixCache: a leftover-only checkpoint list is a lookup miss" {
    const s = mlx.gpuStream();
    var toks: [19]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 17, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null) catch |err| blk: {
        try testing.expectEqual(error.QsaHistoryGap, err);
        break :blk LookupResult{ .matched = 0, .full_match = false };
    };
    try testing.expectEqual(@as(usize, 0), r.matched);
}

test "HotPrefixCache: v3 heir full-reuse clamp satisfies QSA at the restored position, all four residues" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [28]u32 = undefined;
    @memcpy(heir_toks[0..21], donor_toks[0..21]);
    for (heir_toks[21..], 0..) |*t, i| t.* = @intCast(900 + i);

    var donor_live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&donor_live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 16, s);
    donor_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &donor_live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_live = pcBuildQsaHybrid(s, 21, 200.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 20, s);
    try transformer_mod.attachQsaHistoryToLatest(heir_own, &heir_live, s);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var n: usize = 17;
    var hits: usize = 0;
    while (n <= 20) : (n += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, heir_toks[0..n], false, 0, null, null) catch |err| blk: {
            try testing.expectEqual(error.QsaHistoryGap, err);
            break :blk LookupResult{ .matched = 0, .full_match = false };
        };
        try testing.expect(!r.full_match);
        if (r.matched > 0) {
            hits += 1;
            try testing.expect(transformer_mod.qsaRestoreSatisfiesForward(&target, r.matched));
            try testing.expectEqual(r.matched, moe_off);
        }
    }
    try testing.expect(hits == 4);
}

test "HotPrefixCache: restored QSA history values match a cold feed at every position" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const P: usize = 80;
    const keys = try pcQsaArangeKeys(s, @intCast(P), hd);
    defer _ = mlx.mlx_array_free(keys);
    var toks: [P]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;

    // Plain entry: checkpoints at every r in [P-40, P], bank on latest.
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        const n_cps: usize = 41;
        const cps = try testing.allocator.alloc(SSMCheckpoint, n_cps);
        var i: usize = 0;
        while (i < n_cps) : (i += 1) {
            const pos: c_int = @intCast(P - 40 + i);
            const chunk = try pcQsaSliceAxis1(s, keys, 0, pos);
            defer _ = mlx.mlx_array_free(chunk);
            transformer_mod.ssmFreeQsaState(&live[0]);
            live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 };
            try transformer_mod.qsaTestAppendPool(&xfm, &live[0], chunk, 0, s);
            cps[i] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, @intCast(pos), s);
        }
        try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, toks.len);
        _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);
        try pcQsaSweep(&hc, &toks, keys, P - 39, P, s);
    }

    // Inherited entry: donor bank sits above shared; heir is a short-tail commit.
    {
        const shared: usize = 48;
        const donor_len: usize = 64;
        const donor_keys = try pcQsaArangeKeys(s, @intCast(donor_len), hd);
        defer _ = mlx.mlx_array_free(donor_keys);
        var donor_toks: [64]u32 = undefined;
        for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
        var heir_toks: [56]u32 = undefined;
        @memcpy(heir_toks[0..shared], donor_toks[0..shared]);
        for (heir_toks[shared..], 0..) |*t, i| t.* = @intCast(10_000 + i);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;

        var donor_live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&donor_live);
        const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 5);
        for ([_]c_int{ 16, 32, 48, 56, 64 }, 0..) |pos, i| {
            const chunk = try pcQsaSliceAxis1(s, donor_keys, 0, pos);
            defer _ = mlx.mlx_array_free(chunk);
            transformer_mod.ssmFreeQsaState(&donor_live[0]);
            donor_live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 };
            try transformer_mod.qsaTestAppendPool(&xfm, &donor_live[0], chunk, 0, s);
            donor_cps[i] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, @intCast(pos), s);
        }
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &donor_live, s);
        var donor_cache = try KVCache.init(testing.allocator, 3);
        defer donor_cache.deinit();
        try testFillCache(&donor_cache, s, 3, donor_toks.len);
        _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

        var heir_live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&heir_live);
        {
            const chunk = try pcQsaSliceAxis1(s, donor_keys, 0, @intCast(shared));
            defer _ = mlx.mlx_array_free(chunk);
            try transformer_mod.qsaTestAppendPool(&xfm, &heir_live[0], chunk, 0, s);
        }
        const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
        heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, shared, s);
        var heir_cache = try KVCache.init(testing.allocator, 3);
        defer heir_cache.deinit();
        try testFillCache(&heir_cache, s, 3, heir_toks.len);
        _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

        try pcQsaSweep(&hc, heir_toks[0..shared], donor_keys, 17, shared, s);
    }
}

test "HotPrefixCache: QSA trim checkpoint bill matches materialized history" {
    const s = mlx.gpuStream();
    var live = pcBuildQsaHybrid(s, 1027, 100.0);
    defer pcFreeQsaHybrid(&live);
    var cps: [2]SSMCheckpoint = undefined;
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 1027, s);
    defer cps[1].deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(cps[1..], &live, s);
    for ([_]usize{ 256, 257, 258, 259, 512, 1024, 1027 }) |pos| {
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, pos, s);
        defer cps[0].deinit(testing.allocator);
        const predicted = HotPrefixCache.trimmedCheckpointBytes(&cps, 0);
        try sliceQsaHistoryOntoCheckpoint(&cps[0], &cps[1], pos, s);
        try testing.expectEqual(predicted, ssmCheckpointBytes(&cps[0]));
        // Pricing a checkpoint that already carries its bank must not double-bill it.
        try testing.expectEqual(predicted, HotPrefixCache.trimmedCheckpointBytes(&cps, 0));
    }
}

test "HotPrefixCache: oversized QSA trim bills the transferred bank and stays reusable" {
    const s = mlx.gpuStream();
    for ([_]transformer_mod.ThinPolicy{ .min_span_recency, .min_span }) |policy| {
        for ([_]bool{ false, true }) |replace| {
            var toks: [1024]u32 = undefined;
            for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
            var live = pcBuildQsaHybrid(s, 1024, 100.0);
            defer pcFreeQsaHybrid(&live);
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try testFillCache(&cache, s, 3, toks.len);
            var snap = try cache.snapshot();
            defer snap.deinit();
            const cps = try testing.allocator.alloc(SSMCheckpoint, 4);
            for (cps, 0..) |*cp, i| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, (i + 1) * 256, s);
            try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
            // 768 KV rows and all small checkpoints fit, but the 768-row bank does not.
            const budget = HotPrefixCache.snapshotRowBytes(&snap) * 768 + ssmCheckpointBytes(&cps[0]) * 4 + 1024;
            var hc = HotPrefixCache.initWithMem(testing.allocator, 1, budget);
            defer hc.deinit();
            hc.cp_thin = policy;
            hc.qsa_history_required = true;
            if (replace) {
                var seed = try KVCache.init(testing.allocator, 3);
                defer seed.deinit();
                try testFillCache(&seed, s, 3, 256);
                const seed_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
                seed_cps[0] = try transformer_mod.shareSsmCheckpoint(testing.allocator, &cps[0]);
                try sliceQsaHistoryOntoCheckpoint(&seed_cps[0], &cps[3], 256, s);
                _ = try hc.commitWithState(&seed, toks[0..256], false, 0, seed_cps, null, null);
            }
            const chosen = hc.trimLenForBudget(budget, toks.len, HotPrefixCache.snapshotRowBytes(&snap), cps);
            _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);
            try testing.expectEqual(@as(?usize, 512), chosen);
            try testing.expectEqual(@as(usize, 1), hc.entryCount());
            try testing.expect(hc.current_kv_bytes <= budget);
            try testing.expectEqual(@as(usize, 512), hc.entries.items[0].tokens.len);
            var restored = pcEmptySsm();
            defer pcFreeQsaHybrid(&restored);
            var target = try KVCache.init(testing.allocator, 3);
            defer target.deinit();
            var moe_off: usize = 0;
            const hit = try hc.lookupAndRestore(&target, &moe_off, &restored, s, &toks, false, 0, null, null);
            try testing.expectEqual(@as(usize, 512), hit.matched);
        }
    }
}

test "HotPrefixCache: shed rescues an interior QSA bank" {
    const s = mlx.gpuStream();
    var toks: [30]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&live);
    var probe = try KVCache.init(testing.allocator, 3);
    defer probe.deinit();
    try testFillCache(&probe, s, 3, toks.len);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    hc.ssm_checkpoint_max = 8;
    const cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    try transformer_mod.sliceQsaHistoryOntoCheckpoint(&cps[1], &cps[2], cps[1].pos, s);
    for (cps[2].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&probe, &toks, false, 0, cps, null, null);
    const before = hc.current_kv_bytes;
    const listed = hc.entries.items[0].ssm_checkpoints.?;
    hc.max_kv_bytes = before - ssmCheckpointBytes(&listed[1]) + 1;
    hc.shedCheckpointsToFit();
    try testing.expect(hc.entryCount() == 1);
    const kept = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expect(kept.len >= 1);
    try testing.expect(transformer_mod.checkpointListHasQsaPooled(kept));
}

test "HotPrefixCache: dropLastRestored removes the restored entry so the next lookup is cold" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const first = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, null);
    try testing.expect(first.matched > 0);
    try testing.expect(hc.dropLastRestored());
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const second = try hc.lookupAndRestore(&cache2, &moe2, &target2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), second.matched);
}

test "HotPrefixCache: QSA self-heal drops the slot model cache and leaves a sibling cache" {
    const s = mlx.gpuStream();
    const tokens_a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const tokens_b = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

    var last_loaded = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer last_loaded.deinit();
    last_loaded.qsa_history_required = true;
    var cache_a = try KVCache.init(testing.allocator, 3);
    defer cache_a.deinit();
    try testFillCache(&cache_a, s, 3, tokens_a.len);
    var live_a = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live_a);
    const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps_a[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live_a, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps_a, &live_a, s);
    _ = try last_loaded.commitWithState(&cache_a, &tokens_a, false, 0, cps_a, null, null);

    var slot_hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer slot_hc.deinit();
    slot_hc.qsa_history_required = true;
    var cache_b = try KVCache.init(testing.allocator, 3);
    defer cache_b.deinit();
    try testFillCache(&cache_b, s, 3, tokens_b.len);
    var live_b = pcBuildQsaHybrid(s, 10, 200.0);
    defer pcFreeQsaHybrid(&live_b);
    const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps_b[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live_b, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps_b, &live_b, s);
    _ = try slot_hc.commitWithState(&cache_b, &tokens_b, false, 0, cps_b, null, null);

    var tgt_a = try KVCache.init(testing.allocator, 3);
    defer tgt_a.deinit();
    var ssm_a = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm_a);
    var moe_a: usize = 0;
    const ra = try last_loaded.lookupAndRestore(&tgt_a, &moe_a, &ssm_a, s, &tokens_a, false, 0, null, null);
    try testing.expect(ra.matched > 0);

    var tgt_b = try KVCache.init(testing.allocator, 3);
    defer tgt_b.deinit();
    var ssm_b = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm_b);
    var moe_b: usize = 0;
    const rb = try slot_hc.lookupAndRestore(&tgt_b, &moe_b, &ssm_b, s, &tokens_b, false, 0, null, null);
    try testing.expect(rb.matched > 0);

    try testing.expect(HotPrefixCache.dropQsaGapEntry(&slot_hc));
    try testing.expectEqual(@as(usize, 0), slot_hc.entryCount());
    try testing.expectEqual(@as(usize, 1), last_loaded.entryCount());
}

test "HotPrefixCache: inherited checkpoints never exceed the shared prefix" {
    const s = mlx.gpuStream();
    var prefix: [18]u32 = undefined;
    for (&prefix, 0..) |*t, i| t.* = @intCast(i + 1);
    const donor_toks = prefix ++ [_]u32{ 90, 91 };
    const prompt_len: usize = 19;
    const heir_toks = prefix ++ [_]u32{200} ++ [_]u32{ 300, 301, 302, 303, 304, 305, 306, 307 };
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 18, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse continue;
        for (cps) |*cp| try testing.expect(cp.pos <= prompt_len - 1);
    }
    try testing.expect(found);
}

test "HotPrefixCache: inherit does not clone checkpoints past donor.shared" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const P: usize = 20;
    const hd: c_int = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, 40, hd);
    defer _ = mlx.mlx_array_free(keys);
    var donor_toks: [40]u32 = undefined;
    var heir_toks: [40]u32 = undefined;
    for (0..P) |i| {
        donor_toks[i] = @intCast(i + 1);
        heir_toks[i] = @intCast(i + 1);
    }
    for (P..40) |i| {
        donor_toks[i] = @intCast(1000 + i);
        heir_toks[i] = @intCast(2000 + i);
    }
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 4);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 8, 100.0, 8);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 16, 160.0, 16);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 30, 300.0, 30);
    donor_cps[3] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 40, 400.0, 40);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, 40, s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len or e.tokens[P] != heir_toks[P]) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse return error.TestExpectedEqual;
        for (cps) |*cp| try testing.expect(cp.pos <= P);
    }
    try testing.expect(found);

    var rpos: usize = P - 4;
    while (rpos <= P) : (rpos += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const got = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, heir_toks[0..rpos], false, 0, null, null);
        try testing.expect(got.matched > 0);
        try testing.expect(got.matched <= P);
        var cold = pcEmptySsm();
        defer pcFreeQsaHybrid(&cold);
        cold[0].qsa_ratio = 4;
        try pcQsaFeed(&xfm, &cold[0], keys, @intCast(got.matched), s);
        pcQsaExpectViewEqual(s, &target[0], &cold[0], @intCast(got.matched)) catch {
            log.warn("inherit divergence restore mismatch pos={d} restored={d}\n", .{ rpos, got.matched });
            return error.TestExpectedEqual;
        };
    }
}

test "HotPrefixCache: inherit of a greedy continuation stops at the prompt" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const P: usize = 48;
    const prompt_len: usize = P - 1;
    const hd: c_int = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, 80, hd);
    defer _ = mlx.mlx_array_free(keys);

    var donor_toks: [64]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [59]u32 = undefined;
    @memcpy(&heir_toks, donor_toks[0..59]);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 19, 190.0, 19);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(P), 480.0, 48);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 64, 640.0, 64);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, 64, s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithMediaState(&heir_cache, &heir_toks, false, 0, 0, null, null, null, null, prompt_len);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse return error.TestExpectedEqual;
        for (cps) |*cp| try testing.expect(cp.pos <= prompt_len);
    }
    try testing.expect(found);
}

test "HotPrefixCache: sliced qsa bank apply at backoff matches the full bank at every residue" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const B: usize = 19;
    const H: usize = 80;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, @intCast(H), hd);
    defer _ = mlx.mlx_array_free(keys);

    var live: [3]SSMCacheEntry = .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
    };
    defer pcFreeQsaHybrid(&live);
    try pcQsaFeed(&xfm, &live[0], keys, @intCast(B), s);
    var cps = [_]SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, B, s),
        undefined,
    };
    transformer_mod.ssmFreeQsaState(&live[0]);
    live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    try pcQsaFeed(&xfm, &live[0], keys, @intCast(H), s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, H, s);
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &live, s);
    if (live[0].aux_state.ctx != null) {
        if (cps[1].layers[0].aux_state.ctx != null) _ = mlx.mlx_array_free(cps[1].layers[0].aux_state);
        cps[1].layers[0].aux_state = try transformer_mod.materializedOwnedCopy(s, live[0].aux_state);
        _ = mlx.mlx_array_eval(cps[1].layers[0].aux_state);
    }

    var cold = pcEmptySsm();
    defer pcFreeQsaHybrid(&cold);
    cold[0].qsa_ratio = ratio;
    try pcQsaFeed(&xfm, &cold[0], keys, @intCast(B), s);

    var from_full = pcEmptySsm();
    defer pcFreeQsaHybrid(&from_full);
    try transformer_mod.restoreSsmCheckpoint(&from_full, &cps[0]);
    transformer_mod.applyQsaHistoryAt(&from_full, &cps[1], B, s, false) catch |err| {
        std.debug.print("sliced-apply residue fail K=full field=apply {s}\n", .{@errorName(err)});
        return err;
    };
    pcQsaExpectViewEqual(s, &from_full[0], &cold[0], @intCast(B)) catch {
        std.debug.print("sliced-apply residue fail K=full field=cold_view hist={d}/{d} blocks={d}\n", .{
            from_full[0].qsa_hist_rows,
            from_full[0].qsa_key_rows,
            from_full[0].qsa_pooled_blocks,
        });
        return error.TestExpectedEqual;
    };

    var K: usize = B + 1;
    while (K < H) : (K += 1) {
        var sliced = try transformer_mod.shareSsmCheckpoint(testing.allocator, &cps[0]);
        defer sliced.deinit(testing.allocator);
        sliced.pos = K;
        try transformer_mod.sliceQsaHistoryOntoCheckpoint(&sliced, &cps[1], K, s);

        var from_sliced = pcEmptySsm();
        defer pcFreeQsaHybrid(&from_sliced);
        try transformer_mod.restoreSsmCheckpoint(&from_sliced, &cps[0]);
        transformer_mod.applyQsaHistoryAt(&from_sliced, &sliced, B, s, false) catch |err| {
            std.debug.print("sliced-apply residue fail K={d} field=apply {s}\n", .{ K, @errorName(err) });
            return err;
        };

        const field = try pcSsmDiffField(s, &from_full, &from_sliced);
        if (field.len != 0) {
            std.debug.print("sliced-apply residue fail K={d} field={s} full_hist={d}/{d} sliced_hist={d}/{d} full_blocks={d} sliced_blocks={d}\n", .{
                K,
                field,
                from_full[0].qsa_hist_rows,
                from_full[0].qsa_key_rows,
                from_sliced[0].qsa_hist_rows,
                from_sliced[0].qsa_key_rows,
                from_full[0].qsa_pooled_blocks,
                from_sliced[0].qsa_pooled_blocks,
            });
            return error.TestExpectedEqual;
        }
        pcQsaExpectViewEqual(s, &from_sliced[0], &cold[0], @intCast(B)) catch {
            std.debug.print("sliced-apply residue fail K={d} field=cold_view\n", .{K});
            return error.TestExpectedEqual;
        };
    }
}

test "HotPrefixCache: heir restore at backoff matches the donor bit for bit" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const P: usize = 48;
    const backoff: usize = 19;
    const donor_gen: usize = 16;
    const heir_gen: usize = 12;
    const a2_extra: usize = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;

    const donor_len = P + donor_gen;
    const a2_len = donor_len + a2_extra;
    const heir_len = (P - 1) + heir_gen;
    const keys = try pcQsaArangeKeys(s, @intCast(a2_len), hd);
    defer _ = mlx.mlx_array_free(keys);

    var donor_toks: [P + donor_gen]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var a2_toks: [P + donor_gen + a2_extra]u32 = undefined;
    @memcpy(a2_toks[0..donor_toks.len], &donor_toks);
    for (a2_toks[donor_toks.len..], 0..) |*t, i| t.* = @intCast(50_000 + i);
    var heir_toks: [(P - 1) + heir_gen]u32 = undefined;
    @memcpy(heir_toks[0 .. P - 1], donor_toks[0 .. P - 1]);
    for (heir_toks[P - 1 ..], 0..) |*t, i| t.* = @intCast(90_000 + i);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(backoff), 190.0, 19);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(P), 480.0, 48);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(donor_len), 640.0, 64);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, @intCast(donor_len), s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var a2_cache = try KVCache.init(testing.allocator, 3);
    defer a2_cache.deinit();
    try testFillCache(&a2_cache, s, 3, a2_toks.len);
    const a2_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    a2_own[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(a2_len), 720.0, 72);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, @intCast(a2_len), s);
        try transformer_mod.attachQsaHistoryToLatest(a2_own, &live, s);
    }
    _ = try hc.commitWithState(&a2_cache, &a2_toks, false, 0, a2_own, null, null);

    var donor_restored = pcEmptySsm();
    defer pcFreeQsaHybrid(&donor_restored);
    var donor_kv = try KVCache.init(testing.allocator, 3);
    defer donor_kv.deinit();
    var donor_moe: usize = 0;
    const donor_hit = try hc.lookupAndRestore(&donor_kv, &donor_moe, &donor_restored, s, a2_toks[0..P], false, 0, null, null);
    try testing.expectEqual(backoff, donor_hit.matched);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 44, 440.0, 44);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var heir_restored = pcEmptySsm();
    defer pcFreeQsaHybrid(&heir_restored);
    var heir_kv = try KVCache.init(testing.allocator, 3);
    defer heir_kv.deinit();
    var heir_moe: usize = 0;
    const heir_hit = try hc.lookupAndRestore(&heir_kv, &heir_moe, &heir_restored, s, &heir_toks, false, 0, null, null);
    try testing.expectEqual(backoff, heir_hit.matched);

    const field = try pcSsmDiffField(s, &donor_restored, &heir_restored);
    if (field.len != 0) {
        log.warn("heir vs donor restore differed in {s} (donor_pos={d} heir_pos={d} heir_hist={d}/{d} donor_hist={d}/{d})\n", .{
            field,
            donor_hit.matched,
            heir_hit.matched,
            heir_restored[0].qsa_hist_rows,
            heir_restored[0].qsa_key_rows,
            donor_restored[0].qsa_hist_rows,
            donor_restored[0].qsa_key_rows,
        });
        return error.TestExpectedEqual;
    }
    _ = heir_len;
}

test "HotPrefixCache: a verbatim re-send restores at prompt_len-1" {
    const s = mlx.gpuStream();
    var toks: [19]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, toks.len - 1, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null);
    try testing.expectEqual(toks.len - 1, r.matched);
    try testing.expect(!r.full_match);
    try testing.expect(transformer_mod.qsaRestoreSatisfiesForward(&target, r.matched));
}

test "HotPrefixCache: a hybrid full-prefix extend checks out under SSD-first" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
    var toks: [10]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var prompt: [12]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var slot = try KVCache.init(testing.allocator, 3);
    defer slot.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestoreForSlot(&slot, &moe_off, &target, s, &prompt, false, 0, null, null, null, 0xA11CE);
    try testing.expect(r.matched > 0);
    try testing.expect(r.checked_out);
    try testing.expectEqual(@as(?usize, 0xA11CE), hc.entries.items[0].checked_out_by);
}

test "HotPrefixCache: walk-down restores the higher covering checkpoint, not a lower one" {
    const s = mlx.gpuStream();
    var toks: [20]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 16, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 16, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, 16), r.matched);
    try testing.expect(target[0].qsa_pooled.ctx != null);
}

test "HotPrefixCache: skip set on the cache without a lookup does not poison the next request" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const skipped = try hc.lookupAndRestoreWithMedia(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, &.{}, null, null, null, true);
    try testing.expectEqual(@as(usize, 0), skipped.matched);
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    const leaked = try hc.lookupAndRestoreWithMedia(&cache2, &moe2, &target2, s, &tokens, false, 0, null, &.{}, null, null, null, false);
    try testing.expect(leaked.matched > 0);
}

test "HotPrefixCache: skip_prefix_cache lookup is cold even when a covering entry exists" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const skipped = try hc.lookupAndRestoreWithMedia(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, &.{}, null, null, null, true);
    try testing.expectEqual(@as(usize, 0), skipped.matched);

    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const second = try hc.lookupAndRestore(&cache2, &moe2, &target2, s, &tokens, false, 0, null, null);
    try testing.expect(second.matched > 0);
}

test "HotPrefixCache: a skipped lookup of a covering prompt matches a cold cache" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var skipped_cache = try KVCache.init(testing.allocator, 3);
    defer skipped_cache.deinit();
    var skipped_ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&skipped_ssm);
    var skip_off: usize = 7;
    const skipped = try hc.lookupAndRestoreWithMedia(&skipped_cache, &skip_off, &skipped_ssm, s, &tokens, false, 0, null, &.{}, null, null, null, true);

    var cold = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer cold.deinit();
    cold.qsa_history_required = true;
    var cold_cache = try KVCache.init(testing.allocator, 3);
    defer cold_cache.deinit();
    var cold_ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&cold_ssm);
    var cold_off: usize = 7;
    const cold_r = try cold.lookupAndRestore(&cold_cache, &cold_off, &cold_ssm, s, &tokens, false, 0, null, null);
    try testing.expectEqual(cold_r.matched, skipped.matched);
    try testing.expectEqual(cold_off, skip_off);
    try testing.expectEqual(@as(usize, 0), skipped.matched);
}

test "HotPrefixCache: inherit-branch QSA invariant drop unbills the dropped SSM bytes" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [19]u32 = undefined;
    @memcpy(heir_toks[0..17], donor_toks[0..17]);
    heir_toks[17] = 900;
    heir_toks[18] = 901;

    var live = pcBuildQsaHybrid(s, 17, 100.0);
    defer pcFreeQsaHybrid(&live);
    var ctrl = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer ctrl.deinit();
    var ctrl_cache = try KVCache.init(testing.allocator, 3);
    defer ctrl_cache.deinit();
    try testFillCache(&ctrl_cache, s, 3, heir_toks.len);
    _ = try ctrl.commitWithState(&ctrl_cache, &heir_toks, false, 0, null, null, null);
    const snap_only = ctrl.entries.items[0].kv_bytes;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    for (donor_cps[0].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    hc.qsa_history_required = true;
    var heir_live = pcBuildQsaHybrid(s, 19, 100.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 18, s);
    for (heir_own[0].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        try testing.expect(e.ssm_checkpoints == null);
        try testing.expectEqual(@as(u64, 0), e.ssm_bytes);
        try testing.expectEqual(snap_only, e.kv_bytes);
    }
    try testing.expect(found);
}

test "HotPrefixCache: shedding the oldest QSA bank with no lower destination drops the list" {
    const s = mlx.gpuStream();
    var toks: [30]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    var live2 = pcBuildQsaHybrid(s, 25, 200.0);
    defer pcFreeQsaHybrid(&live2);
    var probe = try KVCache.init(testing.allocator, 3);
    defer probe.deinit();
    try testFillCache(&probe, s, 3, toks.len);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    hc.ssm_checkpoint_max = 8;
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live2, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(cps[0..1], &live, s);
    for (cps[1].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&probe, &toks, false, 0, cps, null, null);
    hc.max_kv_bytes = 1;
    hc.shedCheckpointsToFit();
    try testing.expect(hc.entryCount() == 1);
    try testing.expect(hc.entries.items[0].ssm_checkpoints == null);
}

test "HotPrefixCache: a QSA-required restore without indexer history is a miss even when forward-satisfy is vacuous" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildHybrid(s, 123.0, 456.0);
    defer pcFreeHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
    hc.qsa_history_required = true;

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), r.matched);
}

test "spec adopt: a qwen4 head target declines a payload with no QSA half; KV-only targets are unaffected" {
    // The qwen4_exp head's KV is meaningless without its index-key history, so the two halves
    // adopt together or not at all. Everything else is KV-only and keeps the old rule.
    const Tag = std.meta.Tag(SpecAdopt);
    const Plan = struct {
        fn tag(p: SpecAdopt) Tag {
            return std.meta.activeTag(p);
        }
        fn len(p: SpecAdopt) usize {
            return switch (p) {
                .kv_only, .head => |w| w,
                else => std.math.maxInt(usize),
            };
        }
    };
    try testing.expectEqual(Tag.kv_only, Plan.tag(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, false, false))); // starts past the reuse
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, false, false))); // ends short of it
    try testing.expectEqual(@as(usize, 31), Plan.len(specAdoptPlan(0, 31, 31, false, false))); // exact

    try testing.expectEqual(Tag.head, Plan.tag(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(Tag.decline_head_no_history, Plan.tag(specAdoptPlan(10, 40, 31, true, false)));
    // A payload the trunk cannot use is skipped before the aux question.
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, true, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, true, false)));
}

test "qwen4 MTP head persist: the head's row count IS its cache step, so a committed history adopts" {
    // `KVCache.update` advances `step` only at layer 0, so `qwen4MtpAdvance` sets the head's;
    // the commit snapshots it and `qwen4MtpAdopt` demands the key history be exactly that long.
    const s = mlx.gpuStream();
    const head_layer: u32 = 3; // stands in for `num_hidden_layers`
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var seq_offset: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 100, &seq_offset);
    try testing.expectEqual(@as(usize, 100), seq_offset);
    try testing.expectEqual(seq_offset, kv.step);

    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    try testing.expectEqual(seq_offset, snap.snapshot.step);

    const Tag = std.meta.Tag(SpecAdopt);
    const plan = specAdoptPlan(snap.base_pos, snap.snapshot.step, seq_offset, true, true);
    try testing.expectEqual(Tag.head, std.meta.activeTag(plan));
    try testing.expectEqual(seq_offset, plan.head);
    try testing.expectEqual(@as(usize, 100), snap.snapshot.step);

    // The shape the bug had: the step `KVCache.update` left at a non-zero layer is not adoptable.
    try testing.expectEqual(Tag.skip, std.meta.activeTag(specAdoptPlan(0, 0, seq_offset, true, true)));
}

test "spec snap bytes: the qwen4 head's QSA half is billed into the entry" {
    const s = mlx.gpuStream();
    // Head-shaped: one layer at the head's own index, never layer 0.
    const head_layer: u32 = 3;
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var head_rows: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 16, &head_rows);
    try testing.expectEqual(head_rows, kv.step);
    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    const kv_only = HotPrefixCache.specSnapBytes(&snap);
    try testing.expect(kv_only > 0);

    var entry: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer transformer_mod.ssmFreeQsaState(&entry);
    defer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const shape = [_]c_int{ 1, 16, 128 };
    entry.aux_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.aux_state, &shape, 3, .bfloat16, s));
    entry.qsa_ratio = 4;
    snap.head_aux = transformer_mod.ssmSnapshot(&entry);
    const with_head = HotPrefixCache.specSnapBytes(&snap);
    try testing.expectEqual(kv_only + 16 * 128 * 2, with_head);
}

test "Qwen SSD-only: a 40k prefill keeps the 8192 grid and the prompt end on SSD, holding two states, and a grid point restores its state" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const len = 40_000;
    var tokens: [len]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, len);
    var srcs: [5][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-qwen-grid", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.cp_thin = .min_span_recency;
    hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();

    var cps = std.ArrayList(SSMCheckpoint).empty;
    errdefer {
        for (cps.items) |*cp| cp.deinit(testing.allocator);
        cps.deinit(testing.allocator);
    }
    const stride = prefill_plan.effectiveSsmCheckpointStride(256, 8192);
    const loop_end = len - 1 - prefill_plan.SSM_SNAPSHOT_BACKOFF;
    var peak: usize = 0;
    var pos: usize = 0;
    while (pos < loop_end) {
        pos = prefill_plan.nextChunkEnd(pos, loop_end, 8192, true, stride, 0, true);
        if (pos % stride == 0) {
            try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[cps.items.len], pos, s));
            peak = @max(peak, pcHeld(cps.items));
        }
        hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..pos], false, .{}, cps.items, std.math.maxInt(u64), s);
    }
    try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[4], loop_end, s));
    try testing.expect(peak <= 2);

    const owned = try cps.toOwnedSlice(testing.allocator);
    _ = try hc.commitWithSsm(&cache, &tokens, false, owned, null, null);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqualSlices(u32, &.{ 8192, 16384, 24576, 32768, loop_end }, hc.disk.?.entries.items[0].ssm_positions);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst = pcEmptySsm();
    defer pcFreeHybrid(&dst);
    try testing.expectEqual(@as(u32, 16384), try hc.disk.?.restoreIntoHybrid(&cache2, &dst, 0, 16384, s));
    try testing.expectEqual(@as(f32, 200), pcSsmVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 1000), pcSsmVal(dst[0].ssm_state, 0, s));
}

test "an image turn on a QSA hybrid persists the indexer history its checkpoint needs" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const tokens = mediaTurnTokens(1024);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.hybrid = true;
        hc.qsa_history_required = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-qsa", 0, 128);
        defer hc.deinit();
        var live = pcBuildQsaHybrid(s, 1024, 100.0);
        defer pcFreeQsaHybrid(&live);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 4);
        for (cps, 0..) |*cp, i| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, (i + 1) * 256, s);
        // The pooled bank lives on the newest checkpoint, which is past the item.
        try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        _ = try hc.commitWithMediaState(&cache, &tokens, false, 0xABCD, 0, 600, cps, null, null, tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        // Whole, with the pooled bank's checkpoint; the foreign-key restore below still needs the
        // indexer history for its OWN checkpoint, which is the point of this test.
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
    }
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.hybrid = true;
    hc2.qsa_history_required = true;
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-qsa", 0, 128);
    defer hc2.deinit();
    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm);
    var moe: usize = 0;
    const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, &ssm, s, &tokens, false, 0x1234, 600, &.{}, null, null, null, false);
    try testing.expectEqual(@as(usize, 512), res.matched);
}
