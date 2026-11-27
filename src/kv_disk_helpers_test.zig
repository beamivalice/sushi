//! Shared test helpers for the SSD prefix cache test suite.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const trunk = @import("trunk.zig");
const write_stats = @import("disk_write_stats.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const safetensors = @import("safetensors.zig");
const disk = @import("kv_disk_cache.zig");

const KVCache = trunk.KVCache;
const DiskTier = disk.DiskTier;
pub const SSMCacheEntry = trunk.SSMCacheEntry;

pub fn fillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32, head_dim: u32, seed: f64, dtype: mlx.mlx_dtype) !void {
    // Drive the cache through its real update path with deterministic
    // arange-derived K/V so restored values are checkable. Dense tests use
    // float32 (every position stays exactly distinguishable); the affine test
    // uses bf16, the production dtype the quant write path expects.
    // A fill is a request whose last row is its own: it binds the cache there, past any earlier request's end.
    cache.bound(@max(cache.end_tokens, cache.step + tokens));
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const count: f64 = @floatFromInt(step * head_dim);
            const base: f64 = seed + @as(f64, @floatFromInt(written * head_dim + li * 1_000_000));
            try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
            var shaped = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(shaped);
            const shape = [_]c_int{ 1, 1, @intCast(step), @intCast(head_dim) };
            try mlx.check(mlx.mlx_reshape(&shaped, flat, &shape, 4, s));
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_astype(&k, shaped, dtype, s));
            // V = -K so a restore-side K/V swap can't false-pass.
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_negative(&v, k, s));
            var view = try cache.update(li, k, v, s, 0);
            view.deinit();
        }
        written += step;
    }
}

pub fn cacheValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream) !f32 {
    return cacheBufValueAt(cache, layer, pos, d, s, false);
}

pub fn cacheBufValueAt(cache: *KVCache, layer: u32, pos: u32, d: u32, s: mlx.mlx_stream, values: bool) !f32 {
    const entry = &cache.entries[layer];
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, @intCast(pos), @intCast(d) };
    const sp = [_]c_int{ 1, 1, @intCast(pos + 1), @intCast(d + 1) };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    const buf = if (values) entry.values.array else entry.keys.array;
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, sliced, .float32, s));
    _ = mlx.mlx_array_eval(f);
    const ptr = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return ptr[0];
}

pub fn tmpRoot(tmp: *std.testing.TmpDir, io: std.Io, buf: []u8) ![]const u8 {
    const n = try tmp.dir.realPath(io, buf);
    return buf[0..n];
}


pub const conv_shape = [_]c_int{ 1, 3, 8 }; // [B, kernel-1, conv_dim]
pub const ssm_shape = [_]c_int{ 1, 2, 4, 4 }; // [B, Hv, Dv, Dk]

pub fn makeArange(s: mlx.mlx_stream, shape: []const c_int, base: f64) mlx.mlx_array {
    var count: f64 = 1;
    for (shape) |d| count *= @floatFromInt(d);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    _ = mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&out, flat, shape.ptr, @intCast(shape.len), s);
    _ = mlx.mlx_array_eval(out);
    return out;
}

pub fn ssmArrVal(arr: mlx.mlx_array, idx: usize, s: mlx.mlx_stream) f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return mlx.mlx_array_data_float32(f).?[idx];
}

/// Three-layer synthetic hybrid SSM state, covering the full null-state
/// matrix: (0) a GatedDeltaNet layer with both conv+ssm, (1) a layer
/// with conv only (null ssm_state), (2) a plain-attention
/// layer in the hybrid (uninitialized, both null). `conv_base`/`ssm_base`
/// make each capture position's values distinguishable, so a restore-side
/// conv/ssm KEY SWAP fails the value checks (the K/V-swap lesson).
pub fn buildHybridEntries(s: mlx.mlx_stream, conv_base: f64, ssm_base: f64) [3]SSMCacheEntry {
    return .{
        .{
            .conv_state = makeArange(s, &conv_shape, conv_base),
            .ssm_state = makeArange(s, &ssm_shape, ssm_base),
            .initialized = true,
        },
        .{
            .conv_state = makeArange(s, &conv_shape, conv_base + 10_000),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = true,
        },
        .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = false,
        },
    };
}

pub fn freeHybridEntries(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        _ = mlx.mlx_array_free(x.conv_state);
        _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}


pub const grow_divs = [_]u32{ 0, 2, 4, 0 };
pub const grow_span: u32 = 128;
/// Bytes a token adds across those rows: (4 + 2) f32 columns over 2 tokens, and again over 4.
pub const grow_row_bytes: u64 = 18;

/// A restore point at `pos` as an append-only store gives it: the rows are the same numbers in every point, so a
/// shorter point's rows are a prefix of a longer one's; the fixed arrays differ per position.
pub fn growCheckpoint(s: mlx.mlx_stream, pos: usize) !trunk.SSMCheckpoint {
    const layers = try testing.allocator.alloc(trunk.SSMCacheEntrySnapshot, 4);
    const r1: c_int = @intCast(pos / 2);
    const r2: c_int = @intCast(pos / 4);
    const base: f64 = @floatFromInt(pos * 100);
    layers[0] = .{ .conv_state = makeArange(s, &[_]c_int{ 8, 4 }, base), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    layers[1] = .{ .conv_state = makeArange(s, &[_]c_int{ r1, 4 }, 0), .ssm_state = makeArange(s, &[_]c_int{ r1, 2 }, 500_000), .initialized = true };
    layers[2] = .{ .conv_state = makeArange(s, &[_]c_int{ r2, 4 }, 0), .ssm_state = makeArange(s, &[_]c_int{ r2, 2 }, 500_000), .initialized = true };
    layers[3] = .{ .conv_state = makeArange(s, &[_]c_int{ 3, 2 }, base + 7), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    return .{ .pos = pos, .layers = layers };
}

pub fn expectSameArray(want: mlx.mlx_array, got: mlx.mlx_array) !void {
    if (want.ctx == null) {
        try testing.expect(got.ctx == null or mlx.mlx_array_size(got) == 0);
        return;
    }
    try testing.expect(got.ctx != null);
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(u8, safetensors.bytesOf(want).?, safetensors.bytesOf(got).?);
}

/// Loads the restore point at `pos` of entry `id` and checks it holds exactly the arrays `growCheckpoint` builds.
pub fn expectGrowPoint(tier: *DiskTier, id: u64, pos: u32, s: mlx.mlx_stream) !void {
    var want = try growCheckpoint(s, pos);
    defer want.deinit(testing.allocator);
    var got = try tier.loadSsmFile(id, pos, 4);
    defer got.deinit(testing.allocator);
    for (want.layers, got.layers) |w, g| {
        try expectSameArray(w.conv_state, g.conv_state);
        try expectSameArray(w.ssm_state, g.ssm_state);
    }
}

pub fn commitGrowPoints(tier: *DiskTier, cache: *const KVCache, tokens: []const u32, positions: []const usize, s: mlx.mlx_stream) !void {
    const cps = try testing.allocator.alloc(trunk.SSMCheckpoint, positions.len);
    defer testing.allocator.free(cps);
    for (cps, positions) |*cp, p| cp.* = try growCheckpoint(s, p);
    defer for (cps) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens, false, cps, s);
}

pub fn growTier(io: std.Io, base: []const u8, fp: []const u8, split: bool, writer: bool) !DiskTier {
    var tier = try DiskTier.init(testing.allocator, io, base, fp, 0, 128);
    errdefer tier.deinit();
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = 160;
    if (split) try tier.setGrownSpec(grow_span, &grow_divs, grow_row_bytes);
    if (writer) tier.enableBackgroundWriter();
    return tier;
}

pub fn growFileSize(tier: *const DiskTier, alloc_path: []const u8, id: u64, name: []const u8) ?u64 {
    const path = std.fmt.allocPrint(testing.allocator, "{s}/e{d}/{s}", .{ alloc_path, id, name }) catch return null;
    defer testing.allocator.free(path);
    return kv_disk_fs.testSize(tier.io, path);
}


pub const ShareToks = struct { a: [600]u32, b: [600]u32 };
pub fn chunkShareTokens(shared: usize) ShareToks {
    var out: ShareToks = undefined;
    for (&out.a, 0..) |*t, i| t.* = @intCast(i + 7);
    for (&out.b, 0..) |*t, i| t.* = if (i < shared) @intCast(i + 7) else @intCast(9000 + i);
    return out;
}


pub fn wroteKind(tier: *const DiskTier, kind: write_stats.Kind) u64 {
    return tier.stats.snapshot().written[@backingInt(kind)];
}

pub fn sumBytes(items: []const u64) u64 {
    var n: u64 = 0;
    for (items) |b| n += b;
    return n;
}

/// The bytes the layer appended to its packs, journal and snapshot, the pages it punched since included; 0 on plain
/// files.
pub fn appendedOf(tier: *DiskTier) u64 {
    const v = tier.fs.vfs;
    var punched: u64 = 0;
    var it = v.packs.valueIterator();
    while (it.next()) |p| punched += p.holes;
    return v.physicalBytes() + punched;
}

