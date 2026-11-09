//! Tensor files on the prefix VFS against the safetensors path the SSD tier writes today.

const std = @import("std");
const mlx = @import("mlx.zig");
const vt = @import("prefix_vfs_tensors.zig");
const prefix_vfs = @import("prefix_vfs.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");

const testing = std.testing;
const ta = testing.allocator;
const tio = testing.io;
const DiskTier = kv_disk_cache.DiskTier;
const Vfs = prefix_vfs.Vfs;
const RowMap = vt.RowMap;

const TSpec = struct { key: []const u8, shape: []const c_int, dtype: mlx.mlx_dtype, rows: ?RowMap = null };
const KindSpec = struct { name: []const u8, kind: prefix_vfs.Kind, tensors: []const TSpec, meta: []const vt.MetaPair = &.{} };

fn itemSize(d: mlx.mlx_dtype) usize {
    return switch (d) {
        .bool_, .uint8, .int8 => 1,
        .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        else => unreachable,
    };
}

fn makeArray(shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64) mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = ta.alloc(u8, n * itemSize(dtype)) catch @panic("oom");
    defer ta.free(buf);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    switch (dtype) {
        .float32 => for (0..n) |i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], @bitCast((r.float(f32) - 0.5) * 8), .little),
        .float16 => for (0..n) |i| std.mem.writeInt(u16, buf[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast((r.float(f32) - 0.5) * 8))), .little),
        .bfloat16 => for (0..n) |i| std.mem.writeInt(u16, buf[i * 2 ..][0..2], @intCast(@as(u32, @bitCast((r.float(f32) - 0.5) * 8)) >> 16), .little),
        .bool_ => for (buf) |*b| {
            b.* = r.int(u1);
        },
        else => r.bytes(buf),
    }
    return mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), dtype);
}

/// Rows `[13, 13 + rows)` of a buffer with 50 more rows on axis 2: a lazy strided view, as the tier's chunk slices are.
fn slicedArray(t: TSpec, seed: u64) !mlx.mlx_array {
    var big_shape: [4]c_int = undefined;
    @memcpy(&big_shape, t.shape);
    big_shape[2] += 50;
    const big = makeArray(&big_shape, t.dtype, seed);
    defer _ = mlx.mlx_array_free(big);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const start = [_]c_int{ 0, 0, 13, 0 };
    const stop = [_]c_int{ t.shape[0], t.shape[1], 13 + t.shape[2], t.shape[3] };
    const stride = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&out, big, &start, 4, &stop, 4, &stride, 4, mlx.gpuStream()));
    return out;
}

/// A tensor list in each shape a consumer wants: the tier's own `NamedTensor`s, and this module's views.
const Built = struct {
    named: std.ArrayList(DiskTier.NamedTensor) = .empty,
    views: []vt.Tensor,
    maps: []?RowMap,

    /// `sliced` hands every 4-D array over as a strided view.
    fn init(spec: KindSpec, sliced: bool) !Built {
        var b: Built = .{ .views = try ta.alloc(vt.Tensor, spec.tensors.len), .maps = try ta.alloc(?RowMap, spec.tensors.len) };
        errdefer b.deinit();
        for (spec.tensors, 0..) |t, i| {
            const key = try ta.dupe(u8, t.key);
            errdefer ta.free(key);
            const seed = std.hash.Wyhash.hash(i, t.key);
            const arr = if (sliced and t.shape.len == 4) try slicedArray(t, seed) else makeArray(t.shape, t.dtype, seed);
            errdefer _ = mlx.mlx_array_free(arr);
            try b.named.append(ta, .{ .key = key, .arr = arr });
            b.views[i] = .{ .key = key, .arr = arr };
            b.maps[i] = t.rows;
        }
        return b;
    }

    fn deinit(self: *Built) void {
        for (self.named.items) |t| {
            ta.free(t.key);
            _ = mlx.mlx_array_free(t.arr);
        }
        self.named.deinit(ta);
        ta.free(self.views);
        ta.free(self.maps);
    }
};

const Env = struct {
    tmp: std.testing.TmpDir,
    buf: [512]u8 = undefined,
    len: usize = 0,

    fn init(self: *Env) !void {
        self.tmp = testing.tmpDir(.{ .iterate = true });
        self.len = try self.tmp.dir.realPath(tio, &self.buf);
    }
    fn deinit(self: *Env) void {
        self.tmp.cleanup();
    }
    fn root(self: *const Env) []const u8 {
        return self.buf[0..self.len];
    }
    fn path(self: *const Env, comptime fmt: []const u8, args: anytype) ![]u8 {
        const tail = try std.fmt.allocPrint(ta, fmt, args);
        defer ta.free(tail);
        return std.fmt.allocPrint(ta, "{s}/{s}", .{ self.root(), tail });
    }
};

fn cstr(text: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(ta, "{s}", .{text}, 0);
}

fn cpuLoad(path: []const u8) !vt.Loaded {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const path_z = try cstr(path);
    defer ta.free(path_z);
    var tensors = mlx.mlx_map_string_to_array_new();
    var meta = mlx.mlx_map_string_to_string_new();
    errdefer {
        _ = mlx.mlx_map_string_to_array_free(tensors);
        _ = mlx.mlx_map_string_to_string_free(meta);
    }
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &meta, path_z.ptr, cpu));
    return .{ .tensors = tensors, .meta = meta };
}

fn mapCount(map: mlx.mlx_map_string_to_array) !usize {
    const it = mlx.mlx_map_string_to_array_iterator_new(map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    var n: usize = 0;
    while (true) {
        var key: ?[*:0]const u8 = null;
        var val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(val);
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &val, it) != 0 or key == null) break;
        n += 1;
    }
    return n;
}

fn expectSameLoads(spec: KindSpec, want: vt.Loaded, got: vt.Loaded) !void {
    try testing.expectEqual(spec.tensors.len, try mapCount(want.tensors));
    try testing.expectEqual(spec.tensors.len, try mapCount(got.tensors));
    for (spec.tensors) |t| {
        const key_z = try cstr(t.key);
        defer ta.free(key_z);
        var a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a);
        var b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b);
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_get(&a, want.tensors, key_z.ptr));
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_get(&b, got.tensors, key_z.ptr));
        try testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
        try testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
        try mlx.check(mlx.mlx_array_eval(a));
        try mlx.check(mlx.mlx_array_eval(b));
        const ba = vt.bytesOf(a) orelse return error.NoBytes;
        const bb = vt.bytesOf(b) orelse return error.NoBytes;
        try testing.expectEqualSlices(u8, ba, bb);
    }
    for (spec.meta) |m| {
        const key_z = try cstr(m.key);
        defer ta.free(key_z);
        var a: [*:0]const u8 = undefined;
        var b: [*:0]const u8 = undefined;
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_string_get(&a, want.meta, key_z.ptr));
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_string_get(&b, got.meta, key_z.ptr));
        try testing.expectEqualStrings(std.mem.span(a), std.mem.span(b));
        try testing.expectEqualStrings(m.value, std.mem.span(b));
    }
}

fn readWhole(path: []const u8) ![]u8 {
    return kv_disk_cache.readFileAlloc(ta, tio, path, 1 << 30) orelse error.FileNotFound;
}

/// One kind of file written by the plain-file tier's two arms (the writer's staged encoder and the synchronous save)
/// and by the VFS: the loads agree on every name, dtype, shape, byte and metadata value, and the VFS size is the writer
/// arm's.
fn expectKindMatchesTier(spec: KindSpec, sliced: bool) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();

    const sync_path = try env.path("sync.safetensors", .{});
    defer ta.free(sync_path);
    const staged_path = try env.path("staged.safetensors", .{});
    defer ta.free(staged_path);

    {
        var tier = try DiskTier.initBackend(ta, tio, env.root(), "fp-sync", 0, 128, .real);
        defer tier.deinit();
        var b = try Built.init(spec, sliced);
        defer b.deinit();
        _ = try tier.saveNamed(try ta.dupe(u8, sync_path), b.named.items, spec.meta, spec.kind, s);
    }
    var staged_size: u64 = 0;
    {
        var tier = try DiskTier.initBackend(ta, tio, env.root(), "fp-staged", 0, 128, .real);
        defer tier.deinit();
        tier.enableBackgroundWriter();
        var b = try Built.init(spec, sliced);
        defer b.deinit();
        staged_size = try tier.saveNamed(try ta.dupe(u8, staged_path), b.named.items, spec.meta, spec.kind, s);
        tier.drainWriter();
        try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    }
    try testing.expectEqual(staged_size, kv_disk_cache.fileSize(tio, staged_path).?);

    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    try vfs.mkdirp("e1");
    var b = try Built.init(spec, sliced);
    defer b.deinit();
    const size = try vt.saveTensors(vfs, "e1/file.safetensors", b.views, spec.meta, b.maps, spec.kind, s);
    try testing.expectEqual(staged_size, size);
    try testing.expectEqual(staged_size, vfs.stat("e1/file.safetensors").?.size);

    const want_bytes = try readWhole(staged_path);
    defer ta.free(want_bytes);
    const regen = try vt.regenerate(vfs, ta, "e1/file.safetensors");
    defer ta.free(regen);
    try testing.expectEqualSlices(u8, want_bytes, regen);

    const loaded = try vt.loadTensors(vfs, "e1/file.safetensors");
    defer loaded.deinit();
    const from_staged = try cpuLoad(staged_path);
    defer from_staged.deinit();
    const from_sync = try cpuLoad(sync_path);
    defer from_sync.deinit();
    try expectSameLoads(spec, from_staged, loaded);
    try expectSameLoads(spec, from_sync, loaded);
}

const kv_rows_map = RowMap{ .axis = 2, .pos0 = 3 * 1024 };

fn kvLayer(comptime li: usize, comptime rows: c_int, comptime heads: c_int, comptime width: c_int, comptime dtype: mlx.mlx_dtype, comptime map: ?RowMap) [2]TSpec {
    return .{
        .{ .key = std.fmt.comptimePrint("l{d}.k", .{li}), .shape = &.{ 1, heads, rows, width }, .dtype = dtype, .rows = map },
        .{ .key = std.fmt.comptimePrint("l{d}.v", .{li}), .shape = &.{ 1, heads, rows, width }, .dtype = dtype, .rows = map },
    };
}

fn affineLayer(comptime li: usize, comptime rows: c_int, comptime heads: c_int, comptime words: c_int, comptime map: ?RowMap) [6]TSpec {
    const kind = [_][]const u8{ "k", "v", "ks", "kb", "vs", "vb" };
    var out: [6]TSpec = undefined;
    inline for (kind, 0..) |name, i| {
        const quant = i < 2;
        out[i] = .{
            .key = std.fmt.comptimePrint("l{d}.{s}", .{ li, name }),
            .shape = &.{ 1, heads, rows, if (quant) words else 1 },
            .dtype = if (quant) .uint32 else .bfloat16,
            .rows = map,
        };
    }
    return out;
}

const dense_f32 = kvLayer(0, 1024, 2, 16, .float32, kv_rows_map) ++ kvLayer(1, 1024, 2, 16, .float32, kv_rows_map);
const dense_bf16 = kvLayer(0, 1024, 4, 32, .bfloat16, kv_rows_map) ++ kvLayer(2, 1024, 4, 32, .bfloat16, kv_rows_map);
const aff4 = affineLayer(0, 1024, 2, 8, kv_rows_map) ++ affineLayer(3, 1024, 2, 8, kv_rows_map);
const aff6 = affineLayer(0, 1024, 2, 12, kv_rows_map) ++ affineLayer(3, 1024, 2, 12, kv_rows_map);
const aff8 = affineLayer(0, 1024, 2, 16, kv_rows_map) ++ affineLayer(3, 1024, 2, 16, kv_rows_map);
const latent = affineLayer(0, 1024, 1, 72, .{ .axis = 2, .pos0 = 5 * 1024 }) ++ kvLayer(5, 1024, 1, 576, .bfloat16, .{ .axis = 2, .pos0 = 5 * 1024 });
const unmapped = kvLayer(0, 700, 2, 16, .float32, null);

const chunk_kinds = [_]KindSpec{
    .{ .name = "dense f32 chunk", .kind = .kv, .tensors = &dense_f32 },
    .{ .name = "dense bf16 chunk", .kind = .kv, .tensors = &dense_bf16 },
    .{ .name = "affine 4 chunk", .kind = .kv, .tensors = &aff4 },
    .{ .name = "affine 6 chunk", .kind = .kv, .tensors = &aff6 },
    .{ .name = "affine 8 chunk", .kind = .kv, .tensors = &aff8 },
    .{ .name = "latent pseudo-cache chunk", .kind = .kv, .tensors = &latent },
    .{ .name = "chunk without row maps", .kind = .kv, .tensors = &unmapped },
};

const hybrid = [_]TSpec{
    .{ .key = "l0.conv", .shape = &.{ 3, 48 }, .dtype = .float32 },
    .{ .key = "l0.ssm", .shape = &.{ 1, 4, 16, 16 }, .dtype = .float32 },
    .{ .key = "l0.aux", .shape = &.{ 1, 1100, 64 }, .dtype = .bfloat16 },
    .{ .key = "l0.pooled", .shape = &.{ 1, 275, 64 }, .dtype = .bfloat16 },
    .{ .key = "l0.ple", .shape = &.{9}, .dtype = .uint32 },
    .{ .key = "l1.conv", .shape = &.{ 3, 48 }, .dtype = .bfloat16 },
    .{ .key = "l1.ssm", .shape = &.{ 1, 1024, 2100 }, .dtype = .float32 },
};
const hybrid_meta = [_]vt.MetaPair{
    .{ .key = "layers", .value = "2" },
    .{ .key = "init", .value = "0,1" },
    .{ .key = "qsa_ratio", .value = "4" },
    .{ .key = "qsa_rows", .value = "1100" },
};
const grown_cut = [_]TSpec{
    .{ .key = "l0.conv", .shape = &.{ 300, 48 }, .dtype = .float32, .rows = .{ .axis = 0, .pos0 = 8192, .per_row = 4 } },
    .{ .key = "l0.ssm", .shape = &.{ 300, 16 }, .dtype = .bfloat16, .rows = .{ .axis = 0, .pos0 = 8192, .per_row = 4 } },
    .{ .key = "l1.conv", .shape = &.{ 3, 48 }, .dtype = .float32 },
};
const grown_meta = [_]vt.MetaPair{ .{ .key = "layers", .value = "2" }, .{ .key = "init", .value = "0,1" }, .{ .key = "grown", .value = "1" } };
const segment = [_]TSpec{
    .{ .key = "l0.conv", .shape = &.{ 2048, 48 }, .dtype = .float32, .rows = .{ .axis = 0, .pos0 = 16384, .per_row = 4 } },
    .{ .key = "l0.ssm", .shape = &.{ 64, 16 }, .dtype = .bfloat16, .rows = .{ .axis = 0, .pos0 = 16384, .per_row = 128 } },
};
const ring = kvLayer(0, 640, 2, 16, .bfloat16, .{ .axis = 2, .pos0 = 360 }) ++ affineLayer(1, 640, 2, 16, .{ .axis = 2, .pos0 = 360 });
const ring_meta = [_]vt.MetaPair{.{ .key = "layers", .value = "3" }};
// The row maps below are the ones each save site gives: `writeQsaHistoryFile` maps the pooled banks (a row per ratio
// tokens, from the tail past the segments), never the raw leftovers.
const history = [_]TSpec{
    .{ .key = "l0.aux", .shape = &.{ 1, 3, 64 }, .dtype = .bfloat16 },
    .{ .key = "l0.pooled", .shape = &.{ 1, 325, 64 }, .dtype = .bfloat16, .rows = .{ .axis = 1, .pos0 = 8192, .per_row = 4 } },
    .{ .key = "l2.aux", .shape = &.{ 1, 3, 64 }, .dtype = .float32 },
    .{ .key = "l2.pooled", .shape = &.{ 1, 325, 64 }, .dtype = .float32, .rows = .{ .axis = 1, .pos0 = 8192, .per_row = 4 } },
};
const history_meta = [_]vt.MetaPair{ .{ .key = "qsa_rows", .value = "9495" }, .{ .key = "qsa_ratio", .value = "4" } };
const history_segment = [_]TSpec{
    .{ .key = "l0.pooled", .shape = &.{ 1, 2048, 64 }, .dtype = .bfloat16, .rows = .{ .axis = 1, .pos0 = 8192, .per_row = 4 } },
    .{ .key = "l2.pooled", .shape = &.{ 1, 2048, 64 }, .dtype = .float32, .rows = .{ .axis = 1, .pos0 = 8192, .per_row = 4 } },
};
// `writeSpecSidecar`: the DFlash KV from its base, the MTP KV and the head's pooled bank from the rows past the
// segments; the head's ring and marks are fixed.
const sidecar = kvLayer(0, 600, 2, 16, .bfloat16, .{ .axis = 2, .pos0 = 77 }) ++ kvLayer(1, 600, 2, 16, .bfloat16, .{ .axis = 2, .pos0 = 77 }) ++
    [_]TSpec{
        .{ .key = "m0.k", .shape = &.{ 1, 1, 600, 16 }, .dtype = .float16, .rows = .{ .axis = 2, .pos0 = 8192 + 1 } },
        .{ .key = "mh.aux", .shape = &.{ 1, 32, 64 }, .dtype = .bfloat16 },
        .{ .key = "mh.pooled", .shape = &.{ 1, 150, 64 }, .dtype = .bfloat16, .rows = .{ .axis = 1, .pos0 = 8192, .per_row = 4 } },
        .{ .key = "mh.lv0", .shape = &.{ 3, 64 }, .dtype = .bfloat16 },
        .{ .key = "mh.lv1", .shape = &.{ 2, 64 }, .dtype = .bfloat16 },
    };
const sidecar_meta = [_]vt.MetaPair{ .{ .key = "d.pos", .value = "77:600" }, .{ .key = "m.pos", .value = "1:8792" } };
const sidecar_segment = [_]TSpec{
    .{ .key = "m0.k", .shape = &.{ 1, 1, 2048, 16 }, .dtype = .float16, .rows = .{ .axis = 2, .pos0 = 4096 + 1 } },
    .{ .key = "m0.v", .shape = &.{ 1, 1, 2048, 16 }, .dtype = .float16, .rows = .{ .axis = 2, .pos0 = 4096 + 1 } },
    .{ .key = "mh.pooled", .shape = &.{ 1, 512, 64 }, .dtype = .bfloat16, .rows = .{ .axis = 1, .pos0 = 4096, .per_row = 4 } },
};
const odd_axes = [_]TSpec{
    .{ .key = "x", .shape = &.{ 2, 700, 8 }, .dtype = .int32, .rows = .{ .axis = 1, .pos0 = 5 } },
    .{ .key = "y", .shape = &.{ 700, 3 }, .dtype = .uint8, .rows = .{ .axis = 0, .pos0 = 5 } },
    .{ .key = "z", .shape = &.{ 3, 4, 700 }, .dtype = .bool_, .rows = .{ .axis = 2, .pos0 = 5 } },
    .{ .key = "w", .shape = &.{ 2, 700 }, .dtype = .int8, .rows = .{ .axis = 1, .pos0 = 255, .per_row = 7 } },
};

const empties = [_]TSpec{
    .{ .key = "a", .shape = &.{ 1, 2, 0, 16 }, .dtype = .float32, .rows = .{ .axis = 2 } },
    .{ .key = "b", .shape = &.{0}, .dtype = .uint32 },
    .{ .key = "c", .shape = &.{ 3, 4 }, .dtype = .float32 },
};

const other_kinds = [_]KindSpec{
    .{ .name = "hybrid point with aux, pooled and ple", .kind = .ssm, .tensors = &hybrid, .meta = &hybrid_meta },
    .{ .name = "point with a grown cut", .kind = .ssm, .tensors = &grown_cut, .meta = &grown_meta },
    .{ .name = "grown segment", .kind = .ssm, .tensors = &segment },
    .{ .name = "ring file", .kind = .ring, .tensors = &ring, .meta = &ring_meta },
    .{ .name = "history file", .kind = .qsa, .tensors = &history, .meta = &history_meta },
    .{ .name = "sidecar file with the head half and marks", .kind = .spec, .tensors = &sidecar, .meta = &sidecar_meta },
    .{ .name = "rows on axes 0, 1 and 2 of other dtypes", .kind = .kv, .tensors = &odd_axes },
    .{ .name = "empty tensors", .kind = .kv, .tensors = &empties },
    .{ .name = "history segment", .kind = .qsa, .tensors = &history_segment },
    .{ .name = "sidecar segment", .kind = .spec, .tensors = &sidecar_segment },
};

test "tensor files on the vfs: every chunk kind equals what the tier's safetensors path writes and loads" {
    for (chunk_kinds) |spec| expectKindMatchesTier(spec, false) catch |e| {
        std.debug.print("kind: {s}: {s}\n", .{ spec.name, @errorName(e) });
        return e;
    };
}

test "tensor files on the vfs: points, segments, rings, histories and sidecars equal what the tier writes and loads" {
    for (other_kinds) |spec| expectKindMatchesTier(spec, false) catch |e| {
        std.debug.print("kind: {s}: {s}\n", .{ spec.name, @errorName(e) });
        return e;
    };
}

test "tensor files on the vfs: lazy strided slices, as the tier stages them, equal what the tier writes" {
    for (chunk_kinds[0..6]) |spec| expectKindMatchesTier(spec, true) catch |e| {
        std.debug.print("kind: {s}: {s}\n", .{ spec.name, @errorName(e) });
        return e;
    };
    for ([_]usize{ 3, 5 }) |i| expectKindMatchesTier(other_kinds[i], true) catch |e| {
        std.debug.print("kind: {s}: {s}\n", .{ other_kinds[i].name, @errorName(e) });
        return e;
    };
}

/// Rows of a file of `rows` rows, the first `old` of which an earlier file held, that fall in a group the earlier file
/// did not hold whole.
fn newRows(map: RowMap, old: u64, rows: u64) u64 {
    var n: u64 = 0;
    var i: u64 = 0;
    while (i < rows) {
        const g = (map.pos0 + i * map.per_row) / vt.GROUP_TOKENS;
        var end = i + 1;
        while (end < rows and (map.pos0 + end * map.per_row) / vt.GROUP_TOKENS == g) end += 1;
        if ((map.pos0 + old * map.per_row) / vt.GROUP_TOKENS <= g) n += end - i;
        i = end;
    }
    return n;
}

fn prefixRows(arr: mlx.mlx_array, axis: usize, rows: usize) !mlx.mlx_array {
    const shape = mlx.getShape(arr);
    var stop: [8]c_int = undefined;
    const start: [8]c_int = @splat(0);
    const stride: [8]c_int = @splat(1);
    @memcpy(stop[0..shape.len], shape);
    stop[axis] = @intCast(rows);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &start, shape.len, &stop, shape.len, &stride, shape.len, mlx.gpuStream()));
    return out;
}

const Grown = struct { key: []const u8, tail: [2]c_int, dtype: mlx.mlx_dtype };
const growth = [_]Grown{
    .{ .key = "l0.k", .tail = .{ 2, 16 }, .dtype = .bfloat16 },
    .{ .key = "l0.ks", .tail = .{ 2, 1 }, .dtype = .float32 },
};

/// A file of the first `old` rows and then of the first `rows` rows of the same buffers writes only the rows of the groups
/// the first did not hold whole, plus the new header.
fn expectGrowthWritesOnlyNewGroups(pos0: u64, old: usize, rows: usize) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();

    var full: [growth.len]mlx.mlx_array = undefined;
    var part: [growth.len]mlx.mlx_array = undefined;
    var maps: [growth.len]?RowMap = undefined;
    var full_t: [growth.len]vt.Tensor = undefined;
    var part_t: [growth.len]vt.Tensor = undefined;
    defer for (full) |a| {
        _ = mlx.mlx_array_free(a);
    };
    defer for (part) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (growth, 0..) |g, i| {
        const shape = [_]c_int{ 1, g.tail[0], @intCast(rows), g.tail[1] };
        full[i] = makeArray(&shape, g.dtype, i + 7);
        part[i] = try prefixRows(full[i], 2, old);
        maps[i] = .{ .axis = 2, .pos0 = pos0 };
        full_t[i] = .{ .key = g.key, .arr = full[i] };
        part_t[i] = .{ .key = g.key, .arr = part[i] };
    }
    _ = try vt.saveTensors(vfs, "part", &part_t, &.{}, &maps, .kv, s);
    const before = vfs.stats(.kv).data_bytes;
    const size = try vt.saveTensors(vfs, "full", &full_t, &.{}, &maps, .kv, s);
    const wrote = vfs.stats(.kv).data_bytes - before;

    var data: u64 = 0;
    var expect: u64 = 0;
    for (full, maps) |arr, m| {
        const nb = mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr);
        data += nb;
        expect += nb / rows * newRows(m.?, old, rows);
    }
    expect += size - data;
    try testing.expectEqual(expect, wrote);
}

test "tensor files on the vfs: a chunk tail and then the whole chunk write only the rows past the tail's whole groups" {
    try expectGrowthWritesOnlyNewGroups(4096, 600, 1024);
    try expectGrowthWritesOnlyNewGroups(4096, 1023, 1024);
    try expectGrowthWritesOnlyNewGroups(4096, 255, 1024);
    try expectGrowthWritesOnlyNewGroups(0, 256, 1024);
    try expectGrowthWritesOnlyNewGroups(4196, 900, 1024);
    try expectGrowthWritesOnlyNewGroups(77, 1, 1024);
}

test "tensor files on the vfs: an aligned chunk tail then the whole chunk writes C minus the tail's whole groups in rows" {
    const c: u64 = 1024;
    for ([_]u64{ 1, 255, 256, 257, 600, 1023 }) |l| {
        try testing.expectEqual(c - l / 256 * 256, newRows(.{ .axis = 2, .pos0 = 8192 }, l, c));
    }
}

/// `spec`'s file without the last `drop` tokens of its row tensors, then the whole file at the same path: the rewrite
/// stores its header and the rows of the groups the first file did not hold whole, and nothing of its fixed tensors.
fn expectKindGrowth(spec: KindSpec, drop: u64) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();

    var b = try Built.init(spec, false);
    defer b.deinit();
    const short = try ta.alloc(vt.Tensor, spec.tensors.len);
    defer ta.free(short);
    const owned = try ta.alloc(?mlx.mlx_array, spec.tensors.len);
    defer ta.free(owned);
    @memset(owned, null);
    defer for (owned) |o| if (o) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var data: u64 = 0;
    var expect: u64 = 0;
    var changed = false;
    for (spec.tensors, b.views, short, owned) |t, v, *sh, *o| {
        const nb = mlx.mlx_array_size(v.arr) * mlx.mlx_array_itemsize(v.arr);
        data += nb;
        sh.* = v;
        const m = t.rows orelse continue;
        const rows: u64 = @intCast(t.shape[m.axis]);
        const old = rows - drop / m.per_row;
        // A tensor that lost no row is the same blocks.
        if (old == rows) continue;
        changed = true;
        o.* = try prefixRows(v.arr, m.axis, @intCast(old));
        sh.arr = o.*.?;
        expect += nb / rows * newRows(m, old, rows);
    }
    _ = try vt.saveTensors(vfs, "f", short, spec.meta, b.maps, spec.kind, s);
    const before = vfs.stats(spec.kind).data_bytes;
    const size = try vt.saveTensors(vfs, "f", b.views, spec.meta, b.maps, spec.kind, s);
    try testing.expectEqual(if (changed) expect + size - data else 0, vfs.stats(spec.kind).data_bytes - before);
}

test "tensor files on the vfs: every row-mapped kind rewritten with more rows stores only the groups from the old partial one" {
    for (other_kinds) |spec| {
        const mapped = for (spec.tensors) |t| {
            if (t.rows != null and t.shape[t.rows.?.axis] > 0) break true;
        } else false;
        if (!mapped) continue;
        for ([_]u64{ 1, 200, 256, 300 }) |drop| expectKindGrowth(spec, drop) catch |e| {
            std.debug.print("kind: {s}, drop {d}: {s}\n", .{ spec.name, drop, @errorName(e) });
            return e;
        };
    }
}

test "tensor files on the vfs: a history rewritten with more rows writes only the groups from the old partial one" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();

    const rows = 1400;
    const old = 1300;
    const aux = makeArray(&.{ 1, rows, 64 }, .bfloat16, 1);
    defer _ = mlx.mlx_array_free(aux);
    const pooled = makeArray(&.{ 1, rows / 4, 64 }, .bfloat16, 2);
    defer _ = mlx.mlx_array_free(pooled);
    const maps = [_]?RowMap{ .{ .axis = 1 }, .{ .axis = 1, .per_row = 4 } };
    const full = [_]vt.Tensor{ .{ .key = "l0.aux", .arr = aux }, .{ .key = "l0.pooled", .arr = pooled } };
    const aux_old = try prefixRows(aux, 1, old);
    defer _ = mlx.mlx_array_free(aux_old);
    const pooled_old = try prefixRows(pooled, 1, old / 4);
    defer _ = mlx.mlx_array_free(pooled_old);
    const prev = [_]vt.Tensor{ .{ .key = "l0.aux", .arr = aux_old }, .{ .key = "l0.pooled", .arr = pooled_old } };

    _ = try vt.saveTensors(vfs, "qsa", &prev, &.{}, &maps, .qsa, s);
    const before = vfs.stats(.qsa).data_bytes;
    const size = try vt.saveTensors(vfs, "qsa", &full, &.{}, &maps, .qsa, s);
    const wrote = vfs.stats(.qsa).data_bytes - before;

    const row_bytes: u64 = 64 * 2;
    const aux_new = newRows(maps[0].?, old, rows);
    // Group 5 holds positions [1280, 1300) of the old file and [1280, 1400) of the new one.
    try testing.expectEqual(@as(u64, 120), aux_new);
    const header = size - row_bytes * (rows + rows / 4);
    try testing.expectEqual(header + row_bytes * (aux_new + newRows(maps[1].?, old / 4, rows / 4)), wrote);
}

test "tensor files on the vfs: a refused input stages nothing" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();

    const f32_arr = makeArray(&.{ 4, 8 }, .float32, 1);
    defer _ = mlx.mlx_array_free(f32_arr);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var f64_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f64_arr);
    try mlx.check(mlx.mlx_astype(&f64_arr, f32_arr, .float64, cpu));
    const good = [_]vt.Tensor{.{ .key = "a", .arr = f32_arr }};
    const wide = [_]vt.Tensor{.{ .key = "a", .arr = f64_arr }};

    const phys = vfs.physicalBytes();
    try testing.expectError(error.DiskCacheBadDtype, vt.saveTensors(vfs, "w", &wide, &.{}, &.{}, .kv, s));
    try testing.expectError(error.DiskCacheBadShape, vt.saveTensors(vfs, "x", &good, &.{}, &.{.{ .axis = 2 }}, .kv, s));
    try testing.expectError(error.DiskCacheBadShape, vt.saveTensors(vfs, "y", &good, &.{}, &.{.{ .axis = 0, .per_row = 0 }}, .kv, s));
    try testing.expectError(error.DiskCacheBadShape, vt.saveTensors(vfs, "z", &good, &.{}, &.{ null, null }, .kv, s));
    try testing.expectError(error.DiskCacheBadMetadata, vt.saveTensors(vfs, "m", &good, &.{.{ .key = "k", .value = "a\"b" }}, &.{}, .kv, s));
    for ([_][]const u8{ "w", "x", "y", "z", "m" }) |p| try testing.expect(vfs.stat(p) == null);
    try testing.expectEqual(phys, vfs.physicalBytes());
}

test "tensor files on the vfs: a flipped pack byte refuses that file with the tier's MLX error and no other" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    const a = makeArray(&.{ 1, 2, 600, 16 }, .bfloat16, 1);
    defer _ = mlx.mlx_array_free(a);
    const b = makeArray(&.{ 1, 2, 600, 16 }, .bfloat16, 2);
    defer _ = mlx.mlx_array_free(b);
    const maps = [_]?RowMap{.{ .axis = 2 }};
    {
        var vfs = try Vfs.open(ta, tio, vroot, .{});
        defer vfs.close();
        _ = try vt.saveTensors(vfs, "a", &.{.{ .key = "k", .arr = a }}, &.{}, &maps, .kv, s);
        _ = try vt.saveTensors(vfs, "b", &.{.{ .key = "k", .arr = b }}, &.{}, &maps, .kv, s);
    }
    const pack_path = try env.path("vfs/v/packs/0.pack", .{});
    defer ta.free(pack_path);
    {
        const f = try std.Io.Dir.openFileAbsolute(tio, pack_path, .{ .mode = .read_write });
        defer f.close(tio);
        var one: [1]u8 = undefined;
        _ = try f.readPositionalAll(tio, &one, 100);
        one[0] ^= 0xff;
        try f.writePositionalAll(tio, &one, 100);
    }
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    try testing.expectError(error.MlxError, vt.loadTensors(vfs, "a"));
    try testing.expectError(error.MlxError, vt.loadTensors(vfs, "a"));
    const ok = try vt.loadTensors(vfs, "b");
    defer ok.deinit();
    try testing.expectError(error.FileNotFound, vt.loadTensors(vfs, "missing"));
}

test "tensor files on the vfs: a staged file lands from another thread after its arrays are gone" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    const spec = other_kinds[5];

    const ticket = blk: {
        var built = try Built.init(spec, true);
        defer built.deinit();
        break :blk try vt.stageTensors(vfs, "side", .replace, built.views, spec.meta, built.maps, spec.kind, s);
    };
    try testing.expect(vfs.stat("side") == null);

    const Land = struct {
        fn run(v: *Vfs, t: prefix_vfs.Ticket, out: *?anyerror) void {
            _ = v.publish(t) catch |e| {
                out.* = e;
            };
        }
    };
    var failed: ?anyerror = null;
    const th = try std.Thread.spawn(.{}, Land.run, .{ vfs, ticket, &failed });
    th.join();
    try testing.expect(failed == null);
    try testing.expectEqual(ticket.size, vfs.stat("side").?.size);

    var again = try Built.init(spec, true);
    defer again.deinit();
    _ = try vt.saveTensors(vfs, "ref", again.views, spec.meta, again.maps, spec.kind, s);
    const a = try vt.regenerate(vfs, ta, "side");
    defer ta.free(a);
    const b = try vt.regenerate(vfs, ta, "ref");
    defer ta.free(b);
    try testing.expectEqualSlices(u8, b, a);
}

test "tensor files on the vfs: the size is known before any array is read back" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    for (other_kinds) |spec| {
        var built = try Built.init(spec, false);
        defer built.deinit();
        const want = try vt.encodedSize(ta, built.views, spec.meta);
        try testing.expectEqual(want, try vt.saveTensors(vfs, "f", built.views, spec.meta, built.maps, spec.kind, s));
    }
}

/// The tensor keys of a safetensors file in header order, which is the tier's list order.
fn headerKeys(bytes: []const u8, arena: std.mem.Allocator) ![][]const u8 {
    const n = std.mem.readInt(u64, bytes[0..8], .little);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes[8..][0..n], .{ .allocate = .alloc_always });
    var keys: std.ArrayList([]const u8) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (!std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) try keys.append(arena, kv.key_ptr.*);
    }
    return keys.items;
}

/// The chunk files a real commit writes, read back with mlx and stored through the VFS, regenerate byte for byte and
/// load as mlx loads them.
fn expectProductionChunks(config: kv_quant.KVQuantConfig, dtype: mlx.mlx_dtype) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    var arena_state = std.heap.ArenaAllocator.init(ta);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var tier = try DiskTier.initBackend(ta, tio, env.root(), "fp-prod", 0, 0, .real);
    defer tier.deinit();
    tier.enableBackgroundWriter();
    var cache = try trunk.KVCache.initWithConfig(ta, 3, config);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 3, 1100, 64, 0.0, dtype);
    var tokens: [1100]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 3);
    try testing.expectEqual(kv_disk_cache.PersistOutcome.persisted, try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s));
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 0), tier.writeErrors());

    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();

    for (0..2) |idx| {
        const file = try std.fmt.allocPrint(ta, "{s}/e{d}/c{d:0>6}.safetensors", .{ tier.root, tier.entries.items[0].id, idx });
        defer ta.free(file);
        const want = try readWhole(file);
        defer ta.free(want);
        const keys = try headerKeys(want, arena);
        const from_tier = try cpuLoad(file);
        defer from_tier.deinit();

        const views = try ta.alloc(vt.Tensor, keys.len);
        defer ta.free(views);
        const specs = try ta.alloc(TSpec, keys.len);
        defer ta.free(specs);
        const maps = try ta.alloc(?RowMap, keys.len);
        defer ta.free(maps);
        defer for (views) |v| {
            _ = mlx.mlx_array_free(v.arr);
        };
        for (keys, views, specs, maps) |k, *v, *sp, *m| {
            const kz = try cstr(k);
            defer ta.free(kz);
            var arr = mlx.mlx_array_new();
            try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_get(&arr, from_tier.tensors, kz.ptr));
            v.* = .{ .key = k, .arr = arr };
            sp.* = .{ .key = k, .shape = &.{}, .dtype = .float32 };
            m.* = .{ .axis = 2, .pos0 = idx * 1024 };
        }
        const size = try vt.saveTensors(vfs, "chunk", views, &.{}, maps, .kv, s);
        try testing.expectEqual(want.len, size);
        const regen = try vt.regenerate(vfs, ta, "chunk");
        defer ta.free(regen);
        try testing.expectEqualSlices(u8, want, regen);
        const loaded = try vt.loadTensors(vfs, "chunk");
        defer loaded.deinit();
        try expectSameLoads(.{ .name = "", .kind = .kv, .tensors = specs }, from_tier, loaded);
    }
}

test "tensor files on the vfs: the chunk files a real commit writes regenerate byte for byte, dense and affine 4, 6 and 8" {
    try expectProductionChunks(kv_quant.KVQuantConfig.dense, .float32);
    try expectProductionChunks(kv_quant.KVQuantConfig.affine(4), .bfloat16);
    try expectProductionChunks(kv_quant.KVQuantConfig.affine(6), .bfloat16);
    try expectProductionChunks(kv_quant.KVQuantConfig.affine(8), .bfloat16);
}

test "tensor files on the vfs: a file with no row maps is a whole-file write, not a shape error" {
    try mlx.requireGpu();
    const arr = makeArray(&.{ 1, 2, 3, 4 }, .float32, 7);
    defer _ = mlx.mlx_array_free(arr);
    var parts = try vt.Parts.init(ta, &.{.{ .key = "k", .arr = arr }}, &.{}, &.{}, mlx.gpuStream());
    defer parts.deinit();
    try testing.expect(parts.size > 0);
}

test "tensor files on the vfs: a load reads its blocks with no layer lock held" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    const a = makeArray(&.{ 1, 2, 600, 16 }, .bfloat16, 1);
    defer _ = mlx.mlx_array_free(a);
    const maps = [_]?RowMap{.{ .axis = 2 }};
    _ = try vt.saveTensors(vfs, "a", &.{.{ .key = "k", .arr = a }}, &.{}, &maps, .kv, s);
    const want = try vt.regenerate(vfs, ta, "a");
    defer ta.free(want);

    var gate: prefix_vfs.Gate = .{ .at = .read };
    vfs.gate = &gate;
    // The writer's side: a layer mutation while the load is held mid-read. The watcher opens the gate either way.
    const Writer = struct {
        done: std.atomic.Value(bool) = .init(false),
        free: bool = false,
        fn mutate(self: *@This(), v: *Vfs) void {
            v.writeBytes("other", .meta, "xyz") catch {};
            self.done.store(true, .release);
        }
        fn watch(self: *@This(), v: *Vfs, g: *prefix_vfs.Gate) void {
            defer g.open.store(true, .release);
            var waited: usize = 0;
            while (!g.entered.load(.acquire) and waited < 5000) : (waited += 1) std.Io.sleep(tio, .fromMilliseconds(1), .real) catch {};
            if (!g.entered.load(.acquire)) return;
            const t = std.Thread.spawn(.{}, mutate, .{ self, v }) catch return;
            waited = 0;
            while (!self.done.load(.acquire) and waited < 2000) : (waited += 1) std.Io.sleep(tio, .fromMilliseconds(1), .real) catch {};
            self.free = self.done.load(.acquire);
            g.open.store(true, .release);
            t.join();
        }
    };
    var writer: Writer = .{};
    const w = try std.Thread.spawn(.{}, Writer.watch, .{ &writer, vfs, &gate });
    const loaded = vt.loadTensors(vfs, "a");
    w.join();
    const got = try loaded;
    defer got.deinit();
    try testing.expect(gate.entered.load(.acquire));
    try testing.expect(writer.free);
    try testing.expectEqual(@as(u64, 3), vfs.stat("other").?.size);
    const again = try vt.regenerate(vfs, ta, "a");
    defer ta.free(again);
    try testing.expectEqualSlices(u8, want, again);
}

test "tensor files on the vfs: a flipped byte in any block of a file, header, fixed or row tensor, refuses its load" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const vroot = try env.path("vfs", .{});
    defer ta.free(vroot);
    const f = makeArray(&.{ 3, 4000 }, .float32, 5);
    defer _ = mlx.mlx_array_free(f);
    const k = makeArray(&.{ 1, 2, 600, 16 }, .bfloat16, 6);
    defer _ = mlx.mlx_array_free(k);
    const maps = [_]?RowMap{ null, .{ .axis = 2 } };
    var n_blocks: usize = 0;
    {
        var vfs = try Vfs.open(ta, tio, vroot, .{});
        defer vfs.close();
        _ = try vt.saveTensors(vfs, "a", &.{ .{ .key = "f", .arr = f }, .{ .key = "k", .arr = k } }, &.{}, &maps, .kv, s);
        const pf = try vfs.pinFile(ta, "a");
        defer vfs.unpinFile(ta, pf, &.{});
        n_blocks = pf.placed.len;
        var fixed: usize = 0;
        var rows: usize = 0;
        for (pf.placed) |pl| {
            if (pl.tensor == prefix_vfs.UNTAGGED) continue;
            if (pl.tensor >> 24 == 0) fixed += 1 else rows += 1;
        }
        try testing.expect(fixed >= 1 and rows >= 2);
    }

    // A fresh open per block: a block this process wrote or already read whole is trusted.
    for (0..n_blocks) |i| {
        var vfs = try Vfs.open(ta, tio, vroot, .{});
        defer vfs.close();
        const pf = try vfs.pinFile(ta, "a");
        const pl = pf.placed[i];
        vfs.unpinFile(ta, pf, &.{});
        var one: [1]u8 = undefined;
        const at = pl.off + pl.len / 2;
        _ = try pl.file.readPositionalAll(tio, &one, at);
        one[0] ^= 0x5a;
        try pl.file.writePositionalAll(tio, &one, at);
        try testing.expectError(error.MlxError, vt.loadTensors(vfs, "a"));
        one[0] ^= 0x5a;
        try pl.file.writePositionalAll(tio, &one, at);
    }

    var vfs = try Vfs.open(ta, tio, vroot, .{});
    defer vfs.close();
    const got = try vt.loadTensors(vfs, "a");
    defer got.deinit();
    for ([_]struct { []const u8, mlx.mlx_array }{ .{ "f", f }, .{ "k", k } }) |want| {
        const key_z = try cstr(want[0]);
        defer ta.free(key_z);
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        try testing.expectEqual(@as(c_int, 0), mlx.mlx_map_string_to_array_get(&arr, got.tensors, key_z.ptr));
        try mlx.check(mlx.mlx_array_eval(arr));
        try mlx.check(mlx.mlx_array_eval(want[1]));
        try testing.expectEqualSlices(u8, vt.bytesOf(want[1]) orelse return error.NoBytes, vt.bytesOf(arr) orelse return error.NoBytes);
    }
}
