//! `sushi convert vision-affine8`: a Hy4-Flash pack's vision tower rewritten as MLX affine 8 (docs/pack-format.md#hy4-stq).
//! The tower's matrices (`hy4_census.visionAffine`) become affine triples at the tower's group
//! (`hy4_config.visionGroup`); a matrix no group fits stays BF16 and `vision_quant.bf16_kept` names it; the tower's
//! other tensors are kept as stored. Every vision tensor moves into
//! `model-vision.safetensors`. A shard that held one is written again without it (its metadata and other tensors'
//! bytes as they were); every other file is cloned copy-on-write. The index points the tower at the new shard and
//! `config.json` gains `vision_quant`. The source pack is never written.
const std = @import("std");
const io_util = @import("mlx_host").io_util;
const host = @import("mlx_host");
const source = @import("source.zig");
const pack = @import("pack.zig");
const trunk_from = @import("trunk_from.zig");

const hy4_config = host.hy4_config;
const census = host.hy4_census;

pub const vision_file = "model-vision.safetensors";
pub const bits: u32 = 8;

pub const Options = struct {
    model_dir: []const u8,
    out_dir: []const u8,
    version: []const u8,
};

pub const Summary = struct {
    /// Matrices written as affine triples.
    quantized: usize = 0,
    /// Vision tensors kept as stored, and among them the matrices no group fits (`vision_quant.bf16_kept`).
    stored: usize = 0,
    kept: usize = 0,
    kept_bytes: u64 = 0,
    /// Shards written again without their vision tensors.
    rewritten: usize = 0,
    cloned: usize = 0,
    group: u32 = 0,
    /// The tower's data bytes before and after.
    bytes_before: u64 = 0,
    bytes_after: u64 = 0,
};

extern "c" fn clonefile(src: [*:0]const u8, dst: [*:0]const u8, flags: u32) c_int;

/// A source shard's header and tensors in data order.
const Shard = struct {
    header: trunk_from.Header,
    fd: std.c.fd_t,

    fn open(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, root: []const u8, name: []const u8) !Shard {
        var header = try trunk_from.readHeader(a, io, dir, name);
        errdefer header.parsed.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ root, name }, 0);
        defer a.free(path);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.SourceOpenFailed;
        return .{ .header = header, .fd = fd };
    }

    fn close(self: *Shard) void {
        _ = std.c.close(self.fd);
        self.header.parsed.deinit();
    }

    fn entry(self: *const Shard, name: []const u8) !Entry {
        const v = self.header.root.get(name) orelse return error.IndexedTensorMissing;
        return Entry.of(name, v);
    }

    fn read(self: *const Shard, a: std.mem.Allocator, e: Entry) ![]u8 {
        const bytes = try a.alloc(u8, e.end - e.start);
        errdefer a.free(bytes);
        try source.preadAll(self.fd, bytes, 8 + self.header.len + e.start);
        return bytes;
    }
};

const Entry = struct {
    name: []const u8,
    dtype: []const u8,
    shape: [max_dims]usize = undefined,
    rank: usize,
    start: u64,
    end: u64,

    const max_dims = 8;

    fn of(name: []const u8, v: std.json.Value) !Entry {
        if (v != .object) return error.BadSafetensorsHeader;
        const dt = v.object.get("dtype") orelse return error.BadSafetensorsHeader;
        const sh = v.object.get("shape") orelse return error.BadSafetensorsHeader;
        const offs = v.object.get("data_offsets") orelse return error.BadSafetensorsHeader;
        if (dt != .string or sh != .array or offs != .array or offs.array.items.len != 2 or sh.array.items.len > max_dims) return error.BadSafetensorsHeader;
        var e: Entry = .{ .name = name, .dtype = dt.string, .rank = sh.array.items.len, .start = try offset(offs.array.items[0]), .end = try offset(offs.array.items[1]) };
        if (e.end < e.start) return error.BadSafetensorsHeader;
        for (sh.array.items, e.shape[0..e.rank]) |d, *o| o.* = try offset(d);
        return e;
    }

    fn offset(x: std.json.Value) !u64 {
        if (x != .integer) return error.BadSafetensorsHeader;
        return std.math.cast(u64, x.integer) orelse error.BadSafetensorsHeader;
    }

    fn dims(self: *const Entry) []const usize {
        return self.shape[0..self.rank];
    }

    fn elems(self: *const Entry) u64 {
        var n: u64 = 1;
        for (self.dims()) |d| n *= d;
        return n;
    }

    fn lessThan(_: void, x: Entry, y: Entry) bool {
        return x.start < y.start;
    }
};

fn isVision(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "vit.");
}

pub fn run(a: std.mem.Allocator, io: std.Io, opts: Options) !Summary {
    const cwd = std.Io.Dir.cwd();
    var src = try cwd.openDir(io, opts.model_dir, .{ .iterate = true });
    defer src.close(io);
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const src_root = root_buf[0..try src.realPath(io, &root_buf)];

    const config_raw = try src.readFileAlloc(io, "config.json", a, .limited(16 << 20));
    defer a.free(config_raw);
    if (try source.identifyConfigJson(a, config_raw) != .hy_v4_flash) return error.ArchitectureNotConvertible;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const ar = arena_state.allocator();
    const config = std.json.parseFromSliceLeaky(std.json.Value, ar, config_raw, .{}) catch return error.SourceConfigInvalid;
    if (config != .object) return error.SourceConfigInvalid;
    const text = config.object.get("text_config") orelse return error.SourceConfigInvalid;
    if (text != .object) return error.SourceConfigInvalid;
    const cfg = try hy4_config.parse(text.object, config.object);
    const v = cfg.vision orelse return error.Hy4NoVisionTower;
    if (v.quant != null) return error.VisionAlreadyAffine;
    const group = hy4_config.visionGroup(v.hidden) orelse return error.VisionGroupUnsupported;
    if (src.statFile(io, vision_file, .{})) |_| return error.VisionShardExists else |_| {}

    const index_raw = src.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch |err|
        return if (err == error.FileNotFound) error.IndexMissing else err;
    defer a.free(index_raw);
    const index = std.json.parseFromSliceLeaky(std.json.Value, ar, index_raw, .{}) catch return error.BadSafetensorsIndex;
    if (index != .object) return error.BadSafetensorsIndex;
    const wm_val = index.object.getPtr("weight_map") orelse return error.BadSafetensorsIndex;
    if (wm_val.* != .object) return error.BadSafetensorsIndex;
    const wm = &wm_val.object;

    // The tower's names in order, each with the shard the index names for it, and those shards.
    const Owned = struct { name: []const u8, file: []const u8 };
    var names = std.ArrayList(Owned).empty;
    defer names.deinit(a);
    var owners: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer owners.deinit(a);
    {
        var it = wm.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string) return error.BadSafetensorsIndex;
            if (!isVision(e.key_ptr.*)) continue;
            try names.append(a, .{ .name = e.key_ptr.*, .file = e.value_ptr.string });
            try owners.put(a, e.value_ptr.string, {});
        }
    }
    if (names.items.len == 0) return error.Hy4NoVisionTower;
    std.mem.sortUnstable(Owned, names.items, {}, struct {
        fn lt(_: void, x: Owned, y: Owned) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.lt);

    try io_util.makeDirPath(io, opts.out_dir);
    var out = try cwd.openDir(io, opts.out_dir, .{ .iterate = true });
    defer out.close(io);
    {
        var it = out.iterate();
        if (try it.next(io) != null) return error.OutputNotEmpty;
    }

    var shards = std.StringHashMap(Shard).init(a);
    defer {
        var it = shards.valueIterator();
        while (it.next()) |s| s.close();
        shards.deinit();
    }
    for (owners.keys()) |name| try shards.put(name, try Shard.open(a, io, src, src_root, name));

    var summary = Summary{ .group = group };
    var removed: u64 = 0;
    // Each shard that held a vision tensor, written again without it.
    for (owners.keys()) |file| {
        const sh = shards.getPtr(file).?;
        var kept = std.ArrayList(Entry).empty;
        defer kept.deinit(a);
        var metadata = std.ArrayList([2][]const u8).empty;
        defer metadata.deinit(a);
        var it = sh.header.root.iterator();
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) {
                if (e.value_ptr.* != .object) return error.BadSafetensorsHeader;
                var mit = e.value_ptr.object.iterator();
                while (mit.next()) |m| {
                    if (m.value_ptr.* != .string) return error.BadSafetensorsHeader;
                    try metadata.append(a, .{ m.key_ptr.*, m.value_ptr.string });
                }
                continue;
            }
            const en = try Entry.of(e.key_ptr.*, e.value_ptr.*);
            const owner = wm.get(e.key_ptr.*) orelse std.json.Value{ .null = {} };
            if (isVision(e.key_ptr.*) and owner == .string and std.mem.eql(u8, owner.string, file)) {
                removed += en.end - en.start;
                continue;
            }
            try kept.append(a, en);
        }
        std.mem.sortUnstable(Entry, kept.items, {}, Entry.lessThan);
        var b = pack.ShardBuilder.init(a);
        defer b.deinit();
        for (kept.items) |*en| {
            const bytes = try sh.read(a, en.*);
            defer a.free(bytes);
            try b.add(en.name, en.dtype, en.dims(), bytes);
        }
        try b.flush(io, out, file, metadata.items);
        summary.rewritten += 1;
    }

    // The tower: its matrices as affine triples, the rest as stored.
    var b = pack.ShardBuilder.init(a);
    defer b.deinit();
    var bf16_kept = std.json.Array.init(ar);
    var reason_buf: [96]u8 = undefined;
    const reason = try std.fmt.bufPrint(&reason_buf, "input width {d} divides by no MLX affine group (32, 64, 128)", .{v.inter});
    for (names.items) |owned| {
        const name = owned.name;
        const sh = shards.getPtr(owned.file).?;
        const en = try sh.entry(name);
        const bytes = try sh.read(a, en);
        defer a.free(bytes);
        summary.bytes_before += bytes.len;
        const matrix = census.visionAffine(v, name) orelse {
            if (census.visionKept(v, name)) {
                if (!std.mem.eql(u8, en.dtype, "BF16")) return error.VisionMatrixShape;
                var k = std.json.ObjectMap.empty;
                try k.put(ar, "tensor", .{ .string = name });
                try k.put(ar, "reason", .{ .string = reason });
                try bf16_kept.append(.{ .object = k });
                summary.kept += 1;
                summary.kept_bytes += bytes.len;
            }
            try b.add(name, en.dtype, en.dims(), bytes);
            summary.stored += 1;
            continue;
        };
        if (!std.mem.eql(u8, en.dtype, "BF16") or en.elems() != matrix[0] * matrix[1] or en.shape[0] != matrix[0]) {
            std.log.err("{s}: {s} {d} values, the tower's matrix is BF16 [{d}, {d}]", .{ name, en.dtype, en.elems(), matrix[0], matrix[1] });
            return error.VisionMatrixShape;
        }
        const out_rows: usize = @intCast(matrix[0]);
        const in: usize = @intCast(matrix[1]);
        var t = try source.quantizeAffineBf16(a, std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(bytes))), out_rows, in, bits, group);
        defer t.deinit(a);
        const base = name[0 .. name.len - ".weight".len];
        var nb: [2][160]u8 = undefined;
        try b.add(name, "U32", &.{ out_rows, t.w_cols }, std.mem.sliceAsBytes(t.weight));
        try b.add(try std.fmt.bufPrint(&nb[0], "{s}.scales", .{base}), "BF16", &.{ out_rows, t.s_cols }, std.mem.sliceAsBytes(t.scales));
        try b.add(try std.fmt.bufPrint(&nb[1], "{s}.biases", .{base}), "BF16", &.{ out_rows, t.s_cols }, std.mem.sliceAsBytes(t.biases));
        summary.quantized += 1;
    }
    var group_buf: [8]u8 = undefined;
    var bits_buf: [8]u8 = undefined;
    var metadata = std.ArrayList([2][]const u8).empty;
    defer metadata.deinit(a);
    try metadata.appendSlice(a, &.{
        .{ "format", "affine" },
        .{ "bits", try std.fmt.bufPrint(&bits_buf, "{d}", .{bits}) },
        .{ "group_size", try std.fmt.bufPrint(&group_buf, "{d}", .{group}) },
        .{ "encoder", opts.version },
    });
    if (revisionOf(config.object)) |rev| try metadata.append(a, .{ "pack_revision", rev });

    // The index: the tower's names at the new shard, with each triple's grids.
    var entries = std.ArrayList(pack.IndexEntry).empty;
    defer {
        for (entries.items) |e| {
            a.free(e.tensor);
            a.free(e.file);
        }
        entries.deinit(a);
    }
    try b.flush(io, out, vision_file, metadata.items);
    const vision_path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ opts.out_dir, vision_file }, 0);
    defer a.free(vision_path);
    summary.bytes_after = try pack.appendShardEntries(a, vision_path, vision_file, &entries);
    for (names.items) |owned| _ = wm.orderedRemove(owned.name);
    for (entries.items) |e| try wm.put(ar, e.tensor, .{ .string = vision_file });
    const meta = index.object.getPtr("metadata") orelse return error.BadSafetensorsIndex;
    if (meta.* != .object) return error.BadSafetensorsIndex;
    const total = meta.object.get("total_size") orelse return error.BadSafetensorsIndex;
    if (total != .integer or total.integer < removed) return error.BadSafetensorsIndex;
    try meta.object.put(ar, "total_size", .{ .integer = total.integer - @as(i64, @intCast(removed)) + @as(i64, @intCast(summary.bytes_after)) });

    // Every other file of the pack, cloned.
    {
        var it = src.iterate();
        while (try it.next(io)) |e| {
            if (e.name[0] == '.' or std.mem.eql(u8, e.name, "config.json") or
                std.mem.eql(u8, e.name, "model.safetensors.index.json") or owners.contains(e.name)) continue;
            const from = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ src_root, e.name }, 0);
            defer a.free(from);
            const to = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ opts.out_dir, e.name }, 0);
            defer a.free(to);
            // A clone needs --out on the pack's volume: a copy of the pack is not made in its place.
            if (clonefile(from.ptr, to.ptr, 0) != 0) {
                std.log.err("cannot clone {s} into {s}: --out must be on the pack's volume", .{ from, opts.out_dir });
                return error.CloneFailed;
            }
            summary.cloned += 1;
        }
    }

    const index_body = try std.json.Stringify.valueAlloc(a, index, .{});
    defer a.free(index_body);
    try writeAtomic(io, out, "model.safetensors.index.json", index_body);

    var q = std.json.ObjectMap.empty;
    try q.put(ar, "format", .{ .string = "affine" });
    try q.put(ar, "bits", .{ .integer = bits });
    try q.put(ar, "group_size", .{ .integer = group });
    try q.put(ar, "bf16_kept", .{ .array = bf16_kept });
    try q.put(ar, "encoder", .{ .string = opts.version });
    var root = config.object;
    try root.put(ar, "vision_quant", .{ .object = q });
    const config_body = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{ .whitespace = .indent_2 });
    defer a.free(config_body);
    // The config is written last: a folder without one is an unfinished pass.
    try writeAtomic(io, out, "config.json", config_body);
    return summary;
}

fn revisionOf(root: std.json.ObjectMap) ?[]const u8 {
    const rev = root.get("pack_revision") orelse return null;
    if (rev != .object) return null;
    const id = rev.object.get("id") orelse return null;
    return if (id == .string) id.string else null;
}

fn writeAtomic(io: std.Io, dir: std.Io.Dir, name: []const u8, body: []const u8) !void {
    var tmp_buf: [96]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tmp_buf, ".{s}.tmp", .{name});
    try dir.writeFile(io, .{ .sub_path = tmp, .data = body });
    try dir.rename(tmp, dir, name, io);
}

const testing = std.testing;
const st = @import("safetensors.zig");

fn sha(io: std.Io, dir: std.Io.Dir, name: []const u8) ![32]u8 {
    const raw = try dir.readFileAlloc(io, name, testing.allocator, .limited(1 << 30));
    defer testing.allocator.free(raw);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &out, .{});
    return out;
}

/// A whole shard, its header parsed.
const Whole = struct {
    raw: []align(4) u8,
    file: st.File,

    fn open(io: std.Io, dir: std.Io.Dir, name: []const u8) !Whole {
        const a = testing.allocator;
        const bytes = try dir.readFileAlloc(io, name, a, .limited(1 << 30));
        defer a.free(bytes);
        const raw = try a.alignedAlloc(u8, .of(u32), bytes.len);
        errdefer a.free(raw);
        @memcpy(raw, bytes);
        return .{ .raw = raw, .file = try st.parse(a, raw) };
    }

    fn close(self: *Whole) void {
        self.file.deinit(testing.allocator);
        testing.allocator.free(self.raw);
    }
};

fn bf16(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

// `HY4_STQ_TINY`: an encoded Hy4-Flash STQ pack (`sushi convert exl3` of the tiny STQ source), its tower in
// `model-globals` as the encode stores it.
test "hy4 vision affine8: the tower moves into model-vision as affine-8 triples that decode to the source within their rounding, every other file and tensor byte for byte (HY4_STQ_TINY)" {
    const src_env = std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const out_dir = try std.fmt.allocPrint(a, "{s}/va8", .{root});
    defer a.free(out_dir);
    const summary = try run(a, io, .{ .model_dir = std.mem.span(src_env), .out_dir = out_dir, .version = "test" });

    var src = try std.Io.Dir.cwd().openDir(io, std.mem.span(src_env), .{ .iterate = true });
    defer src.close(io);
    var out = try std.Io.Dir.cwd().openDir(io, out_dir, .{ .iterate = true });
    defer out.close(io);
    const config_raw = try out.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(config_raw);
    const cfg = try hy4_config.parseText(config_raw);
    const v = cfg.vision.?;
    try testing.expectEqual(@as(?hy4_config.VisionQuant, .{ .bits = 8, .group = 32 }), v.quant);
    // The tiny's 64-wide down projections take group 32: it keeps no matrix BF16.
    try testing.expectEqual(@as(usize, 0), summary.kept);
    // The tiny's two blocks' six matrices each and the merger's two.
    try testing.expectEqual(@as(usize, 2 * 6 + 2), summary.quantized);
    try testing.expectEqual(@as(usize, 1), summary.rewritten);

    // Every file but the config, the index, the globals and the new shard: the same bytes.
    var other: usize = 0;
    var it = src.iterate();
    while (try it.next(io)) |e| {
        if (std.mem.eql(u8, e.name, "config.json") or std.mem.eql(u8, e.name, "model.safetensors.index.json") or
            std.mem.eql(u8, e.name, "model-globals.safetensors")) continue;
        try testing.expectEqualSlices(u8, &try sha(io, src, e.name), &try sha(io, out, e.name));
        other += 1;
    }
    try testing.expectEqual(summary.cloned, other);
    // The config: the source's, with `vision_quant` added.
    {
        const before_raw = try src.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(before_raw);
        const before = try std.json.parseFromSlice(std.json.Value, a, before_raw, .{});
        defer before.deinit();
        const after = try std.json.parseFromSlice(std.json.Value, a, config_raw, .{});
        defer after.deinit();
        try testing.expectEqual(before.value.object.count() + 1, after.value.object.count());
        var kit = before.value.object.iterator();
        while (kit.next()) |e| {
            const x = try std.json.Stringify.valueAlloc(a, e.value_ptr.*, .{});
            defer a.free(x);
            const y = try std.json.Stringify.valueAlloc(a, after.value.object.get(e.key_ptr.*).?, .{});
            defer a.free(y);
            try testing.expectEqualStrings(x, y);
        }
    }

    var g_src = try Whole.open(io, src, "model-globals.safetensors");
    defer g_src.close();
    var g_out = try Whole.open(io, out, "model-globals.safetensors");
    defer g_out.close();
    var vis = try Whole.open(io, out, vision_file);
    defer vis.close();
    // The globals keep their stamp and every other tensor's bytes.
    try testing.expectEqual(g_src.file.metadata.count(), g_out.file.metadata.count());
    var mit = g_src.file.metadata.iterator();
    while (mit.next()) |m| try testing.expectEqualStrings(m.value_ptr.*, g_out.file.metadata.get(m.key_ptr.*).?);
    var tit = g_src.file.tensors.iterator();
    var vision_names: usize = 0;
    while (tit.next()) |t| {
        if (isVision(t.key_ptr.*)) {
            vision_names += 1;
            try testing.expect(g_out.file.tensors.get(t.key_ptr.*) == null);
            continue;
        }
        const got = g_out.file.tensors.get(t.key_ptr.*).?;
        try testing.expectEqualStrings(t.value_ptr.dtype, got.dtype);
        try testing.expectEqualSlices(usize, t.value_ptr.shape, got.shape);
        try testing.expectEqualSlices(u8, t.value_ptr.bytes, got.bytes);
    }
    try testing.expectEqual(summary.quantized + summary.stored, vision_names);
    try testing.expectEqualStrings("affine", vis.file.metadata.get("format").?);
    try testing.expectEqualStrings("8", vis.file.metadata.get("bits").?);
    try testing.expectEqualStrings("32", vis.file.metadata.get("group_size").?);
    try testing.expectEqualStrings(g_src.file.metadata.get("pack_revision").?, vis.file.metadata.get("pack_revision").?);

    // Each tower tensor: a matrix's codes decode to its source weight within half a step, plus what the scale's and
    // bias's BF16 rounding (half an ulp, 2^-8 of each) adds over up to 255 steps; any other tensor as stored.
    tit = g_src.file.tensors.iterator();
    while (tit.next()) |t| {
        const name = t.key_ptr.*;
        if (!isVision(name)) continue;
        const got = vis.file.tensors.get(name).?;
        const m = census.visionAffine(v, name) orelse {
            try testing.expectEqualStrings(t.value_ptr.dtype, got.dtype);
            try testing.expectEqualSlices(u8, t.value_ptr.bytes, got.bytes);
            continue;
        };
        var nb: [2][160]u8 = undefined;
        const base = name[0 .. name.len - ".weight".len];
        const sc = vis.file.tensors.get(try std.fmt.bufPrint(&nb[0], "{s}.scales", .{base})).?;
        const bi = vis.file.tensors.get(try std.fmt.bufPrint(&nb[1], "{s}.biases", .{base})).?;
        const rows: usize = @intCast(m[0]);
        const in: usize = @intCast(m[1]);
        try testing.expectEqualStrings("U32", got.dtype);
        try testing.expectEqualSlices(usize, &.{ rows, in / 4 }, got.shape);
        try testing.expectEqualSlices(usize, &.{ rows, in / 32 }, sc.shape);
        try testing.expectEqualSlices(usize, &.{ rows, in / 32 }, bi.shape);
        const w = st.slice(u16, t.value_ptr.*);
        const codes = got.bytes;
        const s = st.slice(u16, sc);
        const b = st.slice(u16, bi);
        for (0..rows) |r| for (0..in) |c| {
            const gi = r * (in / 32) + c / 32;
            const step = @abs(bf16(s[gi]));
            const deq = @as(f32, @floatFromInt(codes[r * in + c])) * bf16(s[gi]) + bf16(b[gi]);
            const bound = step * (0.5 + 255.0 / 256.0) + @abs(bf16(b[gi])) / 256.0 + 1e-6;
            if (!(@abs(deq - bf16(w[r * in + c])) <= bound)) {
                std.debug.print("{s}[{d}, {d}]: decodes to {d}, the source {d}, step {d}\n", .{ name, r, c, deq, bf16(w[r * in + c]), step });
                return error.TestOutsideStep;
            }
        };
    }

    // The index names the new shard for the tower, its grids too, the same shard as before for the rest, and sums
    // the bytes its tensors span.
    {
        const raw = try out.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
        defer a.free(raw);
        const index = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer index.deinit();
        const before_raw = try src.readFileAlloc(io, "model.safetensors.index.json", a, .limited(16 << 20));
        defer a.free(before_raw);
        const before = try std.json.parseFromSlice(std.json.Value, a, before_raw, .{});
        defer before.deinit();
        const wm = index.value.object.get("weight_map").?.object;
        var bit = before.value.object.get("weight_map").?.object.iterator();
        while (bit.next()) |e| {
            const want = if (isVision(e.key_ptr.*)) vision_file else e.value_ptr.string;
            try testing.expectEqualStrings(want, wm.get(e.key_ptr.*).?.string);
        }
        try testing.expectEqual(before.value.object.get("weight_map").?.object.count() + 2 * summary.quantized, wm.count());
        var total: u64 = 0;
        var wit = wm.iterator();
        var files = std.StringHashMap(Whole).init(a);
        defer {
            var fit = files.valueIterator();
            while (fit.next()) |f| f.close();
            files.deinit();
        }
        while (wit.next()) |e| {
            // `total_size` counts the tensors, never a pair-major shard's pads.
            if (std.mem.startsWith(u8, e.key_ptr.*, @import("shard.zig").pad_prefix)) continue;
            const g = try files.getOrPut(e.value_ptr.string);
            if (!g.found_existing) g.value_ptr.* = try Whole.open(io, out, e.value_ptr.string);
            total += g.value_ptr.file.tensors.get(e.key_ptr.*).?.bytes.len;
        }
        try testing.expectEqual(@as(i64, @intCast(total)), index.value.object.get("metadata").?.object.get("total_size").?.integer);
    }

    // A pack whose tower is affine already is refused, as is an --out that holds files.
    const again = try std.fmt.allocPrint(a, "{s}/again", .{root});
    defer a.free(again);
    try testing.expectError(error.VisionAlreadyAffine, run(a, io, .{ .model_dir = out_dir, .out_dir = again, .version = "test" }));
    try testing.expectError(error.OutputNotEmpty, run(a, io, .{ .model_dir = std.mem.span(src_env), .out_dir = out_dir, .version = "test" }));
}
