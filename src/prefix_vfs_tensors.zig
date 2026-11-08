//! Tensor files on the prefix VFS: the MLX glue between arrays and `prefix_vfs.zig`'s byte blocks.
//!
//! A file is the safetensors file the SSD tier writes today, stored as blocks:
//!   - a tensor with a `RowMap` is cut at absolute token positions that are multiples of `GROUP_TOKENS`; one block holds
//!     one group's rows of one tensor, head by head (`[outer, rows, inner]` gathered per outer index), so a file that
//!     grows or shifts keeps the blocks of every group it holds whole;
//!   - any other tensor is cut at `FIXED_PART` bytes;
//!   - the header block comes last, so the unchanged blocks of a rewritten file are a prefix of its block list.
//! Every segment is tagged `(tensor | axis + 1 << 24, group or part)`, so a load needs no row map: the file says how to
//! put itself back. All MLX calls (the readback in `stageTensors`, the arrays `loadTensors` builds) run on the caller's
//! thread; the VFS steps that follow a `stageTensors` touch bytes only.

const std = @import("std");
const mlx = @import("mlx.zig");
const prefix_vfs = @import("prefix_vfs.zig");
const disk_format = @import("disk_format.zig");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const Vfs = prefix_vfs.Vfs;
const Segment = prefix_vfs.Segment;

pub const MetaPair = disk_format.MetaPair;

/// Rows are grouped by absolute token position in groups of this many tokens.
pub const GROUP_TOKENS: u64 = 256;
/// A tensor without a row map is cut into blocks of this many bytes.
pub const FIXED_PART: usize = 8 << 20;

const AXIS_SHIFT = 24;
const MAX_TENSORS: usize = 1 << AXIS_SHIFT;
const MAX_AXIS = 16;

pub const Tensor = struct { key: []const u8, arr: mlx.mlx_array };

/// Which axis of a tensor holds rows, the absolute token position of row 0, and the tokens one row covers.
pub const RowMap = struct { axis: u8, pos0: u64 = 0, per_row: u32 = 1 };

/// What `mlx_load_safetensors` returns: the caller owns both maps.
pub const Loaded = struct {
    tensors: mlx.mlx_map_string_to_array,
    meta: mlx.mlx_map_string_to_string,

    pub fn deinit(self: Loaded) void {
        _ = mlx.mlx_map_string_to_array_free(self.tensors);
        _ = mlx.mlx_map_string_to_string_free(self.meta);
    }
};

fn nbytesOf(a: mlx.mlx_array) u64 {
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

fn dtypeName(d: mlx.mlx_dtype) ?[]const u8 {
    return switch (d) {
        .bool_ => "BOOL",
        .uint8 => "U8",
        .uint16 => "U16",
        .uint32 => "U32",
        .uint64 => "U64",
        .int8 => "I8",
        .int16 => "I16",
        .int32 => "I32",
        .int64 => "I64",
        .float16 => "F16",
        .float32 => "F32",
        .bfloat16 => "BF16",
        else => null,
    };
}

fn dtypeFromName(name: []const u8) ?mlx.mlx_dtype {
    for (std.enums.values(mlx.mlx_dtype)) |d| {
        if (dtypeName(d)) |n| if (std.mem.eql(u8, n, name)) return d;
    }
    return null;
}

fn itemSize(d: mlx.mlx_dtype) usize {
    return switch (d) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}

/// The bytes of an evaluated, contiguous array; null for a dtype the tier does not store.
pub fn bytesOf(arr: mlx.mlx_array) ?[]const u8 {
    const n: usize = @intCast(nbytesOf(arr));
    if (n == 0) return &.{};
    const p: [*]const u8 = switch (mlx.mlx_array_dtype(arr)) {
        .bool_ => @ptrCast(mlx.mlx_array_data_bool(arr) orelse return null),
        .uint8, .int8 => @ptrCast(mlx.mlx_array_data_uint8(arr) orelse return null),
        .uint32, .int32 => @ptrCast(mlx.mlx_array_data_uint32(arr) orelse return null),
        .float32 => @ptrCast(mlx.mlx_array_data_float32(arr) orelse return null),
        .float16 => @ptrCast(mlx.mlx_array_data_float16(arr) orelse return null),
        .bfloat16 => @ptrCast(mlx.mlx_array_data_bfloat16(arr) orelse return null),
        else => return null,
    };
    return p[0..n];
}

const HeaderSource = struct {
    tensors: []const Tensor,

    pub fn len(self: HeaderSource) usize {
        return self.tensors.len;
    }

    pub fn desc(self: HeaderSource, i: usize) !disk_format.TensorDesc {
        const t = self.tensors[i];
        return .{
            .key = t.key,
            .dtype = dtypeName(mlx.mlx_array_dtype(t.arr)) orelse return error.DiskCacheBadDtype,
            .shape = mlx.getShape(t.arr),
            .nbytes = nbytesOf(t.arr),
        };
    }
};

fn encodeHeader(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair) ![]u8 {
    return disk_format.encodeHeader(allocator, HeaderSource{ .tensors = tensors }, meta);
}

/// The byte size of the safetensors file for `tensors` and `meta`, without reading any array back.
pub fn encodedSize(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair) !u64 {
    const head = try encodeHeader(allocator, tensors, meta);
    defer allocator.free(head);
    var n: u64 = head.len;
    for (tensors) |t| n += nbytesOf(t.arr);
    return n;
}

fn rowTag(ti: usize, axis: u8) u32 {
    return (@as(u32, axis) + 1) << AXIS_SHIFT | @as(u32, @intCast(ti));
}

fn validate(tensors: []const Tensor, row_maps: []const ?RowMap) !void {
    if (tensors.len >= MAX_TENSORS) return error.DiskCacheBadShape;
    if (row_maps.len != 0 and row_maps.len != tensors.len) return error.DiskCacheBadShape;
    for (row_maps, tensors) |m_opt, t| {
        const m = m_opt orelse continue;
        if (m.per_row == 0 or m.axis >= MAX_AXIS or m.axis >= mlx.mlx_array_ndim(t.arr)) return error.DiskCacheBadShape;
    }
}

/// Every array contiguous and evaluated by one batched `mlx_eval`, as `mlx::core::save_safetensors` does.
fn materialize(allocator: Allocator, tensors: []const Tensor, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const out = try allocator.alloc(mlx.mlx_array, tensors.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (tensors, out) |t, *slot| {
        slot.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_contiguous(slot, t.arr, false, s));
        try mlx.check(mlx.mlx_vector_array_append_value(vec, slot.*));
    }
    try mlx.check(mlx.mlx_eval(vec));
    return out;
}

const Cut = struct {
    group: u32,
    tensor: u32,
    from: usize,
    to: usize,

    fn lessThan(_: void, a: Cut, b: Cut) bool {
        return if (a.group != b.group) a.group < b.group else a.tensor < b.tensor;
    }
};

/// The row ranges of a tensor of `rows` rows, one per group its rows fall in, ascending.
fn cutRows(list: *std.ArrayList(Cut), allocator: Allocator, ti: usize, map: RowMap, rows: usize) !void {
    var i: usize = 0;
    while (i < rows) {
        const g = (map.pos0 + @as(u64, i) * map.per_row) / GROUP_TOKENS;
        if (g > std.math.maxInt(u32)) return error.DiskCacheBadShape;
        // The first row of the next group: the smallest j with pos0 + j * per_row >= (g + 1) * GROUP_TOKENS.
        const edge = (g + 1) * GROUP_TOKENS - map.pos0;
        const next = std.math.divCeil(u64, edge, map.per_row) catch unreachable;
        const to: usize = @intCast(@min(next, rows));
        if (to <= i) return error.DiskCacheBadShape;
        try list.append(allocator, .{ .group = @intCast(g), .tensor = @intCast(ti), .from = i, .to = to });
        i = to;
    }
}

/// Bytes of rows `[from, to)` of a `[outer, rows, inner]` tensor, gathered outer index by outer index.
fn gather(dst: []u8, src: []const u8, outer: usize, rows: usize, inner_bytes: usize, from: usize, to: usize) void {
    const run = (to - from) * inner_bytes;
    for (0..outer) |o| @memcpy(dst[o * run ..][0..run], src[(o * rows + from) * inner_bytes ..][0..run]);
}

fn freeSegments(allocator: Allocator, list: *std.ArrayList(Segment)) void {
    for (list.items) |sg| allocator.free(sg.bytes);
    list.deinit(allocator);
}

fn buildPayload(
    allocator: Allocator,
    tensors: []const Tensor,
    row_maps: []const ?RowMap,
    conts: []const mlx.mlx_array,
    head: []u8,
    kind: prefix_vfs.Kind,
) !prefix_vfs.OwnedPayload {
    var head_live = true;
    errdefer if (head_live) allocator.free(head);
    var segs: std.ArrayList(Segment) = .empty;
    errdefer freeSegments(allocator, &segs);
    var cuts: std.ArrayList(Cut) = .empty;
    defer cuts.deinit(allocator);

    for (tensors, conts, 0..) |_, arr, ti| {
        const m = if (row_maps.len == 0) null else row_maps[ti];
        if (m) |map| {
            const shape = mlx.getShape(arr);
            if (nbytesOf(arr) == 0) continue;
            try cutRows(&cuts, allocator, ti, map, @intCast(shape[map.axis]));
        }
    }
    std.mem.sort(Cut, cuts.items, {}, Cut.lessThan);
    for (cuts.items) |c| {
        const arr = conts[c.tensor];
        const map = row_maps[c.tensor].?;
        const shape = mlx.getShape(arr);
        const rows: usize = @intCast(shape[map.axis]);
        var outer: usize = 1;
        for (shape[0..map.axis]) |d| outer *= @intCast(d);
        const inner_bytes: usize = @intCast(nbytesOf(arr) / (outer * rows));
        const src = bytesOf(arr) orelse return error.DiskCacheUnreadable;
        const buf = try allocator.alloc(u8, outer * (c.to - c.from) * inner_bytes);
        errdefer allocator.free(buf);
        gather(buf, src, outer, rows, inner_bytes, c.from, c.to);
        try segs.append(allocator, .{ .bytes = buf, .tensor = rowTag(c.tensor, map.axis), .group = c.group });
    }

    for (tensors, conts, 0..) |_, arr, ti| {
        if (row_maps.len != 0 and row_maps[ti] != null) continue;
        const src = bytesOf(arr) orelse return error.DiskCacheUnreadable;
        var off: usize = 0;
        var part: u32 = 0;
        while (off < src.len) : (part += 1) {
            const n = @min(FIXED_PART, src.len - off);
            const buf = try allocator.dupe(u8, src[off..][0..n]);
            errdefer allocator.free(buf);
            try segs.append(allocator, .{ .bytes = buf, .tensor = @intCast(ti), .group = part });
            off += n;
        }
    }

    try segs.append(allocator, .{ .bytes = head });
    head_live = false;
    return .{ .kind = kind, .segments = try segs.toOwnedSlice(allocator) };
}

/// Reads `tensors` back from MLX on the caller's thread and holds the file's blocks in `vfs`. The VFS steps that follow
/// (`appendBlocks`, `journalRecord`, `publishTicket`) never touch an array. `row_maps` is empty or parallel to `tensors`.
/// The ticket's `size` is the safetensors file's size. A refused input stages nothing.
pub fn stageTensors(
    vfs: *Vfs,
    path: []const u8,
    mode: prefix_vfs.HoldMode,
    tensors: []const Tensor,
    meta: []const MetaPair,
    row_maps: []const ?RowMap,
    kind: prefix_vfs.Kind,
    s: mlx.mlx_stream,
) !prefix_vfs.Ticket {
    const a = vfs.allocator;
    try validate(tensors, row_maps);
    const head = try encodeHeader(a, tensors, meta);
    const conts = materialize(a, tensors, s) catch |e| {
        a.free(head);
        return e;
    };
    defer {
        for (conts) |c| _ = mlx.mlx_array_free(c);
        a.free(conts);
    }
    const payload = try buildPayload(a, tensors, row_maps, conts, head, kind);
    return vfs.hold(path, mode, payload);
}

/// `stageTensors` and the four VFS steps in one call: the file is visible when it returns. Returns the file's size.
pub fn saveTensors(
    vfs: *Vfs,
    path: []const u8,
    tensors: []const Tensor,
    meta: []const MetaPair,
    row_maps: []const ?RowMap,
    kind: prefix_vfs.Kind,
    s: mlx.mlx_stream,
) !u64 {
    const ticket = try stageTensors(vfs, path, .replace, tensors, meta, row_maps, kind, s);
    return (try vfs.publish(ticket)).size;
}

const Entry = struct {
    key: []const u8,
    dtype: mlx.mlx_dtype,
    shape: []c_int,
    nbytes: usize,
    /// 0 for a fixed tensor, `axis + 1` for a row tensor.
    code: u32 = 0,
    blocks: std.ArrayList(u32) = .empty,
};

/// A file's payload, parsed: the header's tensors and which blocks hold each.
const Decoded = struct {
    arena: std.heap.ArenaAllocator,
    allocator: Allocator,
    payload: prefix_vfs.OwnedPayload,
    head_idx: usize,
    entries: []Entry,
    meta: []MetaPair,

    fn deinit(self: *Decoded) void {
        self.payload.deinit(self.allocator);
        self.arena.deinit();
    }

    fn header(self: *const Decoded) []const u8 {
        return self.payload.segments[self.head_idx].bytes;
    }

    /// Entry `e`'s bytes, put back in file order; its blocks are freed as they are consumed.
    fn tensorBytes(self: *Decoded, e: *const Entry) ![]u8 {
        const a = self.allocator;
        const out = try a.alloc(u8, e.nbytes);
        errdefer a.free(out);
        const segs = self.payload.segments;
        if (e.code == 0) {
            var off: usize = 0;
            var prev: ?u32 = null;
            for (e.blocks.items) |bi| {
                const sg = segs[bi];
                if (prev != null and sg.group <= prev.?) return error.VfsCorrupt;
                prev = sg.group;
                if (sg.bytes.len > e.nbytes - off) return error.VfsCorrupt;
                @memcpy(out[off..][0..sg.bytes.len], sg.bytes);
                off += sg.bytes.len;
                a.free(sg.bytes);
                segs[bi].bytes = &.{};
            }
            if (off != e.nbytes) return error.VfsCorrupt;
            return out;
        }
        const axis = e.code - 1;
        if (axis >= e.shape.len) return error.VfsCorrupt;
        const rows: usize = @intCast(e.shape[axis]);
        var outer: usize = 1;
        for (e.shape[0..axis]) |d| outer *= @intCast(d);
        if (outer * rows == 0) return error.VfsCorrupt;
        const inner_bytes = e.nbytes / (outer * rows);
        var at: usize = 0;
        var prev: ?u32 = null;
        for (e.blocks.items) |bi| {
            const sg = segs[bi];
            if (prev != null and sg.group <= prev.?) return error.VfsCorrupt;
            prev = sg.group;
            const run_unit = outer * inner_bytes;
            if (sg.bytes.len == 0 or sg.bytes.len % run_unit != 0) return error.VfsCorrupt;
            const n = sg.bytes.len / run_unit;
            if (n > rows - at) return error.VfsCorrupt;
            const run = n * inner_bytes;
            for (0..outer) |o| @memcpy(out[(o * rows + at) * inner_bytes ..][0..run], sg.bytes[o * run ..][0..run]);
            at += n;
            a.free(sg.bytes);
            segs[bi].bytes = &.{};
        }
        if (at != rows) return error.VfsCorrupt;
        return out;
    }
};

fn jsonInt(v: std.json.Value) ?i64 {
    return if (v == .integer) v.integer else null;
}

fn decode(vfs: *Vfs, allocator: Allocator, path: []const u8) !Decoded {
    var payload = try vfs.readPayload(allocator, path);
    errdefer payload.deinit(allocator);
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    var head_idx: ?usize = null;
    for (payload.segments, 0..) |sg, i| {
        if (sg.tensor != prefix_vfs.UNTAGGED) continue;
        if (head_idx != null) return error.VfsCorrupt;
        head_idx = i;
    }
    const hi = head_idx orelse return error.VfsCorrupt;
    const head = payload.segments[hi].bytes;
    if (head.len < 8 or std.mem.readInt(u64, head[0..8], .little) != head.len - 8) return error.VfsCorrupt;
    const root = std.json.parseFromSliceLeaky(std.json.Value, aa, head[8..], .{ .allocate = .alloc_always }) catch return error.VfsCorrupt;
    if (root != .object) return error.VfsCorrupt;

    var entries: std.ArrayList(Entry) = .empty;
    var meta: std.ArrayList(MetaPair) = .empty;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) {
            if (kv.value_ptr.* != .object) return error.VfsCorrupt;
            var mit = kv.value_ptr.object.iterator();
            while (mit.next()) |m| {
                if (m.value_ptr.* != .string) return error.VfsCorrupt;
                try meta.append(aa, .{ .key = m.key_ptr.*, .value = m.value_ptr.string });
            }
            continue;
        }
        const obj = kv.value_ptr.*;
        if (obj != .object) return error.VfsCorrupt;
        const dt = obj.object.get("dtype") orelse return error.VfsCorrupt;
        const sh = obj.object.get("shape") orelse return error.VfsCorrupt;
        if (dt != .string or sh != .array) return error.VfsCorrupt;
        const dtype = dtypeFromName(dt.string) orelse return error.VfsCorrupt;
        const shape = try aa.alloc(c_int, sh.array.items.len);
        var nbytes: usize = itemSize(dtype);
        for (sh.array.items, shape) |d, *slot| {
            const v = std.math.cast(c_int, jsonInt(d) orelse return error.VfsCorrupt) orelse return error.VfsCorrupt;
            if (v < 0) return error.VfsCorrupt;
            slot.* = v;
            nbytes *= @intCast(v);
        }
        try entries.append(aa, .{ .key = kv.key_ptr.*, .dtype = dtype, .shape = shape, .nbytes = nbytes });
    }

    for (payload.segments, 0..) |sg, i| {
        if (i == hi) continue;
        const ti = sg.tensor & (MAX_TENSORS - 1);
        const code = sg.tensor >> AXIS_SHIFT;
        if (ti >= entries.items.len) return error.VfsCorrupt;
        const e = &entries.items[ti];
        if (e.blocks.items.len != 0 and e.code != code) return error.VfsCorrupt;
        e.code = code;
        try e.blocks.append(aa, @intCast(i));
    }
    return .{ .arena = arena, .allocator = allocator, .payload = payload, .head_idx = hi, .entries = entries.items, .meta = meta.items };
}

/// The file as `mlx_load_safetensors` would give it: same names, dtypes, shapes, bytes and metadata. A block whose
/// checksum fails, or a file that does not parse, raises `error.MlxError`, the error a corrupt file raises today.
pub fn loadTensors(vfs: *Vfs, path: []const u8) !Loaded {
    var dec = decode(vfs, vfs.allocator, path) catch |e| return loadError(path, e);
    defer dec.deinit();
    const tensors = mlx.mlx_map_string_to_array_new();
    const meta = mlx.mlx_map_string_to_string_new();
    const loaded: Loaded = .{ .tensors = tensors, .meta = meta };
    errdefer loaded.deinit();
    const a = vfs.allocator;

    for (dec.meta) |m| {
        const kz = try std.fmt.allocPrint(a, "{s}\x00", .{m.key});
        defer a.free(kz);
        const vz = try std.fmt.allocPrint(a, "{s}\x00", .{m.value});
        defer a.free(vz);
        try mlx.check(mlx.mlx_map_string_to_string_insert(meta, @ptrCast(kz.ptr), @ptrCast(vz.ptr)));
    }
    for (dec.entries) |*e| {
        const bytes = dec.tensorBytes(e) catch |err| return loadError(path, err);
        defer a.free(bytes);
        const kz = try std.fmt.allocPrint(a, "{s}\x00", .{e.key});
        defer a.free(kz);
        var dummy: u8 = 0;
        const arr = mlx.mlx_array_new_data(if (bytes.len == 0) &dummy else bytes.ptr, e.shape.ptr, @intCast(e.shape.len), e.dtype);
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_map_string_to_array_insert(tensors, @ptrCast(kz.ptr), arr));
    }
    return loaded;
}

fn loadError(path: []const u8, err: anyerror) anyerror {
    return switch (err) {
        error.VfsCorrupt => blk: {
            log.warn("  [prefix-vfs] {s}: {s}\n", .{ path, @errorName(err) });
            break :blk error.MlxError;
        },
        else => err,
    };
}

/// The safetensors file's exact bytes, rebuilt from the blocks.
pub fn regenerate(vfs: *Vfs, allocator: Allocator, path: []const u8) ![]u8 {
    var dec = try decode(vfs, allocator, path);
    defer dec.deinit();
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, dec.header());
    for (dec.entries) |*e| {
        const bytes = try dec.tensorBytes(e);
        defer allocator.free(bytes);
        try out.appendSlice(allocator, bytes);
    }
    return out.toOwnedSlice(allocator);
}
