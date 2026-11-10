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
const xxh3 = @import("xxh3.zig");
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
    if (row_maps.len == 0) return;
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

/// One segment of a file before its bytes exist: rows `[from, to)` of a row-mapped tensor, bytes `[from, to)` of a
/// fixed one, or the header.
const Piece = struct {
    what: enum { rows, fixed, head },
    ti: u32 = 0,
    tag: u32 = prefix_vfs.UNTAGGED,
    group: u32 = prefix_vfs.UNTAGGED,
    from: usize = 0,
    to: usize = 0,
    len: usize,
};

/// The file's segments in order: every tensor's row groups (by group, then tensor), the fixed tensors' parts, then
/// the header.
fn planPieces(allocator: Allocator, row_maps: []const ?RowMap, conts: []const mlx.mlx_array, head_len: usize) ![]Piece {
    var pieces: std.ArrayList(Piece) = .empty;
    errdefer pieces.deinit(allocator);
    var cuts: std.ArrayList(Cut) = .empty;
    defer cuts.deinit(allocator);
    for (conts, 0..) |arr, ti| {
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
        const rows: usize = @intCast(mlx.getShape(arr)[map.axis]);
        const row_bytes: usize = @intCast(nbytesOf(arr) / rows);
        try pieces.append(allocator, .{ .what = .rows, .ti = c.tensor, .tag = rowTag(c.tensor, map.axis), .group = c.group, .from = c.from, .to = c.to, .len = (c.to - c.from) * row_bytes });
    }
    for (conts, 0..) |arr, ti| {
        if (row_maps.len != 0 and row_maps[ti] != null) continue;
        const n: usize = @intCast(nbytesOf(arr));
        var off: usize = 0;
        var part: u32 = 0;
        while (off < n) : (part += 1) {
            const len = @min(FIXED_PART, n - off);
            try pieces.append(allocator, .{ .what = .fixed, .ti = @intCast(ti), .tag = @intCast(ti), .group = part, .from = off, .to = off + len, .len = len });
            off += len;
        }
    }
    try pieces.append(allocator, .{ .what = .head, .len = head_len });
    return pieces.toOwnedSlice(allocator);
}

/// The bytes of a rows or fixed piece, into `dst` (`p.len` bytes).
fn fillPiece(p: Piece, row_maps: []const ?RowMap, conts: []const mlx.mlx_array, dst: []u8) !void {
    const arr = conts[p.ti];
    const src = bytesOf(arr) orelse return error.DiskCacheUnreadable;
    switch (p.what) {
        .rows => {
            const map = row_maps[p.ti].?;
            const shape = mlx.getShape(arr);
            const rows: usize = @intCast(shape[map.axis]);
            var outer: usize = 1;
            for (shape[0..map.axis]) |d| outer *= @intCast(d);
            const inner_bytes: usize = @intCast(nbytesOf(arr) / (outer * rows));
            gather(dst, src, outer, rows, inner_bytes, p.from, p.to);
        },
        .fixed => @memcpy(dst, src[p.from..p.to]),
        .head => unreachable,
    }
}

/// A tensor file read back once and handed out in order as payloads of at most `max` bytes (a piece larger than
/// `max` is one of its own): the parts of a file held with `Vfs.holdPart`, each copied only when it is taken.
pub const Parts = struct {
    allocator: Allocator,
    head: ?[]u8,
    conts: []mlx.mlx_array,
    row_maps: []const ?RowMap,
    pieces: []Piece,
    next: usize = 0,
    /// The safetensors file's size.
    size: u64,

    /// `row_maps` is empty or parallel to `tensors`, and must outlive the `Parts`.
    pub fn init(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair, row_maps: []const ?RowMap, s: mlx.mlx_stream) !Parts {
        try validate(tensors, row_maps);
        const head = try encodeHeader(allocator, tensors, meta);
        errdefer allocator.free(head);
        const conts = try materialize(allocator, tensors, s);
        errdefer freeConts(allocator, conts);
        const pieces = try planPieces(allocator, row_maps, conts, head.len);
        var size: u64 = 0;
        for (pieces) |p| size += p.len;
        return .{ .allocator = allocator, .head = head, .conts = conts, .row_maps = row_maps, .pieces = pieces, .size = size };
    }

    pub fn deinit(self: *Parts) void {
        if (self.head) |h| self.allocator.free(h);
        freeConts(self.allocator, self.conts);
        self.allocator.free(self.pieces);
    }

    pub fn done(self: *const Parts) bool {
        return self.next == self.pieces.len;
    }

    fn end(self: *const Parts, max: u64) usize {
        var n: u64 = 0;
        var i = self.next;
        while (i < self.pieces.len) : (i += 1) {
            if (n > 0 and n + self.pieces[i].len > max) break;
            n += self.pieces[i].len;
        }
        return i;
    }

    /// The byte size of the part `take(max, ...)` returns next.
    pub fn nextLen(self: *const Parts, max: u64) u64 {
        var n: u64 = 0;
        for (self.pieces[self.next..self.end(max)]) |p| n += p.len;
        return n;
    }

    /// The next part, its bytes allocated with the `Parts`' allocator.
    pub fn take(self: *Parts, max: u64, kind: prefix_vfs.Kind) !prefix_vfs.OwnedPayload {
        const a = self.allocator;
        const stop = self.end(max);
        var segs: std.ArrayList(Segment) = .empty;
        errdefer freeSegments(a, &segs);
        try segs.ensureTotalCapacity(a, stop - self.next);
        for (self.pieces[self.next..stop]) |p| {
            if (p.what == .head) {
                segs.appendAssumeCapacity(.{ .bytes = self.head.? });
                self.head = null;
                continue;
            }
            const buf = try a.alloc(u8, p.len);
            errdefer a.free(buf);
            try fillPiece(p, self.row_maps, self.conts, buf);
            segs.appendAssumeCapacity(.{ .bytes = buf, .tensor = p.tag, .group = p.group });
        }
        self.next = stop;
        return .{ .kind = kind, .segments = try segs.toOwnedSlice(a) };
    }
};

fn freeConts(allocator: Allocator, conts: []mlx.mlx_array) void {
    for (conts) |c| _ = mlx.mlx_array_free(c);
    allocator.free(conts);
}

/// Reads `tensors` back from MLX on the caller's thread and holds the file's blocks in `vfs`. The VFS steps that follow
/// (`appendBlocks`, `fillBlocks`, `journalRecord`, `publishTicket`) never touch an array. `row_maps` is empty or parallel to `tensors`.
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
    var parts = try Parts.init(vfs.allocator, tensors, meta, row_maps, s);
    defer parts.deinit();
    return vfs.hold(path, mode, try parts.take(std.math.maxInt(u64), kind));
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
    const tags = try aa.alloc(u32, payload.segments.len);
    for (payload.segments, tags) |sg, *t| t.* = sg.tensor;
    const hi = try headIndex(tags);
    const lay = try parseLayout(aa, payload.segments[hi].bytes, tags, hi);
    return .{ .arena = arena, .allocator = allocator, .payload = payload, .head_idx = hi, .entries = lay.entries, .meta = lay.meta };
}

/// The one block a file's header is: the only untagged one.
fn headIndex(tags: []const u32) !usize {
    var head_idx: ?usize = null;
    for (tags, 0..) |t, i| {
        if (t != prefix_vfs.UNTAGGED) continue;
        if (head_idx != null) return error.VfsCorrupt;
        head_idx = i;
    }
    return head_idx orelse error.VfsCorrupt;
}

const Layout = struct { entries: []Entry, meta: []MetaPair };

/// The header `head` (block `hi` of a file whose blocks carry `tags`) parsed: its tensors, each with the blocks that
/// hold it, and its metadata. Allocated in `aa`.
fn parseLayout(aa: Allocator, head: []const u8, tags: []const u32, hi: usize) !Layout {
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

    for (tags, 0..) |tag, i| {
        if (i == hi) continue;
        const ti = tag & (MAX_TENSORS - 1);
        const code = tag >> AXIS_SHIFT;
        if (ti >= entries.items.len) return error.VfsCorrupt;
        const e = &entries.items[ti];
        if (e.blocks.items.len != 0 and e.code != code) return error.VfsCorrupt;
        e.code = code;
        try e.blocks.append(aa, @intCast(i));
    }
    return .{ .entries = entries.items, .meta = meta.items };
}

/// The file as `mlx_load_safetensors` would give it: same names, dtypes, shapes, bytes and metadata. A block whose
/// checksum fails, or a file that does not parse, raises `error.MlxError`, the error a corrupt file raises today. The
/// file's blocks are pinned and MLX reads them as it reads a plain file: lazily, on its IO threads, into its own
/// buffers, with no layer lock held. Every tensor is evaluated before the load returns, so a bad block fails here.
pub fn loadTensors(vfs: *Vfs, path: []const u8) !Loaded {
    const src = Source.open(vfs, path) catch |e| return loadError(path, e);
    const reader = mlx.mlx_io_reader_new(src, Source.vtable);
    if (reader.ctx == null) {
        src.close();
        return error.MlxError;
    }
    // From here MLX owns `src`: it goes with the reader and the last tensor read from it.
    defer _ = mlx.mlx_io_reader_free(reader);
    return loadFrom(src, reader) catch |e| loadError(path, e);
}

fn loadFrom(src: *Source, reader: mlx.mlx_io_reader) !Loaded {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var loaded: Loaded = .{ .tensors = mlx.mlx_map_string_to_array_new(), .meta = mlx.mlx_map_string_to_string_new() };
    errdefer loaded.deinit();
    try mlx.check(mlx.mlx_load_safetensors_reader(&loaded.tensors, &loaded.meta, reader, cpu));
    const all = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(all);
    const it = mlx.mlx_map_string_to_array_iterator_new(loaded.tensors);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &arr, it) != 0 or key == null) break;
        try mlx.check(mlx.mlx_vector_array_append_value(all, arr));
    }
    try mlx.check(mlx.mlx_eval(all));
    if (src.bad.load(.acquire)) return error.VfsCorrupt;
    return loaded;
}

/// A pinned file served to MLX as its safetensors bytes: the header from memory, each tensor gathered from its blocks
/// when MLX reads it, each block checked against its hash unless this process already has. A read that fails, or that
/// is not one whole tensor, sets `bad` and leaves its bytes unwritten.
const Source = struct {
    vfs: *Vfs,
    arena: std.heap.ArenaAllocator,
    pf: Vfs.PinnedFile,
    checked: []bool,
    head: []u8,
    entries: []Entry,
    starts: []u64,
    size: u64,
    pos: u64 = 0,
    bad: std.atomic.Value(bool) = .init(false),

    const vtable: mlx.mlx_io_vtable = .{
        .is_open = isOpen,
        .good = good,
        .tell = tell,
        .seek = seek,
        .read = read,
        .read_at_offset = readAtOffset,
        .write = write,
        .label = label,
        .free = free,
    };

    fn open(vfs: *Vfs, path: []const u8) !*Source {
        const a = vfs.allocator;
        const pf = try vfs.pinFile(a, path);
        var pinned = true;
        errdefer if (pinned) vfs.unpinFile(a, pf, &.{});
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .vfs = vfs, .arena = .init(a), .pf = pf, .checked = &.{}, .head = &.{}, .entries = &.{}, .starts = &.{}, .size = 0 };
        errdefer self.arena.deinit();
        const aa = self.arena.allocator();
        self.checked = try aa.alloc(bool, pf.placed.len);
        @memset(self.checked, false);
        const tags = try aa.alloc(u32, pf.placed.len);
        for (pf.placed, tags) |pl, *t| t.* = pl.tensor;
        const hi = try headIndex(tags);
        self.head = try aa.alloc(u8, pf.placed[hi].len);
        try readChecked(vfs, pf.placed[hi], self.head, &self.checked[hi]);
        const lay = try parseLayout(aa, self.head, tags, hi);
        self.entries = lay.entries;
        self.starts = try aa.alloc(u64, lay.entries.len);
        var at: u64 = self.head.len;
        for (lay.entries, self.starts) |e, *st| {
            st.* = at;
            at += e.nbytes;
        }
        self.size = at;
        pinned = false;
        return self;
    }

    fn close(self: *Source) void {
        const a = self.vfs.allocator;
        self.vfs.unpinFile(a, self.pf, self.checked);
        self.arena.deinit();
        a.destroy(self);
    }

    fn of(desc: ?*anyopaque) *Source {
        return @ptrCast(@alignCast(desc.?));
    }

    fn readAt(self: *Source, dst: []u8, off: u64) void {
        if (dst.len == 0) return;
        if (off + dst.len <= self.head.len) {
            @memcpy(dst, self.head[@intCast(off)..][0..dst.len]);
            return;
        }
        for (self.entries, self.starts) |*e, st| {
            if (st != off or e.nbytes != dst.len) continue;
            fillEntry(self.vfs, self.pf, self.checked, e, dst) catch break;
            return;
        }
        self.bad.store(true, .release);
    }

    fn isOpen(_: ?*anyopaque) callconv(.c) bool {
        return true;
    }
    fn good(desc: ?*anyopaque) callconv(.c) bool {
        return !of(desc).bad.load(.acquire);
    }
    fn tell(desc: ?*anyopaque) callconv(.c) usize {
        return @intCast(of(desc).pos);
    }
    fn seek(desc: ?*anyopaque, off: i64, whence: c_int) callconv(.c) void {
        const self = of(desc);
        const base: i64 = switch (whence) {
            std.c.SEEK.SET => 0,
            std.c.SEEK.CUR => @intCast(self.pos),
            std.c.SEEK.END => @intCast(self.size),
            else => return self.bad.store(true, .release),
        };
        const to = base + off;
        if (to < 0) return self.bad.store(true, .release);
        self.pos = @intCast(to);
    }
    fn read(desc: ?*anyopaque, data: [*]u8, n: usize) callconv(.c) void {
        const self = of(desc);
        self.readAt(data[0..n], self.pos);
        self.pos += n;
    }
    fn readAtOffset(desc: ?*anyopaque, data: [*]u8, n: usize, off: usize) callconv(.c) void {
        of(desc).readAt(data[0..n], off);
    }
    fn write(desc: ?*anyopaque, _: [*]const u8, _: usize) callconv(.c) void {
        of(desc).bad.store(true, .release);
    }
    fn label(_: ?*anyopaque) callconv(.c) [*:0]const u8 {
        return "prefix-vfs";
    }
    fn free(desc: ?*anyopaque) callconv(.c) void {
        of(desc).close();
    }
};

/// Extent `pl` read whole into `dst`, checked against its hash unless this process already has.
fn readChecked(vfs: *Vfs, pl: Vfs.Placed, dst: []u8, checked: *bool) !void {
    if (dst.len != pl.len) return error.VfsCorrupt;
    try vfs.readPlaced(pl, 0, dst);
    if (pl.verified) return;
    if (xxh3.hash(dst) != pl.xxh) return error.VfsCorrupt;
    checked.* = true;
}

/// Entry `e`'s bytes gathered from its blocks into `out` in file order (a row tensor's runs scattered back per outer
/// index, as `tensorBytes` does).
fn fillEntry(vfs: *Vfs, pf: Vfs.PinnedFile, checked: []bool, e: *const Entry, out: []u8) !void {
    if (out.len != e.nbytes) return error.VfsCorrupt;
    if (e.code == 0) {
        var off: usize = 0;
        var prev: ?u32 = null;
        for (e.blocks.items) |bi| {
            const pl = pf.placed[bi];
            if (prev != null and pl.group <= prev.?) return error.VfsCorrupt;
            prev = pl.group;
            if (pl.len > e.nbytes - off) return error.VfsCorrupt;
            try readChecked(vfs, pl, out[off..][0..pl.len], &checked[bi]);
            off += pl.len;
        }
        if (off != e.nbytes) return error.VfsCorrupt;
        return;
    }
    const axis = e.code - 1;
    if (axis >= e.shape.len) return error.VfsCorrupt;
    const rows: usize = @intCast(e.shape[axis]);
    var outer: usize = 1;
    for (e.shape[0..axis]) |d| outer *= @intCast(d);
    if (outer * rows == 0) return error.VfsCorrupt;
    const inner_bytes = e.nbytes / (outer * rows);
    const run_unit = outer * inner_bytes;
    var at: usize = 0;
    var prev: ?u32 = null;
    for (e.blocks.items) |bi| {
        const pl = pf.placed[bi];
        if (prev != null and pl.group <= prev.?) return error.VfsCorrupt;
        prev = pl.group;
        if (pl.len == 0 or pl.len % run_unit != 0) return error.VfsCorrupt;
        const n = pl.len / run_unit;
        if (n > rows - at) return error.VfsCorrupt;
        const run = n * inner_bytes;
        var h: xxh3.Stream = .{};
        for (0..outer) |o| {
            const d = out[(o * rows + at) * inner_bytes ..][0..run];
            try vfs.readPlaced(pl, o * run, d);
            if (!pl.verified) h.update(d);
        }
        if (!pl.verified) {
            if (h.final() != pl.xxh) return error.VfsCorrupt;
            checked[bi] = true;
        }
        at += n;
    }
    if (at != rows) return error.VfsCorrupt;
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
