//! Tensor files on the prefix VFS: content-addressed block codec between arrays and `prefix_vfs.zig`'s byte blocks.
//!
//! A file is a standard safetensors image stored as blocks:
//!   - a tensor with a `RowMap` is cut at absolute token positions that are multiples of `GROUP_TOKENS`; one block holds
//!     one group's rows of one tensor, head by head (`[outer, rows, inner]` gathered per outer index), so a file that
//!     grows or shifts keeps the blocks of every group it holds whole;
//!   - any other tensor is cut at `FIXED_PART` bytes;
//!   - the header block comes last, so the unchanged blocks of a rewritten file are a prefix of its block list.
//! Every segment is tagged `(tensor | axis + 1 << 24, group or part)`, so a load needs no row map: the file says how to
//! put itself back. `loadTensors` pins the file, releases the VFS lock, checks each block's XXH3 hash, and scatters block
//! bytes directly into Cooler `metal.Buffer` tensors or MLX arrays via `safetensors.zig` without reconstructing any
//! safetensors byte stream or calling `mlx_load_safetensors_reader`.

const std = @import("std");
const xxh3 = @import("xxh3.zig");
const mlx = @import("mlx.zig");
const prefix_vfs = @import("prefix_vfs.zig");
const st = @import("safetensors.zig");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const Vfs = prefix_vfs.Vfs;
const Segment = prefix_vfs.Segment;

pub const MetaPair = st.MetaPair;
pub const Tensor = st.Tensor;
pub const Loaded = st.Loaded;
pub const Named = st.Named;
pub const Sink = st.Sink;
pub const bytesOf = st.bytesOf;

/// Rows are grouped by absolute token position in groups of this many tokens.
pub const GROUP_TOKENS: u64 = 256;
/// A tensor without a row map is cut into blocks of this many bytes.
pub const FIXED_PART: usize = 8 << 20;

const AXIS_SHIFT = 24;
const MAX_TENSORS: usize = 1 << AXIS_SHIFT;
const MAX_AXIS = 16;

/// Which axis of a tensor holds rows, the absolute token position of row 0, and the tokens one row covers.
pub const RowMap = struct { axis: u8, pos0: u64 = 0, per_row: u32 = 1 };

/// The byte size of the safetensors file for `tensors` and `meta`, without reading any array back.
pub fn encodedSize(allocator: Allocator, tensors: []const Tensor, meta: []const MetaPair) !u64 {
    const head = try st.encodeHeader(allocator, tensors, meta);
    defer allocator.free(head);
    var n: u64 = head.len;
    for (tensors) |t| n += st.nbytesOf(t.arr);
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
            if (st.nbytesOf(arr) == 0) continue;
            try cutRows(&cuts, allocator, ti, map, @intCast(shape[map.axis]));
        }
    }
    std.mem.sort(Cut, cuts.items, {}, Cut.lessThan);
    for (cuts.items) |c| {
        const arr = conts[c.tensor];
        const map = row_maps[c.tensor].?;
        const rows: usize = @intCast(mlx.getShape(arr)[map.axis]);
        const row_bytes: usize = @intCast(st.nbytesOf(arr) / rows);
        try pieces.append(allocator, .{ .what = .rows, .ti = c.tensor, .tag = rowTag(c.tensor, map.axis), .group = c.group, .from = c.from, .to = c.to, .len = (c.to - c.from) * row_bytes });
    }
    for (conts, 0..) |arr, ti| {
        if (row_maps.len != 0 and row_maps[ti] != null) continue;
        const n: usize = @intCast(st.nbytesOf(arr));
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
    const src = st.bytesOf(arr) orelse return error.DiskCacheUnreadable;
    switch (p.what) {
        .rows => {
            const map = row_maps[p.ti].?;
            const shape = mlx.getShape(arr);
            const rows: usize = @intCast(shape[map.axis]);
            var outer: usize = 1;
            for (shape[0..map.axis]) |d| outer *= @intCast(d);
            const inner_bytes: usize = @intCast(st.nbytesOf(arr) / (outer * rows));
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
        const head = try st.encodeHeader(allocator, tensors, meta);
        errdefer allocator.free(head);
        const conts = try st.materialize(allocator, tensors, s);
        errdefer st.freeArrays(allocator, conts);
        const pieces = try planPieces(allocator, row_maps, conts, head.len);
        var size: u64 = 0;
        for (pieces) |p| size += p.len;
        return .{ .allocator = allocator, .head = head, .conts = conts, .row_maps = row_maps, .pieces = pieces, .size = size };
    }

    pub fn deinit(self: *Parts) void {
        if (self.head) |h| self.allocator.free(h);
        st.freeArrays(self.allocator, self.conts);
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

/// Reads `tensors` back on the caller's thread and holds the file's blocks in `vfs`. The VFS steps that follow
/// (`appendBlocks`, `fillBlocks`, `journalRecord`, `publishTicket`) never touch an array. `row_maps` is empty or
/// parallel to `tensors`. The ticket's `size` is the safetensors file's size. A refused input stages nothing.
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

/// `stageTensors` and the VFS publish in one call: the file is visible when it returns. Returns the file's size.
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

    fn named(self: *const Entry) Named {
        return .{ .key = self.key, .dtype = self.dtype, .shape = self.shape };
    }
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
    const parsed = st.parseHeader(aa, head) catch return error.VfsCorrupt;
    const entries = try aa.alloc(Entry, parsed.entries.len);
    for (parsed.entries, entries) |pe, *e| {
        e.* = .{ .key = pe.key, .dtype = pe.dtype, .shape = pe.shape, .nbytes = pe.nbytes };
    }
    for (tags, 0..) |tag, i| {
        if (i == hi) continue;
        const ti = tag & (MAX_TENSORS - 1);
        const code = tag >> AXIS_SHIFT;
        if (ti >= entries.len) return error.VfsCorrupt;
        const e = &entries[ti];
        if (e.blocks.items.len != 0 and e.code != code) return error.VfsCorrupt;
        e.code = code;
        try e.blocks.append(aa, @intCast(i));
    }
    return .{ .entries = entries, .meta = parsed.meta };
}

/// Loads `path` directly from its VFS blocks into Cooler Metal buffers (when `s` is Cooler's stream) or evaluated MLX
/// arrays, checking each block's XXH3 hash with no layer lock held. A block whose checksum fails, or a file that does
/// not parse, raises `error.MlxError`.
pub fn loadTensors(vfs: *Vfs, path: []const u8, s: mlx.mlx_stream) !Loaded {
    const src = Source.open(vfs, path) catch |e| return loadError(path, e);
    defer src.close();
    return loadPinned(src, s) catch |e| loadError(path, e);
}

fn loadPinned(src: *Source, s: mlx.mlx_stream) !Loaded {
    const scope = if (mlx.engine.isStream(s)) mlx.engine.enter(s) else null;
    defer if (scope) |sc| sc.end();
    var loaded = Loaded.init(src.vfs.allocator);
    errdefer loaded.deinit();
    const BlockFill = struct {
        src: *Source,
        e: *const Entry,
        fn run(ctx: @This(), dst: []u8) anyerror!void {
            try fillEntry(ctx.src.vfs, ctx.src.pf, ctx.src.checked, ctx.e, dst);
        }
    };
    for (src.entries) |*e| {
        const arr = try st.makeArrayFilled(src.vfs.allocator, e.dtype, e.shape, e.nbytes, s, BlockFill{ .src = src, .e = e }, BlockFill.run);
        errdefer _ = mlx.mlx_array_free(arr);
        try loaded.put(e.key, arr);
    }
    for (src.meta) |m| try loaded.putMeta(m.key, m.value);
    return loaded;
}

/// Loads an in-memory safetensors image directly into Cooler Metal buffers or MLX arrays. A buffer that does not parse
/// raises `error.MlxError`.
pub fn loadBytes(allocator: Allocator, bytes: []const u8, s: mlx.mlx_stream) !Loaded {
    return st.loadBytes(allocator, bytes, s) catch |e| switch (e) {
        error.OutOfMemory => e,
        else => error.MlxError,
    };
}

/// The pinned file `path`'s tensors read into the bytes `sink` names, each block checked as `loadTensors` checks it, and
/// no array made; its metadata, in `aa`. A file that does not parse or a block that fails raises `error.MlxError`.
pub fn readInto(vfs: *Vfs, path: []const u8, aa: Allocator, sink: Sink) ![]const MetaPair {
    return readIntoOr(vfs, path, aa, sink) catch |e| loadError(path, e);
}

fn readIntoOr(vfs: *Vfs, path: []const u8, aa: Allocator, sink: Sink) ![]const MetaPair {
    const src = try Source.open(vfs, path);
    defer src.close();
    try sink.begin(src.meta);
    for (src.entries) |*e| if (try sink.take(e.named(), e.nbytes)) |out| try fillEntry(src.vfs, src.pf, src.checked, e, out);
    const meta = try aa.alloc(MetaPair, src.meta.len);
    for (src.meta, meta) |m, *d| d.* = .{ .key = try aa.dupe(u8, m.key), .value = try aa.dupe(u8, m.value) };
    return meta;
}

/// `readInto` for a tensor file held in memory as its safetensors bytes.
pub fn readBytesInto(bytes: []const u8, aa: Allocator, sink: Sink) ![]const MetaPair {
    return st.readBytesInto(bytes, aa, sink) catch |e| switch (e) {
        error.OutOfMemory => e,
        else => error.MlxError,
    };
}

/// A pinned VFS file's parsed header and block list, held across reads with no layer lock held.
const Source = struct {
    vfs: *Vfs,
    arena: std.heap.ArenaAllocator,
    pf: Vfs.PinnedFile,
    checked: []bool,
    head: []u8,
    entries: []Entry,
    meta: []MetaPair = &.{},

    fn open(vfs: *Vfs, path: []const u8) !*Source {
        const a = vfs.allocator;
        const pf = try vfs.pinFile(a, path);
        var pinned = true;
        errdefer if (pinned) vfs.unpinFile(a, pf, &.{});
        const self = try a.create(Source);
        errdefer a.destroy(self);
        self.* = .{ .vfs = vfs, .arena = .init(a), .pf = pf, .checked = &.{}, .head = &.{}, .entries = &.{} };
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
        self.meta = lay.meta;
        pinned = false;
        return self;
    }

    fn close(self: *Source) void {
        const a = self.vfs.allocator;
        self.vfs.unpinFile(a, self.pf, self.checked);
        self.arena.deinit();
        a.destroy(self);
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
        error.VfsCorrupt, error.SafetensorsCorrupt => blk: {
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
