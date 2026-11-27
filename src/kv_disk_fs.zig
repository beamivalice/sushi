//! The SSD tier's file operations on the content-addressed layer (`prefix_vfs.zig`) at `<root>/v`.
//! Every call takes an absolute path under the tier root and routes it through the VFS.

const std = @import("std");
const io_util = @import("io_util.zig");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const prefix_vfs = @import("prefix_vfs.zig");
const vt = @import("prefix_vfs_tensors.zig");
const safetensors = @import("safetensors.zig");
const disk_writer = @import("kv_disk_writer.zig");
const write_stats = @import("disk_write_stats.zig");

const Allocator = std.mem.Allocator;
const Vfs = prefix_vfs.Vfs;

/// What the tier reads of a file.
pub const Stat = struct { size: u64, nlink: u64, inode: u64, mtime_ns: i128 };
pub const DirEntry = prefix_vfs.DirEntry;

pub const Fs = struct {
    allocator: Allocator,
    io: std.Io,
    /// The tier root every path lives under. Borrowed from the tier unless `own_root`.
    root: []const u8,
    own_root: bool = false,
    /// A view of another `Fs`'s layer: `close` leaves it open.
    borrowed: bool = false,
    vfs: *Vfs,
    /// Where a synchronous mutation of the layer counts the pack and journal bytes it added (the tier's `stored`).
    stats: ?*write_stats.Stats = null,
    /// The writer's `land_mutex` while it runs: held around a synchronous mutation so its bytes are its own.
    land: ?*std.Io.Mutex = null,

    pub fn open(allocator: Allocator, io: std.Io, root: []const u8) !Fs {
        const vfs = try Vfs.open(allocator, io, root, .{});
        if (builtin.is_test) testing_layers.append(std.heap.page_allocator, .{ .root = root, .vfs = vfs }) catch {};
        return .{ .allocator = allocator, .io = io, .root = root, .vfs = vfs };
    }

    /// `open` on a copy of `root`, for a caller that holds no tier.
    pub fn openCopy(allocator: Allocator, io: std.Io, root: []const u8) !Fs {
        const own = try allocator.dupe(u8, root);
        errdefer allocator.free(own);
        var self = try open(allocator, io, own);
        self.own_root = true;
        return self;
    }

    pub fn close(self: *Fs) void {
        if (self.borrowed) return;
        if (builtin.is_test) {
            for (testing_layers.items, 0..) |l, i| if (l.vfs == self.vfs) {
                _ = testing_layers.swapRemove(i);
                break;
            };
        }
        self.vfs.close();
        if (self.own_root) self.allocator.free(self.root);
        self.own_root = false;
    }

    /// `path` below the root, without its `\x00`; null for a path outside it.
    fn rel(self: *const Fs, path_in: []const u8) ?[]const u8 {
        const path = std.mem.trimEnd(u8, path_in, "\x00");
        if (!std.mem.startsWith(u8, path, self.root)) return null;
        const rest = path[self.root.len..];
        if (rest.len == 0) return rest;
        if (rest[0] != '/') return null;
        return rest[1..];
    }

    // ── queries ──

    pub fn stat(self: *const Fs, path: []const u8) ?Stat {
        const st = self.vfs.stat(self.rel(path) orelse return null) orelse return null;
        if (st.is_dir) return null;
        return .{ .size = st.size, .nlink = st.nlink, .inode = st.ino, .mtime_ns = st.mtime_ns };
    }

    pub fn size(self: *const Fs, path: []const u8) ?u64 {
        return (self.stat(path) orelse return null).size;
    }

    /// The whole file, or null when it is missing or longer than `limit`.
    pub fn readAlloc(self: *const Fs, allocator: Allocator, path: []const u8, limit: usize) ?[]u8 {
        const r = self.rel(path) orelse return null;
        const st = self.vfs.stat(r) orelse return null;
        if (st.is_dir or st.size > limit) return null;
        return self.vfs.readBytes(allocator, r) catch null;
    }

    /// Does the file begin with `want`?
    pub fn startsWith(self: *const Fs, path: []const u8, want: []const u8) bool {
        const bytes = self.vfs.readBytes(self.allocator, self.rel(path) orelse return false) catch return false;
        defer self.allocator.free(bytes);
        return std.mem.startsWith(u8, bytes, want);
    }

    /// The names directly inside directory `dir`. The caller frees them with `freeEntries`.
    pub fn list(self: *const Fs, allocator: Allocator, dir: []const u8) ![]DirEntry {
        return self.vfs.iterate(allocator, self.rel(dir) orelse return error.FileNotFound) catch |e| switch (e) {
            error.FileNotFound => return &.{},
            else => return e,
        };
    }

    pub fn freeEntries(allocator: Allocator, entries: []DirEntry) void {
        for (entries) |e| allocator.free(e.name);
        allocator.free(entries);
    }

    // ── namespace ──

    /// Physical bytes before a synchronous mutation of the layer; `counted` ends it.
    fn begin(self: *const Fs) u64 {
        if (self.land) |m| m.lockUncancelable(self.io);
        return self.vfs.physicalBytes();
    }

    /// Counts the bytes the mutation since `before` added under `kind`.
    fn counted(self: *const Fs, before: u64, kind: write_stats.Kind) void {
        const n = self.vfs.physicalBytes() -| before;
        if (self.land) |m| m.unlock(self.io);
        if (self.stats) |st| st.recordStored(kind, n);
    }

    pub fn deleteFile(self: *const Fs, path: []const u8) void {
        const r = self.rel(path) orelse return;
        const before = self.begin();
        defer self.counted(before, .meta);
        self.vfs.unlink(r) catch {};
    }

    pub fn hardLink(self: *const Fs, old: []const u8, new: []const u8) !void {
        const old_sub = self.rel(old) orelse return error.BadPathName;
        const new_sub = self.rel(new) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, .meta);
        return self.vfs.link(old_sub, new_sub);
    }

    pub fn deleteTree(self: *const Fs, dir: []const u8) void {
        const sub = self.rel(dir) orelse return;
        const before = self.begin();
        defer self.counted(before, .meta);
        self.vfs.deleteTree(sub) catch {};
    }

    pub fn mkdirp(self: *const Fs, dir: []const u8) !void {
        const r = self.rel(dir) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, .meta);
        return self.vfs.mkdirp(r);
    }

    /// Creates `dir`; `error.PathAlreadyExists` when it exists.
    pub fn mkdirExclusive(self: *const Fs, dir: []const u8) !void {
        const r = self.rel(dir) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, .meta);
        return self.vfs.mkdirExclusive(r);
    }

    // ── bytes ──

    /// Writes `bytes` at byte `off` of `path` (created; truncated first when `truncate`), then sets its length to
    /// `len`: the token record's in-place append.
    pub fn writeAt(self: *const Fs, path: []const u8, truncate: bool, off: u64, bytes: []const u8, len: u64) !void {
        const r = self.rel(path) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, .meta);
        if (truncate or self.vfs.stat(r) == null) {
            try self.vfs.writeBytes(r, .meta, &.{});
        }
        try self.vfs.appendBytes(r, off, bytes);
        try self.vfs.setLength(r, len);
    }

    /// Writes `bytes` to `tmp`, then renames it to `final`.
    pub fn writeRenamed(self: *const Fs, tmp: []const u8, final: []const u8, bytes: []const u8) !void {
        const t = self.rel(tmp) orelse return error.BadPathName;
        const f = self.rel(final) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, .meta);
        try self.vfs.writeBytes(t, .meta, bytes);
        try self.vfs.rename(t, f);
    }

    /// Writes `bytes` over `path` in place, keeping its inode if it exists.
    pub fn writeInPlace(self: *const Fs, path: []const u8, bytes: []const u8, kind: write_stats.Kind) !void {
        const r = self.rel(path) orelse return error.BadPathName;
        const before = self.begin();
        defer self.counted(before, kind);
        try self.vfs.writeBytes(r, kind, bytes);
    }

    // ── tensors ──

    /// Loads a safetensors file directly into Cooler Metal buffers (on Cooler stream) or MLX arrays.
    pub fn loadTensors(self: *const Fs, path: []const u8, s: mlx.mlx_stream) !safetensors.Loaded {
        const r = self.rel(path) orelse return error.MlxError;
        return vt.loadTensors(self.vfs, r, s) catch |e| switch (e) {
            error.OutOfMemory => e,
            else => error.MlxError,
        };
    }

    /// The tensors of the file at `path` (without its `\x00`) read into the bytes `into` names, no array made; its
    /// metadata, in `aa`. A missing, short or unparsable file fails as `loadTensors` fails on it.
    pub fn readTensorsInto(self: *const Fs, path: []const u8, aa: Allocator, into: vt.Sink) ![]const vt.MetaPair {
        return vt.readInto(self.vfs, self.rel(path) orelse return error.MlxError, aa, into);
    }

    /// Writes a tensor file in place on the calling thread. Returns its size. `maps` is empty or parallel to `tensors`.
    pub fn saveTensorsNow(self: *const Fs, path: []const u8, tensors: []const vt.Tensor, meta: []const vt.MetaPair, maps: []const ?vt.RowMap, kind: write_stats.Kind, s: mlx.mlx_stream) !u64 {
        const r = self.rel(path) orelse return error.BadPathName;
        const t = try vt.stageTensors(self.vfs, r, .overwrite, tensors, meta, maps, kind, s);
        const before = self.begin();
        defer self.counted(before, kind);
        return (try self.vfs.publish(t)).size;
    }

    /// Stages a tensor file behind the writer's queue; it lands as a replace. A file past the writer's part size goes
    /// as parts of one ticket, each waiting for room, and only the last publishes; the caller writes no block. Takes
    /// `path`. `maps` is empty or parallel to `tensors`.
    pub fn stageTensors(self: *const Fs, w: *disk_writer.Writer, path: []u8, tensors: []const vt.Tensor, meta: []const vt.MetaPair, maps: []const ?vt.RowMap, kind: write_stats.Kind, s: mlx.mlx_stream) !u64 {
        var path_owned: ?[]u8 = path;
        defer if (path_owned) |p| self.allocator.free(p);
        const r = self.rel(path) orelse return error.BadPathName;
        // The readback stays here (arrays are inference-thread-owned); the writer gets parts of bytes.
        var parts = try vt.Parts.init(self.allocator, tensors, meta, maps, s);
        defer parts.deinit();
        const part_max = w.partBytes();
        var ticket: ?u64 = null;
        errdefer if (ticket) |t| {
            w.fence(path);
            self.vfs.abortTicket(t);
        };
        while (!parts.done()) {
            const n = parts.nextLen(part_max);
            w.waitForRoom(@intCast(n));
            const payload = try parts.take(part_max, kind);
            const first = ticket == null;
            if (ticket) |t| _ = try self.vfs.holdPart(t, payload) else ticket = (try self.vfs.hold(r, .replace, payload)).id;
            const last = parts.done();
            const part: disk_writer.Part = if (first and last) .whole else if (first) .first else if (last) .last else .middle;
            const op_path = if (last) path else try self.allocator.dupe(u8, path);
            if (last) path_owned = null;
            w.submitOpPart(op_path, self.sink(), .{ .ticket = ticket.?, .host_bytes = n, .logical_bytes = n }, part, kind);
        }
        return parts.size;
    }

    /// Stages a small byte file (`meta.json`) behind the writer's queue as one op. Takes `path` and `bytes`, also on
    /// an error.
    pub fn stageBytes(self: *const Fs, w: *disk_writer.Writer, path: []u8, bytes: []u8, kind: write_stats.Kind) !void {
        const a = self.allocator;
        var path_owned: ?[]u8 = path;
        defer if (path_owned) |p| a.free(p);
        const payload = blk: {
            defer a.free(bytes);
            break :blk try ownedBytes(a, bytes, kind);
        };
        const t = try self.vfs.hold(self.rel(path) orelse {
            payload.deinit(a);
            return error.BadPathName;
        }, .replace, payload);
        path_owned = null;
        w.submitOp(path, self.sink(), .{ .ticket = t.id, .host_bytes = t.size, .logical_bytes = t.size }, kind);
    }

    fn sink(self: *const Fs) disk_writer.OpSink {
        return .{ .ctx = self.vfs, .hashBlocks = Vfs.opHashBlocks, .appendBlocks = Vfs.opAppendBlocks, .fillBlocks = Vfs.opFillBlocks, .journalRecord = Vfs.opJournalRecord, .publish = Vfs.opPublish, .abort = Vfs.opAbort };
    }

    // ── accounting ──

    /// The volume's free bytes expressed in logical file bytes: the layer holds `physical` bytes where the files hold
    /// `logical`.
    pub fn freeAsFiles(self: *const Fs, free: u64) u64 {
        return (free +| self.vfs.physicalBytes()) -| self.vfs.logicalBytes();
    }

    /// Gives the volume back the layer's dead bytes.
    pub fn reclaim(self: *const Fs) void {
        _ = self.vfs.reclaim();
    }

    /// The bytes the tier's files hold on the volume: the layer's packs, journal and snapshot, less the snapshot's state
    /// record an empty layer keeps (`prefix_vfs.state_record_bytes`).
    pub fn heldBytes(self: *const Fs) u64 {
        return self.vfs.bytesBeyondEmpty();
    }

    /// Folds the layer's journal into its snapshot, so the records of what was deleted stop holding the volume.
    pub fn compact(self: *const Fs) void {
        self.vfs.compact() catch |err| log.warn("  [prefix-vfs] compacting the journal failed: {s}\n", .{@errorName(err)});
    }
};

fn ownedBytes(a: Allocator, bytes: []const u8, kind: write_stats.Kind) !prefix_vfs.OwnedPayload {
    const views = try prefix_vfs.byteSegments(a, bytes);
    defer a.free(views);
    const segs = try a.alloc(prefix_vfs.Segment, views.len);
    var made: usize = 0;
    errdefer {
        for (segs[0..made]) |sg| a.free(sg.bytes);
        a.free(segs);
    }
    for (views, segs) |src, *dst| {
        dst.* = .{ .bytes = try a.dupe(u8, src.bytes) };
        made += 1;
    }
    return .{ .kind = kind, .segments = segs };
}

pub fn statFile(io: std.Io, abs_path: []const u8) ?std.Io.File.Stat {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    return f.stat(io) catch null;
}

pub fn readFileAlloc(allocator: Allocator, io: std.Io, abs_path: []const u8, limit: usize) ?[]u8 {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    const f = std.Io.Dir.openFileAbsolute(io, abs_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

comptime {
    // The token record is written as its in-memory bytes.
    std.debug.assert(builtin.cpu.arch.endian() == .little);
}

// ── Test helpers: a tier's files by absolute path, on its CAS layer ──

const Layer = struct { root: []const u8, vfs: *Vfs };
/// Test builds: every open layer, so a test reads a live tier's files through the layer that holds them.
var testing_layers: std.ArrayList(Layer) = .empty;

/// The files of the tier root holding `path` (`<root>/e<id>/...`, or `path` itself when it names no entry): the live
/// tier's layer, or one opened for the call. `close` it.
pub fn testFiles(io: std.Io, path: []const u8) !Fs {
    for (testing_layers.items) |l| {
        if (std.mem.startsWith(u8, path, l.root) and (path.len == l.root.len or path[l.root.len] == '/'))
            return .{ .allocator = std.testing.allocator, .io = io, .root = l.root, .vfs = l.vfs, .borrowed = true };
    }
    const root = entryRoot(path);
    try io_util.makeDirPath(io, root);
    return Fs.openCopy(std.testing.allocator, io, root);
}

/// `path` cut before its `/e<digits>` component.
fn entryRoot(path: []const u8) []const u8 {
    var end = path.len;
    while (std.mem.lastIndexOfScalar(u8, path[0..end], '/')) |slash| : (end = slash) {
        const comp = path[slash + 1 .. end];
        if (comp.len < 2 or comp[0] != 'e') continue;
        for (comp[1..]) |c| {
            if (!std.ascii.isDigit(c)) break;
        } else return path[0..slash];
    }
    return path;
}

pub fn testStat(io: std.Io, path: []const u8) ?Stat {
    var f = testFiles(io, path) catch return null;
    defer f.close();
    return f.stat(path);
}

pub fn testSize(io: std.Io, path: []const u8) ?u64 {
    return (testStat(io, path) orelse return null).size;
}

/// The file's bytes: a tensor file of the layer is regenerated from its blocks.
pub fn testRead(io: std.Io, path: []const u8, limit: usize) ?[]u8 {
    var f = testFiles(io, path) catch return null;
    defer f.close();
    if (!std.mem.endsWith(u8, path, ".safetensors")) return f.readAlloc(std.testing.allocator, path, limit);
    const r = f.rel(path) orelse return null;
    const st = f.vfs.stat(r) orelse return null;
    if (st.is_dir or st.size > limit) return null;
    // A file a test wrote as plain bytes is its own image.
    return vt.regenerate(f.vfs, std.testing.allocator, r) catch f.readAlloc(std.testing.allocator, path, limit);
}

/// Writes `bytes` as the whole of `path`, creating its directory.
pub fn testWrite(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var f = try testFiles(io, path);
    defer f.close();
    try f.mkdirp(std.fs.path.dirname(path).?);
    try f.writeInPlace(path, bytes, .meta);
}

pub fn testDelete(io: std.Io, path: []const u8) void {
    var f = testFiles(io, path) catch return;
    defer f.close();
    f.deleteFile(path);
}

pub fn testMkdir(io: std.Io, path: []const u8) !void {
    var f = try testFiles(io, path);
    defer f.close();
    try f.mkdirp(path);
}

/// Flips one byte inside the first tensor block of file `path` on the content-addressed layer: rot that only the
/// block's hash can tell.
pub fn testFlipTensorByte(fs: *const Fs, path: []const u8) !void {
    const pf = try fs.vfs.pinFile(fs.allocator, fs.rel(path) orelse return error.FileNotFound);
    defer fs.vfs.unpinFile(fs.allocator, pf, &.{});
    for (pf.placed) |pl| {
        if (pl.tensor == prefix_vfs.UNTAGGED) continue;
        return fs.vfs.testFlipPlacedByte(pl);
    }
    return error.FileNotFound;
}

fn joined(buf: []u8, base: []const u8, sub: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ base, sub }) catch @panic("test path too long");
}

pub fn testStatAt(io: std.Io, base: []const u8, sub: []const u8) ?Stat {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    return testStat(io, joined(&buf, base, sub));
}

pub fn testReadAt(io: std.Io, base: []const u8, sub: []const u8, limit: usize) ?[]u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    return testRead(io, joined(&buf, base, sub), limit);
}

pub fn testWriteAt(io: std.Io, base: []const u8, sub: []const u8, bytes: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    return testWrite(io, joined(&buf, base, sub), bytes);
}

pub fn testDeleteAt(io: std.Io, base: []const u8, sub: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    testDelete(io, joined(&buf, base, sub));
}

pub fn testDeleteTreeAt(io: std.Io, base: []const u8, sub: []const u8) void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = joined(&buf, base, sub);
    var f = testFiles(io, path) catch return;
    defer f.close();
    f.deleteTree(path);
}

pub fn testMkdirAt(io: std.Io, base: []const u8, sub: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    return testMkdir(io, joined(&buf, base, sub));
}

/// The names inside directory `dir`; free with `Fs.freeEntries(std.testing.allocator, ...)`.
pub fn testList(io: std.Io, dir: []const u8) ![]DirEntry {
    var f = try testFiles(io, dir);
    defer f.close();
    return f.list(std.testing.allocator, dir);
}

// ── Tests ──

const testing = std.testing;

test "kv_disk_fs: a tensor file reads into the bytes its reader names on the layer, from a file and from its bytes, its metadata seen first and a tensor passed over left unread" {
    try mlx.requireGpu();
    const io = testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var cas = try Fs.open(testing.allocator, io, root);
    defer cas.close();
    var dir_buf: [600]u8 = undefined;
    try cas.mkdirp(try std.fmt.bufPrint(&dir_buf, "{s}/e1", .{root}));
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_arange(&a, 0, 12, 1, .float32, s));
    var a2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a2);
    try mlx.check(mlx.mlx_reshape(&a2, a, &[_]c_int{ 4, 3 }, 2, s));
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_arange(&b, 0, 5, 1, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(a2));
    try mlx.check(mlx.mlx_array_eval(b));
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/e1/s0000012.safetensors", .{root});
    defer testing.allocator.free(path);
    _ = try cas.saveTensorsNow(path, &.{ .{ .key = "a", .arr = a2 }, .{ .key = "b", .arr = b } }, &.{.{ .key = "layers", .value = "2" }}, &.{}, .ssm, s);
    const image = testRead(io, path, 1 << 20).?;
    defer testing.allocator.free(image);
    var plain_tmp = std.testing.tmpDir(.{});
    defer plain_tmp.cleanup();
    var plain_buf: [512]u8 = undefined;
    const plain_root = plain_buf[0..try plain_tmp.dir.realPath(io, &plain_buf)];
    const plain_path = try std.fmt.allocPrint(testing.allocator, "{s}/s0000012.safetensors", .{plain_root});
    defer testing.allocator.free(plain_path);
    {
        const f = try std.Io.Dir.createFileAbsolute(io, plain_path, .{});
        defer f.close(io);
        try f.writePositionalAll(io, image, 0);
    }
    const Into = struct {
        out: [12]f32 = @splat(-1),
        /// The metadata lives only for the read: its `layers` value is kept as read.
        layers: ?u8 = null,
        named_before_meta: bool = false,

        fn meta(ctx: *anyopaque, pairs: []const vt.MetaPair) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            for (pairs) |p| if (std.mem.eql(u8, p.key, "layers") and p.value.len == 1) {
                self.layers = p.value[0];
            };
        }

        fn dest(ctx: *anyopaque, t: vt.Named) anyerror!?[]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (self.layers == null) self.named_before_meta = true;
            if (!std.mem.eql(u8, t.key, "a")) return null;
            try testing.expectEqual(mlx.mlx_dtype.float32, t.dtype);
            try testing.expectEqualSlices(c_int, &.{ 4, 3 }, t.shape);
            return std.mem.sliceAsBytes(&self.out);
        }
    };
    var want: [12]f32 = undefined;
    for (&want, 0..) |*x, i| x.* = @floatFromInt(i);
    for (0..3) |how| {
        var arena = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena.deinit();
        var into: Into = .{};
        const sink: vt.Sink = .{ .ctx = &into, .dest = Into.dest, .meta = Into.meta };
        _ = switch (how) {
            0 => try cas.readTensorsInto(path, arena.allocator(), sink),
            1 => try safetensors.readFileInto(io, plain_path, arena.allocator(), sink),
            else => try vt.readBytesInto(image, arena.allocator(), sink),
        };
        try testing.expect(!into.named_before_meta);
        try testing.expectEqual(@as(?u8, '2'), into.layers);
        try testing.expectEqualSlices(f32, &want, &into.out);
    }
}

test "kv_disk_fs: a file larger than half the writer's permit goes to the writer in parts, the caller appending nothing" {
    try mlx.requireGpu();
    const io = testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var fs = try Fs.open(testing.allocator, io, root);
    defer fs.close();
    var w = disk_writer.Writer.init(testing.allocator, io);
    // Parts of 256 KiB: the 400 KB file is two, and both fit the permit while the writer is held.
    w.permit_bytes = 512 * 1024;
    try w.start();
    defer w.deinit();
    var dir_buf: [600]u8 = undefined;
    try fs.mkdirp(try std.fmt.bufPrint(&dir_buf, "{s}/e1", .{root}));

    var arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(arr);
    try mlx.check(mlx.mlx_arange(&arr, 0, 100_000, 1, .float32, s));
    try mlx.check(mlx.mlx_array_eval(arr));
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/e1/s0000128.safetensors", .{root});
    var probe_buf: [600]u8 = undefined;
    const probe = try std.fmt.bufPrint(&probe_buf, "{s}", .{path});

    w.setPaused(true);
    const phys0 = fs.vfs.physicalBytes();
    const n = try fs.stageTensors(&w, path, &.{.{ .key = "a", .arr = arr }}, &.{}, &.{}, .ssm, s);
    try testing.expectEqual(phys0, fs.vfs.physicalBytes());
    try testing.expect(fs.stat(probe) == null);
    try testing.expect(w.isPending(probe));
    w.setPaused(false);
    w.drain();
    try testing.expect(w.peakBytes() <= 512 * 1024);
    try testing.expectEqual(n, fs.size(probe).?);
    const image = testRead(io, probe, 1 << 20).?;
    defer testing.allocator.free(image);
    const data: [*]const u8 = @ptrCast(mlx.mlx_array_data_float32(arr).?);
    try testing.expectEqualSlices(u8, data[0 .. 100_000 * 4], image[image.len - 100_000 * 4 ..]);
}

test "kv_disk_fs: while the writer hashes or writes a file's blocks, the inference thread reads and writes the layer without waiting" {
    try landsBesideInference(.hash);
    try landsBesideInference(.pack_write);
}

/// The writer held at step `at` of landing a staged file while another thread reads `tokens.bin` and appends to it
/// synchronously, as the inference thread does in a persist.
fn landsBesideInference(at: @FieldType(prefix_vfs.Gate, "at")) !void {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var fs = try Fs.open(testing.allocator, io, root);
    defer fs.close();
    var w = disk_writer.Writer.init(testing.allocator, io);
    try w.start();
    defer w.deinit();
    fs.land = &w.land_mutex;
    var dir_buf: [600]u8 = undefined;
    try fs.mkdirp(try std.fmt.bufPrint(&dir_buf, "{s}/e1", .{root}));
    var tok_buf: [600]u8 = undefined;
    const tokens = try std.fmt.bufPrint(&tok_buf, "{s}/e1/tokens.bin", .{root});
    try fs.writeAt(tokens, true, 0, "abcd", 4);

    var gate: prefix_vfs.Gate = .{ .at = at };
    fs.vfs.gate = &gate;
    const chunk = try std.fmt.allocPrint(testing.allocator, "{s}/e1/c000000.safetensors", .{root});
    const bytes = try testing.allocator.alloc(u8, 1 << 20);
    for (bytes, 0..) |*b, i| b.* = @truncate((i *% 2654435761) >> 7);
    var want: [1 << 10]u8 = undefined;
    @memcpy(&want, bytes[0..want.len]);
    try fs.stageBytes(&w, chunk, bytes, .kv);
    var waited: usize = 0;
    while (!gate.entered.load(.acquire) and waited < 5000) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(1), .real) catch {};
    try testing.expect(gate.entered.load(.acquire));

    // The inference thread's side: a read and a synchronous `tokens.bin` append while the writer is mid-append.
    const Inference = struct {
        read: std.atomic.Value(bool) = .init(false),
        wrote: std.atomic.Value(bool) = .init(false),
        fn run(self: *@This(), f: *const Fs, path: []const u8) void {
            if (f.readAlloc(testing.allocator, path, 64)) |got| testing.allocator.free(got);
            self.read.store(true, .release);
            f.writeAt(path, false, 4, "efgh", 8) catch {};
            self.wrote.store(true, .release);
        }
    };
    var inf: Inference = .{};
    const t = try std.Thread.spawn(.{}, Inference.run, .{ &inf, &fs, tokens });
    waited = 0;
    while (!inf.wrote.load(.acquire) and waited < 2000) : (waited += 1) std.Io.sleep(io, .fromMilliseconds(1), .real) catch {};
    const read_free = inf.read.load(.acquire);
    const write_free = inf.wrote.load(.acquire);
    gate.open.store(true, .release);
    t.join();
    w.drain();
    try testing.expect(read_free);
    try testing.expect(write_free);

    const record = fs.readAlloc(testing.allocator, tokens, 64).?;
    defer testing.allocator.free(record);
    try testing.expectEqualStrings("abcdefgh", record);
    var chunk_buf: [600]u8 = undefined;
    const landed = fs.readAlloc(testing.allocator, try std.fmt.bufPrint(&chunk_buf, "{s}/e1/c000000.safetensors", .{root}), 2 << 20).?;
    defer testing.allocator.free(landed);
    try testing.expectEqual(@as(usize, 1 << 20), landed.len);
    try testing.expectEqualSlices(u8, &want, landed[0..want.len]);
}
