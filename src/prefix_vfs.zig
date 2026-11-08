//! Content-addressed virtual file layer for the SSD prefix cache (host tier, MLX-free).
//!
//! A file is a list of extents; an extent names a block by the 128-bit hash of its bytes, and a block
//! lives once in an append-only pack. The namespace answers `stat` (size, nlink, inode, mtime), `link`,
//! `rename` and `iterate` as a POSIX directory does, so a caller that reads those answers sees the
//! filesystem it would have had. Every namespace mutation is one checksummed journal record, written
//! before it is applied; `open` replays the snapshot, then the journal, and drops a torn tail.
//! Nothing is fsynced: a crash of the process loses nothing that was journaled, a power loss may.

const std = @import("std");
const write_stats = @import("disk_write_stats.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;
const File = std.Io.File;

pub const Kind = write_stats.Kind;

/// Tag of a segment that belongs to no tensor or row group.
pub const UNTAGGED: u32 = std.math.maxInt(u32);
/// Block size for bytes the caller gives unstructured (`tokens.bin`, `meta.json`).
pub const BYTE_BLOCK: usize = 16 * 1024;

const MAX_DEPTH = 32;
const MAX_NAME = 255;
const FRAME_HEAD = 12;

/// One block of a file. `tensor` and `group` are the caller's tags and come back from `readPayload`.
pub const Segment = struct {
    bytes: []const u8,
    tensor: u32 = UNTAGGED,
    group: u32 = UNTAGGED,
};

/// A file's bytes as the segments to store, in order. `kind` attributes the new blocks it writes.
pub const Payload = struct { kind: Kind, segments: []const Segment };

pub const OwnedPayload = struct {
    kind: Kind,
    segments: []Segment,

    pub fn deinit(self: OwnedPayload, allocator: Allocator) void {
        for (self.segments) |s| allocator.free(s.bytes);
        allocator.free(self.segments);
    }
};

pub const Options = struct {
    pack_limit: u64 = 1 << 30,
    /// The journal compacts into the snapshot once it exceeds this and twice the snapshot.
    compact_min_bytes: u64 = 4 << 20,
};

pub const Stat = struct { size: u64, nlink: u32, ino: u64, mtime_ns: i128, is_dir: bool };
pub const DirEntry = struct { name: []u8, is_dir: bool };
/// A staged file: `data_bytes` are the new block bytes already in the pack, `size` the file's logical size.
pub const Ticket = struct { id: u64, data_bytes: u64, size: u64 };
pub const Landed = struct { size: u64, data_bytes: u64, journal_bytes: u64 };
pub const KindStats = struct { data_bytes: u64 = 0, blocks: u64 = 0, dedup_blocks: u64 = 0, dedup_bytes: u64 = 0 };
pub const Recovery = struct { journal_torn_bytes: u64 = 0, pack_tail_bytes: u64 = 0, snapshot_dropped: bool = false };

const Addr = struct { xxh: u64, wy: u64 };
const Key = struct { addr: Addr, len: u32 };
const Ext = struct {
    addr: Addr,
    len: u32,
    tensor: u32,
    group: u32,

    fn key(e: Ext) Key {
        return .{ .addr = e.addr, .len = e.len };
    }
    fn eql(a: Ext, b: Ext) bool {
        return a.addr.xxh == b.addr.xxh and a.addr.wy == b.addr.wy and a.len == b.len and a.tensor == b.tensor and a.group == b.group;
    }
};
const Block = struct { pack: u32, off: u32, kind: Kind, refs: u32, verified: bool };
const Ref = struct { ext: Ext, pack: u32, off: u32, kind: Kind };
const Pack = struct { file: ?File = null, size: u64 = 0, live_blocks: u32 = 0, live_bytes: u64 = 0 };

const Vnode = struct {
    ino: u64,
    nlink: u32,
    mtime: i128,
    kind: Kind,
    size: u64,
    exts: std.ArrayList(Ext) = .empty,
};
const DirNode = struct {
    ino: u64,
    children: std.StringHashMapUnmanaged(Node) = .empty,
};
const Node = union(enum) {
    dir: *DirNode,
    file: *Vnode,

    fn same(a: Node, b: Node) bool {
        return switch (a) {
            .dir => |d| b == .dir and b.dir == d,
            .file => |f| b == .file and b.file == f,
        };
    }
};

const Mode = enum { create, replace, overwrite, edit };
pub const HoldMode = enum { create, replace };
/// A staged file. `held` is the payload until `appendBlocks` interns it; `exts` are pinned from then on.
const Staged = struct {
    path: []u8,
    mode: Mode,
    kind: Kind,
    held: ?OwnedPayload = null,
    exts: []Ext = &.{},
    data_bytes: u64 = 0,
};

/// A journaled record waiting for its `publishTicket`; the Vfs lock stays held in between.
const Pending = struct { id: u64, arena: std.heap.ArenaAllocator, rec: Rec };

const Comps = struct {
    items: [MAX_DEPTH][]const u8 = undefined,
    n: usize = 0,

    fn slice(self: *const Comps) []const []const u8 {
        return self.items[0..self.n];
    }
};

fn split(path: []const u8) !Comps {
    var c: Comps = .{};
    var it = std.mem.tokenizeScalar(u8, path, '/');
    while (it.next()) |name| {
        if (name.len > MAX_NAME or c.n == MAX_DEPTH) return error.InvalidPath;
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.InvalidPath;
        if (std.mem.indexOfScalar(u8, name, 0) != null) return error.InvalidPath;
        c.items[c.n] = name;
        c.n += 1;
    }
    return c;
}

fn joinPath(a: Allocator, comps: []const []const u8) ![]u8 {
    return std.mem.join(a, "/", comps);
}

fn addrOf(bytes: []const u8) Addr {
    return .{ .xxh = std.hash.XxHash3.hash(0, bytes), .wy = std.hash.Wyhash.hash(0, bytes) };
}

/// `bytes` cut into BYTE_BLOCK segments that point into it; the caller frees the list.
pub fn byteSegments(allocator: Allocator, bytes: []const u8) ![]Segment {
    const n = (bytes.len + BYTE_BLOCK - 1) / BYTE_BLOCK;
    const segs = try allocator.alloc(Segment, n);
    for (segs, 0..) |*s, i| s.* = .{ .bytes = bytes[i * BYTE_BLOCK ..][0..@min(BYTE_BLOCK, bytes.len - i * BYTE_BLOCK)] };
    return segs;
}

// ---------------------------------------------------------------- journal records

const Rec = union(enum) {
    mkdir: struct { path: []const u8, ino: u64 },
    put: struct { path: []const u8, ino: u64, mtime: i64, kind: Kind, keep: u32, refs: []const Ref },
    edit: struct { path: []const u8, mtime: i64, keep: u32, refs: []const Ref },
    link: struct { from: []const u8, to: []const u8 },
    unlink: []const u8,
    rename: struct { from: []const u8, to: []const u8 },
    deltree: []const u8,
    touch: struct { path: []const u8, mtime: i64 },
    state: struct { next_ino: u64, last_mtime: i64, last_seq: u64 },
};

const Enc = struct {
    list: std.ArrayList(u8) = .empty,
    a: Allocator,

    fn int(self: *Enc, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try self.list.appendSlice(self.a, &b);
    }
    fn str(self: *Enc, s: []const u8) !void {
        try self.int(u16, @intCast(s.len));
        try self.list.appendSlice(self.a, s);
    }
    fn ref(self: *Enc, r: Ref) !void {
        try self.int(u64, r.ext.addr.xxh);
        try self.int(u64, r.ext.addr.wy);
        try self.int(u32, r.pack);
        try self.int(u32, r.off);
        try self.int(u32, r.ext.len);
        try self.int(u8, @intFromEnum(r.kind));
        try self.int(u32, r.ext.tensor);
        try self.int(u32, r.ext.group);
    }
    fn refs(self: *Enc, keep: u32, rs: []const Ref) !void {
        try self.int(u32, keep);
        try self.int(u32, @intCast(rs.len));
        for (rs) |r| try self.ref(r);
    }
};

const Dec = struct {
    buf: []const u8,
    pos: usize = 0,

    fn int(self: *Dec, comptime T: type) !T {
        if (self.pos + @sizeOf(T) > self.buf.len) return error.VfsCorrupt;
        const v = std.mem.readInt(T, self.buf[self.pos..][0..@sizeOf(T)], .little);
        self.pos += @sizeOf(T);
        return v;
    }
    fn str(self: *Dec) ![]const u8 {
        const n = try self.int(u16);
        if (self.pos + n > self.buf.len) return error.VfsCorrupt;
        defer self.pos += n;
        return self.buf[self.pos..][0..n];
    }
    fn kind(self: *Dec) !Kind {
        return std.enums.fromInt(Kind, try self.int(u8)) orelse error.VfsCorrupt;
    }
    fn ref(self: *Dec) !Ref {
        const xxh = try self.int(u64);
        const wy = try self.int(u64);
        const pack = try self.int(u32);
        const off = try self.int(u32);
        const len = try self.int(u32);
        const k = try self.kind();
        const tensor = try self.int(u32);
        const group = try self.int(u32);
        return .{ .ext = .{ .addr = .{ .xxh = xxh, .wy = wy }, .len = len, .tensor = tensor, .group = group }, .pack = pack, .off = off, .kind = k };
    }
    fn refs(self: *Dec, a: Allocator) !struct { keep: u32, refs: []Ref } {
        const keep = try self.int(u32);
        const n = try self.int(u32);
        if (n > (self.buf.len - self.pos) / 37) return error.VfsCorrupt;
        const out = try a.alloc(Ref, n);
        for (out) |*r| r.* = try self.ref();
        return .{ .keep = keep, .refs = out };
    }
};

const Tag = enum(u8) { mkdir = 1, put, edit, link, unlink, rename, deltree, touch, state };

fn encodeRec(e: *Enc, seq: u64, rec: Rec) !void {
    const start = e.list.items.len;
    try e.list.appendNTimes(e.a, 0, FRAME_HEAD);
    const tag: Tag = switch (rec) {
        inline else => |_, t| @field(Tag, @tagName(t)),
    };
    try e.int(u8, @intFromEnum(tag));
    try e.int(u64, seq);
    switch (rec) {
        .mkdir => |m| {
            try e.str(m.path);
            try e.int(u64, m.ino);
        },
        .put => |p| {
            try e.str(p.path);
            try e.int(u64, p.ino);
            try e.int(i64, p.mtime);
            try e.int(u8, @intFromEnum(p.kind));
            try e.refs(p.keep, p.refs);
        },
        .edit => |p| {
            try e.str(p.path);
            try e.int(i64, p.mtime);
            try e.refs(p.keep, p.refs);
        },
        .link => |l| {
            try e.str(l.from);
            try e.str(l.to);
        },
        .unlink => |p| try e.str(p),
        .rename => |r| {
            try e.str(r.from);
            try e.str(r.to);
        },
        .deltree => |p| try e.str(p),
        .touch => |t| {
            try e.str(t.path);
            try e.int(i64, t.mtime);
        },
        .state => |s| {
            try e.int(u64, s.next_ino);
            try e.int(i64, s.last_mtime);
            try e.int(u64, s.last_seq);
        },
    }
    const payload = e.list.items[start + FRAME_HEAD ..];
    std.mem.writeInt(u32, e.list.items[start..][0..4], @intCast(payload.len), .little);
    std.mem.writeInt(u64, e.list.items[start + 4 ..][0..8], std.hash.XxHash3.hash(0, payload), .little);
}

const Decoded = struct { seq: u64, rec: Rec };

fn decodeRec(a: Allocator, payload: []const u8) !Decoded {
    var d: Dec = .{ .buf = payload };
    const tag = std.enums.fromInt(Tag, try d.int(u8)) orelse return error.VfsCorrupt;
    const seq = try d.int(u64);
    const rec: Rec = switch (tag) {
        .mkdir => .{ .mkdir = .{ .path = try d.str(), .ino = try d.int(u64) } },
        .put => blk: {
            const path = try d.str();
            const ino = try d.int(u64);
            const mtime = try d.int(i64);
            const kind = try d.kind();
            const r = try d.refs(a);
            break :blk .{ .put = .{ .path = path, .ino = ino, .mtime = mtime, .kind = kind, .keep = r.keep, .refs = r.refs } };
        },
        .edit => blk: {
            const path = try d.str();
            const mtime = try d.int(i64);
            const r = try d.refs(a);
            break :blk .{ .edit = .{ .path = path, .mtime = mtime, .keep = r.keep, .refs = r.refs } };
        },
        .link => .{ .link = .{ .from = try d.str(), .to = try d.str() } },
        .unlink => .{ .unlink = try d.str() },
        .rename => .{ .rename = .{ .from = try d.str(), .to = try d.str() } },
        .deltree => .{ .deltree = try d.str() },
        .touch => .{ .touch = .{ .path = try d.str(), .mtime = try d.int(i64) } },
        .state => .{ .state = .{ .next_ino = try d.int(u64), .last_mtime = try d.int(i64), .last_seq = try d.int(u64) } },
    };
    if (d.pos != payload.len) return error.VfsCorrupt;
    return .{ .seq = seq, .rec = rec };
}

// ---------------------------------------------------------------- the VFS

pub const Vfs = struct {
    allocator: Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    opts: Options,
    vdir: Dir,
    root: *DirNode,

    blocks: std.AutoHashMapUnmanaged(Key, Block) = .empty,
    packs: std.AutoHashMapUnmanaged(u32, Pack) = .empty,
    active: ?u32 = null,
    next_pack: u32 = 0,

    journal: File,
    journal_len: u64 = 0,
    snapshot_len: u64 = 0,
    seq: u64 = 0,
    next_ino: u64 = 2,
    last_mtime: i128 = 0,
    logical_total: u64 = 0,

    tickets: std.AutoHashMapUnmanaged(u64, Staged) = .empty,
    next_ticket: u64 = 1,
    kind_stats: [write_stats.kind_count]KindStats = @splat(.{}),

    pending: ?Pending = null,
    replaying: bool = false,
    seen_end: std.AutoHashMapUnmanaged(u32, u64) = .empty,
    scratch: std.ArrayList(u8) = .empty,

    recovered: Recovery = .{},
    /// Blocks hashed on a first read; a block written by this process is never counted.
    verify_count: u64 = 0,

    pub fn open(allocator: Allocator, io: std.Io, root: []const u8, opts: Options) !*Vfs {
        const packs_path = try std.fmt.allocPrint(allocator, "{s}/v/packs", .{root});
        defer allocator.free(packs_path);
        try Dir.cwd().createDirPath(io, packs_path);
        const v_path = try std.fmt.allocPrint(allocator, "{s}/v", .{root});
        defer allocator.free(v_path);
        var vdir = try Dir.openDirAbsolute(io, v_path, .{ .iterate = true });
        errdefer vdir.close(io);
        const journal = try vdir.createFile(io, "journal", .{ .read = true, .truncate = false });
        errdefer journal.close(io);

        const root_node = try allocator.create(DirNode);
        root_node.* = .{ .ino = 1 };
        const self = allocator.create(Vfs) catch |e| {
            allocator.destroy(root_node);
            return e;
        };
        self.* = .{ .allocator = allocator, .io = io, .opts = opts, .vdir = vdir, .root = root_node, .journal = journal };
        self.recover() catch |e| {
            self.freeState();
            allocator.destroy(self);
            return e;
        };
        return self;
    }

    pub fn close(self: *Vfs) void {
        const allocator = self.allocator;
        self.freeState();
        self.journal.close(self.io);
        self.vdir.close(self.io);
        allocator.destroy(self);
    }

    fn freeState(self: *Vfs) void {
        const a = self.allocator;
        self.removeNode(.{ .dir = self.root }, false);
        self.blocks.deinit(a);
        var pit = self.packs.valueIterator();
        while (pit.next()) |p| if (p.file) |f| f.close(self.io);
        self.packs.deinit(a);
        var tit = self.tickets.valueIterator();
        while (tit.next()) |t| {
            a.free(t.path);
            a.free(t.exts);
            if (t.held) |h| h.deinit(a);
        }
        self.tickets.deinit(a);
        self.seen_end.deinit(a);
        self.scratch.deinit(a);
    }

    fn lock(self: *Vfs) void {
        self.mutex.lockUncancelable(self.io);
    }
    fn unlock(self: *Vfs) void {
        self.mutex.unlock(self.io);
    }

    // ------------------------------------------------------------ public: files

    pub fn create(self: *Vfs, path: []const u8, p: Payload) !u64 {
        return (try self.publish(try self.stageCreate(path, p))).size;
    }

    pub fn replace(self: *Vfs, path: []const u8, p: Payload) !u64 {
        return (try self.publish(try self.stageReplace(path, p))).size;
    }

    pub fn stageCreate(self: *Vfs, path: []const u8, p: Payload) !Ticket {
        return self.stageImpl(path, .create, p);
    }

    pub fn stageReplace(self: *Vfs, path: []const u8, p: Payload) !Ticket {
        return self.stageImpl(path, .replace, p);
    }

    /// Appends the blocks, journals the record and makes the file visible, in one call. On an error the
    /// ticket is released.
    pub fn publish(self: *Vfs, t: Ticket) !Landed {
        const appended = self.appendBlocks(t.id) catch |e| {
            self.abortTicket(t.id);
            return e;
        };
        const journal_bytes = self.journalRecord(t.id) catch |e| {
            self.abortTicket(t.id);
            return e;
        };
        self.publishTicket(t.id);
        return .{ .size = t.size, .data_bytes = t.data_bytes + appended, .journal_bytes = journal_bytes };
    }

    /// Drops a staged file; its new blocks stay in the pack as dead bytes.
    pub fn abandon(self: *Vfs, t: Ticket) void {
        self.abortTicket(t.id);
    }

    // The writer's four steps. A ticket is `hold` (or `stage*`), then `appendBlocks`, `journalRecord`, and
    // `publishTicket`, or `abortTicket` at any point before the last. `journalRecord` returns with the Vfs lock
    // held and `publishTicket` releases it, so no other mutation is journaled between the two and the
    // record's view of the namespace is the one it is applied to. The caller sequences the four on one thread.

    /// Stages a file whose bytes the Vfs now owns (allocated with its allocator), also on an error. Nothing is
    /// hashed or written yet.
    pub fn hold(self: *Vfs, path: []const u8, mode: HoldMode, payload: OwnedPayload) !Ticket {
        errdefer payload.deinit(self.allocator);
        const c = try split(path);
        if (c.n == 0) return error.InvalidPath;
        const a = self.allocator;
        const owned = try a.dupe(u8, path);
        errdefer a.free(owned);
        var size: u64 = 0;
        for (payload.segments) |sg| size += sg.bytes.len;
        self.lock();
        defer self.unlock();
        const id = self.next_ticket;
        try self.tickets.put(a, id, .{ .path = owned, .mode = if (mode == .create) .create else .replace, .kind = payload.kind, .held = payload });
        self.next_ticket += 1;
        return .{ .id = id, .data_bytes = 0, .size = size };
    }

    /// Hashes the held payload and appends the blocks the index lacks to the active pack. Returns the pack
    /// bytes appended. The file stays invisible: the blocks are pinned, not yet part of any name.
    pub fn appendBlocks(self: *Vfs, id: u64) !u64 {
        const a = self.allocator;
        self.lock();
        const held = (self.tickets.getPtr(id) orelse {
            self.unlock();
            return error.UnknownTicket;
        }).held;
        self.unlock();
        const payload = held orelse return 0;
        const addrs = try a.alloc(Addr, payload.segments.len);
        defer a.free(addrs);
        for (payload.segments, addrs) |sg, *ad| ad.* = addrOf(sg.bytes);

        self.lock();
        defer self.unlock();
        const st = self.tickets.getPtr(id) orelse return error.UnknownTicket;
        const fresh = try self.internLocked(payload.kind, payload.segments, addrs);
        payload.deinit(a);
        st.held = null;
        st.exts = fresh.exts;
        st.data_bytes = fresh.data_bytes;
        return fresh.data_bytes;
    }

    /// Appends the ticket's journal record and returns its bytes. On success the Vfs lock is still held until
    /// `publishTicket`; on an error it is released and the ticket stays for `abortTicket`.
    pub fn journalRecord(self: *Vfs, id: u64) !u64 {
        self.lock();
        errdefer self.unlock();
        const st = self.tickets.getPtr(id) orelse return error.UnknownTicket;
        if (st.held != null) return error.NotAppended;
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();
        const rec = try self.buildRec(arena.allocator(), st.path, st.mode, st.kind, st.exts);
        const n = try self.journalAppend(rec);
        self.pending = .{ .id = id, .arena = arena, .rec = rec };
        return n;
    }

    /// Applies the journaled record: the file is visible to `stat` from here. Releases the lock
    /// `journalRecord` took.
    pub fn publishTicket(self: *Vfs, id: u64) void {
        const p = self.pending.?;
        std.debug.assert(p.id == id);
        self.pending = null;
        var arena = p.arena;
        defer arena.deinit();
        const st = self.tickets.fetchRemove(id).?.value;
        self.settle(p.rec) catch |e| log.warn("  [prefix-vfs] a journaled record could not be applied: {s}\n", .{@errorName(e)});
        self.releaseAll(st.exts);
        self.allocator.free(st.path);
        self.allocator.free(st.exts);
        self.unlock();
    }

    /// Releases a ticket that never landed; blocks already appended stay in the pack as dead bytes. Never
    /// calls out of the Vfs, so it is safe under a caller's own lock.
    pub fn abortTicket(self: *Vfs, id: u64) void {
        self.lock();
        defer self.unlock();
        const kv = self.tickets.fetchRemove(id) orelse return;
        self.releaseAll(kv.value.exts);
        self.allocator.free(kv.value.path);
        self.allocator.free(kv.value.exts);
        if (kv.value.held) |h| h.deinit(self.allocator);
    }

    /// The four steps as plain functions over `ctx: *Vfs`, the shape the writer's op sink takes.
    pub fn opAppendBlocks(ctx: *anyopaque, ticket: u64) anyerror!u64 {
        return @as(*Vfs, @ptrCast(@alignCast(ctx))).appendBlocks(ticket);
    }
    pub fn opJournalRecord(ctx: *anyopaque, ticket: u64) anyerror!u64 {
        return @as(*Vfs, @ptrCast(@alignCast(ctx))).journalRecord(ticket);
    }
    pub fn opPublish(ctx: *anyopaque, ticket: u64) void {
        @as(*Vfs, @ptrCast(@alignCast(ctx))).publishTicket(ticket);
    }
    pub fn opAbort(ctx: *anyopaque, ticket: u64) void {
        @as(*Vfs, @ptrCast(@alignCast(ctx))).abortTicket(ticket);
    }

    /// `open(O_CREAT | O_TRUNC)` and write: the inode survives an existing file.
    pub fn writeBytes(self: *Vfs, path: []const u8, kind: Kind, bytes: []const u8) !void {
        const segs = try byteSegments(self.allocator, bytes);
        defer self.allocator.free(segs);
        _ = try self.publish(try self.stageImpl(path, .overwrite, .{ .kind = kind, .segments = segs }));
    }

    /// `pwrite` at `off`; a gap past the end reads as zeros.
    pub fn appendBytes(self: *Vfs, path: []const u8, off: u64, bytes: []const u8) !void {
        self.lock();
        defer self.unlock();
        const vn = try self.fileAt(path);
        if (bytes.len == 0) return;
        try self.splice(path, vn, off, bytes);
    }

    pub fn setLength(self: *Vfs, path: []const u8, len: u64) !void {
        self.lock();
        defer self.unlock();
        const a = self.allocator;
        const vn = try self.fileAt(path);
        if (len == vn.size) return;
        if (len > vn.size) {
            const zeros = try a.alloc(u8, len - vn.size);
            defer a.free(zeros);
            @memset(zeros, 0);
            return self.splice(path, vn, vn.size, zeros);
        }
        const exts = vn.exts.items;
        var start: u64 = 0;
        var i: usize = 0;
        while (i < exts.len and start + exts[i].len <= len) : (i += 1) start += exts[i].len;
        var fresh: Interned = .{ .exts = &.{}, .data_bytes = 0 };
        if (len > start) {
            const whole = try a.alloc(u8, exts[i].len);
            defer a.free(whole);
            try self.readBlock(exts[i], whole);
            fresh = try self.internBytes(vn.kind, whole[0..@intCast(len - start)]);
        }
        defer a.free(fresh.exts);
        const list = try self.joinPinned(exts[0..i], fresh.exts, &.{});
        defer a.free(list);
        defer self.releaseAll(list);
        _ = try self.land(path, .edit, vn.kind, list, fresh.data_bytes);
    }

    pub fn readBytes(self: *Vfs, allocator: Allocator, path: []const u8) ![]u8 {
        self.lock();
        defer self.unlock();
        const vn = try self.fileAt(path);
        const out = try allocator.alloc(u8, vn.size);
        errdefer allocator.free(out);
        var off: usize = 0;
        for (vn.exts.items) |e| {
            try self.readBlock(e, out[off..][0..e.len]);
            off += e.len;
        }
        return out;
    }

    pub fn readPayload(self: *Vfs, allocator: Allocator, path: []const u8) !OwnedPayload {
        self.lock();
        defer self.unlock();
        const vn = try self.fileAt(path);
        const segs = try allocator.alloc(Segment, vn.exts.items.len);
        var made: usize = 0;
        errdefer {
            for (segs[0..made]) |s| allocator.free(s.bytes);
            allocator.free(segs);
        }
        for (vn.exts.items, segs) |e, *s| {
            const buf = try allocator.alloc(u8, e.len);
            errdefer allocator.free(buf);
            try self.readBlock(e, buf);
            s.* = .{ .bytes = buf, .tensor = e.tensor, .group = e.group };
            made += 1;
        }
        return .{ .kind = vn.kind, .segments = segs };
    }

    // ------------------------------------------------------------ public: namespace

    pub fn link(self: *Vfs, from: []const u8, to: []const u8) !void {
        self.lock();
        defer self.unlock();
        const cf = try split(from);
        const ct = try split(to);
        const src = try self.find(cf.slice()) orelse return error.FileNotFound;
        if (src == .dir) return error.AccessDenied;
        if (ct.n == 0) return error.PathAlreadyExists;
        const dst = try self.parentOf(ct.slice());
        if (dst.dir.children.contains(dst.name)) return error.PathAlreadyExists;
        const a = self.allocator;
        const pf = try joinPath(a, cf.slice());
        defer a.free(pf);
        const pt = try joinPath(a, ct.slice());
        defer a.free(pt);
        _ = try self.commit(.{ .link = .{ .from = pf, .to = pt } });
    }

    pub fn unlink(self: *Vfs, path: []const u8) !void {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        const node = try self.find(c.slice()) orelse return error.FileNotFound;
        if (node == .dir) return error.IsDir;
        const p = try joinPath(self.allocator, c.slice());
        defer self.allocator.free(p);
        _ = try self.commit(.{ .unlink = p });
    }

    pub fn rename(self: *Vfs, from: []const u8, to: []const u8) !void {
        self.lock();
        defer self.unlock();
        const cf = try split(from);
        const ct = try split(to);
        if (cf.n == 0 or ct.n == 0) return error.InvalidPath;
        const src = try self.find(cf.slice()) orelse return error.FileNotFound;
        const dst_parent = try self.parentOf(ct.slice());
        if (src == .dir and ct.n > cf.n and commonPrefix(cf.slice(), ct.slice())) return error.InvalidPath;
        if (dst_parent.dir.children.get(dst_parent.name)) |dst| {
            if (src.same(dst)) return;
            switch (src) {
                .file => if (dst == .dir) return error.IsDir,
                .dir => switch (dst) {
                    .file => return error.NotDir,
                    .dir => |d| if (d.children.count() != 0) return error.DirNotEmpty,
                },
            }
        }
        const a = self.allocator;
        const pf = try joinPath(a, cf.slice());
        defer a.free(pf);
        const pt = try joinPath(a, ct.slice());
        defer a.free(pt);
        _ = try self.commit(.{ .rename = .{ .from = pf, .to = pt } });
    }

    pub fn mkdirp(self: *Vfs, path: []const u8) !void {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        var cur = self.root;
        for (0..c.n) |i| {
            if (cur.children.get(c.items[i])) |child| {
                switch (child) {
                    .dir => |d| cur = d,
                    .file => return error.NotDir,
                }
                continue;
            }
            const p = try joinPath(self.allocator, c.items[0 .. i + 1]);
            defer self.allocator.free(p);
            const ino = self.next_ino;
            _ = try self.commit(.{ .mkdir = .{ .path = p, .ino = ino } });
            cur = cur.children.get(c.items[i]).?.dir;
        }
    }

    pub fn mkdirExclusive(self: *Vfs, path: []const u8) !void {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        if (c.n == 0) return error.PathAlreadyExists;
        const at = try self.parentOf(c.slice());
        if (at.dir.children.contains(at.name)) return error.PathAlreadyExists;
        const p = try joinPath(self.allocator, c.slice());
        defer self.allocator.free(p);
        _ = try self.commit(.{ .mkdir = .{ .path = p, .ino = self.next_ino } });
    }

    /// Removes a file or a directory with everything below it; a missing path is not an error.
    pub fn deleteTree(self: *Vfs, path: []const u8) !void {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        if (c.n == 0) return error.InvalidPath;
        _ = (try self.find(c.slice())) orelse return;
        const p = try joinPath(self.allocator, c.slice());
        defer self.allocator.free(p);
        _ = try self.commit(.{ .deltree = p });
    }

    pub fn iterate(self: *Vfs, allocator: Allocator, path: []const u8) ![]DirEntry {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        const node = try self.find(c.slice()) orelse return error.FileNotFound;
        const d = switch (node) {
            .dir => |d| d,
            .file => return error.NotDir,
        };
        var out: std.ArrayList(DirEntry) = .empty;
        errdefer {
            for (out.items) |e| allocator.free(e.name);
            out.deinit(allocator);
        }
        var it = d.children.iterator();
        while (it.next()) |kv| {
            const name = try allocator.dupe(u8, kv.key_ptr.*);
            errdefer allocator.free(name);
            try out.append(allocator, .{ .name = name, .is_dir = kv.value_ptr.* == .dir });
        }
        return out.toOwnedSlice(allocator);
    }

    pub fn freeEntries(allocator: Allocator, entries: []DirEntry) void {
        for (entries) |e| allocator.free(e.name);
        allocator.free(entries);
    }

    pub fn stat(self: *Vfs, path: []const u8) ?Stat {
        self.lock();
        defer self.unlock();
        const c = split(path) catch return null;
        const node = (self.find(c.slice()) catch return null) orelse return null;
        return switch (node) {
            .dir => |d| .{ .size = 0, .nlink = 2, .ino = d.ino, .mtime_ns = 0, .is_dir = true },
            .file => |f| .{ .size = f.size, .nlink = f.nlink, .ino = f.ino, .mtime_ns = f.mtime, .is_dir = false },
        };
    }

    pub fn touch(self: *Vfs, path: []const u8, mtime_ns: i128) !void {
        self.lock();
        defer self.unlock();
        const c = try split(path);
        const node = try self.find(c.slice()) orelse return error.FileNotFound;
        if (node == .dir) return;
        const p = try joinPath(self.allocator, c.slice());
        defer self.allocator.free(p);
        _ = try self.commit(.{ .touch = .{ .path = p, .mtime = @intCast(mtime_ns) } });
    }

    /// The mtime the next mutation will take: wall clock, never at or before an earlier mtime.
    pub fn nowNs(self: *Vfs) i128 {
        self.lock();
        defer self.unlock();
        return @max(std.Io.Timestamp.now(self.io, .real).nanoseconds, self.last_mtime + 1);
    }

    // ------------------------------------------------------------ public: accounting

    pub fn physicalBytes(self: *Vfs) u64 {
        self.lock();
        defer self.unlock();
        var n = self.journal_len + self.snapshot_len;
        var it = self.packs.valueIterator();
        while (it.next()) |p| n += p.size;
        return n;
    }

    /// The sum of the sizes of the distinct files, as `tierBytes` counts a directory of hard links.
    pub fn logicalBytes(self: *Vfs) u64 {
        self.lock();
        defer self.unlock();
        return self.logical_total;
    }

    /// Pack bytes no live file references: freed blocks, abandoned staging, failed writes.
    pub fn deadBytes(self: *Vfs) u64 {
        self.lock();
        defer self.unlock();
        var n: u64 = 0;
        var it = self.packs.valueIterator();
        while (it.next()) |p| n += p.size - p.live_bytes;
        return n;
    }

    pub fn stats(self: *Vfs, kind: Kind) KindStats {
        self.lock();
        defer self.unlock();
        return self.kind_stats[@intFromEnum(kind)];
    }

    /// Folds the journal into a fresh snapshot, atomically.
    pub fn compact(self: *Vfs) !void {
        self.lock();
        defer self.unlock();
        try self.compactLocked();
    }

    // ------------------------------------------------------------ staging and blocks

    fn stageImpl(self: *Vfs, path: []const u8, mode: Mode, p: Payload) !Ticket {
        const c = try split(path);
        if (c.n == 0) return error.InvalidPath;
        const a = self.allocator;
        const addrs = try a.alloc(Addr, p.segments.len);
        defer a.free(addrs);
        for (p.segments, addrs) |s, *ad| ad.* = addrOf(s.bytes);

        self.lock();
        defer self.unlock();
        const fresh = try self.internLocked(p.kind, p.segments, addrs);
        errdefer {
            self.releaseAll(fresh.exts);
            a.free(fresh.exts);
        }
        const owned = try a.dupe(u8, path);
        errdefer a.free(owned);
        const id = self.next_ticket;
        try self.tickets.put(a, id, .{ .path = owned, .mode = mode, .kind = p.kind, .exts = fresh.exts, .data_bytes = fresh.data_bytes });
        self.next_ticket += 1;
        var size: u64 = 0;
        for (fresh.exts) |e| size += e.len;
        return .{ .id = id, .data_bytes = fresh.data_bytes, .size = size };
    }

    const Interned = struct { exts: []Ext, data_bytes: u64 };

    /// One pinned extent per non-empty segment: an existing block is shared, a new one is appended to the pack.
    fn internLocked(self: *Vfs, kind: Kind, segs: []const Segment, addrs: []const Addr) !Interned {
        const a = self.allocator;
        var exts: std.ArrayList(Ext) = .empty;
        errdefer {
            self.releaseAll(exts.items);
            exts.deinit(a);
        }
        var data_bytes: u64 = 0;
        const ks = &self.kind_stats[@intFromEnum(kind)];
        for (segs, addrs) |s, ad| {
            if (s.bytes.len == 0) continue;
            if (s.bytes.len > std.math.maxInt(u32)) return error.BlockTooLarge;
            const e: Ext = .{ .addr = ad, .len = @intCast(s.bytes.len), .tensor = s.tensor, .group = s.group };
            try exts.ensureUnusedCapacity(a, 1);
            if (self.blocks.contains(e.key())) {
                ks.dedup_blocks += 1;
                ks.dedup_bytes += e.len;
            } else {
                try self.appendBlock(e, s.bytes, kind);
                ks.blocks += 1;
                ks.data_bytes += e.len;
                data_bytes += e.len;
            }
            self.acquire(e) catch unreachable;
            exts.appendAssumeCapacity(e);
        }
        return .{ .exts = try exts.toOwnedSlice(a), .data_bytes = data_bytes };
    }

    /// `bytes` as 16 KiB blocks, pinned.
    fn internBytes(self: *Vfs, kind: Kind, bytes: []const u8) !Interned {
        const a = self.allocator;
        const segs = try byteSegments(a, bytes);
        defer a.free(segs);
        const addrs = try a.alloc(Addr, segs.len);
        defer a.free(addrs);
        for (segs, addrs) |s, *ad| ad.* = addrOf(s.bytes);
        return self.internLocked(kind, segs, addrs);
    }

    fn appendBlock(self: *Vfs, e: Ext, bytes: []const u8, kind: Kind) !void {
        const a = self.allocator;
        if (self.active) |id| {
            const p = self.packs.getPtr(id).?;
            if (p.size > 0 and p.size + bytes.len > self.opts.pack_limit) self.active = null;
        }
        if (self.active == null) try self.newPack();
        const id = self.active.?;
        const p = self.packs.getPtr(id).?;
        if (p.size + bytes.len > std.math.maxInt(u32)) return error.BlockTooLarge;
        try self.blocks.ensureUnusedCapacity(a, 1);
        try p.file.?.writePositionalAll(self.io, bytes, p.size);
        self.blocks.putAssumeCapacity(e.key(), .{ .pack = id, .off = @intCast(p.size), .kind = kind, .refs = 0, .verified = true });
        p.size += bytes.len;
    }

    fn newPack(self: *Vfs) !void {
        var name_buf: [32]u8 = undefined;
        const name = packName(&name_buf, self.next_pack);
        const f = try self.vdir.createFile(self.io, name, .{ .read = true, .truncate = true });
        errdefer f.close(self.io);
        try self.packs.put(self.allocator, self.next_pack, .{ .file = f });
        self.active = self.next_pack;
        self.next_pack += 1;
    }

    fn packName(buf: []u8, id: u32) []const u8 {
        return std.fmt.bufPrint(buf, "packs/{d}.pack", .{id}) catch unreachable;
    }

    fn acquire(self: *Vfs, e: Ext) !void {
        const b = self.blocks.getPtr(e.key()) orelse return error.VfsCorrupt;
        b.refs += 1;
        if (b.refs == 1) {
            const p = self.packs.getPtr(b.pack).?;
            p.live_blocks += 1;
            p.live_bytes += e.len;
        }
    }

    fn acquireAll(self: *Vfs, exts: []const Ext) void {
        for (exts) |e| self.acquire(e) catch unreachable;
    }

    fn release(self: *Vfs, e: Ext) void {
        const b = self.blocks.getPtr(e.key()).?;
        b.refs -= 1;
        if (b.refs != 0) return;
        const id = b.pack;
        _ = self.blocks.remove(e.key());
        const p = self.packs.getPtr(id).?;
        p.live_blocks -= 1;
        p.live_bytes -= e.len;
        if (p.live_blocks == 0 and !self.replaying) self.dropPack(id);
    }

    fn releaseAll(self: *Vfs, exts: []const Ext) void {
        for (exts) |e| self.release(e);
    }

    fn dropPack(self: *Vfs, id: u32) void {
        const kv = self.packs.fetchRemove(id) orelse return;
        if (kv.value.file) |f| f.close(self.io);
        var name_buf: [32]u8 = undefined;
        self.vdir.deleteFile(self.io, packName(&name_buf, id)) catch {};
        if (self.active == id) self.active = null;
    }

    /// Reads a whole block, verifying its checksum the first time this process reads it.
    fn readBlock(self: *Vfs, e: Ext, dst: []u8) !void {
        const b = self.blocks.getPtr(e.key()) orelse return error.VfsCorrupt;
        const p = self.packs.get(b.pack) orelse return error.VfsCorrupt;
        const f = p.file orelse return error.VfsCorrupt;
        const got = try f.readPositionalAll(self.io, dst, b.off);
        if (got != dst.len) return error.VfsCorrupt;
        if (b.verified) return;
        self.verify_count += 1;
        if (std.hash.XxHash3.hash(0, dst) != e.addr.xxh) return error.VfsCorrupt;
        b.verified = true;
    }

    // ------------------------------------------------------------ namespace lookup

    fn find(self: *Vfs, comps: []const []const u8) !?Node {
        var cur: Node = .{ .dir = self.root };
        for (comps) |name| {
            const d = switch (cur) {
                .dir => |d| d,
                .file => return error.NotDir,
            };
            cur = d.children.get(name) orelse return null;
        }
        return cur;
    }

    const Parent = struct { dir: *DirNode, name: []const u8 };

    fn parentOf(self: *Vfs, comps: []const []const u8) !Parent {
        const p = try self.find(comps[0 .. comps.len - 1]) orelse return error.FileNotFound;
        return switch (p) {
            .dir => |d| .{ .dir = d, .name = comps[comps.len - 1] },
            .file => error.NotDir,
        };
    }

    fn fileAt(self: *Vfs, path: []const u8) !*Vnode {
        const c = try split(path);
        const node = try self.find(c.slice()) orelse return error.FileNotFound;
        return switch (node) {
            .file => |f| f,
            .dir => error.IsDir,
        };
    }

    fn commonPrefix(a: []const []const u8, b: []const []const u8) bool {
        if (b.len < a.len) return false;
        for (a, b[0..a.len]) |x, y| if (!std.mem.eql(u8, x, y)) return false;
        return true;
    }

    // ------------------------------------------------------------ writing a file

    /// Replaces the bytes in `[off, off + bytes.len)`, re-cutting only the blocks that overlap them.
    fn splice(self: *Vfs, path: []const u8, vn: *Vnode, off: u64, bytes: []const u8) !void {
        const a = self.allocator;
        const old = vn.exts.items;
        const from = @min(off, vn.size);
        const end = off + bytes.len;
        var i: usize = 0;
        var start: u64 = 0;
        while (i < old.len and start + old[i].len <= from) : (i += 1) start += old[i].len;
        // A short untagged block at the write point is the tail of a byte file: re-cut it with the new bytes.
        if (i > 0 and start == from and old[i - 1].len < BYTE_BLOCK and old[i - 1].tensor == UNTAGGED and old[i - 1].group == UNTAGGED) {
            i -= 1;
            start -= old[i].len;
        }
        var j = i;
        var region_end = start;
        while (j < old.len and region_end < end and region_end < vn.size) : (j += 1) region_end += old[j].len;
        const new_len = if (end >= vn.size) end - start else region_end - start;
        const buf = try a.alloc(u8, new_len);
        defer a.free(buf);
        @memset(buf, 0);
        var at: usize = 0;
        for (old[i..j]) |e| {
            try self.readBlock(e, buf[at..][0..e.len]);
            at += e.len;
        }
        @memcpy(buf[off - start ..][0..bytes.len], bytes);

        const fresh = try self.internBytes(vn.kind, buf);
        defer a.free(fresh.exts);
        const list = try self.joinPinned(old[0..i], fresh.exts, old[j..]);
        defer a.free(list);
        defer self.releaseAll(list);
        _ = try self.land(path, .edit, vn.kind, list, fresh.data_bytes);
    }

    /// `head ++ fresh ++ tail` as one list in which every extent is pinned; `fresh` already was.
    fn joinPinned(self: *Vfs, head: []const Ext, fresh: []const Ext, tail: []const Ext) ![]Ext {
        const list = self.allocator.alloc(Ext, head.len + fresh.len + tail.len) catch |e| {
            self.releaseAll(fresh);
            return e;
        };
        @memcpy(list[0..head.len], head);
        @memcpy(list[head.len..][0..fresh.len], fresh);
        @memcpy(list[head.len + fresh.len ..], tail);
        self.acquireAll(head);
        self.acquireAll(tail);
        return list;
    }

    /// Journals and applies one file write whose extents are all pinned by the caller.
    fn land(self: *Vfs, path: []const u8, mode: Mode, kind: Kind, exts: []const Ext, data_bytes: u64) !Landed {
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const rec = try self.buildRec(arena.allocator(), path, mode, kind, exts);
        const journal_bytes = try self.commit(rec);
        var size: u64 = 0;
        for (exts) |e| size += e.len;
        return .{ .size = size, .data_bytes = data_bytes, .journal_bytes = journal_bytes };
    }

    /// The record that writes `exts` to `path`, checked against the namespace as it is now. Its strings and
    /// refs are allocated from `a`.
    fn buildRec(self: *Vfs, a: Allocator, path: []const u8, mode: Mode, kind: Kind, exts: []const Ext) !Rec {
        const c = try split(path);
        if (c.n == 0) return error.InvalidPath;
        const at = try self.parentOf(c.slice());
        const existing = at.dir.children.get(at.name);
        var old: ?*Vnode = null;
        if (existing) |node| switch (node) {
            .dir => return if (mode == .create) error.PathAlreadyExists else error.IsDir,
            .file => |f| {
                if (mode == .create) return error.PathAlreadyExists;
                old = f;
            },
        };
        if (mode == .edit and old == null) return error.FileNotFound;

        var keep: u32 = 0;
        if (old) |f| {
            while (keep < f.exts.items.len and keep < exts.len and f.exts.items[keep].eql(exts[keep])) keep += 1;
        }
        const refs = try a.alloc(Ref, exts.len - keep);
        for (exts[keep..], refs) |e, *r| {
            const b = self.blocks.get(e.key()).?;
            r.* = .{ .ext = e, .pack = b.pack, .off = b.off, .kind = b.kind };
        }
        const norm = try joinPath(a, c.slice());
        const mtime: i64 = @intCast(self.tick());
        return if (old != null and mode != .replace)
            .{ .edit = .{ .path = norm, .mtime = mtime, .keep = keep, .refs = refs } }
        else
            .{ .put = .{ .path = norm, .ino = self.next_ino, .mtime = mtime, .kind = kind, .keep = keep, .refs = refs } };
    }

    fn tick(self: *Vfs) i128 {
        const t = @max(std.Io.Timestamp.now(self.io, .real).nanoseconds, self.last_mtime + 1);
        self.last_mtime = t;
        return t;
    }

    // ------------------------------------------------------------ journal

    /// Journals `rec`, applies it, and compacts when the journal has outgrown the snapshot.
    fn commit(self: *Vfs, rec: Rec) !u64 {
        const n = try self.journalAppend(rec);
        try self.settle(rec);
        return n;
    }

    fn journalAppend(self: *Vfs, rec: Rec) !u64 {
        var e: Enc = .{ .a = self.allocator, .list = self.scratch };
        defer {
            self.scratch = e.list;
            self.scratch.clearRetainingCapacity();
        }
        e.list.clearRetainingCapacity();
        try encodeRec(&e, self.seq + 1, rec);
        try self.journal.writePositionalAll(self.io, e.list.items, self.journal_len);
        const n: u64 = e.list.items.len;
        self.journal_len += n;
        self.seq += 1;
        return n;
    }

    /// Applies a journaled record, then folds the journal into the snapshot if it has outgrown it.
    fn settle(self: *Vfs, rec: Rec) !void {
        try self.apply(rec);
        if (self.journal_len >= @max(self.opts.compact_min_bytes, 2 * self.snapshot_len)) {
            // A failed compaction leaves the old snapshot and the journal intact; the next commit retries.
            self.compactLocked() catch {};
        }
    }

    fn apply(self: *Vfs, rec: Rec) !void {
        const a = self.allocator;
        switch (rec) {
            .mkdir => |m| {
                const c = try split(m.path);
                if (c.n == 0) return error.VfsCorrupt;
                const at = try self.parentOf(c.slice());
                if (at.dir.children.contains(at.name)) return error.VfsCorrupt;
                const d = try a.create(DirNode);
                errdefer a.destroy(d);
                d.* = .{ .ino = m.ino };
                const key = try a.dupe(u8, at.name);
                errdefer a.free(key);
                try at.dir.children.put(a, key, .{ .dir = d });
                self.next_ino = @max(self.next_ino, m.ino + 1);
            },
            .put => |p| {
                const c = try split(p.path);
                if (c.n == 0) return error.VfsCorrupt;
                const at = try self.parentOf(c.slice());
                const old: ?*Vnode = if (at.dir.children.get(at.name)) |n| switch (n) {
                    .file => |f| f,
                    .dir => return error.VfsCorrupt,
                } else null;
                const vn = try a.create(Vnode);
                errdefer a.destroy(vn);
                vn.* = .{ .ino = p.ino, .nlink = 1, .mtime = p.mtime, .kind = p.kind, .size = 0 };
                errdefer vn.exts.deinit(a);
                try self.collect(vn, if (old) |f| f.exts.items else &.{}, p.keep, p.refs);
                errdefer self.releaseAll(vn.exts.items);
                if (old) |f| {
                    // The name now points at the new vnode; the old one loses this name.
                    const slot = at.dir.children.getPtr(at.name).?;
                    slot.* = .{ .file = vn };
                    self.removeNode(.{ .file = f }, true);
                } else {
                    const key = try a.dupe(u8, at.name);
                    errdefer a.free(key);
                    try at.dir.children.put(a, key, .{ .file = vn });
                }
                self.logical_total += vn.size;
                self.next_ino = @max(self.next_ino, p.ino + 1);
                self.last_mtime = @max(self.last_mtime, p.mtime);
            },
            .edit => |p| {
                const vn = try self.fileAt(p.path);
                var fresh: Vnode = .{ .ino = vn.ino, .nlink = vn.nlink, .mtime = p.mtime, .kind = vn.kind, .size = 0 };
                errdefer fresh.exts.deinit(a);
                try self.collect(&fresh, vn.exts.items, p.keep, p.refs);
                self.releaseAll(vn.exts.items);
                self.logical_total -= vn.size;
                vn.exts.deinit(a);
                vn.exts = fresh.exts;
                vn.size = fresh.size;
                vn.mtime = p.mtime;
                self.logical_total += vn.size;
                self.last_mtime = @max(self.last_mtime, p.mtime);
            },
            .link => |l| {
                const cf = try split(l.from);
                const ct = try split(l.to);
                const src = (try self.find(cf.slice())) orelse return error.VfsCorrupt;
                if (src != .file or ct.n == 0) return error.VfsCorrupt;
                const at = try self.parentOf(ct.slice());
                if (at.dir.children.contains(at.name)) return error.VfsCorrupt;
                const key = try a.dupe(u8, at.name);
                errdefer a.free(key);
                try at.dir.children.put(a, key, src);
                src.file.nlink += 1;
            },
            .unlink => |p| {
                const c = try split(p);
                if (c.n == 0) return error.VfsCorrupt;
                const at = try self.parentOf(c.slice());
                const kv = at.dir.children.fetchRemove(at.name) orelse return error.VfsCorrupt;
                a.free(kv.key);
                if (kv.value != .file) return error.VfsCorrupt;
                self.removeNode(kv.value, true);
            },
            .rename => |r| try self.applyRename(r.from, r.to),
            .deltree => |p| {
                const c = try split(p);
                if (c.n == 0) return error.VfsCorrupt;
                const at = try self.parentOf(c.slice());
                const kv = at.dir.children.fetchRemove(at.name) orelse return error.VfsCorrupt;
                a.free(kv.key);
                self.removeNode(kv.value, true);
            },
            .touch => |t| {
                const vn = try self.fileAt(t.path);
                vn.mtime = t.mtime;
                self.last_mtime = @max(self.last_mtime, t.mtime);
            },
            .state => |s| {
                self.next_ino = @max(self.next_ino, s.next_ino);
                self.last_mtime = @max(self.last_mtime, s.last_mtime);
            },
        }
    }

    /// `kept` of the old list followed by `refs`, acquired into `vn`, whose size is then the sum.
    fn collect(self: *Vfs, vn: *Vnode, old: []const Ext, keep: u32, refs: []const Ref) !void {
        const a = self.allocator;
        if (keep > old.len) return error.VfsCorrupt;
        try vn.exts.ensureTotalCapacity(a, keep + refs.len);
        vn.exts.appendSliceAssumeCapacity(old[0..keep]);
        for (refs) |r| {
            try self.ensureBlock(r);
            vn.exts.appendAssumeCapacity(r.ext);
        }
        for (vn.exts.items, 0..) |e, n| {
            self.acquire(e) catch |err| {
                self.releaseAll(vn.exts.items[0..n]);
                return err;
            };
            vn.size += e.len;
        }
    }

    /// Replay learns a block's location from the first record that names it.
    fn ensureBlock(self: *Vfs, r: Ref) !void {
        if (!self.replaying) {
            if (!self.blocks.contains(r.ext.key())) return error.VfsCorrupt;
            return;
        }
        const a = self.allocator;
        const end: u64 = @as(u64, r.off) + r.ext.len;
        const seen = try self.seen_end.getOrPutValue(a, r.pack, 0);
        seen.value_ptr.* = @max(seen.value_ptr.*, end);
        if (self.blocks.contains(r.ext.key())) return;
        try self.blocks.put(a, r.ext.key(), .{ .pack = r.pack, .off = r.off, .kind = r.kind, .refs = 0, .verified = false });
        _ = try self.packs.getOrPutValue(a, r.pack, .{});
        self.next_pack = @max(self.next_pack, r.pack + 1);
    }

    fn applyRename(self: *Vfs, from: []const u8, to: []const u8) !void {
        const a = self.allocator;
        const cf = try split(from);
        const ct = try split(to);
        if (cf.n == 0 or ct.n == 0) return error.VfsCorrupt;
        const sp = try self.parentOf(cf.slice());
        const src = sp.dir.children.get(sp.name) orelse return error.VfsCorrupt;
        const dp = try self.parentOf(ct.slice());
        if (dp.dir.children.get(dp.name)) |dst| if (src.same(dst)) return;
        try dp.dir.children.ensureUnusedCapacity(a, 1);
        const new_key = try a.dupe(u8, dp.name);
        if (dp.dir.children.fetchRemove(dp.name)) |kv| {
            a.free(kv.key);
            self.removeNode(kv.value, true);
        }
        const old = sp.dir.children.fetchRemove(sp.name).?;
        a.free(old.key);
        dp.dir.children.putAssumeCapacity(new_key, src);
    }

    /// Frees a subtree. `drop_blocks` is false only at teardown, where blocks are dropped with the index.
    fn removeNode(self: *Vfs, node: Node, drop_blocks: bool) void {
        const a = self.allocator;
        switch (node) {
            .file => |f| {
                f.nlink -= 1;
                if (f.nlink != 0) return;
                if (drop_blocks) {
                    self.releaseAll(f.exts.items);
                    self.logical_total -= f.size;
                }
                f.exts.deinit(a);
                a.destroy(f);
            },
            .dir => |d| {
                var it = d.children.iterator();
                while (it.next()) |kv| {
                    a.free(kv.key_ptr.*);
                    self.removeNode(kv.value_ptr.*, drop_blocks);
                }
                d.children.deinit(a);
                a.destroy(d);
            },
        }
    }

    // ------------------------------------------------------------ snapshot

    fn compactLocked(self: *Vfs) !void {
        const a = self.allocator;
        var e: Enc = .{ .a = a };
        defer e.list.deinit(a);
        var firsts: std.AutoHashMapUnmanaged(*Vnode, []u8) = .empty;
        defer {
            var it = firsts.valueIterator();
            while (it.next()) |p| a.free(p.*);
            firsts.deinit(a);
        }
        try self.snapshotDir(&e, self.root, "", &firsts);
        try encodeRec(&e, self.seq, .{ .state = .{ .next_ino = self.next_ino, .last_mtime = @intCast(self.last_mtime), .last_seq = self.seq } });

        {
            const f = try self.vdir.createFile(self.io, "snapshot.tmp", .{ .truncate = true });
            defer f.close(self.io);
            errdefer self.vdir.deleteFile(self.io, "snapshot.tmp") catch {};
            try f.writePositionalAll(self.io, e.list.items, 0);
        }
        try self.vdir.rename("snapshot.tmp", self.vdir, "snapshot", self.io);
        self.snapshot_len = e.list.items.len;
        try self.journal.setLength(self.io, 0);
        self.journal_len = 0;
    }

    fn snapshotDir(self: *Vfs, e: *Enc, d: *DirNode, prefix: []const u8, firsts: *std.AutoHashMapUnmanaged(*Vnode, []u8)) !void {
        const a = self.allocator;
        var it = d.children.iterator();
        while (it.next()) |kv| {
            const path = if (prefix.len == 0) try a.dupe(u8, kv.key_ptr.*) else try std.fmt.allocPrint(a, "{s}/{s}", .{ prefix, kv.key_ptr.* });
            var keep_path = false;
            defer if (!keep_path) a.free(path);
            switch (kv.value_ptr.*) {
                .dir => |sub| {
                    try encodeRec(e, self.seq, .{ .mkdir = .{ .path = path, .ino = sub.ino } });
                    try self.snapshotDir(e, sub, path, firsts);
                },
                .file => |vn| {
                    if (firsts.get(vn)) |first| {
                        try encodeRec(e, self.seq, .{ .link = .{ .from = first, .to = path } });
                        continue;
                    }
                    const refs = try a.alloc(Ref, vn.exts.items.len);
                    defer a.free(refs);
                    for (vn.exts.items, refs) |x, *r| {
                        const b = self.blocks.get(x.key()).?;
                        r.* = .{ .ext = x, .pack = b.pack, .off = b.off, .kind = b.kind };
                    }
                    try encodeRec(e, self.seq, .{ .put = .{ .path = path, .ino = vn.ino, .mtime = @intCast(vn.mtime), .kind = vn.kind, .keep = 0, .refs = refs } });
                    try firsts.put(a, vn, path);
                    keep_path = true;
                },
            }
        }
    }

    // ------------------------------------------------------------ recovery

    fn recover(self: *Vfs) !void {
        const a = self.allocator;
        self.replaying = true;
        defer self.replaying = false;
        self.vdir.deleteFile(self.io, "snapshot.tmp") catch {};
        var applied: u64 = 0;

        if (self.vdir.readFileAlloc(self.io, "snapshot", a, .limited(1 << 32))) |bytes| {
            defer a.free(bytes);
            if (self.replayFrames(bytes)) |r| {
                self.snapshot_len = bytes.len;
                applied = r.last_seq;
            } else |_| {
                self.recovered.snapshot_dropped = true;
                self.vdir.deleteFile(self.io, "snapshot") catch {};
                try self.resetNamespace();
            }
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }

        const jbytes = try self.vdir.readFileAlloc(self.io, "journal", a, .limited(1 << 32));
        defer a.free(jbytes);
        const r = try self.replayFramesLenient(jbytes, applied);
        if (r.valid < jbytes.len) {
            self.recovered.journal_torn_bytes = jbytes.len - r.valid;
            try self.journal.setLength(self.io, r.valid);
        }
        self.journal_len = r.valid;
        self.seq = @max(applied, r.last_seq);
        try self.attachPacks();
        self.seen_end.clearAndFree(a);
    }

    const Replayed = struct { valid: usize, last_seq: u64 };

    /// Applies every frame of a snapshot; any bad frame, or a missing final state record, fails it whole.
    fn replayFrames(self: *Vfs, bytes: []const u8) !Replayed {
        const r = try self.replayFramesLenient(bytes, 0);
        if (r.valid != bytes.len or !r.saw_state) return error.VfsCorrupt;
        return .{ .valid = r.valid, .last_seq = r.last_seq };
    }

    const Lenient = struct { valid: usize, last_seq: u64, saw_state: bool };

    fn replayFramesLenient(self: *Vfs, bytes: []const u8, skip_upto: u64) !Lenient {
        var arena_state = std.heap.ArenaAllocator.init(self.allocator);
        defer arena_state.deinit();
        var pos: usize = 0;
        var last_seq = skip_upto;
        var saw_state = false;
        while (pos + FRAME_HEAD <= bytes.len) {
            const len = std.mem.readInt(u32, bytes[pos..][0..4], .little);
            const want = std.mem.readInt(u64, bytes[pos + 4 ..][0..8], .little);
            if (len > bytes.len - pos - FRAME_HEAD) break;
            const payload = bytes[pos + FRAME_HEAD ..][0..len];
            if (std.hash.XxHash3.hash(0, payload) != want) break;
            _ = arena_state.reset(.retain_capacity);
            const d = decodeRec(arena_state.allocator(), payload) catch break;
            if (skip_upto == 0 or d.seq > skip_upto) {
                self.apply(d.rec) catch break;
                if (d.seq > last_seq) last_seq = d.seq;
                if (d.rec == .state) {
                    saw_state = true;
                    last_seq = d.rec.state.last_seq;
                }
            }
            pos += FRAME_HEAD + len;
        }
        return .{ .valid = pos, .last_seq = last_seq, .saw_state = saw_state };
    }

    fn resetNamespace(self: *Vfs) !void {
        const a = self.allocator;
        const fresh = try a.create(DirNode);
        fresh.* = .{ .ino = 1 };
        self.removeNode(.{ .dir = self.root }, false);
        self.root = fresh;
        self.blocks.clearAndFree(a);
        self.packs.clearAndFree(a);
        self.seen_end.clearAndFree(a);
        self.next_ino = 2;
        self.next_pack = 0;
        self.last_mtime = 0;
        self.logical_total = 0;
        self.snapshot_len = 0;
    }

    /// Opens the pack files the replay named, trims what no record references, deletes the rest.
    fn attachPacks(self: *Vfs) !void {
        const a = self.allocator;
        var pdir = try self.vdir.openDir(self.io, "packs", .{ .iterate = true });
        defer pdir.close(self.io);
        var stray: std.ArrayList(u32) = .empty;
        defer stray.deinit(a);
        var it = pdir.iterate();
        while (try it.next(self.io)) |ent| {
            if (ent.kind != .file or !std.mem.endsWith(u8, ent.name, ".pack")) continue;
            const id = std.fmt.parseInt(u32, ent.name[0 .. ent.name.len - ".pack".len], 10) catch continue;
            const p = self.packs.getPtr(id) orelse {
                try stray.append(a, id);
                continue;
            };
            const f = try pdir.openFile(self.io, ent.name, .{ .mode = .read_write });
            p.file = f;
            const size = try f.length(self.io);
            const refd = self.seen_end.get(id) orelse 0;
            if (size > refd) {
                self.recovered.pack_tail_bytes += size - refd;
                try f.setLength(self.io, refd);
            }
            p.size = @min(size, refd);
        }
        for (stray.items) |id| {
            var name_buf: [32]u8 = undefined;
            self.vdir.deleteFile(self.io, packName(&name_buf, id)) catch {};
            self.next_pack = @max(self.next_pack, id + 1);
        }
        var dead: std.ArrayList(u32) = .empty;
        defer dead.deinit(a);
        var pit = self.packs.iterator();
        while (pit.next()) |kv| {
            if (kv.value_ptr.live_blocks == 0) try dead.append(a, kv.key_ptr.*);
        }
        for (dead.items) |id| self.dropPack(id);
        var best: ?u32 = null;
        var bit = self.packs.iterator();
        while (bit.next()) |kv| {
            if (kv.value_ptr.file != null and (best == null or kv.key_ptr.* > best.?)) best = kv.key_ptr.*;
        }
        if (best) |id| {
            if (self.packs.get(id).?.size < self.opts.pack_limit) self.active = id;
        }
    }
};

const testing = std.testing;
const tio = testing.io;
const ta = testing.allocator;

const Env = struct {
    tmp: testing.TmpDir,
    buf: [Dir.max_path_bytes]u8 = undefined,
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

fn blob(len: usize, tag: u64) ![]u8 {
    const out = try ta.alloc(u8, len);
    var prng = std.Random.DefaultPrng.init(tag);
    prng.random().bytes(out);
    return out;
}

/// The payload is valid until the next call: every use hands it straight to the VFS.
threadlocal var one_segment: [1]Segment = undefined;

fn onePayload(kind: Kind, bytes: []const u8) Payload {
    one_segment[0] = .{ .bytes = bytes };
    return .{ .kind = kind, .segments = &one_segment };
}

const ROW = 100;
const GROUP = 256;

/// `rows` rows of ROW bytes, cut into row groups of GROUP rows, one segment per group.
const RowFile = struct {
    data: []u8,
    segs: []Segment,

    fn init(rows: usize) !RowFile {
        const data = try ta.alloc(u8, rows * ROW);
        for (0..rows) |r| {
            var prng = std.Random.DefaultPrng.init(r + 1);
            prng.random().bytes(data[r * ROW ..][0..ROW]);
        }
        const n_groups = (rows + GROUP - 1) / GROUP;
        const segs = try ta.alloc(Segment, n_groups);
        for (segs, 0..) |*s, g| {
            const lo = g * GROUP * ROW;
            const hi = @min(data.len, (g + 1) * GROUP * ROW);
            s.* = .{ .bytes = data[lo..hi], .tensor = 0, .group = @intCast(g) };
        }
        return .{ .data = data, .segs = segs };
    }
    fn deinit(self: RowFile) void {
        ta.free(self.data);
        ta.free(self.segs);
    }
    fn payload(self: RowFile, kind: Kind) Payload {
        return .{ .kind = kind, .segments = self.segs };
    }
};

fn openAt(env: *const Env, opts: Options) !*Vfs {
    const root = try env.path("vfs", .{});
    defer ta.free(root);
    return Vfs.open(ta, tio, root, opts);
}

fn lessName(_: void, a: DirEntry, b: DirEntry) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

/// Every name, ino, size, nlink, mtime and content hash of the namespace, in one comparable text.
fn dump(v: *Vfs) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(ta);
    try dumpDir(v, "", &out);
    return out.toOwnedSlice(ta);
}

fn dumpDir(v: *Vfs, path: []const u8, out: *std.ArrayList(u8)) !void {
    const ents = try v.iterate(ta, path);
    defer Vfs.freeEntries(ta, ents);
    std.mem.sort(DirEntry, ents, {}, lessName);
    for (ents) |e| {
        const child = if (path.len == 0) try ta.dupe(u8, e.name) else try std.fmt.allocPrint(ta, "{s}/{s}", .{ path, e.name });
        defer ta.free(child);
        const st = v.stat(child).?;
        if (e.is_dir) {
            try out.print(ta, "{s} D ino={d}\n", .{ child, st.ino });
            try dumpDir(v, child, out);
        } else {
            const bytes = try v.readBytes(ta, child);
            defer ta.free(bytes);
            try out.print(ta, "{s} F size={d} nlink={d} ino={d} mtime={d} h={x}\n", .{ child, st.size, st.nlink, st.ino, st.mtime_ns, std.hash.XxHash3.hash(0, bytes) });
        }
    }
}

const DiskState = struct {
    journal: []u8,
    names: [][]u8,
    packs: [][]u8,

    fn read(env: *const Env) !DiskState {
        const vroot = try env.path("vfs/v", .{});
        defer ta.free(vroot);
        var vdir = try Dir.openDirAbsolute(tio, vroot, .{});
        defer vdir.close(tio);
        const journal = vdir.readFileAlloc(tio, "journal", ta, .limited(1 << 28)) catch try ta.alloc(u8, 0);
        var names: std.ArrayList([]u8) = .empty;
        var packs: std.ArrayList([]u8) = .empty;
        var pdir = try vdir.openDir(tio, "packs", .{ .iterate = true });
        defer pdir.close(tio);
        var it = pdir.iterate();
        while (try it.next(tio)) |e| {
            try names.append(ta, try ta.dupe(u8, e.name));
            try packs.append(ta, try pdir.readFileAlloc(tio, e.name, ta, .limited(1 << 28)));
        }
        return .{ .journal = journal, .names = try names.toOwnedSlice(ta), .packs = try packs.toOwnedSlice(ta) };
    }
    fn deinit(self: DiskState) void {
        ta.free(self.journal);
        for (self.names) |n| ta.free(n);
        for (self.packs) |p| ta.free(p);
        ta.free(self.names);
        ta.free(self.packs);
    }
    fn pack(self: DiskState, name: []const u8) ?[]const u8 {
        for (self.names, self.packs) |n, p| if (std.mem.eql(u8, n, name)) return p;
        return null;
    }
};

/// Writes `journal` and, for each pack name in either state, the longer of its two contents, into `<env>/vfs/v`.
fn materialize(env: *const Env, journal: []const u8, a: DiskState, b: ?DiskState) !void {
    const vroot = try env.path("vfs/v/packs", .{});
    defer ta.free(vroot);
    try Dir.cwd().createDirPath(tio, vroot);
    const vdir_path = try env.path("vfs/v", .{});
    defer ta.free(vdir_path);
    var vdir = try Dir.openDirAbsolute(tio, vdir_path, .{});
    defer vdir.close(tio);
    try vdir.writeFile(tio, .{ .sub_path = "journal", .data = journal });
    for ([_]?DiskState{ a, b }) |maybe| {
        const s = maybe orelse continue;
        for (s.names) |name| {
            const sub = try std.fmt.allocPrint(ta, "packs/{s}", .{name});
            defer ta.free(sub);
            const mine = s.pack(name).?;
            var best = mine;
            for ([_]?DiskState{ a, b }) |other| {
                const o = other orelse continue;
                if (o.pack(name)) |p| if (p.len > best.len) {
                    best = p;
                };
            }
            try vdir.writeFile(tio, .{ .sub_path = sub, .data = best });
        }
    }
}

fn onDiskBytes(env: *const Env) !u64 {
    const vroot = try env.path("vfs/v", .{});
    defer ta.free(vroot);
    var vdir = try Dir.openDirAbsolute(tio, vroot, .{ .iterate = true });
    defer vdir.close(tio);
    var total: u64 = 0;
    var it = vdir.iterate();
    while (try it.next(tio)) |e| {
        if (e.kind == .file) total += (try vdir.statFile(tio, e.name, .{})).size;
    }
    var pdir = try vdir.openDir(tio, "packs", .{ .iterate = true });
    defer pdir.close(tio);
    var pit = pdir.iterate();
    while (try pit.next(tio)) |e| total += (try pdir.statFile(tio, e.name, .{})).size;
    return total;
}

fn packCount(env: *const Env) !usize {
    const p = try env.path("vfs/v/packs", .{});
    defer ta.free(p);
    var pdir = try Dir.openDirAbsolute(tio, p, .{ .iterate = true });
    defer pdir.close(tio);
    var n: usize = 0;
    var it = pdir.iterate();
    while (try it.next(tio)) |_| n += 1;
    return n;
}

// ---------------------------------------------------------------- differential mirror

const OpKind = enum { mkdirp, mkdir_excl, write, create, replace, append, truncate, link, unlink, rename, delete_tree, touch };
const Op = struct { kind: OpKind, p: []const u8, q: []const u8, tag: u8, len: usize, off: u64, split: u8 };

/// Data ops dominate, as they do in the tier; each namespace op appears often enough to reach its edge cases.
const op_mix = [_]OpKind{
    .write,        .write,    .create,   .replace,   .replace,   .append,   .append,    .append,      .truncate, .truncate,
    .link,         .link,     .link,     .link,      .link,      .link,     .unlink,    .unlink,      .rename,   .rename,
    .touch,        .mkdirp,   .mkdirp,   .mkdir_excl,
    .delete_tree,
};

const names_all = [_][]const u8{ "a", "b", "a/x", "f0", "f1", "a/f0", "a/f1", "b/f0", "a/x/f0", "a/x/f1", "b/y/f0" };

fn contentFor(op: Op) ![]u8 {
    const lens = [_]usize{ 0, 1, 1000, 16384, 20000, 40000 };
    const out = try blob(lens[op.len % lens.len], op.tag);
    return out;
}

fn segmentsFor(op: Op, bytes: []const u8) ![]Segment {
    if (op.split % 3 == 0) return byteSegments(ta, bytes);
    const cut = if (bytes.len == 0) 0 else bytes.len * (op.split % 3) / 3;
    const segs = try ta.alloc(Segment, if (cut == 0 or cut == bytes.len) 1 else 2);
    if (segs.len == 1) {
        segs[0] = .{ .bytes = bytes, .tensor = op.split, .group = 0 };
    } else {
        segs[0] = .{ .bytes = bytes[0..cut], .tensor = op.split, .group = 0 };
        segs[1] = .{ .bytes = bytes[cut..], .tensor = op.split, .group = 1 };
    }
    return segs;
}

fn doReal(real: Dir, op: Op) !void {
    switch (op.kind) {
        .mkdirp => try real.createDirPath(tio, op.p),
        .mkdir_excl => try real.createDir(tio, op.p, .default_dir),
        .write => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            const f = try real.createFile(tio, op.p, .{});
            defer f.close(tio);
            try f.writePositionalAll(tio, bytes, 0);
        },
        .create => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            const f = try real.createFile(tio, op.p, .{ .exclusive = true });
            defer f.close(tio);
            try f.writePositionalAll(tio, bytes, 0);
        },
        .replace => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            const tmp = try std.fmt.allocPrint(ta, "{s}.tmp", .{op.p});
            defer ta.free(tmp);
            {
                const f = try real.createFile(tio, tmp, .{});
                defer f.close(tio);
                try f.writePositionalAll(tio, bytes, 0);
            }
            real.rename(tmp, real, op.p, tio) catch |e| {
                real.deleteFile(tio, tmp) catch {};
                return e;
            };
        },
        .append => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            const f = try real.openFile(tio, op.p, .{ .mode = .write_only });
            defer f.close(tio);
            try f.writePositionalAll(tio, bytes, op.off);
        },
        .truncate => {
            const f = try real.openFile(tio, op.p, .{ .mode = .write_only });
            defer f.close(tio);
            try f.setLength(tio, op.off);
        },
        .link => try real.hardLink(op.p, real, op.q, tio, .{}),
        .unlink => try real.deleteFile(tio, op.p),
        .rename => {
            // std treats the EINVAL of a directory moved under itself as a bug and aborts.
            if (op.q.len > op.p.len and std.mem.startsWith(u8, op.q, op.p) and op.q[op.p.len] == '/') return error.InvalidPath;
            try real.rename(op.p, real, op.q, tio);
        },
        .delete_tree => try real.deleteTree(tio, op.p),
        .touch => {
            // std crashes on the ENOTDIR of utimensat, so the lookup reports it first.
            _ = try real.statFile(tio, op.p, .{});
            try real.setTimestampsNow(tio, op.p, .{});
        },
    }
}

fn doVfs(v: *Vfs, op: Op) !void {
    const kinds = [_]Kind{ .kv, .ssm, .qsa, .spec, .ring, .meta };
    const kind = kinds[op.tag % kinds.len];
    switch (op.kind) {
        .mkdirp => try v.mkdirp(op.p),
        .mkdir_excl => try v.mkdirExclusive(op.p),
        .write => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            try v.writeBytes(op.p, kind, bytes);
        },
        .create, .replace => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            const segs = try segmentsFor(op, bytes);
            defer ta.free(segs);
            const payload: Payload = .{ .kind = kind, .segments = segs };
            if (op.kind == .create) _ = try v.create(op.p, payload) else _ = try v.replace(op.p, payload);
        },
        .append => {
            const bytes = try contentFor(op);
            defer ta.free(bytes);
            try v.appendBytes(op.p, op.off, bytes);
        },
        .truncate => try v.setLength(op.p, op.off),
        .link => try v.link(op.p, op.q),
        .unlink => try v.unlink(op.p),
        .rename => try v.rename(op.p, op.q),
        .delete_tree => try v.deleteTree(op.p),
        .touch => try v.touch(op.p, v.nowNs()),
    }
}

fn errAgree(re: ?anyerror, ve: ?anyerror) bool {
    if (re == null and ve == null) return true;
    if (re == null or ve == null) return false;
    const strict = [_]anyerror{ error.FileNotFound, error.PathAlreadyExists, error.NotDir, error.IsDir, error.DirNotEmpty };
    // std reports a file used as a directory by `stat` as FileNotFound.
    if (re.? == error.FileNotFound) return ve.? == error.FileNotFound or ve.? == error.NotDir;
    for (strict) |s| if (re.? == s) return ve.? == s;
    return true;
}

const RInfo = struct { path: []const u8, is_dir: bool, size: u64, nlink: u64, inode: u64, mtime: i128 };

fn walkReal(arena: std.mem.Allocator, real: Dir, prefix: []const u8, out: *std.ArrayList(RInfo)) !void {
    var d = if (prefix.len == 0) try real.openDir(tio, ".", .{ .iterate = true }) else try real.openDir(tio, prefix, .{ .iterate = true });
    defer d.close(tio);
    var it = d.iterate();
    while (try it.next(tio)) |e| {
        const child = if (prefix.len == 0) try arena.dupe(u8, e.name) else try std.fmt.allocPrint(arena, "{s}/{s}", .{ prefix, e.name });
        const st = try real.statFile(tio, child, .{});
        const is_dir = e.kind == .directory;
        try out.append(arena, .{ .path = child, .is_dir = is_dir, .size = st.size, .nlink = st.nlink, .inode = st.inode, .mtime = st.mtime.nanoseconds });
        if (is_dir) try walkReal(arena, real, child, out);
    }
}

const Seen = struct { real_mtime: i128, vfs_mtime: i128, real_ino: u64, vfs_ino: u64 };

fn compareStates(real: Dir, v: *Vfs, seen: *std.StringHashMap(Seen)) !void {
    var arena_state = std.heap.ArenaAllocator.init(ta);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var infos: std.ArrayList(RInfo) = .empty;
    try walkReal(arena, real, "", &infos);

    // Listing: the same set of names with the same kinds, at every depth.
    var vpaths: std.ArrayList([]const u8) = .empty;
    try collectVfs(arena, v, "", &vpaths);
    try testing.expectEqual(infos.items.len, vpaths.items.len);

    var files: std.ArrayList(struct { r: RInfo, s: Stat }) = .empty;
    var distinct = std.AutoHashMap(u64, u64).init(arena);
    for (infos.items) |r| {
        const s = v.stat(r.path) orelse return error.MissingInVfs;
        try testing.expectEqual(r.is_dir, s.is_dir);
        if (r.is_dir) continue;
        try testing.expectEqual(r.size, s.size);
        try testing.expectEqual(r.nlink, @as(u64, s.nlink));
        const bytes = try v.readBytes(arena, r.path);
        const want = try real.readFileAlloc(tio, r.path, arena, .limited(1 << 24));
        try testing.expectEqualSlices(u8, want, bytes);
        try files.append(arena, .{ .r = r, .s = s });
        try distinct.put(r.inode, r.size);
    }
    // Free space: the logical bill is the sum over distinct inodes, as `tierBytes` counts it.
    var want_logical: u64 = 0;
    var dit = distinct.valueIterator();
    while (dit.next()) |sz| want_logical += sz.*;
    try testing.expectEqual(want_logical, v.logicalBytes());

    for (files.items, 0..) |x, i| {
        for (files.items[i + 1 ..]) |y| {
            // Inode identity classes.
            try testing.expectEqual(x.r.inode == y.r.inode, x.s.ino == y.s.ino);
            // mtime ordering, where the real clock tells two files apart.
            if (x.r.mtime < y.r.mtime) try testing.expect(x.s.mtime_ns < y.s.mtime_ns);
            if (x.r.mtime > y.r.mtime) try testing.expect(x.s.mtime_ns > y.s.mtime_ns);
        }
        // A file changes its mtime in the VFS exactly when it does on disk.
        if (seen.get(x.r.path)) |old| {
            if (old.real_ino == x.r.inode and old.vfs_ino == x.s.ino) {
                try testing.expectEqual(old.real_mtime != x.r.mtime, old.vfs_mtime != x.s.mtime_ns);
            }
        }
    }
    var kit = seen.keyIterator();
    while (kit.next()) |k| testing.allocator.free(k.*);
    seen.clearRetainingCapacity();
    for (files.items) |x| {
        const key = try testing.allocator.dupe(u8, x.r.path);
        try seen.put(key, .{ .real_mtime = x.r.mtime, .vfs_mtime = x.s.mtime_ns, .real_ino = x.r.inode, .vfs_ino = x.s.ino });
    }
}

fn collectVfs(arena: std.mem.Allocator, v: *Vfs, path: []const u8, out: *std.ArrayList([]const u8)) !void {
    const ents = try v.iterate(arena, path);
    for (ents) |e| {
        const child = if (path.len == 0) e.name else try std.fmt.allocPrint(arena, "{s}/{s}", .{ path, e.name });
        try out.append(arena, child);
        if (e.is_dir) try collectVfs(arena, v, child, out);
    }
}

fn freeSeen(seen: *std.StringHashMap(Seen)) void {
    var it = seen.keyIterator();
    while (it.next()) |k| testing.allocator.free(k.*);
    seen.deinit();
}

fn differential(seed: u64, steps: usize, opts: Options) !void {
    var env_real: Env = undefined;
    try env_real.init();
    defer env_real.deinit();
    var env_vfs: Env = undefined;
    try env_vfs.init();
    defer env_vfs.deinit();
    const v = try openAt(&env_vfs, opts);
    defer v.close();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    var seen = std.StringHashMap(Seen).init(ta);
    defer freeSeen(&seen);
    var ok_by_kind: [std.enums.values(OpKind).len]u32 = @splat(0);

    for (0..steps) |_| {
        const op: Op = .{
            .kind = op_mix[rnd.uintLessThan(usize, op_mix.len)],
            .p = names_all[rnd.uintLessThan(usize, names_all.len)],
            .q = names_all[rnd.uintLessThan(usize, names_all.len)],
            .tag = rnd.uintLessThan(u8, 6),
            .len = rnd.uintLessThan(usize, 6),
            .off = rnd.uintLessThan(u64, 50000),
            .split = rnd.uintLessThan(u8, 6),
        };
        const re: ?anyerror = if (doReal(env_real.tmp.dir, op)) |_| null else |e| e;
        const ve: ?anyerror = if (doVfs(v, op)) |_| null else |e| e;
        if (!errAgree(re, ve)) {
            if (io_util.diagEnvOn("SUSHI_VFS_TRACE")) std.debug.print("op {s} {s} {s}: real {?} vfs {?}\n", .{ @tagName(op.kind), op.p, op.q, re, ve });
            return error.ErrorMismatch;
        }
        if (re == null) ok_by_kind[@intFromEnum(op.kind)] += 1;
        try compareStates(env_real.tmp.dir, v, &seen);
    }
    if (io_util.diagEnvOn("SUSHI_VFS_TRACE")) std.debug.print("seed {d}: successful ops by kind {any}\n", .{ seed, ok_by_kind });
}

test "prefix_vfs: every operation the tier performs answers as a real directory does" {
    for (1..13) |seed| try differential(seed, 300, .{ .compact_min_bytes = 1 << 30 });
}

test "prefix_vfs: the same sequences with compaction and tiny packs" {
    for (20..26) |seed| try differential(seed, 250, .{ .compact_min_bytes = 3000, .pack_limit = 40000 });
}

// ---------------------------------------------------------------- dedup

test "prefix_vfs: re-creating identical bytes writes no data and both names report nlink 1" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    try v.mkdirp("e1");
    try v.mkdirp("e2");
    const rf = try RowFile.init(700);
    defer rf.deinit();
    _ = try v.create("e1/c.bin", rf.payload(.kv));
    const before = v.stats(.kv);
    const phys = v.physicalBytes();
    const size = try v.create("e2/c.bin", rf.payload(.kv));
    try testing.expectEqual(@as(u64, 700 * ROW), size);
    const after = v.stats(.kv);
    try testing.expectEqual(before.data_bytes, after.data_bytes);
    try testing.expect(after.dedup_blocks > before.dedup_blocks);
    // Only the journal record grew.
    const grew = v.physicalBytes() - phys;
    try testing.expect(grew < 1000);
    const a = v.stat("e1/c.bin").?;
    const b = v.stat("e2/c.bin").?;
    try testing.expectEqual(@as(u32, 1), a.nlink);
    try testing.expectEqual(@as(u32, 1), b.nlink);
    try testing.expect(a.ino != b.ino);
    // Two files, one stored copy: the logical bill counts both.
    try testing.expectEqual(@as(u64, 2 * 700 * ROW), v.logicalBytes());
    try testing.expect(v.physicalBytes() < 700 * ROW + 2000);
}

test "prefix_vfs: a file rewritten with more rows writes only the groups at or past the old partial one" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    try v.mkdirp("e1");
    const old = try RowFile.init(1000);
    defer old.deinit();
    _ = try v.create("e1/qsa", old.payload(.qsa));
    const written = v.stats(.qsa).data_bytes;
    try testing.expectEqual(@as(u64, 1000 * ROW), written);

    const grown = try RowFile.init(1300);
    defer grown.deinit();
    const phys = v.physicalBytes();
    _ = try v.replace("e1/qsa", grown.payload(.qsa));
    const new_bytes = v.stats(.qsa).data_bytes - written;
    // Groups 0-2 (768 rows) are full in both; group 3 grew from 232 to 256 rows, group 4 is new.
    try testing.expectEqual(@as(u64, (1300 - 768) * ROW), new_bytes);
    // Only the unchanged groups' references are shared: the journal record is small.
    try testing.expect(v.physicalBytes() - phys < new_bytes + 600);
    const got = try v.readBytes(ta, "e1/qsa");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, grown.data, got);
}

test "prefix_vfs: rewriting a file with the same bytes (a restore's meta.json) writes no data" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    try v.mkdirp("e1");
    const meta = try blob(10000, 5);
    defer ta.free(meta);
    try v.writeBytes("e1/meta.json", .meta, meta);
    const first = v.stat("e1/meta.json").?;
    const before = v.stats(.meta).data_bytes;
    const segs = try byteSegments(ta, meta);
    defer ta.free(segs);
    _ = try v.replace("e1/meta.json", .{ .kind = .meta, .segments = segs });
    try testing.expectEqual(before, v.stats(.meta).data_bytes);
    const second = v.stat("e1/meta.json").?;
    try testing.expect(second.mtime_ns > first.mtime_ns);
    try testing.expect(second.ino != first.ino);
}

test "prefix_vfs: a tokens file appended in place stores each 16 KiB block once" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const all = try blob(5 * BYTE_BLOCK + 123, 9);
    defer ta.free(all);
    try v.writeBytes("t1", .meta, all[0 .. 2 * BYTE_BLOCK + 10]);
    var off: u64 = 2 * BYTE_BLOCK + 10;
    while (off < all.len) {
        const n = @min(@as(usize, 3000), all.len - off);
        try v.appendBytes("t1", off, all[off..][0..n]);
        off += n;
    }
    const got = try v.readBytes(ta, "t1");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, all, got);
    // A second file with the same prefix reuses every full block.
    const before = v.stats(.meta).data_bytes;
    try v.writeBytes("t2", .meta, all[0 .. 4 * BYTE_BLOCK]);
    try testing.expectEqual(before, v.stats(.meta).data_bytes);
}

test "prefix_vfs: payload tags and bytes round-trip, and bytes equal the file image" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const a = try blob(300, 1);
    defer ta.free(a);
    const b = try blob(5000, 2);
    defer ta.free(b);
    const segs = [_]Segment{ .{ .bytes = a, .tensor = UNTAGGED, .group = 0 }, .{ .bytes = b, .tensor = 3, .group = 7 } };
    const size = try v.create("f", .{ .kind = .ssm, .segments = &segs });
    try testing.expectEqual(@as(u64, 5300), size);
    const p = try v.readPayload(ta, "f");
    defer p.deinit(ta);
    try testing.expectEqual(Kind.ssm, p.kind);
    try testing.expectEqual(@as(usize, 2), p.segments.len);
    try testing.expectEqualSlices(u8, a, p.segments[0].bytes);
    try testing.expectEqualSlices(u8, b, p.segments[1].bytes);
    try testing.expectEqual(@as(u32, 3), p.segments[1].tensor);
    try testing.expectEqual(@as(u32, 7), p.segments[1].group);
    const img = try v.readBytes(ta, "f");
    defer ta.free(img);
    try testing.expectEqualSlices(u8, b, img[300..]);
}

// ---------------------------------------------------------------- liveness

test "prefix_vfs: blocks live while any name holds them and the pack goes with its last block" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const x = try blob(3000, 1);
    defer ta.free(x);
    _ = try v.create("f", onePayload(.kv, x));
    try v.link("f", "g");
    try testing.expectEqual(@as(u32, 2), v.stat("g").?.nlink);
    try testing.expectEqual(@as(usize, 1), try packCount(&env));
    try v.unlink("f");
    const got = try v.readBytes(ta, "g");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, x, got);
    try testing.expectEqual(@as(usize, 1), try packCount(&env));
    try v.unlink("g");
    try testing.expectEqual(@as(usize, 0), try packCount(&env));
    try testing.expectEqual(@as(u64, 0), v.deadBytes());
    try testing.expectEqual(@as(u64, 0), v.logicalBytes());
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

test "prefix_vfs: packs rotate at the limit, and one dies only when none of its blocks is referenced" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{ .pack_limit = 4096 });
    defer v.close();
    var bufs: [6][]u8 = undefined;
    for (&bufs, 0..) |*b, i| b.* = try blob(1500, i + 1);
    defer for (bufs) |b| ta.free(b);
    const names = [_][]const u8{ "f0", "f1", "f2", "f3", "f4", "f5" };
    for (names, bufs) |n, b| _ = try v.create(n, onePayload(.kv, b));
    try testing.expectEqual(@as(usize, 3), try packCount(&env));
    try v.unlink("f0");
    try testing.expectEqual(@as(usize, 3), try packCount(&env));
    try testing.expectEqual(@as(u64, 1500), v.deadBytes());
    try v.unlink("f1");
    try testing.expectEqual(@as(usize, 2), try packCount(&env));
    try testing.expectEqual(@as(u64, 0), v.deadBytes());
    for (names[2..], bufs[2..]) |n, b| {
        const got = try v.readBytes(ta, n);
        defer ta.free(got);
        try testing.expectEqualSlices(u8, b, got);
    }
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

test "prefix_vfs: a staged file is invisible until published; an abandoned one is dead bytes" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    try v.mkdirp("e1");
    const a = try blob(4000, 1);
    defer ta.free(a);
    const b = try blob(7000, 2);
    defer ta.free(b);
    const t1 = try v.stageCreate("e1/a", onePayload(.kv, a));
    const t2 = try v.stageCreate("e1/b", onePayload(.kv, b));
    try testing.expectEqual(@as(u64, 4000), t1.data_bytes);
    try testing.expectEqual(@as(u64, 4000), t1.size);
    try testing.expect(v.stat("e1/a") == null);
    const phys = v.physicalBytes();
    const landed = try v.publish(t1);
    try testing.expectEqual(@as(u64, 4000), landed.size);
    try testing.expect(v.stat("e1/a") != null);
    try testing.expectEqual(landed.journal_bytes, v.physicalBytes() - phys);
    // The abandoned ticket's bytes stay in the pack as dead space and are exactly accounted.
    v.abandon(t2);
    try testing.expect(v.stat("e1/b") == null);
    try testing.expectEqual(@as(u64, 7000), v.deadBytes());
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

test "prefix_vfs: a staged create that lost its race to another name fails and leaves dead bytes" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const a = try blob(2000, 1);
    defer ta.free(a);
    const b = try blob(3000, 2);
    defer ta.free(b);
    const t = try v.stageCreate("f", onePayload(.kv, a));
    _ = try v.create("f", onePayload(.kv, b));
    try testing.expectError(error.PathAlreadyExists, v.publish(t));
    try testing.expectEqual(@as(u64, 2000), v.deadBytes());
    const got = try v.readBytes(ta, "f");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, b, got);
}

test "prefix_vfs: a path with a dot component is refused" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    try testing.expectError(error.InvalidPath, v.mkdirp("a/../b"));
    try testing.expectError(error.InvalidPath, v.create("./f", onePayload(.kv, "x")));
}

// ---------------------------------------------------------------- crash

/// One scripted step; the same script drives the live run and every replay's expectation.
fn script(v: *Vfs, i: usize) !void {
    const sizes = [_]usize{ 6000, 9000, 7000, 12000, 5000 };
    const data = try blob(sizes[i % sizes.len], 100 + i);
    defer ta.free(data);
    switch (i) {
        0 => try v.mkdirp("e1"),
        1 => try v.mkdirExclusive("e1/sub"),
        2 => _ = try v.create("e1/a", onePayload(.kv, data)),
        3 => try v.writeBytes("e1/tokens.bin", .meta, data[0..5000]),
        4 => try v.appendBytes("e1/tokens.bin", 5000, data[0..2000]),
        5 => try v.link("e1/a", "e1/sub/a2"),
        6 => _ = try v.create("e1/b", onePayload(.ssm, data)),
        7 => try v.mkdirExclusive("e2"),
        8 => _ = try v.replace("e1/a", onePayload(.kv, data)),
        9 => try v.setLength("e1/tokens.bin", 3000),
        10 => try v.rename("e1/b", "e2/b"),
        11 => try v.touch("e2/b", 12345),
        12 => try v.unlink("e1/sub/a2"),
        13 => try v.writeBytes("e2/tokens.bin", .meta, data[0..4000]),
        14 => try v.setLength("e2/tokens.bin", 9000),
        15 => try v.rename("e2", "e3"),
        16 => try v.deleteTree("e1"),
        17 => _ = try v.create("f16", onePayload(.qsa, data)),
        18 => _ = try v.create("f17", onePayload(.ring, data)),
        19 => try v.deleteTree("e3"),
        20 => try v.unlink("f16"),
        else => unreachable,
    }
}
const SCRIPT_LEN = 21;

test "prefix_vfs: a journal cut anywhere replays to the namespace of the operations before it" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{ .pack_limit = 20000, .compact_min_bytes = 1 << 30 });

    var dumps: [SCRIPT_LEN + 1][]u8 = undefined;
    var disks: [SCRIPT_LEN + 1]DiskState = undefined;
    dumps[0] = try dump(v);
    disks[0] = try DiskState.read(&env);
    for (0..SCRIPT_LEN) |i| {
        try script(v, i);
        dumps[i + 1] = try dump(v);
        disks[i + 1] = try DiskState.read(&env);
    }
    v.close();
    defer for (0..SCRIPT_LEN + 1) |i| {
        ta.free(dumps[i]);
        disks[i].deinit();
    };

    var env2: Env = undefined;
    try env2.init();
    defer env2.deinit();
    for (1..SCRIPT_LEN + 1) |k| {
        const lo = disks[k - 1].journal.len;
        const hi = disks[k].journal.len;
        const cuts = [_]usize{ lo, lo + 1, lo + 5, lo + 11, lo + 12, lo + (hi - lo) / 2, hi - 1, hi };
        for (cuts) |cut| {
            if (cut > hi or cut < lo) continue;
            const whole = cut == hi;
            const want = if (whole) dumps[k] else dumps[k - 1];
            try materialize(&env2, disks[k].journal[0..cut], if (whole) disks[k] else disks[k - 1], if (whole) null else disks[k]);
            const r = try openAt(&env2, .{ .pack_limit = 20000, .compact_min_bytes = 1 << 30 });
            const got = try dump(r);
            defer ta.free(got);
            try testing.expectEqualStrings(want, got);
            if (!whole and cut != lo) try testing.expect(r.recovered.journal_torn_bytes > 0);
            try testing.expectEqual(try onDiskBytes(&env2), r.physicalBytes());
            r.close();
            const vdir = try env2.path("vfs", .{});
            defer ta.free(vdir);
            try Dir.cwd().deleteTree(tio, vdir);
        }
    }
}

test "prefix_vfs: a flipped byte inside a journal record drops that record and every later one" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
    var dumps: [SCRIPT_LEN + 1][]u8 = undefined;
    var disks: [SCRIPT_LEN + 1]DiskState = undefined;
    dumps[0] = try dump(v);
    disks[0] = try DiskState.read(&env);
    for (0..SCRIPT_LEN) |i| {
        try script(v, i);
        dumps[i + 1] = try dump(v);
        disks[i + 1] = try DiskState.read(&env);
    }
    v.close();
    defer for (0..SCRIPT_LEN + 1) |i| {
        ta.free(dumps[i]);
        disks[i].deinit();
    };
    const j: usize = 9;
    const journal = try ta.dupe(u8, disks[SCRIPT_LEN].journal);
    defer ta.free(journal);
    const mid = disks[j - 1].journal.len + (disks[j].journal.len - disks[j - 1].journal.len) / 2;
    journal[mid] ^= 0x55;
    var env2: Env = undefined;
    try env2.init();
    defer env2.deinit();
    try materialize(&env2, journal, disks[SCRIPT_LEN], null);
    const r = try openAt(&env2, .{ .compact_min_bytes = 1 << 30 });
    defer r.close();
    const got = try dump(r);
    defer ta.free(got);
    try testing.expectEqualStrings(dumps[j - 1], got);
    try testing.expectEqual(try onDiskBytes(&env2), r.physicalBytes());
}

test "prefix_vfs: pack bytes no record references are ignored and truncated" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    {
        const v = try openAt(&env, .{});
        defer v.close();
        const x = try blob(5000, 1);
        defer ta.free(x);
        _ = try v.create("f", onePayload(.kv, x));
    }
    const pack_path = try env.path("vfs/v/packs/0.pack", .{});
    defer ta.free(pack_path);
    {
        const f = try Dir.openFileAbsolute(tio, pack_path, .{ .mode = .write_only });
        defer f.close(tio);
        try f.writePositionalAll(tio, "garbage-tail-of-an-unpublished-block", 5000);
    }
    const v = try openAt(&env, .{});
    defer v.close();
    try testing.expectEqual(@as(u64, 36), v.recovered.pack_tail_bytes);
    try testing.expectEqual(@as(u64, 5000), (try Dir.cwd().statFile(tio, pack_path, .{})).size);
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
    // The next block lands where the garbage was.
    const y = try blob(1000, 2);
    defer ta.free(y);
    _ = try v.create("g", onePayload(.kv, y));
    const got = try v.readBytes(ta, "g");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, y, got);
}

test "prefix_vfs: a pack nothing references is deleted at open" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    {
        const v = try openAt(&env, .{});
        v.close();
    }
    const stray = try env.path("vfs/v/packs/7.pack", .{});
    defer ta.free(stray);
    {
        const f = try Dir.createFileAbsolute(tio, stray, .{});
        defer f.close(tio);
        try f.writePositionalAll(tio, "orphan", 0);
    }
    const v = try openAt(&env, .{});
    defer v.close();
    try testing.expectEqual(@as(usize, 0), try packCount(&env));
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

// ---------------------------------------------------------------- checksum

test "prefix_vfs: a flipped pack byte fails that block's first read and no other" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const a = try blob(1000, 1);
    defer ta.free(a);
    const b = try blob(1000, 2);
    defer ta.free(b);
    const c = try blob(1000, 3);
    defer ta.free(c);
    {
        const v = try openAt(&env, .{});
        defer v.close();
        const segs = [_]Segment{ .{ .bytes = a }, .{ .bytes = b }, .{ .bytes = c } };
        _ = try v.create("f1", .{ .kind = .kv, .segments = &segs });
        _ = try v.create("f2", onePayload(.kv, a));
        _ = try v.create("f3", onePayload(.kv, c));
        // Blocks written by this process were hashed on the way in.
        const ok = try v.readBytes(ta, "f1");
        ta.free(ok);
        try testing.expectEqual(@as(u64, 0), v.verify_count);
    }
    const pack_path = try env.path("vfs/v/packs/0.pack", .{});
    defer ta.free(pack_path);
    {
        const f = try Dir.openFileAbsolute(tio, pack_path, .{ .mode = .read_write });
        defer f.close(tio);
        var one: [1]u8 = undefined;
        _ = try f.readPositionalAll(tio, &one, 1500);
        one[0] ^= 0xff;
        try f.writePositionalAll(tio, &one, 1500);
    }
    const v = try openAt(&env, .{});
    defer v.close();
    try testing.expectError(error.VfsCorrupt, v.readBytes(ta, "f1"));
    try testing.expectError(error.VfsCorrupt, v.readBytes(ta, "f1"));
    const got2 = try v.readBytes(ta, "f2");
    defer ta.free(got2);
    try testing.expectEqualSlices(u8, a, got2);
    const got3 = try v.readBytes(ta, "f3");
    defer ta.free(got3);
    try testing.expectEqualSlices(u8, c, got3);
}

test "prefix_vfs: a block already verified in this process is not hashed again" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const a = try blob(1000, 1);
    defer ta.free(a);
    const b = try blob(2000, 2);
    defer ta.free(b);
    {
        const v = try openAt(&env, .{});
        defer v.close();
        const segs = [_]Segment{ .{ .bytes = a }, .{ .bytes = b } };
        _ = try v.create("f", .{ .kind = .kv, .segments = &segs });
    }
    const v = try openAt(&env, .{});
    defer v.close();
    for (0..3) |_| {
        const got = try v.readBytes(ta, "f");
        ta.free(got);
    }
    try testing.expectEqual(@as(u64, 2), v.verify_count);
}

// ---------------------------------------------------------------- accounting and compaction

test "prefix_vfs: physical bytes are the pack, journal and snapshot sizes; logical are the sizes given" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{ .pack_limit = 30000, .compact_min_bytes = 1 << 30 });
    defer v.close();
    var logical: u64 = 0;
    for (0..SCRIPT_LEN) |i| {
        try script(v, i);
        try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
    }
    const rf = try RowFile.init(600);
    defer rf.deinit();
    logical += try v.create("rows", rf.payload(.kv));
    try testing.expectEqual(@as(u64, 600 * ROW), logical);
    try v.compact();
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
    // Dead = pack bytes - the distinct live blocks' bytes.
    try testing.expect(v.deadBytes() <= v.physicalBytes());
}

test "prefix_vfs: compaction keeps the namespace, empties the journal and survives a reopen" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    var before: []u8 = undefined;
    {
        const v = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
        defer v.close();
        for (0..SCRIPT_LEN) |i| try script(v, i);
        before = try dump(v);
        try v.compact();
        const after = try dump(v);
        defer ta.free(after);
        try testing.expectEqualStrings(before, after);
        const st = try DiskState.read(&env);
        defer st.deinit();
        try testing.expectEqual(@as(usize, 0), st.journal.len);
        const snap = try env.path("vfs/v/snapshot", .{});
        defer ta.free(snap);
        _ = try Dir.cwd().statFile(tio, snap, .{});
    }
    defer ta.free(before);
    var want: []u8 = undefined;
    {
        const v = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
        defer v.close();
        const got = try dump(v);
        defer ta.free(got);
        try testing.expectEqualStrings(before, got);
        // Operations after the snapshot replay on top of it.
        try v.mkdirp("later/dir");
        _ = try v.create("later/dir/f", onePayload(.kv, "payload"));
        want = try dump(v);
    }
    defer ta.free(want);
    const r = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
    defer r.close();
    const again = try dump(r);
    defer ta.free(again);
    try testing.expectEqualStrings(want, again);
}

test "prefix_vfs: a crash during compaction keeps the old snapshot and the journal" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    var want: []u8 = undefined;
    var old_journal: []u8 = undefined;
    var old_snapshot: []u8 = undefined;
    const spath = try env.path("vfs/v/snapshot", .{});
    defer ta.free(spath);
    {
        const v = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
        defer v.close();
        for (0..10) |i| try script(v, i);
        try v.compact();
        old_snapshot = try Dir.cwd().readFileAlloc(tio, spath, ta, .limited(1 << 24));
        for (10..SCRIPT_LEN) |i| try script(v, i);
        want = try dump(v);
        const st = try DiskState.read(&env);
        old_journal = try ta.dupe(u8, st.journal);
        st.deinit();
        try v.compact();
    }
    defer ta.free(want);
    defer ta.free(old_journal);
    defer ta.free(old_snapshot);
    const jpath = try env.path("vfs/v/journal", .{});
    defer ta.free(jpath);
    const tmp_snap = try env.path("vfs/v/snapshot.tmp", .{});
    defer ta.free(tmp_snap);
    // Cut after the rename, before the journal was emptied, then cut before the rename.
    for ([_]bool{ false, true }) |before_rename| {
        try Dir.cwd().writeFile(tio, .{ .sub_path = jpath, .data = old_journal });
        try Dir.cwd().writeFile(tio, .{ .sub_path = tmp_snap, .data = "half a snapshot" });
        if (before_rename) try Dir.cwd().writeFile(tio, .{ .sub_path = spath, .data = old_snapshot });
        const v = try openAt(&env, .{ .compact_min_bytes = 1 << 30 });
        defer v.close();
        const got = try dump(v);
        defer ta.free(got);
        try testing.expectEqualStrings(want, got);
        try testing.expectError(error.FileNotFound, Dir.cwd().statFile(tio, tmp_snap, .{}));
        try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
    }
}

test "prefix_vfs: an unreadable snapshot is dropped and reported, and the tier stays usable" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    {
        const v = try openAt(&env, .{});
        defer v.close();
        for (0..5) |i| try script(v, i);
        try v.compact();
    }
    const snap = try env.path("vfs/v/snapshot", .{});
    defer ta.free(snap);
    {
        const f = try Dir.openFileAbsolute(tio, snap, .{ .mode = .write_only });
        defer f.close(tio);
        try f.writePositionalAll(tio, "xx", 20);
    }
    const v = try openAt(&env, .{});
    defer v.close();
    try testing.expect(v.recovered.snapshot_dropped);
    _ = try v.create("fresh", onePayload(.kv, "ok"));
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

test "prefix_vfs: the journal compacts by itself once it outgrows the threshold" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    var want: []u8 = undefined;
    {
        const v = try openAt(&env, .{ .compact_min_bytes = 2000 });
        defer v.close();
        try v.mkdirp("d");
        for (0..60) |i| {
            const name = try std.fmt.allocPrint(ta, "d/f{d}", .{i});
            defer ta.free(name);
            const x = try blob(100, i);
            defer ta.free(x);
            _ = try v.create(name, onePayload(.kv, x));
        }
        want = try dump(v);
        const snap = try env.path("vfs/v/snapshot", .{});
        defer ta.free(snap);
        _ = try Dir.cwd().statFile(tio, snap, .{});
        const st = try DiskState.read(&env);
        defer st.deinit();
        const snap_len = (try Dir.cwd().statFile(tio, snap, .{})).size;
        try testing.expect(st.journal.len < @max(2000, 2 * snap_len) + 600);
    }
    defer ta.free(want);
    const v = try openAt(&env, .{ .compact_min_bytes = 2000 });
    defer v.close();
    const got = try dump(v);
    defer ta.free(got);
    try testing.expectEqualStrings(want, got);
}

// ---------------------------------------------------------------- the writer's four steps

/// The shape the writer's op sink takes; assigning the Vfs functions to it checks the signatures.
const SinkShape = struct {
    ctx: *anyopaque,
    appendBlocks: *const fn (ctx: *anyopaque, ticket: u64) anyerror!u64,
    journalRecord: *const fn (ctx: *anyopaque, ticket: u64) anyerror!u64,
    publish: *const fn (ctx: *anyopaque, ticket: u64) void,
    abort: *const fn (ctx: *anyopaque, ticket: u64) void,
};

fn sinkOf(v: *Vfs) SinkShape {
    return .{ .ctx = v, .appendBlocks = Vfs.opAppendBlocks, .journalRecord = Vfs.opJournalRecord, .publish = Vfs.opPublish, .abort = Vfs.opAbort };
}

/// A payload of one segment whose bytes the Vfs will own.
fn ownedOne(kind: Kind, bytes: []const u8) !OwnedPayload {
    const segs = try ta.alloc(Segment, 1);
    segs[0] = .{ .bytes = try ta.dupe(u8, bytes) };
    return .{ .kind = kind, .segments = segs };
}

test "prefix_vfs: the writer's steps are separate and only the last shows the file" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const sink = sinkOf(v);
    try v.mkdirp("e1");
    const x = try blob(5000, 1);
    defer ta.free(x);
    const t = try v.hold("e1/a", .create, try ownedOne(.kv, x));
    try testing.expectEqual(@as(u64, 5000), t.size);
    try testing.expect(v.stat("e1/a") == null);
    const phys = v.physicalBytes();
    try testing.expectEqual(@as(u64, 5000), try sink.appendBlocks(sink.ctx, t.id));
    try testing.expect(v.stat("e1/a") == null);
    try testing.expectEqual(@as(u64, 5000), v.physicalBytes() - phys);
    const journal = try sink.journalRecord(sink.ctx, t.id);
    sink.publish(sink.ctx, t.id);
    try testing.expectEqual(@as(u64, 5000), v.stat("e1/a").?.size);
    try testing.expectEqual(5000 + journal, v.physicalBytes() - phys);
    const got = try v.readBytes(ta, "e1/a");
    defer ta.free(got);
    try testing.expectEqualSlices(u8, x, got);

    // The same bytes under another name append nothing.
    const t2 = try v.hold("e1/b", .create, try ownedOne(.kv, x));
    try testing.expectEqual(@as(u64, 0), try sink.appendBlocks(sink.ctx, t2.id));
    _ = try sink.journalRecord(sink.ctx, t2.id);
    sink.publish(sink.ctx, t2.id);
    try testing.expectEqual(@as(u32, 1), v.stat("e1/b").?.nlink);
}

test "prefix_vfs: a ticket aborted at any step leaves no file, frees its payload and keeps the lock free" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    const v = try openAt(&env, .{});
    defer v.close();
    const sink = sinkOf(v);
    const live = try blob(1000, 1);
    defer ta.free(live);
    _ = try v.create("live", onePayload(.kv, live));
    const x = try blob(4000, 2);
    defer ta.free(x);

    const held = try v.hold("a", .create, try ownedOne(.kv, x));
    sink.abort(sink.ctx, held.id);
    try testing.expectError(error.UnknownTicket, sink.appendBlocks(sink.ctx, held.id));
    try testing.expectEqual(@as(u64, 0), v.deadBytes());

    const appended = try v.hold("a", .create, try ownedOne(.kv, x));
    _ = try sink.appendBlocks(sink.ctx, appended.id);
    sink.abort(sink.ctx, appended.id);
    try testing.expect(v.stat("a") == null);
    try testing.expectEqual(@as(u64, 4000), v.deadBytes());

    // A record that fails validation releases the lock; the ticket is then aborted.
    const y = try blob(3000, 3);
    defer ta.free(y);
    const clash = try v.hold("live", .create, try ownedOne(.kv, y));
    _ = try sink.appendBlocks(sink.ctx, clash.id);
    try testing.expectError(error.PathAlreadyExists, sink.journalRecord(sink.ctx, clash.id));
    try testing.expectEqual(@as(u64, 1000), v.stat("live").?.size);
    sink.abort(sink.ctx, clash.id);
    try testing.expectEqual(@as(u64, 4000 + 3000), v.deadBytes());

    const early = try v.hold("b", .create, try ownedOne(.kv, x));
    try testing.expectError(error.NotAppended, sink.journalRecord(sink.ctx, early.id));
    sink.abort(sink.ctx, early.id);
    try testing.expectEqual(try onDiskBytes(&env), v.physicalBytes());
}

test "prefix_vfs: tickets landed in order replay to the same namespace" {
    var env: Env = undefined;
    try env.init();
    defer env.deinit();
    var want: []u8 = undefined;
    {
        const v = try openAt(&env, .{});
        defer v.close();
        const sink = sinkOf(v);
        const a = try blob(3000, 1);
        defer ta.free(a);
        const b = try blob(3000, 2);
        defer ta.free(b);
        // Both are held and their blocks appended before either is journaled, as the writer's queue does.
        const t1 = try v.hold("f", .create, try ownedOne(.kv, a));
        const t2 = try v.hold("f", .replace, try ownedOne(.kv, b));
        _ = try sink.appendBlocks(sink.ctx, t1.id);
        _ = try sink.appendBlocks(sink.ctx, t2.id);
        for ([_]Ticket{ t1, t2 }) |t| {
            _ = try sink.journalRecord(sink.ctx, t.id);
            sink.publish(sink.ctx, t.id);
        }
        want = try dump(v);
    }
    defer ta.free(want);
    const r = try openAt(&env, .{});
    defer r.close();
    const got = try dump(r);
    defer ta.free(got);
    try testing.expectEqualStrings(want, got);
}
