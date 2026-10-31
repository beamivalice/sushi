//! `sushi repack <pack>`: a pack-order DeepSeek-V4.1 STQ pack becomes pair-major, in place (docs/pack-format.md#repack).
//! A shard is replaced only after its rewrite was re-read and compared with the original; the index and `config.json`
//! change last, so a run cut short leaves a pack the loader refuses and a rerun finishes.
const std = @import("std");
const exl3 = @import("sushi_exl3");
const gl = exl3.group_layout;
const expert_io = @import("../expert_io.zig");
const dsv41_source = @import("dsv41_source.zig");
const model = @import("../model.zig");
const cli = @import("../cli.zig");
const disk_placement = @import("../disk_placement.zig");
const log = @import("../log.zig");

const page: u64 = 16384;
const marker = ".mlp.switch_mlp.";
const pad_prefix = dsv41_source.pad_prefix;
const tmp_suffix = ".repack-tmp";
const lock_name = ".sushi-repack.lock";
const index_name = "model.safetensors.index.json";
const unit_bytes: u64 = 8 << 20;
const spare_bytes: u64 = 1 << 30;

extern "c" fn lstat(path: [*:0]const u8, buf: *std.c.Stat) c_int;
extern "c" fn kill(pid: c_int, sig: c_int) c_int;
extern "c" fn proc_listpidspath(kind: u32, typeinfo: u32, path: [*:0]const u8, flags: u32, buffer: ?*anyopaque, size: c_int) c_int;

pub const Options = struct {
    workers: usize = 8,
    /// Test seams: stop after this many shards were replaced; count this process's own open files as users; the
    /// free bytes of the pack's volume.
    fail_after_shards: ?usize = null,
    count_self: bool = false,
    free_bytes: ?u64 = null,
};

pub const Outcome = enum { repacked, already_pair_major };

fn errno() c_int {
    return std.c._errno().*;
}

fn pathZ(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, name }, 0);
}

fn openFd(path: [:0]const u8, write: bool) !std.c.fd_t {
    const fd = if (write)
        std.c.open(path.ptr, .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600))
    else
        std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return if (errno() == @backingInt(std.c.E.NOENT)) error.FileNotFound else error.RepackIoFailed;
    // The pack is far larger than memory: nothing here is worth the page cache.
    _ = std.c.fcntl(fd, std.c.F.NOCACHE, @as(c_int, 1));
    return fd;
}

fn writeExact(fd: std.c.fd_t, src: []const u8, offset: u64) !void {
    var done: usize = 0;
    while (done < src.len) {
        const put = std.c.pwrite(fd, src[done..].ptr, src.len - done, @intCast(offset + done));
        if (put < 0) {
            if (errno() == @backingInt(std.c.E.INTR)) continue;
            return error.RepackIoFailed;
        }
        done += @intCast(put);
    }
}

/// F_FULLFSYNC reaches the platter; plain fsync, which macOS leaves in the drive's cache, is the fallback for a volume
/// that refuses it.
fn fullSync(fd: std.c.fd_t) !void {
    if (std.c.fcntl(fd, std.c.F.FULLFSYNC) == 0) return;
    if (std.c.fsync(fd) != 0) return error.RepackIoFailed;
}

fn syncDir(a: std.mem.Allocator, dir: []const u8) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}", .{dir}, 0);
    defer a.free(path);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY, .DIRECTORY = true });
    if (fd < 0) return error.RepackIoFailed;
    defer _ = std.c.close(fd);
    try fullSync(fd);
}

fn unlinkQuiet(path: [:0]const u8) void {
    _ = std.c.unlink(path.ptr);
}

fn fileSize(fd: std.c.fd_t) !u64 {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or st.size < 0) return error.RepackIoFailed;
    return @intCast(st.size);
}

fn readSmall(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    return d.readFileAlloc(io, name, a, .limited(256 << 20));
}

/// `name` replaced by `data` through a tmp file, fsynced before and after the rename; the file keeps its mode.
fn replaceFile(a: std.mem.Allocator, dir: []const u8, name: []const u8, data: []const u8) !void {
    const final = try pathZ(a, dir, name);
    defer a.free(final);
    const tmp_name = try std.fmt.allocPrint(a, "{s}{s}", .{ name, tmp_suffix });
    defer a.free(tmp_name);
    const tmp = try pathZ(a, dir, tmp_name);
    defer a.free(tmp);
    var st: std.c.Stat = undefined;
    const mode: std.c.mode_t = if (lstat(final.ptr, &st) == 0) st.mode & 0o777 else 0o644;
    const fd = try openFd(tmp, true);
    errdefer unlinkQuiet(tmp);
    {
        defer _ = std.c.close(fd);
        try writeExact(fd, data, 0);
        _ = std.c.fchmod(fd, mode);
        try fullSync(fd);
    }
    if (std.c.rename(tmp.ptr, final.ptr) != 0) return error.RepackIoFailed;
    try syncDir(a, dir);
}

// ── Shard headers ─────────────────────────────────────────────────────────

const Entry = struct { key: []const u8, dtype: []const u8, shape: []const u64, start: u64, end: u64 };

const Info = struct {
    /// `__metadata__` in file order.
    meta: []const [2][]const u8,
    /// Every tensor, in data order.
    entries: []Entry,
    data_start: u64,
    file_size: u64,

    fn read(a: std.mem.Allocator, fd: std.c.fd_t) !Info {
        const header = try expert_io.SafetensorsHeader.read(a, fd);
        var meta: std.ArrayList([2][]const u8) = .empty;
        var entries: std.ArrayList(Entry) = .empty;
        const object = header.parsed.value.object;
        for (object.keys(), object.values()) |key, value| {
            if (value != .object) return error.InvalidSafetensorsHeader;
            if (std.mem.eql(u8, key, "__metadata__")) {
                for (value.object.keys(), value.object.values()) |k, v| {
                    if (v != .string) return error.InvalidSafetensorsHeader;
                    try meta.append(a, .{ k, v.string });
                }
                continue;
            }
            const dtype = value.object.get("dtype") orelse return error.InvalidSafetensorsHeader;
            const shape = value.object.get("shape") orelse return error.InvalidSafetensorsHeader;
            const offsets = value.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
            if (dtype != .string or shape != .array or offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsHeader;
            const dims = try a.alloc(u64, shape.array.items.len);
            for (shape.array.items, dims) |d, *out| out.* = if (d == .integer and d.integer >= 0) @intCast(d.integer) else return error.InvalidSafetensorsHeader;
            const lo = offsets.array.items[0];
            const hi = offsets.array.items[1];
            if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer) return error.InvalidSafetensorsHeader;
            try entries.append(a, .{ .key = key, .dtype = dtype.string, .shape = dims, .start = @intCast(lo.integer), .end = @intCast(hi.integer) });
        }
        std.mem.sort(Entry, entries.items, {}, struct {
            fn lt(_: void, x: Entry, y: Entry) bool {
                return x.start < y.start;
            }
        }.lt);
        return .{ .meta = meta.items, .entries = entries.items, .data_start = header.data_start, .file_size = try fileSize(fd) };
    }

    fn stamp(self: Info, key: []const u8) ?[]const u8 {
        for (self.meta) |kv| if (std.mem.eql(u8, kv[0], key)) return kv[1];
        return null;
    }
};

const Routed = struct { prefix: []const u8, name: gl.Name };

fn routedOf(key: []const u8) ?Routed {
    const at = std.mem.indexOf(u8, key, marker) orelse return null;
    const name = gl.Name.parse(key[at + marker.len ..]) catch return null;
    return .{ .prefix = key[0 .. at + marker.len], .name = name };
}

const State = enum { plain, legacy, pair_major };

/// What the shard holds of the routed experts; both orders in one shard is mixed.
fn stateOf(info: Info) !State {
    var legacy = false;
    var major = false;
    for (info.entries) |e| {
        const r = routedOf(e.key) orelse continue;
        if (r.name.part != 0) continue;
        if (!r.name.grouped) return error.Exl3TileOrderNeedsGroups;
        if (r.name.pair_major) major = true else legacy = true;
    }
    if (legacy and major) return error.Exl3TileOrderMixed;
    return if (major) .pair_major else if (legacy) .legacy else .plain;
}

fn stampsAgree(spec: exl3.Spec, info: Info) !void {
    try exl3.shardStampAgrees(spec, info.stamp("k"), info.stamp("codebook"), info.stamp("window"));
}

fn tilesOf(e: Entry, order: exl3.format.TileOrder) !gl.TileDims {
    if (!std.mem.eql(u8, e.dtype, "U16")) return error.Exl3GroupGeometry;
    const dims = gl.tileDims(order, e.shape) orelse return error.Exl3GroupGeometry;
    const bytes = std.math.mul(u64, e.shape[0], dims.it * dims.ot * dims.n * 2) catch return error.Exl3GroupGeometry;
    if (dims.it == 0 or dims.ot == 0 or dims.ot % 2 != 0 or dims.n == 0 or e.shape[0] == 0 or bytes != e.end - e.start) return error.Exl3GroupGeometry;
    return dims;
}

/// A pair-major shard as `sushi repack` writes it, judged from its own header and pads.
fn checkPairMajor(a: std.mem.Allocator, fd: std.c.fd_t, spec: exl3.Spec) !void {
    const info = try Info.read(a, fd);
    if (info.data_start % page != 0) return error.RepackShardInvalid;
    try exl3.shardTileOrderAgrees(.pair_major, info.stamp("tile_order"));
    try stampsAgree(spec, info);
    var end: u64 = 0;
    for (info.entries) |e| {
        if (e.start != end) return error.RepackShardInvalid;
        end = e.end;
        if (std.mem.startsWith(u8, e.key, pad_prefix)) {
            if (!std.mem.eql(u8, e.dtype, "U8") or e.shape.len != 1 or e.end == e.start or e.end - e.start >= page) return error.RepackShardInvalid;
            const zeros = try a.alloc(u8, @intCast(e.end - e.start));
            defer a.free(zeros);
            try expert_io.readExact(fd, zeros, info.data_start + e.start);
            for (zeros) |b| if (b != 0) return error.RepackShardInvalid;
        } else if (routedOf(e.key)) |r| {
            if (r.name.part != 0) continue;
            if (!r.name.pair_major) return error.Exl3TileOrderMixed;
            _ = tilesOf(e, .pair_major) catch return error.RepackShardInvalid;
            if ((info.data_start + e.start) % page != 0) return error.RepackShardInvalid;
        }
    }
    if (info.data_start + end != info.file_size) return error.RepackShardInvalid;
}

// ── The plan of one shard ─────────────────────────────────────────────────

const Kind = enum { copy, permute, pad };

const Item = struct {
    key: []const u8,
    kind: Kind,
    dtype: []const u8,
    shape: []const u64,
    src: u64,
    dst: u64,
    len: u64,
    tiles: gl.TileDims = .{ .it = 0, .ot = 0, .n = 0 },
};

const Plan = struct {
    /// The length word, the header and its space padding: the file's first bytes.
    head: []u8,
    items: []Item,
    data_len: u64,
};

fn appendJsonString(a: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) !void {
    try out.append(a, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(a, "\\\""),
        '\\' => try out.appendSlice(a, "\\\\"),
        0...0x1f => try out.print(a, "\\u{x:0>4}", .{c}),
        else => try out.append(a, c),
    };
    try out.append(a, '"');
}

const RoutedEntry = struct { entry: Entry, ordinal: usize, name: gl.Name };

/// `info` laid out pair-major: for gate, up, down, each group in turn, a projection's trellis then its scales, each
/// trellis on a page; the other tensors follow in source order.
fn buildPlan(a: std.mem.Allocator, info: Info, stem: []const u8, hidden: u32, inter: u32) !Plan {
    var routed: std.ArrayList(RoutedEntry) = .empty;
    var others: std.ArrayList(Entry) = .empty;
    var prefixes: std.ArrayList([]const u8) = .empty;
    for (info.entries) |e| {
        const r = routedOf(e.key) orelse {
            try others.append(a, e);
            continue;
        };
        var ordinal = prefixes.items.len;
        for (prefixes.items, 0..) |p, i| if (std.mem.eql(u8, p, r.prefix)) {
            ordinal = i;
            break;
        };
        if (ordinal == prefixes.items.len) try prefixes.append(a, r.prefix);
        try routed.append(a, .{ .entry = e, .ordinal = ordinal, .name = r.name });
    }
    std.mem.sort(RoutedEntry, routed.items, {}, struct {
        fn lt(_: void, x: RoutedEntry, y: RoutedEntry) bool {
            if (x.ordinal != y.ordinal) return x.ordinal < y.ordinal;
            if (x.name.projection != y.name.projection) return x.name.projection < y.name.projection;
            if (x.name.group != y.name.group) return x.name.group < y.name.group;
            return x.name.part < y.name.part;
        }
    }.lt);

    var items: std.ArrayList(Item) = .empty;
    var at: u64 = 0;
    var pads: usize = 0;
    for (routed.items) |r| {
        const e = r.entry;
        if (r.name.part != 0) {
            try items.append(a, .{ .key = e.key, .kind = .copy, .dtype = e.dtype, .shape = e.shape, .src = e.start, .dst = at, .len = e.end - e.start });
            at += e.end - e.start;
            continue;
        }
        const dims = try tilesOf(e, .pack);
        _ = try gl.trellisRate(.pack, e.shape, if (r.name.projection == 2) inter else hidden, if (r.name.projection == 2) hidden else inter);
        if (at % page != 0) {
            const pad = page - at % page;
            try items.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}{s}.{d}", .{ pad_prefix, stem, pads }), .kind = .pad, .dtype = "U8", .shape = try a.dupe(u64, &.{pad}), .src = 0, .dst = at, .len = pad });
            pads += 1;
            at += pad;
        }
        const shape = try a.dupe(u64, &.{ e.shape[0], dims.ot / 2, dims.it, 2, dims.n });
        const key = try std.fmt.allocPrint(a, "{s}_pm", .{e.key});
        try items.append(a, .{ .key = key, .kind = .permute, .dtype = e.dtype, .shape = shape, .src = e.start, .dst = at, .len = e.end - e.start, .tiles = dims });
        at += e.end - e.start;
    }
    for (others.items) |e| {
        try items.append(a, .{ .key = e.key, .kind = .copy, .dtype = e.dtype, .shape = e.shape, .src = e.start, .dst = at, .len = e.end - e.start });
        at += e.end - e.start;
    }

    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(a, "{\"__metadata__\":{");
    for (info.meta) |kv| {
        try appendJsonString(a, &json, kv[0]);
        try json.append(a, ':');
        try appendJsonString(a, &json, kv[1]);
        try json.append(a, ',');
    }
    try json.appendSlice(a, "\"tile_order\":\"pair_major\"}");
    for (items.items) |it| {
        try json.append(a, ',');
        try appendJsonString(a, &json, it.key);
        try json.print(a, ":{{\"dtype\":\"{s}\",\"shape\":[", .{it.dtype});
        for (it.shape, 0..) |d, i| try json.print(a, "{s}{d}", .{ if (i > 0) "," else "", d });
        try json.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ it.dst, it.dst + it.len });
    }
    try json.append(a, '}');
    while ((8 + json.items.len) % page != 0) try json.append(a, ' ');
    const head = try a.alloc(u8, 8 + json.items.len);
    std.mem.writeInt(u64, head[0..8], json.items.len, .little);
    @memcpy(head[8..], json.items);
    return .{ .head = head, .items = items.items, .data_len = at };
}

// ── Copy and verify ───────────────────────────────────────────────────────

const Unit = struct { src: u64, dst: u64, len: u64, tiles: ?gl.TileDims };

fn buildUnits(a: std.mem.Allocator, plan: Plan, data_src: u64) ![]Unit {
    var units: std.ArrayList(Unit) = .empty;
    const data_dst: u64 = plan.head.len;
    for (plan.items) |it| switch (it.kind) {
        .pad => {},
        .copy => {
            var off: u64 = 0;
            while (off < it.len) : (off += unit_bytes) {
                try units.append(a, .{ .src = data_src + it.src + off, .dst = data_dst + it.dst + off, .len = @min(unit_bytes, it.len - off), .tiles = null });
            }
        },
        .permute => {
            const per = it.tiles.it * it.tiles.ot * it.tiles.n * 2;
            for (0..it.shape[0]) |e| {
                try units.append(a, .{ .src = data_src + it.src + e * per, .dst = data_dst + it.dst + e * per, .len = per, .tiles = it.tiles });
            }
        },
    };
    return units.items;
}

const Job = struct {
    units: []const Unit,
    src_fd: std.c.fd_t,
    dst_fd: std.c.fd_t,
    verify: bool,
    max_len: usize,
    next: std.atomic.Value(usize) = .init(0),
    failure: std.atomic.Value(u32) = .init(0),

    fn fail(self: *Job, err: anyerror) void {
        _ = self.failure.cmpxchgStrong(0, @intFromError(err), .monotonic, .monotonic);
    }

    fn run(self: *Job) void {
        const gpa = std.heap.page_allocator;
        const bufs = gpa.alloc(u8, self.max_len * 3) catch return self.fail(error.OutOfMemory);
        defer gpa.free(bufs);
        while (self.failure.load(.monotonic) == 0) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= self.units.len) return;
            self.one(self.units[i], bufs) catch |err| return self.fail(err);
        }
    }

    fn one(self: *Job, u: Unit, bufs: []u8) !void {
        const len: usize = @intCast(u.len);
        const a = bufs[0..len];
        const b = bufs[self.max_len..][0..len];
        const c = bufs[2 * self.max_len ..][0..len];
        try expert_io.readExact(self.src_fd, a, u.src);
        if (!self.verify) {
            if (u.tiles) |t| {
                gl.tileMajor(u8, b, a, t.it, t.ot, t.n * 2);
                return writeExact(self.dst_fd, b, u.dst);
            }
            return writeExact(self.dst_fd, a, u.dst);
        }
        try expert_io.readExact(self.dst_fd, b, u.dst);
        var got = b;
        if (u.tiles) |t| {
            gl.tileMajorInverse(u8, c, b, t.it, t.ot, t.n * 2);
            got = c;
        }
        if (!std.mem.eql(u8, a, got)) return error.RepackVerifyFailed;
    }
};

fn runUnits(job: *Job, workers: usize) !void {
    const n = @max(1, @min(workers, job.units.len));
    var threads: [32]?std.Thread = @splat(null);
    const spawned = @min(n - 1, threads.len);
    for (threads[0..spawned]) |*t| t.* = std.Thread.spawn(.{}, Job.run, .{job}) catch null;
    job.run();
    for (threads[0..spawned]) |t| if (t) |th| th.join();
    const failed = job.failure.load(.monotonic);
    if (failed != 0) return @errorFromInt(@as(u16, @intCast(failed)));
}

fn maxUnit(units: []const Unit) usize {
    var most: u64 = 1;
    for (units) |u| most = @max(most, u.len);
    return @intCast(most);
}

/// Writes the pair-major shard for `src_fd` to `tmp` and returns once it is on disk and re-read equal.
fn writeShard(a: std.mem.Allocator, src_fd: std.c.fd_t, info: Info, plan: Plan, tmp: [:0]const u8, spec: exl3.Spec, workers: usize) !void {
    const units = try buildUnits(a, plan, info.data_start);
    const total: u64 = plan.head.len + plan.data_len;
    const dst_fd = try openFd(tmp, true);
    {
        defer _ = std.c.close(dst_fd);
        var st: std.c.Stat = undefined;
        if (std.c.fstat(src_fd, &st) == 0) _ = std.c.fchmod(dst_fd, st.mode & 0o777);
        if (std.c.ftruncate(dst_fd, @intCast(total)) != 0) return error.RepackIoFailed;
        try writeExact(dst_fd, plan.head, 0);
        var job: Job = .{ .units = units, .src_fd = src_fd, .dst_fd = dst_fd, .verify = false, .max_len = maxUnit(units) };
        try runUnits(&job, workers);
        try fullSync(dst_fd);
    }

    const check_fd = try openFd(tmp, false);
    defer _ = std.c.close(check_fd);
    if (try fileSize(check_fd) != total) return error.RepackVerifyFailed;
    const head = try a.alloc(u8, plan.head.len);
    try expert_io.readExact(check_fd, head, 0);
    if (!std.mem.eql(u8, head, plan.head)) return error.RepackVerifyFailed;
    checkPairMajor(a, check_fd, spec) catch return error.RepackVerifyFailed;
    var job: Job = .{ .units = units, .src_fd = src_fd, .dst_fd = check_fd, .verify = true, .max_len = maxUnit(units) };
    try runUnits(&job, workers);
}

// ── The pack ──────────────────────────────────────────────────────────────

fn processAlive(pid: c_int) bool {
    return kill(pid, 0) == 0 or errno() == @backingInt(std.c.E.PERM);
}

const Lock = struct {
    path: [:0]u8,

    fn acquire(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Lock {
        const path = try pathZ(a, dir, lock_name);
        errdefer a.free(path);
        for (0..2) |_| {
            const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true }, @as(std.c.mode_t, 0o644));
            if (fd >= 0) {
                defer _ = std.c.close(fd);
                var buf: [24]u8 = undefined;
                try writeExact(fd, try std.fmt.bufPrint(&buf, "{d}\n", .{std.c.getpid()}), 0);
                return .{ .path = path };
            }
            if (errno() != @backingInt(std.c.E.EXIST)) return error.RepackIoFailed;
            const raw = readSmall(io, a, dir, lock_name) catch continue;
            defer a.free(raw);
            const pid = std.fmt.parseInt(c_int, std.mem.trim(u8, raw, " \n"), 10) catch 0;
            if (pid > 0 and processAlive(pid)) return error.RepackPackLocked;
            unlinkQuiet(path);
        }
        return error.RepackPackLocked;
    }

    fn release(self: Lock, a: std.mem.Allocator) void {
        unlinkQuiet(self.path);
        a.free(self.path);
    }
};

/// Does any other process have `path` open? A running server reads shards through cached descriptors that
/// revalidate by inode and size, so it would read pair-major bytes under the pack order.
fn inUse(path: [:0]const u8, count_self: bool) bool {
    var pids: [4096]c_int = undefined;
    const got = proc_listpidspath(1, 0, path.ptr, 0, &pids, @sizeOf(@TypeOf(pids)));
    if (got <= 0) return false;
    const me = std.c.getpid();
    for (pids[0 .. @as(usize, @intCast(got)) / @sizeOf(c_int)]) |pid| if (pid != 0 and (count_self or pid != me)) return true;
    return false;
}

const Shard = struct { name: []const u8, state: State, size: u64 };

const Geometry = struct { hidden: u32, inter: u32 };

fn intField(obj: std.json.ObjectMap, name: []const u8) !u32 {
    const v = obj.get(name) orelse return error.InvalidConfig;
    if (v != .integer or v.integer <= 0) return error.InvalidConfig;
    return std.math.cast(u32, v.integer) orelse error.InvalidConfig;
}

fn indexShards(a: std.mem.Allocator, index: std.json.ObjectMap) ![][]const u8 {
    const map = index.get("weight_map") orelse return error.InvalidSafetensorsIndex;
    if (map != .object) return error.InvalidSafetensorsIndex;
    var names: std.ArrayList([]const u8) = .empty;
    for (map.object.values()) |v| {
        if (v != .string or std.mem.indexOfScalar(u8, v.string, '/') != null) return error.InvalidSafetensorsIndex;
        var known = false;
        for (names.items) |n| known = known or std.mem.eql(u8, n, v.string);
        if (!known) try names.append(a, v.string);
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    return names.items;
}

/// The index with trellis keys renamed and every shard's pads listed after the tensors.
fn newIndex(a: std.mem.Allocator, index: std.json.ObjectMap, shards: []const Shard, dir: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\n");
    var first = true;
    for (index.keys(), index.values()) |key, value| {
        if (!first) try out.appendSlice(a, ",\n");
        first = false;
        try out.append(a, ' ');
        try appendJsonString(a, &out, key);
        try out.appendSlice(a, ": ");
        if (!std.mem.eql(u8, key, "weight_map")) {
            try out.appendSlice(a, try std.json.Stringify.valueAlloc(a, value, .{}));
            continue;
        }
        try out.appendSlice(a, "{\n");
        var n: usize = 0;
        for (value.object.keys(), value.object.values()) |k, v| {
            if (std.mem.startsWith(u8, k, pad_prefix)) continue;
            if (n > 0) try out.appendSlice(a, ",\n");
            n += 1;
            try out.appendSlice(a, "  ");
            const renamed = if (routedOf(k)) |r| r.name.part == 0 and !r.name.pair_major else false;
            try appendJsonString(a, &out, if (renamed) try std.fmt.allocPrint(a, "{s}_pm", .{k}) else k);
            try out.appendSlice(a, ": ");
            try appendJsonString(a, &out, v.string);
        }
        for (shards) |s| {
            if (s.state == .plain) continue;
            const path = try pathZ(a, dir, s.name);
            const fd = try openFd(path, false);
            defer _ = std.c.close(fd);
            const info = try Info.read(a, fd);
            for (info.entries) |e| if (std.mem.startsWith(u8, e.key, pad_prefix)) {
                try out.appendSlice(a, ",\n  ");
                try appendJsonString(a, &out, e.key);
                try out.appendSlice(a, ": ");
                try appendJsonString(a, &out, s.name);
            };
        }
        try out.appendSlice(a, "\n }");
    }
    try out.appendSlice(a, "\n}\n");
    return out.items;
}

/// `config.json` with `"tile_order": "pair_major"` as the first member of `expert_quant`; every other byte unchanged.
fn newConfig(a: std.mem.Allocator, raw: []const u8) ![]u8 {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, raw, from, "\"expert_quant\"")) |at| {
        from = at + 1;
        var i = at + "\"expert_quant\"".len;
        while (i < raw.len and std.ascii.isWhitespace(raw[i])) i += 1;
        if (i >= raw.len or raw[i] != ':') continue;
        i += 1;
        while (i < raw.len and std.ascii.isWhitespace(raw[i])) i += 1;
        if (i >= raw.len or raw[i] != '{') continue;
        const open = i + 1;
        var j = open;
        while (j < raw.len and std.ascii.isWhitespace(raw[j])) j += 1;
        const gap = raw[open..j];
        const member = "\"tile_order\": \"pair_major\",";
        const lead = if (std.mem.indexOfScalar(u8, gap, '\n') != null) gap else "";
        return std.mem.concat(a, u8, &.{ raw[0..open], lead, member, raw[open..] });
    }
    return error.InvalidConfig;
}

pub fn repackPack(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, opts: Options) !Outcome {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    const config_raw = readSmall(io, a, dir, "config.json") catch return error.RepackNotAPack;
    const config = std.json.parseFromSliceLeaky(std.json.Value, a, config_raw, .{}) catch return error.InvalidConfig;
    if (config != .object) return error.InvalidConfig;
    const model_type = config.object.get("model_type") orelse return error.RepackArchitectureUnsupported;
    if (model_type != .string or !std.mem.eql(u8, model_type.string, "deepseek_v41")) return error.RepackArchitectureUnsupported;
    const quant = config.object.get("expert_quant") orelse return error.Exl3TileOrderUnsupported;
    if (quant != .object) return error.Exl3TileOrderUnsupported;
    const format = quant.object.get("format") orelse return error.Exl3TileOrderUnsupported;
    if (format != .string or !std.mem.eql(u8, format.string, "stq")) return error.Exl3TileOrderUnsupported;
    if (quant.object.get("tile_order")) |order| {
        if (order == .string and exl3.format.TileOrder.fromName(order.string) == .pair_major) return .already_pair_major;
        return error.Exl3TileOrderUnsupported;
    }
    const spec = try exl3.parseExpertQuant(config.object);
    const text = if (config.object.get("text_config")) |t| (if (t == .object) t.object else config.object) else config.object;
    const geometry: Geometry = .{ .hidden = try intField(text, "hidden_size"), .inter = try intField(text, "moe_intermediate_size") };

    const lock = try Lock.acquire(gpa, io, dir);
    defer lock.release(gpa);

    const index_raw = readSmall(io, a, dir, index_name) catch return error.RepackNotAPack;
    const index = std.json.parseFromSliceLeaky(std.json.Value, a, index_raw, .{}) catch return error.InvalidSafetensorsIndex;
    if (index != .object) return error.InvalidSafetensorsIndex;
    var shards: std.ArrayList(Shard) = .empty;
    var remaining: u64 = 0;
    for (try indexShards(a, index.object)) |name| {
        const path = try pathZ(a, dir, name);
        var st: std.c.Stat = undefined;
        if (lstat(path.ptr, &st) != 0) return error.RepackShardMissing;
        const fd = try openFd(path, false);
        defer _ = std.c.close(fd);
        const info = try Info.read(a, fd);
        const state = try stateOf(info);
        if (state != .plain) {
            if (std.c.S.ISLNK(st.mode)) return error.RepackShardIsLink;
            try stampsAgree(spec, info);
            if (state == .legacy) {
                try exl3.shardTileOrderAgrees(.pack, info.stamp("tile_order"));
                remaining = @max(remaining, info.file_size);
            } else checkPairMajor(a, fd, spec) catch |err| return if (err == error.RepackShardInvalid) error.Exl3TileOrderMixed else err;
        }
        try shards.append(a, .{ .name = name, .state = state, .size = info.file_size });
    }
    var routed_shards: usize = 0;
    for (shards.items) |s| routed_shards += @intFromBool(s.state != .plain);
    if (routed_shards == 0) return error.ExpertLayoutUnsupported;

    if (remaining > 0) {
        const free = opts.free_bytes orelse (disk_placement.volumeSpaceNear(dir) orelse return error.RepackNoSpace).free;
        if (free < remaining + spare_bytes) return error.RepackNoSpace;
    }
    for (shards.items) |s| {
        if (s.state != .legacy) continue;
        const path = try pathZ(a, dir, s.name);
        if (inUse(path, opts.count_self)) return error.RepackPackInUse;
    }

    var done: usize = 0;
    for (shards.items) |*s| {
        if (s.state == .plain) continue;
        const final = try pathZ(a, dir, s.name);
        const tmp = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ final, tmp_suffix }, 0);
        unlinkQuiet(tmp);
        if (s.state != .legacy) continue;

        const started = std.Io.Timestamp.now(io, .awake);
        var shard_arena = std.heap.ArenaAllocator.init(gpa);
        defer shard_arena.deinit();
        const sa = shard_arena.allocator();
        const src_fd = try openFd(final, false);
        defer _ = std.c.close(src_fd);
        const info = try Info.read(sa, src_fd);
        const stem = std.mem.trimEnd(u8, s.name, ".safetensors");
        const plan = try buildPlan(sa, info, stem, geometry.hidden, geometry.inter);
        errdefer unlinkQuiet(tmp);
        try writeShard(sa, src_fd, info, plan, tmp, spec, opts.workers);
        if (inUse(final, opts.count_self)) return error.RepackPackInUse;
        if (std.c.rename(tmp.ptr, final.ptr) != 0) return error.RepackIoFailed;
        try syncDir(a, dir);
        s.state = .pair_major;
        done += 1;
        const secs = @as(f64, @floatFromInt(started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s;
        log.info("[repack] {s}  {d} bytes  {d:.1} s\n", .{ s.name, plan.head.len + plan.data_len, secs });
        if (opts.fail_after_shards == done) return error.RepackInterrupted;
    }

    try replaceFile(a, dir, index_name, try newIndex(a, index.object, shards.items, dir));
    try replaceFile(a, dir, "config.json", try newConfig(a, config_raw));

    var loaded = try model.parseConfig(io, gpa, dir);
    defer loaded.deinit(gpa);
    try dsv41_source.validateTileOrder(io, gpa, dir, loaded.expert_tile_order, loaded.hidden_size, loaded.moe_intermediate_size);
    return .repacked;
}

/// A pack by path, relative directory or alias, as `sushi run` names one, but never downloaded.
pub fn resolvePack(a: std.mem.Allocator, io: std.Io, name: []const u8, home: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(name)) return a.dupe(u8, name);
    if (cli.resolveRelativeDir(a, io, name)) |abs| return abs;
    const resolved = cli.resolveShortName(name) orelse return error.UnknownModel;
    const dest = try cli.modelDestPath(a, home, resolved.repo);
    if (!cli.modelPresent(io, dest)) {
        a.free(dest);
        return error.RepackNotDownloaded;
    }
    return dest;
}

fn explain(err: anyerror) []const u8 {
    return switch (err) {
        error.RepackNotDownloaded => "the pack is not on this machine (repack never downloads: `sushi pull` it first)",
        error.UnknownModel => "unknown pack name",
        error.RepackNotAPack => "no config.json or model.safetensors.index.json there",
        error.RepackArchitectureUnsupported => "only DeepSeek-V4.1 packs have a pair-major layout",
        error.Exl3TileOrderUnsupported => "only an STQ pack in the pack order can be repacked",
        error.Exl3TileOrderMixed => "a shard holds both orders, or a pair-major shard is not one this command wrote",
        error.Exl3TileOrderNeedsGroups => "a routed layer without rate groups cannot be pair-major",
        error.Exl3ShardStampMismatch => "a shard's stamp (k, codebook, window, tile_order) disagrees with config.json",
        error.RepackShardIsLink => "a shard that holds routed experts is a symlink; replace it with a real file first",
        error.RepackShardMissing => "a shard the index lists is missing",
        error.RepackNoSpace => "free disk is under the largest shard plus 1 GiB",
        error.RepackPackLocked => "another repack holds .sushi-repack.lock",
        error.RepackPackInUse => "a process has a shard open (stop the server that serves this pack)",
        error.RepackVerifyFailed => "a rewritten shard did not read back equal to its original; the original is untouched",
        else => "see the error name",
    };
}

pub fn cmdRepack(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (args.len == 0) {
        log.err("usage: sushi repack <pack>\n", .{});
        std.process.exit(1);
    }
    if (args.len > 1 or (args[0].len > 0 and args[0][0] == '-')) {
        log.err("sushi repack: unexpected argument '{s}' (usage: sushi repack <pack>)\n", .{if (args.len > 1) args[1] else args[0]});
        std.process.exit(1);
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    const dir = resolvePack(allocator, io, args[0], home) catch |err| {
        log.err("sushi repack: {s}: {s}\n", .{ @errorName(err), explain(err) });
        std.process.exit(1);
    };
    defer allocator.free(dir);
    const outcome = repackPack(allocator, io, dir, .{}) catch |err| {
        log.err("sushi repack: {s}: {s}\n", .{ @errorName(err), explain(err) });
        std.process.exit(1);
    };
    switch (outcome) {
        .already_pair_major => log.info("{s} is already pair-major\n", .{dir}),
        .repacked => log.info("{s} is pair-major\n", .{dir}),
    }
}
