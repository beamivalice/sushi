//! A model's plans (docs/bill_redesign.md#coverage): for each root, a phase at the shape that decides its structure, the
//! traces its probes recorded and the tables packed from them. After a request a miss refused, the shape it missed is
//! recorded and learned (D5): its root packs again with the new trace, and the tables replace the root's when they fit
//! the arena held; when they need more, the root keeps its tables and the shape waits for the next load (`keepShape`).
//! A book read back from its file (`decode`) holds each root's traces too, so it learns as the book recorded does.
//! Pure but for the file of kept shapes: the host tier builds it.
const std = @import("std");
const frame = @import("plan_frame.zig");
const trace_mod = @import("plan_trace.zig");
const shape_mod = @import("plan_shape.zig");
const pack = @import("plan_pack.zig");
const binder = @import("plan_binder.zig");
const FrameId = frame.FrameId;
const Trace = trace_mod.Trace;

pub const Error = error{ OutOfMemory, PlanAmbiguous };
/// What learning into a root read back from a book file can meet besides `Error`: its traces not decoding.
pub const LearnError = Error || pack.DecodeError;

/// The grain a book places its asks at, unless a load names another.
pub const default_alignment: u32 = 256;

pub const Book = struct {
    gpa: std.mem.Allocator,
    options: pack.Options,
    alignment: u32 = default_alignment,
    /// Each root its own record: a pointer `tables` gave stays good while roots are added.
    roots: std.ArrayList(*Root) = .empty,
    /// What a pack refused as ambiguous, set with `PlanAmbiguous`: the frame and what its instances disagree on.
    ambiguity: shape_mod.Ambiguity = undefined,
    /// The bytes a decoded book's roots keep their traces in, released with the book; null when the caller owns them.
    backing: ?Backing = null,

    pub const Backing = struct { bytes: []const u8, release: *const fn ([]const u8) void };

    const Root = struct {
        id: FrameId,
        traces: std.ArrayList(Trace) = .empty,
        tables: ?pack.Tables = null,
        /// A root read back from a book file: its traces as the file holds them, decoded when it is next learned.
        kept: ?Kept = null,
    };

    /// A root's encoded traces in the bytes a book was decoded from, and their hash, checked when they are decoded.
    const Kept = struct { bytes: []const u8, hash: u64 };

    pub fn deinit(b: *Book) void {
        for (b.roots.items) |r| {
            for (r.traces.items) |*t| t.deinit();
            r.traces.deinit(b.gpa);
            if (r.tables) |*t| t.deinit();
            b.gpa.destroy(r);
        }
        b.roots.deinit(b.gpa);
        if (b.backing) |k| k.release(k.bytes);
        b.* = undefined;
    }

    fn rootOf(b: *Book, id: FrameId) Error!*Root {
        for (b.roots.items) |r| if (r.id.eql(id)) return r;
        try b.roots.ensureUnusedCapacity(b.gpa, 1);
        const r = try b.gpa.create(Root);
        r.* = .{ .id = id };
        b.roots.appendAssumeCapacity(r);
        return r;
    }

    /// `trace`, one probe of the root `id`, is the book's from here, on every path.
    pub fn add(b: *Book, id: FrameId, trace: Trace) Error!void {
        var t = trace;
        const r = b.rootOf(id) catch |e| {
            t.deinit();
            return e;
        };
        r.traces.append(b.gpa, t) catch |e| {
            t.deinit();
            return e;
        };
    }

    fn packRootWith(b: *const Book, traces: []const Trace, ambiguity: *shape_mod.Ambiguity) Error!pack.Tables {
        var shapes = try shape_mod.merge(b.gpa, traces, .{ .alignment = b.alignment, .refused = ambiguity });
        defer shapes.deinit();
        return pack.pack(b.gpa, &shapes, b.options);
    }

    fn packRoot(b: *Book, traces: []const Trace) Error!pack.Tables {
        return b.packRootWith(traces, &b.ambiguity);
    }

    /// The traces of a root read back from a book file, decoded into the root; a root that holds them already is left.
    fn materialize(b: *Book, r: *Root) LearnError!void {
        const kept = (r.kept orelse return).bytes;
        if (std.hash.Wyhash.hash(book_version, kept) != r.kept.?.hash) return error.PlanFileDamaged;
        var cur: pack.BinCursor = .{ .b = kept };
        const n = try cur.len(trace_min_bytes);
        try r.traces.ensureUnusedCapacity(b.gpa, n);
        errdefer {
            for (r.traces.items) |*t| t.deinit();
            r.traces.clearRetainingCapacity();
        }
        for (0..n) |_| r.traces.appendAssumeCapacity(try readTrace(b.gpa, &cur));
        if (cur.at != kept.len) return error.PlanFileDamaged;
        r.kept = null;
    }

    /// Packs every root that has no tables yet, each from all its traces, on up to eight worker threads.
    pub fn build(b: *Book) Error!void {
        return b.buildOn(@min(std.Thread.getCpuCount() catch 1, 8));
    }

    /// `build` on at most `max_workers` threads, the calling one among them; 1 packs on the calling thread alone.
    pub fn buildOn(b: *Book, max_workers: usize) Error!void {
        var pending: usize = 0;
        for (b.roots.items) |r| if (r.tables == null) {
            pending += 1;
        };
        if (pending == 0) return;
        const n_workers = @min(pending, max_workers, 8);
        if (n_workers <= 1) {
            for (b.roots.items) |r| if (r.tables == null) {
                r.tables = try b.packRoot(r.traces.items);
            };
            return;
        }
        const indices = try b.gpa.alloc(usize, pending);
        defer b.gpa.free(indices);
        var p: usize = 0;
        for (b.roots.items, 0..) |r, idx| if (r.tables == null) {
            indices[p] = idx;
            p += 1;
        };
        const Ctx = struct {
            book: *Book,
            indices: []const usize,
            next: std.atomic.Value(usize) = .init(0),
            failed: std.atomic.Value(bool) = .init(false),
            // One slot per root, written only by the worker that claimed it. Every root before a failed one was
            // claimed first and finishes, so the lowest failed slot is the root the serial build fails on.
            errs: []?Failure,

            const Failure = struct { err: Error, ambiguity: shape_mod.Ambiguity };

            fn work(ctx: *@This()) void {
                while (!ctx.failed.load(.acquire)) {
                    const k = ctx.next.fetchAdd(1, .monotonic);
                    if (k >= ctx.indices.len) return;
                    const r = ctx.book.roots.items[ctx.indices[k]];
                    var amb: shape_mod.Ambiguity = undefined;
                    const packed_tables = ctx.book.packRootWith(r.traces.items, &amb) catch |e| {
                        ctx.errs[k] = .{ .err = e, .ambiguity = amb };
                        ctx.failed.store(true, .release);
                        return;
                    };
                    r.tables = packed_tables;
                }
            }
        };
        const errs = try b.gpa.alloc(?Ctx.Failure, pending);
        defer b.gpa.free(errs);
        @memset(errs, null);
        var ctx: Ctx = .{ .book = b, .indices = indices, .errs = errs };
        var threads: [8]std.Thread = undefined;
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |th| th.join();
        for (1..n_workers) |_| {
            threads[spawned] = std.Thread.spawn(.{}, Ctx.work, .{&ctx}) catch break;
            spawned += 1;
        }
        ctx.work();
        for (threads[0..spawned]) |th| th.join();
        spawned = 0;
        if (ctx.failed.load(.acquire)) for (errs) |slot| if (slot) |f| {
            if (f.err == error.PlanAmbiguous) b.ambiguity = f.ambiguity;
            return f.err;
        };
    }

    pub const book_magic = "SUSHIPLB";
    pub const book_version: u64 = 2;

    /// The book as its file holds it: a head of each built root's tables and its traces' length and hash, checksummed,
    /// then each root's traces. A load reads the head alone; a root's traces are read when it is learned.
    pub fn encode(b: *const Book, gpa: std.mem.Allocator) ![]u8 {
        var head: std.Io.Writer.Allocating = .init(gpa);
        defer head.deinit();
        var body: std.Io.Writer.Allocating = .init(gpa);
        defer body.deinit();
        const w = &head.writer;
        try w.writeAll(book_magic);
        try w.writeInt(u64, book_version, .little);
        try w.writeInt(u64, 0, .little);
        var built: u32 = 0;
        for (b.roots.items) |r| if (r.tables != null) {
            built += 1;
        };
        try w.writeInt(u32, built, .little);
        for (b.roots.items) |r| if (r.tables) |*tbl| {
            try w.writeInt(u32, r.id.site, .little);
            try w.writeInt(u64, r.id.key, .little);
            try pack.writeTables(w, tbl);
            const at = body.written().len;
            const kept = r.kept orelse blk: {
                try body.writer.writeInt(u32, @intCast(r.traces.items.len), .little);
                for (r.traces.items) |*t| try writeTrace(&body.writer, t);
                const bytes = body.written()[at..];
                break :blk Kept{ .bytes = bytes, .hash = std.hash.Wyhash.hash(book_version, bytes) };
            };
            if (r.kept != null) try body.writer.writeAll(kept.bytes);
            try w.writeInt(u64, kept.bytes.len, .little);
            try w.writeInt(u64, kept.hash, .little);
        };
        std.mem.writeInt(u64, head.written()[book_magic.len + 8 ..][0..8], head.written().len, .little);
        try w.writeInt(u64, std.hash.Wyhash.hash(book_version, head.written()), .little);
        try w.writeAll(body.written());
        return head.toOwnedSlice();
    }

    /// Decodes the head of a book `encode` wrote: each root's tables, its traces left in `bytes` until it is learned,
    /// so `bytes` outlives the book (`backing` releases them with it).
    pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, options: pack.Options, alignment: u32) pack.DecodeError!Book {
        if (bytes.len < book_magic.len + 16 or !std.mem.eql(u8, bytes[0..book_magic.len], book_magic)) return error.PlanFileDamaged;
        var cur: pack.BinCursor = .{ .b = bytes, .at = book_magic.len };
        if (try cur.u64_() != book_version) return error.PlanFileDamaged;
        const head_len = try cur.u64_();
        if (head_len > bytes.len - 8 or head_len < cur.at) return error.PlanFileDamaged;
        const head = bytes[0..@intCast(head_len)];
        if (std.mem.readInt(u64, bytes[head.len..][0..8], .little) != std.hash.Wyhash.hash(book_version, head)) return error.PlanFileDamaged;
        cur.b = head;
        var traces = head.len + 8;
        const n = try cur.len(12);
        var book: Book = .{ .gpa = gpa, .options = options, .alignment = alignment };
        errdefer book.deinit();
        try book.roots.ensureTotalCapacity(gpa, n);
        for (0..n) |_| {
            const id: FrameId = .{ .site = try cur.u32_(), .key = try cur.u64_() };
            var tbl = try pack.readTables(gpa, &cur);
            errdefer tbl.deinit();
            const len = try cur.u64_();
            const hash = try cur.u64_();
            if (len > bytes.len - traces) return error.PlanFileDamaged;
            const r = try gpa.create(Root);
            r.* = .{ .id = id, .tables = tbl, .kept = .{ .bytes = bytes[traces..][0..@intCast(len)], .hash = hash } };
            book.roots.appendAssumeCapacity(r);
            traces += @intCast(len);
        }
        if (cur.at != head.len or traces != bytes.len) return error.PlanFileDamaged;
        return book;
    }

    pub fn tables(b: *const Book, id: FrameId) ?*const pack.Tables {
        for (b.roots.items) |r| if (r.id.eql(id)) return if (r.tables) |*t| t else null;
        return null;
    }

    /// The arena the roots would need packed under `options`, each from all its traces: what another packing costs.
    pub fn layoutWith(b: *Book, options: pack.Options) LearnError!binder.Layout {
        var l: binder.Layout = .{ .g = 0, .h = 0, .alignment = b.alignment };
        for (b.roots.items) |r| {
            try b.materialize(r);
            var shapes = try shape_mod.merge(b.gpa, r.traces.items, .{ .alignment = b.alignment, .refused = &b.ambiguity });
            defer shapes.deinit();
            var t = try pack.pack(b.gpa, &shapes, options);
            defer t.deinit();
            l.g = @max(l.g, t.g);
            l.h = @max(l.h, t.h);
        }
        return l;
    }

    /// The arena the roots need: the largest scratch and the largest host ring among them.
    pub fn layout(b: *const Book) binder.Layout {
        var l: binder.Layout = .{ .g = 0, .h = 0, .alignment = b.alignment };
        for (b.roots.items) |r| if (r.tables) |t| {
            l.g = @max(l.g, t.g);
            l.h = @max(l.h, t.h);
        };
        return l;
    }

    pub const Learned = enum { swapped, next_load };

    /// After a miss, between executions: `trace`, recorded at the shape that missed, joins the root's traces and the root
    /// packs again. Tables that fit `held` replace the root's (`swapped`); tables that need more leave the root as it was
    /// and the trace goes (`next_load`). The trace is the book's from here, on every path.
    pub fn learn(b: *Book, id: FrameId, trace: Trace, held: binder.Layout) LearnError!Learned {
        var one = [_]Trace{trace};
        return b.learnAll(id, &one, held);
    }

    /// `learn` of a probe's traces together, one a pass of its choices: all of them taken, or none. Each is owned here.
    pub fn learnAll(b: *Book, id: FrameId, traces: []Trace, held: binder.Layout) LearnError!Learned {
        var taken: usize = 0;
        defer for (traces[taken..]) |*t| t.deinit();
        const r = try b.rootOf(id);
        try b.materialize(r);
        const had = r.traces.items.len;
        // Run in reverse: the traces taken are freed, then dropped from the root.
        errdefer r.traces.shrinkRetainingCapacity(had);
        errdefer for (r.traces.items[had..]) |*t| t.deinit();
        try r.traces.ensureUnusedCapacity(b.gpa, traces.len);
        r.traces.appendSliceAssumeCapacity(traces);
        taken = traces.len;
        var next = try b.packRoot(r.traces.items);
        if (next.g > held.g or next.h > held.h) {
            next.deinit();
            for (r.traces.items[had..]) |*t| t.deinit();
            r.traces.shrinkRetainingCapacity(had);
            return .next_load;
        }
        if (r.tables) |*old| old.deinit();
        r.tables = next;
        return .swapped;
    }
};

/// The fewest bytes a trace takes in a book file: its three counts and its root's instance with no events or asks.
const trace_min_bytes = 12 + instance_bytes;
const instance_bytes = 4 + 8 + 1 + 4 * 5 + 1 + 8;

fn writeTrace(w: *std.Io.Writer, t: *const Trace) !void {
    try w.writeInt(u32, @intCast(t.instances.len), .little);
    for (t.instances) |*i| {
        try w.writeInt(u32, i.id.site, .little);
        try w.writeInt(u64, i.id.key, .little);
        try w.writeByte(@backingInt(i.kind));
        for ([_]u32{ i.parent, i.at, i.level_in, i.lo, i.hi }) |v| try w.writeInt(u32, v, .little);
        try w.writeByte(@intFromBool(i.open));
        try w.writeInt(u32, @intCast(i.events.items.len), .little);
        for (i.events.items) |e| switch (e) {
            .ask => |o| {
                try w.writeByte(0);
                try w.writeInt(u32, o, .little);
            },
            .child => |c| {
                try w.writeByte(1);
                try w.writeInt(u32, c, .little);
            },
        };
        try w.writeInt(u32, @intCast(i.asks.items.len), .little);
        for (i.asks.items) |a| {
            try w.writeInt(u64, a.bytes, .little);
            try w.writeByte(@intFromBool(a.host));
            try w.writeByte(@intFromBool(a.external));
            try w.writeByte(a.lift);
            for ([_]u32{ a.death, a.first, a.last, a.died }) |v| try w.writeInt(u32, v, .little);
        }
    }
    try w.writeInt(u32, @intCast(t.waits.len), .little);
    for (t.waits) |x| for ([_]u32{ x.instance, x.at, x.deaths }) |v| try w.writeInt(u32, v, .little);
    try w.writeInt(u32, @intCast(t.choices.len), .little);
    for (t.choices) |c| for ([_]u32{ c.site, c.value, c.at, c.frame, c.event }) |v| try w.writeInt(u32, v, .little);
}

fn readBool(c: *pack.BinCursor) pack.DecodeError!bool {
    const v = try c.u8_();
    if (v > 1) return error.PlanFileDamaged;
    return v == 1;
}

/// A trace `writeTrace` wrote. Every index it holds is checked against what it indexes: a file is refused, never trusted.
fn readTrace(gpa: std.mem.Allocator, c: *pack.BinCursor) pack.DecodeError!Trace {
    const n = try c.len(instance_bytes);
    if (n == 0) return error.PlanFileDamaged;
    const instances = try gpa.alloc(trace_mod.Instance, n);
    var made: usize = 0;
    errdefer {
        for (instances[0..made]) |*i| {
            i.events.deinit(gpa);
            i.asks.deinit(gpa);
        }
        gpa.free(instances);
    }
    for (instances, 0..) |*i, at| {
        i.* = .{ .id = .{ .site = try c.u32_(), .key = try c.u64_() }, .kind = if (try readBool(c)) .each else .exact, .parent = try c.u32_(), .at = try c.u32_(), .level_in = try c.u32_(), .lo = try c.u32_(), .hi = try c.u32_(), .open = try readBool(c) };
        made = at + 1;
        if ((at == 0) != (i.parent == trace_mod.none) or (at > 0 and i.parent >= at)) return error.PlanFileDamaged;
        const n_events = try c.len(5);
        try i.events.ensureTotalCapacity(gpa, n_events);
        for (0..n_events) |_| {
            const child = try readBool(c);
            const v = try c.u32_();
            if (child and (v <= at or v >= n)) return error.PlanFileDamaged;
            i.events.appendAssumeCapacity(if (child) .{ .child = v } else .{ .ask = v });
        }
        const n_asks = try c.len(27);
        try i.asks.ensureTotalCapacity(gpa, n_asks);
        for (0..n_asks) |_| i.asks.appendAssumeCapacity(.{ .bytes = try c.u64_(), .host = try readBool(c), .external = try readBool(c), .lift = try c.u8_(), .death = try c.u32_(), .first = try c.u32_(), .last = try c.u32_(), .died = try c.u32_() });
        for (i.events.items) |e| if (e == .ask and e.ask >= n_asks) return error.PlanFileDamaged;
    }
    const waits = try gpa.alloc(trace_mod.Wait, try c.len(12));
    errdefer gpa.free(waits);
    for (waits) |*x| {
        x.* = .{ .instance = try c.u32_(), .at = try c.u32_(), .deaths = try c.u32_() };
        if (x.instance >= n) return error.PlanFileDamaged;
    }
    const choices = try gpa.alloc(trace_mod.Choice, try c.len(20));
    errdefer gpa.free(choices);
    for (choices) |*x| {
        x.* = .{ .site = try c.u32_(), .value = try c.u32_(), .at = try c.u32_(), .frame = try c.u32_(), .event = try c.u32_() };
        if (x.frame >= n) return error.PlanFileDamaged;
    }
    return .{ .gpa = gpa, .instances = instances, .waits = waits, .choices = choices };
}

/// A shape a served miss recorded, for the next load to plan: the arch's probe, as its phase and four numbers the arch
/// reads (rows, position, ...).
pub const Shape = struct { phase: u32, args: [4]u64 = @splat(0) };

/// Where a load keeps the shapes its misses recorded: the directory `name` under `root` (its pipeline manifest's),
/// opened for each read or write. `root` is the process's, never closed here.
pub const Cache = struct {
    io: std.Io,
    root: Root,
    buf: [64]u8 = undefined,
    len: usize = 0,

    /// A root directory as opened: `owned` when the caller closes it.
    pub const Opened = struct { dir: std.Io.Dir, owned: bool };

    /// Where the cache's directory sits: a handle, or the way to open it from its path each time, so a root removed and
    /// made again is never a stale handle.
    pub const Root = union(enum) {
        handle: std.Io.Dir,
        path: *const fn (io: std.Io) ?Opened,
    };

    pub fn init(io: std.Io, root: Root, sub: []const u8) Cache {
        var c: Cache = .{ .io = io, .root = root, .len = sub.len };
        @memcpy(c.buf[0..sub.len], sub);
        return c;
    }

    /// The cache's own directory, made if missing; the caller closes it.
    pub fn dir(c: *const Cache) !std.Io.Dir {
        return c.openWith(.{});
    }

    /// `dir`, opened so its files can be listed.
    pub fn dirIterable(c: *const Cache) !std.Io.Dir {
        return c.openWith(.{ .iterate = true });
    }

    fn openWith(c: *const Cache, options: std.Io.Dir.OpenOptions) !std.Io.Dir {
        const opened: Opened = switch (c.root) {
            .handle => |h| .{ .dir = h, .owned = false },
            .path => |open| open(c.io) orelse return error.FileNotFound,
        };
        defer if (opened.owned) opened.dir.close(c.io);
        return opened.dir.createDirPathOpen(c.io, c.name(), .{ .open_options = options });
    }

    pub fn name(c: *const Cache) []const u8 {
        return c.buf[0..c.len];
    }

    pub fn read(c: *const Cache, a: std.mem.Allocator) ![]Shape {
        var d = try c.dir();
        defer d.close(c.io);
        return readShapes(a, c.io, d);
    }

    pub fn keep(c: *const Cache, a: std.mem.Allocator, shape: Shape) !void {
        var d = try c.dir();
        defer d.close(c.io);
        return keepShape(a, c.io, d, shape);
    }
};

/// The file of kept shapes in a load's cache directory (beside its pipeline manifest).
pub const shapes_file = "plan-shapes";

pub const ShapesError = error{ OutOfMemory, ShapesCorrupt };

/// The shapes `text` lists, one line each: `phase a0 a1 a2 a3`.
pub fn parseShapes(a: std.mem.Allocator, text: []const u8) ShapesError![]Shape {
    var list: std.ArrayList(Shape) = .empty;
    errdefer list.deinit(a);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        var s: Shape = .{ .phase = std.fmt.parseInt(u32, f.next() orelse return error.ShapesCorrupt, 10) catch return error.ShapesCorrupt };
        for (&s.args) |*v| v.* = std.fmt.parseInt(u64, f.next() orelse return error.ShapesCorrupt, 10) catch return error.ShapesCorrupt;
        if (f.next() != null) return error.ShapesCorrupt;
        try list.append(a, s);
    }
    return list.toOwnedSlice(a);
}

const max_shapes_bytes = 1 << 20;

/// The shapes kept in `dir` for this load to record; none when the file is missing.
pub fn readShapes(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) (ShapesError || std.Io.Dir.ReadFileAllocError)![]Shape {
    const text = dir.readFileAlloc(io, shapes_file, a, .limited(max_shapes_bytes)) catch |e| switch (e) {
        error.FileNotFound => return &.{},
        else => return e,
    };
    defer a.free(text);
    return parseShapes(a, text);
}

var temp_seq = std.atomic.Value(u32).init(0);

/// `shape` joins the shapes kept in `dir`, once; written through a temp file, so a reader sees the old file or the new.
pub fn keepShape(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, shape: Shape) !void {
    const had = try readShapes(a, io, dir);
    defer a.free(had);
    for (had) |s| if (std.meta.eql(s, shape)) return;
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    for (had) |s| try out.writer.print("{d} {d} {d} {d} {d}\n", .{ s.phase, s.args[0], s.args[1], s.args[2], s.args[3] });
    try out.writer.print("{d} {d} {d} {d} {d}\n", .{ shape.phase, shape.args[0], shape.args[1], shape.args[2], shape.args[3] });
    var temp_buf: [64]u8 = undefined;
    const temp = std.fmt.bufPrint(&temp_buf, "{s}.{d}-{d}.tmp", .{ shapes_file, std.c.getpid(), temp_seq.fetchAdd(1, .monotonic) }) catch unreachable;
    errdefer dir.deleteFile(io, temp) catch {};
    try dir.writeFile(io, .{ .sub_path = temp, .data = out.written() });
    try dir.rename(temp, dir, shapes_file, io);
}

const Recorder = trace_mod.Recorder;

/// A probe of root site 1: a buffer of `head`, then one of `tail` after it died, then a host-filled ask of `host`.
fn probe(gpa: std.mem.Allocator, head: u64, tail: u64, host: u64) !Trace {
    var r = try Recorder.begin(gpa, .{ .site = 1 });
    defer r.deinit();
    r.free(try r.ask(head, false, false));
    r.free(try r.ask(tail, false, false));
    r.free(try r.ask(host, true, false));
    return r.end();
}

test "a book packs each root from all its probes, and its layout is the largest of them" {
    const t = std.testing;
    var b: Book = .{ .gpa = t.allocator, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    try b.add(.{ .site = 1 }, try probe(t.allocator, 64, 32, 8));
    try b.add(.{ .site = 1 }, try probe(t.allocator, 16, 96, 24));
    try b.add(.{ .site = 1, .key = 2 }, try probe(t.allocator, 200, 8, 8));
    try b.build();
    // One root's two probes plan as one: each place at the larger of them.
    try t.expectEqual(@as(u64, 96), b.tables(.{ .site = 1 }).?.g);
    try t.expectEqual(@as(u64, 24), b.tables(.{ .site = 1 }).?.h);
    try t.expectEqual(binder.Layout{ .g = 200, .h = 24, .alignment = 8 }, b.layout());
    try t.expectEqual(@as(?*const pack.Tables, null), b.tables(.{ .site = 9 }));
}

test "a learned shape that fits the arena held replaces the tables of its root, and one that needs more leaves them as they were" {
    const t = std.testing;
    var b: Book = .{ .gpa = t.allocator, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    try b.add(.{ .site = 1 }, try probe(t.allocator, 64, 32, 8));
    try b.add(.{ .site = 1, .key = 2 }, try probe(t.allocator, 200, 8, 8));
    try b.build();
    const held = b.layout();
    const first = b.tables(.{ .site = 1 }).?;
    try t.expectEqual(Book.Learned.swapped, try b.learn(.{ .site = 1 }, try probe(t.allocator, 72, 32, 8), held));
    try t.expectEqual(@as(u64, 72), b.tables(.{ .site = 1 }).?.g);
    try t.expectEqual(Book.Learned.next_load, try b.learn(.{ .site = 1 }, try probe(t.allocator, 72, 32, 40), held));
    try t.expectEqual(Book.Learned.next_load, try b.learn(.{ .site = 1 }, try probe(t.allocator, 208, 8, 8), held));
    try t.expectEqual(@as(u64, 72), b.tables(.{ .site = 1 }).?.g);
    try t.expectEqual(@as(u64, 8), b.tables(.{ .site = 1 }).?.h);
    // A root no probe made is learned too, when it fits; the tables an execution holds stay where they were.
    for (3..12) |k| try t.expectEqual(Book.Learned.swapped, try b.learn(.{ .site = 1, .key = k }, try probe(t.allocator, 8, 8, 8), held));
    try t.expect(b.tables(.{ .site = 1, .key = 3 }) != null);
    try t.expectEqual(first, b.tables(.{ .site = 1 }).?);
}

test "the shapes kept for the next load read back as written, each once, and a file that is not one is refused by name" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try t.expectEqual(@as(usize, 0), (try readShapes(t.allocator, io, tmp.dir)).len);
    const a: Shape = .{ .phase = 2, .args = .{ 4, 1_048_513, 1_048_576, 1 } };
    const c: Shape = .{ .phase = 1, .args = .{ 1, 4097, 0, 0 } };
    try keepShape(t.allocator, io, tmp.dir, a);
    try keepShape(t.allocator, io, tmp.dir, c);
    try keepShape(t.allocator, io, tmp.dir, a);
    const got = try readShapes(t.allocator, io, tmp.dir);
    defer t.allocator.free(got);
    try t.expectEqualSlices(Shape, &.{ a, c }, got);
    try t.expectError(error.ShapesCorrupt, parseShapes(t.allocator, "2 4 1\n"));
    try t.expectError(error.ShapesCorrupt, parseShapes(t.allocator, "x 1 2 3 4\n"));
}

fn bookUnderFailures(gpa: std.mem.Allocator, workers: usize) !void {
    var b: Book = .{ .gpa = gpa, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    try b.add(.{ .site = 1 }, try probe(gpa, 64, 32, 8));
    try b.add(.{ .site = 1, .key = 2 }, try probe(gpa, 200, 8, 8));
    try b.buildOn(workers);
    const held = b.layout();
    _ = try b.learn(.{ .site = 1 }, try probe(gpa, 72, 32, 8), held);
    _ = try b.learn(.{ .site = 1 }, try probe(gpa, 72, 32, 40), held);
    _ = try b.learn(.{ .site = 1, .key = 3 }, try probe(gpa, 8, 8, 8), held);
}

/// An allocator the build's workers share, one call at a time: a failing allocator's counts are not atomic.
const Serialized = struct {
    child: std.mem.Allocator,
    busy: std.atomic.Value(bool) = .init(false),

    fn allocator(l: *Serialized) std.mem.Allocator {
        return .{ .ptr = l, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn of(ctx: *anyopaque) *Serialized {
        const l: *Serialized = @ptrCast(@alignCast(ctx));
        while (l.busy.swap(true, .acquire)) std.atomic.spinLoopHint();
        return l;
    }
    fn done(l: *Serialized) void {
        l.busy.store(false, .release);
    }
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const l = of(ctx);
        defer l.done();
        return l.child.rawAlloc(len, a, ra);
    }
    fn resize(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) bool {
        const l = of(ctx);
        defer l.done();
        return l.child.rawResize(m, a, n, ra);
    }
    fn remap(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const l = of(ctx);
        defer l.done();
        return l.child.rawRemap(m, a, n, ra);
    }
    fn free(ctx: *anyopaque, m: []u8, a: std.mem.Alignment, ra: usize) void {
        const l = of(ctx);
        defer l.done();
        l.child.rawFree(m, a, ra);
    }
};

fn bookUnderFailuresOnTwoThreads(gpa: std.mem.Allocator) !void {
    var shared: Serialized = .{ .child = gpa };
    return bookUnderFailures(shared.allocator(), 2);
}

test "a book whose every allocation in turn fails frees each thing once and leaks nothing, built on one thread and on two" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, bookUnderFailures, .{1});
    try std.testing.checkAllAllocationFailures(std.testing.allocator, bookUnderFailuresOnTwoThreads, .{});
}

fn decodeBookForTest(gpa: std.mem.Allocator, bytes: []const u8) !void {
    var b = try Book.decode(gpa, bytes, .{ .level_rule = false }, 8);
    defer b.deinit();
    const held: binder.Layout = .{ .g = 1 << 20, .h = 1 << 20, .alignment = 8 };
    _ = try b.learn(.{ .site = 1, .key = 1 }, try probe(gpa, 16, 16, 8), held);
}

test "a book round-trips through its file with its traces, packs on threads as on one, and a root read back and learned is written with the trace it learned" {
    const t = std.testing;
    var b: Book = .{ .gpa = t.allocator, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    var serial: Book = .{ .gpa = t.allocator, .options = .{ .level_rule = false }, .alignment = 8 };
    defer serial.deinit();
    for (1..10) |k| {
        try b.add(.{ .site = 1, .key = k }, try probe(t.allocator, 64 * k, 32, 8 * k));
        try serial.add(.{ .site = 1, .key = k }, try probe(t.allocator, 64 * k, 32, 8 * k));
    }
    try b.buildOn(4);
    try serial.buildOn(1);
    for (1..10) |k| try t.expect(b.tables(.{ .site = 1, .key = k }).?.eql(serial.tables(.{ .site = 1, .key = k }).?));
    const bytes = try b.encode(t.allocator);
    defer t.allocator.free(bytes);

    var dec = try Book.decode(t.allocator, bytes, .{ .level_rule = false }, 8);
    defer dec.deinit();
    try t.expectEqual(b.layout(), dec.layout());
    for (1..10) |k| {
        const want = b.tables(.{ .site = 1, .key = k }).?;
        const got = dec.tables(.{ .site = 1, .key = k }).?;
        try t.expect(want.eql(got));
    }

    // A book read back and written again is the same file, its roots' traces passed through undecoded.
    const again = try dec.encode(t.allocator);
    defer t.allocator.free(again);
    try t.expectEqualSlices(u8, bytes, again);
    // Its roots' traces read back as they were recorded.
    try dec.materialize(dec.roots.items[0]);
    for (b.roots.items[0].traces.items, dec.roots.items[0].traces.items) |*x, *y| {
        try t.expectEqual(@as(?trace_mod.Difference, null), trace_mod.firstDifference(x.instances, y.instances));
        try t.expect(trace_mod.sameWaits(x.waits, y.waits));
        try t.expectEqualSlices(trace_mod.Choice, x.choices, y.choices);
    }
    // A root read back and learned is written with the trace it learned.
    try t.expectEqual(Book.Learned.swapped, try dec.learn(.{ .site = 1, .key = 1 }, try probe(t.allocator, 16, 16, 8), dec.layout()));
    const learned = try dec.encode(t.allocator);
    defer t.allocator.free(learned);
    var dec2 = try Book.decode(t.allocator, learned, .{ .level_rule = false }, 8);
    defer dec2.deinit();
    try dec2.materialize(dec2.roots.items[0]);
    try t.expectEqual(@as(usize, 2), dec2.roots.items[0].traces.items.len);
    try t.expect(dec.tables(.{ .site = 1, .key = 1 }).?.eql(dec2.tables(.{ .site = 1, .key = 1 }).?));

    // A changed byte in the head or a file cut short is refused as it is read; one in a root's traces when they are.
    const bad = try t.allocator.dupe(u8, bytes);
    defer t.allocator.free(bad);
    bad[Book.book_magic.len + 20] ^= 1;
    try t.expectError(error.PlanFileDamaged, Book.decode(t.allocator, bad, .{ .level_rule = false }, 8));
    try t.expectError(error.PlanFileDamaged, Book.decode(t.allocator, bytes[0 .. bytes.len - 1], .{ .level_rule = false }, 8));
    bad[Book.book_magic.len + 20] ^= 1;
    bad[bad.len - 1] ^= 1;
    var torn = try Book.decode(t.allocator, bad, .{ .level_rule = false }, 8);
    defer torn.deinit();
    try t.expectError(error.PlanFileDamaged, torn.learn(.{ .site = 1, .key = 9 }, try probe(t.allocator, 16, 16, 8), torn.layout()));
    try t.expect(torn.tables(.{ .site = 1, .key = 9 }).?.eql(b.tables(.{ .site = 1, .key = 9 }).?));

    try t.checkAllAllocationFailures(t.allocator, decodeBookForTest, .{bytes});
}

test "a decoded root learns a fitting shape as the recorded book does, and the two books then hold the same tables" {
    const t = std.testing;
    var b: Book = .{ .gpa = t.allocator, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    for (1..10) |k| try b.add(.{ .site = 1, .key = k }, try probe(t.allocator, 64 * k, 32, 8 * k));
    try b.build();
    const bytes = try b.encode(t.allocator);
    defer t.allocator.free(bytes);
    var dec = try Book.decode(t.allocator, bytes, .{ .level_rule = false }, 8);
    defer dec.deinit();
    const held: binder.Layout = .{ .g = 1 << 20, .h = 1 << 20, .alignment = 8 };
    try t.expectEqual(Book.Learned.swapped, try b.learn(.{ .site = 1, .key = 1 }, try probe(t.allocator, 40, 40, 8), held));
    try t.expectEqual(Book.Learned.swapped, try dec.learn(.{ .site = 1, .key = 1 }, try probe(t.allocator, 40, 40, 8), held));
    try t.expect(b.tables(.{ .site = 1, .key = 1 }).?.eql(dec.tables(.{ .site = 1, .key = 1 }).?));
}
