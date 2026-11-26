//! A model's plans (docs/bill_redesign.md#coverage): for each root, a phase at the shape that decides its structure, the
//! traces its probes recorded and the tables packed from them. After a request a miss refused, the shape it missed is
//! recorded and learned (D5): its root packs again with the new trace, and the tables replace the root's when they fit
//! the arena held; when they need more, the root keeps its tables and the shape waits for the next load (`keepShape`).
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

    const Root = struct {
        id: FrameId,
        traces: std.ArrayList(Trace) = .empty,
        tables: ?pack.Tables = null,
    };

    pub fn deinit(b: *Book) void {
        for (b.roots.items) |r| {
            for (r.traces.items) |*t| t.deinit();
            r.traces.deinit(b.gpa);
            if (r.tables) |*t| t.deinit();
            b.gpa.destroy(r);
        }
        b.roots.deinit(b.gpa);
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

    fn packRoot(b: *Book, traces: []const Trace) Error!pack.Tables {
        var shapes = try shape_mod.merge(b.gpa, traces, .{ .alignment = b.alignment, .refused = &b.ambiguity });
        defer shapes.deinit();
        return pack.pack(b.gpa, &shapes, b.options);
    }

    /// Packs every root that has no tables yet, each from all its traces.
    pub fn build(b: *Book) Error!void {
        for (b.roots.items) |r| if (r.tables == null) {
            r.tables = try b.packRoot(r.traces.items);
        };
    }

    pub fn tables(b: *const Book, id: FrameId) ?*const pack.Tables {
        for (b.roots.items) |r| if (r.id.eql(id)) return if (r.tables) |*t| t else null;
        return null;
    }

    /// The arena the roots would need packed under `options`, each from all its traces: what another packing costs.
    pub fn layoutWith(b: *Book, options: pack.Options) Error!binder.Layout {
        var l: binder.Layout = .{ .g = 0, .h = 0, .alignment = b.alignment };
        for (b.roots.items) |r| {
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
    pub fn learn(b: *Book, id: FrameId, trace: Trace, held: binder.Layout) Error!Learned {
        var one = [_]Trace{trace};
        return b.learnAll(id, &one, held);
    }

    /// `learn` of a probe's traces together, one a pass of its choices: all of them taken, or none. Each is owned here.
    pub fn learnAll(b: *Book, id: FrameId, traces: []Trace, held: binder.Layout) Error!Learned {
        var taken: usize = 0;
        defer for (traces[taken..]) |*t| t.deinit();
        const r = try b.rootOf(id);
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
        const opened: Opened = switch (c.root) {
            .handle => |h| .{ .dir = h, .owned = false },
            .path => |open| open(c.io) orelse return error.FileNotFound,
        };
        defer if (opened.owned) opened.dir.close(c.io);
        return opened.dir.createDirPathOpen(c.io, c.name(), .{});
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

fn bookUnderFailures(gpa: std.mem.Allocator) !void {
    var b: Book = .{ .gpa = gpa, .options = .{ .level_rule = false }, .alignment = 8 };
    defer b.deinit();
    try b.add(.{ .site = 1 }, try probe(gpa, 64, 32, 8));
    try b.add(.{ .site = 1, .key = 2 }, try probe(gpa, 200, 8, 8));
    try b.build();
    const held = b.layout();
    _ = try b.learn(.{ .site = 1 }, try probe(gpa, 72, 32, 8), held);
    _ = try b.learn(.{ .site = 1 }, try probe(gpa, 72, 32, 40), held);
    _ = try b.learn(.{ .site = 1, .key = 3 }, try probe(gpa, 8, 8, 8), held);
}

test "a book whose every allocation in turn fails frees each thing once and leaks nothing" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, bookUnderFailures, .{});
}
