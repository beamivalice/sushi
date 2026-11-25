//! The plan's surface: what a forward's code and the engine's choke points call (docs/bill_redesign.md#the-plan).
//! Outside an execution every call here is a branch on one flag and changes nothing, so the same forward runs on an
//! MLX stream, on the CPU oracle's and through a load. Pure: the host tier builds it. The inference thread only.
const std = @import("std");
const frame_mod = @import("plan_frame.zig");
const trace = @import("plan_trace.zig");
const binder_mod = @import("plan_binder.zig");

pub const Site = frame_mod.Site;
pub const FrameId = frame_mod.FrameId;
pub const siteOf = frame_mod.siteOf;
pub const keyOf = frame_mod.keyOf;
pub const Recorder = trace.Recorder;
pub const Ref = trace.Ref;
pub const TraceChoice = trace.Choice;
pub const Binder = binder_mod.Binder;
pub const binder = binder_mod;
pub const trace_mod = trace;
pub const shape = @import("plan_shape.zig");
pub const pack = @import("plan_pack.zig");
pub const region = @import("plan_region.zig");

pub const Error = error{ OutOfMemory, PlanMiss, PlanNeedsLease, PlanEscape };

pub const Mode = enum {
    /// No execution: the pool, or no engine at all.
    off,
    /// A real execution on the pool, recorded: the recorder logs what it asks for, and data decides what it decides
    /// when serving (a choice takes its real value, a bound its real size).
    real,
    /// Dry mode: the recorder logs what the code asks for and no GPU work runs; each `choose` takes the driver's value
    /// and each `bound` its most.
    dry,
    /// An execution on the arena: the binder places what the code asks for.
    replay,
    /// A wide execution (D97: a prefill chunk, an image's encode): `core` places what it asks at the region's lowest gap
    /// that holds it. Frames open nothing; a choice takes its real value, a bound its real size; the forward runs as a
    /// planned one (`planned`), the sizes its dry probes measured the most of.
    wide,
};

pub const Session = struct {
    mode: Mode = .off,
    recorder: ?*Recorder = null,
    binder: ?*Binder = null,
    driver: ?*Driver = null,
    /// The engine's mark of all it has encoded so far (`plan_binder.zig`, "Order").
    mark: *const fn () u64 = noMark,
    /// A padded wide execution follows each `choose` it takes (`core.beginWidePadded`).
    on_choice: ?*const fn (Site, u32) void = null,
};

fn noMark() u64 {
    return 0;
}

/// The inference thread's session. The engine sets it as an execution or a recording begins and puts it back after.
pub var session: Session = .{};

pub const Frame = struct {
    on: bool,

    /// No frame: what a site whose key costs work opens outside a recording (`recording`).
    pub const none: Frame = .{ .on = false };

    pub fn end(f: Frame) void {
        if (!f.on) return;
        switch (session.mode) {
            .off, .wide => {},
            .real, .dry => session.recorder.?.exit(),
            .replay => session.binder.?.exit(session.mark()),
        }
    }
};

/// A frame that is part of its parent's structure at this place. `key`: the ints, enums and bools that decide what
/// the frame asks for and in what order; sizes are not part of it. Close it with `defer f.end()`.
pub inline fn frame(comptime src: std.builtin.SourceLocation, key: anytype) Error!Frame {
    return open(comptime siteOf(src), keyOf(key), .exact);
}

/// One iteration of a loop whose count the shape or the data decides. Its iterations may differ in key, and all hand
/// up the same number of results; what the parent keeps of an iteration it keeps to the loop's end.
/// Whether a recording runs: a site whose key costs work builds it only then.
pub fn recording() bool {
    return session.mode == .real or session.mode == .dry;
}

/// Whether a plan is in force: a recording, or an execution on the arena, which opens the frames its recording opened.
pub fn planned() bool {
    return session.mode != .off;
}

pub inline fn each(comptime src: std.builtin.SourceLocation, key: anytype) Error!Frame {
    return open(comptime siteOf(src), keyOf(key), .each);
}

fn open(site: Site, key: frame_mod.Key, kind: frame_mod.Kind) Error!Frame {
    switch (session.mode) {
        .off, .wide => return .{ .on = false },
        .real, .dry => try session.recorder.?.enter(.{ .site = site, .key = key }, kind),
        .replay => try session.binder.?.enter(.{ .site = site, .key = key }, kind),
    }
    return .{ .on = true };
}

/// A structure that data decides (a streamed route's kind, an arm a readback picks): `real` when serving or recording
/// a real execution, and in a dry recording the driver's value, each value of `T` in turn over its passes, so every one
/// is planned. Its result belongs in the key of the frame it guards. A dry session always carries a driver.
pub fn choose(comptime src: std.builtin.SourceLocation, comptime T: type, real: T) T {
    const site = comptime siteOf(src);
    if (session.mode == .wide) {
        if (session.on_choice) |f| f(site, valueIndex(T, real));
        return real;
    }
    if (session.mode != .dry) return real;
    const driver = session.driver.?;
    const v: T = if (T == bool) driver.pick(site, 2) == 1 else (comptime std.enums.values(T))[driver.pick(site, std.enums.values(T).len)];
    if (session.recorder) |r| r.noteChoice(site, valueIndex(T, v));
    return v;
}

fn valueIndex(comptime T: type, v: T) u32 {
    if (T == bool) return @intFromBool(v);
    inline for (comptime std.enums.values(T), 0..) |x, i| if (x == v) return i;
    unreachable;
}

/// A size that data decides, with the most the op's own math allows: a dry recording plans `max`, and a served size past
/// it is refused by name.
pub fn bound(real: usize, max: usize) Error!usize {
    switch (session.mode) {
        .off, .real, .wide => return real,
        .dry => return max,
        .replay => {
            if (real <= max) return real;
            const b = session.binder.?;
            return b.refuse(.{ .kind = .bound, .id = b.innermost(), .asked = real, .cap = max });
        },
    }
}

/// What makes a recording reach every value of every `choose`, the ones that only exist under another's value
/// included. A pass is one run of the probe script; each site holds one value for a whole pass.
pub const Driver = struct {
    gpa: std.mem.Allocator,
    sites: std.AutoArrayHashMapUnmanaged(Site, State) = .empty,
    /// What this pass holds each site to, the site it is exploring included.
    forced: std.AutoHashMapUnmanaged(Site, u32) = .empty,
    /// The choices this pass has made, each site once, in the order met.
    trail: std.ArrayList(Choice) = .empty,
    failed: bool = false,

    pub const Choice = struct { site: Site, value: u32 };
    const State = struct {
        values: u32,
        /// One bit a value recorded.
        tried: u64 = 0,
        /// The choices made before this site was first met: forcing them leads back to it.
        context: []Choice,
    };

    pub fn deinit(d: *Driver) void {
        for (d.sites.values()) |s| d.gpa.free(s.context);
        d.sites.deinit(d.gpa);
        d.forced.deinit(d.gpa);
        d.trail.deinit(d.gpa);
        d.* = undefined;
    }

    /// The value site `site` takes in this pass, of `values` (at most 64).
    pub fn pick(d: *Driver, site: Site, values: usize) u32 {
        std.debug.assert(values >= 1 and values <= 64);
        for (d.trail.items) |c| if (c.site == site) return c.value;
        const value = d.forced.get(site) orelse 0;
        const got = d.sites.getOrPut(d.gpa, site) catch return d.fail();
        if (!got.found_existing) {
            got.value_ptr.* = .{ .values = @intCast(values), .context = &.{} };
            got.value_ptr.context = d.gpa.dupe(Choice, d.trail.items) catch return d.fail();
        }
        got.value_ptr.tried |= @as(u64, 1) << @intCast(value);
        d.trail.append(d.gpa, .{ .site = site, .value = value }) catch return d.fail();
        return value;
    }

    fn fail(d: *Driver) u32 {
        d.failed = true;
        return 0;
    }

    /// Sets up the next pass; false when every value of every site met has been recorded.
    pub fn next(d: *Driver) error{OutOfMemory}!bool {
        if (d.failed) return error.OutOfMemory;
        d.trail.clearRetainingCapacity();
        d.forced.clearRetainingCapacity();
        for (d.sites.keys(), d.sites.values()) |site, state| {
            const untried = ~state.tried & (if (state.values == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(state.values)) - 1);
            if (untried == 0) continue;
            for (state.context) |c| try d.forced.put(d.gpa, c.site, c.value);
            try d.forced.put(d.gpa, site, @ctz(untried));
            return true;
        }
        return false;
    }
};

test "outside an execution a frame, a choice and a bound change nothing" {
    const t = std.testing;
    const Route = enum { cache, pool_union, spill };
    try t.expectEqual(Mode.off, session.mode);
    const f = try frame(@src(), .{ 3, Route.spill });
    defer f.end();
    try t.expect(!f.on);
    try t.expectEqual(Route.pool_union, choose(@src(), Route, .pool_union));
    try t.expectEqual(@as(usize, 900), try bound(900, 16));
}

test "a recording plans a data-sized ask at its bound and a replay refuses a size past the bound by name" {
    const t = std.testing;
    var rec = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer rec.deinit();
    var driver: Driver = .{ .gpa = t.allocator };
    defer driver.deinit();
    session = .{ .mode = .dry, .recorder = &rec, .driver = &driver };
    defer session = .{};
    const n = try bound(3, 48);
    try t.expectEqual(@as(usize, 48), n);
    {
        const f = try each(@src(), .{});
        defer f.end();
        rec.free(try rec.ask(n * 8, false, false));
    }
    var tr = try rec.end();
    defer tr.deinit();
    var shapes = try shape.merge(t.allocator, &.{tr}, .{ .alignment = 8 });
    defer shapes.deinit();
    var tables = try pack.pack(t.allocator, &shapes, .{});
    defer tables.deinit();
    try t.expectEqual(@as(u64, 48 * 8), tables.g);

    var b = Binder.init(t.allocator, .{ .g = tables.g, .h = tables.h, .alignment = 8 });
    defer b.deinit();
    session = .{ .mode = .replay, .binder = &b };
    try b.begin(&tables, .{ .site = 1 }, false, 0);
    try t.expectEqual(@as(usize, 3), try bound(3, 48));
    try t.expectError(error.PlanMiss, bound(49, 48));
    try t.expectEqual(binder_mod.Miss{ .kind = .bound, .id = .{ .site = 1 }, .asked = 49, .cap = 48 }, b.miss.?);
    b.abort();
}

test "a real recording takes the data's choice and size, driver or none, and logs its frames" {
    const t = std.testing;
    const Route = enum { cache, pool_union, spill };
    var rec = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer rec.deinit();
    var driver: Driver = .{ .gpa = t.allocator };
    defer driver.deinit();
    session = .{ .mode = .real, .recorder = &rec, .driver = &driver };
    defer session = .{};
    try t.expectEqual(Route.spill, choose(@src(), Route, .spill));
    try t.expectEqual(@as(usize, 3), try bound(3, 48));
    try t.expectEqual(@as(usize, 0), driver.sites.count());
    {
        const f = try frame(@src(), .{});
        defer f.end();
        try t.expect(f.on);
    }
    try t.expectEqual(@as(usize, 2), rec.instances.items.len);
    var tr = try rec.end();
    tr.deinit();
}

test "the driver records every value of a choice that only another choice leads to" {
    const t = std.testing;
    const A = enum { plain, routed };
    const B = enum { cache, pool_union, spill };
    var driver: Driver = .{ .gpa = t.allocator };
    defer driver.deinit();
    session = .{ .mode = .dry, .driver = &driver };
    defer session = .{};
    var seen: std.AutoHashMapUnmanaged([3]u8, void) = .empty;
    defer seen.deinit(t.allocator);
    var passes: usize = 0;
    while (true) {
        passes += 1;
        // B exists only on the routed arm, and the last choice only under a spill.
        const a = choose(@src(), A, .plain);
        var b: u8 = 9;
        var c: u8 = 9;
        if (a == .routed) for (0..2) |_| {
            const route = choose(@src(), B, .cache);
            b = @backingInt(route);
            if (route == .spill) c = @intFromBool(choose(@src(), bool, false));
        };
        try seen.put(t.allocator, .{ @backingInt(a), b, c }, {});
        if (!try driver.next()) break;
    }
    for ([_][3]u8{ .{ 0, 9, 9 }, .{ 1, 0, 9 }, .{ 1, 1, 9 }, .{ 1, 2, 0 }, .{ 1, 2, 1 } }) |want| try t.expect(seen.contains(want));
    // One pass a value beyond each site's first: 1 + 1 + 2 + 1.
    try t.expectEqual(@as(usize, 5), passes);
}
