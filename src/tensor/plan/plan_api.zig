//! The plan's surface (D97 everywhere): what forward code and engine choke points call.
//! Outside an execution every call here branches on `session.mode` and changes nothing. Pure: host tier.
const std = @import("std");
const trace = @import("plan_trace.zig");

pub const Site = u64;
pub const Recorder = trace.Recorder;
pub const Ref = trace.Ref;
pub const trace_mod = trace;
pub const region = @import("plan_region.zig");

pub const Error = error{ OutOfMemory, PlanNeedsLease, PlanEscape };

pub fn siteOf(comptime src: std.builtin.SourceLocation) Site {
    comptime {
        var h = std.hash.Wyhash.init(0);
        std.hash.autoHash(&h, src.line);
        std.hash.autoHash(&h, src.column);
        h.update(src.file);
        return h.final();
    }
}

pub const Mode = enum {
    /// No planned execution: pool allocation.
    off,
    /// A real execution on the pool, recorded for comparison against dry probes.
    real,
    /// Dry mode: no GPU work runs; each `choose` takes the driver's value and each `bound` its `max`.
    dry,
    /// A planned narrow execution (decode, verify, draft, joined, sample) on `core`'s first-fit arena region.
    narrow,
    /// A planned wide execution (prefill chunk, vision encode) on `core`'s first-fit arena region.
    wide,
};

pub const Session = struct {
    mode: Mode = .off,
    recorder: ?*Recorder = null,
    driver: ?*Driver = null,
};

/// The inference thread's session. The engine sets it when a recording or planned execution begins.
pub var session: Session = .{};

/// True while recording (dry or real).
pub fn recording() bool {
    return session.mode == .real or session.mode == .dry;
}

/// True under any planned mode (recording or arena region execution).
pub fn planned() bool {
    return session.mode != .off;
}

/// True while executing on `core`'s first-fit arena region (`.narrow` or `.wide`).
pub fn arenaActive() bool {
    return session.mode == .narrow or session.mode == .wide;
}

/// A structure branch that data decides: `real` during serving/wide/real-record, and the driver's value in dry probes.
pub fn choose(comptime src: std.builtin.SourceLocation, comptime T: type, real: T) T {
    if (session.mode != .dry) return real;
    const site = comptime siteOf(src);
    const driver = session.driver.?;
    return if (T == bool) driver.pick(site, 2) == 1 else (comptime std.enums.values(T))[driver.pick(site, std.enums.values(T).len)];
}

/// Upper bound hint for dry probes: returns `max` in dry mode and `real` everywhere else.
pub fn bound(real: usize, max: usize) usize {
    return if (session.mode == .dry) max else real;
}

/// Drives dry probe passes so every reachable `choose` value is explored.
pub const Driver = struct {
    gpa: std.mem.Allocator,
    sites: std.AutoArrayHashMapUnmanaged(Site, State) = .empty,
    /// Values pinned for the current pass.
    forced: std.AutoHashMapUnmanaged(Site, u32) = .empty,
    /// Choices taken in the current pass, in encounter order.
    trail: std.ArrayList(Choice) = .empty,
    failed: bool = false,

    pub const Choice = struct { site: Site, value: u32 };
    const State = struct {
        values: u32,
        tried: u64 = 0,
        context: []Choice,
    };

    pub fn deinit(d: *Driver) void {
        for (d.sites.values()) |s| d.gpa.free(s.context);
        d.sites.deinit(d.gpa);
        d.forced.deinit(d.gpa);
        d.trail.deinit(d.gpa);
        d.* = undefined;
    }

    /// Value taken at `site` in this pass, out of `values` (at most 64).
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

    /// Advances to the next unexplored arm, or returns false once all reachable arms ran.
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

test "outside an execution choose and bound are identity" {
    const t = std.testing;
    const Route = enum { cache, pool_union, spill };
    try t.expectEqual(Mode.off, session.mode);
    try t.expect(!arenaActive());
    try t.expectEqual(Route.pool_union, choose(@src(), Route, .pool_union));
    try t.expectEqual(@as(usize, 900), bound(900, 16));
}

test "dry mode returns bound max while real and arena modes return real size" {
    const t = std.testing;
    session = .{ .mode = .dry };
    defer session = .{};
    try t.expectEqual(@as(usize, 48), bound(3, 48));
    session = .{ .mode = .narrow };
    try t.expect(arenaActive());
    try t.expectEqual(@as(usize, 3), bound(3, 48));
    try t.expectEqual(@as(usize, 49), bound(49, 48));
}

test "the driver explores every reachable arm across nested choices" {
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
        const a = choose(@src(), A, .plain);
        var b: u8 = 9;
        var c: u8 = 9;
        if (a == .routed) for (0..2) |_| {
            const route = choose(@src(), B, .cache);
            b = @intFromEnum(route);
            if (route == .spill) c = @intFromBool(choose(@src(), bool, false));
        };
        try seen.put(t.allocator, .{ @intFromEnum(a), b, c }, {});
        if (!try driver.next()) break;
    }
    for ([_][3]u8{ .{ 0, 9, 9 }, .{ 1, 0, 9 }, .{ 1, 1, 9 }, .{ 1, 2, 0 }, .{ 1, 2, 1 } }) |want| try t.expect(seen.contains(want));
    try t.expectEqual(@as(usize, 5), passes);
}
