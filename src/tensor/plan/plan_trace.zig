//! What one execution asked the engine for, as the recorder saw it in dry or real mode: a flat timeline of asks,
//! frees, and GPU waits (D97 everywhere). Pure: the host tier builds it.
const std = @import("std");

pub const none = std.math.maxInt(u32);

pub const Error = error{ OutOfMemory, PlanEscape };

pub const Ask = struct {
    bytes: u64,
    /// The host fills it: it comes from the host ring, never from scratch.
    host: bool,
    /// `plan.into` named its bytes: a state owner's, not the arena's.
    external: bool,
    /// Ordinal among deaths when freed; `none` while alive.
    death: u32 = none,
};

pub const Event = union(enum) {
    ask: Ref,
    free: Ref,
    /// An eval the recording made: every ask with `death < deaths` has died and its host-ring bytes may be reclaimed.
    wait: u32,
};

/// An ask of a trace: its index in `asks`.
pub const Ref = u32;

pub const View = struct {
    asks: []const Ask,
    events: []const Event,
};

pub const Trace = struct {
    gpa: std.mem.Allocator,
    asks: []Ask,
    events: []Event,

    pub fn view(t: *const Trace) View {
        return .{ .asks = t.asks, .events = t.events };
    }

    pub fn deinit(t: *Trace) void {
        t.gpa.free(t.asks);
        t.gpa.free(t.events);
        t.* = undefined;
    }
};

pub const Recorder = struct {
    gpa: std.mem.Allocator,
    asks: std.ArrayList(Ask) = .empty,
    events: std.ArrayList(Event) = .empty,
    deaths: u32 = 0,
    /// The ask `end` refused: alive when the execution ended, and no `into` named it.
    escape: ?Ref = null,
    oom: bool = false,

    pub fn begin(gpa: std.mem.Allocator) Recorder {
        return .{ .gpa = gpa };
    }

    pub fn deinit(r: *Recorder) void {
        r.asks.deinit(r.gpa);
        r.events.deinit(r.gpa);
        r.* = undefined;
    }

    pub fn view(r: *const Recorder) View {
        return .{ .asks = r.asks.items, .events = r.events.items };
    }

    pub fn ask(r: *Recorder, bytes: u64, host: bool, external: bool) Error!Ref {
        const ordinal: Ref = @intCast(r.asks.items.len);
        try r.asks.append(r.gpa, .{ .bytes = bytes, .host = host, .external = external });
        try r.events.append(r.gpa, .{ .ask = ordinal });
        return ordinal;
    }

    /// Marks `ref` dead now.
    pub fn free(r: *Recorder, ref: Ref) void {
        const a = &r.asks.items[ref];
        std.debug.assert(a.death == none);
        a.death = r.deaths;
        r.deaths += 1;
        r.events.append(r.gpa, .{ .free = ref }) catch {
            r.oom = true;
        };
    }

    /// The forward evaluated here: every pass before has run. Two consecutive waits collapse into one.
    pub fn wait(r: *Recorder) Error!void {
        if (r.events.items.len > 0) switch (r.events.items[r.events.items.len - 1]) {
            .wait => |*deaths| {
                deaths.* = r.deaths;
                return;
            },
            else => {},
        };
        try r.events.append(r.gpa, .{ .wait = r.deaths });
    }

    /// The execution ended: its trace, or `PlanEscape` with `escape` set when an arena ask is still alive.
    pub fn end(r: *Recorder) Error!Trace {
        if (r.oom) return error.OutOfMemory;
        for (r.asks.items, 0..) |a, i| {
            if (a.death == none and !a.external) {
                r.escape = @intCast(i);
                return error.PlanEscape;
            }
        }
        const asks = try r.asks.toOwnedSlice(r.gpa);
        errdefer r.gpa.free(asks);
        const events = try r.events.toOwnedSlice(r.gpa);
        return .{ .gpa = r.gpa, .asks = asks, .events = events };
    }
};

/// Where two traces first part.
pub const Difference = struct {
    what: enum { asks, events },
    at: u32,
};

pub fn firstDifference(a: View, b: View) ?Difference {
    const min_asks = @min(a.asks.len, b.asks.len);
    for (a.asks[0..min_asks], b.asks[0..min_asks], 0..) |x, y, i| {
        if (x.bytes != y.bytes or x.host != y.host or x.external != y.external or x.death != y.death) {
            return .{ .what = .asks, .at = @intCast(i) };
        }
    }
    if (a.asks.len != b.asks.len) return .{ .what = .asks, .at = @intCast(min_asks) };
    const min_events = @min(a.events.len, b.events.len);
    for (a.events[0..min_events], b.events[0..min_events], 0..) |x, y, i| {
        if (!std.meta.eql(x, y)) return .{ .what = .events, .at = @intCast(i) };
    }
    if (a.events.len != b.events.len) return .{ .what = .events, .at = @intCast(min_events) };
    return null;
}

test "two traces part where an ask or an event differs, and nowhere when identical" {
    const t = std.testing;
    var recs: [3]Recorder = undefined;
    for (&recs, 0..) |*r, v| {
        r.* = Recorder.begin(t.allocator);
        const a = try r.ask(if (v == 1) 72 else 64, false, false);
        const b = try r.ask(32, true, false);
        r.free(b);
        if (v == 2) try r.wait();
        r.free(a);
    }
    defer for (&recs) |*r| r.deinit();
    try t.expectEqual(@as(?Difference, null), firstDifference(recs[0].view(), recs[0].view()));
    try t.expectEqual(Difference{ .what = .asks, .at = 0 }, firstDifference(recs[0].view(), recs[1].view()).?);
    try t.expectEqual(Difference{ .what = .events, .at = 3 }, firstDifference(recs[0].view(), recs[2].view()).?);
}

test "consecutive waits collapse to latest deaths and external asks may remain alive at end" {
    const t = std.testing;
    var r = Recorder.begin(t.allocator);
    defer r.deinit();
    _ = try r.ask(8, false, true);
    const left = try r.ask(8, false, false);
    try t.expectError(error.PlanEscape, r.end());
    try t.expectEqual(left, r.escape.?);
    r.free(left);
    try r.wait();
    try r.wait();
    var tr = try r.end();
    defer tr.deinit();
    try t.expectEqual(@as(usize, 2), tr.asks.len);
    try t.expectEqual(@as(usize, 4), tr.events.len);
    try t.expectEqual(Event{ .wait = 1 }, tr.events[3]);
}
