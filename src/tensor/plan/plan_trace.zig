//! What one execution asked the engine for, as the recorder saw it in dry mode: a tree of frame instances, each with
//! its asks and child frames in order, and for every ask when it died and which frame was open then
//! (docs/bill_redesign.md#recorder). Pure: the host tier builds it.
const std = @import("std");
const frame = @import("plan_frame.zig");
const FrameId = frame.FrameId;
const Kind = frame.Kind;

pub const none = std.math.maxInt(u32);

pub const Error = error{ OutOfMemory, PlanEscape };

pub const Ask = struct {
    bytes: u64,
    /// The host fills it: it comes from the host ring, never from scratch.
    host: bool,
    /// `plan.into` named its bytes: a state owner's, not the arena's.
    external: bool,
    /// Frame levels it outlived: its owner is the instance `lift` parents up from its home.
    lift: u8 = 0,
    /// The owner's event count when it died: it is dead before the owner's event of that index. `none` while alive.
    death: u32 = none,
    /// The levels of the first and the last dispatch that bound it; `first > last` when none did.
    first: u32 = none,
    last: u32 = 0,
    /// Its place among the recording's deaths, in the order they came; `none` while alive.
    died: u32 = none,
};

/// An eval the recording made before event `at` of `instance`, once `deaths` asks had died: in a wide execution the
/// host asks dead by then give their bytes back (`plan_region`).
pub const Wait = struct { instance: u32, at: u32, deaths: u32 };

/// A `plan.choose` a dry pass met: its site, the value it took, how many scratch asks came before it, and the instance it
/// was made in with that instance's event count then (a recording read back from the plan cache has the root's 0, 0).
pub const Choice = struct { site: frame.Site, value: u32, at: u32, frame: u32 = 0, event: u32 = 0 };

pub const Event = union(enum) {
    /// The ask of this ordinal.
    ask: u32,
    /// A child instance, entered and left here.
    child: u32,
};

pub const Instance = struct {
    id: FrameId,
    kind: Kind,
    parent: u32,
    /// Its event index in `parent`.
    at: u32,
    /// The highest level any dispatch had reached when it opened.
    level_in: u32,
    /// The levels of the dispatches that bound an ask of its subtree while that ask's home was open; `lo > hi` when
    /// none did.
    lo: u32 = none,
    hi: u32 = 0,
    open: bool = true,
    events: std.ArrayList(Event) = .empty,
    asks: std.ArrayList(Ask) = .empty,
};

/// An ask of a trace: its home instance and its ordinal there.
pub const Ref = struct { instance: u32, ordinal: u32 };

pub const Trace = struct {
    gpa: std.mem.Allocator,
    /// The root is 0; a child follows its parent.
    instances: []Instance,
    waits: []Wait = &.{},
    choices: []Choice = &.{},

    pub fn deinit(t: *Trace) void {
        for (t.instances) |*i| {
            i.events.deinit(t.gpa);
            i.asks.deinit(t.gpa);
        }
        t.gpa.free(t.instances);
        t.gpa.free(t.waits);
        t.gpa.free(t.choices);
        t.* = undefined;
    }

    /// The instance `levels` parents up from `instance`, or null past the root.
    pub fn ancestor(t: *const Trace, instance: u32, levels: u32) ?u32 {
        var a = instance;
        for (0..levels) |_| {
            a = t.instances[a].parent;
            if (a == none) return null;
        }
        return a;
    }
};

pub const Recorder = struct {
    gpa: std.mem.Allocator,
    instances: std.ArrayList(Instance) = .empty,
    /// The open instances, the root first.
    stack: std.ArrayList(u32) = .empty,
    level_hi: u32 = 0,
    waits: std.ArrayList(Wait) = .empty,
    deaths: u32 = 0,
    /// The ask `end` refused: alive when the execution ended, and no `plan.into` named it.
    escape: ?Ref = null,
    choices: std.ArrayList(Choice) = .empty,
    /// Scratch asks so far: neither host-filled nor an owner's bytes.
    scratch: u32 = 0,
    choice_oom: bool = false,

    pub fn begin(gpa: std.mem.Allocator, root: FrameId) Error!Recorder {
        var r: Recorder = .{ .gpa = gpa };
        errdefer r.deinit();
        try r.instances.append(gpa, .{ .id = root, .kind = .exact, .parent = none, .at = 0, .level_in = 0 });
        try r.stack.append(gpa, 0);
        return r;
    }

    pub fn deinit(r: *Recorder) void {
        for (r.instances.items) |*i| {
            i.events.deinit(r.gpa);
            i.asks.deinit(r.gpa);
        }
        r.instances.deinit(r.gpa);
        r.stack.deinit(r.gpa);
        r.waits.deinit(r.gpa);
        r.choices.deinit(r.gpa);
        r.* = undefined;
    }

    fn top(r: *Recorder) u32 {
        return r.stack.items[r.stack.items.len - 1];
    }

    pub fn enter(r: *Recorder, id: FrameId, kind: Kind) Error!void {
        const parent = r.top();
        const at: u32 = @intCast(r.instances.items[parent].events.items.len);
        const child: u32 = @intCast(r.instances.items.len);
        try r.instances.append(r.gpa, .{ .id = id, .kind = kind, .parent = parent, .at = at, .level_in = r.level_hi });
        try r.instances.items[parent].events.append(r.gpa, .{ .child = child });
        try r.stack.append(r.gpa, child);
    }

    /// Closes the innermost frame; the root closes at `end`.
    pub fn exit(r: *Recorder) void {
        std.debug.assert(r.stack.items.len > 1);
        r.instances.items[r.stack.pop().?].open = false;
    }

    pub fn ask(r: *Recorder, bytes: u64, host: bool, external: bool) Error!Ref {
        const home = r.top();
        const inst = &r.instances.items[home];
        const ordinal: u32 = @intCast(inst.asks.items.len);
        try inst.asks.append(r.gpa, .{ .bytes = bytes, .host = host, .external = external });
        try inst.events.append(r.gpa, .{ .ask = ordinal });
        if (!host and !external) r.scratch += 1;
        return .{ .instance = home, .ordinal = ordinal };
    }

    /// A `plan.choose` at `site` took `value` here; `end` reports a note that could not be kept.
    pub fn noteChoice(r: *Recorder, site: frame.Site, value: u32) void {
        const in = r.top();
        r.choices.append(r.gpa, .{ .site = site, .value = value, .at = r.scratch, .frame = in, .event = @intCast(r.instances.items[in].events.items.len) }) catch {
            r.choice_oom = true;
        };
    }

    /// The ask died now: its owner is the nearest frame still open at or above its home.
    pub fn free(r: *Recorder, ref: Ref) void {
        var owner = ref.instance;
        var lift: u8 = 0;
        while (!r.instances.items[owner].open) {
            owner = r.instances.items[owner].parent;
            lift += 1;
        }
        const a = &r.instances.items[ref.instance].asks.items[ref.ordinal];
        std.debug.assert(a.death == none);
        a.lift = lift;
        a.death = @intCast(r.instances.items[owner].events.items.len);
        a.died = r.deaths;
        r.deaths += 1;
    }

    /// The forward evaluated here: every pass before has run. Two with no event between are one.
    pub fn wait(r: *Recorder) Error!void {
        const at: Wait = .{ .instance = r.top(), .at = @intCast(r.instances.items[r.top()].events.items.len), .deaths = r.deaths };
        if (r.waits.items.len > 0) {
            const last = &r.waits.items[r.waits.items.len - 1];
            if (last.instance == at.instance and last.at == at.at) {
                last.deaths = at.deaths;
                return;
            }
        }
        try r.waits.append(r.gpa, at);
    }

    /// A dispatch at `level` of the pass bound the ask.
    pub fn touch(r: *Recorder, ref: Ref, level: u32) void {
        r.level_hi = @max(r.level_hi, level);
        const a = &r.instances.items[ref.instance].asks.items[ref.ordinal];
        a.first = @min(a.first, level);
        a.last = @max(a.last, level);
        var at = ref.instance;
        if (!r.instances.items[at].open) return;
        while (at != none) : (at = r.instances.items[at].parent) {
            const i = &r.instances.items[at];
            i.lo = @min(i.lo, level);
            i.hi = @max(i.hi, level);
        }
    }

    /// The execution ended: its trace, or `PlanEscape` with `escape` set when an ask of the arena is still alive.
    pub fn end(r: *Recorder) Error!Trace {
        std.debug.assert(r.stack.items.len == 1);
        for (r.instances.items, 0..) |inst, i| for (inst.asks.items, 0..) |a, o| {
            if (a.death == none and !a.external) {
                r.escape = .{ .instance = @intCast(i), .ordinal = @intCast(o) };
                return error.PlanEscape;
            }
        };
        if (r.choice_oom) return error.OutOfMemory;
        r.instances.items[0].open = false;
        const waits = try r.waits.toOwnedSlice(r.gpa);
        errdefer r.gpa.free(waits);
        const choices = try r.choices.toOwnedSlice(r.gpa);
        errdefer r.gpa.free(choices);
        const instances = try r.instances.toOwnedSlice(r.gpa);
        r.stack.clearRetainingCapacity();
        return .{ .gpa = r.gpa, .instances = instances, .waits = waits, .choices = choices };
    }
};

/// Where two traces of one execution first part: the instance, what of it differs and, for its events or asks, the
/// index. A dry trace is held to the real one by it, and a recording under two fills to itself.
pub const Difference = struct {
    instance: u32,
    what: enum { instances, frame, events, asks, levels },
    at: u32 = 0,
};

/// Whether two recordings waited at the same places, after as many deaths.
pub fn sameWaits(a: []const Wait, b: []const Wait) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (!std.meta.eql(x, y)) return false;
    return true;
}

pub fn firstDifference(a: []const Instance, b: []const Instance) ?Difference {
    return firstDifferenceOf(a, b, true);
}

/// `firstDifference`, with the dispatch levels compared only where `levels`: a wide execution's places follow its sizes
/// and lifetimes alone (`plan_region`), its floors the marks the engine reads as it runs.
pub fn firstDifferenceOf(a: []const Instance, b: []const Instance, levels: bool) ?Difference {
    if (a.len != b.len) return .{ .instance = @intCast(@min(a.len, b.len)), .what = .instances };
    for (a, b, 0..) |x, y, n| {
        const i: u32 = @intCast(n);
        if (!x.id.eql(y.id) or x.kind != y.kind or x.parent != y.parent or x.at != y.at) return .{ .instance = i, .what = .frame };
        if (levels and (x.level_in != y.level_in or x.lo != y.lo or x.hi != y.hi)) return .{ .instance = i, .what = .levels };
        const events = @min(x.events.items.len, y.events.items.len);
        for (x.events.items[0..events], y.events.items[0..events], 0..) |e, f, k| {
            if (!std.meta.eql(e, f)) return .{ .instance = i, .what = .events, .at = @intCast(k) };
        }
        if (x.events.items.len != y.events.items.len) return .{ .instance = i, .what = .events, .at = @intCast(events) };
        for (x.asks.items, y.asks.items, 0..) |p, q, k| {
            if (p.bytes != q.bytes or p.host != q.host or p.external != q.external or p.lift != q.lift or p.death != q.death)
                return .{ .instance = i, .what = .asks, .at = @intCast(k) };
            if (levels and (p.first != q.first or p.last != q.last)) return .{ .instance = i, .what = .levels, .at = @intCast(k) };
        }
    }
    return null;
}

test "two traces part where an ask, a death, a level or a frame differs, and nowhere when the execution is one" {
    const t = std.testing;
    var traces: [5]Recorder = undefined;
    for (&traces, 0..) |*r, v| {
        r.* = try Recorder.begin(t.allocator, .{ .site = 1 });
        const a = try r.ask(if (v == 1) 72 else 64, false, false);
        try r.enter(.{ .site = 2, .key = if (v == 4) 9 else 0 }, .exact);
        const b = try r.ask(32, true, false);
        r.touch(b, if (v == 3) 4 else 3);
        r.free(b);
        r.exit();
        // The third trace frees the first ask a frame earlier.
        if (v != 2) _ = try r.ask(16, false, false);
        r.free(a);
    }
    defer for (&traces) |*r| r.deinit();
    const base = traces[0].instances.items;
    try t.expectEqual(@as(?Difference, null), firstDifference(base, base));
    try t.expectEqual(Difference{ .instance = 0, .what = .asks, .at = 0 }, firstDifference(base, traces[1].instances.items).?);
    try t.expectEqual(Difference{ .instance = 0, .what = .events, .at = 2 }, firstDifference(base, traces[2].instances.items).?);
    try t.expectEqual(Difference{ .instance = 0, .what = .levels }, firstDifference(base, traces[3].instances.items).?);
    try t.expectEqual(Difference{ .instance = 1, .what = .frame }, firstDifference(base, traces[4].instances.items).?);
}

test "an ask freed after its frame closed belongs to the frame open then, and dies before the next event of that frame" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    try r.enter(.{ .site = 2 }, .exact);
    const kept = try r.ask(64, false, false);
    const scratch = try r.ask(32, false, false);
    r.free(scratch);
    r.exit();
    const beside = try r.ask(16, false, false);
    r.free(kept);
    r.free(beside);
    var trace = try r.end();
    defer trace.deinit();
    const child = trace.instances[1].asks.items;
    // The scratch died in its own frame, before that frame's third event.
    try t.expectEqual(@as(u8, 0), child[1].lift);
    try t.expectEqual(@as(u32, 2), child[1].death);
    // The kept one outlived its frame by one level and died after the root's two events (the child, an ask).
    try t.expectEqual(@as(u8, 1), child[0].lift);
    try t.expectEqual(@as(u32, 2), child[0].death);
    try t.expectEqual(@as(u32, 0), trace.instances[1].at);
    try t.expectEqual(@as(?u32, 0), trace.ancestor(1, 1));
    try t.expectEqual(@as(?u32, null), trace.ancestor(1, 2));
}

test "an execution that ends with an arena ask alive is an escape, by name; a state-owned destination is not" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    _ = try r.ask(8, false, true);
    const left = try r.ask(8, false, false);
    try t.expectError(error.PlanEscape, r.end());
    try t.expectEqual(left, r.escape.?);
    r.free(left);
    var trace = try r.end();
    trace.deinit();
}

test "the level of a dispatch reaches the ask it bound and every frame open around it, not a frame already closed" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    try r.enter(.{ .site = 2 }, .each);
    const a = try r.ask(8, false, false);
    r.touch(a, 3);
    r.touch(a, 5);
    r.exit();
    r.touch(a, 9);
    try r.enter(.{ .site = 2 }, .each);
    r.exit();
    r.free(a);
    var trace = try r.end();
    defer trace.deinit();
    const ask = trace.instances[1].asks.items[0];
    try t.expectEqual(@as(u32, 3), ask.first);
    try t.expectEqual(@as(u32, 9), ask.last);
    try t.expectEqual(@as(u32, 5), trace.instances[1].hi);
    try t.expectEqual(@as(u32, 5), trace.instances[0].hi);
    try t.expectEqual(@as(u32, 9), trace.instances[2].level_in);
    try t.expect(trace.instances[2].lo > trace.instances[2].hi);
}
