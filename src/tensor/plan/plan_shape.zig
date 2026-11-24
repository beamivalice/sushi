//! The merge: every instance of a frame that the probes recorded becomes one shape, the structure they share with the
//! largest sizes and the latest deaths any of them showed (docs/bill_redesign.md#the-plan). Two instances of one
//! `(site, key)` that differ in structure are an error naming the frame. Pure: the host tier builds it.
//!
//! A shape's timeline is its nodes in order, a loop unrolled to its longest run: an ask and an exact child take one
//! position, a group one per iteration. A death is the position before which the bytes are dead.
const std = @import("std");
const frame = @import("plan_frame.zig");
const trace_mod = @import("plan_trace.zig");
const FrameId = frame.FrameId;
const Trace = trace_mod.Trace;
const Instance = trace_mod.Instance;
const none = trace_mod.none;

pub const Error = error{ OutOfMemory, PlanAmbiguous };

pub const Group = struct { site: frame.Site, max: u32, keys: std.ArrayList(frame.Key) = .empty };

pub const Node = union(enum) {
    ask: u32,
    exact: FrameId,
    group: Group,
};

pub const AskShape = struct {
    cap: u64 = 0,
    host: bool,
    external: bool,
    lift: u8 = 0,
    /// In this shape's timeline; read only where `lift` is 0.
    death: u32 = 0,
    first: u32 = none,
    last: u32 = 0,

    /// Whether the arena holds it: neither the host ring's nor a state owner's.
    pub fn placed(a: AskShape) bool {
        return !a.host and !a.external;
    }
};

/// One child of the unrolled timeline: an exact child, or one iteration of a group.
pub const Slot = struct {
    node: u32,
    iter: u32,
    pos: u32,
    /// Its exports' first entry in `Shape.through`, and how many.
    through: u32,
    exports: u32,
};

/// What a child hands up, as its parent sees it.
pub const Through = struct {
    /// Levels still to travel: 1 is placed in this frame, more passes through to its parent.
    left: u8,
    cap: u64 = 0,
    death: u32 = 0,
    first: u32 = none,
    last: u32 = 0,
};

/// What a frame hands up: one of its own asks, or what a child handed it.
pub const Export = struct {
    own: bool,
    /// An ordinal when `own`, else an index of `Shape.through`.
    index: u32,
    left: u8,
};

pub const Shape = struct {
    id: FrameId,
    nodes: std.ArrayList(Node) = .empty,
    asks: std.ArrayList(AskShape) = .empty,
    /// The position of each node; a group's is its first iteration's.
    pos: []u32 = &.{},
    len: u32 = 0,
    slots: []Slot = &.{},
    through: []Through = &.{},
    exports: []Export = &.{},
    /// An ask's index in `exports`, or `none`; likewise each entry of `through`.
    own_export: []u32 = &.{},
    through_export: []u32 = &.{},
    /// Per node, the levels of the dispatches inside an exact child or a group; `lo > hi` when none.
    node_lo: []u32 = &.{},
    node_hi: []u32 = &.{},
    state: enum { open, visiting, done } = .open,

    fn deinit(s: *Shape, gpa: std.mem.Allocator) void {
        for (s.nodes.items) |*n| switch (n.*) {
            .group => |*g| g.keys.deinit(gpa),
            else => {},
        };
        s.nodes.deinit(gpa);
        s.asks.deinit(gpa);
        inline for (.{ "pos", "slots", "through", "exports", "own_export", "through_export", "node_lo", "node_hi" }) |f| gpa.free(@field(s, f));
    }

    /// A node's positions: one, or a group's longest run.
    pub fn span(s: *const Shape, node: u32) u32 {
        return switch (s.nodes.items[node]) {
            .group => |g| g.max,
            else => 1,
        };
    }

    /// The slot of iteration `iter` of the child at `node`.
    pub fn slotOf(s: *const Shape, node: u32, iter: u32) u32 {
        for (s.slots, 0..) |slot, i| if (slot.node == node and slot.iter == iter) return @intCast(i);
        unreachable;
    }
};

pub const Why = enum {
    /// Two instances disagree on what comes next: an ask against a child, or two different exact children.
    structure,
    /// An ask is host-filled, or state-owned, in one instance and not in another.
    flags,
    /// An ask outlives more frames than one of its instances has above it.
    lift,
    /// Two keys of one loop hand up a different number of results.
    exports,
    /// A frame site opens inside itself.
    recursion,
};

pub const Ambiguity = struct { id: FrameId, why: Why, node: u32 = 0 };

pub const Shapes = struct {
    gpa: std.mem.Allocator,
    list: std.ArrayList(Shape) = .empty,
    index: std.AutoHashMapUnmanaged(FrameId, u32) = .empty,
    roots: std.ArrayList(FrameId) = .empty,
    alignment: u32,
    refused: ?*Ambiguity = null,

    pub fn deinit(s: *Shapes) void {
        for (s.list.items) |*shape| shape.deinit(s.gpa);
        s.list.deinit(s.gpa);
        s.index.deinit(s.gpa);
        s.roots.deinit(s.gpa);
        s.* = undefined;
    }

    pub fn get(s: *const Shapes, id: FrameId) ?*Shape {
        return &s.list.items[s.index.get(id) orelse return null];
    }

    fn refuse(s: *Shapes, id: FrameId, why: Why, node: usize) Error {
        if (s.refused) |out| out.* = .{ .id = id, .why = why, .node = @intCast(node) };
        return error.PlanAmbiguous;
    }
};

pub const Options = struct {
    alignment: u32 = 256,
    /// Filled when `merge` returns `PlanAmbiguous`: the frame and what its instances disagree on.
    refused: ?*Ambiguity = null,
};

pub fn alignUp(bytes: u64, alignment: u32) u64 {
    return (bytes + alignment - 1) / alignment * alignment;
}

/// One node of an instance, its loop runs joined: the kind, and for a child its first event.
const Seen = struct { node: Node, event: u32 };

fn seenOf(gpa: std.mem.Allocator, t: *const Trace, inst: *const Instance, out: *std.ArrayList(Seen)) Error!void {
    out.clearRetainingCapacity();
    for (inst.events.items, 0..) |ev, e| switch (ev) {
        .ask => |o| try out.append(gpa, .{ .node = .{ .ask = o }, .event = @intCast(e) }),
        .child => |c| {
            const child = &t.instances[c];
            if (child.kind == .exact) {
                try out.append(gpa, .{ .node = .{ .exact = child.id }, .event = @intCast(e) });
                continue;
            }
            // Consecutive iterations of one site are one run.
            if (out.items.len > 0 and e > 0 and inst.events.items[e - 1] == .child) {
                const last = &out.items[out.items.len - 1];
                if (last.node == .group and last.node.group.site == child.id.site) {
                    last.node.group.max += 1;
                    continue;
                }
            }
            try out.append(gpa, .{ .node = .{ .group = .{ .site = child.id.site, .max = 1 } }, .event = @intCast(e) });
        },
    };
}

fn sameGroup(a: Node, b: Node) bool {
    return a == .group and b == .group and a.group.site == b.group.site;
}

/// Lines `seen` up with the shape's nodes: `at[i]` is the node of `seen[i]`. With `grow`, a loop the shape has not
/// met is added and a longer run widens its group; without it the shape is final and holds every instance.
fn lineUp(s: *Shapes, shape: *Shape, seen: []const Seen, at: []u32, grow: bool) Error!void {
    var n: usize = 0;
    for (seen, 0..) |one, i| {
        while (true) {
            if (n < shape.nodes.items.len) {
                const have = &shape.nodes.items[n];
                if (sameGroup(have.*, one.node)) {
                    if (grow) have.group.max = @max(have.group.max, one.node.group.max);
                    break;
                }
                const match = switch (one.node) {
                    .ask => |o| have.* == .ask and have.ask == o,
                    .exact => |id| have.* == .exact and have.exact.eql(id),
                    .group => false,
                };
                if (match) break;
                // A loop that ran no iteration here leaves no event: step over it.
                if (have.* == .group) {
                    n += 1;
                    continue;
                }
            }
            if (one.node != .group or !grow) return s.refuse(shape.id, .structure, n);
            try shape.nodes.insert(s.gpa, n, .{ .group = .{ .site = one.node.group.site, .max = one.node.group.max } });
            break;
        }
        at[i] = @intCast(n);
        n += 1;
    }
    for (shape.nodes.items[n..], n..) |rest, k| if (rest != .group) return s.refuse(shape.id, .structure, k);
}

/// Every instance of every trace, merged by `(site, key)`.
pub fn merge(gpa: std.mem.Allocator, traces: []const Trace, options: Options) Error!Shapes {
    var s: Shapes = .{ .gpa = gpa, .alignment = options.alignment, .refused = options.refused };
    errdefer s.deinit();
    var seen: std.ArrayList(Seen) = .empty;
    defer seen.deinit(gpa);
    var at: std.ArrayList(u32) = .empty;
    defer at.deinit(gpa);

    // The skeletons, the flags and the lifts.
    for (traces) |*t| {
        const root = t.instances[0].id;
        for (s.roots.items) |r| {
            if (r.eql(root)) break;
        } else try s.roots.append(gpa, root);
        for (t.instances) |*inst| {
            try seenOf(gpa, t, inst, &seen);
            const got = try s.index.getOrPut(gpa, inst.id);
            if (!got.found_existing) {
                got.value_ptr.* = @intCast(s.list.items.len);
                try s.list.append(gpa, .{ .id = inst.id });
            }
            const shape = &s.list.items[got.value_ptr.*];
            try at.resize(gpa, seen.items.len);
            if (!got.found_existing) {
                for (seen.items, 0..) |one, i| {
                    try shape.nodes.append(gpa, one.node);
                    at.items[i] = @intCast(i);
                }
                for (inst.asks.items) |a| try shape.asks.append(gpa, .{ .host = a.host, .external = a.external });
            } else {
                if (inst.asks.items.len != shape.asks.items.len) return s.refuse(inst.id, .structure, shape.nodes.items.len);
                try lineUp(&s, shape, seen.items, at.items, true);
            }
            for (inst.asks.items, shape.asks.items) |a, *as| {
                if (a.host != as.host or a.external != as.external) return s.refuse(inst.id, .flags, 0);
                as.lift = @max(as.lift, a.lift);
            }
            // The keys each loop ran under here.
            for (seen.items, at.items) |one, n| if (one.node == .group) {
                const group = &shape.nodes.items[n].group;
                var e = one.event;
                while (e < inst.events.items.len and inst.events.items[e] == .child) : (e += 1) {
                    const child = &t.instances[inst.events.items[e].child];
                    if (child.kind != .each or child.id.site != group.site) break;
                    for (group.keys.items) |k| {
                        if (k == child.id.key) break;
                    } else try group.keys.append(gpa, child.id.key);
                }
            };
        }
    }

    // Positions, slots, and what each frame hands up.
    for (0..s.list.items.len) |i| try finish(&s, @intCast(i));

    // Sizes, deaths and levels, each at the frame that owns the bytes.
    var pos: std.ArrayList([]u32) = .empty;
    defer {
        for (pos.items) |p| gpa.free(p);
        pos.deinit(gpa);
    }
    var slot: std.ArrayList([]u32) = .empty;
    defer {
        for (slot.items) |p| gpa.free(p);
        slot.deinit(gpa);
    }
    for (traces) |*t| {
        for (pos.items) |p| gpa.free(p);
        pos.clearRetainingCapacity();
        for (slot.items) |p| gpa.free(p);
        slot.clearRetainingCapacity();
        // Where each event of each instance falls in its shape's timeline, and each child event's slot.
        for (t.instances) |*inst| {
            const shape = s.get(inst.id).?;
            try seenOf(gpa, t, inst, &seen);
            try at.resize(gpa, seen.items.len);
            try lineUp(&s, shape, seen.items, at.items, false);
            // Room first: once each goes into its list the list's frees are its only ones.
            try pos.ensureUnusedCapacity(gpa, 1);
            try slot.ensureUnusedCapacity(gpa, 1);
            const p = try gpa.alloc(u32, inst.events.items.len + 1);
            errdefer gpa.free(p);
            const sl = try gpa.alloc(u32, inst.events.items.len);
            errdefer gpa.free(sl);
            @memset(sl, none);
            for (seen.items, at.items, 0..) |one, n, i| {
                const until = if (i + 1 < seen.items.len) seen.items[i + 1].event else inst.events.items.len;
                for (one.event..until, 0..) |e, iter| {
                    p[e] = shape.pos[n] + @as(u32, @intCast(iter));
                    if (one.node != .ask) sl[e] = shape.slotOf(n, @intCast(iter));
                }
            }
            p[inst.events.items.len] = shape.len;
            pos.appendAssumeCapacity(p);
            slot.appendAssumeCapacity(sl);
        }
        for (t.instances, 0..) |*inst, ii| {
            const shape = s.get(inst.id).?;
            if (inst.parent != none and inst.lo <= inst.hi) {
                const parent = &t.instances[inst.parent];
                const ps = s.get(parent.id).?;
                const n = ps.slots[slot.items[inst.parent][inst.at]].node;
                ps.node_lo[n] = @min(ps.node_lo[n], inst.lo -| parent.level_in);
                ps.node_hi[n] = @max(ps.node_hi[n], inst.hi -| parent.level_in);
            }
            for (inst.asks.items, shape.asks.items, 0..) |a, *as, o| {
                as.cap = @max(as.cap, a.bytes);
                if (!as.placed()) continue;
                if (as.lift == 0) {
                    as.death = @max(as.death, pos.items[ii][a.death]);
                    if (a.first <= a.last) {
                        as.first = @min(as.first, a.first -| inst.level_in);
                        as.last = @max(as.last, a.last -| inst.level_in);
                    }
                    continue;
                }
                // Up to its owner: each frame on the way names it by what its child handed up. A loop holds what every
                // key of it hands up as far as the key that lifts it furthest: the walk goes on while the frame passes
                // it on, not only to the ask's own lift.
                var cur: u32 = @intCast(ii);
                var e = shape.own_export[o];
                var level: u8 = 1;
                while (true) : (level += 1) {
                    const below = &t.instances[cur];
                    if (below.parent == none) return s.refuse(inst.id, .lift, o);
                    const parent = &t.instances[below.parent];
                    const ps = s.get(parent.id).?;
                    const ti = ps.slots[slot.items[below.parent][below.at]].through + e;
                    if (ps.through[ti].left > 1) {
                        e = ps.through_export[ti];
                        cur = below.parent;
                        continue;
                    }
                    const th = &ps.through[ti];
                    th.cap = @max(th.cap, a.bytes);
                    // An instance that died below its owner died inside the child the owner was running.
                    const died = if (a.lift == level) pos.items[below.parent][a.death] else pos.items[below.parent][below.at] + 1;
                    th.death = @max(th.death, died);
                    if (a.first <= a.last) {
                        th.first = @min(th.first, a.first -| parent.level_in);
                        th.last = @max(th.last, a.last -| parent.level_in);
                    }
                    break;
                }
            }
        }
    }

    for (s.list.items) |*shape| settle(shape);
    return s;
}

/// The positions, slots and exports of shape `i`, its children's first.
fn finish(s: *Shapes, i: u32) Error!void {
    const gpa = s.gpa;
    if (s.list.items[i].state == .done) return;
    if (s.list.items[i].state == .visiting) return s.refuse(s.list.items[i].id, .recursion, 0);
    s.list.items[i].state = .visiting;
    // A child's shape may sit anywhere in the list, and finishing it appends nothing: `shape` stays valid.
    const shape = &s.list.items[i];
    const nodes = shape.nodes.items;
    shape.pos = try gpa.alloc(u32, nodes.len);
    shape.node_lo = try gpa.alloc(u32, nodes.len);
    shape.node_hi = try gpa.alloc(u32, nodes.len);
    @memset(shape.node_lo, none);
    @memset(shape.node_hi, 0);
    var slots: std.ArrayList(Slot) = .empty;
    errdefer slots.deinit(gpa);
    var through: std.ArrayList(Through) = .empty;
    errdefer through.deinit(gpa);
    var exports: std.ArrayList(Export) = .empty;
    errdefer exports.deinit(gpa);
    var lefts: std.ArrayList(u8) = .empty;
    defer lefts.deinit(gpa);

    shape.own_export = try gpa.alloc(u32, shape.asks.items.len);
    for (shape.asks.items, 0..) |a, o| {
        shape.own_export[o] = none;
        if (!a.placed() or a.lift == 0) continue;
        shape.own_export[o] = @intCast(exports.items.len);
        try exports.append(gpa, .{ .own = true, .index = @intCast(o), .left = a.lift });
    }
    var at: u32 = 0;
    for (nodes, 0..) |node, n| {
        shape.pos[n] = at;
        at += shape.span(@intCast(n));
        // What the child at this node hands up: one frame's, or what every key of a loop agrees on.
        lefts.clearRetainingCapacity();
        switch (node) {
            .ask => continue,
            .exact => |id| {
                const c = s.index.get(id).?;
                try finish(s, c);
                for (s.list.items[c].exports) |x| try lefts.append(gpa, x.left);
            },
            .group => |g| for (g.keys.items, 0..) |key, k| {
                const c = s.index.get(.{ .site = g.site, .key = key }).?;
                try finish(s, c);
                const theirs = s.list.items[c].exports;
                if (k == 0) {
                    for (theirs) |x| try lefts.append(gpa, x.left);
                } else {
                    if (theirs.len != lefts.items.len) return s.refuse(shape.id, .exports, n);
                    for (theirs, lefts.items) |x, *l| l.* = @max(l.*, x.left);
                }
            },
        }
        for (0..shape.span(@intCast(n))) |iter| {
            try slots.append(gpa, .{ .node = @intCast(n), .iter = @intCast(iter), .pos = shape.pos[n] + @as(u32, @intCast(iter)), .through = @intCast(through.items.len), .exports = @intCast(lefts.items.len) });
            for (lefts.items) |left| {
                if (left > 1) try exports.append(gpa, .{ .own = false, .index = @intCast(through.items.len), .left = left - 1 });
                try through.append(gpa, .{ .left = left });
            }
        }
    }
    shape.len = at;
    shape.slots = try slots.toOwnedSlice(gpa);
    shape.through = try through.toOwnedSlice(gpa);
    shape.exports = try exports.toOwnedSlice(gpa);
    shape.through_export = try gpa.alloc(u32, shape.through.len);
    @memset(shape.through_export, none);
    for (shape.exports, 0..) |x, e| if (!x.own) {
        shape.through_export[x.index] = @intCast(e);
    };
    shape.state = .done;
}

/// What makes a shape hold for every run of its loops, whatever their counts: a loop's results are one size and are
/// held to the loop's end, and nothing of the frame's ends inside a loop.
fn settle(shape: *Shape) void {
    for (shape.nodes.items, 0..) |node, n| {
        if (node != .group) continue;
        const start = shape.pos[n];
        const end = start + node.group.max;
        const first = shape.slotOf(@intCast(n), 0);
        const per = shape.slots[first].exports;
        for (0..per) |e| {
            var all: Through = .{ .left = shape.through[shape.slots[first].through + e].left };
            for (0..node.group.max) |iter| {
                const th = shape.through[shape.slots[first + iter].through + e];
                all.cap = @max(all.cap, th.cap);
                all.death = @max(all.death, th.death);
                all.first = @min(all.first, th.first);
                all.last = @max(all.last, th.last);
            }
            all.death = @max(all.death, end);
            for (0..node.group.max) |iter| shape.through[shape.slots[first + iter].through + e] = all;
        }
        for (shape.asks.items) |*a| if (a.death > start and a.death < end) {
            a.death = end;
        };
        for (shape.through) |*th| if (th.death > start and th.death < end) {
            th.death = end;
        };
    }
}

const Recorder = trace_mod.Recorder;

fn loopOf(r: *Recorder, site: frame.Site, sizes: []const u64) !void {
    for (sizes) |bytes| {
        try r.enter(.{ .site = site }, .each);
        const a = try r.ask(bytes, false, false);
        r.free(a);
        r.exit();
    }
}

test "two instances of one frame that differ in their asks are ambiguous, and the refusal names the frame" {
    const t = std.testing;
    var a = try Recorder.begin(t.allocator, .{ .site = 1, .key = 7 });
    defer a.deinit();
    a.free(try a.ask(8, false, false));
    a.free(try a.ask(8, false, false));
    var ta = try a.end();
    defer ta.deinit();
    var b = try Recorder.begin(t.allocator, .{ .site = 1, .key = 7 });
    defer b.deinit();
    b.free(try b.ask(8, false, false));
    var tb = try b.end();
    defer tb.deinit();
    // A flag that differs is as much a second structure as a count that does.
    var c = try Recorder.begin(t.allocator, .{ .site = 1, .key = 7 });
    defer c.deinit();
    c.free(try c.ask(8, true, false));
    var tc = try c.end();
    defer tc.deinit();

    var why: Ambiguity = undefined;
    try t.expectError(error.PlanAmbiguous, merge(t.allocator, &.{ ta, tb }, .{ .refused = &why }));
    try t.expectEqual(Ambiguity{ .id = .{ .site = 1, .key = 7 }, .why = .structure, .node = 2 }, why);
    try t.expectError(error.PlanAmbiguous, merge(t.allocator, &.{ tb, tc }, .{ .refused = &why }));
    try t.expectEqual(Why.flags, why.why);
}

test "a loop is one group sized by its longest run, and a run that skips the loop fits the same shape" {
    const t = std.testing;
    var a = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer a.deinit();
    a.free(try a.ask(100, false, false));
    a.free(try a.ask(8, false, false));
    var ta = try a.end();
    defer ta.deinit();
    var b = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer b.deinit();
    const head = try b.ask(40, false, false);
    try loopOf(&b, 5, &.{ 10, 30, 20 });
    b.free(head);
    b.free(try b.ask(16, false, false));
    var tb = try b.end();
    defer tb.deinit();

    var s = try merge(t.allocator, &.{ ta, tb }, .{});
    defer s.deinit();
    const root = s.get(.{ .site = 1 }).?;
    try t.expectEqual(@as(usize, 3), root.nodes.items.len);
    try t.expectEqual(@as(u32, 3), root.nodes.items[1].group.max);
    try t.expectEqual(@as(u32, 5), root.len);
    try t.expectEqualSlices(u32, &.{ 0, 1, 4 }, root.pos);
    // Sizes are the largest seen; a death is the latest: the head lived across the loop in one run.
    try t.expectEqual(@as(u64, 100), root.asks.items[0].cap);
    try t.expectEqual(@as(u32, 4), root.asks.items[0].death);
    try t.expectEqual(@as(u64, 16), root.asks.items[1].cap);
    try t.expectEqual(@as(u64, 30), s.get(.{ .site = 5 }).?.asks.items[0].cap);
    try t.expectEqual(@as(usize, 1), s.roots.items.len);
}

test "a result a loop iteration hands up has one size for every iteration and is held to the end of the loop" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    var kept: [3]trace_mod.Ref = undefined;
    for (&kept, [_]u64{ 64, 256, 128 }, 0..) |*k, bytes, i| {
        try r.enter(.{ .site = 5 }, .each);
        k.* = try r.ask(bytes, false, false);
        // The loop frees each result one iteration later: a carried value.
        if (i > 0) r.free(kept[i - 1]);
        r.exit();
    }
    r.free(kept[2]);
    var tr = try r.end();
    defer tr.deinit();
    var s = try merge(t.allocator, &.{tr}, .{});
    defer s.deinit();
    const root = s.get(.{ .site = 1 }).?;
    try t.expectEqual(@as(usize, 3), root.through.len);
    for (root.through) |th| {
        try t.expectEqual(@as(u8, 1), th.left);
        try t.expectEqual(@as(u64, 256), th.cap);
        try t.expectEqual(@as(u32, 3), th.death);
    }
    try t.expectEqual(@as(u8, 1), s.get(.{ .site = 5 }).?.asks.items[0].lift);
}

test "a loop whose first iteration's result dies in the next and whose last outlives the loop holds both where the loop's results are held, each at its size" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    try r.enter(.{ .site = 2 }, .exact);
    // The first iteration (its own key) carries a sum the second frees; the second's outlives the loop.
    try r.enter(.{ .site = 5, .key = 0 }, .each);
    const first = try r.ask(512, false, false);
    r.exit();
    try r.enter(.{ .site = 5, .key = 1 }, .each);
    const last = try r.ask(512, false, false);
    r.free(first);
    r.exit();
    r.exit();
    r.free(last);
    var tr = try r.end();
    defer tr.deinit();
    var s = try merge(t.allocator, &.{tr}, .{});
    defer s.deinit();
    // Both iterations' results pass through the loop's frame to the root, which holds them at their size.
    const root = s.get(.{ .site = 1 }).?;
    try t.expectEqual(@as(usize, 2), root.through.len);
    for (root.through) |th| {
        try t.expectEqual(@as(u8, 1), th.left);
        try t.expectEqual(@as(u64, 512), th.cap);
    }
}

test "an ask that outlives two frames is placed two levels up and only passes through the frame between" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    try r.enter(.{ .site = 2 }, .exact);
    try r.enter(.{ .site = 3 }, .exact);
    const far = try r.ask(512, false, false);
    r.exit();
    r.exit();
    const after = try r.ask(8, false, false);
    r.free(far);
    r.free(after);
    var tr = try r.end();
    defer tr.deinit();
    var s = try merge(t.allocator, &.{tr}, .{});
    defer s.deinit();
    const inner = s.get(.{ .site = 3 }).?;
    const mid = s.get(.{ .site = 2 }).?;
    const root = s.get(.{ .site = 1 }).?;
    try t.expectEqual(@as(u8, 2), inner.asks.items[0].lift);
    try t.expectEqual(@as(u8, 2), mid.through[0].left);
    try t.expectEqual(@as(u64, 0), mid.through[0].cap);
    try t.expectEqual(@as(usize, 1), mid.exports.len);
    try t.expectEqual(@as(u8, 1), root.through[0].left);
    try t.expectEqual(@as(u64, 512), root.through[0].cap);
    // Born at the root's first position (the child), dead before its end: after the ask beside it.
    try t.expectEqual(@as(u32, 2), root.through[0].death);
}

test "an ask of the frame that dies inside a loop is held to the end of the loop" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    const outer = try r.ask(64, false, false);
    for (0..3) |i| {
        try r.enter(.{ .site = 5 }, .each);
        if (i == 0) r.free(outer);
        r.exit();
    }
    var tr = try r.end();
    defer tr.deinit();
    var s = try merge(t.allocator, &.{tr}, .{});
    defer s.deinit();
    try t.expectEqual(@as(u32, 4), s.get(.{ .site = 1 }).?.asks.items[0].death);
}

test "two keys of one loop that hand up a different number of results are refused" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    try r.enter(.{ .site = 5, .key = 1 }, .each);
    const one = try r.ask(8, false, false);
    r.exit();
    try r.enter(.{ .site = 5, .key = 2 }, .each);
    r.exit();
    r.free(one);
    var tr = try r.end();
    defer tr.deinit();
    try t.expectError(error.PlanAmbiguous, merge(t.allocator, &.{tr}, .{}));
}
