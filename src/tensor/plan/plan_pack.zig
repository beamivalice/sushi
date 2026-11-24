//! The packer: each frame's buffers get offsets in the frame's block, largest first, two sharing bytes only when they
//! never live together (docs/bill_redesign.md#the-plan, "Packing"). Its output, the tables, is everything the binder
//! reads at run time. Pure and deterministic: the host tier builds it.
const std = @import("std");
const frame = @import("plan_frame.zig");
const shape_mod = @import("plan_shape.zig");
const FrameId = frame.FrameId;
const Shape = shape_mod.Shape;
const Shapes = shape_mod.Shapes;
const alignUp = shape_mod.alignUp;
const none = @import("plan_trace.zig").none;

pub const Error = error{OutOfMemory};

pub const Range = struct { at: u32 = 0, len: u32 = 0 };

/// Where one buffer of a frame goes: an offset in the frame's block, the most bytes it may ask for, its number among
/// the frame's buffers, and the buffers that held some of its bytes just before it (`FramePlan.preds`).
pub const Place = struct {
    offset: u64 = 0,
    cap: u64 = 0,
    item: u32 = none,
    preds: Range = .{},
};

pub const AskPlan = struct {
    kind: enum(u2) {
        /// In this frame's block, at `place`.
        here,
        /// In an ancestor's block: export `up` of this frame.
        lifted,
        /// From the host ring.
        host,
        /// A state owner's bytes (`plan.into`).
        external,
    },
    place: Place = .{},
    up: u32 = none,
};

pub const ThroughPlan = struct {
    /// Placed in this frame at `place`; else it is export `up` of this frame.
    here: bool,
    place: Place = .{},
    up: u32 = none,
};

/// A child's block in its parent: every iteration of a group shares one.
pub const ChildPlan = struct {
    /// The first slot of the child; a group's iteration `i` is slot `slot + i`.
    slot: u32,
    offset: u64,
    block: u64,
    /// What the child hands up, the same for every key of a group.
    exports: u32,
    /// What held some of the block's bytes before it (`FramePlan.preds`).
    preds: Range = .{},
};

pub const NodePlan = union(enum) {
    ask: u32,
    exact: struct { id: FrameId, child: ChildPlan },
    group: struct { site: frame.Site, max: u32, child: ChildPlan },
};

pub const FramePlan = struct {
    id: FrameId,
    asks: []AskPlan,
    nodes: []NodePlan,
    /// Per slot, its exports' first entry in `through`.
    slots: []u32,
    through: []ThroughPlan,
    /// A value under `items` is a buffer of this frame; `items + n` is the child block of node `n`.
    preds: []u32,
    items: u32,
    block: u64,
    /// The most bytes alive at once: no packing is smaller.
    peak: u64,
    /// The most host-filled bytes the frame and all inside it ask for: each loop at its longest run of its heaviest
    /// key, since a served run may combine what no one probe ran together.
    host: u64,
    exports: u32,
};

pub const Tables = struct {
    gpa: std.mem.Allocator,
    plans: []FramePlan,
    index: std.AutoHashMapUnmanaged(FrameId, u32),
    roots: []FrameId,
    /// The scratch arena: the largest root block.
    g: u64,
    /// One replica of the host ring.
    h: u64,
    alignment: u32,
    /// Bytes left after each buffer's own (`Options.guard`).
    guard: u64 = 0,

    pub fn deinit(t: *Tables) void {
        for (t.plans) |p| {
            t.gpa.free(p.asks);
            t.gpa.free(p.nodes);
            t.gpa.free(p.slots);
            t.gpa.free(p.through);
            t.gpa.free(p.preds);
        }
        t.gpa.free(t.plans);
        t.gpa.free(t.roots);
        t.index.deinit(t.gpa);
        t.* = undefined;
    }

    pub fn get(t: *const Tables, id: FrameId) ?*const FramePlan {
        return &t.plans[t.index.get(id) orelse return null];
    }
};

pub const Options = struct {
    /// Bytes are shared only where the earlier buffer's last dispatch is below the later one's first, so a reuse never
    /// adds a barrier to a pass.
    level_rule: bool = true,
    /// The largest buffer the level rule keeps apart: a pair either of which is larger shares bytes by lifetimes alone,
    /// the floors ordering the reuse. A large buffer's dispatches are long beside the barrier its reuse takes, and it
    /// is the large ones the rule would hold apart that grow the arena past the lifetimes.
    level_rule_cap: u64 = std.math.maxInt(u64),
    /// Bytes after each buffer that no other buffer takes, rounded up to the alignment: a test build's canary between
    /// places (`core.arena_check`).
    guard: u64 = 0,
};

const Item = struct {
    size: u64,
    /// Alive over the positions `start..end`.
    start: u32,
    end: u32,
    first: u32,
    last: u32,
    what: enum { ask, through, node },
    index: u32,
    /// A loop's block or one of its results: a run in which the loop stops short never makes it.
    optional: bool = false,
    offset: u64 = 0,
    placed: bool = false,
};

fn apart(a: Item, b: Item, options: Options) bool {
    if (a.start < b.end and b.start < a.end) return true;
    if (!options.level_rule or a.size > options.level_rule_cap or b.size > options.level_rule_cap) return false;
    const early = if (a.end <= b.start) a else b;
    const late = if (a.end <= b.start) b else a;
    // A buffer no dispatch bound has no place in a pass to protect.
    if (early.first > early.last or late.first > late.last) return false;
    return early.last >= late.first;
}

fn bySize(_: void, a: Item, b: Item) bool {
    if (a.size != b.size) return a.size > b.size;
    if (a.start != b.start) return a.start < b.start;
    if (a.what != b.what) return @intFromEnum(a.what) < @intFromEnum(b.what);
    return a.index < b.index;
}

fn byOffset(_: void, a: Item, b: Item) bool {
    return a.offset < b.offset;
}

pub fn pack(gpa: std.mem.Allocator, shapes: *const Shapes, options: Options) Error!Tables {
    const n = shapes.list.items.len;
    const plans = try gpa.alloc(FramePlan, n);
    for (plans, shapes.list.items) |*p, shape| p.* = .{ .id = shape.id, .asks = &.{}, .nodes = &.{}, .slots = &.{}, .through = &.{}, .preds = &.{}, .items = 0, .block = 0, .peak = 0, .host = 0, .exports = 0 };
    var opts = options;
    // Every place starts aligned only while every buffer's size is a multiple of the alignment, its guard included.
    opts.guard = alignUp(options.guard, shapes.alignment);
    var t: Tables = .{ .gpa = gpa, .plans = plans, .index = .empty, .roots = &.{}, .g = 0, .h = 0, .alignment = shapes.alignment, .guard = opts.guard };
    errdefer t.deinit();
    const done = try gpa.alloc(bool, n);
    defer gpa.free(done);
    @memset(done, false);
    // A frame's block holds its children's: they are packed first. The merge refused a frame inside itself, so every
    // pass packs at least one.
    var made: usize = 0;
    while (made < n) {
        for (shapes.list.items, 0..) |*shape, i| {
            if (done[i] or !ready(shapes, shape, done)) continue;
            plans[i] = try packOne(gpa, shapes, shape, plans, opts);
            done[i] = true;
            made += 1;
        }
    }
    try t.index.ensureTotalCapacity(gpa, @intCast(n));
    for (shapes.list.items, 0..) |shape, i| t.index.putAssumeCapacity(shape.id, @intCast(i));
    t.roots = try gpa.dupe(FrameId, shapes.roots.items);
    for (t.roots) |r| {
        t.g = @max(t.g, t.get(r).?.block);
        t.h = @max(t.h, t.get(r).?.host);
    }
    return t;
}

fn ready(shapes: *const Shapes, shape: *const Shape, done: []const bool) bool {
    for (shape.nodes.items) |node| switch (node) {
        .ask => {},
        .exact => |id| if (!done[shapes.index.get(id).?]) return false,
        .group => |g| for (g.keys.items) |key| {
            if (!done[shapes.index.get(.{ .site = g.site, .key = key }).?]) return false;
        },
    };
    return true;
}

fn packOne(gpa: std.mem.Allocator, shapes: *const Shapes, shape: *const Shape, plans: []const FramePlan, options: Options) Error!FramePlan {
    const a = shapes.alignment;
    var items: std.ArrayList(Item) = .empty;
    defer items.deinit(gpa);

    const asks = try gpa.alloc(AskPlan, shape.asks.items.len);
    errdefer gpa.free(asks);
    const nodes = try gpa.alloc(NodePlan, shape.nodes.items.len);
    errdefer gpa.free(nodes);
    const slots = try gpa.alloc(u32, shape.slots.len);
    errdefer gpa.free(slots);
    const through = try gpa.alloc(ThroughPlan, shape.through.len);
    errdefer gpa.free(through);
    for (shape.slots, slots) |s, *out| out.* = s.through;

    // The frame's own buffers and what its children handed it number first; the child blocks follow.
    var count: u32 = 0;
    var host: u64 = 0;
    for (shape.nodes.items, 0..) |node, ni| switch (node) {
        .ask => |o| {
            const as = shape.asks.items[o];
            nodes[ni] = .{ .ask = o };
            if (as.host) {
                asks[o] = .{ .kind = .host, .place = .{ .cap = as.cap } };
                host += alignUp(@max(as.cap, 1), a);
            } else if (as.external) {
                asks[o] = .{ .kind = .external, .place = .{ .cap = as.cap } };
            } else if (as.lift > 0) {
                asks[o] = .{ .kind = .lifted, .place = .{ .cap = as.cap }, .up = shape.own_export[o] };
            } else {
                asks[o] = .{ .kind = .here, .place = .{ .cap = as.cap, .item = count } };
                const start = shape.pos[ni];
                try items.append(gpa, .{ .size = alignUp(@max(as.cap, 1), a) + options.guard, .start = start, .end = @max(as.death, start + 1), .first = as.first, .last = as.last, .what = .ask, .index = o });
                count += 1;
            }
        },
        else => {},
    };
    for (shape.slots) |slot| for (0..slot.exports) |e| {
        const ti = slot.through + @as(u32, @intCast(e));
        const th = shape.through[ti];
        if (th.left > 1) {
            through[ti] = .{ .here = false, .place = .{ .cap = th.cap }, .up = shape.through_export[ti] };
            continue;
        }
        through[ti] = .{ .here = true, .place = .{ .cap = th.cap, .item = count } };
        const optional = switch (shape.nodes.items[slot.node]) {
            .group => true,
            .exact => |id| exportOptional(shapes, &shapes.list.items[shapes.index.get(id).?], @intCast(e)),
            .ask => unreachable,
        };
        try items.append(gpa, .{ .size = alignUp(@max(th.cap, 1), a) + options.guard, .start = slot.pos, .end = @max(th.death, slot.pos + 1), .first = th.first, .last = th.last, .what = .through, .index = ti, .optional = optional });
        count += 1;
    };
    for (shape.nodes.items, 0..) |node, ni| {
        const first_slot = switch (node) {
            .ask => continue,
            else => shape.slotOf(@intCast(ni), 0),
        };
        var block: u64 = 0;
        switch (node) {
            .ask => unreachable,
            .exact => |id| {
                block = plans[shapes.index.get(id).?].block;
                host += plans[shapes.index.get(id).?].host;
            },
            .group => |g| {
                var heaviest: u64 = 0;
                for (g.keys.items) |key| {
                    const child = plans[shapes.index.get(.{ .site = g.site, .key = key }).?];
                    block = @max(block, child.block);
                    heaviest = @max(heaviest, child.host);
                }
                host += heaviest * g.max;
            },
        }
        const child: ChildPlan = .{ .slot = first_slot, .offset = 0, .block = block, .exports = shape.slots[first_slot].exports };
        nodes[ni] = switch (node) {
            .ask => unreachable,
            .exact => |id| .{ .exact = .{ .id = id, .child = child } },
            .group => |g| .{ .group = .{ .site = g.site, .max = g.max, .child = child } },
        };
        if (block == 0) continue;
        try items.append(gpa, .{ .size = block, .start = shape.pos[ni], .end = shape.pos[ni] + shape.span(@intCast(ni)), .first = shape.node_lo[ni], .last = shape.node_hi[ni], .what = .node, .index = @intCast(ni), .optional = node == .group });
    }

    // Largest first, each at the lowest offset clear of every placed buffer it must stay apart from.
    std.mem.sort(Item, items.items, {}, bySize);
    var near: std.ArrayList(Item) = .empty;
    defer near.deinit(gpa);
    var block: u64 = 0;
    for (items.items, 0..) |*it, i| {
        near.clearRetainingCapacity();
        for (items.items[0..i]) |other| if (apart(it.*, other, options)) try near.append(gpa, other);
        std.mem.sort(Item, near.items, {}, byOffset);
        var at: u64 = 0;
        for (near.items) |other| {
            if (at + it.size <= other.offset) break;
            at = @max(at, other.offset + other.size);
        }
        it.offset = at;
        it.placed = true;
        block = @max(block, at + it.size);
    }

    // The most alive at any one position.
    var peak: u64 = 0;
    for (0..shape.len) |p| {
        var live: u64 = 0;
        for (items.items) |it| if (it.start <= p and p < it.end) {
            live += it.size;
        };
        peak = @max(peak, live);
    }

    // For each buffer and each child block, the latest earlier tenant of each of its bytes: what must be dead, and
    // ordered before it.
    var preds: std.ArrayList(u32) = .empty;
    errdefer preds.deinit(gpa);
    var earlier: std.ArrayList(Item) = .empty;
    defer earlier.deinit(gpa);
    var covered: std.ArrayList([2]u64) = .empty;
    defer covered.deinit(gpa);
    for (items.items) |it| {
        earlier.clearRetainingCapacity();
        for (items.items) |other| {
            if (other.end <= it.start and other.offset < it.offset + it.size and it.offset < other.offset + other.size) try earlier.append(gpa, other);
        }
        std.mem.sort(Item, earlier.items, {}, struct {
            fn latest(_: void, x: Item, y: Item) bool {
                if (x.end != y.end) return x.end > y.end;
                return bySize({}, x, y);
            }
        }.latest);
        covered.clearRetainingCapacity();
        const at: u32 = @intCast(preds.items.len);
        for (earlier.items) |other| {
            const lo = @max(other.offset, it.offset);
            const hi = @min(other.offset + other.size, it.offset + it.size);
            if (!uncovered(covered.items, lo, hi)) continue;
            // A tenant that may never have been made hides nothing: whatever lay under it is named too.
            if (!other.optional) try covered.append(gpa, .{ lo, hi });
            try preds.append(gpa, switch (other.what) {
                .ask => asks[other.index].place.item,
                .through => through[other.index].place.item,
                .node => count + other.index,
            });
        }
        const range: Range = .{ .at = at, .len = @as(u32, @intCast(preds.items.len)) - at };
        switch (it.what) {
            .ask => {
                asks[it.index].place.offset = it.offset;
                asks[it.index].place.preds = range;
            },
            .through => {
                through[it.index].place.offset = it.offset;
                through[it.index].place.preds = range;
            },
            .node => switch (nodes[it.index]) {
                .ask => unreachable,
                inline else => |*c| {
                    c.child.offset = it.offset;
                    c.child.preds = range;
                },
            },
        }
    }

    return .{
        .id = shape.id,
        .asks = asks,
        .nodes = nodes,
        .slots = slots,
        .through = through,
        .preds = try preds.toOwnedSlice(gpa),
        .items = count,
        .block = block,
        .peak = peak,
        .host = host,
        .exports = @intCast(shape.exports.len),
    };
}

/// Whether export `e` of `shape` may never be made: it came from a loop's iteration somewhere inside, which a run that
/// stops the loop short never makes. An ask of the frame's own is made whenever the frame runs.
fn exportOptional(shapes: *const Shapes, shape: *const Shape, e: u32) bool {
    const ex = shape.exports[e];
    if (ex.own) return false;
    for (shape.slots) |slot| {
        if (ex.index < slot.through or ex.index >= slot.through + slot.exports) continue;
        return switch (shape.nodes.items[slot.node]) {
            .group => true,
            .exact => |id| exportOptional(shapes, &shapes.list.items[shapes.index.get(id).?], ex.index - slot.through),
            .ask => unreachable,
        };
    }
    unreachable;
}

/// Whether some byte of `lo..hi` is in none of `ranges`.
fn uncovered(ranges: []const [2]u64, lo: u64, hi: u64) bool {
    var at = lo;
    while (at < hi) {
        var moved = false;
        for (ranges) |r| if (r[0] <= at and at < r[1]) {
            at = r[1];
            moved = true;
        };
        if (!moved) return true;
    }
    return false;
}

const trace_mod = @import("plan_trace.zig");
const Recorder = trace_mod.Recorder;

fn planOf(gpa: std.mem.Allocator, r: *Recorder, options: Options) !Tables {
    var tr = try r.end();
    defer tr.deinit();
    var shapes = try shape_mod.merge(gpa, &.{tr}, .{ .alignment = 8 });
    defer shapes.deinit();
    return pack(gpa, &shapes, options);
}

test "buffers that never live together share bytes, those that do never share, and each names its last tenants" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    r.free(try r.ask(96, false, false));
    r.free(try r.ask(64, false, false));
    const c = try r.ask(40, false, false);
    const d = try r.ask(40, false, false);
    r.free(c);
    r.free(d);
    var tables = try planOf(t.allocator, &r, .{});
    defer tables.deinit();
    const root = tables.get(.{ .site = 1 }).?;
    const p = root.asks;
    try t.expectEqual(@as(u64, 0), p[0].place.offset);
    try t.expectEqual(@as(u64, 0), p[1].place.offset);
    try t.expectEqual(@as(u64, 0), p[2].place.offset);
    try t.expectEqual(@as(u64, 40), p[3].place.offset);
    try t.expectEqual(@as(u64, 96), root.block);
    try t.expectEqual(root.peak, root.block);
    try t.expectEqual(@as(u64, 96), tables.g);
    // The second takes the first's bytes; the fourth sits on the second's tail and, past it, on the first's.
    try t.expectEqualSlices(u32, &.{0}, root.preds[p[1].place.preds.at..][0..p[1].place.preds.len]);
    try t.expectEqualSlices(u32, &.{1}, root.preds[p[2].place.preds.at..][0..p[2].place.preds.len]);
    try t.expectEqualSlices(u32, &.{ 1, 0 }, root.preds[p[3].place.preds.at..][0..p[3].place.preds.len]);
}

test "bytes are not shared where the reuse would put a dispatch behind one it never waited for" {
    const t = std.testing;
    inline for (.{ true, false }) |rule| {
        var r = try Recorder.begin(t.allocator, .{ .site = 1 });
        defer r.deinit();
        const a = try r.ask(64, false, false);
        r.touch(a, 3);
        r.touch(a, 5);
        r.free(a);
        // Independent work: its first dispatch sits beside the first buffer's last.
        const b = try r.ask(64, false, false);
        r.touch(b, 4);
        r.free(b);
        // Work that already follows the first buffer's readers.
        const c = try r.ask(64, false, false);
        r.touch(c, 6);
        r.free(c);
        var tables = try planOf(t.allocator, &r, .{ .level_rule = rule });
        defer tables.deinit();
        const p = tables.get(.{ .site = 1 }).?.asks;
        try t.expectEqual(@as(u64, 0), p[0].place.offset);
        try t.expectEqual(@as(u64, if (rule) 64 else 0), p[1].place.offset);
        try t.expectEqual(@as(u64, 0), p[2].place.offset);
    }
}

test "a value carried from each exact child to the next takes two places, not one a child" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    var last: ?trace_mod.Ref = null;
    for (0..6) |layer| {
        try r.enter(.{ .site = 2, .key = layer }, .exact);
        const scratch = try r.ask(32, false, false);
        const next = try r.ask(64, false, false);
        if (last) |old| r.free(old);
        r.free(scratch);
        last = next;
        r.exit();
    }
    r.free(last.?);
    var tables = try planOf(t.allocator, &r, .{});
    defer tables.deinit();
    // Two carried values alive at a layer's end, and one layer's scratch.
    try t.expectEqual(@as(u64, 2 * 64 + 32), tables.g);
    try t.expectEqual(@as(u64, 32), tables.get(.{ .site = 2, .key = 3 }).?.block);
}

test "the results of a loop each keep their bytes to its end while its iterations share one block" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    var kept: [3]trace_mod.Ref = undefined;
    for (&kept) |*k| {
        try r.enter(.{ .site = 5 }, .each);
        const scratch = try r.ask(32, false, false);
        k.* = try r.ask(64, false, false);
        r.free(scratch);
        r.exit();
    }
    for (kept) |k| r.free(k);
    const host = try r.ask(10, true, false);
    r.free(host);
    var tables = try planOf(t.allocator, &r, .{});
    defer tables.deinit();
    const root = tables.get(.{ .site = 1 }).?;
    try t.expectEqual(@as(u64, 3 * 64 + 32), root.block);
    try t.expectEqual(@as(u32, 3), root.nodes[0].group.max);
    try t.expectEqual(@as(u64, 32), root.nodes[0].group.child.block);
    try t.expectEqual(@as(u32, 1), root.nodes[0].group.child.exports);
    // The three results sit apart from each other and from the iterations' block.
    var seen: [4]u64 = .{ root.through[0].place.offset, root.through[1].place.offset, root.through[2].place.offset, root.nodes[0].group.child.offset };
    std.mem.sort(u64, &seen, {}, std.sort.asc(u64));
    try t.expectEqualSlices(u64, &.{ 0, 64, 128, 192 }, &seen);
    // The host ring holds the host-filled ask, rounded to the alignment; the arena holds none of it.
    try t.expectEqual(@as(u64, 16), tables.h);
    try t.expect(root.asks[0].kind == .host);
}

test "a guard keeps its bytes after every buffer, and no other buffer takes them" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    const a = try r.ask(40, false, false);
    const b = try r.ask(40, false, false);
    r.free(a);
    r.free(b);
    r.free(try r.ask(40, false, false));
    var tables = try planOf(t.allocator, &r, .{ .guard = 16 });
    defer tables.deinit();
    const p = tables.get(.{ .site = 1 }).?.asks;
    try t.expectEqual(@as(u64, 16), tables.guard);
    try t.expectEqual(@as(u64, 0), p[0].place.offset);
    try t.expectEqual(@as(u64, 40 + 16), p[1].place.offset);
    try t.expectEqual(@as(u64, 2 * (40 + 16)), tables.g);
    // A guard that is not a multiple of the alignment is rounded up to one: every place stays aligned.
    var r2 = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r2.deinit();
    const c = try r2.ask(40, false, false);
    const d = try r2.ask(40, false, false);
    r2.free(c);
    r2.free(d);
    var odd = try planOf(t.allocator, &r2, .{ .guard = 12 });
    defer odd.deinit();
    try t.expectEqual(@as(u64, 16), odd.guard);
    for (odd.get(.{ .site = 1 }).?.asks) |as| try t.expectEqual(@as(u64, 0), as.place.offset % 8);
}
