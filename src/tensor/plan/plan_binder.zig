//! The binder: an execution replayed against its tables. Each ask gets the place the plan recorded for it, and an
//! ask the plan did not foresee is refused by name before a byte is written (docs/bill_redesign.md#arena). It deals
//! in offsets of one arena, `[ host ring x replicas | scratch ]`, and knows nothing of Metal: the engine turns an
//! offset into a store. Pure: the host tier builds it. One thread drives it.
//!
//! Order. Bytes are reused by buffers the engine's hazard tracking cannot tell are the same bytes, so the binder
//! carries the order for it. A mark is the engine's own measure of how far the GPU's work had got (a pass and a
//! level in it), opaque here and compared with `max`: the engine gives one when a buffer's last reference goes, when
//! a frame closes and when an execution begins, and every buffer comes back with the latest mark of whatever held any
//! of its bytes before (`Got.after`). The engine encodes the buffer's first dispatch after that mark.
const std = @import("std");
const frame = @import("plan_frame.zig");
const pack = @import("plan_pack.zig");
const FrameId = frame.FrameId;
const Tables = pack.Tables;
const FramePlan = pack.FramePlan;
const none = @import("plan_trace.zig").none;

pub const Error = error{ OutOfMemory, PlanMiss, PlanNeedsLease };

pub const MissKind = enum {
    /// A frame or an ask no probe recorded: an unknown `(site, key)`, or an ask past its frame's last.
    unplanned,
    /// An ask larger than the largest its place was recorded at, or a frame whose block the arena cannot hold.
    oversize,
    /// What came next is not what the frame's structure has next: a child where an ask was due, a loop past its
    /// longest run, a host-filled ask where scratch was recorded.
    structure,
    /// Bytes the plan had dead are still referenced: an earlier tenant of the place, or a buffer its frame outlived.
    lifetime,
    /// The execution's host-filled asks pass one replica of the host ring.
    host_full,
    /// A size that data decides passed the most its site declared (`plan.bound`).
    bound,
};

/// What the binder refused first in this execution.
pub const Miss = struct { kind: MissKind, id: FrameId, ordinal: u32 = 0, asked: u64 = 0, cap: u64 = 0 };

pub const Want = enum { scratch, host, external };

/// A buffer of the arena's scratch, for `release`.
pub const Slot = struct { depth: u16 = 0, serial: u64 = 0, item: u32 = none };

pub const Got = struct {
    want: Want,
    /// In the arena; unset for `external`.
    offset: u64 = 0,
    slot: Slot = .{},
    /// The latest mark of anything that held some of these bytes before; 0 with none.
    after: u64 = 0,
    /// The bytes of its place: the most recorded, aligned, and the guard after it (`Tables.guard`).
    room: u64 = 0,
};

pub const Layout = struct {
    /// The scratch arena and one replica of the host ring, in bytes.
    g: u64,
    h: u64,
    replicas: u32 = 2,
    /// The scratch a root may use while the lent pools are full; null where nothing is lent.
    keep: ?u64 = null,
    alignment: u32 = 256,

    pub fn arenaBytes(l: Layout) u64 {
        return l.h * l.replicas + l.g;
    }
};

const Open = struct {
    plan: *const FramePlan,
    base: u64,
    /// Its slot and its node in its parent.
    slot: u32,
    parent_node: u32,
    serial: u64,
    /// The latest mark of what held its block before it: every buffer in the block follows it.
    after: u64,
    ordinal: u32 = 0,
    node: u32 = 0,
    /// Iterations taken of the group at `node`.
    iter: u32 = 0,
    items_at: u32,
    marks_at: u32,
};

pub const Binder = struct {
    gpa: std.mem.Allocator,
    layout: Layout,
    tables: ?*const Tables = null,
    stack: std.ArrayList(Open) = .empty,
    /// Per buffer of each open frame: whether it is referenced, and the mark its release gave.
    live: std.ArrayList(bool) = .empty,
    item_marks: std.ArrayList(u64) = .empty,
    /// Per node of each open frame, the latest mark a child there left.
    node_marks: std.ArrayList(u64) = .empty,
    serial: u64 = 0,
    executions: u64 = 0,
    host_used: u64 = 0,
    miss: ?Miss = null,

    pub fn init(gpa: std.mem.Allocator, layout: Layout) Binder {
        return .{ .gpa = gpa, .layout = layout };
    }

    pub fn deinit(b: *Binder) void {
        b.stack.deinit(b.gpa);
        b.live.deinit(b.gpa);
        b.item_marks.deinit(b.gpa);
        b.node_marks.deinit(b.gpa);
        b.* = undefined;
    }

    /// The replica of the host ring this execution fills: the engine waits for the execution that last used it.
    pub fn replica(b: *const Binder) u32 {
        return @intCast(b.executions % b.layout.replicas);
    }

    pub fn active(b: *const Binder) bool {
        return b.stack.items.len != 0;
    }

    fn fail(b: *Binder, m: Miss) Error {
        if (b.miss == null) b.miss = m;
        return error.PlanMiss;
    }

    /// A miss the binder cannot see itself: the execution is refused from here.
    pub fn refuse(b: *Binder, m: Miss) error{PlanMiss} {
        if (b.miss == null) b.miss = m;
        return error.PlanMiss;
    }

    /// The frame open now.
    pub fn innermost(b: *const Binder) FrameId {
        return b.stack.items[b.stack.items.len - 1].plan.id;
    }

    fn top(b: *Binder) *Open {
        return &b.stack.items[b.stack.items.len - 1];
    }

    fn push(b: *Binder, plan: *const FramePlan, base: u64, slot: u32, parent_node: u32, after: u64) Error!void {
        const items_at: u32 = @intCast(b.live.items.len);
        const marks_at: u32 = @intCast(b.node_marks.items.len);
        try b.live.appendNTimes(b.gpa, false, plan.items);
        try b.item_marks.appendNTimes(b.gpa, 0, plan.items);
        try b.node_marks.appendNTimes(b.gpa, 0, plan.nodes.len);
        b.serial += 1;
        try b.stack.append(b.gpa, .{ .plan = plan, .base = base, .slot = slot, .parent_node = parent_node, .serial = b.serial, .after = after, .items_at = items_at, .marks_at = marks_at });
    }

    /// The latest mark among `owner`'s dead buffers and closed children named by `preds`; a miss when one of the
    /// buffers is still referenced.
    fn tenants(b: *Binder, owner: *const Open, preds: pack.Range, refused: Miss) Error!u64 {
        var after: u64 = 0;
        for (owner.plan.preds[preds.at..][0..preds.len]) |pred| {
            if (pred >= owner.plan.items) {
                after = @max(after, b.node_marks.items[owner.marks_at + pred - owner.plan.items]);
                continue;
            }
            if (b.live.items[owner.items_at + pred]) return b.fail(refused);
            after = @max(after, b.item_marks.items[owner.items_at + pred]);
        }
        return after;
    }

    /// An execution of `root` begins. `leased`: the lent pools are empty, so the root may pass `Layout.keep`.
    /// `after`: the engine's mark of all it has encoded so far, which may still read the last execution's bytes.
    pub fn begin(b: *Binder, tables: *const Tables, root: FrameId, leased: bool, after: u64) Error!void {
        std.debug.assert(!b.active());
        b.miss = null;
        b.tables = tables;
        const plan = tables.get(root) orelse return b.fail(.{ .kind = .unplanned, .id = root });
        if (plan.block > b.layout.g) return b.fail(.{ .kind = .oversize, .id = root, .asked = plan.block, .cap = b.layout.g });
        if (b.layout.keep) |keep| if (plan.block > keep and !leased) return error.PlanNeedsLease;
        b.executions += 1;
        b.host_used = 0;
        try b.push(plan, 0, 0, 0, after);
    }

    /// The execution ended. A miss it met, at an ask or at a frame's exit, is its error.
    pub fn end(b: *Binder) Error!void {
        std.debug.assert(b.stack.items.len == 1);
        b.exit(0);
        if (b.miss != null) return error.PlanMiss;
    }

    /// The execution failed: its frames go, and nothing of it is the binder's any more.
    pub fn abort(b: *Binder) void {
        b.stack.clearRetainingCapacity();
        b.live.clearRetainingCapacity();
        b.item_marks.clearRetainingCapacity();
        b.node_marks.clearRetainingCapacity();
    }

    /// A child frame opens in the innermost frame.
    pub fn enter(b: *Binder, id: FrameId, kind: frame.Kind) Error!void {
        if (b.miss != null) return error.PlanMiss;
        const open = b.top();
        const nodes = open.plan.nodes;
        const child, const slot, const node = while (true) {
            if (open.node >= nodes.len) return b.fail(.{ .kind = .structure, .id = id });
            switch (nodes[open.node]) {
                .ask => return b.fail(.{ .kind = .structure, .id = id }),
                .exact => |x| {
                    if (kind != .exact or !x.id.eql(id)) return b.fail(.{ .kind = .structure, .id = id });
                    open.node += 1;
                    break .{ x.child, x.child.slot, open.node - 1 };
                },
                .group => |g| {
                    if (kind != .each or g.site != id.site) {
                        // A loop that ran no further: what follows it is due.
                        open.node += 1;
                        open.iter = 0;
                        continue;
                    }
                    if (open.iter == g.max) return b.fail(.{ .kind = .structure, .id = id, .asked = g.max + 1, .cap = g.max });
                    open.iter += 1;
                    break .{ g.child, g.child.slot + open.iter - 1, open.node };
                },
            }
        };
        const plan = b.tables.?.get(id) orelse return b.fail(.{ .kind = .unplanned, .id = id });
        if (plan.block > child.block) return b.fail(.{ .kind = .oversize, .id = id, .asked = plan.block, .cap = child.block });
        if (plan.exports != child.exports) return b.fail(.{ .kind = .structure, .id = id });
        // Its block follows the frame's own floor, the loop's earlier iterations there, and the block's last tenants.
        var after = @max(open.after, b.node_marks.items[open.marks_at + node]);
        after = @max(after, try b.tenants(open, child.preds, .{ .kind = .lifetime, .id = id }));
        try b.push(plan, open.base + child.offset, slot, node, after);
    }

    /// The innermost frame closes. `mark`: the engine's mark of all it encoded up to now, which covers the frame.
    pub fn exit(b: *Binder, mark: u64) void {
        const open = b.stack.pop().?;
        for (open.plan.nodes[@min(open.node, open.plan.nodes.len)..]) |rest| {
            if (rest != .group and b.miss == null) b.miss = .{ .kind = .structure, .id = open.plan.id, .ordinal = open.ordinal };
        }
        for (b.live.items[open.items_at..]) |alive| {
            if (alive and b.miss == null) b.miss = .{ .kind = .lifetime, .id = open.plan.id };
        }
        b.live.shrinkRetainingCapacity(open.items_at);
        b.item_marks.shrinkRetainingCapacity(open.items_at);
        b.node_marks.shrinkRetainingCapacity(open.marks_at);
        if (b.stack.items.len == 0) return;
        const m = &b.node_marks.items[b.top().marks_at + open.parent_node];
        m.* = @max(m.*, mark);
    }

    /// The innermost frame's next ask, of `bytes`.
    pub fn ask(b: *Binder, bytes: u64, want: Want) Error!Got {
        if (b.miss != null) return error.PlanMiss;
        const open = b.top();
        const id = open.plan.id;
        const nodes = open.plan.nodes;
        const ordinal = open.ordinal;
        while (true) : (open.node += 1) {
            if (open.node >= nodes.len) return b.fail(.{ .kind = .unplanned, .id = id, .ordinal = ordinal, .asked = bytes });
            switch (nodes[open.node]) {
                .ask => break,
                .group => open.iter = 0,
                .exact => return b.fail(.{ .kind = .structure, .id = id, .ordinal = ordinal, .asked = bytes }),
            }
        }
        const plan = open.plan.asks[ordinal];
        const recorded: Want = switch (plan.kind) {
            .host => .host,
            .external => .external,
            .here, .lifted => .scratch,
        };
        if (recorded != want) return b.fail(.{ .kind = .structure, .id = id, .ordinal = ordinal, .asked = bytes });
        if (bytes > plan.place.cap) return b.fail(.{ .kind = .oversize, .id = id, .ordinal = ordinal, .asked = bytes, .cap = plan.place.cap });
        const got: Got = switch (plan.kind) {
            .external => .{ .want = .external },
            .host => blk: {
                const a = b.layout.alignment;
                const size = (@max(bytes, 1) + a - 1) / a * a;
                if (b.host_used + size > b.layout.h) return b.fail(.{ .kind = .host_full, .id = id, .ordinal = ordinal, .asked = b.host_used + size, .cap = b.layout.h });
                defer b.host_used += size;
                break :blk .{ .want = .host, .offset = @as(u64, b.replica()) * b.layout.h + b.host_used };
            },
            .here => try b.place(@intCast(b.stack.items.len - 1), plan.place, bytes, ordinal),
            .lifted => blk: {
                // Up through each frame that only hands it on, to the one whose block holds it.
                var depth = b.stack.items.len - 1;
                var e = plan.up;
                while (true) : (depth -= 1) {
                    if (depth == 0) return b.fail(.{ .kind = .structure, .id = id, .ordinal = ordinal, .asked = bytes });
                    const parent = &b.stack.items[depth - 1];
                    const th = parent.plan.through[parent.plan.slots[b.stack.items[depth].slot] + e];
                    if (!th.here) {
                        e = th.up;
                        continue;
                    }
                    if (bytes > th.place.cap) return b.fail(.{ .kind = .oversize, .id = id, .ordinal = ordinal, .asked = bytes, .cap = th.place.cap });
                    break :blk try b.place(@intCast(depth - 1), th.place, bytes, ordinal);
                }
            },
        };
        open.ordinal += 1;
        open.node += 1;
        return got;
    }

    fn place(b: *Binder, depth: u16, p: pack.Place, bytes: u64, ordinal: u32) Error!Got {
        const owner = &b.stack.items[depth];
        const refused: Miss = .{ .kind = .lifetime, .id = owner.plan.id, .ordinal = ordinal, .asked = bytes };
        if (b.live.items[owner.items_at + p.item]) return b.fail(refused);
        const after = @max(owner.after, try b.tenants(owner, p.preds, refused));
        b.live.items[owner.items_at + p.item] = true;
        const a = b.layout.alignment;
        return .{
            .want = .scratch,
            .offset = b.layout.h * b.layout.replicas + owner.base + p.offset,
            .slot = .{ .depth = depth, .serial = owner.serial, .item = p.item },
            .after = after,
            .room = (@max(p.cap, 1) + a - 1) / a * a + b.tables.?.guard,
        };
    }

    /// The buffer's last reference went, at the engine's `mark` for its last dispatch. After its frame closed this
    /// changes nothing: the exit already judged it.
    pub fn release(b: *Binder, slot: Slot, mark: u64) void {
        if (slot.item == none or slot.depth >= b.stack.items.len) return;
        const owner = b.stack.items[slot.depth];
        if (owner.serial != slot.serial) return;
        b.live.items[owner.items_at + slot.item] = false;
        b.item_marks.items[owner.items_at + slot.item] = mark;
    }
};

const trace_mod = @import("plan_trace.zig");
const shape_mod = @import("plan_shape.zig");
const Recorder = trace_mod.Recorder;

/// `trace` replayed against `tables` as a served execution of its shape would ask, with no engine: each ask, frame and
/// death in the order recorded. Its first miss is `PlanMiss` with `b.miss` set; none means the plan serves that shape
/// (docs/bill_redesign.md#coverage). `b` is idle before and after.
pub fn replayTrace(b: *Binder, tables: *const Tables, trace: *const trace_mod.Trace) Error!void {
    const gpa = b.gpa;
    const inst = trace.instances;
    // Per instance, the asks that die in it: by the index of the event they die before.
    const Death = struct { at: u32, ref: trace_mod.Ref };
    const dies = try gpa.alloc(std.ArrayList(Death), inst.len);
    defer {
        for (dies) |*d| d.deinit(gpa);
        gpa.free(dies);
    }
    @memset(dies, .empty);
    for (inst, 0..) |in, i| for (in.asks.items, 0..) |a, o| {
        if (a.death == none) continue;
        const owner = trace.ancestor(@intCast(i), a.lift) orelse return error.PlanMiss;
        try dies[owner].append(gpa, .{ .at = a.death, .ref = .{ .instance = @intCast(i), .ordinal = @intCast(o) } });
    };
    const slots = try gpa.alloc([]Slot, inst.len);
    defer {
        for (slots) |sl| gpa.free(sl);
        gpa.free(slots);
    }
    for (slots, inst) |*sl, in| {
        sl.* = try gpa.alloc(Slot, in.asks.items.len);
        @memset(sl.*, .{});
    }
    var mark: u64 = 1;
    const Walk = struct {
        b: *Binder,
        inst: []const trace_mod.Instance,
        dies: []const std.ArrayList(Death),
        slots: [][]Slot,
        mark: *u64,

        fn releaseAt(w: @This(), i: u32, at: u32) void {
            for (w.dies[i].items) |d| if (d.at == at) {
                w.mark.* += 1;
                w.b.release(w.slots[d.ref.instance][d.ref.ordinal], w.mark.*);
            };
        }

        fn run(w: @This(), i: u32) Error!void {
            const in = w.inst[i];
            for (in.events.items, 0..) |e, k| {
                w.releaseAt(i, @intCast(k));
                switch (e) {
                    .ask => |o| {
                        const a = in.asks.items[o];
                        const want: Want = if (a.host) .host else if (a.external) .external else .scratch;
                        w.slots[i][o] = (try w.b.ask(a.bytes, want)).slot;
                    },
                    .child => |c| {
                        try w.b.enter(w.inst[c].id, w.inst[c].kind);
                        w.run(c) catch |err| return err;
                        w.mark.* += 1;
                        w.b.exit(w.mark.*);
                    },
                }
            }
            w.releaseAt(i, @intCast(in.events.items.len));
        }
    };
    try b.begin(tables, inst[0].id, false, mark);
    const w: Walk = .{ .b = b, .inst = inst, .dies = dies, .slots = slots, .mark = &mark };
    w.run(0) catch |e| {
        b.abort();
        return e;
    };
    try b.end();
}

test "a trace replayed against the tables of its probes meets no miss at a smaller shape, and an ask past its place by name" {
    const t = std.testing;
    const Probe = struct {
        fn of(gpa: std.mem.Allocator, head: u64, loop: []const u64) !trace_mod.Trace {
            var r = try Recorder.begin(gpa, .{ .site = 1 });
            defer r.deinit();
            const first = try r.ask(head, false, false);
            try r.enter(.{ .site = 2 }, .exact);
            const kept = try r.ask(32, false, false);
            r.exit();
            for (loop) |bytes| {
                try r.enter(.{ .site = 5 }, .each);
                r.free(try r.ask(bytes, false, false));
                r.exit();
            }
            r.free(first);
            r.free(kept);
            r.free(try r.ask(16, true, false));
            return r.end();
        }
    };
    var probe = try Probe.of(t.allocator, 96, &.{ 64, 64, 64 });
    defer probe.deinit();
    var shapes = try shape_mod.merge(t.allocator, &.{probe}, .{ .alignment = 8 });
    defer shapes.deinit();
    var tables = try pack.pack(t.allocator, &shapes, .{});
    defer tables.deinit();
    var b = Binder.init(t.allocator, layoutOf(&tables));
    defer b.deinit();
    var smaller = try Probe.of(t.allocator, 40, &.{ 8, 64 });
    defer smaller.deinit();
    try replayTrace(&b, &tables, &smaller);
    try t.expect(!b.active());
    var larger = try Probe.of(t.allocator, 40, &.{ 8, 72 });
    defer larger.deinit();
    try t.expectError(error.PlanMiss, replayTrace(&b, &tables, &larger));
    try t.expectEqual(Miss{ .kind = .oversize, .id = .{ .site = 5 }, .ordinal = 0, .asked = 72, .cap = 64 }, b.miss.?);
    try t.expect(!b.active());
    var longer = try Probe.of(t.allocator, 40, &.{ 8, 8, 8, 8 });
    defer longer.deinit();
    try t.expectError(error.PlanMiss, replayTrace(&b, &tables, &longer));
    try t.expectEqual(MissKind.structure, b.miss.?.kind);
}

/// A buffer of `head`, then a loop of `sizes.len` iterations that each use scratch and leave a result, then a
/// host-filled ask.
fn recordLoop(gpa: std.mem.Allocator, head: u64, sizes: []const u64) !Tables {
    var r = try Recorder.begin(gpa, .{ .site = 1 });
    defer r.deinit();
    const first = try r.ask(head, false, false);
    const kept = try gpa.alloc(trace_mod.Ref, sizes.len);
    defer gpa.free(kept);
    for (sizes, kept) |bytes, *k| {
        try r.enter(.{ .site = 5 }, .each);
        const scratch = try r.ask(bytes / 2, false, false);
        k.* = try r.ask(bytes, false, false);
        r.free(scratch);
        r.exit();
    }
    r.free(first);
    for (kept) |k| r.free(k);
    r.free(try r.ask(24, true, false));
    var tr = try r.end();
    defer tr.deinit();
    var shapes = try shape_mod.merge(gpa, &.{tr}, .{ .alignment = 8 });
    defer shapes.deinit();
    return pack.pack(gpa, &shapes, .{});
}

fn layoutOf(tables: *const Tables) Layout {
    return .{ .g = tables.g, .h = tables.h, .alignment = tables.alignment };
}

test "a replay with smaller sizes and fewer iterations lands on the recorded places" {
    const t = std.testing;
    var tables = try recordLoop(t.allocator, 96, &.{ 64, 64, 64 });
    defer tables.deinit();
    var b = Binder.init(t.allocator, layoutOf(&tables));
    defer b.deinit();
    const root = tables.get(.{ .site = 1 }).?;
    const scratch = b.layout.h * b.layout.replicas;

    try b.begin(&tables, .{ .site = 1 }, false, 0);
    const first = try b.ask(40, .scratch);
    try t.expectEqual(scratch + root.asks[0].place.offset, first.offset);
    var kept: [2]Got = undefined;
    for (&kept, 0..) |*k, i| {
        try b.enter(.{ .site = 5 }, .each);
        const s = try b.ask(8, .scratch);
        k.* = try b.ask(16, .scratch);
        // A loop result is the parent's, at the place of its iteration.
        try t.expectEqual(scratch + root.through[i].place.offset, k.offset);
        try t.expectEqual(scratch + root.nodes[1].group.child.offset, s.offset);
        b.release(s.slot, 0);
        b.exit(0);
    }
    b.release(first.slot, 0);
    for (kept) |k| b.release(k.slot, 0);
    const host = try b.ask(24, .host);
    try t.expectEqual(@as(u64, b.replica()) * b.layout.h, host.offset);
    try b.end();
    try t.expect(b.miss == null);
}

test "every kind of miss refuses by name and says what was asked against what was recorded" {
    const t = std.testing;
    var tables = try recordLoop(t.allocator, 96, &.{ 64, 64 });
    defer tables.deinit();
    const Case = enum { oversize, unplanned_ask, unknown_frame, ask_due, loop_too_long, host_for_scratch, tenant_alive, outlived, host_full, lift_oversize };
    inline for (comptime std.enums.values(Case)) |case| {
        var layout = layoutOf(&tables);
        if (case == .host_full) layout.h = 8;
        var b = Binder.init(t.allocator, layout);
        defer b.deinit();
        try b.begin(&tables, .{ .site = 1 }, false, 0);
        const want: MissKind = switch (case) {
            .oversize => blk: {
                try t.expectError(error.PlanMiss, b.ask(104, .scratch));
                try t.expectEqual(Miss{ .kind = .oversize, .id = .{ .site = 1 }, .ordinal = 0, .asked = 104, .cap = 96 }, b.miss.?);
                break :blk .oversize;
            },
            .host_for_scratch => blk: {
                try t.expectError(error.PlanMiss, b.ask(8, .host));
                break :blk .structure;
            },
            .unknown_frame => blk: {
                _ = try b.ask(8, .scratch);
                try t.expectError(error.PlanMiss, b.enter(.{ .site = 5, .key = 99 }, .each));
                break :blk .unplanned;
            },
            .ask_due => blk: {
                try t.expectError(error.PlanMiss, b.enter(.{ .site = 5 }, .each));
                break :blk .structure;
            },
            .loop_too_long, .tenant_alive, .outlived, .lift_oversize => blk: {
                const first = try b.ask(8, .scratch);
                for (0..2) |i| {
                    try b.enter(.{ .site = 5 }, .each);
                    const s = try b.ask(8, .scratch);
                    if (case == .lift_oversize) {
                        try t.expectError(error.PlanMiss, b.ask(72, .scratch));
                        break :blk .oversize;
                    }
                    const k = try b.ask(8, .scratch);
                    if (case == .outlived and i == 0) {
                        // The frame closes over its own scratch, still referenced.
                        b.exit(0);
                        try t.expectError(error.PlanMiss, b.ask(8, .scratch));
                        break :blk .lifetime;
                    }
                    b.release(s.slot, 0);
                    b.exit(0);
                    if (case != .tenant_alive) b.release(k.slot, 0);
                }
                if (case == .tenant_alive) {
                    // The execution ends over the loop's results, still referenced.
                    b.release(first.slot, 0);
                    _ = try b.ask(24, .host);
                    try t.expectError(error.PlanMiss, b.end());
                    break :blk .lifetime;
                }
                b.release(first.slot, 0);
                try t.expectError(error.PlanMiss, b.enter(.{ .site = 5 }, .each));
                break :blk .structure;
            },
            .unplanned_ask, .host_full => blk: {
                b.release((try b.ask(8, .scratch)).slot, 0);
                if (case == .host_full) {
                    try t.expectError(error.PlanMiss, b.ask(24, .host));
                    break :blk .host_full;
                }
                _ = try b.ask(24, .host);
                try t.expectError(error.PlanMiss, b.ask(8, .scratch));
                break :blk .unplanned;
            },
        };
        try t.expectEqual(want, b.miss.?.kind);
        b.abort();
        try t.expect(!b.active());
    }
}

test "host-filled asks fill one replica of the ring an execution, and a root past what the lent pools leave needs the lease" {
    const t = std.testing;
    var tables = try recordLoop(t.allocator, 96, &.{64});
    defer tables.deinit();
    var layout = layoutOf(&tables);
    layout.keep = tables.g - 8;
    var b = Binder.init(t.allocator, layout);
    defer b.deinit();
    try t.expectError(error.PlanNeedsLease, b.begin(&tables, .{ .site = 1 }, false, 0));
    try t.expectEqual(tables.h * 2 + tables.g, layout.arenaBytes());
    var offsets: [3]u64 = undefined;
    for (&offsets) |*o| {
        try b.begin(&tables, .{ .site = 1 }, true, 0);
        const first = try b.ask(96, .scratch);
        try b.enter(.{ .site = 5 }, .each);
        b.release((try b.ask(32, .scratch)).slot, 0);
        const k = try b.ask(64, .scratch);
        b.exit(0);
        b.release(first.slot, 0);
        b.release(k.slot, 0);
        o.* = (try b.ask(24, .host)).offset;
        try b.end();
    }
    try t.expectEqual(offsets[0], offsets[2]);
    try t.expectEqual(tables.h, @max(offsets[0], offsets[1]) - @min(offsets[0], offsets[1]));
}

test "a buffer follows whatever held its bytes: the last execution, a closed child frame, an earlier iteration, a dead buffer" {
    const t = std.testing;
    var r = try Recorder.begin(t.allocator, .{ .site = 1 });
    defer r.deinit();
    for (0..2) |_| {
        try r.enter(.{ .site = 2 }, .each);
        r.free(try r.ask(64, false, false));
        r.exit();
    }
    r.free(try r.ask(64, false, false));
    r.free(try r.ask(64, false, false));
    var tr = try r.end();
    defer tr.deinit();
    var shapes = try shape_mod.merge(t.allocator, &.{tr}, .{ .alignment = 8 });
    defer shapes.deinit();
    var tables = try pack.pack(t.allocator, &shapes, .{});
    defer tables.deinit();
    // One place serves all four buffers in turn.
    try t.expectEqual(@as(u64, 64), tables.g);

    var b = Binder.init(t.allocator, layoutOf(&tables));
    defer b.deinit();
    try b.begin(&tables, .{ .site = 1 }, false, 10);
    try b.enter(.{ .site = 2 }, .each);
    const a = try b.ask(64, .scratch);
    try t.expectEqual(@as(u64, 10), a.after);
    b.release(a.slot, 20);
    b.exit(30);
    try b.enter(.{ .site = 2 }, .each);
    const second = try b.ask(64, .scratch);
    try t.expectEqual(@as(u64, 30), second.after);
    b.release(second.slot, 35);
    b.exit(41);
    const over = try b.ask(64, .scratch);
    try t.expectEqual(@as(u64, 41), over.after);
    // A release of a buffer whose frame is gone must not free the buffer now on its bytes.
    b.release(a.slot, 99);
    try t.expectError(error.PlanMiss, b.ask(64, .scratch));
    try t.expectEqual(MissKind.lifetime, b.miss.?.kind);
    b.abort();

    try b.begin(&tables, .{ .site = 1 }, false, 0);
    const o2 = try b.ask(64, .scratch);
    b.release(o2.slot, 77);
    const last = try b.ask(64, .scratch);
    try t.expectEqual(@as(u64, 77), last.after);
    b.release(last.slot, 0);
    try b.end();
}
