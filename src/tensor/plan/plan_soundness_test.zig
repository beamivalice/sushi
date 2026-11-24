//! The plan's soundness, end to end: generated forwards (nested frames, loops of any count, results handed up,
//! values carried from one child to the next, loop bodies of two kinds) are recorded at their largest shapes, merged
//! and packed, then replayed at other shapes against a shadow of the arena. Whatever the shape, no two live buffers
//! share a byte, nothing leaves the arena, and every buffer is ordered after all that held its bytes before.
const std = @import("std");
const frame = @import("plan_frame.zig");
const trace_mod = @import("plan_trace.zig");
const shape_mod = @import("plan_shape.zig");
const pack = @import("plan_pack.zig");
const binder_mod = @import("plan_binder.zig");
const plan = @import("plan_api.zig");
const Binder = binder_mod.Binder;

const at_end = 255;

/// When an ask dies: `up` frames above its home, before that frame's op `after + 1` past the one running, or during
/// the first iteration of the loop that op is (`mid`). `carry`: a loop iteration's result that the next iteration
/// frees as it starts, and the frame after the loop when there is no next.
const Die = struct { up: u8, after: u8, mid: bool, carry: bool = false };

const Op = union(enum) {
    ask: struct { base: u64, per: u64, host: bool, die: Die },
    child: struct { site: frame.Site, body: []const Op },
    loop: struct { site: frame.Site, max: u8, bodies: [2][]const Op, kinds: u8 },
};

const Gen = struct {
    a: std.mem.Allocator,
    rng: std.Random,
    sites: frame.Site = 100,

    fn askOp(g: *Gen, room: u8, lifted: bool) Op {
        const up: u8 = if (lifted) g.rng.intRangeAtMost(u8, 1, room) else if (room > 0 and g.rng.uintLessThan(u8, 4) == 0) g.rng.intRangeAtMost(u8, 1, room) else 0;
        return .{ .ask = .{
            .base = 8 * g.rng.intRangeAtMost(u64, 1, 16),
            .per = 8 * g.rng.uintLessThan(u64, 9),
            .host = !lifted and up == 0 and g.rng.uintLessThan(u8, 12) == 0,
            .die = .{ .up = up, .after = if (g.rng.uintLessThan(u8, 4) == 0) at_end else g.rng.uintLessThan(u8, 4), .mid = g.rng.boolean(), .carry = up == 1 and g.rng.boolean() },
        } };
    }

    /// `room`: the frames above this body that one of its asks may outlive.
    fn body(g: *Gen, depth: u8, room: u8) ![]const Op {
        const n = g.rng.intRangeAtMost(usize, 1, 5);
        const ops = try g.a.alloc(Op, n);
        for (ops) |*op| {
            const pick = g.rng.uintLessThan(u8, 10);
            g.sites += 1;
            const site = g.sites;
            if (depth < 3 and pick < 2) {
                op.* = .{ .child = .{ .site = site, .body = try g.body(depth + 1, room + 1) } };
            } else if (depth < 3 and pick < 4) {
                const kinds: u8 = if (g.rng.boolean()) 2 else 1;
                var bodies: [2][]const Op = undefined;
                if (kinds == 1) {
                    bodies = @splat(try g.body(depth + 1, room + 1));
                } else {
                    // Two kinds of one loop hand up the same results: what differs between them stays inside.
                    const tail = try g.a.alloc(Op, g.rng.uintLessThan(usize, 3));
                    for (tail) |*t| t.* = g.askOp(room + 1, true);
                    for (&bodies) |*b| b.* = try std.mem.concat(g.a, Op, &.{ try g.body(depth + 1, 0), tail });
                }
                op.* = .{ .loop = .{ .site = site, .max = g.rng.intRangeAtMost(u8, 1, 4), .bodies = bodies, .kinds = kinds } };
            } else op.* = g.askOp(room, false);
        }
        return ops;
    }
};

/// The size scale, and what picks each loop's count and each iteration's kind; a probe runs every loop to its longest
/// with the kind the driver holds it to.
const Shape = struct {
    s: u64,
    probe: ?*plan.Driver = null,
    rng: std.Random,
};

const Handle = struct {
    ref: trace_mod.Ref = undefined,
    slot: binder_mod.Slot = .{},
    offset: u64 = 0,
    len: u64 = 0,
    scratch: bool = false,
    /// For a carried result with no next iteration: the ops after the loop it lives through.
    after: u8 = 0,
};

const Running = struct {
    ops: []const Op,
    cur: usize = 0,
    due: []std.ArrayList(Handle),
    due_mid: []std.ArrayList(Handle),
    /// Whether the op running is a loop, and what its iteration so far handed on to the next.
    looping: bool = false,
    carried: std.ArrayList(Handle) = .empty,
};

const Range = struct { offset: u64, len: u64, mark: u64 };

const Run = struct {
    a: std.mem.Allocator,
    shape: Shape,
    rec: ?*trace_mod.Recorder = null,
    bind: ?*Binder = null,
    stack: std.ArrayList(*Running) = .empty,
    /// The shadow: what is alive in scratch now, and every range a dead buffer or an earlier execution left.
    live: std.ArrayList(Range) = .empty,
    dead: *std.ArrayList(Range),
    clock: *u64,
    host_next: u64 = 0,

    fn tick(r: *Run) u64 {
        r.clock.* += 1;
        return r.clock.*;
    }

    fn free(r: *Run, h: Handle) !void {
        if (r.rec) |rec| return rec.free(h.ref);
        if (!h.scratch) return;
        const mark = r.tick();
        r.bind.?.release(h.slot, mark);
        const at = for (r.live.items, 0..) |l, i| {
            if (l.offset == h.offset and l.len == h.len) break i;
        } else return error.ShadowLostABuffer;
        _ = r.live.swapRemove(at);
        try r.dead.append(r.a, .{ .offset = h.offset, .len = h.len, .mark = mark });
    }

    fn ask(r: *Run, bytes: u64, host: bool) !Handle {
        if (r.rec) |rec| return .{ .ref = try rec.ask(bytes, host, false) };
        const b = r.bind.?;
        const got = try b.ask(bytes, if (host) .host else .scratch);
        if (host) {
            const lo = @as(u64, b.replica()) * b.layout.h;
            if (got.offset < @max(lo, r.host_next) or got.offset + bytes > lo + b.layout.h) return error.HostAskOutsideItsReplica;
            r.host_next = got.offset + bytes;
            return .{};
        }
        const end = got.offset + bytes;
        if (got.offset < b.layout.h * b.layout.replicas or end > b.layout.arenaBytes()) return error.BufferOutsideTheArena;
        for (r.live.items) |l| if (got.offset < l.offset + l.len and l.offset < end) return error.TwoLiveBuffersShareBytes;
        for (r.dead.items) |d| if (got.offset < d.offset + d.len and d.offset < end and got.after < d.mark) return error.BufferNotOrderedAfterAnEarlierTenant;
        try r.live.append(r.a, .{ .offset = got.offset, .len = bytes, .mark = 0 });
        return .{ .slot = got.slot, .offset = got.offset, .len = bytes, .scratch = true };
    }

    fn enter(r: *Run, id: frame.FrameId, kind: frame.Kind) !void {
        if (r.rec) |rec| return rec.enter(id, kind);
        try r.bind.?.enter(id, kind);
    }

    fn exit(r: *Run) void {
        if (r.rec) |rec| return rec.exit();
        r.bind.?.exit(r.tick());
    }

    fn freeAll(r: *Run, list: *std.ArrayList(Handle)) !void {
        for (list.items) |h| try r.free(h);
        list.clearRetainingCapacity();
    }

    fn run(r: *Run, ops: []const Op) anyerror!void {
        var me: Running = .{ .ops = ops, .due = try r.a.alloc(std.ArrayList(Handle), ops.len + 1), .due_mid = try r.a.alloc(std.ArrayList(Handle), ops.len + 1) };
        @memset(me.due, .empty);
        @memset(me.due_mid, .empty);
        try r.stack.append(r.a, &me);
        defer _ = r.stack.pop();
        for (ops, 0..) |op, i| {
            me.cur = i;
            try r.freeAll(&me.due[i]);
            switch (op) {
                .ask => |a| {
                    var h = try r.ask(a.base + a.per * r.shape.s, a.host);
                    h.after = a.die.after;
                    const owner = r.stack.items[r.stack.items.len - 1 - a.die.up];
                    if (a.die.carry and owner.looping) {
                        try owner.carried.append(r.a, h);
                        continue;
                    }
                    const idx = if (a.die.after == at_end) owner.ops.len else @min(owner.ops.len, owner.cur + 1 + a.die.after);
                    const mid = a.die.mid and idx < owner.ops.len and owner.ops[idx] == .loop;
                    try (if (mid) &owner.due_mid[idx] else &owner.due[idx]).append(r.a, h);
                },
                .child => |c| {
                    try r.enter(.{ .site = c.site }, .exact);
                    try r.run(c.body);
                    r.exit();
                },
                .loop => |l| {
                    const count = if (r.shape.probe != null) l.max else r.shape.rng.uintAtMost(u8, l.max);
                    me.looping = true;
                    for (0..count) |iter| {
                        const kind: u8 = if (r.shape.probe) |driver| @intCast(driver.pick(l.site, l.kinds)) else r.shape.rng.uintLessThan(u8, l.kinds);
                        try r.enter(.{ .site = l.site, .key = kind }, .each);
                        if (iter == 0) try r.freeAll(&me.due_mid[i]);
                        // What the iteration before handed on dies here; what this one hands on is collected after.
                        var before = me.carried;
                        me.carried = .empty;
                        try r.freeAll(&before);
                        try r.run(l.bodies[kind]);
                        r.exit();
                    }
                    me.looping = false;
                    // A loop that ran no iteration frees nothing inside itself; its last iteration's results outlive it.
                    try r.freeAll(&me.due_mid[i]);
                    for (me.carried.items) |h| {
                        const idx = if (h.after == at_end) ops.len else @min(ops.len, i + 1 + h.after);
                        try me.due[idx].append(r.a, h);
                    }
                    me.carried = .empty;
                },
            }
        }
        try r.freeAll(&me.due[ops.len]);
    }
};

const max_scale = 8;

fn check(seed: u64, guard: u64) !void {
    const t = std.testing;
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(seed);
    var gen: Gen = .{ .a = a, .rng = prng.random() };
    const program = try gen.body(0, 0);
    const root: frame.FrameId = .{ .site = 1 };
    var clock: u64 = 0;
    var dead: std.ArrayList(Range) = .empty;

    // The probes: every loop kind the driver can reach, at the largest sizes and the longest runs; then two shapes
    // between, which the merge must fold into the same structures.
    var traces: std.ArrayList(trace_mod.Trace) = .empty;
    var driver: plan.Driver = .{ .gpa = a };
    var between: u8 = 0;
    while (true) {
        const probing = between == 0;
        var rec = try trace_mod.Recorder.begin(a, root);
        var run: Run = .{ .a = a, .shape = .{ .s = if (probing) max_scale else prng.random().uintAtMost(u64, max_scale), .probe = if (probing) &driver else null, .rng = prng.random() }, .rec = &rec, .dead = &dead, .clock = &clock };
        try run.run(program);
        try traces.append(a, try rec.end());
        if (probing and try driver.next()) continue;
        between += 1;
        if (between == 3) break;
    }
    var why: shape_mod.Ambiguity = undefined;
    var shapes = shape_mod.merge(a, traces.items, .{ .alignment = 8, .refused = &why }) catch |e| {
        if (e == error.PlanAmbiguous) std.debug.print("seed {d}: ambiguous {any}\n", .{ seed, why });
        return e;
    };
    var tables = try pack.pack(a, &shapes, .{ .guard = guard });
    try t.expect(tables.get(root).?.block >= tables.get(root).?.peak);

    var b = Binder.init(a, .{ .g = tables.g, .h = tables.h, .alignment = 8 });
    for (0..24) |_| {
        var run: Run = .{ .a = a, .shape = .{ .s = prng.random().uintAtMost(u64, max_scale), .rng = prng.random() }, .bind = &b, .dead = &dead, .clock = &clock };
        clock += 1;
        try b.begin(&tables, root, false, clock);
        run.run(program) catch |e| {
            if (e == error.PlanMiss) std.debug.print("seed {d}: miss {any}\n", .{ seed, b.miss.? });
            return e;
        };
        b.end() catch |e| {
            std.debug.print("seed {d}: miss at the end {any}\n", .{ seed, b.miss.? });
            return e;
        };
        try t.expectEqual(@as(usize, 0), run.live.items.len);
    }
}

test "a recorded plan replayed at any smaller shape, any loop counts and either loop kind keeps live buffers apart, inside the arena and in order" {
    // Seed 467 holds a result an exact child hands up from a loop inside it, which a short run never makes.
    for ([_]u64{ 0, 16 }) |guard| for (0..6000) |seed| check(seed, guard) catch |e| {
        std.debug.print("soundness: seed {d}, guard {d} failed: {t}\n", .{ seed, guard, e });
        return e;
    };
}
