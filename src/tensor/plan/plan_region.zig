//! The wide phases' region (D97, docs/bill_redesign.md): an execution of a prefill chunk or an image's encode places
//! each ask at the lowest gap that holds it (address-ordered first fit; a dead ask's bytes join their neighbours'), and
//! its host-filled asks one after another in the host ring's replica, which comes back down at each eval to the end of
//! the highest host ask still alive: the GPU has read every one dead by then. `Need` runs those rules over a dry
//! recording's trace: the region a load holds is the most of each over its probes, and `core`'s wide execution places
//! by the same rules. Pure: the host tier builds it.
const std = @import("std");
const trace_mod = @import("plan_trace.zig");
const Trace = trace_mod.Trace;

pub const Error = error{OutOfMemory};

/// What an execution asks of the region at most: its scratch's highest top, and its host ring's.
pub const Need = struct {
    scratch: u64 = 0,
    host: u64 = 0,

    pub fn max(a: Need, b: Need) Need {
        return .{ .scratch = @max(a.scratch, b.scratch), .host = @max(a.host, b.host) };
    }
};

/// Free bytes below the top, each with the mark of the dispatches that last bound them (`core` takes it as the floor
/// of their next tenant; `need` ignores it).
pub const Gap = struct { lo: u64, hi: u64, mark: u64 = 0 };

/// The scratch's placement: gaps sorted by offset, none touching another or the top.
pub const Fit = struct {
    gaps: std.ArrayList(Gap) = .empty,
    top: u64 = 0,
    /// The highest mark of bytes the top came down over: a place from the top lies on them.
    top_mark: u64 = 0,
    high: u64 = 0,

    pub fn deinit(f: *Fit, gpa: std.mem.Allocator) void {
        f.gaps.deinit(gpa);
    }

    /// The lowest gap that holds `len` (already aligned), or the top; the place and the mark its bytes carry.
    pub fn take(f: *Fit, len: u64) struct { at: u64, mark: u64 } {
        for (f.gaps.items, 0..) |*g, i| if (g.hi - g.lo >= len) {
            const at = g.lo;
            const mark = g.mark;
            if (g.hi - g.lo == len) _ = f.gaps.orderedRemove(i) else g.lo += len;
            return .{ .at = at, .mark = mark };
        };
        const at = f.top;
        f.top += len;
        f.high = @max(f.high, f.top);
        return .{ .at = at, .mark = f.top_mark };
    }

    /// The place `[lo, hi)` is dead, its bytes last bound at `mark`.
    pub fn give(f: *Fit, gpa: std.mem.Allocator, lo_in: u64, hi_in: u64, mark_in: u64) Error!void {
        var lo = lo_in;
        var hi = hi_in;
        var mark = mark_in;
        // The gaps either side, which it joins.
        var i: usize = 0;
        while (i < f.gaps.items.len and f.gaps.items[i].hi <= lo) i += 1;
        if (i > 0 and f.gaps.items[i - 1].hi == lo) {
            i -= 1;
            lo = f.gaps.items[i].lo;
            mark = @max(mark, f.gaps.items[i].mark);
            _ = f.gaps.orderedRemove(i);
        }
        if (i < f.gaps.items.len and f.gaps.items[i].lo == hi) {
            hi = f.gaps.items[i].hi;
            mark = @max(mark, f.gaps.items[i].mark);
            _ = f.gaps.orderedRemove(i);
        }
        if (hi == f.top) {
            f.top = lo;
            f.top_mark = @max(f.top_mark, mark);
            return;
        }
        try f.gaps.insert(gpa, i, .{ .lo = lo, .hi = hi, .mark = mark });
    }
};

/// The place of each scratch ask of `t` in the order an execution makes them, each rounded up to `alignment`: what
/// `core.beginWidePadded` pads an execution of the same structure to. The caller frees it.
pub fn padOf(gpa: std.mem.Allocator, t: *const Trace, alignment: u64) Error![]u64 {
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(gpa);
    try padInstance(gpa, t, 0, alignment, &out);
    return out.toOwnedSlice(gpa);
}

fn padInstance(gpa: std.mem.Allocator, t: *const Trace, i: u32, alignment: u64, out: *std.ArrayList(u64)) Error!void {
    const inst = &t.instances[i];
    for (inst.events.items) |e| switch (e) {
        .ask => |o| {
            const a = inst.asks.items[o];
            if (!a.host and !a.external) try out.append(gpa, std.mem.alignForward(u64, @max(a.bytes, 1), alignment));
        },
        .child => |c| try padInstance(gpa, t, c, alignment, out),
    };
}

/// Whether a padded execution may follow its own choices from one of `traces`' passes to another: every pass meets
/// the same sites in the same order, and leaves first fit in one state at each, the frees before its next scratch ask
/// taken. Then each stretch between two choices places as the pass it follows did, from the state that pass had.
pub fn choicesAgree(gpa: std.mem.Allocator, traces: []const Trace, alignment: u64) Error!bool {
    for (traces) |*tr| if (!choicesKeepTheirAsks(tr)) return false;
    if (traces.len < 2) return true;
    const first = try statesAt(gpa, &traces[0], alignment);
    defer gpa.free(first);
    for (traces[1..]) |*tr| {
        if (tr.choices.len != traces[0].choices.len) return false;
        for (tr.choices, traces[0].choices) |a, b| if (a.site != b.site) return false;
        const got = try statesAt(gpa, tr, alignment);
        defer gpa.free(got);
        if (!std.mem.eql(u64, first, got)) return false;
    }
    return true;
}

/// Whether every scratch ask `t` made after a choice, in the instance the choice was made in or below it, died inside
/// that instance. One that outlives it lands where its own arm left room, so the arms' passes part wherever first fit
/// happens to place it apart: an arm hands up only what was made before it.
fn choicesKeepTheirAsks(t: *const Trace) bool {
    for (t.choices) |c| {
        if (c.frame == 0) continue;
        for (t.instances, 0..) |inst, h| {
            // The instance's way into the choice's: its own events when it is the choice's, else the child it is under.
            var below: u32 = 0;
            var c_inst: u32 = @intCast(h);
            while (c_inst != c.frame) : (below += 1) {
                const up = t.instances[c_inst].parent;
                if (up == trace_mod.none) break;
                if (up == c.frame) {
                    if (t.instances[c_inst].at < c.event) break;
                    below += 1;
                    c_inst = up;
                    break;
                }
                c_inst = up;
            }
            if (c_inst != c.frame) continue;
            for (inst.events.items, 0..) |e, k| switch (e) {
                .ask => |o| {
                    const a = inst.asks.items[o];
                    if (a.host or a.external) continue;
                    if (h == c.frame and k < c.event) continue;
                    if (a.death == trace_mod.none or a.lift > below) return false;
                },
                .child => {},
            };
        }
    }
    return true;
}

/// A hash of first fit's state at each of `t`'s choices: before its next scratch ask, or at the end.
fn statesAt(gpa: std.mem.Allocator, t: *const Trace, alignment: u64) Error![]u64 {
    var w: Walk = .{ .gpa = gpa, .t = t, .alignment = alignment };
    defer w.deinit();
    w.states = try gpa.alloc(u64, t.choices.len);
    errdefer gpa.free(w.states);
    try w.run();
    while (w.next_state < t.choices.len) : (w.next_state += 1) w.states[w.next_state] = w.fitHash();
    return w.states;
}

/// The region `t` asks, each place rounded up to `alignment`. An ask `plan.into` named is its owner's bytes, never the
/// region's.
pub fn need(gpa: std.mem.Allocator, t: *const Trace, alignment: u64) Error!Need {
    var w: Walk = .{ .gpa = gpa, .t = t, .alignment = alignment };
    defer w.deinit();
    try w.run();
    return .{ .scratch = w.fit.high, .host = w.host_high };
}

const Walk = struct {
    gpa: std.mem.Allocator,
    t: *const Trace,
    alignment: u64,
    dying: []std.AutoHashMapUnmanaged(u32, std.ArrayList(trace_mod.Ref)) = &.{},
    places: std.AutoHashMapUnmanaged(u64, [2]u64) = .empty,
    fit: Fit = .{},
    /// The live host asks' ends, by ask.
    host_live: std.AutoHashMapUnmanaged(u64, u64) = .empty,
    host_top: u64 = 0,
    host_high: u64 = 0,
    waits: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    /// `statesAt`: first fit's state as each choice is met, and the scratch asks placed so far.
    states: []u64 = &.{},
    next_state: usize = 0,
    scratch: u32 = 0,

    fn run(w: *Walk) Error!void {
        const gpa = w.gpa;
        const t = w.t;
        // Each instance's asks by the index of the owner's event they die before.
        w.dying = try gpa.alloc(std.AutoHashMapUnmanaged(u32, std.ArrayList(trace_mod.Ref)), t.instances.len);
        for (w.dying) |*d| d.* = .empty;
        for (t.instances, 0..) |inst, i| for (inst.asks.items, 0..) |a, o| {
            if (a.death == trace_mod.none) continue;
            var owner: u32 = @intCast(i);
            for (0..a.lift) |_| owner = t.instances[owner].parent;
            const got = try w.dying[owner].getOrPut(gpa, a.death);
            if (!got.found_existing) got.value_ptr.* = .empty;
            try got.value_ptr.append(gpa, .{ .instance = @intCast(i), .ordinal = @intCast(o) });
        };
        for (t.waits) |x| try w.waits.put(gpa, @as(u64, x.instance) << 32 | x.at, x.deaths);
        try w.instance(0);
    }

    fn fitHash(w: *const Walk) u64 {
        var h = std.hash.Wyhash.init(0);
        std.hash.autoHash(&h, w.fit.top);
        for (w.fit.gaps.items) |g| {
            std.hash.autoHash(&h, g.lo);
            std.hash.autoHash(&h, g.hi);
        }
        return h.final();
    }

    fn deinit(w: *Walk) void {
        for (w.dying) |*d| {
            var it = d.valueIterator();
            while (it.next()) |l| l.deinit(w.gpa);
            d.deinit(w.gpa);
        }
        w.gpa.free(w.dying);
        w.places.deinit(w.gpa);
        w.fit.deinit(w.gpa);
        w.host_live.deinit(w.gpa);
        w.waits.deinit(w.gpa);
    }

    fn key(r: trace_mod.Ref) u64 {
        return @as(u64, r.instance) << 32 | r.ordinal;
    }

    fn instance(w: *Walk, i: u32) Error!void {
        const inst = &w.t.instances[i];
        for (inst.events.items, 0..) |e, k| {
            // A wait comes before the deaths placed with it, which may have come after it: each ask's `died` says.
            w.wait(i, @intCast(k));
            try w.kill(i, @intCast(k));
            switch (e) {
                .ask => |o| try w.ask(.{ .instance = i, .ordinal = o }),
                .child => |c| try w.instance(c),
            }
        }
        w.wait(i, @intCast(inst.events.items.len));
        try w.kill(i, @intCast(inst.events.items.len));
    }

    fn wait(w: *Walk, i: u32, at: u32) void {
        const deaths = w.waits.get(@as(u64, i) << 32 | at) orelse return;
        var top: u64 = 0;
        var it = w.host_live.iterator();
        while (it.next()) |e| {
            const k = e.key_ptr.*;
            const died = w.t.instances[@intCast(k >> 32)].asks.items[@intCast(k & 0xFFFF_FFFF)].died;
            if (died >= deaths) top = @max(top, e.value_ptr.*);
        }
        w.host_top = top;
    }

    fn ask(w: *Walk, ref: trace_mod.Ref) Error!void {
        const a = w.t.instances[ref.instance].asks.items[ref.ordinal];
        if (a.external) return;
        const len = std.mem.alignForward(u64, @max(a.bytes, 1), w.alignment);
        if (a.host) {
            w.host_top += len;
            w.host_high = @max(w.host_high, w.host_top);
            try w.host_live.put(w.gpa, key(ref), w.host_top);
            return;
        }
        while (w.next_state < w.states.len and w.t.choices[w.next_state].at == w.scratch) : (w.next_state += 1) w.states[w.next_state] = w.fitHash();
        w.scratch += 1;
        const got = w.fit.take(len);
        try w.places.put(w.gpa, key(ref), .{ got.at, got.at + len });
    }

    fn kill(w: *Walk, owner: u32, at: u32) Error!void {
        const refs = w.dying[owner].get(at) orelse return;
        for (refs.items) |r| {
            if (w.host_live.remove(key(r))) continue;
            const p = w.places.fetchRemove(key(r)) orelse continue;
            try w.fit.give(w.gpa, p.value[0], p.value[1], 0);
        }
    }
};

const testing = std.testing;

/// A recording of `script`: 'a'..'z' asks an ask of that many pages, 'A'..'Z' kills the ask of that letter, '(' and
/// ')' open and close a frame, '!' makes the next ask host-filled, '|' waits for the GPU.
fn record(script: []const u8) !Trace {
    var r = try trace_mod.Recorder.begin(testing.allocator, .{ .site = 1 });
    defer r.deinit();
    var refs: [26]?trace_mod.Ref = @splat(null);
    var host = false;
    for (script) |ch| switch (ch) {
        'a'...'z' => {
            refs[ch - 'a'] = try r.ask(@as(u64, ch - 'a' + 1) * 4096, host, false);
            host = false;
        },
        'A'...'Z' => r.free(refs[ch - 'A'].?),
        '(' => try r.enter(.{ .site = 2 }, .exact),
        ')' => r.exit(),
        '!' => host = true,
        '|' => try r.wait(),
        else => unreachable,
    };
    return r.end();
}

test "region: an ask takes the lowest gap that holds it, a dead ask's bytes join their neighbours', and host asks add up apart" {
    const page: u64 = 4096;
    const cases = [_]struct { script: []const u8, scratch: u64, host: u64 = 0 }{
        // a (1 page), b (2) above it; b dies, then c (3) takes the top: 1 + 3 pages.
        .{ .script = "abBcAC", .scratch = 4 },
        // b (2) dies under a live d (4): c (3) does not fit b's gap and goes on top; then a (1) takes b's gap.
        .{ .script = "bdBcaDCA", .scratch = 2 + 4 + 3 },
        // A result a frame hands up stays where it is; the scratch below it is a gap the next ask fits.
        .{ .script = "e(bc)BaCAE", .scratch = 5 + 2 + 3 },
        // Two dead neighbours join: c (3) fits where a (1) and b (2) were.
        .{ .script = "abdABcCD", .scratch = 1 + 2 + 4 },
        // Host asks never take scratch: they add up.
        .{ .script = "a!b!cABC", .scratch = 1, .host = 5 },
        // A wait gives back the host bytes of every host ask dead before it: c (3) lands where a and b were.
        .{ .script = "!a!bAB|!cC", .scratch = 0, .host = 3 },
        // A host ask alive at the wait keeps its bytes: c lands past a.
        .{ .script = "!a!bB|!cAC", .scratch = 0, .host = 1 + 3 },
        // An ask that dies after the wait, in the same place of the trace, was alive at it.
        .{ .script = "!a|A!bB", .scratch = 0, .host = 1 + 2 },
        // A wait inside a frame gives back an ask dead in its parent before it.
        .{ .script = "!aA(|!c)C", .scratch = 0, .host = 3 },
    };
    for (cases) |c| {
        var t = try record(c.script);
        defer t.deinit();
        const got = try need(testing.allocator, &t, page);
        errdefer std.debug.print("[region] {s}: scratch {d} pages, host {d}\n", .{ c.script, got.scratch / page, got.host / page });
        try testing.expectEqual(c.scratch * page, got.scratch);
        try testing.expectEqual(c.host * page, got.host);
    }
}

test "region: a gap's mark is the highest of the bytes it joined, and the top keeps the mark of what it came down over" {
    var f: Fit = .{};
    defer f.deinit(testing.allocator);
    const a = f.take(4096);
    const b = f.take(4096);
    _ = f.take(4096);
    try f.give(testing.allocator, a.at, a.at + 4096, 7);
    try f.give(testing.allocator, b.at, b.at + 4096, 3);
    try testing.expectEqual(@as(usize, 1), f.gaps.items.len);
    try testing.expectEqual(@as(u64, 7), f.gaps.items[0].mark);
    try testing.expectEqual(@as(u64, 7), f.take(8192).mark);
    try f.give(testing.allocator, 0, 8192, 9);
    try f.give(testing.allocator, 8192, 12288, 2);
    try testing.expectEqual(@as(u64, 0), f.top);
    try testing.expectEqual(@as(u64, 9), f.take(4096).mark);
}
