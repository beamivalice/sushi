//! D97 address-ordered first-fit region sizing and placement (used uniformly for both narrow and wide executions).
//! Each scratch ask takes the lowest gap that holds it (adjacent dead asks coalesce); each host-filled ask bumps a
//! linear host-ring replica cursor, which drops back at each eval wait to the highest live host ask. Pure: host tier.
const std = @import("std");
const builtin = @import("builtin");
const trace_mod = @import("plan_trace.zig");
const Trace = trace_mod.Trace;

pub const Error = error{OutOfMemory};

pub const default_alignment: u64 = 256;
pub const test_guard: u64 = if (builtin.is_test) default_alignment else 0;

/// Shape of the shared Metal arena `[host ring x replicas | scratch]`.
pub const Layout = struct {
    /// Scratch pool bytes.
    g: u64 = 0,
    /// Host-ring bytes per replica.
    h: u64 = 0,
    replicas: u32 = 2,
    alignment: u64 = default_alignment,
    /// Non-null when `core.arenaFor` mmaps `[host | scratch]` with tail `[keep..]` leased from a borrower.
    keep: ?u64 = null,
    /// Extra canary tail bytes appended to each scratch arena allocation when enabled (`test_guard`).
    guard: u64 = 0,

    pub fn arenaBytes(l: Layout) u64 {
        return l.h * l.replicas + l.g;
    }
};

/// Peak scratch and host-ring bytes an execution trace requires.
pub const Need = struct {
    scratch: u64 = 0,
    host: u64 = 0,

    pub fn max(a: Need, b: Need) Need {
        return .{ .scratch = @max(a.scratch, b.scratch), .host = @max(a.host, b.host) };
    }
};

fn addHeadroom(bytes: u64, pct: u64, min_div: u64, floor_cap: u64, alignment: u64) u64 {
    if (bytes == 0) return 0;
    const by_pct = bytes + (bytes * pct + 99) / 100;
    const min_floor = bytes + @min(bytes / min_div, floor_cap);
    return std.mem.alignForward(u64, @max(by_pct, min_floor), alignment);
}

/// Explicit fragmentation headroom for narrow execution (`Layout.keep`).
pub fn withNarrowHeadroom(n: Need, alignment: u64) Need {
    return .{
        .scratch = addHeadroom(n.scratch, 15, 2, 32 << 20, alignment),
        .host = addHeadroom(n.host, 15, 2, 4 << 20, alignment),
    };
}

/// Explicit fragmentation headroom for wide execution (`Layout.g` / `Layout.h`).
pub fn withWideHeadroom(n: Need, alignment: u64) Need {
    return .{
        .scratch = addHeadroom(n.scratch, 5, 4, 16 << 20, alignment),
        .host = addHeadroom(n.host, 10, 4, 4 << 20, alignment),
    };
}

/// Free bytes below the top, each with the dispatch mark of the tenant that last held them.
pub const Gap = struct { lo: u64, hi: u64, mark: u64 = 0 };

/// Scratch placement: gaps sorted by offset, none touching another or the top.
pub const Fit = struct {
    gaps: std.ArrayList(Gap) = .empty,
    top: u64 = 0,
    /// Highest mark of bytes the top came down over: any allocation from the top below `top_mark_hi` lies on them.
    top_mark: u64 = 0,
    top_mark_hi: u64 = 0,
    high: u64 = 0,

    pub const Taken = struct { at: u64, mark: u64 };

    pub fn deinit(f: *Fit, gpa: std.mem.Allocator) void {
        f.gaps.deinit(gpa);
    }

    /// Lowest gap that holds `len` (already aligned), or the top; returns placement offset and carried mark.
    pub fn take(f: *Fit, len: u64) Taken {
        for (f.gaps.items, 0..) |*g, i| if (g.hi - g.lo >= len) {
            const at = g.lo;
            const mark = g.mark;
            if (g.hi - g.lo == len) _ = f.gaps.orderedRemove(i) else g.lo += len;
            return .{ .at = at, .mark = mark };
        };
        const at = f.top;
        const mark = if (at < f.top_mark_hi) f.top_mark else 0;
        f.top += len;
        f.high = @max(f.high, f.top);
        return .{ .at = at, .mark = mark };
    }

    /// Narrow placement preferring cold bytes (`mark < hot_mark`) to avoid spurious intra-pass GPU barriers when
    /// headroom allows, falling back to coolest fitting gap or `top` if bumping past `top_mark_hi` would exceed `cap`.
    pub fn takePreferCold(f: *Fit, gpa: std.mem.Allocator, len: u64, hot_mark: u64, cap: u64, cold_limit: u64) Error!Taken {
        for (f.gaps.items, 0..) |*g, i| if (g.hi - g.lo >= len and g.mark < hot_mark) {
            const at = g.lo;
            const mark = g.mark;
            if (g.hi - g.lo == len) _ = f.gaps.orderedRemove(i) else g.lo += len;
            return .{ .at = at, .mark = mark };
        };
        if (len <= cold_limit) {
            if ((f.top >= f.top_mark_hi or f.top_mark < hot_mark) and f.top + len <= cap) {
                const at = f.top;
                const mark = if (at < f.top_mark_hi) f.top_mark else 0;
                f.top += len;
                f.high = @max(f.high, f.top);
                return .{ .at = at, .mark = mark };
            } else if (f.top < f.top_mark_hi and f.top_mark >= hot_mark and f.top_mark_hi + len <= cap) {
                try f.gaps.append(gpa, .{ .lo = f.top, .hi = f.top_mark_hi, .mark = f.top_mark });
                f.top = f.top_mark_hi;
                f.top_mark = 0;
                const at = f.top;
                f.top += len;
                f.high = @max(f.high, f.top);
                return .{ .at = at, .mark = 0 };
            }
        }
        var best_i: ?usize = null;
        for (f.gaps.items, 0..) |g, i| {
            if (g.hi - g.lo < len) continue;
            if (best_i == null or g.mark < f.gaps.items[best_i.?].mark) best_i = i;
        }
        if (best_i) |bi| {
            const g = &f.gaps.items[bi];
            const at = g.lo;
            const mark = g.mark;
            if (g.hi - g.lo == len) _ = f.gaps.orderedRemove(bi) else g.lo += len;
            return .{ .at = at, .mark = mark };
        }
        const at = f.top;
        const mark = if (at < f.top_mark_hi) f.top_mark else 0;
        f.top += len;
        f.high = @max(f.high, f.top);
        return .{ .at = at, .mark = mark };
    }

    /// Releases `[lo, hi)` back to `Fit`, recording `mark` as the highest dispatch mark over these bytes.
    pub fn give(f: *Fit, gpa: std.mem.Allocator, lo_in: u64, hi_in: u64, mark_in: u64) Error!void {
        var lo = lo_in;
        var hi = hi_in;
        var mark = mark_in;
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
            if (mark > 0 or f.top_mark > 0) {
                f.top_mark = @max(f.top_mark, mark);
                f.top_mark_hi = @max(f.top_mark_hi, hi);
            }
            return;
        }
        try f.gaps.insert(gpa, i, .{ .lo = lo, .hi = hi, .mark = mark });
    }
};

/// Region demand for `t` without canary guards.
pub fn need(gpa: std.mem.Allocator, t: *const Trace, alignment: u64) Error!Need {
    return needWithGuard(gpa, t, alignment, 0);
}

/// Region demand for `t`, adding `guard` bytes to each scratch allocation to match `core` test canary sizing.
pub fn needWithGuard(gpa: std.mem.Allocator, t: *const Trace, alignment: u64, guard: u64) Error!Need {
    var w: Walk = .{ .gpa = gpa, .t = t, .alignment = alignment, .guard = guard };
    defer w.deinit();
    try w.run();
    return .{ .scratch = w.fit.high, .host = w.host_high };
}

const HostEntry = struct { end: u64, death: u32 };

const Walk = struct {
    gpa: std.mem.Allocator,
    t: *const Trace,
    alignment: u64,
    guard: u64,
    places: std.AutoHashMapUnmanaged(trace_mod.Ref, [2]u64) = .empty,
    fit: Fit = .{},
    host_live: std.AutoHashMapUnmanaged(trace_mod.Ref, HostEntry) = .empty,
    host_top: u64 = 0,
    host_high: u64 = 0,

    fn deinit(w: *Walk) void {
        w.places.deinit(w.gpa);
        w.fit.deinit(w.gpa);
        w.host_live.deinit(w.gpa);
    }

    fn run(w: *Walk) Error!void {
        for (w.t.events) |ev| switch (ev) {
            .ask => |ref| try w.ask(ref),
            .free => |ref| try w.free(ref),
            .wait => |deaths| w.wait(deaths),
        };
    }

    fn wait(w: *Walk, deaths: u32) void {
        var top: u64 = 0;
        var it = w.host_live.valueIterator();
        while (it.next()) |e| {
            // `death == none` (maxInt(u32)) is naturally `>= deaths`, as are asks dying at or after this wait.
            if (e.death >= deaths) top = @max(top, e.end);
        }
        w.host_top = top;
    }

    fn ask(w: *Walk, ref: trace_mod.Ref) Error!void {
        const a = w.t.asks[ref];
        if (a.external) return;
        const aligned = std.mem.alignForward(u64, @max(a.bytes, 1), w.alignment);
        if (a.host) {
            w.host_top += aligned;
            w.host_high = @max(w.host_high, w.host_top);
            try w.host_live.put(w.gpa, ref, .{ .end = w.host_top, .death = a.death });
            return;
        }
        const len = aligned + w.guard;
        const got = w.fit.take(len);
        try w.places.put(w.gpa, ref, .{ got.at, got.at + len });
    }

    fn free(w: *Walk, ref: trace_mod.Ref) Error!void {
        if (w.host_live.remove(ref)) return;
        const p = w.places.fetchRemove(ref) orelse return;
        try w.fit.give(w.gpa, p.value[0], p.value[1], 0);
    }
};

const testing = std.testing;

/// Builds a trace from `script`: 'a'..'z' asks `n` pages, 'A'..'Z' frees it, '!' marks next ask host-filled, '|' waits.
fn record(script: []const u8) !Trace {
    var r = trace_mod.Recorder.begin(testing.allocator);
    defer r.deinit();
    var refs: [26]?trace_mod.Ref = @splat(null);
    var host = false;
    for (script) |ch| switch (ch) {
        'a'...'z' => {
            refs[ch - 'a'] = try r.ask(@as(u64, ch - 'a' + 1) * 4096, host, false);
            host = false;
        },
        'A'...'Z' => r.free(refs[ch - 'A'].?),
        '!' => host = true,
        '|' => try r.wait(),
        else => unreachable,
    };
    return r.end();
}

test "region: first-fit takes lowest gap, coalesces neighbours, and reclaims host ring across waits" {
    const page: u64 = 4096;
    const cases = [_]struct { script: []const u8, scratch: u64, host: u64 = 0 }{
        // a (1 page), b (2) above it; b dies, then c (3) takes the top: 1 + 3 pages.
        .{ .script = "abBcAC", .scratch = 4 },
        // b (2) dies under a live d (4): c (3) does not fit b's gap and goes on top; then a (1) takes b's gap.
        .{ .script = "bdBcaDCA", .scratch = 2 + 4 + 3 },
        // Two dead neighbours join: c (3) fits where a (1) and b (2) were.
        .{ .script = "abdABcCD", .scratch = 1 + 2 + 4 },
        // Host asks never take scratch: they add up.
        .{ .script = "a!b!cABC", .scratch = 1, .host = 5 },
        // A wait gives back the host bytes of every host ask dead before it: c (3) lands where a and b were.
        .{ .script = "!a!bAB|!cC", .scratch = 0, .host = 3 },
        // A host ask alive at the wait keeps its bytes: c lands past a.
        .{ .script = "!a!bB|!cAC", .scratch = 0, .host = 1 + 3 },
        // An ask that dies after the wait was still alive at it.
        .{ .script = "!a|A!bB", .scratch = 0, .host = 1 + 2 },
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

test "region: gap marks and headroom helpers respect tiny fixtures and caps" {
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

    const tiny = withNarrowHeadroom(.{ .scratch = 4096, .host = 4096 }, 256);
    try testing.expectEqual(@as(u64, 6144), tiny.scratch);
    try testing.expectEqual(@as(u64, 6144), tiny.host);

    const real_narrow = withNarrowHeadroom(.{ .scratch = 200 << 20, .host = 10 << 20 }, 256);
    // 200 MiB: max(+15% = 230 MiB, +32 MiB = 232 MiB) = 232 MiB.
    try testing.expectEqual(@as(u64, 232 << 20), real_narrow.scratch);
    // 10 MiB host: max(+15% = 11.5 MiB, +min(5,4)=14 MiB) = 14 MiB.
    try testing.expectEqual(@as(u64, 14 << 20), real_narrow.host);

    const real_wide = withWideHeadroom(.{ .scratch = 2000 << 20, .host = 20 << 20 }, 256);
    try testing.expectEqual(@as(u64, 2100 << 20), real_wide.scratch);
    try testing.expectEqual(@as(u64, 24 << 20), real_wide.host);
}

test "region: takePreferCold avoids hot gap when headroom allows, falls back on tight cap, and coalesces" {
    var f: Fit = .{};
    defer f.deinit(testing.allocator);
    const page: u64 = 4096;
    const hot: u64 = 10;
    const p0 = try f.takePreferCold(testing.allocator, page, hot, 4 * page, 2 * page);
    const p1 = try f.takePreferCold(testing.allocator, page, hot, 4 * page, 2 * page);
    try f.give(testing.allocator, p0.at, p0.at + page, hot);
    // Avoids hot gap [0, 4096) at offset 2*page with mark 0.
    const cold = try f.takePreferCold(testing.allocator, page, hot, 4 * page, 2 * page);
    try testing.expectEqual(2 * page, cold.at);
    try testing.expectEqual(@as(u64, 0), cold.mark);

    // Free cold and p1 so top drops down over hot bytes; bumping top past top_mark_hi still picks cold when cap allows.
    try f.give(testing.allocator, cold.at, cold.at + page, 0);
    try f.give(testing.allocator, p1.at, p1.at + page, hot);
    try testing.expectEqual(@as(u64, 0), f.top);
    try testing.expectEqual(@as(u64, 0), f.gaps.items.len);
    const cold_bump = try f.takePreferCold(testing.allocator, page, hot, 4 * page, 2 * page);
    try testing.expectEqual(2 * page, cold_bump.at);
    try testing.expectEqual(@as(u64, 0), cold_bump.mark);

    // When cap leaves no room to bump past top_mark_hi, fall back to the coolest fitting gap.
    try f.give(testing.allocator, cold_bump.at, cold_bump.at + page, hot + 1);
    try testing.expectEqual(@as(u64, 0), f.top);
    const tight = try f.takePreferCold(testing.allocator, page, hot, 3 * page, 2 * page);
    try testing.expectEqual(@as(u64, 0), tight.at);
    try testing.expectEqual(hot + 1, tight.mark);

    // Distinct gaps behind live allocations: tight fallback picks the gap with the smallest mark.
    var f2: Fit = .{};
    defer f2.deinit(testing.allocator);
    const g0 = f2.take(page);
    const d0 = f2.take(page);
    const g1 = f2.take(page);
    const d1 = f2.take(page);
    _ = .{ d0, d1 };
    try f2.give(testing.allocator, g0.at, g0.at + page, hot + 5);
    try f2.give(testing.allocator, g1.at, g1.at + page, hot + 1);
    const coolest = try f2.takePreferCold(testing.allocator, page, hot, 4 * page, 2 * page);
    try testing.expectEqual(g1.at, coolest.at);
    try testing.expectEqual(hot + 1, coolest.mark);
}
