//! The ledger: one line for each owner of the memory Sushi Cooler holds, and one budget. The bill is the sum of the
//! lines, and an owner allocates only what it first reserved here, so the bill and the allocation are one number
//! (docs/bill_redesign.md#memory-map). Pure: the host tier builds it. Any thread may read or reserve.
const std = @import("std");

pub const Line = enum {
    /// What the loader maps: the pack's tensors and the weights a load builds.
    weights,
    /// The expert stream: slabs, bounce, union workspace, kept scales.
    slots,
    /// What a load makes on the engine and keeps: tables, maps, kernel configs' arrays.
    static,
    pipelines,
    /// The scratch arena and the host ring.
    arena,
    /// Metal's own overhead: the one named constant.
    margin,
    /// The state of the conversations being served, at its reserved capacity.
    state,
    /// The state of finished conversations still in memory: admission may flush it and take its room.
    parked,
    /// Host memory that follows a load.
    host,
    /// A request's decoded pixels, host memory that follows the request: given back when it ends.
    media,
};

const lines = std.enums.values(Line).len;

/// What `reserve` refused: the ask against what was committed under what budget.
pub const Refusal = struct { line: Line, asked: u64, committed: u64, budget: u64 };

pub const Snapshot = struct {
    budget: u64,
    committed: u64,
    by_line: [lines]u64,

    pub fn of(s: Snapshot, line: Line) u64 {
        return s.by_line[@intFromEnum(line)];
    }

    /// What the load holds whatever its requests: every line but `state`, `media` and `parked`.
    pub fn modelBill(s: Snapshot) u64 {
        return s.committed -| s.of(.state) -| s.of(.media) -| s.of(.parked);
    }

    /// What admission may take back, by what it waits for.
    pub const Freed = enum { parked, running };

    /// The bytes `freed` gives back: the parked state, and with the running requests' their `state` and `media`.
    pub fn freeable(s: Snapshot, freed: Freed) u64 {
        return switch (freed) {
            .parked => s.of(.parked),
            .running => s.of(.parked) +| s.of(.state) +| s.of(.media),
        };
    }

    /// The room a request of this load has once `freed` is given back.
    pub fn roomAfter(s: Snapshot, freed: Freed) u64 {
        return s.budget -| (s.committed -| s.freeable(freed));
    }
};

pub const Ledger = struct {
    /// No budget until a load sets one.
    limit: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    total: std.atomic.Value(u64) = .init(0),
    by_line: [lines]std.atomic.Value(u64) = @splat(.init(0)),
    seen: u64 = 0,
    unseen: std.atomic.Value(u64) = .init(0),
    /// What the load's requests hold at its fixed chunk and KV width: set once, when the chunk is fixed, and the only
    /// terms any reader of a request's bill on this load takes.
    terms: ?RequestTerms = null,
    /// The context its lines hold (`contextFor` over the room the model bill leaves), fixed once at the seal; 0 before it.
    context: std.atomic.Value(u32) = .init(0),

    /// The budget a load fixes when it ends.
    pub fn setBudget(l: *Ledger, bytes: u64) void {
        l.limit.store(bytes, .release);
    }

    pub fn budget(l: *const Ledger) u64 {
        return l.limit.load(.acquire);
    }

    pub fn committed(l: *const Ledger) u64 {
        return l.total.load(.acquire);
    }

    /// What may still be reserved.
    pub fn room(l: *const Ledger) u64 {
        return l.budget() -| l.committed();
    }

    pub fn of(l: *const Ledger, line: Line) u64 {
        return l.by_line[@intFromEnum(line)].load(.acquire);
    }

    /// Commits `bytes` to `line`, or refuses with nothing changed when the bill would pass the budget. An owner
    /// reserves before it allocates and allocates exactly what it reserved.
    pub fn reserve(l: *Ledger, line: Line, bytes: u64, refused: ?*Refusal) error{OverBudget}!void {
        var have = l.total.load(.acquire);
        while (true) {
            const limit = l.budget();
            if (bytes > limit -| have) {
                if (refused) |out| out.* = .{ .line = line, .asked = bytes, .committed = have, .budget = limit };
                return error.OverBudget;
            }
            have = l.total.cmpxchgWeak(have, have + bytes, .acq_rel, .acquire) orelse break;
        }
        _ = l.by_line[@intFromEnum(line)].fetchAdd(bytes, .acq_rel);
    }

    /// Gives back what a `reserve` committed, once the owner has freed it.
    pub fn release(l: *Ledger, line: Line, bytes: u64) void {
        std.debug.assert(l.of(line) >= bytes);
        _ = l.by_line[@intFromEnum(line)].fetchSub(bytes, .acq_rel);
        _ = l.total.fetchSub(bytes, .acq_rel);
    }

    /// Bytes already held go under `to` as `bytes`, the lines in `from` given back, in one step: a request's end parks
    /// what its state holds. No budget check: the storage exists, so a bill it passes refuses the next reserve.
    pub fn replace(l: *Ledger, from: []const struct { Line, u64 }, to: Line, bytes: u64) void {
        for (from) |f| {
            std.debug.assert(l.of(f[0]) >= f[1]);
            _ = l.by_line[@intFromEnum(f[0])].fetchSub(f[1], .acq_rel);
            _ = l.total.fetchSub(f[1], .acq_rel);
        }
        _ = l.by_line[@intFromEnum(to)].fetchAdd(bytes, .acq_rel);
        _ = l.total.fetchAdd(bytes, .acq_rel);
    }

    /// The load holds its budget whole from its seal (D90): `seen` is what the engine's active count holds of its
    /// model bill there. Every reading outside it then counts the rest (`unseenBeside`).
    pub fn holdWhole(l: *Ledger, seen: u64) void {
        l.seen = seen;
        l.publish();
    }

    /// What of the budget the engine's active count does not show now: the budget less the model bill it counts and
    /// the parked state, which its pool holds. A running request's state is counted beside it (D91). Call after
    /// `parked` moves; one writer, the inference thread.
    pub fn publish(l: *Ledger) void {
        l.setUnseen(l.budget() -| l.seen -| l.of(.parked));
    }

    /// What this load adds to a reading outside it (`unseenBeside`).
    pub fn unseenNow(l: *const Ledger) u64 {
        return l.unseen.load(.acquire);
    }

    /// The load ends: readings count it no more.
    pub fn letGo(l: *Ledger) void {
        l.setUnseen(0);
    }

    // The sum moves first on a rise and last on a fall, so `unseenBeside` errs high between the two.
    fn setUnseen(l: *Ledger, bytes: u64) void {
        const was = l.unseen.load(.acquire);
        if (bytes > was) {
            _ = unseen_total.fetchAdd(bytes - was, .acq_rel);
            l.unseen.store(bytes, .release);
        } else {
            l.unseen.store(bytes, .release);
            _ = unseen_total.fetchSub(was - bytes, .acq_rel);
        }
    }

    pub fn snapshot(l: *const Ledger) Snapshot {
        var s: Snapshot = .{ .budget = l.budget(), .committed = l.committed(), .by_line = undefined };
        for (&s.by_line, &l.by_line) |*out, *line| out.* = line.load(.acquire);
        return s;
    }
};

/// What every load on the ledger holds of its budget past the engine's active count (`Ledger.publish`).
var unseen_total: std.atomic.Value(u64) = .init(0);

/// What a memory reading adds to the active bytes for the loads on the ledger, each at its whole budget (D90): all of
/// them but `reader`'s own, whose bills read its lines.
pub fn unseenBeside(reader: ?*const Ledger) u64 {
    const own = if (reader) |r| r.unseen.load(.acquire) else 0;
    return unseen_total.load(.acquire) -| own;
}

/// What a load holds from its end, by owner (docs/bill_redesign.md part 4): fixed at the seal, committed on the load's
/// ledger before its first request.
pub const ModelBill = struct {
    weights: u64 = 0,
    slots: u64 = 0,
    static: u64 = 0,
    pipelines: u64 = 0,
    arena: u64 = 0,
    host: u64 = 0,
    margin: u64 = 0,

    const owners = [_]struct { Line, []const u8 }{
        .{ .weights, "weights" }, .{ .slots, "slots" }, .{ .static, "static" }, .{ .pipelines, "pipelines" },
        .{ .arena, "arena" },     .{ .host, "host" },   .{ .margin, "margin" },
    };

    pub fn sum(m: ModelBill) u64 {
        var total: u64 = 0;
        inline for (owners) |o| total +|= @field(m, o[1]);
        return total;
    }

    /// Commits every line, or none when the bill passes the budget.
    pub fn reserveOn(m: ModelBill, l: *Ledger, refused: ?*Refusal) error{OverBudget}!void {
        var done: usize = 0;
        errdefer {
            inline for (owners, 0..) |o, i| if (i < done) l.release(o[0], @field(m, o[1]));
        }
        inline for (owners) |o| {
            try l.reserve(o[0], @field(m, o[1]), refused);
            done += 1;
        }
    }
};

/// A request's bill in closed form over its reach (prompt, reply and speculative rows): its state `s` at `per_token` a
/// token plus a block whatever the reach (`fixed`), the restore points it keeps (`points`), and a copy of that state it
/// may keep beside it, at most `copy_cap` (the restore point a writer stages). The arch states each term as an upper
/// bound of what its storage holds at that reach; its forwards run on the load's arena (D97).
pub const RequestTerms = struct {
    per_token: u64,
    fixed: u64 = 0,
    points: u64 = 0,
    copy_cap: u64 = 0,

    pub fn at(t: RequestTerms, reach: u64) u64 {
        const s = t.per_token *| reach +| t.fixed;
        return s +| t.points +| @min(s, t.copy_cap);
    }
};

/// The context a load serves: the most tokens whose `RequestTerms.at` fits `room`, capped by the model's window; 0 when
/// not one token fits. `room` is the budget less the model bill. With `free = room - points`, the state `s` and its copy
/// take `s + min(s, copy_cap)`: `s` reaches `free - copy_cap` once `free` holds two caps, else `free / 2`.
pub fn contextFor(room: u64, t: RequestTerms, window: u32) u32 {
    std.debug.assert(t.per_token > 0);
    if (room < t.at(0)) return 0;
    const free = room -| t.points;
    const state = if (free / 2 >= t.copy_cap) free - t.copy_cap else free / 2;
    return @intCast(@min((state - t.fixed) / t.per_token, window));
}

/// What admission does with a request whose bill is `need`, decided on the ledger's lines alone.
pub const Verdict = enum {
    admit,
    /// It fits once the parked conversations' state is released.
    release_parked,
    /// It fits once the running requests end.
    wait,
    refuse,
};

pub fn decide(s: Snapshot, need: u64) Verdict {
    const fits = struct {
        fn without(snap: Snapshot, ask: u64, freed: u64) bool {
            return (snap.committed -| freed) +| ask <= snap.budget;
        }
    }.without;
    if (fits(s, need, 0)) return .admit;
    if (s.of(.parked) > 0 and fits(s, need, s.freeable(.parked))) return .release_parked;
    if (s.of(.state) > 0 and fits(s, need, s.freeable(.running))) return .wait;
    return .refuse;
}

/// The second gate, read after the ledger admits: free RAM below what the request will allocate means another program
/// holds memory the budget counted. It can only refuse.
pub fn freeRamRefuses(free_ram: u64, need: u64) bool {
    return need > free_ram;
}

/// `/props` `memory.ledger`: each line, the budget and what is committed.
pub fn writeJson(s: Snapshot, w: *std.Io.Writer) std.Io.Writer.Error!void {
    try w.writeAll("{\"lines\":{");
    for (std.enums.values(Line), 0..) |line, i| try w.print("{s}\"{t}\":{d}", .{ if (i == 0) "" else ",", line, s.of(line) });
    try w.print("}},\"budget\":{d},\"committed\":{d}}}", .{ s.budget, s.committed });
}

test "the bill is the sum of its lines, and a reserve past the budget is refused with its numbers and changes nothing" {
    const t = std.testing;
    var l: Ledger = .{};
    try l.reserve(.weights, 1 << 40, null);
    l.release(.weights, 1 << 40);
    l.setBudget(1000);
    try l.reserve(.weights, 600, null);
    try l.reserve(.arena, 150, null);
    try l.reserve(.state, 200, null);
    try t.expectEqual(@as(u64, 950), l.committed());
    try t.expectEqual(@as(u64, 50), l.room());
    var why: Refusal = undefined;
    try t.expectError(error.OverBudget, l.reserve(.state, 51, &why));
    try t.expectEqual(Refusal{ .line = .state, .asked = 51, .committed = 950, .budget = 1000 }, why);
    try t.expectEqual(@as(u64, 950), l.committed());
    try t.expectEqual(@as(u64, 200), l.of(.state));
    try l.reserve(.state, 50, null);
    const s = l.snapshot();
    var sum: u64 = 0;
    for (s.by_line) |b| sum += b;
    try t.expectEqual(s.committed, sum);
    try t.expectEqual(@as(u64, 250), s.of(.state));
}

test "reserves racing from several threads never pass the budget, and each one that was granted is counted" {
    const t = std.testing;
    var l: Ledger = .{};
    l.setBudget(1000 * 7);
    var granted = std.atomic.Value(u64).init(0);
    const Worker = struct {
        fn run(ledger: *Ledger, count: *std.atomic.Value(u64)) void {
            for (0..500) |_| {
                ledger.reserve(.state, 7, null) catch continue;
                _ = count.fetchAdd(1, .acq_rel);
            }
        }
    };
    var threads: [4]std.Thread = undefined;
    for (&threads) |*th| th.* = try std.Thread.spawn(.{}, Worker.run, .{ &l, &granted });
    for (threads) |th| th.join();
    try t.expectEqual(@as(u64, 1000), granted.load(.acquire));
    try t.expectEqual(@as(u64, 7000), l.committed());
    try t.expectEqual(@as(u64, 7000), l.of(.state));
}

test "the model bill is the sum of its lines, committed whole or not at all" {
    const t = std.testing;
    const bill: ModelBill = .{ .weights = 600, .slots = 50, .static = 30, .pipelines = 5, .arena = 70, .host = 15, .margin = 10 };
    try t.expectEqual(@as(u64, 780), bill.sum());
    var l: Ledger = .{};
    l.setBudget(779);
    var why: Refusal = undefined;
    try t.expectError(error.OverBudget, bill.reserveOn(&l, &why));
    try t.expectEqual(@as(u64, 0), l.committed());
    for (std.enums.values(Line)) |line| try t.expectEqual(@as(u64, 0), l.of(line));
    l.setBudget(1000);
    try bill.reserveOn(&l, null);
    try t.expectEqual(@as(u64, 780), l.committed());
    try t.expectEqual(@as(u64, 600), l.of(.weights));
    try t.expectEqual(@as(u64, 70), l.of(.arena));
    try t.expectEqual(@as(u64, 15), l.of(.host));
    try t.expectEqual(@as(u64, 10), l.of(.margin));
}

test "the context is the room less the request's fixed terms over its rate a token, capped by the window, and a request at it fits" {
    const t = std.testing;
    const terms: RequestTerms = .{ .per_token = 100, .fixed = 3500, .points = 500 };
    try t.expectEqual(@as(u64, 100 * 7 + 4000), terms.at(7));
    // (10,000 - 4,000) / 100.
    try t.expectEqual(@as(u32, 60), contextFor(10_000, terms, 1 << 20));
    try t.expect(terms.at(contextFor(10_000, terms, 1 << 20)) <= 10_000);
    try t.expect(terms.at(contextFor(10_000, terms, 1 << 20) + 1) > 10_000);
    try t.expectEqual(@as(u32, 60), contextFor(10_099, terms, 1 << 20));
    try t.expectEqual(@as(u32, 32), contextFor(10_000, terms, 32));
    // Not one token beside the fixed terms.
    try t.expectEqual(@as(u32, 0), contextFor(4_099, terms, 1 << 20));
    try t.expectEqual(@as(u32, 0), contextFor(100, terms, 1 << 20));
    try t.expectEqual(@as(u32, std.math.maxInt(u32)), contextFor(std.math.maxInt(u64), .{ .per_token = 1 }, std.math.maxInt(u32)));
}

fn lineSnapshot(budget: u64, held: []const struct { Line, u64 }) Snapshot {
    var s: Snapshot = .{ .budget = budget, .committed = 0, .by_line = @splat(0) };
    for (held) |h| {
        s.by_line[@intFromEnum(h[0])] += h[1];
        s.committed += h[1];
    }
    return s;
}

test "admission decides on the ledger's lines: admit, release the parked state, wait behind the running requests, or refuse" {
    const t = std.testing;
    const idle = lineSnapshot(1000, &.{.{ .weights, 600 }});
    try t.expectEqual(Verdict.admit, decide(idle, 400));
    try t.expectEqual(Verdict.refuse, decide(idle, 401));
    const parked = lineSnapshot(1000, &.{ .{ .weights, 600 }, .{ .parked, 300 } });
    try t.expectEqual(Verdict.admit, decide(parked, 100));
    try t.expectEqual(Verdict.release_parked, decide(parked, 101));
    try t.expectEqual(Verdict.release_parked, decide(parked, 400));
    try t.expectEqual(Verdict.refuse, decide(parked, 401));
    const running = lineSnapshot(1000, &.{ .{ .weights, 600 }, .{ .state, 200 }, .{ .parked, 100 } });
    try t.expectEqual(Verdict.admit, decide(running, 100));
    try t.expectEqual(Verdict.release_parked, decide(running, 200));
    try t.expectEqual(Verdict.wait, decide(running, 201));
    try t.expectEqual(Verdict.wait, decide(running, 400));
    try t.expectEqual(Verdict.refuse, decide(running, 401));
    // A running image request's pixels are given back at its end beside its state: a wait counts both.
    const image = lineSnapshot(1000, &.{ .{ .weights, 600 }, .{ .state, 200 }, .{ .media, 100 } });
    try t.expectEqual(Verdict.wait, decide(image, 350));
    try t.expectEqual(Verdict.refuse, decide(image, 401));
    try t.expectEqual(@as(u64, 400), image.roomAfter(.running));
    try t.expectEqual(@as(u64, 100), image.roomAfter(.parked));
    // A budget the model bill already passes refuses every request.
    try t.expectEqual(Verdict.refuse, decide(lineSnapshot(500, &.{.{ .weights, 600 }}), 0));
}

test "a load on the ledger holds its budget whole: a reading outside it counts what the engine does not show, less its parked state, and its own reading counts none of it" {
    const t = std.testing;
    var a: Ledger = .{};
    var b: Ledger = .{};
    defer a.letGo();
    defer b.letGo();
    const base = unseenBeside(null);
    a.setBudget(1000);
    try a.reserve(.weights, 600, null);
    a.holdWhole(550);
    try t.expectEqual(base + 450, unseenBeside(null));
    try t.expectEqual(base, unseenBeside(&a));
    b.setBudget(300);
    b.holdWhole(100);
    try t.expectEqual(base + 650, unseenBeside(null));
    try t.expectEqual(base + 200, unseenBeside(&a));
    try t.expectEqual(base + 450, unseenBeside(&b));
    // A request's end parks what its state holds, in the pool the active count reads.
    try a.reserve(.state, 300, null);
    a.replace(&.{.{ .state, 300 }}, .parked, 120);
    a.publish();
    try t.expectEqual(base + 330 + 200, unseenBeside(null));
    a.release(.parked, 120);
    a.publish();
    try t.expectEqual(base + 650, unseenBeside(null));
    a.letGo();
    try t.expectEqual(base + 200, unseenBeside(null));
}

test "free RAM is a gate that only refuses" {
    const t = std.testing;
    try t.expect(!freeRamRefuses(1000, 1000));
    try t.expect(freeRamRefuses(999, 1000));
    try t.expect(!freeRamRefuses(std.math.maxInt(u64), 0));
}

test "a request's end puts what its state holds under another line in one step, past the budget only by bytes already held" {
    const t = std.testing;
    var l: Ledger = .{};
    l.setBudget(1000);
    try l.reserve(.weights, 600, null);
    try l.reserve(.parked, 100, null);
    try l.reserve(.state, 250, null);
    l.replace(&.{ .{ .state, 250 }, .{ .parked, 100 } }, .parked, 180);
    try t.expectEqual(@as(u64, 780), l.committed());
    try t.expectEqual(@as(u64, 0), l.of(.state));
    try t.expectEqual(@as(u64, 180), l.of(.parked));
    // Held past the budget: the line says so, and the next reserve is refused.
    l.replace(&.{.{ .parked, 180 }}, .parked, 450);
    try t.expectEqual(@as(u64, 1050), l.committed());
    try t.expectError(error.OverBudget, l.reserve(.state, 1, null));
}

test "the ledger's JSON lists every line, the budget and what is committed" {
    const t = std.testing;
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeJson(lineSnapshot(1000, &.{ .{ .weights, 600 }, .{ .arena, 70 }, .{ .parked, 30 } }), &w);
    try t.expectEqualStrings("{\"lines\":{\"weights\":600,\"slots\":0,\"static\":0,\"pipelines\":0,\"arena\":70,\"margin\":0,\"state\":0,\"parked\":30,\"host\":0,\"media\":0},\"budget\":1000,\"committed\":700}", w.buffered());
}

test "a request at the context its room gives fits the room, and one token more does not, whatever its terms and its copy's cap" {
    const t = std.testing;
    const per_tokens = [_]u64{ 1, 7, 100, 10_838, 1 << 20 };
    const fixeds = [_]u64{ 0, 3, 14_810_838 };
    const pointss = [_]u64{ 0, 5, 1_000_000 };
    const caps = [_]u64{ 0, 1, 4096, 1_073_741_824 };
    const window: u32 = 1 << 30;
    for (per_tokens) |per_token| for (fixeds) |fixed| for (pointss) |points| for (caps) |cap| {
        const terms: RequestTerms = .{ .per_token = per_token, .fixed = fixed, .points = points, .copy_cap = cap };
        var room: u64 = 0;
        while (room < 1 << 36) : (room += room / 3 + 1) {
            const ctx = contextFor(room, terms, window);
            if (ctx == 0) {
                // Not even a request of one token: nothing fits, or it is the first that does not.
                try t.expect(room < terms.at(0) or terms.at(1) > room);
                continue;
            }
            try t.expect(terms.at(ctx) <= room);
            try t.expect(ctx == window or terms.at(@as(u64, ctx) + 1) > room);
        }
    };
    // Room for two caps: the state is the free room less one cap, (10,000 - 500 - 3,000 - 3,500) / 100.
    const capped: RequestTerms = .{ .per_token = 100, .fixed = 3500, .points = 500, .copy_cap = 3000 };
    try t.expectEqual(@as(u32, 30), contextFor(10_000, capped, window));
    // Short of two caps the copy is the state itself: 2 * 2,000 beside no points.
    try t.expectEqual(@as(u32, 20), contextFor(4_000, .{ .per_token = 100, .copy_cap = 3000 }, window));
    try t.expectEqual(@as(u32, 0), contextFor(capped.at(0) - 1, capped, window));
}
