//! The Sushi Cooler contract's request cells: a request's tokens, plain, speculative, sampled and in rows, and its
//! checked-call faults. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const arch_sched = @import("arch/sched.zig");
const SamplingParams = @import("sampler.zig").SamplingParams;
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const eachArch = contract.eachArch;
const archSite = contract.archSite;
const serial = contract.serial;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Generator = @import("generate.zig").Generator;
const ContractOf = contract.ContractOf;
const Load = contract.Load;
const MixedTick = contract.MixedTick;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const eachQwenServed = contract.eachQwenServed;
const eachResidency = contract.eachResidency;
const firstAsks = contract.firstAsks;
const greedy = contract.greedy;
const qwenTinyServed = contract.qwenTinyServed;
const servedRows = contract.servedRows;
const servedRowsWith = contract.servedRowsWith;


test "cooler contract: a Cooler load that serves more than one row gives each row its solo request's tokens" {
    // Each row's tokens equal its request run alone: rows of ragged lengths decoded as one forward, each against the same
    // prompt run alone. A load of one row has nothing to compare and is skipped.
    try eachFixture(struct {
        fn body(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.prepare(C, dir, true);
            const rows = arch_sched.coolerRows(&ld.config);
            if (rows.rows <= 1) {
                ld.config.deinit(a);
                return error.SkipZigTest;
            }
            if (comptime !@hasDecl(C, "greedyRows")) {
                ld.config.deinit(a);
                return error.CoolerRowsWithoutGreedyRows;
            } else try rowsAgainstSolo(C, &ld, dir, ids, rows.rows);
        }

        fn rowsAgainstSolo(comptime C: type, ld: *Load, dir: []const u8, ids: []const u32, rows: usize) !void {
            const a = testing.allocator;
            try ld.loadServed(dir);
            defer _ = ld.close();
            const n = @min(rows, 3);
            var prompts: [3][]const u32 = undefined;
            for (0..n) |i| prompts[i] = ids[0 .. ids.len - @min(ids.len - 1, i * 2)];
            // The rows run in the model's turn and end as serve ends a request: the arch's decl takes neither.
            const got = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try C.greedyRows(a, ld.xfm, prompts[0..n], 6);
            };
            defer {
                for (got) |*g| g.deinit(a);
                a.free(got);
            }
            for (prompts[0..n], got) |p, row| {
                var solo = try greedy(ld, p, 6);
                defer solo.deinit(a);
                try testing.expectEqualSlices(u32, solo.items, row.items);
            }
        }
    }.body);
}

test "cooler contract: a Cooler load's request and unload ask nothing of MLX" {
    try eachLayout(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            var closed = false;
            defer if (!closed) {
                _ = cooler.close();
            };
            const calls = mlx.engine.mlxCalls();
            var got = try greedy(&cooler, prompt, 12);
            got.deinit(testing.allocator);
            try testing.expectEqual(calls, mlx.engine.mlxCalls());
            // The job's end then pays MLX what it was owed while at rest: not the teardown's.
            closed = true;
            try testing.expectEqual(@as(u64, 0), cooler.close());
        }
    }.run);
}

test "cooler contract: a Cooler load's speculative requests ask nothing of MLX, and the unload gives back everything" {
    try eachLayout(struct {
        /// A speculative request, and the MLX calls it made to its end. The turn's end is not counted: returning to a
        /// side not at rest pays MLX what it is owed for its own earlier launches (`mlx.payOwed`), which no request asked.
        fn specAsked(ld: *Load, comptime C: type, prompt: []const u32, asked: *u64) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            const calls = mlx.engine.mlxCalls();
            defer asked.* = mlx.engine.mlxCalls() - calls;
            defer arch_sched.endRequest(&ld.config);
            return C.specTokens(testing.allocator, ld.xfm, prompt, 12);
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokens")) return error.SkipZigTest;
            const base = Sheet.now();
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            {
                errdefer _ = cooler.close();
                try testing.expect(C.headLoaded(cooler.xfm));
                // One row, past the verify edge, a chunk: the forwards a prompt length can take.
                for ([_]usize{ 1, 9, 40 }) |len| {
                    const prompt = ids[0..@min(ids.len, len)];
                    var asked: u64 = 0;
                    var got = try specAsked(&cooler, C, prompt, &asked);
                    got.deinit(testing.allocator);
                    errdefer std.debug.print("cooler contract: a speculative request over a prompt of {d} tokens asked MLX {d} times\n", .{ prompt.len, asked });
                    try testing.expectEqual(@as(u64, 0), asked);
                }
            }
            _ = cooler.close();
            try testing.expectEqual(base, Sheet.now());
        }
    }.run);
}

test "cooler contract: requests, repeated, leave the engine as they found it, and an unload gives back everything, load after load" {
    try eachResidency(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const prompt = ids[0..@min(ids.len, 30)];
            // A second load is what meets a first load's leftovers.
            for (0..2) |_| {
                const before = Sheet.now();
                var cooler: Load = undefined;
                try cooler.open(C, dir, true);
                {
                    errdefer _ = cooler.close();
                    for (0..2) |_| {
                        var g = try greedy(&cooler, prompt, 6);
                        g.deinit(testing.allocator);
                    }
                    const warm = Sheet.now();
                    var g = try greedy(&cooler, prompt, 6);
                    g.deinit(testing.allocator);
                    try testing.expectEqual(warm, Sheet.now());
                }
                _ = cooler.close();
                try testing.expectEqual(before, Sheet.now());
            }
        }
    }.run);
}

test "cooler contract: two unseeded sampled requests under one sampler decoded at once serve, plain and speculative, each row drawn under its own request's key (#106)" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var long: [120]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            // Rows that sample alike with no seed draw as one: the issue's sampler, temperature alone, and min-p.
            for ([_]SamplingParams{ .{ .temperature = 0.6, .top_p = 0.95, .top_k = 20 }, .{ .temperature = 0.8 }, .{ .temperature = 0.7, .min_p = 0.05 } }) |sp| {
                for ([_]bool{ false, true }) |mtp| try sv.concurrent(&.{ long[0..40], long[0..101] }, 32, sp, mtp);
            }
        }
    }.run);
}

test "cooler contract: a grouped tick after another request's prefill chunk runs under no cap that chunk installed, and both requests serve, plain and speculative (#106)" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var long: [300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            for ([_]bool{ false, true }) |mtp| try groupedAfterPrefill(&sv, &long, mtp);
        }
    }.run);
}

/// A request decodes alone; a second arrives and prefills (wide forwards, each installing its chunk's cap); then the
/// two decode as a group, to their ends.
fn groupedAfterPrefill(sv: *Served, long: []const u32, mtp: bool) !void {
    const a = testing.allocator;
    const scheduler = Served.scheduler;
    const sch = &sv.sch;
    const head = if (mtp) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
    var slots: [2]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |slot| sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    const uncapped = std.math.maxInt(u64);
    for ([_][]const u32{ long[0..40], long }) |p| {
        const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 24, .enable_mtp = mtp, .mtp = head });
        slots[made] = slot;
        made += 1;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
        if (made == 1) try scheduler.runDecodeTick(sch, slots[0..1]);
    }
    // The second request's last chunk left its cap in force.
    try testing.expect((mlx.engine.poolLimit() orelse uncapped) < uncapped);
    try scheduler.runDecodeTick(sch, slots[0..2]);
    // A grouped tick is a narrow forward: the prefill's cap is lifted for it, as for a one-request step.
    try testing.expectEqual(uncapped, mlx.engine.poolLimit() orelse uncapped);
    while (true) {
        var live: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (slots[0..made]) |slot| if (slot.joinsTick()) {
            live[k] = slot;
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(sch, live[0..k]);
        scheduler.cullDecoding(sch);
    }
    for (slots[0..made]) |slot| if (slot.error_code) |e| {
        std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
        return error.RequestFailed;
    };
}

test "cooler contract: two served requests as rows say their solo tokens, and a rows tick refused at any pool buffer site the solo requests lack leaves the next rows theirs and the pool as it was" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var p1: [37]u32 = undefined;
            for (&p1, 0..) |*t, i| t.* = ids[(i * 3) % ids.len];
            const prompts = [2][]const u32{ ids, &p1 };
            var solo: [2]std.ArrayList(u32) = undefined;
            for (0..2) |i| {
                var f: u32 = 0;
                solo[i] = try sv.request(prompts[i], 16, &f);
            }
            defer for (&solo) |*s| s.deinit(a);
            const Req = struct {
                sv: *Served,
                p: [2][]const u32,
                fn go(q: @This()) anyerror!void {
                    var o: [2]std.ArrayList(u32) = undefined;
                    try servedRows(q.sv, q.p, 16, false, &o);
                    for (&o) |*x| x.deinit(testing.allocator);
                }
            };
            var rows: [2]std.ArrayList(u32) = undefined;
            try servedRows(&sv, prompts, 16, false, &rows);
            for (&rows, 0..) |*x, i| {
                defer x.deinit(a);
                try testing.expectEqualSlices(u32, solo[i].items, x.items);
            }
            // The sweep's sites are the cut's too: it runs on the tiny, over the rows tick's own sites.
            if (!std.mem.eql(u8, dir, std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse ""))) return;
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            const Solo = struct {
                sv: *Served,
                p: [2][]const u32,
                fn go(q: @This()) anyerror!void {
                    for (q.p) |p| {
                        var f: u32 = 0;
                        var got = try q.sv.request(p, 16, &f);
                        got.deinit(testing.allocator);
                    }
                }
            };
            a.free(try firstAsks(a, &seen, Q, Solo{ .sv = &sv, .p = prompts }, Solo.go));
            const points = try firstAsks(a, &seen, Q, Req{ .sv = &sv, .p = prompts }, Req.go);
            defer a.free(points);
            const base = Sheet.now();
            for (points, 0..) |pt, at| {
                errdefer std.debug.print("cooler contract: rows point {d} of {d}\n", .{ at, points.len });
                mlx.engine.fault.armAt(pt);
                var o: [2]std.ArrayList(u32) = undefined;
                const r = servedRows(&sv, prompts, 16, false, &o);
                mlx.engine.fault.disarm();
                if (r) |_| {
                    for (&o, 0..) |*x, i| {
                        defer x.deinit(a);
                        try testing.expectEqualSlices(u32, solo[i].items, x.items);
                    }
                } else |_| {}
                try testing.expect(!mlx.errorPending());
                var o2: [2]std.ArrayList(u32) = undefined;
                try servedRows(&sv, prompts, 16, false, &o2);
                for (&o2, 0..) |*x, i| {
                    defer x.deinit(a);
                    try testing.expectEqualSlices(u32, solo[i].items, x.items);
                }
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u64, 0), now.open);
            }
        }
    }.run);
}

test "cooler contract: speculative rows failed at any checked MLX call site their solo plain requests lack, at its first call or its last, leave the next rows their solo tokens and the pool as it was" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            depth_bounds.active = .{ .min = 3, .max = 3 };
            var p1: [37]u32 = undefined;
            for (&p1, 0..) |*t, i| t.* = ids[(i * 3) % ids.len];
            const prompts = [2][]const u32{ ids[0..@min(ids.len, 23)], &p1 };
            var solo: [2]std.ArrayList(u32) = undefined;
            for (0..2) |i| {
                var f: u32 = 0;
                solo[i] = try sv.request(prompts[i], 12, &f);
            }
            defer for (&solo) |*s| s.deinit(a);
            const Rows = struct {
                sv: *Served,
                p: [2][]const u32,
                mtp: bool,
                fn go(q: @This()) anyerror!void {
                    var o: [2]std.ArrayList(u32) = undefined;
                    try servedRows(q.sv, q.p, 12, q.mtp, &o);
                    for (&o) |*x| x.deinit(testing.allocator);
                }
            };
            // A site the plain rows reach is the plain request's sweep's; the speculative ones are swept here, at
            // their last call too, where an array made by an earlier call of the step is still held.
            const sites = &mlx.fault.sites;
            defer sites.* = .{};
            const plain = try mlx.sweepSites(a, sites, Rows{ .sv = &sv, .p = prompts, .mtp = false }, Rows.go);
            defer a.free(plain);
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            for (plain) |pt| try seen.put(a, pt.site, {});
            sites.* = .{ .lasts = true };
            const all = try mlx.sweepSites(a, sites, Rows{ .sv = &sv, .p = prompts, .mtp = true }, Rows.go);
            defer a.free(all);
            sv.quiet_failures = true;
            const base = Sheet.now();
            for (all, 0..) |pt, at| {
                if (seen.contains(pt.site)) continue;
                errdefer std.debug.print("cooler contract: speculative rows failed at MLX point {d} of {d} (call {d} of its site)\n", .{ at, all.len, pt.n });
                sites.armAt(pt);
                var o: [2]std.ArrayList(u32) = undefined;
                const r = servedRows(&sv, prompts, 12, true, &o);
                mlx.fault.disarm();
                if (r) |_| {
                    for (&o, 0..) |*x, i| {
                        defer x.deinit(a);
                        try testing.expectEqualSlices(u32, solo[i].items, x.items);
                    }
                } else |_| {}
                try testing.expect(!mlx.errorPending());
                var o2: [2]std.ArrayList(u32) = undefined;
                try servedRows(&sv, prompts, 12, true, &o2);
                for (&o2, 0..) |*x, i| {
                    defer x.deinit(a);
                    try testing.expectEqualSlices(u32, solo[i].items, x.items);
                }
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u64, 0), now.open);
            }
        }
    }.run);
}

/// `n` ids of real text: Pride and Prejudice (public domain, the bench's corpus) through the fixture's own tokenizer.
pub fn realTextIds(a: std.mem.Allocator, dir: []const u8, n: usize) ![]u32 {
    const prose = @embedFile("fixtures/prose_pride_and_prejudice.txt");
    var tok = try @import("tokenizer.zig").loadTokenizer(io, a, dir);
    defer tok.deinit();
    const text = try tok.encode(a, prose[0..@min(prose.len, n * 8)]);
    defer a.free(text);
    try testing.expect(text.len >= n);
    return a.dupe(u32, text[0..n]);
}

test "cooler contract: on real text, joined decode rows say their solo rows' tokens for a long chunked prompt beside a short one (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4200);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // A long chunked row past the two-pass context beside a short one; three short rows are the tiny's.
    const sets = [_][]const usize{&.{ 4097, 17 }};
    for (sets) |lens| {
        var prompts: [8][]const u32 = undefined;
        for (lens, 0..) |l, i| prompts[i] = ids[i * 7 .. i * 7 + l];
        const got = blk: {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            break :blk try Q.greedyRows(a, ld.xfm, prompts[0..lens.len], 40);
        };
        defer {
            for (got) |*g| g.deinit(a);
            a.free(got);
        }
        for (prompts[0..lens.len], got) |p, row| {
            const one = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.greedyRows(a, ld.xfm, &.{p}, 40);
            };
            defer {
                for (one) |*g| g.deinit(a);
                a.free(one);
            }
            errdefer std.debug.print("cooler contract: rows {any}: the row of {d} tokens differs from its solo run at {?d}\n", .{ lens, p.len, std.mem.indexOfDiff(u32, one[0].items, row.items) });
            try testing.expectEqualSlices(u32, one[0].items, row.items);
        }
    }
}

/// An allocator that refuses its `at`-th allocation alone; every other call is its child's.
const RefuseOnce = struct {
    child: std.mem.Allocator,
    at: usize = std.math.maxInt(usize),
    n: usize = 0,
    fired: bool = false,

    fn allocator(self: *RefuseOnce) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *RefuseOnce = @ptrCast(@alignCast(ctx));
        defer self.n += 1;
        if (self.n == self.at) {
            self.fired = true;
            return null;
        }
        return self.child.rawAlloc(len, alignment, ra);
    }
    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *RefuseOnce = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(memory, alignment, new_len, ra);
    }
    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *RefuseOnce = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(memory, alignment, new_len, ra);
    }
    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *RefuseOnce = @ptrCast(@alignCast(ctx));
        self.child.rawFree(memory, alignment, ra);
    }
};

test "cooler contract: a tick unit failed by an error no latch explains fails its own requests only, not a drafting request or the joined rows beside it" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const scheduler = Served.scheduler;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if ((arch_sched.rowCap(&ld.config) orelse 1) < 2) return error.SkipZigTest;
            // The bucket path: a planner's round is one unit over every row it plans.
            const planner = @import("mtp_group_planner.zig");
            const planner_was = planner.enabled_override;
            planner.enabled_override = false;
            defer planner.enabled_override = planner_was;
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const sch = &sv.sch;
            sch.queue_cap = 3;
            var prompts: [3][40]u32 = undefined;
            for (&prompts, 0..) |*p, r| for (p, 0..) |*t, i| {
                t.* = ids[(i * (5 + 2 * r) + r + 1) % ids.len];
            };
            // The requests' own allocator, from their submit on (their generators keep it): refused once in a tick. It
            // lives past every tick, so a structure that kept it still reaches its child.
            var refuse: RefuseOnce = .{ .child = a };
            sch.allocator = refuse.allocator();
            defer sch.allocator = a;
            var asks: usize = 0;
            var k: usize = 0;
            var fired_n: usize = 0;
            var collateral: usize = 0;
            while (true) : (k += 1) {
                // One drafting request (a unit of its own) and two plain rows (one joined group), each decoding.
                var slots: [3]*scheduler.Slot = undefined;
                for (&slots, 0..) |*slot, r| {
                    const spec = r == 0;
                    const head = if (spec) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
                    slot.* = try sch.submit(.{ .model = sv.entry, .prompt_ids = &prompts[r], .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 24, .enable_mtp = spec, .mtp = head });
                    for (sch.pending.items, 0..) |p, i| if (p == slot.*) {
                        _ = sch.pending.orderedRemove(i);
                        break;
                    };
                    scheduler.runPrefill(sch, slot.*) catch |e| scheduler.failPrefill(slot.*, e);
                    if (!scheduler.postPrefillTerminal(sch, slot.*)) try sch.decoding.append(a, slot.*);
                }
                MixedTick.tick(sch);
                refuse.n = 0;
                refuse.fired = false;
                refuse.at = if (k == 0) std.math.maxInt(usize) else k - 1;
                MixedTick.tick(sch);
                refuse.at = std.math.maxInt(usize);
                if (k == 0) asks = refuse.n;
                const drafting_failed = slots[0].error_code != null;
                const rows_failed = slots[1].error_code != null or slots[2].error_code != null;
                if (k == 0) try testing.expect(!drafting_failed and !rows_failed);
                if (refuse.fired) fired_n += 1;
                if (refuse.fired and drafting_failed and rows_failed) {
                    collateral += 1;
                    if (collateral <= 3) std.debug.print("allocation {d} of {d} refused in a tick: drafting {s}, rows {s} / {s}\n", .{ k - 1, asks, slots[0].error_code.?, slots[1].error_code orelse "ok", slots[2].error_code orelse "ok" });
                }
                for (slots) |s| sch.complete(s);
                MixedTick.drain(sch);
                if (k > asks) break;
            }
            try testing.expect(fired_n > 0);
            try testing.expectEqual(@as(usize, 0), collateral);
        }
    }.run);
}

test "cooler contract: joined speculative rows under a logit bias, failed at any checked MLX call their unbiased rows lack, leave the next rows their tokens and the pool as it was" {
    // A repeat, presence or frequency penalty keeps a row out of a joined round; a logit bias does not, and shifts each
    // row's verify block after the joined forward made every row's.
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            depth_bounds.active = .{ .min = 3, .max = 3 };
            var p1: [37]u32 = undefined;
            for (&p1, 0..) |*t, i| t.* = ids[(i * 3) % ids.len];
            const prompts = [2][]const u32{ ids[0..@min(ids.len, 23)], &p1 };
            const bias = [_]@import("logit_bias.zig").Bias{.{ .id = ids[1], .delta = -2 }};
            const biased: @import("sampler.zig").SamplingParams = .{ .temperature = 0.0, .think_penalty = .{ .biases = &bias } };
            var want: [2]std.ArrayList(u32) = undefined;
            try servedRowsWith(&sv, prompts, 12, true, biased, &want);
            defer for (&want) |*x| x.deinit(a);
            const Rows = struct {
                sv: *Served,
                p: [2][]const u32,
                sampling: @import("sampler.zig").SamplingParams,
                fn go(q: @This()) anyerror!void {
                    var o: [2]std.ArrayList(u32) = undefined;
                    try servedRowsWith(q.sv, q.p, 12, true, q.sampling, &o);
                    for (&o) |*x| x.deinit(testing.allocator);
                }
            };
            const sites = &mlx.fault.sites;
            defer sites.* = .{};
            const plain = try mlx.sweepSites(a, sites, Rows{ .sv = &sv, .p = prompts, .sampling = .{ .temperature = 0.0 } }, Rows.go);
            defer a.free(plain);
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(a);
            for (plain) |pt| try seen.put(a, pt.site, {});
            sites.* = .{};
            const all = try mlx.sweepSites(a, sites, Rows{ .sv = &sv, .p = prompts, .sampling = biased }, Rows.go);
            defer a.free(all);
            sv.quiet_failures = true;
            const base = Sheet.now();
            var swept: usize = 0;
            for (all, 0..) |pt, at| {
                if (seen.contains(pt.site)) continue;
                errdefer std.debug.print("cooler contract: biased speculative rows failed at MLX point {d} of {d} (call {d} of its site)\n", .{ at, all.len, pt.n });
                sites.armAt(pt);
                var o: [2]std.ArrayList(u32) = undefined;
                const r = servedRowsWith(&sv, prompts, 12, true, biased, &o);
                swept += @intFromBool(sites.target == null);
                mlx.fault.disarm();
                if (r) |_| {
                    for (&o) |*x| x.deinit(a);
                } else |_| {}
                try testing.expect(!mlx.errorPending());
                var o2: [2]std.ArrayList(u32) = undefined;
                try servedRowsWith(&sv, prompts, 12, true, biased, &o2);
                for (&o2, 0..) |*x, i| {
                    defer x.deinit(a);
                    try testing.expectEqualSlices(u32, want[i].items, x.items);
                }
                const now = Sheet.now();
                try testing.expectEqual(base.in_use, now.in_use);
                try testing.expectEqual(@as(u64, 0), now.open);
            }
            try testing.expect(swept > 0);
        }
    }.run);
}

test "cooler contract: a Cooler load's sampled speculative requests, seeded or not, serve and ask nothing of MLX" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensWith")) return error.SkipZigTest;
            const a = testing.allocator;
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            const cases = [_]@import("sampler.zig").SamplingParams{
                .{ .temperature = 0.8, .top_p = 0.95 },
                .{ .temperature = 0.8, .top_p = 0.95, .seed = 7 },
                .{ .temperature = 1.0, .top_k = 20 },
            };
            // The sampler's draw follows no prompt length: each length's forward is the greedy cell's.
            const prompt = ids[0..@min(ids.len, 9)];
            for (cases) |sampling| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const calls = mlx.engine.mlxCalls();
                errdefer std.debug.print("cooler contract: sampled speculative request over {d} tokens, seed {?d}, top_k {d}\n", .{ prompt.len, sampling.seed, sampling.top_k });
                var got = try C.specTokensWith(a, cooler.xfm, prompt, 12, sampling);
                defer got.deinit(a);
                try testing.expectEqual(@as(usize, 12), got.items.len);
                try testing.expectEqual(calls, mlx.engine.mlxCalls());
            }
        }
    }.run);
}

test "cooler contract: a seeded sampled speculative request says the seeded serial request's tokens, again after another request and at every pinned depth" {
    try eachFixture(struct {
        fn spec(ld: *Load, comptime C: type, prompt: []const u32, n: u32, sampling: SamplingParams, depth: u32) !std.ArrayList(u32) {
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            defer arch_sched.endRequest(&ld.config);
            return C.specTokensAt(testing.allocator, ld.xfm, prompt, n, sampling, depth);
        }

        fn differs(what: []const u8, depth: u32, sampling: SamplingParams, len: usize, want: []const u32, got: []const u32) bool {
            const at = std.mem.indexOfDiff(u32, want, got) orelse return false;
            std.debug.print("cooler contract: seed {?d} over {d} tokens: {s} (pinned depth {d}) differs from token {d}\n", .{ sampling.seed, len, what, depth, at });
            return true;
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensAt") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            // Verify rows another plan computes at each row count give no position one token.
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            var long: [300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
            const cases = [_]SamplingParams{
                .{ .temperature = 0.8, .top_k = 20, .seed = 42 },
                .{ .temperature = 1.0, .top_p = 0.95, .top_k = 40, .min_p = 0.05, .seed = 11 },
            };
            var bad: u32 = 0;
            const len = long.len;
            const prompt: []const u32 = &long;
            for (cases) |sampling| {
                var want = try serial(&ld, prompt, 32, sampling);
                defer want.deinit(a);
                var first = try spec(&ld, C, prompt, 32, sampling, 0);
                defer first.deinit(a);
                if (differs("the speculative request against the serial one", 0, sampling, len, want.items, first.items)) bad += 1;
                // Another request between publishes the acceptance the next one plans from.
                {
                    var other = try spec(&ld, C, long[0..100], 24, .{ .temperature = 0 }, 0);
                    other.deinit(a);
                    var again = try spec(&ld, C, prompt, 32, sampling, 0);
                    defer again.deinit(a);
                    if (differs("the speculative request after another against the first", 0, sampling, len, first.items, again.items)) bad += 1;
                }
                for ([_]u32{ 1, 3, 6 }) |depth| {
                    depth_bounds.active = .{ .min = depth, .max = depth };
                    defer depth_bounds.active = saved;
                    var pinned = try spec(&ld, C, prompt, 32, sampling, depth);
                    defer pinned.deinit(a);
                    if (differs("the speculative request against the first", depth, sampling, len, first.items, pinned.items)) bad += 1;
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

/// One greedy request under a thinking bound, ticked in the scheduler's order (`loopGuardTick` before every tick,
/// `commitThoughtClose` when it fires).
const BoundRun = struct {
    /// `drained` decodes serial but leaves the pipeline before the thought's last token, as an MTP serial block's exit
    /// does, so the bound fires with no token in flight whatever the model accepts; `mtp` runs rounds at a pinned depth.
    const Mode = union(enum) { serial, drained, mtp: u32 };

    tokens: std.ArrayList(u32) = .empty,
    /// The logits the forced commit leaves for the token after it.
    logits: []f32 = &.{},
    /// The bound fired on a speculative round's exit (no pipelined token in flight).
    at_exit: bool = false,

    fn deinit(self: *BoundRun, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        a.free(self.logits);
    }

    fn run(ld: *Load, comptime C: type, prompt: []const u32, n: u32, budget: u32, forced: []const u32, mode: Mode) !BoundRun {
        const a = testing.allocator;
        const scheduler = @import("scheduler.zig");
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var bound = @import("mtp_plan.zig").ThinkBound{ .budget = budget, .opener_id = null, .closer_id = forced[forced.len - 2], .forced = forced, .in_think = true };
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        const options: Generator.InitOptions = if (mode == .mtp) C.specOptions(ld.xfm, mode.mtp) else .{};
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0, .think_bound = &bound }, &.{}, options);
        defer gen.deinit(a);
        var out: BoundRun = .{};
        errdefer out.deinit(a);
        while (true) {
            bound.observe(gen.generated_ids.items);
            if (bound.due()) {
                bound.fired = true;
                out.at_exit = gen.pending == null;
                const r = try gen.commitForcedTokens(a, forced);
                defer a.free(r.emitted);
                try out.tokens.appendSlice(a, r.emitted);
                if (r.stopped) break;
                out.logits = try test_support.qwen4ReadF32(a, gen.pending.?.logits, ld.xfm.s);
                if (!try gen.mtpResumeAfterForced(a)) {
                    gen.spec_disabled_runtime = true;
                    gen.spec_disable_reason = .think_bound;
                }
                continue;
            }
            if (mode == .mtp and !scheduler.thinkBoundForcesPlain(&gen)) {
                gen.mtp_group_cap = 0;
                scheduler.capMtpRoundToThinkBound(&gen);
                const r = (try gen.nextMtp(a)) orelse break;
                defer a.free(r.tokens);
                try out.tokens.appendSlice(a, r.tokens);
            } else if (mode == .drained and bound.room() == 1) switch (try gen.drainPipelineForSpec(a)) {
                .drained => |t| try out.tokens.append(a, t),
                .already_clean => return error.NoPipelineToDrain,
                .stopped => break,
            } else try out.tokens.append(a, (try gen.next(a)) orelse break);
        }
        if (!bound.fired) return error.BoundNeverFired;
        return out;
    }
};

test "cooler contract: a greedy request's thought closed by its budget says the serial request's tokens, whether the close lands on a serial tick or a speculative round's exit" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specOptions") or !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            // As wide as a served close (an early-stop line, the closer, a separator), past the verify rows.
            var forced: [24]u32 = undefined;
            for (&forced, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
            forced[forced.len - 2] = @intCast(ld.config.vocab_size - 1);
            var bad: u32 = 0;
            for ([_]u32{ 3, 6, 10, 15 }) |budget| {
                var want = try BoundRun.run(&ld, C, ids, 64, budget, &forced, .serial);
                defer want.deinit(a);
                try testing.expect(!want.at_exit);
                const modes = [_]BoundRun.Mode{ .drained, .{ .mtp = 1 }, .{ .mtp = 2 }, .{ .mtp = 3 }, .{ .mtp = 6 } };
                for (modes) |mode| {
                    if (mode == .mtp) depth_bounds.active = .{ .min = mode.mtp, .max = mode.mtp };
                    defer depth_bounds.active = saved;
                    var got = try BoundRun.run(&ld, C, ids, 64, budget, &forced, mode);
                    defer got.deinit(a);
                    if (mode == .drained) try testing.expect(got.at_exit);
                    const tokens_at = std.mem.indexOfDiff(u32, want.tokens.items, got.tokens.items);
                    const logits_same = std.mem.eql(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(got.logits));
                    if (tokens_at == null and logits_same) continue;
                    bad += 1;
                    const depth = if (mode == .mtp) mode.mtp else 0;
                    const flight = if (got.at_exit) "no token in flight" else "a token in flight";
                    const logits = if (logits_same) "equal" else "differ";
                    std.debug.print("cooler contract: budget {d}, {s} depth {d}, closed with {s}: tokens differ from {?d}, logits after the close {s}\n", .{ budget, @tagName(mode), depth, flight, tokens_at, logits });
                }
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

test "cooler contract: an exact Cooler speculative request says its greedy tokens, over one row, past the verify edge and a chunk" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specIsGreedy")) return error.SkipZigTest;
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            if (!C.specIsGreedy(&ld.config)) return error.SkipZigTest;
            for ([_]usize{ 1, 9, 40 }) |len| {
                const prompt = ids[0..@min(ids.len, len)];
                errdefer std.debug.print("cooler contract: a prompt of {d} tokens\n", .{prompt.len});
                var g = try greedy(&ld, prompt, 12);
                defer g.deinit(a);
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                var sp = try C.specTokens(a, ld.xfm, prompt, 12);
                defer sp.deinit(a);
                try testing.expectEqualSlices(u32, g.items, sp.items);
            }
        }
    }.run);
}

test "cooler contract: on real text, greedy speculative at depths 1, 3 and 6 says greedy's tokens, short and past the two-pass context (cut)" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    errdefer {
        var msg: [512]u8 = undefined;
        std.debug.print("cooler contract: fixture {s}: {s}\n", .{ dir, mlx.takeError(&msg) orelse "no MLX or engine error latched" });
    }
    const a = testing.allocator;
    const ids = try realTextIds(a, dir, 4097);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // Below and past the 2048 context where verify takes the two-pass kernel, the second a chunked prefill; the short
    // prompt arms are the tiny's.
    for ([_]usize{ 17, 4097 }) |len| {
        var greedy_said = try greedyOn(&ld, ids[0..len], 40);
        defer greedy_said.deinit(a);
        // The shallowest and deepest round and one between: each depth re-prefills the prompt.
        for ([_]u32{ 1, 3, 6 }) |depth| {
            var spec = blk: {
                const turn = arch_handle.turnFor(ld.h, &ld.config);
                defer turn.end();
                defer arch_sched.endRequest(&ld.config);
                break :blk try Q.specTokensAt(a, ld.xfm, ids[0..len], 40, .{ .temperature = 0.0 }, depth);
            };
            defer spec.deinit(a);
            errdefer std.debug.print("cooler contract: real text of {d} tokens at depth {d}: speculative differs from greedy at {?d}\n", .{ len, depth, std.mem.indexOfDiff(u32, greedy_said.items, spec.items) });
            try testing.expectEqualSlices(u32, greedy_said.items, spec.items);
        }
    }
}

fn greedyOn(ld: *Load, ids: []const u32, n: u32) !std.ArrayList(u32) {
    const a = testing.allocator;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.next(a)) |t| try got.append(a, t);
    return got;
}

test "cooler contract: a Cooler request failed at any of its checked MLX call sites leaves the engine as a whole request does, and the next says the same tokens" {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(testing.allocator);
    const S = struct {
        var sites: *std.AutoHashMapUnmanaged(u64, void) = undefined;
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            defer mlx.fault.disarm();
            const prompt = ids[0..@min(ids.len, 30)];
            var cooler: Load = undefined;
            try cooler.open(C, dir, true);
            defer _ = cooler.close();
            var want = try greedy(&cooler, prompt, 6);
            defer want.deinit(testing.allocator);
            {
                var again = try greedy(&cooler, prompt, 6);
                again.deinit(testing.allocator);
            }
            const base = Sheet.now();
            const Req = struct {
                ld: *Load,
                prompt: []const u32,
                fn go(r: @This()) anyerror!void {
                    var got = try greedy(r.ld, r.prompt, 6);
                    got.deinit(testing.allocator);
                }
            };
            // Each op under its callers fails once, at its first call, on the first fixture of the arch that reaches it.
            var list: std.ArrayList(u64) = .empty;
            defer list.deinit(testing.allocator);
            mlx.fault.sites = .{ .list = &list, .gpa = testing.allocator };
            const recorded = Req.go(.{ .ld = &cooler, .prompt = prompt });
            const unrecorded = mlx.fault.sites.failed;
            mlx.fault.sites = .{};
            try recorded;
            if (unrecorded) return error.OutOfMemory;
            var points: std.ArrayList(u64) = .empty;
            defer points.deinit(testing.allocator);
            for (list.items, 1..) |site, k| {
                if (!(try sites.getOrPut(testing.allocator, archSite(C, site))).found_existing) try points.append(testing.allocator, k);
            }
            for (points.items) |k| {
                mlx.fault.arm(k);
                var faulted = greedy(&cooler, prompt, 6);
                mlx.fault.disarm();
                if (faulted) |*g| g.deinit(testing.allocator) else |_| {}
                if (!mlx.fault.didFire()) {
                    std.debug.print("cooler contract: the request's checked call {d} was never reached: a request makes other calls than the one recorded\n", .{k});
                    return error.TestUnexpectedResult;
                }
                var msg: [300]u8 = undefined;
                _ = mlx.takeError(&msg);
                var got = try greedy(&cooler, prompt, 6);
                defer got.deinit(testing.allocator);
                errdefer std.debug.print("cooler contract: a request failed at checked MLX call {d}\n", .{k});
                try testing.expectEqualSlices(u32, want.items, got.items);
                try testing.expectEqual(base, Sheet.now());
            }
        }
    };
    S.sites = &seen;
    try eachLayout(S.run);
}
