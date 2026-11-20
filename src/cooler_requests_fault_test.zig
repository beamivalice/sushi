//! The Sushi Cooler contract's request fault sweeps: a request, a tick or joined rows failed or refused at each of their
//! sites leave the next request its tokens and the pool as it was. A step of its own beside
//! `cooler_requests_test.zig`'s; the runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const arch_sched = @import("arch/sched.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const archSite = contract.archSite;
const Load = contract.Load;
const MixedTick = contract.MixedTick;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const eachQwenServed = contract.eachQwenServed;
const firstAsks = contract.firstAsks;
const greedy = contract.greedy;
const qwenTinyServed = contract.qwenTinyServed;
const servedRows = contract.servedRows;
const servedRowsWith = contract.servedRowsWith;

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
