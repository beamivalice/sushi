//! The Sushi Cooler contract's restore cells: the SSD tier, a conversation's held restore points, and DeepSeek's
//! state handed between requests. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const model_load = @import("model_load.zig");
const arch_sched = @import("arch/sched.zig");
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const Served = contract.Served;
const firstAsks = contract.firstAsks;
const qwenTinyServed = contract.qwenTinyServed;
const servedOn = contract.servedOn;
const Sheet = contract.Sheet;

/// DeepSeek's one-row tiny load: the exclusive load, whose model state the next request takes.
fn dsv41OneRowServed(comptime body: fn (comptime D: type, dir: []const u8, ids: []const u32) anyerror!void) !void {
    return servedOn("dsv41", &.{"DSV41_TINY"}, body);
}

/// A DeepSeek request `a` prefilled over `long` whose client left as its prefill ended, and a next request over `other`
/// that fails before it publishes. `race`: the next one takes the model's state before `a`'s connection completes it,
/// as exclusive admission allows; else `a` is completed and drained first.
const CancelledAtPrefillEnd = struct {
    const scheduler = Served.scheduler;

    fn run(sv: *Served, long: []const u32, other: []const u32, race: bool) !void {
        const sch = &sv.sch;
        const a = try sch.submit(.{ .model = sv.entry, .prompt_ids = long, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 12 });
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, a);
        a.cancelled.store(true, .release);
        try testing.expect(scheduler.postPrefillTerminal(sch, a));
        if (!race) {
            sch.complete(a);
            drain(sv);
        }
        const b = try sch.submit(.{ .model = sv.entry, .prompt_ids = other, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 12 });
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, b);
        b.markError("Injected");
        scheduler.releaseNativeState(b);
        if (race) sch.complete(a);
        sch.complete(b);
        drain(sv);
    }

    /// As the inference thread's cleanup drain does.
    fn drain(sv: *Served) void {
        const sch = &sv.sch;
        for (sch.cleanup_queue.items) |s| {
            if (s.cancelled.load(.acquire) and !s.finished and s.error_code == null) {
                scheduler.commitCancelledAtCleanup(sch, s, scheduler.commitSlotIfApplicable);
                sv.entry.prefix_cache.?.flushPendingDisk(arch_handle.trunk(sv.entry.handle.?).s);
            }
            s.deinit();
        }
        sch.cleanup_queue.clearRetainingCapacity();
    }
};

fn cancelledPrompts(ids: []const u32, long: *[1300]u32, other: *[600]u32, cont: *[1340]u32) void {
    for (long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
    for (other, 0..) |*t, i| t.* = ids[(i * 5 + 2 + i / ids.len) % ids.len];
    @memcpy(cont[0..long.len], long);
    for (cont[long.len..], 0..) |*t, i| t.* = ids[(i * 3 + 5) % ids.len];
}

test "cooler contract: a DeepSeek request cancelled as its prefill ends, its cleanup run after the next request took the state, gives its continuation what a cleanup run first does, from RAM or the SSD tier" {
    try dsv41OneRowServed(struct {
        fn run(comptime D: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.openLight(D, dir, true);
            defer _ = ld.close();
            try testing.expectEqual(@as(usize, 1), arch_sched.rowCap(&ld.config) orelse 1);
            var long: [1300]u32 = undefined;
            var other: [600]u32 = undefined;
            var cont: [1340]u32 = undefined;
            cancelledPrompts(ids, &long, &other, &cont);
            // The continuation restores from the live record, or, after another conversation's handover flushed that
            // record, from the SSD tier; a restore path may change tokens, so each is held to its own ordered twin.
            for ([_]bool{ false, true }) |handover| {
                errdefer std.debug.print("cooler contract: the continuation {s}\n", .{if (handover) "after a handover" else "from RAM"});
                var got: [2]std.ArrayList(u32) = undefined;
                var reused: [2]u32 = undefined;
                for ([_]bool{ false, true }, 0..) |race, k| {
                    var sv: Served = undefined;
                    try sv.init(&ld, dir);
                    defer sv.deinit();
                    try CancelledAtPrefillEnd.run(&sv, &long, &other, race);
                    var f: u32 = 0;
                    if (handover) {
                        var o = try sv.request(ids[0..@min(ids.len, 40)], 4, &f);
                        o.deinit(a);
                    }
                    got[k] = try sv.request(&cont, 12, &f);
                    reused[k] = sv.reused;
                }
                defer for (&got) |*g| g.deinit(a);
                try testing.expect(reused[0] > 0);
                try testing.expectEqual(reused[0], reused[1]);
                try testing.expectEqualSlices(u32, got[0].items, got[1].items);
            }
        }
    }.run);
}

test "cooler contract: a DeepSeek request committed after another took the state publishes no record over that state" {
    try dsv41OneRowServed(struct {
        fn run(comptime D: type, dir: []const u8, ids: []const u32) !void {
            const scheduler = Served.scheduler;
            var ld: Load = undefined;
            try ld.openLight(D, dir, true);
            defer _ = ld.close();
            var long: [1300]u32 = undefined;
            var other: [600]u32 = undefined;
            var cont: [1340]u32 = undefined;
            cancelledPrompts(ids, &long, &other, &cont);
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const sch = &sv.sch;
            const first = try sch.submit(.{ .model = sv.entry, .prompt_ids = &long, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 12 });
            _ = sch.pending.orderedRemove(0);
            try scheduler.runPrefill(sch, first);
            const next = try sch.submit(.{ .model = sv.entry, .prompt_ids = &other, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 12 });
            _ = sch.pending.orderedRemove(0);
            try scheduler.runPrefill(sch, next);
            defer {
                for ([_]*scheduler.Slot{ first, next }) |s| {
                    s.markError("Injected");
                    scheduler.releaseNativeState(s);
                    sch.complete(s);
                }
                CancelledAtPrefillEnd.drain(&sv);
            }
            // A commit of `first` run now, whatever deferred it, finds the state `next`'s.
            const live = @import("dsv41/dsv41_prefix.zig").liveOf(&sv.entry.prefix_cache.?);
            const was = .{ live.ready, live.tokens.len, live.points.items.len };
            scheduler.commitSlotIfApplicable(sch, first);
            try testing.expectEqual(was, .{ live.ready, live.tokens.len, live.points.items.len });
        }
    }.run);
}

/// Admission that refuses a warm share the request did not take over: a copy of the prefix does not fit.
pub fn fitsOnlyTaken(_: *const @import("model.zig").ModelConfig, _: usize, _: u32, _: @import("trunk.zig").KVQuantConfig, _: bool, matched: u64, _: u64, will_donate: bool, _: bool, _: @import("scheduler.zig").CheckpointPlan) bool {
    return matched == 0 or will_donate;
}

/// The #109 conversation served: turn 1, a turn 2 that extends it (its entry holds both prompt ends' restore points),
/// and the prompt of a turn 3 that leaves turn 2's text between those points (turn 2 less its last 100 tokens, then
/// `say` new ones): a partial share of turn 2's entry.
pub const PartialShare = struct {
    /// Turn 2's prompt, and the tokens of the entry its commit left (the prompt and its reply).
    second: std.ArrayList(u32) = .empty,
    entry: std.ArrayList(u32) = .empty,
    third: std.ArrayList(u32) = .empty,

    pub fn serve(sv: *Served, ids: []const u32, say: usize) !PartialShare {
        const a = testing.allocator;
        var c: PartialShare = .{};
        errdefer c.deinit();
        var f: u32 = 0;
        var first = try heldPrompt(a, ids, 1500, 0);
        defer first.deinit(a);
        var reply = try sv.request(first.items, 16, &f);
        defer reply.deinit(a);
        try c.second.appendSlice(a, first.items);
        try c.second.appendSlice(a, reply.items);
        var said = try heldPrompt(a, ids, 1500, 2);
        defer said.deinit(a);
        try c.second.appendSlice(a, said.items);
        var reply2 = try sv.request(c.second.items, 16, &f);
        defer reply2.deinit(a);
        try c.entry.appendSlice(a, c.second.items);
        try c.entry.appendSlice(a, reply2.items);
        try c.third.appendSlice(a, c.second.items[0 .. c.second.items.len - 100]);
        var more = try heldPrompt(a, ids, say, 1);
        defer more.deinit(a);
        try c.third.appendSlice(a, more.items);
        return c;
    }

    pub fn deinit(self: *PartialShare) void {
        self.second.deinit(testing.allocator);
        self.entry.deinit(testing.allocator);
        self.third.deinit(testing.allocator);
    }
};

/// A hot cache at rest: no entry is checked out, and the cache bills the sum of its entries' bills.
pub fn takeoverAtRest(hc: *const @import("prefix_cache.zig").HotPrefixCache) bool {
    var sum: u64 = 0;
    for (hc.entries.items) |*e| {
        if (e.checkout != null) return false;
        sum += e.kv_bytes;
    }
    return sum == hc.current_kv_bytes;
}

test "cooler contract: a served Qwen takeover turn refused at each pool site it reaches, under a roomy and a tight RAM budget, leaves the hot cache at rest and the pool as it was, and the next turn is served" {
    try qwenTinyServed(struct {
        ld: *Load,
        dir: []const u8,
        ids: []const u32,
        ram: u64,

        /// Turns 1 and 2, then the partial takeover turn 3, refused at `at` (none: served). A sweep records turn 3 alone.
        fn turns(q: @This(), sv: *Served, at: ?mlx.FaultSites.SitePoint) !void {
            const rec = &mlx.engine.fault.sites;
            const recording = rec.list != null;
            if (recording) rec.paused = true;
            var c = try PartialShare.serve(sv, q.ids, 200);
            defer c.deinit();
            const fits_was = Served.scheduler.prefill_admission_fits;
            defer Served.scheduler.prefill_admission_fits = fits_was;
            Served.scheduler.prefill_admission_fits = fitsOnlyTaken;
            if (recording) rec.paused = false;
            if (at) |pt| {
                // As the inference loop runs a failing request: failPrefill or failTick, then the cleanup drain.
                mlx.engine.fault.armAt(pt);
                _ = try sv.requestToCleanup(c.third.items, 16, false);
                return;
            }
            var f: u32 = 0;
            var got = try sv.request(c.third.items, 16, &f);
            got.deinit(testing.allocator);
        }

        fn go(q: @This()) anyerror!void {
            var sv: Served = undefined;
            try sv.initTiers(q.ld, q.dir, q.ram);
            defer sv.deinit();
            try q.turns(&sv, null);
        }

        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            // The tight budget restores from the SSD tier, a QSA overlay included: the same sites as the roomy budget's
            // unwind through another caller there, so each budget sweeps its own.
            for ([_]u64{ 1 << 30, 24 << 20 }) |ram| {
                var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
                defer seen.deinit(a);
                const q: @This() = .{ .ld = &ld, .dir = dir, .ids = ids, .ram = ram };
                const points = try firstAsks(a, &seen, Q, q, go);
                defer a.free(points);
                try testing.expect(points.len > 0);
                const base = Sheet.now();
                for (points, 1..) |pt, k| {
                    errdefer std.debug.print("cooler contract: the takeover turn under a {d} MB RAM tier refused at pool site {d} of {d}\n", .{ ram >> 20, k, points.len });
                    {
                        var sv: Served = undefined;
                        try sv.initTiers(&ld, dir, ram);
                        defer sv.deinit();
                        sv.quiet_failures = true;
                        try q.turns(&sv, pt);
                        if (pt.strict) try testing.expect(!mlx.engine.fault.armed());
                        mlx.engine.fault.disarm();
                        const hc = &sv.entry.prefix_cache.?;
                        if (hc.disk) |*d| d.drainWriter();
                        try testing.expect(takeoverAtRest(hc));
                        try testing.expect(!mlx.errorPending());
                        // The conversation goes on past its third turn.
                        var f: u32 = 0;
                        var next = try heldPrompt(a, ids, 3300, 0);
                        defer next.deinit(a);
                        var got = try sv.request(next.items, 8, &f);
                        got.deinit(a);
                        try testing.expect(takeoverAtRest(hc));
                    }
                    const now = Sheet.now();
                    try testing.expectEqual(base.in_use, now.in_use);
                    try testing.expectEqual(@as(u32, 0), now.open);
                }
            }
        }
    }.run);
}

test "cooler contract: a streamed Qwen Cooler load decodes one row per forward, so a restore while two requests decode patches no live group and the engine refuses nothing" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const route = @import("tensor/route.zig");
            const scheduler = Served.scheduler;
            var ld: Load = undefined;
            try ld.prepare(Q, dir, true);
            {
                errdefer ld.config.deinit(a);
                // The smallest budget the pack plans: its experts read from the SSD, as `--ssd-budget-gb` loads it.
                var budget: u64 = 1 << 30;
                const plan = while (budget <= 64 << 30) : (budget += 1 << 30) {
                    break model_load.planExpertStreamingWith(io, a, &ld.config, dir, budget, false, 0) catch continue;
                } else return error.TestUnexpectedResult;
                model_load.applyStreamingPlan(&ld.config, plan, budget);
                if (ld.config.expert_source_dir == null) ld.config.expert_source_dir = try a.dupe(u8, dir);
                try testing.expect(ld.config.expert_streaming and !ld.config.expert_resident);
            }
            try ld.loadLight(dir);
            defer _ = ld.close();
            try testing.expect(ld.onEngine());
            // The gate: one row per forward, so no tick joins two requests and no group goes live.
            try testing.expectEqual(@as(?usize, 1), arch_sched.rowCap(&ld.config));
            try testing.expect(!arch_sched.modelBatches(ld.xfm));
            const group = &ld.xfm.arch_state.qwen4.ssm_group;
            var sv: Served = undefined;
            try sv.initTiers(&ld, dir, 1 << 30);
            defer sv.deinit();
            sv.sch.queue_cap = 3;
            const refused0 = route.refusals;
            var f: u32 = 0;
            var first = try heldPrompt(a, ids, 1500, 0);
            defer first.deinit(a);
            var reply = try sv.request(first.items, 8, &f);
            defer reply.deinit(a);
            const sch = &sv.sch;
            var slots: [3]*scheduler.Slot = undefined;
            var made: usize = 0;
            defer {
                for (slots[0..made]) |slot| sch.complete(slot);
                for (sch.cleanup_queue.items) |s| s.deinit();
                sch.cleanup_queue.clearRetainingCapacity();
            }
            var other: [2]std.ArrayList(u32) = undefined;
            for (&other, 0..) |*p, i| p.* = try heldPrompt(a, ids, 300, 3 + i);
            defer for (&other) |*p| p.deinit(a);
            // The third extends the first conversation: it restores a checkpoint while the other two decode.
            var next: std.ArrayList(u32) = .empty;
            defer next.deinit(a);
            try next.appendSlice(a, first.items);
            try next.appendSlice(a, reply.items);
            try next.appendSlice(a, other[0].items[0..40]);
            for ([_][]const u32{ other[0].items, other[1].items, next.items }) |p| {
                const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8 });
                slots[made] = slot;
                made += 1;
                _ = sch.pending.orderedRemove(0);
                try scheduler.runPrefill(sch, slot);
                if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
                if (made == 2) try scheduler.runDecodeTick(sch, slots[0..2]);
                try testing.expect(!group.live);
            }
            try testing.expect(slots[2].cached_tokens > 0);
            while (true) {
                var live: [3]*scheduler.Slot = undefined;
                var k: usize = 0;
                for (slots[0..made]) |slot| if (slot.joinsTick()) {
                    live[k] = slot;
                    k += 1;
                };
                if (k == 0) break;
                try scheduler.runDecodeTick(sch, live[0..k]);
                try testing.expect(!group.live);
                scheduler.cullDecoding(sch);
            }
            for (slots[0..made]) |slot| if (slot.error_code) |e| {
                std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
                return error.RequestFailed;
            };
            try testing.expectEqual(@as(u64, 0), route.refusals - refused0);
        }
    }.run);
}

/// A Qwen conversation of `n` tokens, its own: `seed` picks the ids.
pub fn heldPrompt(a: std.mem.Allocator, ids: []const u32, n: usize, seed: usize) !std.ArrayList(u32) {
    var p: std.ArrayList(u32) = .empty;
    errdefer p.deinit(a);
    for (0..n) |i| try p.append(a, ids[(i * 7 + i / ids.len + seed * 131) % ids.len]);
    return p;
}
