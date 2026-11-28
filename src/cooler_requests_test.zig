//! The Sushi Cooler contract's request cells: a request's tokens, plain, speculative, sampled and in rows, and its
//! checked-call faults. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const arch_sched = @import("arch/sched.zig");
const SamplingParams = @import("sampler.zig").SamplingParams;
const contract = @import("cooler_contract_test.zig");
const eachArch = contract.eachArch;
const serial = contract.serial;
const ContractOf = contract.ContractOf;
const Load = contract.Load;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachLayout = contract.eachLayout;
const eachResidency = contract.eachResidency;
const greedy = contract.greedy;


test "cooler contract: a Cooler load that serves more than one row gives each row its solo request's tokens" {
    // Each row's tokens equal its request run alone: rows of ragged lengths decoded as one forward, each against the same
    // prompt run alone. A load of one row has nothing to compare and is skipped.
    try eachArch(struct {
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
            try ld.loadLight(dir);
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
            try cooler.openLight(C, dir, true);
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
            return C.specTokens(testing.allocator, ld.xfm, prompt, 6);
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokens")) return error.SkipZigTest;
            const base = Sheet.now();
            var cooler: Load = undefined;
            try cooler.prepare(C, dir, true);
            if (cooler.config.expert_streaming and !cooler.config.stream_mtp_head) {
                cooler.config.deinit(testing.allocator);
                return error.SkipZigTest;
            }
            try cooler.loadLight(dir);
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
                try cooler.openLight(C, dir, true);
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

test "cooler contract: a wide forward's cap is its own model's to lift: another Cooler model's narrow forward between leaves it, and the model's own next narrow forward lifts it" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    mlx.engine.hold();
    defer arch_handle.engineHoldEnd();
    const cooler = @import("arch/cooler.zig");
    const config: @import("model.zig").ModelConfig = .{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4, .expert_quant_codebook = .spec1, .cooler_for_test = true };
    var made = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(made);
    try mlx.check(mlx.mlx_zeros(&made, &[_]c_int{64}, 1, .float32, mlx.engine.stream()));
    try mlx.check(mlx.mlx_array_eval(made));
    var a: mlx.engine.BudgetSlot = .{};
    var b: mlx.engine.BudgetSlot = .{};
    defer mlx.engine.forgetSlot(&a);
    defer mlx.engine.forgetSlot(&b);
    {
        const turn = mlx.engine.turnToModel(true, &a);
        defer turn.end();
        cooler.enterWide(&a, 1 << 20);
    }
    {
        const turn = mlx.engine.turnToModel(true, &b);
        defer turn.end();
        cooler.enterJoined(&config, &b);
    }
    const turn = mlx.engine.turnToModel(true, &a);
    defer turn.end();
    cooler.enterJoined(&config, &a);
    try testing.expectEqual(@as(?u64, null), capInForce());
}

/// The pool's cap in force; null when none is (a lifted cap is the largest one).
pub fn capInForce() ?u64 {
    const limit = mlx.engine.poolLimit() orelse return null;
    return if (limit == std.math.maxInt(u64)) null else limit;
}

pub fn poolInUse() u64 {
    const st = mlx.engine.stats();
    return st.pool_bytes - st.pool_free;
}

test "cooler contract: Qwen grouped MTP ticks after a DeepSeek request ends run under no cap the DeepSeek model left, and all serve" {
    try qwenBesideDeepSeek(struct {
        fn run(qsv: *Served, dsv: *Served, qids: []const u32, dids: []const u32) !void {
            const got = try ticksAfterDeepSeek(qsv, dsv, qids, dids, .{ 40, 60, 80 });
            errdefer std.debug.print("cooler contract: {d} of {d} Qwen ticks after the DeepSeek request ran under a cap, the last {d} KB over {d} KB the pool's tensors held\n", .{ got.capped, got.ticks, got.last_cap >> 10, got.last_in_use >> 10 });
            try testing.expectEqual(@as(u32, 0), got.capped);
            try got.served();
        }
    }.run);
}

test "cooler contract: Qwen grouped MTP ticks that grow their caches after a DeepSeek request add nothing to the DeepSeek model's cap" {
    try qwenBesideDeepSeek(struct {
        fn run(qsv: *Served, dsv: *Served, qids: []const u32, dids: []const u32) !void {
            try contract.requirePoolCap(dsv.entry.config.?);
            const got = try ticksAfterDeepSeek(qsv, dsv, qids, dids, .{ 200, 220, 240 });
            try testing.expect(got.ds_before != null);
            try testing.expectEqual(got.ds_before, got.ds_after);
            try got.served();
        }
    }.run);
}

/// The Qwen and DeepSeek STQ tinies loaded at once on Sushi Cooler, each served by a scheduler of its own.
fn qwenBesideDeepSeek(comptime body: fn (qsv: *Served, dsv: *Served, qids: []const u32, dids: []const u32) anyerror!void) !void {
    return qwenAndDeepSeekLoaded(struct {
        fn run(qld: *Load, qdir: []const u8, dld: *Load, ddir: []const u8, qids: []const u32, dids: []const u32) !void {
            var qsv: Served = undefined;
            try qsv.init(qld, qdir);
            defer qsv.deinit();
            var dsv: Served = undefined;
            try dsv.init(dld, ddir);
            defer dsv.deinit();
            try body(&qsv, &dsv, qids, dids);
        }
    }.run);
}

/// The Qwen and DeepSeek STQ tinies loaded at once on Sushi Cooler, for a body that serves them as it needs.
fn qwenAndDeepSeekLoaded(comptime body: fn (qld: *Load, qdir: []const u8, dld: *Load, ddir: []const u8, qids: []const u32, dids: []const u32) anyerror!void) !void {
    try @import("test_note.zig").needArchs(&.{ "qwen4", "dsv41" });
    try mlx.requireGpu();
    const Q = ContractOf("qwen4");
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(Q, "prompt") or !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.engine.fault.disarm();
    const qdir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const ddir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const qids = try Q.prompt(a, qdir);
    defer a.free(qids);
    const dids = try D.prompt(a, ddir);
    defer a.free(dids);
    var qld: Load = undefined;
    try qld.openLight(Q, qdir, true);
    defer _ = qld.close();
    var dld: Load = undefined;
    try dld.openLight(D, ddir, true);
    defer _ = dld.close();
    try body(&qld, qdir, &dld, ddir, qids, dids);
}

/// What the Qwen ticks after a DeepSeek request ran under (`ticksAfterDeepSeek`).
const AfterDeepSeek = struct {
    ticks: u32 = 0,
    /// Ticks that left a cap in force, the last such cap and the pool's bytes in use then.
    capped: u32 = 0,
    last_cap: u64 = 0,
    last_in_use: u64 = 0,
    /// The DeepSeek model's cap as its request ended, and after the Qwen ticks.
    ds_before: ?u64 = null,
    ds_after: ?u64 = null,
    refused: u64 = 0,
    failed: ?[]const u8 = null,

    /// No ask of the Qwen ticks was refused and every request served.
    fn served(self: AfterDeepSeek) !void {
        try testing.expectEqual(@as(u64, 0), self.refused);
        if (self.failed) |e| {
            std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
            return error.RequestFailed;
        }
    }
};

/// Three Qwen speculative requests of `lens` tokens, prefilled in turn, tick together at depth 2; after their third
/// tick a DeepSeek request runs to its cleanup, whose end leaves the DeepSeek model's resting cap in force, and the
/// Qwen requests tick on to their ends.
fn ticksAfterDeepSeek(qsv: *Served, dsv: *Served, qids: []const u32, dids: []const u32, lens: [3]usize) !AfterDeepSeek {
    const a = testing.allocator;
    const scheduler = Served.scheduler;
    const sch = &qsv.sch;
    const depth_bounds = @import("mtp_depth_bounds.zig");
    const saved = depth_bounds.active;
    defer depth_bounds.active = saved;
    depth_bounds.active = .{ .min = 2, .max = 2 };
    const head = @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(qsv.entry.handle.?), true, null);
    var long: [256]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = qids[(i * 7 + i / qids.len) % qids.len];
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |slot| sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    sch.queue_cap = 3;
    for (lens) |n| {
        const slot = try sch.submit(.{ .model = qsv.entry, .prompt_ids = long[0..n], .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 40, .enable_mtp = true, .mtp = head });
        slots[made] = slot;
        made += 1;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
    }
    const ds_slot = &arch_handle.trunk(dsv.entry.handle.?).cooler_budget;
    var got: AfterDeepSeek = .{};
    var refusals: u64 = 0;
    var at: u32 = 0;
    while (true) : (at += 1) {
        var live: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (slots[0..made]) |s| if (s.joinsTick()) {
            live[k] = s;
            k += 1;
        };
        if (k == 0) break;
        if (at == 3) {
            var prompt: [24]u32 = undefined;
            for (&prompt, 0..) |*t, i| t.* = dids[(i * 5 + 2) % dids.len];
            _ = try dsv.requestToCleanup(&prompt, 6, false);
            got.ds_before = ds_slot.cap;
            refusals = mlx.engine.poolRefusals();
        }
        try scheduler.runDecodeTick(sch, live[0..k]);
        scheduler.cullDecoding(sch);
        if (at < 3) continue;
        got.ticks += 1;
        if (capInForce()) |c| {
            got.capped += 1;
            got.last_cap = c;
            got.last_in_use = poolInUse();
        }
    }
    try testing.expect(got.ticks > 0);
    got.refused = mlx.engine.poolRefusals() - refusals;
    for (slots[0..made]) |slot| if (slot.error_code) |e| {
        got.failed = e;
    };
    got.ds_after = ds_slot.cap;
    return got;
}

/// DeepSeek DSpark requests decoded together: each slot's tokens, and the DSpark rounds every slot had run when all three
/// were decoding after one tick, and when the last one was left alone.
const DsparkCompany = struct {
    tokens: [3]std.ArrayList(u32) = .{ .empty, .empty, .empty },
    rounds_before_joined_tick: [3]u64 = @splat(0),
    rounds_after_joined_tick: [3]u64 = @splat(0),
    last_alone_rounds: u64 = 0,
    last_end_rounds: u64 = 0,
    failed: usize = 0,

    fn deinit(self: *DsparkCompany) void {
        for (&self.tokens) |*t| t.deinit(testing.allocator);
    }
};

/// DSpark requests over `prompts`, each to its own `max`, decoded in one scheduler as the server ticks them.
fn dsparkCompany(ld: *Load, dir: []const u8, prompts: []const []const u32, max: []const u32, sampling: SamplingParams) !DsparkCompany {
    const scheduler = Served.scheduler;
    const a = testing.allocator;
    var sv: Served = undefined;
    try sv.init(ld, dir);
    defer sv.deinit();
    const sch = &sv.sch;
    // Every request in flight at once: the harness's queue holds two.
    sch.queue_cap = @intCast(prompts.len);
    var run: DsparkCompany = .{};
    errdefer run.deinit();
    var slots: [3]?*scheduler.Slot = .{ null, null, null };
    defer {
        for (slots) |s| if (s) |x| sch.complete(x);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    const head = @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null);
    for (prompts, max, 0..) |p, m, i| {
        const s = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = sampling, .eos_token_ids = &.{}, .max_tokens = m, .enable_mtp = true, .mtp = head });
        slots[i] = s;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, s);
        if (!scheduler.postPrefillTerminal(sch, s)) try sch.decoding.append(a, s);
    }
    const n = prompts.len;
    const roundsOf = struct {
        fn f(s: *scheduler.Slot) u64 {
            return s.legacy_gen.?.drafter_stats.rounds;
        }
    }.f;
    var ticks: usize = 0;
    var alone_seen = false;
    while (ticks < 2000) : (ticks += 1) {
        var act: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (sch.decoding.items) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k == 0) break;
        if (ticks == 0 and k == n) for (slots[0..n], 0..) |s, i| {
            run.rounds_before_joined_tick[i] = roundsOf(s.?);
        };
        if (k == 1 and !alone_seen) {
            alone_seen = true;
            run.last_alone_rounds = roundsOf(act[0]);
        }
        scheduler.runDecodeTick(sch, act[0..k]) catch |e| for (act[0..k]) |s| s.markError(@errorName(e));
        if (ticks == 0 and k == n) for (slots[0..n], 0..) |s, i| {
            run.rounds_after_joined_tick[i] = roundsOf(s.?);
        };
        scheduler.cullDecoding(sch);
    }
    run.last_end_rounds = roundsOf(slots[n - 1].?);
    for (slots[0..n], 0..) |s, i| {
        const x = s.?;
        if (x.error_code) |code| {
            run.failed += 1;
            var msg: [300]u8 = undefined;
            const why = mlx.takeError(&msg);
            std.debug.print("dsparkCompany: request {d} of {d} failed: {s} ({?s})\n", .{ i, n, code, why });
            continue;
        }
        try run.tokens[i].appendSlice(a, x.legacy_gen.?.generated_ids.items);
    }
    return run;
}

test "cooler contract: three DeepSeek DSpark requests decoding at once run grouped DSpark rounds, each its solo DSpark text greedy and seeded, and the one left alone drafts again" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try D.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.openLight(D, dir, true);
    defer _ = ld.close();
    if ((arch_sched.rowCap(&ld.config) orelse 1) < 3) return error.SkipZigTest;
    var bufs: [3][18]u32 = undefined;
    var prompts: [3][]const u32 = undefined;
    for (&bufs, &prompts, 0..) |*b, *p, i| {
        for (b, 0..) |*t, j| t.* = ids[(j * (i + 2) + 3 * i) % ids.len];
        p.* = b[0 .. 10 + 4 * i];
    }
    const max = [3]u32{ 8, 14, 24 };
    // A sampled request draws each position once (its seed's), so its text is the same however its rounds ran.
    for ([_]SamplingParams{ .{ .temperature = 0.0 }, .{ .temperature = 0.8, .top_p = 0.95, .seed = 7 } }) |sampling| {
        var solo: [3]DsparkCompany = undefined;
        var made: usize = 0;
        defer for (solo[0..made]) |*r| r.deinit();
        for (0..3) |i| {
            solo[i] = try dsparkCompany(&ld, dir, prompts[i .. i + 1], max[i .. i + 1], sampling);
            made += 1;
            try testing.expectEqual(@as(usize, 0), solo[i].failed);
            try testing.expect(solo[i].last_end_rounds > 0);
        }
        var together = try dsparkCompany(&ld, dir, &prompts, &max, sampling);
        defer together.deinit();
        try testing.expectEqual(@as(usize, 0), together.failed);
        errdefer std.debug.print("dsparkCompany temp={d:.2}: before={any} after={any} alone={d} end={d} lens={d},{d},{d}\n", .{
            sampling.temperature,
            together.rounds_before_joined_tick,
            together.rounds_after_joined_tick,
            together.last_alone_rounds,
            together.last_end_rounds,
            together.tokens[0].items.len,
            together.tokens[1].items.len,
            together.tokens[2].items.len,
        });
        // Joined: requests decoding at once run grouped DSpark rounds and match solo DSpark token-for-token.
        try testing.expect(together.last_alone_rounds > 0);
        for (0..3) |i| {
            try testing.expectEqualSlices(u32, solo[i].tokens[0].items, together.tokens[i].items);
        }
        // Left alone, the last request drafts again.
        try testing.expect(together.last_end_rounds > together.last_alone_rounds);
    }
}

/// Requests through one scheduler as serve runs them: each prefilled at its tick, then every live row ticked together
/// in one `runDecodeTick`. A row's tokens are what it published; the planner's counts are its own.
pub const Group = struct {
    pub const Req = struct {
        prompt: []const u32,
        max: u32,
        mtp: bool = true,
        sampling: SamplingParams = .{ .temperature = 0 },
        /// Submitted once this many ticks have run (0: before the first).
        at_tick: u32 = 0,
        /// An M-RoPE table of text positions, as a request with an image builds (the row's tokens are not compared).
        mrope: bool = false,
        /// Another model's entry on the same scheduler; null: the served one.
        entry: ?*@import("model_registry.zig").LoadedModel = null,
    };

    pub const Out = struct {
        published: [8]std.ArrayList(u32) = @splat(.empty),
        failed: [8]?[]const u8 = @splat(null),
        ticks: u32 = 0,
        /// Per row: the planner's plain ticks, prime ticks and speculative rounds.
        planner: [8][3]u32 = @splat(.{ 0, 0, 0 }),

        pub fn deinit(self: *Out) void {
            for (&self.published) |*t| t.deinit(testing.allocator);
        }
    };

    pub fn run(sv: *Served, reqs: []const Req) !Out {
        const a = testing.allocator;
        const scheduler = Served.scheduler;
        const sch = &sv.sch;
        sch.queue_cap = @intCast(reqs.len + 1);
        var slots: [8]*scheduler.Slot = undefined;
        var made: usize = 0;
        var out: Out = .{};
        errdefer out.deinit();
        defer {
            for (slots[0..made]) |slot| sch.complete(slot);
            for (sch.cleanup_queue.items) |s| s.deinit();
            sch.cleanup_queue.clearRetainingCapacity();
        }
        var tick: u32 = 0;
        while (true) : (tick += 1) {
            while (made < reqs.len and reqs[made].at_tick <= tick) {
                const r = reqs[made];
                const entry = r.entry orelse sv.entry;
                const head = @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(entry.handle.?), true, null);
                var table: ?[]i32 = null;
                if (r.mrope) {
                    const t = try a.alloc(i32, 3 * r.prompt.len);
                    for (0..3) |d| for (0..r.prompt.len) |i| {
                        t[d * r.prompt.len + i] = @intCast(i);
                    };
                    table = t;
                }
                const slot = try sch.submit(.{ .model = entry, .prompt_ids = r.prompt, .sampling = r.sampling, .eos_token_ids = &.{}, .max_tokens = r.max, .enable_mtp = r.mtp, .mtp = if (r.mtp) head else null, .mrope_pos = table, .mrope_total = r.prompt.len });
                slots[made] = slot;
                _ = sch.pending.orderedRemove(0);
                try scheduler.runPrefill(sch, slot);
                made += 1;
                if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
            }
            var live: [8]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (slots[0..made]) |s| if (s.joinsTick()) {
                live[k] = s;
                k += 1;
            };
            if (k == 0 and made == reqs.len) break;
            if (k > 0) {
                try scheduler.runDecodeTick(sch, live[0..k]);
                scheduler.cullDecoding(sch);
            }
            if (tick > 4000) return error.TooManyTicks;
        }
        out.ticks = tick;
        for (slots[0..made], 0..) |s, i| {
            out.failed[i] = s.error_code;
            if (s.legacy_gen) |*g| out.planner[i] = .{ g.mtp_planner_plain_ticks, g.mtp_planner_prime_ticks, g.mtp_planner_spec_rounds };
            try out.published[i].appendSlice(a, s.out_buf.items);
        }
        return out;
    }

    /// The request alone, plain, on a fresh scheduler: the tokens a row in company must say.
    pub fn solo(ld: *Load, dir: []const u8, req: Req) !std.ArrayList(u32) {
        var sv: Served = undefined;
        try sv.init(ld, dir);
        defer sv.deinit();
        var one = [_]Req{.{ .prompt = req.prompt, .max = req.max, .mtp = false, .sampling = req.sampling, .mrope = req.mrope }};
        var o = try run(&sv, &one);
        defer o.deinit();
        if (o.failed[0]) |e| {
            std.debug.print("cooler contract: the solo request failed ({s})\n", .{e});
            return error.SoloFailed;
        }
        var got: std.ArrayList(u32) = .empty;
        try got.appendSlice(testing.allocator, o.published[0].items);
        return got;
    }

    /// Every row says its solo tokens.
    pub fn expectSolo(ld: *Load, dir: []const u8, reqs: []const Req, got: *const Out) !void {
        for (reqs, 0..) |r, i| {
            var want = try solo(ld, dir, r);
            defer want.deinit(testing.allocator);
            if (got.failed[i]) |e| {
                std.debug.print("cooler contract: row {d} failed ({s})\n", .{ i, e });
                return error.RequestFailed;
            }
            if (std.mem.indexOfDiff(u32, want.items, got.published[i].items)) |at| {
                std.debug.print("cooler contract: row {d} differs from its solo tokens at {d}\n", .{ i, at });
                return error.RowDiffersFromSolo;
            }
        }
    }
};

test "cooler contract: grouped speculative rows beside another Cooler model's request keep the planner's rounds and say their solo tokens" {
    try qwenAndDeepSeekLoaded(struct {
        fn run(qld: *Load, qdir: []const u8, dld: *Load, ddir: []const u8, qids: []const u32, dids: []const u32) !void {
            var bufs: [3][300]u32 = undefined;
            for (&bufs, 0..) |*b, r| for (b, 0..) |*t, i| {
                t.* = qids[(i * (r + 2) + 3 * r + 7) % qids.len];
            };
            var dbuf: [40]u32 = undefined;
            for (&dbuf, 0..) |*t, i| t.* = dids[(i * 5 + 2) % dids.len];
            var qsv: Served = undefined;
            try qsv.init(qld, qdir);
            defer qsv.deinit();
            var dsv: Served = undefined;
            try dsv.init(dld, ddir);
            defer dsv.deinit();
            // The DeepSeek request outlasts the Qwen rows, which are long enough to calibrate and choose.
            const set = [_]Group.Req{
                .{ .prompt = bufs[0][0..200], .max = 128 },
                .{ .prompt = bufs[1][0..250], .max = 132 },
                .{ .prompt = &dbuf, .max = 160, .mtp = false, .entry = dsv.entry },
                .{ .prompt = bufs[2][0..300], .max = 136, .at_tick = 4 },
            };
            var got = try Group.run(&qsv, &set);
            defer got.deinit();
            errdefer std.debug.print("cooler contract: beside a DeepSeek request: planner plain/prime/spec {any}\n", .{got.planner[0..4]});
            for ([_]usize{ 0, 1, 3 }) |i| try testing.expect(got.planner[i][2] > 0);
            const qwen = [_]Group.Req{ set[0], set[1], set[3] };
            const qgot = Group.Out{
                .published = .{ got.published[0], got.published[1], got.published[3], .empty, .empty, .empty, .empty, .empty },
                .failed = .{ got.failed[0], got.failed[1], got.failed[3], null, null, null, null, null },
            };
            try Group.expectSolo(qld, qdir, &qwen, &qgot);
            const dgot = Group.Out{ .published = .{ got.published[2], .empty, .empty, .empty, .empty, .empty, .empty, .empty }, .failed = .{ got.failed[2], null, null, null, null, null, null, null } };
            try Group.expectSolo(dld, ddir, set[2..3], &dgot);
        }
    }.run);
}

test "cooler contract: a served Hy4 STQ request with DFly2 says its serial tokens greedy and seeded, and its adaptive rounds take more than one width" {
    try contract.servedOn("hy4flash", &.{"HY4_STQ_TINY"}, struct {
        fn runDrafted(sv: *Served, prompt: []const u32, n: u32, sp: SamplingParams, head: @import("generate.zig").MtpHeadRef, hist: *[18]u32) !std.ArrayList(u32) {
            const a = testing.allocator;
            const sch = &sv.sch;
            const scheduler = Served.scheduler;
            const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = sp, .eos_token_ids = &.{}, .max_tokens = n, .enable_mtp = true, .mtp = head });
            defer {
                sch.complete(slot);
                for (sch.cleanup_queue.items) |s| s.deinit();
                sch.cleanup_queue.clearRetainingCapacity();
            }
            _ = sch.pending.orderedRemove(0);
            try scheduler.runPrefill(sch, slot);
            if (!scheduler.postPrefillTerminal(sch, slot)) {
                try sch.decoding.append(a, slot);
                var one = [_]*scheduler.Slot{slot};
                while (slot.joinsTick()) try scheduler.runDecodeTick(sch, &one);
                scheduler.cullDecoding(sch);
            }
            if (slot.error_code) |e| {
                std.debug.print("cooler contract: served Hy4 DFly2 request failed ({s})\n", .{e});
                return error.RequestFailed;
            }
            const g = slot.legacy_gen.?;
            hist.* = g.drafter_stats.rows_hist;
            var got: std.ArrayList(u32) = .empty;
            errdefer got.deinit(a);
            try got.appendSlice(a, g.generated_ids.items);
            return got;
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.openLight(C, dir, true);
            defer _ = ld.close();
            const head = @import("generate.zig").MtpHeadRef.fromModel(ld.xfm, true, null) orelse return error.Hy4DflyMissing;
            for ([_]usize{ 13, 30 }) |plen| {
                const prompt = ids[0..@min(ids.len, plen)];
                for ([_]SamplingParams{ .{ .temperature = 0.0 }, .{ .temperature = 0.8, .top_k = 20, .seed = 42 } }) |sp| {
                    var want = try serial(&ld, prompt, 32, sp);
                    defer want.deinit(a);
                    var sv: Served = undefined;
                    try sv.init(&ld, dir);
                    defer sv.deinit();
                    var hist: [18]u32 = @splat(0);
                    var got = try runDrafted(&sv, prompt, 32, sp, head, &hist);
                    defer got.deinit(a);
                    try testing.expectEqualSlices(u32, want.items, got.items);
                    var widths: usize = 0;
                    var rounds: u32 = 0;
                    for (hist) |c| {
                        rounds += c;
                        if (c > 0) widths += 1;
                    }
                    errdefer std.debug.print("cooler contract: Hy4 DFly2 plen={d} temp={d} took {d} distinct width(s) over {d} rounds, hist={any}\n", .{ plen, sp.temperature, widths, rounds, hist[1..9] });
                    try testing.expect(widths > 1 and rounds - hist[3] > 2);
                }
            }
        }
    }.run);
}
