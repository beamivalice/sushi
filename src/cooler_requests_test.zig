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
const serial = contract.serial;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Generator = @import("generate.zig").Generator;
const ContractOf = contract.ContractOf;
const Load = contract.Load;
const Served = contract.Served;
const Sheet = contract.Sheet;
const eachFixture = contract.eachFixture;
const eachLayout = contract.eachLayout;
const eachQwenServed = contract.eachQwenServed;
const eachResidency = contract.eachResidency;
const greedy = contract.greedy;


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
            return C.specTokens(testing.allocator, ld.xfm, prompt, 12);
        }

        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokens")) return error.SkipZigTest;
            const base = Sheet.now();
            var cooler: Load = undefined;
            try cooler.openLight(C, dir, true);
            {
                errdefer _ = cooler.close();
                if (cooler.config.expert_streaming and !cooler.config.stream_mtp_head) return error.SkipZigTest;
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

test "cooler contract: two unseeded sampled requests under one sampler decoded at once serve, plain and speculative, each row drawn under its own request's key (#106)" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
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
            try ld.openLight(Q, dir, true);
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

test "cooler contract: four requests, three speculative and one plain, the last prefilled beside three decoding, tick under no cap a prefill chunk installed, at every tick and every turn of the model after it, and all serve (#106)" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 4;
            var long: [101]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            // The issue's widths: every speculative row at depth 2. The plain row's tick joins the speculative rounds.
            try joinedAfterPrefill(&sv, &.{ long[0..40], long[0..60], long[0..80], long[0..101] }, &.{ true, true, true, false });
        }
    }.run);
}

test "cooler contract: three speculative requests, one restored from its last turn and prefilled beside two decoding, tick under no cap that prefill installed, at every tick and every turn of the model after it, and all serve (#112)" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 3;
            const a = testing.allocator;
            var conv: std.ArrayList(u32) = .empty;
            defer conv.deinit(a);
            for (0..1500) |i| try conv.append(a, ids[(i * 7 + i / ids.len) % ids.len]);
            var f: u32 = 0;
            var reply = try sv.request(conv.items, 16, &f);
            defer reply.deinit(a);
            try conv.appendSlice(a, reply.items);
            for (0..200) |i| try conv.append(a, ids[(i * 13 + 101) % ids.len]);
            // Rows of like lengths, as the issue's long sessions: the planner groups them only within its pad waste.
            var other: [1600]u32 = undefined;
            for (&other, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            try joinedAfterPrefill(&sv, &.{ other[0..1500], other[0..1600], conv.items }, &.{ true, true, true });
            try testing.expect(sv.reused > 0);
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

/// Requests arrive one at a time, each prefilled (wide forwards, each installing its chunk's cap) while those before it
/// decode, speculative rows at depth 2 where `mtp` says; then all of them tick together to their ends. After every
/// tick, and at the model's next turn, no cap is in force: a tick's forwards are narrow, and lift a chunk's cap.
fn joinedAfterPrefill(sv: *Served, prompts: []const []const u32, mtp: []const bool) !void {
    const a = testing.allocator;
    const scheduler = Served.scheduler;
    const sch = &sv.sch;
    const depth_bounds = @import("mtp_depth_bounds.zig");
    const saved = depth_bounds.active;
    defer depth_bounds.active = saved;
    depth_bounds.active = .{ .min = 2, .max = 2 };
    const head = @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null);
    var slots: [4]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer {
        for (slots[0..made]) |slot| sch.complete(slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    const refusals = mlx.engine.poolRefusals();
    var ticks: u32 = 0;
    for (prompts, mtp) |p, spec| {
        const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = p, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 24, .enable_mtp = spec, .mtp = if (spec) head else null });
        slots[made] = slot;
        made += 1;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, slot);
        sv.reused = slot.cached_tokens;
        if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(a, slot);
        if (made == prompts.len) try testing.expect(capInForce() != null);
        while (true) {
            var live: [4]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (slots[0..made]) |s| if (s.joinsTick()) {
                live[k] = s;
                k += 1;
            };
            if (k == 0) break;
            try scheduler.runDecodeTick(sch, live[0..k]);
            scheduler.cullDecoding(sch);
            ticks += 1;
            errdefer std.debug.print("cooler contract: after tick {d} of {d} rows a cap of {d} KB is in force over {d} KB the pool's tensors hold\n", .{ ticks, k, (capInForce() orelse 0) >> 10, poolInUse() >> 10 });
            try testing.expectEqual(@as(?u64, null), capInForce());
            {
                const turn = arch_handle.turnFor(sv.entry.handle, sv.entry.config);
                defer turn.end();
                errdefer std.debug.print("cooler contract: at the model's turn after tick {d} a cap of {d} KB is in force over {d} KB the pool's tensors hold\n", .{ ticks, (capInForce() orelse 0) >> 10, poolInUse() >> 10 });
                try testing.expectEqual(@as(?u64, null), capInForce());
            }
            // Requests before the last decode a few ticks while the next arrives; the last group ticks to its end.
            if (made < prompts.len and ticks % 3 == 0) break;
        }
    }
    try testing.expectEqual(refusals, mlx.engine.poolRefusals());
    for (slots[0..made]) |slot| if (slot.error_code) |e| {
        std.debug.print("cooler contract: a served request failed ({s})\n", .{e});
        return error.RequestFailed;
    };
}

/// The pool's cap in force; null when none is (a lifted cap is the largest one).
fn capInForce() ?u64 {
    const limit = mlx.engine.poolLimit() orelse return null;
    return if (limit == std.math.maxInt(u64)) null else limit;
}

fn poolInUse() u64 {
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
    try ld.openLight(Q, dir, true);
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

test "cooler contract: a Cooler load's sampled speculative requests, seeded or not, serve and ask nothing of MLX" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime !@hasDecl(C, "specTokensWith")) return error.SkipZigTest;
            const a = testing.allocator;
            var cooler: Load = undefined;
            try cooler.openLight(C, dir, true);
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
                // A fixture with no drafter skips, which is no failure to name.
                var skipped = false;
                errdefer if (!skipped) std.debug.print("cooler contract: sampled speculative request over {d} tokens, seed {?d}, top_k {d}\n", .{ prompt.len, sampling.seed, sampling.top_k });
                var got = C.specTokensWith(a, cooler.xfm, prompt, 12, sampling) catch |e| {
                    skipped = e == error.SkipZigTest;
                    return e;
                };
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
            try ld.openLight(C, dir, true);
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
            try ld.openLight(C, dir, true);
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
            try ld.openLight(C, dir, true);
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
    try ld.openLight(Q, dir, true);
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
const Group = struct {
    const Req = struct {
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

    const Out = struct {
        published: [8]std.ArrayList(u32) = @splat(.empty),
        failed: [8]?[]const u8 = @splat(null),
        ticks: u32 = 0,
        /// Per row: the planner's plain ticks, prime ticks and speculative rounds.
        planner: [8][3]u32 = @splat(.{ 0, 0, 0 }),

        fn deinit(self: *Out) void {
            for (&self.published) |*t| t.deinit(testing.allocator);
        }
    };

    fn run(sv: *Served, reqs: []const Req) !Out {
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
    fn solo(ld: *Load, dir: []const u8, req: Req) !std.ArrayList(u32) {
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
    fn expectSolo(ld: *Load, dir: []const u8, reqs: []const Req, got: *const Out) !void {
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

test "cooler contract: a plain request, or one near its token budget, beside long speculative rows leaves them the planner's speculation" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            var bufs: [3][120]u32 = undefined;
            for (&bufs, 0..) |*b, r| for (b, 0..) |*t, i| {
                t.* = ids[(i * (r + 3) + 5 * r + 1) % ids.len];
            };
            // A speculative third row is the control: the long rows speculate beside it. A plain third row cannot
            // speculate, and a short-budget one is always under the 96 tokens a calibration round needs.
            const thirds = [_]Group.Req{
                .{ .prompt = bufs[2][0..51], .max = 240 },
                .{ .prompt = bufs[2][0..51], .max = 240, .mtp = false },
            };
            for (thirds, 0..) |third, ti| {
                var sv: Served = undefined;
                try sv.init(&ld, dir);
                defer sv.deinit();
                const set = [_]Group.Req{ .{ .prompt = bufs[0][0..23], .max = 200 }, .{ .prompt = bufs[1][0..37], .max = 200 }, third };
                var got = try Group.run(&sv, &set);
                defer got.deinit();
                errdefer std.debug.print("cooler contract: third row {s}: planner plain/prime/spec {any}\n", .{ if (ti == 0) "speculative" else "plain", got.planner[0..3] });
                try testing.expect(got.planner[0][2] > 0);
                try testing.expect(got.planner[1][2] > 0);
                try Group.expectSolo(&ld, dir, &set, &got);
            }
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var shorts: [8]Group.Req = undefined;
            shorts[0] = .{ .prompt = bufs[0][0..23], .max = 200 };
            shorts[1] = .{ .prompt = bufs[1][0..37], .max = 200 };
            for (shorts[2..], 0..) |*r, k| r.* = .{ .prompt = bufs[2][k .. 40 + k], .max = 40, .at_tick = @intCast(k * 40) };
            var got = try Group.run(&sv, &shorts);
            defer got.deinit();
            errdefer std.debug.print("cooler contract: short-budget rows beside: planner plain/prime/spec {any}\n", .{got.planner[0..2]});
            try testing.expect(got.planner[0][2] > 0);
            try testing.expect(got.planner[1][2] > 0);
            try Group.expectSolo(&ld, dir, shorts[0..2], &got);
        }
    }.run);
}

test "cooler contract: a request that cannot join the planner's tick leaves the rows that can their speculation" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.openLight(Q, dir, true);
            defer _ = ld.close();
            var bufs: [3][700]u32 = undefined;
            for (&bufs, 0..) |*b, r| for (b, 0..) |*t, i| {
                t.* = ids[(i * (r + 3) + 5 * r + 1) % ids.len];
            };
            // Each third row has a reason of its own to decode outside the planner: a penalty its plain tick does not
            // sample, a position table it does not read, a context so much longer than
            // its neighbours' that padding them to it wastes the verify.
            const thirds = [_]struct { name: []const u8, req: Group.Req }{
                .{ .name = "penalty", .req = .{ .prompt = bufs[2][0..51], .max = 120, .sampling = .{ .temperature = 0, .presence_penalty = 0.5 } } },
                .{ .name = "mrope", .req = .{ .prompt = bufs[2][0..51], .max = 120, .mrope = true } },
                .{ .name = "pad waste", .req = .{ .prompt = bufs[2][0..650], .max = 120 } },
            };
            var bad: u32 = 0;
            for (thirds) |third| {
                var sv: Served = undefined;
                try sv.init(&ld, dir);
                defer sv.deinit();
                const set = [_]Group.Req{ .{ .prompt = bufs[0][0..23], .max = 200 }, .{ .prompt = bufs[1][0..37], .max = 200 }, third.req };
                var got = try Group.run(&sv, &set);
                defer got.deinit();
                // Every row says its solo tokens: the two that group, and the third that decodes in the regular path.
                if (got.planner[0][2] == 0 or got.planner[1][2] == 0) {
                    std.debug.print("cooler contract: a {s} row beside: planner plain/prime/spec {any}\n", .{ third.name, got.planner[0..3] });
                    bad += 1;
                }
                Group.expectSolo(&ld, dir, &set, &got) catch |e| {
                    std.debug.print("cooler contract: a {s} row beside: {s}\n", .{ third.name, @errorName(e) });
                    bad += 1;
                };
            }
            try testing.expectEqual(@as(u32, 0), bad);
        }
    }.run);
}

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
                .{ .prompt = bufs[0][0..200], .max = 200 },
                .{ .prompt = bufs[1][0..250], .max = 210 },
                .{ .prompt = &dbuf, .max = 400, .mtp = false, .entry = dsv.entry },
                .{ .prompt = bufs[2][0..300], .max = 220, .at_tick = 4 },
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

test "cooler contract: a served request on a load that holds its drafter drafts, greedy and seeded" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            // A load with no drafter has nothing to draft with.
            const head = @import("generate.zig").MtpHeadRef.fromModel(ld.xfm, true, null) orelse return error.SkipZigTest;
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.8, .top_k = 20, .seed = 42 } }) |sp| {
                var drafted: u64 = 0;
                var got = try contract.servedDrafts(&sv, ids[0..@min(ids.len, 30)], 16, sp, head, &drafted);
                defer got.deinit(a);
                errdefer std.debug.print("cooler contract: a served request at temperature {d} drafted nothing over {d} tokens\n", .{ sp.temperature, got.items.len });
                try testing.expect(drafted > 0);
            }
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
            try ld.open(C, dir, true);
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
