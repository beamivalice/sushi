//! The Sushi Cooler contract's late-kernel cells: after a load's served warm nothing compiles. Each runs alone, in a
//! process of its own (`alone_tests` in build.zig), since the pipeline cache is the process's. The runner and the
//! shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const arch_handle = @import("arch/handle.zig");
const arch_sched = @import("arch/sched.zig");
const SamplingParams = @import("sampler.zig").SamplingParams;
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const Load = contract.Load;
const Served = contract.Served;
const eachFixture = contract.eachFixture;
const eachTinyFixture = contract.eachTinyFixture;
const onTheCut = contract.onTheCut;
const greedy = contract.greedy;
const serialWith = contract.serialWith;

const every_length = struct {
    fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
        var cooler: Load = undefined;
        try cooler.open(C, dir, true);
        defer _ = cooler.close();
        {
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) @import("generate.zig").Generator.mtpVerifyDraftsMax(&cooler.config) else null;
            try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
        }
        const late = mlx.engine.latePipelines();
        mlx.engine.late_names.clear();
        var long: [40]u32 = undefined;
        for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
        var first_late: ?usize = null;
        for (1..long.len + 1) |len| {
            const prompt = long[0..len];
            var got = try greedy(&cooler, prompt, 6);
            got.deinit(testing.allocator);
            // Each length is its own prefill shape; a speculative request's verify and draw take the same shapes
            // after any of them, so one length runs them.
            if (len != long.len) {
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = len;
                continue;
            }
            if (@hasDecl(C, "specTokens")) {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokens(testing.allocator, cooler.xfm, prompt, 6);
                spec.deinit(testing.allocator);
            }
            if (@hasDecl(C, "specTokensWith")) {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var sampled = try C.specTokensWith(testing.allocator, cooler.xfm, prompt, 6, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 });
                sampled.deinit(testing.allocator);
            }
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = len;
        }
        errdefer std.debug.print("cooler contract: from a prompt of {?d} tokens on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
        try testing.expectEqual(late, mlx.engine.latePipelines());
    }
};

test "cooler contract: after a Cooler load's served warm, greedy requests of every prompt length to 40, and speculative and sampled ones, compile no kernel, on the tiny fixtures" {
    try eachTinyFixture(every_length.run);
}

test "cooler contract: after a Cooler load's served warm, greedy requests of every prompt length to 40, and speculative and sampled ones, compile no kernel, on the cut" {
    try onTheCut(every_length.run);
}

const widest_and_past_budget = struct {
    fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
        var cooler: Load = undefined;
        try cooler.open(C, dir, true);
        defer _ = cooler.close();
        const budget: usize = cooler.config.indexer_budget;
        if (budget == 0) return;
        var widest: u32 = 0;
        {
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            // The served cap: the adaptive depth row of this machine, and the widest a prompt-lookup chain drafts.
            const G = @import("generate.zig").Generator;
            const depth_cap = @import("mtp_policy.zig").mtpDepthCapForProfile(0, true, .generic);
            const verify_cap: ?u32 = if (@hasDecl(C, "specTokens")) G.mtpWarmVerifyCap(&cooler.config, depth_cap) else null;
            widest = @min(@max(depth_cap, G.mtpLookupDraftCap(&cooler.config)), G.mtpVerifyDraftsMax(&cooler.config));
            try @import("model_load.zig").warmLoad(io, testing.allocator, &cooler.config, cooler.xfm, verify_cap, null, cooler.xfm.cache.config);
        }
        const late = mlx.engine.latePipelines();
        mlx.engine.late_names.clear();
        const a = testing.allocator;
        // A chunk of 4096 rows is what a short prompt's request takes; a cycle of 61 tokens is what prompt lookup drafts.
        const wide = try a.alloc(u32, 4097);
        defer a.free(wide);
        for (wide, 0..) |*t, i| t.* = ids[i % ids.len] +% @as(u32, @intCast(i / ids.len));
        var first_late: ?[]const u8 = null;
        var got = try greedy(&cooler, wide, 4);
        got.deinit(a);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "the widest chunk";
        const past = try a.alloc(u32, budget + 400);
        defer a.free(past);
        for (past, 0..) |*t, i| t.* = ids[i % 61 % ids.len] +% @as(u32, @intCast(i % 61));
        got = try greedy(&cooler, past, 24);
        got.deinit(a);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "greedy past the budget";
        {
            // A prompt restored from the prefix cache prefills its tail past the budget, at any narrow width.
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            defer arch_sched.endRequest(&cooler.config);
            try @import("cooler_warm.zig").narrowAfter(io, a, cooler.xfm, past[0 .. budget + 61], 0);
        }
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "every narrow width past the budget";
        // A long chunk's remainder, or a restored prompt's tail, past the budget: wider than the narrow sweep,
        // narrower than the widest chunk.
        got = try serialWith(&cooler, past, 1, .{ .temperature = 0 }, .{ .pinned_prefill_chunk = 100 });
        got.deinit(a);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a 100-row chunk past the budget";
        if (@hasDecl(C, "specTokensAt")) {
            // Each draft width a round can take, past the budget: the planner picks them by the acceptance it measures.
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            var width: u32 = 1;
            while (width <= widest) : (width += 1) {
                depth_bounds.active = .{ .min = width, .max = width };
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensAt(a, cooler.xfm, past, width + 2, .{ .temperature = 0 }, widest);
                spec.deinit(a);
            }
            depth_bounds.active = saved;
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "each pinned width past the budget";
            for ([_]@import("sampler.zig").SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past, 48, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = if (sp.temperature == 0) "speculative past the budget" else "sampled speculative past the budget";
            }
            // Every round shape at the widest chunk's prompt: the planner opens them from the costs it measures.
            {
                const policy = @import("mtp_policy.zig");
                var plans: [64]policy.MtpRoundPlan = undefined;
                policy.warm_plans = policy.everyPlan(&plans, widest);
                defer policy.warm_plans = &.{};
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                const rounds_before = policy.warm_plan_at;
                var spec = try C.specTokensWith(a, cooler.xfm, wide, 96, .{ .temperature = 0 });
                spec.deinit(a);
                try testing.expect(policy.warm_plan_at - rounds_before >= policy.warm_plans.len);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "every round shape at the widest chunk";
            }
            // The release bench's request: the widest chunk's prompt, greedy with a nucleus and a top-k given.
            for ([_]@import("sampler.zig").SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 } }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, wide, 48, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = if (sp.top_k == 0) "speculative at the widest chunk" else "speculative at the widest chunk with top-p and top-k";
            }
            // Each penalty keeps MTP on: greedy and sampled, every penalty a request may set.
            for ([_]@import("sampler.zig").SamplingParams{
                .{ .temperature = 0, .repeat_penalty = 1.3 },
                .{ .temperature = 0, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
                .{ .temperature = 0.8, .top_k = 20, .seed = 3, .repeat_penalty = 1.3, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
            }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past, 48, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative with penalties";
            }
            // A context that crosses the budget: attention stays dense to there, over keys enough for two passes.
            if (budget > 16) for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past[0 .. budget - 8], 48, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative across the budget";
            };
            // Concurrent requests whose drafter rounds join, two rows and three, greedy and sampled.
            const drafter = @import("generate.zig").MtpHeadRef.fromModel(cooler.xfm, true, null).?;
            for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const prompts = [_][]const u32{ past[0..40], past[0..101], past[0..77] };
                for ([_]usize{ 2, 3 }) |rows| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    try @import("cooler_warm.zig").joinedRounds(io, a, cooler.xfm, drafter, prompts[0..rows], sp, 24);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "joined speculative rows";
                }
            }
            // An image request: its prefill and its verify rows splice the tower's rows and rotate by M-RoPE.
            if (cooler.config.image_token_id != 0 and cooler.config.image_token_id < cooler.config.vocab_size) {
                for ([_]?@import("generate.zig").MtpHeadRef{ drafter, null }) |d| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    try @import("cooler_warm.zig").imageRequest(io, a, cooler.xfm, d, .{ .temperature = 0 }, 300, 64, 24);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "an image request";
                }
            }
            // Through the scheduler: two requests decoded at once with the MTP head, whose ticks join their rows,
            // and a request that restores a longer one's prompt past the budget and drafts from there.
            var sv: Served = undefined;
            try sv.init(&cooler, dir);
            defer sv.deinit();
            try sv.concurrent(&.{ wide[0..40], wide[0..101] }, 64, .{ .temperature = 0 }, true);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "two concurrent requests";
            // Long enough that the tier holds restore points under it, as the SSD contract's prompt is.
            const restored = @max(budget, 1300) + 300;
            try sv.concurrent(&.{wide[0..restored]}, 8, .{ .temperature = 0 }, true);
            try sv.concurrent(&.{wide[0 .. restored + 400]}, 32, .{ .temperature = 0 }, true);
            try testing.expect(sv.reused > 0);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a request restored past the budget";
        }
        if (cooler.config.image_token_id != 0) {
            // An image request's prefill splices the tower's rows in at its placeholders, in the trunk's forward.
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            const s = cooler.xfm.s;
            const rows: c_int = 300;
            const hidden: c_int = @intCast(cooler.config.hidden_size);
            var tids: [300]i32 = undefined;
            for (&tids, 0..) |*t, i| t.* = if (i >= 40 and i < 104) @intCast(cooler.config.image_token_id) else @intCast(ids[i % ids.len]);
            const scope = mlx.engine.enter(s);
            defer scope.end();
            const tid = try mlx.newData(&tids, &[_]c_int{ 1, rows }, .int32);
            defer _ = mlx.mlx_array_free(tid);
            var ve = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ve);
            try mlx.check(mlx.mlx_zeros(&ve, &[_]c_int{ 1, 64, hidden }, 3, .bfloat16, s));
            var h = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&h, &[_]c_int{ 1, rows, hidden }, 3, .bfloat16, s));
            const spliced = try @import("trunk.zig").Trunk.spliceVisionRows(h, tid, ve, cooler.config.image_token_id, cooler.config.video_token_id, 0, s);
            defer _ = mlx.mlx_array_free(spliced);
            try mlx.check(mlx.mlx_array_eval(spliced));
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "an image's rows spliced into the prompt";
        }
        errdefer std.debug.print("cooler contract: from {?s} on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
        try testing.expectEqual(late, mlx.engine.latePipelines());
    }
};

test "cooler contract: after a Cooler load's served warm, a prompt at the widest chunk and a repetitive one past a sparse indexer's budget, greedy, speculative and sampled, compile no kernel, on the tiny fixtures" {
    try eachTinyFixture(widest_and_past_budget.run);
}

test "cooler contract: after a Cooler load's served warm, a prompt at the widest chunk and a repetitive one past a sparse indexer's budget, greedy, speculative and sampled, compile no kernel, on the cut" {
    try onTheCut(widest_and_past_budget.run);
}

test "cooler contract: after a Cooler load's served warm, a served request that restores the live state behind its reply compiles no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // Long enough that the request holds restore points under it, as the SSD contract's prompt does.
            var long: [1300]u32 = undefined;
            for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var from: u32 = 0;
            var first = try sv.request(&long, 24, &from);
            first.deinit(a);
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            // The bench's repeat: the same prompt again, its state rolled back from behind the first reply.
            var again = try sv.request(&long, 24, &from);
            again.deinit(a);
            errdefer std.debug.print("cooler contract: the repeat reused {d} tokens and compiled:\n{s}", .{ sv.reused, mlx.engine.late_names.text() });
            try testing.expect(sv.reused > 0);
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

test "cooler contract: after a Cooler load's served warm, the release bench's first request through the scheduler compiles no kernel" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            // The bench's first: a 4096-token prompt, greedy with a nucleus and a top-k given, its rounds as the planner
            // picks them.
            const prompt = try a.alloc(u32, 4096);
            defer a.free(prompt);
            for (prompt, 0..) |*t, i| t.* = ids[i % ids.len];
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const late = mlx.engine.latePipelines();
            mlx.engine.late_names.clear();
            var from: u32 = 0;
            var got = try sv.requestWith(prompt, 64, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 }, false, &from);
            got.deinit(a);
            errdefer std.debug.print("cooler contract: the bench's first request compiled:\n{s}", .{mlx.engine.late_names.text()});
            try testing.expectEqual(late, mlx.engine.latePipelines());
        }
    }.run);
}

/// A box whose free memory is `base` plus what the decoder pools gave back: a prefill chunk that does not fit it
/// releases pools until it does (the lend `dsv41_borrow.beforeForward` makes).
const LendProbe = struct {
    var engine: ?*@import("expert_stream.zig").Engine = null;
    var base: u64 = 0;
    /// The most pools out on loan at one read.
    var most: usize = 0;

    fn headroom(_: *const @import("model.zig").ModelConfig) @import("bills.zig").Headroom {
        const e = engine.?;
        var lent: u64 = 0;
        var out: usize = 0;
        for (e.borrow_from..e.layers.len) |li| {
            lent += e.lentBytes(@intCast(li));
            out += @intFromBool(e.isBorrowed(@intCast(li)));
        }
        most = @max(most, out);
        return .{ .live = base + lent };
    }
};

/// A prompt whose chunks after the first need more than the free memory holds, so its decoder pools are lent, through
/// the scheduler; the pools released and the kernels compiled late. Null where the load lends nothing.
fn lendingPrefill(ld: *Load, dir: []const u8, ids: []const u32) !?struct { released: usize, late: u64 } {
    const borrow = @import("dsv41/dsv41_borrow.zig");
    const xfm = arch_handle.trunk(ld.h);
    const engine = xfm.expert_stream orelse return null;
    if (engine.borrow_from >= engine.layers.len) return null;
    const m = @import("dsv41/dsv41_state.zig").modelOf(xfm).?;
    ld.config.pinned_prefill_chunk = 2048;
    const need = borrow.prefillNeed(&m.cfg, @import("cooler_side.zig").onEngine(&ld.config), 2048, 4096);
    LendProbe.engine = engine;
    defer LendProbe.engine = null;
    LendProbe.most = 0;
    LendProbe.base = need - engine.poolBytes(@intCast(engine.layers.len - 1)) / 2;
    borrow.headroom_fn = &LendProbe.headroom;
    defer borrow.headroom_fn = null;
    // Three chunks and the held-back span: the first fits, the second and third release pools.
    const prompt = try testing.allocator.alloc(u32, 3 * 2048 + @import("prefill_plan.zig").SSM_SNAPSHOT_BACKOFF + 1);
    defer testing.allocator.free(prompt);
    for (prompt, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
    var sv: Served = undefined;
    try sv.init(ld, dir);
    defer sv.deinit();
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var from: u32 = 0;
    var got = try sv.requestWith(prompt, 4, .{ .temperature = 0 }, false, &from);
    got.deinit(testing.allocator);
    return .{ .released = LendProbe.most, .late = mlx.engine.latePipelines() - late };
}

test "cooler contract: after a Cooler load's warm, and after one whose served warm its manifest skipped, a prefill that lends decoder pools compiles no kernel" {
    try eachTinyFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            if (comptime C != @import("dsv41/arch.zig").cooler_contract) return error.SkipZigTest;
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            var tmp = testing.tmpDir(.{ .iterate = true });
            defer tmp.cleanup();
            shared.root_hook = tmp.dir;
            defer shared.root_hook = null;
            defer shared.forget();
            mlx.engine.pipeline_registry.resetForTest();
            // The second load is a new process as far as its manifest goes: the pipelines the first made stay made in
            // this one, so what it shows is that the manifest's trust skips the served warm and the lend still compiles
            // nothing.
            for (0..2) |load| {
                if (load == 1) mlx.engine.pipeline_registry.resetForTest();
                const runs = cooler_warm.runs_for_test;
                var ld: Load = undefined;
                try ld.open(C, dir, true);
                defer _ = ld.close();
                try testing.expectEqual(@as(u32, if (load == 0) 1 else 0), cooler_warm.runs_for_test - runs);
                const r = (try lendingPrefill(&ld, dir, ids)) orelse return error.SkipZigTest;
                errdefer std.debug.print("cooler contract: load {d}: a prefill that lent {d} pools compiled:\n{s}", .{ load, r.released, mlx.engine.late_names.text() });
                try testing.expect(r.released > 0);
                try testing.expectEqual(@as(u64, 0), r.late);
            }
        }
    }.run);
}
