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
const greedy = contract.greedy;
const serialWith = contract.serialWith;

/// Opens `dir` on `C` with the full verify cap warmed once (no double-warm), then verifies that:
/// 1. greedy lengths 1..40 (`n = 1` for `len < 40`, `n = 6` at `len == 40`, plus speculative & sampled at 40),
/// 2. when `indexer_budget > 0`, the widest chunk, past-budget, penalties, joined rows, image, and restore-past-budget checks,
/// 3. live-state restore behind a reply, and
/// 4. the release bench's first request (`n = 16`)
/// compile no kernel after the load's served warm.
fn checkLateFixture(comptime C: type, dir: []const u8, ids: []const u32) !void {
    const a = testing.allocator;
    const G = @import("generate.zig").Generator;
    var cooler: Load = undefined;
    try cooler.prepare(C, dir, true);
    const verify_cap: ?usize = if (@hasDecl(C, "specTokens")) G.mtpVerifyDraftsMax(&cooler.config) else null;
    try cooler.loadServedWith(dir, verify_cap);
    defer _ = cooler.close();
    if (!std.meta.eql(cooler.xfm.cache.config, @import("bills.zig").kvCacheForLaunch(&cooler.config, null).config)) {
        const turn = arch_handle.turnFor(cooler.h, &cooler.config);
        defer turn.end();
        const cap: ?u32 = if (verify_cap) |v| @intCast(v) else null;
        try @import("model_load.zig").warmLoad(io, a, &cooler.config, cooler.xfm, cap, null, cooler.xfm.cache.config);
    }

    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var first_late: ?[]const u8 = null;

    // 1. Prompt lengths 1..40, plus speculative & sampled at 40.
    {
        var long: [40]u32 = undefined;
        for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
        for (1..long.len + 1) |len| {
            const prompt = long[0..len];
            var got = try greedy(&cooler, prompt, if (len < long.len) 1 else 6);
            got.deinit(a);
            if (len != long.len) {
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a short prompt length";
                continue;
            }
            if (@hasDecl(C, "specTokens") and C.headLoaded(cooler.xfm)) {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokens(a, cooler.xfm, prompt, 6);
                spec.deinit(a);
            }
            if (@hasDecl(C, "specTokensWith") and C.headLoaded(cooler.xfm)) {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var sampled = try C.specTokensWith(a, cooler.xfm, prompt, 6, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 });
                sampled.deinit(a);
            }
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a 40-token prompt";
        }
    }

    // 2. Widest chunk and past-budget sparse-indexer checks (when the arch has an indexer budget).
    const budget: usize = cooler.config.indexer_budget;
    if (budget > 0) {
        const depth_cap = @import("mtp_policy.zig").mtpDepthCapForProfile(0, true, .generic);
        const widest = @min(@max(depth_cap, G.mtpLookupDraftCap(&cooler.config)), G.mtpVerifyDraftsMax(&cooler.config));
        // A chunk of 4096 rows is what a short prompt's request takes; a cycle of 61 tokens is what prompt lookup drafts.
        const wide = try a.alloc(u32, 4097);
        defer a.free(wide);
        for (wide, 0..) |*t, i| t.* = ids[i % ids.len] +% @as(u32, @intCast(i / ids.len));
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
            const turn = arch_handle.turnFor(cooler.h, &cooler.config);
            defer turn.end();
            defer arch_sched.endRequest(&cooler.config);
            try @import("cooler_warm.zig").narrowAfter(io, a, cooler.xfm, past[0 .. budget + 61], 0);
        }
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "every narrow width past the budget";

        got = try serialWith(&cooler, past, 1, .{ .temperature = 0 }, .{ .pinned_prefill_chunk = 100 });
        got.deinit(a);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a 100-row chunk past the budget";

        if (@hasDecl(C, "specTokensAt")) {
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

            for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past, 12, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = if (sp.temperature == 0) "speculative past the budget" else "sampled speculative past the budget";
            }

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

            {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, wide, 12, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 });
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative at the widest chunk with top-p and top-k";
            }

            for ([_]SamplingParams{
                .{ .temperature = 0, .repeat_penalty = 1.3 },
                .{ .temperature = 0, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
                .{ .temperature = 0.8, .top_k = 20, .seed = 3, .repeat_penalty = 1.3, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
            }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past, 12, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative with penalties";
            }

            if (budget > 16) for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                defer turn.end();
                defer arch_sched.endRequest(&cooler.config);
                var spec = try C.specTokensWith(a, cooler.xfm, past[0 .. budget - 8], 16, sp);
                spec.deinit(a);
                if (first_late == null and mlx.engine.latePipelines() != late) first_late = "speculative across the budget";
            };

            const MtpHeadRef = @import("generate.zig").MtpHeadRef;
            const drafter = MtpHeadRef.fromModel(cooler.xfm, true, null).?;
            for ([_]SamplingParams{ .{ .temperature = 0 }, .{ .temperature = 0.7, .top_p = 0.9, .seed = 7 } }) |sp| {
                const prompts = [_][]const u32{ past[0..40], past[0..101], past[0..77] };
                for ([_]usize{ 2, 3 }) |rows| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    try @import("cooler_warm.zig").joinedRounds(io, a, cooler.xfm, drafter, prompts[0..rows], sp, 12);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "joined speculative rows";
                }
            }

            if (cooler.config.image_token_id != 0 and cooler.config.image_token_id < cooler.config.vocab_size) {
                for ([_]?MtpHeadRef{ drafter, null }) |d| {
                    const turn = arch_handle.turnFor(cooler.h, &cooler.config);
                    defer turn.end();
                    defer arch_sched.endRequest(&cooler.config);
                    try @import("cooler_warm.zig").imageRequest(io, a, cooler.xfm, d, .{ .temperature = 0 }, 300, 64, 24);
                    if (first_late == null and mlx.engine.latePipelines() != late) first_late = "an image request";
                }
            }

            var sv: Served = undefined;
            try sv.init(&cooler, dir);
            defer sv.deinit();
            try sv.concurrent(&.{ wide[0..40], wide[0..101] }, 24, .{ .temperature = 0 }, true);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "two concurrent requests";
            const restored = @max(budget, 1300) + 300;
            try sv.concurrent(&.{wide[0..restored]}, 8, .{ .temperature = 0 }, true);
            try sv.concurrent(&.{wide[0 .. restored + 400]}, 24, .{ .temperature = 0 }, true);
            try testing.expect(sv.reused > 0);
            if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a request restored past the budget";
        }

        if (cooler.config.image_token_id != 0) {
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
    }

    // 3. Served request that restores live state behind its reply.
    {
        var long: [1300]u32 = undefined;
        for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
        var sv: Served = undefined;
        try sv.init(&cooler, dir);
        defer sv.deinit();
        var from: u32 = 0;
        var first = try sv.request(&long, 24, &from);
        first.deinit(a);
        var again = try sv.request(&long, 24, &from);
        again.deinit(a);
        try testing.expect(sv.reused > 0);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "a request restored behind its reply";
    }

    // 4. Release bench's first request through the scheduler (fresh Served so no prefix cache hit).
    {
        const prompt = try a.alloc(u32, 4096);
        defer a.free(prompt);
        for (prompt, 0..) |*t, i| t.* = ids[i % ids.len];
        var sv: Served = undefined;
        try sv.init(&cooler, dir);
        defer sv.deinit();
        var from: u32 = 0;
        var got = try sv.requestWith(prompt, 16, .{ .temperature = 0, .top_p = 0.95, .top_k = 20 }, false, &from);
        got.deinit(a);
        if (first_late == null and mlx.engine.latePipelines() != late) first_late = "the release bench's first request";
    }

    errdefer std.debug.print("cooler contract: from {?s} on, requests compiled:\n{s}", .{ first_late, mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "cooler contract: after a Cooler load's served warm, lengths, widest chunk, restore and bench compile no kernel, on the Qwen tinies" {
    try contract.servedOn("qwen4", &.{ "QWEN4_STQ_TINY", "QWEN4_BF16_TINY" }, checkLateFixture);
}

test "cooler contract: after a Cooler load's served warm, lengths, widest chunk, restore and bench compile no kernel, on the Qwen cut" {
    try contract.servedOn("qwen4", &.{"QWEN4_TEST_MODEL"}, checkLateFixture);
}

test "cooler contract: after a Cooler load's served warm, lengths, restore and bench compile no kernel, on DeepSeek" {
    try contract.servedOn("dsv41", &.{ "DSV41_STQ_TINY", "DSV41_TINY" }, checkLateFixture);
}

test "cooler contract: after a Cooler load's served warm, lengths, restore and bench compile no kernel, on Hy4" {
    try contract.servedOn("hy4flash", &.{ "HY4_BF16_TINY", "HY4_STQ_TINY" }, checkLateFixture);
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
    if (@import("bills.zig").ledgerArch(&ld.config)) {
        var aliased: usize = 0;
        for (engine.borrow_from..engine.layers.len) |li| aliased += @intFromBool(engine.layers[li].aliased);
        // Static-bill loads alias the decoder pools on every wide chunk (from chunk 1) and plan tiny fixtures to
        // 2 * chunk + 128 (4224) tokens; two chunks plus the held-back span (4129 + 4 reply tokens) stays within reach.
        const prompt = try testing.allocator.alloc(u32, 2 * 2048 + @import("prefill_plan.zig").SSM_SNAPSHOT_BACKOFF + 1);
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
        return .{ .released = aliased, .late = mlx.engine.latePipelines() - late };
    }
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
    try contract.servedOn("dsv41", &.{"DSV41_STQ_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const shared = @import("arch/cooler_pipelines.zig");
            const cooler_warm = @import("cooler_warm.zig");
            const dsv41_plan = @import("dsv41/dsv41_plan.zig");
            const was_force_alias = dsv41_plan.force_alias_for_test;
            dsv41_plan.force_alias_for_test = true;
            defer dsv41_plan.force_alias_for_test = was_force_alias;
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
