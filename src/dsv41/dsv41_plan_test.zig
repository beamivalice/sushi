//! Tests for DeepSeek-V4.1's static bill plans (D97 Everywhere): two-fill determinism (`recordChecked`),
//! arena replay parity vs pool across decode, verify, DSpark draft, joined ticks, wide region measurement,
//! canary checking, and tail alias sharing.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_contract = @import("dsv41_contract.zig");
const dsv41_plan = @import("dsv41_plan.zig");
const exl3 = @import("sushi_exl3");

const Packed = struct {
    loaded: contract.Load,
    mdl: *dsv41_model.Model,

    fn open(env_name: [:0]const u8, dspark: bool) !Packed {
        const raw = std.c.getenv(env_name) orelse return error.SkipZigTest;
        const dir = std.mem.span(raw);
        var loaded: contract.Load = undefined;
        try loaded.prepare(dsv41_contract, dir, true);
        loaded.config.stream_mtp_head = dspark;
        loaded.config.test_skip_warm = true;
        try loaded.load(dir);
        return .{
            .loaded = loaded,
            .mdl = dsv41_state.modelOf(loaded.xfm) orelse return error.Dsv41NoModel,
        };
    }

    fn deinit(self: *Packed) void {
        _ = self.loaded.close();
    }
};

fn evalSlice(arr: mlx.mlx_array) ![]const f32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const ptr = mlx.mlx_array_data_float32(arr) orelse return error.InvalidTestOutput;
    return ptr[0..mlx.mlx_array_size(arr)];
}

test "dsv41 plan: recordChecked passes two-fill check across decode, narrow verify, joined, DSpark draft, and wide chunks (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    var pack = try Packed.open("DSV41_STQ_TINY", true);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 128);

    // 1. Decode L = 1 & narrow L = 4
    var p1 = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .forward, .rows = 1, .n = @min(reach - 1, m.cfg.window), .want = .last, .route_spec = m.route_spec }, reach);
    defer p1.deinit();
    var p4 = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .forward, .rows = 4, .n = 0, .want = .last, .route_spec = m.route_spec }, reach);
    defer p4.deinit();

    // 2. Verify & finish_verify + draft if DSpark loaded
    if (m.dspark != null) {
        var pv = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .forward, .rows = 3, .n = 1, .want = .all, .verify = true, .lazy_head = m.lazy_head, .route_spec = m.route_spec }, reach);
        defer pv.deinit();
        var pfv = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .finish_verify, .rows = 3, .n = 1, .keep = 2, .verify = true }, reach);
        defer pfv.deinit();
        var pd = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .draft, .rows = @min(m.cfg.dspark.block, 3), .n = 1, .draft_heads = .conf, .draft_short = true }, reach);
        defer pd.deinit();
    }

    // 3. Joined forward (2 rows)
    const groups = [_]u32{ 1, 1 };
    const ns = [_]usize{ 0, 4 };
    var pj = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .joined, .rows = 2, .groups = &groups, .ns = &ns, .lazy_head = true }, reach);
    defer pj.deinit();

    // 4. Wide chunk
    var pw = try dsv41_plan.recordChecked(m, testing.allocator, .{ .phase = .wide, .rows = 24, .n = 0, .want = .last, .route_spec = false }, reach);
    defer pw.deinit();
}

fn checkReplayParity(m: *dsv41_model.Model, reach: usize, chunk_max: usize) !void {
    const own = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, own);
    const lent = dsv41_model.lendOwn(m);
    defer lent.end();

    // Reference outputs on pool before installing plans
    const prompt = [_]u32{ 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69 };
    const prefill_ref = blk: {
        try dsv41_model.reserveNew(m, reach);
        const a_pre = m.arm(.prefill);
        defer a_pre.end();
        const lg = try dsv41_model.extendVision(m, &prompt, .last, dsv41_model.SPAN, null);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(prefill_ref);
    const step_ref = blk: {
        const a_step = m.arm(.step);
        defer a_step.end();
        const lg = try dsv41_model.extend(m, &.{7}, .last);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(step_ref);
    var draft_ref = if (m.dspark != null) try dsv41_model.draftMeasured(m, 7, .conf, null, null) else null;
    defer if (draft_ref) |*d| d.deinit();
    dsv41_model.resetState(m);

    // Record + makeArena
    const plans = try dsv41_plan.record(m, testing.allocator, reach, chunk_max, 0);
    m.plans = plans;
    try dsv41_plan.makeArena(m, plans);
    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;

    try dsv41_model.reserveNew(m, reach);
    {
        const wide_lg = try dsv41_plan.wide(m, &prompt, .last, null);
        defer _ = mlx.mlx_array_free(wide_lg);
        try testing.expectEqualSlices(f32, prefill_ref, try evalSlice(wide_lg));
    }

    {
        const step_lg = try dsv41_plan.run(m, &.{7}, .last, null);
        defer _ = mlx.mlx_array_free(step_lg);
        try testing.expectEqualSlices(f32, step_ref, try evalSlice(step_lg));
    }

    if (draft_ref) |*dr| {
        var got_d = try dsv41_plan.runDraft(m, 7, .conf, null, null);
        defer got_d.deinit();
        try testing.expectEqualSlices(u32, dr.ids, got_d.ids);
        try testing.expectEqualSlices(f32, dr.conf, got_d.conf);
        const v_ids = got_d.ids[0..@min(got_d.ids.len, 3)];
        const v_lg = try dsv41_plan.runVerify(m, v_ids);
        _ = mlx.mlx_array_free(v_lg);
        try dsv41_plan.runFinishVerify(m, 0);
    }
    try testing.expectEqual(hits, core.arena_check.hits);
    dsv41_model.resetState(m);
}

test "dsv41 plan: record, arena replay bit-exact parity vs pool, DSpark round, and wide chunk with clean canaries (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    var pack = try Packed.open("DSV41_STQ_TINY", true);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 96);
    try checkReplayParity(m, reach, @min(32, reach));
}

test "dsv41 plan: record, arena replay bit-exact parity vs pool on FP8/MXFP4 pack with clean canaries (DSV41_TINY)" {
    try mlx.requireGpu();
    var pack = try Packed.open("DSV41_TINY", true);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 96);
    try checkReplayParity(m, reach, @min(32, reach));
}

test "dsv41 plan: joined replay bit-exact parity vs pool under closed pool (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    var pack = try Packed.open("DSV41_STQ_TINY", false);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 64);
    const chunk_max: usize = @min(24, reach);

    const own = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, own);
    const serves_was = m.serves_rows;
    m.serves_rows = true;
    defer m.serves_rows = serves_was;

    var req0 = try dsv41_model.Request.init(testing.allocator, 0);
    defer req0.deinit();
    var req1 = try dsv41_model.Request.init(testing.allocator, 0);
    defer req1.deinit();
    try m.makeRequestLogits(&req0);
    try m.makeRequestLogits(&req1);

    const prompt0 = [_]u32{ 1, 5, 9, 13, 17, 21 };
    const prompt1 = [_]u32{ 2, 6, 10, 14 };
    const step0 = [_]u32{25};
    const step1 = [_]u32{18};
    const segs = [_]dsv41_model.Seg{
        .{ .request = &req0, .ids = &step0 },
        .{ .request = &req1, .ids = &step1 },
    };

    const ref_rows = blk: {
        {
            const l0 = dsv41_model.lend(m, &req0);
            defer l0.end();
            try dsv41_model.reserveNew(m, reach);
            const ap = m.arm(.prefill);
            defer ap.end();
            _ = mlx.mlx_array_free(try dsv41_model.extendVision(m, &prompt0, .last, dsv41_model.SPAN, null));
        }
        {
            const l1 = dsv41_model.lend(m, &req1);
            defer l1.end();
            try dsv41_model.reserveNew(m, reach);
            const ap = m.arm(.prefill);
            defer ap.end();
            _ = mlx.mlx_array_free(try dsv41_model.extendVision(m, &prompt1, .last, dsv41_model.SPAN, null));
        }
        const ar = m.arm(.rows);
        defer ar.end();
        const jlg = try dsv41_model.extendRows(m, &segs);
        defer _ = mlx.mlx_array_free(jlg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(jlg));
    };
    defer testing.allocator.free(ref_rows);

    {
        const l0 = dsv41_model.lend(m, &req0);
        defer l0.end();
        dsv41_model.resetState(m);
    }
    {
        const l1 = dsv41_model.lend(m, &req1);
        defer l1.end();
        dsv41_model.resetState(m);
    }

    const plans = try dsv41_plan.record(m, testing.allocator, reach, chunk_max, 2);
    m.plans = plans;
    try dsv41_plan.makeArena(m, plans);
    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};

    {
        const l0 = dsv41_model.lend(m, &req0);
        defer l0.end();
        try dsv41_model.reserveNew(m, reach);
        _ = mlx.mlx_array_free(try dsv41_plan.wide(m, &prompt0, .last, null));
    }
    {
        const l1 = dsv41_model.lend(m, &req1);
        defer l1.end();
        try dsv41_model.reserveNew(m, reach);
        _ = mlx.mlx_array_free(try dsv41_plan.wide(m, &prompt1, .last, null));
    }

    var views: [2]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (views) |v| if (v.ctx != null) {
        _ = mlx.mlx_array_free(v);
    };
    try dsv41_plan.runJoined(m, &segs, false, &views);
    const vocab = m.cfg.vocab;
    try testing.expectEqualSlices(f32, ref_rows[0..vocab], try evalSlice(views[0]));
    try testing.expectEqualSlices(f32, ref_rows[vocab .. 2 * vocab], try evalSlice(views[1]));
}

test "dsv41 plan: borrow-as-alias shares arena tail across wide prefill and narrow decode under closed pool (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    var pack = try Packed.open("DSV41_STQ_TINY", false);
    defer pack.deinit();
    const m = pack.mdl;
    try testing.expect(m.cfg.layers > 4);
    const stq = dsv41_state.stqExpertsOf(pack.loaded.xfm) orelse return error.MissingExpertStream;
    const engine = stq.engine;
    engine.borrow_from = 4;

    const own = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, own);
    const lent = dsv41_model.lendOwn(m);
    defer lent.end();

    const reach: usize = @min(@as(usize, m.cfg.max_pos), 96);
    var prompt: [24]u32 = undefined;
    for (&prompt, 0..) |*tok, i| tok.* = @intCast(1 + i * 4);

    // Reference outputs on pool before installing plans / alias slabs.
    const prefill_ref = blk: {
        try dsv41_model.reserveNew(m, reach);
        const a_pre = m.arm(.prefill);
        defer a_pre.end();
        const lg = try dsv41_model.extendVision(m, &prompt, .last, dsv41_model.SPAN, null);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(prefill_ref);
    const step_ref = blk: {
        const a_step = m.arm(.step);
        defer a_step.end();
        const lg = try dsv41_model.extendVision(m, &.{7}, .last, dsv41_model.SPAN, null);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(step_ref);
    dsv41_model.resetState(m);

    const plans = try dsv41_plan.record(m, testing.allocator, reach, 24, 0);
    m.plans = plans;
    try dsv41_plan.makeArenaWithBorrow(m, plans, engine);
    try testing.expect(core.arenaTail() != null);
    try testing.expect(!core.arenaLeaseHeld());
    try testing.expect(engine.layers[4].aliased);
    try testing.expect(!engine.layers[3].aliased);

    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};

    for (0..2) |_| {
        try dsv41_model.reserveNew(m, reach);
        {
            const wide_lg = try dsv41_plan.wide(m, &prompt, .last, null);
            defer _ = mlx.mlx_array_free(wide_lg);
            try testing.expect(core.arenaLeaseHeld());
            try testing.expect(engine.layers[4].borrowed);
            try testing.expectEqualSlices(f32, prefill_ref, try evalSlice(wide_lg));
        }
        {
            const step_lg = try dsv41_plan.run(m, &.{7}, .last, null);
            defer _ = mlx.mlx_array_free(step_lg);
            try testing.expect(!core.arenaLeaseHeld());
            try testing.expect(!engine.layers[4].borrowed);
            try testing.expect(engine.layers[4].aliased);
            try testing.expectEqualSlices(f32, step_ref, try evalSlice(step_lg));
        }
        dsv41_model.resetState(m);
    }
}

test "dsv41 plan: held logits view with rotated logits_turn stays in-place without PlanEscape (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    var pack = try Packed.open("DSV41_STQ_TINY", false);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 64);

    const own = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, own);
    const lent = dsv41_model.lendOwn(m);
    defer lent.end();

    const prompt = [_]u32{ 1, 5, 9, 13, 17, 21 };
    const ref0 = blk: {
        try dsv41_model.reserveNew(m, reach);
        const ap = m.arm(.prefill);
        defer ap.end();
        _ = mlx.mlx_array_free(try dsv41_model.extendVision(m, &prompt, .last, dsv41_model.SPAN, null));
        const as0 = m.arm(.step);
        defer as0.end();
        const lg = try dsv41_model.extend(m, &.{7}, .last);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(ref0);
    const ref1 = blk: {
        const as1 = m.arm(.step);
        defer as1.end();
        const lg = try dsv41_model.extend(m, &.{8}, .last);
        defer _ = mlx.mlx_array_free(lg);
        break :blk try testing.allocator.dupe(f32, try evalSlice(lg));
    };
    defer testing.allocator.free(ref1);
    dsv41_model.resetState(m);

    const plans = try dsv41_plan.record(m, testing.allocator, reach, @min(16, reach), 0);
    m.plans = plans;
    try dsv41_plan.makeArena(m, plans);
    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};

    try dsv41_model.reserveNew(m, reach);
    defer dsv41_model.resetState(m);
    _ = mlx.mlx_array_free(try dsv41_plan.wide(m, &prompt, .last, null));

    const lg0 = try dsv41_plan.run(m, &.{7}, .last, null);
    defer _ = mlx.mlx_array_free(lg0);
    const blocks: *[2]?mlx.mlx_array = if (m.bound) |r| &r.logits_fixed else &m.logits_fixed;
    const turn_ptr: *u1 = if (m.bound) |r| &r.logits_turn else &m.logits_turn;
    // `lg0` holds the previously written slot (`turn_ptr.* ^ 1`); the other slot is still sole.
    const held_slot = turn_ptr.* ^ 1;
    try testing.expect(!try mlx.engine.sole(blocks[held_slot].?));
    try testing.expect(try mlx.engine.sole(blocks[turn_ptr.*].?));
    // Reproduce `queuedHead` + `settleLast` rollback: rotate `logits_turn` back onto the slot `lg0` still holds.
    turn_ptr.* = held_slot;

    const lg1 = try dsv41_plan.run(m, &.{8}, .last, null);
    defer _ = mlx.mlx_array_free(lg1);
    try testing.expect(!try mlx.engine.sole(blocks[held_slot].?));
    try testing.expect(!try mlx.engine.sole(blocks[held_slot ^ 1].?));
    try testing.expectEqualSlices(f32, ref0, try evalSlice(lg0));
    try testing.expectEqualSlices(f32, ref1, try evalSlice(lg1));
}
