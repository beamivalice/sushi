//! Tests for DeepSeek-V4.1's static-arena sample plan (`dsv41_sample_plan`): bit-exact parity vs pool across all
//! planned mixes (suppress, penalties, grammar, verdicts) and forced tokens under a closed pool.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const sampler = @import("../sampler.zig");
const cfg_mod = @import("dsv41_config.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_store = @import("dsv41_store.zig");
const dsv41_contract = @import("dsv41_contract.zig");
const dsv41_plan = @import("dsv41_plan.zig");
const sample_plan = @import("dsv41_sample_plan.zig");
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

fn fillLogits(a: std.mem.Allocator, m: *dsv41_model.Model) !void {
    const rows: usize = @intCast(cfg_mod.ring_slack);
    const vocab: usize = m.cfg.vocab;
    const host = try a.alloc(f32, rows * vocab);
    defer a.free(host);
    for (host, 0..) |*x, i| x.* = @floatCast(@sin(@as(f64, @floatFromInt(i)) * 0.37) * 6.0 + @as(f64, @floatFromInt(i % 7)) * 0.25);
    const s = dsv41_model.stateStream(m);
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const arr = try mlx.newData(host.ptr, &[_]c_int{ @intCast(rows), @intCast(vocab) }, .float32);
    defer _ = mlx.mlx_array_free(arr);
    try dsv41_store.writeFixed(&m.logits_fixed[0], arr, &.{ @intCast(rows), @intCast(vocab) }, s);
}

fn tokenOf(a: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(a));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, a));
    return @intCast(v);
}

const Heard = union(enum) {
    step: u32,
    sync: u32,
    verdict: @FieldType(sample_plan.Drawn, "verdict"),
};

fn heard(d: sample_plan.Drawn) !Heard {
    defer d.deinit();
    return switch (d) {
        .step => |t| .{ .step = try tokenOf(t) },
        .sync => |r| .{ .sync = r.token_id },
        .verdict => |v| .{ .verdict = v },
    };
}

fn expectSame(want: Heard, got: sample_plan.Drawn) !void {
    try testing.expectEqualStrings(@tagName(want), @tagName(got));
    switch (want) {
        .step => |w| try testing.expectEqual(w, try tokenOf(got.step)),
        .sync => |w| try testing.expectEqual(w, got.sync.token_id),
        .verdict => |w| try testing.expectEqual(w, got.verdict),
    }
}

test "dsv41 sample plan: every planned mix runs as one execution with pool parity, zero asks, and clean canaries (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    const a = testing.allocator;
    var pack = try Packed.open("DSV41_STQ_TINY", true);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 64);

    const s = dsv41_model.stateStream(m);
    const suppress = try sampler.buildSuppressMask(&.{ 5, 9 }, 0, m.cfg.vocab, s);
    defer _ = mlx.mlx_array_free(suppress);

    const p = try dsv41_plan.record(m, a, reach, @min(24, reach), 0);
    m.plans = p;
    try sample_plan.record(m, p, .{ .suppress = suppress });
    try dsv41_plan.makeArena(m, p);

    const st = &p.sample;
    try testing.expect(st.planned);
    try testing.expect(st.rows_max > 0);
    try fillLogits(a, m);

    const all = try sample_plan.mixes(a, st);
    defer a.free(all);

    const pool_was = core.poolGate();
    try core.closePool(.census);
    defer core.closePool(pool_was) catch {};
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;

    var verdicts: usize = 0;
    for (all) |mix| for ([_]bool{ false, true }) |other| {
        verdicts += @intFromBool(mix.entry == .verdict);
        var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix, .other = other });
        defer in.deinit();
        const at = st.turn;
        st.planned = false;
        const pool_draw = sample_plan.draw(m, st, &in, mix);
        st.planned = true;
        st.turn = at;
        const want = try heard(try pool_draw);

        const before = core.stats();
        try testing.expect(try sample_plan.begin(m, mix));
        const got = sample_plan.draw(m, st, &in, mix) catch |e| {
            sample_plan.fail(m);
            return e;
        };
        defer got.deinit();
        try sample_plan.end(m);
        const after = core.stats();
        try testing.expectEqual(@as(u64, 1), after.executions - before.executions);
        try testing.expectEqual(before.closed_asks, after.closed_asks);
        try testing.expectEqual(before.asks, after.asks);
        try expectSame(want, got);
    };
    try testing.expect(verdicts > 0);
    try testing.expectEqual(hits, core.arena_check.hits);
}

test "dsv41 sample plan: forced tokens write in place under closed pool asking nothing (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!exl3.kernels.gemmNaxOn()) return error.SkipZigTest;
    const a = testing.allocator;
    var pack = try Packed.open("DSV41_STQ_TINY", false);
    defer pack.deinit();
    const m = pack.mdl;
    const reach: usize = @min(@as(usize, m.cfg.max_pos), 64);

    const p = try dsv41_plan.record(m, a, reach, @min(24, reach), 0);
    m.plans = p;
    try sample_plan.record(m, p, .{});
    try dsv41_plan.makeArena(m, p);

    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};
    const before = core.stats().closed_asks;
    for ([_]u32{ 7, 11, 7 }) |id| {
        const t = (try sample_plan.forced(m, id)) orelse return error.TestUnplanned;
        defer _ = mlx.mlx_array_free(t);
        try testing.expectEqual(id, (try mlx.hostResult(u32, t, 1))[0]);
    }
    try testing.expectEqual(before, core.stats().closed_asks);
}
