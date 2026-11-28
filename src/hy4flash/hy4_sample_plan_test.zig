//! Hy4-Flash's samples on the arena (docs/bill_redesign.md, D24) on the tiny fixtures: each mix a load plans runs as
//! one execution with the pool's tokens, and a served request's draws ask the pool nothing.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_held = @import("hy4_held.zig");
const sampler = @import("../sampler.zig");
const hy4_model = @import("hy4_model.zig");
const sample_plan = @import("hy4_sample_plan.zig");
const fwd = @import("hy4_forward.zig");

/// A served load of `dir` whose draws read a reserved-token mask, made as a load with a tokenizer makes it
/// (`sampler.installSuppressMask`), before its warm plans.
fn openMasked(ld: *contract.Load, dir: []const u8) !void {
    try ld.prepare(hy4_held.Drafting, dir, true);
    ld.config.test_skip_warm = true;
    try ld.load(dir);
    errdefer ld.discard();
    const vocab: usize = ld.xfm.config.vocab_size;
    ld.xfm.suppress_mask = try sampler.buildSuppressMask(&.{ 5, 9 }, 0, vocab, ld.xfm.outStream());
    try ld.serve();
}

/// Logits of every row of the load's first block, distinct enough that a draw and an argmax depend on them.
fn fillLogits(a: std.mem.Allocator, m: *hy4_model.Model) !void {
    const rows: usize = @intCast(fwd.verify_rows_max);
    const vocab: usize = m.config.vocab;
    const host = try a.alloc(f32, rows * vocab);
    defer a.free(host);
    for (host, 0..) |*x, i| x.* = @floatCast(@sin(@as(f64, @floatFromInt(i)) * 0.37) * 6.0 + @as(f64, @floatFromInt(i % 7)) * 0.25);
    const scope = mlx.engine.enter(m.s);
    defer scope.end();
    const arr = try mlx.newData(host.ptr, &[_]c_int{ @intCast(rows), @intCast(vocab) }, .float32);
    defer _ = mlx.mlx_array_free(arr);
    try hy4_model.writeFixed(&m.logits_fixed[0], arr, &.{ @intCast(rows), @intCast(vocab) }, m.s);
}

fn tokenOf(a: mlx.mlx_array) !u32 {
    try mlx.check(mlx.mlx_array_eval(a));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, a));
    return @intCast(v);
}

/// What a draw gave, read out and let go: a step's token is a view of its block, which the draw after it writes.
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

fn checkAllMixes(a: std.mem.Allocator, ld: *contract.Load) !void {
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const st = &m.plans.?.sample;
    try testing.expect(st.planned);
    try testing.expectEqual(@as(u8, if (m.plans.?.drafter != null) @intCast(fwd.verify_rows_max) else 0), st.rows_max);
    try fillLogits(a, m);
    const all = try sample_plan.mixes(a, st);
    defer a.free(all);
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;
    var verdicts: usize = 0;
    for (all) |mix| for ([_]bool{ false, true }) |other| {
        verdicts += @intFromBool(mix.entry == .verdict);
        var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix, .other = other });
        defer in.deinit();
        // The pool's draw and the planned one write the same block: the pool's token is read before the planned write.
        const at = st.turn;
        st.planned = false;
        const pool = sample_plan.draw(m, st, &in, mix);
        st.planned = true;
        st.turn = at;
        const want = try heard(try pool);
        const before = core.stats();
        try testing.expect(try sample_plan.begin(m, mix));
        const got = sample_plan.draw(m, st, &in, mix) catch |e| {
            sample_plan.fail(m);
            return e;
        };
        defer got.deinit();
        try sample_plan.end(m);
        const after = core.stats();
        errdefer std.debug.print("hy4 sample plan: mix {any} (other {})\n", .{ mix, other });
        try testing.expectEqual(@as(u64, 1), after.executions - before.executions);
        try testing.expectEqual(before.closed_asks, after.closed_asks);
        try testing.expectEqual(before.asks, after.asks);
        try expectSame(want, got);
    };
    try testing.expectEqual(m.plans.?.drafter != null, verdicts > 0);
    try testing.expectEqual(hits, core.arena_check.hits);
}

test "hy4 sample plan: a draw of every mix the load plans runs as one execution of its root with the pool's token and verdict, asks the pool nothing, canaries clean (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    if (std.c.getenv("HY4_BF16_TINY")) |env_dir| {
        ran = true;
        hy4_held.releaseHeld();
        var ld: contract.Load = undefined;
        try openMasked(&ld, std.mem.span(env_dir));
        defer _ = ld.close();
        try checkAllMixes(a, &ld);
    }
    if (std.c.getenv("HY4_STQ_TINY") != null) {
        ran = true;
        const h = try hy4_held.acquire(.stq_drafting_suppress);
        try checkAllMixes(a, h.ld);
    }
    if (!ran) return error.SkipZigTest;
}

const Generator = @import("../generate.zig").Generator;
const CallForce = @import("../generate.zig").CallForce;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

/// A served request's tokens, its draws as executions and the pool's asks past the
/// seal over its decode (the census).
const Served = struct {
    tokens: [16]u32 = undefined,
    n: usize = 0,
    draws: u64 = 0,
    closed: u64 = 0,
};

/// A request of `prompt` for 16 tokens under `sampling` and `opts`, its rounds DFly2's where `opts` drafts, on the
/// load's plans or, `planned` false, on the pool.
fn serve(a: std.mem.Allocator, ld: *contract.Load, prompt: []const u32, sampling: sampler.SamplingParams, opts: Generator.InitOptions, eos: []const u32, planned: bool) !Served {
    const m = ld.xfm.arch_state.hy4flash.model.?;
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, ld.xfm, &tok, prompt, 16, sampling, eos, opts);
    defer gen.deinit(a);
    var out: Served = .{};
    const draws = if (saved) |p| p.sample.draws else 0;
    const before = core.stats().closed_asks;
    while (out.n < out.tokens.len) {
        if (opts.drafter_enabled) {
            const r = (try gen.nextDrafterRound(a)) orelse break;
            defer a.free(r.tokens);
            for (r.tokens) |t| if (out.n < out.tokens.len) {
                out.tokens[out.n] = t;
                out.n += 1;
            };
        } else {
            out.tokens[out.n] = (try gen.next(a)) orelse break;
            out.n += 1;
        }
    }
    out.closed = core.stats().closed_asks - before;
    if (planned) out.draws = saved.?.sample.draws - draws;
    return out;
}

test "hy4 sample plan: a served request's draws, greedy and sampled, under penalties, plain and in DFly2 rounds, give the pool's tokens, each draw an execution (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_drafting_suppress);
    const ids = try @import("hy4_contract.zig").prompt(a, h.dir);
    defer a.free(ids);
    const prompt = ids[0..24];
    const ld = h.ld;
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try testing.expect(m.plans.?.drafter != null);
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const Case = struct { sampling: sampler.SamplingParams, opts: Generator.InitOptions = .{} };
    const cases = [_]Case{
        .{ .sampling = .{ .temperature = 0 } },
        .{ .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.9, .min_p = 0.05, .seed = 11 } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 11, .repeat_penalty = 1.2, .presence_penalty = 0.4 } },
        .{ .sampling = .{ .temperature = 0 }, .opts = .{ .drafter_enabled = true } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 7 }, .opts = .{ .drafter_enabled = true } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 7, .repeat_penalty = 1.2 }, .opts = .{ .drafter_enabled = true } },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("hy4 sample plan: request {d} differs\n", .{i});
        const want = try serve(a, ld, prompt, c.sampling, c.opts, &.{}, false);
        const got = try serve(a, ld, prompt, c.sampling, c.opts, &.{}, true);
        try testing.expectEqual(want.n, got.n);
        try testing.expectEqualSlices(u32, want.tokens[0..want.n], got.tokens[0..got.n]);
        // A plain request draws a token a step; a DFly2 request, a verdict a round.
        try testing.expect(got.draws >= (if (c.opts.drafter_enabled) 2 else got.n));
        // A served request's decode asks the pool nothing, the serial draw's (penalties) too: its next forward
        // takes its id from the load's input block (`Model.idsInput`).
        try testing.expectEqual(@as(u64, 0), got.closed);
    }
}

test "hy4 sample plan: a request whose stop set holds a token the model emits ends at it with the pool's tokens before it and none sampled after, plain and in DFly2 rounds (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_drafting_suppress);
    const ids = try @import("hy4_contract.zig").prompt(a, h.dir);
    defer a.free(ids);
    const ld = h.ld;
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const prompt = ids[0..24];
    for ([_]Generator.InitOptions{ .{}, .{ .drafter_enabled = true } }) |opts| {
        const free = try serve(a, ld, prompt, .{ .temperature = 0 }, opts, &.{}, true);
        var stop_at: usize = 0;
        for (1..free.n) |k| {
            if (std.mem.indexOfScalar(u32, free.tokens[0..k], free.tokens[k]) == null) {
                stop_at = k;
                break;
            }
        }
        try testing.expect(stop_at > 0);
        for ([_]bool{ false, true }) |planned| {
            const stopped = try serve(a, ld, prompt, .{ .temperature = 0 }, opts, &.{free.tokens[stop_at]}, planned);
            try testing.expectEqualSlices(u32, free.tokens[0..stop_at], stopped.tokens[0..stopped.n]);
        }
    }
}

test "hy4 sample plan: a request whose forced call ends samples on after the forced token with the pool's tokens, no draw missing (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_drafting_suppress);
    const ids = try @import("hy4_contract.zig").prompt(a, h.dir);
    defer a.free(ids);
    const ld = h.ld;
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const call = [_]u32{ 12, 13 };
    var served: [2]Served = undefined;
    for ([_]bool{ false, true }, &served) |planned, *out| {
        var cf: CallForce = .{ .forced = &call, .call_at = 0, .closer_id = null, .phase = .answer, .last = ids[23] };
        const sampling: sampler.SamplingParams = .{ .temperature = 0, .call_force = &cf };
        out.* = try serve(a, ld, ids[0..24], sampling, .{}, &.{}, planned);
    }
    try testing.expectEqual(served[0].n, served[1].n);
    try testing.expectEqualSlices(u32, served[0].tokens[0..served[0].n], served[1].tokens[0..served[1].n]);
    try testing.expectEqualSlices(u32, &call, served[1].tokens[0..call.len]);
}

test "hy4 sample plan: a forced call's token made outside any draw past the seal is the load's token block, written in place, asking the closed pool nothing (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.stq_drafting_suppress);
    const ld = h.ld;
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try core.closePool(.refuse);
    defer core.closePool(.open) catch {};
    const before = core.stats().closed_asks;
    for ([_]u32{ 7, 11, 7 }) |id| {
        const t = (try sample_plan.forced(m, id)) orelse return error.TestUnplanned;
        defer _ = mlx.mlx_array_free(t);
        try testing.expectEqual(id, (try mlx.hostResult(u32, t, 1))[0]);
    }
    try testing.expectEqual(before, core.stats().closed_asks);
}

test "hy4 sample plan: a planned draw that misses while its token is written is refused by name, the token blocks it leaves are sound, and the next draw is served (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_drafting_suppress);
    const ld = h.ld;
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const st = &m.plans.?.sample;
    try fillLogits(a, m);
    defer core.closePool(.open) catch {};
    const mix: sampler.Mix = .{ .entry = .step, .suppress = true, .greedy = true };
    // Either block may be the one the failed draw was writing.
    for ([_]u1{ 0, 1 }) |at| {
        st.turn = at;
        var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix });
        defer in.deinit();
        // A view of every block: the draw takes none that is sole.
        var views: [sample_plan.token_slots]mlx.mlx_array = @splat(.{ .ctx = null });
        defer {
            for (views) |v| _ = mlx.mlx_array_free(v);
        }
        for (&views, st.tokens) |*v, block| {
            v.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(v, block.?, &[_]c_int{1}, 1, m.s));
        }
        try core.closePool(.refuse);
        try testing.expect(try sample_plan.begin(m, mix));
        try testing.expect(std.meta.isError(sample_plan.draw(m, st, &in, mix)));
        sample_plan.fail(m);
        try testing.expectEqual(at, st.turn);
        for (&views) |*v| {
            _ = mlx.mlx_array_free(v.*);
            v.* = .{ .ctx = null };
        }
        // What the missed draw left is no other execution's: the next draw writes the block it was writing, and the
        // pool's draw agrees with it.
        var next = try sample_plan.makeInputs(m, st, a, .{ .mix = mix });
        defer next.deinit();
        try core.closePool(.open);
        st.planned = false;
        const want = try heard(try sample_plan.draw(m, st, &next, mix));
        st.planned = true;
        st.turn = at;
        try core.closePool(.refuse);
        try testing.expect(try sample_plan.begin(m, mix));
        const got = try sample_plan.draw(m, st, &next, mix);
        defer got.deinit();
        try sample_plan.end(m);
        try expectSame(want, got);
    }
}

test "hy4 sample plan: on a GPU that grants the draw fewer threads than its widest rung, the load plans the rung it serves at and a sampled request misses no draw (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    hy4_held.releaseHeld();
    const draw = @import("../sampler_draw.zig");
    // A fresh process's ladder, under the grant an M1 gives this kernel.
    draw.ladderForTest(512);
    defer draw.ladderForTest(std.math.maxInt(usize));
    const ids = try @import("hy4_contract.zig").prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try openMasked(&ld, dir);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const got = try serve(a, &ld, ids[0..24], .{ .temperature = 0.8, .seed = 11 }, .{}, &.{}, true);
    try testing.expect(got.n > 0);
}
