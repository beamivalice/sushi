//! Hy4-Flash's samples on the arena (docs/bill_redesign.md, D24) on the tiny fixtures: each mix a load plans runs as
//! one execution with the pool's tokens, and a served request's draws ask the pool nothing.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const sampler = @import("../sampler.zig");
const think_penalty = @import("../think_penalty.zig");
const hy4_model = @import("hy4_model.zig");
const sample_plan = @import("hy4_sample_plan.zig");
const fwd = @import("hy4_forward.zig");

/// Experts resident, DFly2 loaded where the fixture ships it, as a drafting request runs.
const Drafting = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        config.expert_streaming = false;
        config.expert_source_dir = try a.dupe(u8, dir);
        config.stream_mtp_head = @import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir);
    }
};

/// A served load of `dir` whose draws read a reserved-token mask and think markers, made as a load with a tokenizer
/// makes them (`sampler.installSuppressMask`, `generate.installThinkMarkers`), before its warm plans.
fn openMasked(ld: *contract.Load, dir: []const u8) !void {
    try ld.prepare(Drafting, dir, true);
    try ld.load(dir);
    errdefer ld.discard();
    const vocab: usize = ld.xfm.config.vocab_size;
    ld.xfm.suppress_mask = try sampler.buildSuppressMask(&.{ 5, 9 }, 0, vocab, ld.xfm.outStream());
    ld.xfm.think_marker_mask = try think_penalty.buildMask(ld.a, &.{ 3, 4, 11 }, vocab);
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

test "hy4 sample plan: a draw of every mix the load plans runs as one execution of its root with the pool's token and verdict, asks the pool nothing, canaries clean (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try openMasked(&ld, dir);
        defer _ = ld.close();
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
                try sample_plan.fail(m);
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
        try testing.expectEqual(@as(u64, 0), st.misses);
        try testing.expectEqual(hits, core.arena_check.hits);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 sample plan: a forced call's token in flight reaches the planned draw of every think kind that reads ids, which gives the pool's token (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try openMasked(&ld, dir);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const st = &m.plans.?.sample;
    try fillLogits(a, m);
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const kinds = [_]sampler.Mix.Think{ .mask_inside, .mask_opens, .mask_unopened, .bias_inside, .bias_opens, .bias_unopened };
    for (kinds) |think| for ([_]u32{ 1, 2, 7 }) |forced| {
        const mix: sampler.Mix = .{ .entry = .step, .suppress = true, .greedy = true, .think = think };
        errdefer std.debug.print("hy4 sample plan: think {t}, forced token {d}\n", .{ think, forced });
        var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix });
        defer in.deinit();
        _ = mlx.mlx_array_free(in.pending.?);
        in.pending = try @import("../generate.zig").forcedToken(ld.xfm, forced);
        // The pool's draw and the planned one are one step, each writing the block after the one in flight; the pool's
        // token is read and let go before the planned draw writes its block.
        const turn_of_step = st.turn;
        st.planned = false;
        const pool = sample_plan.draw(m, st, &in, mix);
        st.planned = true;
        st.turn = turn_of_step;
        const want = blk: {
            const drawn = try pool;
            defer drawn.deinit();
            break :blk try tokenOf(drawn.step);
        };
        try testing.expect(try sample_plan.begin(m, mix));
        const got = sample_plan.draw(m, st, &in, mix) catch |e| {
            try sample_plan.fail(m);
            return e;
        };
        defer got.deinit();
        try sample_plan.end(m);
        try testing.expectEqual(want, try tokenOf(got.step));
    };
    try testing.expectEqual(@as(u64, 0), st.misses);
}

test "hy4 sample plan: a planned draw that misses while its token is written is refused by name, the token blocks it leaves are sound, and the next draw is served (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try openMasked(&ld, dir);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const st = &m.plans.?.sample;
    try fillLogits(a, m);
    defer core.closePool(.open) catch {};
    const mix: sampler.Mix = .{ .entry = .step, .suppress = true, .greedy = true, .think = .mask_inside };
    // Either block may be the one the missed draw was writing.
    for ([_]u1{ 0, 1 }) |at| {
        st.turn = at;
        var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix });
        defer in.deinit();
        // A view of the block the draw writes: its write is not in place, which the plan did not record.
        _ = mlx.mlx_array_free(in.pending.?);
        in.pending = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&in.pending.?, st.tokens[at].?, &[_]c_int{1}, 1, m.s));
        const misses = st.misses;
        try core.closePool(.refuse);
        try testing.expect(try sample_plan.begin(m, mix));
        try testing.expect(std.meta.isError(sample_plan.draw(m, st, &in, mix)));
        try testing.expectError(error.PlanMiss, sample_plan.fail(m));
        try testing.expectEqual(misses + 1, st.misses);
        try testing.expectEqual(at, st.turn);
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

/// A request of `prompt` for 16 tokens under `sampling` and `opts`, its rounds DSpark's where `opts` drafts, on the
/// load's plans or, `planned` false, on the pool.
fn serve(a: std.mem.Allocator, ld: *contract.Load, prompt: []const u32, sampling: sampler.SamplingParams, opts: Generator.InitOptions, planned: bool) !Served {
    const m = ld.xfm.arch_state.hy4flash.model.?;
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, ld.xfm, &tok, prompt, 16, sampling, &.{}, opts);
    defer gen.deinit(a);
    var out: Served = .{};
    const draws = if (saved) |p| p.sample.draws else 0;
    const before = core.stats().closed_asks;
    while (out.n < out.tokens.len) {
        if (opts.dspark_enabled) {
            const r = (try gen.nextDspark(a)) orelse break;
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

test "hy4 sample plan: a served request's draws, greedy and sampled, under penalties and the think penalty, plain and in DSpark rounds, give the pool's tokens, each draw an execution (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try @import("hy4_contract.zig").prompt(a, dir);
    defer a.free(ids);
    const prompt = ids[0..24];
    var ld: contract.Load = undefined;
    try openMasked(&ld, dir);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try testing.expect(m.plans.?.drafter != null);
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    const think: think_penalty.ThinkPenalty = .{ .lambda = 2, .opener_id = 3, .closer_id = 4, .phase = .inside };
    const Case = struct { sampling: sampler.SamplingParams, opts: Generator.InitOptions = .{} };
    const cases = [_]Case{
        .{ .sampling = .{ .temperature = 0 } },
        .{ .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.9, .min_p = 0.05, .seed = 11 } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 11, .repeat_penalty = 1.2, .presence_penalty = 0.4 } },
        .{ .sampling = .{ .temperature = 0, .think_penalty = think } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 5, .think_penalty = think } },
        .{ .sampling = .{ .temperature = 0 }, .opts = .{ .dspark_enabled = true } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 7 }, .opts = .{ .dspark_enabled = true } },
        .{ .sampling = .{ .temperature = 0.8, .seed = 7, .repeat_penalty = 1.2, .think_penalty = think }, .opts = .{ .dspark_enabled = true } },
    };
    for (cases, 0..) |c, i| {
        errdefer std.debug.print("hy4 sample plan: request {d} differs\n", .{i});
        const want = try serve(a, &ld, prompt, c.sampling, c.opts, false);
        const got = try serve(a, &ld, prompt, c.sampling, c.opts, true);
        try testing.expectEqual(want.n, got.n);
        try testing.expectEqualSlices(u32, want.tokens[0..want.n], got.tokens[0..got.n]);
        // A plain request draws a token a step; a DSpark request, a verdict a round.
        try testing.expect(got.draws >= (if (c.opts.dspark_enabled) 2 else got.n));
        // A served request's decode asks the pool nothing, the serial draw's (penalties) too: its next forward
        // takes its id from the load's input block (`Model.idsInput`).
        try testing.expectEqual(@as(u64, 0), got.closed);
    }
    try testing.expectEqual(@as(u64, 0), m.plans.?.sample.misses);
}

test "hy4 sample plan: on a GPU that grants the draw fewer threads than its widest rung, the load plans the rung it serves at and a sampled request misses no draw (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
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
    const got = try serve(a, &ld, ids[0..24], .{ .temperature = 0.8, .seed = 11 }, .{}, true);
    try testing.expect(got.n > 0);
    try testing.expectEqual(@as(u64, 0), m.plans.?.sample.misses);
}

test "hy4 sample plan: a request whose forced call ends in the think span samples on after the forced token with the pool's tokens, no draw missing (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
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
    const think: think_penalty.ThinkPenalty = .{ .lambda = 2, .opener_id = 3, .closer_id = 4, .phase = .inside };
    const call = [_]u32{ 12, 13 };
    var served: [2]Served = undefined;
    for ([_]bool{ false, true }, &served) |planned, *out| {
        var cf: CallForce = .{ .forced = &call, .call_at = 0, .closer_id = null, .phase = .answer, .last = ids[23] };
        const sampling: sampler.SamplingParams = .{ .temperature = 0, .think_penalty = think, .call_force = &cf };
        out.* = try serve(a, &ld, ids[0..24], sampling, .{}, planned);
    }
    try testing.expectEqual(served[0].n, served[1].n);
    try testing.expectEqualSlices(u32, served[0].tokens[0..served[0].n], served[1].tokens[0..served[1].n]);
    try testing.expectEqualSlices(u32, &call, served[1].tokens[0..call.len]);
    try testing.expectEqual(@as(u64, 0), m.plans.?.sample.misses);
}

test "hy4 sample plan: a draw of a mix the load did not plan is refused by name, learned in place and kept, and served the next time with the pool's token (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try ld.open(Drafting, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const p = m.plans.?;
    const st = &p.sample;
    try fillLogits(a, m);
    const mix: sampler.Mix = .{ .entry = .step, .repeat = true, .additive = true };
    // The load planned no root for it: its tables and traces go.
    const r = for (p.book.roots.items) |r| {
        if (r.id.eql(sample_plan.root(mix))) break r;
    } else return error.TestNoRoot;
    for (r.traces.items) |*t| t.deinit();
    r.traces.clearRetainingCapacity();
    r.tables.?.deinit();
    r.tables = null;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    p.cache = .init(testing.io, tmp.dir, "scope-build");
    defer p.cache = null;
    var in = try sample_plan.makeInputs(m, st, a, .{ .mix = mix });
    defer in.deinit();
    try testing.expectError(error.PlanMiss, sample_plan.begin(m, mix));
    try testing.expectEqual(@as(u64, 1), st.misses);
    try testing.expectEqual(@as(u64, 1), st.learned);
    const kept = try p.cache.?.read(a);
    defer a.free(kept);
    try testing.expectEqual(@as(usize, 1), kept.len);
    try testing.expectEqual(sample_plan.shapeOf(mix), kept[0]);
    try testing.expectEqual(mix, sample_plan.mixOf(kept[0]).?);
    st.planned = false;
    const want = try heard(try sample_plan.draw(m, st, &in, mix));
    st.planned = true;
    const before = core.stats().executions;
    try testing.expect(try sample_plan.begin(m, mix));
    const got = sample_plan.draw(m, st, &in, mix) catch |e| {
        try sample_plan.fail(m);
        return e;
    };
    defer got.deinit();
    try sample_plan.end(m);
    try testing.expectEqual(before + 1, core.stats().executions);
    try expectSame(want, got);
    try testing.expectEqual(@as(u64, 1), st.misses);
}
