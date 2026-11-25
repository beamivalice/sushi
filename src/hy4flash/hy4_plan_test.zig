//! The static bill's plan on Hy4-Flash's tiny fixtures (docs/bill_redesign.md, stage 1): a forward's trace as the
//! recorder sees it at the engine's choke points.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const contract = @import("../cooler_contract_test.zig");
const arch_forward = @import("../arch/forward.zig");
const hy4_contract = @import("hy4_contract.zig");
const fwd = @import("hy4_forward.zig");

test "hy4 plan: a served request on the pool yields each forward's real trace, every ask the pool counts with its bytes and its death (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(hy4_contract, dir, true);
    defer _ = ld.close();
    // The pool's traces: the load's one-row forwards off the arena for this request.
    const m = ld.xfm.arch_state.hy4flash.model.?;
    const plans = m.plans;
    m.plans = null;
    defer m.plans = plans;
    var recs: arch_forward.Recordings = .{ .gpa = a };
    defer recs.deinit();
    {
        arch_forward.recordings = &recs;
        defer arch_forward.recordings = null;
        // Off its plans past the seal, the load's request runs on the pool by design.
        const wide = core.widePhase();
        defer wide.end();
        var got = try contract.greedy(&ld, ids[0..20], 4);
        got.deinit(a);
    }
    // The prefill and the decodes, each an execution of its own.
    try testing.expect(recs.list.items.len >= 3);
    for (recs.list.items) |one| {
        var n: u64 = 0;
        var bytes: u64 = 0;
        var died: u64 = 0;
        for (one.recorder.instances.items) |inst| for (inst.asks.items) |as| {
            n += 1;
            bytes += as.bytes;
            died += @intFromBool(as.death != plan_trace.none);
        };
        try testing.expect(one.pool_asks > 0);
        try testing.expectEqual(one.pool_asks, n);
        try testing.expectEqual(one.pool_ask_bytes, bytes);
        // Most of a forward's asks are its scratch, dead before it ends; the rest outlive it as state or logits.
        try testing.expect(died * 2 > n);
    }
}

/// A Hy4 load with every expert resident: the decode names no expert stream.
const Resident = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        config.expert_streaming = false;
        config.expert_source_dir = try a.dupe(u8, dir);
    }
};

/// `Resident` with DFly2 loaded where the fixture ships it, as a drafting request runs.
const Drafting = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        try Resident.prepare(a, config, dir);
        config.stream_mtp_head = true;
    }
};

const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");

/// What the engine did for the GPU while a recording ran.
const Work = struct { dispatches: u64, submits: u64, pool_bytes: u64, asks: u64, pipelines: u64 };

/// Position `p` of `ids` decoded after a real prefill of the ones before it, recorded `how` into `rec` when given.
fn decodeAt(m: anytype, ids: []const u32, p: usize, how: ?core.How, rec: ?*plan.Recorder) !Work {
    m.reset();
    const pre = try m.forwardReadout(ids[0..p], .none);
    _ = mlx.mlx_array_free(pre);
    mlx.engine.close();
    const before = core.stats();
    const compiled = core.pipelines;
    if (rec) |r| try core.beginRecording(how.?, r);
    {
        // A dry decode leaves the model's state over stores with no bytes.
        defer if (how != null and how.? == .dry) m.reset();
        defer if (rec != null) core.endRecording() catch {};
        const lg = try m.forwardReadout(ids[p .. p + 1], .last);
        try mlx.check(mlx.mlx_array_eval(lg));
        _ = mlx.mlx_array_free(lg);
    }
    const after = core.stats();
    return .{ .dispatches = after.dispatches - before.dispatches, .submits = after.submits - before.submits, .pool_bytes = after.pool_bytes -| before.pool_bytes, .asks = after.asks - before.asks, .pipelines = core.pipelines - compiled };
}

test "hy4 plan: a dry decode asks what the real decode asks, ask for ask, death for death and level for level, and runs no GPU work (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try m.probeWide();
    // Position 20 opens a group: the decode makes state that outlives it.
    const p = 20;
    _ = try decodeAt(m, ids, p, null, null);
    var real = try plan.Recorder.begin(a, .{ .site = 1 });
    defer real.deinit();
    const real_work = try decodeAt(m, ids, p, .real, &real);
    var dry = try plan.Recorder.begin(a, .{ .site = 1 });
    defer dry.deinit();
    var driver: plan.Driver = .{ .gpa = a };
    defer driver.deinit();
    const dry_work = try decodeAt(m, ids, p, .{ .dry = .{ .driver = &driver } }, &dry);
    try testing.expect(real_work.dispatches > 0 and real_work.asks > 100);
    var asked: u64 = 0;
    for (real.instances.items) |inst| asked += inst.asks.items.len;
    try testing.expectEqual(real_work.asks, asked);
    try testing.expectEqual(@as(?plan_trace.Difference, null), plan_trace.firstDifference(real.instances.items, dry.instances.items));
    try testing.expectEqual(Work{ .dispatches = 0, .submits = 0, .pool_bytes = 0, .asks = 0, .pipelines = 0 }, dry_work);
}

test "hy4 plan: on the host embedding gather an M1 or M2 load takes, a dry decode decodes no block and asks what the real decode asks, ask for ask (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const hy4_stq = @import("hy4_stq.zig");
    // Where the GPU gathers the embedding, the gather a GPU without the NAX decode takes is made from its bank.
    const gpu_was = m.embed_gpu;
    var forced: ?hy4_stq.EmbedGather = null;
    if (gpu_was) |g| {
        forced = try hy4_stq.EmbedGather.of(a, g.bank);
        m.embed_gather = forced;
        m.embed_gpu = null;
    }
    defer if (forced) |*f| {
        m.embed_gather = null;
        m.embed_gpu = gpu_was;
        f.deinit(a);
    };
    try testing.expect(m.embed_gather != null);
    try m.probeWide();
    const p = 20;
    _ = try decodeAt(m, ids, p, null, null);
    var real = try plan.Recorder.begin(a, .{ .site = 1 });
    defer real.deinit();
    _ = try decodeAt(m, ids, p, .real, &real);
    var dry = try plan.Recorder.begin(a, .{ .site = 1 });
    defer dry.deinit();
    var driver: plan.Driver = .{ .gpa = a };
    defer driver.deinit();
    _ = try decodeAt(m, ids, p, .{ .dry = .{ .driver = &driver } }, &dry);
    try testing.expectEqual(@as(?plan_trace.Difference, null), plan_trace.firstDifference(real.instances.items, dry.instances.items));
    // The blocks the recorded forward decodes, counted over it alone: `decodeAt`'s prefill to `p` runs for real, and
    // the blocks it kept are dropped, so the real arm decodes its row's block.
    for ([_]bool{ false, true }) |is_dry| {
        m.reset();
        const pre = try m.forwardReadout(ids[0..p], .none);
        _ = mlx.mlx_array_free(pre);
        mlx.engine.close();
        m.embed_gather.?.dropBlocks();
        var rec = try plan.Recorder.begin(a, .{ .site = 1 });
        defer rec.deinit();
        var d: plan.Driver = .{ .gpa = a };
        defer d.deinit();
        const before = hy4_stq.EmbedGather.blocks_decoded;
        try core.beginRecording(if (is_dry) .{ .dry = .{ .driver = &d } } else .real, &rec);
        {
            defer if (is_dry) m.reset();
            defer core.endRecording() catch {};
            const lg = try m.forwardReadout(ids[p .. p + 1], .last);
            try mlx.check(mlx.mlx_array_eval(lg));
            _ = mlx.mlx_array_free(lg);
        }
        const decoded = hy4_stq.EmbedGather.blocks_decoded - before;
        if (is_dry) try testing.expectEqual(@as(usize, 0), decoded) else try testing.expectEqual(@as(usize, 1), decoded);
    }
}

test "hy4 plan: a GPU that refuses one form's wide mix and grants the rest: the probe before a recording latches each pipeline's own grant, the granted forms launch wide and a dry decode plans what the real one runs (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    const hy4_hc = @import("hy4_hc.zig");
    const kn = mlx.engine.kernel;
    // The GPU grants the expand form's wide pipeline half the threads it asks; every other pipeline what it asks.
    hy4_hc.forgetGrants();
    kn.grant_cap = @import("../hc_kernels.zig").hc_tg / 2;
    kn.grant_cap_name = "sushi_hy4_hc_mix_expand_norm_wide";
    defer {
        kn.grant_cap = std.math.maxInt(usize);
        kn.grant_cap_name = "";
        hy4_hc.forgetGrants();
    }
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try m.probeWide();
    // Dry first, so the plan rests on what the probe latched; the prefill of 20 rows takes the tiled mix.
    const p = 20;
    var dry = try plan.Recorder.begin(a, .{ .site = 1 });
    defer dry.deinit();
    var driver: plan.Driver = .{ .gpa = a };
    defer driver.deinit();
    _ = try decodeAt(m, ids, p, .{ .dry = .{ .driver = &driver } }, &dry);
    var real = try plan.Recorder.begin(a, .{ .site = 1 });
    defer real.deinit();
    const wide_before = hy4_hc.wide_launches;
    _ = try decodeAt(m, ids, p, .real, &real);
    try testing.expectEqual(true, hy4_hc.wide_launches > wide_before);
    try testing.expectEqual(@as(?plan_trace.Difference, null), plan_trace.firstDifference(real.instances.items, dry.instances.items));
}

/// A forward of `ids` recorded dry into `rec`.
fn dryForward(m: anytype, ids: []const u32, rec: *plan.Recorder) !void {
    var driver: plan.Driver = .{ .gpa = testing.allocator };
    defer driver.deinit();
    try core.beginRecording(.{ .dry = .{ .driver = &driver } }, rec);
    defer core.endRecording() catch {};
    const lg = try m.forwardReadout(ids, if (ids.len == 1) .last else .all);
    _ = mlx.mlx_array_free(lg);
}

test "hy4 plan: a dry state of n rows, made without the forwards that reach it, records the decode and the verify block a prefill to n leaves to record (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer {
            m.reset();
            m.keep_hidden = false;
        }
        try m.probeWide();
        // About the tiny's group of 4, window of 8 and threshold of 16; with a drafter's handover kept and without.
        for ([_]bool{ false, true }) |hidden| for ([_]usize{ 1, 3, 8, 9, 17, 20, 30 }) |n| for ([_]usize{ 1, 4 }) |w| {
            var walked = try plan.Recorder.begin(a, .{ .site = 1 });
            defer walked.deinit();
            var made = try plan.Recorder.begin(a, .{ .site = 1 });
            defer made.deinit();
            try (Prefill{ .ids = ids, .p = n, .reach = ids.len, .hidden = hidden }).go(m);
            try dryForward(m, ids[n .. n + w], &walked);
            m.reset();
            {
                var own = try plan.Recorder.begin(a, .{ .site = 1 });
                defer own.deinit();
                var driver: plan.Driver = .{ .gpa = a };
                defer driver.deinit();
                try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &own);
                defer core.endRecording() catch {};
                try m.dryState(n, ids.len);
            }
            try dryForward(m, ids[n .. n + w], &made);
            m.reset();
            try testing.expectEqual(@as(?plan_trace.Difference, null), plan_trace.firstDifference(walked.instances.items, made.instances.items));
        };
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a request's state is reserved at its own reach: a smaller reach after a larger one sizes each store to its bill, and a rollback inside the request keeps the reservation, so the verify after it keeps nothing (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const row_store = @import("../row_store.zig");
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        try m.probeWide();
        const g = m.config.indexer.group;
        const div = @import("hy4_model.zig").rows_grow_div;
        // A request that reached 200 rows, then the next turn, in place, reaching 24: its stores are the bill's for 24.
        try (Prefill{ .ids = ids, .p = 10, .reach = 200, .hidden = false }).go(m);
        try m.reserveFor(24);
        for (m.state[0..m.config.layers], m.layers[0..m.config.layers]) |*st, l| if (l.indexer != null) {
            try testing.expectEqual(row_store.capBound(24, div), st.rows.?.cap);
            if (st.pooled) |*p| try testing.expectEqual(row_store.capBound(24 / g, div), p.cap);
        };
        // A verify block rolled back inside a request reserved past it keeps the reservation: the next grows nothing.
        try (Prefill{ .ids = ids, .p = 10, .reach = 200, .hidden = false }).go(m);
        var caps: [64]usize = @splat(0);
        for (m.state[0..m.config.layers], caps[0..m.config.layers]) |*st, *c| c.* = if (st.rows) |r| r.cap else 0;
        const back: RolledBack = .{ .ids = ids, .p = 10, .reach = 200 };
        try back.go(m);
        for (m.state[0..m.config.layers], caps[0..m.config.layers]) |*st, c| try testing.expectEqual(c, if (st.rows) |r| r.cap else 0);
        var verify: Phase = .{};
        defer verify.deinit();
        try verify.probe(m, .{ .site = 2, .key = 4 }, back, Rows{ .ids = ids, .p = 11, .w = 4 });
    }
    if (!ran) return error.SkipZigTest;
}

/// A request at `p` reserved for `reach`, whose verify block of four rows was rolled back to one kept row.
const RolledBack = struct {
    ids: []const u32,
    p: usize,
    reach: usize,
    fn go(self: @This(), m: anytype) !void {
        try (Prefill{ .ids = self.ids, .p = self.p, .reach = self.reach, .hidden = false }).go(m);
        var mark = try m.mark();
        defer m.dropMark(&mark);
        _ = mlx.mlx_array_free(try m.forwardReadout(self.ids[self.p .. self.p + 4], .all));
        try m.rollbackTo(&mark, self.ids[self.p .. self.p + 1]);
        mlx.engine.close();
    }
};

const plan_shape = @import("../tensor/plan/plan_shape.zig");
const plan_pack = @import("../tensor/plan/plan_pack.zig");

/// The probes of one phase, each recorded dry under zeros and under the ramp: the two must be one trace.
const Phase = struct {
    traces: std.ArrayList(plan_trace.Trace) = .empty,

    fn deinit(p: *Phase) void {
        for (p.traces.items) |*t| t.deinit();
        p.traces.deinit(testing.allocator);
    }

    /// `run` recorded dry after `setup` runs for real, once under each fill; the trace kept.
    fn probe(p: *Phase, m: anytype, root: plan.FrameId, setup: anytype, run: anytype) !void {
        const a = testing.allocator;
        var recs: [2]plan.Recorder = undefined;
        var made: usize = 0;
        defer for (recs[0..made]) |*r| r.deinit();
        for ([_]core.Fill{ .zeros, .ramp }) |fill| {
            try setup.go(m);
            recs[made] = try plan.Recorder.begin(a, root);
            made += 1;
            var driver: plan.Driver = .{ .gpa = a };
            defer driver.deinit();
            try core.beginRecording(.{ .dry = .{ .driver = &driver, .fill = fill } }, &recs[made - 1]);
            defer m.reset();
            defer core.endRecording() catch {};
            try run.go(m);
        }
        try testing.expectEqual(@as(?plan_trace.Difference, null), plan_trace.firstDifference(recs[0].instances.items, recs[1].instances.items));
        // Every survivor went into its owner: nothing the forward asked for outlives it.
        const trace = recs[0].end() catch |e| {
            const at = recs[0].escape.?;
            std.log.err("hy4 plan: an ask of {d} B outlives the forward (frame {d}, ordinal {d})", .{ recs[0].instances.items[at.instance].asks.items[at.ordinal].bytes, at.instance, at.ordinal });
            return e;
        };
        try p.traces.append(a, trace);
    }

    /// `traces` merged into one shape a frame and packed, or what the merge found ambiguous.
    fn plans(traces: []const plan_trace.Trace) !?plan_shape.Ambiguity {
        var why: plan_shape.Ambiguity = undefined;
        var shapes = plan_shape.merge(testing.allocator, traces, .{ .refused = &why }) catch |e| switch (e) {
            error.PlanAmbiguous => return why,
            else => return e,
        };
        defer shapes.deinit();
        var tables = try plan_pack.pack(testing.allocator, &shapes, .{});
        tables.deinit();
        return null;
    }

    /// Each probe plans on its own, and all of them plan as one.
    fn quiet(p: *const Phase) !void {
        for (p.traces.items) |*t| try testing.expectEqual(@as(?plan_shape.Ambiguity, null), try plans(t[0..1]));
        try testing.expectEqual(@as(?plan_shape.Ambiguity, null), try plans(p.traces.items));
    }
};

/// The state a probe starts from, as a served request leaves it: the prompt to `p` prefilled with the drafter's
/// handover kept, the state reserved for the request's `reach` and, before a verify block, the round's mark.
const Prefill = struct {
    ids: []const u32,
    p: usize,
    reach: usize,
    hidden: bool,
    mark: ?*?@import("hy4_model.zig").Model.Mark = null,
    fn go(self: @This(), m: anytype) !void {
        if (self.mark) |mk| if (mk.*) |*held| {
            m.dropMark(held);
            mk.* = null;
        };
        m.reset();
        m.keep_hidden = self.hidden;
        try m.reserveFor(self.reach);
        const pre = try m.forwardReadout(self.ids[0..self.p], .none);
        _ = mlx.mlx_array_free(pre);
        mlx.engine.close();
        if (self.mark) |mk| mk.* = try m.mark();
    }
};

const Rows = struct {
    ids: []const u32,
    p: usize,
    w: usize,
    fn go(self: @This(), m: anytype) !void {
        const lg = try m.forwardReadout(self.ids[self.p .. self.p + self.w], if (self.w == 1) .last else .all);
        try mlx.check(mlx.mlx_array_eval(lg));
        _ = mlx.mlx_array_free(lg);
    }
};

/// A prefill chunk of `w` rows at `p`, its last row read out.
const Chunk = struct {
    ids: []const u32,
    p: usize,
    w: usize,
    fn go(self: @This(), m: anytype) !void {
        const lg = try m.forwardReadout(self.ids[self.p .. self.p + self.w], .last);
        try mlx.check(mlx.mlx_array_eval(lg));
        _ = mlx.mlx_array_free(lg);
    }
};

const Draft = struct {
    drafter: *const @import("hy4_dfly.zig").Drafter,
    t1: u32,
    fn go(self: @This(), m: anytype) !void {
        var d = try self.drafter.draft(m, self.t1);
        d.deinit();
    }
};

/// The gate's probes on the fixture `env` names, its DFly2 loaded when `drafts`: null when it is absent, else whether the
/// load drafts (its draft phase ran).
fn quietOn(env: [:0]const u8, drafts: bool) !?bool {
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv(env) orelse return null);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    if (drafts) try ld.openLight(Drafting, dir, true) else try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const st = &ld.xfm.arch_state.hy4flash;
    const m = st.model.?;
    defer {
        m.reset();
        m.keep_hidden = false;
    }
    try m.probeWide();
    // About the tiny's group of 4 (a decode at 3, 7, 15 or 23 closes one), its window of 8 and its selection
    // threshold of 16, and its prompt's end.
    const positions = [_]usize{ 1, 3, 4, 7, 8, 9, 15, 16, 17, 20, 23, 30, 36 };
    const hidden = st.drafter != null;
    const reach = ids.len;
    var decode: Phase = .{};
    defer decode.deinit();
    for (positions) |p| try decode.probe(m, .{ .site = 1, .key = 1 }, Prefill{ .ids = ids, .p = p, .reach = reach, .hidden = hidden }, Rows{ .ids = ids, .p = p, .w = 1 });
    try decode.quiet();
    // Every block width, from starts before the window fills, across its edge, past it and past the threshold.
    var w: usize = 2;
    while (w <= @as(usize, @intCast(fwd.verify_rows_max))) : (w += 1) {
        var verify: Phase = .{};
        defer verify.deinit();
        var mark: ?@import("hy4_model.zig").Model.Mark = null;
        defer if (mark) |*mk| m.dropMark(mk);
        for ([_]usize{ 1, 3, 6, 8, 16, 21 }) |p| try verify.probe(m, .{ .site = 2, .key = w }, Prefill{ .ids = ids, .p = p, .reach = reach, .hidden = hidden, .mark = &mark }, Rows{ .ids = ids, .p = p, .w = w });
        try verify.quiet();
    }
    if (st.drafter) |*d| {
        var draft: Phase = .{};
        defer draft.deinit();
        for ([_]usize{ 4, 17, 30 }) |p| try draft.probe(m, .{ .site = 3 }, Prefill{ .ids = ids, .p = p, .reach = reach, .hidden = true }, Draft{ .drafter = d, .t1 = ids[p] });
        try draft.quiet();
    }
    // A prefill chunk keeps nothing past it either; its probes are planned apart (span and group frames are stage 4's).
    // A real chunk of the width first: a process-wide cache a width fills refuses its fill while a forward records dry.
    const width = 12;
    try (Prefill{ .ids = ids, .p = width, .reach = reach, .hidden = hidden }).go(m);
    var prefill: Phase = .{};
    defer prefill.deinit();
    for ([_]usize{ 1, 4, 9, 20, 28 }) |p| try prefill.probe(m, .{ .site = 4, .key = width }, Prefill{ .ids = ids, .p = p, .reach = reach, .hidden = hidden }, Chunk{ .ids = ids, .p = p, .w = width });
    return st.drafter != null;
}

test "hy4 plan: dry decode, verify at every block width, draft and a prefill chunk, from a served request's state, keep nothing past the forward, read back nothing unclassified, give one trace under both fills, and plan each narrow phase as one shape (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const bf16 = try quietOn("HY4_BF16_TINY", false);
    // The STQ tiny ships DFly2: loaded, it drafts, so its draft phase runs.
    const stq = try quietOn("HY4_STQ_TINY", true);
    if (stq) |drafted| try testing.expectEqual(true, drafted);
    if (bf16 == null and stq == null) return error.SkipZigTest;
}

test "hy4 plan: a process-wide cache that misses while a forward records dry refuses the fill by name and keeps nothing" {
    try mlx.requireGpu();
    const exl3 = @import("sushi_exl3");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const scope = mlx.engine.enter(mlx.engine.stream());
    defer scope.end();
    const v: c_int = 917_331;
    {
        var rec = try plan.Recorder.begin(testing.allocator, .{ .site = 1 });
        defer rec.deinit();
        var driver: plan.Driver = .{ .gpa = testing.allocator };
        defer driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
        defer core.endRecording() catch {};
        try testing.expectError(error.CacheFillDry, exl3.side_cache.int(1, v));
        try testing.expectError(error.CacheFillDry, exl3.linear.gemmInputs(3, 917, 5));
    }
    // A real forward after it fills the cache for real: the dry miss left no entry behind.
    const got = try exl3.side_cache.int(1, v);
    var x: c_int = 0;
    try mlx.check(mlx.mlx_array_item_int32(&x, got));
    try testing.expectEqual(v, x);
    _ = try exl3.linear.gemmInputs(3, 917, 5);
}

const hy4_plan = @import("hy4_plan.zig");

/// The logits of a one-row forward at `p` after a real prefill of the rows before it, on the arena when `planned`, else
/// on the pool; the pool's asks and the executions it took.
fn oneRow(m: anytype, ids: []const u32, p: usize, planned: bool, out: []f32) !struct { asks: u64, executions: u64, barriers: u64 } {
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    m.reset();
    try m.reserveFor(ids.len);
    if (p > 0) _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..p], .none));
    mlx.engine.close();
    const before = core.stats();
    const lg = try hy4_plan.run(m, ids[p .. p + 1], null, .last);
    defer _ = mlx.mlx_array_free(lg);
    try mlx.check(mlx.mlx_array_eval(lg));
    @memcpy(out, (mlx.mlx_array_data_float32(lg) orelse return error.TestNoLogits)[0..out.len]);
    const after = core.stats();
    return .{ .asks = after.asks - before.asks, .executions = after.executions - before.executions, .barriers = after.barriers - before.barriers };
}

/// At each of `positions`, after a real prefill of the rows before it: a one-row forward on the arena gives the pool's
/// bytes in one execution and asks the pool for nothing, poison and canaries on.
fn arenaMatchesPool(a: std.mem.Allocator, m: anytype, ids: []const u32, positions: []const usize) !void {
    try testing.expect(m.plans != null);
    const vocab: usize = @intCast(m.config.vocab);
    const got = try a.alloc(f32, 2 * vocab);
    defer a.free(got);
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;
    for (positions) |p| {
        const pool = try oneRow(m, ids, p, false, got[0..vocab]);
        const arena = try oneRow(m, ids, p, true, got[vocab..]);
        try testing.expectEqual(@as(u64, 0), pool.executions);
        try testing.expectEqual(@as(u64, 1), arena.executions);
        try testing.expectEqual(@as(u64, 0), arena.asks);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[0..vocab]), std.mem.sliceAsBytes(got[vocab..]));
    }
    try testing.expectEqual(hits, core.arena_check.hits);
}

test "hy4 plan: a resident load plans its one-row forwards, and each runs on the arena with the bytes of the pool and no ask of it, poison and canaries on (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        // About the tiny's group of 4, window of 8 and threshold of 16, and its prompt's end.
        try arenaMatchesPool(a, m, ids, &.{ 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 20, 23, 30, ids.len - 1 });
    }
    if (!ran) return error.SkipZigTest;
}

/// A forward of `t` rows of `ids` from position `p`, read out by its last row, after a real prefill of the rows before
/// it: in the region the load measured when `planned`, else on the pool.
fn wideRows(m: anytype, ids: []const u32, p: usize, t: usize, planned: bool, out: []f32) !struct { asks: u64, wide: u64, executions: u64 } {
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    m.reset();
    try m.reserveFor(p + t);
    if (p > 0) _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..p], .none));
    mlx.engine.close();
    const before = core.stats();
    const lg = try hy4_plan.run(m, ids[p .. p + t], null, .last);
    defer _ = mlx.mlx_array_free(lg);
    try mlx.check(mlx.mlx_array_eval(lg));
    @memcpy(out, (mlx.mlx_array_data_float32(lg) orelse return error.TestNoLogits)[0..out.len]);
    const after = core.stats();
    return .{ .asks = after.asks - before.asks, .wide = after.wide_executions - before.wide_executions, .executions = after.executions - before.executions };
}

/// `n` token ids of the tiny's vocabulary, none of them its image id.
fn spreadIds(a: std.mem.Allocator, c: *const @import("hy4_config.zig").Config, n: usize) ![]u32 {
    const ids = try a.alloc(u32, n);
    for (ids, 0..) |*id, i| {
        id.* = @intCast((i * 7919 + 3) % c.vocab);
        if (c.image_id != 0 and id.* == c.image_id) id.* = (id.* + 1) % c.vocab;
    }
    return ids;
}

test "hy4 plan: a prefill chunk of every width from past a verify block to the load's chunk runs in the region its load measured, at the state's start and past the threshold, with the bytes of the pool and no ask of it; an ask past the region is refused by name (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        const chunk = p.region_rows;
        try testing.expectEqual(@import("../prefill_plan.zig").widestPrefillWidth(&ld.xfm.config), chunk);
        try testing.expect(p.region.scratch > 0);
        const ids = try spreadIds(a, &m.config, chunk + 20);
        defer a.free(ids);
        const vocab: usize = m.config.vocab;
        const got = try a.alloc(f32, 2 * vocab);
        defer a.free(got);
        core.arena_check.byte = 0xFF;
        const verify_rows: usize = @intCast(@import("hy4_forward.zig").verify_rows_max);
        // Past the tiny's threshold of 16; the final span's width; a chunk of a few spans; the load's chunk.
        for ([_]usize{ 0, 20 }) |at| for ([_]usize{ verify_rows + 1, @import("../prefill_plan.zig").final_span_rows, 100, chunk }) |t| {
            if (t > chunk) continue;
            errdefer std.debug.print("{s}: a forward of {d} rows at {d}\n", .{ env, t, at });
            const pool = try wideRows(m, ids, at, t, false, got[0..vocab]);
            const region = try wideRows(m, ids, at, t, true, got[vocab..]);
            try testing.expectEqual(@as(u64, 0), pool.wide);
            try testing.expectEqual(@as(u64, 1), region.wide);
            try testing.expectEqual(@as(u64, 0), region.executions);
            try testing.expectEqual(@as(u64, 0), region.asks);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[0..vocab]), std.mem.sliceAsBytes(got[vocab..]));
        };
        core.arena_check.byte = null;
        // An arena whose scratch holds less than the region: the chunk's forward is refused by name, its state reset.
        const held = core.arenaLayout().?;
        try core.arenaRelease();
        var small = held;
        small.g = 1 << 20;
        try core.arenaFor(small);
        m.reset();
        try m.reserveFor(chunk);
        try testing.expectError(error.WideFull, hy4_plan.run(m, ids[0..chunk], null, .last));
        var said: [512]u8 = undefined;
        try testing.expect(std.mem.indexOf(u8, mlx.takeError(&said) orelse "", "WideFull: a wide execution's scratch reached") != null);
        try testing.expectEqual(@as(usize, 0), m.pos);
        try testing.expectEqual(@as(u64, 1 << 20), core.refused_wide.?.room);
        try core.arenaRelease();
        try core.arenaFor(held);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a prefill chunk asks the same region whatever blocks of the vocabulary its ids fall in, a prompt of a few as one spread over all (HY4_STQ_TINY, HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_STQ_TINY", "HY4_BF16_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        const chunk = p.region_rows;
        const spread = try spreadIds(a, &m.config, chunk);
        defer a.free(spread);
        // The warm's prompt: a few hundred distinct ids, all in the vocabulary's first blocks.
        const few = try a.alloc(u32, chunk);
        defer a.free(few);
        for (few, 0..) |*id, i| {
            id.* = @intCast(1 + i % 300);
            if (m.config.image_id != 0 and id.* == m.config.image_id) id.* += 1;
        }
        const got = try a.alloc(f32, m.config.vocab);
        defer a.free(got);
        _ = try wideRows(m, spread, 0, chunk, true, got);
        const spread_high = core.stats().wide_last_high;
        _ = try wideRows(m, few, 0, chunk, true, got);
        const few_high = core.stats().wide_last_high;
        errdefer std.debug.print("{s}: a {d}-row chunk tops its region at {d} B with ids spread over the vocabulary, {d} B with ids in its first blocks; the load measured {d} B\n", .{ env, chunk, spread_high, few_high, p.region.scratch });
        try testing.expectEqual(spread_high, few_high);
        try testing.expect(few_high <= p.region.scratch);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a prefill forward a row short of the load's chunk, its last attention span short, asks no more than the region its load measured, at the state's start and at its deepest, and runs in it (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        const rows = p.region_rows - 1;
        try testing.expect(rows % @as(usize, @intCast(@import("hy4_model.zig").gathered_span)) != 0);
        for ([_]usize{ 0, p.reach - rows }) |n| {
            errdefer std.debug.print("{s}: a forward of {d} rows at {d} against the region of {d}-row chunks ({d} B, host {d} B)\n", .{ env, rows, n, p.region_rows, p.region.scratch, p.region.host });
            var passes = try hy4_plan.recordChecked(m, a, .{ .rows = rows, .readout = .last, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n, .phase = .wide }, p.reach);
            defer passes.deinit();
            for (passes.traces) |*t| {
                const need = try plan.region.need(a, t, p.book.alignment);
                errdefer std.debug.print("{s}: asks {d} B, host {d} B\n", .{ env, need.scratch, need.host });
                try testing.expect(need.scratch <= p.region.scratch);
                try testing.expect(need.host <= p.region.host);
            }
        }
        const ids = try spreadIds(a, &m.config, rows);
        defer a.free(ids);
        m.reset();
        try m.reserveFor(rows);
        const lg = try hy4_plan.run(m, ids, null, .last);
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a one-row forward on the arena orders its dispatches with the pool's barriers but a few, its bytes reused off work the GPU could run side by side (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const vocab: usize = @intCast(m.config.vocab);
        const got = try a.alloc(f32, 2 * vocab);
        defer a.free(got);
        for ([_]usize{ 0, 1, 4, 8, 9, 16, 17, 30, ids.len - 1 }) |p| {
            const pool = try oneRow(m, ids, p, false, got[0..vocab]);
            const arena = try oneRow(m, ids, p, true, got[vocab..]);
            errdefer std.debug.print("[hy4-plan] {s} at {d}: {d} barriers on the arena, {d} on the pool\n", .{ env, p, arena.barriers, pool.barriers });
            // A few more at most: the logits' write into the load's block (D19), a dispatch the pool does not make, and
            // the reuses the level rule allows where the replay's levels part from the recording's (D23); with lifetimes
            // alone a forward here takes 22 more. The release's decode takes the pool's.
            try testing.expect(arena.barriers <= pool.barriers + 4);
        }
    }
    if (!ran) return error.SkipZigTest;
}

/// A verify block of `rows` rows at `p` after a real prefill of the rows before it, every row read out and the handover
/// kept for the drafter, on the arena when `planned`, else on the pool: its logits' and its handover's bytes appended to
/// `out`, the pool's asks and the executions it took.
fn oneBlock(a: std.mem.Allocator, m: anytype, ids: []const u32, p: usize, rows: usize, planned: bool, out: *std.ArrayList(u8)) !struct { asks: u64, executions: u64 } {
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.reset();
    try m.reserveFor(ids.len);
    m.keep_hidden = false;
    if (p > 0) _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..p], .none));
    m.keep_hidden = true;
    mlx.engine.close();
    const before = core.stats();
    const lg = try hy4_plan.run(m, ids[p .. p + rows], null, .all);
    defer _ = mlx.mlx_array_free(lg);
    try mlx.check(mlx.mlx_array_eval(lg));
    try out.appendSlice(a, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(lg) orelse return error.TestNoLogits)[0..mlx.mlx_array_size(lg)]));
    const h = m.last_hidden orelse return error.TestNoHandover;
    var hc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hc);
    try mlx.check(mlx.mlx_astype(&hc, h, .float32, m.s));
    try mlx.check(mlx.mlx_array_eval(hc));
    try out.appendSlice(a, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(hc) orelse return error.TestNoHandover)[0..mlx.mlx_array_size(hc)]));
    const after = core.stats();
    return .{ .asks = after.asks - before.asks, .executions = after.executions - before.executions };
}

test "hy4 plan: a load that drafts plans its verify blocks of 2 to 8 rows, and each runs on the arena with the bytes of the pool, its logits and its handover, and no ask of it (HY4_BF16_TINY, HY4_STQ_TINY, where they ship DFly2)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        // A fixture that ships no DFly2 layers has no drafter to verify for.
        if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) continue;
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Drafting, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        try testing.expect(ld.xfm.arch_state.hy4flash.drafter != null);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        core.arena_check.byte = 0xFF;
        defer core.arena_check.byte = null;
        const hits = core.arena_check.hits;
        var pool: std.ArrayList(u8) = .empty;
        defer pool.deinit(a);
        var arena: std.ArrayList(u8) = .empty;
        defer arena.deinit(a);
        // Every width at every position the prompt holds: the probes are a few a width (`blockPositions`), so this is
        // the check that the layers' keys name every structure a block takes.
        for (2..@as(usize, @intCast(@import("hy4_forward.zig").verify_rows_max)) + 1) |rows| for (0..ids.len - rows + 1) |p| {
            pool.clearRetainingCapacity();
            arena.clearRetainingCapacity();
            const on_pool = try oneBlock(a, m, ids, p, rows, false, &pool);
            const on_arena = try oneBlock(a, m, ids, p, rows, true, &arena);
            errdefer std.debug.print("[hy4-plan] {s}: a block of {d} rows at {d}\n", .{ env, rows, p });
            try testing.expectEqual(@as(u64, 0), on_pool.executions);
            try testing.expectEqual(@as(u64, 1), on_arena.executions);
            try testing.expectEqual(@as(u64, 0), on_arena.asks);
            try testing.expectEqualSlices(u8, pool.items, arena.items);
        };
        try testing.expectEqual(hits, core.arena_check.hits);
    }
    if (!ran) return error.SkipZigTest;
}

/// A Hy4 load whose every routed expert the stream holds resident, as a served load takes it where its budget covers
/// the pack: each rate-grouped layer's picks map to their slots on the GPU.
const ResidentInStream = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        config.expert_streaming = true;
        config.expert_resident = true;
        config.expert_cache_bytes = std.math.maxInt(u64) / 2;
        config.expert_source_dir = try a.dupe(u8, dir);
    }
};

test "hy4 plan: a load whose STQ experts the stream holds resident plans its one-row forwards, and each runs on the arena with the bytes of the pool and no ask of it (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    // A stream of BF16 experts (HY4_BF16_TINY_STQ's) routes through a read-back even when resident: not planned (D20).
    for ([_][:0]const u8{"HY4_STQ_TINY"}) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(ResidentInStream, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        // Every MoE layer's picks map to their slots on the GPU: a streamed route would read them back, refused dry.
        try testing.expect(m.experts != null and ld.config.expert_resident);
        try arenaMatchesPool(a, m, ids, &.{ 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 20, 23, 30, ids.len - 1 });
    }
    if (!ran) return error.SkipZigTest;
}

/// A streamed load of the contract's whose cache holds `bytes` of experts.
fn StreamedCache(comptime bytes: u64) type {
    return struct {
        pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
            try hy4_contract.prepare(a, config, dir);
            config.expert_cache_bytes = bytes;
        }
    };
}

test "hy4 plan: a streamed load plans its one-row forwards, and each runs on the arena with the bytes of the pool and no ask of it, its experts filled as the pool's are, at a cache of one group a route and of more (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    // Each case's cache slots a layer: fewer than the route's picks (2) make more than one group of a route.
    inline for (.{
        .{ "HY4_BF16_TINY", 40_000, 1 },
        .{ "HY4_BF16_TINY", 70_000, 2 },
        .{ "HY4_STQ_TINY", 200_000, 1 },
        .{ "HY4_STQ_TINY", 400_000, 3 },
    }) |case| {
        if (std.c.getenv(case[0])) |env_dir| {
            const dir = std.mem.span(env_dir);
            ran = true;
            const ids = try hy4_contract.prompt(a, dir);
            defer a.free(ids);
            var ld: contract.Load = undefined;
            try ld.openLight(StreamedCache(case[1]), dir, true);
            defer _ = ld.close();
            const turn = hy4_contract.openTurn(ld.h, &ld.config);
            defer turn.end();
            defer @import("../arch/sched.zig").endRequest(&ld.config);
            const m = ld.xfm.arch_state.hy4flash.model.?;
            defer m.reset();
            const engine = m.experts.?;
            errdefer std.debug.print("[hy4-plan] {s} at a cache of {d} B: {d} slots a layer\n", .{ case[0], case[1], engine.slotsPerLayer() });
            try testing.expectEqual(@as(u16, case[2]), engine.slotsPerLayer());
            // Every MoE layer's route reads its picks back: the streamed route, not a resident one.
            for (m.config.dense_layers..m.config.layers) |l| try testing.expect(engine.residentRoute(@intCast(l)) == null);
            try arenaMatchesPool(a, m, ids, &.{ 0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 20, 23, 30, ids.len - 1 });
        }
    }
    if (!ran) return error.SkipZigTest;
}

/// `StreamedCache` with `--expert-cache-bias` at its most: a decode-width route favours the experts the cache holds.
fn BiasedCache(comptime bytes: u64) type {
    return struct {
        pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
            try StreamedCache(bytes).prepare(a, config, dir);
            config.expert_cache_bias_override = @import("../model_settings.zig").expert_cache_bias_max;
        }
    };
}

test "hy4 plan: a streamed load with --expert-cache-bias plans its one-row forwards, and each runs on the arena with the bytes of the pool and no ask of it, its route reading the cache's penalty (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(BiasedCache(200_000), dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    try testing.expect(m.experts.?.cache_bias > 0);
    // Past a decode width the prefill routes exact, so both arms' decodes read one cache: one slot a layer, its last fill.
    try arenaMatchesPool(a, m, ids, &.{ 17, 30, ids.len - 1 });
}

test "hy4 plan: on the pack HY4_PLAN_PACK names, a one-row forward at each edge of its structure to 8k runs on the arena with the bytes of the pool and no ask of it" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_PLAN_PACK") orelse return error.SkipZigTest);
    // As a served load takes a pack its budget covers: every expert resident in the stream.
    var ld: contract.Load = undefined;
    try ld.open(ResidentInStream, dir, true);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    // The reading counts only as the resident load's: every MoE layer's picks map to their slots on the GPU.
    for (m.config.dense_layers..m.config.layers) |l| try testing.expect(m.experts.?.residentRoute(@intCast(l)) != null);
    const ids = try a.alloc(u32, 8193);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
    // About the release's group of 4, window of 512 and threshold of 2048, and the fast bench's rungs.
    try arenaMatchesPool(a, m, ids, &.{ 0, 1, 3, 4, 7, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 8191 });
}

test "hy4 plan: a load that drafts plans DFly2's draft, and at every depth the prompt holds it runs on the arena with the bytes of the pool, its drafts and their confidence, and no ask of it (HY4_BF16_TINY, HY4_STQ_TINY, where they ship DFly2)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) continue;
        ran = true;
        const ids = try hy4_contract.prompt(a, dir);
        defer a.free(ids);
        var ld: contract.Load = undefined;
        try ld.openLight(Drafting, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const d = &ld.xfm.arch_state.hy4flash.drafter.?;
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const saved = m.plans;
        defer m.plans = saved;
        core.arena_check.byte = 0xFF;
        defer core.arena_check.byte = null;
        const hits = core.arena_check.hits;
        const t: usize = m.config.dfly.draft + 1;
        for (1..ids.len - t + 1) |n| {
            m.reset();
            try m.reserveFor(ids.len);
            m.keep_hidden = true;
            _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..n], .none));
            m.plans = null;
            var pool = try hy4_plan.runDraft(m, d, ids[n]);
            defer pool.deinit();
            m.plans = saved;
            mlx.engine.close();
            const before = core.stats();
            var arena = try hy4_plan.runDraft(m, d, ids[n]);
            defer arena.deinit();
            const after = core.stats();
            errdefer std.debug.print("[hy4-plan] {s}: a draft at {d}\n", .{ env, n });
            try testing.expectEqual(@as(u64, 1), after.executions - before.executions);
            try testing.expectEqual(@as(u64, 0), after.asks - before.asks);
            try testing.expectEqualSlices(u32, pool.ids, arena.ids);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(pool.conf), std.mem.sliceAsBytes(arena.conf));
        }
        try testing.expectEqual(hits, core.arena_check.hits);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a drafting load whose context is narrower than a verify block or a draft plans the widths it can run and serves its decode (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) return error.SkipZigTest;
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Drafting, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const d = &ld.xfm.arch_state.hy4flash.drafter.?;
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const saved = m.plans;
    defer m.plans = saved;
    const reach: usize = 5;
    const narrow = try hy4_plan.build(m, a, reach, d, m.kv, null, 0);
    defer {
        narrow.deinit(a);
        core.arenaRelease() catch {};
    }
    m.plans = narrow;
    m.reset();
    try m.reserveFor(reach);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..2], .none));
    const before = core.stats();
    const logits = try hy4_plan.run(m, ids[2..3], null, .last);
    _ = mlx.mlx_array_free(logits);
    try testing.expectEqual(@as(u64, 1), core.stats().executions - before.executions);
}

/// `ResidentInStream` with DFly2 loaded, as a drafting request on a pack its budget covers runs.
const ResidentDrafting = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        try ResidentInStream.prepare(a, config, dir);
        config.stream_mtp_head = true;
    }
};

test "hy4 plan: on the pack HY4_PLAN_PACK names, with DFly2, a verify block of 2 to 8 rows at each edge of its structure to 8k runs on the arena with the bytes of the pool, its logits and its handover, and no ask of it" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_PLAN_PACK") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try ld.open(ResidentDrafting, dir, true);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    try testing.expect(ld.xfm.arch_state.hy4flash.drafter != null);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    for (m.config.dense_layers..m.config.layers) |l| try testing.expect(m.experts.?.residentRoute(@intCast(l)) != null);
    const saved = m.plans;
    defer m.plans = saved;
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    const ids = try a.alloc(u32, 8200);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
    var pool: std.ArrayList(u8) = .empty;
    defer pool.deinit(a);
    var arena: std.ArrayList(u8) = .empty;
    defer arena.deinit(a);
    m.reset();
    try m.reserveFor(ids.len);
    var at: usize = 0;
    // About the release's group of 4, window of 512 and threshold of 2048, each block straddling or below them.
    for ([_]usize{ 1, 506, 509, 510, 511, 2041, 2044, 2045, 2046, 2047, 4093, 8180 }) |p| {
        m.keep_hidden = false;
        if (p > at) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..p], .none));
        at = p;
        var mk = try m.mark();
        defer m.dropMark(&mk);
        for ([_]usize{ 2, 3, 5, 6, 8 }) |rows| {
            for ([_]bool{ false, true }, [_]*std.ArrayList(u8){ &pool, &arena }) |planned, out| {
                out.clearRetainingCapacity();
                m.plans = if (planned) saved else null;
                m.keep_hidden = true;
                m.dropHidden();
                mlx.engine.close();
                const before = core.stats();
                const lg = try hy4_plan.run(m, ids[p .. p + rows], null, .all);
                defer _ = mlx.mlx_array_free(lg);
                try mlx.check(mlx.mlx_array_eval(lg));
                try out.appendSlice(a, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(lg) orelse return error.TestNoLogits)[0..mlx.mlx_array_size(lg)]));
                var hc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(hc);
                try mlx.check(mlx.mlx_astype(&hc, m.last_hidden orelse return error.TestNoHandover, .float32, m.s));
                try mlx.check(mlx.mlx_array_eval(hc));
                try out.appendSlice(a, std.mem.sliceAsBytes((mlx.mlx_array_data_float32(hc) orelse return error.TestNoHandover)[0..mlx.mlx_array_size(hc)]));
                const after = core.stats();
                errdefer std.debug.print("[hy4-plan] a block of {d} rows at {d} on the {s}\n", .{ rows, p, if (planned) "arena" else "pool" });
                try testing.expectEqual(@as(u64, @intFromBool(planned)), after.executions - before.executions);
                if (planned) try testing.expectEqual(@as(u64, 0), after.asks - before.asks);
                m.plans = saved;
                try m.rollbackTo(&mk, &.{});
            }
            errdefer std.debug.print("[hy4-plan] a block of {d} rows at {d}: the arena's bytes are not the pool's\n", .{ rows, p });
            try testing.expectEqualSlices(u8, pool.items, arena.items);
        }
    }
}

/// `arr`'s bytes appended to `out`.
fn appendHost(a: std.mem.Allocator, arr: mlx.mlx_array, out: *std.ArrayList(u8)) !void {
    try mlx.check(mlx.mlx_array_eval(arr));
    const n = mlx.mlx_array_size(arr);
    try out.appendSlice(a, switch (mlx.mlx_array_dtype(arr)) {
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(arr) orelse return error.HostRead)[0..n]),
        .bfloat16 => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(arr) orelse return error.HostRead)[0..n]),
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(arr) orelse return error.HostRead)[0..n]),
        else => return error.TestStateDtype,
    });
}

/// The request's state as its next forward reads it, appended to `out`: every layer's blocks and stores, whole, and the
/// drafter's handover.
fn appendState(a: std.mem.Allocator, m: anytype, out: *std.ArrayList(u8)) !void {
    const hy4_model = @import("hy4_model.zig");
    var arrays: [hy4_model.max_state_arrays]mlx.mlx_array = undefined;
    for (m.state[0..m.config.layers]) |*st| for (hy4_model.stateArrays(st, &arrays)) |arr| try appendHost(a, arr, out);
    if (m.last_hidden) |h| try appendHost(a, h, out);
}

/// A DSpark round at `p` as `hy4_dspark` runs it, after a real prefill of the rows before it: the mark, a verify block of
/// `rows` rows, the state rolled back to the mark and its first `keep` rows run again; on the arena when `planned`, else
/// on the pool. The verify's logits and the state the round leaves appended to `out`; the executions, and the asks of a
/// closed pool the mark and the rollback made.
fn oneRound(a: std.mem.Allocator, m: anytype, ids: []const u32, p: usize, rows: usize, keep: usize, planned: bool, out: *std.ArrayList(u8)) !struct { executions: u64, mark_asks: u64, back_asks: u64 } {
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    m.reset();
    try m.reserveFor(ids.len);
    if (p > 0) _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..p], .none));
    mlx.engine.close();
    const s0 = core.stats();
    var mk = try m.mark();
    defer m.dropMark(&mk);
    const s1 = core.stats();
    const lg = try hy4_plan.run(m, ids[p .. p + rows], null, .all);
    defer _ = mlx.mlx_array_free(lg);
    const s2 = core.stats();
    try m.rollbackTo(&mk, ids[p .. p + keep]);
    mlx.engine.close();
    const s3 = core.stats();
    try appendHost(a, lg, out);
    try appendState(a, m, out);
    return .{ .executions = s3.executions - s0.executions, .mark_asks = s1.closed_asks - s0.closed_asks, .back_asks = s3.closed_asks - s2.closed_asks };
}

test "hy4 plan: a DSpark round's mark and replay ask nothing of a closed pool: at depths across the window, the selection threshold and the latent width, every kept count leaves the pool's logits and state, its verify and replay one execution each (HY4_BF16_TINY, HY4_STQ_TINY, where they ship DFly2)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) continue;
        ran = true;
        var ld: contract.Load = undefined;
        try ld.openLight(Drafting, dir, true);
        defer _ = ld.close();
        const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        try testing.expect(ld.xfm.arch_state.hy4flash.drafter != null);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const ids = try a.alloc(u32, 144);
        defer a.free(ids);
        for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
        try core.closePool(.census);
        defer core.closePool(.open) catch {};
        core.arena_check.byte = 0xFF;
        defer core.arena_check.byte = null;
        const hits = core.arena_check.hits;
        var pool: std.ArrayList(u8) = .empty;
        defer pool.deinit(a);
        var arena: std.ArrayList(u8) = .empty;
        defer arena.deinit(a);
        const rows: usize = @intCast(@import("hy4_forward.zig").verify_rows_max);
        // About the tiny's group of 4, window of 8, selection threshold of 16 and latent width of 128.
        var depths: std.ArrayList(usize) = .empty;
        defer depths.deinit(a);
        for (1..25) |p| try depths.append(a, p);
        for (116..ids.len - rows + 1) |p| try depths.append(a, p);
        for (depths.items) |p| for (1..rows) |keep| {
            pool.clearRetainingCapacity();
            arena.clearRetainingCapacity();
            const on_pool = try oneRound(a, m, ids, p, rows, keep, false, &pool);
            const on_arena = try oneRound(a, m, ids, p, rows, keep, true, &arena);
            errdefer std.debug.print("[hy4-plan] {s}: a round at {d} keeping {d} of {d}: {d} executions, the mark asked {d}, the rollback {d}\n", .{ env, p, keep, rows, on_arena.executions, on_arena.mark_asks, on_arena.back_asks });
            try testing.expectEqual(@as(u64, 0), on_pool.executions);
            try testing.expectEqual(@as(u64, 2), on_arena.executions);
            try testing.expectEqual(@as(u64, 0), on_arena.mark_asks);
            try testing.expectEqual(@as(u64, 0), on_arena.back_asks);
            try testing.expectEqualSlices(u8, pool.items, arena.items);
        };
        try testing.expectEqual(hits, core.arena_check.hits);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: on the pack HY4_PLAN_PACK names, with DFly2, a DSpark round at each edge of its structure to 8k marks and replays asking nothing of a closed pool, its replay of each kept count one execution, the state it leaves the pool's" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_PLAN_PACK") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try ld.open(ResidentDrafting, dir, true);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    try testing.expect(ld.xfm.arch_state.hy4flash.drafter != null);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    for (m.config.dense_layers..m.config.layers) |l| try testing.expect(m.experts.?.residentRoute(@intCast(l)) != null);
    const saved = m.plans;
    defer m.plans = saved;
    const ids = try a.alloc(u32, 8200);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
    try core.closePool(.census);
    defer core.closePool(.open) catch {};
    var pool: std.ArrayList(u8) = .empty;
    defer pool.deinit(a);
    var arena: std.ArrayList(u8) = .empty;
    defer arena.deinit(a);
    const rows: usize = @intCast(@import("hy4_forward.zig").verify_rows_max);
    m.reset();
    try m.reserveFor(ids.len);
    var at: usize = 0;
    // About the release's group of 4, window of 512, latent width of 512 and threshold of 2048, each round straddling
    // or below them.
    for ([_]usize{ 1, 504, 506, 509, 511, 2041, 2044, 2046, 2047, 4093, 8180 }) |p| {
        if (p > at) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..p], .none));
        at = p;
        mlx.engine.close();
        const s0 = core.stats();
        var mk = try m.mark();
        defer m.dropMark(&mk);
        try testing.expectEqual(s0.closed_asks, core.stats().closed_asks);
        for ([_]usize{ 1, 2, 4, 7 }) |keep| {
            for ([_]bool{ false, true }, [_]*std.ArrayList(u8){ &pool, &arena }) |planned, out| {
                out.clearRetainingCapacity();
                m.plans = if (planned) saved else null;
                _ = mlx.mlx_array_free(try hy4_plan.run(m, ids[p .. p + rows], null, .all));
                mlx.engine.close();
                const before = core.stats();
                try m.rollbackTo(&mk, ids[p .. p + keep]);
                mlx.engine.close();
                const after = core.stats();
                errdefer std.debug.print("[hy4-plan] a round at {d} keeping {d} on the {s}: {d} executions, {d} asks of the closed pool\n", .{ p, keep, if (planned) "arena" else "pool", after.executions - before.executions, after.closed_asks - before.closed_asks });
                try testing.expectEqual(@as(u64, @intFromBool(planned)), after.executions - before.executions);
                if (planned) try testing.expectEqual(before.closed_asks, after.closed_asks);
                // The state the replay leaves, as the next decode row reads it, and the handover beside it.
                try appendHost(a, m.last_hidden orelse return error.TestNoHandover, out);
                const next = try hy4_plan.run(m, ids[p + keep .. p + keep + 1], null, .last);
                defer _ = mlx.mlx_array_free(next);
                try appendHost(a, next, out);
                m.plans = saved;
                try m.rollbackTo(&mk, &.{});
            }
            errdefer std.debug.print("[hy4-plan] a round at {d} keeping {d}: the arena's state is not the pool's\n", .{ p, keep });
            try testing.expectEqualSlices(u8, pool.items, arena.items);
        }
    }
}

/// A served DSpark request of `n` tokens after `prompt`, every round drafted and verifying `rows` rows, on the arena when
/// `planned`, else on the pool: its tokens and the state it leaves appended to `out`. Each round's executions are
/// checked against what it ran: its draft, its verify and, where it kept fewer rows than it verified, its replay.
fn servedRounds(a: std.mem.Allocator, ld: *contract.Load, prompt: []const u32, n: u32, rows: u8, planned: bool, out: *std.ArrayList(u8)) !void {
    const Generator = @import("../generate.zig").Generator;
    const m = ld.xfm.arch_state.hy4flash.model.?;
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    var tok = @import("../tokenizer.zig").Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
    defer gen.deinit(a);
    gen.dspark_diag = .{ .rows = rows };
    var rounds: u64 = 0;
    while (true) {
        const hist = gen.dspark_stats.rows_hist;
        const plain = gen.dspark_stats.plain;
        const before = core.stats();
        const r = (try gen.nextDspark(a)) orelse break;
        defer a.free(r.tokens);
        const ran = core.stats().executions - before.executions;
        try out.appendSlice(a, std.mem.sliceAsBytes(r.tokens));
        // A plain step (the generator's pending token) is decode's, not a round's.
        if (gen.dspark_stats.plain != plain) continue;
        const verified = for (gen.dspark_stats.rows_hist, hist, 0..) |now, was, w| {
            if (now != was) break w;
        } else return error.TestNoRound;
        rounds += 1;
        // Rows forced past one draft every round, whatever the room left cuts the verify to: the draft, the verify, its
        // verdict's draw (D24), and the replay where the draft was not all kept.
        const want: u64 = 3 + @as(u64, @intFromBool(r.accepted_tokens + 1 < verified));
        errdefer std.debug.print("[hy4-plan] a served round of {d} rows kept {d}: {d} executions\n", .{ verified, r.accepted_tokens + 1, ran });
        try testing.expectEqual(if (planned) want else 0, ran);
    }
    try testing.expect(rounds > 0);
    try appendState(a, m, out);
}

test "hy4 plan: served DSpark rounds give the pool's tokens and state, each round's draft, verify and replay an execution on the arena, from prompts before the window and the threshold and across the latent width (HY4_BF16_TINY, HY4_STQ_TINY, where they ship DFly2)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) continue;
        ran = true;
        var ld: contract.Load = undefined;
        try ld.openLight(Drafting, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const ids = try a.alloc(u32, 112);
        defer a.free(ids);
        for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
        core.arena_check.byte = 0xFF;
        defer core.arena_check.byte = null;
        const hits = core.arena_check.hits;
        var pool: std.ArrayList(u8) = .empty;
        defer pool.deinit(a);
        var arena: std.ArrayList(u8) = .empty;
        defer arena.deinit(a);
        // About the tiny's window of 8, selection threshold of 16 and latent width of 128, at two block widths.
        for ([_]struct { prompt: usize, n: u32, rows: u8 }{ .{ .prompt = 5, .n = 24, .rows = 8 }, .{ .prompt = 6, .n = 20, .rows = 5 }, .{ .prompt = 112, .n = 28, .rows = 8 } }) |req| {
            pool.clearRetainingCapacity();
            arena.clearRetainingCapacity();
            try servedRounds(a, &ld, ids[0..req.prompt], req.n, req.rows, false, &pool);
            try servedRounds(a, &ld, ids[0..req.prompt], req.n, req.rows, true, &arena);
            errdefer std.debug.print("[hy4-plan] {s}: a request of {d} tokens after {d}, rounds of {d} rows, is not the pool's\n", .{ env, req.n, req.prompt, req.rows });
            try testing.expectEqualSlices(u8, pool.items, arena.items);
        }
        try testing.expectEqual(hits, core.arena_check.hits);
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a one-row forward whose shape no probe recorded is refused by name, learned in place, and served the next time with the bytes of the pool (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const p = m.plans.?;
    // The decode's root planned from one position only: a group still open, below the window.
    const id = hy4_plan.root(.{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv });
    var book: @import("../tensor/plan/plan_book.zig").Book = .{ .gpa = a, .options = p.book.options };
    const passes = try hy4_plan.recordChecked(m, a, .{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv, .n = 1 }, p.reach);
    defer a.free(passes.traces);
    try testing.expectEqual(@as(usize, 1), passes.traces.len);
    try book.add(id, passes.traces[0]);
    try book.build();
    std.mem.swap(@TypeOf(book), &book, &p.book);
    // Swapped back first, so the book freed is the one this test made.
    defer book.deinit();
    defer std.mem.swap(@TypeOf(book), &book, &p.book);
    const vocab: usize = m.config.vocab;
    const got = try a.alloc(f32, 2 * vocab);
    defer a.free(got);
    // The missed shape is kept beside the load's manifest.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    p.cache = .init(testing.io, tmp.dir, "scope-build");
    defer p.cache = null;
    // Past the window and a group's edge: layers whose keys the one probe never took.
    try testing.expectError(error.PlanMiss, oneRow(m, ids, 20, true, got[vocab..]));
    try testing.expectEqual(@as(u64, 1), p.misses);
    try testing.expectEqual(@as(u64, 1), p.learned);
    const kept = try p.cache.?.read(a);
    defer a.free(kept);
    try testing.expectEqual(@as(usize, 1), kept.len);
    try testing.expectEqual(hy4_plan.shapeOf(.{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv, .n = 20 }), kept[0]);
    const pool = try oneRow(m, ids, 20, false, got[0..vocab]);
    const arena = try oneRow(m, ids, 20, true, got[vocab..]);
    try testing.expectEqual(@as(u64, 0), pool.executions);
    try testing.expectEqual(@as(u64, 1), arena.executions);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(got[0..vocab]), std.mem.sliceAsBytes(got[vocab..]));
}

test "hy4 plan: a load plans the shapes earlier misses kept beside its manifest, and passes over one it cannot run by name (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const p = m.plans.?;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cache: @import("../tensor/plan/plan_book.zig").Cache = .init(testing.io, tmp.dir, "scope-build");
    // One the load can run, and one past its reach.
    try cache.keep(a, hy4_plan.shapeOf(.{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv, .n = 21 }));
    try cache.keep(a, hy4_plan.shapeOf(.{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv, .n = p.reach }));
    const plain = try hy4_plan.build(m, a, p.reach, null, p.kv, null, 0);
    defer {
        plain.deinit(a);
        core.arenaRelease() catch {};
    }
    const with = try hy4_plan.build(m, a, p.reach, null, p.kv, cache, 0);
    defer {
        with.deinit(a);
        core.arenaRelease() catch {};
    }
    const id = hy4_plan.root(.{ .rows = 1, .readout = .last, .keep_hidden = false, .kv = p.kv });
    const traces = struct {
        fn of(b: *const @import("../tensor/plan/plan_book.zig").Book, want: plan.FrameId) usize {
            for (b.roots.items) |r| if (r.id.eql(want)) return r.traces.items.len;
            return 0;
        }
    }.of;
    try testing.expectEqual(traces(&plain.book, id) + 1, traces(&with.book, id));
}

/// The engine's work over `n` one-row forwards from position `p`, on the arena when `planned`, else on the pool: per
/// forward, from a real prefill of the rows before `p`.
fn perForward(m: anytype, ids: []const u32, p: usize, n: usize, planned: bool) !struct { dispatches: f64, barriers: f64, submits: f64, waits: f64, busy_ms: f64, host_ms: f64 } {
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    m.reset();
    try m.reserveFor(p + n);
    if (p > 0) _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..p], .none));
    mlx.engine.close();
    const clock = @import("../load_clock.zig");
    const before = core.stats();
    const t0 = clock.now();
    for (p..p + n) |at| {
        const lg = try hy4_plan.run(m, ids[at .. at + 1], null, .last);
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
    }
    const wall = clock.now() - t0;
    const after = core.stats();
    const k: f64 = @floatFromInt(n);
    return .{
        .dispatches = @as(f64, @floatFromInt(after.dispatches - before.dispatches)) / k,
        .barriers = @as(f64, @floatFromInt(after.barriers - before.barriers)) / k,
        .submits = @as(f64, @floatFromInt(after.submits - before.submits)) / k,
        .waits = @as(f64, @floatFromInt(after.waits - before.waits)) / k,
        .busy_ms = @as(f64, @floatFromInt(after.gpu_busy_ns - before.gpu_busy_ns)) / k / 1e6,
        .host_ms = @as(f64, @floatFromInt(wall)) / k / 1e6,
    };
}

test "hy4 plan: on the pack HY4_PLAN_PACK names, with HY4_PLAN_REPORT set, the engine's work a one-row forward on the pool and on the arena, arms interleaved" {
    try mlx.requireGpu();
    if (std.c.getenv("HY4_PLAN_REPORT") == null) return error.SkipZigTest;
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_PLAN_PACK") orelse return error.SkipZigTest);
    // As a served load takes a pack its budget covers: every expert resident in the stream.
    var ld: contract.Load = undefined;
    try ld.open(ResidentInStream, dir, true);
    defer _ = ld.close();
    const turn = @import("../arch/handle.zig").turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const steps = 32;
    const ids = try a.alloc(u32, 8192 + steps);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % @as(usize, @intCast(m.config.vocab)));
    for ([_]usize{ 4096 - steps, 8192 - steps }) |p| for ([_]bool{ false, true, true, false }) |planned| {
        const w = try perForward(m, ids, p, steps, planned);
        std.debug.print("[plan-report] {s} at {d}: {d:.1} dispatches, {d:.1} barriers, {d:.2} command buffers, {d:.2} waits, GPU busy {d:.3} ms, host {d:.3} ms a forward\n", .{ if (planned) "arena" else "pool ", p, w.dispatches, w.barriers, w.submits, w.waits, w.busy_ms, w.host_ms });
    };
}

test "hy4 plan: a forward at another KV width than the load planned, or one that splices an image's rows, runs on the pool with no miss; an image request's decode past its image runs on the arena (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const p = m.plans.?;
    const vocab: usize = m.config.vocab;
    const got = try a.alloc(f32, 2 * vocab);
    defer a.free(got);
    // A request that names BF16 KV.
    const kv_was = m.kv;
    m.kv = .{ .scheme = .off, .bits = 16, .group_size = 64 };
    defer m.kv = kv_was;
    const bf16 = try oneRow(m, ids, 9, true, got[0..vocab]);
    try testing.expectEqual(@as(u64, 0), bf16.executions);
    try testing.expectEqual(@as(u64, 0), p.misses);
    m.kv = kv_was;
    // An image of two rows, both spliced already: the decode after it is the text forward.
    const dim: usize = m.config.dim;
    const zeros = try a.alloc(f32, 2 * dim);
    defer a.free(zeros);
    @memset(zeros, 0);
    const rows = mlx.mlx_array_new_data(zeros.ptr, &[_]c_int{ 2, @intCast(dim) }, 2, .float32);
    defer _ = mlx.mlx_array_free(rows);
    m.reset();
    try m.reserveFor(ids.len);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..9], .none));
    const before = core.stats().executions;
    const past = try hy4_plan.run(m, ids[9..10], .{ .rows = rows, .offset = 2 }, .last);
    _ = mlx.mlx_array_free(past);
    try testing.expectEqual(before + 1, core.stats().executions);
    // An image id with rows still behind it splices them: on the pool.
    const image_id = [_]u32{m.config.image_id};
    const spliced = try hy4_plan.run(m, &image_id, .{ .rows = rows, .offset = 0 }, .last);
    _ = mlx.mlx_array_free(spliced);
    try testing.expectEqual(before + 1, core.stats().executions);
    try testing.expectEqual(@as(u64, 0), p.misses);
}

test "hy4 plan: a forward that splices an image's rows moves the request's splice offset past them, so a decode that samples the image token is the text forward and runs on the arena (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const p = m.plans.?;
    const dim: usize = m.config.dim;
    const zeros = try a.alloc(f32, 2 * dim);
    defer a.free(zeros);
    @memset(zeros, 0);
    const rows = mlx.mlx_array_new_data(zeros.ptr, &[_]c_int{ 2, @intCast(dim) }, 2, .float32);
    defer _ = mlx.mlx_array_free(rows);
    var ctx = ld.xfm.defaultCtx();
    ctx.vision_embeddings = rows;
    ctx.vision_splice_offset = 0;
    ld.xfm.bindOwnRows(32);
    // The image ends in the prompt's last forward, as when it sits in the final span.
    var prompt: [10]i32 = undefined;
    for (&prompt, ids[0..10]) |*d, v| d.* = @intCast(v);
    prompt[7] = @intCast(m.config.image_id);
    prompt[8] = @intCast(m.config.image_id);
    const prompt_arr = mlx.mlx_array_new_data(&prompt, &[_]c_int{ 1, 10 }, 2, .int32);
    defer _ = mlx.mlx_array_free(prompt_arr);
    _ = mlx.mlx_array_free(try ld.xfm.forwardWith(&ctx, prompt_arr));
    try testing.expectEqual(@as(usize, 2), ctx.vision_splice_offset);
    // A sampled image token past the image's rows: no splice, the planned text forward.
    const before = core.stats().executions;
    const token = [_]i32{@intCast(m.config.image_id)};
    const token_arr = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);
    _ = mlx.mlx_array_free(try ld.xfm.forwardWith(&ctx, token_arr));
    try testing.expectEqual(before + 1, core.stats().executions);
    try testing.expectEqual(@as(u64, 0), p.misses);
}

test "hy4 plan: a Cooler load of an STQ pack proves its experts' and its trunk's NAX GEMMs before any forward, and a dry recording that reaches one no load proved is refused by name (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const kernels = @import("sushi_exl3").kernels;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const q = try @import("hy4_stq.zig").readQuant(testing.io, testing.allocator, dir);
    // The slots are the process's: emptied first, so only this load can fill them.
    var expert = kernels.takeGemmNaxForTest(q.expert.?, .tile_major);
    defer expert.putBack();
    var trunk = kernels.takeGemmNaxForTest(q.trunk.?, .pack);
    defer trunk.putBack();
    var ld: contract.Load = undefined;
    try ld.prepare(hy4_contract, dir, true);
    try ld.load(dir);
    defer ld.discard();
    try testing.expect(kernels.gemmNaxProven(q.expert.?, .tile_major));
    try testing.expect(kernels.gemmNaxProven(q.trunk.?, .pack));
    var again = kernels.takeGemmNaxForTest(q.expert.?, .tile_major);
    defer again.putBack();
    var rec = try plan.Recorder.begin(testing.allocator, .{ .site = plan.siteOf(@src()) });
    defer rec.deinit();
    var driver: plan.Driver = .{ .gpa = testing.allocator };
    defer driver.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
        defer core.endRecording() catch {};
        try testing.expectError(error.Exl3KernelProofDry, kernels.prepareGemmNax(q.expert.?, .tile_major, mlx.engine.stream()));
    }
    try testing.expect(!kernels.gemmNaxProven(q.expert.?, .tile_major));
}

test "hy4 plan: a prefill forward of any width at any depth the layers turn at asks no more than the region its load measured (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        try expectSweep(a, m, p);
        // A drafting load keeps the handover, whose rows ask the host ring more.
        const keep_was = m.keep_hidden;
        defer m.keep_hidden = keep_was;
        m.keep_hidden = true;
        const kept = try hy4_plan.record(m, a, p.reach, null, p.kv, null, null);
        defer kept.deinit(a);
        kept.region = try hy4_plan.regionAt(m, kept, p.region_rows);
        kept.region_rows = p.region_rows;
        try expectSweep(a, m, kept);
    }
    if (!ran) return error.SkipZigTest;
}

/// Every width a prefill takes at every depth the layers turn at, none asking more than the region of `p`.
fn expectSweep(a: std.mem.Allocator, m: anytype, p: *hy4_plan.Plans) !void {
    const chunk = p.region_rows;
    const widths = [_]usize{ 2, 5, 9, 17, 31, 32, 33, 63, 65, 255, 256, 257, 1000, 2047, 2049, 4095, 4097, 8000, 8160, chunk - 1, chunk };
    for (widths) |t| {
        if (t > chunk) continue;
        // Every depth through the tiny's group (4), window (8) and threshold (16), and a few past them.
        for (0..40) |n| try expectInRegion(a, m, p, t, n);
        for ([_]usize{ 64, 1000, p.reach / 2, p.reach -| t }) |n| try expectInRegion(a, m, p, t, n);
    }
}

/// A wide forward of `t` rows after `n`, read out by its last row and by none, asks no more than the region of `p`.
fn expectInRegion(a: std.mem.Allocator, m: anytype, p: *hy4_plan.Plans, t: usize, n: usize) !void {
    if (n + t > p.reach) return;
    for ([_]@TypeOf(m.*).Readout{ .none, .last }) |readout| {
        var passes = try hy4_plan.recordChecked(m, a, .{ .rows = t, .readout = readout, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n, .phase = .wide }, p.reach);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            const need = try plan.region.need(a, tr, p.book.alignment);
            errdefer std.debug.print("a forward of {d} rows at {d} ({t}) asks scratch {d} B, host {d} B; the region of {d}-row chunks holds {d} B, host {d} B\n", .{ t, n, readout, need.scratch, need.host, p.region_rows, p.region.scratch, p.region.host });
            try testing.expect(need.scratch <= p.region.scratch and need.host <= p.region.host);
        }
    }
}

test "hy4 plan: a round that reads out more rows than the logits blocks hold is refused by name on a planned load, and a block of them is not (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    if (!@import("hy4_load.zig").shipsHead(std.Io.Threaded.global_single_threaded.io(), a, dir)) return error.SkipZigTest;
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: contract.Load = undefined;
    try ld.openLight(Drafting, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    defer m.reset();
    const block: usize = @intCast(fwd.verify_rows_max);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    _ = try oneBlock(a, m, ids, 0, block, true, &out);
    out.clearRetainingCapacity();
    try testing.expectError(error.Hy4ReadoutPastBlock, oneBlock(a, m, ids, 0, block + 1, true, &out));
    // Off its plans the pool serves the wider round.
    _ = try oneBlock(a, m, ids, 0, block + 1, false, &out);
}

test "hy4 plan: the region of a chunk is measured at each width's two ends and at the one depth a span is cut at the threshold with a verify block's rows of it dense (HY4_BF16_TINY, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    var ran = false;
    for ([_][:0]const u8{ "HY4_BF16_TINY", "HY4_STQ_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const p = ld.xfm.arch_state.hy4flash.model.?.plans.?;
        try testing.expect(p.measured.items.len > 0);
        for (p.measured.items) |r| {
            const span = @min(@as(usize, @import("../prefill_plan.zig").final_span_rows), r.rows);
            const widths: u64 = if (r.rows - 1 > span) 3 else 2;
            // Each width: the state's start and its deepest, read out by none and by its last row, and the cut; and each
            // class of the deep zone at its deepest, read out both ways.
            try testing.expect(r.classes > 0);
            try testing.expectEqual(widths * 5 + 2 * r.classes, r.probes);
        }
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: prefill ubench honours kv context and rolls back between iterations (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    var ld: contract.Load = undefined;
    try ld.openLight(Resident, dir, true);
    defer _ = ld.close();
    const turn = hy4_contract.openTurn(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    const kv: usize = 24;
    const rows: usize = 8;
    const res = try @import("../model_load.zig").runPrefillUbench(testing.allocator, ld.xfm, null, .{
        .n = 2,
        .rows = rows,
        .kv = kv,
        .quiet = true,
    });
    try testing.expectEqual(@as(usize, 3), res.iterations);
    try testing.expectEqual(kv, res.last_start_pos);
    try testing.expectEqual(kv + rows, res.last_end_pos);
}

test "hy4 plan: a load-width prefill chunk past the selection threshold and the window makes its class's asks in its class's order, none larger than at the class's deepest, at every depth and edge (HY4_STQ_TINY, HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_STQ_TINY", "HY4_BF16_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        const chunk = p.region_rows;
        const deepest = p.reach - chunk;
        const w: usize = m.config.window;
        var depths: std.ArrayList(usize) = .empty;
        defer depths.deinit(a);
        // Every residue a class may turn on near the zone's start, each span width's edges, and the deepest.
        for (0..2 * w + 8) |i| try depths.append(a, p.deep + i);
        var g: usize = 1 << 12;
        while (g * 4 < p.reach) : (g *= 2) for ([_]usize{ 0, 1, 3 }) |d| {
            const at = (g * 4) -| (chunk + d);
            if (at >= p.deep and at <= deepest) try depths.append(a, at);
        };
        try depths.append(a, deepest);
        for (depths.items) |at| {
            const class = p.pads.get(m.wideClass(@intCast(at), @intCast(chunk), @import("hy4_model.zig").Model.Readout.last)) orelse {
                std.debug.print("{s}: a {d}-row chunk at {d} has no class its load recorded\n", .{ env, chunk, at });
                return error.TestUnexpectedResult;
            };
            const asks = try scratchAsks(m, p, chunk, at);
            defer a.free(asks);
            const top = class.passes[0].sizes;
            errdefer std.debug.print("{s}: a {d}-row chunk at {d} asks {d} scratch blocks, its class's deepest ({d}) {d}\n", .{ env, chunk, at, asks.len, class.at, top.len });
            try testing.expectEqual(top.len, asks.len);
            for (asks, top, 0..) |x, y, i| if (x > y) {
                std.debug.print("{s}: ask {d} of a {d}-row chunk at {d} is {d} B, {d} B at its class's deepest ({d})\n", .{ env, i, chunk, at, x, y, class.at });
                return error.TestUnexpectedResult;
            };
        }
    }
    if (!ran) return error.SkipZigTest;
}

test "hy4 plan: a load-width prefill chunk past the selection threshold tops out where its class's recording does, at any depth, within the region its load measured (HY4_STQ_TINY, HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var ran = false;
    for ([_][:0]const u8{ "HY4_STQ_TINY", "HY4_BF16_TINY" }) |env| {
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        ran = true;
        var ld: contract.Load = undefined;
        try ld.open(Resident, dir, true);
        defer _ = ld.close();
        const turn = hy4_contract.openTurn(ld.h, &ld.config);
        defer turn.end();
        defer @import("../arch/sched.zig").endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        defer m.reset();
        const p = m.plans.?;
        const chunk = p.region_rows;
        const ids = try spreadIds(a, &m.config, 4 * chunk);
        defer a.free(ids);
        for ([_]usize{ chunk, 2 * chunk + 1, 3 * chunk - 1 }) |at| {
            if (at < p.deep) continue;
            m.reset();
            try m.reserveFor(at + chunk);
            var done: usize = 0;
            while (done < at) {
                const n = @min(chunk, at - done);
                _ = mlx.mlx_array_free(try m.forwardReadout(ids[done .. done + n], .none));
                done += n;
            }
            mlx.engine.close();
            const lg = try hy4_plan.run(m, ids[at .. at + chunk], null, .last);
            defer _ = mlx.mlx_array_free(lg);
            try mlx.check(mlx.mlx_array_eval(lg));
            const class = p.pads.get(m.wideClass(@intCast(at), @intCast(chunk), @import("hy4_model.zig").Model.Readout.last)).?;
            errdefer std.debug.print("{s}: a {d}-row chunk at {d} tops out at {d} B, its class's recording at {d} B; the region {d} B\n", .{ env, chunk, at, core.stats().wide_last_high, class.need, p.region.scratch });
            try testing.expectEqual(class.need, core.stats().wide_last_high);
            try testing.expect(class.need <= p.region.scratch);
        }
    }
    if (!ran) return error.SkipZigTest;
}

/// Each scratch ask's place in a dry `rows`-row wide forward at `n`, in the order an execution makes them, every pass's
/// in turn (`plan.region.padOf`).
fn scratchAsks(m: anytype, p: anytype, rows: usize, n: usize) ![]u64 {
    var passes = try hy4_plan.recordChecked(m, testing.allocator, .{ .rows = rows, .readout = .last, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n, .phase = .wide }, p.reach);
    defer passes.deinit();
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(testing.allocator);
    for (passes.traces) |*t| {
        const pad = try plan.region.padOf(testing.allocator, t, p.book.alignment);
        defer testing.allocator.free(pad);
        try out.appendSlice(testing.allocator, pad);
    }
    return out.toOwnedSlice(testing.allocator);
}
