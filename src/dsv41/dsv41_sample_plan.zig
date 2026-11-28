//! DeepSeek-V4.1's samples on the static arena (D97 Everywhere): served draws (lazy `.step`, synchronous `.sync`,
//! and DSpark `.verdict`) run as narrow executions on the shared arena sized from worst-case sample envelopes.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const sampler = @import("../sampler.zig");
const cfg_mod = @import("dsv41_config.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_store = @import("dsv41_store.zig");
const dsv41_dspark = @import("dsv41_dspark.zig");
const dsv41_plan = @import("dsv41_plan.zig");

const Plans = dsv41_plan.Plans;
const Model = dsv41_model.Model;
const Mix = sampler.Mix;
const Arr = mlx.mlx_array;

pub const token_slots = 32;

/// What a load's samples hold beside its forwards' plans (`dsv41_plan.Plans.sample`).
pub const State = struct {
    planned: bool = false,
    masks: Masks = .{},
    tokens: [token_slots]?Arr = @splat(null),
    turn: usize = 0,
    open: bool = false,
    wide: ?core.Wide = null,
    rows_max: u8 = 0,
    draws: u64 = 0,

    pub fn deinit(st: *State) void {
        freeTokens(&st.tokens);
    }
};

fn freeTokens(tokens: *[token_slots]?Arr) void {
    for (tokens) |*b| if (b.*) |a| {
        _ = mlx.mlx_array_free(a);
        b.* = null;
    };
}

/// Reserved-token mask borrowed from the load.
pub const Masks = struct {
    suppress: ?Arr = null,
};

pub fn begin(m: *Model, mix: Mix) !bool {
    const p = m.plans orelse return false;
    const st = &p.sample;
    if (!st.planned) return false;
    const s = dsv41_model.stateStream(m);
    try remakeTokens(st, s);
    if (mix.rows > @max(st.rows_max, 1)) {
        st.wide = core.widePhase();
        return true;
    }
    try core.beginNarrow();
    st.open = true;
    st.draws += 1;
    return true;
}

fn endWide(st: *State) bool {
    const w = st.wide orelse return false;
    w.end();
    st.wide = null;
    return true;
}

pub fn end(m: *Model) !void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    st.open = false;
    try core.endNarrow();
}

pub fn fail(m: *Model) void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    if (!st.open) return;
    st.open = false;
    core.abortNarrow();
}

pub fn token(m: *Model, lazy: Arr) !Arr {
    if (endWide(&m.plans.?.sample)) return lazy;
    const view = kept(m, &m.plans.?.sample, lazy) catch |e| {
        fail(m);
        return e;
    };
    errdefer _ = mlx.mlx_array_free(view);
    try end(m);
    return view;
}

pub fn forced(m: *Model, id: u32) !?Arr {
    const p = m.plans orelse return null;
    if (!p.sample.planned) return null;
    const st = &p.sample;
    const s = dsv41_model.stateStream(m);
    try remakeTokens(st, s);
    const turn = try freeToken(st, true);
    const host = std.mem.bytesAsSlice(u32, try mlx.engine.hostWritable(st.tokens[turn].?));
    host[0] = id;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, st.tokens[turn].?, &[_]c_int{1}, 1, s));
    return view;
}

fn freeToken(st: *State, sole_only: bool) !usize {
    var pick: ?usize = if (sole_only) null else st.turn % token_slots;
    for (0..token_slots) |off| {
        const idx = (st.turn + off) % token_slots;
        if (st.tokens[idx]) |b| {
            if (try mlx.engine.sole(b)) {
                pick = idx;
                break;
            }
        } else if (!sole_only) {
            pick = idx;
            break;
        }
    }
    const turn = pick orelse return error.Dsv41TokenBlocksHeld;
    st.turn = (turn + 1) % token_slots;
    return turn;
}

fn kept(m: *Model, st: *State, lazy: Arr) !Arr {
    defer _ = mlx.mlx_array_free(lazy);
    const turn = try freeToken(st, true);
    errdefer {
        st.turn = turn;
        if (st.tokens[turn]) |b| _ = mlx.mlx_array_free(b);
        st.tokens[turn] = null;
    }
    const s = dsv41_model.stateStream(m);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, lazy, &[_]c_int{1}, 1, s));
    try dsv41_store.writeFixed(&st.tokens[turn], flat, &.{1}, s);
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    const sh = mlx.getShape(lazy);
    try mlx.check(mlx.mlx_reshape(&view, st.tokens[turn].?, sh.ptr, sh.len, s));
    return view;
}

// ── Probes ──

pub const Probe = struct { mix: Mix, other: bool = false };

/// Mixes used to validate full parity across all combinations in tests.
pub fn mixes(gpa: std.mem.Allocator, st: *const State) ![]Mix {
    const suppress = st.masks.suppress != null;
    var list: std.ArrayList(Mix) = .empty;
    errdefer list.deinit(gpa);
    const bools = [_]bool{ false, true };
    for (bools) |greedy| for (bools) |repeat| for (bools) |additive| {
        const base: Mix = .{
            .entry = .step,
            .suppress = suppress,
            .greedy = greedy,
            .repeat = repeat,
            .additive = additive,
        };
        for (bools) |grammar| {
            var step = base;
            step.grammar = grammar;
            try list.append(gpa, step);
            if (st.rows_max == 0) continue;
            for (1..@as(usize, st.rows_max) + 1) |rows| {
                var v = step;
                v.entry = .verdict;
                v.rows = @intCast(rows);
                try list.append(gpa, v);
            }
        }
        var sync = base;
        sync.entry = .sync;
        try list.append(gpa, sync);
    };
    return list.toOwnedSlice(gpa);
}

fn envelopeMixes(gpa: std.mem.Allocator, st: *const State) ![]Mix {
    const suppress = st.masks.suppress != null;
    var list: std.ArrayList(Mix) = .empty;
    errdefer list.deinit(gpa);
    for ([_]bool{ false, true }) |greedy| {
        try list.append(gpa, .{
            .entry = .step,
            .suppress = suppress,
            .greedy = greedy,
            .repeat = true,
            .additive = true,
            .grammar = true,
        });
        try list.append(gpa, .{
            .entry = .sync,
            .suppress = suppress,
            .greedy = greedy,
            .repeat = true,
            .additive = true,
        });
        if (st.rows_max > 0) {
            try list.append(gpa, .{
                .entry = .verdict,
                .rows = st.rows_max,
                .suppress = suppress,
                .greedy = greedy,
                .repeat = true,
                .additive = true,
                .grammar = true,
            });
            if (st.rows_max > 1) {
                try list.append(gpa, .{
                    .entry = .verdict,
                    .rows = 1,
                    .suppress = suppress,
                    .greedy = greedy,
                    .repeat = true,
                    .additive = true,
                    .grammar = true,
                });
            }
        }
    }
    return list.toOwnedSlice(gpa);
}

pub const Inputs = struct {
    gpa: std.mem.Allocator,
    logits: Arr,
    masks: ?[]bool = null,
    sampling: sampler.SamplingParams,
    generated: []const u32,
    ids: [cfg_mod.ring_slack]u32 = undefined,

    pub fn deinit(in: *Inputs) void {
        _ = mlx.mlx_array_free(in.logits);
        if (in.masks) |b| in.gpa.free(b);
    }
};

const few_ids = [_]u32{ 1, 2, 3 };
const many_ids: [40]u32 = @splat(7);

pub fn makeInputs(m: *Model, st: *const State, gpa: std.mem.Allocator, probe: Probe) !Inputs {
    const mix = probe.mix;
    const s = dsv41_model.stateStream(m);
    const vocab: c_int = @intCast(m.cfg.vocab);
    const rows: c_int = mix.rows;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, m.logits_fixed[0].?, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ rows, vocab }, 2, &[_]c_int{ 1, 1 }, 2, s));
    var in: Inputs = .{
        .gpa = gpa,
        .logits = mlx.mlx_array_new(),
        .sampling = .{ .draw_root = 1 },
        .generated = if (probe.other) &many_ids else &few_ids,
    };
    errdefer in.deinit();
    try mlx.check(mlx.mlx_reshape(&in.logits, head, &[_]c_int{ 1, rows, vocab }, 3, s));
    for (&in.ids, 0..) |*id, i| id.* = @intCast(i + 1);
    var sp = &in.sampling;
    sp.suppress_mask = if (mix.suppress) st.masks.suppress else null;
    sp.temperature = if (mix.greedy) 0 else if (probe.other) 1.3 else 0.7;
    if (!mix.greedy and probe.other) {
        sp.top_k = 40;
        sp.top_p = 0.9;
        sp.min_p = 0.05;
        sp.seed = 7;
    }
    if (mix.repeat) sp.repeat_penalty = 1.1;
    if (mix.additive) {
        sp.presence_penalty = 0.5;
        if (probe.other) sp.frequency_penalty = 0.25;
    }
    if (mix.grammar) {
        in.masks = try gpa.alloc(bool, @as(usize, mix.rows) * m.cfg.vocab);
        @memset(in.masks.?, true);
    }
    return in;
}

pub const Drawn = union(enum) {
    step: Arr,
    sync: sampler.SampleResult,
    verdict: Verdict,

    pub fn deinit(d: Drawn) void {
        switch (d) {
            .step => |a| _ = mlx.mlx_array_free(a),
            .sync, .verdict => {},
        }
    }
};

const Verdict = dsv41_dspark.Verdict;

pub fn draw(m: *Model, st: *State, in: *const Inputs, mix: Mix) !Drawn {
    const s = dsv41_model.stateStream(m);
    switch (mix.entry) {
        .step => {
            const step: sampler.Step = .{
                .logits = in.logits,
                .sampling = in.sampling,
                .generated = in.generated,
                .grammar = if (in.masks) |b| b else null,
            };
            if (!std.meta.eql(step.mix(), mix)) return error.Dsv41SampleProbeMix;
            return .{ .step = try kept(m, st, try sampler.sampleStep(std.heap.page_allocator, step, s)) };
        },
        .sync => {
            if (!std.meta.eql(sampler.syncMix(in.sampling, in.generated), mix)) return error.Dsv41SampleProbeMix;
            return .{ .sync = try sampler.sampleToken(in.logits, in.sampling, in.generated, s) };
        },
        .verdict => {
            const v: dsv41_dspark.VerdictIn = .{
                .logits = in.logits,
                .ids = in.ids[0..mix.rows],
                .sampling = in.sampling,
                .committed = if (mix.repeat or mix.additive) in.generated else null,
                .masks = if (in.masks) |b| b else null,
                .streams = .{ .ops = s, .greedy = s, .draws = s },
            };
            if (!std.meta.eql(v.mix(), mix)) return error.Dsv41SampleProbeMix;
            return .{ .verdict = try dsv41_dspark.verdictOf(v) };
        },
    }
}

fn recordPass(m: *Model, st: *State, gpa: std.mem.Allocator, probe: Probe, fill: core.Fill) !plan_trace.Trace {
    var in = try makeInputs(m, st, gpa, probe);
    defer in.deinit();
    const held = st.tokens;
    const turn = st.turn;
    st.tokens = @splat(null);
    defer {
        freeTokens(&st.tokens);
        st.tokens = held;
        st.turn = turn;
    }
    var driver: plan.Driver = .{ .gpa = gpa };
    defer driver.deinit();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var prep_driver: plan.Driver = .{ .gpa = gpa };
        defer prep_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &prep_driver } }, &rec);
        defer core.endRecording() catch {};
        try makeTokens(m, st);
    }
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = &driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        (try draw(m, st, &in, probe.mix)).deinit();
    }
    return rec.end();
}

pub fn makeTokens(m: *Model, st: *State) !void {
    freeTokens(&st.tokens);
    st.turn = 0;
    const wide = core.widePhase();
    defer wide.end();
    try fillTokens(st, dsv41_model.stateStream(m));
}

fn fillTokens(st: *State, s: mlx.mlx_stream) !void {
    for (&st.tokens) |*b| if (b.* == null) {
        b.* = (try @import("../mlx_pages.zig").ringBuffer(&.{1}, .uint32, s)).array;
    };
}

fn remakeTokens(st: *State, s: mlx.mlx_stream) !void {
    for (st.tokens) |b| if (b == null) {
        const wide = core.widePhase();
        defer wide.end();
        try fillTokens(st, s);
        return;
    };
}

fn recordChecked(m: *Model, st: *State, gpa: std.mem.Allocator, probe: Probe) !plan_trace.Trace {
    var zeros = try recordPass(m, st, gpa, probe, .zeros);
    errdefer zeros.deinit();
    var ramp = try recordPass(m, st, gpa, probe, .ramp);
    defer ramp.deinit();
    if (plan_trace.firstDifference(zeros.view(), ramp.view())) |d| {
        log.err("[dsv41-plan] a sample ({t}, {d} rows) records two shapes under two fills ({t} at {d})\n", .{ probe.mix.entry, probe.mix.rows, d.what, d.at });
        return error.Dsv41PlanReadbackSteers;
    }
    return zeros;
}

/// Measures worst-case D97 sample demand and folds it into `p.narrow`.
pub fn record(m: *Model, p: *Plans, masks: Masks) !void {
    const t0 = @import("../load_clock.zig").now();
    const gpa = p.gpa;
    const st = &p.sample;
    const rows_max: u8 = if (m.dspark != null) @intCast(@min(@as(usize, m.cfg.dspark.block) + 1, cfg_mod.ring_slack)) else 0;
    st.* = .{ .planned = true, .masks = masks, .rows_max = rows_max };
    errdefer st.deinit();
    const s = dsv41_model.stateStream(m);
    if (m.logits_fixed[0] == null) try m.makeLogitsBlocks();
    try @import("../sampler_draw.zig").settle(m.cfg.vocab, .float32, s);
    try makeTokens(m, st);
    const envs = try envelopeMixes(gpa, st);
    defer gpa.free(envs);
    var sample_need: plan.region.Need = .{};
    for (envs) |mix| {
        for ([_]bool{ false, true }) |other| {
            var tr = try recordChecked(m, st, gpa, .{ .mix = mix, .other = other });
            defer tr.deinit();
            sample_need = sample_need.max(try plan.region.needWithGuard(gpa, &tr, p.alignment, p.guard));
        }
    }
    p.narrow = p.narrow.max(sample_need);
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[dsv41-plan] samples: {d} envelopes, verdicts to {d} rows; scratch {d} KB, host ring {d} KB in {d} ms\n", .{ envs.len, st.rows_max, sample_need.scratch >> 10, sample_need.host >> 10, ms });
}
