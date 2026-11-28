//! Hy4-Flash's samples on the arena (D97 everywhere): a served draw, from the logits to its token, is one narrow
//! execution on the shared arena region, sized by dry-probing sample mixes beside forwards during load planning.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const sampler = @import("../sampler.zig");
const hy4_model = @import("hy4_model.zig");
const fwd = @import("hy4_forward.zig");
const Plans = @import("hy4_plan.zig").Plans;
const Model = hy4_model.Model;
const Mix = sampler.Mix;
const Arr = mlx.mlx_array;

pub const token_slots = 32;

/// What a load's samples hold beside its forwards' plans (`hy4_plan.Plans.sample`).
pub const State = struct {
    /// Whether the load's samples are planned.
    planned: bool = false,
    /// The load's reserved-token mask, which its requests' draws read.
    masks: Masks = .{},
    /// The ring blocks a step's token is written into, in turn: the generator holds one token across the next draw.
    tokens: [token_slots]?Arr = @splat(null),
    turn: usize = 0,
    /// Whether a narrow execution is open for the current draw.
    open: bool = false,
    /// A draw no root holds that runs on the pool by design, open until it ends: a verdict wider than a verify block.
    wide: ?core.Wide = null,
    /// The widest verdict planned: a verify block's (D18) where the load drafts, else none.
    rows_max: u8 = 0,
    /// Draws run as narrow executions since load.
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

/// What a load's requests' draws read of the load's own: its reserved-token mask (`Trunk.suppress_mask`), borrowed.
pub const Masks = struct { suppress: ?Arr = null };

/// A served draw of `mix` begins: a narrow execution on the arena where the load planned it (`true`), or `false` on a
/// load without planned samples. Verdicts wider than a verify block run on the pool past the seal (`State.wide`).
pub fn begin(m: *Model, mix: Mix) !bool {
    const p = m.plans orelse return false;
    const st = &p.sample;
    if (!st.planned) return false;
    try remakeTokens(st, m.s);
    // A verdict wider than a verify block (a forced DFly2 width, D42) draws on the pool past the seal by design (D64).
    if (mix.rows > @max(st.rows_max, 1)) {
        st.wide = core.widePhase();
        return true;
    }
    try core.beginNarrow();
    st.open = true;
    st.draws += 1;
    return true;
}

/// A draw on the pool (`State.wide`) ends: its wide phase goes. Whether one was open.
fn endWide(st: *State) bool {
    const w = st.wide orelse return false;
    w.end();
    st.wide = null;
    return true;
}

/// Ends the active sample execution.
pub fn end(m: *Model) !void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    st.open = false;
    try core.endNarrow();
}

/// Aborts the active sample execution on error.
pub fn fail(m: *Model) void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    if (!st.open) return;
    st.open = false;
    core.abortNarrow();
}

/// A step's token `lazy` (owned) kept past its execution, which ends: written into the token block of this turn, the
/// caller handed a view of it in `lazy`'s shape.
pub fn token(m: *Model, lazy: Arr) !Arr {
    // A draw on the pool keeps its token as the sampler made it.
    if (endWide(&m.plans.?.sample)) return lazy;
    const view = kept(m, &m.plans.?.sample, lazy) catch |e| {
        fail(m);
        return e;
    };
    errdefer _ = mlx.mlx_array_free(view);
    try end(m);
    return view;
}

/// A forced call's token `id` kept as a draw's is (D36): null on a load that plans no samples.
pub fn forced(m: *Model, id: u32) !?Arr {
    const p = m.plans orelse return null;
    if (!p.sample.planned) return null;
    const st = &p.sample;
    try remakeTokens(st, m.s);
    // Written in place on the host: past the seal a fresh array for the id is an ask the pool refuses.
    const turn = try freeToken(st, true);
    const host = std.mem.bytesAsSlice(u32, try mlx.engine.hostWritable(st.tokens[turn].?));
    host[0] = id;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, st.tokens[turn].?, &[_]c_int{1}, 1, m.s));
    return view;
}

/// The token block the next kept token takes, in turn: the first no other handle reads, else the turn's own. `sole_only`
/// refuses the turn's own when another handle reads it, for a write that lands in place.
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
    const turn = pick orelse return error.Hy4TokenBlocksHeld;
    st.turn = (turn + 1) % token_slots;
    return turn;
}

fn kept(m: *Model, st: *State, lazy: Arr) !Arr {
    defer _ = mlx.mlx_array_free(lazy);
    const turn = try freeToken(st, true);
    // A write that fails inside an execution leaves its block owing bytes of the arena, which the execution's end takes
    // back: the block goes, and `remakeTokens` makes it again outside one.
    errdefer {
        st.turn = turn;
        if (st.tokens[turn]) |b| _ = mlx.mlx_array_free(b);
        st.tokens[turn] = null;
    }
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, lazy, &[_]c_int{1}, 1, m.s));
    try hy4_model.writeFixed(&st.tokens[turn], flat, &.{1}, m.s);
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    const sh = mlx.getShape(lazy);
    try mlx.check(mlx.mlx_reshape(&view, st.tokens[turn].?, sh.ptr, sh.len, m.s));
    return view;
}

// ── Probes ──

pub const Probe = struct { mix: Mix, other: bool = false };

/// Mixes used to envelope sample demand (`hy4_sample_plan.record`) and verified by tests.
pub fn mixes(gpa: std.mem.Allocator, st: *const State) ![]Mix {
    const suppress = st.masks.suppress != null;
    var list: std.ArrayList(Mix) = .empty;
    errdefer list.deinit(gpa);
    const bools = [_]bool{ false, true };
    for (bools) |greedy| for (bools) |repeat| for (bools) |additive| {
        const base: Mix = .{ .entry = .step, .suppress = suppress, .greedy = greedy, .repeat = repeat, .additive = additive };
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

/// Worst-case envelope mixes to size D97 sample demand quickly at boot.
fn envelopeMixes(gpa: std.mem.Allocator, st: *const State) ![]Mix {
    const suppress = st.masks.suppress != null;
    var list: std.ArrayList(Mix) = .empty;
    errdefer list.deinit(gpa);
    for ([_]bool{ false, true }) |greedy| {
        for ([_]bool{ false, true }) |grammar| {
            try list.append(gpa, .{ .entry = .step, .suppress = suppress, .greedy = greedy, .repeat = true, .additive = true, .grammar = grammar });
            if (st.rows_max > 0) {
                try list.append(gpa, .{ .entry = .verdict, .rows = 1, .suppress = suppress, .greedy = greedy, .repeat = true, .additive = true, .grammar = grammar });
                if (st.rows_max > 1) {
                    try list.append(gpa, .{ .entry = .verdict, .rows = st.rows_max, .suppress = suppress, .greedy = greedy, .repeat = true, .additive = true, .grammar = grammar });
                }
            }
        }
        try list.append(gpa, .{ .entry = .sync, .suppress = suppress, .greedy = greedy, .repeat = true, .additive = true });
    }
    return list.toOwnedSlice(gpa);
}

/// What a probe's draw reads, made before its recording as a served draw finds them: the logits in the load's block.
pub const Inputs = struct {
    gpa: std.mem.Allocator,
    logits: Arr,
    masks: ?[]bool = null,
    sampling: sampler.SamplingParams,
    generated: []const u32,
    ids: [@intCast(fwd.verify_rows_max)]u32 = undefined,

    pub fn deinit(in: *Inputs) void {
        _ = mlx.mlx_array_free(in.logits);
        if (in.masks) |b| in.gpa.free(b);
    }
};

const few_ids = [_]u32{ 1, 2, 3 };
const many_ids: [40]u32 = @splat(7);

pub fn makeInputs(m: *Model, st: *const State, gpa: std.mem.Allocator, probe: Probe) !Inputs {
    const mix = probe.mix;
    const vocab: c_int = @intCast(m.config.vocab);
    const rows: c_int = mix.rows;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, m.logits_fixed[0].?, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ rows, vocab }, 2, &[_]c_int{ 1, 1 }, 2, m.s));
    var in: Inputs = .{ .gpa = gpa, .logits = mlx.mlx_array_new(), .sampling = .{ .draw_root = 1 }, .generated = if (probe.other) &many_ids else &few_ids };
    errdefer in.deinit();
    try mlx.check(mlx.mlx_reshape(&in.logits, head, &[_]c_int{ 1, rows, vocab }, 3, m.s));
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
        in.masks = try gpa.alloc(bool, @as(usize, mix.rows) * m.config.vocab);
        @memset(in.masks.?, true);
    }
    return in;
}

/// What a draw hands back: a step's token (a view of its block), a serial draw's token, or a verdict.
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

const Verdict = @import("hy4_drafter.zig").rounds.Verdict;

/// A draw of `mix` over `in`, as a served draw of its mix runs it.
pub fn draw(m: *Model, st: *State, in: *const Inputs, mix: Mix) !Drawn {
    const s = m.s;
    switch (mix.entry) {
        .step => {
            const step: sampler.Step = .{ .logits = in.logits, .sampling = in.sampling, .generated = in.generated, .grammar = if (in.masks) |b| b else null };
            if (!std.meta.eql(step.mix(), mix)) return error.Hy4SampleProbeMix;
            return .{ .step = try kept(m, st, try sampler.sampleStep(std.heap.page_allocator, step, s)) };
        },
        .sync => {
            if (!std.meta.eql(sampler.syncMix(in.sampling, in.generated), mix)) return error.Hy4SampleProbeMix;
            return .{ .sync = try sampler.sampleToken(in.logits, in.sampling, in.generated, s) };
        },
        .verdict => {
            const rounds = @import("hy4_drafter.zig").rounds;
            const v: rounds.VerdictIn = .{ .logits = in.logits, .ids = in.ids[0..mix.rows], .sampling = in.sampling, .committed = if (mix.repeat or mix.additive) in.generated else null, .masks = if (in.masks) |b| b else null, .streams = .{ .ops = s, .greedy = s, .draws = s } };
            if (!std.meta.eql(v.mix(), mix)) return error.Hy4SampleProbeMix;
            return .{ .verdict = try rounds.verdictOf(v) };
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
        try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
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
    try fillTokens(st, m.s);
}

fn fillTokens(st: *State, s: mlx.mlx_stream) !void {
    for (&st.tokens) |*b| if (b.* == null) {
        b.* = (try @import("../mlx_pages.zig").ringBuffer(&.{1}, .uint32, s)).array;
    };
}

/// The token blocks a failed draw let go (`kept`), made again: the pool's gate stays closed to every other ask.
fn remakeTokens(st: *State, s: mlx.mlx_stream) !void {
    for (st.tokens) |b| if (b == null) {
        const wide = core.widePhase();
        defer wide.end();
        try fillTokens(st, s);
        return;
    };
}

/// `probe` recorded under both fills: verifies readbacks do not steer allocation shapes.
fn recordChecked(m: *Model, st: *State, gpa: std.mem.Allocator, probe: Probe) !plan_trace.Trace {
    var zeros = try recordPass(m, st, gpa, probe, .zeros);
    errdefer zeros.deinit();
    var ramp = try recordPass(m, st, gpa, probe, .ramp);
    defer ramp.deinit();
    if (plan_trace.firstDifference(zeros.view(), ramp.view())) |d| {
        log.err("[plan] a sample ({t}, {d} rows) records two shapes under two fills ({t} at {d}): a readback steers it\n", .{ probe.mix.entry, probe.mix.rows, d.what, d.at });
        return error.Hy4PlanReadbackSteers;
    }
    return zeros;
}

/// Measures worst-case D97 sample demand and folds it into `p.narrow`.
pub fn record(m: *Model, p: *Plans, masks: Masks) !void {
    const t0 = @import("../load_clock.zig").now();
    const gpa = p.gpa;
    const st = &p.sample;
    st.* = .{ .planned = true, .masks = masks, .rows_max = if (p.drafter != null) @intCast(fwd.verify_rows_max) else 0 };
    errdefer st.deinit();
    try @import("../sampler_draw.zig").settle(m.config.vocab, .float32, m.s);
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
    log.info("[plan] samples: {d} envelopes, verdicts to {d} rows; scratch {d} KB, host ring {d} KB in {d} ms\n", .{ envs.len, st.rows_max, sample_need.scratch >> 10, sample_need.host >> 10, ms });
}
