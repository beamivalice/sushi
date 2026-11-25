//! Hy4-Flash's samples on the arena (docs/bill_redesign.md, D24): a served draw, from the logits to its token, is one
//! execution of a `sample` root keyed by its mix (`sampler.Mix`). Each mix the sampler can take is recorded dry at the
//! load's end beside the forwards' roots (`build`); a served draw of a mix no probe recorded, or one that misses inside
//! its execution, is refused by name and its mix recorded and learned (D5), the shape kept for the next load.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_book = @import("../tensor/plan/plan_book.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const sampler = @import("../sampler.zig");
const think_penalty = @import("../think_penalty.zig");
const hy4_model = @import("hy4_model.zig");
const fwd = @import("hy4_forward.zig");
const Plans = @import("hy4_plan.zig").Plans;
const Model = hy4_model.Model;
const Mix = sampler.Mix;
const Arr = mlx.mlx_array;

/// A sample's root: its mix, a verdict's rows only as one row or a block (its rows are a size).
pub fn root(mix: Mix) plan.FrameId {
    return .{ .site = plan.siteOf(@src()), .key = plan.keyOf(.{ mix.entry, mix.rows <= 1, mix.suppress, mix.greedy, mix.think, mix.repeat, mix.additive, mix.grammar }) };
}

/// What a load's samples hold beside its forwards' plans (`hy4_plan.Plans.sample`).
pub const State = struct {
    /// Whether the load's samples are planned (`build`).
    planned: bool = false,
    /// The load's reserved-token and think-marker masks, which its requests' draws read.
    masks: Masks = .{},
    /// The two blocks a step's token is written into, in turn: the generator holds one token across the next draw.
    tokens: [2]?Arr = .{ null, null },
    turn: u1 = 0,
    /// The mix of the execution open, when one is.
    open: ?Mix = null,
    /// A draw no root holds that runs on the pool by design, open until it ends: a verdict wider than a verify block.
    wide: ?core.Wide = null,
    /// The widest verdict planned: a verify block's (D18) where the load drafts, else none.
    rows_max: u8 = 0,
    /// Draws run as executions since the load; misses served, and of them the mixes learned in place.
    draws: u64 = 0,
    misses: u64 = 0,
    learned: u64 = 0,

    pub fn deinit(st: *State) void {
        freeTokens(&st.tokens);
    }
};

fn freeTokens(tokens: *[2]?Arr) void {
    for (tokens) |*b| if (b.*) |a| {
        _ = mlx.mlx_array_free(a);
        b.* = null;
    };
}

/// What a load's requests' draws read of the load's own: its reserved-token mask (`Trunk.suppress_mask`) and its think
/// markers' (`Trunk.think_marker_mask`), borrowed.
pub const Masks = struct { suppress: ?Arr = null, think: ?Arr = null };

/// A served draw of `mix` begins: an execution of its root where the load planned it (true), else none (false): a
/// load without plans, or a verdict wider than a verify block, which is a prefill block's on the pool (D18). A mix the
/// load planned no root for is refused by name (`PlanMiss`), recorded and learned.
pub fn begin(m: *Model, mix: Mix) !bool {
    const p = m.plans orelse return false;
    const st = &p.sample;
    if (!st.planned) return false;
    try remakeTokens(st, m.s);
    // A verdict wider than a verify block (a forced DSpark width, D42) draws on the pool past the seal by design (D64).
    if (mix.rows > @max(st.rows_max, 1)) {
        st.wide = core.widePhase();
        return true;
    }
    const tables = p.book.tables(root(mix)) orelse return missed(m, mix, null);
    try core.beginExecution(tables, root(mix));
    st.open = mix;
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

/// The execution ends; a miss it met is `PlanMiss`.
pub fn end(m: *Model) !void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    const mix = st.open.?;
    st.open = null;
    core.endExecution() catch |e| return if (core.executionMiss()) |miss| missed(m, mix, miss) else e;
}

/// The sampler failed inside the execution: it goes, and a miss it met is `PlanMiss`.
pub fn fail(m: *Model) error{PlanMiss}!void {
    const st = &m.plans.?.sample;
    if (endWide(st)) return;
    const mix = st.open orelse return;
    st.open = null;
    core.abortExecution();
    if (core.executionMiss()) |miss| return missed(m, mix, miss);
}

/// A step's token `lazy` (owned) kept past its execution, which ends: written into the token block of this turn, the
/// caller handed a view of it in `lazy`'s shape.
pub fn token(m: *Model, lazy: Arr) !Arr {
    // A draw on the pool keeps its token as the sampler made it.
    if (endWide(&m.plans.?.sample)) return lazy;
    const view = kept(m, &m.plans.?.sample, lazy) catch |e| {
        try fail(m);
        return e;
    };
    errdefer _ = mlx.mlx_array_free(view);
    try end(m);
    return view;
}

/// A forced call's token `id` kept as a draw's is (D36): the next draw's probes took a pending token as a view of a
/// token block, so it is one. Null on a load that plans no samples.
pub fn forced(m: *Model, id: u32) !?Arr {
    const p = m.plans orelse return null;
    if (!p.sample.planned) return null;
    try remakeTokens(&p.sample, m.s);
    return try kept(m, &p.sample, try mlx.newData(&id, &.{1}, .uint32));
}

fn kept(m: *Model, st: *State, lazy: Arr) !Arr {
    defer _ = mlx.mlx_array_free(lazy);
    const turn = st.turn;
    st.turn +%= 1;
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

/// A kept shape's phase: a sample (`Mix`), its mix packed in the first number.
pub const shape_phase: u32 = 3;

pub fn shapeOf(mix: Mix) plan_book.Shape {
    const flags: u64 = @as(u64, @intFromBool(mix.suppress)) | @as(u64, @intFromBool(mix.greedy)) << 1 | @as(u64, @intFromBool(mix.repeat)) << 2 |
        @as(u64, @intFromBool(mix.additive)) << 3 | @as(u64, @intFromBool(mix.grammar)) << 4;
    return .{ .phase = shape_phase, .args = .{ @intFromEnum(mix.entry), mix.rows, @intFromEnum(mix.think), flags } };
}

/// The mix a kept shape names, or null when it names none.
pub fn mixOf(s: plan_book.Shape) ?Mix {
    if (s.phase != shape_phase or s.args[1] == 0 or s.args[1] > 255 or s.args[3] >> 5 != 0) return null;
    const entry = std.enums.fromInt(Mix.Entry, s.args[0]) orelse return null;
    const think = std.enums.fromInt(Mix.Think, s.args[2]) orelse return null;
    const f = s.args[3];
    if (entry != .verdict and s.args[1] != 1) return null;
    return .{ .entry = entry, .rows = @intCast(s.args[1]), .think = think, .suppress = f & 1 != 0, .greedy = f & 2 != 0, .repeat = f & 4 != 0, .additive = f & 8 != 0, .grammar = f & 16 != 0 };
}

/// A served draw of `mix` missed (`miss`, null for a mix no root was planned for): it is refused by name, its mix
/// recorded and learned (`plan_book.Book.learnAll`) and kept beside the load's manifest either way.
fn missed(m: *Model, mix: Mix, miss: ?plan.binder.Miss) error{PlanMiss} {
    const p = m.plans.?;
    const st = &p.sample;
    st.misses += 1;
    if (miss) |x| {
        log.warn("[plan] miss: a sample ({t}, {d} rows, think {t}) asked {d} B against {d} B ({t}, frame {d}/{d}, ordinal {d})\n", .{ mix.entry, mix.rows, mix.think, x.asked, x.cap, x.kind, x.id.site, x.id.key, x.ordinal });
    } else log.warn("[plan] miss: a sample ({t}, {d} rows, think {t}) of a mix the load did not plan\n", .{ mix.entry, mix.rows, mix.think });
    remakeTokens(st, m.s) catch |e| {
        log.warn("[plan] the token blocks of a missed sample were not made again ({t}); the next draw makes them or fails\n", .{e});
        return error.PlanMiss;
    };
    const held = core.arenaLayout() orelse return error.PlanMiss;
    // A mix's probes make their inputs on the pool, past the seal.
    const wide = core.widePhase();
    const passes = recordMix(m, st, p.book.gpa, mix);
    wide.end();
    var traces = passes catch |e| {
        log.warn("[plan] the missed sample was not recorded ({t}); it misses again until a load plans it\n", .{e});
        return error.PlanMiss;
    };
    defer traces.deinit(p.book.gpa);
    const learned = p.book.learnAll(root(mix), traces.items, held) catch |e| {
        traces.items.len = 0;
        log.warn("[plan] the missed sample was recorded and not learned ({t})\n", .{e});
        return error.PlanMiss;
    };
    traces.items.len = 0;
    switch (learned) {
        .swapped => {
            st.learned += 1;
            log.warn("[plan] the missed sample is learned in place: the next such draw is served\n", .{});
        },
        .next_load => log.warn("[plan] the missed sample needs more arena than the load holds: the next load plans it\n", .{}),
    }
    if (p.cache) |*c| c.keep(p.book.gpa, shapeOf(mix)) catch |e| log.warn("[plan] the missed sample was not kept in {s} ({t}): a restart misses it again\n", .{ c.name(), e });
    return error.PlanMiss;
}

// ── The probes ──

/// One shape a sample's root is recorded at: its mix, and which of two sets of the values its key leaves out (the
/// temperature's value, top-k, top-p, min-p, the seed and the penalties' ids).
pub const Probe = struct { mix: Mix, other: bool = false };

/// Every mix a draw of the load can take: each entry under each think penalty's shift the load can make (its marker
/// mask's where it has one, a request's biases always), each penalty, greedy or not, a grammar mask; the load's own
/// reserved-token mask; a verdict at every rows up to a verify block's where the load drafts.
pub fn mixes(gpa: std.mem.Allocator, st: *const State) ![]Mix {
    const has_mask = st.masks.think != null;
    const suppress = st.masks.suppress != null;
    var list: std.ArrayList(Mix) = .empty;
    errdefer list.deinit(gpa);
    const bools = [_]bool{ false, true };
    for (std.enums.values(Mix.Think)) |think| for (bools) |greedy| for (bools) |repeat| for (bools) |additive| {
        const rows_shift = think != .off and think != .mask and think != .bias;
        const by_mask = think == .mask or think == .mask_inside or think == .mask_opens or think == .mask_unopened;
        if (by_mask and !has_mask) continue;
        const base: Mix = .{ .entry = .step, .suppress = suppress, .greedy = greedy, .think = think, .repeat = repeat, .additive = additive };
        for (bools) |grammar| {
            var step = base;
            step.grammar = grammar;
            try list.append(gpa, step);
            if (st.rows_max == 0 or (think != .off and !rows_shift)) continue;
            for (1..@as(usize, st.rows_max) + 1) |rows| {
                var v = step;
                v.entry = .verdict;
                v.rows = @intCast(rows);
                try list.append(gpa, v);
            }
        }
        if (rows_shift) continue;
        var sync = base;
        sync.entry = .sync;
        try list.append(gpa, sync);
    };
    return list.toOwnedSlice(gpa);
}

/// What a probe's draw reads, made before its recording as a served draw finds them: the logits in the load's block,
/// a pending token in a token block, a request's prepared biases.
pub const Inputs = struct {
    gpa: std.mem.Allocator,
    logits: Arr,
    pending: ?Arr = null,
    prepared: ?think_penalty.Prepared = null,
    masks: ?[]bool = null,
    sampling: sampler.SamplingParams,
    generated: []const u32,
    ids: [@intCast(fwd.verify_rows_max)]u32 = undefined,

    pub fn deinit(in: *Inputs) void {
        _ = mlx.mlx_array_free(in.logits);
        if (in.pending) |a| _ = mlx.mlx_array_free(a);
        if (in.prepared) |*pr| pr.deinit();
        if (in.masks) |b| in.gpa.free(b);
    }

    fn shift(in: *const Inputs, mix: Mix, mask: ?Arr) ?sampler.Shift {
        if (mix.think == .off) return null;
        return .{ .mask = mask, .state = in.sampling.think_penalty, .ids = in.pending };
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
    var tp: think_penalty.ThinkPenalty = .{ .lambda = 1, .opener_id = 1, .closer_id = 2, .phase = .inside };
    switch (mix.think) {
        .off => tp = .{},
        .mask, .mask_inside, .bias, .bias_inside => {},
        .mask_opens, .bias_opens => tp.phase = .before,
        .mask_unopened, .bias_unopened => {
            tp.phase = .before;
            tp.opener_id = null;
        },
        .bias_after => tp.phase = .after,
    }
    switch (mix.think) {
        .bias, .bias_inside, .bias_opens, .bias_unopened, .bias_after => {
            const one = [_]@import("../logit_bias.zig").Bias{.{ .id = 1, .delta = 1 }};
            in.prepared = try think_penalty.Prepared.init(gpa, m.config.vocab, null, .{ .biases = &one }, m.s);
            tp.prepared = in.prepared;
        },
        else => {},
    }
    sp.think_penalty = tp;
    if (mix.entry == .step and mix.think != .off and mix.think != .mask and mix.think != .bias) {
        var p = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(p);
        // The token in flight is in the block before the one the next draw writes.
        try mlx.check(mlx.mlx_reshape(&p, st.tokens[st.turn -% 1].?, &[_]c_int{1}, 1, m.s));
        in.pending = p;
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

const Verdict = @import("hy4_dspark.zig").rounds.Verdict;

/// A draw of `mix` over `in`, as a served draw of its mix runs it (a step's token kept in its block, the execution, if
/// any, left open).
pub fn draw(m: *Model, st: *State, in: *const Inputs, mix: Mix) !Drawn {
    const s = m.s;
    const think_mask = st.masks.think;
    switch (mix.entry) {
        .step => {
            const step: sampler.Step = .{ .logits = in.logits, .sampling = in.sampling, .generated = in.generated, .grammar = if (in.masks) |b| b else null, .shift = in.shift(mix, think_mask) };
            if (!std.meta.eql(step.mix(), mix)) return error.Hy4SampleProbeMix;
            return .{ .step = try kept(m, st, try sampler.sampleStep(std.heap.page_allocator, step, s)) };
        },
        .sync => {
            const shift: ?sampler.MarkerShift = if (in.shift(mix, think_mask)) |sh| .{ .mask = sh.mask, .state = sh.state } else null;
            if (!std.meta.eql(sampler.syncMix(in.sampling, in.generated, shift), mix)) return error.Hy4SampleProbeMix;
            return .{ .sync = try sampler.sampleToken(in.logits, in.sampling, in.generated, shift, s) };
        },
        .verdict => {
            const rounds = @import("hy4_dspark.zig").rounds;
            const v: rounds.VerdictIn = .{ .logits = in.logits, .ids = in.ids[0..mix.rows], .sampling = in.sampling, .think = in.shift(mix, think_mask), .committed = if (mix.repeat or mix.additive) in.generated else null, .masks = if (in.masks) |b| b else null, .streams = .{ .ops = s, .greedy = s, .draws = s } };
            if (!std.meta.eql(v.mix(), mix)) return error.Hy4SampleProbeMix;
            return .{ .verdict = try rounds.verdictOf(v) };
        },
    }
}

/// `probe` recorded dry under `fill`. The token blocks it writes are its own, made dry beside it: a dry recording leaves
/// the blocks it writes unusable, and the generator may hold a view of the load's.
fn recordPass(m: *Model, st: *State, gpa: std.mem.Allocator, probe: Probe, fill: core.Fill) !plan_trace.Trace {
    var in = try makeInputs(m, st, gpa, probe);
    defer in.deinit();
    const held = st.tokens;
    const turn = st.turn;
    st.tokens = .{ null, null };
    defer {
        freeTokens(&st.tokens);
        st.tokens = held;
        st.turn = turn;
    }
    var driver: plan.Driver = .{ .gpa = gpa };
    defer driver.deinit();
    {
        var rec = try plan.Recorder.begin(gpa, .{ .site = plan.siteOf(@src()) });
        defer rec.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
        defer core.endRecording() catch {};
        try makeTokens(m, st);
    }
    var rec = try plan.Recorder.begin(gpa, root(probe.mix));
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = &driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        (try draw(m, st, &in, probe.mix)).deinit();
    }
    return rec.end();
}

fn makeTokens(m: *Model, st: *State) !void {
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
    if (st.tokens[0] != null and st.tokens[1] != null) return;
    const wide = core.widePhase();
    defer wide.end();
    try fillTokens(st, s);
}

/// `probe` recorded under both fills: the zeros' trace, once the ramp's is found to be the same execution (a readback
/// that steered a shape would part them).
fn recordChecked(m: *Model, st: *State, gpa: std.mem.Allocator, probe: Probe) !plan_trace.Trace {
    var zeros = try recordPass(m, st, gpa, probe, .zeros);
    errdefer zeros.deinit();
    var ramp = try recordPass(m, st, gpa, probe, .ramp);
    defer ramp.deinit();
    if (plan_trace.firstDifference(zeros.instances, ramp.instances)) |d| {
        log.err("[plan] a sample ({t}, {d} rows) records two shapes under two fills (instance {d}, {t} at {d}): a readback steers it\n", .{ probe.mix.entry, probe.mix.rows, d.instance, d.what, d.at });
        return error.Hy4PlanReadbackSteers;
    }
    return zeros;
}

/// The traces of `mix`'s probes: each set of the values its key leaves out.
fn recordMix(m: *Model, st: *State, gpa: std.mem.Allocator, mix: Mix) !std.ArrayList(plan_trace.Trace) {
    var list: std.ArrayList(plan_trace.Trace) = .empty;
    errdefer {
        for (list.items) |*t| t.deinit();
        list.deinit(gpa);
    }
    for ([_]bool{ false, true }) |other| {
        try list.ensureUnusedCapacity(gpa, 1);
        list.appendAssumeCapacity(try recordChecked(m, st, gpa, .{ .mix = mix, .other = other }));
    }
    return list;
}

/// The samples of `m`'s load, planned beside its forwards in `p` (after `hy4_plan.record`, before its arena): every mix
/// (`mixes`) recorded under both fills, merged and packed into the book, and the shapes earlier misses kept learned.
pub fn record(m: *Model, p: *Plans, masks: Masks) !void {
    const t0 = @import("../load_clock.zig").now();
    const gpa = p.book.gpa;
    const st = &p.sample;
    st.* = .{ .planned = true, .masks = masks, .rows_max = if (p.drafter != null) @intCast(fwd.verify_rows_max) else 0 };
    errdefer st.deinit();
    // The draw's threads a group settle on what the GPU grants before the probes record them (an M1 grants it 512).
    try @import("../sampler_draw.zig").settle(m.config.vocab, .float32, m.s);
    try makeTokens(m, st);
    const all = try mixes(gpa, st);
    defer gpa.free(all);
    for (all) |mix| {
        var traces = try recordMix(m, st, gpa, mix);
        defer traces.deinit(gpa);
        for (traces.items, 0..) |t, i| p.book.add(root(mix), t) catch |e| {
            for (traces.items[i + 1 ..]) |*rest| rest.deinit();
            return e;
        };
    }
    p.book.build() catch |e| {
        if (e == error.PlanAmbiguous) log.err("[plan] a sample's frame {d}/{d} is ambiguous: {t} at node {d}\n", .{ p.book.ambiguity.id.site, p.book.ambiguity.id.key, p.book.ambiguity.why, p.book.ambiguity.node });
        return e;
    };
    if (p.cache) |*c| learnKept(m, p, c);
    const l = p.book.layout();
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[plan] samples: {d} mixes, verdicts to {d} rows; scratch {d} KB, host ring {d} KB x {d} with them; recorded in {d} ms\n", .{ all.len, st.rows_max, l.g >> 10, l.h >> 10, l.replicas, ms });
}

/// The sample shapes earlier loads' misses kept, each recorded and learned into the book; one this load cannot run is
/// passed over by name.
fn learnKept(m: *Model, p: anytype, c: *const plan_book.Cache) void {
    const gpa = p.book.gpa;
    const kept_shapes = c.read(gpa) catch return;
    defer gpa.free(kept_shapes);
    const unbounded: plan.binder.Layout = .{ .g = std.math.maxInt(u64), .h = std.math.maxInt(u64), .alignment = p.book.alignment };
    var learned: usize = 0;
    var named: usize = 0;
    for (kept_shapes) |s| if (s.phase == shape_phase) {
        named += 1;
        const mix = mixOf(s) orelse {
            log.warn("[plan] a kept shape is no sample of this load ({d} {d} {d} {d})\n", .{ s.args[0], s.args[1], s.args[2], s.args[3] });
            continue;
        };
        var traces = recordMix(m, &p.sample, gpa, mix) catch |e| {
            log.warn("[plan] a kept sample ({t}, {d} rows) was not recorded ({t})\n", .{ mix.entry, mix.rows, e });
            continue;
        };
        defer traces.deinit(gpa);
        _ = p.book.learnAll(root(mix), traces.items, unbounded) catch |e| {
            traces.items.len = 0;
            log.warn("[plan] a kept sample ({t}, {d} rows) was not planned ({t})\n", .{ mix.entry, mix.rows, e });
            continue;
        };
        traces.items.len = 0;
        learned += 1;
    };
    if (named > 0) log.info("[plan] {d} of {d} samples kept by earlier misses planned\n", .{ learned, named });
}
