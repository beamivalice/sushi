//! Hy4-Flash on the static bill (D97 Everywhere): at load time, worst-case narrow envelopes (decode, verify, replay,
//! draft, joined, samples) and worst-case wide envelopes (prefill chunk + image encode) are dry-probed via
//! `plan_region.needWithGuard` to size one shared Metal arena `[host ring x 2 | scratch]` with explicit fragmentation
//! headroom. Both narrow and wide phases execute through the same D97 first-fit region allocator (`core.beginNarrow` /
//! `core.beginWide`).
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_config = @import("hy4_config.zig");
const fwd = @import("hy4_forward.zig");
const Model = hy4_model.Model;
const Readout = Model.Readout;
const KVQuantConfig = @import("../kv_quant.zig").KVQuantConfig;
const Drafter = @import("hy4_dfly.zig").Drafter;
const GemmScope = @import("sushi_exl3").linear.GemmScope;

const joined_rows_max: usize = @intCast(@import("sushi_exl3").project_rows.NAX_ROWS_MAX);

/// Maximum joined rows for a streamed Cooler load whose expert cache holds `slots` experts per layer:
/// requires at least `2 * top_k` slots so >= 2 concurrent rows fit in one cache working set without intra-tick
/// eviction, capped at `min(hw_max, slots / top_k)`. Tighter caches stay exclusive (1 row).
pub fn streamedRowsCap(experts: usize, top_k: usize, slots: usize, hw_max: usize) usize {
    if (experts == 0 or top_k == 0 or hw_max <= 1) return 1;
    const by_slots = slots / top_k;
    if (by_slots < 2) return 1;
    return @min(hw_max, by_slots);
}

pub fn joinedRows(config: *const @import("../model.zig").ModelConfig) usize {
    if (!@import("../cooler_side.zig").onEngine(config)) return 1;
    const hw_max: usize = @intCast(fwd.joinedRowsMax());
    if (config.expert_resident) return hw_max;
    const c = config.hy4flash orelse return 1;
    return streamedRowsCap(c.experts, c.top_k, config.expert_cache_slots, hw_max);
}

/// One shape a dry probe executes: `rows` rows after a state `n` rows deep.
pub const Probe = struct {
    rows: usize,
    readout: Readout,
    keep_hidden: bool,
    kv: KVQuantConfig,
    n: usize = 0,
    phase: Phase = .forward,
    drafter: ?*const Drafter = null,
    groups: []const u16 = &.{},
    ns: []const usize = &.{},
};

pub const Phase = enum {
    forward,
    draft,
    wide,
    joined,
};

pub const Encode = struct { tower: *const @import("hy4_vision.zig").Tower, max_patches: u64 };

pub const Plans = struct {
    gpa: std.mem.Allocator,
    alignment: u64 = plan.region.default_alignment,
    guard: u64 = plan.region.test_guard,
    reach: usize,
    joined: usize = 0,
    kv: KVQuantConfig,
    drafter: ?*const Drafter = null,
    sample: @import("hy4_sample_plan.zig").State = .{},
    narrow: plan.region.Need = .{},
    encode: plan.region.Need = .{},
    region: plan.region.Need = .{},
    region_rows: usize = 0,
    wide_cache: std.AutoHashMapUnmanaged(usize, plan.region.Need) = .empty,

    /// Combined arena layout across narrow envelopes (with narrow headroom) and wide prefill/encode envelopes (with wide headroom).
    pub fn layoutWith(p: *const Plans, wide_need: plan.region.Need) plan.region.Layout {
        const n = plan.region.withNarrowHeadroom(p.narrow, p.alignment);
        const w = plan.region.withWideHeadroom(wide_need.max(p.encode), p.alignment);
        return .{
            .g = @max(n.scratch, w.scratch),
            .h = @max(n.host, w.host),
            .alignment = p.alignment,
            .guard = p.guard,
        };
    }

    pub fn deinit(p: *Plans, gpa: std.mem.Allocator) void {
        p.sample.deinit();
        p.wide_cache.deinit(gpa);
        gpa.destroy(p);
    }
};

pub const Passes = struct {
    gpa: std.mem.Allocator,
    traces: []plan_trace.Trace,

    pub fn deinit(p: *Passes) void {
        for (p.traces) |*t| t.deinit();
        p.gpa.free(p.traces);
        p.* = undefined;
    }
};

pub const Ids = enum { zeros, spread };

pub fn draftProbe(d: *const Drafter, c: *const hy4_config.Config, kv: KVQuantConfig, n: usize) Probe {
    return .{ .rows = c.dfly.draft + 1, .readout = .all, .keep_hidden = true, .kv = kv, .n = n, .phase = .draft, .drafter = d };
}

fn recordPasses(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids: Ids) !Passes {
    var passes: std.ArrayList(plan_trace.Trace) = .empty;
    errdefer {
        for (passes.items) |*t| t.deinit();
        passes.deinit(gpa);
    }
    var driver: plan.Driver = .{ .gpa = gpa };
    defer driver.deinit();
    while (true) {
        var t = try recordPass(m, gpa, probe, reach, fill, ids, &driver);
        passes.append(gpa, t) catch |e| {
            t.deinit();
            return e;
        };
        if (!try driver.next()) break;
    }
    return .{ .gpa = gpa, .traces = try passes.toOwnedSlice(gpa) };
}

fn recordPass(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    if (probe.phase == .draft) return if (probe.groups.len > 0) recordDraftBatch(m, gpa, probe, reach, fill, ids_kind, driver) else recordDraft(m, gpa, probe, reach, fill, driver);
    if (probe.phase == .joined) return if (probe.groups.len > 0) recordJoinedGroups(m, gpa, probe, reach, fill, ids_kind, driver) else recordJoined(m, gpa, probe, reach, fill, ids_kind, driver);
    const own = m.lendOwn();
    defer own.end();
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = probe.keep_hidden;
    const kv_was = m.own.kv;
    defer m.own.kv = kv_was;
    m.own.kv = probe.kv;
    const held = m.logits_fixed;
    const turn = m.logits_turn;
    m.logits_fixed = .{ null, null };
    defer {
        m.freeLogitsBlocks();
        m.logits_fixed = held;
        m.logits_turn = turn;
    }
    m.reset();
    defer m.reset();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        try m.dryState(probe.n, reach);
        if (held[0] != null) try m.makeLogitsBlocks();
    }
    const ids = try gpa.alloc(u32, probe.rows);
    defer gpa.free(ids);
    spreadIds(&m.config, ids, ids_kind);
    const wide_ok = probe.phase == .wide;
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        const scope = if (wide_ok) try GemmScope.begin() else null;
        defer if (scope) |sc| sc.end();
        const lg = try m.forwardServed(ids, null, probe.readout);
        if (probe.readout != .none) _ = mlx.mlx_array_free(lg);
    }
    return rec.end() catch |e| {
        if (e == error.PlanEscape) {
            const at = rec.escape.?;
            log.err("[plan] an ask of {d} B outlives a forward of {d} rows at {d} (ordinal {d})\n", .{ rec.asks.items[at].bytes, probe.rows, probe.n, at });
        }
        return e;
    };
}

fn recordJoined(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    const max_rows = joined_rows_max;
    const n = probe.rows;
    if (n == 0 or n > max_rows) return error.Hy4RowsWidth;
    const own = m.lendOwn();
    defer own.end();
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = probe.keep_hidden;
    var reqs: [max_rows]hy4_model.Request = undefined;
    var made: usize = 0;
    defer for (reqs[0..made]) |*r| r.deinit();
    var segs: [max_rows]Model.Seg = undefined;
    var ids: [max_rows]u32 = undefined;
    m.reset();
    defer m.reset();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        for (0..n) |i| {
            reqs[i] = try hy4_model.Request.init(gpa, 0);
            made = i + 1;
            reqs[i].kv = probe.kv;
            const lent = m.lend(&reqs[i]);
            defer lent.end();
            const ni = if (probe.ns.len > 0) probe.ns[i] else probe.n;
            try m.dryState(ni, reach);
        }
    }
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        const scope = try GemmScope.begin();
        defer scope.end();
        spreadIds(&m.config, ids[0..n], ids_kind);
        for (0..n) |i| segs[i] = .{ .request = &reqs[i], .id = ids[i] };
        const joined = try m.extendRows(segs[0..n]);
        _ = mlx.mlx_array_free(joined);
    }
    return rec.end() catch |e| {
        if (e == error.PlanEscape) {
            const at = rec.escape.?;
            log.err("[plan] an ask of {d} B outlives a joined tick of {d} rows at {d} (ordinal {d})\n", .{ rec.asks.items[at].bytes, probe.rows, probe.n, at });
        }
        return e;
    };
}

fn recordJoinedGroups(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    const max_rows = joined_rows_max;
    const n = probe.groups.len;
    if (n == 0 or n > max_rows) return error.Hy4RowsWidth;
    const own = m.lendOwn();
    defer own.end();
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = probe.keep_hidden;
    const held = m.logits_fixed;
    const turn = m.logits_turn;
    m.logits_fixed = .{ null, null };
    defer {
        m.freeLogitsBlocks();
        m.logits_fixed = held;
        m.logits_turn = turn;
    }
    var reqs: [max_rows]hy4_model.Request = undefined;
    var made: usize = 0;
    defer for (reqs[0..made]) |*r| r.deinit();
    var ids: [max_rows][max_rows]u32 = undefined;
    var groups: [max_rows]Model.Group = undefined;
    m.reset();
    defer m.reset();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        for (0..n) |i| {
            reqs[i] = try hy4_model.Request.init(gpa, 0);
            made = i + 1;
            reqs[i].kv = probe.kv;
            const lent = m.lend(&reqs[i]);
            defer lent.end();
            const ni = if (probe.ns.len > 0) probe.ns[i] else probe.n;
            try m.dryState(ni, reach);
            try m.makeRequestLogits(&reqs[i]);
        }
        if (held[0] != null) try m.makeLogitsBlocks();
    }
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        const scope = try GemmScope.begin();
        defer scope.end();
        for (0..n) |i| {
            const w: usize = probe.groups[i];
            spreadIds(&m.config, ids[i][0..w], ids_kind);
            groups[i] = .{ .request = &reqs[i], .ids = ids[i][0..w] };
        }
        const joined = try m.extendGroups(groups[0..n]);
        var start: usize = 0;
        for (groups[0..n], 0..) |g, i| {
            const v = try hy4_model.writeGroupBlock(g.request, joined, @intCast(start), @intCast(start + probe.groups[i]), m.logits_rows, m.s);
            _ = mlx.mlx_array_free(v);
            start += probe.groups[i];
        }
        _ = mlx.mlx_array_free(joined);
    }
    return rec.end();
}

fn recordDraft(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, driver: *plan.Driver) !plan_trace.Trace {
    const d = probe.drafter orelse return error.Hy4DflyMissing;
    if (probe.n == 0) return error.Hy4DflyNoContext;
    const own = m.lendOwn();
    defer own.end();
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = true;
    const kv_was = m.own.kv;
    defer m.own.kv = kv_was;
    m.own.kv = probe.kv;
    m.reset();
    defer m.reset();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        try m.dryState(probe.n - 1, reach);
        _ = mlx.mlx_array_free(try m.forwardReadout(&.{0}, .none));
    }
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        var drafted = try d.draftLimit(m, 0, probe.rows - 1);
        drafted.deinit();
    }
    return rec.end();
}

fn recordDraftBatch(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    const d = probe.drafter orelse return error.Hy4DflyMissing;
    const n = probe.groups.len;
    if (n < 2 or n > Drafter.max_batch_slots) return error.Hy4RowsWidth;
    const own = m.lendOwn();
    defer own.end();
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = true;
    var reqs: [Drafter.max_batch_slots]hy4_model.Request = undefined;
    var made: usize = 0;
    defer for (reqs[0..made]) |*r| r.deinit();
    m.reset();
    defer m.reset();
    {
        var rec = plan.Recorder.begin(gpa);
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        for (0..n) |i| {
            const ni = if (probe.ns.len > 0) probe.ns[i] else probe.n;
            if (ni == 0) return error.Hy4DflyNoContext;
            reqs[i] = try hy4_model.Request.init(gpa, 0);
            made = i + 1;
            reqs[i].kv = probe.kv;
            const lent = m.lend(&reqs[i]);
            defer lent.end();
            try m.dryState(ni, reach);
            try m.dryHandover();
        }
    }
    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        const scope = try GemmScope.begin();
        defer scope.end();
        var req_ptrs: [Drafter.max_batch_slots]*hy4_model.Request = undefined;
        for (0..n) |i| req_ptrs[i] = &reqs[i];
        var t1s: [Drafter.max_batch_slots]u32 = undefined;
        spreadIds(&m.config, t1s[0..n], ids_kind);
        var out: [Drafter.max_batch_slots]Drafter.Draft = undefined;
        try d.draftBatch(m, req_ptrs[0..n], t1s[0..n], probe.rows, out[0..n]);
        for (out[0..n]) |*dr| dr.deinit();
    }
    return rec.end();
}

/// Records `probe` under both fills and zero vs spread ids to prove data never steers unannotated allocations.
pub fn recordChecked(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize) !Passes {
    var zeros = try recordPasses(m, gpa, probe, reach, .zeros, .zeros);
    errdefer zeros.deinit();
    var other = try recordPasses(m, gpa, probe, reach, .ramp, .spread);
    defer other.deinit();
    if (zeros.traces.len != other.traces.len) {
        log.err("[plan] a forward of {d} rows at {d} records {d} passes over one input and {d} over another: its data steers a choice no `plan.choose` declares\n", .{ probe.rows, probe.n, zeros.traces.len, other.traces.len });
        return error.Hy4PlanDataSteers;
    }
    for (zeros.traces, other.traces, 0..) |z, r, pass| if (plan_trace.firstDifference(z.view(), r.view())) |d| {
        const ask = if (d.what == .asks and d.at < z.asks.len) z.asks[d.at].bytes else 0;
        const other_ask = if (d.what == .asks and d.at < r.asks.len) r.asks[d.at].bytes else 0;
        log.err("[plan] a forward of {d} rows at {d} records two shapes over two inputs (pass {d}, {t} at {d}: {d} B against {d} B): its data steers a size no `plan.bound` declares\n", .{ probe.rows, probe.n, pass, d.what, d.at, ask, other_ask });
        return error.Hy4PlanDataSteers;
    };
    return zeros;
}

fn spreadIds(c: *const hy4_config.Config, ids: []u32, kind: Ids) void {
    switch (kind) {
        .zeros => @memset(ids, 0),
        .spread => for (ids, 0..) |*id, i| {
            var v: u32 = @intCast((i * 7919 + 3) % c.vocab);
            if (c.image_id != 0 and v == c.image_id) v = (v + 1) % c.vocab;
            id.* = v;
        },
    }
}

fn measureProbe(m: *Model, p: *Plans, probe: Probe) !void {
    var passes = try recordChecked(m, p.gpa, probe, p.reach);
    defer passes.deinit();
    for (passes.traces) |*tr| {
        p.narrow = p.narrow.max(try plan.region.needWithGuard(p.gpa, tr, p.alignment, p.guard));
    }
}

/// Sizes Hy4-Flash's D97 narrow arena envelopes from worst-case dry probes (no exact-trace book or cartesian sweeps).
pub fn record(m: *Model, gpa: std.mem.Allocator, reach: usize, drafter: ?*const Drafter, kv: KVQuantConfig, encode: ?Encode, joined: usize) !*Plans {
    const drafts = drafter != null;
    const t0 = @import("../load_clock.zig").now();
    try m.probeWide();
    const p = try gpa.create(Plans);
    errdefer gpa.destroy(p);
    p.* = .{
        .gpa = gpa,
        .reach = reach,
        .joined = if (joined > 1) @min(joined, joined_rows_max) else 0,
        .kv = kv,
        .drafter = drafter,
    };
    errdefer p.deinit(gpa);

    m.logits_rows = @intCast(@max(@as(usize, @intCast(fwd.verify_rows_max)), p.joined));
    try m.makeLogitsBlocks();
    try m.makeIdsBlock();

    const c = &m.config;
    var probes: usize = 0;

    // 1. Serial 1-row decode at shallow (past group=1) and deepest context (past window/threshold/split).
    const serial_pos = [_]usize{ 0, @min(1, reach -| 1), @min(c.window + 1, reach -| 1), reach -| 1 };
    for (serial_pos, 0..) |n, idx| {
        if (idx > 0 and n == serial_pos[idx - 1]) continue;
        try measureProbe(m, p, .{ .rows = 1, .readout = .last, .keep_hidden = drafts or m.keep_hidden, .kv = kv, .n = n });
        probes += 1;
        if (drafts) {
            try measureProbe(m, p, .{ .rows = 1, .readout = .all, .keep_hidden = true, .kv = kv, .n = n });
            probes += 1;
        }
    }

    // 2. Drafter verify blocks, rollback replay, single draft, and batched draft.
    const verify_max: usize = @min(@as(usize, @intCast(fwd.verify_rows_max)), reach);
    if (drafts) {
        if (verify_max >= 2) {
            for ([_]usize{ 2, verify_max }) |vr| {
                const n_deep = reach -| vr;
                try measureProbe(m, p, .{ .rows = vr, .readout = .all, .keep_hidden = true, .kv = kv, .n = n_deep });
                probes += 1;
                if (vr == verify_max) break;
            }
            if (verify_max > 2) {
                try measureProbe(m, p, .{ .rows = 2, .readout = .all, .keep_hidden = true, .kv = kv, .n = reach -| 2 });
                probes += 1;
            }
        }
        const replay_max = @min(verify_max -| 1, reach -| 1);
        if (replay_max >= 1) {
            try measureProbe(m, p, .{ .rows = replay_max, .readout = .none, .keep_hidden = true, .kv = kv, .n = reach -| replay_max });
            probes += 1;
        }
        if (drafter) |d| if (reach > c.dfly.draft + 1) {
            const draft_deep = reach -| (c.dfly.draft + 1);
            try measureProbe(m, p, draftProbe(d, c, kv, draft_deep));
            probes += 1;
            if (reach > 2) {
                try measureProbe(m, p, draftProbe(d, c, kv, 1));
                probes += 1;
            }
            if (p.joined > 1) {
                const max_slots = @min(p.joined, Drafter.max_batch_slots);
                for ([_]usize{ 2, max_slots }) |slots| {
                    const n_use = groupDraftWidth(@as(usize, c.dfly.draft), p.joined, slots);
                    if (n_use == 0 or slots * (n_use + 1) > 64) continue;
                    var widths: [Drafter.max_batch_slots]u16 = @splat(@intCast(n_use));
                    try measureProbe(m, p, .{ .rows = n_use, .readout = .all, .keep_hidden = true, .kv = kv, .n = draft_deep, .phase = .draft, .drafter = d, .groups = widths[0..slots] });
                    probes += 1;
                    if (slots == 2) break;
                }
                if (max_slots > 2) {
                    const n_use = groupDraftWidth(@as(usize, c.dfly.draft), p.joined, max_slots);
                    if (n_use > 0 and max_slots * (n_use + 1) <= 64) {
                        var widths: [Drafter.max_batch_slots]u16 = @splat(@intCast(n_use));
                        try measureProbe(m, p, .{ .rows = n_use, .readout = .all, .keep_hidden = true, .kv = kv, .n = draft_deep, .phase = .draft, .drafter = d, .groups = widths[0..max_slots] });
                        probes += 1;
                    }
                }
            }
        };
    }

    // 3. Joined decode and joined-groups verify envelopes.
    if (p.joined > 1) {
        const deep = reach -| 1;
        try measureProbe(m, p, .{ .rows = p.joined, .readout = .all, .keep_hidden = m.keep_hidden, .kv = kv, .n = deep, .phase = .joined });
        probes += 1;
        if (drafter != null) {
            const slots = @min(p.joined, Drafter.max_batch_slots);
            var templates: [20][joined_rows_max]u16 = undefined;
            const n_tpl = groupWidthTemplates(@as(usize, c.dfly.draft), p.joined, slots, &templates);
            if (n_tpl > 0) {
                var best_sum: usize = 0;
                var best_ti: usize = 0;
                for (0..n_tpl) |ti| {
                    var sum: usize = 0;
                    for (templates[ti][0..slots]) |w| sum += w;
                    if (sum >= best_sum) {
                        best_sum = sum;
                        best_ti = ti;
                    }
                }
                const max_w = templates[best_ti][0];
                try measureProbe(m, p, .{ .rows = best_sum, .readout = .all, .keep_hidden = m.keep_hidden, .kv = kv, .n = reach -| max_w, .phase = .joined, .groups = templates[best_ti][0..slots] });
                probes += 1;
            }
        }
    }
    // 4. Image encode envelope.
    if (encode) |e| {
        p.encode = try @import("hy4_encode_plan.zig").need(e.tower, gpa, e.max_patches, p.alignment, p.guard);
    }

    const narrow_layout = plan.region.withNarrowHeadroom(p.narrow, p.alignment);
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[plan] {d} worst-case probes to {d}: raw narrow scratch {d} KB (with headroom {d} KB), host ring {d} KB x 2; recorded in {d} ms\n", .{ probes, reach, p.narrow.scratch >> 10, narrow_layout.scratch >> 10, narrow_layout.host >> 10, ms });
    return p;
}

pub fn build(m: *Model, gpa: std.mem.Allocator, reach: usize, drafter: ?*const Drafter, kv: KVQuantConfig, wide_rows: usize) !*Plans {
    const p = try record(m, gpa, reach, drafter, kv, null, 0);
    errdefer p.deinit(gpa);
    try makeArena(m, p, wide_rows);
    return p;
}

fn measureWide(m: *Model, p: *const Plans, most: *plan.region.Need, rows: usize, n: usize, readout: Readout) !void {
    if (rows == 0) return;
    var passes = try recordChecked(m, p.gpa, .{ .rows = rows, .readout = readout, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n, .phase = .wide }, p.reach);
    defer passes.deinit();
    for (passes.traces) |*t| {
        most.* = most.*.max(try plan.region.needWithGuard(p.gpa, t, p.alignment, p.guard));
    }
}

fn straddleDepth(c: *const hy4_config.Config, reach: usize, rows: usize) ?usize {
    const thr: usize = @as(usize, c.indexer.group_topk) * c.indexer.group;
    const dense = @min(@as(usize, @intCast(fwd.verify_rows_max)), thr, rows -| 1);
    return if (thr - dense + rows > reach) null else thr - dense;
}

/// Measures worst-case wide chunk region demand at `rows` width (cached by `rows` on `p.wide_cache`).
pub fn regionAt(m: *Model, p: *Plans, rows: usize) !plan.region.Need {
    if (p.wide_cache.get(rows)) |cached| return cached;
    const t0 = @import("../load_clock.zig").now();
    const span = @min(@as(usize, @import("../prefill_plan.zig").final_span_rows), rows);
    const widths = [_]usize{ rows, span, rows -| 1 };
    const n_widths: usize = if (rows > 1 and rows - 1 > span) 3 else if (span != rows and span > 0) 2 else 1;
    var most: plan.region.Need = .{};
    for (widths[0..n_widths]) |r| {
        try measureWide(m, p, &most, r, p.reach -| r, .last);
        try measureWide(m, p, &most, r, 0, .none);
        if (straddleDepth(&m.config, p.reach, r)) |n| {
            try measureWide(m, p, &most, r, n, .last);
        }
    }
    try p.wide_cache.put(p.gpa, rows, most);
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[plan] {d}-row prefill region: raw scratch {d} KB, host {d} KB; measured in {d} ms\n", .{ rows, most.scratch >> 10, most.host >> 10, ms });
    return most;
}

pub fn makeArena(m: *Model, p: *Plans, wide_rows: usize) !void {
    const wide_need: plan.region.Need = if (wide_rows > 0) try regionAt(m, p, wide_rows) else .{};
    p.region = wide_need;
    p.region_rows = wide_rows;
    const layout = p.layoutWith(wide_need);
    try core.arenaFor(layout);
    log.info("[plan] arena ready: scratch {d} KB, host ring {d} KB x {d} (wide chunk {d})\n", .{ layout.g >> 10, layout.h >> 10, layout.replicas, wide_rows });
}

fn isNarrowForward(m: *const Model, p: *const Plans, ids: []const u32, vision: ?Model.VisionRows, req_: *const hy4_model.Request) bool {
    if (m.splicesImage(ids, vision) or !std.meta.eql(req_.kv, p.kv)) return false;
    const verify_max: usize = if (p.drafter != null) @intCast(fwd.verify_rows_max) else 1;
    return ids.len >= 1 and ids.len <= verify_max;
}

/// Runs a forward on the shared arena: narrow decode/verify/replay uses `.narrow`; wider prefill chunks, image
/// splices, or alternate KV widths use `.wide`.
pub fn run(m: *Model, ids: []const u32, vision: ?Model.VisionRows, readout: Readout) !mlx.mlx_array {
    const p = m.plans orelse return m.forwardServed(ids, vision, readout);
    const req_ = m.reqConst();
    if (!isNarrowForward(m, p, ids, vision, req_)) return wide(m, p, ids, vision, readout);
    try core.beginNarrow();
    const logits = m.forwardServed(ids, vision, readout) catch |e| {
        core.abortNarrow();
        return e;
    };
    core.endNarrow() catch |e| {
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

fn wide(m: *Model, p: *const Plans, ids: []const u32, vision: ?Model.VisionRows, readout: Readout) !mlx.mlx_array {
    const at = m.reqConst().pos;
    try core.beginWide();
    const scope = GemmScope.begin() catch |e| {
        core.abortWide();
        return e;
    };
    const logits = m.forwardServed(ids, vision, readout) catch |e| {
        scope.end();
        core.abortWide();
        if (core.refused_wide) |r| {
            log.err("[plan] a forward of {d} rows at {d} asked {d} KB of the region's {s}, past the {d} KB the load holds (measured at {d}-row chunks)\n", .{ ids.len, at, r.asked >> 10, if (r.host) "host ring" else "scratch", r.room >> 10, p.region_rows });
            return error.WideFull;
        }
        return e;
    };
    scope.end();
    core.endWide() catch |e| {
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

pub fn runDraft(m: *Model, d: *const Drafter, t1: u32) !Drafter.Draft {
    return runDraftLimit(m, d, t1, null);
}

/// `runDraft` capped to `max_drafts` slots (clamped to `m.config.dfly.draft`).
pub fn runDraftLimit(m: *Model, d: *const Drafter, t1: u32, max_drafts: ?usize) !Drafter.Draft {
    const cap: usize = m.config.dfly.draft;
    const n_draft: usize = @min(max_drafts orelse cap, cap);
    const p = m.plans orelse return d.draftLimit(m, t1, n_draft);
    if (!std.meta.eql(m.reqConst().kv, p.kv)) return d.draftLimit(m, t1, n_draft);
    try core.beginNarrow();
    var drafted = d.draftLimit(m, t1, n_draft) catch |e| {
        core.abortNarrow();
        return e;
    };
    core.endNarrow() catch |e| {
        drafted.deinit();
        return e;
    };
    return drafted;
}

pub fn runDraftBatch(
    m: *Model,
    d: *const Drafter,
    reqs: []const *hy4_model.Request,
    t1s: []const u32,
    max_drafts: usize,
    out_drafts: []Drafter.Draft,
) !void {
    const n = reqs.len;
    if (n == 0 or t1s.len != n or out_drafts.len != n) return error.Hy4RowsWidth;
    const n_use = @min(max_drafts, @as(usize, m.config.dfly.draft));
    if (n == 1 or n * (n_use + 1) > 64) {
        var made: usize = 0;
        errdefer for (out_drafts[0..made]) |*b| b.deinit();
        for (reqs, t1s, out_drafts) |r, t1, *dst| {
            const lent = m.lend(r);
            defer lent.end();
            dst.* = try runDraft(m, d, t1);
            made += 1;
        }
        return;
    }
    const p = m.plans orelse {
        const scope = try GemmScope.begin();
        defer scope.end();
        return d.draftBatch(m, reqs, t1s, n_use, out_drafts);
    };
    for (reqs) |r| if (!std.meta.eql(r.kv, p.kv)) {
        var made: usize = 0;
        errdefer for (out_drafts[0..made]) |*b| b.deinit();
        for (reqs, t1s, out_drafts) |rq, t1, *dst| {
            const lent = m.lend(rq);
            defer lent.end();
            dst.* = try runDraft(m, d, t1);
            made += 1;
        }
        return;
    };
    try core.beginNarrow();
    const scope = GemmScope.begin() catch |e| {
        core.abortNarrow();
        return e;
    };
    d.draftBatch(m, reqs, t1s, n_use, out_drafts) catch |e| {
        scope.end();
        core.abortNarrow();
        return e;
    };
    scope.end();
    core.endNarrow() catch |e| {
        for (out_drafts) |*b| b.deinit();
        return e;
    };
}

pub fn beginJoined(m: *Model, segs: []const Model.Seg) !void {
    const p = m.plans orelse return error.Hy4ModelMissing;
    for (segs) |sg| if (!std.meta.eql(sg.request.kv, p.kv)) {
        log.err("[plan] a joined tick's request holds a kv the load's plans do not: refused, never served in the wide region\n", .{});
        return error.Hy4JoinedKvUnplanned;
    };
    if (segs.len > p.joined) {
        log.err("[plan] a joined tick of {d} rows: the load plans joined ticks up to {d}\n", .{ segs.len, p.joined });
        return error.Hy4RowsWidth;
    }
    try core.beginNarrow();
}

pub fn endJoined(_: *Model, segs: []const Model.Seg) !void {
    core.endNarrow() catch |e| {
        for (segs) |sg| sg.request.reset();
        return e;
    };
}

pub fn groupWidthTemplates(drafts: usize, joined: usize, slots: usize, out: *[20][joined_rows_max]u16) usize {
    const max_g = joined_rows_max;
    if (slots < 2 or slots > max_g or slots >= joined) return 0;
    const uniform_max = fwd.groupRows(drafts, joined, slots);
    if (uniform_max < 2) return 0;
    const max_w = uniform_max;
    var count: usize = 0;
    const Helper = struct {
        fn add(dst: *[20][max_g]u16, cnt: *usize, sl: usize, lim: usize, cap_w: usize, vals: []const u16) void {
            if (cnt.* >= dst.len or vals.len != sl) return;
            var sum: usize = 0;
            var any_multi = false;
            for (vals, 0..) |v, i| {
                if (v == 0 or v > cap_w) return;
                if (i > 0 and v > vals[i - 1]) return;
                if (v >= 2) any_multi = true;
                sum += v;
            }
            if (!any_multi or sum > lim) return;
            for (0..cnt.*) |j| if (std.mem.eql(u16, dst[j][0..sl], vals)) return;
            @memcpy(dst[cnt.*][0..sl], vals);
            cnt.* += 1;
        }
        fn addFill(dst: *[20][max_g]u16, cnt: *usize, sl: usize, lim: usize, cap_w: usize, k: usize, hi: u16) void {
            if (k == 0 or k > sl) return;
            var row: [max_g]u16 = @splat(1);
            @memset(row[0..k], hi);
            add(dst, cnt, sl, lim, cap_w, row[0..sl]);
        }
    };
    for (2..uniform_max + 1) |w| {
        var row: [max_g]u16 = @splat(@intCast(w));
        Helper.add(out, &count, slots, joined, max_w, row[0..slots]);
    }
    switch (slots) {
        2 => {
            for ([_]u16{ 2, 4, 6 }) |w0| Helper.add(out, &count, 2, joined, max_w, &.{ w0, 1 });
            Helper.add(out, &count, 2, joined, max_w, &.{ 6, 2 });
        },
        3 => {
            for ([_]u16{ 2, 4, 5 }) |w| Helper.add(out, &count, 3, joined, max_w, &.{ w, w, 1 });
            for ([_]u16{ 2, 4, 5 }) |w| Helper.add(out, &count, 3, joined, max_w, &.{ w, 1, 1 });
        },
        4 => {
            for ([_]u16{ 2, 3, 4 }) |w| Helper.add(out, &count, 4, joined, max_w, &.{ w, w, 1, 1 });
            Helper.add(out, &count, 4, joined, max_w, &.{ 4, 4, 4, 1 });
            Helper.add(out, &count, 4, joined, max_w, &.{ 3, 3, 3, 1 });
            for ([_]u16{ 2, 3, 4 }) |w| Helper.add(out, &count, 4, joined, max_w, &.{ w, 1, 1, 1 });
        },
        else => {
            for ([_]usize{ 1, 2, 3, 4 }) |k| if (k < slots) Helper.addFill(out, &count, slots, joined, max_w, k, 3);
            for ([_]usize{ 1, 2, slots / 2, slots -| 2, slots -| 1 }) |k| if (k >= 1 and k < slots) Helper.addFill(out, &count, slots, joined, max_w, k, 2);
        },
    }
    return count;
}

pub fn groupDraftWidth(drafts: usize, joined: usize, slots: usize) usize {
    if (drafts == 0 or slots < 2 or slots > Drafter.max_batch_slots or slots >= joined) return 0;
    var templates: [20][joined_rows_max]u16 = undefined;
    const n_tpl = groupWidthTemplates(drafts, joined, slots, &templates);
    var max_w: usize = 0;
    for (templates[0..n_tpl]) |tpl| {
        if (tpl[0] > max_w) max_w = tpl[0];
    }
    if (max_w < 2) return 0;
    const by_batch = 64 / slots;
    if (by_batch < 2) return 0;
    return @min(drafts, @min(max_w - 1, by_batch - 1));
}

pub fn beginJoinedGroups(_: *Model) !void {
    try core.beginNarrow();
}

pub fn endJoinedGroups(_: *Model, groups: []const Model.Group) !void {
    core.endNarrow() catch |e| {
        for (groups) |g| g.request.reset();
        return e;
    };
}
