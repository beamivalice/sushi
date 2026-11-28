//! DeepSeek-V4.1 on the static bill (D97 Everywhere): measures worst-case narrow envelopes (decode, verify,
//! finish_verify, DSpark draft, joined rows, samples) and worst-case wide chunks + image encode to size one
//! shared Metal arena `[host ring x 2 | scratch | optional tail alias]`, running both narrow and wide forwards
//! through `core.beginNarrow` / `core.beginWide`.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const model = @import("../model.zig");
const cfg_mod = @import("dsv41_config.zig");
const dsv41_model = @import("dsv41_model.zig");
const store_mod = @import("dsv41_store.zig");
const dsv41_sched = @import("dsv41_sched.zig");
const expert_stream = @import("../expert_stream.zig");
const exl3 = @import("sushi_exl3");
const GemmScope = exl3.linear.GemmScope;

const Model = dsv41_model.Model;
const Want = dsv41_model.Want;
const Heads = dsv41_model.Heads;

pub const Phase = enum {
    forward,
    joined,
    draft,
    draft_batch,
    finish_verify,
    finish_verify_joined,
    wide,
};

pub const Probe = struct {
    phase: Phase = .forward,
    rows: usize = 1,
    n: usize = 0,
    want: Want = .last,
    verify: bool = false,
    has_img: bool = false,
    ced_skip: bool = false,
    route_spec: bool = true,
    lazy_head: bool = false,
    draft_heads: Heads = .conf,
    draft_short: bool = false,
    keep: usize = 0,
    groups: []const u32 = &.{},
    ns: []const usize = &.{},
    accepteds: []const usize = &.{},
};

pub fn joinedRows(config: *const model.ModelConfig) usize {
    return dsv41_sched.coolerRows(config).rows;
}

pub const SampleState = @import("dsv41_sample_plan.zig").State;

pub const Plans = struct {
    gpa: std.mem.Allocator,
    alignment: u64 = plan.region.default_alignment,
    guard: u64 = plan.region.test_guard,
    reach: usize,
    joined: usize = 0,
    narrow: plan.region.Need = .{},
    encode: plan.region.Need = .{},
    region: plan.region.Need = .{},
    region_rows: usize = 0,
    wide_cache: std.AutoHashMapUnmanaged(usize, plan.region.Need) = .empty,
    sample: SampleState = .{},
    has_arena: bool = false,
    alias_engine: ?*expert_stream.Engine = null,
    keep_g: ?u64 = null,
    alias_bytes: u64 = 0,

    pub fn deinit(p: *Plans) void {
        if (p.alias_engine) |eng| {
            eng.releaseAliasedLayers();
            p.alias_engine = null;
        }
        if (p.has_arena) {
            core.arenaRelease() catch {};
            p.has_arena = false;
        }
        p.sample.deinit();
        p.wide_cache.deinit(p.gpa);
    }

    pub fn layout(p: *const Plans) plan.region.Layout {
        return p.layoutWith(p.region);
    }

    pub fn layoutWith(p: *const Plans, wide_need: plan.region.Need) plan.region.Layout {
        const n = plan.region.withNarrowHeadroom(p.narrow, p.alignment);
        const w = plan.region.withWideHeadroom(wide_need.max(p.encode), p.alignment);
        var l: plan.region.Layout = .{
            .g = @max(n.scratch, w.scratch),
            .h = @max(n.host, w.host),
            .replicas = 2,
            .alignment = p.alignment,
            .guard = p.guard,
        };
        if (p.keep_g) |keep| {
            l.keep = keep;
            l.g = @max(l.g, keep + p.alias_bytes);
        }
        return l;
    }

    pub fn acquireAliasLease(p: *const Plans) !void {
        if (p.alias_engine) |eng| try eng.acquireAliasLease();
    }

    pub fn releaseAliasLease(p: *const Plans) !void {
        if (p.alias_engine) |eng| try eng.releaseAliasLease();
    }
};

pub const Ids = enum { zeros, spread };

pub const Passes = struct {
    gpa: std.mem.Allocator,
    traces: []plan_trace.Trace,

    pub fn deinit(self: *Passes) void {
        for (self.traces) |*t| t.deinit();
        self.gpa.free(self.traces);
        self.* = undefined;
    }
};

fn spreadIds(c: *const cfg_mod.Config, ids: []u32, kind: Ids, has_img: bool) void {
    const vocab = @max(c.vocab, 2);
    if (has_img and c.image_token_id != 0 and ids.len > 1) {
        ids[0] = 0;
        for (ids[1..]) |*t| t.* = c.image_token_id;
        return;
    }
    switch (kind) {
        .zeros => @memset(ids, 0),
        .spread => {
            for (ids, 0..) |*t, i| {
                var tok: u32 = @intCast((i * 7 + 3) % vocab);
                if (c.image_token_id != 0 and tok == c.image_token_id) tok = (tok + 1) % vocab;
                t.* = tok;
            }
        },
    }
}

fn dryPend(m: *const Model, st: *dsv41_model.State, n: usize) !void {
    const s = dsv41_model.stateStream(m);
    for (st.layers, 0..) |*ls, li| {
        const r = m.layers[li].ratio;
        if (ls.comp) |*rows| rows.used = if (r > 0) @min(rows.cap, n / r) else 0;
        if (ls.idx_k) |*rows| rows.used = if (r > 0) @min(rows.cap, n / r) else 0;
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = .{ .ctx = null };
        ls.pend_gate = .{ .ctx = null };
        if (ls.comp != null and r > 1 and (n % r) > 0) {
            const rem = n % r;
            if (ls.pend_kv_buf) |b| ls.pend_kv = try store_mod.headRows(b, rem, s);
            if (ls.pend_gate_buf) |b| ls.pend_gate = try store_mod.headRows(b, rem, s);
        }
    }
}

/// Seeds `m`'s live state at token position `n` reserved to `reach` tokens without executing prior forwards.
/// Must be called inside a `.dry` recorder scope so fixed Cooler owners live on the dry recorder, never the pool.
pub fn dryState(m: *Model, n: usize, reach: usize) !void {
    const st = try dsv41_model.ensureState(m);
    try store_mod.reserveState(st, &m.cfg, m.act, m.dspark != null, m.onEngine(), m.serves_rows, reach, dsv41_model.stateStream(m));
    st.n = n;
    try dryPend(m, st, n);
    if (m.hash != null) {
        try st.hist.resize(m.gpa, n);
        @memset(st.hist.items, 0);
    }
}

fn ensureSpecFixed(m: *Model) !void {
    const s = dsv41_model.stateStream(m);
    if (m.cfg.trunk_dec) |dec| {
        try exl3.kernels.prepareGemmNax(dec, .pack, s);
        try exl3.kernels.prepareGemmNax(dec, .tile_major, s);
    }
    if (m.cfg.expert_dec) |dec| {
        try exl3.kernels.prepareGemmNax(dec, .pack, s);
        try exl3.kernels.prepareGemmNax(dec, .tile_major, s);
        try exl3.rate_groups.prepareSharedCaches(s);
    }
    if (!m.onEngine()) return;
    switch (m.experts) {
        .hook => |hk| if (hk.fix_spec) |fix| try fix(hk.ctx),
        .resident => {},
    }
}

fn freeJoinedViewsAndFree(joined: mlx.mlx_array, reqs: []dsv41_model.Request, segs: []const dsv41_model.Seg, m: *Model) !void {
    defer _ = mlx.mlx_array_free(joined);
    const s = dsv41_model.stateStream(m);
    var start: usize = 0;
    for (segs, 0..) |sg, i| {
        const v = try store_mod.writeGroupBlock(&reqs[i].logits_fixed, &reqs[i].logits_turn, joined, @intCast(start), @intCast(start + sg.ids.len), m.logits_rows, s);
        _ = mlx.mlx_array_free(v);
        start += sg.ids.len;
    }
}

pub fn recordPasses(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids) !Passes {
    var driver: plan.Driver = .{ .gpa = gpa };
    defer driver.deinit();
    var traces: std.ArrayList(plan_trace.Trace) = .empty;
    errdefer {
        for (traces.items) |*t| t.deinit();
        traces.deinit(gpa);
    }
    while (true) {
        const tr = try recordOnePass(m, gpa, probe, reach, fill, ids_kind, &driver);
        try traces.append(gpa, tr);
        if (!try driver.next()) break;
    }
    return .{ .gpa = gpa, .traces = try traces.toOwnedSlice(gpa) };
}

fn recordOnePass(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    try ensureSpecFixed(m);
    const own = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, own);
    const lent = dsv41_model.lendOwn(m);
    defer lent.end();

    const spec_was = m.route_spec;
    defer m.route_spec = spec_was;
    m.route_spec = probe.route_spec;

    const serves_was = m.serves_rows;
    defer m.serves_rows = serves_was;
    if (probe.phase == .joined or probe.phase == .draft_batch or probe.phase == .finish_verify_joined) {
        m.serves_rows = true;
    }

    const held_logits = m.logits_fixed;
    const held_turn = m.logits_turn;
    const held_ids = m.ids_fixed;
    const held_pf = m.pass_fixed;
    m.logits_fixed = .{ null, null };
    m.ids_fixed = null;
    m.pass_fixed = null;
    defer {
        m.freeLogitsBlocks();
        if (m.ids_fixed) |a| _ = mlx.mlx_array_free(a);
        m.freePassFixed();
        m.logits_fixed = held_logits;
        m.logits_turn = held_turn;
        m.ids_fixed = held_ids;
        m.pass_fixed = held_pf;
    }

    var reqs: [16]dsv41_model.Request = undefined;
    var req_made: usize = 0;
    defer for (reqs[0..req_made]) |*r| r.deinit();

    var seg_ids: [16][cfg_mod.ring_slack]u32 = undefined;
    var segs: [16]dsv41_model.Seg = undefined;
    var rptrs: [16]*dsv41_model.Request = undefined;
    var t1s: [16]u32 = undefined;
    var accepteds_buf: [16]usize = undefined;

    const ids_buf = try gpa.alloc(u32, @max(probe.rows, 1));
    defer gpa.free(ids_buf);
    spreadIds(&m.cfg, ids_buf[0..probe.rows], ids_kind, probe.has_img);

    dsv41_model.resetState(m);
    defer dsv41_model.resetState(m);

    {
        var prep_rec = plan.Recorder.begin(gpa);
        defer prep_rec.deinit();
        var prep_driver: plan.Driver = .{ .gpa = gpa };
        defer prep_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &prep_driver } }, &prep_rec);
        defer core.endRecording() catch {};

        try m.makeLogitsBlocks();
        try m.makeIdsBlock();
        // Under the unified D97 arena (`beginNarrow .. endNarrow`), every per-pass scratch buffer is freed before
        // `endNarrow()`, so `PassFixed` copies (~111 `sushi_ops_copy_into` dispatches/forward) are redundant.

        switch (probe.phase) {
            .forward, .wide, .draft => try dryState(m, probe.n, reach),
            .finish_verify => {
                try dryState(m, probe.n, reach);
                const vlg = try dsv41_model.verify(m, ids_buf[0..probe.rows]);
                _ = mlx.mlx_array_free(vlg);
            },
            .joined, .draft_batch, .finish_verify_joined => {
                for (0..probe.groups.len) |i| {
                    reqs[i] = try dsv41_model.Request.init(gpa, 0);
                    req_made = i + 1;
                    const gl = dsv41_model.lend(m, &reqs[i]);
                    defer gl.end();
                    const ni = if (i < probe.ns.len) probe.ns[i] else probe.n;
                    try dryState(m, ni, reach);
                    try m.makeRequestLogits(&reqs[i]);
                    rptrs[i] = &reqs[i];
                    const w: usize = probe.groups[i];
                    spreadIds(&m.cfg, seg_ids[i][0..w], ids_kind, false);
                    segs[i] = .{ .request = &reqs[i], .ids = seg_ids[i][0..w] };
                    t1s[i] = seg_ids[i][0];
                    if ((probe.phase == .joined and w > 1) or probe.phase == .finish_verify_joined) {
                        if (w > 1) try dsv41_model.beginVerifyReq(m, &reqs[i], w);
                        accepteds_buf[i] = if (i < probe.accepteds.len) probe.accepteds[i] else 0;
                    }
                }
                if (probe.phase == .finish_verify_joined) {
                    const armed = m.arm(.rows);
                    defer armed.end();
                    const jlg = try dsv41_model.extendRows(m, segs[0..probe.groups.len]);
                    try freeJoinedViewsAndFree(jlg, reqs[0..probe.groups.len], segs[0..probe.groups.len], m);
                }
            },
        }
    }

    var rec = plan.Recorder.begin(gpa);
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        const scope: ?GemmScope = if (probe.phase == .wide or probe.phase == .joined or probe.phase == .draft_batch or (probe.phase == .forward and probe.rows > 1)) try GemmScope.begin() else null;
        defer if (scope) |sc| sc.end();

        switch (probe.phase) {
            .forward, .wide => {
                var vis_arr = mlx.mlx_array_new();
                defer if (vis_arr.ctx != null) {
                    _ = mlx.mlx_array_free(vis_arr);
                };
                const vr: ?dsv41_model.VisionRows = if (probe.has_img) blk: {
                    try mlx.check(mlx.mlx_zeros(&vis_arr, &[_]c_int{ @intCast(probe.rows), @intCast(m.cfg.dim) }, 2, m.act, dsv41_model.stateStream(m)));
                    break :blk .{ .rows = vis_arr, .offset = 0 };
                } else null;

                if (probe.ced_skip) {
                    errdefer dsv41_model.pipeDrain(m) catch {};
                    try dsv41_model.extendState(m, ids_buf[0..probe.rows], dsv41_model.SPAN, vr, &.{});
                    try dsv41_model.pipeDrain(m);
                } else if (probe.verify) {
                    const lg = try dsv41_model.verify(m, ids_buf[0..probe.rows]);
                    _ = mlx.mlx_array_free(lg);
                } else {
                    const fwd_kind: Model.Forward = if (probe.phase == .wide or probe.rows > exl3.kernels.DECODE_ROWS_MAX) .prefill else .step;
                    const armed = m.arm(fwd_kind);
                    defer armed.end();
                    armed.lazyHead(probe.lazy_head);
                    const lg = try dsv41_model.extendVision(m, ids_buf[0..probe.rows], probe.want, dsv41_model.SPAN, vr);
                    _ = mlx.mlx_array_free(lg);
                }
            },
            .draft => {
                var d = try dsv41_model.draftMeasured(m, ids_buf[0], probe.draft_heads, @intCast(probe.rows), null);
                d.deinit();
            },
            .finish_verify => {
                try dsv41_model.finishVerify(m, probe.keep - 1);
            },
            .joined => {
                const n = probe.groups.len;
                const armed = m.arm(.rows);
                defer armed.end();
                armed.lazyHead(probe.lazy_head);
                const joined = try dsv41_model.extendRows(m, segs[0..n]);
                try freeJoinedViewsAndFree(joined, reqs[0..n], segs[0..n], m);
            },
            .draft_batch => {
                const n = probe.groups.len;
                var drafts: [16]dsv41_model.Draft = undefined;
                try dsv41_model.draftBatch(m, rptrs[0..n], t1s[0..n], probe.groups[0..n], probe.draft_heads, drafts[0..n]);
                for (drafts[0..n]) |*d| d.deinit();
            },
            .finish_verify_joined => {
                const n = probe.groups.len;
                try dsv41_model.finishVerifyJoined(m, rptrs[0..n], accepteds_buf[0..n]);
            },
        }
    }
    return rec.end() catch |e| {
        if (e == error.PlanEscape) {
            const at = rec.escape.?;
            log.err("[dsv41-plan] PlanEscape: {s} rows={d} n={d} ord={d} bytes={d}\n", .{
                @tagName(probe.phase),
                probe.rows,
                probe.n,
                at,
                rec.asks.items[at].bytes,
            });
        }
        return e;
    };
}

/// Records `probe` under `.zeros` and `.ramp` fills and verifies identical trace shape, ask sizes, dispatch levels,
/// and wait points across both fills (`D2`).
pub fn recordChecked(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize) !Passes {
    var z = try recordPasses(m, gpa, probe, reach, .zeros, .zeros);
    errdefer z.deinit();
    var r = try recordPasses(m, gpa, probe, reach, .ramp, .spread);
    defer r.deinit();
    if (z.traces.len != r.traces.len) return error.Dsv41PlanFillMismatch;
    for (z.traces, r.traces, 0..) |zt, rt, pass| {
        if (plan_trace.firstDifference(zt.view(), rt.view())) |diff| {
            log.err("[dsv41-plan] two-fill mismatch at {s} rows={d} n={d}: pass {d} {t} at {d}\n", .{
                @tagName(probe.phase),
                probe.rows,
                probe.n,
                pass,
                diff.what,
                diff.at,
            });
            return error.Dsv41PlanFillMismatch;
        }
    }
    return z;
}

fn maxRatio(c: *const cfg_mod.Config) usize {
    var mr: usize = 1;
    for (c.ratios[0..c.layers]) |r| mr = @max(mr, r);
    return mr;
}

fn pendPos(max_p0: usize, max_ratio: usize) usize {
    if (max_ratio <= 1 or max_p0 < max_ratio - 1) return max_p0;
    const rem = max_p0 % max_ratio;
    if (rem == max_ratio - 1) return max_p0;
    return max_p0 - (rem + 1);
}

fn measureProbe(m: *Model, p: *Plans, probe: Probe) !void {
    var passes = try recordChecked(m, p.gpa, probe, p.reach);
    defer passes.deinit();
    for (passes.traces) |*tr| {
        p.narrow = p.narrow.max(try plan.region.needWithGuard(p.gpa, tr, p.alignment, p.guard));
    }
}

/// Sizes DeepSeek-V4.1's D97 narrow arena envelopes from worst-case dry probes (no exact-trace book or cartesian sweeps).
pub fn record(m: *Model, gpa: std.mem.Allocator, reach: usize, chunk_max: usize, joined: usize) !*Plans {
    const t0 = @import("../load_clock.zig").now();
    try ensureSpecFixed(m);
    const p = try gpa.create(Plans);
    errdefer gpa.destroy(p);
    p.* = .{
        .gpa = gpa,
        .reach = reach,
        .joined = if (joined > 1) @min(joined, 16) else 0,
    };
    errdefer p.deinit();

    const max_r = maxRatio(&m.cfg);
    const has_vision = m.cfg.image_token_id != 0 and m.layers[0].gate_bias_vl.ctx != null;
    const spec_off_differs = m.route_spec and m.onEngine();
    var probes: usize = 0;

    // 1. Decode step (L = 1) and narrow forwards (2, route_spec_rows, DECODE_ROWS_MAX).
    const max_narrow = @min(@as(usize, exl3.kernels.DECODE_ROWS_MAX), reach);
    if (max_narrow > 0) {
        const row_candidates = [_]usize{
            1,
            @min(2, max_narrow),
            @min(@as(usize, cfg_mod.route_spec_rows), max_narrow),
            max_narrow,
        };
        for (row_candidates, 0..) |rows, ri| {
            var seen_row = false;
            for (row_candidates[0..ri]) |prev| if (prev == rows) {
                seen_row = true;
                break;
            };
            if (seen_row) continue;
            const max_p0 = reach -| rows;
            const pend_p0 = pendPos(max_p0, max_r);
            if (rows == 1) {
                const pos = [_]usize{ 0, @min(m.cfg.window + 1, max_p0), pend_p0, max_p0 };
                for (pos, 0..) |p0, pi| {
                    var seen_p = false;
                    for (pos[0..pi]) |prev_p| if (prev_p == p0) {
                        seen_p = true;
                        break;
                    };
                    if (seen_p) continue;
                    try measureProbe(m, p, .{ .phase = .forward, .rows = rows, .n = p0, .want = .last, .lazy_head = m.lazy_head, .route_spec = m.route_spec });
                    probes += 1;
                }
                if (m.lazy_head) {
                    try measureProbe(m, p, .{ .phase = .forward, .rows = 1, .n = max_p0, .want = .last, .lazy_head = false, .route_spec = m.route_spec });
                    probes += 1;
                }
                if (spec_off_differs) {
                    try measureProbe(m, p, .{ .phase = .forward, .rows = 1, .n = max_p0, .want = .last, .lazy_head = m.lazy_head, .route_spec = false });
                    probes += 1;
                }
            } else {
                const pos = [_]usize{ 0, pend_p0, max_p0 };
                for (pos, 0..) |p0, pi| {
                    var seen_p = false;
                    for (pos[0..pi]) |prev_p| if (prev_p == p0) {
                        seen_p = true;
                        break;
                    };
                    if (seen_p) continue;
                    try measureProbe(m, p, .{ .phase = .forward, .rows = rows, .n = p0, .want = .last, .lazy_head = false, .route_spec = m.route_spec });
                    probes += 1;
                }
                if (rows == max_narrow and spec_off_differs) {
                    try measureProbe(m, p, .{ .phase = .forward, .rows = rows, .n = max_p0, .want = .last, .lazy_head = false, .route_spec = false });
                    probes += 1;
                }
            }
        }
        if (has_vision) {
            try measureProbe(m, p, .{ .phase = .forward, .rows = 1, .n = reach -| 1, .want = .last, .has_img = true, .route_spec = false });
            probes += 1;
            if (max_narrow > 1) {
                try measureProbe(m, p, .{ .phase = .forward, .rows = max_narrow, .n = reach -| max_narrow, .want = .last, .has_img = true, .route_spec = false });
                probes += 1;
            }
        }
    }

    // 2. DSpark verify, finish_verify, and draft.
    if (m.dspark != null) {
        const block_max: usize = @min(@as(usize, m.cfg.dspark.block), cfg_mod.ring_slack);
        const v_max = @min(block_max + 1, @min(@as(usize, cfg_mod.ring_slack), reach));
        if (v_max >= 2) {
            const vrows_list = [_]usize{ 2, v_max };
            for (vrows_list, 0..) |vrows, vi| {
                if (vi > 0 and vrows == vrows_list[0]) continue;
                const max_v = reach -| vrows;
                const pend_v = pendPos(max_v, max_r);
                try measureProbe(m, p, .{ .phase = .forward, .rows = vrows, .n = pend_v, .want = .all, .verify = true, .lazy_head = m.lazy_head, .route_spec = m.route_spec });
                probes += 1;
                if (max_v != pend_v) {
                    try measureProbe(m, p, .{ .phase = .forward, .rows = vrows, .n = max_v, .want = .all, .verify = true, .lazy_head = m.lazy_head, .route_spec = m.route_spec });
                    probes += 1;
                }
                if (vrows == v_max and spec_off_differs) {
                    try measureProbe(m, p, .{ .phase = .forward, .rows = vrows, .n = max_v, .want = .all, .verify = true, .lazy_head = m.lazy_head, .route_spec = false });
                    probes += 1;
                }
            }
            const max_v = reach -| v_max;
            const pend_v = pendPos(max_v, max_r);
            const fv_pos = [_]usize{ pend_v, max_v };
            for (fv_pos, 0..) |p0, pi| {
                if (pi > 0 and p0 == fv_pos[0]) continue;
                for ([_]usize{ 1, v_max }) |keep| {
                    try measureProbe(m, p, .{ .phase = .finish_verify, .rows = v_max, .n = p0, .keep = keep, .verify = true });
                    probes += 1;
                }
            }
        }
        if (block_max >= 1 and reach >= block_max) {
            const w_candidates = [_]usize{ 1, block_max };
            for (w_candidates, 0..) |width, wi| {
                if (wi > 0 and width == w_candidates[0]) continue;
                const n_deep = reach -| width;
                const n_short = @min(1, reach -| 1);
                try measureProbe(m, p, .{ .phase = .draft, .rows = width, .n = n_short, .draft_heads = .conf, .draft_short = true });
                probes += 1;
                if (n_deep != n_short or !(n_deep < m.cfg.window)) {
                    try measureProbe(m, p, .{ .phase = .draft, .rows = width, .n = n_deep, .draft_heads = .conf, .draft_short = n_deep < m.cfg.window });
                    probes += 1;
                }
                if (width == block_max) {
                    try measureProbe(m, p, .{ .phase = .draft, .rows = width, .n = n_deep, .draft_heads = .full, .draft_short = n_deep < m.cfg.window });
                    probes += 1;
                }
            }
        }
    }

    // 3. Joined forward envelopes.
    if (p.joined > 1) {
        const was_serves = m.serves_rows;
        defer m.serves_rows = was_serves;
        m.serves_rows = true;

        const max_p0 = reach -| 1;
        const pend_p0 = pendPos(max_p0, max_r);
        const nrows_list = [_]usize{ 2, p.joined };
        for (nrows_list, 0..) |nrows, ni| {
            if (ni > 0 and nrows == nrows_list[0]) continue;
            var groups: [16]u32 = @splat(1);
            var ns: [16]usize = @splat(max_p0);
            try measureProbe(m, p, .{
                .phase = .joined,
                .rows = nrows,
                .groups = groups[0..nrows],
                .ns = ns[0..nrows],
                .lazy_head = m.lazy_head,
                .route_spec = m.route_spec,
            });
            probes += 1;
            if (nrows == p.joined) {
                if (m.lazy_head) {
                    try measureProbe(m, p, .{
                        .phase = .joined,
                        .rows = nrows,
                        .groups = groups[0..nrows],
                        .ns = ns[0..nrows],
                        .lazy_head = false,
                        .route_spec = m.route_spec,
                    });
                    probes += 1;
                }
                for (0..nrows) |k| {
                    ns[k] = switch (k % 3) {
                        0 => 0,
                        1 => pend_p0,
                        else => max_p0,
                    };
                }
                try measureProbe(m, p, .{
                    .phase = .joined,
                    .rows = nrows,
                    .groups = groups[0..nrows],
                    .ns = ns[0..nrows],
                    .lazy_head = m.lazy_head,
                    .route_spec = m.route_spec,
                });
                probes += 1;
                if (m.dspark != null and nrows >= 3) {
                    const v_w: u32 = @intCast(@min(@as(usize, m.cfg.dspark.block) + 1, nrows));
                    if (v_w >= 2) {
                        var vgroups: [16]u32 = @splat(1);
                        vgroups[0] = v_w;
                        const slots = nrows - v_w + 1;
                        var vns: [16]usize = @splat(reach -| v_w);
                        try measureProbe(m, p, .{
                            .phase = .joined,
                            .rows = nrows,
                            .groups = vgroups[0..slots],
                            .ns = vns[0..slots],
                            .lazy_head = false,
                            .route_spec = m.route_spec,
                        });
                        probes += 1;
                    }
                }
            }
        }
    }

    if (chunk_max > 0) _ = try regionAt(m, p, @min(chunk_max, reach));

    const narrow_layout = plan.region.withNarrowHeadroom(p.narrow, p.alignment);
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[dsv41-plan] {d} worst-case probes to {d}: raw narrow scratch {d} KB (with headroom {d} KB), host ring {d} KB x 2; recorded in {d} ms\n", .{
        probes,
        reach,
        p.narrow.scratch >> 10,
        narrow_layout.scratch >> 10,
        narrow_layout.host >> 10,
        ms,
    });
    return p;
}

/// Measures the wide scratch and host-ring bytes needed by `rows`-row prefill chunks (normal + vision + CED split).
pub fn regionAt(m: *Model, p: *Plans, rows: usize) !plan.region.Need {
    const gpa = p.gpa;
    if (p.wide_cache.get(rows)) |cached| {
        p.region = cached;
        p.region_rows = rows;
        return cached;
    }
    if (rows == 0 or p.reach == 0) {
        p.region = .{};
        p.region_rows = 0;
        return .{};
    }

    var most: plan.region.Need = .{};
    const max_r = maxRatio(&m.cfg);
    const p0_hi = p.reach -| rows;
    const p0_pend = pendPos(p0_hi, max_r);

    // Deepest chunk under both fills & token seeds.
    inline for (.{ .{ .zeros, .zeros }, .{ .ramp, .spread } }) |seed| {
        var passes = try recordPasses(m, gpa, .{ .phase = .wide, .rows = rows, .n = p0_hi, .want = .last, .route_spec = false }, p.reach, seed[0], seed[1]);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            most = most.max(try plan.region.needWithGuard(gpa, tr, p.alignment, p.guard));
        }
    }
    // Shallow & pending-KV-boundary wide chunks.
    const extra_pos = [_]usize{ 0, p0_pend };
    for (extra_pos, 0..) |p0, pi| {
        if (p0 == p0_hi) continue;
        if (pi > 0 and p0 == extra_pos[0]) continue;
        var passes = try recordPasses(m, gpa, .{ .phase = .wide, .rows = rows, .n = p0, .want = .last, .route_spec = false }, p.reach, .zeros, .zeros);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            most = most.max(try plan.region.needWithGuard(gpa, tr, p.alignment, p.guard));
        }
    }
    // Near-chunk tails (final span and chunk - 1).
    const span = @min(@as(usize, @import("../prefill_plan.zig").final_span_rows), rows);
    const tails = [_]usize{ rows -| 1, span };
    for (tails, 0..) |extra_rows, ti| {
        if (extra_rows <= 16 or extra_rows >= rows) continue;
        if (ti > 0 and extra_rows == tails[0]) continue;
        const p0 = p.reach -| extra_rows;
        var passes = try recordPasses(m, gpa, .{ .phase = .wide, .rows = extra_rows, .n = p0, .want = .last, .route_spec = false }, p.reach, .zeros, .zeros);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            most = most.max(try plan.region.needWithGuard(gpa, tr, p.alignment, p.guard));
        }
    }
    // Vision-spliced wide chunk.
    if (m.cfg.image_token_id != 0 and m.layers[0].gate_bias_vl.ctx != null and p.reach >= rows) {
        var passes = try recordPasses(m, gpa, .{ .phase = .wide, .rows = rows, .n = p0_hi, .want = .last, .has_img = true, .route_spec = false }, p.reach, .zeros, .zeros);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            most = most.max(try plan.region.needWithGuard(gpa, tr, p.alignment, p.guard));
        }
    }
    // CED split wide chunk.
    if (m.ced_split != null and rows > m.cfg.window and p.reach >= rows) {
        const p0s = [_]usize{ 0, p0_hi };
        for (p0s, 0..) |p0, pi| {
            if (pi > 0 and p0 == 0) continue;
            inline for (.{ .{ .zeros, .zeros }, .{ .ramp, .spread } }) |seed| {
                var passes = try recordPasses(m, gpa, .{ .phase = .wide, .rows = rows, .n = p0, .want = .last, .ced_skip = true, .route_spec = false }, p.reach, seed[0], seed[1]);
                defer passes.deinit();
                for (passes.traces) |*tr| {
                    most = most.max(try plan.region.needWithGuard(gpa, tr, p.alignment, p.guard));
                }
            }
        }
    }

    try p.wide_cache.put(gpa, rows, most);
    p.region = most;
    p.region_rows = rows;
    return most;
}

pub const Encode = struct {
    tower: *@import("dsv41_vision.zig").Dsv41Vision,
    max_patches: u64,
};

pub fn recordEncode(p: *Plans, encode: Encode) !void {
    p.encode = try @import("dsv41_encode_plan.zig").need(encode.tower, p.gpa, encode.max_patches, p.alignment, p.guard);
}

/// Allocates the shared static Metal arena covering all planned narrow envelopes and wide regions, plus fixed
/// Cooler logit/id blocks on `m`.
pub fn makeArena(m: *Model, p: *Plans) !void {
    return makeArenaWithBorrow(m, p, null);
}

pub const BorrowMode = enum { all_from_borrow, within_wide };

/// Test hook: allow `planBuild` to alias decoder slabs even on a resident load so tiny-fixture tests can
/// exercise lent-route late-compile contracts without artificially forcing SSD streaming.
pub var force_alias_for_test: bool = false;

/// Returns `engine` for tail-aliasing only when experts stream from SSD (`!resident`), or under a test hook.
/// A resident load holds every routed expert in RAM ("routes read no SSD") and already fits the full wide arena
/// inside its static-bill budget (`openLedger`); aliasing a resident layer's slabs into the wide arena tail forces
/// SSD reads on wide prefill and a ~300 ms synchronous SSD refill on every request's first decode tick.
pub fn aliasEngineForServing(engine: ?*expert_stream.Engine) ?*expert_stream.Engine {
    const eng = engine orelse return null;
    if (!eng.plan.resident) return eng;
    if (comptime @import("builtin").is_test) if (force_alias_for_test) return eng;
    return null;
}

pub fn makeArenaWithBorrow(m: *Model, p: *Plans, engine: ?*expert_stream.Engine) !void {
    return makeArenaWithBorrowBudget(m, p, engine, .all_from_borrow);
}

/// Allocates fixed model/sample blocks and creates the shared static arena, optionally aliasing eligible expert
/// stream layers (`borrow_from..`) into a page-aligned tail past narrow scratch (`Layout.keep`). `.within_wide` only
/// aliases layers that fit inside `wide_g - keep` without growing the arena.
pub fn makeArenaWithBorrowBudget(m: *Model, p: *Plans, engine: ?*expert_stream.Engine, mode: BorrowMode) !void {
    try ensureSpecFixed(m);
    if (m.logits_fixed[0] == null) try m.makeLogitsBlocks();
    if (m.ids_fixed == null) try m.makeIdsBlock();
    if (p.sample.planned) try @import("dsv41_sample_plan.zig").makeTokens(m, &p.sample);
    p.keep_g = null;
    p.alias_bytes = 0;
    p.alias_engine = null;
    if (engine) |eng| if (eng.borrow_from < eng.layers.len) {
        const base_l = p.layout();
        const narrow_g = plan.region.withNarrowHeadroom(p.narrow, p.alignment).scratch;
        const page: u64 = @max(std.heap.pageSize(), std.heap.page_size_max);
        const h_bytes = base_l.h * base_l.replicas;
        const tail_off = std.mem.alignForward(u64, h_bytes + narrow_g, page);
        const keep = tail_off - h_bytes;
        const budget: ?u64 = switch (mode) {
            .all_from_borrow => null,
            .within_wide => base_l.g -| keep,
        };
        const ap = eng.aliasPlan(eng.borrow_from, budget);
        if (ap.total_bytes > 0) {
            p.keep_g = keep;
            p.alias_bytes = ap.total_bytes;
            errdefer {
                p.keep_g = null;
                p.alias_bytes = 0;
            }
            try core.arenaFor(p.layout());
            p.has_arena = true;
            const tail = core.arenaTail() orelse {
                core.arenaRelease() catch {};
                p.has_arena = false;
                return error.TensorUnsupported;
            };
            const installed = eng.installAliasedLayers(eng.borrow_from, tail) catch |e| {
                core.arenaRelease() catch {};
                p.has_arena = false;
                return e;
            };
            if (installed > 0) p.alias_engine = eng;
            return;
        }
    };
    try core.arenaFor(p.layout());
    p.has_arena = true;
}

/// Executes a narrow `extendVision` (1..DECODE_ROWS_MAX rows) on the static arena (or delegates to `wide`).
pub fn run(m: *Model, ids: []const u32, want: Want, vision: ?dsv41_model.VisionRows) !mlx.mlx_array {
    const eff_want: Want = if (ids.len == 1) .last else want;
    const p = m.plans orelse {
        return dsv41_model.extendVision(m, ids, eff_want, dsv41_model.SPAN, vision);
    };
    if (ids.len > exl3.kernels.DECODE_ROWS_MAX) return wide(m, ids, eff_want, vision);
    try p.releaseAliasLease();
    const dh = m.defer_head;
    const armed = m.arm(.step);
    defer armed.end();
    m.defer_head = dh;
    try core.beginNarrow();
    const scope: ?GemmScope = if (ids.len > 1) GemmScope.begin() catch |e| {
        core.abortNarrow();
        return e;
    } else null;
    const logits = dsv41_model.extendVision(m, ids, eff_want, dsv41_model.SPAN, vision) catch |e| {
        if (scope) |sc| sc.end();
        core.abortNarrow();
        return e;
    };
    if (scope) |sc| sc.end();
    core.endNarrow() catch |e| {
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

/// Executes a wide prefill chunk inside the measured wide region on the shared arena.
pub fn wide(m: *Model, ids: []const u32, want: Want, vision: ?dsv41_model.VisionRows) !mlx.mlx_array {
    return wideAhead(m, ids, want, vision, &.{});
}

/// Executes a wide prefill chunk inside the measured wide region on the shared arena, chaining `ahead` rows.
pub fn wideAhead(m: *Model, ids: []const u32, want: Want, vision: ?dsv41_model.VisionRows, ahead: []const u32) !mlx.mlx_array {
    const armed = m.arm(.prefill);
    defer armed.end();
    const spec_was = m.route_spec;
    defer m.route_spec = spec_was;
    m.route_spec = false;
    const p = m.plans orelse return dsv41_model.extendAhead(m, ids, want, dsv41_model.SPAN, vision, ahead);
    try p.acquireAliasLease();
    try core.beginWide();
    const scope = GemmScope.begin() catch |e| {
        core.abortWide();
        return e;
    };
    const logits = dsv41_model.extendAhead(m, ids, want, dsv41_model.SPAN, vision, ahead) catch |e| {
        scope.end();
        core.abortWide();
        return e;
    };
    scope.end();
    errdefer _ = mlx.mlx_array_free(logits);
    try core.endWide();
    return logits;
}

pub fn wideState(m: *Model, ids: []const u32, vision: ?dsv41_model.VisionRows, ahead: []const u32) !void {
    const armed = m.arm(.prefill);
    defer armed.end();
    const spec_was = m.route_spec;
    defer m.route_spec = spec_was;
    m.route_spec = false;
    const p = m.plans orelse return dsv41_model.extendState(m, ids, dsv41_model.SPAN, vision, ahead);
    try p.acquireAliasLease();
    try core.beginWide();
    const scope = GemmScope.begin() catch |e| {
        core.abortWide();
        return e;
    };
    dsv41_model.extendState(m, ids, dsv41_model.SPAN, vision, ahead) catch |e| {
        dsv41_model.pipeDrain(m) catch {};
        scope.end();
        core.abortWide();
        return e;
    };
    dsv41_model.pipeDrain(m) catch |e| {
        scope.end();
        core.abortWide();
        return e;
    };
    scope.end();
    try core.endWide();
}

pub fn runVerify(m: *Model, ids: []const u32) !mlx.mlx_array {
    if (ids.len == 1) return run(m, ids, .last, null);
    const p = m.plans orelse return dsv41_model.verify(m, ids);
    const v_max = @min(@as(usize, m.cfg.dspark.block) + 1, cfg_mod.ring_slack);
    if (ids.len > v_max) {
        try p.releaseAliasLease();
        const w = core.widePhase();
        defer w.end();
        return dsv41_model.verify(m, ids);
    }
    try p.releaseAliasLease();
    try core.beginNarrow();
    const scope = GemmScope.begin() catch |e| {
        core.abortNarrow();
        return e;
    };
    const logits = dsv41_model.verify(m, ids) catch |e| {
        scope.end();
        core.abortNarrow();
        return e;
    };
    scope.end();
    core.endNarrow() catch |e| {
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

pub fn runFinishVerify(m: *Model, accepted: usize) !void {
    const p = m.plans orelse return dsv41_model.finishVerify(m, accepted);
    const st = (try dsv41_model.currentState(m)) orelse return dsv41_model.finishVerify(m, accepted);
    const v = st.verify orelse return dsv41_model.finishVerify(m, accepted);
    const v_max = @min(@as(usize, m.cfg.dspark.block) + 1, cfg_mod.ring_slack);
    if (v.rows > v_max) {
        try p.releaseAliasLease();
        const w = core.widePhase();
        defer w.end();
        return dsv41_model.finishVerify(m, accepted);
    }
    try p.releaseAliasLease();
    try core.beginNarrow();
    dsv41_model.finishVerify(m, accepted) catch |e| {
        core.abortNarrow();
        return e;
    };
    try core.endNarrow();
}

pub fn runDraft(m: *Model, t1: u32, heads_wanted: Heads, limit: ?u32, meter: ?*@import("../round_meter.zig").RoundMeter) !dsv41_model.Draft {
    const p = m.plans orelse return dsv41_model.draftMeasured(m, t1, heads_wanted, limit, meter);
    try p.releaseAliasLease();
    try core.beginNarrow();
    var drafted = dsv41_model.draftMeasured(m, t1, heads_wanted, limit, meter) catch |e| {
        core.abortNarrow();
        return e;
    };
    core.endNarrow() catch |e| {
        drafted.deinit();
        return e;
    };
    return drafted;
}

pub fn runJoined(m: *Model, segs: []const dsv41_model.Seg, lazy_head: bool, out_views: []mlx.mlx_array) !void {
    const p = m.plans orelse return error.Dsv41NoModel;
    if (segs.len > 16 or out_views.len < segs.len) return error.Dsv41RowsWidth;
    for (0..segs.len) |i| out_views[i] = .{ .ctx = null };
    const armed = m.arm(.rows);
    defer armed.end();
    armed.lazyHead(lazy_head);
    try p.releaseAliasLease();
    try core.beginNarrow();
    const scope = GemmScope.begin() catch |e| {
        core.abortNarrow();
        return e;
    };
    const joined = dsv41_model.extendRows(m, segs) catch |e| {
        scope.end();
        core.abortNarrow();
        return e;
    };
    const s = dsv41_model.stateStream(m);
    var start: usize = 0;
    var wrote: usize = 0;
    for (segs, 0..) |sg, i| {
        out_views[i] = store_mod.writeGroupBlock(&sg.request.logits_fixed, &sg.request.logits_turn, joined, @intCast(start), @intCast(start + sg.ids.len), m.logits_rows, s) catch |e| {
            for (out_views[0..wrote]) |v| _ = mlx.mlx_array_free(v);
            @memset(out_views[0..segs.len], .{ .ctx = null });
            _ = mlx.mlx_array_free(joined);
            scope.end();
            core.abortNarrow();
            return e;
        };
        wrote += 1;
        start += sg.ids.len;
    }
    _ = mlx.mlx_array_free(joined);
    scope.end();
    core.endNarrow() catch |e| {
        for (out_views[0..segs.len]) |v| _ = mlx.mlx_array_free(v);
        @memset(out_views[0..segs.len], .{ .ctx = null });
        return e;
    };
}
