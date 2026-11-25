//! Hy4-Flash on the static bill (docs/bill_redesign.md, stage 3): before the load's warm each planned phase is recorded
//! dry at the shapes that decide its structure and its sizes, merged per root and packed (`plan_book`), and the region
//! its wide forwards run in is measured (D97); at the warm's end the arena is made for both. A served forward of a
//! planned root runs as an execution on the arena, and a miss refuses it by name, the shape it missed recorded and
//! learned before the error returns (D5); any other forward runs in the region, and one past it is refused by name.
const std = @import("std");
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_book = @import("../tensor/plan/plan_book.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const log = @import("../log.zig");
const sparse = @import("../sparse_select.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_config = @import("hy4_config.zig");
const Model = hy4_model.Model;
const Readout = Model.Readout;
const KVQuantConfig = @import("../kv_quant.zig").KVQuantConfig;
const Drafter = @import("hy4_dfly.zig").Drafter;
const GemmScope = @import("sushi_exl3").linear.GemmScope;

/// A planned forward's root: its rows, how it reads out, whether it keeps the drafter's handover, and its KV width (a
/// request may name its own), each of which decides its structure.
pub fn root(p: Probe) plan.FrameId {
    // A draft's root is its own: the block `[t1, mask x draft]`, its width the config's, read from the handover.
    if (p.phase == .draft) return .{ .site = plan.siteOf(@src()), .key = plan.keyOf(.{ p.kv.scheme, p.kv.bits, p.kv.group_size }) };
    return .{ .site = plan.siteOf(@src()), .key = plan.keyOf(.{ p.rows, p.readout, p.keep_hidden, p.kv.scheme, p.kv.bits, p.kv.group_size }) };
}

/// One shape a root is recorded at: its rows after a state `n` rows deep; a target's forward, or DFly2's draft
/// (`Drafter.draft`) from a state `n` rows deep and the handover of its last row.
pub const Probe = struct { rows: usize, readout: Readout, keep_hidden: bool, kv: KVQuantConfig, n: usize = 0, phase: Phase = .forward, drafter: ?*const Drafter = null };

pub const Phase = enum {
    forward,
    draft,
    /// A forward the region runs (D97), recorded to measure it: its GEMM inputs are its own (`GemmScope`), as its
    /// execution's are.
    wide,
};

/// A draft at a state `n` rows deep, by `d`.
pub fn draftProbe(d: *const Drafter, c: *const hy4_config.Config, kv: KVQuantConfig, n: usize) Probe {
    return .{ .rows = c.dfly.draft + 1, .readout = .all, .keep_hidden = true, .kv = kv, .n = n, .phase = .draft, .drafter = d };
}

/// The positions a draft is recorded at, for a load whose requests reach `reach` rows: the state's depth either side of
/// each edge its layers' shared rows and keys turn at (the target's selection past its threshold, the window, the
/// latent's width where its products turn arms), the first rows, and the end. Sorted, each once.
pub fn draftPositions(gpa: std.mem.Allocator, c: *const hy4_config.Config, reach: usize) ![]usize {
    const t: usize = c.dfly.draft + 1;
    if (reach <= t) return error.Hy4PlanNoContext;
    const ratio: usize = c.indexer.group;
    const last = reach - t;
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    for (1..@min(ratio + t, last) + 1) |n| try list.append(gpa, n);
    const edges = [_]usize{ ratio, c.window, c.kv_lora, @as(usize, c.indexer.group_topk) * ratio };
    for (edges) |e| {
        var n = @max(e -| (t + ratio), 1);
        while (n <= @min(e + ratio, last)) : (n += 1) try list.append(gpa, n);
    }
    var n = @max(last -| ratio, 1);
    while (n <= last) : (n += 1) try list.append(gpa, n);
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    var kept: usize = 0;
    for (list.items) |v| {
        if (kept > 0 and list.items[kept - 1] == v) continue;
        list.items[kept] = v;
        kept += 1;
    }
    list.shrinkRetainingCapacity(kept);
    return list.toOwnedSlice(gpa);
}

/// The rows a decode's state may reach before its row: the position of the last row a request of the load can run.
fn lastPosition(c: *const hy4_config.Config, reach: usize) usize {
    _ = c;
    return reach - 1;
}

/// The positions a one-row forward is recorded at, for a load whose requests reach `reach` rows: every structure its
/// layers' keys take (`Model.layerKey`: a group of `ratio` opening and closing, the window filling, the selection's
/// threshold, its top-k splitting), each at the largest position that takes it, which holds its largest sizes: every
/// residue of a group just below and just past each edge, and at the end. Sorted, each once.
pub fn decodePositions(gpa: std.mem.Allocator, c: *const hy4_config.Config, reach: usize) ![]usize {
    if (reach == 0) return error.Hy4PlanNoContext;
    const ratio: usize = c.indexer.group;
    const last = lastPosition(c, reach);
    var list: std.ArrayList(usize) = .empty;
    errdefer list.deinit(gpa);
    // A row at position p scores the groups closed by p + 1: the top-k splits from the position before its edge. Keys
    // past the latent's width turn the score and value products between split-K and fused (`matmul.steelPath`).
    const edges = [_]usize{ ratio, c.window, c.kv_lora, @as(usize, c.indexer.group_topk) * ratio, sparse.topk_split_min_n_one_row * ratio - 1, last + 1 };
    for (0..@min(ratio, last + 1)) |n| try list.append(gpa, n);
    for (edges) |e| {
        const lo = e -| ratio;
        const hi = @min(e + ratio, last + 1);
        var n = lo;
        while (n < hi) : (n += 1) try list.append(gpa, n);
    }
    std.mem.sort(usize, list.items, {}, std.sort.asc(usize));
    var kept: usize = 0;
    for (list.items) |n| {
        if (kept > 0 and list.items[kept - 1] == n) continue;
        list.items[kept] = n;
        kept += 1;
    }
    list.shrinkRetainingCapacity(kept);
    return list.toOwnedSlice(gpa);
}

/// The positions a forward of `rows` rows (a verify block, 2 to `verify_rows_max`) is recorded at, for a load whose
/// requests reach `reach` rows: candidates either side of every edge its layers' keys turn at (`Model.layerKey`: a group
/// of `ratio` closing, the window filling and wrapping, the selection's threshold, its top-k splitting at a block's
/// count of groups), every one past the window's last wrap before the end, and the latent's width, where the score and
/// value products turn arms (`matmul.steelPath`); of the candidates that take one set of keys, the largest, which holds
/// its largest sizes. Sorted.
pub fn blockPositions(gpa: std.mem.Allocator, m: *const Model, reach: usize, rows: usize) ![]usize {
    const c = &m.config;
    if (reach < rows) return error.Hy4PlanNoContext;
    const ratio: usize = c.indexer.group;
    const last = reach - rows;
    var list: std.ArrayList(usize) = .empty;
    defer list.deinit(gpa);
    for (0..@min(ratio + rows, last) + 1) |n| try list.append(gpa, n);
    const split = sparse.topk_split_min_n * ratio;
    const edges = [_]usize{ ratio, c.window, 2 * c.window, c.kv_lora, @as(usize, c.indexer.group_topk) * ratio, split };
    for (edges) |e| {
        var n = e -| (rows + ratio);
        while (n <= @min(e + ratio, last)) : (n += 1) try list.append(gpa, n);
    }
    var n = last -| (c.window + rows + ratio);
    while (n <= last) : (n += 1) try list.append(gpa, n);
    std.mem.sort(usize, list.items, {}, std.sort.desc(usize));
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(gpa);
    var kept: std.ArrayList(usize) = .empty;
    errdefer kept.deinit(gpa);
    for (list.items, 0..) |p, i| {
        if (i > 0 and list.items[i - 1] == p) continue;
        var h = std.hash.Wyhash.init(0);
        for (0..c.layers) |l| std.hash.autoHash(&h, m.layerKey(l, @intCast(p), @intCast(rows)));
        std.hash.autoHash(&h, p + rows > c.kv_lora);
        if ((try seen.getOrPut(gpa, h.final())).found_existing) continue;
        try kept.append(gpa, p);
    }
    std.mem.sort(usize, kept.items, {}, std.sort.asc(usize));
    return kept.toOwnedSlice(gpa);
}

/// The verify blocks a load that drafts plans (D18): a DSpark round's `[t1, drafts...]`, 2 to `verify_rows_max` rows,
/// every row read out, the handover kept for the drafter. A wider round is a prefill block, on the pool.
fn verifyRoot(rows: usize, kv: KVQuantConfig) Probe {
    return .{ .rows = rows, .readout = .all, .keep_hidden = true, .kv = kv };
}

/// The replay a DSpark round's rollback runs (`Model.rollbackTo`): the 1 to `verify_rows_max - 1` rows it keeps, from the
/// mark, read out by none, the handover kept. Its state is a plain forward's at the mark: the rows cut back, the rings
/// and open groups written back in place into the blocks they were made in.
fn replayRoot(rows: usize, kv: KVQuantConfig) Probe {
    return .{ .rows = rows, .readout = .none, .keep_hidden = true, .kv = kv };
}

/// A load's plans: the book of its planned roots, the reach every probe reserved the state for, the KV width they were
/// recorded at (a request's own runs on the pool, billed so), and where the shapes its misses record are kept for the
/// next load.
pub const Plans = struct {
    book: plan_book.Book,
    reach: usize,
    kv: KVQuantConfig,
    cache: ?plan_book.Cache = null,
    /// The pack's identity (`pipeline_manifest.addPack`), which names the region kept beside the manifest; none keeps
    /// no region and reads none.
    pack: ?[16]u8 = null,
    /// The load's DFly2, whose drafts are planned; null for a load that does not draft.
    drafter: ?*const Drafter = null,
    /// Misses served since the load, and of them the shapes learned in place.
    misses: u64 = 0,
    learned: u64 = 0,
    /// The load's samples (D24), whose roots share the book.
    sample: @import("hy4_sample_plan.zig").State = .{},
    /// The region the load's wide forwards run in (D97, `makeArena`), and what each chunk width measured asked
    /// (`regionAt`); and what its image encode asks, where it keeps the tower (`hy4_encode_plan.need`).
    region: plan.region.Need = .{},
    encode: plan.region.Need = .{},
    region_rows: usize = 0,
    measured: std.ArrayList(Measured) = .empty,
    /// Where the deep zone starts (`deepZone`; 0: no class measured), and each class of a chunk width's forwards
    /// past it (`Model.wideClass`), which runs padded to its recording (`core.beginWidePadded`).
    deep: usize = 0,
    pads: std.AutoHashMapUnmanaged(u64, Class) = .empty,

    /// `probes`: the wide forwards recorded to measure it; `classes`: the deep zone's classes among them.
    pub const Measured = struct { rows: usize, need: plan.region.Need, probes: u64, classes: u64 = 0 };

    /// A class's deepest member, recorded: its chunk width, where it is, the scratch it tops out at, and each pass's
    /// asks and choices.
    pub const Class = struct { rows: usize, at: usize, need: u64, passes: []core.PadPass };

    pub fn tables(p: *const Plans, probe: Probe) ?*const plan.pack.Tables {
        return p.book.tables(root(probe));
    }

    /// The arena the plans, a prefill's `region` and the encode ask: each part the largest of the three.
    pub fn layoutWith(p: *const Plans, region: plan.region.Need) plan.binder.Layout {
        const most = region.max(p.encode);
        var l = p.book.layout();
        l.g = @max(l.g, most.scratch);
        l.h = @max(l.h, most.host);
        return l;
    }

    pub fn deinit(p: *Plans, gpa: std.mem.Allocator) void {
        p.sample.deinit();
        p.book.deinit();
        p.measured.deinit(gpa);
        var it = p.pads.valueIterator();
        while (it.next()) |c| freeClass(gpa, c.*);
        p.pads.deinit(gpa);
        gpa.destroy(p);
    }
};

/// A probe's traces, one a pass of the driver: a pass holds each `choose` site to one value, and passes are run until
/// every value of every site met is recorded.
pub const Passes = struct {
    gpa: std.mem.Allocator,
    traces: []plan_trace.Trace,

    pub fn deinit(p: *Passes) void {
        for (p.traces) |*t| t.deinit();
        p.gpa.free(p.traces);
        p.* = undefined;
    }
};

/// The ids a probe forwards: zeros, or ids spread over the vocabulary (none the image id), which `recordChecked` runs
/// beside them so a size that follows the ids with no `plan.bound` parts the two recordings.
pub const Ids = enum { zeros, spread };

/// `probe` recorded dry under `fill` over `ids`, a pass for each value of each `choose` it meets (`recordPass`).
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

/// One pass of `probe` recorded dry from a state of `n` rows reserved for `reach`, made without forwards
/// (`Model.dryState`), under `fill`, its choices `driver`'s: its trace. The model's state is reset on every path, and
/// its drafter flag left as it was.
fn recordPass(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, ids_kind: Ids, driver: *plan.Driver) !plan_trace.Trace {
    if (probe.phase == .draft) return recordDraft(m, gpa, probe, reach, fill, driver);
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = probe.keep_hidden;
    const kv_was = m.kv;
    defer m.kv = kv_was;
    m.kv = probe.kv;
    // A dry recording leaves the blocks it writes unusable: it writes logits blocks of its own, and the caller may still
    // hold a view of the load's pair.
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
        var rec = try plan.Recorder.begin(gpa, .{ .site = plan.siteOf(@src()) });
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        try m.dryState(probe.n, reach);
        // Made dry beside the state, as the load's pair stands before a served forward, holding no memory.
        if (held[0] != null) try m.makeLogitsBlocks();
    }
    var rec = try plan.Recorder.begin(gpa, root(probe));
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        // Its builds die inside the recording, as the execution's die inside it.
        const scope: ?GemmScope = if (probe.phase == .wide) try GemmScope.begin() else null;
        defer if (scope) |sc| sc.end();
        const ids = try gpa.alloc(u32, probe.rows);
        defer gpa.free(ids);
        spreadIds(&m.config, ids, ids_kind);
        const lg = try m.forwardServed(ids, null, probe.readout);
        _ = mlx.mlx_array_free(lg);
    }
    return rec.end() catch |e| {
        if (e == error.PlanEscape) {
            const at = rec.escape.?;
            log.err("[plan] an ask of {d} B outlives a forward of {d} rows at {d} (frame {d}, ordinal {d})\n", .{ rec.instances.items[at.instance].asks.items[at.ordinal].bytes, probe.rows, probe.n, at.instance, at.ordinal });
        }
        return e;
    };
}

/// A draft probe recorded dry: the state `n - 1` rows deep made without forwards, then one row run dry with the
/// handover kept (its hidden row and each sparse layer's selection, as a verify's last row leaves them), then the draft.
fn recordDraft(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize, fill: core.Fill, driver: *plan.Driver) !plan_trace.Trace {
    const d = probe.drafter orelse return error.Hy4DflyMissing;
    if (probe.n == 0) return error.Hy4DflyNoContext;
    const keep_was = m.keep_hidden;
    defer m.keep_hidden = keep_was;
    m.keep_hidden = true;
    const kv_was = m.kv;
    defer m.kv = kv_was;
    m.kv = probe.kv;
    m.reset();
    defer m.reset();
    {
        var rec = try plan.Recorder.begin(gpa, .{ .site = plan.siteOf(@src()) });
        defer rec.deinit();
        var state_driver: plan.Driver = .{ .gpa = gpa };
        defer state_driver.deinit();
        try core.beginRecording(.{ .dry = .{ .driver = &state_driver } }, &rec);
        defer core.endRecording() catch {};
        try m.dryState(probe.n - 1, reach);
        _ = mlx.mlx_array_free(try m.forwardReadout(&.{0}, .none));
    }
    var rec = try plan.Recorder.begin(gpa, root(probe));
    defer rec.deinit();
    {
        try core.beginRecording(.{ .dry = .{ .driver = driver, .fill = fill } }, &rec);
        defer core.endRecording() catch {};
        var drafted = try d.draft(m, 0);
        drafted.deinit();
    }
    return rec.end();
}

/// `probe` recorded twice (docs/bill_redesign.md, "A classification is checked", D99): over zero ids under the zeros
/// fill, and over spread ids under the ramp; the first's passes, once each of the second's is found to be the same
/// execution (a wide probe's dispatch levels aside, which its region does not read). A size or a choice that follows
/// the forward's data, its ids or a value it reads back, with no `plan.bound` or `plan.choose` to declare it, parts
/// them, and the probe is refused by name.
pub fn recordChecked(m: *Model, gpa: std.mem.Allocator, probe: Probe, reach: usize) !Passes {
    var zeros = try recordPasses(m, gpa, probe, reach, .zeros, .zeros);
    errdefer zeros.deinit();
    var other = try recordPasses(m, gpa, probe, reach, .ramp, .spread);
    defer other.deinit();
    if (zeros.traces.len != other.traces.len) {
        log.err("[plan] a forward of {d} rows at {d} records {d} passes over one input and {d} over another: its data steers a choice no `plan.choose` declares\n", .{ probe.rows, probe.n, zeros.traces.len, other.traces.len });
        return error.Hy4PlanDataSteers;
    }
    for (zeros.traces, other.traces, 0..) |z, r, pass| if (plan_trace.firstDifferenceOf(z.instances, r.instances, probe.phase != .wide)) |d| {
        const ask = if (d.what == .asks) z.instances[d.instance].asks.items[d.at].bytes else 0;
        const other_ask = if (d.what == .asks) r.instances[d.instance].asks.items[d.at].bytes else 0;
        log.err("[plan] a forward of {d} rows at {d} records two shapes over two inputs (pass {d}, frame {d}/{d}, {t} at {d}: {d} B against {d} B): its data steers a size no `plan.bound` declares\n", .{ probe.rows, probe.n, pass, z.instances[d.instance].id.site, z.instances[d.instance].id.key, d.what, d.at, ask, other_ask });
        return error.Hy4PlanDataSteers;
    };
    for (zeros.traces, other.traces, 0..) |z, r, pass| if (!plan_trace.sameWaits(z.waits, r.waits)) {
        log.err("[plan] a forward of {d} rows at {d} waits for the GPU at other places over two inputs (pass {d}): its data steers an eval\n", .{ probe.rows, probe.n, pass });
        return error.Hy4PlanDataSteers;
    };
    return zeros;
}

/// `ids` as `kind` names them.
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

/// A kept shape's phase: a forward of a planned root (`Probe`).
const shape_phase: u32 = 1;

/// The third number: the handover's bit, then the KV width's scheme, bits and group size.
pub fn shapeOf(p: Probe) plan_book.Shape {
    const kv: u64 = @intFromBool(p.keep_hidden) | @as(u64, @backingInt(p.kv.scheme)) << 8 | @as(u64, p.kv.bits) << 16 | @as(u64, p.kv.group_size) << 24;
    return .{ .phase = if (p.phase == .draft) draft_phase else shape_phase, .args = .{ p.rows, @backingInt(p.readout), kv, p.n } };
}

/// A kept shape's phase: a draft (`draftProbe`).
const draft_phase: u32 = 2;

/// The probe a kept shape names, or null when it names none this load can run.
fn probeOf(s: plan_book.Shape, reach: usize, drafter: ?*const Drafter, c: *const hy4_config.Config) ?Probe {
    if (s.phase == draft_phase) {
        const d = drafter orelse return null;
        const fwd_probe = probeOf(.{ .phase = shape_phase, .args = .{ 1, @backingInt(Readout.last), s.args[2] & ~@as(u64, 0xff), s.args[3] } }, reach, null, c) orelse return null;
        if (s.args[3] == 0 or s.args[3] + c.dfly.draft + 1 > reach) return null;
        return draftProbe(d, c, fwd_probe.kv, s.args[3]);
    }
    if (s.phase != shape_phase or s.args[2] & 0xfe != 0) return null;
    const rows = s.args[0];
    if (rows == 0 or rows > @import("hy4_forward.zig").verify_rows_max or s.args[3] + rows > reach) return null;
    const readout: Readout = for (std.enums.values(Readout)) |r| {
        if (@backingInt(r) == s.args[1]) break r;
    } else return null;
    const scheme = for (std.enums.values(@FieldType(KVQuantConfig, "scheme"))) |v| {
        if (@backingInt(v) == (s.args[2] >> 8) & 0xff) break v;
    } else return null;
    const kv: KVQuantConfig = .{ .scheme = scheme, .bits = @intCast((s.args[2] >> 16) & 0xff), .group_size = @intCast(s.args[2] >> 24) };
    return .{ .rows = rows, .readout = readout, .keep_hidden = s.args[2] & 1 == 1, .kv = kv, .n = s.args[3] };
}

/// The shapes earlier loads' misses kept, each recorded and learned into the book. One this load cannot run, or that
/// does not record or merge, is passed over by name: the load serves without it, and such a forward misses again.
fn learnKept(m: *Model, p: *Plans, c: *const plan_book.Cache) void {
    const gpa = p.book.gpa;
    const kept = c.read(gpa) catch |e| {
        log.warn("[plan] the kept shapes in {s} were not read ({t}): the load plans without them\n", .{ c.name(), e });
        return;
    };
    defer gpa.free(kept);
    const unbounded: @import("../tensor/plan/plan_binder.zig").Layout = .{ .g = std.math.maxInt(u64), .h = std.math.maxInt(u64), .alignment = p.book.alignment };
    var learned: usize = 0;
    var others: usize = 0;
    for (kept) |s| {
        // A sample's shape is the samples' to learn (`hy4_sample_plan.record`).
        if (s.phase == @import("hy4_sample_plan.zig").shape_phase) {
            others += 1;
            continue;
        }
        const probe = probeOf(s, p.reach, p.drafter, &m.config) orelse {
            log.warn("[plan] a kept shape is no forward of this load (phase {d}: {d} {d} {d} {d})\n", .{ s.phase, s.args[0], s.args[1], s.args[2], s.args[3] });
            continue;
        };
        const passes = recordChecked(m, gpa, probe, p.reach) catch |e| {
            log.warn("[plan] a kept shape, {d} rows at {d}, was not recorded ({t})\n", .{ probe.rows, probe.n, e });
            continue;
        };
        defer gpa.free(passes.traces);
        _ = p.book.learnAll(root(probe), passes.traces, unbounded) catch |e| {
            log.warn("[plan] a kept shape, {d} rows at {d}, was not planned ({t})\n", .{ probe.rows, probe.n, e });
            continue;
        };
        learned += 1;
    }
    if (kept.len > others) log.info("[plan] {d} of {d} shapes kept by earlier misses planned\n", .{ learned, kept.len - others });
}

/// The roots a load runs on the arena: its one-row forwards, read out as a step reads them (`last`) and, with a drafter,
/// as its verify reads its rows (`all`), with the handover kept where a drafter runs and without.
fn decodeRoots(drafts: bool, kv: KVQuantConfig, out: *[4]Probe) []Probe {
    var n: usize = 0;
    for ([_]bool{ false, true }) |keep| {
        if (keep and !drafts) continue;
        out[n] = .{ .rows = 1, .readout = .last, .keep_hidden = keep, .kv = kv };
        n += 1;
        if (!drafts) continue;
        out[n] = .{ .rows = 1, .readout = .all, .keep_hidden = keep, .kv = kv };
        n += 1;
    }
    return out[0..n];
}

/// The largest buffer the level rule keeps apart (`plan_pack.Options.level_rule_cap`, D23).
pub const level_rule_cap: u64 = 4 << 20;

/// `probe` recorded under both fills, each pass added to `p`'s book.
fn addProbe(m: *Model, gpa: std.mem.Allocator, p: *Plans, probe: Probe, reach: usize) !void {
    const passes = try recordChecked(m, gpa, probe, reach);
    defer gpa.free(passes.traces);
    for (passes.traces, 0..) |t, i| p.book.add(root(probe), t) catch |e| {
        for (passes.traces[i + 1 ..]) |*rest| rest.deinit();
        return e;
    };
}

/// The plans of `m` for requests that reach at most `reach` rows, recorded before the load's warm (D98): the wide
/// pipelines latched first (D20), then each root recorded at every decode position, merged and packed, the shapes
/// earlier misses kept learned, and the logits blocks made (D19). No arena is made: `makeArena` makes it with the
/// region.
pub fn record(m: *Model, gpa: std.mem.Allocator, reach: usize, drafter: ?*const Drafter, kv: KVQuantConfig, cache: ?plan_book.Cache, encode: ?Encode) !*Plans {
    const drafts = drafter != null;
    const t0 = @import("../load_clock.zig").now();
    try m.probeWide();
    const p = try gpa.create(Plans);
    errdefer gpa.destroy(p);
    // A test build keeps a canary after every place (`core.arena_check`).
    // The level rule keeps reuse off work the GPU could run side by side (D23): the floors order every reuse, the rule
    // keeps their barriers to the pool's. Past its cap a buffer reuses by lifetimes: the key-sized ones a verify block
    // makes at a long context would take the arena from 70 MB to 4 GB.
    p.* = .{ .book = .{ .gpa = gpa, .options = .{ .level_rule = true, .level_rule_cap = level_rule_cap, .guard = if (@import("builtin").is_test) 256 else 0 } }, .reach = reach, .kv = kv, .cache = cache, .drafter = drafter };
    errdefer p.book.deinit();
    errdefer p.measured.deinit(gpa);
    const positions = try decodePositions(gpa, &m.config, reach);
    defer gpa.free(positions);
    // Made first: each recording writes the logits into blocks of their shape, as a served forward does.
    try m.makeLogitsBlocks();
    try m.makeIdsBlock();
    var roots: [4]Probe = undefined;
    var probes: usize = 0;
    for (decodeRoots(drafts, kv, &roots)) |r| for (positions) |n| {
        var probe = r;
        probe.n = n;
        try addProbe(m, gpa, p, probe, reach);
        probes += 1;
    };
    // A block wider than the load's reach is one no request of it runs.
    // Typed: @min with a comptime operand narrows to the smallest integer that holds it, and the `+ 1` would wrap.
    const verify_max: usize = @min(@as(usize, @intCast(@import("hy4_forward.zig").verify_rows_max)), reach);
    if (drafts) for (2..verify_max + 1) |rows| {
        const at = try blockPositions(gpa, m, reach, rows);
        defer gpa.free(at);
        for (at) |n| {
            var probe = verifyRoot(rows, kv);
            probe.n = n;
            try addProbe(m, gpa, p, probe, reach);
            probes += 1;
        }
    };
    // A one-row replay takes the one-row forward's structures, a wider one a block's.
    const replay_max: usize = @min(@as(usize, @intCast(@import("hy4_forward.zig").verify_rows_max)) - 1, reach);
    if (drafts) for (1..replay_max + 1) |rows| {
        const at = if (rows == 1) try gpa.dupe(usize, positions) else try blockPositions(gpa, m, reach, rows);
        defer gpa.free(at);
        for (at) |n| {
            var probe = replayRoot(rows, kv);
            probe.n = n;
            try addProbe(m, gpa, p, probe, reach);
            probes += 1;
        }
    };
    if (drafter) |d| if (reach > m.config.dfly.draft + 1) {
        const at = try draftPositions(gpa, &m.config, reach);
        defer gpa.free(at);
        for (at) |n| {
            try addProbe(m, gpa, p, draftProbe(d, &m.config, kv, n), reach);
            probes += 1;
        }
    };
    if (encode) |e| p.encode = try @import("hy4_encode_plan.zig").need(e.tower, gpa, e.max_patches, p.book.alignment);
    p.book.build() catch |e| {
        if (e == error.PlanAmbiguous) {
            // The first root without tables is the one the merge refused.
            const at = for (p.book.roots.items) |r| {
                if (r.tables == null) break r.id.key;
            } else 0;
            log.err("[plan] frame {d}/{d} is ambiguous: {t} at node {d} (root {d}, {d} traces)\n", .{ p.book.ambiguity.id.site, p.book.ambiguity.id.key, p.book.ambiguity.why, p.book.ambiguity.node, at, for (p.book.roots.items) |r| {
                if (r.tables == null) break r.traces.items.len;
            } else 0 });
        }
        return e;
    };
    if (p.cache) |*c| learnKept(m, p, c);
    const l = p.book.layout();
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[plan] {d} roots from {d} probes to {d}: scratch {d} KB, host ring {d} KB x {d}; recorded in {d} ms\n", .{ p.book.roots.items.len, probes, reach, l.g >> 10, l.h >> 10, l.replicas, ms });
    return p;
}

/// `record`, then the arena made with the region of `wide_rows`-row prefill chunks (`makeArena`; 0 measures none, and
/// every wide forward is refused).
pub fn build(m: *Model, gpa: std.mem.Allocator, reach: usize, drafter: ?*const Drafter, kv: KVQuantConfig, cache: ?plan_book.Cache, wide_rows: usize) !*Plans {
    const p = try record(m, gpa, reach, drafter, kv, cache, null);
    errdefer p.deinit(gpa);
    try makeArena(m, p, wide_rows);
    return p;
}

/// The image encode a load runs in its region (`hy4_encode_plan`): its tower, and the most patches an image the load
/// accepts takes.
pub const Encode = struct { tower: *const @import("hy4_vision.zig").Tower, max_patches: u64 };

/// The depth, besides the state's start and its deepest, a wide forward of `rows` rows is recorded at to measure the
/// region: where the selection threshold cuts a sparse layer's span with a dense part of a verify block's most rows
/// (`isVerifyBlock`), each of which asks the host ring a row's more; a chunk keeps one selection row at least. Null where
/// a chunk there reaches past the load's.
fn straddleDepth(c: *const hy4_config.Config, reach: usize, rows: usize) ?usize {
    const thr: usize = @as(usize, c.indexer.group_topk) * c.indexer.group;
    const dense = @min(@as(usize, @intCast(@import("hy4_forward.zig").verify_rows_max)), thr, rows -| 1);
    return if (thr - dense + rows > reach) null else thr - dense;
}

/// The region a prefill of `rows`-row chunks asks at most (D97): a chunk, a row short of one, and the final span
/// (`prefill_plan.final_span_rows`) each at the state's start and at its deepest, read out by none and by its last row,
/// and at the depth a span is cut at the threshold (`straddleDepth`), read out by its last row, which asks no less
/// than none (its head's asks come after the layers'); each recorded dry at every pass of its choices, at the load's
/// handover and KV width. The scratch grows with the keys a chunk reads, so the deepest holds its most; the state's
/// start, the readout and the cut each add host asks of their own.
/// A forward whose last attention span is short launches it apart and joins it to the whole spans
/// (`hy4_forward.linearSpans`): a row short of the chunk holds the widest such pair, which can ask more than the chunk.
pub fn regionAt(m: *Model, p: *Plans, rows: usize) !plan.region.Need {
    for (p.measured.items) |r| if (r.rows == rows) return r.need;
    const gpa = p.book.gpa;
    const t0 = @import("../load_clock.zig").now();
    if (p.cache) |*c| if (p.pack != null) if (try readRegion(m, p, c, rows)) |kept| {
        const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
        log.info("[plan] a prefill of {d}-row chunks asks a region of {d} KB, host {d} KB, its {d} classes past {d} rows padded to their deepest; read from the load's cache and checked in {d} ms\n", .{ rows, kept.scratch >> 10, kept.host >> 10, p.measured.items[p.measured.items.len - 1].classes, p.deep, ms });
        return kept;
    };
    const span = @min(@as(usize, @import("../prefill_plan.zig").final_span_rows), rows);
    const widths = [_]usize{ rows, span, rows - 1 };
    var most: plan.region.Need = .{};
    var probes: u64 = 0;
    for (widths[0..if (rows - 1 > span) 3 else 2]) |r| {
        for ([_]usize{ 0, p.reach -| r }) |n| for ([_]Readout{ .none, .last }) |readout| {
            try measureWide(m, p, &most, r, n, readout);
            probes += 1;
        };
        if (straddleDepth(&m.config, p.reach, r)) |n| {
            try measureWide(m, p, &most, r, n, .last);
            probes += 1;
        }
    }
    const classes = try measureClasses(m, p, rows, &most);
    probes += classes;
    try p.measured.append(gpa, .{ .rows = rows, .need = most, .probes = probes, .classes = classes });
    if (p.cache) |*c| if (p.pack != null) keepRegion(m, p, c, rows) catch |e| log.warn("[plan] the region of {d}-row chunks not kept for the next load: {s}\n", .{ rows, @errorName(e) });
    const ms = (@import("../load_clock.zig").now() - t0) / std.time.ns_per_ms;
    log.info("[plan] a prefill of {d}-row chunks asks a region of {d} KB, host {d} KB, its {d} classes past {d} rows padded to their deepest; measured in {d} ms\n", .{ rows, most.scratch >> 10, most.host >> 10, classes, p.deep, ms });
    return most;
}

// ── The region kept for the next load ──
//
// A load keeps what `regionAt` measured beside its manifest (`Plans.cache`, keyed by the build and the kernels' scope),
// in a file named by the pack and what else sizes the region (`regionFileName`): the build, the kernels' scope, the
// pack and the load's settings decide it. The next load reads it back where the file is whole and agrees with the live
// load, and one probe recorded now, a load-width chunk at the deepest read out by its last row, asks what it asked
// when the file was written; anything else is measured again.

const region_version: u64 = 2;
const region_magic = "HY4REGN\x01";

/// The file of `rows`-row chunks' region: named by what sizes it past the directory's build and kernel scope.
pub fn regionFileName(m: *const Model, p: *const Plans, rows: usize, buf: *[40]u8) []const u8 {
    var h = std.hash.Wyhash.init(region_version);
    std.hash.autoHash(&h, p.pack.?);
    for ([_]u64{ rows, p.reach, p.book.alignment, @intFromBool(m.keep_hidden), @intFromBool(m.streamed), @backingInt(p.kv.scheme), p.kv.bits, p.kv.group_size, deepZone(&m.config), if (m.experts) |e| e.plan.slots_per_layer else 0 }) |v| std.hash.autoHash(&h, v);
    return std.fmt.bufPrint(buf, "region-{x:0>16}.bin", .{h.final()}) catch unreachable;
}

/// What a region file is checked by: a load-width chunk at the deepest read out by its last row, over one input.
fn checkNeed(m: *Model, p: *const Plans, rows: usize) !plan.region.Need {
    var passes = try recordPasses(m, p.book.gpa, .{ .rows = rows, .readout = .last, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = p.reach -| rows, .phase = .wide }, p.reach, .zeros, .zeros);
    defer passes.deinit();
    var most: plan.region.Need = .{};
    for (passes.traces) |*t| most = most.max(try plan.region.need(p.book.gpa, t, p.book.alignment));
    return most;
}

/// A region file's content: what `regionAt` measured for a width, its check, and its classes.
const Kept = struct {
    measured: Plans.Measured,
    check: plan.region.Need,
    deep: u64,
    keys: []u64,
    classes: []Plans.Class,

    fn deinit(k: *Kept, gpa: std.mem.Allocator) void {
        for (k.classes) |c| freeClass(gpa, c);
        gpa.free(k.classes);
        gpa.free(k.keys);
    }
};

fn encodeRegion(gpa: std.mem.Allocator, k: Kept) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll(region_magic);
    for ([_]u64{ k.measured.rows, k.measured.need.scratch, k.measured.need.host, k.measured.probes, k.measured.classes, k.check.scratch, k.check.host, k.deep, k.classes.len }) |v| try w.writeInt(u64, v, .little);
    for (k.keys, k.classes) |key, c| {
        for ([_]u64{ key, c.rows, c.at, c.need, c.passes.len }) |v| try w.writeInt(u64, v, .little);
        for (c.passes) |ps| {
            try w.writeInt(u64, ps.sizes.len, .little);
            for (ps.sizes) |v| try w.writeInt(u64, v, .little);
            try w.writeInt(u64, ps.choices.len, .little);
            for (ps.choices) |ch| for ([_]u64{ ch.site, ch.value, ch.at }) |v| try w.writeInt(u64, v, .little);
        }
    }
    try w.writeInt(u64, std.hash.Wyhash.hash(region_version, out.written()), .little);
    return out.toOwnedSlice();
}

const Cursor = struct {
    b: []const u8,
    at: usize = 0,

    fn int(c: *Cursor) !u64 {
        if (c.b.len - c.at < 8) return error.Hy4RegionFileDamaged;
        defer c.at += 8;
        return std.mem.readInt(u64, c.b[c.at..][0..8], .little);
    }

    /// The next value, refused as damage where it does not fit `T`.
    fn narrow(c: *Cursor, comptime T: type) !T {
        return std.math.cast(T, try c.int()) orelse error.Hy4RegionFileDamaged;
    }

    fn len(c: *Cursor, each: usize) !usize {
        const n = try c.int();
        if (n > (c.b.len - c.at) / each) return error.Hy4RegionFileDamaged;
        return @intCast(n);
    }
};

fn decodeRegion(gpa: std.mem.Allocator, b: []const u8) !Kept {
    if (b.len < region_magic.len + 8 or !std.mem.eql(u8, b[0..region_magic.len], region_magic)) return error.Hy4RegionFileDamaged;
    const body = b[0 .. b.len - 8];
    if (std.mem.readInt(u64, b[b.len - 8 ..][0..8], .little) != std.hash.Wyhash.hash(region_version, body)) return error.Hy4RegionFileDamaged;
    var c: Cursor = .{ .b = body, .at = region_magic.len };
    var k: Kept = .{ .measured = undefined, .check = undefined, .deep = 0, .keys = &.{}, .classes = &.{} };
    k.measured = .{ .rows = try c.narrow(usize), .need = .{ .scratch = try c.int(), .host = try c.int() }, .probes = try c.int(), .classes = try c.int() };
    k.check = .{ .scratch = try c.int(), .host = try c.int() };
    k.deep = try c.int();
    const n = try c.len(5 * 8);
    k.keys = try gpa.alloc(u64, n);
    errdefer gpa.free(k.keys);
    var classes: std.ArrayList(Plans.Class) = .empty;
    errdefer {
        for (classes.items) |cl| freeClass(gpa, cl);
        classes.deinit(gpa);
    }
    for (k.keys) |*key| {
        key.* = try c.int();
        var cl: Plans.Class = .{ .rows = try c.narrow(usize), .at = try c.narrow(usize), .need = try c.int(), .passes = &.{} };
        cl.passes = try gpa.alloc(core.PadPass, try c.len(2 * 8));
        var made: usize = 0;
        errdefer freePasses(gpa, cl.passes, made);
        for (cl.passes) |*ps| {
            const sizes = try gpa.alloc(u64, try c.len(8));
            errdefer gpa.free(sizes);
            for (sizes) |*v| v.* = try c.int();
            const choices = try gpa.alloc(plan.TraceChoice, try c.len(3 * 8));
            errdefer gpa.free(choices);
            for (choices) |*ch| ch.* = .{ .site = try c.narrow(@FieldType(plan.TraceChoice, "site")), .value = try c.narrow(u32), .at = try c.narrow(u32) };
            ps.* = .{ .sizes = sizes, .choices = choices };
            made += 1;
        }
        try classes.append(gpa, cl);
    }
    if (c.at != body.len) return error.Hy4RegionFileDamaged;
    k.classes = try classes.toOwnedSlice(gpa);
    return k;
}

/// Writes what `regionAt` just measured for `rows`-row chunks, with its check, through a temp file.
fn keepRegion(m: *Model, p: *Plans, c: *const plan_book.Cache, rows: usize) !void {
    const gpa = p.book.gpa;
    var keys: std.ArrayList(u64) = .empty;
    defer keys.deinit(gpa);
    var classes: std.ArrayList(Plans.Class) = .empty;
    defer classes.deinit(gpa);
    var it = p.pads.iterator();
    while (it.next()) |e| if (e.value_ptr.rows == rows) {
        try keys.append(gpa, e.key_ptr.*);
        try classes.append(gpa, e.value_ptr.*);
    };
    const measured = p.measured.items[p.measured.items.len - 1];
    const bytes = try encodeRegion(gpa, .{ .measured = measured, .check = try checkNeed(m, p, rows), .deep = p.deep, .keys = keys.items, .classes = classes.items });
    defer gpa.free(bytes);
    var dir = try c.root.createDirPathOpen(c.io, c.name(), .{});
    defer dir.close(c.io);
    var name_buf: [40]u8 = undefined;
    const name = regionFileName(m, p, rows, &name_buf);
    var temp_buf: [64]u8 = undefined;
    const temp = std.fmt.bufPrint(&temp_buf, "{s}.{d}.tmp", .{ name, std.c.getpid() }) catch unreachable;
    errdefer dir.deleteFile(c.io, temp) catch {};
    try dir.writeFile(c.io, .{ .sub_path = temp, .data = bytes });
    try dir.rename(temp, dir, name, c.io);
}

/// The region of `rows`-row chunks the cache keeps, installed on `p` with its classes, where its check holds; null
/// where there is none, it is damaged, or its check differs (each said in the log), and `regionAt` measures.
fn readRegion(m: *Model, p: *Plans, c: *const plan_book.Cache, rows: usize) !?plan.region.Need {
    const gpa = p.book.gpa;
    var name_buf: [40]u8 = undefined;
    const name = regionFileName(m, p, rows, &name_buf);
    var dir = c.root.createDirPathOpen(c.io, c.name(), .{}) catch return null;
    defer dir.close(c.io);
    const bytes = dir.readFileAlloc(c.io, name, gpa, .limited(1 << 30)) catch |e| {
        if (e != error.FileNotFound) log.warn("[plan] the kept region of {d}-row chunks unread: {s}; measured again\n", .{ rows, @errorName(e) });
        return null;
    };
    defer gpa.free(bytes);
    var k = decodeRegion(gpa, bytes) catch |e| {
        log.warn("[plan] the kept region of {d}-row chunks unread: {s}; measured again\n", .{ rows, @errorName(e) });
        return null;
    };
    defer k.deinit(gpa);
    // The deep zone and the class count are the live load's own: a file that names others is not this load's.
    const deep: u64 = if (p.reach >= rows and p.reach - rows >= deepZone(&m.config)) deepZone(&m.config) else 0;
    if (k.measured.rows != rows or k.deep != deep or k.measured.classes != k.keys.len) {
        log.warn("[plan] the kept region of {d}-row chunks is not this load's: {d} rows, deep zone {d}, {d} classes of {d} recorded; measured again\n", .{ rows, k.measured.rows, k.deep, k.keys.len, k.measured.classes });
        return null;
    }
    const live = try checkNeed(m, p, rows);
    if (live.scratch != k.check.scratch or live.host != k.check.host) {
        log.info("[plan] the kept region of {d}-row chunks no longer holds: its check asks {d} KB, host {d} KB, now against {d} KB, host {d} KB, kept; measured again\n", .{ rows, live.scratch >> 10, live.host >> 10, k.check.scratch >> 10, k.check.host >> 10 });
        return null;
    }
    try p.measured.ensureUnusedCapacity(gpa, 1);
    try p.pads.ensureUnusedCapacity(gpa, @intCast(k.classes.len));
    for (k.keys, k.classes) |key, cl| {
        const got = p.pads.getOrPutAssumeCapacity(key);
        if (got.found_existing) freeClass(gpa, got.value_ptr.*);
        got.value_ptr.* = cl;
    }
    gpa.free(k.classes);
    k.classes = &.{};
    if (k.deep > 0) p.deep = @intCast(k.deep);
    p.measured.appendAssumeCapacity(.{ .rows = rows, .need = k.measured.need, .probes = 1, .classes = k.measured.classes });
    return k.measured.need;
}

/// Test seam: `bytes`, a kept region, with its check replaced. The caller frees it.
pub fn regionWithCheckForTest(gpa: std.mem.Allocator, bytes: []const u8, check: plan.region.Need) ![]u8 {
    var k = try decodeRegion(gpa, bytes);
    defer k.deinit(gpa);
    k.check = check;
    return encodeRegion(gpa, k);
}

/// Test seam: `bytes`, a kept region, with its deep zone replaced. The caller frees it.
pub fn regionWithDeepForTest(gpa: std.mem.Allocator, bytes: []const u8, deep: u64) ![]u8 {
    var k = try decodeRegion(gpa, bytes);
    defer k.deinit(gpa);
    k.deep = deep;
    return encodeRegion(gpa, k);
}

/// Test seam: `p` forgets the classes of `rows`-row chunks.
pub fn forgetClassesForTest(p: *Plans, rows: usize) void {
    var doomed: std.ArrayList(u64) = .empty;
    defer doomed.deinit(p.book.gpa);
    var it = p.pads.iterator();
    while (it.next()) |e| if (e.value_ptr.rows == rows) doomed.append(p.book.gpa, e.key_ptr.*) catch return;
    for (doomed.items) |key| if (p.pads.fetchRemove(key)) |kv| freeClass(p.book.gpa, kv.value);
}

/// Where the deep zone starts: past it no row of a chunk meets the selection threshold or the window's first fill, so
/// a few facts decide a chunk's structure (`Model.wideClass`).
fn deepZone(c: *const hy4_config.Config) usize {
    return @max(@as(usize, c.indexer.group_topk) * c.indexer.group, c.window);
}

/// Each class of `rows`-row forwards in the deep zone, recorded at its deepest member, whose asks are its largest,
/// read out by its last row as a served chunk is: its scratch joins `most`, and a chunk of it runs padded to the
/// recording. The classes found.
/// First fit is not monotone, so a class's shallower members are padded to it rather than measured.
fn measureClasses(m: *Model, p: *Plans, rows: usize, most: *plan.region.Need) !u64 {
    const gpa = p.book.gpa;
    const deep = deepZone(&m.config);
    if (p.reach < rows or p.reach - rows < deep) return 0;
    p.deep = deep;
    var deepest: std.AutoHashMapUnmanaged(u64, usize) = .empty;
    defer deepest.deinit(gpa);
    var at = deep;
    while (at <= p.reach - rows) : (at += 1) try deepest.put(gpa, m.wideClass(@intCast(at), @intCast(rows), Readout.last), at);
    var it = deepest.valueIterator();
    while (it.next()) |n| {
        const readout: Readout = .last;
        // One input: the start's and the deepest's probes (`recordChecked`) and the class cells hold that data steers
        // no structure; a chunk whose asks part from its class's is refused by name, never placed.
        var passes = try recordPasses(m, gpa, .{ .rows = rows, .readout = readout, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n.*, .phase = .wide }, p.reach, .zeros, .zeros);
        defer passes.deinit();
        if (!try plan.region.choicesAgree(gpa, passes.traces, p.book.alignment)) {
            log.err("[plan] a forward of {d} rows at {d} leaves its scratch apart across the passes of its choices: a choice's asks outlive it\n", .{ rows, n.* });
            return error.Hy4PlanChoicesApart;
        }
        var class: Plans.Class = .{ .rows = rows, .at = n.*, .need = 0, .passes = try gpa.alloc(core.PadPass, passes.traces.len) };
        var made: usize = 0;
        errdefer freePasses(gpa, class.passes, made);
        for (passes.traces, class.passes) |*t, *ps| {
            const sizes = try plan.region.padOf(gpa, t, p.book.alignment);
            errdefer gpa.free(sizes);
            ps.* = .{ .sizes = sizes, .choices = try gpa.dupe(plan.TraceChoice, t.choices) };
            made += 1;
            const need = try plan.region.need(gpa, t, p.book.alignment);
            class.need = @max(class.need, need.scratch);
            most.* = most.max(need);
        }
        const got = try p.pads.getOrPut(gpa, m.wideClass(@intCast(n.*), @intCast(rows), readout));
        if (got.found_existing) freeClass(gpa, got.value_ptr.*);
        got.value_ptr.* = class;
    }
    return deepest.count();
}

fn freeClass(gpa: std.mem.Allocator, c: Plans.Class) void {
    freePasses(gpa, c.passes, c.passes.len);
}

/// `passes`, of which the first `made` are filled: their asks and choices, then the whole array as allocated.
fn freePasses(gpa: std.mem.Allocator, passes: []core.PadPass, made: usize) void {
    for (passes[0..made]) |ps| {
        gpa.free(ps.sizes);
        gpa.free(ps.choices);
    }
    gpa.free(passes);
}

/// `most`, and what a wide forward of `rows` rows after `n` read out by `readout` asks.
fn measureWide(m: *Model, p: *const Plans, most: *plan.region.Need, rows: usize, n: usize, readout: Readout) !void {
    var passes = try recordChecked(m, p.book.gpa, .{ .rows = rows, .readout = readout, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = n, .phase = .wide }, p.reach);
    defer passes.deinit();
    for (passes.traces) |*t| most.* = most.max(try plan.region.need(p.book.gpa, t, p.book.alignment));
}

/// The arena of `p` and the region of `wide_rows`-row prefill chunks (0: none), made where the load holds it.
pub fn makeArena(m: *Model, p: *Plans, wide_rows: usize) !void {
    p.region = if (wide_rows > 0) try regionAt(m, p, wide_rows) else .{};
    p.region_rows = wide_rows;
    const l = p.layoutWith(p.region);
    try core.arenaFor(l);
    log.info("[plan] arena {d} KB: scratch {d} KB, host ring {d} KB x {d}; the region of {d}-row chunks {d} KB of it, an image's encode {d} KB\n", .{ l.arenaBytes() >> 10, l.g >> 10, l.h >> 10, l.replicas, wide_rows, p.region.scratch >> 10, p.encode.scratch >> 10 });
}

/// A served forward of `ids` read out by `readout`: an execution on the arena when the load planned its root, else one
/// in its region (`wide`). A miss refuses the forward by name (`PlanMiss`): the model's state is reset, as a forward
/// that fails leaves it, the shape that missed is recorded and learned (`plan_book.Book.learn`), and the line says which
/// way it went.
pub fn run(m: *Model, ids: []const u32, vision: ?Model.VisionRows, readout: Readout) !mlx.mlx_array {
    const p = m.plans orelse return m.forwardServed(ids, vision, readout);
    const probe: Probe = .{ .rows = ids.len, .readout = readout, .keep_hidden = m.keep_hidden, .kv = m.kv, .n = m.pos };
    // A forward that splices an image's rows is not planned; an image request's decode past its prompt splices none.
    const tables = (if (!m.splicesImage(ids, vision)) p.tables(probe) else null) orelse return wide(m, p, ids, vision, readout);
    const id = root(probe);
    try core.beginExecution(tables, id);
    const logits = m.forwardServed(ids, vision, readout) catch |e| {
        core.abortExecution();
        return if (core.executionMiss()) |miss| missed(m, p, probe, miss) else e;
    };
    core.endExecution() catch |e| {
        _ = mlx.mlx_array_free(logits);
        return if (core.executionMiss()) |miss| missed(m, p, probe, miss) else e;
    };
    return logits;
}

/// A forward of a planned load that no root holds, in the region the load measured (D97): a prefill chunk (an image's
/// splice included), a round wider than a verify block (D37, D42), a request at another KV width than the load planned.
/// An ask past the region refuses the forward by name (`WideFull`).
fn wide(m: *Model, p: *const Plans, ids: []const u32, vision: ?Model.VisionRows, readout: Readout) !mlx.mlx_array {
    const at = m.pos;
    // A load-width chunk in the deep zone read out by its last row, as a served chunk is and as the load recorded its
    // classes, runs padded to its class's recording, so it tops out where that did; another readout keeps D97's measure.
    if (p.deep > 0 and vision == null and ids.len == p.region_rows and at >= p.deep and readout == .last) {
        const class = p.pads.get(m.wideClass(@intCast(at), @intCast(ids.len), readout)) orelse {
            log.err("[plan] a forward of {d} rows at {d} is of no class the load recorded\n", .{ ids.len, at });
            return error.Hy4WidePlanMiss;
        };
        try core.beginWidePadded(class.passes);
    } else try core.beginWide();
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

/// DFly2's draft of `t1` (`Drafter.draft`): an execution of the load's draft root where it planned one for the
/// request's KV width, else on the pool.
pub fn runDraft(m: *Model, d: *const Drafter, t1: u32) !Drafter.Draft {
    const p = m.plans orelse return d.draft(m, t1);
    const probe = draftProbe(d, &m.config, m.kv, m.pos);
    const tables = p.tables(probe) orelse return d.draft(m, t1);
    try core.beginExecution(tables, root(probe));
    var drafted = d.draft(m, t1) catch |e| {
        core.abortExecution();
        return if (core.executionMiss()) |miss| missed(m, p, probe, miss) else e;
    };
    core.endExecution() catch |e| {
        drafted.deinit();
        return if (core.executionMiss()) |miss| missed(m, p, probe, miss) else e;
    };
    return drafted;
}

fn missed(m: *Model, p: *Plans, probe: Probe, miss: plan.binder.Miss) error{PlanMiss} {
    p.misses += 1;
    m.reset();
    log.warn("[plan] miss: a {s} of {d} rows at {d} ({t}, keep {}) asked {d} B against {d} B ({t}, frame {d}/{d}, ordinal {d})\n", .{ if (probe.phase == .draft) "draft" else "forward", probe.rows, probe.n, probe.readout, probe.keep_hidden, miss.asked, miss.cap, miss.kind, miss.id.site, miss.id.key, miss.ordinal });
    const passes = recordChecked(m, p.book.gpa, probe, p.reach) catch |e| {
        log.warn("[plan] the missed shape was not recorded ({t}); it misses again until a load plans it\n", .{e});
        return error.PlanMiss;
    };
    defer p.book.gpa.free(passes.traces);
    const held = core.arenaLayout() orelse {
        for (passes.traces) |*t| t.deinit();
        return error.PlanMiss;
    };
    const learned = p.book.learnAll(root(probe), passes.traces, held) catch |e| {
        log.warn("[plan] the missed shape was recorded and not learned ({t})\n", .{e});
        return error.PlanMiss;
    };
    switch (learned) {
        .swapped => {
            p.learned += 1;
            log.warn("[plan] the missed shape is learned in place: the next such forward is served\n", .{});
        },
        .next_load => log.warn("[plan] the missed shape needs more arena than the load holds: the next load plans it\n", .{}),
    }
    // Kept either way, so a restart plans it too.
    if (p.cache) |*c| c.keep(p.book.gpa, shapeOf(probe)) catch |e| log.warn("[plan] the missed shape was not kept in {s} ({t}): a restart misses it again\n", .{ c.name(), e });
    return error.PlanMiss;
}

const testing = std.testing;

test "hy4 plan: the decode positions take every residue of a group below and past each edge, the first group and the end, each once" {
    var c = try hy4_config.parseText(hy4_config.release_file);
    _ = &c;
    const got = try decodePositions(testing.allocator, &c, 1 << 20);
    defer testing.allocator.free(got);
    // A group of 4, the window of 512, the threshold of 2048, the top-k's split at 131,072 groups and the last position.
    const want = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 508, 509, 510, 511, 512, 513, 514, 515, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 524_283, 524_284, 524_285, 524_286, 524_287, 524_288, 524_289, 524_290, 1_048_572, 1_048_573, 1_048_574, 1_048_575 };
    try testing.expectEqualSlices(usize, &want, got);
}

/// A kept region of two classes of two passes, each pass with asks and a choice: what a decoder walks.
fn keptForTest(gpa: std.mem.Allocator) !Kept {
    const k: Kept = .{ .measured = .{ .rows = 512, .need = .{ .scratch = 4096, .host = 64 }, .probes = 9, .classes = 2 }, .check = .{ .scratch = 2048, .host = 32 }, .deep = 2048, .keys = try gpa.dupe(u64, &.{ 11, 22 }), .classes = try gpa.alloc(Plans.Class, 2) };
    for (k.classes, 0..) |*c, i| {
        c.* = .{ .rows = 512, .at = 4000 + i, .need = 1024, .passes = try gpa.alloc(core.PadPass, 2) };
        for (c.passes, 0..) |*ps, j| ps.* = .{ .sizes = try gpa.dupe(u64, &.{ 256, 512 + j }), .choices = try gpa.dupe(plan.TraceChoice, &.{.{ .site = @intCast(0xabcd00 + 2 * i + j), .value = 1, .at = 1 }}) };
    }
    return k;
}

fn decodeAndFree(gpa: std.mem.Allocator, bytes: []const u8) !void {
    var k = try decodeRegion(gpa, bytes);
    k.deinit(gpa);
}

test "hy4 plan: a kept region's decoder frees all it made when any of its allocations fails" {
    const a = testing.allocator;
    var k = try keptForTest(a);
    defer k.deinit(a);
    const bytes = try encodeRegion(a, k);
    defer a.free(bytes);
    try testing.checkAllAllocationFailures(a, decodeAndFree, .{bytes});
}

test "hy4 plan: a kept region whose value does not fit its field is refused as damaged under a whole checksum, never narrowed" {
    const a = testing.allocator;
    var k = try keptForTest(a);
    defer k.deinit(a);
    const bytes = try encodeRegion(a, k);
    defer a.free(bytes);
    var site: [8]u8 = undefined;
    std.mem.writeInt(u64, &site, 0xabcd00, .little);
    const at = std.mem.indexOf(u8, bytes, &site).?;
    std.mem.writeInt(u64, bytes[at..][0..8], 1 << 32, .little);
    std.mem.writeInt(u64, bytes[bytes.len - 8 ..][0..8], std.hash.Wyhash.hash(region_version, bytes[0 .. bytes.len - 8]), .little);
    try testing.expectError(error.Hy4RegionFileDamaged, decodeRegion(a, bytes));
}
