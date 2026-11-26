//! The audit (docs/bill_redesign.md, stage 1e): a dry model of the release, built from its config and its 1,964
//! tensors with no pack and no byte of weight, records each phase dry at release shapes, and each recording packs into
//! the arena it would take. `SUSHI_PLAN_AUDIT=<abs file>` writes the table
//! there (one TSV row a recording).
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const plan_trace = @import("../tensor/plan/plan_trace.zig");
const plan_shape = @import("../tensor/plan/plan_shape.zig");
const plan_pack = @import("../tensor/plan/plan_pack.zig");
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_load = @import("hy4_load.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_dfly = @import("hy4_dfly.zig");

const Model = hy4_model.Model;

extern "c" fn time(t: ?*i64) i64;
extern "c" fn fflush(f: *std.c.FILE) c_int;

/// One recording dry: a frame of its own for `root`, its survivors state-owned (stage 2 gives them their owners).
const Dry = struct {
    rec: plan.Recorder,
    driver: plan.Driver,

    fn begin(self: *Dry, root: plan.FrameId) !void {
        self.rec = try plan.Recorder.begin(testing.allocator, root);
        errdefer self.rec.deinit();
        self.driver = .{ .gpa = testing.allocator };
        try core.beginRecording(.{ .dry = .{ .driver = &self.driver } }, &self.rec);
    }

    /// The recording's end, packed: the arena its asks take (`Tables.g`) where a reuse adds no barrier and where any may,
    /// and the host ring (`Tables.h`).
    fn planned(self: *Dry) !struct { g: u64, lifetimes: u64, h: u64 } {
        try core.endRecording();
        defer {
            self.driver.deinit();
            self.rec.deinit();
        }
        for (self.rec.instances.items) |*inst| for (inst.asks.items) |*as| {
            if (as.death == plan_trace.none) as.external = true;
        };
        var trace = try self.rec.end();
        defer trace.deinit();
        var why: plan_shape.Ambiguity = undefined;
        var shapes = plan_shape.merge(testing.allocator, (&trace)[0..1], .{ .refused = &why }) catch |e| {
            std.log.err("hy4 plan audit: one recording is ambiguous: frame {d}/{d}, {t} at node {d}", .{ why.id.site, why.id.key, why.why, why.node });
            return e;
        };
        defer shapes.deinit();
        var tables = try plan_pack.pack(testing.allocator, &shapes, .{});
        defer tables.deinit();
        var loose = try plan_pack.pack(testing.allocator, &shapes, .{ .level_rule = false });
        defer loose.deinit();
        return .{ .g = tables.g, .lifetimes = loose.g, .h = tables.h };
    }

    /// The recording dropped: the load's, whose asks are the model's for good.
    fn abort(self: *Dry) void {
        core.endRecording() catch {};
        self.driver.deinit();
        self.rec.deinit();
    }

    fn failed(self: *Dry, e: anyerror) anyerror {
        self.abort();
        return e;
    }
};

/// The release's tensors as a load keeps them, every one with a shape and no bytes; each Engram table resident (a
/// served load reads it by row from the checkpoint) and each token map a ramp, the host reads it.
fn dryWeights(c: *const hy4_config.Config) !model_mod.Weights {
    var w = model_mod.Weights.init(testing.allocator);
    errdefer w.deinit();
    var lines = std.mem.tokenizeScalar(u8, @embedFile("fixtures/hy4_tensors.txt"), '\n');
    while (lines.next()) |line| {
        var f = std.mem.tokenizeScalar(u8, line, ' ');
        const name = f.next().?;
        const dt = f.next().?;
        var dims: [8]c_int = undefined;
        var n: usize = 0;
        var it = std.mem.tokenizeScalar(u8, f.next().?, ',');
        while (it.next()) |d| : (n += 1) dims[n] = try std.fmt.parseInt(c_int, d, 10);
        const table = std.mem.endsWith(u8, name, "multi_head_embedding.embedding.weight");
        if (!table and !try hy4_load.keptWith(c, false, true, true, name)) continue;
        var a = mlx.mlx_array_new();
        if (std.mem.endsWith(u8, name, "lookup_table")) {
            const ramp = try testing.allocator.alloc(i32, @intCast(dims[0]));
            defer testing.allocator.free(ramp);
            for (ramp, 0..) |*v, i| v.* = @intCast(i);
            a = mlx.mlx_array_new_data(ramp.ptr, dims[0..n].ptr, @intCast(n), .int32);
        } else {
            const dtype: mlx.mlx_dtype = if (std.mem.eql(u8, dt, "BF16")) .bfloat16 else if (std.mem.eql(u8, dt, "F32")) .float32 else return error.TestUnknownDtype;
            try mlx.check(mlx.engine.freshArray(&a, dims[0..n], dtype));
        }
        try w.map.put(try testing.allocator.dupe(u8, name), a);
    }
    return w;
}

const Row = struct { phase: []const u8, rows: u64, seq: u64, packed_g: u64, lifetimes: u64, host: u64 };

/// The table as it is made, a row a recording, when `SUSHI_PLAN_AUDIT` names a file.
const Table = struct {
    rows: std.ArrayList(Row) = .empty,
    file: ?*std.c.FILE = null,
    start: i64 = 0,

    fn add(self: *Table, r: Row) !void {
        try self.rows.append(testing.allocator, r);
        const f = self.file orelse return;
        var buf: [256]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf, "{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{d}\n", .{ r.phase, r.rows, r.seq, r.packed_g, r.lifetimes, r.host, time(null) - self.start });
        _ = std.c.fwrite(line.ptr, 1, line.len, f);
        _ = fflush(f);
    }
};

/// Rows `n` of a made-up prompt: ids the tokenizer could give, spread over the vocabulary.
fn idsFrom(buf: []u32, from: u64, vocab: u32) []const u32 {
    for (buf, 0..) |*v, i| v.* = @intCast((from + i) * 7919 % vocab);
    return buf;
}

test "hy4 plan audit: a dry model of the release plans decode, verify at every block width, draft and the prefill chunk at 4k to the full context, each from a dry state and recorded on its own" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const c = try hy4_config.parseText(hy4_config.release_config);
    // The machine arms a first launch latches, latched for real before anything runs dry (the wide mix's once the
    // dry model stands: its census runs the model).
    try @import("../hc_kernels.zig").probeIdentity(s, c.hc_mult, c.dim);

    var load: Dry = undefined;
    try load.begin(.{ .site = 100 });
    var w = dryWeights(&c) catch |e| return load.failed(e);
    defer w.deinit();
    const m = Model.initWith(a, c, &w, s, .{}) catch |e| return load.failed(e);
    defer m.deinit();
    var drafter = hy4_dfly.Drafter.init(a, m, &w) catch |e| return load.failed(e);
    defer drafter.deinit();
    load.abort();
    try m.probeWide();
    m.keep_hidden = true;

    var table: Table = .{ .start = time(null) };
    defer table.rows.deinit(a);
    if (std.c.getenv("SUSHI_PLAN_AUDIT")) |path| table.file = std.c.fopen(path, "w") orelse return error.TestAuditFile;
    defer if (table.file) |f| {
        _ = std.c.fclose(f);
    };
    const chunk: usize = 4096;
    const contexts = [_]u64{ 4096, 32768, 131072, 1048576 - 64 };
    const ids = try a.alloc(u32, chunk);
    defer a.free(ids);
    const widths = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16 };
    var at: u64 = 0;
    for (contexts) |ctx| {
        // The chunk that reaches `ctx`, the widest key set, planned from a dry state of the rows before it, reserved
        // for what this context's probes reach (`Model.dryState`: no forward runs to reach it).
        const last: usize = @intCast(@min(chunk, ctx));
        {
            var d: Dry = undefined;
            try d.begin(.{ .site = 4 });
            m.dryState(@intCast(ctx - last), @intCast(ctx + 64)) catch |e| return d.failed(e);
            d.abort();
        }
        at = ctx - last;
        {
            var d: Dry = undefined;
            try d.begin(.{ .site = 1, .key = last });
            const lg = m.forwardReadout(idsFrom(ids[0..last], at, c.vocab), .none) catch |e| return d.failed(e);
            _ = mlx.mlx_array_free(lg);
            at += last;
            const p = try d.planned();
            try table.add(.{ .phase = "prefill", .rows = last, .seq = at, .packed_g = p.g, .lifetimes = p.lifetimes, .host = p.h });
        }
        // A decode, a verify block of every width (and 16 rows, past them), and a draft, each from where the last left.
        for (widths) |t| {
            var d: Dry = undefined;
            try d.begin(.{ .site = 2, .key = t });
            const lg = m.forwardReadout(idsFrom(ids[0..t], at, c.vocab), if (t == 1) .last else .all) catch |e| return d.failed(e);
            _ = mlx.mlx_array_free(lg);
            const p = try d.planned();
            at += t;
            try table.add(.{ .phase = if (t == 1) "decode" else if (t <= 8) "verify" else "rows", .rows = t, .seq = at, .packed_g = p.g, .lifetimes = p.lifetimes, .host = p.h });
        }
        {
            var d: Dry = undefined;
            try d.begin(.{ .site = 3 });
            var dr = drafter.draft(m, ids[0]) catch |e| return d.failed(e);
            dr.deinit();
            const p = try d.planned();
            try table.add(.{ .phase = "draft", .rows = c.dfly.draft + 1, .seq = at, .packed_g = p.g, .lifetimes = p.lifetimes, .host = p.h });
        }
    }
    try testing.expectEqual(contexts.len * (widths.len + 2), table.rows.items.len);
    for (table.rows.items) |r| try testing.expect(r.packed_g > 0);
}

test "hy4 plan audit: a dry model of the release builds the plans of its one-row forwards to the full context, of its image encode to the most patches it accepts, and of every draw its requests take, every structure its probes take merged with no ambiguity" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const c = try hy4_config.parseText(hy4_config.release_config);
    try @import("../hc_kernels.zig").probeIdentity(s, c.hc_mult, c.dim);
    var load: Dry = undefined;
    try load.begin(.{ .site = 100 });
    var w = dryWeights(&c) catch |e| return load.failed(e);
    defer w.deinit();
    const m = Model.initWith(a, c, &w, s, .{}) catch |e| return load.failed(e);
    defer m.deinit();
    var drafter = hy4_dfly.Drafter.init(a, m, &w) catch |e| return load.failed(e);
    defer drafter.deinit();
    load.abort();
    var tower = try @import("hy4_vision.zig").Tower.init(a, c.vision.?, &w);
    defer tower.deinit();
    var mc: model_mod.ModelConfig = .{ .model_type = "hy_v4_flash", .has_vision = true };
    mc.hy4flash = c;
    const media = @import("hy4_media.zig");
    const hy4_plan = @import("hy4_plan.zig");
    const p = try hy4_plan.record(m, a, 1 << 20, &drafter, .engine_default, null, .{ .tower = &tower, .max_patches = media.maxImagePatches(media.preproc(&mc)) }, 0);
    m.recorded = p;
    // A served load's draws read a reserved-token mask.
    const suppress = try @import("../sampler.zig").buildSuppressMask(&.{ 5, 9 }, 0, c.vocab, s);
    defer _ = mlx.mlx_array_free(suppress);
    try @import("hy4_sample_plan.zig").record(m, p, .{ .suppress = suppress });
}

fn releaseModel(a: std.mem.Allocator, s: mlx.mlx_stream, c: *const hy4_config.Config, w: *model_mod.Weights) !*Model {
    var load: Dry = undefined;
    try load.begin(.{ .site = 100 });
    w.* = dryWeights(c) catch |e| return load.failed(e);
    const m = Model.initWith(a, c.*, w, s, .{}) catch |e| {
        w.deinit();
        return load.failed(e);
    };
    load.abort();
    try m.probeWide();
    return m;
}

/// Every `[rows, depth]` of `shapes`, read out by its last row and by none, asks no more than the region a load of
/// `chunk`-row chunks measures (`hy4_plan.regionAt`).
fn expectRegionHolds(a: std.mem.Allocator, m: *Model, chunk: usize, reach: usize, shapes: []const [2]usize) !void {
    const hy4_plan = @import("hy4_plan.zig");
    const p = try a.create(hy4_plan.Plans);
    p.* = .{ .book = .{ .gpa = a, .options = .{ .level_rule = true, .level_rule_cap = hy4_plan.level_rule_cap } }, .reach = reach, .kv = .engine_default };
    defer p.deinit(a);
    const region = try hy4_plan.regionAt(m, p, chunk);
    for (shapes) |sh| for ([_]Model.Readout{ .none, .last }) |readout| {
        var passes = try hy4_plan.recordChecked(m, a, .{ .rows = sh[0], .readout = readout, .keep_hidden = m.keep_hidden, .kv = p.kv, .n = sh[1], .phase = .wide }, reach);
        defer passes.deinit();
        for (passes.traces) |*tr| {
            const need = try plan.region.need(a, tr, p.book.alignment);
            errdefer std.debug.print("a forward of {d} rows at {d} ({t}) asks scratch {d} B, host {d} B; the region of {d}-row chunks holds {d} B, host {d} B\n", .{ sh[0], sh[1], readout, need.scratch, need.host, chunk, region.scratch, region.host });
            try testing.expect(need.scratch <= region.scratch and need.host <= region.host);
        }
    };
}

test "hy4 plan audit: the release's prefill chunks, at every depth their selection span straddles the threshold at and past it, ask no more than the region their load measured (dry release)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const c = try hy4_config.parseText(hy4_config.release_config);
    try @import("../hc_kernels.zig").probeIdentity(s, c.hc_mult, c.dim);
    var w: model_mod.Weights = undefined;
    const m = try releaseModel(a, s, &c, &w);
    defer w.deinit();
    defer m.deinit();
    const thr: usize = @as(usize, c.indexer.group_topk) * c.indexer.group;
    // A chunk that ends at the threshold or crosses it, its first span cut at each row of a selection span and in the
    // periods after it, then the depths where a sliding layer's window turns.
    var shapes: std.ArrayList([2]usize) = .empty;
    defer shapes.deinit(a);
    for (0..41) |cut| for ([_]usize{ 4096, 4095 }) |t| try shapes.append(a, .{ t, thr - cut });
    for ([_]usize{ 72, 104, 520, 1000, 2040 }) |cut| try shapes.append(a, .{ 4096, thr - cut });
    try shapes.appendSlice(a, &.{ .{ 4096, 0 }, .{ 4096, 1 }, .{ 4096, @as(usize, c.window) }, .{ 4096, @as(usize, c.window) + 4 }, .{ 4095, 1028 }, .{ 4096, thr }, .{ 4096, 20000 } });
    try expectRegionHolds(a, m, 4096, 65536, shapes.items);
    try expectRegionHolds(a, m, 8192, 65536, &.{ .{ 8192, thr - 1 }, .{ 8192, thr - 4 }, .{ 8192, thr - 8 }, .{ 8192, thr - 9 }, .{ 8192, thr - 31 }, .{ 8191, thr - 8 } });
    // A chunk of 512 rows, which ends near the threshold at the depths below it.
    try expectRegionHolds(a, m, 512, 8192, &.{ .{ 512, thr - 512 }, .{ 512, thr - 256 }, .{ 512, thr - 31 }, .{ 511, thr - 511 }, .{ 300, thr - 300 }, .{ 512, thr - 8 }, .{ 512, thr - 1 }, .{ 512, 1024 } });
    // Every depth a chunk of 512 rows touches the threshold at, so the threshold cuts each of its spans at each row: the
    // region holds each, though it records one cut. The chunk a row short of it and the final span take the first and the
    // last of their spans, the last one short.
    var every: std.ArrayList([2]usize) = .empty;
    defer every.deinit(a);
    for (thr - 512..thr + 1) |n| try every.append(a, .{ 512, n });
    for ([_]usize{ 511, 33 }) |t| for (thr - t..thr + 1) |n| if (n < thr - t + 40 or n + 40 > thr) try every.append(a, .{ t, n });
    try expectRegionHolds(a, m, 512, 8192, every.items);
}
