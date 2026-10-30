//! DeepSeek-V4.1-Flash's indexer, attention span and top-k kernels at the release's shapes (32 index heads x 128,
//! top-512, blocks of 8; 64 attention heads x 512, window 128), on synthetic rows: the decode indexer's stages, the
//! indexer's and the row stores' working set, the attention span arms, and the decode top-k. Run by `sushi-ubench`,
//! never by a test.
const std = @import("std");
const mlx = @import("../mlx.zig");
const io_util = @import("../io_util.zig");
const report = @import("../ubench_report.zig");
const cfg_mod = @import("dsv41_config.zig");
const sc_mod = @import("dsv41_scope.zig");
const qat_mod = @import("dsv41_qat.zig");
const kernels = @import("dsv41_kernels.zig");
const model = @import("dsv41_model.zig");

pub const benches = [_]report.Bench{ index_decode, index_peak, index_block_peak, rows_growth, attn_span, picks_span, attn_epilogue, topk, hc_expand, hc_stages };

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const none = sc_mod.none;
const ci = sc_mod.ci;
const evalAll = sc_mod.evalAll;

const alloc = std.heap.page_allocator;
const neg_inf = -std.math.inf(f32);

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

fn fakeModel(m: *model.Model) void {
    m.fused = true;
    m.trace = null;
}

fn axisInts(run: *const report.Run, name: []const u8, out: []usize) ![]const usize {
    const values = run.axis(name);
    for (values, 0..) |text, i| out[i] = try std.fmt.parseInt(usize, text, 10);
    return out[0..values.len];
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}

/// The lowest and highest peak over a bench's samples: a peak read once is raised by when freed buffers are
/// released, so the floor and the worst case differ from run to run.
const Range = struct {
    lo: usize = std.math.maxInt(usize),
    hi: usize = 0,

    fn add(r: *Range, v: usize) void {
        r.lo = @min(r.lo, v);
        r.hi = @max(r.hi, v);
    }

    fn metrics(r: Range, run: *report.Run, arm: []const u8, name: []const u8) !void {
        const lo = try std.fmt.allocPrint(run.a, "{s} min", .{name});
        const hi = try std.fmt.allocPrint(run.a, "{s} max", .{name});
        try run.metric(arm, lo, "MB", mb(r.lo));
        try run.metric(arm, hi, "MB", mb(r.hi));
    }
};

/// MLX bytes beyond the active ones that a sample's builds and evaluations hold at their peak.
const Peak = struct {
    active: usize = 0,

    fn begin(gpu: mlx.mlx_stream) Peak {
        _ = mlx.mlx_synchronize(gpu);
        _ = mlx.mlx_clear_cache();
        _ = mlx.mlx_reset_peak_memory();
        var p: Peak = .{};
        _ = mlx.mlx_get_active_memory(&p.active);
        return p;
    }

    fn end(p: Peak) usize {
        var pk: usize = 0;
        _ = mlx.mlx_get_peak_memory(&pk);
        return pk -| p.active;
    }
};

fn mb(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

// ── dsv41.index_decode ──────────────────────────────────────────────────

pub const index_decode: report.Bench = .{
    .name = "dsv41.index_decode",
    .scope = .dsv41,
    .what = "a decode block's indexer stages over packed keys: scores, candidate mask applied, top-512, candidate-source blocks, scores + top-k",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "keys", .values = &.{ "32768", "65536", "131072", "262144" } },
        .{ .name = "rows", .values = &.{ "1", "2", "4" } },
    },
    .arms = &.{ "scores", "mask", "topk", "cand", "scores_topk" },
    .ref = "scores",
    .run = runIndexDecode,
};

const IndexStage = enum { scores, mask, topk, cand, scores_topk };

const IndexDecode = struct {
    gpu: mlx.mlx_stream,
    fake: *const model.Model,
    nb: usize,
    L: usize,
    q: A,
    w: A,
    lens: A,
    codes: A,
    scl: A,
    mode: qat_mod.Qat,
    group: usize,
    full: A,
    mask: A,
    masked: A,
};

fn sampleIndexDecode(c: *const IndexDecode, arm: usize) anyerror!f64 {
    var t = Scope.init(alloc, c.gpu);
    defer t.deinit();
    var sw = io_util.Stopwatch.init(io());
    switch (@as(IndexStage, @fromBackingInt(@intCast(arm)))) {
        .scores => try evalAll(&.{try kernels.indexScoresPacked(&t, c.q, c.codes, c.scl, c.w, 0.01, c.nb - c.L, 1, c.mode, c.group)}),
        .mask => {
            var o = mlx.mlx_array_new();
            const rep = try t.res(mlx.mlx_repeat_axis(&o, c.mask, 8, -1, t.s), &o);
            try evalAll(&.{try t.where(try t.slice(rep, -1, 0, c.nb), c.full, try t.f(neg_inf))});
        },
        .topk => try evalAll(&.{try model.topkRows(&t, c.fake, c.masked, 512)}),
        .cand => try evalAll(&.{try model.candidateBlocks(&t, c.fake, 0, c.full, c.lens, 8, 2048)}),
        .scores_topk => {
            const scores = try kernels.indexScoresPacked(&t, c.q, c.codes, c.scl, c.w, 0.01, c.nb - c.L, 1, c.mode, c.group);
            try evalAll(&.{try model.topkRows(&t, c.fake, scores, 512)});
        },
    }
    return @floatFromInt(sw.read());
}

fn runIndexDecode(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    var fake: model.Model = undefined;
    fakeModel(&fake);
    var keys_buf: [8]usize = undefined;
    var rows_buf: [8]usize = undefined;
    for (try axisInts(run, "keys", &keys_buf)) |nb| {
        var sc = Scope.init(alloc, gpu);
        defer sc.deinit();
        var rows = try model.streamKeys(&sc, nb, 4096, 0x4d0);
        defer rows.deinit();
        const pk = rows.pack.?;
        for (try axisInts(run, "rows", &rows_buf)) |L| {
            const q = try sc.astype(try model.sc_mod_randn(&sc, &.{ ci(L), 32, 128 }, 0x5d0), .float32);
            const w = try sc.astype(try model.sc_mod_randn(&sc, &.{ ci(L), 32 }, 0x6d0), .bfloat16);
            const lens_h = try alloc.alloc(i32, L);
            defer alloc.free(lens_h);
            for (lens_h, 0..) |*v, i| v.* = @intCast(nb - L + i + 1);
            const lens = try sc.ints(lens_h, &.{ ci(L), 1 });
            const codes = try sc.slice(rows.buf, 0, 0, nb);
            const scl = try sc.slice(rows.sbuf, 0, 0, nb);
            const full = try kernels.indexScoresPacked(&sc, q, codes, scl, w, 0.01, nb - L, 1, pk.mode, pk.group);
            const mask = try model.candidateBlocks(&sc, &fake, 0, full, lens, 8, 2048);
            if (mask.ctx == null) return error.KeysUnderTheCandidateBlocks;
            var o = mlx.mlx_array_new();
            const rep = try sc.res(mlx.mlx_repeat_axis(&o, mask, 8, -1, sc.s), &o);
            const masked = try sc.where(try sc.slice(rep, -1, 0, nb), full, try sc.f(neg_inf));
            try evalAll(&.{ q, w, lens, codes, scl, full, mask, masked });
            const c: IndexDecode = .{ .gpu = gpu, .fake = &fake, .nb = nb, .L = L, .q = q, .w = w, .lens = lens, .codes = codes, .scl = scl, .mode = pk.mode, .group = pk.group, .full = full, .mask = mask, .masked = masked };
            const label = try std.fmt.allocPrint(run.a, "keys {d} rows {d}", .{ nb, L });
            try run.interleave(label, &c, sampleIndexDecode);
        }
    }
}

// ── dsv41.index_peak ────────────────────────────────────────────────────

pub const index_peak: report.Bench = .{
    .name = "dsv41.index_peak",
    .scope = .dsv41,
    .what = "one wide span's indexer at the release shapes: the whole scores against the streamed top-k, MLX peak beyond the inputs",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "span", .values = &.{ "enc", "ced" }, .numeric = false },
        .{ .name = "ctx", .values = &.{ "16384", "65536", "262144", "524288", "1048576" } },
    },
    .arms = &.{ "whole", "streamed" },
    .ref = "whole",
    .run = runIndexPeak,
};

const SpanArgs = struct { rows: *const model.Rows, nb: usize, q: A, w: A, wscale: f32, p0: usize, r: usize, k: usize, bs: usize, with_blocks: bool, lens: A };

const SpanSample = struct { gpu: mlx.mlx_stream, a: SpanArgs, tile: usize, peak: [2]Range = @splat(.{}) };

fn sampleSpan(c: *SpanSample, arm: usize) anyerror!f64 {
    var fake: model.Model = undefined;
    fakeModel(&fake);
    const p = Peak.begin(c.gpu);
    var t = Scope.init(alloc, c.gpu);
    defer t.deinit();
    const a = c.a;
    var sw = io_util.Stopwatch.init(io());
    if (arm == 0) {
        const keys = try a.rows.rows(&t, a.nb);
        const scores = try model.indexScores(&t, &fake, a.q, keys, a.w, a.wscale, a.p0, a.r);
        const idx = try model.topkRows(&t, &fake, scores, a.k);
        try evalAll(&.{ idx, if (a.with_blocks) try model.candidateBlocks(&t, &fake, 0, scores, a.lens, a.bs, 2048) else idx });
    } else {
        const got = try model.indexStream(&t, a.rows, a.nb, a.q, a.w, a.wscale, a.p0, a.r, a.k, c.tile, none, a.bs, if (a.with_blocks) 2048 else 0, a.lens);
        try evalAll(&.{ got.idx, if (a.with_blocks) got.cand else got.idx });
    }
    const ns = sw.read();
    c.peak[arm].add(p.end());
    return @floatFromInt(ns);
}

fn runIndexPeak(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const tile: usize = @intCast(cfg_mod.index_stream_tile);
    var ctx_buf: [8]usize = undefined;
    // Encoder owners (ratio 2, the 256-row span) and decoder owners (ratio 1, the 128-row CED tail).
    const Span = struct { name: []const u8, L: usize, r: usize };
    for ([_]Span{ .{ .name = "enc", .L = 256, .r = 2 }, .{ .name = "ced", .L = 128, .r = 1 } }) |g| {
        if (!listed(run.axis("span"), g.name)) continue;
        for (try axisInts(run, "ctx", &ctx_buf)) |ctx| {
            const nb = ctx / g.r;
            var sc = Scope.init(alloc, gpu);
            defer sc.deinit();
            var rows = try model.streamKeys(&sc, nb, 4096, 0x4d0);
            defer rows.deinit();
            const q = try sc.astype(try model.sc_mod_randn(&sc, &.{ ci(g.L), 32, 128 }, 0x5d0), .float32);
            const w = try sc.astype(try model.sc_mod_randn(&sc, &.{ ci(g.L), 32 }, 0x6d0), .bfloat16);
            const lens_h = try alloc.alloc(i32, g.L);
            defer alloc.free(lens_h);
            for (lens_h, 0..) |*v, i| v.* = @intCast((ctx - g.L + i + 1) / g.r);
            const lens = try sc.ints(lens_h, &.{ ci(g.L), 1 });
            try evalAll(&.{ q, w, rows.buf, rows.sbuf, lens });
            const a: SpanArgs = .{ .rows = &rows, .nb = nb, .q = q, .w = w, .wscale = 0.01, .p0 = ctx - g.L, .r = g.r, .k = 512, .bs = 8, .with_blocks = nb / 8 > 2048, .lens = lens };
            var c: SpanSample = .{ .gpu = gpu, .a = a, .tile = tile };
            const label = try std.fmt.allocPrint(run.a, "{s} ctx {d}", .{ g.name, ctx });
            try run.interleave(label, &c, sampleSpan);
            try c.peak[0].metrics(run, "whole", "peak");
            try c.peak[1].metrics(run, "streamed", "peak");
        }
    }
}

// ── dsv41.index_block_peak ──────────────────────────────────────────────

pub const index_block_peak: report.Bench = .{
    .name = "dsv41.index_block_peak",
    .scope = .dsv41,
    .what = "a 16-row decode or verify block's indexer over every packed key, with and without the candidate mask a later layer applies: MLX peak beyond the inputs",
    .unit = .ms,
    .steps = 1,
    .axes = &.{.{ .name = "ctx", .values = &.{ "65536", "262144", "524288", "1048576" } }},
    .arms = &.{ "packed", "masked" },
    .ref = "packed",
    .run = runIndexBlockPeak,
};

const block_rows = 16;

const BlockSample = struct { gpu: mlx.mlx_stream, a: SpanArgs, mask: A, peak: [2]Range = @splat(.{}) };

fn sampleBlock(c: *BlockSample, arm: usize) anyerror!f64 {
    var fake: model.Model = undefined;
    fakeModel(&fake);
    const p = Peak.begin(c.gpu);
    var t = Scope.init(alloc, c.gpu);
    defer t.deinit();
    const a = c.a;
    const pk = a.rows.pack.?;
    var sw = io_util.Stopwatch.init(io());
    var scores = try kernels.indexScoresPacked(&t, a.q, try t.slice(a.rows.buf, 0, 0, a.nb), try t.slice(a.rows.sbuf, 0, 0, a.nb), a.w, a.wscale, a.p0, a.r, pk.mode, pk.group);
    if (arm == 1) {
        var o = mlx.mlx_array_new();
        const rep = try t.res(mlx.mlx_repeat_axis(&o, c.mask, ci(a.bs), -1, t.s), &o);
        scores = try t.where(try t.slice(rep, -1, 0, a.nb), scores, try t.f(neg_inf));
    }
    try evalAll(&.{try model.topkRows(&t, &fake, scores, a.k)});
    const ns = sw.read();
    c.peak[arm].add(p.end());
    return @floatFromInt(ns);
}

fn runIndexBlockPeak(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    var fake: model.Model = undefined;
    fakeModel(&fake);
    var ctx_buf: [8]usize = undefined;
    for (try axisInts(run, "ctx", &ctx_buf)) |ctx| {
        var sc = Scope.init(alloc, gpu);
        defer sc.deinit();
        const L = block_rows;
        var rows = try model.streamKeys(&sc, ctx, 4096, 0x4d0);
        defer rows.deinit();
        const q = try sc.astype(try model.sc_mod_randn(&sc, &.{ L, 32, 128 }, 0x5d0), .float32);
        const w = try sc.astype(try model.sc_mod_randn(&sc, &.{ L, 32 }, 0x6d0), .bfloat16);
        var lens_h: [L]i32 = undefined;
        for (&lens_h, 0..) |*v, i| v.* = @intCast(ctx - L + i + 1);
        const lens = try sc.ints(&lens_h, &.{ L, 1 });
        const pk = rows.pack.?;
        const full = try kernels.indexScoresPacked(&sc, q, try sc.slice(rows.buf, 0, 0, ctx), try sc.slice(rows.sbuf, 0, 0, ctx), w, 0.01, ctx - L, 1, pk.mode, pk.group);
        const mask = try model.candidateBlocks(&sc, &fake, 0, full, lens, 8, 2048);
        if (mask.ctx == null) return error.KeysUnderTheCandidateBlocks;
        try evalAll(&.{ q, w, rows.buf, rows.sbuf, lens, mask });
        const a: SpanArgs = .{ .rows = &rows, .nb = ctx, .q = q, .w = w, .wscale = 0.01, .p0 = ctx - L, .r = 1, .k = 512, .bs = 8, .with_blocks = false, .lens = lens };
        var c: BlockSample = .{ .gpu = gpu, .a = a, .mask = mask };
        const label = try std.fmt.allocPrint(run.a, "ctx {d}", .{ctx});
        try run.interleave(label, &c, sampleBlock);
        try c.peak[0].metrics(run, "packed", "peak");
        try c.peak[1].metrics(run, "masked", "peak");
    }
}

// ── dsv41.rows_growth ───────────────────────────────────────────────────

pub const rows_growth: report.Bench = .{
    .name = "dsv41.rows_growth",
    .scope = .dsv41,
    .what = "a full row store growing by one 2048-row chunk: the time, the bytes it holds and the peak beyond them",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "store", .values = &.{ "comp", "index" }, .numeric = false },
        .{ .name = "used", .values = &.{ "262144", "1048576" } },
    },
    .arms = &.{"grow"},
    .ref = "grow",
    .run = runRowsGrowth,
};

const GrowSample = struct { gpu: mlx.mlx_stream, base: *const model.Rows, chunk: A, peak: Range = .{} };

/// The store's rows are shared, not copied: growing frees only the copy's own handles.
fn sampleGrow(c: *GrowSample, _: usize) anyerror!f64 {
    var rows = c.base.*;
    rows.buf = mlx.mlx_array_new();
    rows.sbuf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&rows.buf, c.base.buf));
    try mlx.check(mlx.mlx_array_set(&rows.sbuf, c.base.sbuf));
    defer rows.deinit();
    const p = Peak.begin(c.gpu);
    var t = Scope.init(alloc, c.gpu);
    defer t.deinit();
    var sw = io_util.Stopwatch.init(io());
    try rows.append(&t, c.chunk);
    try evalAll(&.{ rows.buf, rows.sbuf });
    const ns = sw.read();
    c.peak.add(p.end());
    return @floatFromInt(ns);
}

fn runRowsGrowth(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const Store = struct { name: []const u8, width: usize, mode: qat_mod.Qat, group: usize };
    var used_buf: [8]usize = undefined;
    for ([_]Store{ .{ .name = "comp", .width = 512, .mode = .fp4_e4m3, .group = 16 }, .{ .name = "index", .width = 128, .mode = .fp4_ue8m0, .group = 32 } }) |g| {
        if (!listed(run.axis("store"), g.name)) continue;
        for (try axisInts(run, "used", &used_buf)) |n| {
            var sc = Scope.init(alloc, gpu);
            defer sc.deinit();
            var rows: model.Rows = .{ .width = g.width, .dtype = .bfloat16, .pack = .{ .mode = g.mode, .group = g.group } };
            defer rows.deinit();
            const base = try sc.astype(try model.sc_mod_randn(&sc, &.{ 4096, ci(g.width) }, 0x7d0), .bfloat16);
            const full = try sc.broadcast(try sc.expand(base, 0), &.{ ci(n / 4096), 4096, ci(g.width) });
            try rows.append(&sc, try sc.reshape(full, &.{ ci(n), ci(g.width) }));
            const chunk = try sc.slice(base, 0, 0, 2048);
            try evalAll(&.{ rows.buf, rows.sbuf, chunk });
            var c: GrowSample = .{ .gpu = gpu, .base = &rows, .chunk = chunk };
            const label = try std.fmt.allocPrint(run.a, "{s} used {d}", .{ g.name, n });
            try run.interleave(label, &c, sampleGrow);
            try run.metric("grow", "held", "MB", mb(n * (g.width / 2 + g.width / g.group)));
            try c.peak.metrics(run, "grow", "peak beyond held");
        }
    }
}

// ── dsv41.attn_span ─────────────────────────────────────────────────────

pub const attn_span: report.Bench = .{
    .name = "dsv41.attn_span",
    .scope = .dsv41,
    .what = "one attention span at the release geometry (64 heads x 512, window 128): the f32 composed core, the f16 wide arm, the fused kernel, the fused kernel on the matrix units (NAX GPUs); MLX peak beyond the inputs",
    .unit = .ms,
    .steps = 1,
    .axes = &.{.{ .name = "span", .values = &.{ "256k512", "128k512", "256k0" }, .numeric = false }},
    .arms = &.{ "core", "wide", "fused", "nax" },
    .ref = "core",
    .run = runAttnSpan,
};

const AttnArm = enum { core, wide, fused, nax };

const AttnSample = struct { gpu: mlx.mlx_stream, d: model.AttnData, c: model.AttnCase, peak: [4]Range = @splat(.{}) };

fn sampleAttn(s: *AttnSample, arm: usize) anyerror!f64 {
    const scale = 1.0 / @sqrt(@as(f32, 512));
    const d = s.d;
    const c = s.c;
    var sc = Scope.init(alloc, s.gpu);
    defer sc.deinit();
    const p = Peak.begin(s.gpu);
    var sw = io_util.Stopwatch.init(io());
    if (arm == @backingInt(AttnArm.nax) and !kernels.attnNaxFits(c.H, c.hd)) return error.NaxUnavailable;
    const y = try switch (@as(AttnArm, @fromBackingInt(@intCast(arm)))) {
        .core => model.attnCore(&sc, d.sink, d.q, d.keys, d.band, d.comp, d.topk, scale, .bfloat16),
        .wide => kernels.attnWide(&sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .bfloat16),
        .fused => kernels.attnFused(&sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .bfloat16, null, false),
        .nax => kernels.attnFused(&sc, d.sink, d.q, d.keys, c.wp, c.window, d.comp, d.topk, scale, .bfloat16, null, true),
    };
    try mlx.check(mlx.mlx_array_eval(y));
    const ns = sw.read();
    s.peak[arm].add(p.end());
    return @floatFromInt(ns);
}

fn runAttnSpan(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const Case = struct { name: []const u8, c: model.AttnCase };
    const cases = [_]Case{
        .{ .name = "256k512", .c = .{ .L = 256, .wp = 127, .K = 512, .nc = 4096, .holes = false } },
        .{ .name = "128k512", .c = .{ .L = 128, .wp = 127, .K = 512, .nc = 4096, .holes = false } },
        .{ .name = "256k0", .c = .{ .L = 256, .wp = 127, .K = 0, .nc = 0 } },
    };
    for (cases) |cs| {
        if (!listed(run.axis("span"), cs.name)) continue;
        var base = Scope.init(alloc, gpu);
        defer base.deinit();
        const d = try model.attnCaseData(&base, cs.c, 0xa77);
        try evalAll(&.{ d.q, d.keys, d.band, d.sink });
        if (cs.c.K > 0) try evalAll(&.{ d.comp, d.topk });
        var s: AttnSample = .{ .gpu = gpu, .d = d, .c = cs.c };
        const label = try std.fmt.allocPrint(run.a, "span {d} K {d}", .{ cs.c.L, cs.c.K });
        try run.interleave(label, &s, sampleAttn);
        for (std.enums.values(AttnArm)) |arm| try s.peak[@backingInt(arm)].metrics(run, @tagName(arm), "peak");
    }
}

// ── dsv41.attn_epilogue ─────────────────────────────────────────────────

pub const attn_epilogue: report.Bench = .{
    .name = "dsv41.attn_epilogue",
    .scope = .dsv41,
    .what = "one prefill span's heads as the grouped wo_a takes them, at the release geometry (64 heads x 512, 8 groups, QAT on, span at 30,720): the fused kernel then the rotation, fp8 round trip and transpose as ops, against the kernel's store doing all three; MLX peak beyond the inputs",
    .unit = .ms,
    .steps = 1,
    .axes = &.{.{ .name = "span", .values = &.{ "256k512", "128k512", "256k0" }, .numeric = false }},
    .arms = &.{ "chain", "fold" },
    .ref = "chain",
    .run = runAttnEpilogue,
};

const epi_pos0: usize = 30720;
const epi_groups = 8;

const EpiSample = struct { gpu: mlx.mlx_stream, d: model.AttnData, c: model.AttnCase, rows: ?*const model.Rows, freqs: A, pos: A, peak: [2]Range = @splat(.{}) };

fn sampleEpilogue(s: *EpiSample, arm: usize) anyerror!f64 {
    const scale = 1.0 / @sqrt(@as(f32, 512));
    const c = s.c;
    var sc = Scope.init(alloc, s.gpu);
    defer sc.deinit();
    const p = Peak.begin(s.gpu);
    var sw = io_util.Stopwatch.init(io());
    const picks: model.Picks = if (s.rows) |r| try model.packedPicks(&sc, r, s.d.topk, true) else .{};
    const epi: kernels.Epilogue = .{ .freqs = s.freqs, .rd = 64, .pos0 = epi_pos0, .qat = true, .og = epi_groups };
    const y = if (arm == 1)
        try model.fusedHeads(&sc, s.d.sink, s.d.q, s.d.keys, c.wp, c.window, picks, scale, .bfloat16, epi, false)
    else blk: {
        const heads = try model.fusedHeads(&sc, s.d.sink, s.d.q, s.d.keys, c.wp, c.window, picks, scale, .bfloat16, null, false);
        const rot = try kernels.ropeTail(&sc, heads, s.freqs, s.pos, 64, true, .none, null);
        const grouped = try sc.reshape(rot, &.{ ci(c.L), epi_groups, ci(c.H * c.hd / epi_groups) });
        break :blk try sc.transpose(try qat_mod.qat(&sc, grouped, .fp8_ue8m0, 32), &.{ 1, 0, 2 });
    };
    try mlx.check(mlx.mlx_array_eval(y));
    const ns = sw.read();
    s.peak[arm].add(p.end());
    return @floatFromInt(ns);
}

fn runAttnEpilogue(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const Case = struct { name: []const u8, c: model.AttnCase };
    const cases = [_]Case{
        .{ .name = "256k512", .c = .{ .L = 256, .wp = 127, .K = 512, .nc = 4096, .holes = false } },
        .{ .name = "128k512", .c = .{ .L = 128, .wp = 127, .K = 512, .nc = 4096, .holes = false } },
        .{ .name = "256k0", .c = .{ .L = 256, .wp = 127, .K = 0, .nc = 0 } },
    };
    var f: [32]f32 = undefined;
    qat_mod.invFreqs(&f, 64, 65536, 160000, 16, 32, 1);
    for (cases) |cs| {
        if (!listed(run.axis("span"), cs.name)) continue;
        var base = Scope.init(alloc, gpu);
        defer base.deinit();
        const d = try model.attnCaseData(&base, cs.c, 0xa77);
        var store: ?model.Rows = if (cs.c.K > 0) try model.packedStore(&base, d.comp) else null;
        defer if (store) |*r| r.deinit();
        const freqs = try base.floats(&f, &.{32});
        const pos = try model.posArray(&base, epi_pos0, cs.c.L, 1);
        try evalAll(&.{ d.q, d.keys, d.sink, freqs, pos });
        if (store) |r| try evalAll(&.{ d.topk, r.buf, r.sbuf });
        var s: EpiSample = .{ .gpu = gpu, .d = d, .c = cs.c, .rows = if (store) |*r| r else null, .freqs = freqs, .pos = pos };
        const label = try std.fmt.allocPrint(run.a, "span {d} K {d}", .{ cs.c.L, cs.c.K });
        try run.interleave(label, &s, sampleEpilogue);
        try s.peak[0].metrics(run, "chain", "peak");
        try s.peak[1].metrics(run, "fold", "peak");
    }
}

// ── dsv41.picks_span ────────────────────────────────────────────────────

pub const picks_span: report.Bench = .{
    .name = "dsv41.picks_span",
    .scope = .dsv41,
    .what = "one prefill span's compressed picks and fused attention at the release geometry: the picked rows unpacked and gathered first, against the kernel decoding them from the packed store, on the SIMD or the matrix units (NAX GPUs); MLX peak beyond the inputs",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "span", .values = &.{ "256k512", "128k512" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "8192", "262144" } },
    },
    .arms = &.{ "gather", "packed", "packed_nax" },
    .ref = "gather",
    .run = runPicksSpan,
};

const PicksSample = struct { gpu: mlx.mlx_stream, d: model.AttnData, c: model.AttnCase, rows: *const model.Rows, peak: [3]Range = @splat(.{}) };

fn samplePicks(s: *PicksSample, arm: usize) anyerror!f64 {
    const scale = 1.0 / @sqrt(@as(f32, 512));
    var sc = Scope.init(alloc, s.gpu);
    defer sc.deinit();
    const p = Peak.begin(s.gpu);
    var sw = io_util.Stopwatch.init(io());
    const nax = arm == 2;
    if (nax and !kernels.attnNaxFits(s.c.H, s.c.hd)) return error.NaxUnavailable;
    const picks = try model.packedPicks(&sc, s.rows, s.d.topk, arm >= 1);
    const y = try model.fusedHeads(&sc, s.d.sink, s.d.q, s.d.keys, s.c.wp, s.c.window, picks, scale, .bfloat16, null, nax);
    try mlx.check(mlx.mlx_array_eval(y));
    const ns = sw.read();
    s.peak[arm].add(p.end());
    return @floatFromInt(ns);
}

fn runPicksSpan(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    var rows_buf: [8]usize = undefined;
    const Case = struct { name: []const u8, L: usize };
    for ([_]Case{ .{ .name = "256k512", .L = 256 }, .{ .name = "128k512", .L = 128 } }) |cs| {
        if (!listed(run.axis("span"), cs.name)) continue;
        for (try axisInts(run, "rows", &rows_buf)) |nc| {
            const c: model.AttnCase = .{ .L = cs.L, .wp = 127, .K = 512, .nc = nc, .holes = false };
            var base = Scope.init(alloc, gpu);
            defer base.deinit();
            const d = try model.attnCaseData(&base, c, 0xa77);
            var store = try model.packedStore(&base, d.comp);
            defer store.deinit();
            try evalAll(&.{ d.q, d.keys, d.sink, d.topk, store.buf, store.sbuf });
            var s: PicksSample = .{ .gpu = gpu, .d = d, .c = c, .rows = &store };
            const label = try std.fmt.allocPrint(run.a, "span {d} rows {d}", .{ cs.L, nc });
            try run.interleave(label, &s, samplePicks);
            try s.peak[0].metrics(run, "gather", "peak");
            try s.peak[1].metrics(run, "packed", "peak");
            try s.peak[2].metrics(run, "packed_nax", "peak");
        }
    }
}

// ── dsv41.topk ──────────────────────────────────────────────────────────

pub const topk: report.Bench = .{
    .name = "dsv41.topk",
    .scope = .dsv41,
    .what = "the decode top-k over index scores: one threadgroup a row against the served split, per row of work in one eval of one or eight",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "ctx", .values = &.{ "65536", "262144", "524288" } },
        .{ .name = "shape", .values = &.{ "ratio2", "ratio1", "blocks", "gathered" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "4", "16" } },
        .{ .name = "batch", .values = &.{ "1", "8" } },
    },
    .arms = &.{ "one", "served" },
    .ref = "one",
    .run = runTopk,
};

const TopkSample = struct { gpu: mlx.mlx_stream, xs: *const [8]A, batch: usize, k: usize };

fn sampleTopk(c: *const TopkSample, arm: usize) anyerror!f64 {
    var t = Scope.init(alloc, c.gpu);
    defer t.deinit();
    var outs: [8]A = undefined;
    var sw = io_util.Stopwatch.init(io());
    for (outs[0..c.batch], c.xs[0..c.batch]) |*o, x| o.* = if (arm == 0) try kernels.topkIndicesSplit(&t, x, c.k, 1) else try kernels.topkIndices(&t, x, c.k);
    try evalAll(outs[0..c.batch]);
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(c.batch));
}

fn runTopk(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    var ctx_buf: [8]usize = undefined;
    var rows_buf: [8]usize = undefined;
    var batch_buf: [8]usize = undefined;
    const ctxs = try axisInts(run, "ctx", &ctx_buf);
    const Shape = struct { name: []const u8, n: usize, k: usize };
    for (ctxs, 0..) |ctx, ci_| {
        // The candidate source's block maxima and a later source's gathered keys, then the two score widths.
        const shapes = [_]Shape{ .{ .name = "ratio2", .n = ctx / 2, .k = 512 }, .{ .name = "ratio1", .n = ctx, .k = 512 }, .{ .name = "blocks", .n = ctx / 8, .k = 2048 }, .{ .name = "gathered", .n = 16384, .k = 512 } };
        for (shapes) |sh| {
            if (!listed(run.axis("shape"), sh.name)) continue;
            // The gathered width does not depend on the context: time it once.
            if (std.mem.eql(u8, sh.name, "gathered") and ci_ > 0) continue;
            for (try axisInts(run, "rows", &rows_buf)) |L| {
                var sc = Scope.init(alloc, gpu);
                defer sc.deinit();
                var xs: [8]A = undefined;
                for (&xs, 0..) |*x, i| x.* = try sc.add(try kernels.randn(&sc, &.{ ci(L), ci(sh.n) }, .float32, 90 + i, 1.0), try sc.f(@floatFromInt(i)));
                try evalAll(&xs);
                for (try axisInts(run, "batch", &batch_buf)) |batch| {
                    if (batch > xs.len) return error.BatchOverEight;
                    const c: TopkSample = .{ .gpu = gpu, .xs = &xs, .batch = batch, .k = sh.k };
                    const ctx_label = if (std.mem.eql(u8, sh.name, "gathered")) "" else try std.fmt.allocPrint(run.a, "ctx {d} ", .{ctx});
                    const label = try std.fmt.allocPrint(run.a, "{s}{s} rows {d} x{d}", .{ ctx_label, sh.name, L, batch });
                    try run.interleave(label, &c, sampleTopk);
                }
            }
        }
    }
}

// ── dsv41.hc_expand ─────────────────────────────────────────────────────

pub const hc_expand: report.Bench = .{
    .name = "dsv41.hc_expand",
    .scope = .dsv41,
    .what = "the hyper-connection expand of a wide chunk (hc 4 x 5120, bf16 stream): the op chain against the fused kernel, alone (the layer tail) and followed by the stage's mix and norm kernels (the FFN stage)",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "rows", .values = &.{ "512", "2048" } },
        .{ .name = "stage", .values = &.{ "expand", "ffn" }, .numeric = false },
    },
    .arms = &.{ "chain", "fused" },
    .ref = "chain",
    .run = runHcExpand,
};

const HcExpandData = struct { x: A, y: A, post: A, comb: A, cpre: A, w: A, hw: kernels.HcWeights, eps: A };

const HcExpandSample = struct { gpu: mlx.mlx_stream, d: HcExpandData, ffn: bool };

fn sampleHcExpand(s: *HcExpandSample, arm: usize) anyerror!f64 {
    const d = s.d;
    var sc = Scope.init(alloc, s.gpu);
    defer sc.deinit();
    var sw = io_util.Stopwatch.init(io());
    const x1 = if (arm == 0) try model.hcExpandOps(&sc, d.x, d.y, d.post, d.comb) else try kernels.hcExpandFused(&sc, d.x, d.y, d.post, d.comb);
    var ev: [2]A = .{ x1, x1 };
    if (s.ffn) {
        const n = try kernels.hcNorm(&sc, try kernels.hcMix(&sc, x1, d.hw, d.eps), x1, d.cpre, d.w, 20, d.eps);
        ev[1] = n.h;
    }
    try evalAll(&ev);
    return @floatFromInt(sw.read());
}

fn runHcExpand(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const D = 5120;
    const mix = 24;
    try kernels.probeHc(alloc, gpu, 4, D, 20);
    var rows_buf: [8]usize = undefined;
    for (try axisInts(run, "rows", &rows_buf)) |L| {
        var sc = Scope.init(alloc, gpu);
        defer sc.deinit();
        const l: c_int = ci(L);
        const f32t = mlx.mlx_dtype.float32;
        const d: HcExpandData = .{
            .x = try sc.astype(try kernels.randn(&sc, &.{ l, 4, D }, f32t, 201, 1.5), .bfloat16),
            .y = try sc.astype(try kernels.randn(&sc, &.{ l, D }, f32t, 202, 1.0), .bfloat16),
            .post = try kernels.randn(&sc, &.{ l, 4 }, f32t, 203, 0.7),
            .comb = try kernels.randn(&sc, &.{ l, 4, 4 }, f32t, 204, 0.5),
            .cpre = try sc.op1(mlx.mlx_abs, try kernels.randn(&sc, &.{ l, 4 }, f32t, 205, 0.6)),
            .w = try sc.astype(try sc.add(try kernels.randn(&sc, &.{D}, f32t, 206, 0.2), try sc.f(1.0)), .bfloat16),
            .hw = .{ .fn_w = try kernels.randn(&sc, &.{ mix, 4 * D }, f32t, 207, 0.02), .scale = try sc.floats(&.{ 0.3, 0.5, 0.7 }, &.{3}), .base = try kernels.randn(&sc, &.{mix}, f32t, 208, 0.5) },
            .eps = try kernels.hcEps(&sc, 1e-6, 1e-6),
        };
        try evalAll(&.{ d.x, d.y, d.post, d.comb, d.cpre, d.w, d.hw.fn_w, d.hw.scale, d.hw.base, d.eps });
        const want = try sc_mod.hostF32(try sc.astype(try model.hcExpandOps(&sc, d.x, d.y, d.post, d.comb), f32t));
        const got = try sc_mod.hostF32(try sc.astype(try kernels.hcExpandFused(&sc, d.x, d.y, d.post, d.comb), f32t));
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) run.invalid("the fused expand's bytes differ from the chain's");
        for ([_][]const u8{ "expand", "ffn" }) |stage| {
            if (!listed(run.axis("stage"), stage)) continue;
            var s: HcExpandSample = .{ .gpu = gpu, .d = d, .ffn = std.mem.eql(u8, stage, "ffn") };
            const label = try std.fmt.allocPrint(run.a, "{s} rows {d}", .{ stage, L });
            try run.interleave(label, &s, sampleHcExpand);
        }
    }
}

// ── dsv41.hc_stages ─────────────────────────────────────────────────────

pub const hc_stages: report.Bench = .{
    .name = "dsv41.hc_stages",
    .scope = .dsv41,
    .what = "the wide chunk's hyper-connection dispatches one by one (hc 4 x 5120, bf16 stream): mix, norm, split, expand, the attention stage (mix + norm) and the FFN stage (expand + mix + norm), against a copy of the stream; the arms are the mix kernel's row x mix tile",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "rows", .values = &.{ "512", "2048" } },
        .{ .name = "stage", .values = &.{ "copy", "mix", "norm", "split", "expand", "attn", "ffn" }, .numeric = false },
    },
    .arms = &.{ "r4m6", "r2m12" },
    .ref = "r4m6",
    .run = runHcStages,
};

const hc_stage_names = [_][]const u8{ "copy", "mix", "norm", "split", "expand", "attn", "ffn" };
const hc_tiles = [_]kernels.HcMixTile{ .{ .rows = 4, .cols = 6 }, .{ .rows = 2, .cols = 12 } };
const hc_reps = 8;

const HcStagesSample = struct { gpu: mlx.mlx_stream, d: HcExpandData, z: A, stage: []const u8 };

fn sampleHcStages(s: *HcStagesSample, arm: usize) anyerror!f64 {
    const d = s.d;
    const tile = hc_tiles[arm];
    var sc = Scope.init(alloc, s.gpu);
    defer sc.deinit();
    var ev: [hc_reps]A = undefined;
    var sw = io_util.Stopwatch.init(io());
    for (&ev) |*e| {
        if (std.mem.eql(u8, s.stage, "copy")) {
            e.* = try sc.add(d.x, try sc.astype(try sc.f(1.0), .bfloat16));
        } else if (std.mem.eql(u8, s.stage, "mix")) {
            e.* = try kernels.hcMixTiled(&sc, d.x, d.hw, d.eps, tile);
        } else if (std.mem.eql(u8, s.stage, "norm")) {
            e.* = (try kernels.hcNorm(&sc, s.z, d.x, d.cpre, d.w, 20, d.eps)).h;
        } else if (std.mem.eql(u8, s.stage, "split")) {
            e.* = (try kernels.hcSplitZ(&sc, s.z, 4, 20, d.eps)).comb;
        } else if (std.mem.eql(u8, s.stage, "expand")) {
            e.* = try kernels.hcExpandFused(&sc, d.x, d.y, d.post, d.comb);
        } else {
            const x1 = if (std.mem.eql(u8, s.stage, "ffn")) try kernels.hcExpandFused(&sc, d.x, d.y, d.post, d.comb) else d.x;
            e.* = (try kernels.hcNorm(&sc, try kernels.hcMixTiled(&sc, x1, d.hw, d.eps, tile), x1, d.cpre, d.w, 20, d.eps)).h;
        }
    }
    try evalAll(&ev);
    return @as(f64, @floatFromInt(sw.read())) / hc_reps;
}

fn runHcStages(run: *report.Run) anyerror!void {
    const gpu = mlx.gpuStream();
    const D = 5120;
    const mix = 24;
    try kernels.probeHc(alloc, gpu, 4, D, 20);
    var rows_buf: [8]usize = undefined;
    for (try axisInts(run, "rows", &rows_buf)) |L| {
        var sc = Scope.init(alloc, gpu);
        defer sc.deinit();
        const l: c_int = ci(L);
        const f32t = mlx.mlx_dtype.float32;
        const d: HcExpandData = .{
            .x = try sc.astype(try kernels.randn(&sc, &.{ l, 4, D }, f32t, 201, 1.5), .bfloat16),
            .y = try sc.astype(try kernels.randn(&sc, &.{ l, D }, f32t, 202, 1.0), .bfloat16),
            .post = try kernels.randn(&sc, &.{ l, 4 }, f32t, 203, 0.7),
            .comb = try kernels.randn(&sc, &.{ l, 4, 4 }, f32t, 204, 0.5),
            .cpre = try sc.op1(mlx.mlx_abs, try kernels.randn(&sc, &.{ l, 4 }, f32t, 205, 0.6)),
            .w = try sc.astype(try sc.add(try kernels.randn(&sc, &.{D}, f32t, 206, 0.2), try sc.f(1.0)), .bfloat16),
            .hw = .{ .fn_w = try kernels.randn(&sc, &.{ mix, 4 * D }, f32t, 207, 0.02), .scale = try sc.floats(&.{ 0.3, 0.5, 0.7 }, &.{3}), .base = try kernels.randn(&sc, &.{mix}, f32t, 208, 0.5) },
            .eps = try kernels.hcEps(&sc, 1e-6, 1e-6),
        };
        const z = try kernels.hcMixTiled(&sc, d.x, d.hw, d.eps, hc_tiles[0]);
        try evalAll(&.{ d.x, d.y, d.post, d.comb, d.cpre, d.w, d.hw.fn_w, d.hw.scale, d.hw.base, d.eps, z });
        const zbytes = std.mem.sliceAsBytes(try sc_mod.hostF32(z));
        for (hc_tiles[1..]) |tile| {
            const got = try sc_mod.hostF32(try kernels.hcMixTiled(&sc, d.x, d.hw, d.eps, tile));
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), zbytes)) run.invalid("a mix tile's bytes differ from the first arm's");
        }
        for (hc_stage_names) |stage| {
            if (!listed(run.axis("stage"), stage)) continue;
            var s: HcStagesSample = .{ .gpu = gpu, .d = d, .z = z, .stage = stage };
            const label = try std.fmt.allocPrint(run.a, "{s} rows {d}", .{ stage, L });
            try run.interleave(label, &s, sampleHcStages);
        }
    }
}
