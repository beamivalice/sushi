//! Hy4-Flash's microbenches at the release's shapes, on synthetic banks. `hy4.span_linears` times a layer's attention
//! gate and `W_o` through trellis banks (the STQ pack's trunk) over a prefill chunk: one launch each per attention span
//! against one over the chunk's whole spans (`hy4_forward.linearSpans`). Run by `sushi-ubench`, never by a test; the
//! bytes of both arms are asserted by `hy4 attention spans` in `hy4_forward.zig`. `hy4.attn_core` times the attention
//! core of one layer's chunk on Sushi Cooler's stream: the per-span op chain over kv8 keys at f32 products against the
//! matrix-unit core (`hy4_attn`: the keys read to bf16 once, one launch a span), the GEMMs of its useful work and a
//! read of the bytes it must read.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const kv_quant = @import("../kv_quant.zig");
const fwd = @import("hy4_forward.zig");
const hy4_latent = @import("hy4_latent.zig");
const hy4_attn = @import("hy4_attn.zig");
const report = @import("../ubench_report.zig");
const api = @import("sushi_exl3");
const bench = @import("sushi_exl3_ubench").bench_util;

const kernels = api.kernels;
const Arr = mlx.mlx_array;

pub const benches = [_]report.Bench{ span_linears, attn_core };

const dim = 3072;

/// A layer kind's attention width (heads x v) and the rows its spans take.
const Kind = struct { name: []const u8, width: usize, span: usize };
const KINDS = [_]Kind{
    .{ .name = "sliding", .width = 64 * 256, .span = 256 },
    .{ .name = "gathered", .width = 32 * 256, .span = 32 },
};

pub const span_linears: report.Bench = .{
    .name = "hy4.span_linears",
    .scope = .hy4,
    .what = "a layer's attention gate (3072 -> heads x 256) and W_o (back to 3072) through trellis banks over a prefill chunk: one launch of each per attention span (256 rows on a sliding layer, 32 on a sparse layer past its selection) against one over the chunk, with TFLOP/s",
    .unit = .us,
    .steps = 4,
    .axes = &.{
        .{ .name = "kind", .values = &.{ "sliding", "gathered" }, .numeric = false },
        .{ .name = "k", .values = &.{ "4", "6" } },
        .{ .name = "rows", .values = &.{ "1024", "4096" } },
    },
    .arms = &.{ "per_span", "whole" },
    .ref = "per_span",
    .run = runSpanLinears,
};

/// Copies of each bank a chain cycles through, so a step never reads the previous step's bank from cache.
const COPIES = 2;

const Bank = struct { w: Arr, suh: Arr, svh: Arr };

const Ctx = struct {
    s: mlx.mlx_stream,
    kind: Kind,
    x0: Arr,
    gate: [COPIES]Bank,
    o: [COPIES]Bank,
    link_ns: f64,
};

fn project(s: mlx.mlx_stream, x: Arr, b: Bank) !Arr {
    return api.trunk.matmul(s, x, b.w, b.suh, b.svh, kernels.SPEC1_W15, false);
}

/// One layer's gate then `W_o` over `rows`, a launch of each per span (`per_span`) or over every row at once.
fn spanStep(c: *const Ctx, s: mlx.mlx_stream, i: usize, x: Arr, per_span: bool) !Arr {
    const ci = i % COPIES;
    const t: usize = @intCast(mlx.getShape(x)[0]);
    if (!per_span) {
        const g = try project(s, x, c.gate[ci]);
        defer _ = mlx.mlx_array_free(g);
        return project(s, g, c.o[ci]);
    }
    var parts: [128]Arr = undefined;
    var n: usize = 0;
    defer for (parts[0..n]) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var r0: usize = 0;
    while (r0 < t) : (r0 += c.kind.span) {
        var xs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xs);
        try mlx.check(mlx.mlx_slice(&xs, x, &[_]c_int{ @intCast(r0), 0 }, 2, &[_]c_int{ @intCast(@min(t, r0 + c.kind.span)), dim }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const g = try project(s, xs, c.gate[ci]);
        defer _ = mlx.mlx_array_free(g);
        parts[n] = try project(s, g, c.o[ci]);
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_concatenate_axis(&y, vec, 0, s));
    return y;
}

const ArmCtx = struct { c: *const Ctx, per_span: bool };

fn armStep(ac: ArmCtx, s: mlx.mlx_stream, i: usize, x: Arr) anyerror!Arr {
    return spanStep(ac.c, s, i, x, ac.per_span);
}

const Sample = struct { c: *const Ctx, steps: usize };

fn sample(smp: Sample, arm: usize) anyerror!f64 {
    kernels.setDecodeParams(kernels.SPEC1_W15);
    const ns = try bench.chainNs(smp.c.s, smp.c.x0, smp.steps, ArmCtx{ .c = smp.c, .per_span = arm == 0 }, armStep);
    return (@as(f64, @floatFromInt(ns)) - smp.c.link_ns) / @as(f64, @floatFromInt(smp.steps));
}

fn newBank(a: std.mem.Allocator, rnd: std.Random, s: mlx.mlx_stream, in: usize, out: usize, n: u32, owned: *std.ArrayList(Arr)) !Bank {
    const words = try a.alloc(u16, (in / 16) * (out / 16) * n);
    bench.randomWords(rnd, words);
    const suh_v = try a.alloc(u16, in);
    @memset(suh_v, api.format.f32ToF16Bits(1.0 / @sqrt(@as(f32, @floatFromInt(in)))));
    const svh_v = try a.alloc(u16, out);
    @memset(svh_v, api.format.f32ToF16Bits(1.0));
    const b: Bank = .{
        .w = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, @intCast(in / 16), @intCast(out / 16), @intCast(n) }, 4, .uint16),
        .suh = mlx.mlx_array_new_data(suh_v.ptr, &[_]c_int{ 1, @intCast(in) }, 2, .float16),
        .svh = mlx.mlx_array_new_data(svh_v.ptr, &[_]c_int{ 1, @intCast(out) }, 2, .float16),
    };
    try owned.appendSlice(a, &.{ b.w, b.suh, b.svh });
    _ = s;
    return b;
}

fn runSpanLinears(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x4879);
    const rnd = prng.random();
    try kernels.SPEC1_W15.requireTable();
    for (KINDS) |kind| {
        if (!listed(run.axis("kind"), kind.name)) continue;
        for (run.axis("k")) |k_text| {
            const n: u32 = if (std.mem.eql(u8, k_text, "4")) 64 else if (std.mem.eql(u8, k_text, "6")) 96 else return error.BadK;
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const a = arena.allocator();
            var owned: std.ArrayList(Arr) = .empty;
            defer for (owned.items) |x| {
                _ = mlx.mlx_array_free(x);
            };
            var c: Ctx = .{ .s = s, .kind = kind, .x0 = undefined, .gate = undefined, .o = undefined, .link_ns = 0 };
            for (0..COPIES) |ci| {
                c.gate[ci] = try newBank(a, rnd, s, dim, kind.width, n, &owned);
                c.o[ci] = try newBank(a, rnd, s, kind.width, dim, n, &owned);
            }
            const ev = mlx.mlx_vector_array_new_data(owned.items.ptr, owned.items.len);
            defer _ = mlx.mlx_vector_array_free(ev);
            try mlx.check(mlx.mlx_eval(ev));
            for (run.axis("rows")) |rows_text| {
                const rows = try std.fmt.parseInt(usize, rows_text, 10);
                if (rows / kind.span > 128) return error.TooManySpans;
                const xs = try a.alloc(u16, rows * dim);
                for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) - 0.5)) >> 16);
                c.x0 = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(rows), dim }, 2, .bfloat16);
                defer _ = mlx.mlx_array_free(c.x0);
                const steps = run.steps(4);
                var t: [5]u64 = undefined;
                for (&t) |*v| v.* = try bench.chainNs(s, c.x0, steps, {}, bench.linkOnly);
                c.link_ns = bench.median(&t);
                const label = try std.fmt.allocPrint(run.a, "{s} [3072 -> {d} -> 3072] K{s} rows {d}", .{ kind.name, kind.width, k_text, rows });
                try run.interleave(label, Sample{ .c = &c, .steps = steps }, sample);
                const flop: f64 = 4.0 * @as(f64, @floatFromInt(rows * dim * kind.width));
                for ([_][]const u8{ "per_span", "whole" }) |arm| {
                    if (run.hasArm(arm)) try run.metric(arm, "TFLOP/s", "TFLOP/s", flop / run.lastMedianNs(arm) / 1e3);
                }
            }
        }
    }
}

fn listed(values: []const []const u8, name: []const u8) bool {
    for (values) |v| if (std.mem.eql(u8, v, name)) return true;
    return false;
}

// ---- hy4.attn_core: one layer's attention core over a prefill chunk, the chain and the core against their floor ----

const nope = 192;
const rope = 64;
const v_dim = 256;
const latent = 512;
const window = 512;
/// The keys a sparse row reads past its selection: 512 groups of 4.
const picks = 2048;
const ctx_keys = 32768;

/// A layer kind's heads and the rows a span takes.
const CoreKind = struct { name: []const u8, heads: c_int, span: c_int, gathered: bool };
const CORE_KINDS = [_]CoreKind{
    .{ .name = "sliding", .heads = 64, .span = 256, .gathered = false },
    .{ .name = "gathered", .heads = 32, .span = 32, .gathered = true },
};

pub const attn_core: report.Bench = .{
    .name = "hy4.attn_core",
    .scope = .hy4,
    .what = "one layer's attention core over a prefill chunk ending at 32K keys, on Sushi Cooler's stream at f32 (dk 576, dv 512, absorbed): the per-span op chain over kv8 keys (sliding: 256-row spans over S + 511 keys; gathered: 32-row spans, each row its 2,048 picked keys) against the matrix-unit core over the same spans (the kv8 keys read to bf16 once a layer), their useful work as four batched GEMMs (f32 and bf16) and a read of the bytes it must read; ms per layer",
    .unit = .us,
    .steps = 2,
    .axes = &.{
        .{ .name = "kind", .values = &.{ "sliding", "gathered" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1024", "4096" } },
    },
    .arms = &.{ "chain", "fused", "gemm_f32", "gemm_bf16", "read" },
    .ref = "chain",
    .run = runAttnCore,
};

const CoreCtx = struct {
    s: mlx.mlx_stream,
    kind: CoreKind,
    rows: c_int,
    /// The chain's and the core's inputs: the queries [H, T, 256] f32, the keys as the state stores them (kv8 codes, scales,
    /// biases; rope rows BF16), the gate rows, kv_b and its widening, the sinks; a gathered layer's picks.
    q: Arr,
    codes: Arr,
    scales: Arr,
    biases: Arr,
    k_pe: Arr,
    gate: Arr,
    kv_b32: Arr,
    attn: fwd.Attn,
    geo: fwd.AttnGeometry,
    flat: Arr = .{ .ctx = null },
    keep: Arr = .{ .ctx = null },
    /// The GEMM arms' operands, f32 and bf16: queries [H, T, 576], keys [H, 576, K], probabilities [H, T, K], values
    /// [K, 512], absorb [H, T, 192] x [H, 192, 512]; value-out reads [H, 512, 256] of the keys.
    g: [2][6]Arr,
    /// The read arm's buffer, the bytes a fused kernel must read and write once.
    floor: Arr,
    link_ns: [5]f64 = @splat(0),
};

fn keysOf(c: *const CoreCtx) c_int {
    return if (c.kind.gathered) picks else window;
}

/// The chain over the chunk, span by span, as `hy4_model.attendSpan` runs it where the core is off.
fn chain(c: *const CoreCtx, s: mlx.mlx_stream, q: Arr) !Arr {
    var outer: Ops = .{ .s = s };
    defer outer.deinit();
    var parts: [128]Arr = undefined;
    var n: usize = 0;
    var r0: c_int = 0;
    while (r0 < c.rows) : (r0 += c.kind.span) {
        const r1 = @min(c.rows, r0 + c.kind.span);
        const rows = r1 - r0;
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const qs = try o.slice(q, 1, r0, r1);
        const in: fwd.SpanIn = .{ .gate = try o.slice(c.gate, 0, r0, r1), .kv_b32 = c.kv_b32 };
        const y = if (c.kind.gathered) blk: {
            const idx = try o.slice(c.flat, 0, r0 * picks, r1 * picks);
            const lat = try o.cast(try o.own(try kv_quant.dequantizeAffine(s, try o.take(c.codes, idx, 0), try o.take(c.scales, idx, 0), try o.take(c.biases, idx, 0), 64, 8)), .float32);
            break :blk try fwd.attendRowsGated(&o, in, qs, lat, try o.take(c.k_pe, idx, 0), try o.slice(c.keep, 0, r0, r1), c.attn, c.geo, null);
        } else blk: {
            // The window buffer holds the 511 rows before the chunk, then the chunk's: span rows r0..r1 read r0..r1+511.
            const k1 = r1 + window - 1;
            const lat = try o.cast(try o.own(try kv_quant.dequantizeAffine(s, try o.slice(c.codes, 0, r0, k1), try o.slice(c.scales, 0, r0, k1), try o.slice(c.biases, 0, r0, k1), 64, 8)), .float32);
            break :blk try fwd.attendGated(&o, in, qs, lat, try o.slice(c.k_pe, 0, r0, k1), c.attn, c.geo, try fwd.causalMask(&o, rows, k1 - r0, window - 1, window), null);
        };
        parts[n] = try outer.own(try o.result(y));
        n += 1;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

/// The matrix-unit core over the same spans: the layer's kv8 rows read to bf16 once, then a launch a span.
fn fused(c: *const CoreCtx, s: mlx.mlx_stream, q: Arr) !Arr {
    var outer: Ops = .{ .s = s };
    defer outer.deinit();
    var parts: [128]Arr = undefined;
    var n: usize = 0;
    const keys: hy4_attn.Keys = .{
        .c = try outer.own(try hy4_latent.rangeBf16(s, c.codes, c.scales, c.biases, 64)),
        .k_pe = c.k_pe,
    };
    var r0: c_int = 0;
    while (r0 < c.rows) : (r0 += c.kind.span) {
        const r1 = @min(c.rows, r0 + c.kind.span);
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const qs = try o.slice(q, 1, r0, r1);
        const in: fwd.SpanIn = .{ .gate = try o.slice(c.gate, 0, r0, r1), .kv_b32 = c.kv_b32 };
        const core: fwd.CoreKeys = if (c.kind.gathered)
            .{ .picks = .{ .keys = keys, .rows = try o.reshape(try o.slice(c.flat, 0, r0 * picks, r1 * picks), &.{ r1 - r0, picks }), .keep = try o.slice(c.keep, 0, r0, r1) } }
        else
            .{ .range = .{ .keys = .{ .c = try o.slice(keys.c, 0, r0, r1 + window - 1), .k_pe = try o.slice(keys.k_pe, 0, r0, r1 + window - 1) }, .p0 = window - 1, .window = window } };
        parts[n] = try outer.own(try o.result(try fwd.attendCoreGated(&o, in, qs, core, c.attn, c.geo, null)));
        n += 1;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

/// The useful work as four batched GEMMs: absorb, scores, the value sum, value-out.
fn gemms(g: [6]Arr, s: mlx.mlx_stream) !Arr {
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const a = try o.binary(.mm, g[4], g[5]);
    const sc = try o.binary(.mm, g[0], g[1]);
    const pv = try o.binary(.mm, g[2], g[3]);
    const vo = try o.binary(.mm, pv, try o.slice(try o.slice(g[1], 1, 0, latent), 2, 0, v_dim));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    // One output that waits on all four.
    try mlx.check(mlx.mlx_add(&out, try corner(&o, sc), try o.binary(.add, try corner(&o, a), try corner(&o, vo)), s));
    return out;
}

/// `x`'s first element of each batch, [H, 1, 1].
fn corner(o: *Ops, x: Arr) !Arr {
    return o.slice(try o.slice(x, 2, 0, 1), 1, 0, 1);
}

const CoreArm = struct { c: *const CoreCtx, arm: usize };

fn coreStep(ac: CoreArm, s: mlx.mlx_stream, _: usize, x: Arr) anyerror!Arr {
    return switch (ac.arm) {
        0 => chain(ac.c, s, x),
        1 => fused(ac.c, s, x),
        2, 3 => blk: {
            var g = ac.c.g[ac.arm - 2];
            g[0] = x;
            break :blk gemms(g, s);
        },
        else => blk: {
            var y = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_sum_axis(&y, x, 0, true, s));
            break :blk y;
        },
    };
}

fn armInput(c: *const CoreCtx, arm: usize) Arr {
    return switch (arm) {
        0, 1 => c.q,
        2, 3 => c.g[arm - 2][0],
        else => c.floor,
    };
}

const CoreSample = struct { c: *const CoreCtx, steps: usize };

fn coreSample(smp: CoreSample, arm: usize) anyerror!f64 {
    const ns = try bench.chainNs(smp.c.s, armInput(smp.c, arm), smp.steps, CoreArm{ .c = smp.c, .arm = arm }, coreStep);
    return (@as(f64, @floatFromInt(ns)) - smp.c.link_ns[arm]) / @as(f64, @floatFromInt(smp.steps));
}

fn hostArr(a: std.mem.Allocator, rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, scale: f32) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    if (dt == .float32) {
        const v = try a.alloc(f32, n);
        for (v) |*e| e.* = rnd.floatNorm(f32) * scale;
        return mlx.newData(v.ptr, shape, .float32);
    }
    const v = try a.alloc(u16, n);
    for (v) |*e| e.* = @truncate(@as(u32, @bitCast(rnd.floatNorm(f32) * scale)) >> 16);
    return mlx.newData(v.ptr, shape, dt);
}

fn runAttnCore(run: *report.Run) anyerror!void {
    // On for the process from here, as a Cooler load turns it on.
    mlx.engine.enable();
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    var prng = std.Random.DefaultPrng.init(0x4879_a77e);
    const rnd = prng.random();
    for (CORE_KINDS) |kind| {
        if (!listed(run.axis("kind"), kind.name)) continue;
        for (run.axis("rows")) |rows_text| {
            const rows = try std.fmt.parseInt(c_int, rows_text, 10);
            var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena.deinit();
            const a = arena.allocator();
            var o: Ops = .{ .s = s };
            defer o.deinit();
            const h = kind.heads;
            const key_rows: c_int = if (kind.gathered) ctx_keys else rows + window - 1;
            const lat_rows = try o.own(try hostArr(a, rnd, &.{ key_rows, latent }, .bfloat16, 1));
            var qk = try kv_quant.quantizeAffine(s, lat_rows, 64, 8);
            defer qk.deinit();
            const geo: fwd.AttnGeometry = .{ .heads = h, .nope = nope, .rope = rope, .v = v_dim, .latent = latent, .theta = 1e4, .eps = 1e-5 };
            const kv_b = try o.own(try hostArr(a, rnd, &.{ h * (nope + v_dim), latent }, .bfloat16, 0.02));
            const tiny = try o.own(try hostArr(a, rnd, &.{ 16, 16 }, .bfloat16, 1));
            const lin = fwd.Lin.dense(tiny);
            var c: CoreCtx = .{
                .s = s,
                .kind = kind,
                .rows = rows,
                .q = try o.own(try hostArr(a, rnd, &.{ h, rows, nope + rope }, .float32, 0.1)),
                .codes = qk.q,
                .scales = qk.scales,
                .biases = qk.biases,
                .k_pe = try o.own(try hostArr(a, rnd, &.{ key_rows, rope }, .bfloat16, 1)),
                .gate = try o.own(try hostArr(a, rnd, &.{ rows, h * v_dim }, .bfloat16, 1)),
                .kv_b32 = try o.cast(kv_b, .float32),
                .attn = .{ .q_a = lin, .q_a_norm = tiny, .q_b = lin, .kv_a = lin, .kv_a_norm = tiny, .kv_b = kv_b, .gate = lin, .o = lin, .post_norm = tiny, .sinks = try o.own(try hostArr(a, rnd, &.{h}, .float32, 1)) },
                .geo = geo,
                .g = undefined,
                .floor = undefined,
            };
            if (kind.gathered) {
                const idx = try a.alloc(u32, @as(usize, @intCast(rows)) * picks);
                for (idx, 0..) |*e, i| e.* = rnd.uintLessThan(u32, ctx_keys / 4) * 4 + @as(u32, @intCast(i % 4));
                c.flat = try o.own(try mlx.newData(idx.ptr, &.{@intCast(idx.len)}, .uint32));
                const keep = try a.alloc(bool, idx.len);
                @memset(keep, true);
                c.keep = try o.own(try mlx.newData(keep.ptr, &.{ rows, picks }, .bool_));
            }
            const k = keysOf(&c);
            for (&c.g, [_]mlx.mlx_dtype{ .float32, .bfloat16 }) |*g, dt| g.* = .{
                try o.own(try hostArr(a, rnd, &.{ h, rows, latent + rope }, dt, 0.1)),
                try o.own(try hostArr(a, rnd, &.{ h, latent + rope, k }, dt, 0.1)),
                try o.own(try hostArr(a, rnd, &.{ h, rows, k }, dt, 0.01)),
                try o.own(try hostArr(a, rnd, &.{ k, latent }, dt, 0.1)),
                try o.own(try hostArr(a, rnd, &.{ h, rows, nope }, dt, 0.1)),
                try o.own(try hostArr(a, rnd, &.{ h, nope, latent }, dt, 0.1)),
            };
            // What a fused kernel reads and writes once: the queries (f32), the distinct keys as stored, kv_b, the
            // gate rows in and the gated rows out (BF16).
            const key_bytes: usize = @as(usize, @intCast(@min(key_rows, if (kind.gathered) ctx_keys else key_rows))) * (latent + latent / 64 * 4 + rope * 2);
            const bytes: usize = @as(usize, @intCast(h * rows * (nope + rope))) * 4 + key_bytes + @as(usize, @intCast(h * (nope + v_dim) * latent)) * 2 + 2 * @as(usize, @intCast(rows * h * v_dim)) * 2;
            c.floor = try o.own(try hostArr(a, rnd, &.{@intCast(bytes / 4)}, .float32, 1));
            try mlx.check(mlx.mlx_array_eval(c.kv_b32));
            const steps = run.steps(2);
            for (0..5) |arm| {
                var t: [5]u64 = undefined;
                for (&t) |*v| v.* = try bench.chainNs(s, armInput(&c, arm), steps, {}, bench.linkOnly);
                c.link_ns[arm] = bench.median(&t);
            }
            const label = try std.fmt.allocPrint(run.a, "{s} H{d} rows {d} keys {d}", .{ kind.name, h, rows, k });
            try run.interleave(label, CoreSample{ .c = &c, .steps = steps }, coreSample);
            const flop: f64 = 2.0 * @as(f64, @floatFromInt(h)) * @as(f64, @floatFromInt(rows)) * (@as(f64, nope * latent) + @as(f64, @floatFromInt(k)) * (latent + rope + latent) + @as(f64, latent * v_dim));
            for ([_][]const u8{ "chain", "fused", "gemm_f32", "gemm_bf16" }) |arm| {
                if (run.hasArm(arm)) try run.metric(arm, "TFLOP/s", "useful TFLOP/s", flop / run.lastMedianNs(arm) / 1e3);
            }
            if (run.hasArm("read")) try run.metric("read", "GB/s", "GB/s", @as(f64, @floatFromInt(bytes)) / run.lastMedianNs("read"));
        }
    }
}
