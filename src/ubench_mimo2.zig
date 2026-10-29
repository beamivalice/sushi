//! MiMo-V2.6-Flash microbenches: global-layer decode and prefill attention, the FP8 trunk linears at the
//! checkpoint's own shapes, and the MiMo-ViT encode on a real tower. Run by `sushi-ubench`, never by a test.
const std = @import("std");
const mlx = @import("mlx.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const report = @import("ubench_report.zig");
const fp8_block = @import("fp8_block.zig");
const transformer = @import("transformer.zig");
const attn_kernels = @import("attn_kernels.zig");
const kv_quant = @import("kv_quant.zig");
const mimo_forward = @import("mimo2/mimo_forward.zig");
const model_mod = @import("model.zig");
const mimo_source = @import("mimo2/mimo_source.zig");
const mimo_vision = @import("mimo2/mimo_vision.zig");

const RowSplit = fp8_block.RowSplit;

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .boot).nanoseconds;
}

// ── mimo2.fp8_gemv ──────────────────────────────────────────────────────

/// One trunk linear at MiMo's per-rank geometry.
const Shape = struct { name: []const u8, split: RowSplit, k: c_int };

const SHAPES = [_]Shape{
    .{ .name = "qkv_global", .split = .{ .tp = 4, .parts = .{ 3072, 192, 128 } }, .k = 4096 },
    .{ .name = "qkv_sliding", .split = .{ .tp = 4, .parts = .{ 3072, 384, 256 } }, .k = 4096 },
    .{ .name = "gate_up", .split = RowSplit.dense(16384), .k = 4096 },
    .{ .name = "down", .split = RowSplit.dense(4096), .k = 16384 },
};

/// `fp8` is `project` (the GEMV to 16 rows, dequant and matmul above), `fp8_serving` is `projectServing` (the
/// matrix-unit tile at 9..128 rows, NAX chips only), `bf16` and `affine8` are the stock ops on the dequantized weight.
const Arm = enum { bf16, fp8, fp8_serving, affine8 };

pub const fp8_gemv: report.Bench = .{
    .name = "mimo2.fp8_gemv",
    .scope = .mimo2,
    .what = "FP8 trunk linear at MiMo's per-rank shapes: GEMV and tile against bf16 and affine-8, us per linear",
    .unit = .us,
    .steps = copies_default,
    .axes = &.{
        .{ .name = "shape", .values = &.{ "qkv_global", "qkv_sliding", "gate_up", "down" }, .numeric = false },
        .{ .name = "rows", .values = &.{ "1", "2", "4", "8", "9", "12", "16", "32", "64", "128", "256", "512", "2048" } },
    },
    .arms = &.{ "bf16", "fp8", "fp8_serving", "affine8" },
    .ref = "bf16",
    .run = runGemv,
};

/// Distinct weight copies one sample cycles through, so no weight stays in cache between calls.
const copies_default = 6;

const Weights = struct {
    fp8_w: []mlx.mlx_array,
    fp8_s: []mlx.mlx_array,
    /// Per copy, per output: the bf16 route's `[in, out]` view and the affine triple, as the served paths bind them.
    bf16_t: [][3]mlx.mlx_array,
    aq: [][3][3]mlx.mlx_array,

    fn init(a: std.mem.Allocator, copies: usize) !Weights {
        const fp8_w = try a.alloc(mlx.mlx_array, copies);
        const fp8_s = try a.alloc(mlx.mlx_array, copies);
        const bf16_t = try a.alloc([3]mlx.mlx_array, copies);
        const aq = try a.alloc([3][3]mlx.mlx_array, copies);
        @memset(fp8_w, .{});
        @memset(fp8_s, .{});
        @memset(bf16_t, @splat(.{}));
        @memset(aq, @splat(@splat(.{})));
        return .{ .fp8_w = fp8_w, .fp8_s = fp8_s, .bf16_t = bf16_t, .aq = aq };
    }

    fn deinit(self: *Weights) void {
        for (self.fp8_w) |x| _ = mlx.mlx_array_free(x);
        for (self.fp8_s) |x| _ = mlx.mlx_array_free(x);
        for (self.bf16_t) |row| for (row) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for (self.aq) |row| for (row) |t| for (t) |x| {
            _ = mlx.mlx_array_free(x);
        };
    }
};

fn buildWeights(a: std.mem.Allocator, s: mlx.mlx_stream, shape: Shape, copies: usize, seed: u64) !Weights {
    var bw = try Weights.init(a, copies);
    errdefer bw.deinit();
    const n: c_int = @intCast(shape.split.tp * shape.split.rowsPerRank());
    const srows: c_int = @intCast(shape.split.tp * shape.split.blocksPerRank());
    for (0..copies) |c| {
        var rk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rk);
        try mlx.check(mlx.mlx_random_key(&rk, seed + c));
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        const lo = mlx.mlx_array_new_int(0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_int(126);
        defer _ = mlx.mlx_array_free(hi);
        try mlx.check(mlx.mlx_random_randint(&bits, lo, hi, &[_]c_int{ n, shape.k }, 2, .int32, rk, s));
        try mlx.check(mlx.mlx_astype(&bw.fp8_w[c], bits, .uint8, s));
        const scale = mlx.mlx_array_new_float(3e-4);
        defer _ = mlx.mlx_array_free(scale);
        try mlx.check(mlx.mlx_full(&bw.fp8_s[c], &[_]c_int{ srows, @divExact(shape.k, 128) }, 2, scale, .float32, s));
        var dense: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
        defer for (&dense) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try fp8_block.dequantize(s, bw.fp8_w[c], bw.fp8_s[c], shape.split, &dense);
        for (0..shape.split.outputs()) |p| {
            try mlx.check(mlx.mlx_array_eval(dense[p]));
            try mlx.check(mlx.mlx_transpose(&bw.bf16_t[c][p], dense[p], s));
            var parts = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(parts);
            try mlx.check(mlx.mlx_quantize(&parts, dense[p], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
            for (0..3) |i| {
                try mlx.check(mlx.mlx_vector_array_get(&bw.aq[c][p][i], parts, i));
                try mlx.check(mlx.mlx_array_eval(bw.aq[c][p][i]));
            }
        }
        try mlx.check(mlx.mlx_array_eval(bw.fp8_w[c]));
        try mlx.check(mlx.mlx_array_eval(bw.fp8_s[c]));
    }
    return bw;
}

fn applyArm(s: mlx.mlx_stream, arm: Arm, shape: Shape, bw: *const Weights, c: usize, x: mlx.mlx_array, v: mlx.mlx_vector_array) !void {
    var outs: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    defer for (&outs) |x_| if (x_.ctx != null) {
        _ = mlx.mlx_array_free(x_);
    };
    switch (arm) {
        .fp8 => try fp8_block.project(s, x, bw.fp8_w[c], bw.fp8_s[c], shape.split, &outs),
        .fp8_serving => try fp8_block.projectServing(s, x, bw.fp8_w[c], bw.fp8_s[c], shape.split, &outs),
        .bf16 => for (0..shape.split.outputs()) |p| {
            outs[p] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&outs[p], x, bw.bf16_t[c][p], s));
        },
        .affine8 => for (0..shape.split.outputs()) |p| {
            outs[p] = mlx.mlx_array_new();
            const t = bw.aq[c][p];
            try mlx.check(mlx.mlx_quantized_matmul(&outs[p], x, t[0], t[1], t[2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
        },
    }
    for (outs[0..shape.split.outputs()]) |o| _ = mlx.mlx_vector_array_append_value(v, o);
}

/// Bytes one call reads: the weight and its scales in the arm's storage.
fn armBytes(arm: Arm, shape: Shape) f64 {
    const nk: f64 = @as(f64, @floatFromInt(shape.split.tp * shape.split.rowsPerRank())) * @as(f64, @floatFromInt(shape.k));
    return switch (arm) {
        .bf16 => 2 * nk,
        .fp8, .fp8_serving => nk + @as(f64, @floatFromInt(shape.split.tp * shape.split.blocksPerRank())) * @as(f64, @floatFromInt(shape.k)) / 128.0 * 4.0,
        .affine8 => nk + nk / 64.0 * 4.0,
    };
}

const GemvSample = struct { s: mlx.mlx_stream, io: std.Io, shape: Shape, bw: *const Weights, x: mlx.mlx_array, copies: usize };

/// Nanoseconds per linear: `copies` calls on distinct weights, evaluated together.
fn sampleGemv(smp: GemvSample, arm_index: usize) anyerror!f64 {
    const arm: Arm = @fromBackingInt(@intCast(arm_index));
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (0..smp.copies) |c| try applyArm(smp.s, arm, smp.shape, smp.bw, c, smp.x, v);
    const t0 = nowNs(smp.io);
    try mlx.check(mlx.mlx_eval(v));
    return @as(f64, @floatFromInt(nowNs(smp.io) - t0)) / @as(f64, @floatFromInt(smp.copies));
}

fn runGemv(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const nax = transformer.verifyQmmNaxAvailable();
    if (run.hasArm("fp8_serving") and !nax) {
        run.invalid("fp8_serving is the matrix-unit tile and this GPU has none: pass --arms bf16,fp8,affine8");
        return;
    }
    const armed = fp8_block.tile_nax;
    defer fp8_block.tile_nax = armed;
    fp8_block.tile_nax = nax;
    const copies = run.steps(copies_default);
    for (SHAPES, 0..) |shape, si| {
        if (!listed(run.axis("shape"), shape.name)) continue;
        var bw = try buildWeights(run.a, s, shape, copies, 0xF8 + si * 16);
        defer bw.deinit();
        for (run.axis("rows")) |rows_text| {
            const m = try std.fmt.parseInt(c_int, rows_text, 10);
            var rk = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rk);
            try mlx.check(mlx.mlx_random_key(&rk, 7));
            var xf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xf);
            try mlx.check(mlx.mlx_random_normal(&xf, &[_]c_int{ 1, m, shape.k }, 3, .float32, 0, 1, rk, s));
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
            try mlx.check(mlx.mlx_array_eval(x));
            const label = try std.fmt.allocPrint(run.a, "{s} rows {d}", .{ shape.name, m });
            try run.interleave(label, GemvSample{ .s = s, .io = io, .shape = shape, .bw = &bw, .x = x, .copies = copies }, sampleGemv);
            for (run.plan.arms) |i| {
                const arm: Arm = @fromBackingInt(@intCast(i));
                const ns = run.lastMedianNs(@tagName(arm));
                try run.metric(@tagName(arm), "GB/s", "GB/s", armBytes(arm, shape) / ns);
            }
        }
        _ = mlx.mlx_clear_cache();
    }
}

// ── mimo2.vision_encode ─────────────────────────────────────────────────

pub const vision_encode: report.Bench = .{
    .name = "mimo2.vision_encode",
    .scope = .mimo2,
    .kind = .pack,
    .what = "MiMo-ViT encode of one image on the real tower (--pack: the MiMo checkpoint), ms and scratch peak",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "grid", .values = &.{ "14x14", "68x120", "96x96" }, .numeric = false },
    },
    .arms = &.{"encode"},
    .ref = "encode",
    .run = runVision,
};

const VisionSample = struct {
    tower: *mimo_vision.MimoVision,
    io: std.Io,
    pv: mlx.mlx_array,
    gh: u32,
    gw: u32,
    first_peak: *?usize,
    steady_peak: *usize,
};

/// Nanoseconds of one encode; the transient bytes of the first call and the largest of the rest ride alongside.
fn sampleVision(smp: VisionSample, _: usize) anyerror!f64 {
    const s = smp.tower.s;
    try mlx.check(mlx.mlx_synchronize(s));
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    _ = mlx.mlx_reset_peak_memory();
    const t0 = nowNs(smp.io);
    const out = try smp.tower.forward(smp.pv, smp.gh, smp.gw);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    try mlx.check(mlx.mlx_synchronize(s));
    const ns = nowNs(smp.io) - t0;
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    const extra = peak -| before;
    if (smp.first_peak.* == null) smp.first_peak.* = extra else smp.steady_peak.* = @max(smp.steady_peak.*, extra);
    return @floatFromInt(ns);
}

fn runVision(run: *report.Run) anyerror!void {
    const a = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = run.pack.?;
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    try mimo_source.loadVisionWeightsInto(&weights, io, a, dir);
    var tower = try mimo_vision.MimoVision.init(a, config, &weights);
    defer tower.deinit();
    const feat: c_int = @intCast(3 * config.qv_temporal_patch * config.qv_patch * config.qv_patch);
    for (run.axis("grid")) |grid| {
        const x_at = std.mem.indexOfScalar(u8, grid, 'x').?;
        const gh = try std.fmt.parseInt(u32, grid[0..x_at], 10);
        const gw = try std.fmt.parseInt(u32, grid[x_at + 1 ..], 10);
        const n: c_int = @intCast(gh * gw);
        var pv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pv);
        try mlx.check(mlx.mlx_random_normal(&pv, &[_]c_int{ n, feat }, 2, .float32, 0, 1, .{ .ctx = null }, tower.s));
        try mlx.check(mlx.mlx_array_eval(pv));
        var first_peak: ?usize = null;
        var steady_peak: usize = 0;
        const label = try std.fmt.allocPrint(run.a, "{s} ({d} patches)", .{ grid, n });
        try run.interleave(label, VisionSample{ .tower = &tower, .io = io, .pv = pv, .gh = gh, .gw = gw, .first_peak = &first_peak, .steady_peak = &steady_peak }, sampleVision);
        const mb = 1e6;
        try run.metric("encode", "scratch first", "MB", @as(f64, @floatFromInt(first_peak.?)) / mb);
        try run.metric("encode", "scratch steady", "MB", @as(f64, @floatFromInt(steady_peak)) / mb);
        try run.metric("encode", "bill", "MB", @as(f64, @floatFromInt(mimo_vision.encodeScratchBytes(&config, @intCast(n)))) / mb);
    }
}

// ── mimo2.global_decode_attn ────────────────────────────────────────────

const decode_hq = 64;
const decode_hkv = 4;
const decode_dk = 192;
const decode_dv = 128;

/// `dense` is SDPA over the view's dense arrays, which the view builds once. `dense_rebuild` dequantizes the triples on
/// every call, as a decode step below the packed arms' floor does. `mpp` is the matmul2d kernel (matrix units only)
/// and `split_k` the range kernel every GPU runs.
const DecodeArm = enum { dense, dense_rebuild, mpp, split_k };

pub const global_decode_attn: report.Bench = .{
    .name = "mimo2.global_decode_attn",
    .scope = .mimo2,
    .what = "MiMo global-layer decode attention (64/4 heads, qk 192 / v 128, packed cache) as a dependent layer chain, " ++
        "each layer over its own cache, chain overhead subtracted: dense SDPA, dequant rebuild, matmul2d, split-K",
    .unit = .us,
    .steps = 9,
    .axes = &.{
        .{ .name = "kv", .values = &.{ "2048", "4096", "8192", "16384", "65536", "262144", "524288" } },
    },
    .arms = &.{ "dense", "dense_rebuild", "mpp", "split_k" },
    .ref = "dense",
    .opts = &.{.{ .name = "kv-quant", .help = "the cache's bits: 8 (default) or 4" }},
    .run = runGlobalDecode,
};

const DecodeCase = struct {
    s: mlx.mlx_stream,
    views: []transformer.DenseKVView,
    q0: mlx.mlx_array,
    bits: u8,
    /// Nanoseconds of the chain with no attention in it, taken off every sample.
    base_ns: f64,
};

fn benchIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// The next layer's query depends on this layer's output: [1,64,1,128] -> [1,64,1,192].
fn nextQuery(s: mlx.mlx_stream, out: mlx.mlx_array) !mlx.mlx_array {
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, out, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, decode_hq, 1, decode_dk - decode_dv }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
    const parts = [_]mlx.mlx_array{ out, head };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&q, vec, 3, s));
    return q;
}

/// Nanoseconds of the whole chain; a null arm runs the chain's plumbing alone.
fn decodeChainNs(c: DecodeCase, arm: ?DecodeArm) !f64 {
    const scale: f32 = 1.0 / @sqrt(@as(f32, decode_dk));
    const none = mlx.mlx_array{ .ctx = null };
    const sw = io_util.Stopwatch.init(benchIo());
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&q, c.q0));
    defer _ = mlx.mlx_array_free(q);
    for (c.views) |*view| {
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        if (arm) |a| switch (a) {
            .dense => try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, view.k, view.v, scale, "", none, none, false, c.s)),
            .dense_rebuild => {
                const kd = try kv_quant.dequantizeAffine(c.s, view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.group_size, c.bits);
                defer _ = mlx.mlx_array_free(kd);
                const vd = try kv_quant.dequantizeAffine(c.s, view.v_triple_q, view.v_triple_scales, view.v_triple_biases, view.group_size, c.bits);
                defer _ = mlx.mlx_array_free(vd);
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kd, vd, scale, "", none, none, false, c.s));
            },
            .mpp => {
                _ = mlx.mlx_array_free(out);
                out = (try mimo_forward.qkvAttnMppKernel(c.s, q, view, scale, "")) orelse return error.KernelDeclined;
            },
            .split_k => {
                _ = mlx.mlx_array_free(out);
                out = (try mimo_forward.qkvAttnSplitKKernel(c.s, q, view, scale, "")) orelse return error.KernelDeclined;
            },
        } else try mlx.check(mlx.mlx_slice(&out, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, decode_hq, 1, decode_dv }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, c.s));
        const next = try nextQuery(c.s, out);
        _ = mlx.mlx_array_free(q);
        q = next;
    }
    try mlx.check(mlx.mlx_array_eval(q));
    return @floatFromInt(sw.read());
}

/// Nanoseconds per layer beyond the chain's own plumbing.
fn sampleDecode(c: DecodeCase, arm_index: usize) anyerror!f64 {
    const arm: DecodeArm = @fromBackingInt(@intCast(arm_index));
    const ns = try decodeChainNs(c, arm);
    return @max(1.0, (ns - c.base_ns) / @as(f64, @floatFromInt(c.views.len)));
}

fn uniformBf16(seed: u64, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

fn triples(view: *const transformer.DenseKVView) [6]mlx.mlx_array {
    return .{ view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.v_triple_q, view.v_triple_scales, view.v_triple_biases };
}

fn runGlobalDecode(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const kvq = run.opt("kv-quant") orelse "8";
    const bits: u8 = if (std.mem.eql(u8, kvq, "8")) 8 else if (std.mem.eql(u8, kvq, "4")) 4 else {
        run.invalid("--opt kv-quant is 8 or 4");
        return;
    };
    if (run.hasArm("mpp") and !mimo_forward.mimoDecodeUsesNax()) {
        run.invalid("mpp is the matmul2d kernel and this GPU has no matrix units: pass --arms dense,dense_rebuild,split_k");
        return;
    }
    const layers = run.steps(9);
    if (layers > 64) return error.TooManyLayers;
    const a = std.heap.page_allocator;
    const caches = try a.alloc(transformer.KVCache, layers);
    defer a.free(caches);
    const views = try a.alloc(transformer.DenseKVView, layers);
    defer a.free(views);
    for (run.axis("kv")) |kv_text| {
        const kv = try std.fmt.parseInt(c_int, kv_text, 10);
        var built: usize = 0;
        defer for (0..built) |i| {
            views[i].deinit();
            caches[i].deinit();
        };
        for (0..layers) |i| {
            const seed = @as(u64, 0x3D2D_BE7C) + i * 4;
            caches[i] = try transformer.KVCache.initWithConfig(a, 1, kv_quant.KVQuantConfig.affine(bits));
            {
                const k = try uniformBf16(seed, &[_]c_int{ 1, decode_hkv, kv - 1, decode_dk }, -0.5, 0.5, s);
                defer _ = mlx.mlx_array_free(k);
                const v = try uniformBf16(seed + 1, &[_]c_int{ 1, decode_hkv, kv - 1, decode_dv }, -0.5, 0.5, s);
                defer _ = mlx.mlx_array_free(v);
                var first = try caches[i].update(0, k, v, s, 0);
                first.deinit();
            }
            const k1 = try uniformBf16(seed + 2, &[_]c_int{ 1, decode_hkv, 1, decode_dk }, -0.5, 0.5, s);
            defer _ = mlx.mlx_array_free(k1);
            const v1 = try uniformBf16(seed + 3, &[_]c_int{ 1, decode_hkv, 1, decode_dv }, -0.5, 0.5, s);
            defer _ = mlx.mlx_array_free(v1);
            views[i] = try caches[i].update(0, k1, v1, s, 0);
            built += 1;
            const trip = triples(&views[i]);
            const ev = mlx.mlx_vector_array_new_data(&trip, trip.len);
            defer _ = mlx.mlx_vector_array_free(ev);
            try mlx.check(mlx.mlx_eval(ev));
        }
        var bytes: f64 = 0;
        for (triples(&views[0])) |t| bytes += @floatFromInt(mlx.mlx_array_size(t) * mlx.mlx_array_itemsize(t));
        const q0 = try uniformBf16(0x51DE, &[_]c_int{ 1, decode_hq, 1, decode_dk }, -4.0, 4.0, s);
        defer _ = mlx.mlx_array_free(q0);
        try mlx.check(mlx.mlx_array_eval(q0));

        var c: DecodeCase = .{ .s = s, .views = views[0..layers], .q0 = q0, .bits = bits, .base_ns = 0 };
        var plumbing: [5]f64 = undefined;
        for (&plumbing) |*p| p.* = try decodeChainNs(c, null);
        c.base_ns = report.stats(&plumbing).median;
        try run.interleave(kv_text, c, sampleDecode);
        for (run.plan.arms) |i| {
            const arm: DecodeArm = @fromBackingInt(@intCast(i));
            try run.metric(@tagName(arm), "GB/s", "GB/s", bytes / run.lastMedianNs(@tagName(arm)));
        }
        _ = mlx.mlx_clear_cache();
    }
}

// ── mimo2.prefill_attn ──────────────────────────────────────────────────

const prefill_hq = 64;
const sliding_hkv = 8;
const sliding_window: c_int = 128;
const max_keys: c_int = 524288;

/// `served` is the kv8 chain the forward runs on a global layer (per-dispatch slice and dequant, fp32 carries) and
/// `dense` the same dispatches over K/V dequantized beforehand. `band` is one sliding layer, its ring view dequantized
/// per call, on the fused band-and-sink kernel; `band_composed` is the same layer on MLX's masked SDPA.
const PrefillArm = enum { served, dense, band, band_composed };

pub const prefill_attn: report.Bench = .{
    .name = "mimo2.prefill_attn",
    .scope = .mimo2,
    .what = "MiMo prefill attention of one chunk (64 query heads, kv8 g64): a global layer at the end of kL keys, served " ++
        "against pre-dequantized K/V, and a sliding layer's ring, fused band kernel against composed SDPA. " ++
        "Global rows read against served, the sliding row against --ref band",
    .unit = .us,
    .steps = 1,
    .axes = &.{
        .{ .name = "ql", .values = &.{"2048"} },
        .{ .name = "kl", .values = &.{ "2048", "16384", "32768", "65536", "131072", "262144" } },
    },
    .arms = &.{ "served", "dense", "band", "band_composed" },
    .ref = "served",
    .run = runPrefill,
};

/// One global rung; the kernel dispatches and the arm that engaged ride alongside the timings.
const GlobalCase = struct {
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const transformer.DenseKVView,
    calls: usize,
    dispatches: *[2]u32,
    engaged: *transformer.AttnPdArm,
};

fn sampleGlobal(c: GlobalCase, arm_index: usize) anyerror!f64 {
    const arm: PrefillArm = @fromBackingInt(@intCast(arm_index));
    const scale: f32 = 1.0 / @sqrt(192.0);
    const sw = io_util.Stopwatch.init(benchIo());
    for (0..c.calls) |_| {
        const out = (switch (arm) {
            .served => try transformer.fusedSdpaPrefillKv(c.s, c.q, c.view, scale, 0, .{ .ctx = null }),
            else => try transformer.fusedSdpaPrefill(c.s, c.q, c.view.k, c.view.v, scale, 0),
        }) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
    }
    const ns = sw.read();
    c.dispatches[arm_index] = attn_kernels.fused256_last_dispatch_count;
    c.engaged.* = attn_kernels.attn_pd_last_arm;
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(c.calls));
}

const SlidingCase = struct {
    s: mlx.mlx_stream,
    cfg: *const model_mod.ModelConfig,
    q: mlx.mlx_array,
    sinks: mlx.mlx_array,
    mask: mlx.mlx_array,
    kq: kv_quant.QuantizedKV,
    vq: kv_quant.QuantizedKV,
    calls: usize,
};

fn sampleSliding(c: SlidingCase, arm_index: usize) anyerror!f64 {
    const arm: PrefillArm = @fromBackingInt(@intCast(arm_index));
    const scale: f32 = 1.0 / @sqrt(192.0);
    const sw = io_util.Stopwatch.init(benchIo());
    for (0..c.calls) |_| {
        const kd = try kv_quant.dequantizeAffine(c.s, c.kq.q, c.kq.scales, c.kq.biases, 64, 8);
        defer _ = mlx.mlx_array_free(kd);
        const vd = try kv_quant.dequantizeAffine(c.s, c.vq.q, c.vq.scales, c.vq.biases, 64, 8);
        defer _ = mlx.mlx_array_free(vd);
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        if (arm == .band) {
            const view = transformer.DenseKVView{ .k = kd, .v = vd, .owned = false };
            _ = mlx.mlx_array_free(out);
            out = (try transformer.slidingPrefillAttn(c.s, c.cfg, c.q, &view, scale, c.sinks)) orelse return error.FusedDeclined;
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, c.q, kd, vd, scale, "array", c.mask, c.sinks, false, c.s));
        }
        try mlx.check(mlx.mlx_array_eval(out));
    }
    return @as(f64, @floatFromInt(sw.read())) / @as(f64, @floatFromInt(c.calls));
}

/// The first `kl` cached rows as the forward's view: packed slices, plus the dense arrays evaluated once for the arm
/// that skips the dequant.
const PrefixView = struct {
    parts: [6]mlx.mlx_array,
    view: transformer.DenseKVView,

    fn init(s: mlx.mlx_stream, trip: [6]mlx.mlx_array, kl: c_int) !PrefixView {
        var parts: [6]mlx.mlx_array = undefined;
        var n: usize = 0;
        errdefer for (parts[0..n]) |p| {
            _ = mlx.mlx_array_free(p);
        };
        for (trip) |src| {
            const sh = mlx.getShape(src);
            parts[n] = mlx.mlx_array_new();
            n += 1;
            try mlx.check(mlx.mlx_slice(&parts[n - 1], src, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], kl, sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        }
        const kd = try kv_quant.dequantizeAffine(s, parts[0], parts[1], parts[2], 64, 8);
        errdefer _ = mlx.mlx_array_free(kd);
        const vd = try kv_quant.dequantizeAffine(s, parts[3], parts[4], parts[5], 64, 8);
        errdefer _ = mlx.mlx_array_free(vd);
        try mlx.check(mlx.mlx_array_eval(kd));
        try mlx.check(mlx.mlx_array_eval(vd));
        return .{ .parts = parts, .view = .{
            .k = kd,
            .v = vd,
            .owned = true,
            .k_triple_q = parts[0],
            .k_triple_scales = parts[1],
            .k_triple_biases = parts[2],
            .v_triple_q = parts[3],
            .v_triple_scales = parts[4],
            .v_triple_biases = parts[5],
            .has_quant_triple = true,
            .bits = 8,
            .group_size = 64,
        } };
    }

    fn deinit(self: *PrefixView) void {
        self.view.deinit();
        for (self.parts) |p| _ = mlx.mlx_array_free(p);
    }
};

/// The served query of one chunk: projection rows split into heads, transposed, partially roped.
fn chunkQuery(seed: u64, ql: c_int, base: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const proj = try uniformBf16(seed, &[_]c_int{ 1, ql, prefill_hq * 192 }, -2.0, 2.0, s);
    defer _ = mlx.mlx_array_free(proj);
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, proj, &[_]c_int{ 1, ql, prefill_hq, 192 }, 4, s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, r, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_fast_rope(&q, t, 64, false, .{ .value = base, .has_value = true }, 1.0, 0, .{ .ctx = null }, s));
    try mlx.check(mlx.mlx_array_eval(q));
    return q;
}

/// `[1, 1, ql, kl]` bool band, bottom-right aligned: the mask the composed arm reads.
fn bandMask(a: std.mem.Allocator, ql: c_int, kl: c_int) !mlx.mlx_array {
    const cols: usize = @intCast(kl);
    const buf = try a.alloc(bool, @as(usize, @intCast(ql)) * cols);
    defer a.free(buf);
    for (0..@intCast(ql)) |r| {
        const row: c_int = kl - ql + @as(c_int, @intCast(r));
        for (0..cols) |c| {
            const col: c_int = @intCast(c);
            buf[r * cols + c] = col <= row and row - col < sliding_window;
        }
    }
    return mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, 1, ql, kl }, 4, .bool_);
}

/// Times the arms of `names` that are in force as one rung; false when none is.
fn interleaveSome(run: *report.Run, label: []const u8, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize) anyerror!f64, names: []const []const u8) !bool {
    var picked: [report.max_arms]usize = undefined;
    var n: usize = 0;
    for (run.plan.arms) |i| {
        for (names) |name| if (std.mem.eql(u8, run.plan.bench.arms[i], name)) {
            picked[n] = i;
            n += 1;
        };
    }
    if (n == 0) return false;
    const all = run.plan.arms;
    run.plan.arms = picked[0..n];
    defer run.plan.arms = all;
    try run.interleave(label, ctx, f);
    return true;
}

fn causalFlops(ql: c_int, kl: c_int) f64 {
    const qf: f64 = @floatFromInt(ql);
    const kf: f64 = @floatFromInt(kl);
    return 2.0 * prefill_hq * (192.0 + 128.0) * (qf * (kf - qf) + qf * (qf + 1.0) / 2.0);
}

fn runPrefill(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const a = std.heap.page_allocator;
    var max_kl: c_int = 0;
    for (run.axis("ql")) |t| max_kl = @max(max_kl, try std.fmt.parseInt(c_int, t, 10));
    for (run.axis("kl")) |t| max_kl = @max(max_kl, try std.fmt.parseInt(c_int, t, 10));
    if (max_kl > max_keys) return error.TooManyKeys;

    // One global layer's kv8 cache, filled once; each rung reads the first kL rows of its views.
    var cache = try transformer.KVCache.initWithConfig(a, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    var full = blk: {
        const k = try uniformBf16(0xA77D, &[_]c_int{ 1, 4, max_kl, 192 }, -2.0, 2.0, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try uniformBf16(0xA77E, &[_]c_int{ 1, 4, max_kl, 128 }, -1.0, 1.0, s);
        defer _ = mlx.mlx_array_free(v);
        break :blk try cache.update(0, k, v, s, 0);
    };
    defer full.deinit();
    const trip = triples(&full);
    {
        const ev = mlx.mlx_vector_array_new_data(&trip, trip.len);
        defer _ = mlx.mlx_vector_array_free(ev);
        try mlx.check(mlx.mlx_eval(ev));
    }

    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "mimo_v2";
    cfg.has_sliding_window = true;
    cfg.sliding_window = @intCast(sliding_window);
    cfg.head_dim = 192;
    cfg.v_head_dim = 128;

    var engaged: ?transformer.AttnPdArm = null;
    for (run.axis("ql")) |ql_text| {
        const ql = try std.fmt.parseInt(c_int, ql_text, 10);
        const q = try chunkQuery(0xB001, ql, 1.0e7, s);
        defer _ = mlx.mlx_array_free(q);
        for (run.axis("kl")) |kl_text| {
            const kl = try std.fmt.parseInt(c_int, kl_text, 10);
            if (kl < ql) continue;
            var view = try PrefixView.init(s, trip, kl);
            defer view.deinit();
            var dispatches: [2]u32 = .{ 0, 0 };
            var arm_seen: transformer.AttnPdArm = .simd;
            const label = try std.fmt.allocPrint(run.a, "global qL {d} kL {d}", .{ ql, kl });
            const case: GlobalCase = .{ .s = s, .q = q, .view = &view.view, .calls = run.steps(1), .dispatches = &dispatches, .engaged = &arm_seen };
            if (!try interleaveSome(run, label, case, sampleGlobal, &.{ "served", "dense" })) continue;
            if (engaged == null) {
                engaged = arm_seen;
                log.info("[ubench] mimo2.prefill_attn engaged: {s}\n", .{@tagName(arm_seen)});
            } else if (engaged.? != arm_seen) run.invalid("the prefill kernel arm changed between rungs");
            for ([_]PrefillArm{ .served, .dense }) |arm| {
                if (!run.hasArm(@tagName(arm))) continue;
                try run.metric(@tagName(arm), "TFLOP/s", "TFLOP/s", causalFlops(ql, kl) / run.lastMedianNs(@tagName(arm)) / 1e3);
            }
            if (run.hasArm("served")) try run.metric("served", "dispatches", "n", @floatFromInt(dispatches[0]));
            _ = mlx.mlx_clear_cache();
        }

        // One sliding layer: its ring view (window - 1 + chunk rows) is dequantized whole per call.
        const ring = ql + sliding_window - 1;
        const qs = try chunkQuery(0xB002, ql, 1.0e4, s);
        defer _ = mlx.mlx_array_free(qs);
        const sinks = try uniformBf16(0xB003, &[_]c_int{prefill_hq}, -1.0, 1.0, s);
        defer _ = mlx.mlx_array_free(sinks);
        const ks = try uniformBf16(0xB004, &[_]c_int{ 1, sliding_hkv, ring, 192 }, -2.0, 2.0, s);
        defer _ = mlx.mlx_array_free(ks);
        const vs = try uniformBf16(0xB005, &[_]c_int{ 1, sliding_hkv, ring, 128 }, -1.0, 1.0, s);
        defer _ = mlx.mlx_array_free(vs);
        var kq = try kv_quant.quantizeAffine(s, ks, 64, 8);
        defer kq.deinit();
        var vq = try kv_quant.quantizeAffine(s, vs, 64, 8);
        defer vq.deinit();
        const mask = try bandMask(a, ql, ring);
        defer _ = mlx.mlx_array_free(mask);
        for ([_]mlx.mlx_array{ kq.q, kq.scales, kq.biases, vq.q, vq.scales, vq.biases, mask }) |x| try mlx.check(mlx.mlx_array_eval(x));
        const label = try std.fmt.allocPrint(run.a, "sliding qL {d}", .{ql});
        const case: SlidingCase = .{ .s = s, .cfg = &cfg, .q = qs, .sinks = sinks, .mask = mask, .kq = kq, .vq = vq, .calls = run.steps(1) };
        _ = try interleaveSome(run, label, case, sampleSliding, &.{ "band", "band_composed" });
        _ = mlx.mlx_clear_cache();
    }
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}
