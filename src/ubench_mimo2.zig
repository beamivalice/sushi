//! MiMo-V2.6-Flash microbenches: the FP8 trunk linears at the checkpoint's own shapes, and the MiMo-ViT encode on a
//! real tower. Run by `sushi-ubench`, never by a test.
const std = @import("std");
const mlx = @import("mlx.zig");
const report = @import("ubench_report.zig");
const fp8_block = @import("fp8_block.zig");
const transformer = @import("transformer.zig");
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

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}
