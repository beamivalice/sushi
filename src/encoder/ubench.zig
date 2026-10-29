//! The encoder's two hot loops, timed on synthetic data: the block-LDL factor of a Hessian bank (GPU float32 against
//! the float64 CPU oracle) and the trellis search over a batch of tiles. Run by `sushi-ubench`, never by a test.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const report = @import("mlx_host").report;
const format = @import("sushi_exl3").format;
const gpu_factor = @import("gpu_factor.zig");
const metal_search = @import("metal_search.zig");
const quantize = @import("quantize.zig");
const stages = @import("stages.zig");

const Shape = struct { key: []const u8, n: usize, batch: usize };

/// Flash-Next's hidden width in 32-matrix banks, and a small width in 16.
const SHAPES = [_]Shape{
    .{ .key = "n512x16", .n = 512, .batch = 16 },
    .{ .key = "n2560x32", .n = 2560, .batch = 32 },
};

pub const factor: report.Bench = .{
    .name = "enc.factor",
    .scope = .enc,
    .what = "block-LDL factor of a Hessian bank: float32 on the GPU against the float64 CPU oracle, ms per matrix",
    .unit = .ms,
    .steps = 1,
    .axes = &.{
        .{ .name = "shape", .values = &.{ "n512x16", "n2560x32" }, .numeric = false },
    },
    .arms = &.{ "gpu", "cpu" },
    .ref = "cpu",
    .run = runFactor,
};

const FactorRung = struct {
    s: mlx.mlx_stream,
    n: usize,
    batch: usize,
    raw: []const f32,
    many: []const []const f32,
    means: []const f64,
    ls: [][]f32,
};

/// Nanoseconds per matrix: the whole bank on the GPU, or one matrix on the CPU.
fn factorSample(c: FactorRung, arm: usize) anyerror!f64 {
    const alloc = std.heap.page_allocator;
    const t0 = stages.nowNs();
    if (arm == 0) {
        var stats: gpu_factor.Stats = .{};
        try gpu_factor.factorBatch(alloc, c.s, c.many, c.means, c.n, .{}, &stats, c.ls);
        const dt = stages.nowNs() - t0;
        for (c.ls) |l| alloc.free(l);
        return @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(c.batch));
    }
    const l = try quantize.ldlFactor(alloc, c.raw, c.n);
    const dt = stages.nowNs() - t0;
    alloc.free(l);
    return @floatFromInt(dt);
}

fn runFactor(run: *report.Run) anyerror!void {
    const s = mlx.gpuStream();
    const alloc = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(77);
    for (SHAPES) |shape| {
        if (!listed(run.axis("shape"), shape.key)) continue;
        const raw = try gpu_factor.spdHessian(alloc, prng.random(), shape.n, 64, 0.5);
        defer alloc.free(raw);
        const damped = try alloc.dupe(f32, raw);
        defer alloc.free(damped);
        const mean = quantize.dampDiagonal(damped, shape.n);
        const many = try run.a.alloc([]const f32, shape.batch);
        @memset(many, damped);
        const means = try run.a.alloc(f64, shape.batch);
        @memset(means, mean);
        const ls = try run.a.alloc([]f32, shape.batch);
        const c: FactorRung = .{ .s = s, .n = shape.n, .batch = shape.batch, .raw = raw, .many = many, .means = means, .ls = ls };
        try run.interleave(shape.key, c, factorSample);
    }
}

const SEARCH_TILES: usize = 4096;

pub const search: report.Bench = .{
    .name = "enc.search",
    .scope = .enc,
    .what = "trellis search throughput at K2.5 (n40, MCG, window 15) over 4096 random tiles, us per tile",
    .unit = .us,
    .steps = 50,
    .arms = &.{"search"},
    .ref = "search",
    .run = runSearch,
};

const SearchRung = struct {
    s: mlx.mlx_stream,
    tiles: []const f32,
    packed_out: []u16,
    recon: []f32,
    steps: usize,
};

/// Nanoseconds per tile over `steps` back-to-back searches: a conversion keeps the GPU busy for minutes, so the
/// sample is sustained work, not one burst.
fn searchSample(c: SearchRung, _: usize) anyerror!f64 {
    const rate: format.Rate = .{ .n = 40 };
    const dec: format.Decode = .{ .codebook = .mcg, .window = .w15 };
    const t0 = stages.nowNs();
    for (0..c.steps) |_| try metal_search.encodeTiles(std.heap.page_allocator, c.s, rate, dec, c.tiles, c.packed_out, c.recon);
    const dt = stages.nowNs() - t0;
    return @as(f64, @floatFromInt(dt)) / @as(f64, @floatFromInt(c.steps * SEARCH_TILES));
}

fn runSearch(run: *report.Run) anyerror!void {
    const alloc = std.heap.page_allocator;
    var prng = std.Random.DefaultPrng.init(4096);
    const r = prng.random();
    const tiles = try alloc.alloc(f32, SEARCH_TILES * metal_search.tile_len);
    defer alloc.free(tiles);
    for (tiles) |*v| v.* = r.floatNorm(f32);
    const packed_out = try alloc.alloc(u16, SEARCH_TILES * 40);
    defer alloc.free(packed_out);
    const recon = try alloc.alloc(f32, SEARCH_TILES * metal_search.tile_len);
    defer alloc.free(recon);
    const c: SearchRung = .{ .s = mlx.gpuStream(), .tiles = tiles, .packed_out = packed_out, .recon = recon, .steps = run.steps(50) };
    try run.interleave("K2.5 mcg w15, 4096 tiles", c, searchSample);
}

fn listed(values: []const []const u8, v: []const u8) bool {
    for (values) |x| if (std.mem.eql(u8, x, v)) return true;
    return false;
}
