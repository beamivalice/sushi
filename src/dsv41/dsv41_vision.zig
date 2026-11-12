//! DeepSeek-V4.1-Flash image input: the release's processor (`inference/image_processor.py`) and tower
//! (`inference/vision.py`). The pack keeps the release's tensor names: `vision.*`, `aligner.*`,
//! `image_start/end/newline`, all bf16.
//!
//! A 32-block ViT with 2-D RoPE and full attention over one image's 14 px patches, then the aligner:
//! a 3x3 unfold into the text width through `w2(gelu(w1(x)))`. The trunk sees an image as
//!
//!     [start] + ([row of n_w aligner tokens] + [newline]) * n_h + [end]
//!
//! every position of which carries `image_token_id`, so `forward` returns the whole span (the three
//! delimiters are learned embeddings) and the trunk's splice lands it row by row.
//!
//! Plan, letterbox and unfold geometry follow ddalcu/mlx-serve's `dsv41_vision.zig` (MIT), re-derived
//! against the release's reference.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const common = @import("../vision_common.zig");
const cfg_mod = @import("dsv41_config.zig");
const sc_mod = @import("dsv41_scope.zig");
const log = @import("../log.zig");
const transient_peak = @import("transient_peak.zig");
const dflash = @import("../dflash.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_pipelines = @import("dsv41_pipelines.zig");
const quantized = @import("../tensor/quantized.zig");
const matmul = @import("../tensor/matmul.zig");
const metal = @import("../metal.zig");
const manifest = @import("../pipeline_manifest.zig");
const dsv41_model = @import("dsv41_model.zig");
const io_util = @import("../io_util.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const none = sc_mod.none;
const ci = sc_mod.ci;

/// Query rows per attention call: bounds the score sheet a non-fused SDPA would build.
pub const QUERY_CHUNK: usize = 512;
const EPS: f32 = 1e-6;

/// What the image processor reads of the config.
pub const Processor = struct {
    patch: u32,
    downsample: u32,
    max_tokens: u32,
    min_pixels: u32,
    /// An image wider than this many heights is stretched into the capped canvas; 0 = no cap.
    max_wh_ratio: u32,

    pub fn of(v: cfg_mod.Vision) Processor {
        return .{ .patch = v.patch, .downsample = v.downsample, .max_tokens = v.max_tokens, .min_pixels = v.min_pixels, .max_wh_ratio = v.max_wh_ratio };
    }
};

/// `plan_image_grid` + `load_image`: the canvas in pixels, the ViT patch grid and the aligner grid, and
/// whether the image is stretched into the canvas (the `max_wh_ratio` arm) rather than letterboxed.
pub const Plan = struct { h: u32, w: u32, vit_h: u32, vit_w: u32, llm_h: u32, llm_w: u32, stretch: bool };

/// Tokens one image costs: its aligner grid, a newline per row, start and end.
pub fn spanTokens(llm_h: u32, llm_w: u32) usize {
    return @as(usize, llm_h) * (@as(usize, llm_w) + 1) + 2;
}

/// The aligner grid of a ViT grid: `ceil(n / downsample)` per axis.
pub fn llmCells(vit: u32, downsample: u32) u32 {
    return (vit + downsample - 1) / downsample;
}

/// Tokens an image with ViT grid `grid_h x grid_w` occupies in the prompt.
pub fn spanRows(grid_h: u32, grid_w: u32, downsample: u32) usize {
    return spanTokens(llmCells(grid_h, downsample), llmCells(grid_w, downsample));
}

fn ceilDiv(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

fn llmGrid(best_h: u64, best_w: u64, patch: u64, ds: u64) [2]u64 {
    return .{ ceilDiv(best_h / patch, ds), ceilDiv(best_w / patch, ds) };
}

/// `plan_image_grid` for an image of `width` x `height` pixels. Null only for a degenerate config.
pub fn planGrid(width: u32, height: u32, v: Processor) ?Plan {
    const patch = v.patch;
    const ds = v.downsample;
    const max_tokens = v.max_tokens;
    if (width == 0 or height == 0 or patch == 0 or ds == 0 or max_tokens < 4) return null;
    const p: u64 = patch;
    var w: u64 = width;
    var h: u64 = height;
    const stretch = v.max_wh_ratio != 0 and w >= @as(u64, v.max_wh_ratio) * h;
    if (v.max_wh_ratio != 0 and w > @as(u64, v.max_wh_ratio) * h) w = @as(u64, v.max_wh_ratio) * h;
    if (w * h < v.min_pixels) {
        const ratio = @sqrt(@as(f64, @floatFromInt(v.min_pixels)) / @as(f64, @floatFromInt(w * h)));
        w = @intFromFloat(@trunc(@as(f64, @floatFromInt(w)) * ratio));
        h = @intFromFloat(@trunc(@as(f64, @floatFromInt(h)) * ratio));
        if (w == 0 or h == 0) return null;
    }
    var best_w = ceilDiv(w, p) * p;
    var best_h = ceilDiv(h, p) * p;
    var grid = llmGrid(best_h, best_w, p, ds);
    if (spanTokens(@intCast(grid[0]), @intCast(grid[1])) > max_tokens) {
        // `solve_resize_ratio`: the largest aspect-preserving size whose grid fits the budget.
        const r = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w));
        const max_w_f = @sqrt(@as(f64, @floatFromInt(max_tokens - 2)) / r + 0.25) - 0.5;
        const max_h_f = max_w_f * r;
        const cell = p * ds;
        if (max_w_f < 1.0) {
            best_h = (@as(u64, max_tokens) - 2) / 2 * cell;
            best_w = cell;
        } else if (max_h_f < 1.0) {
            best_h = cell;
            best_w = (@as(u64, max_tokens) - 3) * cell;
        } else {
            const cf: f64 = @floatFromInt(cell);
            const beta = @min(@floor(max_w_f) * cf / @as(f64, @floatFromInt(w)), @floor(max_h_f) * cf / @as(f64, @floatFromInt(h)));
            best_h = @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(h)) * beta / @as(f64, @floatFromInt(p))))) * p;
            best_w = @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(w)) * beta / @as(f64, @floatFromInt(p))))) * p;
        }
        grid = llmGrid(best_h, best_w, p, ds);
        if (best_h == 0 or best_w == 0 or spanTokens(@intCast(grid[0]), @intCast(grid[1])) > max_tokens) return null;
    }
    return .{
        .h = @intCast(best_h),
        .w = @intCast(best_w),
        .vit_h = @intCast(best_h / p),
        .vit_w = @intCast(best_w / p),
        .llm_h = @intCast(grid[0]),
        .llm_w = @intCast(grid[1]),
        .stretch = stretch,
    };
}

/// Python's `round`: half to even.
fn pyRound(x: f64) u32 {
    const f = @floor(x);
    const d = x - f;
    const r: f64 = if (d > 0.5) f + 1 else if (d < 0.5) f else if (@mod(f, 2.0) == 0) f else f + 1;
    return @intFromFloat(r);
}

/// Pillow `ImageOps.pad(image, (w, h), BICUBIC, 127)`: `contain` picks the resized size, the rest is
/// the fill, centered (half to even on an odd margin).
pub const Letterbox = struct { rw: u32, rh: u32, x: u32, y: u32 };

pub fn letterbox(src_w: u32, src_h: u32, w: u32, h: u32) Letterbox {
    const im_ratio = @as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h));
    const dest_ratio = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    var rw = w;
    var rh = h;
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            rh = pyRound(@as(f64, @floatFromInt(src_h)) / @as(f64, @floatFromInt(src_w)) * @as(f64, @floatFromInt(w)));
        } else {
            rw = pyRound(@as(f64, @floatFromInt(src_w)) / @as(f64, @floatFromInt(src_h)) * @as(f64, @floatFromInt(h)));
        }
    }
    return .{
        .rw = rw,
        .rh = rh,
        .x = if (rw != w) pyRound(@as(f64, @floatFromInt(w - rw)) * 0.5) else 0,
        .y = if (rw == w and rh != h) pyRound(@as(f64, @floatFromInt(h - rh)) * 0.5) else 0,
    };
}

/// `(x / 255 - 0.5) / 0.5`, the processor's normalization, in f32 as the reference does it.
fn normalizePixel(v: u8) f32 {
    return (@as(f32, @floatFromInt(v)) / 255.0 - 0.5) / 0.5;
}

/// One image's ViT patches: f32 bytes of `[grid_h * grid_w, 3 * patch * patch]` in `(c, py, px)` order,
/// row-major over the grid. `h`/`w` are the canvas in pixels.
pub const Pixels = struct { bytes: []u8, h: u32, w: u32, grid_h: u32, grid_w: u32 };

/// `load_image` for `rgb` (HWC u8): plan, letterbox (or stretch), normalize, patchify. Caller owns `bytes`.
pub fn preprocess(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, v: Processor) !Pixels {
    const patch = v.patch;
    const plan = planGrid(src_w, src_h, v) orelse return error.InvalidImageDimensions;
    const box: Letterbox = if (plan.stretch) .{ .rw = plan.w, .rh = plan.h, .x = 0, .y = 0 } else letterbox(src_w, src_h, plan.w, plan.h);
    const resized: ?[]u8 = if (box.rw == src_w and box.rh == src_h)
        null
    else
        try common.resizeRgbU8(allocator, rgb, src_h, src_w, box.rh, box.rw, .bicubic);
    defer if (resized) |r| allocator.free(r);
    const content = resized orelse rgb;

    const n: usize = @as(usize, plan.vit_h) * plan.vit_w;
    const feat: usize = 3 * @as(usize, patch) * patch;
    const bytes = try allocator.alloc(u8, n * feat * 4);
    errdefer allocator.free(bytes);
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. n * feat];
    const fill = normalizePixel(127);
    for (0..plan.vit_h) |vh| for (0..plan.vit_w) |vw| {
        const row = out[(vh * plan.vit_w + vw) * feat ..][0..feat];
        for (0..3) |c| for (0..patch) |py| for (0..patch) |px| {
            const y = vh * patch + py;
            const x = vw * patch + px;
            const inside = y >= box.y and y < box.y + box.rh and x >= box.x and x < box.x + box.rw;
            row[(c * patch + py) * patch + px] = if (inside) normalizePixel(content[((y - box.y) * box.rw + (x - box.x)) * 3 + c]) else fill;
        };
    };
    return .{ .bytes = bytes, .h = plan.h, .w = plan.w, .grid_h = plan.vit_h, .grid_w = plan.vit_w };
}

pub const QUANT_GROUP: u32 = 128;

/// Does the tower run the pack tensor `key` quantized? The ViT blocks' and the aligner's bf16 weight matrices
/// whose input width is whole groups; the patch embedding, norms, biases and delimiters stay as stored.
pub fn quantizesAtLoad(key: []const u8, bf16: bool, shape: []const u64) bool {
    if (!bf16 or shape.len != 2 or shape[1] % QUANT_GROUP != 0) return false;
    if (std.mem.eql(u8, key, "aligner.w1.weight") or std.mem.eql(u8, key, "aligner.w2.weight")) return true;
    if (!std.mem.startsWith(u8, key, "vision.blocks.")) return false;
    for ([_][]const u8{ ".attn.wqkv.weight", ".attn.wo.weight", ".mlp.w1.weight", ".mlp.w2.weight" }) |tail|
        if (std.mem.endsWith(u8, key, tail)) return true;
    return false;
}

/// Bytes a quantized `[out, in]` weight holds: the codes and a bf16 scale and bias per group.
pub fn quantizedBytes(out: u64, in: u64) u64 {
    return common.quantizedBytes(out, in, QUANT_GROUP);
}

/// The rows the tower sees for the images a processor admits: ViT patches (every ViT linear's rows, and the keys of
/// every attention call) and aligner cells (the aligner's rows). Read off `planGrid` over image sizes from 1 px to
/// 16,384 px a side: a geometric grid on both axes (`step`), and every width under 128 px (and every height) against a
/// geometric grid of the other side, where the thin canvases are. A sample cannot undercut the true minimum; it can
/// miss it, so `engineGap` asks for twice what a plan needs.
pub const Reach = struct { min_patches: u64, max_patches: u64, min_cells: u64, max_cells: u64 };

fn grow(r: *?Reach, p: Processor, w: u32, h: u32) void {
    const plan = planGrid(w, h, p) orelse return;
    const patches = @as(u64, plan.vit_h) * plan.vit_w;
    const cells = @as(u64, plan.llm_h) * plan.llm_w;
    r.* = if (r.*) |x|
        .{ .min_patches = @min(x.min_patches, patches), .max_patches = @max(x.max_patches, patches), .min_cells = @min(x.min_cells, cells), .max_cells = @max(x.max_cells, cells) }
    else
        .{ .min_patches = patches, .max_patches = patches, .min_cells = cells, .max_cells = cells };
}

pub fn reachOf(p: Processor, step: f64) ?Reach {
    var out: ?Reach = null;
    var a: f64 = 1;
    while (a <= 16384) : (a = @max(a + 1, a * step)) {
        var b: f64 = 1;
        while (b <= 16384) : (b = @max(b + 1, b * step)) grow(&out, p, @intFromFloat(a), @intFromFloat(b));
        for (1..128) |thin| {
            grow(&out, p, @intCast(thin), @intFromFloat(a));
            grow(&out, p, @intFromFloat(a), @intCast(thin));
        }
    }
    return out;
}

/// An image the engine's tower would have to refuse: the plan MLX takes for it and the engine lacks.
pub const Gap = struct { what: []const u8, rows: u64, out: usize, in: usize };

/// The fewest rows from which a linear of this shape takes only ported plans.
fn portedFrom(g: matmul.Gpu, out: usize, in: usize) usize {
    var m: usize = 1;
    while (m < 1 << 14 and quantized.unported(g, m, out, in, QUANT_GROUP, .bfloat16) != null) m += 1;
    return m;
}

/// The first shape an admissible image can give a quantized linear or the attention that the engine does not port, else
/// null. Decided at the load, so no request meets a refusal MLX would not give.
pub fn engineGap(v: cfg_mod.Vision, out_dim: u32, quantize: bool) !?Gap {
    if (v.headDim() != 64) return .{ .what = "attention head dimension other than 64", .rows = 0, .out = v.headDim(), .in = 0 };
    const r = reachOf(Processor.of(v), 1.06) orelse return .{ .what = "no admissible image", .rows = 0, .out = 0, .in = 0 };
    // The last chunk of a query is at most as long as the keys.
    if (r.min_patches < 8) return .{ .what = "fewer keys than a vector attention chunk", .rows = r.min_patches, .out = 0, .in = 0 };
    if (!quantize) return null;
    const g = try matmul.gpu();
    const h: usize = v.hidden;
    const shapes = [_]struct { out: usize, in: usize, rows: u64 }{
        .{ .out = 3 * h, .in = h, .rows = r.min_patches },
        .{ .out = h, .in = h, .rows = r.min_patches },
        .{ .out = 2 * @as(usize, v.inter), .in = h, .rows = r.min_patches },
        .{ .out = h, .in = v.inter, .rows = r.min_patches },
        .{ .out = out_dim, .in = h * v.downsample * v.downsample, .rows = r.min_cells },
        .{ .out = out_dim, .in = out_dim, .rows = r.min_cells },
    };
    for (shapes) |sh| {
        if (sh.in % QUANT_GROUP != 0) continue;
        const from = portedFrom(g, sh.out, sh.in);
        if (sh.rows < 2 * from) return .{ .what = "a plan the engine does not port", .rows = sh.rows, .out = sh.out, .in = sh.in };
    }
    return null;
}

/// A linear: `w` is a bf16 `[out, in]`, or `q` holds it as MLX affine codes.
const Lin = struct {
    w: A = none,
    bias: A = none,
    q: ?dflash.DflashLinear = null,

    fn apply(self: Lin, sc: *Scope, x: A) !A {
        const y = if (self.q) |q| blk: {
            var o = mlx.mlx_array_new();
            break :blk try sc.res(common.quantizedMatmul(&o, x, q, sc.s), &o);
        } else try sc.matmul(x, try sc.t2(self.w));
        return if (self.bias.ctx != null) sc.add(y, self.bias) else y;
    }

    /// Frees what the tower owns of this linear: the quantized triple.
    fn release(self: *Lin) void {
        if (self.q) |*q| q.deinit();
        self.q = null;
    }
};

const Block = struct { norm1: A = none, norm2: A = none, wqkv: Lin = .{}, wo: Lin = .{}, w1: Lin = .{}, w2: Lin = .{} };

fn tensor(weights: *const model.Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, args) catch return error.MissingVisionWeights;
    return weights.get(name) orelse {
        log.warn("MISSING DEEPSEEK VISION WEIGHT: {s}\n", .{name});
        return error.MissingVisionWeights;
    };
}

/// What the tower's load does to its weights: the map the linears come from, and the quantization it ran.
const Load = struct {
    weights: *model.Weights,
    s: mlx.mlx_stream,
    quantize: bool,
    linears: u32 = 0,
    bf16_bytes: u64 = 0,
    quant_bytes: u64 = 0,
};

fn lin(ld: *Load, comptime fmt: []const u8, args: anytype, out_dim: usize, in_dim: usize, with_bias: bool) !Lin {
    var l: Lin = .{ .w = try tensor(ld.weights, fmt ++ ".weight", args) };
    if (with_bias) l.bias = try tensor(ld.weights, fmt ++ ".bias", args);
    const shape = mlx.getShape(l.w);
    if (shape.len != 2 or shape[0] != ci(out_dim) or shape[1] != ci(in_dim) or mlx.mlx_array_dtype(l.w) != .bfloat16) {
        log.warn("DEEPSEEK VISION: " ++ fmt ++ " is not a bf16 [{d}, {d}] weight\n", args ++ .{ out_dim, in_dim });
        return error.InvalidVisionWeights;
    }
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, fmt ++ ".weight", args);
    if (!quantizesAtLoad(name, true, &.{ out_dim, in_dim })) return l;
    if (!ld.quantize) {
        ld.bf16_bytes += 2 * out_dim * in_dim;
        return l;
    }
    if (mlx.engine.isStream(ld.s)) {
        // The codes, scales and biases outlive every forward: the pool's cap rises by them.
        const groups = 2 * out_dim * (in_dim / QUANT_GROUP);
        mlx.engine.room(&.{ out_dim * in_dim * common.QUANT_BITS / 8, groups, groups });
    }
    // The bf16 copy goes as soon as its codes exist: the load holds one linear's worth of both.
    const q = try common.requantize(l.w, QUANT_GROUP, ld.s);
    _ = mlx.mlx_array_free(ld.weights.take(name).?);
    ld.linears += 1;
    ld.bf16_bytes += 2 * out_dim * in_dim;
    ld.quant_bytes += quantizedBytes(out_dim, in_dim);
    l.w = none;
    l.q = q;
    return l;
}

pub const Dsv41Vision = struct {
    allocator: std.mem.Allocator,
    /// MLX's stream: the whole forward runs on it when `own` is off, and so does that load's requantization.
    s: mlx.mlx_stream,
    /// The forward runs on the engine of our own, on its stream and with its ops: set at the load that serves on the engine.
    own: bool = false,
    v: cfg_mod.Vision,
    /// The text width an aligner row and a delimiter take.
    out_dim: u32,
    patch: Lin,
    blocks: []Block,
    norm: A,
    a_w1: Lin,
    a_w2: Lin,
    image_start: A,
    image_end: A,
    image_newline: A,

    /// The serving tower: its block and aligner matrices are requantized to affine 8-bit and their bf16 copies leave
    /// `weights`. The tower's other handles are borrowed from `weights`, which outlives it.
    pub fn init(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Dsv41Vision {
        const quantize_at_load = !common.vision_bf16;
        const c = config.dsv41 orelse return error.MissingVisionWeights;
        const v = c.vision orelse return error.MissingVisionWeights;
        var own = false;
        if (dsv41_borrow.ownDecode(&c)) {
            // Decided here, once: a Cooler load runs nothing on MLX, so a tower the engine cannot run refuses the load.
            if (try engineGap(v, c.dim, quantize_at_load)) |g| {
                log.err("[vision] the engine does not port this tower: an image of {d} rows reaches {s} for a {d} x {d} linear; --ds41-mlx-engine serves it\n", .{ g.rows, g.what, g.out, g.in });
                return error.VisionTowerNotPorted;
            }
            if (!mlx.engine.enabled) return error.EngineNotEnabled;
            own = true;
        }
        // The requantization's kernel is this load's to make, as the warm's are.
        const making: ?mlx.engine.Seal = if (own) mlx.engine.unsealed() else null;
        defer if (making) |seal| seal.end();
        var self = try load(allocator, config, weights, quantize_at_load, own);
        errdefer self.deinit();
        if (own) try self.useEngine();
        return self;
    }

    /// From here the forward runs on the engine of our own, after a run of every pipeline an image can reach.
    pub fn useEngine(self: *Dsv41Vision) !void {
        if (!mlx.engine.enabled) return error.EngineNotEnabled;
        self.own = true;
        errdefer self.own = false;
        try self.warmEngine();
    }

    fn stream(self: *const Dsv41Vision) mlx.mlx_stream {
        return if (self.own) mlx.engine.stream() else self.s;
    }

    /// The engine's stream when the tower runs there: where its caller makes patch arrays and joins spans.
    pub fn ownStream(self: *const Dsv41Vision) ?mlx.mlx_stream {
        return if (self.own) mlx.engine.stream() else null;
    }

    /// The engine runs this pass: its ops take over, an MLX stream among them is refused by name, and its pool may
    /// hold the pass's scratch. A no-op on MLX's side.
    const Run = struct {
        makers: ?mlx.engine.Makers,

        fn end(r: Run) void {
            if (r.makers) |made| made.end();
        }
    };

    fn engage(self: *const Dsv41Vision, patches: usize) !Run {
        if (!self.own) return .{ .makers = null };
        try mlx.engine.budget(scratchOf(self.v, self.out_dim, patches));
        return .{ .makers = mlx.engine.makers(true) };
    }

    /// `quantize` false keeps every linear bf16 (the reference-parity tests).
    pub fn initWith(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights, quantize: bool) !Dsv41Vision {
        return load(allocator, config, weights, quantize, false);
    }

    /// `own`: the requantization runs on the engine of our own, which then holds the codes.
    fn load(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights, quantize: bool, own: bool) !Dsv41Vision {
        const c = config.dsv41 orelse return error.MissingVisionWeights;
        const v = c.vision orelse return error.MissingVisionWeights;
        const d = v.hidden;
        const feat = 3 * v.patch * v.patch;
        const fold = d * v.downsample * v.downsample;
        const blocks = try allocator.alloc(Block, v.layers);
        errdefer allocator.free(blocks);
        for (blocks) |*b| b.* = .{};
        var self: Dsv41Vision = .{
            .allocator = allocator,
            .s = if (own) mlx.engine.stream() else mlx.mlx_default_gpu_stream_new(),
            .v = v,
            .out_dim = c.dim,
            .patch = .{},
            .blocks = blocks,
            .norm = none,
            .a_w1 = .{},
            .a_w2 = .{},
            .image_start = none,
            .image_end = none,
            .image_newline = none,
        };
        errdefer {
            self.release();
            _ = mlx.mlx_stream_free(self.s);
        }
        var ld: Load = .{ .weights = weights, .s = if (own) mlx.engine.stream() else self.s, .quantize = quantize };
        var clock = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        self.patch = try lin(&ld, "vision.patch_embed.proj", .{}, d, feat, true);
        for (blocks, 0..) |*b, i| {
            b.norm1 = try tensor(weights, "vision.blocks.{d}.norm1.weight", .{i});
            b.norm2 = try tensor(weights, "vision.blocks.{d}.norm2.weight", .{i});
            b.wqkv = try lin(&ld, "vision.blocks.{d}.attn.wqkv", .{i}, 3 * d, d, true);
            b.wo = try lin(&ld, "vision.blocks.{d}.attn.wo", .{i}, d, d, true);
            b.w1 = try lin(&ld, "vision.blocks.{d}.mlp.w1", .{i}, 2 * v.inter, d, false);
            b.w2 = try lin(&ld, "vision.blocks.{d}.mlp.w2", .{i}, d, v.inter, false);
        }
        self.norm = try tensor(weights, "vision.norm.weight", .{});
        self.a_w1 = try lin(&ld, "aligner.w1", .{}, c.dim, fold, true);
        self.a_w2 = try lin(&ld, "aligner.w2", .{}, c.dim, c.dim, true);
        self.image_start = try tensor(weights, "image_start", .{});
        self.image_end = try tensor(weights, "image_end", .{});
        self.image_newline = try tensor(weights, "image_newline", .{});
        log.info("Vision encoder: DeepSeek-V4.1 ViT (depth={d}, hidden={d}, heads={d}, patch={d}, downsample={d})\n", .{ v.layers, d, v.heads, v.patch, v.downsample });
        if (!quantize and common.vision_bf16) log.info("[vision] tower kept bf16 (--vision-bf16): {d:.2} GB\n", .{@as(f64, @floatFromInt(ld.bf16_bytes)) / 1e9});
        if (ld.linears > 0) log.info("[vision] tower requantized at load: bf16 {d:.2} GB -> affine{d} g{d} {d:.2} GB ({d} linears, {d} ms)\n", .{
            @as(f64, @floatFromInt(ld.bf16_bytes)) / 1e9,
            common.QUANT_BITS,
            QUANT_GROUP,
            @as(f64, @floatFromInt(ld.quant_bytes)) / 1e9,
            ld.linears,
            clock.read() / std.time.ns_per_ms,
        });
        return self;
    }

    fn release(self: *Dsv41Vision) void {
        self.patch.release();
        for (self.blocks) |*b| for ([_]*Lin{ &b.wqkv, &b.wo, &b.w1, &b.w2 }) |l| l.release();
        self.a_w1.release();
        self.a_w2.release();
    }

    pub fn deinit(self: *Dsv41Vision) void {
        self.release();
        self.allocator.free(self.blocks);
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Encode one image: `patches` is `[grid_h * grid_w, 3 * p * p]`. Returns the trunk's whole span,
    /// `[1, spanRows, out_dim]`, evaluated.
    pub fn forward(self: *Dsv41Vision, patches: A, grid_h: u32, grid_w: u32) !A {
        const out = try self.encode(patches, grid_h, grid_w);
        // The pass's scratch leaves the pool; the span stays.
        if (self.own) mlx.engine.trim();
        return out;
    }

    fn encode(self: *Dsv41Vision, patches: A, grid_h: u32, grid_w: u32) !A {
        transient_peak.begin();
        const run = try self.engage(@as(usize, grid_h) * grid_w);
        defer run.end();
        const hidden = try self.towerHidden(patches, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        const out = try self.tail(hidden, grid_h, grid_w);
        transient_peak.end(.vision_encode, grid_h * grid_w);
        return out;
    }

    /// The aligner and the span of a tower output `hidden`, evaluated.
    fn tail(self: *Dsv41Vision, hidden: A, grid_h: u32, grid_w: u32) !A {
        var sc = Scope.plain(self.allocator, self.stream());
        defer sc.deinit();
        const feats = try self.aligner(&sc, hidden, grid_h, grid_w);
        const out = sc.out(try self.span(&sc, feats, grid_h, grid_w));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// Patch counts whose one-block runs between them reach every pipeline of an image under the release's
    /// geometry: query chunks whole or short of a 64, key counts whole 32s or not, and a last chunk of 1 to 4 and
    /// of 5 to 8 rows (the vector attention kernels).
    const warm_patches = [_]usize{ 1536, 1509, 1632, 1537, 1544 };

    /// Warms that ran in this process: tests read it to see that a load from a manifest ran none.
    pub var warms: u32 = 0;

    fn warmEngine(self: *Dsv41Vision) !void {
        var key_buf: [96]u8 = undefined;
        const v = self.v;
        const key = std.fmt.bufPrint(&key_buf, "{s}/{d}.{d}.{d}.{d}.{d}.{d}/w1", .{ if (self.blocks[0].wqkv.q != null) "q8" else "bf16", v.hidden, v.heads, v.inter, v.patch, v.downsample, self.out_dim }) catch return error.TensorShape;
        // A manifest of this load's key that holds the warm: its pipelines are made, so the run would find them all.
        if (mlx.engine.pipeline_registry.takeMarker(key)) {
            log.info("[vision] engine tower: pipelines made from the manifest, no warm\n", .{});
            return;
        }
        warms += 1;
        const making = mlx.engine.unsealed();
        defer making.end();
        var clock = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        const made_before = mlx.engine.pipelinesMade().count;
        const feat: usize = 3 * self.v.patch * self.v.patch;
        for (warm_patches) |n| {
            const zeros = try self.allocator.alloc(f32, n * feat);
            defer self.allocator.free(zeros);
            @memset(zeros, 0);
            // The engine's own array, made between forwards: the pool's cap rises by it until the pass's cap replaces it.
            mlx.engine.room(&.{zeros.len * @sizeOf(f32)});
            const scope = mlx.engine.enter(self.stream());
            const patches = mlx.mlx_array_new_data(zeros.ptr, &[_]c_int{ ci(n), ci(feat) }, 2, .float32);
            scope.end();
            defer _ = mlx.mlx_array_free(patches);
            const run = try self.engage(n);
            defer run.end();
            const hidden = try self.towerAfter(patches, 1, @intCast(n), 1);
            defer _ = mlx.mlx_array_free(hidden);
            _ = mlx.mlx_array_free(try self.tail(hidden, 1, @intCast(n)));
        }
        mlx.engine.trim();
        log.info("[vision] engine tower warmed: {d} pipelines, {d} ms\n", .{ mlx.engine.pipelinesMade().count - made_before, clock.read() / std.time.ns_per_ms });
        mlx.engine.pipeline_registry.recordMarker("vision tower", key);
        dsv41_pipelines.rewriteLate();
    }

    /// The ViT through its final norm: `[grid_h * grid_w, hidden]`, evaluated; the caller frees it.
    pub fn towerHidden(self: *Dsv41Vision, patches: A, grid_h: u32, grid_w: u32) !A {
        return self.towerAfter(patches, grid_h, grid_w, self.blocks.len);
    }

    /// The stream after the patch embedding and the first `n_blocks` blocks (the final norm only after all of them).
    pub fn towerAfter(self: *Dsv41Vision, patches: A, grid_h: u32, grid_w: u32, n_blocks: usize) !A {
        const v = self.v;
        const n: usize = @as(usize, grid_h) * grid_w;
        const shape = mlx.getShape(patches);
        if (n == 0 or shape.len != 2 or shape[0] != ci(n) or shape[1] != ci(3 * v.patch * v.patch)) return error.InvalidPatchGrid;
        const run = try self.engage(n);
        defer run.end();
        var x: A = blk: {
            var sc = Scope.plain(self.allocator, self.stream());
            defer sc.deinit();
            const e = sc.out(try self.patch.apply(&sc, try sc.astype(patches, .bfloat16)));
            errdefer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_array_eval(e));
            break :blk e;
        };
        defer _ = mlx.mlx_array_free(x);
        var rope_sc = Scope.plain(self.allocator, self.stream());
        defer rope_sc.deinit();
        const rope = try ropeTables(&rope_sc, v, grid_h, grid_w);
        for (self.blocks[0..n_blocks]) |*b| {
            var sc = Scope.plain(self.allocator, self.stream());
            defer sc.deinit();
            const attn = try self.attention(&sc, try sc.rms(x, b.norm1, EPS), b, rope);
            const mid = try sc.add(x, attn);
            const next = sc.out(try sc.add(mid, try self.mlp(&sc, try sc.rms(mid, b.norm2, EPS), b)));
            errdefer _ = mlx.mlx_array_free(next);
            try mlx.check(mlx.mlx_array_eval(next));
            _ = mlx.mlx_array_free(x);
            x = next;
        }
        if (n_blocks < self.blocks.len) {
            var partial = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&partial, x);
            return partial;
        }
        var sc = Scope.plain(self.allocator, self.stream());
        defer sc.deinit();
        const out = sc.out(try sc.rms(x, self.norm, EPS));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    const Rope = struct { cos: A, sin: A };

    /// The release's 2-D rope: each rotate-half pair of a head carries a row angle in its first half of
    /// pairs and a column angle in its second, `[n, 1, head_dim / 2]` f32 apiece.
    fn ropeTables(sc: *Scope, v: cfg_mod.Vision, grid_h: u32, grid_w: u32) !Rope {
        const half: usize = v.headDim() / 2;
        const axis = half / 2;
        var inv: [64]f32 = undefined;
        for (0..axis) |i| {
            const e = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(half));
            inv[i] = @floatCast(1.0 / std.math.pow(f64, v.rope_theta, e));
        }
        const n: usize = @as(usize, grid_h) * grid_w;
        const cos = try sc.a.alloc(f32, n * half);
        defer sc.a.free(cos);
        const sin = try sc.a.alloc(f32, n * half);
        defer sc.a.free(sin);
        for (0..grid_h) |r| for (0..grid_w) |col| {
            const row = r * grid_w + col;
            for (0..axis) |i| {
                const hr = @as(f32, @floatFromInt(r)) * inv[i];
                const wc = @as(f32, @floatFromInt(col)) * inv[i];
                cos[row * half + i] = @cos(hr);
                sin[row * half + i] = @sin(hr);
                cos[row * half + axis + i] = @cos(wc);
                sin[row * half + axis + i] = @sin(wc);
            }
        };
        const shape = [_]c_int{ ci(n), 1, ci(half) };
        return .{ .cos = try sc.keep(mlx.mlx_array_new_data(cos.ptr, &shape, 3, .float32)), .sin = try sc.keep(mlx.mlx_array_new_data(sin.ptr, &shape, 3, .float32)) };
    }

    /// `x [n, heads, head_dim]` rotated in f32, back in its dtype.
    fn rotate(sc: *Scope, x: A, r: Rope) !A {
        const f = try sc.astype(x, .float32);
        const half = sc_mod.dim(x, 2) / 2;
        const x1 = try sc.slice(f, 2, 0, half);
        const x2 = try sc.slice(f, 2, half, 2 * half);
        const lo = try sc.sub(try sc.mul(x1, r.cos), try sc.mul(x2, r.sin));
        const hi = try sc.add(try sc.mul(x2, r.cos), try sc.mul(x1, r.sin));
        return sc.astype(try sc.concat(&.{ lo, hi }, 2), mlx.mlx_array_dtype(x));
    }

    fn attention(self: *Dsv41Vision, sc: *Scope, x: A, b: *const Block, r: Rope) !A {
        const heads: usize = self.v.heads;
        const hd: usize = self.v.headDim();
        const n = sc_mod.dim(x, 0);
        const qkv = try sc.reshape(try b.wqkv.apply(sc, x), &.{ ci(n), 3, ci(heads), ci(hd) });
        var parts: [3]A = undefined;
        inline for (0..3) |j| {
            var p = try sc.reshape(try sc.slice(qkv, 1, j, j + 1), &.{ ci(n), ci(heads), ci(hd) });
            if (j < 2) p = try rotate(sc, p, r);
            parts[j] = try sc.transpose(p, &.{ 1, 0, 2 }); // [heads, n, hd]
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const q = try sc.expand(parts[0], 0);
        const k = try sc.expand(parts[1], 0);
        const val = try sc.expand(parts[2], 0);
        var pieces: std.ArrayList(A) = .empty;
        defer pieces.deinit(self.allocator);
        var lo: usize = 0;
        while (lo < n) : (lo += QUERY_CHUNK) {
            const hi = @min(n, lo + QUERY_CHUNK);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, try sc.slice(q, 2, lo, hi), k, val, scale, "", none, none, false, sc.s));
            const piece = try sc.keep(o);
            try mlx.check(mlx.mlx_array_eval(piece));
            try pieces.append(self.allocator, piece);
        }
        const joined = try sc.concat(pieces.items, 2);
        return b.wo.apply(sc, try sc.reshape(try sc.transpose(joined, &.{ 0, 2, 1, 3 }), &.{ ci(n), ci(heads * hd) }));
    }

    /// `w2(silu(gate) * up)`, gate and up the halves of `w1`'s output; silu rounds to bf16 before the product.
    fn mlp(self: *Dsv41Vision, sc: *Scope, x: A, b: *const Block) !A {
        const gu = try b.w1.apply(sc, x);
        const gate = try sc.astype(try sc.slice(gu, 1, 0, self.v.inter), .float32);
        const up = try sc.slice(gu, 1, self.v.inter, 2 * self.v.inter);
        const act = try sc.astype(try sc.mul(gate, try sc.sigmoid(gate)), .bfloat16);
        return b.w2.apply(sc, try sc.mul(act, up));
    }

    /// `Aligner.forward`: the grid zero-padded to multiples of the ratio, every block unfolded channel-major
    /// into one row, then `w2(gelu(w1(x)))` -> `[llm_h * llm_w, out_dim]`.
    fn aligner(self: *Dsv41Vision, sc: *Scope, hidden: A, grid_h: u32, grid_w: u32) !A {
        const r: usize = self.v.downsample;
        const c: usize = self.v.hidden;
        const lh: usize = llmCells(grid_h, self.v.downsample);
        const lw: usize = llmCells(grid_w, self.v.downsample);
        var x = try sc.reshape(hidden, &.{ ci(grid_h), ci(grid_w), ci(c) });
        if (lh * r != grid_h or lw * r != grid_w) {
            var padded = mlx.mlx_array_new();
            const zero = try sc.keep(mlx.mlx_array_new_float(0.0));
            x = try sc.res(mlx.mlx_pad(&padded, x, &[_]c_int{ 0, 1 }, 2, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(lh * r - grid_h), ci(lw * r - grid_w) }, 2, zero, "constant", sc.s), &padded);
        }
        // [lh, r, lw, r, C] -> [lh, lw, C, r, r]: torch's unfold orders a block's features (c, ky, kx).
        const blocks = try sc.transpose(try sc.reshape(x, &.{ ci(lh), ci(r), ci(lw), ci(r), ci(c) }), &.{ 0, 2, 4, 1, 3 });
        const rows = try sc.reshape(blocks, &.{ ci(lh * lw), ci(c * r * r) });
        const h1 = try self.a_w1.apply(sc, rows);
        return self.a_w2.apply(sc, try gelu(sc, h1));
    }

    /// The trunk's span: start, each aligner row followed by newline, end -> `[1, spanRows, out_dim]`.
    fn span(self: *Dsv41Vision, sc: *Scope, feats: A, grid_h: u32, grid_w: u32) !A {
        const lh: usize = llmCells(grid_h, self.v.downsample);
        const lw: usize = llmCells(grid_w, self.v.downsample);
        const d: c_int = ci(self.out_dim);
        const dt = mlx.mlx_array_dtype(feats);
        const rows = try sc.reshape(feats, &.{ ci(lh), ci(lw), d });
        const nl = try sc.broadcast(try sc.reshape(try sc.astype(self.image_newline, dt), &.{ 1, 1, d }), &.{ ci(lh), 1, d });
        const lines = try sc.reshape(try sc.concat(&.{ rows, nl }, 1), &.{ ci(lh * (lw + 1)), d });
        const start = try sc.reshape(try sc.astype(self.image_start, dt), &.{ 1, d });
        const end = try sc.reshape(try sc.astype(self.image_end, dt), &.{ 1, d });
        return sc.expand(try sc.concat(&.{ start, lines, end }, 0), 0);
    }
};

/// Exact (erf) GELU in f32, rounded back to the input's dtype as torch's bf16 op does.
fn gelu(sc: *Scope, x: A) !A {
    const f = try sc.astype(x, .float32);
    const e = try sc.op1(mlx.mlx_erf, try sc.mul(f, try sc.f(1.0 / @sqrt(2.0))));
    return sc.astype(try sc.mul(try sc.mul(f, try sc.f(0.5)), try sc.add(e, try sc.f(1.0))), mlx.mlx_array_dtype(x));
}

/// GPU bytes one encode of `patches` ViT patches peaks at: activations per block, the query-chunked score
/// sheet and the aligner. A bound, not a measurement.
pub fn encodeScratchBytes(cfg: *const model.ModelConfig, patches: u64) u64 {
    return scratchBound(cfg.qv_heads, cfg.qv_hidden, cfg.qv_intermediate, cfg.qv_out_hidden, cfg.qv_merge, patches);
}

fn scratchOf(v: cfg_mod.Vision, out_hidden: u32, patches: usize) u64 {
    return scratchBound(v.heads, v.hidden, v.inter, out_hidden, v.downsample, patches);
}

fn scratchBound(heads: u64, hidden: u64, inter: u64, out_hidden: u64, merge: u64, patches: u64) u64 {
    const scores = 4 * heads * @min(patches, @as(u64, QUERY_CHUNK)) * patches;
    const activations = 4 * patches * (12 * hidden + 6 * inter);
    const aligner = 4 * (patches / @max(merge * merge, 1)) * (hidden * merge * merge + 4 * out_hidden);
    return (64 << 20) + scores * 13 / 10 + @max(activations, aligner);
}

// ── Tests ──

const testing = std.testing;

const release_vision: Processor = .{ .patch = 14, .downsample = 3, .max_tokens = 1024, .min_pixels = 295936, .max_wh_ratio = 0 };

test "dsv41 vision: the resize plan and letterbox match the release's image processor" {
    // Rows of the release's `plan_image_grid` and Pillow's `ImageOps.pad` box at the release config
    // (tests/dsv41/dump_dsv41_vision_fixtures.py writes them from the reference as plan.json).
    const cases = [_]struct { w: u32, h: u32, bw: u32, bh: u32, lh: u32, lw: u32, rw: u32, rh: u32, x: u32, y: u32 }{
        .{ .w = 730, .h = 487, .bw = 742, .bh = 490, .lh = 12, .lw = 18, .rw = 734, .rh = 490, .x = 4, .y = 0 },
        .{ .w = 97, .h = 61, .bw = 686, .bh = 434, .lh = 11, .lw = 17, .rw = 686, .rh = 431, .x = 0, .y = 2 },
        .{ .w = 7000, .h = 1000, .bw = 3528, .bh = 504, .lh = 12, .lw = 84, .rw = 3528, .rh = 504, .x = 0, .y = 0 },
        .{ .w = 4000, .h = 3000, .bw = 1512, .bh = 1134, .lh = 27, .lw = 36, .rw = 1512, .rh = 1134, .x = 0, .y = 0 },
        .{ .w = 14, .h = 14, .bw = 546, .bh = 546, .lh = 13, .lw = 13, .rw = 546, .rh = 546, .x = 0, .y = 0 },
        .{ .w = 1, .h = 1, .bw = 546, .bh = 546, .lh = 13, .lw = 13, .rw = 546, .rh = 546, .x = 0, .y = 0 },
        .{ .w = 5000, .h = 50, .bw = 5446, .bh = 56, .lh = 2, .lw = 130, .rw = 5446, .rh = 54, .x = 0, .y = 1 },
        .{ .w = 50, .h = 5000, .bw = 56, .bh = 5446, .lh = 130, .lw = 2, .rw = 54, .rh = 5446, .x = 1, .y = 0 },
        .{ .w = 1920, .h = 1080, .bw = 1708, .bh = 966, .lh = 23, .lw = 41, .rw = 1708, .rh = 961, .x = 0, .y = 2 },
        .{ .w = 640, .h = 640, .bw = 644, .bh = 644, .lh = 16, .lw = 16, .rw = 644, .rh = 644, .x = 0, .y = 0 },
        .{ .w = 1000, .h = 7000, .bw = 462, .bh = 3234, .lh = 77, .lw = 11, .rw = 462, .rh = 3234, .x = 0, .y = 0 },
        .{ .w = 300, .h = 2900, .bw = 308, .bh = 2912, .lh = 70, .lw = 8, .rw = 301, .rh = 2912, .x = 4, .y = 0 },
    };
    for (cases) |c| {
        const p = planGrid(c.w, c.h, release_vision).?;
        const box = letterbox(c.w, c.h, p.w, p.h);
        try testing.expectEqual(Plan{ .h = c.bh, .w = c.bw, .vit_h = c.bh / 14, .vit_w = c.bw / 14, .llm_h = c.lh, .llm_w = c.lw, .stretch = false }, p);
        try testing.expectEqual(Letterbox{ .rw = c.rw, .rh = c.rh, .x = c.x, .y = c.y }, box);
        try testing.expect(spanTokens(p.llm_h, p.llm_w) <= 1024);
        try testing.expectEqual(spanTokens(p.llm_h, p.llm_w), spanRows(p.vit_h, p.vit_w, 3));
    }
    // A wh cap stretches a wider image into the capped canvas instead of letterboxing it.
    var capped = release_vision;
    capped.max_wh_ratio = 2;
    const wide = planGrid(4000, 1000, capped).?;
    try testing.expectEqual(Plan{ .h = 924, .w = 1848, .vit_h = 66, .vit_w = 132, .llm_h = 22, .llm_w = 44, .stretch = true }, wide);
    var degenerate = release_vision;
    degenerate.min_pixels = 0;
    try testing.expect(planGrid(0, 10, degenerate) == null);
}

test "dsv41 vision: patches hold the letterboxed image in (c, py, px) order and gray elsewhere" {
    const a = testing.allocator;
    var v = release_vision;
    v.min_pixels = 0;
    // A 28x14 image already at the canvas size: patch 0 is the left half, patch 1 the right.
    var rgb: [28 * 14 * 3]u8 = undefined;
    for (0..14) |y| for (0..28) |x| {
        const val: u8 = if (x < 14) 255 else 0;
        rgb[(y * 28 + x) * 3 + 0] = val;
        rgb[(y * 28 + x) * 3 + 1] = 51;
        rgb[(y * 28 + x) * 3 + 2] = val;
    };
    const px = try preprocess(a, &rgb, 14, 28, v);
    defer a.free(px.bytes);
    try testing.expectEqual(@as(u32, 1), px.grid_h);
    try testing.expectEqual(@as(u32, 2), px.grid_w);
    const f = @as([*]const f32, @ptrCast(@alignCast(px.bytes.ptr)))[0 .. px.bytes.len / 4];
    try testing.expectEqual(@as(usize, 2 * 588), f.len);
    try testing.expectApproxEqAbs(@as(f32, 1.0), f[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 51.0 / 255.0 * 2.0 - 1.0), f[196], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1.0), f[588], 1e-6);

    // A 28x7 image lands in a 28x14 canvas centered (rows 4..10): the rows above and below are the fill.
    const small = try preprocess(a, rgb[0 .. 28 * 7 * 3], 7, 28, v);
    defer a.free(small.bytes);
    const g = @as([*]const f32, @ptrCast(@alignCast(small.bytes.ptr)))[0 .. small.bytes.len / 4];
    try testing.expectEqual(@as(u32, 2), small.grid_w);
    try testing.expectApproxEqAbs(normalizePixel(127), g[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), g[4 * 14], 1e-6);
    try testing.expectApproxEqAbs(normalizePixel(127), g[11 * 14], 1e-6);
}

test "dsv41 vision: the encode bill grows with the patches and stays under 2 GB at the processor's cap" {
    const cfg = model.ModelConfig{ .qv_heads = 16, .qv_hidden = 1024, .qv_intermediate = 2816, .qv_out_hidden = 5120, .qv_merge = 3 };
    const max_patches: u64 = 9 * 1022;
    const cap = encodeScratchBytes(&cfg, max_patches);
    try testing.expect(cap > encodeScratchBytes(&cfg, max_patches / 2));
    try testing.expect(cap < 2 * (1 << 30));
}

// ── Parity against the release's reference (tests/dsv41/dump_dsv41_vision_fixtures.py) ──
//
// The fixture holds the reference's bf16 processor patches and tower/aligner outputs on two images, and the
// pack's own tower tensors; both sit under `kld-teacher/dsv41-vision` in the models root. The tests skip
// when it is absent.

fn fixtureFile(buf: []u8, comptime name: [:0]const u8) ![:0]const u8 {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("../test_models.zig").packPath(&dir_buf, "kld-teacher/dsv41-vision");
    const text = std.fmt.bufPrint(buf, "{s}/" ++ name ++ "\x00", .{dir}) catch return error.SkipZigTest;
    const path = text[0 .. text.len - 1 :0];
    (std.Io.Dir.openFileAbsolute(testing.io, path, .{}) catch return error.SkipZigTest).close(testing.io);
    return path;
}

fn loadFile(path: [*:0]const u8) !model.Weights {
    var weights = model.Weights.init(testing.allocator);
    errdefer weights.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try model.loadSafetensorsFile(testing.allocator, &weights, path, cpu, true);
    return weights;
}

fn fixtureTensor(fx: *const model.Weights, case: []const u8, comptime what: []const u8) !A {
    var buf: [64]u8 = undefined;
    const a = fx.get(try std.fmt.bufPrint(&buf, "{s}_" ++ what, .{case})) orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(a));
    return a;
}

/// Round to the nearest bf16 (ties to even), as a float.
fn bf16Round(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    const rounded = (bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000;
    return @bitCast(rounded);
}

fn releaseConfig() !model.ModelConfig {
    var cfg: model.ModelConfig = .{};
    cfg.dsv41 = try cfg_mod.parseText(cfg_mod.release_config);
    return cfg;
}

test "dsv41 vision: the processor's patches equal the reference's, bf16 for bf16" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var fx = try loadFile((try fixtureFile(&buf, "fixture.safetensors")).ptr);
    defer fx.deinit();
    const proc = Processor.of((try releaseConfig()).dsv41.?.vision.?);
    for ([_][]const u8{ "odd", "photo" }) |case| {
        const rgb_arr = try fixtureTensor(&fx, case, "rgb");
        const shape = mlx.getShape(rgb_arr);
        const h: u32 = @intCast(shape[0]);
        const w: u32 = @intCast(shape[1]);
        const rgb = mlx.mlx_array_data_uint8(rgb_arr).?[0 .. @as(usize, h) * w * 3];
        const px = try preprocess(testing.allocator, rgb, h, w, proc);
        defer testing.allocator.free(px.bytes);
        const grid_arr = try fixtureTensor(&fx, case, "grid");
        const grid = mlx.mlx_array_data_int32(grid_arr).?[0..4];
        try testing.expectEqual(@as(u32, @intCast(grid[0])), px.grid_h);
        try testing.expectEqual(@as(u32, @intCast(grid[1])), px.grid_w);
        try testing.expectEqual(@as(usize, @intCast(grid[2] * grid[3])), llmCells(px.grid_h, 3) * llmCells(px.grid_w, 3));
        const want_arr = try fixtureTensor(&fx, case, "patches");
        const got = @as([*]const f32, @ptrCast(@alignCast(px.bytes.ptr)))[0 .. px.bytes.len / 4];
        try testing.expectEqual(mlx.mlx_array_size(want_arr), got.len);
        const want = mlx.mlx_array_data_float32(want_arr).?[0..got.len];
        var worst: f32 = 0;
        for (got, want) |g, x| worst = @max(worst, @abs(bf16Round(g) - x));
        errdefer std.debug.print("[dsv41-vision] {s} patches max|diff|={e}\n", .{ case, worst });
        try testing.expectEqual(@as(f32, 0), worst);
    }
}

test "dsv41 vision: tower, aligner and span match the reference" {
    try mlx.requireGpu();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var fx = try loadFile((try fixtureFile(&buf, "fixture.safetensors")).ptr);
    defer fx.deinit();
    var tower_buf: [std.fs.max_path_bytes]u8 = undefined;
    var weights = try loadFile((try fixtureFile(&tower_buf, "tower.safetensors")).ptr);
    defer weights.deinit();
    var dv = try Dsv41Vision.initWith(testing.allocator, try releaseConfig(), &weights, false);
    defer dv.deinit();
    for ([_][]const u8{ "odd", "photo" }) |case| {
        const grid_arr = try fixtureTensor(&fx, case, "grid");
        const grid = mlx.mlx_array_data_int32(grid_arr).?[0..4];
        const gh: u32 = @intCast(grid[0]);
        const gw: u32 = @intCast(grid[1]);
        const patches = try fixtureTensor(&fx, case, "patches");
        const embed = try dv.towerAfter(patches, gh, gw, 0);
        defer _ = mlx.mlx_array_free(embed);
        const block0 = try dv.towerAfter(patches, gh, gw, 1);
        defer _ = mlx.mlx_array_free(block0);
        const tower = try dv.towerHidden(patches, gh, gw);
        defer _ = mlx.mlx_array_free(tower);
        const span = try dv.forward(patches, gh, gw);
        defer _ = mlx.mlx_array_free(span);
        try testing.expectEqual(@as(c_int, @intCast(spanRows(gh, gw, 3))), mlx.getShape(span)[1]);
        const Stage = struct { name: []const u8, got: A, want: A, min_cos: f32 };
        for ([_]Stage{
            // The first stages are bf16 rounding noise only; deeper ones carry the 32-block drift of two bf16 stacks.
            .{ .name = "embed", .got = embed, .want = try fixtureTensor(&fx, case, "embed"), .min_cos = 0.99999 },
            .{ .name = "block0", .got = block0, .want = try fixtureTensor(&fx, case, "block0"), .min_cos = 0.9999 },
            .{ .name = "tower", .got = tower, .want = try fixtureTensor(&fx, case, "tower"), .min_cos = 0.99 },
            .{ .name = "span", .got = span, .want = try fixtureTensor(&fx, case, "span"), .min_cos = 0.995 },
        }) |stage| {
            const cos = try common.cosineSim(stage.got, stage.want, dv.s);
            const ratio = try common.rmsRatio(stage.got, stage.want, dv.s);
            errdefer std.debug.print("[dsv41-vision] {s} {s} cosine={d:.7} rms_ratio={d:.7}\n", .{ case, stage.name, cos, ratio });
            try testing.expect(cos > stage.min_cos);
            try testing.expect(ratio > 0.99 and ratio < 1.01);
        }
        // The yardstick for the drift: the reference's own fp32 run. Our bf16 tower is as close to it as the
        // reference's bf16 tower is.
        const f32_tower = try fixtureTensor(&fx, case, "tower_f32");
        const ours = try common.cosineSim(tower, f32_tower, dv.s);
        const theirs = try common.cosineSim(try fixtureTensor(&fx, case, "tower"), f32_tower, dv.s);
        errdefer std.debug.print("[dsv41-vision] {s} cosine to the fp32 tower: ours {d:.7}, reference bf16 {d:.7}\n", .{ case, ours, theirs });
        try testing.expect(ours > theirs - 0.002);
    }
}

test "dsv41 vision: the tokenizer reads the image atom as the config's image token, once" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("../test_models.zig").packPath(&buf, "deepseek-ai/DeepSeek-V4.1-Flash-1017");
    var tok = @import("../tokenizer.zig").loadTokenizer(testing.io, testing.allocator, dir) catch return error.SkipZigTest;
    defer tok.deinit();
    const c = (try releaseConfig()).dsv41.?;
    const ids = try tok.encode(testing.allocator, "<\xef\xbd\x9cUser\xef\xbd\x9c><\xef\xbd\x9cdeepseek_image\xef\xbd\x9c>\n\nWhat is this?");
    defer testing.allocator.free(ids);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u32, ids, &.{c.image_token_id}));
}

// ── Requantization at load ──

test "dsv41 vision: the load requantizes the block and aligner matrices of whole groups and nothing else" {
    const bf = true;
    try testing.expect(quantizesAtLoad("vision.blocks.0.attn.wqkv.weight", bf, &.{ 3072, 1024 }));
    try testing.expect(quantizesAtLoad("vision.blocks.31.attn.wo.weight", bf, &.{ 1024, 1024 }));
    try testing.expect(quantizesAtLoad("vision.blocks.5.mlp.w1.weight", bf, &.{ 5632, 1024 }));
    try testing.expect(quantizesAtLoad("vision.blocks.5.mlp.w2.weight", bf, &.{ 1024, 2816 }));
    try testing.expect(quantizesAtLoad("aligner.w1.weight", bf, &.{ 5120, 9216 }));
    try testing.expect(quantizesAtLoad("aligner.w2.weight", bf, &.{ 5120, 5120 }));
    // The patch embedding is 588 wide; norms, biases and the delimiters are not block matrices.
    try testing.expect(!quantizesAtLoad("vision.patch_embed.proj.weight", bf, &.{ 1024, 588 }));
    try testing.expect(!quantizesAtLoad("vision.blocks.0.norm1.weight", bf, &.{1024}));
    try testing.expect(!quantizesAtLoad("vision.blocks.0.attn.wqkv.bias", bf, &.{3072}));
    try testing.expect(!quantizesAtLoad("image_start", bf, &.{5120}));
    try testing.expect(!quantizesAtLoad("layers.3.ffn.gate.weight", bf, &.{ 256, 4096 }));
    try testing.expect(!quantizesAtLoad("vision.blocks.0.mlp.w2.weight", bf, &.{ 1024, 2800 }));
    try testing.expect(!quantizesAtLoad("vision.blocks.0.mlp.w2.weight", false, &.{ 1024, 2816 }));
    // Codes plus a bf16 scale and bias per group.
    try testing.expectEqual(@as(u64, 1024 * 1024 + 4 * 1024 * 8), quantizedBytes(1024, 1024));
}

fn putBf16(w: *model.Weights, name: []const u8, shape: []const c_int, seed: u64, scale: f32, base: f32) !void {
    try common.putRandomBf16(w.allocator, &w.map, name, shape, seed, scale, base);
}

const tiny_vision: cfg_mod.Vision = .{ .layers = 2, .hidden = 256, .heads = 4, .inter = 256, .patch = 4, .rope_theta = 10000, .downsample = 2, .max_tokens = 64, .min_pixels = 0, .max_wh_ratio = 0 };
const tiny_dim: u32 = 128;

/// A synthetic tower of the release's structure at widths of whole 128-groups.
fn tinyTower() !model.Weights {
    return towerOf(tiny_vision, tiny_dim);
}

fn towerOf(v: cfg_mod.Vision, dim: u32) !model.Weights {
    const d: c_int = @intCast(v.hidden);
    const inter: c_int = @intCast(v.inter);
    var w = model.Weights.init(testing.allocator);
    errdefer w.deinit();
    var buf: [96]u8 = undefined;
    var seed: u64 = 1;
    const feat: c_int = @intCast(3 * v.patch * v.patch);
    try putBf16(&w, "vision.patch_embed.proj.weight", &.{ d, feat }, 1, 0.1, 0);
    try putBf16(&w, "vision.patch_embed.proj.bias", &.{d}, 2, 0.05, 0);
    for (0..v.layers) |i| {
        const Part = struct { name: []const u8, shape: [2]c_int, rank: u2 };
        const parts = [_]Part{
            .{ .name = "norm1.weight", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "norm2.weight", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "attn.wqkv.weight", .shape = .{ 3 * d, d }, .rank = 2 },
            .{ .name = "attn.wqkv.bias", .shape = .{ 3 * d, 0 }, .rank = 1 },
            .{ .name = "attn.wo.weight", .shape = .{ d, d }, .rank = 2 },
            .{ .name = "attn.wo.bias", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "mlp.w1.weight", .shape = .{ 2 * inter, d }, .rank = 2 },
            .{ .name = "mlp.w2.weight", .shape = .{ d, inter }, .rank = 2 },
        };
        for (parts) |p| {
            seed += 1;
            const norm = std.mem.startsWith(u8, p.name, "norm");
            try putBf16(&w, try std.fmt.bufPrint(&buf, "vision.blocks.{d}.{s}", .{ i, p.name }), p.shape[0..p.rank], seed, if (norm) 0.1 else 0.1, if (norm) 1 else 0);
        }
    }
    try putBf16(&w, "vision.norm.weight", &.{d}, 80, 0.1, 1);
    const out: c_int = @intCast(dim);
    try putBf16(&w, "aligner.w1.weight", &.{ out, d * @as(c_int, @intCast(v.downsample * v.downsample)) }, 81, 0.1, 0);
    try putBf16(&w, "aligner.w1.bias", &.{out}, 82, 0.05, 0);
    try putBf16(&w, "aligner.w2.weight", &.{ out, out }, 83, 0.1, 0);
    try putBf16(&w, "aligner.w2.bias", &.{out}, 84, 0.05, 0);
    for ([_][]const u8{ "image_start", "image_end", "image_newline" }) |name| try putBf16(&w, name, &.{out}, 70, 0.1, 0);
    return w;
}

fn tinyConfig() model.ModelConfig {
    return configOf(tiny_vision, tiny_dim);
}

fn configOf(v: cfg_mod.Vision, dim: u32) model.ModelConfig {
    var cfg: model.ModelConfig = .{};
    cfg.dsv41 = .{ .dim = dim, .vision = v };
    return cfg;
}

test "dsv41 vision: requantized tower holds affine 8-bit g128 codes, frees the bf16 copies and tracks the bf16 tower" {
    try mlx.requireGpu();
    var wq = try tinyTower();
    defer wq.deinit();
    var wb = try tinyTower();
    defer wb.deinit();
    const before = wq.count();
    var dq = try Dsv41Vision.init(testing.allocator, tinyConfig(), &wq);
    defer dq.deinit();
    var db = try Dsv41Vision.initWith(testing.allocator, tinyConfig(), &wb, false);
    defer db.deinit();

    // Two blocks of wqkv, wo, w1, w2 and the aligner's two: their bf16 entries leave the map, the rest stay.
    const linears: u32 = 2 * 4 + 2;
    try testing.expectEqual(before - linears, wq.count());
    try testing.expectEqual(before, wb.count());
    try testing.expect(wq.get("vision.blocks.1.mlp.w2.weight") == null and wq.get("aligner.w1.weight") == null);
    try testing.expect(wq.get("vision.patch_embed.proj.weight") != null and wq.get("vision.blocks.0.attn.wqkv.bias") != null);
    try testing.expect(dq.patch.q == null and dq.patch.w.ctx != null);

    var bytes: u64 = 0;
    for ([_]*const Lin{ &dq.blocks[0].wqkv, &dq.blocks[1].w2, &dq.a_w1 }) |l| {
        const q = l.q.?;
        try testing.expectEqual(@as(u32, 8), q.bits);
        try testing.expectEqual(@as(u32, 128), q.group_size);
        try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(q.w));
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(q.scales));
        try testing.expectEqual(@divExact(mlx.getShape(q.w)[1] * 4, 128), mlx.getShape(q.scales)[1]);
        bytes += mlx.mlx_array_size(q.w) * 4 + (mlx.mlx_array_size(q.scales) + mlx.mlx_array_size(q.biases)) * 2;
    }
    try testing.expectEqual(quantizedBytes(768, 256) + quantizedBytes(256, 256) + quantizedBytes(128, 1024), bytes);

    var prng = std.Random.DefaultPrng.init(7);
    const grid_h: u32 = 6;
    const grid_w: u32 = 8;
    var px: [6 * 8 * 48]f32 = undefined;
    for (&px) |*x| x.* = prng.random().float(f32) * 2 - 1;
    const patches = mlx.mlx_array_new_data(&px, &[_]c_int{ grid_h * grid_w, 48 }, 2, .float32);
    defer _ = mlx.mlx_array_free(patches);
    const got = try dq.forward(patches, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(got);
    const want = try db.forward(patches, grid_h, grid_w);
    defer _ = mlx.mlx_array_free(want);
    const cos = try common.cosineSim(got, want, dq.s);
    errdefer std.debug.print("[dsv41-vision] tiny q8 vs bf16 cosine={d:.6}\n", .{cos});
    try testing.expect(cos > 0.999);
}

test "dsv41 vision: --vision-bf16 keeps bf16 linears and matches the bf16 arm byte for byte" {
    try mlx.requireGpu();
    var wk = try tinyTower();
    defer wk.deinit();
    var wb = try tinyTower();
    defer wb.deinit();
    const before = wk.count();
    common.vision_bf16 = true;
    defer common.vision_bf16 = false;
    var dk = try Dsv41Vision.init(testing.allocator, tinyConfig(), &wk);
    defer dk.deinit();
    var db = try Dsv41Vision.initWith(testing.allocator, tinyConfig(), &wb, false);
    defer db.deinit();

    try testing.expectEqual(before, wk.count());
    for ([_]*const Lin{ &dk.blocks[0].wqkv, &dk.blocks[1].w2, &dk.a_w1, &dk.patch }) |l| {
        try testing.expect(l.q == null);
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(l.w));
    }

    var px: [6 * 8 * 48]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&px) |*x| x.* = prng.random().float(f32) * 2 - 1;
    const patches = mlx.mlx_array_new_data(&px, &[_]c_int{ 6 * 8, 48 }, 2, .float32);
    defer _ = mlx.mlx_array_free(patches);
    const got = try dk.forward(patches, 6, 8);
    defer _ = mlx.mlx_array_free(got);
    const want = try db.forward(patches, 6, 8);
    defer _ = mlx.mlx_array_free(want);
    const g = try common.gap(got, want, dk.s);
    try testing.expectEqual(@as(f64, 0), g.max_abs);
}

/// Every linear's inner width is one group, so MLX's quantized matmul never splits K for these rows.
const engine_vision: cfg_mod.Vision = .{ .layers = 2, .hidden = 128, .heads = 2, .inter = 128, .patch = 4, .rope_theta = 10000, .downsample = 1, .max_tokens = 4096, .min_pixels = 0, .max_wh_ratio = 0 };

test "dsv41 vision: the tower on the engine's stream gives MLX's bytes, with no MLX op, no lend and no pipeline made after the warm" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;
    var wm = try towerOf(engine_vision, tiny_dim);
    defer wm.deinit();
    var we = try towerOf(engine_vision, tiny_dim);
    defer we.deinit();
    var on_mlx = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &wm);
    defer on_mlx.deinit();
    var on_engine = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &we);
    defer on_engine.deinit();
    try on_engine.useEngine();
    defer mlx.engine.seal(false);
    // A pass leaves its pool budget in place, as a forward does; a later test sees no limit.
    defer mlx.engine.budget(1 << 62) catch {};

    // 36 x 44: chunks of 512 and a last of 48; 1 x 1544: a last chunk of 8 rows, the vector kernels.
    for ([_][2]u32{ .{ 36, 44 }, .{ 1, 1544 } }) |grid| {
        const n: usize = @as(usize, grid[0]) * grid[1];
        const px = try testing.allocator.alloc(f32, n * 48);
        defer testing.allocator.free(px);
        var prng = std.Random.DefaultPrng.init(n);
        for (px) |*x| x.* = prng.random().float(f32) * 2 - 1;
        const patches = mlx.mlx_array_new_data(px.ptr, &[_]c_int{ ci(n), 48 }, 2, .float32);
        defer _ = mlx.mlx_array_free(patches);
        const want = try on_mlx.forward(patches, grid[0], grid[1]);
        defer _ = mlx.mlx_array_free(want);

        const ops_before = mlx.engine.mlx_ops;
        const lends_before = mlx.engine.stats().lends;
        const made_before = mlx.engine.pipelinesMade().count;
        const got = try on_engine.forward(patches, grid[0], grid[1]);
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqual(ops_before, mlx.engine.mlx_ops);
        try testing.expectEqual(lends_before, mlx.engine.stats().lends);
        try testing.expectEqual(made_before, mlx.engine.pipelinesMade().count);

        try testing.expect(mlx.engine.isOurs(got) and !mlx.engine.isOurs(want));
        try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
        try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
        const bytes = mlx.mlx_array_size(want) * mlx.mlx_array_itemsize(want);
        try testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(want).?[0..bytes], mlx.mlx_array_data_uint8(got).?[0..bytes]);
    }
}

/// `towerOf` with every weight the engine's own tensor, as a Cooler load holds them.
fn ownTowerOf(v: cfg_mod.Vision, dim: u32) !model.Weights {
    var w = try towerOf(v, dim);
    errdefer w.deinit();
    var it = w.map.valueIterator();
    while (it.next()) |a| {
        const shape = mlx.getShape(a.*);
        const ours = mlx.engine.mlx_array_new_data(mlx.mlx_array_data_uint8(a.*).?, shape.ptr, @intCast(shape.len), mlx.mlx_array_dtype(a.*));
        if (ours.ctx == null) return error.OutOfMemory;
        _ = mlx.mlx_array_free(a.*);
        a.* = ours;
    }
    return w;
}

test "dsv41 vision: an image the engine's quantized matmul cannot take is refused by name, not run on MLX" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;
    var w = try towerOf(engine_vision, tiny_dim);
    defer w.deinit();
    var tower = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &w);
    defer tower.deinit();
    tower.own = true;
    defer mlx.engine.budget(1 << 62) catch {};
    var px: [4 * 4 * 48]f32 = @splat(0.25);
    const patches = mlx.mlx_array_new_data(&px, &[_]c_int{ 16, 48 }, 2, .float32);
    defer _ = mlx.mlx_array_free(patches);
    var buf: [512]u8 = undefined;
    _ = mlx.takeError(&buf);
    const ops_before = mlx.engine.mlx_ops;
    try testing.expectError(error.MlxError, tower.forward(patches, 4, 4));
    try testing.expect(std.mem.indexOf(u8, mlx.takeError(&buf).?, "[engine] quantized_matmul") != null);
    try testing.expectEqual(ops_before, mlx.engine.mlx_ops);
    try testing.expect(!mlx.engine.active);
}

// Rows the tower can see, for any image the release processor admits (`reachOf`; `planGrid` over every width under 128 px
// and a geometric grid of sizes up to 16,384 px, and exhaustively over widths 1 to 79 and heights 1 to 200,000 below):
//   - ViT patches, the rows of every ViT linear and the keys of every attention call: 1,084 at the fewest (a 79 x 14,281
//     image, cut by the token cap to a 28 x 7,588 canvas; the 546 px square a tiny image scales up to has 1,521) to
//     9,198 (`maxPatches`, 1,022 aligner cells of nine);
//   - aligner cells, the rows of the aligner's two linears: 164 at the fewest (a 3 x 481 image) to 1,022;
//   - an attention call's query chunk: 512 rows, and a last chunk of 1 to 512, against those keys.
// MLX takes its vector kernels below 33 rows (13 for the aligner's 9,216 inner width) and its split-K plan while the
// output tiles are few: from 257 rows against 1,024 outputs and 33 against the aligner's 5,120 the plan is the NAX
// tile this port covers (`quantized.unported`). Every admitted image gives 4x that or more, so the vector kernels and
// split-K are reached by no request; `engineGap` checks it at the load, with a factor of two for the sampling, and a
// pack whose vision config could reach them keeps its tower on MLX, said in a log line.

fn releaseVision() !cfg_mod.Vision {
    return (try cfg_mod.parseText(cfg_mod.release_config)).vision.?;
}

test "dsv41 vision: every image the release processor admits gives the tower rows the engine's plans cover" {
    try mlx.requireGpu();
    const v = try releaseVision();
    const fine = reachOf(Processor.of(v), 1.013).?;
    try testing.expect(fine.min_patches >= 1084 and fine.min_patches < 1100);
    try testing.expect(fine.max_patches <= v.maxPatches() and fine.max_patches > 9000);
    try testing.expect(fine.min_cells >= 164 and fine.min_cells < 170);
    try testing.expect(fine.max_cells <= 1022 and fine.max_cells > 1000);
    // The rows from which MLX's plan is one the engine ports, per linear.
    const g = try matmul.gpu();
    try testing.expectEqual(@as(usize, 257), portedFrom(g, 1024, 1024));
    try testing.expectEqual(@as(usize, 33), portedFrom(g, 5120, 9216));
    try testing.expectEqual(@as(usize, 65), portedFrom(g, 3072, 1024));
    try testing.expect(try engineGap(v, 5120, true) == null);

    // A processor with no pixel floor admits a one-patch image, which no chunk of keys covers; with a floor of about
    // forty patches the linears take MLX's vector kernels.
    var floorless = v;
    floorless.min_pixels = 0;
    try testing.expectEqualStrings("fewer keys than a vector attention chunk", (try engineGap(floorless, 5120, true)).?.what);
    floorless.min_pixels = 196 * 40;
    const gap = (try engineGap(floorless, 5120, true)).?;
    try testing.expectEqualStrings("a plan the engine does not port", gap.what);
    try testing.expect(gap.rows < 100);
    var wide_head = v;
    wide_head.hidden = 2048;
    try testing.expect((try engineGap(wide_head, 5120, true)) != null);
}

test "dsv41 vision: a Cooler load refuses a tower the engine does not port by name; the MLX route loads it" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;
    // No pixel floor: a one-patch image is admissible, and no chunk of keys covers it.
    var w = try towerOf(engine_vision, tiny_dim);
    defer w.deinit();
    try testing.expect(try engineGap(engine_vision, tiny_dim, true) != null);
    dsv41_borrow.own_override = true;
    defer dsv41_borrow.own_override = null;
    const calls = mlx.engine.mlxCalls();
    try testing.expectError(error.VisionTowerNotPorted, Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &w));
    try testing.expectEqual(calls, mlx.engine.mlxCalls());
    dsv41_borrow.own_override = false;
    var on_mlx = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &w);
    defer on_mlx.deinit();
    try testing.expect(!on_mlx.own and on_mlx.ownStream() == null);
}

test "dsv41 vision: the fewest patches and aligner cells any image gives the release tower, over every thin size" {
    const p = Processor.of(try releaseVision());
    var least: ?Reach = null;
    for (1..80) |w| {
        for (1..200_001) |h| grow(&least, p, @intCast(w), @intCast(h));
    }
    try testing.expectEqual(@as(u64, 1084), least.?.min_patches);
    try testing.expectEqual(@as(u64, 164), least.?.min_cells);
}

test "dsv41 vision: at the release's geometry the engine's tower gives MLX's bytes for the smallest, a thin and a large image" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;
    dsv41_pipelines.forget();
    var v = try releaseVision();
    v.layers = 2;
    var wm = try towerOf(v, 5120);
    defer wm.deinit();
    var we = try ownTowerOf(v, 5120);
    defer we.deinit();
    var on_mlx = try Dsv41Vision.init(testing.allocator, configOf(v, 5120), &wm);
    defer on_mlx.deinit();
    defer mlx.engine.budget(1 << 62) catch {};
    // A load that follows a forward meets that forward's cap: here a sixth of the aligner's first codes.
    try mlx.engine.budget(8 << 20);
    const load_ops = mlx.engine.mlx_ops;
    const load_stats = mlx.engine.stats();
    dsv41_borrow.own_override = true;
    defer dsv41_borrow.own_override = null;
    // The model's load has sealed the engine by the time the tower loads.
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();
    var on_engine = try Dsv41Vision.init(testing.allocator, configOf(v, 5120), &we);
    defer on_engine.deinit();
    try testing.expectEqual(late, mlx.engine.latePipelines());
    // The Cooler load requantized on the engine: nothing asked of MLX, no weight taken from it or lent to it.
    try testing.expect(on_engine.own and mlx.engine.isOurs(on_engine.blocks[0].wqkv.q.?.w));
    try testing.expectEqual(load_ops, mlx.engine.mlx_ops);
    try testing.expectEqual(load_stats.imports, mlx.engine.stats().imports);
    try testing.expectEqual(load_stats.lends, mlx.engine.stats().lends);

    // 1 x 1 scales up to the 546 px square (1,521 patches); 5000 x 50 is the thin canvas; 2000 x 1500 a large one.
    for ([_][2]u32{ .{ 1, 1 }, .{ 5000, 50 }, .{ 2000, 1500 } }) |size| {
        const rgb = try testing.allocator.alloc(u8, @as(usize, size[0]) * size[1] * 3);
        defer testing.allocator.free(rgb);
        var prng = std.Random.DefaultPrng.init(size[0]);
        prng.random().bytes(rgb);
        const px = try preprocess(testing.allocator, rgb, size[1], size[0], Processor.of(v));
        defer testing.allocator.free(px.bytes);
        const n: usize = @as(usize, px.grid_h) * px.grid_w;
        const patches = mlx.mlx_array_new_data(px.bytes.ptr, &[_]c_int{ ci(n), 588 }, 2, .float32);
        defer _ = mlx.mlx_array_free(patches);
        const want = try on_mlx.forward(patches, px.grid_h, px.grid_w);
        defer _ = mlx.mlx_array_free(want);

        const ops_before = mlx.engine.mlx_ops;
        const lends_before = mlx.engine.stats().lends;
        const made_before = mlx.engine.pipelinesMade().count;
        const calls_before = mlx.calls.load(.monotonic);
        var named_before: std.ArrayList(mlx.NameCount) = .empty;
        defer named_before.deinit(testing.allocator);
        try mlx.callsByName(&named_before, testing.allocator);
        const got = try on_engine.forward(patches, px.grid_h, px.grid_w);
        defer _ = mlx.mlx_array_free(got);
        try testing.expectEqual(ops_before, mlx.engine.mlx_ops);
        try testing.expectEqual(lends_before, mlx.engine.stats().lends);
        // One array is MLX's, this test's patches: a pass reads it 12 times (availability, bytes, buffer, shape,
        // strides, dtype).
        try testing.expect(mlx.calls.load(.monotonic) - calls_before < 20);
        try expectOnlyImportReads(named_before.items);
        // The warm covers the pipelines an image of any size reaches.
        try testing.expectEqual(made_before, mlx.engine.pipelinesMade().count);
        try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
        const bytes = mlx.mlx_array_size(want) * mlx.mlx_array_itemsize(want);
        try testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(want).?[0..bytes], mlx.mlx_array_data_uint8(got).?[0..bytes]);
    }
}

test "dsv41 vision: the tower's pipelines ride the manifest: a first load warms and stores them, a same-key load warms nothing" {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;
    defer mlx.engine.seal(false);
    defer dsv41_pipelines.forget();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    const key: manifest.Key = .{ .scope = "70e570e570e570e5".*, .build = "b2b2b2b2b2b2b2b2".* };
    var w1 = try towerOf(engine_vision, tiny_dim);
    defer w1.deinit();
    var w2 = try towerOf(engine_vision, tiny_dim);
    defer w2.deinit();
    const pipelines_registry = mlx.engine.pipeline_registry;

    // First load: no manifest, the tower warms and the manifest is written with the warm's marker.
    pipelines_registry.resetForTest();
    var first = dsv41_pipelines.begin(testing.allocator, io, tmp.dir, "", key, 0);
    defer first.deinit();
    try testing.expectEqual(dsv41_pipelines.Case.none, first.case);
    _ = first.settle();
    var one = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &w1);
    defer one.deinit();
    const warms = Dsv41Vision.warms;
    try one.useEngine();
    defer mlx.engine.budget(1 << 62) catch {};
    try testing.expectEqual(warms + 1, Dsv41Vision.warms);
    var stored = try manifest.lookup(testing.allocator, io, tmp.dir, key);
    defer stored.deinit();
    try testing.expect(stored == .same);
    var markers: usize = 0;
    for (stored.same.recipes) |r| markers += @intFromBool(r.kind == .marker and std.mem.startsWith(u8, r.key, "q8/"));
    try testing.expectEqual(@as(usize, 1), markers);

    // Same-key load, as a new process: the manifest's recipes make, the marker is trusted, and the tower runs no warm.
    pipelines_registry.resetForTest();
    var second = dsv41_pipelines.begin(testing.allocator, io, tmp.dir, "", key, 0);
    defer second.deinit();
    try testing.expectEqual(dsv41_pipelines.Case.same, second.case);
    try testing.expectEqual(dsv41_model.WarmScope.mlx, second.settle());
    var two = try Dsv41Vision.init(testing.allocator, configOf(engine_vision, tiny_dim), &w2);
    defer two.deinit();
    const late = mlx.engine.latePipelines();
    const made = mlx.engine.pipelinesMade().count;
    try two.useEngine();
    try testing.expectEqual(warms + 1, Dsv41Vision.warms);
    try testing.expectEqual(late, mlx.engine.latePipelines());
    try testing.expectEqual(made, mlx.engine.pipelinesMade().count);

    // A manifest whose recipes did not all make is not trusted: its marker is dropped and the tower warms.
    pipelines_registry.resetForTest();
    pipelines_registry.putMarker(.{ .kind = .marker, .function = "vision tower", .key = "q8/stale" }) catch unreachable;
    pipelines_registry.settleMarkers(false);
    try testing.expect(!pipelines_registry.takeMarker("q8/stale"));
}

test "dsv41 vision: a stock kernel with an int function constant is made from its recipe, and a launch finds it" {
    try mlx.requireGpu();
    const stock = mlx.engine.stock_kernels;
    const cache = mlx.engine.pipeline_cache;
    const name = "sdpa_vector_2pass_1_float16_t_64_64";
    const flags = [_]metal.Flag{ .{ .index = 20, .value = false }, .{ .index = 21, .value = false }, .{ .index = 22, .value = false }, .{ .index = 23, .value = false }, .{ .index = 24, .value = false }, .{ .index = 25, .value = false } };
    const ints = [_]metal.IntConstant{.{ .index = 26, .value = 160 }};
    _ = try stock.kernelConstants(name, "int_recipe_probe_made", &flags, &ints);
    var recipe: ?manifest.Recipe = null;
    for (mlx.engine.pipeline_registry.recipes()) |r| {
        if (std.mem.eql(u8, r.key, "int_recipe_probe_made")) recipe = r;
    }
    var made = recipe orelse return error.RecipeNotRecorded;
    try testing.expectEqual(@as(i32, 160), made.ints[0].value);
    // Another process names the same pipeline by another key; the file carries the constants.
    made.key = "int_recipe_probe_waiting";
    const bytes = try manifest.encode(testing.allocator, "k", &.{made});
    defer testing.allocator.free(bytes);
    var loaded = try manifest.decode(testing.allocator, bytes);
    defer loaded.deinit();
    const report = cache.finish(try cache.start(loaded.recipes, 1));
    try testing.expectEqual(@as(usize, 1), report.made);
    const before = mlx.engine.pipelinesMade().count;
    const preloaded = mlx.engine.preloadedPipelines();
    _ = try stock.kernelConstants(name, "int_recipe_probe_waiting", &flags, &ints);
    try testing.expectEqual(before, mlx.engine.pipelinesMade().count);
    try testing.expectEqual(preloaded + 1, mlx.engine.preloadedPipelines());
}

// ── The requantized tower against the bf16 tower on a real image (the pack's tower, SUSHI_MODELS_DIR) ──

test "dsv41 vision: the requantized tower stays within bf16's drift of the bf16 tower on house.jpeg at 642 tokens" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tower_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tower_path = try fixtureFile(&tower_buf, "tower.safetensors");
    const house = try common.loadHouse(a, testing.io);
    defer a.free(house.rgb);
    const src_w = house.w;
    const src_h = house.h;
    const rgb = house.rgb;

    // The picture letterboxed on a 1302 x 812 canvas: a 58 x 93 ViT grid and 20 x 31 aligner cells, 642 span tokens.
    const cw: u32 = 1302;
    const chh: u32 = 812;
    const box = letterbox(src_w, src_h, cw, chh);
    const fitted = try common.resizeRgbU8(a, rgb, src_h, src_w, box.rh, box.rw, .bicubic);
    defer a.free(fitted);
    const canvas = try a.alloc(u8, @as(usize, cw) * chh * 3);
    defer a.free(canvas);
    @memset(canvas, 127);
    for (0..box.rh) |y| @memcpy(canvas[((y + box.y) * cw + box.x) * 3 ..][0 .. @as(usize, box.rw) * 3], fitted[y * box.rw * 3 ..][0 .. @as(usize, box.rw) * 3]);
    const px = try preprocess(a, canvas, chh, cw, Processor.of((try releaseConfig()).dsv41.?.vision.?));
    defer a.free(px.bytes);
    try testing.expectEqual(@as(u32, 58), px.grid_h);
    try testing.expectEqual(@as(u32, 93), px.grid_w);
    const patches = mlx.mlx_array_new_data(px.bytes.ptr, &[_]c_int{ ci(px.grid_h * px.grid_w), 588 }, 2, .float32);
    defer _ = mlx.mlx_array_free(patches);

    var wb = try loadFile(tower_path.ptr);
    defer wb.deinit();
    var wq = try loadFile(tower_path.ptr);
    defer wq.deinit();
    var active: [2]usize = undefined;
    _ = mlx.mlx_get_active_memory(&active[0]);
    var load = io_util.Stopwatch.init(testing.io);
    var dq = try Dsv41Vision.init(a, try releaseConfig(), &wq);
    const load_ms = @as(f64, @floatFromInt(load.read())) / 1e6;
    defer dq.deinit();
    _ = mlx.mlx_get_active_memory(&active[1]);
    var db = try Dsv41Vision.initWith(a, try releaseConfig(), &wb, false);
    defer db.deinit();

    const stages = [_]struct { name: []const u8, blocks: usize }{ .{ .name = "after 16 blocks", .blocks = 16 }, .{ .name = "after 31 blocks", .blocks = 31 } };
    const report = std.c.getenv("SUSHI_VISION_Q8_REPORT") != null;
    var worst_cos: f64 = 1;
    for (stages) |st| {
        const got = try dq.towerAfter(patches, px.grid_h, px.grid_w, st.blocks);
        defer _ = mlx.mlx_array_free(got);
        const want = try db.towerAfter(patches, px.grid_h, px.grid_w, st.blocks);
        defer _ = mlx.mlx_array_free(want);
        const g = try common.gap(got, want, dq.s);
        worst_cos = @min(worst_cos, g.cos);
        if (report) std.debug.print("[vision-q8] {s}: cos {d:.6} max|d| {d:.4} rel-rms {d:.5}\n", .{ st.name, g.cos, g.max_abs, g.rel_rms });
    }
    const tower_q = try dq.towerHidden(patches, px.grid_h, px.grid_w);
    defer _ = mlx.mlx_array_free(tower_q);
    const tower_b = try db.towerHidden(patches, px.grid_h, px.grid_w);
    defer _ = mlx.mlx_array_free(tower_b);
    const gt = try common.gap(tower_q, tower_b, dq.s);
    const span_q = try dq.forward(patches, px.grid_h, px.grid_w);
    defer _ = mlx.mlx_array_free(span_q);
    const span_b = try db.forward(patches, px.grid_h, px.grid_w);
    defer _ = mlx.mlx_array_free(span_b);
    const gs = try common.gap(span_q, span_b, dq.s);
    if (report) {
        std.debug.print("[vision-q8] final norm: cos {d:.6} max|d| {d:.4} rel-rms {d:.5}\n", .{ gt.cos, gt.max_abs, gt.rel_rms });
        std.debug.print("[vision-q8] span (aligner out): cos {d:.6} max|d| {d:.4} rel-rms {d:.5}\n", .{ gs.cos, gs.max_abs, gs.rel_rms });
        std.debug.print("[vision-q8] init {d:.0} ms; active {d:.3} GB -> {d:.3} GB\n", .{ load_ms, @as(f64, @floatFromInt(active[0])) / 1e9, @as(f64, @floatFromInt(active[1])) / 1e9 });
        var probe: [2]usize = undefined;
        _ = mlx.mlx_get_active_memory(&probe[0]);
        var one = try dflash.quantizeDense(wb.get("aligner.w1.weight").?, common.QUANT_BITS, QUANT_GROUP, dq.s);
        for ([_]A{ one.w, one.scales, one.biases }) |x| try mlx.check(mlx.mlx_array_eval(x));
        _ = mlx.mlx_get_active_memory(&probe[1]);
        one.deinit();
        std.debug.print("[vision-q8] aligner.w1 quantized: active +{d} B, billed {d} B\n", .{ probe[1] - probe[0], quantizedBytes(5120, 9216) });
        for ([_]*Dsv41Vision{ &db, &dq }, [_][]const u8{ "bf16", "q8" }) |d, name| {
            mlx.resetTransientPeak();
            const o = try d.forward(patches, px.grid_h, px.grid_w);
            _ = mlx.mlx_array_free(o);
            std.debug.print("[vision-q8] {s} encode transient peak {d:.3} GiB\n", .{ name, @as(f64, @floatFromInt(mlx.transientPeak())) / (1 << 30) });
        }
        // Interleaved in this process, warm: the tower through its final norm, then with the aligner.
        const reps = 9;
        var t: [4][reps]f64 = undefined;
        for (0..reps + 1) |i| {
            for (0..2) |arm| {
                const d: *Dsv41Vision = if (arm == 0) &db else &dq;
                var sw = io_util.Stopwatch.init(testing.io);
                const o = try d.towerHidden(patches, px.grid_h, px.grid_w);
                _ = mlx.mlx_array_free(o);
                const tower_ms = @as(f64, @floatFromInt(sw.read())) / 1e6;
                sw.reset();
                const o2 = try d.forward(patches, px.grid_h, px.grid_w);
                _ = mlx.mlx_array_free(o2);
                const full_ms = @as(f64, @floatFromInt(sw.read())) / 1e6;
                if (i > 0) {
                    t[arm * 2][i - 1] = tower_ms;
                    t[arm * 2 + 1][i - 1] = full_ms;
                }
            }
        }
        std.debug.print("[vision-q8] 642 tokens, median of {d}: bf16 tower {d:.1} ms, +aligner {d:.1} ms; q8 tower {d:.1} ms, +aligner {d:.1} ms\n", .{ reps, common.median(&t[0]), common.median(&t[1]), common.median(&t[2]), common.median(&t[3]) });
    }
    try testing.expect(worst_cos > 0.998);
    try testing.expect(gt.cos > 0.998);
    try testing.expect(gs.cos > 0.998);
}

const request_media = @import("../request_media.zig");
const arch_media = @import("../arch/media.zig");
const chat_mod = @import("../chat.zig");

test "a tower config without a vision block reserves the pixel-cap image, not a token budget it does not have" {
    var cfg = arch_media.testConfig("deepseek_v41") orelse return error.SkipZigTest;
    cfg.dsv41.?.vision = null;
    const vp = request_media.visionPreprocFromConfig(&cfg);
    try std.testing.expectEqual(.gemma, vp.mode);
    const patches: u64 = common.effectivePixelBounds(vp.min_pixels, vp.max_pixels).max / (@as(u64, vp.patch) * vp.patch);
    const img = [_]chat_mod.ImageData{.{ .pixels = &.{}, .width = 0, .height = 0, .grid_h = 1, .grid_w = @intCast(patches) }};
    const want = request_media.visionEncodeBill(&cfg, &img, &.{}, @intCast(patches / (vp.merge * vp.merge))).bytes;
    try std.testing.expectEqual(want, request_media.largestImageEncodeBytes(&cfg));
}

/// The calls into mlx-c since `named_before` are the reads of imported weights and nothing else: no op, no eval.
fn expectOnlyImportReads(named_before: []const mlx.NameCount) !void {
    var now: std.ArrayList(mlx.NameCount) = .empty;
    defer now.deinit(testing.allocator);
    try mlx.callsByName(&now, testing.allocator);
    const allowed = [_][]const u8{ "mlx_array_free", "mlx_array_ndim", "mlx_array_shape", "mlx_array_dtype", "_mlx_array_is_available", "mlx_array_data_uint8", "_mlx_array_metal_buffer", "mlx_array_strides", "mlx_array_set" };
    for (now.items) |n| {
        var was: u64 = 0;
        for (named_before) |b| if (std.mem.eql(u8, b.name, n.name)) {
            was = b.count;
        };
        if (n.count == was) continue;
        for (allowed) |name| {
            if (std.mem.eql(u8, name, n.name)) break;
        } else return error.AnMlxCallThatIsNotAnImportRead;
    }
}
