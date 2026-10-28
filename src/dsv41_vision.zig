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
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const common = @import("vision_common.zig");
const cfg_mod = @import("dsv41_config.zig");
const sc_mod = @import("dsv41_scope.zig");
const log = @import("log.zig");
const transient_peak = @import("transient_peak.zig");
const dflash = @import("dflash.zig");
const io_util = @import("io_util.zig");

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
    s: mlx.mlx_stream,
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
        return initWith(allocator, config, weights, !common.vision_bf16);
    }

    /// `quantize` false keeps every linear bf16 (the reference-parity tests).
    pub fn initWith(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights, quantize: bool) !Dsv41Vision {
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
            .s = mlx.mlx_default_gpu_stream_new(),
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
        var ld: Load = .{ .weights = weights, .s = self.s, .quantize = quantize };
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
        transient_peak.begin();
        const hidden = try self.towerHidden(patches, grid_h, grid_w);
        defer _ = mlx.mlx_array_free(hidden);
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const feats = try self.aligner(&sc, hidden, grid_h, grid_w);
        const out = sc.out(try self.span(&sc, feats, grid_h, grid_w));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        transient_peak.end(.vision_encode, grid_h * grid_w);
        return out;
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
        var x: A = blk: {
            var sc = Scope.init(self.allocator, self.s);
            defer sc.deinit();
            const e = sc.out(try self.patch.apply(&sc, try sc.astype(patches, .bfloat16)));
            errdefer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_array_eval(e));
            break :blk e;
        };
        defer _ = mlx.mlx_array_free(x);
        var rope_sc = Scope.init(self.allocator, self.s);
        defer rope_sc.deinit();
        const rope = try ropeTables(&rope_sc, v, grid_h, grid_w);
        for (self.blocks[0..n_blocks]) |*b| {
            var sc = Scope.init(self.allocator, self.s);
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
        var sc = Scope.init(self.allocator, self.s);
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
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, try sc.slice(q, 2, lo, hi), k, val, scale, "", none, none, false, self.s));
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
    const scores = 4 * @as(u64, cfg.qv_heads) * @min(patches, @as(u64, QUERY_CHUNK)) * patches;
    const activations = 4 * patches * (12 * @as(u64, cfg.qv_hidden) + 6 * @as(u64, cfg.qv_intermediate));
    const aligner = 4 * (patches / @max(@as(u64, cfg.qv_merge) * cfg.qv_merge, 1)) * (@as(u64, cfg.qv_hidden) * cfg.qv_merge * cfg.qv_merge + 4 * @as(u64, cfg.qv_out_hidden));
    return (64 << 20) + scores * 13 / 10 + @max(activations, aligner);
}

// ── Tests ──

const testing = std.testing;

const release_vision: Processor = .{ .patch = 14, .downsample = 3, .max_tokens = 1024, .min_pixels = 295936, .max_wh_ratio = 0 };

test "dsv41 vision: the resize plan and letterbox match the release's image processor" {
    // Rows of the release's `plan_image_grid` and Pillow's `ImageOps.pad` box at the release config
    // (tests/dump_dsv41_vision_fixtures.py writes them from the reference as plan.json).
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

// ── Parity against the release's reference (tests/dump_dsv41_vision_fixtures.py) ──
//
// The fixture holds the reference's bf16 processor patches and tower/aligner outputs on two images, and the
// pack's own tower tensors; both sit under `kld-teacher/dsv41-vision` in the models root. The tests skip
// when it is absent.

fn fixtureFile(buf: []u8, comptime name: [:0]const u8) ![:0]const u8 {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("test_models.zig").packPath(&dir_buf, "kld-teacher/dsv41-vision");
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
    const dir = try @import("test_models.zig").packPath(&buf, "deepseek-ai/DeepSeek-V4.1-Flash-1017");
    var tok = @import("tokenizer.zig").loadTokenizer(testing.io, testing.allocator, dir) catch return error.SkipZigTest;
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
    const v = tiny_vision;
    const d: c_int = @intCast(v.hidden);
    const inter: c_int = @intCast(v.inter);
    var w = model.Weights.init(testing.allocator);
    errdefer w.deinit();
    var buf: [96]u8 = undefined;
    var seed: u64 = 1;
    try putBf16(&w, "vision.patch_embed.proj.weight", &.{ d, 48 }, 1, 0.1, 0);
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
    const out: c_int = @intCast(tiny_dim);
    try putBf16(&w, "aligner.w1.weight", &.{ out, d * 4 }, 81, 0.1, 0);
    try putBf16(&w, "aligner.w1.bias", &.{out}, 82, 0.05, 0);
    try putBf16(&w, "aligner.w2.weight", &.{ out, out }, 83, 0.1, 0);
    try putBf16(&w, "aligner.w2.bias", &.{out}, 84, 0.05, 0);
    for ([_][]const u8{ "image_start", "image_end", "image_newline" }) |name| try putBf16(&w, name, &.{out}, 70, 0.1, 0);
    return w;
}

fn tinyConfig() model.ModelConfig {
    var cfg: model.ModelConfig = .{};
    cfg.dsv41 = .{ .dim = tiny_dim, .vision = tiny_vision };
    return cfg;
}

test "dsv41 vision: requantized tower holds affine 8-bit g128 codes, frees the bf16 copies and tracks the bf16 tower" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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

// ── The requantized tower against the bf16 tower on a real image (the pack's tower, SUSHI_MODELS_DIR) ──

test "dsv41 vision: the requantized tower stays within bf16's drift of the bf16 tower on house.jpeg at 642 tokens" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
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
