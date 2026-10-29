//! GLM-5.3-Flash vision tower and padded CLIP image processor.
//! Matches HF transformers/models/glm5_next/{modeling,image_processing}_glm5_next.py.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const base = @import("glm5_model.zig");
const common = @import("vision_common.zig");
const Arr = mlx.mlx_array;
const Ops = base.Ops;
pub const Resized = common.Resized;
pub const QUERY_CHUNK: u64 = 256;
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

fn alignPixels(value: u32, factor: u32) u32 {
    return ((value - 1) / factor + 1) * factor;
}

/// HF's aligned canvas; budgets count merged spatiotemporal tokens.
pub fn smartResize(frames: u32, height: u32, width: u32, temporal: u32, factor: u32, min_tokens: u32, max_tokens: u32) Resized {
    std.debug.assert(frames > 0 and height > 0 and width > 0 and temporal > 0 and factor > 0 and max_tokens > 0);
    const aligned_frames: u64 = @intFromFloat(@max(@as(f64, @floatFromInt(temporal)), common.roundHalfEven(@as(f64, @floatFromInt(frames)) / @as(f64, @floatFromInt(temporal))) * @as(f64, @floatFromInt(temporal))));
    const pixels_per_token: u64 = @as(u64, temporal) * factor * factor;
    const min_pixels = @as(u64, min_tokens) * pixels_per_token;
    const max_pixels = @as(u64, max_tokens) * pixels_per_token;
    var h = alignPixels(height, factor);
    var w = alignPixels(width, factor);
    if (aligned_frames * h * w < min_pixels) {
        const scale = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (@as(f64, @floatFromInt(frames)) * @as(f64, @floatFromInt(height)) * @as(f64, @floatFromInt(width))));
        h = alignPixels(@intFromFloat(@max(1, @ceil(@as(f64, @floatFromInt(height)) * scale))), factor);
        w = alignPixels(@intFromFloat(@max(1, @ceil(@as(f64, @floatFromInt(width)) * scale))), factor);
    }
    if (aligned_frames * h * w > max_pixels) {
        var low: u32 = 1;
        var high = height;
        h = factor;
        w = factor;
        while (low <= high) {
            const content_h = low + (high - low) / 2;
            const content_w: u32 = @intCast(@max(1, @as(u64, width) * content_h / height));
            const candidate_h = alignPixels(content_h, factor);
            const candidate_w = alignPixels(content_w, factor);
            if (aligned_frames * candidate_h * candidate_w <= max_pixels) {
                h = candidate_h;
                w = candidate_w;
                low = content_h + 1;
            } else high = content_h - 1;
        }
    }
    return .{ .h = h, .w = w };
}

/// Aspect-preserving bicubic content resize, right/bottom black padding, CLIP normalization.
pub fn resizeNormalizedChw(a: std.mem.Allocator, dst: []f32, rgb: []const u8, sh: u32, sw: u32, dh: u32, dw: u32, frames: u32, temporal: u32, factor: u32, min_tokens: u32) !void {
    if (sh == 0 or sw == 0 or dh == 0 or dw == 0 or dst.len != @as(usize, dh) * dw * 3) return error.InvalidImageDimensions;
    var scale = @min(@as(f64, @floatFromInt(dh)) / @as(f64, @floatFromInt(sh)), @as(f64, @floatFromInt(dw)) / @as(f64, @floatFromInt(sw)));
    if (@as(u64, frames) * sh * sw >= @as(u64, temporal) * factor * factor * min_tokens) scale = @min(1, scale);
    const ch: u32 = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(dh)), @floor(@as(f64, @floatFromInt(sh)) * scale))));
    const cw: u32 = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(dw)), @floor(@as(f64, @floatFromInt(sw)) * scale))));
    const content = try a.alloc(f32, @as(usize, ch) * cw * 3);
    defer a.free(content);
    try common.resizeRgbBicubicNormalizedChw(a, content, rgb, sh, sw, ch, cw);
    const plane = @as(usize, dh) * dw;
    for (0..3) |c| {
        @memset(dst[c * plane ..][0..plane], -MEAN[c] / STD[c]);
        for (0..ch) |y| for (0..cw) |x| {
            // The reused resampler produces 2*(pixel/255)-1; undo its normalization.
            const pixel = (content[c * @as(usize, ch) * cw + y * cw + x] + 1) * 0.5;
            dst[c * plane + y * dw + x] = (pixel - MEAN[c]) / STD[c];
        };
    }
}

const Projection = struct {
    linear: base.Linear,
    bias: ?Arr = null,
    fn load(weights: *const model.Weights, prefix: []const u8, input: u32, output: u32, with_bias: bool) !Projection {
        const linear = try base.Linear.load(weights, prefix, input);
        if (linear.output != output) return error.InvalidGlmVisionWeights;
        var buf: [256]u8 = undefined;
        const bias = weights.get(try std.fmt.bufPrint(&buf, "{s}.bias", .{prefix}));
        if (with_bias and bias == null) return error.MissingVisionWeights;
        if (bias) |b| if (!std.mem.eql(c_int, mlx.getShape(b), &.{@intCast(output)})) return error.InvalidGlmVisionWeights;
        return .{ .linear = linear, .bias = bias };
    }
    fn apply(self: Projection, ops: *Ops, x: Arr) !Arr {
        const out = try self.linear.apply(ops, x);
        return if (self.bias) |b| ops.binary(.add, out, b) else out;
    }
};
const Mlp = struct {
    gate: Projection,
    up: Projection,
    down: Projection,
    fn load(weights: *const model.Weights, prefix: []const u8, input: u32, middle: u32, with_bias: bool) !Mlp {
        var b: [256]u8 = undefined;
        return .{
            .gate = try Projection.load(weights, try std.fmt.bufPrint(&b, "{s}.gate_proj", .{prefix}), input, middle, with_bias),
            .up = try Projection.load(weights, try std.fmt.bufPrint(&b, "{s}.up_proj", .{prefix}), input, middle, with_bias),
            .down = try Projection.load(weights, try std.fmt.bufPrint(&b, "{s}.down_proj", .{prefix}), middle, input, with_bias),
        };
    }
    fn apply(self: Mlp, ops: *Ops, x: Arr, limit: f32) !Arr {
        const gate = try self.gate.apply(ops, x);
        const up = try self.up.apply(ops, x);
        const hi = try ops.scalar(limit, mlx.mlx_array_dtype(gate));
        const lo = try ops.scalar(-limit, mlx.mlx_array_dtype(up));
        return self.down.apply(ops, try ops.binary(.mul, try ops.silu(try ops.binary(.min, gate, hi)), try ops.binary(.max, try ops.binary(.min, up, hi), lo)));
    }
};
const Block = struct { norm1: Arr, norm2: Arr, q_norm: Arr, k_norm: Arr, qkv: Projection, proj: Projection, mlp: Mlp };
fn tensor(weights: *const model.Weights, prefix: []const u8, leaf: []const u8) !Arr {
    var b: [256]u8 = undefined;
    return weights.get(try std.fmt.bufPrint(&b, "{s}.{s}", .{ prefix, leaf })) orelse error.MissingVisionWeights;
}

pub const GlmVision = struct {
    allocator: std.mem.Allocator,
    config: model.ModelConfig,
    s: mlx.mlx_stream,
    blocks: []Block,
    patch: Projection,
    downsample: Projection,
    post_norm: Arr,
    merger_proj: Projection,
    merger_norm_w: Arr,
    merger_norm_b: Arr,
    merger_mlp: Mlp,

    pub fn init(a: std.mem.Allocator, cfg: model.ModelConfig, weights: *const model.Weights) !GlmVision {
        const s = mlx.mlx_default_gpu_stream_new();
        errdefer _ = mlx.mlx_stream_free(s);
        const blocks = try a.alloc(Block, cfg.qv_depth);
        errdefer a.free(blocks);
        const prefix = "model.visual";
        var b: [256]u8 = undefined;
        for (blocks, 0..) |*block, i| {
            const p = try std.fmt.bufPrint(&b, "{s}.blocks.{d}", .{ prefix, i });
            var l: [256]u8 = undefined;
            block.* = .{
                .norm1 = try tensor(weights, p, "norm1.weight"),
                .norm2 = try tensor(weights, p, "norm2.weight"),
                .q_norm = try tensor(weights, p, "attn.q_norm.weight"),
                .k_norm = try tensor(weights, p, "attn.k_norm.weight"),
                .qkv = try Projection.load(weights, try std.fmt.bufPrint(&l, "{s}.attn.qkv", .{p}), cfg.qv_hidden, 3 * cfg.qv_hidden, true),
                .proj = try Projection.load(weights, try std.fmt.bufPrint(&l, "{s}.attn.proj", .{p}), cfg.qv_hidden, cfg.qv_hidden, true),
                .mlp = try Mlp.load(weights, try std.fmt.bufPrint(&l, "{s}.mlp", .{p}), cfg.qv_hidden, cfg.qv_intermediate, true),
            };
        }
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const patch_w = try tensor(weights, prefix, "patch_embed.proj.weight");
        const down_w = try tensor(weights, prefix, "downsample.weight");
        if (!std.mem.eql(c_int, mlx.getShape(patch_w), &.{ @intCast(cfg.qv_hidden), 3, @intCast(cfg.qv_temporal_patch), @intCast(cfg.qv_patch), @intCast(cfg.qv_patch) }) or
            !std.mem.eql(c_int, mlx.getShape(down_w), &.{ @intCast(cfg.qv_out_hidden), @intCast(cfg.qv_hidden), @intCast(cfg.qv_merge), @intCast(cfg.qv_merge) })) return error.InvalidGlmVisionWeights;
        const feature = 3 * cfg.qv_temporal_patch * cfg.qv_patch * cfg.qv_patch;
        const flat_patch = try ops.result(try ops.reshape(patch_w, &.{ @intCast(cfg.qv_hidden), @intCast(feature) }));
        errdefer _ = mlx.mlx_array_free(flat_patch);
        const flat_down = try ops.result(try ops.reshape(down_w, &.{ @intCast(cfg.qv_out_hidden), @intCast(cfg.qv_hidden * cfg.qv_merge * cfg.qv_merge) }));
        errdefer _ = mlx.mlx_array_free(flat_down);
        return .{
            .allocator = a,
            .config = cfg,
            .s = s,
            .blocks = blocks,
            .patch = .{ .linear = .{ .w = flat_patch, .input = @intCast(feature), .output = @intCast(cfg.qv_hidden) }, .bias = try tensor(weights, prefix, "patch_embed.proj.bias") },
            .downsample = .{ .linear = .{ .w = flat_down, .input = @intCast(cfg.qv_hidden * cfg.qv_merge * cfg.qv_merge), .output = @intCast(cfg.qv_out_hidden) }, .bias = try tensor(weights, prefix, "downsample.bias") },
            .post_norm = try tensor(weights, prefix, "post_layernorm.weight"),
            .merger_proj = try Projection.load(weights, "model.visual.merger.proj", cfg.qv_out_hidden, cfg.qv_out_hidden, false),
            .merger_norm_w = try tensor(weights, prefix, "merger.post_projection_norm.weight"),
            .merger_norm_b = try tensor(weights, prefix, "merger.post_projection_norm.bias"),
            .merger_mlp = try Mlp.load(weights, "model.visual.merger", cfg.qv_out_hidden, cfg.glmv_projection_intermediate, false),
        };
    }
    pub fn deinit(self: *GlmVision) void {
        _ = mlx.mlx_array_free(self.patch.linear.w);
        _ = mlx.mlx_array_free(self.downsample.linear.w);
        self.allocator.free(self.blocks);
        _ = mlx.mlx_stream_free(self.s);
    }

    const Rope = struct {
        cos: Arr,
        sin: Arr,
        fn deinit(r: Rope) void {
            _ = mlx.mlx_array_free(r.cos);
            _ = mlx.mlx_array_free(r.sin);
        }
    };
    fn rope(self: *GlmVision, gh: u32, gw: u32) !Rope {
        const cfg = self.config;
        const n: usize = @as(usize, gh) * gw;
        const d: usize = cfg.qv_head_dim;
        const cos = try self.allocator.alloc(f32, n * d);
        defer self.allocator.free(cos);
        const sin = try self.allocator.alloc(f32, n * d);
        defer self.allocator.free(sin);
        var row: usize = 0;
        for (0..gh / cfg.qv_merge) |by| for (0..gw / cfg.qv_merge) |bx| for (0..cfg.qv_merge) |dy| for (0..cfg.qv_merge) |dx| {
            for (0..d / 4) |j| {
                const freq = std.math.pow(f32, cfg.glmv_rope_theta, -@as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(d / 2)));
                for (0..2) |axis| {
                    const pos = if (axis == 0) by * cfg.qv_merge + dy else bx * cfg.qv_merge + dx;
                    const angle = @as(f32, @floatFromInt(pos)) * freq;
                    const off = axis * (d / 4) + j;
                    cos[row * d + off] = @cos(angle);
                    cos[row * d + off + d / 2] = @cos(angle);
                    sin[row * d + off] = @sin(angle);
                    sin[row * d + off + d / 2] = @sin(angle);
                }
            }
            row += 1;
        };
        return .{
            .cos = mlx.mlx_array_new_data(cos.ptr, &[_]c_int{ @intCast(n), 1, @intCast(d) }, 3, .float32),
            .sin = mlx.mlx_array_new_data(sin.ptr, &[_]c_int{ @intCast(n), 1, @intCast(d) }, 3, .float32),
        };
    }
    fn rotate(_: *GlmVision, ops: *Ops, x: Arr, r: Rope, d: c_int) !Arr {
        const f = try ops.cast(x, .float32);
        const halves = [_]Arr{ try ops.unary(.negative, try ops.slice(f, 2, @divExact(d, 2), d)), try ops.slice(f, 2, 0, @divExact(d, 2)) };
        return ops.cast(try ops.binary(.add, try ops.binary(.mul, f, r.cos), try ops.binary(.mul, try ops.concat(&halves, 2), r.sin)), mlx.mlx_array_dtype(x));
    }
    fn attention(self: *GlmVision, ops: *Ops, x: Arr, block: Block, r: Rope, n: c_int) !Arr {
        const h: c_int = @intCast(self.config.qv_heads);
        const d: c_int = @intCast(self.config.qv_head_dim);
        const qkv = try ops.reshape(try block.qkv.apply(ops, x), &.{ n, 3, h, d });
        const q0 = try ops.rms(try ops.reshape(try ops.slice(qkv, 1, 0, 1), &.{ n, h, d }), block.q_norm, self.config.glmv_eps);
        const k0 = try ops.rms(try ops.reshape(try ops.slice(qkv, 1, 1, 2), &.{ n, h, d }), block.k_norm, self.config.glmv_eps);
        const q = try ops.reshape(try ops.transpose(try self.rotate(ops, q0, r, d), &.{ 1, 0, 2 }), &.{ 1, h, n, d });
        const k = try ops.reshape(try ops.transpose(try self.rotate(ops, k0, r, d), &.{ 1, 0, 2 }), &.{ 1, h, n, d });
        const v = try ops.transpose(try ops.reshape(try ops.slice(qkv, 1, 2, 3), &.{ 1, n, h, d }), &.{ 0, 2, 1, 3 });
        // Bound any fallback score sheet. Each query chunk sees every spatial key.
        var pieces = std.ArrayList(Arr).empty;
        defer {
            for (pieces.items) |piece| _ = mlx.mlx_array_free(piece);
            pieces.deinit(self.allocator);
        }
        var start: c_int = 0;
        while (start < n) : (start += @intCast(QUERY_CHUNK)) {
            var piece_ops = Ops{ .s = self.s };
            defer piece_ops.deinit();
            const queries = try piece_ops.slice(q, 2, start, @min(n, start + @as(c_int, @intCast(QUERY_CHUNK))));
            const out = try piece_ops.slot();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(out, queries, k, v, 1 / @sqrt(@as(f32, @floatFromInt(d))), "", .{ .ctx = null }, .{ .ctx = null }, false, self.s));
            try mlx.check(mlx.mlx_array_eval(out.*));
            const piece = try piece_ops.result(out.*);
            pieces.append(self.allocator, piece) catch |err| {
                _ = mlx.mlx_array_free(piece);
                return err;
            };
        }
        const joined = try ops.concat(pieces.items, 2);
        return block.proj.apply(ops, try ops.reshape(try ops.transpose(joined, &.{ 0, 2, 1, 3 }), &.{ n, h * d }));
    }
    pub fn forward(self: *GlmVision, patches: Arr, gh: u32, gw: u32) !Arr {
        const cfg = self.config;
        const shape = mlx.getShape(patches);
        const n: c_int = @intCast(@as(u64, gh) * gw);
        if (n == 0 or gh % cfg.qv_merge != 0 or gw % cfg.qv_merge != 0 or shape.len != 2 or shape[0] != n or shape[1] != self.patch.linear.input) return error.InvalidPatchGrid;
        const r = try self.rope(gh, gw);
        defer r.deinit();
        var x: Arr = undefined;
        {
            var ops = Ops{ .s = self.s };
            defer ops.deinit();
            x = try ops.result(try self.patch.apply(&ops, try ops.cast(patches, mlx.mlx_array_dtype(self.patch.linear.w))));
        }
        defer _ = mlx.mlx_array_free(x);
        for (self.blocks) |block| {
            var ops = Ops{ .s = self.s };
            defer ops.deinit();
            const hidden = try ops.binary(.add, x, try self.attention(&ops, try ops.rms(x, block.norm1, cfg.glmv_eps), block, r, n));
            const result = try ops.binary(.add, hidden, try block.mlp.apply(&ops, try ops.rms(hidden, block.norm2, cfg.glmv_eps), cfg.glmv_swiglu_limit));
            try mlx.check(mlx.mlx_array_eval(result));
            _ = mlx.mlx_array_free(x);
            x = try ops.result(result);
        }
        var ops = Ops{ .s = self.s };
        defer ops.deinit();
        const merged = @divExact(n, @as(c_int, @intCast(cfg.qv_merge * cfg.qv_merge)));
        const normalized = try ops.rms(x, self.post_norm, cfg.glmv_eps);
        const grouped = try ops.reshape(normalized, &.{ merged, @intCast(cfg.qv_merge), @intCast(cfg.qv_merge), @intCast(cfg.qv_hidden) });
        const down = try self.downsample.apply(&ops, try ops.reshape(try ops.transpose(grouped, &.{ 0, 3, 1, 2 }), &.{ merged, @intCast(cfg.qv_hidden * cfg.qv_merge * cfg.qv_merge) }));
        const proj = try self.merger_proj.apply(&ops, down);
        const norm = try ops.layerNorm(proj, self.merger_norm_w, self.merger_norm_b, 1e-5);
        const f = try ops.cast(norm, .float32);
        const erf = try ops.slot();
        try mlx.check(mlx.mlx_erf(erf, try ops.binary(.mul, f, try ops.scalar(1 / @sqrt(@as(f32, 2)), .float32)), self.s));
        const gelu = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, f, try ops.scalar(0.5, .float32)), try ops.binary(.add, erf.*, try ops.scalar(1, .float32))), mlx.mlx_array_dtype(norm));
        const out = try ops.reshape(try self.merger_mlp.apply(&ops, gelu, cfg.glmv_swiglu_limit), &.{ 1, merged, @intCast(cfg.qv_out_hidden) });
        try mlx.check(mlx.mlx_array_eval(out));
        return ops.result(out);
    }
    pub fn forwardVideo(self: *GlmVision, patches: Arr, gt: u32, gh: u32, gw: u32) !Arr {
        if (gt == 0) return error.InvalidPatchGrid;
        const n: c_int = @intCast(@as(u64, gh) * gw);
        const shape = mlx.getShape(patches);
        if (shape.len != 2 or shape[0] != @as(u64, gt) * @as(u64, @intCast(n))) return error.InvalidPatchGrid;
        var ops = Ops{ .s = self.s };
        defer ops.deinit();
        const parts = try self.allocator.alloc(Arr, gt);
        var built: usize = 0;
        defer {
            for (parts[0..built]) |part| _ = mlx.mlx_array_free(part);
            self.allocator.free(parts);
        }
        for (parts, 0..) |*part, i| {
            var slice_ops = Ops{ .s = self.s };
            defer slice_ops.deinit();
            part.* = try self.forward(try slice_ops.slice(patches, 0, @as(c_int, @intCast(i)) * n, @as(c_int, @intCast(i + 1)) * n), gh, gw);
            built += 1;
        }
        return ops.result(try ops.concat(parts, 1));
    }
};

/// Evaluated per block and per query chunk, matching the bounded execution above.
pub fn encodeScratchBytes(cfg: *const model.ModelConfig, patches: u64) u64 {
    const scores = 4 * @as(u64, cfg.qv_heads) * @min(patches, QUERY_CHUNK) * patches;
    const activations = 4 * patches * (12 * @as(u64, cfg.qv_hidden) + 6 * @as(u64, cfg.qv_intermediate));
    const merged = patches / @max(@as(u64, cfg.qv_merge) * cfg.qv_merge, 1);
    const merger = 4 * merged * (8 * @as(u64, cfg.qv_out_hidden) + 6 * @as(u64, cfg.glmv_projection_intermediate));
    return (64 << 20) + scores * 13 / 10 + @max(activations, merger);
}

fn tinyConfig() model.ModelConfig {
    return .{ .glm5_vision = true, .has_vision = true, .hidden_size = 64, .qv_depth = 2, .qv_hidden = 32, .qv_heads = 4, .qv_head_dim = 8, .qv_intermediate = 48, .qv_patch = 4, .qv_temporal_patch = 2, .qv_merge = 2, .qv_out_hidden = 64, .glmv_projection_intermediate = 80, .glmv_swiglu_limit = 0.7 };
}
fn tinyWeights(tmp: *std.testing.TmpDir) !model.Weights {
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "tiny.safetensors", .data = @embedFile("fixtures/glm5_vision_tiny.safetensors") });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(std.testing.io, &buf);
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, "{s}/tiny.safetensors", .{buf[0..len]}, 0);
    defer std.testing.allocator.free(path);
    var weights = model.Weights.init(std.testing.allocator);
    errdefer weights.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try model.loadSafetensorsFile(std.testing.allocator, &weights, path, cpu, true);
    return weights;
}
fn assertReference(actual: Arr, expected: Arr, stream: mlx.mlx_stream) !void {
    const cos = try common.cosineSim(actual, expected, stream);
    const ratio = try common.rmsRatio(actual, expected, stream);
    errdefer std.debug.print("[glm-vision parity] cosine={d:.7} rms_ratio={d:.7}\n", .{ cos, ratio });
    try std.testing.expect(cos > 0.99995);
    try std.testing.expect(ratio > 0.998 and ratio < 1.002);
}

test "GLM vision tiny matches original HF tower across query chunks and video groups" {
    try mlx.requireGpu();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var weights = try tinyWeights(&tmp);
    defer weights.deinit();
    var tower = try GlmVision.init(std.testing.allocator, tinyConfig(), &weights);
    defer tower.deinit();
    const out = try tower.forward(weights.get("fixture.pixel_values").?, 16, 20);
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expectEqualSlices(c_int, &.{ 1, 80, 64 }, mlx.getShape(out));
    const expected = weights.get("fixture.features").?;
    try assertReference(out, expected, tower.s);
    const video = try tower.forwardVideo(weights.get("fixture.video_pixel_values").?, 2, 6, 8);
    defer _ = mlx.mlx_array_free(video);
    try std.testing.expectEqualSlices(c_int, &.{ 1, 24, 64 }, mlx.getShape(video));
    try assertReference(video, weights.get("fixture.video_features").?, tower.s);
}

test "GLM vision processor matches original HF padded canvas CLIP pixels and patch order" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var weights = try tinyWeights(&tmp);
    defer weights.deinit();
    const rgb = weights.get("fixture.rgb").?;
    try mlx.check(mlx.mlx_array_eval(rgb));
    const rs = smartResize(2, 55, 81, 2, 8, 16, 32);
    try std.testing.expectEqual(@as(u32, 40), rs.h);
    try std.testing.expectEqual(@as(u32, 48), rs.w);
    const a = std.testing.allocator;
    const chw = try a.alloc(f32, @as(usize, rs.h) * rs.w * 3);
    defer a.free(chw);
    try resizeNormalizedChw(a, chw, mlx.mlx_array_data_uint8(rgb).?[0 .. 55 * 81 * 3], 55, 81, rs.h, rs.w, 2, 2, 8, 16);
    const patches = try a.alloc(f32, 10 * 12 * 3 * 2 * 4 * 4);
    defer a.free(patches);
    common.buildPixelValues(patches, chw, 3, rs.h, rs.w, 4, 2, 2);
    const expected = weights.get("fixture.processed_pixels").?;
    try mlx.check(mlx.mlx_array_eval(expected));
    const want = mlx.mlx_array_data_float32(expected).?[0..patches.len];
    var max_diff: f32 = 0;
    for (want, patches) |v, p| max_diff = @max(max_diff, @abs(v - p));
    errdefer std.debug.print("[glm-vision preprocessing] max_abs={d:.7}\n", .{max_diff});
    // Torchvision uint8 bicubic and Pillow's uint8 kernel differ by at most one pixel.
    try std.testing.expect(max_diff < 0.016);
    const pad_index: usize = 39 * rs.w + 47;
    for (0..3) |c| try std.testing.expectApproxEqAbs(-MEAN[c] / STD[c], chw[c * @as(usize, rs.h) * rs.w + pad_index], 1e-6);
}

test "GLM vision resize respects token floor canvas padding and large-image budget" {
    try std.testing.expectEqual(Resized{ .h = 476, .w = 672 }, smartResize(2, 450, 650, 2, 28, 16, 8000));
    const small = smartResize(2, 7, 11, 2, 28, 16, 8000);
    try std.testing.expect(small.h * small.w >= 16 * 28 * 28);
    const large = smartResize(2, 10000, 12000, 2, 28, 16, 8000);
    try std.testing.expect(large.h * large.w <= 8000 * 28 * 28);
    const video = smartResize(32, 2160, 3840, 2, 28, 16, 240000);
    try std.testing.expect(@as(u64, 16) * video.h * video.w <= 240000 * 28 * 28);
    const cfg = model.ModelConfig{ .qv_heads = 16, .qv_hidden = 1024, .qv_intermediate = 4096, .qv_out_hidden = 4096, .qv_merge = 2 };
    const bill = encodeScratchBytes(&cfg, 32000);
    try std.testing.expect(bill < 8_000_000_000);
    // Doubling a large grid doubles the bill; query chunks keep it linear in patches.
    const floor: u64 = 64 << 20;
    try std.testing.expectEqual(2 * (bill - floor), encodeScratchBytes(&cfg, 64000) - floor);
}
