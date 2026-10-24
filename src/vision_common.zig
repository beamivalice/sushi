//! Image preprocessing shared by the vision towers: smart-resize geometry,
//! Pillow-compatible resampling and the Qwen2-VL patch layout.

const std = @import("std");
const mlx = @import("mlx.zig");

/// Qwen3VLImageProcessor fallbacks (processing_qwen3_vl.py:96-98). Checkpoint
/// processor metadata overrides these when present.
pub const FACTOR: u32 = 32; // patch_size(16) * merge_size(2)
pub const MIN_PIXELS: u32 = 56 * 56; // 3136
pub const MAX_PIXELS: u32 = 14 * 14 * 4 * 1280; // 1003520

pub const Resized = struct { h: u32, w: u32 };

/// Engine ceiling on the image area, whatever the checkpoint's processor
/// declares. Qwen3.8 packs ship `longest_edge: 16777216` (16.7 Mpx); the
/// reference survives that behind flash attention, while our ViT
/// MATERIALIZES the full bidirectional score matrix per layer — a 5100x3300
/// photo became 65k patches and an uncatchable Metal OOM (live, 26.8.9).
/// 1536x1536: ~9.2k patches, ~2.7 GB of bf16 scores per layer at 16 heads;
/// a 1920x1080 screenshot is barely touched (2.07 -> 2.36 Mpx cap).
pub const ENGINE_MAX_PIXELS: u32 = 1536 * 1536;

pub const PixelBounds = struct { min: u32, max: u32, clamped: bool };

/// The bounds the resize actually uses: the checkpoint's when sane, the
/// processor defaults otherwise, and never above ENGINE_MAX_PIXELS.
pub fn effectivePixelBounds(cfg_min: u32, cfg_max: u32) PixelBounds {
    const requested_min = if (cfg_min > 0) cfg_min else MIN_PIXELS;
    const min = @min(requested_min, ENGINE_MAX_PIXELS);
    const declared = if (cfg_max >= requested_min) cfg_max else @max(MAX_PIXELS, requested_min);
    const max = @max(min, @min(declared, ENGINE_MAX_PIXELS));
    return .{ .min = min, .max = max, .clamped = max < declared };
}

/// Python `round()` is round-half-to-EVEN (banker's rounding); Zig's
/// `std.math.round` is round-half-away-from-zero. `_smart_resize_image` uses the
/// Python builtin, so we must match it for byte-faithful grids. Inputs here are
/// always positive (image dimensions / factor).
pub fn roundHalfEven(x: f64) f64 {
    const fl = std.math.floor(x);
    const frac = x - fl;
    if (frac < 0.5) return fl;
    if (frac > 0.5) return fl + 1.0;
    // Exactly .5 → choose the even neighbour.
    return if (@mod(fl, 2.0) == 0.0) fl else fl + 1.0;
}

/// Faithful port of `_smart_resize_image` (processing_qwen3_vl.py:93-116): snap
/// each side to a multiple of `factor`, then rescale into [min_pixels, max_pixels]
/// preserving aspect ratio. Returns the resized (H, W) in pixels.
pub fn smartResizeImage(height: u32, width: u32, factor: u32, min_pixels: u32, max_pixels: u32) Resized {
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    const ff: f64 = @floatFromInt(factor);
    const fmin: f64 = @floatFromInt(min_pixels);
    const fmax: f64 = @floatFromInt(max_pixels);

    var h_bar = roundHalfEven(fh / ff) * ff;
    var w_bar = roundHalfEven(fw / ff) * ff;
    if (h_bar * w_bar > fmax) {
        const beta = @sqrt((fh * fw) / fmax);
        h_bar = @max(ff, std.math.floor(fh / beta / ff) * ff);
        w_bar = @max(ff, std.math.floor(fw / beta / ff) * ff);
    } else if (h_bar * w_bar < fmin) {
        const beta = @sqrt(fmin / (fh * fw));
        h_bar = std.math.ceil(fh * beta / ff) * ff;
        w_bar = std.math.ceil(fw * beta / ff) * ff;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

/// Convenience wrapper using the Qwen3VL processor defaults.
pub fn smartResizeDefault(height: u32, width: u32) Resized {
    return smartResizeImage(height, width, FACTOR, MIN_PIXELS, MAX_PIXELS);
}

/// Resampling kernels, matching Pillow's. Qwen's processor asks for BICUBIC,
/// Muse-Glimmer's for LANCZOS.
pub const Filter = enum {
    bilinear,
    bicubic,
    lanczos,

    fn support(self: Filter) f64 {
        return switch (self) {
            .bilinear => 1.0,
            .bicubic => 2.0,
            .lanczos => 3.0,
        };
    }

    /// Triangle / Keys bicubic (a = -0.5) / Lanczos-3.
    fn weight(self: Filter, distance: f64) f64 {
        const x = @abs(distance);
        switch (self) {
            .bilinear => return if (x < 1.0) 1.0 - x else 0.0,
            .bicubic => {
                if (x <= 1.0) return (1.5 * x - 2.5) * x * x + 1.0;
                if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
                return 0.0;
            },
            .lanczos => {
                if (x >= 3.0) return 0.0;
                return sinc(x) * sinc(x / 3.0);
            },
        }
    }
};

fn sinc(x: f64) f64 {
    if (x == 0.0) return 1.0;
    const t = x * std.math.pi;
    return @sin(t) / t;
}

const RESAMPLE_PRECISION_BITS: u6 = 22;
const RESAMPLE_PRECISION_SCALE: f64 = @floatFromInt(@as(i64, 1) << RESAMPLE_PRECISION_BITS);
const RESAMPLE_ROUNDING_BIAS: i64 = @as(i64, 1) << (RESAMPLE_PRECISION_BITS - 1);

const ResampleBound = struct {
    start: usize,
    len: usize,
};

const ResampleCoefficients = struct {
    bounds: []ResampleBound,
    weights: []i32,
    kernel_size: usize,

    fn deinit(self: *ResampleCoefficients, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.bounds);
        self.* = undefined;
    }
};

/// Precompute one separable Pillow-style resize axis. Downscaling widens the
/// filter footprint by `input_len / output_len`; this anti-aliasing step is the
/// material difference between Image.resize(BICUBIC) and a fixed 4-tap sampler.
/// Separable-axis geometry, shared by every consumer of this filter so the
/// fixed-point image path and the float matrix path cannot drift apart.
const AxisGeometry = struct {
    scale: f64,
    support: f64,
    inverse_filter_scale: f64,

    fn init(input_len: usize, output_len: usize, filter: Filter) AxisGeometry {
        const scale = @as(f64, @floatFromInt(input_len)) / @as(f64, @floatFromInt(output_len));
        const filter_scale = @max(scale, 1.0);
        return .{
            .scale = scale,
            .support = filter.support() * filter_scale,
            .inverse_filter_scale = 1.0 / filter_scale,
        };
    }

    fn kernelSize(self: AxisGeometry) usize {
        return @intFromFloat(@ceil(self.support) * 2.0 + 1.0);
    }

    /// The clipped input window contributing to one output coordinate, plus its
    /// filter center. Pillow's half-pixel convention throughout.
    fn window(self: AxisGeometry, output_index: usize, input_len: usize) !struct { start: usize, count: usize, center: f64 } {
        const center = (@as(f64, @floatFromInt(output_index)) + 0.5) * self.scale;
        const raw_start: i64 = @intFromFloat(center - self.support + 0.5);
        const raw_end: i64 = @intFromFloat(center + self.support + 0.5);
        const start = if (raw_start <= 0) 0 else @min(@as(usize, @intCast(raw_start)), input_len);
        const end = if (raw_end <= 0) 0 else @min(@as(usize, @intCast(raw_end)), input_len);
        if (end <= start) return error.InvalidResampleCoefficients;
        return .{ .start = start, .count = end - start, .center = center };
    }

    /// Normalized (sum == 1) tap weights for one output coordinate.
    fn taps(self: AxisGeometry, dst: []f64, start: usize, center: f64, filter: Filter) !void {
        var total: f64 = 0.0;
        for (dst, 0..) |*w, i| {
            const source_center = @as(f64, @floatFromInt(start + i)) + 0.5;
            w.* = filter.weight((source_center - center) * self.inverse_filter_scale);
            total += w.*;
        }
        if (total == 0.0) return error.InvalidResampleCoefficients;
        for (dst) |*w| w.* /= total;
    }
};

fn buildResampleCoefficients(
    allocator: std.mem.Allocator,
    input_len: usize,
    output_len: usize,
    filter: Filter,
) !ResampleCoefficients {
    if (input_len == 0 or output_len == 0) return error.InvalidImageDimensions;

    const geo = AxisGeometry.init(input_len, output_len, filter);
    const kernel_size = geo.kernelSize();
    const weights_len = try std.math.mul(usize, output_len, kernel_size);

    const bounds = try allocator.alloc(ResampleBound, output_len);
    errdefer allocator.free(bounds);
    const weights = try allocator.alloc(i32, weights_len);
    errdefer allocator.free(weights);

    const taps = try allocator.alloc(f64, kernel_size);
    defer allocator.free(taps);

    for (0..output_len) |output_index| {
        const win = try geo.window(output_index, input_len);
        try geo.taps(taps[0..win.count], win.start, win.center, filter);

        // Pillow converts each normalized coefficient to signed 22-bit fixed
        // point before applying it to uint8 image rows.
        const row = weights[output_index * kernel_size ..][0..win.count];
        for (row, taps[0..win.count]) |*weight, value| {
            const scaled = value * RESAMPLE_PRECISION_SCALE;
            weight.* = @intFromFloat(if (scaled < 0.0) scaled - 0.5 else scaled + 0.5);
        }
        bounds[output_index] = .{ .start = win.start, .len = win.count };
    }

    return .{ .bounds = bounds, .weights = weights, .kernel_size = kernel_size };
}

/// Dense `[output_len, input_len]` row-major resample matrix in f32, so an
/// axis can be resampled by a matmul instead of a gather. Same taps as the
/// image path — including the anti-aliasing footprint widening on downscale,
/// which is what makes this equal to torch's `interpolate(..., antialias=True)`
/// and unequal to a fixed 2-tap bilinear.
pub fn resampleWeightMatrix(
    allocator: std.mem.Allocator,
    dst: []f32,
    input_len: usize,
    output_len: usize,
    filter: Filter,
) !void {
    if (input_len == 0 or output_len == 0) return error.InvalidImageDimensions;
    std.debug.assert(dst.len == output_len * input_len);
    @memset(dst, 0);

    const geo = AxisGeometry.init(input_len, output_len, filter);
    const taps = try allocator.alloc(f64, geo.kernelSize());
    defer allocator.free(taps);

    for (0..output_len) |output_index| {
        const win = try geo.window(output_index, input_len);
        try geo.taps(taps[0..win.count], win.start, win.center, filter);
        const row = dst[output_index * input_len ..][0..input_len];
        for (taps[0..win.count], 0..) |value, i| row[win.start + i] = @floatCast(value);
    }
}

fn clipFixedResample(value: i64) u8 {
    const rounded = @divFloor(value, @as(i64, 1) << RESAMPLE_PRECISION_BITS);
    return @intCast(std.math.clamp(rounded, 0, 255));
}

fn normalizeQwenPixel(value: u8) f32 {
    return @as(f32, @floatFromInt(value)) / 127.5 - 1.0;
}

pub fn resizeRgbBicubicNormalizedChw(
    allocator: std.mem.Allocator,
    dst: []f32,
    rgb: []const u8,
    source_h: u32,
    source_w: u32,
    target_h: u32,
    target_w: u32,
) !void {
    return resizeRgbNormalizedChw(allocator, dst, rgb, source_h, source_w, target_h, target_w, .bicubic);
}

/// Pillow-compatible resize of interleaved RGB, followed by float32 CHW
/// normalization (mean/std 0.5 — both processors use it). Each separable pass
/// is quantized to uint8, matching the reference's `PIL.Image.resize` boundary.
pub fn resizeRgbNormalizedChw(
    allocator: std.mem.Allocator,
    dst: []f32,
    rgb: []const u8,
    source_h: u32,
    source_w: u32,
    target_h: u32,
    target_w: u32,
    filter: Filter,
) !void {
    const source_plane: usize = @as(usize, source_h) * source_w;
    const target_plane: usize = @as(usize, target_h) * target_w;
    if (source_h == 0 or source_w == 0 or target_h == 0 or target_w == 0)
        return error.InvalidImageDimensions;
    if (rgb.len != source_plane * 3 or dst.len != target_plane * 3)
        return error.InvalidImageBuffer;
    const resized = try resizeRgbU8(allocator, rgb, source_h, source_w, target_h, target_w, filter);
    defer allocator.free(resized);
    for (0..target_plane) |p| {
        inline for (0..3) |channel| dst[channel * target_plane + p] = normalizeQwenPixel(resized[p * 3 + channel]);
    }
}

/// Pillow's `Image.resize` of interleaved RGB, uint8 in and out (the caller frees): a horizontal
/// pass then a vertical one, each rounded to uint8 as Pillow does.
pub fn resizeRgbU8(
    allocator: std.mem.Allocator,
    rgb: []const u8,
    source_h: u32,
    source_w: u32,
    target_h: u32,
    target_w: u32,
    filter: Filter,
) ![]u8 {
    if (source_h == 0 or source_w == 0 or target_h == 0 or target_w == 0)
        return error.InvalidImageDimensions;
    if (rgb.len != @as(usize, source_h) * source_w * 3) return error.InvalidImageBuffer;

    var horizontal_owned: ?[]u8 = null;
    defer if (horizontal_owned) |buffer| allocator.free(buffer);
    const horizontal: []const u8 = if (target_w != source_w) resize: {
        const horizontal_len = try std.math.mul(usize, @as(usize, source_h) * target_w, 3);
        const buffer = try allocator.alloc(u8, horizontal_len);
        horizontal_owned = buffer;

        var coefficients = try buildResampleCoefficients(allocator, source_w, target_w, filter);
        defer coefficients.deinit(allocator);
        for (0..source_h) |y| {
            for (0..target_w) |x| {
                const bound = coefficients.bounds[x];
                const weights = coefficients.weights[x * coefficients.kernel_size ..][0..bound.len];
                inline for (0..3) |channel| {
                    var sum: i64 = RESAMPLE_ROUNDING_BIAS;
                    for (weights, 0..) |weight, source_offset| {
                        const source = (y * source_w + bound.start + source_offset) * 3 + channel;
                        sum += @as(i64, rgb[source]) * @as(i64, weight);
                    }
                    buffer[(y * target_w + x) * 3 + channel] = clipFixedResample(sum);
                }
            }
        }
        break :resize buffer;
    } else rgb;

    const out = try allocator.alloc(u8, @as(usize, target_h) * target_w * 3);
    errdefer allocator.free(out);
    if (target_h != source_h) {
        var coefficients = try buildResampleCoefficients(allocator, source_h, target_h, filter);
        defer coefficients.deinit(allocator);
        for (0..target_h) |y| {
            const bound = coefficients.bounds[y];
            const weights = coefficients.weights[y * coefficients.kernel_size ..][0..bound.len];
            for (0..target_w) |x| {
                inline for (0..3) |channel| {
                    var sum: i64 = RESAMPLE_ROUNDING_BIAS;
                    for (weights, 0..) |weight, source_offset| {
                        const source = ((bound.start + source_offset) * target_w + x) * 3 + channel;
                        sum += @as(i64, horizontal[source]) * @as(i64, weight);
                    }
                    out[(y * target_w + x) * 3 + channel] = clipFixedResample(sum);
                }
            }
        }
    } else @memcpy(out, horizontal);
    return out;
}

/// Number of LLM image-pad tokens an image of resized (H,W) expands to.
pub fn imageTokenCount(resized: Resized, patch: u32, merge: u32) u32 {
    const gh = resized.h / patch;
    const gw = resized.w / patch;
    return (gh / merge) * (gw / merge);
}

/// Build one temporal-patch group's `pixel_values` [N, C*tps*ps*ps] from `tps`
/// REAL consecutive decoded frames, in merge-block token order with feature
/// layout [C, tps, py, px] — the exact ordering of mlx-vlm's `_process_one`
/// transpose. Each `frames[tt]` is `[C, rh, rw]`, already rescaled+normalized
/// and resized to the SAME (rh, rw) as every other frame in the group (a
/// video's whole frame set shares one patch grid). Caller owns `out`.
pub fn buildPixelValuesVideo(
    out: []f32,
    frames: []const []const f32,
    C: u32,
    rh: u32,
    rw: u32,
    patch: u32,
    merge: u32,
) void {
    const tps: u32 = @intCast(frames.len);
    const gh = rh / patch;
    const gw = rw / patch;
    const mh = gh / merge;
    const mw = gw / merge;
    const feat = C * tps * patch * patch;
    std.debug.assert(out.len == @as(usize, gh) * gw * feat);

    const plane: usize = @as(usize, rh) * rw;
    var token: usize = 0;
    var bh: u32 = 0;
    while (bh < mh) : (bh += 1) {
        var bw: u32 = 0;
        while (bw < mw) : (bw += 1) {
            var ir: u32 = 0;
            while (ir < merge) : (ir += 1) {
                var ic: u32 = 0;
                while (ic < merge) : (ic += 1) {
                    const row = bh * merge + ir;
                    const col = bw * merge + ic;
                    const base = token * feat;
                    var f: usize = 0;
                    var c: u32 = 0;
                    while (c < C) : (c += 1) {
                        var tt: u32 = 0;
                        while (tt < tps) : (tt += 1) {
                            const frame = frames[tt];
                            var py: u32 = 0;
                            while (py < patch) : (py += 1) {
                                const y = row * patch + py;
                                var px: u32 = 0;
                                while (px < patch) : (px += 1) {
                                    const x = col * patch + px;
                                    out[base + f] = frame[@as(usize, c) * plane + @as(usize, y) * rw + x];
                                    f += 1;
                                }
                            }
                        }
                    }
                    token += 1;
                }
            }
        }
    }
}

/// Build one still IMAGE's `pixel_values` — the `tps` slots the Conv3d-as-Linear
/// weight expects are filled by duplicating the single frame, per Qwen's own
/// image processor (there is no second real frame for a still image). Thin
/// wrapper over `buildPixelValuesVideo`'s real per-frame path.
pub fn buildPixelValues(
    out: []f32,
    img_chw: []const f32,
    C: u32,
    rh: u32,
    rw: u32,
    patch: u32,
    tps: u32,
    merge: u32,
) void {
    std.debug.assert(tps <= 8);
    var reps: [8][]const f32 = undefined;
    for (0..tps) |i| reps[i] = img_chw;
    buildPixelValuesVideo(out, reps[0..tps], C, rh, rw, patch, merge);
}

test "qwen smart_resize matches reference table" {
    const cases = [_]struct { h: u32, w: u32, eh: u32, ew: u32 }{
        .{ .h = 768, .w = 768, .eh = 768, .ew = 768 }, // in-range, 48x48 grid → 576 tokens
        .{ .h = 1024, .w = 768, .eh = 1024, .ew = 768 }, // 64x48 → 768 tokens
        .{ .h = 480, .w = 640, .eh = 480, .ew = 640 }, // 30x40 → 300 tokens
        .{ .h = 4000, .w = 3000, .eh = 1152, .ew = 864 }, // > max → 72x54 → 972 tokens
        .{ .h = 100, .w = 100, .eh = 96, .ew = 96 }, // round(3.125)=3 → 96, 6x6 grid
        .{ .h = 40, .w = 40, .eh = 64, .ew = 64 }, // < min → upscaled to 64x64
    };
    for (cases) |c| {
        const r = smartResizeDefault(c.h, c.w);
        try std.testing.expectEqual(c.eh, r.h);
        try std.testing.expectEqual(c.ew, r.w);
    }
}

test "qwen: a checkpoint's 16.7 Mpx bound is clamped to what the ViT can attend" {
    const b = effectivePixelBounds(65536, 16777216);
    try std.testing.expectEqual(ENGINE_MAX_PIXELS, b.max);
    try std.testing.expect(b.clamped);
    // The live crash: 5100x3300 -> 16377 merged tokens. Under the cap it
    // is ~2300, and the engine never builds a 65k-patch score matrix.
    const r = smartResizeImage(3300, 5100, 32, b.min, b.max);
    try std.testing.expect(imageTokenCount(r, 16, 2) <= ENGINE_MAX_PIXELS / (32 * 32));
    try std.testing.expect(imageTokenCount(r, 16, 2) > 2000);
    // Defaults and sane checkpoints pass through unchanged.
    const d = effectivePixelBounds(0, 0);
    try std.testing.expectEqual(MIN_PIXELS, d.min);
    try std.testing.expectEqual(MAX_PIXELS, d.max);
    try std.testing.expect(!d.clamped);
    try std.testing.expectEqual(@as(u32, 1003520), effectivePixelBounds(3136, 1003520).max);
}

test "qwen smart_resize honors checkpoint processor pixel bounds" {
    const r = smartResizeImage(971, 1619, 32, 65536, 16777216);
    try std.testing.expectEqual(@as(u32, 960), r.h);
    try std.testing.expectEqual(@as(u32, 1632), r.w);
    try std.testing.expectEqual(@as(u32, 1530), imageTokenCount(r, 16, 2));
}

test "qwen pixel minimum cannot raise the engine ceiling" {
    for ([_]u32{ 0, 16777216 }) |max| {
        const bounds = effectivePixelBounds(16777216, max);
        try std.testing.expectEqual(ENGINE_MAX_PIXELS, bounds.max);
        try std.testing.expect(bounds.min <= bounds.max);
        try std.testing.expect(bounds.clamped);
        const resized = smartResizeImage(512, 512, FACTOR, bounds.min, bounds.max);
        try std.testing.expect(@as(u64, resized.h) * resized.w <= ENGINE_MAX_PIXELS);
    }
}

test "qwen bicubic RGB preprocessing preserves colors and interpolates" {
    const rgb = [_]u8{
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        255, 255, 255, // white
    };
    var same: [3 * 2 * 2]f32 = undefined;
    try resizeRgbBicubicNormalizedChw(std.testing.allocator, &same, &rgb, 2, 2, 2, 2);
    const expected = [_]f32{
        1,  -1, -1, 1,
        -1, 1,  -1, 1,
        -1, -1, 1,  1,
    };
    for (same, expected) |got, want| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }

    const checker = [_]u8{
        0,   0,   0,
        255, 255, 255,
        255, 255, 255,
        0,   0,   0,
    };
    var upsampled: [3 * 3 * 3]f32 = undefined;
    try resizeRgbBicubicNormalizedChw(std.testing.allocator, &upsampled, &checker, 2, 2, 3, 3);
    const center: usize = 4;
    const pillow_center = normalizeQwenPixel(128);
    try std.testing.expectApproxEqAbs(pillow_center, upsampled[center], 1e-6);
    try std.testing.expectApproxEqAbs(pillow_center, upsampled[9 + center], 1e-6);
    try std.testing.expectApproxEqAbs(pillow_center, upsampled[18 + center], 1e-6);
}

test "resampleWeightMatrix reproduces torch's antialiased bilinear on one axis" {
    // Reference: torch.nn.functional.interpolate(mode="bilinear",
    // align_corners=False, antialias=True) on a 16-long signal — the exact call
    // Siglip2 makes to resample its 16x16 position table onto an image's own
    // patch grid. A grid axis SHORTER than 16 is a real downscale, where a
    // fixed 2-tap bilinear (no footprint widening) diverges; 16 -> 8 below is
    // that case, and its last entry (204.571426, not 210.5) is the clipped
    // edge window renormalizing, which a hand-rolled sampler also misses.
    const signal = [_]f64{ 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225 };
    const cases = [_]struct { out: usize, want: []const f32 }{
        .{ .out = 14, .want = &.{ 0.166667, 1.833334, 5.944445, 12.500000, 21.500004, 32.944450, 47.736851, 65.894753, 86.277786, 108.166679, 132.500015, 159.277786, 188.500031, 220.166687 } },
        .{ .out = 20, .want = &.{ 0.000000, 0.700000, 2.500000, 5.500000, 9.700001, 15.300001, 22.300003, 30.500000, 39.900002, 50.500000, 62.500008, 75.899994, 90.500000, 106.300011, 123.300011, 141.700012, 161.500000, 182.500000, 204.700012, 225.000000 } },
        .{ .out = 8, .want = &.{ 1.000000, 7.000000, 21.000000, 43.000000, 73.000000, 111.000000, 157.000000, 204.571426 } },
        .{ .out = 32, .want = &.{ 0.000000, 0.250000, 0.750000, 1.750000, 3.250000, 5.250000, 7.750000, 10.750000, 14.250000, 18.250000, 22.750000, 27.750000, 33.250000, 39.250000, 45.750000, 52.750000, 60.250000, 68.250000, 76.750000, 85.750000, 95.250000, 105.250000, 115.750000, 126.750000, 138.250000, 150.250000, 162.750000, 175.750000, 189.250000, 203.250000, 217.750000, 225.000000 } },
    };
    const a = std.testing.allocator;
    for (cases) |c| {
        const m = try a.alloc(f32, c.out * signal.len);
        defer a.free(m);
        try resampleWeightMatrix(a, m, signal.len, c.out, .bilinear);
        for (0..c.out) |i| {
            var acc: f64 = 0;
            for (signal, 0..) |v, j| acc += v * m[i * signal.len + j];
            std.testing.expectApproxEqAbs(c.want[i], @as(f32, @floatCast(acc)), 2e-4) catch |e| {
                std.debug.print("16->{d} [{d}] = {d:.6}, want {d:.6}\n", .{ c.out, i, acc, c.want[i] });
                return e;
            };
        }
    }
}

test "qwen bicubic downscale matches Pillow anti-aliased RGB output" {
    var rgb: [8 * 8 * 3]u8 = undefined;
    for (0..8) |y| {
        for (0..8) |x| {
            for (0..3) |channel| {
                rgb[(y * 8 + x) * 3 + channel] =
                    @intCast((x * 31 + y * 17 + channel * 53 + x * y * 7) % 256);
            }
        }
    }

    var actual: [3 * 2 * 3]f32 = undefined;
    try resizeRgbBicubicNormalizedChw(std.testing.allocator, &actual, &rgb, 8, 8, 2, 3);
    // Pillow 12.3.0: Image.fromarray(rgb).resize((3, 2), Image.Resampling.BICUBIC)
    // flattened as CHW after the uint8 resize boundary.
    const expected_u8 = [_]u8{
        73,  135, 130, 141, 123, 119,
        124, 129, 112, 156, 115, 120,
        156, 116, 104, 119, 125, 114,
    };
    for (actual, expected_u8) |got, expected| {
        try std.testing.expectApproxEqAbs(normalizeQwenPixel(expected), got, 1e-6);
    }
}

test "qwen buildPixelValues merge-order + [C,tps,py,px] feature layout" {
    const a = std.testing.allocator;
    // 2x2 image, patch=1, merge=2, tps=2, C=3. img[c,y,x] = c*100 + y*10 + x.
    const C: u32 = 3;
    const rh: u32 = 2;
    const rw: u32 = 2;
    var img: [3 * 2 * 2]f32 = undefined;
    for (0..C) |c| for (0..rh) |y| for (0..rw) |x| {
        img[c * 4 + y * 2 + x] = @floatFromInt(c * 100 + y * 10 + x);
    };
    const pv = try a.alloc(f32, 4 * 6);
    defer a.free(pv);
    buildPixelValues(pv, &img, C, rh, rw, 1, 2, 2);
    // 4 tokens (merge-block order over a single 2x2 block) × feat 6 (=C*tps*1*1).
    const expect = [_]f32{
        0, 0, 100, 100, 200, 200, // token0 row0col0
        1, 1, 101, 101, 201, 201, // token1 row0col1
        10, 10, 110, 110, 210, 210, // token2 row1col0
        11, 11, 111, 111, 211, 211, // token3 row1col1
    };
    try std.testing.expectEqualSlices(f32, &expect, pv);
}

test "qwen buildPixelValuesVideo reads REAL per-frame data, not one frame duplicated" {
    const a = std.testing.allocator;
    // Two distinct 2x2 "frames" (single channel, patch=1, merge=2 → one token
    // covering the whole 2x2 grid). frame0[y,x] = y*10+x; frame1 = frame0+1000
    // so the two temporal slots are trivially distinguishable in the output.
    const C: u32 = 1;
    const rh: u32 = 2;
    const rw: u32 = 2;
    var f0: [4]f32 = .{ 0, 1, 10, 11 };
    var f1: [4]f32 = .{ 1000, 1001, 1010, 1011 };
    const frames = [_][]const f32{ &f0, &f1 };
    const pv = try a.alloc(f32, 4 * 2); // 4 tokens × feat 2 (C*tps*1*1)
    defer a.free(pv);
    buildPixelValuesVideo(pv, &frames, C, rh, rw, 1, 2);
    // Merge-block order over the single 2x2 block; each token's feature is
    // [frame0_px, frame1_px] — proving both real frames land in the output,
    // not one frame duplicated tps times (that would make both slots equal).
    const expect = [_]f32{
        0, 1000, // token0 row0col0
        1, 1001, // token1 row0col1
        10, 1010, // token2 row1col0
        11, 1011, // token3 row1col1
    };
    try std.testing.expectEqualSlices(f32, &expect, pv);
}

test "qwen image token count" {
    try std.testing.expectEqual(@as(u32, 576), imageTokenCount(.{ .h = 768, .w = 768 }, 16, 2));
    try std.testing.expectEqual(@as(u32, 972), imageTokenCount(.{ .h = 1152, .w = 864 }, 16, 2));
    try std.testing.expectEqual(@as(u32, 300), imageTokenCount(.{ .h = 480, .w = 640 }, 16, 2));
}

test "qwen roundHalfEven matches python banker's rounding" {
    try std.testing.expectEqual(@as(f64, 0.0), roundHalfEven(0.5)); // → even 0
    try std.testing.expectEqual(@as(f64, 2.0), roundHalfEven(1.5)); // → even 2
    try std.testing.expectEqual(@as(f64, 2.0), roundHalfEven(2.5)); // → even 2
    try std.testing.expectEqual(@as(f64, 4.0), roundHalfEven(3.5)); // → even 4
    try std.testing.expectEqual(@as(f64, 3.0), roundHalfEven(3.125));
    try std.testing.expectEqual(@as(f64, 94.0), roundHalfEven(93.75));
}

pub fn binary(comptime f: anytype, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(f(&out, a, b, s));
    return out;
}

pub fn astype(a: mlx.mlx_array, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, a, dtype, s));
    return out;
}

pub fn reshape(a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

fn sumProduct(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const af = try astype(a, .float32, s);
    defer _ = mlx.mlx_array_free(af);
    const bf = try astype(b, .float32, s);
    defer _ = mlx.mlx_array_free(bf);
    const flat = [_]c_int{-1};
    const a1 = try reshape(af, &flat, s);
    defer _ = mlx.mlx_array_free(a1);
    const b1 = try reshape(bf, &flat, s);
    defer _ = mlx.mlx_array_free(b1);
    const prod = try binary(mlx.mlx_multiply, a1, b1, s);
    defer _ = mlx.mlx_array_free(prod);
    var total = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(total);
    try mlx.check(mlx.mlx_sum(&total, prod, false, s));
    try mlx.check(mlx.mlx_array_eval(total));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, total));
    return v;
}

/// Cosine of two tensors read as flat vectors, in f32. NaN fails every bar.
pub fn cosineSim(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const ab = try sumProduct(a, b, s);
    const aa = try sumProduct(a, a, s);
    const bb = try sumProduct(b, b, s);
    if (!std.math.isFinite(ab) or aa <= 0 or bb <= 0) return std.math.nan(f32);
    return ab / (@sqrt(aa) * @sqrt(bb));
}

/// ||a|| / ||b||: the scale error a cosine cannot see.
pub fn rmsRatio(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const aa = try sumProduct(a, a, s);
    const bb = try sumProduct(b, b, s);
    if (!std.math.isFinite(aa) or bb <= 0) return std.math.nan(f32);
    return @sqrt(aa / bb);
}
