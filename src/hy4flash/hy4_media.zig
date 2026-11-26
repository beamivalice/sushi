//! Hy4-Flash's media slot (`arch/media.zig` reads it): the tower and merger (hy4_vision.zig) over the release
//! processor's image: smart resize to whole 32-pixel cells (bicubic), mean and std 0.5, and raster patches of (C, P, P)
//! values. The rows splice inside the model's own pass (`Model.VisionRows`). Images only: no video processor.
const std = @import("std");
const model = @import("../model.zig");
const mlx = @import("../mlx.zig");
const chat = @import("../chat.zig");
const log = @import("../log.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const vision_common = @import("../vision_common.zig");
const arch_media = @import("../arch/media.zig");
const hy4_vision = @import("hy4_vision.zig");
const hy4_encode_plan = @import("hy4_encode_plan.zig");

pub const mode = .hy4flash;

/// An image's pixel bounds, from the release's processor_config.json; the engine caps the top at
/// `vision_common.ENGINE_MAX_PIXELS`, as it does Qwen's.
const min_pixels: u32 = 4096;
const max_pixels: u32 = 4194304;

pub const Tower = struct {
    tower: hy4_vision.Tower,
    out_hidden: u32,

    pub fn deinit(self: *Tower) void {
        self.tower.deinit();
    }

    /// The engine runs every Hy4 forward, the tower's too.
    pub fn ownStream(_: *const Tower) ?mlx.mlx_stream {
        return mlx.engine.stream();
    }

    /// A request's `Media`: its images' `rows` rows (`hy4_encode_plan.makeMedia`).
    pub fn makeMedia(self: *Tower, rows: usize) !mlx.mlx_array {
        return hy4_encode_plan.makeMedia(self.tower.v, rows);
    }

    /// One image's rows from its raster patches (host f32 [N, 3·P·P]) written into the request's `media` from `row0`:
    /// an execution on the arena where the model's load planned the encode (`hy4_encode_plan.run`), else on the pool.
    pub fn encodeInto(self: *Tower, xfm: *@import("../trunk.zig").Trunk, media: *mlx.mlx_array, row0: usize, pixels: []const u8, grid_h: u32, grid_w: u32) !void {
        const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
        return hy4_encode_plan.run(m.plans, &self.tower, media, row0, pixels, grid_h, grid_w);
    }

    /// The image's prompt rows [1, rows, out_hidden] f32 from its raster patches [N, 3·P·P].
    pub fn forward(self: *Tower, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        var o: Ops = .{ .s = mlx.engine.stream() };
        defer o.deinit();
        const rows = try self.tower.forward(&o, patches, grid_h, grid_w, null);
        const out = try o.result(try o.reshape(rows, &.{ 1, -1, @intCast(self.out_hidden) }));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }
};

pub fn claims(config: *const model.ModelConfig) bool {
    const c = config.hy4flash orelse return false;
    return c.vision != null;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    const v = (config.hy4flash orelse return .{}).vision orelse return .{};
    return .{ .mode = .hy4flash, .patch = v.patch, .tps = 1, .merge = v.merge, .min_pixels = min_pixels, .max_pixels = max_pixels };
}

pub fn initTower(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
    const v = (config.hy4flash orelse return error.MissingVisionWeights).vision orelse return error.MissingVisionWeights;
    if (weights.get("vit.embeddings.patch_embedding.weight") == null) return error.MissingVisionWeights;
    var tower = try hy4_vision.Tower.init(allocator, v, weights);
    errdefer tower.deinit();
    try tower.warmEngine();
    log.info("Vision encoder: Hy4-Flash ViT (depth={d}, hidden={d}, heads={d}, merge={d}, out_hidden={d})\n", .{ v.layers, v.hidden, v.heads, v.merge, v.out_hidden });
    return .{ .tower = tower, .out_hidden = v.out_hidden };
}

/// An encode of `patches` patches at its worst shape (one merge cell wide: the most newline rows), plus the margin.
pub fn scratchBytes(config: *const model.ModelConfig, patches: u64) u64 {
    const v = (config.hy4flash orelse return 0).vision orelse return 0;
    if (patches == 0) return 0;
    const m: u64 = v.merge;
    const held = hy4_vision.encodeHeldBytes(v, hy4_vision.attn_query_rows, @max(patches / m, m), m) catch |e| {
        // No bill, no admission: the image is refused rather than under-billed.
        log.err("hy_v4_flash: the vision encode's bill: {t}\n", .{e});
        return std.math.maxInt(u64);
    };
    return held + @import("../bills.zig").BILL_MARGIN_BYTES;
}

/// A request's encode bill (`arch/media.zig` `mediaBill`): its `Media`, closed form from its `rows`, and its images'
/// host f32 pixels (`input_bytes`), the request's while they wait for and feed the encode. The encode's work is the
/// load's region's (D97).
pub fn mediaBill(config: *const model.ModelConfig, _: u64, input_bytes: u64, rows: usize) u64 {
    const v = (config.hy4flash orelse return 0).vision orelse return 0;
    return hy4_encode_plan.mediaBytes(v, rows) +| input_bytes;
}

fn bounds(vp: chat.VisionPreproc) vision_common.PixelBounds {
    return vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
}

pub fn maxImagePatches(vp: chat.VisionPreproc) u64 {
    return bounds(vp).max / @max(@as(u64, vp.patch) * vp.patch, 1);
}

/// The tallest admitted grid is one merged cell wide: every cell is a row beside its newline.
pub fn maxImageRows(vp: chat.VisionPreproc) usize {
    const merge: u64 = @max(vp.merge, 1);
    return @intCast(2 * (maxImagePatches(vp) / (merge * merge)));
}

/// Each merged row of an image and its newline.
pub fn imageRows(merge: u32, grid_h: u32, grid_w: u32) usize {
    return @as(usize, grid_h / merge) * (grid_w / merge + 1);
}

pub fn planImage(vp: chat.VisionPreproc, src_w: u32, src_h: u32) ?arch_media.ImagePlan {
    const factor = std.math.mul(u32, vp.patch, vp.merge) catch return null;
    if (factor == 0 or src_w == 0 or src_h == 0) return null;
    const b = bounds(vp);
    const r = vision_common.smartResizeImage(src_h, src_w, factor, b.min, b.max);
    return .{ .width = r.w, .height = r.h, .grid_h = r.h / vp.patch, .grid_w = r.w / vp.patch };
}

/// The processor's patches of an RGB8 image: resized (Pillow's bicubic), normalized, and cut into raster patches
/// whose values run (C, P, P).
pub fn decodeImage(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, vp: chat.VisionPreproc) ?chat.ImageData {
    const plan = planImage(vp, src_w, src_h) orelse return null;
    const h: usize = plan.height;
    const w: usize = plan.width;
    const p: usize = vp.patch;
    const chw = allocator.alloc(f32, 3 * h * w) catch return null;
    defer allocator.free(chw);
    vision_common.resizeRgbNormalizedChw(allocator, chw, rgb[0 .. @as(usize, src_h) * src_w * 3], src_h, src_w, plan.height, plan.width, .bicubic) catch return null;
    const feat = 3 * p * p;
    const bytes = allocator.alloc(u8, @as(usize, plan.grid_h) * plan.grid_w * feat * 4) catch return null;
    const out = @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
    for (0..plan.grid_h) |pi| for (0..plan.grid_w) |pj| {
        const base = (pi * plan.grid_w + pj) * feat;
        for (0..3) |c| for (0..p) |y| {
            const src = chw[c * h * w + (pi * p + y) * w + pj * p ..][0..p];
            @memcpy(out[base + c * p * p + y * p ..][0..p], src);
        };
    };
    log.info("  Decoded {d}x{d} image → hy4 grid {d}x{d} ({d} tokens)\n", .{ src_w, src_h, plan.grid_h, plan.grid_w, imageRows(vp.merge, plan.grid_h, plan.grid_w) });
    return .{ .pixels = bytes, .width = plan.width, .height = plan.height, .grid_h = plan.grid_h, .grid_w = plan.grid_w };
}
