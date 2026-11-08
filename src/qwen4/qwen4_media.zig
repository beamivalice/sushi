//! Flash-Next media: the Qwen3-VL tower, its smart-resize processor and the M-RoPE positions of a prompt's blocks.
const std = @import("std");
const chat = @import("../chat.zig");
const log = @import("../log.zig");
const model = @import("../model.zig");
const mrope = @import("../mrope.zig");
const vision_common = @import("../vision_common.zig");
const arch_media = @import("../arch/media.zig");
const qwen_vision = @import("qwen_vision.zig");

pub const mode = .qwen;
pub const Tower = qwen_vision.QwenVision;
pub const scratchBytes = qwen_vision.encodeScratchBytes;
pub const quantGroupAtLoad = qwen_vision.quantGroupAtLoad;

pub fn claims(config: *const model.ModelConfig) bool {
    return config.qwen_vision;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    return .{
        .mode = .qwen,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_pixels = config.qv_min_pixels,
        .max_pixels = config.qv_max_pixels,
    };
}

pub fn initTower(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
    const tower = try Tower.init(allocator, config, weights);
    log.info("Vision encoder: Qwen3-VL ViT (depth={d}, hidden={d}, heads={d}, merge={d}, out_hidden={d})\n", .{
        config.qv_depth, config.qv_hidden, config.qv_heads, config.qv_merge, config.qv_out_hidden,
    });
    return tower;
}

var pixel_clamp_logged: bool = false;

fn pixelBounds(vp: chat.VisionPreproc) vision_common.PixelBounds {
    const bounds = vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
    if (bounds.clamped and !pixel_clamp_logged) {
        pixel_clamp_logged = true;
        log.info("[vision] image area capped at {d} px (checkpoint declares {d}): the ViT materializes its full attention\n", .{ vision_common.ENGINE_MAX_PIXELS, vp.max_pixels });
    }
    return bounds;
}

pub fn resize(vp: chat.VisionPreproc, factor: u32, src_h: u32, src_w: u32) vision_common.Resized {
    const bounds = pixelBounds(vp);
    return vision_common.smartResizeImage(src_h, src_w, factor, bounds.min, bounds.max);
}

pub fn normalizeChw(allocator: std.mem.Allocator, vp: chat.VisionPreproc, dst: []f32, rgb: []const u8, src_h: u32, src_w: u32, dst_h: u32, dst_w: u32, frames: u32) !void {
    _ = vp;
    _ = frames;
    return vision_common.resizeRgbNormalizedChw(allocator, dst, rgb, src_h, src_w, dst_h, dst_w, .bicubic);
}

pub fn videoResize(vp: chat.VisionPreproc, frames: u32, first_h: u32, first_w: u32, factor: u32) ?vision_common.Resized {
    _ = frames;
    const bounds = pixelBounds(vp);
    return vision_common.smartResizeImage(first_h, first_w, factor, bounds.min, bounds.max);
}

/// Interleaved M-RoPE over the FINAL prompt (after pad expansion) and every image and video grid in prompt order.
/// Caller owns `pos`.
pub fn mropePositions(allocator: std.mem.Allocator, prompt_ids: []const u32, images: []const chat.ImageData, videos: []const chat.VideoData, config: *const model.ModelConfig) !arch_media.Mrope {
    // Each modality's grids in its own document order: getRopeIndex
    // interleaves the two lists by whichever marker occurs first in tokens.
    var image_grids = std.ArrayList(mrope.ImageGrid).empty;
    defer image_grids.deinit(allocator);
    var video_grids = std.ArrayList(mrope.ImageGrid).empty;
    defer video_grids.deinit(allocator);
    for (images) |im| {
        if (im.grid_h > 0) try image_grids.append(allocator, .{ .t = 1, .h = im.grid_h, .w = im.grid_w });
    }
    for (videos) |vd| {
        try video_grids.append(allocator, .{ .t = vd.grid_t, .h = vd.grid_h, .w = vd.grid_w });
    }
    if (image_grids.items.len == 0 and video_grids.items.len == 0) return .{};

    var ri = mrope.getRopeIndex(allocator, prompt_ids, image_grids.items, video_grids.items, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge) catch |err| {
        log.warn("M-RoPE get_rope_index failed ({s}); falling back to scalar RoPE\n", .{@errorName(err)});
        return .{};
    };
    defer ri.deinit();
    const total = prompt_ids.len;
    const flat = try allocator.alloc(i32, 3 * total);
    @memcpy(flat[0..total], ri.pos[0]);
    @memcpy(flat[total .. 2 * total], ri.pos[1]);
    @memcpy(flat[2 * total .. 3 * total], ri.pos[2]);
    log.info("  M-RoPE: {d} images, {d} videos, position ids over {d} tokens, decode delta {d}\n", .{ image_grids.items.len, video_grids.items.len, total, ri.delta });
    return .{ .pos = flat, .total = total, .delta = ri.delta };
}
