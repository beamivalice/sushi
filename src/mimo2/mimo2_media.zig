//! MiMo-V2.6-Flash media: the MiMo-ViT tower and its smart-resize processor. Images only, no M-RoPE.
const std = @import("std");
const chat = @import("../chat.zig");
const model = @import("../model.zig");
const vision_common = @import("../vision_common.zig");
const arch_media = @import("../arch/media.zig");
const mimo_vision = @import("mimo_vision.zig");

pub const mode: arch_media.Mode = .mimo;
pub const Tower = mimo_vision.MimoVision;
pub const scratchBytes = mimo_vision.encodeScratchBytes;

pub fn claims(config: *const model.ModelConfig) bool {
    return config.mimo_vision;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    return .{
        .mode = .mimo,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_pixels = config.qv_min_pixels,
        .max_pixels = config.qv_max_pixels,
    };
}

pub fn initTower(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
    return Tower.init(allocator, config, weights);
}

pub fn resize(vp: chat.VisionPreproc, factor: u32, src_h: u32, src_w: u32) vision_common.Resized {
    const bounds = vision_common.effectivePixelBounds(vp.min_pixels, vp.max_pixels);
    return mimo_vision.smartResize(src_h, src_w, factor, bounds.min, bounds.max);
}

pub fn normalizeChw(allocator: std.mem.Allocator, vp: chat.VisionPreproc, dst: []f32, rgb: []const u8, src_h: u32, src_w: u32, dst_h: u32, dst_w: u32, frames: u32) !void {
    _ = allocator;
    _ = vp;
    _ = frames;
    return mimo_vision.resizeNormalizedChw(dst, rgb, src_h, src_w, dst_h, dst_w);
}
