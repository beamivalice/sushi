//! DeepSeek-V4.1-Flash media: the aligner tower, its letterboxing processor and the image span a prompt holds.
//! Images only, no M-RoPE. The rows are spliced inside the model's own pass.
const std = @import("std");
const chat = @import("../chat.zig");
const log = @import("../log.zig");
const model = @import("../model.zig");
const arch_media = @import("../arch/media.zig");
const dsv41_vision = @import("dsv41_vision.zig");

pub const mode: arch_media.Mode = .dsv41;
pub const Tower = dsv41_vision.Dsv41Vision;
pub const scratchBytes = dsv41_vision.encodeScratchBytes;

pub fn claims(config: *const model.ModelConfig) bool {
    return config.isDsv41() or config.dsv41 != null;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    const c = config.dsv41 orelse return .{};
    const v = c.vision orelse return .{};
    return .{
        .mode = .dsv41,
        .patch = v.patch,
        .merge = v.downsample,
        .min_pixels = v.min_pixels,
        .max_tokens = v.max_tokens,
        .max_wh_ratio = v.max_wh_ratio,
    };
}

pub fn initTower(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
    return Tower.init(allocator, config, weights);
}

fn processor(vp: chat.VisionPreproc) dsv41_vision.Processor {
    return .{ .patch = vp.patch, .downsample = vp.merge, .max_tokens = vp.max_tokens, .min_pixels = vp.min_pixels, .max_wh_ratio = vp.max_wh_ratio };
}

pub fn maxImagePatches(vp: chat.VisionPreproc) u64 {
    const merge: u64 = @max(vp.merge, 1);
    return merge * merge * (vp.max_tokens - 2);
}

/// An image's span: its aligner cells, a newline per row, start and end.
pub fn imageRows(merge: u32, grid_h: u32, grid_w: u32) usize {
    return dsv41_vision.spanRows(grid_h, grid_w, merge);
}

pub fn planImage(vp: chat.VisionPreproc, src_w: u32, src_h: u32) ?arch_media.ImagePlan {
    const plan = dsv41_vision.planGrid(src_w, src_h, processor(vp)) orelse return null;
    return .{ .width = plan.w, .height = plan.h, .grid_h = plan.vit_h, .grid_w = plan.vit_w };
}

/// The release letterboxes into a canvas of whole patches.
pub fn decodeImage(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, vp: chat.VisionPreproc) ?chat.ImageData {
    const pv = dsv41_vision.preprocess(allocator, rgb[0 .. @as(usize, src_h) * src_w * 3], src_h, src_w, processor(vp)) catch return null;
    log.info("  Decoded {d}x{d} image → deepseek grid {d}x{d} ({d} tokens, canvas {d}x{d})\n", .{ src_w, src_h, pv.grid_h, pv.grid_w, dsv41_vision.spanRows(pv.grid_h, pv.grid_w, vp.merge), pv.w, pv.h });
    return .{ .pixels = pv.bytes, .width = pv.w, .height = pv.h, .grid_h = pv.grid_h, .grid_w = pv.grid_w };
}

pub fn testConfig() model.ModelConfig {
    return model.parseConfigFromJson(std.testing.allocator, @import("dsv41_config.zig").release_config) catch unreachable;
}
