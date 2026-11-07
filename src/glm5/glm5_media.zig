//! GLM-5.3-Flash media: the GLM tower, its token-budget processor and the timestamped video blocks. No M-RoPE.
const std = @import("std");
const chat = @import("../chat.zig");
const model = @import("../model.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const vision_common = @import("../vision_common.zig");
const arch_media = @import("../arch/media.zig");
const glm5_vision = @import("glm5_vision.zig");

pub const mode = .glm5;
pub const Tower = glm5_vision.GlmVision;
pub const scratchBytes = glm5_vision.encodeScratchBytes;

pub fn claims(config: *const model.ModelConfig) bool {
    return config.glm5_vision;
}

pub fn preproc(config: *const model.ModelConfig) chat.VisionPreproc {
    return .{
        .mode = .glm5,
        .patch = config.qv_patch,
        .tps = config.qv_temporal_patch,
        .merge = config.qv_merge,
        .min_tokens = config.glmv_min_image_tokens,
        .max_tokens = config.glmv_max_image_tokens,
        .max_video_tokens = config.glmv_max_video_tokens,
    };
}

pub fn initTower(allocator: std.mem.Allocator, config: model.ModelConfig, weights: *model.Weights) !Tower {
    return Tower.init(allocator, config, weights);
}

pub fn maxImagePatches(vp: chat.VisionPreproc) u64 {
    const merge: u64 = @max(vp.merge, 1);
    return @as(u64, vp.max_tokens) * merge * merge;
}

pub fn resize(vp: chat.VisionPreproc, factor: u32, src_h: u32, src_w: u32) vision_common.Resized {
    return glm5_vision.smartResize(vp.tps, src_h, src_w, vp.tps, factor, vp.min_tokens, vp.max_tokens);
}

pub fn normalizeChw(allocator: std.mem.Allocator, vp: chat.VisionPreproc, dst: []f32, rgb: []const u8, src_h: u32, src_w: u32, dst_h: u32, dst_w: u32, frames: u32) !void {
    const factor = std.math.mul(u32, vp.patch, vp.merge) catch return error.InvalidImageDimensions;
    return glm5_vision.resizeNormalizedChw(allocator, dst, rgb, src_h, src_w, dst_h, dst_w, frames, vp.tps, factor, vp.min_tokens);
}

pub fn videoResize(vp: chat.VisionPreproc, frames: u32, first_h: u32, first_w: u32, factor: u32) ?vision_common.Resized {
    if (@as(u64, (frames + vp.tps - 1) / vp.tps) > vp.max_video_tokens) return null;
    return glm5_vision.smartResize(frames, first_h, first_w, vp.tps, factor, vp.min_tokens, vp.max_video_tokens);
}

/// Each temporal group is its own start, image-pad run, end and a "N.N seconds" stamp.
pub fn expandVideo(allocator: std.mem.Allocator, out: *std.ArrayList(u32), config: *const model.ModelConfig, tokenizer: ?*const tokenizer_mod.Tokenizer, video: chat.VideoData, rows: usize) !void {
    const tok = tokenizer orelse return error.MediaLayoutUnsupported;
    const per_group = rows / video.grid_t;
    for (0..video.grid_t) |group| {
        try out.append(allocator, config.vision_start_token_id);
        try out.appendNTimes(allocator, config.image_token_id, per_group);
        try out.append(allocator, config.vision_end_token_id);
        var stamp: [64]u8 = undefined;
        const seconds = @as(f64, @floatFromInt(group * config.qv_temporal_patch)) / video.fps;
        const text = try std.fmt.bufPrint(&stamp, "{d:.1} seconds", .{seconds});
        const stamp_ids = try tok.encode(allocator, text);
        defer allocator.free(stamp_ids);
        try out.appendSlice(allocator, stamp_ids);
    }
}

pub fn testConfig() model.ModelConfig {
    return model.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json")) catch unreachable;
}
