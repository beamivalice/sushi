//! The vision geometries of the served archs for the media tests: their processors, their hand-flagged towers and the
//! processor a wire parse is given.
const arch_media = @import("arch/media.zig");
const chat_mod = @import("chat.zig");
const model_mod = @import("model.zig");
const request_media = @import("request_media.zig");

pub const vision_modes = [_]chat_mod.VisionPreproc{
    .{ .mode = arch_media.modeNamed("qwen") orelse .gemma, .patch = 16, .tps = 2, .merge = 2, .min_pixels = 65536, .max_pixels = 1048576 },
    .{ .mode = arch_media.modeNamed("mimo") orelse .gemma, .patch = 16, .tps = 2, .merge = 2, .min_pixels = 8192, .max_pixels = 8388608 },
    .{ .mode = arch_media.modeNamed("glm5") orelse .gemma, .patch = 14, .tps = 2, .merge = 2, .min_tokens = 16, .max_tokens = 8000 },
};

/// Hand-flagged tower configs (Flash-Next, MiMo, GLM); a row is null when its arch is not in the build.
pub fn towerConfigs() [3]?model_mod.ModelConfig {
    var rows: [3]?model_mod.ModelConfig = .{
        // Qwen3.8 packs: no processor config, so the 1,003,520 px default.
        .{ .has_vision = true, .qwen_vision = true, .qv_heads = 16, .qv_hidden = 1152, .qv_intermediate = 4304, .qv_out_hidden = 2560, .qv_patch = 16, .qv_temporal_patch = 2, .qv_merge = 2 },
        .{ .has_vision = true, .mimo_vision = true, .qv_heads = 32, .mvit_kv_heads = 8, .qv_head_dim = 64, .qv_hidden = 1280, .qv_intermediate = 4608, .qv_out_hidden = 4096, .qv_patch = 16, .qv_temporal_patch = 2, .qv_merge = 2, .qv_min_pixels = 8192, .qv_max_pixels = 8388608 },
        .{ .has_vision = true, .glm5_vision = true, .qv_heads = 16, .qv_head_dim = 64, .qv_hidden = 1024, .qv_intermediate = 4096, .qv_out_hidden = 4096, .glmv_projection_intermediate = 10240, .qv_patch = 14, .qv_temporal_patch = 2, .qv_merge = 2, .glmv_min_image_tokens = 16, .glmv_max_image_tokens = 8000 },
    };
    for (&rows) |*row| if (!arch_media.claimed(&row.*.?)) {
        row.* = null;
    };
    return rows;
}

pub const Processors = struct { qwen: ?chat_mod.VisionPreproc, mimo: ?chat_mod.VisionPreproc, glm5: ?chat_mod.VisionPreproc, dsv41: ?chat_mod.VisionPreproc };

pub fn processors() Processors {
    const cfgs = towerConfigs();
    return .{
        .qwen = if (cfgs[0]) |*c| request_media.visionPreprocFromConfig(c) else null,
        .mimo = if (cfgs[1]) |*c| request_media.visionPreprocFromConfig(c) else null,
        .glm5 = if (cfgs[2]) |*c| request_media.visionPreprocFromConfig(c) else null,
        .dsv41 = request_media.testVp("deepseek_v41"),
    };
}

/// The processor the wire-shape parses read images with.
pub const wire_vp: chat_mod.VisionPreproc = .{ .mode = arch_media.modeNamed("qwen") orelse .gemma, .patch = 16, .tps = 2, .merge = 2 };
