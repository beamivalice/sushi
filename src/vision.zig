const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const qwen_vision = @import("qwen_vision.zig");
const mimo_vision = @import("mimo_vision.zig");
const glm5_vision = @import("glm5_vision.zig");
const dsv41_vision = @import("dsv41_vision.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

/// The resident image tower of a served model: exactly one patch-grid ViT
/// (Qwen3-VL, MiMo-ViT, GLM or DeepSeek-V4.1), each owning its own weights and GPU stream.
pub const VisionEncoder = struct {
    allocator: std.mem.Allocator,
    /// Stream for the ops that join per-image embeddings.
    s: mlx.mlx_stream,
    qwen: ?qwen_vision.QwenVision = null,
    mimo: ?mimo_vision.MimoVision = null,
    glm5: ?glm5_vision.GlmVision = null,
    dsv41: ?dsv41_vision.Dsv41Vision = null,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights) !VisionEncoder {
        if (config.qwen_vision) {
            const qv = try qwen_vision.QwenVision.init(allocator, config, weights);
            log.info("Vision encoder: Qwen3-VL ViT (depth={d}, hidden={d}, heads={d}, merge={d}, out_hidden={d})\n", .{
                config.qv_depth, config.qv_hidden, config.qv_heads, config.qv_merge, config.qv_out_hidden,
            });
            return .{ .allocator = allocator, .s = mlx.mlx_default_gpu_stream_new(), .qwen = qv };
        }
        if (config.mimo_vision) {
            const mv = try mimo_vision.MimoVision.init(allocator, config, weights);
            return .{ .allocator = allocator, .s = mlx.mlx_default_gpu_stream_new(), .mimo = mv };
        }
        if (config.glm5_vision) {
            const gv = try glm5_vision.GlmVision.init(allocator, config, weights);
            return .{ .allocator = allocator, .s = mlx.mlx_default_gpu_stream_new(), .glm5 = gv };
        }
        if (config.isDsv41()) {
            const dv = try dsv41_vision.Dsv41Vision.init(allocator, config, weights);
            return .{ .allocator = allocator, .s = mlx.mlx_default_gpu_stream_new(), .dsv41 = dv };
        }
        return error.MissingVisionWeights;
    }

    pub fn deinit(self: *VisionEncoder) void {
        if (self.qwen) |*q| q.deinit();
        if (self.mimo) |*m| m.deinit();
        if (self.glm5) |*g| g.deinit();
        if (self.dsv41) |*d| d.deinit();
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Encode one image: `patches` is the processor's pixel_values [N, feat];
    /// `grid_h/grid_w` is the full patch grid. Returns [1, N/merge², out_hidden]
    /// (DeepSeek: the whole span, delimiters included).
    pub fn forwardPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.qwen) |*qv| return qv.forward(patches, grid_h, grid_w);
        if (self.mimo) |*mv| return mv.forward(patches, grid_h, grid_w);
        if (self.glm5) |*g| return g.forward(patches, grid_h, grid_w);
        if (self.dsv41) |*d| return d.forward(patches, grid_h, grid_w);
        return error.NoPatchGridEncoder;
    }

    /// Encode one VIDEO: `patches` holds `grid_t` temporal-patch groups'
    /// pixel_values concatenated (see `vision_common.buildPixelValuesVideo`).
    /// Only Qwen3-VL-family and GLM checkpoints have a video path.
    pub fn forwardVideoPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.qwen) |*qv| return qv.forwardVideo(patches, grid_t, grid_h, grid_w);
        if (self.glm5) |*g| return g.forwardVideo(patches, grid_t, grid_h, grid_w);
        return error.NoVideoEncoder;
    }
};
