const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const arch_media = @import("arch/media.zig");

const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;

/// The resident image tower of a served model: exactly one patch-grid ViT of the arch that claims the config,
/// owning its own weights and GPU stream.
pub const VisionEncoder = struct {
    allocator: std.mem.Allocator,
    /// Stream the patch arrays are made for and the per-image embeddings are joined on: the tower's own when it
    /// brings one.
    s: mlx.mlx_stream,
    tower: ?arch_media.Tower = null,

    pub fn init(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights) !VisionEncoder {
        const tower = try arch_media.Tower.init(allocator, config, weights);
        return .{ .allocator = allocator, .s = tower.ownStream() orelse mlx.mlx_default_gpu_stream_new(), .tower = tower };
    }

    pub fn deinit(self: *VisionEncoder) void {
        if (self.tower) |*t| t.deinit();
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Encode one image: `patches` is the processor's pixel_values [N, feat];
    /// `grid_h/grid_w` is the full patch grid. Returns [1, N/merge², out_hidden]
    /// (DeepSeek: the whole span, delimiters included).
    pub fn forwardPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.tower) |*t| return t.forward(patches, grid_h, grid_w);
        return error.NoPatchGridEncoder;
    }

    /// Encode one VIDEO: `patches` holds `grid_t` temporal-patch groups'
    /// pixel_values concatenated (see `vision_common.buildPixelValuesVideo`).
    /// Only some towers have a video path.
    pub fn forwardVideoPatches(self: *VisionEncoder, patches: mlx.mlx_array, grid_t: u32, grid_h: u32, grid_w: u32) !mlx.mlx_array {
        if (self.tower) |*t| return t.forwardVideo(patches, grid_t, grid_h, grid_w);
        return error.NoVideoEncoder;
    }
};
