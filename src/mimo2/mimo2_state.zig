//! MiMo-V2.6-Flash keeps nothing beyond the trunk except, on a streamed load, the expert engine.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const trunk = @import("../trunk.zig");
const Trunk = trunk.Trunk;

/// The expert engine of a streamed MiMo load, null for a resident one.
pub fn streamParts(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, s: mlx.mlx_stream) !?Trunk.ExpertStreamParts {
    if (!config.expert_streaming) return null;
    return try Trunk.initExpertStream(allocator, config, s);
}
