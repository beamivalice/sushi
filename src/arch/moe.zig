//! The MoE family's arch hooks: where an arch's router or routing departs from the trunk's softmax top-k. An arch
//! declares them in a `moe` struct in its `arch.zig`. They are asked by the config's arch, so a trunk built by hand
//! needs only its `model_type`.
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const trunk = @import("../trunk.zig");
const registry = @import("registry.zig");
const arch_config = @import("config.zig");
const Trunk = trunk.Trunk;

fn has(comptime t: registry.Tag, comptime name: []const u8) bool {
    const M = registry.Mod(t);
    return @hasDecl(M, "moe") and @hasDecl(M.moe, name);
}

/// The router logits of an arch that projects them its own way; null takes the router's own projection.
pub fn routerLogits(t: *Trunk, x: mlx.mlx_array, mw: *const trunk.MoeMlpWeights) !?mlx.mlx_array {
    const tag = arch_config.tagOf(&t.config) orelse return null;
    return switch (tag) {
        inline else => |g| if (comptime has(g, "routerLogits")) try registry.Mod(g).moe.routerLogits(t, x, mw) else null,
    };
}

/// The routing of a layer that carries a selection bias (`MoeMlpWeights.expert_bias`): only the arch that loads one
/// defines it.
pub fn biasRouting(t: *const Trunk, logits: mlx.mlx_array, bias: mlx.mlx_array) !Trunk.MoeRouting {
    const tag = arch_config.tagOf(&t.config) orelse return error.BiasRoutingUnsupported;
    return switch (tag) {
        inline else => |g| if (comptime has(g, "biasRouting")) registry.Mod(g).moe.biasRouting(t, logits, bias) else error.BiasRoutingUnsupported,
    };
}

/// Whether the loader widens a dense router to f32 once, for an arch that routes in f32 on this config.
pub fn widensRouter(config: *const model.ModelConfig) bool {
    const tag = arch_config.tagOf(config) orelse return false;
    return switch (tag) {
        inline else => |g| if (comptime has(g, "widensRouter")) registry.Mod(g).moe.widensRouter(config) else false,
    };
}
