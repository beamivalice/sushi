//! MiMo-V2.6-Flash (`mimo_v2`) as the registry sees it: its one public face. It runs the generic MoE trunk, so it
//! is the registry's default arch and has no `loaded`.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trf = @import("../transformer.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Transformer = trf.Transformer;

pub const model_types = [_][]const u8{"mimo_v2"};
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Transformer;

pub const packs = .{}; // slot:packs (S2h)
pub const flags = .{}; // slot:flags (S2h)

// slot:drafter (S2a)

// slot:sched (S2b)

// slot:prefix (S2c)

// slot:bill (S2d)

// slot:config (S2e)

// slot:media (S2f)

// slot:chat (S2g)

// slot:offline (S2i)

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trf.Weights, s: mlx.mlx_stream) !Transformer {
    return Transformer.initTrunk(io, allocator, config, weights, s);
}

pub fn forward(xfm: Model, ctx: *trf.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return xfm.forwardMoeWith(ctx, ids);
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    return xfm.lmHeadProject(x, false);
}

pub fn ownsModuleDecodeState(xfm: Model) bool {
    _ = xfm;
    return false;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    _ = config;
    return .{};
}

test "a model loaded as mimo2 is dispatched to mimo2" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Transformer = undefined;
    xfm.glm5 = null;
    xfm.dsv41 = null;
    xfm.qwen4 = null;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
