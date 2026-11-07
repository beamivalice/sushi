//! GLM-5.3-Flash (`glm5_next`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trf = @import("../transformer.zig");
const serving = @import("glm5_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Transformer = trf.Transformer;

pub const model_types = [_][]const u8{"glm5_next"};
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

pub fn loaded(xfm: Model) bool {
    return xfm.glm5 != null;
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trf.Weights, s: mlx.mlx_stream) !Transformer {
    _ = io;
    return serving.initGlm5(allocator, config, weights, s);
}

pub fn forward(xfm: Model, ctx: *trf.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return serving.forwardGlm5WithImpl(xfm, ctx, ids, xfm.glm5.?);
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    return xfm.glm5.?.projectHead(x);
}

// A streamed GLM load (the KLD teacher) admits one claimant per expert stream.
pub fn ownsModuleDecodeState(xfm: Model) bool {
    return xfm.glm5.?.expert_stream != null;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    _ = config;
    return .{};
}

test "a model loaded as glm5 is dispatched to glm5" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Transformer = undefined;
    xfm.glm5 = @ptrFromInt(@alignOf(@TypeOf(xfm.glm5.?.*)));
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
