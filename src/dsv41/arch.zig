//! DeepSeek-V4.1-Flash (`deepseek_v41`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trf = @import("../transformer.zig");
const serving = @import("dsv41_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Transformer = trf.Transformer;

pub const model_types = [_][]const u8{"deepseek_v41"};
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
    return xfm.dsv41 != null;
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trf.Weights, s: mlx.mlx_stream) !Transformer {
    _ = io;
    return serving.initDsv41(allocator, config, weights, s);
}

pub fn forward(xfm: Model, ctx: *trf.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return serving.forwardDsv41With(xfm, ctx, ids, xfm.dsv41.?);
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    return xfm.lmHeadProject(x, false);
}

pub fn ownsModuleDecodeState(xfm: Model) bool {
    _ = xfm;
    return true;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    return .{ .drop_expert = true, .tile_major = config.expert_layout == .exl3_k4 };
}

test "a model loaded as dsv41 is dispatched to dsv41" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Transformer = undefined;
    xfm.glm5 = null;
    xfm.dsv41 = @ptrFromInt(@alignOf(@TypeOf(xfm.dsv41.?.*)));
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
