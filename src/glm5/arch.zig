//! GLM-5.3-Flash (`glm5_next`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const glm5_state = @import("glm5_state.zig");
const serving = @import("glm5_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;

pub const packs = .{}; // slot:packs (S2h)
pub const flags = .{}; // slot:flags (S2h)

pub const drafter = @import("glm5_drafter.zig");

// slot:sched (S2b)

pub const prefix = @import("glm5_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("glm5_bill.zig"); // slot:bill (S2d)

const config_file = @import("glm5_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = [_][]const u8{ "glm5_next", "glm5_next_text" };
pub const discovery_types = model_types;
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true };
pub const parseConfig = config_file.parseFamily;

pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    return try @import("glm5_diagnostic.zig").streamedSplit(io, allocator, model_dir, config.num_hidden_layers);
}

pub const media = @import("glm5_media.zig"); // slot:media (S2f)

pub const chat = @import("glm5_chat.zig"); // slot:chat (S2g)

pub const kld = @import("glm5_kld_capture.zig");

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    _ = io;
    return serving.initGlm5(allocator, config, weights, s);
}

pub fn deinit(xfm: Model) void {
    glm5_state.deinit(xfm);
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    glm5_state.reset(xfm);
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
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

test "a model stamped as glm5 is dispatched to glm5" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
