//! MiMo-V2.6-Flash (`mimo_v2`) as the registry sees it: its one public face. It runs the generic MoE trunk, so it
//! is the registry's default arch (`default_arch`).
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = [_][]const u8{"mimo_v2"};
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;

/// A model type no arch claims runs the generic MoE trunk, which is this arch.
pub const default_arch = true;

pub const packs = .{}; // slot:packs (S2h)
pub const flags = .{}; // slot:flags (S2h)

// slot:drafter (S2a)

// slot:sched (S2b)

// slot:prefix (S2c)

// slot:bill (S2d)

// slot:config (S2e)

// slot:media (S2f)

// slot:chat (S2g)

pub const kld = @import("mimo2_kld.zig");
test {
    _ = @import("mimo2_kld.zig");
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    return Trunk.initTrunk(io, allocator, config, weights, s);
}

pub fn deinit(xfm: Model) void {
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
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

test "a model stamped as mimo2 is dispatched to mimo2" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}

test "a MiMo config leaves the QSA budget zero, so the QSA arming is qwen4's alone" {
    var config = try @import("../model.zig").parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/model-configs/mimo_v2.json"));
    defer config.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), config.indexer_budget);
}
