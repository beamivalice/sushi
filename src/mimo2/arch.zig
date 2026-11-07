//! MiMo-V2.6-Flash (`mimo_v2`) as the registry sees it: its one public face. It runs the generic MoE trunk, so it
//! is the registry's default arch (`default_arch`).
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;

/// A model type no arch claims runs the generic MoE trunk, which is this arch.
pub const default_arch = true;

pub const packs = .{}; // slot:packs (S2h)
pub const flags = .{}; // slot:flags (S2h)

pub const drafter = @import("mimo2_drafter.zig");

// slot:sched (S2b)

pub const prefix = @import("mimo2_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("mimo_bill.zig"); // slot:bill (S2d)

const config_file = @import("mimo_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = model_types;
pub const discovery_types = model_types;
pub const config_facts: arch_config.Facts = .{
    // The vendor template's own default is on: only an explicit `enable_thinking is false` closes the think block.
    .thinks_by_default = true,
    .stub_first_moe = 1,
    .dense_prefix_in_layer_freq = true,
    .serves_video = false,
};
pub const parseConfig = config_file.parseFamily;

pub fn adoptLayout(config: *Config) void {
    // The MiMo binder reads the source QKV itself; the generic fused-QKV names never apply to it.
    if (config.usesMimoSourceTrunk()) config.attn_fused_qkv = false;
}

pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    if (!config.usesMimoSourceTrunk()) return null;
    var streamed = config.*;
    streamed.expert_streaming = true;
    const source = @import("mimo_source.zig");
    return .{
        .trunk = try source.residentBytesWithConfig(io, allocator, model_dir, &streamed),
        .mtp = 0,
        .vision = try source.visionResidentBytes(io, allocator, model_dir),
    };
}

pub const media = @import("mimo2_media.zig"); // slot:media (S2f)

pub const chat = @import("mimo_chat.zig"); // slot:chat (S2g)

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
