//! DeepSeek-V4.1-Flash (`deepseek_v41`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const dsv41_state = @import("dsv41_state.zig");
const serving = @import("dsv41_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;

pub const packs = .{}; // slot:packs (S2h)
pub const flags = .{}; // slot:flags (S2h)

pub const drafter = @import("dsv41_drafter.zig");

// slot:sched (S2b)

pub const prefix = @import("dsv41_prefix.zig"); // slot:prefix (S2c)

// slot:bill (S2d)

const config_file = @import("dsv41_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = model_types;
pub const discovery_types = model_types;
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true, .reads_pair_major = true };
pub const parseConfig = config_file.parseFamily;

pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    const source = @import("dsv41_source.zig");
    return .{
        .trunk = try source.residentBytes(io, allocator, model_dir),
        .mtp = if (config.dsv41.?.dspark.block > 0) try source.dsparkBytes(io, allocator, model_dir) else 0,
        .vision = if (config.has_vision) try source.visionBytes(io, allocator, model_dir, @import("../vision_common.zig").vision_bf16) else 0,
    };
}

pub const media = @import("dsv41_media.zig"); // slot:media (S2f)

pub const chat = @import("dsv41_chat.zig"); // slot:chat (S2g)

pub const kld = @import("kld_capture.zig");

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    _ = io;
    return serving.initDsv41(allocator, config, weights, s);
}

pub fn deinit(xfm: Model) void {
    dsv41_state.deinit(xfm);
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    dsv41_state.reset(xfm);
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
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

test "a model stamped as dsv41 is dispatched to dsv41" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
