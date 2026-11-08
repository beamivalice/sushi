//! MiMo-V2.6-Flash (`mimo_v2`) as the registry sees it: its one public face. It runs the generic MoE trunk, so it
//! is the registry's default arch (`default_arch`).
const std = @import("std");
const mimo_forward = @import("mimo_forward.zig");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;
pub const State = @import("mimo2_state.zig").State;
pub const Ctx = @import("mimo2_state.zig").Ctx;

/// A model type no arch claims runs the generic MoE trunk, which is this arch.
pub const default_arch = true;

pub const packs = struct { // slot:packs (S2h)
    pub const product = "MiMo-V2.6-Flash";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "mimo-v2.6-flash", .tag = "2.3bpw", .repo = "beamster/MiMo-V2.6-Flash-Sushi-2.3bpw", .is_default = true, .tier_gb = 128, .recommend_gb = 128, .serve_flags = "--ctx-size 1048576" },
    };
};
pub const flags = @import("mimo2_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("mimo2_drafter.zig");

pub const sched = @import("mimo2_sched.zig"); // slot:sched (S2b)

pub const prefix = @import("mimo2_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("mimo_bill.zig"); // slot:bill (S2d)

pub const load = @import("mimo2_load.zig");

/// The arch's microbenches, which `sushi-ubench` lists; nothing else reads them.
pub const ubench = @import("ubench.zig");

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
    .residual_from_embedding = true,
    .moe_scores_f32 = true,
    .qkv_split_rows = true,
    .bias_routed_moe = true,
};
pub const pack_layout: arch_config.PackLayout = .{ .prefix = "model", .stamped = true, .head = .none };
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

/// MiMo routes in f32: its router reads f32 activations and weights, and a layer's selection bias picks the experts
/// (sigmoid, noaux_tc) while the unbiased scores weight them.
pub const moe = struct {
    pub const routerLogits = mimo_forward.mimoRouterLogits;
    pub const biasRouting = mimo_forward.computeMimoRouting;

    /// A source trunk's dense router is widened once at load, which `mimo_source.countResidentBytes` bills.
    pub fn widensRouter(config: *const Config) bool {
        return config.usesMimoSourceTrunk();
    }
};

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
    xfm.arch_state.mimo2.deinit(xfm.allocator);
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return mimo_forward.forwardMoeWith(xfm, ctx, ids);
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
    var config = try @import("../model.zig").parseConfigFromJson(std.testing.allocator, chat.fixtures.config);
    defer config.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u32, 0), config.indexer_budget);
}

test "a MiMo source trunk widens its dense router at load; a converted MiMo pack and another arch's config do not" {
    const arch_moe = @import("../arch/moe.zig");
    try std.testing.expect(arch_moe.widensRouter(&.{ .model_type = "mimo_v2", .expert_layout = .mxfp4_individual }));
    try std.testing.expect(arch_moe.widensRouter(&.{ .model_type = "mimo_v2", .expert_layout = .exl3_k4 }));
    try std.testing.expect(!arch_moe.widensRouter(&.{ .model_type = "mimo_v2" }));
    try std.testing.expect(!arch_moe.widensRouter(&.{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4 }));
}

test "a MiMo config states the trunk rules MiMo keeps, and another arch's config states none of them" {
    const mimo: Config = .{ .model_type = model_types[0] };
    const other: Config = .{ .model_type = "facts-test" };
    inline for (.{ "residual_from_embedding", "moe_scores_f32", "qkv_split_rows", "bias_routed_moe" }) |name| {
        try std.testing.expect(arch_config.fact(&mimo, name));
        try std.testing.expect(!arch_config.fact(&other, name));
    }
}
