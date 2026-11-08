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
pub const State = glm5_state.State;
pub const Ctx = glm5_state.Ctx;

pub const packs = struct { // slot:packs (S2h)
    pub const dflash_target = true;
    pub const product = "GLM-5.3-Flash";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "glm-5.3-flash", .tag = "2bpw", .repo = "beamster/GLM-5.3-Flash-Sushi-2bpw", .tier_gb = 96, .recommend_gb = 96 },
        .{ .name = "glm-5.3-flash", .tag = "2.4bpw", .repo = "beamster/GLM-5.3-Flash-Sushi-2.4bpw", .is_default = true, .tier_gb = 128, .recommend_gb = 128 },
    };
};
pub const flags = @import("glm5_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("glm5_drafter.zig");

pub const load = @import("glm5_load.zig");

/// The offline tools this folder builds, by their `-D` option (`build_roots.zon`); `src/tool_root.zig` runs one.
pub const ubench = @import("ubench_glm5.zig");

pub const tools = struct {
    pub const @"glm-ffn-replay" = @import("glm5_ffn_replay.zig");
    pub const @"glm-trunk-cov" = @import("glm5_trunk_cov.zig");
};

pub const sched = @import("glm5_sched.zig"); // slot:sched (S2b)

pub const prefix = @import("glm5_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("glm5_bill.zig"); // slot:bill (S2d)

const config_file = @import("glm5_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = [_][]const u8{ "glm5_next", "glm5_next_text" };
pub const discovery_types = model_types;
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true, .expert_source = .individual_source, .state_beside_kv = true, .no_mtp_head = true, .mtp_unserved = true, .latent_kv = true, .kda_state = true, .evicts_hot_cache = true, .declared_wired_ceiling = true, .own_dflash_bill = true, .prefill_start_width = @import("glm5_forward.zig").prefill_chunk };

/// A pack stores the MTP layers after the trunk's, under the same prefix (`pack_index`'s range check).
pub fn trailingPackLayers(config: *const Config) u32 {
    return config.mtp_layers;
}
pub const pack_layout: arch_config.PackLayout = .{ .prefix = "model.language_model", .stamped = true, .head = .trailing, .refuses_rate_groups = true };
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
    xfm.arch_state.glm5.deinit(xfm.allocator);
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    xfm.arch_state.glm5.reset();
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return serving.forwardGlm5WithImpl(xfm, ctx, ids, xfm.arch_state.glm5.model.?);
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    return xfm.arch_state.glm5.model.?.projectHead(x);
}

// A streamed GLM load (the KLD teacher) admits one claimant per expert stream.
pub fn ownsModuleDecodeState(xfm: Model) bool {
    return xfm.arch_state.glm5.model.?.expert_stream != null;
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
