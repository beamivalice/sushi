//! DeepSeek-V4.1-Flash (`deepseek_v41`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const serving = @import("dsv41_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;
pub const State = dsv41_state.State;

pub const packs = struct { // slot:packs (S2h)
    pub const product = "DeepSeek-V4.1-Flash";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "deepseek-v4.1-flash", .tag = "1.75bpw", .repo = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", .is_default = true, .tier_gb = 96, .recommend_gb = 96, .streams_experts = true },
    };
};
pub const flags = @import("dsv41_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("dsv41_drafter.zig");

pub const sched = @import("dsv41_sched.zig"); // slot:sched (S2b)

pub const prefix = @import("dsv41_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("dsv41_bill.zig"); // slot:bill (S2d)

const config_file = @import("dsv41_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = model_types;
pub const discovery_types = model_types;
/// Its DSpark stages ride a streamed load on both of its layouts.
pub const streamed_mtp_head = struct {
    pub const phrase = "DeepSeek-V4.1-Flash";
    pub fn supported(layout: @import("../expert_quant.zig").Layout) bool {
        return layout == .mxfp4_w123 or layout == .exl3_k4;
    }
};
pub const config_facts: arch_config.Facts = .{
    .thinks_by_default = true,
    .reads_pair_major = true,
    .prefill_start_width = config_file.prefill_width,
    .expert_pool_bytes = config_file.dsv41_pool_bytes,
    .streams_experts = true,
    .renders_own_chat_template = true,
    .expert_source = .mxfp4_w123,
    .state_beside_kv = true,
    .no_mtp_head = true,
    .vision_on_streamed = true,
    .own_kv_state = true,
    .scores_picked_keys = true,
    .fills_slabs_in_place = true,
    .streams_rate_groups = true,
    .reads_own_source = true,
};
pub const OwnConfig = config_file.Config;
pub const serveDefaults = config_file.Config.serveDefaults;

pub fn releaseOwn() !OwnConfig {
    return config_file.parseText(config_file.release_config);
}

pub fn kvBytesPerToken(config: *const Config) ?u64 {
    const c = config.dsv41 orelse return null;
    return c.stateBytesPerToken();
}

pub fn cacheLayoutNamespace(config: *const Config) ?[]const u8 {
    return dsv41_prefix.namespaceTag(dsparkOn(config));
}

/// Records who asked for decode on MLX, which `dsv41_borrow.decodeSideOf` and the bills read.
pub fn stampCooler(config: *Config, on: bool, by_flag: bool) void {
    if (config.dsv41) |*c| c.mlx_ask = if (on) .none else if (by_flag) .flag else .setting;
}

/// Does this load keep the DSpark stages resident? They are its MTP head: `--mtp` under streaming.
pub fn dsparkOn(config: *const Config) bool {
    const c = config.dsv41 orelse return false;
    return config.stream_mtp_head and c.dspark.block > 0;
}

pub const load = @import("dsv41_load.zig");
pub const parseConfig = config_file.parseFamily;

pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    const source = @import("dsv41_source.zig");
    return .{
        .trunk = try source.residentBytes(io, allocator, model_dir),
        .mtp = if (config.dsv41.?.dspark.block > 0) try source.dsparkBytes(io, allocator, model_dir) else 0,
        .vision = if (config.has_vision) try source.visionBytes(io, allocator, model_dir, @import("../vision_common.zig").vision_bf16) else 0,
    };
}

pub fn streamedResidentExtra(config: *const Config) u64 {
    const c = config.dsv41 orelse return 0;
    return @import("dsv41_engram.zig").cacheBytes(c.engram);
}

pub const media = @import("dsv41_media.zig"); // slot:media (S2f)

pub const chat = @import("dsv41_chat.zig"); // slot:chat (S2g)

pub const kld = @import("kld_capture.zig");

/// The arch's microbenches, which `sushi-ubench` lists; nothing else reads them.
pub const ubench = @import("ubench.zig");

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    _ = io;
    return serving.initDsv41(allocator, config, weights, s);
}

/// A Cooler load runs on the engine's stream.
pub const ownStream = dsv41_prefix.arraysOn;

pub fn deinit(xfm: Model) void {
    const held = if (xfm.arch_state.dsv41.model) |m| m.engine_decode else false;
    xfm.arch_state.dsv41.deinit(xfm.allocator);
    xfm.deinitTrunk();
    if (held) mlx.engine.unhold();
}

pub fn resetCache(xfm: Model) !void {
    xfm.arch_state.dsv41.reset();
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    return serving.forwardDsv41With(xfm, ctx, ids, xfm.arch_state.dsv41.model.?);
}

// DeepSeek drafts with DSpark from its own stages and loads no DFlash drafter, so nothing projects through the shell's
// empty head.
pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    _ = .{ xfm, x };
    return error.DraftHeadUnsupported;
}

pub fn ownsModuleDecodeState(xfm: Model) bool {
    _ = xfm;
    return true;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    return .{ .drop_expert = true, .prefetch = true, .cooler = true, .tile_major = config.expert_layout == .exl3_k4, .stq_experts = if (config.dsv41) |c| c.expert_dec != null else false };
}

test "a model stamped as dsv41 is dispatched to dsv41" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
