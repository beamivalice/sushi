//! DeepSeek-V4.1-Flash (`deepseek_v41`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const trunk = @import("../trunk.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const cooler_side = @import("../cooler_side.zig");
const serving = @import("dsv41_serving.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;
pub const State = dsv41_state.State;
pub const Ctx = dsv41_state.Ctx;

pub const packs = struct { // slot:packs (S2h)
    pub const product = "DeepSeek-V4.1-Flash";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "deepseek-v4.1-flash", .tag = "1.75bpw", .repo = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", .is_default = true, .tier_gb = 96, .recommend_gb = 96, .streams_experts = true },
        .{ .name = "deepseek-v4.1-flash", .tag = "2.7bpw", .repo = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-2.7bpw", .tier_gb = 64, .recommend_gb = 256, .streams_experts = true },
    };
};
pub const flags = @import("dsv41_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("dsv41_drafter.zig");

/// Its forwards install their own caps beside the expert pools they release (`dsv41_borrow`), and a request's end leaves
/// the engine the decode pool the ledger parks (`arch/cooler.zig`).
pub const cooler_own_caps = true;
pub fn coolerRestCap(config: *const Config) u64 {
    return config.expertPoolBytes();
}
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
/// The pool parked beside the model between requests: MLX's decode pool, or on a Cooler load the engine's free
/// buffers under the cap a request's end leaves it (`dsv41_borrow.endRequest`). MLX's pool is empty under such a load.
pub fn expertPoolBytes(config: *const Config) u64 {
    const c = config.dsv41 orelse return config_file.dsv41_pool_bytes;
    const borrow = @import("dsv41_borrow.zig");
    return if (cooler_side.onEngine(config)) borrow.ownCap(&c, true, 1) else config_file.dsv41_pool_bytes;
}

/// An STQ pack on Sushi Cooler keeps the scales of the experts it has read (`dsv41_borrow.keepsScales`).
pub fn expertKeepsScales(config: *const Config) bool {
    if (config.dsv41 == null) return false;
    return @import("dsv41_borrow.zig").keepsScales(config);
}

/// Where a prompt's last forward starts (`dsv41_prefix.snapshotBackoff`); `tok` names the assistant opener.
pub fn snapshotBackoff(tok: anytype, prompt_ids: []const u32) ?usize {
    return dsv41_prefix.snapshotBackoff(tok.specialTokenId("<｜Assistant｜>") orelse return null, prompt_ids);
}

pub const config_facts: arch_config.Facts = .{
    .thinks_by_default = true,
    .ssd_tables = &.{".engram.embed."},
    .reads_pair_major = true,
    .prefill_start_width_of = config_file.prefillWidthFor,
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
/// Its STQ packs' routed experts, trunk and DSpark stages, take the pair-major layout (`sushi repack`).
pub const repack: arch_config.Repack = .{};
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
    return dsv41_prefix.namespaceTag(drafterOn(config));
}

pub const cooler: @import("../arch/config.zig").Cooler = .{ .opt_out = .{
    .flag = "--ds41-mlx-engine",
    .setting = "ds41_cooler",
} };

/// What the shared Sushi Cooler contract suite asks of this arch (`src/cooler_contract_test.zig`).
pub const cooler_contract = @import("dsv41_contract.zig");

/// Does this load keep the DSpark stages resident? They are its MTP head: `--mtp` under streaming.
pub fn drafterOn(config: *const Config) bool {
    const c = config.dsv41 orelse return false;
    return config.stream_mtp_head and c.dspark.block > 0;
}

pub const load = @import("dsv41_load.zig");

/// The pack's widest rate-grouped layer, which the forward cap's routing worst case reads.
pub fn adoptLayout(config: *Config) void {
    if (config.dsv41) |*c| c.rate_groups = config.expert_rate_groups;
}

/// The pools a routed chain makes follow the expert plan: a resident plan holds every expert in the cache, so no route spills.
pub fn adoptPlan(config: *Config) void {
    if (config.dsv41) |*c| c.spill_pools = !config.expert_resident;
}

/// The wide forward's arms the load recorded (`dsv41_model.setFused`), which admission prices.
pub fn adoptLoaded(config: *Config, loaded: *const Config) void {
    if (config.dsv41) |*c| {
        c.wide_arms = loaded.dsv41.?.wide_arms;
        c.spill_pools = loaded.dsv41.?.spill_pools;
    }
}
pub const parseConfig = config_file.parseFamily;

pub fn streamedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const Config) !?@import("../model.zig").ResidentSplit {
    const source = @import("dsv41_source.zig");
    const engine = cooler_side.onEngine(config);
    return try source.streamedSplit(
        io,
        allocator,
        model_dir,
        config.dsv41.?.dspark.block > 0,
        config.has_vision,
        @import("../vision_common.zig").vision_bf16,
        engine,
    );
}

pub fn residentExtra(config: *const Config) u64 {
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
/// Every DeepSeek pack runs on Sushi Cooler, STQ or the release's own FP8 trunk and MXFP4 experts, on any GPU (the
/// arms a GPU has are `device`'s): only the user's opt-out keeps one on MLX.
pub fn coolerGap(_: *const Config) ?[]const u8 {
    return null;
}

pub fn deinit(xfm: Model) void {
    const held = if (xfm.arch_state.dsv41.model) |m| m.onEngine() else false;
    xfm.arch_state.dsv41.deinit(xfm.allocator);
    xfm.deinitTrunk();
    if (!held) return;
    const breaks = mlx.rest_breaks.load(.monotonic);
    const swallows = mlx.rest_swallows.load(.monotonic);
    if (breaks != 0 or swallows != 0) log.warn("[engine] while MLX was at rest it was called {d} times and asked {d} settings that reached nothing (process totals): work outside its model's turn\n", .{ breaks, swallows });
}

pub fn resetCache(xfm: Model) !void {
    xfm.arch_state.dsv41.reset();
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    // Runs on a warm manifest too: its first forward takes the expert routes' first cost, which a skip moves into the
    // first request.
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

/// One request at a time, unless the load joins rows: each request then runs a decode state of its own.
pub fn ownsModuleDecodeState(xfm: Model) bool {
    const m = dsv41_state.modelOf(xfm) orelse return true;
    return !m.serves_rows;
}

pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    return .{ .drop_expert = true, .prefetch = true, .cache_bias = true, .tile_major = config.expert_layout == .exl3_k4, .stq_experts = if (config.dsv41) |c| c.expert_dec != null else false };
}

test "a model stamped as dsv41 is dispatched to dsv41" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}
