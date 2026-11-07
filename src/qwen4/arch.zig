//! Flash-Next (`qwen4_exp`) as the registry sees it: its one public face.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const qwen4_state = @import("qwen4_state.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const qwen4_ssm_group = @import("qwen4_ssm_group.zig");
const admission = @import("../arch/admission.zig");
const registry = @import("../arch/registry.zig");
const Trunk = trunk.Trunk;

pub const model_types = chat.model_types;
pub const Config = @import("../model.zig").ModelConfig;
pub const Model = *Trunk;

pub const packs = struct { // slot:packs (S2h)
    pub const product = "Qwen3.8-Flash-Next";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "qwen3.8-flash-next", .tag = "2bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-2bpw", .tier_gb = 48, .recommend_gb = 32, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 131072 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "2.75bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .is_default = true, .tier_gb = 64, .recommend_gb = 64, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 250000 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "4bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-4bpw", .tier_gb = 96, .recommend_gb = 96, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 --prefill-chunk 2048 --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
    };
};
pub const flags = @import("qwen4_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("qwen4_drafter.zig");

pub const sched = @import("qwen4_sched.zig"); // slot:sched (S2b)

pub const prefix = @import("qwen4_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("qwen4_bill.zig"); // slot:bill (S2d)

const config_file = @import("qwen4_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = [_][]const u8{ "qwen4_exp", "qwen4_exp_text" };
pub const discovery_types = config_types;
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true };
pub const parseConfig = config_file.parseFamily;
pub const parseDir = config_file.parseDir;
pub const samplingDefaults = config_file.samplingDefaults;

pub const media = @import("qwen4_media.zig"); // slot:media (S2f)

pub const chat = @import("qwen4_chat.zig"); // slot:chat (S2g)

pub const kld = @import("qwen4_kld.zig");
test {
    _ = @import("qwen4_kld.zig");
}

pub fn init(io: std.Io, allocator: std.mem.Allocator, config: Config, weights: *trunk.Weights, s: mlx.mlx_stream) !Trunk {
    if (config.indexer_budget != 0) {
        qwen4_qsa_select.qsaScoreFusedArm();
        qwen4_qsa_gather.qsaNaxArm();
    }
    return Trunk.initTrunk(io, allocator, config, weights, s);
}

pub fn deinit(xfm: Model) void {
    qwen4_state.deinit(xfm);
    xfm.deinitTrunk();
}

pub fn resetCache(xfm: Model) !void {
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    // Only a batched GDN tick binds the persistent group, so no other arch has one to release.
    if (ctx.batch_slots == null) try xfm.ssmGroupRelease(ctx);
    return xfm.forwardQwen4With(ctx, ids);
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

test "a model stamped as qwen4 is dispatched to qwen4" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}

test "an unbound persistent group releases nothing, so a forward on another arch needs no release" {
    var group: @TypeOf(@as(trunk.Trunk, undefined).ssm_group) = .{};
    var ctx: trunk.ForwardCtx = undefined;
    ctx.ssm_entries = &.{};
    try qwen4_ssm_group.ssmGroupReleaseCtx(&group, mlx.mlx_stream{ .ctx = null }, &ctx);
    try std.testing.expect(!group.live);
}
