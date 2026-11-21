//! Flash-Next (`qwen4_exp`) as the registry sees it: its one public face.
const std = @import("std");
const qwen4_forward = @import("qwen4_forward.zig");
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
pub const State = qwen4_state.State;
pub const Ctx = qwen4_state.Ctx;

pub const packs = struct { // slot:packs (S2h)
    pub const dflash_target = true;
    pub const product = "Qwen3.8-Flash-Next";
    pub const aliases = [_]@import("../arch/packs.zig").Alias{
        .{ .name = "qwen3.8-flash-next", .tag = "2bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-2bpw", .tier_gb = 48, .recommend_gb = 32, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 131072 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "2.75bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .is_default = true, .tier_gb = 64, .recommend_gb = 64, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --qwen-gdn 16 --ctx-size 250000 --max-tokens 32000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
        .{ .name = "qwen3.8-flash-next", .tag = "4bpw", .repo = "beamster/Qwen3.8-Flash-Next-Sushi-4bpw", .tier_gb = 96, .recommend_gb = 96, .serve_flags = "--mtp --kv-quant 8 --mtp-head-kv-quant --ctx-size 500000 --prefill-chunk 2048 --max-tokens 64000 --prefix-cache-disk 20GB --prefix-cache-entries 1 --temp 1" },
    };
};
/// A streamed Sushi EXL3 pack keeps its MTP head, whose resident tensors carry the canonical `language_model.mtp.` keys.
pub const streamed_mtp_head = struct {
    pub const phrase = "a Sushi EXL3 Qwen3.8-Flash-Next pack";
    pub fn supported(layout: @import("../expert_quant.zig").Layout) bool {
        return layout == .exl3_k4;
    }
};

pub const flags = @import("qwen4_flags.zig"); // slot:flags (S2h)

pub const drafter = @import("qwen4_drafter.zig");

pub const load = @import("qwen4_load.zig");

pub const ubench = @import("ubench_qwen4.zig");

pub const sched = @import("qwen4_sched.zig"); // slot:sched (S2b)

pub const prefix = @import("qwen4_prefix.zig"); // slot:prefix (S2c)

pub const bill = @import("qwen4_bill.zig"); // slot:bill (S2d)

pub const kernels = struct {
    pub const batchedGatherOn = qwen4_qsa_gather.qsaBatchedGatherOn;
    pub const batchedGatherFloor = qwen4_qsa_gather.qsaBatchedGatherFloor;
    pub const scoreFusedFor = qwen4_qsa_select.qsaScoreFusedActiveFor;
    pub const warmEnvCaches = @import("qwen4_qsa_attn.zig").warmQsaEnvCaches;
    pub const diagArmed = @import("qwen4_profile_env.zig").profileArmed;

    pub fn sharedFoldEngaged() void {
        @import("../log.zig").info("[qwen] shared expert rides the routed launches engaged\n", .{});
    }

    pub inline fn moeStandin() @import("../arch/kernels.zig").MoeStandin {
        const s = qwen4_forward.qwen4Standin();
        return .{ .moe_shared = s.moe_shared, .moe_router = s.moe_router, .moe_gateup = s.moe_gateup, .moe_down = s.moe_down };
    }
};

const config_file = @import("qwen4_config.zig");
const arch_config = @import("../arch/config.zig");
pub const config_types = [_][]const u8{ "qwen4_exp", "qwen4_exp_text" };
pub const discovery_types = config_types;
pub const config_facts: arch_config.Facts = .{ .thinks_by_default = true, .hyper_connected = true, .chip_capped_drafter_block = false, .dflash_by_flag = true, .long_ctx_gated = true, .sushi_quant_memory_bill = true, .thinking_off_checkpoint = true, .skips_mtp_weights_when_off = true, .expert_source = .language_model_banks, .mtp_head_marker = "\"language_model.mtp.fc_hidden.weight\"", .streams_rate_groups = true, .mtp_draft_temp = 1.0, .cost_layout = .long, .reads_yarn = true };
pub const pack_layout: arch_config.PackLayout = .{ .prefix = "language_model.model", .head = .{ .layer = "language_model.mtp.layers.0" }, .streams_groups = true };
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
    var line_buf: [256]u8 = undefined;
    @import("../log.zig").info("{s}\n", .{@import("../cooler_side.zig").line(&line_buf, "[qwen4] forwards", &config)});
    const held = mlx.engine.isStream(s);
    if (config.indexer_budget != 0) {
        try qwen4_qsa_select.qsaScoreFusedArm(s);
        try qwen4_qsa_gather.qsaNaxArm(s);
    }
    var xfm = try Trunk.initTrunk(io, allocator, config, weights, s);
    if (held) qwen4_cooler.ownBuiltWeights(&xfm) catch |e| {
        xfm.arch_state.qwen4.deinit(allocator);
        xfm.deinitTrunk();
        return e;
    };
    return xfm;
}

pub fn deinit(xfm: Model) void {
    xfm.arch_state.qwen4.deinit(xfm.allocator);
    xfm.deinitTrunk();
}

const qwen4_cooler = @import("qwen4_cooler.zig");

/// What the shared Sushi Cooler contract suite asks of this arch (`src/cooler_contract_test.zig`).
pub const cooler_opt_out: @import("../arch/config.zig").CoolerOptOut = .{
    .flag = "--qwen-mlx-engine",
    .setting = "qwen_cooler",
};

pub const cooler_contract = @import("qwen4_contract.zig");

/// A Cooler load runs on the engine's stream.
pub const coolerGap = qwen4_cooler.coolerGap;

/// The pipeline manifest a Qwen load reads and writes (`arch/cooler_pipelines.zig`): kernels follow the pack's shapes
/// and formats, never its weights, so two packs alike in these share one.
/// The served warm's packed gather past the indexer's budget (`qwen4_qsa.warmPackedGather`), over the warm's cache.
pub const warmPackedGather = @import("qwen4_qsa.zig").warmPackedGather;

pub fn coolerPipelineScope(config: *const Config) [16]u8 {
    const manifest = @import("../pipeline_manifest.zig");
    var d: manifest.Digest = .{};
    d.add("model type", config.model_type);
    for ([_]struct { []const u8, u64 }{
        .{ "layers", config.num_hidden_layers },        .{ "hidden", config.hidden_size },
        .{ "heads", config.num_attention_heads },       .{ "vocab", config.vocab_size },
        .{ "experts", config.num_experts },             .{ "expert width", config.moe_intermediate_size },
        .{ "indexer budget", config.indexer_budget },   .{ "expert layout", @intFromEnum(config.expert_layout) },
        .{ "codebook", @intFromEnum(config.expert_quant_codebook) }, .{ "window", @intFromEnum(config.expert_quant_window) },
        .{ "table", config.expert_quant_table orelse 0xff }, .{ "rate", config.expert_quant_rate.n },
        .{ "rates", @truncate(config.expert_quant_rates orelse 0) }, .{ "tile order", @intFromEnum(config.expert_tile_order) },
        .{ "quant bits", config.quant_bits },           .{ "quant group", config.quant_group_size },
        .{ "trunk codebook", if (config.trunk_quant) |t| @intFromEnum(t.codebook) else 0xff },
    }) |f| d.addInt(f[0], f[1]);
    @import("../arch/cooler_pipelines.zig").addRecipeEnv(&d);
    return d.hex();
}

pub fn resetCache(xfm: Model) !void {
    return xfm.resetTrunk();
}

pub fn warmup(xfm: Model) !void {
    return xfm.warmupTrunk();
}

pub fn forward(xfm: Model, ctx: *trunk.ForwardCtx, ids: mlx.mlx_array) !mlx.mlx_array {
    // Only a batched GDN tick binds the persistent group, so no other arch has one to release.
    if (ctx.batch_slots == null) try qwen4_ssm_group.ssmGroupRelease(xfm, ctx);
    return qwen4_forward.forwardQwen4With(xfm, ctx, ids);
}

pub fn lmHeadForDraft(xfm: Model, x: mlx.mlx_array) !mlx.mlx_array {
    return xfm.lmHeadProject(x, false);
}

pub fn ownsModuleDecodeState(xfm: Model) bool {
    _ = xfm;
    return false;
}

/// A quantized pack drops experts by its usage list and its router can favour the cached ones; the BF16 checkpoint's
/// fused banks are read whole.
pub fn expertStream(config: *const Config) admission.ExpertStreamTraits {
    const bf16 = config.expert_layout == .bf16_fused or config.expert_layout == .bf16_individual;
    return .{ .drop_expert = !bf16, .stq_experts = !bf16, .cache_bias = !bf16 };
}

test "a model stamped as qwen4 is dispatched to qwen4" {
    const tag = registry.ofModelType(model_types[0]) orelse return;
    var xfm: Trunk = undefined;
    xfm.arch = tag;
    try std.testing.expectEqual(tag, std.meta.activeTag(registry.handleOf(&xfm)));
}

test "an unbound persistent group releases nothing, so a forward on another arch needs no release" {
    var group: trunk.PersistentSsmGroup = .{};
    var ctx: trunk.ForwardCtx = undefined;
    ctx.ssm_entries = &.{};
    try qwen4_ssm_group.ssmGroupReleaseCtx(&group, mlx.mlx_stream{ .ctx = null }, &ctx);
    try std.testing.expect(!group.live);
}
