const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const mtp_mod = @import("../mtp.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const model_mod = @import("../model.zig");
const model_registry_mod = @import("../model_registry.zig");
const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;
const Generator = generate_mod.Generator;
const expert_stream_mod = @import("../expert_stream.zig");
const testing = std.testing;
const scheduler_mod = @import("../scheduler.zig");
const mtp_tick = @import("../mtp_tick.zig");
const model_load = @import("../model_load.zig");
const Slot = scheduler_mod.Slot;
const anyPenalised = mtp_tick.anyPenalised;
const mimoCoarseHeadBytes = model_load.mimoCoarseHeadBytes;
const mimoVerifyWarmWanted = model_load.mimoVerifyWarmWanted;
const resolveExpertCache = model_load.resolveExpertCache;
const slotMimoMtpCrowdable = scheduler_mod.slotMimoMtpCrowdable;

test "a resident MiMo warms its verify rows without its heads; a streamed one and another arch do not" {
    var mimo = ModelConfig{};
    mimo.model_type = "mimo_v2";
    try testing.expect(mimoVerifyWarmWanted(&mimo));
    mimo.expert_streaming = true;
    try testing.expect(!mimoVerifyWarmWanted(&mimo));
    var qwen = ModelConfig{};
    qwen.model_type = "qwen4_exp";
    try testing.expect(!mimoVerifyWarmWanted(&qwen));
}

test "a MiMo source trunk bills one coarse lm_head copy, streamed or resident; another arch none" {
    var mimo = ModelConfig{ .model_type = "mimo_v2", .vocab_size = 152576, .hidden_size = 4096 };
    const want = mtp_mod.rerankCoarseBytes(152576, 4096, mimo_mtp.rerankBits());
    for ([_]@import("../expert_quant.zig").Layout{ .exl3_k4, .mxfp4_individual }) |layout| {
        mimo.expert_layout = layout;
        mimo.expert_streaming = false;
        try std.testing.expectEqual(want, mimoCoarseHeadBytes(&mimo));
        mimo.expert_streaming = true;
        try std.testing.expectEqual(want, mimoCoarseHeadBytes(&mimo));
    }
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .vocab_size = 248320, .hidden_size = 2560, .expert_layout = .exl3_k4 };
    try std.testing.expectEqual(@as(u64, 0), mimoCoarseHeadBytes(&qwen));
}

test "mimo_v2 expert cache budget excludes the dense prefix" {
    const per_expert: u64 = 1024 * 1024;
    const config = model_mod.ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 4,
        .first_k_dense_replace = 1,
        .num_experts = 8,
        .num_experts_per_tok = 2,
    };
    const fixed = per_expert + 8 * per_expert + 2 * per_expert + expert_stream_mod.BOUNCE_BYTES;
    const result = try resolveExpertCache(fixed + 9 * per_expert, &config, .{ .trunk = per_expert, .mtp = 0 }, false, per_expert);
    try std.testing.expectEqual(@as(u16, 3), result.ledger.?.slots_per_layer);
    try std.testing.expectEqual(9 * per_expert, result.cache_bytes);
}

test "MiMo crowded MTP takes an eligible slot" {
    var xfm: Transformer = undefined;
    xfm.config = .{ .model_type = "mimo_v2" };
    xfm.moe_layers = &.{};
    xfm.expert_stream = null;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = &xfm;
    var gen: Generator = undefined;
    gen.mtp = .{ .mimo = undefined };
    gen.mtp_cache = .{ .mimo = undefined };
    gen.has_last_hidden = true;
    gen.spec_disabled_runtime = false;
    gen.mtp_serial_left = 0;
    gen.mtp_serial_exit = .none;
    gen.dflash = null;
    var slot: Slot = undefined;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.sampling = .{};
    slot.logprobs_n = 0;
    try testing.expect(slotMimoMtpCrowdable(&slot));
}

test "a penalised MTP slot never joins a plain batched tick, which samples no penalty" {
    var xfm: Transformer = undefined;
    xfm.config = .{ .model_type = "mimo_v2" };
    xfm.moe_layers = &.{};
    xfm.expert_stream = null;
    var model: model_registry_mod.LoadedModel = undefined;
    model.transformer = &xfm;
    var gen: Generator = undefined;
    gen.mtp = .{ .mimo = undefined };
    gen.mtp_cache = .{ .mimo = undefined };
    gen.has_last_hidden = true;
    gen.spec_disabled_runtime = false;
    gen.mtp_serial_left = 0;
    gen.mtp_serial_exit = .none;
    gen.dflash = null;
    var slot: Slot = undefined;
    slot.model = &model;
    slot.legacy_gen = gen;
    slot.enable_mtp = true;
    slot.enable_drafter = false;
    slot.logprobs_n = 0;
    slot.sampling = .{ .presence_penalty = 1.5 };
    try testing.expect(!slotMimoMtpCrowdable(&slot));
    var plain: Slot = slot;
    plain.sampling = .{};
    try testing.expect(slotMimoMtpCrowdable(&plain));
    try testing.expect(anyPenalised(&.{ &plain, &slot }));
    try testing.expect(!anyPenalised(&.{&plain}));
}
