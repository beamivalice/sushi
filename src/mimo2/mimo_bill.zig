//! MiMo's own terms of the bill: the batched decode rows, the FP8 dequant scratch and the load warmup.
const std = @import("std");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const fp8_block = @import("../fp8_block.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const mtp_mod = @import("../mtp.zig");
const slot_admission = @import("../slot_admission.zig");
const mimo_forward = @import("mimo_forward.zig");



// Hooks: bills.zig reaches these through the arch's `bill` declaration; a hook returns null where it has no say.

/// The 2-bit lm_head copy a source trunk keeps, resident or streamed.
pub fn coarseHead(config: *const model_mod.ModelConfig) ?u64 {
    if (!config.usesMimoSourceTrunk() or mimo_mtp.rerankBits() == 0) return 0;
    return mtp_mod.rerankCoarseBytes(@intCast(config.vocab_size), @intCast(config.hidden_size), mimo_mtp.rerankBits());
}

/// The head's rings and catch-up rows.
pub fn headRings(config: *const model_mod.ModelConfig) ?u64 {
    return mimo_mtp.State.billedBytes(config);
}

/// What the other rows of a batched decode hold beside this slot's tick: each row rebuilds one global layer
/// dense below the packed arms' floor, where a solo tick's rebuild was alone. Zero where decode does not batch.
pub fn decodeRows(config: *const model_mod.ModelConfig) ?u64 {
    if (!config.supportsBatchedMimoDecode()) return 0;
    const keys = mimo_forward.mimoGlobalDecodeRebuildMaxKeys();
    var widest: u64 = 0;
    var li: u32 = 0;
    while (li < config.num_hidden_layers) : (li += 1) {
        if (config.isKvPerTokenLayer(li)) widest = @max(widest, config.layerKvBytes(li));
    }
    return @as(u64, slot_admission.batchGroupCap(config) - 1) *| keys *| widest;
}

/// The bf16 copy `fp8_block.project` dequantizes a source FP8 linear into when a forward is wider than its GEMV:
/// the widest linear's, and one more for the copy MLX's buffer cache still holds when the next linear dequantizes.
pub fn fp8Scratch(config: *const model_mod.ModelConfig, fwd: u64) ?u64 {
    if (!config.usesMimoSourceTrunk() or fwd <= @as(u64, @intCast(fp8_block.gemv_max_rows))) return 0;
    var widest: u64 = config.intermediate_size;
    for (0..config.num_hidden_layers) |i| {
        const l: u32 = @intCast(i);
        const qkv: u64 = @as(u64, config.layerNumHeads(l)) * config.layerHeadDim(l) +
            @as(u64, config.layerKVHeads(l)) * (config.layerHeadDim(l) + config.layerVHeadDim(l));
        widest = @max(widest, qkv);
    }
    return 2 *| widest *| config.hidden_size *| 2;
}

pub fn loadWarmup(config: *const model_mod.ModelConfig, mtp_on: bool) ?u64 {
    _ = config;
    _ = mtp_on;
    return (1 << 30) / 4;
}

