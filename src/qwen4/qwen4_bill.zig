//! Flash-Next's own terms of the bill: the MTP head's KV, ring and state, and the thought-point checkpoint.
const std = @import("std");
const qwen4_forward = @import("qwen4_forward.zig");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const Transformer = trunk_mod.Transformer;
const trunk_mod = @import("../trunk.zig");

// Hooks: bills.zig reaches these through the arch's `bill` declaration; a hook returns null where it has no say.

/// Per-token bytes of the MTP head's own KV at the scheme it was LOADED with, billed once per request when MTP is on.
pub fn headKvPerToken(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    const head_cfg = qwen4_forward.qwen4MtpHeadKvConfig(bills.configuredKvQuantFor(config));
    const bits: u64 = if (head_cfg.scheme == .off) 16 else head_cfg.bits;
    return bills.kvBytesPerTokenAtBits(config.kvBytesPerToken() / n, bits);
}

pub fn headQsaRing(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return config.qsaRingBytes() / n;
}

pub fn headStatePerToken(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return bills.statePerTokenBilled(config) / n;
}

/// The checkpoint state a thinking request holds past its prefill's: the restore point behind its thought's closer, taken
/// mid-decode and held to the commit (`generate.ThoughtPoint`).
pub fn thoughtPoint(config: *const model_mod.ModelConfig, warm: bills.WarmPrefix) ?u64 {
    if (!warm.thought or bills.launch_state.ssm_checkpoint_stride == 0) return 0;
    return config.ssmCheckpointBytes();
}

/// The most checkpoint states a request holds at once when its model runs an SSD tier: the write-through
/// frees each one once its file is written, and a request it cannot serve is thinned to the same number
/// (`scheduler.qwenCheckpointCap`). Null = every checkpoint stays in RAM.
pub fn residentCheckpointCap(config: *const model_mod.ModelConfig) ?u64 {
    if (!bills.diskTierOn(config)) return null;
    return trunk_mod.SSM_RESIDENT_MAX;
}

