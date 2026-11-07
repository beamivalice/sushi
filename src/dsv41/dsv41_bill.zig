//! DeepSeek-V4.1's own terms of the bill: the compressed state, the restore points and the forward cap.
const std = @import("std");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_config = @import("dsv41_config.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const generate_mod = @import("../generate.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const prefill_plan = @import("../prefill_plan.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");

/// The restore points a DeepSeek prefill holds in RAM while it runs: with a RAM tier the capped list plus the copy
/// taken before a thin, each a full copy of the state at most `seq` tokens deep; SSD-first only what is in flight to
/// the tier. Zero with no prefix cache at all.
pub fn checkpointReserve(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    const c = config.dsv41 orelse return 0;
    const base = bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config));
    const stride = prefill_plan.effectiveSsmCheckpointStride(base, @max(@as(usize, @intCast(chunk)), generate_mod.prefill_chunk_override));
    if (stride == 0) return 0;
    const one = c.stateFixedBytes(config.dsparkOn()) +| seq *| c.stateBytesPerToken();
    // SSD-first, a point the tier takes shares the request's rows and leaves RAM as the writer copies it: what is in
    // flight is the writer's bounded host buffer, and the fixed rings each point copies; the deepest few of those
    // stay in RAM as the next turn's rollback points (`dsv41_prefix.Live`).
    if (seq >= kv_disk_cache.MIN_PERSIST_TOKENS and prefix_cache_mod.ssdFirstActive(config, bills.diskTierOn(config), bills.launch_state.prefix_cache_ram_enabled))
        return @min(bills.launch_state.disk_writer_permit_bytes, one) +| (1 + @as(u64, dsv41_prefix.live_point_cap)) *| c.stateFixedBytes(config.dsparkOn());
    const held = @min(@as(u64, dsv41_prefix.checkpointMax(bills.launch_state.ssm_checkpoint_max)), seq / stride + 1) + 1;
    return held *| one;
}

// Hooks: bills.zig reaches these through the arch's `bill` declaration; a hook returns null where it has no say.

pub fn requestBill(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, chunk: u64, warm: bills.WarmPrefix) ?u64 {
    _ = kv_bits;
    _ = warm;
    const c = config.dsv41 orelse return null;
    const held = (@as(u64, seq) +| max_tokens) *| c.stateBytesPerToken() +| c.stateFixedBytes(config.dsparkOn());
    // The transient cap a forward runs under (`Config.forwardCap`) is its bill. The prefill's comes out of the decoder
    // pools it may release (`dsv41_borrow`); a decode forward's never does. The decode pool is parked beside either.
    const work = @max(c.forwardCap(dsv41_borrow.verify_rows, seq), c.chunkCap(chunk, seq) -| bills.dsv41BorrowableBytes(config)) +| config.expertPoolBytes();
    return (held +| checkpointReserve(config, seq, chunk)) *| 5 / 4 +| work +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
}

pub fn transientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64, seq: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    return c.chunkCap(chunk, seq) +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
}

pub fn sessionBytes(config: *const model_mod.ModelConfig, kv_bits: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    return c.stateBytesPerToken();
}

pub fn slotRing(config: *const model_mod.ModelConfig, kv_bits: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    return c.stateFixedBytes(config.dsparkOn());
}

pub fn neutralChunkWidth(config: *const model_mod.ModelConfig) ?u32 {
    _ = config;
    return dsv41_config.prefill_width;
}

pub fn diskTokenBytes(config: *const model_mod.ModelConfig, kv_bits: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    return dsv41_prefix.diskBytesPerToken(&c, config.dsparkOn());
}

/// DeepSeek's request bill grows with the prompt (the indexer's scores), so its context is the longest that bill admits.
pub fn memoryContext(config: *const model_mod.ModelConfig, kv_bits: u64, ceiling: u64, active_mem: u64, chunk: u64) ?u32 {
    // Searched past the checkpoint's window, up to the memory a margined default needs, so the margin is judged against memory.
    const upper = @max(config.contextCap(), bills.memoryContextForMargined(config.defaultContext()));
    return bills.maxSushiContextUpTo(config, upper, kv_bits, bills.admissionAvailable(ceiling, active_mem) -| bills.ctxCacheShare(ceiling, active_mem, bills.ctxSizingCacheReserve(config)), chunk, .{});
}

