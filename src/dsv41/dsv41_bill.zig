//! DeepSeek-V4.1's own terms of the bill: the compressed state, the restore points and the forward cap.
const std = @import("std");
const settings_resolve = @import("../settings_resolve.zig");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_config = @import("dsv41_config.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const prefill_plan = @import("../prefill_plan.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");

/// The restore points a DeepSeek prefill holds in RAM while it runs: with a RAM tier the capped list plus the copy
/// taken before a thin, each a full copy of the state at most `seq` tokens deep; SSD-first only what is in flight to
/// the tier. Zero with no prefix cache at all.
pub fn checkpointReserve(config: *const model_mod.ModelConfig, seq: u64, chunk: u64) u64 {
    const c = config.dsv41 orelse return 0;
    const base = bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config));
    const stride = prefill_plan.effectiveSsmCheckpointStride(base, @max(@as(usize, @intCast(chunk)), settings_resolve.prefill_chunk.value));
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
    // The cap a forward runs under (`Config.forwardCap`) is its bill. The prefill's comes out of the decoder pools it
    // may release (`dsv41_borrow`); a decode forward's never does, and the last runs at the reply's end, its verify
    // block past it. On MLX the decode pool is parked beside either. On a Cooler load the engine's pool is the cap
    // itself (its free buffers are inside it) and MLX holds nothing.
    const engine = @import("../cooler_side.zig").onEngine(config);
    const decode_end = seq +| max_tokens +| dsv41_borrow.verify_rows;
    const capped = @max(c.forwardCap(dsv41_borrow.verify_rows, decode_end), c.chunkCapOn(engine, chunk, seq) -| bills.borrowableBytes(config));
    // A Cooler cap carries the bill's one margin, which `bills.billMargin` adds to the bill: the bill takes the cost.
    const forward = if (engine) capped -| dsv41_config.BILL_MARGIN_BYTES else capped;
    const work = forward +| (if (engine) 0 else config.expertPoolBytes());
    const stored = held +| checkpointReserve(config, seq, chunk);
    // On Sushi Cooler the bill is exact (`bills.billMargin` adds the one margin); MLX keeps its calibrated quarter.
    return (if (@import("../cooler_side.zig").onEngine(config)) stored else stored *| 5 / 4) +| work +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
}

/// A fully resident load joins rows (`dsv41_sched.coolerRows`): a request's decode state is its own, made at its first
/// forward, so all of it grows after admission. At one row the model's parked state is resident already.
pub fn commitBytes(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64, mtp_on: bool) ?u64 {
    _ = kv_bits;
    _ = mtp_on;
    const c = config.dsv41 orelse return null;
    // As `dsv41_sched.coolerRows`: rows are a resident load's on Sushi Cooler.
    if (!config.expert_resident or !@import("../cooler_side.zig").onEngine(config)) return null;
    return (@as(u64, prompt_len) +| max_tokens) *| c.stateBytesPerToken() +| c.stateFixedBytes(config.dsparkOn());
}

pub fn transientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64, seq: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    return c.chunkCapOn(@import("../cooler_side.zig").onEngine(config), chunk, seq) +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
}

/// A chunk's live price at `kv_len` (the width hook, the served warm): the forward it becomes with the tail it may absorb
/// under the per-chunk width (`prefill_plan.tailMergeMax`), at the position that forward ends at.
pub fn chunkTransientAt(config: *const model_mod.ModelConfig, kv_bits: u64, width: u64, kv_len: u64) ?u64 {
    _ = kv_bits;
    const c = config.dsv41 orelse return null;
    const rows = width +| (prefill_plan.tailMergeMax(@intCast(width)) - 1);
    return c.forwardCapOn(@import("../cooler_side.zig").onEngine(config), rows, kv_len +| rows) +| (if (config.expert_streaming) config.expert_fill_peak_bytes else 0);
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
    return bills.maxSushiContextUpTo(config, upper, kv_bits, bills.contextBudget(config, ceiling, active_mem), chunk, .{});
}


/// The width line of the load log: DeepSeek starts at its own prefill width and says when memory sized it down.
pub fn prefillChunkLoadLine(config: *const model_mod.ModelConfig, buf: []u8, pinned: u32, launch: usize) ?[]const u8 {
    _ = config;
    if (launch <= dsv41_config.prefill_width) return null;
    if (pinned == dsv41_config.prefill_width)
        return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (DeepSeek's default width; --prefill-chunk overrides)\n", .{pinned}) catch null;
    return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (memory-sized down from DeepSeek's {d}; --prefill-chunk overrides)\n", .{ pinned, dsv41_config.prefill_width }) catch null;
}

/// `dsv41_borrow.headroom_fn`: the one live headroom (`bills.headroomNow`).
pub fn dsv41Headroom(config: *const model_mod.ModelConfig) bills.Headroom {
    return bills.headroomNow(config, 0);
}

/// PURE: `dsv41Headroom` over the readings. `active` is `gpu_mem.activeBytes`, MLX's and the engine's bytes in use.
pub fn dsv41HeadroomFrom(ceiling: u64, active: u64, engine_free: u64, config: *const model_mod.ModelConfig) u64 {
    return bills.liveHeadroomFrom(config, ceiling, active, 0, engine_free);
}

/// On a Cooler load the engine's free buffers are parked (`parkedBytes`) and a forward or an encode under its own cap
/// reuses them (`bills.Headroom.reusable`).
pub fn reusableUnderCap(config: *const model_mod.ModelConfig, engine_free: u64) ?u64 {
    return if (@import("../cooler_side.zig").onEngine(config)) engine_free else 0;
}

/// The decode pool a request's bill holds itself (`requestBill`): MLX's parked pool; on a Cooler load none, the pool being
/// the engine's free buffers, which the live headroom already leaves out.
pub fn billedParkedBytes(config: *const model_mod.ModelConfig) u64 {
    return if (@import("../cooler_side.zig").onEngine(config)) 0 else config.expertPoolBytes();
}

/// Off every live headroom of a DeepSeek forward (`bills.liveHeadroomFrom`): the decode pool the ledger parks, no part
/// of the active bytes, and the engine's free pool buffers, which nothing on MLX's side can reclaim. On a Cooler load
/// the parked pool IS the engine's free buffers under its cap: off once, at the larger of the two.
pub fn parkedBytes(config: *const model_mod.ModelConfig, engine_free: u64) u64 {
    if (@import("../cooler_side.zig").onEngine(config)) return @max(engine_free, config.expertPoolBytes());
    return engine_free +| config.expertPoolBytes();
}
