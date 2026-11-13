//! The live GPU readings behind the bill: MLX's own counters and Metal's working-set limit.
const std = @import("std");
const arch_serve = @import("arch/serve.zig");
const log = @import("log.zig");
const metrics = @import("status.zig");
const mlx = @import("mlx.zig");
const bills = @import("bills.zig");
const device = @import("sushi_exl3").device;

// The host test root links no MLX: a read of the live GPU there is a bug in the test, not a number.
const host_tier = @import("build_options").host_tier;

fn liveRead() noreturn {
    @panic("a host-tier test read the live GPU: it belongs in bills_live_test.zig");
}

/// Bytes MLX has in use plus the engine's: its pool buffers bound to live tensors and the mappings it wraps (weights).
/// Safe off the inference thread: the engine's side is atomics, and a tensor lent to MLX is in both counters, so the
/// sum errs high, never low.
pub fn activeBytes() usize {
    if (host_tier) return liveRead();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    return active + engineInUse();
}

/// MLX's reclaimable pool plus the engine's free pool buffers: held by the process, reusable only by their owner.
pub fn cacheBytes() usize {
    if (host_tier) return liveRead();
    var cache: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache);
    return cache + mlx.engine.poolFree();
}

/// The high-water since `resetPeak`: MLX's own peak, and beside it the engine's pool at its fullest with the mappings
/// it wraps now. The two peaks need not have met, so the sum errs high. MLX starts its peak from zero at a reset, so
/// until it next allocates this reads below `activeBytes` by what MLX holds.
pub fn peakBytes() usize {
    if (host_tier) return liveRead();
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    return peak + mlx.engine.poolPeak(.process) + mlx.engine.mappedTotal();
}

pub fn resetPeak() void {
    if (host_tier) return liveRead();
    _ = mlx.mlx_reset_peak_memory();
    mlx.engine.resetPoolPeak(.process);
}

fn engineInUse() usize {
    return (mlx.engine.poolTotal() -| mlx.engine.poolFree()) + mlx.engine.mappedTotal();
}

/// Read once on the main thread in `resolveMlxCadence`: the limit is static per device, and a
/// connection thread asking for it must not call MLX.
var working_set_at_start = std.atomic.Value(u64).init(0);

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process: ggml's
/// global std::terminate handler prints the backtrace, but the throw is MLX's).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
pub fn getGpuWorkingSetLimit() u64 {
    const cached = working_set_at_start.load(.monotonic);
    if (cached > 0) return cached;
    if (host_tier) return liveRead();
    const max_rec = mlx.maxRecommendedWorkingSet();
    if (max_rec > 0) return @as(u64, max_rec);
    return bills.getMetalBufferLimit();
}

/// Read the floor once, at startup, before MLX holds a model: the only moment the counter is
/// clean both ways: right after a load it lags low while the weights are still being wired, and
/// after an unload it lags high while freed pages unwire.
pub fn measureGpuFloor() void {
    const active = activeBytes();
    const cache = cacheBytes();
    const in_use = metrics.getGpuInUseSystemMemBytes();
    const floor = bills.gpuFloorFrom(in_use, active, cache);
    bills.gpu_floor_bytes.store(floor, .monotonic);
    log.info("[mem] GPU floor {d} MB (system GPU in use {d} MB, MLX active {d} MB + cache {d} MB)\n", .{
        floor >> 20,
        in_use >> 20,
        active >> 20,
        cache >> 20,
    });
}

/// Inference thread, no model resident: other processes may have released GPU memory since
/// startup. Lowers the floor, never raises it (`bills.idleGpuFloor`).
pub fn refreshIdleGpuFloor() void {
    const active = activeBytes();
    const cache = cacheBytes();
    const in_use = metrics.getGpuInUseSystemMemBytes();
    const was = bills.gpu_floor_bytes.load(.monotonic);
    const floor = bills.idleGpuFloor(was, in_use, active, cache);
    if (floor == was) return;
    bills.gpu_floor_bytes.store(floor, .monotonic);
    bills.wired_floor_logged.store(false, .monotonic);
    log.info("[mem] GPU floor {d} MB (was {d} MB): system GPU in use {d} MB, MLX active {d} MB + cache {d} MB\n", .{
        floor >> 20,
        was >> 20,
        in_use >> 20,
        active >> 20,
        cache >> 20,
    });
}

/// Main thread, before other threads: resolve MLX's command-buffer cadence the way MLX does
/// and say what it is, since MLX reads the env once and reports nothing.
pub fn resolveMlxCadence() void {
    const class = device.current().class;
    var arch_buf: [64]u8 = undefined;
    const arch = device.archName(&arch_buf);
    const env_mb: ?[]const u8 = if (std.c.getenv("MLX_MAX_MB_PER_BUFFER")) |v| std.mem.span(v) else null;
    const env_ops: ?[]const u8 = if (std.c.getenv("MLX_MAX_OPS_PER_BUFFER")) |v| std.mem.span(v) else null;
    bills.mlx_mb_per_buffer = bills.mlxPerBufferFromEnv(env_mb, bills.mlxDefaultMbPerBuffer(class));
    const ops = bills.mlxPerBufferFromEnv(env_ops, bills.mlxDefaultOpsPerBuffer(class));
    log.info("[mem] MLX command buffers: {d} ops / {d} MB ({s}, {s}); prefill bill +{d} MB in flight\n", .{
        ops,
        bills.mlx_mb_per_buffer,
        if (env_mb != null or env_ops != null) "env" else "default",
        if (arch.len > 0) arch else "unknown arch",
        bills.cadenceInflightBytes(bills.mlx_mb_per_buffer) >> 20,
    });
    measureGpuFloor();
    working_set_at_start.store(getGpuWorkingSetLimit(), .monotonic);
}

/// Impure wrapper: apply the cap to the live MLX allocator. Called ONCE from
/// `main()`, above every subcommand branch — not per serve path, because a
/// hand-rolled per-path config is exactly how `runHeadlessServe` (the mode the
/// app always launches) came to silently eat launch flags.
///
/// Never RAISES a tighter existing cap.
pub fn applyMlxCacheLimit() void {
    const cap = bills.mlxCacheLimitBytes(metrics.getTotalMemBytes());
    if (cap == 0) return;
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(cap));
    if (prev < cap) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    log.info("[mem] MLX buffer-pool cap {d} MB (was {d} MB)\n", .{ cap >> 20, prev >> 20 });
}

/// Lowers MLX's buffer-pool cap to a per-model ledger term; never raises the RAM-wide cap.
pub fn capMlxPool(bytes: u64) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(bytes));
    if (prev < bytes) {
        var tmp: usize = 0;
        _ = mlx.mlx_set_cache_limit(&tmp, prev);
        return;
    }
    arch_serve.poolCapped(prev);
    log.info("[mem] MLX buffer-pool cap {d} MB from the ssd-budget ledger (was {d} MB)\n", .{ bytes >> 20, prev >> 20 });
}
