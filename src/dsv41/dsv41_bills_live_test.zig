//! DeepSeek-V4.1's bill on the tiny fixture, which needs MLX (`DSV41_TINY`).
const std = @import("std");
const bills = @import("../bills.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const launch_state = @import("../server_state.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");

test "dsv41 bills: a prefill holds its restore points beside the state with a RAM tier, and only the writer's in-flight bytes SSD-first (DSV41_TINY)" {
    try mlx.requireGpu();
    const dir = try @import("dsv41_model.zig").fixtureDir();
    const t = std.testing;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model_mod.parseConfig(io, t.allocator, dir);
    defer cfg.deinit(t.allocator);
    const saved = .{ launch_state.prefix_cache_ram_enabled, launch_state.ssm_checkpoint_stride, launch_state.ssm_checkpoint_max, launch_state.prefix_cache_disk_bytes, launch_state.prefix_cache_capacity };
    defer {
        launch_state.prefix_cache_ram_enabled = saved[0];
        launch_state.ssm_checkpoint_stride = saved[1];
        launch_state.ssm_checkpoint_max = saved[2];
        launch_state.prefix_cache_disk_bytes = saved[3];
        launch_state.prefix_cache_capacity = saved[4];
    }
    launch_state.ssm_checkpoint_stride = 256;
    launch_state.ssm_checkpoint_max = 16;
    launch_state.prefix_cache_capacity = 32;
    const c = cfg.dsv41.?;
    const seq: u64 = 20_000;
    const one = c.stateFixedBytes(cfg.dsparkOn()) + seq * c.stateBytesPerToken();
    launch_state.prefix_cache_ram_enabled = false;
    launch_state.prefix_cache_disk_bytes = 0;
    const off = bills.prefillNeededAtChunk(&cfg, seq, 0, 8, 512, .{});
    try t.expectEqual(@as(u64, 0), bills.checkpointReserve(&cfg, seq, 512));
    for ([_][2]bool{ .{ true, false }, .{ true, true } }) |tiers| {
        launch_state.prefix_cache_ram_enabled = tiers[0];
        launch_state.prefix_cache_disk_bytes = if (tiers[1]) 20 << 30 else 0;
        // One checkpoint per 8192-token stride and the prompt end, plus the copy taken before a thin.
        try t.expectEqual(@as(u64, 4) * one, bills.checkpointReserve(&cfg, seq, 512));
        try t.expectEqual(off + 4 * one * 5 / 4, bills.prefillNeededAtChunk(&cfg, seq, 0, 8, 512, .{}));
        // The list is capped, whatever the prompt.
        try t.expectEqual(@as(u64, 5) * (c.stateFixedBytes(cfg.dsparkOn()) + 1_000_000 * c.stateBytesPerToken()), bills.checkpointReserve(&cfg, 1_000_000, 512));
    }
    // SSD-first: each point goes to the tier as it is taken, so the bill is the writer's bounded host buffer and one
    // point's fixed rings, whatever the prompt; under the tier's floor the points wait for the commit in RAM.
    launch_state.prefix_cache_ram_enabled = false;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const rings = c.stateFixedBytes(cfg.dsparkOn());
    // The rollback points the next turn keeps in RAM are billed with it: each is one set of fixed rings.
    const kept = @as(u64, dsv41_prefix.live_point_cap) * rings;
    try t.expectEqual(one + rings + kept, bills.checkpointReserve(&cfg, seq, 512));
    const big: u64 = 1 << 40;
    try t.expectEqual(@min(launch_state.disk_writer_permit_bytes, rings + big * c.stateBytesPerToken()) + rings + kept, bills.checkpointReserve(&cfg, big, 512));
    try t.expect(bills.checkpointReserve(&cfg, 100, 512) == 2 * (rings + 100 * c.stateBytesPerToken()));
    // An entry on disk keeps its rows once and the fixed rings of one point per grid step.
    try t.expectEqual(c.stateBytesPerToken() + (rings + dsv41_prefix.row_span - 1) / dsv41_prefix.row_span, bills.diskBytesPerToken(&cfg, 8));
    launch_state.ssm_checkpoint_stride = 0;
    try t.expectEqual(@as(u64, 0), bills.checkpointReserve(&cfg, seq, 512));
}

test "the peak reading counts the engine's pool at its fullest beside MLX's, and a reset starts the pool's from what is in use" {
    try mlx.requireGpu();
    const t = std.testing;
    const gpu_mem = @import("../gpu_mem.zig");
    const core = @import("../tensor/core.zig");
    const mb = 1 << 20;
    try core.sync();
    gpu_mem.resetPeak();
    const base = gpu_mem.peakBytes();
    // Each part starts from what it has in use.
    try t.expect(base >= (mlx.engine.poolTotal() -| mlx.engine.poolFree()) + mlx.engine.mappedTotal());
    const a = try core.fresh(.uint8, &.{9 * mb + 5});
    core.release(a);
    try t.expect(gpu_mem.peakBytes() >= base + 9 * mb);
    gpu_mem.resetPeak();
    try t.expectEqual(base, gpu_mem.peakBytes());
}
