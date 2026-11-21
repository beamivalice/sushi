//! MiMo's bill tests that need MLX: they run in the MLX tier; the rest of mimo_bill.zig runs in test-host.
const std = @import("std");
const bills = @import("../bills.zig");
const mlx = @import("../mlx.zig");

/// MLX memory above where it started that one sliding layer's kv8 ring holds across one-row appends after its first
/// compaction, the pool emptied and, when `seed`, left 64 free 3-page buffers.
fn ringAppendHeld(seed: bool) !u64 {
    const kv_cache = @import("../kv_cache.zig");
    const s = mlx.gpuStream();
    var cache = try kv_cache.KVCache.initWithConfig(std.testing.allocator, 1, @import("../kv_quant.zig").KVQuantConfig.affine(8));
    defer cache.deinit();
    cache.setSwaRing(128);
    var arrays: [4]mlx.mlx_array = undefined;
    for (&arrays, [_][4]c_int{ .{ 1, 8, 700, 192 }, .{ 1, 8, 700, 128 }, .{ 1, 8, 1, 192 }, .{ 1, 8, 1, 128 } }) |*a, shape| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(a, &shape, 4, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(a.*));
    }
    defer for (arrays) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try @import("../test_support.zig").seedMlxPool(s, 3 * 16384, if (seed) 64 else 0);
    var before: usize = 0;
    for (0..5) |step| {
        var view = try cache.update(0, arrays[if (step == 0) 0 else 2], arrays[if (step == 0) 1 else 3], s, 128);
        view.deinit();
        const e = &cache.entries[0];
        const bufs = [_]mlx.mlx_array{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array };
        const vec = mlx.mlx_vector_array_new_data(&bufs, bufs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        try mlx.check(mlx.mlx_synchronize(s));
        if (step == 0) {
            _ = mlx.mlx_get_active_memory(&before);
            _ = mlx.mlx_reset_peak_memory();
        }
    }
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    return peak -| before;
}

test "a kv8 sliding ring's appends on MLX stay inside the slot bill when MLX's pool holds buffers two pages past its scale rows" {
    try mlx.requireGpu();
    const cfg = @import("../bills_fixtures_test.zig").mimoV2FlashBillConfig();
    // Each append copies the ring's scale and bias rows off the page grid (640 rows of 48 B and 32 B); the slot bill
    // holds every sliding layer's copy.
    const copied = try ringAppendHeld(true) -| try ringAppendHeld(false);
    try std.testing.expect(copied > 0);
    try std.testing.expect(copied <= bills.ringAppendCopyBytes(&cfg, 8) / 39);
}
