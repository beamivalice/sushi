//! `SUSHI_DSV41_TRANSIENT_PEAK=1`: one `[transient-peak]` line per DeepSeek forward phase, the high-water of the
//! bytes MLX allocated itself (imported weights left out) since the phase began.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const transformer = @import("transformer.zig");

pub const Phase = enum {
    decode,
    dspark_draft,
    dspark_verify,
    prefill_chunk,
    vision_encode,

    fn label(self: Phase) []const u8 {
        return switch (self) {
            .decode => "decode",
            .dspark_draft => "dspark draft",
            .dspark_verify => "dspark verify",
            .prefill_chunk => "prefill chunk",
            .vision_encode => "vision encode",
        };
    }
};

var enabled_cache: ?bool = null;

fn enabled() bool {
    return transformer.diagEnvOnCached(&enabled_cache, "SUSHI_DSV41_TRANSIENT_PEAK");
}

pub fn begin() void {
    if (enabled()) mlx.resetTransientPeak();
}

/// `rows` is the phase's width: tokens of the pass, or patches of an image.
pub fn end(phase: Phase, rows: usize) void {
    if (!enabled()) return;
    var cache: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache);
    var line: [160]u8 = undefined;
    log.info("{s}", .{formatLine(&line, phase, rows, mlx.transientPeak(), cache)});
}

fn formatLine(buf: []u8, phase: Phase, rows: usize, peak: usize, cache: usize) []const u8 {
    const gb = 1024.0 * 1024.0 * 1024.0;
    return std.fmt.bufPrint(buf, "[transient-peak] {s} rows={d} peak_gb={d:.3} pool_gb={d:.3}\n", .{
        phase.label(), rows, @as(f64, @floatFromInt(peak)) / gb, @as(f64, @floatFromInt(cache)) / gb,
    }) catch "[transient-peak] line too long\n";
}

test "a transient-peak line names the phase, its rows and both sizes in GiB" {
    var buf: [160]u8 = undefined;
    try std.testing.expectEqualStrings(
        "[transient-peak] prefill chunk rows=2048 peak_gb=4.350 pool_gb=0.500\n",
        formatLine(&buf, .prefill_chunk, 2048, 4670776934, 536870912),
    );
}
