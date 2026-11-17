//! `SUSHI_TRANSIENT_PEAK=1`: one `[transient-peak]` line per forward phase of any arch, the high-water of the bytes MLX
//! and the engine allocated themselves (weights left out) past what was in use when the phase began, both pools' free
//! bytes, and when the caller knows it the bill the phase was admitted under. A phase begun inside another is the outer
//! one's.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");

pub const Phase = enum {
    decode,
    verify,
    draft,
    prefill_chunk,
    vision_encode,

    fn label(self: Phase) []const u8 {
        return switch (self) {
            .decode => "decode",
            .verify => "verify",
            .draft => "draft",
            .prefill_chunk => "prefill chunk",
            .vision_encode => "vision encode",
        };
    }

    /// A forward's phase by its rows: one is a decode step, a verify block's are a verify, wider is a prefill chunk.
    pub fn ofRows(rows: usize) Phase {
        return if (rows <= 1) .decode else if (rows <= @import("sampler_draw.zig").max_rows) .verify else .prefill_chunk;
    }
};

var enabled_cache: ?bool = null;
var depth: u32 = 0;

pub fn enabled() bool {
    return io_util.diagEnvOnCached(&enabled_cache, "SUSHI_TRANSIENT_PEAK");
}

pub fn begin() void {
    if (!enabled()) return;
    depth += 1;
    if (depth > 1) return;
    mlx.resetTransientPeak();
    mlx.engine.resetPoolPeak(.phase);
    // Both high-waters restart from what is in use (the KV cache, a slot's state): the phase's own bytes are past it.
    in_use_at_begin = highWater();
    held_at_begin = mlx.engine.poolHeldPeak(.phase);
}

var in_use_at_begin: usize = 0;
var held_at_begin: usize = 0;

/// `rows` is the phase's width: tokens of the pass, or patches of an image; `bill` what it was admitted under, if known.
pub fn end(phase: Phase, rows: usize, bill: ?u64) void {
    if (!enabled()) return;
    depth -= 1;
    if (depth > 0) return;
    var cache: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache);
    var line: [192]u8 = undefined;
    log.info("{s}", .{formatLine(&line, phase, rows, phaseBytes(), mlx.engine.poolHeldPeak(.phase) -| held_at_begin, cache + mlx.engine.poolFree(), bill)});
}

fn highWater() usize {
    return mlx.transientPeak() + mlx.engine.poolPeak(.phase);
}

fn phaseBytes() usize {
    return highWater() -| in_use_at_begin;
}

/// `held`: what the engine's pool grew to hold over the phase, its free buffers included (what Metal counts).
fn formatLine(buf: []u8, phase: Phase, rows: usize, peak: usize, held: usize, cache: usize, bill: ?u64) []const u8 {
    const gb = 1024.0 * 1024.0 * 1024.0;
    const head = std.fmt.bufPrint(buf, "[transient-peak] {s} rows={d} peak_gb={d:.3} held_gb={d:.3} pool_gb={d:.3}", .{
        phase.label(), rows, @as(f64, @floatFromInt(peak)) / gb, @as(f64, @floatFromInt(held)) / gb, @as(f64, @floatFromInt(cache)) / gb,
    }) catch return "[transient-peak] line too long\n";
    const tail = if (bill) |b|
        std.fmt.bufPrint(buf[head.len..], " bill_gb={d:.3}\n", .{@as(f64, @floatFromInt(b)) / gb})
    else
        std.fmt.bufPrint(buf[head.len..], "\n", .{});
    return buf[0 .. head.len + (tail catch return "[transient-peak] line too long\n").len];
}

test "a transient-peak line names the phase, its rows, its peak in use, what the pool held and the cache in GiB, and the bill when known" {
    var buf: [192]u8 = undefined;
    try std.testing.expectEqualStrings(
        "[transient-peak] prefill chunk rows=2048 peak_gb=4.350 held_gb=4.850 pool_gb=0.500\n",
        formatLine(&buf, .prefill_chunk, 2048, 4670776934, 5207647846, 536870912, null),
    );
    try std.testing.expectEqualStrings(
        "[transient-peak] verify rows=4 peak_gb=0.250 held_gb=0.000 pool_gb=0.500 bill_gb=1.000\n",
        formatLine(&buf, .verify, 4, 268435456, 0, 536870912, 1 << 30),
    );
}

test "a phase's transient counts what it added past what was in use when it began, not what stayed resident" {
    try mlx.requireGpu();
    const was = enabled_cache;
    enabled_cache = true;
    defer enabled_cache = was;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const mb: c_int = 1 << 20;
    var resident = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(resident);
    try mlx.check(mlx.mlx_zeros(&resident, &[_]c_int{ 64, mb }, 2, .uint8, s));
    try mlx.check(mlx.mlx_array_eval(resident));
    begin();
    {
        var scratch = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scratch);
        try mlx.check(mlx.mlx_zeros(&scratch, &[_]c_int{ 16, mb }, 2, .uint8, s));
        try mlx.check(mlx.mlx_array_eval(scratch));
    }
    const added = phaseBytes();
    depth -= 1;
    try std.testing.expect(added >= 16 << 20);
    try std.testing.expect(added < 64 << 20);
}

