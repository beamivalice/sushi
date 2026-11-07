//! Timing helpers for the kernel microbenches (`sushi-ubench`): a chain of dependent calls, so launch latency counts
//! as it does in a forward, and the effective read rate of a weight against the roofline.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const io_util = @import("mlx_host").io_util;
const root = @import("sushi_exl3");
const kernels = root.kernels;
const synth = root.synth;

/// Streaming read rate the M5 Max delivers to a read kernel ([perf-baselines](../../docs/perf-baselines.md#roofline)).
pub const ROOFLINE_GBPS: f64 = 540.0;

var link_kernel: ?mlx.mlx_fast_metal_kernel = null;
const LINK_SOURCE: [:0]const u8 =
    \\const uint i = thread_position_in_grid.x;
    \\if (i >= uint(LEN)) return;
    \\out[i] = a[i];
    \\if (i == 0u && float(d[0]) == 1.5e30f) out[0] = a[1];
;

/// A copy of `x0` that waits on `y`: the next step's input, so steps run one after another.
pub fn link(s: mlx.mlx_stream, x0: mlx.mlx_array, y: mlx.mlx_array) !mlx.mlx_array {
    const k = try kernels.getNamedKernel(&link_kernel, "sushi_exl3_bench_link", &.{ "a", "d" }, &.{"out"}, LINK_SOURCE, "");
    const sh = mlx.getShape(x0);
    const len: c_int = @intCast(mlx.mlx_array_size(x0));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, mlx.mlx_array_dtype(x0)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, len, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LEN", len));
    const ov = try kernels.applyOuts(s, k, &.{ x0, y }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Nanoseconds for `steps` dependent calls of `f` (each fed the previous call's link), the link included.
pub fn chainNs(s: mlx.mlx_stream, x0: mlx.mlx_array, steps: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), mlx.mlx_stream, usize, mlx.mlx_array) anyerror!mlx.mlx_array) !u64 {
    var x = try link(s, x0, x0);
    defer _ = mlx.mlx_array_free(x);
    for (0..steps) |i| {
        const y = try f(ctx, s, i, x);
        defer _ = mlx.mlx_array_free(y);
        const next = try link(s, x0, y);
        _ = mlx.mlx_array_free(x);
        x = next;
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(x));
    return sw.read();
}

/// The link alone: what every arm's chain pays per step beyond its own kernels.
pub fn linkOnly(_: void, s: mlx.mlx_stream, _: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_set(&y, x));
    _ = s;
    return y;
}

pub fn median(times: []u64) f64 {
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    return @floatFromInt(times[times.len / 2]);
}

/// The fastest round: a box shared with another GPU job shows its best case there, a quiet box its median.
pub fn fastest(times: []u64) f64 {
    return @floatFromInt(std.mem.min(u64, times));
}

/// GB/s a call reads `bytes` at in `us`.
pub fn gbps(bytes: f64, us: f64) f64 {
    return bytes / us / 1e3;
}

pub const randomWords = synth.randomWords;
pub const makeBank = synth.makeBank;
pub const freeBank = synth.freeBank;
