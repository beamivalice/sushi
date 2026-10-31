//! The key MLX caches a custom kernel's compiled library under: its name and source, its template args, whether each
//! input binds as a scalar, a `constant` array (under 8 elements) or a device array, and its output dtypes. A key
//! MLX has not seen is a Metal compile of tens of ms on the inference thread. `SUSHI_KERNEL_BUILD_LOG=1` logs every
//! new key; a test counts them with `trackForTest`.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

/// MLX's `max_constant_array_size`: a smaller input binds as `constant`, which changes the generated source.
const constant_below = 8;

var mode: enum(u8) { unknown, off, on } = .unknown;
var log_on = false;
var mu: std.c.pthread_mutex_t = .{};
const gpa = std.heap.page_allocator;

const Kernel = struct { name: []u8, source: u64 };
var kernels: std.AutoHashMapUnmanaged(usize, Kernel) = .empty;
/// A config's template args and output dtypes in the order MLX writes them.
var configs: std.AutoHashMapUnmanaged(usize, std.ArrayList(u8)) = .empty;
var keys: std.AutoHashMapUnmanaged(u64, void) = .empty;
var new_keys: usize = 0;
var last_key: std.ArrayList(u8) = .empty;

/// Tests track from the first kernel, so a library an earlier test compiled is no new key to a later one.
pub fn on() bool {
    if (mode == .unknown) {
        log_on = diagOn(std.c.getenv("SUSHI_KERNEL_BUILD_LOG"));
        mode = if (log_on or builtin.is_test) .on else .off;
    }
    return mode == .on;
}

fn diagOn(v: ?[*:0]const u8) bool {
    const s = std.mem.span(v orelse return false);
    return s.len > 0 and !std.mem.eql(u8, s, "0");
}

/// The count so far, with tracking on.
pub fn trackForTest() usize {
    _ = on();
    mode = .on;
    return count();
}

/// Distinct keys seen since tracking began.
pub fn count() usize {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    return new_keys;
}

/// The newest key, for a failing test's message.
pub fn last() []const u8 {
    return last_key.items;
}

pub fn kernelNew(k: mlx.mlx_fast_metal_kernel, name: [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8, row_contiguous: bool, atomic: bool) void {
    if (!on() or k.ctx == null) return;
    var h = std.hash.Wyhash.init(0);
    h.update(std.mem.span(source));
    h.update(std.mem.span(header));
    h.update(&.{ @intFromBool(row_contiguous), @intFromBool(atomic) });
    const copy = gpa.dupe(u8, std.mem.span(name)) catch return;
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    if (kernels.fetchPut(gpa, @intFromPtr(k.ctx), .{ .name = copy, .source = h.final() }) catch null) |old| gpa.free(old.value.name);
}

pub fn kernelFree(k: mlx.mlx_fast_metal_kernel) void {
    if (mode != .on) return;
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    if (kernels.fetchRemove(@intFromPtr(k.ctx))) |old| gpa.free(old.value.name);
}

pub fn configNote(cfg: mlx.mlx_fast_metal_kernel_config, comptime fmt: []const u8, args: anytype) void {
    if (!on()) return;
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    const e = configs.getOrPut(gpa, @intFromPtr(cfg.ctx)) catch return;
    if (!e.found_existing) e.value_ptr.* = .empty;
    e.value_ptr.print(gpa, fmt, args) catch {};
}

pub fn configFree(cfg: mlx.mlx_fast_metal_kernel_config) void {
    if (mode != .on) return;
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    if (configs.fetchRemove(@intFromPtr(cfg.ctx))) |old| {
        var l = old.value;
        l.deinit(gpa);
    }
}

pub fn apply(k: mlx.mlx_fast_metal_kernel, inputs: mlx.mlx_vector_array, cfg: mlx.mlx_fast_metal_kernel_config) void {
    if (!on()) return;
    var key: std.ArrayList(u8) = .empty;
    defer key.deinit(gpa);
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    if (kernels.get(@intFromPtr(k.ctx))) |kn| {
        key.print(gpa, "{s} src={x}", .{ kn.name, kn.source }) catch return;
    } else key.print(gpa, "kernel@{x}", .{@intFromPtr(k.ctx)}) catch return;
    if (configs.get(@intFromPtr(cfg.ctx))) |c| {
        key.print(gpa, " {s}", .{c.items}) catch return;
    } else key.print(gpa, " cfg@{x}", .{@intFromPtr(cfg.ctx)}) catch return;
    key.appendSlice(gpa, " in:") catch return;
    for (0..mlx.mlx_vector_array_size(inputs)) |i| {
        var a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a);
        if (mlx.mlx_vector_array_get(&a, inputs, i) != 0) return;
        const bind: []const u8 = if (mlx.mlx_array_ndim(a) == 0) "s" else if (mlx.mlx_array_size(a) < constant_below) "c" else "";
        key.print(gpa, " {s}{s}", .{ @tagName(mlx.mlx_array_dtype(a)), bind }) catch return;
    }
    const e = keys.getOrPut(gpa, std.hash.Wyhash.hash(0, key.items)) catch return;
    if (e.found_existing) return;
    new_keys += 1;
    last_key.clearRetainingCapacity();
    last_key.appendSlice(gpa, key.items) catch {};
    if (log_on) log.info("[kernel-build] #{d} {s}\n", .{ new_keys, key.items });
}

const probe_source =
    \\uint i = thread_position_in_grid.x;
    \\out[i] = x[i] + b[i & 3u];
;

/// `out = x + b[i & 3]` over n values; x is 0.5 * i, so every dtype below holds each value exactly.
fn probeRun(k: mlx.mlx_fast_metal_kernel, s: mlx.mlx_stream, dtype: mlx.mlx_dtype, b_len: c_int, out: []f32) !void {
    const n: c_int = @intCast(out.len);
    var x_vals: [64]f32 = undefined;
    var b_vals: [64]f32 = undefined;
    for (0..out.len) |i| x_vals[i] = 0.5 * @as(f32, @floatFromInt(i));
    for (0..@intCast(b_len)) |i| b_vals[i] = 0.25 * @as(f32, @floatFromInt(i + 1));
    const x32 = mlx.mlx_array_new_data(&x_vals, &[_]c_int{n}, 1, .float32);
    defer _ = mlx.mlx_array_free(x32);
    const b32 = mlx.mlx_array_new_data(&b_vals, &[_]c_int{b_len}, 1, .float32);
    defer _ = mlx.mlx_array_free(b32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&x, x32, dtype, s));
    try mlx.check(mlx.mlx_astype(&b, b32, dtype, s));
    const inputs = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ x, b }, 2);
    defer _ = mlx.mlx_vector_array_free(inputs);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{n}, 1, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 16, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "UNUSED", 7));
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, k, inputs, cfg, s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(y32));
    @memcpy(out, mlx.mlx_array_data_float32(y32).?[0..out.len]);
}

test "one kernel object across input dtypes and bindings: same outputs, one library key per source" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const names = [_][*:0]const u8{ "x", "b" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_kernel_keys_probe", iv, ov, probe_source, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);

    // Constant-bound b in f32 and f16, then a device-bound b (16 values) in f32: three sources, three keys.
    const Case = struct { dtype: mlx.mlx_dtype, b_len: c_int };
    const cases = [_]Case{ .{ .dtype = .float32, .b_len = 4 }, .{ .dtype = .float16, .b_len = 4 }, .{ .dtype = .float32, .b_len = 16 } };
    const base = trackForTest();
    var got: [32]f32 = undefined;
    for (0..3) |_| {
        for (cases) |c| {
            try probeRun(k, s, c.dtype, c.b_len, &got);
            for (got, 0..) |v, i| try t.expectEqual(0.5 * @as(f32, @floatFromInt(i)) + 0.25 * @as(f32, @floatFromInt((i & 3) + 1)), v);
        }
        try t.expectEqual(base + 3, count());
    }
}

extern "c" fn mlx_fast_metal_kernel_config_add_template_arg_bool(cls: mlx.mlx_fast_metal_kernel_config, name: [*:0]const u8, value: bool) c_int;

test "template args of one value and different kinds each get their own source" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const names = [_][*:0]const u8{"x"};
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    // An int 1 and a bool true both name the kernel `_1`, but sizeof(N) tells the two sources apart.
    const k = mlx.mlx_fast_metal_kernel_new("sushi_kernel_keys_kind", iv, ov, "out[thread_position_in_grid.x] = float(sizeof(N)) + x[0] * 0.0f;", "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x = mlx.mlx_array_new_data(&data, &[_]c_int{10}, 1, .float32);
    defer _ = mlx.mlx_array_free(x);
    const inputs = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{x}, 1);
    defer _ = mlx.mlx_vector_array_free(inputs);
    for (0..3) |_| for ([_]bool{ false, true }) |as_bool| {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{4}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 4, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 4, 1, 1));
        try mlx.check(if (as_bool) mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "N", true) else mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", 1));
        var outs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs_vec, k, inputs, cfg, s));
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_vector_array_get(&y, outs_vec, 0));
        try mlx.check(mlx.mlx_array_eval(y));
        const got = mlx.mlx_array_data_float32(y).?;
        try t.expectEqual(@as(f32, if (as_bool) 1 else 4), got[0]);
    };
}
