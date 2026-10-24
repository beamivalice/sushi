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
