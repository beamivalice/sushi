//! Hy4's kernel launch configs, made once per launch shape and kept: a decode forward launches each of its kernels at a
//! few shapes, so a launch makes no config calls (about ten FFI calls a fresh config) and the engine finds the
//! pipeline the config last named. A cache past `cap` shapes (prefill chunks of many widths) starts again.
const std = @import("std");
const mlx = @import("../mlx.zig");

pub const Config = mlx.mlx_fast_metal_kernel_config;

const cap = 256;
const gpa = std.heap.c_allocator;

pub fn Cache(comptime Key: type) type {
    return struct {
        map: std.AutoHashMapUnmanaged(Key, Config) = .empty,

        /// The config of `key`, made by `build(key)` the first time. The cache owns it.
        pub fn get(self: *@This(), key: Key, comptime build: fn (Key) anyerror!Config) !Config {
            if (self.map.get(key)) |c| return c;
            if (self.map.count() >= cap) self.clear();
            const c = try build(key);
            self.map.put(gpa, key, c) catch |e| {
                _ = mlx.mlx_fast_metal_kernel_config_free(c);
                return e;
            };
            return c;
        }

        fn clear(self: *@This()) void {
            var it = self.map.valueIterator();
            while (it.next()) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c.*);
            self.map.clearRetainingCapacity();
        }
    };
}

/// An f32 constant as a template int (its bits): the kernel reads it as `as_type<float>(NAME)`, a value fixed by the
/// model's geometry, so it never makes a second pipeline.
pub fn bits(v: f32) c_int {
    return @bitCast(v);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

/// The FFI calls one call of `f` makes.
fn callsOf(comptime f: anytype, args: anytype) !u64 {
    const before = mlx.op_count.load(.monotonic);
    const out = try @call(.auto, f, args);
    const n = mlx.op_count.load(.monotonic) - before;
    _ = mlx.mlx_array_free(out);
    return n;
}

test "hy4 kcfg: a Hy4 kernel launched again at a shape it ran makes none of its config's calls (rope, norm)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const v: [144]f32 = @splat(0.5);
    const x = try mlx.newData(&v, &.{ 3, 1, 48 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const rope = @import("hy4_rope.zig");
    const norm = @import("hy4_norm.zig");
    // A shape no other test launches, so the first call makes its config.
    const first = try callsOf(rope.tail, .{ es, x, 16, 1234.5, 7, 1, .float32 });
    const again = try callsOf(rope.tail, .{ es, x, 16, 1234.5, 8, 1, .float32 });
    try testing.expect(again < first);
    const r = try mlx.newData(&v, &.{ 1, 144 }, .float32);
    defer _ = mlx.mlx_array_free(r);
    const w = try mlx.newData(&v, &.{144}, .float32);
    defer _ = mlx.mlx_array_free(w);
    const n_first = try callsOf(norm.rows, .{ es, r, null, w, 0.25, .float32 });
    const n_again = try callsOf(norm.rows, .{ es, r, null, w, 0.25, .float32 });
    try testing.expect(n_again < n_first);
}
