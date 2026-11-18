//! Arrays a module keeps for the process, one set for each side of the host (1 the engine's, 0 MLX's): an array belongs
//! to the side that made it. A cache keeps only an array that was made (an engine's maker can be refused and hand back
//! an empty handle), and its first put enrolls it, so the last model of a side gives back every cache's arrays of that
//! side (`dropAll`) with no list kept by hand. Inference thread only.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const host_side = @import("host_side.zig");

const Node = struct {
    drop: *const fn (*Node, u1) void,
    next: ?*Node = null,
    enrolled: bool = false,
};

var enrolled: ?*Node = null;

/// Frees side `side`'s arrays in every cache that ever kept one.
pub fn dropAll(side: u1) void {
    var at = enrolled;
    while (at) |node| : (at = node.next) node.drop(node, side);
}

fn unenroll(node: *Node) void {
    if (!node.enrolled) return;
    var at = &enrolled;
    while (at.*) |n| : (at = &n.next) if (n == node) {
        at.* = node.next;
        node.* = .{ .drop = node.drop };
        return;
    };
}

/// Up to `n` arrays a side, by `K`; when a side is full the least recently used goes. A handle from the cache is the
/// cache's: valid until its side drops or `n` other keys are put after it, never freed by the caller.
pub fn SideCache(comptime K: type, comptime n: usize) type {
    return struct {
        const Self = @This();
        const Slot = struct { key: K, arr: mlx.mlx_array, tick: u64 };

        slots: [2][n]Slot = undefined,
        len: [2]usize = .{ 0, 0 },
        tick: u64 = 0,
        node: Node = .{ .drop = dropNode },

        pub fn get(self: *Self, side: u1, key: K) ?mlx.mlx_array {
            for (self.slots[side][0..self.len[side]]) |*s| if (std.meta.eql(s.key, key)) {
                self.tick +%= 1;
                s.tick = self.tick;
                return s.arr;
            };
            return null;
        }

        /// Keeps `arr` for `key` on `side` and hands it back, or refuses an empty handle and keeps nothing.
        pub fn put(self: *Self, side: u1, key: K, arr: mlx.mlx_array) error{HostArrayRefused}!mlx.mlx_array {
            const kept = try host_side.made(arr);
            if (!self.node.enrolled) {
                self.node.next = enrolled;
                self.node.enrolled = true;
                enrolled = &self.node;
            }
            self.tick +%= 1;
            const side_slots = &self.slots[side];
            const len = &self.len[side];
            for (side_slots[0..len.*]) |*s| if (std.meta.eql(s.key, key)) {
                _ = mlx.mlx_array_free(s.arr);
                s.* = .{ .key = key, .arr = kept, .tick = self.tick };
                return kept;
            };
            if (len.* < n) {
                side_slots[len.*] = .{ .key = key, .arr = kept, .tick = self.tick };
                len.* += 1;
                return kept;
            }
            var oldest: usize = 0;
            for (side_slots, 0..) |s, i| if (s.tick < side_slots[oldest].tick) {
                oldest = i;
            };
            _ = mlx.mlx_array_free(side_slots[oldest].arr);
            side_slots[oldest] = .{ .key = key, .arr = kept, .tick = self.tick };
            return kept;
        }

        pub fn drop(self: *Self, side: u1) void {
            for (self.slots[side][0..self.len[side]]) |s| _ = mlx.mlx_array_free(s.arr);
            self.len[side] = 0;
        }

        /// A cache that does not live for the process (a test's) frees both sides and leaves the enrolled list, which
        /// would otherwise point into its dead frame.
        pub fn deinit(self: *Self) void {
            self.drop(0);
            self.drop(1);
            unenroll(&self.node);
        }

        fn dropNode(node: *Node, side: u1) void {
            const self: *Self = @fieldParentPtr("node", node);
            self.drop(side);
        }
    };
}

/// Int scalars by value, as kernels take their sizes.
var ints: SideCache(c_int, 16) = .{};

/// A 0-d int32 scalar of `v` on `side`, the cache's.
pub fn int(side: u1, v: c_int) error{HostArrayRefused}!mlx.mlx_array {
    return ints.get(side, v) orelse ints.put(side, v, mlx.mlx_array_new_int(v));
}

const testing = std.testing;

test "a side cache refuses an empty handle and keeps nothing, and keeps a made one for its side alone" {
    var c: SideCache(u8, 2) = .{};
    defer c.deinit();
    try testing.expectError(error.HostArrayRefused, c.put(0, 1, .{}));
    try testing.expect(c.get(0, 1) == null);
    const a = try c.put(0, 1, mlx.mlx_array_new_int(7));
    try testing.expectEqual(a.ctx, c.get(0, 1).?.ctx);
    try testing.expect(c.get(1, 1) == null);
}

test "a full side gives up its least recently used array, and a key put again replaces its array" {
    var c: SideCache(u8, 2) = .{};
    defer c.deinit();
    _ = try c.put(0, 1, mlx.mlx_array_new_int(1));
    _ = try c.put(0, 2, mlx.mlx_array_new_int(2));
    _ = c.get(0, 1);
    _ = try c.put(0, 3, mlx.mlx_array_new_int(3));
    try testing.expect(c.get(0, 2) == null);
    try testing.expect(c.get(0, 1) != null);
    const again = try c.put(0, 1, mlx.mlx_array_new_int(10));
    try testing.expectEqual(again.ctx, c.get(0, 1).?.ctx);
    try testing.expectEqual(@as(usize, 2), c.len[0]);
}

test "dropAll frees a side in every cache that kept an array, and leaves the other side" {
    const S = struct {
        var a: SideCache(u8, 1) = .{};
        var b: SideCache(u8, 1) = .{};
    };
    defer S.a.deinit();
    defer S.b.deinit();
    _ = try S.a.put(0, 1, mlx.mlx_array_new_int(1));
    _ = try S.a.put(1, 1, mlx.mlx_array_new_int(2));
    _ = try S.b.put(0, 1, mlx.mlx_array_new_int(3));
    dropAll(0);
    try testing.expect(S.a.get(0, 1) == null);
    try testing.expect(S.b.get(0, 1) == null);
    try testing.expect(S.a.get(1, 1) != null);
}

test "a full side never gives up the two arrays a call just took: a kernel holds its K and N scalars together" {
    var c: SideCache(c_int, 4) = .{};
    defer c.deinit();
    for (0..4) |i| _ = try c.put(0, @intCast(i), mlx.mlx_array_new_int(@intCast(i)));
    var v: c_int = 100;
    for (0..40) |_| {
        const k = c.get(0, v) orelse try c.put(0, v, mlx.mlx_array_new_int(v));
        const n = c.get(0, v + 1) orelse try c.put(0, v + 1, mlx.mlx_array_new_int(v + 1));
        try testing.expectEqual(k.ctx, c.get(0, v).?.ctx);
        try testing.expectEqual(n.ctx, c.get(0, v + 1).?.ctx);
        v += 2;
    }
}

fn enrolledCount() usize {
    var n: usize = 0;
    var at = enrolled;
    while (at) |node| : (at = node.next) n += 1;
    return n;
}

test "a cache that goes leaves the enrolled list, so dropAll never reaches a cache past its scope" {
    const before = enrolledCount();
    {
        var c: SideCache(u8, 1) = .{};
        defer c.deinit();
        _ = try c.put(0, 1, mlx.mlx_array_new_int(1));
        _ = try c.put(1, 1, mlx.mlx_array_new_int(2));
        try testing.expectEqual(before + 1, enrolledCount());
    }
    try testing.expectEqual(before, enrolledCount());
    dropAll(0);
}
