//! The routed experts `--drop-expert` leaves unloaded: one bit row a layer. The expert store reads it; the arch that
//! ranks the experts fills it. Std only, so the MLX-free test tier can reach it.
const std = @import("std");

/// The widest router a set covers (the window kernels hold one thread per expert).
pub const max_experts = 512;
pub const words = max_experts / 64;

pub const DropSet = struct {
    allocator: std.mem.Allocator,
    n: u16,
    first_layer: u16,
    rows: [][words]u64,
    /// Where the ranking came from, as the file states it.
    source: []u8,

    pub fn deinit(self: *DropSet) void {
        self.allocator.free(self.rows);
        self.allocator.free(self.source);
    }

    /// Names the dropped set: two sets with the same digest leave out the same experts in every layer.
    pub fn digest(self: *const DropSet) u64 {
        var h = std.hash.XxHash64.init(0x6472_6f70_6578_7031);
        h.update(std.mem.asBytes(&self.n));
        h.update(std.mem.asBytes(&self.first_layer));
        h.update(std.mem.sliceAsBytes(self.rows));
        return h.final();
    }

    /// How many of `lo..hi` this layer drops.
    pub fn droppedIn(self: *const DropSet, layer: usize, lo: usize, hi: usize) usize {
        var count: usize = 0;
        for (lo..hi) |e| count += @intFromBool(self.isDropped(layer, e));
        return count;
    }

    pub fn isDropped(self: *const DropSet, layer: usize, expert: usize) bool {
        if (layer < self.first_layer or layer - self.first_layer >= self.rows.len or expert >= max_experts) return false;
        return self.rows[layer - self.first_layer][expert / 64] >> @intCast(expert % 64) & 1 != 0;
    }
};

test "a drop set answers per layer and expert and counts a range" {
    const t = std.testing;
    var rows = [_][words]u64{ @splat(0), @splat(0) };
    rows[0][0] = 0b0110;
    rows[1][1] = 1;
    var source: [0]u8 = .{};
    var set: DropSet = .{ .allocator = t.allocator, .n = 2, .first_layer = 3, .rows = &rows, .source = &source };
    try t.expect(set.isDropped(3, 1) and set.isDropped(3, 2) and !set.isDropped(3, 0));
    try t.expect(set.isDropped(4, 64));
    try t.expect(!set.isDropped(2, 1) and !set.isDropped(5, 0) and !set.isDropped(3, max_experts));
    try t.expectEqual(@as(usize, 2), set.droppedIn(3, 0, 4));
    try t.expectEqual(@as(usize, 1), set.droppedIn(3, 2, 8));
    const before = set.digest();
    rows[0][0] = 0b0100;
    try t.expect(set.digest() != before);
}
