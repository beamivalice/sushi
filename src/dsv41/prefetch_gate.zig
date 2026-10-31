//! Whether the expert prefetch pays: it reads ahead only while the cache misses little, because a read that
//! competes with the misses for the SSD costs more than it saves. Pure data, no MLX.
const std = @import("std");

/// Decode forwards of 2 to 16 rows are measured; a wider one is a prefill and says nothing about the decode cache.
const max_rows = 16;
/// Missed experts per row above which the SSD is the bottleneck and the prefetch's extra reads slow the round.
const on_below: f32 = 8;
const window: f32 = 8;

pub const PrefetchGate = struct {
    /// Smoothed distinct missed experts per row per decode forward.
    ema: f32 = 0,

    pub fn observe(self: *PrefetchGate, rows: usize, misses: u64) void {
        if (rows == 0 or rows > max_rows) return;
        const per_row = @as(f32, @floatFromInt(misses)) / @as(f32, @floatFromInt(rows));
        self.ema += (per_row - self.ema) / window;
    }

    pub fn on(self: *const PrefetchGate) bool {
        return self.ema < on_below;
    }
};

test "prefetch gate: starts on" {
    try std.testing.expect((PrefetchGate{}).on());
}

test "prefetch gate: twenty forwards at 17 misses a row turn it off" {
    var g = PrefetchGate{};
    for (0..20) |_| g.observe(4, 68);
    try std.testing.expect(!g.on());
}

test "prefetch gate: seven forwards at 5 a row from 1 a row stay on" {
    var g = PrefetchGate{};
    for (0..30) |_| g.observe(1, 1);
    for (0..7) |_| g.observe(4, 20);
    try std.testing.expect(g.on());
}

test "prefetch gate: it comes back on after forwards at 1 a row" {
    var g = PrefetchGate{};
    for (0..20) |_| g.observe(4, 68);
    try std.testing.expect(!g.on());
    for (0..30) |_| g.observe(1, 1);
    try std.testing.expect(g.on());
}

test "prefetch gate: a prefill and a zero-row forward do not move it" {
    var g = PrefetchGate{};
    g.observe(17, 1000);
    g.observe(4096, 1000);
    g.observe(0, 1000);
    try std.testing.expectEqual(@as(f32, 0), g.ema);
}
