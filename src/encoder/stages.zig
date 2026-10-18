const std = @import("std");

pub const Stage = enum { read, hessian, prep, factor, wait, scale, feedback, search, pack };

const count = std.enums.values(Stage).len;

pub const Stages = struct {
    ns: [count]u64 = @splat(0),

    pub fn add(self: *Stages, stage: Stage, ns: u64) void {
        _ = @atomicRmw(u64, &self.ns[@intFromEnum(stage)], .Add, ns, .monotonic);
    }

    pub fn get(self: *const Stages, stage: Stage) u64 {
        return self.ns[@intFromEnum(stage)];
    }

    pub fn total(self: *const Stages) u64 {
        var sum: u64 = 0;
        for (self.ns) |v| sum += v;
        return sum;
    }

    pub fn merge(self: *Stages, other: *const Stages) void {
        for (other.ns, 0..) |v, i| _ = @atomicRmw(u64, &self.ns[i], .Add, v, .monotonic);
    }

    pub fn write(self: *const Stages, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (std.enums.values(Stage), 0..) |stage, i| {
            try w.print("{s}{s} {d:.1}s", .{ if (i == 0) "" else ", ", @tagName(stage), @as(f64, @floatFromInt(self.get(stage))) / 1e9 });
        }
    }
};

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const Lap = struct {
    last: u64,
    stages: ?*Stages,

    pub fn start(stages: ?*Stages) Lap {
        return .{ .last = if (stages != null) nowNs() else 0, .stages = stages };
    }

    /// Restart the lap without charging any stage: the time between was charged elsewhere.
    pub fn skip(self: *Lap) void {
        if (self.stages != null) self.last = nowNs();
    }

    pub fn mark(self: *Lap, stage: Stage) void {
        const s = self.stages orelse return;
        const now = nowNs();
        s.add(stage, now - self.last);
        self.last = now;
    }
};

test "stages: laps land in their stage and total sums them" {
    var s = Stages{};
    var lap = Lap.start(&s);
    const t0 = nowNs();
    while (nowNs() - t0 < 2_000_000) {}
    lap.mark(.scale);
    try std.testing.expect(s.get(.scale) >= 1_000_000);
    try std.testing.expectEqual(@as(u64, 0), s.get(.search));
    var other = Stages{};
    other.add(.search, 7);
    s.merge(&other);
    try std.testing.expectEqual(s.get(.scale) + 7, s.total());
    var none = Lap.start(null);
    none.mark(.read);
}
