const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const Stopwatch = @import("io_util.zig").Stopwatch;

pub const RoundMeter = struct {
    pub const Phase = enum { entry, stage, head, markov, draft_read, plan, verify, verdict, finish, publish };
    const Part = struct { calls: u32 = 0, wall_ns: u64 = 0, wait_ns: u64 = 0, waits: u64 = 0, dispatches: u64 = 0, barriers: u64 = 0 };
    const count = std.enums.values(Phase).len;

    fence_draft: bool,
    timer: Stopwatch,
    last_ns: u64 = 0,
    last: @TypeOf(mlx.engine.stats()),
    parts: [count]Part = @splat(.{}),

    pub fn init(io: std.Io) RoundMeter {
        return .{ .fence_draft = @import("io_util.zig").diagEnvOn("SUSHI_DSV41_DSPARK_STAGE_TIMES"), .timer = Stopwatch.init(io), .last = mlx.engine.stats() };
    }

    pub fn part(self: *RoundMeter, phase: Phase) *Part {
        return &self.parts[@intFromEnum(phase)];
    }

    pub fn mark(self: *RoundMeter, phase: Phase) void {
        const now = self.timer.read();
        const stats = mlx.engine.stats();
        const p = self.part(phase);
        p.calls += 1;
        if (self.fence_draft and phase == .stage) log.info("[dspark-stage] stage={d} wall_ns={d} wait_ns={d} dispatches={d}\n", .{ p.calls, now - self.last_ns, stats.wait_ns - self.last.wait_ns, stats.dispatches - self.last.dispatches });
        p.wall_ns += now - self.last_ns;
        p.wait_ns += stats.wait_ns - self.last.wait_ns;
        p.waits += stats.waits - self.last.waits;
        p.dispatches += stats.dispatches - self.last.dispatches;
        p.barriers += stats.barriers - self.last.barriers;
        self.last_ns = now;
        self.last = stats;
    }

    fn column(self: *const RoundMeter, comptime field: []const u8) [count]u64 {
        var out: [count]u64 = undefined;
        for (self.parts, &out) |p, *v| v.* = @field(p, field);
        return out;
    }

    pub fn report(self: *const RoundMeter, ctx: usize, rows: usize, accepted: u32, source: []const u8) void {
        log.info("[drafter-meter] ctx={d} rows={d} accepted={d} source={s} start_ns={d} entry/stage/head/markov/draft_read/plan/verify/verdict/finish/publish wall_ns={any} wait_ns={any} waits={any} dispatches={any} barriers={any}\n", .{
            ctx, rows, accepted, source, self.timer.started_at.nanoseconds, self.column("wall_ns"), self.column("wait_ns"), self.column("waits"), self.column("dispatches"), self.column("barriers"),
        });
    }
};
