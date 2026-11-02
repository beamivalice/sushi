//! The GPU clock under a timed run. A thread reads the active GPU frequency (IOReport, `gpu_clock_live.zig`) and the
//! thermal pressure level (the system notification) every interval, no sudo; `sushi bench` reads it beside every
//! request so a throttled run is named, and stops.
//! Docs: process-measurement.md#release-bench.

const std = @import("std");
const builtin = @import("builtin");
const io_util = @import("io_util.zig");
const live = @import("gpu_clock_live.zig");
const status = @import("status.zig");
const testing = std.testing;

/// Ordered by severity, so the worst level seen is the largest.
pub const Pressure = enum { nominal, moderate, heavy, trapping, sleeping, other };

/// A throttled M5 Max sat at 1025-1120 MHz under `Heavy` pressure; an unthrottled one runs 1578-1620 MHz.
pub const throttle_mhz: f64 = 1100;
/// A sample below this GPU utilization (%) is idle time and says nothing about the clock under load.
pub const busy_residency: f64 = 50;
pub const interval_ms: u32 = 500;

pub const Reading = struct {
    /// Monotonic ms when the sample was printed, which is the end of its interval.
    at_ms: i64,
    span_ms: f64,
    mhz: f64,
    residency: f64,
    pressure: Pressure,
};

pub fn pressureLabel(p: Pressure) []const u8 {
    return switch (p) {
        .nominal => "Nominal",
        .moderate => "Moderate",
        .heavy => "Heavy",
        .trapping => "Trapping",
        .sleeping => "Sleeping",
        .other => "Unknown",
    };
}

extern "c" fn notify_register_check(name: [*:0]const u8, token: *c_int) u32;
extern "c" fn notify_get_state(token: c_int, state: *u64) u32;

/// `com.apple.system.thermalpressurelevel`: 0 Nominal, 1 Moderate, 2 Heavy (read on a box powermetrics called Heavy),
/// then Trapping and Sleeping in powermetrics' order.
pub fn pressureFromLevel(level: u64) Pressure {
    return switch (level) {
        0 => .nominal,
        1 => .moderate,
        2 => .heavy,
        3 => .trapping,
        4 => .sleeping,
        else => .other,
    };
}

/// Null when the notification is unavailable.
fn readPressure(token: *?c_int) ?Pressure {
    if (token.* == null) {
        var t: c_int = 0;
        if (notify_register_check("com.apple.system.thermalpressurelevel", &t) != 0) return null;
        token.* = t;
    }
    var level: u64 = 0;
    if (notify_get_state(token.*.?, &level) != 0) return null;
    return pressureFromLevel(level);
}

pub const Summary = struct {
    /// Median active MHz of the busy samples; 0 when none was busy.
    mhz: f64 = 0,
    /// Worst pressure level of the samples; null when no sample covered the request.
    pressure: ?Pressure = null,
    samples: usize = 0,
};

fn medianOf(a: std.mem.Allocator, vals: []const f64) !f64 {
    if (vals.len == 0) return 0;
    const sorted = try a.dupe(f64, vals);
    defer a.free(sorted);
    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
    const n = sorted.len;
    return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/// The readings that cover mostly `[from_ms, to_ms]`: a sample is the interval that ends at `at_ms`, and counts when
/// at least half of it (or of the request, if shorter) lies inside.
pub fn summarize(a: std.mem.Allocator, readings: []const Reading, from_ms: i64, to_ms: i64) !Summary {
    var busy: std.ArrayList(f64) = .empty;
    defer busy.deinit(a);
    var out: Summary = .{};
    const from: f64 = @floatFromInt(from_ms);
    const to: f64 = @floatFromInt(to_ms);
    for (readings) |r| {
        const end: f64 = @floatFromInt(r.at_ms);
        const overlap = @min(end, to) - @max(end - r.span_ms, from);
        if (overlap <= 0 or overlap * 2 < @min(r.span_ms, to - from)) continue;
        out.samples += 1;
        if (out.pressure == null or @intFromEnum(r.pressure) > @intFromEnum(out.pressure.?)) out.pressure = r.pressure;
        if (r.residency >= busy_residency) try busy.append(a, r.mhz);
    }
    out.mhz = try medianOf(a, busy.items);
    return out;
}

pub const Verdict = enum { ok, clock };

/// The abort rule is the busy clock at or under `throttle_mhz`. The pressure level is only recorded: a long run on a
/// fan-cooled box reports `Heavy` while the GPU still holds 1500+ MHz, and the clock is what the tok/s follows.
pub fn verdict(mhz: f64, pressure: ?Pressure) Verdict {
    _ = pressure;
    return if (mhz > 0 and mhz <= throttle_mhz) .clock else .ok;
}

/// A whole run's readings, for the report header.
pub const RunSummary = struct {
    monitored: bool = false,
    min_mhz: f64 = 0,
    median_mhz: f64 = 0,
    busy_samples: usize = 0,
    /// Bit `@intFromEnum(Pressure)` is set for every level seen.
    seen: u8 = 0,
};

pub fn summarizeRun(a: std.mem.Allocator, readings: []const Reading) !RunSummary {
    var busy: std.ArrayList(f64) = .empty;
    defer busy.deinit(a);
    var out: RunSummary = .{ .monitored = true };
    for (readings) |r| {
        out.seen |= @as(u8, 1) << @intFromEnum(r.pressure);
        if (r.residency >= busy_residency) try busy.append(a, r.mhz);
    }
    out.busy_samples = busy.items.len;
    out.median_mhz = try medianOf(a, busy.items);
    if (busy.items.len > 0) out.min_mhz = std.mem.min(f64, busy.items);
    return out;
}

pub fn writeRunHeader(w: *std.Io.Writer, s: RunSummary) !void {
    if (!s.monitored) {
        try w.writeAll("GPU clock: not monitored (the IOReport clock was unreadable), so a thermally throttled run would not be flagged.\n\n");
        return;
    }
    try w.print("GPU clock, sampled every {d} ms: median {d:.0} MHz, minimum {d:.0} MHz over {d} busy samples; thermal pressure seen:", .{ interval_ms, s.median_mhz, s.min_mhz, s.busy_samples });
    var first = true;
    for (std.enums.values(Pressure)) |tag| {
        if (s.seen & (@as(u8, 1) << @intFromEnum(tag)) != 0) {
            try w.print("{s} {s}", .{ if (first) "" else ",", pressureLabel(tag) });
            first = false;
        }
    }
    try w.writeAll(if (first) " none.\n\n" else ".\n\n");
}

/// Reads the clock and the pressure every `interval_ms` on a thread into `readings`.
pub const Sampler = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    reader: live.Reader,
    thread: std.Thread = undefined,
    mu: std.Io.Mutex = .init,
    readings: std.ArrayList(Reading) = .empty,
    stopping: std.atomic.Value(bool) = .init(false),

    /// Fails when the clock cannot be read on this box.
    pub fn start(gpa: std.mem.Allocator, io: std.Io) !*Sampler {
        const self = try gpa.create(Sampler);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .reader = live.Reader.open() orelse return error.GpuClockUnavailable };
        self.thread = try std.Thread.spawn(.{}, readMain, .{self});
        return self;
    }

    pub fn stop(self: *Sampler) void {
        self.stopping.store(true, .release);
        self.thread.join();
        self.readings.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    /// True once a reading has arrived; false when `timeout_ms` passed.
    pub fn waitFirst(self: *Sampler, timeout_ms: u32) bool {
        return self.waitPast(std.math.minInt(i64), timeout_ms);
    }

    /// Waits until a reading ends at or after `t_ms`, so the sample that covers a request's last moments is in.
    /// False when `timeout_ms` passed first.
    pub fn waitPast(self: *Sampler, t_ms: i64, timeout_ms: u32) bool {
        var waited: u32 = 0;
        while (waited <= timeout_ms) : (waited += 50) {
            if (self.lastAt()) |at| if (at >= t_ms) return true;
            std.Io.sleep(self.io, .fromMilliseconds(50), .real) catch {};
        }
        return false;
    }

    fn lastAt(self: *Sampler) ?i64 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return if (self.readings.items.len > 0) self.readings.items[self.readings.items.len - 1].at_ms else null;
    }

    pub fn summarizeWindow(self: *Sampler, a: std.mem.Allocator, from_ms: i64, to_ms: i64) !Summary {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return summarize(a, self.readings.items, from_ms, to_ms);
    }

    pub fn summarizeAll(self: *Sampler, a: std.mem.Allocator) !RunSummary {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return summarizeRun(a, self.readings.items);
    }

    fn readMain(self: *Sampler) void {
        var token: ?c_int = null;
        var last = io_util.nowMsMonotonic(self.io);
        while (!self.stopping.load(.acquire)) {
            std.Io.sleep(self.io, .fromMilliseconds(interval_ms), .real) catch {};
            const mhz = self.reader.sample() orelse continue;
            const pressure = readPressure(&token) orelse continue;
            const at = io_util.nowMsMonotonic(self.io);
            const reading: Reading = .{ .at_ms = at, .span_ms = @floatFromInt(@max(at - last, 1)), .mhz = @floatFromInt(mhz), .residency = @floatFromInt(status.getGpuPct()), .pressure = pressure };
            last = at;
            self.mu.lockUncancelable(self.io);
            self.readings.append(self.gpa, reading) catch {};
            self.mu.unlock(self.io);
        }
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

test "gpu_clock: notification levels map onto the labels, unknown ones stay non-nominal" {
    try testing.expectEqual(Pressure.nominal, pressureFromLevel(0));
    try testing.expectEqual(Pressure.moderate, pressureFromLevel(1));
    try testing.expectEqual(Pressure.heavy, pressureFromLevel(2));
    try testing.expectEqual(Pressure.trapping, pressureFromLevel(3));
    try testing.expectEqual(Pressure.sleeping, pressureFromLevel(4));
    try testing.expectEqual(Pressure.other, pressureFromLevel(7));
}

fn rd(at_ms: i64, mhz: f64, residency: f64, pressure: Pressure) Reading {
    return .{ .at_ms = at_ms, .span_ms = 500, .mhz = mhz, .residency = residency, .pressure = pressure };
}

test "gpu_clock: a request's summary is the median clock of its busy samples and its worst pressure" {
    const a = testing.allocator;
    const rs = [_]Reading{
        rd(500, 300, 5, .nominal), // before the request
        rd(1000, 1600, 100, .nominal),
        rd(1500, 1580, 100, .nominal),
        rd(2000, 1620, 100, .moderate),
        rd(2500, 400, 10, .nominal), // idle tail: not busy
        rd(5000, 200, 100, .heavy), // after the request
    };
    const s = try summarize(a, &rs, 600, 2400);
    try testing.expectEqual(@as(f64, 1600), s.mhz);
    try testing.expectEqual(Pressure.moderate, s.pressure.?);
    try testing.expectEqual(@as(usize, 4), s.samples);
    const none = try summarize(a, &rs, 3000, 4000);
    try testing.expectEqual(@as(?Pressure, null), none.pressure);
    try testing.expectEqual(@as(f64, 0), none.mhz);
    // Only idle samples: a worst pressure but no clock to judge.
    const idle = try summarize(a, &rs, 2400, 2500);
    try testing.expectEqual(@as(f64, 0), idle.mhz);
    try testing.expectEqual(Pressure.nominal, idle.pressure.?);
}

test "gpu_clock: a sample that mostly lies outside the request does not count" {
    const a = testing.allocator;
    // Covers 1000..1500; the request starts at 1400.
    const rs = [_]Reading{rd(1500, 1050, 100, .heavy)};
    const s = try summarize(a, &rs, 1400, 9000);
    try testing.expectEqual(@as(usize, 0), s.samples);
}

test "gpu_clock: the abort rule" {
    try testing.expectEqual(Verdict.ok, verdict(1600, .nominal));
    try testing.expectEqual(Verdict.ok, verdict(1101, .nominal));
    try testing.expectEqual(Verdict.clock, verdict(1100, .nominal));
    try testing.expectEqual(Verdict.clock, verdict(1050, .nominal));
    // The pressure level never aborts on its own: a Heavy box holding 1539 MHz is a measurement, a 1040 MHz one is not.
    try testing.expectEqual(Verdict.ok, verdict(1600, .moderate));
    try testing.expectEqual(Verdict.ok, verdict(1539, .heavy));
    try testing.expectEqual(Verdict.ok, verdict(1600, .other));
    try testing.expectEqual(Verdict.clock, verdict(1100, .moderate));
    try testing.expectEqual(Verdict.clock, verdict(1040, .heavy));
    // No busy sample: nothing to judge.
    try testing.expectEqual(Verdict.ok, verdict(0, .other));
    try testing.expectEqual(Verdict.ok, verdict(0, null));
}

test "gpu_clock: the report header names the clock and the pressure levels seen, or says it was not monitored" {
    const a = testing.allocator;
    var rs: std.ArrayList(Reading) = .empty;
    defer rs.deinit(a);
    try rs.appendSlice(a, &.{ rd(1, 1600, 100, .nominal), rd(2, 1000, 100, .heavy), rd(3, 300, 5, .nominal) });
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try writeRunHeader(&out.writer, try summarizeRun(a, rs.items));
    try testing.expectEqualStrings("GPU clock, sampled every 500 ms: median 1300 MHz, minimum 1000 MHz over 2 busy samples; thermal pressure seen: Nominal, Heavy.\n\n", out.written());
    out.clearRetainingCapacity();
    try writeRunHeader(&out.writer, .{});
    try testing.expect(std.mem.startsWith(u8, out.written(), "GPU clock: not monitored"));
}

test "gpu_clock: a sampler delivers readings and stops" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;
    const s = Sampler.start(testing.allocator, testing.io) catch return;
    defer s.stop();
    try testing.expect(s.waitFirst(5000));
    const run = try s.summarizeAll(testing.allocator);
    try testing.expect(run.monitored and run.seen != 0);
}
