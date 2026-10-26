//! The GPU clock under a timed run. `sudo -n powermetrics` prints the active GPU frequency and the thermal
//! pressure level every interval; `sushi bench` reads it beside every request so a throttled run is named, and stops.
//! Docs: process-measurement.md#release-bench.

const std = @import("std");
const io_util = @import("io_util.zig");
const testing = std.testing;

/// Ordered by severity, so the worst level seen is the largest.
pub const Pressure = enum { nominal, moderate, heavy, trapping, sleeping, other };

/// A throttled M5 Max sat at 1025-1120 MHz under `Heavy` pressure; an unthrottled one runs 1578-1620 MHz.
pub const throttle_mhz: f64 = 1100;
/// A sample below this GPU residency is idle time and says nothing about the clock under load.
pub const busy_residency: f64 = 50;
pub const interval_ms: u32 = 500;

pub const Reading = struct {
    /// Monotonic ms when the sample was printed, which is the end of its interval.
    at_ms: i64,
    span_ms: f64,
    mhz: f64,
    residency: f64,
    power_mw: f64,
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

/// Folds powermetrics' text, a line at a time, into readings. A sample is complete at its `GPU Power` line;
/// one that lacks the frequency, residency or pressure line is dropped rather than guessed.
pub const LineParser = struct {
    span_ms: f64 = 0,
    mhz: ?f64 = null,
    residency: ?f64 = null,
    pressure: ?Pressure = null,

    pub fn feed(p: *LineParser, line: []const u8, at_ms: i64) ?Reading {
        if (std.mem.startsWith(u8, line, "*** Sampled system activity")) {
            p.* = .{};
            if (std.mem.indexOf(u8, line, "ms elapsed")) |end| {
                const open = std.mem.lastIndexOfScalar(u8, line[0..end], '(') orelse return null;
                p.span_ms = std.fmt.parseFloat(f64, line[open + 1 .. end]) catch 0;
            }
        } else if (valueAfter(line, "Current pressure level:")) |v| {
            const name = firstWord(v);
            p.pressure = .other;
            inline for (.{ .nominal, .moderate, .heavy, .trapping, .sleeping }) |tag| {
                if (std.ascii.eqlIgnoreCase(name, @tagName(tag))) p.pressure = tag;
            }
        } else if (valueAfter(line, "GPU HW active frequency:")) |v| {
            p.mhz = std.fmt.parseFloat(f64, firstWord(v)) catch null;
        } else if (valueAfter(line, "GPU HW active residency:")) |v| {
            p.residency = std.fmt.parseFloat(f64, std.mem.trimEnd(u8, firstWord(v), "%")) catch null;
        } else if (valueAfter(line, "GPU Power:")) |v| {
            const done = p.*;
            p.* = .{};
            const power = std.fmt.parseFloat(f64, firstWord(v)) catch return null;
            return .{
                .at_ms = at_ms,
                .span_ms = if (done.span_ms > 0) done.span_ms else interval_ms,
                .mhz = done.mhz orelse return null,
                .residency = done.residency orelse return null,
                .power_mw = power,
                .pressure = done.pressure orelse return null,
            };
        }
        return null;
    }

    fn valueAfter(line: []const u8, key: []const u8) ?[]const u8 {
        return if (std.mem.startsWith(u8, line, key)) std.mem.trim(u8, line[key.len..], " \t\r") else null;
    }

    fn firstWord(v: []const u8) []const u8 {
        return v[0 .. std.mem.indexOfAny(u8, v, " (") orelse v.len];
    }
};

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
        try w.writeAll("GPU clock: not monitored (no passwordless `sudo -n powermetrics`), so a thermally throttled run would not be flagged.\n\n");
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

/// powermetrics under `sudo -n`, which never prompts: without a sudoers rule it exits at once and the sampler reads
/// nothing. `-b 1` makes it write each line as it is produced instead of in blocks.
pub const powermetrics_argv = [_][]const u8{ "/usr/bin/sudo", "-n", "powermetrics", "--samplers", "gpu_power,thermal", "-i", std.fmt.comptimePrint("{d}", .{interval_ms}), "-b", "1" };

/// A child process that prints powermetrics text, read on a thread into `readings`.
pub const Sampler = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    child: std.process.Child,
    pid: std.c.pid_t,
    thread: std.Thread = undefined,
    mu: std.Io.Mutex = .init,
    readings: std.ArrayList(Reading) = .empty,
    done: std.atomic.Value(bool) = .init(false),

    pub fn start(gpa: std.mem.Allocator, io: std.Io, argv: []const []const u8) !*Sampler {
        const self = try gpa.create(Sampler);
        errdefer gpa.destroy(self);
        const child = try std.process.spawn(io, .{ .argv = argv, .stdin = .ignore, .stdout = .pipe, .stderr = .ignore });
        self.* = .{ .gpa = gpa, .io = io, .child = child, .pid = child.id.? };
        self.thread = std.Thread.spawn(.{}, readMain, .{self}) catch |e| {
            self.child.kill(io);
            return e;
        };
        return self;
    }

    /// Ends the child (SIGTERM, which sudo passes on to powermetrics) and frees the sampler.
    pub fn stop(self: *Sampler) void {
        if (!self.done.load(.acquire)) std.posix.kill(self.pid, .TERM) catch {};
        self.thread.join();
        self.readings.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    /// True once a reading has arrived; false when the child ended without one or `timeout_ms` passed.
    pub fn waitFirst(self: *Sampler, timeout_ms: u32) bool {
        return self.waitPast(std.math.minInt(i64), timeout_ms);
    }

    /// Waits until a reading ends at or after `t_ms`, so the sample that covers a request's last moments is in.
    /// False when the child ended or `timeout_ms` passed first.
    pub fn waitPast(self: *Sampler, t_ms: i64, timeout_ms: u32) bool {
        var waited: u32 = 0;
        while (waited <= timeout_ms) : (waited += 50) {
            if (self.lastAt()) |at| if (at >= t_ms) return true;
            if (self.done.load(.acquire)) return false;
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
        var buf: [4096]u8 = undefined;
        var r = self.child.stdout.?.readerStreaming(self.io, &buf);
        var parser: LineParser = .{};
        while (r.interface.takeDelimiter('\n') catch null) |line| {
            const at = io_util.nowMsMonotonic(self.io);
            const reading = parser.feed(line, at) orelse continue;
            self.mu.lockUncancelable(self.io);
            self.readings.append(self.gpa, reading) catch {};
            self.mu.unlock(self.io);
        }
        _ = self.child.wait(self.io) catch {};
        self.done.store(true, .release);
    }
};

// ── Tests ───────────────────────────────────────────────────────────────

fn block(comptime pressure: []const u8, comptime mhz: u32, comptime residency: []const u8, comptime power_mw: u32) []const u8 {
    return std.fmt.comptimePrint(
        \\*** Sampled system activity (Mon Oct 26 17:54:29 2026 +0700) (503.30ms elapsed) ***
        \\
        \\
        \\
        \\**** Thermal pressure ****
        \\
        \\Current pressure level: {s}
        \\
        \\**** GPU usage ****
        \\
        \\GPU HW active frequency: {d} MHz
        \\GPU HW active residency: {s}% (338 MHz:   0% 486 MHz:   0% 636 MHz:   0% 796 MHz:   0% 888 MHz:   0% 988 MHz:   0% 1084 MHz:   0% 1182 MHz:   0% 1278 MHz:   0% 1374 MHz:   0% 1470 MHz:   0% 1578 MHz:  74% 1620 MHz:  26%)
        \\GPU SW requested state: (P1 :   0% P2 :   0% P3 :   0% P4 :   0% P5 :   0% P6 :   0% P7 :   0% P8 :   0% P9 :   0% P10 :   0% P11 :   0% P12 :   0% P13 : 100%)
        \\GPU idle residency:   0.00%
        \\GPU Power: {d} mW
        \\
        \\
    , .{ pressure, mhz, residency, power_mw });
}

fn parseAll(a: std.mem.Allocator, text: []const u8, readings: *std.ArrayList(Reading)) !void {
    var p: LineParser = .{};
    var at: i64 = 1000;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |l| {
        if (p.feed(l, at)) |r| {
            try readings.append(a, r);
            at += 500;
        }
    }
}

test "gpu_clock: powermetrics text becomes one reading per sample" {
    const a = testing.allocator;
    var rs: std.ArrayList(Reading) = .empty;
    defer rs.deinit(a);
    try parseAll(a, "Machine model: Mac17,6\nBoot time: Sat Oct 24 17:10:32 2026\n\n" ++
        comptime block("Nominal", 1589, " 100.00", 50761) ++ block("Heavy", 1061, "  31.88", 19400), &rs);
    try testing.expectEqual(@as(usize, 2), rs.items.len);
    try testing.expectEqual(@as(f64, 1589), rs.items[0].mhz);
    try testing.expectEqual(@as(f64, 100), rs.items[0].residency);
    try testing.expectEqual(@as(f64, 50761), rs.items[0].power_mw);
    try testing.expectApproxEqAbs(@as(f64, 503.3), rs.items[0].span_ms, 0.001);
    try testing.expectEqual(Pressure.nominal, rs.items[0].pressure);
    try testing.expectEqual(@as(i64, 1000), rs.items[0].at_ms);
    try testing.expectEqual(Pressure.heavy, rs.items[1].pressure);
    try testing.expectEqual(@as(f64, 1061), rs.items[1].mhz);
    try testing.expectApproxEqAbs(@as(f64, 31.88), rs.items[1].residency, 0.001);
}

test "gpu_clock: a sample missing its thermal line is dropped, not read as Nominal" {
    const a = testing.allocator;
    var rs: std.ArrayList(Reading) = .empty;
    defer rs.deinit(a);
    const full = comptime block("Nominal", 1589, " 100.00", 50761);
    const cut = comptime std.mem.indexOf(u8, full, "Current pressure level").?;
    const eol = comptime std.mem.indexOfScalarPos(u8, full, cut, '\n').?;
    try parseAll(a, full[0..cut] ++ full[eol + 1 ..], &rs);
    try testing.expectEqual(@as(usize, 0), rs.items.len);
}

test "gpu_clock: pressure names, unknown ones stay non-nominal" {
    const a = testing.allocator;
    var rs: std.ArrayList(Reading) = .empty;
    defer rs.deinit(a);
    try parseAll(a, comptime block("Moderate", 1500, " 100.00", 1) ++ block("Trapping", 900, " 100.00", 1) ++ block("Sleeping", 900, " 100.00", 1) ++ block("Weird", 900, " 100.00", 1), &rs);
    try testing.expectEqual(Pressure.moderate, rs.items[0].pressure);
    try testing.expectEqual(Pressure.trapping, rs.items[1].pressure);
    try testing.expectEqual(Pressure.sleeping, rs.items[2].pressure);
    try testing.expectEqual(Pressure.other, rs.items[3].pressure);
}

fn rd(at_ms: i64, mhz: f64, residency: f64, pressure: Pressure) Reading {
    return .{ .at_ms = at_ms, .span_ms = 500, .mhz = mhz, .residency = residency, .power_mw = 0, .pressure = pressure };
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

test "gpu_clock: a sampler reads a child's powermetrics text and stops it; a child that ends silently reads nothing" {
    const a = testing.allocator;
    const io = testing.io;
    const text = comptime block("Nominal", 1589, " 100.00", 50761) ++ block("Heavy", 1061, " 100.00", 19400);
    const s = try Sampler.start(a, io, &.{ "/bin/sh", "-c", "printf '%s' \"$0\"; exec sleep 60", text });
    try testing.expect(s.waitFirst(5000));
    while ((try s.summarizeAll(a)).busy_samples < 2) std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};
    const run = try s.summarizeAll(a);
    try testing.expectEqual(@as(f64, 1061), run.min_mhz);
    try testing.expectEqual(@as(u8, 0b101), run.seen);
    s.stop();

    const dead = try Sampler.start(a, io, &.{ "/bin/sh", "-c", "exit 1" });
    try testing.expect(!dead.waitFirst(5000));
    dead.stop();
}
