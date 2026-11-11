//! Energy and host work per token for `sushi bench`: this process's own CPU joules, CPU seconds and instructions
//! (`proc_pid_rusage`), the whole machine's GPU joules (IOReport, an undocumented library bound at run time) and the
//! unit and interpolation arithmetic between them. A missing library, symbol, group or channel leaves a reason, never a
//! zero. Docs: process-measurement.md#energy-per-token.
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const is_macos = builtin.os.tag == .macos;

// ── Units ───────────────────────────────────────────────────────────────

/// mach_timebase_info: ticks * numer / denom = nanoseconds. Read from the machine, never assumed.
pub const Timebase = extern struct { numer: u32 = 1, denom: u32 = 1 };

pub fn machToNs(ticks: u64, tb: Timebase) u64 {
    if (tb.denom == 0) return 0;
    return @intCast(@as(u128, ticks) * tb.numer / tb.denom);
}

/// Nanojoules in one of the unit a channel names, null for a unit this build does not know.
pub fn unitNj(label: []const u8) ?u64 {
    const table = [_]struct { []const u8, u64 }{ .{ "nJ", 1 }, .{ "uJ", 1_000 }, .{ "\u{b5}J", 1_000 }, .{ "mJ", 1_000_000 }, .{ "J", 1_000_000_000 } };
    for (table) |row| if (std.mem.eql(u8, row[0], label)) return row[1];
    return null;
}

/// A channel's integer value in its own unit, as nanojoules; null for a negative value or an unknown unit.
pub fn toNj(value: i64, label: []const u8) ?u64 {
    if (value < 0) return null;
    const per = unitNj(label) orelse return null;
    return std.math.mul(u64, @intCast(value), per) catch null;
}

// ── Own CPU ─────────────────────────────────────────────────────────────

/// This process's CPU counters; each only grows. A null is a counter the OS did not give or that went backwards.
pub const Cpu = struct { nj: ?u64 = null, ns: ?u64 = null, instructions: ?u64 = null };

fn grew(a: ?u64, b: ?u64) ?u64 {
    const x = a orelse return null;
    const y = b orelse return null;
    return if (y >= x) y - x else null;
}

pub fn cpuDelta(a: Cpu, b: Cpu) Cpu {
    return .{ .nj = grew(a.nj, b.nj), .ns = grew(a.ns, b.ns), .instructions = grew(a.instructions, b.instructions) };
}

pub const Stamped = struct { t_ns: u64, cpu: Cpu };

fn stepAlong(a: ?u64, b: ?u64, f: f64) ?u64 {
    const x = a orelse return null;
    const step = grew(a, b) orelse return null;
    return x + @as(u64, @intFromFloat(@round(@as(f64, @floatFromInt(step)) * f)));
}

/// The counters at `t_ns`, interpolated linearly between the points that bracket it; null outside them.
/// `points` are sorted by time.
pub fn cpuAt(points: []const Stamped, t_ns: u64) ?Cpu {
    if (points.len == 0 or t_ns < points[0].t_ns or t_ns > points[points.len - 1].t_ns) return null;
    var lo: usize = 0;
    var hi: usize = points.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (points[mid].t_ns < t_ns) lo = mid + 1 else hi = mid;
    }
    const after = points[lo];
    if (after.t_ns == t_ns or lo == 0) return after.cpu;
    const before = points[lo - 1];
    const f = @as(f64, @floatFromInt(t_ns - before.t_ns)) / @as(f64, @floatFromInt(after.t_ns - before.t_ns));
    return .{
        .nj = stepAlong(before.cpu.nj, after.cpu.nj, f),
        .ns = stepAlong(before.cpu.ns, after.cpu.ns, f),
        .instructions = stepAlong(before.cpu.instructions, after.cpu.instructions, f),
    };
}

/// The CPU spent in `[from_ns, to_ns]`; null when the points do not cover it.
pub fn cpuWindow(points: []const Stamped, from_ns: u64, to_ns: u64) ?Cpu {
    const a = cpuAt(points, from_ns) orelse return null;
    const b = cpuAt(points, to_ns) orelse return null;
    return cpuDelta(a, b);
}

pub const PerToken = struct { cpu_mj: ?f64, cpu_ms: ?f64, instructions: ?f64 };

fn rate(v: ?u64, scale: f64, tokens: f64) ?f64 {
    return @as(f64, @floatFromInt(v orelse return null)) / scale / tokens;
}

/// Null for no tokens: a rate over nothing is not a reading.
pub fn perToken(d: Cpu, tokens: u64) ?PerToken {
    if (tokens == 0) return null;
    const n: f64 = @floatFromInt(tokens);
    return .{ .cpu_mj = rate(d.nj, 1e6, n), .cpu_ms = rate(d.ns, 1e6, n), .instructions = rate(d.instructions, 1, n) };
}

pub fn gpuMjPerToken(nj: u64, tokens: u64) ?f64 {
    if (tokens == 0) return null;
    return @as(f64, @floatFromInt(nj)) / 1e6 / @as(f64, @floatFromInt(tokens));
}

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn mach_timebase_info(info: *Timebase) c_int;
extern "c" fn getpid() c_int;

// `struct rusage_info_v6` (RUSAGE_INFO_V6 = 6) is 464 bytes; these are its u64 slots.
const rusage_v6_words = 58;
const slot_user_time = 2;
const slot_system_time = 3;
const slot_instructions = 31;
const slot_energy_nj = 42;

var timebase_cache: ?Timebase = null;

fn timebase() ?Timebase {
    if (timebase_cache) |t| return t;
    var t: Timebase = .{};
    if (mach_timebase_info(&t) != 0 or t.denom == 0) return null;
    timebase_cache = t;
    return t;
}

/// The counters of one `rusage_info_v6` record. A counter reading 0 is the OS not counting it, not an idle process;
/// with neither counter, or without the machine's timebase, there is no reading.
pub fn cpuFromRusage(ri: *const [rusage_v6_words]u64, tb: ?Timebase) ?Cpu {
    const base = tb orelse return null;
    if (base.denom == 0) return null;
    const nj = ri[slot_energy_nj];
    const instructions = ri[slot_instructions];
    if (nj == 0 and instructions == 0) return null;
    return .{
        .nj = if (nj == 0) null else nj,
        .ns = machToNs(ri[slot_user_time] +| ri[slot_system_time], base),
        .instructions = if (instructions == 0) null else instructions,
    };
}

/// This process's counters now; null where the OS gives none.
pub fn readCpu() ?Cpu {
    if (!is_macos) return null;
    var ri: [rusage_v6_words]u64 = undefined;
    if (proc_pid_rusage(getpid(), 6, &ri) != 0) return null;
    return cpuFromRusage(&ri, timebase());
}

/// Monotonic nanoseconds on the clock the timeline is stamped with.
pub fn nowNs(io: std.Io) u64 {
    return @intCast(std.Io.Timestamp.now(io, .boot).nanoseconds);
}

/// Reads this process's counters every `period_ms` on a thread, so a request can be cut at a phase boundary that is
/// only known afterwards. A reading is a syscall.
pub const CpuTimeline = struct {
    pub const period_ms = 5;

    gpa: std.mem.Allocator,
    io: std.Io,
    thread: std.Thread = undefined,
    mu: std.Io.Mutex = .init,
    points: std.ArrayList(Stamped) = .empty,
    stopping: std.atomic.Value(bool) = .init(false),
    read: *const fn () ?Cpu = readCpu,

    /// Fails where the counters cannot be read.
    pub fn start(gpa: std.mem.Allocator, io: std.Io) !*CpuTimeline {
        if (readCpu() == null) return error.CpuCountersUnavailable;
        const self = try gpa.create(CpuTimeline);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io };
        _ = self.mark();
        self.thread = try std.Thread.spawn(.{}, sampleMain, .{self});
        return self;
    }

    pub fn stop(self: *CpuTimeline) void {
        self.stopping.store(true, .release);
        self.thread.join();
        self.points.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    /// A reading now, so a window's two ends are exact instants; false when it could not be taken.
    pub fn mark(self: *CpuTimeline) bool {
        return self.add(false);
    }

    /// Like `mark`, and forgets every point before it: no window looks before the request it opens.
    pub fn begin(self: *CpuTimeline) bool {
        return self.add(true);
    }

    fn add(self: *CpuTimeline, forget_older: bool) bool {
        const t = nowNs(self.io);
        const cpu = self.read() orelse return false;
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (forget_older) {
            const keep = std.sort.partitionPoint(Stamped, self.points.items, t, startsBefore);
            const items = self.points.items;
            std.mem.copyForwards(Stamped, items[0 .. items.len - keep], items[keep..]);
            self.points.shrinkRetainingCapacity(items.len - keep);
        }
        var at = self.points.items.len;
        self.points.append(self.gpa, .{ .t_ns = t, .cpu = cpu }) catch return false;
        while (at > 0 and self.points.items[at - 1].t_ns > t) : (at -= 1) std.mem.swap(Stamped, &self.points.items[at - 1], &self.points.items[at]);
        return true;
    }

    fn startsBefore(t: u64, p: Stamped) bool {
        return p.t_ns < t;
    }

    pub fn window(self: *CpuTimeline, from_ns: u64, to_ns: u64) ?Cpu {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return cpuWindow(self.points.items, from_ns, to_ns);
    }

    fn sampleMain(self: *CpuTimeline) void {
        while (!self.stopping.load(.acquire)) {
            std.Io.sleep(self.io, .fromMilliseconds(period_ms), .real) catch {};
            _ = self.mark();
        }
    }
};

// ── GPU energy (IOReport, bound at run time) ────────────────────────────

pub const OpenError = error{ NoGroup, NoChannel, NoSubscription, UnknownUnit };

/// The GPU energy source: cumulative nanojoules since it was opened, null when one read failed.
pub const GpuSource = struct {
    ctx: ?*anyopaque = null,
    read: *const fn (ctx: ?*anyopaque) ?u64,
};

pub const Gpu = union(enum) { source: GpuSource, unavailable: []const u8 };

/// Resolves a symbol of the IOReport library by name; null when it has none.
pub const Loader = struct {
    ctx: ?*anyopaque = null,
    lookup: *const fn (ctx: ?*anyopaque, name: [:0]const u8) ?*anyopaque,
};

pub const symbol_names = [_][:0]const u8{
    "IOReportCopyChannelsInGroup",
    "IOReportCreateSubscription",
    "IOReportCreateSamples",
    "IOReportCreateSamplesDelta",
    "IOReportChannelGetChannelName",
    "IOReportChannelGetUnitLabel",
    "IOReportSimpleGetIntegerValue",
};

const missing_symbol_reasons = blk: {
    var out: [symbol_names.len][]const u8 = undefined;
    for (symbol_names, 0..) |n, i| out[i] = "libIOReport.dylib has no " ++ n;
    break :blk out;
};

pub const Raw = struct { ptrs: [symbol_names.len]*anyopaque };

pub const group_name = "Energy Model";
pub const channel_name = "GPU Energy";

fn openErrorReason(e: OpenError) []const u8 {
    return switch (e) {
        error.NoGroup => "IOReport has no '" ++ group_name ++ "' group",
        error.NoChannel => "the '" ++ group_name ++ "' group has no '" ++ channel_name ++ "' channel",
        error.NoSubscription => "IOReport refused a subscription to '" ++ channel_name ++ "'",
        error.UnknownUnit => "the '" ++ channel_name ++ "' channel reports in a unit this build does not know",
    };
}

/// Every step that can be absent leaves a reason and nothing else: a macOS without the library, a library without a
/// call, a machine without the group or the channel.
pub fn openGpu(loader: ?Loader, open_fn: *const fn (Raw) OpenError!GpuSource) Gpu {
    const l = loader orelse return .{ .unavailable = "libIOReport.dylib did not open" };
    var raw: Raw = undefined;
    for (symbol_names, 0..) |name, i| raw.ptrs[i] = l.lookup(l.ctx, name) orelse return .{ .unavailable = missing_symbol_reasons[i] };
    const source = open_fn(raw) catch |e| return .{ .unavailable = openErrorReason(e) };
    return .{ .source = source };
}

fn dlLookup(ctx: ?*anyopaque, name: [:0]const u8) ?*anyopaque {
    const lib: *std.DynLib = @ptrCast(@alignCast(ctx.?));
    return lib.lookup(*anyopaque, name);
}

var system_lib: std.DynLib = undefined;

/// The system's IOReport library; null where it does not open.
pub fn systemLoader() ?Loader {
    if (!is_macos) return null;
    system_lib = std.DynLib.open("/usr/lib/libIOReport.dylib") catch return null;
    return .{ .ctx = &system_lib, .lookup = dlLookup };
}

/// The GPU energy source of this machine, or why there is none.
pub fn openGpuLive() Gpu {
    return openGpu(systemLoader(), liveOpen);
}

const CF = ?*anyopaque;
const utf8: u32 = 0x08000100;
extern "c" const kCFTypeArrayCallBacks: anyopaque;
extern "c" fn CFStringCreateWithCString(alloc: CF, s: [*:0]const u8, enc: u32) CF;
extern "c" fn CFStringGetCString(s: CF, buf: [*]u8, len: isize, enc: u32) bool;
extern "c" fn CFDictionaryGetValue(d: CF, k: CF) CF;
extern "c" fn CFDictionarySetValue(d: CF, k: CF, v: CF) void;
extern "c" fn CFDictionaryCreateMutableCopy(alloc: CF, cap: isize, d: CF) CF;
extern "c" fn CFArrayGetCount(a: CF) isize;
extern "c" fn CFArrayGetValueAtIndex(a: CF, i: isize) CF;
extern "c" fn CFArrayCreateMutable(alloc: CF, cap: isize, callbacks: *const anyopaque) CF;
extern "c" fn CFArrayAppendValue(a: CF, v: CF) void;
extern "c" fn CFRelease(o: CF) void;

const Fns = struct {
    copyGroup: *const fn (CF, CF, u64, u64, u64) callconv(.c) CF,
    createSub: *const fn (CF, CF, *CF, u64, CF) callconv(.c) CF,
    samples: *const fn (CF, CF, CF) callconv(.c) CF,
    delta: *const fn (CF, CF, CF) callconv(.c) CF,
    chName: *const fn (CF) callconv(.c) CF,
    unitLabel: *const fn (CF) callconv(.c) CF,
    simpleInt: *const fn (CF, i32) callconv(.c) i64,
};

fn cfs(s: [*:0]const u8) CF {
    return CFStringCreateWithCString(null, s, utf8);
}

fn cfEquals(v: CF, text: []const u8) bool {
    const str = v orelse return false;
    var buf: [48]u8 = undefined;
    if (!CFStringGetCString(str, &buf, buf.len, utf8)) return false;
    return std.mem.eql(u8, std.mem.sliceTo(&buf, 0), text);
}

/// A channel's unit label; null for a channel that names none.
fn channelUnit(f: Fns, ch: CF, buf: []u8) ?[]const u8 {
    const label = f.unitLabel(ch) orelse return null;
    if (!CFStringGetCString(label, buf.ptr, @intCast(buf.len), utf8)) return null;
    return std.mem.sliceTo(buf, 0);
}

fn channelIndex(f: Fns, channels: CF) ?isize {
    var i: isize = 0;
    while (i < CFArrayGetCount(channels)) : (i += 1) {
        if (cfEquals(f.chName(CFArrayGetValueAtIndex(channels, i)), channel_name)) return i;
    }
    return null;
}

const Live = struct {
    fns: Fns,
    sub: CF,
    subbed: CF,
    base: CF,

    fn read(ctx: ?*anyopaque) ?u64 {
        const self: *Live = @ptrCast(@alignCast(ctx.?));
        const now = self.fns.samples(self.sub, self.subbed, null) orelse return null;
        defer CFRelease(now);
        const diff = self.fns.delta(self.base, now, null) orelse return null;
        defer CFRelease(diff);
        const key = cfs("IOReportChannels");
        defer CFRelease(key);
        const channels = CFDictionaryGetValue(diff, key) orelse return null;
        const at = channelIndex(self.fns, channels) orelse return null;
        const ch = CFArrayGetValueAtIndex(channels, at);
        var unit: [16]u8 = undefined;
        const label = channelUnit(self.fns, ch, &unit) orelse return null;
        return toNj(self.fns.simpleInt(ch, 0), label);
    }
};

fn liveOpen(raw: Raw) OpenError!GpuSource {
    const f: Fns = .{
        .copyGroup = @ptrCast(@alignCast(raw.ptrs[0])),
        .createSub = @ptrCast(@alignCast(raw.ptrs[1])),
        .samples = @ptrCast(@alignCast(raw.ptrs[2])),
        .delta = @ptrCast(@alignCast(raw.ptrs[3])),
        .chName = @ptrCast(@alignCast(raw.ptrs[4])),
        .unitLabel = @ptrCast(@alignCast(raw.ptrs[5])),
        .simpleInt = @ptrCast(@alignCast(raw.ptrs[6])),
    };
    const group = cfs(group_name);
    defer CFRelease(group);
    const all = f.copyGroup(group, null, 0, 0, 0) orelse return error.NoGroup;
    defer CFRelease(all);
    const key = cfs("IOReportChannels");
    defer CFRelease(key);
    const channels = CFDictionaryGetValue(all, key) orelse return error.NoChannel;
    const at = channelIndex(f, channels) orelse return error.NoChannel;
    const ch = CFArrayGetValueAtIndex(channels, at);
    var unit: [16]u8 = undefined;
    const label = channelUnit(f, ch, &unit) orelse return error.UnknownUnit;
    if (unitNj(label) == null) return error.UnknownUnit;
    // A subscription to this channel alone: a read then snapshots one channel, not the group.
    const kept = CFArrayCreateMutable(null, 1, &kCFTypeArrayCallBacks) orelse return error.NoSubscription;
    defer CFRelease(kept);
    CFArrayAppendValue(kept, ch);
    const pruned = CFDictionaryCreateMutableCopy(null, 0, all) orelse return error.NoSubscription;
    defer CFRelease(pruned);
    CFDictionarySetValue(pruned, key, kept);
    var subbed: CF = null;
    const sub = f.createSub(null, pruned, &subbed, 0, null) orelse return error.NoSubscription;
    const base = f.samples(sub, subbed, null) orelse return error.NoSubscription;
    const live = std.heap.page_allocator.create(Live) catch return error.NoSubscription;
    live.* = .{ .fns = f, .sub = sub, .subbed = subbed, .base = base };
    return .{ .ctx = live, .read = Live.read };
}

// ── Tests ───────────────────────────────────────────────────────────────

test "energy units: nJ, uJ, mJ and J scale to nanojoules, anything else is refused" {
    try testing.expectEqual(@as(?u64, 5), toNj(5, "nJ"));
    try testing.expectEqual(@as(?u64, 5_000), toNj(5, "uJ"));
    try testing.expectEqual(@as(?u64, 5_000_000), toNj(5, "mJ"));
    try testing.expectEqual(@as(?u64, 5_000_000_000), toNj(5, "J"));
    try testing.expectEqual(@as(?u64, null), toNj(5, "kWh"));
    try testing.expectEqual(@as(?u64, null), toNj(5, ""));
    try testing.expectEqual(@as(?u64, null), toNj(-1, "nJ"));
    try testing.expectEqual(@as(?u64, 0), toNj(0, "mJ"));
}

test "energy mach time: ticks scale by the machine's own timebase" {
    try testing.expectEqual(@as(u64, 1_000), machToNs(1_000, .{ .numer = 1, .denom = 1 }));
    try testing.expectEqual(@as(u64, 41_666), machToNs(1_000, .{ .numer = 125, .denom = 3 }));
    try testing.expectEqual(@as(u64, 0), machToNs(1_000, .{ .numer = 1, .denom = 0 }));
    try testing.expectEqual(@as(u64, 12_500_000_000_000_000_000), machToNs(300_000_000_000_000_000, .{ .numer = 125, .denom = 3 }));
}

test "energy delta: counters subtract, one that did not move reads 0, one that went back or is missing is null" {
    const a: Cpu = .{ .nj = 100, .ns = 50, .instructions = 1_000 };
    const b: Cpu = .{ .nj = 350, .ns = 50, .instructions = 4_000 };
    const d = cpuDelta(a, b);
    try testing.expectEqual(@as(?u64, 250), d.nj);
    try testing.expectEqual(@as(?u64, 0), d.ns);
    try testing.expectEqual(@as(?u64, 3_000), d.instructions);
    const back = cpuDelta(b, a);
    try testing.expectEqual(@as(?u64, null), back.nj);
    try testing.expectEqual(@as(?u64, 0), back.ns);
    try testing.expectEqual(@as(?u64, null), back.instructions);
    const missing = cpuDelta(.{ .nj = null, .ns = 1, .instructions = 1 }, .{ .nj = 5, .ns = 2, .instructions = null });
    try testing.expectEqual(@as(?u64, null), missing.nj);
    try testing.expectEqual(@as(?u64, 1), missing.ns);
    try testing.expectEqual(@as(?u64, null), missing.instructions);
}

const line = [_]Stamped{
    .{ .t_ns = 1_000, .cpu = .{ .nj = 0, .ns = 0, .instructions = 0 } },
    .{ .t_ns = 2_000, .cpu = .{ .nj = 1_000, .ns = 400, .instructions = 10_000 } },
    .{ .t_ns = 4_000, .cpu = .{ .nj = 1_000, .ns = 400, .instructions = 10_000 } },
};

test "energy timeline: a time between two points is interpolated, a point is exact, outside the points is not a reading" {
    try testing.expectEqual(@as(?u64, 500), cpuAt(&line, 1_500).?.nj);
    try testing.expectEqual(@as(?u64, 1_000), cpuAt(&line, 2_000).?.nj);
    try testing.expectEqual(@as(?u64, 0), cpuAt(&line, 1_000).?.nj);
    try testing.expectEqual(@as(?u64, 1_000), cpuAt(&line, 4_000).?.nj);
    try testing.expectEqual(@as(?Cpu, null), cpuAt(&line, 999));
    try testing.expectEqual(@as(?Cpu, null), cpuAt(&line, 4_001));
    try testing.expectEqual(@as(?Cpu, null), cpuAt(&.{}, 10));
}

test "energy timeline: a window is the difference of its two ends, an idle stretch is 0, an uncovered one is null" {
    const w = cpuWindow(&line, 1_500, 2_000).?;
    try testing.expectEqual(@as(?u64, 500), w.nj);
    try testing.expectEqual(@as(?u64, 5_000), w.instructions);
    const idle = cpuWindow(&line, 2_500, 3_500).?;
    try testing.expectEqual(@as(?u64, 0), idle.nj);
    try testing.expectEqual(@as(?u64, 0), idle.instructions);
    try testing.expectEqual(@as(?Cpu, null), cpuWindow(&line, 500, 2_000));
    try testing.expectEqual(@as(?Cpu, null), cpuWindow(&line, 2_000, 9_000));
    try testing.expectEqual(@as(?u64, 0), cpuWindow(&line, 3_000, 2_000).?.nj);
}

test "energy per token: millijoules, milliseconds and instructions divide by the tokens, no tokens is no reading" {
    const d: Cpu = .{ .nj = 12_600_000, .ns = 3_500_000, .instructions = 40_000_000 };
    const p = perToken(d, 1).?;
    try testing.expectApproxEqAbs(@as(f64, 12.6), p.cpu_mj.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 3.5), p.cpu_ms.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 40_000_000), p.instructions.?, 1e-6);
    const q = perToken(d, 4).?;
    try testing.expectApproxEqAbs(@as(f64, 3.15), q.cpu_mj.?, 1e-9);
    const half = perToken(.{ .nj = null, .ns = 3_500_000, .instructions = 40_000_000 }, 1).?;
    try testing.expectEqual(@as(?f64, null), half.cpu_mj);
    try testing.expect(half.cpu_ms != null and half.instructions != null);
    try testing.expectEqual(@as(?PerToken, null), perToken(d, 0));
    try testing.expectApproxEqAbs(@as(f64, 1262), gpuMjPerToken(1_262_000_000_000, 1000).?, 1e-9);
    try testing.expectEqual(@as(?f64, null), gpuMjPerToken(5, 0));
}

const fake_ptr: *anyopaque = @ptrFromInt(0x1000);

fn lookupAll(ctx: ?*anyopaque, name: [:0]const u8) ?*anyopaque {
    _ = ctx;
    _ = name;
    return fake_ptr;
}

var skipped: []const u8 = "";

fn lookupExcept(ctx: ?*anyopaque, name: [:0]const u8) ?*anyopaque {
    _ = ctx;
    return if (std.mem.eql(u8, name, skipped)) null else fake_ptr;
}

fn readFake(ctx: ?*anyopaque) ?u64 {
    _ = ctx;
    return 7;
}

fn openOk(raw: Raw) OpenError!GpuSource {
    _ = raw;
    return .{ .read = readFake };
}

fn openNoGroup(raw: Raw) OpenError!GpuSource {
    _ = raw;
    return error.NoGroup;
}

fn openNoChannel(raw: Raw) OpenError!GpuSource {
    _ = raw;
    return error.NoChannel;
}

test "energy GPU, library present: the source opens and reads" {
    const g = openGpu(.{ .lookup = lookupAll }, openOk);
    try testing.expectEqual(@as(?u64, 7), g.source.read(g.source.ctx));
}

test "energy GPU, library missing: unavailable with the reason, nothing else fails" {
    const g = openGpu(null, openOk);
    try testing.expect(std.mem.indexOf(u8, g.unavailable, "libIOReport") != null);
}

test "energy GPU, any one symbol missing: unavailable and the reason names that symbol" {
    for (symbol_names) |name| {
        skipped = name;
        const g = openGpu(.{ .lookup = lookupExcept }, openOk);
        try testing.expect(std.mem.indexOf(u8, g.unavailable, name) != null);
    }
    skipped = "";
}

test "energy GPU, group or channel missing: unavailable with its own reason" {
    const no_group = openGpu(.{ .lookup = lookupAll }, openNoGroup);
    try testing.expect(std.mem.indexOf(u8, no_group.unavailable, "Energy Model") != null);
    const no_channel = openGpu(.{ .lookup = lookupAll }, openNoChannel);
    try testing.expect(std.mem.indexOf(u8, no_channel.unavailable, "GPU Energy") != null);
    try testing.expect(!std.mem.eql(u8, no_group.unavailable, no_channel.unavailable));
}

test "energy live: own CPU counters and the GPU source never go backwards between two reads" {
    if (!is_macos) return error.SkipZigTest;
    const a = readCpu() orelse return error.SkipZigTest;
    var spin: u64 = 0;
    for (0..2_000_000) |i| spin +%= i * i;
    std.mem.doNotOptimizeAway(spin);
    const b = readCpu() orelse return error.TestUnexpectedResult;
    try testing.expect(cpuDelta(a, b).ns != null);
    try testing.expect(cpuDelta(a, b).instructions == null or cpuDelta(a, b).instructions.? > 0);
    switch (openGpuLive()) {
        .source => |g| {
            const x = g.read(g.ctx) orelse return error.TestUnexpectedResult;
            const y = g.read(g.ctx) orelse return error.TestUnexpectedResult;
            try testing.expect(y >= x);
        },
        .unavailable => {},
    }
}

test "energy timeline: a started timeline answers for a window it covers and not for one it does not" {
    if (!is_macos) return error.SkipZigTest;
    const io = testing.io;
    const tl = CpuTimeline.start(testing.allocator, io) catch return error.SkipZigTest;
    defer tl.stop();
    const t0 = nowNs(io);
    _ = tl.mark();
    var spin: u64 = 0;
    for (0..4_000_000) |i| spin +%= i * i;
    std.mem.doNotOptimizeAway(spin);
    const t1 = nowNs(io);
    _ = tl.mark();
    try testing.expect(tl.window(t0, t1) != null);
    try testing.expect(tl.window(t0, t1 + 60 * std.time.ns_per_s) == null);
}

test "energy interpolation: a counter that is missing or went back between two points has no value in between" {
    const gap = [_]Stamped{
        .{ .t_ns = 1_000, .cpu = .{ .nj = null, .ns = 0, .instructions = 500 } },
        .{ .t_ns = 2_000, .cpu = .{ .nj = 1_000, .ns = 400, .instructions = 100 } },
    };
    const mid = cpuAt(&gap, 1_500).?;
    try testing.expectEqual(@as(?u64, null), mid.nj);
    try testing.expectEqual(@as(?u64, 200), mid.ns);
    try testing.expectEqual(@as(?u64, null), mid.instructions);
}

fn rusage(nj: u64, instructions: u64) [rusage_v6_words]u64 {
    var ri: [rusage_v6_words]u64 = @splat(0);
    ri[slot_user_time] = 3_000;
    ri[slot_system_time] = 1_000;
    ri[slot_energy_nj] = nj;
    ri[slot_instructions] = instructions;
    return ri;
}

test "energy rusage: no timebase is no reading, and a counter that reads 0 is null on its own" {
    const tb: Timebase = .{ .numer = 125, .denom = 3 };
    try testing.expectEqual(@as(?Cpu, null), cpuFromRusage(&rusage(10, 20), null));
    try testing.expectEqual(@as(?Cpu, null), cpuFromRusage(&rusage(10, 20), .{ .numer = 1, .denom = 0 }));
    const both = cpuFromRusage(&rusage(10, 20), tb).?;
    try testing.expectEqual(@as(?u64, 10), both.nj);
    try testing.expectEqual(@as(?u64, 20), both.instructions);
    try testing.expectEqual(@as(?u64, 166_666), both.ns);
    const no_energy = cpuFromRusage(&rusage(0, 20), tb).?;
    try testing.expectEqual(@as(?u64, null), no_energy.nj);
    try testing.expectEqual(@as(?u64, 20), no_energy.instructions);
    const no_instructions = cpuFromRusage(&rusage(10, 0), tb).?;
    try testing.expectEqual(@as(?u64, 10), no_instructions.nj);
    try testing.expectEqual(@as(?u64, null), no_instructions.instructions);
    try testing.expectEqual(@as(?Cpu, null), cpuFromRusage(&rusage(0, 0), tb));
}

var next_nj: u64 = 0;

fn countingRead() ?Cpu {
    next_nj += 10;
    return .{ .nj = next_nj, .ns = next_nj, .instructions = next_nj };
}

fn failingRead() ?Cpu {
    return null;
}

test "energy timeline: begin forgets the points before it, and a mark that cannot read says so" {
    const io = testing.io;
    var tl: CpuTimeline = .{ .gpa = testing.allocator, .io = io, .read = countingRead };
    defer tl.points.deinit(testing.allocator);
    for (0..50) |_| try testing.expect(tl.mark());
    try testing.expectEqual(@as(usize, 50), tl.points.items.len);
    std.Io.sleep(io, .fromMilliseconds(2), .real) catch {};
    try testing.expect(tl.begin());
    try testing.expectEqual(@as(usize, 1), tl.points.items.len);
    std.Io.sleep(io, .fromMilliseconds(2), .real) catch {};
    try testing.expect(tl.mark());
    const from = tl.points.items[0].t_ns;
    try testing.expect(tl.window(from, tl.points.items[1].t_ns) != null);
    tl.read = failingRead;
    try testing.expect(!tl.mark());
    try testing.expect(!tl.begin());
    try testing.expectEqual(@as(usize, 2), tl.points.items.len);
}

// A stand-in for IOReport: real CoreFoundation containers around fake channels, so the bound-at-run-time path runs
// as it does live.
extern "c" const kCFTypeDictionaryKeyCallBacks: anyopaque;
extern "c" const kCFTypeDictionaryValueCallBacks: anyopaque;
extern "c" fn CFDictionaryCreateMutable(alloc: CF, cap: isize, key_cb: *const anyopaque, value_cb: *const anyopaque) CF;

const fake_io = struct {
    var elems: [2]CF = .{ null, null };
    var names: [2]?[*:0]const u8 = .{ null, null };
    var units: [2]?[*:0]const u8 = .{ null, null };

    fn set(n0: ?[*:0]const u8, n1: ?[*:0]const u8, ua: ?[*:0]const u8, ub: ?[*:0]const u8) void {
        elems = .{ cfs("channel 0"), cfs("channel 1") };
        names = .{ n0, n1 };
        units = .{ ua, ub };
    }

    fn indexOf(ch: CF) usize {
        return if (ch == elems[0]) 0 else 1;
    }

    fn group(_: CF, _: CF, _: u64, _: u64, _: u64) callconv(.c) CF {
        const d = CFDictionaryCreateMutable(null, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        const a = CFArrayCreateMutable(null, 2, &kCFTypeArrayCallBacks);
        CFArrayAppendValue(a, elems[0]);
        CFArrayAppendValue(a, elems[1]);
        CFDictionarySetValue(d, cfs("IOReportChannels"), a);
        return d;
    }

    fn sub(_: CF, _: CF, subbed: *CF, _: u64, _: CF) callconv(.c) CF {
        subbed.* = cfs("subscribed");
        return cfs("subscription");
    }

    fn sample(_: CF, _: CF, _: CF) callconv(.c) CF {
        return cfs("sample");
    }

    fn delta(_: CF, _: CF, _: CF) callconv(.c) CF {
        return group(null, null, 0, 0, 0);
    }

    fn name(ch: CF) callconv(.c) CF {
        const n = names[indexOf(ch)] orelse return null;
        return cfs(n);
    }

    fn unit(ch: CF) callconv(.c) CF {
        const u = units[indexOf(ch)] orelse return null;
        return cfs(u);
    }

    fn value(_: CF, _: i32) callconv(.c) i64 {
        return 5;
    }

    fn lookup(_: ?*anyopaque, symbol: [:0]const u8) ?*anyopaque {
        inline for (symbol_names, .{ &group, &sub, &sample, &delta, &name, &unit, &value }) |n, f| {
            if (std.mem.eql(u8, symbol, n)) return @constCast(@ptrCast(f));
        }
        return null;
    }

    fn open() Gpu {
        return openGpu(.{ .lookup = lookup }, liveOpen);
    }
};

test "energy IOReport: a channel without a name is skipped and the next one is read" {
    if (!is_macos) return error.SkipZigTest;
    fake_io.set(null, "GPU Energy", "nJ", "uJ");
    const g = fake_io.open();
    const source = switch (g) {
        .source => |s| s,
        .unavailable => return error.TestUnexpectedResult,
    };
    try testing.expectEqual(@as(?u64, 5_000), source.read(source.ctx));
}

test "energy IOReport: no channel has a name, the reason is the missing channel" {
    if (!is_macos) return error.SkipZigTest;
    fake_io.set(null, null, "nJ", "nJ");
    const g = fake_io.open();
    try testing.expect(std.mem.indexOf(u8, g.unavailable, "no 'GPU Energy' channel") != null);
}

test "energy IOReport: a channel without a unit is refused at open with the unit as the reason" {
    if (!is_macos) return error.SkipZigTest;
    fake_io.set("GPU Energy", null, null, "nJ");
    const g = fake_io.open();
    try testing.expect(std.mem.indexOf(u8, g.unavailable, "unit") != null);
}

test "energy IOReport: a unit that goes missing after open reads null" {
    if (!is_macos) return error.SkipZigTest;
    fake_io.set("GPU Energy", null, "nJ", "nJ");
    const source = switch (fake_io.open()) {
        .source => |s| s,
        .unavailable => return error.TestUnexpectedResult,
    };
    try testing.expectEqual(@as(?u64, 5), source.read(source.ctx));
    fake_io.units[0] = null;
    try testing.expectEqual(@as(?u64, null), source.read(source.ctx));
}
