//! The box under a timed run, shared by `sushi bench` and `sushi-ubench`: fans and die temperature, the power
//! source, the process priority, other GPU jobs, builds, the GPU lock and the QUIET marker. MLX-free.
//! Docs: process-measurement.md#quiet-box.
const std = @import("std");
const io_util = @import("io_util.zig");
const testing = std.testing;

/// The target and the cap of the idle between the arms; the same rule `tests/bench.sh` applies before a boot.
pub const cooldown_target_c: f64 = 50;
pub const cooldown_cap_s: u32 = 180;
const cooldown_poll_s: u32 = 10;

pub const Cooldown = struct {
    waited_s: u32,
    start_c: ?f64 = null,
    end_c: ?f64 = null,
};

/// Without a first reading (no `fanctl`) there is nothing to wait for: the idle is skipped, not a fixed wait.
pub fn cooldownSkipped(start_c: ?f64) bool {
    return start_c == null;
}

/// Once readings come, a missing one means only the cap ends the wait.
pub fn cooldownDone(die_c: ?f64, waited_s: u32) bool {
    if (waited_s >= cooldown_cap_s) return true;
    return if (die_c) |t| t < cooldown_target_c else false;
}

/// The hottest `topTemps` sensor of a `fanctl status` document; null when it holds none or is not that document.
pub fn hottestDie(allocator: std.mem.Allocator, status: []const u8) ?f64 {
    const parsed = std.json.parseFromSlice(struct { topTemps: []const struct { celsius: f64 } }, allocator, status, .{ .ignore_unknown_fields = true }) catch return null;
    defer parsed.deinit();
    var hottest: ?f64 = null;
    for (parsed.value.topTemps) |t| hottest = if (hottest) |h| @max(h, t.celsius) else t.celsius;
    return hottest;
}

pub fn runOut(a: std.mem.Allocator, io: std.Io, argv: []const []const u8) ?[]u8 {
    const r = std.process.run(a, io, .{ .argv = argv, .stdout_limit = .limited(1 << 20) }) catch return null;
    a.free(r.stderr);
    const ok = switch (r.term) {
        .exited => |code| code == 0,
        else => false,
    };
    if (ok) return r.stdout;
    a.free(r.stdout);
    return null;
}

pub fn readDie(a: std.mem.Allocator, io: std.Io) ?f64 {
    const out = runOut(a, io, &.{ "fanctl", "status" }) orelse return null;
    defer a.free(out);
    return hottestDie(a, out);
}

/// Fans to max (a request lasts 15 minutes, so a long run renews it), then idles until the hottest die is under
/// the target or the cap passes.
pub fn cooldown(a: std.mem.Allocator, io: std.Io) Cooldown {
    fansMax(a, io);
    var clock = io_util.Stopwatch.init(io);
    const start_c = readDie(a, io);
    if (cooldownSkipped(start_c)) return .{ .waited_s = 0 };
    var die = start_c;
    while (true) {
        const waited: u32 = @intCast(clock.read() / std.time.ns_per_s);
        if (cooldownDone(die, waited)) break;
        std.Io.sleep(io, .fromSeconds(cooldown_poll_s), .real) catch {};
        die = readDie(a, io);
    }
    const waited: u32 = @intCast(clock.read() / std.time.ns_per_s);
    return .{ .waited_s = waited, .start_c = start_c, .end_c = die };
}

pub fn fansMax(a: std.mem.Allocator, io: std.Io) void {
    if (runOut(a, io, &.{ "fanctl", "max", "--ttl", "900" })) |out| a.free(out);
}

pub fn fansAuto(a: std.mem.Allocator, io: std.Io) void {
    if (runOut(a, io, &.{ "fanctl", "auto" })) |out| a.free(out);
}

pub const Power = enum { ac, battery, unknown };

pub fn parsePower(pmset: []const u8) Power {
    if (std.mem.indexOf(u8, pmset, "'AC Power'") != null) return .ac;
    if (std.mem.indexOf(u8, pmset, "'Battery Power'") != null) return .battery;
    return .unknown;
}

pub fn power(a: std.mem.Allocator, io: std.Io) Power {
    const out = runOut(a, io, &.{ "pmset", "-g", "batt" }) orelse return .unknown;
    defer a.free(out);
    return parsePower(out);
}

/// `ps -o pri=` for one process: 31 is the base priority of an interactive launch, 4 is what an agent harness
/// or `taskpolicy -b` leaves, which never reaches the Super cores.
pub fn parsePri(ps: []const u8) ?i32 {
    return std.fmt.parseInt(i32, std.mem.trim(u8, ps, " \t\r\n"), 10) catch null;
}

pub const interactive_pri: i32 = 31;

pub fn priority(a: std.mem.Allocator, io: std.Io) ?i32 {
    var pid_buf: [16]u8 = undefined;
    const pid = std.fmt.bufPrint(&pid_buf, "{d}", .{std.c.getpid()}) catch return null;
    const out = runOut(a, io, &.{ "ps", "-o", "pri=", "-p", pid }) orelse return null;
    defer a.free(out);
    return parsePri(out);
}

/// The processes that may not share the GPU with a timed run, the list `tests/bench.sh` refuses on.
pub fn liveGpuJobs(a: std.mem.Allocator, io: std.Io) ?[]u8 {
    const out = runOut(a, io, &.{ "pgrep", "-lx", "sushi|sushi-repack|sushi-grid" }) orelse return null;
    if (std.mem.trim(u8, out, " \r\n").len == 0) {
        a.free(out);
        return null;
    }
    return out;
}

pub fn buildsRunning(a: std.mem.Allocator, io: std.Io) bool {
    const out = runOut(a, io, &.{ "pgrep", "-x", "zig|clang|clang\\+\\+|ld" }) orelse return false;
    defer a.free(out);
    return std.mem.trim(u8, out, " \r\n").len > 0;
}

/// What the quiet box needs to be true before a timed run; the first thing wrong is the refusal.
pub const Facts = struct {
    stale: ?[]const u8 = null,
    dirty: bool = false,
    pri: ?i32 = null,
    gpu_jobs: ?[]const u8 = null,
    power: Power = .ac,
};

pub fn firstRefusal(f: Facts, buf: []u8) ?[]const u8 {
    if (f.stale) |why| return std.fmt.bufPrint(buf, "stale binary: {s}; rebuild with `zig build ubench`", .{why}) catch why;
    if (f.dirty) return "built from a dirty tree: commit, then rebuild with `zig build ubench` (or --contended for a look)";
    if (f.pri) |p| if (p < interactive_pri) return std.fmt.bufPrint(buf, "base priority {d} is below {d}: run it under `taskpolicy -a sushi-ubench ...`", .{ p, interactive_pri }) catch "base priority too low";
    if (f.gpu_jobs) |jobs| return std.fmt.bufPrint(buf, "another GPU job is live ({s})", .{std.mem.trim(u8, jobs, " \r\n")}) catch "another GPU job is live";
    if (f.power != .ac) return "not on AC power";
    return null;
}

/// Pid of a `gpu-lock.sh acquire` in progress, for a signal handler that must end the wait.
pub var acquiring_pid: std.atomic.Value(i32) = .init(0);

pub fn lockAcquire(io: std.Io, script: []const u8, owner: []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = &.{ script, "acquire", owner }, .stdin = .ignore });
    acquiring_pid.store(child.id.?, .release);
    defer acquiring_pid.store(0, .release);
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code == 0) return,
        else => {},
    }
    return error.LockNotAcquired;
}

pub fn lockRelease(io: std.Io, script: []const u8, owner: []const u8) void {
    var child = std.process.spawn(io, .{ .argv = &.{ script, "release", owner }, .stdin = .ignore }) catch return;
    _ = child.wait(io) catch {};
}

/// `~/.sushi/runs/QUIET`: while it exists no worker starts a build or test.
pub fn quietPath(a: std.mem.Allocator) ![]u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoHome);
    return std.fs.path.join(a, &.{ home, ".sushi", "runs", "QUIET" });
}

pub fn quietTouch(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |dir| try std.Io.Dir.cwd().createDirPath(io, dir);
    const f = try std.Io.Dir.cwd().createFile(io, path, .{});
    f.close(io);
}

pub fn quietRemove(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
}

// ── Tests ───────────────────────────────────────────────────────────────

test "box: the power source, and the process priority, read from the tools' text" {
    try testing.expectEqual(Power.ac, parsePower("Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged"));
    try testing.expectEqual(Power.battery, parsePower("Now drawing from 'Battery Power'\n"));
    try testing.expectEqual(Power.unknown, parsePower(""));
    try testing.expectEqual(@as(?i32, 31), parsePri(" 31\n"));
    try testing.expectEqual(@as(?i32, 4), parsePri("4"));
    try testing.expectEqual(@as(?i32, null), parsePri(""));
}

test "box: the first thing wrong is the refusal, and a clean box has none" {
    var buf: [256]u8 = undefined;
    try testing.expectEqual(@as(?[]const u8, null), firstRefusal(.{ .pri = 31 }, &buf));
    try testing.expectEqual(@as(?[]const u8, null), firstRefusal(.{}, &buf));
    try testing.expect(std.mem.indexOf(u8, firstRefusal(.{ .stale = "built abc, HEAD def", .dirty = true }, &buf).?, "stale binary") != null);
    try testing.expect(std.mem.indexOf(u8, firstRefusal(.{ .dirty = true, .pri = 4 }, &buf).?, "dirty tree") != null);
    try testing.expect(std.mem.indexOf(u8, firstRefusal(.{ .pri = 4, .gpu_jobs = "1 sushi" }, &buf).?, "taskpolicy -a") != null);
    try testing.expect(std.mem.indexOf(u8, firstRefusal(.{ .pri = 31, .gpu_jobs = "812 sushi\n" }, &buf).?, "812 sushi") != null);
    try testing.expect(std.mem.indexOf(u8, firstRefusal(.{ .power = .battery }, &buf).?, "AC power") != null);
}

test "bench cooldown: no first reading skips the idle, any reading keeps it" {
    try testing.expect(cooldownSkipped(null));
    try testing.expect(!cooldownSkipped(30));
    try testing.expect(!cooldownSkipped(87));
}

test "bench cooldown: stops under 50 C, is capped at 180 s, and waits the full cap without a reading" {
    try testing.expect(cooldownDone(49.9, 0));
    try testing.expect(!cooldownDone(50.0, 0));
    try testing.expect(!cooldownDone(87, 170));
    try testing.expect(cooldownDone(87, 180));
    try testing.expect(!cooldownDone(null, 0));
    try testing.expect(!cooldownDone(null, 170));
    try testing.expect(cooldownDone(null, 180));
}

test "bench cooldown: the hottest die in a fanctl status document, and nothing from a bad one" {
    const status =
        \\{
        \\  "fans" : [ { "actualRPM" : 5356, "index" : 0, "mode" : 0 } ],
        \\  "state" : "max",
        \\  "topTemps" : [
        \\    { "celsius" : 55.165626525878906, "key" : "TCMb" },
        \\    { "celsius" : 87, "key" : "TCHP" },
        \\    { "celsius" : 47.078125, "key" : "TCDX" }
        \\  ]
        \\}
    ;
    try testing.expectEqual(@as(f64, 87), hottestDie(testing.allocator, status).?);
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "{\"topTemps\":[]}"));
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "not json"));
    try testing.expectEqual(@as(?f64, null), hottestDie(testing.allocator, "{\"state\":\"max\"}"));
}
