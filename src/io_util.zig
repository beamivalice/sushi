//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.

const std = @import("std");

/// Seconds since the Unix epoch (wall-clock).
pub fn nowSecs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toSeconds();
}

/// Milliseconds since the Unix epoch (wall-clock).
pub fn nowMs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toMilliseconds();
}

/// Milliseconds on the MONOTONIC `boot` clock (arbitrary epoch — only
/// differences are meaningful). Use for deadlines/intervals, never for
/// timestamps a client sees: an NTP step or a manual clock change must not be
/// able to stall (or spam) a timer. `.boot` counts across pmset sleep, so a
/// deadline that expired while the lid was shut fires immediately on wake.
pub fn nowMsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMilliseconds();
}

/// Drop-in replacement for `std.time.Timer` — uses the `boot` clock, which
/// counts wall-time across pmset sleep (vs `.awake`, which stops during
/// suspend). Important for long-running stopwatches that span a lid close.
/// Resolution is set by the `Io` implementation.
pub const Stopwatch = struct {
    io: std.Io,
    started_at: std.Io.Timestamp,

    pub fn init(io: std.Io) Stopwatch {
        return .{ .io = io, .started_at = std.Io.Timestamp.now(io, .boot) };
    }

    /// Nanoseconds since `init` (or last `reset`).
    pub fn read(s: Stopwatch) u64 {
        return @intCast(s.started_at.untilNow(s.io, .boot).nanoseconds);
    }

    /// Reset the start point to "now".
    pub fn reset(s: *Stopwatch) void {
        s.started_at = std.Io.Timestamp.now(s.io, .boot);
    }
};

/// Clock ties and rollback must not alias independent response envelopes.
pub fn nextCompletionId(counter: *std.atomic.Value(i64), now_ms: i64) i64 {
    var previous = counter.load(.monotonic);
    while (true) {
        const next = @max(now_ms, previous + 1);
        previous = counter.cmpxchgWeak(previous, next, .monotonic, .monotonic) orelse return next;
    }
}

var call_id_seq = std.atomic.Value(u64).init(0);

/// A tool call's id, `<prefix>_<16 random hex digits>_<per-process count>`: unique across the requests running
/// together, and across restarts, where a clock tick (the id's old part) repeats. The one maker of every surface's call id.
pub fn callId(io: std.Io, allocator: std.mem.Allocator, prefix: []const u8) std.mem.Allocator.Error![]u8 {
    var random: [8]u8 = undefined;
    io.random(&random);
    const seq = call_id_seq.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "{s}_{x:0>16}_{x}", .{ prefix, std.mem.readInt(u64, &random, .little), seq });
}

/// Diagnostic switches are off for empty values and values beginning with `0`.
pub fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

pub fn diagEnvOn(name: [*:0]const u8) bool {
    return diagEnvValueOn(std.c.getenv(name));
}
/// Cache the effective switch once.
pub fn diagEnvOnCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = diagEnvOn(name);
    cache.* = v;
    return v;
}

/// Reads exactly `dst.len` bytes at `offset`, retrying an interrupted read.
pub fn readExact(fd: std.c.fd_t, dst: []u8, offset: u64) !void {
    var done: usize = 0;
    while (done < dst.len) {
        const got = std.c.pread(fd, dst[done..].ptr, dst.len - done, @intCast(offset + done));
        if (got < 0) {
            if (std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
            return error.FillReadFailed;
        }
        if (got == 0) return error.FillShortRead;
        done += @intCast(got);
    }
}

const IdBatch = struct {
    ids: [2000][]u8 = undefined,
    made: usize = 0,

    fn run(self: *IdBatch, prefix: []const u8) void {
        for (&self.ids) |*slot| {
            slot.* = callId(std.testing.io, std.testing.allocator, prefix) catch return;
            self.made += 1;
        }
    }
};

/// Makes the directory `path` and its missing parents. Unlike `std.Io.Dir.createDirPath`, which refuses a path whose last
/// component is a symlink (`NotDir`), it takes one that leads to a directory as made: a user's `~/.sushi` or cache
/// folder linked to another volume. A dangling link or a file there is still `NotDir`.
pub fn makeDirPath(io: std.Io, path: []const u8) !void {
    std.Io.Dir.cwd().createDirPath(io, path) catch |e| switch (e) {
        error.NotDir => {
            // Opening follows the link: only a folder at its end makes the path.
            var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return e;
            dir.close(io);
        },
        else => return e,
    };
}

test "tool call ids made by concurrent requests, in the same millisecond or not, never repeat" {
    var a: IdBatch = .{};
    var b: IdBatch = .{};
    const ta = try std.Thread.spawn(.{}, IdBatch.run, .{ &a, "call" });
    const tb = try std.Thread.spawn(.{}, IdBatch.run, .{ &b, "call" });
    ta.join();
    tb.join();
    defer for ([_]*IdBatch{ &a, &b }) |batch| for (batch.ids[0..batch.made]) |id| std.testing.allocator.free(id);
    try std.testing.expectEqual(a.ids.len, a.made);
    try std.testing.expectEqual(b.ids.len, b.made);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(std.testing.allocator);
    for ([_]*IdBatch{ &a, &b }) |batch| for (batch.ids) |id| {
        try std.testing.expect(std.mem.startsWith(u8, id, "call_"));
        const entry = try seen.getOrPut(std.testing.allocator, id);
        if (entry.found_existing) {
            std.debug.print("tool call id made twice: {s}\n", .{id});
            return error.ToolCallIdRepeated;
        }
    };
}

test "makeDirPath takes a path whose last component links to a directory as made, and refuses a dangling link or a file" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    try tmp.dir.createDirPath(io, "volume/cache");
    try tmp.dir.symLink(io, "volume/cache", "linked", .{ .is_directory = true });
    try tmp.dir.symLink(io, "volume/gone", "dangling", .{ .is_directory = true });
    try tmp.dir.writeFile(io, .{ .sub_path = "file", .data = "" });
    var p: [std.fs.max_path_bytes]u8 = undefined;
    try makeDirPath(io, try std.fmt.bufPrint(&p, "{s}/linked", .{root}));
    try makeDirPath(io, try std.fmt.bufPrint(&p, "{s}/linked/fp/e1", .{root}));
    var made = try tmp.dir.openDir(io, "volume/cache/fp/e1", .{});
    made.close(io);
    try std.testing.expectError(error.NotDir, makeDirPath(io, try std.fmt.bufPrint(&p, "{s}/dangling", .{root})));
    try std.testing.expectError(error.NotDir, makeDirPath(io, try std.fmt.bufPrint(&p, "{s}/file", .{root})));
}
