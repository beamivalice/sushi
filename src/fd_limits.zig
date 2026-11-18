//! The process's open-file limit: raised at startup toward the hard limit, and the census the accept loop logs once
//! the descriptors run out (host tier, MLX-free).

const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");

const rlim_t = std.posix.rlim_t;
const RLIM_INFINITY = std.c.RLIM.INFINITY;

/// The soft RLIMIT_NOFILE to ask for: the hard limit, capped at the kernel's per-process maximum (macOS refuses a soft
/// limit above `kern.maxfilesperproc`), and never below the current one.
pub fn raisedSoftLimit(current: rlim_t, hard: rlim_t, max_per_proc: ?rlim_t) rlim_t {
    const want = if (max_per_proc) |m| @min(hard, m) else hard;
    return @max(current, want);
}

/// What `raiseOpenFileLimit` did: `after` is the soft limit now in force, `failed` the error that left it at `before`.
pub const Raise = struct { before: rlim_t, after: rlim_t, hard: rlim_t, failed: ?anyerror = null, wanted: rlim_t = 0 };

/// Raises the soft RLIMIT_NOFILE as `raisedSoftLimit` says and logs the `[limits]` line. A refusal leaves the limit as
/// it was; the line says so.
pub fn raiseOpenFileLimit() void {
    var buf: [256]u8 = undefined;
    const lim = std.posix.getrlimit(.NOFILE) catch |e| return log.warn("[limits] open files: the limit could not be read ({s})\n", .{@errorName(e)});
    const want = raisedSoftLimit(lim.cur, lim.max, maxFilesPerProc());
    var r: Raise = .{ .before = lim.cur, .after = lim.cur, .hard = lim.max, .wanted = want };
    if (want != lim.cur) {
        if (std.posix.setrlimit(.NOFILE, .{ .cur = want, .max = lim.max })) r.after = want else |e| r.failed = e;
    }
    const line = limitsLine(&buf, r);
    if (r.failed != null) log.warn("{s}\n", .{line}) else log.info("{s}\n", .{line});
}

fn maxFilesPerProc() ?rlim_t {
    if (builtin.os.tag != .macos) return null;
    var n: c_int = 0;
    var len: usize = @sizeOf(c_int);
    if (std.c.sysctlbyname("kern.maxfilesperproc", &n, &len, null, 0) != 0 or n <= 0) return null;
    return @intCast(n);
}

const Limit = struct {
    v: rlim_t,
    pub fn format(self: Limit, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.v == RLIM_INFINITY) return w.writeAll("unlimited");
        return w.print("{d}", .{self.v});
    }
};

/// The `[limits]` load line for `r`.
pub fn limitsLine(buf: []u8, r: Raise) []const u8 {
    const hard: Limit = .{ .v = r.hard };
    const out = if (r.failed) |e|
        std.fmt.bufPrint(buf, "[limits] open files: soft {f}, raising it to {f} failed ({s}); serving goes on (hard {f})", .{ Limit{ .v = r.before }, Limit{ .v = r.wanted }, @errorName(e), hard })
    else if (r.after != r.before)
        std.fmt.bufPrint(buf, "[limits] open files: soft {f} -> {f} (hard {f})", .{ Limit{ .v = r.before }, Limit{ .v = r.after }, hard })
    else
        std.fmt.bufPrint(buf, "[limits] open files: soft {f}, unchanged (hard {f})", .{ Limit{ .v = r.after }, hard });
    return out catch buf[0..0];
}

/// Out of descriptors the listener stays readable and every accept fails at once: the accept loop waits this long before
/// its next poll instead of spinning.
pub const fd_quota_pause_ms: u32 = 100;

/// The pause before the accept loop's next poll after an accept failed with `err`.
pub fn acceptPauseMs(err: anyerror) u32 {
    return switch (err) {
        error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded => fd_quota_pause_ms,
        else => 0,
    };
}

/// The accept loop's one line when the process or the system has no descriptor left (`err`): the process's count
/// against its soft limit, the holder of most of them, and the pause between tries.
pub fn quotaLine(buf: []u8, err: anyerror, c: Census) []const u8 {
    const head = "accept error: {s}: {d} descriptors open of a soft limit of {f}, {d} of them ";
    const tail = "; connections are dropped until one closes, accepts retried every {d} ms (logged once until an accept succeeds)";
    const soft: Limit = .{ .v = c.soft };
    const name = @errorName(err);
    const pause = acceptPauseMs(err);
    const out = switch (c.holder) {
        .dir => std.fmt.bufPrint(buf, head ++ "files under {s}" ++ tail, .{ name, c.open, soft, c.held, c.dir, pause }),
        .sockets, .pipes => std.fmt.bufPrint(buf, head ++ "{t}" ++ tail, .{ name, c.open, soft, c.held, c.holder, pause }),
        .other => std.fmt.bufPrint(buf, head ++ "other descriptors" ++ tail, .{ name, c.open, soft, c.held, pause }),
    };
    return out catch buf[0..0];
}

/// The process's open descriptors and the group that holds most of them: the directory of a file, or sockets, pipes or
/// other descriptors.
pub const Census = struct {
    open: usize,
    soft: rlim_t,
    held: usize,
    holder: Holder,
    /// The directory, for `holder == .dir`.
    dir: []const u8 = "",

    pub const Holder = enum { dir, sockets, pipes, other };
};

/// Counts the descriptors below the soft limit (all of them once it is reached) and groups them by holder. `dir_buf`
/// holds the returned directory.
pub fn takeCensus(gpa: std.mem.Allocator, dir_buf: []u8) !Census {
    const soft = (try std.posix.getrlimit(.NOFILE)).cur;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var dirs: std.StringHashMapUnmanaged(usize) = .empty;
    var kinds: std.EnumArray(Census.Holder, usize) = .initFill(0);
    var open: usize = 0;
    var path_buf: [std.posix.PATH_MAX]u8 = undefined;
    for (0..@min(soft, census_probe_limit)) |i| {
        const fd: std.c.fd_t = @intCast(i);
        if (std.c.fcntl(fd, std.c.F.GETFD) == -1) continue;
        open += 1;
        var st: std.c.Stat = undefined;
        const mode = if (std.c.fstat(fd, &st) == 0) st.mode & std.c.S.IFMT else 0;
        if (mode == std.c.S.IFSOCK) {
            kinds.getPtr(.sockets).* += 1;
        } else if (mode == std.c.S.IFIFO) {
            kinds.getPtr(.pipes).* += 1;
        } else if (builtin.os.tag == .macos and mode != 0 and std.c.fcntl(fd, std.c.F.GETPATH, &path_buf) != -1) {
            const path = std.mem.sliceTo(&path_buf, 0);
            const dir = if (mode == std.c.S.IFDIR) path else std.fs.path.dirname(path) orelse path;
            const gop = try dirs.getOrPut(a, dir);
            if (!gop.found_existing) {
                gop.key_ptr.* = try a.dupe(u8, dir);
                gop.value_ptr.* = 0;
            }
            gop.value_ptr.* += 1;
        } else {
            kinds.getPtr(.other).* += 1;
        }
    }
    var c: Census = .{ .open = open, .soft = soft, .held = 0, .holder = .other };
    var it = dirs.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* > c.held) {
        c.held = kv.value_ptr.*;
        c.dir = kv.key_ptr.*;
        c.holder = .dir;
    };
    for ([_]Census.Holder{ .sockets, .pipes, .other }) |h| if (kinds.get(h) > c.held) {
        c.held = kinds.get(h);
        c.holder = h;
    };
    if (c.holder == .dir) {
        const n = @min(c.dir.len, dir_buf.len);
        @memcpy(dir_buf[0..n], c.dir[0..n]);
        c.dir = dir_buf[0..n];
    } else c.dir = "";
    return c;
}

/// Past this many descriptors the census stops probing: a soft limit can be in the hundreds of thousands.
const census_probe_limit = 1 << 18;

const testing = std.testing;

test "fd limits: the soft limit rises to the hard limit, capped by the kernel's per-process maximum, never lower" {
    try testing.expectEqual(@as(rlim_t, 245760), raisedSoftLimit(256, RLIM_INFINITY, 245760));
    try testing.expectEqual(@as(rlim_t, 10240), raisedSoftLimit(256, 10240, 245760));
    try testing.expectEqual(@as(rlim_t, 300000), raisedSoftLimit(300000, RLIM_INFINITY, 245760));
    try testing.expectEqual(@as(rlim_t, 256), raisedSoftLimit(256, 256, 245760));
    try testing.expectEqual(RLIM_INFINITY, raisedSoftLimit(256, RLIM_INFINITY, null));
}

test "fd limits: the load line says what changed, what stayed and what failed" {
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("[limits] open files: soft 256 -> 245760 (hard unlimited)", limitsLine(&buf, .{ .before = 256, .after = 245760, .hard = RLIM_INFINITY }));
    try testing.expectEqualStrings("[limits] open files: soft 10240, unchanged (hard 10240)", limitsLine(&buf, .{ .before = 10240, .after = 10240, .hard = 10240 }));
    try testing.expectEqualStrings(
        "[limits] open files: soft 256, raising it to 245760 failed (PermissionDenied); serving goes on (hard unlimited)",
        limitsLine(&buf, .{ .before = 256, .after = 256, .hard = RLIM_INFINITY, .wanted = 245760, .failed = error.PermissionDenied }),
    );
}

test "fd limits: the accept loop pauses only when the process or the system is out of descriptors" {
    try testing.expectEqual(fd_quota_pause_ms, acceptPauseMs(error.ProcessFdQuotaExceeded));
    try testing.expectEqual(fd_quota_pause_ms, acceptPauseMs(error.SystemFdQuotaExceeded));
    try testing.expectEqual(@as(u32, 0), acceptPauseMs(error.ConnectionAborted));
}

test "fd limits: the quota line counts the descriptors against the soft limit and names what holds most" {
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings(
        "accept error: ProcessFdQuotaExceeded: 256 descriptors open of a soft limit of 256, 197 of them files under /tmp/kv/v/packs; connections are dropped until one closes, accepts retried every 100 ms (logged once until an accept succeeds)",
        quotaLine(&buf, error.ProcessFdQuotaExceeded, .{ .open = 256, .soft = 256, .held = 197, .holder = .dir, .dir = "/tmp/kv/v/packs" }),
    );
    try testing.expectEqualStrings(
        "accept error: SystemFdQuotaExceeded: 300 descriptors open of a soft limit of 256, 120 of them sockets; connections are dropped until one closes, accepts retried every 100 ms (logged once until an accept succeeds)",
        quotaLine(&buf, error.SystemFdQuotaExceeded, .{ .open = 300, .soft = 256, .held = 120, .holder = .sockets }),
    );
}

test "fd limits: the census counts this process's descriptors and names the directory that holds most" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var want_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const want = want_buf[0..try tmp.dir.realPath(testing.io, &want_buf)];
    var files: [40]std.Io.File = undefined;
    var name_buf: [16]u8 = undefined;
    for (&files, 0..) |*f, i| f.* = try tmp.dir.createFile(testing.io, std.fmt.bufPrint(&name_buf, "f{d}", .{i}) catch unreachable, .{});
    defer for (files) |f| f.close(testing.io);
    var dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const c = try takeCensus(testing.allocator, &dir_buf);
    try testing.expectEqual(Census.Holder.dir, c.holder);
    try testing.expectEqualStrings(want, c.dir);
    // The files and the directory's own descriptor.
    try testing.expectEqual(@as(usize, files.len + 1), c.held);
    try testing.expect(c.open > files.len);
    try testing.expectEqual((try std.posix.getrlimit(.NOFILE)).cur, c.soft);
}

test "fd limits: the startup raise puts the soft limit where raisedSoftLimit says" {
    const saved = try std.posix.getrlimit(.NOFILE);
    defer std.posix.setrlimit(.NOFILE, saved) catch {};
    try std.posix.setrlimit(.NOFILE, .{ .cur = 256, .max = saved.max });
    raiseOpenFileLimit();
    try testing.expectEqual(raisedSoftLimit(256, saved.max, maxFilesPerProc()), (try std.posix.getrlimit(.NOFILE)).cur);
}
