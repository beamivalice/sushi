//! The operating-system side of `sushibar`: pid files, what the process table says about a pid, and spawning a
//! program in a session of its own.
const std = @import("std");
const logic = @import("sushibar_logic.zig");

extern fn proc_pidpath(pid: c_int, buffer: *anyopaque, size: u32) c_int;
extern fn getsid(pid: c_int) c_int;

const ctl_kern = 1;
const kern_procargs2 = 49;
const args_cap = 128 * 1024;

pub fn exeOfPid(pid: i32, buf: []u8) ?[]const u8 {
    const n = proc_pidpath(pid, buf.ptr, @intCast(buf.len));
    return if (n > 0) buf[0..@intCast(n)] else null;
}

fn argsOfPid(pid: i32, buf: []u8) ?[]const u8 {
    const mib = [_]c_int{ ctl_kern, kern_procargs2, pid };
    var len: usize = buf.len;
    if (std.c.sysctl(&mib, mib.len, buf.ptr, &len, null, 0) != 0) return null;
    return buf[0..len];
}

pub fn selfPid() i32 {
    return std.c.getpid();
}

/// A process that leads its own session was detached on purpose, here or by launchd.
pub fn isSessionLeader() bool {
    return getsid(0) == selfPid();
}

pub fn pidFilePath(buf: []u8, home: []const u8, name: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.sushi/{s}", .{ home, name }) catch null;
}

pub fn writePid(io: std.Io, home: []const u8, path: []const u8, entry: logic.PidEntry) !void {
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/.sushi", .{home}) catch return error.NameTooLong;
    try std.Io.Dir.cwd().createDirPath(io, dir);
    var text: [32]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = logic.formatPidFile(&text, entry) });
}

pub fn removePid(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
}

/// The pid file's entry while its pid is still the program that wrote it; a stale file is removed.
pub fn verifiedPid(io: std.Io, path: []const u8, want_exe: []const u8, server: bool) ?logic.PidEntry {
    var text: [64]u8 = undefined;
    var r = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    const n = r.readPositionalAll(io, &text, 0) catch 0;
    r.close(io);
    const entry = logic.parsePidFile(text[0..n]) orelse {
        removePid(io, path);
        return null;
    };
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    var args_buf: [args_cap]u8 = undefined;
    const probe: logic.Probe = .{
        .exe = exeOfPid(entry.pid, &exe_buf),
        .args = if (server) argsOfPid(entry.pid, &args_buf) else null,
    };
    if (logic.pidAlive(entry, probe, want_exe, server)) return entry;
    removePid(io, path);
    return null;
}

pub const SpawnError = error{SpawnFailed};

/// Starts `argv` in a new session with the inherited environment and stdin on /dev/null. Output goes to
/// /dev/null too, unless `keep_stderr`.
pub fn spawnDetached(argv: []const [*:0]const u8, keep_stderr: bool) SpawnError!i32 {
    var args: [8:null]?[*:0]const u8 = @splat(null);
    if (argv.len > args.len) return error.SpawnFailed;
    for (argv, 0..) |a, i| args[i] = a;

    var attr: std.c.posix_spawnattr_t = undefined;
    var actions: std.c.posix_spawn_file_actions_t = undefined;
    if (std.c.posix_spawnattr_init(&attr) != 0) return error.SpawnFailed;
    defer _ = std.c.posix_spawnattr_destroy(&attr);
    if (std.c.posix_spawn_file_actions_init(&actions) != 0) return error.SpawnFailed;
    defer _ = std.c.posix_spawn_file_actions_destroy(&actions);
    if (std.c.posix_spawnattr_setflags(&attr, .{ .SETSID = true, .CLOEXEC_DEFAULT = true }) != 0) return error.SpawnFailed;

    const o_rdwr: std.c.O = .{ .ACCMODE = .RDWR };
    if (std.c.posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", @bitCast(o_rdwr), 0) != 0) return error.SpawnFailed;
    if (std.c.posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", @bitCast(o_rdwr), 0) != 0) return error.SpawnFailed;
    if (keep_stderr) {
        if (std.c.posix_spawn_file_actions_addinherit_np(&actions, 2) != 0) return error.SpawnFailed;
    } else if (std.c.posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", @bitCast(o_rdwr), 0) != 0) return error.SpawnFailed;

    var pid: std.c.pid_t = 0;
    if (std.c.posix_spawn(&pid, argv[0], &actions, &attr, &args, std.c.environ) != 0) return error.SpawnFailed;
    return pid;
}

pub fn terminate(pid: i32) void {
    _ = std.c.kill(pid, .TERM);
}

test "pid file: written, verified against the process table, and swept when stale" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    var home_buf: [std.fs.max_path_bytes]u8 = undefined;
    const home = home_buf[0..try tmp.dir.realPath(io, &home_buf)];
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pidFilePath(&path_buf, home, "t.pid").?;

    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe = exeOfPid(selfPid(), &exe_buf).?;
    const base = std.fs.path.basename(self_exe);
    try writePid(io, home, path, .{ .pid = selfPid(), .port = 18801 });
    try std.testing.expectEqual(selfPid(), verifiedPid(io, path, base, false).?.pid);
    try std.testing.expect(verifiedPid(io, path, "no-such-program", false) == null);
    try std.testing.expect(verifiedPid(io, path, base, false) == null);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = "garbage" });
    try std.testing.expect(verifiedPid(io, path, base, false) == null);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openFile(io, path, .{}));
}

test "spawnDetached: the program leads a session of its own, with the arguments it was given" {
    const pid = try spawnDetached(&.{ "/bin/sleep", "30" }, false);
    defer {
        terminate(pid);
        _ = std.c.waitpid(pid, null, 0);
    }
    try std.testing.expectEqual(pid, getsid(pid));
    try std.testing.expect(getsid(pid) != getsid(0));
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    try std.testing.expectEqualStrings("sleep", std.fs.path.basename(exeOfPid(pid, &buf).?));
    var args: [args_cap]u8 = undefined;
    const got = argsOfPid(pid, &args).?;
    try std.testing.expect(std.mem.indexOf(u8, got, "/bin/sleep\x0030\x00") != null);
}
