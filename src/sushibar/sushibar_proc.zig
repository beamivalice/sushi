//! The operating-system side of `sushibar`: pid files, what the process table says about a pid, and spawning a
//! program in a session of its own.
const std = @import("std");
const builtin = @import("builtin");
const logic = @import("sushibar_logic.zig");

extern fn proc_pidpath(pid: c_int, buffer: *anyopaque, size: u32) c_int;
extern fn getsid(pid: c_int) c_int;
extern fn flock(fd: c_int, op: c_int) c_int;

const lock_ex = 2;
const lock_nb = 4;

const ctl_kern = 1;
const kern_procargs2 = 49;
pub const args_cap = 128 * 1024;

pub fn exeOfPid(pid: i32, buf: []u8) ?[]const u8 {
    const n = proc_pidpath(pid, buf.ptr, @intCast(buf.len));
    return if (n > 0) buf[0..@intCast(n)] else null;
}

pub fn argsOfPid(pid: i32, buf: []u8) ?[]const u8 {
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

/// Takes the exclusive lock on `path` for the rest of the process (the descriptor stays open; `exec` closes it, and
/// the new image takes it again). False when another process holds it.
pub fn acquireLock(path: [*:0]const u8) bool {
    const fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return false;
    if (flock(fd, lock_ex | lock_nb) == 0) return true;
    _ = std.c.close(fd);
    return false;
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

pub fn kill(pid: i32) void {
    _ = std.c.kill(pid, .KILL);
}

const prox_fdtype_socket = 2;
const proc_all_pids = 1;
const proc_pidlistfds = 1;
const proc_pidfdsocketinfo = 3;
const sockinfo_tcp = 2;

extern fn proc_listpids(kind: u32, info: u32, buffer: ?*anyopaque, size: c_int) c_int;
extern fn proc_pidinfo(pid: c_int, flavor: c_int, arg: u64, buffer: ?*anyopaque, size: c_int) c_int;
extern fn proc_pidfdinfo(pid: c_int, fd: c_int, flavor: c_int, buffer: ?*anyopaque, size: c_int) c_int;

const FdInfo = extern struct { fd: i32, fd_type: u32 };

// Offsets into `struct socket_fdinfo` (sys/proc_info.h): the 24-byte `proc_fileinfo`, then `socket_info`, whose
// `soi_kind` is at 232 and whose protocol union starts at 240 with `in_sockinfo` (foreign port, local port,
// `insi_vflag` at +24, the local address at +48: an IPv4 one in the last four bytes of the 16-byte slot).
const off_family = 24 + 160;
const off_kind = 24 + 232;
const off_fport = 24 + 240;
const off_lport = 24 + 244;
const off_vflag = 24 + 264;
const off_laddr = 24 + 288;
const socket_info_cap = 1024;
const ini_ipv6 = 2;

const Bound = struct { port: u16, addr: logic.Addr };

fn listeningSocket(pid: i32, fd: i32) ?Bound {
    var info: [socket_info_cap]u8 align(8) = undefined;
    const n = proc_pidfdinfo(pid, fd, proc_pidfdsocketinfo, &info, socket_info_cap);
    if (n < off_lport + 2) return null;
    const family = std.mem.readInt(i32, info[off_family..][0..4], builtin.cpu.arch.endian());
    const kind = std.mem.readInt(i32, info[off_kind..][0..4], builtin.cpu.arch.endian());
    if (kind != sockinfo_tcp or (family != 2 and family != 30)) return null;
    // A listening socket has no peer: an accepted connection carries the same local port but a foreign one.
    if (std.mem.readInt(u16, info[off_fport..][0..2], .big) != 0) return null;
    const port = std.mem.readInt(u16, info[off_lport..][0..2], .big);
    if (port == 0) return null;
    const addr: logic.Addr = if (info[off_vflag] & ini_ipv6 != 0)
        .{ .v6 = info[off_laddr..][0..16].* }
    else
        .{ .v4 = info[off_laddr + 12 ..][0..4].* };
    return .{ .port = port, .addr = addr };
}

/// Every process listening on a TCP port (own user's only), optionally only those whose executable is `only_exe`.
pub fn listeners(out: []logic.Listener, only_exe: ?[]const u8) []logic.Listener {
    var pids: [8192]i32 = undefined;
    const bytes = proc_listpids(proc_all_pids, 0, &pids, @sizeOf(@TypeOf(pids)));
    if (bytes <= 0) return out[0..0];
    var n: usize = 0;
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    var fds: [2048]FdInfo = undefined;
    for (pids[0 .. @as(usize, @intCast(bytes)) / @sizeOf(i32)]) |pid| {
        if (pid <= 0) continue;
        if (only_exe) |want| {
            const exe = exeOfPid(pid, &exe_buf) orelse continue;
            if (!std.mem.eql(u8, std.fs.path.basename(exe), want)) continue;
        }
        const got = proc_pidinfo(pid, proc_pidlistfds, 0, &fds, @sizeOf(@TypeOf(fds)));
        if (got <= 0) continue;
        for (fds[0 .. @as(usize, @intCast(got)) / @sizeOf(FdInfo)]) |f| {
            if (f.fd_type != prox_fdtype_socket) continue;
            const bound = listeningSocket(pid, f.fd) orelse continue;
            if (n == out.len) return out[0..n];
            out[n] = .{ .pid = pid, .port = bound.port, .addr = bound.addr };
            n += 1;
        }
    }
    return out[0..n];
}

/// True when something accepts a TCP connection on 127.0.0.1:`port`. Asked only when no sushi listens there, and
/// Start binds loopback, so a holder on another address does not block it.
pub fn portInUse(port: u16) bool {
    const fd = std.c.socket(std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return false;
    defer _ = std.c.close(fd);
    var addr: std.c.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    return std.c.connect(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr))) == 0;
}

/// Runs `argv` to its end with stdout and stderr in the file `out_path`; the exit code, or null when it did not
/// start or was killed.
pub fn runCapture(argv: []const [*:0]const u8, out_path: [*:0]const u8) ?u8 {
    var args: [8:null]?[*:0]const u8 = @splat(null);
    if (argv.len > args.len) return null;
    for (argv, 0..) |a, i| args[i] = a;
    var actions: std.c.posix_spawn_file_actions_t = undefined;
    if (std.c.posix_spawn_file_actions_init(&actions) != 0) return null;
    defer _ = std.c.posix_spawn_file_actions_destroy(&actions);
    const wr: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
    const rd: std.c.O = .{ .ACCMODE = .RDWR };
    if (std.c.posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", @bitCast(rd), 0) != 0) return null;
    if (std.c.posix_spawn_file_actions_addopen(&actions, 1, out_path, @bitCast(wr), 0o600) != 0) return null;
    if (std.c.posix_spawn_file_actions_adddup2(&actions, 1, 2) != 0) return null;
    var attr: std.c.posix_spawnattr_t = undefined;
    if (std.c.posix_spawnattr_init(&attr) != 0) return null;
    defer _ = std.c.posix_spawnattr_destroy(&attr);
    _ = std.c.posix_spawnattr_setflags(&attr, .{ .CLOEXEC_DEFAULT = true });
    var pid: std.c.pid_t = 0;
    if (std.c.posix_spawn(&pid, argv[0], &actions, &attr, &args, std.c.environ) != 0) return null;
    var status: c_int = 0;
    if (std.c.waitpid(pid, &status, 0) != pid) return null;
    const bits: u32 = @bitCast(status);
    return if (bits & 0x7f == 0) @intCast((bits >> 8) & 0xff) else null;
}

/// `path` with every symlink resolved: Homebrew links `bin/sushibar` to the keg's `libexec/sushibar`, and the
/// sibling `sushi` and the Cellar check belong to the keg. Resolve again at each use: an upgrade replaces the keg.
pub fn realPath(buf: []u8, path: []const u8) ?[]const u8 {
    var in: [std.fs.max_path_bytes:0]u8 = undefined;
    if (path.len >= in.len or buf.len < std.fs.max_path_bytes) return null;
    @memcpy(in[0..path.len], path);
    in[path.len] = 0;
    const r = std.c.realpath(&in, buf.ptr) orelse return null;
    return std.mem.span(r);
}

/// Replaces this process with `argv` (same pid, fds 0-2 kept, every other fd closed). Returns only on failure.
pub fn execReplace(argv: []const [*:0]const u8) void {
    var args: [8:null]?[*:0]const u8 = @splat(null);
    if (argv.len > args.len) return;
    for (argv, 0..) |a, i| args[i] = a;
    var actions: std.c.posix_spawn_file_actions_t = undefined;
    if (std.c.posix_spawn_file_actions_init(&actions) != 0) return;
    defer _ = std.c.posix_spawn_file_actions_destroy(&actions);
    for ([_]c_int{ 0, 1, 2 }) |fd| _ = std.c.posix_spawn_file_actions_addinherit_np(&actions, fd);
    var attr: std.c.posix_spawnattr_t = undefined;
    if (std.c.posix_spawnattr_init(&attr) != 0) return;
    defer _ = std.c.posix_spawnattr_destroy(&attr);
    _ = std.c.posix_spawnattr_setflags(&attr, .{ .SETEXEC = true, .CLOEXEC_DEFAULT = true });
    var pid: std.c.pid_t = 0;
    _ = std.c.posix_spawn(&pid, argv[0], &actions, &attr, &args, std.c.environ);
}

/// A listening socket of this process on an ephemeral port of the loopback address `host`, for tests.
const TestListener = struct { fd: c_int, port: u16 };

fn testListenerOn(host: logic.Addr) !TestListener {
    const v6 = host == .v6;
    const fd = std.c.socket(if (v6) std.c.AF.INET6 else std.c.AF.INET, std.c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    errdefer _ = std.c.close(fd);
    var a4: std.c.sockaddr.in = .{ .port = 0, .addr = if (v6) 0 else @bitCast(host.v4) };
    var a6: std.c.sockaddr.in6 = .{ .port = 0, .addr = if (v6) host.v6 else @splat(0), .flowinfo = 0, .scope_id = 0 };
    const sa: *std.c.sockaddr = if (v6) @ptrCast(&a6) else @ptrCast(&a4);
    var len: std.c.socklen_t = if (v6) @sizeOf(@TypeOf(a6)) else @sizeOf(@TypeOf(a4));
    if (std.c.bind(fd, sa, len) != 0) return error.BindFailed;
    if (std.c.listen(fd, 4) != 0) return error.ListenFailed;
    if (std.c.getsockname(fd, sa, &len) != 0) return error.NameFailed;
    return .{ .fd = fd, .port = std.mem.bigToNative(u16, if (v6) a6.port else a4.port) };
}

fn testListener() !TestListener {
    return testListenerOn(logic.loopback);
}

test "listeners: finds the TCP listener this process opened, and the name filter keeps only sushi" {
    const l = try testListener();
    defer _ = std.c.close(l.fd);
    var buf: [64]logic.Listener = undefined;
    const found = for (listeners(&buf, null)) |x| {
        if (x.port == l.port) break x;
    } else return error.ListenerNotFound;
    try std.testing.expectEqual(selfPid(), found.pid);
    for (listeners(&buf, "sushi")) |x| try std.testing.expect(x.port != l.port);
    var exe: [std.fs.max_path_bytes]u8 = undefined;
    const me = std.fs.path.basename(exeOfPid(selfPid(), &exe).?);
    var seen = false;
    for (listeners(&buf, me)) |x| {
        if (x.port == l.port) seen = true;
    }
    try std.testing.expect(seen);
}

test "listeners: one listener is listed once, and portInUse sees it" {
    const l = try testListener();
    defer _ = std.c.close(l.fd);
    try std.testing.expect(portInUse(l.port));
    var buf: [64]logic.Listener = undefined;
    var hits: usize = 0;
    for (listeners(&buf, null)) |x| {
        if (x.port == l.port and x.pid == selfPid()) hits += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), hits);
}

test "listeners: carry the address the socket is bound to" {
    var lo6: [16]u8 = @splat(0);
    lo6[15] = 1;
    for ([_]logic.Addr{ logic.loopback, .{ .v6 = lo6 } }) |host| {
        const l = testListenerOn(host) catch |e| switch (e) {
            error.SocketFailed, error.BindFailed => continue,
            else => return e,
        };
        defer _ = std.c.close(l.fd);
        var buf: [64]logic.Listener = undefined;
        const found = for (listeners(&buf, null)) |x| {
            if (x.port == l.port) break x;
        } else return error.ListenerNotFound;
        try std.testing.expectEqualDeep(host, found.addr);
    }
}

test "portInUse: a closed port is free" {
    const l = try testListener();
    _ = std.c.close(l.fd);
    try std.testing.expect(!portInUse(l.port));
}

test "runCapture: output and exit code of a short program, and a program that cannot start" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/out.txt", .{dir}, 0);
    try std.testing.expectEqual(@as(?u8, 3), runCapture(&.{ "/bin/sh", "-c", "echo out; echo err >&2; exit 3" }, path));
    var text: [64]u8 = undefined;
    var f = try std.Io.Dir.cwd().openFile(std.testing.io, path, .{});
    defer f.close(std.testing.io);
    const n = try f.readPositionalAll(std.testing.io, &text, 0);
    try std.testing.expectEqualStrings("out\nerr\n", text[0..n]);
    try std.testing.expectEqual(@as(?u8, null), runCapture(&.{"/no/such/program"}, path));
}

test "acquireLock: a second holder is refused while the first lives, in this process or another" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(std.testing.io, &dir_buf)];
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, "{s}/bar.lock", .{dir}, 0);
    try std.testing.expect(acquireLock(path));
    try std.testing.expect(!acquireLock(path));
    try std.testing.expectEqual(@as(?u8, 1), runCapture(&.{ "/usr/bin/perl", "-e", "use Fcntl qw(:flock); open(my $f, '>>', $ARGV[0]) or exit 2; exit(flock($f, LOCK_EX|LOCK_NB) ? 0 : 1)", path }, "/dev/null"));
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

test "realPath: a Homebrew bin symlink resolves into the keg, where the sibling sushi and the Cellar check live" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    try tmp.dir.createDirPath(io, "Cellar/sushi/1.0/libexec");
    try tmp.dir.createDirPath(io, "bin");
    try tmp.dir.writeFile(io, .{ .sub_path = "Cellar/sushi/1.0/libexec/sushibar", .data = "" });
    var target_buf: [std.fs.max_path_bytes]u8 = undefined;
    const target = try std.fmt.bufPrintSentinel(&target_buf, "{s}/Cellar/sushi/1.0/libexec/sushibar", .{root}, 0);
    var link_buf: [std.fs.max_path_bytes]u8 = undefined;
    const link = try std.fmt.bufPrintSentinel(&link_buf, "{s}/bin/sushibar", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.symlink(target.ptr, link.ptr));

    var real_buf: [std.fs.max_path_bytes]u8 = undefined;
    const real = realPath(&real_buf, link).?;
    try std.testing.expectEqualStrings(target, real);
    var sib: [std.fs.max_path_bytes]u8 = undefined;
    const sushi = logic.siblingPath(&sib, real, "sushi").?;
    try std.testing.expect(std.mem.endsWith(u8, sushi, "/Cellar/sushi/1.0/libexec/sushi"));
    try std.testing.expect(logic.isBrewKeg(sushi));
    try std.testing.expect(realPath(&real_buf, "/no/such/path") == null);
}
