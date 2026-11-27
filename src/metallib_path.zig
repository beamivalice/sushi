//! Where a running binary finds Sushi Cooler's `sushi.metallib`: one search, read by the stock kernel loader and by the
//! round-cost table's install stamp, so the stamp always names the library the engine loaded. Std only: the MLX-free
//! host tests reach it through `round_cost.zig`.

const std = @import("std");
const builtin = @import("builtin");

pub const Error = error{MetallibNotFound};

/// The layouts a binary ships in, probed in order, relative to the binary's real directory: the release tarball and the
/// Homebrew keg (`lib/` beside it), a `zig build` install (`../lib/`), a flat copy beside it. A test binary run from the
/// Zig cache also probes the checkout's `zig-out/lib/`; a shipped binary never does.
pub const candidates = [_][]const u8{ "lib/sushi.metallib", "../lib/sushi.metallib", "sushi.metallib" } ++
    (if (builtin.is_test) [_][]const u8{"../../../zig-out/lib/sushi.metallib"} else [_][]const u8{});

/// The candidates as one line, for the message a binary without its library logs.
pub const searched = blk: {
    var line: []const u8 = "";
    for (candidates, 0..) |c, i| line = line ++ (if (i > 0) ", " else "") ++ c;
    break :blk line;
};

/// `override` when set (`SUSHI_METALLIB`), else the first candidate that exists beside `exe_path`. Never `mlx.metallib`.
pub fn resolve(buf: []u8, exe_path: []const u8, override: ?[]const u8) Error![:0]const u8 {
    if (override) |p| if (p.len > 0) return std.fmt.bufPrintSentinel(buf, "{s}", .{p}, 0) catch error.MetallibNotFound;
    const dir = std.fs.path.dirname(exe_path) orelse return error.MetallibNotFound;
    inline for (candidates) |rel| {
        const path = std.fmt.bufPrintSentinel(buf, "{s}/" ++ rel, .{dir}, 0) catch return error.MetallibNotFound;
        if (std.c.access(path.ptr, std.c.F_OK) == 0) return path;
    }
    return error.MetallibNotFound;
}

/// The running binary's real path (a Homebrew `bin/` symlink resolves to its keg), into `buf`.
pub fn exePath(buf: *[std.fs.max_path_bytes]u8) ?[]const u8 {
    var raw: [std.fs.max_path_bytes]u8 = undefined;
    var n: u32 = @intCast(raw.len);
    if (std.c._NSGetExecutablePath(&raw, &n) != 0) return null;
    const raw_z: [*:0]const u8 = @ptrCast(&raw);
    if (std.c.realpath(raw_z, buf)) |p| return std.mem.span(p);
    const s = std.mem.span(raw_z);
    @memcpy(buf[0..s.len], s);
    return buf[0..s.len];
}

/// This process's `sushi.metallib`: `SUSHI_METALLIB` (the build's test steps set it), else the search beside the binary.
pub fn forThisProcess(buf: []u8) Error![:0]const u8 {
    const override: ?[]const u8 = if (std.c.getenv("SUSHI_METALLIB")) |v| std.mem.span(v) else null;
    if (override) |p| if (p.len > 0) return resolve(buf, "", p);
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    return resolve(buf, exePath(&exe_buf) orelse return error.MetallibNotFound, null);
}

const testing = std.testing;

test "metallib path: every shipped layout resolves to sushi.metallib, an override wins, and mlx.metallib alone is refused" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(io, &root_buf)];
    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    var buf: [std.fs.max_path_bytes]u8 = undefined;

    try tmp.dir.createDirPath(io, "a/b/c/only_mlx/lib");
    try tmp.dir.writeFile(io, .{ .sub_path = "a/b/c/only_mlx/lib/mlx.metallib", .data = "mlx" });
    // Deep enough that a test binary's checkout probe (`../../../zig-out`) stays inside the temporary folder.
    const only_mlx = try std.fmt.bufPrint(&exe_buf, "{s}/a/b/c/only_mlx/sushi", .{root});
    try testing.expectError(error.MetallibNotFound, resolve(&buf, only_mlx, null));
    try testing.expectEqualStrings("/custom/sushi.metallib", try resolve(&buf, only_mlx, "/custom/sushi.metallib"));

    const Layout = struct { dir: []const u8, file: []const u8, exe: []const u8, ends: []const u8 };
    for ([_]Layout{
        .{ .dir = "keg/lib", .file = "keg/lib/sushi.metallib", .exe = "keg/sushi", .ends = "/keg/lib/sushi.metallib" },
        .{ .dir = "dev/lib", .file = "dev/lib/sushi.metallib", .exe = "dev/bin/sushi", .ends = "/dev/bin/../lib/sushi.metallib" },
        .{ .dir = "flat", .file = "flat/sushi.metallib", .exe = "flat/sushi", .ends = "/flat/sushi.metallib" },
    }) |l| {
        try tmp.dir.createDirPath(io, l.dir);
        if (std.fs.path.dirname(l.exe)) |exe_dir| try tmp.dir.createDirPath(io, exe_dir);
        try tmp.dir.writeFile(io, .{ .sub_path = l.file, .data = "lib" });
        const exe = try std.fmt.bufPrint(&exe_buf, "{s}/{s}", .{ root, l.exe });
        try testing.expect(std.mem.endsWith(u8, try resolve(&buf, exe, null), l.ends));
    }
}
