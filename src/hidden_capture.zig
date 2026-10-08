//! The residual stream at every block boundary of a prompt forward, appended to
//! one directory while `sushi kld capture` runs the teacher:
//!   boundary-XX.bin   XX = 00..num_layers, raw bf16 row-major [tokens, width], no header;
//!                     boundary 0 is the input to layer 0, boundary b the output of layer b-1
//!                     width is hidden_size (Qwen4 and GLM: hc_count * hidden_size, [tokens, hc, hidden])
//!   tokens.bin        u32 token ids [tokens]
//! Every file is opened for append, so prompts (and runs) concatenate in token
//! order. A prompt's ids land after all of its rows: `tokens.bin` counts only
//! prompts whose rows are complete, which is where a resumed run truncates to.
//!
//! Opt-in through `SUSHI_HIDDEN_OUT=<abs dir>`; absent or empty = off.
//!
//! Spool mode (`SUSHI_HIDDEN_SPOOL_BYTES=<budget>`, `SUSHI_HIDDEN_OUT` = spool root,
//! `SUSHI_HIDDEN_SPOOL_FIRST=<n>`, default 0) writes each prompt as its own window
//! `<root>/w-<8-digit n>/{boundary-XX.bin,tokens.bin}`, built in `w-<n>.partial` and
//! renamed once every file and the directory are fsynced. Before a window it waits
//! while the complete windows hold >= the budget; the consumer deletes windows it used.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const ENV_VAR = "SUSHI_HIDDEN_OUT";
const SPOOL_BYTES_VAR = "SUSHI_HIDDEN_SPOOL_BYTES";
const SPOOL_FIRST_VAR = "SUSHI_HIDDEN_SPOOL_FIRST";
const spool_poll_ms = 100;

pub const SpoolConfig = struct {
    budget: u64,
    first: u64 = 0,
    /// Called instead of sleeping between budget polls.
    poll_hook: ?*const fn (?*anyopaque) void = null,
    poll_ctx: ?*anyopaque = null,
};

fn envInt(name: [:0]const u8) !?u64 {
    const raw = std.c.getenv(name) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    if (s.len == 0) return null;
    return std.fmt.parseInt(u64, s, 10) catch error.HiddenCaptureBadSpoolEnv;
}

/// The spool settings from the environment, or null when spool mode is off.
pub fn spoolFromEnv() !?SpoolConfig {
    const budget = (try envInt(SPOOL_BYTES_VAR)) orelse return null;
    return .{ .budget = budget, .first = (try envInt(SPOOL_FIRST_VAR)) orelse 0 };
}

/// The output directory from the environment, or null when capture is off.
pub fn envPath() ?[]const u8 {
    const raw = std.c.getenv(ENV_VAR) orelse return null;
    const s = std.mem.sliceTo(raw, 0);
    return if (s.len == 0) null else s;
}

pub const Writer = struct {
    allocator: std.mem.Allocator,
    hidden: usize,
    /// One append descriptor per boundary, then `tokens`.
    boundaries: []std.c.fd_t,
    tokens: std.c.fd_t = -1,
    spool: ?Spool = null,

    pub fn open(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, num_layers: usize, hidden: usize) !*Writer {
        return openWith(allocator, io, dir, num_layers, hidden, try spoolFromEnv());
    }

    pub fn openWith(allocator: std.mem.Allocator, io: std.Io, dir: []const u8, num_layers: usize, hidden: usize, spool: ?SpoolConfig) !*Writer {
        if (num_layers == 0 or hidden == 0) return error.HiddenCaptureBadGeometry;
        try std.Io.Dir.cwd().createDirPath(io, dir);
        const fds = try allocator.alloc(std.c.fd_t, num_layers + 1);
        @memset(fds, -1);
        const self = allocator.create(Writer) catch |e| {
            allocator.free(fds);
            return e;
        };
        self.* = .{ .allocator = allocator, .hidden = hidden, .boundaries = fds };
        errdefer self.close();
        if (spool) |cfg| {
            self.spool = .{ .io = io, .root = try allocator.dupe(u8, dir), .cfg = cfg, .next = cfg.first };
            log.info("[hidden] spool root {s}, budget {d} bytes, first window {d}\n", .{ dir, cfg.budget, cfg.first });
            return self;
        }
        var name: [32]u8 = undefined;
        for (fds, 0..) |*fd, b| fd.* = try openAppend(dir, try std.fmt.bufPrint(&name, "boundary-{d:0>2}.bin", .{b}));
        self.tokens = try openAppend(dir, "tokens.bin");
        return self;
    }

    pub fn close(self: *Writer) void {
        for (self.boundaries) |fd| if (fd >= 0) {
            _ = std.c.close(fd);
        };
        if (self.tokens >= 0) _ = std.c.close(self.tokens);
        if (self.spool) |sp| self.allocator.free(sp.root);
        self.allocator.free(self.boundaries);
        self.allocator.destroy(self);
    }

    /// One prompt forward: `rows[0]` the residual entering layer 0 and `rows[b]`
    /// layer b-1's output, each [1, n, hidden] or [n, hidden] bf16, for the n `ids`.
    /// Every row is checked before anything is written.
    pub fn append(self: *Writer, s: mlx.mlx_stream, ids: []const u32, rows: []const mlx.mlx_array) !void {
        if (rows.len != self.boundaries.len) return error.HiddenCaptureBoundaryCount;
        const n = ids.len;
        if (n == 0) return error.HiddenCaptureEmpty;
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (rows) |r| {
            if (r.ctx == null) return error.HiddenCaptureMissingBoundary;
            if (mlx.mlx_array_dtype(r) != .bfloat16) return error.HiddenCaptureNotBf16;
            const shape = mlx.getShape(r);
            const fits = switch (shape.len) {
                2 => shape[0] == n and shape[1] == self.hidden,
                3 => shape[0] == 1 and shape[1] == n and shape[2] == self.hidden,
                else => false,
            };
            if (!fits) return error.HiddenCaptureBadShape;
            try mlx.check(mlx.mlx_vector_array_append_value(vec, r));
        }
        try mlx.check(mlx.mlx_eval(vec));
        var window: ?Window = null;
        if (self.spool) |*sp| window = try sp.begin();
        for (rows, self.boundaries, 0..) |r, base_fd, b| {
            var fd = base_fd;
            if (window) |*wd| fd = try wd.create(b);
            defer if (window != null) {
                _ = std.c.close(fd);
            };
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_contiguous(&c, r, false, s));
            try mlx.check(mlx.mlx_array_eval(c));
            const p = mlx.mlx_array_data_bfloat16(c) orelse return error.HiddenCaptureUnreadable;
            try writeAll(fd, std.mem.sliceAsBytes(p[0 .. n * self.hidden]));
            if (window != null and std.c.fsync(fd) != 0) return error.HiddenCaptureSyncFailed;
        }
        if (window) |*wd| {
            const fd = try wd.create(null);
            defer _ = std.c.close(fd);
            try writeAll(fd, std.mem.sliceAsBytes(ids));
            if (std.c.fsync(fd) != 0) return error.HiddenCaptureSyncFailed;
            try wd.commit(&self.spool.?);
            return;
        }
        try writeAll(self.tokens, std.mem.sliceAsBytes(ids));
    }

    /// One boundary's rows of one forward piece, BF16 with `hidden` values per token in any leading shape;
    /// `hash` sees exactly the appended bytes. Returns the token count.
    pub fn appendRows(self: *Writer, s: mlx.mlx_stream, boundary: usize, rows: mlx.mlx_array, hash: ?*std.crypto.hash.sha2.Sha256) !usize {
        if (self.spool != null) return error.HiddenCaptureSpoolUnsupported;
        if (boundary >= self.boundaries.len) return error.HiddenCaptureBoundaryCount;
        if (mlx.mlx_array_dtype(rows) != .bfloat16) return error.HiddenCaptureNotBf16;
        const size = mlx.mlx_array_size(rows);
        if (size == 0 or size % self.hidden != 0) return error.HiddenCaptureBadShape;
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, rows, false, s));
        try mlx.check(mlx.mlx_array_eval(c));
        const bytes = std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(c) orelse return error.HiddenCaptureUnreadable)[0..size]);
        if (hash) |h| h.update(bytes);
        try writeAll(self.boundaries[boundary], bytes);
        return size / self.hidden;
    }

    pub fn appendTokens(self: *Writer, ids: []const u32) !void {
        if (self.spool != null) return error.HiddenCaptureSpoolUnsupported;
        try writeAll(self.tokens, std.mem.sliceAsBytes(ids));
    }

    /// Bytes reach the device before a caller records them as committed.
    pub fn sync(self: *Writer) !void {
        if (self.spool != null) return;
        for (self.boundaries) |fd| if (std.c.fsync(fd) != 0) return error.HiddenCaptureSyncFailed;
        if (std.c.fsync(self.tokens) != 0) return error.HiddenCaptureSyncFailed;
    }

    /// Drops whatever an interrupted run appended past `tokens` committed tokens; a file shorter than that is refused.
    pub fn truncateTo(self: *Writer, tokens: u64) !void {
        if (self.spool != null) return error.HiddenCaptureSpoolUnsupported;
        for (self.boundaries) |fd| try truncateFd(fd, tokens * self.hidden * 2);
        try truncateFd(self.tokens, tokens * 4);
    }
};

const Spool = struct {
    io: std.Io,
    root: []u8,
    cfg: SpoolConfig,
    next: u64,

    /// Waits for room under the budget, then creates this window's `.partial` directory.
    fn begin(self: *Spool) !Window {
        while (true) {
            const used = try self.completeBytes();
            if (used.windows == 0 or used.bytes < self.cfg.budget) break;
            if (self.cfg.poll_hook) |h| h(self.cfg.poll_ctx) else std.Io.sleep(self.io, .fromMilliseconds(spool_poll_ms), .real) catch {};
        }
        var wd: Window = .{};
        const final = try std.fmt.bufPrintSentinel(&wd.final, "{s}/w-{d:0>8}", .{ self.root, self.next }, 0);
        var st: std.c.Stat = undefined;
        if (std.c.stat(final.ptr, &st) == 0) return error.HiddenCaptureWindowExists;
        wd.final_len = final.len;
        const part = try std.fmt.bufPrintSentinel(&wd.partial, "{s}.partial", .{final}, 0);
        wd.partial_len = part.len;
        if (std.c.mkdir(part.ptr, 0o755) != 0) {
            return if (std.c._errno().* == @intFromEnum(std.c.E.EXIST)) error.HiddenCaptureWindowExists else error.HiddenCaptureOpenFailed;
        }
        return wd;
    }

    const Used = struct { bytes: u64 = 0, windows: usize = 0 };

    fn completeBytes(self: *Spool) !Used {
        var root = std.Io.Dir.cwd().openDir(self.io, self.root, .{ .iterate = true }) catch return error.HiddenCaptureOpenFailed;
        defer root.close(self.io);
        var used: Used = .{};
        var it = root.iterate();
        while (it.next(self.io) catch return error.HiddenCaptureOpenFailed) |e| {
            if (e.kind != .directory or !std.mem.startsWith(u8, e.name, "w-") or std.mem.endsWith(u8, e.name, ".partial")) continue;
            // The consumer may delete the window between listing and reading it.
            var wdir = root.openDir(self.io, e.name, .{ .iterate = true }) catch continue;
            defer wdir.close(self.io);
            used.windows += 1;
            var fit = wdir.iterate();
            while (fit.next(self.io) catch null) |f| {
                const st = wdir.statFile(self.io, f.name, .{}) catch continue;
                used.bytes += st.size;
            }
        }
        return used;
    }
};

const Window = struct {
    partial: [std.fs.max_path_bytes]u8 = undefined,
    partial_len: usize = 0,
    final: [std.fs.max_path_bytes]u8 = undefined,
    final_len: usize = 0,

    /// A boundary file, or `tokens.bin` for null; created exclusively, write-only.
    fn create(self: *Window, boundary: ?usize) !std.c.fd_t {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = if (boundary) |b|
            try std.fmt.bufPrintSentinel(&buf, "{s}/boundary-{d:0>2}.bin", .{ self.partial[0..self.partial_len], b }, 0)
        else
            try std.fmt.bufPrintSentinel(&buf, "{s}/tokens.bin", .{self.partial[0..self.partial_len]}, 0);
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.HiddenCaptureOpenFailed;
        return fd;
    }

    fn commit(self: *Window, sp: *Spool) !void {
        self.partial[self.partial_len] = 0;
        try fsyncDir(self.partial[0..self.partial_len :0]);
        if (std.c.rename(self.partial[0..self.partial_len :0], self.final[0..self.final_len :0]) != 0) return error.HiddenCaptureWriteFailed;
        var rootz: [std.fs.max_path_bytes]u8 = undefined;
        try fsyncDir(try std.fmt.bufPrintSentinel(&rootz, "{s}", .{sp.root}, 0));
        sp.next += 1;
    }
};

fn fsyncDir(path: [:0]const u8) !void {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.HiddenCaptureOpenFailed;
    defer _ = std.c.close(fd);
    if (std.c.fsync(fd) != 0) return error.HiddenCaptureSyncFailed;
}

/// Whether any capture file of `dir` (boundaries 0..`boundaries`-1 and `tokens.bin`) holds bytes.
pub fn holdsData(dir: []const u8, boundaries: usize) !bool {
    var name: [32]u8 = undefined;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    for (0..boundaries + 1) |b| {
        const leaf = if (b == boundaries) "tokens.bin" else try std.fmt.bufPrint(&name, "boundary-{d:0>2}.bin", .{b});
        const fd = std.c.open(try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, leaf }, 0), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.NOENT)) continue;
            return error.HiddenCaptureOpenFailed;
        }
        defer _ = std.c.close(fd);
        if (try fdBytes(fd) != 0) return true;
    }
    return false;
}

fn fdBytes(fd: std.c.fd_t) !u64 {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.HiddenCaptureOpenFailed;
    return @intCast(st.size);
}

fn truncateFd(fd: std.c.fd_t, bytes: u64) !void {
    if (try fdBytes(fd) < bytes) return error.HiddenCaptureShorterThanCommitted;
    if (std.c.ftruncate(fd, @intCast(bytes)) != 0) return error.HiddenCaptureWriteFailed;
}

/// Output files are private: a new one is created exclusively without following
/// a link, and an existing one is appended to only when it is a regular file with
/// a single link, so a write can never reach another file's bytes.
fn openAppend(dir: []const u8, name: []const u8) !std.c.fd_t {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fresh = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .APPEND = true }, @as(std.c.mode_t, 0o644));
    if (fresh >= 0) return fresh;
    if (std.c._errno().* != @intFromEnum(std.c.E.EXIST)) return error.HiddenCaptureOpenFailed;
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .NOFOLLOW = true, .APPEND = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.c.E.LOOP)) error.HiddenCaptureNotPrivateFile else error.HiddenCaptureOpenFailed;
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or !std.c.S.ISREG(@intCast(st.mode)) or st.nlink != 1) {
        _ = std.c.close(fd);
        return error.HiddenCaptureNotPrivateFile;
    }
    return fd;
}

fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.HiddenCaptureWriteFailed;
        }
        if (got == 0) return error.HiddenCaptureWriteFailed;
        done += @intCast(got);
    }
}

// ── tests ──

const testing = std.testing;

fn bf16Rows(values: []const f32, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = mlx.mlx_array_new_data(values.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn readAll(io: std.Io, dir: std.Io.Dir, name: []const u8) ![]u8 {
    return dir.readFileAlloc(io, name, testing.allocator, .limited(1 << 20));
}

test "hidden capture appends each boundary's rows and then the ids, prompt after prompt" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const dir = try std.fs.path.join(testing.allocator, &.{ root, "hidden" });
    defer testing.allocator.free(dir);

    // Two layers => three boundaries of hidden 2; bf16-exact values.
    const w = try Writer.open(testing.allocator, io, dir, 2, 2);
    const first = [_][4]f32{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 }, .{ 9, 10, 11, 12 } };
    var rows: [3]mlx.mlx_array = undefined;
    for (&rows, first) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2, 2 }, s);
    try w.append(s, &.{ 7, 3 }, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    const second = [_][2]f32{ .{ -1, -2 }, .{ -3, -4 }, .{ -5, -6 } };
    for (&rows, second) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2 }, s);
    try w.append(s, &.{9}, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    w.close();

    var d = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer d.close(io);
    const toks = try readAll(io, d, "tokens.bin");
    defer testing.allocator.free(toks);
    try testing.expectEqual(@as(usize, 3 * 4), toks.len);
    for ([_]u32{ 7, 3, 9 }, 0..) |want, i| try testing.expectEqual(want, std.mem.readInt(u32, toks[i * 4 ..][0..4], .little));
    for (0..3) |b| {
        var name: [32]u8 = undefined;
        const bytes = try readAll(io, d, try std.fmt.bufPrint(&name, "boundary-{d:0>2}.bin", .{b}));
        defer testing.allocator.free(bytes);
        try testing.expectEqual(@as(usize, 3 * 2 * 2), bytes.len);
        for (0..6) |i| {
            const want: f32 = if (i < 4) first[b][i] else second[b][i - 4];
            const bits: u32 = @as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16;
            try testing.expectEqual(want, @as(f32, @bitCast(bits)));
        }
    }

    // A second writer on the same directory appends after what is there.
    const again = try Writer.open(testing.allocator, io, dir, 2, 2);
    for (&rows, second) |*r, v| r.* = try bf16Rows(&v, &.{ 1, 2 }, s);
    try again.append(s, &.{4}, &rows);
    for (rows) |r| _ = mlx.mlx_array_free(r);
    again.close();
    const more = try readAll(io, d, "tokens.bin");
    defer testing.allocator.free(more);
    try testing.expectEqual(@as(usize, 4 * 4), more.len);
}

test "hidden capture refuses a row it cannot store exactly, before writing anything" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const w = try Writer.open(testing.allocator, io, dir, 1, 2);
    defer w.close();
    const v = [_]f32{ 1, 2 };
    const good = try bf16Rows(&v, &.{ 1, 2 }, s);
    defer _ = mlx.mlx_array_free(good);
    const wide = mlx.mlx_array_new_data(&v, &[_]c_int{ 1, 2 }, 2, .float32);
    defer _ = mlx.mlx_array_free(wide);
    const unset = mlx.mlx_array_new();
    try testing.expectError(error.HiddenCaptureNotBf16, w.append(s, &.{1}, &.{ good, wide }));
    try testing.expectError(error.HiddenCaptureMissingBoundary, w.append(s, &.{1}, &.{ good, unset }));
    try testing.expectError(error.HiddenCaptureBadShape, w.append(s, &.{ 1, 2 }, &.{ good, good }));
    try testing.expectError(error.HiddenCaptureBoundaryCount, w.append(s, &.{1}, &.{good}));

    for ([_][]const u8{ "boundary-00.bin", "boundary-01.bin", "tokens.bin" }) |name| {
        const st = try tmp.dir.statFile(io, name, .{});
        try testing.expectEqual(@as(u64, 0), st.size);
    }
}

test "hidden capture never writes through a link: a hard-linked or symlinked output is refused, its target untouched" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.writeFile(io, .{ .sub_path = "precious.bin", .data = "checkpoint bytes" });
    const target = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/precious.bin", .{root}, 0);
    defer testing.allocator.free(target);

    for ([_][]const u8{ "hard", "soft" }) |kind| {
        const dir = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ root, kind });
        defer testing.allocator.free(dir);
        try tmp.dir.createDirPath(io, kind);
        const link = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/boundary-01.bin", .{dir}, 0);
        defer testing.allocator.free(link);
        if (std.mem.eql(u8, kind, "hard")) {
            try testing.expectEqual(@as(c_int, 0), std.c.link(target, link));
        } else {
            try testing.expectEqual(@as(c_int, 0), std.c.symlink(target, link));
        }
        try testing.expectError(error.HiddenCaptureNotPrivateFile, Writer.open(testing.allocator, io, dir, 1, 2));
        const kept = try readAll(io, tmp.dir, "precious.bin");
        defer testing.allocator.free(kept);
        try testing.expectEqualStrings("checkpoint bytes", kept);
    }
}

test "hidden capture is off without the environment variable" {
    try testing.expect(envPath() == null);
}

fn spoolRoot(io: std.Io, tmp: *testing.TmpDir, buf: []u8) ![]const u8 {
    try tmp.dir.createDirPath(io, "spool");
    const base = buf[0..try tmp.dir.realPath(io, buf)];
    return std.fmt.bufPrint(buf[base.len..], "{s}/spool", .{base});
}

fn appendOne(w: *Writer, s: mlx.mlx_stream, base: f32, ids: []const u32) !void {
    var rows: [2]mlx.mlx_array = undefined;
    var vals: [4]f32 = undefined;
    for (&rows, 0..) |*r, i| {
        for (0..ids.len) |t| vals[t * 2 ..][0..2].* = .{ base + @as(f32, @floatFromInt(i)), -base };
        r.* = try bf16Rows(vals[0 .. ids.len * 2], &.{ 1, @intCast(ids.len), 2 }, s);
    }
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    try w.append(s, ids, &rows);
}

test "hidden spool writes one complete window per append, numbered from FIRST" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [2 * std.fs.max_path_bytes]u8 = undefined;
    const root = try spoolRoot(io, &tmp, &buf);

    const w = try Writer.openWith(testing.allocator, io, root, 1, 2, .{ .budget = 1 << 30, .first = 7 });
    defer w.close();
    try appendOne(w, s, 1, &.{ 5, 6 });
    try appendOne(w, s, 3, &.{9});

    var spool = try tmp.dir.openDir(io, "spool", .{ .iterate = true });
    defer spool.close(io);
    var it = spool.iterate();
    var seen: usize = 0;
    while (try it.next(io)) |e| {
        try testing.expect(std.mem.eql(u8, e.name, "w-00000007") or std.mem.eql(u8, e.name, "w-00000008"));
        seen += 1;
    }
    try testing.expectEqual(@as(usize, 2), seen);

    var d = try spool.openDir(io, "w-00000007", .{});
    defer d.close(io);
    const toks = try readAll(io, d, "tokens.bin");
    defer testing.allocator.free(toks);
    try testing.expectEqual(@as(usize, 8), toks.len);
    try testing.expectEqual(@as(u32, 6), std.mem.readInt(u32, toks[4..8], .little));
    const b1 = try readAll(io, d, "boundary-01.bin");
    defer testing.allocator.free(b1);
    try testing.expectEqual(@as(usize, 8), b1.len);
    const want = [_]f32{ 2, -1 };
    for (want, 0..) |v, i| {
        const bits: u32 = @as(u32, std.mem.readInt(u16, b1[i * 2 ..][0..2], .little)) << 16;
        try testing.expectEqual(v, @as(f32, @bitCast(bits)));
    }
}

test "hidden spool refuses a window that already exists" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [2 * std.fs.max_path_bytes]u8 = undefined;
    const root = try spoolRoot(io, &tmp, &buf);
    try tmp.dir.createDirPath(io, "spool/w-00000002");
    try tmp.dir.createDirPath(io, "spool/w-00000003.partial");

    const w = try Writer.openWith(testing.allocator, io, root, 1, 2, .{ .budget = 1 << 30, .first = 2 });
    defer w.close();
    try testing.expectError(error.HiddenCaptureWindowExists, appendOne(w, s, 1, &.{1}));
    w.spool.?.next = 3;
    try testing.expectError(error.HiddenCaptureWindowExists, appendOne(w, s, 1, &.{1}));
}

const PollDeleter = struct {
    io: std.Io,
    dir: std.Io.Dir,
    polls: usize = 0,

    fn hook(ctx: ?*anyopaque) void {
        const self: *PollDeleter = @ptrCast(@alignCast(ctx.?));
        self.polls += 1;
        self.dir.deleteTree(self.io, "spool/w-00000000") catch unreachable;
    }
};

test "hidden spool waits while complete windows fill the budget and proceeds once one is deleted" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [2 * std.fs.max_path_bytes]u8 = undefined;
    const root = try spoolRoot(io, &tmp, &buf);
    try tmp.dir.createDirPath(io, "spool/w-00000099.partial");

    var del: PollDeleter = .{ .io = io, .dir = tmp.dir };
    const w = try Writer.openWith(testing.allocator, io, root, 1, 2, .{ .budget = 1, .poll_hook = PollDeleter.hook, .poll_ctx = &del });
    defer w.close();
    try appendOne(w, s, 1, &.{1});
    try testing.expectEqual(@as(usize, 0), del.polls);
    try appendOne(w, s, 2, &.{2});
    try testing.expectEqual(@as(usize, 1), del.polls);
    try tmp.dir.access(io, "spool/w-00000001", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "spool/w-00000000", .{}));
}

test "hidden spool refuses the chunked GLM path" {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [2 * std.fs.max_path_bytes]u8 = undefined;
    const root = try spoolRoot(io, &tmp, &buf);
    const w = try Writer.openWith(testing.allocator, io, root, 1, 2, .{ .budget = 1 << 30 });
    defer w.close();
    const rows = try bf16Rows(&.{ 1, 2 }, &.{ 1, 2 }, s);
    defer _ = mlx.mlx_array_free(rows);
    try testing.expectError(error.HiddenCaptureSpoolUnsupported, w.appendRows(s, 0, rows, null));
    try testing.expectError(error.HiddenCaptureSpoolUnsupported, w.appendTokens(&.{1}));
    try testing.expectError(error.HiddenCaptureSpoolUnsupported, w.truncateTo(0));
}
