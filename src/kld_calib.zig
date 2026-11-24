//! The files of a layer-major calibration capture (`sushi kld capture --layer-major`'s `calib/`), whatever the arch:
//! append-only row files truncated back to the committed windows on a resume, the trunk sums (`sum x x^T` per linear
//! input, f32) folded per layer into `trunk-LL.<windows>.safetensors` off the inference thread, and the run's ledger
//! (`layer-major.json` binds the run, `windows.jsonl` commits a window once every file of its batch is synced).
const std = @import("std");
const mlx = @import("mlx.zig");
const fixture = @import("kld_fixture.zig");
const util = @import("kld_capture_util.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

// ── row files ────────────────────────────────────────────────────────────────────────────────────

pub fn openAppend(dir: []const u8, name: []const u8) !std.c.fd_t {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ dir, name }, 0);
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .APPEND = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CalibCaptureOpenFailed;
    return fd;
}

pub fn writeAll(fd: std.c.fd_t, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const got = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (got < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.CalibCaptureWriteFailed;
        }
        if (got == 0) return error.CalibCaptureWriteFailed;
        done += @intCast(got);
    }
}

/// A file shorter than `bytes` lost committed rows; longer is an interrupted batch's tail.
pub fn truncateFd(fd: std.c.fd_t, bytes: u64) !void {
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or @as(u64, @intCast(st.size)) < bytes) return error.CalibCaptureRowsMissing;
    if (std.c.ftruncate(fd, @intCast(bytes)) != 0) return error.CalibCaptureOpenFailed;
}

/// The array's bytes on the host, contiguous (the caller frees).
pub fn hostBytes(a: std.mem.Allocator, s: mlx.mlx_stream, x: mlx.mlx_array) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const data: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull);
    return a.dupe(u8, data[0..n]);
}

/// Name, size and mtime of every `model*.safetensors` shard: a source changed under a run changes this.
pub fn sourceStatSha(a: std.mem.Allocator, io: std.Io, dir_path: []const u8) ![64]u8 {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |n| a.free(n);
        names.deinit(a);
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.startsWith(u8, entry.name, "model") or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try names.ensureUnusedCapacity(a, 1);
        names.appendAssumeCapacity(try a.dupe(u8, entry.name));
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn less(_: void, x: []u8, y: []u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.less);
    if (names.items.len == 0) return error.CalibSourceHasNoShards;
    var hash = Sha256.init(.{});
    for (names.items) |name| {
        const stat = try dir.statFile(io, name, .{});
        const stamp = try std.fmt.allocPrint(a, "{s}:{d}:{d}\n", .{ name, stat.size, stat.mtime.nanoseconds });
        defer a.free(stamp);
        hash.update(stamp);
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

// ── trunk sums ───────────────────────────────────────────────────────────────────────────────────

/// One summed input: `name` (`layers.L.<input>` or `head_in`), its `[in, in]` f32 total and the rows it adds.
pub const Entry = struct {
    name: []u8,
    shape: []usize,
    data: []f32,
    rows: u64,

    pub fn deinit(self: *Entry, a: std.mem.Allocator) void {
        a.free(self.name);
        a.free(self.shape);
        a.free(self.data);
    }
};

/// A layer's sums over windows `[from, to)`, to be added to the running total of `[0, from)`. `layer` null: the head.
pub const Job = struct { layer: ?usize, from: usize, to: usize, entries: []Entry };

pub fn sumsPath(a: std.mem.Allocator, dir: []const u8, layer: ?usize, windows: usize, tmp: bool) ![]u8 {
    const suffix = if (tmp) ".tmp.safetensors" else ".safetensors";
    if (layer) |l| return std.fmt.allocPrint(a, "{s}/trunk-{d:0>2}.{d:0>6}{s}", .{ dir, l, windows, suffix });
    return std.fmt.allocPrint(a, "{s}/trunk-head.{d:0>6}{s}", .{ dir, windows, suffix });
}

/// Adds the previous file's sums into the job's, writes the new file whole and renames it into place.
pub fn foldJob(a: std.mem.Allocator, dir: []const u8, job: Job) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    if (job.from > 0) {
        const prev_path = try sumsPath(a, dir, job.layer, job.from, false);
        defer a.free(prev_path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, prev_path, a, .limited(32 << 30));
        defer a.free(raw);
        if (raw.len < 8) return error.CalibSumsCorrupt;
        const hlen = std.mem.readInt(u64, raw[0..8], .little);
        if (8 + hlen > raw.len) return error.CalibSumsCorrupt;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw[8 .. 8 + hlen], .{});
        defer parsed.deinit();
        const body = raw[8 + hlen ..];
        for (job.entries) |*e| {
            const span = try tensorSpan(parsed.value, e.name, body.len);
            if (span[1] - span[0] != e.data.len * 4) return error.CalibSumsCorrupt;
            for (e.data, 0..) |*v, i| v.* += @bitCast(std.mem.readInt(u32, body[span[0] + i * 4 ..][0..4], .little));
            var key_buf: [160]u8 = undefined;
            const rows = try tensorSpan(parsed.value, try std.fmt.bufPrint(&key_buf, "{s}.rows", .{e.name}), body.len);
            if (rows[1] - rows[0] != 8) return error.CalibSumsCorrupt;
            e.rows += std.mem.readInt(u64, body[rows[0]..][0..8], .little);
        }
    }
    var header: std.Io.Writer.Allocating = .init(a);
    defer header.deinit();
    const w = &header.writer;
    try w.print("{{\"__metadata__\":{{\"windows\":\"{d}\",\"producer\":\"sushi kld capture --layer-major\"}}", .{job.to});
    var offset: u64 = 0;
    for (job.entries) |e| {
        try w.print(",\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{e.name});
        for (e.shape, 0..) |d, i| try w.print("{s}{d}", .{ if (i == 0) "" else ",", d });
        try w.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + e.data.len * 4 });
        offset += e.data.len * 4;
    }
    for (job.entries) |e| {
        try w.print(",\"{s}.rows\":{{\"dtype\":\"I64\",\"shape\":[1],\"data_offsets\":[{d},{d}]}}", .{ e.name, offset, offset + 8 });
        offset += 8;
    }
    try w.writeAll("}");
    while (header.written().len % 8 != 0) try w.writeByte(' ');

    const tmp = try sumsPath(a, dir, job.layer, job.to, true);
    defer a.free(tmp);
    const final = try sumsPath(a, dir, job.layer, job.to, false);
    defer a.free(final);
    const tmp_z = try std.fmt.allocPrintSentinel(a, "{s}", .{tmp}, 0);
    defer a.free(tmp_z);
    const fd = std.c.open(tmp_z.ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.CalibCaptureOpenFailed;
    {
        defer _ = std.c.close(fd);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.written().len, .little);
        try writeAll(fd, &len);
        try writeAll(fd, header.written());
        for (job.entries) |e| try writeAll(fd, std.mem.sliceAsBytes(e.data));
        for (job.entries) |e| try writeAll(fd, std.mem.asBytes(&e.rows));
        if (std.c.fsync(fd) != 0) return error.CalibCaptureWriteFailed;
    }
    const cwd = std.Io.Dir.cwd();
    try cwd.rename(tmp, cwd, final, io);
}

/// Where tensor `name` sits in a safetensors body of `body_len` bytes, by its header `header`.
fn tensorSpan(header: std.json.Value, name: []const u8, body_len: usize) ![2]usize {
    if (header != .object) return error.CalibSumsCorrupt;
    const t = header.object.get(name) orelse return error.CalibSumsCorrupt;
    if (t != .object) return error.CalibSumsCorrupt;
    const off = t.object.get("data_offsets") orelse return error.CalibSumsCorrupt;
    if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) return error.CalibSumsCorrupt;
    const lo = std.math.cast(usize, off.array.items[0].integer) orelse return error.CalibSumsCorrupt;
    const hi = std.math.cast(usize, off.array.items[1].integer) orelse return error.CalibSumsCorrupt;
    if (lo > hi or hi > body_len) return error.CalibSumsCorrupt;
    return .{ lo, hi };
}

/// Folds at most two layers' sums at a time off the inference thread; MLX stays on the caller.
pub const Folder = struct {
    a: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    mu: std.Io.Mutex = .init,
    work: std.Io.Condition = .init,
    idle: std.Io.Condition = .init,
    queue: [2]?Job = .{ null, null },
    busy: bool = false,
    stopping: bool = false,
    failure: ?anyerror = null,
    thread: ?std.Thread = null,

    pub fn start(self: *Folder) !void {
        self.thread = try std.Thread.spawn(.{}, loop, .{self});
    }

    pub fn stop(self: *Folder) void {
        self.mu.lockUncancelable(self.io);
        self.stopping = true;
        self.work.broadcast(self.io);
        self.mu.unlock(self.io);
        if (self.thread) |t| t.join();
        self.thread = null;
        for (&self.queue) |*slot| if (slot.*) |job| {
            for (job.entries) |*e| e.deinit(self.a);
            self.a.free(job.entries);
            slot.* = null;
        };
    }

    /// Takes the job's entries; waits while both slots are taken.
    pub fn submit(self: *Folder, job: Job) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.failure == null and self.queue[0] != null and self.queue[1] != null) self.idle.waitUncancelable(self.io, &self.mu);
        if (self.failure) |e| {
            for (job.entries) |*x| @constCast(x).deinit(self.a);
            self.a.free(job.entries);
            return e;
        }
        const slot: usize = if (self.queue[0] == null) 0 else 1;
        self.queue[slot] = job;
        self.work.signal(self.io);
    }

    /// Returns once every submitted job has been written.
    pub fn drain(self: *Folder) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.failure == null and (self.busy or self.queue[0] != null or self.queue[1] != null)) self.idle.waitUncancelable(self.io, &self.mu);
        if (self.failure) |e| return e;
    }

    fn loop(self: *Folder) void {
        while (true) {
            self.mu.lockUncancelable(self.io);
            while (!self.stopping and self.queue[0] == null and self.queue[1] == null) self.work.waitUncancelable(self.io, &self.mu);
            if (self.stopping) {
                self.mu.unlock(self.io);
                return;
            }
            const slot: usize = if (self.queue[0] != null) 0 else 1;
            const job = self.queue[slot].?;
            self.busy = true;
            self.mu.unlock(self.io);
            const result = foldJob(self.a, self.dir, job);
            for (job.entries) |*e| e.deinit(self.a);
            self.a.free(job.entries);
            self.mu.lockUncancelable(self.io);
            self.queue[slot] = null;
            self.busy = false;
            if (result) |_| {} else |err| self.failure = err;
            self.idle.broadcast(self.io);
            self.mu.unlock(self.io);
        }
    }
};

/// The finished running totals become `trunk-LL.safetensors` and `trunk-head.safetensors`.
pub fn finalizeSums(a: std.mem.Allocator, io: std.Io, calib: []const u8, layers: usize, windows: usize) !void {
    const cwd = std.Io.Dir.cwd();
    for (0..layers + 1) |k| {
        const layer: ?usize = if (k == layers) null else k;
        const from = try sumsPath(a, calib, layer, windows, false);
        defer a.free(from);
        const to = if (layer) |l| try std.fmt.allocPrint(a, "{s}/trunk-{d:0>2}.safetensors", .{ calib, l }) else try std.fmt.allocPrint(a, "{s}/trunk-head.safetensors", .{calib});
        defer a.free(to);
        try cwd.rename(from, cwd, to, io);
    }
}

// ── the ledger ───────────────────────────────────────────────────────────────────────────────────

/// One line of `windows.jsonl`, written only after the window's rows, routes, sums and logits reached the device.
pub const WindowRecord = struct {
    window: usize,
    id: []const u8,
    dir: []const u8,
    tokens: usize,
    token_offset: u64,
    chosen: u32,
    nll_bits: u64,
    logits_dtype: []const u8,
    ids_sha256: []const u8,
    logits_sha256: []const u8,
};

/// `<out>.partial`: the manifest that binds the run, the committed windows and the token ids beside them.
pub const Ledger = struct {
    a: std.mem.Allocator,
    io: std.Io,
    staging: []const u8,
    calib: []const u8,
    arena: std.heap.ArenaAllocator,
    log_fd: std.c.fd_t = -1,
    tokens_fd: std.c.fd_t = -1,
    records: []WindowRecord = &.{},
    committed_tokens: u64 = 0,

    /// Creates the staging directory bound to `expected`, or reopens one bound to it; another run's is refused.
    pub fn open(a: std.mem.Allocator, io: std.Io, staging: []const u8, calib: []const u8, expected: []const u8) !Ledger {
        var self: Ledger = .{ .a = a, .io = io, .staging = staging, .calib = calib, .arena = .init(a) };
        errdefer self.close();
        const cwd = std.Io.Dir.cwd();
        const manifest = try std.fmt.allocPrint(a, "{s}/layer-major.json", .{staging});
        defer a.free(manifest);
        if (cwd.access(io, staging, .{})) |_| {
            const stored = cwd.readFileAlloc(io, manifest, a, .limited(1 << 20)) catch |err| return if (err == error.FileNotFound) error.CalibStagingUnknown else err;
            defer a.free(stored);
            if (!std.mem.eql(u8, stored, expected)) {
                @import("log.zig").err("[kld] layer-major: {s} belongs to another run.\nstored:\n{s}\nthis run:\n{s}\n", .{ staging, stored, expected });
                return error.CalibResumeMismatch;
            }
        } else |err| {
            if (err != error.FileNotFound) return err;
            try cwd.createDir(io, staging, .default_dir);
            const tmp = try std.fmt.allocPrint(a, "{s}.tmp", .{manifest});
            defer a.free(tmp);
            try cwd.writeFile(io, .{ .sub_path = tmp, .data = expected });
            try util.fsyncPath(tmp);
            try cwd.rename(tmp, cwd, manifest, io);
        }
        try cwd.createDirPath(io, calib);
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const log_path = try std.fmt.bufPrintSentinel(&buf, "{s}/windows.jsonl", .{staging}, 0);
        self.log_fd = std.c.open(log_path.ptr, .{ .ACCMODE = .RDWR, .CREAT = true, .APPEND = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
        if (self.log_fd < 0) return error.CalibLogUnreadable;
        const arena = self.arena.allocator();
        const body = try cwd.readFileAlloc(io, log_path, arena, .limited(1 << 30));
        // A line without its newline was being written when the run died; it commits nothing.
        const whole = if (std.mem.lastIndexOfScalar(u8, body, '\n')) |e| e + 1 else 0;
        if (whole != body.len and std.c.ftruncate(self.log_fd, @intCast(whole)) != 0) return error.CalibLogUnreadable;
        var lines: std.ArrayList(WindowRecord) = .empty;
        var it = std.mem.splitScalar(u8, body[0..whole], '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            try lines.append(arena, std.json.parseFromSliceLeaky(WindowRecord, arena, line, .{}) catch return error.CalibLogCorrupt);
        }
        self.records = lines.items;
        self.tokens_fd = try openAppend(calib, "tokens.bin");
        return self;
    }

    pub fn close(self: *Ledger) void {
        if (self.log_fd >= 0) _ = std.c.close(self.log_fd);
        if (self.tokens_fd >= 0) _ = std.c.close(self.tokens_fd);
        self.log_fd = -1;
        self.tokens_fd = -1;
        self.arena.deinit();
    }

    /// Every committed window must still be the same input and logits row, and `tokens.bin` must hold its ids.
    pub fn verify(self: *Ledger, prompts: []const fixture.Prompt, inputs: []const []u32) !void {
        var offset: u64 = 0;
        for (self.records, 0..) |r, i| {
            if (r.window != i or i >= prompts.len or !std.mem.eql(u8, r.id, prompts[i].id) or r.tokens != inputs[i].len or r.token_offset != offset) return error.CalibWindowChanged;
            const ids_sha = util.shaHex(std.mem.sliceAsBytes(inputs[i]));
            const path = try std.fmt.allocPrint(self.a, "{s}/{s}/logits.f32", .{ self.staging, r.dir });
            defer self.a.free(path);
            const logits = std.Io.Dir.cwd().readFileAlloc(self.io, path, self.a, .limited(64 << 20)) catch return error.CalibWindowChanged;
            defer self.a.free(logits);
            const logits_sha = util.shaHex(logits);
            if (!std.mem.eql(u8, r.ids_sha256, &ids_sha) or !std.mem.eql(u8, r.logits_sha256, &logits_sha)) return error.CalibWindowChanged;
            offset += r.tokens;
        }
        self.committed_tokens = offset;
        if (self.records.len == 0) return;
        const bytes = try self.a.alloc(u8, @intCast(offset * 4));
        defer self.a.free(bytes);
        const path = try std.fmt.allocPrint(self.a, "{s}/tokens.bin", .{self.calib});
        defer self.a.free(path);
        var file = try std.Io.Dir.cwd().openFile(self.io, path, .{});
        defer file.close(self.io);
        if (file.readPositionalAll(self.io, bytes, 0) catch 0 != bytes.len) return error.CalibWindowChanged;
        var at: usize = 0;
        for (self.records, 0..) |r, i| {
            if (!std.mem.eql(u8, bytes[at * 4 ..][0 .. r.tokens * 4], std.mem.sliceAsBytes(inputs[i]))) return error.CalibWindowChanged;
            at += r.tokens;
        }
    }

    /// `tokens.bin` back to the committed windows: an interrupted batch's ids go.
    pub fn truncateTokens(self: *Ledger) !void {
        try truncateFd(self.tokens_fd, self.committed_tokens * 4);
    }

    /// Commits windows whose every file is synced: their ids into `tokens.bin`, then one line each.
    pub fn commit(self: *Ledger, inputs: []const []u32, lines: []const WindowRecord) !void {
        for (inputs) |ids| try writeAll(self.tokens_fd, std.mem.sliceAsBytes(ids));
        if (std.c.fsync(self.tokens_fd) != 0) return error.CalibCaptureWriteFailed;
        for (lines) |r| {
            const line = try std.json.Stringify.valueAlloc(self.a, r, .{});
            defer self.a.free(line);
            try util.appendLine(self.log_fd, line);
            self.committed_tokens += r.tokens;
        }
        if (std.c.fsync(self.log_fd) != 0) return error.LayerMajorSyncFailed;
    }

    /// Removes the run's recoverable leftovers except those of the first `keep` windows: imatrix states in the staging
    /// directory, trunk sums in `calib/`.
    pub fn dropStale(self: *Ledger, keep: usize) void {
        const cwd = std.Io.Dir.cwd();
        var keep_state: [64]u8 = undefined;
        const keep_name = std.fmt.bufPrint(&keep_state, "imatrix-{d}.safetensors", .{keep}) catch return;
        var keep_suffix: [32]u8 = undefined;
        const keep_sums = std.fmt.bufPrint(&keep_suffix, ".{d:0>6}.safetensors", .{keep}) catch return;
        if (cwd.openDir(self.io, self.staging, .{ .iterate = true })) |opened| {
            var dir = opened;
            defer dir.close(self.io);
            var it = dir.iterate();
            while (it.next(self.io) catch null) |entry| {
                if (std.mem.startsWith(u8, entry.name, "imatrix-") and !std.mem.eql(u8, entry.name, keep_name)) dir.deleteFile(self.io, entry.name) catch {};
            }
        } else |_| {}
        if (cwd.openDir(self.io, self.calib, .{ .iterate = true })) |opened| {
            var dir = opened;
            defer dir.close(self.io);
            var it = dir.iterate();
            while (it.next(self.io) catch null) |entry| {
                if (std.mem.startsWith(u8, entry.name, "trunk-") and !std.mem.endsWith(u8, entry.name, keep_sums)) dir.deleteFile(self.io, entry.name) catch {};
            }
        } else |_| {}
    }
};
