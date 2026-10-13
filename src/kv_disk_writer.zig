//! The SSD tier's background writer. The inference thread keeps the device->host readback and
//! hands one writer thread host byte buffers; no mlx handle crosses. Files land `tmp` + `rename`,
//! FIFO, so an entry's `meta.json` (enqueued last) is the last to land. A file larger than the
//! permit arrives in parts that append to one `tmp`, the last renaming it. A host-byte permit
//! bounds the staged bytes (~1 GiB); an epoch fence drops staged bytes for a directory about to
//! be removed. POSIX syscalls: this runs off the main thread. Nothing is fsynced: a rename that
//! landed survives a crash of the process, not a loss of power.

const std = @import("std");
const log = @import("log.zig");
const io_util = @import("io_util.zig");

/// One staged file, or one part of a file larger than the permit; both buffers are owned by the
/// queue once `submit` accepts them.
pub const Blob = struct {
    path: []u8,
    bytes: []u8,
    epoch: u64,
    part: Part = .whole,
};

/// A file staged in parts: the first truncates its `tmp`, the rest append, the last renames.
pub const Part = enum { whole, first, middle, last };

pub const DEFAULT_PERMIT_BYTES: u64 = 1024 * 1024 * 1024;
pub const MIN_PERMIT_BYTES: u64 = 128 * 1024 * 1024;

/// The permit on a host with `ram` bytes (0 = unknown: the cap): a 128th of it, floor to cap. The SSD
/// drains a prefill chunk's blob before the next is built, and admission bills the whole permit, so on a
/// small Mac a larger one is context taken from every request.
pub fn permitForRam(ram: u64) u64 {
    if (ram == 0) return DEFAULT_PERMIT_BYTES;
    return std.math.clamp(ram / 128, MIN_PERMIT_BYTES, DEFAULT_PERMIT_BYTES);
}

/// One blob the writer could not write; the tier invalidates the entry it belonged to.
/// `path` is owned by whoever takes it out of `takeFailures`.
pub const Failure = struct { path: []u8, err_name: []const u8 };

/// Where an injected failure strikes. See `Writer.fail_at`.
pub const FailAt = enum { write, submit };

/// Failed paths kept for attribution; past this `unattributed_failure` is raised.
pub const MAX_RECORDED_FAILURES: usize = 256;

pub const Writer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    work: std.Io.Condition = .init,
    done: std.Io.Condition = .init,
    queue: std.ArrayList(Blob) = .empty,
    pending_bytes: u64 = 0,
    inflight_bytes: u64 = 0,
    /// Valid only while `inflight_bytes > 0`; the blob owns the memory.
    inflight_path: ?[]const u8 = null,
    permit_bytes: u64 = DEFAULT_PERMIT_BYTES,
    epoch: u64 = 1,
    running: bool = false,
    /// Test-only: hold the queue so submission order can be inspected.
    paused: bool = false,
    deinited: bool = false,
    thread: ?std.Thread = null,
    /// Failed blobs since the last `takeFailures`.
    failures: std.ArrayList(Failure) = .empty,
    /// A failure that could not be recorded; read-and-clear via `takeUnattributed`.
    unattributed_failure: bool = false,
    /// Test-only: every blob whose path contains this substring fails like a full volume. Owned.
    fail_substr: ?[]u8 = null,
    fail_at: FailAt = .write,
    /// Diagnostics / test bars. Written under the mutex.
    files_written: u64 = 0,
    bytes_written: u64 = 0,
    files_dropped: u64 = 0,
    write_errors: u64 = 0,
    /// Time the producer spent blocked on the permit.
    waited_ns: u64 = 0,
    /// Most host bytes staged and in flight at once.
    peak_bytes: u64 = 0,
    /// A file whose earlier part failed or was dropped: its later parts are dropped too. Owned.
    broken_path: ?[]u8 = null,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) Writer {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn start(self: *Writer) !void {
        self.mutex.lockUncancelable(self.io);
        if (self.running) {
            self.mutex.unlock(self.io);
            return;
        }
        self.running = true;
        self.mutex.unlock(self.io);
        self.thread = std.Thread.spawn(.{}, loop, .{self}) catch |err| {
            self.mutex.lockUncancelable(self.io);
            self.running = false;
            self.mutex.unlock(self.io);
            return err;
        };
    }

    /// Drain, stop the thread, free anything left. Safe to call twice.
    pub fn deinit(self: *Writer) void {
        self.mutex.lockUncancelable(self.io);
        if (self.deinited) {
            self.mutex.unlock(self.io);
            return;
        }
        self.deinited = true;
        // Lift a pause and wake both condvars before stopping the loop, or `drain` parks forever.
        self.paused = false;
        self.work.broadcast(self.io);
        self.done.broadcast(self.io);
        self.mutex.unlock(self.io);
        self.mutex.lockUncancelable(self.io);
        self.running = false;
        self.paused = false;
        self.work.broadcast(self.io);
        self.done.broadcast(self.io);
        self.mutex.unlock(self.io);
        if (self.thread) |t| t.join();
        self.thread = null;
        self.mutex.lockUncancelable(self.io);
        for (self.queue.items) |*b| self.freeBlob(b);
        self.queue.clearRetainingCapacity();
        self.pending_bytes = 0;
        self.queue.deinit(self.allocator);
        for (self.failures.items) |f| self.allocator.free(f.path);
        self.failures.deinit(self.allocator);
        if (self.fail_substr) |fs| self.allocator.free(fs);
        self.fail_substr = null;
        if (self.broken_path) |bp| self.allocator.free(bp);
        self.broken_path = null;
        self.mutex.unlock(self.io);
    }

    fn freeBlob(self: *Writer, b: *Blob) void {
        self.allocator.free(b.path);
        self.allocator.free(b.bytes);
    }

    /// Stage one file. Takes ownership of both slices on every path, including errors.
    /// Blocks while the unwritten queue is over the permit (the only place the inference
    /// thread waits on the writer). Single producer: the inference thread.
    pub fn submit(self: *Writer, path: []u8, bytes: []u8) void {
        self.submitPart(path, bytes, .whole);
    }

    /// `submit` for one part of a file staged in parts (`Part`). A part that cannot be staged
    /// drops the rest of its file.
    pub fn submitPart(self: *Writer, path: []u8, bytes: []u8, part: Part) void {
        std.debug.assert(!self.deinited);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.running) {
            // No writer: dropping is correct, the index file rides the same queue.
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        }
        if (self.injectedLocked(path, .submit)) {
            log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedSubmitFailure", path });
            self.noteFailureLocked(path, "InjectedSubmitFailure");
            if (part != .whole) self.markBrokenLocked(path);
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        }
        self.waitForRoomLocked(bytes.len);
        self.queue.append(self.allocator, .{
            .path = path,
            .bytes = bytes,
            .epoch = self.epoch,
            .part = part,
        }) catch |err| {
            self.noteFailureLocked(path, @errorName(err));
            if (part != .whole) self.markBrokenLocked(path);
            self.allocator.free(path);
            self.allocator.free(bytes);
            self.files_dropped += 1;
            return;
        };
        self.pending_bytes += bytes.len;
        self.peak_bytes = @max(self.peak_bytes, self.pending_bytes + self.inflight_bytes);
        self.work.signal(self.io);
    }

    /// Caller holds the mutex. The later parts of `path` are dropped until its last.
    fn markBrokenLocked(self: *Writer, path: []const u8) void {
        if (self.broken_path) |bp| {
            if (std.mem.eql(u8, bp, path)) return;
            self.allocator.free(bp);
        }
        self.broken_path = self.allocator.dupe(u8, path) catch null;
    }

    fn isBrokenLocked(self: *const Writer, path: []const u8) bool {
        const bp = self.broken_path orelse return false;
        return std.mem.eql(u8, bp, path);
    }

    /// Caller holds the mutex: forget a broken file once its last part is gone.
    fn clearBrokenLocked(self: *Writer, path: []const u8) void {
        if (!self.isBrokenLocked(path)) return;
        self.allocator.free(self.broken_path.?);
        self.broken_path = null;
    }

    /// Block until `n` more staged bytes fit under the permit. Called before a blob's host bytes
    /// are allocated, so the bytes staged plus the one being built never pass the permit as long
    /// as no blob does (a larger file is staged in parts, `partBytes`); the single producer keeps
    /// that room until its `submit`.
    pub fn waitForRoom(self: *Writer, n: usize) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.running) return;
        self.waitForRoomLocked(n);
    }

    fn waitForRoomLocked(self: *Writer, n: usize) void {
        if (!self.overPermitLocked(n)) return;
        const sw = io_util.Stopwatch.init(self.io);
        while (self.overPermitLocked(n)) self.done.waitUncancelable(self.io, &self.mutex);
        self.waited_ns += sw.read();
    }

    fn overPermitLocked(self: *const Writer, n: usize) bool {
        return self.pending_bytes + self.inflight_bytes + n > self.permit_bytes and
            (self.queue.items.len > 0 or self.inflight_bytes > 0);
    }

    pub fn waitedNs(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.waited_ns;
    }

    /// The largest blob one file stages at once: a file past it goes in parts.
    pub fn partBytes(self: *const Writer) u64 {
        return @max(self.permit_bytes / 2, 1);
    }

    pub fn peakBytes(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.peak_bytes;
    }

    /// Wait until the files staged for `path_prefix` have been written (or dropped); null = all.
    pub fn drainPrefix(self: *Writer, path_prefix: ?[]const u8) void {
        const pre = path_prefix orelse {
            self.drain();
            return;
        };
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.running) {
            var waiting = false;
            if (self.inflight_path) |p| {
                if (std.mem.startsWith(u8, p, pre)) waiting = true;
            }
            if (!waiting) {
                for (self.queue.items) |b| {
                    if (std.mem.startsWith(u8, b.path, pre)) {
                        waiting = true;
                        break;
                    }
                }
            }
            if (!waiting) return;
            self.done.waitUncancelable(self.io, &self.mutex);
        }
    }

    /// Non-blocking twin of `drainPrefix`: is any blob for `path_prefix` still staged or in flight?
    pub fn pendingPrefix(self: *Writer, path_prefix: []const u8) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.inflight_path) |p| {
            if (std.mem.startsWith(u8, p, path_prefix)) return true;
        }
        for (self.queue.items) |b| {
            if (std.mem.startsWith(u8, b.path, path_prefix)) return true;
        }
        return false;
    }

    /// Wait until every staged file has been written (or dropped).
    pub fn drain(self: *Writer) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        while (self.running and (self.queue.items.len > 0 or self.inflight_bytes > 0)) {
            self.done.waitUncancelable(self.io, &self.mutex);
        }
    }

    /// Epoch fence: staged bytes for `path_prefix` (null = everything) are discarded rather than
    /// written, and anything in flight is waited out, so the caller can remove the directory.
    pub fn fence(self: *Writer, path_prefix: ?[]const u8) void {
        self.mutex.lockUncancelable(self.io);
        if (path_prefix == null) self.epoch += 1;
        // A file cut between its parts leaves its `tmp`, removed once the part in flight lands.
        var cut: ?[]u8 = null;
        defer if (cut) |c| self.allocator.free(c);
        var i: usize = 0;
        while (i < self.queue.items.len) {
            const b = &self.queue.items[i];
            const doomed = if (path_prefix) |pre| std.mem.startsWith(u8, b.path, pre) else true;
            if (!doomed) {
                i += 1;
                continue;
            }
            self.pending_bytes -|= b.bytes.len;
            var owned = self.queue.orderedRemove(i);
            if (owned.part != .whole and cut == null) cut = self.allocator.dupe(u8, owned.path) catch null;
            if (owned.part == .whole or owned.part == .last) self.clearBrokenLocked(owned.path);
            self.freeBlob(&owned);
            self.files_dropped += 1;
        }
        self.done.broadcast(self.io);
        while (self.running and self.inflight_bytes > 0) self.done.waitUncancelable(self.io, &self.mutex);
        self.mutex.unlock(self.io);
        if (cut) |c| unlinkTmp(c);
    }

    /// Test-only: hold / release the writer thread.
    /// Is a write to `path` still queued or in flight? Read-only on the queue.
    pub fn isPending(self: *Writer, path: []const u8) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.inflight_path) |p| {
            if (std.mem.eql(u8, p, path)) return true;
        }
        for (self.queue.items) |b| {
            if (std.mem.eql(u8, b.path, path)) return true;
        }
        return false;
    }

    pub fn setPaused(self: *Writer, v: bool) void {
        self.mutex.lockUncancelable(self.io);
        self.paused = v;
        self.work.broadcast(self.io);
        self.mutex.unlock(self.io);
    }

    /// Test-only: the staged paths in write order, duped into `a`. Caller frees each item.
    pub fn stagedPaths(self: *Writer, out: *std.ArrayList([]const u8), a: std.mem.Allocator) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.queue.items) |b| try out.append(a, try a.dupe(u8, b.path));
    }

    pub fn pendingBytes(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.pending_bytes + self.inflight_bytes;
    }

    /// Hand over every failure recorded since the last call; the caller owns the slice and each `path`.
    pub fn takeFailures(self: *Writer) []Failure {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.failures.toOwnedSlice(self.allocator) catch blk: {
            self.unattributed_failure = true;
            break :blk &[_]Failure{};
        };
    }

    /// Read-and-clear: did a failure go unrecorded since the last call?
    pub fn takeUnattributed(self: *Writer) bool {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const v = self.unattributed_failure;
        self.unattributed_failure = false;
        return v;
    }

    /// Test-only: fail every blob whose path contains `substr` (null clears), the way an ENOSPC does.
    pub fn injectFailure(self: *Writer, substr: ?[]const u8, at: FailAt) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.fail_substr) |fs| self.allocator.free(fs);
        self.fail_substr = if (substr) |sub| (self.allocator.dupe(u8, sub) catch null) else null;
        self.fail_at = at;
    }

    /// Caller holds the mutex.
    fn injectedLocked(self: *Writer, path: []const u8, at: FailAt) bool {
        if (self.fail_at != at) return false;
        const fs = self.fail_substr orelse return false;
        return std.mem.indexOf(u8, path, fs) != null;
    }

    /// Count + record one failed blob. Caller must NOT hold the mutex.
    fn noteFailure(self: *Writer, path: []const u8, err_name: []const u8) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.noteFailureLocked(path, err_name);
    }

    fn noteFailureLocked(self: *Writer, path: []const u8, err_name: []const u8) void {
        self.write_errors += 1;
        if (self.failures.items.len >= MAX_RECORDED_FAILURES) {
            self.unattributed_failure = true;
            return;
        }
        const p = self.allocator.dupe(u8, path) catch {
            self.unattributed_failure = true;
            return;
        };
        self.failures.append(self.allocator, .{ .path = p, .err_name = err_name }) catch {
            self.allocator.free(p);
            self.unattributed_failure = true;
        };
    }

    pub fn writeErrorCount(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.write_errors;
    }

    pub fn filesWritten(self: *Writer) u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.files_written;
    }

    fn loop(self: *Writer) void {
        while (true) {
            self.mutex.lockUncancelable(self.io);
            while (self.running and (self.paused or self.queue.items.len == 0)) self.work.waitUncancelable(self.io, &self.mutex);
            if (!self.running and self.queue.items.len == 0) {
                self.mutex.unlock(self.io);
                return;
            }
            var blob = self.queue.orderedRemove(0);
            self.pending_bytes -|= blob.bytes.len;
            self.inflight_bytes = blob.bytes.len;
            self.inflight_path = blob.path;
            self.mutex.unlock(self.io);

            // Re-read the epoch under the lock, immediately before the write.
            self.mutex.lockUncancelable(self.io);
            const live_epoch = self.epoch;
            const inject = self.injectedLocked(blob.path, .write);
            if (blob.part == .first) self.clearBrokenLocked(blob.path);
            const broken = (blob.part == .middle or blob.part == .last) and self.isBrokenLocked(blob.path);
            self.mutex.unlock(self.io);

            var dropped = false;
            if (broken or blob.epoch != live_epoch) {
                dropped = true;
            } else if (inject) {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedWriteFailure", blob.path });
                self.noteFailure(blob.path, "InjectedWriteFailure");
                dropped = true;
            } else if (writePart(blob.path, blob.bytes, blob.part)) |_| {} else |err| {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ @errorName(err), blob.path });
                self.noteFailure(blob.path, @errorName(err));
                dropped = true;
            }
            if (dropped and blob.part != .whole) unlinkTmp(blob.path);

            self.mutex.lockUncancelable(self.io);
            if (blob.part == .first or blob.part == .middle) {
                if (dropped) self.markBrokenLocked(blob.path);
            } else self.clearBrokenLocked(blob.path);
            if (dropped) {
                self.files_dropped += 1;
            } else {
                if (blob.part == .whole or blob.part == .last) self.files_written += 1;
                self.bytes_written += blob.bytes.len;
            }
            self.inflight_bytes = 0;
            self.inflight_path = null;
            self.freeBlob(&blob);
            self.done.broadcast(self.io);
            self.mutex.unlock(self.io);
        }
    }
};

/// The staging file then rename.
fn writeAtomic(path: []const u8, bytes: []const u8) !void {
    return writePart(path, bytes, .whole);
}

const TmpBuf = [std.fs.max_path_bytes + 24]u8;

/// `<path>.<pid>.tmp`: a staging name no other process writes.
fn tmpPathZ(buf: *TmpBuf, path: []const u8) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}.{d}.tmp", .{ path, std.c.getpid() }, 0) catch error.NameTooLong;
}

/// Remove `path`'s staging file; a caller only does this once no part of it is queued or in flight.
pub fn unlinkTmp(path: []const u8) void {
    var tmp_buf: TmpBuf = undefined;
    const tmp = tmpPathZ(&tmp_buf, path) catch return;
    _ = std.c.unlink(tmp.ptr);
}

/// One part of `path` (`Part`): a whole or first part truncates its `tmp`, the others append; a
/// whole or last part renames it into place.
fn writePart(path: []const u8, bytes: []const u8, part: Part) !void {
    var tmp_buf: TmpBuf = undefined;
    const tmp = try tmpPathZ(&tmp_buf, path);
    const fresh = part == .whole or part == .first;
    const fd = std.c.open(tmp.ptr, .{ .ACCMODE = .WRONLY, .CREAT = fresh, .TRUNC = fresh, .APPEND = !fresh }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.OpenFailed;
    errdefer _ = std.c.unlink(tmp.ptr);
    {
        defer _ = std.c.close(fd);
        var off: usize = 0;
        while (off < bytes.len) {
            const n = std.c.write(fd, bytes.ptr + off, bytes.len - off);
            if (n < 0) {
                const e = std.c._errno().*;
                if (e == @intFromEnum(std.c.E.INTR) or e == @intFromEnum(std.c.E.AGAIN)) continue;
                return error.WriteFailed;
            }
            if (n == 0) return error.WriteFailed;
            off += @intCast(n);
        }
    }
    if (part == .first or part == .middle) return;

    var final_buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= final_buf.len) return error.NameTooLong;
    @memcpy(final_buf[0..path.len], path);
    final_buf[path.len] = 0;
    const final: [:0]const u8 = final_buf[0..path.len :0];
    if (std.c.rename(tmp.ptr, final.ptr) != 0) return error.RenameFailed;
}

// ── Tests ──

const testing = std.testing;

test "kv_disk_writer: files land off-thread, in FIFO order, and atomically" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    try w.start();
    defer w.deinit();

    var i: usize = 0;
    while (i < 10) : (i += 1) {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/f{d}.bin", .{ root, i });
        const bytes = try testing.allocator.alloc(u8, 4096);
        @memset(bytes, @intCast(i));
        w.submit(path, bytes);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 10), w.filesWritten());
    try testing.expectEqual(@as(u64, 0), w.pendingBytes());

    i = 0;
    while (i < 10) : (i += 1) {
        var name: [64]u8 = undefined;
        const n = try std.fmt.bufPrint(&name, "f{d}.bin", .{i});
        const got = try tmp.dir.readFileAlloc(std.testing.io, n, testing.allocator, .limited(1 << 20));
        defer testing.allocator.free(got);
        try testing.expectEqual(@as(usize, 4096), got.len);
        try testing.expectEqual(@as(u8, @intCast(i)), got[0]);
    }
    try expectNoStagingFile(tmp.dir);
}

/// No staging file is left in `dir`, whatever its name.
fn expectNoStagingFile(dir: std.Io.Dir) !void {
    var it = dir.iterate();
    while (try it.next(testing.io)) |dent| try testing.expect(!std.mem.endsWith(u8, dent.name, ".tmp"));
}

test "kv_disk_writer: the epoch fence drops staged bytes instead of writing them" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    const p0 = try std.fmt.allocPrint(testing.allocator, "{s}/never.bin", .{root});
    const b0 = try testing.allocator.alloc(u8, 16);
    w.submit(p0, b0);
    try testing.expectEqual(@as(u64, 1), w.files_dropped);

    try w.start();
    defer w.deinit();
    w.fence(null);
    const after_fence = w.epoch;
    try testing.expect(after_fence > 1);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, "never.bin", .{}));
}

test "kv_disk_writer: the permit scales with host RAM between its floor and its cap" {
    const gib: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 256 << 20), permitForRam(32 * gib));
    try testing.expectEqual(@as(u64, 512 << 20), permitForRam(64 * gib));
    try testing.expectEqual(DEFAULT_PERMIT_BYTES, permitForRam(128 * gib));
    try testing.expectEqual(DEFAULT_PERMIT_BYTES, permitForRam(512 * gib));
    try testing.expectEqual(MIN_PERMIT_BYTES, permitForRam(8 * gib));
    try testing.expectEqual(DEFAULT_PERMIT_BYTES, permitForRam(0));
}

test "kv_disk_writer: the host-byte permit bounds staged bytes" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    w.permit_bytes = 64 * 1024;
    try w.start();
    defer w.deinit();

    var i: usize = 0;
    while (i < 32) : (i += 1) {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/p{d}.bin", .{ root, i });
        const bytes = try testing.allocator.alloc(u8, 16 * 1024);
        @memset(bytes, 7);
        w.submit(path, bytes);
        try testing.expect(w.pendingBytes() <= w.permit_bytes + 16 * 1024);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 32), w.filesWritten());
}

test "kv_disk_writer: room for a file is waited for before its bytes exist" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    w.permit_bytes = 64 * 1024;
    try w.start();
    defer w.deinit();
    w.setPaused(true);
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/r{d}.bin", .{ root, i });
        const bytes = try testing.allocator.alloc(u8, 16 * 1024);
        @memset(bytes, 1);
        w.submit(path, bytes);
    }
    const Waiter = struct {
        fn run(wr: *Writer, got: *std.atomic.Value(bool)) void {
            wr.waitForRoom(16 * 1024);
            got.store(true, .release);
        }
    };
    var got = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, Waiter.run, .{ &w, &got });
    // The permit is full and the writer paused: the room cannot exist yet.
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .real) catch {};
    try testing.expect(!got.load(.acquire));
    w.setPaused(false);
    t.join();
    try testing.expect(got.load(.acquire));
    try testing.expect(w.pendingBytes() + 16 * 1024 <= w.permit_bytes);
    try testing.expect(w.waitedNs() > 0);
}

test "kv_disk_writer: a file staged in parts lands whole at its last part, and a failed part publishes nothing" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    w.permit_bytes = 64 * 1024;
    try w.start();
    defer w.deinit();
    const parts = [_]Part{ .first, .middle, .last };
    w.setPaused(true);
    for (parts, 0..) |part, i| {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/big.bin", .{root});
        const bytes = try testing.allocator.alloc(u8, 16 * 1024);
        @memset(bytes, @intCast(i + 1));
        w.submitPart(path, bytes, part);
    }
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, "big.bin", .{}));
    w.setPaused(false);
    w.drain();
    const got = try tmp.dir.readFileAlloc(std.testing.io, "big.bin", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(usize, 48 * 1024), got.len);
    for (0..3) |i| try testing.expectEqual(@as(u8, @intCast(i + 1)), got[i * 16 * 1024]);
    try expectNoStagingFile(tmp.dir);
    try testing.expectEqual(@as(u64, 1), w.filesWritten());

    // The first part fails like a full volume: the rest of the file is dropped, nothing lands.
    w.injectFailure("broken.bin", .write);
    for (parts) |part| {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/broken.bin", .{root});
        w.submitPart(path, try testing.allocator.alloc(u8, 1024), part);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 1), w.writeErrorCount());
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(std.testing.io, "broken.bin", .{}));
    try expectNoStagingFile(tmp.dir);
    // The same path written afresh lands.
    w.injectFailure(null, .write);
    const again = try std.fmt.allocPrint(testing.allocator, "{s}/broken.bin", .{root});
    const bytes = try testing.allocator.alloc(u8, 8);
    @memset(bytes, 9);
    w.submit(again, bytes);
    w.drain();
    try testing.expectEqual(@as(u64, 8), (try tmp.dir.statFile(std.testing.io, "broken.bin", .{})).size);
}

test "kv_disk_writer: a PAUSED writer deinits without blocking" {
    // A test that pauses the writer and then fails must not hang the suite.
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];

    var w = Writer.init(testing.allocator, std.testing.io);
    try w.start();
    w.setPaused(true);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/held.bin", .{root});
    const bytes = try testing.allocator.alloc(u8, 4096);
    @memset(bytes, 3);
    w.submit(path, bytes);
    try testing.expect(w.pendingBytes() > 0);

    w.deinit();
    try testing.expect(w.thread == null);
    w.deinit();
}

test "kv_disk_writer: failed publication removes staged temporary bytes" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDir(testing.io, "blocked", .default_dir);
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/blocked", .{root});
    defer testing.allocator.free(path);

    try testing.expectError(error.RenameFailed, writeAtomic(path, "unpublished bytes"));
    try expectNoStagingFile(tmp.dir);
}

test "kv_disk_writer: queue allocation failure invalidates persistence" {
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    var writer = Writer.init(failing.allocator(), testing.io);
    defer writer.deinit();
    writer.running = true;
    const path = try testing.allocator.dupe(u8, "e1/c000000.safetensors");
    const bytes = try testing.allocator.dupe(u8, "chunk bytes");

    writer.submit(path, bytes);

    try testing.expectEqual(@as(u64, 1), writer.writeErrorCount());
    try testing.expect(writer.takeUnattributed());
    try testing.expectEqual(@as(u64, 0), writer.pendingBytes());
}
