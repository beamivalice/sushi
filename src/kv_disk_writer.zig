//! The SSD tier's background writer. The inference thread keeps the device->host readback and
//! hands one writer thread host byte buffers; no mlx handle crosses. Files land `tmp` + `rename`,
//! FIFO, so an entry's `meta.json` (enqueued last) is the last to land. A file larger than the
//! permit arrives in parts that append to one `tmp`, the last renaming it. A host-byte permit
//! bounds the staged bytes (~1 GiB); an epoch fence drops staged bytes for a directory about to
//! be removed. POSIX syscalls: this runs off the main thread. Nothing is fsynced: a rename that
//! landed survives a crash of the process, not a loss of power.
//!
//! An op (`submitOp`) is a virtual file of the content-addressed layer: it rides the same FIFO, permit and
//! fence, but lands by `OpSink` steps instead of `tmp` + `rename`. It stays pending and unpublished for as
//! long as the file write of its logical bytes would take (`pace_bytes_per_s`), so no decision reads a
//! landed write sooner than it does with plain files.

const std = @import("std");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const write_stats = @import("disk_write_stats.zig");

/// One staged file, or one part of a file larger than the permit; both buffers are owned by the
/// queue once `submit` accepts them.
pub const Blob = struct {
    path: []u8,
    bytes: []u8,
    epoch: u64,
    part: Part = .whole,
    kind: write_stats.Kind,
    /// Set for an op: `bytes` is then empty and the sink holds the payload.
    op: ?QueuedOp = null,

    fn cost(self: Blob) u64 {
        return if (self.op) |o| o.host_bytes else self.bytes.len;
    }
};

/// Landed bytes of one op, kept until `takeLanding` reads them.
pub const Landing = struct { ticket: u64, bytes: u64 };
pub const MAX_LANDINGS: usize = 1024;

/// The content-addressed layer's side of an op. The sink owns the ticket's payload; the writer sequences
/// the steps on its thread (or the caller's, `landNow`) and calls exactly one of `publish` and `abort`.
pub const OpSink = struct {
    ctx: *anyopaque,
    /// Hash the payload and append the blocks the index lacks to the active pack. Returns the pack bytes
    /// appended. The blocks are unreferenced until `journalRecord`, so a stop here leaves no visible file.
    appendBlocks: *const fn (ctx: *anyopaque, ticket: u64) anyerror!u64,
    /// Append the ticket's journal record. Returns the journal bytes appended.
    journalRecord: *const fn (ctx: *anyopaque, ticket: u64) anyerror!u64,
    /// Make the op visible to `stat`: the counterpart of a file's `rename`.
    publish: *const fn (ctx: *anyopaque, ticket: u64) void,
    /// Release the ticket without publishing it; blocks already appended are reclaimed as dead.
    abort: *const fn (ctx: *anyopaque, ticket: u64) void,
};

/// `host_bytes` is what the staged payload holds in RAM (the permit); `logical_bytes` is the size of the file
/// the plain path would write (the pace).
pub const Op = struct { ticket: u64, host_bytes: u64, logical_bytes: u64 };

const QueuedOp = struct { sink: OpSink, ticket: u64, host_bytes: u64, logical_bytes: u64 };

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
    /// Where written and failed bytes are reported, by kind. Null in a bare writer.
    stats: ?*write_stats.Stats = null,
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
    /// Throughput of the plain file write, which an op may not beat (0 = no floor). Set before `start`.
    pace_bytes_per_s: u64 = 0,
    /// Serializes the pack and journal appends of the writer thread with `landNow`.
    land_mutex: std.Io.Mutex = .init,
    /// Landed bytes per op since `takeLanding` read them; the oldest go past `MAX_LANDINGS`.
    landings: std.ArrayList(Landing) = .empty,

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
        self.landings.deinit(self.allocator);
        if (self.fail_substr) |fs| self.allocator.free(fs);
        self.fail_substr = null;
        if (self.broken_path) |bp| self.allocator.free(bp);
        self.broken_path = null;
        self.mutex.unlock(self.io);
    }

    fn noteWritten(self: *Writer, kind: write_stats.Kind, n: usize) void {
        if (self.stats) |st| st.recordWritten(kind, n);
    }

    fn noteStored(self: *Writer, kind: write_stats.Kind, n: u64) void {
        if (self.stats) |st| st.recordStored(kind, n);
    }

    fn noteFailed(self: *Writer, n: usize) void {
        if (self.stats) |st| st.recordFailed(n);
    }

    fn freeBlob(self: *Writer, b: *Blob) void {
        if (b.op) |o| o.sink.abort(o.sink.ctx, o.ticket);
        self.allocator.free(b.path);
        self.allocator.free(b.bytes);
    }

    /// Stage one file. Takes ownership of both slices on every path, including errors.
    /// Blocks while the unwritten queue is over the permit (the only place the inference
    /// thread waits on the writer). Single producer: the inference thread.
    pub fn submit(self: *Writer, path: []u8, bytes: []u8, kind: write_stats.Kind) void {
        self.submitPart(path, bytes, .whole, kind);
    }

    /// `submit` for one part of a file staged in parts (`Part`). A part that cannot be staged
    /// drops the rest of its file.
    pub fn submitPart(self: *Writer, path: []u8, bytes: []u8, part: Part, kind: write_stats.Kind) void {
        self.stage(.{ .path = path, .bytes = bytes, .epoch = 0, .part = part, .kind = kind });
    }

    /// Stage one op of the content-addressed layer behind the files already queued. Takes ownership of
    /// `path` and of the ticket on every path, including errors (a dropped ticket is aborted).
    pub fn submitOp(self: *Writer, path: []u8, sink: OpSink, op: Op, kind: write_stats.Kind) void {
        self.submitOpPart(path, sink, op, .whole, kind);
    }

    /// `submitOp` for one part of a ticket held in parts (`Vfs.holdPart`): each part appends its blocks, the
    /// last also journals and publishes. A part that cannot land drops the rest of its file, as a file's does.
    pub fn submitOpPart(self: *Writer, path: []u8, sink: OpSink, op: Op, part: Part, kind: write_stats.Kind) void {
        self.stage(.{
            .path = path,
            .bytes = @constCast(@as([]const u8, &.{})),
            .epoch = 0,
            .part = part,
            .kind = kind,
            .op = .{ .sink = sink, .ticket = op.ticket, .host_bytes = op.host_bytes, .logical_bytes = op.logical_bytes },
        });
    }

    fn stage(self: *Writer, staged: Blob) void {
        var blob = staged;
        const part = blob.part;
        const cost = blob.cost();
        std.debug.assert(!self.deinited);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (!self.running) {
            // No writer: dropping is correct, the index file rides the same queue.
            self.noteFailed(cost);
            self.freeBlob(&blob);
            self.files_dropped += 1;
            return;
        }
        if (self.injectedLocked(blob.path, .submit)) {
            log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedSubmitFailure", blob.path });
            self.noteFailureLocked(blob.path, "InjectedSubmitFailure");
            if (part != .whole) self.markBrokenLocked(blob.path);
            self.noteFailed(cost);
            self.freeBlob(&blob);
            self.files_dropped += 1;
            return;
        }
        self.waitForRoomLocked(cost);
        blob.epoch = self.epoch;
        self.queue.append(self.allocator, blob) catch |err| {
            self.noteFailureLocked(blob.path, @errorName(err));
            if (part != .whole) self.markBrokenLocked(blob.path);
            self.noteFailed(cost);
            self.freeBlob(&blob);
            self.files_dropped += 1;
            return;
        };
        self.pending_bytes += cost;
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
            self.pending_bytes -|= b.cost();
            var owned = self.queue.orderedRemove(i);
            if (owned.part != .whole and cut == null) cut = self.allocator.dupe(u8, owned.path) catch null;
            if (owned.part == .whole or owned.part == .last) self.clearBrokenLocked(owned.path);
            self.noteFailed(owned.cost());
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

    /// Caller holds the mutex.
    fn recordLandingLocked(self: *Writer, ticket: u64, bytes: u64) void {
        for (self.landings.items) |*l| if (l.ticket == ticket) {
            l.bytes += bytes;
            return;
        };
        if (self.landings.items.len >= MAX_LANDINGS) _ = self.landings.orderedRemove(0);
        self.landings.append(self.allocator, .{ .ticket = ticket, .bytes = bytes }) catch {};
    }

    /// The pack and journal bytes op `ticket` landed, once; null if unknown or already taken.
    pub fn takeLanding(self: *Writer, ticket: u64) ?u64 {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        for (self.landings.items, 0..) |l, i| {
            if (l.ticket != ticket) continue;
            _ = self.landings.orderedRemove(i);
            return l.bytes;
        }
        return null;
    }

    /// Land `ticket` on the calling thread, as a synchronous syscall of the plain path does: no queue, no
    /// pace. The caller blocks until it is published, so it cannot read it sooner than the plain path would.
    /// Returns the landed bytes; on error the ticket is aborted.
    pub fn landNow(self: *Writer, sink: OpSink, ticket: u64, kind: write_stats.Kind) !u64 {
        self.land_mutex.lockUncancelable(self.io);
        const landed = landSteps(sink, ticket);
        self.land_mutex.unlock(self.io);
        const n = landed catch |err| {
            sink.abort(sink.ctx, ticket);
            return err;
        };
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.noteStored(kind, n);
        self.recordLandingLocked(ticket, n);
        return n;
    }

    /// Append the blocks, wait out the file write's duration, then (`finish`: a whole file or its last part)
    /// journal and publish. The blocks are unreferenced until the journal record, so the wait leaves the
    /// namespace as the plain path's `tmp` does.
    fn landPaced(self: *Writer, op: QueuedOp, finish: bool) !u64 {
        const started = io_util.Stopwatch.init(self.io);
        self.land_mutex.lockUncancelable(self.io);
        const appended = op.sink.appendBlocks(op.sink.ctx, op.ticket);
        self.land_mutex.unlock(self.io);
        const pack = try appended;
        self.padToFloor(started, op.logical_bytes);
        if (!finish) return pack;
        self.land_mutex.lockUncancelable(self.io);
        defer self.land_mutex.unlock(self.io);
        return pack + try finishSteps(op.sink, op.ticket);
    }

    /// Sleep until `logical_bytes` at the pace rate have elapsed since `started`.
    fn padToFloor(self: *Writer, started: io_util.Stopwatch, logical_bytes: u64) void {
        if (self.pace_bytes_per_s == 0) return;
        const floor_ns: u64 = @intCast(@min(@as(u128, logical_bytes) * std.time.ns_per_s / self.pace_bytes_per_s, std.math.maxInt(u64)));
        while (true) {
            const elapsed = started.read();
            if (elapsed >= floor_ns) return;
            std.Io.sleep(self.io, .fromNanoseconds(@intCast(floor_ns - elapsed)), .real) catch {};
        }
    }

    /// Measure the file-write rate of the volume under `dir_abs` with the plain path's own write, and make
    /// it the pace floor. A probe file is written and removed.
    pub fn calibratePace(self: *Writer, dir_abs: []const u8) !void {
        const probe_bytes: usize = 64 * 1024 * 1024;
        const buf = try self.allocator.alloc(u8, probe_bytes);
        defer self.allocator.free(buf);
        @memset(buf, 0xA5);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/.pace-probe", .{dir_abs});
        const sw = io_util.Stopwatch.init(self.io);
        try writeAtomic(path, buf);
        const ns = @max(sw.read(), 1);
        unlinkTmp(path);
        var z: [std.fs.max_path_bytes + 1]u8 = undefined;
        @memcpy(z[0..path.len], path);
        z[path.len] = 0;
        _ = std.c.unlink(@ptrCast(&z));
        self.pace_bytes_per_s = @max(@as(u64, @intCast(@as(u128, probe_bytes) * std.time.ns_per_s / ns)), 1);
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
            const cost = blob.cost();
            self.pending_bytes -|= cost;
            self.inflight_bytes = cost;
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
            var landed: u64 = blob.bytes.len;
            var landed_ticket: ?u64 = null;
            const logical: u64 = if (blob.op) |o| o.logical_bytes else blob.bytes.len;
            const is_op = blob.op != null;
            const finish = blob.part == .whole or blob.part == .last;
            if (broken or blob.epoch != live_epoch) {
                dropped = true;
            } else if (inject) {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ "InjectedWriteFailure", blob.path });
                self.noteFailure(blob.path, "InjectedWriteFailure");
                dropped = true;
            } else if (blob.op) |op| {
                if (self.landPaced(op, finish)) |n| {
                    landed = n;
                    landed_ticket = op.ticket;
                    blob.op = null;
                    if (finish) noteLanding();
                } else |err| {
                    log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ @errorName(err), blob.path });
                    self.noteFailure(blob.path, @errorName(err));
                    dropped = true;
                }
            } else if (writePart(blob.path, blob.bytes, blob.part)) |_| {
                if (blob.part == .whole or blob.part == .last) noteLanding();
            } else |err| {
                log.warn("  [disk-cache] background write failed: {s} ({s})\n", .{ @errorName(err), blob.path });
                self.noteFailure(blob.path, @errorName(err));
                dropped = true;
            }
            if (dropped and blob.part != .whole and !is_op) unlinkTmp(blob.path);

            self.mutex.lockUncancelable(self.io);
            if (blob.part == .first or blob.part == .middle) {
                if (dropped) self.markBrokenLocked(blob.path);
            } else self.clearBrokenLocked(blob.path);
            if (dropped) {
                self.files_dropped += 1;
                self.noteFailed(cost);
            } else {
                if (blob.part == .whole or blob.part == .last) self.files_written += 1;
                self.bytes_written += landed;
                // An op counts its file's bytes as written and what the layer added as stored.
                self.noteWritten(blob.kind, logical);
                if (landed_ticket) |t| {
                    self.noteStored(blob.kind, landed);
                    self.recordLandingLocked(t, landed);
                }
            }
            self.inflight_bytes = 0;
            self.inflight_path = null;
            self.freeBlob(&blob);
            self.done.broadcast(self.io);
            self.mutex.unlock(self.io);
        }
    }
};

/// Journal then publish: after this the op is visible. Returns the journal bytes.
fn finishSteps(sink: OpSink, ticket: u64) !u64 {
    const journal = try sink.journalRecord(sink.ctx, ticket);
    sink.publish(sink.ctx, ticket);
    return journal;
}

/// All steps of a synchronous op.
fn landSteps(sink: OpSink, ticket: u64) !u64 {
    const pack = try sink.appendBlocks(sink.ctx, ticket);
    return pack + try finishSteps(sink, ticket);
}

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
                if (e == @backingInt(std.c.E.INTR) or e == @backingInt(std.c.E.AGAIN)) continue;
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

var landings: std.atomic.Value(u64) = .init(0);

/// Files and ops landed in this process so far: what `SUSHI_TEST_CRASH_AFTER_LANDINGS` counts.
pub fn landingCount() u64 {
    return landings.load(.monotonic);
}
var crash_after_read: bool = false;
var crash_after: ?u64 = null;

pub fn parseCrashAfter(raw: ?[*:0]const u8) ?u64 {
    const v = raw orelse return null;
    const k = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return null;
    return if (k == 0) null else k;
}

/// Test seam, `SUSHI_TEST_CRASH_AFTER_LANDINGS=k`: the process is killed once the writer has landed k of a tier's
/// files (a rename, or an op's publish; never the pace probe), so a crash test can cut it between two landings.
fn noteLanding() void {
    if (!crash_after_read) {
        crash_after = parseCrashAfter(std.c.getenv("SUSHI_TEST_CRASH_AFTER_LANDINGS"));
        crash_after_read = true;
    }
    const n = landings.fetchAdd(1, .monotonic) + 1;
    if (crash_after == n) _ = std.c.kill(std.c.getpid(), .KILL);
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
        w.submit(path, bytes, .kv);
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
    w.submit(p0, b0, .kv);
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
        w.submit(path, bytes, .kv);
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
        w.submit(path, bytes, .kv);
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
        w.submitPart(path, bytes, part, .kv);
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
        w.submitPart(path, try testing.allocator.alloc(u8, 1024), part, .kv);
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
    w.submit(again, bytes, .kv);
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
    w.submit(path, bytes, .kv);
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

    writer.submit(path, bytes, .kv);

    try testing.expectEqual(@as(u64, 1), writer.writeErrorCount());
    try testing.expect(writer.takeUnattributed());
    try testing.expectEqual(@as(u64, 0), writer.pendingBytes());
}

const StatKind = write_stats.Kind;

fn stagedBlob(root: []const u8, name: []const u8, len: usize) !struct { path: []u8, bytes: []u8 } {
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ root, name });
    const bytes = try testing.allocator.alloc(u8, len);
    @memset(bytes, 5);
    return .{ .path = path, .bytes = bytes };
}

test "kv_disk_writer: landed bytes count under their kind, a rewrite counts its full bytes again" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, std.testing.io);
    w.stats = &stats;
    try w.start();
    defer w.deinit();

    const plan = [_]struct { name: []const u8, len: usize, kind: StatKind }{
        .{ .name = "c000000.safetensors", .len = 4000, .kind = .kv },
        .{ .name = "s0000128.safetensors", .len = 300, .kind = .ssm },
        .{ .name = "qsa.safetensors", .len = 200, .kind = .qsa },
        .{ .name = "qsa.safetensors", .len = 260, .kind = .qsa },
        .{ .name = "spec.safetensors", .len = 90, .kind = .spec },
        .{ .name = "r0000128.safetensors", .len = 70, .kind = .ring },
        .{ .name = "meta.json", .len = 11, .kind = .meta },
    };
    for (plan) |p| {
        const b = try stagedBlob(root, p.name, p.len);
        w.submit(b.path, b.bytes, p.kind);
    }
    w.drain();
    const s = stats.snapshot();
    try testing.expectEqual(@as(u64, 4000), s.written[@backingInt(StatKind.kv)]);
    try testing.expectEqual(@as(u64, 300), s.written[@backingInt(StatKind.ssm)]);
    try testing.expectEqual(@as(u64, 460), s.written[@backingInt(StatKind.qsa)]);
    try testing.expectEqual(@as(u64, 90), s.written[@backingInt(StatKind.spec)]);
    try testing.expectEqual(@as(u64, 70), s.written[@backingInt(StatKind.ring)]);
    try testing.expectEqual(@as(u64, 11), s.written[@backingInt(StatKind.meta)]);
    try testing.expectEqual(@as(u64, 0), s.failed);
    try testing.expectEqual(w.bytes_written, s.totalWritten());
}

test "kv_disk_writer: a failed, dropped or fenced blob lands in failed, never in written" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, std.testing.io);
    w.stats = &stats;

    const unarmed = try stagedBlob(root, "c000000.safetensors", 50);
    w.submit(unarmed.path, unarmed.bytes, .kv);
    try testing.expectEqual(@as(u64, 50), stats.snapshot().failed);

    try w.start();
    defer w.deinit();
    w.injectFailure("c000001", .write);
    const write_fail = try stagedBlob(root, "c000001.safetensors", 1000);
    w.submit(write_fail.path, write_fail.bytes, .kv);
    w.drain();
    w.injectFailure("c000002", .submit);
    const submit_fail = try stagedBlob(root, "c000002.safetensors", 700);
    w.submit(submit_fail.path, submit_fail.bytes, .kv);
    w.injectFailure(null, .write);

    w.setPaused(true);
    const fenced = try stagedBlob(root, "r0000128.safetensors", 400);
    w.submit(fenced.path, fenced.bytes, .ring);
    w.fence(root);
    w.setPaused(false);
    w.drain();

    const s = stats.snapshot();
    try testing.expectEqual(@as(u64, 50 + 1000 + 700 + 400), s.failed);
    try testing.expectEqual(@as(u64, 0), s.totalWritten());
}

test "kv_disk_writer: a file staged in parts counts the parts that landed, a broken file's rest as failed" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, std.testing.io);
    w.stats = &stats;
    try w.start();
    defer w.deinit();

    const parts = [_]Part{ .first, .middle, .last };
    for (parts) |part| {
        const b = try stagedBlob(root, "c000000.safetensors", 1024);
        w.submitPart(b.path, b.bytes, part, .kv);
    }
    w.drain();
    try testing.expectEqual(@as(u64, 3072), stats.snapshot().written[@backingInt(StatKind.kv)]);

    w.injectFailure("c000001", .write);
    for (parts) |part| {
        const b = try stagedBlob(root, "c000001.safetensors", 512);
        w.submitPart(b.path, b.bytes, part, .kv);
    }
    w.drain();
    const s = stats.snapshot();
    try testing.expectEqual(@as(u64, 3072), s.written[@backingInt(StatKind.kv)]);
    try testing.expectEqual(@as(u64, 1536), s.failed);
}

// ── Op tests: a content-addressed VFS op rides the same queue as a file blob ──

const FakeSink = struct {
    io: std.Io,
    mu: std.Io.Mutex = .init,
    clock: io_util.Stopwatch,
    published: std.ArrayList(u64) = .empty,
    published_ns: std.ArrayList(u64) = .empty,
    aborted: std.ArrayList(u64) = .empty,
    /// New block bytes a ticket appends to the pack (index = ticket), and the bytes of its journal record.
    blocks: [32]u64 = @splat(1000),
    journal: u64 = 100,
    pack_bytes: u64 = 0,
    journal_bytes: u64 = 0,
    fail_append: ?u64 = null,
    fail_journal: ?u64 = null,
    land_ns: u64 = 0,
    in_land: std.atomic.Value(bool) = .init(false),
    overlapped: std.atomic.Value(bool) = .init(false),

    fn init(io: std.Io) FakeSink {
        return .{ .io = io, .clock = io_util.Stopwatch.init(io) };
    }

    fn deinit(self: *FakeSink) void {
        self.published.deinit(testing.allocator);
        self.published_ns.deinit(testing.allocator);
        self.aborted.deinit(testing.allocator);
    }

    fn sink(self: *FakeSink) OpSink {
        return .{ .ctx = self, .appendBlocks = appendBlocks, .journalRecord = journalRecord, .publish = publish, .abort = abort };
    }

    fn appendBlocks(ctx: *anyopaque, ticket: u64) anyerror!u64 {
        const self: *FakeSink = @ptrCast(@alignCast(ctx));
        if (self.in_land.swap(true, .acq_rel)) self.overlapped.store(true, .release);
        if (self.land_ns > 0) std.Io.sleep(self.io, .fromNanoseconds(@intCast(self.land_ns)), .real) catch {};
        if (self.fail_append == ticket) return error.InjectedAppend;
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.pack_bytes += self.blocks[ticket];
        return self.blocks[ticket];
    }

    fn journalRecord(ctx: *anyopaque, ticket: u64) anyerror!u64 {
        const self: *FakeSink = @ptrCast(@alignCast(ctx));
        if (self.fail_journal == ticket) return error.InjectedJournal;
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.journal_bytes += self.journal;
        return self.journal;
    }

    fn publish(ctx: *anyopaque, ticket: u64) void {
        const self: *FakeSink = @ptrCast(@alignCast(ctx));
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.published.append(testing.allocator, ticket) catch unreachable;
        self.published_ns.append(testing.allocator, self.clock.read()) catch unreachable;
        self.in_land.store(false, .release);
    }

    fn abort(ctx: *anyopaque, ticket: u64) void {
        const self: *FakeSink = @ptrCast(@alignCast(ctx));
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.aborted.append(testing.allocator, ticket) catch unreachable;
        self.in_land.store(false, .release);
    }

    fn isVisible(self: *FakeSink, ticket: u64) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return std.mem.indexOfScalar(u64, self.published.items, ticket) != null;
    }
};

fn opPath(name: []const u8) ![]u8 {
    return std.fmt.allocPrint(testing.allocator, "v/e1/{s}", .{name});
}

fn submitFakeOp(w: *Writer, fake: *FakeSink, name: []const u8, ticket: u64, host: u64, kind: StatKind) !void {
    w.submitOp(try opPath(name), fake.sink(), .{ .ticket = ticket, .host_bytes = host, .logical_bytes = host }, kind);
}

test "kv_disk_writer op: visible only once published, in FIFO order beside file blobs" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    try w.start();
    defer w.deinit();

    w.setPaused(true);
    try submitFakeOp(&w, &fake, "c000000", 1, 4096, .kv);
    const file = try stagedBlob(root, "mid.bin", 64);
    w.submit(file.path, file.bytes, .meta);
    try submitFakeOp(&w, &fake, "c000001", 2, 4096, .kv);

    try testing.expect(!fake.isVisible(1) and !fake.isVisible(2));
    try testing.expect(w.isPending("v/e1/c000000") and w.isPending("v/e1/c000001"));
    try testing.expect(w.pendingPrefix("v/e1/"));
    var staged: std.ArrayList([]const u8) = .empty;
    defer {
        for (staged.items) |p| testing.allocator.free(p);
        staged.deinit(testing.allocator);
    }
    try w.stagedPaths(&staged, testing.allocator);
    try testing.expectEqual(@as(usize, 3), staged.items.len);
    try testing.expectEqualStrings("v/e1/c000000", staged.items[0]);
    try testing.expectEqualStrings("v/e1/c000001", staged.items[2]);

    w.setPaused(false);
    w.drainPrefix("v/e1/");
    try testing.expectEqualSlices(u64, &.{ 1, 2 }, fake.published.items);
    try testing.expect(!w.isPending("v/e1/c000000") and !w.pendingPrefix("v/e1/"));
    w.drain();
    try testing.expectEqual(@as(u64, 3), w.filesWritten());
}

test "kv_disk_writer op: a fenced op never becomes visible and its ticket is released" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    var stats: write_stats.Stats = .{};
    w.stats = &stats;
    try w.start();
    defer w.deinit();

    w.setPaused(true);
    try submitFakeOp(&w, &fake, "c000000", 1, 4096, .kv);
    w.submitOp(try std.fmt.allocPrint(testing.allocator, "v/e2/c000000", .{}), fake.sink(), .{ .ticket = 2, .host_bytes = 512, .logical_bytes = 512 }, .kv);
    w.fence("v/e1/");
    w.setPaused(false);
    w.drain();

    try testing.expectEqualSlices(u64, &.{2}, fake.published.items);
    try testing.expectEqualSlices(u64, &.{1}, fake.aborted.items);
    try testing.expectEqual(@as(u64, 4096), stats.snapshot().failed);
    try testing.expectEqual(@as(u64, 1), w.files_dropped);

    // A whole-writer fence also discards what an older epoch staged.
    w.setPaused(true);
    try submitFakeOp(&w, &fake, "c000001", 3, 100, .kv);
    w.fence(null);
    w.setPaused(false);
    w.drain();
    try testing.expect(!fake.isVisible(3));
    try testing.expectEqualSlices(u64, &.{ 1, 3 }, fake.aborted.items);
}

test "kv_disk_writer op: injected failures report the way a file blob's do" {
    inline for (.{ FailAt.write, FailAt.submit }) |at| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
        var fake = FakeSink.init(testing.io);
        defer fake.deinit();

        const Run = struct { errors: u64, dropped: u64, written: u64, failed_bytes: u64, err_name: []const u8 };
        var runs: [2]Run = undefined;
        for (0..2) |as_op| {
            var stats: write_stats.Stats = .{};
            var w = Writer.init(testing.allocator, testing.io);
            w.stats = &stats;
            try w.start();
            defer w.deinit();
            w.injectFailure("bad", at);
            const names = [_][]const u8{ "c000000", "bad000", "c000002" };
            for (names, 1..) |n, t| {
                if (as_op == 1) {
                    try submitFakeOp(&w, &fake, n, t, 700, .kv);
                } else {
                    const b = try stagedBlob(root, n, 700);
                    w.submit(b.path, b.bytes, .kv);
                }
            }
            w.drain();
            const fails = w.takeFailures();
            defer {
                for (fails) |f| testing.allocator.free(f.path);
                testing.allocator.free(fails);
            }
            try testing.expectEqual(@as(usize, 1), fails.len);
            try testing.expectEqualStrings("bad000", std.fs.path.basename(fails[0].path));
            runs[as_op] = .{
                .errors = w.writeErrorCount(),
                .dropped = w.files_dropped,
                .written = w.filesWritten(),
                .failed_bytes = stats.snapshot().failed,
                .err_name = fails[0].err_name,
            };
        }
        try testing.expectEqual(runs[0].errors, runs[1].errors);
        try testing.expectEqual(runs[0].dropped, runs[1].dropped);
        try testing.expectEqual(runs[0].written, runs[1].written);
        try testing.expectEqual(runs[0].failed_bytes, runs[1].failed_bytes);
        try testing.expectEqualStrings(runs[0].err_name, runs[1].err_name);
        try testing.expect(!fake.isVisible(2));
        try testing.expect(fake.isVisible(1) and fake.isVisible(3));
    }
}

test "kv_disk_writer op: a sink error in either step drops the op, releases the ticket and counts a failure" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    fake.fail_append = 1;
    fake.fail_journal = 2;
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, testing.io);
    w.stats = &stats;
    try w.start();
    defer w.deinit();
    try submitFakeOp(&w, &fake, "c000000", 1, 300, .kv);
    try submitFakeOp(&w, &fake, "c000001", 2, 200, .kv);
    try submitFakeOp(&w, &fake, "c000002", 3, 100, .kv);
    w.drain();

    try testing.expectEqualSlices(u64, &.{3}, fake.published.items);
    try testing.expectEqualSlices(u64, &.{ 1, 2 }, fake.aborted.items);
    try testing.expectEqual(@as(u64, 2), w.writeErrorCount());
    try testing.expectEqual(@as(u64, 500), stats.snapshot().failed);
    const fails = w.takeFailures();
    defer {
        for (fails) |f| testing.allocator.free(f.path);
        testing.allocator.free(fails);
    }
    try testing.expectEqualStrings("InjectedAppend", fails[0].err_name);
    try testing.expectEqualStrings("InjectedJournal", fails[1].err_name);
}

test "kv_disk_writer op: the permit bounds the host bytes staged ops hold" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    fake.land_ns = 200_000;
    var w = Writer.init(testing.allocator, testing.io);
    w.permit_bytes = 64 * 1024;
    try w.start();
    defer w.deinit();
    for (0..12) |i| {
        try submitFakeOp(&w, &fake, "p", i + 1, 16 * 1024, .kv);
        try testing.expect(w.pendingBytes() <= w.permit_bytes + 16 * 1024);
    }
    w.drain();
    try testing.expectEqual(@as(usize, 12), fake.published.items.len);

    w.setPaused(true);
    for (0..4) |i| try submitFakeOp(&w, &fake, "q", 13 + i, 16 * 1024, .kv);
    const Waiter = struct {
        fn run(wr: *Writer, got: *std.atomic.Value(bool)) void {
            wr.waitForRoom(16 * 1024);
            got.store(true, .release);
        }
    };
    var got = std.atomic.Value(bool).init(false);
    const t = try std.Thread.spawn(.{}, Waiter.run, .{ &w, &got });
    std.Io.sleep(testing.io, .fromMilliseconds(20), .real) catch {};
    try testing.expect(!got.load(.acquire));
    w.setPaused(false);
    t.join();
    try testing.expect(got.load(.acquire));
}

test "kv_disk_writer op: a ticket's landed bytes equal the pack and journal growth it caused, counted as stored" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    fake.blocks[1] = 5000;
    fake.blocks[2] = 0;
    fake.blocks[3] = 777;
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, testing.io);
    w.stats = &stats;
    try w.start();
    defer w.deinit();
    try submitFakeOp(&w, &fake, "qsa.safetensors", 1, 9000, .qsa);
    try submitFakeOp(&w, &fake, "qsa.safetensors", 2, 9000, .qsa);
    try submitFakeOp(&w, &fake, "s0000128", 3, 2000, .ssm);
    w.drain();

    try testing.expectEqual(@as(?u64, 5100), w.takeLanding(1));
    try testing.expectEqual(@as(?u64, 100), w.takeLanding(2));
    try testing.expectEqual(@as(?u64, 877), w.takeLanding(3));
    try testing.expectEqual(@as(?u64, null), w.takeLanding(3));
    const s = stats.snapshot();
    try testing.expectEqual(@as(u64, 5200), s.stored[@backingInt(StatKind.qsa)]);
    try testing.expectEqual(@as(u64, 877), s.stored[@backingInt(StatKind.ssm)]);
    try testing.expectEqual(fake.pack_bytes + fake.journal_bytes, s.totalStored());
    try testing.expectEqual(w.bytes_written, s.totalStored());
    // `written` is each file's own size, as the plain files would have taken.
    try testing.expectEqual(@as(u64, 18000), s.written[@backingInt(StatKind.qsa)]);
    try testing.expectEqual(@as(u64, 2000), s.written[@backingInt(StatKind.ssm)]);
}

test "kv_disk_writer op: a file staged in parts appends each part, publishes at the last, and a dropped part drops the rest" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    fake.blocks[1] = 300;
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, testing.io);
    w.stats = &stats;
    try w.start();
    defer w.deinit();
    for ([_]Part{ .first, .middle, .last }) |part| {
        w.submitOpPart(try opPath("s0000128"), fake.sink(), .{ .ticket = 1, .host_bytes = 64, .logical_bytes = 64 }, part, .ssm);
    }
    w.drain();
    try testing.expectEqualSlices(u64, &.{1}, fake.published.items);
    try testing.expectEqual(@as(usize, 0), fake.aborted.items.len);
    try testing.expectEqual(@as(?u64, 3 * 300 + 100), w.takeLanding(1));
    try testing.expectEqual(@as(u64, 3 * 300 + 100), stats.snapshot().stored[@backingInt(StatKind.ssm)]);
    try testing.expectEqual(@as(u64, 3 * 64), stats.snapshot().written[@backingInt(StatKind.ssm)]);

    fake.fail_append = 2;
    for ([_]Part{ .first, .middle, .last }) |part| {
        w.submitOpPart(try opPath("s0000256"), fake.sink(), .{ .ticket = 2, .host_bytes = 64, .logical_bytes = 64 }, part, .ssm);
    }
    w.drain();
    try testing.expect(!fake.isVisible(2));
    try testing.expect(std.mem.indexOfScalar(u64, fake.aborted.items, 2) != null);
    try testing.expectEqual(@as(u64, 3 * 64), stats.snapshot().failed);
    try testing.expectEqual(@as(u64, 3 * 64), stats.snapshot().written[@backingInt(StatKind.ssm)]);
}

test "kv_disk_writer op: an op stays pending and unpublished for as long as the file write of its logical bytes takes" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    // 4 MiB at 16 MiB/s: a 250 ms floor, far above the instant fake landing.
    w.pace_bytes_per_s = 16 << 20;
    try w.start();
    defer w.deinit();

    const t0 = fake.clock.read();
    w.submitOp(try opPath("c000000"), fake.sink(), .{ .ticket = 1, .host_bytes = 4096, .logical_bytes = 4 << 20 }, .kv);
    std.Io.sleep(testing.io, .fromMilliseconds(60), .real) catch {};
    try testing.expect(w.isPending("v/e1/c000000"));
    try testing.expect(w.pendingPrefix("v/e1/"));
    try testing.expect(!fake.isVisible(1));
    w.drain();
    try testing.expect(fake.isVisible(1));
    try testing.expect(fake.published_ns.items[0] - t0 >= 250 * std.time.ns_per_ms);
    try testing.expect(!w.isPending("v/e1/c000000"));
}

test "kv_disk_writer op: a paced queue delays each op by the ops before it, as a busy writer does" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    w.pace_bytes_per_s = 16 << 20;
    try w.start();
    defer w.deinit();

    w.setPaused(true);
    const t0 = fake.clock.read();
    for (1..4) |t| w.submitOp(try opPath("c"), fake.sink(), .{ .ticket = t, .host_bytes = 64, .logical_bytes = 1 << 20 }, .kv);
    w.setPaused(false);
    w.drain();
    // 1 MiB at 16 MiB/s is 62.5 ms: the third op lands no sooner than three floors after the first began.
    try testing.expectEqual(@as(usize, 3), fake.published_ns.items.len);
    try testing.expect(fake.published_ns.items[0] - t0 >= 62 * std.time.ns_per_ms);
    try testing.expect(fake.published_ns.items[2] - t0 >= 187 * std.time.ns_per_ms);
}

test "kv_disk_writer op: no pace rate means no floor" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    try w.start();
    defer w.deinit();
    const t0 = fake.clock.read();
    w.submitOp(try opPath("c"), fake.sink(), .{ .ticket = 1, .host_bytes = 64, .logical_bytes = 1 << 40 }, .kv);
    w.drain();
    try testing.expect(fake.published_ns.items[0] - t0 < 100 * std.time.ns_per_ms);
}

test "kv_disk_writer op: calibratePace measures the file-write rate of a directory and leaves nothing behind" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var w = Writer.init(testing.allocator, testing.io);
    defer w.deinit();
    try w.calibratePace(root);
    try testing.expect(w.pace_bytes_per_s > 0);
    var it = tmp.dir.iterate();
    try testing.expect((try it.next(testing.io)) == null);
}

test "kv_disk_writer op: landNow lands on the caller, serialized with the writer thread" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    fake.land_ns = 100_000;
    var stats: write_stats.Stats = .{};
    var w = Writer.init(testing.allocator, testing.io);
    w.stats = &stats;

    // No thread needed: a synchronous op lands before the call returns.
    const n = try w.landNow(fake.sink(), 1, .meta);
    try testing.expectEqual(@as(u64, 1100), n);
    try testing.expect(fake.isVisible(1));
    try testing.expectEqual(@as(?u64, 1100), w.takeLanding(1));
    try testing.expectEqual(@as(u64, 1100), stats.snapshot().stored[@backingInt(StatKind.meta)]);
    try testing.expectEqual(@as(u64, 0), stats.snapshot().totalWritten());

    try w.start();
    defer w.deinit();
    for (0..8) |i| try submitFakeOp(&w, &fake, "c", 2 + i, 64, .kv);
    for (0..4) |i| _ = try w.landNow(fake.sink(), 10 + i, .meta);
    w.drain();
    try testing.expect(!fake.overlapped.load(.acquire));
    try testing.expectEqual(@as(usize, 13), fake.published.items.len);
}

test "kv_disk_writer op: deinit lands what a started writer holds and releases what a stopped one drops" {
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    try w.start();
    w.setPaused(true);
    try submitFakeOp(&w, &fake, "c", 1, 64, .kv);
    w.deinit();
    try testing.expectEqualSlices(u64, &.{1}, fake.published.items);
    try testing.expectEqual(@as(usize, 0), fake.aborted.items.len);
    // A writer that never started drops it.
    var idle = Writer.init(testing.allocator, testing.io);
    defer idle.deinit();
    try submitFakeOp(&idle, &fake, "c", 2, 64, .kv);
    try testing.expectEqualSlices(u64, &.{2}, fake.aborted.items);
}

test "kv_disk_writer: the crash seam counts the files and ops a tier lands, never the pace probe" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    var fake = FakeSink.init(testing.io);
    defer fake.deinit();
    var w = Writer.init(testing.allocator, testing.io);
    defer w.deinit();
    const before = landings.load(.monotonic);
    try w.calibratePace(root);
    try testing.expectEqual(before, landings.load(.monotonic));
    w.pace_bytes_per_s = 0;
    try w.start();
    w.submit(try std.fmt.allocPrint(testing.allocator, "{s}/c000000.safetensors", .{root}), try testing.allocator.dupe(u8, "file"), .kv);
    try submitFakeOp(&w, &fake, "c000001.safetensors", 1, 64, .kv);
    w.drain();
    try testing.expectEqual(before + 2, landings.load(.monotonic));
}

test "parseCrashAfter: a positive count arms the seam, anything else leaves it off" {
    try testing.expectEqual(@as(?u64, null), parseCrashAfter(null));
    try testing.expectEqual(@as(?u64, null), parseCrashAfter(""));
    try testing.expectEqual(@as(?u64, null), parseCrashAfter("0"));
    try testing.expectEqual(@as(?u64, null), parseCrashAfter("x3"));
    try testing.expectEqual(@as(?u64, 17), parseCrashAfter("17"));
}
