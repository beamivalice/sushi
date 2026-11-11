//! Host memory a fill is writing. A pass of the engine that reads any of it is refused at its commit: the GPU would
//! read bytes the host has not finished.
const std = @import("std");

const max_marks = 256;
var mu: std.Io.Mutex = .init;
var begin: [max_marks]usize = @splat(0);
var end: [max_marks]usize = @splat(0);
/// Nonzero while any fill is in flight: the commit of a pass costs this one compare when none is.
pub var live: std.atomic.Value(u32) = .init(0);

fn lock() std.Io {
    const io = std.Io.Threaded.global_single_threaded.io();
    mu.lockUncancelable(io);
    return io;
}

/// A fill of `[base, base + len)` has begun. More marks than `max_marks` is a programming error, not a quiet drop.
pub fn add(base: usize, len: usize) void {
    const io = lock();
    defer mu.unlock(io);
    for (&begin, &end) |*b, *e| {
        if (b.* != 0) continue;
        b.* = base;
        e.* = base + len;
        _ = live.fetchAdd(1, .release);
        return;
    }
    @panic("fill_marks: too many fills in flight");
}

/// The fill that began at `base` is over.
pub fn remove(base: usize) void {
    const io = lock();
    defer mu.unlock(io);
    for (&begin) |*b| {
        if (b.* != base) continue;
        b.* = 0;
        _ = live.fetchSub(1, .release);
        return;
    }
}

/// The fills in flight, copied under the lock once so that a pass's binds are compared without it.
pub const Snapshot = struct {
    begin: [max_marks]usize = undefined,
    end: [max_marks]usize = undefined,
    n: usize = 0,

    pub fn meets(self: *const Snapshot, ptr: usize, len: usize) bool {
        for (self.begin[0..self.n], self.end[0..self.n]) |b, e| if (ptr < e and b < ptr + len) return true;
        return false;
    }
};

pub fn snapshot() Snapshot {
    const io = lock();
    defer mu.unlock(io);
    var out: Snapshot = .{};
    for (begin, end) |b, e| {
        if (b == 0) continue;
        out.begin[out.n] = b;
        out.end[out.n] = e;
        out.n += 1;
    }
    return out;
}

const testing = std.testing;

test "fill marks: a range meets a fill in flight only between its add and remove" {
    try testing.expect(!snapshot().meets(1000, 100));
    add(1000, 100);
    const held = snapshot();
    try testing.expect(held.meets(1050, 10));
    try testing.expect(held.meets(900, 101));
    try testing.expect(!held.meets(900, 100));
    try testing.expect(!held.meets(1100, 10));
    remove(1000);
    try testing.expect(!snapshot().meets(1050, 10));
    try testing.expectEqual(@as(u32, 0), live.load(.acquire));
}
