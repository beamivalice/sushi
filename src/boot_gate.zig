//! The launch model's boot as the HTTP side sees it: the listener serves while the model loads, `/health` says
//! whether it is up, and a request that got the model before the boot finished waits for the boot's post-load steps.
const std = @import("std");

pub const Phase = enum(u8) { loading, ready, failed };

pub const Health = struct { status: []const u8, body: []const u8 };

/// A loading or failed boot is a 503, so a client that polls `/health` for readiness (the `run` REPL, scripts, load
/// balancers) still waits for the model, as it did when the port only opened after the load.
pub fn healthFor(phase: Phase) Health {
    return switch (phase) {
        .ready => .{ .status = "200 OK", .body = "{\"status\":\"ok\"}" },
        .loading => .{ .status = "503 Service Unavailable", .body = "{\"status\":\"loading\"}" },
        .failed => .{ .status = "503 Service Unavailable", .body = "{\"status\":\"error\"}" },
    };
}

pub const Gate = struct {
    phase: std.atomic.Value(Phase) = .init(.ready),
    opened: std.Io.Event = .is_set,

    /// Before the listener accepts: nothing waits on the gate yet.
    pub fn begin(self: *Gate) void {
        self.phase.store(.loading, .release);
        self.opened.reset();
    }

    pub fn open(self: *Gate, io: std.Io) void {
        self.phase.store(.ready, .release);
        self.opened.set(io);
    }

    /// A waiter released here finds the model's load error where it looks next, so it never runs on a failed boot.
    pub fn fail(self: *Gate, io: std.Io) void {
        self.phase.store(.failed, .release);
        self.opened.set(io);
    }

    pub fn waitOpen(self: *Gate, io: std.Io) void {
        self.opened.waitUncancelable(io);
    }

    pub fn health(self: *const Gate) Health {
        return healthFor(self.phase.load(.acquire));
    }
};

/// The process's one boot; a headless server never begins it, so it stays open and healthy.
pub var process: Gate = .{};

const testing = std.testing;

test "boot gate: a server that loads nothing at boot is healthy and never holds a request" {
    var g: Gate = .{};
    try testing.expectEqualStrings("200 OK", g.health().status);
    try testing.expectEqualStrings("{\"status\":\"ok\"}", g.health().body);
    g.waitOpen(testing.io);
}

test "boot gate: /health is a 503 naming the load until the boot opens, then ok" {
    var g: Gate = .{};
    g.begin();
    try testing.expectEqualStrings("503 Service Unavailable", g.health().status);
    try testing.expectEqualStrings("{\"status\":\"loading\"}", g.health().body);
    g.open(testing.io);
    try testing.expectEqualStrings("200 OK", g.health().status);
    try testing.expectEqualStrings("{\"status\":\"ok\"}", g.health().body);
}

test "boot gate: a failed boot reports the error and releases its waiters" {
    var g: Gate = .{};
    g.begin();
    g.fail(testing.io);
    try testing.expectEqualStrings("503 Service Unavailable", g.health().status);
    try testing.expectEqualStrings("{\"status\":\"error\"}", g.health().body);
    g.waitOpen(testing.io);
}

test "boot gate: a request waiting for the boot runs only once it opens" {
    const Waiter = struct {
        gate: *Gate,
        done: std.atomic.Value(bool) = .init(false),
        fn run(self: *@This()) void {
            self.gate.waitOpen(testing.io);
            self.done.store(true, .release);
        }
    };
    var g: Gate = .{};
    g.begin();
    var w: Waiter = .{ .gate = &g };
    const t = try std.Thread.spawn(.{}, Waiter.run, .{&w});
    var slept: u32 = 0;
    while (slept < 50) : (slept += 1) {
        try testing.expect(!w.done.load(.acquire));
        const ts = std.c.timespec{ .sec = 0, .nsec = 1_000_000 };
        _ = std.c.nanosleep(&ts, null);
    }
    g.open(testing.io);
    t.join();
    try testing.expect(w.done.load(.acquire));
}
