//! Small helpers around `std.Io.Timestamp` to ease the Zig 0.15 -> 0.16 migration.
//!
//! In 0.16 the legacy `std.time.Timer`, `std.time.timestamp()`, and
//! `std.time.milliTimestamp()` are gone — all clocks live under `std.Io` and
//! require an `Io` parameter. These helpers wrap those calls so the rest of the
//! codebase reads naturally.

const std = @import("std");

/// Seconds since the Unix epoch (wall-clock).
pub fn nowSecs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toSeconds();
}

/// Milliseconds since the Unix epoch (wall-clock).
pub fn nowMs(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .real).toMilliseconds();
}

/// Milliseconds on the MONOTONIC `boot` clock (arbitrary epoch — only
/// differences are meaningful). Use for deadlines/intervals, never for
/// timestamps a client sees: an NTP step or a manual clock change must not be
/// able to stall (or spam) a timer. `.boot` counts across pmset sleep, so a
/// deadline that expired while the lid was shut fires immediately on wake.
pub fn nowMsMonotonic(io: std.Io) i64 {
    return std.Io.Timestamp.now(io, .boot).toMilliseconds();
}

/// Drop-in replacement for `std.time.Timer` — uses the `boot` clock, which
/// counts wall-time across pmset sleep (vs `.awake`, which stops during
/// suspend). Important for long-running stopwatches that span a lid close.
/// Resolution is set by the `Io` implementation.
pub const Stopwatch = struct {
    io: std.Io,
    started_at: std.Io.Timestamp,

    pub fn init(io: std.Io) Stopwatch {
        return .{ .io = io, .started_at = std.Io.Timestamp.now(io, .boot) };
    }

    /// Nanoseconds since `init` (or last `reset`).
    pub fn read(s: Stopwatch) u64 {
        return @intCast(s.started_at.untilNow(s.io, .boot).nanoseconds);
    }

    /// Reset the start point to "now".
    pub fn reset(s: *Stopwatch) void {
        s.started_at = std.Io.Timestamp.now(s.io, .boot);
    }
};

/// Clock ties and rollback must not alias independent response envelopes.
pub fn nextCompletionId(counter: *std.atomic.Value(i64), now_ms: i64) i64 {
    var previous = counter.load(.monotonic);
    while (true) {
        const next = @max(now_ms, previous + 1);
        previous = counter.cmpxchgWeak(previous, next, .monotonic, .monotonic) orelse return next;
    }
}

/// Diagnostic switches are off for empty values and values beginning with `0`.
pub fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

pub fn diagEnvOn(name: [*:0]const u8) bool {
    return diagEnvValueOn(std.c.getenv(name));
}
/// Cache the effective switch once.
pub fn diagEnvOnCached(cache: *?bool, name: [*:0]const u8) bool {
    if (cache.*) |v| return v;
    const v = diagEnvOn(name);
    cache.* = v;
    return v;
}
