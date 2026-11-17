//! A kernel arm a load keeps only if its trial on this GPU passes (a matrix-unit kernel against the SIMD one it would
//! replace). One rule for every arch: on the engine's stream a trial that could not run is the load's error, since a
//! launch there is never refused for the GPU's sake; on MLX's stream it declines the arm, as a GPU without the
//! kernel's features needs, and drops the error the trial latched. A verdict is kept for each side of the host, so a
//! trial on MLX's side decides nothing for a later Cooler load.
const std = @import("std");
const mlx = @import("mlx.zig");

/// `trial(args)` on stream `s` by the rule above: whether the arm is kept.
pub fn run(s: mlx.mlx_stream, comptime trial: anytype, args: anytype) !bool {
    if (mlx.engine.isStream(s)) return @call(.auto, trial, args);
    const had = mlx.errorPending();
    const ok = @call(.auto, trial, args) catch false;
    if (!ok) mlx.dropLatchedErrorUnless(had);
    return ok;
}

/// The side of stream `s`: 1 the engine's, 0 MLX's.
pub fn sideOf(s: mlx.mlx_stream) u1 {
    return @intFromBool(mlx.engine.isStream(s));
}

/// The side arrays are made on now, for a reader inside a forward with no stream to hand.
pub fn making() u1 {
    return @intFromBool(mlx.engine.active);
}

/// A trial's verdict, once a process for each side.
pub const Verdict = struct {
    sides: [2]std.atomic.Value(u8) = .{ .init(untried), .init(untried) },

    const untried = 0;
    const declined_v = 1;
    const kept_v = 2;

    pub fn tried(self: *const Verdict, side: u1) bool {
        return self.sides[side].load(.acquire) != untried;
    }
    pub fn kept(self: *const Verdict, side: u1) bool {
        return self.sides[side].load(.acquire) == kept_v;
    }
    pub fn declined(self: *const Verdict, side: u1) bool {
        return self.sides[side].load(.acquire) == declined_v;
    }
    pub fn set(self: *Verdict, side: u1, ok: bool) void {
        self.sides[side].store(if (ok) kept_v else declined_v, .release);
    }
    /// Both sides untried again (tests).
    pub fn reset(self: *Verdict) void {
        for (&self.sides) |*v| v.store(untried, .release);
    }
};

const testing = std.testing;

const Fake = struct {
    fn passes() !bool {
        return true;
    }
    fn differs() !bool {
        return false;
    }
    fn cannotRun() !bool {
        mlx.latchError("[test] a trial's launch failed");
        return error.MlxError;
    }
};

test "a trial on MLX's stream that cannot run declines the arm and drops what it latched; on the engine's it fails the load" {
    try mlx.requireGpu();
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    try testing.expect(try run(gpu, Fake.passes, .{}));
    try testing.expect(!try run(gpu, Fake.differs, .{}));
    try testing.expect(!try run(gpu, Fake.cannotRun, .{}));
    try testing.expect(!mlx.errorPending());

    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    try testing.expect(try run(es, Fake.passes, .{}));
    try testing.expect(!try run(es, Fake.differs, .{}));
    try testing.expectError(error.MlxError, run(es, Fake.cannotRun, .{}));
    var said: [128]u8 = undefined;
    try testing.expect(mlx.takeError(&said) != null);
}

test "a verdict on one side decides nothing for the other" {
    var v: Verdict = .{};
    v.set(0, false);
    try testing.expect(v.declined(0));
    try testing.expect(!v.tried(1));
    v.set(1, true);
    try testing.expect(v.kept(1) and !v.kept(0));
    v.reset();
    try testing.expect(!v.tried(0) and !v.tried(1));
}
