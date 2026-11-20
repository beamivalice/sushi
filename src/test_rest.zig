//! After each test (`scripts/test_runner.zig` calls `sushi_after_each`): the process state a test leaves for the next
//! one, named and put back at rest, so the test that leaves it fails by name and the next one starts clean.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const handle = @import("arch/handle.zig");

var report: [512]u8 = undefined;

/// Loads a run step keeps across its tests (the Cooler contract's held loads): the engine holds they keep, what a test
/// left of them (`rest`, written to `w`; true when it left anything) and their unload as the process ends (`close`).
pub const Held = struct {
    holds: u32 = 0,
    rest: ?*const fn (w: *std.Io.Writer) bool = null,
    close: ?*const fn (w: *std.Io.Writer) bool = null,
};
pub var held: Held = .{};

var note_buf: [512]u8 = undefined;
var note_len: usize = 0;

/// A part of the running test that did not run (an arch whose fixture is missing): the runner writes it on the test's
/// log line, so a run that passes says what it did not cover.
pub fn notePartSkipped(what: []const u8) void {
    var w = std.Io.Writer.fixed(note_buf[note_len .. note_buf.len - 1]);
    w.print("{s}{s}", .{ if (note_len > 0) "; " else "", what }) catch {};
    note_len += w.end;
}

/// The running test's notes so far, for a test that checks its own.
pub fn notes() []const u8 {
    return note_buf[0..note_len];
}

/// The running test's notes, cleared as they are read (`scripts/test_runner.zig`).
fn testNote() callconv(.c) ?[*:0]const u8 {
    if (note_len == 0) return null;
    note_buf[note_len] = 0;
    note_len = 0;
    return @ptrCast(&note_buf);
}

fn afterEach() callconv(.c) ?[*:0]const u8 {
    var w = std.Io.Writer.fixed(report[0 .. report.len - 1]);
    var left = mlx.engine.leftByTest(&w, held.holds) catch true;
    while (mlx.engine.holdCount() > held.holds) handle.engineHoldEnd();
    if (held.rest) |rest| left = rest(&w) or left;
    if (mlx.fault.armed()) {
        w.writeAll(" the MLX fault injector armed;") catch {};
        mlx.fault.disarm();
        left = true;
    }
    if (mlx.errorPending()) {
        var msg: [200]u8 = undefined;
        w.print(" an MLX error latched ({s});", .{mlx.takeError(&msg) orelse ""}) catch {};
        left = true;
    }
    if (model.cooler_off.count() != 0) {
        w.writeAll(" a Sushi Cooler opt-out flag still given;") catch {};
        model.cooler_off = .empty;
        left = true;
    }
    if (!left) return null;
    const n = w.end;
    report[n] = 0;
    return @ptrCast(report[0..n :0]);
}

/// As the process ends (`scripts/test_runner.zig`): the held loads unloaded, and what that left.
fn afterAll() callconv(.c) ?[*:0]const u8 {
    const close = held.close orelse return null;
    var w = std.Io.Writer.fixed(report[0 .. report.len - 1]);
    if (!close(&w)) return null;
    const n = w.end;
    report[n] = 0;
    return @ptrCast(report[0..n :0]);
}

comptime {
    @export(&afterEach, .{ .name = "sushi_after_each" });
    @export(&afterAll, .{ .name = "sushi_after_all" });
    @export(&testNote, .{ .name = "sushi_test_note" });
}
