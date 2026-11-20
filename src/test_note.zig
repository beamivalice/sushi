//! What of the running test did not run, said on its log line (`scripts/test_runner.zig` reads `sushi_test_note` after
//! each test), so a run that passes says what it did not cover. MLX-free: the host tier links it too.
const std = @import("std");
const registry = @import("arch/registry.zig");

var buf: [512]u8 = undefined;
var len: usize = 0;

/// A part of the running test that did not run (an arch whose fixture is missing, or that this build leaves out).
pub fn notePartSkipped(what: []const u8) void {
    var w = std.Io.Writer.fixed(buf[len .. buf.len - 1]);
    w.print("{s}{s}", .{ if (len > 0) "; " else "", what }) catch {};
    len += w.end;
}

/// The running test's notes so far, for a test that checks its own.
pub fn notes() []const u8 {
    return buf[0..len];
}

/// A test that reads these archs' configs or fixtures skips, each one left out named, when `-Darchs` leaves any of them
/// out of this build: it never runs on numbers its arch's code would have made.
pub fn needArchs(comptime names: []const []const u8) error{SkipZigTest}!void {
    var missing = false;
    inline for (names) |n| {
        comptime {
            for (registry.registered_names) |r| {
                if (std.mem.eql(u8, r, n)) break;
            } else @compileError("needArchs names \"" ++ n ++ "\", which src/arch/registry.zig does not register");
        }
        if (!@hasField(registry.Tag, n)) {
            notePartSkipped(n ++ ": not in this build (-Darchs)");
            missing = true;
        }
    }
    if (missing) return error.SkipZigTest;
}

/// The running test's notes, cleared as they are read.
fn testNote() callconv(.c) ?[*:0]const u8 {
    if (len == 0) return null;
    buf[len] = 0;
    len = 0;
    return @ptrCast(&buf);
}

comptime {
    @export(&testNote, .{ .name = "sushi_test_note" });
}
