//! The smallest host root `sushi_exl3` compiles against: exactly the three names a host owes the module as `mlx_host`.
//! `zig build exl3-host-check` compiles the module against it, so a fourth name the module reaches is a compile error.
const std = @import("std");
const files = @import("host_files");

pub const mlx = files.mlx;
pub const log = files.log;
pub const io_util = files.io_util;

// Zig analyses a declaration only when something names it; naming the module's public surface down to the methods of
// a type in a file (module, file, type, member) makes the check cover code no test calls.
fn reach(comptime T: type, comptime depth: u8) void {
    inline for (comptime std.meta.declarations(T)) |name| {
        const decl = @field(T, name);
        if (@TypeOf(decl) != type) {
            _ = &decl;
        } else if (depth > 0) switch (@typeInfo(decl)) {
            .@"struct", .@"union", .@"enum", .@"opaque" => reach(decl, depth - 1),
            else => {},
        };
    }
}

test "the module's public declarations compile against this host" {
    reach(@import("sushi_exl3"), 2);
}
