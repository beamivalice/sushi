//! Root of `zig build arch-slice`: it names the public surface of an arch folder built as a module of its own, so
//! Zig analyses code that no test calls. Zig reads a module only inside its root folder, so an import that leaves
//! the folder is a compile error here; the engine arrives only as `arch_host`.
const std = @import("std");

// Naming a declaration analyses it; a type is walked down to its members, `depth` levels.
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

test "the arch's public declarations compile against the host stub" {
    reach(@import("arch_slice"), 3);
}
