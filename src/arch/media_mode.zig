//! Which image processor produced a request's pixels: `gemma`'s fixed square, or the mode each enabled arch's `media`
//! declares as an enum literal (`pub const mode = .qwen;`). A separate file so `chat.zig`, which is MLX-free, can name it.
const std = @import("std");
const registry = @import("registry.zig");

pub const Mode = blk: {
    const tags = std.meta.tags(registry.Tag);
    var names: [tags.len + 1][]const u8 = undefined;
    var values: [tags.len + 1]u8 = undefined;
    names[0] = "gemma";
    values[0] = 0;
    for (tags, 1..) |t, i| {
        names[i] = @tagName(registry.Mod(t).media.mode);
        values[i] = i;
    }
    break :blk @Enum(u8, .exhaustive, &names, &values);
};

/// The mode a processor goes by when an enabled arch declares it; null when the build left that arch out.
pub fn modeNamed(comptime name: []const u8) ?Mode {
    return if (@hasField(Mode, name)) @field(Mode, name) else null;
}
