//! The port `sushi --serve` binds when neither `--port` nor the settings file's `server.port` says otherwise;
//! `sushibar` shares it without linking the server.
const std = @import("std");

pub const port: u16 = 12345;

/// A `server.port` value from `model-settings.json`: a whole number from 1 to 65535, anything else is not one.
pub fn fromValue(v: std.json.Value) ?u16 {
    return switch (v) {
        .integer => |i| if (i >= 1 and i <= 65535) @intCast(i) else null,
        else => null,
    };
}
