//! Value rules the launch-flag parser applies before anything loads. MLX-free, so a host test pins them.
const std = @import("std");

/// True when the flag at `args[i]` is followed by its value: a flag never takes the next flag as its value.
pub fn follows(args: []const []const u8, i: usize) bool {
    return i + 1 < args.len and !std.mem.startsWith(u8, args[i + 1], "--");
}

pub fn isHelp(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h");
}

/// A TCP port a server can bind: 1..65535.
pub fn parsePort(text: []const u8) ?u16 {
    const n = std.fmt.parseInt(u16, text, 10) catch return null;
    return if (n == 0) null else n;
}

/// A sampling temperature: finite and not negative.
pub fn parseTemperature(text: []const u8) ?f32 {
    const v = std.fmt.parseFloat(f32, text) catch return null;
    return if (std.math.isFinite(v) and v >= 0) v else null;
}

/// A top-p mass: above 0, at most 1.
pub fn parseTopP(text: []const u8) ?f32 {
    const v = std.fmt.parseFloat(f32, text) catch return null;
    return if (std.math.isFinite(v) and v > 0 and v <= 1) v else null;
}

test "flag value: a flag does not take the next flag as its value" {
    const t = std.testing;
    try t.expect(follows(&.{ "--api-key", "k" }, 0));
    try t.expect(follows(&.{ "--temp", "-1" }, 0));
    try t.expect(!follows(&.{ "--api-key", "--port", "18901" }, 0));
    try t.expect(!follows(&.{"--api-key"}, 0));
}

test "flag value: ports are 1..65535" {
    const t = std.testing;
    try t.expectEqual(@as(?u16, 18901), parsePort("18901"));
    try t.expectEqual(@as(?u16, 65535), parsePort("65535"));
    try t.expectEqual(@as(?u16, null), parsePort("0"));
    try t.expectEqual(@as(?u16, null), parsePort("70000"));
    try t.expectEqual(@as(?u16, null), parsePort("-1"));
    try t.expectEqual(@as(?u16, null), parsePort("http"));
}

test "flag value: temperature is not negative and top-p is in (0, 1]" {
    const t = std.testing;
    try t.expectEqual(@as(?f32, 0), parseTemperature("0"));
    try t.expectEqual(@as(?f32, 1.5), parseTemperature("1.5"));
    try t.expectEqual(@as(?f32, null), parseTemperature("-1"));
    try t.expectEqual(@as(?f32, null), parseTemperature("nan"));
    try t.expectEqual(@as(?f32, 1), parseTopP("1"));
    try t.expectEqual(@as(?f32, 0.9), parseTopP("0.9"));
    try t.expectEqual(@as(?f32, null), parseTopP("0"));
    try t.expectEqual(@as(?f32, null), parseTopP("2"));
}
