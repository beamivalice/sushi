//! What names a frame of a plan: the code site that opens it and the key that decides its structure
//! (docs/bill_redesign.md#the-plan). Pure: the host tier builds it.
const std = @import("std");

/// A `plan.frame` or `plan.each` call site, never 0.
pub const Site = u32;
/// A hash of a frame's key tuple; 0 for the empty key.
pub const Key = u64;

pub const FrameId = struct {
    site: Site,
    key: Key = 0,

    pub fn eql(a: FrameId, b: FrameId) bool {
        return a.site == b.site and a.key == b.key;
    }
};

/// How a child frame sits in its parent: `exact` is part of the parent's structure at its place; `each` is one
/// iteration of a loop whose count varies, so consecutive ones of a site are one group.
pub const Kind = enum(u1) { exact, each };

/// The site of a call at `src`. Two calls on one line and column of one file are one site.
pub fn siteOf(comptime src: std.builtin.SourceLocation) Site {
    return comptime blk: {
        var h = std.hash.Wyhash.init(0);
        h.update(src.file);
        h.update(std.mem.asBytes(&src.line));
        h.update(std.mem.asBytes(&src.column));
        const v: u32 = @truncate(h.final());
        break :blk if (v == 0) 1 else v;
    };
}

/// The key of a tuple of ints, enums and bools, in field order. `{}` and `.{}` give 0.
pub fn keyOf(key: anytype) Key {
    const T = @TypeOf(key);
    if (T == void) return 0;
    var h = std.hash.Wyhash.init(0);
    var n: usize = 0;
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| {
            feed(&h, @field(key, name));
            n += 1;
        },
        else => {
            feed(&h, key);
            n = 1;
        },
    }
    if (n == 0) return 0;
    const v = h.final();
    return if (v == 0) 1 else v;
}

fn feed(h: *std.hash.Wyhash, value: anytype) void {
    const word: u64 = switch (@typeInfo(@TypeOf(value))) {
        .int, .comptime_int => @bitCast(@as(i64, @intCast(value))),
        .@"enum" => @intCast(@intFromEnum(value)),
        .bool => @intFromBool(value),
        else => @compileError("a frame key holds ints, enums and bools, not " ++ @typeName(@TypeOf(value))),
    };
    h.update(std.mem.asBytes(&word));
}

test "a frame key follows its values, whatever their widths, and the empty key is 0" {
    const t = std.testing;
    const Arm = enum { fused, split };
    try t.expectEqual(@as(Key, 0), keyOf({}));
    try t.expectEqual(@as(Key, 0), keyOf(.{}));
    try t.expectEqual(keyOf(.{ @as(u8, 3), Arm.split, true }), keyOf(.{ @as(c_int, 3), Arm.split, true }));
    try t.expect(keyOf(.{ 3, Arm.split }) != keyOf(.{ 3, Arm.fused }));
    try t.expect(keyOf(.{ 3, 4 }) != keyOf(.{ 4, 3 }));
    try t.expect(keyOf(.{0}) != 0);
    try t.expectEqual(keyOf(@as(u32, 7)), keyOf(.{7}));
}

test "two call sites are two frame sites" {
    const a = siteOf(@src());
    const b = siteOf(@src());
    try std.testing.expect(a != 0 and b != 0 and a != b);
}
