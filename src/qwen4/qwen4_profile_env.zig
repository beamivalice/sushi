//! The `QWEN4_PROFILE_*` diagnostic switches: a run with any of them armed does not persist its round-cost table.
const std = @import("std");
const io_util = @import("../io_util.zig");

pub fn armedIn(environ: [*:null]const ?[*:0]const u8) bool {
    var i: usize = 0;
    while (environ[i]) |entry| : (i += 1) {
        const s = std.mem.span(entry);
        if (!std.mem.startsWith(u8, s, "QWEN4_PROFILE_")) continue;
        const eq = std.mem.indexOfScalar(u8, s, '=') orelse continue;
        if (io_util.diagEnvValueOn(@ptrCast(s[eq + 1 ..].ptr))) return true;
    }
    return false;
}

pub fn profileArmed() bool {
    return armedIn(std.c.environ);
}

test "a QWEN4_PROFILE_ variable armed with a diagnostic value arms the profile" {
    const unarmed = [_:null]?[*:0]const u8{ "HOME=/tmp", "QWEN4_PROFILE_X=0", "QWEN4_PROFILE_NOEQ" };
    try std.testing.expect(!armedIn(&unarmed));
    const armed = [_:null]?[*:0]const u8{ "HOME=/tmp", "QWEN4_PROFILE_X=1" };
    try std.testing.expect(armedIn(&armed));
    const other = [_:null]?[*:0]const u8{"SUSHI_PROFILE_X=1"};
    try std.testing.expect(!armedIn(&other));
}
