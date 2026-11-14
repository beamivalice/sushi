//! The one decode of a UTF-8 sequence from a slice: `std.unicode.utf8Decode` is `unreachable` when the slice is
//! shorter than its lead byte announces (a character cut by a segment end), so nothing else calls it (lint
//! `utf8-decode-raw`).
const std = @import("std");

/// The code point of `seq`, which must be exactly one whole sequence; null for anything else.
pub fn decode(seq: []const u8) ?u21 {
    if (seq.len == 0) return null;
    const want = std.unicode.utf8ByteSequenceLength(seq[0]) catch return null;
    if (seq.len != want) return null;
    return std.unicode.utf8Decode(seq) catch null;
}

const testing = std.testing;

test "decode: a whole sequence is its code point, a cut or malformed one is null" {
    try testing.expectEqual(@as(?u21, 'a'), decode("a"));
    try testing.expectEqual(@as(?u21, 0xE4), decode("\xc3\xa4"));
    try testing.expectEqual(@as(?u21, 0x4E2D), decode("\xe4\xb8\xad"));
    try testing.expectEqual(@as(?u21, 0x1F600), decode("\xf0\x9f\x98\x80"));
    for ([_][]const u8{ "", "\xc3", "\xe4\xb8", "\xf0\x9f\x98", "\xe4\xb8\xad\xad", "\x80", "\xc3\x28", "\xc0\x80", "\xed\xa0\x80" }) |bad| {
        try testing.expectEqual(@as(?u21, null), decode(bad));
    }
}
