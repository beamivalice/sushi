//! What one message read off the wire may cost, for the chunked request reader and the WebSocket reader alike: its
//! decoded payload against `payload_cap` (a message of exactly the cap is read) and every raw byte, framing included,
//! against `raw_cap`, so framing that decodes to little or nothing (chunk extensions, empty fragments) still ends.
const std = @import("std");

pub const ByteBudget = struct {
    payload_cap: usize,
    raw_cap: usize,
    payload: usize = 0,
    raw: usize = 0,

    /// What framing may add to a payload of the cap: a chunk's size line and CRLFs, or a client frame's head, is 6
    /// bytes for a piece under 256 bytes, so pieces of 64 bytes and up stay under an eighth; 1 KiB covers a small
    /// message's last chunk, trailers or few frames.
    pub const framing_fixed: usize = 1024;

    pub fn init(payload_cap: usize) ByteBudget {
        return .{ .payload_cap = payload_cap, .raw_cap = payload_cap +| payload_cap / 8 +| framing_fixed };
    }

    pub const Error = error{
        /// The payload is over the cap.
        TooLarge,
        /// The payload fits, but the framing around it is over its allowance: pieces too small.
        FramingOverAllowance,
    };

    /// Charges `raw` wire bytes and `payload` decoded bytes.
    pub fn take(b: *ByteBudget, raw: usize, payload: usize) Error!void {
        b.raw +|= raw;
        b.payload +|= payload;
        if (b.payload > b.payload_cap) return error.TooLarge;
        if (b.raw > b.raw_cap) return error.FramingOverAllowance;
    }
};

test "ByteBudget: a payload of exactly the cap is taken, a byte more of either is not" {
    var b = ByteBudget.init(8);
    try b.take(8, 8);
    try b.take(ByteBudget.framing_fixed + 1, 0);
    try std.testing.expectError(error.FramingOverAllowance, b.take(1, 0));
    var c = ByteBudget.init(8);
    try std.testing.expectError(error.TooLarge, c.take(9, 9));
}
