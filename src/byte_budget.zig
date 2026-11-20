//! What one message read off the wire may cost, for the chunked request reader and the WebSocket reader alike: its
//! decoded payload against `payload_cap` (a message of exactly the cap is read, in pieces of any size: a piece's own
//! minimal framing rides free with the bytes it carries), and framing that carries nothing (chunk extensions and
//! padding, the last chunk and trailers, empty fragments) against a fixed allowance, so it cannot run on.
const std = @import("std");

pub const ByteBudget = struct {
    payload_cap: usize,
    payload: usize = 0,
    idle: usize = 0,

    /// Framing that carries no payload a message may hold: a client sends none or a few bytes of it (a last chunk, a
    /// trailer, an empty final fragment); 8 KiB is the trailer room the chunked reader always allowed.
    pub const idle_framing_cap: usize = 8 * 1024;

    pub fn init(payload_cap: usize) ByteBudget {
        return .{ .payload_cap = payload_cap };
    }

    pub const Error = error{
        /// The payload is over the cap.
        TooLarge,
        /// Framing that carries nothing is over `idle_framing_cap`.
        FramingOverAllowance,
    };

    /// Charges `payload` decoded bytes and `idle` framing bytes that carried none.
    pub fn take(b: *ByteBudget, payload: usize, idle: usize) Error!void {
        b.payload +|= payload;
        b.idle +|= idle;
        if (b.payload > b.payload_cap) return error.TooLarge;
        if (b.idle > idle_framing_cap) return error.FramingOverAllowance;
    }
};

test "ByteBudget: a payload of exactly the cap is taken, a byte more is not, and idle framing has its own allowance" {
    var b = ByteBudget.init(8);
    try b.take(8, 0);
    try std.testing.expectError(error.TooLarge, b.take(1, 0));
    var c = ByteBudget.init(8);
    try c.take(0, ByteBudget.idle_framing_cap);
    try std.testing.expectError(error.FramingOverAllowance, c.take(0, 1));
}
