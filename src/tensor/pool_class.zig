//! The engine pool's size classes, pure (host tier): what a buffer of `len` bytes takes from the pool, which the
//! pool (`core.zig`) allocates by and the bills (`bills.zig`) count by.

const std = @import("std");

/// Pool buffers are whole pages: every page count up to `steps`, then `steps` sizes to an octave, so a buffer is
/// under 1/16 larger than what was asked.
pub const page_bytes = 16 * 1024;
pub const steps = 16;
pub const classes = steps * 22;

pub fn classOf(len: usize) usize {
    const pages = (len + page_bytes - 1) / page_bytes;
    if (pages <= steps) return pages - 1;
    var shift: usize = std.math.log2_int(usize, pages) - std.math.log2_int(usize, steps);
    var top = (pages + (@as(usize, 1) << @intCast(shift)) - 1) >> @intCast(shift);
    if (top == 2 * steps) {
        shift += 1;
        top = steps;
    }
    return steps - 1 + shift * steps + (top - steps);
}

/// What a tensor of `len` bytes takes from the pool: its size class.
pub fn poolBytes(len: usize) usize {
    return if (len == 0) 0 else classBytes(classOf(len));
}

pub fn classBytes(class: usize) usize {
    if (class < steps) return (class + 1) * page_bytes;
    const k = class - (steps - 1);
    return ((steps + k % steps) << @intCast(k / steps)) * page_bytes;
}
