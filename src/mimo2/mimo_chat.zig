//! MiMo-V2.6's chat face: the effort words its template takes (`arch/chat.zig` reads it).
const eff = @import("../effort.zig");

pub const model_types = [_][]const u8{"mimo_v2"};

// The template has only on/off; no artificial effort levels are advertised.
pub const effort: eff.Offer = .{ .fixed = &.{
    .{ .effort = .off },
    .{ .effort = .on },
} };
