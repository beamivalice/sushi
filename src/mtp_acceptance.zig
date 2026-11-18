const std = @import("std");
const testing = std.testing;

pub fn specAcceptProb(p_draft: f32, q_draft: f32) f32 {
    return @min(1.0, p_draft / @max(q_draft, 1e-12));
}

/// The longest prefix of drafts a sampled round keeps: draft k stands with probability min(1, p/q), or p alone for a
/// draft with no proposal density (a one-hot: greedy or copied).
pub fn exactPrefix(p: []const f32, q: ?[]const f32, prng: *std.Random.DefaultPrng) u32 {
    var accepted: u32 = 0;
    for (p, 0..) |mass, k| {
        const accept_prob = if (q) |proposal| specAcceptProb(mass, proposal[k]) else @min(1.0, mass);
        if (prng.random().float(f32) >= accept_prob) break;
        accepted += 1;
    }
    return accepted;
}

test "exactPrefix keeps a certain draft and stops at an impossible one" {
    var prng = std.Random.DefaultPrng.init(7);
    try testing.expectEqual(@as(u32, 2), exactPrefix(&.{ 1.0, 1.0 }, null, &prng));
    try testing.expectEqual(@as(u32, 1), exactPrefix(&.{ 1.0, 0.0, 1.0 }, null, &prng));
    try testing.expectEqual(@as(u32, 2), exactPrefix(&.{ 0.3, 0.2, 0.0 }, &.{ 0.3, 0.1, 0.5 }, &prng));
}
