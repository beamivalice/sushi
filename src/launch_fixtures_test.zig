//! The reasoning words each served arch advertises, read from the engine's own tables, for the launch tests.
const effort_mod = @import("effort.zig");
const launch = @import("launch.zig");

fn listedCount(comptime model_type: []const u8) usize {
    var n: usize = 0;
    for (effort_mod.effortArms(model_type) orelse &.{}) |a| n += @intFromBool(effort_mod.isListed(a));
    return n;
}

/// A served arch's `/v1/models` `reasoning_efforts`, read from the engine's own table.
fn advertised(comptime model_type: []const u8) [listedCount(model_type)][]const u8 {
    var words: [listedCount(model_type)][]const u8 = undefined;
    var i: usize = 0;
    for (effort_mod.effortArms(model_type) orelse &.{}) |a| if (effort_mod.isListed(a)) {
        words[i] = @tagName(a.effort);
        i += 1;
    };
    return words;
}
pub const qwen4_efforts = advertised("qwen4_exp");
pub const mimo_efforts = advertised("mimo_v2");
pub const glm5_efforts = advertised("glm5_next");

pub fn glmEntry() launch.Entry {
    return .{ .id = "GLM-5.3-Flash-Sushi-2.5bpw", .budget = launch.budgetForContext(1048576), .vision = true, .loaded = true, .efforts = &glm5_efforts, .default_effort = "high" };
}
