const std = @import("std");
const testing = std.testing;

/// The engine's thinking-effort vocabulary. Each served arch accepts a subset
/// (`effortArms`); a word outside it is refused. `selectEffortPick` folds Qwen
/// `high`/`max` to `xhigh`, GLM `medium` to `high` and `xhigh` to `max`, and
/// MiMo's thinking words to `on` (that fold is silent).
pub const Effort = enum { off, on, minimal, low, medium, high, xhigh, max };

/// Explicit --think default for serving, CLI and later model loads.
pub var think_effort_flag: ?Effort = null;

/// The arm `--think` selects on this arch; null leaves the model on its own default.
pub fn thinkFlagArm(model_type: []const u8) ?EffortArm {
    const e = think_effort_flag orelse return null;
    return armForWord(effortArms(model_type) orelse return null, @tagName(e));
}

/// One accepted effort word on one arch. `budget` is the decode-time thinking
/// cap in tokens; null = `--reasoning-budget` (unlimited by default). The word
/// itself reaches a template that reads it (qwen4_exp: low|medium|xhigh).
pub const EffortArm = struct { effort: Effort, budget: ?i32 = null };

pub const qwen4_exp_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .low, .budget = 2048 },
    .{ .effort = .medium, .budget = 8192 },
    .{ .effort = .xhigh },
};

// MiMo's template has only on/off; do not advertise artificial effort levels.
const mimo_v2_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .on },
};

// DeepSeek's encoder takes low|high|max and renders a numeric budget; medium folds to high.
const deepseek_v41_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .low },
    .{ .effort = .high },
    .{ .effort = .max },
};

const glm5_efforts = [_]EffortArm{
    .{ .effort = .low },
    .{ .effort = .high },
    .{ .effort = .max },
};

/// null = an inherited arch: its effort words keep `responses.effortBudget`.
pub fn effortArms(model_type: []const u8) ?[]const EffortArm {
    if (std.mem.eql(u8, model_type, "glm5_next")) return &glm5_efforts;
    if (std.mem.eql(u8, model_type, "qwen4_exp")) return &qwen4_exp_efforts;
    if (std.mem.eql(u8, model_type, "mimo_v2")) return &mimo_v2_efforts;
    if (std.mem.eql(u8, model_type, "deepseek_v41")) return &deepseek_v41_efforts;
    return null;
}

/// `none` is an alias of off. No served table lists `minimal`; inherited arches
/// keep its legacy budget (`responses.effortBudget`).
pub fn parseEffort(word: []const u8) ?Effort {
    if (std.mem.eql(u8, word, "none")) return .off;
    return std.meta.stringToEnum(Effort, word);
}

pub fn findEffortArm(arms: []const EffortArm, effort: Effort) ?EffortArm {
    for (arms) |a| if (a.effort == effort) return a;
    return null;
}

/// The effort `e` selects on a model offering `offered`; null = refused.
pub const EffortPick = struct {
    effort: Effort,
    /// Set when the asked word was re-routed. MiMo's on-fold leaves this null.
    rerouted_from: ?Effort = null,
};

pub fn selectEffortPick(offered: []const Effort, e: Effort) ?EffortPick {
    if (std.mem.indexOfScalar(Effort, offered, e) != null) return .{ .effort = e };
    if ((e == .high or e == .max) and std.mem.indexOfScalar(Effort, offered, .xhigh) != null)
        return .{ .effort = .xhigh, .rerouted_from = e };
    if (e == .medium and std.mem.indexOfScalar(Effort, offered, .high) != null)
        return .{ .effort = .high, .rerouted_from = e };
    if (e == .xhigh and std.mem.indexOfScalar(Effort, offered, .max) != null)
        return .{ .effort = .max, .rerouted_from = e };
    if (e != .off and std.mem.indexOfScalar(Effort, offered, .on) != null) return .{ .effort = .on };
    return null;
}

pub fn selectEffort(offered: []const Effort, e: Effort) ?Effort {
    const pick = selectEffortPick(offered, e) orelse return null;
    return pick.effort;
}

/// `{model} has no `{asked}` effort; served at `{served}`.
pub const effort_reroute_notice = "{s} has no `{s}` effort; served at `{s}`";

pub fn formatEffortRerouteNotice(allocator: std.mem.Allocator, model_name: []const u8, from: Effort, to: Effort) ![]u8 {
    return std.fmt.allocPrint(allocator, effort_reroute_notice, .{ model_name, @tagName(from), @tagName(to) });
}

/// The arm a client's effort word selects (`selectEffort`); null = refused.
pub fn armForWord(arms: []const EffortArm, word: []const u8) ?EffortArm {
    var offered: [std.enums.values(Effort).len]Effort = undefined;
    for (arms, 0..) |a, i| offered[i] = a.effort;
    const e = selectEffort(offered[0..arms.len], parseEffort(word) orelse return null) orelse return null;
    return findEffortArm(arms, e);
}

test "selectEffort: Qwen high and max re-route to xhigh; other unknowns stay refused" {
    const qwen = [_]Effort{ .off, .low, .medium, .xhigh };
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .high));
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .max));
    try testing.expectEqual(@as(?Effort, .xhigh), selectEffort(&qwen, .xhigh));
    try testing.expectEqual(@as(?Effort, null), selectEffort(&qwen, .minimal));
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .high));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .max));
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .medium));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .xhigh));
    const mimo = [_]Effort{ .off, .on };
    try testing.expectEqual(@as(?Effort, .on), selectEffort(&mimo, .high));
    try testing.expectEqual(@as(?Effort, .on), selectEffort(&mimo, .max));
    try testing.expectEqual(@as(?Effort, .off), selectEffort(&mimo, .off));
}

test "selectEffort: GLM medium re-routes to high and xhigh to max; other missing words stay refused" {
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expectEqual(@as(?Effort, .high), selectEffort(&glm, .medium));
    try testing.expectEqual(@as(?Effort, .max), selectEffort(&glm, .xhigh));
    try testing.expectEqual(Effort.medium, selectEffortPick(&glm, .medium).?.rerouted_from.?);
    try testing.expectEqual(Effort.xhigh, selectEffortPick(&glm, .xhigh).?.rerouted_from.?);
    try testing.expect(selectEffortPick(&glm, .high).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&glm, .max).?.rerouted_from == null);
    try testing.expect(selectEffort(&glm, .off) == null);
    try testing.expect(selectEffort(&glm, .minimal) == null);
    try testing.expect(selectEffort(&glm, .on) == null);
    const mimo = [_]Effort{ .off, .on };
    try testing.expect(selectEffortPick(&mimo, .medium).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&mimo, .xhigh).?.rerouted_from == null);
}

test "selectEffortPick: only Qwen high and max carry a re-route; MiMo's on-fold stays quiet" {
    const qwen = [_]Effort{ .off, .low, .medium, .xhigh };
    const high = selectEffortPick(&qwen, .high).?;
    try testing.expectEqual(Effort.xhigh, high.effort);
    try testing.expectEqual(Effort.high, high.rerouted_from.?);
    try testing.expect(selectEffortPick(&qwen, .xhigh).?.rerouted_from == null);
    try testing.expect(selectEffortPick(&qwen, .minimal) == null);
    const mimo = [_]Effort{ .off, .on };
    try testing.expect(selectEffortPick(&mimo, .max).?.rerouted_from == null);
    try testing.expectEqual(Effort.on, selectEffortPick(&mimo, .max).?.effort);
    const glm = [_]Effort{ .low, .high, .max };
    try testing.expect(selectEffortPick(&glm, .high).?.rerouted_from == null);
    try testing.expectEqual(Effort.max, selectEffortPick(&glm, .xhigh).?.effort);
    try testing.expectEqual(Effort.xhigh, selectEffortPick(&glm, .xhigh).?.rerouted_from.?);
    const notice = try formatEffortRerouteNotice(testing.allocator, "Qwen3.8-Flash-Next", .high, .xhigh);
    defer testing.allocator.free(notice);
    try testing.expectEqualStrings("Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`", notice);
}
