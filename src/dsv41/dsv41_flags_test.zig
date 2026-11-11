//! DeepSeek-V4.1's launch-flag refusals that read its own config. MLX-free, so the host tier runs it.
const std = @import("std");
const flags = @import("../arch/flags.zig");
const model = @import("../model.zig");

test "an MTP depth flag above the DSpark block is refused, naming the flag and the block" {
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    dsv41.dsv41 = .{ .dspark = .{ .block = 5 } };
    const qwen4 = model.ModelConfig{ .model_type = "qwen4_exp" };
    try std.testing.expectEqualStrings("--mtp-max-depth 6 is above deepseek_v41's DSpark block of 5 drafts; use 5 or lower", flags.archFlagRefusal(&dsv41, .{ .mtp_max_depth = 6 }).?);
    try std.testing.expectEqualStrings("--mtp-min-depth 6 is above deepseek_v41's DSpark block of 5 drafts; use 5 or lower", flags.archFlagRefusal(&dsv41, .{ .mtp_min_depth = 6 }).?);
    try std.testing.expect(flags.archFlagRefusal(&dsv41, .{ .mtp_max_depth = 5, .mtp_min_depth = 5 }) == null);
    try std.testing.expect(flags.archFlagRefusal(&dsv41, .{}) == null);
    try std.testing.expect(flags.archFlagRefusal(&qwen4, .{ .mtp_max_depth = 8 }) == null);
}

test "the settings page disables ds41_cooler, with the reason, for a DeepSeek pack the engine cannot serve" {
    var dsv41 = model.ModelConfig{ .model_type = "deepseek_v41" };
    dsv41.dsv41 = .{};
    const fit = flags.settingsFit(&dsv41);
    const lim = @import("../model_settings.zig").limit("ds41_cooler", fit);
    try std.testing.expect(lim.disabled);
    try std.testing.expect(std.mem.indexOf(u8, lim.why, "not STQ") != null);
}

test "the Sushi Cooler choice reaches the DeepSeek config: the flag, else the setting, else nothing asked" {
    var cfg = model.ModelConfig{ .model_type = "deepseek_v41" };
    cfg.dsv41 = .{};
    cfg.stampCooler();
    try std.testing.expectEqual(.none, cfg.dsv41.?.mlx_ask);
    cfg.cooler_override = false;
    cfg.stampCooler();
    try std.testing.expectEqual(.setting, cfg.dsv41.?.mlx_ask);
    cfg.cooler_flag = false;
    cfg.stampCooler();
    try std.testing.expectEqual(.flag, cfg.dsv41.?.mlx_ask);
    cfg.cooler_flag = null;
    cfg.cooler_override = true;
    cfg.stampCooler();
    try std.testing.expectEqual(.none, cfg.dsv41.?.mlx_ask);
}
