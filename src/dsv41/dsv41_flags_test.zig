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
