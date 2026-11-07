//! What an arch tells admission and the expert stream, from its config alone.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const spec = @import("spec.zig");

pub const ExpertStreamTraits = struct {
    /// `--drop-expert` is accepted.
    drop_expert: bool = false,
    /// `--ssd-prefetch` is accepted: the arch has an expert prefetcher.
    prefetch: bool = false,
    /// The expert slabs are read tile-major.
    tile_major: bool = false,
};

/// A config no enabled arch claims streams plain, with no drop.
pub fn expertStream(config: *const model.ModelConfig) ExpertStreamTraits {
    const s = spec.of(config) orelse return .{};
    return switch (s) {
        inline else => |c, t| registry.Mod(t).expertStream(c),
    };
}

test "expert-stream traits per model type and layout" {
    const T = ExpertStreamTraits;
    const rows = [_]struct { model_type: []const u8, layout: @TypeOf(@as(model.ModelConfig, .{}).expert_layout), want: T }{
        .{ .model_type = "deepseek_v41", .layout = .exl3_k4, .want = .{ .drop_expert = true, .prefetch = true, .tile_major = true } },
        .{ .model_type = "deepseek_v41", .layout = .mxfp4_w123, .want = .{ .drop_expert = true, .prefetch = true, .tile_major = false } },
        .{ .model_type = "glm5_next", .layout = .exl3_k4, .want = .{} },
        .{ .model_type = "qwen4_exp", .layout = .exl3_k4, .want = .{} },
        .{ .model_type = "mimo_v2", .layout = .exl3_k4, .want = .{} },
        .{ .model_type = "mistral", .layout = .exl3_k4, .want = .{} },
    };
    for (rows) |r| {
        const claimed = registry.ofModelType(r.model_type) != null or !model.isServedArch(r.model_type);
        if (!claimed) continue;
        var config = model.ModelConfig{ .model_type = r.model_type, .expert_layout = r.layout };
        try std.testing.expectEqual(r.want, expertStream(&config));
    }
}
