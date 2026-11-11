//! What an arch tells admission and the expert stream, from its config alone.
const std = @import("std");
const model = @import("../model.zig");
const registry = @import("registry.zig");
const spec = @import("spec.zig");
const packs = @import("packs.zig");
const arch_config = @import("config.zig");
const expert_quant = @import("../expert_quant.zig");

pub const ExpertStreamTraits = struct {
    /// `--drop-expert` is accepted.
    drop_expert: bool = false,
    /// `--ssd-prefetch` is accepted: the arch has an expert prefetcher.
    prefetch: bool = false,
    /// `--ds41-mlx-engine` is accepted: decode may run on Sushi Cooler, the engine of our own, or on MLX.
    cooler: bool = false,
    /// The expert slabs are read tile-major.
    tile_major: bool = false,
    /// The routed experts are STQ rate groups, which the prefetcher reads.
    stq_experts: bool = false,
};

/// Bytes a streamed load keeps resident beside the trunk that only the arch knows of; 0 where it has none.
pub fn streamedResidentExtra(config: *const model.ModelConfig) u64 {
    const s = spec.of(config) orelse return 0;
    return switch (s) {
        inline else => |c, t| if (@hasDecl(registry.Mod(t), "streamedResidentExtra")) registry.Mod(t).streamedResidentExtra(c) else 0,
    };
}

/// Whether a streamed pack of `config`'s arch at `layout` keeps its MTP head resident and verifies on the streamed
/// decode path (an arch's `streamed_mtp_head.supported`).
pub fn streamedMtpHead(config: *const model.ModelConfig, layout: expert_quant.Layout) bool {
    const tag = arch_config.tagOf(config) orelse return false;
    return switch (tag) {
        inline else => |t| if (@hasDecl(registry.Mod(t), "streamed_mtp_head")) registry.Mod(t).streamed_mtp_head.supported(layout) else false,
    };
}

/// What the load refusal says a streamed MTP head needs: the `streamed_mtp_head.phrase` of every built arch that has one.
pub const mtp_unsupported = blk: {
    var text: []const u8 = "MTP under expert streaming needs --mtp on ";
    var first = true;
    for (packs.gate_order) |t| {
        if (@hasDecl(registry.Mod(t), "streamed_mtp_head")) {
            text = text ++ (if (first) "" else " or ") ++ registry.Mod(t).streamed_mtp_head.phrase;
            first = false;
        }
    }
    break :blk text;
};

/// A config no enabled arch claims streams plain, with no drop.
pub fn expertStream(config: *const model.ModelConfig) ExpertStreamTraits {
    const s = spec.of(config) orelse return .{};
    return switch (s) {
        inline else => |c, t| registry.Mod(t).expertStream(c),
    };
}

test "a streamed MTP head is an arch's own declaration, and the refusal names the archs that have one" {
    if (registry.ofModelType("qwen4_exp") != null) {
        const q = model.ModelConfig{ .model_type = "qwen4_exp" };
        try std.testing.expect(streamedMtpHead(&q, .exl3_k4));
        try std.testing.expect(!streamedMtpHead(&q, .bf16_fused));
    }
    if (registry.ofModelType("deepseek_v41") != null) {
        const d = model.ModelConfig{ .model_type = "deepseek_v41" };
        try std.testing.expect(streamedMtpHead(&d, .mxfp4_w123));
        try std.testing.expect(streamedMtpHead(&d, .exl3_k4));
        try std.testing.expect(!streamedMtpHead(&d, .mxfp4_individual));
    }
    const m = model.ModelConfig{ .model_type = "mimo_v2" };
    try std.testing.expect(!streamedMtpHead(&m, .exl3_k4));
    if (registry.ofModelType("qwen4_exp") != null and registry.ofModelType("deepseek_v41") != null)
        try std.testing.expectEqualStrings("MTP under expert streaming needs --mtp on a Sushi EXL3 Qwen3.8-Flash-Next pack or DeepSeek-V4.1-Flash", mtp_unsupported);
}

test "expert-stream traits per model type and layout" {
    const T = ExpertStreamTraits;
    const rows = [_]struct { model_type: []const u8, layout: @TypeOf(@as(model.ModelConfig, .{}).expert_layout), want: T }{
        .{ .model_type = "deepseek_v41", .layout = .exl3_k4, .want = .{ .drop_expert = true, .prefetch = true, .cooler = true, .tile_major = true } },
        .{ .model_type = "deepseek_v41", .layout = .mxfp4_w123, .want = .{ .drop_expert = true, .prefetch = true, .cooler = true, .tile_major = false } },
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
