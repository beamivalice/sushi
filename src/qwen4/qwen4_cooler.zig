//! Where a Flash-Next load runs, decided once at the load: Sushi Cooler, the engine of our own, or MLX.
const std = @import("std");
const mlx = @import("../mlx.zig");
const ModelConfig = @import("../model.zig").ModelConfig;

/// An STQ pack: its routed experts are trellis banks of the STQ codebook, its trunk affine or STQ banks.
pub fn isStq(config: *const ModelConfig) bool {
    if (config.expert_layout != .exl3_k4 or config.expert_quant_codebook != .spec1) return false;
    const t = config.trunk_quant orelse return true;
    return t.codebook == .spec1;
}

/// The formats a Flash-Next load runs on Sushi Cooler, and no other: an STQ pack (trellis experts and trunk) and the
/// release's own BF16 checkpoint. Affine and MXFP4 packs keep MLX.
pub const Format = enum { stq, bf16 };

pub fn formatOf(config: *const ModelConfig) ?Format {
    if (isStq(config)) return .stq;
    const bf16_experts = config.expert_layout == .bf16_individual or config.expert_layout == .bf16_fused;
    if (bf16_experts and config.quant_bits == 0 and config.trunk_quant == null) return .bf16;
    return null;
}

/// The loads a Flash-Next pack cannot run on Sushi Cooler (`cooler_side.sideOf` asks): any but its formats.
pub fn coolerGap(config: *const ModelConfig) ?[]const u8 {
    if (formatOf(config) == null) return "Sushi Cooler runs Flash-Next's STQ packs and its BF16 checkpoint only";
    return null;
}

test "qwen4 cooler: an STQ pack and the BF16 checkpoint take the engine, no other format, and the opt-out outranks it" {
    const cooler_side = @import("../cooler_side.zig");
    var c: ModelConfig = .{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4, .expert_quant_codebook = .spec1, .cooler_for_test = true };
    try std.testing.expect(cooler_side.sideOf(&c) == .engine);
    c.trunk_quant = .{ .codebook = .spec1 };
    try std.testing.expect(cooler_side.sideOf(&c) == .engine);
    c.cooler_ask = .flag;
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_by_flag);
    c.cooler_ask = .setting;
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_by_setting);
    c.cooler_ask = .none;
    c.trunk_quant = .{ .codebook = .mcg };
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_gap);
    c.trunk_quant = null;
    c.expert_quant_codebook = .mcg;
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_gap);
    c.expert_quant_codebook = .spec1;
    c.expert_layout = .bf16_individual;
    try std.testing.expect(cooler_side.sideOf(&c) == .engine);
    c.quant_bits = 4;
    try std.testing.expectEqualStrings("Sushi Cooler runs Flash-Next's STQ packs and its BF16 checkpoint only", cooler_side.sideOf(&c).mlx_gap);
    c.quant_bits = 0;
    c.expert_layout = .mxfp4_split;
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_gap);
    c.expert_layout = .exl3_k4;
    c.cooler_for_test = null;
    try std.testing.expect(cooler_side.sideOf(&c) == .mlx_in_test);
}

const Trunk = @import("../trunk.zig").Trunk;
const layer_weights = @import("../layer_weights.zig");
const model = @import("../model.zig");

/// Every weight a Flash-Next load holds: the trunk's, the final mixer's and the MTP head's.
pub fn appendWeights(xfm: *const Trunk, vec: mlx.mlx_vector_array) !void {
    try xfm.appendWeights(vec);
    const q = &xfm.arch_state.qwen4;
    if (q.mixer) |*mx| layer_weights.appendStructArrays(vec, mx);
    if (q.mtp) |*h| {
        layer_weights.appendStructArrays(vec, h);
        layer_weights.appendStructArrays(vec, &h.mixer);
        layer_weights.appendLayerWeights(vec, &h.layer);
        for (h.owned) |arr| if (arr.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(vec, arr);
        };
    }
}

/// On a Cooler load, every weight the init built with an op moves from the engine's pool into an owned region.
pub fn ownBuiltWeights(xfm: *const Trunk) !void {
    const all = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(all);
    try appendWeights(xfm, all);
    for (0..mlx.mlx_vector_array_size(all)) |i| {
        var arr: mlx.mlx_array = .{};
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_vector_array_get(&arr, all, i));
        try model.ownInPlace(arr);
    }
}
