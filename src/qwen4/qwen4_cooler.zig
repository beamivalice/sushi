//! Where a Flash-Next load runs, decided once at the load: Sushi Cooler, the engine of our own, or MLX.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("../mlx.zig");
const io_util = @import("../io_util.zig");
const ModelConfig = @import("../model.zig").ModelConfig;

pub const Side = enum {
    engine,
    /// `--qwen-mlx-engine`.
    mlx_by_flag,
    /// The model setting turned Sushi Cooler off.
    mlx_by_setting,
    /// Only an STQ pack runs on Sushi Cooler.
    mlx_not_stq,
    /// Until Qwen on Sushi Cooler passes its gate, a served load takes it only under `SUSHI_QWEN4_COOLER`.
    mlx_until_gated,
};

/// An STQ pack: its routed experts are trellis banks of the STQ codebook, its trunk affine or STQ banks.
pub fn isStq(config: *const ModelConfig) bool {
    if (config.expert_layout != .exl3_k4 or config.expert_quant_codebook != .spec1) return false;
    const t = config.trunk_quant orelse return true;
    return t.codebook == .spec1;
}

pub fn sideOf(config: *const ModelConfig) Side {
    switch (config.cooler_ask) {
        .flag => return .mlx_by_flag,
        .setting => return .mlx_by_setting,
        .none => {},
    }
    if (!isStq(config)) return .mlx_not_stq;
    // The tests written against MLX keep it; the engine's own tests name the engine.
    if (comptime builtin.is_test) return if (config.cooler_for_test orelse false) .engine else .mlx_until_gated;
    return if (io_util.diagEnvOnCached(&dev_knob, "SUSHI_QWEN4_COOLER")) .engine else .mlx_until_gated;
}

var dev_knob: ?bool = null;

/// Whether this load's forwards run on the engine of our own. The load and every bill ask here.
pub fn ownDecode(config: *const ModelConfig) bool {
    return sideOf(config) == .engine;
}

/// The stream a Cooler load runs on; null for the thread's GPU stream.
pub fn ownStream(config: *const ModelConfig) ?mlx.mlx_stream {
    return if (ownDecode(config)) mlx.engine.stream() else null;
}

/// The load line for a side. The engine of our own ships as Sushi Cooler.
pub fn line(side: Side) []const u8 {
    return switch (side) {
        .engine => "[qwen4] forwards: on Sushi Cooler, the engine of our own",
        .mlx_by_flag => "[qwen4] forwards: on MLX because --qwen-mlx-engine was given (Sushi Cooler is off)",
        .mlx_by_setting => "[qwen4] forwards: on MLX because the model setting turned Sushi Cooler off",
        .mlx_not_stq => "[qwen4] forwards: on MLX: Sushi Cooler runs STQ packs only",
        .mlx_until_gated => "[qwen4] forwards: on MLX: Sushi Cooler for Qwen is in development (SUSHI_QWEN4_COOLER=1 turns it on)",
    };
}

test "qwen4 cooler: only an STQ pack takes the engine, and the opt-out outranks it" {
    var c: ModelConfig = .{ .model_type = "qwen4_exp", .expert_layout = .exl3_k4, .expert_quant_codebook = .spec1, .cooler_for_test = true };
    try std.testing.expectEqual(Side.engine, sideOf(&c));
    c.trunk_quant = .{ .codebook = .spec1 };
    try std.testing.expectEqual(Side.engine, sideOf(&c));
    c.cooler_ask = .flag;
    try std.testing.expectEqual(Side.mlx_by_flag, sideOf(&c));
    c.cooler_ask = .setting;
    try std.testing.expectEqual(Side.mlx_by_setting, sideOf(&c));
    c.cooler_ask = .none;
    c.trunk_quant = .{ .codebook = .mcg };
    try std.testing.expectEqual(Side.mlx_not_stq, sideOf(&c));
    c.trunk_quant = null;
    c.expert_quant_codebook = .mcg;
    try std.testing.expectEqual(Side.mlx_not_stq, sideOf(&c));
    c.expert_quant_codebook = .spec1;
    c.expert_layout = .bf16_fused;
    try std.testing.expectEqual(Side.mlx_not_stq, sideOf(&c));
    c.expert_layout = .exl3_k4;
    c.cooler_for_test = null;
    try std.testing.expectEqual(Side.mlx_until_gated, sideOf(&c));
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
