//! GLM's pack preflight: the MTP layers after the trunk's, and the refused rate groups.
const std = @import("std");
const model = @import("../model.zig");
const pack_index = @import("../pack_index.zig");
const SourceIndex = pack_index.SourceIndex;
const TensorMeta = pack_index.TensorMeta;
const ShardStamp = pack_index.ShardStamp;
const validateExl3Source = pack_index.validateExl3Source;

test "GLM EXL3 preflight refuses orphan MTP and out of range packed tensors" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var source = SourceIndex{ .weight_map = std.StringHashMap([]const u8).init(a), .files = std.StringHashMap(void).init(a), .tensors = std.StringHashMap(TensorMeta).init(a), .stamps = std.StringHashMap(ShardStamp).init(a) };
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .exl3_k4, .expert_quant_rate = .{ .n = 36 }, .num_hidden_layers = 2, .first_k_dense_replace = 1, .num_experts = 2, .num_experts_per_tok = 1, .hidden_size = 128, .moe_intermediate_size = 128, .glm_mtp_layers = 1 };
    const code = TensorMeta{ .dtype = .u16, .shape = &.{ 2, 8, 8, 36 }, .data_start = 0, .data_end = 2 * 8 * 8 * 36 * 2, .data_base = 0, .file = "test.safetensors" };
    const scale = TensorMeta{ .dtype = .f16, .shape = &.{ 2, 128 }, .data_start = 0, .data_end = 2 * 128 * 2, .data_base = 0, .file = "test.safetensors" };
    for ([_][]const u8{ "gate", "up", "down" }) |p| for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
        try source.tensors.put(try std.fmt.allocPrint(a, "model.language_model.layers.1.mlp.switch_mlp.{s}_proj.{s}", .{ p, part }), if (i == 0) code else scale);
    };
    try source.tensors.put("model.language_model.layers.1.mlp.gate.weight", .{ .dtype = .bf16, .shape = &.{ 2, 128 }, .data_start = 0, .data_end = 512, .data_base = 0, .file = "test.safetensors" });
    try validateExl3Source(&source, a, &cfg);
    for ([_]u32{ 2, 0, 3 }) |li| {
        const key = try std.fmt.allocPrint(a, "model.language_model.layers.{d}.mlp.switch_mlp.up_proj.trellis", .{li});
        try source.tensors.put(key, code);
        defer _ = source.tensors.remove(key);
        if (validateExl3Source(&source, a, &cfg)) |_| return error.ExpectedGlmPackRejection else |_| {}
    }
}

test "GLM EXL3 preflight refuses a rate-grouped layer by name" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var source = SourceIndex{ .weight_map = std.StringHashMap([]const u8).init(a), .files = std.StringHashMap(void).init(a), .tensors = std.StringHashMap(TensorMeta).init(a), .stamps = std.StringHashMap(ShardStamp).init(a) };
    const cfg = model.ModelConfig{ .model_type = "glm5_next", .expert_layout = .exl3_k4, .expert_quant_rate = .{ .n = 36 }, .num_hidden_layers = 2, .first_k_dense_replace = 1, .num_experts = 2, .num_experts_per_tok = 1, .hidden_size = 128, .moe_intermediate_size = 128 };
    for ([_]u64{ 32, 36 }, 0..) |n, group| {
        const code = TensorMeta{ .dtype = .u16, .shape = try a.dupe(u64, &.{ 1, 8, 8, n }), .data_start = 0, .data_end = 8 * 8 * n * 2, .data_base = 0, .file = "test.safetensors" };
        const scale = TensorMeta{ .dtype = .f16, .shape = &.{ 1, 128 }, .data_start = 0, .data_end = 128 * 2, .data_base = 0, .file = "test.safetensors" };
        for ([_][]const u8{ "gate", "up", "down" }) |p| for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
            try source.tensors.put(try std.fmt.allocPrint(a, "model.language_model.layers.1.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ p, group, part }), if (i == 0) code else scale);
        };
    }
    try source.tensors.put("model.language_model.layers.1.mlp.gate.weight", .{ .dtype = .bf16, .shape = &.{ 2, 128 }, .data_start = 0, .data_end = 512, .data_base = 0, .file = "test.safetensors" });
    try t.expectError(error.Exl3RateGroupsUnsupported, validateExl3Source(&source, a, &cfg));
}
