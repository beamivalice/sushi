//! Independent oMLX MLA fixture checks; native attention remains separately tested.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const base = @import("glm5_model.zig");
const forward = @import("glm5_forward.zig");
const attention = @import("glm5_attention.zig");
const Arr = mlx.mlx_array;

fn fixture() !model.Weights {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "mla.safetensors", .data = @embedFile("../fixtures/glm5_mla.safetensors") });
    var buffer: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(t.io, &buffer);
    const path = try std.fmt.allocPrintSentinel(t.allocator, "{s}/mla.safetensors", .{buffer[0..n]}, 0);
    defer t.allocator.free(path);
    var tensors = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensors);
    var metadata = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(metadata);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try mlx.check(mlx.mlx_load_safetensors(&tensors, &metadata, path.ptr, cpu));
    const it = mlx.mlx_map_string_to_array_iterator_new(tensors);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
    var out = model.Weights.init(t.allocator);
    errdefer out.deinit();
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        if (mlx.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        try mlx.check(mlx.mlx_array_eval(value));
        try out.map.put(try t.allocator.dupe(u8, std.mem.span(key.?)), value);
    }
    return out;
}

fn compare(ops: *base.Ops, actual: Arr, expected: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(actual));
    const a = try ops.contiguous(try ops.cast(actual, .float32));
    const b = try ops.contiguous(try ops.cast(expected, .float32));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    var max_abs: f32 = 0;
    var error_sum: f64 = 0;
    var reference_sum: f64 = 0;
    const n = mlx.mlx_array_size(a);
    for (mlx.mlx_array_data_float32(a).?[0..n], mlx.mlx_array_data_float32(b).?[0..n]) |got, want| {
        const err = @as(f64, got) - want;
        max_abs = @max(max_abs, @abs(got - want));
        error_sum += err * err;
        reference_sum += @as(f64, want) * want;
    }
    const relative_l2 = @sqrt(error_sum / @max(reference_sum, 1e-30));
    // Expanded K/V and absorbed Q/V introduce different BF16 rounding points.
    // Bound the observed approximation without asserting bitwise equivalence.
    try std.testing.expect(max_abs <= 0.008);
    try std.testing.expect(relative_l2 <= 0.01);
}

fn runOracle(dense_prefill: bool) !void {
    var weights = try fixture();
    defer weights.deinit();
    const cfg = model.ModelConfig{
        .model_type = "glm5_next",
        .hidden_size = 128,
        .num_attention_heads = 2,
        .mla_q_lora_rank = 128,
        .mla_kv_lora_rank = 512,
        .mla_qk_nope_head_dim = 256,
        .mla_v_head_dim = 256,
        .indexer_n_heads = 2,
        .indexer_head_dim = 32,
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
        .rms_norm_eps = 1e-5,
    };
    var layer = try forward.Mla.load(&weights, "m", &cfg, mlx.gpuStream());
    defer layer.deinit();
    const Case = struct { name: []const u8, chunks: []const c_int, prefix: usize = 0, input: []const u8 = "input" };
    const ones: [33]c_int = @splat(1);
    for ([_]Case{ .{ .name = "full", .chunks = &.{33} }, .{ .name = "irregular", .chunks = &.{ 17, 1, 15 } }, .{ .name = "serial", .chunks = &ones }, .{ .name = "boundary.serial", .chunks = &.{ 1, 1, 1, 1, 1, 1 }, .prefix = 2047, .input = "boundary.input" }, .{ .name = "boundary.chunk", .chunks = &.{ 3, 3 }, .prefix = 2047, .input = "boundary.input" }, .{ .name = "boundary.dense", .chunks = &.{17}, .prefix = 2034, .input = "boundary.prefill.input" }, .{ .name = "boundary.fallback", .chunks = &.{17}, .prefix = 2035, .input = "boundary.prefill.input" } }) |case| {
        var state = attention.State.init();
        defer state.deinit();
        const prefix = case.prefix;
        if (prefix != 0) {
            var prep = base.Ops{ .s = mlx.gpuStream() };
            defer prep.deinit();
            _ = try state.append(try prep.slice(weights.get("boundary.latent").?, 0, 0, @intCast(prefix)), try prep.slice(weights.get("boundary.keys").?, 0, 0, @intCast(prefix)), try prep.slice(weights.get("boundary.gates").?, 0, 0, @intCast(prefix)), weights.get("m.indexer.index_kpool_compress_ape").?, mlx.gpuStream());
            for (state.arrays()) |a| if (a.ctx != null) try mlx.check(mlx.mlx_array_eval(a));
        }
        var pos: c_int = 0;
        var key: [64]u8 = undefined;
        const expected = weights.get(try std.fmt.bufPrint(&key, "{s}.output", .{case.name})).?;
        for (case.chunks) |count| {
            var ops = base.Ops{ .s = mlx.gpuStream() };
            defer ops.deinit();
            const x = try ops.slice(weights.get(case.input).?, 1, pos, pos + count);
            const y = try layer.applyMode(&ops, x, &cfg, &state, dense_prefill);
            try compare(&ops, y, try ops.slice(expected, 1, pos, pos + count));
            for (state.arrays()) |a| if (a.ctx != null) try mlx.check(mlx.mlx_array_eval(a));
            pos += count;
            try std.testing.expectEqual(prefix + @as(usize, @intCast(pos)), state.processed);
        }
    }
}

test "GLM MLA full source oracle covers prefill serial and cached irregular chunks" {
    try mlx.requireGpu();
    try runOracle(false);
}

test "GLM MLA dense prefill source oracle covers causal cached chunks" {
    try mlx.requireGpu();
    try runOracle(true);
}

test "GLM MLA dense prefill eligibility excludes decode and sparse boundary" {
    const cfg = model.ModelConfig{ .mla_qk_nope_head_dim = 256, .mla_v_head_dim = 256 };
    try std.testing.expect(forward.Mla.densePrefillEligible(true, 33, 0, &cfg, .bfloat16));
    try std.testing.expect(forward.Mla.densePrefillEligible(true, 17, 2034, &cfg, .bfloat16));
    try std.testing.expect(!forward.Mla.densePrefillEligible(true, 17, 2035, &cfg, .bfloat16));
    try std.testing.expect(!forward.Mla.densePrefillEligible(true, 1, 0, &cfg, .bfloat16));
    try std.testing.expect(!forward.Mla.densePrefillEligible(true, 8, 0, &cfg, .bfloat16));
    try std.testing.expect(!forward.Mla.densePrefillEligible(false, 33, 0, &cfg, .bfloat16));
    try std.testing.expect(!forward.Mla.densePrefillEligible(true, 33, 0, &cfg, .float32));
    var unsupported = cfg;
    unsupported.mla_qk_nope_head_dim = 512;
    try std.testing.expect(!forward.Mla.densePrefillEligible(true, 33, 0, &unsupported, .bfloat16));
}

fn putRandom(weights: *model.Weights, ops: *base.Ops, name: []const u8, shape: []const c_int, seed: u64, deviation: f32) !void {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    var value = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(value);
    try mlx.check(mlx.mlx_random_normal(&value, shape.ptr, shape.len, .bfloat16, 1, deviation, key.*, ops.s));
    try mlx.check(mlx.mlx_array_eval(value));
    try weights.map.put(try std.testing.allocator.dupe(u8, name), value);
}

test "GLM BF16 MLA layer takes the head-batched projections after the dense window and tracks the staged chain" {
    try mlx.requireGpu();
    if (!base.naxArms()) return error.SkipZigTest;
    const batch = @import("glm5_mla_prefill_batch.zig");
    var weights = model.Weights.init(std.testing.allocator);
    defer weights.deinit();
    var prep = base.Ops{ .s = mlx.gpuStream() };
    defer prep.deinit();
    const hidden = 128;
    const shapes = [_]struct { []const u8, []const c_int, f32 }{
        .{ "m.kv_b_proj.weight", &.{ 64 * 512, 512 }, 0.04 },
        .{ "m.q_a_proj.weight", &.{ 128, hidden }, 0.1 },
        .{ "m.q_b_proj.weight", &.{ 64 * 256, 128 }, 0.1 },
        .{ "m.kv_a_proj_with_mqa.weight", &.{ 512, hidden }, 0.1 },
        .{ "m.o_proj.weight", &.{ hidden, 64 * 256 }, 0.02 },
        .{ "m.indexer.wq_b.weight", &.{ 2 * 32, 128 }, 0.1 },
        .{ "m.indexer.wk.weight", &.{ 32, hidden }, 0.1 },
        .{ "m.indexer.weights_proj.weight", &.{ 2, hidden }, 0.1 },
        .{ "m.indexer.index_kpool_compress_gate", &.{ 32, hidden }, 0.1 },
        .{ "m.indexer.index_kpool_compress_ape", &.{ 4, 32 }, 0.1 },
        .{ "m.q_a_layernorm.weight", &.{128}, 0.1 },
        .{ "m.kv_a_layernorm.weight", &.{512}, 0.1 },
        .{ "m.indexer.k_norm.weight", &.{32}, 0.1 },
        .{ "m.indexer.k_norm.bias", &.{32}, 0.1 },
    };
    for (shapes, 0..) |entry, i| try putRandom(&weights, &prep, entry[0], entry[1], 100 + i, entry[2]);
    const cfg = model.ModelConfig{
        .model_type = "glm5_next",
        .hidden_size = hidden,
        .num_attention_heads = 64,
        .mla_q_lora_rank = 128,
        .mla_kv_lora_rank = 512,
        .mla_qk_nope_head_dim = 256,
        .mla_v_head_dim = 256,
        .indexer_n_heads = 2,
        .indexer_head_dim = 32,
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
        .rms_norm_eps = 1e-5,
    };
    var layer = try forward.Mla.load(&weights, "m", &cfg, mlx.gpuStream());
    defer layer.deinit();
    const rows: c_int = 130;
    var outputs: [2]Arr = undefined;
    var made: usize = 0;
    defer for (outputs[0..made]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    defer base.leaveTeacher();
    for ([_]bool{ false, true }) |reference| {
        base.reference_numerics = reference;
        var state = attention.State.init();
        defer state.deinit();
        var ops = base.Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 7));
        const x = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(x, &.{ 1, rows, hidden }, 3, .bfloat16, 0, 1, key.*, ops.s));
        const before = batch.dispatchCount();
        const y = try layer.applyMode(&ops, x.*, &cfg, &state, false);
        outputs[made] = try ops.result(try ops.contiguous(try ops.cast(y, .float32)));
        made += 1;
        try mlx.check(mlx.mlx_array_eval(outputs[made - 1]));
        try std.testing.expectEqual(before + @as(usize, if (reference) 0 else 2), batch.dispatchCount());
    }
    var num: f64 = 0;
    var den: f64 = 0;
    const n = mlx.mlx_array_size(outputs[0]);
    for (mlx.mlx_array_data_float32(outputs[0]).?[0..n], mlx.mlx_array_data_float32(outputs[1]).?[0..n]) |got, want| {
        try std.testing.expect(std.math.isFinite(got));
        num += (@as(f64, got) - want) * (@as(f64, got) - want);
        den += @as(f64, want) * want;
    }
    try std.testing.expect(den > 0 and @sqrt(num / den) <= 0.01);
}
