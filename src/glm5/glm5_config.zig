const std = @import("std");
const expectError = @import("../test_expect.zig").expectError;
const model_mod = @import("../model.zig");
const ModelConfig = model_mod.ModelConfig;
const cfgF32 = model_mod.cfgF32;
const cfgInt = model_mod.cfgInt;
const cfgObject = model_mod.cfgObject;
const parseConfigFromJson = model_mod.parseConfigFromJson;
const parseVisionProcessorDefaultsFromJson = model_mod.parseVisionProcessorDefaultsFromJson;
const testing = std.testing;

fn glmU32(obj: std.json.ObjectMap, key: []const u8) !u32 {
    const value = try cfgInt(u32, obj.get(key) orelse return error.InvalidGlmConfig);
    if (value > std.math.maxInt(c_int)) return error.UnsupportedGlmConfig;
    return value;
}

fn glmF32(obj: std.json.ObjectMap, key: []const u8) !f32 {
    const value = try cfgF32(obj.get(key) orelse return error.InvalidGlmConfig);
    if (!std.math.isFinite(value)) return error.InvalidGlmConfig;
    return value;
}

fn glmRequireBool(obj: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const value = obj.get(key) orelse return error.InvalidGlmConfig;
    if (value != .bool) return error.InvalidGlmConfig;
    if (value.bool != expected) return error.UnsupportedGlmConfig;
}

fn glmRequireString(obj: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const value = obj.get(key) orelse return error.InvalidGlmConfig;
    if (value != .string) return error.InvalidGlmConfig;
    if (!std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
}

fn glmLayerList(obj: std.json.ObjectMap, key: []const u8, layers: u32, linear: bool) !void {
    const value = obj.get(key) orelse return;
    if (value != .array) return error.InvalidGlmConfig;
    var item: usize = 0;
    for (0..layers) |li| {
        if (((li + 1) % 4 != 0) != linear) continue;
        if (item >= value.array.items.len or value.array.items[item] != .integer or
            value.array.items[item].integer != li) return error.UnsupportedGlmConfig;
        item += 1;
    }
    if (item != value.array.items.len) return error.UnsupportedGlmConfig;
}

pub fn parseGlm5VisionFields(c: *ModelConfig, root: std.json.ObjectMap) !void {
    const value = root.get("vision_config") orelse return;
    if (value != .object) return error.InvalidGlmVisionConfig;
    const v = value.object;
    try glmRequireString(v, "model_type", "glm5_next_vision");
    try glmRequireString(v, "hidden_act", "silu");
    try glmRequireBool(v, "attention_bias", true);
    c.qv_depth = try glmU32(v, "depth");
    c.qv_hidden = try glmU32(v, "hidden_size");
    c.qv_heads = try glmU32(v, "num_heads");
    c.qv_intermediate = try glmU32(v, "intermediate_size");
    c.qv_patch = try glmU32(v, "patch_size");
    c.qv_temporal_patch = try glmU32(v, "temporal_patch_size");
    c.qv_merge = try glmU32(v, "spatial_merge_size");
    c.qv_out_hidden = try glmU32(v, "out_hidden_size");
    c.glmv_projection_intermediate = try glmU32(v, "projection_intermediate_size");
    c.glmv_eps = try glmF32(v, "rms_norm_eps");
    c.glmv_swiglu_limit = try glmF32(v, "swiglu_limit");
    if (c.qv_heads == 0 or c.qv_hidden % c.qv_heads != 0 or c.qv_out_hidden != c.hidden_size or
        c.qv_depth == 0 or c.qv_hidden == 0 or c.qv_intermediate == 0 or c.qv_patch == 0 or c.qv_patch > 64 or
        c.qv_temporal_patch == 0 or c.qv_temporal_patch > 8 or c.qv_merge != 2 or
        c.glmv_projection_intermediate == 0 or !std.math.isFinite(c.glmv_eps) or c.glmv_eps <= 0 or
        !std.math.isFinite(c.glmv_swiglu_limit) or c.glmv_swiglu_limit <= 0) return error.InvalidGlmVisionConfig;
    c.qv_head_dim = c.qv_hidden / c.qv_heads;
    if (c.qv_head_dim % 4 != 0) return error.InvalidGlmVisionConfig;
    if (v.get("rope_parameters")) |rp| {
        if (rp == .object) {
            try glmRequireString(rp.object, "rope_type", "axial");
            c.glmv_rope_theta = try glmF32(rp.object, "rope_theta");
            if (!std.math.isFinite(c.glmv_rope_theta) or c.glmv_rope_theta <= 0) return error.InvalidGlmVisionConfig;
        }
    }
    c.image_token_id = try glmU32(root, "image_token_id");
    c.video_token_id = try glmU32(root, "video_token_id");
    c.vision_start_token_id = try glmU32(root, "image_start_token_id");
    c.vision_end_token_id = try glmU32(root, "image_end_token_id");
    for ([_]u32{ c.image_token_id, c.video_token_id, c.vision_start_token_id, c.vision_end_token_id }) |id| if (id >= c.vocab_size) return error.InvalidGlmVisionConfig;
    c.glm5_vision = true;
    c.has_vision = true;
}

pub fn parseGlm5Fields(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    // These fields define the supported computation, not optional loader hints.
    try glmRequireBool(obj, "mhc", true);
    try glmRequireBool(obj, "attention_bias", false);
    try glmRequireBool(obj, "index_kpool_compress", true);
    try glmRequireBool(obj, "tie_word_embeddings", false);
    try glmRequireString(obj, "scoring_func", "sigmoid");
    try glmRequireString(obj, "topk_method", "noaux_tc");
    try glmRequireString(obj, "hidden_act", "silu");
    try glmRequireString(obj, "moe_router_dtype", "float32");
    c.hidden_size = try glmU32(obj, "hidden_size");
    c.vocab_size = try glmU32(obj, "vocab_size");
    c.intermediate_size = try glmU32(obj, "intermediate_size");
    c.num_hidden_layers = try glmU32(obj, "num_hidden_layers");
    c.num_attention_heads = try glmU32(obj, "num_attention_heads");
    c.num_experts_per_tok = try glmU32(obj, "num_experts_per_tok");
    c.moe_intermediate_size = try glmU32(obj, "moe_intermediate_size");
    c.max_position_embeddings = try glmU32(obj, "max_position_embeddings");
    c.rms_norm_eps = try glmF32(obj, "rms_norm_eps");
    c.has_vision = false;
    c.model_type = "glm5_next";
    c.weight_prefix = "model.language_model";
    c.norm_has_offset = false;
    c.scale_embeddings = false;
    c.has_pre_ff_norm = true;
    c.has_final_norm = true;
    c.has_qk_norm = false;
    c.hidden_act = .silu;
    c.has_sliding_window = false;
    c.attn_output_gate = false;
    c.partial_rotary_factor = 0;
    c.expert_layout = .bf16_individual;
    c.num_experts = try glmU32(obj, "n_routed_experts");
    c.first_k_dense_replace = try glmU32(obj, "first_k_dense_replace");
    c.shared_expert_intermediate_size = try std.math.mul(u32, c.moe_intermediate_size, try glmU32(obj, "n_shared_experts"));
    c.router_scaling_factor = try glmF32(obj, "routed_scaling_factor");
    c.moe_sigmoid_router = true;
    const norm = obj.get("norm_topk_prob") orelse return error.InvalidGlmConfig;
    if (norm != .bool) return error.InvalidGlmConfig;
    c.moe_route_norm = norm.bool;
    c.moe_n_group = try glmU32(obj, "n_group");
    c.moe_topk_group = try glmU32(obj, "topk_group");
    if (c.moe_n_group != 1 or c.moe_topk_group != 1) return error.UnsupportedGlmConfig;
    c.mla_q_lora_rank = try glmU32(obj, "q_lora_rank");
    c.mla_kv_lora_rank = try glmU32(obj, "kv_lora_rank");
    c.mla_qk_nope_head_dim = try glmU32(obj, "qk_nope_head_dim");
    c.mla_qk_rope_head_dim = try glmU32(obj, "qk_rope_head_dim");
    c.mla_v_head_dim = try glmU32(obj, "v_head_dim");
    const nope = obj.get("mla_use_nope") orelse return error.InvalidGlmConfig;
    if (nope != .bool or !nope.bool or c.mla_qk_rope_head_dim != 0) return error.UnsupportedGlmConfig;
    // Attention operates on one compressed latent, shared by every query head.
    c.head_dim = c.mla_kv_lora_rank;
    c.v_head_dim = c.mla_kv_lora_rank;
    c.num_key_value_heads = 1;
    c.query_pre_attn_scalar = c.mla_qk_nope_head_dim;
    const linear = try cfgObject(obj.get("linear_attn_config") orelse return error.InvalidGlmConfig);
    c.linear_num_key_heads = try glmU32(linear, "num_heads");
    c.linear_num_value_heads = c.linear_num_key_heads;
    c.linear_key_head_dim = try glmU32(linear, "head_dim");
    c.linear_value_head_dim = c.linear_key_head_dim;
    c.linear_conv_kernel_dim = try glmU32(linear, "short_conv_kernel_size");
    c.kda_vector_gate = true;
    c.kda_sigmoid_out_gate = true;
    c.kda_gate_lower_bound = try glmF32(linear, "gate_lower_bound");
    c.hc_count = try glmU32(obj, "hc_mult");
    c.glm_hc_sinkhorn_iters = try glmU32(obj, "hc_sinkhorn_iters");
    c.glm_hc_eps = try glmF32(obj, "hc_eps");
    c.glm_swiglu_limit = try glmF32(obj, "swiglu_limit");
    c.indexer_n_heads = try glmU32(obj, "index_n_heads");
    c.indexer_head_dim = try glmU32(obj, "index_head_dim");
    c.indexer_budget = try glmU32(obj, "index_topk");
    c.indexer_compress_ratio = try glmU32(obj, "index_kpool");
    const tail = obj.get("index_kpool_always_select_tail") orelse return error.InvalidGlmConfig;
    if (tail != .bool) return error.InvalidGlmConfig;
    c.glm_index_tail = tail.bool;
    if (obj.get("num_nextn_predict_layers")) |v| c.glm_mtp_layers = try cfgInt(u32, v);
    c.full_attention_interval = 4;
    if (c.hc_count != 4 or c.linear_key_head_dim != 128 or c.linear_conv_kernel_dim < 1 or
        c.num_experts == 0 or c.num_experts > 512 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > 32 or
        c.num_experts_per_tok > c.num_experts or c.first_k_dense_replace >= c.num_hidden_layers or
        c.num_hidden_layers > 128 or c.hidden_size == 0 or c.hidden_size % 128 != 0 or
        c.moe_intermediate_size == 0 or c.moe_intermediate_size % 128 != 0 or
        c.mla_kv_lora_rank == 0 or c.mla_q_lora_rank == 0 or c.mla_qk_nope_head_dim == 0 or
        c.glm_hc_sinkhorn_iters == 0 or c.glm_hc_sinkhorn_iters > 100 or c.glm_hc_eps <= 0 or
        c.glm_swiglu_limit != 10 or c.indexer_compress_ratio != 4 or c.indexer_budget < 4 or
        c.indexer_budget % 4 != 0 or c.router_scaling_factor <= 0 or c.kda_gate_lower_bound >= 0)
        return error.UnsupportedGlmConfig;
    if (c.vocab_size == 0 or c.intermediate_size == 0 or c.num_attention_heads == 0 or
        c.linear_num_key_heads == 0 or c.indexer_n_heads == 0 or c.indexer_head_dim == 0 or
        c.mla_v_head_dim == 0 or c.max_position_embeddings == 0 or c.rms_norm_eps <= 0) return error.UnsupportedGlmConfig;
    // Array dimensions and kernel index arithmetic use signed 32-bit products.
    for ([_][2]u32{
        .{ c.linear_num_key_heads, c.linear_key_head_dim },
        .{ c.indexer_n_heads, c.indexer_head_dim },
        .{ c.num_attention_heads, c.mla_qk_nope_head_dim },
        .{ c.num_attention_heads, c.mla_v_head_dim },
        .{ c.hc_count, c.hidden_size },
    }) |dims| {
        const width = std.math.mul(u32, dims[0], dims[1]) catch return error.UnsupportedGlmConfig;
        if (width > std.math.maxInt(c_int)) return error.UnsupportedGlmConfig;
    }
    if (obj.get("qk_head_dim")) |value| {
        if (try cfgInt(u32, value) != c.mla_qk_nope_head_dim) return error.UnsupportedGlmConfig;
    }
    try glmLayerList(linear, "kda_layers", c.num_hidden_layers, true);
    try glmLayerList(linear, "full_attn_layers", c.num_hidden_layers, false);
    const mlps = obj.get("mlp_layer_types") orelse return error.InvalidGlmConfig;
    if (mlps != .array or mlps.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
    for (mlps.array.items, 0..) |value, i| {
        const expected = if (i < c.first_k_dense_replace) "dense" else "sparse";
        if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
    }
    if (obj.get("indexer_types")) |types| {
        if (types != .array or types.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
        for (types.array.items) |value| {
            if (value != .string or !std.mem.eql(u8, value.string, "full")) return error.UnsupportedGlmConfig;
        }
    }
    const kinds = obj.get("layer_types") orelse return error.InvalidGlmConfig;
    if (kinds != .array or kinds.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
    for (kinds.array.items, 0..) |value, i| {
        const expected = if ((i + 1) % 4 == 0) "deepseek_sparse_attention" else "linear_attention";
        if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
    }
}

test "GLM config maps compressed MLA and FP32 recurrent state without Qwen assumptions" {
    const c = try parseConfigFromJson(testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    try testing.expect(c.isGlm5());
    try testing.expect(c.has_vision and c.glm5_vision);
    try testing.expectEqual(@as(u32, 14), c.qv_patch);
    try testing.expectEqual(@as(u32, 154854), c.image_token_id);
    try testing.expectEqual(@as(f32, 0.0625), c.attnScale());
    try testing.expectEqualStrings("model.language_model", c.weight_prefix);
    try testing.expectEqual(@as(u32, 45), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 288), c.num_experts);
    try testing.expectEqual(@as(u32, 8), c.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 3), c.first_k_dense_replace);
    try testing.expectEqual(@as(u32, 64), c.linear_num_value_heads);
    try testing.expectEqual(@as(u32, 128), c.linear_key_head_dim);
    try testing.expectEqual(@as(u32, 11), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u32, 512), c.mla_kv_lora_rank);
    try testing.expectEqual(@as(u32, 0), c.mla_qk_rope_head_dim);
    try testing.expectEqual(@as(u32, 20), c.glm_hc_sinkhorn_iters);
    try testing.expectEqual(@as(u64, 11 * 512 * 2), c.kvBytesPerToken());
    try testing.expectEqual(@as(u64, 34 * (64 * 128 * 128 * 4 + 3 * 3 * 64 * 128 * 2)), c.ssmCheckpointBytes());
    try testing.expectEqual(@as(u64, 11 * 128 * 2 / 4), c.qsaHistoryBytesPerToken());
    try testing.expectEqual(@as(u64, 0), c.qsaScoreBankBytesPerToken());
    try testing.expectEqual(@as(u64, 11 * 3 * 128 * 2 * 2), c.qsaRingBytes());
    try testing.expectApproxEqAbs(@as(f32, -5), c.kda_gate_lower_bound, 1e-6);
    try testing.expect(c.moe_sigmoid_router and c.moe_route_norm and !c.norm_has_offset);
}

test "GLM config refuses unsupported semantics and invalid geometry" {
    const source = @embedFile("../fixtures/glm5_config.json");
    const cases = [_][2][]const u8{
        .{ "\"mhc\": true", "\"mhc\": false" },
        .{ "\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"" },
        .{ "\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\"" },
        .{ "\"index_kpool_compress\": true", "\"index_kpool_compress\": false" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true" },
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"relu\"" },
        .{ "\"moe_router_dtype\": \"float32\"", "\"moe_router_dtype\": \"bfloat16\"" },
        .{ "\"dense\"", "\"sparse\"" },
        .{ "\"full\"", "\"other\"" },
        .{ "\"num_attention_heads\": 64", "\"num_attention_heads\": 0" },
        .{ "\"num_heads\": 64", "\"num_heads\": 0" },
        .{ "\"index_n_heads\": 32", "\"index_n_heads\": 0" },
        .{ "\"index_head_dim\": 128", "\"index_head_dim\": 0" },
        .{ "\"v_head_dim\": 256", "\"v_head_dim\": 0" },
        .{ "\"intermediate_size\": 12288", "\"intermediate_size\": 0" },
        .{ "\"vocab_size\": 154880", "\"vocab_size\": 0" },
        .{ "\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": 0" },
        .{ "\"max_position_embeddings\": 1048576", "\"max_position_embeddings\": 0" },
        .{ "\"qk_head_dim\": 256", "\"qk_head_dim\": 128" },
        .{ "\"hidden_size\": 4096,", "" },
        .{ "\"num_attention_heads\": 64,", "" },
        .{ "\"full_attn_layers\": [", "\"full_attn_layers\": [0," },
        .{ "\"kda_layers\": [", "\"kda_layers\": [99," },
        .{ "\"index_head_dim\": 128", "\"index_head_dim\": 2147483648" },
        .{ "\"index_n_heads\": 32", "\"index_n_heads\": 2147483647" },
    };
    for (cases) |change| {
        const raw = try std.mem.replaceOwned(u8, testing.allocator, source, change[0], change[1]);
        defer testing.allocator.free(raw);
        if (parseConfigFromJson(testing.allocator, raw)) |_| return error.ExpectedGlmConfigRejection else |err| {
            try testing.expect(err == error.InvalidGlmConfig or err == error.UnsupportedGlmConfig);
        }
    }
}

test "GLM config preserves optional shared expert counts" {
    for ([_]u32{ 0, 2 }) |count| {
        const replacement = try std.fmt.allocPrint(testing.allocator, "\"n_shared_experts\": {d}", .{count});
        defer testing.allocator.free(replacement);
        const raw = try std.mem.replaceOwned(u8, testing.allocator, @embedFile("../fixtures/glm5_config.json"), "\"n_shared_experts\": 1", replacement);
        defer testing.allocator.free(raw);
        const config = try parseConfigFromJson(testing.allocator, raw);
        try testing.expectEqual(count * 2048, config.shared_expert_intermediate_size);
    }
}

test "GLM vision config rejects unsupported tower geometry and accepts a text-only checkpoint" {
    const a = testing.allocator;
    const source = @embedFile("../fixtures/glm5_config.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, source, .{});
    defer parsed.deinit();
    var root = parsed.value.object;
    const vision = root.get("vision_config").?.object;
    var cfg = try parseConfigFromJson(a, source);
    try testing.expect(cfg.has_vision and cfg.glm5_vision and !cfg.qwen_vision);
    try testing.expectEqual(@as(u32, 64), cfg.qv_head_dim);
    try testing.expectEqual(@as(f32, 10000), cfg.glmv_rope_theta);
    try testing.expectEqual(@as(u32, 10240), cfg.glmv_projection_intermediate);
    var invalid = vision;
    invalid.getPtr("num_heads").?.* = .{ .integer = 15 };
    cfg.glm5_vision = false;
    try testing.expectError(error.InvalidGlmVisionConfig, parseGlm5VisionFields(&cfg, root));
    _ = root.swapRemove("vision_config");
    cfg.glm5_vision = false;
    cfg.has_vision = false;
    try parseGlm5VisionFields(&cfg, root);
    try testing.expect(!cfg.has_vision and !cfg.glm5_vision);
    const processor = parseVisionProcessorDefaultsFromJson("{\"image_processor\":{\"min_image_tokens\":16,\"max_image_tokens\":8000},\"video_processor\":{\"max_image_tokens\":240000}}");
    try testing.expectEqual(@as(?u32, 16), processor.min_image_tokens);
    try testing.expectEqual(@as(?u32, 8000), processor.max_image_tokens);
    try testing.expectEqual(@as(?u32, 240000), processor.max_video_tokens);
}

test "GLM raw FP8 config identifies source storage without changing native KV" {
    const source = @embedFile("../fixtures/glm5_config.json");
    const raw = try std.fmt.allocPrint(std.testing.allocator, "{{\"sushi_pack\":{{\"trunk_storage\":\"source-fp8-e4m3fn-block128\"}},{s}", .{source[1..]});
    defer std.testing.allocator.free(raw);
    const cfg = try parseConfigFromJson(std.testing.allocator, raw);
    try std.testing.expect(cfg.glm_fp8_trunk);
    try std.testing.expectEqual(@as(u64, 11 * 512 * 2), cfg.kvBytesPerToken());
    const affine = try parseConfigFromJson(std.testing.allocator, source);
    try std.testing.expect(!affine.glm_fp8_trunk);
    const release = try std.fmt.allocPrint(std.testing.allocator, "{{\"quantization_config\":{{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\",\"weight_block_size\":[128,128]}},{s}", .{source[1..]});
    defer std.testing.allocator.free(release);
    try std.testing.expect((try parseConfigFromJson(std.testing.allocator, release)).glm_fp8_trunk);
    const other_block = try std.fmt.allocPrint(std.testing.allocator, "{{\"quantization_config\":{{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"weight_block_size\":[64,64]}},{s}", .{source[1..]});
    defer std.testing.allocator.free(other_block);
    try expectError(error.UnsupportedGlmConfig, parseConfigFromJson(std.testing.allocator, other_block));
}
