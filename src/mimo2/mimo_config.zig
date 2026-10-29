const std = @import("std");
const expectError = @import("../test_expect.zig").expectError;
const expert_quant = @import("../expert_quant.zig");
const model_discovery = @import("../model_discovery.zig");
const model_mod = @import("../model.zig");
const qwen4_config = @import("../qwen4/qwen4_config.zig");
const MAX_VISION_LAYERS = model_mod.MAX_VISION_LAYERS;
const MimoVitAttn = model_mod.MimoVitAttn;
const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;
const jsonFloat = model_mod.jsonFloat;
const mergeConfigJson = model_mod.mergeConfigJson;
const mimoSourceResidentBytes = model_mod.mimoSourceResidentBytes;
const parseConfig = model_mod.parseConfig;
const parseConfigFromJson = model_mod.parseConfigFromJson;
const streamingResidentSplit = model_mod.streamingResidentSplit;
const testing = std.testing;

fn mimoUint(obj: std.json.ObjectMap, key: []const u8, fallback: u32) !u32 {
    const v = obj.get(key) orelse return fallback;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32))
        return error.UnsupportedMimoV2Config;
    return @intCast(v.integer);
}

fn mimoFloat(obj: std.json.ObjectMap, key: []const u8, fallback: f32) !f32 {
    const v = obj.get(key) orelse return fallback;
    if (v != .integer and v != .float) return error.UnsupportedMimoV2Config;
    const f = jsonFloat(v);
    if (!std.math.isFinite(f)) return error.UnsupportedMimoV2Config;
    return f;
}

fn mimoBool(obj: std.json.ObjectMap, key: []const u8, fallback: bool) !bool {
    const v = obj.get(key) orelse return fallback;
    if (v != .bool) return error.UnsupportedMimoV2Config;
    return v.bool;
}

/// MiMo-ViT geometry, the image token ids, and the pixel bounds the vendor
/// processors read from config.json's own `processor_config` (they ignore
/// preprocessor_config.json).
fn parseMimoVision(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    const vc_val = obj.get("vision_config") orelse return;
    if (vc_val != .object) return error.UnsupportedMimoV2Config;
    const vc = vc_val.object;
    c.qv_depth = try mimoUint(vc, "depth", 0);
    c.qv_hidden = try mimoUint(vc, "hidden_size", 0);
    c.qv_heads = try mimoUint(vc, "num_heads", 0);
    c.qv_head_dim = try mimoUint(vc, "qk_channels", 64);
    c.mvit_kv_heads = try mimoUint(vc, "num_key_value_heads", c.qv_heads);
    c.qv_intermediate = try mimoUint(vc, "intermediate_size", 0);
    c.qv_out_hidden = try mimoUint(vc, "out_hidden_size", c.hidden_size);
    c.qv_patch = try mimoUint(vc, "patch_size", 16);
    c.qv_merge = try mimoUint(vc, "spatial_merge_size", 2);
    c.qv_temporal_patch = try mimoUint(vc, "temporal_patch_size", 2);
    c.mvit_window = try mimoUint(vc, "visual_token_window_size", 0);
    c.mvit_sinks = try mimoBool(vc, "use_sink", false);
    if (c.qv_depth == 0 or c.qv_depth > MAX_VISION_LAYERS or c.qv_hidden == 0 or c.qv_heads == 0 or
        c.qv_head_dim == 0 or c.mvit_kv_heads == 0 or c.qv_heads % c.mvit_kv_heads != 0 or
        c.qv_intermediate == 0 or c.qv_out_hidden != c.hidden_size or c.qv_patch == 0 or c.qv_merge == 0 or
        c.qv_temporal_patch == 0 or c.mvit_window == 0)
        return error.UnsupportedMimoV2Config;

    var full: [MAX_VISION_LAYERS]bool = @splat(false);
    if (vc.get("fullatt_block_indexes")) |v| {
        if (v != .array) return error.UnsupportedMimoV2Config;
        for (v.array.items) |item| {
            if (item != .integer or item.integer < 0 or item.integer >= c.qv_depth) return error.UnsupportedMimoV2Config;
            full[@intCast(item.integer)] = true;
        }
    }
    const types = vc.get("vit_window_attn_types") orelse return error.UnsupportedMimoV2Config;
    if (types != .array or types.array.items.len != c.qv_depth) return error.UnsupportedMimoV2Config;
    for (types.array.items, 0..) |item, i| {
        if (item != .integer) return error.UnsupportedMimoV2Config;
        c.mvit_attn[i] = if (full[i]) .full else switch (item.integer) {
            -1, 0 => .row,
            1 => .col,
            else => return error.UnsupportedMimoV2Config,
        };
    }

    c.image_token_id = try mimoUint(obj, "image_token_id", 0);
    c.vision_start_token_id = try mimoUint(obj, "vision_start_token_id", 0);
    c.vision_end_token_id = try mimoUint(obj, "vision_end_token_id", 0);
    if (c.image_token_id == 0 or c.vision_start_token_id == 0 or c.vision_end_token_id == 0)
        return error.UnsupportedMimoV2Config;
    if (obj.get("processor_config")) |pc| {
        if (pc != .object) return error.UnsupportedMimoV2Config;
        c.qv_min_pixels = try mimoUint(pc.object, "image_min_pixels", 0);
        c.qv_max_pixels = try mimoUint(pc.object, "image_max_pixels", 0);
    }
    c.mimo_vision = true;
}

pub fn parseMimoConfig(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    c.model_type = "mimo_v2";
    c.weight_prefix = "model";
    c.norm_has_offset = false;
    c.scale_embeddings = false;
    c.has_pre_ff_norm = false;
    c.has_qk_norm = false;
    c.hidden_act = .silu;
    try parseMimoVision(c, obj);
    c.has_vision = c.mimo_vision;
    // Video is not served yet; its pads must never join the splice.
    c.video_token_id = 0;
    c.has_sliding_window = true;
    c.has_explicit_layer_types = true;
    c.rope_scaling_factor = 1;
    c.rms_norm_eps = try mimoFloat(obj, "layernorm_epsilon", c.rms_norm_eps);
    c.partial_rotary_factor = try mimoFloat(obj, "partial_rotary_factor", c.partial_rotary_factor);
    c.rope_local_base_freq = try mimoFloat(obj, "swa_rope_theta", 10000);
    c.attention_value_scale = try mimoFloat(obj, "attention_value_scale", 1);
    c.sliding_window = try mimoUint(obj, "sliding_window", try mimoUint(obj, "sliding_window_size", 128));
    if (c.num_hidden_layers == 0 or c.num_hidden_layers > c.layer_is_global.len or
        c.sliding_window == 0 or c.hidden_size == 0 or c.rms_norm_eps <= 0 or
        c.rope_theta <= 0 or c.rope_local_base_freq <= 0 or
        c.partial_rotary_factor <= 0 or c.partial_rotary_factor > 1)
        return error.UnsupportedMimoV2Config;

    const pattern = obj.get("hybrid_layer_pattern") orelse return error.UnsupportedMimoV2Config;
    if (pattern != .array or pattern.array.items.len != c.num_hidden_layers)
        return error.UnsupportedMimoV2Config;
    for (pattern.array.items, 0..) |v, i| {
        if (v != .integer or (v.integer != 0 and v.integer != 1)) return error.UnsupportedMimoV2Config;
        c.layer_is_global[i] = v.integer == 0;
    }

    // The config's plain geometry is global; our layer helpers use sliding defaults.
    c.global_head_dim = c.head_dim;
    c.global_v_head_dim = try mimoUint(obj, "v_head_dim", c.head_dim);
    c.num_global_key_value_heads = c.num_key_value_heads;
    c.head_dim = try mimoUint(obj, "swa_head_dim", c.global_head_dim);
    c.v_head_dim = try mimoUint(obj, "swa_v_head_dim", c.global_v_head_dim);
    c.num_key_value_heads = try mimoUint(obj, "swa_num_key_value_heads", c.num_global_key_value_heads);
    const swa_heads = try mimoUint(obj, "swa_num_attention_heads", c.num_attention_heads);
    if (c.global_head_dim == 0 or c.global_v_head_dim == 0 or c.num_global_key_value_heads == 0 or
        c.head_dim == 0 or c.v_head_dim == 0 or c.num_key_value_heads == 0 or
        c.num_attention_heads == 0 or swa_heads == 0)
        return error.UnsupportedMimoV2Config;
    c.has_per_layer_heads = true;
    for (0..c.num_hidden_layers) |i| {
        c.num_attention_heads_per_layer[i] = if (c.layer_is_global[i]) c.num_attention_heads else swa_heads;
        const li: u32 = @intCast(i);
        const heads = c.layerNumHeads(li);
        const kv = c.layerKVHeads(li);
        const hd = c.layerHeadDim(li);
        if (heads == 0 or kv == 0 or heads % kv != 0 or hd == 0 or c.layerVHeadDim(li) == 0)
            return error.UnsupportedMimoV2Config;
        const rd: u32 = @intFromFloat(@as(f64, @floatFromInt(hd)) * c.partial_rotary_factor);
        if (rd == 0 or rd % 2 != 0) return error.UnsupportedMimoV2Config;
    }
    c.query_pre_attn_scalar = c.global_head_dim;
    c.attn_sinks_sliding = try mimoBool(obj, "add_swa_attention_sink_bias", true);
    c.attn_sinks_global = try mimoBool(obj, "add_full_attention_sink_bias", false);
    c.has_attn_sinks = c.attn_sinks_sliding or c.attn_sinks_global;
    if (obj.get("attention_projection_layout")) |v| {
        if (v != .string) return error.UnsupportedMimoV2Config;
        c.attn_fused_qkv = std.mem.eql(u8, v.string, "fused_qkv");
        if (!c.attn_fused_qkv and !std.mem.eql(u8, v.string, "split_qkv") and
            !std.mem.eql(u8, v.string, "split"))
            return error.UnsupportedMimoV2Config;
    }

    c.num_experts = try mimoUint(obj, "n_routed_experts", c.num_experts);
    c.moe_sigmoid_router = true;
    c.moe_n_group = try mimoUint(obj, "n_group", 1);
    c.moe_topk_group = try mimoUint(obj, "topk_group", 1);
    c.moe_route_norm = try mimoBool(obj, "norm_topk_prob", true);
    if (obj.get("routed_scaling_factor")) |v| {
        if (v != .null) c.router_scaling_factor = try mimoFloat(obj, "routed_scaling_factor", 1);
    }
    for ([_][]const u8{ "scoring_func", "topk_method", "hidden_act" }, [_][]const u8{ "sigmoid", "noaux_tc", "silu" }) |key, expected| {
        if (obj.get(key)) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedMimoV2Config;
        }
    }
    if (obj.get("n_shared_experts")) |v| {
        if (v != .null and (v != .integer or v.integer != 0)) return error.UnsupportedMimoV2Config;
    }
    if (c.num_experts == 0 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > c.num_experts or
        c.moe_intermediate_size == 0 or c.moe_n_group == 0 or c.num_experts % c.moe_n_group != 0 or
        c.moe_topk_group == 0 or c.moe_topk_group > c.moe_n_group or
        c.num_experts_per_tok > c.num_experts / c.moe_n_group * c.moe_topk_group or
        (c.moe_n_group > 1 and c.num_experts / c.moe_n_group < 2))
        return error.UnsupportedMimoV2Config;
    const freq = obj.get("moe_layer_freq") orelse return error.UnsupportedMimoV2Config;
    c.first_k_dense_replace = try model_discovery.denseMoePrefix(freq, c.num_hidden_layers);
    // A pack stores its packed trunk linears (docs/pack-format.md); the engine
    // quantizes nothing at load, so a config asking it to is refused.
    if (obj.get("trunk_quant") != null) return error.UnsupportedMimoV2Config;
}

const MIMO_V2_VISION_JSON =
    \\{
    \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
    \\  "num_hidden_layers": 2, "intermediate_size": 1536,
    \\  "num_attention_heads": 4, "num_key_value_heads": 2,
    \\  "head_dim": 192, "v_head_dim": 128,
    \\  "hybrid_layer_pattern": [0,1], "sliding_window": 128,
    \\  "partial_rotary_factor": 0.334, "moe_layer_freq": [0,1],
    \\  "n_routed_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 192,
    \\  "image_token_id": 101, "video_token_id": 102, "audio_token_id": 103,
    \\  "vision_start_token_id": 104, "vision_end_token_id": 105,
    \\  "vision_config": {
    \\    "depth": 4, "hidden_size": 64, "num_heads": 4, "num_key_value_heads": 2,
    \\    "intermediate_size": 96, "out_hidden_size": 384, "patch_size": 16,
    \\    "spatial_merge_size": 2, "temporal_patch_size": 2, "use_sink": true,
    \\    "fullatt_block_indexes": [0, 3], "vit_window_attn_types": [-1, 0, 1, -1],
    \\    "visual_token_window_size": 64, "window_size": 128
    \\  },
    \\  "processor_config": {"image_min_pixels": 8192, "image_max_pixels": 8388608}
    \\}
;

/// The mimo_v2 geometry of "ModelConfig parses mimo_v2 hybrid geometry", kept
/// beside the bill assertions so the expected bytes read against one spelling.
const MIMO_V2_BILL_JSON =
    \\{
    \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
    \\  "num_hidden_layers": 4, "intermediate_size": 1536,
    \\  "num_attention_heads": 4, "num_key_value_heads": 2,
    \\  "head_dim": 192, "v_head_dim": 128,
    \\  "swa_num_attention_heads": 6, "swa_num_key_value_heads": 3,
    \\  "swa_head_dim": 192, "swa_v_head_dim": 128,
    \\  "hybrid_layer_pattern": [0,1,1,0], "sliding_window": 128,
    \\  "rope_theta": 10000000, "swa_rope_theta": 10000,
    \\  "partial_rotary_factor": 0.334, "attention_value_scale": 0.707,
    \\  "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": false,
    \\  "attention_projection_layout": "split_qkv", "layernorm_epsilon": 0.00001,
    \\  "n_routed_experts": 16, "num_experts_per_tok": 4,
    \\  "moe_intermediate_size": 192, "moe_layer_freq": [0,1,1,1],
    \\  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    \\  "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
    \\  "routed_scaling_factor": null, "n_shared_experts": null,
    \\  "eos_token_id": 17, "tie_word_embeddings": false,
    \\  "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    \\}
;

test "defaultEnableThinking: mimo_v2 thinks by default, with and without tools" {
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    try testing.expect(mimo.defaultEnableThinking());
}

test "ModelConfig parses mimo_v2 hybrid geometry and sigmoid routing" {
    const json =
        \\{
        \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
        \\  "num_hidden_layers": 4, "intermediate_size": 1536,
        \\  "num_attention_heads": 4, "num_key_value_heads": 2,
        \\  "head_dim": 192, "v_head_dim": 128,
        \\  "swa_num_attention_heads": 6, "swa_num_key_value_heads": 3,
        \\  "swa_head_dim": 192, "swa_v_head_dim": 128,
        \\  "hybrid_layer_pattern": [0,1,1,0], "sliding_window": 128,
        \\  "rope_theta": 10000000, "swa_rope_theta": 10000,
        \\  "partial_rotary_factor": 0.334, "attention_value_scale": 0.707,
        \\  "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": false,
        \\  "attention_projection_layout": "split_qkv", "layernorm_epsilon": 0.00001,
        \\  "n_routed_experts": 16, "num_experts_per_tok": 4,
        \\  "moe_intermediate_size": 192, "moe_layer_freq": [0,1,1,1],
        \\  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
        \\  "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        \\  "routed_scaling_factor": null, "n_shared_experts": null,
        \\  "eos_token_id": 17, "tie_word_embeddings": false,
        \\  "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        \\}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("mimo_v2", c.model_type);
    try testing.expectEqualStrings("model", c.weight_prefix);
    try testing.expect(c.has_explicit_layer_types);
    try testing.expect(c.isGlobalLayer(0) and c.isGlobalLayer(3));
    try testing.expect(!c.isGlobalLayer(1) and !c.isGlobalLayer(2));
    try testing.expectEqual(@as(u32, 4), c.layerNumHeads(0));
    try testing.expectEqual(@as(u32, 6), c.layerNumHeads(1));
    try testing.expectEqual(@as(u32, 2), c.layerKVHeads(0));
    try testing.expectEqual(@as(u32, 3), c.layerKVHeads(1));
    try testing.expectEqual(@as(u32, 192), c.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(0));
    try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(1));
    try testing.expect(!c.layerHasAttnSinks(0) and c.layerHasAttnSinks(1));
    try testing.expectEqual(@as(f32, 0.707), c.attention_value_scale);
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(@as(f32, 0.334), c.partial_rotary_factor);
    try testing.expectEqual(@as(f32, 1e7), c.rope_theta);
    try testing.expectEqual(@as(f32, 1e4), c.rope_local_base_freq);
    try testing.expectEqual(@as(f32, 1e-5), c.rms_norm_eps);
    try testing.expectEqual(@as(u32, 16), c.num_experts);
    try testing.expectEqual(@as(u32, 1), c.first_k_dense_replace);
    try testing.expect(c.moe_sigmoid_router and c.moe_route_norm);
    try testing.expectEqual(@as(f32, 1), c.router_scaling_factor);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expect(!c.norm_has_offset and !c.scale_embeddings and !c.has_qk_norm);
    try testing.expect(!c.has_pre_ff_norm and !c.has_vision);
    try testing.expect(c.isEosToken(17));
}

test "mimo_v2 config reads the MiMo-ViT geometry and the processor's own pixel bounds" {
    const c = try parseConfigFromJson(testing.allocator, MIMO_V2_VISION_JSON);
    try testing.expect(c.has_vision and c.mimo_vision and !c.qwen_vision);
    try testing.expectEqual(@as(u32, 4), c.qv_depth);
    try testing.expectEqual(@as(u32, 64), c.qv_hidden);
    try testing.expectEqual(@as(u32, 4), c.qv_heads);
    // The reference's `qk_channels` default, not hidden / heads.
    try testing.expectEqual(@as(u32, 64), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 2), c.mvit_kv_heads);
    try testing.expectEqual(@as(u32, 96), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 384), c.qv_out_hidden);
    try testing.expectEqual(@as(u32, 16), c.qv_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_merge);
    try testing.expectEqual(@as(u32, 2), c.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 64), c.mvit_window);
    try testing.expect(c.mvit_sinks);
    try testing.expectEqualSlices(MimoVitAttn, &.{ .full, .row, .col, .full }, c.mvit_attn[0..4]);
    try testing.expectEqual(@as(u32, 8192), c.qv_min_pixels);
    try testing.expectEqual(@as(u32, 8388608), c.qv_max_pixels);
    try testing.expectEqual(@as(u32, 101), c.image_token_id);
    try testing.expectEqual(@as(u32, 104), c.vision_start_token_id);
    try testing.expectEqual(@as(u32, 105), c.vision_end_token_id);
    // Video and audio are not served yet: their placeholders must never join the splice.
    try testing.expectEqual(@as(u32, 0), c.video_token_id);
}

test "mimo_v2 config refuses a MiMo-ViT whose block tables disagree with its depth" {
    const cases = [_][]const u8{
        "\"fullatt_block_indexes\": [0, 4], \"vit_window_attn_types\": [-1, 0, 1, -1]",
        "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 1]",
        "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 2, -1]",
    };
    for (cases) |tables| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, MIMO_V2_VISION_JSON, "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 1, -1]", tables);
        defer testing.allocator.free(json);
        try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
    }
}

test "mimo_v2 config rejects unsupported routing and malformed layer geometry" {
    const base =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":384,
        \\ "num_attention_heads":4, "num_key_value_heads":2, "head_dim":192,
        \\ "v_head_dim":128, "partial_rotary_factor":0.334,
        \\ "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
        \\ "n_routed_experts":16, "num_experts_per_tok":4, "moe_intermediate_size":192}
    ;
    const good = try parseConfigFromJson(testing.allocator, base);
    try testing.expectEqualStrings("mimo_v2", good.model_type);
    for ([_][]const u8{
        \\{"scoring_func":"softmax"}
        ,
        \\{"topk_method":"greedy"}
        ,
        \\{"hidden_act":"gelu"}
        ,
        \\{"n_shared_experts":1}
        ,
        \\{"hybrid_layer_pattern":[0]}
        ,
        \\{"hybrid_layer_pattern":[0,2]}
        ,
        \\{"moe_layer_freq":[1,0]}
        ,
        \\{"moe_layer_freq":2}
        ,
        \\{"swa_num_attention_heads":3}
        ,
        \\{"swa_v_head_dim":0}
        ,
        \\{"partial_rotary_factor":0.34}
        ,
        \\{"partial_rotary_factor":2}
        ,
        \\{"attention_projection_layout":"interleaved"}
        ,
        \\{"sliding_window":0}
        ,
        \\{"n_group":0}
        ,
        \\{"n_group":3}
        ,
        \\{"n_group":4,"topk_group":5}
        ,
        \\{"n_group":16,"topk_group":4}
        ,
        \\{"layernorm_epsilon":0}
        ,
    }) |override| {
        const json = try mergeConfigJson(testing.allocator, base, override);
        defer testing.allocator.free(json);
        try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
    }
    const fused_json = try mergeConfigJson(testing.allocator, base,
        \\{"attention_projection_layout":"fused_qkv","routed_scaling_factor":2.5,
        \\ "norm_topk_prob":false,"add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":true}
    );
    defer testing.allocator.free(fused_json);
    const fused = try parseConfigFromJson(testing.allocator, fused_json);
    try testing.expect(fused.attn_fused_qkv and !fused.moe_route_norm);
    try testing.expectEqual(@as(f32, 2.5), fused.router_scaling_factor);
    try testing.expect(fused.layerHasAttnSinks(0) and !fused.layerHasAttnSinks(1));
}

test "mimo_v2 refuses a config asking to quantize trunk linears at load" {
    const base =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":384,
        \\ "num_attention_heads":4, "num_key_value_heads":2, "head_dim":192,
        \\ "v_head_dim":128, "partial_rotary_factor":0.334,
        \\ "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
        \\ "n_routed_experts":16, "num_experts_per_tok":4, "moe_intermediate_size":192}
    ;
    _ = try parseConfigFromJson(testing.allocator, base);
    const json = try mergeConfigJson(testing.allocator, base,
        \\{"trunk_quant":{"o_proj":{"mode":"affine","bits":8,"group_size":64}}}
    );
    defer testing.allocator.free(json);
    try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
}

test "real mimo_v2 Flash config agrees with source geometry" {
    const raw = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    var c = try parseConfig(testing.io, testing.allocator, std.mem.span(raw));
    defer c.deinit(testing.allocator);
    try testing.expectEqualStrings("mimo_v2", c.model_type);
    try testing.expectEqual(@as(u32, 48), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 256), c.num_experts);
    try testing.expectEqual(@as(u32, 8), c.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 1), c.first_k_dense_replace);
    var global: u32 = 0;
    for (0..c.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        global += @intFromBool(c.isGlobalLayer(li));
        try testing.expectEqual(@as(u32, 64), c.layerNumHeads(li));
        try testing.expectEqual(@as(u32, if (c.isGlobalLayer(li)) 4 else 8), c.layerKVHeads(li));
        try testing.expectEqual(@as(u32, 192), c.layerHeadDim(li));
        try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(li));
        try testing.expectEqual(!c.isGlobalLayer(li), c.layerHasAttnSinks(li));
    }
    try testing.expectEqual(@as(u32, 9), global);
    // What the session costs: 9 global layers per token, the 39 sliding ones a
    // ring held once per slot.
    try testing.expectEqual(@as(u64, 9 * 4 * (192 + 128) * 2), c.kvBytesPerToken());
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_SLACK, c.swaRingTokens());
    try testing.expectEqual(c.swaRingTokens() * 39 * 8 * (192 + 128) * 2, c.swaRingBytes());
    try testing.expectEqual(@as(f32, 0.707), c.attention_value_scale);
    try testing.expect(c.isEosToken(151643) and c.isEosToken(151645) and c.isEosToken(151672));
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expectEqual(@as(u32, 4), c.quant_bits);
    try testing.expectEqual(@as(u32, 32), c.quant_group_size);
    try testing.expect(c.expertStreamingRequired());

    try testing.expect(c.mimo_vision);
    try testing.expectEqual(@as(u32, 28), c.qv_depth);
    try testing.expectEqual(@as(u32, 1280), c.qv_hidden);
    try testing.expectEqual(@as(u32, 32), c.qv_heads);
    try testing.expectEqual(@as(u32, 8), c.mvit_kv_heads);
    try testing.expectEqual(@as(u32, 64), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 4608), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 64), c.mvit_window);
    var full: u32 = 0;
    for (c.mvit_attn[0..c.qv_depth]) |kind| full += @intFromBool(kind == .full);
    try testing.expectEqual(@as(u32, 4), full);
    try testing.expect(c.mvit_attn[0] == .full and c.mvit_attn[27] == .full and c.mvit_attn[1] == .row and c.mvit_attn[5] == .col);
    try testing.expectEqual(@as(u32, 8192), c.qv_min_pixels);
    try testing.expectEqual(@as(u32, 8388608), c.qv_max_pixels);
    try testing.expectEqual(@as(u32, 151655), c.image_token_id);
}

test "mimo_v2 bills per-layer KV geometry and the sliding window once per slot" {
    const c = try parseConfigFromJson(testing.allocator, MIMO_V2_BILL_JSON);
    // Global layers 0 and 3 at 2 KV heads x (qk 192 + v 128) x 2 bytes. The
    // sliding pair contributes NOTHING per token — its storage is a ring.
    try testing.expectEqual(@as(u64, 2 * 2 * (192 + 128) * 2), c.kvBytesPerToken());
    // The ring: two sliding layers at 3 KV heads x 320 x 2 bytes, held for
    // `swaRingTokens` rows however long the session runs.
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_SLACK, c.swaRingTokens());
    try testing.expectEqual(c.swaRingTokens() * 2 * 3 * (192 + 128) * 2, c.swaRingBytes());
    // A restore point is the window plus the backoff, per sliding layer, at the same width.
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_CHECKPOINT_BACKOFF, c.swaRingCheckpointTokens());
    try testing.expectEqual(c.swaRingCheckpointTokens() * 2 * 3 * (192 + 128) * 2, c.swaRingCheckpointBytes());
    // What one chunk token stages in those layers before the ring compacts,
    // for as many of them as one eval-cadence window lets coexist.
    try testing.expectEqual(@as(u64, 2 * 3 * (192 + 128) * 2), c.swaStreamBytesPerToken(5));
    try testing.expectEqual(@as(u64, 1 * 3 * (192 + 128) * 2), c.swaStreamBytesPerToken(1));
    try testing.expectEqual(@as(u64, 0), c.swaStreamBytesPerToken(0));
}

test "real mimo_v2 original and converted packs bill the same resident trunk" {
    const source = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const reference = std.c.getenv("MIMO_PACK_REFERENCE") orelse return error.SkipZigTest;
    var config = try parseConfig(testing.io, testing.allocator, std.mem.span(source));
    defer config.deinit(testing.allocator);
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, config.expert_layout);
    const original = try streamingResidentSplit(testing.io, testing.allocator, std.mem.span(source), &config);
    const converted = try streamingResidentSplit(testing.io, testing.allocator, std.mem.span(reference), &.{ .expert_layout = .mxfp4_split });
    try testing.expect(original.trunk > 0);
    try testing.expectEqual(converted, original);
}

test "real mimo_v2 bill carries the bf16 vision tower exactly when it is loaded" {
    const source = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const dir = std.mem.span(source);
    const text = try mimoSourceResidentBytes(testing.io, testing.allocator, dir, false);
    const with_tower = try mimoSourceResidentBytes(testing.io, testing.allocator, dir, true);
    // 364 `visual.*` tensors, all bf16, as stored.
    try testing.expectEqual(@as(u64, 1_457_188_864), with_tower - text);
}

test "mimo_v2 original config selects split QKV and native expert quantization" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try expert_quant.writeTinyMxfp4IndividualCheckpoint(a, tmp.dir, 4, 128, 128);
    defer a.free(raw);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"mimo_v2","num_hidden_layers":2,"hidden_size":128,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
        \\ "v_head_dim":32,"swa_head_dim":32,"swa_v_head_dim":32,
        \\ "swa_num_attention_heads":4,"swa_num_key_value_heads":2,
        \\ "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
        \\ "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,
        \\ "attention_projection_layout":"fused_qkv",
        \\ "quantization_config":{"quant_method":"fp8","store_dtype":"mxfp4"}}
        ,
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &path_buf);
    var c = try parseConfig(testing.io, a, path_buf[0..n]);
    defer c.deinit(a);
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, c.expert_layout);
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expectEqual(@as(u32, 4), c.quant_bits);
    try testing.expectEqual(@as(u32, 32), c.quant_group_size);
    try testing.expect(c.expertStreamingRequired());
}

test "mimo_v2 EXL3 routed banks stream on request and take the source trunk loader" {
    var c = ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 2,
        .first_k_dense_replace = 1,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 0,
        .expert_layout = .exl3_k4,
    };
    try testing.expect(c.supportsExpertStreaming());
    try testing.expect(!c.expertStreamingRequired());
    try testing.expect(c.usesMimoSourceTrunk());
    c.expert_layout = .mxfp4_individual;
    try testing.expect(c.expertStreamingRequired());
    try testing.expect(c.usesMimoSourceTrunk());
    c.expert_layout = .mxfp4_split;
    try testing.expect(!c.usesMimoSourceTrunk());
    var q = ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 2,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 0,
        .expert_layout = .exl3_k4,
    };
    try testing.expect(!q.usesMimoSourceTrunk());
    try testing.expect(!q.expertStreamingRequired());
}

test "mimo_v2 original MXFP4 experts require streaming despite their quantized width" {
    var c = ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 2,
        .first_k_dense_replace = 1,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 4,
        .quant_group_size = 32,
        .quant_mode = .mxfp4,
        .expert_layout = .mxfp4_individual,
    };
    try testing.expect(c.expertStreamingRequired());
    c.expert_layout = .mxfp4_split;
    try testing.expect(!c.expertStreamingRequired());
}

test "mimo_v2 original streaming retains resident names and excludes experts and MTP" {
    var buf: [256]u8 = undefined;
    const key = "model.layers.0.self_attn.qkv_proj.weight";
    try testing.expectEqualStrings(key, qwen4_config.qwen4StreamingWeightKey(.mxfp4_individual, &buf, key).?);
    try testing.expect(qwen4_config.qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.layers.1.mlp.experts.0.gate_proj.weight") == null);
    try testing.expect(qwen4_config.qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.layers.1.mlp.experts.0.gate_proj.weight_scale") == null);
    try testing.expect(qwen4_config.qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.mtp.layers.0.self_attn.qkv_proj.weight") == null);
}

test "mimo_v2 streaming leaves trunk keys intact and excludes only routed banks" {
    var buf: [256]u8 = undefined;
    for ([_][]const u8{
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
    }) |key| {
        try testing.expectEqualStrings(key, qwen4_config.qwen4StreamingWeightKey(.mxfp4_split, &buf, key).?);
    }
    try testing.expect(qwen4_config.qwen4StreamingWeightKey(.mxfp4_split, &buf, "model.layers.1.mlp.switch_mlp.gate_proj.weight") == null);
    try testing.expect(qwen4_config.qwen4StreamingWeightKey(.mxfp4_split, &buf, "model.layers.1.mlp.switch_mlp.down_proj.scales") == null);
}
