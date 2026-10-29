const std = @import("std");
const expectError = @import("../test_expect.zig").expectError;
const expert_quant = @import("../expert_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const ModelConfig = model_mod.ModelConfig;
const QWEN4_GOOD_FIELDS = model_mod.QWEN4_GOOD_FIELDS;
const QWEN4_SHIPPED = model_mod.QWEN4_SHIPPED;
const QWEN4_YARN = model_mod.QWEN4_YARN;
const cfgF32 = model_mod.cfgF32;
const cfgField = model_mod.cfgField;
const cfgInt = model_mod.cfgInt;
const loadWeightsStreaming = model_mod.loadWeightsStreaming;
const parseConfigFromJson = model_mod.parseConfigFromJson;
const qwen4CaseJson = model_mod.qwen4CaseJson;
const setConfigOverrides = model_mod.setConfigOverrides;
const streamingResidentSplit = model_mod.streamingResidentSplit;
const testing = std.testing;

/// One qwen4_exp integer bound, read strictly: wrong-typed or negative refuses.
pub fn qwen4ConfigU64(cfg_obj: std.json.ObjectMap, key: []const u8) !?u64 {
    const v = cfg_obj.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

pub fn qwen4ConfigU32(cfg_obj: std.json.ObjectMap, key: []const u8) !?u32 {
    const v = try qwen4ConfigU64(cfg_obj, key) orelse return null;
    if (v > std.math.maxInt(u32)) return error.InvalidQwen4ConfigField;
    return @intCast(v);
}

/// Range-check every qwen4_exp bound the forward indexes a fixed array with or divides by.
/// Names travel to the client as "Model load failed: <name>".
pub fn validateQwen4Config(config: *const ModelConfig) !void {
    if (config.hidden_size == 0 or config.head_dim == 0 or config.full_attention_interval == 0 or
        config.num_attention_heads == 0 or config.num_key_value_heads == 0 or
        config.num_attention_heads % config.num_key_value_heads != 0 or
        config.num_experts_per_tok == 0 or config.num_experts_per_tok > config.num_experts)
    {
        return error.InvalidQwen4Geometry;
    }
    // `NgramHash.multipliers` is [MAX_NGRAM_SIZE]i64; `ple_prev` is written ngram_size-1 deep.
    if (config.ngram_size < 2 or config.ngram_size > qwen4_exp.MAX_NGRAM_SIZE) {
        return error.InvalidQwen4NgramSize;
    }
    // `vocab`/`offsets` are [MAX_HEADS]i64, written n_heads deep.
    if (config.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (config.heads_per_ngram > qwen4_exp.MAX_HEADS / (config.ngram_size - 1)) {
        return error.InvalidQwen4NgramHeads;
    }
    if (config.ngram_vocab_divisor == 0 or config.ngram_vocab_base < 2) {
        return error.InvalidQwen4NgramVocab;
    }
    // The forward divides kv by the ratio and selects `budget / ratio` blocks.
    if (config.indexer_n_heads > 0) {
        if (config.indexer_head_dim == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_budget < config.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    if (config.ple_layer_idx < 0 or config.ple_layer_idx >= @as(i32, @intCast(config.num_hidden_layers))) {
        return error.InvalidQwen4PleLayer;
    }
}

/// True when the layer loop installed the PLE on exactly the layer the config names. A negative
/// index asks for no PLE (the MTP head's own layer) and is satisfied by a loop that installed none.
pub fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
/// Qwen3-VL-family vision + M-RoPE fields of the qwen4_exp arm (`vision_config` keys, `rope_parameters.mrope_*`,
/// vision token ids). The generic vision_config block already set
/// `has_vision`; this reads Qwen's own keys into `qv_*`.
pub fn parseQwenVisionFields(config: *ModelConfig, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    if (root.get("vision_config")) |vc_val| {
        if (vc_val == .object) {
            const vc = vc_val.object;
            config.has_vision = true;
            config.qwen_vision = true;
            if (vc.get("depth")) |v| {
                if (v == .integer) config.qv_depth = try cfgInt(u32, v);
            }
            if (vc.get("hidden_size")) |v| {
                if (v == .integer) config.qv_hidden = try cfgInt(u32, v);
            }
            if (vc.get("num_heads")) |v| {
                if (v == .integer) config.qv_heads = try cfgInt(u32, v);
            }
            if (vc.get("intermediate_size")) |v| {
                if (v == .integer) config.qv_intermediate = try cfgInt(u32, v);
            }
            if (vc.get("patch_size")) |v| {
                if (v == .integer) config.qv_patch = try cfgInt(u32, v);
            }
            if (vc.get("temporal_patch_size")) |v| {
                if (v == .integer) config.qv_temporal_patch = try cfgInt(u32, v);
            }
            if (vc.get("spatial_merge_size")) |v| {
                if (v == .integer) config.qv_merge = try cfgInt(u32, v);
            }
            if (vc.get("num_position_embeddings")) |v| {
                if (v == .integer) config.qv_num_pos_emb = try cfgInt(u32, v);
            }
            if (vc.get("out_hidden_size")) |v| {
                if (v == .integer) config.qv_out_hidden = try cfgInt(u32, v);
            }
            if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
            if (config.qv_out_hidden == 0) config.qv_out_hidden = config.hidden_size;
        }
    }
    // Interleaved M-RoPE sections (text_config.rope_parameters). rope_theta /
    // partial_rotary_factor already parsed in the generic rope block above.
    if (cfg_obj.get("rope_parameters")) |rp| {
        if (rp == .object) {
            if (rp.object.get("mrope_interleaved")) |v| {
                if (v == .bool) config.mrope_interleaved = v.bool;
            }
            if (rp.object.get("mrope_section")) |v| {
                if (v == .array) {
                    for (v.array.items, 0..) |item, i| {
                        if (i >= 3) break;
                        if (item == .integer) config.mrope_section[i] = try cfgInt(u32, item);
                    }
                }
            }
        }
    }
    // Qwen vision token ids (top-level).
    if (root.get("video_token_id")) |v| {
        if (v == .integer) config.video_token_id = try cfgInt(u32, v);
    }
    if (root.get("vision_start_token_id")) |v| {
        if (v == .integer) config.vision_start_token_id = try cfgInt(u32, v);
    }
    if (root.get("vision_end_token_id")) |v| {
        if (v == .integer) config.vision_end_token_id = try cfgInt(u32, v);
    }
}

/// Flat HF `rope_parameters` carrying `rope_type: "yarn"` — the YaRN context
/// extension, i.e. exactly what vLLM's `--hf-overrides` recipe for Qwen
/// writes:
///
///   {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
///     "original_max_position_embeddings": 262144, "rope_theta": 10000000,
///     "partial_rotary_factor": 0.25, "mrope_interleaved": true,
///     "mrope_section": [11,11,10]}}}
///
/// `factor` may be omitted, in which case HF derives it from
/// `max_position_embeddings / original_max_position_embeddings` (as vLLM's
/// `_get_and_verify_max_len` does). `attention_factor` is HF's key and
/// REPLACES the computed mscale; `attn_factor` is vLLM's and MULTIPLIES
/// `yarnMscale(factor)`. Neither is present in a vendor config, and per HF's
/// default the mscale is then COMPUTED as 0.1·ln(factor)+1 — the value the
/// scaling was calibrated with. Nested per-layer-type `rope_parameters`
/// never reach here: they have no top-level `rope_type`.
pub fn parseYarnRopeParameters(config: *ModelConfig, cfg_obj: std.json.ObjectMap) !void {
    const rp_val = cfg_obj.get("rope_parameters") orelse return;
    if (rp_val != .object) return;
    const rp = rp_val.object;
    const rt = rp.get("rope_type") orelse return;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "yarn"))) return;

    if (rp.get("original_max_position_embeddings")) |v| {
        if (v == .integer) config.yarn_orig_max_pos = try cfgInt(u32, v);
    }
    // A YaRN block with no window to scale FROM is not a scaling we can
    // reproduce: the ramp bounds (and so every mid-band frequency) come from
    // it. Refuse the load rather than serve a silently-wrong rotation.
    if (config.yarn_orig_max_pos == 0) return error.YarnRopeNeedsOriginalMaxPos;
    if (cfgField(rp, "factor")) |v| config.yarn_factor = try cfgF32(v);
    if (cfgField(rp, "beta_fast")) |v| config.yarn_beta_fast = try cfgF32(v);
    if (cfgField(rp, "beta_slow")) |v| config.yarn_beta_slow = try cfgF32(v);
    if (rp.get("truncate")) |v| {
        if (v == .bool) config.yarn_truncate = v.bool;
    }
    if (config.yarn_factor <= 0.0) return error.InvalidRopeScalingFactor;
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block leaves it out (the config then only states the window).
    if (cfgField(rp, "factor") == null and config.max_position_embeddings > config.yarn_orig_max_pos) {
        config.yarn_factor = @as(f32, @floatFromInt(config.max_position_embeddings)) /
            @as(f32, @floatFromInt(config.yarn_orig_max_pos));
    }
    // HF `attention_factor` replaces; vLLM `attn_factor` multiplies the
    // computed 0.1·ln(factor)+1. Both present → HF wins.
    if (cfgField(rp, "attention_factor")) |v| {
        config.yarn_attention_factor = try cfgF32(v);
    } else if (cfgField(rp, "attn_factor")) |v| {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor) * try cfgF32(v);
    } else {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor);
    }
    config.rope_yarn = true;
}

/// HF's default YaRN mscale (`attention_factor`) for a scaling `factor`.
fn yarnMscale(factor: f32) f32 {
    if (factor <= 1.0) return 1.0;
    return 0.1 * @log(@as(f32, factor)) + 1.0;
}

pub fn qwen4StreamingWeightKey(layout: expert_quant.Layout, buf: []u8, key: []const u8) ?[]const u8 {
    if (expert_quant.isRoutedExpertKey(layout, key)) return null;
    if (layout == .mxfp4_split or layout == .mxfp4_individual) {
        if (std.mem.startsWith(u8, key, "mtp.") or std.mem.startsWith(u8, key, "model.mtp.")) return null;
        return key;
    }
    if (layout == .quantized_split or layout == .exl3_k4) return key;
    const trunk_prefix = "model.language_model.";
    if (std.mem.indexOf(u8, key, ".ple.ple_embedding.ngram_embedding.shard_") != null) return null;
    if (std.mem.startsWith(u8, key, trunk_prefix)) {
        return std.fmt.bufPrint(buf, "language_model.model.{s}", .{key[trunk_prefix.len..]}) catch null;
    }
    if (std.mem.startsWith(u8, key, "mtp.")) {
        return std.fmt.bufPrint(buf, "language_model.mtp.{s}", .{key[4..]}) catch null;
    }
    if (std.mem.eql(u8, key, "lm_head.weight")) return "language_model.lm_head.weight";
    return key;
}

pub fn qwen4NormFold(key: []const u8) bool {
    const suffixes = [_][]const u8{
        "hc_norm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "ple.norm_key.weight",
        "ple.norm_query.weight",
        "ple.norm_conv.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

pub fn qwen4FoldNorm(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one_f32 = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f32);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_astype(&one, one_f32, mlx.mlx_array_dtype(value), s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, value, one, s));
    return result;
}

fn qwen4ConvShape(shape: []const c_int) !void {
    if (shape.len != 3 or shape[1] != 1) return error.InvalidQwen4ConvShape;
}

pub fn qwen4TransposeConv(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    try qwen4ConvShape(mlx.getShape(value));
    const axes = [_]c_int{ 0, 2, 1 };
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_transpose_axes(&view, value, &axes, 3, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&result, view, false, s));
    return result;
}

pub fn qwen4SplitGateUp(value: mlx.mlx_array, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const shape = mlx.getShape(value);
    if (shape.len != 3 or shape[1] == 0 or @mod(shape[1], 2) != 0) return error.BadPackedGateUpShape;
    const half = @divExact(shape[1], 2);
    const strides = [_]c_int{ 1, 1, 1 };
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer {
        for (result) |arr| _ = mlx.mlx_array_free(arr);
    }
    for (0..2) |i| {
        const start = [_]c_int{ 0, @intCast(i * @as(usize, @intCast(half))), 0 };
        const stop = [_]c_int{ shape[0], @intCast((i + 1) * @as(usize, @intCast(half))), shape[2] };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, value, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_contiguous(&result[i], view, false, s));
    }
    return result;
}

test "qwen4 convolution transform rejects a non-unit channel axis" {
    try std.testing.expectError(error.InvalidQwen4ConvShape, qwen4ConvShape(&.{ 4, 2, 8 }));
    try qwen4ConvShape(&.{ 4, 1, 8 });
}

test "parseConfigFromJson: qwen4_exp YaRN rope_parameters extends 262144 to 1048576" {
    const c = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expect(c.isQwen4());
    // The scaling is recognised and lands where the engine reads it —
    // transformer.yarnSpec() consumes exactly these fields.
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);
    try testing.expectEqual(@as(u32, 262_144), c.yarn_orig_max_pos);
    try testing.expectApproxEqAbs(@as(f32, 32.0), c.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_beta_slow, 1e-9);
    try testing.expect(c.yarn_truncate); // HF's default, absent from the block
    // The mscale is COMPUTED (no `attention_factor` in the block, so HF's
    // default applies): 0.1·ln 4 + 1 — the value the extension is calibrated to.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), c.yarn_attention_factor, 1e-6);
    // qwen4_exp has ONE rope for the trunk, so the YaRN table spans exactly the
    // 64 dims attention rotates: `partial_rotary_factor`, not the full head dim.
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.yarnPartial(), 1e-9);
    try testing.expectEqual(@as(u32, 64), c.yarnRotaryDim());
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), c.rope_theta, 1.0);
    // The 32 frequencies of the scaled table are the 32 halves the interleaved
    // M-RoPE selector splits [11,11,10] across. If they disagreed, half the
    // table would rotate against an axis the position table doesn't have.
    try testing.expectEqual(
        c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2],
        c.yarnRotaryDim() / 2,
    );
    // The window the server may advertise: original × factor, and the config's
    // own declaration agrees.
    try testing.expectEqual(@as(u32, 1_048_576), c.max_position_embeddings);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Cost of that window: only the 12 interval-full layers bill KV, so
    // 12 layers × 2 kv heads × (K+V) × 256 dims × 2 bytes = 24 KiB per token.
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 24_576), c.kvBytesPerToken());
}

test "parseConfigFromJson: qwen4_exp (Qwen3.8-Flash-Next) reads the hyper-connection, PLE, QSA and text-config eos fields" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "hc_count":4,"hc_lowrank":320,"ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,
        \\ "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,
        \\ "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,"output_gate_type":"sigmoid",
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4());
    try testing.expectEqualStrings("language_model.model", c.weight_prefix);
    try testing.expectEqual(@as(u32, 4), c.hc_count);
    try testing.expectEqual(@as(u32, 320), c.hc_lowrank);
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx); // 1-based [2] → layer 1
    try testing.expectEqual(@as(u32, 2560), c.ple_embed_dim);
    try testing.expectEqual(@as(u32, 4), c.indexer_n_heads);
    try testing.expectEqual(@as(u32, 2048), c.indexer_budget);
    try testing.expectEqual(@as(u32, 4), c.indexer_compress_ratio);
    try testing.expectEqual(@as(u32, 248044), c.ngram_eos);
    try testing.expectEqual(@as(u32, 4), c.full_attention_interval);
    try testing.expect(c.isLinearLayer(0) and !c.isLinearLayer(3));
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 12 * 128 * 2 / 4), c.qsaHistoryBytesPerToken());
    try testing.expectEqual(@as(u64, 12 * @as(u64, @intCast(@import("../transformer.zig").QSA_RING_ROWS)) * 128 * 2), c.qsaRingBytes());
    try testing.expect(c.attn_output_gate and c.kda_sigmoid_out_gate and !c.has_final_norm and !c.norm_has_offset);
    try testing.expect(c.isMoe() and c.supportsBatchedGdnDecode()); // per-slot state on the SSMCacheEntry: batches
    try testing.expectEqual(@as(f32, 0.25), c.partial_rotary_factor);
    try testing.expectEqual(@as(f32, 10000000.0), c.rope_theta);
    try testing.expect(!c.qwen_vision and !c.has_vision);
}

test "parseConfigFromJson accepts dense qwen4 as an expert streaming architecture" {
    const json =
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    const config = try parseConfigFromJson(std.testing.allocator, json);
    try std.testing.expectEqual(@as(u32, 0), config.quant_bits);
    try std.testing.expect(config.supportsExpertStreaming());
}

test "expert streaming is a capability of every qwen4 pack and required only by the dense one" {
    const t = std.testing;
    const dense =
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    const quantized =
        \\{"model_type":"qwen4_exp","quantization":{"group_size":64,"bits":4,"mode":"affine"},"text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    var dense_config = try parseConfigFromJson(t.allocator, dense);
    defer dense_config.deinit(t.allocator);
    try t.expect(dense_config.supportsExpertStreaming() and dense_config.expertStreamingRequired());
    var quant_config = try parseConfigFromJson(t.allocator, quantized);
    defer quant_config.deinit(t.allocator);
    try t.expectEqual(@as(u32, 4), quant_config.quant_bits);
    try t.expect(quant_config.supportsExpertStreaming() and !quant_config.expertStreamingRequired());
    var other = ModelConfig{ .model_type = "not_a_served_arch", .num_hidden_layers = 48, .num_experts = 512, .num_experts_per_tok = 10, .hidden_size = 2560, .moe_intermediate_size = 640 };
    try t.expect(!other.supportsExpertStreaming() and !other.expertStreamingRequired());
}

test "qwen4 streaming loader canonicalizes resident keys and excludes disk tensors" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.7.mlp.gate.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.gate.weight").?);
    try std.testing.expectEqualStrings("language_model.mtp.fc_hidden.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "mtp.fc_hidden.weight").?);
    try std.testing.expectEqualStrings("language_model.lm_head.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "lm_head.weight").?);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.experts.gate_up_proj") == null);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.experts.down_proj") == null);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_12.weight") == null);
}

test "the quantized streaming loader drops the nine routed banks and keeps everything else" {
    var buf: [256]u8 = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            var key_buf: [192]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "language_model.model.layers.7.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part });
            try std.testing.expect(qwen4StreamingWeightKey(.quantized_split, &buf, key) == null);
        }
    }
    try std.testing.expectEqualStrings(
        "language_model.model.layers.7.mlp.shared_expert.gate_proj.scales",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.model.layers.7.mlp.shared_expert.gate_proj.scales").?,
    );
    try std.testing.expectEqualStrings(
        "language_model.model.layers.7.mlp.gate.weight",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.model.layers.7.mlp.gate.weight").?,
    );
    try std.testing.expectEqualStrings(
        "language_model.lm_head.weight",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.lm_head.weight").?,
    );
}

test "qwen4 streaming loader materializes only transformed resident tensors" {
    const t = std.testing;
    const io = t.io;
    const allocator = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..path_len];
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[0,4]},\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[4,8]},\"model.language_model.layers.0.mlp.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[8,10]},\"model.language_model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[10,12]},\"model.language_model.layers.0.linear_attn.conv1d.weight\":{\"dtype\":\"BF16\",\"shape\":[1,1,2],\"data_offsets\":[12,16]},\"mtp.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[16,20]},\"mtp.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[1,1,1],\"data_offsets\":[20,22]}}";
    const padded_header_len = std.mem.alignForward(usize, header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + padded_header_len + 22);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], padded_header_len, .little);
    @memset(file_bytes[8 .. 8 + padded_header_len], ' ');
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const tensor_data = [_]u16{ 0x3f80, 0x4000, 0x3f80, 0x4000, 0x3f80, 0, 0x3f80, 0x4000, 0x3f80, 0x4000, 0x3f80 };
    @memcpy(file_bytes[8 + padded_header_len ..], std.mem.sliceAsBytes(&tensor_data));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });

    var weights = try loadWeightsStreaming(io, allocator, model_dir, .bf16_fused, false, false);
    defer weights.deinit();
    try t.expect(weights.get("model.language_model.layers.0.mlp.experts.gate_up_proj") == null);
    try t.expect(weights.get("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight") == null);
    try t.expect(weights.get("language_model.model.layers.0.mlp.gate.weight") != null);
    const norm = weights.get("language_model.model.layers.0.self_attn.q_norm.weight").?;
    try t.expectEqualSlices(c_int, &.{1}, mlx.getShape(norm));
    try t.expectEqualSlices(c_int, &.{ 1, 2, 1 }, mlx.getShape(weights.get("language_model.model.layers.0.linear_attn.conv1d.weight").?));
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight") == null);
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight") == null);
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight") == null);
}

test "qwen4 streaming resident byte estimate excludes experts PLE and vision" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,2],\"data_offsets\":[0,8]},\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":{\"dtype\":\"BF16\",\"shape\":[1,2],\"data_offsets\":[8,12]},\"model.visual.x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[12,14]},\"model.language_model.layers.0.mlp.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[1,3],\"data_offsets\":[14,20]},\"mtp.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[1,3,2],\"data_offsets\":[20,32]}}";
    const bytes = try t.allocator.alloc(u8, 8 + header.len + 32);
    defer t.allocator.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "s.safetensors", .data = bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"s.safetensors\",\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":\"s.safetensors\",\"model.visual.x\":\"s.safetensors\",\"model.language_model.layers.0.mlp.gate.weight\":\"s.safetensors\",\"mtp.layers.0.mlp.experts.down_proj\":\"s.safetensors\"}}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const split = try streamingResidentSplit(io, t.allocator, path_buf[0..path_len], &.{ .expert_layout = .bf16_fused });
    try t.expectEqual(@as(u64, 6), split.trunk);
    try t.expectEqual(@as(u64, 12), split.mtp);
}

test "parseConfigFromJson: qwen4_exp with vision_config reads the Qwen3-VL tower, M-RoPE and vision token ids" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "image_token_id":248056,"video_token_id":248057,"vision_start_token_id":248053,"vision_end_token_id":248054,
        \\ "vision_config":{"depth":27,"hidden_size":1152,"num_heads":16,"intermediate_size":4304,"patch_size":16,
        \\   "temporal_patch_size":2,"spatial_merge_size":2,"num_position_embeddings":2304,"out_hidden_size":2560,"model_type":"qwen4_exp_vision"},
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "ple_layer_ids":[2],"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4() and c.has_vision and c.qwen_vision);
    try testing.expectEqual(@as(u32, 27), c.qv_depth);
    try testing.expectEqual(@as(u32, 1152), c.qv_hidden);
    try testing.expectEqual(@as(u32, 16), c.qv_heads);
    try testing.expectEqual(@as(u32, 72), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 4304), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 16), c.qv_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_merge);
    try testing.expectEqual(@as(u32, 2304), c.qv_num_pos_emb);
    try testing.expectEqual(@as(u32, 2560), c.qv_out_hidden);
    try testing.expect(c.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, c.mrope_section);
    try testing.expectEqual(@as(u32, 248056), c.image_token_id);
    try testing.expectEqual(@as(u32, 248057), c.video_token_id);
    try testing.expectEqual(@as(u32, 248053), c.vision_start_token_id);
    try testing.expectEqual(@as(u32, 248054), c.vision_end_token_id);
}

test "parseConfigFromJson: the shipped (unscaled) qwen4_exp config is untouched" {
    // The regression guard for every checkpoint that predates the extension:
    // no YaRN, and `contextCap` is just max_position_embeddings, so no server
    // sizing path can shift for a model that did not ask to be scaled.
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.isQwen4());
    try testing.expect(!c.rope_yarn);
    try testing.expectEqual(@as(f32, 1.0), c.yarn_factor);
    try testing.expectEqual(@as(u32, 262_144), c.max_position_embeddings);
    try testing.expectEqual(c.max_position_embeddings, c.contextCap());
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_attention_factor, 1e-9);
    // Same geometry otherwise — YaRN is a rotation, not an architecture change.
    const y = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(c.kvBytesPerToken(), y.kvBytesPerToken());
    try testing.expectEqual(c.num_hidden_layers, y.num_hidden_layers);
    try testing.expectEqual(c.head_dim, y.head_dim);
    try testing.expectEqual(c.mrope_section, y.mrope_section);
}

test "parseConfigFromJson: YaRN reads beta_fast/beta_slow/truncate and honours a pinned mscale" {
    defer setConfigOverrides(null);
    // HF's `attention_factor` REPLACES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attention_factor":1.25}}}
    );
    const pinned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(pinned.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 1.25), pinned.yarn_attention_factor, 1e-9);
    // Still reads theta/partial from the merged block (the base config's values).
    try testing.expectApproxEqAbs(@as(f32, 0.25), pinned.yarnPartial(), 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), pinned.rope_theta, 1.0);

    // vLLM's `attn_factor` MULTIPLIES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attn_factor":0.5}}}
    );
    const vl = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 0.5 * 1.138629436111989), vl.yarn_attention_factor, 1e-6);

    // Both keys present: HF's attention_factor wins (replace, not multiply).
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "attention_factor":1.25,"attn_factor":0.5}}}
    );
    const both = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 1.25), both.yarn_attention_factor, 1e-9);

    // The ramp knobs are read too — they move the blend, and so every frequency
    // between the bands.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "beta_fast":16,"beta_slow":2,"truncate":false}}}
    );
    const tuned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 16.0), tuned.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 2.0), tuned.yarn_beta_slow, 1e-9);
    try testing.expect(!tuned.yarn_truncate);
    // With no pinned mscale, the computed default returns.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), tuned.yarn_attention_factor, 1e-6);
}

test "parseConfigFromJson: YaRN derives factor from the window when the block omits it (HF)" {
    defer setConfigOverrides(null);
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block names only the window. Here the override widens the
    // declared window to 2M out of 262144 → factor 8, mscale 0.1·ln 8 + 1.
    setConfigOverrides(
        \\{"text_config":{"max_position_embeddings":2097152,
        \\  "rope_parameters":{"rope_type":"yarn","original_max_position_embeddings":262144}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 8.0), c.yarn_factor, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.2079441541679836), c.yarn_attention_factor, 1e-6);
    try testing.expectEqual(@as(u32, 2_097_152), c.contextCap());
}

test "parseConfigFromJson: YaRN with no pre-trained window, or a zero factor, fails the load" {
    defer setConfigOverrides(null);
    // The ramp bounds come from `original_max_position_embeddings`. Without it
    // every blended frequency is a guess — refuse the load rather than serve a
    // rope that looks fine at short contexts and decays beyond the window.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0}}}
    );
    try expectError(
        error.YarnRopeNeedsOriginalMaxPos,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
    // A zero factor is not "no scaling", it is a divide-by-zero waiting to run.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":0.0,
        \\  "original_max_position_embeddings":262144}}}
    );
    try expectError(
        error.InvalidRopeScalingFactor,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
}

test "qwen4_exp config: an n-gram bound past the fixed arrays is a named load error" {
    const good = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(u32, 3), good.ngram_size);
    try testing.expectEqual(@as(u32, 8), good.heads_per_ngram);

    try expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":9,\"heads_per_ngram\":8"),
    ));
    try expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":1,\"heads_per_ngram\":8"),
    ));
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":0"),
    ));
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":5,\"heads_per_ngram\":16"),
    ));
    try expectError(error.InvalidQwen4NgramVocab, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"make_ngram_vocab_size_divisible_by\":0"),
    ));
}

test "n-gram head count overflow is refused by the config" {
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":2147483656"),
    ));
}

test "qwen4_exp config: a wrong-typed or negative bound is a refusal, never a silent default" {
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":-1"),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":\"3\""),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"heads_per_ngram\":3.5"),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_compress_ratio\":-4"),
    ));
}

test "qwen4_exp config: an armed QSA indexer must carry a usable budget and ratio" {
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048"),
    ));
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2,\"indexer_compress_ratio\":4"),
    ));
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_budget\":2048,\"indexer_compress_ratio\":4"),
    ));
    const dense = try parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8"),
    );
    try testing.expectEqual(@as(u32, 0), dense.indexer_n_heads);
    try testing.expectEqual(@as(u32, 0), dense.indexer_compress_ratio);
}

test "qwen4_exp config: the PLE layer id must name exactly one layer that exists" {
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ngram_size\":3"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[0]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[49]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2,5]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":2"),
    ));
    const c = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx);
}

test "qwen4 PLE placement: the layer loop must install exactly one PLE, at the configured layer" {
    try testing.expect(qwen4PleInstalledAt(&.{ false, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, false, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ true, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 2));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 4));
    // A negative index is a build that asks for no PLE (`loadQwen4Mtp` sets -1 for the head's layer).
    try testing.expect(qwen4PleInstalledAt(&.{false}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{true}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, -1));
    try testing.expect(qwen4PleInstalledAt(&.{ false, false }, -1));
    // ...while a config that DOES name a layer is unchanged.
    try testing.expect(!qwen4PleInstalledAt(&.{false}, 0));
    try testing.expect(qwen4PleInstalledAt(&.{true}, 0));
}

test "the n-gram table source: the bf16 override outranks streaming, streaming outranks the pack table" {
    var c = ModelConfig{ .model_type = "qwen4_exp", .weight_prefix = "language_model.model" };
    try std.testing.expectEqual(ModelConfig.NgramTableSource.quantized, c.ngramTableSource());
    c.expert_streaming = true;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_streamed, c.ngramTableSource());
    c.expert_layout = .quantized_split;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.quantized, c.ngramTableSource());
    c.expert_layout = .bf16_fused;
    var dir = [_]u8{ '/', 'x' };
    c.ngram_bf16_dir = dir[0..];
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_override, c.ngramTableSource());
    c.expert_streaming = false;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_override, c.ngramTableSource());
}

test "qwen4_exp config: a geometry the forward divides by or indexes with is a named error" {
    const t = testing;
    const base = "\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"moe_intermediate_size\":640," ++
        "\"vocab_size\":248320," ++ QWEN4_GOOD_FIELDS;
    for ([_][]const u8{
        "{" ++ base ++ ",\"full_attention_interval\":0,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":0,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":5,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"num_experts\":8,\"num_experts_per_tok\":10}",
    }) |doc| {
        try expectError(error.InvalidQwen4Geometry, parseConfigFromJson(t.allocator, doc));
    }
}

test "qwen4PleInstalledAt accepts the MTP head's deliberate -1 and still refuses a stray PLE" {
    try testing.expect(qwen4PleInstalledAt(&.{false}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{true}, -1));
}
