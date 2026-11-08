//! MiMo-V2.6-Flash's committed configs, MLX-free, for its own tests and the shared ones (`arch_config.fixtureConfig`,
//! the config pins), reached through its chat face.

/// The checkpoint's `config.json`.
pub const config = @embedFile("fixtures/mimo_v2.json");

pub const MIMO_V2_VISION_JSON =
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

/// The mimo_v2 geometry of "ModelConfig parses mimo_v2 hybrid geometry", which the bill assertions read.
pub const MIMO_V2_BILL_JSON =
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

/// The configs the shared config pins parse (`fixtures/model-configs/parsed.pins`), by section name.
pub const config_pins = .{
    .{ "model-configs/mimo_v2.json", config },
    .{ "mimo vision", MIMO_V2_VISION_JSON },
    .{ "mimo bill", MIMO_V2_BILL_JSON },
};
