//! Flash-Next's committed configs and chat template, MLX-free, for its own tests and the shared ones
//! (`arch/chat.zig` `Fixtures`, the config pins), reached through its chat face.

/// The `config.json` cut from Qwen3.8-Flash-Next-Sushi-3bpw.
pub const config = @embedFile("../fixtures/model-configs/qwen4_exp.json");

/// The stock chat template, byte-matched by the late-system adapter.
pub const template = @embedFile("../fixtures/qwen38_flash_next_chat_template.jinja");

/// The wikitext2 teacher fixture of `sushi kld`, under the models root.
pub const teacher_pack = "kld-teacher/Qwen3.8-Flash-Next-wikitext2-60x64";

// Both documents below are the SHIPPED checkpoint's text config (Qwen3.8-Flash-Next, `model_type: qwen4_exp`): one as
// it ships (plain rope, 262144) and one with the YaRN block vLLM's `--hf-overrides` recipe writes. They stay literals
// so the parser is tested against the real file shape, braces and all, rather than a synthesized one.

/// The checkpoint as it ships: `rope_type: "default"`, a 262144 window.
pub const shipped =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 262144,
    \\    "rope_parameters": {
    \\      "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// The same checkpoint with its rope scaled 4x and the window widened, exactly
/// `vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {...}}}'
/// --max-model-len 1010000` expressed as config instead of a flag.
pub const yarn =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 1048576,
    \\    "rope_parameters": {
    \\      "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
    \\      "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// One qwen4_exp config document with `extra` fields spliced in.
pub fn caseJson(comptime extra: []const u8) []const u8 {
    return "{\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48," ++
        "\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"ple_conv_kernel_size\":4," ++
        "\"num_experts\":512,\"num_experts_per_tok\":10,\"moe_intermediate_size\":640," ++
        "\"eos_token_id\":248044,\"vocab_size\":248320,\"rms_norm_eps\":1e-6," ++
        extra ++ "}";
}

pub const good_fields =
    "\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8," ++
    "\"ngram_vocab_size_base\":20000000,\"make_ngram_vocab_size_divisible_by\":128," ++
    "\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048,\"indexer_compress_ratio\":4";

/// `caseJson` of the fields a valid config needs, then `extra` (empty, or starting with a comma).
pub fn goodJson(comptime extra: []const u8) []const u8 {
    return caseJson(good_fields ++ extra);
}

const trunk_quant_fields = ",\"trunk_quant\":{\"format\":\"stq\",\"k\":6,\"codebook\":\"spec1\",\"window\":15,\"codebook_table\":\"spec1-w15\",\"codebook_digest\":\"37f72431\"}";

/// The configs the pins file names, one `{ section, json }` each.
pub const config_pins = [_][2][]const u8{
    .{ "model-configs/qwen4_exp.json", config },
    .{ "qwen4 shipped", shipped },
    .{ "qwen4 yarn", yarn },
    .{ "qwen4 trunk_quant", goodJson(trunk_quant_fields) },
};

/// Pinned configs with one spelling swapped before the parse.
pub const config_variants = [_]struct { name: []const u8, json: []const u8, from: []const u8, to: []const u8 }{
    .{ .name = "qwen4 text alias", .json = shipped, .from = "  \"model_type\": \"qwen4_exp\",\n  \"text_config\"", .to = "  \"model_type\": \"qwen4_exp_text\",\n  \"text_config\"" },
};
