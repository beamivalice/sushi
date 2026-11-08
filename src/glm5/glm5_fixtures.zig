//! GLM-5.3-Flash's committed config and chat template, MLX-free, for its own tests and the shared ones
//! (`arch/chat.zig` `Fixtures`, the config pins), reached through its chat face.

/// The checkpoint's `config.json`.
pub const config = @embedFile("../fixtures/glm5_config.json");

/// GLM-5.3's own chat template.
pub const template = @embedFile("../fixtures/glm53_chat_template.jinja");

/// The configs the pins file names, one `{ section, json }` each.
pub const config_pins = [_][2][]const u8{
    .{ "glm5_config.json", config },
};

/// Pinned configs with one spelling swapped before the parse.
pub const config_variants = [_]struct { name: []const u8, json: []const u8, from: []const u8, to: []const u8 }{
    .{ .name = "glm5 text alias", .json = config, .from = "\n  \"model_type\": \"glm5_next\"", .to = "\n  \"model_type\": \"glm5_next_text\"" },
    .{ .name = "glm5 raw fp8", .json = config, .from = "{\n  \"architectures\"", .to = "{\"quantization_config\":{\"quant_method\":\"fp8\",\"weight_block_size\":[128,128]},\n  \"architectures\"" },
};
