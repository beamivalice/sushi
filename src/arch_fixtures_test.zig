//! Arch data for tests of shared code, read through each arch's chat face (`arch/chat.zig` `Fixtures`). A value is
//! null (or empty) where the build left its arch out, so the test skips.
const arch_chat = @import("arch/chat.zig");
const chat = @import("chat.zig");
const model_mod = @import("model.zig");
const std = @import("std");

fn declOf(comptime name: []const u8, comptime decl: []const u8) ?[]const u8 {
    const fx = arch_chat.Fixtures(name);
    return if (@hasDecl(fx, decl)) @field(fx, decl) else null;
}

pub const qwen_template = declOf("qwen4", "template");
pub const qwen_model_type = arch_chat.modelTypeOf("qwen4");
pub const glm_model_type = arch_chat.modelTypeOf("glm5");
pub const mimo_template = declOf("mimo2", "template");
pub const glm_template = declOf("glm5", "template");

/// The Flash-Next teacher fixture of `sushi kld`, under the models root.
pub const qwen_teacher_pack = declOf("qwen4", "teacher_pack");

/// The stock templates the late-system adapter byte-matches.
pub const qwen_late_templates: []const []const u8 = if (qwen_template) |t| &.{t} else &.{};

/// Flash-Next's stock template with turns ending `<|im_end|>`, compiled as a load compiles it; the caller calls
/// `dropCompiled`.
pub fn qwenChatConfig(allocator: std.mem.Allocator) !?chat.ChatConfig {
    var cc: chat.ChatConfig = .{
        .chat_template = qwen_template orelse return null,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
        .model_type = qwen_model_type,
    };
    try cc.compile();
    return cc;
}

/// A Flash-Next-shaped model that counts a request's tokens: no layers, thinking declared off so only a test that
/// names a thinking default meets one.
pub fn qwenCountConfig() ?model_mod.ModelConfig {
    if (qwen_template == null) return null;
    return .{ .num_hidden_layers = 0, .pinned_context = 4096, .model_type = "qwen4_exp", .gen_enable_thinking = false };
}

/// How a template spells a call: its opener, what follows the name (the first of its tails), the gap between a thought
/// and the call, and whether the template leaves the think opener to the model.
pub const ForcedFamily = struct { tpl: []const u8, head: []const u8, name_tail: []const u8, sep: []const u8, leaves_opener: bool = false };

/// The families whose template a forced `tool_choice` test renders, for the archs the build holds.
pub const forced_families: []const ForcedFamily = blk: {
    var out: []const ForcedFamily = &.{};
    if (qwen_template) |t| out = out ++ &[_]ForcedFamily{.{ .tpl = t, .head = "<tool_call>\n<function=", .name_tail = ">\n", .sep = "\n\n" }};
    if (mimo_template) |t| out = out ++ &[_]ForcedFamily{.{ .tpl = t, .head = "<tool_call><function=", .name_tail = ">", .sep = "", .leaves_opener = true }};
    if (glm_template) |t| out = out ++ &[_]ForcedFamily{.{ .tpl = t, .head = "<tool_call>", .name_tail = "<arg_key>", .sep = "" }};
    break :blk out;
};
