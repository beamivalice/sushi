//! MiMo-V2.6's own chat template, rendered through the template engine: late system turns, tools and XML tool history.
const std = @import("std");
const chat = @import("../chat.zig");
const fixtures = @import("mimo_fixtures.zig");
const think_tags = @import("../think_tags.zig");
const ChatConfig = chat.ChatConfig;
const Message = chat.Message;
const ToolCall = chat.ToolCall;
const promptTailOpensThink = think_tags.promptTailOpensThink;
const renderChatTemplate = chat.renderChatTemplate;
const testing = std.testing;

fn mimoTemplateConfig(allocator: std.mem.Allocator) ChatConfig {
    return .{
        .chat_template = fixtures.template,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
}

test "mimo_v2 template renders a late system turn in place" {
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "system", .content = "late" },
        .{ .role = "user", .content = "again" },
    };
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqualStrings("<|im_start|>system\nS<|im_end|><|im_start|>user\nhi<|im_end|>" ++
        "<|im_start|>system\nlate<|im_end|><|im_start|>user\nagain<|im_end|><|im_start|>assistant\n", rendered);
}

test "mimo_v2 template renders a tools request as the reference does" {
    // Expected bytes: transformers' apply_chat_template environment (jinja2, its
    // json.dumps tojson) over the same template and request.
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const tools_json =
        \\[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    ;
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .reasoning_content = "Need weather.", .tool_calls = &calls },
        .{ .role = "tool", .content = "18C", .tool_call_id = "call_1" },
        .{ .role = "user", .content = "Thanks." },
    };
    const expected = "<|im_start|>system\nYou are provided with the following tools:\n\n<tools>\n" ++
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Current weather\", " ++
        "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}" ++
        "\n</tools><|im_end|><|im_start|>system\nS<|im_end|><|im_start|>user\nWeather in Paris?<|im_end|>" ++
        "<|im_start|>assistant\n<think>Need weather.</think><tool_call><function=get_weather><parameter=city>Paris</parameter>" ++
        "</function></tool_call><|im_end|><|im_start|>tool\n18C<|im_end|><|im_start|>user\nThanks.<|im_end|>" ++
        "<|im_start|>assistant\n";
    const thinking = try renderChatTemplate(a, &messages, &config, tools_json, null, true, null, false);
    defer a.free(thinking);
    try testing.expectEqualStrings(expected, thinking);
    const plain = try renderChatTemplate(a, &messages, &config, tools_json, null, false, null, false);
    defer a.free(plain);
    try testing.expectEqualStrings(expected ++ "<think></think>", plain);
}

test "mimo_v2 template preserves reasoning and XML tool history" {
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "sum", .arguments = "{\"x\":2}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Compute." },
        .{ .role = "assistant", .content = "", .reasoning_content = "Use sum.", .tool_calls = &calls },
        .{ .role = "tool", .content = "2" },
        .{ .role = "user", .content = "Continue." },
    };
    const expected = "<|im_start|>user\nCompute.<|im_end|>" ++
        "<|im_start|>assistant\n<think>Use sum.</think>" ++
        "<tool_call><function=sum><parameter=x>2</parameter></function></tool_call><|im_end|>" ++
        "<|im_start|>tool\n2<|im_end|>" ++
        "<|im_start|>user\nContinue.<|im_end|><|im_start|>assistant\n";
    const thinking = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(thinking);
    try testing.expectEqualStrings(expected, thinking);
    try testing.expect(!promptTailOpensThink(thinking));
    const plain = try renderChatTemplate(a, &messages, &config, null, null, false, null, false);
    defer a.free(plain);
    try testing.expectEqualStrings(expected ++ "<think></think>", plain);
}

test "real mimo_v2 template is the committed fixture" {
    const raw = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    var config = try chat.loadChatConfig(testing.io, testing.allocator, std.mem.span(raw));
    defer config.deinit();
    try testing.expectEqualStrings(fixtures.template, config.chat_template);
}
