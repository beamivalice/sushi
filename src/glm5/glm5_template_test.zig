//! GLM-5.3's own chat template, rendered through jinja.cpp: tool round trips and tool results in their list forms.
const std = @import("std");
const chat = @import("../chat.zig");
const fixtures = @import("glm5_fixtures.zig");
const jinja_c = @import("jinja_c");
const ChatConfig = chat.ChatConfig;
const Message = chat.Message;
const ToolCall = chat.ToolCall;
const renderChatTemplate = chat.renderChatTemplate;
const template_fallbacks = &chat.template_fallbacks;
const testing = std.testing;

fn templateConfig(allocator: std.mem.Allocator) ChatConfig {
    return .{ .chat_template = fixtures.template, .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = allocator };
}

test "GLM-5.3 template renders a tool round-trip itself, never the generic fallback" {
    const a = testing.allocator;
    var config = templateConfig(a);
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    ;
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What is the weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "Need the weather." },
        .{ .role = "tool", .content = "Sunny, 21C", .tool_call_id = "call_1" },
    };
    const want = "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>\n# Tools\n\n" ++
        "You may call one or more functions to assist with the user query.\n\n" ++
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" ++
        "{\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n" ++
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n" ++
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>" ++
        "<|system|>You are helpful.<|user|>What is the weather in Paris?" ++
        "<|assistant|><think>Need the weather.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
        "<|observation|><tool_response>Sunny, 21C</tool_response><|assistant|><think>";
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(a, &messages, &config, tools, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqual(before, template_fallbacks.load(.monotonic));
    try testing.expectEqualStrings(want, rendered);
}

/// Renders `messages_json` through GLM-5.3's own template; the text from the first user turn on.
fn renderRaw(messages_json: [:0]const u8, tools_json: ?[:0]const u8, out: *std.ArrayList(u8)) !void {
    const extra = "{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}";
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(fixtures.template, messages_json, if (tools_json) |t| t.ptr else null, extra, 1, &len) orelse {
        std.debug.print("\njinja error: {s}\n", .{std.mem.span(jinja_c.jinja_last_error().?)});
        return error.RenderFailed;
    };
    defer jinja_c.jinja_str_free(ptr);
    const text = ptr[0..len];
    try out.appendSlice(testing.allocator, text[std.mem.indexOf(u8, text, "<|user|>").?..]);
}

test "GLM-5.3 template: a tool result given as a list of outputs, text parts or tool references renders in place" {
    const user = "{\"role\":\"user\",\"content\":\"Q\"}";
    const call = "{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"R\",\"tool_calls\":[" ++
        "{\"type\":\"function\",\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}}," ++
        "{\"type\":\"function\",\"id\":\"call_2\",\"function\":{\"name\":\"get_time\",\"arguments\":{}}}]}";
    const head = "<|user|>Q<|assistant|><think>R</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
        "<tool_call>get_time</tool_call><|observation|>";
    const tail = "<|assistant|><think>";
    const tools =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{}}}}]
    ;
    const cases = [_]struct { tool: [:0]const u8, tools: ?[:0]const u8, want: []const u8 }{
        .{
            .tool = "{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\",\"output\":\"12:00\"},{\"tool_call_id\":\"call_1\",\"output\":\"Sunny\"}]}",
            .tools = null,
            .want = head ++ "<tool_response>Sunny</tool_response><tool_response>12:00</tool_response>" ++ tail,
        },
        .{
            .tool = "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":[{\"type\":\"text\",\"text\":\"Sunny\"},{\"type\":\"image\"}]}",
            .tools = null,
            .want = head ++ "<tool_response>Sunny<|begin_of_image|><|image|><|end_of_image|></tool_response>" ++ tail,
        },
        .{
            .tool = "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":[{\"type\":\"tool_reference\",\"name\":\"get_time\"}]}",
            .tools = tools,
            .want = head ++ "<tool_response><tools>\n{\"name\": \"get_time\", \"description\": \"Get time\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}\n</tools></tool_response>" ++ tail,
        },
    };
    for (cases) |c| {
        const json = try std.fmt.allocPrintSentinel(testing.allocator, "[{s},{s},{s}]", .{ user, call, c.tool }, 0);
        defer testing.allocator.free(json);
        var out = std.ArrayList(u8).empty;
        defer out.deinit(testing.allocator);
        try renderRaw(json, c.tools, &out);
        try testing.expectEqualStrings(c.want, out.items);
    }
}

test "GLM-5.3 template: parallel results come back in the order of the calls" {
    const a = testing.allocator;
    var config = templateConfig(a);
    const calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" },
        .{ .id = "call_2", .name = "get_time", .arguments = "{}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "Q" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "R" },
        .{ .role = "tool", .content = "12:00", .tool_call_id = "call_2" },
        .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
    };
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqual(before, template_fallbacks.load(.monotonic));
    const from = std.mem.indexOf(u8, rendered, "<|user|>").?;
    try testing.expectEqualStrings(
        "<|user|>Q<|assistant|><think>R</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
            "<tool_call>get_time</tool_call><|observation|><tool_response>Sunny</tool_response><tool_response>12:00</tool_response><|assistant|><think>",
        rendered[from..],
    );
}
