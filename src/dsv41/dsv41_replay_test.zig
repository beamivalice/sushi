//! DeepSeek-V4.1 turns replayed the way a client does: the model's tokens for a raw reply, the server's split and
//! parse of its text, and the next request's prompt tokens through the real template.
const std = @import("std");
const arch_chat = @import("../arch/chat.zig");
const chat = @import("../chat.zig");
const think_tags = @import("../think_tags.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const tool_calls = @import("../tool_calls.zig");
const ChatConfig = chat.ChatConfig;
const Message = chat.Message;
const ToolCall = chat.ToolCall;
const Tokenizer = tokenizer_mod.Tokenizer;
const parseToolCalls = tool_calls.parseToolCalls;
const renderChatTemplate = chat.renderChatTemplate;
const splitThinkBlock = think_tags.splitThinkBlock;
const testing = std.testing;

const Dsv41Replay = struct {
    gen: []u32,
    /// Generated tokens the next prompt repeats, from the first one.
    reused: usize,
    /// The next prompt starts with the first prompt.
    prefix_kept: bool,
    think_close_at: ?usize,
    markup_at: ?usize,

    fn deinit(self: *Dsv41Replay, a: std.mem.Allocator) void {
        a.free(self.gen);
    }
};

fn dsv41Replay(a: std.mem.Allocator, tok: *const Tokenizer, tools: ?[]const u8, thinking: bool, raw: []const u8) !Dsv41Replay {
    var cfg = ChatConfig{ .chat_template = @embedFile("../fixtures/dsv41_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = (arch_chat.pinOf("deepseek_v41") orelse return error.SkipZigTest).image_placeholder };
    const first = [_]Message{.{ .role = "user", .content = "q" }};
    const p1_text = try renderChatTemplate(a, &first, &cfg, tools, null, thinking, "high", false);
    defer a.free(p1_text);
    const p1 = try tok.encode(a, p1_text);
    defer a.free(p1);
    const gen = try tok.encode(a, raw);
    errdefer a.free(gen);

    const split = splitThinkBlock(raw, true, thinking);
    const parsed = if (tools != null) try parseToolCalls(a, raw) else null;
    defer if (parsed) |calls| {
        for (calls) |c| {
            a.free(c.name);
            a.free(c.arguments);
        }
        a.free(calls);
    };
    var calls: [4]ToolCall = undefined;
    if (parsed) |pc| for (pc, 0..) |c, i| {
        calls[i] = .{ .id = "call_1", .name = c.name, .arguments = c.arguments };
    };
    const assistant: Message = .{
        .role = "assistant",
        .content = split.content,
        .reasoning_content = split.reasoning_content,
        .tool_calls = if (parsed) |pc| calls[0..pc.len] else null,
    };
    const tail: Message = if (parsed != null) .{ .role = "tool", .content = "ok", .tool_call_id = "call_1" } else .{ .role = "user", .content = "next" };
    const second = [_]Message{ first[0], assistant, tail };
    const p2_text = try renderChatTemplate(a, &second, &cfg, tools, null, thinking, "high", false);
    defer a.free(p2_text);
    const p2 = try tok.encode(a, p2_text);
    defer a.free(p2);

    const keep = p1.len <= p2.len and std.mem.eql(u32, p1, p2[0..p1.len]);
    var reused: usize = 0;
    if (keep) {
        const tail_ids = p2[p1.len..];
        while (reused < gen.len and reused < tail_ids.len and gen[reused] == tail_ids[reused]) reused += 1;
    }
    const close_ids = try tok.encode(a, "</think>");
    defer a.free(close_ids);
    const markup_ids = try tok.encode(a, "\xef\xbd\x9cDSML\xef\xbd\x9c");
    defer a.free(markup_ids);
    return .{
        .gen = gen,
        .reused = reused,
        .prefix_kept = keep,
        .think_close_at = std.mem.indexOfScalar(u32, gen, close_ids[0]),
        .markup_at = std.mem.indexOfScalar(u32, gen, markup_ids[0]),
    };
}

const dsv41_replay_tools =
    \\[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"tags":{"type":"array"}}}}}]
;

test "dsv41 replay: what the next prompt shares with a generated turn (DeepSeek tokenizer.json)" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("../test_models.zig").packPath(&buf, "DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw");
    var tok = tokenizer_mod.loadTokenizer(testing.io, testing.allocator, dir) catch return error.SkipZigTest;
    defer tok.deinit();
    const a = testing.allocator;
    const D = "\xef\xbd\x9cDSML\xef\xbd\x9c";
    const call = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"read_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">src/main.zig</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const call_json = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"read_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">src/main.zig</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"tags\" string=\"false\">[\"a\",\"b\"]</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"limit\" string=\"false\">40</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const call_code = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">a.zig</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"content\" string=\"true\">const s = \"a\\nb\"; // <x> & 'y' \\ \xe2\x80\x94 \n\tfn f() void {}\n</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const reasoning = "The user wants the entry point. I should read src/main.zig first, then look at how the arguments are parsed.";
    const Reuse = enum {
        /// Every generated token comes back.
        whole,
        /// Through the call's first marker, then the re-render differs.
        into_call,
        /// Through the thought minus the whitespace the server trims off its end.
        thought_less_trim,
        /// The turn is not re-rendered as it was generated.
        none,
    };
    const cases = [_]struct { name: []const u8, tools: ?[]const u8, thinking: bool, raw: []const u8, reuse: Reuse }{
        .{ .name = "thinking, tools, content then a call", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>I will read the file." ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call alone", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call, .reuse = .whole },
        // jinja's `tojson` spaces the array's items, the model wrote none.
        .{ .name = "thinking, tools, a call with JSON values", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call_json, .reuse = .into_call },
        .{ .name = "thinking, tools, a call with code in a string", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call_code, .reuse = .whole },
        .{ .name = "thinking, tools, plain answer", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>The entry point is `src/main.zig`.\n\nIt parses the flags first.", .reuse = .whole },
        .{ .name = "thinking, tools, reasoning ends in a newline", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>The entry point is `src/main.zig`.", .reuse = .thought_less_trim },
        // Without tools the template drops an earlier turn's thought and opens it `</think>`, not `<think>`.
        .{ .name = "thinking, no tools", .tools = null, .thinking = true, .raw = reasoning ++ "</think>The entry point is `src/main.zig`.", .reuse = .none },
        .{ .name = "chat, tools, content then a call", .tools = dsv41_replay_tools, .thinking = false, .raw = "I will read the file." ++ call, .reuse = .whole },
        .{ .name = "chat, no tools", .tools = null, .thinking = false, .raw = "The entry point is `src/main.zig`.", .reuse = .whole },
    };
    for (cases) |c| {
        var r = try dsv41Replay(a, &tok, c.tools, c.thinking, c.raw);
        defer r.deinit(a);
        errdefer std.debug.print("{s}: {d} tokens, prefix kept {}, reused {d}, </think> at {?d}, markup at {?d}\n", .{ c.name, r.gen.len, r.prefix_kept, r.reused, r.think_close_at, r.markup_at });
        switch (c.reuse) {
            .whole => try testing.expectEqual(r.gen.len, r.reused),
            .into_call => try testing.expect(r.reused > r.markup_at.? and r.reused < r.gen.len),
            .thought_less_trim => try testing.expectEqual(r.think_close_at.? - 1, r.reused),
            .none => try testing.expect(!r.prefix_kept),
        }
        // The point behind the closer is inside what the next prompt shares, unless the thought's edge is whitespace.
        if (c.thinking and c.tools != null and c.reuse != .thought_less_trim) try testing.expect(r.reused > r.think_close_at.?);
    }
}
