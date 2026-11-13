//! The template oracle: every call `chat.formatChat` makes into the template engine over a corpus of requests,
//! through each served template, rendered by Python Jinja2 under transformers' chat-template rules
//! (`tests/dump_template_oracle.py`, committed as `fixtures/template_oracle.json`) and by jinja.cpp.
//!
//! The requests: the shapes of the template tests in `format_corpus_test.zig` and its recorded model outputs,
//! `fixtures/tool_traffic.jsonl` (every tools table and a selection of its outputs), the V4.1 encoder cases of
//! `fixtures/dsv41_template_cases.json`, and edge inputs (control bytes, U+FDD0, special-token text in every role,
//! empty and missing fields, large integers and floats). Each runs at the thinking, effort and preserve-thinking
//! settings the server sends, and the probes formatChat and the forced-call reader make are captured with them.

const std = @import("std");
const testing = std.testing;
const jinja_c = @import("jinja_c");
const zig_template = @import("template/template.zig");
const chat = @import("chat.zig");
const arch_chat = @import("arch/chat.zig");
const tokenizer_mod = @import("tokenizer.zig");
const tool_calls_mod = @import("tool_calls.zig");
const format_corpus = @import("format_corpus_test.zig");
const arch_fx = @import("arch_fixtures_test.zig");

const oracle_json = @embedFile("fixtures/template_oracle.json");

const Message = chat.Message;
const ToolCall = chat.ToolCall;

/// One engine call as the fixture holds it; `id` is a hash of the inputs.
const Case = struct {
    id: []const u8,
    label: []const u8,
    template: []const u8,
    messages: []const u8,
    tools: ?[]const u8,
    extra: []const u8,
    add_generation_prompt: bool,
};

const OracleCase = struct {
    id: []const u8,
    label: []const u8,
    template: []const u8,
    messages: []const u8,
    tools: ?[]const u8,
    extra: []const u8,
    add_generation_prompt: bool,
    python: ?[]const u8,
    python_error: ?[]const u8,
};

const Oracle = struct {
    generator: []const u8,
    templates: std.json.ArrayHashMap([]const u8),
    cases: []const OracleCase,
};

/// A template as a pack serves it. `tok` null: no tokenizer of that pack on the release box, rendered unmarked.
const Served = struct { id: []const u8, config: chat.ChatConfig, tok: ?*const tokenizer_mod.Tokenizer };

const Arm = struct { thinking: bool, effort: ?[]const u8 = null, preserve: ?bool = null };
const both_arms = [_]Arm{ .{ .thinking = true }, .{ .thinking = false } };
const on_arm = [_]Arm{.{ .thinking = true }};

const Request = struct {
    label: []const u8,
    messages: []const Message,
    tools: ?[]const u8 = null,
    instr: ?[]const u8 = null,
    continue_final: bool = false,
    arms: []const Arm = &both_arms,
};

/// A template text the capture names: a served one, or its arch-adapted copy (`+late_system`).
const Known = struct { id: []const u8, text: []const u8 };

const Capture = struct {
    arena: std.mem.Allocator,
    known: []const Known,
    label: []const u8 = "",
    cases: std.ArrayList(Case) = .empty,
    seen: std.StringHashMapUnmanaged(void) = .empty,
    failure: ?anyerror = null,

    fn see(ctx: *anyopaque, call: chat.EngineCall) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        self.record(call) catch |e| {
            self.failure = e;
        };
    }

    fn record(self: *Capture, call: chat.EngineCall) !void {
        const template = for (self.known) |k| {
            if (std.mem.eql(u8, k.text, call.template)) break k.id;
        } else return error.TemplateNotServed;
        const id = caseId(template, call.messages, call.tools, call.extra, call.add_generation_prompt);
        const gop = try self.seen.getOrPut(self.arena, try self.arena.dupe(u8, &id));
        if (gop.found_existing) return;
        try self.cases.append(self.arena, .{
            .id = gop.key_ptr.*,
            .label = self.label,
            .template = template,
            .messages = try self.arena.dupe(u8, call.messages),
            .tools = if (call.tools) |t| try self.arena.dupe(u8, t) else null,
            .extra = try self.arena.dupe(u8, call.extra),
            .add_generation_prompt = call.add_generation_prompt,
        });
    }
};

fn caseId(template: []const u8, messages: []const u8, tools: ?[]const u8, extra: []const u8, add_generation_prompt: bool) [24]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for ([_][]const u8{ template, messages, tools orelse "\x01", extra, if (add_generation_prompt) "1" else "0" }) |part| {
        h.update(part);
        h.update("\x00");
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest[0..12].*, .lower);
}

fn testTokenizer(a: std.mem.Allocator, arena: std.mem.Allocator, pack: []const u8) !*tokenizer_mod.Tokenizer {
    const tok = try arena.create(tokenizer_mod.Tokenizer);
    tok.* = (try tokenizer_mod.packTokenizerForTests(a, arena, pack)).tok;
    return tok;
}

/// The four templates with the bos/eos their packs' tokenizer_config.json carries (DeepSeek's are objects, read as
/// none). A template whose serving arch is not in the build is left out: its arch adapts or pins it.
fn servedTemplates(a: std.mem.Allocator, arena: std.mem.Allocator, toks: *std.ArrayList(*tokenizer_mod.Tokenizer)) ![]Served {
    var out = std.ArrayList(Served).empty;
    if (arch_fx.qwen_template) |qwen| {
        const tok = try testTokenizer(a, arena, "qwen38_flash_next");
        try toks.append(arena, tok);
        try out.append(arena, .{ .id = "qwen38_flash_next", .tok = tok, .config = .{ .chat_template = qwen, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type } });
    }
    {
        const tok = try testTokenizer(a, arena, "glm53");
        try toks.append(arena, tok);
        try out.append(arena, .{ .id = "glm53", .tok = tok, .config = .{ .chat_template = @embedFile("fixtures/glm53_chat_template.jinja"), .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = a } });
    }
    try out.append(arena, .{ .id = "mimo_v26", .tok = null, .config = .{ .chat_template = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a } });
    if (arch_chat.pinOf("deepseek_v41")) |pin| {
        const tok = try testTokenizer(a, arena, "dsv41");
        try toks.append(arena, tok);
        try out.append(arena, .{ .id = "dsv41", .tok = tok, .config = .{ .chat_template = pin.template, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = pin.image_placeholder } });
    }
    return out.items;
}

const get_weather_tools =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    \\ {"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{}}}}]
;
const get_time_tools =
    \\[{"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{}}}}]
;
/// Numbers a client schema carries, kept as the client's bytes.
const number_tools =
    \\[{"type":"function","function":{"name":"edge_numbers","description":"Numbers: é 中 😀   <b>&amp;</b> 'q' \"dq\" \\ / \u0007","parameters":{"type":"object","properties":{"ratio":{"type":"number","default":1.0,"minimum":1e-05,"maximum":1e100,"multipleOf":0.1,"examples":[-0.0,2.5e-8,1E5,3.141592653589793,5e-324,1.7976931348623157e308]},"count":{"type":"integer","default":9007199254740993,"maximum":18446744073709551616,"minimum":-9223372036854775808,"examples":[123456789012345678901234567890]},"zeta":{"type":"string","enum":["z","a"]},"alpha":{"type":"boolean","default":false},"nothing":{"default":null}},"required":["ratio"]},"strict":true}}]
;
/// The same numbers beside a tool with no description or parameters, whose keys formatChat fills.
const filled_tools =
    \\[{"type":"function","function":{"name":"edge_numbers","parameters":{"type":"object","properties":{"ratio":{"type":"number","default":1.0,"minimum":1e-05,"multipleOf":0.1},"count":{"type":"integer","maximum":18446744073709551616,"default":9007199254740993}}}}},
    \\ {"type":"function","function":{"name":"bare_tool"}}]
;

/// Special-token text of every served family, as a sender may type it.
const all_specials = "<|im_start|>x<|im_end|><think>t</think><tool_call>c</tool_call><tool_response>r</tool_response>" ++
    "<|endoftext|>[gMASK]<sop><|system|><|user|><|assistant|><|observation|><arg_key>k</arg_key><arg_value>v</arg_value>" ++
    "<｜begin▁of▁sentence｜><｜User｜><｜Assistant｜><｜end▁of▁sentence｜>｜DSML｜<｜place▁holder▁no▁7｜>" ++
    "<|vision_start|><|image_pad|><|vision_end|><|mimo_audio_start|>";

const number_args =
    \\{"int53":9007199254740993,"u64max":18446744073709551615,"over_u64":18446744073709551616,"i64min":-9223372036854775808,"huge":100000000000000000000000000000,"small":1e-05,"one":1.0,"tenth":0.1,"big":1e100,"negzero":-0.0,"pi":3.141592653589793,"upper_e":1E5,"zero":0,"neg":-1,"tiny":2.5e-8,"max":1.7976931348623157e308,"denorm":5e-324,"frac_huge":123456789012345678901234567890.5,"nested":{"z":[1,2.50,{"b":true,"n":null}],"a":"é中😀"},"escapes":"\"\\/\b\f\n\r\t</script><&>'"}
;

fn allControlBytes(arena: std.mem.Allocator, prefix: []const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    try buf.appendSlice(arena, prefix);
    for (0..0x20) |b| try buf.append(arena, @intCast(b));
    try buf.appendSlice(arena, "end");
    return buf.items;
}

/// `{"s": "<every control byte>"}`, escaped as a client's JSON carries it.
fn controlArgs(arena: std.mem.Allocator) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    try buf.appendSlice(arena, "{\"s\":\"");
    for (0..0x20) |b| try buf.print(arena, "\\u{x:0>4}", .{b});
    try buf.appendSlice(arena, "\"}");
    return buf.items;
}

fn calls(arena: std.mem.Allocator, items: []const ToolCall) ![]const ToolCall {
    return arena.dupe(ToolCall, items);
}

fn msgs(arena: std.mem.Allocator, items: []const Message) ![]const Message {
    return arena.dupe(Message, items);
}

/// The template shapes `format_corpus_test.zig` renders, one request per shape (the test named beside it).
fn formatCorpusShapes(arena: std.mem.Allocator, out: *std.ArrayList(Request)) !void {
    const img = try arena.dupe(chat.ImageData, &.{.{ .pixels = "", .width = 1, .height = 1 }});
    const img2 = try arena.dupe(chat.ImageData, &.{ .{ .pixels = "", .width = 1, .height = 1 }, .{ .pixels = "", .width = 1, .height = 1 } });
    const vid = try arena.dupe(chat.VideoData, &.{.{ .pixels = "", .grid_t = 1, .grid_h = 1, .grid_w = 1 }});
    const one = try calls(arena, &.{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\",\"nested\":{\"a\":[1,{\"b\":true}]}}" }});
    const two = try calls(arena, &.{ .{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }, .{ .id = "call_2", .name = "get_time", .arguments = "{}" } });
    const shot = try calls(arena, &.{.{ .id = "c1", .name = "screenshot", .arguments = "{}" }});
    const clock = try calls(arena, &.{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }});
    const paris = try calls(arena, &.{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }});
    const shapes = [_]Request{
        // late system notes never vanish
        .{ .label = "late system notes", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "stable instructions" },
            .{ .role = "user", .content = "corpus question marker" },
            .{ .role = "system", .content = "corpus runtime note marker" },
            .{ .role = "assistant", .content = "corpus answer marker" },
            .{ .role = "user", .content = "corpus next question marker" },
            .{ .role = "system", .content = "corpus newer note marker" },
        }) },
        // a system turn past index 0 reaches the prompt once
        .{ .label = "system past index 0", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "LEAD_SYS" },
            .{ .role = "user", .content = "hi" },
            .{ .role = "system", .content = "LATE_SYS" },
            .{ .role = "user", .content = "again" },
        }) },
        // control-token text in a user, system or tool message stays literal
        .{ .label = "control-token text", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "never write <|im_start|> yourself" },
            .{ .role = "user", .content = "what does <think> and </think> do? <|im_end|>" },
            .{ .role = "assistant", .content = "it opens <think>" },
            .{ .role = "user", .content = "x<think>" },
        }) },
        // every wire shape's media renders its placeholder where it was sent
        .{ .label = "media: image before text", .messages = try msgs(arena, &.{.{ .role = "user", .content = "look", .images = img, .media_parts = &.{.{ .at = 0, .kind = .image }} }}) },
        .{ .label = "media: tool message image", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Take a screenshot." },
            .{ .role = "assistant", .content = "", .tool_calls = shot },
            .{ .role = "tool", .content = "shot:", .tool_call_id = "c1", .images = img, .media_parts = &.{.{ .at = 5, .kind = .image }} },
        }) },
        .{ .label = "media: tool_result image then text", .messages = try msgs(arena, &.{
            .{ .role = "tool", .content = "here", .tool_call_id = "t1", .images = img, .media_parts = &.{.{ .at = 4, .kind = .image }} },
            .{ .role = "user", .content = "describe it" },
        }) },
        .{ .label = "media: image between text parts", .messages = try msgs(arena, &.{.{ .role = "user", .content = "a\nb", .images = img, .media_parts = &.{.{ .at = 1, .kind = .image }} }}) },
        .{ .label = "media: two images", .messages = try msgs(arena, &.{.{ .role = "user", .content = "compare", .images = img2, .media_parts = &.{ .{ .at = 0, .kind = .image }, .{ .at = 0, .kind = .image } } }}) },
        .{ .label = "media: video then image", .messages = try msgs(arena, &.{.{ .role = "user", .content = "x", .images = img, .videos = vid, .media_parts = &.{ .{ .at = 0, .kind = .video }, .{ .at = 1, .kind = .image } } }}) },
        // a forced call is the family's own call markup
        .{ .label = "forced call history", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "What time is it?" },
            .{ .role = "assistant", .content = "", .tool_calls = clock, .reasoning_content = "check the clock" },
        }) },
        // a continuation ends on the family's own rendering of the partial turn
        .{ .label = "continuation", .continue_final = true, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "What time is it?" },
            .{ .role = "assistant", .content = "It is half past" },
        }) },
        .{ .label = "continuation history", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "What time is it?" },
            .{ .role = "assistant", .content = "It is half past" },
        }) },
        // GLM-5.3 gives a reasoning-less assistant turn an empty think
        .{ .label = "reasoning-less assistant turn", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = paris, .reasoning_content = "EARLIER_THOUGHT" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
            .{ .role = "assistant", .content = "final answer" },
            .{ .role = "user", .content = "Thanks" },
        }) },
        // a conversation with no user turn is refused by name
        .{ .label = "no user turn: system only", .messages = try msgs(arena, &.{.{ .role = "system", .content = "You are terse." }}) },
        .{ .label = "no user turn: assistant only", .messages = try msgs(arena, &.{.{ .role = "assistant", .content = "hi" }}) },
        .{ .label = "no user turn: system then user", .messages = try msgs(arena, &.{ .{ .role = "system", .content = "You are terse." }, .{ .role = "user", .content = "hi" } }) },
    };
    try out.appendSlice(arena, &shapes);

    // a forced tool_choice reaches the prompt once
    for ([_]tool_calls_mod.ToolChoice{ .required, .{ .named = "get_time" } }) |choice| {
        const instr = (try choice.instruction(arena)).?;
        try out.append(arena, .{ .label = "forced tool_choice", .tools = get_time_tools, .instr = instr, .messages = try msgs(arena, &.{.{ .role = "user", .content = "What time is it?" }}) });
    }
    // a string value parses back byte-exact from each family's own call rendering
    for ([_][]const u8{ "plain", "last line\n", "\nafter a blank line", "two blank lines\n\n", "  indented\n", "\n" }) |value| {
        const args = try std.json.Stringify.valueAlloc(arena, .{ .path = "a.txt", .content = value }, .{});
        try out.append(arena, .{ .label = "string value round trip", .arms = &on_arm, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Write it." },
            .{ .role = "assistant", .content = "", .tool_calls = try calls(arena, &.{.{ .id = "tc_0", .name = "write_file", .arguments = args }}), .reasoning_content = "r" },
        }) });
    }
    // a history call whose arguments are not an object still renders its own template
    for ([_][]const u8{ "", "null", "[]", "[1,2]", "5", "\"text\"", "not json", "{\"a\":" }) |args| {
        try out.append(arena, .{ .label = "history call arguments not an object", .arms = &on_arm, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = try calls(arena, &.{.{ .id = "call_1", .name = "get_weather", .arguments = args }}), .reasoning_content = "r" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
        }) });
    }
    // tool traffic renders through every served template
    const traffic_shapes = [_]struct { label: []const u8, messages: []const Message }{
        .{ .label = "tool round trip", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "Be brief." },
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = one, .reasoning_content = "r" },
            .{ .role = "tool", .content = "Sunny, 21C", .tool_call_id = "call_1" },
        }) },
        .{ .label = "tool result then a new user turn", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = one, .reasoning_content = "r" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
            .{ .role = "assistant", .content = "It is sunny.", .reasoning_content = "done" },
            .{ .role = "user", .content = "And tomorrow?" },
        }) },
        .{ .label = "parallel results, reversed", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Both?" },
            .{ .role = "assistant", .content = "", .tool_calls = two, .reasoning_content = "r" },
            .{ .role = "tool", .content = "12:00", .tool_call_id = "call_2" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
        }) },
        .{ .label = "tool result without an id, special characters", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "Checking.", .tool_calls = one },
            .{ .role = "tool", .content = "Soleil \"\xe2\x98\x80\" <b>&</b>\nline2\t\xe4\xb8\xad" },
        }) },
        .{ .label = "empty tool result", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = one, .reasoning_content = "r" },
            .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
        }) },
        .{ .label = "image turn then a tool round", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Look: ", .images = img, .media_parts = &.{.{ .at = 6, .kind = .image }} },
            .{ .role = "assistant", .content = "", .tool_calls = one, .reasoning_content = "r" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
        }) },
        .{ .label = "video turn", .messages = try msgs(arena, &.{.{ .role = "user", .content = "Watch: ", .videos = vid, .media_parts = &.{.{ .at = 7, .kind = .video }} }}) },
    };
    const effort_arms = [_]Arm{ .{ .thinking = true }, .{ .thinking = false }, .{ .thinking = true, .effort = "low" }, .{ .thinking = false, .effort = "low" } };
    for (traffic_shapes) |s| for ([_]?[]const u8{ null, get_weather_tools }) |tools| {
        try out.append(arena, .{ .label = s.label, .messages = s.messages, .tools = tools, .arms = &effort_arms });
    };
}

/// The recorded model outputs of the format corpus, as the assistant turns of a history, eight to a conversation,
/// and the hostile tool results its round-trip test sends back.
fn recordedOutputs(arena: std.mem.Allocator, out: *std.ArrayList(Request)) !void {
    var raws = std.ArrayList([]const u8).empty;
    for (format_corpus.corpus) |e| try raws.append(arena, e.raw);
    for (format_corpus.dialects) |d| try raws.append(arena, d.raw);
    try historiesOf(arena, out, "format corpus output", raws.items, 8);

    const bash = try calls(arena, &.{.{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" }});
    for ([_][]const u8{
        "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?\n\u{2502}  \u{25cf} SvelteKit minimal",
        try allControlBytes(arena, ""),
        "\xff\xfe\x80 binary",
    }) |result| {
        try out.append(arena, .{ .label = "hostile tool result", .arms = &on_arm, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "make me a sveltekit app" },
            .{ .role = "assistant", .content = format_corpus.corpus[0].raw, .tool_calls = bash },
            .{ .role = "tool", .content = result, .tool_call_id = "tc_0" },
        }) });
    }
}

fn historiesOf(arena: std.mem.Allocator, out: *std.ArrayList(Request), label: []const u8, raws: []const []const u8, per: usize) !void {
    var i: usize = 0;
    while (i < raws.len) : (i += per) {
        const group = raws[i..@min(raws.len, i + per)];
        var history = std.ArrayList(Message).empty;
        for (group) |raw| {
            try history.append(arena, .{ .role = "user", .content = "Go on." });
            try history.append(arena, .{ .role = "assistant", .content = raw });
        }
        try history.append(arena, .{ .role = "user", .content = "Next." });
        try out.append(arena, .{ .label = try std.fmt.allocPrint(arena, "{s} {d}..{d}", .{ label, i, i + group.len }), .messages = history.items, .arms = &on_arm });
    }
}

/// Every tools table of the traffic fixture in one short turn; of its model outputs, every one carrying a think tag,
/// a served family's turn-marker text or a control byte, and the first of each other kind per model (tool-call
/// markup or not, `<|` markup or not, ASCII or not).
fn toolTraffic(arena: std.mem.Allocator, out: *std.ArrayList(Request)) !void {
    const greet = try msgs(arena, &.{ .{ .role = "system", .content = "Be brief." }, .{ .role = "user", .content = "Weather?" } });
    var kinds_seen = std.StringHashMapUnmanaged(void).empty;
    var raws = std.ArrayList([]const u8).empty;
    var lines = std.mem.splitScalar(u8, @embedFile("fixtures/tool_traffic.jsonl"), '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len == 0) continue;
        const rec = (try std.json.parseFromSliceLeaky(std.json.Value, arena, line, .{})).object;
        const raw = if (rec.get("raw")) |r| r.string else {
            const tools = try std.json.Stringify.valueAlloc(arena, rec.get("tools").?, .{});
            try out.append(arena, .{ .label = try std.fmt.allocPrint(arena, "tool traffic tools {s}", .{rec.get("tools_id").?.string}), .messages = greet, .tools = tools, .arms = &on_arm });
            continue;
        };
        const has = struct {
            fn has(text: []const u8, needle: []const u8) bool {
                return std.mem.indexOf(u8, text, needle) != null;
            }
        }.has;
        const rare = has(raw, "<think>") or has(raw, "</think>") or has(raw, "<|im_") or has(raw, "<|endoftext|>") or
            has(raw, "<\u{ff5c}") or for (raw) |c| {
            if (c < 0x20 and c != '\n' and c != '\t') break true;
        } else false;
        if (!rare) {
            const non_ascii = for (raw) |c| {
                if (c >= 0x80) break true;
            } else false;
            const model = if (rec.get("model")) |m| m.string else "";
            const kind = try std.fmt.allocPrint(arena, "{s}|{}|{}|{}", .{ model, has(raw, "<tool_call"), has(raw, "<|"), non_ascii });
            if ((try kinds_seen.getOrPut(arena, kind)).found_existing) continue;
        }
        try raws.append(arena, raw);
    }
    try historiesOf(arena, out, "tool traffic output", raws.items, 6);
}

/// The V4.1 encoder cases, each at its own thinking mode and effort.
fn dsv41Cases(arena: std.mem.Allocator, out: *std.ArrayList(Request)) !void {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, @embedFile("fixtures/dsv41_template_cases.json"), .{});
    for (doc.array.items) |case| {
        const o = case.object;
        var history = std.ArrayList(Message).empty;
        for (o.get("messages").?.array.items) |mv| {
            const m = mv.object;
            var tcs: ?[]const ToolCall = null;
            if (m.get("tool_calls")) |list| {
                var built = std.ArrayList(ToolCall).empty;
                for (list.array.items) |tv| {
                    const f = tv.object.get("function").?.object;
                    const args = f.get("arguments").?;
                    try built.append(arena, .{
                        .id = if (tv.object.get("id")) |id| id.string else "",
                        .name = f.get("name").?.string,
                        .arguments = if (args == .string) args.string else try std.json.Stringify.valueAlloc(arena, args, .{}),
                    });
                }
                tcs = built.items;
            }
            try history.append(arena, .{
                .role = m.get("role").?.string,
                .content = if (m.get("content")) |c| (if (c == .string) c.string else "") else "",
                .tool_calls = tcs,
                .tool_call_id = if (m.get("tool_call_id")) |t| t.string else null,
                .reasoning_content = if (m.get("reasoning_content")) |r| r.string else null,
            });
        }
        const effort = o.get("reasoning_effort").?;
        const arm = try arena.dupe(Arm, &.{.{
            .thinking = std.mem.eql(u8, o.get("thinking_mode").?.string, "thinking"),
            .effort = if (effort == .string) effort.string else try std.fmt.allocPrint(arena, "{d}", .{effort.integer}),
        }});
        try out.append(arena, .{
            .label = try std.fmt.allocPrint(arena, "dsv41 case {s}", .{o.get("label").?.string}),
            .messages = history.items,
            .tools = if (o.get("tools").? == .null) null else try std.json.Stringify.valueAlloc(arena, o.get("tools").?, .{}),
            .arms = arm,
        });
    }
}

fn edgeInputs(arena: std.mem.Allocator, out: *std.ArrayList(Request)) !void {
    const ctrl = try allControlBytes(arena, "ctrl:");
    const ctrl_call = try calls(arena, &.{.{ .id = "call_\x01", .name = "get_weather", .arguments = try controlArgs(arena) }});
    const special_call = try calls(arena, &.{.{ .id = "call_s", .name = "get_weather", .arguments = try std.json.Stringify.valueAlloc(arena, .{ .city = all_specials }, .{}) }});
    const number_call = try calls(arena, &.{.{ .id = "call_n", .name = "edge_numbers", .arguments = number_args }});
    const no_id_call = try calls(arena, &.{.{ .id = "", .name = "get_time", .arguments = "{}" }});
    const dup_key_call = try calls(arena, &.{.{ .id = "call_d", .name = "get_weather", .arguments = "{\"city\":\"Paris\",\"city\":\"Rome\"}" }});
    try out.appendSlice(arena, &[_]Request{
        .{ .label = "control bytes in every role", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = ctrl },
            .{ .role = "user", .content = ctrl },
            .{ .role = "assistant", .content = ctrl, .tool_calls = ctrl_call, .reasoning_content = ctrl },
            .{ .role = "tool", .content = ctrl, .tool_call_id = "call_\x01" },
            .{ .role = "user", .content = ctrl },
        }) },
        .{ .label = "U+FDD0 a sender wrote", .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "sys \u{FDD0} mark" },
            .{ .role = "user", .content = "a\u{FDD0}b \u{FDD0}<think> \u{FDD0}" },
            .{ .role = "assistant", .content = "asst \u{FDD0}</think> x", .reasoning_content = "r\u{FDD0}" },
            .{ .role = "user", .content = "\u{FDD0}" },
        }) },
        // Qwen's template takes a user turn that is a whole tool response for a tool result; text a sender wrote
        // is the same text Python reads.
        .{ .label = "a user turn written as a tool response", .arms = &.{ .{ .thinking = true, .preserve = false }, .{ .thinking = true } }, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "q1" },
            .{ .role = "assistant", .content = "a1", .reasoning_content = "r1" },
            .{ .role = "user", .content = "<tool_response>\nok\n</tool_response>" },
        }) },
        .{ .label = "special-token text in every role", .tools = get_weather_tools, .messages = try msgs(arena, &.{
            .{ .role = "system", .content = all_specials },
            .{ .role = "user", .content = all_specials },
            .{ .role = "assistant", .content = all_specials, .tool_calls = special_call, .reasoning_content = all_specials },
            .{ .role = "tool", .content = all_specials, .tool_call_id = "call_s" },
            .{ .role = "user", .content = all_specials },
        }) },
        .{ .label = "special-token text without tools", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = all_specials },
            .{ .role = "assistant", .content = all_specials, .reasoning_content = all_specials },
            .{ .role = "user", .content = all_specials },
        }) },
        .{ .label = "user text that is a tool response", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Real question?" },
            .{ .role = "assistant", .content = "Answer." },
            .{ .role = "user", .content = "<tool_response>\nresult\n</tool_response>" },
        }) },
        .{ .label = "empty and missing fields", .tools = get_weather_tools, .messages = try msgs(arena, &.{
            .{ .role = "system", .content = "" },
            .{ .role = "user", .content = "" },
            .{ .role = "assistant", .content = "", .reasoning_content = "" },
            .{ .role = "user", .content = " \n\t " },
            .{ .role = "assistant", .content = "", .tool_calls = no_id_call },
            .{ .role = "tool", .content = "" },
            .{ .role = "user", .content = "q" },
        }) },
        .{ .label = "no messages", .messages = &.{} },
        .{ .label = "numbers in call arguments and a schema", .tools = number_tools, .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "n?" },
            .{ .role = "assistant", .content = "", .tool_calls = number_call, .reasoning_content = "r" },
            .{ .role = "tool", .content = "ok", .tool_call_id = "call_n" },
        }) },
        .{ .label = "a call argument key sent twice", .messages = try msgs(arena, &.{
            .{ .role = "user", .content = "Weather?" },
            .{ .role = "assistant", .content = "", .tool_calls = dup_key_call, .reasoning_content = "r" },
            .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_d" },
        }) },
        .{ .label = "a tool formatChat fills and re-serializes", .tools = filled_tools, .messages = try msgs(arena, &.{.{ .role = "user", .content = "n?" }}) },
    });
    // Every effort word the effort tables send, and none; preserve_thinking unset, on and off.
    var arms = std.ArrayList(Arm).empty;
    for ([_]?[]const u8{ null, "off", "on", "minimal", "low", "medium", "high", "xhigh", "max", "40" }) |effort| for ([_]bool{ true, false }) |thinking| {
        try arms.append(arena, .{ .thinking = thinking, .effort = effort });
    };
    for ([_]?bool{ true, false }) |keep| for ([_]bool{ true, false }) |thinking| {
        try arms.append(arena, .{ .thinking = thinking, .preserve = keep });
    };
    try out.append(arena, .{ .label = "settings ladder", .arms = arms.items, .messages = try msgs(arena, &.{
        .{ .role = "user", .content = "Why is the sky blue?" },
        .{ .role = "assistant", .content = "Rayleigh scattering.", .reasoning_content = "Shorter wavelengths scatter more." },
        .{ .role = "user", .content = "And sunsets?" },
    }) });
}

fn corpusRequests(arena: std.mem.Allocator) ![]const Request {
    var out = std.ArrayList(Request).empty;
    try formatCorpusShapes(arena, &out);
    try recordedOutputs(arena, &out);
    try toolTraffic(arena, &out);
    try dsv41Cases(arena, &out);
    try edgeInputs(arena, &out);
    return out.items;
}

/// Runs every request through every served template as the server does, `capture` watching the engine.
fn runCorpus(a: std.mem.Allocator, arena: std.mem.Allocator, capture: *Capture, served: []const Served) !void {
    const requests = try corpusRequests(arena);
    chat.engine_spy = .{ .ctx = capture, .see = Capture.see };
    defer chat.engine_spy = null;
    for (served) |s| {
        for (requests) |r| for (r.arms) |arm| {
            capture.label = try std.fmt.allocPrint(arena, "{s} | {s} | thinking={} effort={s} preserve={?}{s}", .{
                s.id, r.label, arm.thinking, arm.effort orelse "none", arm.preserve, if (r.continue_final) " continue" else "",
            });
            var cfg = s.config;
            cfg.preserve_thinking = arm.preserve;
            if (s.tok) |tok| {
                const ids = chat.formatChat(a, tok, r.messages, &cfg, r.tools, r.instr, arm.thinking, arm.effort, r.continue_final) catch |e| switch (e) {
                    error.ChatNeedsUserTurn, error.ChatTemplateUnusable => continue,
                    else => return e,
                };
                defer a.free(ids);
                // The prompt the tokenizer encodes is the rendered text, byte for byte: the spans after the think-tag
                // collapse, the thinking-off tail and the continuation cut join to what `renderChatTemplate` gives.
                const text = try chat.renderChatTemplate(a, r.messages, &cfg, r.tools, r.instr, arm.thinking, arm.effort, r.continue_final);
                defer a.free(text);
                const decoded = try tok.decode(a, ids);
                defer a.free(decoded);
                testing.expectEqualStrings(text, decoded) catch |e| {
                    std.debug.print("spans differ from the text: {s}\n", .{capture.label});
                    return e;
                };
            } else {
                const text = chat.renderChatTemplate(a, r.messages, &cfg, r.tools, r.instr, arm.thinking, arm.effort, r.continue_final) catch |e| switch (e) {
                    error.ChatNeedsUserTurn, error.ChatTemplateUnusable => continue,
                    else => return e,
                };
                a.free(text);
            }
            if (capture.failure) |e| return e;
        };
        capture.label = try std.fmt.allocPrint(arena, "{s} | forced-call reader", .{s.id});
        if (try chat.forcedCallText(a, &s.config, "get_time")) |fc| {
            var owned = fc;
            owned.deinit(a);
        }
        if (try chat.callOpener(a, &s.config)) |o| a.free(o);
        if (capture.failure) |e| return e;
    }
}

const Dump = struct {
    templates: std.json.ArrayHashMap([]const u8),
    cases: []const Case,
};

test "template oracle: the corpus is every call formatChat makes into the engine over its requests" {
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var toks = std.ArrayList(*tokenizer_mod.Tokenizer).empty;
    defer for (toks.items) |t| tokenizer_mod.deinitTestTokenizer(t);
    const served = try servedTemplates(a, arena, &toks);

    var known = std.ArrayList(Known).empty;
    for (served) |s| try known.append(arena, .{ .id = s.id, .text = s.config.chat_template });
    if (arch_fx.qwen_template) |qwen| {
        const adapted = (try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, arena, qwen)) orelse return error.LateSystemAdapterDisengaged;
        try known.append(arena, .{ .id = "qwen38_flash_next+late_system", .text = adapted });
    }
    var capture = Capture{ .arena = arena, .known = known.items };
    try runCorpus(a, arena, &capture, served);

    if (std.c.getenv("SUSHI_TEMPLATE_ORACLE_DUMP")) |path| {
        var templates = std.json.ArrayHashMap([]const u8){};
        for (known.items) |k| try templates.map.put(arena, k.id, k.text);
        const text = try std.json.Stringify.valueAlloc(arena, Dump{ .templates = templates, .cases = capture.cases.items }, .{});
        const file = try std.Io.Dir.createFileAbsolute(testing.io, std.mem.span(path), .{});
        defer file.close(testing.io);
        try file.writeStreamingAll(testing.io, text);
        return;
    }

    const oracle = try std.json.parseFromSliceLeaky(Oracle, arena, oracle_json, .{});
    var by_id = std.StringHashMapUnmanaged(OracleCase).empty;
    for (oracle.cases) |c| try by_id.put(arena, c.id, c);
    var missing: usize = 0;
    for (capture.cases.items) |c| {
        const o = by_id.get(c.id) orelse {
            missing += 1;
            if (missing <= 5) std.debug.print("\nnot in the oracle fixture: {s} [{s}]\n", .{ c.id, c.label });
            continue;
        };
        try testing.expectEqualStrings(c.template, o.template);
        try testing.expectEqualStrings(c.messages, o.messages);
        try testing.expectEqualStrings(c.extra, o.extra);
        try testing.expectEqualStrings(c.tools orelse "", o.tools orelse "");
        try testing.expectEqual(c.add_generation_prompt, o.add_generation_prompt);
    }
    if (missing > 0) {
        std.debug.print("{d} engine calls are not in the oracle fixture: rerun tests/dump_template_oracle.py\n", .{missing});
        return error.OracleFixtureStale;
    }
    // Every fixture case of a template this build serves is still made.
    var unreached: usize = 0;
    for (oracle.cases) |o| {
        const served_here = for (known.items) |k| {
            if (std.mem.eql(u8, k.id, o.template)) break true;
        } else false;
        if (served_here and !capture.seen.contains(o.id)) {
            unreached += 1;
            if (unreached <= 5) std.debug.print("\noracle case no request makes: {s} [{s}]\n", .{ o.id, o.label });
        }
    }
    if (unreached > 0) return error.OracleFixtureStale;
}

/// A case where jinja.cpp's render is not Python's, and why. Every listed case must still differ.
const Difference = struct { case: []const u8, what: []const u8, why: []const u8 };

const number_what = "tojson of a number in the tools list and the call arguments: jinja.cpp writes a float that holds an integer " ++
    "without its `.0` (1.0 -> 1, 1E5 -> 100000, -0.0 -> -0), any other float to six significant digits " ++
    "(3.141592653589793 -> 3.14159, 1.7976931348623157e308 -> 1.79769e+308, 5e-324 -> 4.94066e-324), an integer past " ++
    "2^64-1 as such a float (18446744073709551616 -> 1.84467e+19, 10^29 -> 1e+29), and 2^64-1 as -1";
const filled_what = "tojson of the integer 18446744073709551616 in a tools list whose missing keys formatChat filled (a tool " ++
    "had no description): jinja.cpp writes 1.84467e+19";
const number_why = "jinja.cpp keeps a JSON number as an int64 or a double (an unsigned past 2^63 wraps, a larger integer " ++
    "becomes a double) and prints a double with %g; transformers' tojson is json.dumps over Python's int and float, " ++
    "which print exactly. Python is right. A served prompt has these bytes only when a client's tool schema or call " ++
    "arguments carry such a number; no recorded request does.";

const differences = [_]Difference{
    .{ .case = "a4f63c214f484b68b135e51d", .what = number_what, .why = number_why }, // qwen38_flash_next, thinking true
    .{ .case = "ce5b77f2cc9d42325822694f", .what = number_what, .why = number_why }, // qwen38_flash_next, thinking false
    .{ .case = "7edccf3fcc3a12ea107d1913", .what = number_what, .why = number_why }, // glm53, thinking true
    .{ .case = "e7b17f16d6be96b433531339", .what = number_what, .why = number_why }, // glm53, thinking false
    .{ .case = "d63a1a94383b5293e3238fe1", .what = number_what, .why = number_why }, // mimo_v26, thinking true
    .{ .case = "0885ebb8423c54c30f524565", .what = number_what, .why = number_why }, // mimo_v26, thinking false
    .{ .case = "ca48cb49cf7a32bb3e8f0bec", .what = number_what, .why = number_why }, // dsv41, thinking true
    .{ .case = "095f228afd32b3a8d0218f79", .what = number_what, .why = number_why }, // dsv41, thinking false
    .{ .case = "d409857ee965d2a05873e9cd", .what = filled_what, .why = number_why }, // qwen38_flash_next, thinking true
    .{ .case = "915a88f33d13cefdd96654a4", .what = filled_what, .why = number_why }, // qwen38_flash_next, thinking false
    .{ .case = "9245030cd0fa62de0c8e8536", .what = filled_what, .why = number_why }, // glm53, thinking true
    .{ .case = "af5ce95c5b0319468b4b73d2", .what = filled_what, .why = number_why }, // glm53, thinking false
    .{ .case = "027c37e1f049ee354df073a9", .what = filled_what, .why = number_why }, // mimo_v26, thinking true
    .{ .case = "898599c9ee6773f3012e63cf", .what = filled_what, .why = number_why }, // mimo_v26, thinking false
    .{ .case = "7a752a9ffea1cdf9c0c49cb0", .what = filled_what, .why = number_why }, // dsv41, thinking true
    .{ .case = "a5e007eba80bdc5355798f6d", .what = filled_what, .why = number_why }, // dsv41, thinking false
};

test "template oracle: jinja.cpp renders the corpus as Python Jinja2 does, except the listed differences" {
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const oracle = try std.json.parseFromSliceLeaky(Oracle, arena, oracle_json, .{});
    try testing.expect(oracle.cases.len > 0);

    var listed = std.StringHashMapUnmanaged(bool).empty;
    for (differences) |d| try listed.put(arena, d.case, false);
    var unlisted: usize = 0;
    for (oracle.cases) |c| {
        const tpl = try arena.dupeSentinel(u8, oracle.templates.map.get(c.template).?, 0);
        const msgs_z = try arena.dupeSentinel(u8, c.messages, 0);
        const extra_z = try arena.dupeSentinel(u8, c.extra, 0);
        const tools_z: ?[:0]u8 = if (c.tools) |t| try arena.dupeSentinel(u8, t, 0) else null;
        var len: usize = 0;
        const ptr = jinja_c.jinja_render_chat(tpl.ptr, msgs_z.ptr, if (tools_z) |t| t.ptr else null, extra_z.ptr, @intFromBool(c.add_generation_prompt), &len);
        defer if (ptr) |p| jinja_c.jinja_str_free(p);
        const cpp: ?[]const u8 = if (ptr) |p| p[0..len] else null;
        const same = if (cpp) |r| (c.python != null and std.mem.eql(u8, r, c.python.?)) else c.python == null;
        if (same) continue;
        if (listed.getPtr(c.id)) |hit| {
            hit.* = true;
            continue;
        }
        unlisted += 1;
        if (unlisted <= 5) {
            const at = if (cpp != null and c.python != null) std.mem.indexOfDiff(u8, cpp.?, c.python.?) orelse 0 else 0;
            std.debug.print("\n[{s}] {s}\n  jinja.cpp at {d}: {?s}\n  python    at {d}: {?s} {?s}\n", .{
                c.id,                                                 c.label,
                at,                                                   if (cpp) |r| r[at..@min(r.len, at + 120)] else null,
                at,                                                   if (c.python) |p| p[at..@min(p.len, at + 120)] else null,
                c.python_error,
            });
        }
    }
    var stale: usize = 0;
    var it = listed.iterator();
    while (it.next()) |e| if (!e.value_ptr.*) {
        stale += 1;
        std.debug.print("\nlisted difference renders the same now: {s}\n", .{e.key_ptr.*});
    };
    if (unlisted > 0) std.debug.print("\n{d} unlisted differences\n", .{unlisted});
    try testing.expectEqual(@as(usize, 0), unlisted);
    try testing.expectEqual(@as(usize, 0), stale);
}

test "template oracle: the Zig engine renders the corpus as Python Jinja2 does, on every case" {
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const oracle = try std.json.parseFromSliceLeaky(Oracle, arena, oracle_json, .{});
    try testing.expect(oracle.cases.len > 0);

    var failed: usize = 0;
    for (oracle.cases) |c| {
        var case_arena = std.heap.ArenaAllocator.init(a);
        defer case_arena.deinit();
        var diag = zig_template.Diagnostic{};
        const got = zig_template.renderJson(case_arena.allocator(), .{
            .template = oracle.templates.map.get(c.template).?,
            .messages_json = c.messages,
            .tools_json = c.tools,
            .extra_json = c.extra,
            .add_generation_prompt = c.add_generation_prompt,
        }, &diag);
        const same = if (got) |text|
            c.python != null and std.mem.eql(u8, text, c.python.?)
        else |e| e == error.TemplateRaised and c.python_error != null and
            std.mem.eql(u8, c.python_error.?, try std.fmt.allocPrint(case_arena.allocator(), "TemplateError: {s}", .{diag.message()}));
        if (same) continue;
        failed += 1;
        if (failed > 5) continue;
        if (got) |text| {
            const want = c.python orelse "";
            const at = std.mem.indexOfDiff(u8, text, want) orelse 0;
            const from = at -| 60;
            std.debug.print("\n[{s}] {s}\n  zig    at {d}: {s}\n  python at {d}: {s} {?s}\n", .{
                c.id,                            c.label,
                at,                              text[from..@min(text.len, at + 120)],
                at,                              want[@min(want.len, from)..@min(want.len, at + 120)],
                c.python_error,
            });
        } else |e| std.debug.print("\n[{s}] {s}\n  zig: {s} at {d}:{d}: {s}\n  python: {?s}\n", .{ c.id, c.label, @errorName(e), diag.line, diag.column, diag.message(), c.python_error });
    }
    if (failed > 0) std.debug.print("\n{d} of {d} cases differ\n", .{ failed, oracle.cases.len });
    try testing.expectEqual(@as(usize, 0), failed);
}

const BuiltinCase = struct {
    covers: []const u8,
    label: []const u8,
    template: []const u8,
    extra: []const u8,
    python: ?[]const u8,
    python_error: ?[]const u8,
};

/// Names Python's sandbox refuses and jinja.cpp serves: the engine gives jinja.cpp's result.
const jinja_cpp_only = [_][]const u8{ "filter strip", "method indent", "method append" };

/// A builtin case whose Zig render is not Python's, and why. Every listed case must still differ.
const builtin_differences = [_]struct { label: []const u8, why: []const u8 }{
    .{ .label = "filter lower #2", .why = "case mapping is ASCII-only, as jinja.cpp's" },
    .{ .label = "filter upper #2", .why = "case mapping is ASCII-only, as jinja.cpp's" },
    .{ .label = "filter wordcount #3", .why = "`\\w` past ASCII is approximated: symbols such as an arrow count as word characters, and Latin-1 numerics such as one half do not" },
    .{ .label = "filter string #2", .why = "repr's isprintable is approximated: an unassigned code point prints as itself, not as a \\u escape" },
    .{ .label = "operator + #2", .why = "arithmetic on an integer past 64 bits is a TemplateRuntime error, never a wrong number" },
    .{ .label = "filter int #2", .why = "an integer past 64 bits from int() is a TemplateRuntime error, never a wrong number" },
    .{ .label = "filter sort #4", .why = "a NaN sort key fails by name: the order Python gives it depends on its sort algorithm" },
    .{ .label = "operator slice #10", .why = "Jinja2 folds an all-literal slice at compile time through getitem, which makes one Python refuses undefined; the engine raises as Python does at run time" },
    .{ .label = "operator slice #11", .why = "Jinja2 folds an all-literal slice at compile time through getitem, which makes one Python refuses undefined; the engine raises as Python does at run time" },
    .{ .label = "method title #2", .why = "case mapping is ASCII-only: a non-ASCII letter does not join a word, so the letter after it starts one" },
    .{ .label = "test sameas #2", .why = "`sameas` is identity of the engine's values: equal literals are not one object as CPython's interned constants are" },
};

fn requiredBuiltinNames(arena: std.mem.Allocator) ![]const []const u8 {
    const builtins = @import("template/builtins.zig");
    const value = @import("template/value.zig");
    var names = std.ArrayList([]const u8).empty;
    inline for (.{ .{ "filter", builtins.Filter }, .{ "test", builtins.Test }, .{ "method", value.Method }, .{ "global", value.Global } }) |kind| {
        inline for (comptime std.meta.tags(kind[1])) |tag| try names.append(arena, kind[0] ++ " " ++ @tagName(tag));
    }
    for ([_][]const u8{ "index", "index0", "revindex", "revindex0", "first", "last", "length", "previtem", "nextitem" }) |f|
        try names.append(arena, try std.fmt.allocPrint(arena, "loop {s}", .{f}));
    for ([_][]const u8{ "in", "not in", "~", "==", "!=", "<", "<=", ">", ">=", "and", "or", "not", "+", "-", "*", "/", "%", "unary -", "unary +", "slice", "subscript", "if else", "literals" }) |o|
        try names.append(arena, try std.fmt.allocPrint(arena, "operator {s}", .{o}));
    for ([_][]const u8{ "if", "for", "break", "continue", "set", "set block", "macro", "call", "filter", "generation", "comment", "whitespace" }) |s|
        try names.append(arena, try std.fmt.allocPrint(arena, "statement {s}", .{s}));
    return names.items;
}

test "template builtins: every filter, test, method, global, loop field, operator and statement renders as Python Jinja2 does" {
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const doc = try std.json.parseFromSliceLeaky(struct { generator: []const u8, cases: []const BuiltinCase }, arena, @embedFile("fixtures/template_builtins.json"), .{});

    var uncovered: usize = 0;
    for (try requiredBuiltinNames(arena)) |name| {
        const hit = for (doc.cases) |c| {
            if (std.mem.eql(u8, c.covers, name)) break true;
        } else false;
        if (!hit) {
            uncovered += 1;
            std.debug.print("\nno builtin case covers '{s}': add one to tests/dump_template_oracle.py\n", .{name});
        }
    }

    var failed: usize = 0;
    var differ_seen: [builtin_differences.len]bool = @splat(false);
    for (doc.cases) |c| {
        var case_arena = std.heap.ArenaAllocator.init(a);
        defer case_arena.deinit();
        const ca = case_arena.allocator();
        var diag = zig_template.Diagnostic{};
        const got = zig_template.renderJson(ca, .{ .template = c.template, .messages_json = "[]", .tools_json = null, .extra_json = c.extra, .add_generation_prompt = false }, &diag);

        const cpp_only = for (jinja_cpp_only) |n| {
            if (std.mem.eql(u8, n, c.covers)) break true;
        } else false;
        var want: ?[]const u8 = c.python;
        var cpp_out: ?[*]u8 = null;
        defer if (cpp_out) |p| jinja_c.jinja_str_free(p);
        if (cpp_only) {
            var len: usize = 0;
            cpp_out = jinja_c.jinja_render_chat((try ca.dupeSentinel(u8, c.template, 0)).ptr, "[]", null, (try ca.dupeSentinel(u8, c.extra, 0)).ptr, 0, &len);
            want = if (cpp_out) |p| p[0..len] else null;
        }
        const same = if (got) |text| (want != null and std.mem.eql(u8, text, want.?)) else |e| blk: {
            if (cpp_only or c.python_error == null) break :blk false;
            const perr = c.python_error.?;
            if (std.mem.startsWith(u8, perr, "TemplateError: "))
                break :blk e == error.TemplateRaised and std.mem.eql(u8, perr["TemplateError: ".len..], diag.message());
            const syntax = std.mem.startsWith(u8, perr, "TemplateSyntaxError") or std.mem.startsWith(u8, perr, "TemplateAssertionError");
            break :blk e == if (syntax) error.TemplateSyntax else error.TemplateRuntime;
        };
        const listed = for (builtin_differences, 0..) |d, i| {
            if (std.mem.eql(u8, d.label, c.label)) break i;
        } else null;
        if (listed) |i| {
            differ_seen[i] = !same;
            if (!same) continue;
        } else if (same) continue;
        failed += 1;
        if (got) |text| {
            std.debug.print("\n[{s}] {s}\n  zig:    {s}\n  expect: {?s} {?s}\n", .{ c.label, if (listed != null) "is listed as a difference and renders the same now" else "", text, want, if (cpp_only) "(jinja.cpp)" else c.python_error });
        } else |e| std.debug.print("\n[{s}]\n  zig: {s} at {d}:{d}: {s}\n  expect: {?s} {?s}\n", .{ c.label, @errorName(e), diag.line, diag.column, diag.message(), want, c.python_error });
    }
    for (builtin_differences, differ_seen) |d, seen| if (!seen) {
        std.debug.print("\nlisted builtin difference not found or renders the same: {s}\n", .{d.label});
        failed += 1;
    };
    try testing.expectEqual(@as(usize, 0), uncovered);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "template oracle: a special spelled in an assistant turn is its token only if the model can write it in a turn" {
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var toks = std.ArrayList(*tokenizer_mod.Tokenizer).empty;
    defer for (toks.items) |t| tokenizer_mod.deinitTestTokenizer(t);
    const served = try servedTemplates(a, arena, &toks);
    // Per pack: what an assistant turn reads as text (a stop, or a flagged special the template never prints) and
    // what stays its token (a special the template prints, or one the pack does not flag).
    const Rule = struct { pack: []const u8, text: []const []const u8, token: []const []const u8 };
    const rules = [_]Rule{
        .{ .pack = "qwen38_flash_next", .text = &.{ "<|im_end|>", "<|endoftext|>", "<|box_start|>" }, .token = &.{ "<|im_start|>", "<tool_call>", "<think>" } },
        .{ .pack = "glm53", .text = &.{ "<|user|>", "<|observation|>", "<|assistant|>", "<|endoftext|>" }, .token = &.{ "<tool_call>", "/nothink", "<think>" } },
        .{ .pack = "dsv41", .text = &.{ "<｜end▁of▁sentence｜>", "<｜place▁holder▁no▁7｜>" }, .token = &.{ "<｜User｜>", "<｜tool▁calls▁begin｜>" } },
    };
    var checked: usize = 0;
    for (served) |s| {
        const rule = for (rules) |r| {
            if (std.mem.eql(u8, r.pack, s.id)) break r;
        } else continue;
        const pt = try tokenizer_mod.packTokenizerForTests(a, arena, s.id);
        const tok = try arena.create(tokenizer_mod.Tokenizer);
        tok.* = pt.tok;
        try toks.append(arena, tok);
        var cfg = s.config;
        try cfg.compile();
        defer cfg.dropCompiled();
        try tok.bindTemplate(cfg.printedFacts(), pt.stops);
        for ([_][]const []const u8{ rule.text, rule.token }, [_]bool{ false, true }) |specials, is_token| for (specials) |special| {
            const id = tok.specialTokenId(special).?;
            for ([_][]const u8{ "assistant", "user" }) |role| {
                const with = try std.mem.concat(arena, u8, &.{ "say ", special, " now" });
                var counts: [2]usize = undefined;
                for ([_][]const u8{ "say  now", with }, &counts) |content, *n| {
                    const turn = [_]Message{
                        .{ .role = "user", .content = if (std.mem.eql(u8, role, "user")) content else "hi" },
                        .{ .role = "assistant", .content = if (std.mem.eql(u8, role, "assistant")) content else "hello" },
                        .{ .role = "user", .content = "ok" },
                    };
                    const ids = try chat.formatChat(a, tok, &turn, &cfg, null, null, true, null, false);
                    defer a.free(ids);
                    n.* = std.mem.count(u32, ids, &.{id});
                }
                const want: usize = if (is_token and std.mem.eql(u8, role, "assistant")) 1 else 0;
                testing.expectEqual(want, counts[1] - counts[0]) catch |e| {
                    std.debug.print("{s}: {s} spelled in a {s} turn\n", .{ s.id, special, role });
                    return e;
                };
                checked += 1;
            }
        };
    }
    try testing.expect(checked > 0);
}

test "template oracle: a prompt encoded through the segment cache is the prompt encoded without it, from eight threads at once" {
    const a = std.heap.page_allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var toks = std.ArrayList(*tokenizer_mod.Tokenizer).empty;
    defer for (toks.items) |t| tokenizer_mod.deinitTestTokenizer(t);
    const served = try servedTemplates(a, arena, &toks);
    const requests = try corpusRequests(arena);
    const Call = struct { r: Request, arm: Arm, want: []const u32 };
    const segment_cache = @import("segment_cache.zig");
    var checked: usize = 0;
    for (served) |s| {
        if (s.tok == null) continue;
        const pt = try tokenizer_mod.packTokenizerForTests(a, arena, s.id);
        const tok = try arena.create(tokenizer_mod.Tokenizer);
        tok.* = pt.tok;
        try toks.append(arena, tok);
        var cfg = s.config;
        try cfg.compile();
        defer cfg.dropCompiled();
        try tok.bindTemplate(cfg.printedFacts(), pt.stops);
        // Every prompt of the corpus, encoded without a cache.
        var prompts = std.ArrayList(Call).empty;
        for (requests) |r| for (r.arms) |arm| {
            var c2 = cfg;
            c2.preserve_thinking = arm.preserve;
            const ids = chat.formatChat(arena, tok, r.messages, &c2, r.tools, r.instr, arm.thinking, arm.effort, r.continue_final) catch |e| switch (e) {
                error.ChatNeedsUserTurn, error.ChatTemplateUnusable => continue,
                else => return e,
            };
            try prompts.append(arena, .{ .r = r, .arm = arm, .want = ids });
        };
        // The default budget, and one small enough that the threads evict each other's segments.
        for ([_]usize{ segment_cache.default_budget, 64 << 10 }) |budget| {
            try tok.enableSegmentCache(budget);
            const Worker = struct {
                fn run(t: *const tokenizer_mod.Tokenizer, config: *const chat.ChatConfig, list: []const Call, ok: *bool) void {
                    for (list) |c| {
                        var c2 = config.*;
                        c2.preserve_thinking = c.arm.preserve;
                        const ids = chat.formatChat(std.heap.page_allocator, t, c.r.messages, &c2, c.r.tools, c.r.instr, c.arm.thinking, c.arm.effort, c.r.continue_final) catch return;
                        defer std.heap.page_allocator.free(ids);
                        if (!std.mem.eql(u32, ids, c.want)) return;
                    }
                    ok.* = true;
                }
            };
            // Cold and warm on one thread, then eight at once.
            for (0..2) |_| {
                var ok = false;
                Worker.run(tok, &cfg, prompts.items, &ok);
                try testing.expect(ok);
            }
            var oks: [8]bool = @splat(false);
            var threads: [8]std.Thread = undefined;
            for (&threads, &oks) |*t, *ok| t.* = try std.Thread.spawn(.{}, Worker.run, .{ tok, &cfg, prompts.items, ok });
            for (threads) |t| t.join();
            for (oks) |ok| try testing.expect(ok);
            try testing.expect(tok.segment_cache.?.bytes() <= budget);
        }
        checked += prompts.items.len;
    }
    try testing.expect(checked > 0);
}
