//! What a generation's text means to every reply surface, decided in one place: whether it holds tool calls and the
//! finish that goes with them. Chat, Anthropic and Responses, streamed or not, ask here.
const std = @import("std");
const think_tags = @import("think_tags.zig");
const tool_calls_mod = @import("tool_calls.zig");
const tool_schema = @import("tool_schema.zig");
const server_state = @import("server_state.zig");
const log = @import("log.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const listed_tokens = @import("listed_tokens.zig");

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when calls the model finished before the cut
/// are in the reply — clients key their truncation recovery on it. Overriding
/// to "tool_calls" hides the cut (an open call is declined, `settle`, so the
/// client learns the model was stopped mid-call only from "length").
pub fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// A repetition-loop cut may land inside an otherwise recognizable tool call.
/// Never emit that buffer as executable work: unlike a genuine max-token cut,
/// this intentional stop must not ask clients to recover by compacting/retrying.
pub fn shouldParseToolCalls(finish_details: ?[]const u8) bool {
    const d = finish_details orelse return true;
    return !std.mem.eql(u8, d, "repetition_loop") and !std.mem.eql(u8, d, thought_loop);
}

/// `finish_details` of a reply whose turn ended inside its thought (`server.continue_after_thought`).
pub const thought_continued = "continued_after_thought";
/// The continuation then stopped on a repetition loop.
pub const thought_loop = "repetition_loop_after_thought";

/// Why a reply whose turn ended inside its thought was not continued.
pub const ThoughtKept = enum {
    /// max_tokens (or the context the request was billed) leaves no token for the continuation.
    max_tokens,
    /// The template writes no thought for the turn (DeepSeek's chat mode): its continuation would drop it.
    template_drops_thought,
    /// The template could not render the continuation.
    render_failed,
    /// The request carries images: the continuation does not encode them again.
    images,
    /// The request asked for logprobs, which describe one generation.
    logprobs,
    /// A JSON schema constrains the reply.
    json_schema,
    /// The request is itself a continuation of an assistant message.
    continuation_request,

    /// The `finish_details` value naming it.
    pub fn details(self: ThoughtKept) []const u8 {
        return switch (self) {
            inline else => |k| "thought_not_continued:" ++ @tagName(k),
        };
    }
};

/// The `finish_details` object (with no key) a details value names, or null for none this module defines.
pub fn thoughtDetailsJson(details: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, details, thought_continued)) return "{\"type\":\"continued_after_thought\",\"message\":\"continued after the thought ended the turn\"}";
    if (std.mem.eql(u8, details, thought_loop)) return "{\"type\":\"repetition_loop\",\"continued_after_thought\":true}";
    inline for (comptime std.enums.values(ThoughtKept)) |k| {
        if (std.mem.eql(u8, details, k.details())) return "{\"type\":\"thought_ended_turn\",\"not_continued\":\"" ++ @tagName(k) ++ "\"}";
    }
    return null;
}

/// The thought's details object as a field named `key`, with its leading comma, or "" when `details` names none: the
/// surfaces whose finish carries no `finish_details` (Anthropic's `stop_details`, the Responses object).
pub fn thoughtDetailsField(comptime key: []const u8, details: ?[]const u8) []const u8 {
    const d = details orelse return "";
    inline for ([_][]const u8{ thought_continued, thought_loop } ++ kept_details) |v| {
        if (std.mem.eql(u8, d, v)) return ",\"" ++ key ++ "\":" ++ comptime thoughtDetailsJson(v).?;
    }
    return "";
}

const kept_details = blk: {
    const kept = std.enums.values(ThoughtKept);
    var out: [kept.len][]const u8 = undefined;
    for (kept, 0..) |k, i| out[i] = k.details();
    break :blk out;
};

/// The thought of a reply that holds only its thought (no visible text, no call), trimmed and owned, or null.
pub fn soleThought(allocator: std.mem.Allocator, text: []const u8, ask: Ask) !?[]u8 {
    const settled = try settle(allocator, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
    defer settled.deinit(allocator);
    if (settled.reply != null) return null;
    const split = think_tags.splitThinkBlock(settled.text, true, ask.opens_think);
    const thought = std.mem.trim(u8, split.reasoning_content orelse return null, " \t\r\n");
    if (thought.len == 0 or std.mem.trim(u8, split.content, " \t\r\n").len > 0) return null;
    return try allocator.dupe(u8, thought);
}

/// How the text came to be.
pub const Origin = struct {
    /// The served template's call dialect (`chat.callDialect`), read first; null reads the families in the legacy order.
    dialect: ?tool_calls_mod.Dialect,
    /// The prompt ended inside a think block (`promptOpensThink`): text with no closer is reasoning.
    opens_think: bool = false,
    /// A stop sequence, the length cap or a cancel ended the generation, not the model: markup it left open is no call.
    cut: bool = false,
};

/// A reply that holds calls: the text before them and the calls, as every surface lists them.
pub const ToolReply = struct {
    /// The text before the markup, byte for byte as a stream flushes it; empty when there is none.
    preamble: []const u8,
    calls: []tool_schema.ParsedToolCall,

    pub fn deinit(self: ToolReply, allocator: std.mem.Allocator) void {
        for (self.calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(self.calls);
    }
};

/// The calls of a generation and the text before them, or null when it holds none. A call is a call wherever it
/// stands in the generation: one the model wrote inside a thought it never closed ships too, and its markup is cut
/// out of the reasoning (`splitThinkBlock`). `text` outlives the result: the preamble slices it.
pub fn toolReply(allocator: std.mem.Allocator, text: []const u8, tools_json: ?[]const u8, allow_parallel: bool, origin: Origin) !?ToolReply {
    const calls = (try parseToolCallsForRequest(allocator, text, tools_json, allow_parallel, origin)) orelse return null;
    const split = think_tags.splitThinkBlock(text, true, origin.opens_think);
    return .{ .preamble = think_tags.visibleToolPreamble(split.content), .calls = calls };
}

/// A generation as a surface got it.
pub const Generated = struct {
    text: []const u8,
    /// Where the stop that ended it is read from (`stop_sequences.cut`).
    source: stop_seq_mod.Source,
    finish_reason: []const u8,
    finish_details: ?[]const u8,
};

/// What the request asks of the end handling.
pub const Ask = struct {
    stops: []const []const u8,
    /// Merge re-opened mid-text thought channels into the leading block so no parse below leaks raw tags (Gemma 12B
    /// re-opens channels mid-turn). A span-routed constrained response sets false: its payload is authoritative and
    /// marker text inside it is DATA.
    normalize: bool,
    has_tools: bool,
    tools_json: ?[]const u8,
    allow_parallel: bool,
    dialect: ?tool_calls_mod.Dialect,
    opens_think: bool,
};

/// How a generation ended, decided once for every surface, streamed or not.
pub const Settled = struct {
    /// What the model said, cut at the stop that ended it and normalized: the text every later split reads.
    text: []const u8,
    /// The stop that ended the generation.
    stop: ?[]const u8,
    /// "stop" when a stop ended it, else the generation's own; a surface maps it with `toolCallFinishReason` when `reply` holds calls.
    finish_reason: []const u8,
    reply: ?ToolReply,
    normalized: ?[]u8,

    pub fn deinit(self: Settled, allocator: std.mem.Allocator) void {
        if (self.reply) |r| r.deinit(allocator);
        if (self.normalized) |n| allocator.free(n);
    }
};

/// The one end handling of a reply: the stop that ended the generation, the text it leaves, the calls that text
/// holds and the finish a stop leaves. A call a stop, the length cap or a cancel left open is not a call (`parseToolCallsForRequest`).
pub fn settle(allocator: std.mem.Allocator, gen: Generated, ask: Ask) !Settled {
    const cut = stop_seq_mod.cut(gen.text, ask.stops, gen.source);
    var out: Settled = .{
        .text = cut.text,
        .stop = cut.stop,
        .finish_reason = if (cut.stop != null) "stop" else gen.finish_reason,
        .reply = null,
        .normalized = null,
    };
    errdefer out.deinit(allocator);
    if (ask.normalize) {
        out.normalized = try think_tags.normalizeEmbeddedThinkBlocks(allocator, cut.text);
        if (out.normalized) |n| out.text = n;
    }
    if (ask.has_tools and shouldParseToolCalls(gen.finish_details)) {
        // The model's own end is the stop finish; any other (length, cancelled) cut it short.
        const model_ended = cut.stop == null and std.mem.eql(u8, gen.finish_reason, "stop");
        out.reply = try toolReply(allocator, out.text, ask.tools_json, ask.allow_parallel, .{ .dialect = ask.dialect, .opens_think = ask.opens_think, .cut = !model_ended });
    }
    return out;
}

/// The ONE tool-call extraction path for every HTTP surface: parse, fall back to
/// bare-JSON inference, then coerce the arguments to the schema the request
/// declared. Every surface must go through this — a site that calls
/// `tool_calls.parseToolCalls` directly silently reintroduces the value-spelling
/// type-inference bug (a boolean param arriving as the string "False") on that
/// surface alone, which no output-equality test can see.
pub fn parseToolCallsForRequest(
    allocator: std.mem.Allocator,
    text: []const u8,
    tools_json: ?[]const u8,
    /// OpenAI `parallel_tool_calls` (Anthropic spelling:
    /// !tool_choice.disable_parallel_tool_use). false = the client accepts AT
    /// MOST ONE call per response, so the chokepoint keeps the first parsed
    /// call and drops the rest (the model re-issues them next round after the
    /// first result). Deliberately NOT gated by --no-tool-autocorrect:
    /// client-requested behavior, not a repair heuristic.
    allow_parallel: bool,
    origin: Origin,
) !?[]tool_schema.ParsedToolCall {
    // An unfinished call is never shipped, whatever ended the generation: the calls the model finished before it stay.
    var body = text;
    if (origin.cut) if (think_tags.openToolCallStart(text)) |open| {
        tool_schema.noteUnparsedToolMarkup(text);
        body = text[0..open];
    };
    return tool_schema.parseForRequest(allocator, body, tools_json, .{
        .allow_parallel = allow_parallel,
        .autocorrect = server_state.g_tool_autocorrect,
        .dialect = origin.dialect,
    });
}

test "parseToolCallsForRequest coerces args to the schema (server-side chokepoint wiring)" {
    // The chokepoint every HTTP surface routes through must actually invoke the
    // schema coercion — not just parse. This pins the WIRING (a mis-wire that
    // parsed but skipped coerce would reintroduce the string-"False" bug on every
    // surface at once). Uses the exact Hermes-XML `replace_all=False` shape from
    // the live Claude Code failure.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .bool); // coerced from the STRING "False"
    try std.testing.expectEqual(false, ra.bool);
    // Null tools_json path must still parse (no coercion, no crash).
    const calls2 = try parseToolCallsForRequest(allocator, text, null, true, .{ .dialect = null });
    defer if (calls2) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls2 != null);
}

test "parseToolCallsForRequest hoists a misplaced required param (server-side chokepoint wiring)" {
    // Live 2026-07-13 pi session, gemma-4-26B-A4B: the model buried `path` inside
    // each edits[] item and emitted none at the top level, so pi rejected three
    // consecutive multi-thousand-token generations with
    //   Validation failed for tool "edit": - path: must have required properties path
    // This pins the WIRING — every HTTP surface reaches the hoist through this one
    // function, and the hoist must run BEFORE coercion so the lifted value is
    // type-checked like any other top-level arg.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]}}},"required":["path","edits"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"edit\",\"arguments\":{\"edits\":[{\"newText\":\"new\",\"oldText\":\"old\"," ++
        "\"path\":\"us_presidents/generate_site.sh\"}]}}</tool_call>";

    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try std.testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    try std.testing.expect(parsed.value.object.get("edits").?.array.items[0].object.get("path") == null);
    try std.testing.expect(tool_schema.toolCallConformsToSchema(allocator, calls[0], tools));

    // The escape hatch covers this correction too: it repairs the MODEL's output
    // (unlike the inferred-name filter, which corrects our own heuristic).
    server_state.g_tool_autocorrect = false;
    defer server_state.g_tool_autocorrect = true;
    const raw_calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })).?;
    defer {
        for (raw_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(raw_calls);
    }
    const raw_parsed = try std.json.parseFromSlice(std.json.Value, allocator, raw_calls[0].arguments, .{});
    defer raw_parsed.deinit();
    try std.testing.expect(raw_parsed.value.object.get("path") == null); // verbatim: still buried
}

test "parseToolCallsForRequest: --no-tool-autocorrect leaves args verbatim" {
    // The escape hatch: with g_tool_autocorrect off, the schema coercion is
    // skipped and the model's value passes through EXACTLY (here the STRING
    // "False", which is what a strict client would then reject — the user's
    // choice). Parse-repair + valid-JSON safety net still run, so args stay
    // well-formed JSON regardless.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";

    server_state.g_tool_autocorrect = false;
    defer server_state.g_tool_autocorrect = true; // restore the default for other tests
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const ra = parsed.value.object.get("replace_all").?;
    try std.testing.expect(ra == .string); // NOT coerced — verbatim
    try std.testing.expectEqualStrings("False", ra.string);
}

test "parseToolCallsForRequest: truncated DATA object is not promoted to a tool call (George Washington class)" {
    // Live pi capture 2026-07-13 (Qwen3.6-35B-A3B distilled): generation hit
    // max_tokens midway through a presidents data script. The raw-JSON fallback
    // found the first balanced object — {"name": "George Washington", "num": 1,
    // …} — and the flat-shape synthesis promoted it to a TOOL CALL named
    // "George Washington" (args = every key but "name"). pi answered "Tool
    // George Washington not found", the model retried the identical mega-write,
    // and the session burned two full 16K-token turns making zero progress.
    // A HEURISTICALLY inferred call (no tag syntax — the model never said
    // "tool call") must carry a name the request actually declared; otherwise
    // the text stays visible and finish_reason="length" reaches the client
    // untouched, so its truncation recovery fires instead of a bogus tool loop.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "Now let me build a generator script that creates all 46 president pages with real historical data:\n\n" ++
        "presidents = [\n" ++
        "  {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None (Federalist-leaning)\", \"term\": \"1789\u{2013}1797\", \"vice\": \"John Adams\"},\n" ++
        "  {\"name\": \"John Adams\", \"num\": 2, \"party\": \"Federalist\",";
    const calls = try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null });
    defer if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    };
    try std.testing.expect(calls == null);
}

test "parseToolCallsForRequest: tool markup that parses to no call leaves one log line, prose leaves none" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const log_path = try std.fmt.allocPrint(allocator, "{s}/s.log", .{root});
    defer allocator.free(log_path);
    const tools =
        \\[{"type":"function","function":{"name":"bash","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    // A stray token inside the invoke tag: the block names no call.
    const wrecked = "I'll list the files.\n\n<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}inv\u{793c}oke name=\"bash\">\n" ++
        "<\u{ff5c}DSML\u{ff5c}parameter name=\"command\" string=\"true\">ls</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>";
    try log.openFile(log_path, 0);
    const none = parseToolCallsForRequest(allocator, wrecked, tools, true, .{ .dialect = null }) catch |e| {
        log.closeFile();
        return e;
    };
    const prose = parseToolCallsForRequest(allocator, "The files are listed above.", tools, true, .{ .dialect = null }) catch |e| {
        log.closeFile();
        return e;
    };
    log.closeFile();
    try std.testing.expect(none == null);
    try std.testing.expect(prose == null);
    const logged = try tmp.dir.readFileAlloc(io, "s.log", allocator, .limited(1 << 16));
    defer allocator.free(logged);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, logged, "[tool-parse]"));
    try std.testing.expect(std.mem.indexOf(u8, logged, "inv\u{793c}oke name=") != null);
    try std.testing.expect(std.mem.indexOf(u8, logged, "from byte 22,") != null);
}

test "parseToolCallsForRequest: raw-JSON call with a DECLARED name still parses" {
    // The counterweight to the George Washington guard: models without a
    // trained tool format (Gemma 3) emit fenced raw-JSON calls, and those must
    // keep working when the name matches a declared tool.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    ;
    const text = "```json\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.txt\", \"content\": \"hi\"}}\n```";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("write", calls[0].name);
}

test "parseToolCallsForRequest: tag-format call with an UNDECLARED name is kept" {
    // An EXPLICIT tag-format call (<tool_call>…) to a name the request never
    // declared still goes to the client — "tool not found" is model-visible
    // feedback the model can correct from. Only HEURISTIC raw-JSON inference
    // gets schema-name validation; this pins the filter against over-reach.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    const text = "<tool_call>{\"name\":\"searchWeb\",\"arguments\":{\"q\":\"zig\"}}</tool_call>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("searchWeb", calls[0].name);
}

test "parseToolCallsForRequest: parallel_tool_calls=false clamps to the FIRST call" {
    // OpenAI contract: parallel_tool_calls=false means AT MOST ONE call per
    // response (the SDK sets it in strict structured-output mode; those
    // clients execute tool_calls[0] and assume length 1). We can't constrain
    // generation, so the chokepoint clamps post-parse: keep the first, drop
    // the rest — the model re-issues them next round after seeing the first
    // result. Whole valid calls only; nothing partial ships.
    const allocator = std.testing.allocator;
    const text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>\n" ++
        "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":\"b\"}}</tool_call>";

    // allow_parallel=true (default wiring): both calls survive.
    const both = (try parseToolCallsForRequest(allocator, text, null, true, .{ .dialect = null })).?;
    defer {
        for (both) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(both);
    }
    try std.testing.expectEqual(@as(usize, 2), both.len);

    // allow_parallel=false: only the FIRST survives, intact.
    const one = (try parseToolCallsForRequest(allocator, text, null, false, .{ .dialect = null })).?;
    defer {
        for (one) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(one);
    }
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("read", one[0].name);

    // A single call with the flag off passes through untouched.
    const single_text = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"a\"}}</tool_call>";
    const single = (try parseToolCallsForRequest(allocator, single_text, null, false, .{ .dialect = null })).?;
    defer {
        for (single) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(single);
    }
    try std.testing.expectEqual(@as(usize, 1), single.len);
}

test "parseToolCallsForRequest: MiniCPM5 V3 XML through the real server chokepoint" {
    // Exercises the EXACT function every HTTP dispatch site in this file calls,
    // proving the new dialect is wired all the way through — not just reachable
    // from tool_calls.parseToolCalls in isolation. Uses the Agent shell tool's real
    // declared schema shape.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
    ;
    const text = "<function name=\"shell\">\n  <param name=\"command\">git status</param>\n</function>";
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{ .dialect = null })) orelse
        return error.ExpectedToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try std.testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("git status", parsed.value.object.get("command").?.string);
}

test "parseToolCallsForRequest: coercion fires across think on/off × qwen/gemma" {
    // The tool-call parse strips the leading think block BEFORE parsing args, and
    // the two families strip DIFFERENTLY (Qwen `<think>…</think>`, Gemma
    // `<|channel>thought…<channel|>`). Pin that a preceding reasoning block never
    // blocks the schema coercion, for BOTH families, WITH and WITHOUT thinking.
    const allocator = std.testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;
    const gtools =
        \\[{"type":"function","function":{"name":"Edit","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["file_path"]}}}]
    ;

    const Case = struct { name: []const u8, text: []const u8, tools: []const u8 };
    const cases = [_]Case{
        // Qwen3.6-MoE thinking OFF — Hermes XML, no reasoning block.
        .{ .name = "qwen think-off", .tools = tools, .text = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Qwen3.6-MoE thinking ON — a `<think>…</think>` block precedes the call.
        .{ .name = "qwen think-on", .tools = tools, .text = "<think>\nThe user wants replace_all disabled, so I'll pass it false.\n</think>\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>" },
        // Gemma thinking OFF — custom `call:` form, no channel block.
        .{ .name = "gemma think-off", .tools = gtools, .text = "<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
        // Gemma thinking ON — a `<|channel>thought…<channel|>` block precedes.
        .{ .name = "gemma think-on", .tools = gtools, .text = "<|channel>thought\nI should disable replace_all for a single edit.<channel|>\n<|tool_call>call:Edit{file_path:<|\"|>/tmp/x<|\"|>,replace_all:False}<tool_call|>" },
    };

    for (cases) |c| {
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools, true, .{ .dialect = null })) orelse {
            std.debug.print("\n[{s}] no tool call parsed\n", .{c.name});
            return error.NoToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try std.testing.expectEqualStrings("Edit", calls[0].name);
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        const ra = parsed.value.object.get("replace_all") orelse {
            std.debug.print("\n[{s}] replace_all missing in {s}\n", .{ c.name, calls[0].arguments });
            return error.ArgMissing;
        };
        if (ra != .bool or ra.bool != false) {
            std.debug.print("\n[{s}] replace_all NOT coerced to bool false: {s}\n", .{ c.name, calls[0].arguments });
            return error.NotCoerced;
        }
    }
}

test "toolCallFinishReason preserves truncation over parsed tool calls" {
    // A truncated generation ("length": max_tokens OR request timeout) whose
    // cut-off text still salvaged a partial tool call must KEEP reporting
    // "length" — clients key their truncation recovery (chunk-and-retry
    // nudges) on it. Reporting "tool_calls" hides the cut: the client
    // executes a half-argument call and tells the model IT made a mistake.
    // Live capture 2026-07-03: Qwen3.6-27B 33KB writeFile guillotined by the
    // 300s default timeout arrived as {"path":...} with finish "tool_calls".
    try std.testing.expectEqualStrings("length", toolCallFinishReason("length"));
    // Everything else stays the normal tool-call finish.
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("stop"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("tool_calls"));
    try std.testing.expectEqualStrings("tool_calls", toolCallFinishReason("client_disconnect"));
}

test "repetition-loop cuts decline tool-call parsing" {
    // The loop detector can fire inside a tool argument. The response must
    // remain a safe terminal stop without emitting an executable fragment.
    try std.testing.expect(shouldParseToolCalls(null));
    try std.testing.expect(!shouldParseToolCalls("repetition_loop"));
    try std.testing.expect(shouldParseToolCalls("unknown_future_detail"));
}

const testing = std.testing;

const test_tools =
    \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    \\{"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
;

fn expectCalls(text: []const u8, origin: Origin, want: usize) !void {
    const calls = try parseToolCallsForRequest(testing.allocator, text, test_tools, true, origin);
    defer if (calls) |cs| {
        for (cs) |c| {
            testing.allocator.free(c.name);
            testing.allocator.free(c.arguments);
        }
        testing.allocator.free(cs);
    };
    errdefer std.debug.print("expected {d} calls, got {d} for {f}\n", .{ want, if (calls) |cs| cs.len else 0, std.zig.fmtString(text) });
    try testing.expectEqual(want, if (calls) |cs| cs.len else 0);
}

test "a call inside a thought the prompt opened and the model never closed is a call, cut out of the reasoning" {
    const qwen = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const glm = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>";
    for ([_][]const u8{ qwen, glm }) |call| {
        const open = try std.mem.concat(testing.allocator, u8, &.{ "I'll call it.\n", call });
        defer testing.allocator.free(open);
        const after = try std.mem.concat(testing.allocator, u8, &.{ "Plan.\n</think>\n\n", call });
        defer testing.allocator.free(after);
        const inside = try std.mem.concat(testing.allocator, u8, &.{ "Maybe ", call, " no.\n</think>\n\nIt is sunny." });
        defer testing.allocator.free(inside);
        try expectCalls(open, .{ .dialect = null, .opens_think = true }, 1);
        try expectCalls(after, .{ .dialect = null, .opens_think = true }, 1);
        // A call inside a thought that closed stays skipped.
        try expectCalls(inside, .{ .dialect = null, .opens_think = true }, 0);

        const reply = (try toolReply(testing.allocator, open, test_tools, true, .{ .dialect = null, .opens_think = true })).?;
        defer reply.deinit(testing.allocator);
        try testing.expectEqualStrings("get_weather", reply.calls[0].name);
        try testing.expectEqualStrings("", reply.preamble);
        const split = think_tags.splitThinkBlock(open, true, true);
        try testing.expectEqualStrings("I'll call it.", split.reasoning_content.?);
    }
}

test "a generation cut inside a call (a stop, the length cap, a cancel) leaves no call, in every dialect" {
    const cut = [_][]const u8{
        "Plan.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nline one\n",
        "Plan.</think><tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>line one\n",
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"line one\n",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">notes.md</｜DSML｜parameter>\n<｜DSML｜parameter name=\"content\" string=\"true\">line one\n",
        "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>line one\n",
    };
    for (cut) |text| {
        try expectCalls(text, .{ .dialect = null, .cut = true }, 0);
        // The model's own end inside a call keeps the salvage it had.
        errdefer std.debug.print("salvage changed for {f}\n", .{std.zig.fmtString(text)});
        try expectCalls(text, .{ .dialect = null }, 1);
    }
}

test "a cut after a call closed leaves the call" {
    const whole = [_][]const u8{
        "Let me look.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    };
    for (whole) |text| try expectCalls(text, .{ .dialect = null, .cut = true }, 1);
}

test "a call the model finished before a cut stays and the open one after it is declined, in every dialect" {
    const Pair = struct { done: []const u8, open: []const u8 };
    const pairs = [_]Pair{
        .{ .done = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>", .open = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nline one\n" },
        .{ .done = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>", .open = "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>line one\n" },
        .{ .done = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>", .open = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">notes.md</｜DSML｜parameter>\n<｜DSML｜parameter name=\"content\" string=\"true\">line one\n" },
        .{ .done = "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>", .open = "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>line one\n" },
    };
    for (pairs) |p| {
        const text = try std.mem.concat(testing.allocator, u8, &.{ p.done, "\n", p.open });
        defer testing.allocator.free(text);
        const calls = (try parseToolCallsForRequest(testing.allocator, text, test_tools, true, .{ .dialect = null, .cut = true })) orelse return error.FinishedCallDropped;
        defer {
            for (calls) |c| {
                testing.allocator.free(c.name);
                testing.allocator.free(c.arguments);
            }
            testing.allocator.free(calls);
        }
        errdefer std.debug.print("calls kept for {f}\n", .{std.zig.fmtString(text)});
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("get_weather", calls[0].name);
    }
}

/// One generated text as a stream hands it to a surface (its gate delivers the text and reports the match) and as a
/// non-stream reply does (the whole text, the stop still in it), settled under the same ask.
fn expectSameEnd(allocator: std.mem.Allocator, generated: []const u8, stops: []const []const u8, gen_finish: []const u8, ask: Ask, want_stop: ?[]const u8, want_finish: []const u8, want_calls: ?usize) !void {
    var frags = std.ArrayList([]const u8).empty;
    defer frags.deinit(allocator);
    var at: usize = 0;
    while (at < generated.len) : (at += 5) try frags.append(allocator, generated[at..@min(at + 5, generated.len)]);
    var matched: ?[]const u8 = null;
    const delivered = try stop_seq_mod.streamDelivered(allocator, frags.items, stops, &matched);
    defer allocator.free(delivered);

    // What each path's surface passes: the stream ends "stop" when its gate matched, the non-stream keeps the slot's own.
    const streamed = try settle(allocator, .{ .text = delivered, .source = .{ .gated = matched }, .finish_reason = if (matched != null) "stop" else gen_finish, .finish_details = null }, ask);
    defer streamed.deinit(allocator);
    const whole = try settle(allocator, .{ .text = generated, .source = .whole, .finish_reason = gen_finish, .finish_details = null }, ask);
    defer whole.deinit(allocator);

    for ([_]Settled{ streamed, whole }) |got| {
        if (want_stop) |w| try testing.expectEqualStrings(w, got.stop.?) else try testing.expect(got.stop == null);
        try testing.expectEqualStrings(want_finish, got.finish_reason);
        try testing.expectEqual(want_calls, if (got.reply) |r| @as(?usize, r.calls.len) else null);
    }
    try testing.expectEqualStrings(streamed.text, whole.text);
    if (streamed.reply) |s| {
        const w = whole.reply.?;
        try testing.expectEqualStrings(s.preamble, w.preamble);
        try testing.expectEqual(s.calls.len, w.calls.len);
        for (s.calls, w.calls) |a, b| {
            try testing.expectEqualStrings(a.name, b.name);
            try testing.expectEqualStrings(a.arguments, b.arguments);
        }
    }
}

/// The tokens a stream lists logprobs for, one token per fragment: its gate releases the text, a content chunk goes out
/// when the think split of the delivered text grows, and the final chunk lists what lagged behind (the newest entry
/// has not landed when its chunk goes out). The content arm withholds a content's trailing whitespace; it ships at
/// the end of the reply when the non-stream content reaches the end of the text (`tail_ships`), else it is cut.
fn streamListed(allocator: std.mem.Allocator, frags: []const []const u8, stops: []const []const u8, opens_think: bool, tail_ships: bool, listed: []bool, cut: *listed_tokens.Cut) !void {
    // A tools stream buffers from the first tool markup on and sends no content after it (`ToolHoldScan`).
    var tool_scan: think_tags.ToolHoldScan = .{};
    var listing: listed_tokens.Listing = .{};
    defer listing.deinit(allocator);
    var gate = stop_seq_mod.Gate{ .stops = stops };
    defer gate.deinit(allocator);
    var delivered = std.ArrayList(u8).empty;
    defer delivered.deinit(allocator);
    var sent_content: usize = 0;
    var tail = false;
    var i: usize = 0;
    while (true) : (i += 1) {
        const piece = if (i < frags.len) blk: {
            try listing.note(allocator, frags[i].len);
            break :blk try gate.push(allocator, frags[i]);
        } else if (gate.matched == null and gate.hasHeld() and !tail) blk: {
            tail = true;
            break :blk try gate.finish(allocator);
        } else break;
        defer allocator.free(piece);
        listing.released = gate.released;
        listing.ended = listing.ended or gate.matched != null or tail;
        try delivered.appendSlice(allocator, piece);
        // The content arm holds a tail that may still grow into a think opener.
        const content = if (think_tags.endsWithPartialThinkOpen(delivered.items) or think_tags.streamShouldBufferForToolsScan(delivered.items, &tool_scan)) delivered.items[0..0] else think_tags.splitThinkBlock(delivered.items, true, opens_think).content;
        if (content.len > sent_content) {
            listing.skipTo(@intFromPtr(content.ptr) - @intFromPtr(delivered.items.ptr));
            sent_content = content.len;
            const content_end = @intFromPtr(content.ptr) - @intFromPtr(delivered.items.ptr) + content.len;
            const withheld = listing.released -| content_end + (content.len - std.mem.trimEnd(u8, content, "\n ").len);
            noteListed(listing.sent(@min(i, frags.len), withheld), listed, cut);
        }
        if (gate.matched != null) break;
    }
    listing.ended = true;
    if (tail_ships) noteListed(listing.sent(frags.len, 0), listed, cut);
    noteListed(listing.late(frags.len), listed, cut);
}

/// A drained range joins the listing: the first token listed carries the head cut, the last the tail cut.
fn noteListed(r: listed_tokens.Range, listed: []bool, cut: *listed_tokens.Cut) void {
    if (r.end <= r.start) return;
    var first = true;
    for (listed) |l| first = first and !l;
    if (first) cut.head = r.head;
    cut.tail = r.tail;
    for (r.start..r.end) |k| listed[k] = true;
}

/// A stream and a non-stream reply list logprobs for the same tokens: those whose bytes the reply returned.
fn expectSameListing(allocator: std.mem.Allocator, generated: []const u8, frag_len: usize, stops: []const []const u8, ask_in: Ask) !void {
    var ask = ask_in;
    ask.stops = stops;
    var frags = std.ArrayList([]const u8).empty;
    defer frags.deinit(allocator);
    var at: usize = 0;
    while (at < generated.len) : (at += frag_len) try frags.append(allocator, generated[at..@min(at + frag_len, generated.len)]);
    const got = try allocator.alloc(bool, frags.items.len);
    defer allocator.free(got);
    @memset(got, false);
    const whole = try settle(allocator, .{ .text = generated, .source = .whole, .finish_reason = "length", .finish_details = null }, ask);
    defer whole.deinit(allocator);
    var listing: listed_tokens.Listing = .{};
    defer listing.deinit(allocator);
    for (frags.items) |f| try listing.note(allocator, f.len);
    // A reply with calls lists its preamble: the message's content, never a call's tokens.
    const content = if (whole.reply) |r| r.preamble else think_tags.splitThinkBlock(whole.text, true, ask.opens_think).content;
    const from = if (content.len > 0) @intFromPtr(content.ptr) - @intFromPtr(whole.text.ptr) else 0;
    const r = listing.touching(from, from + content.len);
    var streamed_cut: listed_tokens.Cut = .{};
    try streamListed(allocator, frags.items, stops, ask.opens_think, content.len > 0 and from + content.len == whole.text.len, got, &streamed_cut);
    for (got, 0..) |g, k| try testing.expectEqual(k >= r.start and k < r.end, g);
    // A token the reply returned part of lists those bytes on both.
    try testing.expectEqual(r.cut(), streamed_cut);
}

test "a stream and a non-stream reply end the same: a stop in a call, a stop after one, a length end, a cancel" {
    const a = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const open = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTok";
    const stops = [_][]const u8{"STOP"};
    // Chat and Messages ask alike; Responses asks the same, or skips the normalization of a constrained reply; a
    // template-opened think block changes where the thought ends.
    const asks = [_]Ask{
        .{ .stops = &stops, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false },
        .{ .stops = &stops, .normalize = false, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false },
        .{ .stops = &stops, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = true },
    };
    for (asks) |ask| {
        // A stop cut the call open: no call. (A stream's gate never holds the stop; the text it cut ends in the open call.)
        try expectSameEnd(a, open ++ "STOP and more", &stops, "length", ask, "STOP", "stop", null);
        // A stop after a closed call: the call stays.
        try expectSameEnd(a, call ++ "\nSTOP and more", &stops, "length", ask, "STOP", "stop", 1);
        // No stop: a length end and a cancel inside an open call decline it as a stop does, on both paths, and say why.
        try expectSameEnd(a, open, &stops, "length", ask, null, "length", null);
        try expectSameEnd(a, open, &stops, "cancelled", ask, null, "cancelled", null);
        // A call the model finished before the end ships as it is, however the end came; only the open one is declined.
        for ([_][]const u8{ "length", "cancelled" }) |finish| {
            try expectSameEnd(a, call, &stops, finish, ask, null, finish, 1);
            try expectSameEnd(a, call ++ "\n" ++ open, &stops, finish, ask, null, finish, 1);
        }
        try expectSameEnd(a, call ++ "\n" ++ open ++ "STOP and more", &stops, "length", ask, "STOP", "stop", 1);
        // Logprobs list the text returned: a stop's own tokens, and a thought the reply does not ship, are never listed.
        try expectSameListing(a, "an answer that runs on STOP and more", 5, &stops, ask);
        try expectSameListing(a, "<think>a thought the cap cut off", 5, &stops, ask);
        try expectSameListing(a, "<think>a thought</think>then an answer STOP more", 5, &stops, ask);
        // A call's own tokens are not listed: logprobs describe the content before it, as `logprobs.content` does.
        try expectSameListing(a, "Check it.\n" ++ call, 5, &stops, ask);
        try expectSameListing(a, "Check it.\n" ++ call ++ "\nSTOP and more", 5, &stops, ask);
        // A stop prefix one token holds is not listed while a later token releases the text before it.
        try expectSameListing(a, "ab STXQ more", 1, &.{ "STOP", "TXQ" }, ask);
    }
}

test "a reply holds only its thought when it has no visible text and no call, its thought open or closed" {
    const a = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const Case = struct { text: []const u8, opens: bool, want: ?[]const u8 };
    const cases = [_]Case{
        .{ .text = "I should check the clock.\n", .opens = true, .want = "I should check the clock." },
        .{ .text = "I should check the clock.\n</think>\n\n", .opens = true, .want = "I should check the clock." },
        .{ .text = "<think>plan</think>", .opens = false, .want = "plan" },
        .{ .text = "plan</think>\n\nIt is noon.", .opens = true, .want = null },
        .{ .text = "plan\n" ++ call, .opens = true, .want = null },
        .{ .text = "plan</think>\n\n" ++ call, .opens = true, .want = null },
        .{ .text = " \n", .opens = true, .want = null },
        .{ .text = "It is noon.", .opens = false, .want = null },
    };
    for (cases) |c| {
        errdefer std.debug.print("text: {f}\n", .{std.zig.fmtString(c.text)});
        const got = try soleThought(a, c.text, .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = c.opens });
        defer if (got) |g| a.free(g);
        if (c.want) |w| try testing.expectEqualStrings(w, got.?) else try testing.expect(got == null);
    }
    // Every details value this module writes has its object, and a loop after the continuation still parses no call.
    try testing.expect(thoughtDetailsJson(thought_continued) != null);
    try testing.expect(!shouldParseToolCalls(thought_loop));
    for (std.enums.values(ThoughtKept)) |k| try testing.expect(thoughtDetailsJson(k.details()) != null);
}
