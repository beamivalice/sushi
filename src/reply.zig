//! What a generation's text means to every reply surface, decided in one place: whether it holds tool calls and the
//! finish that goes with them. Chat, Anthropic and Responses, streamed or not, ask here.
const std = @import("std");
const think_tags = @import("think_tags.zig");
const tool_calls_mod = @import("tool_calls.zig");
const tool_schema = @import("tool_schema.zig");
const server_state = @import("server_state.zig");
const log = @import("log.zig");

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when the cut-off text salvaged a partial tool
/// call (the truncated-opener recovery in tool_calls.parseToolCalls) — clients key
/// their truncation recovery on it (the app's chunk-and-retry nudge fires on
/// finish_reason "length" + tool calls present). Overriding to "tool_calls"
/// hides the cut: the client executes a half-argument call, blames the model
/// ("you sent no content"), and the model re-emits the same doomed mega-call.
pub fn toolCallFinishReason(pre_parse: []const u8) []const u8 {
    return if (std.mem.eql(u8, pre_parse, "length")) "length" else "tool_calls";
}

/// A repetition-loop cut may land inside an otherwise recognizable tool call.
/// Never emit that buffer as executable work: unlike a genuine max-token cut,
/// this intentional stop must not ask clients to recover by compacting/retrying.
pub fn shouldParseToolCalls(finish_details: ?[]const u8) bool {
    const d = finish_details orelse return true;
    return !std.mem.eql(u8, d, "repetition_loop");
}

/// How the text came to be.
pub const Origin = struct {
    /// The prompt ended inside a think block (`promptOpensThink`): text with no closer is reasoning.
    opens_think: bool = false,
    /// A stop sequence cut `text`: markup it left open is no call.
    stop_cut: bool = false,
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
    // A stop that cut a call open left no complete call (the loop cut declines the same way).
    if (origin.stop_cut and think_tags.endsInsideToolCall(text)) {
        noteUnparsedToolMarkup(text);
        return null;
    }
    var parsed_calls = try tool_calls_mod.parseToolCalls(allocator, text);
    // Heuristically-inferred raw-JSON calls must name a DECLARED tool — a
    // truncated data object ({"name": "George Washington", …}) is not a call.
    // Deliberately NOT gated on g_tool_autocorrect: this corrects our own
    // heuristic's false positive, not the model's output.
    if (parsed_calls) |c| {
        if (tools_json) |tj| parsed_calls = try tool_schema.filterInferredBySchema(allocator, c, tj);
    }
    var calls = parsed_calls orelse
        (if (tools_json) |tj| try tool_schema.inferBareJsonToolCalls(allocator, text, tj) else null) orelse {
        noteUnparsedToolMarkup(text);
        return null;
    };
    if (!allow_parallel and calls.len > 1) {
        log.info("  parallel_tool_calls=false: clamping {d} parsed calls to the first\n", .{calls.len});
        for (calls[1..]) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        const kept = calls[0];
        allocator.free(calls);
        const one = try allocator.alloc(tool_schema.ParsedToolCall, 1);
        one[0] = kept;
        calls = one;
    }
    if (server_state.g_tool_autocorrect) {
        if (tools_json) |tj| {
            // Order matters: put a buried required param back where the schema
            // says it lives BEFORE coercing types, so the hoisted value is
            // type-checked like any other top-level arg.
            try tool_schema.hoistMisplacedRequiredParams(allocator, calls, tj);
            try tool_schema.coerceToolArgsToSchema(allocator, calls, tj);
        }
    }
    return calls;
}

/// A reply whose tool markup gave no call goes out as text with the markup cut, and its turn ends as a plain stop:
/// this line is the log's one trace of why.
fn noteUnparsedToolMarkup(text: []const u8) void {
    const at = think_tags.leakedToolMarkupStart(text) orelse return;
    var quoted: [512]u8 = undefined;
    const n = log.appendQuoted(&quoted, 0, text[at..], 96);
    log.info("[tool-parse] tool markup parsed to no call and is cut from the reply: {d} bytes from byte {d}, {s}\n", .{ text.len - at, at, quoted[0..n] });
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
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})).?;
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
    const calls2 = try parseToolCallsForRequest(allocator, text, null, true, .{});
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

    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})).?;
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
    const raw_calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})).?;
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
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})).?;
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
    const calls = try parseToolCallsForRequest(allocator, text, tools, true, .{});
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
    const none = parseToolCallsForRequest(allocator, wrecked, tools, true, .{}) catch |e| {
        log.closeFile();
        return e;
    };
    const prose = parseToolCallsForRequest(allocator, "The files are listed above.", tools, true, .{}) catch |e| {
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
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})) orelse
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
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})) orelse
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
    const both = (try parseToolCallsForRequest(allocator, text, null, true, .{})).?;
    defer {
        for (both) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(both);
    }
    try std.testing.expectEqual(@as(usize, 2), both.len);

    // allow_parallel=false: only the FIRST survives, intact.
    const one = (try parseToolCallsForRequest(allocator, text, null, false, .{})).?;
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
    const single = (try parseToolCallsForRequest(allocator, single_text, null, false, .{})).?;
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
    const calls = (try parseToolCallsForRequest(allocator, text, tools, true, .{})) orelse
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
        const calls = (try parseToolCallsForRequest(allocator, c.text, c.tools, true, .{})) orelse {
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
        try expectCalls(open, .{ .opens_think = true }, 1);
        try expectCalls(after, .{ .opens_think = true }, 1);
        // A call inside a thought that closed stays skipped.
        try expectCalls(inside, .{ .opens_think = true }, 0);

        const reply = (try toolReply(testing.allocator, open, test_tools, true, .{ .opens_think = true })).?;
        defer reply.deinit(testing.allocator);
        try testing.expectEqualStrings("get_weather", reply.calls[0].name);
        try testing.expectEqualStrings("", reply.preamble);
        const split = think_tags.splitThinkBlock(open, true, true);
        try testing.expectEqualStrings("I'll call it.", split.reasoning_content.?);
    }
}

test "a stop that cut a call open leaves no call, in every dialect" {
    const cut = [_][]const u8{
        "Plan.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nline one\n",
        "Plan.</think><tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>line one\n",
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"line one\n",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">notes.md</｜DSML｜parameter>\n<｜DSML｜parameter name=\"content\" string=\"true\">line one\n",
        "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>line one\n",
    };
    for (cut) |text| {
        try expectCalls(text, .{ .stop_cut = true }, 0);
        // Without a stop the same text is a length cut, whose salvage ships as it did.
        errdefer std.debug.print("salvage changed for {f}\n", .{std.zig.fmtString(text)});
        try expectCalls(text, .{}, 1);
    }
}

test "a stop after a call closed leaves the call" {
    const whole = [_][]const u8{
        "Let me look.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    };
    for (whole) |text| try expectCalls(text, .{ .stop_cut = true }, 1);
}
