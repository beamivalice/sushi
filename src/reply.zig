//! What a generation's text means to every reply surface, decided in one place: whether it holds tool calls and the
//! finish that goes with them. Chat, Anthropic and Responses, streamed or not, ask here.
const std = @import("std");
const think_tags = @import("think_tags.zig");
const tool_reading = @import("tool_reading.zig");
const tool_calls_mod = @import("tool_calls.zig");
const tool_schema = @import("tool_schema.zig");
const server_state = @import("server_state.zig");
const log = @import("log.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const listed_tokens = @import("listed_tokens.zig");
const http = @import("http.zig");

/// Finish reason for a response whose generated text yielded tool calls.
/// A TRUNCATED generation ("length": max_tokens or the request timeout) must
/// keep reporting "length" even when calls the model finished before the cut
/// are in the reply — clients key their truncation recovery on it. Overriding
/// to "tool_calls" hides the cut (an open call is declined and named by `settle`).
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
    /// The model itself ended the generation (its stop token): a wrapper whose call closed ships without its own close.
    model_ended: bool = false,
};

/// `finish_details.type` (Anthropic `stop_details`, the Responses `finish_details`) of a reply that declined a call.
pub const tool_call_cut = "tool_call_cut";

/// A tool call the generation left open. It is no call, and the reply names it.
pub const Declined = struct {
    /// The tool the parser read from the markup; null when the markup stopped before the name.
    name: ?[]u8,

    pub fn deinit(self: Declined, allocator: std.mem.Allocator) void {
        if (self.name) |n| allocator.free(n);
    }
};

/// The details object that names a declined call, owned.
pub fn declinedObject(allocator: std.mem.Allocator, d: Declined) ![]u8 {
    const name = d.name orelse return allocator.dupe(u8, "{\"type\":\"" ++ tool_call_cut ++ "\"}");
    const escaped = try http.jsonEscape(allocator, name);
    defer allocator.free(escaped);
    return std.fmt.allocPrint(allocator, "{{\"type\":\"" ++ tool_call_cut ++ "\",\"name\":{s}}}", .{escaped});
}

/// The details field named `key` of a finish, with its leading comma, owned: the declined call when the reply declined
/// one (it outranks a thought's note, which asks nothing of the client), else a copy of `otherwise`, the surface's own.
pub fn detailsField(allocator: std.mem.Allocator, comptime key: []const u8, declined: ?Declined, otherwise: []const u8) ![]u8 {
    const d = declined orelse return allocator.dupe(u8, otherwise);
    const object = try declinedObject(allocator, d);
    defer allocator.free(object);
    return std.fmt.allocPrint(allocator, ",\"" ++ key ++ "\":{s}", .{object});
}

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
    return .{ .preamble = preambleOf(text, split.content, calls), .calls = calls };
}

/// The text the model said before its calls: `content` (the split of `text`) up to the markup the split cuts, and, when
/// the calls are bare JSON (no markup anywhere in `text`), up to the JSON, a fence around it included, which is the
/// call and not content.
pub fn preambleOf(text: []const u8, content: []const u8, calls: []const tool_schema.ParsedToolCall) []const u8 {
    var from_bare_json = calls.len > 0;
    for (calls) |c| from_bare_json = from_bare_json and c.inferred;
    if (!from_bare_json or think_tags.leakedToolMarkupStart(text) != null) return think_tags.visibleToolPreamble(content);
    const at = tool_calls_mod.bareJsonCallStart(content) orelse return think_tags.visibleToolPreamble(content);
    return think_tags.visibleToolPreamble(std.mem.trimEnd(u8, content[0..at], " \t\r\n"));
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
    /// The call the generation left open, whatever ended it.
    declined: ?Declined,
    normalized: ?[]u8,

    pub fn deinit(self: Settled, allocator: std.mem.Allocator) void {
        if (self.reply) |r| r.deinit(allocator);
        if (self.declined) |d| d.deinit(allocator);
        if (self.normalized) |n| allocator.free(n);
    }
};

/// The one end handling of a reply: the stop that ended the generation, the text it leaves, the calls that text
/// holds and the finish a stop leaves. A call left open, by whatever ended the generation, is not a call and is named
/// (`Settled.declined`).
pub fn settle(allocator: std.mem.Allocator, gen: Generated, ask: Ask) !Settled {
    const cut = stop_seq_mod.cut(gen.text, ask.stops, gen.source);
    var out: Settled = .{
        .text = cut.text,
        .stop = cut.stop,
        .finish_reason = if (cut.stop != null) "stop" else gen.finish_reason,
        .reply = null,
        .declined = null,
        .normalized = null,
    };
    errdefer out.deinit(allocator);
    if (ask.normalize) {
        out.normalized = try think_tags.normalizeEmbeddedThinkBlocks(allocator, cut.text);
        if (out.normalized) |n| out.text = n;
    }
    if (ask.has_tools and shouldParseToolCalls(gen.finish_details)) {
        const model_ended = cut.stop == null and std.mem.eql(u8, gen.finish_reason, "stop");
        const origin: Origin = .{ .dialect = ask.dialect, .opens_think = ask.opens_think, .model_ended = model_ended };
        out.reply = try toolReply(allocator, out.text, ask.tools_json, ask.allow_parallel, origin);
        out.declined = try declinedCall(allocator, out.text, origin, ask.tools_json);
    }
    return out;
}

/// The call `text` leaves unfinished (open at its end, or a lost value close), named as far as its markup went, or
/// null when it leaves none. A bare JSON call has no marker: it is a call only once its name is whole and `tools_json`
/// declares it, as the parse chain reads it.
fn declinedCall(allocator: std.mem.Allocator, text: []const u8, origin: Origin, tools_json: ?[]const u8) !?Declined {
    const defects = try tool_reading.defects(allocator, text, origin.model_ended);
    defer allocator.free(defects);
    if (defects.len == 0) return declinedBareCall(allocator, text, origin, tools_json);
    const calls = (try tool_calls_mod.parseToolCallsAs(allocator, text[defects[0].start..defects[0].end], origin.dialect)) orelse {
        // A body cut before it names the tool: the opener's own `name="…"` attribute does.
        const attr = tool_reading.attributeName(text, defects[0].start) orelse return .{ .name = null };
        return .{ .name = try allocator.dupe(u8, attr) };
    };
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    return .{ .name = try allocator.dupe(u8, calls[0].name) };
}

fn declinedBareCall(allocator: std.mem.Allocator, text: []const u8, origin: Origin, tools_json: ?[]const u8) !?Declined {
    var open: ?[]u8 = null;
    errdefer if (open) |name| allocator.free(name);
    if (try tool_calls_mod.parseToolCallsOpen(allocator, text, origin.dialect, &open)) |calls| {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    const name = open orelse return null;
    if (tools_json) |tj| if (!tool_schema.toolNameIsDeclared(allocator, tj, name)) {
        allocator.free(name);
        return null;
    };
    return .{ .name = name };
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
    // An unfinished call is never shipped, whatever ended the generation (the model's own end included): the calls the
    // model finished stay, and `declinedCall` names the first unfinished one.
    const defects = try tool_reading.defects(allocator, text, origin.model_ended);
    defer allocator.free(defects);
    var rest = std.ArrayList(u8).empty;
    defer rest.deinit(allocator);
    var body = text;
    if (defects.len > 0) {
        tool_schema.noteUnparsedToolMarkup(text);
        var at: usize = 0;
        for (defects) |d| {
            try rest.appendSlice(allocator, text[at..d.start]);
            at = d.end;
        }
        try rest.appendSlice(allocator, text[at..]);
        body = rest.items;
    }
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

test "a generation that ended inside a call (the model's own end, a stop, the length cap, a cancel) leaves no call, in every dialect" {
    const cut = [_][]const u8{
        "Plan.\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nline one\n",
        "Plan.</think><tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>line one\n",
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"line one\n",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n<｜DSML｜parameter name=\"path\" string=\"true\">notes.md</｜DSML｜parameter>\n<｜DSML｜parameter name=\"content\" string=\"true\">line one\n",
        "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>line one\n",
    };
    for (cut) |text| {
        try expectCalls(text, .{ .dialect = null }, 0);
        // The call it left open is named by its markup, in each dialect.
        const left = (try declinedCall(testing.allocator, text, .{ .dialect = null }, test_tools)) orelse return error.DeclineNotNamed;
        defer left.deinit(testing.allocator);
        errdefer std.debug.print("declined {?s} for {f}\n", .{ left.name, std.zig.fmtString(text) });
        try testing.expectEqualStrings("write_file", left.name orelse return error.DeclineNotNamed);
    }
}

test "a cut after a call closed leaves the call" {
    const whole = [_][]const u8{
        "Let me look.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>\n",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
        "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Tokyo</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    };
    for (whole) |text| try expectCalls(text, .{ .dialect = null }, 1);
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
        const calls = (try parseToolCallsForRequest(testing.allocator, text, test_tools, true, .{ .dialect = null })) orelse return error.FinishedCallDropped;
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

/// What a reply says of the call its generation left open.
const Cut = union(enum) { none, named: []const u8, unnamed };

/// One generated text as a stream hands it to a surface (its gate delivers the text and reports the match) and as a
/// non-stream reply does (the whole text, the stop still in it), settled under the same ask.
fn expectSameEnd(allocator: std.mem.Allocator, generated: []const u8, stops: []const []const u8, gen_finish: []const u8, ask: Ask, want_stop: ?[]const u8, want_finish: []const u8, want_calls: ?usize, want_cut: Cut) !void {
    errdefer std.debug.print("generation ending {s}: {f}\n", .{ gen_finish, std.zig.fmtString(generated) });
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
        switch (want_cut) {
            .none => try testing.expect(got.declined == null),
            .named => |n| try testing.expectEqualStrings(n, (got.declined orelse return error.DeclineNotNamed).name orelse return error.DeclineNotNamed),
            .unnamed => try testing.expect((got.declined orelse return error.DeclineNotNamed).name == null),
        }
    }
    try testing.expectEqual(streamed.declined == null, whole.declined == null);
    if (streamed.declined) |s| {
        const sn = s.name orelse "";
        try testing.expectEqualStrings(sn, whole.declined.?.name orelse "");
        const a_obj = try declinedObject(allocator, s);
        defer allocator.free(a_obj);
        const b_obj = try declinedObject(allocator, whole.declined.?);
        defer allocator.free(b_obj);
        try testing.expectEqualStrings(a_obj, b_obj);
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
    const named: Cut = .{ .named = "get_weather" };
    for (asks) |ask| {
        // A stop cut the call open: no call. (A stream's gate never holds the stop; the text it cut ends in the open call.)
        try expectSameEnd(a, open ++ "STOP and more", &stops, "length", ask, "STOP", "stop", null, named);
        // A stop after a closed call: the call stays, and nothing was declined.
        try expectSameEnd(a, call ++ "\nSTOP and more", &stops, "length", ask, "STOP", "stop", 1, .none);
        // No stop: the model's own end, a length end and a cancel inside an open call decline it alike, on both paths,
        // and name it; the finish stays the end's own.
        try expectSameEnd(a, open, &stops, "stop", ask, null, "stop", null, named);
        try expectSameEnd(a, open, &stops, "length", ask, null, "length", null, named);
        try expectSameEnd(a, open, &stops, "cancelled", ask, null, "cancelled", null, named);
        // A call whose name the markup never reached is declined unnamed.
        try expectSameEnd(a, "Let me check.\n<tool_call>\n<function=", &stops, "stop", ask, null, "stop", null, .unnamed);
        // A call the model finished before the end ships as it is, however the end came; only the open one is declined.
        for ([_][]const u8{ "stop", "length", "cancelled" }) |finish| {
            try expectSameEnd(a, call, &stops, finish, ask, null, finish, 1, .none);
            try expectSameEnd(a, call ++ "\n" ++ open, &stops, finish, ask, null, finish, 1, named);
        }
        try expectSameEnd(a, call ++ "\n" ++ open ++ "STOP and more", &stops, "length", ask, "STOP", "stop", 1, named);
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

test "an unfinished call of every dialect the parse chain reads is declined and named, whatever ended the generation" {
    const a = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    const open = [_][]const u8{
        "<function name=\"get_weather\"><param name=\"city\">Tok",
        "<function=get_weather>\n<parameter=city>\nTok",
        "<invoke name=\"get_weather\">\n<parameter name=\"city\">Tok",
        "<function_calls>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Tok",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"city\" string=\"true\">Tok",
        "<tool name=\"get_weather\">{\"city\": \"Tokyo\"}",
        "<tool_request>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tok",
    };
    const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false };
    for (open) |call| for ([_][]const u8{ "", "Let me check.\n" }) |lead| for ([_][]const u8{ "stop", "length" }) |finish| {
        const text = try std.mem.concat(a, u8, &.{ lead, call });
        defer a.free(text);
        try expectSameEnd(a, text, &.{}, finish, ask, null, finish, null, .{ .named = "get_weather" });
    };
}

test "bare JSON cut short is declined and named, never repaired; a finished call keeps its repair, and an array keeps its complete elements" {
    const a = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    const stops = [_][]const u8{"STOP"};
    const asks = [_]Ask{
        .{ .stops = &stops, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false },
        .{ .stops = &stops, .normalize = true, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = true },
    };
    const whole = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}";
    const one_short = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}";
    const first = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}";
    const named: Cut = .{ .named = "get_weather" };
    for (asks) |ask| for ([_][]const u8{ "", "Let me check.\n" }) |lead| {
        // A whole object ships; the same object one `}` short at the end of the generation is declined and named.
        for ([_][]const u8{ "stop", "length", "cancelled" }) |finish| {
            const call = try std.mem.concat(a, u8, &.{ lead, whole });
            defer a.free(call);
            try expectSameEnd(a, call, &stops, finish, ask, null, finish, 1, .none);
            const cut = try std.mem.concat(a, u8, &.{ lead, one_short });
            defer a.free(cut);
            try expectSameEnd(a, cut, &stops, finish, ask, null, finish, null, named);
            // A cut in the middle of a value or a key is declined the same, once the name is read.
            const mid = try std.mem.concat(a, u8, &.{ lead, "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tok" });
            defer a.free(mid);
            try expectSameEnd(a, mid, &stops, finish, ask, null, finish, null, named);
            // An array whose second element is cut
            // ships the complete first and declines the second.
            const arr = try std.mem.concat(a, u8, &.{ "[", first, ",{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Osa" });
            defer a.free(arr);
            try expectSameEnd(a, arr, &stops, finish, ask, null, finish, 1, named);
            // Both elements complete and the array not closed: nothing is cut, both ship.
            const open_arr = try std.mem.concat(a, u8, &.{ "[", first, ",", first });
            defer a.free(open_arr);
            try expectSameEnd(a, open_arr, &stops, finish, ask, null, finish, 2, .none);
        }
        // A stop that cut the object short.
        const stopped = try std.mem.concat(a, u8, &.{ lead, one_short, "STOP and more" });
        defer a.free(stopped);
        try expectSameEnd(a, stopped, &stops, "length", ask, "STOP", "stop", null, named);
        // A name the request never declared is a data object, not a call: nothing is declined.
        try expectSameEnd(a, "{\"name\":\"George Washington\",\"num\":1,", &stops, "length", ask, null, "length", null, .none);
        try expectSameEnd(a, "{\"city\": \"Tok", &stops, "length", ask, null, "length", null, .none);
    };
    // A call the model finished (a close tag after it) keeps its repair: one `}` short, then its own close.
    const finished = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}</tool_call>";
    try expectSameEnd(a, finished, &stops, "stop", asks[0], null, "stop", 1, .none);
}

test "a closer spelled inside a parameter value does not close the call: the generation cut there leaves it open, in every dialect" {
    const a = testing.allocator;
    const stops = [_][]const u8{"STOP"};
    const ask: Ask = .{ .stops = &stops, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
    // Each text is a write_file call whose `content` value spells a closer of its own dialect, cut right after it or a few
    // bytes on; none of them has closed.
    const Open = struct { text: []const u8, cut: Cut = .{ .named = "write_file" } };
    const open = [_]Open{
        .{ .text = "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nprint(\"</function></tool_call>\")" },
        .{ .text = "<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nprint(\"</function>\")" },
        .{ .text = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"a </tool_call> b" },
        .{ .text = "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>a </tool_call> b" },
        .{ .text = "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>a <tool_call|> b" },
        .{ .text = "<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">a </\u{ff5c}DSML\u{ff5c}invoke> b" },
        .{ .text = "<invoke name=\"write_file\">\n<parameter name=\"path\">notes.md</parameter>\n<parameter name=\"content\">a </invoke> b" },
        .{ .text = "<function name=\"write_file\"><param name=\"content\">a </function> b" },
        // The attribute names the tool: a body cut inside its JSON is declined under that name.
        .{ .text = "<tool name=\"write_file\">{\"path\": \"notes.md\", \"content\": \"a </tool> b" },
        .{ .text = "<tool name=\"write_file\">{\"path\": \"notes.md\", \"content\": \"abc" },
        .{ .text = "<tool_call name=\"write_file\">{\"path\": \"notes.md\", \"content\": \"abc" },
        .{ .text = "<tool_request>{\"name\": \"write_file\", \"arguments\": {\"content\": \"a </tool_request> b" },
    };
    for (open) |call| for ([_][]const u8{ "", "Let me write it.\n" }) |lead| for ([_][]const u8{ "stop", "length" }) |finish| {
        const text = try std.mem.concat(a, u8, &.{ lead, call.text });
        defer a.free(text);
        try expectSameEnd(a, text, &stops, finish, ask, null, finish, null, call.cut);
    };
}

test "a declined call is named in the details field of every surface's key, and outranks a thought's note" {
    const a = testing.allocator;
    const named: Declined = .{ .name = try a.dupe(u8, "read\"file") };
    defer named.deinit(a);
    const unnamed: Declined = .{ .name = null };
    const note = thoughtDetailsField("stop_details", thought_continued);
    const got = try detailsField(a, "stop_details", named, note);
    defer a.free(got);
    try testing.expectEqualStrings(",\"stop_details\":{\"type\":\"tool_call_cut\",\"name\":\"read\\\"file\"}", got);
    const bare = try detailsField(a, "finish_details", unnamed, "");
    defer a.free(bare);
    try testing.expectEqualStrings(",\"finish_details\":{\"type\":\"tool_call_cut\"}", bare);
    // No decline: the surface's own field, byte for byte.
    const own = try detailsField(a, "stop_details", null, note);
    defer a.free(own);
    try testing.expectEqualStrings(note, own);
}

test "a span-routed constrained reply keeps the marker text in its payload as written, streamed or not, and a call after it still ships" {
    const a = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
    ;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const payload = "{\"note\":\"a <think>b</think> c <think>d</think> e\"}\n";
    const text = payload ++ call;
    const routed: Ask = .{ .stops = &.{}, .normalize = false, .has_tools = true, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false };
    try expectSameEnd(a, text, &.{}, "stop", routed, null, "stop", 1, .none);
    const kept = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, routed);
    defer kept.deinit(a);
    try testing.expectEqualStrings(text, kept.text);
    // The witness: the same text normalized is another text, which is why every surface asks by one predicate.
    var merged = routed;
    merged.normalize = true;
    const changed = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, merged);
    defer changed.deinit(a);
    try testing.expect(!std.mem.eql(u8, text, changed.text));
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




test "r173-f-foreign-opener: an opener mentioned inside a thought that closed is no open call" {
    const a = testing.allocator;
    const texts = [_][]const u8{
        "<think>I could emit <tool_call> here, but the answer is known.</think>\n\nIt is sunny.",
        "<think>Maybe <|tool_call>call:get_weather{city: no.</think>\n\nIt is sunny.",
    };
    for (texts) |text| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        errdefer std.debug.print("thought {f}: declined={?s}\n", .{ std.zig.fmtString(text), if (s.declined) |d| d.name orelse "(unnamed)" else null });
        try testing.expect(s.reply == null);
        try testing.expect(s.declined == null);
    }
}

test "r173-f-foreign-opener: served dialects: a finished call whose value spells a foreign opener keeps every argument" {
    const a = testing.allocator;
    const Case = struct { dialect: tool_calls_mod.Dialect, text: []const u8 };
    const value = "x = 1\nif s.startswith(\"<|tool_call>\"): pass";
    const cases = [_]Case{
        .{ .dialect = .function_tag, .text = "<tool_call>\n<function=write_file>\n<parameter=path>\nparse.py\n</parameter>\n<parameter=content>\n" ++ value ++ "\n</parameter>\n</function>\n</tool_call>" },
        .{ .dialect = .arg_key, .text = "<tool_call>write_file<arg_key>path</arg_key><arg_value>parse.py</arg_value><arg_key>content</arg_key><arg_value>" ++ value ++ "</arg_value></tool_call>" },
        .{ .dialect = .dsml, .text = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">parse.py</\u{ff5c}DSML\u{ff5c}parameter>\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">" ++ value ++ "</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>" },
    };
    var bad: usize = 0;
    for (cases) |c| for ([_][]const u8{ "stop", "length" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = c.dialect, .opens_think = false };
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        const args = if (s.reply) |r| r.calls[0].arguments else "(no call)";
        const ok = s.declined == null and s.reply != null and std.mem.indexOf(u8, args, "\"content\"") != null;
        if (!ok) {
            bad += 1;
            std.debug.print("{s}/{s}: args {s} declined={?s}\n", .{ @tagName(c.dialect), finish, args, if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: a foreign opener mentioned in a closed thought does not drop the real call after it" {
    const a = testing.allocator;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const leads = [_][]const u8{
        "<think>Gemma would write <|tool_call> here; I use my own format.</think>\n\n",
        "Gemma would write <|tool_call> here; I use my own format.</think>\n\n",
    };
    var bad: usize = 0;
    for (leads, [_]bool{ false, true }) |lead, opens| {
        const text = try std.mem.concat(a, u8, &.{ lead, call });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = .function_tag, .opens_think = opens };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const n = if (s.reply) |r| r.calls.len else 0;
        if (n != 1 or s.declined != null) {
            bad += 1;
            std.debug.print("thought-lead opens={}: calls={d} declined={?s}\n", .{ opens, n, if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: a finished call followed by prose that mentions an opener is not declined" {
    const a = testing.allocator;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    const tails = [_][]const u8{ "\nI used the `<tool_call>` tag.", "\nGemma would write `<|tool_call>`." };
    var bad: usize = 0;
    for (tails) |tail| {
        const text = try std.mem.concat(a, u8, &.{ call, tail });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = .function_tag, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const n = if (s.reply) |r| r.calls.len else 0;
        if (n != 1 or s.declined != null) {
            bad += 1;
            std.debug.print("tail {f}: calls={d} declined={?s}\n", .{ std.zig.fmtString(tail), n, if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-lost-close-wrapper: a call whose inner call closed but whose wrapper did not ships at the model's own end" {
    const a = testing.allocator;
    const dsml = "\u{ff5c}DSML\u{ff5c}";
    const Case = struct { dialect: ?tool_calls_mod.Dialect, text: []const u8 };
    const cases = [_]Case{
        .{ .dialect = .function_tag, .text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n" },
        .{ .dialect = .dsml, .text = "<" ++ dsml ++ "tool_calls>\n<" ++ dsml ++ "invoke name=\"get_weather\">\n<" ++ dsml ++ "parameter name=\"city\" string=\"true\">Tokyo</" ++ dsml ++ "parameter>\n</" ++ dsml ++ "invoke>\n" },
        .{ .dialect = null, .text = "<function_calls>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Tokyo</parameter>\n</invoke>\n" },
    };
    var bad: usize = 0;
    for (cases) |c| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = c.dialect, .opens_think = false };
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        if ((if (s.reply) |r| r.calls.len else 0) != 1 or s.declined != null) {
            bad += 1;
            std.debug.print("wrapper open {s}: calls={d} declined={?s}\n", .{ if (c.dialect) |d| @tagName(d) else "null", if (s.reply) |r| r.calls.len else 0, if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}


test "r173-f-lost-close-merge: a call that lost its last parameter close, followed by a whole call, is declined and the whole call ships" {
    const a = testing.allocator;
    const dsml = "\u{ff5c}DSML\u{ff5c}";
    const texts = [_][]const u8{
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOsaka\n</parameter>\n</function>\n</tool_call>",
        "<" ++ dsml ++ "tool_calls>\n<" ++ dsml ++ "invoke name=\"write_file\">\n<" ++ dsml ++ "parameter name=\"path\" string=\"true\">a.md\n</" ++ dsml ++ "invoke>\n<" ++ dsml ++ "invoke name=\"get_weather\">\n<" ++ dsml ++ "parameter name=\"city\" string=\"true\">Osaka</" ++ dsml ++ "parameter>\n</" ++ dsml ++ "invoke>\n</" ++ dsml ++ "tool_calls>",
        "<invoke name=\"write_file\">\n<parameter name=\"path\">a.md\n</invoke>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Osaka</parameter>\n</invoke>",
        "<function_calls>\n<invoke name=\"write_file\">\n<parameter name=\"path\">a.md\n</invoke>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Osaka</parameter>\n</invoke>\n</function_calls>",
        "<function name=\"write_file\"><param name=\"path\">a.md</function>\n<function name=\"get_weather\"><param name=\"city\">Osaka</param></function>",
    };
    var bad: usize = 0;
    for (texts) |text| for ([_]?tool_calls_mod.Dialect{ null, .function_tag, .dsml }) |dialect| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = dialect, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const shipped = if (s.reply) |r| r.calls.len == 1 and std.mem.eql(u8, r.calls[0].name, "get_weather") and std.mem.indexOf(u8, r.calls[0].arguments, "\"Osaka\"") != null else false;
        const named = if (s.declined) |d| if (d.name) |n| std.mem.eql(u8, n, "write_file") else false else false;
        if (!shipped or !named) {
            bad += 1;
            std.debug.print("lost close {f} [{s}]: shipped={} declined={?s}\n", .{ std.zig.fmtString(text[0..@min(text.len, 60)]), if (dialect) |d| @tagName(d) else "null", shipped, if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}


test "r173-f-foreign-opener: Gemma: a string value ending in a brace right before a spelled closer keeps the call whole" {
    const a = testing.allocator;
    const text = "<|tool_call>call:write_file{path:<|\"|>a.rs<|\"|>,content:<|\"|>fn f() {}<tool_call|> tail<|\"|>}<tool_call|>";
    var bad: usize = 0;
    for ([_][]const u8{ "stop", "length" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        const ok = if (s.reply) |r| r.calls.len == 1 and std.mem.indexOf(u8, r.calls[0].arguments, "tail") != null else false;
        if (!ok or s.declined != null) {
            bad += 1;
            std.debug.print("gemma-brace [{s}]: declined={?s}\n", .{ finish, if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: Hermes JSON: a string value ending in a brace right before a spelled closer" {
    const a = testing.allocator;
    const Case = struct { text: []const u8, finish: []const u8, calls: ?usize, declined: bool, want: ?[]const u8 = null };
    const cases = [_]Case{
        .{ .text = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"content\": \"x = {}</tool_call> y", .finish = "length", .calls = null, .declined = true },
        .{ .text = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"content\": \"x = {}</tool_call> y\"}}\n</tool_call>", .finish = "stop", .calls = 1, .declined = false, .want = "x = {}</tool_call> y" },
        .{ .text = "<tool_call>\n<function=write_file>\n<parameter=path>\na\n</parameter>\n<parameter=content>\nx = {}</tool_call> y", .finish = "length", .calls = null, .declined = true },
        .{ .text = "<invoke name=\"write_file\">\n<parameter name=\"path\">a</parameter>\n<parameter name=\"content\">x = {}</invoke> y", .finish = "length", .calls = null, .declined = true },
    };
    var bad: usize = 0;
    for (cases) |c| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = c.finish, .finish_details = null }, ask);
        defer s.deinit(a);
        var ok = (if (s.reply) |r| @as(?usize, r.calls.len) else null) == c.calls and (s.declined != null) == c.declined;
        if (c.want) |w| if (s.reply) |r| {
            const p = try std.json.parseFromSlice(std.json.Value, a, r.calls[0].arguments, .{});
            defer p.deinit();
            ok = ok and if (p.value.object.get("content")) |v| std.mem.eql(u8, v.string, w) else false;
        };
        if (!ok) {
            bad += 1;
            std.debug.print("brace [{s}] {f}: declined={?s}\n", .{ c.finish, std.zig.fmtString(c.text[0..@min(c.text.len, 70)]), if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}


test "r173-f-foreign-opener: a finished call whose value spells its own dialect's opener ships whole" {
    const a = testing.allocator;
    const Case = struct { dialect: ?tool_calls_mod.Dialect, text: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{ .dialect = null, .text = "<tool_call>{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.md\",\"content\":\"see <tool_call> tag\"}}</tool_call>", .want = "see <tool_call> tag" },
        .{ .dialect = null, .text = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.md\", \"content\": \"wrap: <tool_call>x</tool_call> done\"}}\n</tool_call>", .want = "wrap: <tool_call>x</tool_call> done" },
        .{ .dialect = .function_tag, .text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n<parameter=content>\nsee <tool_call> tag\n</parameter>\n</function>\n</tool_call>", .want = "see <tool_call> tag" },
        .{ .dialect = .function_tag, .text = "<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n<parameter=content>\nuse <function=x> then\n</parameter>\n</function>\n</tool_call>", .want = "use <function=x> then" },
        .{ .dialect = .arg_key, .text = "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.md</arg_value><arg_key>content</arg_key><arg_value>see <tool_call> tag</arg_value></tool_call>", .want = "see <tool_call> tag" },
        .{ .dialect = .dsml, .text = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">a.md</\u{ff5c}DSML\u{ff5c}parameter>\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">see <\u{ff5c}DSML\u{ff5c}invoke name=\"y\"> tag</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>", .want = "see <\u{ff5c}DSML\u{ff5c}invoke name=\"y\"> tag" },
        .{ .dialect = null, .text = "<|tool_call>call:write_file{path:<|\"|>a.md<|\"|>,content:<|\"|>see <|tool_call> tag<|\"|>}<tool_call|>", .want = "see <|tool_call> tag" },
    };
    var bad: usize = 0;
    for (cases) |c| for ([_][]const u8{ "stop", "length" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = c.dialect, .opens_think = false };
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        var got: []const u8 = "(no call)";
        var parsed: ?std.json.Parsed(std.json.Value) = null;
        defer if (parsed) |p| p.deinit();
        if (s.reply) |r| {
            parsed = try std.json.parseFromSlice(std.json.Value, a, r.calls[0].arguments, .{});
            got = if (parsed.?.value.object.get("content")) |v| v.string else r.calls[0].arguments;
        }
        if (s.declined != null or !std.mem.eql(u8, got, c.want)) {
            bad += 1;
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
            std.debug.print("own {f} [{s}]: content {f} declined={?s}\n", .{ std.zig.fmtString(c.text[0..@min(40, c.text.len)]), finish, std.zig.fmtString(got), if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: a finished Hermes JSON call whose string spells the closer keeps its arguments, under every served dialect" {
    const a = testing.allocator;
    const texts = [_][]const u8{
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.md\", \"content\": \"end </tool_call> here\"}}\n</tool_call>",
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.md\", \"content\": \"wrap: <tool_call>x</tool_call> done\"}}\n</tool_call>",
    };
    var bad: usize = 0;
    for (texts) |text| for ([_]?tool_calls_mod.Dialect{ null, .function_tag, .arg_key, .dsml }) |dialect| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = dialect, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const ok = if (s.reply) |r| r.calls.len == 1 and std.mem.eql(u8, r.calls[0].name, "write_file") and std.mem.indexOf(u8, r.calls[0].arguments, "here") != null or std.mem.indexOf(u8, r.calls[0].arguments, "done") != null else false;
        if (!ok or s.declined != null) {
            bad += 1;
            std.debug.print("hermes-closer {s} {f}: declined={?s}\n", .{ if (dialect) |d| @tagName(d) else "null", std.zig.fmtString(text[40..@min(text.len, 110)]), if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: a finished arg_key call whose value spells a whole call of its own dialect keeps its arguments" {
    const a = testing.allocator;
    const texts = [_][]const u8{
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.md</arg_value><arg_key>content</arg_key><arg_value>wrap: <tool_call>x</tool_call> done</arg_value></tool_call>",
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.md</arg_value><arg_key>content</arg_key><arg_value>end </tool_call> done</arg_value></tool_call>",
    };
    var bad: usize = 0;
    for (texts) |text| for ([_][]const u8{ "stop", "length" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = .arg_key, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        const ok = if (s.reply) |r| r.calls.len == 1 and std.mem.eql(u8, r.calls[0].name, "write_file") and std.mem.indexOf(u8, r.calls[0].arguments, "done") != null else false;
        if (!ok or s.declined != null) {
            bad += 1;
            std.debug.print("glm [{s}] {f}: declined={?s}\n", .{ finish, std.zig.fmtString(text[80..]), if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: a finished function-tag call whose value spells its closers keeps every argument" {
    const a = testing.allocator;
    const values = [_][]const u8{ "print(\"</tool_call>\")", "print(\"</function>\")", "print(\"</function></tool_call>\")" };
    var bad: usize = 0;
    for (values) |v| for ([_]?tool_calls_mod.Dialect{ .function_tag, null }) |dialect| {
        const text = try std.mem.concat(a, u8, &.{ "<tool_call>\n<function=write_file>\n<parameter=content>\n", v, "\n</parameter>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>" });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = dialect, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const ok = if (s.reply) |r| r.calls.len == 1 and std.mem.indexOf(u8, r.calls[0].arguments, "a.py") != null else false;
        if (!ok or s.declined != null) {
            bad += 1;
            std.debug.print("qwen-closer {s} {s}: declined={?s}\n", .{ if (dialect) |d| @tagName(d) else "null", v, if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: DSML: a finished call whose value spells its closers keeps every argument" {
    const a = testing.allocator;
    const values = [_][]const u8{ "a </\u{ff5c}DSML\u{ff5c}invoke></\u{ff5c}DSML\u{ff5c}tool_calls> b", "a </\u{ff5c}DSML\u{ff5c}tool_calls> b", "<\u{ff5c}DSML\u{ff5c}tool_calls> b" };
    var bad: usize = 0;
    for (values) |v| for ([_][]const u8{ "stop", "length" }) |finish| {
        const text = try std.mem.concat(a, u8, &.{ "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">", v, "</\u{ff5c}DSML\u{ff5c}parameter>\n<\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">a.py</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>" });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = .dsml, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        const ok = if (s.reply) |r| r.calls.len == 1 and std.mem.indexOf(u8, r.calls[0].arguments, "a.py") != null else false;
        if (!ok or s.declined != null) {
            bad += 1;
            std.debug.print("dsml-closer [{s}] {f}: declined={?s}\n", .{ finish, std.zig.fmtString(v), if (s.declined) |d| d.name orelse "(unnamed)" else null });
            if (s.reply) |r| for (r.calls) |cc| std.debug.print("  call {s} {s}\n", .{ cc.name, cc.arguments });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}


test "r173-f-foreign-opener: a finished call whose value spells think tags keeps the value as written" {
    const a = testing.allocator;
    const value = "a <think>b</think> c";
    const leads = [_][]const u8{ "", "<think>plan</think>\n\n", "plan</think>\n\n" };
    var bad: usize = 0;
    for (leads, [_]bool{ false, false, true }) |lead, opens| for ([_]?tool_calls_mod.Dialect{ .function_tag, .arg_key, .dsml }) |dialect| {
        const call = switch (dialect.?) {
            .function_tag => "<tool_call>\n<function=write_file>\n<parameter=path>\na.md\n</parameter>\n<parameter=content>\n" ++ value ++ "\n</parameter>\n</function>\n</tool_call>",
            .arg_key => "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.md</arg_value><arg_key>content</arg_key><arg_value>" ++ value ++ "</arg_value></tool_call>",
            .dsml => "<\u{ff5c}DSML\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"write_file\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">a.md</\u{ff5c}DSML\u{ff5c}parameter>\n<\u{ff5c}DSML\u{ff5c}parameter name=\"content\" string=\"true\">" ++ value ++ "</\u{ff5c}DSML\u{ff5c}parameter>\n</\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>",
        };
        const text = try std.mem.concat(a, u8, &.{ lead, call });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = dialect, .opens_think = opens };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        var got: []const u8 = "(no call)";
        var p: ?std.json.Parsed(std.json.Value) = null;
        defer if (p) |q| q.deinit();
        if (s.reply) |r| {
            p = try std.json.parseFromSlice(std.json.Value, a, r.calls[0].arguments, .{});
            got = if (p.?.value.object.get("content")) |v| v.string else r.calls[0].arguments;
        }
        if (!std.mem.eql(u8, got, value) or s.declined != null) {
            bad += 1;
            std.debug.print("think-value {s} lead={f}: content {f} declined={?s}\n", .{ @tagName(dialect.?), std.zig.fmtString(lead), std.zig.fmtString(got), if (s.declined) |d| d.name orelse "(unnamed)" else null });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}


test "r173-f-foreign-opener: a declined call's markup never reaches content, streamed or not" {
    const a = testing.allocator;
    _ = a;
    const open = [_][]const u8{
        "<function name=\"get_weather\"><param name=\"city\">Tok",
        "<function=get_weather>\n<parameter=city>\nTok",
        "<invoke name=\"get_weather\">\n<parameter name=\"city\">Tok",
        "<function_calls>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Tok",
        "<\u{ff5c}DSML\u{ff5c}invoke name=\"get_weather\">\n<\u{ff5c}DSML\u{ff5c}parameter name=\"city\" string=\"true\">Tok",
        "<tool name=\"get_weather\">{\"city\": \"Tokyo\"}",
        "<tool_request>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tok",
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nTok",
        "<|tool_call>call:get_weather{city:<|\"|>Tok",
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tok",
    };
    var bad: usize = 0;
    for (open) |call| {
        var buf: [512]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "Let me check.\n{s}", .{call});
        const content = think_tags.splitThinkBlock(text, true, false).content;
        if (!std.mem.eql(u8, std.mem.trim(u8, content, " \n"), "Let me check.")) {
            bad += 1;
            std.debug.print("non-stream content leaks for {f}: {f}\n", .{ std.zig.fmtString(call), std.zig.fmtString(content) });
        }
        // The stream holds from the opener on: no prefix holding the opener's first byte is released.
        var scan: think_tags.ToolHoldScan = .{};
        const lead = "Let me check.\n".len;
        var released_markup: ?usize = null;
        for (lead + 1..text.len + 1) |end| {
            if (!think_tags.streamShouldBufferForToolsScan(text[0..end], &scan) and released_markup == null) released_markup = end;
        }
        if (released_markup) |end| {
            bad += 1;
            std.debug.print("stream releases markup for {f} at prefix {f}\n", .{ std.zig.fmtString(call), std.zig.fmtString(text[lead..end]) });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-foreign-opener: the preamble of a finished call holds no markup, in every dialect the chain reads" {
    const a = testing.allocator;
    const calls = [_][]const u8{
        "<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>",
        "<function name=\"get_weather\"><param name=\"city\">Tokyo</param></function>",
        "<invoke name=\"get_weather\">\n<parameter name=\"city\">Tokyo</parameter>\n</invoke>",
        "<tool name=\"get_weather\">{\"city\": \"Tokyo\"}</tool>",
        "<tool_request>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</tool_request>",
    };
    var bad: usize = 0;
    for (calls) |call| {
        const text = try std.mem.concat(a, u8, &.{ "Let me check.\n", call });
        defer a.free(text);
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
        const s = try settle(a, .{ .text = text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const pre = if (s.reply) |r| r.preamble else "(no call)";
        if (!std.mem.eql(u8, std.mem.trim(u8, pre, " \n"), "Let me check.")) {
            bad += 1;
            std.debug.print("preamble for {f}: {f}\n", .{ std.zig.fmtString(call[0..@min(call.len, 30)]), std.zig.fmtString(pre) });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-array-after-prose: a bare JSON array of calls after prose ships every complete element, a cut one is declined and named, streamed or not" {
    const a = testing.allocator;
    const first = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Tokyo\"}}";
    const second = "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Osaka\"}}";
    const asks = [_]Ask{
        .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false },
        .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = true },
    };
    for (asks) |ask| for ([_][]const u8{ "Let me check.\n", "Let me check:\n```json\n" }) |lead| for ([_][]const u8{ "stop", "length", "cancelled" }) |finish| {
        const whole = try std.mem.concat(a, u8, &.{ lead, "[", first, ",", second, "]" });
        defer a.free(whole);
        try expectSameEnd(a, whole, &.{}, finish, ask, null, finish, 2, .none);
        const cut = try std.mem.concat(a, u8, &.{ lead, "[", first, ",{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Osa" });
        defer a.free(cut);
        try expectSameEnd(a, cut, &.{}, finish, ask, null, finish, 1, .{ .named = "get_weather" });
        const unclosed = try std.mem.concat(a, u8, &.{ lead, "[", first, ",", second });
        defer a.free(unclosed);
        try expectSameEnd(a, unclosed, &.{}, finish, ask, null, finish, 2, .none);
    };
}

test "r173-f-span-cut-call: a call a model backticks ships whole, and its value's foreign markup is no call, under every dialect" {
    const a = testing.allocator;
    const dsml = "\u{ff5c}DSML\u{ff5c}";
    const Case = struct { text: []const u8, dialect: ?tool_calls_mod.Dialect, name: []const u8 };
    const hermes_dsml_value = "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.md\", \"content\": \"<" ++ dsml ++ "invoke name=\\\"get_weather\\\"><" ++ dsml ++ "parameter name=\\\"city\\\" string=\\\"true\\\">X</" ++ dsml ++ "parameter></" ++ dsml ++ "invoke>\"}}</tool_call>";
    const hermes_glm_value = "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"a.md\", \"content\": \"<tool_call>get_weather<arg_key>city</arg_key><arg_value>X</arg_value></tool_call>\"}}</tool_call>";
    const cases = [_]Case{
        .{ .text = "Run `" ++ hermes_dsml_value ++ "`", .dialect = .dsml, .name = "write_file" },
        .{ .text = "Run `" ++ hermes_glm_value ++ "`", .dialect = .arg_key, .name = "write_file" },
        .{ .text = "Run `<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</tool_call>`", .dialect = null, .name = "get_weather" },
        .{ .text = "I use ` here <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}</tool_call>", .dialect = null, .name = "get_weather" },
        .{ .text = "I use ` here <tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>", .dialect = .function_tag, .name = "get_weather" },
    };
    for (cases) |c| for ([_][]const u8{ "stop", "length" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = c.dialect, .opens_think = false };
        errdefer std.debug.print("[{s}] {f}\n", .{ finish, std.zig.fmtString(c.text) });
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        try testing.expect(s.declined == null);
        const r = s.reply orelse return error.CallDropped;
        try testing.expectEqual(@as(usize, 1), r.calls.len);
        try testing.expectEqualStrings(c.name, r.calls[0].name);
    };
}

/// What one settle of a reply may read, per byte of it: a reading pass sweeps the text once for each needle it must rule
/// out (the next `<`, the call's closers, a think or channel tag), about five sweeps, and a settle makes about twenty-five
/// passes (the normalize, the parse, the unfinished-call reading, the split, the decline).
const settle_work_per_byte = 140;

/// The reading work of one `settle` of `text`: every byte a reading search covered (`tool_reading.work`).
fn settleWork(allocator: std.mem.Allocator, text: []const u8, finish: []const u8) !usize {
    const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = false };
    const before = tool_reading.work;
    const s = try settle(allocator, .{ .text = text, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
    defer s.deinit(allocator);
    return tool_reading.work - before;
}

test "r173-f-settle-linear: settling k whole calls and a think tag reads the text a bounded number of times, however many calls" {
    const a = testing.allocator;
    const call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>\n";
    var work: [3]usize = undefined;
    var bytes: [3]usize = undefined;
    for ([_]usize{ 200, 400, 800 }, 0..) |k, i| {
        var text = std.ArrayList(u8).empty;
        defer text.deinit(a);
        for (0..k) |_| try text.appendSlice(a, call);
        try text.appendSlice(a, "\n<think>");
        bytes[i] = text.items.len;
        work[i] = try settleWork(a, text.items, "stop");
    }
    errdefer std.debug.print("k=200/400/800 bytes={any} work={any}\n", .{ bytes, work });
    for (work, bytes) |w, n| try testing.expect(w <= settle_work_per_byte * n);
    try testing.expect(work[2] <= 5 * work[1] / 2);
}

test "r173-f-settle-linear: settling k thoughts each followed by a call reads the text a bounded number of times, however many thoughts" {
    const a = testing.allocator;
    const unit = "<think>plan</think><tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}\n</tool_call>\n";
    var work: [4]usize = undefined;
    var bytes: [4]usize = undefined;
    for ([_]usize{ 100, 200, 400, 800 }, 0..) |k, i| {
        var text = std.ArrayList(u8).empty;
        defer text.deinit(a);
        for (0..k) |_| try text.appendSlice(a, unit);
        bytes[i] = text.items.len;
        work[i] = try settleWork(a, text.items, "stop");
    }
    errdefer std.debug.print("k=100/200/400/800 bytes={any} work={any}\n", .{ bytes, work });
    for (work, bytes) |w, n| try testing.expect(w <= settle_work_per_byte * n);
    try testing.expect(work[3] <= 5 * work[2] / 2);
}

test "r173-f-cut-think-tag: a think tag inside a cut call's value is that call's text: the call is declined and named, nothing of its value ships as content, in every dialect" {
    const a = testing.allocator;
    const dsml = "\u{ff5c}DSML\u{ff5c}";
    const texts = [_][]const u8{
        "<tool_call>\n<function=write_file>\n<parameter=path>\nnotes.md\n</parameter>\n<parameter=content>\nuse </think> tags\nline one\n",
        "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"use </think> tags, line",
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>notes.md</arg_value><arg_key>content</arg_key><arg_value>use </think> tags\nline",
        "<" ++ dsml ++ "tool_calls>\n<" ++ dsml ++ "invoke name=\"write_file\">\n<" ++ dsml ++ "parameter name=\"path\" string=\"true\">notes.md</" ++ dsml ++ "parameter>\n<" ++ dsml ++ "parameter name=\"content\" string=\"true\">use </think> tags\nline",
        "<|tool_call>call:write_file{path:<|\"|>notes.md<|\"|>,content:<|\"|>use </think> tags\nline",
        "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"notes.md\", \"content\": \"use <think>x</think> tags, line",
    };
    var bad: usize = 0;
    for (texts) |t| for ([_]bool{ false, true }) |opens| for ([_][]const u8{ "length", "stop" }) |finish| {
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = opens };
        const s = try settle(a, .{ .text = t, .source = .whole, .finish_reason = finish, .finish_details = null }, ask);
        defer s.deinit(a);
        const content = think_tags.splitThinkBlock(s.text, true, opens).content;
        if (s.reply != null or s.declined == null or std.mem.indexOf(u8, content, "tags") != null or std.mem.indexOf(u8, content, "line") != null) {
            bad += 1;
            std.debug.print("opens_think={} [{s}] {f}: calls={d} declined={?s} content={f}\n", .{ opens, finish, std.zig.fmtString(t), if (s.reply) |r| r.calls.len else 0, if (s.declined) |d| d.name orelse "(unnamed)" else null, std.zig.fmtString(content) });
        }
    };
    try testing.expectEqual(@as(usize, 0), bad);
}

test "r173-f-bare-json-preamble: a finished bare-JSON call's own JSON is the call, never the content: the preamble is the text before it, a fence around it goes with it, and a stream holds what the whole reply cuts" {
    const a = testing.allocator;
    const one = "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}}";
    const two = "[" ++ one ++ ", {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Osaka\"}}]";
    const Case = struct { text: []const u8, calls: usize, preamble: []const u8 };
    const cases = [_]Case{
        .{ .text = one, .calls = 1, .preamble = "" },
        .{ .text = "[" ++ one ++ "]", .calls = 1, .preamble = "" },
        .{ .text = "Let me check.\n" ++ one, .calls = 1, .preamble = "Let me check." },
        .{ .text = "Let me check.\n" ++ two, .calls = 2, .preamble = "Let me check." },
        .{ .text = "```json\n" ++ one ++ "\n```", .calls = 1, .preamble = "" },
        .{ .text = "Let me check:\n```json\n" ++ one ++ "\n```", .calls = 1, .preamble = "Let me check:" },
        .{ .text = "Let me check:\n```json\n" ++ two ++ "\n```", .calls = 2, .preamble = "Let me check:" },
    };
    var bad: usize = 0;
    for (cases) |c| {
        const opens = false;
        const ask: Ask = .{ .stops = &.{}, .normalize = true, .has_tools = true, .tools_json = test_tools, .allow_parallel = true, .dialect = null, .opens_think = opens };
        const s = try settle(a, .{ .text = c.text, .source = .whole, .finish_reason = "stop", .finish_details = null }, ask);
        defer s.deinit(a);
        const r = s.reply orelse {
            bad += 1;
            std.debug.print("no reply for {f}\n", .{std.zig.fmtString(c.text)});
            continue;
        };
        if (r.calls.len != c.calls or !std.mem.eql(u8, r.preamble, c.preamble)) {
            bad += 1;
            std.debug.print("{f}: calls={d} preamble={f}\n", .{ std.zig.fmtString(c.text), r.calls.len, std.zig.fmtString(r.preamble) });
        }
        // What a stream sends as content before it holds: the split content of the longest prefix the hold released.
        var scan: think_tags.ToolHoldScan = .{};
        var sent: []const u8 = "";
        for (1..c.text.len + 1) |end| {
            if (think_tags.streamShouldBufferForToolsScan(c.text[0..end], &scan)) break;
            if (think_tags.endsWithPartialThinkOpen(c.text[0..end])) continue;
            sent = think_tags.splitThinkBlock(c.text[0..end], true, opens).content;
        }
        if (!std.mem.startsWith(u8, c.preamble, std.mem.trim(u8, sent, " \n"))) {
            bad += 1;
            std.debug.print("stream sent {f} before holding {f}\n", .{ std.zig.fmtString(sent), std.zig.fmtString(c.text) });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
