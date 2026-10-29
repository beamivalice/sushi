const std = @import("std");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const rp_mod = @import("reasoning_protocol.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_mod = @import("model.zig");
const effort_mod = @import("effort.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const generation_stream = @import("generation_stream.zig");
const GenErrorWire = generation_stream.GenErrorWire;
const sendGenerationError = generation_stream.sendGenerationError;
const nonStreamingViaScheduler = generation_stream.nonStreamingViaScheduler;
const StreamingTokenStream = generation_stream.StreamingTokenStream;
const requestSpecModes = generation_stream.requestSpecModes;
const noteMtpRefused = generation_stream.noteMtpRefused;
const pickStreamMode = generation_stream.pickStreamMode;
const contextOverflowMessage = generation_stream.contextOverflowMessage;
const formatTimingsObject = generation_stream.formatTimingsObject;
const sendStreamIdle = generation_stream.sendStreamIdle;
const beatStreamKeepalive = generation_stream.beatStreamKeepalive;
const http_conn = @import("http_conn.zig");
const Conn = http_conn.Conn;
const sendSseHeaders = http_conn.sendSseHeaders;
const request_media = @import("request_media.zig");
const responsesMediaRefusal = request_media.responsesMediaRefusal;
const decodeImageUrlInto = request_media.decodeImageUrlInto;
const PreparedMedia = request_media.PreparedMedia;
const prepareRequestMedia = request_media.prepareRequestMedia;
const visionPreprocFromConfig = request_media.visionPreprocFromConfig;
const request_policy = @import("request_policy.zig");
const resolveSamplingDefault = request_policy.resolveSamplingDefault;
const resolveThinkPenalty = request_policy.resolveThinkPenalty;
const armThinkPenalty = request_policy.armThinkPenalty;
const omittedMaxTokensDefault = request_policy.omittedMaxTokensDefault;
const resolveRequestMaxTokens = request_policy.resolveRequestMaxTokens;
const clampMaxTokens = request_policy.clampMaxTokens;
const effortWordOnly = request_policy.effortWordOnly;
const reasoningEffortFromWord = request_policy.reasoningEffortFromWord;
const EffortNotice = request_policy.EffortNotice;
const effortModelName = request_policy.effortModelName;
const clearEffortNotice = request_policy.clearEffortNotice;
const holdEffortNotice = request_policy.holdEffortNotice;
const withEffortNoticeJson = request_policy.withEffortNoticeJson;
const effortRefusalFor = request_policy.effortRefusalFor;
const responsesEnableThinking = request_policy.responsesEnableThinking;
const checkThinkingSupport = request_policy.checkThinkingSupport;
const schemaMasksThinking = request_policy.schemaMasksThinking;
const parseRepeatPenalty = request_policy.parseRepeatPenalty;
const parseAdditivePenalty = request_policy.parseAdditivePenalty;
const io_util = @import("io_util.zig");
const prefill_plan = @import("prefill_plan.zig");
const sampler = @import("sampler.zig");
const nowSecs = io_util.nowSecs;
const Stopwatch = io_util.Stopwatch;
const testing = std.testing;
const server_mod = @import("server.zig");
const SSE_ALLOW_HEADERS_DEFAULT = server_mod.SSE_ALLOW_HEADERS_DEFAULT;
const TOOL_CHOICE_UNDECLARED = server_mod.TOOL_CHOICE_UNDECLARED;
const admitMtpForCtx = server_mod.admitMtpForCtx;
const armCallForce = server_mod.armCallForce;
const armThinkBound = server_mod.armThinkBound;
const cachedFormatChat = server_mod.cachedFormatChat;
const checkAttentionMemory = server_mod.checkAttentionMemory;
const clampJsonI32 = server_mod.clampJsonI32;
const countRejected = server_mod.countRejected;
const defaultEnableMtpForStreaming = server_mod.defaultEnableMtpForStreaming;
const forceMtpFor = server_mod.forceMtpFor;
const getEffectiveContextLength = server_mod.getEffectiveContextLength;
const getTimeoutNs = server_mod.getTimeoutNs;
const jsonEscape = server_mod.jsonEscape;
const logHttpSseData = server_mod.logHttpSseData;
const nativeMeasuredMoeHead = server_mod.nativeMeasuredMoeHead;
const parseJsonFloatOpt = server_mod.parseJsonFloatOpt;
const parseJsonTopKOpt = server_mod.parseJsonTopKOpt;
const parseKvQuantOverride = server_mod.parseKvQuantOverride;
const parseRequestSeed = server_mod.parseRequestSeed;
const parseToolCallsForRequest = server_mod.parseToolCallsForRequest;
const promptOpensThink = server_mod.promptOpensThink;
const renderConfigFor = server_mod.renderConfigFor;
const renderQwenForResponsesTest = server_mod.renderQwenForResponsesTest;
const requestBillsMtpHead = server_mod.requestBillsMtpHead;
const requestCacheKey = server_mod.requestCacheKey;
const resolveReasoningProtocol = server_mod.resolveReasoningProtocol;
const rowMaskedDrafter = server_mod.rowMaskedDrafter;
const sendAnthropicEvent = server_mod.sendAnthropicEvent;
const sendErrorResponse = server_mod.sendErrorResponse;
const sendResponse = server_mod.sendResponse;
const shouldParseToolCalls = server_mod.shouldParseToolCalls;

/// In-memory store for OpenAI Responses API state (`store: true` requests).
/// Bounded LRU; lost on restart.
var global_response_store: ?responses_mod.ResponseStore = null;
var global_response_store_gpa: ?std.mem.Allocator = null;
const RESPONSE_STORE_CAP: usize = 256;
const DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS: u32 = 2048;

pub fn getOrInitResponseStore(io: std.Io, gpa: std.mem.Allocator) *responses_mod.ResponseStore {
    if (global_response_store == null) {
        global_response_store = responses_mod.ResponseStore.init(io, gpa, RESPONSE_STORE_CAP);
        global_response_store_gpa = gpa;
    }
    return &global_response_store.?;
}

pub fn deinitGlobalResponseStore() void {
    if (global_response_store) |*store| {
        store.deinit();
        global_response_store = null;
        global_response_store_gpa = null;
    }
}

/// The `/v1/responses` terminal `error` event payload (`sendResponsesEvent` adds `sequence_number`).
pub fn buildResponsesStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"error","code":"{s}","message":{s},"param":null}}
    , .{ w.openai_type, esc });
}

fn responsesStoresFinish(finish_reason: []const u8, requested: bool) bool {
    if (std.mem.eql(u8, finish_reason, "client_disconnect")) return false;
    return requested;
}

fn emitConstrainedResponses(a: std.mem.Allocator, stream: *Conn, seq: *u64, reasoning_id: *?[]u8, reasoning_index: *u32, reasoning_started: *bool, message_id: *?[]u8, message_index: *u32, message_started: *bool, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        if (!reasoning_started.*) {
            reasoning_id.* = try responses_mod.makeId(stream.io, a, "rs");
            reasoning_index.* = 0;
            try emitResponsesReasoningStart(a, stream, seq, reasoning_index.*, reasoning_id.*.?);
            reasoning_started.* = true;
        }
        try emitResponsesReasoningDelta(a, stream, seq, reasoning_index.*, reasoning_id.*.?, d.reasoning.items);
    }
    if (d.content.items.len > 0) {
        if (!message_started.*) {
            message_id.* = try responses_mod.makeId(stream.io, a, "msg");
            message_index.* = if (reasoning_started.*) 1 else 0;
            try emitResponsesMessageStart(a, stream, seq, message_index.*, message_id.*.?);
            message_started.* = true;
        }
        try emitResponsesMessageDelta(a, stream, seq, message_index.*, message_id.*.?, d.content.items);
    }
}

pub fn splitConstrainedResponse(allocator: std.mem.Allocator, routed: *?rp_mod.Delivery, sampling: sampler.SamplingParams, text: []const u8, payload_byte: ?usize, keep_markup: bool, opened: bool) !chat_mod.ThinkSplit {
    if (sampling.constraint) |c| {
        if (c.proto) |p| {
            routed.* = rp_mod.Delivery.init(p);
            const d = &routed.*.?;
            d.payload_byte = payload_byte;
            try d.feed(allocator, text);
            try d.finish(allocator);
            const reasoning = std.mem.trim(u8, d.reasoning.items, "\n ");
            return .{ .reasoning_content = if (reasoning.len == 0) null else reasoning, .content = d.content.items };
        }
    }
    return if (keep_markup) chat_mod.splitThinkBlockKeepingMarkup(text, true, opened) else chat_mod.splitThinkBlock(text, true, opened);
}

/// Wrap a Responses-API streaming event payload with `sequence_number` (which
/// the OpenAI Responses streaming schema requires on every event), then send.
/// The `payload` is expected to be a JSON object string ending in `}`.
pub fn sendResponsesEvent(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    event_name: []const u8,
    payload: []const u8,
) !void {
    if (payload.len < 2 or payload[0] != '{' or payload[payload.len - 1] != '}') {
        // Malformed payload — fall back to raw send (defensive; should not happen).
        try sendAnthropicEvent(stream, event_name, payload);
        return;
    }
    var num_buf: [32]u8 = undefined;
    const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{seq.*});
    seq.* += 1;
    // Detect whether the object has any existing fields (decides leading comma).
    var has_fields = false;
    for (payload[1 .. payload.len - 1]) |c| {
        if (!std.ascii.isWhitespace(c)) {
            has_fields = true;
            break;
        }
    }
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, payload[0 .. payload.len - 1]);
    if (has_fields) try buf.append(allocator, ',');
    try buf.appendSlice(allocator, "\"sequence_number\":");
    try buf.appendSlice(allocator, num_str);
    try buf.append(allocator, '}');
    try sendAnthropicEvent(stream, event_name, buf.items);
}

// ─── /v1/responses (OpenAI Responses API) ────────────────────────────────

pub fn handleResponsesGet(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.get(id)) |sr| {
        try sendResponse(stream, "200 OK", "application/json", sr.body_json);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}

pub fn handleResponsesDelete(allocator: std.mem.Allocator, stream: *Conn, id: []const u8) !void {
    const store = getOrInitResponseStore(stream.io, allocator);
    if (store.delete(id)) {
        const esc_id = try jsonEscape(allocator, id);
        defer allocator.free(esc_id);
        const body = try std.fmt.allocPrint(allocator,
            \\{{"id":{s},"object":"response","deleted":true}}
        , .{esc_id});
        defer allocator.free(body);
        try sendResponse(stream, "200 OK", "application/json", body);
    } else {
        try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Response not found", 404);
    }
}

fn responsesToolExists(tools_val: ?std.json.Value, name: []const u8) bool {
    const v = tools_val orelse return false;
    if (v != .array) return false;
    for (v.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (!std.mem.eql(u8, t, "function")) continue;
        const tool_name = if (tool.get("name")) |nv| (if (nv == .string) nv.string else "") else "";
        if (std.mem.eql(u8, tool_name, name)) return true;
    }
    return false;
}

fn buildResponsesJsonInstruction(allocator: std.mem.Allocator, schema_val: ?std.json.Value, tools_active: bool) ![]const u8 {
    var instruction_buf = std.ArrayList(u8).empty;
    defer instruction_buf.deinit(allocator);

    if (tools_active) {
        try instruction_buf.appendSlice(allocator, "If you answer without calling a function, respond with valid JSON only. Do not include markdown or explanation outside the JSON. If a function call is needed, call the function instead; this JSON format applies only to final assistant messages. ");
    } else {
        try instruction_buf.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");
    }

    if (schema_val) |sv| {
        var out: std.Io.Writer.Allocating = .init(allocator);
        defer out.deinit();
        var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
        sv.jsonStringify(&jws) catch {};
        try instruction_buf.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
        try instruction_buf.appendSlice(allocator, out.written());
    }

    return try allocator.dupe(u8, instruction_buf.items);
}

fn shouldInjectResponsesJsonInstruction(wants_json: bool, tools_active: bool, tool_choice_instruction: ?[]const u8) bool {
    if (!wants_json) return false;
    if (!tools_active) return true;

    // Required/forced tool calls must keep the prompt focused on tool syntax.
    // Structured-output instructions apply after the tool result is supplied.
    return tool_choice_instruction == null;
}

fn isJsonObjectString(allocator: std.mem.Allocator, text: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

/// Compact a conversation into a single opaque, round-trippable item.
///
/// The OpenAI Responses spec treats `encrypted_content` as provider-defined.
/// We synthesize a base64-encoded JSON envelope of the resolved messages so
/// the returned `compaction` item can be fed back into `response.create` as
/// an `input` item — exercising the full round-trip without an LLM call.
pub fn handleResponsesCompact(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses/compact -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // ── model (required) ──
    const model_val = root.get("model");
    const has_model = model_val != null and model_val.? == .string and model_val.?.string.len > 0;
    if (!has_model) {
        try sendErrorResponse(allocator, stream, "422 Unprocessable Entity", "invalid_request_error", "model is required", 422);
        return;
    }

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions (optional) ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id (optional) ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    var prev_messages: ?[]const chat_mod.Message = null;
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse → resolved message history ──
    // Compaction drops images, so none is decoded.
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, null, .{}) catch |err| {
        log.warn("POST /v1/responses/compact -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    // ── synthesize the opaque blob ──
    const blob = try responses_mod.encodeCompactionBlob(allocator, pi.messages.items);
    defer allocator.free(blob);

    // ── mint ids ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "comp");
    defer allocator.free(resp_id);
    const item_id = try responses_mod.makeId(stream.io, allocator, "cmp");
    defer allocator.free(item_id);

    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_item_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_item_id);
    const esc_blob = try jsonEscape(allocator, blob);
    defer allocator.free(esc_blob);

    const created_ts = nowSecs(stream.io);

    const out = try std.fmt.allocPrint(allocator,
        \\{{"id":{s},"object":"response.compaction","created_at":{d},"output":[{{"type":"compaction","id":{s},"encrypted_content":{s}}}],"usage":{{"input_tokens":0,"output_tokens":0,"total_tokens":0,"input_tokens_details":{{"cached_tokens":0}},"output_tokens_details":{{"reasoning_tokens":0}}}}}}
    , .{ esc_resp_id, created_ts, esc_item_id, esc_blob });
    defer allocator.free(out);

    log.info("POST /v1/responses/compact ({d} msgs -> {d}b blob)\n", .{ pi.messages.items.len, blob.len });
    try sendResponse(stream, "200 OK", "application/json", out);
}

/// Route entry for `/v1/responses`: gives the handler an error arm (it was the one generative
/// surface with no `catch` on its path) and owns the SSE sequence counter the terminal `error`
/// event must carry.
pub fn handleResponses(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    var seq_num: u64 = 0;
    handleResponsesInner(allocator, stream, body, lm, &seq_num) catch |err| {
        // The WS transport keeps its error: it has its own terminal frame and cleanup.
        if (stream.ws_mode != null) return err;
        log.err("  -> responses error: {s}\n", .{@errorName(err)});
        sendGenerationError(allocator, stream, err, .{ .responses = &seq_num }) catch {};
    };
}

fn handleResponsesInner(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
    /// Owned by `handleResponses` so its error arm can number the terminal `error` event.
    seq_num: *u64,
) !void {
    // No `lm.transformer.?` needed here.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/responses -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        log.warn("POST /v1/responses -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;

    // ── input (required) ──
    const input_val = root.get("input") orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "'input' is a required field", 400);
        return;
    };

    // ── instructions ──
    const instructions: ?[]const u8 = if (root.get("instructions")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── previous_response_id ──
    const prev_id: ?[]const u8 = if (root.get("previous_response_id")) |v|
        (if (v == .string) v.string else null)
    else
        null;

    // ── store (default true) ──
    const should_store = if (root.get("store")) |v| (if (v == .bool) v.bool else true) else true;

    // ── streaming ──
    const is_stream = if (root.get("stream")) |v| (v == .bool and v.bool) else false;

    // ── text.format (or chat-style response_format alias) ──
    var text_format = responses_mod.parseTextFormat(root.get("text"));
    if (text_format.schema_value == null and !std.mem.eql(u8, text_format.kind, "json_schema") and !std.mem.eql(u8, text_format.kind, "json_object")) {
        // Fall back to top-level `response_format` (some clients reuse their
        // chat-completions adapter for /v1/responses).
        const alias = responses_mod.parseResponseFormatAlias(root.get("response_format"));
        if (alias.schema_value != null or std.mem.eql(u8, alias.kind, "json_schema") or std.mem.eql(u8, alias.kind, "json_object")) {
            text_format = alias;
            log.debug("[responses] using top-level response_format as text.format alias\n", .{});
        }
    }
    if (std.mem.eql(u8, text_format.kind, "json_schema") and (text_format.schema_value == null or text_format.schema_value.? != .object)) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "text.format.schema must be a JSON object when format.type is json_schema (use {\"type\":\"json_object\"} for unconstrained JSON)", 400);
        return;
    }
    const wants_json = std.mem.eql(u8, text_format.kind, "json_schema") or std.mem.eql(u8, text_format.kind, "json_object");
    // Belt + braces, mirroring the chat-completions path: bare `json_object`
    // carries no schema, so without a synthesized permissive one there is no
    // grammar constraint at all and prompt-ignoring models (Gemma 3 wraps the
    // object in a ```json fence — caught live by llmprobe
    // structured-json-mode-valid, 2026-06-10) emit invalid JSON.
    var json_object_schema_holder: ?std.json.Parsed(std.json.Value) = null;
    defer if (json_object_schema_holder) |*p| p.deinit();
    if (text_format.schema_value == null and std.mem.eql(u8, text_format.kind, "json_object")) {
        json_object_schema_holder = std.json.parseFromSlice(std.json.Value, allocator, "{\"type\":\"object\",\"additionalProperties\":true}", .{}) catch null;
    }
    const grammar_schema_val: ?std.json.Value = if (json_object_schema_holder) |*p|
        p.value
    else
        text_format.schema_value;
    const has_current_tool_output = responses_mod.inputContainsFunctionCallOutput(input_val);

    // ── sampling params ──
    // Track whether the user explicitly supplied max_output_tokens so we can
    // echo `null` (vs. our internal default) in the response envelope.
    const req_max_output_tokens: ?u32 = blk: {
        const v = root.get("max_output_tokens") orelse root.get("max_tokens");
        // <= 0 is "auto" (the omitted-field default, the `--max-tokens` launch value when given).
        const r = resolveRequestMaxTokens(v, 0);
        break :blk if (r > 0) r else null;
    };
    const max_tokens: u32 = req_max_output_tokens orelse
        (if (wants_json) DEFAULT_STRUCTURED_OUTPUT_MAX_TOKENS else omittedMaxTokensDefault(getEffectiveContextLength(config)));
    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_mod.server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_mod.server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_mod.server_config.default_top_k, config.gen_top_k, 0);
    const frequency_penalty = parseAdditivePenalty(root, "frequency_penalty");
    const repeat_penalty = parseRepeatPenalty(root);
    const presence_penalty = parseAdditivePenalty(root, "presence_penalty");

    // ── echo fields (parsed but not consumed by generation; round-tripped
    // back into the response envelope to satisfy the OpenAI Responses schema) ──
    const top_logprobs_echo: u32 = if (root.get("top_logprobs")) |v| switch (v) {
        .integer => |i| if (i >= 0 and i <= 20) @intCast(i) else 0,
        else => 0,
    } else 0;
    const max_tool_calls_echo: ?u32 = if (root.get("max_tool_calls")) |v| switch (v) {
        .integer => |i| if (i >= 0) @as(?u32, @intCast(i)) else null,
        else => null,
    } else null;
    const truncation_echo: []const u8 = if (root.get("truncation")) |v|
        (if (v == .string and (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "disabled"))) v.string else "disabled")
    else
        "disabled";
    const parallel_tool_calls_echo: bool = if (root.get("parallel_tool_calls")) |v|
        (if (v == .bool) v.bool else true)
    else
        true;
    const background_echo: bool = if (root.get("background")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;
    // Honest rejection: there is no queue/poll machinery behind `background`.
    // Accepting it and running synchronously returns status "completed" on a
    // request the caller expects to come back "queued" — a silent lie strict
    // clients flag. The WS transport already rejects it at body-build time.
    if (background_echo) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "background mode is not supported; remove \"background\": true and run the request synchronously", 400);
        return;
    }
    const service_tier_echo: []const u8 = if (root.get("service_tier")) |v|
        (if (v == .string) v.string else "default")
    else
        "default";
    const safety_identifier_echo: ?[]const u8 = if (root.get("safety_identifier")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const prompt_cache_key_echo: ?[]const u8 = if (root.get("prompt_cache_key")) |v|
        (if (v == .string) v.string else null)
    else
        null;
    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // ── stop sequences ──
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |sv| switch (sv) {
        .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
        .array => |arr| for (arr.items) |it| {
            if (it == .string and it.string.len > 0) try stop_sequences.append(allocator, it.string);
        },
        else => {},
    };

    // ── reasoning ──
    // Budget precedence as on chat: reasoning_budget_tokens > the effort word's budget >
    // --reasoning-budget.
    var reasoning_cfg = responses_mod.parseReasoning(root.get("reasoning"), server_mod.server_config.default_reasoning_budget);
    if (reasoning_cfg.effort == null and (root.get("reasoning") == null or reasoning_cfg.enable)) reasoning_cfg.effort = model_mod.defaultEffortWord(config);
    var enable_thinking = reasoning_cfg.enable;
    var effort_budget: i32 = model_mod.defaultThinkBudget(config, server_mod.server_config.default_reasoning_budget);
    var effort_notice: ?EffortNotice = null;
    defer if (effort_notice) |n| n.deinit(allocator);
    defer clearEffortNotice(stream);
    if (reasoning_cfg.effort) |word| {
        const cfg = reasoningEffortFromWord(word, server_mod.server_config.default_reasoning_budget, effortWordOnly(allocator, lm, tok), effort_mod.effortArms(config.model_type)) catch {
            const msg = try effortRefusalFor(allocator, lm, word);
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
            return;
        };
        reasoning_cfg.effort = cfg.effort;
        enable_thinking = cfg.enable;
        effort_budget = cfg.budget;
        effort_notice = try holdEffortNotice(allocator, stream, effortModelName(lm), cfg);
    }
    const reasoning_budget: i32 = if (root.get("reasoning_budget_tokens")) |v| switch (v) {
        .integer => |i| clampJsonI32(i),
        else => effort_budget,
    } else effort_budget;

    // ── tools ──
    var tools_json: ?[]const u8 = null;
    var tools_json_owned = false;
    defer if (tools_json_owned) {
        if (tools_json) |tj| allocator.free(tj);
    };
    var has_tools = root.get("tools") != null;

    const tool_choice = chat_mod.parseToolChoice(root.get("tool_choice"));
    const tool_choice_instruction = try tool_choice.instruction(allocator);
    defer if (tool_choice_instruction) |ins| allocator.free(ins);
    if (tool_choice == .none) has_tools = false;

    if (has_tools) {
        if (root.get("tools")) |tools_val| if (tools_val == .array) {
            const reshaped = try responses_mod.buildToolsJson(allocator, tools_val.array);
            tools_json = reshaped;
            tools_json_owned = true;
            if (reshaped.len <= 2) has_tools = false; // "[]" — no function tools
        };
    }
    if (has_tools and chat_mod.toolChoiceNamesUndeclared(allocator, tool_choice, tools_json)) {
        log.warn("POST /v1/responses -> 400 (tool_choice names an undeclared function)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", TOOL_CHOICE_UNDECLARED, 400);
        return;
    }

    // Once tool outputs are supplied for a structured-output request, this turn
    // must produce the final JSON answer. Keeping tools available lets local
    // models reinterpret schema fields as fake tool calls and loop forever.
    const final_answer_mode = wants_json and has_current_tool_output;
    const active_has_tools = has_tools and !final_answer_mode;
    const active_tools_json: ?[]const u8 = if (active_has_tools) tools_json else null;
    const active_tool_choice_instruction: ?[]const u8 = if (active_has_tools) tool_choice_instruction else null;
    enable_thinking = responsesEnableThinking(root, enable_thinking, config.defaultEnableThinking());
    if (!try checkThinkingSupport(allocator, stream, config, enable_thinking, false)) return;
    if (final_answer_mode and has_tools) {
        log.info("[responses] final-answer mode - tools disabled after function_call_output\n", .{});
    }
    // ── model name ──
    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    // ── previous response — fetch stored history ──
    var prev_messages: ?[]const chat_mod.Message = null;
    if (prev_id) |pid| {
        const store = getOrInitResponseStore(stream.io, allocator);
        if (store.get(pid)) |sr| {
            prev_messages = sr.history;
        } else {
            try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "previous_response_id not found", 404);
            return;
        }
    }

    // ── parse input → messages ──
    var pi = responses_mod.parseInput(allocator, input_val, instructions, prev_messages, decodeImageUrlInto, visionPreprocFromConfig(config)) catch |err| {
        log.warn("POST /v1/responses -> 400 (input parse: {s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Failed to parse input", 400);
        return;
    };
    defer pi.deinit();

    if (responsesMediaRefusal(&pi)) |fault| {
        log.warn("POST /v1/responses -> 400 ({s})\n", .{fault.text()});
        try sendErrorResponse(allocator, stream, fault.status(), fault.errorType(), fault.text(), fault.code());
        return;
    }
    if (pi.messages.items.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "No valid messages found in 'input'", 400);
        return;
    }

    // ── inject json_schema instruction into system msg (prompt-side belt) ──
    var rf_allocs = std.ArrayList([]const u8).empty;
    defer {
        for (rf_allocs.items) |a| allocator.free(a);
        rf_allocs.deinit(allocator);
    }
    const inject_json_instruction = shouldInjectResponsesJsonInstruction(wants_json, active_has_tools, active_tool_choice_instruction);
    if (inject_json_instruction) {
        const owned = try buildResponsesJsonInstruction(allocator, grammar_schema_val, active_has_tools);
        try rf_allocs.append(allocator, owned);
        if (pi.messages.items.len > 0 and std.mem.eql(u8, pi.messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ pi.messages.items[0].content, owned });
            try rf_allocs.append(allocator, combined);
            pi.messages.items[0].content = combined;
        } else {
            try pi.messages.insert(allocator, 0, .{ .role = "system", .content = owned });
        }
        if (active_has_tools) {
            log.info("[grammar] injected JSON schema prompt while tools are available (mask deferred for tool calls)\n", .{});
        }
    }

    // ── format chat template ──
    // Iteration 1 timing + Iteration 2 cache. Responses sees the same
    // cache as chat-completions / messages because they all hash the
    // same canonical (messages, tools, flags) tuple.
    var tokenize_sw = Stopwatch.init(stream.io);
    const render_config = renderConfigFor(chat_config, root, model_settings.preserve_thinking_flag, config.preserve_thinking_override);
    var prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, pi.messages.items, active_tools_json, active_tool_choice_instruction, enable_thinking, reasoning_cfg.effort, false);
    var schema_proto: rp_mod.Protocol = undefined;
    var schema_proto_active = false;
    if (grammar_schema_val != null and !active_has_tools and enable_thinking) {
        schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, true);
    }
    switch (schemaMasksThinking(
        grammar_schema_val != null,
        active_has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            schema_proto_active = false;
            allocator.free(prompt_ids_raw);
            enable_thinking = false;
            prompt_ids_raw = try cachedFormatChat(allocator, stream.io, lm, tok, &render_config, pi.messages.items, active_tools_json, active_tool_choice_instruction, false, reasoning_cfg.effort, false);
            log.info("[grammar] reasoning protocol unsupported for this model/prompt; rerendered with thinking off\n", .{});
        },
        .no_mask, .token_zero => schema_proto_active = false,
    }
    if (grammar_schema_val != null and !active_has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, prompt_ids_raw, &schema_proto, false)) {
            schema_proto = rp_mod.Protocol.jsonOnly();
        }
        schema_proto_active = true;
    }
    const tokenize_ns = tokenize_sw.read();
    log.info("POST /v1/responses ({d} msgs, max_out={d}, temp={d:.2}, stream={}, thinking={}, prev={?s})\n", .{
        pi.messages.items.len, max_tokens, temperature, is_stream, enable_thinking, prev_id,
    });

    // ── media ──
    const cache_key = requestCacheKey(root);
    var req_media: PreparedMedia = .{};
    defer req_media.deinit(allocator);
    switch (try prepareRequestMedia(allocator, lm, pi.messages.items, &prompt_ids_raw, getEffectiveContextLength(config))) {
        .ready => |r| req_media = r,
        .refused => |fault| {
            allocator.free(prompt_ids_raw);
            log.warn("POST /v1/responses -> {d} ({s})\n", .{ fault.code(), fault.text() });
            try sendErrorResponse(allocator, stream, fault.status(), fault.errorType(), fault.text(), fault.code());
            return;
        },
    }
    const prompt_ids = prompt_ids_raw;
    defer allocator.free(prompt_ids);

    // ── context limit ──
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }
    const kv_quant_override = parseKvQuantOverride(root);
    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    if (expert_stream_mod.mtpRefusal(config.expert_streaming, if (root.get("enable_mtp")) |v| (v == .bool and v.bool) else false, lm.mtp != null)) |why| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
        return;
    }
    var enable_mtp_resp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtpForStreaming(lm.mtp != null, config.isMoe(), forceMtpFor(config), nativeMeasuredMoeHead(lm), model_mod.isServedArch(config.model_type), config.expert_streaming and !config.stream_mtp_head);
    if (enable_mtp_resp and lm.mtp == null) enable_mtp_resp = false;
    enable_mtp_resp = admitMtpForCtx(enable_mtp_resp, prompt_ids.len);

    // Drafter (Responses-side parsing). Same disable rules as the chat
    // and Anthropic parse sites.
    const lm_default_enable_drafter_resp: bool = lm.dflash != null;
    var enable_drafter_resp: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter_resp;
    if (enable_drafter_resp and lm.dflash == null) enable_drafter_resp = false;

    // Check if attention computation would exceed GPU memory.
    const bill_head = requestBillsMtpHead(enable_drafter_resp, enable_mtp_resp, lm.dflash != null, lm.mtp != null, .{ .repeat_penalty = repeat_penalty, .presence_penalty = presence_penalty, .frequency_penalty = frequency_penalty }, 0);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, lm.id, false, kv_quant_override, prefill_plan.visionPrefillUnchunked(req_media.deferred != null), bill_head)) return;

    // ── sampling ──
    var sampling = sampler.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = repeat_penalty,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .seed = seed,
    };
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();
    if (grammar_schema_val) |sv| {
        if (active_has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try lm.grammarTokenBytes(allocator, stream.io);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                var deferred = false;
                if (schema_proto_active) {
                    sc.deferWithProtocol(&schema_proto);
                    deferred = schema_proto.startState().phase != .json_body;
                }
                sc.constraint.row_masks = rowMaskedDrafter(lm);
                sampling.constraint = &sc.constraint;
                if (deferred) {
                    log.info("[grammar] deferring JSON schema until the reasoning boundary (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                } else {
                    log.info("[grammar] enforcing JSON schema from token 0 (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                }
            } else |err| {
                log.warn("[grammar] schema parse failed ({s})\n", .{@errorName(err)});
            }
        }
    }

    // Enforced at decode: the bound closes the thought, so this surface parses a closed block.
    sampling.think_penalty = armThinkPenalty(allocator, lm, tok, prompt_ids, enable_thinking, resolveThinkPenalty(root, model_settings.think_penalty_flag, config.think_penalty_override));
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    if (think_bound) |*tb| sampling.think_bound = tb;
    var call_force = if (active_has_tools and lm.transformer != null)
        armCallForce(allocator, tok, chat_config, prompt_ids, enable_thinking, tool_choice, active_tools_json)
    else
        null;
    defer if (call_force) |*cf| cf.deinit(allocator);
    if (call_force) |*cf| sampling.call_force = cf;

    // ── pre-allocate response id (used in streaming envelopes too) ──
    const resp_id = try responses_mod.makeId(stream.io, allocator, "resp");
    defer allocator.free(resp_id);
    const esc_resp_id = try jsonEscape(allocator, resp_id);
    defer allocator.free(esc_resp_id);
    const esc_model = try jsonEscape(allocator, model_name);
    defer allocator.free(esc_model);

    // ── pre-render request echoes (live for both the streaming skeleton
    // and the final completed envelope) ──
    const echo_tools_json = try renderResponsesToolsEcho(allocator, root.get("tools"));
    defer allocator.free(echo_tools_json);
    const echo_tool_choice_json = try renderResponsesToolChoiceEcho(allocator, root.get("tool_choice"));
    defer allocator.free(echo_tool_choice_json);
    const echo_text_json = try renderResponsesTextEcho(allocator, root);
    defer allocator.free(echo_text_json);
    const echo_reasoning_json = try renderResponsesReasoningEcho(allocator, root);
    defer allocator.free(echo_reasoning_json);
    const echo_metadata_json = try renderResponsesMetadataEcho(allocator, root);
    defer allocator.free(echo_metadata_json);

    const response_echo = ResponseEcho{
        .tools_json = echo_tools_json,
        .tool_choice_json = echo_tool_choice_json,
        .text_json = echo_text_json,
        .reasoning_json = echo_reasoning_json,
        .metadata_json = echo_metadata_json,
        .instructions = instructions,
        .truncation = truncation_echo,
        .service_tier = service_tier_echo,
        .safety_identifier = safety_identifier_echo,
        .prompt_cache_key = prompt_cache_key_echo,
        .temperature = temperature,
        .top_p = top_p,
        .presence_penalty = presence_penalty,
        .frequency_penalty = frequency_penalty,
        .top_logprobs = top_logprobs_echo,
        .parallel_tool_calls = parallel_tool_calls_echo,
        .background = background_echo,
        .max_output_tokens = req_max_output_tokens,
        .max_tool_calls = max_tool_calls_echo,
    };

    // SSE event sequence counter, owned by the wrapper (see `handleResponses`).

    // ── streaming: send SSE headers + response.created + response.in_progress ──
    if (is_stream) {
        if (stream.ws_mode == null) {
            try sendSseHeaders(stream, "responses", SSE_ALLOW_HEADERS_DEFAULT);
        }

        // Skeleton envelope (status:in_progress, output:[])
        // Timings are all zero on the skeleton — the real numbers appear on
        // the final `response.completed` envelope after generation.
        const skel = try buildResponsesEnvelope(
            stream.io,
            allocator,
            esc_resp_id,
            esc_model,
            "in_progress",
            "[]",
            0,
            0,
            0,
            0,
            should_store,
            prev_id,
            false,
            false,
            response_echo,
            0,
            0,
            tokenize_ns,
            0,
        );
        defer allocator.free(skel);
        const skel_noted = try withEffortNoticeJson(allocator, skel, stream.effort_notice_json);
        defer allocator.free(skel_noted);
        const created_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.created\",\"response\":{s}}}", .{skel_noted});
        defer allocator.free(created_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.created", created_payload);
        const ip_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.in_progress\",\"response\":{s}}}", .{skel_noted});
        defer allocator.free(ip_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.in_progress", ip_payload);
    }

    // ── generate (streaming path: emit deltas live; non-streaming: existing) ──
    const eos_slice = config.eosTokenSlice();

    // Streaming bookkeeping. When we live-stream a reasoning or message item,
    // we record its id + index so the post-loop block emits just the END
    // events instead of full start+delta+end via emit*Events.
    var streamed_reasoning_id: ?[]u8 = null;
    var streamed_reasoning_index: u32 = 0;
    var streamed_reasoning_started = false;
    var streamed_message_id: ?[]u8 = null;
    var streamed_message_index: u32 = 0;
    var streamed_message_started = false;
    defer if (streamed_reasoning_id) |id| allocator.free(id);
    defer if (streamed_message_id) |id| allocator.free(id);

    // Wave 1.A: per-request KV-quant override (Responses path mirror).

    var result: generate_mod.GenerationResult = undefined;
    if (is_stream) {
        // Pick speculative-decoding mode for the streaming Responses path.
        const stream_mode = pickStreamMode(enable_drafter_resp, enable_mtp_resp, lm.dflash != null, lm.mtp != null, sampling, 0);
        if (stream_mode == .drafter) log.info("  drafter=enabled (streaming responses, block_size={d})\n", .{lm.drafter_block_size});
        if (stream_mode == .mtp) log.info("  mtp=enabled (streaming responses, depth={d})\n", .{lm.mtp_depth});
        noteMtpRefused(enable_mtp_resp, lm.mtp != null, sampling, 0);

        var slot_handle: ?*scheduler_mod.Slot = null;
        defer if (slot_handle) |s| server_mod.global_scheduler.?.complete(s);

        // Transfer vision ownership into the slot.
        const slot_ve_resp = req_media.deferred;
        req_media.deferred = null;
        const slot_mrope_resp = req_media.mrope;
        req_media.mrope = .{};
        const sch = server_mod.global_scheduler.?;
        slot_handle = try sch.submit(.{
            .model = lm,
            .prompt_ids = prompt_ids,
            .full_prompt = prompt_ids,
            .cached_tokens = 0,
            .has_tools = active_has_tools,
            .enable_thinking = enable_thinking,
            .sampling = sampling,
            .eos_token_ids = eos_slice,
            .max_tokens = effective_max_tokens,
            .timeout_ns = getTimeoutNs(),
            .enable_drafter = stream_mode == .drafter,
                .dflash = if (stream_mode == .drafter) lm.dflash else null,
            .drafter_block_size = lm.drafter_block_size,
            .allow_batch_mtp = allow_batch_mtp,
            .enable_mtp = stream_mode == .mtp,
            .mtp = if (stream_mode == .mtp) lm.mtp else null,
            .mtp_depth = lm.mtp_depth,
            .deferred_media = slot_ve_resp,
            .vision_key = prefix_cache_mod.chainKey(req_media.chain),
            .media_chain = req_media.chain,
            .cache_key = cache_key,
            .mrope_pos = slot_mrope_resp.pos,
            .mrope_total = slot_mrope_resp.total,
            .mrope_delta = slot_mrope_resp.delta,
            .logprobs_n = 0,
            .kv_quant_config = kv_quant_override,
        });
        var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_slice);
        defer ts.deinit(allocator);

        var raw_buf = std.ArrayList(u8).empty;
        defer raw_buf.deinit(allocator);
        var token_ids_buf = std.ArrayList(u32).empty;
        defer token_ids_buf.deinit(allocator);

        var stopped = false;
        var stop_gate = stop_seq_mod.Gate{ .stops = stop_sequences.items };
        defer stop_gate.deinit(allocator);
        var tail_flush = false; // the generation ended: this turn delivers the gate's held tail
        var client_gone = false;
        // Prompt-opened thinking streams as reasoning even when the request
        // asked for thinking off — generated reasoning is delivered, never
        // dropped (thinking-off is enforced prompt-side, noThinkTailSuffix).
        // Prompt-derived, never flag-derived — see the chat streaming arm.
        const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
        var in_think_block = promptOpensThink(allocator, tok, prompt_ids) and !constrained_proto;
        var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
        defer if (delivery) |*d| d.deinit(allocator);
        var channel_armed = false; // the `<|channel>` marker that may open a Gemma 4 thought
        var think_buf = std.ArrayList(u8).empty;
        defer think_buf.deinit(allocator);
        var thought_shipped = false; // this thought has streamed a delta (`think_tags.openThoughtFlush`)
        var skipped_think_open = false;
        var live_output_index: u32 = 0;

        while (true) {
            // A stop cut resolved on the previous token ends the turn here.
            if (stopped) break;
            const token_id: u32 = switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
                .token => |t| t,
                .done => if (stop_gate.hasHeld()) flush: {
                    tail_flush = true;
                    break :flush 0;
                } else break,
                .idle => {
                    // No tokens yet (long prefill). Probe the peer: an abandoned
                    // request must cancel instead of grinding a ghost prefill
                    // (Claude Code retries pile up serially otherwise), and the
                    // keepalive stops clients timing out on stream silence.
                    if (stream.peerClosed()) {
                        log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                        slot_handle.?.cancel();
                        client_gone = true;
                        break;
                    }
                    sendStreamIdle(stream, slot_admission.queueWait(server_mod.global_scheduler.?, slot_handle.?)) catch {
                        log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                        slot_handle.?.cancel();
                        client_gone = true;
                        break;
                    };
                    continue;
                },
            };
            if (stream.peerClosed()) {
                slot_handle.?.cancel();
                client_gone = true;
                break;
            }
            const token_text = if (tail_flush) try stop_gate.finish(allocator) else blk: {
                try token_ids_buf.append(allocator, token_id);
                const raw_decoded = try tok.decode(allocator, &[_]u32{token_id});
                defer allocator.free(raw_decoded);
                if (delivery) |*d| d.noteToken(raw_decoded.len, slot_handle.?.constraintPayloadStart());
                break :blk try stop_gate.push(allocator, raw_decoded);
            };
            defer allocator.free(token_text);

            stopped = stop_gate.matched != null;
            if (stopped) slot_handle.?.stop_hit.store(true, .release);
            if (token_text.len == 0) {
                if (stopped) break;
                continue;
            }

            try raw_buf.appendSlice(allocator, token_text);

            // Beat BEFORE the tool early-continue below: a tool-active request
            // emits nothing for its whole generation, and the thinking branch
            // holds until its close tag. Both look identical to a dead server
            // from the client's socket.
            beatStreamKeepalive(stream, .sse_comment) catch {
                log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                slot_handle.?.cancel();
                client_gone = true;
                break;
            };

            // Tool-active requests buffer entirely — we cannot emit text deltas
            // before knowing whether the output is a tool call.
            if (active_has_tools) continue;

            if (delivery) |*d| {
                try d.feed(allocator, token_text);
                try emitConstrainedResponses(allocator, stream, seq_num, &streamed_reasoning_id, &streamed_reasoning_index, &streamed_reasoning_started, &streamed_message_id, &streamed_message_index, &streamed_message_started, d);
                continue;
            }

            if (in_think_block) {
                try think_buf.appendSlice(allocator, token_text);

                // Skip a literal think opener if the template did not pre-inject one.
                if (!skipped_think_open and (think_buf.items.len >= 7 or !chat_mod.thinkOpenerPossible(think_buf.items))) {
                    if (std.mem.startsWith(u8, think_buf.items, "<think>")) {
                        var skip: usize = 7;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                        var skip: usize = 17;
                        while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                        const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                        think_buf.clearAndFree(allocator);
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                        skipped_think_open = true;
                    } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                        // partial prefix of channel-thought tag; wait for more
                    } else {
                        skipped_think_open = true;
                    }
                }

                // Detect close tag (</think> or <channel|>).
                const tp = std.mem.indexOf(u8, think_buf.items, "</think>");
                const cp = std.mem.indexOf(u8, think_buf.items, "<channel|>");
                const close_match: ?struct { pos: usize, tag: []const u8 } = blk: {
                    if (tp == null and cp == null) break :blk null;
                    if (tp == null) break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                    if (cp == null) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    if (tp.? <= cp.?) break :blk .{ .pos = tp.?, .tag = "</think>" };
                    break :blk .{ .pos = cp.?, .tag = "<channel|>" };
                };

                if (close_match) |m| {
                    const before = chat_mod.closedThoughtDelta(think_buf.items[0..m.pos], thought_shipped);
                    thought_shipped = false;
                    if (before.len > 0) {
                        if (!streamed_reasoning_started) {
                            streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                            streamed_reasoning_index = live_output_index;
                            try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                            streamed_reasoning_started = true;
                        }
                        try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, before);
                    }
                    if (streamed_reasoning_started) live_output_index += 1;

                    const after = m.pos + m.tag.len;
                    var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                    if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                        content_after = content_after[11..];
                    } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                        content_after = content_after[10..];
                    }
                    content_after = std.mem.trimStart(u8, content_after, "\n ");
                    if (content_after.len > 0) {
                        if (!streamed_message_started) {
                            streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                            streamed_message_index = live_output_index;
                            try emitResponsesMessageStart(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?);
                            streamed_message_started = true;
                        }
                        try emitResponsesMessageDelta(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, content_after);
                    }
                    think_buf.clearRetainingCapacity();
                    in_think_block = false;
                } else if (skipped_think_open) {
                    // Flush what the closed thought will certainly deliver: a partial
                    // close tag, a trailing "\n " run and a split code point wait.
                    const flush = chat_mod.openThoughtFlush(think_buf.items, thought_shipped);
                    const safe_len = flush.skip + flush.ship;
                    if (safe_len > 0) {
                        if (flush.ship > 0) {
                            if (!streamed_reasoning_started) {
                                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                                streamed_reasoning_index = live_output_index;
                                try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                                streamed_reasoning_started = true;
                            }
                            try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_buf.items[flush.skip..safe_len]);
                            thought_shipped = true;
                        }
                        const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                        think_buf.clearRetainingCapacity();
                        try think_buf.appendSlice(allocator, remaining);
                        allocator.free(remaining);
                    }
                }
            } else {
                // Skip Gemma 4 channel tags that may leak after the thinking block.
                // A think block the MODEL opens: seeded like a template-injected one.
                // Checked before the marker skip, which would swallow a bare `<think>`.
                if (!streamed_message_started) if (chat_mod.modelThinkOpener(channel_armed, token_text)) |opener| {
                    channel_armed = false;
                    in_think_block = true;
                    try think_buf.appendSlice(allocator, opener);
                    continue;
                };
                if (chat_mod.isChannelMarkerToken(token_text)) {
                    channel_armed = !streamed_message_started and std.mem.eql(u8, token_text, "<|channel>");
                    continue;
                }
                channel_armed = false;
                if (!streamed_message_started) {
                    streamed_message_id = try responses_mod.makeId(stream.io, allocator, "msg");
                    streamed_message_index = live_output_index;
                    try emitResponsesMessageStart(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?);
                    streamed_message_started = true;
                }
                try emitResponsesMessageDelta(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, token_text);
            }
        }

        if (!client_gone) {
            if (delivery) |*d| {
                d.clearOutput();
                try d.finish(allocator);
                try emitConstrainedResponses(allocator, stream, seq_num, &streamed_reasoning_id, &streamed_reasoning_index, &streamed_reasoning_started, &streamed_message_id, &streamed_message_index, &streamed_message_started, d);
            }
        }

        // Flush any remaining think buffer (no close tag found) as reasoning.
        const think_tail = chat_mod.cutThoughtDelta(think_buf.items, skipped_think_open, thought_shipped);
        if (!client_gone and in_think_block and think_tail.len > 0 and !active_has_tools) {
            if (!streamed_reasoning_started) {
                streamed_reasoning_id = try responses_mod.makeId(stream.io, allocator, "rs");
                streamed_reasoning_index = live_output_index;
                try emitResponsesReasoningStart(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?);
                streamed_reasoning_started = true;
            }
            try emitResponsesReasoningDelta(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, think_tail);
        }

        ts.finalize();

        if (client_gone) {
            // Peer disconnected mid-decode. We've already cancelled the slot;
            // tear down without spending more I/O on terminal envelope events
            // (which would just fail or pile into a dead socket buffer).
            log.info("  <- {d}+{d} tokens streamed [client_disconnect]\n", .{ ts.prompt_tokens, ts.completion_tokens });
            return;
        }

        result = .{
            .constraint_payload_byte = if (delivery) |d| d.payload_byte else null,
            .text = try raw_buf.toOwnedSlice(allocator),
            .token_ids = try token_ids_buf.toOwnedSlice(allocator),
            .prompt_tokens = ts.prompt_tokens,
            .completion_tokens = ts.completion_tokens,
            .finish_reason = if (stopped) "stop" else ts.finish_reason,
            .prefill_tps = 0.0,
            .decode_tps = 0.0,
            .finish_details = ts.finish_details,
        };
    } else {
        // Non-streaming Responses: `requestSpecModes` (DFlash > MTP > drafter
        // > drafter) so /v1/responses gets the same speedup as /v1/chat/completions.
        const spec = requestSpecModes(enable_drafter_resp, enable_mtp_resp, lm.dflash != null, lm.mtp != null, sampling, 0);
        const use_mtp = spec.use_mtp;
        const use_drafter = spec.use_drafter;
        // Transfer vision ownership into the slot.
        const slot_ve_ns = req_media.deferred;
        req_media.deferred = null;
        const slot_mrope_ns = req_media.mrope;
        req_media.mrope = .{};
        // Propagates to `handleResponses`' one error arm, shared with the streaming half.
        result = try nonStreamingViaScheduler(allocator, server_mod.global_scheduler.?, lm, tok, prompt_ids, prompt_ids, effective_max_tokens, sampling, eos_slice, 0, active_has_tools, enable_thinking, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve_ns, req_media.chain, cache_key, slot_mrope_ns, 0, kv_quant_override, stream, .{ .stops = stop_sequences.items });
    }
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // ── apply stop sequences ──
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    if (stop_seq_mod.earliest(final_text, stop_sequences.items)) |hit| {
        final_text = final_text[0..hit.index];
        finish_reason = "stop";
    }

    if (!responsesStoresFinish(finish_reason, true)) {
        log.info("  <- {d}+{d} tokens [client_disconnect]\n", .{ result.prompt_tokens, result.completion_tokens });
        return;
    }

    const status_str: []const u8 = if (std.mem.eql(u8, finish_reason, "length")) "incomplete" else "completed";

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels
    // mid-turn). A span-routed constrained response skips this: its payload
    // is authoritative and marker text inside it is DATA.
    var normalized_text: ?[]u8 = null;
    defer if (normalized_text) |n| allocator.free(n);
    if (sampling.constraint == null or sampling.constraint.?.proto == null) {
        normalized_text = try chat_mod.normalizeEmbeddedThinkBlocks(allocator, final_text);
        if (normalized_text) |n| final_text = n;
    }

    // ── split thinking & tool calls ──
    // Reasoning the model actually generated is always delivered — the
    // request's thinking flag shaped the PROMPT (think_tags.noThinkTailSuffix),
    // never the delivery.
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, false, promptOpensThink(allocator, tok, prompt_ids));
    const reasoning_text: ?[]const u8 = think_split.reasoning_content;
    const visible_text: []const u8 = think_split.content;

    var tool_calls: ?[]chat_mod.ParsedToolCall = null;
    if (active_has_tools and shouldParseToolCalls(result.finish_details)) {
        tool_calls = try parseToolCallsForRequest(allocator, final_text, active_tools_json, parallel_tool_calls_echo);
    }
    defer if (tool_calls) |tcs| {
        for (tcs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(tcs);
    };

    // ── build output[] array (shared between streaming + non-streaming) ──
    var out_buf = std.ArrayList(u8).empty;
    defer out_buf.deinit(allocator);
    try out_buf.append(allocator, '[');
    var emitted: usize = 0;
    var output_index: u32 = 0;
    var emitted_tool_calls = std.ArrayList(chat_mod.ToolCall).empty;
    defer {
        for (emitted_tool_calls.items) |tc| allocator.free(tc.id);
        emitted_tool_calls.deinit(allocator);
    }

    if (reasoning_text) |rt| if (rt.len > 0) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_reasoning_started) {
            // Live deltas already streamed; emit just the closing events with
            // the canonical reasoning text from splitThinkBlock.
            try responses_mod.appendReasoningItem(allocator, &out_buf, streamed_reasoning_id.?, rt);
            try emitResponsesReasoningEnd(allocator, stream, seq_num, streamed_reasoning_index, streamed_reasoning_id.?, rt);
        } else {
            const rid = try responses_mod.makeId(stream.io, allocator, "rs");
            defer allocator.free(rid);
            try responses_mod.appendReasoningItem(allocator, &out_buf, rid, rt);
            if (is_stream) {
                try emitResponsesReasoningEvents(allocator, stream, seq_num, output_index, rid, rt);
            }
        }
        emitted += 1;
        output_index += 1;
    };

    if (tool_calls) |tcs| if (tcs.len > 0) {
        for (tcs) |tc| {
            if (!responsesToolExists(root.get("tools"), tc.name)) {
                log.warn("[responses] dropping undeclared tool call: {s}\n", .{tc.name});
                continue;
            }
            if (!isJsonObjectString(allocator, tc.arguments)) {
                log.warn("[responses] dropping tool call with non-object arguments: {s}\n", .{tc.name});
                continue;
            }
            const fc_id = try responses_mod.makeId(stream.io, allocator, "fc");
            defer allocator.free(fc_id);
            const call_id = try responses_mod.makeId(stream.io, allocator, "call");
            defer allocator.free(call_id);
            const stored_call_id = try allocator.dupe(u8, call_id);
            emitted_tool_calls.append(allocator, .{ .id = stored_call_id, .name = tc.name, .arguments = tc.arguments }) catch |err| {
                allocator.free(stored_call_id);
                return err;
            };
            if (emitted > 0) try out_buf.append(allocator, ',');
            try responses_mod.appendFunctionCallItem(allocator, &out_buf, fc_id, call_id, tc.name, tc.arguments);
            emitted += 1;
            if (is_stream) {
                try emitResponsesFunctionCallEvents(allocator, stream, seq_num, output_index, fc_id, call_id, tc.name, tc.arguments);
            }
            output_index += 1;
        }
    };

    const has_tool_calls = emitted_tool_calls.items.len > 0;
    if (!has_tool_calls) {
        if (emitted > 0) try out_buf.append(allocator, ',');
        if (is_stream and streamed_message_started) {
            // Live deltas already streamed; emit just the closing events.
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, streamed_message_id.?, visible_text);
            try emitResponsesMessageEnd(allocator, stream, seq_num, streamed_message_index, streamed_message_id.?, visible_text);
        } else {
            const mid = try responses_mod.makeId(stream.io, allocator, "msg");
            defer allocator.free(mid);
            try responses_mod.appendOutputTextMessage(allocator, &out_buf, mid, visible_text);
            if (is_stream) {
                try emitResponsesMessageEvents(allocator, stream, seq_num, output_index, mid, visible_text);
            }
        }
        emitted += 1;
        output_index += 1;
    }

    try out_buf.append(allocator, ']');

    // ── envelope ──
    const is_incomplete = std.mem.eql(u8, status_str, "incomplete");
    const is_completed_status = std.mem.eql(u8, status_str, "completed") or is_incomplete;
    // reasoning token count for usage.output_tokens_details (re-encode of the
    // split reasoning text — exact modulo merge boundaries).
    const reasoning_tok_count: u32 = blk: {
        const rt = reasoning_text orelse break :blk 0;
        const rids = tok.encode(allocator, rt) catch break :blk 0;
        defer allocator.free(rids);
        break :blk @intCast(rids.len);
    };
    const envelope = try buildResponsesEnvelope(
        stream.io,
        allocator,
        esc_resp_id,
        esc_model,
        status_str,
        out_buf.items,
        result.prompt_tokens,
        result.completion_tokens,
        result.cached_tokens,
        reasoning_tok_count,
        should_store,
        prev_id,
        is_incomplete,
        is_completed_status,
        response_echo,
        result.prefill_ns,
        result.decode_ns,
        tokenize_ns,
        result.completion_tokens,
    );
    defer allocator.free(envelope);
    const envelope_noted = try withEffortNoticeJson(allocator, envelope, stream.effort_notice_json);
    defer allocator.free(envelope_noted);

    // ── store response ──
    if (responsesStoresFinish(finish_reason, should_store)) {
        const stored_tool_calls: ?[]const chat_mod.ToolCall = if (emitted_tool_calls.items.len > 0) emitted_tool_calls.items else null;
        storeResponse(stream.io, allocator, resp_id, model_name, status_str, envelope_noted, pi.messages.items, visible_text, reasoning_text, stored_tool_calls, result.finish_details) catch |err| {
            log.warn("[responses] store failed: {s}\n", .{@errorName(err)});
        };
    }

    if (is_stream) {
        const completed_payload = try std.fmt.allocPrint(allocator, "{{\"type\":\"response.completed\",\"response\":{s}}}", .{envelope_noted});
        defer allocator.free(completed_payload);
        try sendResponsesEvent(allocator, stream, seq_num, "response.completed", completed_payload);
        // OpenAI terminates the Responses HTTP SSE stream with the same
        // `data: [DONE]` sentinel as chat completions; generic SSE middleware
        // keys stream end off it. The WS transport must NOT get one — its
        // per-response terminator is the `response.completed` event, and a
        // trailing [DONE] frame would be misread as the NEXT turn's marker
        // on a chained session (see the WS turn loop below).
        if (stream.ws_mode == null) {
            logHttpSseData("[DONE]");
            try stream.writeAll("data: [DONE]\n\n");
        }
    } else {
        try sendResponse(stream, "200 OK", "application/json", envelope_noted);
    }
}

// ─── Response envelope echo helpers ──────────────────────────────────────
// The OpenAI Responses API ResponseResource schema requires the response to
// echo most request configuration (tools, tool_choice, text.format, reasoning,
// metadata, etc.). These helpers re-render those values from the parsed
// request JSON in the exact shape the schema demands. Caller owns the result.

fn renderResponsesToolsEcho(allocator: std.mem.Allocator, tools_val: ?std.json.Value) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    var emitted: usize = 0;
    if (tools_val) |tv| if (tv == .array) {
        for (tv.array.items) |tool_val| {
            if (tool_val != .object) continue;
            const tool = tool_val.object;
            const t = if (tool.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (!std.mem.eql(u8, t, "function")) continue;

            // Accept both flat (Responses API) and nested-under-"function" (chat-completions) shapes.
            var fn_obj_opt: ?std.json.ObjectMap = null;
            if (tool.get("function")) |fv| if (fv == .object) {
                fn_obj_opt = fv.object;
            };

            const name_v: ?std.json.Value = tool.get("name") orelse if (fn_obj_opt) |fo| fo.get("name") else null;
            const desc_v: ?std.json.Value = tool.get("description") orelse if (fn_obj_opt) |fo| fo.get("description") else null;
            const params_v: ?std.json.Value = tool.get("parameters") orelse if (fn_obj_opt) |fo| fo.get("parameters") else null;
            const strict_v: ?std.json.Value = tool.get("strict") orelse if (fn_obj_opt) |fo| fo.get("strict") else null;

            if (emitted > 0) try buf.append(allocator, ',');
            emitted += 1;
            try buf.appendSlice(allocator, "{\"type\":\"function\",\"name\":");
            if (name_v) |nv| if (nv == .string) {
                const e = try jsonEscape(allocator, nv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "\"\"");
            } else {
                try buf.appendSlice(allocator, "\"\"");
            }
            try buf.appendSlice(allocator, ",\"description\":");
            if (desc_v) |dv| if (dv == .string) {
                const e = try jsonEscape(allocator, dv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"parameters\":");
            if (params_v) |pv| if (pv == .object) {
                try responses_mod.serializeJsonValue(allocator, &buf, pv);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"strict\":");
            if (strict_v) |sv| if (sv == .bool) {
                try buf.appendSlice(allocator, if (sv.bool) "true" else "false");
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.append(allocator, '}');
        }
    };
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

fn renderResponsesToolChoiceEcho(allocator: std.mem.Allocator, tc_val: ?std.json.Value) ![]const u8 {
    if (tc_val) |v| switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required")) {
                return try std.fmt.allocPrint(allocator, "\"{s}\"", .{s});
            }
        },
        .object => |obj| {
            const t = if (obj.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (std.mem.eql(u8, t, "function")) {
                var name: []const u8 = "";
                if (obj.get("name")) |x| {
                    if (x == .string) name = x.string;
                } else if (obj.get("function")) |fv| if (fv == .object) {
                    if (fv.object.get("name")) |nv| if (nv == .string) {
                        name = nv.string;
                    };
                };
                if (name.len > 0) {
                    const esc = try jsonEscape(allocator, name);
                    defer allocator.free(esc);
                    return try std.fmt.allocPrint(allocator, "{{\"type\":\"function\",\"name\":{s}}}", .{esc});
                }
                return try allocator.dupe(u8, "{\"type\":\"function\"}");
            }
        },
        else => {},
    };
    return try allocator.dupe(u8, "\"auto\"");
}

fn renderResponsesTextEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("text")) |tv| if (tv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, tv);
        return try buf.toOwnedSlice(allocator);
    };
    if (root.get("response_format")) |rf| if (rf == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try buf.appendSlice(allocator, "{\"format\":");
        try responses_mod.serializeJsonValue(allocator, &buf, rf);
        try buf.append(allocator, '}');
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{\"format\":{\"type\":\"text\"}}");
}

fn renderResponsesReasoningEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    var effort_buf: [32]u8 = undefined;
    var effort_str: []const u8 = "null";
    var summary_str: []const u8 = "null";
    if (root.get("reasoning")) |rv| if (rv == .object) {
        if (rv.object.get("effort")) |ev| if (ev == .string) {
            const s = ev.string;
            const valid = std.mem.eql(u8, s, "minimal") or
                std.mem.eql(u8, s, "low") or
                std.mem.eql(u8, s, "medium") or
                std.mem.eql(u8, s, "high");
            if (valid) {
                effort_str = std.fmt.bufPrint(&effort_buf, "\"{s}\"", .{s}) catch "null";
            }
        };
        if (rv.object.get("summary")) |sv| if (sv == .string) {
            const s = sv.string;
            const valid = std.mem.eql(u8, s, "auto") or
                std.mem.eql(u8, s, "concise") or
                std.mem.eql(u8, s, "detailed");
            if (valid) {
                // share buffer is fine since alloc happens immediately after
                if (std.mem.eql(u8, s, "auto")) summary_str = "\"auto\"" else if (std.mem.eql(u8, s, "concise")) summary_str = "\"concise\"" else summary_str = "\"detailed\"";
            }
        };
    };
    return try std.fmt.allocPrint(allocator, "{{\"effort\":{s},\"summary\":{s}}}", .{ effort_str, summary_str });
}

fn renderResponsesMetadataEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("metadata")) |mv| if (mv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, mv);
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{}");
}

/// Echoed-back fields that round out the OpenAI Responses envelope.
/// Owned by the caller (POST handler); freed after the final envelope is built.
const ResponseEcho = struct {
    // Pre-rendered JSON fragments (raw object/array text — caller owns).
    tools_json: []const u8 = "[]",
    tool_choice_json: []const u8 = "\"auto\"",
    text_json: []const u8 = "{\"format\":{\"type\":\"text\"}}",
    reasoning_json: []const u8 = "{\"effort\":null,\"summary\":null}",
    metadata_json: []const u8 = "{}",
    // Plain values; serialized inline.
    instructions: ?[]const u8 = null,
    truncation: []const u8 = "disabled",
    service_tier: []const u8 = "default",
    safety_identifier: ?[]const u8 = null,
    prompt_cache_key: ?[]const u8 = null,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    top_logprobs: u32 = 0,
    parallel_tool_calls: bool = true,
    background: bool = false,
    max_output_tokens: ?u32 = null,
    max_tool_calls: ?u32 = null,
};

/// Build the Responses envelope JSON body. Used for both the response.created
/// skeleton (in_progress, output:[]) and the final response.completed body.
/// Shape matches the OpenAI Responses API ResponseResource schema.
fn buildResponsesEnvelope(
    io: std.Io,
    allocator: std.mem.Allocator,
    esc_resp_id: []const u8,
    esc_model: []const u8,
    status_str: []const u8,
    output_json: []const u8,
    input_tokens: u32,
    output_tokens: u32,
    cached_input_tokens: u32,
    reasoning_output_tokens: u32,
    should_store: bool,
    prev_id: ?[]const u8,
    incomplete: bool,
    completed: bool,
    echo: ResponseEcho,
    /// Iteration 1: timings extension. The Responses spec doesn't model
    /// per-stage timings, so we attach a sibling `timings` block at the
    /// envelope root with the same shape as /v1/chat/completions. Pass
    /// zeroes to omit the field (legacy callers stay shape-compatible).
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
    completion_tokens_for_timings: u32,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var num_buf: [32]u8 = undefined;

    try buf.append(allocator, '{');
    try buf.appendSlice(allocator, "\"id\":");
    try buf.appendSlice(allocator, esc_resp_id);

    try buf.appendSlice(allocator, ",\"object\":\"response\",\"created_at\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));

    if (completed) {
        try buf.appendSlice(allocator, ",\"completed_at\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));
    } else {
        try buf.appendSlice(allocator, ",\"completed_at\":null");
    }

    try buf.appendSlice(allocator, ",\"status\":\"");
    try buf.appendSlice(allocator, status_str);
    try buf.append(allocator, '"');

    if (incomplete) {
        try buf.appendSlice(allocator, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    } else {
        try buf.appendSlice(allocator, ",\"incomplete_details\":null");
    }

    try buf.appendSlice(allocator, ",\"model\":");
    try buf.appendSlice(allocator, esc_model);

    if (prev_id) |pid| {
        const esc_pid = try jsonEscape(allocator, pid);
        defer allocator.free(esc_pid);
        try buf.appendSlice(allocator, ",\"previous_response_id\":");
        try buf.appendSlice(allocator, esc_pid);
    } else {
        try buf.appendSlice(allocator, ",\"previous_response_id\":null");
    }

    if (echo.instructions) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"instructions\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"instructions\":null");
    }

    try buf.appendSlice(allocator, ",\"output\":");
    try buf.appendSlice(allocator, output_json);

    try buf.appendSlice(allocator, ",\"error\":null");

    try buf.appendSlice(allocator, ",\"tools\":");
    try buf.appendSlice(allocator, echo.tools_json);

    try buf.appendSlice(allocator, ",\"tool_choice\":");
    try buf.appendSlice(allocator, echo.tool_choice_json);

    try buf.appendSlice(allocator, ",\"truncation\":\"");
    try buf.appendSlice(allocator, echo.truncation);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"parallel_tool_calls\":");
    try buf.appendSlice(allocator, if (echo.parallel_tool_calls) "true" else "false");

    try buf.appendSlice(allocator, ",\"text\":");
    try buf.appendSlice(allocator, echo.text_json);

    try buf.appendSlice(allocator, ",\"top_p\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_p}));
    try buf.appendSlice(allocator, ",\"presence_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.presence_penalty}));
    try buf.appendSlice(allocator, ",\"frequency_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.frequency_penalty}));
    try buf.appendSlice(allocator, ",\"top_logprobs\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_logprobs}));
    try buf.appendSlice(allocator, ",\"temperature\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.temperature}));

    try buf.appendSlice(allocator, ",\"reasoning\":");
    try buf.appendSlice(allocator, echo.reasoning_json);

    try buf.appendSlice(allocator, ",\"usage\":{\"input_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens}));
    try buf.appendSlice(allocator, ",\"output_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{output_tokens}));
    try buf.appendSlice(allocator, ",\"total_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens + output_tokens}));
    try buf.appendSlice(allocator, ",\"input_tokens_details\":{\"cached_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{cached_input_tokens}));
    try buf.appendSlice(allocator, "},\"output_tokens_details\":{\"reasoning_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{reasoning_output_tokens}));
    try buf.appendSlice(allocator, "}}");

    if (echo.max_output_tokens) |n| {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":null");
    }

    if (echo.max_tool_calls) |n| {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":null");
    }

    try buf.appendSlice(allocator, ",\"store\":");
    try buf.appendSlice(allocator, if (should_store) "true" else "false");

    try buf.appendSlice(allocator, ",\"background\":");
    try buf.appendSlice(allocator, if (echo.background) "true" else "false");

    try buf.appendSlice(allocator, ",\"service_tier\":\"");
    try buf.appendSlice(allocator, echo.service_tier);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"metadata\":");
    try buf.appendSlice(allocator, echo.metadata_json);

    if (echo.safety_identifier) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"safety_identifier\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"safety_identifier\":null");
    }

    if (echo.prompt_cache_key) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":null");
    }

    // Iteration 1 timings extension. Reuses the chat-completions
    // formatter so any future field added there appears on Responses too
    // without a second touch point.
    const timings_obj = try formatTimingsObject(allocator, input_tokens, cached_input_tokens, completion_tokens_for_timings, prefill_ns, decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    if (timings_obj.len > 0) {
        try buf.appendSlice(allocator, ",\"timings\":");
        try buf.appendSlice(allocator, timings_obj);
    }

    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

/// Emit output_item.added (type=reasoning) + reasoning_summary_part.added.
/// Pair with `emitResponsesReasoningEnd` after deltas are streamed.
fn emitResponsesReasoningStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.added","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":""}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.added", part_added);
    }
}

/// Emit a single reasoning_summary_text.delta event with the given chunk.
fn emitResponsesReasoningDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.reasoning_summary_text.delta","item_id":{s},"output_index":{d},"summary_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.delta", delta);
}

/// Emit reasoning_summary_text.done + reasoning_summary_part.done + output_item.done.
fn emitResponsesReasoningEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_text.done","item_id":{s},"output_index":{d},"summary_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.done","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":{s}}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[{{"type":"summary_text","text":{s}}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full reasoning output item in one shot. Used when the entire
/// reasoning text is known up-front (non-streaming generation paths).
fn emitResponsesReasoningEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    reasoning_text: []const u8,
) !void {
    try emitResponsesReasoningStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesReasoningDelta(allocator, stream, seq, output_index, item_id, reasoning_text);
    try emitResponsesReasoningEnd(allocator, stream, seq, output_index, item_id, reasoning_text);
}

/// Emit the SSE event sequence for a function_call output item: output_item.added,
/// function_call_arguments.delta (single full-args delta), .done, output_item.done.
fn emitResponsesFunctionCallEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    fc_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, fc_id);
    defer allocator.free(esc_id);
    const esc_call = try jsonEscape(allocator, call_id);
    defer allocator.free(esc_call);
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments_json);
    defer allocator.free(esc_args);

    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s},"arguments":"","status":"in_progress"}}}}
        , .{ output_index, esc_id, esc_call, esc_name });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const delta = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.delta","item_id":{s},"output_index":{d},"delta":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(delta);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.delta", delta);
    }
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.done","item_id":{s},"output_index":{d},"arguments":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.done", done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s},"arguments":{s},"status":"completed"}}}}
        , .{ output_index, esc_id, esc_call, esc_name, esc_args });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit output_item.added (type=message) + content_part.added.
/// Pair with `emitResponsesMessageEnd` after output_text.delta events stream.
fn emitResponsesMessageStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"in_progress","content":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.added","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":"","annotations":[]}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.added", part_added);
    }
}

/// Emit a single output_text.delta event for an in-progress message.
fn emitResponsesMessageDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.delta","item_id":{s},"output_index":{d},"content_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.output_text.delta", delta);
}

/// Emit output_text.done + content_part.done + output_item.done.
fn emitResponsesMessageEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_text.done","item_id":{s},"output_index":{d},"content_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.done","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":{s},"annotations":[]}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"completed","content":[{{"type":"output_text","text":{s},"annotations":[]}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full message output item in one shot. Used when the entire visible
/// text is known up-front (non-streaming generation paths).
fn emitResponsesMessageEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    text: []const u8,
) !void {
    try emitResponsesMessageStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesMessageDelta(allocator, stream, seq, output_index, item_id, text);
    try emitResponsesMessageEnd(allocator, stream, seq, output_index, item_id, text);
}

/// Persist a finished response to the in-memory store. The stored history is
/// the input messages plus a non-loop-cut assistant turn, deep-copied into
/// the entry's arena so it stays valid across requests.
fn storeResponse(
    io: std.Io,
    gpa: std.mem.Allocator,
    resp_id: []const u8,
    model_name: []const u8,
    status_str: []const u8,
    body_json: []const u8,
    input_messages: []const chat_mod.Message,
    visible_text: []const u8,
    reasoning_text: ?[]const u8,
    tool_calls: ?[]const chat_mod.ToolCall,
    finish_details: ?[]const u8,
) !void {
    const sr = try gpa.create(responses_mod.StoredResponse);
    errdefer gpa.destroy(sr);
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    // Keep the response retrievable, but never replay a loop-cut assistant turn.
    const keep_assistant = shouldParseToolCalls(finish_details);

    // Deep-copy input messages.
    const total_msgs = input_messages.len + @intFromBool(keep_assistant);
    const history = try a.alloc(chat_mod.Message, total_msgs);
    for (input_messages, 0..) |m, i| {
        history[i] = .{
            .role = try a.dupe(u8, m.role),
            .content = try a.dupe(u8, m.content),
            .reasoning_content = if (m.reasoning_content) |rc| try a.dupe(u8, rc) else null,
            .tool_call_id = if (m.tool_call_id) |tid| try a.dupe(u8, tid) else null,
            .tool_calls = if (m.tool_calls) |tcs| blk: {
                const copied = try a.alloc(chat_mod.ToolCall, tcs.len);
                for (tcs, 0..) |tc, j| copied[j] = .{
                    .id = try a.dupe(u8, tc.id),
                    .name = try a.dupe(u8, tc.name),
                    .arguments = try a.dupe(u8, tc.arguments),
                };
                break :blk copied;
            } else null,
            // images: not preserved across requests (would require deep-copying pixel buffers).
            .images = null,
        };
    }
    if (keep_assistant) {
        const assistant_content = try a.dupe(u8, visible_text);
        const assistant_reasoning: ?[]const u8 = if (reasoning_text) |rt| try a.dupe(u8, rt) else null;

        var assistant_tool_calls: ?[]chat_mod.ToolCall = null;
        if (tool_calls) |tcs| if (tcs.len > 0) {
            const arr = try a.alloc(chat_mod.ToolCall, tcs.len);
            for (tcs, 0..) |tc, i| {
                arr[i] = .{
                    .id = try a.dupe(u8, tc.id),
                    .name = try a.dupe(u8, tc.name),
                    .arguments = try a.dupe(u8, tc.arguments),
                };
            }
            assistant_tool_calls = arr;
        };

        history[total_msgs - 1] = .{
            .role = try a.dupe(u8, "assistant"),
            .content = assistant_content,
            .reasoning_content = assistant_reasoning,
            .tool_calls = assistant_tool_calls,
        };
    }

    sr.* = .{
        .id = try a.dupe(u8, resp_id),
        .created_at = nowSecs(io),
        .model = try a.dupe(u8, model_name),
        .status = try a.dupe(u8, status_str),
        .body_json = try a.dupe(u8, body_json),
        .history = history,
        .arena = arena,
    };
    errdefer sr.deinit();

    const store = getOrInitResponseStore(io, gpa);
    try store.put(sr);
}

test "responsesToolExists validates Responses function tool names" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"type":"function","name":"smartSearch","parameters":{}},{"type":"web_search"}]
    , .{});
    defer parsed.deinit();

    try testing.expect(responsesToolExists(parsed.value, "smartSearch"));
    try testing.expect(!responsesToolExists(parsed.value, "cruise_cards"));
}

test "buildResponsesJsonInstruction scopes schema prompt when tools are active" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\{"type":"object","properties":{"blocks":{"type":"array"}},"required":["blocks"]}
    , .{});
    defer parsed.deinit();

    const with_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, true);
    defer testing.allocator.free(with_tools);
    try testing.expect(std.mem.indexOf(u8, with_tools, "If you answer without calling a function") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "function call is needed") != null);
    try testing.expect(std.mem.indexOf(u8, with_tools, "\"blocks\"") != null);

    const without_tools = try buildResponsesJsonInstruction(testing.allocator, parsed.value, false);
    defer testing.allocator.free(without_tools);
    try testing.expect(std.mem.indexOf(u8, without_tools, "Respond with valid JSON only") != null);
    try testing.expect(std.mem.indexOf(u8, without_tools, "without calling a function") == null);
}

test "shouldInjectResponsesJsonInstruction skips required tool turns" {
    try testing.expect(!shouldInjectResponsesJsonInstruction(false, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, false, null));
    try testing.expect(shouldInjectResponsesJsonInstruction(true, true, null));
    try testing.expect(!shouldInjectResponsesJsonInstruction(true, true, "required"));
}

test "Responses history omits loop cuts while retaining the response and input" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();
    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    const calls = [_]chat_mod.ToolCall{.{ .id = "call_1", .name = "writeFile", .arguments = "{}" }};
    try storeResponse(testing.io, testing.allocator, "loop", "model", "completed", "{\"status\":\"completed\"}", &messages, "loop fragment", "repeated thought", &calls, "repetition_loop");
    const stored = global_response_store.?.map.get("loop").?;
    try testing.expectEqualStrings("completed", stored.status);
    try testing.expectEqualStrings("{\"status\":\"completed\"}", stored.body_json);
    try testing.expectEqual(@as(usize, 1), stored.history.len);
    try testing.expectEqualStrings("user", stored.history[0].role);
    try testing.expectEqualStrings("hi", stored.history[0].content);

    // A continuation keeps earlier input and its healthy assistant output.
    try storeResponse(testing.io, testing.allocator, "next", "model", "completed", "{}", stored.history, "answer", null, null, null);
    const next = global_response_store.?.map.get("next").?;
    try testing.expectEqual(@as(usize, 2), next.history.len);
    try testing.expectEqualStrings("hi", next.history[0].content);
    try testing.expectEqualStrings("answer", next.history[1].content);
}

test "Responses: a stored chain and an input replay render the same Qwen prompt" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();
    const a = testing.allocator;
    const calls = [_]chat_mod.ToolCall{.{ .id = "c1", .name = "lookup", .arguments = "{\"id\":1}" }};
    const first = [_]chat_mod.Message{.{ .role = "user", .content = "compare" }};
    try storeResponse(testing.io, a, "r1", "model", "completed", "{}", &first, "", "inspect it", &calls, null);
    const stored = global_response_store.?.map.get("r1").?;

    const out = try std.json.parseFromSlice(std.json.Value, a,
        \\[{"type":"function_call_output","call_id":"c1","output":"done"}]
    , .{});
    defer out.deinit();
    var chained = try responses_mod.parseInput(a, out.value, null, stored.history, null, .{});
    defer chained.deinit();

    const replay = try std.json.parseFromSlice(std.json.Value, a,
        \\[{"role":"user","content":"compare"},
        \\ {"type":"reasoning","summary":[{"type":"summary_text","text":"inspect it"}]},
        \\ {"type":"function_call","call_id":"c1","name":"lookup","arguments":"{\"id\":1}"},
        \\ {"type":"function_call_output","call_id":"c1","output":"done"}]
    , .{});
    defer replay.deinit();
    var replayed = try responses_mod.parseInput(a, replay.value, null, null, null, .{});
    defer replayed.deinit();

    const by_chain = try renderQwenForResponsesTest(a, chained.messages.items);
    defer a.free(by_chain);
    const by_replay = try renderQwenForResponsesTest(a, replayed.messages.items);
    defer a.free(by_replay);
    try testing.expectEqualStrings(by_replay, by_chain);
    try testing.expect(std.mem.indexOf(u8, by_chain, "inspect it") != null);
}

test "Responses history keeps reasoning as reasoning_content, as an input replay does" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();
    const messages = [_]chat_mod.Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "", .reasoning_content = "replayed thought" },
    };
    try storeResponse(testing.io, testing.allocator, "r1", "model", "completed", "{}", &messages, "answer", "fresh thought", null, null);
    const stored = global_response_store.?.map.get("r1").?;
    try testing.expectEqual(@as(usize, 3), stored.history.len);
    try testing.expectEqualStrings("replayed thought", stored.history[1].reasoning_content.?);
    try testing.expectEqualStrings("answer", stored.history[2].content);
    try testing.expectEqualStrings("fresh thought", stored.history[2].reasoning_content.?);
}

test "deinitGlobalResponseStore frees stored responses" {
    deinitGlobalResponseStore();
    defer deinitGlobalResponseStore();

    const messages = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    try storeResponse(testing.io, testing.allocator, "resp_test", "sushi", "completed", "{}", &messages, "hello", null, null, null);

    if (global_response_store) |*store| {
        try testing.expectEqual(@as(usize, 1), store.map.count());
    } else {
        return error.TestUnexpectedResult;
    }

    deinitGlobalResponseStore();
    try testing.expect(global_response_store == null);
}

test "isJsonObjectString only accepts JSON objects" {
    try testing.expect(isJsonObjectString(testing.allocator, "{\"destination\":\"CARIBBEAN\"}"));
    try testing.expect(!isJsonObjectString(testing.allocator, "[]"));
    try testing.expect(!isJsonObjectString(testing.allocator, "not-json"));
}

test "a responses disconnect is never stored" {
    try std.testing.expect(!responsesStoresFinish("client_disconnect", true));
    try std.testing.expect(!responsesStoresFinish("client_disconnect", false));
    try std.testing.expect(responsesStoresFinish("stop", true));
    try std.testing.expect(!responsesStoresFinish("length", false));
}
