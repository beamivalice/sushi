const std = @import("std");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const tool_calls_mod = @import("tool_calls.zig");
const think_tags = @import("think_tags.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const rp_mod = @import("reasoning_protocol.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_mod = @import("model.zig");
const effort_mod = @import("effort.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const tokenize_cache_mod = @import("tokenize_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const slot_admission = @import("slot_admission.zig");
const vision_queue = @import("vision_queue.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const responses_api = @import("responses_api.zig");
const splitConstrainedResponse = responses_api.splitConstrainedResponse;
const handleResponses = responses_api.handleResponses;
const generation_stream = @import("generation_stream.zig");
const GenErrorWire = generation_stream.GenErrorWire;
const sendGenerationError = generation_stream.sendGenerationError;
const visibleToolPreamble = generation_stream.visibleToolPreamble;
const nonStreamingViaScheduler = generation_stream.nonStreamingViaScheduler;
const StreamingTokenStream = generation_stream.StreamingTokenStream;
const requestSpecModes = generation_stream.requestSpecModes;
const noteMtpRefused = generation_stream.noteMtpRefused;
const pickStreamMode = generation_stream.pickStreamMode;
const contextOverflowMessage = generation_stream.contextOverflowMessage;
const formatTimingsObject = generation_stream.formatTimingsObject;
const formatPerfBracket = generation_stream.formatPerfBracket;
const sendStreamIdle = generation_stream.sendStreamIdle;
const beatStreamKeepalive = generation_stream.beatStreamKeepalive;
const http_conn = @import("http_conn.zig");
const WsBridge = http_conn.WsBridge;
const Conn = http_conn.Conn;
const sendSseHeaders = http_conn.sendSseHeaders;
const request_media = @import("request_media.zig");
const flattenMedia = request_media.flattenMedia;
const PreparedMedia = request_media.PreparedMedia;
const placeRequestMedia = request_media.placeRequestMedia;
const prepareRequestMedia = request_media.prepareRequestMedia;
const visionPreprocFromConfig = request_media.visionPreprocFromConfig;
const request_policy = @import("request_policy.zig");
const resolveSamplingDefault = request_policy.resolveSamplingDefault;
const resolveThinkPenalty = request_policy.resolveThinkPenalty;
const armThinkPenalty = request_policy.armThinkPenalty;
const launchMaxTokensDefault = request_policy.launchMaxTokensDefault;
const resolveRequestMaxTokens = request_policy.resolveRequestMaxTokens;
const maxTokensOrigin = request_policy.maxTokensOrigin;
const describeMaxTokens = request_policy.describeMaxTokens;
const clampMaxTokens = request_policy.clampMaxTokens;
const effortWordOnly = request_policy.effortWordOnly;
const reasoningEffortFromWord = request_policy.reasoningEffortFromWord;
const EffortNotice = request_policy.EffortNotice;
const effortModelName = request_policy.effortModelName;
const clearEffortNotice = request_policy.clearEffortNotice;
const holdEffortNotice = request_policy.holdEffortNotice;
const effortRefusalFor = request_policy.effortRefusalFor;
const checkThinkingSupport = request_policy.checkThinkingSupport;
const schemaMasksThinking = request_policy.schemaMasksThinking;
const io_util = @import("io_util.zig");
const prefill_plan = @import("prefill_plan.zig");
const sampler = @import("sampler.zig");
const nowMs = io_util.nowMs;
const Stopwatch = io_util.Stopwatch;
const nextCompletionId = io_util.nextCompletionId;
const byteTokenizerForTests = tokenizer_mod.byteTokenizerForTests;
const deinitTestTokenizer = tokenizer_mod.deinitTestTokenizer;
const server_mod = @import("server.zig");
const MAX_TOKENS_DESC_LEN = server_mod.MAX_TOKENS_DESC_LEN;
const MropeData = server_mod.MropeData;
const ReadContent = server_mod.ReadContent;
const TOOL_CHOICE_UNDECLARED = server_mod.TOOL_CHOICE_UNDECLARED;
const WireMessages = server_mod.WireMessages;
const WirePos = server_mod.WirePos;
const admitMtpForCtx = server_mod.admitMtpForCtx;
const appendRawToolDump = server_mod.appendRawToolDump;
const armCallForce = server_mod.armCallForce;
const armThinkBound = server_mod.armThinkBound;
const buildOpenAIToolsJson = server_mod.buildOpenAIToolsJson;
const cachedFormatChat = server_mod.cachedFormatChat;
const checkAttentionMemory = server_mod.checkAttentionMemory;
const clampJsonI32 = server_mod.clampJsonI32;
const countRejected = server_mod.countRejected;
const defaultEnableMtpForStreaming = server_mod.defaultEnableMtpForStreaming;
const expectSurfaceForcesCall = server_mod.expectSurfaceForcesCall;
const forceMtpFor = server_mod.forceMtpFor;
const getEffectiveContextLength = server_mod.getEffectiveContextLength;
const getTimeoutNs = server_mod.getTimeoutNs;
const handleChatCompletions = server_mod.handleChatCompletions;
const joinedTextParts = server_mod.joinedTextParts;
const jsonEscape = server_mod.jsonEscape;
const logHttpBody = server_mod.logHttpBody;
const mtpCapable = server_mod.mtpCapable;
const nativeMeasuredMoeHead = server_mod.nativeMeasuredMoeHead;
const parseJsonFloatOpt = server_mod.parseJsonFloatOpt;
const parseJsonTopKOpt = server_mod.parseJsonTopKOpt;
const parseKvQuantOverride = server_mod.parseKvQuantOverride;
const parseRequestSeed = server_mod.parseRequestSeed;
const parseToolCallsForRequest = server_mod.parseToolCallsForRequest;
const promptOpensThink = server_mod.promptOpensThink;
const readAllFromPair = server_mod.readAllFromPair;
const readContentParts = server_mod.readContentParts;
const reasoningWithinBudget = server_mod.reasoningWithinBudget;
const renderConfigFor = server_mod.renderConfigFor;
const requestBillsMtpHead = server_mod.requestBillsMtpHead;
const requestCacheKey = server_mod.requestCacheKey;
const resolveReasoningProtocol = server_mod.resolveReasoningProtocol;
const responseBody = server_mod.responseBody;
const rowMaskedDrafter = server_mod.rowMaskedDrafter;
const sendErrorResponse = server_mod.sendErrorResponse;
const sendResponse = server_mod.sendResponse;
const serializeJsonValue = server_mod.serializeJsonValue;
const shouldParseToolCalls = server_mod.shouldParseToolCalls;
const testWire = server_mod.testWire;
const test_png_blue = server_mod.test_png_blue;
const test_png_red = server_mod.test_png_red;
const toolCallFinishReason = server_mod.toolCallFinishReason;

/// The `/v1/messages` terminal `error` event payload.
pub fn buildAnthropicStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\{{"type":"error","error":{{"type":"{s}","message":{s}}}}}
    , .{ w.anthropic_type, esc });
}

/// The routes whose errors take Anthropic's shape.
pub fn isAnthropicRoute(path: []const u8) bool {
    return std.mem.eql(u8, path, "/v1/messages") or std.mem.eql(u8, path, "/v1/messages/count_tokens");
}

/// Anthropic `output_config` — the 2026 spelling Claude Code sends: `effort`
/// in place of the older `thinking` budget object, and `format` where this
/// surface previously had no structured-output request at all. Ignored, it
/// served every Claude Code request at the arch default with an UNLIMITED
/// thinking budget (live 2026-08-16: 8-minute retries at 16k thinking tokens
/// each) and answered `json_schema` requests with markdown prose the client
/// rejects — which is what fed the retry loop.
const AnthropicOutputConfig = struct {
    effort: ?[]const u8 = null,
    /// `format.schema` when `format.type == "json_schema"`; null otherwise.
    schema: ?std.json.Value = null,
    /// `format.type == "json_schema"` with no object schema to enforce.
    schema_invalid: bool = false,
};

fn parseAnthropicOutputConfig(root: std.json.ObjectMap) AnthropicOutputConfig {
    var out: AnthropicOutputConfig = .{};
    const oc = root.get("output_config") orelse return out;
    if (oc != .object) return out;
    if (oc.object.get("effort")) |e| {
        if (e == .string) out.effort = e.string;
    }
    if (oc.object.get("format")) |f| {
        if (f == .object) {
            const ftype = if (f.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, ftype, "json_schema")) {
                out.schema = f.object.get("schema");
                if (out.schema == null or out.schema.? != .object) {
                    out.schema = null;
                    out.schema_invalid = true;
                }
            }
        }
    }
    return out;
}

fn emitConstrainedAnthropic(a: std.mem.Allocator, stream: *Conn, index: *u32, thinking_open: *bool, text_open: *bool, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        if (!thinking_open.*) {
            const data = try std.fmt.allocPrint(a,
                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
            , .{index.*});
            defer a.free(data);
            try sendAnthropicEvent(stream, "content_block_start", data);
            thinking_open.* = true;
        }
        try emitAnthropicThinkingDelta(a, stream, index.*, d.reasoning.items);
    }
    if (d.content.items.len > 0) {
        if (thinking_open.*) {
            try closeAnthropicThinkingBlock(a, stream, index.*);
            thinking_open.* = false;
            index.* += 1;
        }
        if (!text_open.*) {
            const data = try std.fmt.allocPrint(a,
                \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
            , .{index.*});
            defer a.free(data);
            try sendAnthropicEvent(stream, "content_block_start", data);
            text_open.* = true;
        }
        try emitAnthropicTextDelta(a, stream, index.*, d.content.items);
    }
}

const SSE_ALLOW_HEADERS_ANTHROPIC = "Content-Type, Authorization, x-api-key, anthropic-version";

fn logHttpSseEvent(event_name: []const u8, data: []const u8) void {
    if (!log.isDebug()) return;
    log.debug("[http] <- sse event={s} data={d}b\n", .{ event_name, data.len });
    logHttpBody("[http] sse data", data);
}

/// An Anthropic image block's source as a URL the decoder reads: base64 as a
/// data URL, a `url` source as given. Caller frees.
pub fn anthropicImageUrl(allocator: std.mem.Allocator, block: std.json.Value) !?[]u8 {
    const src = block.object.get("source") orelse return null;
    if (src != .object) return null;
    const stype = if (src.object.get("type")) |t| (if (t == .string) t.string else "") else "";
    if (std.mem.eql(u8, stype, "base64")) {
        const media_type = if (src.object.get("media_type")) |v| (if (v == .string) v.string else "image/png") else "image/png";
        const data = if (src.object.get("data")) |v| (if (v == .string) v.string else "") else "";
        if (data.len == 0) return null;
        return try std.fmt.allocPrint(allocator, "data:{s};base64,{s}", .{ media_type, data });
    }
    if (std.mem.eql(u8, stype, "url")) {
        const url = if (src.object.get("url")) |v| (if (v == .string) v.string else "") else "";
        if (url.len == 0) return null;
        return try allocator.dupe(u8, url);
    }
    return null;
}

/// A /v1/messages conversation: the top-level `system` first, then every message.
fn readAnthropicConversation(wire: *WireMessages, system: ?std.json.Value, items: []const std.json.Value, vp: chat_mod.VisionPreproc) !void {
    const allocator = wire.allocator;
    // Array form JOINS every text block — Claude Code sends 2+ (identity + the
    // instructions block), and first-wins dropped everything after the first.
    if (system) |sys_val| {
        const sys_text: []const u8 = switch (sys_val) {
            .string => |s| s,
            .array => |arr| blk: {
                const joined = try joinedTextParts(allocator, arr.items);
                if (joined.owned) try wire.texts.append(allocator, joined.text);
                break :blk joined.text;
            },
            else => "",
        };
        if (sys_text.len > 0) {
            try wire.messages.append(allocator, .{ .role = "system", .content = sys_text, .tool_calls = null, .tool_call_id = null });
        }
    }
    try readAnthropicMessages(wire, items, vp);
}

/// Every message of a /v1/messages body. A user message's `tool_result`
/// blocks become tool messages ahead of its own text, and every image — in a
/// tool result or beside the text — is decoded and placed where it sat.
pub fn readAnthropicMessages(wire: *WireMessages, items: []const std.json.Value, vp: chat_mod.VisionPreproc) !void {
    const allocator = wire.allocator;
    for (items, 0..) |msg_val, raw_index| {
        if (msg_val != .object) continue;
        const msg_obj = msg_val.object;
        const role_val = msg_obj.get("role") orelse continue;
        if (role_val != .string) continue;
        const role = role_val.string;
        const content_val = msg_obj.get("content");
        const pos: WirePos = .{ .list = "messages", .index = raw_index };

        if (std.mem.eql(u8, role, "user")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try wire.messages.append(allocator, .{ .role = "user", .content = s });
                },
                .array => |arr| {
                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";
                        if (!std.mem.eql(u8, btype, "tool_result")) continue;

                        const tool_use_id = if (block.object.get("tool_use_id")) |v| (if (v == .string) v.string else "") else "";
                        var result: ReadContent = .{};
                        if (block.object.get("content")) |rc| switch (rc) {
                            .string => |s| result.text = s,
                            .array => |result_arr| result = try readContentParts(allocator, &wire.media, &wire.texts, result_arr.items, .anthropic, pos, vp),
                            else => {},
                        };
                        try wire.messages.append(allocator, .{
                            .role = "tool",
                            .content = result.text,
                            .tool_call_id = tool_use_id,
                            .images = result.images,
                            .media_parts = result.media_parts,
                        });
                    }

                    const read = try readContentParts(allocator, &wire.media, &wire.texts, arr.items, .anthropic, pos, vp);
                    if (read.text.len > 0 or read.images != null) {
                        try wire.messages.append(allocator, .{
                            .role = "user",
                            .content = read.text,
                            .images = read.images,
                            .media_parts = read.media_parts,
                        });
                    }
                },
                else => {},
            };
        } else if (std.mem.eql(u8, role, "assistant")) {
            if (content_val) |cv| switch (cv) {
                .string => |s| {
                    try wire.messages.append(allocator, .{ .role = "assistant", .content = s });
                },
                .array => |arr| {
                    // Extract text, tool_use and thinking blocks
                    var text_content = std.ArrayList(u8).empty;
                    defer text_content.deinit(allocator);
                    var think_content = std.ArrayList(u8).empty;
                    defer think_content.deinit(allocator);
                    var tcs = std.ArrayList(chat_mod.ToolCall).empty;

                    for (arr.items) |block| {
                        if (block != .object) continue;
                        const btype = if (block.object.get("type")) |t| (if (t == .string) t.string else "") else "";

                        if (std.mem.eql(u8, btype, "text")) {
                            const text = if (block.object.get("text")) |t| (if (t == .string) t.string else "") else "";
                            try text_content.appendSlice(allocator, text);
                        } else if (std.mem.eql(u8, btype, "tool_use")) {
                            const tc_id = if (block.object.get("id")) |v| (if (v == .string) v.string else "") else "";
                            const tc_name = if (block.object.get("name")) |v| (if (v == .string) v.string else "") else "";
                            // Serialize input object to JSON string
                            var args_buf = std.ArrayList(u8).empty;
                            if (block.object.get("input")) |input_val| {
                                try serializeJsonValue(allocator, &args_buf, input_val);
                            } else {
                                try args_buf.appendSlice(allocator, "{}");
                            }
                            const args_str = try args_buf.toOwnedSlice(allocator);
                            try wire.texts.append(allocator, args_str);
                            try tcs.append(allocator, .{ .id = tc_id, .name = tc_name, .arguments = args_str });
                        } else if (std.mem.eql(u8, btype, "thinking")) {
                            // History reasoning → Message.reasoning_content, same
                            // round-trip as chat completions' reasoning_content:
                            // templates that persist reasoning across turns
                            // starve into nothink without it.
                            // "redacted_thinking" stays skipped (opaque payload).
                            const t = if (block.object.get("thinking")) |v| (if (v == .string) v.string else "") else "";
                            if (t.len > 0) {
                                if (think_content.items.len > 0) try think_content.append(allocator, '\n');
                                try think_content.appendSlice(allocator, t);
                            }
                        }
                    }

                    var msg_tool_calls: ?[]const chat_mod.ToolCall = null;
                    if (tcs.items.len > 0) {
                        const owned = try tcs.toOwnedSlice(allocator);
                        try wire.tool_call_lists.append(allocator, owned);
                        msg_tool_calls = owned;
                    } else {
                        tcs.deinit(allocator);
                    }

                    const content: []const u8 = if (text_content.items.len > 0) blk: {
                        const duped = try allocator.dupe(u8, text_content.items);
                        try wire.texts.append(allocator, duped);
                        break :blk duped;
                    } else "";
                    const msg_reasoning: ?[]const u8 = if (think_content.items.len > 0) blk: {
                        const duped = try allocator.dupe(u8, think_content.items);
                        try wire.texts.append(allocator, duped);
                        break :blk duped;
                    } else null;
                    try wire.messages.append(allocator, .{ .role = "assistant", .content = content, .tool_calls = msg_tool_calls, .reasoning_content = msg_reasoning });
                },
                else => {},
            };
        } else if (std.mem.eql(u8, role, "system")) {
            // Claude Code puts SessionStart hook output here; dropping it is silent context loss.
            const sys_text: []const u8 = if (content_val) |cv| switch (cv) {
                .string => |s| s,
                .array => |arr| blk: {
                    const joined = try joinedTextParts(allocator, arr.items);
                    if (joined.owned) try wire.texts.append(allocator, joined.text);
                    break :blk joined.text;
                },
                else => "",
            } else "";
            if (sys_text.len > 0) {
                try wire.messages.append(allocator, .{ .role = "system", .content = sys_text });
            }
        }
    }
}

test "readAnthropicMessages: a tool_result image is the tool message's placed media" {
    var t = try testWire("{\"messages\":[" ++
        "{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"" ++ test_png_red ++ "\"}},{\"type\":\"text\",\"text\":\"open it\"}]}," ++
        "{\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"screenshot\",\"input\":{}}]}," ++
        "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":[{\"type\":\"text\",\"text\":\"here\"},{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"" ++ test_png_blue ++ "\"}}]},{\"type\":\"text\",\"text\":\"describe it\"}]}" ++
        "]}", .anthropic);
    defer t.parsed.deinit();
    defer t.wire.deinit();
    try std.testing.expect(t.wire.media.refusal() == null);
    const msgs = t.wire.messages.items;
    try std.testing.expectEqual(@as(usize, 4), msgs.len);
    try std.testing.expectEqual(@as(usize, 1), msgs[0].images.?.len);
    try std.testing.expectEqualStrings("tool", msgs[2].role);
    try std.testing.expectEqualStrings("t1", msgs[2].tool_call_id.?);
    try std.testing.expectEqualStrings("here", msgs[2].content);
    try std.testing.expectEqual(@as(usize, 1), msgs[2].images.?.len);
    try std.testing.expectEqualSlices(chat_mod.MediaPart, &.{.{ .at = 4, .kind = .image }}, msgs[2].media_parts.?);
    try std.testing.expectEqualStrings("user", msgs[3].role);
    try std.testing.expectEqualStrings("describe it", msgs[3].content);
    try std.testing.expect(msgs[3].images == null);
}

// ── Anthropic Messages API ──

pub fn sendAnthropicError(allocator: std.mem.Allocator, stream: *Conn, err_type: []const u8, message: []const u8, status_code: u32) !void {
    const escaped_msg = try jsonEscape(allocator, message);
    defer allocator.free(escaped_msg);
    const body = try std.fmt.allocPrint(allocator,
        \\{{"type":"error","error":{{"type":"{s}","message":{s}}}}}
    , .{ err_type, escaped_msg });
    defer allocator.free(body);
    var status_buf: [32]u8 = undefined;
    const status = std.fmt.bufPrint(&status_buf, "{d} Error", .{status_code}) catch "500 Error";
    try sendResponse(stream, status, "application/json", body);
}

/// The Anthropic surface's idle event: its ping, preceded by the queue comment while the request waits.
fn sendAnthropicIdle(stream: *Conn, queued: ?slot_admission.QueueWait) !void {
    if (queued) |w| try sendStreamIdle(stream, w);
    try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}");
}

pub fn sendAnthropicEvent(stream: *Conn, event_name: []const u8, data: []const u8) !void {
    if (stream.ws_mode) |bridge| {
        // WS transport: emit only the JSON payload as a text frame; the
        // event name lives inside the JSON as `"type": "..."`. (Anthropic
        // events are not currently WS-bridged — only Responses.)
        try bridge.sendText(data);
        return;
    }
    logHttpSseEvent(event_name, data);
    try stream.writeAllNoFlush("event: ");
    try stream.writeAllNoFlush(event_name);
    try stream.writeAllNoFlush("\ndata: ");
    try stream.writeAllNoFlush(data);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

fn mapFinishToStopReason(finish_reason: []const u8) []const u8 {
    if (std.mem.eql(u8, finish_reason, "stop")) return "end_turn";
    if (std.mem.eql(u8, finish_reason, "length")) return "max_tokens";
    if (std.mem.eql(u8, finish_reason, "tool_calls")) return "tool_use";
    return "end_turn";
}

/// Anthropic stop_reason with the client-stop-sequence case: a matched stop
/// sequence reports "stop_sequence" (callers learn WHICH stop fired via the
/// echoed `stop_sequence` field) — but only over a plain "stop"; it never
/// masks a max_tokens truncation or a parsed tool call.
fn anthropicStopReason(finish_reason: []const u8, matched_stop_seq: ?[]const u8) []const u8 {
    if (matched_stop_seq != null and std.mem.eql(u8, finish_reason, "stop")) return "stop_sequence";
    return mapFinishToStopReason(finish_reason);
}

pub fn handleAnthropicMessages(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    // No `lm.transformer.?` needed here.
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;
    const parsed = (try parseAnthropicBody(allocator, stream, body, "/v1/messages")) orelse return;
    defer parsed.deinit();
    const root = parsed.value.object;

    // max_tokens is required in the Anthropic API, but through the one parse helper so a negative
    // value is an omission and the `--max-tokens` launch default reaches this surface too.
    const req_max_tokens: u32 = resolveRequestMaxTokens(root.get("max_tokens"), 0);
    const max_tokens: u32 = if (req_max_tokens > 0) req_max_tokens else request_policy.filledMaxTokens(config);
    // For the request log line only. This surface 400s on a missing budget, so
    // the origin here is never `.auto`.
    const max_tokens_origin = maxTokensOrigin(root.get("max_tokens"), launchMaxTokensDefault(), config.defaultMaxTokens());
    if (max_tokens == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'max_tokens' is required and must be > 0", 400);
        return;
    }

    // Sampling parameters. Omitted fields resolve through CLI flags and the
    // model's generation_config.json — Claude Code omits ALL of them, and the
    // bare temp=1.0/top_p=1.0/no-top_k fallback sampled far outside Qwen's
    // intended envelope (model card wants top_k=20, top_p=0.95).
    const temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_mod.server_config.default_temperature, config.gen_temperature, 1.0);
    const top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_mod.server_config.default_top_p, config.gen_top_p, 1.0);
    const top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_mod.server_config.default_top_k, config.gen_top_k, 0);
    const seed: ?u64 = parseRequestSeed(root.get("seed"));

    // Stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop_sequences")) |stop_val| {
        if (stop_val == .array) {
            for (stop_val.array.items) |item| {
                if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
            }
        }
    }

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;
    const model_name = if (root.get("model")) |v| (if (v == .string) v.string else config.model_type) else config.model_type;

    // Wave 1.A: per-request KV-quant override (Anthropic mirror).
    const kv_quant_override = parseKvQuantOverride(root);

    // Drafter: same disable rules as chat-completions parse site.
    const lm_default_enable_drafter: bool = lm.dflash != null;
    var enable_drafter: bool = if (root.get("enable_drafter")) |v|
        (v == .bool and v.bool)
    else
        lm_default_enable_drafter;
    if (enable_drafter and lm.dflash == null) enable_drafter = false;
    const allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true;
    var enable_mtp: bool = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtpForStreaming(lm.mtp != null, config.isMoe(), forceMtpFor(config), nativeMeasuredMoeHead(lm), model_mod.isServedArch(config.model_type), config.expert_streaming and !config.stream_mtp_head);
    if (enable_mtp and lm.mtp == null) enable_mtp = false;

    var prompt: AnthropicPrompt = .{ .wire = WireMessages.init(allocator) };
    defer prompt.deinit();
    defer clearEffortNotice(stream);
    if (!try renderAnthropicPrompt(allocator, stream, lm, root, .messages, &prompt)) return;
    if (expert_stream_mod.mtpRefusal(config.expert_streaming, if (root.get("enable_mtp")) |v| (v == .bool and v.bool) else false, lm.mtp != null)) |why| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
        return;
    }
    const messages = &prompt.wire.messages;
    const tools_json = prompt.tools_json;
    const has_tools = prompt.has_tools;
    const enable_thinking = prompt.enable_thinking;
    const reasoning_budget = prompt.reasoning_budget;
    const output_cfg = prompt.output_cfg;
    const continue_final = prompt.continue_final;
    const tokenize_ns = prompt.tokenize_ns;

    // Log request
    const last_msg = messages.items[messages.items.len - 1];
    const preview_len = @min(last_msg.content.len, 80);
    var tool_msg_count: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "tool")) tool_msg_count += 1;
    }
    const tools_len = if (tools_json) |tj| tj.len else 0;

    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    var system_chars: usize = 0;
    for (messages.items) |msg| {
        if (std.mem.eql(u8, msg.role, "system")) system_chars += msg.content.len;
    }
    log.info("POST /v1/messages ({d} msgs, max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}, thinking={}, sys={d}b, tools={d}b, tool_msgs={d})\n", .{
        messages.items.len, describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), temperature, top_p, top_k, is_stream, enable_thinking, system_chars, tools_len, tool_msg_count,
    });
    log.info("  > \"{s}{s}\"\n", .{ last_msg.content[0..preview_len], if (last_msg.content.len > 80) "..." else "" });

    // Vision encoder: encode any images on the last user message and splice
    // image tokens into the prompt at the model's configured image_token_id.
    // Phase A8: per-request ownership.
    const cache_key = requestCacheKey(root);
    var req_media: PreparedMedia = .{};
    defer req_media.deinit(allocator);
    switch (try prepareRequestMedia(allocator, lm, messages.items, &prompt.ids, getEffectiveContextLength(config))) {
        .ready => |r| req_media = r,
        .refused => |fault| {
            log.warn("POST /v1/messages -> {d} ({s})\n", .{ fault.code(), fault.text() });
            try sendAnthropicError(allocator, stream, if (fault.server_error) "api_error" else "invalid_request_error", fault.text(), fault.code());
            return;
        },
    }
    const prompt_ids = prompt.ids;
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Context size enforcement
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/messages -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendAnthropicError(allocator, stream, "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    const bill_head = requestBillsMtpHead(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), .{}, 0);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, lm.id, true, kv_quant_override, prefill_plan.visionPrefillUnchunked(req_media.deferred != null), bill_head)) return;

    log.info("  prompt={d} tokens, max_gen={d}, ctx={d}\n", .{ prompt_ids.len, effective_max_tokens, effective_ctx });

    const eos_slice = config.eosTokenSlice();
    var sampling = sampler.SamplingParams{
        .temperature = temperature,
        .top_p = top_p,
        .top_k = top_k,
        .repeat_penalty = 1.0,
        .presence_penalty = 0.0,
        .seed = seed,
    };

    // Grammar-constrained sampling for `output_config.format` json_schema —
    // the chat-completions block, same lifetime rule: the SchemaConstraint
    // must NOT be moved (the embedded Constraint holds pointers into it).
    var sc: generate_mod.SchemaConstraint = undefined;
    var sc_init = false;
    defer if (sc_init) sc.deinit();
    if (output_cfg.schema) |sv| {
        if (has_tools) {
            log.info("[grammar] skipped JSON schema mask while tools are available (tool calls must remain reachable)\n", .{});
        } else {
            const tb = try lm.grammarTokenBytes(allocator, stream.io);
            if (sc.initFromValue(allocator, sv, tb)) {
                sc_init = true;
                var deferred = false;
                if (prompt.schema_proto_active) {
                    sc.deferWithProtocol(&prompt.schema_proto);
                    deferred = prompt.schema_proto.startState().phase != .json_body;
                }
                sc.constraint.row_masks = rowMaskedDrafter(lm);
                sampling.constraint = &sc.constraint;
                if (deferred) {
                    log.info("[grammar] deferring JSON schema until the reasoning boundary (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                } else {
                    log.info("[grammar] enforcing JSON schema from token 0 (vocab={d}, mask={d}b)\n", .{ tb.bytes.len, sc.mask_buf.len });
                }
            } else |err| {
                log.warn("[grammar] schema parse failed ({s}); falling back to prompt-only enforcement\n", .{@errorName(err)});
            }
        }
    }

    sampling.think_penalty = armThinkPenalty(allocator, lm, tok, prompt_ids, enable_thinking, resolveThinkPenalty(root, model_settings.think_penalty_flag, config.think_penalty_override));
    var think_bound = armThinkBound(allocator, lm, tok, prompt_ids, enable_thinking, reasoning_budget);
    defer if (think_bound) |tb| allocator.free(tb.forced);
    const surface_budget: i32 = if (think_bound != null) -1 else reasoning_budget;
    if (think_bound) |*tb| sampling.think_bound = tb;
    var call_force = if (has_tools and lm.transformer != null and !continue_final)
        armCallForce(allocator, tok, chat_config, prompt_ids, enable_thinking, prompt.tool_choice, tools_json)
    else
        null;
    defer if (call_force) |*cf| cf.deinit(allocator);
    if (call_force) |*cf| sampling.call_force = cf;

    // Hand vision ownership to the sub-handler (slot takes it on submit).
    const sub_ve = req_media.deferred;
    req_media.deferred = null;
    const sub_mrope = req_media.mrope;
    req_media.mrope = .{}; // ownership transferred to the sub-handler → slot
    if (is_stream) {
        handleAnthropicStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, prompt.allow_parallel_tools, enable_thinking, surface_budget, @intCast(prompt_ids.len), enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            sendGenerationError(allocator, stream, err, .anthropic) catch {};
        };
    } else {
        handleAnthropicNonStreaming(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, stop_sequences.items, model_name, has_tools, tools_json, prompt.allow_parallel_tools, enable_thinking, surface_budget, @intCast(prompt_ids.len), enable_drafter, enable_mtp, allow_batch_mtp, sub_ve, req_media.chain, cache_key, sub_mrope, kv_quant_override, tokenize_ns) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .anthropic) catch {};
        };
    }
}

/// A /v1/messages body as a JSON object; null once the 400 is sent.
fn parseAnthropicBody(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, route: []const u8) !?std.json.Parsed(std.json.Value) {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST {s} -> 400 (invalid JSON)\n", .{route});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Invalid JSON in request body", 400);
        return null;
    };
    if (parsed.value != .object) {
        parsed.deinit();
        log.warn("POST {s} -> 400 (body is not a JSON object)\n", .{route});
        try sendAnthropicError(allocator, stream, "invalid_request_error", "Request body must be a JSON object", 400);
        return null;
    }
    return parsed;
}

/// What a /v1/messages body renders to, before media placement; /v1/messages and
/// count_tokens both build it through `renderAnthropicPrompt`.
const AnthropicPrompt = struct {
    wire: WireMessages,
    /// OpenAI-shape tools; null when the request declares none.
    tools_json: ?[]const u8 = null,
    has_tools: bool = false,
    allow_parallel_tools: bool = true,
    tool_choice: tool_calls_mod.ToolChoice = .auto,
    tool_choice_instruction: ?[]const u8 = null,
    enable_thinking: bool = false,
    reasoning_budget: i32 = -1,
    output_cfg: AnthropicOutputConfig = .{},
    continue_final: bool = false,
    ids: []u32 = &.{},
    schema_proto: rp_mod.Protocol = undefined,
    schema_proto_active: bool = false,
    tokenize_ns: u64 = 0,
    /// Armed on the connection when an effort word is served as another; the handler clears it.
    effort_notice: ?EffortNotice = null,

    fn deinit(p: *AnthropicPrompt) void {
        const a = p.wire.allocator;
        if (p.effort_notice) |n| n.deinit(a);
        if (p.tools_json) |tj| a.free(tj);
        if (p.tool_choice_instruction) |tci| a.free(tci);
        a.free(p.ids);
        p.wire.deinit();
    }
};

/// The route a /v1/messages body renders for. Only `.messages` uses the tokenize LRU: a count
/// changes no cache.
const AnthropicRoute = enum {
    messages,
    count_tokens,

    fn path(r: AnthropicRoute) []const u8 {
        return switch (r) {
            .messages => "/v1/messages",
            .count_tokens => "/v1/messages/count_tokens",
        };
    }
};

/// Renders `root` into `p` exactly as /v1/messages prefills it; false once a 400 is sent.
fn renderAnthropicPrompt(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, root: std.json.ObjectMap, route: AnthropicRoute, p: *AnthropicPrompt) !bool {
    const tok = lm.tokenizer.?;
    const chat_config = lm.chat_config.?;
    const config = lm.config.?;

    const messages_val = root.get("messages") orelse {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' is required", 400);
        return false;
    };
    if (messages_val != .array) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "'messages' must be an array", 400);
        return false;
    }
    const messages = &p.wire.messages;
    try readAnthropicConversation(&p.wire, root.get("system"), messages_val.array.items, visionPreprocFromConfig(config));
    if (p.wire.media.refusal()) |fault| {
        log.warn("POST {s} -> 400 ({s})\n", .{ route.path(), fault.text() });
        try sendAnthropicError(allocator, stream, if (fault.server_error) "api_error" else "invalid_request_error", fault.text(), fault.code());
        return false;
    }
    if (messages.items.len == 0) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "No valid messages found in request", 400);
        return false;
    }

    if (root.get("tools")) |tools_val| {
        if (tools_val == .array and tools_val.array.items.len > 0) {
            p.has_tools = true;
            // Convert Anthropic tools to OpenAI format for chat template
            p.tools_json = try buildOpenAIToolsJson(allocator, tools_val.array);

            if (root.get("tool_choice")) |tc| {
                if (tc == .object) {
                    // Anthropic spelling of the parallel clamp.
                    if (tc.object.get("disable_parallel_tool_use")) |d| {
                        if (d == .bool and d.bool) p.allow_parallel_tools = false;
                    }
                }
            }
            p.tool_choice = tool_calls_mod.parseToolChoice(root.get("tool_choice"));
            if (p.tool_choice == .none) p.has_tools = false;
            p.tool_choice_instruction = try p.tool_choice.instruction(allocator);
            if (tool_calls_mod.toolChoiceNamesUndeclared(allocator, p.tool_choice, p.tools_json)) {
                log.warn("POST {s} -> 400 (tool_choice names an undeclared tool)\n", .{route.path()});
                try sendAnthropicError(allocator, stream, "invalid_request_error", TOOL_CHOICE_UNDECLARED, 400);
                return false;
            }
        }
    }

    // Thinking config. An absent `thinking` object consults the arch default
    // (an arch that always reasons delivers it as thinking blocks rather
    // than paying for it and discarding it); a PRESENT object decides
    // explicitly, exactly as before.
    var enable_thinking = if (root.get("thinking") == null)
        config.defaultEnableThinking()
    else
        false;
    var reasoning_budget: i32 = model_mod.defaultThinkBudget(config, server_mod.server_config.default_reasoning_budget);
    var budget_explicit = false;
    if (root.get("thinking")) |think_val| {
        if (think_val == .object) {
            const think_type = if (think_val.object.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, think_type, "enabled") or std.mem.eql(u8, think_type, "adaptive")) {
                enable_thinking = true;
            }
            if (think_val.object.get("budget_tokens")) |bt| {
                if (bt == .integer) {
                    reasoning_budget = clampJsonI32(bt.integer);
                    budget_explicit = true;
                }
            }
        }
    }

    // `output_config` (Claude Code's spelling; see parseAnthropicOutputConfig).
    // The effort word is an explicit thinking signal, so it displaces the arch
    // default and OR's with a present `thinking` object — the OpenAI surface's
    // resolveEnableThinking rule. An explicit `budget_tokens` outranks the
    // budget derived from the word; the word itself rides through to templates
    // that read it (qwen3.8's preamble, dsv4's).
    const output_cfg = parseAnthropicOutputConfig(root);
    p.output_cfg = output_cfg;
    if (output_cfg.schema_invalid) {
        try sendAnthropicError(allocator, stream, "invalid_request_error", "output_config.format.schema must be a JSON object when format.type is json_schema", 400);
        return false;
    }
    var effort_word: ?[]const u8 = model_mod.defaultEffortWord(config);
    if (output_cfg.effort) |word| {
        const cfg = reasoningEffortFromWord(word, server_mod.server_config.default_reasoning_budget, effortWordOnly(allocator, lm, tok), effort_mod.effortArms(config.model_type)) catch {
            const msg = try effortRefusalFor(allocator, lm, word);
            defer allocator.free(msg);
            try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400);
            return false;
        };
        effort_word = cfg.effort;
        p.effort_notice = try holdEffortNotice(allocator, stream, effortModelName(lm), cfg);
        if (!budget_explicit) reasoning_budget = cfg.budget;
        enable_thinking = if (root.get("thinking") == null) cfg.enable else (enable_thinking or cfg.enable);
    }
    if (!try checkThinkingSupport(allocator, stream, config, enable_thinking, true)) return false;

    // `output_config.format` json_schema — the same two-layer enforcement as
    // chat-completions' `response_format`: a schema instruction in the system
    // prompt (so the model aims at the shape) plus the grammar mask below (so
    // the shape is guaranteed). Before the render, or the instruction never
    // reaches the prompt.
    if (output_cfg.schema != null) {
        var schema_instruction = std.ArrayList(u8).empty;
        defer schema_instruction.deinit(allocator);
        try schema_instruction.appendSlice(allocator, "Respond with valid JSON only. No other text, no markdown, no explanation. ");
        {
            var out: std.Io.Writer.Allocating = .init(allocator);
            defer out.deinit();
            var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
            output_cfg.schema.?.jsonStringify(&jws) catch {};
            try schema_instruction.appendSlice(allocator, "Your response MUST conform to this JSON schema:\n");
            try schema_instruction.appendSlice(allocator, out.written());
        }
        if (messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system")) {
            const combined = try std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ messages.items[0].content, schema_instruction.items });
            try p.wire.texts.append(allocator, combined);
            messages.items[0].content = combined;
        } else {
            const instruction = try allocator.dupe(u8, schema_instruction.items);
            try p.wire.texts.append(allocator, instruction);
            try messages.insert(allocator, 0, .{ .role = "system", .content = instruction, .tool_calls = null, .tool_call_id = null });
        }
    }

    // Format chat template. Iteration 1 timing + Iteration 2 cache. The
    // `effective_tools_json` swap (`null` when has_tools is false) keeps
    // the cache key consistent with what the encoder actually sees.
    const effective_tools_json: ?[]const u8 = if (p.has_tools) p.tools_json else null;
    // Anthropic's own contract: a conversation ending in an assistant message
    // means "continue this", with no flag. Implicit here and explicit on the
    // OpenAI surface is not an inconsistency — it is each surface's documented
    // behaviour, so an Anthropic SDK client gets prefill for free.
    //
    // And because it is INFERRED, a model that cannot serve one gets an
    // ordinary turn rather than a 400: the client never asked for a
    // continuation, so refusing its request would take away an answer this
    // endpoint has always given.
    p.continue_final = think_tags.continuationRequested(messages.items);
    var tokenize_sw = Stopwatch.init(stream.io);
    // The `thinking` budget object carries no effort string, but
    // `output_config.effort` does — templates that read the word (dsv4,
    // qwen3.8) get it; requests without one still render their default.
    const render_config = renderConfigFor(chat_config, root, model_settings.preserve_thinking_flag, config.preserve_thinking_override);
    p.ids = (if (route == .messages)
        cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, effective_tools_json, p.tool_choice_instruction, enable_thinking, effort_word, p.continue_final)
    else
        chat_mod.formatChat(allocator, tok, messages.items, &render_config, effective_tools_json, p.tool_choice_instruction, enable_thinking, effort_word, p.continue_final)) catch |err| {
        try sendGenerationError(allocator, stream, err, .anthropic);
        return false;
    };
    if (output_cfg.schema != null and !p.has_tools and enable_thinking and reasoning_budget < 0) {
        p.schema_proto_active = resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, p.ids, &p.schema_proto, true);
    }
    switch (schemaMasksThinking(
        output_cfg.schema != null,
        p.has_tools,
        enable_thinking,
        reasoning_budget >= 0,
        p.schema_proto_active,
    )) {
        .deferred => {},
        .fallback_thinking_off => {
            p.schema_proto_active = false;
            allocator.free(p.ids);
            p.ids = &.{};
            enable_thinking = false;
            p.ids = (if (route == .messages)
                cachedFormatChat(allocator, stream.io, lm, tok, &render_config, messages.items, effective_tools_json, p.tool_choice_instruction, false, effort_word, p.continue_final)
            else
                chat_mod.formatChat(allocator, tok, messages.items, &render_config, effective_tools_json, p.tool_choice_instruction, false, effort_word, p.continue_final)) catch |err| {
                try sendGenerationError(allocator, stream, err, .anthropic);
                return false;
            };
            log.info("[grammar] {s}; rerendered with thinking off\n", .{if (reasoning_budget >= 0) "finite reasoning budget" else "reasoning protocol unsupported for this model/prompt"});
        },
        .no_mask, .token_zero => p.schema_proto_active = false,
    }
    if (output_cfg.schema != null and !p.has_tools and !enable_thinking and lm.transformer != null) {
        if (!resolveReasoningProtocol(allocator, stream.io, lm, tok, chat_config, p.ids, &p.schema_proto, false)) {
            p.schema_proto = rp_mod.Protocol.jsonOnly();
        }
        p.schema_proto_active = true;
    }
    p.tokenize_ns = tokenize_sw.read();
    p.enable_thinking = enable_thinking;
    p.reasoning_budget = reasoning_budget;
    return true;
}

/// POST /v1/messages/count_tokens: the prompt tokens /v1/messages would prefill for the
/// same body, media rows included, with no generation, queue or cache write.
pub fn handleAnthropicCountTokens(allocator: std.mem.Allocator, stream: *Conn, body: []const u8, lm: *LoadedModel) !void {
    const parsed = (try parseAnthropicBody(allocator, stream, body, "/v1/messages/count_tokens")) orelse return;
    defer parsed.deinit();
    var prompt: AnthropicPrompt = .{ .wire = WireMessages.init(allocator) };
    defer prompt.deinit();
    defer clearEffortNotice(stream);
    if (!try renderAnthropicPrompt(allocator, stream, lm, parsed.value.object, .count_tokens, &prompt)) return;

    var input_tokens = prompt.ids.len;
    var flat = try flattenMedia(allocator, prompt.wire.messages.items);
    defer flat.deinit(allocator);
    if (flat.images.len > 0 or flat.videos.len > 0 or flat.has_audio) {
        var placed = switch (try placeRequestMedia(allocator, lm, prompt.wire.messages.items, flat, prompt.ids)) {
            .placed => |pl| pl,
            .refused => |fault| {
                log.warn("POST /v1/messages/count_tokens -> {d} ({s})\n", .{ fault.code(), fault.text() });
                try sendAnthropicError(allocator, stream, if (fault.server_error) "api_error" else "invalid_request_error", fault.text(), fault.code());
                return;
            },
        };
        defer placed.deinit(allocator);
        input_tokens = placed.ids.len;
    }
    log.info("POST /v1/messages/count_tokens -> {d} tokens\n", .{input_tokens});
    const out = try std.fmt.allocPrint(allocator, "{{\"input_tokens\":{d}}}", .{input_tokens});
    defer allocator.free(out);
    try sendResponse(stream, "200 OK", "application/json", out);
}

fn handleAnthropicNonStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    deferred_media: ?*vision_queue.DeferredMedia,
    media_chain: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1 instrumentation: nanoseconds of render+tokenize measured
    /// by the parent handleAnthropicMessages. Threaded through so the
    /// non-streaming response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Vision-array ownership: nulled below before scheduler.submit so the
    // early-return defer doesn't double-free.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    var timer = Stopwatch.init(stream.io);

    // Speculative decoding dispatch — same `requestSpecModes` as
    // chat-completions (DFlash > MTP).
    const spec = requestSpecModes(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, 0);
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;

    // Anthropic responses never carry logprobs (the API doesn't expose
    // them). Vision-bearing requests transfer ownership of `ve_local` into
    // the slot.
    const slot_ve: ?*vision_queue.DeferredMedia = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // M-RoPE: Anthropic path uses scalar-RoPE fallback for now (faithful M-RoPE
    // wired for /v1/chat/completions; see computeQwenMrope). Qwen image requests
    // still decode correctly — M-RoPE refines spatial grounding only.
    // Propagates to `handleAnthropicMessages`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, server_mod.global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve, media_chain, cache_key, mrope, 0, kv_quant_override, stream, .{ .stops = stop_sequences });
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // Apply stop sequences; remember WHICH one matched (Anthropic reports it
    // as stop_reason "stop_sequence" + the echoed `stop_sequence` field).
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    var matched_stop_seq: ?[]const u8 = null;
    if (stop_seq_mod.earliest(final_text, stop_sequences)) |hit| {
        final_text = final_text[0..hit.index];
        finish_reason = "stop";
        matched_stop_seq = hit.stop;
    }

    // Merge re-opened mid-text thought channels into the leading block so the
    // split/parse below never leaks raw tags (Gemma 12B re-opens channels
    // mid-turn). A span-routed constrained response skips this: its payload
    // is authoritative and marker text inside it is DATA.
    var normalized_text: ?[]u8 = null;
    defer if (normalized_text) |n| allocator.free(n);
    if (sampling.constraint == null or sampling.constraint.?.proto == null) {
        normalized_text = try think_tags.normalizeEmbeddedThinkBlocks(allocator, final_text);
        if (normalized_text) |n| final_text = n;
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Build content blocks array
    var content = std.ArrayList(u8).empty;
    defer content.deinit(allocator);
    try content.append(allocator, '[');

    var block_count: u32 = 0;

    // Thinking block.
    //
    // The split/strip here KEEP unparsed tool markup: unlike every other
    // surface, this path reassigns `final_text` from the result and hands it
    // to parseToolCallsForRequest below. Cutting the markup first would make a
    // real call unparseable. The visible text block is cut at emission instead
    // (`think_tags.trimLeakedToolMarkup`), which is where the leak matters.
    // Reasoning the model generated is always delivered as a thinking block —
    // the request's thinking flag shaped the prompt, never the delivery.
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    {
        const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, true, promptOpensThink(allocator, tok, prompt_ids));
        // Reasoning is never fed back to the parser, so it is cut here.
        const split_reasoning: ?[]const u8 = if (think_split.reasoning_content) |r| blk: {
            const t = think_tags.trimLeakedToolMarkup(r);
            break :blk if (t.len > 0) t else null;
        } else null;
        if (split_reasoning) |reasoning| {
            // Apply budget truncation
            var truncated_reasoning = reasoning;
            var trunc_allocated = false;
            defer if (trunc_allocated) allocator.free(truncated_reasoning);
            if (reasoning_budget >= 0) {
                const r_ids = try tok.encode(allocator, reasoning);
                defer allocator.free(r_ids);
                const budget_usize: usize = @intCast(reasoning_budget);
                if (r_ids.len > budget_usize) {
                    truncated_reasoning = try tok.decode(allocator, r_ids[0..budget_usize]);
                    trunc_allocated = true;
                }
            }
            const esc_r = try jsonEscape(allocator, truncated_reasoning);
            defer allocator.free(esc_r);
            const thinking_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"thinking","thinking":{s},"signature":"sushi-local"}}
            , .{esc_r});
            defer allocator.free(thinking_block);
            try content.appendSlice(allocator, thinking_block);
            block_count += 1;
        }
        final_text = think_split.content;
    }

    // Check for tool calls
    if (has_tools and shouldParseToolCalls(result.finish_details)) {
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json, allow_parallel_tools);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tu_perf_buf: [160]u8 = undefined;
            const tu_perf = formatPerfBracket(&tu_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_use: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tu_perf, tool_calls.len,
            });

            // The same pre-markup text the stream flushes as a text delta.
            const tu_preamble = visibleToolPreamble(think_tags.trimLeakedToolMarkup(final_text));
            if (tu_preamble.len > 0) {
                if (block_count > 0) try content.append(allocator, ',');
                const esc_pre = try jsonEscape(allocator, tu_preamble);
                defer allocator.free(esc_pre);
                const pre_block = try std.fmt.allocPrint(allocator,
                    \\{{"type":"text","text":{s}}}
                , .{esc_pre});
                defer allocator.free(pre_block);
                try content.appendSlice(allocator, pre_block);
                block_count += 1;
            }

            // Emit tool_use content blocks
            for (tool_calls, 0..) |tc, i| {
                if (block_count > 0) try content.append(allocator, ',');
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);
                const tc_block = try std.fmt.allocPrint(allocator,
                    \\{{"type":"tool_use","id":"{s}","name":{s},"input":{s}}}
                , .{ tc_id, esc_name, tc.arguments });
                defer allocator.free(tc_block);
                try content.appendSlice(allocator, tc_block);
                block_count += 1;
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls — emit text block. Nothing parsed, so any tool
            // markup still in here is unparsed wreckage: cut it at emission.
            if (block_count > 0) try content.append(allocator, ',');
            const esc_text = try jsonEscape(allocator, if (routed != null) final_text else think_tags.trimLeakedToolMarkup(final_text));
            defer allocator.free(esc_text);
            const text_block = try std.fmt.allocPrint(allocator,
                \\{{"type":"text","text":{s}}}
            , .{esc_text});
            defer allocator.free(text_block);
            try content.appendSlice(allocator, text_block);
        }
    } else {
        // No tools — emit text block
        if (block_count > 0) try content.append(allocator, ',');
        const esc_text = try jsonEscape(allocator, if (routed != null) final_text else think_tags.trimLeakedToolMarkup(final_text));
        defer allocator.free(esc_text);
        const text_block = try std.fmt.allocPrint(allocator,
            \\{{"type":"text","text":{s}}}
        , .{esc_text});
        defer allocator.free(text_block);
        try content.appendSlice(allocator, text_block);
    }

    try content.append(allocator, ']');

    const stop_reason = anthropicStopReason(finish_reason, matched_stop_seq);
    // Echo the matched sequence exactly when the reason is "stop_sequence".
    const echo_stop_seq = std.mem.eql(u8, stop_reason, "stop_sequence");
    const stop_seq_json: []const u8 = if (echo_stop_seq) try jsonEscape(allocator, matched_stop_seq.?) else "null";
    defer if (echo_stop_seq) allocator.free(stop_seq_json);
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, stop_reason,
    });

    // Anthropic spec doesn't standardize a `timings` field; we attach one as
    // an extension (mirrors what /v1/chat/completions already does) so bench
    // tooling can read tokenize_ms / prompt_ms / predicted_ms from either
    // surface without re-implementing the SSE accumulator.
    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);
    const usage = try formatAnthropicUsage(allocator, prompt_token_count, result.completion_tokens, result.cached_tokens);
    defer allocator.free(usage);

    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"msg_{d}","type":"message","role":"assistant","content":{s},"model":"{s}","stop_reason":"{s}","stop_sequence":{s},"usage":{s}{s}}}
    , .{
        nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io)),
        content.items,
        model_name,
        stop_reason,
        stop_seq_json,
        usage,
        timings_field,
    });
    defer allocator.free(response);
    try sendResponse(stream, "200 OK", "application/json", response);
}

/// Anthropic counts cache reads apart from `input_tokens`: a client adds the two to size its context.
fn formatAnthropicUsage(allocator: std.mem.Allocator, prompt_tokens: u32, output_tokens: u32, cached_tokens: u32) ![]u8 {
    return std.fmt.allocPrint(allocator, "{{\"input_tokens\":{d},\"output_tokens\":{d},\"cache_read_input_tokens\":{d}}}", .{ prompt_tokens -| cached_tokens, output_tokens, cached_tokens });
}

fn handleAnthropicStreaming(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    enable_thinking: bool,
    reasoning_budget: i32,
    prompt_token_count: u32,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    deferred_media: ?*vision_queue.DeferredMedia,
    media_chain: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns from parent handler. Anthropic streaming
    /// doesn't currently emit `timings` over SSE (spec doesn't model it),
    /// but plumbing the value through keeps the signature consistent with
    /// non-streaming and unblocks a future message_delta extension.
    tokenize_ns: u64,
) !void {
    _ = tokenize_ns; // reserved; see doc comment
    // Vision-array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    // Pick speculative-decoding mode (regular / drafter / MTP). The token-
    // stream adapter below feeds the per-token Anthropic state machine the
    // same way for all three modes.
    const stream_mode = pickStreamMode(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, 0);
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});
    noteMtpRefused(enable_mtp, mtpCapable(lm), sampling, 0);

    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| server_mod.global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_anth = ve_local;
    ve_local = null;
    const sch = server_mod.global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = getTimeoutNs(),
        .enable_drafter = stream_mode == .drafter,
        .dflash = if (stream_mode == .drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .allow_batch_mtp = allow_batch_mtp,
        .enable_mtp = stream_mode == .mtp,
        .mtp = if (stream_mode == .mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .logprobs_n = 0,
        .deferred_media = slot_ve_anth,
        .vision_key = prefix_cache_mod.chainKey(media_chain),
        .media_chain = media_chain,
        .cache_key = cache_key,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);

    // SSE headers
    try sendSseHeaders(stream, "anthropic.messages", SSE_ALLOW_HEADERS_ANTHROPIC);

    // message_start
    {
        const data = try std.fmt.allocPrint(allocator,
            \\{{"type":"message_start","message":{{"id":"msg_{d}","type":"message","role":"assistant","content":[],"model":"{s}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{d},"output_tokens":1}}{s}}}}}
        , .{ nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io)), model_name, prompt_token_count, stream.effort_notice_json orelse "" });
        defer allocator.free(data);
        try sendAnthropicEvent(stream, "message_start", data);
    }
    try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}");

    // State
    var block_index: u32 = 0;
    var text_block_open = false;
    var thinking_block_open = false;
    // Template-opened think (Qwen 3.5/3.6 render `…assistant\n<think>\n` into
    // the generation prompt): the output's thinking carries NO opener tag.
    // Needed up front by the tools branch — by the time the close tag is the
    // only evidence, visible-text flushing has already leaked the thoughts.
    // See the streaming chat site for why the prompt fact is NOT ANDed with
    // enable_thinking (LFM2.5's unconditional pre-opened `<think>`).
    const prompt_opened_think = promptOpensThink(allocator, tok, prompt_ids);
    const opens_think = prompt_opened_think;
    // Prompt-opened reasoning is DELIVERED as thinking blocks even when the
    // request didn't ask for thinking — generated tokens are never dropped.
    // A stream starts inside a think block only when the RENDERED PROMPT ends
    // inside one — never because the request asked for thinking. Qwen/Gemma
    // templates render the opener when thinking is on, so `promptOpensThink`
    // sees it; LFM2-VL's generation prompt is a bare `<|im_start|>assistant`
    // and its model answers directly, so seeding from the flag routed the whole
    // answer into reasoning_content and left `content` empty (live 2026-08-13).
    // A model that opens the block itself is picked up by `saw_think_open`.
    var channel_armed = false; // the `<|channel>` marker that may open a Gemma 4 thought
    const gated_stream = has_tools;
    // Constrained requests route through the protocol, not the response-side
    // close scan. The stream seeds the think arm while the payload has not
    // begun (a generated opener must strip, not leak), and flips straight to
    // the content arm when the generator reports a DIRECT answer — the whole
    // point is that no reasoning block exists to scan for.
    const constrained_proto = sampling.constraint != null and sampling.constraint.?.proto != null;
    // A stream starts in a think block from the rendered prompt, never from the request flag.
    var in_think_block = prompt_opened_think and !constrained_proto;
    var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
    defer if (delivery) |*d| d.deinit(allocator);
    // Set once the buffered think block has been split + emitted (tools
    // branch). Releases the buffer hold AND tells the end-of-stream split
    // that the remaining text has no template-opened semantics.
    var think_closed = false;
    var content_started = false;
    var think_buf = std.ArrayList(u8).empty;
    defer think_buf.deinit(allocator);
    var thought_shipped = false; // this thought has streamed a delta (`think_tags.openThoughtFlush`)
    var think_close_tag: []const u8 = "</think>";
    var skipped_think_open = false;
    var think_tokens: i32 = 0;
    var budget_exhausted = false;
    var reasoning_streamed: usize = 0; // bytes of the held thought already sent (tools path)

    var text_buf = std.ArrayList(u8).empty;
    defer text_buf.deinit(allocator);
    // The buffer keeps whole text (the tool gate scans it), so a split reads only what the client has
    // NOT received: a thought the model opens after visible text is its own bytes, not that text again.
    var text_delivered: usize = 0;
    // Memoized marker scan for the think gate. Owned BESIDE text_buf and reset
    // with it — the gate is otherwise O(buffer) per token for as long as a
    // think block is open and unclosed (47.99 us/token at 113 KB).
    var think_scan: think_tags.ThinkScan = .{};
    // Gate through streamThinkGateScan only, and reset the scan wherever the buffer is cleared.
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var stop_gate = stop_seq_mod.Gate{ .stops = stop_sequences };
    defer stop_gate.deinit(allocator);
    var tail_flush = false; // the generation ended: this turn delivers the gate's held tail
    var client_gone = false;

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
                sendAnthropicIdle(stream, slot_admission.queueWait(server_mod.global_scheduler.?, slot_handle.?)) catch {
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
        // The gate remembers WHICH stop matched (reported as stop_reason "stop_sequence" + the
        // echoed `stop_sequence` field in message_delta).
        const token_text = if (tail_flush) try stop_gate.finish(allocator) else blk: {
            const raw_decoded = try tok.decode(allocator, &[_]u32{token_id});
            defer allocator.free(raw_decoded);
            if (delivery) |*d| d.noteToken(raw_decoded.len, slot_handle.?.constraintPayloadStart());
            break :blk try stop_gate.push(allocator, raw_decoded);
        };
        stopped = stop_gate.matched != null;
        if (stopped) slot_handle.?.stop_hit.store(true, .release);
        if (token_text.len == 0) {
            allocator.free(token_text);
            if (stopped) break;
            continue;
        }

        if (gated_stream) {
            try text_buf.appendSlice(allocator, token_text);
        }

        if (delivery) |*d| {
            defer allocator.free(token_text);
            try d.feed(allocator, token_text);
            try emitConstrainedAnthropic(allocator, stream, &block_index, &thinking_block_open, &text_block_open, d);
            try beatStreamKeepalive(stream, .anthropic_ping);
            continue;
        }

        if (gated_stream) {
            // Buffer for tool detection. Detection rules live in
            // `think_tags.streamShouldBufferForTools` — the SAME predicate as the
            // chat-completions stream (this path once carried its own inline
            // subset, which drifted: the class the think gate was unified for).
            try token_texts.append(allocator, token_text);
            const buf = text_buf.items;
            const undelivered = buf[text_delivered..];
            const maybe_tool = think_tags.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // Shared gate with the chat-completions stream (the two paths
                // drifted once: this one only recognized think openers present
                // in the OUTPUT, so Qwen-family template-opened thinking
                // streamed as visible text_deltas and a raw `</think>` leaked
                // into Claude Code transcripts). Hermetically pinned per
                // recorded model family by the corpus streaming-gate test.
                switch (think_tags.streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think_scan)) {
                    .hold_thinking => {
                        // Safe to stream: the tool gate just returned false, so these bytes
                        // are reasoning. Each iteration is one token, so the budget counts them.
                        if (!budget_exhausted) if (heldThoughtSoFar(buf, opens_think, think_closed)) |settled| {
                            if (think_tags.unstreamedReasoning(settled, reasoning_streamed)) |fresh| {
                                try emitAnthropicThinkingAfterText(allocator, stream, &block_index, &text_block_open, &thinking_block_open, fresh);
                                reasoning_streamed = settled.len;
                                think_tokens += 1;
                                if (reasoning_budget >= 0 and think_tokens >= reasoning_budget) {
                                    budget_exhausted = true;
                                    log.info("  reasoning budget exhausted ({d}/{d} tokens, streamed)\n", .{ think_tokens, reasoning_budget });
                                }
                            }
                        };
                    },
                    .split_think => {
                        // Complete think block — split once: thinking block
                        // first, then the visible remainder as text. Reasoning
                        // ships regardless of the request's thinking flag.
                        const split = think_tags.splitThinkBlock(undelivered, true, opens_think);
                        if (!budget_exhausted) if (split.reasoning_content) |rc| {
                            const capped = try reasoningWithinBudget(allocator, tok, rc, reasoning_budget);
                            defer if (capped.owned) allocator.free(capped.text);
                            if (think_tags.unstreamedReasoning(capped.text, reasoning_streamed)) |rest| {
                                try emitAnthropicThinkingAfterText(allocator, stream, &block_index, &text_block_open, &thinking_block_open, rest);
                            }
                        };
                        if (thinking_block_open) try closeAnthropicBlock(allocator, stream, &block_index, &text_block_open, &thinking_block_open);
                        reasoning_streamed = 0;
                        if (split.content.len > 0) {
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            content_started = true;
                            try emitAnthropicTextDelta(allocator, stream, block_index, split.content);
                        }
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        text_delivered = 0;
                        think_scan.reset();
                        think_closed = true;
                    },
                    .flush_text => {
                        for (token_texts.items) |tt| {
                            defer allocator.free(tt);
                            // Skip bare think/channel tags that leak without a block
                            if (think_tags.isChannelMarkerToken(tt)) {
                                continue;
                            }
                            const vis = think_tags.streamContentLead(tt, content_started);
                            if (vis.len == 0) continue;
                            content_started = true;
                            if (!text_block_open) {
                                const sd = try std.fmt.allocPrint(allocator,
                                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                                , .{block_index});
                                defer allocator.free(sd);
                                try sendAnthropicEvent(stream, "content_block_start", sd);
                                text_block_open = true;
                            }
                            try emitAnthropicTextDelta(allocator, stream, block_index, vis);
                        }
                        token_texts.clearRetainingCapacity();
                        text_delivered = text_buf.items.len;
                    },
                }
            }
        } else if (in_think_block) {
            // Thinking block handling. Also covers thinking-off +
            // template-opened (LFM2.5 class): the block streams as thinking
            // deltas instead of being paid for and dropped.
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            if (!skipped_think_open and (think_buf.items.len >= 7 or !think_tags.thinkOpenerPossible(think_buf.items))) {
                if (think_tags.thinkOpenTagLenAt(think_buf.items)) |olen| {
                    var skip: usize = olen;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    think_close_tag = "<channel|>";
                    var skip: usize = 17;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Partial prefix of `<|channel>thought` — wait for more tokens.
                } else if (think_buf.items.len < 32 and think_tags.endsWithPartialThinkOpen(think_buf.items)) {
                    // Still a growing opener (`…<thi`) — wait.
                } else {
                    // No opener in the model's output — template injected one.
                    // Stay in the think block; close tag is detected dynamically.
                    skipped_think_open = true;
                }
            }

            // A spent budget closes the DELIVERED thinking block; the tail is withheld, never text.
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                if (!thinking_block_open) try openAnthropicThinkingBlock(allocator, stream, block_index);
                try closeAnthropicThinkingBlock(allocator, stream, block_index);
                thinking_block_open = false;
                block_index += 1;
            }

            // Check for the close tag — accept whichever appears first.
            const think_match = think_tags.indexOfThinkCloseTag(think_buf.items, 0);
            const channel_pos = std.mem.indexOf(u8, think_buf.items, "<channel|>");
            const close_match: ?struct { pos: usize, len: usize, is_channel: bool } = blk: {
                if (think_match == null and channel_pos == null) break :blk null;
                if (think_match == null) break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
                if (channel_pos == null) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                if (think_match.?.pos <= channel_pos.?) break :blk .{ .pos = think_match.?.pos, .len = think_match.?.len, .is_channel = false };
                break :blk .{ .pos = channel_pos.?, .len = "<channel|>".len, .is_channel = true };
            };

            if (close_match) |m| {
                const last = think_tags.closedThoughtDelta(think_buf.items[0..m.pos], thought_shipped);
                thought_shipped = false;
                if (!budget_exhausted and last.len > 0) {
                    try emitAnthropicThinkingOpening(allocator, stream, block_index, &thinking_block_open, last);
                }
                if (thinking_block_open) {
                    try closeAnthropicThinkingBlock(allocator, stream, block_index);
                    thinking_block_open = false;
                    block_index += 1;
                }
                const after = m.pos + m.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) content_after = content_after[11..];
                if (std.mem.startsWith(u8, content_after, "<|channel>")) content_after = content_after[10..];
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    content_started = true;
                    if (!text_block_open) {
                        const sd = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd);
                        try sendAnthropicEvent(stream, "content_block_start", sd);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, content_after);
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = if (m.is_channel) "<channel|>" else "</think>";
            } else if (skipped_think_open) {
                // Flush what the closed thought will certainly deliver: a partial
                // close tag, a trailing "\n " run and a split code point wait.
                const flush = think_tags.openThoughtFlush(think_buf.items, thought_shipped);
                const safe_len = flush.skip + flush.ship;
                if (safe_len > 0) {
                    if (!budget_exhausted and flush.ship > 0) try emitAnthropicThinkingOpening(allocator, stream, block_index, &thinking_block_open, think_buf.items[flush.skip..safe_len]);
                    if (flush.ship > 0) thought_shipped = true;
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            // Regular content token
            defer allocator.free(token_text);
            // A think block the MODEL opens: seeded like a template-injected one.
            // Checked before the marker skip, which would swallow a bare `<think>`.
            if (!content_started) if (think_tags.modelThinkOpener(channel_armed, token_text)) |opener| {
                channel_armed = false;
                in_think_block = true;
                try think_buf.appendSlice(allocator, opener);
                continue;
            };
            if (think_tags.isChannelMarkerToken(token_text)) {
                channel_armed = !content_started and std.mem.eql(u8, token_text, "<|channel>");
                continue;
            }
            channel_armed = false;
            const vis_token = think_tags.streamContentLead(token_text, content_started);
            if (vis_token.len == 0) continue;
            content_started = true;
            if (!text_block_open) {
                const sd = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                , .{block_index});
                defer allocator.free(sd);
                try sendAnthropicEvent(stream, "content_block_start", sd);
                text_block_open = true;
            }
            try emitAnthropicTextDelta(allocator, stream, block_index, vis_token);
        }

        // See the chat-completions loop: a buffered tool call / thinking block
        // emits no events at all, so liveness must be driven off socket
        // silence, not off token arrival.
        beatStreamKeepalive(stream, .anthropic_ping) catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            slot_handle.?.cancel();
            client_gone = true;
            break;
        };
    }

    if (!client_gone) {
        if (delivery) |*d| {
            d.clearOutput();
            try d.finish(allocator);
            try emitConstrainedAnthropic(allocator, stream, &block_index, &thinking_block_open, &text_block_open, d);
        }
    }

    // Flush remaining think buffer
    const think_tail = think_tags.cutThoughtDelta(think_buf.items, skipped_think_open, thought_shipped);
    if (!client_gone and !budget_exhausted and think_tail.len > 0) {
        try emitAnthropicThinkingOpening(allocator, stream, block_index, &thinking_block_open, think_tail);
    }

    // Post-generation: snapshot stats from whichever source was active.
    ts.finalize();

    // Handle tool calls
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;

    if (gated_stream and !client_gone) {
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            if (std.c.getenv("SUSHI_RAW_DUMP_FILE")) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B re-opens
        // channels mid-turn — observed live via Claude Code on this surface).
        const norm_owned = try think_tags.normalizeEmbeddedThinkBlocks(allocator, text_buf.items);
        defer if (norm_owned) |n| allocator.free(n);
        const gen_text: []const u8 = norm_owned orelse text_buf.items;
        const found_calls = if (has_tools and shouldParseToolCalls(ts.finish_details)) try parseToolCallsForRequest(allocator, gen_text, tools_json, allow_parallel_tools) else null;
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            // Emit thinking from buffered text if present — delivery does not
            // key on the request's thinking flag. Once the in-loop split
            // already emitted it, the remaining buffer has no template-opened
            // semantics — passing `opens_think` unguarded would misfile the
            // visible tail as reasoning.
            const think_split = think_tags.splitThinkBlock(gen_text, true, opens_think and !think_closed);
            if (!budget_exhausted) if (think_split.reasoning_content) |reasoning| {
                const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                defer if (capped.owned) allocator.free(capped.text);
                if (think_tags.unstreamedReasoning(capped.text, reasoning_streamed)) |rest| {
                    try emitAnthropicThinkingAfterText(allocator, stream, &block_index, &text_block_open, &thinking_block_open, rest);
                }
            };
            try closeAnthropicBlock(allocator, stream, &block_index, &text_block_open, &thinking_block_open);

            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "toolu_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const esc_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(esc_name);

                // content_block_start
                const start = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"tool_use","id":"{s}","name":{s},"input":{{}}}}}}
                , .{ block_index, tc_id, esc_name });
                defer allocator.free(start);
                try sendAnthropicEvent(stream, "content_block_start", start);

                // input_json_delta
                const esc_args_full = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(esc_args_full);
                const args_inner = esc_args_full[1 .. esc_args_full.len - 1];
                const delta = try std.fmt.allocPrint(allocator,
                    \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"input_json_delta","partial_json":"{s}"}}}}
                , .{ block_index, args_inner });
                defer allocator.free(delta);
                try sendAnthropicEvent(stream, "content_block_delta", delta);

                // content_block_stop
                const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{block_index});
                defer allocator.free(stop);
                try sendAnthropicEvent(stream, "content_block_stop", stop);
                block_index += 1;
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls — flush buffered tokens, splitting thinking from
            // content regardless of the request's thinking flag (split.content
            // passes trimLeakedToolMarkup, so the unparsed-markup cut the old
            // thinking-off arm did still happens).
            {
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| try full_text.appendSlice(allocator, t);
                const flush_norm = try think_tags.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = think_tags.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                if (!budget_exhausted) if (think_split.reasoning_content) |reasoning| {
                    const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                    defer if (capped.owned) allocator.free(capped.text);
                    if (think_tags.unstreamedReasoning(capped.text, reasoning_streamed)) |rest| {
                        try emitAnthropicThinkingAfterText(allocator, stream, &block_index, &text_block_open, &thinking_block_open, rest);
                    }
                };
                if (thinking_block_open) try closeAnthropicBlock(allocator, stream, &block_index, &text_block_open, &thinking_block_open);
                if (think_split.content.len > 0) {
                    content_started = true;
                    if (!text_block_open) {
                        const sd2 = try std.fmt.allocPrint(allocator,
                            \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"text","text":""}}}}
                        , .{block_index});
                        defer allocator.free(sd2);
                        try sendAnthropicEvent(stream, "content_block_start", sd2);
                        text_block_open = true;
                    }
                    try emitAnthropicTextDelta(allocator, stream, block_index, think_split.content);
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    const stop_reason = anthropicStopReason(finish_reason, stop_gate.matched);
    const echo_stop_seq = std.mem.eql(u8, stop_reason, "stop_sequence");
    const stop_seq_json: []const u8 = if (echo_stop_seq) try jsonEscape(allocator, stop_gate.matched.?) else "null";
    defer if (echo_stop_seq) allocator.free(stop_seq_json);

    if (!client_gone) {
        try closeAnthropicBlock(allocator, stream, &block_index, &text_block_open, &thinking_block_open);

        // Ensure at least one content block
        if (block_index == 0) {
            const sd = try std.fmt.allocPrint(allocator,
                \\{{"type":"content_block_start","index":0,"content_block":{{"type":"text","text":""}}}}
            , .{});
            defer allocator.free(sd);
            try sendAnthropicEvent(stream, "content_block_start", sd);
            const sd2 = "{\"type\":\"content_block_stop\",\"index\":0}";
            try sendAnthropicEvent(stream, "content_block_stop", sd2);
        }

        // The prefix-cache hit count is only known after prefill, so message_delta
        // restates input_tokens; clients take its usage over message_start's.
        {
            const usage = try formatAnthropicUsage(allocator, prompt_token_count, ts.completion_tokens, ts.cached_tokens);
            defer allocator.free(usage);
            const md = try std.fmt.allocPrint(allocator,
                \\{{"type":"message_delta","delta":{{"stop_reason":"{s}","stop_sequence":{s}}},"usage":{s}}}
            , .{ stop_reason, stop_seq_json, usage });
            defer allocator.free(md);
            try sendAnthropicEvent(stream, "message_delta", md);
        }
        try sendAnthropicEvent(stream, "message_stop", "{\"type\":\"message_stop\"}");
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, stop_reason,
    });
}

/// Emit a text_delta event for Anthropic streaming.
fn emitAnthropicTextDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, text: []const u8) !void {
    const esc = try jsonEscape(allocator, text);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"text_delta","text":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// Emit a thinking_delta event for Anthropic streaming.
fn emitAnthropicThinkingDelta(allocator: std.mem.Allocator, stream: *Conn, index: u32, thinking: []const u8) !void {
    const esc = try jsonEscape(allocator, thinking);
    defer allocator.free(esc);
    const inner = esc[1 .. esc.len - 1];
    const data = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"thinking_delta","thinking":"{s}"}}}}
    , .{ index, inner });
    defer allocator.free(data);
    try sendAnthropicEvent(stream, "content_block_delta", data);
}

/// A streamed thinking block opens at its first delta, so a thought that
/// delivers nothing streams no block, as the non-stream reply carries none.
fn emitAnthropicThinkingOpening(allocator: std.mem.Allocator, stream: *Conn, index: u32, open: *bool, thinking: []const u8) !void {
    if (!open.*) try openAnthropicThinkingBlock(allocator, stream, index);
    open.* = true;
    try emitAnthropicThinkingDelta(allocator, stream, index, thinking);
}

/// Close a thinking block with a fake signature and content_block_stop.
fn openAnthropicThinkingBlock(allocator: std.mem.Allocator, stream: *Conn, block_index: u32) !void {
    const sd = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_start","index":{d},"content_block":{{"type":"thinking","thinking":"","signature":""}}}}
    , .{block_index});
    defer allocator.free(sd);
    try sendAnthropicEvent(stream, "content_block_start", sd);
}

fn closeAnthropicThinkingBlock(allocator: std.mem.Allocator, stream: *Conn, index: u32) !void {
    const sig = try std.fmt.allocPrint(allocator,
        \\{{"type":"content_block_delta","index":{d},"delta":{{"type":"signature_delta","signature":"sushi-local"}}}}
    , .{index});
    defer allocator.free(sig);
    try sendAnthropicEvent(stream, "content_block_delta", sig);
    const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{index});
    defer allocator.free(stop);
    try sendAnthropicEvent(stream, "content_block_stop", stop);
}

/// End the open content block, so the next block starts at its own index.
fn closeAnthropicBlock(allocator: std.mem.Allocator, stream: *Conn, index: *u32, text_open: *bool, thinking_open: *bool) !void {
    if (thinking_open.*) {
        try closeAnthropicThinkingBlock(allocator, stream, index.*);
    } else if (text_open.*) {
        const stop = try std.fmt.allocPrint(allocator, "{{\"type\":\"content_block_stop\",\"index\":{d}}}", .{index.*});
        defer allocator.free(stop);
        try sendAnthropicEvent(stream, "content_block_stop", stop);
    } else return;
    text_open.* = false;
    thinking_open.* = false;
    index.* += 1;
}

/// A thought may follow visible text: the text block ends before the thinking block opens.
fn emitAnthropicThinkingAfterText(allocator: std.mem.Allocator, stream: *Conn, index: *u32, text_open: *bool, thinking_open: *bool, thinking: []const u8) !void {
    if (text_open.*) try closeAnthropicBlock(allocator, stream, index, text_open, thinking_open);
    try emitAnthropicThinkingOpening(allocator, stream, index.*, thinking_open, thinking);
}

/// The settled reasoning of a thought still open on the tools path, or null when it must wait for
/// its close: only a thought that starts the buffer (prompt-opened, or its opener first) splits to
/// the same prefix while open as once closed, so its deltas add up to the closed thought.
fn heldThoughtSoFar(buf: []const u8, opens_think: bool, think_closed: bool) ?[]const u8 {
    const prompt_opened = opens_think and !think_closed;
    if (!prompt_opened and think_tags.thinkOpenTagLenAt(buf) == null) return null;
    const reasoning = think_tags.splitThinkBlock(buf, true, prompt_opened).reasoning_content orelse return null;
    return think_tags.settledReasoning(reasoning);
}

test "anthropicStopReason: a matched client stop sequence reports stop_sequence, truncation stays honest" {
    // Anthropic spec: when a CLIENT stop sequence cut the text, stop_reason is
    // "stop_sequence" (with the matched string echoed) — that's how callers
    // learn WHICH stop fired. But it never masks the other reasons: a
    // max_tokens cut stays "max_tokens" (clients key truncation recovery on
    // it), a parsed tool call stays "tool_use".
    try std.testing.expectEqualStrings("stop_sequence", anthropicStopReason("stop", "beta"));
    try std.testing.expectEqualStrings("end_turn", anthropicStopReason("stop", null));
    try std.testing.expectEqualStrings("max_tokens", anthropicStopReason("length", "beta"));
    try std.testing.expectEqualStrings("tool_use", anthropicStopReason("tool_calls", "beta"));
}

test "parseAnthropicOutputConfig: Claude Code's shape yields effort AND the schema" {
    const allocator = std.testing.allocator;
    // Verbatim shape from the live 2026-08-16 capture (schema abbreviated).
    const body =
        \\{"output_config":{"effort":"high","format":{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}}}}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const oc = parseAnthropicOutputConfig(parsed.value.object);
    try std.testing.expectEqualStrings("high", oc.effort.?);
    try std.testing.expect(oc.schema != null);
    try std.testing.expect(oc.schema.? == .object);
}

test "parseAnthropicOutputConfig: absent, non-object, and non-schema shapes stay null" {
    const allocator = std.testing.allocator;
    const cases = [_][]const u8{
        "{}",
        \\{"output_config":"high"}
        ,
        // effort alone — no format means no schema, never a crash.
        \\{"output_config":{"effort":"low"}}
        ,
        // a format we don't serve is not a schema request.
        \\{"output_config":{"format":{"type":"text"}}}
        ,
    };
    for (cases) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const oc = parseAnthropicOutputConfig(parsed.value.object);
        try std.testing.expect(oc.schema == null);
        try std.testing.expect(!oc.schema_invalid);
    }
    // json_schema declared with nothing to enforce is flagged, never a silent fall-open.
    for ([_][]const u8{
        \\{"output_config":{"format":{"type":"json_schema"}}}
        ,
        \\{"output_config":{"format":{"type":"json_schema","schema":"nope"}}}
        ,
    }) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        const oc = parseAnthropicOutputConfig(parsed.value.object);
        try std.testing.expect(oc.schema == null and oc.schema_invalid);
    }
    // The effort-alone case still carries its word.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"output_config":{"effort":"low"}}
    , .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("low", parseAnthropicOutputConfig(parsed.value.object).effort.?);
}

test "anthropic messages: a forced tool_choice is committed at the answer boundary on both families" {
    const a = std.testing.allocator;
    const wire =
        \\[{"name":"get_time","description":"Get time","input_schema":{"type":"object","properties":{"timezone":{"type":"string"}}}},
        \\ {"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, wire, .{});
    defer parsed.deinit();
    const tools = try buildOpenAIToolsJson(a, parsed.value.array);
    defer a.free(tools);
    try expectSurfaceForcesCall("{\"type\":\"tool\",\"name\":\"get_time\"}", tools, "get_time");
    try expectSurfaceForcesCall("{\"type\":\"any\"}", tools, null);
}

/// Render one turn as the surface builds it: Responses input (with `instructions`)
/// or a /v1/messages body.
fn renderSurfaceTurnForTests(a: std.mem.Allocator, tpl: []const u8, surface: enum { responses, messages }, body_json: []const u8) ![]const u8 {
    var config = chat_mod.ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    const parsed = try std.json.parseFromSlice(std.json.Value, a, body_json, .{});
    defer parsed.deinit();
    switch (surface) {
        .responses => {
            var pi = try responses_mod.parseInput(a, parsed.value, "I", null, null, .{});
            defer pi.deinit();
            return chat_mod.renderChatTemplate(a, pi.messages.items, &config, null, null, true, null, false);
        },
        .messages => {
            var wire = WireMessages.init(a);
            defer wire.deinit();
            try readAnthropicConversation(&wire, parsed.value.object.get("system"), parsed.value.object.get("messages").?.array.items, .{});
            return chat_mod.renderChatTemplate(a, wire.messages.items, &config, null, null, true, null, false);
        },
    }
}

test "a late system turn keeps each turn's prompt a prefix of the next where the template places it" {
    const a = std.testing.allocator;
    // Codex sends the whole input each turn, with a new developer turn mid-conversation;
    // Claude Code's hook output arrives as a system message between turns.
    const responses_n =
        \\[{"role":"developer","content":"env: /repo"},{"role":"user","content":"u1"}]
    ;
    const responses_next =
        \\[{"role":"developer","content":"env: /repo"},{"role":"user","content":"u1"},
        \\ {"type":"message","role":"assistant","content":[{"type":"output_text","text":"a1"}]},
        \\ {"role":"developer","content":"approvals: never"},{"role":"user","content":"u2"}]
    ;
    const messages_n =
        \\{"system":"S","messages":[{"role":"user","content":"u1"}]}
    ;
    const messages_next =
        \\{"system":"S","messages":[{"role":"user","content":"u1"},{"role":"assistant","content":"a1"},
        \\ {"role":"system","content":"hook: started"},{"role":"user","content":"u2"}]}
    ;
    const Template = struct { tpl: []const u8, places_late_system: bool };
    const templates = [_]Template{
        .{ .tpl = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .places_late_system = true },
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .places_late_system = true },
        .{ .tpl = "{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('system must be first') }}{% endif %}{{ m.role + ':' + (m.content or '') + ';' }}{% endfor %}", .places_late_system = false },
    };
    for (templates) |t| {
        inline for (.{ .{ .responses, responses_n, responses_next, "approvals: never" }, .{ .messages, messages_n, messages_next, "hook: started" } }) |c| {
            const turn_n = try renderSurfaceTurnForTests(a, t.tpl, c[0], c[1]);
            defer a.free(turn_n);
            const turn_next = try renderSurfaceTurnForTests(a, t.tpl, c[0], c[2]);
            defer a.free(turn_next);
            try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, turn_next, c[3]));
            if (t.places_late_system) try std.testing.expect(std.mem.startsWith(u8, turn_next, turn_n));
        }
    }
}

test "completion IDs stay distinct with a repeated or backwards clock" {
    var counter = std.atomic.Value(i64).init(0);
    const times = [_]i64{ 100, 100, 99, 200, 200, 100 };
    var ids: [times.len]i64 = undefined;
    for (times, 0..) |now_ms, i| {
        ids[i] = nextCompletionId(&counter, now_ms);
        for (ids[0..i]) |previous| try std.testing.expect(previous != ids[i]);
    }
}

const AnthropicEventLog = struct {
    allocator: std.mem.Allocator,
    out: std.ArrayList(u8) = .empty,

    fn record(impl: *anyopaque, data: []const u8) anyerror!void {
        const self: *AnthropicEventLog = @ptrCast(@alignCast(impl));
        try self.out.appendSlice(self.allocator, data);
        try self.out.append(self.allocator, '\n');
    }
};

test "anthropic stream: a thought after visible text ends the text block and takes the next index" {
    const a = std.testing.allocator;
    var events = AnthropicEventLog{ .allocator = a };
    defer events.out.deinit(a);
    var bridge: WsBridge = .{ .impl = &events, .sendTextFn = &AnthropicEventLog.record };
    var conn: Conn = undefined;
    conn.ws_mode = &bridge;

    var index: u32 = 1;
    var text_open = true;
    var thinking_open = false;
    try emitAnthropicThinkingAfterText(a, &conn, &index, &text_open, &thinking_open, "plan");
    try emitAnthropicThinkingAfterText(a, &conn, &index, &text_open, &thinking_open, " A");
    try closeAnthropicBlock(a, &conn, &index, &text_open, &thinking_open);
    try std.testing.expectEqual(@as(u32, 3), index);
    try std.testing.expect(!text_open and !thinking_open);
    try std.testing.expectEqualStrings(
        \\{"type":"content_block_stop","index":1}
        \\{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"","signature":""}}
        \\{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"plan"}}
        \\{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":" A"}}
        \\{"type":"content_block_delta","index":2,"delta":{"type":"signature_delta","signature":"sushi-local"}}
        \\{"type":"content_block_stop","index":2}
        \\
    , events.out.items);
}

test "anthropic stream: a held thought streams only while its start cannot move" {
    try std.testing.expectEqualStrings("The user asks", heldThoughtSoFar("The user asks\n</thi", true, false).?);
    try std.testing.expectEqualStrings("plan", heldThoughtSoFar("<think>plan", false, true).?);
    try std.testing.expect(heldThoughtSoFar("Done. <think>plan", true, true) == null);
    try std.testing.expect(heldThoughtSoFar("Done. <th", true, true) == null);
}

test "formatAnthropicUsage: input_tokens leaves out the cache read a client adds back" {
    const a = std.testing.allocator;
    const warm = try formatAnthropicUsage(a, 3017, 1, 2986);
    defer a.free(warm);
    try std.testing.expectEqualStrings("{\"input_tokens\":31,\"output_tokens\":1,\"cache_read_input_tokens\":2986}", warm);
    const cold = try formatAnthropicUsage(a, 3017, 1, 0);
    defer a.free(cold);
    try std.testing.expectEqualStrings("{\"input_tokens\":3017,\"output_tokens\":1,\"cache_read_input_tokens\":0}", cold);
}

test "an Anthropic stream waiting in the queue gets the queue comment before each ping, and a plain ping otherwise" {
    const t = std.testing;
    const wait = slot_admission.QueueWait{ .ahead = 1, .running_prompt = 3000, .running_generated = 8000, .running_max = 32768, .tps = 0 };
    var sv: [2]std.posix.fd_t = undefined;
    try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
    try sendAnthropicIdle(&conn, wait);
    try sendAnthropicIdle(&conn, null);
    _ = std.c.close(sv[0]);
    var buf: [4096]u8 = undefined;
    const raw = readAllFromPair(sv[1], &buf);
    const ping = "event: ping\ndata: {\"type\":\"ping\"}\n\n";
    try t.expectEqualStrings(": queued behind 1 request(s); running 8000 of 32768 tokens\n\n" ++ ping ++ ping, raw);
}

/// A Qwen3.8-template model on the byte tokenizer: enough to count a /v1/messages body.
const CountTokensFixture = struct {
    arena: std.heap.ArenaAllocator,
    tok: Tokenizer,
    cfg: model_mod.ModelConfig,
    chat_cfg: chat_mod.ChatConfig,
    lm: LoadedModel,

    fn init(f: *CountTokensFixture) !void {
        const a = std.testing.allocator;
        f.arena = .init(a);
        f.tok = try byteTokenizerForTests(a, f.arena.allocator(), &.{ "<|im_start|>", "<|im_end|>", "<think>", "</think>" });
        // Thinking declared off, so only a test that names a thinking default meets one.
        f.cfg = .{ .num_hidden_layers = 0, .pinned_context = 4096, .model_type = "qwen4_exp", .gen_enable_thinking = false };
        f.chat_cfg = .{ .chat_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        f.lm = undefined;
        f.lm.id = "";
        f.lm.config = &f.cfg;
        f.lm.tokenizer = &f.tok;
        f.lm.chat_config = &f.chat_cfg;
        f.lm.transformer = null;
        f.lm.vision_encoder = null;
        f.lm.dflash = null;
        f.lm.mtp = null;
        f.lm.tokenize_cache = tokenize_cache_mod.TokenizeCache.init(a, 4);
    }

    fn deinit(f: *CountTokensFixture) void {
        f.lm.tokenize_cache.?.deinit();
        deinitTestTokenizer(&f.tok);
        f.arena.deinit();
    }

    /// The whole HTTP response count_tokens writes for `body`.
    fn post(f: *CountTokensFixture, body: []const u8) ![]u8 {
        return f.postTo(handleAnthropicCountTokens, body);
    }

    fn postTo(f: *CountTokensFixture, handler: *const fn (std.mem.Allocator, *Conn, []const u8, *LoadedModel) anyerror!void, body: []const u8) ![]u8 {
        const t = std.testing;
        var sv: [2]std.posix.fd_t = undefined;
        try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0); // AF_UNIX, SOCK_STREAM
        defer _ = std.c.close(sv[1]);
        var conn: Conn = undefined;
        Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
        try handler(t.allocator, &conn, body, &f.lm);
        _ = std.c.close(sv[0]);
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(t.allocator);
        var buf: [4096]u8 = undefined;
        while (true) {
            const n = std.c.read(sv[1], &buf, buf.len);
            if (n <= 0) break;
            try out.appendSlice(t.allocator, buf[0..@intCast(n)]);
        }
        return out.toOwnedSlice(t.allocator);
    }

    /// `input_tokens` from a 200 count_tokens answer.
    fn count(f: *CountTokensFixture, body: []const u8) !i64 {
        const t = std.testing;
        const response = try f.post(body);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 200 "));
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, responseBody(response), .{});
        defer parsed.deinit();
        try t.expectEqual(@as(usize, 1), parsed.value.object.count());
        return parsed.value.object.get("input_tokens").?.integer;
    }

    /// The oracle: `body`'s prompt rendered with the tools and thinking the test names, not resolves.
    fn rendered(f: *CountTokensFixture, body: []const u8, tools: bool, thinking: bool, effort: ?[]const u8) !i64 {
        const a = std.testing.allocator;
        const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        var wire = WireMessages.init(a);
        defer wire.deinit();
        try readAnthropicConversation(&wire, root.get("system"), root.get("messages").?.array.items, .{});
        const tools_json: ?[]const u8 = if (tools) try buildOpenAIToolsJson(a, root.get("tools").?.array) else null;
        defer if (tools_json) |tj| a.free(tj);
        const ids = try chat_mod.formatChat(a, &f.tok, wire.messages.items, &f.chat_cfg, tools_json, null, thinking, effort, false);
        defer a.free(ids);
        return @intCast(ids.len);
    }
};

test "a refused chat template answers a named 500 on every generative surface, streamed or not" {
    const t = std.testing;
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    f.chat_cfg.chat_template = "{{ raise_exception('no turn format') }}";
    f.chat_cfg.eos_token = "<|endoftext|>";
    const Case = struct { handler: *const fn (std.mem.Allocator, *Conn, []const u8, *LoadedModel) anyerror!void, body: []const u8 };
    const cases = [_]Case{
        .{ .handler = handleChatCompletions, .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .handler = handleChatCompletions, .body = "{\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .handler = handleAnthropicMessages, .body = "{\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .handler = handleAnthropicMessages, .body = "{\"max_tokens\":8,\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .handler = handleAnthropicCountTokens, .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" },
        .{ .handler = handleResponses, .body = "{\"input\":\"hi\"}" },
        .{ .handler = handleResponses, .body = "{\"stream\":true,\"input\":\"hi\"}" },
    };
    for (cases) |c| {
        const response = try f.postTo(c.handler, c.body);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 500 "));
        try t.expect(std.mem.indexOf(u8, responseBody(response), "chat template unusable: ") != null);
        try t.expect(std.mem.indexOf(u8, responseBody(response), "no turn format") != null);
    }
}

const count_tokens_sample = @embedFile("fixtures/count_tokens_sample.json");
const count_tokens_tools_body =
    \\{"model":"sushi","messages":[{"role":"user","content":"What time is it in Paris?"}],
    \\ "tools":[{"name":"get_time","description":"Get the time","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}
;
const count_tokens_thinking_body =
    \\{"model":"sushi","max_tokens":16,"thinking":{"type":"enabled","budget_tokens":1024},
    \\ "messages":[{"role":"user","content":"This is some text that Claude Code wants token count of"}]}
;

test "count_tokens: the sample counts its /v1/messages prompt, empty tools as no tools, any model name alike" {
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    const want = try f.rendered(count_tokens_sample, false, false, null);
    try std.testing.expectEqual(want, try f.count(count_tokens_sample));
    // Routing picks the model; an unknown name reaching the default counts the same body the same.
    const renamed = try std.mem.replaceOwned(u8, std.testing.allocator, count_tokens_sample, "Qwen3.8-Flash-Next-Sushi-2bpw", "gpt-4");
    defer std.testing.allocator.free(renamed);
    try std.testing.expectEqual(want, try f.count(renamed));
}

test "count_tokens: an absent thinking field takes the model's default, as /v1/messages does" {
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    // Qwen's own default: thinking on at medium effort.
    f.cfg.gen_enable_thinking = null;
    const thinking = try f.rendered(count_tokens_sample, false, true, "medium");
    try std.testing.expect(thinking != try f.rendered(count_tokens_sample, false, false, null));
    try std.testing.expectEqual(thinking, try f.count(count_tokens_sample));
}

test "count_tokens: tools and an enabled thinking block count as /v1/messages renders them" {
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    const with_tools = try f.rendered(count_tokens_tools_body, true, false, null);
    try std.testing.expect(with_tools != try f.rendered(count_tokens_tools_body, false, false, null));
    try std.testing.expectEqual(with_tools, try f.count(count_tokens_tools_body));

    const thinking = try f.rendered(count_tokens_thinking_body, false, true, null);
    try std.testing.expect(thinking != try f.rendered(count_tokens_thinking_body, false, false, null));
    try std.testing.expectEqual(thinking, try f.count(count_tokens_thinking_body));
}

test "count_tokens: a prompt past the context is still counted" {
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    f.cfg.pinned_context = 8;
    const want = try f.rendered(count_tokens_sample, false, false, null);
    try std.testing.expect(want > 8);
    try std.testing.expectEqual(want, try f.count(count_tokens_sample));
}

test "count_tokens: the tokenize LRU is the same before and after, on a hit and on a miss" {
    const a = std.testing.allocator;
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    // Seed the entry /v1/messages leaves for the sample, so counting it would be a hit.
    {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, count_tokens_sample, .{});
        defer parsed.deinit();
        var wire = WireMessages.init(a);
        defer wire.deinit();
        try readAnthropicConversation(&wire, null, parsed.value.object.get("messages").?.array.items, .{});
        const ids = try cachedFormatChat(a, std.testing.io, &f.lm, &f.tok, &f.chat_cfg, wire.messages.items, null, null, false, null, false);
        a.free(ids);
    }
    const cache = &f.lm.tokenize_cache.?;
    const Snap = struct { key: u64, last_used: u64 };
    var before: [4]Snap = undefined;
    const n = cache.entries.items.len;
    try std.testing.expectEqual(@as(usize, 1), n);
    for (cache.entries.items, before[0..n]) |e, *s| s.* = .{ .key = e.key, .last_used = e.last_used };
    const counter = cache.counter;

    _ = try f.count(count_tokens_sample);
    _ = try f.count(count_tokens_tools_body);

    try std.testing.expectEqual(n, cache.entries.items.len);
    try std.testing.expectEqual(counter, cache.counter);
    for (cache.entries.items, before[0..n]) |e, s| {
        try std.testing.expectEqual(s.key, e.key);
        try std.testing.expectEqual(s.last_used, e.last_used);
    }
}

test "count_tokens: a malformed body is an Anthropic invalid_request_error" {
    const t = std.testing;
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    for ([_][]const u8{ "{not json", "[]", "{\"model\":\"sushi\"}", "{\"messages\":\"hi\"}" }) |body| {
        const response = try f.post(body);
        defer t.allocator.free(response);
        try t.expect(std.mem.startsWith(u8, response, "HTTP/1.1 400 "));
        try t.expect(std.mem.startsWith(u8, responseBody(response), "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\""));
    }
}

test "count_tokens: a rerouted effort word counts as served and says so, as /v1/messages does" {
    const t = std.testing;
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"output_config\":{\"effort\":\"high\"}}";
    // qwen4_exp has no `high`: /v1/messages serves it as `xhigh`, which renders its own preamble.
    const want = try f.rendered(body, false, true, "xhigh");
    try t.expect(want != try f.rendered(body, false, true, null));
    const response = try f.post(body);
    defer t.allocator.free(response);
    try t.expect(std.mem.indexOf(u8, response, "\r\nX-Sushi-Effort-Notice: ") != null);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, responseBody(response), .{});
    defer parsed.deinit();
    try t.expectEqual(want, parsed.value.object.get("input_tokens").?.integer);
    try t.expect(parsed.value.object.get("effort_notice").? == .string);
}

test "count_tokens: both routes render a body to the same prompt, schema, tools, system blocks and effort included" {
    const t = std.testing;
    var f: CountTokensFixture = undefined;
    try f.init();
    defer f.deinit();
    f.cfg.gen_enable_thinking = null;
    const bodies = [_][]const u8{
        count_tokens_sample,
        count_tokens_tools_body,
        count_tokens_thinking_body,
        "{\"system\":[{\"type\":\"text\",\"text\":\"Be brief.\",\"cache_control\":{\"type\":\"ephemeral\"}}],\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\",\"cache_control\":{\"type\":\"ephemeral\"}}]}],\"output_config\":{\"effort\":\"high\"}}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"Sure,\"}]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"output_config\":{\"format\":{\"type\":\"json_schema\",\"schema\":{\"type\":\"object\"}}}}",
    };
    for (bodies) |body| {
        var sv: [2]std.posix.fd_t = undefined;
        try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
        defer _ = std.c.close(sv[0]);
        defer _ = std.c.close(sv[1]);
        var conn: Conn = undefined;
        Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, body, .{});
        defer parsed.deinit();
        var served: AnthropicPrompt = .{ .wire = WireMessages.init(t.allocator) };
        defer served.deinit();
        try t.expect(try renderAnthropicPrompt(t.allocator, &conn, &f.lm, parsed.value.object, .messages, &served));
        clearEffortNotice(&conn);
        var counted: AnthropicPrompt = .{ .wire = WireMessages.init(t.allocator) };
        defer counted.deinit();
        try t.expect(try renderAnthropicPrompt(t.allocator, &conn, &f.lm, parsed.value.object, .count_tokens, &counted));
        clearEffortNotice(&conn);
        try t.expect(served.ids.len > 0);
        try t.expectEqualSlices(u32, served.ids, counted.ids);
        try t.expectEqual(served.enable_thinking, counted.enable_thinking);
        try t.expectEqual(served.has_tools, counted.has_tools);
    }
}
