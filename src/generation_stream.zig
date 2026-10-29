const std = @import("std");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const generate_mod = @import("generate.zig");
const chat_mod = @import("chat.zig");
const think_tags = @import("think_tags.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const rp_mod = @import("reasoning_protocol.zig");
const expert_stream_mod = @import("expert_stream.zig");
const log = @import("log.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const scheduler_mod = @import("scheduler.zig");
const vision_queue = @import("vision_queue.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const LoadedModel = model_registry_mod.LoadedModel;
const http_conn = @import("http_conn.zig");
const Conn = http_conn.Conn;
const sendSseHeaders = http_conn.sendSseHeaders;
const request_policy = @import("request_policy.zig");
const withEffortNoticeJson = request_policy.withEffortNoticeJson;
const logprobs = @import("logprobs.zig");
const StreamLogprobs = logprobs.StreamLogprobs;
const formatLogprobsObject = logprobs.formatLogprobsObject;
const io_util = @import("io_util.zig");
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
const Stopwatch = io_util.Stopwatch;
const nextCompletionId = io_util.nextCompletionId;
const server_mod = @import("server.zig");
const anthropic = @import("anthropic.zig");
const responses_api = @import("responses_api.zig");
const completions = @import("completions.zig");
const FakeSlot = server_mod.FakeSlot;
const GEN_OOM_MSG = server_mod.GEN_OOM_MSG;
const MropeData = server_mod.MropeData;
const SSE_ALLOW_HEADERS_DEFAULT = server_mod.SSE_ALLOW_HEADERS_DEFAULT;
const TickPlayer = server_mod.TickPlayer;
const appendRawToolDump = server_mod.appendRawToolDump;
const buildAnthropicStreamError = anthropic.buildAnthropicStreamError;
const buildResponsesStreamError = responses_api.buildResponsesStreamError;
const completionShowsToken = completions.completionShowsToken;
const completionText = completions.completionText;
const finishDetailsField = server_mod.finishDetailsField;
const getTimeoutNs = server_mod.getTimeoutNs;
const insufficient_free_memory_message = server_mod.insufficient_free_memory_message;
const jsonEscape = server_mod.jsonEscape;
const jsonEscapeOrEmpty = server_mod.jsonEscapeOrEmpty;
const logHttpSseData = server_mod.logHttpSseData;
const memoryContextRefusalMessage = server_mod.memoryContextRefusalMessage;
const mtpCapable = server_mod.mtpCapable;
const not_enough_memory_message = server_mod.not_enough_memory_message;
const parseToolCallsForRequest = server_mod.parseToolCallsForRequest;
const promptOpensThink = server_mod.promptOpensThink;
const reasoningWithinBudget = server_mod.reasoningWithinBudget;
const sendAnthropicError = anthropic.sendAnthropicError;
const sendAnthropicEvent = anthropic.sendAnthropicEvent;
const sendErrorResponse = server_mod.sendErrorResponse;
const sendResponse = server_mod.sendResponse;
const sendResponsesEvent = responses_api.sendResponsesEvent;
const shouldParseToolCalls = server_mod.shouldParseToolCalls;
const splitConstrainedResponse = responses_api.splitConstrainedResponse;
const toolCallFinishReason = server_mod.toolCallFinishReason;

/// Tokens a client should be shown after a loop cut: everything before the
/// degenerate span. `start` is an index into the SAME emitted-token sequence,
/// but it is computed from the generator's own list, so it is clamped rather
/// than trusted — a mismatch must degrade to "emit everything", never to a
/// slice out of bounds.
fn loopTrimmedIds(ids: []const u32, start: ?usize) []const u32 {
    const s = start orelse return ids;
    if (s >= ids.len) return ids;
    return ids[0..s];
}

/// A slot that errored reports why: the MLX error latch's `error.OutOfMemory` is a memory
/// failure the operator can act on. One helper for the three slot-drain sites.
fn slotFailure(slot: *scheduler_mod.Slot) anyerror {
    server_mod.slot_memory_context_refusal = slot.memory_context_refusal;
    server_mod.slot_media_refusal = slot.media_refusal;
    if (slot.shutdown_cancelled.load(.acquire)) return error.ServerShutdown;
    if (slot.errorNameIs("MediaRefused")) return error.MediaRefused;
    // A pre-prefill admission refusal is not an abandoned generation: the client gets a 400.
    if (slot.errorNameIs("PrefillDoesNotFit")) return error.PrefillDoesNotFit;
    if (slot.errorNameIs("TransientCapExceeded")) return error.TransientCapExceeded;
    if (slot.errorIsMemory()) return error.GenerationOutOfMemory;
    return error.GenerationFailed;
}

/// The message `error.TransientCapExceeded` sends: MLX refused an allocation past the transient memory cap.
const TRANSIENT_CAP_MSG = "The request needed more transient GPU memory than the engine's cap allows and was abandoned. The server is still running. Reduce the prompt length or the image size.";

/// The message `error.PrefillDoesNotFit` sends: refused before its first forward by the same
/// estimator the guard uses, re-asked after the hot cache gave back everything it could.
pub const PREFILL_NOFIT_MSG = "This prompt does not fit in GPU memory even after freeing the prefix cache; it was refused before any work started. Reduce the prompt length, lower --ctx-size, or free memory on the machine (the server log quotes the byte counts it compared).";

/// The message `error.EmptyPrompt` sends: generation continues from the prompt's last token.
pub const EMPTY_PROMPT_MSG = "The prompt tokenizes to zero tokens; send non-empty text to continue.";

/// A generation failure as the wire sees it: one status, one message, both dialects' type.
pub const GenErrorWire = struct {
    status_line: []const u8,
    code: u32,
    openai_type: []const u8,
    anthropic_type: []const u8,
    message: []const u8,
};

/// The mapping from a failed generation to what the client is told. It existed only on the
/// non-streaming arms; streaming arms wrote a bare `server_error` into an SSE frame, so agents
/// never saw the actionable 400/503. The unknown arm keeps the error name at a mapped 500.
pub fn mapGenerationError(err: anyerror, buf: []u8) GenErrorWire {
    const context_refusal = server_mod.slot_memory_context_refusal;
    server_mod.slot_memory_context_refusal = null;
    const media_refusal = server_mod.slot_media_refusal;
    server_mod.slot_media_refusal = null;
    return switch (err) {
        // The media encode the prefix restore left to do could not run: the same text and status the
        // connection thread gave before the encode moved behind the restore.
        error.MediaRefused => if (media_refusal) |r| .{
            .status_line = if (r.server_error) "500 Internal Server Error" else "400 Bad Request",
            .code = if (r.server_error) 500 else 400,
            .openai_type = if (r.server_error) "server_error" else "invalid_request_error",
            .anthropic_type = if (r.server_error) "api_error" else "invalid_request_error",
            .message = std.fmt.bufPrint(buf, "{s}", .{r.text()}) catch "media refused",
        } else .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = "media refused",
        },
        // The MLX working-set latch (#353): a memory 503.
        error.GenerationOutOfMemory, error.OutOfMemory => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = GEN_OOM_MSG,
        },
        error.TransientCapExceeded => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = TRANSIENT_CAP_MSG,
        },
        // Refused before the first forward: the request is too large, a named 400.
        error.PrefillDoesNotFit => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = if (context_refusal) |r| memoryContextRefusalMessage(buf, r) else PREFILL_NOFIT_MSG,
        },
        // Refused at submit, so every surface answers it the same way.
        error.EmptyPrompt => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = EMPTY_PROMPT_MSG,
        },
        // The shutdown drain ended a request that was queued or running.
        error.ServerShutdown => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = "Server is shutting down",
        },
        error.GenerationFailed => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = "generation failed",
        },
        error.ChatTemplateUnusable => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = std.fmt.bufPrint(buf, "chat template unusable: {s}", .{chat_mod.templateRefusal()}) catch "chat template unusable",
        },
        else => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = std.fmt.bufPrint(buf, "Internal server error: {s}", .{@errorName(err)}) catch
                "Internal server error",
        },
    };
}

/// The terminal SSE frame for the OpenAI dialects, with a `finish_reason: "error"` choice so a
/// chunk-only client terminates.
fn buildOpenAiStreamError(allocator: std.mem.Allocator, w: GenErrorWire) ![]u8 {
    const esc = try jsonEscape(allocator, w.message);
    defer allocator.free(esc);
    return std.fmt.allocPrint(allocator,
        \\data: {{"error":{{"message":{s},"type":"{s}","param":null,"code":{d}}},"choices":[{{"index":0,"delta":{{}},"finish_reason":"error"}}]}}
    ++ "\n\n", .{ esc, w.openai_type, w.code });
}

/// Which wire dialect a generative surface speaks its errors in.
const ErrorSurface = union(enum) {
    openai,
    anthropic,
    responses: *u64,
};

/// The error arm for every generative surface, streaming or not: before the SSE head the
/// stream gets the same response its non-streaming twin would; past it the same `type` and
/// `message` ride the surface's terminal `error` event.
pub fn sendGenerationError(allocator: std.mem.Allocator, stream: *Conn, err: anyerror, surface: ErrorSurface) !void {
    var msg_buf: [192]u8 = undefined;
    const w = mapGenerationError(err, &msg_buf);

    if (!stream.sse_headers_sent) {
        return switch (surface) {
            .anthropic => sendAnthropicError(allocator, stream, w.anthropic_type, w.message, w.code),
            else => sendErrorResponse(allocator, stream, w.status_line, w.openai_type, w.message, w.code),
        };
    }

    switch (surface) {
        .openai => {
            const body = try buildOpenAiStreamError(allocator, w);
            defer allocator.free(body);
            try stream.writeAllNoFlush(body);
            try stream.writeAll("data: [DONE]\n\n");
        },
        .anthropic => {
            const body = try buildAnthropicStreamError(allocator, w);
            defer allocator.free(body);
            // Anthropic streams have no `[DONE]`; the `error` event is the terminator.
            try sendAnthropicEvent(stream, "error", body);
        },
        .responses => |seq| {
            const body = try buildResponsesStreamError(allocator, w);
            defer allocator.free(body);
            try sendResponsesEvent(allocator, stream, seq, "error", body);
            // Same gate as the success path: no `[DONE]` over the WS bridge.
            if (stream.ws_mode == null) try stream.writeAll("data: [DONE]\n\n");
        },
    }
}

/// A client that vanished mid-generation did not hit a token cap.
fn nonStreamFinishReason(client_gone: bool, slot_reason: []const u8) []const u8 {
    return if (client_gone) "client_disconnect" else slot_reason;
}

/// The text the model said before the tool markup, byte for byte as the stream flushes it.
pub fn visibleToolPreamble(content: []const u8) []const u8 {
    return if (std.mem.trim(u8, content, " \t\r\n").len > 0) content else "";
}

fn deliveredReasoning(capped: ?[]const u8, raw: ?[]const u8) ?[]const u8 {
    return capped orelse raw;
}

/// Client stop sequences a non-stream surface hands the scheduler wait, so generation ends when one completes.
const StopSpec = struct {
    stops: []const []const u8 = &.{},
    /// Completions decode the raw continuation (null = chat decode); true drops flagged special tokens.
    completion_skip_special: ?bool = null,
};

/// Feeds the stream path's `Gate` token by token. The answer is still cut by `stop_sequences.earliest`
/// over the whole decoded text, so generation ends only on a stop that text also contains.
const EarlyStop = struct {
    tok: *const Tokenizer,
    gate: stop_seq_mod.Gate,
    completion_skip_special: ?bool,
    spent: bool = false,

    /// `ids` holds every token so far; the last one is new. True once the generation can end.
    fn observe(self: *EarlyStop, allocator: std.mem.Allocator, ids: []const u32) !bool {
        if (self.spent) return false;
        const id = ids[ids.len - 1];
        if (self.completion_skip_special) |skip| if (!completionShowsToken(self.tok, id, skip)) return false;
        const piece = try self.tok.decode(allocator, &[_]u32{id});
        defer allocator.free(piece);
        allocator.free(try self.gate.push(allocator, piece));
        if (self.gate.matched == null) return false;
        self.spent = true;
        const text = if (self.completion_skip_special) |skip|
            try completionText(allocator, self.tok, ids, skip)
        else
            try self.tok.decode(allocator, ids);
        defer allocator.free(text);
        return stop_seq_mod.earliest(text, self.gate.stops) != null;
    }
};

const DrainEnd = enum { done, client_gone, stopped, failed };

/// Connection thread: collect the slot's tokens until it ends, the client leaves or `early` sees a stop complete.
fn drainSlotTokens(slot: anytype, conn: ?*Conn, allocator: std.mem.Allocator, output_ids: *std.ArrayList(u32), early: ?*EarlyStop) !DrainEnd {
    while (true) {
        const nr = slot.waitNextTimeout(Conn.STREAM_KEEPALIVE_MS) orelse {
            // Idle (long prefill). If the client is gone, cancel and serve
            // whatever accumulated — the response write will fail upstream,
            // which is fine; the win is freeing the GPU.
            if (conn) |c| {
                if (c.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting (non-stream) — cancelling slot\n", .{});
                    slot.cancel();
                    return .client_gone;
                }
            }
            continue;
        };
        switch (nr) {
            .token => |t| try output_ids.append(allocator, t),
            .done => return .done,
            .err => return .failed,
        }
        if (conn) |c| {
            if (c.peerClosed()) {
                log.info("  [cancel] client disconnected while decoding (non-stream) — cancelling slot\n", .{});
                slot.cancel();
                return .client_gone;
            }
        }
        if (early) |e| if (try e.observe(allocator, output_ids.items)) {
            slot.cancelOnStop();
            return .stopped;
        };
    }
}

/// `drainSlotTokens`, then, after an early stop, wait out the inference pass still publishing and accounting
/// the cancelled slot: its statistics are plain fields that only a quiet slot lets the caller read.
fn drainSettled(sch: *scheduler_mod.Scheduler, slot: *scheduler_mod.Slot, conn: ?*Conn, allocator: std.mem.Allocator, output_ids: *std.ArrayList(u32), early: ?*EarlyStop) !DrainEnd {
    const end = try drainSlotTokens(slot, conn, allocator, output_ids, early);
    if (end == .stopped) sch.quiesce(slot);
    return end;
}

/// Run a non-streaming generation through the scheduler. Returns the same
/// shape as `generate.generate` so the calling handler's response builder
/// is unchanged.
///
/// `deferred_media`: when non-null, ownership transfers into the slot, which
/// encodes it after the prefix restore. Each handler holds a per-request local;
/// it nulls its copy before passing here so its own `defer` doesn't double-free.
pub fn nonStreamingViaScheduler(
    allocator: std.mem.Allocator,
    sch: *scheduler_mod.Scheduler,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    full_prompt: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    cached_tokens: u32,
    has_tools: bool,
    enable_thinking: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    timeout_ns: u64,
    deferred_media: ?*vision_queue.DeferredMedia,
    media_chain: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    logprobs_n: u32,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// When non-null, the peer socket is probed on idle wakeups during the
    /// wait — a vanished client cancels the slot (aborting its prefill)
    /// instead of grinding out a ghost generation nobody will read.
    conn: ?*Conn,
    stop: StopSpec,
) !generate_mod.GenerationResult {
    var slot = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = full_prompt,
        .cached_tokens = cached_tokens,
        .has_tools = has_tools,
        .enable_thinking = enable_thinking,
        .sampling = sampling,
        .eos_token_ids = eos_token_ids,
        .max_tokens = max_tokens,
        .timeout_ns = timeout_ns,
        .enable_drafter = enable_drafter and (lm.dflash != null),
        .dflash = if (enable_drafter) lm.dflash else null,
        .drafter_block_size = lm.drafter_block_size,
        .allow_batch_mtp = allow_batch_mtp,
        .enable_mtp = enable_mtp and mtpCapable(lm),
        .mtp = if (enable_mtp) lm.mtp else null,
        .mtp_depth = lm.mtp_depth,
        .deferred_media = deferred_media,
        .vision_key = prefix_cache_mod.chainKey(media_chain),
        .media_chain = media_chain,
        .cache_key = cache_key,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .logprobs_n = logprobs_n,
        .kv_quant_config = kv_quant_override,
    });
    defer sch.complete(slot);

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var early: ?EarlyStop = if (stop.stops.len > 0) .{
        .tok = tok,
        .gate = .{ .stops = stop.stops },
        .completion_skip_special = stop.completion_skip_special,
    } else null;
    defer if (early) |*e| e.gate.deinit(allocator);
    const end = try drainSettled(sch, slot, conn, allocator, &output_ids, if (early) |*e| e else null);
    if (end == .failed) return slotFailure(slot);
    const client_gone = end == .client_gone;

    // The scheduler measures prefill_ns / decode_ns per-slot directly. Pull
    // those instead of the old single-wall-clock approximation, which
    // double-counted total time into both phases.
    const prefill_tps = generate_mod.prefillTokensPerSec(slot.prompt_tokens, slot.cached_tokens, slot.prefill_ns);
    const decode_tps = generate_mod.tokensPerSec(slot.completion_tokens, slot.decode_ns);

    // A loop cut is a truncation, and the degenerate span is the part the
    // client must not get: an agent re-sends the cut turn as history, the
    // model reads its own loop back and resumes it — five loop-stops in a
    // row, each firing sooner than the last (live 2026-08-05, under pi).
    // Only reachable non-streaming: a delta cannot be retracted, so a
    // streaming client has already received the tail.
    const emit_ids = loopTrimmedIds(output_ids.items, slot.loop_trim_start);
    const text = try tok.decode(allocator, emit_ids);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    // Convert the generator's (token index, byte offset) payload boundary
    // into a byte offset inside `text`. The prefix decode uses the same
    // leading-space convention as the full text, so offsets are exact.
    var constraint_payload_byte: ?usize = null;
    if (slot.constraintPayloadStart()) |ps| {
        var pb: ?usize = ps.byte_offset;
        if (ps.token_index > 0) {
            if (ps.token_index <= emit_ids.len) {
                const prefix = tok.decode(allocator, emit_ids[0..ps.token_index]) catch null;
                if (prefix) |p| {
                    pb = p.len + ps.byte_offset;
                    allocator.free(p);
                } else pb = null;
            } else pb = null;
        }
        if (pb) |b| {
            if (b <= text.len) constraint_payload_byte = b;
        }
    }

    // Phase A5: take ownership of the slot's accumulated logprobs. After
    // `toOwnedSlice` the slot's list is empty, so `Slot.deinit` doesn't try
    // to free what we just transferred to the caller.
    const logprobs_slice: ?[]sampler.LogprobResult = slot.takeLogprobs(if (end == .stopped) token_ids.len else std.math.maxInt(usize));

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = slot.prompt_tokens,
        // An early stop reports the tokens it returned, not the ones decoded past the stop.
        .completion_tokens = if (end == .stopped) @intCast(token_ids.len) else slot.completion_tokens,
        .finish_reason = nonStreamFinishReason(client_gone, slot.finish_reason),
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .prefill_ns = slot.prefill_ns,
        .decode_ns = slot.decode_ns,
        .cached_tokens = slot.cached_tokens,
        .logprobs = logprobs_slice,
        .finish_details = slot.finish_details,
        .constraint_payload_byte = constraint_payload_byte,
    };
}

/// Generation is authoritative for constrained responses: when the generator
/// published a payload boundary, the split is a byte cut at that offset —
/// reasoning is everything before it (delimiter markup stripped, whitespace
/// policy matched to the streaming suppression), content is everything after,
/// UNREParsed. A standalone re-scan of the payload would read marker text
/// inside JSON string data ("the close delimiter") as protocol structure and
/// truncate valid JSON. Without a boundary the existing parser applies.
fn emitConstrainedChat(a: std.mem.Allocator, stream: *Conn, id: i64, model: []const u8, lps: *StreamLogprobs, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = null, .reasoning_content = d.reasoning.items }, null, null, null, .{ .logprobs_json = null });
    }
    if (d.content.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = d.content.items }, null, null, null, .{ .logprobs_json = try lps.take() });
    } else lps.dropPending();
}

pub fn handleNonStreamingGeneration(
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
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    deferred_media: ?*vision_queue.DeferredMedia,
    media_chain: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns from the parent handleChatCompletions, so
    /// the non-streaming chat response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
) !void {
    // Phase A8: this handler owns the vision array on entry; ownership
    // transfers to the slot on submit (the scheduler's `Slot.deinit` frees
    // it). Nulled before transfer so the early-return defer is a no-op.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    var timer = Stopwatch.init(stream.io);

    // Speculative-decoding dispatch (priority: DFlash > MTP > regular).
    //   1. Drafter wins if loaded AND requested AND no logprobs (drafter
    //      cannot expose realized log-probs from the draft side).
    //   2. PLD next if requested AND no logprobs AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   3. Otherwise the regular pipeline.
    const spec = requestSpecModes(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, logprobs_n);
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;

    // Transfer vision ownership into the slot via the scheduler.
    const slot_ve: ?*vision_queue.DeferredMedia = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // Propagates to `handleChatCompletions`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, server_mod.global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve, media_chain, cache_key, mrope, logprobs_n, kv_quant_override, stream, .{ .stops = stop_sequences });
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);
    defer if (result.logprobs) |lps| {
        for (lps) |*lp| allocator.free(lp.top_logprobs);
        allocator.free(lps);
    };

    // Apply stop sequences: truncate at the earliest occurrence (`stop_sequences.zig`)
    var final_text: []const u8 = result.text;
    var finish_reason = result.finish_reason;
    if (stop_seq_mod.earliest(final_text, stop_sequences)) |hit| {
        final_text = final_text[0..hit.index];
        finish_reason = "stop";
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

    // Template-opened think block (Qwen 3.5/3.6): unclosed output is reasoning.
    // Not ANDed with enable_thinking: generated reasoning is always DELIVERED,
    // never paid-for-and-dropped — thinking-off is enforced on the PROMPT side
    // (think_tags.noThinkTailSuffix commits the channel), so any reasoning that
    // still shows up here is real work the client gets to see.
    const opens_think = promptOpensThink(allocator, tok, prompt_ids);

    // Apply reasoning budget: truncate reasoning by token count
    // For non-streaming, we truncate after generation since we can't interrupt mid-generation
    var budget_truncated_reasoning: ?[]const u8 = null;
    var budget_reasoning_allocated = false;
    defer if (budget_reasoning_allocated) allocator.free(budget_truncated_reasoning.?);

    if (enable_thinking and reasoning_budget >= 0) {
        const think_split = think_tags.splitThinkBlock(final_text, true, opens_think);
        if (think_split.reasoning_content) |reasoning| {
            // Count tokens in reasoning by encoding it
            const reasoning_ids = try tok.encode(allocator, reasoning);
            defer allocator.free(reasoning_ids);
            if (reasoning_ids.len > @as(usize, @intCast(reasoning_budget))) {
                // Truncate: decode only budget-many tokens
                const budget_usize: usize = @intCast(reasoning_budget);
                const truncated_ids = reasoning_ids[0..budget_usize];
                const truncated_text = try tok.decode(allocator, truncated_ids);
                budget_truncated_reasoning = truncated_text;
                budget_reasoning_allocated = true;
                log.info("  reasoning budget truncated ({d}/{d} tokens)\n", .{ budget_usize, reasoning_ids.len });
            }
        }
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    // Check for tool calls in the output
    if (has_tools and shouldParseToolCalls(result.finish_details)) {
        log.debug("  checking {d} bytes of generated text for tool calls\n", .{final_text.len});
        const found_calls = try parseToolCallsForRequest(allocator, final_text, tools_json, allow_parallel_tools);
        if (found_calls) |tool_calls| {
            defer {
                for (tool_calls) |tc| {
                    allocator.free(tc.name);
                    allocator.free(tc.arguments);
                }
                allocator.free(tool_calls);
            }

            var tc_perf_buf: [160]u8 = undefined;
            const tc_perf = formatPerfBracket(&tc_perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
            log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [tool_calls: {d}]\n", .{
                result.prompt_tokens, result.completion_tokens, elapsed_ms, tc_perf, tool_calls.len,
            });

            // Build tool_calls JSON array
            var tc_buf = std.ArrayList(u8).empty;
            defer tc_buf.deinit(allocator);
            try tc_buf.appendSlice(allocator, "[");
            for (tool_calls, 0..) |tc, i| {
                if (i > 0) try tc_buf.appendSlice(allocator, ",");
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ nowMs(stream.io), i });
                defer allocator.free(tc_id);
                const escaped_name = try jsonEscape(allocator, tc.name);
                defer allocator.free(escaped_name);
                const escaped_args = try jsonEscape(allocator, tc.arguments);
                defer allocator.free(escaped_args);
                const tc_json = try std.fmt.allocPrint(allocator,
                    \\{{"id":"{s}","type":"function","function":{{"name":{s},"arguments":{s}}}}}
                , .{ tc_id, escaped_name, escaped_args });
                defer allocator.free(tc_json);
                try tc_buf.appendSlice(allocator, tc_json);
            }
            try tc_buf.appendSlice(allocator, "]");

            // Reasoning is delivered whenever the model produced it — the
            // request's thinking flag shaped the prompt, not the delivery.
            const tc_think_split = think_tags.splitThinkBlock(final_text, true, opens_think);
            var tc_reasoning_json: []const u8 = "";
            var tc_reasoning_allocated = false;
            {
                if (deliveredReasoning(budget_truncated_reasoning, tc_think_split.reasoning_content)) |reasoning| {
                    const escaped_reasoning = try jsonEscape(allocator, reasoning);
                    tc_reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
                    allocator.free(escaped_reasoning);
                    tc_reasoning_allocated = true;
                }
            }
            defer if (tc_reasoning_allocated) allocator.free(tc_reasoning_json);

            // The same pre-markup text the stream flushes as a text delta.
            const tc_preamble = visibleToolPreamble(tc_think_split.content);
            const tc_content_esc = jsonEscapeOrEmpty(allocator, tc_preamble);
            defer if (tc_content_esc.owned) allocator.free(tc_content_esc.slice);
            const tc_content_json: []const u8 = if (tc_preamble.len > 0) tc_content_esc.slice else "null";

            const tc_timings = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
            defer allocator.free(tc_timings);
            const tc_timings_field = if (tc_timings.len > 0)
                try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{tc_timings})
            else
                try allocator.alloc(u8, 0);
            defer allocator.free(tc_timings_field);

            const tc_details = try formatReasoningDetails(allocator, tok, deliveredReasoning(budget_truncated_reasoning, tc_think_split.reasoning_content) orelse "");
            defer allocator.free(tc_details);
            const tc_usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, tc_details);
            defer allocator.free(tc_usage_obj);

            // finish_reason always carries finish_details beside it.
            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s},"tool_calls":{s}}},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
            , .{
                nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io)),
                nowSecs(stream.io),
                model_name,
                tc_content_json,
                tc_reasoning_json,
                tc_buf.items,
                toolCallFinishReason(finish_reason),
                finishDetailsField(toolCallFinishReason(finish_reason), result.finish_details),
                tc_usage_obj,
                tc_timings_field,
            });
            defer allocator.free(response);
            try sendResponse(stream, "200 OK", "application/json", response);
            return;
        }
    }

    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    // Split thinking content from response. Always the think-capable split:
    // whatever reasoning was generated ships as reasoning_content, never
    // stripped (tokens we discarded still counted against tok/s).
    var routed: ?rp_mod.Delivery = null;
    defer if (routed) |*d| d.deinit(allocator);
    const think_split = try splitConstrainedResponse(allocator, &routed, sampling, final_text, result.constraint_payload_byte, false, opens_think);
    const content_text = think_split.content;

    const escaped = jsonEscapeOrEmpty(allocator, content_text);
    const escaped_text = escaped.slice;
    defer if (escaped.owned) allocator.free(escaped.slice);

    // Build logprobs JSON if requested. The array describes `message.content`,
    // so it stops at the tokens that survived the think split — see
    // `contentTokenRange`.
    var logprobs_json: []const u8 = "null";
    var logprobs_allocated = false;
    if (result.logprobs) |lps| {
        const n = @min(result.token_ids.len, lps.len);
        const r = contentTokenRange(allocator, tok, result.token_ids[0..n], final_text, content_text);
        const a = @min(r.start, n);
        const b = @min(@max(r.end, a), n);
        logprobs_json = try formatLogprobsObject(allocator, tok, result.token_ids[a..b], lps[a..b]);
        logprobs_allocated = true;
    }
    defer if (logprobs_allocated) allocator.free(logprobs_json);

    // Build reasoning_content whenever reasoning exists — delivery does not
    // key on the request's thinking flag.
    var reasoning_json: []const u8 = "";
    var reasoning_allocated = false;
    const reasoning_text = deliveredReasoning(budget_truncated_reasoning, think_split.reasoning_content);
    const usage_details_json = try formatReasoningDetails(allocator, tok, reasoning_text orelse "");
    defer allocator.free(usage_details_json);
    if (reasoning_text) |reasoning| {
        const escaped_reasoning = try jsonEscape(allocator, reasoning);
        defer allocator.free(escaped_reasoning);
        reasoning_json = try std.fmt.allocPrint(allocator, ",\"reasoning_content\":{s}", .{escaped_reasoning});
        reasoning_allocated = true;
    }
    defer if (reasoning_allocated) allocator.free(reasoning_json);

    const timings_obj = try formatTimingsObject(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    const timings_field = if (timings_obj.len > 0)
        try std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{timings_obj})
    else
        try allocator.alloc(u8, 0);
    defer allocator.free(timings_field);

    const usage_obj = try formatChatUsage(allocator, result.prompt_tokens, result.completion_tokens, result.cached_tokens, usage_details_json);
    defer allocator.free(usage_obj);

    // finish_reason always carries finish_details beside it.
    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}}},"logprobs":{s},"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
    , .{
        nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io)),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        reasoning_json,
        logprobs_json,
        finish_reason,
        finishDetailsField(finish_reason, result.finish_details),
        usage_obj,
        timings_field,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

/// Token-stream adapter that drives the streaming SSE state machine the same
/// way regardless of whether speculative decoding (drafter / MTP) is on. Each
/// `next()` call yields exactly one token (or null on EOS / max_tokens), so
/// the per-token state machine in `handleStreamingGeneration` /
/// `handleAnthropicStreaming` / `handleResponses` does not need to know
/// anything about multi-token batches that the drafter and MTP emit per step.
///
/// EOS-in-batch behavior: the stop token is NOT yielded — the loop just
/// terminates. This keeps tokens leaking past EOS impossible regardless of
/// where in an N-token batch the EOS lands.
///
/// `pending_buf` is a heap-owned pending buffer (the drafter yields up to
/// `block_size` tokens per step). Caller must `deinit`.
pub const StreamMode = enum { regular, drafter, mtp };
/// Source of streamed tokens. Either a legacy Generator (single-slot mlx on
/// the calling thread) or a Scheduler Slot (mlx on the scheduler's inference
/// thread, this thread reads via waitNext). `next()` yields one token id at
/// a time regardless of source. Post-generation stats are surfaced via the
/// `prompt_tokens` / `completion_tokens` / `finish_reason` fields populated
/// by `finalize()`.
pub const StreamingTokenStream = struct {
    /// Active source. Exactly one of `gen`/`slot` is set per stream.
    gen: ?*Generator = null,
    slot: ?*scheduler_mod.Slot = null,
    mode: StreamMode,
    eos_token_ids: []const u32,
    /// Pending tokens from a multi-token speculative step (legacy path only).
    /// Drained one at a time before the next call to nextDflash / nextMtp.
    pending_buf: std.ArrayList(u32) = .empty,
    pending_idx: usize = 0,
    finished: bool = false,

    /// Stats populated by `finalize()` after the stream ends. Consumers read
    /// these instead of touching the underlying gen/slot directly so the same
    /// post-generation code works for both legacy and scheduler paths.
    prompt_tokens: u32 = 0,
    cached_tokens: u32 = 0,
    completion_tokens: u32 = 0,
    finish_reason: []const u8 = "stop",
    /// Set only by the degenerate-tail guard (scheduler path). Rides the
    /// FINAL chunk beside `finish_reason`; the tokens themselves are already
    /// gone — a delta cannot be retracted, so streaming gets the signal, not
    /// the trim.
    finish_details: ?[]const u8 = null,
    /// Wall-clock ns for prefill / decode (scheduler path only; the legacy
    /// Generator path leaves these at 0, in which case the server omits the
    /// `timings` block from the usage chunk).
    prefill_ns: u64 = 0,
    decode_ns: u64 = 0,

    /// Phase A2-streaming: build a token stream backed by a scheduler Slot.
    /// `mode` is recorded for telemetry but doesn't drive next()'s dispatch
    /// (the scheduler's inference thread already routed drafter/MTP inside
    /// runSingleDecodeTick — this side just drains the resulting token ring).
    pub fn initFromSlot(slot: *scheduler_mod.Slot, mode: StreamMode, eos: []const u32) StreamingTokenStream {
        return .{
            .slot = slot,
            .mode = mode,
            .eos_token_ids = eos,
        };
    }

    pub fn deinit(self: *StreamingTokenStream, allocator: std.mem.Allocator) void {
        self.pending_buf.deinit(allocator);
    }

    /// Snapshot prompt/completion tokens + finish reason from the underlying
    /// source so callers can read them after the loop exits without having
    /// to know which source was active. Safe to call multiple times; idempotent.
    pub fn finalize(self: *StreamingTokenStream) void {
        if (self.slot) |s| {
            self.prompt_tokens = s.prompt_tokens;
            self.cached_tokens = s.cached_tokens;
            self.completion_tokens = s.completion_tokens;
            self.finish_reason = s.finish_reason;
            self.finish_details = s.finish_details;
            self.prefill_ns = s.prefill_ns;
            self.decode_ns = s.decode_ns;
        } else if (self.gen) |g| {
            self.prompt_tokens = g.prompt_tokens;
            self.completion_tokens = g.completion_tokens;
            self.finish_reason = g.finish_reason;
        }
    }

    const NextOrIdle = union(enum) { token: u32, done, idle };

    /// `next` with an idle timeout (scheduler path only): returns `.idle`
    /// after `timeout_ms` with no progress so the caller can poll the peer
    /// socket and emit SSE keepalives during long prefills. The local-
    /// generator path computes synchronously and never idles.
    pub fn nextOrIdle(self: *StreamingTokenStream, allocator: std.mem.Allocator, timeout_ms: i64) !NextOrIdle {
        if (self.slot) |s| {
            if (self.finished) return .done;
            const nr = s.waitNextTimeout(timeout_ms) orelse return .idle;
            switch (nr) {
                .token => |t| return .{ .token = t },
                .done => {
                    self.finished = true;
                    return .done;
                },
                .err => return slotFailure(s),
            }
        }
        if (try self.next(allocator)) |t| return .{ .token = t };
        return .done;
    }

    /// Yield the next decoded token id, or null if generation is complete.
    /// Mirrors the contract of `Generator.next` for the regular path.
    fn next(self: *StreamingTokenStream, allocator: std.mem.Allocator) !?u32 {
        // Scheduler path: drain the slot's output ring one token at a time.
        // The inference thread already handled regular/drafter/MTP dispatch
        // via runSingleDecodeTick; we just consume what it pushed.
        if (self.slot) |s| {
            if (self.finished) return null;
            switch (s.waitNext()) {
                .token => |t| return t,
                .done => {
                    self.finished = true;
                    return null;
                },
                .err => return slotFailure(s),
            }
        }
        // Drain any pending tokens from a previous speculative step FIRST,
        // before honoring `self.finished`. The drafter/MTP branches set
        // `self.finished = true` mid-batch when an EOS lands at index > 0;
        // tokens *before* that EOS were pushed to pending_buf and must still
        // flush before we terminate. Skipping the drain here would silently
        // truncate the last few output tokens (observed on Gemma 4 drafter
        // with `print(sum_two(10, 20))` losing its trailing `))`).
        if (self.pending_idx < self.pending_buf.items.len) {
            const tok = self.pending_buf.items[self.pending_idx];
            self.pending_idx += 1;
            return tok;
        }
        // Reset the pending buffer once drained so the speculative path can
        // refill it.
        if (self.pending_idx > 0) {
            self.pending_buf.clearRetainingCapacity();
            self.pending_idx = 0;
        }

        if (self.finished) return null;

        const gen = self.gen.?;
        switch (self.mode) {
            .regular => return gen.next(allocator),
            .mtp => {
                // Mirror the drafter branch — `nextMtp` returns the same
                // `{tokens, accepted_tokens}` shape.
                const r = (try gen.nextMtp(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
            .drafter => {
                // Mirror the MTP branch — `nextDflash` returns the same
                // `{tokens, accepted_tokens}` shape (full accept yields
                // `block_size` tokens, partial accept yields `1+j`). Walk the
                // batch, stop at the first EOS, push the rest into pending.
                const r = (try gen.nextDflash(allocator)) orelse return null;
                defer allocator.free(r.tokens);
                if (r.tokens.len == 0) {
                    self.finished = true;
                    return null;
                }
                var first_idx: usize = 0;
                while (first_idx < r.tokens.len and generate_mod.isEosId(r.tokens[first_idx], self.eos_token_ids)) : (first_idx += 1) {}
                if (first_idx >= r.tokens.len) {
                    gen.done = true;
                    gen.finish_reason = "stop";
                    self.finished = true;
                    return null;
                }
                const first_tok = r.tokens[first_idx];
                var i: usize = first_idx + 1;
                while (i < r.tokens.len) : (i += 1) {
                    if (generate_mod.isEosId(r.tokens[i], self.eos_token_ids)) {
                        gen.done = true;
                        gen.finish_reason = "stop";
                        self.finished = true;
                        break;
                    }
                    try self.pending_buf.append(allocator, r.tokens[i]);
                }
                return first_tok;
            },
        }
    }
};

pub const RequestSpec = struct { use_mtp: bool, use_drafter: bool };

/// The ONE request-level spec dispatch, shared by every surface x stream /
/// non-stream. Priority: DFlash > MTP, mirroring
/// `scheduler.specInitWiring`. A loaded DFlash sidecar is an explicit choice
/// (`--drafter` or the pack's own `drafter/`) while the MTP head ships with
/// the checkpoint, so the sidecar wins; `--no-drafter` / `enable_drafter:
/// false` hand the round back to MTP. logprobs and a grammar constraint disable every mode; a repeat or presence
/// penalty disables only the sidecar and the drafter, since MTP shifts its verify rows for it.
pub fn requestSpecModes(
    enable_drafter: bool,
    enable_mtp: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    sampling: sampler.SamplingParams,
    logprobs_n: u32,
) RequestSpec {
    const spec_ok = !mtp_plan.mtpUnsupported(sampling, logprobs_n);
    const sidecar = spec_ok and !sampler.penaltyActive(sampling) and enable_drafter and dflash_loaded;
    const use_dflash = sidecar and dflash_loaded;
    const use_mtp = !use_dflash and spec_ok and enable_mtp and mtp_loaded;
    const use_drafter = use_dflash or (!use_mtp and sidecar);
    return .{
        .use_mtp = use_mtp,
        .use_drafter = use_drafter,
    };
}

/// A request that asked for MTP and runs without it says why; the boot line says MTP is on.
pub fn noteMtpRefused(enable_mtp: bool, mtp_loaded: bool, sampling: sampler.SamplingParams, logprobs_n: u32) void {
    if (!enable_mtp or !mtp_loaded) return;
    if (mtp_plan.mtpRefusal(sampling, logprobs_n)) |why| log.info("  mtp=disabled ({s})\n", .{why});
}

pub fn pickStreamMode(
    enable_drafter: bool,
    enable_mtp: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    sampling: sampler.SamplingParams,
    logprobs_n: u32,
) StreamMode {
    const r = requestSpecModes(enable_drafter, enable_mtp, dflash_loaded, mtp_loaded, sampling, logprobs_n);
    if (r.use_mtp) return .mtp;
    if (r.use_drafter) return .drafter;
    return .regular;
}

test "requestSpecModes: a loaded DFlash sidecar outranks the checkpoint's MTP head" {
    const plain = sampler.SamplingParams{};
    // dflash + mtp both loaded and enabled -> dflash rides the drafter arm.
    var r = requestSpecModes(true, true, true, true, plain, 0);
    try std.testing.expect(r.use_drafter and !r.use_mtp);
    try std.testing.expectEqual(StreamMode.drafter, pickStreamMode(true, true, true, true, plain, 0));
    // No sidecar loaded -> MTP keeps its rank.
    r = requestSpecModes(true, true, false, true, plain, 0);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, false, true, plain, 0));
    // enable_drafter:false hands the round back to MTP.
    r = requestSpecModes(false, true, true, true, plain, 0);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    // logprobs / grammar disable every spec mode, dflash included.
    r = requestSpecModes(true, true, true, true, plain, 3);
    try std.testing.expect(!r.use_mtp and !r.use_drafter);
    var dummy: generate_mod.Constraint = .{ .grammar = undefined, .token_bytes = undefined, .mask_buf = &.{} };
    r = requestSpecModes(true, true, true, true, .{ .constraint = &dummy }, 0);
    try std.testing.expect(!r.use_mtp and !r.use_drafter);
    // Nothing loaded -> regular decode.
    r = requestSpecModes(true, true, false, false, plain, 0);
    try std.testing.expect(!r.use_mtp and !r.use_drafter);
    try std.testing.expectEqual(StreamMode.regular, pickStreamMode(true, true, false, false, plain, 0));
}

pub fn handleStreamingGeneration(
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
    include_usage: bool,
    has_tools: bool,
    /// OpenAI-shape tools JSON (for bare-args tool-call inference); null when
    /// the request defined no tools.
    tools_json: ?[]const u8,
    /// false = client set parallel_tool_calls:false (Anthropic:
    /// tool_choice.disable_parallel_tool_use) — at most one call per response.
    allow_parallel_tools: bool,
    logprobs_n: u32,
    enable_thinking: bool,
    reasoning_budget: i32,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    deferred_media: ?*vision_queue.DeferredMedia,
    media_chain: []const prefix_cache_mod.MediaSpan,
    cache_key: u64,
    mrope: MropeData,
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?transformer_mod.KVQuantConfig,
    /// Iteration 1: tokenize_ns measured by the request handler before
    /// dispatching here. Surfaced via `timings.tokenize_ms` on the final
    /// usage SSE chunk so streaming clients see the same metric as
    /// non-streaming.
    tokenize_ns: u64,
) !void {
    // Vision array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer so
    // the early-return defer is a no-op.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    const chat_id = nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io));

    // Template-opened think block (Qwen 3.5/3.6): unclosed buffered output is
    // reasoning, never content (mirrors the non-streaming split policy).
    // `prompt_opened_think` drops the `enable_thinking` conjunct on purpose:
    // whether the prompt ends inside a think block is a property of the
    // RENDERED BYTES. LFM2.5 pre-opens `<think>` unconditionally, so with
    // thinking off its reasoning is tag-free prose the gate flushed as the
    // visible answer (live 2026-08-04). Families that gate the opener on
    // enable_thinking render the closed signature and stay false here.
    const prompt_opened_think = promptOpensThink(allocator, tok, prompt_ids);
    const opens_think = prompt_opened_think;

    // Pick the speculative-decoding mode (regular / drafter / MTP). The
    // per-token state machine below is driven by `StreamingTokenStream`,
    // which feeds `next` (regular), `nextPld` (1..1+draft_len tokens/step),
    // or `nextDrafter` (1..block_size tokens/step) through the same
    // one-token-at-a-time interface.
    const stream_mode = pickStreamMode(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, logprobs_n);
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});
    noteMtpRefused(enable_mtp, mtpCapable(lm), sampling, logprobs_n);

    // Scheduler's inference thread runs prefill + per-tick decode (regular
    // / PLD / drafter) and pushes generated tokens into the slot's output
    // ring; this thread reads via `slot.waitNext` through the
    // `StreamingTokenStream.initFromSlot` adapter.
    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| server_mod.global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_s = ve_local;
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
        .logprobs_n = logprobs_n,
        .deferred_media = slot_ve_s,
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

    // Send SSE headers (no Content-Length — we stream until done)
    try sendSseHeaders(stream, "chat.completions", SSE_ALLOW_HEADERS_DEFAULT);

    // Logprobs ride the chunks beside their deltas. Requesting them already
    // forced this stream off every speculative path (see `pickStreamMode`), so
    // the cost is paid whether or not they are delivered — dropping them was
    // the worst of both.
    var lps = StreamLogprobs{
        .allocator = allocator,
        .tok = tok,
        .slot = slot_handle,
        .enabled = logprobs_n > 0,
    };
    defer lps.deinit();

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null, null, .{ .logprobs_json = try lps.take() });

    // Buffer for stop sequence and tool call detection
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
    // When tools are present, buffer individual token texts for deferred streaming
    var token_texts = std.ArrayList([]const u8).empty;
    defer {
        for (token_texts.items) |t| allocator.free(t);
        token_texts.deinit(allocator);
    }
    var stopped = false;
    var client_gone = false;

    // Thinking state for real-time streaming of reasoning_content vs content
    // Supports both <think>...</think> and Gemma 4's <|channel>thought\n...<channel|>
    // Starts true when thinking is enabled (model outputs <think> first) OR
    // the prompt itself opened a think block: generated reasoning is always
    // DELIVERED as reasoning_content, never paid-for-and-dropped. Thinking-off
    // is enforced prompt-side (think_tags.noThinkTailSuffix commits the channel).
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
    var think_closed = false; // a complete think block was already split+emitted this stream
    // Leading whitespace is suppressed until the first visible byte, so the
    // stream reaches the same content bytes as splitThinkBlock's own
    // trimStart (think_tags.streamContentLead).
    var content_started = false;
    var think_buf = std.ArrayList(u8).empty; // buffer to detect close tag across token boundaries
    var thought_shipped = false; // this thought has streamed a delta (`think_tags.openThoughtFlush`)
    defer think_buf.deinit(allocator);
    var think_close_tag: []const u8 = "</think>"; // will be updated if Gemma 4 format detected
    var skipped_think_open = false; // track if we've skipped the initial think tag
    // Positive evidence the model itself emitted a think OPENER. Distinct
    // from `skipped_think_open`, whose else-branch also fires for "no known
    // opener — the template must have injected one", which is the very case
    // that has to be told apart at the end-of-stream flush below.
    var saw_think_open = false;
    // Bytes of THIS turn's reasoning already streamed. The tools path emits the
    // thought incrementally now, so every later emit site sends the remainder.
    var reasoning_streamed: usize = 0;
    // Tokenize concatenated delivered text; SSE boundaries are not token boundaries.
    var delivered_reasoning = std.ArrayList(u8).empty;
    defer delivered_reasoning.deinit(allocator);
    var reasoning_tokens_sent: usize = 0; // reasoning deltas actually emitted (the budget is counted in these)
    var think_tokens: i32 = 0; // count of tokens generated in think block
    var budget_exhausted = false; // true when reasoning budget hit
    var stop_gate = stop_seq_mod.Gate{ .stops = stop_sequences };
    defer stop_gate.deinit(allocator);
    var tail_flush = false; // the generation ended: this turn delivers the gate's held tail

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, MTP, or drafter.
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
        // The gate delivers the bytes before a match and holds back any tail that could still become
        // one, or a character the token left unfinished.
        const token_text = if (tail_flush) try stop_gate.finish(allocator) else blk: {
            try lps.note(token_id);
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

        // Accumulate for tool call detection
        if (gated_stream) {
            try text_buf.appendSlice(allocator, token_text);
        }

        if (delivery) |*d| {
            defer allocator.free(token_text);
            try d.feed(allocator, token_text);
            if (include_usage) try delivered_reasoning.appendSlice(allocator, d.reasoning.items);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
            try beatStreamKeepalive(stream, .sse_comment);
            continue;
        }

        if (gated_stream) {
            // Stream tokens until we detect a tool call pattern starting, then buffer.
            // Detection rules live in `think_tags.streamShouldBufferForTools` — it
            // covers the full `<tool…>` family (including bare DSV4 `<tool>`),
            // Gemma 4 `<|tool_call`, raw JSON, plus partial-prefix growth from
            // a single `<` all the way to `<|tool_cal`. Without the partial
            // coverage, a `<tool` single-BPE-token leaks as visible content
            // (the bug surfaced on DSV4 where the model emits `<tool` then
            // gets stuck looping the partial).
            try token_texts.append(allocator, token_text);
            // text_buf already updated above

            const buf = text_buf.items;
            const undelivered = buf[text_delivered..];
            const maybe_tool = think_tags.streamShouldBufferForTools(buf);

            if (!maybe_tool) {
                // No tool call pattern — ask the shared gate (think_tags.streamThinkGate,
                // also used by /v1/messages) whether the buffer is thinking that
                // must be held, a completed think block to split, or visible
                // prose to flush. Hermetically pinned per recorded model family
                // by the format corpus streaming-gate test.
                switch (think_tags.streamThinkGateScan(buf, enable_thinking, think_closed, prompt_opened_think, &think_scan)) {
                    .hold_thinking => {
                        // Stream the thought AS IT ARRIVES instead of holding the
                        // whole block. Safe here by construction:
                        // `streamShouldBufferForTools` returned false just above,
                        // so the buffer carries no tool markup and no partial
                        // marker at its tail — these bytes are reasoning and
                        // nothing else. Without it a tools request shows NOTHING
                        // until `</think>` (measured 4.4 s on a 2.6B, and it grows
                        // with the thought) while the no-tools path is already
                        // streaming at 0.05 s.
                        //
                        // A reasoning BUDGET is a cap, not a reason to hide the
                        // thought: you never exceed a cap you STOP emitting at.
                        // This was gated on `reasoning_budget < 0`, so a capped
                        // tools request showed nothing for the whole generation
                        // and got one dump at the end — a capped agent session
                        // looked frozen (live 2026-08-14, pi on Qwen3.8-27B).
                        // Each loop iteration is one model token, so an
                        // iteration that produced fresh reasoning IS one
                        // reasoning token; at the cap the latch stops every
                        // later emitter from shipping the remainder.
                        if (!budget_exhausted) {
                            const so_far = think_tags.splitThinkBlock(buf, true, opens_think);
                            if (so_far.reasoning_content) |rc| {
                                const settled = think_tags.settledReasoning(rc);
                                if (think_tags.unstreamedReasoning(settled, reasoning_streamed)) |fresh| {
                                    if (include_usage) try delivered_reasoning.appendSlice(allocator, fresh);
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{ .logprobs_json = null });
                                    reasoning_streamed = settled.len;
                                    reasoning_tokens_sent += 1;
                                    if (reasoning_budget >= 0 and reasoning_tokens_sent >= @as(usize, @intCast(reasoning_budget))) {
                                        budget_exhausted = true;
                                        log.info("  reasoning budget exhausted ({d}/{d} tokens, streamed)\n", .{ reasoning_tokens_sent, reasoning_budget });
                                    }
                                }
                            }
                        }
                    },
                    .split_think => {
                        // Complete thinking block — split into reasoning + content
                        const split = think_tags.splitThinkBlock(undelivered, true, opens_think);
                        // The thought's tokens never reach the client, so their
                        // entries must not ride the content chunk below — the
                        // reasoning emitters above deliberately do not drain.
                        // Empty content means the block closed with nothing
                        // after it: everything buffered so far was reasoning,
                        // and leaving it pending hands it to the NEXT chunk
                        // (measured: 42 entries on a 1-char delta).
                        if (split.content.len > 0) lps.skipToContent(undelivered, split.content, stop_gate.carry_len) else lps.dropPending();
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        text_delivered = 0;
                        think_scan.reset();
                        think_closed = true;
                        // Past the cap the tail is exactly what the budget
                        // exists to withhold. This arm applied no budget at all,
                        // so a capped request received the WHOLE thought the
                        // moment the block closed.
                        if (!budget_exhausted) {
                            if (split.reasoning_content) |rc| {
                                if (think_tags.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    if (include_usage) try delivered_reasoning.appendSlice(allocator, fresh);
                                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = fresh }, null, null, null, .{ .logprobs_json = null });
                                }
                            }
                        }
                        // Block done and `text_buf` was just cleared — a thought
                        // re-opened later in the turn starts counting from zero.
                        reasoning_streamed = 0;
                        if (split.content.len > 0) {
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                    },
                    .flush_text => {
                        for (token_texts.items) |tt| {
                            defer allocator.free(tt);
                            // Skip bare channel/think tags that leak without a full block
                            if (think_tags.isChannelMarkerToken(tt)) {
                                continue;
                            }
                            const vis = think_tags.streamContentLead(tt, content_started);
                            // Suppressed lead: nothing pending describes text the
                            // client received, so retire it rather than letting it
                            // ride the next chunk (logprobs.content ↔ content).
                            if (vis.len == 0) {
                                lps.dropPending();
                                continue;
                            }
                            content_started = true;
                            try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis }, null, null, null, .{ .logprobs_json = try lps.take() });
                        }
                        token_texts.clearRetainingCapacity();
                        text_delivered = text_buf.items.len;
                    },
                }
            }
            // Otherwise keep buffering — tool call may be in progress
        } else if (in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think>
            // detection. Also covers thinking-off + template-opened (LFM2.5
            // renders `…assistant\n<think>` unconditionally, live 2026-08-04):
            // the block used to be buffered and DROPPED there; now it streams
            // as reasoning_content like any other thought — generated tokens
            // are always delivered.
            defer allocator.free(token_text);
            try think_buf.appendSlice(allocator, token_text);
            think_tokens += 1;

            // Skip the initial think tag prefix (<think> or <|channel>thought\n).
            // Many templates (e.g. Qwen 3.5/3.6, some Gemma 4 variants) pre-inject
            // the opener into the prompt so the model's first tokens are already
            // INSIDE the thinking block — no opener appears in the streamed text.
            if (!skipped_think_open and (think_buf.items.len >= 7 or !think_tags.thinkOpenerPossible(think_buf.items))) {
                if (think_tags.thinkOpenTagLenAt(think_buf.items)) |olen| {
                    // Remove the opener and any leading newline.
                    saw_think_open = true;
                    var skip: usize = olen;
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len >= 17 and std.mem.startsWith(u8, think_buf.items, "<|channel>thought")) {
                    // Gemma 4 think format — switch close tag
                    saw_think_open = true;
                    think_close_tag = "<channel|>";
                    var skip: usize = 17; // len of "<|channel>thought"
                    while (skip < think_buf.items.len and think_buf.items[skip] == '\n') skip += 1;
                    const remaining = try allocator.dupe(u8, think_buf.items[skip..]);
                    think_buf.clearAndFree(allocator);
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                    skipped_think_open = true;
                } else if (think_buf.items.len < 17 and std.mem.startsWith(u8, "<|channel>thought", think_buf.items)) {
                    // Buffer is still a partial prefix of `<|channel>thought` —
                    // wait for more tokens before deciding.
                } else if (think_buf.items.len < 32 and think_tags.endsWithPartialThinkOpen(think_buf.items)) {
                    // Still a growing opener (`…<thi`) —
                    // wait before deciding, or the raw tag leaks as reasoning.
                } else {
                    // Not a known opener — template already injected one.
                    // Stay inside the think block; close tag is detected dynamically below.
                    skipped_think_open = true;
                }
            }

            // A spent budget caps DELIVERY only: the tail stays withheld reasoning until the close tag.
            if (!budget_exhausted and reasoning_budget >= 0 and think_tokens >= reasoning_budget and skipped_think_open) {
                budget_exhausted = true;
                log.info("  reasoning budget exhausted ({d}/{d} tokens)\n", .{ think_tokens, reasoning_budget });
            }

            // Check for the close tag — accept whichever appears first.
            // Models with templates that pre-inject the opener (Qwen 3.5/3.6,
            // some Gemma 4 variants) don't reveal which
            // format they use until the close tag arrives, so look for both
            // families.
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
                if (last.len > 0 and !budget_exhausted) {
                    if (include_usage) try delivered_reasoning.appendSlice(allocator, last);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = last }, null, null, null, .{ .logprobs_json = null });
                }
                const after = m.pos + m.len;
                var content_after = std.mem.trimStart(u8, think_buf.items[after..], "\n ");
                // Strip Gemma 4 content channel tag: <|channel>\n or <|channel>
                if (std.mem.startsWith(u8, content_after, "<|channel>\n")) {
                    content_after = content_after[11..];
                } else if (std.mem.startsWith(u8, content_after, "<|channel>")) {
                    content_after = content_after[10..];
                }
                content_after = std.mem.trimStart(u8, content_after, "\n ");
                if (content_after.len > 0) {
                    // The reasoning above this point never reaches the client,
                    // so its entries stop here rather than riding the answer.
                    lps.skipToContent(think_buf.items, content_after, stop_gate.carry_len);
                    const vis_content_after = think_tags.streamContentLead(content_after, content_started);
                    if (vis_content_after.len > 0) {
                        content_started = true;
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_content_after }, null, null, null, .{ .logprobs_json = try lps.take() });
                    }
                } else {
                    lps.dropPending();
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
                    if (flush.ship > 0 and !budget_exhausted) {
                        if (include_usage) try delivered_reasoning.appendSlice(allocator, think_buf.items[flush.skip..safe_len]);
                        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = think_buf.items[flush.skip..safe_len] }, null, null, null, .{ .logprobs_json = null });
                    }
                    if (flush.ship > 0) thought_shipped = true;
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        } else {
            defer allocator.free(token_text);
            // Skip Gemma 4 channel tags that leak after thinking blocks
            // A think block the MODEL opens: seeded like a template-injected one.
            // Checked before the marker skip, which would swallow a bare `<think>`.
            if (!content_started) if (think_tags.modelThinkOpener(channel_armed, token_text)) |opener| {
                channel_armed = false;
                in_think_block = true;
                saw_think_open = true;
                try think_buf.appendSlice(allocator, opener);
                continue;
            };
            if (think_tags.isChannelMarkerToken(token_text)) {
                channel_armed = !content_started and std.mem.eql(u8, token_text, "<|channel>");
                continue;
            }
            channel_armed = false;
            const vis_token_text = think_tags.streamContentLead(token_text, content_started);
            if (vis_token_text.len == 0) {
                lps.dropPending();
            } else {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_token_text }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }

        // Every branch above may have written NOTHING (tool-call detection and
        // unclosed thinking blocks buffer silently, sometimes for minutes on a
        // one-shot whole-file `write_file`). Tokens flowing is not bytes
        // flowing — without this the client's idle-body timeout kills the
        // stream mid-generation.
        beatStreamKeepalive(stream, .sse_comment) catch {
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
            if (include_usage) try delivered_reasoning.appendSlice(allocator, d.reasoning.items);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, &lps, d);
        }
    }

    // Flush any remaining think buffer
    if (!client_gone and think_buf.items.len > 0) {
        // No close tag arrived. Reasoning ONLY with positive evidence a block
        // was open — the prompt injected the opener, or the model emitted one.
        // `in_think_block` alone is not that evidence: it STARTS true whenever
        // thinking was requested, so a Gemma turn answered directly (its
        // template renders a bare `<|turn>model` and lets the model decide)
        // shipped its whole answer as reasoning with EMPTY content, while
        // non-streaming returned it correctly (live 2026-08-04).
        if (think_tags.streamTailIsReasoning(in_think_block, prompt_opened_think, saw_think_open)) {
            const tail = think_tags.cutThoughtDelta(think_buf.items, skipped_think_open, thought_shipped);
            if (!budget_exhausted and tail.len > 0) {
                if (include_usage) try delivered_reasoning.appendSlice(allocator, tail);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = tail }, null, null, null, .{ .logprobs_json = null });
            }
        } else {
            const vis_tail = think_tags.streamContentLead(think_buf.items, content_started);
            if (vis_tail.len > 0) {
                content_started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_tail }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
        }
    }

    // After generation: capture stats from whichever source was active
    // (Generator on legacy, Slot on scheduler).
    ts.finalize();

    // Check for tool calls in accumulated text
    var finish_reason: []const u8 = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    if (gated_stream and !client_gone) {
        log.debug("  checking {d} bytes of streamed text for tool calls\n", .{text_buf.items.len});
        if (log.isDebug() and text_buf.items.len > 0) {
            log.debug("  raw generated text before tool parse ({d}b): {s}\n", .{ text_buf.items.len, text_buf.items[0..@min(text_buf.items.len, 4000)] });
            // Corpus-harvest aid: the inline dump caps at 4KB, useless for
            // >30KB mega-tool-calls. Set SUSHI_RAW_DUMP_FILE=<abs path> to
            // APPEND the FULL pre-parse buffer of every tools request, framed so
            // a harvester can slice each record exactly (the text is arbitrary
            // bytes, so the byte count — not a delimiter — defines the record).
            if (std.c.getenv("SUSHI_RAW_DUMP_FILE")) |dump_path| {
                appendRawToolDump(std.mem.span(dump_path), tools_json, text_buf.items);
            }
        }
        // Merge re-opened mid-text thought channels into the leading block so
        // the split/parse below never leaks raw tags (Gemma 12B tail behavior).
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

            // Emit reasoning_content before tool calls — whatever the model
            // generated is delivered, thinking flag or not.
            {
                const think_split = think_tags.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (think_tags.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                    defer if (capped.owned) allocator.free(capped.text);
                    if (include_usage) try delivered_reasoning.appendSlice(allocator, capped.text);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = capped.text }, null, null, null, .{ .logprobs_json = null });
                }
            }

            // Emit tool call deltas in OpenAI streaming format
            for (tool_calls, 0..) |tc, i| {
                const tc_id = try std.fmt.allocPrint(allocator, "call_{d}_{d}", .{ chat_id, i });
                defer allocator.free(tc_id);

                // Escape the full arguments string for embedding in JSON
                // Strip outer quotes from jsonEscape result (it wraps in "...")
                // First delta: name + id + full arguments (clients accumulate these)
                const first_delta = try formatChatStreamToolCall(allocator, i, tc_id, tc.name, tc.arguments);
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null, null, .{ .logprobs_json = try lps.take() });
            }
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            // No tool calls found — flush buffered tokens, splitting thinking
            // from content. Runs regardless of the request's thinking flag:
            // reasoning the model produced ships as reasoning_content, and
            // split.content already passes trimLeakedToolMarkup, so unparsed
            // wreckage that held the buffer (markers the tokenizer split
            // across tokens — live 2026-08-01, DSV4 `<` then `｜DSML｜`) is
            // still cut once, at the end.
            {
                // Concatenate all buffered tokens and split thinking from content
                var full_text = std.ArrayList(u8).empty;
                defer full_text.deinit(allocator);
                for (token_texts.items) |t| {
                    try full_text.appendSlice(allocator, t);
                }
                const flush_norm = try think_tags.normalizeEmbeddedThinkBlocks(allocator, full_text.items);
                defer if (flush_norm) |n| allocator.free(n);
                const flush_text: []const u8 = flush_norm orelse full_text.items;
                const think_split = think_tags.splitThinkBlock(flush_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (think_tags.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                    defer if (capped.owned) allocator.free(capped.text);
                    if (include_usage) try delivered_reasoning.appendSlice(allocator, capped.text);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = capped.text }, null, null, null, .{ .logprobs_json = null });
                }
                if (think_split.content.len > 0) {
                    // Locate against the RAW concatenation: it is always a true
                    // suffix of the generation, where `flush_text` may be a
                    // normalized rewrite. If the content cannot be found there
                    // (normalization moved it) nothing is skipped, which is the
                    // old behaviour rather than a wrong boundary.
                    lps.skipToContent(full_text.items, think_split.content, stop_gate.carry_len);
                    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = think_split.content }, null, null, null, .{ .logprobs_json = try lps.take() });
                } else {
                    // All reasoning, no content: those entries describe text the
                    // client never received.
                    lps.dropPending();
                }
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    if (!client_gone) {
        // Final chunk with finish_reason
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, .{ .reason = finish_reason, .details = ts.finish_details }, null, null, .{ .logprobs_json = try lps.take() });

        // Usage chunk (if requested via stream_options.include_usage). Scheduler
        // accounts for any prompt-cache hits in `ts.prompt_tokens` directly.
        // OpenAI ships this chunk with an EMPTY choices array — restating
        // finish_reason/finish_details here made every per-event client render
        // the ending twice (PR #147's doubled truncation banner). Pending
        // logprobs all drained on the final chunk above.
        if (include_usage) {
            const details = try formatReasoningDetails(allocator, tok, delivered_reasoning.items);
            defer allocator.free(details);
            const usage_json = try formatChatUsage(allocator, total_prompt, ts.completion_tokens, ts.cached_tokens, details);
            defer allocator.free(usage_json);
            const timings_obj = try formatTimingsObject(allocator, total_prompt, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns, tokenize_ns);
            defer allocator.free(timings_obj);
            const timings_opt: ?[]const u8 = if (timings_obj.len > 0) timings_obj else null;
            try sendSSEUsageChunk(allocator, stream, chat_id, model_name, usage_json, timings_opt);
        }

        // Done sentinel
        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    // Per-slot timings come from the scheduler (ts.prefill_ns / ts.decode_ns,
    // populated in finalize); the bracket reports compute + any prefix reuse.
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, perf, finish_reason,
    });
}

const DeltaFields = struct {
    role: ?[]const u8,
    content: ?[]const u8,
    reasoning_content: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

/// Per-choice fields that sit BESIDE `delta`, not inside it. Defaulted so a
/// call site that has nothing to add passes `.{}` — every existing emitter
/// keeps its exact bytes, which is what makes this additive.
const ChunkExtras = struct {
    /// Already-rendered `{"content":[…]}` from `formatLogprobsObject`, or null
    /// to omit the key entirely.
    logprobs_json: ?[]const u8,
};

/// How a stream ended: the wire `finish_reason` plus, when the
/// degenerate-tail guard cut it, the sibling cause (`finishDetailsField`).
const Finish = struct {
    reason: []const u8,
    details: ?[]const u8 = null,
};

fn sendSSEChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    delta: DeltaFields,
    /// `null` on every mid-stream delta; the two FINAL chunks pass the reason
    /// and, when the degenerate-tail guard cut the turn, its cause. Modelled
    /// as one value so a site cannot report the reason and forget the cause.
    finish: ?Finish,
    usage_json: ?[]const u8,
    timings_json: ?[]const u8,
    extras: ChunkExtras,
) !void {
    // Build the delta JSON object
    var delta_buf = std.ArrayList(u8).empty;
    defer delta_buf.deinit(allocator);

    try delta_buf.appendSlice(allocator, "{");
    var need_comma = false;

    if (delta.role) |role| {
        try delta_buf.appendSlice(allocator, "\"role\":\"");
        try delta_buf.appendSlice(allocator, role);
        try delta_buf.appendSlice(allocator, "\"");
        need_comma = true;
    }

    if (delta.content) |content| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"content\":");
        const escaped = try jsonEscape(allocator, content);
        defer allocator.free(escaped);
        try delta_buf.appendSlice(allocator, escaped);
        need_comma = true;
    }

    if (delta.reasoning_content) |reasoning| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"reasoning_content\":");
        const escaped_r = try jsonEscape(allocator, reasoning);
        defer allocator.free(escaped_r);
        try delta_buf.appendSlice(allocator, escaped_r);
        need_comma = true;
    }

    if (delta.tool_calls_json) |tc_json| {
        if (need_comma) try delta_buf.appendSlice(allocator, ",");
        try delta_buf.appendSlice(allocator, "\"tool_calls\":");
        try delta_buf.appendSlice(allocator, tc_json);
    }

    try delta_buf.appendSlice(allocator, "}");

    // Build the finish_reason field (+ its sibling cause, when there is one)
    var fr_buf: [64]u8 = undefined;
    const fr_str = if (finish) |f|
        std.fmt.bufPrint(&fr_buf, "\"{s}\"", .{f.reason}) catch "null"
    else
        "null";
    const fd_str = if (finish) |f| finishDetailsField(f.reason, f.details) else "";

    // `logprobs` is a sibling of `delta` on the choice, not a field inside it.
    // Emitted only when the request asked; OpenAI's own chunk shape carries
    // `"logprobs":null` otherwise, which is what the empty case renders.
    var lp_buf = std.ArrayList(u8).empty;
    defer lp_buf.deinit(allocator);
    if (extras.logprobs_json) |lp| {
        try lp_buf.appendSlice(allocator, ",\"logprobs\":");
        try lp_buf.appendSlice(allocator, lp);
    }

    // Build usage field
    const usage_str = if (usage_json) |u| u else "null";

    // Optional `timings` tail. Inserted as a top-level field next to `usage`
    // — matches the llama.cpp shape so clients that key off it work as-is.
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }

    // Build the full SSE chunk
    // finish_reason always carries finish_details beside it.
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"delta":{s},"finish_reason":{s}{s}{s}}}],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, fd_str, lp_buf.items, usage_str, timings_tail_buf.items });
    defer allocator.free(chunk);

    const shipped = if (finish != null) try withEffortNoticeJson(allocator, chunk, stream.effort_notice_json) else try allocator.dupe(u8, chunk);
    defer allocator.free(shipped);
    logHttpSseData(shipped);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(shipped);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

/// The `stream_options.include_usage` chunk: OpenAI's shape is an EMPTY
/// `choices` array beside the usage object — no delta, no finish_reason, no
/// logprobs (all per-choice fields; the final chunk already carried them).
fn sendSSEUsageChunk(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    usage_json: []const u8,
    timings_json: ?[]const u8,
) !void {
    var timings_tail_buf = std.ArrayList(u8).empty;
    defer timings_tail_buf.deinit(allocator);
    if (timings_json) |t| {
        try timings_tail_buf.appendSlice(allocator, ",\"timings\":");
        try timings_tail_buf.appendSlice(allocator, t);
    }
    const chunk = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, usage_json, timings_tail_buf.items });
    defer allocator.free(chunk);
    const shipped = try withEffortNoticeJson(allocator, chunk, stream.effort_notice_json);
    defer allocator.free(shipped);

    logHttpSseData(shipped);
    try stream.writeAllNoFlush("data: ");
    try stream.writeAllNoFlush(shipped);
    try stream.writeAllNoFlush("\n\n");
    try stream.flush();
}

test "every load refusal the registry preserves answers under its own name" {
    const t = std.testing;
    const names = [_][]const u8{
        "InsufficientMemory",
        "OutOfMemory",
        "ArchitectureUnsupported",
        "ModelFormatUnsupported",
        "ExpertCacheDoesNotFit",
        "ExpertStreamingRequired",
        "SsdBudgetBelowResident",
        "SsdBudgetBelowVision",
        "SsdBudgetExceedsWiredLimit",
        "ExpertStreamingMtpUnsupported",
        "ExpertStreamingUnsupportedLayout",
        "ExpertSlabImportCopied",
        "ExpertLayoutUnsupported",
        "Exl3TopKExceedsReduceBank",
        "Exl3TrellisGeometry",
        "Exl3WindowUnsupported",
        "Exl3ShardStampMismatch",
        "Exl3CodebookTableUnknown",
        "Exl3CodebookTableDigestMismatch",
        "Exl3CodebookTableWindowMismatch",
        "Exl3CodebookTableCodebookMismatch",
        "Exl3CodebookTableRateMissing",
        "Exl3CodebookTableLegacyConfig",
        "Exl3CodebookTableUndecodable",
    };
    for (names) |name| {
        const refusal = loadRefusalFor(model_registry_mod.ModelRegistry.loadErrorFromName(name)) orelse {
            std.debug.print("no named refusal for {s}\n", .{name});
            return error.TestUnexpectedResult;
        };
        try t.expect(refusal.type.len > 0);
        try t.expect(refusal.message.len > 0);
    }
    try t.expectEqualStrings("architecture_unsupported", loadRefusalFor(error.ArchitectureUnsupported).?.type);
    try t.expectEqualStrings("model_format_unsupported", loadRefusalFor(error.ModelFormatUnsupported).?.type);
    try t.expectEqualStrings("expert_streaming_unsupported_layout", loadRefusalFor(error.ExpertStreamingUnsupportedLayout).?.type);
    try t.expectEqualStrings("expert_slab_import_copied", loadRefusalFor(error.ExpertSlabImportCopied).?.type);
    try t.expectEqualStrings("expert_layout_unsupported", loadRefusalFor(error.ExpertLayoutUnsupported).?.type);
    try t.expectEqualStrings("exl3_topk_exceeds_reduce_bank", loadRefusalFor(error.Exl3TopKExceedsReduceBank).?.type);
    try t.expectEqualStrings("exl3_trellis_geometry", loadRefusalFor(error.Exl3TrellisGeometry).?.type);
    try t.expectEqualStrings("exl3_window_unsupported", loadRefusalFor(error.Exl3WindowUnsupported).?.type);
    try t.expectEqualStrings("exl3_shard_stamp_mismatch", loadRefusalFor(error.Exl3ShardStampMismatch).?.type);
    try t.expectEqualStrings("exl3_codebook_table_unknown", loadRefusalFor(error.Exl3CodebookTableUnknown).?.type);
    try t.expectEqualStrings("exl3_codebook_table_digest_mismatch", loadRefusalFor(error.Exl3CodebookTableDigestMismatch).?.type);
    try t.expectEqualStrings("exl3_codebook_table_window_mismatch", loadRefusalFor(error.Exl3CodebookTableWindowMismatch).?.type);
    try t.expectEqualStrings("exl3_codebook_table_codebook_mismatch", loadRefusalFor(error.Exl3CodebookTableCodebookMismatch).?.type);
    try t.expectEqualStrings("exl3_codebook_table_rate_missing", loadRefusalFor(error.Exl3CodebookTableRateMissing).?.type);
    try t.expectEqualStrings("exl3_codebook_table_legacy_config", loadRefusalFor(error.Exl3CodebookTableLegacyConfig).?.type);
    try t.expectEqualStrings("exl3_codebook_table_undecodable", loadRefusalFor(error.Exl3CodebookTableUndecodable).?.type);
    try t.expectEqualStrings("ssd_budget_below_vision", loadRefusalFor(error.SsdBudgetBelowVision).?.type);
    try t.expect(std.mem.indexOf(u8, loadRefusalFor(error.SsdBudgetBelowVision).?.message, "--vision") != null);
    try t.expect(loadRefusalFor(error.LoadFailed) == null);
    try t.expect(loadRefusalFor(error.UnknownModelId) == null);
}

pub const LoadRefusal = struct { type: []const u8, message: []const u8 };

test "streaming layout refusal explains MiMo repacking without assuming Qwen" {
    const layout = loadRefusalFor(error.ExpertStreamingUnsupportedLayout).?;
    try std.testing.expect(std.mem.indexOf(u8, layout.message, "For MiMo, repack") != null);
    const budget = loadRefusalFor(error.ExpertStreamingRequired).?;
    try std.testing.expect(std.mem.indexOf(u8, budget.message, "qwen4_exp") == null);
    try std.testing.expect(std.mem.indexOf(u8, budget.message, "--ssd-budget-gb") != null);
}

pub fn loadRefusalFor(err: anyerror) ?LoadRefusal {
    return switch (err) {
        error.NotEnoughMemory => .{ .type = "out_of_memory", .message = not_enough_memory_message },
        error.InsufficientMemory => .{ .type = "out_of_memory", .message = insufficient_free_memory_message },
        error.ArchitectureUnsupported => .{ .type = "architecture_unsupported", .message = "This checkpoint's model_type is not served by this build, which loads qwen4_exp (Qwen3.8-Flash-Next), mimo_v2 (MiMo-V2.6-Flash), and glm5_next (GLM-5.3-Flash)." },
        error.ModelFormatUnsupported => .{ .type = "model_format_unsupported", .message = "This checkpoint's file format is not supported. Serve an MLX safetensors checkpoint (qwen4_exp or mimo_v2)." },
        error.ExpertCacheDoesNotFit => .{ .type = "expert_cache_does_not_fit", .message = "The requested expert cache, full-union workspace, bounce buffers, resident trunk, and serving state do not fit under the GPU memory ceiling. Lower --ssd-budget-gb or free memory." },
        error.ExpertStreamingMtpUnsupported => .{ .type = "expert_streaming_mtp_unsupported", .message = expert_stream_mod.MTP_UNSUPPORTED },
        error.ExpertStreamingRequired => .{ .type = "expert_streaming_required", .message = "This checkpoint streams its experts from SSD and needs a resident budget: set this model's \"ssd_budget_gb\" in model-settings.json, or launch with --ssd-budget-gb <n>." },
        error.ExpertStreamingUnsupportedLayout => .{ .type = "expert_streaming_unsupported_layout", .message = "This checkpoint has no complete expert-bank layout this build can stream. Check the pack and all indexed shards. For MiMo, repack the checkpoint first; raw per-expert HF shards cannot be streamed directly." },
        error.ExpertSlabImportCopied => .{ .type = "expert_slab_import_copied", .message = "MLX copied the expert slab instead of aliasing it, so this machine cannot stream experts zero-copy. Report the Mac model and macOS version." },
        error.ExpertLayoutUnsupported => .{ .type = "expert_layout_unsupported", .message = "This checkpoint's routed experts are not a pack layout this build can load. An EXL3 pack's expert_quant names format exl3, a k, and codebook mcg or mul1, or format stq and codebook std1, spec1 or pair1 (with their codebook_table and codebook_digest); re-convert the pack." },
        error.Exl3GroupMissing => .{ .type = "exl3_group_missing", .message = "An EXL3 rate group or projection tensor is missing." },
        error.Exl3GroupNameInvalid => .{ .type = "exl3_group_name_invalid", .message = "EXL3 rate groups must use contiguous g0 through g31 names." },
        error.Exl3MixedGroupLayout => .{ .type = "exl3_mixed_group_layout", .message = "A layer mixes grouped and ungrouped EXL3 tensors." },
        error.Exl3GroupGeometry => .{ .type = "exl3_group_geometry", .message = "EXL3 group expert counts or projection dimensions disagree." },
        error.Exl3GroupDtype => .{ .type = "exl3_group_dtype", .message = "EXL3 trellises must be U16 and axis scales must be F16." },
        error.Exl3RouterWidthMismatch => .{ .type = "exl3_router_width_mismatch", .message = "The router width must equal the sum of its layer expert group sizes." },
        error.Exl3TopKExceedsExperts => .{ .type = "exl3_topk_exceeds_experts", .message = "The routed top-k exceeds this layer expert count." },
        error.Exl3RaggedStreamingUnsupported => .{ .type = "exl3_ragged_streaming_unsupported", .message = "EXL3 rate groups and pruned expert layers require resident loading." },
        error.Exl3RateGroupsUnsupported => .{ .type = "exl3_rate_groups_unsupported", .message = "GLM-5.3-Flash packs serve one rate per layer; a rate-grouped layer loads on Qwen3.8-Flash-Next and MiMo-V2.6-Flash only." },
        error.Exl3RouterGroupsUnsupported => .{ .type = "exl3_router_groups_unsupported", .message = "Reordered or pruned EXL3 experts require n_group=1 routing." },
        error.Exl3TopKExceedsReduceBank => .{ .type = "exl3_topk_exceeds_reduce_bank", .message = "This EXL3 pack's num_experts_per_tok exceeds the decode reduce-bank (32). Re-convert with top-k <= 32." },
        error.Exl3RateGroupsStreamingUnsupported => .{ .type = "exl3_rate_groups_streaming_unsupported", .message = "SSD streaming does not support EXL3 rate-group (.gN) packs. Use uniform leading-index expert banks." },
        error.Exl3GateUpRateMismatch => .{ .type = "exl3_gate_up_rate_mismatch", .message = "The EXL3 gate and up trellises must use the same packed rate." },
        error.Exl3TrellisGeometry => .{ .type = "exl3_trellis_geometry", .message = "This EXL3 pack has a routed-expert trellis this build cannot decode, or one that disagrees with the expert count, shape or k its config.json names. Re-convert the pack." },
        error.Exl3WindowUnsupported => .{ .type = "exl3_window_unsupported", .message = "This EXL3 pack names a codeword window this build cannot decode: expert_quant.window must be an integer from 8 to 16, or absent for 16." },
        error.Exl3ShardStampMismatch => .{ .type = "exl3_shard_stamp_mismatch", .message = "An EXL3 shard in this pack was written for a different decoder than config.json's expert_quant names (k, codebook or window). Re-convert the pack, or fix expert_quant to match the shards." },
        error.Exl3CodebookTableUnknown => .{ .type = "exl3_codebook_table_unknown", .message = "This STQ pack names no codebook table this build holds: expert_quant.codebook_table must be one of std1, spec1-w15, spec1-w16 or pair1-w16-v1. A pack that names no table is reconverted." },
        error.Exl3CodebookTableDigestMismatch => .{ .type = "exl3_codebook_table_digest_mismatch", .message = "This STQ pack's expert_quant.codebook_digest is not its codebook table's digest in this build. Re-convert the pack." },
        error.Exl3CodebookTableWindowMismatch => .{ .type = "exl3_codebook_table_window_mismatch", .message = "The PAIR1 codebook table this STQ pack names decodes only at its own window (pair1-w16-v1: 16), not at expert_quant.window." },
        error.Exl3CodebookTableCodebookMismatch => .{ .type = "exl3_codebook_table_codebook_mismatch", .message = "The codebook table this STQ pack names is for another codebook than expert_quant.codebook." },
        error.Exl3CodebookTableRateMissing => .{ .type = "exl3_codebook_table_rate_missing", .message = "This STQ pack has a routed bank at a rate its codebook table has no row for (each table covers the even n it lists: spec1-w15 n16 to n128, the others n24 to n64)." },
        error.Exl3CodebookTableLegacyConfig => .{ .type = "exl3_codebook_table_legacy_config", .message = "This pack's expert_quant names a codebook table by a digest from before the table grew (spec1-w15 2ac2594d, now 37f72431). Re-convert the pack." },
        error.Exl3CodebookTableUndecodable => .{ .type = "exl3_codebook_table_undecodable", .message = "This STQ pack's codebook table holds PAIR1 constants this build's kernels have not compiled." },
        error.SsdBudgetBelowResident => .{ .type = "ssd_budget_below_resident", .message = "--ssd-budget-gb leaves no room for an expert cache after the resident trunk, the prefill union and the fill buffers. Raise the budget." },
        error.SsdBudgetBelowVision => .{ .type = "ssd_budget_below_vision", .message = "--ssd-budget-gb cannot hold the vision tower (--vision, or the default on DeepSeek-V4.1) beside the minimum expert cache. Raise the budget or pass --no-vision." },
        error.SsdBudgetExceedsWiredLimit => .{ .type = "ssd_budget_exceeds_wired_limit", .message = "--ssd-budget-gb plus the planned KV cache (and, under --vision, the largest image encode) exceeds the machine's residency limit. Raise iogpu.wired_limit_mb (the server log names the value) or lower the budget." },
        else => null,
    };
}

pub fn sendNamedLoadRefusal(allocator: std.mem.Allocator, stream: *Conn, scheduler: *scheduler_mod.Scheduler, id: []const u8, err: anyerror) !bool {
    const refusal = loadRefusalFor(err) orelse return false;
    var buf: [256]u8 = undefined;
    const message = if (err == error.InsufficientMemory) blk: {
        const context = scheduler.takeLoadContextRefusal(id) orelse break :blk refusal.message;
        break :blk memoryContextRefusalMessage(&buf, context);
    } else refusal.message;
    try sendErrorResponse(allocator, stream, "503 Service Unavailable", refusal.type, message, 503);
    return true;
}

/// #144: `error.LoadFailed` used to surface as a bare "Model load failed" for
/// every on-demand load failure — the inference thread's error name was
/// dropped at the thread boundary. The registry keeps it; echo it.
pub fn sendLoadFailedResponse(allocator: std.mem.Allocator, stream: *Conn, sched: *scheduler_mod.Scheduler, id: []const u8) !void {
    if (sched.loadErrorName(allocator, id)) |name| {
        defer allocator.free(name);
        if (std.fmt.allocPrint(allocator, "Model load failed: {s}", .{name}) catch null) |msg| {
            defer allocator.free(msg);
            try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
            return;
        }
    }
    try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
}

pub fn contextOverflowMessage(buf: []u8, prompt_tokens: usize, ctx: usize) []const u8 {
    return std.fmt.bufPrint(
        buf,
        "Prompt exceeds maximum context length: {d} tokens requested, {d} available",
        .{ prompt_tokens, ctx },
    ) catch "Prompt exceeds maximum context length";
}

/// Delivered reasoning is a subset of completion_tokens, not an additional charge.
fn formatReasoningDetails(allocator: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer, text: []const u8) ![]u8 {
    if (text.len == 0) return reasoningDetailsJson(allocator, 0);
    const ids = try tok.encode(allocator, text);
    defer allocator.free(ids);
    return reasoningDetailsJson(allocator, ids.len);
}

fn reasoningDetailsJson(allocator: std.mem.Allocator, count: usize) ![]u8 {
    return std.fmt.allocPrint(allocator, ",\"completion_tokens_details\":{{\"reasoning_tokens\":{d}}}", .{count});
}

/// The chat-completions `usage` object, shared by the non-stream, tool-call
/// and streaming-final-chunk emitters so the shape never drifts per path.
/// `prompt_tokens_details.cached_tokens` is ALWAYS present (0 when cold) —
/// OpenAI emits it unconditionally, and black-box conformance/cost tooling
/// (llmprobe) treats a missing field as "no prompt caching" even though the
/// engine caches. `extra_details` is a pre-formatted `,"key":{...}` fragment
/// (completion_tokens_details) or "".
fn formatChatUsage(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    extra_details: []const u8,
) ![]u8 {
    return try std.fmt.allocPrint(allocator,
        \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d},"prompt_tokens_details":{{"cached_tokens":{d}}}{s}}}
    , .{ prompt_tokens, completion_tokens, prompt_tokens + completion_tokens, cached_tokens, extra_details });
}

pub fn formatTimingsObject(
    allocator: std.mem.Allocator,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
) ![]u8 {
    if (prefill_ns == 0 and decode_ns == 0 and tokenize_ns == 0) return try allocator.alloc(u8, 0);
    const ns_per_ms_f: f64 = @as(f64, @floatFromInt(std.time.ns_per_ms));
    const p_ms = @as(f64, @floatFromInt(prefill_ns)) / ns_per_ms_f;
    const d_ms = @as(f64, @floatFromInt(decode_ns)) / ns_per_ms_f;
    const t_ms = @as(f64, @floatFromInt(tokenize_ns)) / ns_per_ms_f;
    // prompt_per_second reflects compute: divide by the tokens actually run
    // (prompt minus the KV-cache prefix). `cached_n` exposes the reuse so a
    // bench / client can tell a warm hit from a cold prefill.
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    // Always emit `tokenize_ms` (even at 0.0) so clients can rely on the key
    // being present — they can branch on the value, not on presence/absence.
    return try std.fmt.allocPrint(
        allocator,
        \\{{"prompt_n":{d},"cached_n":{d},"prompt_ms":{d:.3},"prompt_per_second":{d:.3},"predicted_n":{d},"predicted_ms":{d:.3},"predicted_per_second":{d:.3},"tokenize_ms":{d:.3}}}
    ,
        .{ prompt_tokens, cached_tokens, p_ms, p_tps, completion_tokens, d_ms, d_tps, t_ms },
    );
}

/// Format the "prefill: X tok/s [(C cached / P total)], decode: Y tok/s" perf
/// bracket into `buf` so every API path logs prefill compute (and any prefix
/// reuse) identically. `cached > 0` adds the cached/total hint; otherwise the
/// short form keeps the existing log shape for cold prefills. Single source of
/// truth so the format never drifts across the chat/anthropic/streaming logs.
pub fn formatPerfBracket(
    buf: []u8,
    prompt_tokens: u32,
    cached_tokens: u32,
    completion_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
) []const u8 {
    const p_tps = generate_mod.prefillTokensPerSec(prompt_tokens, cached_tokens, prefill_ns);
    const d_tps = generate_mod.tokensPerSec(completion_tokens, decode_ns);
    const s = if (cached_tokens > 0)
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s ({d} cached / {d} total), decode: {d:.1} tok/s", .{ p_tps, cached_tokens, prompt_tokens, d_tps })
    else
        std.fmt.bufPrint(buf, "prefill: {d:.1} tok/s, decode: {d:.1} tok/s", .{ p_tps, d_tps });
    return s catch buf[0..0];
}

fn quotedJsonInner(escaped: []const u8) []const u8 {
    if (escaped.len >= 2 and escaped[0] == '"' and escaped[escaped.len - 1] == '"') return escaped[1 .. escaped.len - 1];
    return escaped;
}

fn formatChatStreamToolCall(allocator: std.mem.Allocator, index: usize, id: []const u8, name: []const u8, arguments: []const u8) ![]u8 {
    const escaped_name = try jsonEscape(allocator, name);
    defer allocator.free(escaped_name);
    const escaped_args = try jsonEscape(allocator, arguments);
    defer allocator.free(escaped_args);
    return std.fmt.allocPrint(allocator,
        \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]
    , .{ index, id, quotedJsonInner(escaped_name), quotedJsonInner(escaped_args) });
}

/// The half-open token range whose decoded text lands inside `content`.
const ContentTokenRange = struct { start: usize, end: usize };

/// Which of the generated tokens survive into `message.content`.
///
/// OpenAI defines `logprobs.content` as the tokens of the message CONTENT, but
/// the generation also carries the reasoning block, any leaked tool markup and
/// a loop-trimmed tail — all cut before the text reaches the client. Emitting
/// entries for cut text makes the array correspond to nothing the client can
/// see: on every model that thinks, `logprobs.content[0]` was the first token
/// of the reasoning.
///
/// The split helpers return raw slices into `full_text`, so the offset is exact
/// pointer arithmetic. A content that is not a slice of it (a future transform
/// that rewrites rather than cuts) falls back to a search, and failing that to
/// the full range — an array we cannot align is still better than none.
pub fn contentTokenRange(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    full_text: []const u8,
    content: []const u8,
) ContentTokenRange {
    const all = ContentTokenRange{ .start = 0, .end = token_ids.len };
    if (content.len == 0) return .{ .start = 0, .end = 0 };

    const base = @intFromPtr(full_text.ptr);
    const cptr = @intFromPtr(content.ptr);
    const start_off = if (cptr >= base and cptr + content.len <= base + full_text.len)
        cptr - base
    else
        std.mem.indexOf(u8, full_text, content) orelse return all;
    const end_off = start_off + content.len;

    var start: ?usize = null;
    var off: usize = 0;
    for (token_ids, 0..) |id, i| {
        if (off >= end_off) return .{ .start = start orelse i, .end = i };
        const text = tok.decode(allocator, &[_]u32{id}) catch return all;
        defer allocator.free(text);
        // A token straddling the boundary belongs to content: it put bytes there.
        if (start == null and off + text.len > start_off) start = i;
        off += text.len;
    }
    return .{ .start = start orelse token_ids.len, .end = token_ids.len };
}

/// SSE comment keepalive for the OpenAI-style streaming surfaces. Comments
/// are SSE-spec-legal and skipped by every SSE parser. No-op on the
/// WebSocket transport — a raw comment line would corrupt WS framing, and
/// WS has protocol-level liveness of its own.
fn sendStreamKeepalive(stream: *Conn) !void {
    if (stream.ws_mode != null) return;
    try stream.writeAll(": keepalive\n\n");
}

/// The keepalive of a stream still waiting for its first token: while the request is queued
/// behind another, the comment says so. It is a comment, so no client reads it as content.
pub fn sendStreamIdle(stream: *Conn, queued: ?slot_admission.QueueWait) !void {
    const w = queued orelse return sendStreamKeepalive(stream);
    if (stream.ws_mode != null) return;
    var buf: [196]u8 = undefined;
    const line = slot_admission.queueWaitText(buf[0..192], w, false);
    buf[line.len] = '\n';
    buf[line.len + 1] = '\n';
    try stream.writeAll(buf[0 .. line.len + 2]);
}

/// Which no-op event a surface uses to prove liveness.
const KeepaliveStyle = enum { sse_comment, anthropic_ping };

/// Emit a keepalive iff the socket has been SILENT for `STREAM_KEEPALIVE_MS`.
///
/// Every streaming token loop calls this once per iteration — not just on the
/// `.idle` (waiting-for-first-token) path. Those two are different facts: a
/// surface buffering tokens for tool-call or thinking detection is receiving
/// tokens steadily while writing nothing, and only bytes on the wire hold off
/// a client's idle-body timeout (Node's `fetch`/undici: 300 s, then
/// `TypeError: terminated`). A one-shot `write_file` of a whole source file
/// buffers for exactly that long.
pub fn beatStreamKeepalive(stream: *Conn, style: KeepaliveStyle) !void {
    if (!stream.keepaliveDue()) return;
    switch (style) {
        .sse_comment => try sendStreamKeepalive(stream),
        .anthropic_ping => try sendAnthropicEvent(stream, "ping", "{\"type\":\"ping\"}"),
    }
    // The WS transport no-ops both senders; stamp regardless so a bridged
    // connection doesn't re-evaluate the deadline on every token.
    stream.heartbeat.noteWrite(nowMsMonotonic(stream.io));
}

test "loopTrimmedIds: the degenerate span is cut, and a bad index degrades to emitting everything" {
    const ids = [_]u32{ 1, 2, 3, 4, 5 };
    // No cut → untouched (identity for every non-loop response).
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, null));
    try std.testing.expectEqualSlices(u32, ids[0..3], loopTrimmedIds(&ids, 3));
    // A whole-generation loop leaves nothing: honest, and finish_reason
    // "length" already says the server cut it.
    try std.testing.expectEqualSlices(u32, ids[0..0], loopTrimmedIds(&ids, 0));
    // The index comes from the generator's own token list, so a mismatch
    // must emit everything rather than slice out of bounds.
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 5));
    try std.testing.expectEqualSlices(u32, &ids, loopTrimmedIds(&ids, 99));
}

test "chat stream tool-call delta escapes the function name" {
    const a = std.testing.allocator;
    const delta = try formatChatStreamToolCall(a, 0, "call_1", "say\"hi", "{\"x\":1}");
    defer a.free(delta);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, delta, .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value == .array);
    const fn_obj = parsed.value.array.items[0].object.get("function").?.object;
    try std.testing.expectEqualStrings("say\"hi", fn_obj.get("name").?.string);
    try std.testing.expectEqualStrings("{\"x\":1}", fn_obj.get("arguments").?.string);
}

test "a tool-call reply delivers the budget-capped thought" {
    try std.testing.expectEqualStrings("short", deliveredReasoning("short", "short and the rest the budget withholds").?);
    try std.testing.expectEqualStrings("full", deliveredReasoning(null, "full").?);
    try std.testing.expect(deliveredReasoning(null, null) == null);
}

test "formatChatUsage: prompt_tokens_details.cached_tokens always present (llmprobe chat caching)" {
    const t = std.testing;
    const a = t.allocator;

    // Cold request: field present with 0, matching OpenAI's unconditional emission.
    const cold = try formatChatUsage(a, 100, 5, 0, "");
    defer a.free(cold);
    try t.expectEqualStrings(
        \\{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":0}}
    , cold);

    // Warm request: cached count surfaces; cached ≤ prompt by construction
    // (scheduler folds cached into prompt_tokens).
    const warm = try formatChatUsage(a, 300, 12, 250, "");
    defer a.free(warm);
    try t.expectEqualStrings(
        \\{"prompt_tokens":300,"completion_tokens":12,"total_tokens":312,"prompt_tokens_details":{"cached_tokens":250}}
    , warm);

    // Reasoning responses append completion_tokens_details after the prompt details.
    const with_details = try formatChatUsage(a, 10, 20, 4, ",\"completion_tokens_details\":{\"reasoning_tokens\":7}");
    defer a.free(with_details);
    try t.expectEqualStrings(
        \\{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":7}}
    , with_details);
}

test "a slot the transient cap failed answers a named 503, not the memory 503" {
    const t = std.testing;
    var slot: scheduler_mod.Slot = undefined;
    slot.io = t.io;
    slot.out_mu = .init;
    slot.memory_context_refusal = null;
    slot.media_refusal = null;
    slot.shutdown_cancelled = .init(false);
    slot.error_code = "TransientCapExceeded";

    try t.expect(slotFailure(&slot) == error.TransientCapExceeded);
    var buf: [192]u8 = undefined;
    const w = mapGenerationError(error.TransientCapExceeded, &buf);
    try t.expectEqual(@as(u32, 503), w.code);
    try t.expectEqualStrings(TRANSIENT_CAP_MSG, w.message);
    try t.expect(!std.mem.eql(u8, w.message, GEN_OOM_MSG));

    slot.error_code = "OutOfMemory";
    try t.expect(slotFailure(&slot) == error.GenerationOutOfMemory);
}

test "a streaming fault answers with the SAME mapped error a non-streaming one does" {
    // The 400/503 mapping existed only on the non-streaming arms; streaming arms wrote a bare `server_error` frame.
    const t = std.testing;
    const a = t.allocator;

    var buf: [192]u8 = undefined;
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        try t.expectEqual(@as(u32, 503), w.code);
        try t.expectEqualStrings("503 Service Unavailable", w.status_line);
        try t.expectEqualStrings(GEN_OOM_MSG, w.message);
        try t.expectEqualStrings("server_error", w.openai_type);
        try t.expectEqualStrings("api_error", w.anthropic_type);
    }
    {
        const w = mapGenerationError(error.PrefillDoesNotFit, &buf);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings(PREFILL_NOFIT_MSG, w.message);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
        try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    }
    {
        const w = mapGenerationError(error.ServerShutdown, &buf);
        try t.expectEqual(@as(u32, 503), w.code);
        try t.expectEqualStrings("server_error", w.openai_type);
        try t.expectEqualStrings("Server is shutting down", w.message);
    }
    {
        const w = mapGenerationError(error.GenerationFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("generation failed", w.message);
    }
    {
        // Anything else keeps its name in the message, at a mapped status.
        const w = mapGenerationError(error.WriteFailed, &buf);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("Internal server error: WriteFailed", w.message);
    }

    // The SSE bodies carry the same type and message, plus the surface's terminal marker.
    {
        const w = mapGenerationError(error.GenerationOutOfMemory, &buf);
        const body = try buildOpenAiStreamError(a, w);
        defer a.free(body);
        try t.expect(std.mem.indexOf(u8, body, "\"type\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, body, "\"code\":503") != null);
        try t.expect(std.mem.indexOf(u8, body, GEN_OOM_MSG) != null);
        try t.expect(std.mem.indexOf(u8, body, "\"finish_reason\":\"error\"") != null);

        const anth = try buildAnthropicStreamError(a, w);
        defer a.free(anth);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"api_error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, GEN_OOM_MSG) != null);

        const resp = try buildResponsesStreamError(a, w);
        defer a.free(resp);
        try t.expect(std.mem.indexOf(u8, resp, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, "\"code\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, GEN_OOM_MSG) != null);
    }
}

test "an empty prompt refused at submit is a 400 invalid_request_error in both dialects" {
    const t = std.testing;
    var buf: [192]u8 = undefined;
    const w = mapGenerationError(error.EmptyPrompt, &buf);
    try t.expectEqual(@as(u32, 400), w.code);
    try t.expectEqualStrings("400 Bad Request", w.status_line);
    try t.expectEqualStrings("invalid_request_error", w.openai_type);
    try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    try t.expectEqualStrings(EMPTY_PROMPT_MSG, w.message);
}

test "contextOverflowMessage: the 400 names both counts so a client can act on it" {
    const t = std.testing;
    var buf: [160]u8 = undefined;

    // The counts ARE the feature: without them a client can only say "too
    // long", and our own app can't offer "raise the context to N" — which is
    // the one action that fixes it.
    const msg = contextOverflowMessage(&buf, 4108, 4096);
    try t.expectEqualStrings("Prompt exceeds maximum context length: 4108 tokens requested, 4096 available", msg);

    // Legacy prefix preserved verbatim — APIError.looksLikeContextOverflow and
    // every third-party client key on this exact phrase.
    try t.expect(std.mem.startsWith(u8, msg, "Prompt exceeds maximum context length"));

    // No quotes or control bytes, so the message is safe to drop into a JSON
    // error body without escaping (the appendJsonString class).
    for (msg) |c| try t.expect(c >= 0x20 and c != '"' and c != '\\');
}

test "contextOverflowMessage: a buffer too small falls back rather than sending nothing" {
    const t = std.testing;
    // bufPrint failing must not propagate — the media-gen 400 that sent NO body
    // at all when its fixed buffer overflowed is the class this guards.
    var tiny: [8]u8 = undefined;
    try t.expectEqualStrings("Prompt exceeds maximum context length", contextOverflowMessage(&tiny, 999999, 1));
}

test "a vanished client is not a token cap: the non-stream finish reason says so" {
    // Bar: the non-streaming path reports what the three streaming paths do.
    const t = std.testing;
    try t.expectEqualStrings("client_disconnect", nonStreamFinishReason(true, "length"));
    try t.expectEqualStrings("stop", nonStreamFinishReason(false, "stop"));
    try t.expectEqualStrings("length", nonStreamFinishReason(false, "length"));
}

test "a tool-call answer keeps the model's visible preamble" {
    // Bar: non-streamed content == what the stream flushes before the tool markup.
    const t = std.testing;
    const raw = "<think>pick a tool</think>Checking the weather.\n<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>";
    const split = think_tags.splitThinkBlock(raw, true, false);
    try t.expectEqualStrings("Checking the weather.", visibleToolPreamble(split.content));
    const bare = think_tags.splitThinkBlock("<tool_call>{\"name\":\"w\",\"arguments\":{}}</tool_call>", true, false);
    try t.expectEqualStrings("", visibleToolPreamble(bare.content));
    try t.expectEqualStrings("", visibleToolPreamble("  \n "));
}

test "sushi coder load refusals retain named group errors" {
    for ([_]anyerror{ error.Exl3GroupMissing, error.Exl3GroupNameInvalid, error.Exl3MixedGroupLayout, error.Exl3GroupGeometry, error.Exl3GroupDtype, error.Exl3RouterWidthMismatch, error.Exl3TopKExceedsExperts, error.Exl3RaggedStreamingUnsupported, error.Exl3RouterGroupsUnsupported, error.Exl3RateGroupsUnsupported }) |err| {
        try std.testing.expect(loadRefusalFor(err) != null);
    }
}

fn expectSettledAfterEarlyStop(published: u32) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    try tok.id_to_token.put(1, "S");
    var sch: scheduler_mod.Scheduler = undefined;
    sch.io = io;
    sch.queue_mu = .init;
    var slot: scheduler_mod.Slot = undefined;
    slot.io = io;
    slot.allocator = a;
    slot.out_mu = .init;
    slot.out_cond = .init;
    slot.out_event = .unset;
    slot.out_buf = .empty;
    defer slot.out_buf.deinit(a);
    slot.logprobs_buf = .empty;
    slot.out_idx = 0;
    slot.handover_token = null;
    slot.cancelled = .init(false);
    slot.finished = false;
    slot.error_code = null;
    slot.completion_tokens = 0;
    slot.decode_ns = 0;
    slot.in_pass = .init(1);

    var player = TickPlayer{ .slot = &slot, .n = published, .gap_ms = 2 };
    const t = try std.Thread.spawn(.{}, TickPlayer.run, .{&player});
    const stops = [_][]const u8{"S"};
    var early = EarlyStop{ .tok = &tok, .gate = .{ .stops = &stops }, .completion_skip_special = null };
    defer early.gate.deinit(a);
    var out = std.ArrayList(u32).empty;
    defer out.deinit(a);
    const end = try drainSettled(&sch, &slot, null, a, &out, &early);
    // Read exactly as the result builder does, before the player is joined.
    const settled_tokens = slot.completion_tokens;
    const settled_ns = slot.decode_ns;
    t.join();
    try std.testing.expectEqual(DrainEnd.stopped, end);
    try std.testing.expectEqual(@as(usize, 1), out.items.len);
    try std.testing.expectEqual(published, settled_tokens);
    try std.testing.expectEqual(@as(u64, 777), settled_ns);
}

test "a non-stream early stop reads usage and timing only after the tick that published its token settles" {
    try expectSettledAfterEarlyStop(1);
}

test "a non-stream early stop inside a speculative block settles the whole block's accounting" {
    try expectSettledAfterEarlyStop(4);
}

fn expectNonStreamStopsAt(stop: []const u8, pieces: []const []const u8, stop_token: usize) !void {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var ids: [16]u32 = undefined;
    for (pieces, 0..) |p, i| {
        ids[i] = @intCast(i + 1);
        try tok.id_to_token.put(ids[i], p);
    }
    var slot = FakeSlot{ .tokens = ids[0..pieces.len] };
    const stops = [_][]const u8{stop};
    var early = EarlyStop{ .tok = &tok, .gate = .{ .stops = &stops }, .completion_skip_special = null };
    defer early.gate.deinit(a);
    var out = std.ArrayList(u32).empty;
    defer out.deinit(a);
    const end = try drainSlotTokens(&slot, null, a, &out, &early);
    try std.testing.expectEqual(DrainEnd.stopped, end);
    try std.testing.expectEqual(stop_token, out.items.len);
    try std.testing.expect(slot.cancelled);
    try std.testing.expect(slot.stop_hit);
}

test "a non-stream generation ends on the token that completes a stop" {
    try expectNonStreamStopsAt("END", &.{ "a", "b", "E", "N", "D", "c", "d", "e" }, 5);
    try expectNonStreamStopsAt("\n\n", &.{ "x", "\n", "\n", "y", "z" }, 3);
}

test "a non-stream generation without a stop in its text runs to the end" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    try tok.id_to_token.put(1, "E");
    try tok.id_to_token.put(2, "N");
    var slot = FakeSlot{ .tokens = &.{ 1, 2, 1 } };
    const stops = [_][]const u8{"END"};
    var early = EarlyStop{ .tok = &tok, .gate = .{ .stops = &stops }, .completion_skip_special = null };
    defer early.gate.deinit(a);
    var out = std.ArrayList(u32).empty;
    defer out.deinit(a);
    try std.testing.expectEqual(DrainEnd.done, try drainSlotTokens(&slot, null, a, &out, &early));
    try std.testing.expectEqual(@as(usize, 3), out.items.len);
}

test "reasoning usage preserves totals and reports zero" {
    const a = std.testing.allocator;
    for ([_]usize{ 0, 7 }) |count| {
        const details = try reasoningDetailsJson(a, count);
        defer a.free(details);
        const raw = try formatChatUsage(a, 10, 20, 4, details);
        defer a.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(@as(i64, @intCast(count)), parsed.value.object.get("completion_tokens_details").?.object.get("reasoning_tokens").?.integer);
        try std.testing.expectEqual(@as(i64, 30), parsed.value.object.get("total_tokens").?.integer);
        try std.testing.expectEqual(@as(i64, 4), parsed.value.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer);
    }
}

test "reasoning usage tokenization ignores SSE boundaries" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.vocab.put("a", 1);
    try tok.vocab.put("b", 2);
    try tok.vocab.put("ab", 3);
    try tok.merge_ranks.put(.{ .left = "a", .right = "b" }, 0);
    var delivered = std.ArrayList(u8).empty;
    defer delivered.deinit(a);
    try delivered.appendSlice(a, "a");
    try delivered.appendSlice(a, "b");
    const details = try formatReasoningDetails(a, &tok, delivered.items);
    defer a.free(details);
    try std.testing.expectEqualStrings(",\"completion_tokens_details\":{\"reasoning_tokens\":1}", details);
    const empty = try formatReasoningDetails(a, &tok, "");
    defer a.free(empty);
    try std.testing.expectEqualStrings(",\"completion_tokens_details\":{\"reasoning_tokens\":0}", empty);
}
