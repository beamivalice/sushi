const std = @import("std");
const test_note = @import("test_note.zig");
const arch_load = @import("arch/load.zig");
const http = @import("http.zig");
const trunk = @import("trunk.zig");
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
const model_load_mod = @import("model_load.zig");
const vision_queue = @import("vision_queue.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const bills_mod = @import("bills.zig");
const forward_refusal_mod = @import("forward_refusal.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const Generator = generate_mod.Generator;
const LoadedModel = model_registry_mod.LoadedModel;
const http_conn = @import("http_conn.zig");
const Conn = http_conn.Conn;
const sendSseHeaders = http_conn.sendSseHeaders;
const request_policy = @import("request_policy.zig");
const withEffortNoticeJson = request_policy.withEffortNoticeJson;
const io_util = @import("io_util.zig");
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");
const tool_calls_mod = @import("tool_calls.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const nowMsMonotonic = io_util.nowMsMonotonic;
const Stopwatch = io_util.Stopwatch;
const nextCompletionId = io_util.nextCompletionId;
const server_mod = @import("server.zig");
const reply_mod = @import("reply.zig");
const tool_progress = @import("tool_progress.zig");
const server_state = @import("server_state.zig");
const arch_packs = @import("arch/packs.zig");
const anthropic = @import("anthropic.zig");
const responses_api = @import("responses_api.zig");
const completions = @import("completions.zig");
const FakeSlot = server_mod.FakeSlot;
const GEN_OOM_MSG = server_mod.GEN_OOM_MSG;
const MropeData = server_mod.MropeData;
const SSE_ALLOW_HEADERS_DEFAULT = http.SSE_ALLOW_HEADERS_DEFAULT;
const TickPlayer = server_mod.TickPlayer;
const appendRawToolDump = server_mod.appendRawToolDump;
const completionShowsToken = completions.completionShowsToken;
const completionText = completions.completionText;
const finishDetailsField = server_mod.finishDetailsField;
const getTimeoutNs = server_mod.getTimeoutNs;
const insufficient_free_memory_message = http.insufficient_free_memory_message;
const jsonEscape = http.jsonEscape;
const jsonEscapeOrEmpty = server_mod.jsonEscapeOrEmpty;
const logHttpSseData = http.logHttpSseData;
const memoryContextRefusalMessage = server_mod.memoryContextRefusalMessage;
const mtpCapable = server_mod.mtpCapable;
const not_enough_memory_message = http.not_enough_memory_message;
const promptOpensThink = server_mod.promptOpensThink;
const reasoningWithinBudget = server_mod.reasoningWithinBudget;
const sendAnthropicEvent = anthropic.sendAnthropicEvent;
const sendErrorResponse = http.sendErrorResponse;
const sendErrorResponseWithHeaders = http.sendErrorResponseWithHeaders;
const resident_models_cap_message = http.resident_models_cap_message;
const BUSY_RETRY_AFTER_SECS = http.BUSY_RETRY_AFTER_SECS;
const sendResponse = http.sendResponse;
const sendResponsesEvent = responses_api.sendResponsesEvent;
const shouldParseToolCalls = reply_mod.shouldParseToolCalls;
const splitConstrainedResponse = responses_api.splitConstrainedResponse;
const toolCallFinishReason = reply_mod.toolCallFinishReason;

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
    server_state.slot_memory_context_refusal = slot.memory_context_refusal;
    server_state.slot_media_refusal = slot.media_refusal;
    server_state.slot_forward_refusal = slot.forward_refusal;
    if (slot.shutdown_cancelled.load(.acquire)) return error.ServerShutdown;
    if (slot.errorNameIs("MediaRefused")) return error.MediaRefused;
    // A pre-prefill admission refusal is not an abandoned generation: the client gets a 400.
    if (slot.errorNameIs("PrefillDoesNotFit")) return error.PrefillDoesNotFit;
    if (slot.errorNameIs("TransientCapExceeded")) return error.TransientCapExceeded;
    if (slot.errorNameIs("PrefillChunkDoesNotFit")) return error.PrefillChunkDoesNotFit;
    if (slot.errorIsMemory()) return error.GenerationOutOfMemory;
    return error.GenerationFailed;
}

/// The message `error.TransientCapExceeded` sends: MLX refused an allocation past the transient memory cap.
const TRANSIENT_CAP_MSG = "This request ran out of GPU memory while it ran and was stopped; the server is still running.\nTo serve it:\n- Quit other apps or unload another model on the Models page, then retry.\n- Lower max_tokens or shorten the prompt: a smaller request needs less memory.";

/// The message `error.PrefillDoesNotFit` sends: refused before its first forward by the same
/// estimator the guard uses, re-asked after the hot cache gave back everything it could. It knows no pack, so it
/// offers none (the context refusal's text does, from the load's stamp).
pub const PREFILL_NOFIT_MSG = "This request does not fit in GPU memory, even with the prefix cache emptied; nothing ran.\nTo serve it:\n- Quit other apps or unload another model on the Models page, then retry.\n- Lower max_tokens or shorten the prompt.";

/// The message `error.PrefillChunkDoesNotFit` sends when the arch left no text of its own.
pub const PREFILL_CHUNK_NOFIT_MSG = "This prompt ran out of GPU memory partway through reading it.\nTo serve it:\n- Quit other apps or unload another model on the Models page, then retry.\n- Shorten the prompt.\n- Restart sushi with a smaller --prefill-chunk (a launch flag; Settings has none): a narrower chunk needs less memory.";

/// The message `error.EmptyPrompt` sends: generation continues from the prompt's last token.
pub const EMPTY_PROMPT_MSG = "The prompt tokenizes to zero tokens; send non-empty text to continue.";

/// A generation failure as the wire sees it: one status, one message, both dialects' type.
pub const GenErrorWire = struct {
    status_line: []const u8,
    code: u32,
    openai_type: []const u8,
    anthropic_type: []const u8,
    /// Owned: a refusal's text reaches the client whole, whatever its length.
    message: []const u8,

    pub fn deinit(self: GenErrorWire, allocator: std.mem.Allocator) void {
        allocator.free(self.message);
    }

    /// The answer in every dialect's frame, built by `http.ErrorEnvelope`.
    pub fn envelope(self: GenErrorWire) http.ErrorEnvelope {
        return .{ .status = self.code, .types = .{ .openai = self.openai_type, .anthropic = self.anthropic_type }, .message = self.message, .code = self.code };
    }
};

/// The mapping from a failed generation to what the client is told. It existed only on the
/// non-streaming arms; streaming arms wrote a bare `server_error` into an SSE frame, so agents
/// never saw the actionable 400/503. The unknown arm keeps the error name at a mapped 500.
pub fn mapGenerationError(allocator: std.mem.Allocator, err: anyerror) !GenErrorWire {
    const context_refusal = server_state.slot_memory_context_refusal;
    server_state.slot_memory_context_refusal = null;
    const media_refusal = server_state.slot_media_refusal;
    server_state.slot_media_refusal = null;
    const forward_refusal = server_state.slot_forward_refusal;
    server_state.slot_forward_refusal = null;
    var said: std.Io.Writer.Allocating = .init(allocator);
    defer said.deinit();
    var wire: GenErrorWire = switch (err) {
        // The media encode the prefix restore left to do could not run: the same text and status the
        // connection thread gave before the encode moved behind the restore.
        error.MediaRefused => if (media_refusal) |*r| .{
            .status_line = if (r.server_error) "500 Internal Server Error" else "400 Bad Request",
            .code = if (r.server_error) 500 else 400,
            .openai_type = if (r.server_error) "server_error" else "invalid_request_error",
            .anthropic_type = if (r.server_error) "api_error" else "invalid_request_error",
            .message = r.text(),
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
        // Refused mid-prefill for memory: the arch's text names the chunk, the shortfall and what to change.
        error.PrefillChunkDoesNotFit => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = if (forward_refusal) |*r| r.text() else PREFILL_CHUNK_NOFIT_MSG,
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
            .message = if (context_refusal) |r| blk: {
                try bills_mod.writeMemoryContextRefusal(&said.writer, r);
                break :blk said.written();
            } else PREFILL_NOFIT_MSG,
        },
        // Refused at submit, so every surface answers it the same way.
        error.EmptyPrompt => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = EMPTY_PROMPT_MSG,
        },
        // The shutdown drain ended a request that was queued or running, or refused one at submit.
        error.ServerShutdown, error.Shutdown => .{
            .status_line = "503 Service Unavailable",
            .code = 503,
            .openai_type = http.shutting_down_type,
            .anthropic_type = "api_error",
            .message = http.shutting_down_message,
        },
        error.GenerationFailed => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = "generation failed",
        },
        error.ChatNeedsUserTurn => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = "messages hold no user turn to answer: this model's chat template needs one",
        },
        error.ChatRoleUnknown => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = blk: {
                try said.writer.print("a message role must be one of system, user, assistant or tool; got \"{s}\"", .{chat_mod.roleRefusal()});
                break :blk said.written();
            },
        },
        error.ContinuationThoughtInContent => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = "the continued assistant message holds a think close tag in its text: send its thought in reasoning_content (a thinking block on /v1/messages) and only the visible reply as content",
        },
        error.ContinuationDropsThought => .{
            .status_line = "400 Bad Request",
            .code = 400,
            .openai_type = "invalid_request_error",
            .anthropic_type = "invalid_request_error",
            .message = "the continued assistant message holds only its thought, and this model's chat template writes no thought for it in this request (a template whose thinking-off turns carry none drops it): turn thinking on, or send the visible reply so far as content",
        },
        error.ChatTemplateUnusable => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = blk: {
                try said.writer.print("chat template unusable: {s}", .{chat_mod.templateRefusal()});
                break :blk said.written();
            },
        },
        else => .{
            .status_line = "500 Internal Server Error",
            .code = 500,
            .openai_type = "server_error",
            .anthropic_type = "api_error",
            .message = blk: {
                try said.writer.print("Internal server error: {s}", .{@errorName(err)});
                break :blk said.written();
            },
        },
    };
    wire.message = try allocator.dupe(u8, wire.message);
    return wire;
}

/// Which stream a generative surface ends with its error event.
pub const ErrorSurface = union(enum) {
    openai,
    completions,
    anthropic,
    responses: *u64,
};

/// The error arm for every generative surface, streaming or not: before the SSE head the
/// stream gets the same response its non-streaming twin would; past it the same `type` and
/// `message` ride the surface's terminal `error` event.
pub fn sendGenerationError(allocator: std.mem.Allocator, stream: *Conn, err: anyerror, surface: ErrorSurface) !void {
    const w = try mapGenerationError(allocator, err);
    defer w.deinit(allocator);

    if (!stream.sse_headers_sent) return http.sendError(allocator, stream, w.status_line, .{ .openai = w.openai_type, .anthropic = w.anthropic_type }, w.message, w.code, "");

    switch (surface) {
        .openai, .completions => {
            const body = try w.envelope().openaiStreamFrame(allocator, surface == .completions);
            defer allocator.free(body);
            try stream.writeAllNoFlush(body);
            try stream.writeAll("data: [DONE]\n\n");
        },
        .anthropic => {
            const body = try w.envelope().anthropicStreamEvent(allocator);
            defer allocator.free(body);
            // Anthropic streams have no `[DONE]`; the `error` event is the terminator.
            try sendAnthropicEvent(stream, "error", body);
        },
        .responses => |seq| {
            const body = try w.envelope().responsesStreamEvent(allocator);
            defer allocator.free(body);
            try sendResponsesEvent(allocator, stream, seq, "error", body);
            // Same gate as the success path: no `[DONE]` over the WS bridge.
            if (stream.ws_mode == null) try stream.writeAll("data: [DONE]\n\n");
        },
    }
}

/// A client that vanished mid-generation, or a slot the cancel API ended, did not hit a token cap.
fn nonStreamFinishReason(end: DrainEnd, slot_reason: []const u8) []const u8 {
    return switch (end) {
        .client_gone => "client_disconnect",
        .cancelled => "cancelled",
        else => slot_reason,
    };
}

/// A generation result as the reply end handling reads it.
pub fn generatedOf(r: *const generate_mod.GenerationResult) reply_mod.Generated {
    return .{ .text = r.text, .source = r.stop_source, .finish_reason = r.finish_reason, .finish_details = r.finish_details };
}

pub const visibleToolPreamble = think_tags.visibleToolPreamble;

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

/// What a request gives the server to continue a reply whose turn ended inside its thought
/// (`server.continue_after_thought`): the render a client-side `continue_final_message` of that thought-only reply gives.
pub const ThoughtResume = struct {
    chat_config: *const chat_mod.ChatConfig,
    messages: []const chat_mod.Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// The reply's end handling, as its surface settles it (`reply.Ask`).
    stops: []const []const u8,
    has_tools: bool,
    allow_parallel: bool,
    /// The request's tool_choice: a forced call stays forced in the continuation.
    tool_choice: tool_calls_mod.ToolChoice = .auto,
    /// Why this request is not continued; `finish_details` names it when its turn ends inside the thought.
    kept: ?reply_mod.ThoughtKept = null,

    /// What a request that cannot be continued is kept for, or null when it can be.
    pub fn keptFor(continue_final: bool, images: bool, constrained: bool) ?reply_mod.ThoughtKept {
        if (continue_final) return .continuation_request;
        if (images) return .images;
        if (constrained) return .json_schema;
        return null;
    }
};

const ThoughtStep = union(enum) {
    /// The turn did not end inside its thought, or the setting is off: the reply is the generation's own.
    none,
    /// It did, and is not continued: the `finish_details` naming why.
    kept: []const u8,
    /// The continuation's slot, and the close tag the reply carries before its tokens (empty when the model closed it).
    continued: struct { slot: *scheduler_mod.Slot, bridge: []u32 },
};

/// After `slot`'s turn ends: whether it ended on its end-of-turn token with a thought and nothing else, and if so the
/// once-only continuation, submitted, or why none runs. `params` is what `slot` was submitted with.
/// `force` holds the continuation's forced call, if any, until its slot completes.
fn continueAfterThought(allocator: std.mem.Allocator, sch: *scheduler_mod.Scheduler, tok: *const Tokenizer, r: *const ThoughtResume, slot: *scheduler_mod.Slot, generated: []const u32, params: scheduler_mod.SubmitParams, force: *?generate_mod.CallForce) !ThoughtStep {
    if (!std.mem.eql(u8, slot.finish_reason, "stop") or slot.finish_details != null) return .none;
    const stop = slot.stop_token orelse return .none;
    if (!generate_mod.isEosId(stop, slot.eos_token_ids) or !server_state.continueAfterThought()) return .none;
    const text = try tok.decode(allocator, generated);
    defer allocator.free(text);
    const ask: reply_mod.Ask = .{
        .stops = r.stops,
        .normalize = true,
        .has_tools = r.has_tools,
        .tools_json = r.tools_json,
        .allow_parallel = r.allow_parallel,
        .dialect = if (r.has_tools) try chat_mod.callDialect(allocator, r.chat_config) else null,
        .opens_think = promptOpensThink(allocator, tok, params.prompt_ids),
    };
    const thought = (try reply_mod.soleThought(allocator, text, ask)) orelse return .none;
    defer allocator.free(thought);
    if (r.kept) |k| return keptThought(slot, k);
    const turns = try std.mem.concat(allocator, chat_mod.Message, &.{ r.messages, &.{.{ .role = "assistant", .content = "", .reasoning_content = thought }} });
    defer allocator.free(turns);
    const prompt = chat_mod.formatChat(allocator, tok, turns, r.chat_config, r.tools_json, r.tool_choice_instruction, r.enable_thinking, r.effort, true) catch |err| switch (err) {
        error.OutOfMemory => return err,
        error.ContinuationDropsThought => return keptThought(slot, .template_drops_thought),
        else => {
            log.warn("  [thought] request {d}: the continuation did not render ({s})\n", .{ slot.request_id, @errorName(err) });
            return keptThought(slot, .render_failed);
        },
    };
    defer allocator.free(prompt);
    // The continuation spends what the thought left of max_tokens, inside the prompt plus max_tokens the request was billed.
    const billed = params.prompt_ids.len + params.max_tokens;
    const left = @min(params.max_tokens -| slot.completion_tokens, billed -| prompt.len);
    if (left == 0) return keptThought(slot, .max_tokens);
    var next = params;
    next.prompt_ids = prompt;
    next.full_prompt = prompt;
    next.cached_tokens = 0;
    next.max_tokens = left;
    next.sampling = continuationSampling(allocator, tok, r, prompt, params.sampling, force);
    const bridge = if (think_tags.indexOfThinkCloseTag(text, 0) == null) try tok.encode(allocator, think_tags.BARE_THINK_CLOSER) else try allocator.alloc(u32, 0);
    errdefer allocator.free(bridge);
    const cont = try sch.submit(next);
    log.info("  [thought] request {d} ended its turn inside its thought: continued once after its closed think block as request {d} ({d} prompt tokens, {d} max_tokens left)\n", .{ slot.request_id, cont.request_id, prompt.len, left });
    return .{ .continued = .{ .slot = cont, .bridge = bridge } };
}

/// The request's sampling for its continuation, which starts after the closed thought.
fn continuationSampling(allocator: std.mem.Allocator, tok: *const Tokenizer, r: *const ThoughtResume, prompt: []const u32, base: sampler.SamplingParams, force: *?generate_mod.CallForce) sampler.SamplingParams {
    var s = base;
    s.think_bound = null;
    s.think_penalty.phase = .after;
    // The prompt closed the thought, so a forced call is armed as an answer's: committed at the first token.
    force.* = server_mod.armCallForce(allocator, tok, r.chat_config, prompt, false, r.tool_choice, r.tools_json);
    s.call_force = if (force.*) |*f| f else null;
    return s;
}

fn keptThought(slot: *const scheduler_mod.Slot, k: reply_mod.ThoughtKept) ThoughtStep {
    log.info("  [thought] request {d} ended its turn inside its thought: not continued ({s})\n", .{ slot.request_id, @tagName(k) });
    return .{ .kept = k.details() };
}

/// What the continuation reused: its prompt's tokens the prefix cache restored, and its prefill time.
fn logContinuation(cont: *const scheduler_mod.Slot) void {
    log.debug("  [thought] request {d}: {d} of {d} prompt tokens restored, prefill {d} us\n", .{ cont.request_id, cont.cached_tokens, cont.prompt_tokens, cont.prefill_ns / std.time.ns_per_us });
}

/// The `finish_details` of a reply its continuation ended: the continuation's own loop stop, or that it ran.
fn continuedDetails(last: ?[]const u8) []const u8 {
    if (last) |d| if (std.mem.eql(u8, d, "repetition_loop")) return reply_mod.thought_loop;
    return reply_mod.thought_continued;
}

const DrainEnd = enum { done, client_gone, stopped, cancelled, failed };

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
            .cancelled => return .cancelled,
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

/// `drainSlotTokens`, then, once the slot is cancelled (an early stop, the cancel API, a client that left), wait out
/// the inference pass still publishing and accounting it: its statistics are plain fields only a quiet slot lets the
/// caller read.
fn drainSettled(sch: *scheduler_mod.Scheduler, slot: *scheduler_mod.Slot, conn: ?*Conn, allocator: std.mem.Allocator, output_ids: *std.ArrayList(u32), early: ?*EarlyStop) !DrainEnd {
    const end = try drainSlotTokens(slot, conn, allocator, output_ids, early);
    if (slot.cancelled.load(.acquire)) sch.quiesce(slot);
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
    /// Wave 1.A: per-request KV-quant override; null = inherit scheduler default.
    kv_quant_override: ?trunk.KVQuantConfig,
    /// When non-null, the peer socket is probed on idle wakeups during the
    /// wait — a vanished client cancels the slot (aborting its prefill)
    /// instead of grinding out a ghost generation nobody will read.
    conn: ?*Conn,
    stop: StopSpec,
    /// Continue a turn that ends inside its thought (`server.continue_after_thought`); null on a surface with none.
    thought: ?*const ThoughtResume,
) !generate_mod.GenerationResult {
    const params: scheduler_mod.SubmitParams = .{
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
        .kv_quant_config = kv_quant_override,
    };
    const slot = try sch.submit(params);
    defer sch.complete(slot);

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var early: ?EarlyStop = if (stop.stops.len > 0) .{
        .tok = tok,
        .gate = .{ .stops = stop.stops },
        .completion_skip_special = stop.completion_skip_special,
    } else null;
    defer if (early) |*e| e.gate.deinit(allocator);
    var end = try drainSettled(sch, slot, conn, allocator, &output_ids, if (early) |*e| e else null);
    if (end == .failed) return slotFailure(slot);
    // The part a continuation after the thought generates: its forced call, its slot, and where its tokens start in
    // `output_ids`. The force outlives the slot.
    var force: ?generate_mod.CallForce = null;
    defer if (force) |*f| f.deinit(allocator);
    var cont: ?*scheduler_mod.Slot = null;
    defer if (cont) |c| sch.complete(c);
    var cont_from: usize = 0;
    var bridged: usize = 0;
    var details = slot.finish_details;
    if (end == .done) if (thought) |r| switch (try continueAfterThought(allocator, sch, tok, r, slot, output_ids.items, params, &force)) {
        .none => {},
        .kept => |d| details = d,
        .continued => |c| {
            defer allocator.free(c.bridge);
            cont = c.slot;
            try output_ids.appendSlice(allocator, c.bridge);
            bridged = c.bridge.len;
            cont_from = output_ids.items.len;
            end = try drainSettled(sch, c.slot, conn, allocator, &output_ids, if (early) |*e| e else null);
            if (end == .failed) return slotFailure(c.slot);
            logContinuation(c.slot);
            details = continuedDetails(c.slot.finish_details);
        },
    };
    const last = cont orelse slot;
    // An early stop or a cancel reports the tokens it returned, not the ones decoded past its end.
    const returned = end == .stopped or end == .cancelled;
    const generated: u32 = slot.completion_tokens + if (cont) |c| c.completion_tokens else 0;
    const prefill_ns = slot.prefill_ns + if (cont) |c| c.prefill_ns else 0;
    const decode_ns = slot.decode_ns + if (cont) |c| c.decode_ns else 0;

    // The scheduler measures prefill_ns / decode_ns per-slot directly. Pull
    // those instead of the old single-wall-clock approximation, which
    // double-counted total time into both phases.
    const prefill_tps = generate_mod.prefillTokensPerSec(slot.prompt_tokens, slot.cached_tokens, prefill_ns);
    const decode_tps = generate_mod.tokensPerSec(generated, decode_ns);

    // A loop cut is a truncation, and the degenerate span is the part the
    // client must not get: an agent re-sends the cut turn as history, the
    // model reads its own loop back and resumes it — five loop-stops in a
    // row, each firing sooner than the last (live 2026-08-05, under pi).
    // Only reachable non-streaming: a delta cannot be retracted, so a
    // streaming client has already received the tail.
    const emit_ids = loopTrimmedIds(output_ids.items, if (last.loop_trim_start) |t| cont_from + t else null);
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

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = slot.prompt_tokens,
        .completion_tokens = if (returned) @intCast(token_ids.len - bridged) else generated,
        .finish_reason = nonStreamFinishReason(end, last.finish_reason),
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .prefill_ns = prefill_ns,
        .decode_ns = decode_ns,
        .cached_tokens = slot.cached_tokens,
        .finish_details = details,
        .constraint_payload_byte = constraint_payload_byte,
        .restore_ns = slot.restore_ns,
        .spec_mode = slot.spec_mode,
    };
}

/// Generation is authoritative for constrained responses: when the generator
/// published a payload boundary, the split is a byte cut at that offset —
/// reasoning is everything before it (delimiter markup stripped, whitespace
/// policy matched to the streaming suppression), content is everything after,
/// UNREParsed. A standalone re-scan of the payload would read marker text
/// inside JSON string data ("the close delimiter") as protocol structure and
/// truncate valid JSON. Without a boundary the existing parser applies.
/// A chat stream's content delta.
fn sendContentDelta(allocator: std.mem.Allocator, stream: *Conn, chat_id: i64, model_name: []const u8, bytes: []const u8) !void {
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = bytes }, null, null, null);
}

fn emitConstrainedChat(a: std.mem.Allocator, stream: *Conn, id: i64, model: []const u8, d: *rp_mod.Delivery) !void {
    if (d.reasoning.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = null, .reasoning_content = d.reasoning.items }, null, null, null);
    }
    if (d.content.items.len > 0) {
        try sendSSEChunk(a, stream, id, model, .{ .role = null, .content = d.content.items }, null, null, null);
    }
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
    kv_quant_override: ?trunk.KVQuantConfig,
    /// Iteration 1: tokenize_ns from the parent handleChatCompletions, so
    /// the non-streaming chat response carries `timings.tokenize_ms`.
    tokenize_ns: u64,
    thought: ?*const ThoughtResume,
) !void {
    // Phase A8: this handler owns the vision array on entry; ownership
    // transfers to the slot on submit (the scheduler's `Slot.deinit` frees
    // it). Nulled before transfer so the early-return defer is a no-op.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    var timer = Stopwatch.init(stream.io);

    // Speculative-decoding dispatch (priority: DFlash > MTP > regular).
    //   1. Drafter wins if loaded AND requested.
    //   2. PLD next if requested AND no grammar constraint
    //      (constrained decode requires per-token state advancement).
    //   3. Otherwise the regular pipeline.
    const spec = requestSpecModes(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling);
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;

    // Transfer vision ownership into the slot via the scheduler.
    const slot_ve: ?*vision_queue.DeferredMedia = blk: {
        const v = ve_local;
        ve_local = null;
        break :blk v;
    };
    // Propagates to `handleChatCompletions`' one error arm, shared with the streaming twin.
    const result = try nonStreamingViaScheduler(allocator, server_state.global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, has_tools, enable_thinking, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), slot_ve, media_chain, cache_key, mrope, kv_quant_override, stream, .{ .stops = stop_sequences }, thought);
    defer result.deinit(allocator);

    // Template-opened think block (Qwen 3.5/3.6): unclosed output is reasoning.
    // Not ANDed with enable_thinking: generated reasoning is always DELIVERED,
    // never paid-for-and-dropped — thinking-off is enforced on the PROMPT side
    // (think_tags.noThinkTailSuffix commits the channel), so any reasoning that
    // still shows up here is real work the client gets to see.
    const opens_think = promptOpensThink(allocator, tok, prompt_ids);

    // The stop that ended the generation, the text it leaves and the calls in it (`reply.settle`).
    const settled = try reply_mod.settle(allocator, generatedOf(&result), .{
        .stops = stop_sequences,
        .normalize = !generate_mod.payloadRouted(sampling.constraint),
        .has_tools = has_tools,
        .tools_json = tools_json,
        .allow_parallel = allow_parallel_tools,
        .dialect = if (has_tools) try chat_mod.callDialect(allocator, lm.chat_config.?) else null,
        .opens_think = opens_think,
    });
    defer settled.deinit(allocator);
    const final_text = settled.text;
    const finish_reason = settled.finish_reason;

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
        log.debug("  checked {d} bytes of generated text for tool calls\n", .{final_text.len});
        if (settled.reply) |tool_reply| {
            const tool_calls = tool_reply.calls;

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
                const tc_id = try io_util.callId(stream.io, allocator, "call");
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
            const tc_preamble = tool_reply.preamble;
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

            const tc_finish_details = try reply_mod.detailsField(allocator, "finish_details", settled.declined, finishDetailsField(toolCallFinishReason(finish_reason), result.finish_details));
            defer allocator.free(tc_finish_details);
            // finish_reason always carries finish_details beside it.
            const response = try std.fmt.allocPrint(allocator,
                \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s},"tool_calls":{s}}},"logprobs":null,"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
            , .{
                nextCompletionId(&server_state.completion_id_counter, nowMs(stream.io)),
                nowSecs(stream.io),
                model_name,
                tc_content_json,
                tc_reasoning_json,
                tc_buf.items,
                toolCallFinishReason(finish_reason),
                tc_finish_details,
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

    const finish_details = try reply_mod.detailsField(allocator, "finish_details", settled.declined, finishDetailsField(finish_reason, result.finish_details));
    defer allocator.free(finish_details);
    // finish_reason always carries finish_details beside it.
    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-{d}","object":"chat.completion","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"message":{{"role":"assistant","content":{s}{s}}},"logprobs":null,"finish_reason":"{s}"{s}}}],"usage":{s}{s}}}
    , .{
        nextCompletionId(&server_state.completion_id_counter, nowMs(stream.io)),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        reasoning_json,
        finish_reason,
        finish_details,
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
    /// Tokens `nextOrIdle` has handed out: what a client stop sequence or a cancel leaves as the completion count.
    delivered: u32 = 0,
    /// The slot was cancelled under the reader before it finished (`NextResult.cancelled`).
    cancelled: bool = false,
    /// Continues a turn that ends inside its thought (`armThought`); `slot` is then the part generating now.
    thought: ?ThoughtArm = null,

    const ThoughtArm = struct {
        sch: *scheduler_mod.Scheduler,
        tok: *const Tokenizer,
        render: *const ThoughtResume,
        params: scheduler_mod.SubmitParams,
        /// The thought part's tokens, read when its turn ends.
        ids: std.ArrayList(u32) = .empty,
        /// The thought part's slot once the continuation runs: the handler that submitted it completes it.
        first: ?*scheduler_mod.Slot = null,
        /// The close tag handed out before the continuation's tokens.
        bridge: []u32 = &.{},
        bridge_at: usize = 0,
        /// Why the turn that ended inside its thought was not continued.
        kept: ?[]const u8 = null,
        /// The continuation's forced call, freed once its slot completes.
        force: ?generate_mod.CallForce = null,
    };

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
        if (self.thought) |*t| {
            t.ids.deinit(allocator);
            allocator.free(t.bridge);
            if (t.first != null) t.sch.complete(self.slot.?);
            if (t.force) |*f| f.deinit(allocator);
        }
    }

    /// Continue the turn once when it ends inside its thought; `params` is what the stream's slot was submitted with.
    pub fn armThought(self: *StreamingTokenStream, sch: *scheduler_mod.Scheduler, tok: *const Tokenizer, render: ?*const ThoughtResume, params: scheduler_mod.SubmitParams) void {
        const r = render orelse return;
        self.thought = .{ .sch = sch, .tok = tok, .render = r, .params = params };
    }

    /// At the end of the thought part: switch to the continuation when its turn ended inside its thought.
    fn resumeThought(self: *StreamingTokenStream, allocator: std.mem.Allocator) !bool {
        const t = if (self.thought) |*t| t else return false;
        if (t.first != null) return false;
        const s = self.slot.?;
        switch (try continueAfterThought(allocator, t.sch, t.tok, t.render, s, t.ids.items, t.params, &t.force)) {
            .none => return false,
            .kept => |d| {
                t.kept = d;
                return false;
            },
            .continued => |c| {
                t.first = s;
                t.bridge = c.bridge;
                self.slot = c.slot;
                return true;
            },
        }
    }

    /// `finalize` after the request's own stop sequence ended the stream: the slot is cancelled and waited
    /// out, and usage counts the tokens delivered, not the ones a speculative block decoded past the stop.
    pub fn finalizeStopped(self: *StreamingTokenStream, sch: *scheduler_mod.Scheduler) void {
        if (self.slot) |s| s.cancelOnStop();
        self.finalize(sch);
        self.completion_tokens = self.delivered;
    }

    /// Snapshot prompt/completion tokens + finish reason from the underlying
    /// source so callers can read them after the loop exits without having
    /// to know which source was active. Safe to call multiple times; idempotent.
    /// A cancelled slot is waited out first: a pass may still write the fields read here.
    pub fn finalize(self: *StreamingTokenStream, sch: *scheduler_mod.Scheduler) void {
        if (self.slot) |s| {
            if (s.cancelled.load(.acquire)) sch.quiesce(s);
            const first: ?*scheduler_mod.Slot = if (self.thought) |t| t.first else null;
            if (first != null) logContinuation(s);
            const head = first orelse s;
            self.prompt_tokens = head.prompt_tokens;
            self.cached_tokens = head.cached_tokens;
            // A cancel ends the stream on what it handed out; no finish set the slot's reason.
            self.completion_tokens = if (self.cancelled) self.delivered else s.completion_tokens + if (first) |f| f.completion_tokens else 0;
            self.finish_reason = if (self.cancelled) "cancelled" else s.finish_reason;
            self.finish_details = if (first != null) continuedDetails(s.finish_details) else if (self.thought) |t| t.kept orelse s.finish_details else s.finish_details;
            self.prefill_ns = s.prefill_ns + if (first) |f| f.prefill_ns else 0;
            self.decode_ns = s.decode_ns + if (first) |f| f.decode_ns else 0;
        } else if (self.gen) |g| {
            self.prompt_tokens = g.prompt_tokens;
            self.completion_tokens = g.completion_tokens;
            self.finish_reason = g.finish_reason;
        }
    }

    /// The finalized stream as the result a non-stream generation returns: the one place the stats are copied.
    /// `gate_matched` is the stop the stream's `Gate` matched (null when none): `text` is what the gate delivered, cut before it.
    pub fn result(self: *const StreamingTokenStream, text: []u8, token_ids: []u32, finish_reason: []const u8, constraint_payload_byte: ?usize, gate_matched: ?[]const u8) generate_mod.GenerationResult {
        return .{
            .text = text,
            .token_ids = token_ids,
            .prompt_tokens = self.prompt_tokens,
            .completion_tokens = self.completion_tokens,
            .finish_reason = finish_reason,
            .prefill_tps = generate_mod.prefillTokensPerSec(self.prompt_tokens, self.cached_tokens, self.prefill_ns),
            .decode_tps = generate_mod.tokensPerSec(self.completion_tokens, self.decode_ns),
            .prefill_ns = self.prefill_ns,
            .decode_ns = self.decode_ns,
            .cached_tokens = self.cached_tokens,
            .finish_details = self.finish_details,
            .constraint_payload_byte = constraint_payload_byte,
            .restore_ns = if (self.slot) |sl| sl.restore_ns else null,
            .spec_mode = if (self.slot) |sl| sl.spec_mode else .none,
            .stop_source = .{ .gated = gate_matched },
        };
    }

    const NextOrIdle = union(enum) { token: u32, done, idle };

    /// `next` with an idle timeout (scheduler path only): returns `.idle`
    /// after `timeout_ms` with no progress so the caller can poll the peer
    /// socket and emit SSE keepalives during long prefills. The local-
    /// generator path computes synchronously and never idles.
    pub fn nextOrIdle(self: *StreamingTokenStream, allocator: std.mem.Allocator, timeout_ms: i64) !NextOrIdle {
        while (self.slot) |s| {
            if (self.thought) |*t| if (t.bridge_at < t.bridge.len) {
                t.bridge_at += 1;
                return .{ .token = t.bridge[t.bridge_at - 1] };
            };
            if (self.finished) return .done;
            const nr = s.waitNextTimeout(timeout_ms) orelse return .idle;
            switch (nr) {
                .token => |t| {
                    self.delivered += 1;
                    if (self.thought) |*th| if (th.first == null) try th.ids.append(allocator, t);
                    return .{ .token = t };
                },
                .done => {
                    if (try self.resumeThought(allocator)) continue;
                    self.finished = true;
                    return .done;
                },
                .cancelled => {
                    self.finished = true;
                    self.cancelled = true;
                    return .done;
                },
                .err => return slotFailure(s),
            }
        }
        if (try self.next(allocator)) |t| {
            self.delivered += 1;
            return .{ .token = t };
        }
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
                .cancelled => {
                    self.finished = true;
                    self.cancelled = true;
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
/// `sched_policy.specInitWiring`. A loaded DFlash sidecar is an explicit choice
/// (`--drafter` or the pack's own `drafter/`) while the MTP head ships with
/// the checkpoint, so the sidecar wins; `--no-drafter` / `enable_drafter:
/// false` hand the round back to MTP. A grammar constraint disables every mode; a repeat or presence
/// penalty disables only the sidecar and the drafter, since MTP shifts its verify rows for it.
pub fn requestSpecModes(
    enable_drafter: bool,
    enable_mtp: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    sampling: sampler.SamplingParams,
) RequestSpec {
    const spec_ok = !mtp_plan.mtpUnsupported(sampling);
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
pub fn noteMtpRefused(enable_mtp: bool, mtp_loaded: bool, sampling: sampler.SamplingParams) void {
    if (!enable_mtp or !mtp_loaded) return;
    if (mtp_plan.mtpRefusal(sampling)) |why| log.info("  mtp=disabled ({s})\n", .{why});
}

pub fn pickStreamMode(
    enable_drafter: bool,
    enable_mtp: bool,
    dflash_loaded: bool,
    mtp_loaded: bool,
    sampling: sampler.SamplingParams,
) StreamMode {
    const r = requestSpecModes(enable_drafter, enable_mtp, dflash_loaded, mtp_loaded, sampling);
    if (r.use_mtp) return .mtp;
    if (r.use_drafter) return .drafter;
    return .regular;
}

test "requestSpecModes: a loaded DFlash sidecar outranks the checkpoint's MTP head" {
    const plain = sampler.SamplingParams{};
    // dflash + mtp both loaded and enabled -> dflash rides the drafter arm.
    var r = requestSpecModes(true, true, true, true, plain);
    try std.testing.expect(r.use_drafter and !r.use_mtp);
    try std.testing.expectEqual(StreamMode.drafter, pickStreamMode(true, true, true, true, plain));
    // No sidecar loaded -> MTP keeps its rank.
    r = requestSpecModes(true, true, false, true, plain);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, false, true, plain));
    // enable_drafter:false hands the round back to MTP.
    r = requestSpecModes(false, true, true, true, plain);
    try std.testing.expect(r.use_mtp and !r.use_drafter);
    // A grammar disables every spec mode, dflash included.
    var dummy: generate_mod.Constraint = .{ .grammar = undefined, .token_bytes = undefined, .mask_buf = &.{} };
    r = requestSpecModes(true, true, true, true, .{ .constraint = &dummy });
    try std.testing.expect(!r.use_mtp and !r.use_drafter);
    // Nothing loaded -> regular decode.
    r = requestSpecModes(true, true, false, false, plain);
    try std.testing.expect(!r.use_mtp and !r.use_drafter);
    try std.testing.expectEqual(StreamMode.regular, pickStreamMode(true, true, false, false, plain));
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
    kv_quant_override: ?trunk.KVQuantConfig,
    /// Iteration 1: tokenize_ns measured by the request handler before
    /// dispatching here. Surfaced via `timings.tokenize_ms` on the final
    /// usage SSE chunk so streaming clients see the same metric as
    /// non-streaming.
    tokenize_ns: u64,
    thought: ?*const ThoughtResume,
) !void {
    // Vision array ownership: held by this handler on entry, transfers to
    // the slot on submit (slot.deinit frees). Nulled before transfer so
    // the early-return defer is a no-op.
    var ve_local = deferred_media;
    defer if (ve_local) |d| d.deinit();

    const chat_id = nextCompletionId(&server_state.completion_id_counter, nowMs(stream.io));

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
    const stream_mode = pickStreamMode(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling);
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});
    noteMtpRefused(enable_mtp, mtpCapable(lm), sampling);

    // Scheduler's inference thread runs prefill + per-tick decode (regular
    // / PLD / drafter) and pushes generated tokens into the slot's output
    // ring; this thread reads via `slot.waitNext` through the
    // `StreamingTokenStream.initFromSlot` adapter.
    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| server_state.global_scheduler.?.complete(s);

    // Transfer vision ownership into the slot.
    const slot_ve_s = ve_local;
    ve_local = null;
    const sch = server_state.global_scheduler.?;
    const params: scheduler_mod.SubmitParams = .{
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
        .deferred_media = slot_ve_s,
        .vision_key = prefix_cache_mod.chainKey(media_chain),
        .media_chain = media_chain,
        .cache_key = cache_key,
        .mrope_pos = mrope.pos,
        .mrope_total = mrope.total,
        .mrope_delta = mrope.delta,
        .kv_quant_config = kv_quant_override,
    };
    slot_handle = try sch.submit(params);
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);
    defer ts.deinit(allocator);
    ts.armThought(sch, tok, thought, params);

    // Send SSE headers (no Content-Length — we stream until done)
    try sendSseHeaders(stream, "chat.completions", SSE_ALLOW_HEADERS_DEFAULT);

    // First chunk: role announcement
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = "assistant", .content = "" }, null, null, null);

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
    // The tool gate and the held thought are memoized the same way; the thought scan also resets
    // whenever `text_delivered` moves, for it reads only the undelivered rest.
    var tool_scan: think_tags.ToolHoldScan = .{};
    var thought_scan: think_tags.ThoughtScan = .{};
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
    // The call the stream opens to the client while it is still being written (`tool_progress`).
    var tool_stream: tool_progress.Progress = .{};
    defer tool_stream.deinit(allocator);
    // The `finish_details` field naming a call the reply declined, set where the end is settled.
    var declined_field: []const u8 = "";
    defer allocator.free(declined_field);
    // Constrained requests route through the protocol, not the response-side
    // close scan. The stream seeds the think arm while the payload has not
    // begun (a generated opener must strip, not leak), and flips straight to
    // the content arm when the generator reports a DIRECT answer — the whole
    // point is that no reasoning block exists to scan for.
    const constrained_proto = generate_mod.payloadRouted(sampling.constraint);
    // A stream starts in a think block from the rendered prompt, never from the request flag.
    var in_think_block = prompt_opened_think and !constrained_proto;
    var delivery: ?rp_mod.Delivery = if (constrained_proto) rp_mod.Delivery.init(sampling.constraint.?.proto.?) else null;
    defer if (delivery) |*d| d.deinit(allocator);
    var think_closed = false; // a complete think block was already split+emitted this stream
    // The content arm delivers the non-stream content bytes (think_tags.ContentStream).
    var content: think_tags.ContentStream = .{ .mark = .{ .on = !gated_stream }, .opens_thoughts = !gated_stream };
    defer content.deinit(allocator);
    var thought_mark: think_tags.MarkupCut = .{ .on = !gated_stream };
    var join: think_tags.ThoughtJoin = .{};
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
    // The reply ended: one more turn, with no token, reads what the arms held for bytes that decide it (`think_tags.ContentStream.ready`).
    var reply_over = false;

    // Generate tokens via the adapter — yields one decoded token id per call
    // regardless of whether the underlying decode is regular, MTP, or drafter.
    while (true) {
        const arms_wait = !gated_stream and delivery == null and (content.waiting() or (in_think_block and think_buf.items.len > 0));
        // A stop cut resolved on the previous token ends the turn here, once the arms have read what they held.
        if (reply_over or (stopped and !arms_wait)) break;
        if (stopped) reply_over = true;
        const token_id: u32 = if (reply_over) 0 else switch (try ts.nextOrIdle(allocator, Conn.STREAM_KEEPALIVE_MS)) {
            .token => |t| t,
            .done => if (stop_gate.hasHeld()) flush: {
                tail_flush = true;
                break :flush 0;
            } else if (arms_wait) flush: {
                reply_over = true;
                break :flush 0;
            } else break,
            .idle => {
                // No tokens yet (long prefill). Probe the peer: an abandoned
                // request must cancel instead of grinding a ghost prefill
                // (Claude Code retries pile up serially otherwise), and the
                // keepalive stops clients timing out on stream silence.
                if (stream.peerClosed()) {
                    log.info("  [cancel] client disconnected while waiting for tokens — cancelling slot\n", .{});
                    ts.slot.?.cancel();
                    client_gone = true;
                    break;
                }
                sendStreamIdle(stream, slot_admission.queueWait(server_state.global_scheduler.?, ts.slot.?)) catch {
                    log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
                    ts.slot.?.cancel();
                    client_gone = true;
                    break;
                };
                continue;
            },
        };
        if (stream.peerClosed()) {
            ts.slot.?.cancel();
            client_gone = true;
            break;
        }
        // The gate delivers the bytes before a match and holds back any tail that could still become
        // one, or a character the token left unfinished.
        const token_text = if (reply_over) try allocator.alloc(u8, 0) else if (tail_flush) try stop_gate.finish(allocator) else blk: {
            const raw_decoded = try tok.decode(allocator, &[_]u32{token_id});
            defer allocator.free(raw_decoded);
            if (delivery) |*d| d.noteToken(raw_decoded.len, ts.slot.?.constraintPayloadStart());
            break :blk try stop_gate.push(allocator, raw_decoded);
        };
        stopped = stop_gate.matched != null;
        if (stopped) ts.slot.?.stop_hit.store(true, .release);
        if (token_text.len == 0 and !reply_over) {
            allocator.free(token_text);
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
            try emitConstrainedChat(allocator, stream, chat_id, model_name, d);
            try beatStreamKeepalive(stream, .sse_comment);
            continue;
        }
        defer if (!gated_stream) allocator.free(token_text);
        // The thought holds the token already when it opened here: it took the tokens that followed its opener.
        var think_has_token = false;

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
            const maybe_tool = think_tags.streamShouldBufferForToolsScan(buf, &tool_scan);

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
                            if (think_tags.heldThoughtSoFarScan(undelivered, opens_think, think_closed, &tool_scan, &thought_scan)) |settled| {
                                if (think_tags.unstreamedReasoning(settled, reasoning_streamed)) |fresh| {
                                    try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, fresh);
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
                        for (token_texts.items) |tt| allocator.free(tt);
                        token_texts.clearRetainingCapacity();
                        text_buf.clearRetainingCapacity();
                        text_delivered = 0;
                        think_scan.reset();
                        tool_scan = .{};
                        thought_scan.reset();
                        think_closed = true;
                        // Past the cap the tail is exactly what the budget
                        // exists to withhold. This arm applied no budget at all,
                        // so a capped request received the WHOLE thought the
                        // moment the block closed.
                        if (!budget_exhausted) {
                            if (split.reasoning_content) |rc| {
                                if (think_tags.unstreamedReasoning(rc, reasoning_streamed)) |fresh| {
                                    try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, fresh);
                                }
                            }
                        }
                        join.close();
                        // Block done and `text_buf` was just cleared — a thought
                        // re-opened later in the turn starts counting from zero.
                        reasoning_streamed = 0;
                        content.endPart();
                        if (split.content.len > 0) switch (try content.deliver(allocator, split.content)) {
                            .send => |bytes| try sendContentDelta(allocator, stream, chat_id, model_name, bytes),
                            else => {},
                        };
                    },
                    .flush_text => {
                        for (token_texts.items) |tt| {
                            defer allocator.free(tt);
                            try content.arrive(allocator, tt);
                        }
                        // The gate flushes no think tag that waits on a backtick's partner: what it flushed is read as it stands.
                        while (try content.ready(allocator, false, true)) |r| switch (r) {
                            .opens => unreachable,
                            .step => |st| {
                                switch (st.step) {
                                    .marker, .hold, .drop => {},
                                    .send => |bytes| try sendContentDelta(allocator, stream, chat_id, model_name, bytes),
                                }
                            },
                        };
                        token_texts.clearRetainingCapacity();
                        text_delivered = text_buf.items.len;
                        thought_scan.reset();
                    },
                }
            }
            // Otherwise keep buffering — tool call may be in progress
        }

        // A think block the MODEL opens, first or after content: seeded like a template-injected one, and holds the tokens
        // that followed its opener while the content arm waited on the tag.
        if (!gated_stream and !in_think_block) {
            if (token_text.len > 0) try content.arrive(allocator, token_text);
            while (try content.ready(allocator, channel_armed, reply_over)) |r| switch (r) {
                .opens => |opener| {
                    // The opener and any whitespace held before it never reach content.
                    content.endPart();
                    channel_armed = false;
                    in_think_block = true;
                    saw_think_open = true;
                    skipped_think_open = false;
                    try think_buf.appendSlice(allocator, opener);
                    const rest = content.takeWaiting();
                    try think_buf.appendSlice(allocator, rest.bytes);
                    think_tokens += @intCast(rest.tokens);
                    think_has_token = true;
                    break;
                },
                .step => |st| {
                    channel_armed = st.step == .marker and std.mem.eql(u8, st.text, "<|channel>");
                    switch (st.step) {
                        .marker, .hold, .drop => {},
                        .send => |bytes| try sendContentDelta(allocator, stream, chat_id, model_name, bytes),
                    }
                },
            };
        }
        if (!gated_stream and in_think_block) {
            // Inside <think> block — stream as reasoning_content with </think>
            // detection. Also covers thinking-off + template-opened (LFM2.5
            // renders `…assistant\n<think>` unconditionally, live 2026-08-04):
            // the block used to be buffered and DROPPED there; now it streams
            // as reasoning_content like any other thought — generated tokens
            // are always delivered.
            if (!think_has_token) {
                try think_buf.appendSlice(allocator, token_text);
                think_tokens += @intFromBool(token_text.len > 0);
            }

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
            // A think close that waits on a backtick's partner is no close yet; the reply's end settles it.
            if (thought_mark.thoughtClose(think_buf.items, reply_over)) |m| {
                const closed = think_tags.closedThoughtDelta(think_buf.items[0..m.pos], thought_shipped);
                const last = closed[0..thought_mark.final(closed)];
                thought_shipped = false;
                if (!budget_exhausted) try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, last);
                join.close();
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
                    switch (try content.deliver(allocator, content_after)) {
                        .send => |bytes| try sendContentDelta(allocator, stream, chat_id, model_name, bytes),
                        else => {},
                    }
                }
                think_buf.clearRetainingCapacity();
                in_think_block = false;
                think_close_tag = if (m.is_channel) "<channel|>" else "</think>";
            } else if (skipped_think_open) {
                // Flush what the closed thought will certainly deliver: a partial
                // close tag, a trailing "\n " run and a split code point wait.
                const flush = think_tags.openThoughtFlush(think_buf.items, thought_shipped);
                const ship = thought_mark.sendable(think_buf.items[flush.skip..][0..flush.ship]);
                const safe_len = flush.skip + ship;
                if (safe_len > 0) {
                    if (!budget_exhausted) try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, think_buf.items[flush.skip..safe_len]);
                    if (ship > 0) thought_shipped = true;
                    const remaining = try allocator.dupe(u8, think_buf.items[safe_len..]);
                    think_buf.clearRetainingCapacity();
                    try think_buf.appendSlice(allocator, remaining);
                    allocator.free(remaining);
                }
            }
        }

        // Every branch above may have written NOTHING (tool-call detection and
        // unclosed thinking blocks buffer silently, sometimes for minutes on a
        // one-shot whole-file `write_file`). Tokens flowing is not bytes
        // flowing — without this the client's idle-body timeout kills the
        // stream mid-generation.
        const beaten = if (gated_stream and stream.keepaliveDue())
            beatToolCall(allocator, stream, chat_id, model_name, &tool_stream, text_buf.items, .{
                .dialect = try chat_mod.callDialect(allocator, lm.chat_config.?),
                .opens_think = opens_think and !think_closed,
            }, tools_json)
        else
            beatStreamKeepalive(stream, .sse_comment);
        beaten catch {
            log.info("  [cancel] keepalive write failed (client disconnected) — cancelling slot\n", .{});
            ts.slot.?.cancel();
            client_gone = true;
            break;
        };
    }

    if (!client_gone) {
        if (delivery) |*d| {
            d.clearOutput();
            try d.finish(allocator);
            if (include_usage) try delivered_reasoning.appendSlice(allocator, d.reasoning.items);
            try emitConstrainedChat(allocator, stream, chat_id, model_name, d);
        }
    }

    // Whitespace the content arm withheld ships unless the non-stream reply trims it: after a closed
    // thought, or before tool markup (a tools request decides below).
    if (!client_gone and !gated_stream) {
        const held = try content.finish(allocator, join.closed);
        if (held.len > 0) try sendContentDelta(allocator, stream, chat_id, model_name, held);
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
            const rest = join.cutTail(think_tags.cutThoughtDelta(think_buf.items, skipped_think_open, thought_shipped));
            const tail = rest[0..thought_mark.final(rest)];
            if (!budget_exhausted) try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, tail);
        } else {
            const lead = think_tags.streamContentLead(think_buf.items, content.started);
            const vis_tail = lead[0..content.mark.final(lead)];
            if (vis_tail.len > 0) {
                content.started = true;
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = vis_tail }, null, null, null);
            }
        }
    }

    // After generation: capture stats from whichever source was active
    // (Generator on legacy, Slot on scheduler).
    if (stopped) ts.finalizeStopped(server_state.global_scheduler.?) else ts.finalize(server_state.global_scheduler.?);

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
        // The text is what the gate delivered; its match is the stop that ended the stream (`reply.settle`).
        const settled = try reply_mod.settle(allocator, .{
            .text = text_buf.items,
            .source = .{ .gated = stop_gate.matched },
            .finish_reason = finish_reason,
            .finish_details = ts.finish_details,
        }, .{
            .stops = stop_sequences,
            .normalize = !constrained_proto,
            .has_tools = has_tools,
            .tools_json = tools_json,
            .allow_parallel = allow_parallel_tools,
            .dialect = if (has_tools) try chat_mod.callDialect(allocator, lm.chat_config.?) else null,
            .opens_think = opens_think and !think_closed,
        });
        defer settled.deinit(allocator);
        declined_field = try reply_mod.detailsField(allocator, "finish_details", settled.declined, "");
        const gen_text = settled.text;
        if (settled.reply) |tool_reply| {
            const tool_calls = tool_reply.calls;

            // Emit reasoning_content before tool calls — whatever the model
            // generated is delivered, thinking flag or not.
            {
                const think_split = think_tags.splitThinkBlock(gen_text, true, opens_think and !think_closed);
                // `budget_exhausted` means the cap was already streamed in full.
                if (think_tags.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                    defer if (capped.owned) allocator.free(capped.text);
                    try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, capped.text);
                }
            }

            // Emit tool call deltas in OpenAI streaming format
            tool_stream.settle(if (tool_calls.len > 0) tool_calls[0] else null);
            for (tool_calls, 0..) |tc, i| {
                // A call that went out early owes its arguments (or nothing); any other is delivered whole.
                const first_delta = try tool_progress.chatEnd(allocator, stream.io, &tool_stream, i, tc);
                defer allocator.free(first_delta);
                try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = first_delta }, null, null, null);
            }
            try endCutCall(allocator, stream, chat_id, model_name, &tool_stream);
            finish_reason = toolCallFinishReason(finish_reason);
        } else {
            tool_stream.settle(null);
            try endCutCall(allocator, stream, chat_id, model_name, &tool_stream);
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
                const held = try content.finish(allocator, join.closed and think_split.content.len == 0);
                if (held.len > 0) try sendContentDelta(allocator, stream, chat_id, model_name, held);
                // `budget_exhausted` means the cap was already streamed in full.
                if (think_tags.unstreamedReasoning(if (budget_exhausted) "" else think_split.reasoning_content orelse "", reasoning_streamed)) |reasoning| {
                    // Apply reasoning budget truncation if set
                    const capped = try reasoningWithinBudget(allocator, tok, reasoning, reasoning_budget);
                    defer if (capped.owned) allocator.free(capped.text);
                    try sendReasoning(allocator, stream, chat_id, model_name, &join, if (include_usage) &delivered_reasoning else null, capped.text);
                }
                if (think_split.content.len > 0) try sendContentDelta(allocator, stream, chat_id, model_name, think_split.content);
            }
        }
    }

    const total_prompt = ts.prompt_tokens;
    if (!client_gone) {
        // Final chunk with finish_reason
        try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null }, .{ .reason = finish_reason, .details = ts.finish_details, .declined_field = if (declined_field.len > 0) declined_field else null }, null, null);

        // Usage chunk (if requested via stream_options.include_usage). Scheduler
        // accounts for any prompt-cache hits in `ts.prompt_tokens` directly.
        // OpenAI ships this chunk with an EMPTY choices array — restating
        // finish_reason/finish_details here made every per-event client render
        // the ending twice (PR #147's doubled truncation banner). The
        // final chunk above listed what content had left unlisted.
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

/// How a stream ended: the wire `finish_reason` plus, when the
/// degenerate-tail guard cut it, the sibling cause (`finishDetailsField`).
const Finish = struct {
    reason: []const u8,
    details: ?[]const u8 = null,
    /// The `finish_details` field naming a call the reply declined (`reply.detailsField`); it outranks `details`.
    declined_field: ?[]const u8 = null,
};

/// A reasoning delta, led by the blank line that joins a later thought to an earlier one (`ThoughtJoin`).
fn sendReasoning(allocator: std.mem.Allocator, stream: *Conn, chat_id: i64, model_name: []const u8, join: *think_tags.ThoughtJoin, usage: ?*std.ArrayList(u8), delta: []const u8) !void {
    if (delta.len == 0) return;
    const sep = join.lead(delta);
    if (usage) |u| {
        try u.appendSlice(allocator, sep);
        try u.appendSlice(allocator, delta);
    }
    if (sep.len > 0) try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = sep }, null, null, null);
    try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .reasoning_content = delta }, null, null, null);
}

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
    const fd_str = if (finish) |f| (f.declined_field orelse finishDetailsField(f.reason, f.details)) else "";

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
        \\{{"id":"chatcmpl-{d}","object":"chat.completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"delta":{s},"finish_reason":{s}{s}}}],"usage":{s}{s}}}
    , .{ chat_id, nowSecs(stream.io), model_name, delta_buf.items, fr_str, fd_str, usage_str, timings_tail_buf.items });
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
/// (all per-choice fields; the final chunk already carried them).
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
        "ContextSizeAboveWindow",
        "LaunchFlagUnsupported",
        "GlmKvQuantUnsupported",
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
        "Exl3TileOrderUnsupported",
        "Exl3TileOrderMixed",
        "Exl3TileOrderMismatch",
        "Exl3TileOrderNeedsGroups",
        "Exl3CodebookTableUnknown",
        "Exl3CodebookTableDigestMismatch",
        "Exl3CodebookTableWindowMismatch",
        "Exl3CodebookTableCodebookMismatch",
        "Exl3CodebookTableRateMissing",
        "Exl3CodebookTableLegacyConfig",
        "Exl3CodebookTableUndecodable",
        "PackRevisionMixed",
    };
    for (names) |name| {
        const refusal = loadRefusalFor(model_registry_mod.ModelRegistry.loadErrorFromName(name)) orelse {
            std.debug.print("no named refusal for {s}\n", .{name});
            return error.TestUnexpectedResult;
        };
        try t.expect(refusal.type.len > 0);
        try t.expect(refusal.message.len > 0);
    }
    try t.expectEqualStrings("launch_flag_unsupported", loadRefusalFor(error.LaunchFlagUnsupported).?.type);
    try t.expectEqualStrings("kv_quant_unsupported", loadRefusalFor(error.GlmKvQuantUnsupported).?.type);
    try t.expectEqualStrings("context_size_above_window", loadRefusalFor(error.ContextSizeAboveWindow).?.type);
    try t.expectEqualStrings("architecture_unsupported", loadRefusalFor(error.ArchitectureUnsupported).?.type);
    try t.expectEqualStrings("model_format_unsupported", loadRefusalFor(error.ModelFormatUnsupported).?.type);
    try t.expectEqualStrings("expert_streaming_unsupported_layout", loadRefusalFor(error.ExpertStreamingUnsupportedLayout).?.type);
    try t.expectEqualStrings("expert_slab_import_copied", loadRefusalFor(error.ExpertSlabImportCopied).?.type);
    try t.expectEqualStrings("expert_layout_unsupported", loadRefusalFor(error.ExpertLayoutUnsupported).?.type);
    try t.expectEqualStrings("exl3_topk_exceeds_reduce_bank", loadRefusalFor(error.Exl3TopKExceedsReduceBank).?.type);
    try t.expectEqualStrings("exl3_trellis_geometry", loadRefusalFor(error.Exl3TrellisGeometry).?.type);
    try t.expectEqualStrings("exl3_window_unsupported", loadRefusalFor(error.Exl3WindowUnsupported).?.type);
    try t.expectEqualStrings("pack_revision_mixed", loadRefusalFor(error.PackRevisionMixed).?.type);
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

pub const LoadRefusal = struct { type: []const u8, message: []const u8, retry_after: ?u32 = null };

test "streaming layout refusal explains MiMo repacking without assuming Qwen" {

    try test_note.needArchs(&.{"mimo2"});
    const layout = loadRefusalFor(error.ExpertStreamingUnsupportedLayout).?;
    try std.testing.expect(std.mem.indexOf(u8, layout.message, "For MiMo, repack") != null);
    const budget = loadRefusalFor(error.ExpertStreamingRequired).?;
    try std.testing.expect(std.mem.indexOf(u8, budget.message, "qwen4_exp") == null);
    try std.testing.expect(std.mem.indexOf(u8, budget.message, "--ssd-budget-gb") != null);
}

test "a fixed refusal text, which knows no pack, offers no smaller pack" {
    var buf: [384]u8 = undefined;
    const gib: u64 = 1 << 30;
    const texts = [_][]const u8{
        PREFILL_NOFIT_MSG,
        loadRefusalFor(error.ExpertCacheDoesNotFit).?.message,
        loadRefusalFor(error.SsdBudgetExceedsWiredLimit).?.message,
        // Past what this Mac may wire: the process's wired bytes span every loaded model.
        bills_mod.userWireWarning(&buf, 117 * gib, .{ 116 * gib, 116 * gib }, 0, 128 * gib).?,
    };
    for (texts) |text| {
        errdefer std.debug.print("{s}\n", .{text});
        try std.testing.expect(std.mem.indexOf(u8, text, "smaller") == null);
    }
}

test "the load refusals list the archs this build serves" {
    if (arch_packs.gate_order.len != 4) return error.SkipZigTest;
    try std.testing.expectEqualStrings(
        "This checkpoint's model_type is not served by this build, which loads qwen4_exp (Qwen3.8-Flash-Next), mimo_v2 (MiMo-V2.6-Flash), glm5_next (GLM-5.3-Flash), and deepseek_v41 (DeepSeek-V4.1-Flash).",
        loadRefusalFor(error.ArchitectureUnsupported).?.message,
    );
    try std.testing.expectEqualStrings(
        "This checkpoint's file format is not supported. Serve an MLX safetensors checkpoint (qwen4_exp, mimo_v2, glm5_next or deepseek_v41).",
        loadRefusalFor(error.ModelFormatUnsupported).?.message,
    );
}

pub fn loadRefusalFor(err: anyerror) ?LoadRefusal {
    return switch (err) {
        error.NotEnoughMemory => .{ .type = "out_of_memory", .message = not_enough_memory_message },
        error.ResidentModelsCapReached => .{ .type = "resident_models_cap", .message = resident_models_cap_message, .retry_after = BUSY_RETRY_AFTER_SECS },
        error.InsufficientMemory => .{ .type = "out_of_memory", .message = insufficient_free_memory_message },
        error.ContextSizeAboveWindow => .{ .type = "context_size_above_window", .message = "--ctx-size, or this model's saved ctx_size, is above the context window the model has under its yarn_factor and --config-overrides. Lower it, or raise yarn_factor." },
        error.LaunchFlagUnsupported => .{ .type = "launch_flag_unsupported", .message = "A launch flag this server was started with has no effect on this model. The server log names the flag; restart without it." },
        error.GlmKvQuantUnsupported => .{ .type = "kv_quant_unsupported", .message = "The KV-cache quantization in force (--kv-quant or the model's saved kv_quant) is not one this model stores. The server log names the accepted widths." },
        error.ArchitectureUnsupported => .{ .type = "architecture_unsupported", .message = "This checkpoint's model_type is not served by this build, which loads " ++ arch_packs.served_list ++ "." },
        error.ModelFormatUnsupported => .{ .type = "model_format_unsupported", .message = "This checkpoint's file format is not supported. Serve an MLX safetensors checkpoint (" ++ arch_packs.served_types_or ++ ")." },
        error.ExpertCacheDoesNotFit => .{ .type = "expert_cache_does_not_fit", .message = "Not enough memory to load this model: its SSD streaming memory and what serving needs beside it do not fit under the GPU memory limit. Lower Memory for SSD streaming (GiB) in its settings (--ssd-budget-gb), or quit other apps or unload another model." },
        error.ExpertStreamingMtpUnsupported => .{ .type = "expert_streaming_mtp_unsupported", .message = expert_stream_mod.MTP_UNSUPPORTED },
        error.ExpertStreamingRequired => .{ .type = "expert_streaming_required", .message = model_load_mod.budget_required_message },
        error.ExpertStreamingUnsupportedLayout => .{ .type = "expert_streaming_unsupported_layout", .message = arch_load.refusalText("ExpertStreamingUnsupportedLayout") orelse "This checkpoint has no complete expert-bank layout this build can stream. Check the pack and all indexed shards." },
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
        error.Exl3RateGroupsUnsupported => .{ .type = "exl3_rate_groups_unsupported", .message = arch_load.refusalText("Exl3RateGroupsUnsupported") orelse "This model serves one rate per layer; a rate-grouped layer needs a model that reads rate groups." },
        error.Exl3RouterGroupsUnsupported => .{ .type = "exl3_router_groups_unsupported", .message = "Reordered or pruned EXL3 experts require n_group=1 routing." },
        error.Exl3TopKExceedsReduceBank => .{ .type = "exl3_topk_exceeds_reduce_bank", .message = "This EXL3 pack's num_experts_per_tok exceeds the decode reduce-bank (32). Re-convert with top-k <= 32." },
        error.Exl3RateGroupsStreamingUnsupported => .{ .type = "exl3_rate_groups_streaming_unsupported", .message = "SSD streaming does not support EXL3 rate-group (.gN) packs. Use uniform leading-index expert banks." },
        error.Exl3GateUpRateMismatch => .{ .type = "exl3_gate_up_rate_mismatch", .message = "The EXL3 gate and up trellises must use the same packed rate." },
        error.Exl3TrellisGeometry => .{ .type = "exl3_trellis_geometry", .message = "This EXL3 pack has a routed-expert trellis this build cannot decode, or one that disagrees with the expert count, shape or k its config.json names. Re-convert the pack." },
        error.Exl3WindowUnsupported => .{ .type = "exl3_window_unsupported", .message = "This EXL3 pack names a codeword window this build cannot decode: expert_quant.window must be an integer from 8 to 16, or absent for 16." },
        error.Exl3TileOrderUnsupported => .{ .type = "exl3_tile_order_unsupported", .message = "This pack's expert_quant.tile_order is not one this model reads: pair_major is read by one architecture only (docs/pack-format.md#tile-order)." },
        error.Exl3TileOrderMixed => .{ .type = "exl3_tile_order_mixed", .message = "This pack stores some routed trellis tensors as trellis and some as trellis_pm. Re-convert the pack." },
        error.Exl3TileOrderMismatch => .{ .type = "exl3_tile_order_mismatch", .message = "This pack's routed trellis tensor names do not match the tile_order its config.json names. Re-convert the pack, or fix expert_quant to match." },
        error.Exl3TileOrderNeedsGroups => .{ .type = "exl3_tile_order_needs_groups", .message = "A pair_major pack stores every routed layer as rate groups (gN); this layer is not." },
        error.PackRevisionMixed => .{ .type = "pack_revision_mixed", .message = "This pack folder mixes shards of two pack revisions, or has an unstamped shard in a stamped pack; the server log names the files. Run `sushi pull <name>` again or re-download the pack." },
        error.Exl3ShardStampMismatch => .{ .type = "exl3_shard_stamp_mismatch", .message = "An EXL3 shard in this pack was written for a different decoder than config.json's expert_quant names (k, codebook or window). Re-convert the pack, or fix expert_quant to match the shards." },
        error.Exl3CodebookTableUnknown => .{ .type = "exl3_codebook_table_unknown", .message = "This STQ pack names no codebook table this build holds: expert_quant.codebook_table must be one of std1, spec1-w15, spec1-w16 or pair1-w16-v1. A pack that names no table is reconverted." },
        error.Exl3CodebookTableDigestMismatch => .{ .type = "exl3_codebook_table_digest_mismatch", .message = "This STQ pack's expert_quant.codebook_digest is not its codebook table's digest in this build. Re-convert the pack." },
        error.Exl3CodebookTableWindowMismatch => .{ .type = "exl3_codebook_table_window_mismatch", .message = "The PAIR1 codebook table this STQ pack names decodes only at its own window (pair1-w16-v1: 16), not at expert_quant.window." },
        error.Exl3CodebookTableCodebookMismatch => .{ .type = "exl3_codebook_table_codebook_mismatch", .message = "The codebook table this STQ pack names is for another codebook than expert_quant.codebook." },
        error.Exl3CodebookTableRateMissing => .{ .type = "exl3_codebook_table_rate_missing", .message = "This STQ pack has a routed bank at a rate its codebook table has no row for (each table covers the even n it lists: spec1-w15 n16 to n128, the others n24 to n64)." },
        error.Exl3CodebookTableLegacyConfig => .{ .type = "exl3_codebook_table_legacy_config", .message = "This pack's expert_quant names a codebook table by a digest from before the table grew (spec1-w15 2ac2594d, now 37f72431). Re-convert the pack." },
        error.Exl3CodebookTableUndecodable => .{ .type = "exl3_codebook_table_undecodable", .message = "This STQ pack's codebook table holds PAIR1 constants this build's kernels have not compiled." },
        error.SsdBudgetBelowResident => .{ .type = "ssd_budget_below_resident", .message = "This model's SSD streaming memory is too small to load it: raise Memory for SSD streaming (GiB) in its settings (--ssd-budget-gb)." },
        error.SsdBudgetBelowVision => .{ .type = "ssd_budget_below_vision", .message = arch_load.refusalText("SsdBudgetBelowVision") orelse "--ssd-budget-gb cannot hold the vision tower (--vision) beside the minimum expert cache. Raise the budget or pass --no-vision." },
        error.SsdBudgetExceedsWiredLimit => .{ .type = "ssd_budget_exceeds_wired_limit", .message = "Not enough memory to load this model: its SSD streaming memory and the context it plans are more than this Mac's GPU memory limit. Lower Memory for SSD streaming (GiB) in its settings (--ssd-budget-gb)." },
        else => null,
    };
}

pub fn sendNamedLoadRefusal(allocator: std.mem.Allocator, stream: *Conn, scheduler: *scheduler_mod.Scheduler, id: []const u8, err: anyerror, report: *const scheduler_mod.LoadReport) !bool {
    if (err == error.PackDownloading) {
        var gate_buf: [512]u8 = undefined;
        const entry = scheduler.registry.resolveEntry(id) catch null;
        const why = (if (entry) |e| scheduler.registry.loadBlocked(&gate_buf, e) else null) orelse
            "A download into this model's folder was running: load it again";
        try sendErrorResponse(allocator, stream, "409 Conflict", "pack_downloading", why, 409);
        return true;
    }
    const refusal = loadRefusalFor(err) orelse return false;
    // Only this request's own load names a context: a fast answer from a stored failure is never a refusal.
    const context = if (err == error.InsufficientMemory) report.context else null;
    const message = try loadRefusalMessage(allocator, err, refusal, context, report.refusalText());
    defer allocator.free(message);
    var retry_buf: [40]u8 = undefined;
    const retry = if (refusal.retry_after) |secs| std.fmt.bufPrint(&retry_buf, "Retry-After: {d}\r\n", .{secs}) catch "" else "";
    try sendErrorResponseWithHeaders(allocator, stream, "503 Service Unavailable", refusal.type, message, 503, retry);
    return true;
}

/// The message a load refusal's answer carries: the text the load wrote when it wrote one (the memory check's names the
/// bound and what to free or change), else the refusal's own. The caller frees it.
pub fn loadRefusalMessage(allocator: std.mem.Allocator, err: anyerror, refusal: LoadRefusal, context: ?model_registry_mod.MemoryContextRefusal, text: ?[]const u8) ![]u8 {
    const written = err == error.InsufficientMemory or err == error.ContextSizeAboveWindow or err == error.LaunchFlagUnsupported or err == error.GlmKvQuantUnsupported or
        err == error.SsdBudgetExceedsWiredLimit or err == error.ExpertCacheDoesNotFit or err == error.SsdBudgetBelowResident or err == error.NotEnoughMemory;
    if (written) if (text) |t| return allocator.dupe(u8, t);
    if (context) |c| return memoryContextRefusalMessage(allocator, c);
    return allocator.dupe(u8, refusal.message);
}

test "a request refused for memory with no text of its own gets the actions that can help it, one per line" {
    const a = std.testing.allocator;
    for ([_]anyerror{ error.TransientCapExceeded, error.GenerationOutOfMemory, error.PrefillDoesNotFit, error.PrefillChunkDoesNotFit }) |e| {
        const w = try mapGenerationError(a, e);
        defer w.deinit(a);
        errdefer std.debug.print("{s}: {s}\n", .{ @errorName(e), w.message });
        try std.testing.expect(std.mem.indexOf(u8, w.message, "\n- Quit other apps or unload another model") != null);
        try std.testing.expect(std.mem.indexOf(u8, w.message, "max_tokens") != null or std.mem.indexOf(u8, w.message, "prompt") != null);
        try std.testing.expect(std.mem.indexOf(u8, w.message, "cap allows") == null and std.mem.indexOf(u8, w.message, "image size") == null);
    }
    const shut = try mapGenerationError(a, error.ServerShutdown);
    defer shut.deinit(a);
    try std.testing.expect(std.mem.indexOf(u8, shut.message, "retry") != null);
}

test "a prefill chunk refused mid-request for memory answers 503 with the arch's whole text, not the admission refusal" {
    const text = "This prompt ran out of GPU memory partway through its prefill.\nIts 2048-token chunk, tokens 4096 to 6144, needs 5.00 GB, and the free memory with every expert-cache pool the prefill can borrow holds 4.50 GB: about 0.5 GB short. The chunks before it ran.\nTo serve it:\n- Lower --prefill-chunk to 1024: a narrower chunk needs less memory at this length.\n- Retry when other requests finish or other apps quit: the memory they hold comes back.\n- Shorten the prompt to 4096 tokens or fewer: the chunks up to there fit.";
    forward_refusal_mod.record(text);
    server_state.slot_forward_refusal = forward_refusal_mod.take();
    const a = std.testing.allocator;
    const w = try mapGenerationError(a, error.PrefillChunkDoesNotFit);
    defer w.deinit(a);
    try std.testing.expectEqual(@as(u32, 503), w.code);
    try std.testing.expectEqualStrings(text, w.message);
    // With no text left, the answer still names a mid-prefill refusal, never the before-any-work one.
    const bare = try mapGenerationError(a, error.PrefillChunkDoesNotFit);
    defer bare.deinit(a);
    try std.testing.expectEqualStrings(PREFILL_CHUNK_NOFIT_MSG, bare.message);
    try std.testing.expect(std.mem.indexOf(u8, bare.message, "before any work started") == null);
}

test "a load refused by a check that wrote its text answers with that text, as the memory preflight does" {
    const a = std.testing.allocator;
    for ([_]anyerror{ error.SsdBudgetExceedsWiredLimit, error.ExpertCacheDoesNotFit, error.SsdBudgetBelowResident, error.NotEnoughMemory }) |e| {
        const message = try loadRefusalMessage(a, e, loadRefusalFor(e).?, null, "Not enough memory to load this model.\nIt needs 150.0 GB ...");
        defer a.free(message);
        try std.testing.expectEqualStrings("Not enough memory to load this model.\nIt needs 150.0 GB ...", message);
    }
}

test "a memory-refused load's 503 carries the check's whole text, its lines intact" {
    const a = std.testing.allocator;
    var text_buf: [2048]u8 = undefined;
    const gib: u64 = 1 << 30;
    const text = bills_mod.loadMemoryRefusalText(&text_buf, .{ .needed = 20 * gib, .avail = bills_mod.sushiAvailable(12 * gib, 0, 29 * gib, 0, 28 * gib), .wired_beside = 0 });
    const context = model_registry_mod.MemoryContextRefusal{ .requested = 9, .maximum = 4, .needed = 1, .available = 0, .kv_bits = 8, .chunk = 512 };
    const message = try loadRefusalMessage(a, error.InsufficientMemory, loadRefusalFor(error.InsufficientMemory).?, context, text);
    defer a.free(message);
    try std.testing.expectEqualStrings(text, message);
    // The body the web UI and Sushibar read `error.message` from.
    const escaped = try http.jsonEscape(a, message);
    defer a.free(escaped);
    const body = try std.fmt.allocPrint(a, "{{\"error\":{{\"message\":{s}}}}}", .{escaped});
    defer a.free(body);
    const E = struct { @"error": struct { message: []const u8 } };
    const parsed = try std.json.parseFromSlice(E, a, body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(text, parsed.value.@"error".message);
    try std.testing.expect(std.mem.indexOf(u8, parsed.value.@"error".message, "Free about 8.0 GB more.\nTo load it:\n- Quit other apps") != null);
    // An answer with no text written falls back to the context figures.
    const again = try loadRefusalMessage(a, error.InsufficientMemory, loadRefusalFor(error.InsufficientMemory).?, context, null);
    defer a.free(again);
    try std.testing.expect(std.mem.startsWith(u8, again, "Context 9 does not fit"));
}

/// #144: `error.LoadFailed` used to surface as a bare "Model load failed" for
/// every on-demand load failure — the inference thread's error name was
/// dropped at the thread boundary. The registry keeps it; echo it.
pub fn sendLoadFailedResponse(allocator: std.mem.Allocator, stream: *Conn, report: *const scheduler_mod.LoadReport) !void {
    if (report.failed) |err| return sendLoadFailedNamed(allocator, stream, @errorName(err));
    // No load of this request's own: the stored failure it was answered from fast, an earlier load's by design.
    if (report.stored.get()) |name| return sendLoadFailedNamed(allocator, stream, name);
    try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
}

fn sendLoadFailedNamed(allocator: std.mem.Allocator, stream: *Conn, name: []const u8) !void {
    const msg = std.fmt.allocPrint(allocator, "Model load failed: {s}", .{name}) catch
        return sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", "Model load failed", 500);
    defer allocator.free(msg);
    try sendErrorResponse(allocator, stream, "500 Internal Server Error", "model_load_failed", msg, 500);
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

/// The keepalive of a stream that is buffering tool markup: while a call is being written, a data chunk that adds no
/// bytes to it (`tool_progress`), so a client that counts events, not comments, sees the stream move.
fn beatToolCall(
    allocator: std.mem.Allocator,
    stream: *Conn,
    chat_id: i64,
    model_name: []const u8,
    progress: *tool_progress.Progress,
    buffered: []const u8,
    origin: reply_mod.Origin,
    tools_json: ?[]const u8,
) !void {
    // A WebSocket has liveness of its own, and its frames are not a stream a client times.
    if (stream.ws_mode == null) {
        if (try progress.beat(allocator, stream.io, buffered, origin, tools_json, .{})) |beat| {
            const delta = try tool_progress.chatBeat(allocator, progress, beat);
            defer allocator.free(delta);
            return sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = delta }, null, null, null);
        }
    }
    try beatStreamKeepalive(stream, .sse_comment);
}

/// A call sent to the client that the reply does not hold (the generation ended inside it) stays as it was sent,
/// `arguments` empty; the finish is never `tool_calls` and names the cut.
fn endCutCall(allocator: std.mem.Allocator, stream: *Conn, chat_id: i64, model_name: []const u8, progress: *const tool_progress.Progress) !void {
    const open = progress.cut() orelse return;
    log.info("  [tool-stream] the call \"{s}\" sent to the client was left unfinished; it ends under a finish that is not tool_calls\n", .{open.name});
    if (progress.cutEnding()) |ending| try sendSSEChunk(allocator, stream, chat_id, model_name, .{ .role = null, .content = null, .tool_calls_json = ending }, null, null, null);
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
    const w = try mapGenerationError(t.allocator, error.TransientCapExceeded);
    defer w.deinit(t.allocator);
    try t.expectEqual(@as(u32, 503), w.code);
    try t.expectEqualStrings(TRANSIENT_CAP_MSG, w.message);
    try t.expect(!std.mem.eql(u8, w.message, GEN_OOM_MSG));

    slot.error_code = "OutOfMemory";
    try t.expect(slotFailure(&slot) == error.GenerationOutOfMemory);
}

test "a refused message role is a 400 that names the role, on both dialects" {
    const t = std.testing;
    var config = chat_mod.ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = t.allocator };
    const msgs = [_]chat_mod.Message{.{ .role = "narrator", .content = "x" }};
    try t.expectError(error.ChatRoleUnknown, chat_mod.renderChatTemplate(t.allocator, &msgs, &config, null, null, true, null, false));
    const w = try mapGenerationError(t.allocator, error.ChatRoleUnknown);
    defer w.deinit(t.allocator);
    try t.expectEqual(@as(u32, 400), w.code);
    try t.expectEqualStrings("invalid_request_error", w.openai_type);
    try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    try t.expect(std.mem.indexOf(u8, w.message, "\"narrator\"") != null);
}

test "a streaming fault answers with the SAME mapped error a non-streaming one does" {
    // The 400/503 mapping existed only on the non-streaming arms; streaming arms wrote a bare `server_error` frame.
    const t = std.testing;
    const a = t.allocator;

    {
        const w = try mapGenerationError(a, error.GenerationOutOfMemory);
        defer w.deinit(a);
        try t.expectEqual(@as(u32, 503), w.code);
        try t.expectEqualStrings("503 Service Unavailable", w.status_line);
        try t.expectEqualStrings(GEN_OOM_MSG, w.message);
        try t.expectEqualStrings("server_error", w.openai_type);
        try t.expectEqualStrings("api_error", w.anthropic_type);
    }
    {
        const w = try mapGenerationError(a, error.PrefillDoesNotFit);
        defer w.deinit(a);
        try t.expectEqual(@as(u32, 400), w.code);
        try t.expectEqualStrings(PREFILL_NOFIT_MSG, w.message);
        try t.expectEqualStrings("invalid_request_error", w.openai_type);
        try t.expectEqualStrings("invalid_request_error", w.anthropic_type);
    }
    for ([_]anyerror{ error.ServerShutdown, error.Shutdown }) |e| {
        const w = try mapGenerationError(a, e);
        defer w.deinit(a);
        try t.expectEqual(@as(u32, 503), w.code);
        // The type every other shutdown refusal sends (an unload, a load, a download, a cache job).
        try t.expectEqualStrings("shutting_down", w.openai_type);
        try t.expectEqualStrings(http.shutting_down_message, w.message);
    }
    {
        const w = try mapGenerationError(a, error.GenerationFailed);
        defer w.deinit(a);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("generation failed", w.message);
    }
    {
        // Anything else keeps its name in the message, at a mapped status.
        const w = try mapGenerationError(a, error.WriteFailed);
        defer w.deinit(a);
        try t.expectEqual(@as(u32, 500), w.code);
        try t.expectEqualStrings("Internal server error: WriteFailed", w.message);
    }

    // The SSE bodies carry the same type and message, plus the surface's terminal marker.
    {
        const w = try mapGenerationError(a, error.GenerationOutOfMemory);
        defer w.deinit(a);
        const quoted = try http.jsonEscape(a, GEN_OOM_MSG);
        defer a.free(quoted);
        const escaped = quoted[1 .. quoted.len - 1];
        const body = try w.envelope().openaiStreamFrame(a, false);
        defer a.free(body);
        try t.expect(std.mem.indexOf(u8, body, "\"type\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, body, "\"code\":503") != null);
        try t.expect(std.mem.indexOf(u8, body, escaped) != null);
        try t.expect(std.mem.indexOf(u8, body, "\"finish_reason\":\"error\"") != null);

        const anth = try w.envelope().anthropicStreamEvent(a);
        defer a.free(anth);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, "\"type\":\"api_error\"") != null);
        try t.expect(std.mem.indexOf(u8, anth, escaped) != null);

        const resp = try w.envelope().responsesStreamEvent(a);
        defer a.free(resp);
        try t.expect(std.mem.indexOf(u8, resp, "\"type\":\"error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, "\"code\":\"server_error\"") != null);
        try t.expect(std.mem.indexOf(u8, resp, escaped) != null);
    }
}

test "every error frame of a generation failure is the same bytes in its dialect" {
    const a = std.testing.allocator;
    const w: GenErrorWire = .{
        .status_line = "503 Service Unavailable",
        .code = 503,
        .openai_type = "server_error",
        .anthropic_type = "api_error",
        .message = "He said \"hi\"\n",
    };
    const quoted = "\"He said \\\"hi\\\"\\n\"";
    const chat = try w.envelope().openaiStreamFrame(a, false);
    defer a.free(chat);
    try std.testing.expectEqualStrings("data: {\"error\":{\"message\":" ++ quoted ++ ",\"type\":\"server_error\",\"param\":null,\"code\":503}," ++
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"error\"}]}\n\n", chat);
    const text = try w.envelope().openaiStreamFrame(a, true);
    defer a.free(text);
    try std.testing.expectEqualStrings("data: {\"error\":{\"message\":" ++ quoted ++ ",\"type\":\"server_error\",\"param\":null,\"code\":503}," ++
        "\"choices\":[{\"index\":0,\"text\":\"\",\"logprobs\":null,\"finish_reason\":\"error\"}]}\n\n", text);
    const anth = try w.envelope().anthropicStreamEvent(a);
    defer a.free(anth);
    try std.testing.expectEqualStrings("{\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":" ++ quoted ++ "}}", anth);
    const resp = try w.envelope().responsesStreamEvent(a);
    defer a.free(resp);
    try std.testing.expectEqualStrings("{\"type\":\"error\",\"code\":\"server_error\",\"message\":" ++ quoted ++ ",\"param\":null}", resp);
    const ws = try w.envelope().wsFrame(a);
    defer a.free(ws);
    try std.testing.expectEqualStrings("{\"type\":\"error\",\"status\":503,\"error\":{\"code\":\"server_error\",\"message\":" ++ quoted ++ "}}", ws);
}

test "an empty prompt refused at submit is a 400 invalid_request_error in both dialects" {
    const t = std.testing;
    const w = try mapGenerationError(t.allocator, error.EmptyPrompt);
    defer w.deinit(t.allocator);
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
    try t.expectEqualStrings("client_disconnect", nonStreamFinishReason(.client_gone, "length"));
    try t.expectEqualStrings("cancelled", nonStreamFinishReason(.cancelled, "length"));
    try t.expectEqualStrings("stop", nonStreamFinishReason(.done, "stop"));
    try t.expectEqualStrings("length", nonStreamFinishReason(.done, "length"));
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
    var tok = Tokenizer.initEmpty(a);
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

test "a streamed stop reports the tokens it delivered, not the speculative block decoded past it" {
    const a = std.testing.allocator;
    const io = std.testing.io;
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
    slot.out_idx = 0;
    slot.handover_token = null;
    slot.cancelled = .init(false);
    slot.stop_hit = .init(false);
    slot.shutdown_cancelled = .init(false);
    slot.finished = false;
    slot.error_code = null;
    slot.completion_tokens = 0;
    slot.prompt_tokens = 0;
    slot.cached_tokens = 0;
    slot.finish_reason = "stop";
    slot.finish_details = null;
    slot.prefill_ns = 0;
    slot.decode_ns = 0;
    slot.in_pass = .init(1);

    var player = TickPlayer{ .slot = &slot, .n = 4, .gap_ms = 2 };
    const t = try std.Thread.spawn(.{}, TickPlayer.run, .{&player});
    var ts = StreamingTokenStream.initFromSlot(&slot, .regular, &.{});
    switch (try ts.nextOrIdle(a, 1000)) {
        .token => {},
        else => return error.TestUnexpectedResult,
    }
    ts.finalizeStopped(&sch);
    t.join();
    try std.testing.expectEqual(@as(u32, 4), slot.completion_tokens);
    try std.testing.expectEqual(@as(u32, 1), ts.completion_tokens);
}

/// The inference pass still holding a slot the cancel API ended: it writes the prompt count, two tokens decoded past
/// what the reader took and the decode time, then lets go.
const LatePass = struct {
    slot: *scheduler_mod.Slot,
    fn run(self: *LatePass) void {
        std.Io.sleep(self.slot.io, .fromMilliseconds(5), .real) catch {};
        self.slot.prompt_tokens = 12;
        self.slot.completion_tokens += 2;
        self.slot.decode_ns = 777;
        _ = self.slot.in_pass.fetchSub(1, .acq_rel);
    }
};

test "a reply the cancel API ends mid-decode says cancelled and counts the tokens it returned, read once its pass settles" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    for ([_]bool{ false, true }) |streamed| {
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
        slot.out_idx = 0;
        slot.handover_token = null;
        slot.cancelled = .init(false);
        slot.stop_hit = .init(false);
        slot.shutdown_cancelled = .init(false);
        slot.finished = false;
        slot.error_code = null;
        // What `Slot.init` leaves until a finish publishes its own.
        slot.finish_reason = "length";
        slot.finish_details = null;
        slot.prompt_tokens = 0;
        slot.cached_tokens = 0;
        slot.completion_tokens = 3;
        slot.prefill_ns = 0;
        slot.decode_ns = 0;
        slot.in_pass = .init(1);
        for (0..3) |i| slot.pushToken(@intCast(i + 1));
        // POST /v1/slots/cancel, from another connection, while a pass holds the slot.
        slot.cancel();
        var pass = LatePass{ .slot = &slot };
        const t = try std.Thread.spawn(.{}, LatePass.run, .{&pass});
        defer t.join();
        if (streamed) {
            var ts = StreamingTokenStream.initFromSlot(&slot, .regular, &.{});
            var got: u32 = 0;
            while (try ts.nextOrIdle(a, 1000) == .token) got += 1;
            ts.finalize(&sch);
            try std.testing.expectEqual(@as(u32, 3), got);
            try std.testing.expectEqualStrings("cancelled", ts.finish_reason);
            try std.testing.expectEqual(@as(u32, 3), ts.completion_tokens);
            try std.testing.expectEqual(@as(u32, 12), ts.prompt_tokens);
            try std.testing.expectEqual(@as(u64, 777), ts.decode_ns);
        } else {
            var out = std.ArrayList(u32).empty;
            defer out.deinit(a);
            const end = try drainSettled(&sch, &slot, null, a, &out, null);
            // Read exactly as the result builder does.
            const prompt = slot.prompt_tokens;
            try std.testing.expectEqualStrings("cancelled", @tagName(end));
            try std.testing.expectEqual(@as(usize, 3), out.items.len);
            try std.testing.expectEqual(@as(u32, 12), prompt);
        }
    }
}

test "a non-stream early stop reads usage and timing only after the tick that published its token settles" {
    try expectSettledAfterEarlyStop(1);
}

test "a non-stream early stop inside a speculative block settles the whole block's accounting" {
    try expectSettledAfterEarlyStop(4);
}

fn expectNonStreamStopsAt(stop: []const u8, pieces: []const []const u8, stop_token: usize) !void {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmpty(a);
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
    var tok = Tokenizer.initEmpty(a);
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
    var tok = Tokenizer.initEmpty(a);
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

test "a streamed result carries the prefix reuse and the timings its stream finalized" {
    var text = [_]u8{'x'};
    var ids = [_]u32{7};
    const ts = StreamingTokenStream{
        .mode = .regular,
        .eos_token_ids = &.{},
        .prompt_tokens = 683,
        .cached_tokens = 457,
        .completion_tokens = 20,
        .prefill_ns = 500_000_000,
        .decode_ns = 1_000_000_000,
        .finish_details = "repetition_loop",
    };
    const r = ts.result(&text, &ids, "stop", 3, "cd");
    try std.testing.expectEqual(@as(u32, 683), r.prompt_tokens);
    try std.testing.expectEqual(@as(u32, 457), r.cached_tokens);
    try std.testing.expectEqual(@as(u32, 20), r.completion_tokens);
    try std.testing.expectEqual(@as(u64, 500_000_000), r.prefill_ns);
    try std.testing.expectEqual(@as(u64, 1_000_000_000), r.decode_ns);
    try std.testing.expectEqual(generate_mod.prefillTokensPerSec(683, 457, 500_000_000), r.prefill_tps);
    try std.testing.expectEqual(@as(f64, 20), r.decode_tps);
    try std.testing.expectEqualStrings("repetition_loop", r.finish_details.?);
    try std.testing.expectEqual(@as(?usize, 3), r.constraint_payload_byte);
    // The stream's own end: the stop its gate matched, which `reply.settle` reads instead of searching the cut text.
    try std.testing.expectEqualStrings("cd", generatedOf(&r).source.gated.?);
}

/// The answer `sendNamedLoadRefusal` writes for `err`, read back off a socket pair. Only the scheduler's registry is read.
fn answerOf(id: []const u8, err: anyerror, report: *const scheduler_mod.LoadReport, reg: *model_registry_mod.ModelRegistry, out: []u8) ![]u8 {
    var sv: [2]std.posix.fd_t = undefined;
    try std.testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, std.testing.io);
    var sch: scheduler_mod.Scheduler = undefined;
    sch.registry = reg;
    try std.testing.expect(try sendNamedLoadRefusal(std.testing.allocator, &conn, &sch, id, err, report));
    _ = std.posix.system.shutdown(sv[1], std.posix.SHUT.WR);
    conn.close();
    var got: usize = 0;
    while (got < out.len) {
        const n = std.c.recv(sv[1], out[got..].ptr, out.len - got, 0);
        if (n <= 0) break;
        got += @intCast(n);
    }
    return out[0..got];
}

test "a stream's final chunk names the call its reply declined, ahead of a thought's note" {
    const a = std.testing.allocator;
    const named: reply_mod.Declined = .{ .name = try a.dupe(u8, "read_file") };
    defer named.deinit(a);
    const field = try reply_mod.detailsField(a, "finish_details", named, "");
    defer a.free(field);
    const Case = struct { finish: Finish, want: []const u8 };
    const cases = [_]Case{
        .{ .finish = .{ .reason = "stop" }, .want = "\"finish_reason\":\"stop\"}" },
        .{ .finish = .{ .reason = "stop", .details = reply_mod.thought_continued }, .want = "\"finish_details\":{\"type\":\"continued_after_thought\"" },
        .{ .finish = .{ .reason = "length", .declined_field = field }, .want = "\"finish_reason\":\"length\",\"finish_details\":{\"type\":\"tool_call_cut\",\"name\":\"read_file\"}" },
        .{ .finish = .{ .reason = "stop", .details = reply_mod.thought_continued, .declined_field = field }, .want = "\"finish_reason\":\"stop\",\"finish_details\":{\"type\":\"tool_call_cut\",\"name\":\"read_file\"}" },
    };
    for (cases) |c| {
        var sv: [2]std.posix.fd_t = undefined;
        try std.testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
        defer _ = std.c.close(sv[1]);
        var conn: Conn = undefined;
        Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, std.testing.io);
        try sendSSEChunk(a, &conn, 1, "m", .{ .role = null, .content = null }, c.finish, null, null);
        _ = std.posix.system.shutdown(sv[1], std.posix.SHUT.WR);
        conn.close();
        var buf: [4096]u8 = undefined;
        var got: usize = 0;
        while (got < buf.len) {
            const n = std.c.recv(sv[1], buf[got..].ptr, buf.len - got, 0);
            if (n <= 0) break;
            got += @intCast(n);
        }
        errdefer std.debug.print("chunk: {s}\n", .{buf[0..got]});
        try std.testing.expect(std.mem.indexOf(u8, buf[0..got], c.want) != null);
    }
}

test "a files read that ran out of memory is answered with its own refusal, not an earlier load's context figures" {
    var reg = try model_registry_mod.ModelRegistry.init(std.testing.allocator, std.testing.io, null, 2, 0, null);
    defer reg.deinit();
    const e = try reg.registerStub("b", "b", 10);
    // An earlier load of b was refused for its context, which the entry keeps.
    e.memory_context_refusal = .{ .requested = 500000, .maximum = 128000, .needed = 100, .available = 90, .kv_bits = 8, .chunk = 2048 };
    const report: scheduler_mod.LoadReport = .{ .failed = error.OutOfMemory };
    var buf: [4096]u8 = undefined;
    const raw = try answerOf("b", error.InsufficientMemory, &report, reg, &buf);
    try std.testing.expect(std.mem.indexOf(u8, raw, "500000") == null);
}

test "a continued thought keeps the request's forced tool_choice: the call is committed at the continuation's first token on every family" {
    const a = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var tok = try tokenizer_mod.byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_start|>", "<|im_end|>", "<arg_key>" });
    defer tokenizer_mod.deinitTestTokenizer(&tok);
    const families = @import("arch_fixtures_test.zig").forced_families;
    if (families.len == 0) return error.SkipZigTest;
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
    const user: chat_mod.Message = .{ .role = "user", .content = "What time is it?" };
    for (families) |f| {
        var config = chat_mod.ChatConfig{ .chat_template = f.tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        try config.compile();
        defer config.dropCompiled();
        for ([_]tool_calls_mod.ToolChoice{ .required, .{ .named = "get_time" } }) |choice| {
            const instruction = try choice.instruction(a);
            defer if (instruction) |i| a.free(i);
            const r: ThoughtResume = .{ .chat_config = &config, .messages = &.{user}, .tools_json = tools, .tool_choice_instruction = instruction, .enable_thinking = true, .effort = null, .stops = &.{}, .has_tools = true, .allow_parallel = true, .tool_choice = choice };
            const turns = [_]chat_mod.Message{ user, .{ .role = "assistant", .content = "", .reasoning_content = "hmm" } };
            const prompt = try chat_mod.formatChat(a, &tok, &turns, &config, tools, instruction, true, null, true);
            defer a.free(prompt);
            var force: ?generate_mod.CallForce = null;
            defer if (force) |*cf| cf.deinit(a);
            const sampling = continuationSampling(a, &tok, &r, prompt, .{}, &force);
            errdefer std.debug.print("[{s}] {s}\n", .{ f.head, @tagName(choice) });
            const cf = sampling.call_force orelse return error.ForceNotCarried;
            // The thought is closed in the prompt: the call starts at the continuation's first token.
            var published = std.ArrayList(u32).empty;
            defer published.deinit(a);
            while (true) switch (cf.decide(published.items)) {
                .force => |id| try published.append(a, id),
                .choose => |ids| try published.append(a, ids[0]),
                .sample => if (cf.pending()) return error.CallNotAtFirstToken else break,
            };
            const text = try tok.decode(a, published.items);
            defer a.free(text);
            const want = try std.mem.concat(a, u8, &.{ f.head, "get_time", f.name_tail });
            defer a.free(want);
            try std.testing.expectEqualStrings(want, text);
        }
    }
}

/// A tool whose call the cells below stream; `tool_call_head` is the text up to the first argument's value.
const stream_cell_tools =
    \\[{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
;
const stream_cell_head = "Writing it.\n<tool_call>\n<function=write_file>\n<parameter=path>\nreport.md\n</parameter>\n<parameter=content>\n";

test "tool stream: a chat call that buffers for 23 s sends a data event within every beat interval, never only a comment" {
    const a = std.testing.allocator;
    var wire: http_conn.TestWire = undefined;
    try wire.init();
    defer wire.deinit();
    var progress: tool_progress.Progress = .{};
    defer progress.deinit(a);
    var buffered = std.ArrayList(u8).empty;
    defer buffered.deinit(a);
    var since: i64 = 0;
    var worst: i64 = 0;
    for (0..230) |i| {
        wire.advance(100);
        since += 100;
        try buffered.appendSlice(a, if (i == 0) stream_cell_head else "- item: pending review\n");
        if (!wire.conn.keepaliveDue()) continue;
        worst = @max(worst, since);
        since = 0;
        try beatToolCall(a, &wire.conn, 7, "m", &progress, buffered.items, .{ .dialect = null }, stream_cell_tools);
    }
    var buf: [8192]u8 = undefined;
    const raw = wire.read(&buf);
    try std.testing.expect(worst <= Conn.STREAM_KEEPALIVE_MS);
    try std.testing.expect(std.mem.indexOf(u8, raw, ": keepalive") == null);
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, raw, "data: "));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, raw, "\"name\":\"write_file\""));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, raw, "\"arguments\":\"\""));
}
