const std = @import("std");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const stop_seq_mod = @import("stop_sequences.zig");
const expert_stream_mod = @import("expert_stream.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const scheduler_mod = @import("scheduler.zig");
const generate_mod = @import("generate.zig");
const slot_admission = @import("slot_admission.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const generation_stream = @import("generation_stream.zig");
const EMPTY_PROMPT_MSG = generation_stream.EMPTY_PROMPT_MSG;
const sendGenerationError = generation_stream.sendGenerationError;
const nonStreamingViaScheduler = generation_stream.nonStreamingViaScheduler;
const StreamingTokenStream = generation_stream.StreamingTokenStream;
const requestSpecModes = generation_stream.requestSpecModes;
const noteMtpRefused = generation_stream.noteMtpRefused;
const pickStreamMode = generation_stream.pickStreamMode;
const contextOverflowMessage = generation_stream.contextOverflowMessage;
const formatTimingsObject = generation_stream.formatTimingsObject;
const formatPerfBracket = generation_stream.formatPerfBracket;
const sendStreamIdle = generation_stream.sendStreamIdle;
const http_conn = @import("http_conn.zig");
const Conn = http_conn.Conn;
const sendSseHeaders = http_conn.sendSseHeaders;
const request_policy = @import("request_policy.zig");
const resolveSamplingDefault = request_policy.resolveSamplingDefault;
const resolveThinkPenalty = request_policy.resolveThinkPenalty;
const armThinkPenalty = request_policy.armThinkPenalty;
const omittedMaxTokensDefault = request_policy.omittedMaxTokensDefault;
const launchMaxTokensDefault = request_policy.launchMaxTokensDefault;
const resolveRequestMaxTokens = request_policy.resolveRequestMaxTokens;
const maxTokensOrigin = request_policy.maxTokensOrigin;
const describeMaxTokens = request_policy.describeMaxTokens;
const clampMaxTokens = request_policy.clampMaxTokens;
const parseRepeatPenalty = request_policy.parseRepeatPenalty;
const parseAdditivePenalty = request_policy.parseAdditivePenalty;
const logprobs = @import("logprobs.zig");
const StreamLogprobs = logprobs.StreamLogprobs;
const formatCompletionsLogprobs = logprobs.formatCompletionsLogprobs;
const io_util = @import("io_util.zig");
const sampler = @import("sampler.zig");
const nowSecs = io_util.nowSecs;
const nowMs = io_util.nowMs;
const Stopwatch = io_util.Stopwatch;
const nextCompletionId = io_util.nextCompletionId;
const testing = std.testing;
const byteTokenizerForTests = tokenizer_mod.byteTokenizerForTests;
const deinitTestTokenizer = tokenizer_mod.deinitTestTokenizer;
const server_mod = @import("server.zig");
const MAX_TOKENS_DESC_LEN = server_mod.MAX_TOKENS_DESC_LEN;
const SSE_ALLOW_HEADERS_DEFAULT = server_mod.SSE_ALLOW_HEADERS_DEFAULT;
const admitMtpForCtx = server_mod.admitMtpForCtx;
const chatIgnoreEosRejectReason = server_mod.chatIgnoreEosRejectReason;
const checkAttentionMemory = server_mod.checkAttentionMemory;
const countRejected = server_mod.countRejected;
const defaultEnableMtpForStreaming = server_mod.defaultEnableMtpForStreaming;
const finishDetailsField = server_mod.finishDetailsField;
const forceMtpFor = server_mod.forceMtpFor;
const getEffectiveContextLength = server_mod.getEffectiveContextLength;
const getTimeoutNs = server_mod.getTimeoutNs;
const ignoreEosRequested = server_mod.ignoreEosRequested;
const jsonEscape = server_mod.jsonEscape;
const jsonEscapeOrEmpty = server_mod.jsonEscapeOrEmpty;
const logHttpSseData = server_mod.logHttpSseData;
const mtpCapable = server_mod.mtpCapable;
const nChoicesRejectReason = server_mod.nChoicesRejectReason;
const nativeMeasuredMoeHead = server_mod.nativeMeasuredMoeHead;
const parseJsonFloatOpt = server_mod.parseJsonFloatOpt;
const parseJsonTopKOpt = server_mod.parseJsonTopKOpt;
const parseKvQuantOverride = server_mod.parseKvQuantOverride;
const parseRequestSeed = server_mod.parseRequestSeed;
const requestBillsMtpHead = server_mod.requestBillsMtpHead;
const requestCacheKey = server_mod.requestCacheKey;
const sendErrorResponse = server_mod.sendErrorResponse;
const sendResponse = server_mod.sendResponse;

/// vLLM's `ignore_eos`: an EOS id does not end the reply, so it runs to `max_tokens`. Stop
/// sequences and the loop stops still end it.
fn requestEosSlice(config: *const model_mod.ModelConfig, root: std.json.ObjectMap) []const u32 {
    return if (ignoreEosRequested(root)) &.{} else config.eosTokenSlice();
}

/// vLLM's default `skip_special_tokens` for a completion that decodes past EOS: the
/// `special: true` turn markers it writes there stay out of its text, on both paths.
pub fn completionShowsToken(tok: *const Tokenizer, id: u32, skip_special: bool) bool {
    if (!skip_special) return true;
    for (tok.flagged_specials) |sp| if (sp.id == id) return false;
    return true;
}

pub fn completionText(allocator: std.mem.Allocator, tok: *const Tokenizer, ids: []const u32, skip_special: bool) ![]u8 {
    if (!skip_special) return tok.decode(allocator, ids);
    var shown = std.ArrayList(u32).empty;
    defer shown.deinit(allocator);
    for (ids) |id| if (completionShowsToken(tok, id, true)) try shown.append(allocator, id);
    return tok.decode(allocator, shown.items);
}

fn withTextPromptBos(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, body: []const u32) ![]u32 {
    const bos = config.textPromptBosId() orelse return allocator.dupe(u32, body);
    const ids = try allocator.alloc(u32, body.len + 1);
    ids[0] = bos;
    @memcpy(ids[1..], body);
    return ids;
}

const CompletionPrompt = union(enum) {
    text: []const u8,
    /// Caller frees.
    ids: []u32,
    invalid: []const u8,
    missing,
};

/// `/v1/completions` `prompt`: a string or token ids (OpenAI's `int[]`, what lm-eval sends),
/// either also as a one-item list. A real batch is a named 400, like `n`.
fn parseCompletionPrompt(allocator: std.mem.Allocator, v: ?std.json.Value, vocab_size: u32) !CompletionPrompt {
    const val = v orelse return .missing;
    switch (val) {
        .null => return .missing,
        .string => |s| return .{ .text = s },
        .array => |arr| {
            if (arr.items.len == 0) return .{ .invalid = "'prompt' is empty" };
            if (arr.items[0] == .string or arr.items[0] == .array) {
                if (arr.items.len > 1) return .{ .invalid = "batched prompts are not supported: send one prompt per request" };
                return parseCompletionPrompt(allocator, arr.items[0], vocab_size);
            }
            const ids = try allocator.alloc(u32, arr.items.len);
            for (arr.items, ids) |item, *id| {
                if (item != .integer or item.integer < 0 or item.integer >= vocab_size) {
                    allocator.free(ids);
                    return .{ .invalid = "'prompt' token ids must be integers inside the model's vocabulary" };
                }
                id.* = @intCast(item.integer);
            }
            return .{ .ids = ids };
        },
        else => return .{ .invalid = "'prompt' must be a string or a list of token ids" },
    }
}

/// Prompt-token logprobs are not computed, so an echo would hand lm-eval an empty sum it
/// reads as perplexity 1; refused by name instead.
fn echoRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    const v = root.get("echo") orelse return null;
    if (v != .bool or !v.bool) return null;
    return "'echo' is not supported: the prompt is not echoed and its tokens carry no logprobs";
}

pub fn handleCompletions(
    allocator: std.mem.Allocator,
    stream: *Conn,
    body: []const u8,
    lm: *LoadedModel,
) !void {
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        log.warn("POST /v1/completions -> 400 (invalid JSON)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON in request body", 400);
        return;
    };
    defer parsed.deinit();

    if (parsed.value != .object) {
        log.warn("POST /v1/completions -> 400 (body is not a JSON object)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400);
        return;
    }
    const root = parsed.value.object;
    const cache_key = requestCacheKey(root);
    const kv_quant_override = parseKvQuantOverride(root);

    if (nChoicesRejectReason(root)) |reason| {
        log.warn("POST /v1/completions -> 400 (unsupported n)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    // Legacy-completions `logprobs` is an INTEGER (how many alternatives per
    // token), not chat's bool + `top_logprobs`. It was parsed nowhere and the
    // handler passed a hardcoded 0 while still emitting a `logprobs` key — a
    // silently ignored field, which reads to a client as "this model has no
    // opinion" rather than "this server never asked".
    const logprobs_n: u32 = if (root.get("logprobs")) |v| switch (v) {
        .integer => |i| @intCast(@min(@max(i, 0), 20)),
        else => 0,
    } else 0;

    if (echoRejectReason(root)) |reason| {
        log.warn("POST /v1/completions -> 400 (unsupported echo)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
        return;
    }

    const prompt = try parseCompletionPrompt(allocator, root.get("prompt"), @min(config.vocab_size, tok.definedVocabSize()));
    defer if (prompt == .ids) allocator.free(prompt.ids);
    switch (prompt) {
        .missing, .invalid => {
            const reason = if (prompt == .invalid) prompt.invalid else "'prompt' is a required field";
            log.warn("POST /v1/completions -> 400 ({s})\n", .{reason});
            try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", reason, 400);
            return;
        },
        .text, .ids => {},
    }

    const requested_max_tokens = root.get("max_tokens") orelse root.get("max_completion_tokens");
    const max_tokens: u32 = resolveRequestMaxTokens(
        requested_max_tokens,
        omittedMaxTokensDefault(config),
    );
    // For the request log line only: which of the three sources this came from.
    const max_tokens_origin = maxTokensOrigin(requested_max_tokens, launchMaxTokensDefault(), config.defaultMaxTokens());

    const is_stream = if (root.get("stream")) |v| v == .bool and v.bool else false;

    var sampling = requestSampling(config, root);

    // Parse stop sequences
    var stop_sequences = std.ArrayList([]const u8).empty;
    defer stop_sequences.deinit(allocator);
    if (root.get("stop")) |stop_val| {
        switch (stop_val) {
            .string => |s| if (s.len > 0) try stop_sequences.append(allocator, s),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .string and item.string.len > 0) try stop_sequences.append(allocator, item.string);
                }
            },
            else => {},
        }
    }

    const model_name = if (root.get("model")) |v|
        (if (v == .string) v.string else config.model_type)
    else
        config.model_type;

    const include_usage = if (root.get("stream_options")) |so| blk: {
        if (so != .object) break :blk false;
        if (so.object.get("include_usage")) |iu| {
            break :blk iu == .bool and iu.bool;
        }
        break :blk false;
    } else false;

    // Spec-decode flags (mirror chat-completions).
    if (expert_stream_mod.mtpRefusal(config.expert_streaming, if (root.get("enable_mtp")) |v| (v == .bool and v.bool) else false, lm.mtp != null)) |why| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", why, 400);
        return;
    }
    const spec_request = requestSpec(config, root, lm.dflash != null, lm.mtp != null, nativeMeasuredMoeHead(lm));
    const enable_drafter = spec_request.drafter;
    const allow_batch_mtp = spec_request.allow_batch_mtp;
    var enable_mtp = spec_request.mtp;

    // Log the request
    var max_tokens_desc_buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    log.info("POST /v1/completions (max_tokens={s}, temp={d:.2}, top_p={d:.2}, top_k={d}, stream={}) \n", .{ describeMaxTokens(&max_tokens_desc_buf, max_tokens, max_tokens_origin), sampling.temperature, sampling.top_p, sampling.top_k, is_stream });
    switch (prompt) {
        .text => |s| log.info("  > \"{s}{s}\"\n", .{ s[0..@min(s.len, 80)], if (s.len > 80) "..." else "" }),
        .ids => |ids| log.info("  > [{d} token ids]\n", .{ids.len}),
        else => unreachable,
    }

    // Tokenize prompt directly (no chat template); token ids are taken as sent.
    const prompt_ids = switch (prompt) {
        .text => |s| blk: {
            const piece = try tok.encode(allocator, s);
            defer allocator.free(piece);
            break :blk try withTextPromptBos(allocator, config, piece);
        },
        .ids => |ids| try allocator.dupe(u32, ids),
        else => unreachable,
    };
    defer allocator.free(prompt_ids);
    if (prompt_ids.len == 0) {
        log.warn("POST /v1/completions -> 400 (prompt tokenizes to zero tokens)\n", .{});
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", EMPTY_PROMPT_MSG, 400);
        return;
    }
    enable_mtp = admitMtpForCtx(enable_mtp, prompt_ids.len);

    // Enforce context size limit
    const effective_ctx = getEffectiveContextLength(config);
    if (prompt_ids.len > effective_ctx) {
        log.warn("POST /v1/completions -> 400 (prompt {d} tokens exceeds ctx_size {d})\n", .{ prompt_ids.len, effective_ctx });
        countRejected();
        var ovf_buf: [160]u8 = undefined;
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", contextOverflowMessage(&ovf_buf, prompt_ids.len, effective_ctx), 400);
        return;
    }

    // Clamp max_tokens first: the memory guard bills the reserved generation headroom, and an
    // omitted max_tokens is the maxInt(u32)/4 sentinel.
    const effective_max_tokens = clampMaxTokens(max_tokens, prompt_ids.len, effective_ctx);

    // Check if attention computation would exceed GPU memory.
    const bill_head = requestBillsMtpHead(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), .{ .repeat_penalty = sampling.repeat_penalty, .presence_penalty = sampling.presence_penalty, .frequency_penalty = sampling.frequency_penalty }, logprobs_n);
    if (!try checkAttentionMemory(allocator, stream, prompt_ids, effective_max_tokens, config, lm.id, false, kv_quant_override, false, bill_head)) return;

    const eos_slice = requestEosSlice(config, root);
    sampling.think_penalty = requestThinkPenalty(allocator, lm, root, prompt_ids);
    const request_bias: []const @import("logit_bias.zig").Bias = if (root.get("logit_bias")) |bias_value| @import("logit_bias.zig").request(allocator, bias_value, @min(config.vocab_size, tok.definedVocabSize())) catch |err| {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", @errorName(err), null);
        return;
    } else &.{};
    defer allocator.free(request_bias);
    sampling.think_penalty.biases = request_bias;

    if (is_stream) {
        handleStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, ignoreEosRequested(root), stop_sequences.items, model_name, include_usage, enable_drafter, enable_mtp, allow_batch_mtp, logprobs_n, cache_key, kv_quant_override) catch |err| {
            log.err("  -> streaming error: {}\n", .{err});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    } else {
        handleNonStreamingCompletion(allocator, stream, lm, tok, prompt_ids, effective_max_tokens, sampling, eos_slice, ignoreEosRequested(root), stop_sequences.items, model_name, enable_drafter, enable_mtp, allow_batch_mtp, logprobs_n, cache_key, kv_quant_override) catch |err| {
            log.err("  -> {s}\n", .{@errorName(err)});
            sendGenerationError(allocator, stream, err, .openai) catch {};
        };
    }
}

/// The sampling a completions request resolves, before the think penalty: the body, then the launch flags,
/// then the model's generation config.
fn requestSampling(config: *const model_mod.ModelConfig, root: std.json.ObjectMap) sampler.SamplingParams {
    return .{
        .temperature = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "temperature", 0.0, 2.0), server_mod.server_config.default_temperature, config.gen_temperature, 1.0),
        .top_p = resolveSamplingDefault(f32, parseJsonFloatOpt(root, "top_p", 0.0, 1.0), server_mod.server_config.default_top_p, config.gen_top_p, 1.0),
        .top_k = resolveSamplingDefault(u32, parseJsonTopKOpt(root, "top_k"), server_mod.server_config.default_top_k, config.gen_top_k, 0),
        .repeat_penalty = parseRepeatPenalty(root),
        .presence_penalty = parseAdditivePenalty(root, "presence_penalty"),
        .frequency_penalty = parseAdditivePenalty(root, "frequency_penalty"),
        .seed = parseRequestSeed(root.get("seed")),
    };
}

fn requestThinkPenalty(allocator: std.mem.Allocator, lm: *LoadedModel, root: std.json.ObjectMap, prompt_ids: []const u32) @import("think_penalty.zig").ThinkPenalty {
    const config = lm.config.?;
    return armThinkPenalty(allocator, lm, lm.tokenizer.?, prompt_ids, true, resolveThinkPenalty(root, model_settings.think_penalty_flag, config.think_penalty_override));
}

const SpecRequest = struct { drafter: bool, mtp: bool, allow_batch_mtp: bool };

/// The speculative switches a request resolves: its own fields, else what the loaded model defaults to.
fn requestSpec(config: *const model_mod.ModelConfig, root: std.json.ObjectMap, dflash_loaded: bool, mtp_loaded: bool, native_measured: bool) SpecRequest {
    const drafter = if (root.get("enable_drafter")) |v| (v == .bool and v.bool) else dflash_loaded;
    const mtp = if (root.get("enable_mtp")) |v|
        (v == .bool and v.bool)
    else
        defaultEnableMtpForStreaming(mtp_loaded, config.isMoe(), forceMtpFor(config), native_measured, model_mod.isServedArch(config.model_type), config.expert_streaming and !config.stream_mtp_head);
    return .{
        .drafter = drafter and dflash_loaded,
        .mtp = mtp and mtp_loaded,
        .allow_batch_mtp = if (root.get("enable_batch_mtp")) |v| v != .bool or v.bool else true,
    };
}

/// A non-streaming completion of token ids with no connection, for `sushi bench` in this process: the request
/// `handleCompletions` resolves from `{"prompt": ids, "max_tokens": n, "temperature": 0, "ignore_eos": true}`,
/// through the same resolution and the same scheduler path.
pub fn completeIds(allocator: std.mem.Allocator, lm: *LoadedModel, ids: []const u32, max_tokens: u32) !generate_mod.GenerationResult {
    const tok = lm.tokenizer.?;
    const config = lm.config.?;
    var body_buf: [96]u8 = undefined;
    const body = try std.fmt.bufPrint(&body_buf, "{{\"max_tokens\":{d},\"temperature\":0,\"ignore_eos\":true}}", .{max_tokens});
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const effective_ctx = getEffectiveContextLength(config);
    if (ids.len > effective_ctx) return error.PromptExceedsContext;
    var sampling = requestSampling(config, root);
    sampling.think_penalty = requestThinkPenalty(allocator, lm, root, ids);
    const spec_request = requestSpec(config, root, lm.dflash != null, lm.mtp != null, nativeMeasuredMoeHead(lm));
    const enable_mtp = admitMtpForCtx(spec_request.mtp, ids.len);
    const spec = requestSpecModes(spec_request.drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, 0);

    return nonStreamingViaScheduler(allocator, server_mod.global_scheduler.?, lm, tok, ids, ids, clampMaxTokens(max_tokens, ids.len, effective_ctx), sampling, requestEosSlice(config, root), 0, false, false, spec.use_drafter, spec.use_mtp, spec_request.allow_batch_mtp, getTimeoutNs(), null, &.{}, requestCacheKey(root), .{}, 0, parseKvQuantOverride(root), null, .{ .stops = &.{}, .completion_skip_special = true });
}

fn handleNonStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    skip_special: bool,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    logprobs_n: u32,
    cache_key: u64,
    kv_quant_override: ?transformer_mod.KVQuantConfig,
) !void {
    var timer = Stopwatch.init(stream.io);

    // Spec dispatch: `requestSpecModes` (DFlash > MTP).
    // logprobs needs every step's own distribution, so it disables speculation
    // here exactly as it does on chat.
    const spec = requestSpecModes(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, logprobs_n);
    const use_mtp = spec.use_mtp;
    const use_drafter = spec.use_drafter;

    // Every failure class propagates to the surface's one error arm (`sendGenerationError`), shared with the streaming twin.
    var result = try nonStreamingViaScheduler(allocator, server_mod.global_scheduler.?, lm, tok, prompt_ids, prompt_ids, max_tokens, sampling, eos_token_ids, 0, false, false, use_drafter, use_mtp, allow_batch_mtp, getTimeoutNs(), null, &.{}, cache_key, .{}, logprobs_n, kv_quant_override, stream, .{ .stops = stop_sequences, .completion_skip_special = skip_special });
    _ = &result;
    defer allocator.free(result.text);
    defer allocator.free(result.token_ids);

    // Text completion is a raw continuation: FIM clients rely on exact
    // indentation (stream and non-stream bytes match).
    const raw_text = try completionText(allocator, tok, result.token_ids, skip_special);
    defer allocator.free(raw_text);

    var final_text: []const u8 = raw_text;
    var finish_reason = result.finish_reason;
    if (stop_seq_mod.earliest(final_text, stop_sequences)) |hit| {
        final_text = final_text[0..hit.index];
        finish_reason = "stop";
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    log.info("  <- {d}+{d} tokens ({d}ms) [{s}] [{s}]\n", .{
        result.prompt_tokens, result.completion_tokens, elapsed_ms, perf, finish_reason,
    });

    const escaped = jsonEscapeOrEmpty(allocator, final_text);
    const escaped_text = escaped.slice;
    defer if (escaped.owned) allocator.free(escaped.slice);

    // `null` unless the request asked: OpenAI omits the field's content rather
    // than shipping an empty object that reads as "no alternatives exist".
    var lp_offset_base: usize = 0;
    const lp_json: []const u8 = if (logprobs_n > 0 and result.logprobs != null)
        try formatCompletionsLogprobs(allocator, tok, result.token_ids, result.logprobs.?, &lp_offset_base)
    else
        "null";
    defer if (!std.mem.eql(u8, lp_json, "null")) allocator.free(lp_json);

    const timings_field = try completionTimingsField(allocator, result.prompt_tokens, result.cached_tokens, result.completion_tokens, result.prefill_ns, result.decode_ns);
    defer allocator.free(timings_field);

    // finish_reason always carries finish_details beside it.
    const response = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"text":{s},"logprobs":{s},"finish_reason":"{s}"{s}}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}{s}}}
    , .{
        nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io)),
        nowSecs(stream.io),
        model_name,
        escaped_text,
        lp_json,
        finish_reason,
        finishDetailsField(finish_reason, result.finish_details),
        result.prompt_tokens,
        result.completion_tokens,
        result.prompt_tokens + result.completion_tokens,
        timings_field,
    });
    defer allocator.free(response);

    try sendResponse(stream, "200 OK", "application/json", response);
}

/// `,"timings":{...}` for a non-streaming completion, empty when nothing was timed. The same object a chat reply carries.
fn completionTimingsField(allocator: std.mem.Allocator, prompt_tokens: u32, cached_tokens: u32, completion_tokens: u32, prefill_ns: u64, decode_ns: u64) ![]u8 {
    const obj = try formatTimingsObject(allocator, prompt_tokens, cached_tokens, completion_tokens, prefill_ns, decode_ns, 0);
    defer allocator.free(obj);
    if (obj.len == 0) return allocator.alloc(u8, 0);
    return std.fmt.allocPrint(allocator, ",\"timings\":{s}", .{obj});
}

fn handleStreamingCompletion(
    allocator: std.mem.Allocator,
    stream: *Conn,
    lm: *LoadedModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    skip_special: bool,
    stop_sequences: []const []const u8,
    model_name: []const u8,
    include_usage: bool,
    enable_drafter: bool,
    enable_mtp: bool,
    allow_batch_mtp: bool,
    logprobs_n: u32,
    cache_key: u64,
    kv_quant_override: ?transformer_mod.KVQuantConfig,
) !void {
    const cmpl_id = nextCompletionId(&server_mod.completion_id_counter, nowMs(stream.io));
    const created_ts = nowSecs(stream.io);
    var timer = Stopwatch.init(stream.io);

    const stream_mode = pickStreamMode(enable_drafter, enable_mtp, lm.dflash != null, mtpCapable(lm), sampling, logprobs_n);
    if (stream_mode == .drafter) log.info("  drafter=enabled (streaming, block_size={d})\n", .{lm.drafter_block_size});
    if (stream_mode == .mtp) log.info("  mtp=enabled (streaming, depth={d})\n", .{lm.mtp_depth});
    noteMtpRefused(enable_mtp, mtpCapable(lm), sampling, logprobs_n);

    var slot_handle: ?*scheduler_mod.Slot = null;
    defer if (slot_handle) |s| server_mod.global_scheduler.?.complete(s);

    const sch = server_mod.global_scheduler.?;
    slot_handle = try sch.submit(.{
        .model = lm,
        .prompt_ids = prompt_ids,
        .full_prompt = prompt_ids,
        .cached_tokens = 0,
        .has_tools = false,
        .enable_thinking = false,
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
        .cache_key = cache_key,
        .kv_quant_config = kv_quant_override,
    });
    var ts = StreamingTokenStream.initFromSlot(slot_handle.?, stream_mode, eos_token_ids);

    // Legacy completions carries logprobs as four parallel arrays whose
    // `text_offset` indexes the whole completion, so the collector renders
    // the legacy shape and keeps the running offset across chunks.
    var lps = StreamLogprobs{
        .allocator = allocator,
        .tok = tok,
        .slot = slot_handle,
        .enabled = logprobs_n > 0,
        .legacy = true,
    };
    defer lps.deinit();
    defer ts.deinit(allocator);

    // SSE headers
    try sendSseHeaders(stream, "completions", SSE_ALLOW_HEADERS_DEFAULT);

    var stop_gate = stop_seq_mod.Gate{ .stops = stop_sequences };
    defer stop_gate.deinit(allocator);
    var tail_flush = false; // the generation ended: this turn delivers the gate's held tail
    var stopped = false;
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
            try lps.note(token_id);
            if (!completionShowsToken(tok, token_id, skip_special)) continue;
            const raw_decoded_c = try tok.decode(allocator, &[_]u32{token_id});
            defer allocator.free(raw_decoded_c);
            break :blk try stop_gate.push(allocator, raw_decoded_c);
        };
        defer allocator.free(token_text);

        stopped = stop_gate.matched != null;
        if (stopped) slot_handle.?.stop_hit.store(true, .release);
        if (token_text.len == 0) {
            if (stopped) break;
            continue;
        }

        const escaped = try jsonEscape(allocator, token_text);
        defer allocator.free(escaped);
        // Legacy `logprobs` is a sibling of `text` on the choice. OpenAI sends
        // `null` on a chunk that has none, never an absent key.
        const lp_field = (try lps.take()) orelse "null";
        const chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"text":{s},"logprobs":{s},"finish_reason":null}}]}}
        , .{ cmpl_id, created_ts, model_name, escaped, lp_field });
        defer allocator.free(chunk);

        logHttpSseData(chunk);
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(chunk);
        try stream.writeAllNoFlush("\n\n");
        try stream.flush();
    }

    // Final chunk with finish_reason
    ts.finalize();
    const finish_reason = if (client_gone) "client_disconnect" else if (stopped) "stop" else ts.finish_reason;
    const total_prompt = ts.prompt_tokens;

    if (!client_gone) {
        const final_lp = (try lps.take()) orelse "null";
        // finish_reason always carries finish_details beside it.
        const final_chunk = try std.fmt.allocPrint(allocator,
            \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[{{"index":0,"text":"","logprobs":{s},"finish_reason":"{s}"{s}}}]}}
        , .{ cmpl_id, created_ts, model_name, final_lp, finish_reason, finishDetailsField(finish_reason, ts.finish_details) });
        defer allocator.free(final_chunk);

        logHttpSseData(final_chunk);
        try stream.writeAllNoFlush("data: ");
        try stream.writeAllNoFlush(final_chunk);
        try stream.writeAllNoFlush("\n\n");

        // Usage rides its own empty-choices chunk (OpenAI's stream_options
        // shape) — never the finish chunk, so the ending is stated once.
        if (include_usage) {
            const usage_chunk = try std.fmt.allocPrint(allocator,
                \\{{"id":"cmpl-{d}","object":"text_completion.chunk","created":{d},"model":"{s}","system_fingerprint":"sushi","choices":[],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
            , .{ cmpl_id, created_ts, model_name, total_prompt, ts.completion_tokens, total_prompt + ts.completion_tokens });
            defer allocator.free(usage_chunk);
            logHttpSseData(usage_chunk);
            try stream.writeAllNoFlush("data: ");
            try stream.writeAllNoFlush(usage_chunk);
            try stream.writeAllNoFlush("\n\n");
        }

        logHttpSseData("[DONE]");
        try stream.writeAll("data: [DONE]\n\n");
    }

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    var perf_buf: [160]u8 = undefined;
    const perf = formatPerfBracket(&perf_buf, ts.prompt_tokens, ts.cached_tokens, ts.completion_tokens, ts.prefill_ns, ts.decode_ns);
    log.info("  <- {d}+{d} tokens streamed ({d}ms) [{s}] [{s}]\n", .{
        total_prompt, ts.completion_tokens, elapsed_ms, perf, finish_reason,
    });
}

test "parseCompletionPrompt: a token-id prompt is a prompt, a batch is a named 400" {
    const allocator = std.testing.allocator;
    const Want = union(enum) { text: []const u8, ids: []const u32, invalid, missing };
    const cases = [_]struct { body: []const u8, want: Want }{
        .{ .body = "{\"prompt\":\"hi\"}", .want = .{ .text = "hi" } },
        .{ .body = "{\"prompt\":[\"hi\"]}", .want = .{ .text = "hi" } },
        .{ .body = "{\"prompt\":[1,2,3]}", .want = .{ .ids = &.{ 1, 2, 3 } } },
        .{ .body = "{\"prompt\":[[5,6]]}", .want = .{ .ids = &.{ 5, 6 } } },
        .{ .body = "{\"prompt\":[[1],[2]]}", .want = .invalid },
        .{ .body = "{\"prompt\":[\"a\",\"b\"]}", .want = .invalid },
        .{ .body = "{\"prompt\":[]}", .want = .invalid },
        .{ .body = "{\"prompt\":[1,-2]}", .want = .invalid },
        .{ .body = "{\"prompt\":[1,100]}", .want = .invalid },
        .{ .body = "{\"prompt\":7}", .want = .invalid },
        .{ .body = "{}", .want = .missing },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = try parseCompletionPrompt(allocator, parsed.value.object.get("prompt"), 100);
        defer if (got == .ids) allocator.free(got.ids);
        switch (case.want) {
            .text => |t| try std.testing.expectEqualStrings(t, got.text),
            .ids => |ids| try std.testing.expectEqualSlices(u32, ids, got.ids),
            .invalid => try std.testing.expect(got == .invalid),
            .missing => try std.testing.expect(got == .missing),
        }
    }
}

test "a text prompt opens with BOS on DeepSeek only, never with Qwen's endoftext" {
    const a = testing.allocator;
    var ds = model_mod.ModelConfig{};
    ds.model_type = "deepseek_v41";
    ds.bos_token_id = 0;
    const with = try withTextPromptBos(a, &ds, &.{ 671, 6102 });
    defer a.free(with);
    try testing.expectEqualSlices(u32, &.{ 0, 671, 6102 }, with);

    var qwen = model_mod.ModelConfig{};
    qwen.model_type = "qwen4_exp";
    qwen.bos_token_id = 248044;
    const without = try withTextPromptBos(a, &qwen, &.{ 671, 6102 });
    defer a.free(without);
    try testing.expectEqualSlices(u32, &.{ 671, 6102 }, without);
}

test "echoRejectReason: echo is refused by name, never silently dropped" {
    const allocator = std.testing.allocator;
    for ([_]struct { body: []const u8, rejected: bool }{
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"echo\":false}", .rejected = false },
        .{ .body = "{\"echo\":true}", .rejected = true },
    }) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(case.rejected, echoRejectReason(parsed.value.object) != null);
    }
}

test "ignore_eos: a completion's text past EOS drops the special turn markers, the same on both paths" {
    const t = std.testing;
    const a = t.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    var tok = try byteTokenizerForTests(a, arena_state.allocator(), &.{ "<think>", "<|im_end|>", "<|endoftext|>" });
    defer deinitTestTokenizer(&tok);
    // `<think>` is an added token but not `special: true`, as in the Qwen vocabulary.
    tok.flagged_specials = &.{ .{ .id = 257, .content = "<|im_end|>" }, .{ .id = 258, .content = "<|endoftext|>" } };
    const ids = [_]u32{ 'P', 'a', 'r', 'i', 's', 257, '\n', 258, 256, 'x' };

    const kept = try completionText(a, &tok, &ids, false);
    defer a.free(kept);
    try t.expectEqualStrings("Paris<|im_end|>\n<|endoftext|><think>x", kept);
    const skipped = try completionText(a, &tok, &ids, true);
    defer a.free(skipped);
    try t.expectEqualStrings("Paris\n<think>x", skipped);
    // The stream decodes token by token and drops what `completionShowsToken` hides.
    var streamed = std.ArrayList(u8).empty;
    defer streamed.deinit(a);
    for (ids) |id| {
        if (!completionShowsToken(&tok, id, true)) continue;
        const piece = try tok.decode(a, &.{id});
        defer a.free(piece);
        try streamed.appendSlice(a, piece);
    }
    try t.expectEqualStrings(skipped, streamed.items);
}

test "ignore_eos: a completion that sets it decodes past EOS; a chat request that sets it is refused by name" {
    const t = std.testing;
    var config = model_mod.ModelConfig{};
    config.addEosToken(248044);
    config.addEosToken(248046);
    var arena_state = std.heap.ArenaAllocator.init(t.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    for ([_]struct { body: []const u8, stops: usize }{
        .{ .body = "{\"ignore_eos\":true}", .stops = 0 },
        .{ .body = "{\"ignore_eos\":false}", .stops = 2 },
        .{ .body = "{\"max_tokens\":64}", .stops = 2 },
        .{ .body = "{\"ignore_eos\":1}", .stops = 2 },
    }) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        try t.expectEqual(c.stops, requestEosSlice(&config, parsed.value.object).len);
        try t.expectEqual(c.stops == 0, chatIgnoreEosRejectReason(parsed.value.object) != null);
    }
}

test "completionTimingsField: a non-streaming completion reports cached and prefill timings" {
    const a = testing.allocator;
    const field = try completionTimingsField(a, 8223, 31, 512, 2_000_000_000, 4_000_000_000);
    defer a.free(field);
    try testing.expect(std.mem.startsWith(u8, field, ",\"timings\":{"));
    try testing.expect(std.mem.indexOf(u8, field, "\"cached_n\":31") != null);
    try testing.expect(std.mem.indexOf(u8, field, "\"prompt_ms\":2000.000") != null);
    const none = try completionTimingsField(a, 1, 0, 1, 0, 0);
    defer a.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);
}

fn rootOf(arena: std.mem.Allocator, body: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{})).object;
}

test "completions sampling: the body outranks the launch flag, which outranks the generation config, which outranks the fallback" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var config = model_mod.ModelConfig{};
    const bare = requestSampling(&config, try rootOf(a, "{}"));
    try testing.expectEqual(@as(f32, 1.0), bare.temperature);
    try testing.expectEqual(@as(f32, 1.0), bare.top_p);
    try testing.expectEqual(@as(u32, 0), bare.top_k);
    try testing.expectEqual(@as(f32, 1.0), bare.repeat_penalty);
    try testing.expectEqual(@as(f32, 0.0), bare.presence_penalty);
    try testing.expect(bare.seed == null);

    config.gen_temperature = 0.6;
    config.gen_top_p = 0.95;
    config.gen_top_k = 20;
    const gen = requestSampling(&config, try rootOf(a, "{}"));
    try testing.expectEqual(@as(f32, 0.6), gen.temperature);
    try testing.expectEqual(@as(f32, 0.95), gen.top_p);
    try testing.expectEqual(@as(u32, 20), gen.top_k);

    const saved = server_mod.server_config;
    defer server_mod.server_config = saved;
    server_mod.server_config.default_temperature = 0.3;
    server_mod.server_config.default_top_k = 5;
    const flag = requestSampling(&config, try rootOf(a, "{}"));
    try testing.expectEqual(@as(f32, 0.3), flag.temperature);
    try testing.expectEqual(@as(u32, 5), flag.top_k);
    try testing.expectEqual(@as(f32, 0.95), flag.top_p);

    const body = requestSampling(&config, try rootOf(a, "{\"temperature\":0,\"top_k\":0,\"repetition_penalty\":1.1,\"presence_penalty\":0.5,\"seed\":7}"));
    try testing.expectEqual(@as(f32, 0.0), body.temperature);
    try testing.expectEqual(@as(u32, 0), body.top_k);
    try testing.expectEqual(@as(f32, 1.1), body.repeat_penalty);
    try testing.expectEqual(@as(f32, 0.5), body.presence_penalty);
    try testing.expectEqual(@as(?u64, 7), body.seed);
}

test "completions spec: a request's own switches outrank the model's defaults, and a head that is not loaded stays off" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var config = model_mod.ModelConfig{};
    config.model_type = "qwen4_exp";
    const loaded = requestSpec(&config, try rootOf(a, "{}"), true, true, false);
    try testing.expect(loaded.drafter and loaded.mtp and loaded.allow_batch_mtp);
    const off = requestSpec(&config, try rootOf(a, "{\"enable_drafter\":false,\"enable_mtp\":false,\"enable_batch_mtp\":false}"), true, true, false);
    try testing.expect(!off.drafter and !off.mtp and !off.allow_batch_mtp);
    const asked = requestSpec(&config, try rootOf(a, "{\"enable_drafter\":true,\"enable_mtp\":true}"), false, false, false);
    try testing.expect(!asked.drafter and !asked.mtp);
    const unserved = model_mod.ModelConfig{};
    const other = requestSpec(&unserved, try rootOf(a, "{}"), false, true, false);
    try testing.expect(other.mtp);
    var moe = model_mod.ModelConfig{};
    moe.model_type = "unserved_moe";
    moe.num_experts = 8;
    try testing.expect(!requestSpec(&moe, try rootOf(a, "{}"), false, true, false).mtp);
    try testing.expect(requestSpec(&moe, try rootOf(a, "{}"), false, true, true).mtp);
}
