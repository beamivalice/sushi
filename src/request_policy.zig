const std = @import("std");
const transformer_mod = @import("transformer.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const think_tags = @import("think_tags.zig");
const model_mod = @import("model.zig");
const effort_mod = @import("effort.zig");
const log = @import("log.zig");
const responses_mod = @import("responses.zig");
const model_registry_mod = @import("model_registry.zig");
const model_settings = @import("model_settings.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const LoadedModel = model_registry_mod.LoadedModel;
const sampler = @import("sampler.zig");
const think_penalty = @import("think_penalty.zig");
const testing = std.testing;
const server = @import("server.zig");
const http_conn = @import("http_conn.zig");
const generation_stream = @import("generation_stream.zig");
const Conn = http_conn.Conn;
const MAX_TOKENS_DESC_LEN = server.MAX_TOKENS_DESC_LEN;
const StreamMode = generation_stream.StreamMode;
const atomicTokenId = server.atomicTokenId;
const autoBudgetWindowTight = server.autoBudgetWindowTight;
const chatTemplateKwargBool = server.chatTemplateKwargBool;
const clampJsonI32 = server.clampJsonI32;
const jsonEscape = server.jsonEscape;
const parseJsonFloatOpt = server.parseJsonFloatOpt;
const parseOnOff = server.parseOnOff;
const parseRequestSeed = server.parseRequestSeed;
const pickStreamMode = generation_stream.pickStreamMode;
const prefillRequestTerms = server.prefillRequestTerms;
const promptOpensThink = server.promptOpensThink;
const qsaScoreFusedOffGuard = server.qsaScoreFusedOffGuard;
const qwen4ExpOomConfig = server.qwen4ExpOomConfig;
const renderConfigFor = server.renderConfigFor;
const requestSpecModes = generation_stream.requestSpecModes;
const sendAnthropicError = @import("anthropic.zig").sendAnthropicError;
const sendErrorResponse = server.sendErrorResponse;

/// Sampling-default resolution chain: request body > CLI launch flag >
/// model generation_config.json > hardcoded fallback. An explicit request
/// value of 0 (greedy / disabled) is a value, not an omission.
pub fn resolveSamplingDefault(comptime T: type, request: ?T, cli: ?T, gen_config: ?T, fallback: T) T {
    return request orelse cli orelse gen_config orelse fallback;
}

/// The in-stream thinking bound needs atomic think markers.
fn thinkMarkersAtomic(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    if (lm.transformer == null) return false;
    return atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER) != null;
}

/// The think penalty's lambda: the request's `think_penalty` > `--think-penalty` >
/// model-settings.json > off.
pub fn resolveThinkPenalty(root: std.json.ObjectMap, flag: ?f32, setting: ?f32) f32 {
    if (parseJsonFloatOpt(root, "think_penalty", 0, model_settings.think_penalty_max)) |v| return v;
    return model_settings.pick(f32, flag, setting, 0).value;
}

/// The penalty for one request: off unless it thinks, lambda > 0 and its span is trackable
/// (a single-token closer, and an opener when the prompt leaves the thought to the model).
fn thinkPenaltyFor(lambda: f32, enable_thinking: bool, opener: ?u32, closer: ?u32, prompt_opened: bool) think_penalty.ThinkPenalty {
    if (lambda <= 0 or !enable_thinking) return .{};
    const close = closer orelse return .{};
    if (opener == null and !prompt_opened) return .{};
    return .{ .lambda = lambda, .opener_id = opener, .closer_id = close, .phase = if (prompt_opened) .inside else .before };
}

pub fn armThinkPenalty(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer, prompt_ids: []const u32, enable_thinking: bool, lambda: f32) think_penalty.ThinkPenalty {
    if (!enable_thinking or lm.transformer == null) return .{};
    if (!@import("think_penalty.zig").needsSpan(lambda, lm.transformer.?.logit_bias)) return .{};
    const closer = atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER);
    var tp = thinkPenaltyFor(@max(lambda, 1), enable_thinking, atomicTokenId(allocator, tok, think_tags.BARE_THINK_OPENER), closer, promptOpensThink(allocator, tok, prompt_ids));
    tp.lambda = if (tp.lambda > 0) lambda else 0;
    if (tp.armed()) log.info("  think penalty {d}: markers lowered inside the reasoning span\n", .{lambda});
    return tp;
}

/// Default when an OpenAI-style client omits `max_tokens` /
/// `max_completion_tokens` / `max_output_tokens`. OpenAI semantics: omitted
/// means "generate until EOS, bounded by the context window" — NOT a small
/// fixed cap. (The old 256 default silently broke agent clients like pi:
/// every thinking-enabled turn hit `length` mid-reasoning, so no tool call or
/// answer ever came back.) The sentinel is huge so the downstream
/// `clampMaxTokens` resolves it to the remaining context; only when the context
/// is genuinely unknown (0) do we fall back to a finite 4096.
///
/// Takes the EFFECTIVE context, not `server_config.max_context_size`: that
/// global is set only by `--ctx-size`, so under auto-context this used to
/// return 4096 — an omitted-max_tokens client silently capped at 4096 tokens,
/// truncating any large tool call. Same class as the 256 default it replaced.
pub fn omittedMaxTokensDefault(config: *const model_mod.ModelConfig) u32 {
    return omittedMaxTokensDefaultWith(server.getEffectiveContextLength(config), launchMaxTokensDefault(), config.defaultMaxTokens());
}

/// Pure core of the above with the launch default (`--max-tokens N`, 0 = unset) passed in.
/// The flag outranks both fallbacks.
fn omittedMaxTokensDefaultWith(effective_ctx: u32, launch_default: u32, model_default: u32) u32 {
    if (launch_default > 0) return launch_default;
    if (model_default > 0) return model_default;
    return if (effective_ctx > 0) AUTO_MAX_TOKENS_SENTINEL else 4096;
}

/// The "no budget named, peg it to the window" sentinel, in one place; `clampMaxTokens` reduces it to `context - prompt`.
const AUTO_MAX_TOKENS_SENTINEL: u32 = std.math.maxInt(u32) / 4;

/// The budget a surface that needs one fills in when the request names none: `--max-tokens`, else the model's own; 0 = neither.
pub fn filledMaxTokens(config: *const model_mod.ModelConfig) u32 {
    const launch = launchMaxTokensDefault();
    return if (launch > 0) launch else config.defaultMaxTokens();
}

/// The one read of `ServerConfig.default_max_tokens`. 0 = flag not given.
pub fn launchMaxTokensDefault() u32 {
    return server.server_config.default_max_tokens;
}

/// Resolve a request's `max_tokens` (or its aliases) to an effective cap.
/// Absent OR `<= 0` means **auto**: peg generation to the remaining context
/// window via `auto_default` (the `omittedMaxTokensDefault` sentinel, which
/// `clampMaxTokens` then reduces to `context - prompt`). The app's "Auto"
/// setting sends 0 (or omits the field); both land on the context-pegged budget
/// instead of a fixed ceiling. A positive integer is an explicit cap.
pub fn resolveRequestMaxTokens(v: ?std.json.Value, auto_default: u32) u32 {
    const val = v orelse return auto_default;
    return switch (val) {
        .integer => |i| if (i > 0) @intCast(@min(i, std.math.maxInt(u32))) else auto_default,
        else => auto_default,
    };
}

/// Who decided the effective `max_tokens`: the client, the `--max-tokens` launch default, or nobody (auto).
const MaxTokensOrigin = enum { client, launch_default, model_default, auto };

/// The origin of the number `resolveRequestMaxTokens` returns, from the same two inputs.
pub fn maxTokensOrigin(requested: ?std.json.Value, launch_default: u32, model_default: u32) MaxTokensOrigin {
    if (requested) |val| switch (val) {
        .integer => |i| if (i > 0) return .client,
        else => {},
    };
    if (launch_default > 0) return .launch_default;
    return if (model_default > 0) .model_default else .auto;
}

/// Render the request log's `max_tokens=` field: auto prints as `auto` (or `auto (N)` for the
/// finite unknown-context fallback), a launch-filled budget names its source, a client's number
/// prints bare. The sentinel used to print as a client asking for a billion tokens.
pub fn describeMaxTokens(buf: []u8, value: u32, origin: MaxTokensOrigin) []const u8 {
    return switch (origin) {
        .client => std.fmt.bufPrint(buf, "{d}", .{value}) catch "?",
        .launch_default => std.fmt.bufPrint(buf, "{d} (launch default)", .{value}) catch "?",
        .model_default => std.fmt.bufPrint(buf, "{d} (model default)", .{value}) catch "?",
        .auto => if (value == AUTO_MAX_TOKENS_SENTINEL)
            "auto"
        else
            std.fmt.bufPrint(buf, "auto ({d})", .{value}) catch "auto",
    };
}

/// Does the generation budget warrant the "budget squeezed" warning? The auto sentinel is
/// excluded: it is always more than 4x anything that remains.
fn maxTokensBudgetSqueezed(max_tokens: u32, remaining: u32) bool {
    if (max_tokens == AUTO_MAX_TOKENS_SENTINEL) return false;
    return remaining < max_tokens / 4;
}

/// Clamp max_tokens so prompt + completion doesn't exceed context length.
/// Clamp a request's `max_tokens` so `prompt + generation` stays inside
/// `effective_ctx`. `effective_ctx == 0` means "unknown" and imposes no limit.
///
/// Takes the context explicitly rather than reading `server_config.max_context_size`:
/// that global is only set by `--ctx-size`, so under AUTO context this function
/// used to return `max_tokens` untouched — the server never trimmed a client's
/// budget, never emitted the "budget squeezed" warning, and generation could run
/// past the memory-safe window. Callers pass `getEffectiveContextLength(config)`,
/// which is the same number the prompt-length guard and `/v1/models` report.
pub fn clampMaxTokens(max_tokens: u32, prompt_len: usize, effective_ctx: u32) u32 {
    if (effective_ctx == 0) return max_tokens;
    const prompt: u32 = @intCast(@min(prompt_len, effective_ctx));
    if (prompt >= effective_ctx) return 1; // at least 1 token
    const remaining = effective_ctx - prompt;
    if (maxTokensBudgetSqueezed(max_tokens, remaining)) {
        log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, effective_ctx });
    } else if (max_tokens == AUTO_MAX_TOKENS_SENTINEL and autoBudgetWindowTight(remaining, effective_ctx)) {
        // The client named no budget, so this is context, not a deviation: debug.
        log.debug("  generation budget squeezed: {d} of the window remain (auto budget; prompt={d}, ctx={d})\n", .{ remaining, prompt, effective_ctx });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx={d}, prompt={d})\n", .{ max_tokens, remaining, effective_ctx, prompt });
        return remaining;
    }
    return max_tokens;
}

/// Heuristic: chat templates that contain a thinking-block opener indicate the
/// model can produce reasoning_content. Covers Qwen (`enable_thinking`,
/// `<think>`), Gemma 4 (`<|channel>thought`), and generic `<think>` templates.
pub fn chatTemplateSupportsThinking(tmpl: []const u8) bool {
    return std.mem.indexOf(u8, tmpl, "enable_thinking") != null or
        std.mem.indexOf(u8, tmpl, "<think>") != null or
        std.mem.indexOf(u8, tmpl, "thought") != null or
        std.mem.indexOf(u8, tmpl, "<|channel>") != null;
}

/// Render an optional model-author sampling recommendation (from the model's
/// generation_config.json) as a JSON scalar: the number when present, the
/// literal `null` when the model ships no value. Caller owns the slice.
/// Used for the `gen_temperature`/`gen_top_p`/`gen_top_k` meta fields the
/// Swift Settings UI reads to show "model recommends" guidance pills.
pub fn optSamplingRecJson(allocator: std.mem.Allocator, comptime T: type, v: ?T) ![]u8 {
    if (v) |val| return std.fmt.allocPrint(allocator, "{d}", .{val});
    return allocator.dupe(u8, "null");
}

/// `effort` is the client's raw string, borrowed from the parsed request JSON
/// (which outlives the handler), or on a served arch the selected arm's static
/// word — DeepSeek templates map it into the
/// render via `chat.dsv41EffortFor`; every other consumer only reads
/// enable/budget.
const ReasoningEffort = struct { enable: bool, budget: i32, effort: ?[]const u8 = null, rerouted_from: ?[]const u8 = null };

/// OpenAI-standard `reasoning_effort` on chat/completions (none | minimal |
/// low | medium | high | xhigh — values are model-dependent, so unknown
/// strings still enable). "none" is an explicit off (the gpt-5.1 default
/// spelling). Absent or non-string → null: the vendor `enable_thinking` bool
/// stays in charge and existing clients see zero behavior change.
/// The effort word is the ONLY lever when the template reads it and no
/// decode-time bound can enforce a budget derived from it (a budget that is
/// only a delivery cap fights the word: `chat.templateConsumesEffort`).
pub fn effortWordOnly(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    const cc = lm.chat_config orelse return false;
    return chat_mod.templateConsumesEffort(cc.chat_template) and !thinkMarkersAtomic(allocator, lm, tok);
}

pub fn parseReasoningEffort(root: std.json.ObjectMap, default_budget: i32, template_consumes_effort: bool, arms: ?[]const effort_mod.EffortArm) error{EffortRefused}!?ReasoningEffort {
    const v = root.get("reasoning_effort") orelse return null;
    if (v != .string) return null;
    return try reasoningEffortFromWord(v.string, default_budget, template_consumes_effort, arms);
}

/// One effort word → one thinking config, whatever field carried the word —
/// OpenAI's flat `reasoning_effort`, Responses' `reasoning.effort` and
/// Anthropic's `output_config.effort` must not drift on what "low" means.
/// `arms` is the model's `effort.effortArms` table, read through `effort.selectEffortPick`.
pub fn reasoningEffortFromWord(word: []const u8, default_budget: i32, template_consumes_effort: bool, arms: ?[]const effort_mod.EffortArm) error{EffortRefused}!ReasoningEffort {
    if (arms) |table| {
        var offered: [std.enums.values(effort_mod.Effort).len]effort_mod.Effort = undefined;
        for (table, 0..) |a, i| offered[i] = a.effort;
        const asked = effort_mod.parseEffort(word) orelse return error.EffortRefused;
        const pick = effort_mod.selectEffortPick(offered[0..table.len], asked) orelse return error.EffortRefused;
        const arm = effort_mod.findEffortArm(table, pick.effort) orelse return error.EffortRefused;
        const from: ?[]const u8 = if (pick.rerouted_from) |f| @tagName(f) else null;
        if (arm.effort == .off) return .{ .enable = false, .budget = default_budget, .effort = word, .rerouted_from = from };
        const budget = if (template_consumes_effort) default_budget else arm.budget orelse default_budget;
        return .{ .enable = true, .budget = budget, .effort = @tagName(arm.effort), .rerouted_from = from };
    }
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = default_budget, .effort = word };
    // Where the TEMPLATE reads the effort word, the word is the behavioral
    // lever and a budget derived from the same string is pure display
    // truncation on top of it (`chat.templateConsumesEffort`). An explicit
    // `--reasoning-budget` still rides `default_budget`.
    const budget = if (template_consumes_effort)
        default_budget
    else
        responses_mod.effortBudget(word, default_budget);
    return .{ .enable = true, .budget = budget, .effort = word };
}

pub const EffortNotice = struct {
    text: []u8,
    json: []u8,

    pub fn deinit(self: EffortNotice, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.json);
    }
};

pub fn effortModelName(lm: *LoadedModel) []const u8 {
    const config = lm.config.?;
    return if (lm.id.len > 0) lm.id else config.model_type;
}

/// One line for the log, the header and `effort_notice`. Caller frees.
fn makeEffortNotice(allocator: std.mem.Allocator, model_name: []const u8, from_word: []const u8, to_word: []const u8) !EffortNotice {
    const from = effort_mod.parseEffort(from_word) orelse return error.EffortRefused;
    const to = effort_mod.parseEffort(to_word) orelse return error.EffortRefused;
    const text = try effort_mod.formatEffortRerouteNotice(allocator, model_name, from, to);
    errdefer allocator.free(text);
    const esc = try jsonEscape(allocator, text);
    defer allocator.free(esc);
    const json_field = try std.fmt.allocPrint(allocator, ",\"effort_notice\":{s}", .{esc});
    log.info("[effort] {s}\n", .{text});
    return .{ .text = text, .json = json_field };
}

fn armEffortNotice(stream: *Conn, notice: EffortNotice) void {
    stream.effort_notice = notice.text;
    stream.effort_notice_json = notice.json;
}

pub fn clearEffortNotice(stream: *Conn) void {
    stream.effort_notice = null;
    stream.effort_notice_json = null;
}

pub fn holdEffortNotice(allocator: std.mem.Allocator, stream: *Conn, model_name: []const u8, cfg: ReasoningEffort) !?EffortNotice {
    const from = cfg.rerouted_from orelse return null;
    const notice = try makeEffortNotice(allocator, model_name, from, cfg.effort orelse return error.EffortRefused);
    armEffortNotice(stream, notice);
    return notice;
}

fn effortNoticeHeaderSafe(notice: []const u8) bool {
    for (notice) |c| if (c < 0x20 or c == 0x7f) return false;
    return true;
}

pub fn writeEffortNoticeHeader(stream: *Conn) !void {
    const notice = stream.effort_notice orelse return;
    if (!effortNoticeHeaderSafe(notice)) return;
    try stream.writeAllNoFlush("X-Sushi-Effort-Notice: ");
    try stream.writeAllNoFlush(notice);
    try stream.writeAllNoFlush("\r\nAccess-Control-Expose-Headers: X-Sushi-Effort-Notice\r\n");
}

pub fn withEffortNoticeJson(allocator: std.mem.Allocator, json: []const u8, field: ?[]const u8) ![]u8 {
    const extra = field orelse return allocator.dupe(u8, json);
    if (extra.len == 0 or json.len == 0 or json[json.len - 1] != '}') return allocator.dupe(u8, json);
    if (std.mem.indexOf(u8, json, "\"effort_notice\":") != null) return allocator.dupe(u8, json);
    return std.fmt.allocPrint(allocator, "{s}{s}}}", .{ json[0 .. json.len - 1], extra });
}

/// The 400 text for an effort word the model's table refuses; caller frees.
pub fn effortRefusal(allocator: std.mem.Allocator, word: []const u8, model_name: []const u8, arms: []const effort_mod.EffortArm) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "reasoning effort '{s}' is not supported by {s}; use one of: ", .{ word, model_name });
    for (arms, 0..) |a, i| try out.print(allocator, "{s}{s}", .{ if (i > 0) ", " else "", @tagName(a.effort) });
    return out.toOwnedSlice(allocator);
}

/// The `/v1/models` row fields listing the accepted effort words, the thinking-token budget of
/// each capped word, and the one a request naming none runs at; "" for an arch without a table.
pub fn reasoningEffortsJson(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig) ![]u8 {
    const table = effort_mod.effortArms(config.model_type) orelse return allocator.dupe(u8, "");
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, ",\"reasoning_efforts\":[");
    for (table, 0..) |a, i| try out.print(allocator, "{s}\"{s}\"", .{ if (i > 0) "," else "", @tagName(a.effort) });
    try out.appendSlice(allocator, "],\"reasoning_effort_budgets\":{");
    var first = true;
    for (table) |a| if (a.budget) |b| {
        try out.print(allocator, "{s}\"{s}\":{d}", .{ if (first) "" else ",", @tagName(a.effort), b });
        first = false;
    };
    try out.append(allocator, '}');
    if (model_mod.defaultReasoningEffort(config)) |w| try out.print(allocator, ",\"default_reasoning_effort\":\"{s}\"", .{w});
    return out.toOwnedSlice(allocator);
}

/// The refusal text for `word` on `lm`'s model; caller frees.
pub fn effortRefusalFor(allocator: std.mem.Allocator, lm: *LoadedModel, word: []const u8) ![]u8 {
    const config = lm.config.?;
    return effortRefusal(allocator, word, if (lm.id.len > 0) lm.id else config.model_type, effort_mod.effortArms(config.model_type).?);
}

/// The client's thinking bool: top-level `enable_thinking`, else `chat_template_kwargs.enable_thinking`.
fn requestEnableThinking(root: std.json.ObjectMap) ?bool {
    if (root.get("enable_thinking")) |v| if (v == .bool) return v.bool;
    return chatTemplateKwargBool(root, "enable_thinking");
}

/// Responses thinking: a `reasoning` object decides alone; without one the request
/// bool (`requestEnableThinking`), else the arch default, as on chat.
pub fn responsesEnableThinking(root: std.json.ObjectMap, from_reasoning: bool, arch_default: bool) bool {
    if (root.get("reasoning")) |r| if (r != .null) return from_reasoning;
    return requestEnableThinking(root) orelse arch_default;
}

/// Resolve thinking for a chat request. An EXPLICIT client signal always wins —
/// the vendor `enable_thinking` bool, or an OpenAI `reasoning_effort` string
/// ("none" being an explicit OFF). Only a request that names NEITHER falls
/// through to the arch default (`ModelConfig.defaultEnableThinking`).
///
/// The two knobs stay OR'd when both are present, as they always were.
pub fn checkThinkingSupport(allocator: std.mem.Allocator, stream: *Conn, config: *const model_mod.ModelConfig, enabled: bool, anthropic: bool) !bool {
    if (!config.isGlm5() or enabled) return true;
    const message = "GLM-5.3 supports thinking low, high, max; thinking off is not supported by its original template.";
    if (anthropic) try sendAnthropicError(allocator, stream, "invalid_request_error", message, 400) else try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", message, 400);
    return false;
}

pub fn resolveEnableThinking(root: std.json.ObjectMap, effort_cfg: ?ReasoningEffort, arch_default: bool) bool {
    const et = requestEnableThinking(root);
    if (et == null and effort_cfg == null) return arch_default;
    return (et orelse false) or (if (effort_cfg) |e| e.enable else false);
}

const SchemaThinkingPolicy = enum {
    no_mask,
    token_zero,
    deferred,
    fallback_thinking_off,
};

/// A schema grammar is safe with thinking only when generation can hand the
/// channel machinery to a resolved reasoning protocol (a prompt-opened think
/// block whose close the tokenizer can represent, or — for think-tag
/// templates — the unresolved opener choice). Every other masked surface
/// keeps the upstream thinking-off safety net. A finite response-side
/// reasoning budget also keeps that fallback because it can end the visible
/// think block before generation reaches the grammar boundary. Tools still
/// skip the mask.
/// Every JSON grammar mask site goes through here.
pub fn schemaMasksThinking(
    has_schema: bool,
    has_tools: bool,
    enable_thinking: bool,
    has_finite_reasoning_budget: bool,
    protocol_ready: bool,
) SchemaThinkingPolicy {
    if (!has_schema or has_tools) return .no_mask;
    if (!enable_thinking) return .token_zero;
    if (has_finite_reasoning_budget) return .fallback_thinking_off;
    if (protocol_ready) return .deferred;
    return .fallback_thinking_off;
}

test "requestSpecModes: a repeat, presence or frequency penalty keeps the sidecar off but MTP on" {
    const presence = sampler.SamplingParams{ .presence_penalty = 1.5 };
    const repeat = sampler.SamplingParams{ .repeat_penalty = 1.1 };
    const frequency = sampler.SamplingParams{ .frequency_penalty = 0.5 };
    for ([_]sampler.SamplingParams{ presence, repeat, frequency }) |sampling| {
        // A loaded sidecar cannot take the request, so the round falls to MTP.
        var r = requestSpecModes(true, true, true, true, sampling, 0);
        try std.testing.expect(r.use_mtp and !r.use_drafter);
        try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, true, true, sampling, 0));
        r = requestSpecModes(true, true, false, true, sampling, 0);
        try std.testing.expect(r.use_mtp and !r.use_drafter);
        // Without MTP the penalty leaves the sidecar nothing to run.
        r = requestSpecModes(true, false, true, false, sampling, 0);
        try std.testing.expect(!r.use_mtp and !r.use_drafter);
        // logprobs still beat it.
        r = requestSpecModes(true, true, false, true, sampling, 2);
        try std.testing.expect(!r.use_mtp and !r.use_drafter);
    }
}

/// The template's `preserve_thinking`: request > `--preserve-thinking` >
/// model-settings.json; null leaves the template's own default.
pub fn resolvePreserveThinking(request: ?bool, flag: ?bool, setting: ?bool) ?bool {
    return request orelse flag orelse setting;
}

/// The multiplicative `repeat_penalty` (llama.cpp's, at most 10), else vLLM's `repetition_penalty`;
/// 0 or absent is 1.0 (off). OpenAI's `frequency_penalty` is a separate additive penalty.
pub fn parseRepeatPenalty(root: std.json.ObjectMap) f32 {
    const rp = parseJsonFloat(root, "repeat_penalty", 0.0, 0.0, 10.0);
    if (rp > 0.0) return rp;
    const rep_p = parseJsonFloat(root, "repetition_penalty", 0.0, 0.0, 10.0);
    return if (rep_p > 0.0) rep_p else 1.0;
}

/// OpenAI's `presence_penalty` and `frequency_penalty`: -2 to 2, 0 = off.
pub fn parseAdditivePenalty(root: std.json.ObjectMap, key: []const u8) f32 {
    return parseJsonFloat(root, key, 0.0, -2.0, 2.0);
}

fn parseJsonFloat(root: std.json.ObjectMap, key: []const u8, default: f32, min: f32, max: f32) f32 {
    const raw = if (root.get(key)) |v| switch (v) {
        .float => |f| @as(f32, @floatCast(f)),
        .integer => |i| @as(f32, @floatFromInt(i)),
        else => default,
    } else default;
    return std.math.clamp(raw, min, max);
}

test "repeat_penalty: 0 or below is unset, then off, on chat and completions alike" {
    const a = std.testing.allocator;
    const Case = struct { body: []const u8, want: f32 };
    const cases = [_]Case{
        .{ .body = "{}", .want = 1.0 },
        .{ .body = "{\"repeat_penalty\":1.3}", .want = 1.3 },
        .{ .body = "{\"repeat_penalty\":0}", .want = 1.0 },
        .{ .body = "{\"repeat_penalty\":-2,\"frequency_penalty\":0.5}", .want = 1.0 },
        .{ .body = "{\"repeat_penalty\":\"x\",\"frequency_penalty\":0.1}", .want = 1.0 },
        .{ .body = "{\"frequency_penalty\":1}", .want = 1.0 },
        .{ .body = "{\"repeat_penalty\":50}", .want = 10.0 },
        .{ .body = "{\"repetition_penalty\":1.3}", .want = 1.3 },
        .{ .body = "{\"repetition_penalty\":0}", .want = 1.0 },
        .{ .body = "{\"repetition_penalty\":-2,\"frequency_penalty\":0.5}", .want = 1.0 },
        .{ .body = "{\"repetition_penalty\":50}", .want = 10.0 },
        .{ .body = "{\"repeat_penalty\":1.2,\"repetition_penalty\":1.5}", .want = 1.2 },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        try std.testing.expectApproxEqAbs(c.want, parseRepeatPenalty(parsed.value.object), 1e-6);
    }
}

test "OpenAI frequency and presence penalties are additive fields in -2..2, never a repeat_penalty multiplier" {
    const a = std.testing.allocator;
    const Case = struct { body: []const u8, frequency: f32, presence: f32 };
    const cases = [_]Case{
        .{ .body = "{}", .frequency = 0, .presence = 0 },
        .{ .body = "{\"frequency_penalty\":1,\"presence_penalty\":0}", .frequency = 1, .presence = 0 },
        .{ .body = "{\"frequency_penalty\":0.5,\"presence_penalty\":-0.25}", .frequency = 0.5, .presence = -0.25 },
        .{ .body = "{\"frequency_penalty\":-9,\"presence_penalty\":9}", .frequency = -2, .presence = 2 },
        .{ .body = "{\"frequency_penalty\":\"x\"}", .frequency = 0, .presence = 0 },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        try std.testing.expectEqual(c.frequency, parseAdditivePenalty(root, "frequency_penalty"));
        try std.testing.expectEqual(c.presence, parseAdditivePenalty(root, "presence_penalty"));
        try std.testing.expectEqual(@as(f32, 1.0), parseRepeatPenalty(root));
    }
}

test "parseJsonFloat returns value when present" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":0.7}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 0.7), result, 0.001);
}

test "parseJsonFloat returns default when missing" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "parseJsonFloat clamps to range" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":5.0}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 1.0, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 2.0), result, 0.001);
}

test "parseJsonFloat handles integer value" {
    const allocator = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{\"temp\":1}", .{});
    defer parsed.deinit();
    const result = parseJsonFloat(parsed.value.object, "temp", 0.5, 0.0, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, 0.001);
}

test "clampMaxTokens honors the effective context (auto-pinned, not just --ctx-size)" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;
    server.server_config.max_context_size = 0; // auto: the old code never clamped here

    // 4096-token window, 4000-token prompt -> only 96 tokens of generation left.
    try testing.expectEqual(@as(u32, 96), clampMaxTokens(8192, 4000, 4096));
    // Comfortable budget is passed through untouched.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 100, 65536));
    // Prompt at/over the window still yields at least one token.
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 4096, 4096));
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, 9000, 4096));
    // ctx 0 (unknown) = no limit.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, 4000, 0));
}

test "omittedMaxTokensDefault: context-bound when ctx is known, finite fallback otherwise" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;

    // OpenAI semantics: omitted max_tokens = generate until EOS bounded by the
    // context window. The old fixed 256 default broke agent clients (pi) whose
    // thinking-enabled turns hit `length` mid-reasoning on EVERY request.
    server.server_config.max_context_size = 32768;
    const sentinel = omittedMaxTokensDefault(&model_mod.ModelConfig{});
    // Big enough that clampMaxTokens (the downstream bound) always wins…
    try testing.expect(sentinel > 32768);
    // …and the composition resolves to exactly the remaining context.
    try testing.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(sentinel, 1500, 32768));

    // ctx unknown (max_context_size=0): clampMaxTokens won't bound anything,
    // so the default itself must stay finite.
    server.server_config.max_context_size = 0;
    try testing.expectEqual(@as(u32, 4096), omittedMaxTokensDefaultWith(0, 0, 0));
}

test "resolveRequestMaxTokens: absent / 0 / negative / non-int → auto; positive → value" {
    const auto: u32 = 12345;
    // Absent → auto (the omitted-default path the app's "Auto" setting rides).
    try testing.expectEqual(auto, resolveRequestMaxTokens(null, auto));
    // 0 and negatives are "auto" too — a client must not be able to request a
    // 0-token (generate-nothing) response by selecting Auto.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = 0 }, auto));
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .integer = -5 }, auto));
    // Wrong type (e.g. a string) → auto rather than crashing.
    try testing.expectEqual(auto, resolveRequestMaxTokens(.{ .string = "999" }, auto));
    // A real positive cap is honored verbatim.
    try testing.expectEqual(@as(u32, 4096), resolveRequestMaxTokens(.{ .integer = 4096 }, auto));
}

test "--max-tokens is the omitted-field default: unset is byte-identical, set wins" {
    const t = std.testing;
    // With no flag the omitted-field default is byte-for-byte what it was.
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefaultWith(32768, 0, 0));
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefaultWith(0, 0, 0));

    // Flag set: the launch value outranks both fallbacks.
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(786_432, 65536, 0));
    try t.expectEqual(@as(u32, 65536), omittedMaxTokensDefaultWith(0, 65536, 0));

    // The context clamp still sits on top.
    try t.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(omittedMaxTokensDefaultWith(32768, 65536, 0), 1500, 32768));

    // A request that sends max_tokens is untouched by the flag.
    const auto = omittedMaxTokensDefaultWith(786_432, 65536, 0);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, auto));
    try t.expectEqual(@as(u32, 200_000), resolveRequestMaxTokens(.{ .integer = 200_000 }, auto));
    try t.expectEqual(@as(u32, 65536), resolveRequestMaxTokens(.{ .integer = 0 }, auto));

    // The live wrapper reads the ServerConfig field through the one accessor.
    const slot = &@field(server.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    try t.expectEqual(@as(u32, 0), launchMaxTokensDefault());
    var cfg = model_mod.ModelConfig{ .pinned_context = 32768 };
    try t.expectEqual(@as(u32, std.math.maxInt(u32) / 4), omittedMaxTokensDefault(&cfg));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(&cfg));
}

test "an omitted max_tokens: --max-tokens, else the architecture's own default, else the window" {
    const t = std.testing;
    const sentinel = std.math.maxInt(u32) / 4;
    // DeepSeek with no flag: its own 256,000, in every arm that is not the flag's.
    try t.expectEqual(@as(u32, 256_000), omittedMaxTokensDefaultWith(1_000_000, 0, 256_000));
    try t.expectEqual(@as(u32, 256_000), omittedMaxTokensDefaultWith(0, 0, 256_000));
    // The flag outranks the architecture.
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefaultWith(1_000_000, 4096, 256_000));
    // An architecture with no default keeps the window.
    try t.expectEqual(@as(u32, sentinel), omittedMaxTokensDefaultWith(1_000_000, 0, 0));
    // The context clamp still sits on top, and a client's own number is untouched.
    try t.expectEqual(@as(u32, 100_000 - 1500), clampMaxTokens(256_000, 1500, 100_000));
    const dflt = omittedMaxTokensDefaultWith(1_000_000, 0, 256_000);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, dflt));
    try t.expectEqual(@as(u32, 256_000), resolveRequestMaxTokens(.{ .integer = 0 }, dflt));

    // The log names the origin, and a launch value still wins the label.
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    try t.expectEqual(MaxTokensOrigin.model_default, maxTokensOrigin(null, 0, 256_000));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 4096, 256_000));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0, 256_000));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(null, 0, 0));
    try t.expectEqualStrings("256000 (model default)", describeMaxTokens(&buf, 256_000, .model_default));

    // The live wrapper reads the flag and the model's own default.
    const slot = &@field(server.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    var dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .pinned_context = 1_000_000, .serve_defaults = @import("dsv41/dsv41_config.zig").Config.serveDefaults(model_mod.ServeDefaults) };
    var qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp", .pinned_context = 1_000_000 };
    try t.expectEqual(@as(u32, 256_000), omittedMaxTokensDefault(&dsv41));
    try t.expectEqual(@as(u32, sentinel), omittedMaxTokensDefault(&qwen));
    try t.expectEqual(@as(u32, 256_000), filledMaxTokens(&dsv41));
    try t.expectEqual(@as(u32, 0), filledMaxTokens(&qwen));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), omittedMaxTokensDefault(&dsv41));
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&dsv41));
}

test "describeMaxTokens: an omitted max_tokens logs as auto, never as the sentinel" {
    const t = std.testing;
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;

    try t.expectEqualStrings("auto", describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .auto));

    // With an unknown context the auto budget is a finite 4096 that really caps generation.
    try t.expectEqualStrings("auto (4096)", describeMaxTokens(&buf, 4096, .auto));

    // `--max-tokens N` filled an omitted field: the number is real, but it is
    // the operator's, not the client's, and the line says whose it is.
    try t.expectEqualStrings("65536 (launch default)", describeMaxTokens(&buf, 65536, .launch_default));

    // A client that sent the field gets its own number back, unadorned.
    try t.expectEqualStrings("4096", describeMaxTokens(&buf, 4096, .client));

    // Origin comes from the SAME inputs `resolveRequestMaxTokens` reads, so
    // the label can't disagree with the number beside it.
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(null, 0, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = 0 }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .integer = -5 }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.auto, maxTokensOrigin(.{ .string = "999" }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 65536, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(.{ .integer = 0 }, 65536, 0));
    // A sent field outranks the flag, so it is the CLIENT's number either way.
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 65536, 0));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0, 0));

    try t.expectEqualStrings(
        "1073741823 (launch default)",
        describeMaxTokens(&buf, AUTO_MAX_TOKENS_SENTINEL, .launch_default),
    );
}

test "the max_tokens squeeze warning is silent for the auto budget, loud for a real over-ask" {
    const t = std.testing;

    // A real over-ask: the client named a number and the window can't pay it.
    // That is the warning's whole point (the tool-call truncation nudge).
    try t.expect(maxTokensBudgetSqueezed(32768, 4000));
    try t.expect(!maxTokensBudgetSqueezed(32768, 30000));

    // The sentinel is always > 4x whatever remains; there is nothing to be short of.
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 4000));
    try t.expect(!maxTokensBudgetSqueezed(AUTO_MAX_TOKENS_SENTINEL, 1));

    try t.expect(autoBudgetWindowTight(4000, 32768));
    try t.expect(!autoBudgetWindowTight(30000, 32768));
    // A launch default is a real number and takes the real predicate.
    try t.expect(maxTokensBudgetSqueezed(65536, 4000));
}

test "an omitted max_tokens reserves the LAUNCH default, not the whole window" {
    const t = std.testing;
    // The admission reservation reads the resolved budget, so the flag reaches it.
    const KVCache = transformer_mod.KVCache;
    const ctx: u64 = 786_432;
    const seq: u64 = 400_000; // past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;

    // No flag: the sentinel becomes the whole remaining window; only the headroom cap keeps that off the bill.
    const no_flag = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 0, 0), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, @intCast(ctx - seq)), no_flag);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // Flag set below the headroom: the reservation is prompt + the flag.
    const flagged = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 2048, 0), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 2048), flagged);
    try t.expectEqual(seq + 2048 + chunk, KVCache.reservedTokens(seq, flagged, chunk, ctx));
    try t.expect(KVCache.reservedTokens(seq, flagged, chunk, ctx) < KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // A flag above the headroom bills only the headroom.
    const big = clampMaxTokens(omittedMaxTokensDefaultWith(@intCast(ctx), 65536, 0), seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 65536), big);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, KVCache.reservedTokens(seq, big, chunk, ctx));
}

test "clampMaxTokens no limit when ctx_size=0" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;
    server.server_config.max_context_size = 0;

    try testing.expectEqual(@as(u32, 1000), clampMaxTokens(1000, 500, 0));
}

test "clampMaxTokens clamps when would exceed" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;
    server.server_config.max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, 3000, 4096));
}

test "clampMaxTokens no clamp when fits" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;
    server.server_config.max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, 100, 4096));
}

test "clampMaxTokens at boundary" {
    const original = server.server_config.max_context_size;
    defer server.server_config.max_context_size = original;
    server.server_config.max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4096, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, 4095, 4096));
}

test "parseReasoningEffort: standard chat reasoning_effort opt-in maps to thinking + budget" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, expect: ?ReasoningEffort }{
        // Absent or wrong type → null: the vendor enable_thinking flag stays
        // in charge and nothing changes for existing clients.
        .{ .body = "{}", .expect = null },
        .{ .body = "{\"reasoning_effort\":42}", .expect = null },
        // "none" is an explicit OFF (gpt-5.1 default), never an enable.
        .{ .body = "{\"reasoning_effort\":\"none\"}", .expect = .{ .enable = false, .budget = -1 } },
        // Known efforts enable thinking with the shared Responses budget map.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .expect = .{ .enable = true, .budget = 1024 } },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .expect = .{ .enable = true, .budget = 2048 } },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .expect = .{ .enable = true, .budget = 8192 } },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .expect = .{ .enable = true, .budget = -1 } },
        // Unknown efforts (xhigh, future values) enable with the default
        // budget — spec values are model-dependent, never reject them.
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .expect = .{ .enable = true, .budget = -1 } },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = try parseReasoningEffort(parsed.value.object, -1, false, null);
        if (case.expect) |want| {
            try std.testing.expectEqual(want.enable, got.?.enable);
            try std.testing.expectEqual(want.budget, got.?.budget);
            // The raw client string always rides along (dsv4 templates map it
            // into the render); it is the request body's own bytes.
            const v = parsed.value.object.get("reasoning_effort").?;
            try std.testing.expectEqualStrings(v.string, got.?.effort.?);
        } else {
            try std.testing.expect(got == null);
        }
    }
}

test "parseReasoningEffort: a template that READS the effort word gets no budget from it unless a decode-time bound can enforce it" {
    // Two levers, one word. Where the template acts on `reasoning_effort` the
    // word already shortens the THOUGHT; a budget that is only a delivery cap
    // fights it: pi asking Qwen3.8 for `medium` got the model's unguided (long)
    // thinking AND a 2048-token cut, followed by 25.9k tokens of invisible
    // generation (live 2026-08-14). The third argument is `effortWordOnly`:
    // true when the template reads the word AND `armThinkBound` cannot close
    // the thought at the budget (non-atomic markers). With atomic markers the
    // budget is enforced in-stream and the word maps like everywhere else.
    // An EXPLICIT cap is someone asking on purpose and still applies — that is
    // `default_budget` here, which carries `--reasoning-budget`.
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, consumes: bool, default_budget: i32, want: i32 }{
        // Consuming template (qwen3.8 / deepseek / glm): no effort-derived cap.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = true, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = true, .default_budget = -1, .want = -1 },
        // ...but `--reasoning-budget` still binds there.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .default_budget = 4096, .want = 4096 },
        // Non-consuming template: the effort word reaches no template, so the budget is the ONLY
        // thing it does and every mapping stays exactly as it shipped.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .consumes = false, .default_budget = -1, .want = 1024 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = false, .default_budget = -1, .want = 2048 },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = false, .default_budget = -1, .want = 8192 },
        // high and xhigh are uncapped: pi's own ladder is 2048 / 8192 / 16384.
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = false, .default_budget = -1, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .consumes = false, .default_budget = -1, .want = -1 },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = (try parseReasoningEffort(parsed.value.object, case.default_budget, case.consumes, null)).?;
        try std.testing.expect(got.enable);
        try std.testing.expectEqual(case.want, got.budget);
        // The raw string rides along either way — the template still renders it.
        try std.testing.expectEqualStrings(parsed.value.object.get("reasoning_effort").?.string, got.effort.?);
    }
}

test "reasoningEffortFromWord: none disables, words budget exactly like the OpenAI surface" {
    // "none" is an explicit off with the word kept for the template.
    const off = try reasoningEffortFromWord("none", -1, false, null);
    try std.testing.expect(!off.enable);
    try std.testing.expectEqualStrings("none", off.effort.?);
    // A word maps through the ONE effortBudget table…
    const low = try reasoningEffortFromWord("low", -1, false, null);
    try std.testing.expect(low.enable);
    try std.testing.expectEqual(@as(i32, 2048), low.budget);
    // …unless the template consumes the word (qwen3.8 class): then the word is
    // the lever and the budget stays the launch default.
    const consumed = try reasoningEffortFromWord("low", -1, true, null);
    try std.testing.expect(consumed.enable);
    try std.testing.expectEqual(@as(i32, -1), consumed.budget);
}

test "reasoningEffortFromWord: a served arch refuses unknown words, and re-routes only the announced folds" {
    const qwen = effort_mod.effortArms("qwen4_exp").?;
    const mimo = effort_mod.effortArms("mimo_v2").?;
    const glm = effort_mod.effortArms("glm5_next").?;
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", -1, false, qwen));
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", -1, false, qwen));
    for ([_][]const u8{ "high", "max" }) |w| {
        const got = try reasoningEffortFromWord(w, -1, false, qwen);
        try std.testing.expect(got.enable);
        try std.testing.expectEqualStrings("xhigh", got.effort.?);
        try std.testing.expectEqualStrings(w, got.rerouted_from.?);
        try std.testing.expectEqual(@as(i32, -1), got.budget);
    }
    const consumed = try reasoningEffortFromWord("max", 4096, true, qwen);
    try std.testing.expectEqual(@as(i32, 4096), consumed.budget);
    try std.testing.expectEqualStrings("xhigh", consumed.effort.?);
    const glm_med = try reasoningEffortFromWord("medium", -1, false, glm);
    try std.testing.expect(glm_med.enable);
    try std.testing.expectEqualStrings("high", glm_med.effort.?);
    try std.testing.expectEqualStrings("medium", glm_med.rerouted_from.?);
    const glm_xh = try reasoningEffortFromWord("xhigh", -1, false, glm);
    try std.testing.expect(glm_xh.enable);
    try std.testing.expectEqualStrings("max", glm_xh.effort.?);
    try std.testing.expectEqualStrings("xhigh", glm_xh.rerouted_from.?);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("off", -1, false, glm));
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", -1, false, glm));
    const glm_high = try reasoningEffortFromWord("high", -1, false, glm);
    try std.testing.expect(glm_high.rerouted_from == null);
    try std.testing.expectEqualStrings("high", glm_high.effort.?);
    const mimo_high = try reasoningEffortFromWord("high", 4096, false, mimo);
    try std.testing.expectEqualStrings("on", mimo_high.effort.?);
    try std.testing.expect(mimo_high.rerouted_from == null);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", -1, false, mimo));
    for ([_][]const u8{ "off", "none" }) |w| {
        try std.testing.expect(!(try reasoningEffortFromWord(w, -1, false, qwen)).enable);
        try std.testing.expect(!(try reasoningEffortFromWord(w, -1, false, mimo)).enable);
    }
    const q_low = try reasoningEffortFromWord("low", -1, false, qwen);
    try std.testing.expect(q_low.enable);
    try std.testing.expectEqual(@as(i32, 2048), q_low.budget);
    try std.testing.expectEqualStrings("low", q_low.effort.?);
    try std.testing.expectEqual(@as(i32, -1), (try reasoningEffortFromWord("low", -1, true, qwen)).budget);
    try std.testing.expectEqualStrings("xhigh", (try reasoningEffortFromWord("xhigh", -1, false, qwen)).effort.?);
    // An uncapped arm takes `--reasoning-budget`.
    try std.testing.expectEqual(@as(i32, 4096), (try reasoningEffortFromWord("on", 4096, false, mimo)).budget);
    try std.testing.expectEqual(@as(i32, 4096), (try reasoningEffortFromWord("medium", 4096, false, mimo)).budget);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", -1, false, qwen));
    try std.testing.expect((try reasoningEffortFromWord("ultra", -1, false, null)).enable);
}

test "withEffortNoticeJson appends the notice and leaves the answer bytes" {
    const a = std.testing.allocator;
    const body = "{\"content\":\"page summary\"}";
    const field = ",\"effort_notice\":\"Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`\"";
    const out = try withEffortNoticeJson(a, body, field);
    defer a.free(out);
    try std.testing.expectEqualStrings("{\"content\":\"page summary\",\"effort_notice\":\"Qwen3.8-Flash-Next has no `high` effort; served at `xhigh`\"}", out);
    const same = try withEffortNoticeJson(a, body, null);
    defer a.free(same);
    try std.testing.expectEqualStrings(body, same);
    const twice = try withEffortNoticeJson(a, out, field);
    defer a.free(twice);
    try std.testing.expectEqualStrings(out, twice);
}

test "effort refusal and /v1/models list name the model's accepted words" {
    const allocator = std.testing.allocator;
    const qwen = effort_mod.effortArms("qwen4_exp").?;
    const msg = try effortRefusal(allocator, "minimal", "Qwen3.8-Flash-Next-EXL3-K4", qwen);
    defer allocator.free(msg);
    try std.testing.expectEqualStrings("reasoning effort 'minimal' is not supported by Qwen3.8-Flash-Next-EXL3-K4; use one of: off, low, medium, xhigh", msg);
    const row = try reasoningEffortsJson(allocator, &.{ .model_type = "qwen4_exp" });
    defer allocator.free(row);
    try std.testing.expectEqualStrings(",\"reasoning_efforts\":[\"off\",\"low\",\"medium\",\"xhigh\"],\"reasoning_effort_budgets\":{\"low\":2048,\"medium\":8192},\"default_reasoning_effort\":\"medium\"", row);
    const legacy = try reasoningEffortsJson(allocator, &.{ .model_type = "not_a_served_arch" });
    defer allocator.free(legacy);
    try std.testing.expectEqualStrings("", legacy);
}

test "resolveEnableThinking: an explicit request value outranks the arch default, silence takes it" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, arch: bool, want: bool }{
        // Silent request → the arch default, either way.
        .{ .body = "{}", .arch = false, .want = false },
        .{ .body = "{}", .arch = true, .want = true },
        // An explicit false must survive a thinking-on arch — silently
        // ignoring a field the client set is the class this exists to avoid.
        .{ .body = "{\"enable_thinking\":false}", .arch = true, .want = false },
        .{ .body = "{\"reasoning_effort\":\"none\"}", .arch = true, .want = false },
        // An explicit enable works on an arch that defaults off.
        .{ .body = "{\"enable_thinking\":true}", .arch = false, .want = true },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .arch = false, .want = true },
        // Both present stay OR'd, as they always were.
        .{ .body = "{\"enable_thinking\":false,\"reasoning_effort\":\"high\"}", .arch = false, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"reasoning_effort\":\"none\"}", .arch = false, .want = true },
        // A non-bool `enable_thinking` is not a signal; with nothing else in
        // the body the arch default still applies.
        .{ .body = "{\"enable_thinking\":\"yes\"}", .arch = true, .want = true },
        // vLLM / SGLang clients send the template switch nested; the top-level field wins over it.
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .arch = true, .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":true}}", .arch = false, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}", .arch = false, .want = true },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":\"no\"}}", .arch = true, .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const effort = try parseReasoningEffort(parsed.value.object, -1, false, null);
        try std.testing.expectEqual(case.want, resolveEnableThinking(parsed.value.object, effort, case.arch));
    }
}

test "responsesEnableThinking: no reasoning object takes the request bool, else the arch default" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, arch: bool, want: bool }{
        .{ .body = "{}", .arch = true, .want = true },
        .{ .body = "{}", .arch = false, .want = false },
        .{ .body = "{\"reasoning\":null}", .arch = true, .want = true },
        .{ .body = "{\"enable_thinking\":false}", .arch = true, .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .arch = true, .want = false },
        // A reasoning object decides alone.
        .{ .body = "{\"reasoning\":{\"effort\":\"none\"}}", .arch = true, .want = false },
        .{ .body = "{\"reasoning\":{}}", .arch = false, .want = true },
        .{ .body = "{\"reasoning\":{\"effort\":\"none\"},\"enable_thinking\":true}", .arch = false, .want = false },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const root = parsed.value.object;
        const from_reasoning = responses_mod.parseReasoning(root.get("reasoning"), -1).enable;
        try std.testing.expectEqual(case.want, responsesEnableThinking(root, from_reasoning, case.arch));
    }
}

test "schema thinking policy defers only a resolved reasoning protocol" {
    const decide = schemaMasksThinking;
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(false, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.no_mask, decide(true, true, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.token_zero, decide(true, false, false, true, false));
    try std.testing.expectEqual(SchemaThinkingPolicy.deferred, decide(true, false, true, false, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, true, true));
    try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, false));

    // A think tail classifies as resolvable (the resolver owns the
    // capability checks); channel-family and plain ChatML tails do not, and
    // keep the fallback — capability is never the tail's alone to decide.
    try std.testing.expect(think_tags.promptThinkTailClass("<|im_start|>assistant\n<think>") == .bare);
    const unsupported_tails = [_][]const u8{
        "<|channel|>analysis<|message|>",
        "<|im_start|>assistant\n",
    };
    for (unsupported_tails) |tail| {
        try std.testing.expectEqual(SchemaThinkingPolicy.fallback_thinking_off, decide(true, false, true, false, think_tags.promptThinkTailClass(tail) != .none));
    }
}

test "resolveSamplingDefault: request > CLI > generation_config > fallback" {
    // Request value always wins.
    try std.testing.expectEqual(@as(f32, 0.2), resolveSamplingDefault(f32, 0.2, 0.7, 1.0, 1.0));
    // Omitted in request -> CLI launch flag (the app passes Settings here).
    try std.testing.expectEqual(@as(f32, 0.7), resolveSamplingDefault(f32, null, 0.7, 1.0, 1.0));
    // No CLI flag -> the model's generation_config.json recommendation.
    try std.testing.expectEqual(@as(u32, 20), resolveSamplingDefault(u32, null, null, 20, 0));
    // Nothing anywhere -> hardcoded fallback (pre-existing behavior).
    try std.testing.expectEqual(@as(f32, 1.0), resolveSamplingDefault(f32, null, null, null, 1.0));
    // Explicit request 0 (greedy) must not be treated as omitted.
    try std.testing.expectEqual(@as(f32, 0.0), resolveSamplingDefault(f32, 0.0, 0.7, 1.0, 1.0));
}

test "optSamplingRecJson emits number when present, null when absent" {
    const a = std.testing.allocator;

    // Present -> bare JSON number (no quotes), so the Swift side decodes it
    // straight to Double/Int.
    const top_k = try optSamplingRecJson(a, u32, 20);
    defer a.free(top_k);
    try std.testing.expectEqualStrings("20", top_k);

    const top_p = try optSamplingRecJson(a, f32, 0.95);
    defer a.free(top_p);
    try std.testing.expectEqualStrings("0.95", top_p);

    // Absent -> JSON null literal so the model-author recommendation reads as
    // "no opinion" rather than a spurious 0.
    const none = try optSamplingRecJson(a, u32, null);
    defer a.free(none);
    try std.testing.expectEqualStrings("null", none);
}

test "the reservation is bounded by the CONTEXT: an omitted max_tokens cannot bill 26 TB" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // An omitted `max_tokens` is maxInt(u32)/4; the memory guard used to run before
    // `clampMaxTokens`, so the reservation billed ~26 TB of KV and refused every long prompt.
    const KVCache = transformer_mod.KVCache;
    const ctx: u64 = 262_144;
    const seq: u64 = 50_000;
    const sentinel: u64 = omittedMaxTokensDefaultWith(@intCast(ctx), 0, 0);
    try t.expectEqual(@as(u64, std.math.maxInt(u32) / 4), sentinel);

    // (1) The cache clamps: the reservation cannot exceed the context plus a chunk.
    const reserved = KVCache.reservedTokens(seq, sentinel, 4096, ctx);
    // Bounded twice: by the generation headroom first, by the context second.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, reserved);
    try t.expect(reserved < ctx);
    // (2) The four surfaces clamp first and hand the guard the clamped value.
    const clamped: u64 = clampMaxTokens(@intCast(sentinel), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped);
    try t.expectEqual(reserved, KVCache.reservedTokens(seq, clamped, 4096, ctx));

    try t.expectEqual(seq + 2048 + 4096, KVCache.reservedTokens(seq, 2048, 4096, ctx));
    // A caller that names no context is not context-clamped, but the headroom bound still applies.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, KVCache.reservedTokens(seq, sentinel, 4096, 0));
    try t.expectEqual(ctx + 4096, KVCache.reservedTokens(ctx, sentinel, 4096, ctx));

    const cfg = qwen4ExpOomConfig();
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, 4096, .{});
    try t.expect(terms.reserved_kv_bytes < 4 * 1024 * 1024 * 1024);
}

test "a request integer from a client is clamped, never blindly @intCast" {
    // Bar: `seed: -1`, a 2^32 max_tokens and an out-of-range budget resolve without a cast.
    const t = std.testing;
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .integer = -1 }));
    try t.expectEqual(@as(?u64, 0), parseRequestSeed(.{ .integer = 0 }));
    try t.expectEqual(@as(?u64, 42), parseRequestSeed(.{ .integer = 42 }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(.{ .string = "42" }));
    try t.expectEqual(@as(?u64, null), parseRequestSeed(null));

    try t.expectEqual(@as(u32, std.math.maxInt(u32)), resolveRequestMaxTokens(.{ .integer = 1 << 32 }, 7));
    try t.expectEqual(@as(u32, 512), resolveRequestMaxTokens(.{ .integer = 512 }, 7));
    try t.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .integer = 0 }, 7));

    try t.expectEqual(@as(i32, std.math.maxInt(i32)), clampJsonI32(1 << 40));
    try t.expectEqual(@as(i32, std.math.minInt(i32)), clampJsonI32(-(1 << 40)));
    try t.expectEqual(@as(i32, -1), clampJsonI32(-1));
    try t.expectEqual(@as(i32, 1024), clampJsonI32(1024));
}

test "preserve_thinking: request > --preserve-thinking > model-settings.json > the template's default" {
    const t = std.testing;
    try t.expectEqual(@as(?bool, false), resolvePreserveThinking(false, true, true));
    try t.expectEqual(@as(?bool, true), resolvePreserveThinking(true, false, false));
    try t.expectEqual(@as(?bool, true), resolvePreserveThinking(null, true, false));
    try t.expectEqual(@as(?bool, false), resolvePreserveThinking(null, null, false));
    try t.expectEqual(@as(?bool, null), resolvePreserveThinking(null, null, null));
}

test "think penalty: request > --think-penalty > model-settings.json > off, clamped" {
    const a = std.testing.allocator;
    const Case = struct { body: []const u8, flag: ?f32, setting: ?f32, want: f32 };
    const cases = [_]Case{
        .{ .body = "{\"think_penalty\":0}", .flag = 2, .setting = 3, .want = 0 },
        .{ .body = "{\"think_penalty\":1.5}", .flag = null, .setting = null, .want = 1.5 },
        .{ .body = "{\"think_penalty\":-4}", .flag = 2, .setting = null, .want = 0 },
        .{ .body = "{\"think_penalty\":1e9}", .flag = null, .setting = null, .want = model_settings.think_penalty_max },
        .{ .body = "{\"think_penalty\":\"2\"}", .flag = 2, .setting = 3, .want = 2 },
        .{ .body = "{}", .flag = null, .setting = 3, .want = 3 },
        .{ .body = "{}", .flag = null, .setting = null, .want = 0 },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(c.want, resolveThinkPenalty(parsed.value.object, c.flag, c.setting));
    }
}

test "think penalty: arms a thinking request whose think markers are single tokens" {
    const t = std.testing;
    const open: u32 = 7;
    const close: u32 = 8;
    const prompt_opened = thinkPenaltyFor(1.5, true, open, close, true);
    try t.expectEqual(@as(f32, 1.5), prompt_opened.lambda);
    try t.expectEqual(think_penalty.ThinkPenalty.Phase.inside, prompt_opened.phase);
    try t.expectEqual(close, prompt_opened.closer_id);
    try t.expectEqual(think_penalty.ThinkPenalty.Phase.before, thinkPenaltyFor(1.5, true, open, close, false).phase);
    try t.expect(!thinkPenaltyFor(0, true, open, close, true).armed());
    try t.expect(!thinkPenaltyFor(1.5, false, open, close, true).armed());
    try t.expect(!thinkPenaltyFor(1.5, true, open, null, true).armed());
    try t.expect(!thinkPenaltyFor(1.5, true, null, close, false).armed());
    try t.expect(thinkPenaltyFor(1.5, true, null, close, true).armed());
}

test "--preserve-thinking takes on or off" {
    try std.testing.expectEqual(@as(?bool, true), parseOnOff("on"));
    try std.testing.expectEqual(@as(?bool, false), parseOnOff("off"));
    try std.testing.expectEqual(@as(?bool, null), parseOnOff("yes"));
}

test "a request renders with its own preserve_thinking and leaves the model's config alone" {
    const a = std.testing.allocator;
    var model_config = chat_mod.ChatConfig{ .chat_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"chat_template_kwargs\":{\"preserve_thinking\":false}}", .{});
    defer parsed.deinit();
    const per_request = renderConfigFor(&model_config, parsed.value.object, null, true);
    try std.testing.expectEqual(@as(?bool, false), per_request.preserve_thinking);
    try std.testing.expectEqual(@as(?bool, null), model_config.preserve_thinking);
    const empty = try std.json.parseFromSlice(std.json.Value, a, "{}", .{});
    defer empty.deinit();
    try std.testing.expectEqual(@as(?bool, true), renderConfigFor(&model_config, empty.value.object, null, true).preserve_thinking);
    try std.testing.expectEqual(@as(?bool, false), renderConfigFor(&model_config, empty.value.object, false, true).preserve_thinking);
}

test "format corpus: one-shot thinking reaches served templates and effort budgets" {
    const a = std.testing.allocator;
    const cli = @import("cli.zig");
    const families = .{
        .{ "qwen4_exp", @embedFile("fixtures/qwen38_flash_next_chat_template.jinja") },
        .{ "mimo_v2", @embedFile("fixtures/mimo_v26_chat_template.jinja") },
    };
    inline for (families) |family| {
        var choices = std.ArrayList(cli.Think).empty;
        defer choices.deinit(a);
        try choices.appendSlice(a, &.{ .model_default, .on });
        for (effort_mod.effortArms(family[0]).?) |arm| try choices.append(a, .{ .effort = arm.effort });
        for (choices.items) |think| {
            const body = try cli.buildPromptBody(a, "  literal prompt\n", think, .{});
            defer a.free(body);
            const parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
            defer parsed.deinit();
            const root = parsed.value.object;
            const effort = try parseReasoningEffort(root, 77, false, effort_mod.effortArms(family[0]));
            const enable = resolveEnableThinking(root, effort, true);
            try std.testing.expectEqual(think != .effort or think.effort != .off, enable);
            if (think == .effort) {
                try std.testing.expectEqual(@as(i32, switch (think.effort) {
                    .low => 2048,
                    .medium => 8192,
                    else => 77,
                }), effort.?.budget);
            }
            var cc: chat_mod.ChatConfig = .{ .chat_template = family[1], .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
            const rendered = try chat_mod.renderChatTemplate(a, &.{.{ .role = "user", .content = root.get("messages").?.array.items[0].object.get("content").?.string }}, &cc, null, null, enable, if (effort) |e| e.effort else null, false);
            defer a.free(rendered);
            try std.testing.expect(std.mem.indexOf(u8, rendered, "literal prompt") != null);
            if (std.mem.eql(u8, family[0], "qwen4_exp")) {
                try std.testing.expectEqual(enable, think_tags.promptTailOpensThink(rendered));
            } else {
                try std.testing.expect(std.mem.endsWith(u8, rendered, if (enable) "<|im_start|>assistant\n" else "<think></think>"));
            }
            if (std.mem.eql(u8, family[0], "qwen4_exp") and think == .effort and think.effort == .xhigh)
                try std.testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to xhigh.") != null);
        }
    }
}

test "thinking policy HTTP: Qwen and GLM re-route the missing graded words, MiMo folds to on" {
    for ([_]struct { arch: []const u8, accepted: []const effort_mod.Effort }{
        .{ .arch = "glm5_next", .accepted = &.{ .low, .high, .max } },
        .{ .arch = "qwen4_exp", .accepted = &.{ .off, .low, .medium, .xhigh } },
    }) |c| {
        for (std.enums.values(effort_mod.Effort)) |e| {
            const result = reasoningEffortFromWord(@tagName(e), -1, true, effort_mod.effortArms(c.arch));
            if (std.mem.indexOfScalar(effort_mod.Effort, c.accepted, e) != null) {
                const got = try result;
                try std.testing.expectEqual(e != .off, got.enable);
                try std.testing.expect(got.rerouted_from == null);
            } else if (std.mem.eql(u8, c.arch, "qwen4_exp") and (e == .high or e == .max)) {
                const got = try result;
                try std.testing.expect(got.enable);
                try std.testing.expectEqualStrings("xhigh", got.effort.?);
                try std.testing.expectEqualStrings(@tagName(e), got.rerouted_from.?);
            } else if (std.mem.eql(u8, c.arch, "glm5_next") and e == .medium) {
                const got = try result;
                try std.testing.expect(got.enable);
                try std.testing.expectEqualStrings("high", got.effort.?);
                try std.testing.expectEqualStrings("medium", got.rerouted_from.?);
            } else if (std.mem.eql(u8, c.arch, "glm5_next") and e == .xhigh) {
                const got = try result;
                try std.testing.expect(got.enable);
                try std.testing.expectEqualStrings("max", got.effort.?);
                try std.testing.expectEqualStrings("xhigh", got.rerouted_from.?);
            } else try std.testing.expectError(error.EffortRefused, result);
        }
        try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", -1, true, effort_mod.effortArms(c.arch)));
    }
    const mimo = effort_mod.effortArms("mimo_v2");
    for ([_][]const u8{ "on", "minimal", "low", "medium", "high", "xhigh", "max" }) |w| {
        const got = try reasoningEffortFromWord(w, -1, true, mimo);
        try std.testing.expect(got.enable);
        try std.testing.expectEqualStrings("on", got.effort.?);
        try std.testing.expect(got.rerouted_from == null);
    }
    for ([_][]const u8{ "off", "none" }) |w| try std.testing.expect(!(try reasoningEffortFromWord(w, -1, true, mimo)).enable);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", -1, true, mimo));
}
