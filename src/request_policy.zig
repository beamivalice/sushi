const std = @import("std");
const test_note = @import("test_note.zig");
const arch_chat = @import("arch/chat.zig");
const arch_handle = @import("arch/handle.zig");
const settings_resolve = @import("settings_resolve.zig");
const request_parse = @import("request_parse.zig");
const request_fields = @import("request_fields.zig");
const http = @import("http.zig");
const trunk = @import("trunk.zig");
const tokenizer_mod = @import("tokenizer.zig");
const chat_mod = @import("chat.zig");
const think_tags = @import("think_tags.zig");
const model_mod = @import("model.zig");
const arch_config = @import("arch/config.zig");
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
const server_state = @import("server_state.zig");
const http_conn = @import("http_conn.zig");
const generation_stream = @import("generation_stream.zig");
const Conn = http_conn.Conn;
const MAX_TOKENS_DESC_LEN = request_parse.MAX_TOKENS_DESC_LEN;
const StreamMode = generation_stream.StreamMode;
const atomicTokenId = server.atomicTokenId;
const jsonEscape = http.jsonEscape;
const parseOnOff = server.parseOnOff;
const pickStreamMode = generation_stream.pickStreamMode;
const promptOpensThink = server.promptOpensThink;
const renderConfigFor = server.renderConfigFor;
const requestSpecModes = generation_stream.requestSpecModes;
const sendAnthropicError = @import("anthropic.zig").sendAnthropicError;
const sendErrorResponse = http.sendErrorResponse;

/// Sampling-default resolution chain: request body > CLI launch flag >
/// model generation_config.json > hardcoded fallback. An explicit request
/// value of 0 (greedy / disabled) is a value, not an omission.
pub fn resolveSamplingDefault(comptime T: type, request: ?T, cli: ?T, gen_config: ?T, fallback: T) T {
    return request orelse cli orelse gen_config orelse fallback;
}

/// The sampler values a request that omits them gets on this model; `request` is the body's own value.
pub fn samplingTemperature(config: *const model_mod.ModelConfig, request: ?f32) f32 {
    return resolveSamplingDefault(f32, request, server_state.server_config.default_temperature, config.gen_temperature, 1.0);
}

pub fn samplingTopP(config: *const model_mod.ModelConfig, request: ?f32) f32 {
    return resolveSamplingDefault(f32, request, server_state.server_config.default_top_p, config.gen_top_p, 1.0);
}

pub fn samplingTopK(config: *const model_mod.ModelConfig, request: ?u32) u32 {
    return resolveSamplingDefault(u32, request, server_state.server_config.default_top_k, config.gen_top_k, 0);
}

/// The in-stream thinking bound needs atomic think markers.
fn thinkMarkersAtomic(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    if (lm.handle == null) return false;
    return atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER) != null;
}

/// The think penalty's lambda: the request's `think_penalty` > `--think-penalty` >
/// model-settings.json > off.
pub fn resolveThinkPenalty(request: ?f32, flag: ?f32, setting: ?f32) f32 {
    if (request) |v| return v;
    return settings_resolve.resolve(f32, flag, setting, model_settings.default_think_penalty).value;
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
    if (!enable_thinking or lm.handle == null) return .{};
    if (!@import("think_penalty.zig").needsSpan(lambda, arch_handle.trunk(lm.handle.?).logit_bias)) return .{};
    const closer = atomicTokenId(allocator, tok, think_tags.BARE_THINK_CLOSER);
    var tp = thinkPenaltyFor(@max(lambda, 1), enable_thinking, atomicTokenId(allocator, tok, think_tags.BARE_THINK_OPENER), closer, promptOpensThink(allocator, tok, prompt_ids));
    tp.lambda = if (tp.lambda > 0) lambda else 0;
    if (tp.armed()) log.info("  think penalty {d}: markers lowered inside the reasoning span\n", .{lambda});
    return tp;
}

/// The budget of a request that omits `max_tokens`, never 0 and never pegged to the window:
/// `--max-tokens`, else the model's default, else `ENGINE_DEFAULT_MAX_TOKENS`.
pub fn filledMaxTokens(config: *const model_mod.ModelConfig) u32 {
    return filledMaxTokensWith(launchMaxTokensDefault(), config.defaultMaxTokens());
}

/// Pure core of `filledMaxTokens`; 0 = unset for both inputs.
pub fn filledMaxTokensWith(launch_default: u32, model_default: u32) u32 {
    if (launch_default > 0) return launch_default;
    if (model_default > 0) return model_default;
    return ENGINE_DEFAULT_MAX_TOKENS;
}

/// What an omitted `max_tokens` gets when neither the operator nor the model names a budget.
pub const ENGINE_DEFAULT_MAX_TOKENS: u32 = 32768;

/// The `model` an answer echoes when the request names none: the id it is served under.
pub fn servedModelId(lm: *const LoadedModel, config: *const model_mod.ModelConfig) []const u8 {
    return if (lm.id.len > 0) lm.id else config.model_type;
}

/// The one read of `ServerConfig.default_max_tokens`. 0 = flag not given.
pub fn launchMaxTokensDefault() u32 {
    return server_state.server_config.default_max_tokens;
}

/// Resolve a request's `max_tokens` (or its aliases) to an effective cap. Absent, `<= 0` or not
/// a whole number takes `filled` (`filledMaxTokens`): the app's "Auto" sends 0. A positive
/// integer is the client's own cap.
pub fn resolveRequestMaxTokens(v: ?std.json.Value, filled: u32) u32 {
    const i = request_parse.wholeNumber(v orelse return filled) orelse return filled;
    return if (i > 0) std.math.cast(u32, i) orelse filled else filled;
}

/// Who decided the effective `max_tokens`: the client, `--max-tokens`, the model, or the engine.
const MaxTokensOrigin = enum { client, launch_default, model_default, engine_default };

/// The origin of the number `resolveRequestMaxTokens` returns, from the same inputs.
pub fn maxTokensOrigin(requested: ?std.json.Value, launch_default: u32, model_default: u32) MaxTokensOrigin {
    if (requested) |val| if (request_parse.wholeNumber(val)) |i| {
        if (i > 0) return .client;
    };
    if (launch_default > 0) return .launch_default;
    return if (model_default > 0) .model_default else .engine_default;
}

/// Render the request log's `max_tokens=` field: a filled budget names its source, a client's number prints bare.
pub fn describeMaxTokens(buf: []u8, value: u32, origin: MaxTokensOrigin) []const u8 {
    return switch (origin) {
        .client => std.fmt.bufPrint(buf, "{d}", .{value}) catch "?",
        .launch_default => std.fmt.bufPrint(buf, "{d} (launch default)", .{value}) catch "?",
        .model_default => std.fmt.bufPrint(buf, "{d} (model default)", .{value}) catch "?",
        .engine_default => std.fmt.bufPrint(buf, "{d} (engine default)", .{value}) catch "?",
    };
}

/// Does the generation budget warrant the "budget squeezed" note?
fn maxTokensBudgetSqueezed(max_tokens: u32, remaining: u32) bool {
    return remaining < max_tokens / 4;
}

/// Clamp a request's `max_tokens` so `prompt + generation` stays inside `effective_ctx`, which
/// is always concrete (`getEffectiveContextLength` never returns 0). A squeeze on the client's
/// own number warns; on a budget the server filled it is context, not a deviation: debug.
pub fn clampMaxTokens(max_tokens: u32, client_asked: bool, prompt_len: usize, effective_ctx: u32) u32 {
    const prompt: u32 = @intCast(@min(prompt_len, effective_ctx));
    if (prompt >= effective_ctx) return 1; // at least 1 token
    const remaining = effective_ctx - prompt;
    if (maxTokensBudgetSqueezed(max_tokens, remaining)) {
        if (client_asked)
            log.warn("  generation budget squeezed: {d}/{d} tokens remaining (prompt={d}, ctx={d}) — tool call arguments may be truncated\n", .{ remaining, max_tokens, prompt, effective_ctx })
        else
            log.debug("  generation budget squeezed: {d}/{d} tokens remaining (filled budget; prompt={d}, ctx={d})\n", .{ remaining, max_tokens, prompt, effective_ctx });
    }
    if (max_tokens > remaining) {
        log.debug("  max_tokens clamped: {d} -> {d} (ctx={d}, prompt={d})\n", .{ max_tokens, remaining, effective_ctx, prompt });
        return remaining;
    }
    return max_tokens;
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
/// word. `sends` is what the template's `reasoning_effort` receives
/// (`effort.templateValue`, static), the value every render passes.
pub const ReasoningEffort = struct { enable: bool, budget: i32, effort: ?[]const u8 = null, rerouted_from: ?[]const u8 = null, sends: ?[]const u8 = null };

/// OpenAI-standard `reasoning_effort` on chat/completions (none | minimal |
/// low | medium | high | xhigh — values are model-dependent, so unknown
/// strings still enable). "none" is an explicit off (the gpt-5.1 default
/// spelling). Absent or not a string → null (a non-word is refused first, `nonWordEffort`): the vendor `enable_thinking` bool
/// stays in charge and existing clients see zero behavior change.
/// The effort word is the ONLY lever when the template reads it and no
/// decode-time bound can enforce a budget derived from it (a budget that is
/// only a delivery cap fights the word: `chat.templateConsumesEffort`).
pub fn effortWordOnly(allocator: std.mem.Allocator, lm: *LoadedModel, tok: *const Tokenizer) bool {
    const cc = lm.chat_config orelse return false;
    return chat_mod.templateConsumesEffort(cc) and !thinkMarkersAtomic(allocator, lm, tok);
}

pub fn parseReasoningEffort(root: std.json.ObjectMap, template_consumes_effort: bool, arms: ?[]const effort_mod.EffortArm) error{EffortRefused}!?ReasoningEffort {
    const v = root.get("reasoning_effort") orelse return null;
    if (v != .string) return null;
    return try reasoningEffortFromWord(v.string, template_consumes_effort, arms);
}

/// One effort word → one thinking config, whatever field carried the word —
/// OpenAI's flat `reasoning_effort`, Responses' `reasoning.effort` and
/// Anthropic's `output_config.effort` must not drift on what "low" means.
/// `arms` is the model's served table (`effort.servedArms`), read through `effort.resolve`.
pub fn reasoningEffortFromWord(word: []const u8, template_consumes_effort: bool, arms: ?[]const effort_mod.EffortArm) error{EffortRefused}!ReasoningEffort {
    if (arms) |table| {
        const asked = effort_mod.parseEffort(word) orelse return error.EffortRefused;
        const pick = effort_mod.resolve(table, asked) orelse return error.EffortRefused;
        const arm = pick.arm;
        const from: ?[]const u8 = if (pick.rerouted_from) |f| @tagName(f) else null;
        if (arm.effort == .off) return .{ .enable = false, .budget = -1, .effort = word, .rerouted_from = from, .sends = word };
        // Where the template reads the word, the word is the lever and the table's cap is dropped, unless the row
        // sends the template another level's word (Hy4-Flash's low and medium send `high`): there the cap is what
        // tells the level apart, and it holds.
        const cap_is_lever = switch (arm.sends) {
            .word => |w| w != arm.effort,
            else => false,
        };
        const budget = if (template_consumes_effort and !cap_is_lever) -1 else arm.budget orelse -1;
        return .{ .enable = true, .budget = budget, .effort = @tagName(arm.effort), .rerouted_from = from, .sends = effort_mod.templateValue(arm) };
    }
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = -1, .effort = word, .sends = word };
    // Where the TEMPLATE reads the effort word, the word is the behavioral
    // lever and a budget derived from the same string is pure display
    // truncation on top of it (`chat.templateConsumesEffort`).
    const budget = if (template_consumes_effort) -1 else responses_mod.effortBudget(word);
    return .{ .enable = true, .budget = budget, .effort = word, .sends = word };
}

/// The thinking cap a request runs under: its own `reasoning_budget_tokens`, else its effort's (`effort_budget`).
pub fn requestThinkBudget(req: anytype, effort_budget: i32) i32 {
    return req.reasoningBudget() orelse effort_budget;
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

/// `clampMaxTokens` for a request: a cut the window made to the client's own number also rides the
/// response head, so the client sees it was not granted its ask. A filled budget asked for nothing.
pub fn clampMaxTokensFor(stream: *Conn, max_tokens: u32, client_asked: bool, prompt_len: usize, effective_ctx: u32) u32 {
    const granted = clampMaxTokens(max_tokens, client_asked, prompt_len, effective_ctx);
    stream.max_tokens_cut_len = 0;
    if (client_asked and granted < max_tokens) {
        const text = std.fmt.bufPrint(&stream.max_tokens_cut, "{d} -> {d}", .{ max_tokens, granted }) catch return granted;
        stream.max_tokens_cut_len = @intCast(text.len);
    }
    return granted;
}

pub fn writeMaxTokensCutHeader(stream: *Conn) !void {
    if (stream.max_tokens_cut_len == 0) return;
    try stream.writeAllNoFlush("X-Sushi-Max-Tokens-Cut: ");
    try stream.writeAllNoFlush(stream.max_tokens_cut[0..stream.max_tokens_cut_len]);
    try stream.writeAllNoFlush("\r\nAccess-Control-Expose-Headers: X-Sushi-Max-Tokens-Cut\r\n");
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
    var buf: [std.enums.values(effort_mod.Effort).len]effort_mod.Effort = undefined;
    for (effort_mod.listedWords(arms, &buf), 0..) |e, i| try out.print(allocator, "{s}{s}", .{ if (i > 0) ", " else "", @tagName(e) });
    return out.toOwnedSlice(allocator);
}

/// The `/v1/models` row fields listing the accepted effort words, the thinking-token budget of
/// each capped word, the template value of each thinking word, and the one a request naming none
/// runs at; "" for an arch without a table. Read from the served table on every call.
pub fn reasoningEffortsJson(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig) ![]u8 {
    const arms = effort_mod.servedArms(config.model_type) orelse return allocator.dupe(u8, "");
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, ",\"reasoning_efforts\":[");
    var first = true;
    for (arms.slice()) |a| if (effort_mod.isListed(a)) {
        try out.print(allocator, "{s}\"{s}\"", .{ if (first) "" else ",", @tagName(a.effort) });
        first = false;
    };
    try out.appendSlice(allocator, "],\"reasoning_effort_budgets\":{");
    first = true;
    for (arms.slice()) |a| if (effort_mod.isListed(a)) if (a.budget) |b| {
        try out.print(allocator, "{s}\"{s}\":{d}", .{ if (first) "" else ",", @tagName(a.effort), b });
        first = false;
    };
    try out.appendSlice(allocator, "},\"reasoning_effort_values\":{");
    first = true;
    for (arms.slice()) |a| if (effort_mod.isListed(a)) switch (a.sends) {
        .none => {},
        .word => |w| {
            try out.print(allocator, "{s}\"{s}\":\"{s}\"", .{ if (first) "" else ",", @tagName(a.effort), @tagName(w) });
            first = false;
        },
        .scale => |n| {
            try out.print(allocator, "{s}\"{s}\":{d}", .{ if (first) "" else ",", @tagName(a.effort), n });
            first = false;
        },
    };
    try out.append(allocator, '}');
    if (model_mod.defaultReasoningEffort(config)) |w| try out.print(allocator, ",\"default_reasoning_effort\":\"{s}\"", .{w});
    return out.toOwnedSlice(allocator);
}

pub const EffortSpelling = enum { chat, responses, anthropic };

/// The request's effort field in `spelling`'s place when it holds something other than a word or null (a number, an
/// object): refused like an unknown word, never ignored.
pub fn nonWordEffort(root: std.json.ObjectMap, spelling: EffortSpelling) ?std.json.Value {
    const v = switch (spelling) {
        .chat => root.get("reasoning_effort"),
        .responses => if (root.get("reasoning")) |r| (if (r == .object) r.object.get("effort") else null) else null,
        .anthropic => if (root.get("output_config")) |o| (if (o == .object) o.object.get("effort") else null) else null,
    } orelse return null;
    return switch (v) {
        .string, .null => null,
        else => v,
    };
}

/// The 400 for a `nonWordEffort`: the unknown-word refusal, naming the value as sent.
pub fn sendEffortValueRefusal(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, v: std.json.Value, snap: *const effort_mod.Snapshot, anthropic: bool) !void {
    const text = try std.json.Stringify.valueAlloc(allocator, v, .{});
    defer allocator.free(text);
    const config = lm.config.?;
    const msg = if (effort_mod.servedArmsIn(snap, config.model_type) != null)
        try effortRefusalFor(allocator, lm, text, snap)
    else
        try std.fmt.allocPrint(allocator, "reasoning effort {s} is not a word", .{text});
    defer allocator.free(msg);
    if (anthropic) try sendAnthropicError(allocator, stream, "invalid_request_error", msg, 400) else try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", msg, 400);
}

/// The refusal text for `word` on `lm`'s model; caller frees.
pub fn effortRefusalFor(allocator: std.mem.Allocator, lm: *LoadedModel, word: []const u8, snap: *const effort_mod.Snapshot) ![]u8 {
    const config = lm.config.?;
    const arms = effort_mod.servedArmsIn(snap, config.model_type).?;
    return effortRefusal(allocator, word, if (lm.id.len > 0) lm.id else config.model_type, arms.slice());
}

/// The client's thinking bool: top-level `enable_thinking`, else `chat_template_kwargs.enable_thinking`.
/// Responses thinking: a `reasoning` object decides alone; without one the request
/// bool (`Checked.enableThinking`), else the arch default, as on chat.
pub fn responsesEnableThinking(req: request_fields.Checked(.responses), from_reasoning: bool, arch_default: bool) bool {
    if (req.root.get("reasoning")) |r| if (r != .null) return from_reasoning;
    return req.enableThinking() orelse arch_default;
}

/// Resolve thinking for a chat request. An EXPLICIT client signal always wins —
/// the vendor `enable_thinking` bool, or an OpenAI `reasoning_effort` string
/// ("none" being an explicit OFF). Only a request that names NEITHER falls
/// through to the arch default (`ModelConfig.defaultEnableThinking`).
///
/// The two knobs stay OR'd when both are present, as they always were.
pub fn checkThinkingSupport(allocator: std.mem.Allocator, stream: *Conn, lm: *LoadedModel, snap: *const effort_mod.Snapshot, enabled: bool, anthropic: bool) !bool {
    const config = lm.config.?;
    if (!thinkingOffRefused(config, snap, enabled)) return true;
    const message = if (arch_chat.thinkOffRefusal(config.model_type)) |refusal|
        try allocator.dupe(u8, refusal)
    else
        try effortRefusalFor(allocator, lm, "off", snap);
    defer allocator.free(message);
    if (anthropic) try sendAnthropicError(allocator, stream, "invalid_request_error", message, 400) else try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", message, 400);
    return false;
}

/// Thinking off where the model's served table does not offer `off`, as `reasoning_effort: "off"` is refused there. A
/// request that thinks off only because the model does (a checkpoint declaring it) passes; GLM's template has no off.
pub fn thinkingOffRefused(config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot, enabled: bool) bool {
    if (enabled) return false;
    if (arch_chat.thinkOffRefusal(config.model_type) != null) return true;
    const arms = effort_mod.servedArmsIn(snap, config.model_type) orelse return false;
    if (effort_mod.findEffortArm(arms.slice(), .off) != null) return false;
    return config.defaultEnableThinkingIn(snap);
}

pub fn resolveEnableThinking(req: anytype, effort_cfg: ?ReasoningEffort, arch_default: bool) bool {
    const et = req.enableThinking();
    if (et == null and effort_cfg == null) return arch_default;
    return (et orelse false) or (if (effort_cfg) |e| e.enable else false);
}

/// A request's thinking: on or off, its cap (-1 = none) and the template's `reasoning_effort`.
pub const Thinking = struct { enable: bool, budget: i32, template: ?[]const u8 };

/// Qwen under an `off` default (`--think off`, the table's): a request naming no effort word thinks only when it names a
/// thinking budget, at the level silence renders (`chat.qwen38EffortFor`); its on switch alone leaves it off.
fn holdOffDefault(t: *Thinking, config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot, names_effort: bool, names_budget: bool, explicit_off: bool) void {
    if (names_effort or !arch_config.fact(config, "thinking_off_checkpoint") or !model_mod.defaultEffortIsOff(config, snap)) return;
    t.enable = names_budget and !explicit_off;
    if (!t.enable) t.budget = -1;
    t.template = model_mod.defaultTemplateEffortIn(config, snap);
}

fn namesThinkBudget(req: anytype) bool {
    return req.reasoningBudget() != null;
}

/// `/v1/chat/completions`: `effort_cfg` is the request's `reasoning_effort`; the cap is `reasoning_budget_tokens`, else the word's.
/// `snap` is the effort table the request read its word from (`effort.snapshot`).
pub fn chatThinking(req: request_fields.Checked(.chat), effort_cfg: ?ReasoningEffort, config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot) Thinking {
    const effort_budget = if (effort_cfg) |e| e.budget else model_mod.defaultThinkBudgetIn(config, snap);
    var t: Thinking = .{
        .enable = resolveEnableThinking(req, effort_cfg, config.defaultEnableThinkingIn(snap)),
        .budget = requestThinkBudget(req, effort_budget),
        .template = if (effort_cfg) |e| e.sends else model_mod.defaultTemplateEffortIn(config, snap),
    };
    holdOffDefault(&t, config, snap, effort_cfg != null, namesThinkBudget(req), req.enableThinking() == false);
    return t;
}

/// `/v1/messages`: a `thinking` object decides, OR'd with `output_config.effort` (`effort_cfg`); `budget_tokens`
/// outranks the word's cap.
pub fn anthropicThinking(req: request_fields.Checked(.messages), effort_cfg: ?ReasoningEffort, config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot) Thinking {
    const asked = req.thinking();
    var t: Thinking = .{
        .enable = if (asked == null) config.defaultEnableThinkingIn(snap) else false,
        .budget = model_mod.defaultThinkBudgetIn(config, snap),
        .template = model_mod.defaultTemplateEffortIn(config, snap),
    };
    var budget_explicit = false;
    var disabled = false;
    if (asked) |a| {
        t.enable = a.kind != .disabled;
        disabled = a.kind == .disabled;
        if (a.budget_tokens) |b| {
            t.budget = b;
            budget_explicit = true;
        }
    }
    if (effort_cfg) |cfg| {
        t.template = cfg.sends;
        if (!budget_explicit) t.budget = cfg.budget;
        t.enable = if (asked == null) cfg.enable else (t.enable or cfg.enable);
    }
    holdOffDefault(&t, config, snap, effort_cfg != null, budget_explicit, disabled);
    return t;
}

/// The effort word a `/v1/responses` request runs at: `reasoning.effort`, else the default word where it has no
/// `reasoning` or one that turns thinking on.
pub fn responsesEffortWord(root: std.json.ObjectMap, parsed: responses_mod.ReasoningConfig, config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot) ?[]const u8 {
    if (parsed.effort) |w| return w;
    if (root.get("reasoning") == null or parsed.enable) return model_mod.defaultEffortWordIn(config, snap);
    return null;
}

/// `/v1/responses`: `word_cfg` is the resolved `responsesEffortWord`; `named` says the request named it. A `reasoning`
/// object decides alone.
pub fn responsesThinking(req: request_fields.Checked(.responses), reasoning_enable: bool, word_cfg: ?ReasoningEffort, named: bool, config: *const model_mod.ModelConfig, snap: *const effort_mod.Snapshot) Thinking {
    const effort_budget = if (word_cfg) |c| c.budget else model_mod.defaultThinkBudgetIn(config, snap);
    var t: Thinking = .{
        .enable = responsesEnableThinking(req, if (word_cfg) |c| c.enable else reasoning_enable, config.defaultEnableThinkingIn(snap)),
        .budget = requestThinkBudget(req, effort_budget),
        .template = if (word_cfg) |c| c.sends else null,
    };
    holdOffDefault(&t, config, snap, named, namesThinkBudget(req), req.enableThinking() == false);
    return t;
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
        var r = requestSpecModes(true, true, true, true, sampling);
        try std.testing.expect(r.use_mtp and !r.use_drafter);
        try std.testing.expectEqual(StreamMode.mtp, pickStreamMode(true, true, true, true, sampling));
        r = requestSpecModes(true, true, false, true, sampling);
        try std.testing.expect(r.use_mtp and !r.use_drafter);
        // Without MTP the penalty leaves the sidecar nothing to run.
        r = requestSpecModes(true, false, true, false, sampling);
        try std.testing.expect(!r.use_mtp and !r.use_drafter);
    }
}

/// The template's `preserve_thinking`: request > `--preserve-thinking` >
/// model-settings.json; null leaves the template's own default.
pub fn resolvePreserveThinking(request: ?bool, flag: ?bool, setting: ?bool) ?bool {
    return request orelse flag orelse setting;
}

test "clampMaxTokens honors the effective context (auto-pinned, not just --ctx-size)" {
    const original = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = original;
    server_state.server_config.max_context_size = 0; // auto: the old code never clamped here

    // 4096-token window, 4000-token prompt -> only 96 tokens of generation left.
    try testing.expectEqual(@as(u32, 96), clampMaxTokens(8192, true, 4000, 4096));
    // Comfortable budget is passed through untouched.
    try testing.expectEqual(@as(u32, 8192), clampMaxTokens(8192, true, 100, 65536));
    // Prompt at/over the window still yields at least one token.
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, true, 4096, 4096));
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(8192, true, 9000, 4096));
}

test "an omitted max_tokens is a concrete budget an agent's thinking turn fits in, still bounded by the window" {
    // The old fixed 256 default broke agent clients (pi) whose thinking-enabled turns hit
    // `length` mid-reasoning on EVERY request.
    const slot = &@field(server_state.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    const filled = filledMaxTokens(&model_mod.ModelConfig{});
    try testing.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, filled);
    try testing.expect(filled > 256);
    // The window still bounds it.
    try testing.expectEqual(@as(u32, 16384 - 1500), clampMaxTokens(filled, false, 1500, 16384));
}

test "resolveRequestMaxTokens: absent / 0 / negative / non-int → the filled budget; positive → value" {
    const filled: u32 = 12345;
    // Absent → the filled budget (the omitted-default path the app's "Auto" setting rides).
    try testing.expectEqual(filled, resolveRequestMaxTokens(null, filled));
    // 0 and negatives take it too — a client must not be able to request a
    // 0-token (generate-nothing) response by selecting Auto.
    try testing.expectEqual(filled, resolveRequestMaxTokens(.{ .integer = 0 }, filled));
    try testing.expectEqual(filled, resolveRequestMaxTokens(.{ .integer = -5 }, filled));
    // Wrong type (e.g. a string) → the filled budget rather than crashing.
    try testing.expectEqual(filled, resolveRequestMaxTokens(.{ .string = "999" }, filled));
    // A real positive cap is honored verbatim.
    try testing.expectEqual(@as(u32, 4096), resolveRequestMaxTokens(.{ .integer = 4096 }, filled));
    // A whole float is the integer `json.dumps(7.0)` sends.
    try testing.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .float = 7.0 }, filled));
    try testing.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .float = 7.0 }, 0, 0));
}

test "--max-tokens is the omitted-field default: unset is the engine default, set wins" {
    const t = std.testing;
    try t.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, filledMaxTokensWith(0, 0));

    // Flag set: the launch value outranks both fallbacks.
    try t.expectEqual(@as(u32, 65536), filledMaxTokensWith(65536, 0));
    try t.expectEqual(@as(u32, 65536), filledMaxTokensWith(65536, 256_000));

    // The context clamp still sits on top.
    try t.expectEqual(@as(u32, 32768 - 1500), clampMaxTokens(filledMaxTokensWith(65536, 0), false, 1500, 32768));

    // A request that sends max_tokens is untouched by the flag.
    const filled = filledMaxTokensWith(65536, 0);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, filled));
    try t.expectEqual(@as(u32, 200_000), resolveRequestMaxTokens(.{ .integer = 200_000 }, filled));
    try t.expectEqual(@as(u32, 65536), resolveRequestMaxTokens(.{ .integer = 0 }, filled));

    // The live wrapper reads the ServerConfig field through the one accessor.
    const slot = &@field(server_state.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    try t.expectEqual(@as(u32, 0), launchMaxTokensDefault());
    var cfg = model_mod.ModelConfig{ .pinned_context = 32768 };
    try t.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, filledMaxTokens(&cfg));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), launchMaxTokensDefault());
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&cfg));
}

test "an omitted max_tokens: --max-tokens, else the architecture's own default, else the engine's" {
    if (arch_config.ofConfigType("deepseek_v41") == null) return error.SkipZigTest;
    const t = std.testing;
    // DeepSeek with no flag: its own 256,000.
    try t.expectEqual(@as(u32, 256_000), filledMaxTokensWith(0, 256_000));
    // The flag outranks the architecture.
    try t.expectEqual(@as(u32, 4096), filledMaxTokensWith(4096, 256_000));
    // An architecture with no default takes the engine's.
    try t.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, filledMaxTokensWith(0, 0));
    // The context clamp still sits on top, and a client's own number is untouched.
    try t.expectEqual(@as(u32, 100_000 - 1500), clampMaxTokens(256_000, false, 1500, 100_000));
    const dflt = filledMaxTokensWith(0, 256_000);
    try t.expectEqual(@as(u32, 128), resolveRequestMaxTokens(.{ .integer = 128 }, dflt));
    try t.expectEqual(@as(u32, 256_000), resolveRequestMaxTokens(.{ .integer = 0 }, dflt));

    // The log names the origin, and a launch value still wins the label.
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    try t.expectEqual(MaxTokensOrigin.model_default, maxTokensOrigin(null, 0, 256_000));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 4096, 256_000));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0, 256_000));
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(null, 0, 0));
    try t.expectEqualStrings("256000 (model default)", describeMaxTokens(&buf, 256_000, .model_default));

    // The live wrapper reads the flag and the model's own default.
    const slot = &@field(server_state.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    var dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .pinned_context = 1_000_000, .serve_defaults = arch_config.releaseServeDefaults("dsv41") };
    var qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp", .pinned_context = 1_000_000 };
    try t.expectEqual(@as(u32, 256_000), filledMaxTokens(&dsv41));
    try t.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, filledMaxTokens(&qwen));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&dsv41));
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&qwen));
}

test "an omitted max_tokens: --max-tokens, else the model's max_tokens setting, else the architecture's default, else the engine's" {
    if (arch_config.ofConfigType("deepseek_v41") == null) return error.SkipZigTest;
    const t = std.testing;
    const slot = &@field(server_state.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    var dsv41 = model_mod.ModelConfig{ .model_type = "deepseek_v41", .pinned_context = 1_000_000, .serve_defaults = arch_config.releaseServeDefaults("dsv41") };
    var qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp", .pinned_context = 1_000_000 };
    dsv41.max_tokens_override = 50_000;
    qwen.max_tokens_override = 12_000;
    try t.expectEqual(@as(u32, 50_000), filledMaxTokens(&dsv41));
    try t.expectEqual(@as(u32, 12_000), filledMaxTokens(&qwen));
    slot.* = 4096;
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&dsv41));
    try t.expectEqual(@as(u32, 4096), filledMaxTokens(&qwen));
}

test "an omitted max_tokens with no flag and no model default is the engine's 32768, never the rest of the window" {
    const t = std.testing;
    const slot = &@field(server_state.server_config, "default_max_tokens");
    const orig = slot.*;
    defer slot.* = orig;
    slot.* = 0;
    const qwen = model_mod.ModelConfig{ .model_type = "qwen4_exp", .pinned_context = 1_000_000 };
    try t.expectEqual(@as(u32, 0), qwen.defaultMaxTokens());
    try t.expectEqual(@as(u32, 32768), filledMaxTokens(&qwen));
    try t.expectEqual(@as(u32, 32768), resolveRequestMaxTokens(null, filledMaxTokens(&qwen)));
    try t.expectEqual(@as(u32, 32768), clampMaxTokens(filledMaxTokens(&qwen), false, 1500, 1_000_000));
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(null, launchMaxTokensDefault(), qwen.defaultMaxTokens()));
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;
    try t.expectEqualStrings("32768 (engine default)", describeMaxTokens(&buf, filledMaxTokens(&qwen), .engine_default));
}

test "describeMaxTokens: a filled budget names its source, a client's number prints bare" {
    const t = std.testing;
    var buf: [MAX_TOKENS_DESC_LEN]u8 = undefined;

    try t.expectEqualStrings("32768 (engine default)", describeMaxTokens(&buf, ENGINE_DEFAULT_MAX_TOKENS, .engine_default));

    // `--max-tokens N` filled an omitted field: the number is real, but it is
    // the operator's, not the client's, and the line says whose it is.
    try t.expectEqualStrings("65536 (launch default)", describeMaxTokens(&buf, 65536, .launch_default));

    // A client that sent the field gets its own number back, unadorned.
    try t.expectEqualStrings("4096", describeMaxTokens(&buf, 4096, .client));

    // Origin comes from the SAME inputs `resolveRequestMaxTokens` reads, so
    // the label can't disagree with the number beside it.
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(null, 0, 0));
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(.{ .integer = 0 }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(.{ .integer = -5 }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.engine_default, maxTokensOrigin(.{ .string = "999" }, 0, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(null, 65536, 0));
    try t.expectEqual(MaxTokensOrigin.launch_default, maxTokensOrigin(.{ .integer = 0 }, 65536, 0));
    // A sent field outranks the flag, so it is the CLIENT's number either way.
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 65536, 0));
    try t.expectEqual(MaxTokensOrigin.client, maxTokensOrigin(.{ .integer = 128 }, 0, 0));

    // The widest value fits the buffer.
    try t.expectEqualStrings("4294967295 (engine default)", describeMaxTokens(&buf, std.math.maxInt(u32), .engine_default));
}

test "the max_tokens squeeze note fires when under a quarter of the budget remains" {
    const t = std.testing;
    // A real over-ask: the window can't pay the budget. That is the warning's whole point
    // (the tool-call truncation nudge); a filled budget logs the same note at debug.
    try t.expect(maxTokensBudgetSqueezed(32768, 4000));
    try t.expect(!maxTokensBudgetSqueezed(32768, 30000));
    try t.expect(maxTokensBudgetSqueezed(65536, 4000));
}

test "an omitted max_tokens reserves the LAUNCH default, not the whole window" {
    const t = std.testing;
    // The admission reservation reads the resolved budget, so the flag reaches it.
    const KVCache = trunk.KVCache;
    const ctx: u64 = 786_432;
    const seq: u64 = 400_000; // past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;

    // No flag: the engine default, reserved whole.
    const no_flag = clampMaxTokens(filledMaxTokensWith(0, 0), false, seq, @intCast(ctx));
    try t.expectEqual(ENGINE_DEFAULT_MAX_TOKENS, no_flag);
    try t.expectEqual(seq + ENGINE_DEFAULT_MAX_TOKENS + chunk, KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // A flag: the reservation is prompt + the flag.
    const flagged = clampMaxTokens(filledMaxTokensWith(2048, 0), false, seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 2048), flagged);
    try t.expectEqual(seq + 2048 + chunk, KVCache.reservedTokens(seq, flagged, chunk, ctx));
    try t.expect(KVCache.reservedTokens(seq, flagged, chunk, ctx) < KVCache.reservedTokens(seq, no_flag, chunk, ctx));

    // A large one is reserved whole too.
    const big = clampMaxTokens(filledMaxTokensWith(65536, 0), false, seq, @intCast(ctx));
    try t.expectEqual(@as(u32, 65536), big);
    try t.expectEqual(seq + 65536 + chunk, KVCache.reservedTokens(seq, big, chunk, ctx));
}

test "clampMaxTokens clamps when would exceed" {
    const original = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = original;
    server_state.server_config.max_context_size = 4096;

    // prompt=3000, max_tokens=2000 → clamp to 1096
    try testing.expectEqual(@as(u32, 1096), clampMaxTokens(2000, true, 3000, 4096));
}

test "clampMaxTokens no clamp when fits" {
    const original = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = original;
    server_state.server_config.max_context_size = 4096;

    // prompt=100, max_tokens=200 → fits, no clamp
    try testing.expectEqual(@as(u32, 200), clampMaxTokens(200, true, 100, 4096));
}

test "clampMaxTokens at boundary" {
    const original = server_state.server_config.max_context_size;
    defer server_state.server_config.max_context_size = original;
    server_state.server_config.max_context_size = 4096;

    // prompt=4096 → only 1 token allowed
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, true, 4096, 4096));
    // prompt=4095 → only 1 token remaining
    try testing.expectEqual(@as(u32, 1), clampMaxTokens(100, true, 4095, 4096));
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
        const got = try parseReasoningEffort(parsed.value.object, false, null);
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
    // An EXPLICIT cap is someone asking on purpose: the request's own
    // `reasoning_budget_tokens` (`requestThinkBudget`).
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, consumes: bool, want: i32 }{
        // Consuming template (qwen3.8 / deepseek / glm): no effort-derived cap.
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = true, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = true, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = true, .want = -1 },
        // Non-consuming template: the effort word reaches no template, so the budget is the ONLY
        // thing it does and every mapping stays exactly as it shipped.
        .{ .body = "{\"reasoning_effort\":\"minimal\"}", .consumes = false, .want = 1024 },
        .{ .body = "{\"reasoning_effort\":\"low\"}", .consumes = false, .want = 2048 },
        .{ .body = "{\"reasoning_effort\":\"medium\"}", .consumes = false, .want = 8192 },
        // high and xhigh are uncapped: pi's own ladder is 2048 / 8192 / 16384.
        .{ .body = "{\"reasoning_effort\":\"high\"}", .consumes = false, .want = -1 },
        .{ .body = "{\"reasoning_effort\":\"xhigh\"}", .consumes = false, .want = -1 },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = (try parseReasoningEffort(parsed.value.object, case.consumes, null)).?;
        try std.testing.expect(got.enable);
        try std.testing.expectEqual(case.want, got.budget);
        // The raw string rides along either way — the template still renders it.
        try std.testing.expectEqualStrings(parsed.value.object.get("reasoning_effort").?.string, got.effort.?);
    }
}

test "reasoningEffortFromWord: none disables, words budget exactly like the OpenAI surface" {
    // "none" is an explicit off with the word kept for the template.
    const off = try reasoningEffortFromWord("none", false, null);
    try std.testing.expect(!off.enable);
    try std.testing.expectEqualStrings("none", off.effort.?);
    // A word maps through the ONE effortBudget table…
    const low = try reasoningEffortFromWord("low", false, null);
    try std.testing.expect(low.enable);
    try std.testing.expectEqual(@as(i32, 2048), low.budget);
    // …unless the template consumes the word (qwen3.8 class): then the word is
    // the lever and the budget stays the launch default.
    const consumed = try reasoningEffortFromWord("low", true, null);
    try std.testing.expect(consumed.enable);
    try std.testing.expectEqual(@as(i32, -1), consumed.budget);
}

test "reasoningEffortFromWord: a table's cap holds where the template reads the word only when the row sends another level's word" {
    // Hy4-Flash (owner, 2026-11-18): low and medium think at the template's `high` under 2,048 and 16,384 tokens. The
    // template cannot tell them from high, so the table's cap holds; Qwen's low sends `low` itself and keeps none.
    try test_note.needArchs(&.{"hy4flash"});
    const hy4 = effort_mod.effortArms("hy_v4_flash").?;
    const low = try reasoningEffortFromWord("low", true, hy4);
    try std.testing.expectEqual(@as(i32, 2048), low.budget);
    try std.testing.expectEqualStrings("high", low.sends.?);
    const medium = try reasoningEffortFromWord("medium", true, hy4);
    try std.testing.expectEqual(@as(i32, 16384), medium.budget);
    try std.testing.expectEqualStrings("high", medium.sends.?);
    const high = try reasoningEffortFromWord("high", true, hy4);
    try std.testing.expectEqual(@as(i32, -1), high.budget);
    const xhigh = try reasoningEffortFromWord("xhigh", true, hy4);
    try std.testing.expectEqual(@as(i32, -1), xhigh.budget);
    try std.testing.expectEqualStrings("max", xhigh.sends.?);
    try std.testing.expect(!(try reasoningEffortFromWord("off", true, hy4)).enable);
}

test "reasoningEffortFromWord: Hy4-Flash aliases minimal and on follow their rows and refuse when unoffered" {
    try test_note.needArchs(&.{"hy4flash"});
    defer effort_mod.publish(.hy_v4_flash, .{});
    const hy4 = effort_mod.servedArms("hy_v4_flash").?;
    const min_arm = try reasoningEffortFromWord("minimal", true, hy4.slice());
    try std.testing.expect(min_arm.enable);
    try std.testing.expectEqual(@as(i32, 2048), min_arm.budget);
    try std.testing.expectEqualStrings("high", min_arm.sends.?);
    try std.testing.expect(min_arm.rerouted_from == null);
    const on_arm = try reasoningEffortFromWord("on", true, hy4.slice());
    try std.testing.expect(on_arm.enable);
    try std.testing.expectEqual(@as(i32, -1), on_arm.budget);
    try std.testing.expectEqualStrings("high", on_arm.sends.?);
    try std.testing.expect(on_arm.rerouted_from == null);
    try std.testing.expect(!(try reasoningEffortFromWord("none", true, hy4.slice())).enable);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", true, hy4.slice()));
    var edits: effort_mod.Edits = .{};
    edits.rows[effort_mod.rowIndex(.low).?].offered = false;
    effort_mod.publish(.hy_v4_flash, edits);
    const edited = effort_mod.servedArms("hy_v4_flash").?;
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("low", true, edited.slice()));
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", true, edited.slice()));
    const still_on = try reasoningEffortFromWord("on", true, edited.slice());
    try std.testing.expectEqualStrings("high", still_on.sends.?);
}

test "reasoningEffortFromWord: a served arch refuses unknown words, and re-routes only the announced folds" {

    try test_note.needArchs(&.{ "qwen4", "mimo2", "glm5" });
    const qwen = effort_mod.effortArms("qwen4_exp").?;
    const mimo = effort_mod.effortArms("mimo_v2").?;
    const glm = effort_mod.effortArms("glm5_next").?;
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", false, qwen));
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", false, qwen));
    for ([_][]const u8{ "high", "max" }) |w| {
        const got = try reasoningEffortFromWord(w, false, qwen);
        try std.testing.expect(got.enable);
        try std.testing.expectEqualStrings("xhigh", got.effort.?);
        try std.testing.expectEqualStrings(w, got.rerouted_from.?);
        try std.testing.expectEqual(@as(i32, -1), got.budget);
    }
    const consumed = try reasoningEffortFromWord("max", true, qwen);
    try std.testing.expectEqual(@as(i32, -1), consumed.budget);
    try std.testing.expectEqualStrings("xhigh", consumed.effort.?);
    const glm_med = try reasoningEffortFromWord("medium", false, glm);
    try std.testing.expect(glm_med.enable);
    try std.testing.expectEqualStrings("high", glm_med.effort.?);
    try std.testing.expectEqualStrings("medium", glm_med.rerouted_from.?);
    const glm_xh = try reasoningEffortFromWord("xhigh", false, glm);
    try std.testing.expect(glm_xh.enable);
    try std.testing.expectEqualStrings("max", glm_xh.effort.?);
    try std.testing.expectEqualStrings("xhigh", glm_xh.rerouted_from.?);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("off", false, glm));
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", false, glm));
    const glm_high = try reasoningEffortFromWord("high", false, glm);
    try std.testing.expect(glm_high.rerouted_from == null);
    try std.testing.expectEqualStrings("high", glm_high.effort.?);
    const mimo_high = try reasoningEffortFromWord("high", false, mimo);
    try std.testing.expectEqualStrings("on", mimo_high.effort.?);
    try std.testing.expect(mimo_high.rerouted_from == null);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", false, mimo));
    for ([_][]const u8{ "off", "none" }) |w| {
        try std.testing.expect(!(try reasoningEffortFromWord(w, false, qwen)).enable);
        try std.testing.expect(!(try reasoningEffortFromWord(w, false, mimo)).enable);
    }
    const q_low = try reasoningEffortFromWord("low", false, qwen);
    try std.testing.expect(q_low.enable);
    try std.testing.expectEqual(@as(i32, 2048), q_low.budget);
    try std.testing.expectEqualStrings("low", q_low.effort.?);
    try std.testing.expectEqual(@as(i32, -1), (try reasoningEffortFromWord("low", true, qwen)).budget);
    try std.testing.expectEqualStrings("xhigh", (try reasoningEffortFromWord("xhigh", false, qwen)).effort.?);
    // An uncapped arm has no cap.
    try std.testing.expectEqual(@as(i32, -1), (try reasoningEffortFromWord("on", false, mimo)).budget);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", false, qwen));
    try std.testing.expect((try reasoningEffortFromWord("ultra", false, null)).enable);
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

    try test_note.needArchs(&.{"qwen4"});
    const allocator = std.testing.allocator;
    const qwen = effort_mod.effortArms("qwen4_exp").?;
    const msg = try effortRefusal(allocator, "minimal", "Qwen3.8-Flash-Next-EXL3-K4", qwen);
    defer allocator.free(msg);
    try std.testing.expectEqualStrings("reasoning effort 'minimal' is not supported by Qwen3.8-Flash-Next-EXL3-K4; use one of: off, low, medium, xhigh", msg);
    const row = try reasoningEffortsJson(allocator, &.{ .model_type = "qwen4_exp" });
    defer allocator.free(row);
    try std.testing.expectEqualStrings(",\"reasoning_efforts\":[\"off\",\"low\",\"medium\",\"xhigh\"],\"reasoning_effort_budgets\":{\"low\":2048,\"medium\":8192},\"reasoning_effort_values\":{\"low\":\"low\",\"medium\":\"medium\",\"xhigh\":\"xhigh\"},\"default_reasoning_effort\":\"medium\"", row);
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
        // vLLM / SGLang clients send the template switch nested; the top-level field wins over it.
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .arch = true, .want = false },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":true}}", .arch = false, .want = true },
        .{ .body = "{\"enable_thinking\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}", .arch = false, .want = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const effort = try parseReasoningEffort(parsed.value.object, false, null);
        try std.testing.expectEqual(case.want, resolveEnableThinking(try checkedOf(.chat, parsed.value.object), effort, case.arch));
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
        const from_reasoning = responses_mod.parseReasoning(root.get("reasoning")).enable;
        try std.testing.expectEqual(case.want, responsesEnableThinking(try checkedOf(.responses, root), from_reasoning, case.arch));
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

test "a request integer from a client is read without a blind cast" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .integer = 1 << 32 }, 7));
    try t.expectEqual(@as(u32, 512), resolveRequestMaxTokens(.{ .integer = 512 }, 7));
    try t.expectEqual(@as(u32, 7), resolveRequestMaxTokens(.{ .integer = 0 }, 7));
}

test "preserve_thinking: request > --preserve-thinking > model-settings.json > the template's default" {
    const t = std.testing;
    try t.expectEqual(@as(?bool, false), resolvePreserveThinking(false, true, true));
    try t.expectEqual(@as(?bool, true), resolvePreserveThinking(true, false, false));
    try t.expectEqual(@as(?bool, true), resolvePreserveThinking(null, true, false));
    try t.expectEqual(@as(?bool, false), resolvePreserveThinking(null, null, false));
    try t.expectEqual(@as(?bool, null), resolvePreserveThinking(null, null, null));
}

test "think penalty: request > --think-penalty > model-settings.json > off" {
    const Case = struct { request: ?f32, flag: ?f32, setting: ?f32, want: f32 };
    const cases = [_]Case{
        .{ .request = 0, .flag = 2, .setting = 3, .want = 0 },
        .{ .request = 1.5, .flag = null, .setting = null, .want = 1.5 },
        .{ .request = null, .flag = 2, .setting = 3, .want = 2 },
        .{ .request = null, .flag = null, .setting = 3, .want = 3 },
        .{ .request = null, .flag = null, .setting = null, .want = 0 },
    };
    for (cases) |c| try std.testing.expectEqual(c.want, resolveThinkPenalty(c.request, c.flag, c.setting));
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
    const per_request = renderConfigFor(&model_config, false, null, true);
    try std.testing.expectEqual(@as(?bool, false), per_request.preserve_thinking);
    try std.testing.expectEqual(@as(?bool, null), model_config.preserve_thinking);
    try std.testing.expectEqual(@as(?bool, true), renderConfigFor(&model_config, null, null, true).preserve_thinking);
    try std.testing.expectEqual(@as(?bool, false), renderConfigFor(&model_config, null, false, true).preserve_thinking);
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
            const effort = try parseReasoningEffort(root, false, effort_mod.effortArms(family[0]));
            const enable = resolveEnableThinking(try checkedOf(.chat, root), effort, true);
            try std.testing.expectEqual(think != .effort or think.effort != .off, enable);
            if (think == .effort) {
                try std.testing.expectEqual(@as(i32, switch (think.effort) {
                    .low => 2048,
                    .medium => 8192,
                    else => -1,
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

    try test_note.needArchs(&.{ "qwen4", "mimo2", "glm5" });
    for ([_]struct { arch: []const u8, accepted: []const effort_mod.Effort }{
        .{ .arch = "glm5_next", .accepted = &.{ .low, .high, .max } },
        .{ .arch = "qwen4_exp", .accepted = &.{ .off, .low, .medium, .xhigh } },
    }) |c| {
        for (std.enums.values(effort_mod.Effort)) |e| {
            const result = reasoningEffortFromWord(@tagName(e), true, effort_mod.effortArms(c.arch));
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
        try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("minimal", true, effort_mod.effortArms(c.arch)));
    }
    const mimo = effort_mod.effortArms("mimo_v2");
    for ([_][]const u8{ "on", "minimal", "low", "medium", "high", "xhigh", "max" }) |w| {
        const got = try reasoningEffortFromWord(w, true, mimo);
        try std.testing.expect(got.enable);
        try std.testing.expectEqualStrings("on", got.effort.?);
        try std.testing.expect(got.rerouted_from == null);
    }
    for ([_][]const u8{ "off", "none" }) |w| try std.testing.expect(!(try reasoningEffortFromWord(w, true, mimo)).enable);
    try std.testing.expectError(error.EffortRefused, reasoningEffortFromWord("ultra", true, mimo));
}

test "the sampler values an omitted field gets: launch flag, else the model's generation_config, else the fallback" {
    const t = std.testing;
    const cfg_slot = &server_state.server_config;
    const saved = .{ cfg_slot.default_temperature, cfg_slot.default_top_p, cfg_slot.default_top_k };
    defer {
        cfg_slot.default_temperature, cfg_slot.default_top_p, cfg_slot.default_top_k = saved;
    }
    cfg_slot.default_temperature = null;
    cfg_slot.default_top_p = null;
    cfg_slot.default_top_k = null;
    var bare = model_mod.ModelConfig{};
    try t.expectEqual(@as(f32, 1.0), samplingTemperature(&bare, null));
    try t.expectEqual(@as(f32, 1.0), samplingTopP(&bare, null));
    try t.expectEqual(@as(u32, 0), samplingTopK(&bare, null));
    var tuned = model_mod.ModelConfig{ .gen_temperature = 0.6, .gen_top_p = 0.95, .gen_top_k = 20 };
    try t.expectEqual(@as(f32, 0.6), samplingTemperature(&tuned, null));
    try t.expectEqual(@as(u32, 20), samplingTopK(&tuned, null));
    cfg_slot.default_temperature = 0.3;
    try t.expectEqual(@as(f32, 0.3), samplingTemperature(&tuned, null));
    try t.expectEqual(@as(f32, 0.9), samplingTemperature(&tuned, 0.9));
}

const CharResolved = struct { enable: bool, budget: i32, effort: ?[]const u8, from: ?[]const u8 = null, template: ?[]const u8 };

fn charResolve(arch: []const u8, word: []const u8) error{EffortRefused}!CharResolved {
    const arms = effort_mod.servedArms(arch);
    const r = try reasoningEffortFromWord(word, false, if (arms) |*a| a.slice() else null);
    return .{ .enable = r.enable, .budget = r.budget, .effort = r.effort, .from = r.rerouted_from, .template = r.sends };
}

fn charDefault(config: *const model_mod.ModelConfig) CharResolved {
    const word = model_mod.defaultEffortWord(config);
    return .{ .enable = config.defaultEnableThinking(), .budget = model_mod.defaultThinkBudget(config), .effort = word, .template = model_mod.defaultTemplateEffort(config) };
}

test "effort table: with no edits, every word on every family resolves and renders as before the table" {

    try test_note.needArchs(&.{ "qwen4", "mimo2", "glm5", "dsv41" });
    const a = std.testing.allocator;
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    effort_mod.think_effort_flag = null;
    const families = .{
        .{ "qwen4_exp", @embedFile("fixtures/qwen38_flash_next_chat_template.jinja") },
        .{ "deepseek_v41", @embedFile("fixtures/dsv41_chat_template.jinja") },
        .{ "glm5_next", @embedFile("fixtures/glm53_chat_template.jinja") },
        .{ "mimo_v2", @embedFile("fixtures/mimo_v26_chat_template.jinja") },
    };
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    inline for (families) |family| {
        var cc: chat_mod.ChatConfig = .{ .chat_template = family[1], .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
        const msgs = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
        const words = [_]?[]const u8{ null, "off", "none", "on", "minimal", "low", "medium", "high", "xhigh", "max" };
        for (words) |maybe_word| {
            const config = model_mod.ModelConfig{ .model_type = family[0] };
            const r: CharResolved = if (maybe_word) |word| charResolve(family[0], word) catch {
                try w.print("{s} {s}: refused\n", .{ family[0], word });
                continue;
            } else charDefault(&config);
            const rendered = try chat_mod.renderChatTemplate(a, &msgs, &cc, null, null, r.enable, r.template, false);
            defer a.free(rendered);
            try w.print("{s} {s}: on={} budget={d} effort={s} from={s} prompt={x}\n", .{ family[0], maybe_word orelse "(none)", r.enable, r.budget, r.effort orelse "-", r.from orelse "-", std.hash.Wyhash.hash(0, rendered) });
        }
    }
    try std.testing.expectEqualStrings(
        \\qwen4_exp (none): on=true budget=8192 effort=medium from=- prompt=6c54ad9c9be9f8a8
        \\qwen4_exp off: on=false budget=-1 effort=off from=- prompt=75ced9bd25b4f3fe
        \\qwen4_exp none: on=false budget=-1 effort=none from=- prompt=75ced9bd25b4f3fe
        \\qwen4_exp on: refused
        \\qwen4_exp minimal: refused
        \\qwen4_exp low: on=true budget=2048 effort=low from=- prompt=d698954f74b0f08e
        \\qwen4_exp medium: on=true budget=8192 effort=medium from=- prompt=6c54ad9c9be9f8a8
        \\qwen4_exp high: on=true budget=-1 effort=xhigh from=high prompt=feb55f714fd50d51
        \\qwen4_exp xhigh: on=true budget=-1 effort=xhigh from=- prompt=feb55f714fd50d51
        \\qwen4_exp max: on=true budget=-1 effort=xhigh from=max prompt=feb55f714fd50d51
        \\deepseek_v41 (none): on=true budget=-1 effort=high from=- prompt=8bbeed4af7ee1125
        \\deepseek_v41 off: on=false budget=-1 effort=off from=- prompt=b2110a89b409fdf1
        \\deepseek_v41 none: on=false budget=-1 effort=none from=- prompt=b2110a89b409fdf1
        \\deepseek_v41 on: refused
        \\deepseek_v41 minimal: refused
        \\deepseek_v41 low: on=true budget=-1 effort=low from=- prompt=58b1c1f9b941b279
        \\deepseek_v41 medium: on=true budget=-1 effort=medium from=- prompt=8cca813d5afbbd21
        \\deepseek_v41 high: on=true budget=-1 effort=high from=- prompt=8bbeed4af7ee1125
        \\deepseek_v41 xhigh: on=true budget=-1 effort=xhigh from=- prompt=2b7a3626e78b8a3a
        \\deepseek_v41 max: on=true budget=-1 effort=max from=- prompt=7c02131b55c141f9
        \\glm5_next (none): on=true budget=-1 effort=high from=- prompt=d6493052f6da6c3
        \\glm5_next off: refused
        \\glm5_next none: refused
        \\glm5_next on: refused
        \\glm5_next minimal: refused
        \\glm5_next low: on=true budget=-1 effort=low from=- prompt=7e5f3e07767348d1
        \\glm5_next medium: on=true budget=-1 effort=high from=medium prompt=d6493052f6da6c3
        \\glm5_next high: on=true budget=-1 effort=high from=- prompt=d6493052f6da6c3
        \\glm5_next xhigh: on=true budget=-1 effort=max from=xhigh prompt=a4e3fd2d3dbc083e
        \\glm5_next max: on=true budget=-1 effort=max from=- prompt=a4e3fd2d3dbc083e
        \\mimo_v2 (none): on=true budget=-1 effort=- from=- prompt=77031caa3b7f5fac
        \\mimo_v2 off: on=false budget=-1 effort=off from=- prompt=d2d1836d0fd8236d
        \\mimo_v2 none: on=false budget=-1 effort=none from=- prompt=d2d1836d0fd8236d
        \\mimo_v2 on: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 minimal: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 low: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 medium: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 high: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 xhigh: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\mimo_v2 max: on=true budget=-1 effort=on from=- prompt=77031caa3b7f5fac
        \\
    , out.written());
}

fn publishRow(f: effort_mod.Family, word: effort_mod.Effort, row: effort_mod.RowEdit) void {
    var e = effort_mod.publishedEdits(f);
    e.rows[effort_mod.rowIndex(word).?] = row;
    effort_mod.publish(f, e);
}

fn resetTables() void {
    for (effort_mod.families) |f| effort_mod.publish(f, .{});
}

test "effort table: a remapped word is served as itself with its own value and cap, with no notice" {

    try test_note.needArchs(&.{ "qwen4", "glm5", "dsv41" });
    defer resetTables();
    publishRow(.qwen4_exp, .high, .{ .offered = true, .sends = .{ .word = .medium }, .cap = @as(?i32, 64) });
    publishRow(.deepseek_v41, .medium, .{ .sends = .{ .scale = 40 }, .cap = @as(?i32, 64) });
    publishRow(.glm5_next, .low, .{ .cap = @as(?i32, 64) });
    for ([_]struct { arch: []const u8, word: []const u8, sends: []const u8 }{
        .{ .arch = "qwen4_exp", .word = "high", .sends = "medium" },
        .{ .arch = "deepseek_v41", .word = "medium", .sends = "40" },
        .{ .arch = "glm5_next", .word = "low", .sends = "low" },
    }) |c| {
        const r = try charResolve(c.arch, c.word);
        try std.testing.expect(r.enable);
        try std.testing.expectEqual(@as(i32, 64), r.budget);
        try std.testing.expectEqualStrings(c.word, r.effort.?);
        try std.testing.expectEqualStrings(c.sends, r.template.?);
        try std.testing.expect(r.from == null);
    }
    // A row the table leaves uncapped has no cap.
    try std.testing.expectEqual(@as(i32, -1), (try charResolve("deepseek_v41", "high")).budget);
    try std.testing.expectEqual(@as(i32, -1), (try charResolve("glm5_next", "high")).budget);
    // Qwen max is still not offered: it folds to xhigh, with the notice.
    const max = try charResolve("qwen4_exp", "max");
    try std.testing.expectEqualStrings("xhigh", max.effort.?);
    try std.testing.expectEqualStrings("max", max.from.?);
}

test "effort table: a word turned off is refused; a built-in fold still folds, with its notice" {

    try test_note.needArchs(&.{ "qwen4", "glm5", "dsv41" });
    defer resetTables();
    publishRow(.qwen4_exp, .low, .{ .offered = false });
    publishRow(.qwen4_exp, .high, .{ .offered = false });
    publishRow(.deepseek_v41, .medium, .{ .offered = false });
    publishRow(.glm5_next, .low, .{ .offered = false });
    try std.testing.expectError(error.EffortRefused, charResolve("qwen4_exp", "low"));
    try std.testing.expectError(error.EffortRefused, charResolve("qwen4_exp", "high"));
    try std.testing.expectError(error.EffortRefused, charResolve("glm5_next", "low"));
    try std.testing.expectError(error.EffortRefused, charResolve("deepseek_v41", "medium"));
    const folded = try charResolve("qwen4_exp", "max");
    try std.testing.expectEqualStrings("xhigh", folded.effort.?);
    try std.testing.expectEqualStrings("max", folded.from.?);
    try std.testing.expectEqualStrings("xhigh", folded.template.?);
    const arms = effort_mod.servedArms("qwen4_exp").?;
    const msg = try effortRefusal(std.testing.allocator, "low", "Qwen", arms.slice());
    defer std.testing.allocator.free(msg);
    try std.testing.expectEqualStrings("reasoning effort 'low' is not supported by Qwen; use one of: off, medium, xhigh", msg);
}

test "effort table: a request's own budget outranks the table's cap" {

    try test_note.needArchs(&.{"qwen4"});
    defer resetTables();
    publishRow(.qwen4_exp, .high, .{ .offered = true, .cap = @as(?i32, 64) });
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"reasoning_effort\":\"high\",\"reasoning_budget_tokens\":300}", .{});
    defer parsed.deinit();
    const arms = effort_mod.servedArms("qwen4_exp").?;
    const cfg = (try parseReasoningEffort(parsed.value.object, false, arms.slice())).?;
    try std.testing.expectEqual(@as(i32, 64), cfg.budget);
    try std.testing.expectEqual(@as(i32, 300), requestThinkBudget(try checkedOf(.chat, parsed.value.object), cfg.budget));
    const plain = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"reasoning_effort\":\"high\"}", .{});
    defer plain.deinit();
    try std.testing.expectEqual(@as(i32, 64), requestThinkBudget(try checkedOf(.chat, plain.value.object), cfg.budget));
}

test "effort table: /v1/models rows follow the served table, with each word's template value" {

    try test_note.needArchs(&.{ "qwen4", "mimo2", "dsv41" });
    defer resetTables();
    const a = std.testing.allocator;
    {
        const row = try reasoningEffortsJson(a, &.{ .model_type = "deepseek_v41" });
        defer a.free(row);
        try std.testing.expectEqualStrings(",\"reasoning_efforts\":[\"off\",\"low\",\"medium\",\"high\",\"xhigh\",\"max\"],\"reasoning_effort_budgets\":{},\"reasoning_effort_values\":{\"low\":50,\"medium\":62,\"high\":75,\"xhigh\":87,\"max\":100},\"default_reasoning_effort\":\"high\"", row);
    }
    publishRow(.qwen4_exp, .high, .{ .offered = true, .sends = .{ .word = .medium }, .cap = @as(?i32, 64) });
    publishRow(.qwen4_exp, .low, .{ .offered = false });
    var e = effort_mod.publishedEdits(.qwen4_exp);
    e.default = .high;
    effort_mod.publish(.qwen4_exp, e);
    const row = try reasoningEffortsJson(a, &.{ .model_type = "qwen4_exp" });
    defer a.free(row);
    try std.testing.expectEqualStrings(",\"reasoning_efforts\":[\"off\",\"medium\",\"high\",\"xhigh\"],\"reasoning_effort_budgets\":{\"medium\":8192,\"high\":64},\"reasoning_effort_values\":{\"medium\":\"medium\",\"high\":\"medium\",\"xhigh\":\"xhigh\"},\"default_reasoning_effort\":\"high\"", row);
    const mimo = try reasoningEffortsJson(a, &.{ .model_type = "mimo_v2" });
    defer a.free(mimo);
    try std.testing.expectEqualStrings(",\"reasoning_efforts\":[\"off\",\"on\"],\"reasoning_effort_budgets\":{},\"reasoning_effort_values\":{},\"default_reasoning_effort\":\"on\"", mimo);
}

const OffSurface = enum { chat, anthropic, responses };

/// What `surface`'s handler resolves for `body` before the render (no `effortWordOnly` template).
/// A test body's fields checked for `surface`: a body the table refuses is a test defect.
fn checkedOf(comptime surface: request_fields.Surface, root: std.json.ObjectMap) error{RequestRefused}!request_fields.Checked(surface) {
    var buf: [256]u8 = undefined;
    return switch (request_fields.check(surface, &buf, root)) {
        .ok => |c| c,
        .fault => error.RequestRefused,
    };
}

fn surfaceThinking(a: std.mem.Allocator, surface: OffSurface, config: *const model_mod.ModelConfig, body: []const u8) !Thinking {
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{})).object;
    const snap = effort_mod.snapshot();
    const arms = effort_mod.servedArmsIn(&snap, config.model_type);
    const table: ?[]const effort_mod.EffortArm = if (arms) |*x| x.slice() else null;
    switch (surface) {
        .chat => return chatThinking(try checkedOf(.chat, root), try parseReasoningEffort(root, false, table), config, &snap),
        .anthropic => {
            const word: ?[]const u8 = if (root.get("output_config")) |oc| (if (oc.object.get("effort")) |e| e.string else null) else null;
            return anthropicThinking(try checkedOf(.messages, root), if (word) |w| try reasoningEffortFromWord(w, false, table) else null, config, &snap);
        },
        .responses => {
            var rc = responses_mod.parseReasoning(root.get("reasoning"));
            const named = rc.effort != null;
            rc.effort = responsesEffortWord(root, rc, config, &snap);
            return responsesThinking(try checkedOf(.responses, root), rc.enable, if (rc.effort) |w| try reasoningEffortFromWord(w, false, table) else null, named, config, &snap);
        },
    }
}

fn renderHi(a: std.mem.Allocator, template: []const u8, enable: bool, effort: ?[]const u8) ![]const u8 {
    var cc: chat_mod.ChatConfig = .{ .chat_template = template, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
    const msgs = [_]chat_mod.Message{.{ .role = "user", .content = "hi" }};
    return chat_mod.renderChatTemplate(a, &msgs, &cc, null, null, enable, effort, false);
}

const OffDefault = enum { think_flag, table };

/// `--think off`, or the family table's default `off`; the other cleared.
fn setOffDefault(how: OffDefault, f: effort_mod.Family) void {
    effort_mod.think_effort_flag = if (how == .think_flag) .off else null;
    effort_mod.publish(f, if (how == .table) .{ .default = .off } else .{});
}

test "effort off default: Qwen stays off on chat, Anthropic and Responses for a request naming no effort word or budget" {

    try test_note.needArchs(&.{"qwen4"});
    const qwen38_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    defer effort_mod.publish(.qwen4_exp, .{});
    const config = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const cases = [_]struct { OffSurface, []const u8 }{
        .{ .chat, "{\"enable_thinking\":true}" },
        .{ .chat, "{\"chat_template_kwargs\":{\"enable_thinking\":true}}" },
        .{ .chat, "{}" },
        .{ .chat, "{\"enable_thinking\":false,\"reasoning_budget_tokens\":512}" },
        .{ .anthropic, "{\"thinking\":{\"type\":\"enabled\"}}" },
        .{ .anthropic, "{\"thinking\":{\"type\":\"adaptive\"}}" },
        .{ .anthropic, "{}" },
        .{ .anthropic, "{\"thinking\":{\"type\":\"disabled\",\"budget_tokens\":512}}" },
        .{ .responses, "{\"reasoning\":{}}" },
        .{ .responses, "{\"reasoning\":{\"summary\":\"auto\"}}" },
        .{ .responses, "{\"enable_thinking\":true}" },
        .{ .responses, "{}" },
    };
    for ([_]OffDefault{ .think_flag, .table }) |how| {
        setOffDefault(how, .qwen4_exp);
        const asked_off = try surfaceThinking(a, .chat, &config, "{\"reasoning_effort\":\"off\"}");
        const off = try renderHi(a, qwen38_template, asked_off.enable, asked_off.template);
        for (cases) |c| {
            const t = try surfaceThinking(a, c[0], &config, c[1]);
            try testing.expect(!t.enable);
            try testing.expectEqual(@as(i32, -1), t.budget);
            try testing.expectEqualStrings(off, try renderHi(a, qwen38_template, t.enable, t.template));
        }
    }
}

test "effort off default: a Qwen request naming an effort word or a budget still thinks, at that word or Qwen's unnamed level" {

    try test_note.needArchs(&.{"qwen4"});
    const qwen38_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    defer effort_mod.publish(.qwen4_exp, .{});
    const config = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    // Surface, body, budget, and the word whose render it is: a budget alone thinks at low, the level silence renders.
    const cases = [_]struct { OffSurface, []const u8, i32, []const u8 }{
        .{ .chat, "{\"reasoning_effort\":\"low\"}", 2048, "low" },
        .{ .chat, "{\"reasoning_effort\":\"xhigh\"}", -1, "xhigh" },
        .{ .chat, "{\"reasoning_budget_tokens\":512}", 512, "low" },
        .{ .chat, "{\"enable_thinking\":true,\"reasoning_budget_tokens\":512}", 512, "low" },
        .{ .anthropic, "{\"output_config\":{\"effort\":\"medium\"}}", 8192, "medium" },
        .{ .anthropic, "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}}", 1024, "low" },
        .{ .responses, "{\"reasoning\":{\"effort\":\"low\"}}", 2048, "low" },
        .{ .responses, "{\"reasoning\":{},\"reasoning_budget_tokens\":512}", 512, "low" },
        .{ .responses, "{\"reasoning_budget_tokens\":512}", 512, "low" },
    };
    for ([_]OffDefault{ .think_flag, .table }) |how| {
        setOffDefault(how, .qwen4_exp);
        for (cases) |c| {
            const t = try surfaceThinking(a, c[0], &config, c[1]);
            try testing.expect(t.enable);
            try testing.expectEqual(c[2], t.budget);
            const word = try reasoningEffortFromWord(c[3], false, effort_mod.servedArms("qwen4_exp").?.slice());
            try testing.expectEqualStrings(try renderHi(a, qwen38_template, true, word.sends), try renderHi(a, qwen38_template, t.enable, t.template));
        }
    }
}

test "effort off default: DeepSeek and GLM resolve and render as before" {

    try test_note.needArchs(&.{ "glm5", "dsv41" });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    defer effort_mod.publish(.deepseek_v41, .{});
    defer effort_mod.publish(.glm5_next, .{});
    const ds_template = @embedFile("fixtures/dsv41_chat_template.jinja");
    const ds = model_mod.ModelConfig{ .model_type = "deepseek_v41" };
    const ds_high = (try reasoningEffortFromWord("high", false, effort_mod.servedArms("deepseek_v41").?.slice())).sends;
    for ([_]OffDefault{ .think_flag, .table }) |how| {
        setOffDefault(how, .deepseek_v41);
        const off = try renderHi(a, ds_template, false, null);
        // An on switch with no word thinks at high, uncapped; silence and a bare `reasoning` object stay off.
        for ([_]struct { OffSurface, []const u8, bool }{
            .{ .chat, "{\"enable_thinking\":true}", true },
            .{ .anthropic, "{\"thinking\":{\"type\":\"enabled\"}}", true },
            .{ .responses, "{\"enable_thinking\":true}", true },
            .{ .chat, "{}", false },
            .{ .anthropic, "{}", false },
            .{ .responses, "{\"reasoning\":{}}", false },
        }) |c| {
            const t = try surfaceThinking(a, c[0], &ds, c[1]);
            try testing.expectEqual(c[2], t.enable);
            try testing.expectEqual(@as(i32, -1), t.budget);
            const want = if (c[2]) try renderHi(a, ds_template, true, ds_high) else off;
            try testing.expectEqualStrings(want, try renderHi(a, ds_template, t.enable, t.template));
        }
    }
    // GLM cannot think off: `--think off` does not bind on it, and every surface keeps its default.
    const glm_template = @embedFile("fixtures/glm53_chat_template.jinja");
    const glm = model_mod.ModelConfig{ .model_type = "glm5_next" };
    for ([_]OffSurface{ .chat, .anthropic, .responses }) |surface| {
        // Messages has no `enable_thinking`: its on switch is a `thinking` object.
        const on = if (surface == .anthropic) "{\"thinking\":{\"type\":\"enabled\"}}" else "{\"enable_thinking\":true}";
        for ([_][]const u8{ "{}", on }) |body| {
            effort_mod.think_effort_flag = null;
            const before = try surfaceThinking(a, surface, &glm, body);
            effort_mod.think_effort_flag = .off;
            const t = try surfaceThinking(a, surface, &glm, body);
            try testing.expect(t.enable);
            try testing.expectEqual(before.budget, t.budget);
            try testing.expectEqualStrings(try renderHi(a, glm_template, before.enable, before.template), try renderHi(a, glm_template, t.enable, t.template));
        }
    }
}

test "effort table: a request resolves against one table while a save publishes another" {
    defer effort_mod.publish(.qwen4_exp, .{});
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    effort_mod.think_effort_flag = null;
    const Flipper = struct {
        fn run(stop: *std.atomic.Value(bool)) void {
            var off = true;
            while (!stop.load(.acquire)) : (off = !off) effort_mod.publish(.qwen4_exp, .{ .default = if (off) .off else .low });
        }
    };
    effort_mod.publish(.qwen4_exp, .{ .default = .off });
    var stop = std.atomic.Value(bool).init(false);
    const flipper = try std.Thread.spawn(.{}, Flipper.run, .{&stop});
    defer flipper.join();
    defer stop.store(true, .release);
    const config = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    const root: std.json.ObjectMap = .empty;
    for (0..20_000) |_| {
        const snap = effort_mod.snapshot();
        const t = chatThinking(.{ .root = root }, null, &config, &snap);
        // Default off: off, no cap, no level. Default low: on at low, its 2048 cap.
        const whole = if (t.enable) t.budget == 2048 and t.template != null and std.mem.eql(u8, t.template.?, "low") else t.budget == -1 and t.template == null;
        try testing.expect(whole);
    }
}

test "effort table: with off unoffered, chat, Anthropic and Responses refuse a request that turns thinking off, as GLM does" {

    try test_note.needArchs(&.{ "qwen4", "glm5", "dsv41" });
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const saved = effort_mod.think_effort_flag;
    defer effort_mod.think_effort_flag = saved;
    effort_mod.think_effort_flag = null;
    defer effort_mod.publish(.qwen4_exp, .{});
    defer effort_mod.publish(.deepseek_v41, .{});
    const asks_off = [_]struct { OffSurface, []const u8 }{
        .{ .chat, "{\"enable_thinking\":false}" },
        .{ .chat, "{\"chat_template_kwargs\":{\"enable_thinking\":false}}" },
        .{ .anthropic, "{\"thinking\":{\"type\":\"disabled\"}}" },
        .{ .responses, "{\"enable_thinking\":false}" },
    };
    const thinks = [_]struct { OffSurface, []const u8 }{
        .{ .chat, "{}" },
        .{ .chat, "{\"enable_thinking\":true}" },
        .{ .anthropic, "{}" },
        .{ .responses, "{}" },
    };
    for ([_]effort_mod.Family{ .qwen4_exp, .deepseek_v41 }) |f| {
        const config = model_mod.ModelConfig{ .model_type = @tagName(f) };
        for ([_]bool{ false, true }) |offered| {
            var edits: effort_mod.Edits = .{};
            if (!offered) edits.rows[effort_mod.rowIndex(.off).?] = .{ .offered = false };
            effort_mod.publish(f, edits);
            const snap = effort_mod.snapshot();
            for (asks_off) |c| try testing.expectEqual(!offered, thinkingOffRefused(&config, &snap, (try surfaceThinking(a, c[0], &config, c[1])).enable));
            for (thinks) |c| try testing.expect(!thinkingOffRefused(&config, &snap, (try surfaceThinking(a, c[0], &config, c[1])).enable));
        }
    }
    const glm = model_mod.ModelConfig{ .model_type = "glm5_next" };
    const snap = effort_mod.snapshot();
    for (asks_off) |c| try testing.expect(thinkingOffRefused(&glm, &snap, (try surfaceThinking(a, c[0], &glm, c[1])).enable));
    for (thinks) |c| try testing.expect(!thinkingOffRefused(&glm, &snap, (try surfaceThinking(a, c[0], &glm, c[1])).enable));
}

test "effort table: a non-word effort is refused on every surface, a word or null is read as before" {
    const a = testing.allocator;
    const cases = [_]struct { EffortSpelling, []const u8, bool }{
        .{ .chat, "{\"reasoning_effort\":40}", true },
        .{ .chat, "{\"reasoning_effort\":true}", true },
        .{ .chat, "{\"reasoning_effort\":{\"level\":\"low\"}}", true },
        .{ .chat, "{\"reasoning_effort\":\"40\"}", false },
        .{ .chat, "{\"reasoning_effort\":null}", false },
        .{ .chat, "{}", false },
        .{ .responses, "{\"reasoning\":{\"effort\":40}}", true },
        .{ .responses, "{\"reasoning\":{\"effort\":\"low\"}}", false },
        .{ .responses, "{\"reasoning\":{}}", false },
        .{ .anthropic, "{\"output_config\":{\"effort\":40}}", true },
        .{ .anthropic, "{\"output_config\":{\"effort\":\"low\"}}", false },
        .{ .anthropic, "{\"output_config\":{\"effort\":null}}", false },
        // Each surface reads only its own spelling.
        .{ .anthropic, "{\"reasoning_effort\":40}", false },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, c[1], .{});
        defer parsed.deinit();
        try testing.expectEqual(c[2], nonWordEffort(parsed.value.object, c[0]) != null);
    }
}
