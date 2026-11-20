//! Reading a request: the model id out of a raw body or query, the sampling and budget fields, the cache key, the
//! content parts and the tool definitions. Pure functions of the bytes, so the host tier tests them.
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const std = @import("std");
const json_text = @import("json_text.zig");
const jsonEscape = @import("http.zig").jsonEscape;
const trunk = @import("trunk.zig");
const json_schema = @import("json_schema.zig");
const model_settings = @import("model_settings.zig");
const testing = std.testing;

/// A float body field after `generationFieldFault`: null when omitted, so the resolution chain supplies the default.
pub fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8) ?f32 {
    return switch (root.get(key) orelse return null) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// The request's `min_p` after `generationFieldFault`: omitted or null is 0, which is off.
pub fn parseMinP(root: std.json.ObjectMap) f32 {
    return parseJsonFloatOpt(root, "min_p") orelse 0;
}

/// Optional top_k body field after `generationFieldFault`; null = omitted, 0 disables top-k.
pub fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    return switch (root.get(key) orelse return null) {
        .integer => |i| std.math.cast(u32, i),
        .float => |f| if (f >= 0 and f <= std.math.maxInt(u32)) @intFromFloat(f) else null,
        else => null,
    };
}

/// Plan 05 — extract the `"model":"..."` value from a JSON request body
/// without doing a full parse. Returns a borrowed slice into `body` (valid
/// until the body buffer is freed) or null if the field is missing or
/// malformed. Used by `handleConnection` to route each POST to the right
/// LoadedModel before invoking the handler.
///
/// Cheap (single linear scan, ~tens of nanoseconds for typical bodies);
/// the per-handler full JSON parse runs immediately after and rejects any
/// truly malformed body, so robustness here is fine.
pub fn parseModelFromBody(body: []const u8) ?[]const u8 {
    var idx: usize = 0;
    while (idx < body.len) {
        const key_start = std.mem.indexOfPos(u8, body, idx, "\"model\"") orelse return null;
        // Must be JSON object key: preceded by `{` or `,` (skipping whitespace).
        var prev = key_start;
        while (prev > 0) {
            prev -= 1;
            const c = body[prev];
            if (c == ' ' or c == '\t' or c == '\n' or c == '\r') continue;
            if (c == '{' or c == ',') break;
            // Not a top-level key — keep searching after this match.
            idx = key_start + 1;
            return parseModelFromBody(body[idx..]);
        }
        // Skip past the key + the colon.
        var pos = key_start + "\"model\"".len;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t')) pos += 1;
        if (pos >= body.len or body[pos] != ':') return null;
        pos += 1;
        while (pos < body.len and (body[pos] == ' ' or body[pos] == '\t' or body[pos] == '\n' or body[pos] == '\r')) pos += 1;
        if (pos >= body.len) return null;
        // Accept "string" only — null/numbers/etc fall back to default.
        if (body[pos] != '"') return null;
        pos += 1;
        const val_start = pos;
        while (pos < body.len and body[pos] != '"') {
            if (body[pos] == '\\' and pos + 1 < body.len) pos += 1; // skip escape
            pos += 1;
        }
        if (pos >= body.len) return null;
        return body[val_start..pos];
    }
    return null;
}

/// `?model=<id>` for bodiless GETs (`/props`), percent-decoded into `buf`.
pub fn queryModel(buf: []u8, raw_path: []const u8) ?[]const u8 {
    const q = raw_path[(std.mem.indexOfScalar(u8, raw_path, '?') orelse return null) + 1 ..];
    var it = std.mem.splitScalar(u8, q, '&');
    while (it.next()) |kv| {
        if (!std.mem.startsWith(u8, kv, "model=") or kv.len == "model=".len) continue;
        const v = kv["model=".len..];
        if (v.len > buf.len) return null;
        @memcpy(buf[0..v.len], v);
        return std.Uri.percentDecodeInPlace(buf[0..v.len]);
    }
    return null;
}

/// Assistant-history reasoning field on an incoming chat message:
/// `reasoning_content` (our own SSE/vLLM field, what pi rounds-trips) with
/// `reasoning` (the vLLM request spelling some templates read first) as
/// the fallback. Non-empty strings only — an empty string would render an
/// empty <think></think> block, which is exactly the nothink signature this
/// field exists to avoid.
pub fn messageReasoningFromObj(obj: std.json.ObjectMap) ?[]const u8 {
    inline for (.{ "reasoning_content", "reasoning" }) |key| {
        if (obj.get(key)) |v| {
            if (v == .string and v.string.len > 0) return v.string;
        }
    }
    return null;
}

/// A request's `seed`; negative is the llama.cpp "random" convention and means unseeded.
pub fn parseRequestSeed(v: ?std.json.Value) ?u64 {
    const i = wholeNumber(v orelse return null) orelse return null;
    return if (i < 0) null else @as(u64, @intCast(i));
}

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
pub const MAX_TOKENS_DESC_LEN: usize = 27;

pub fn ignoreEosRequested(root: std.json.ObjectMap) bool {
    return boolField(root, .ignore_eos) orelse false;
}

/// Chat refuses `ignore_eos`: past its end of turn the model writes another turn, whose think
/// block the non-stream reply merges into the reasoning and the live stream cannot.
pub fn chatIgnoreEosRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    if (!ignoreEosRequested(root)) return null;
    return "'ignore_eos' is supported on /v1/completions only: a chat reply ends at its end of turn";
}

pub fn parseCancelRequestId(allocator: std.mem.Allocator, body: []const u8) ?u64 {
    if (body.len == 0) return null;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    const v = obj.get("id") orelse obj.get("request_id") orelse return null;
    return switch (v) {
        .integer => |i| std.math.cast(u64, i),
        .string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

/// A request body read with every number as the client spelled it (`number_string`). The readers that copy client
/// JSON into the prompt (messages, tools) read this tree, so `1.0` and `1e22` reach the template as sent; every other
/// field is read from the default parse.
pub fn parseSpelled(allocator: std.mem.Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, body, .{ .parse_numbers = false });
}

/// A number from either parse: an integer, a float, or a spelled number.
pub fn jsonNumber(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |n| std.fmt.parseFloat(f64, n) catch null,
        else => null,
    };
}

/// A chat body's `tools` (from `parseSpelled`), written by the one writer; null when it declares none. Caller frees.
pub fn chatToolsJson(allocator: std.mem.Allocator, spelled_root: std.json.ObjectMap) !?[]u8 {
    const tools = spelled_root.get("tools") orelse return null;
    if (tools != .array or tools.array.items.len == 0) return null;
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try json_text.appendValue(allocator, &buf, tools);
    return try buf.toOwnedSlice(allocator);
}

pub const JoinedText = chat_mod.JoinedText;
pub const TextJoin = chat_mod.TextJoin;

/// Every `{"type":"text"}` part of a content array, joined by `TextJoin`.
pub fn joinedTextParts(allocator: std.mem.Allocator, parts: []const std.json.Value) !JoinedText {
    var join: TextJoin = .{};
    errdefer join.deinit(allocator);
    for (parts) |part| {
        if (part != .object) continue;
        const ptype = part.object.get("type") orelse continue;
        if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
        const tv = part.object.get("text") orelse continue;
        if (tv == .string) try join.add(allocator, tv.string);
    }
    return join.finish(allocator);
}

/// Workload key for hot-cache eviction (#378): `prompt_cache_key` (OpenAI's own
/// routing field) > `metadata.user_id` (Anthropic; Claude Code sends its session
/// id) > the system prompt (OpenAI `messages[0]`, Anthropic `system`, Responses
/// `instructions`) > 0 = anonymous. Text parts hash in order like `joinedTextParts`.
pub fn requestCacheKey(root: std.json.ObjectMap) u64 {
    if (root.get("prompt_cache_key")) |v| if (v == .string and v.string.len > 0)
        return std.hash.Wyhash.hash(1, v.string);
    if (root.get("metadata")) |mv| if (mv == .object) {
        if (mv.object.get("user_id")) |v| if (v == .string and v.string.len > 0)
            return std.hash.Wyhash.hash(2, v.string);
    };
    if (root.get("messages")) |mv| if (mv == .array and mv.array.items.len > 0) {
        const first = mv.array.items[0];
        if (first == .object) if (first.object.get("role")) |r| if (r == .string and std.mem.eql(u8, chat_mod.canonicalRole(r.string), "system")) {
            if (first.object.get("content")) |c| return hashTextValue(c) orelse 0;
        };
    };
    if (root.get("system")) |v| return hashTextValue(v) orelse 0;
    if (root.get("instructions")) |v| return hashTextValue(v) orelse 0;
    return 0;
}

fn hashTextValue(v: std.json.Value) ?u64 {
    var h = std.hash.Wyhash.init(3);
    var n: usize = 0;
    switch (v) {
        .string => |t| {
            if (t.len == 0) return null;
            h.update(t);
            n = 1;
        },
        .array => |parts| for (parts.items) |part| {
            if (part != .object) continue;
            const ptype = part.object.get("type") orelse continue;
            if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
            const tv = part.object.get("text") orelse continue;
            if (tv != .string or tv.string.len == 0) continue;
            if (n > 0) h.update("\n");
            h.update(tv.string);
            n += 1;
        },
        else => {},
    }
    return if (n == 0) null else h.final();
}

pub fn nChoicesRejectReason(root: std.json.ObjectMap) ?[]const u8 {
    const v = root.get("n") orelse return null;
    const n = wholeNumber(v) orelse switch (v) {
        .null => return null,
        else => return "'n' must be the integer 1: this server returns a single choice per request",
    };
    if (n == 1) return null;
    if (n < 1) return "'n' must be a positive integer, and this server returns a single choice per request: omit 'n' or set it to 1";
    return "'n' > 1 is not supported: this server returns a single choice per request; omit 'n' or set it to 1";
}

/// Legacy completions fields this server cannot honour: a fill-in-the-middle `suffix`, and `best_of` above 1.
pub fn completionsFieldFault(root: std.json.ObjectMap) ?[]const u8 {
    if (root.get("suffix")) |v| switch (v) {
        .null => {},
        .string => |text| if (text.len > 0) return "'suffix' is not supported: this server does not fill in the middle; omit it",
        else => return "'suffix' is not supported: this server does not fill in the middle; omit it",
    };
    if (root.get("best_of")) |v| if (v != .null and wholeNumber(v) != 1) {
        return "'best_of' must be 1: this server returns one completion per request and ranks no candidates";
    };
    if (root.get("logprobs")) |v| if (v != .null) {
        const n = wholeNumber(v) orelse return "'logprobs' must be an integer from 0 to 20";
        if (n < 0 or n > MAX_TOP_LOGPROBS) return "'logprobs' must be an integer from 0 to 20";
    };
    return null;
}

fn numberOf(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| if (std.math.isFinite(f)) f else null,
        else => null,
    };
}

/// A JSON integer, or a float with no fractional part (Python's `json.dumps(7.0)`), as an i64.
pub fn wholeNumber(v: std.json.Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (std.math.isFinite(f) and f == @floor(f) and @abs(f) < 9.0e15) @intFromFloat(f) else null,
        else => null,
    };
}

/// Each surface's spelling of the token budget, in precedence order.
pub const chat_max_tokens = [_][]const u8{ "max_completion_tokens", "max_tokens" };
pub const responses_max_tokens = [_][]const u8{ "max_output_tokens", "max_tokens" };

/// The request value a token budget reads: the first of a surface's `keys` that carries a positive number, so an
/// absent, null or 0 field (the app's Auto) leaves the next in force. `generationFieldFault` refused another type.
pub fn requestedMaxTokens(root: std.json.ObjectMap, keys: []const []const u8) ?std.json.Value {
    for (keys) |k| {
        const v = root.get(k) orelse continue;
        if (wholeNumber(v)) |n| if (n > 0) return v;
    }
    return null;
}

/// The bool fields a request carries. `boolField` is the one reader and `boolFieldFault` refuses any other type by
/// name, so a surface cannot read a wrong-typed one as absent.
pub const BoolField = enum { stream, store, echo, ignore_eos, background, enable_mtp, enable_drafter, enable_batch_mtp, parallel_tool_calls, enable_thinking, continue_final_message };

/// `field` as the request set it; null when absent or null, so the surface's default applies. A value of another type
/// reads as null too, which `generationFieldFault` has refused before any surface asks.
pub fn boolField(root: std.json.ObjectMap, field: BoolField) ?bool {
    const v = root.get(@tagName(field)) orelse return null;
    return if (v == .bool) v.bool else null;
}

/// `stream_options.include_usage`: off unless the client set it true (`boolFieldFault` refuses another type).
pub fn includeUsage(root: std.json.ObjectMap) bool {
    const so = root.get("stream_options") orelse return false;
    if (so != .object) return false;
    const v = so.object.get("include_usage") orelse return false;
    return v == .bool and v.bool;
}

/// `chat_template_kwargs` keys that take a bool; `chatTemplateKwargBool` reads them.
const kwarg_bools = [_][]const u8{ "enable_thinking", "preserve_thinking" };

/// The first bool field (or `stream_options`, `chat_template_kwargs` entry) of another type, as a message naming it.
pub fn boolFieldFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    inline for (comptime std.meta.tags(BoolField)) |f| if (root.get(@tagName(f))) |v| if (v != .null and v != .bool) {
        return fieldFault(buf, "'{s}' must be true or false", .{@tagName(f)});
    };
    if (root.get("stream_options")) |so| switch (so) {
        .null => {},
        .object => |o| if (o.get("include_usage")) |v| if (v != .null and v != .bool) return "'stream_options.include_usage' must be true or false",
        else => return "'stream_options' must be an object",
    };
    if (root.get("chat_template_kwargs")) |kw| switch (kw) {
        .null => {},
        .object => |o| for (kwarg_bools) |k| if (o.get(k)) |v| if (v != .null and v != .bool) {
            return fieldFault(buf, "'chat_template_kwargs.{s}' must be true or false", .{k});
        },
        else => return "'chat_template_kwargs' must be an object",
    };
    return null;
}

/// Where a surface spells the fields `generationFieldFault` checks.
pub const GenerationKeys = struct {
    max_tokens: []const []const u8 = &chat_max_tokens,
    /// Null where the surface has no stop field.
    stop: ?[]const u8 = "stop",
    /// False where the surface's API has no `min_p`: a value there is refused rather than read.
    min_p: bool = true,
};

fn fieldFault(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch "a request field has an invalid value";
}

/// A number field: a value of another type or outside `min..max` is refused, never clamped.
fn numberFault(buf: []u8, root: std.json.ObjectMap, key: []const u8, min: f64, max: f64) ?[]const u8 {
    const v = root.get(key) orelse return null;
    if (v == .null) return null;
    const n = numberOf(v) orelse return fieldFault(buf, "'{s}' must be a number from {d} to {d}", .{ key, min, max });
    if (n < min or n > max) return fieldFault(buf, "'{s}' must be a number from {d} to {d}", .{ key, min, max });
    return null;
}

const NumberRange = struct { key: []const u8, min: f64, max: f64 };
// OpenAI's ranges; `top_k` is capped by the sampler's rank search, and 0 turns a filter off.
const sampling_ranges = [_]NumberRange{
    .{ .key = "temperature", .min = 0, .max = 2 },
    .{ .key = "top_p", .min = 0, .max = 1 },
    .{ .key = "top_k", .min = 0, .max = 1000 },
    .{ .key = "min_p", .min = 0, .max = 1 },
    .{ .key = "presence_penalty", .min = -2, .max = 2 },
    .{ .key = "frequency_penalty", .min = -2, .max = 2 },
    .{ .key = "repeat_penalty", .min = 0, .max = 10 },
    .{ .key = "repetition_penalty", .min = 0, .max = 10 },
    .{ .key = "think_penalty", .min = 0, .max = model_settings.think_penalty_max },
};

/// Samplers other servers run and this one does not: each is accepted only at the value that turns it off.
const unsupported_samplers = [_]struct { key: []const u8, off: f64 }{
    .{ .key = "typical_p", .off = 1 },
    .{ .key = "tfs_z", .off = 1 },
    .{ .key = "top_a", .off = 0 },
    .{ .key = "mirostat", .off = 0 },
    .{ .key = "dry_multiplier", .off = 0 },
    .{ .key = "xtc_probability", .off = 0 },
};

/// `reasoning_budget_tokens` and Anthropic's `thinking.budget_tokens` are 32-bit integers; -1 is no cap.
fn reasoningBudgetFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    const thinking: ?std.json.ObjectMap = if (root.get("thinking")) |t| (if (t == .object) t.object else null) else null;
    const sources = [_]struct { name: []const u8, obj: ?std.json.ObjectMap, key: []const u8 }{
        .{ .name = "reasoning_budget_tokens", .obj = root, .key = "reasoning_budget_tokens" },
        .{ .name = "thinking.budget_tokens", .obj = thinking, .key = "budget_tokens" },
    };
    for (sources) |src| {
        const v = (src.obj orelse continue).get(src.key) orelse continue;
        if (v == .null) continue;
        if (v != .integer or v.integer < std.math.minInt(i32) or v.integer > std.math.maxInt(i32)) {
            return fieldFault(buf, "'{s}' must be an integer that fits in 32 bits", .{src.name});
        }
    }
    return null;
}

pub const MAX_TOP_LOGPROBS: i64 = 20;

/// What a request asks of logprobs when it asks: `top` alternatives beside each sampled token's own logprob, 0 for
/// the sampled token alone. Null, where a request carries one, is off: `top` 0 is not.
pub const Logprobs = struct {
    top: u32,

    /// The engine's per-step count, which reads 0 as off: a request for the sampled token alone still ranks one
    /// candidate, and the reply lists `top` of them.
    pub fn engineN(self: Logprobs) u32 {
        return @max(self.top, 1);
    }
};

/// Legacy completions' integer `logprobs` (null or absent: off; the range is `completionsFieldFault`'s).
pub fn completionsLogprobs(root: std.json.ObjectMap) ?Logprobs {
    const v = root.get("logprobs") orelse return null;
    const n = wholeNumber(v) orelse return null;
    return .{ .top = @intCast(@max(n, 0)) };
}

/// Chat's `logprobs: true` with its `top_logprobs` (absent or null: none; the range is `generationFieldFault`'s).
pub fn chatLogprobs(root: std.json.ObjectMap) ?Logprobs {
    const on = root.get("logprobs") orelse return null;
    if (on != .bool or !on.bool) return null;
    const top = root.get("top_logprobs") orelse return .{ .top = 0 };
    return .{ .top = if (wholeNumber(top)) |n| @intCast(@max(n, 0)) else 0 };
}

/// The first bad value among the fields every generation surface reads the same way, as a message naming the
/// field. A field that is absent or null is the default; a value of the wrong type or outside its range is
/// refused, never clamped or read as absent. A `max_tokens` of 0 is an omission: the server fills the default.
pub fn generationFieldFault(buf: []u8, root: std.json.ObjectMap, keys: GenerationKeys) ?[]const u8 {
    for (keys.max_tokens) |k| {
        const v = root.get(k) orelse continue;
        if (v == .null) continue;
        const n = wholeNumber(v) orelse return fieldFault(buf, "'{s}' must be an integer", .{k});
        if (n < 0 or n > std.math.maxInt(u32)) return fieldFault(buf, "'{s}' must be an integer from 0 (the default) to {d}", .{ k, std.math.maxInt(u32) });
    }
    if (!keys.min_p) if (root.get("min_p")) |v| if (v != .null) {
        return "'min_p' is not a field of this API: send it to /v1/chat/completions, /v1/completions or /v1/responses";
    };
    for (sampling_ranges) |r| {
        if (numberFault(buf, root, r.key, r.min, r.max)) |m| return m;
    }
    for (unsupported_samplers) |u| if (root.get(u.key)) |v| if (v != .null and numberOf(v) != u.off) {
        return fieldFault(buf, "'{s}' is not supported: this server samples with temperature, top_k, top_p and min_p only; omit it or send {d}", .{ u.key, u.off });
    };
    if (root.get("top_logprobs")) |v| if (v != .null) {
        const n = wholeNumber(v) orelse return "'top_logprobs' must be an integer from 0 to 20";
        if (n < 0 or n > MAX_TOP_LOGPROBS) return "'top_logprobs' must be an integer from 0 to 20";
    };
    if (reasoningBudgetFault(buf, root)) |m| return m;
    if (boolFieldFault(buf, root)) |m| return m;
    if (root.get("service_tier")) |v| if (v != .null and v != .string) return "'service_tier' must be a string";
    // A negative seed is the documented "unseeded"; an integer wider than i64 is no seed at all.
    if (root.get("seed")) |v| if (v != .null and wholeNumber(v) == null) return "'seed' must be an integer that fits in 64 bits; a negative one means unseeded";
    if (keys.stop) |sk| if (root.get(sk)) |v| switch (v) {
        .null, .string => {},
        .array => |arr| for (arr.items) |item| {
            if (item != .string) return fieldFault(buf, "'{s}' must be a string or an array of strings", .{sk});
        },
        else => return fieldFault(buf, "'{s}' must be a string or an array of strings", .{sk}),
    };
    if (root.get("kv_quant")) |v| if (v != .null and trunk.KVQuantConfig.fromJsonValue(v) == null) {
        return "'kv_quant' must be one of off, 4, 6 or 8 (0 and 16 also mean off)";
    };
    return null;
}

/// Responses `tool_choice`: auto, none, required or a function (`{"type":"function","name":...}`); null is absent.
pub fn responsesToolChoiceFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    const tc = root.get("tool_choice") orelse return null;
    switch (tc) {
        .null => return null,
        .string => |s| {
            const ok = std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required");
            if (ok) return null;
            return fieldFault(buf, "'tool_choice' must be auto, none, required or a function object; got \"{s}\"", .{s[0..@min(s.len, 32)]});
        },
        .object => |o| {
            const t = o.get("type") orelse return "'tool_choice.type' is required: this server calls function tools only, so it is function";
            if (t != .string or !std.mem.eql(u8, t.string, "function")) return "'tool_choice.type' must be function: this server calls function tools only";
            const holder = if (o.get("function")) |f| (if (f == .object) f.object else o) else o;
            const name = holder.get("name") orelse return "'tool_choice.name' is required: the function the model must call";
            if (name != .string or name.string.len == 0) return "'tool_choice.name' must be the name of a declared function";
            return null;
        },
        else => return "'tool_choice' must be auto, none, required or a function object",
    }
}

/// Anthropic's `tool_choice` (an object typed auto, any, tool or none, `tool` naming a function) and each
/// `tool_result.is_error` of the messages; null is absent.
pub fn anthropicToolFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    if (root.get("tool_choice")) |tc| switch (tc) {
        .null => {},
        .object => |o| {
            const t = o.get("type") orelse return "'tool_choice.type' is required: auto, any, tool or none";
            if (t != .string) return "'tool_choice.type' must be auto, any, tool or none";
            if (std.mem.eql(u8, t.string, "tool")) {
                const name = o.get("name") orelse return "'tool_choice.name' is required when 'tool_choice.type' is tool";
                if (name != .string or name.string.len == 0) return "'tool_choice.name' must be the name of a declared tool";
            } else if (!std.mem.eql(u8, t.string, "auto") and !std.mem.eql(u8, t.string, "any") and !std.mem.eql(u8, t.string, "none")) {
                return "'tool_choice.type' must be auto, any, tool or none";
            }
            if (o.get("disable_parallel_tool_use")) |d| if (d != .null and d != .bool) {
                return "'tool_choice.disable_parallel_tool_use' must be true or false";
            };
        },
        else => return "'tool_choice' must be an object typed auto, any, tool or none",
    };
    const messages = root.get("messages") orelse return null;
    if (messages != .array) return null;
    for (messages.array.items, 0..) |m, mi| {
        if (m != .object) continue;
        const content = m.object.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array.items, 0..) |block, bi| {
            if (block != .object) continue;
            const flag = block.object.get("is_error") orelse continue;
            if (flag != .null and flag != .bool) {
                return fieldFault(buf, "'messages[{d}].content[{d}].is_error' must be true or false", .{ mi, bi });
            }
        }
    }
    return null;
}

/// Whether a `tool_result` block says its tool failed.
pub fn toolResultIsError(block: std.json.ObjectMap) bool {
    const v = block.get("is_error") orelse return false;
    return v == .bool and v.bool;
}

/// `tool_choice.disable_parallel_tool_use`: at most one call per reply.
pub fn disableParallelToolUse(root: std.json.ObjectMap) bool {
    const tc = root.get("tool_choice") orelse return false;
    if (tc != .object) return false;
    const v = tc.object.get("disable_parallel_tool_use") orelse return false;
    return v == .bool and v.bool;
}

/// Whether the request declares at least one tool: `tools` null or `[]` is no tools.
pub fn toolsDeclared(root: std.json.ObjectMap) bool {
    const t = root.get("tools") orelse return false;
    return t == .array and t.array.items.len > 0;
}

/// Where a surface carries a structured-output schema: the object holding `schema` and `strict`, named for messages.
pub const SchemaHolder = struct {
    obj: std.json.ObjectMap,
    path: []const u8,
    /// Anthropic's `output_config.format` has no `strict`: its schema is always meant to be enforced.
    implied_strict: bool = false,
};

/// `strict: true` promises the output conforms. A strict schema the grammar cannot enforce, or one that cannot be
/// enforced because `has_tools` keeps the grammar off, is a message naming why (caller frees). A non-strict one falls
/// back to prompt-only guidance.
pub fn strictSchemaFault(allocator: std.mem.Allocator, holder: SchemaHolder, has_tools: bool) ?[]u8 {
    const flag = holder.obj.get("strict");
    const explicit = flag != null and flag.? == .bool;
    const strict = if (explicit) flag.?.bool else holder.implied_strict;
    if (!strict) return null;
    if (has_tools) {
        if (!explicit) return null;
        return std.fmt.allocPrint(allocator, "{s}.strict is true, but the schema is not enforced while tools are present: remove strict or the tools", .{holder.path}) catch null;
    }
    const schema = holder.obj.get("schema") orelse return null;
    var detail: json_schema.Detail = .{};
    var parsed = json_schema.parseExplained(allocator, schema, &detail) catch |err| {
        const what = if (detail.len > 0) detail.text() else @errorName(err);
        return std.fmt.allocPrint(allocator, "{s}.schema cannot be enforced as a strict schema: {s}", .{ holder.path, what }) catch null;
    };
    defer parsed.deinit();
    const relaxed = parsed.relaxed orelse return null;
    return std.fmt.allocPrint(allocator, "{s}.schema cannot be enforced as a strict schema: {s}", .{ holder.path, relaxed }) catch null;
}

/// The first entry of a `/detokenize` list that is no id of a `vocab`-sized vocabulary, as a message naming it.
pub fn tokenIdFault(buf: []u8, items: []const std.json.Value, vocab: usize) ?[]const u8 {
    for (items, 0..) |item, i| {
        const id = wholeNumber(item) orelse -1;
        if (id < 0 or id >= vocab) return fieldFault(buf, "'tokens[{d}]' must be a token id between 0 and {d}", .{ i, vocab -| 1 });
    }
    return null;
}

test "tokenIdFault names the first entry that is no token id" {
    var buf: [128]u8 = undefined;
    var ok = try std.json.parseFromSlice(std.json.Value, testing.allocator, "[0, 5.0, 99]", .{});
    defer ok.deinit();
    try testing.expectEqual(@as(?[]const u8, null), tokenIdFault(&buf, ok.value.array.items, 100));
    for ([_][]const u8{ "[1, -1]", "[1, 100]", "[1, \"2\"]", "[2.5]", "[9999999999]" }) |body| {
        var bad = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer bad.deinit();
        try testing.expect(std.mem.indexOf(u8, tokenIdFault(&buf, bad.value.array.items, 100).?, "'tokens[") != null);
    }
}


/// A tool name the call markup can carry: no whitespace, quote, angle bracket or control byte, at most 128 long.
fn toolNameCarriable(name: []const u8) bool {
    if (name.len == 0 or name.len > 128) return false;
    for (name) |c| switch (c) {
        0...0x20, 0x7f, '<', '>', '"' => return false,
        else => {},
    };
    return true;
}

/// The first malformed entry of a `tools` array (OpenAI chat, flat Responses and Anthropic shapes alike), as a
/// message naming it. `typed` also requires an OpenAI/Responses `type` of "function" where one is given.
pub fn toolsFault(buf: []u8, tools: std.json.Value, typed: bool) ?[]const u8 {
    if (tools == .null) return null;
    if (tools != .array) return "'tools' must be an array";
    for (tools.array.items, 0..) |tool, i| {
        if (tool != .object) return fieldFault(buf, "'tools[{d}]' must be an object", .{i});
        if (typed) if (tool.object.get("type")) |t| {
            if (t != .string or !std.mem.eql(u8, t.string, "function")) return fieldFault(buf, "'tools[{d}].type' must be \"function\"", .{i});
        };
        const holder = if (tool.object.get("function")) |f| (if (f == .object) f.object else tool.object) else tool.object;
        const name = holder.get("name") orelse return fieldFault(buf, "'tools[{d}]' has no function name", .{i});
        if (name != .string or name.string.len == 0) return fieldFault(buf, "'tools[{d}]' needs a function name that is a non-empty string", .{i});
        if (!toolNameCarriable(name.string)) return fieldFault(buf, "'tools[{d}]' names the function \"{s}\", which the model cannot call: a name holds no whitespace, quote or angle bracket and is at most 128 bytes", .{ i, name.string[0..@min(name.string.len, 64)] });
        for (tools.array.items[0..i]) |prev| {
            if (prev != .object) continue;
            const ph = if (prev.object.get("function")) |f| (if (f == .object) f.object else prev.object) else prev.object;
            const pn = ph.get("name") orelse continue;
            if (pn == .string and std.mem.eql(u8, pn.string, name.string)) return fieldFault(buf, "'tools[{d}]' repeats the function name \"{s}\"", .{ i, name.string[0..@min(name.string.len, 64)] });
        }
        for ([_][]const u8{ "parameters", "input_schema" }) |pk| {
            const p = holder.get(pk) orelse continue;
            if (p != .object and p != .null) return fieldFault(buf, "'tools[{d}].{s}' must be a JSON schema object", .{ i, pk });
        }
    }
    return null;
}

/// `a, b or c`: the roles a surface lists in its refusal.
fn roleList(comptime roles: []const []const u8) []const u8 {
    comptime var out: []const u8 = "";
    inline for (roles, 0..) |r, i| out = out ++ (if (i == 0) "" else if (i == roles.len - 1) " or " else ", ") ++ r;
    return out;
}

/// The first of `items` that names a role outside `roles`, as a message naming where it sits: a request defect,
/// refused before anything renders (a template raises on a role it does not know, and a render must never fall back).
/// An item with no string role is not a message here, nor, when `typed`, one whose `type` is not "message" (a call, its
/// output): Responses items carry a type, the other surfaces' messages never do.
fn roleFault(buf: []u8, comptime list: []const u8, items: []const std.json.Value, comptime roles: []const []const u8, typed: bool) ?[]const u8 {
    for (items, 0..) |m, i| {
        if (m != .object) continue;
        if (typed) if (m.object.get("type")) |t| if (t != .string or !std.mem.eql(u8, t.string, "message")) continue;
        const role = m.object.get("role") orelse continue;
        if (role != .string) continue;
        for (roles) |r| {
            if (std.mem.eql(u8, role.string, r)) break;
        } else return fieldFault(buf, "'" ++ list ++ "[{d}].role' must be one of " ++ comptime roleList(roles) ++ "; got \"{s}\"", .{ i, role.string[0..@min(role.string.len, 32)] });
    }
    return null;
}

/// The roles /v1/messages carries: its API has no system or tool message (a `system` field and `tool_result` blocks).
const anthropic_roles = [_][]const u8{ "user", "assistant" };

/// The first message of a /v1/messages body whose role is neither user nor assistant.
pub fn anthropicFieldFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    const mv = root.get("messages") orelse return null;
    if (mv != .array) return null;
    return roleFault(buf, "messages", mv.array.items, &anthropic_roles, false);
}

/// The first Responses input message whose role is not one a chat request may spell. An item of another `type`
/// (a call, its output, reasoning) has no role to check.
pub fn responsesFieldFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    const iv = root.get("input") orelse return null;
    if (iv != .array) return null;
    return roleFault(buf, "input", iv.array.items, &chat_mod.request_roles, true);
}

/// The first bad value among the chat-completions-only fields, as a message naming the field.
pub fn chatFieldFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    if (root.get("logprobs")) |v| if (v != .null and v != .bool) return "'logprobs' must be a boolean";
    if (root.get("messages")) |mv| if (mv == .array) if (roleFault(buf, "messages", mv.array.items, &chat_mod.request_roles, false)) |m| return m;
    if (root.get("response_format")) |rf| switch (rf) {
        .null => {},
        .object => |o| {
            const t = o.get("type") orelse return "'response_format.type' is required: text, json_object or json_schema";
            const ok = t == .string and (std.mem.eql(u8, t.string, "text") or std.mem.eql(u8, t.string, "json_object") or std.mem.eql(u8, t.string, "json_schema"));
            if (!ok) return "'response_format.type' must be text, json_object or json_schema";
        },
        else => return "'response_format' must be an object",
    };
    if (root.get("tools")) |tv| if (toolsFault(buf, tv, true)) |m| return m;
    if (root.get("tool_choice")) |tc| switch (tc) {
        .null, .object => {},
        .string => |s| {
            const ok = std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required") or std.mem.eql(u8, s, "any");
            if (!ok) return "'tool_choice' must be auto, none, required or a function object";
        },
        else => return "'tool_choice' must be auto, none, required or a function object",
    };
    return null;
}

test "generationFieldFault names the field a bad value sits in" {
    const Case = struct { body: []const u8, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{}", .field = null },
        .{ .body = "{\"max_tokens\":7,\"temperature\":0,\"top_p\":1,\"top_k\":40,\"seed\":-1,\"stop\":[\"a\"],\"kv_quant\":\"off\"}", .field = null },
        .{ .body = "{\"max_tokens\":7.0,\"seed\":5.0,\"top_k\":0,\"max_completion_tokens\":null,\"kv_quant\":16}", .field = null },
        .{ .body = "{\"max_tokens\":0}", .field = null },
        .{ .body = "{\"temperature\":2,\"top_p\":0,\"top_k\":1000,\"presence_penalty\":-2,\"frequency_penalty\":2,\"repeat_penalty\":10,\"top_logprobs\":20,\"seed\":-5,\"max_output_tokens\":0}", .field = null },
        .{ .body = "{\"stop\":null,\"temperature\":null,\"top_k\":null,\"top_logprobs\":null}", .field = null },
        .{ .body = "{\"temperature\":-1}", .field = "temperature" },
        .{ .body = "{\"temperature\":100}", .field = "temperature" },
        .{ .body = "{\"top_p\":2}", .field = "top_p" },
        .{ .body = "{\"top_p\":-0.1}", .field = "top_p" },
        .{ .body = "{\"top_k\":-3}", .field = "top_k" },
        .{ .body = "{\"top_k\":1001}", .field = "top_k" },
        .{ .body = "{\"presence_penalty\":3}", .field = "presence_penalty" },
        .{ .body = "{\"frequency_penalty\":-2.5}", .field = "frequency_penalty" },
        .{ .body = "{\"repeat_penalty\":11}", .field = "repeat_penalty" },
        .{ .body = "{\"repetition_penalty\":-1}", .field = "repetition_penalty" },
        .{ .body = "{\"top_logprobs\":50}", .field = "top_logprobs" },
        .{ .body = "{\"top_logprobs\":-1}", .field = "top_logprobs" },
        .{ .body = "{\"top_logprobs\":\"5\"}", .field = "top_logprobs" },
        .{ .body = "{\"top_logprobs\":2.5}", .field = "top_logprobs" },
        .{ .body = "{\"seed\":9223372036854775808}", .field = "seed" },
        .{ .body = "{\"max_tokens\":-1}", .field = "max_tokens" },
        .{ .body = "{\"max_completion_tokens\":-5}", .field = "max_completion_tokens" },
        .{ .body = "{\"max_tokens\":4294967296}", .field = "max_tokens" },
        .{ .body = "{\"think_penalty\":-4}", .field = "think_penalty" },
        .{ .body = "{\"think_penalty\":1e9}", .field = "think_penalty" },
        .{ .body = "{\"reasoning_budget_tokens\":-1,\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":2048},\"think_penalty\":20}", .field = null },
        .{ .body = "{\"reasoning_budget_tokens\":1099511627776}", .field = "reasoning_budget_tokens" },
        .{ .body = "{\"reasoning_budget_tokens\":\"64\"}", .field = "reasoning_budget_tokens" },
        .{ .body = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":99999999999}}", .field = "thinking.budget_tokens" },
        .{ .body = "{\"max_tokens\":7.9}", .field = "max_tokens" },
        .{ .body = "{\"max_tokens\":\"7\"}", .field = "max_tokens" },
        .{ .body = "{\"max_tokens\":true}", .field = "max_tokens" },
        .{ .body = "{\"max_completion_tokens\":\"7\"}", .field = "max_completion_tokens" },
        .{ .body = "{\"temperature\":\"hot\"}", .field = "temperature" },
        .{ .body = "{\"top_p\":true}", .field = "top_p" },
        .{ .body = "{\"top_k\":\"5\"}", .field = "top_k" },
        .{ .body = "{\"seed\":\"abc\"}", .field = "seed" },
        .{ .body = "{\"seed\":1.5}", .field = "seed" },
        .{ .body = "{\"stop\":5}", .field = "stop" },
        .{ .body = "{\"stop\":[\"a\",3]}", .field = "stop" },
        .{ .body = "{\"frequency_penalty\":\"x\"}", .field = "frequency_penalty" },
        .{ .body = "{\"repeat_penalty\":\"x\"}", .field = "repeat_penalty" },
        .{ .body = "{\"kv_quant\":5}", .field = "kv_quant" },
        .{ .body = "{\"kv_quant\":\"5\"}", .field = "kv_quant" },
        .{ .body = "{\"kv_quant\":true}", .field = "kv_quant" },
        .{ .body = "{\"min_p\":0,\"service_tier\":\"auto\"}", .field = null },
        .{ .body = "{\"min_p\":1}", .field = null },
        .{ .body = "{\"min_p\":0.05}", .field = null },
        .{ .body = "{\"min_p\":-0.1}", .field = "min_p" },
        .{ .body = "{\"min_p\":1.5}", .field = "min_p" },
        .{ .body = "{\"min_p\":\"0.1\"}", .field = "min_p" },
        .{ .body = "{\"typical_p\":1,\"tfs_z\":1.0,\"top_a\":0,\"mirostat\":0,\"dry_multiplier\":0,\"xtc_probability\":0.0}", .field = null },
        .{ .body = "{\"typical_p\":null,\"mirostat\":null}", .field = null },
        .{ .body = "{\"typical_p\":0.9}", .field = "typical_p" },
        .{ .body = "{\"tfs_z\":0.95}", .field = "tfs_z" },
        .{ .body = "{\"top_a\":0.1}", .field = "top_a" },
        .{ .body = "{\"mirostat\":2}", .field = "mirostat" },
        .{ .body = "{\"dry_multiplier\":0.8}", .field = "dry_multiplier" },
        .{ .body = "{\"xtc_probability\":0.5}", .field = "xtc_probability" },
        .{ .body = "{\"mirostat\":\"off\"}", .field = "mirostat" },
        .{ .body = "{\"stream\":true,\"store\":false,\"echo\":false,\"ignore_eos\":true,\"enable_mtp\":true,\"enable_drafter\":false,\"enable_batch_mtp\":true,\"parallel_tool_calls\":false,\"enable_thinking\":true,\"service_tier\":\"auto\"}", .field = null },
        .{ .body = "{\"stream\":null,\"store\":null,\"enable_mtp\":null,\"stream_options\":null,\"chat_template_kwargs\":null,\"service_tier\":null}", .field = null },
        .{ .body = "{\"stream_options\":{\"include_usage\":true},\"chat_template_kwargs\":{\"enable_thinking\":false,\"preserve_thinking\":true,\"other\":5}}", .field = null },
        .{ .body = "{\"stream\":\"true\"}", .field = "stream" },
        .{ .body = "{\"stream\":1}", .field = "stream" },
        .{ .body = "{\"store\":\"false\"}", .field = "store" },
        .{ .body = "{\"echo\":0}", .field = "echo" },
        .{ .body = "{\"ignore_eos\":\"true\"}", .field = "ignore_eos" },
        .{ .body = "{\"enable_mtp\":\"yes\"}", .field = "enable_mtp" },
        .{ .body = "{\"enable_drafter\":1}", .field = "enable_drafter" },
        .{ .body = "{\"enable_batch_mtp\":\"off\"}", .field = "enable_batch_mtp" },
        .{ .body = "{\"parallel_tool_calls\":\"false\"}", .field = "parallel_tool_calls" },
        .{ .body = "{\"enable_thinking\":\"false\"}", .field = "enable_thinking" },
        .{ .body = "{\"stream_options\":\"all\"}", .field = "stream_options" },
        .{ .body = "{\"stream_options\":{\"include_usage\":\"true\"}}", .field = "stream_options.include_usage" },
        .{ .body = "{\"chat_template_kwargs\":5}", .field = "chat_template_kwargs" },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":\"no\"}}", .field = "chat_template_kwargs.enable_thinking" },
        .{ .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":1}}", .field = "chat_template_kwargs.preserve_thinking" },
        .{ .body = "{\"service_tier\":5}", .field = "service_tier" },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        var buf: [256]u8 = undefined;
        const got = generationFieldFault(&buf, parsed.value.object, .{});
        errdefer std.debug.print("\nbody: {s}\ngot: {?s}\n", .{ c.body, got });
        if (c.field) |f| {
            const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
            defer testing.allocator.free(quoted);
            try testing.expect(std.mem.indexOf(u8, got.?, quoted) != null);
        } else try testing.expectEqual(@as(?[]const u8, null), got);
    }
}

test "strictSchemaFault refuses a strict schema the grammar cannot enforce or tools would leave unenforced" {
    const a = testing.allocator;
    const bad = "\"schema\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"nonsense\"}}}";
    const good = "\"schema\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}";
    const Case = struct { body: []const u8, tools: bool, implied: bool = false, names: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{\"strict\":true," ++ bad ++ "}", .tools = false, .names = "type 'nonsense'" },
        .{ .body = "{\"strict\":false," ++ bad ++ "}", .tools = false, .names = null },
        .{ .body = "{" ++ bad ++ "}", .tools = false, .names = null },
        .{ .body = "{\"strict\":true," ++ good ++ "}", .tools = false, .names = null },
        .{ .body = "{\"strict\":true," ++ good ++ "}", .tools = true, .names = "not enforced while tools are present" },
        .{ .body = "{\"strict\":false," ++ good ++ "}", .tools = true, .names = null },
        .{ .body = "{" ++ good ++ "}", .tools = true, .names = null },
        .{ .body = "{" ++ bad ++ "}", .tools = false, .implied = true, .names = "type 'nonsense'" },
        .{ .body = "{" ++ good ++ "}", .tools = true, .implied = true, .names = null },
        .{ .body = "{\"strict\":false," ++ bad ++ "}", .tools = false, .implied = true, .names = null },
        // The parser owns what the grammar enforces: a construct it relaxes, or a schema it refuses, is the fault.
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"r\":{\"$ref\":\"https://example.com/r.json\"}}}}", .tools = false, .names = "$ref" },
        .{ .body = "{\"strict\":false,\"schema\":{\"type\":\"object\",\"properties\":{\"r\":{\"$ref\":\"#/$defs/none\"}}}}", .tools = false, .names = null },
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{},\"required\":[\"b\"]}}", .tools = false, .names = "'b'" },
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"string\",\"maxLength\":4294967296}}", .tools = false, .names = "'maxLength'" },
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"array\",\"items\":false}}", .tools = false, .names = null },
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"object\",\"required\":[\"b\"],\"additionalProperties\":{\"type\":\"integer\"}}}", .tools = false, .names = null },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        const got = strictSchemaFault(a, .{ .obj = parsed.value.object, .path = "text.format", .implied_strict = c.implied }, c.tools);
        defer if (got) |g| a.free(g);
        errdefer std.debug.print("\nbody: {s} tools: {}\ngot: {?s}\n", .{ c.body, c.tools, got });
        if (c.names) |n| {
            try testing.expect(std.mem.startsWith(u8, got.?, "text.format."));
            try testing.expect(std.mem.indexOf(u8, got.?, n) != null);
        } else try testing.expectEqual(@as(?[]u8, null), got);
    }
}

test "generationFieldFault reads each surface's own spelling" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"stop\":5,\"stop_sequences\":7,\"max_output_tokens\":\"1\"}", .{});
    defer parsed.deinit();
    var buf: [256]u8 = undefined;
    try testing.expectEqual(@as(?[]const u8, null), generationFieldFault(&buf, parsed.value.object, .{ .max_tokens = &.{"max_tokens"}, .stop = null }));
    try testing.expect(std.mem.indexOf(u8, generationFieldFault(&buf, parsed.value.object, .{ .max_tokens = &.{"max_output_tokens"}, .stop = null }).?, "'max_output_tokens'") != null);
    try testing.expect(std.mem.indexOf(u8, generationFieldFault(&buf, parsed.value.object, .{ .max_tokens = &.{}, .stop = "stop_sequences" }).?, "'stop_sequences'") != null);
}

test "generationFieldFault: a sampler this server does not run is refused as unsupported, and min_p only where the surface has it" {
    var buf: [256]u8 = undefined;
    var off = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"xtc_probability\":0.5}", .{});
    defer off.deinit();
    try testing.expect(std.mem.indexOf(u8, generationFieldFault(&buf, off.value.object, .{}).?, "not supported") != null);
    var min_p = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"min_p\":0.1}", .{});
    defer min_p.deinit();
    const messages: GenerationKeys = .{ .max_tokens = &.{"max_tokens"}, .stop = "stop_sequences", .min_p = false };
    try testing.expect(std.mem.indexOf(u8, generationFieldFault(&buf, min_p.value.object, messages).?, "'min_p'") != null);
    var none = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"min_p\":null}", .{});
    defer none.deinit();
    try testing.expectEqual(@as(?[]const u8, null), generationFieldFault(&buf, none.value.object, messages));
}

test "completionsFieldFault refuses suffix and a best_of above 1 by name" {
    const Case = struct { body: []const u8, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{}", .field = null },
        .{ .body = "{\"suffix\":null,\"best_of\":1}", .field = null },
        .{ .body = "{\"suffix\":\"\",\"best_of\":1.0}", .field = null },
        .{ .body = "{\"suffix\":\"}\"}", .field = "suffix" },
        .{ .body = "{\"suffix\":5}", .field = "suffix" },
        .{ .body = "{\"best_of\":3}", .field = "best_of" },
        .{ .body = "{\"best_of\":0}", .field = "best_of" },
        .{ .body = "{\"best_of\":\"2\"}", .field = "best_of" },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        const got = completionsFieldFault(parsed.value.object);
        if (c.field) |f| {
            const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
            defer testing.allocator.free(quoted);
            try testing.expect(std.mem.indexOf(u8, got.?, quoted) != null);
        } else try testing.expectEqual(@as(?[]const u8, null), got);
    }
}

test "parseMinP: an omitted or null min_p is off" {
    for ([_]struct { body: []const u8, want: f32 }{
        .{ .body = "{}", .want = 0 },
        .{ .body = "{\"min_p\":null}", .want = 0 },
        .{ .body = "{\"min_p\":0.05}", .want = 0.05 },
        .{ .body = "{\"min_p\":1}", .want = 1 },
    }) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        try testing.expectEqual(c.want, parseMinP(parsed.value.object));
    }
}

test "a surface's first positive token budget key wins, and a null or zero one leaves the next in force" {
    const Case = struct { body: []const u8, keys: []const []const u8, want: ?i64 };
    const cases = [_]Case{
        .{ .body = "{\"max_tokens\":20,\"max_completion_tokens\":7}", .keys = &chat_max_tokens, .want = 7 },
        .{ .body = "{\"max_tokens\":20,\"max_completion_tokens\":null}", .keys = &chat_max_tokens, .want = 20 },
        .{ .body = "{\"max_tokens\":5,\"max_completion_tokens\":0}", .keys = &chat_max_tokens, .want = 5 },
        .{ .body = "{\"max_tokens\":5.0}", .keys = &chat_max_tokens, .want = 5 },
        .{ .body = "{\"max_tokens\":0,\"max_completion_tokens\":null}", .keys = &chat_max_tokens, .want = null },
        .{ .body = "{}", .keys = &chat_max_tokens, .want = null },
        .{ .body = "{\"max_output_tokens\":null,\"max_tokens\":1}", .keys = &responses_max_tokens, .want = 1 },
        .{ .body = "{\"max_output_tokens\":0,\"max_tokens\":1}", .keys = &responses_max_tokens, .want = 1 },
        .{ .body = "{\"max_output_tokens\":9,\"max_tokens\":1}", .keys = &responses_max_tokens, .want = 9 },
        .{ .body = "{\"max_completion_tokens\":3,\"max_tokens\":4}", .keys = &.{"max_tokens"}, .want = 4 },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        errdefer std.debug.print("\nbody: {s}\n", .{c.body});
        const got = requestedMaxTokens(parsed.value.object, c.keys);
        try testing.expectEqual(c.want, if (got) |v| wholeNumber(v) else null);
    }
}

test "chatFieldFault refuses an unknown role, response_format type, malformed tools and tool_choice" {
    const Case = struct { body: []const u8, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"developer\"}],\"tool_choice\":\"required\"}", .field = null },
        .{ .body = "{\"messages\":[{\"content\":\"no role\"},5],\"response_format\":null,\"tools\":null}", .field = null },
        .{ .body = "{\"response_format\":{\"type\":\"json_schema\"},\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"a\"}}}", .field = null },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a.b/c:d\",\"parameters\":{\"type\":\"object\"}}},{\"name\":\"flat\",\"parameters\":null}]}", .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"wizard\",\"content\":\"hi\"}]}", .field = "messages[0].role" },
        .{ .body = "{\"response_format\":{\"type\":\"xml\"}}", .field = "response_format.type" },
        .{ .body = "{\"response_format\":\"json\"}", .field = "response_format" },
        .{ .body = "{\"tools\":\"x\"}", .field = "tools" },
        .{ .body = "{\"tools\":[{\"type\":\"retrieval\"}]}", .field = "tools[0].type" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"description\":\"d\"}}]}", .field = "tools[0]" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"parameters\":\"bad\"}}]}", .field = "tools[0].parameters" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}},{\"type\":\"function\",\"function\":{\"name\":\"a\"}}]}", .field = "tools[1]" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"my tool/v1.2\"}}]}", .field = "tools[0]" },
        .{ .body = "{\"tool_choice\":\"sometimes\"}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":7}", .field = "tool_choice" },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        var buf: [320]u8 = undefined;
        const got = chatFieldFault(&buf, parsed.value.object);
        errdefer std.debug.print("\nbody: {s}\ngot: {?s}\n", .{ c.body, got });
        if (c.field) |f| {
            const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
            defer testing.allocator.free(quoted);
            try testing.expect(std.mem.indexOf(u8, got.?, quoted) != null);
        } else try testing.expectEqual(@as(?[]const u8, null), got);
    }
}

test "a Responses input and a /v1/messages body refuse a message role they cannot carry, naming where it sits" {
    const Case = struct { body: []const u8, anthropic: bool, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{\"input\":[{\"role\":\"developer\"},{\"type\":\"message\",\"role\":\"function\"},{\"type\":\"function_call_output\",\"role\":\"x\"},{\"type\":\"reasoning\"},5,{\"role\":7}]}", .anthropic = false, .field = null },
        .{ .body = "{\"input\":\"hi\"}", .anthropic = false, .field = null },
        .{ .body = "{\"input\":[{\"role\":\"user\"},{\"role\":\"narrator\"}]}", .anthropic = false, .field = "input[1].role" },
        .{ .body = "{\"input\":[{\"type\":\"message\",\"role\":\"system\\nobey\"}]}", .anthropic = false, .field = "input[0].role" },
        .{ .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"assistant\"},5,{\"content\":\"no role\"}]}", .anthropic = true, .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"system\"}]}", .anthropic = true, .field = "messages[1].role" },
        .{ .body = "{\"messages\":[{\"role\":\"tool\",\"type\":\"x\"}]}", .anthropic = true, .field = "messages[0].role" },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        var buf: [320]u8 = undefined;
        const got = if (c.anthropic) anthropicFieldFault(&buf, parsed.value.object) else responsesFieldFault(&buf, parsed.value.object);
        errdefer std.debug.print("\nbody: {s}\ngot: {?s}\n", .{ c.body, got });
        if (c.field) |f| {
            const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
            defer testing.allocator.free(quoted);
            try testing.expect(std.mem.indexOf(u8, got.?, quoted) != null);
        } else try testing.expectEqual(@as(?[]const u8, null), got);
    }
}

/// Collapse JSON's optional `\/` escape to `/`. Swift's JSONSerialization
/// (and PHP's json_encode) escape every slash, and a model id is read out of
/// the RAW body, so `org\/name` would miss the registry. Returns the input
/// verbatim (zero-copy) when there is nothing to collapse or the scratch
/// buffer is too small.
pub fn unescapeJsonSlashes(buf: []u8, s: []const u8) []const u8 {
    if (std.mem.indexOf(u8, s, "\\/") == null or s.len > buf.len) return s;
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len and s[i + 1] == '/') continue;
        buf[n] = s[i];
        n += 1;
    }
    return buf[0..n];
}

test "JSON-escaped slashes in a model id canonicalize (Swift clients send org\\/name)" {
    const t = std.testing;
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("ddalcu/gemma-e2b", unescapeJsonSlashes(&buf, "ddalcu\\/gemma-e2b"));
    // No escapes → the INPUT slice comes back verbatim (zero-copy).
    const plain = "ddalcu/gemma-e2b";
    try t.expect(unescapeJsonSlashes(&buf, plain).ptr == plain.ptr);
    // Only the two-byte sequence `\/` collapses; other backslashes survive.
    try t.expectEqualStrings("a\\b/c", unescapeJsonSlashes(&buf, "a\\b\\/c"));
    // Oversized input degrades to verbatim rather than truncating.
    var tiny: [4]u8 = undefined;
    try t.expectEqualStrings("x\\/y", unescapeJsonSlashes(tiny[0..2], "x\\/y"));
}

/// Build a llama.cpp-style `timings` JSON object (no surrounding key) from
/// raw nanosecond counts and token totals. Caller frees. Returns an empty
/// string when `prefill_ns`, `decode_ns`, AND `tokenize_ns` are all zero
/// (legacy paths that don't measure timing) so the field can be omitted.
///
/// `tokenize_ns` is the wall-clock cost of the synchronous
/// `renderChatTemplate + tokenizer.encode` step on the request thread.
/// Phase 4 #1 of `performance-plan.md` calls for instrumentation-first:
/// before we move tokenize onto a worker thread, we need numbers per
/// engine / prompt length. Pass 0 for legacy paths that don't measure it
/// (the field is then omitted from the JSON; existing callers stay shape-
/// compatible while the new ones surface the metric).
///
/// Iteration 2: chat-template render+tokenize wrapper that consults a
/// per-LoadedModel LRU cache before calling the underlying engine's
/// encoder. On hit the returned slice is a fresh allocation owned by the
/// caller (drop-in replacement for the engine encoders). On miss it
/// stores a copy of the result in the cache.
/// A bool the client set in `chat_template_kwargs` (the vLLM/SGLang spelling for
/// template variables); null when absent or not a bool. The one reader of that object.
pub fn chatTemplateKwargBool(root: std.json.ObjectMap, key: []const u8) ?bool {
    const kwargs = root.get("chat_template_kwargs") orelse return null;
    if (kwargs != .object) return null;
    const v = kwargs.object.get(key) orelse return null;
    return if (v == .bool) v.bool else null;
}

/// Wave 1.A — parse the optional per-request `kv_quant` body field. Accepts
/// the string forms `"off"`, `"0"`, `"4"`, `"6"`, `"8"` and the integer forms `0`,
/// `4`, `6`, `8`. Returns null when the field is absent or unrecognized (the
/// caller falls back to the process-level `--kv-quant` default carried on
/// the scheduler). Returns `KVQuantConfig.dense` for "off"/0.
pub fn parseKvQuantOverride(root: std.json.ObjectMap) ?trunk.KVQuantConfig {
    const v = root.get("kv_quant") orelse return null;
    const kq = trunk.KVQuantConfig.fromJsonValue(v) orelse return null;
    switch (kq.scheme) {
        .off => log.info("  kv-quant override: off (per-request)\n", .{}),
        .affine => log.info("  kv-quant override: affine {d}-bit (per-request)\n", .{kq.bits}),
    }
    return kq;
}

/// Convert Anthropic tools format to OpenAI tools format for chat template compatibility.
pub fn buildOpenAIToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (tools_array.items) |tool_val| {
        if (tool_val != .object) continue;
        if (buf.items.len > 1) try buf.append(allocator, ',');
        const tool = tool_val.object;
        const name = if (tool.get("name")) |v| (if (v == .string) v.string else "") else "";
        const desc = if (tool.get("description")) |v| (if (v == .string) v.string else "") else "";
        const esc_n = try jsonEscape(allocator, name);
        defer allocator.free(esc_n);
        const esc_d = try jsonEscape(allocator, desc);
        defer allocator.free(esc_d);
        try buf.appendSlice(allocator, "{\"type\":\"function\",\"function\":{\"name\":");
        try buf.appendSlice(allocator, esc_n);
        try buf.appendSlice(allocator, ",\"description\":");
        try buf.appendSlice(allocator, esc_d);
        try buf.appendSlice(allocator, ",\"parameters\":");
        if (tool.get("input_schema")) |schema_val| {
            try json_text.appendValue(allocator, &buf, schema_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

test "content-array text parts JOIN in order — plan-mode's [prompt, reminder] shape (issue #195)" {
    // opencode plan mode appends its system-reminder as a SECOND text part on
    // the user message. Last-wins parsing kept only the reminder, so the model
    // saw no user prompt at all.
    const body =
        \\[{"type":"text","text":"analyze project"},
        \\ {"type":"image_url","image_url":{"url":"data:x"}},
        \\ {"type":"text","text":"<system-reminder>plan mode</system-reminder>"}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    defer if (j.owned) testing.allocator.free(j.text);
    try testing.expect(j.owned);
    try testing.expectEqualStrings("analyze project\n<system-reminder>plan mode</system-reminder>", j.text);
}

test "joinedTextParts: single text part borrows; empty and non-text parts ignored" {
    const body =
        \\[{"type":"text","text":""}, {"type":"input_audio"}, {"type":"text","text":"hello"}, {"nope":1}]
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const j = try joinedTextParts(testing.allocator, parsed.value.array.items);
    try testing.expect(!j.owned);
    try testing.expectEqualStrings("hello", j.text);

    const none = try joinedTextParts(testing.allocator, &.{});
    try testing.expect(!none.owned);
    try testing.expectEqualStrings("", none.text);
}

fn cacheKeyOf(body: []const u8) !u64 {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    return requestCacheKey(parsed.value.object);
}

test "canonicalRole: developer is the system turn, a function result a tool's; every other role is itself" {
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("developer"));
    try testing.expectEqualStrings("tool", chat_mod.canonicalRole("function"));
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("system"));
    try testing.expectEqualStrings("tool", chat_mod.canonicalRole("tool"));
    try testing.expectEqualStrings("assistant", chat_mod.canonicalRole("assistant"));
}

test "requestCacheKey: prompt_cache_key > metadata.user_id > system prompt > anonymous" {
    const sys_str = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are S."},{"role":"user","content":"hi"}]}
    );
    const sys_parts = try cacheKeyOf(
        \\{"messages":[{"role":"system","content":[{"type":"text","text":"You are S."}]}]}
    );
    const anth = try cacheKeyOf(
        \\{"system":[{"type":"text","text":"You are S."}],"messages":[{"role":"user","content":"hi"}]}
    );
    const resp = try cacheKeyOf(
        \\{"instructions":"You are S.","input":"hi"}
    );
    try testing.expect(sys_str != 0);
    try testing.expectEqual(sys_str, sys_parts);
    try testing.expectEqual(sys_str, anth);
    try testing.expectEqual(sys_str, resp);
    // OpenAI's reasoning-model spelling of the system turn (pi sends it).
    try testing.expectEqual(sys_str, try cacheKeyOf(
        \\{"messages":[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]}
    ));
    try testing.expect(sys_str != try cacheKeyOf(
        \\{"messages":[{"role":"system","content":"You are T."}]}
    ));

    const uid = try cacheKeyOf(
        \\{"metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(uid != 0 and uid != sys_str);
    const pck = try cacheKeyOf(
        \\{"prompt_cache_key":"batch","metadata":{"user_id":"sess-1"},"messages":[{"role":"system","content":"You are S."}]}
    );
    try testing.expect(pck != 0 and pck != uid);
    try testing.expectEqual(pck, try cacheKeyOf(
        \\{"prompt_cache_key":"batch","prompt":"doc"}
    ));

    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"messages":[{"role":"user","content":"hi"}]}
    ));
    try testing.expectEqual(@as(u64, 0), try cacheKeyOf(
        \\{"prompt_cache_key":"","prompt":"doc"}
    ));
}

test "nChoicesRejectReason: n>1 earns an honest 400, single-choice spellings pass" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { body: []const u8, rejected: bool }{
        // Absent, explicit null, and the two spellings of "one choice" pass.
        .{ .body = "{}", .rejected = false },
        .{ .body = "{\"n\":null}", .rejected = false },
        .{ .body = "{\"n\":1}", .rejected = false },
        .{ .body = "{\"n\":1.0}", .rejected = false },
        // Anything else is a request for multi-choice (or garbage) — reject
        // instead of the silent one-choice no-op.
        .{ .body = "{\"n\":2}", .rejected = true },
        .{ .body = "{\"n\":0}", .rejected = true },
        .{ .body = "{\"n\":\"2\"}", .rejected = true },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const reason = nChoicesRejectReason(parsed.value.object);
        try std.testing.expectEqual(case.rejected, reason != null);
    }
    // n=0 and n=-1 are not "more than one".
    for ([_][]const u8{ "{\"n\":0}", "{\"n\":-1}" }) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
        defer parsed.deinit();
        try std.testing.expect(std.mem.indexOf(u8, nChoicesRejectReason(parsed.value.object).?, "positive integer") != null);
    }
}

test "messageReasoningFromObj: reasoning_content round-trip, reasoning fallback, empty dropped" {
    const t = std.testing;
    const allocator = t.allocator;
    const cases = [_]struct { body: []const u8, want: ?[]const u8 }{
        // pi/vLLM SSE spelling (live 2026-07-29).
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"two plus two\"}", .want = "two plus two" },
        // vLLM request spelling.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning\":\"twice two\"}", .want = "twice two" },
        // reasoning_content wins when both are present.
        .{ .body = "{\"role\":\"assistant\",\"reasoning_content\":\"a\",\"reasoning\":\"b\"}", .want = "a" },
        // Empty string is NOT reasoning — it would render the empty
        // <think></think> nothink signature the field exists to avoid.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":\"\"}", .want = null },
        // Non-string shapes are ignored, never crash.
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\",\"reasoning_content\":{\"x\":1}}", .want = null },
        .{ .body = "{\"role\":\"assistant\",\"content\":\"4\"}", .want = null },
    };
    for (cases) |case| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, case.body, .{});
        defer parsed.deinit();
        const got = messageReasoningFromObj(parsed.value.object);
        if (case.want) |w| {
            try t.expectEqualStrings(w, got.?);
        } else {
            try t.expect(got == null);
        }
    }
}

test "the model id is the key, not a value spelled \"model\"" {
    try testing.expectEqualStrings("a/b", parseModelFromBody("{\"model\":\"a/b\",\"messages\":[]}").?);
    try testing.expectEqualStrings("m", parseModelFromBody("{\"role\":\"model\", \"model\" :\n \"m\"}").?);
    try testing.expect(parseModelFromBody("{\"model\":null}") == null);
    try testing.expect(parseModelFromBody("{\"messages\":[]}") == null);
}

test "?model= is percent-decoded out of a bodiless GET and a missing or empty one is null" {
    var buf: [64]u8 = undefined;
    try testing.expectEqualStrings("org/name x", queryModel(&buf, "/props?x=1&model=org%2Fname%20x").?);
    try testing.expect(queryModel(&buf, "/props") == null);
    try testing.expect(queryModel(&buf, "/props?model=") == null);
}

test "a seed is a non-negative integer, a cancel id is a number or a numeric string" {
    try testing.expectEqual(@as(?u64, 7), parseRequestSeed(.{ .integer = 7 }));
    try testing.expectEqual(@as(?u64, 7), parseRequestSeed(.{ .float = 7.0 }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(.{ .integer = -1 }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(.{ .string = "7" }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(null));
    const a = testing.allocator;
    try testing.expectEqual(@as(?u64, 12), parseCancelRequestId(a, "{\"id\":12}"));
    try testing.expectEqual(@as(?u64, 12), parseCancelRequestId(a, "{\"request_id\":\"12\"}"));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, "{\"id\":-3}"));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, ""));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, "[1]"));
}

test "chat_template_kwargs answers only a bool" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"chat_template_kwargs\":{\"enable_thinking\":false,\"n\":1}}", .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(?bool, false), chatTemplateKwargBool(parsed.value.object, "enable_thinking"));
    try testing.expectEqual(@as(?bool, null), chatTemplateKwargBool(parsed.value.object, "n"));
    try testing.expectEqual(@as(?bool, null), chatTemplateKwargBool(parsed.value.object, "absent"));
}

test "chatToolsJson reads the body's own top-level tools, whatever else names tools before them" {
    const want = "[{\"type\":\"function\",\"function\":{\"name\":\"f\",\"parameters\":{\"type\":\"object\",\"properties\":{\"x\":{\"maximum\":1e22}}}}}]";
    const bodies = [_][]const u8{
        "{\"messages\":[{\"role\":\"user\",\"content\":\"tools\"}],\"tools\":" ++ want ++ "}",
        "{\"messages\":[],\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"tools\"}},\"tools\":" ++ want ++ "}",
        "{\"messages\":[],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"schema\":{\"properties\":{\"tools\":{\"type\":\"array\"}}}}},\"tools\":" ++ want ++ "}",
        "{\"messages\":[],\"metadata\":{\"tools\":[\"x\"]},\"tools\":" ++ want ++ "}",
        "{\"messages\":[],\"\\u0074ools\":" ++ want ++ "}",
    };
    for (bodies) |body| {
        const spelled = try parseSpelled(testing.allocator, body);
        defer spelled.deinit();
        const got = (try chatToolsJson(testing.allocator, spelled.value.object)).?;
        defer testing.allocator.free(got);
        try testing.expectEqualStrings(want, got);
    }
    const none = try parseSpelled(testing.allocator, "{\"messages\":[],\"tools\":[]}");
    defer none.deinit();
    try testing.expect((try chatToolsJson(testing.allocator, none.value.object)) == null);
}

test "responsesToolChoiceFault refuses a tool_choice this server cannot honour, by name" {
    const Case = struct { body: []const u8, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{}", .field = null },
        .{ .body = "{\"tool_choice\":null}", .field = null },
        .{ .body = "{\"tool_choice\":\"auto\"}", .field = null },
        .{ .body = "{\"tool_choice\":\"none\"}", .field = null },
        .{ .body = "{\"tool_choice\":\"required\"}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}", .field = null },
        .{ .body = "{\"tool_choice\":\"bogus\"}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":\"any\"}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":5}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":true}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":[]}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":{}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":5}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"allowed_tools\",\"mode\":\"auto\"}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":7}}", .field = "tool_choice.name" },
    };
    for (cases) |c| try expectFault(responsesToolChoiceFault, c.body, c.field);
}

test "anthropicToolFault refuses a tool_choice type it does not serve and a tool_result flag that is no bool, by name" {
    const Case = struct { body: []const u8, field: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{}", .field = null },
        .{ .body = "{\"tool_choice\":null}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"auto\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"any\",\"disable_parallel_tool_use\":true}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"none\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"tool\",\"name\":\"f\",\"disable_parallel_tool_use\":false}}", .field = null },
        .{ .body = "{\"tool_choice\":\"auto\"}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":5}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":{}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"bogus\"}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"tool\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"tool\",\"name\":\"\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":\"yes\"}}", .field = "tool_choice.disable_parallel_tool_use" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":true}]}]}", .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":null}]}]}", .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":\"yes\"}]}]}", .field = "messages[0].content[0].is_error" },
    };
    for (cases) |c| try expectFault(anthropicToolFault, c.body, c.field);
}

fn expectFault(check: *const fn ([]u8, std.json.ObjectMap) ?[]const u8, body: []const u8, field: ?[]const u8) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    var buf: [256]u8 = undefined;
    const got = check(&buf, parsed.value.object);
    errdefer std.debug.print("\nbody: {s}\ngot: {?s}\n", .{ body, got });
    if (field) |f| {
        const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
        defer testing.allocator.free(quoted);
        try testing.expect(std.mem.indexOf(u8, got.?, quoted) != null);
    } else try testing.expectEqual(@as(?[]const u8, null), got);
}

test "toolResultIsError reads the flag, and disableParallelToolUse the Anthropic parallel clamp" {
    const Case = struct { body: []const u8, error_flag: bool, no_parallel: bool };
    const cases = [_]Case{
        .{ .body = "{}", .error_flag = false, .no_parallel = false },
        .{ .body = "{\"is_error\":true,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":true}}", .error_flag = true, .no_parallel = true },
        .{ .body = "{\"is_error\":false,\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":false}}", .error_flag = false, .no_parallel = false },
        .{ .body = "{\"is_error\":null,\"tool_choice\":{\"type\":\"auto\"}}", .error_flag = false, .no_parallel = false },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        try testing.expectEqual(c.error_flag, toolResultIsError(parsed.value.object));
        try testing.expectEqual(c.no_parallel, disableParallelToolUse(parsed.value.object));
    }
}

test "chat logprobs: true is on whatever top_logprobs says, 0 and absent included; only a missing true is off" {
    const cases = [_]struct { body: []const u8, want: ?u32 }{
        .{ .body = "{}", .want = null },
        .{ .body = "{\"logprobs\":false,\"top_logprobs\":3}", .want = null },
        .{ .body = "{\"top_logprobs\":3}", .want = null },
        .{ .body = "{\"logprobs\":true}", .want = 0 },
        .{ .body = "{\"logprobs\":true,\"top_logprobs\":null}", .want = 0 },
        .{ .body = "{\"logprobs\":true,\"top_logprobs\":0}", .want = 0 },
        .{ .body = "{\"logprobs\":true,\"top_logprobs\":20}", .want = 20 },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.body, .{});
        defer parsed.deinit();
        const got = chatLogprobs(parsed.value.object);
        try std.testing.expectEqual(c.want, if (got) |l| l.top else null);
        // The engine reads 0 as off: a request that is on always asks it for at least one candidate.
        if (got) |l| try std.testing.expect(l.engineN() >= 1);
    }
}

test "logprobs of the wrong type or range is refused by name on each route, null is absent" {
    var buf: [256]u8 = undefined;
    const chat = [_]struct { body: []const u8, refused: bool }{
        .{ .body = "{\"logprobs\":null}", .refused = false },
        .{ .body = "{\"logprobs\":false}", .refused = false },
        .{ .body = "{\"logprobs\":1}", .refused = true },
        .{ .body = "{\"logprobs\":\"true\"}", .refused = true },
    };
    for (chat) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(c.refused, chatFieldFault(&buf, parsed.value.object) != null);
    }
    const legacy = [_]struct { body: []const u8, top: ?u32, refused: bool }{
        .{ .body = "{}", .top = null, .refused = false },
        .{ .body = "{\"logprobs\":null}", .top = null, .refused = false },
        .{ .body = "{\"logprobs\":0}", .top = 0, .refused = false },
        .{ .body = "{\"logprobs\":20}", .top = 20, .refused = false },
        .{ .body = "{\"logprobs\":21}", .top = null, .refused = true },
        .{ .body = "{\"logprobs\":true}", .top = null, .refused = true },
        .{ .body = "{\"logprobs\":\"2\"}", .top = null, .refused = true },
    };
    for (legacy) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, c.body, .{});
        defer parsed.deinit();
        try std.testing.expectEqual(c.refused, completionsFieldFault(parsed.value.object) != null);
        if (!c.refused) try std.testing.expectEqual(c.top, if (completionsLogprobs(parsed.value.object)) |l| l.top else null);
    }
}
