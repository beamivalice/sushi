//! Reading a request: the model id out of a raw body or query, the sampling and budget fields, the cache key, the
//! content parts and the tool definitions. Pure functions of the bytes, so the host tier tests them.
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const std = @import("std");
const jsonEscape = @import("http.zig").jsonEscape;
const trunk = @import("trunk.zig");
const json_schema = @import("json_schema.zig");
const testing = std.testing;

/// parseJsonFloat variant that distinguishes "omitted / wrong type" (null)
/// from an explicit value, for fields whose default comes from the
/// resolution chain above.
pub fn parseJsonFloatOpt(root: std.json.ObjectMap, key: []const u8, min: f32, max: f32) ?f32 {
    const v = root.get(key) orelse return null;
    const raw: f32 = switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => return null,
    };
    return std.math.clamp(raw, min, max);
}

/// Optional top_k body-field parse (positive integer, capped at 1000).
/// null = omitted or unusable; explicit 0 means "disable top-k".
pub fn parseJsonTopKOpt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = root.get(key) orelse return null;
    return switch (v) {
        .integer => |i| if (i > 0) @intCast(@min(i, 1000)) else 0,
        .float => |f| if (f > 0) @intFromFloat(@min(f, 1000)) else 0,
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

/// A client-supplied JSON integer narrowed to i32 by clamping, never by `@intCast`.
pub fn clampJsonI32(i: i64) i32 {
    return @intCast(std.math.clamp(i, std.math.minInt(i32), std.math.maxInt(i32)));
}

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
pub const MAX_TOKENS_DESC_LEN: usize = 27;

pub fn ignoreEosRequested(root: std.json.ObjectMap) bool {
    const v = root.get("ignore_eos") orelse return false;
    return v == .bool and v.bool;
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
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .string => |s| std.fmt.parseInt(u64, s, 10) catch null,
        else => null,
    };
}

const JoinedText = struct { text: []const u8, owned: bool };

/// Collect every `{"type":"text"}` part of a content array into one string,
/// in order, '\n'-joined. A single (or no) text part borrows the JSON's
/// bytes (`owned=false`); 2+ parts allocate (`owned=true`, caller frees).
///
/// Multiple text parts are real traffic: opencode plan mode appends its
/// "Plan Mode - System Reminder" as a SECOND text part on the user message,
/// and last-wins here dropped the user's actual prompt — the model answered
/// "What would you like to accomplish?" to every first plan-mode message
/// (issue #195).
pub fn joinedTextParts(allocator: std.mem.Allocator, parts: []const std.json.Value) !JoinedText {
    var single: []const u8 = "";
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var n: usize = 0;
    for (parts) |part| {
        if (part != .object) continue;
        const ptype = part.object.get("type") orelse continue;
        if (ptype != .string or !std.mem.eql(u8, ptype.string, "text")) continue;
        const tv = part.object.get("text") orelse continue;
        if (tv != .string or tv.string.len == 0) continue;
        n += 1;
        if (n == 1) {
            single = tv.string;
            continue;
        }
        if (n == 2) try buf.appendSlice(allocator, single);
        try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, tv.string);
    }
    if (n <= 1) return .{ .text = single, .owned = false };
    return .{ .text = try buf.toOwnedSlice(allocator), .owned = true };
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

/// The request value a token budget reads: OpenAI's `max_completion_tokens` supersedes `max_tokens`.
pub fn requestedMaxTokens(root: std.json.ObjectMap) ?std.json.Value {
    if (root.get("max_completion_tokens")) |v| if (v != .null) return v;
    return root.get("max_tokens");
}

/// Where a surface spells the fields `generationFieldFault` checks.
pub const GenerationKeys = struct {
    max_tokens: []const []const u8 = &.{ "max_completion_tokens", "max_tokens" },
    /// Null where the surface has no stop field.
    stop: ?[]const u8 = "stop",
};

fn fieldFault(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch "a request field has an invalid value";
}

/// A number field: a value of another type is refused. A number outside the range the sampler reads is clamped
/// into it, not refused.
fn numberFault(buf: []u8, root: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = root.get(key) orelse return null;
    if (v == .null or numberOf(v) != null) return null;
    return fieldFault(buf, "'{s}' must be a number", .{key});
}

/// The first bad value among the fields every generation surface reads the same way, as a message naming the
/// field. A field that is absent or null is the default; a value of the wrong type or outside its range is
/// refused, never silently read as absent. `max_tokens` of 0 or less is the documented "rest of the window".
pub fn generationFieldFault(buf: []u8, root: std.json.ObjectMap, keys: GenerationKeys) ?[]const u8 {
    for (keys.max_tokens) |k| {
        const v = root.get(k) orelse continue;
        if (v != .null and wholeNumber(v) == null) return fieldFault(buf, "'{s}' must be an integer", .{k});
    }
    for ([_][]const u8{ "temperature", "top_p", "presence_penalty", "frequency_penalty", "repeat_penalty", "repetition_penalty", "top_k" }) |k| {
        if (numberFault(buf, root, k)) |m| return m;
    }
    // An integer wider than i64 reads as unseeded.
    if (root.get("seed")) |v| if (v != .null and v != .number_string and wholeNumber(v) == null) return "'seed' must be an integer; a negative one means unseeded";
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

/// `strict: true` promises the output conforms: a schema the grammar cannot enforce is a message naming the
/// construct (caller frees), where a non-strict one falls back to prompt-only guidance.
pub fn strictSchemaFault(allocator: std.mem.Allocator, json_schema_obj: std.json.ObjectMap) ?[]u8 {
    const strict = json_schema_obj.get("strict") orelse return null;
    if (strict != .bool or !strict.bool) return null;
    const schema = json_schema_obj.get("schema") orelse return null;
    var detail: json_schema.Detail = .{};
    var parsed = json_schema.parseExplained(allocator, schema, &detail) catch |err| {
        const what = if (detail.len > 0) detail.text() else @errorName(err);
        return std.fmt.allocPrint(allocator, "response_format.json_schema.schema cannot be enforced while strict is true: unsupported {s}", .{what}) catch null;
    };
    parsed.deinit();
    return null;
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

const chat_roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool", "function" };

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

/// The first bad value among the chat-completions-only fields, as a message naming the field.
pub fn chatFieldFault(buf: []u8, root: std.json.ObjectMap) ?[]const u8 {
    if (root.get("messages")) |mv| if (mv == .array) for (mv.array.items, 0..) |m, i| {
        if (m != .object) continue;
        const role = m.object.get("role") orelse continue;
        if (role != .string) continue;
        for (chat_roles) |r| {
            if (std.mem.eql(u8, role.string, r)) break;
        } else return fieldFault(buf, "'messages[{d}].role' must be one of system, developer, user, assistant, tool or function; got \"{s}\"", .{ i, role.string[0..@min(role.string.len, 32)] });
    };
    if (root.get("top_logprobs")) |v| if (v != .null) {
        if (wholeNumber(v) == null) return "'top_logprobs' must be an integer";
    };
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
        .{ .body = "{\"temperature\":-1,\"top_p\":2,\"top_k\":-5,\"presence_penalty\":3,\"top_logprobs\":50,\"seed\":9223372036854775808}", .field = null },
        .{ .body = "{\"max_tokens\":-1}", .field = null },
        .{ .body = "{\"stop\":null,\"temperature\":null,\"top_k\":-1}", .field = null },
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

test "strictSchemaFault refuses only a strict schema the grammar cannot enforce, naming the construct" {
    const a = testing.allocator;
    const bad = "\"schema\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"nonsense\"}}}";
    const Case = struct { body: []const u8, names: ?[]const u8 };
    const cases = [_]Case{
        .{ .body = "{\"strict\":true," ++ bad ++ "}", .names = "type 'nonsense'" },
        .{ .body = "{\"strict\":false," ++ bad ++ "}", .names = null },
        .{ .body = "{" ++ bad ++ "}", .names = null },
        .{ .body = "{\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}}", .names = null },
    };
    for (cases) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, a, c.body, .{});
        defer parsed.deinit();
        const got = strictSchemaFault(a, parsed.value.object);
        defer if (got) |g| a.free(g);
        if (c.names) |n| {
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

test "a request's max_completion_tokens supersedes its max_tokens" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"max_tokens\":20,\"max_completion_tokens\":7}", .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 7), requestedMaxTokens(parsed.value.object).?.integer);
    var only = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"max_tokens\":20,\"max_completion_tokens\":null}", .{});
    defer only.deinit();
    try testing.expectEqual(@as(i64, 20), requestedMaxTokens(only.value.object).?.integer);
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
        .{ .body = "{\"top_logprobs\":\"5\"}", .field = "top_logprobs" },
        .{ .body = "{\"top_logprobs\":20,\"logprobs\":true}", .field = null },
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

/// Extract a JSON field's raw value from a JSON body string.
/// Returns the raw substring for the field value (e.g., the array or object).
pub fn extractJsonField(body: []const u8, field: []const u8) ?[]const u8 {
    // Search for "field": or "field" :
    var pos: usize = 0;
    while (pos < body.len) {
        const quote_pos = std.mem.indexOf(u8, body[pos..], "\"") orelse return null;
        const key_start = pos + quote_pos + 1;
        if (key_start + field.len >= body.len) return null;

        if (std.mem.eql(u8, body[key_start .. key_start + field.len], field) and
            body[key_start + field.len] == '"')
        {
            // Found the key, skip to colon
            var i = key_start + field.len + 1;
            while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\n' or body[i] == '\r' or body[i] == '\t')) {
                i += 1;
            }
            if (i >= body.len) return null;

            // Now extract the value - find matching bracket/brace
            const start = i;
            const open = body[start];
            const close: u8 = if (open == '[') ']' else if (open == '{') '}' else return null;
            var depth: usize = 1;
            var j = start + 1;
            var in_string = false;
            while (j < body.len and depth > 0) {
                if (body[j] == '\\' and in_string) {
                    j += 1; // skip escaped char
                } else if (body[j] == '"') {
                    in_string = !in_string;
                } else if (!in_string) {
                    if (body[j] == open) depth += 1;
                    if (body[j] == close) depth -= 1;
                }
                j += 1;
            }
            if (depth == 0) return body[start..j];
            return null;
        }
        pos = key_start;
    }
    return null;
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

/// Serialize a std.json.Value to JSON text, appending to buf.
pub fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var num_buf: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var num_buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&num_buf, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const escaped = try jsonEscape(allocator, s);
            defer allocator.free(escaped);
            try buf.appendSlice(allocator, escaped);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, idx| {
                if (idx > 0) try buf.append(allocator, ',');
                try serializeJsonValue(allocator, buf, item);
            }
            try buf.append(allocator, ']');
        },
        .object => |obj| {
            try buf.append(allocator, '{');
            var iter = obj.iterator();
            var first = true;
            while (iter.next()) |entry| {
                if (!first) try buf.append(allocator, ',');
                first = false;
                const ek = try jsonEscape(allocator, entry.key_ptr.*);
                defer allocator.free(ek);
                try buf.appendSlice(allocator, ek);
                try buf.append(allocator, ':');
                try serializeJsonValue(allocator, buf, entry.value_ptr.*);
            }
            try buf.append(allocator, '}');
        },
        .number_string => |s| try buf.appendSlice(allocator, s),
    }
}

/// Convert Anthropic tools format to OpenAI tools format for chat template compatibility.
pub fn buildOpenAIToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    for (tools_array.items, 0..) |tool_val, i| {
        if (i > 0) try buf.append(allocator, ',');
        if (tool_val != .object) continue;
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
            try serializeJsonValue(allocator, &buf, schema_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

test "extractJsonField extracts array" {
    const body =
        \\{"messages":[{"role":"user","content":"hi"}],"temperature":0.7}
    ;
    const result = extractJsonField(body, "messages").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
}

test "extractJsonField extracts nested object" {
    const body =
        \\{"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}}
    ;
    const result = extractJsonField(body, "response_format").?;
    try testing.expect(std.mem.startsWith(u8, result, "{"));
    try testing.expect(std.mem.endsWith(u8, result, "}"));
}

test "extractJsonField returns null for missing field" {
    const body = "{\"messages\":[]}";
    try testing.expect(extractJsonField(body, "tools") == null);
}

test "extractJsonField handles escaped quotes in strings" {
    const body =
        \\{"tools":[{"type":"function","function":{"name":"say_\"hello\""}}]}
    ;
    const result = extractJsonField(body, "tools").?;
    try testing.expect(std.mem.startsWith(u8, result, "["));
    try testing.expect(std.mem.endsWith(u8, result, "]"));
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

test "canonicalRole: developer is the system turn; every other role is itself" {
    try testing.expectEqualStrings("system", chat_mod.canonicalRole("developer"));
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

test "a seed is a non-negative integer, a wide client integer clamps to i32, a cancel id is a number or a numeric string" {
    try testing.expectEqual(@as(?u64, 7), parseRequestSeed(.{ .integer = 7 }));
    try testing.expectEqual(@as(?u64, 7), parseRequestSeed(.{ .float = 7.0 }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(.{ .integer = -1 }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(.{ .string = "7" }));
    try testing.expectEqual(@as(?u64, null), parseRequestSeed(null));
    try testing.expectEqual(@as(i32, std.math.maxInt(i32)), clampJsonI32(1 << 40));
    try testing.expectEqual(@as(i32, std.math.minInt(i32)), clampJsonI32(-(1 << 40)));
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
