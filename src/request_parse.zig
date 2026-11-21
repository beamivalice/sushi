//! Reading a request: the model id out of a raw body or query, the sampling and budget fields, the cache key, the
//! content parts and the tool definitions. Pure functions of the bytes, so the host tier tests them.
const chat_mod = @import("chat.zig");
const log = @import("log.zig");
const std = @import("std");
const json_text = @import("json_text.zig");
const jsonEscape = @import("http.zig").jsonEscape;
const trunk = @import("trunk.zig");
const json_schema = @import("json_schema.zig");
const testing = std.testing;

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

/// Widest string `describeMaxTokens` writes: a 10-digit budget plus
/// " (launch default)".
pub const MAX_TOKENS_DESC_LEN: usize = 27;

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

/// A request's `n`: absent, null and 1 are served; this server returns a single choice per request.
pub fn nChoicesFault(v: std.json.Value) ?[]const u8 {
    const n = wholeNumber(v) orelse switch (v) {
        .null => return null,
        else => return "'n' must be the integer 1: this server returns a single choice per request",
    };
    if (n == 1) return null;
    if (n < 1) return "'n' must be a positive integer, and this server returns a single choice per request: omit 'n' or set it to 1";
    return "'n' > 1 is not supported: this server returns a single choice per request; omit 'n' or set it to 1";
}

/// A JSON integer, or a float with no fractional part (Python's `json.dumps(7.0)`), as an i64.
pub fn wholeNumber(v: std.json.Value) ?i64 {
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (std.math.isFinite(f) and f == @floor(f) and @abs(f) < 9.0e15) @intFromFloat(f) else null,
        else => null,
    };
}

pub fn fieldFault(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch "a request field has an invalid value";
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

pub fn responsesToolChoiceShape(buf: []u8, tc: std.json.Value) ?[]const u8 {
    switch (tc) {
        .null => return null,
        .string => |s| {
            const ok = std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required");
            if (ok) return null;
            return fieldFault(buf, "'tool_choice' must be auto, none, required or a function object; got \"{s}\"", .{s[0..@min(s.len, 32)]});
        },
        .object => |o| return functionChoiceFault(o),
        else => return "'tool_choice' must be auto, none, required or a function object",
    }
}

/// Chat's `tool_choice`: a mode word or an object `functionChoiceFault` accepts.
pub fn chatToolChoiceFault(tc: std.json.Value) ?[]const u8 {
    switch (tc) {
        .null => {},
        .object => |o| return functionChoiceFault(o),
        .string => |s| {
            const ok = std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required") or std.mem.eql(u8, s, "any");
            if (!ok) return "'tool_choice' must be auto, none, required or a function object";
        },
        else => return "'tool_choice' must be auto, none, required or a function object",
    }
    return null;
}

/// An object `tool_choice` on chat and Responses: a form `parseToolChoice` honours, a mode (none, auto, any, required)
/// or a function to call (type function or tool, named flat or under `function`).
fn functionChoiceFault(o: std.json.ObjectMap) ?[]const u8 {
    const t = o.get("type") orelse return "'tool_choice.type' is required: none, auto, required or function";
    if (t != .string) return "'tool_choice.type' must be none, auto, required or function";
    for ([_][]const u8{ "none", "auto", "any", "required" }) |mode| if (std.mem.eql(u8, t.string, mode)) return null;
    if (std.mem.eql(u8, t.string, "allowed_tools")) return "'tool_choice.type' allowed_tools is not supported: send only the allowed tools in 'tools' and set 'tool_choice' to their mode";
    if (!std.mem.eql(u8, t.string, "function") and !std.mem.eql(u8, t.string, "tool")) return "'tool_choice.type' must be none, auto, required or function: this server calls function tools only";
    const holder = if (o.get("function")) |f| (if (f == .object) f.object else o) else o;
    const name = holder.get("name") orelse return "'tool_choice.name' is required: the function the model must call";
    if (name != .string or name.string.len == 0) return "'tool_choice.name' must be the name of a declared function";
    return null;
}

/// Anthropic's `tool_choice`: an object typed auto, any, tool or none, `tool` naming a function.
pub fn anthropicToolChoiceFault(tc: std.json.Value) ?[]const u8 {
    switch (tc) {
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
    }
    return null;
}

/// Each `tool_result.is_error` of a /v1/messages conversation must be a bool.
pub fn toolResultFault(buf: []u8, items: []const std.json.Value) ?[]const u8 {
    for (items, 0..) |m, mi| {
        if (m != .object) continue;
        const content = m.object.get("content") orelse continue;
        if (content != .array) continue;
        for (content.array.items, 0..) |block, bi| {
            if (block != .object) continue;
            const kind = block.object.get("type") orelse continue;
            if (kind != .string or !std.mem.eql(u8, kind.string, "tool_result")) continue;
            if (block.object.get("content")) |c| switch (c) {
                .null, .string, .array => {},
                else => return fieldFault(buf, "'messages[{d}].content[{d}].content' must be a string or an array of content parts", .{ mi, bi }),
            };
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
        // Anthropic's server tools and its own client tools (bash, text editor, computer) carry no schema this server
        // can hand a model, and it runs none of them: only a custom tool is served.
        if (!typed) if (tool.object.get("type")) |t| {
            if (t != .string) return fieldFault(buf, "'tools[{d}].type' must be a string", .{i});
            if (!std.mem.eql(u8, t.string, "custom")) return fieldFault(buf, "'tools[{d}]' is a \"{s}\" tool: this server serves custom tools only (a name, a description and an input_schema)", .{ i, t.string[0..@min(t.string.len, 64)] });
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

test "a cancel id is a number or a numeric string" {
    const a = testing.allocator;
    try testing.expectEqual(@as(?u64, 12), parseCancelRequestId(a, "{\"id\":12}"));
    try testing.expectEqual(@as(?u64, 12), parseCancelRequestId(a, "{\"request_id\":\"12\"}"));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, "{\"id\":-3}"));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, ""));
    try testing.expectEqual(@as(?u64, null), parseCancelRequestId(a, "[1]"));
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

