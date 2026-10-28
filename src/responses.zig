//! OpenAI Responses API helpers.
//!
//! Pure data-handling for `POST /v1/responses` — input-item parsing, tool-shape
//! translation, output-item JSON builders, and the in-memory response store.
//! HTTP plumbing and generation orchestration live in `server.zig`.

const std = @import("std");
const chat_mod = @import("chat.zig");

// ─── small json helpers (intentionally duplicated from server.zig to avoid
// ─── a circular import; identical behavior) ──────────────────────────────

/// Escape into a JSON string literal. Every string here is built from model
/// bytes, and a token is a BPE fragment — see the same chokepoint in server.zig.
pub fn jsonEscape(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (!std.unicode.utf8ValidateSlice(input)) {
        const clean = try chat_mod.utf8Sanitize(allocator, input);
        defer allocator.free(clean);
        return jsonEscapeValid(allocator, clean);
    }
    return jsonEscapeValid(allocator, input);
}

fn jsonEscapeValid(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '"');
    for (input) |c| {
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            0x08 => try buf.appendSlice(allocator, "\\b"),
            0x0C => try buf.appendSlice(allocator, "\\f"),
            0...0x07, 0x0B, 0x0E...0x1F => {
                var hex_buf: [8]u8 = undefined;
                const s = try std.fmt.bufPrint(&hex_buf, "\\u{x:0>4}", .{c});
                try buf.appendSlice(allocator, s);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
    return try buf.toOwnedSlice(allocator);
}

pub fn serializeJsonValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) !void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |b| try buf.appendSlice(allocator, if (b) "true" else "false"),
        .integer => |i| {
            var n: [24]u8 = undefined;
            const s = std.fmt.bufPrint(&n, "{d}", .{i}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .float => |f| {
            var n: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&n, "{d}", .{f}) catch "0";
            try buf.appendSlice(allocator, s);
        },
        .string => |s| {
            const e = try jsonEscape(allocator, s);
            defer allocator.free(e);
            try buf.appendSlice(allocator, e);
        },
        .array => |arr| {
            try buf.append(allocator, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try buf.append(allocator, ',');
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

// ─── ID generation ────────────────────────────────────────────────────────

var id_counter: std.atomic.Value(u64) = .{ .raw = 0 };

pub fn makeId(io: std.Io, allocator: std.mem.Allocator, prefix: []const u8) ![]u8 {
    const ms = std.Io.Timestamp.now(io, .real).toMilliseconds();
    const seq = id_counter.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "{s}_{d}_{x}", .{ prefix, ms, seq });
}

// ─── reasoning effort ────────────────────────────────────────────────────

pub const ReasoningConfig = struct {
    enable: bool,
    budget: i32,
    /// The client's raw `reasoning.effort` string, borrowed from the parsed
    /// request JSON — DeepSeek templates map it into the render
    /// (`chat.dsv41EffortFor`); null when the request didn't send one.
    effort: ?[]const u8 = null,
};

/// Map `reasoning.effort` → (enable_thinking, reasoning_budget).
/// "none" is an explicit off, matching the chat and Anthropic surfaces;
/// `null` / non-object → thinking disabled, budget unchanged.
pub fn parseReasoning(reasoning_val: ?std.json.Value, default_budget: i32) ReasoningConfig {
    const v = reasoning_val orelse return .{ .enable = false, .budget = default_budget };
    if (v != .object) return .{ .enable = false, .budget = default_budget };
    const effort_val = v.object.get("effort") orelse return .{ .enable = true, .budget = default_budget };
    if (effort_val != .string) return .{ .enable = true, .budget = default_budget };
    const word = effort_val.string;
    if (std.mem.eql(u8, word, "none")) return .{ .enable = false, .budget = default_budget, .effort = word };
    return .{ .enable = true, .budget = effortBudget(word, default_budget), .effort = word };
}

/// Effort → thinking-budget mapping shared by the Responses `reasoning.effort`
/// object and the chat-completions `reasoning_effort` string. Unknown efforts
/// (model-dependent spec values like "xhigh") fall back to the default budget.
pub fn effortBudget(effort: []const u8, default_budget: i32) i32 {
    if (std.mem.eql(u8, effort, "minimal")) return 1024;
    if (std.mem.eql(u8, effort, "low")) return 2048;
    if (std.mem.eql(u8, effort, "medium")) return 8192;
    return default_budget;
}

// ─── text.format → schema constraint ──────────────────────────────────────

pub const TextFormat = struct {
    /// "text" | "json_object" | "json_schema"
    kind: []const u8,
    /// When kind == "json_schema": the schema value to enforce.
    schema_value: ?std.json.Value,
};

/// Decode the `text` field of a Responses request. Accepts both shapes:
///   • flat (current OpenAI Responses spec):
///       text.format = {type, name, schema, strict}
///   • nested (chat-completions-style, used by some clients/benches):
///       text.format = {type, json_schema: {name, schema, strict}}
/// Returns text-format ("text" by default) and the schema value to enforce.
pub fn parseTextFormat(text_val: ?std.json.Value) TextFormat {
    const v = text_val orelse return .{ .kind = "text", .schema_value = null };
    if (v != .object) return .{ .kind = "text", .schema_value = null };
    const fmt_val = v.object.get("format") orelse return .{ .kind = "text", .schema_value = null };
    if (fmt_val != .object) return .{ .kind = "text", .schema_value = null };
    const t_val = fmt_val.object.get("type") orelse return .{ .kind = "text", .schema_value = null };
    const t = if (t_val == .string) t_val.string else "text";
    // Prefer flat `schema`; fall back to nested `json_schema.schema`.
    var schema = fmt_val.object.get("schema");
    if (schema == null) {
        if (fmt_val.object.get("json_schema")) |js| if (js == .object) {
            schema = js.object.get("schema");
        };
    }
    return .{ .kind = t, .schema_value = schema };
}

/// Decode a chat-completions-style `response_format` field as an alternative to
/// `text.format` on /v1/responses. Some clients/benches send their /v1/chat/
/// completions adapter body to /v1/responses; accept it as an alias to avoid
/// silently dropping the schema constraint.
///   response_format = {type, json_schema: {name, schema, strict}}
///   response_format = {type, schema, name, strict}        (flat, also accepted)
pub fn parseResponseFormatAlias(rf_val: ?std.json.Value) TextFormat {
    const v = rf_val orelse return .{ .kind = "text", .schema_value = null };
    if (v != .object) return .{ .kind = "text", .schema_value = null };
    const t_val = v.object.get("type") orelse return .{ .kind = "text", .schema_value = null };
    const t = if (t_val == .string) t_val.string else "text";
    var schema = v.object.get("schema");
    if (schema == null) {
        if (v.object.get("json_schema")) |js| if (js == .object) {
            schema = js.object.get("schema");
        };
    }
    return .{ .kind = t, .schema_value = schema };
}

pub fn inputContainsFunctionCallOutput(input_val: std.json.Value) bool {
    if (input_val != .array) return false;
    for (input_val.array.items) |item| {
        if (item != .object) continue;
        const t_val = item.object.get("type") orelse continue;
        if (t_val == .string and std.mem.eql(u8, t_val.string, "function_call_output")) return true;
    }
    return false;
}

// ─── tools: Responses (flat) → OpenAI (nested) ───────────────────────────

/// Re-shape Responses tools (`{type:"function", name, parameters, description}`)
/// into the nested OpenAI form (`{type:"function", function:{name, parameters,
/// description}}`) that `chat_mod.formatChat` expects. Returns owned JSON.
pub fn buildToolsJson(allocator: std.mem.Allocator, tools_array: std.json.Array) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    var emitted: usize = 0;
    for (tools_array.items) |tool_val| {
        if (tool_val != .object) continue;
        const tool = tool_val.object;
        // Only function tools are supported locally (web_search/file_search/computer_use are not)
        const t = if (tool.get("type")) |tv| (if (tv == .string) tv.string else "") else "";
        if (!std.mem.eql(u8, t, "function")) continue;

        if (emitted > 0) try buf.append(allocator, ',');
        emitted += 1;
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
        if (tool.get("parameters")) |params_val| {
            try serializeJsonValue(allocator, &buf, params_val);
        } else {
            try buf.appendSlice(allocator, "{}");
        }
        try buf.appendSlice(allocator, "}}");
    }
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

// ─── output-item JSON builders ────────────────────────────────────────────

pub fn appendOutputTextMessage(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, text);
    defer allocator.free(esc_text);
    try buf.appendSlice(allocator, "{\"type\":\"message\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":");
    try buf.appendSlice(allocator, esc_text);
    try buf.appendSlice(allocator, ",\"annotations\":[]}]}");
}

pub fn appendReasoningItem(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    summary_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, summary_text);
    defer allocator.free(esc_text);
    try buf.appendSlice(allocator, "{\"type\":\"reasoning\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"summary\":[{\"type\":\"summary_text\",\"text\":");
    try buf.appendSlice(allocator, esc_text);
    try buf.appendSlice(allocator, "}]}");
}

pub fn appendFunctionCallItem(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    item_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_call = try jsonEscape(allocator, call_id);
    defer allocator.free(esc_call);
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments_json);
    defer allocator.free(esc_args);
    try buf.appendSlice(allocator, "{\"type\":\"function_call\",\"id\":");
    try buf.appendSlice(allocator, esc_id);
    try buf.appendSlice(allocator, ",\"call_id\":");
    try buf.appendSlice(allocator, esc_call);
    try buf.appendSlice(allocator, ",\"name\":");
    try buf.appendSlice(allocator, esc_name);
    try buf.appendSlice(allocator, ",\"arguments\":");
    try buf.appendSlice(allocator, esc_args);
    try buf.appendSlice(allocator, ",\"status\":\"completed\"}");
}

// ─── input-item parser ────────────────────────────────────────────────────

/// Owns parsed messages and their backing buffers. Free with `deinit`.
pub const ParsedInput = struct {
    messages: std.ArrayList(chat_mod.Message),
    /// Owned heap allocations backing message fields (tool_calls slices, image
    /// pixel bufs, concatenated content, etc.). Not arena-allocated because
    /// some pieces (image pixels) are freed by other paths.
    owned_strings: std.ArrayList([]const u8),
    owned_tool_calls: std.ArrayList([]chat_mod.ToolCall),
    owned_images: std.ArrayList([]chat_mod.ImageData),
    owned_parts: std.ArrayList([]chat_mod.MediaPart) = .empty,
    allocator: std.mem.Allocator,
    /// `input_image` parts named anywhere in the input, decoded or not.
    n_images: usize = 0,
    /// The first media part refused.
    media_fault: ?MediaFault = null,

    pub fn deinit(self: *ParsedInput) void {
        for (self.owned_strings.items) |s| self.allocator.free(s);
        for (self.owned_tool_calls.items) |tcs| self.allocator.free(tcs);
        for (self.owned_images.items) |imgs| {
            for (imgs) |img| self.allocator.free(img.pixels);
            self.allocator.free(imgs);
        }
        for (self.owned_parts.items) |parts| self.allocator.free(parts);
        self.owned_strings.deinit(self.allocator);
        self.owned_tool_calls.deinit(self.allocator);
        self.owned_images.deinit(self.allocator);
        self.owned_parts.deinit(self.allocator);
        self.messages.deinit(self.allocator);
    }
};

/// A media part refused: the `input` item it sat in, what it was, and why.
pub const MediaFault = struct {
    item: usize,
    kind: enum { undecodable_image, unsupported_audio } = .undecodable_image,
    reason: []const u8 = "",
};

/// Decode a single image_url string into preprocessed pixels. Provided as a
/// callback because the actual decoder lives in `server.zig` (uses stb_image
/// + libwebp). Returns null when it appended the image, else the reason it
/// could not, recorded as `image_fault` so the surface refuses by name.
/// Appends one entry per tower call an `image_url` expands into — usually one,
/// but LFM2-VL splits a large source into tiles plus a thumbnail. Appending
/// rather than returning is what lets a single URL produce several.
pub const ImageUrlDecoder = *const fn (
    allocator: std.mem.Allocator,
    list: *std.ArrayList(chat_mod.ImageData),
    url: []const u8,
    vp: chat_mod.VisionPreproc,
) ?[]const u8;

/// How one `input` item's image parts decode: the callback, the processor,
/// and the item index a fault names.
const ImageSink = struct {
    decoder: ?ImageUrlDecoder,
    vp: chat_mod.VisionPreproc,
    item: usize,
};

/// Translate a Responses `input` value (string or array of input items) into
/// `chat_mod.Message`s. Optionally prepends `instructions` as the leading
/// `system` msg, replacing the stored history's leading one. A system or
/// developer turn later in the conversation stays where it was sent; the render
/// folds it only for a template that cannot place it.
/// `previous_messages` are deep-referenced (not copied) into the result if
/// non-null — caller must keep them alive.
pub fn parseInput(
    allocator: std.mem.Allocator,
    input_val: std.json.Value,
    instructions: ?[]const u8,
    previous_messages: ?[]const chat_mod.Message,
    image_decoder: ?ImageUrlDecoder,
    vp: chat_mod.VisionPreproc,
) !ParsedInput {
    var pi: ParsedInput = .{
        .messages = std.ArrayList(chat_mod.Message).empty,
        .owned_strings = std.ArrayList([]const u8).empty,
        .owned_tool_calls = std.ArrayList([]chat_mod.ToolCall).empty,
        .owned_images = std.ArrayList([]chat_mod.ImageData).empty,
        .allocator = allocator,
    };
    errdefer pi.deinit();

    const fresh_instructions = if (instructions) |ins| (if (ins.len > 0) ins else null) else null;
    if (fresh_instructions) |ins| {
        try pi.messages.append(allocator, .{
            .role = "system",
            .content = ins,
        });
    }

    if (previous_messages) |prev| {
        for (prev, 0..) |m, i| {
            if (fresh_instructions != null and i == 0 and std.mem.eql(u8, m.role, "system")) {
                continue;
            }
            try pi.messages.append(allocator, m);
        }
    }

    switch (input_val) {
        .string => |s| {
            try pi.messages.append(allocator, .{ .role = "user", .content = s });
        },
        .array => |arr| {
            // The assistant message a preceding reasoning item opened, still open to join.
            var reasoning_turn: ?usize = null;
            for (arr.items, 0..) |item, item_index| {
                if (item != .object) {
                    reasoning_turn = null;
                    continue;
                }
                const obj = item.object;
                const sink: ImageSink = .{ .decoder = image_decoder, .vp = vp, .item = item_index };
                const before = pi.messages.items.len;
                const t_val = obj.get("type") orelse {
                    // Bare {role, content} (some clients omit "type":"message")
                    try appendMessageItem(allocator, &pi, obj, sink);
                    joinReasoningTurn(&pi, &reasoning_turn, before);
                    continue;
                };
                if (t_val != .string) {
                    reasoning_turn = null;
                    continue;
                }
                const t = t_val.string;
                if (!std.mem.eql(u8, t, "message") and !std.mem.eql(u8, t, "function_call")) reasoning_turn = null;
                if (std.mem.eql(u8, t, "message")) {
                    try appendMessageItem(allocator, &pi, obj, sink);
                    joinReasoningTurn(&pi, &reasoning_turn, before);
                } else if (std.mem.eql(u8, t, "function_call")) {
                    try appendFunctionCallInputItem(allocator, &pi, obj);
                    joinReasoningTurn(&pi, &reasoning_turn, before);
                } else if (std.mem.eql(u8, t, "function_call_output")) {
                    try appendFunctionCallOutputItem(allocator, &pi, obj, sink);
                } else if (std.mem.eql(u8, t, "reasoning")) {
                    if (try appendReasoningInputItem(allocator, &pi, obj)) reasoning_turn = pi.messages.items.len - 1;
                } else if (std.mem.eql(u8, t, "compaction")) {
                    appendCompactionInputItem(allocator, &pi, obj) catch {};
                } else {
                    // Unknown item type — skip silently.
                    continue;
                }
            }
        },
        else => {},
    }

    return pi;
}

/// Appends the item's plaintext reasoning (`content[].reasoning_text`, else `summary[].summary_text`)
/// as an assistant turn. An item this server cannot read (encrypted body, malformed or empty) is
/// dropped: false = nothing appended.
fn appendReasoningInputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
) !bool {
    if (obj.get("encrypted_content")) |value| switch (value) {
        .null => {},
        .string => |enc| if (enc.len > 0) return false,
        else => return false,
    };
    const content: ?std.json.Value = if (obj.get("content")) |value| (if (value == .null) null else value) else null;
    const parts = content orelse obj.get("summary") orelse return false;
    if (parts != .array) return false;
    var text = std.ArrayList(u8).empty;
    defer text.deinit(allocator);
    for (parts.array.items) |part| {
        if (part != .object) return false;
        const kind = part.object.get("type") orelse return false;
        const value = part.object.get("text") orelse return false;
        if (kind != .string or value != .string) return false;
        const valid = if (content != null)
            std.mem.eql(u8, kind.string, "reasoning_text") or std.mem.eql(u8, kind.string, "text")
        else
            std.mem.eql(u8, kind.string, "summary_text");
        if (!valid) return false;
        try text.appendSlice(allocator, value.string);
    }
    if (text.items.len == 0) return false;
    const owned = try text.toOwnedSlice(allocator);
    errdefer allocator.free(owned);
    try pi.owned_strings.append(allocator, owned);
    try pi.messages.append(allocator, .{ .role = "assistant", .content = "", .reasoning_content = owned });
    return true;
}

/// Folds the assistant message just appended (index `before`) into the reasoning turn it follows, so
/// reasoning, text and parallel calls render as ONE assistant turn like chat completions. Never joins
/// across a user/tool turn or two nonempty assistant texts.
fn joinReasoningTurn(pi: *ParsedInput, turn: *?usize, before: usize) void {
    const index = turn.* orelse return;
    if (pi.messages.items.len != before + 1) {
        turn.* = null;
        return;
    }
    const next = pi.messages.items[before];
    const target = &pi.messages.items[index];
    if (!std.mem.eql(u8, next.role, "assistant") or
        (target.content.len > 0 and next.content.len > 0) or
        (target.images != null and next.images != null))
    {
        turn.* = null;
        return;
    }
    if (next.tool_calls) |calls| {
        if (target.tool_calls) |existing| {
            const joined = pi.allocator.alloc(chat_mod.ToolCall, existing.len + calls.len) catch {
                turn.* = null;
                return;
            };
            pi.owned_tool_calls.append(pi.allocator, joined) catch {
                pi.allocator.free(joined);
                turn.* = null;
                return;
            };
            @memcpy(joined[0..existing.len], existing);
            @memcpy(joined[existing.len..], calls);
            target.tool_calls = joined;
        } else target.tool_calls = calls;
    }
    if (next.content.len > 0) target.content = next.content;
    if (next.images) |images| {
        target.images = images;
        target.media_parts = next.media_parts;
    }
    pi.messages.items.len = before;
}

/// One item's content parts: text joined by '\n', and each `input_image`
/// decoded and placed at the text offset it sat at.
const InputParts = struct {
    text: []const u8 = "",
    images: ?[]const chat_mod.ImageData = null,
    media_parts: ?[]const chat_mod.MediaPart = null,
};

fn readInputParts(allocator: std.mem.Allocator, pi: *ParsedInput, parts: []const std.json.Value, sink: ImageSink) !InputParts {
    var text_parts = std.ArrayList(u8).empty;
    defer text_parts.deinit(allocator);
    var image_list = std.ArrayList(chat_mod.ImageData).empty;
    errdefer {
        for (image_list.items) |img| allocator.free(img.pixels);
        image_list.deinit(allocator);
    }
    var placed = std.ArrayList(chat_mod.MediaPart).empty;
    defer placed.deinit(allocator);
    for (parts) |part| {
        if (part != .object) continue;
        const pt_val = part.object.get("type") orelse continue;
        if (pt_val != .string) continue;
        const pt = pt_val.string;
        if (std.mem.eql(u8, pt, "input_text") or std.mem.eql(u8, pt, "text") or std.mem.eql(u8, pt, "output_text")) {
            const tx = part.object.get("text") orelse continue;
            if (tx == .string) {
                if (text_parts.items.len > 0) try text_parts.append(allocator, '\n');
                try text_parts.appendSlice(allocator, tx.string);
            }
        } else if (std.mem.eql(u8, pt, "input_image")) {
            pi.n_images += 1;
            const dec = sink.decoder orelse continue;
            if (pi.n_images > chat_mod.MAX_REQUEST_IMAGES or pi.media_fault != null) continue;
            const url: ?[]const u8 = if (part.object.get("image_url")) |url_val| switch (url_val) {
                .string => |u| u,
                .object => |io| if (io.get("url")) |u| (if (u == .string) u.string else null) else null,
                else => null,
            } else null;
            const reason = if (url) |u| dec(allocator, &image_list, u, sink.vp) else "the part carries no image_url";
            if (reason) |r| {
                pi.media_fault = .{ .item = sink.item, .reason = r };
            } else {
                try placed.append(allocator, .{ .at = text_parts.items.len, .kind = .image });
            }
        } else if (std.mem.eql(u8, pt, "input_audio")) {
            if (pi.media_fault == null) pi.media_fault = .{ .item = sink.item, .kind = .unsupported_audio };
        }
    }
    var out: InputParts = .{};
    if (text_parts.items.len > 0) {
        const owned = try allocator.dupe(u8, text_parts.items);
        try pi.owned_strings.append(allocator, owned);
        out.text = owned;
    }
    if (image_list.items.len > 0) {
        const owned = try image_list.toOwnedSlice(allocator);
        try pi.owned_images.append(allocator, owned);
        out.images = owned;
    } else {
        image_list.deinit(allocator);
    }
    if (placed.items.len > 0) {
        const owned = try placed.toOwnedSlice(allocator);
        try pi.owned_parts.append(allocator, owned);
        out.media_parts = owned;
    }
    return out;
}

fn appendMessageItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
    sink: ImageSink,
) !void {
    const role_val = obj.get("role") orelse return;
    if (role_val != .string) return;
    const role = chat_mod.canonicalRole(role_val.string);

    const content_val = obj.get("content") orelse return;
    var read: InputParts = .{};
    switch (content_val) {
        .string => |s| read.text = s,
        .array => |arr| read = try readInputParts(allocator, pi, arr.items, sink),
        else => {},
    }

    if (read.text.len == 0 and read.images == null) return;
    try pi.messages.append(allocator, .{
        .role = role,
        .content = read.text,
        .images = read.images,
        .media_parts = read.media_parts,
    });
}

fn appendFunctionCallInputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
) !void {
    const call_id = if (obj.get("call_id")) |v| (if (v == .string) v.string else "") else "";
    const name = if (obj.get("name")) |v| (if (v == .string) v.string else "") else "";
    const args = if (obj.get("arguments")) |v| (if (v == .string) v.string else "{}") else "{}";

    const tcs = try allocator.alloc(chat_mod.ToolCall, 1);
    errdefer allocator.free(tcs);
    tcs[0] = .{ .id = call_id, .name = name, .arguments = args };
    try pi.owned_tool_calls.append(allocator, tcs);
    try pi.messages.append(allocator, .{
        .role = "assistant",
        .content = "",
        .tool_calls = tcs,
    });
}

fn appendFunctionCallOutputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
    sink: ImageSink,
) !void {
    const call_id = if (obj.get("call_id")) |v| (if (v == .string) v.string else "") else "";
    // `output` is a string, or a part list that can carry `input_image`.
    var read: InputParts = .{};
    if (obj.get("output")) |v| switch (v) {
        .string => |out| read.text = out,
        .array => |arr| read = try readInputParts(allocator, pi, arr.items, sink),
        else => {},
    };
    try pi.messages.append(allocator, .{
        .role = "tool",
        .content = read.text,
        .tool_call_id = call_id,
        .images = read.images,
        .media_parts = read.media_parts,
    });
}

// ─── compaction (round-trippable opaque blob) ────────────────────────────
//
// The OpenAI Responses spec treats `encrypted_content` as opaque, provider-
// defined data. We synthesize a self-describing blob: base64-encoded JSON of
// `{v:1, msgs:[{role, content}, ...]}`. No LLM call is required — the message
// list is a faithful (lossy on tool-calls / images) snapshot of the resolved
// input that round-trips back into a fresh response.create as `input`.

/// Encode a sequence of messages as a compaction blob (base64 over JSON).
/// Caller owns the returned slice. Tool calls / images are dropped — the blob
/// only carries text-form turns. That matches the spec's "summarized" framing
/// while staying self-contained.
pub fn encodeCompactionBlob(allocator: std.mem.Allocator, messages: []const chat_mod.Message) ![]u8 {
    var json_buf = std.ArrayList(u8).empty;
    defer json_buf.deinit(allocator);
    try json_buf.appendSlice(allocator, "{\"v\":1,\"msgs\":[");
    var emitted: usize = 0;
    for (messages) |m| {
        // Skip empty turns and tool-role messages (no faithful round-trip path).
        if (m.content.len == 0) continue;
        if (emitted > 0) try json_buf.append(allocator, ',');
        emitted += 1;
        const esc_role = try jsonEscape(allocator, m.role);
        defer allocator.free(esc_role);
        const esc_content = try jsonEscape(allocator, m.content);
        defer allocator.free(esc_content);
        try json_buf.appendSlice(allocator, "{\"role\":");
        try json_buf.appendSlice(allocator, esc_role);
        try json_buf.appendSlice(allocator, ",\"content\":");
        try json_buf.appendSlice(allocator, esc_content);
        try json_buf.append(allocator, '}');
    }
    try json_buf.appendSlice(allocator, "]}");

    const enc = std.base64.standard.Encoder;
    const out_len = enc.calcSize(json_buf.items.len);
    const out = try allocator.alloc(u8, out_len);
    _ = enc.encode(out, json_buf.items);
    return out;
}

fn appendCompactionInputItem(
    allocator: std.mem.Allocator,
    pi: *ParsedInput,
    obj: std.json.ObjectMap,
) !void {
    const enc_val = obj.get("encrypted_content") orelse return;
    if (enc_val != .string) return;
    const enc_str = enc_val.string;
    if (enc_str.len == 0) return;

    const dec = std.base64.standard.Decoder;
    const dec_len = dec.calcSizeForSlice(enc_str) catch return;
    const decoded = try allocator.alloc(u8, dec_len);
    defer allocator.free(decoded);
    dec.decode(decoded, enc_str) catch return;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, decoded, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const root = parsed.value.object;
    const v_val = root.get("v") orelse return;
    if (v_val != .integer or v_val.integer != 1) return;
    const msgs_val = root.get("msgs") orelse return;
    if (msgs_val != .array) return;

    for (msgs_val.array.items) |m_val| {
        if (m_val != .object) continue;
        const role_val = m_val.object.get("role") orelse continue;
        if (role_val != .string) continue;
        const content_val = m_val.object.get("content") orelse continue;
        if (content_val != .string) continue;

        // Inner JSON values are owned by `parsed` (freed at scope end).
        // Dupe both fields so they outlive this function.
        const role_owned = try allocator.dupe(u8, role_val.string);
        try pi.owned_strings.append(allocator, role_owned);
        const content_owned = try allocator.dupe(u8, content_val.string);
        try pi.owned_strings.append(allocator, content_owned);

        try pi.messages.append(allocator, .{
            .role = role_owned,
            .content = content_owned,
        });
    }
}

// ─── in-memory response store ────────────────────────────────────────────

pub const StoredResponse = struct {
    id: []u8,
    created_at: i64,
    model: []u8,
    status: []u8, // "completed" | "failed" | "incomplete"
    /// Pre-rendered final response JSON envelope (full body returned by
    /// `GET /v1/responses/{id}`). Owned by the arena.
    body_json: []u8,
    /// Snapshot of input + assistant messages used to produce this response.
    /// Used when a later request supplies `previous_response_id` — the saved
    /// messages are concatenated in front of the new input items. Owned by
    /// the arena (including all inner []const u8 slices and tool_calls).
    history: []chat_mod.Message,

    arena: std.heap.ArenaAllocator,

    list_node: std.DoublyLinkedList.Node = .{},

    pub fn deinit(self: *StoredResponse) void {
        var arena = self.arena;
        const gpa = arena.child_allocator;
        arena.deinit();
        gpa.destroy(self);
    }
};

pub const ResponseStore = struct {
    mu: std.Io.Mutex = .init,
    map: std.StringHashMapUnmanaged(*StoredResponse) = .{},
    lru: std.DoublyLinkedList = .{},
    cap: usize,
    gpa: std.mem.Allocator,
    io: std.Io,

    pub fn init(io: std.Io, gpa: std.mem.Allocator, cap: usize) ResponseStore {
        return .{ .io = io, .gpa = gpa, .cap = cap };
    }

    pub fn deinit(self: *ResponseStore) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        var node = self.lru.first;
        while (node) |n| {
            const next = n.next;
            const sr: *StoredResponse = @fieldParentPtr("list_node", n);
            sr.deinit();
            node = next;
        }
        self.map.deinit(self.gpa);
        self.lru = .{};
    }

    /// Take ownership of `sr`. Evicts the LRU tail if at capacity.
    /// `sr.id` must already be set.
    pub fn put(self: *ResponseStore, sr: *StoredResponse) !void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);

        // If id already exists, evict the old entry first
        if (self.map.fetchRemove(sr.id)) |kv| {
            const old = kv.value;
            self.lru.remove(&old.list_node);
            old.deinit();
        }

        if (self.map.count() >= self.cap) {
            // Evict LRU tail
            if (self.lru.last) |tail_node| {
                const tail: *StoredResponse = @fieldParentPtr("list_node", tail_node);
                _ = self.map.remove(tail.id);
                self.lru.remove(tail_node);
                tail.deinit();
            }
        }

        try self.map.put(self.gpa, sr.id, sr);
        self.lru.prepend(&sr.list_node);
    }

    /// Returns a borrowed reference (do not free). Touches LRU.
    pub fn get(self: *ResponseStore, id: []const u8) ?*StoredResponse {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const sr = self.map.get(id) orelse return null;
        self.lru.remove(&sr.list_node);
        self.lru.prepend(&sr.list_node);
        return sr;
    }

    /// Returns true if removed.
    pub fn delete(self: *ResponseStore, id: []const u8) bool {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const kv = self.map.fetchRemove(id) orelse return false;
        self.lru.remove(&kv.value.list_node);
        kv.value.deinit();
        return true;
    }
};

// ─── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

test "parseReasoning maps effort levels" {
    const v_low = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"low\"}", .{});
    defer v_low.deinit();
    try testing.expectEqual(@as(i32, 2048), parseReasoning(v_low.value, -1).budget);

    // high is uncapped: the default budget rides through.
    const v_high = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"high\"}", .{});
    defer v_high.deinit();
    try testing.expectEqual(@as(i32, -1), parseReasoning(v_high.value, -1).budget);

    try testing.expectEqual(false, parseReasoning(null, -1).enable);
    try testing.expectEqual(@as(i32, -1), parseReasoning(null, -1).budget);
}

// `none` is the OpenAI/gpt-5.1 spelling of an explicit thinking-off on the chat
// and Anthropic surfaces; Responses must agree, not treat a present effort as on.
test "parseReasoning: effort none is an explicit thinking-off" {
    const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"effort\":\"none\"}", .{});
    defer v.deinit();
    const cfg = parseReasoning(v.value, -1);
    try testing.expectEqual(false, cfg.enable);
    try testing.expectEqualStrings("none", cfg.effort.?);
}

test "parseTextFormat extracts schema from flat shape" {
    const json =
        \\{"format":{"type":"json_schema","name":"x","schema":{"type":"object"}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseTextFormat(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseTextFormat default is text" {
    const tf = parseTextFormat(null);
    try testing.expectEqualStrings("text", tf.kind);
    try testing.expect(tf.schema_value == null);
}

test "parseTextFormat extracts schema from nested json_schema shape" {
    const json =
        \\{"format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"},"strict":true}}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseTextFormat(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias accepts chat-style nested shape" {
    const json =
        \\{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"},"strict":true}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseResponseFormatAlias(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias accepts flat shape too" {
    const json =
        \\{"type":"json_schema","name":"x","schema":{"type":"object"},"strict":true}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const tf = parseResponseFormatAlias(parsed.value);
    try testing.expectEqualStrings("json_schema", tf.kind);
    try testing.expect(tf.schema_value != null);
}

test "parseResponseFormatAlias default is text" {
    const tf = parseResponseFormatAlias(null);
    try testing.expectEqualStrings("text", tf.kind);
    try testing.expect(tf.schema_value == null);
}

test "inputContainsFunctionCallOutput detects tool result items" {
    const json =
        \\[
        \\  {"type":"message","role":"user","content":"hi"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"{}"}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    try testing.expect(inputContainsFunctionCallOutput(parsed.value));
    try testing.expect(!inputContainsFunctionCallOutput(.{ .string = "hi" }));
}

test "buildToolsJson nests Responses-shape into OpenAI-shape" {
    const json =
        \\[{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const out = try buildToolsJson(testing.allocator, parsed.value.array);
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "\"function\":{\"name\":\"get_weather\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "\"parameters\":{") != null);
    // Make sure top-level "type" wraps it
    try testing.expect(std.mem.startsWith(u8, out, "[{\"type\":\"function\""));
}

test "buildToolsJson skips non-function tools" {
    const json =
        \\[{"type":"web_search"},{"type":"function","name":"f","parameters":{}}]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const out = try buildToolsJson(testing.allocator, parsed.value.array);
    defer testing.allocator.free(out);
    // Only the function tool should be emitted, no leading comma
    try testing.expect(std.mem.startsWith(u8, out, "[{"));
    try testing.expect(std.mem.indexOf(u8, out, "web_search") == null);
}

test "parseInput string becomes single user message" {
    const v: std.json.Value = .{ .string = "hello" };
    var pi = try parseInput(testing.allocator, v, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 1), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("hello", pi.messages.items[0].content);
}

test "parseInput reads a developer item as the system turn" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator,
        \\[{"role":"developer","content":"You are S."},{"role":"user","content":"hi"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("You are S.", pi.messages.items[0].content);
}

fn testRejectingDecoder(_: std.mem.Allocator, _: *std.ArrayList(chat_mod.ImageData), _: []const u8, _: chat_mod.VisionPreproc) ?[]const u8 {
    return "unreadable";
}

test "parseInput records an input_image the decoder could not read" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":[{"type":"input_text","text":"what is this"},{"type":"input_image","image_url":"http://example.invalid/x.png"}]}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, null, null, testRejectingDecoder, .{});
    defer pi.deinit();
    try std.testing.expectEqual(@as(usize, 0), pi.media_fault.?.item);
    try std.testing.expectEqualStrings("unreadable", pi.media_fault.?.reason);
    try std.testing.expectEqual(@as(usize, 1), pi.messages.items.len);
}

fn testAcceptingDecoder(allocator: std.mem.Allocator, list: *std.ArrayList(chat_mod.ImageData), _: []const u8, _: chat_mod.VisionPreproc) ?[]const u8 {
    const pixels = allocator.alloc(u8, 4) catch return "oom";
    list.append(allocator, .{ .pixels = pixels, .width = 1, .height = 1 }) catch {
        allocator.free(pixels);
        return "oom";
    };
    return null;
}

test "parseInput places input_image parts wherever they appear, tool outputs included" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,AAAA"},{"type":"input_text","text":"open it"}]},
        \\ {"type":"function_call","call_id":"c1","name":"screenshot","arguments":"{}"},
        \\ {"type":"function_call_output","call_id":"c1","output":[{"type":"input_text","text":"shot:"},{"type":"input_image","image_url":{"url":"data:image/png;base64,AAAA"}}]}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, null, null, testAcceptingDecoder, .{});
    defer pi.deinit();
    try std.testing.expect(pi.media_fault == null);
    try std.testing.expectEqual(@as(usize, 2), pi.n_images);
    const msgs = pi.messages.items;
    try std.testing.expectEqual(@as(usize, 3), msgs.len);
    try std.testing.expectEqualSlices(chat_mod.MediaPart, &.{.{ .at = 0, .kind = .image }}, msgs[0].media_parts.?);
    try std.testing.expectEqualStrings("tool", msgs[2].role);
    try std.testing.expectEqualStrings("shot:", msgs[2].content);
    try std.testing.expectEqual(@as(usize, 1), msgs[2].images.?.len);
    try std.testing.expectEqualSlices(chat_mod.MediaPart, &.{.{ .at = 5, .kind = .image }}, msgs[2].media_parts.?);
}

test "parseInput names the input item of an image it could not decode" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":"hi"},
        \\ {"type":"function_call_output","call_id":"c1","output":[{"type":"input_image","image_url":"http://example.invalid/x.png"}]}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, null, null, testRejectingDecoder, .{});
    defer pi.deinit();
    try std.testing.expectEqual(@as(usize, 1), pi.media_fault.?.item);
}

test "parseInput refuses an input_audio part by name" {
    const allocator = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":[{"type":"input_text","text":"listen"},{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, null, null, testAcceptingDecoder, .{});
    defer pi.deinit();
    try std.testing.expectEqual(@as(usize, 0), pi.media_fault.?.item);
    try std.testing.expect(pi.media_fault.?.kind == .unsupported_audio);
}

test "parseInput with instructions prepends system" {
    const v: std.json.Value = .{ .string = "hi" };
    var pi = try parseInput(testing.allocator, v, "You are a pirate", null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("user", pi.messages.items[1].role);
}

test "parseInput replaces stored system when fresh instructions are provided" {
    const v: std.json.Value = .{ .string = "next" };
    const prev = [_]chat_mod.Message{
        .{ .role = "system", .content = "old instructions" },
        .{ .role = "user", .content = "first" },
        .{ .role = "assistant", .content = "answer" },
    };
    var pi = try parseInput(testing.allocator, v, "new instructions", &prev, null, .{});
    defer pi.deinit();

    try testing.expectEqual(@as(usize, 4), pi.messages.items.len);
    try testing.expectEqualStrings("system", pi.messages.items[0].role);
    try testing.expectEqualStrings("new instructions", pi.messages.items[0].content);
    try testing.expectEqualStrings("user", pi.messages.items[1].role);
    try testing.expectEqualStrings("assistant", pi.messages.items[2].role);
    try testing.expectEqualStrings("user", pi.messages.items[3].role);
    for (pi.messages.items[1..]) |m| {
        try testing.expect(!std.mem.eql(u8, m.role, "system"));
    }
}

test "parseInput keeps a mid-input developer turn where it was sent" {
    // Codex sends a mid-input `developer` turn beside `instructions`; the render
    // folds it only for a template that cannot place it.
    const allocator = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator,
        \\[{"role":"user","content":"hi"},{"role":"developer","content":"mid"},{"role":"user","content":"again"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(allocator, parsed.value, "You are S.", null, null, .{});
    defer pi.deinit();
    const want = [_][2][]const u8{ .{ "system", "You are S." }, .{ "user", "hi" }, .{ "system", "mid" }, .{ "user", "again" } };
    try testing.expectEqual(want.len, pi.messages.items.len);
    for (want, pi.messages.items) |w, m| {
        try testing.expectEqualStrings(w[0], m.role);
        try testing.expectEqualStrings(w[1], m.content);
    }
}

test "parseInput: fresh instructions replace only the leading stored system turn" {
    const v: std.json.Value = .{ .string = "next" };
    const prev = [_]chat_mod.Message{
        .{ .role = "system", .content = "old instructions" },
        .{ .role = "user", .content = "first" },
        .{ .role = "system", .content = "mid" },
        .{ .role = "assistant", .content = "answer" },
    };
    var pi = try parseInput(testing.allocator, v, "new instructions", &prev, null, .{});
    defer pi.deinit();
    const want = [_][2][]const u8{ .{ "system", "new instructions" }, .{ "user", "first" }, .{ "system", "mid" }, .{ "assistant", "answer" }, .{ "user", "next" } };
    try testing.expectEqual(want.len, pi.messages.items.len);
    for (want, pi.messages.items) |w, m| {
        try testing.expectEqualStrings(w[0], m.role);
        try testing.expectEqualStrings(w[1], m.content);
    }
}

test "parseInput function_call + function_call_output round-trip" {
    const json =
        \\[
        \\  {"type":"message","role":"user","content":"what's the weather?"},
        \\  {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"sf\"}"},
        \\  {"type":"function_call_output","call_id":"call_1","output":"sunny"}
        \\]
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 3), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("assistant", pi.messages.items[1].role);
    try testing.expect(pi.messages.items[1].tool_calls != null);
    try testing.expectEqualStrings("call_1", pi.messages.items[1].tool_calls.?[0].id);
    try testing.expectEqualStrings("tool", pi.messages.items[2].role);
    try testing.expectEqualStrings("call_1", pi.messages.items[2].tool_call_id.?);
}

test "compaction blob round-trips through encode + parseInput" {
    const msgs = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello there" },
        .{ .role = "assistant", .content = "hi back" },
    };
    const blob = try encodeCompactionBlob(testing.allocator, &msgs);
    defer testing.allocator.free(blob);

    // Build the input array with a compaction item carrying the blob.
    const input_json = try std.fmt.allocPrint(testing.allocator,
        \\[{{"type":"compaction","id":"cmp_1","encrypted_content":"{s}"}}]
    , .{blob});
    defer testing.allocator.free(input_json);

    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, input_json, .{});
    defer parsed.deinit();

    var pi = try parseInput(testing.allocator, parsed.value, null, null, null, .{});
    defer pi.deinit();

    try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
    try testing.expectEqualStrings("user", pi.messages.items[0].role);
    try testing.expectEqualStrings("hello there", pi.messages.items[0].content);
    try testing.expectEqualStrings("assistant", pi.messages.items[1].role);
    try testing.expectEqualStrings("hi back", pi.messages.items[1].content);
}

test "compaction with malformed envelope is silently skipped" {
    // Bad base64 + bogus inner JSON shouldn't crash, just produces no messages.
    const inputs = [_][]const u8{
        \\[{"type":"compaction","encrypted_content":"!!!not-base64!!!"}]
        ,
        \\[{"type":"compaction","encrypted_content":""}]
        ,
    };
    for (inputs) |body| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
        defer parsed.deinit();
        var pi = try parseInput(testing.allocator, parsed.value, null, null, null, .{});
        defer pi.deinit();
        try testing.expectEqual(@as(usize, 0), pi.messages.items.len);
    }
}

fn makeTestStored(gpa: std.mem.Allocator, id: []const u8) !*StoredResponse {
    const sr = try gpa.create(StoredResponse);
    var arena = std.heap.ArenaAllocator.init(gpa);
    sr.* = .{
        .id = try arena.allocator().dupe(u8, id),
        .created_at = 0,
        .model = try arena.allocator().dupe(u8, "m"),
        .status = try arena.allocator().dupe(u8, "completed"),
        .body_json = try arena.allocator().dupe(u8, "{}"),
        .history = &[_]chat_mod.Message{},
        .arena = arena,
    };
    return sr;
}

test "ResponseStore basic put/get/delete" {
    const gpa = testing.allocator;
    var store = ResponseStore.init(testing.io, gpa, 4);
    defer store.deinit();

    const sr = try makeTestStored(gpa, "resp_1");
    try store.put(sr);

    try testing.expect(store.get("resp_1") != null);
    try testing.expect(store.get("missing") == null);
    try testing.expectEqual(true, store.delete("resp_1"));
    try testing.expect(store.get("resp_1") == null);
}

test "ResponseStore evicts LRU at cap" {
    const gpa = testing.allocator;
    var store = ResponseStore.init(testing.io, gpa, 2);
    defer store.deinit();

    var ids: [3][]const u8 = undefined;
    inline for (0..3) |i| ids[i] = std.fmt.comptimePrint("id_{d}", .{i});
    for (ids) |id| {
        const sr = try makeTestStored(gpa, id);
        try store.put(sr);
    }
    // Cap is 2, inserted 3 — first one should be evicted
    try testing.expect(store.get("id_0") == null);
    try testing.expect(store.get("id_1") != null);
    try testing.expect(store.get("id_2") != null);
}

fn expectSameQwenPrompt(actual: []const chat_mod.Message, expected: []const chat_mod.Message) !void {
    const a = testing.allocator;
    const config = chat_mod.ChatConfig{
        .chat_template = @embedFile("fixtures/qwen38_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = a,
        .preserve_thinking = true,
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"integer\"}}}}}]";
    const got = try chat_mod.renderChatTemplate(a, actual, &config, tools, null, true, null, false);
    defer a.free(got);
    const want = try chat_mod.renderChatTemplate(a, expected, &config, tools, null, true, null, false);
    defer a.free(want);
    try testing.expectEqualStrings(want, got);
    try testing.expect(std.mem.indexOf(u8, got, "inspect both") != null);
}

test "parseInput: a replayed reasoning item renders the chat-completions prompt" {
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\[{"role":"user","content":"compare"},
        \\ {"type":"reasoning","summary":[{"type":"summary_text","text":"inspect both"}]},
        \\ {"type":"function_call","call_id":"c1","name":"lookup","arguments":"{\"id\":1}"},
        \\ {"type":"function_call","call_id":"c2","name":"lookup","arguments":"{\"id\":2}"},
        \\ {"type":"function_call_output","call_id":"c1","output":"first"},
        \\ {"type":"function_call_output","call_id":"c2","output":"second"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(a, parsed.value, null, null, null, .{});
    defer pi.deinit();
    const calls = [_]chat_mod.ToolCall{
        .{ .id = "c1", .name = "lookup", .arguments = "{\"id\":1}" },
        .{ .id = "c2", .name = "lookup", .arguments = "{\"id\":2}" },
    };
    const expected = [_]chat_mod.Message{
        .{ .role = "user", .content = "compare" },
        .{ .role = "assistant", .content = "", .reasoning_content = "inspect both", .tool_calls = &calls },
        .{ .role = "tool", .content = "first", .tool_call_id = "c1" },
        .{ .role = "tool", .content = "second", .tool_call_id = "c2" },
    };
    try testing.expectEqual(expected.len, pi.messages.items.len);
    try expectSameQwenPrompt(pi.messages.items, &expected);
}

test "parseInput: an unreadable reasoning item is dropped without an error" {
    const a = testing.allocator;
    for ([_][]const u8{
        \\{"type":"reasoning","encrypted_content":"opaque","summary":[{"type":"summary_text","text":"s"}]}
        ,
        \\{"type":"reasoning","encrypted_content":17}
        ,
        \\{"type":"reasoning","summary":"invalid"}
        ,
        \\{"type":"reasoning","content":[{"type":"unknown","text":"lost"}]}
        ,
        \\{"type":"reasoning","summary":[]}
        ,
        \\{"type":"reasoning"}
        ,
    }) |bad| {
        const json = try std.fmt.allocPrint(a, "[{{\"role\":\"user\",\"content\":\"hi\"}},{s},{{\"role\":\"assistant\",\"content\":\"hello\"}}]", .{bad});
        defer a.free(json);
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer parsed.deinit();
        var pi = try parseInput(a, parsed.value, null, null, null, .{});
        defer pi.deinit();
        try testing.expectEqual(@as(usize, 2), pi.messages.items.len);
        try testing.expectEqualStrings("hello", pi.messages.items[1].content);
        try testing.expect(pi.messages.items[1].reasoning_content == null);
    }
}

test "parseInput: reasoning joins the assistant message that follows it" {
    const a = testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\[{"role":"user","content":"q"},
        \\ {"type":"reasoning","content":[{"type":"reasoning_text","text":"think "},{"type":"reasoning_text","text":"hard"}]},
        \\ {"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
        \\ {"role":"user","content":"next"}]
    , .{});
    defer parsed.deinit();
    var pi = try parseInput(a, parsed.value, null, null, null, .{});
    defer pi.deinit();
    try testing.expectEqual(@as(usize, 3), pi.messages.items.len);
    try testing.expectEqualStrings("answer", pi.messages.items[1].content);
    try testing.expectEqualStrings("think hard", pi.messages.items[1].reasoning_content.?);
}
