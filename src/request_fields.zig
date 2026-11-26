//! The request fields of every generation surface, in ONE table: the key, its type and the surfaces that read it.
//! `check` is the only way into a body: it refuses, by name, a field of the wrong type or range and a field the
//! surface does not read, and hands back `Checked`, whose readers take only the fields the table gives that surface,
//! at the type the check proved. A field is therefore read as it was checked, and never checked on one path and read
//! on another. Pure functions of the bytes, so the host tier tests them.
const std = @import("std");
const request_parse = @import("request_parse.zig");
const stop_sequences = @import("stop_sequences.zig");
const model_settings = @import("model_settings.zig");
const trunk = @import("trunk.zig");
const chat_mod = @import("chat.zig");

const wholeNumber = request_parse.wholeNumber;
const fieldFault = request_parse.fieldFault;

pub const Surface = enum {
    chat,
    completions,
    messages,
    responses,
    /// `/v1/responses/compact`: it generates nothing, so it reads only the conversation it folds.
    compact,

    pub fn route(s: Surface) []const u8 {
        return switch (s) {
            .chat => "/v1/chat/completions",
            .completions => "/v1/completions",
            .messages => "/v1/messages",
            .responses => "/v1/responses",
            .compact => "/v1/responses/compact",
        };
    }
};

const Range = struct { min: f64, max: f64 };
const Checker = *const fn (buf: []u8, s: Surface, v: std.json.Value) ?[]const u8;

/// The value that turns an unserved feature off: `false` for a flag, `0` for a count, or none (any value asks for it;
/// only absent or null is off).
pub const Off = enum { flag, count, absent };

pub const Kind = union(enum) {
    /// A finite number in the range, never clamped.
    number: Range,
    /// A whole number in the range: a float with no fractional part counts, 2.5 is refused.
    whole: Range,
    flag,
    text,
    /// A string or an array of strings (`stop_sequences.valid`).
    stop,
    /// A token budget: a whole number from 0 (the default) to 2^32 - 1.
    budget,
    /// `seed`: a whole number that fits 64 bits; a negative one means unseeded.
    seed,
    /// A sampler other servers run and this one does not: accepted only at the value that turns it off.
    sampler_off: f64,
    /// A feature this server does not serve: accepted only at the value that turns it off (null is absent), and an
    /// `absent` one never.
    not_served: Off,
    /// `n`: the single choice this server returns.
    one_choice,
    /// A value with a shape of its own, checked by the named function.
    shaped: Checker,
};

const Row = struct {
    key: []const u8,
    kind: Kind,
    on: []const Surface,
    /// Surfaces that accept the field without reading it: a field of their own API that this server serves without
    /// acting on, type-checked. Every other surface that does not read the row refuses it by name.
    carried: []const Surface = &.{},
    /// The refusal's sentence where the surface has a better one than the generic.
    unread: ?[]const u8 = null,

    fn reads(r: Row, s: Surface) bool {
        for (r.on) |o| if (o == s) return true;
        return false;
    }

    fn carries(r: Row, s: Surface) bool {
        for (r.carried) |o| if (o == s) return true;
        return false;
    }
};

const all = [_]Surface{ .chat, .completions, .messages, .responses };
const every_surface = [_]Surface{ .chat, .completions, .messages, .responses, .compact };
const not_messages = [_]Surface{ .chat, .completions, .responses };
const chat_responses = [_]Surface{ .chat, .responses };
const chat_completions = [_]Surface{ .chat, .completions };
const chat_messages_responses = [_]Surface{ .chat, .messages, .responses };

const constraint_hint = "constrain the output with a JSON schema: response_format (chat), text.format (Responses) or output_config.format (Messages)";
const sampler_hint = "this server samples with temperature, top_k, top_p and min_p only";

pub const rows = [_]Row{
    // The routing id: every surface reads it, and only null or absence means "the default", never a value of another type.
    .{ .key = "model", .kind = .text, .on = &every_surface },
    .{ .key = "temperature", .kind = .{ .number = .{ .min = 0, .max = 2 } }, .on = &all },
    .{ .key = "top_p", .kind = .{ .number = .{ .min = 0, .max = 1 } }, .on = &all },
    .{ .key = "top_k", .kind = .{ .whole = .{ .min = 0, .max = 1000 } }, .on = &all },
    .{ .key = "min_p", .kind = .{ .number = .{ .min = 0, .max = 1 } }, .on = &not_messages, .unread = "'min_p' is not a field of /v1/messages: send it to /v1/chat/completions, /v1/completions or /v1/responses" },
    .{ .key = "presence_penalty", .kind = .{ .number = .{ .min = -2, .max = 2 } }, .on = &not_messages },
    .{ .key = "frequency_penalty", .kind = .{ .number = .{ .min = -2, .max = 2 } }, .on = &not_messages },
    .{ .key = "repeat_penalty", .kind = .{ .number = .{ .min = 0, .max = 10 } }, .on = &not_messages },
    .{ .key = "repetition_penalty", .kind = .{ .number = .{ .min = 0, .max = 10 } }, .on = &not_messages },
    .{ .key = "think_penalty", .kind = .{ .number = .{ .min = 0, .max = model_settings.think_penalty_max } }, .on = &all },
    .{ .key = "typical_p", .kind = .{ .sampler_off = 1 }, .on = &all },
    .{ .key = "tfs_z", .kind = .{ .sampler_off = 1 }, .on = &all },
    .{ .key = "top_a", .kind = .{ .sampler_off = 0 }, .on = &all },
    .{ .key = "mirostat", .kind = .{ .sampler_off = 0 }, .on = &all },
    .{ .key = "dry_multiplier", .kind = .{ .sampler_off = 0 }, .on = &all },
    .{ .key = "xtc_probability", .kind = .{ .sampler_off = 0 }, .on = &all },
    .{ .key = "seed", .kind = .seed, .on = &all },
    .{ .key = "stop", .kind = .stop, .on = &not_messages },
    .{ .key = "stop_sequences", .kind = .stop, .on = &.{.messages} },
    .{ .key = "max_completion_tokens", .kind = .budget, .on = &chat_completions },
    .{ .key = "max_tokens", .kind = .budget, .on = &all },
    .{ .key = "max_output_tokens", .kind = .budget, .on = &.{.responses} },
    .{ .key = "kv_quant", .kind = .{ .shaped = kvQuantFault }, .on = &all },
    .{ .key = "n", .kind = .one_choice, .on = &all },
    .{ .key = "logit_bias", .kind = .{ .shaped = anyShape }, .on = &chat_completions },
    .{ .key = "logprobs", .kind = .{ .not_served = .flag }, .on = &.{.chat} },
    .{ .key = "logprobs", .kind = .{ .not_served = .absent }, .on = &.{.completions} },
    .{ .key = "top_logprobs", .kind = .{ .not_served = .count }, .on = &chat_responses },
    .{ .key = "include", .kind = .{ .shaped = includeFault }, .on = &.{.responses} },
    .{ .key = "enable_mtp", .kind = .flag, .on = &all },
    .{ .key = "enable_drafter", .kind = .flag, .on = &all },
    .{ .key = "enable_batch_mtp", .kind = .flag, .on = &all },
    .{ .key = "stream", .kind = .flag, .on = &all },
    .{ .key = "return_progress", .kind = .flag, .on = &chat_completions },
    .{ .key = "stream_options", .kind = .{ .shaped = streamOptionsFault }, .on = &chat_completions, .carried = &.{.responses} },
    .{ .key = "enable_thinking", .kind = .flag, .on = &chat_responses },
    .{ .key = "reasoning_budget_tokens", .kind = .{ .whole = .{ .min = std.math.minInt(i32), .max = std.math.maxInt(i32) } }, .on = &chat_responses },
    .{ .key = "reasoning_effort", .kind = .{ .shaped = anyShape }, .on = &.{.chat} },
    .{ .key = "chat_template_kwargs", .kind = .{ .shaped = templateKwargsFault }, .on = &chat_messages_responses },
    .{ .key = "continue_final_message", .kind = .flag, .on = &.{.chat} },
    .{ .key = "thinking", .kind = .{ .shaped = thinkingFault }, .on = &.{.messages} },
    .{ .key = "reasoning", .kind = .{ .shaped = reasoningFault }, .on = &.{.responses} },
    .{ .key = "output_config", .kind = .{ .shaped = outputConfigFault }, .on = &.{.messages} },
    .{ .key = "tools", .kind = .{ .shaped = toolsShape }, .on = &chat_messages_responses },
    .{ .key = "tool_choice", .kind = .{ .shaped = toolChoiceFault }, .on = &chat_messages_responses },
    .{ .key = "parallel_tool_calls", .kind = .flag, .on = &chat_responses },
    .{ .key = "response_format", .kind = .{ .shaped = responseFormatFault }, .on = &chat_responses },
    .{ .key = "text", .kind = .{ .shaped = textFault }, .on = &.{.responses} },
    .{ .key = "messages", .kind = .{ .shaped = messagesFault }, .on = &.{ .chat, .messages } },
    .{ .key = "system", .kind = .{ .shaped = systemFault }, .on = &.{.messages} },
    .{ .key = "input", .kind = .{ .shaped = inputFault }, .on = &.{ .responses, .compact } },
    .{ .key = "instructions", .kind = .text, .on = &.{ .responses, .compact } },
    .{ .key = "previous_response_id", .kind = .text, .on = &.{ .responses, .compact } },
    .{ .key = "prompt", .kind = .{ .shaped = anyShape }, .on = &.{.completions} },
    .{ .key = "suffix", .kind = .{ .shaped = suffixFault }, .on = &.{.completions} },
    .{ .key = "best_of", .kind = .{ .shaped = bestOfFault }, .on = &.{.completions} },
    .{ .key = "echo", .kind = .flag, .on = &.{.completions} },
    .{ .key = "ignore_eos", .kind = .flag, .on = &.{.completions}, .unread = "'ignore_eos' is supported on /v1/completions only: a chat reply ends at its end of turn" },
    .{ .key = "truncation", .kind = .{ .shaped = truncationFault }, .on = &.{.responses} },
    .{ .key = "max_tool_calls", .kind = .{ .whole = .{ .min = 0, .max = std.math.maxInt(u32) } }, .on = &.{.responses} },
    .{ .key = "background", .kind = .flag, .on = &.{.responses} },
    .{ .key = "metadata", .kind = .{ .shaped = metadataFault }, .on = &.{ .messages, .responses }, .carried = &all },
    .{ .key = "prompt_cache_key", .kind = .text, .on = &all },
    .{ .key = "safety_identifier", .kind = .text, .on = &.{.responses}, .carried = &all },
    .{ .key = "store", .kind = .flag, .on = &.{.responses}, .carried = &all },
    .{ .key = "service_tier", .kind = .text, .on = &.{.responses}, .carried = &all },
    // Behaviour other servers give these and this one does not: refused on every surface, never dropped.
    .{ .key = "functions", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'functions' is OpenAI's legacy tool API and is not read: send 'tools'" },
    .{ .key = "function_call", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'function_call' is OpenAI's legacy tool API and is not read: send 'tool_choice'" },
    .{ .key = "grammar", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'grammar' is not read: " ++ constraint_hint },
    .{ .key = "guided_json", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'guided_json' is not read: " ++ constraint_hint },
    .{ .key = "guided_regex", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'guided_regex' is not read: " ++ constraint_hint },
    .{ .key = "guided_choice", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'guided_choice' is not read: " ++ constraint_hint },
    .{ .key = "guided_grammar", .kind = .{ .shaped = anyShape }, .on = &.{}, .unread = "'guided_grammar' is not read: " ++ constraint_hint },
};

/// Whether `surface` reads `key` at all, whatever the type it reads it at.
fn readsAnywhere(surface: Surface, key: []const u8) bool {
    for (rows) |r| if (std.mem.eql(u8, r.key, key) and r.reads(surface)) return true;
    return false;
}

fn numberOf(v: std.json.Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| if (std.math.isFinite(f)) f else null,
        else => null,
    };
}

fn kindFault(buf: []u8, r: Row, s: Surface, v: std.json.Value) ?[]const u8 {
    switch (r.kind) {
        .number => |g| {
            const n = numberOf(v) orelse return fieldFault(buf, "'{s}' must be a number from {d} to {d}", .{ r.key, g.min, g.max });
            if (n < g.min or n > g.max) return fieldFault(buf, "'{s}' must be a number from {d} to {d}", .{ r.key, g.min, g.max });
        },
        .whole => |g| {
            const n = wholeNumber(v) orelse return fieldFault(buf, "'{s}' must be a whole number from {d} to {d}", .{ r.key, g.min, g.max });
            const f: f64 = @floatFromInt(n);
            if (f < g.min or f > g.max) return fieldFault(buf, "'{s}' must be a whole number from {d} to {d}", .{ r.key, g.min, g.max });
        },
        .flag => if (v != .bool) return fieldFault(buf, "'{s}' must be true or false", .{r.key}),
        .text => if (v != .string) return fieldFault(buf, "'{s}' must be a string", .{r.key}),
        .stop => if (!stop_sequences.valid(v)) return fieldFault(buf, "'{s}' must be a string or an array of strings", .{r.key}),
        .budget => {
            const n = wholeNumber(v) orelse return fieldFault(buf, "'{s}' must be an integer", .{r.key});
            if (n < 0 or n > std.math.maxInt(u32)) return fieldFault(buf, "'{s}' must be an integer from 0 (the default) to {d}", .{ r.key, std.math.maxInt(u32) });
        },
        // A negative seed is the documented "unseeded"; an integer wider than i64 is no seed at all.
        .seed => if (wholeNumber(v) == null) return "'seed' must be an integer that fits in 64 bits; a negative one means unseeded",
        .sampler_off => |off| if (numberOf(v) != off) {
            return fieldFault(buf, "'{s}' is not supported: " ++ sampler_hint ++ "; omit it or send {d}", .{ r.key, off });
        },
        .not_served => |off| {
            const is_off = switch (off) {
                .flag => v == .bool and !v.bool,
                .count => numberOf(v) == 0,
                .absent => false,
            };
            if (!is_off) return switch (off) {
                .flag => fieldFault(buf, "'{s}' is not supported: this server returns no token probabilities; omit it or send false", .{r.key}),
                .count => fieldFault(buf, "'{s}' is not supported: this server returns no token probabilities; omit it or send 0", .{r.key}),
                .absent => fieldFault(buf, "'{s}' is not supported: this server returns no token probabilities; omit it", .{r.key}),
            };
        },
        .one_choice => if (request_parse.nChoicesFault(v)) |m| return m,
        .shaped => |checker| return checker(buf, s, v),
    }
    return null;
}

/// The first field of `root` that is of the wrong type or range, or that `surface` does not read, as a message naming
/// it; null and absent are the same, the field's default.
pub fn fault(buf: []u8, surface: Surface, root: std.json.ObjectMap) ?[]const u8 {
    for (rows) |r| {
        const v = root.get(r.key) orelse continue;
        if (v == .null) continue;
        if (!r.reads(surface)) {
            if (readsAnywhere(surface, r.key)) continue;
            // A compaction carries the Responses request it folds and generates nothing from the rest.
            if (surface == .compact) continue;
            if (!r.carries(surface)) return r.unread orelse fieldFault(buf, "'{s}' is not a field {s} reads: remove it", .{ r.key, surface.route() });
        }
        if (kindFault(buf, r, surface, v)) |m| return m;
    }
    return null;
}

fn rowFor(comptime s: Surface, comptime key: []const u8) Row {
    for (rows) |r| if (comptime std.mem.eql(u8, r.key, key) and r.reads(s)) return r;
    @compileError("request_fields: " ++ @tagName(s) ++ " reads no field '" ++ key ++ "'");
}

fn expectKind(comptime s: Surface, comptime key: []const u8, comptime tag: std.meta.Tag(Kind)) void {
    if (std.meta.activeTag(rowFor(s, key).kind) != tag) @compileError("request_fields: '" ++ key ++ "' is not a " ++ @tagName(tag) ++ " on " ++ @tagName(s));
}

/// A body `fault` passed, for `surface`: the only reader of its fields.
pub fn Checked(comptime surface: Surface) type {
    return struct {
        const Self = @This();

        /// The body, for the readers of shaped fields (`messages`, `tools`...) that read their own checked shape.
        root: std.json.ObjectMap,

        /// A `number` field as the client sent it; null when absent.
        pub fn number(self: Self, comptime key: []const u8) ?f32 {
            comptime expectKind(surface, key, .number);
            return numberValue(self.root.get(key) orelse return null);
        }

        /// A `whole` field; null when absent.
        pub fn whole(self: Self, comptime key: []const u8) ?i64 {
            comptime expectKind(surface, key, .whole);
            return wholeValue(self.root.get(key) orelse return null);
        }

        /// A `flag` field; null when absent, so the surface's own default applies.
        pub fn flag(self: Self, comptime key: []const u8) ?bool {
            comptime expectKind(surface, key, .flag);
            const v = self.root.get(key) orelse return null;
            return switch (v) {
                .bool => |b| b,
                .null => null,
                else => unreachable,
            };
        }

        /// A `text` field; null when absent.
        pub fn text(self: Self, comptime key: []const u8) ?[]const u8 {
            comptime expectKind(surface, key, .text);
            const v = self.root.get(key) orelse return null;
            return switch (v) {
                .string => |s| s,
                .null => null,
                else => unreachable,
            };
        }

        /// `top_k`; null when absent, 0 turns the filter off.
        pub fn topK(self: Self) ?u32 {
            const n = self.whole("top_k") orelse return null;
            return @intCast(n);
        }

        /// The request's seed; null when absent or negative (unseeded).
        pub fn seed(self: Self) ?u64 {
            comptime expectKind(surface, "seed", .seed);
            const n = wholeValue(self.root.get("seed") orelse return null) orelse return null;
            return if (n < 0) null else @intCast(n);
        }

        /// The multiplicative repeat penalty (`repeat_penalty`, else `repetition_penalty`); 0 or absent is 1, off.
        pub fn repeatPenalty(self: Self) f32 {
            const rp = self.number("repeat_penalty") orelse 0;
            if (rp > 0) return rp;
            const alt = self.number("repetition_penalty") orelse 0;
            return if (alt > 0) alt else 1;
        }

        /// A `-2..2` additive penalty (`presence_penalty`, `frequency_penalty`); 0 is off.
        pub fn additivePenalty(self: Self, comptime key: []const u8) f32 {
            return self.number(key) orelse 0;
        }

        /// The request's `min_p`: absent is 0, off.
        pub fn minP(self: Self) f32 {
            return self.number("min_p") orelse 0;
        }

        /// The request's own `think_penalty`; null when absent.
        pub fn thinkPenalty(self: Self) ?f32 {
            return self.number("think_penalty");
        }

        /// The first positive token budget among the surface's spellings, so an absent, null or 0 one (the app's
        /// Auto) leaves the next in force.
        pub fn requestedMaxTokens(self: Self) ?std.json.Value {
            inline for (budget_keys) |k| if (comptime readsAnywhere(surface, k)) {
                if (self.root.get(k)) |v| if (wholeValue(v)) |n| if (n > 0) return v;
            };
            return null;
        }

        const budget_keys = switch (surface) {
            .chat, .completions => [_][]const u8{ "max_completion_tokens", "max_tokens" },
            .messages => [_][]const u8{"max_tokens"},
            .responses, .compact => [_][]const u8{ "max_output_tokens", "max_tokens" },
        };

        /// Appends the request's stop sequences, the surface's own spelling of them.
        pub fn stops(self: Self, allocator: std.mem.Allocator, list: *std.ArrayList([]const u8)) !void {
            const key = comptime if (surface == .messages) "stop_sequences" else "stop";
            comptime expectKind(surface, key, .stop);
            try stop_sequences.append(allocator, list, self.root.get(key));
        }

        /// `return_progress`: llama.cpp's prompt processing progress chunks on a
        /// stream. Off unless set true; only a stream reads it.
        pub fn returnProgress(self: Self) bool {
            comptime _ = rowFor(surface, "return_progress");
            return self.flag("return_progress") orelse false;
        }

        /// `stream_options.include_usage`: off unless set true.
        pub fn includeUsage(self: Self) bool {
            comptime _ = rowFor(surface, "stream_options");
            const so = self.root.get("stream_options") orelse return false;
            const v = (switch (so) {
                .object => |o| o.get("include_usage"),
                else => null,
            }) orelse return false;
            return v == .bool and v.bool;
        }

        /// A bool of `chat_template_kwargs` (`enable_thinking` on chat and Responses, `preserve_thinking`).
        pub fn templateKwarg(self: Self, comptime key: []const u8) ?bool {
            comptime _ = rowFor(surface, "chat_template_kwargs");
            if (comptime std.mem.eql(u8, key, "enable_thinking") and surface == .messages) @compileError("/v1/messages reads no chat_template_kwargs.enable_thinking");
            const kw = self.root.get("chat_template_kwargs") orelse return null;
            if (kw != .object) return null;
            const v = kw.object.get(key) orelse return null;
            return if (v == .bool) v.bool else null;
        }

        /// Anthropic's `thinking` object; null when absent or null.
        pub fn thinking(self: Self) ?AnthropicThinking {
            comptime _ = rowFor(surface, "thinking");
            const v = self.root.get("thinking") orelse return null;
            if (v == .null) return null;
            if (v != .object) unreachable;
            const t = v.object.get("type") orelse unreachable;
            if (t != .string) unreachable;
            const kind = std.meta.stringToEnum(ThinkingKind, t.string) orelse unreachable;
            const budget: ?i32 = if (v.object.get("budget_tokens")) |b| switch (b) {
                .integer => |i| std.math.cast(i32, i) orelse unreachable,
                else => null,
            } else null;
            return .{ .kind = kind, .budget_tokens = budget };
        }

        /// Responses' `truncation` (auto or disabled, the default).
        pub fn truncation(self: Self) []const u8 {
            comptime _ = rowFor(surface, "truncation");
            const v = self.root.get("truncation") orelse return "disabled";
            return switch (v) {
                .string => |t| t,
                .null => "disabled",
                else => unreachable,
            };
        }

        /// The thinking switch the client set: top-level `enable_thinking`, else the vLLM `chat_template_kwargs` one.
        pub fn enableThinking(self: Self) ?bool {
            return self.flag("enable_thinking") orelse self.templateKwarg("enable_thinking");
        }

        /// A `reasoning_budget_tokens` the client set; null when absent.
        pub fn reasoningBudget(self: Self) ?i32 {
            const n = self.whole("reasoning_budget_tokens") orelse return null;
            return @intCast(n);
        }

    };
}

fn numberValue(v: std.json.Value) ?f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        .null => null,
        else => unreachable,
    };
}

fn wholeValue(v: std.json.Value) ?i64 {
    return switch (v) {
        .null => null,
        else => wholeNumber(v) orelse unreachable,
    };
}

pub const ThinkingKind = enum { enabled, disabled, adaptive };
pub const AnthropicThinking = struct { kind: ThinkingKind, budget_tokens: ?i32 };

pub fn CheckResult(comptime surface: Surface) type {
    return union(enum) { fault: []const u8, ok: Checked(surface) };
}

/// The body's fields checked for `surface`: a fault naming the first bad or unread field, else the readers.
pub fn check(comptime surface: Surface, buf: []u8, root: std.json.ObjectMap) CheckResult(surface) {
    if (fault(buf, surface, root)) |why| return .{ .fault = why };
    return .{ .ok = .{ .root = root } };
}

// ---- the shapes of fields that carry more than a scalar ----

fn anyShape(_: []u8, _: Surface, _: std.json.Value) ?[]const u8 {
    return null;
}

fn kvQuantFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (trunk.KVQuantConfig.fromJsonValue(v) == null) return "'kv_quant' must be one of off, 4, 6 or 8 (0 and 16 also mean off)";
    return null;
}

fn streamOptionsFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'stream_options' must be an object";
    if (v.object.get("include_usage")) |u| if (u != .null and u != .bool) return "'stream_options.include_usage' must be true or false";
    return null;
}

fn templateKwargsFault(buf: []u8, s: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'chat_template_kwargs' must be an object";
    const bools: []const []const u8 = if (s == .messages) &.{"preserve_thinking"} else &.{ "enable_thinking", "preserve_thinking" };
    for (bools) |k| if (v.object.get(k)) |b| if (b != .null and b != .bool) {
        return fieldFault(buf, "'chat_template_kwargs.{s}' must be true or false", .{k});
    };
    if (s == .messages) if (v.object.get("enable_thinking")) |b| if (b != .null) {
        return "'chat_template_kwargs.enable_thinking' is not read by /v1/messages: send 'thinking' or 'output_config.effort'";
    };
    return null;
}

/// Anthropic's `thinking`: `{type: enabled | disabled | adaptive, budget_tokens}`.
fn thinkingFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'thinking' must be an object typed enabled, disabled or adaptive";
    const t = v.object.get("type") orelse return "'thinking.type' is required: enabled, disabled or adaptive";
    const ok = t == .string and (std.mem.eql(u8, t.string, "enabled") or std.mem.eql(u8, t.string, "disabled") or std.mem.eql(u8, t.string, "adaptive"));
    if (!ok) return "'thinking.type' must be enabled, disabled or adaptive";
    if (v.object.get("budget_tokens")) |b| if (b != .null) {
        if (b != .integer or b.integer < std.math.minInt(i32) or b.integer > std.math.maxInt(i32)) return "'thinking.budget_tokens' must be an integer that fits in 32 bits";
    };
    return null;
}

/// Responses' `reasoning`: an object whose `effort` is a word (`request_policy.nonWordEffort` names a non-word) and
/// whose `summary` is one of the three the envelope echoes.
fn reasoningFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'reasoning' must be an object";
    if (v.object.get("summary")) |sv| if (sv != .null) {
        const ok = sv == .string and (std.mem.eql(u8, sv.string, "auto") or std.mem.eql(u8, sv.string, "concise") or std.mem.eql(u8, sv.string, "detailed"));
        if (!ok) return "'reasoning.summary' must be auto, concise or detailed";
    };
    return null;
}

/// Anthropic's `output_config`: `effort` (a word, checked where it resolves) and `format`, a json_schema.
fn outputConfigFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'output_config' must be an object";
    const f = v.object.get("format") orelse return null;
    switch (f) {
        .null => {},
        .object => |o| {
            const t = o.get("type") orelse return "'output_config.format.type' is required: json_schema";
            if (t != .string or !std.mem.eql(u8, t.string, "json_schema")) return "'output_config.format.type' must be json_schema";
        },
        else => return "'output_config.format' must be an object",
    }
    return null;
}

fn formatTypeFault(path: []const u8, o: std.json.ObjectMap) ?[]const u8 {
    const t = o.get("type") orelse return if (std.mem.eql(u8, path, "text.format")) "'text.format.type' is required: text, json_object or json_schema" else "'response_format.type' is required: text, json_object or json_schema";
    const ok = t == .string and (std.mem.eql(u8, t.string, "text") or std.mem.eql(u8, t.string, "json_object") or std.mem.eql(u8, t.string, "json_schema"));
    if (ok) return null;
    return if (std.mem.eql(u8, path, "text.format")) "'text.format.type' must be text, json_object or json_schema" else "'response_format.type' must be text, json_object or json_schema";
}

fn responseFormatFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'response_format' must be an object";
    return formatTypeFault("response_format", v.object);
}

/// `include` asks for extra output; the one entry this server cannot give is the output text's token probabilities.
fn includeFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .array) return null;
    for (v.array.items) |entry| {
        if (entry == .string and std.mem.eql(u8, entry.string, "message.output_text.logprobs")) {
            return "'include' is not supported with 'message.output_text.logprobs': this server returns no token probabilities; remove that entry";
        }
    }
    return null;
}

fn textFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .object) return "'text' must be an object";
    const f = v.object.get("format") orelse return null;
    return switch (f) {
        .null => null,
        .object => |o| formatTypeFault("text.format", o),
        else => "'text.format' must be an object",
    };
}

fn toolsShape(buf: []u8, s: Surface, v: std.json.Value) ?[]const u8 {
    return request_parse.toolsFault(buf, v, switch (s) {
        .messages => .messages,
        .responses => .responses,
        else => .chat,
    });
}

fn toolChoiceFault(buf: []u8, s: Surface, v: std.json.Value) ?[]const u8 {
    return switch (s) {
        .messages => request_parse.anthropicToolChoiceFault(v),
        .responses => request_parse.responsesToolChoiceShape(buf, v),
        else => request_parse.chatToolChoiceFault(v),
    };
}

fn suffixFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (v == .string and v.string.len == 0) return null;
    return "'suffix' is not supported: this server does not fill in the middle; omit it";
}

fn bestOfFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    if (wholeNumber(v) != 1) return "'best_of' must be 1: this server returns one completion per request and ranks no candidates";
    return null;
}

fn truncationFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    const ok = v == .string and (std.mem.eql(u8, v.string, "auto") or std.mem.eql(u8, v.string, "disabled"));
    return if (ok) null else "'truncation' must be auto or disabled";
}

fn metadataFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    return if (v == .object) null else "'metadata' must be an object";
}

// ---- the conversation: one reading of a message's content, one check of its shape ----

/// A message's `content`: text, content parts, or nothing (absent or null: the message may carry only tool calls, and
/// one that carries nothing is dropped, `Message.carriesNothing`).
pub const Content = union(enum) {
    none,
    text: []const u8,
    parts: []const std.json.Value,
};

/// The one reading of a message's `content`, for every surface. A value that is neither text nor an array is no
/// content: `conversationFault` refused it before any reader ran.
pub fn contentOf(v: ?std.json.Value) error{ContentWrongType}!Content {
    const c = v orelse return .none;
    return switch (c) {
        .null => .none,
        .string => |s| .{ .text = s },
        .array => |a| .{ .parts = a.items },
        else => error.ContentWrongType,
    };
}

/// A conversation entry as the readers take it: an object with a string `role`. `conversationFault` refused any other
/// before a reader ran, so this is an error only for a reader called without it.
pub const Message = struct { obj: std.json.ObjectMap, role: []const u8 };

pub fn messageOf(v: std.json.Value) error{MessageWrongShape}!Message {
    if (v != .object) return error.MessageWrongShape;
    const role = v.object.get("role") orelse return error.MessageWrongShape;
    if (role != .string) return error.MessageWrongShape;
    return .{ .obj = v.object, .role = role.string };
}

/// The first entry of a conversation list that is no message: not an object, a role that is no string or not one of
/// `roles`, or content that is neither text nor parts; a Responses item (`typed`) of another `type` (a call, its
/// output, reasoning) carries no role and is not checked here.
fn conversationFault(buf: []u8, comptime list: []const u8, items: []const std.json.Value, comptime roles: []const []const u8, typed: bool) ?[]const u8 {
    for (items, 0..) |m, i| {
        if (m != .object) return fieldFault(buf, "'" ++ list ++ "[{d}]' must be an object", .{i});
        if (typed) if (m.object.get("type")) |t| {
            if (t != .string) return fieldFault(buf, "'" ++ list ++ "[{d}].type' must be a string", .{i});
            if (!std.mem.eql(u8, t.string, "message")) continue;
        };
        const role = m.object.get("role") orelse return fieldFault(buf, "'" ++ list ++ "[{d}].role' is required", .{i});
        if (role != .string) return fieldFault(buf, "'" ++ list ++ "[{d}].role' must be a string", .{i});
        for (roles) |r| {
            if (std.mem.eql(u8, role.string, r)) break;
        } else return fieldFault(buf, "'" ++ list ++ "[{d}].role' must be one of " ++ comptime roleList(roles) ++ "; got \"{s}\"", .{ i, role.string[0..@min(role.string.len, 32)] });
        _ = contentOf(m.object.get("content")) catch return fieldFault(buf, "'" ++ list ++ "[{d}].content' must be a string or an array of content parts", .{i});
    }
    return null;
}

/// `a, b or c`: the roles a surface lists in its refusal.
fn roleList(comptime roles: []const []const u8) []const u8 {
    comptime var out: []const u8 = "";
    inline for (roles, 0..) |r, i| out = out ++ (if (i == 0) "" else if (i == roles.len - 1) " or " else ", ") ++ r;
    return out;
}

/// The roles /v1/messages carries: a `system` message inside `messages` is Claude Code's hook output, read into the
/// prompt where it sits; a tool answer is a `tool_result` block, never a role.
const anthropic_roles = [_][]const u8{ "user", "assistant", "system" };

fn messagesFault(buf: []u8, s: Surface, v: std.json.Value) ?[]const u8 {
    if (v != .array) return "'messages' must be an array";
    if (s == .messages) {
        if (conversationFault(buf, "messages", v.array.items, &anthropic_roles, false)) |m| return m;
        return request_parse.toolResultFault(buf, v.array.items);
    }
    return conversationFault(buf, "messages", v.array.items, &chat_mod.request_roles, false);
}

fn systemFault(_: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string, .array => null,
        else => "'system' must be a string or an array of text blocks",
    };
}

fn inputFault(buf: []u8, _: Surface, v: std.json.Value) ?[]const u8 {
    return switch (v) {
        .string => null,
        .array => |a| conversationFault(buf, "input", a.items, &chat_mod.request_roles, true),
        else => "'input' must be a string or an array of input items",
    };
}

const testing = std.testing;

fn expectFault(surface: Surface, body: []const u8, field: ?[]const u8) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    var buf: [320]u8 = undefined;
    const got = fault(&buf, surface, parsed.value.object);
    errdefer std.debug.print("\n{s} body: {s}\ngot: {?s}\n", .{ @tagName(surface), body, got });
    if (field) |f| {
        const quoted = try std.fmt.allocPrint(testing.allocator, "'{s}'", .{f});
        defer testing.allocator.free(quoted);
        try testing.expect(std.mem.indexOf(u8, got orelse return error.NotRefused, quoted) != null);
    } else try testing.expectEqual(@as(?[]const u8, null), got);
}

const Case = struct { surface: Surface = .chat, body: []const u8, field: ?[]const u8 };

fn expectCases(cases: []const Case) !void {
    for (cases) |c| try expectFault(c.surface, c.body, c.field);
}

test "a scalar field of the wrong type or range is refused by name, on the surface that reads it" {
    try expectCases(&.{
        .{ .body = "{}", .field = null },
        .{ .body = "{\"max_tokens\":7,\"temperature\":0,\"top_p\":1,\"top_k\":40,\"seed\":-1,\"stop\":[\"a\"],\"kv_quant\":\"off\"}", .field = null },
        .{ .body = "{\"max_tokens\":7.0,\"seed\":5.0,\"top_k\":40.0,\"max_completion_tokens\":null,\"kv_quant\":16,\"top_p\":1.0}", .field = null },
        .{ .body = "{\"max_tokens\":0}", .field = null },
        .{ .body = "{\"temperature\":2,\"top_p\":0,\"top_k\":1000,\"presence_penalty\":-2,\"frequency_penalty\":2,\"repeat_penalty\":10,\"seed\":-5}", .field = null },
        .{ .surface = .responses, .body = "{\"max_output_tokens\":0}", .field = null },
        .{ .body = "{\"stop\":null,\"temperature\":null,\"top_k\":null,\"top_logprobs\":null}", .field = null },
        .{ .body = "{\"temperature\":-1}", .field = "temperature" },
        .{ .body = "{\"temperature\":100}", .field = "temperature" },
        .{ .body = "{\"top_p\":2}", .field = "top_p" },
        .{ .body = "{\"top_p\":-0.1}", .field = "top_p" },
        .{ .body = "{\"top_k\":-3}", .field = "top_k" },
        .{ .body = "{\"top_k\":1001}", .field = "top_k" },
        .{ .body = "{\"top_k\":2.5}", .field = "top_k" },
        .{ .surface = .messages, .body = "{\"top_k\":2.5}", .field = "top_k" },
        .{ .surface = .responses, .body = "{\"top_k\":2.5}", .field = "top_k" },
        .{ .body = "{\"presence_penalty\":3}", .field = "presence_penalty" },
        .{ .body = "{\"frequency_penalty\":-2.5}", .field = "frequency_penalty" },
        .{ .body = "{\"repeat_penalty\":11}", .field = "repeat_penalty" },
        .{ .body = "{\"repetition_penalty\":-1}", .field = "repetition_penalty" },
        .{ .body = "{\"seed\":9223372036854775808}", .field = "seed" },
        .{ .body = "{\"max_tokens\":-1}", .field = "max_tokens" },
        .{ .body = "{\"max_completion_tokens\":-5}", .field = "max_completion_tokens" },
        .{ .body = "{\"max_tokens\":4294967296}", .field = "max_tokens" },
        .{ .surface = .responses, .body = "{\"max_output_tokens\":\"1\"}", .field = "max_output_tokens" },
        .{ .body = "{\"think_penalty\":-4}", .field = "think_penalty" },
        .{ .body = "{\"think_penalty\":1e9}", .field = "think_penalty" },
        .{ .body = "{\"reasoning_budget_tokens\":-1,\"think_penalty\":20}", .field = null },
        .{ .body = "{\"reasoning_budget_tokens\":1099511627776}", .field = "reasoning_budget_tokens" },
        .{ .body = "{\"reasoning_budget_tokens\":\"64\"}", .field = "reasoning_budget_tokens" },
        .{ .body = "{\"max_tokens\":7.9}", .field = "max_tokens" },
        .{ .body = "{\"max_tokens\":\"7\"}", .field = "max_tokens" },
        .{ .body = "{\"max_tokens\":true}", .field = "max_tokens" },
        .{ .body = "{\"max_completion_tokens\":\"7\"}", .field = "max_completion_tokens" },
        .{ .body = "{\"temperature\":\"hot\"}", .field = "temperature" },
        .{ .body = "{\"top_p\":true}", .field = "top_p" },
        .{ .body = "{\"top_k\":\"5\"}", .field = "top_k" },
        .{ .body = "{\"seed\":\"abc\"}", .field = "seed" },
        .{ .body = "{\"seed\":1.5}", .field = "seed" },
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
        .{ .body = "{\"n\":1,\"service_tier\":null}", .field = null },
        .{ .body = "{\"n\":1.0}", .field = null },
        .{ .body = "{\"n\":2}", .field = "n" },
        .{ .body = "{\"n\":0}", .field = "n" },
        .{ .body = "{\"n\":\"2\"}", .field = "n" },
        .{ .surface = .messages, .body = "{\"n\":2}", .field = "n" },
        .{ .body = "{\"service_tier\":5}", .field = "service_tier" },
    });
}

test "a sampler this server does not run is accepted only at the value that turns it off, on every surface" {
    for ([_]Surface{ .chat, .completions, .messages, .responses }) |s| try expectCases(&.{
        .{ .surface = s, .body = "{\"typical_p\":1,\"tfs_z\":1.0,\"top_a\":0,\"mirostat\":0,\"dry_multiplier\":0,\"xtc_probability\":0.0}", .field = null },
        .{ .surface = s, .body = "{\"typical_p\":null,\"mirostat\":null}", .field = null },
        .{ .surface = s, .body = "{\"typical_p\":0.9}", .field = "typical_p" },
        .{ .surface = s, .body = "{\"tfs_z\":0.95}", .field = "tfs_z" },
        .{ .surface = s, .body = "{\"top_a\":0.1}", .field = "top_a" },
        .{ .surface = s, .body = "{\"mirostat\":2}", .field = "mirostat" },
        .{ .surface = s, .body = "{\"dry_multiplier\":0.8}", .field = "dry_multiplier" },
        .{ .surface = s, .body = "{\"xtc_probability\":0.5}", .field = "xtc_probability" },
        .{ .surface = s, .body = "{\"mirostat\":\"off\"}", .field = "mirostat" },
    });
}

test "one stop field per surface, a string or an array of strings, and no other surface's spelling" {
    try expectCases(&.{
        .{ .surface = .chat, .body = "{\"stop\":\"x\"}", .field = null },
        .{ .surface = .completions, .body = "{\"stop\":[\"a\",\"b\"]}", .field = null },
        .{ .surface = .responses, .body = "{\"stop\":[\"a\"]}", .field = null },
        .{ .surface = .messages, .body = "{\"stop_sequences\":[\"a\"]}", .field = null },
        .{ .surface = .chat, .body = "{\"stop\":5}", .field = "stop" },
        .{ .surface = .chat, .body = "{\"stop\":[\"a\",3]}", .field = "stop" },
        .{ .surface = .completions, .body = "{\"stop\":{}}", .field = "stop" },
        .{ .surface = .responses, .body = "{\"stop\":5}", .field = "stop" },
        .{ .surface = .responses, .body = "{\"stop\":[\"a\",7]}", .field = "stop" },
        .{ .surface = .messages, .body = "{\"stop_sequences\":\"a\"}", .field = null },
        .{ .surface = .messages, .body = "{\"stop_sequences\":5}", .field = "stop_sequences" },
        .{ .surface = .messages, .body = "{\"stop_sequences\":[\"a\",null]}", .field = "stop_sequences" },
        .{ .surface = .messages, .body = "{\"stop\":[\"a\"]}", .field = "stop" },
        .{ .surface = .chat, .body = "{\"stop_sequences\":[\"a\"]}", .field = "stop_sequences" },
    });
}

test "a bool field is true or false, and chat_template_kwargs and stream_options carry their own" {
    try expectCases(&.{
        .{ .body = "{\"stream\":true,\"store\":false,\"enable_mtp\":true,\"enable_drafter\":false,\"enable_batch_mtp\":true,\"parallel_tool_calls\":false,\"enable_thinking\":true,\"continue_final_message\":false,\"service_tier\":\"auto\"}", .field = null },
        .{ .surface = .completions, .body = "{\"echo\":false,\"ignore_eos\":true,\"stream\":null}", .field = null },
        .{ .body = "{\"stream\":null,\"store\":null,\"enable_mtp\":null,\"stream_options\":null,\"chat_template_kwargs\":null,\"service_tier\":null}", .field = null },
        .{ .body = "{\"stream_options\":{\"include_usage\":true},\"chat_template_kwargs\":{\"enable_thinking\":false,\"preserve_thinking\":true,\"other\":5}}", .field = null },
        .{ .body = "{\"stream\":\"true\"}", .field = "stream" },
        .{ .body = "{\"stream\":1}", .field = "stream" },
        .{ .body = "{\"store\":\"false\"}", .field = "store" },
        .{ .surface = .completions, .body = "{\"echo\":0}", .field = "echo" },
        .{ .surface = .completions, .body = "{\"ignore_eos\":\"true\"}", .field = "ignore_eos" },
        .{ .body = "{\"enable_mtp\":\"yes\"}", .field = "enable_mtp" },
        .{ .body = "{\"enable_drafter\":1}", .field = "enable_drafter" },
        .{ .body = "{\"enable_batch_mtp\":\"off\"}", .field = "enable_batch_mtp" },
        .{ .body = "{\"parallel_tool_calls\":\"false\"}", .field = "parallel_tool_calls" },
        .{ .body = "{\"enable_thinking\":\"false\"}", .field = "enable_thinking" },
        .{ .body = "{\"continue_final_message\":1}", .field = "continue_final_message" },
        .{ .body = "{\"stream_options\":\"all\"}", .field = "stream_options" },
        .{ .body = "{\"stream_options\":{\"include_usage\":\"true\"}}", .field = "stream_options.include_usage" },
        .{ .body = "{\"chat_template_kwargs\":5}", .field = "chat_template_kwargs" },
        .{ .body = "{\"chat_template_kwargs\":{\"enable_thinking\":\"no\"}}", .field = "chat_template_kwargs.enable_thinking" },
        .{ .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":1}}", .field = "chat_template_kwargs.preserve_thinking" },
        .{ .surface = .messages, .body = "{\"chat_template_kwargs\":{\"preserve_thinking\":true}}", .field = null },
        .{ .surface = .messages, .body = "{\"chat_template_kwargs\":{\"enable_thinking\":false}}", .field = "chat_template_kwargs.enable_thinking" },
    });
}

test "a field a surface does not read is refused by name, never dropped" {
    try expectCases(&.{
        // The penalties, the thinking switches and the other surfaces' spellings.
        .{ .surface = .messages, .body = "{\"presence_penalty\":1}", .field = "presence_penalty" },
        .{ .surface = .messages, .body = "{\"frequency_penalty\":1}", .field = "frequency_penalty" },
        .{ .surface = .messages, .body = "{\"repeat_penalty\":1.1}", .field = "repeat_penalty" },
        .{ .surface = .messages, .body = "{\"repetition_penalty\":1.1}", .field = "repetition_penalty" },
        .{ .surface = .messages, .body = "{\"min_p\":0.1}", .field = "min_p" },
        .{ .surface = .messages, .body = "{\"min_p\":null}", .field = null },
        .{ .surface = .messages, .body = "{\"enable_thinking\":true}", .field = "enable_thinking" },
        .{ .surface = .messages, .body = "{\"reasoning_budget_tokens\":64}", .field = "reasoning_budget_tokens" },
        .{ .surface = .messages, .body = "{\"top_logprobs\":3}", .field = "top_logprobs" },
        .{ .surface = .messages, .body = "{\"parallel_tool_calls\":false}", .field = "parallel_tool_calls" },
        .{ .surface = .messages, .body = "{\"continue_final_message\":true}", .field = "continue_final_message" },
        .{ .surface = .messages, .body = "{\"stream_options\":{\"include_usage\":true}}", .field = "stream_options" },
        .{ .surface = .messages, .body = "{\"reasoning_effort\":\"low\"}", .field = "reasoning_effort" },
        .{ .surface = .chat, .body = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":5}}", .field = "thinking" },
        .{ .surface = .chat, .body = "{\"output_config\":{}}", .field = "output_config" },
        .{ .surface = .chat, .body = "{\"ignore_eos\":true}", .field = "ignore_eos" },
        .{ .surface = .chat, .body = "{\"echo\":true}", .field = "echo" },
        .{ .surface = .responses, .body = "{\"stream_options\":{\"include_usage\":true}}", .field = null },
        .{ .surface = .responses, .body = "{\"stream_options\":\"all\"}", .field = "stream_options" },
        .{ .surface = .responses, .body = "{\"logit_bias\":{}}", .field = "logit_bias" },
        .{ .surface = .completions, .body = "{\"messages\":[]}", .field = "messages" },
        .{ .surface = .completions, .body = "{\"tools\":[]}", .field = "tools" },
        .{ .surface = .completions, .body = "{\"enable_thinking\":false}", .field = "enable_thinking" },
        .{ .surface = .completions, .body = "{\"response_format\":{\"type\":\"text\"}}", .field = "response_format" },
        .{ .surface = .completions, .body = "{\"reasoning_effort\":\"low\"}", .field = "reasoning_effort" },
        // `return_progress` reads on the two OpenAI completion surfaces only.
        .{ .surface = .messages, .body = "{\"return_progress\":true}", .field = "return_progress" },
        .{ .surface = .responses, .body = "{\"return_progress\":true}", .field = "return_progress" },
        .{ .surface = .chat, .body = "{\"return_progress\":\"yes\"}", .field = "return_progress" },
        .{ .surface = .chat, .body = "{\"return_progress\":true}", .field = null },
        .{ .surface = .chat, .body = "{\"return_progress\":false}", .field = null },
        .{ .surface = .completions, .body = "{\"return_progress\":true}", .field = null },
        // A null field is absent on every surface.
        .{ .surface = .completions, .body = "{\"messages\":null,\"tools\":null,\"enable_thinking\":null}", .field = null },
        // A field of the surface's own API the server serves without acting on is type-checked, not refused.
        .{ .surface = .chat, .body = "{\"store\":true,\"service_tier\":\"auto\",\"safety_identifier\":\"u\",\"metadata\":{}}", .field = null },
        .{ .surface = .messages, .body = "{\"service_tier\":\"auto\",\"metadata\":{\"user_id\":\"u\"}}", .field = null },
        .{ .surface = .chat, .body = "{\"metadata\":\"x\"}", .field = "metadata" },
        .{ .surface = .chat, .body = "{\"safety_identifier\":5}", .field = "safety_identifier" },
        // A compaction carries the Responses request it folds and generates nothing from the rest.
        .{ .surface = .compact, .body = "{\"input\":\"x\",\"instructions\":\"y\",\"tools\":[],\"reasoning\":{},\"text\":{},\"parallel_tool_calls\":true,\"temperature\":5}", .field = null },
        .{ .surface = .compact, .body = "{\"instructions\":5}", .field = "instructions" },
        .{ .surface = .compact, .body = "{\"previous_response_id\":5}", .field = "previous_response_id" },
    });
}

test "logprobs are refused by name on every surface that read them, accepted only at the value that turns them off" {
    try expectCases(&.{
        // Chat: `logprobs` is off at false, `top_logprobs` at 0; null is absent.
        .{ .surface = .chat, .body = "{\"logprobs\":false,\"top_logprobs\":0}", .field = null },
        .{ .surface = .chat, .body = "{\"logprobs\":null,\"top_logprobs\":null}", .field = null },
        .{ .surface = .chat, .body = "{\"logprobs\":true}", .field = "logprobs" },
        .{ .surface = .chat, .body = "{\"logprobs\":true,\"top_logprobs\":3}", .field = "logprobs" },
        .{ .surface = .chat, .body = "{\"logprobs\":1}", .field = "logprobs" },
        .{ .surface = .chat, .body = "{\"logprobs\":\"true\"}", .field = "logprobs" },
        .{ .surface = .chat, .body = "{\"top_logprobs\":3}", .field = "top_logprobs" },
        .{ .surface = .chat, .body = "{\"logprobs\":false,\"top_logprobs\":3}", .field = "top_logprobs" },
        .{ .surface = .chat, .body = "{\"top_logprobs\":\"0\"}", .field = "top_logprobs" },
        // Legacy completions: any integer asks, 0 included (the sampled token's logprob, no alternatives); only absent or null is off.
        .{ .surface = .completions, .body = "{}", .field = null },
        .{ .surface = .completions, .body = "{\"logprobs\":null}", .field = null },
        .{ .surface = .completions, .body = "{\"logprobs\":0}", .field = "logprobs" },
        .{ .surface = .completions, .body = "{\"logprobs\":1}", .field = "logprobs" },
        .{ .surface = .completions, .body = "{\"logprobs\":20}", .field = "logprobs" },
        .{ .surface = .completions, .body = "{\"logprobs\":true}", .field = "logprobs" },
        .{ .surface = .completions, .body = "{\"logprobs\":\"2\"}", .field = "logprobs" },
        // Responses: `top_logprobs` is off at 0, and the output-text logprobs entry of `include` is refused.
        .{ .surface = .responses, .body = "{\"top_logprobs\":0}", .field = null },
        .{ .surface = .responses, .body = "{\"top_logprobs\":5}", .field = "top_logprobs" },
        .{ .surface = .responses, .body = "{\"include\":[\"message.output_text.logprobs\"]}", .field = "include" },
        .{ .surface = .responses, .body = "{\"include\":[\"reasoning.encrypted_content\",\"message.output_text.logprobs\"]}", .field = "include" },
        .{ .surface = .responses, .body = "{\"include\":[\"reasoning.encrypted_content\"]}", .field = null },
        .{ .surface = .responses, .body = "{\"include\":[]}", .field = null },
        // Messages never read either field.
        .{ .surface = .messages, .body = "{\"logprobs\":true}", .field = "logprobs" },
        .{ .surface = .messages, .body = "{\"top_logprobs\":3}", .field = "top_logprobs" },
    });
}

test "completions' suffix and best_of are refused by name unless they ask for nothing" {
    try expectCases(&.{
        .{ .surface = .completions, .body = "{}", .field = null },
        .{ .surface = .completions, .body = "{\"suffix\":null,\"best_of\":1}", .field = null },
        .{ .surface = .completions, .body = "{\"suffix\":\"\",\"best_of\":1.0}", .field = null },
        .{ .surface = .completions, .body = "{\"suffix\":\"}\"}", .field = "suffix" },
        .{ .surface = .completions, .body = "{\"suffix\":5}", .field = "suffix" },
        .{ .surface = .completions, .body = "{\"best_of\":3}", .field = "best_of" },
        .{ .surface = .completions, .body = "{\"best_of\":0}", .field = "best_of" },
        .{ .surface = .completions, .body = "{\"best_of\":\"2\"}", .field = "best_of" },
    });
}

test "chat refuses an unknown role, response_format type, malformed tools and tool_choice by name" {
    try expectCases(&.{
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"developer\"}],\"tool_choice\":\"required\"}", .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[]},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"a\"}]}],\"response_format\":null,\"tools\":null}", .field = null },
        .{ .body = "{\"response_format\":{\"type\":\"json_schema\"},\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"a\"}}}", .field = null },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a.b/c:d\",\"parameters\":{\"type\":\"object\"}}},{\"name\":\"flat\",\"parameters\":null}]}", .field = null },
        .{ .body = "{\"messages\":[{\"role\":\"wizard\",\"content\":\"hi\"}]}", .field = "messages[0].role" },
        .{ .body = "{\"messages\":[{\"content\":\"no role\"}]}", .field = "messages[0].role" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},5]}", .field = "messages[1]" },
        .{ .body = "{\"messages\":[{\"role\":7,\"content\":\"hi\"}]}", .field = "messages[0].role" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":5}]}", .field = "messages[0].content" },
        .{ .body = "{\"messages\":[{\"role\":\"user\",\"content\":{\"type\":\"text\"}}]}", .field = "messages[0].content" },
        .{ .body = "{\"messages\":\"hi\"}", .field = "messages" },
        .{ .body = "{\"response_format\":{\"type\":\"xml\"}}", .field = "response_format.type" },
        .{ .body = "{\"response_format\":{}}", .field = "response_format.type" },
        .{ .body = "{\"response_format\":\"json\"}", .field = "response_format" },
        .{ .body = "{\"tools\":\"x\"}", .field = "tools" },
        .{ .body = "{\"tools\":[{\"type\":\"retrieval\"}]}", .field = "tools[0].type" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"description\":\"d\"}}]}", .field = "tools[0]" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"parameters\":\"bad\"}}]}", .field = "tools[0].parameters" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}},{\"type\":\"function\",\"function\":{\"name\":\"a\"}}]}", .field = "tools[1]" },
        .{ .body = "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"my tool/v1.2\"}}]}", .field = "tools[0]" },
        .{ .body = "{\"tool_choice\":\"sometimes\"}", .field = "tool_choice" },
        .{ .body = "{\"tool_choice\":7}", .field = "tool_choice" },
        // An object tool_choice is a form the reader honours, or it is refused by name: never read as auto.
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"a\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"none\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"auto\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"any\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"required\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"tool\",\"name\":\"a\"}}", .field = null },
        .{ .body = "{\"tool_choice\":{\"type\":\"allowed_tools\",\"mode\":\"required\",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"a\"}}]}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"tool\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"bogus\"}}", .field = "tool_choice.type" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\"}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"\"}}}", .field = "tool_choice.name" },
        .{ .body = "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":7}}}", .field = "tool_choice.name" },
    });
}

test "a Responses field the envelope echoes or the turn reads is refused by name when its type is wrong, never read as absent" {
    try expectCases(&.{
        .{ .surface = .responses, .body = "{\"instructions\":\"be brief\",\"previous_response_id\":\"resp_1\",\"safety_identifier\":\"u\",\"prompt_cache_key\":\"k\",\"truncation\":\"auto\",\"max_tool_calls\":3}", .field = null },
        .{ .surface = .responses, .body = "{\"instructions\":null,\"previous_response_id\":null,\"safety_identifier\":null,\"prompt_cache_key\":null,\"truncation\":\"disabled\",\"max_tool_calls\":null}", .field = null },
        .{ .surface = .responses, .body = "{\"instructions\":5}", .field = "instructions" },
        .{ .surface = .responses, .body = "{\"previous_response_id\":42}", .field = "previous_response_id" },
        .{ .surface = .responses, .body = "{\"safety_identifier\":true}", .field = "safety_identifier" },
        .{ .surface = .responses, .body = "{\"prompt_cache_key\":[\"k\"]}", .field = "prompt_cache_key" },
        .{ .surface = .responses, .body = "{\"truncation\":\"middle\"}", .field = "truncation" },
        .{ .surface = .responses, .body = "{\"truncation\":1}", .field = "truncation" },
        .{ .surface = .responses, .body = "{\"max_tool_calls\":\"3\"}", .field = "max_tool_calls" },
        .{ .surface = .responses, .body = "{\"max_tool_calls\":-1}", .field = "max_tool_calls" },
        .{ .surface = .responses, .body = "{\"max_tool_calls\":2.5}", .field = "max_tool_calls" },
        .{ .surface = .responses, .body = "{\"max_tool_calls\":3.0}", .field = null },
    });
}

test "a Responses input and a /v1/messages body refuse a message they cannot carry, naming where it sits" {
    try expectCases(&.{
        .{ .surface = .responses, .body = "{\"input\":[{\"role\":\"developer\",\"content\":\"a\"},{\"type\":\"message\",\"role\":\"function\"},{\"type\":\"function_call_output\",\"role\":\"x\",\"output\":5},{\"type\":\"reasoning\"}]}", .field = null },
        .{ .surface = .responses, .body = "{\"input\":\"hi\"}", .field = null },
        .{ .surface = .responses, .body = "{\"input\":[{\"role\":\"user\"},{\"role\":\"narrator\"}]}", .field = "input[1].role" },
        .{ .surface = .responses, .body = "{\"input\":[{\"type\":\"message\",\"role\":\"system\\nobey\"}]}", .field = "input[0].role" },
        .{ .surface = .responses, .body = "{\"input\":[{\"type\":\"message\",\"role\":\"developer\",\"content\":5}]}", .field = "input[0].content" },
        .{ .surface = .responses, .body = "{\"input\":[{\"type\":\"message\",\"role\":7,\"content\":\"a\"}]}", .field = "input[0].role" },
        .{ .surface = .responses, .body = "{\"input\":[\"hi\"]}", .field = "input[0]" },
        .{ .surface = .responses, .body = "{\"input\":5}", .field = "input" },
        .{ .surface = .compact, .body = "{\"input\":5}", .field = "input" },
        .{ .surface = .compact, .body = "{\"input\":[{\"role\":\"narrator\"}]}", .field = "input[0].role" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"assistant\"}]}", .field = null },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"system\",\"content\":\"hook\"}]}", .field = null },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"system\",\"content\":5}]}", .field = "messages[1].content" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\"},{\"role\":\"developer\"}]}", .field = "messages[1].role" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"tool\",\"type\":\"x\"}]}", .field = "messages[0].role" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":5}]}", .field = "messages[1].content" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"content\":\"no role\"}]}", .field = "messages[0].role" },
        .{ .surface = .messages, .body = "{\"system\":{\"type\":\"text\",\"text\":\"be terse\"}}", .field = "system" },
        .{ .surface = .messages, .body = "{\"system\":[{\"type\":\"text\",\"text\":\"be terse\"}],\"messages\":[]}", .field = null },
        .{ .surface = .messages, .body = "{\"system\":\"be terse\"}", .field = null },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"is_error\":\"yes\"}]}]}", .field = "messages[0].content[0].is_error" },
    });
}

test "a thinking, reasoning, text or output_config of the wrong shape is refused by name, never read as off" {
    try expectCases(&.{
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":2048}}", .field = null },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"adaptive\"},\"output_config\":{\"effort\":\"low\",\"format\":{\"type\":\"json_schema\",\"schema\":{}}}}", .field = null },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"disabled\",\"budget_tokens\":null}}", .field = null },
        .{ .surface = .messages, .body = "{\"thinking\":\"enabled\"}", .field = "thinking" },
        .{ .surface = .messages, .body = "{\"thinking\":{}}", .field = "thinking.type" },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"on\"}}", .field = "thinking.type" },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":7}}", .field = "thinking.type" },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":99999999999}}", .field = "thinking.budget_tokens" },
        .{ .surface = .messages, .body = "{\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":\"many\"}}", .field = "thinking.budget_tokens" },
        .{ .surface = .messages, .body = "{\"output_config\":\"high\"}", .field = "output_config" },
        .{ .surface = .messages, .body = "{\"output_config\":{\"format\":\"json\"}}", .field = "output_config.format" },
        .{ .surface = .messages, .body = "{\"output_config\":{\"format\":{\"type\":\"xml\"}}}", .field = "output_config.format.type" },
        .{ .surface = .responses, .body = "{\"reasoning\":{\"effort\":\"low\",\"summary\":\"auto\"},\"text\":{\"format\":{\"type\":\"json_schema\"}}}", .field = null },
        .{ .surface = .responses, .body = "{\"reasoning\":null,\"text\":{\"format\":null},\"response_format\":{\"type\":\"json_object\"}}", .field = null },
        .{ .surface = .responses, .body = "{\"reasoning\":\"high\"}", .field = "reasoning" },
        .{ .surface = .responses, .body = "{\"reasoning\":{\"summary\":\"verbose\"}}", .field = "reasoning.summary" },
        .{ .surface = .responses, .body = "{\"text\":\"json\"}", .field = "text" },
        .{ .surface = .responses, .body = "{\"text\":{\"format\":\"json\"}}", .field = "text.format" },
        .{ .surface = .responses, .body = "{\"text\":{\"format\":{\"type\":\"json_shema\"}}}", .field = "text.format.type" },
        .{ .surface = .responses, .body = "{\"text\":{\"format\":{}}}", .field = "text.format.type" },
        .{ .surface = .responses, .body = "{\"response_format\":{\"type\":\"xml\"}}", .field = "response_format.type" },
    });
}

test "kv_quant, metadata and the tools of Messages and Responses are checked where they are read" {
    try expectCases(&.{
        .{ .surface = .messages, .body = "{\"tools\":[{\"name\":\"t\",\"input_schema\":{\"type\":\"object\"}}],\"tool_choice\":{\"type\":\"tool\",\"name\":\"t\",\"disable_parallel_tool_use\":true}}", .field = null },
        .{ .surface = .messages, .body = "{\"tools\":[{\"name\":\"my tool\",\"input_schema\":{\"type\":\"object\"}}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":{\"name\":\"t\"}}", .field = "tools" },
        .{ .surface = .messages, .body = "{\"tool_choice\":\"auto\"}", .field = "tool_choice" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"any\",\"disable_parallel_tool_use\":1}}", .field = "tool_choice.disable_parallel_tool_use" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"tool\"}}", .field = "tool_choice.name" },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"required\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"any\"}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"allowed_tools\"}}", .field = "tool_choice.type" },
        .{ .surface = .responses, .body = "{\"metadata\":{\"k\":\"v\"}}", .field = null },
        .{ .surface = .responses, .body = "{\"metadata\":[]}", .field = "metadata" },
    });
}

test "the readers give the values the check accepted" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"top_k\":40.0,\"temperature\":1,\"seed\":-3,\"repetition_penalty\":1.25,\"stop\":[\"a\",\"\",\"b\"],\"max_completion_tokens\":0,\"max_tokens\":9,\"stream_options\":{\"include_usage\":true},\"reasoning_budget_tokens\":-1}", .{});
    defer parsed.deinit();
    var buf: [256]u8 = undefined;
    const req = switch (check(.chat, &buf, parsed.value.object)) {
        .ok => |c| c,
        .fault => |m| {
            std.debug.print("{s}\n", .{m});
            return error.Refused;
        },
    };
    try testing.expectEqual(@as(?u32, 40), req.topK());
    try testing.expectEqual(@as(?f32, 1), req.number("temperature"));
    try testing.expectEqual(@as(?u64, null), req.seed());
    try testing.expectEqual(@as(f32, 1.25), req.repeatPenalty());
    try testing.expectEqual(@as(f32, 0), req.additivePenalty("presence_penalty"));
    var stops: std.ArrayList([]const u8) = .empty;
    defer stops.deinit(testing.allocator);
    try req.stops(testing.allocator, &stops);
    try testing.expectEqual(@as(usize, 2), stops.items.len);
    try testing.expectEqual(@as(i64, 9), wholeNumber(req.requestedMaxTokens().?).?);
    try testing.expect(req.includeUsage());
    try testing.expectEqual(@as(?i32, -1), req.reasoningBudget());

    // `return_progress` reads as the bool the client sent, on both completion surfaces.
    var progress = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"return_progress\":true}", .{});
    defer progress.deinit();
    const chat_progress = check(.chat, &buf, progress.value.object).ok;
    try testing.expect(chat_progress.returnProgress());
    const cmpl_progress = check(.completions, &buf, progress.value.object).ok;
    try testing.expect(cmpl_progress.returnProgress());
    var off = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{}", .{});
    defer off.deinit();
    try testing.expect(!check(.chat, &buf, off.value.object).ok.returnProgress());

    var responses = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"max_tool_calls\":3.0,\"truncation\":\"auto\",\"service_tier\":\"flex\"}", .{});
    defer responses.deinit();
    const resp = check(.responses, &buf, responses.value.object).ok;
    try testing.expectEqual(@as(?i64, 3), resp.whole("max_tool_calls"));
    try testing.expectEqualStrings("auto", resp.truncation());
    try testing.expectEqualStrings("flex", resp.text("service_tier").?);
}

test "contentOf reads text, parts or nothing, and refuses any other type" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "[\"t\",[{\"type\":\"text\"}],null,5,{}]", .{});
    defer parsed.deinit();
    const items = parsed.value.array.items;
    try testing.expectEqualStrings("t", (try contentOf(items[0])).text);
    try testing.expectEqual(@as(usize, 1), (try contentOf(items[1])).parts.len);
    try testing.expect(try contentOf(items[2]) == .none);
    try testing.expect(try contentOf(null) == .none);
    try testing.expectError(error.ContentWrongType, contentOf(items[3]));
    try testing.expectError(error.ContentWrongType, contentOf(items[4]));
}

test "Messages refuses a tool_choice type it does not serve, and a tool_result flag or content that is no bool or text, by name" {
    try expectCases(&.{
        .{ .surface = .messages, .body = "{}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":null}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"auto\"}}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"any\",\"disable_parallel_tool_use\":true}}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"none\"}}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"tool\",\"name\":\"f\",\"disable_parallel_tool_use\":false}}", .field = null },
        .{ .surface = .messages, .body = "{\"tool_choice\":\"auto\"}", .field = "tool_choice" },
        .{ .surface = .messages, .body = "{\"tool_choice\":5}", .field = "tool_choice" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{}}", .field = "tool_choice.type" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"bogus\"}}", .field = "tool_choice.type" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}}", .field = "tool_choice.type" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"tool\"}}", .field = "tool_choice.name" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"tool\",\"name\":\"\"}}", .field = "tool_choice.name" },
        .{ .surface = .messages, .body = "{\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":\"yes\"}}", .field = "tool_choice.disable_parallel_tool_use" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":true}]}]}", .field = null },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":null}]}]}", .field = null },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":\"x\",\"is_error\":\"yes\"}]}]}", .field = "messages[0].content[0].is_error" },
        .{ .surface = .messages, .body = "{\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"t\",\"content\":5}]}]}", .field = "messages[0].content[0].content" },
    });
}

test "a surface's first positive token budget key wins, and a null or zero one leaves the next in force" {
    const Budget = struct { surface: Surface, body: []const u8, want: ?i64 };
    const cases = [_]Budget{
        .{ .surface = .chat, .body = "{\"max_tokens\":20,\"max_completion_tokens\":7}", .want = 7 },
        .{ .surface = .chat, .body = "{\"max_tokens\":20,\"max_completion_tokens\":null}", .want = 20 },
        .{ .surface = .chat, .body = "{\"max_tokens\":5,\"max_completion_tokens\":0}", .want = 5 },
        .{ .surface = .completions, .body = "{\"max_tokens\":5.0}", .want = 5 },
        .{ .surface = .chat, .body = "{\"max_tokens\":0,\"max_completion_tokens\":null}", .want = null },
        .{ .surface = .chat, .body = "{}", .want = null },
        .{ .surface = .responses, .body = "{\"max_output_tokens\":null,\"max_tokens\":1}", .want = 1 },
        .{ .surface = .responses, .body = "{\"max_output_tokens\":0,\"max_tokens\":1}", .want = 1 },
        .{ .surface = .responses, .body = "{\"max_output_tokens\":9,\"max_tokens\":1}", .want = 9 },
        .{ .surface = .messages, .body = "{\"max_tokens\":4}", .want = 4 },
    };
    inline for (comptime std.meta.tags(Surface)) |s| {
        if (s == .compact) continue;
        for (cases) |c| {
            if (c.surface != s) continue;
            var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
            defer parsed.deinit();
            var buf: [128]u8 = undefined;
            const req = check(s, &buf, parsed.value.object).ok;
            errdefer std.debug.print("\nbody: {s}\n", .{c.body});
            const got = req.requestedMaxTokens();
            try testing.expectEqual(c.want, if (got) |v| wholeNumber(v) else null);
        }
    }
}

test "Responses tool_choice refuses a form this server cannot honour, by name" {
    try expectCases(&.{
        .{ .surface = .responses, .body = "{}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":null}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"auto\"}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"none\"}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"required\"}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"f\"}}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"f\"}}}", .field = null },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"bogus\"}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":\"any\"}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":5}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":true}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":[]}", .field = "tool_choice" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{}}", .field = "tool_choice.type" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":5}}", .field = "tool_choice.type" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"allowed_tools\",\"mode\":\"auto\"}}", .field = "tool_choice.type" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"function\"}}", .field = "tool_choice.name" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":\"\"}}", .field = "tool_choice.name" },
        .{ .surface = .responses, .body = "{\"tool_choice\":{\"type\":\"function\",\"name\":7}}", .field = "tool_choice.name" },
    });
}

test "a seed reads as an unsigned number, a negative one is unseeded" {
    for ([_]struct { body: []const u8, want: ?u64 }{
        .{ .body = "{\"seed\":7}", .want = 7 },
        .{ .body = "{\"seed\":7.0}", .want = 7 },
        .{ .body = "{\"seed\":-1}", .want = null },
        .{ .body = "{\"seed\":null}", .want = null },
        .{ .body = "{}", .want = null },
    }) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        var buf: [128]u8 = undefined;
        try testing.expectEqual(c.want, check(.chat, &buf, parsed.value.object).ok.seed());
    }
}

test "a bool of chat_template_kwargs reads as the bool the client sent" {
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"chat_template_kwargs\":{\"enable_thinking\":false,\"n\":1}}", .{});
    defer parsed.deinit();
    var buf: [128]u8 = undefined;
    const req = check(.chat, &buf, parsed.value.object).ok;
    try testing.expectEqual(@as(?bool, false), req.templateKwarg("enable_thinking"));
    try testing.expectEqual(@as(?bool, null), req.templateKwarg("preserve_thinking"));
}

test "repeat_penalty: 0 is unset, then off; frequency and presence are additive and never a repeat multiplier" {
    const Penalties = struct { body: []const u8, repeat: f32, frequency: f32 = 0, presence: f32 = 0 };
    for ([_]Penalties{
        .{ .body = "{}", .repeat = 1.0 },
        .{ .body = "{\"repeat_penalty\":1.3}", .repeat = 1.3 },
        .{ .body = "{\"repeat_penalty\":0}", .repeat = 1.0 },
        .{ .body = "{\"frequency_penalty\":1}", .repeat = 1.0, .frequency = 1 },
        .{ .body = "{\"frequency_penalty\":0.5,\"presence_penalty\":-0.25}", .repeat = 1.0, .frequency = 0.5, .presence = -0.25 },
        .{ .body = "{\"frequency_penalty\":-2,\"presence_penalty\":2}", .repeat = 1.0, .frequency = -2, .presence = 2 },
        .{ .body = "{\"repetition_penalty\":1.3}", .repeat = 1.3 },
        .{ .body = "{\"repetition_penalty\":0}", .repeat = 1.0 },
        .{ .body = "{\"repeat_penalty\":1.2,\"repetition_penalty\":1.5}", .repeat = 1.2 },
    }) |c| {
        var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.body, .{});
        defer parsed.deinit();
        var buf: [128]u8 = undefined;
        const req = check(.chat, &buf, parsed.value.object).ok;
        try testing.expectApproxEqAbs(c.repeat, req.repeatPenalty(), 1e-6);
        try testing.expectEqual(c.frequency, req.additivePenalty("frequency_penalty"));
        try testing.expectEqual(c.presence, req.additivePenalty("presence_penalty"));
    }
}

test "a key that drops behaviour on every surface is refused by name, with the way to ask for it" {
    for ([_]Surface{ .chat, .completions, .messages, .responses }) |surface| try expectCases(&.{
        .{ .surface = surface, .body = "{\"functions\":[{\"name\":\"f\"}]}", .field = "functions" },
        .{ .surface = surface, .body = "{\"function_call\":\"auto\"}", .field = "function_call" },
        .{ .surface = surface, .body = "{\"grammar\":\"root ::= x\"}", .field = "grammar" },
        .{ .surface = surface, .body = "{\"guided_json\":{}}", .field = "guided_json" },
        .{ .surface = surface, .body = "{\"guided_regex\":\"a+\"}", .field = "guided_regex" },
        .{ .surface = surface, .body = "{\"guided_choice\":[\"a\"]}", .field = "guided_choice" },
        .{ .surface = surface, .body = "{\"guided_grammar\":\"x\"}", .field = "guided_grammar" },
        .{ .surface = surface, .body = "{\"grammar\":null,\"functions\":null,\"guided_json\":null}", .field = null },
    });
}

/// codex-cli 0.154's `tools` (a captured request, trimmed to one member): a function, a namespace, a hosted web_search.
const codex_function = "{\"type\":\"function\",\"name\":\"exec_command\",\"description\":\"Runs a command.\",\"strict\":false,\"parameters\":{\"type\":\"object\",\"properties\":{\"cmd\":{\"type\":\"string\"}},\"required\":[\"cmd\"],\"additionalProperties\":false}}";
const codex_namespace = "{\"type\":\"namespace\",\"name\":\"multi_agent_v1\",\"description\":\"Tools for spawning and managing sub-agents.\",\"tools\":[{\"type\":\"function\",\"name\":\"close_agent\",\"description\":\"Close an agent.\",\"strict\":false,\"parameters\":{\"type\":\"object\",\"properties\":{\"target\":{\"type\":\"string\"}},\"required\":[\"target\"],\"additionalProperties\":false}}]}";
const codex_web_search = "{\"type\":\"web_search\",\"external_web_access\":false}";

test "Responses serves a namespace of function tools and refuses a malformed one by name" {
    try expectCases(&.{
        .{ .surface = .responses, .body = "{\"tools\":[" ++ codex_function ++ "," ++ codex_namespace ++ "]}", .field = null },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\"}]}", .field = "tools[0].tools" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":{}}]}", .field = "tools[0].tools" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[]}]}", .field = "tools[0].tools" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]}]}", .field = "tools[0].name" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]}]}", .field = "tools[0].name" },
        .{ .surface = .responses, .body = "{\"tools\":[" ++ codex_function ++ ",{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"},7]}]}", .field = "tools[1].tools[1]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"custom\",\"name\":\"a\"}]}]}", .field = "tools[0].tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"namespace\",\"name\":\"in\",\"tools\":[]}]}]}", .field = "tools[0].tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\"}]}]}", .field = "tools[0].tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"a\",\"parameters\":\"bad\"}]}]}", .field = "tools[0].tools[0].parameters" },
        // One name is one tool: the model is shown the members beside the top-level functions.
        .{ .surface = .responses, .body = "{\"tools\":[" ++ codex_function ++ ",{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"exec_command\"}]}]}", .field = "tools[1].tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"ns\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]},{\"type\":\"function\",\"name\":\"a\"}]}", .field = "tools[1]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"namespace\",\"name\":\"one\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]},{\"type\":\"namespace\",\"name\":\"two\",\"tools\":[{\"type\":\"function\",\"name\":\"a\"}]}]}", .field = "tools[1].tools[0]" },
        // Namespaces are a Responses tool: chat and Anthropic refuse one as they refuse any other type.
        .{ .surface = .chat, .body = "{\"tools\":[" ++ codex_namespace ++ "]}", .field = "tools[0].type" },
        .{ .surface = .messages, .body = "{\"tools\":[" ++ codex_namespace ++ "]}", .field = "tools[0]" },
    });
}

test "Responses refuses a hosted web_search, naming the tool and the client setting that turns it off" {
    const body = "{\"tools\":[" ++ codex_function ++ "," ++ codex_namespace ++ "," ++ codex_web_search ++ "]}";
    try expectCases(&.{
        .{ .surface = .responses, .body = body, .field = "tools[2]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"web_search_preview\"}]}", .field = "tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":\"file_search\"}]}", .field = "tools[0]" },
        .{ .surface = .responses, .body = "{\"tools\":[{\"type\":7}]}", .field = "tools[0].type" },
    });
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    var buf: [480]u8 = undefined;
    const why = fault(&buf, .responses, parsed.value.object).?;
    try testing.expect(std.mem.indexOf(u8, why, "\"web_search\"") != null);
    try testing.expect(std.mem.indexOf(u8, why, "web_search = \"disabled\"") != null);
    try testing.expect(std.mem.indexOf(u8, why, "config.toml") != null);
}

test "Messages serves a custom tool and refuses every other typed tool, naming its type" {
    try expectCases(&.{
        .{ .surface = .messages, .body = "{\"tools\":[{\"name\":\"t\",\"input_schema\":{\"type\":\"object\"}},{\"type\":\"custom\",\"name\":\"u\",\"input_schema\":{\"type\":\"object\"}}]}", .field = null },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":\"web_search_20250305\",\"name\":\"web_search\"}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"name\":\"t\",\"input_schema\":{}},{\"type\":\"web_fetch_20250910\",\"name\":\"web_fetch\"}]}", .field = "tools[1]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":\"code_execution_20250522\",\"name\":\"code_execution\"}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":\"bash_20250124\",\"name\":\"bash\"}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":\"text_editor_20250124\",\"name\":\"str_replace_editor\"}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":\"computer_20250124\",\"name\":\"computer\",\"display_width_px\":1024}]}", .field = "tools[0]" },
        .{ .surface = .messages, .body = "{\"tools\":[{\"type\":7,\"name\":\"t\"}]}", .field = "tools[0].type" },
    });
    var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"tools\":[{\"type\":\"bash_20250124\",\"name\":\"bash\"}]}", .{});
    defer parsed.deinit();
    var buf: [320]u8 = undefined;
    const why = fault(&buf, .messages, parsed.value.object).?;
    try testing.expect(std.mem.indexOf(u8, why, "bash_20250124") != null);
}
