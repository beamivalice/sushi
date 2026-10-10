//! `sushi launch <agent>` — configure and launch a third-party coding
//! agent against the local server, ollama-style (issue #188).
//!
//! Configs normally use dedicated dirs or inline settings. OpenCode --persist
//! explicitly merges into the user config after backing up the original.
//!
//! Flow: probe the server; if it's down, print how to start one. Then read
//! `/v1/models`, derive each model's budget from its ADVERTISED context
//! (AgentBudget's formula: output = clamp(ctx/2, 1024, 65536) — never a
//! hardcoded window), write the agent's config, and exec it through a login
//! zsh so the user's PATH (nvm, Homebrew, ~/.local/bin) resolves.

const std = @import("std");
const log = @import("log.zig");
const model_mod = @import("model.zig");

pub const Budget = struct { context: u64, output: u64 };

/// Used when the server advertises no
/// context (older build, unloaded stub with no readable config).
pub const FALLBACK_BUDGET = Budget{ .context = 32768, .output = 8192 };

/// output = clamp(ctx/2, 1024, 65536).
pub fn budgetForContext(ctx: u64) Budget {
    if (ctx == 0) return FALLBACK_BUDGET;
    return .{ .context = ctx, .output = @min(65536, @max(1024, ctx / 2)) };
}

/// Room an agent keeps free before compacting, and what it keeps after: a
/// quarter of the window, capped where pi's own 20000-token defaults (sized
/// for 200k windows) take over.
pub fn compactionReserve(ctx: u64) u64 {
    return @min(20000, @max(1024, ctx / 4));
}

/// One chat-capable /v1/models row as declared to an agent CLI.
pub const Entry = struct {
    id: []const u8,
    budget: Budget,
    vision: bool,
    loaded: bool,
    /// The row's `reasoning_efforts`; null when the server lists none.
    efforts: ?[]const []const u8 = null,
    /// The row's `default_reasoning_effort`: the word a request naming none runs at.
    default_effort: ?[]const u8 = null,
};

/// pi's thinking levels, in its own order.
const pi_levels = [_][]const u8{ "off", "minimal", "low", "medium", "high", "xhigh" };
/// The server's effort vocabulary, in order; `on` ranks below every graded word.
const server_efforts = @typeInfo(model_mod.Effort).@"enum".field_names;

fn listed(words: []const []const u8, w: []const u8) bool {
    for (words) |x| if (std.mem.eql(u8, x, w)) return true;
    return false;
}

/// The accepted word a pi level reaches the server as: the level itself, else the nearest
/// accepted word above it (minimal reads as low), else below, which on an on/off model is `on`.
/// A thinking level never lands on off; null = no accepted word, which pi's map reads as an
/// unsupported level.
pub fn piEffortFor(level: []const u8, accepted: []const []const u8) ?[]const u8 {
    if (std.mem.eql(u8, level, "off")) return if (listed(accepted, "off")) "off" else null;
    const want = if (std.mem.eql(u8, level, "minimal")) "low" else level;
    var rank: usize = 1;
    while (rank < server_efforts.len and !std.mem.eql(u8, server_efforts[rank], want)) rank += 1;
    if (rank == server_efforts.len) return null;
    for (server_efforts[rank..]) |w| if (listed(accepted, w)) return w;
    var i = rank;
    while (i > 1) {
        i -= 1;
        if (listed(accepted, server_efforts[i])) return server_efforts[i];
    }
    return null;
}

/// pi's `thinkingLevelMap` for one model: every level to a word the server accepts, which pi
/// sends as `reasoning_effort`. Without a listed vocabulary only off is spelled, as `none`.
fn writePiLevelMap(allocator: std.mem.Allocator, out: *std.ArrayList(u8), efforts: ?[]const []const u8) !void {
    const accepted = efforts orelse return out.appendSlice(allocator, "{\"off\": \"none\"}");
    try out.append(allocator, '{');
    for (pi_levels, 0..) |lvl, i| {
        if (i > 0) try out.appendSlice(allocator, ", ");
        if (piEffortFor(lvl, accepted)) |w| {
            try out.print(allocator, "\"{s}\": \"{s}\"", .{ lvl, w });
        } else {
            try out.print(allocator, "\"{s}\": null", .{lvl});
        }
    }
    try out.append(allocator, '}');
}

pub const AgentKind = enum {
    claude,
    pi,
    omp,
    opencode,
    codex,
    hermes,
    aider,
    zcode,
    grok,

    pub fn fromName(name: []const u8) ?AgentKind {
        // The codex rebrand: issue #188 asks for `sushi launch chatgpt`.
        if (std.mem.eql(u8, name, "chatgpt")) return .codex;
        inline for (@typeInfo(AgentKind).@"enum".field_names, 0..) |f, i| {
            if (std.mem.eql(u8, name, f)) return @fromBackingInt(@intCast(i));
        }
        return null;
    }

    pub const names = "claude, pi, omp, opencode, codex, hermes, aider, zcode, grok";
};

pub const zcodeConfigJson = @import("zcode_launch.zig").configJson;

// ── Config builders (pure — unit-tested below) ──────────────────────────

/// `{f}` writes a string's content escaped for a JSON, TOML basic or YAML double-quoted
/// string; the caller supplies the quotes.
const Esc = struct {
    s: []const u8,

    pub fn format(self: Esc, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (self.s) |c| switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            0...9, 11...31, 127 => try w.print("\\u{x:0>4}", .{c}),
            else => try w.writeByte(c),
        };
    }
};

fn esc(s: []const u8) Esc {
    return .{ .s = s };
}

/// pi `models.json`, with every chat-capable model in the array so
/// in-session `/model` can switch (a launch-time snapshot).
pub fn piModelsJson(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\{{
        \\  "providers": {{
        \\    "sushi": {{
        \\      "baseUrl": "{f}/v1",
        \\      "api": "openai-completions",
        \\      "apiKey": "sushi",
        \\      "compat": {{
        \\        "supportsDeveloperRole": false,
        \\        "supportsReasoningEffort": true,
        \\        "maxTokensField": "max_tokens"
        \\      }},
        \\      "models": [
    , .{esc(base_url)});
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}
            \\        {{"id": "{f}", "name": "{f} (sushi)", "input": [{s}],
            \\         "contextWindow": {d}, "maxTokens": {d}, "reasoning": true, "thinkingLevelMap":
        , .{
            if (i == 0) "" else ",",
            esc(e.id),
            esc(e.id),
            if (e.vision) "\"text\", \"image\"" else "\"text\"",
            e.budget.context,
            e.budget.output,
        });
        try writePiLevelMap(allocator, &out, e.efforts);
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator,
        \\
        \\      ]
        \\    }
        \\  }
        \\}
    );
    return out.toOwnedSlice(allocator);
}

/// oh-my-pi `models.yml` — static chat-capable list, deliberately not omp's
/// openai-models-list discovery (it would put every media model in the
/// coding picker at omp's 128k default).
pub fn ompModelsYml(allocator: std.mem.Allocator, base_url: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\# written by sushi — custom `sushi` provider for oh-my-pi (omp).
        \\# Regenerated at each launch; edits here are overwritten.
        \\providers:
        \\  sushi:
        \\    baseUrl: "{f}/v1"
        \\    api: openai-completions
        \\    apiKey: sushi
        \\    compat:
        \\      streamIdleTimeoutMs: 0
        \\      supportsDeveloperRole: false
        \\      supportsReasoningEffort: true
        \\      maxTokensField: max_tokens
        \\      thinkingFormat: qwen
        \\      qwenTemplateReasoningEffort: false
        \\      whenThinking:
        \\        thinkingFormat: openai
        \\    models:
        \\
    , .{esc(base_url)});
    for (entries) |e| {
        try out.print(allocator,
            \\      - id: "{f}"
            \\        name: "{f} (sushi)"
            \\        reasoning: true
            \\
        , .{ esc(e.id), esc(e.id) });
        if (e.efforts) |accepted| try writeOmpThinking(allocator, &out, accepted);
        try out.print(allocator,
            \\        input: [{s}]
            \\        cost:
            \\          input: 0
            \\          output: 0
            \\          cacheRead: 0
            \\          cacheWrite: 0
            \\        contextWindow: {d}
            \\        maxTokens: {d}
            \\
        , .{ if (e.vision) "text, image" else "text", e.budget.context, e.budget.output });
    }
    return out.toOwnedSlice(allocator);
}

/// omp's thinking levels past off; off rides the provider's qwen dialect (`enable_thinking:
/// false`), every other level `whenThinking`'s openai `reasoning_effort`.
const omp_levels = [_][]const u8{ "minimal", "low", "medium", "high", "xhigh", "max" };

/// One model's `thinking` block: the levels that land on an accepted word, remapped where they
/// differ. `requiresEffort: false` keeps omp from clamping off to the lowest effort.
fn writeOmpThinking(allocator: std.mem.Allocator, out: *std.ArrayList(u8), accepted: []const []const u8) !void {
    // omp refuses an empty `efforts`; a model that accepts no thinking word keeps omp's defaults.
    if (piEffortFor("max", accepted) == null) return;
    try out.appendSlice(allocator, "        thinking:\n          mode: effort\n          requiresEffort: false\n          efforts: [");
    var n: usize = 0;
    for (omp_levels) |lvl| if (piEffortFor(lvl, accepted) != null) {
        try out.print(allocator, "{s}{s}", .{ if (n > 0) ", " else "", lvl });
        n += 1;
    };
    try out.appendSlice(allocator, "]\n          effortMap: {");
    n = 0;
    for (omp_levels) |lvl| if (piEffortFor(lvl, accepted)) |w| if (!std.mem.eql(u8, w, lvl)) {
        try out.print(allocator, "{s}{s}: {s}", .{ if (n > 0) ", " else "", lvl, w });
        n += 1;
    };
    try out.appendSlice(allocator, "}\n");
}

/// opencode config — carried inline via OPENCODE_CONFIG_CONTENT (merges over
/// the user's own config, no file writes).
/// `limit.output` is the room opencode keeps free before compacting (it
/// never sends max_tokens), so it carries the reserve, not the response cap.
/// opencode 2.x has no `--model` on its default command, so the model rides `model`. A row
/// listing efforts is a reasoning model: its graded words become the `variants` opencode sends
/// as `reasoning_effort`, and its `reasoning_content` is carried back.
pub fn opencodeJson(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry) ![]u8 {
    return opencodeJsonWithEffort(allocator, base_url, model, entries, null);
}

/// An explicit launch preference applies only to the selected model.
/// With no preference, retain the server/client default behavior.
pub fn opencodeJsonWithEffort(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry, requested_effort: ?[]const u8) ![]u8 {
    if (requested_effort) |effort| {
        var supported = false;
        for (entries) |e| {
            if (!std.mem.eql(u8, e.id, model)) continue;
            if (e.efforts) |words| supported = listed(words, effort);
            break;
        }
        if (!supported) return error.UnsupportedReasoningEffort;
    }
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "{{\"$schema\": \"https://opencode.ai/config.json\", \"model\": \"sushi/{f}\", ", .{esc(model)});
    // A long prefill plus a buffered tool call outlasts opencode's 5-minute request and chunk defaults.
    try out.print(allocator,
        \\"provider": {{"sushi": {{"npm": "@ai-sdk/openai-compatible", "name": "sushi (local)", "options": {{"baseURL": "{f}/v1", "timeout": 3600000, "chunkTimeout": 3600000}}, "models": {{
    , .{esc(base_url)});
    for (entries, 0..) |e, i| {
        try out.print(allocator, "{s}\"{f}\": {{\"name\": \"{f} (sushi)\", \"tool_call\": true,{s} \"limit\": {{\"context\": {d}, \"output\": {d}}}", .{
            if (i == 0) "" else ", ",
            esc(e.id),
            esc(e.id),
            if (e.vision) " \"attachment\": true," else "",
            e.budget.context,
            compactionReserve(e.budget.context),
        });
        if (e.efforts) |words| try writeOpencodeReasoning(allocator, &out, words);
        if (std.mem.eql(u8, e.id, model)) {
            if (requested_effort) |effort| {
                try out.print(allocator, ", \"options\": {{\"reasoningEffort\": \"{f}\"}}", .{esc(effort)});
            }
        }
        try out.append(allocator, '}');
    }
    try out.appendSlice(allocator, "}}}}");
    return out.toOwnedSlice(allocator);
}

fn writeOpencodeReasoning(allocator: std.mem.Allocator, out: *std.ArrayList(u8), words: []const []const u8) !void {
    try out.appendSlice(allocator, ", \"reasoning\": true, \"interleaved\": {\"field\": \"reasoning_content\"}");
    var n: usize = 0;
    for (words) |w| {
        if (std.mem.eql(u8, w, "on") or std.mem.eql(u8, w, "off")) continue;
        try out.print(allocator, "{s}\"{s}\": {{\"reasoningEffort\": \"{s}\"}}", .{ if (n == 0) ", \"variants\": {" else ", ", w, w });
        n += 1;
    }
    if (n > 0) try out.append(allocator, '}');
}

/// OpenCode 2 needs --standalone to receive this process's inline config.
/// OpenCode 1 has no such flag. Probe CLI support instead of assuming a version.
fn appendOpencodeInvocation(out: *std.ArrayList(u8), allocator: std.mem.Allocator, extras: []const []const u8) !void {
    try out.appendSlice(allocator,
        \\sushi_opencode_flags=()
        \\if opencode --help 2>&1 | grep -q -- '--standalone'; then
        \\  sushi_opencode_flags=(--standalone)
        \\fi
        \\opencode
    );
    if (extras.len > 0 and extras[0].len > 0 and extras[0][0] != '-') {
        try out.append(allocator, ' ');
        try appendQuoted(out, allocator, extras[0]);
        try out.appendSlice(allocator, " \"${sushi_opencode_flags[@]}\"");
        return appendExtras(out, allocator, extras[1..]);
    }
    try out.appendSlice(allocator, " \"${sushi_opencode_flags[@]}\"");
    try appendExtras(out, allocator, extras);
}

/// grok's `[session] auto_compact_threshold_percent` is its only compaction knob: the share of the
/// window used before it compacts, which leaves `compactionReserve` free.
pub fn grokCompactPercent(ctx: u64) u64 {
    if (ctx == 0) return 90;
    return @max(50, (ctx - compactionReserve(ctx)) * 100 / ctx);
}

fn appendTomlString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        0...9, 11...31, 127 => try out.print(allocator, "\\u{x:0>4}", .{c}),
        else => try out.append(allocator, c),
    };
    try out.append(allocator, '"');
}

/// grok `config.toml` under its own GROK_HOME: one `[model.*]` per chat row on the chat
/// completions wire, a dummy key (grok sends it as the bearer), the row's advertised context
/// and output budget, and its reasoning words as the effort picker.
pub fn grokConfigToml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, budget: Budget, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "# written by sushi, regenerated at each launch.\n[cli]\nauto_update = false\n\n[features]\ntelemetry = false\nfeedback = false\n\n[models]\ndefault = ");
    try appendTomlString(&out, allocator, model);
    try out.print(allocator, "\n\n[session]\nauto_compact_threshold_percent = {d}\n", .{grokCompactPercent(budget.context)});
    for (entries) |e| {
        try out.appendSlice(allocator, "\n[model.");
        try appendTomlString(&out, allocator, e.id);
        try out.appendSlice(allocator, "]\nmodel = ");
        try appendTomlString(&out, allocator, e.id);
        try out.print(allocator, "\nbase_url = \"{f}/v1\"\nname = ", .{esc(base_url)});
        const label = try std.fmt.allocPrint(allocator, "{s} (sushi)", .{e.id});
        defer allocator.free(label);
        try appendTomlString(&out, allocator, label);
        try out.print(allocator,
            \\
            \\api_backend = "chat_completions"
            \\api_key = "sushi"
            \\context_window = {d}
            \\max_completion_tokens = {d}
            \\
        , .{ e.budget.context, e.budget.output });
        const words = e.efforts orelse continue;
        if (words.len == 0) continue;
        const default = if (e.default_effort) |d| (if (listed(words, d)) d else words[0]) else words[0];
        try out.appendSlice(allocator, "supports_reasoning_effort = true\nreasoning_effort = ");
        try appendTomlString(&out, allocator, default);
        try out.append(allocator, '\n');
        for (words) |w| {
            try out.appendSlice(allocator, "\n[[model.");
            try appendTomlString(&out, allocator, e.id);
            try out.appendSlice(allocator, ".reasoning_efforts]]\nid = ");
            try appendTomlString(&out, allocator, w);
            try out.appendSlice(allocator, "\nvalue = ");
            try appendTomlString(&out, allocator, w);
            try out.appendSlice(allocator, "\nlabel = ");
            try appendTomlString(&out, allocator, w);
            try out.print(allocator, "\ndefault = {s}\n", .{if (std.mem.eql(u8, w, default)) "true" else "false"});
        }
    }
    return out.toOwnedSlice(allocator);
}

/// pi `settings.json`: compaction numbers scaled to the window, everything
/// else (theme, packages, the user's own `enabled`) kept. pi compacts when
/// context exceeds window - reserveTokens and keeps keepRecentTokens; its
/// defaults (16384 / 20000) never compact a 24k window while max_tokens
/// shrinks to 1.
pub fn mergePiSettingsJson(allocator: std.mem.Allocator, existing: []const u8, ctx: u64) ![]u8 {
    const trimmed = std.mem.trim(u8, existing, " \t\r\n");
    const body = if (trimmed.len == 0) "{}" else existing;
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch
        try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    if (parsed.value != .object) {
        parsed.deinit();
        parsed = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    }
    defer parsed.deinit();
    const a = parsed.arena.allocator();

    var compaction: std.json.ObjectMap = .empty;
    if (parsed.value.object.get("compaction")) |c| {
        if (c == .object) compaction = c.object;
    }
    const reserve = compactionReserve(ctx);
    try compaction.put(a, "reserveTokens", .{ .integer = @intCast(@min(16384, reserve + 4096)) });
    try compaction.put(a, "keepRecentTokens", .{ .integer = @intCast(reserve) });

    var obj = parsed.value.object;
    try obj.put(a, "compaction", .{ .object = compaction });
    parsed.value = .{ .object = obj };
    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

/// codex `config.toml` — Responses wire API only (codex-rs `WireApi` has one
/// variant), pointing at our /v1/responses. Keyless: no `env_key` and
/// `requires_openai_auth` unset means codex skips login; the loopback server
/// ignores keys anyway.
pub fn codexConfigToml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, budget: Budget) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\# written by sushi — dedicated CODEX_HOME, regenerated at each launch.
        \\model = "{f}"
        \\model_provider = "sushi"
        \\model_context_window = {d}
        \\
        \\[model_providers.sushi]
        \\name = "sushi (local)"
        \\base_url = "{f}/v1"
        \\wire_api = "responses"
        \\
    , .{ esc(model), budget.context, esc(base_url) });
}

/// hermes `config.yaml` — mirrors what `hermes setup`'s custom-endpoint flow
/// saves (verified against hermes_cli source).
pub fn hermesConfigYaml(allocator: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.print(allocator,
        \\# written by sushi — regenerated at each launch. Mirrors what
        \\# `hermes setup`'s custom-endpoint flow saves, so the first run starts
        \\# configured instead of launching the wizard.
        \\model:
        \\  default: "{f}"
        \\  provider: custom
        \\  base_url: "{f}/v1"
        \\  api_key: "sushi"
        \\  api_mode: chat_completions
        \\custom_providers:
        \\  - name: sushi
        \\    base_url: "{f}/v1"
        \\    api_key: "sushi"
        \\    model: "{f}"
        \\    api_mode: chat_completions
        \\    models:
        \\
    , .{ esc(model), esc(base_url), esc(base_url), esc(model) });
    for (entries) |e| {
        try out.print(allocator, "      \"{f}\":\n        context_length: {d}\n", .{ esc(e.id), e.budget.context });
    }
    return out.toOwnedSlice(allocator);
}

/// hermes `.env` — the first-run wizard kill switch: OPENAI_BASE_URL alone
/// marks a provider as configured. Lives under HERMES_HOME like config.yaml.
pub fn hermesEnvFile(allocator: std.mem.Allocator, base_url: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\# written by sushi — OPENAI_BASE_URL marks a provider as configured,
        \\# which is what keeps the first-run setup wizard out of the session.
        \\OPENAI_BASE_URL={s}/v1
        \\OPENAI_API_KEY=sushi
        \\
    , .{base_url});
}

/// aider model metadata (litellm's registry format) — the real context
/// window for every openai/<id> model.
pub fn aiderMetadataJson(allocator: std.mem.Allocator, entries: []const Entry) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, "{\n");
    for (entries, 0..) |e, i| {
        try out.print(allocator,
            \\{s}  "openai/{f}": {{
            \\    "max_input_tokens": {d},
            \\    "max_output_tokens": {d},
            \\    "max_tokens": {d},
            \\    "input_cost_per_token": 0,
            \\    "output_cost_per_token": 0,
            \\    "litellm_provider": "openai",
            \\    "mode": "chat"
            \\  }}
        , .{ if (i == 0) "" else ",\n", esc(e.id), e.budget.context, e.budget.output, e.budget.output });
    }
    try out.appendSlice(allocator, "\n}\n");
    return out.toOwnedSlice(allocator);
}

// ── Launch script assembly ──────────────────────────────────────────────

/// Shell-quote one extra passthrough arg (single quotes, '\'' escape).
fn appendQuoted(out: *std.ArrayList(u8), allocator: std.mem.Allocator, arg: []const u8) !void {
    try out.append(allocator, '\'');
    for (arg) |c| {
        if (c == '\'') try out.appendSlice(allocator, "'\\''") else try out.append(allocator, c);
    }
    try out.append(allocator, '\'');
}

/// One shell word: `prefix` + `arg` + `suffix`, single-quoted as a whole.
fn quotedAlloc(allocator: std.mem.Allocator, prefix: []const u8, arg: []const u8, suffix: []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    const word = try std.mem.concat(allocator, u8, &.{ prefix, arg, suffix });
    defer allocator.free(word);
    try appendQuoted(&out, allocator, word);
    return out.toOwnedSlice(allocator);
}

fn appendExtras(out: *std.ArrayList(u8), allocator: std.mem.Allocator, extras: []const []const u8) !void {
    for (extras) |a| {
        try out.append(allocator, ' ');
        try appendQuoted(out, allocator, a);
    }
}

/// The script body run through `/bin/zsh -l -c` (login shell = the user's
/// real PATH). Configs are written by `writeConfigs` BEFORE this runs; the
/// script only exports env and execs the agent.
/// Below this the agent's own fixed prompt leaves every turn compacting or
/// truncated: Claude Code sends 40-70k before the first word (tool + MCP
/// schemas, skills catalogue), opencode ~8k, pi ~2k.
pub fn contextFloor(kind: AgentKind) u64 {
    return switch (kind) {
        .claude => 65536,
        .opencode => 32768,
        else => 16384,
    };
}

pub fn scriptFor(allocator: std.mem.Allocator, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, opencode_config: ?[]const u8, extras: []const []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    if (budget.context > 0 and budget.context < contextFloor(kind)) {
        try out.print(allocator, "echo 'sushi: the model advertises a {d}-token context; {s} needs {d}+ to work well (raise --ctx-size or Settings > Server > Context size).' >&2\n", .{ budget.context, @tagName(kind), contextFloor(kind) });
    }
    const q_base = try quotedAlloc(allocator, "", base_url, "");
    defer allocator.free(q_base);
    const q_model = try quotedAlloc(allocator, "", model, "");
    defer allocator.free(q_model);
    const q_sushi_model = try quotedAlloc(allocator, "sushi/", model, "");
    defer allocator.free(q_sushi_model);
    const q_openai_model = try quotedAlloc(allocator, "openai/", model, "");
    defer allocator.free(q_openai_model);
    const q_api_base = try quotedAlloc(allocator, "", base_url, "/v1");
    defer allocator.free(q_api_base);
    const q_config = try quotedAlloc(allocator, "", opencode_config orelse "", "");
    defer allocator.free(q_config);
    switch (kind) {
        .claude => {
            try out.print(allocator,
                \\export ANTHROPIC_BASE_URL={s}
                \\export ANTHROPIC_API_KEY=
                \\export ANTHROPIC_AUTH_TOKEN=sushi
                \\export CLAUDE_CODE_ATTRIBUTION_HEADER=0
                \\export ANTHROPIC_DEFAULT_OPUS_MODEL={s}
                \\export ANTHROPIC_DEFAULT_SONNET_MODEL={s}
                \\export ANTHROPIC_DEFAULT_HAIKU_MODEL={s}
                \\export CLAUDE_CODE_SUBAGENT_MODEL={s}
                \\export CLAUDE_CODE_MAX_OUTPUT_TOKENS={d}
                \\export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
                \\export API_TIMEOUT_MS=3600000
                \\export CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=1800000
                \\export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
                \\export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=1800000
                \\
            , .{ q_base, q_model, q_model, q_model, q_model, budget.output });
            // A long prefill and a long think on a local model outlast Claude Code's stream watchdogs; a fallback
            // re-sends the whole prompt as a non-stream request, which then times out and retries.
            // Claude Code assumes 200k for a model outside its catalog; declare the advertised context verbatim.
            if (budget.context > 0) {
                try out.print(allocator, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS={d}\n", .{budget.context});
            }
            try out.print(allocator, "claude --model {s}", .{q_model});
        },
        .pi => {
            try out.print(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.sushi/pi"
                \\pi --provider sushi --model {s}
            , .{q_model});
        },
        .omp => {
            // omp still reads pi's env spelling (measured on v17 — the OMP_
            // rename reached only its help text); export both.
            try out.print(allocator,
                \\export PI_CODING_AGENT_DIR="$HOME/.sushi/omp"
                \\export OMP_CODING_AGENT_DIR="$HOME/.sushi/omp"
                \\omp --model {s}
            , .{q_sushi_model});
        },
        .opencode => {
            try out.print(allocator, "export OPENCODE_CONFIG_CONTENT={s}\n", .{q_config});
            try appendOpencodeInvocation(&out, allocator, extras);
            try out.append(allocator, '\n');
            return out.toOwnedSlice(allocator);
        },
        .grok => {
            try out.appendSlice(allocator,
                \\export GROK_HOME="$HOME/.sushi/grok"
                \\if ! command -v grok >/dev/null 2>&1; then echo "grok is not installed" >&2; exit 127; fi
                \\grok --model
            );
            try out.append(allocator, ' ');
            try out.appendSlice(allocator, q_model);
        },
        .codex => {
            // PATH first, then the CLI the desktop app bundles (codex's
            // rebranded app installs as ChatGPT.app or Codex.app, bundle id
            // com.openai.codex, CLI at Contents/Resources/codex) — a
            // desktop-app-only user has no codex on PATH.
            try out.appendSlice(allocator,
                \\export CODEX_HOME="$HOME/.sushi/codex"
                \\CODEX_BIN="$(command -v codex)"
                \\if [ -z "$CODEX_BIN" ]; then
                \\  for app in "/Applications/ChatGPT.app" "/Applications/Codex.app" "$HOME/Applications/ChatGPT.app" "$HOME/Applications/Codex.app"; do
                \\    if [ -x "$app/Contents/Resources/codex" ]; then CODEX_BIN="$app/Contents/Resources/codex"; break; fi
                \\  done
                \\fi
                \\if [ -z "$CODEX_BIN" ]; then echo "codex is not installed: npm install -g @openai/codex, or install the ChatGPT app"; exit 127; fi
                \\"$CODEX_BIN"
            );
        },
        .hermes => {
            try out.appendSlice(allocator,
                \\export HERMES_HOME="$HOME/.sushi/hermes"
                \\hermes
            );
        },
        .zcode => {
            try out.appendSlice(allocator,
                \\export ZCODE_DATA_BASE_DIR="$HOME/.sushi/zcode"
                \\export ZCODE_STORAGE_DIR="$HOME/.sushi/zcode/storage"
                \\export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="$HOME/.sushi/zcode/provider_config.json"
                \\if ! command -v zcode >/dev/null 2>&1; then echo "zcode is not installed: build or install ZCode (https://github.com/zai-org/ZCode)" >&2; exit 127; fi
                \\zcode
            );
        },
        .aider => {
            try out.print(allocator,
                \\export OPENAI_API_BASE={s}
                \\export OPENAI_API_KEY=sushi
                \\aider --model {s} --weak-model {s} --model-metadata-file ~/.sushi/aider/model-metadata.json
            , .{ q_api_base, q_openai_model, q_openai_model });
        },
    }
    try appendExtras(&out, allocator, extras);
    try out.append(allocator, '\n');
    return out.toOwnedSlice(allocator);
}

// ── Server discovery / model pick ───────────────────────────────────────

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

fn curlGet(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
    // Plain fetch, no HF token header — this talks to OUR server, never HF.
    const result = std.process.run(allocator, io, .{
        .argv = &.{ "curl", "-fsS", "-m", "5", url },
        .stdout_limit = .limited(16 * 1024 * 1024),
    }) catch return error.FetchFailed;
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);
    switch (result.term) {
        .exited => |code| if (code != 0) return error.FetchFailed,
        else => return error.FetchFailed,
    }
    return result.stdout;
}

fn serverUp(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8) bool {
    const url = std.fmt.allocPrint(allocator, "{s}/health", .{base_url}) catch return false;
    defer allocator.free(url);
    const body = curlGet(allocator, io, url) catch return false;
    allocator.free(body);
    return true;
}

const Models = struct {
    arena: std.heap.ArenaAllocator,
    entries: []Entry,

    fn deinit(self: *Models) void {
        self.arena.deinit();
    }
};

/// Parse /v1/models into the chat-capable entries (media/embedding models
/// never enter a coding agent's picker). Context comes from meta.context_length,
/// falling back to the top-level twin.
fn fetchChatEntries(allocator: std.mem.Allocator, io: std.Io, base_url: []const u8) !Models {
    const url = try std.fmt.allocPrint(allocator, "{s}/v1/models", .{base_url});
    defer allocator.free(url);
    const body = try curlGet(allocator, io, url);
    defer allocator.free(body);
    return parseChatEntries(allocator, body);
}

fn parseChatEntries(allocator: std.mem.Allocator, body: []const u8) !Models {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return error.BadModelsJson;
    const data = switch (parsed) {
        .object => |o| o.get("data") orelse return error.BadModelsJson,
        else => return error.BadModelsJson,
    };
    if (data != .array) return error.BadModelsJson;

    var list = std.ArrayList(Entry).empty;
    for (data.array.items) |row| {
        if (row != .object) continue;
        const obj = row.object;
        const id_val = obj.get("id") orelse continue;
        if (id_val != .string or id_val.string.len == 0) continue;

        // Chat-capable only; a row with no capabilities key is an old build
        // that serves chat.
        var chat = true;
        var vision = false;
        if (obj.get("capabilities")) |caps| {
            if (caps == .array) {
                chat = caps.array.items.len == 0;
                for (caps.array.items) |c| {
                    if (c != .string) continue;
                    if (std.mem.eql(u8, c.string, "chat")) chat = true;
                    if (std.mem.eql(u8, c.string, "vision")) vision = true;
                    if (std.mem.eql(u8, c.string, "embeddings")) chat = false;
                }
            }
        }
        if (!chat) continue;

        var ctx: u64 = 0;
        if (obj.get("meta")) |meta| {
            if (meta == .object) {
                if (meta.object.get("context_length")) |v| {
                    if (v == .integer and v.integer > 0) ctx = @intCast(v.integer);
                }
            }
        }
        if (ctx == 0) {
            if (obj.get("context_length")) |v| {
                if (v == .integer and v.integer > 0) ctx = @intCast(v.integer);
            }
        }
        var loaded = false;
        if (obj.get("loaded")) |v| loaded = v == .bool and v.bool;
        var efforts: ?[]const []const u8 = null;
        if (obj.get("reasoning_efforts")) |v| if (v == .array) {
            var words = std.ArrayList([]const u8).empty;
            for (v.array.items) |w| if (w == .string) try words.append(a, try a.dupe(u8, w.string));
            efforts = try words.toOwnedSlice(a);
        };

        try list.append(a, .{
            .id = try a.dupe(u8, id_val.string),
            .budget = budgetForContext(ctx),
            .vision = vision,
            .loaded = loaded,
            .efforts = efforts,
            .default_effort = if (obj.get("default_reasoning_effort")) |v| (if (v == .string) try a.dupe(u8, v.string) else null) else null,
        });
    }
    return .{ .arena = arena, .entries = try list.toOwnedSlice(a) };
}

// ── Config writes ───────────────────────────────────────────────────────

fn writeAgentFile(allocator: std.mem.Allocator, io: std.Io, subdir: []const u8, name: []const u8, content: []const u8) !void {
    const dir_path = try std.fmt.allocPrint(allocator, "{s}/.sushi/{s}", .{ homeDir(), subdir });
    defer allocator.free(dir_path);
    try std.Io.Dir.cwd().createDirPath(io, dir_path);
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = name, .data = content });
}

/// Write the agent's config files. opencode
/// carries its config inline and writes nothing.
fn writeConfigs(allocator: std.mem.Allocator, io: std.Io, kind: AgentKind, base_url: []const u8, model: []const u8, budget: Budget, entries: []const Entry) !void {
    switch (kind) {
        .claude, .opencode => {},
        .grok => {
            const toml = try grokConfigToml(allocator, base_url, model, budget, entries);
            defer allocator.free(toml);
            try writeAgentFile(allocator, io, "grok", "config.toml", toml);
        },
        .pi => {
            const json = try piModelsJson(allocator, base_url, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "pi", "models.json", json);
            const settings_path = try std.fmt.allocPrint(allocator, "{s}/.sushi/pi/settings.json", .{homeDir()});
            defer allocator.free(settings_path);
            const existing = std.Io.Dir.cwd().readFileAlloc(io, settings_path, allocator, .limited(1 << 20)) catch
                try allocator.dupe(u8, "{}");
            defer allocator.free(existing);
            const settings = try mergePiSettingsJson(allocator, existing, budget.context);
            defer allocator.free(settings);
            try writeAgentFile(allocator, io, "pi", "settings.json", settings);
        },
        .omp => {
            const yml = try ompModelsYml(allocator, base_url, entries);
            defer allocator.free(yml);
            try writeAgentFile(allocator, io, "omp", "models.yml", yml);
        },
        .codex => {
            const toml = try codexConfigToml(allocator, base_url, model, budget);
            defer allocator.free(toml);
            try writeAgentFile(allocator, io, "codex", "config.toml", toml);
        },
        .hermes => {
            const yaml = try hermesConfigYaml(allocator, base_url, model, entries);
            defer allocator.free(yaml);
            try writeAgentFile(allocator, io, "hermes", "config.yaml", yaml);
            const env = try hermesEnvFile(allocator, base_url);
            defer allocator.free(env);
            try writeAgentFile(allocator, io, "hermes", ".env", env);
        },
        .zcode => {
            const json = try zcodeConfigJson(allocator, base_url, model, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "zcode", "provider_config.json", json);
        },
        .aider => {
            const json = try aiderMetadataJson(allocator, entries);
            defer allocator.free(json);
            try writeAgentFile(allocator, io, "aider", "model-metadata.json", json);
        },
    }
}

// ── Command entry ───────────────────────────────────────────────────────

const LaunchArgs = struct {
    kind: AgentKind,
    model: ?[]const u8 = null,
    url: ?[]const u8 = null,
    port: u16 = 12345,
    print_only: bool = false,
    think: ?[]const u8 = null,
    persist: bool = false,
    extras: []const []const u8 = &.{},
};

fn parseLaunchArgs(args: []const []const u8) !LaunchArgs {
    if (args.len == 0) return error.Usage;
    const kind = AgentKind.fromName(args[0]) orelse return error.UnknownAgent;
    var out = LaunchArgs{ .kind = kind };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--")) {
            out.extras = args[i + 1 ..];
            break;
        } else if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.model = args[i];
        } else if (std.mem.eql(u8, arg, "--url")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.url = std.mem.trimEnd(u8, args[i], "/");
        } else if (std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.port = std.fmt.parseInt(u16, args[i], 10) catch return error.Usage;
        } else if (std.mem.eql(u8, arg, "--think")) {
            i += 1;
            if (i >= args.len) return error.Usage;
            out.think = args[i];
        } else if (std.mem.eql(u8, arg, "--persist")) {
            out.persist = true;
        } else if (std.mem.eql(u8, arg, "--print")) {
            out.print_only = true;
        } else if (std.mem.eql(u8, arg, "--no-start")) {
            // Accepted for existing scripts: a down server is never auto-started.
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            return error.Usage;
        } else {
            return error.Usage;
        }
    }
    if ((out.think != null or out.persist) and out.kind != .opencode) return error.Usage;
    return out;
}

fn printLaunchUsage() void {
    log.err(
        \\usage: sushi launch <agent> [options] [-- <extra agent args>]
        \\
        \\agents: {s}
        \\
        \\options:
        \\  --model <id>   Serve this model (default: the server's default model)
        \\  --url <base>   Server base URL (default: http://127.0.0.1:<port>)
        \\  --port <n>     Server port for the default URL (default: 12345)
        \\  --think <word> OpenCode only: explicitly request a server-advertised
        \\                 reasoning effort (including off when supported)
        \\  --persist      OpenCode only: back up and merge into the user config
        \\  --print        Write the config files and print the launch script
        \\                 instead of running the agent
        \\
        \\Anything after `--` is passed to the agent, e.g.:
        \\  sushi launch codex -- resume
        \\
    , .{AgentKind.names});
}

pub fn cmdLaunch(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    const parsed = parseLaunchArgs(args) catch |err| {
        switch (err) {
            error.UnknownAgent => log.err("unknown agent '{s}' — supported: {s}\n", .{ args[0], AgentKind.names }),
            else => {},
        }
        printLaunchUsage();
        std.process.exit(1);
    };

    var url_buf: [64]u8 = undefined;
    const base_url = parsed.url orelse std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}", .{parsed.port}) catch unreachable;

    if (!serverUp(allocator, io, base_url)) {
        log.err("no sushi server at {s}.\n", .{base_url});
        log.err("start one first:  sushi serve   (or: sushi run <model>)\n", .{});
        std.process.exit(1);
    }

    // The server may still be scanning/loading right after boot — poll until
    // a chat-capable model shows up (a stub is fine: the first request
    // hot-loads it).
    var models: Models = undefined;
    var polls: usize = 0;
    while (true) : (polls += 1) {
        models = fetchChatEntries(allocator, io, base_url) catch |err| {
            log.err("could not read {s}/v1/models: {s}\n", .{ base_url, @errorName(err) });
            std.process.exit(1);
        };
        if (models.entries.len > 0) break;
        models.deinit();
        if (polls >= 30) {
            log.err("no chat-capable model on {s} — pull one first (sushi pull <model>)\n", .{base_url});
            std.process.exit(1);
        }
        std.Io.sleep(io, .fromMilliseconds(1000), .real) catch {};
    }
    defer models.deinit();

    // Pick: --model must exist on the server; default = first loaded chat
    // row (/v1/models sorts the default first), else the first chat row.
    var pick: ?Entry = null;
    if (parsed.model) |want| {
        for (models.entries) |e| {
            if (std.mem.eql(u8, e.id, want)) pick = e;
        }
        if (pick == null) {
            log.err("model '{s}' is not on {s} — available:\n", .{ want, base_url });
            for (models.entries) |e| log.err("  {s}\n", .{e.id});
            std.process.exit(1);
        }
    } else {
        for (models.entries) |e| {
            if (e.loaded) {
                pick = e;
                break;
            }
        }
        if (pick == null) pick = models.entries[0];
    }
    const chosen = pick.?;

    writeConfigs(allocator, io, parsed.kind, base_url, chosen.id, chosen.budget, models.entries) catch |err| {
        log.err("could not write the {s} config: {s}\n", .{ @tagName(parsed.kind), @errorName(err) });
        std.process.exit(1);
    };

    const oc_config: ?[]u8 = if (parsed.kind == .opencode)
        opencodeJsonWithEffort(allocator, base_url, chosen.id, models.entries, parsed.think) catch |err| {
            if (err != error.UnsupportedReasoningEffort) return err;
            log.err("reasoning effort '{s}' is not supported by {s}; accepted:", .{ parsed.think orelse "", chosen.id });
            if (chosen.efforts) |words| {
                for (words) |word| log.err(" {s}", .{word});
            }
            log.err("\n", .{});
            std.process.exit(1);
        }
    else
        null;
    defer if (oc_config) |c| allocator.free(c);

    if (parsed.persist) {
        const config_root = if (std.c.getenv("XDG_CONFIG_HOME")) |v| std.mem.span(v) else null;
        if (config_root) |root| {
            if (!std.fs.path.isAbsolute(root)) return error.InvalidConfigDirectory;
        }
        const dir_path = if (config_root) |root|
            try std.fmt.allocPrint(allocator, "{s}/opencode", .{root})
        else
            try std.fmt.allocPrint(allocator, "{s}/.config/opencode", .{homeDir()});
        defer allocator.free(dir_path);
        @import("opencode_persist.zig").save(allocator, io, dir_path, oc_config.?, models.entries) catch |err| {
            log.err("could not persist OpenCode config: {s}\n", .{@errorName(err)});
            return err;
        };
        log.info("saved OpenCode settings in {s} (original backed up when changed)\n", .{dir_path});
    }

    const script = try scriptFor(allocator, parsed.kind, base_url, chosen.id, chosen.budget, oc_config, parsed.extras);
    defer allocator.free(script);

    if (parsed.print_only) {
        var stdout_buf: [8192]u8 = undefined;
        var stdout_w = std.Io.File.stdout().writer(io, &stdout_buf);
        stdout_w.interface.writeAll(script) catch {};
        stdout_w.interface.flush() catch {};
        return;
    }

    log.info("launching {s} with {s} ({d}K context) via {s}\n", .{
        @tagName(parsed.kind), chosen.id, chosen.budget.context / 1024, base_url,
    });
    var child = std.process.spawn(io, .{
        .argv = &.{ "/bin/zsh", "-l", "-c", script },
        .stdin = .inherit,
        .stdout = .inherit,
        .stderr = .inherit,
    }) catch {
        log.err("could not start /bin/zsh\n", .{});
        std.process.exit(1);
    };
    const term = child.wait(io) catch std.process.exit(1);
    switch (term) {
        .exited => |code| std.process.exit(code),
        else => std.process.exit(1),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

const t = std.testing;

test "budgetForContext mirrors AgentBudget: ctx/2 clamped to [1024, 65536], 0 = fallback" {
    // Thinking shares the response cap: a 24k window at ctx/4 gave pi 6144,
    // which one xhigh design turn on Qwen3.8 spent entirely on thinking.
    try t.expectEqual(FALLBACK_BUDGET, budgetForContext(0));
    try t.expectEqual(Budget{ .context = 4096, .output = 2048 }, budgetForContext(4096));
    try t.expectEqual(Budget{ .context = 2048, .output = 1024 }, budgetForContext(2048));
    try t.expectEqual(Budget{ .context = 24576, .output = 12288 }, budgetForContext(24576));
    try t.expectEqual(Budget{ .context = 90112, .output = 45056 }, budgetForContext(90112));
    try t.expectEqual(Budget{ .context = 1048576, .output = 65536 }, budgetForContext(1048576));
}

test "omp models.yml: static per-model entries, no discovery, pi-compat vocabulary" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
        .{ .id = "m2", .budget = .{ .context = 262144, .output = 65536 }, .vision = true, .loaded = false },
    };
    const yml = try ompModelsYml(t.allocator, "http://127.0.0.1:12345", &entries);
    defer t.allocator.free(yml);
    try t.expect(std.mem.indexOf(u8, yml, "discovery") == null);
    try t.expect(std.mem.indexOf(u8, yml, "baseUrl: \"http://127.0.0.1:12345/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, yml, "contextWindow: 4096") != null);
    try t.expect(std.mem.indexOf(u8, yml, "contextWindow: 262144") != null);
    try t.expect(std.mem.indexOf(u8, yml, "input: [text, image]") != null);
    try t.expect(std.mem.indexOf(u8, yml, "thinkingFormat: qwen") != null);
}

test "omp models.yml: buffered tool calls have no provider progress deadline" {
    const entries = [_]Entry{.{ .id = "local", .budget = .{ .context = 131072, .output = 32768 }, .vision = false, .loaded = true }};
    const yml = try ompModelsYml(t.allocator, "http://127.0.0.1:12345", &entries);
    defer t.allocator.free(yml);
    try t.expect(std.mem.indexOf(u8, yml, "    compat:\n      streamIdleTimeoutMs: 0\n") != null);
    const script = try scriptFor(t.allocator, .omp, "http://127.0.0.1:12345", "local", entries[0].budget, null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "STREAM_IDLE_TIMEOUT") == null);
}

test "codex config: responses wire API, keyless, context at the root" {
    const toml = try codexConfigToml(t.allocator, "http://127.0.0.1:12345", "m1", .{ .context = 90112, .output = 22528 });
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "wire_api = \"responses\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "model_context_window = 90112") != null);
    try t.expect(std.mem.indexOf(u8, toml, "base_url = \"http://127.0.0.1:12345/v1\"") != null);
    try t.expect(std.mem.indexOf(u8, toml, "env_key") == null);
}

test "pi models.json and opencode config parse as JSON and stay single-quote-free" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = true, .loaded = true },
        .{ .id = "m2", .budget = .{ .context = 8192, .output = 2048 }, .vision = false, .loaded = false },
    };
    const pi_json = try piModelsJson(t.allocator, "http://127.0.0.1:12345", &entries);
    defer t.allocator.free(pi_json);
    const oc_json = try opencodeJson(t.allocator, "http://127.0.0.1:12345", "m1", &entries);
    defer t.allocator.free(oc_json);
    for ([_][]const u8{ pi_json, oc_json }) |json| {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
        defer parsed.deinit();
        // opencode's config rides single-quoted inside the launch script.
        try t.expect(std.mem.indexOf(u8, json, "'") == null);
    }
}

/// A served arch's `/v1/models` `reasoning_efforts`, read from the engine's own table.
fn advertised(comptime model_type: []const u8) [model_mod.effortArms(model_type).?.len][]const u8 {
    const arms = comptime model_mod.effortArms(model_type).?;
    var words: [arms.len][]const u8 = undefined;
    for (arms, 0..) |a, i| words[i] = @tagName(a.effort);
    return words;
}
const qwen4_efforts = advertised("qwen4_exp");
const mimo_efforts = advertised("mimo_v2");
const glm5_efforts = advertised("glm5_next");

test "piEffortFor: every pi level lands on a word the model accepts" {
    const want_qwen = [_][]const u8{ "off", "low", "low", "medium", "xhigh", "xhigh" };
    const want_mimo = [_][]const u8{ "off", "on", "on", "on", "on", "on" };
    const want_glm = [_]?[]const u8{ null, "low", "low", "high", "high", "max" };
    for (pi_levels, want_qwen, want_mimo, want_glm) |lvl, q, m, g| {
        try t.expectEqualStrings(q, piEffortFor(lvl, &qwen4_efforts).?);
        try t.expectEqualStrings(m, piEffortFor(lvl, &mimo_efforts).?);
        if (g) |w| try t.expectEqualStrings(w, piEffortFor(lvl, &glm5_efforts).?) else try t.expect(piEffortFor(lvl, &glm5_efforts) == null);
    }
    // Rounds up first, then down, and a thinking level never becomes off.
    const only_low = [_][]const u8{ "off", "low" };
    try t.expectEqualStrings("low", piEffortFor("xhigh", &only_low).?);
    try t.expect(piEffortFor("off", &[_][]const u8{"low"}) == null);
    try t.expect(piEffortFor("low", &[_][]const u8{"off"}) == null);
}

test "pi models.json sends each thinking level as reasoning_effort the model accepts" {
    // thinkingFormat "qwen" made pi send only enable_thinking, so low/medium never reached the server.
    const entries = [_]Entry{
        .{ .id = "qwen", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true, .efforts = &qwen4_efforts },
        .{ .id = "mimo", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true, .efforts = &mimo_efforts },
        .{ .id = "old", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
    };
    const json = try piModelsJson(t.allocator, "http://127.0.0.1:12345", &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const p = parsed.value.object.get("providers").?.object.get("sushi").?.object;
    try t.expect(p.get("compat").?.object.get("thinkingFormat") == null);
    const models = p.get("models").?.array.items;
    for (models[0..2], [_][]const []const u8{ &qwen4_efforts, &mimo_efforts }) |m, accepted| {
        const map = m.object.get("thinkingLevelMap").?.object;
        for (pi_levels) |lvl| try t.expect(listed(accepted, map.get(lvl).?.string));
    }
    try t.expectEqualStrings("xhigh", models[0].object.get("thinkingLevelMap").?.object.get("high").?.string);
    // No listed vocabulary: pi's own words, with off spelled as the server's none.
    const old = models[2].object.get("thinkingLevelMap").?.object;
    try t.expectEqual(@as(usize, 1), old.count());
    try t.expectEqualStrings("none", old.get("off").?.string);
}

test "omp models.yml: off rides enable_thinking, every other level an accepted reasoning_effort" {
    // The shape verified against omp 18.3.0: the qwen dialect sends off as enable_thinking false,
    // `whenThinking` switches a thinking request to reasoning_effort, remapped per model.
    const entries = [_]Entry{
        .{ .id = "qwen", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true, .efforts = &qwen4_efforts },
        .{ .id = "mimo", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true, .efforts = &mimo_efforts },
        .{ .id = "old", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
    };
    const yml = try ompModelsYml(t.allocator, "http://127.0.0.1:12345", &entries);
    defer t.allocator.free(yml);
    try t.expect(std.mem.indexOf(u8, yml, "      qwenTemplateReasoningEffort: false\n      whenThinking:\n        thinkingFormat: openai\n") != null);
    const qwen_block = "      - id: \"qwen\"\n        name: \"qwen (sushi)\"\n        reasoning: true\n        thinking:\n" ++
        "          mode: effort\n          requiresEffort: false\n          efforts: [minimal, low, medium, high, xhigh, max]\n" ++
        "          effortMap: {minimal: low, high: xhigh, max: xhigh}\n        input: [text]\n";
    try t.expect(std.mem.indexOf(u8, yml, qwen_block) != null);
    const mimo_block = "      - id: \"mimo\"\n        name: \"mimo (sushi)\"\n        reasoning: true\n        thinking:\n" ++
        "          mode: effort\n          requiresEffort: false\n          efforts: [minimal, low, medium, high, xhigh, max]\n" ++
        "          effortMap: {minimal: on, low: on, medium: on, high: on, xhigh: on, max: on}\n        input: [text]\n";
    try t.expect(std.mem.indexOf(u8, yml, mimo_block) != null);
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, yml, "thinking:\n"));
    try t.expect(std.mem.indexOf(u8, yml, "      - id: \"old\"\n        name: \"old (sushi)\"\n        reasoning: true\n        input: [text]\n") != null);
}

test "parseChatEntries reads each row's reasoning_efforts" {
    const body =
        \\{"data":[{"id":"q","capabilities":["chat"],"context_length":8192,"reasoning_efforts":["off","low","medium","xhigh"]},
        \\ {"id":"old","context_length":8192}]}
    ;
    var models = try parseChatEntries(t.allocator, body);
    defer models.deinit();
    try t.expectEqual(@as(usize, 2), models.entries.len);
    try t.expectEqual(@as(usize, 4), models.entries[0].efforts.?.len);
    try t.expectEqualStrings("xhigh", models.entries[0].efforts.?[3]);
    try t.expect(models.entries[1].efforts == null);
}

test "compactionReserve: a quarter of the window, capped where the agents' own defaults take over" {
    // pi keeps 20000 recent tokens by default, sized for a 200k window: a 24k
    // window would never compact.
    try t.expectEqual(@as(u64, 6144), compactionReserve(24576));
    try t.expectEqual(@as(u64, 2048), compactionReserve(8192));
    try t.expectEqual(@as(u64, 1024), compactionReserve(2048));
    try t.expectEqual(@as(u64, 20000), compactionReserve(262144));
}

test "pi settings.json merge scales compaction to the window and keeps the rest" {
    const existing =
        \\{"theme":"dark","defaultProvider":"sushi","compaction":{"enabled":false,"reserveTokens":1}}
    ;
    const json = try mergePiSettingsJson(t.allocator, existing, 24576);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try t.expectEqualStrings("dark", obj.get("theme").?.string);
    const c = obj.get("compaction").?.object;
    // The user's own enabled flag survives; the numbers are ours.
    try t.expectEqual(false, c.get("enabled").?.bool);
    try t.expectEqual(@as(i64, 10240), c.get("reserveTokens").?.integer);
    try t.expectEqual(@as(i64, 6144), c.get("keepRecentTokens").?.integer);

    // A big window keeps pi's own defaults (16384 / 20000); an empty file is fine.
    const big = try mergePiSettingsJson(t.allocator, "", 262144);
    defer t.allocator.free(big);
    const bp = try std.json.parseFromSlice(std.json.Value, t.allocator, big, .{});
    defer bp.deinit();
    const bc = bp.value.object.get("compaction").?.object;
    try t.expectEqual(@as(i64, 16384), bc.get("reserveTokens").?.integer);
    try t.expectEqual(@as(i64, 20000), bc.get("keepRecentTokens").?.integer);
}

test "opencode config: limit.output is the compaction reserve" {
    // opencode never sends max_tokens; `limit.output` is only the room it
    // keeps free before compacting.
    const entries = [_]Entry{
        .{ .id = "m1", .budget = budgetForContext(24576), .vision = false, .loaded = true },
    };
    const v1 = try opencodeJson(t.allocator, "http://127.0.0.1:12345", "m1", &entries);
    defer t.allocator.free(v1);
    const p1 = try std.json.parseFromSlice(std.json.Value, t.allocator, v1, .{});
    defer p1.deinit();
    const limit = p1.value.object.get("provider").?.object.get("sushi").?.object.get("models").?.object.get("m1").?.object.get("limit").?.object;
    try t.expectEqual(@as(i64, 6144), limit.get("output").?.integer);
    try t.expect(p1.value.object.get("compaction") == null);
    try t.expectEqualStrings("sushi/m1", p1.value.object.get("model").?.string);
}

test "aider metadata: litellm keys per openai/<id> entry" {
    const entries = [_]Entry{
        .{ .id = "m1", .budget = .{ .context = 4096, .output = 1024 }, .vision = false, .loaded = true },
    };
    const json = try aiderMetadataJson(t.allocator, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const row = parsed.value.object.get("openai/m1").?.object;
    try t.expectEqual(@as(i64, 4096), row.get("max_input_tokens").?.integer);
    try t.expectEqual(@as(i64, 1024), row.get("max_output_tokens").?.integer);
}

test "launch args: passthrough after --, unknown agent named, url trailing slash trimmed" {
    const parsed = try parseLaunchArgs(&.{ "codex", "--url", "http://x:1/", "--print", "--", "resume", "-a" });
    try t.expectEqual(AgentKind.codex, parsed.kind);
    try t.expectEqualStrings("http://x:1", parsed.url.?);
    try t.expect(parsed.print_only);
    try t.expectEqual(@as(usize, 2), parsed.extras.len);
    try t.expectEqualStrings("resume", parsed.extras[0]);
    try t.expectError(error.UnknownAgent, parseLaunchArgs(&.{"cursor"}));
    // The rebrand alias from issue #188's own wording.
    try t.expectEqual(AgentKind.codex, (try parseLaunchArgs(&.{"chatgpt"})).kind);
}

test "launch args: the default port is sushi's own, clear of mlx-serve's 11234" {
    try t.expectEqual(@as(u16, 12345), (try parseLaunchArgs(&.{"opencode"})).port);
}

test "script assembly: extras are shell-quoted onto the invocation line" {
    const script = try scriptFor(t.allocator, .codex, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, null, &.{ "resume", "it's" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "\"$CODEX_BIN\" 'resume' 'it'\\''s'") != null);
    try t.expect(std.mem.indexOf(u8, script, "export CODEX_HOME=\"$HOME/.sushi/codex\"") != null);
}

test "codex script falls back to the desktop app's bundled CLI (ChatGPT.app rebrand)" {
    const script = try scriptFor(t.allocator, .codex, "http://x:1", "m1", .{ .context = 4096, .output = 1024 }, null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "/Applications/ChatGPT.app") != null);
    try t.expect(std.mem.indexOf(u8, script, "/Applications/Codex.app") != null);
    try t.expect(std.mem.indexOf(u8, script, "$HOME/Applications") != null);
    try t.expect(std.mem.indexOf(u8, script, "Contents/Resources/codex") != null);
    // Never exec an empty resolution — refuse with the install hint.
    try t.expect(std.mem.indexOf(u8, script, "exit 127") != null);
    try t.expect(std.mem.indexOf(u8, script, "\n\"$CODEX_BIN\"") != null);
}

test "claude script keeps a slow local turn on one streamed request" {
    const script = try scriptFor(t.allocator, .claude, "http://x:1", "m1", budgetForContext(786432), null, &.{});
    defer t.allocator.free(script);
    for ([_][]const u8{
        "export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1\n",
        "export API_TIMEOUT_MS=3600000\n",
        "export CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS=1800000\n",
        "export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000\n",
        "export CLAUDE_BYTE_STREAM_IDLE_TIMEOUT_MS=1800000\n",
    }) |line| try t.expect(std.mem.indexOf(u8, script, line) != null);
}

test "claude script declares the advertised context window (CLAUDE_CODE_MAX_CONTEXT_TOKENS)" {
    // Claude Code assumes 200k outside its catalog; CLAUDE_CODE_MAX_CONTEXT_TOKENS is the override.
    const script = try scriptFor(t.allocator, .claude, "http://x:1", "m1", budgetForContext(786432), null, &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_CONTEXT_TOKENS=786432") != null);
    try t.expect(std.mem.indexOf(u8, script, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=65536") != null);
    try t.expect(std.mem.indexOf(u8, script, "\nclaude --model 'm1'") != null);

    // An unknown context is not a claim: omit the export rather than pin a
    // number the server never advertised.
    const unknown = try scriptFor(t.allocator, .claude, "http://x:1", "m1", .{ .context = 0, .output = 8192 }, null, &.{});
    defer t.allocator.free(unknown);
    try t.expect(std.mem.indexOf(u8, unknown, "CLAUDE_CODE_MAX_CONTEXT_TOKENS") == null);
    try t.expect(std.mem.indexOf(u8, unknown, "export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192") != null);
}

test "zcode config escapes arbitrary model ids and declares server budgets and wire maps" {
    const entries = [_]Entry{
        .{ .id = "org/model\"quoted", .budget = .{ .context = 98304, .output = 49152 }, .vision = true, .loaded = false },
        .{ .id = "glm-native", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true },
    };
    const json = try zcodeConfigJson(t.allocator, "http://127.0.0.1:12345", entries[0].id, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try t.expectEqualStrings(entries[0].id, config.get("defaultModelSelection").?.object.get("modelId").?.string);
    const rules = config.get("modelConfigRules").?.object.get("providerModelRules").?.array.items;
    try t.expectEqual(@as(usize, 2), rules.len);
    const m = rules[0].object.get("config").?.object;
    try t.expectEqual(@as(i64, 98304), m.get("properties").?.object.get("contextWindow").?.integer);
    try t.expectEqualStrings("{\"max_tokens\": maxOutputTokens}", m.get("optionSpecs").?.object.get("maxOutputTokens").?.object.get("map").?.string);
    const script = try scriptFor(t.allocator, .zcode, "http://127.0.0.1:12345", entries[0].id, entries[0].budget, null, &.{ "--prompt", "it's a prompt" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "ZCODE_PERSONAL_PROVIDER_CONFIG_FILE") != null);
    try t.expect(std.mem.indexOf(u8, script, "zcode '--prompt' 'it'\\''s a prompt'") != null);
}

test "zcode starts MiMo thinking, its own default" {
    const entries = [_]Entry{.{ .id = "mimo", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true, .efforts = &mimo_efforts }};
    const json = try zcodeConfigJson(t.allocator, "http://localhost:12345", entries[0].id, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    try t.expectEqualStrings("on", parsed.value.object.get("config").?.object.get("defaultModelSelection").?.object.get("options").?.object.get("reasoningLevel").?.string);
}

test "zcode exposes only a model's advertised reasoning efforts" {
    const entries = [_]Entry{.{ .id = "arbitrary-native", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true, .efforts = &.{ "off", "max" } }};
    const json = try zcodeConfigJson(t.allocator, "http://localhost:12345", entries[0].id, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try t.expectEqualStrings("max", config.get("defaultModelSelection").?.object.get("options").?.object.get("reasoningLevel").?.string);
    const levels = config.get("modelConfigRules").?.object.get("providerModelRules").?.array.items[0].object.get("config").?.object.get("optionSpecs").?.object.get("reasoningLevel").?.object.get("values").?.array.items;
    try t.expectEqual(@as(usize, 2), levels.len);
    try t.expectEqualStrings("off", levels[0].string);
    try t.expectEqualStrings("max", levels[1].string);
}

fn glmEntry() Entry {
    return .{ .id = "GLM-5.3-Flash-Sushi-2.5bpw", .budget = budgetForContext(1048576), .vision = true, .loaded = true, .efforts = &glm5_efforts, .default_effort = "high" };
}

test "opencode config for GLM: tool calls on, reasoning_content round-trips, variants are only the words GLM accepts" {
    const entries = [_]Entry{glmEntry()};
    const json = try opencodeJson(t.allocator, "http://127.0.0.1:12345", entries[0].id, &entries);
    defer t.allocator.free(json);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try t.expectEqualStrings("sushi/GLM-5.3-Flash-Sushi-2.5bpw", root.get("model").?.string);
    const sushi = root.get("provider").?.object.get("sushi").?.object;
    try t.expectEqual(@as(i64, 3600000), sushi.get("options").?.object.get("chunkTimeout").?.integer);
    const m = sushi.get("models").?.object.get(entries[0].id).?.object;
    try t.expectEqual(true, m.get("tool_call").?.bool);
    try t.expectEqual(true, m.get("reasoning").?.bool);
    try t.expectEqualStrings("reasoning_content", m.get("interleaved").?.object.get("field").?.string);
    try t.expectEqual(@as(i64, 1048576), m.get("limit").?.object.get("context").?.integer);
    const variants = m.get("variants").?.object;
    try t.expectEqual(@as(usize, 3), variants.count());
    for (glm5_efforts) |w| try t.expectEqualStrings(w, variants.get(w).?.object.get("reasoningEffort").?.string);
    // GLM refuses `medium`: a default effort option would fail every request.
    try t.expect(m.get("options") == null);
}

test "opencode config: a row without efforts is not declared a reasoning model, on/off make no variants" {
    const plain = [_]Entry{.{ .id = "old", .budget = budgetForContext(65536), .vision = false, .loaded = true }};
    const a = try opencodeJson(t.allocator, "http://x:1", "old", &plain);
    defer t.allocator.free(a);
    try t.expect(std.mem.indexOf(u8, a, "\"reasoning\"") == null);
    const mimo = [_]Entry{.{ .id = "mimo", .budget = budgetForContext(65536), .vision = false, .loaded = true, .efforts = &mimo_efforts }};
    const b = try opencodeJson(t.allocator, "http://x:1", "mimo", &mimo);
    defer t.allocator.free(b);
    try t.expect(std.mem.indexOf(u8, b, "\"reasoning\": true") != null);
    try t.expect(std.mem.indexOf(u8, b, "variants") == null);
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, b, .{});
    parsed.deinit();
}

test "opencode script: optional --standalone follows the subcommand, the model rides the config" {
    // opencode 2.x connects to a background service that never sees this env, refuses --model on its
    // default command, and refuses a flag placed before a subcommand.
    const b = budgetForContext(1048576);
    const tui = try scriptFor(t.allocator, .opencode, "http://x:1", "m1", b, "{}", &.{});
    defer t.allocator.free(tui);
    try t.expect(std.mem.indexOf(u8, tui, "\nopencode \"${sushi_opencode_flags[@]}\"\n") != null);
    try t.expect(std.mem.indexOf(u8, tui, "--model") == null);
    const run = try scriptFor(t.allocator, .opencode, "http://x:1", "m1", b, "{}", &.{ "run", "read it's" });
    defer t.allocator.free(run);
    try t.expect(std.mem.indexOf(u8, run, "\nopencode 'run' \"${sushi_opencode_flags[@]}\" 'read it'\\''s'\n") != null);
    const flag = try scriptFor(t.allocator, .opencode, "http://x:1", "m1", b, "{}", &.{"--continue"});
    defer t.allocator.free(flag);
    try t.expect(std.mem.indexOf(u8, flag, "\nopencode \"${sushi_opencode_flags[@]}\" '--continue'\n") != null);
}

test "grok names" {
    try t.expectEqual(AgentKind.grok, AgentKind.fromName("grok").?);
    try t.expect(std.mem.indexOf(u8, AgentKind.names, "grok") != null);
}

test "grok compaction threshold leaves the compaction reserve free" {
    try t.expectEqual(@as(u64, 98), grokCompactPercent(1048576));
    try t.expectEqual(@as(u64, 75), grokCompactPercent(24576));
    try t.expectEqual(@as(u64, 50), grokCompactPercent(2048));
}

test "grok config.toml for GLM: chat completions at the server, advertised budgets, only accepted efforts" {
    const entries = [_]Entry{glmEntry()};
    const toml = try grokConfigToml(t.allocator, "http://127.0.0.1:12345", entries[0].id, entries[0].budget, &entries);
    defer t.allocator.free(toml);
    for ([_][]const u8{
        "[models]\ndefault = \"GLM-5.3-Flash-Sushi-2.5bpw\"\n",
        "[session]\nauto_compact_threshold_percent = 98\n",
        "[model.\"GLM-5.3-Flash-Sushi-2.5bpw\"]\n",
        "model = \"GLM-5.3-Flash-Sushi-2.5bpw\"\n",
        "base_url = \"http://127.0.0.1:12345/v1\"\n",
        "api_backend = \"chat_completions\"\n",
        "api_key = \"sushi\"\n",
        "context_window = 1048576\n",
        "max_completion_tokens = 65536\n",
        "supports_reasoning_effort = true\n",
        "reasoning_effort = \"high\"\n",
        "[[model.\"GLM-5.3-Flash-Sushi-2.5bpw\".reasoning_efforts]]\nid = \"max\"\nvalue = \"max\"\n",
    }) |needle| try t.expect(std.mem.indexOf(u8, toml, needle) != null);
    try t.expect(std.mem.indexOf(u8, toml, "id = \"medium\"") == null);
    try t.expect(std.mem.indexOf(u8, toml, "id = \"off\"") == null);
}

test "grok config.toml escapes ids, lists every row, and leaves effort off for a row without any" {
    const entries = [_]Entry{
        .{ .id = "org/m\"q", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true },
        .{ .id = "second", .budget = budgetForContext(65536), .vision = false, .loaded = false, .efforts = &.{ "low", "xhigh" } },
    };
    const toml = try grokConfigToml(t.allocator, "http://h:1", entries[0].id, entries[0].budget, &entries);
    defer t.allocator.free(toml);
    try t.expect(std.mem.indexOf(u8, toml, "[model.\"org/m\\\"q\"]\n") != null);
    const second = std.mem.indexOf(u8, toml, "[model.\"second\"]").?;
    try t.expect(std.mem.indexOf(u8, toml[0..second], "supports_reasoning_effort") == null);
    // No advertised default: the first listed word.
    try t.expect(std.mem.indexOf(u8, toml[second..], "reasoning_effort = \"low\"\n") != null);
}

test "grok script: its own GROK_HOME, the chosen model, extras quoted" {
    const e = glmEntry();
    const script = try scriptFor(t.allocator, .grok, "http://x:1", e.id, e.budget, null, &.{ "-p", "read README.md" });
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export GROK_HOME=\"$HOME/.sushi/grok\"\n") != null);
    try t.expect(std.mem.indexOf(u8, script, "\ngrok --model 'GLM-5.3-Flash-Sushi-2.5bpw' '-p' 'read README.md'\n") != null);
}

test "parseChatEntries reads default_reasoning_effort" {
    var models = try parseChatEntries(t.allocator,
        \\{"data":[{"id":"glm","reasoning_efforts":["low","high","max"],"default_reasoning_effort":"high"},{"id":"old"}]}
    );
    defer models.deinit();
    try t.expectEqualStrings("high", models.entries[0].default_effort.?);
    try t.expect(models.entries[1].default_effort == null);
}

const hostile_id = "x'; touch PWNED #$(id)\"\\";
const hostile_quoted = "'x'\\''; touch PWNED #$(id)\"\\'";

test "script: a model id with spaces, quotes, ; and $() stays one shell word in every agent" {
    const b = budgetForContext(786432);
    const spaced = try scriptFor(t.allocator, .claude, "http://x:1", "My Model (copy)", b, null, &.{});
    defer t.allocator.free(spaced);
    try t.expect(std.mem.indexOf(u8, spaced, "\nclaude --model 'My Model (copy)'\n") != null);
    try t.expect(std.mem.indexOf(u8, spaced, "export ANTHROPIC_DEFAULT_OPUS_MODEL='My Model (copy)'\n") != null);

    const claude = try scriptFor(t.allocator, .claude, "http://x:1", hostile_id, b, null, &.{});
    defer t.allocator.free(claude);
    try t.expect(std.mem.indexOf(u8, claude, "export CLAUDE_CODE_SUBAGENT_MODEL=" ++ hostile_quoted ++ "\n") != null);
    try t.expect(std.mem.indexOf(u8, claude, "\nclaude --model " ++ hostile_quoted ++ "\n") != null);

    const pi = try scriptFor(t.allocator, .pi, "http://x:1", hostile_id, b, null, &.{});
    defer t.allocator.free(pi);
    try t.expect(std.mem.indexOf(u8, pi, "\npi --provider sushi --model " ++ hostile_quoted ++ "\n") != null);

    const omp = try scriptFor(t.allocator, .omp, "http://x:1", hostile_id, b, null, &.{});
    defer t.allocator.free(omp);
    try t.expect(std.mem.indexOf(u8, omp, "\nomp --model 'sushi/x'\\''; touch PWNED #$(id)\"\\'\n") != null);

    const aider = try scriptFor(t.allocator, .aider, "http://x:1", hostile_id, b, null, &.{});
    defer t.allocator.free(aider);
    const oai = "'openai/x'\\''; touch PWNED #$(id)\"\\'";
    try t.expect(std.mem.indexOf(u8, aider, "\naider --model " ++ oai ++ " --weak-model " ++ oai ++ " --model-metadata-file") != null);

    const hostile_url = try scriptFor(t.allocator, .aider, "http://h'; touch PWNED #", "m1", b, null, &.{});
    defer t.allocator.free(hostile_url);
    try t.expect(std.mem.indexOf(u8, hostile_url, "export OPENAI_API_BASE='http://h'\\''; touch PWNED #/v1'\n") != null);
    const claude_url = try scriptFor(t.allocator, .claude, "http://h'; touch PWNED #", "m1", b, null, &.{});
    defer t.allocator.free(claude_url);
    try t.expect(std.mem.indexOf(u8, claude_url, "export ANTHROPIC_BASE_URL='http://h'\\''; touch PWNED #'\n") != null);
}

test "script: the opencode config is single-quote escaped as one env value" {
    const script = try scriptFor(t.allocator, .opencode, "http://x:1", "m1", FALLBACK_BUDGET, "{\"model\": \"it's\"}", &.{});
    defer t.allocator.free(script);
    try t.expect(std.mem.indexOf(u8, script, "export OPENCODE_CONFIG_CONTENT='{\"model\": \"it'\\''s\"}'\n") != null);
}

test "config writers escape ids and urls so hostile ones stay data" {
    const id = "a\"b\\c'd";
    const entries = [_]Entry{.{ .id = id, .budget = FALLBACK_BUDGET, .vision = false, .loaded = true }};
    const oc = try opencodeJson(t.allocator, "http://h\"x:1", id, &entries);
    defer t.allocator.free(oc);
    const oc_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, oc, .{});
    defer oc_parsed.deinit();
    try t.expectEqualStrings("sushi/" ++ id, oc_parsed.value.object.get("model").?.string);
    try t.expect(oc_parsed.value.object.get("provider").?.object.get("sushi").?.object.get("models").?.object.get(id) != null);

    const pi = try piModelsJson(t.allocator, "http://h\"x:1", &entries);
    defer t.allocator.free(pi);
    const pi_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, pi, .{});
    defer pi_parsed.deinit();
    const pi_models = pi_parsed.value.object.get("providers").?.object.get("sushi").?.object.get("models").?.array.items;
    try t.expectEqualStrings(id, pi_models[0].object.get("id").?.string);

    const aider = try aiderMetadataJson(t.allocator, &entries);
    defer t.allocator.free(aider);
    const aider_parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, aider, .{});
    defer aider_parsed.deinit();
    try t.expect(aider_parsed.value.object.get("openai/" ++ id) != null);

    const codex = try codexConfigToml(t.allocator, "http://h:1", id, FALLBACK_BUDGET);
    defer t.allocator.free(codex);
    try t.expect(std.mem.indexOf(u8, codex, "\nmodel = \"a\\\"b\\\\c'd\"\n") != null);

    const omp = try ompModelsYml(t.allocator, "http://h:1", &entries);
    defer t.allocator.free(omp);
    try t.expect(std.mem.indexOf(u8, omp, "      - id: \"a\\\"b\\\\c'd\"\n") != null);

    const hermes = try hermesConfigYaml(t.allocator, "http://h:1", id, &entries);
    defer t.allocator.free(hermes);
    try t.expect(std.mem.indexOf(u8, hermes, "  default: \"a\\\"b\\\\c'd\"\n") != null);
    try t.expect(std.mem.indexOf(u8, hermes, "      \"a\\\"b\\\\c'd\":\n") != null);
}

test "opencode --think applies only to selected model and uses SDK spelling" {
    const entries = [_]Entry{
        .{ .id = "qwen", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true, .efforts = &qwen4_efforts },
        .{ .id = "glm", .budget = FALLBACK_BUDGET, .vision = false, .loaded = false, .efforts = &glm5_efforts },
    };
    for ([_][]const u8{ "off", "low", "medium", "xhigh" }) |effort| {
        const text = try opencodeJsonWithEffort(t.allocator, "http://localhost:12345", "qwen", &entries, effort);
        defer t.allocator.free(text);
        const json = try std.json.parseFromSlice(std.json.Value, t.allocator, text, .{});
        defer json.deinit();
        const models = json.value.object.get("provider").?.object.get("sushi").?.object.get("models").?.object;
        const options = models.get("qwen").?.object.get("options").?.object;
        try t.expectEqualStrings(effort, options.get("reasoningEffort").?.string);
        try t.expect(options.get("reasoning_effort") == null);
        try t.expect(models.get("glm").?.object.get("options") == null);
    }
    try t.expectError(error.UnsupportedReasoningEffort, opencodeJsonWithEffort(t.allocator, "http://localhost:12345", "qwen", &entries, "high"));
}

test "opencode --think rejects unadvertised or absent reasoning support" {
    const plain = [_]Entry{.{ .id = "plain", .budget = FALLBACK_BUDGET, .vision = false, .loaded = true }};
    try t.expectError(error.UnsupportedReasoningEffort, opencodeJsonWithEffort(t.allocator, "http://localhost:12345", "plain", &plain, "xhigh"));
    const parsed = try parseLaunchArgs(&.{ "opencode", "--think", "xhigh", "--", "run", "hello" });
    try t.expectEqualStrings("xhigh", parsed.think.?);
    try t.expectEqualStrings("run", parsed.extras[0]);
    try t.expectError(error.Usage, parseLaunchArgs(&.{ "opencode", "--think" }));
    try t.expectError(error.Usage, parseLaunchArgs(&.{ "pi", "--think", "xhigh" }));
}

test "opencode persist flag is explicit and agent scoped" {
    try t.expect((try parseLaunchArgs(&.{ "opencode", "--persist", "--think", "xhigh" })).persist);
    try t.expect(!(try parseLaunchArgs(&.{"opencode"})).persist);
    try t.expectError(error.Usage, parseLaunchArgs(&.{ "pi", "--persist" }));
}
