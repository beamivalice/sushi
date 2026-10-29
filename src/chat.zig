const std = @import("std");
const jinja_c = @import("jinja_c");
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;
const tool_calls_mod = @import("tool_calls.zig");
pub const parseToolCalls = tool_calls_mod.parseToolCalls;
const balancedJsonObject = tool_calls_mod.balancedJsonObject;
pub const BARE_INVOKE = tool_calls_mod.BARE_INVOKE;
pub const FUNCTION_CALLS_TAG = tool_calls_mod.FUNCTION_CALLS_TAG;
const looseRepairContainer = tool_calls_mod.looseRepairContainer;
pub const miniCpm5AttrValue = tool_calls_mod.miniCpm5AttrValue;
pub const ToolChoice = tool_calls_mod.ToolChoice;
pub const parseToolChoice = tool_calls_mod.parseToolChoice;
pub const declaredToolNames = tool_calls_mod.declaredToolNames;
pub const toolChoiceNamesUndeclared = tool_calls_mod.toolChoiceNamesUndeclared;
const appendToolSystemPrompt = tool_calls_mod.appendToolSystemPrompt;
const think_tags = @import("think_tags.zig");
const tagLenAt = think_tags.tagLenAt;
pub const thinkOpenTagLenAt = think_tags.thinkOpenTagLenAt;
pub const indexOfThinkCloseTag = think_tags.indexOfThinkCloseTag;
pub const tool_markup_openers = think_tags.tool_markup_openers;
pub const trimLeakedToolMarkup = think_tags.trimLeakedToolMarkup;
pub const ThinkSplit = think_tags.ThinkSplit;
pub const modelThinkOpener = think_tags.modelThinkOpener;
pub const isChannelMarkerToken = think_tags.isChannelMarkerToken;
pub const BARE_THINK_OPENER = think_tags.BARE_THINK_OPENER;
pub const BARE_THINK_CLOSER = think_tags.BARE_THINK_CLOSER;
pub const promptTailOpensThink = think_tags.promptTailOpensThink;
pub const promptThinkTailClass = think_tags.promptThinkTailClass;
pub const noThinkTailSuffix = think_tags.noThinkTailSuffix;
const contentChannelTail = think_tags.contentChannelTail;
pub const continuationRequested = think_tags.continuationRequested;
const continuationPrefill = think_tags.continuationPrefill;
pub const splitThinkBlock = think_tags.splitThinkBlock;
pub const splitThinkBlockKeepingMarkup = think_tags.splitThinkBlockKeepingMarkup;
pub const normalizeEmbeddedThinkBlocks = think_tags.normalizeEmbeddedThinkBlocks;
pub const thinkOpenerPossible = think_tags.thinkOpenerPossible;
pub const endsWithPartialThinkOpen = think_tags.endsWithPartialThinkOpen;
pub const streamShouldBufferForTools = think_tags.streamShouldBufferForTools;
pub const StreamThinkGate = think_tags.StreamThinkGate;
pub const streamThinkGate = think_tags.streamThinkGate;
pub const ThinkScan = think_tags.ThinkScan;
pub const streamThinkGateScan = think_tags.streamThinkGateScan;
pub const streamTailIsReasoning = think_tags.streamTailIsReasoning;
pub const streamContentLead = think_tags.streamContentLead;
pub const settledReasoning = think_tags.settledReasoning;
pub const openThoughtFlush = think_tags.openThoughtFlush;
pub const closedThoughtDelta = think_tags.closedThoughtDelta;
pub const cutThoughtDelta = think_tags.cutThoughtDelta;
pub const unstreamedReasoning = think_tags.unstreamedReasoning;

pub const ToolCall = struct {
    id: []const u8,
    name: []const u8,
    arguments: []const u8, // JSON string
};

/// Raw image pixel data for vision encoder (float32, CHW format).
pub const ImageData = struct {
    pixels: []const u8, // Gemma: CHW float32 [3*H*W*4]. Qwen: merge-order patches [N*1536*4].
    width: u32,
    height: u32,
    // Qwen3-VL only: full patch grid (h=resized_H/patch, w=resized_W/patch). 0 ⇒
    // Gemma CHW layout. When >0, `pixels` holds the processor's merge-order
    // pixel_values and the encoder is QwenVision (see src/qwen_vision.zig).
    grid_h: u32 = 0,
    grid_w: u32 = 0,
    /// `pixels` is still the encoded file (JPEG/PNG/WebP) and `width`/`height`/`grid_*` are the plan the
    /// processor will reach; the tower's turn decodes it, so a request holds the compressed bytes only.
    undecoded: bool = false,
};

/// Per-request image preprocessing selector, derived from the loaded model's
/// config. Threaded into `parseImageUrlContent`/`decodeImageToPixels` so decode
/// stays race-safe (no global state) under `--max-concurrent ≥ 2`.
pub const VisionPreproc = struct {
    /// Which processor produced `ImageData.pixels`: Gemma's fixed CHW square,
    /// or one of the patch-grid towers (each with its own resize + patch order).
    mode: enum { gemma, qwen, mimo, glm5, dsv41 } = .gemma,
    patch: u32 = 16,
    tps: u32 = 2,
    merge: u32 = 2,
    min_pixels: u32 = 0,
    max_pixels: u32 = 0,
    /// glm5: the resize cap and floor are on MERGED tokens, not pixels.
    max_tokens: u32 = 0,
    max_video_tokens: u32 = 0,
    min_tokens: u32 = 0,
    /// dsv41: an image wider than this many heights is stretched into the capped canvas (0 = no cap).
    max_wh_ratio: u32 = 0,
};

/// Raw mono 16 kHz audio samples for the Gemma 4 12B unified audio embedder.
/// Bytes are little-endian float32 PCM; the encoder frames them into 640-sample
/// tokens (40 ms @ 16 kHz) and projects each straight into language-model space.
pub const AudioData = struct {
    samples: []const u8, // Raw float32-LE bytes (n_samples * 4)
};

/// Qwen3-VL video: pre-patchified pixel_values for ALL `grid_t` temporal-patch
/// groups, concatenated (see `vision_common.buildPixelValuesVideo` /
/// `QwenVision.forwardVideo`) — the video-equivalent of `ImageData.pixels`'s
/// Qwen merge-order layout. `grid_t` is a TEMPORAL PATCH count (raw sampled
/// frames grouped `tps`-at-a-time), not a raw frame count; `grid_h`/`grid_w`
/// are the shared per-frame patch grid every frame in the video was resized to.
pub const VideoData = struct {
    pixels: []const u8, // merge-order patches [grid_t*(grid_h*grid_w)*feat*4]
    grid_t: u32,
    grid_h: u32,
    grid_w: u32,
    /// Rate of the pre-sampled input frames. GLM uses it for frame timestamps.
    fps: f64 = 24,
};

/// OpenAI spells the system turn `developer` for reasoning models (pi does
/// when `supportsDeveloperRole` is unset). No template we serve tells the two
/// apart and Qwen's raises on the unknown role, which is a SILENT fallback.
pub fn canonicalRole(role: []const u8) []const u8 {
    return if (std.mem.eql(u8, role, "developer")) "system" else role;
}

/// Fold every `system` message past index 0 into the leading one (created
/// when absent). Qwen3.8's template raises on a system turn that is not first
/// and the raise is a silent generic fallback; MiMo's renders it in place.
/// Returns the joined buffer the caller owns, null when nothing moved.
pub fn foldSystemMessages(allocator: std.mem.Allocator, messages: *std.ArrayList(Message)) !?[]const u8 {
    var extra: usize = 0;
    for (messages.items[@min(messages.items.len, 1)..]) |m| {
        if (std.mem.eql(u8, m.role, "system")) extra += 1;
    }
    if (extra == 0) return null;
    const lead_is_system = messages.items.len > 0 and std.mem.eql(u8, messages.items[0].role, "system");
    var joined = std.ArrayList(u8).empty;
    errdefer joined.deinit(allocator);
    if (lead_is_system) try joined.appendSlice(allocator, messages.items[0].content);
    var i: usize = if (lead_is_system) 1 else 0;
    while (i < messages.items.len) {
        if (!std.mem.eql(u8, messages.items[i].role, "system")) {
            i += 1;
            continue;
        }
        if (joined.items.len > 0) try joined.appendSlice(allocator, "\n\n");
        try joined.appendSlice(allocator, messages.items[i].content);
        _ = messages.orderedRemove(i);
    }
    const text = try joined.toOwnedSlice(allocator);
    if (lead_is_system) {
        messages.items[0].content = text;
    } else {
        try messages.insert(allocator, 0, .{ .role = "system", .content = text });
    }
    return text;
}

/// The most images one request may carry, counted across every message.
pub const MAX_REQUEST_IMAGES: usize = 1024;

/// Where one image or video sits in `Message.content`: the template renders
/// its placeholder at byte offset `at`. Parts of a kind consume the message's
/// `images` / `videos` in order.
pub const MediaPart = struct {
    at: usize,
    kind: Kind,

    pub const Kind = enum { image, video };
};

pub const Message = struct {
    role: []const u8,
    content: []const u8,
    tool_calls: ?[]const ToolCall = null,
    tool_call_id: ?[]const u8 = null,
    images: ?[]const ImageData = null, // Preprocessed image data for vision
    videos: ?[]const VideoData = null, // Preprocessed video data for vision
    audio: ?[]const AudioData = null, // Raw PCM for the unified audio embedder
    media_parts: ?[]const MediaPart = null,
    /// Reasoning the client round-trips on assistant HISTORY messages
    /// (`reasoning_content`/`reasoning` on chat completions, `thinking`
    /// blocks on /v1/messages). Templates that persist reasoning across
    /// turns read it; templates that strip history reasoning
    /// (Qwen, Gemma) never reference the field and render unchanged.
    reasoning_content: ?[]const u8 = null,
};

/// Chat template configuration loaded from tokenizer_config.json.
pub const ChatConfig = struct {
    chat_template: []const u8,
    bos_token: ?[]const u8,
    eos_token: ?[]const u8,
    add_bos_token: bool,
    allocator: std.mem.Allocator,
    /// The template's `preserve_thinking`, resolved per request; null leaves it
    /// undefined, which Qwen3.8 reads as keeping every turn's thinking.
    preserve_thinking: ?bool = null,
    /// A template that reads `content` as a STRING (DeepSeek-V4.1): each image becomes this atom where it sits,
    /// joined to the text around it by blank lines, as the release's encoder renders image blocks.
    image_placeholder: ?[]const u8 = null,

    pub fn deinit(self: *ChatConfig) void {
        self.allocator.free(self.chat_template);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
    }
};

/// True when `<model_dir>/config.json` declares the given model_type. Any
/// read/parse failure is false — this only ARMS a family fallback, never
/// blocks a load.
fn dirModelTypeIs(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, want: []const u8) bool {
    const cfg_path = std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir}) catch return false;
    defer allocator.free(cfg_path);
    const file = std.Io.Dir.openFileAbsolute(io, cfg_path, .{}) catch return false;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var rs = file.reader(io, &buf);
    const content = rs.interface.allocRemaining(allocator, .limited(1 << 22)) catch return false;
    defer allocator.free(content);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    const mt = parsed.value.object.get("model_type") orelse return false;
    return mt == .string and std.mem.eql(u8, mt.string, want);
}

/// Is this `chat_template` entry just an `{% include %}` POINTER at the
/// sidecar file? transformers >= 5 saves the real template to
/// `chat_template.jinja` and writes that one-liner into tokenizer_config.json
/// (mlx-community oQ4e / oQ5e builds, issue #169). jinja.cpp has
/// no `include` statement, so rendering the pointer fails and drops SILENTLY to
/// the generic format — the model then answers in another family's markers.
/// The file it names is always the sidecar the caller already falls back to.
fn isIncludeStub(template: []const u8) bool {
    const t = std.mem.trim(u8, template, " \t\r\n");
    if (!std.mem.startsWith(u8, t, "{%") or !std.mem.endsWith(u8, t, "%}")) return false;
    const body = std.mem.trim(u8, t[2 .. t.len - 2], " \t\r\n-");
    if (!std.mem.startsWith(u8, body, "include")) return false;
    // Exactly one statement: a second `{%` would make it a real template.
    return std.mem.indexOf(u8, t[2..], "{%") == null;
}

/// The `chat_template` entry of tokenizer_config.json, as a string.
///
/// HF allows TWO shapes and we only handled one: a bare string, or a LIST of
/// `{name, template}` objects (Mistral-7B-Instruct-v0.3 ships `default` +
/// `tool_use`). Reading the list as `v.string` is not a graceful failure — it
/// is a Zig union panic that kills the process at LOAD, so the checkpoint
/// could not be served at all (live 2026-08-07: `access of union field
/// 'string' while field 'array' is active`).
///
/// Selection follows transformers: the entry named "default", which is what
/// `apply_chat_template` uses unless the caller names another. Falls back to
/// the first entry when nothing is named "default", and returns null for any
/// other shape so the caller drops to its chat_template.jinja / family
/// fallback instead of panicking.
fn chatTemplateFromValue(v: ?std.json.Value) ?[]const u8 {
    const val = v orelse return null;
    switch (val) {
        .string => |s| return if (isIncludeStub(s)) null else s,
        .array => |arr| {
            var first: ?[]const u8 = null;
            for (arr.items) |entry| {
                if (entry != .object) continue;
                const tpl = entry.object.get("template") orelse continue;
                if (tpl != .string) continue;
                if (first == null) first = tpl.string;
                const name = entry.object.get("name") orelse continue;
                if (name == .string and std.mem.eql(u8, name.string, "default")) return tpl.string;
            }
            return first;
        },
        else => return null,
    }
}

test "chat_template accepts HF's list-of-named-templates shape" {
    // Mistral-7B-Instruct-v0.3 ships a LIST (`default` + `tool_use`). Reading
    // it as `.string` panicked the process at load, so the checkpoint could
    // not be served at all.
    const a = std.testing.allocator;

    {   // the shape that panicked: pick the entry named "default"
        const json =
            \\{"chat_template":[{"name":"tool_use","template":"TOOLS"},
            \\{"name":"default","template":"PLAIN"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("PLAIN", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // no "default" named: fall back to the first usable entry
        const json = \\{"chat_template":[{"name":"tool_use","template":"TOOLS"}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("TOOLS", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // the ordinary string shape is untouched
        const json = \\{"chat_template":"BARE"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("BARE", chatTemplateFromValue(p.value.object.get("chat_template")).?);
    }
    {   // junk shapes return null so the caller uses its jinja/family fallback
        const json = \\{"chat_template":[{"name":"x"},{"nope":1}]}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    try std.testing.expect(chatTemplateFromValue(null) == null);

    {   // transformers >= 5 saves the template to chat_template.jinja and leaves
        // an `{% include %}` POINTER in tokenizer_config.json (mlx-community
        // oQ4e builds, issue #169). jinja.cpp has no `include`, so
        // taking the pointer literally fails the render and SILENTLY drops to
        // the generic fallback — wrong-family markers the model then echoes.
        // Read it as "no inline template" so the sidecar file is used.
        const json = \\{"chat_template":"{% include 'chat_template.jinja' %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // whitespace/dash variants are the same pointer
        const json = \\{"chat_template":"\n  {%- include \"chat_template.jinja\" -%}\n"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) == null);
    }
    {   // a real template that merely CONTAINS the word include is untouched
        const json = \\{"chat_template":"{% if x %}include{% endif %}"}
        ;
        var p = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer p.deinit();
        try std.testing.expect(chatTemplateFromValue(p.value.object.get("chat_template")) != null);
    }
}

/// Load chat template configuration from tokenizer_config.json.
pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ChatConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/tokenizer_config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const root = parsed.value.object;

    // DeepSeek's V4.1 template merges dicts with `**`, which jinja.cpp cannot parse; ours is pinned to the encoder.
    const dsv41 = dirModelTypeIs(io, allocator, model_dir, "deepseek_v41");
    const chat_template: []const u8 = if (dsv41)
        try allocator.dupe(u8, @embedFile("fixtures/dsv41_chat_template.jinja"))
    else if (chatTemplateFromValue(root.get("chat_template"))) |t|
        try allocator.dupe(u8, t)
    else blk: {
        // Fall back to chat_template.jinja file (e.g. Qwen3.5 models)
        const jinja_path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{model_dir});
        defer allocator.free(jinja_path);
        if (std.Io.Dir.openFileAbsolute(io, jinja_path, .{})) |f| {
            defer f.close(io);
            var jinja_buf: [4096]u8 = undefined;
            var jinja_reader = f.reader(io, &jinja_buf);
            break :blk try jinja_reader.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
        } else |_| {
            break :blk try allocator.dupe(u8, "");
        }
    };

    const bos_token: ?[]const u8 = if (root.get("bos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const eos_token: ?[]const u8 = if (root.get("eos_token")) |v|
        (if (v == .string) try allocator.dupe(u8, v.string) else null)
    else
        null;

    const add_bos_token = if (root.get("add_bos_token")) |v|
        (if (v == .bool) v.bool else false)
    else
        false;

    return .{
        .chat_template = chat_template,
        .bos_token = bos_token,
        .eos_token = eos_token,
        .add_bos_token = add_bos_token,
        .allocator = allocator,
        .image_placeholder = if (dsv41) dsv41_image_atom else null,
    };
}

/// V4.1's image placeholder token, which the processor expands into the image's span.
pub const dsv41_image_atom = "<\xef\xbd\x9cdeepseek_image\xef\xbd\x9c>";

/// Format chat messages into token IDs using the model's Jinja chat template.
/// `effort` is the client's raw `reasoning_effort` string (see
/// `serializeExtraContext`) — null when the request didn't send one.
pub fn formatChat(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it.
    continue_final: bool,
) ![]u32 {
    var literal_arena = std.heap.ArenaAllocator.init(allocator);
    defer literal_arena.deinit();
    const marked = try markLiteralControlText(literal_arena.allocator(), tok, messages);
    const rendered = try renderChatTemplate(allocator, marked, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer allocator.free(rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    try encodeWithSpecialTokens(allocator, tok, rendered, &ids);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// Renders whose shipped template raised and fell to the generic format; `/props` reports it.
pub var template_fallbacks = std.atomic.Value(u64).init(0);

/// Render the Jinja chat template with the given messages.
pub fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    /// Extend the trailing assistant message instead of answering after it —
    /// the prompt ends mid-turn and generation resumes from its last word.
    continue_final: bool,
) ![]const u8 {
    // Split the partial reply off FIRST, so every transform below (tool
    // fallback synthesis) sees exactly the message list
    // an ordinary turn would have. The template must open the assistant turn
    // once, which it only does when the trailing assistant is not in `messages`.
    const prefill = continuationPrefill(messages, continue_final);
    const msgs = if (prefill != null) messages[0 .. messages.len - 1] else messages;

    if (chat_config.chat_template.len == 0) {
        if (!fallbackIsChatMl(chat_config)) return refuseTemplate("the model has no chat template and no fallback format");
        const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
        if (prefill) |partial| {
            defer allocator.free(base);
            return std.mem.concat(allocator, u8, &.{ base, partial });
        }
        return base;
    }

    // Some chat templates (e.g. DeepSeek V4) don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    // A byte-pinned stock Qwen template renders a late system note in place; the fold below would
    // rewrite every earlier byte of the prompt. Only the render and its probe see the adapted text.
    const adapted = if (hasLateSystem(msgs)) try qwenLateSystemTemplate(allocator, chat_config.chat_template) else null;
    defer if (adapted) |a| allocator.free(a);
    const render_tpl = adapted orelse chat_config.chat_template;
    const tpl = chat_config.chat_template;
    const tpl_has_tools = std.mem.indexOf(u8, tpl, "tools") != null;
    const messages_have_tool_content = messagesHaveToolContent(msgs);
    const extra_json = try serializeExtraContext(allocator, chat_config, enable_thinking, effort);
    defer allocator.free(extra_json);

    // A generic role header is not proof that the template emits every role:
    // some templates guard the header to user/assistant/system. Probe only
    // tool-bearing requests; explicit tool branches remain the zero-cost path.
    const tpl_has_tool_role = !messages_have_tool_content or
        templateReferencesToolRole(tpl) or
        try templateProbePreservesToolContent(allocator, tpl, extra_json);
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and messages_have_tool_content;

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = msgs;
    // A template that refuses a system turn past index 0 (Qwen's raises) gets it folded
    // into the leading one; a template that renders it in place (MiMo's) keeps it there.
    if (hasLateSystem(msgs) and !try templateProbeRendersLateSystem(allocator, render_tpl, extra_json)) {
        fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        var folded = try std.ArrayList(Message).initCapacity(arena_alloc, msgs.len);
        folded.appendSliceAssumeCapacity(msgs);
        _ = try foldSystemMessages(arena_alloc, &folded);
        effective_messages = folded.items;
    }
    var effective_tools_json = tools_json;
    // Optional keys a template may dump blindly (see fillOptionalToolDefKeys).
    // Arena-owned, so the rewritten bytes outlive the render with no free path
    // of their own.
    if (tools_json) |tj| {
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        if (try fillOptionalToolDefKeys(fallback_arena.?.allocator(), tj)) |filled| {
            effective_tools_json = filled;
        }
    }
    if (needs_inject_tools or needs_rewrite_tool_role) {
        // NEVER re-init: the tool-def fill above may already own this arena,
        // and clobbering it orphans everything allocated there (a leak here, a
        // use-after-free the moment `effective_tools_json` points into it).
        if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        effective_messages = try synthesizeToolFallbackMessages(
            arena_alloc,
            effective_messages,
            tools_json,
            tool_choice_instruction,
            needs_inject_tools,
            !tpl_has_tool_role,
            !tpl_has_tools,
        );
        if (needs_inject_tools) {
            effective_tools_json = null; // already inlined as system content
        }
    }
    // The fallback above inlines the instruction with the tools; a template that
    // renders tools itself gets it on the last turn instead. A continuation keeps
    // the client's own prefill.
    if (tool_choice_instruction) |instr| {
        if (tools_json != null and !needs_inject_tools and prefill == null) {
            if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
            effective_messages = try appendTurnInstruction(fallback_arena.?.allocator(), effective_messages, std.mem.trim(u8, instr, "\n"));
        }
    }

    // Serialize messages to JSON — Gemma 4 templates handle role:"tool" natively
    // (producing <|turn>tool in the rendered output). No transformation needed.
    const messages_json = try serializeMessagesJsonFor(allocator, effective_messages, chat_config);
    defer allocator.free(messages_json);

    // Null-terminate strings for C
    const tmpl_z = try allocator.dupeSentinel(u8, render_tpl, 0);
    defer allocator.free(tmpl_z);
    const msgs_z = try allocator.dupeSentinel(u8, messages_json, 0);
    defer allocator.free(msgs_z);
    const extra_z = try allocator.dupeSentinel(u8, extra_json, 0);
    defer allocator.free(extra_z);

    var tools_z: ?[:0]const u8 = null;
    defer if (tools_z) |tz| allocator.free(tz);
    if (effective_tools_json) |tj| {
        tools_z = try allocator.dupeSentinel(u8, tj, 0);
    }

    if (prefill) |partial| {
        const turn = messages[messages.len - 1];
        if (try renderContinuation(allocator, tmpl_z, extra_z, tools_z, effective_messages, turn, partial, chat_config)) |cont| return cont;
    }

    // Length-delimited: a message may carry a raw 0x00, and a C-string read would cut the prompt there.
    var rendered_len: usize = 0;
    const result_ptr = jinja_c.jinja_render_chat(
        tmpl_z.ptr,
        msgs_z.ptr,
        if (tools_z) |tz| tz.ptr else null,
        extra_z.ptr,
        1,
        &rendered_len,
    );

    if (result_ptr) |ptr| {
        defer jinja_c.jinja_str_free(ptr);
        const collapsed = try collapseDoubledThinkTags(allocator, ptr[0..rendered_len]);
        // A continuation commits the content channel and then hands the model
        // its own unfinished sentence. It runs INSTEAD of the thinking-off
        // tail, never beside it — both append the same channel commit, and
        // twice is a malformed turn header.
        if (prefill) |partial| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{
                collapsed, contentChannelTail(collapsed), partial,
            });
        }
        if (noThinkTailSuffix(collapsed, enable_thinking)) |suffix| {
            defer allocator.free(collapsed);
            return std.mem.concat(allocator, u8, &.{ collapsed, suffix });
        }
        return collapsed;
    }

    // WARN, not debug: a failed render silently swaps a ChatML model's prompt into
    // fallbackFormatChat's format, so the downgrade must be visible at the default
    // log level. Any other model has no fallback: its turn tokens are not ChatML's.
    if (!fallbackIsChatMl(chat_config)) {
        const reason = if (jinja_c.jinja_last_error()) |e| std.mem.span(e) else "the template failed to render";
        log.warn("jinja render failed ({s}), and this model has no fallback chat format\n", .{reason});
        return refuseTemplate(reason);
    }
    _ = template_fallbacks.fetchAdd(1, .monotonic);
    if (jinja_c.jinja_last_error()) |e| {
        log.warn("jinja render failed ({s}), falling back to generic chat format\n", .{std.mem.span(e)});
    }
    const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (prefill) |partial| {
        defer allocator.free(base);
        return std.mem.concat(allocator, u8, &.{ base, partial });
    }
    return base;
}

/// A continuation as the template writes the partial turn when it is history,
/// cut right after its text (transformers' `continue_final_message` rule): the
/// think block and separators match every earlier turn, thinking on or off.
/// Null when the render fails or does not carry the text, and the caller falls
/// back to the generation prompt plus the content-channel tail.
fn renderContinuation(
    allocator: std.mem.Allocator,
    tmpl_z: [:0]const u8,
    extra_z: [:0]const u8,
    tools_z: ?[:0]const u8,
    history: []const Message,
    turn: Message,
    partial: []const u8,
    chat_config: *const ChatConfig,
) !?[]const u8 {
    const turns = try allocator.alloc(Message, history.len + 1);
    defer allocator.free(turns);
    @memcpy(turns[0..history.len], history);
    turns[history.len] = .{ .role = "assistant", .content = partial, .reasoning_content = turn.reasoning_content };
    const json = try serializeMessagesJsonFor(allocator, turns, chat_config);
    defer allocator.free(json);
    const json_z = try allocator.dupeSentinel(u8, json, 0);
    defer allocator.free(json_z);
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(tmpl_z.ptr, json_z.ptr, if (tools_z) |tz| tz.ptr else null, extra_z.ptr, 0, &len) orelse return null;
    defer jinja_c.jinja_str_free(ptr);
    const rendered = try collapseDoubledThinkTags(allocator, ptr[0..len]);
    defer allocator.free(rendered);
    // A template may trim the text's leading whitespace, as transformers allows for.
    const text = std.mem.trimStart(u8, partial, " \t\r\n");
    const at = std.mem.lastIndexOf(u8, rendered, text) orelse return null;
    return try allocator.dupe(u8, rendered[0 .. at + text.len]);
}

/// DSV4's chat template emits `</think></think>` between every user→assistant
/// pair in chat mode — the user role appends `</think>` after `<|Assistant|>`
/// and the assistant role prepends `</think>` before its content. The doubling
/// is structurally invalid (no legitimate rendered prompt should contain
/// `</think></think>`), and at 2-bit DQ it pushes the model into prompt-echo
/// / phrase-loop collapse on the second turn (a 24-token "Hello / Hi! / What
/// is 2+2?" chat reliably degenerates). The HF tokenizer renders the same
/// doubling so this is a template-side artifact, not our renderer; mlx-lm
/// must be tolerating it via reduction-order quirks we don't share. Strip
/// it here. Safe across all model families because `</think></think>` never
/// appears legitimately.
fn collapseDoubledThinkTags(allocator: std.mem.Allocator, rendered: []const u8) ![]u8 {
    const tag = "</think>";
    const needle = "</think></think>";
    if (std.mem.indexOf(u8, rendered, needle) == null) {
        return allocator.dupe(u8, rendered);
    }
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.ensureTotalCapacity(allocator, rendered.len);
    var i: usize = 0;
    while (i < rendered.len) {
        // If we see `</think></think>...`, drop the FIRST `</think>` and let
        // the next iteration handle whatever follows — that way 3x, 4x, …
        // also collapse to a single tag without a separate pass.
        if (i + needle.len <= rendered.len and
            std.mem.eql(u8, rendered[i .. i + needle.len], needle))
        {
            i += tag.len; // skip the first `</think>`; next iter writes the survivor
        } else {
            try out.append(allocator, rendered[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice(allocator);
}

/// Explicit tool branches are authoritative and skip the render probe below.
/// A generic header is not enough: it may be guarded to known non-tool roles.
fn templateReferencesToolRole(tpl: []const u8) bool {
    const patterns = [_][]const u8{ "'tool'", "\"tool\"" };
    for (patterns) |p| {
        if (std.mem.indexOf(u8, tpl, p) != null) return true;
    }
    return false;
}

/// Execute a tiny role probe instead of trying to infer Jinja control flow
/// from source spelling. A missing marker means the template filtered the tool
/// message, so the caller rewrites it through the lossless tool fallback.
fn templateProbePreservesToolContent(
    allocator: std.mem.Allocator,
    tpl: []const u8,
    extra_json: []const u8,
) !bool {
    const marker = "__mlx_tool_role_probe__";
    return templateProbeRendersMarker(allocator, tpl, extra_json, &.{
        .{ .role = "tool", .content = marker },
    }, marker);
}

fn hasLateSystem(messages: []const Message) bool {
    if (messages.len < 2) return false;
    for (messages[1..]) |m| {
        if (std.mem.eql(u8, m.role, "system")) return true;
    }
    return false;
}

/// The stock Flash-Next template raises on a system turn that is not first. Replace only that
/// branch so the turn renders in place; any other revision keeps its native contract and the fold.
fn qwenLateSystemTemplate(allocator: std.mem.Allocator, tpl: []const u8) !?[]const u8 {
    const source = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    const content = "render_content(message.content, false, true)|trim";
    const branch =
        \\{%- if message.role == "system" %}
        \\        {%- if not loop.first %}
        \\            {{- raise_exception('System message must be at the beginning.') }}
        \\        {%- endif %}
    ;
    if (!std.mem.eql(u8, tpl, source)) return null;
    if (std.mem.count(u8, tpl, branch) != 1) return null;
    const replacement = try std.mem.concat(allocator, u8, &.{
        "{%- if message.role == \"system\" %}{%- if not loop.first %}{{- '<|im_start|>system\\n' + (",
        content,
        ") + '<|im_end|>\\n' }}{%- endif %}",
    });
    defer allocator.free(replacement);
    return try std.mem.replaceOwned(u8, allocator, tpl, branch, replacement);
}

/// False when the template raises on a system turn that is not first, or drops it.
fn templateProbeRendersLateSystem(
    allocator: std.mem.Allocator,
    tpl: []const u8,
    extra_json: []const u8,
) !bool {
    const marker = "__sushi_late_system_probe__";
    return templateProbeRendersMarker(allocator, tpl, extra_json, &.{
        .{ .role = "user", .content = "u" },
        .{ .role = "system", .content = marker },
        .{ .role = "user", .content = "u" },
    }, marker);
}

fn templateProbeRendersMarker(
    allocator: std.mem.Allocator,
    tpl: []const u8,
    extra_json: []const u8,
    probe_messages: []const Message,
    marker: []const u8,
) !bool {
    const rendered = (try templateProbeRender(allocator, tpl, extra_json, probe_messages)) orelse return false;
    defer allocator.free(rendered);
    return std.mem.indexOf(u8, rendered, marker) != null;
}

/// The template's render of `probe_messages` without a generation prompt, or
/// null when it fails. Probe failures are conservative: the real render clears
/// the thread-local Jinja error and either renders or logs its own failure.
fn templateProbeRender(
    allocator: std.mem.Allocator,
    tpl: []const u8,
    extra_json: []const u8,
    probe_messages: []const Message,
) !?[]u8 {
    const messages_json = try serializeMessagesJson(allocator, probe_messages);
    defer allocator.free(messages_json);

    const tmpl_z = try allocator.dupeSentinel(u8, tpl, 0);
    defer allocator.free(tmpl_z);
    const msgs_z = try allocator.dupeSentinel(u8, messages_json, 0);
    defer allocator.free(msgs_z);
    const extra_z = try allocator.dupeSentinel(u8, extra_json, 0);
    defer allocator.free(extra_z);

    var rendered_len: usize = 0;
    const result_ptr = jinja_c.jinja_render_chat(
        tmpl_z.ptr,
        msgs_z.ptr,
        null,
        extra_z.ptr,
        0,
        &rendered_len,
    ) orelse return null;
    defer jinja_c.jinja_str_free(result_ptr);
    return try allocator.dupe(u8, result_ptr[0..rendered_len]);
}

/// What a forced tool_choice commits once the thought is closed, spelled the
/// way the template renders a call after a closed think block.
pub const ForcedCall = struct {
    /// Whitespace between the think closer and the call.
    sep: []u8,
    /// The call up to the model's first free choice: through the function name
    /// and its closing delimiter when one is named, else up to the name.
    call: []u8,
    /// What closes a name: its delimiter and the whitespace after it.
    name_tail: []u8,

    pub fn deinit(self: *ForcedCall, allocator: std.mem.Allocator) void {
        allocator.free(self.sep);
        allocator.free(self.call);
        allocator.free(self.name_tail);
    }
};

/// Read the forced call from the template's own rendering of an assistant tool
/// call. Null when the template renders no think closer before the call to
/// anchor on, or no delimiter after the function name.
const probe_name = "__sushi_forced_call_probe__";

/// The template's rendering of a turn whose assistant calls `probe_name` after a closed thought.
fn renderProbeCall(allocator: std.mem.Allocator, chat_config: *const ChatConfig) !?[]u8 {
    const calls = [_]ToolCall{.{ .id = "call_probe", .name = probe_name, .arguments = "{\"a\": \"v\"}" }};
    const probe = [_]Message{
        .{ .role = "user", .content = "u" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "r" },
    };
    const extra_json = try serializeExtraContext(allocator, chat_config, true, null);
    defer allocator.free(extra_json);
    return templateProbeRender(allocator, chat_config.chat_template, extra_json, &probe);
}

/// Where the probe's call markup sits: `between` runs from the think closer to the name.
const ProbeCall = struct { between: []const u8, head: []const u8, after: []const u8 };

fn locateProbeCall(rendered: []const u8) ?ProbeCall {
    const name_at = std.mem.indexOf(u8, rendered, probe_name) orelse return null;
    const closer_at = std.mem.lastIndexOf(u8, rendered[0..name_at], BARE_THINK_CLOSER) orelse return null;
    const between = rendered[closer_at + BARE_THINK_CLOSER.len .. name_at];
    const head = std.mem.trimStart(u8, between, " \t\r\n");
    if (head.len == 0) return null;
    return .{ .between = between, .head = head, .after = rendered[name_at + probe_name.len ..] };
}

/// The `<tag>` the template opens every tool call with (`<tool_call>`), or null when its call
/// markup does not start with one.
pub fn callOpener(allocator: std.mem.Allocator, chat_config: *const ChatConfig) !?[]u8 {
    const rendered = (try renderProbeCall(allocator, chat_config)) orelse return null;
    defer allocator.free(rendered);
    const head = (locateProbeCall(rendered) orelse return null).head;
    if (head[0] != '<') return null;
    const end = std.mem.indexOfScalar(u8, head, '>') orelse return null;
    return try allocator.dupe(u8, head[0 .. end + 1]);
}

pub fn forcedCallText(allocator: std.mem.Allocator, chat_config: *const ChatConfig, name: ?[]const u8) !?ForcedCall {
    const rendered = (try renderProbeCall(allocator, chat_config)) orelse return null;
    defer allocator.free(rendered);
    const at = locateProbeCall(rendered) orelse return null;
    const head = at.head;
    const sep = at.between[0 .. at.between.len - head.len];

    const after = at.after;
    var delim: usize = 0;
    while (delim < after.len and delim < 4 and !nameByte(after[delim]) and !std.ascii.isWhitespace(after[delim]) and after[delim] != '<') delim += 1;
    if (delim == 0) return null;
    var ws = delim;
    while (ws < after.len and std.ascii.isWhitespace(after[ws])) ws += 1;
    const tail = after[0..ws];

    const owned_sep = try allocator.dupe(u8, sep);
    errdefer allocator.free(owned_sep);
    const owned_tail = try allocator.dupe(u8, tail);
    errdefer allocator.free(owned_tail);
    const call = try std.mem.concat(allocator, u8, &.{ head, name orelse "", if (name != null) tail else "" });
    return .{ .sep = owned_sep, .call = call, .name_tail = owned_tail };
}

fn nameByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.';
}

/// `messages` with `text` closing the conversation: appended to a trailing user
/// turn, else a user turn of its own. Every byte before the last message is
/// unchanged, so the prefix cache still matches.
fn appendTurnInstruction(arena: std.mem.Allocator, messages: []const Message, text: []const u8) ![]const Message {
    const out = try arena.alloc(Message, messages.len + 1);
    @memcpy(out[0..messages.len], messages);
    if (messages.len > 0 and std.mem.eql(u8, messages[messages.len - 1].role, "user")) {
        const last = &out[messages.len - 1];
        last.content = if (last.content.len == 0) text else try std.fmt.allocPrint(arena, "{s}\n\n{s}", .{ last.content, text });
        return out[0..messages.len];
    }
    out[messages.len] = .{ .role = "user", .content = text };
    return out;
}

/// True if any message has `role: "tool"` or an assistant message with tool_calls.
fn messagesHaveToolContent(messages: []const Message) bool {
    for (messages) |m| {
        if (std.mem.eql(u8, m.role, "tool")) return true;
        if (m.tool_calls != null) return true;
    }
    return false;
}

/// Synthesize a messages array suitable for templates that have no tool/role support.
/// - inject_tool_prompt: prepend (or merge into) a system message with the tool prompt
/// - rewrite_tool_role: rewrite role:"tool" messages to role:"user" with <tool_response> wrapping
/// - rewrite_tool_calls: rewrite assistant messages that carry tool_calls into plain
///   content emitting <tool_call>...</tool_call> blocks (matches what we instruct the
///   model to produce, so multi-turn context stays consistent).
pub fn synthesizeToolFallbackMessages(
    arena: std.mem.Allocator,
    messages: []const Message,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    inject_tool_prompt: bool,
    rewrite_tool_role: bool,
    rewrite_tool_calls: bool,
) ![]Message {
    var out = std.ArrayList(Message).empty;

    var tool_prompt: ?[]const u8 = null;
    if (inject_tool_prompt and tools_json != null) {
        var buf = std.ArrayList(u8).empty;
        try appendToolSystemPrompt(arena, &buf, tools_json.?, tool_choice_instruction);
        tool_prompt = try buf.toOwnedSlice(arena);
    }

    var injected = false;
    if (tool_prompt) |tp| {
        if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) {
            try out.append(arena, .{ .role = "system", .content = tp });
            injected = true;
        }
    }

    for (messages, 0..) |msg, i| {
        if (tool_prompt) |tp| {
            if (!injected and i == 0 and std.mem.eql(u8, msg.role, "system")) {
                const merged = if (msg.content.len > 0)
                    try std.fmt.allocPrint(arena, "{s}\n\n{s}", .{ msg.content, tp })
                else
                    try arena.dupe(u8, tp);
                try out.append(arena, .{
                    .role = "system",
                    .content = merged,
                    .tool_calls = msg.tool_calls,
                    .tool_call_id = msg.tool_call_id,
                    .images = msg.images,
                });
                injected = true;
                continue;
            }
        }

        if (rewrite_tool_role and std.mem.eql(u8, msg.role, "tool")) {
            const wrapped = try std.fmt.allocPrint(arena, "<tool_response>\n{s}\n</tool_response>", .{msg.content});
            try out.append(arena, .{
                .role = "user",
                .content = wrapped,
            });
            continue;
        }

        if (rewrite_tool_calls and std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var buf = std.ArrayList(u8).empty;
            if (msg.content.len > 0) {
                try buf.appendSlice(arena, msg.content);
                try buf.append(arena, '\n');
            }
            for (msg.tool_calls.?) |tc| {
                try buf.appendSlice(arena, "<tool_call>\n{\"name\": \"");
                try buf.appendSlice(arena, tc.name);
                try buf.appendSlice(arena, "\", \"arguments\": ");
                try buf.appendSlice(arena, tc.arguments);
                try buf.appendSlice(arena, "}\n</tool_call>\n");
            }
            const owned = try buf.toOwnedSlice(arena);
            try out.append(arena, .{
                .role = "assistant",
                .content = std.mem.trimEnd(u8, owned, "\n"),
                .images = msg.images,
            });
            continue;
        }

        try out.append(arena, msg);
    }

    return out.toOwnedSlice(arena);
}

/// Deepest `{`/`[` nesting a JSON request may carry: the recursive consumers (echo, schema, template) have no cap of their own.
pub const max_json_nesting: usize = 256;

pub fn jsonNestingExceeds(body: []const u8, limit: usize) bool {
    const first = std.mem.indexOfNone(u8, body, " \t\r\n") orelse return false;
    if (body[first] != '{' and body[first] != '[') return false;
    var depth: usize = 0;
    var in_string = false;
    var escaped = false;
    for (body[first..]) |c| {
        if (in_string) {
            if (escaped) escaped = false else if (c == '\\') escaped = true else if (c == '"') in_string = false;
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{', '[' => {
                depth += 1;
                if (depth > limit) return true;
            },
            '}', ']' => depth -|= 1,
            else => {},
        }
    }
    return false;
}

pub fn serializeMessagesJson(allocator: std.mem.Allocator, messages: []const Message) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, null);
}

pub fn serializeMessagesJsonFor(allocator: std.mem.Allocator, messages: []const Message, chat_config: *const ChatConfig) ![]const u8 {
    return serializeMessagesJsonImpl(allocator, messages, chat_config.image_placeholder);
}

/// `content` as the typed part list vision templates render, each media part
/// a `{"type":"image"|"video"}` placeholder at its offset.
fn appendContentParts(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), content: []const u8, parts: []const MediaPart) !void {
    try buf.append(allocator, '[');
    var from: usize = 0;
    var first = true;
    for (parts) |p| {
        const at = @max(from, @min(p.at, content.len));
        if (at > from) {
            if (!first) try buf.append(allocator, ',');
            try buf.appendSlice(allocator, "{\"type\":\"text\",\"text\":");
            try appendJsonString(allocator, buf, content[from..at]);
            try buf.append(allocator, '}');
            from = at;
            first = false;
        }
        if (!first) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, switch (p.kind) {
            .image => "{\"type\":\"image\"}",
            .video => "{\"type\":\"video\"}",
        });
        first = false;
    }
    if (from < content.len) {
        if (!first) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"type\":\"text\",\"text\":");
        try appendJsonString(allocator, buf, content[from..]);
        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');
}

/// `content` as one string for a template that has no part list: an `atom` for each image at its offset, the
/// text around them, all joined by blank lines. Video has no atom there and is dropped.
fn appendContentText(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), content: []const u8, parts: []const MediaPart, atom: []const u8) !void {
    var text = std.ArrayList(u8).empty;
    defer text.deinit(allocator);
    var from: usize = 0;
    for (parts) |p| {
        const at = @max(from, @min(p.at, content.len));
        if (p.kind != .image) continue;
        if (at > from) try appendJoined(allocator, &text, content[from..at]);
        try appendJoined(allocator, &text, atom);
        from = at;
    }
    if (from < content.len) try appendJoined(allocator, &text, content[from..]);
    try appendJsonString(allocator, buf, text.items);
}

fn appendJoined(allocator: std.mem.Allocator, text: *std.ArrayList(u8), piece: []const u8) !void {
    if (text.items.len > 0) try text.appendSlice(allocator, "\n\n");
    try text.appendSlice(allocator, piece);
}

fn serializeMessagesJsonImpl(allocator: std.mem.Allocator, messages: []const Message, image_atom: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '[');
    for (messages, 0..) |msg, i| {
        if (i > 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "{\"role\":");
        try appendJsonString(allocator, &buf, msg.role);

        try buf.appendSlice(allocator, ",\"content\":");
        if (msg.media_parts != null and msg.media_parts.?.len > 0) {
            if (image_atom) |atom| try appendContentText(allocator, &buf, msg.content, msg.media_parts.?, atom) else try appendContentParts(allocator, &buf, msg.content, msg.media_parts.?);
        } else if (msg.content.len > 0) {
            try appendJsonString(allocator, &buf, msg.content);
        } else try buf.appendSlice(allocator, "null");

        if (msg.tool_calls) |tcs| {
            try buf.appendSlice(allocator, ",\"tool_calls\":[");
            for (tcs, 0..) |tc, ti| {
                if (ti > 0) try buf.append(allocator, ',');
                try buf.appendSlice(allocator, "{\"type\":\"function\",");
                if (tc.id.len > 0) {
                    // A template that names a tool RESULT by matching
                    // `message.tool_call_id` against the history call's `id`
                    // needs the id; inert for templates that never read it.
                    try buf.appendSlice(allocator, "\"id\":");
                    try appendJsonString(allocator, &buf, tc.id);
                    try buf.append(allocator, ',');
                }
                try buf.appendSlice(allocator, "\"function\":{\"name\":");
                try appendJsonString(allocator, &buf, tc.name);
                try buf.appendSlice(allocator, ",\"arguments\":");
                // Templates iterate `arguments|items`, so anything but a JSON object
                // (empty, null, an array, scalar, malformed or over-nested text) would
                // raise or recurse unbounded: embed `{}` instead.
                const is_object = !jsonNestingExceeds(tc.arguments, max_json_nesting) and
                    if (std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{})) |parsed| blk: {
                    defer parsed.deinit();
                    break :blk parsed.value == .object;
                } else |_| false;
                try buf.appendSlice(allocator, if (is_object) tc.arguments else "{}");
                try buf.appendSlice(allocator, "}}");
            }
            try buf.append(allocator, ']');
        }

        if (msg.tool_call_id) |tid| {
            try buf.appendSlice(allocator, ",\"tool_call_id\":");
            try appendJsonString(allocator, &buf, tid);
            // No tool_responses field needed — Gemma 4 templates handle role:"tool"
            // natively via the content field. Adding tool_responses causes the template
            // to render duplicate content, wasting tokens.
        }

        if (msg.reasoning_content) |rc| {
            // Key omitted entirely when absent: templates gate on
            // `message.reasoning_content is string`, and an explicit null
            // would still pass `is defined`-style checks.
            try buf.appendSlice(allocator, ",\"reasoning_content\":");
            try appendJsonString(allocator, &buf, rc);
        }

        try buf.append(allocator, '}');
    }
    try buf.append(allocator, ']');

    return buf.toOwnedSlice(allocator);
}

/// Fill the OPTIONAL tool-definition keys a template may dump blindly.
///
/// `description` is optional in OpenAI's function schema and a no-arg tool
/// needs no `parameters`, but a template that renders `fn.description | tojson`
/// or `tool | tojson` wholesale hits `tojson` on Undefined, which is a
/// jinja RAISE — i.e. a silent `fallbackFormatChat` with no tools preamble at
/// all, for a request that was perfectly legal. Returns null when every
/// function already carries both keys, so the common path re-emits nothing and
/// the caller keeps the client's own bytes (provenance, not content, decides
/// ownership).
///
/// Defaults are the empty ones: `""` says the client supplied no description,
/// and an empty-properties object is what a no-argument tool means. Neither
/// invents a capability the client did not declare.
pub fn fillOptionalToolDefKeys(allocator: std.mem.Allocator, tools_json: []const u8) !?[]u8 {
    // Own the intermediates: growing a parsed ObjectMap allocates OUTSIDE the
    // parse's arena, so mutating with the caller's allocator leaks unless the
    // caller happens to be an arena too. Only the returned bytes are theirs.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const parsed = std.json.parseFromSlice(std.json.Value, a, tools_json, .{}) catch return null;
    if (parsed.value != .array) return null;

    var changed = false;
    for (parsed.value.array.items) |*tool| {
        if (tool.* != .object) continue;
        const fn_val = tool.object.getPtr("function") orelse continue;
        if (fn_val.* != .object) continue;
        if (fn_val.object.get("description") == null) {
            try fn_val.object.put(a, "description", .{ .string = "" });
            changed = true;
        }
        if (fn_val.object.get("parameters") == null) {
            var params: std.json.ObjectMap = .empty;
            try params.put(a, "type", .{ .string = "object" });
            try params.put(a, "properties", .{ .object = .empty });
            try fn_val.object.put(a, "parameters", .{ .object = params });
            changed = true;
        }
    }
    if (!changed) return null;

    return try std.json.Stringify.valueAlloc(allocator, parsed.value, .{});
}

/// Whether the template READS `reasoning_effort` — i.e. the word is a
/// BEHAVIORAL lever on this checkpoint (DeepSeek's preamble, Qwen3.8's and
/// GLM's effort arms) rather than a string we hand over and it ignores.
/// Where it is, the server must NOT also derive a token budget from the same
/// word: the budget only truncates what the client is shown, so the two
/// levers fight — pi asking for `medium` on Qwen3.8 got the model's default
/// (unguided, long) thinking AND a 2048-token display cut, then 25.9k tokens
/// of invisible generation (live 2026-08-14). An explicit
/// `reasoning_budget_tokens` or `--reasoning-budget` is someone asking for a
/// cap on purpose and still applies.
///
/// Substring on the KEY, so a family reading a different one is untouched.
pub fn templateConsumesEffort(tpl: []const u8) bool {
    return std.mem.indexOf(u8, tpl, "reasoning_effort") != null;
}

/// OpenAI's effort vocabulary -> DeepSeek-V4.1's low|high|max (the encoder's default is high).
fn dsv41EffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "high";
    if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) return "low";
    if (std.mem.eql(u8, e, "xhigh") or std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "ultra")) return "max";
    return "high";
}

/// OpenAI's effort vocabulary -> Qwen3.8's xhigh|medium|low. The template
/// raise_exception's on any other string and its own default is xhigh, so an
/// absent or unrecognized effort keeps the checkpoint default. The template
/// never reads the key on its thinking-off arm, so the value is inert there.
fn qwen38EffortFor(effort: ?[]const u8, enable_thinking: bool) []const u8 {
    if (!enable_thinking) return "low";
    // An ABSENT effort takes OUR default, not the checkpoint's. The template
    // defaults to xhigh, whose preamble asks the model to validate assumptions
    // and consider alternatives, and nothing bounds the result: a pi agent turn
    // that sent no effort ran 16803 reasoning tokens before the client gave up
    // (live 2026-08-14). The server-side `reasoning_effort` budget is NOT the
    // answer to that — it truncates rather than shortens, and on exhaustion the
    // model is still inside its block, so the rest of the thought streams as
    // CONTENT. Low is the only lever that makes the model itself stop early
    // ("keep your thinking brief and focused, moving directly to the
    // conclusion"). A client that wants the checkpoint default still gets it by
    // asking for it; only silence changed meaning.
    const e = effort orelse return "low";
    if (std.mem.eql(u8, e, "medium")) return "medium";
    if (std.mem.eql(u8, e, "low") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "none")) return "low";
    return "xhigh";
}

/// `effort` is the client's raw `reasoning_effort` string (null when the
/// request didn't send one) — today only the dsv4 family maps it into the
/// template; other families keep their fixed vocabulary.
fn serializeExtraContext(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool, effort: ?[]const u8) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);

    try buf.append(allocator, '{');
    var need_comma = false;

    if (chat_config.bos_token) |bos| {
        try buf.appendSlice(allocator, "\"bos_token\":");
        try appendJsonString(allocator, &buf, bos);
        need_comma = true;
    }
    if (chat_config.eos_token) |eos| {
        if (need_comma) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, "\"eos_token\":");
        try appendJsonString(allocator, &buf, eos);
        need_comma = true;
    }
    // Qwen3.8's effort vocabulary, sniffed on the jinja STRING LITERAL 'xhigh'
    // — it appears in both the `reasoning_effort|default('xhigh')` line and the
    // accepted-values tuple, so a reworded raise message can't drift the
    // detection. EVERY 3.8 template raises on OpenAI's "high", so this keys the
    // mapping for the whole family.
    const qwen38_style = std.mem.indexOf(u8, chat_config.chat_template, "'xhigh'") != null;
    if (need_comma) try buf.append(allocator, ',');
    if (enable_thinking) {
        try buf.appendSlice(allocator, "\"enable_thinking\":true");
    } else {
        try buf.appendSlice(allocator, "\"enable_thinking\":false");
    }
    // Templates that ignore `reasoning_effort` get high/no_think. The families
    // that read it each have their own vocabulary (GLM's free word, Qwen3.8's
    // xhigh|medium|low, DeepSeek's low|high|max) and raise on any other string:
    // a raise swaps in fallbackFormatChat (wrong-family tags), so every one is
    // mapped below instead of sent the default.
    const dsv41_style = std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null;
    const glm5_style = std.mem.indexOf(u8, chat_config.chat_template, "effective_reasoning_effort") != null and
        std.mem.indexOf(u8, chat_config.chat_template, "[gMASK]<sop>") != null;
    if (glm5_style) {
        // Sushi defaults to high; depth is a prompt instruction, never a token cap.
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, effort orelse "high");
        try buf.append(allocator, '"');
    } else if (dsv41_style) {
        // V4.1 renders a numeric budget for every thinking turn, on its own ladder.
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, dsv41EffortFor(effort));
        try buf.append(allocator, '"');
    } else if (qwen38_style) {
        try buf.appendSlice(allocator, ",\"reasoning_effort\":\"");
        try buf.appendSlice(allocator, qwen38EffortFor(effort, enable_thinking));
        try buf.append(allocator, '"');
    } else try buf.appendSlice(allocator, if (enable_thinking)
        ",\"reasoning_effort\":\"high\""
    else
        ",\"reasoning_effort\":\"no_think\"");

    // DeepSeek-V4.1's template switches on `thinking_mode` ("chat"|"thinking",
    // the reference encoder's own vocabulary). Only emitted when the template
    // reads it — other families never see the key.
    if (std.mem.indexOf(u8, chat_config.chat_template, "thinking_mode") != null) {
        try buf.appendSlice(allocator, if (enable_thinking)
            ",\"thinking_mode\":\"thinking\""
        else
            ",\"thinking_mode\":\"chat\"");
    }

    if (chat_config.preserve_thinking) |keep| {
        if (std.mem.indexOf(u8, chat_config.chat_template, "preserve_thinking") != null) {
            try buf.appendSlice(allocator, if (keep) ",\"preserve_thinking\":true" else ",\"preserve_thinking\":false");
        }
    }

    try buf.append(allocator, '}');
    return buf.toOwnedSlice(allocator);
}

/// Encode a rendered prompt: its own control tokens (turn markers, think tags) become ids, and
/// the control-token text a sender put in a message (see `markLiteralControlText`) stays text.
fn encodeWithSpecialTokens(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    text: []const u8,
    ids: *std.ArrayList(u32),
) !void {
    const segment_ids = try tok.encodeMarked(allocator, text);
    defer allocator.free(segment_ids);
    try ids.appendSlice(allocator, segment_ids);
}

/// Message text a client sends is data, so a `<think>` or `<|im_end|>` in a user, system or tool
/// message must not become a control token the model reads as a real boundary. Assistant text is
/// left alone: the template parses it (`</think>` split) and it is the model's own earlier output.
pub fn markLiteralControlText(arena: std.mem.Allocator, tok: *const Tokenizer, messages: []const Message) ![]const Message {
    var out: ?[]Message = null;
    for (messages, 0..) |m, i| {
        if (std.mem.eql(u8, m.role, "assistant")) continue;
        const text = try tok.markLiteralSpecials(arena, m.content);
        if (std.mem.eql(u8, text, m.content)) continue;
        if (out == null) out = try arena.dupe(Message, messages);
        out.?[i].content = text;
        if (m.media_parts) |parts| {
            const shifted = try arena.dupe(MediaPart, parts);
            for (shifted) |*p| p.at = markedOffset(text, p.at);
            out.?[i].media_parts = shifted;
        }
    }
    return out orelse messages;
}

/// Where byte `raw_at` of the unmarked content sits in the marked text, just before any mark there.
fn markedOffset(marked: []const u8, raw_at: usize) usize {
    var k: usize = 0;
    var raw: usize = 0;
    while (k < marked.len and raw < raw_at) {
        if (std.mem.startsWith(u8, marked[k..], tokenizer_mod.literal_mark)) {
            k += tokenizer_mod.literal_mark.len;
        } else {
            k += 1;
            raw += 1;
        }
    }
    return k;
}

const SpecialEntry = struct {
    text: []const u8,
    id: u32,
};

threadlocal var template_refusal_buf: [160]u8 = undefined;
threadlocal var template_refusal_len: usize = 0;

/// Why the last render on this thread was refused; the HTTP error names it.
pub fn templateRefusal() []const u8 {
    return template_refusal_buf[0..template_refusal_len];
}

fn refuseTemplate(reason: []const u8) error{ChatTemplateUnusable} {
    template_refusal_len = @min(reason.len, template_refusal_buf.len);
    @memcpy(template_refusal_buf[0..template_refusal_len], reason[0..template_refusal_len]);
    return error.ChatTemplateUnusable;
}

/// The fallback format is ChatML, so only a model whose turns end with `<|im_end|>` may use it.
fn fallbackIsChatMl(chat_config: *const ChatConfig) bool {
    const eos = chat_config.eos_token orelse return false;
    return std.mem.indexOf(u8, eos, "<|im_end|>") != null;
}

/// Fallback chat formatting (ChatML) for when Jinja rendering fails.
fn fallbackFormatChat(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
) ![]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);

    const has_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

    if (tools_json != null and !has_system) {
        try result.appendSlice(allocator, "<|im_start|>system\n");
        try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
        try result.appendSlice(allocator, "<|im_end|>\n");
    }

    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system") and tools_json != null) {
            try result.appendSlice(allocator, "<|im_start|>system\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "\n\n");
            try appendToolSystemPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            try result.appendSlice(allocator, "<|im_end|>\n");
        } else if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            try result.appendSlice(allocator, "<|im_start|>assistant\n");
            if (msg.content.len > 0) {
                try result.appendSlice(allocator, msg.content);
                try result.appendSlice(allocator, "\n");
            }
            for (msg.tool_calls.?) |tc| {
                try result.appendSlice(allocator, "<tool_call>\n");
                try result.appendSlice(allocator, "{\"name\": \"");
                try result.appendSlice(allocator, tc.name);
                try result.appendSlice(allocator, "\", \"arguments\": ");
                try result.appendSlice(allocator, tc.arguments);
                try result.appendSlice(allocator, "}\n</tool_call>");
            }
            try result.appendSlice(allocator, "<|im_end|>\n");
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            try result.appendSlice(allocator, "<|im_start|>user\n");
            try result.appendSlice(allocator, "<tool_response>\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "\n</tool_response>");
            try result.appendSlice(allocator, "<|im_end|>\n");
        } else {
            try result.appendSlice(allocator, "<|im_start|>");
            try result.appendSlice(allocator, msg.role);
            try result.appendSlice(allocator, "\n");
            try result.appendSlice(allocator, msg.content);
            try result.appendSlice(allocator, "<|im_end|>\n");
        }
    }
    try result.appendSlice(allocator, "<|im_start|>assistant\n");
    if (std.mem.indexOf(u8, chat_config.chat_template, "enable_thinking") != null) {
        if (enable_thinking) {
            try result.appendSlice(allocator, "<think>\n");
        } else {
            try result.appendSlice(allocator, "<think>\n\n</think>\n\n");
        }
    }

    return result.toOwnedSlice(allocator);
}

/// True when `s` strict-parses as a JSON object. Cheap arbiter for the
/// parseToolCalls safety net.
pub fn jsonIsValidObject(allocator: std.mem.Allocator, s: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
    /// True when the call was HEURISTICALLY inferred from bare JSON in the
    /// output (no tag syntax — the model never said "this is a tool call").
    /// The request chokepoint validates inferred names against the declared
    /// tools schema: a truncated DATA object like {"name": "George
    /// Washington", "num": 1, …} must never become a tool call (live pi
    /// capture 2026-07-13). Explicit tag-format calls keep flowing even with
    /// unknown names — "tool not found" is feedback the model corrects from.
    inferred: bool = false,
};

/// Locate a tool's `properties` map in an OpenAI-shaped tools array. Accepts
/// both the wrapped (`{"type":"function","function":{…}}`) and flat forms.
/// The tool's JSON-Schema `parameters` object (Anthropic `input_schema` too).
/// Callers that need only the property map use `toolPropertiesFor`; the hoist
/// also needs `required`, which lives beside `properties` here.
fn toolParametersFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    if (tools != .array) return null;
    for (tools.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv != .string or !std.mem.eql(u8, nv.string, name)) continue;
        const params = func.get("parameters") orelse func.get("input_schema") orelse return null;
        if (params != .object) return null;
        return params.object;
    }
    return null;
}

fn propertiesOf(params: std.json.ObjectMap) ?std.json.ObjectMap {
    const props = params.get("properties") orelse return null;
    if (props != .object) return null;
    return props.object;
}

fn toolPropertiesFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    return propertiesOf(toolParametersFor(tools, name) orelse return null);
}

/// The types a misplaced value may be hoisted on. A required OBJECT/ARRAY param
/// found inside another container is too speculative to move — only scalars.
fn isScalarJsonType(want: []const u8) bool {
    return std.mem.eql(u8, want, "string") or std.mem.eql(u8, want, "integer") or
        std.mem.eql(u8, want, "number") or std.mem.eql(u8, want, "boolean");
}

const SchemaTypes = struct {
    items: [4][]const u8 = undefined,
    len: usize = 0,

    fn add(self: *SchemaTypes, t: []const u8) void {
        if (std.mem.eql(u8, t, "null") or self.len == self.items.len) return;
        for (self.items[0..self.len]) |have| if (std.mem.eql(u8, have, t)) return;
        self.items[self.len] = t;
        self.len += 1;
    }

    fn slice(self: *const SchemaTypes) []const []const u8 {
        return self.items[0..self.len];
    }
};

const schema_ref_depth = 8;

/// The non-null JSON types a property declares, in declaration order. `"type"` may be a union
/// array; without one, `anyOf`/`oneOf` contribute every branch, `allOf` its first typed branch,
/// and a local `$ref` into `$defs`/`definitions` of `root` its target. Odd/absent → empty,
/// which means "leave the value alone".
fn collectJsonTypes(prop: std.json.Value, root: std.json.ObjectMap, out: *SchemaTypes, depth: u8) void {
    if (prop != .object or depth > schema_ref_depth) return;
    if (prop.object.get("type")) |t| {
        switch (t) {
            .string => |s| out.add(s),
            .array => |arr| for (arr.items) |item| if (item == .string) out.add(item.string),
            else => {},
        }
        return;
    }
    if (prop.object.get("$ref")) |r| {
        if (r != .string) return;
        inline for (.{ "#/$defs/", "#/definitions/" }) |prefix| {
            if (std.mem.startsWith(u8, r.string, prefix)) {
                const defs = root.get(prefix[2 .. prefix.len - 1]) orelse return;
                if (defs != .object) return;
                if (defs.object.get(r.string[prefix.len..])) |target| collectJsonTypes(target, root, out, depth + 1);
                return;
            }
        }
        return;
    }
    inline for (.{ "anyOf", "oneOf" }) |key| {
        if (prop.object.get(key)) |branches| {
            if (branches == .array) for (branches.array.items) |b| collectJsonTypes(b, root, out, depth + 1);
            return;
        }
    }
    if (prop.object.get("allOf")) |branches| {
        if (branches == .array) for (branches.array.items) |b| {
            collectJsonTypes(b, root, out, depth + 1);
            if (out.len > 0) return;
        };
    }
}

/// The first type a property declares (see `collectJsonTypes`), or null.
fn declaredJsonType(prop: std.json.Value, root: std.json.ObjectMap) ?[]const u8 {
    var types = SchemaTypes{};
    collectJsonTypes(prop, root, &types, 0);
    return if (types.len > 0) types.items[0] else null;
}

/// Tolerant boolean spelling — the union of what weak models actually emit:
/// Python's `False`, shell's `0`/`1`, prose `yes`/`no`, plus the trailing-comma
/// weld the app's `appendFlagIsTrue` already tolerates. Anything else → null
/// (never guess; an honest client validation error beats a wrong value).
fn looseBoolSpelling(raw: []const u8) ?bool {
    const s = std.mem.trim(u8, raw, " \t\r\n,");
    if (s.len == 0 or s.len > 5) return null;
    var buf: [5]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..s.len], s);
    if (std.mem.eql(u8, lower, "true") or std.mem.eql(u8, lower, "yes") or
        std.mem.eql(u8, lower, "on") or std.mem.eql(u8, lower, "1")) return true;
    if (std.mem.eql(u8, lower, "false") or std.mem.eql(u8, lower, "no") or
        std.mem.eql(u8, lower, "off") or std.mem.eql(u8, lower, "0")) return false;
    return null;
}

/// Strict parse of a container string that admits a key repeated with the SAME
/// value (issue #402: a model spelling `header` twice per item) and refuses a
/// key repeated with different values: the two are told apart by parsing
/// first-wins and last-wins and comparing the documents.
fn parseContainerAllowingRepeats(allocator: std.mem.Allocator, text: []const u8) ?std.json.Parsed(std.json.Value) {
    var first = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_first }) catch return null;
    var last = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_last }) catch {
        first.deinit();
        return null;
    };
    defer last.deinit();
    const a = std.json.Stringify.valueAlloc(allocator, first.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(a);
    const b = std.json.Stringify.valueAlloc(allocator, last.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(b);
    if (!std.mem.eql(u8, a, b)) {
        first.deinit();
        return null;
    }
    return first;
}

/// Coerce ONE value to the declared type. Returns null when the value already
/// matches or the spelling is undecidable. Allocated replacements are appended
/// to `owned`; parsed sub-documents to `docs` — both are freed by the caller
/// AFTER the object is re-serialized, because the returned Value borrows them.
fn coerceValueToType(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    want: []const u8,
    owned: *std.ArrayList([]const u8),
    docs: *std.ArrayList(std.json.Parsed(std.json.Value)),
) !?std.json.Value {
    // A CONTAINER param carried through a tag format arrives as a STRING — no tag
    // format types its values. Live (15 hits in one session): pi's `edit.edits`,
    // an array of {oldText,newText}, shipped as `"[{\"oldText\":…}]"` on every
    // call. Strict parse only: a value that isn't well-formed JSON of the
    // DECLARED kind is left alone rather than guessed at.
    if (std.mem.eql(u8, want, "array") or std.mem.eql(u8, want, "object")) {
        if (v != .string) return null;
        const t = std.mem.trim(u8, v.string, " \t\r\n");
        const want_array = want[0] == 'a';
        if (t.len == 0 or t[0] != (if (want_array) @as(u8, '[') else @as(u8, '{'))) return null;

        // Strict first; a well-formed container is the common case.
        if (parseContainerAllowingRepeats(allocator, t)) |doc| {
            const kind_ok = if (want_array) doc.value == .array else doc.value == .object;
            if (kind_ok) {
                try docs.append(allocator, doc);
                return doc.value;
            }
            var d = doc;
            d.deinit();
            return null;
        }

        // The container string is itself MANGLED (live: pi's `edits` array with a
        // missing comma between two object values). Same big-file escaping class
        // as looseRepairToolCallJson — re-serialize tolerantly, then STRICT-parse
        // the result so a mis-repair that yields invalid JSON is discarded.
        const repaired = looseRepairContainer(allocator, t) orelse return null;
        defer allocator.free(repaired);
        var rdoc = parseContainerAllowingRepeats(allocator, repaired) orelse {
            log.debug("[tool-autocorrect] {s} param left as a string: not well-formed JSON of that kind\n", .{want});
            return null;
        };
        const rkind_ok = if (want_array) rdoc.value == .array else rdoc.value == .object;
        if (!rkind_ok) {
            rdoc.deinit();
            return null;
        }
        try docs.append(allocator, rdoc); // ownership transfers; never deinit here
        return rdoc.value;
    }
    if (std.mem.eql(u8, want, "boolean")) {
        return switch (v) {
            .bool => null,
            .string => |s| if (looseBoolSpelling(s)) |b| std.json.Value{ .bool = b } else null,
            .integer => |i| if (i == 0 or i == 1) std.json.Value{ .bool = i == 1 } else null,
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "integer")) {
        return switch (v) {
            .string => |s| blk: {
                const n = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\r\n"), 10) catch break :blk null;
                break :blk std.json.Value{ .integer = n };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "number")) {
        return switch (v) {
            .string => |s| blk: {
                const t = std.mem.trim(u8, s, " \t\r\n");
                if (std.fmt.parseInt(i64, t, 10)) |n| break :blk std.json.Value{ .integer = n } else |_| {}
                const f = std.fmt.parseFloat(f64, t) catch break :blk null;
                if (std.math.isNan(f) or std.math.isInf(f)) break :blk null; // not representable in JSON
                break :blk std.json.Value{ .float = f };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "string")) {
        // The inverse repair: the tag/custom formats infer type from the value's
        // SPELLING (isJsonLiteral), so a string param whose content happens to
        // read as `false`/`42` was shipped as a bool/number. The schema is the
        // only thing that can tell those apart.
        return switch (v) {
            .bool => |b| std.json.Value{ .string = if (b) "true" else "false" },
            .integer => |i| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .float => |f| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .number_string => |s| std.json.Value{ .string = s },
            else => null,
        };
    }
    return null; // a type we don't model — never guessed
}

fn jsonTypeMatches(v: std.json.Value, want: []const u8) bool {
    if (v == .null) return true; // an explicit null satisfies any optional field
    if (std.mem.eql(u8, want, "boolean")) return v == .bool;
    if (std.mem.eql(u8, want, "integer")) return v == .integer or v == .number_string;
    if (std.mem.eql(u8, want, "number")) return v == .integer or v == .float or v == .number_string;
    if (std.mem.eql(u8, want, "string")) return v == .string;
    if (std.mem.eql(u8, want, "array")) return v == .array;
    if (std.mem.eql(u8, want, "object")) return v == .object;
    return true; // a type we don't model constrains nothing
}

/// True when every argument whose type the tool DECLARES actually carries that
/// JSON type. This is the invariant strict clients (Claude Code, pi, opencode)
/// enforce and reject on; the format corpus asserts it universally so a new
/// entry in any family is covered without a bespoke assertion. Undeclared keys
/// and unmodelled types constrain nothing.
pub fn toolCallConformsToSchema(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return true;
    const props = propertiesOf(params) orelse return true;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const prop = props.get(entry.key_ptr.*) orelse continue;
        var types = SchemaTypes{};
        collectJsonTypes(prop, params, &types, 0);
        if (types.len == 0) continue;
        for (types.slice()) |want| {
            if (jsonTypeMatches(entry.value_ptr.*, want)) break;
        } else return false;
    }
    return true;
}

/// True when a REQUIRED scalar param is MISSING at the top level while sitting
/// buried inside one of the tool's declared container args — exactly the shape
/// `hoistMisplacedRequiredParams` exists to repair, and exactly what a strict
/// client rejects with "must have required properties X". The format corpus
/// asserts this is FALSE for every entry that declares tools, so a future family
/// that produces the buried shape is covered without a bespoke assertion.
pub fn requiredParamIsBuried(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return false;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return false;
    const props_v = params.get("properties") orelse return false;
    const required_v = params.get("required") orelse return false;
    if (props_v != .object or required_v != .array) return false;
    const props = props_v.object;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    for (required_v.array.items) |req_v| {
        if (req_v != .string) continue;
        const name = req_v.string;
        if (parsed.value.object.get(name) != null) continue;
        const want = declaredJsonType(props.get(name) orelse continue, params) orelse continue;
        if (!isScalarJsonType(want)) continue;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const prop = props.get(e.key_ptr.*) orelse continue;
            if (containerItemDeclares(prop, name)) continue;
            if (buriedParamIn(e.value_ptr, name, want, false) != null) return true;
        }
    }
    return false;
}

/// Coerce every parsed call's `arguments` to the types its tool DECLARES.
///
/// Class: **type inference from value spelling.** Neither the tag formats
/// (`<parameter=x>False</parameter>`, Gemma's `key:false`) nor a model writing
/// JSON by hand carries type information a parser can trust — `isJsonLiteral`
/// has to guess from the bytes, and it guesses wrong in both directions:
///   • `False` / `"false"` on a boolean param → shipped as the STRING "false";
///   • `false` / `42` as a string param's CONTENT → shipped as a bool/number.
/// Strict clients (Claude Code, pi, opencode) reject both and the model, which
/// cannot see its own serialized request, loops forever "fixing" a value that
/// was already correct. The tool schema is the only disambiguator, and it is
/// already threaded to every parse site as `tools_json`.
///
/// Contract: only where the schema declares a type, and only when the spelling
/// is unambiguous. An undecidable value is left alone so the client's
/// validation error stays honest. Correct calls from strong models are
/// byte-unchanged (nothing re-serializes unless a coercion actually fired).
///
/// Scalars AND containers: a tag format types nothing, so a declared
/// `array`/`object` arrives as a string of JSON and `coerceValueToType` parses
/// it back (strict, then one tolerant repair re-validated strictly). This
/// comment used to say "only SCALARS are touched" — it was two weeks stale by
/// the time it was read as a spec and filed as issue #94. A contract comment is
/// load-bearing documentation: pin it with a test (`issue #94: a container
/// param carried as literal TEXT…`), don't let it drift.
pub fn coerceToolArgsToSchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props = propertiesOf(params) orelse continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var owned = std.ArrayList([]const u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }
        // Parsed container sub-documents: the rewritten object BORROWS from these,
        // so they must outlive the Stringify below.
        var docs = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
        defer {
            for (docs.items) |*d| d.deinit();
            docs.deinit(allocator);
        }

        var changed = false;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const prop = props.get(entry.key_ptr.*) orelse continue;
            var types = SchemaTypes{};
            collectJsonTypes(prop, params, &types, 0);
            if (types.len == 0) continue;
            // Among several candidates a value that already fits one is kept.
            if (types.len > 1) {
                const fits = for (types.slice()) |want| {
                    if (jsonTypeMatches(entry.value_ptr.*, want)) break true;
                } else false;
                if (fits) continue;
            }
            const new_val = for (types.slice()) |want| {
                if (try coerceValueToType(allocator, entry.value_ptr.*, want, &owned, &docs)) |nv| break nv;
            } else continue;
            entry.value_ptr.* = new_val;
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        log.info("  [tool-parse] coerced {s} arguments to the declared schema types\n", .{call.name});
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does this container's ITEM schema declare `name` as one of its own
/// properties? If it does, a `name` sitting inside it BELONGS there and must
/// never be lifted out — a multi-file edit tool would lose its per-item paths.
/// This is the whole safety argument for the hoist below, so it is deliberately
/// conservative: no declared item `properties` map ⇒ nothing is proven ⇒ false.
fn containerItemDeclares(prop: std.json.Value, name: []const u8) bool {
    if (prop != .object) return false;
    // Array property → the item schema. Object property → the schema itself.
    const inner = if (prop.object.get("items")) |items| items else prop;
    if (inner != .object) return false;
    const props = inner.object.get("properties") orelse return false;
    if (props != .object) return false;
    return props.object.get(name) != null;
}

/// Scalar equality — the only shapes we ever hoist on. A container value is
/// never a candidate, so a structural compare is not needed.
fn jsonScalarEql(a: std.json.Value, b: std.json.Value) bool {
    return switch (a) {
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .integer => |i| b == .integer and i == b.integer,
        .bool => |x| b == .bool and x == b.bool,
        else => false,
    };
}

/// Find `name` buried one level down inside `container`, requiring UNANIMITY:
/// every object in the container carries `name`, all with the same scalar value,
/// and that value matches the type the schema declares for the top-level param.
/// Anything less is ambiguous → null. With `remove`, also deletes it from each
/// object (the returned value keeps pointing into the parse arena, which
/// outlives the removal).
fn buriedParamIn(
    container: *std.json.Value,
    name: []const u8,
    want: []const u8,
    remove: bool,
) ?std.json.Value {
    switch (container.*) {
        .array => |*arr| {
            if (arr.items.len == 0) return null;
            var found: ?std.json.Value = null;
            for (arr.items) |*item| {
                if (item.* != .object) return null;
                const v = item.object.get(name) orelse return null; // must be in EVERY item
                if (v == .null or !jsonTypeMatches(v, want)) return null;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) return null; // items disagree → ambiguous
                } else found = v;
            }
            if (remove) for (arr.items) |*item| {
                _ = item.object.orderedRemove(name);
            };
            return found;
        },
        .object => |*obj| {
            const v = obj.get(name) orelse return null;
            if (v == .null or !jsonTypeMatches(v, want)) return null;
            if (remove) _ = obj.orderedRemove(name);
            return v;
        },
        else => return null,
    }
}

/// Lift a REQUIRED top-level parameter the model BURIED inside a container arg.
///
/// Class: **misplaced required param.** A weak model that has internalized "the
/// edit object holds everything about the edit" writes the file path INSIDE each
/// `edits[]` item and never emits it at the top level:
///
///     {"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
///
/// The call is valid JSON and correctly escaped, so no repair path fires and no
/// type is wrong — it is purely in the wrong PLACE, which only the schema can
/// know. Strict clients reject it ("path: must have required properties path"),
/// and the model, which cannot see its own serialized request, re-emits the same
/// call until it gives up and falls back to rewriting the whole file with
/// `write` — three dead multi-thousand-token rounds in the live 2026-07-13 pi
/// session (gemma-4-26B-A4B). Same shape as the Claude Code replace_all loop.
///
/// Contract — every condition is read off the SCHEMA, never guessed:
///   • the param is declared REQUIRED at the top level and is ABSENT there;
///   • it is declared a SCALAR (hoisting a container is too speculative);
///   • it sits inside a DECLARED container arg whose item schema does NOT
///     declare it (`containerItemDeclares` — otherwise it belongs where it is);
///   • every object in that container carries it with the SAME value and the
///     declared type.
/// Ambiguity of any kind (items disagreeing, two containers offering different
/// values, a wrong-typed value) leaves the call untouched so the client's
/// validation error stays honest. A compliant call never re-serializes.
pub fn hoistMisplacedRequiredParams(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props_v = params.get("properties") orelse continue;
        const required_v = params.get("required") orelse continue;
        if (props_v != .object or required_v != .array) continue;
        const props = props_v.object;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var changed = false;
        for (required_v.array.items) |req_v| {
            if (req_v != .string) continue;
            const name = req_v.string; // borrows from `tools`, which outlives the Stringify
            if (parsed.value.object.get(name) != null) continue; // present — nothing misplaced

            const want = declaredJsonType(props.get(name) orelse continue, params) orelse continue;
            if (!isScalarJsonType(want)) continue;

            // Probe every declared container WITHOUT mutating: a value found in
            // two containers that disagree is ambiguous, and we must not have
            // already gutted the first one by then.
            var found: ?std.json.Value = null;
            var ambiguous = false;
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                const v = buriedParamIn(e.value_ptr, name, want, false) orelse continue;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) {
                        ambiguous = true;
                        break;
                    }
                } else found = v;
            }
            if (ambiguous or found == null) continue;

            // Commit: strip it from the containers, install it at the top level.
            it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                _ = buriedParamIn(e.value_ptr, name, want, true);
            }
            try parsed.value.object.put(parsed.arena.allocator(), name, found.?);
            log.info("  [tool-parse] hoisted misplaced required arg '{s}' to the top level for {s}\n", .{ name, call.name });
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does the OpenAI-shaped tools array declare a function with this name?
/// Accepts both the wrapped ({"type":"function","function":{…}}) and flat
/// forms, mirroring `toolPropertiesFor`. An unparseable/odd `tools_json`
/// returns TRUE — never drop a call because OUR schema parse failed.
pub fn toolNameIsDeclared(allocator: std.mem.Allocator, tools_json: []const u8, name: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .array) return true;
    for (parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv == .string and std.mem.eql(u8, nv.string, name)) return true;
    }
    return false;
}

/// Drop HEURISTICALLY-inferred raw-JSON calls (`ParsedToolCall.inferred`)
/// whose name the request never declared. The George Washington class: a
/// generation truncated mid-data-script leaves `{"name": "George Washington",
/// "num": 1, …}` as the first balanced object, and the flat-shape synthesis
/// promoted it to a tool call the client can only answer with "tool not
/// found" — burning the turn. Explicit tag-format calls are NEVER dropped
/// here (undeclared name → honest client feedback the model corrects from).
///
/// Returns the retained slice (shrunk in place), or null when every call was
/// dropped — the caller treats that exactly like "no tool calls", so the text
/// stays visible content and finish_reason (e.g. "length") is preserved.
pub fn filterInferredBySchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !?[]ParsedToolCall {
    var kept: usize = 0;
    for (calls) |tc| {
        if (!tc.inferred or toolNameIsDeclared(allocator, tools_json, tc.name)) {
            calls[kept] = tc;
            kept += 1;
        } else {
            log.warn("  [tool-parse] dropped inferred raw-JSON call '{s}' — name not among the request's tools (data object, not a call)\n", .{tc.name});
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    if (kept == calls.len) return calls;
    if (kept == 0) {
        allocator.free(calls);
        return null;
    }
    return try allocator.realloc(calls, kept);
}

/// Last-resort tool-call inference for models that emit JUST the arguments
/// object — no tool name, no wrapper syntax. Observed live from Gemma 4 12B
/// via Claude Code: ```` ```json\n{"file_path": …, "content": …}\n``` ````
/// with the Write tool defined; parseToolCalls finds nothing (no "name" key)
/// and the un-executed JSON leaked into the visible text.
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • the visible content (after a leading think block) must START with the
///     JSON object, optionally wrapped in a markdown fence;
///   • the object's keys must satisfy exactly ONE tool in `tools_json`
///     (OpenAI shape): required ⊆ keys AND keys ⊆ properties;
///   • zero or multiple matching tools → null (ambiguity never guesses).
pub fn inferBareJsonToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    // Strip a leading think block, mirroring parseToolCalls.
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    var trimmed = std.mem.trimStart(u8, effective_text, "\n \t");
    // Skip one opening markdown fence line (``` or ```json etc.).
    if (std.mem.startsWith(u8, trimmed, "```")) {
        const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return null;
        trimmed = std.mem.trimStart(u8, trimmed[nl + 1 ..], "\n \t");
    }
    // The object must LEAD the visible content — a JSON example mid-prose is
    // never promoted to a call.
    if (!std.mem.startsWith(u8, trimmed, "{")) return null;
    const obj_slice = balancedJsonObject(trimmed) orelse return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, obj_slice, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    if (obj.count() == 0) return null;

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var match_name: ?[]const u8 = null;
    for (tools_parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        const params_val = func_val.object.get("parameters") orelse continue;
        if (params_val != .object) continue;
        const props_val = params_val.object.get("properties") orelse continue;
        if (props_val != .object) continue;
        const props = props_val.object;
        if (props.count() == 0) continue;

        // keys ⊆ properties
        var keys_ok = true;
        var it = obj.iterator();
        while (it.next()) |kv| {
            if (props.get(kv.key_ptr.*) == null) {
                keys_ok = false;
                break;
            }
        }
        if (!keys_ok) continue;

        // required ⊆ keys
        if (params_val.object.get("required")) |req_val| {
            if (req_val == .array) {
                var req_ok = true;
                for (req_val.array.items) |r| {
                    if (r != .string) continue;
                    if (obj.get(r.string) == null) {
                        req_ok = false;
                        break;
                    }
                }
                if (!req_ok) continue;
            }
        }

        if (match_name != null) return null; // ambiguous — never guess
        match_name = name_val.string;
    }

    const name = match_name orelse return null;
    const calls = try allocator.alloc(ParsedToolCall, 1);
    errdefer allocator.free(calls);
    const name_owned = try allocator.dupe(u8, name);
    errdefer allocator.free(name_owned);
    calls[0] = .{ .name = name_owned, .arguments = try allocator.dupe(u8, obj_slice) };
    log.info("  [tool-parse] inferred bare-args call to '{s}' via unique schema match\n", .{name});
    return calls;
}

/// One UTF-8 sequence starting at `i`: `valid` = well-formed, else `end` is the maximal
/// subpart to replace with U+FFFD (a character split across two tokens costs ONE replacement).
pub const Utf8Seq = struct { end: usize, valid: bool };

pub fn utf8Next(s: []const u8, i: usize) Utf8Seq {
    const len = std.unicode.utf8ByteSequenceLength(s[i]) catch return .{ .end = i + 1, .valid = false };
    if (i + len <= s.len and std.unicode.utf8ValidateSlice(s[i .. i + len])) return .{ .end = i + len, .valid = true };
    var j = i + 1;
    while (j < s.len and j < i + len and s[j] & 0xC0 == 0x80) j += 1;
    return .{ .end = j, .valid = false };
}

/// The lossy view of bytes that are not valid UTF-8: one U+FFFD per invalid sequence.
/// A token is a BPE fragment, so ANY string built from model bytes can end mid-character;
/// raw, that makes a response body unparseable and a prompt render fall back silently.
pub fn utf8Sanitize(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var clean = std.ArrayList(u8).empty;
    errdefer clean.deinit(allocator);
    var i: usize = 0;
    while (i < input.len) {
        const seq = utf8Next(input, i);
        try clean.appendSlice(allocator, if (seq.valid) input[i..seq.end] else "\u{FFFD}");
        i = seq.end;
    }
    return clean.toOwnedSlice(allocator);
}

/// THE JSON string escaper (quoted, control bytes \u-escaped). Public so other
/// request builders reuse it instead of hand-rolling a second one — a duplicate
/// escaper is exactly how the control-byte class shipped twice before.
pub fn appendJsonString(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), s: []const u8) !void {
    try buf.append(allocator, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            // nlohmann rejects ill-formed UTF-8 like a raw control byte.
            const seq = utf8Next(s, i);
            try buf.appendSlice(allocator, if (seq.valid) s[i..seq.end] else "\u{FFFD}");
            i = seq.end;
            continue;
        }
        i += 1;
        switch (c) {
            '"' => try buf.appendSlice(allocator, "\\\""),
            '\\' => try buf.appendSlice(allocator, "\\\\"),
            '\n' => try buf.appendSlice(allocator, "\\n"),
            '\r' => try buf.appendSlice(allocator, "\\r"),
            '\t' => try buf.appendSlice(allocator, "\\t"),
            // Every other control char (e.g. ESC from ANSI codes in tool
            // results) must be \u-escaped — nlohmann inside jinja_render_chat
            // rejects raw control bytes, and the render failure silently
            // downgrades the prompt to fallbackFormatChat.
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(allocator, n);
            },
            else => try buf.append(allocator, c),
        }
    }
    try buf.append(allocator, '"');
}

// ── Tests ──

const testing = std.testing;

/// src/fixtures/mimo_v26_chat_template.jinja is the `chat_template` of
/// XiaomiMiMo/MiMo-V2.6-Flash-RL's tokenizer_config.json, verbatim.
pub fn mimoTemplateConfig(allocator: std.mem.Allocator) ChatConfig {
    return .{
        .chat_template = @embedFile("fixtures/mimo_v26_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
}

test "real mimo_v2 template is the committed fixture" {
    const raw = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    var config = try loadChatConfig(testing.io, testing.allocator, std.mem.span(raw));
    defer config.deinit();
    try testing.expectEqualStrings(@embedFile("fixtures/mimo_v26_chat_template.jinja"), config.chat_template);
}

test "mimo_v2 template renders a late system turn in place" {
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "system", .content = "late" },
        .{ .role = "user", .content = "again" },
    };
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqualStrings("<|im_start|>system\nS<|im_end|><|im_start|>user\nhi<|im_end|>" ++
        "<|im_start|>system\nlate<|im_end|><|im_start|>user\nagain<|im_end|><|im_start|>assistant\n", rendered);
}

test "mimo_v2 template renders a tools request as the reference does" {
    // Expected bytes: transformers' apply_chat_template environment (jinja2, its
    // json.dumps tojson) over the same template and request.
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const tools_json =
        \\[{"type":"function","function":{"name":"get_weather","description":"Current weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    ;
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "Weather in Paris?" },
        .{ .role = "assistant", .content = "", .reasoning_content = "Need weather.", .tool_calls = &calls },
        .{ .role = "tool", .content = "18C", .tool_call_id = "call_1" },
        .{ .role = "user", .content = "Thanks." },
    };
    const expected = "<|im_start|>system\nYou are provided with the following tools:\n\n<tools>\n" ++
        "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Current weather\", " ++
        "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}" ++
        "\n</tools><|im_end|><|im_start|>system\nS<|im_end|><|im_start|>user\nWeather in Paris?<|im_end|>" ++
        "<|im_start|>assistant\n<think>Need weather.</think><tool_call><function=get_weather><parameter=city>Paris</parameter>" ++
        "</function></tool_call><|im_end|><|im_start|>tool\n18C<|im_end|><|im_start|>user\nThanks.<|im_end|>" ++
        "<|im_start|>assistant\n";
    const thinking = try renderChatTemplate(a, &messages, &config, tools_json, null, true, null, false);
    defer a.free(thinking);
    try testing.expectEqualStrings(expected, thinking);
    const plain = try renderChatTemplate(a, &messages, &config, tools_json, null, false, null, false);
    defer a.free(plain);
    try testing.expectEqualStrings(expected ++ "<think></think>", plain);
}

test "collapseDoubledThinkTags collapses 2x → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<|Assistant|></think></think>Hi!");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<|Assistant|></think>Hi!", out);
}

test "collapseDoubledThinkTags collapses triple → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, "</think></think></think>X");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("</think>X", out);
}

test "collapseDoubledThinkTags leaves single </think> unchanged" {
    const out = try collapseDoubledThinkTags(testing.allocator, "<think>r</think>Hello");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("<think>r</think>Hello", out);
}

test "collapseDoubledThinkTags handles multiple separated doublings" {
    const out = try collapseDoubledThinkTags(testing.allocator,
        "A</think></think>B</think></think>C");
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("A</think>B</think>C", out);
}

const test_tools_write_bash =
    \\[{"type":"function","function":{"name":"Write","description":"Write a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\ {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
;

test "inferBareJsonToolCalls maps bare-args fenced JSON to the unique matching tool (Claude Code capture)" {
    const allocator = testing.allocator;
    // Shape captured live from gemma-4-12b via Claude Code /v1/messages: thought
    // block, then a ```json fence holding ONLY the Write tool's arguments.
    const text = "<|channel>thought\nI will create this file using Write.<channel|>```json\n{\n  \"file_path\": \"/Users/user/mlx_info.html\",\n  \"content\": \"<h1>MLX</h1>\"\n}\n```\nI've created the file.";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("Write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/Users/user/mlx_info.html", parsed.value.object.get("file_path").?.string);
}

test "inferBareJsonToolCalls unfenced bare object at content start" {
    const allocator = testing.allocator;
    const text = "{\"command\": \"ls -la\"}";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("Bash", calls[0].name);
}

test "inferBareJsonToolCalls refuses ambiguous and non-matching objects" {
    const allocator = testing.allocator;
    // Two tools share the same key → ambiguous → null.
    const dup_tools =
        \\[{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}},
        \\ {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}}]
    ;
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"x\": \"1\"}", dup_tools)) == null);
    // Keys not in any tool's properties → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"zzz\": 1}", test_tools_write_bash)) == null);
    // Missing a required key → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"file_path\": \"a.txt\"}", test_tools_write_bash)) == null);
}

test "inferBareJsonToolCalls ignores JSON that does not lead the content" {
    const allocator = testing.allocator;
    // Example object mid-prose must never be promoted to a call.
    const text = "Here is an example of the payload you could send: {\"command\": \"ls\"} — adjust as needed.";
    try testing.expect((try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) == null);
}

test "filterInferredBySchema drops undeclared inferred calls, keeps declared and explicit ones" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    // Inferred + undeclared → dropped; slice becomes null.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "George Washington"),
            .arguments = try allocator.dupe(u8, "{\"num\":1}"),
            .inferred = true,
        };
        try testing.expect((try filterInferredBySchema(allocator, calls, tools)) == null);
    }
    // Inferred + declared → kept.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "write"),
            .arguments = try allocator.dupe(u8, "{\"path\":\"a\"}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
    }
    // Explicit + undeclared → kept (client feedback, not our guess); the
    // inferred sibling with an undeclared name is dropped and the slice shrinks.
    {
        var calls = try allocator.alloc(ParsedToolCall, 2);
        calls[0] = .{
            .name = try allocator.dupe(u8, "searchWeb"),
            .arguments = try allocator.dupe(u8, "{\"q\":\"zig\"}"),
            .inferred = false,
        };
        calls[1] = .{
            .name = try allocator.dupe(u8, "John Adams"),
            .arguments = try allocator.dupe(u8, "{\"num\":2}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
        try testing.expectEqualStrings("searchWeb", kept[0].name);
    }
}

test "toolNameIsDeclared: wrapped + flat forms; unparseable schema never drops" {
    const allocator = testing.allocator;
    const wrapped =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{}}}}]
    ;
    const flat =
        \\[{"name":"write","parameters":{"type":"object","properties":{}}}]
    ;
    try testing.expect(toolNameIsDeclared(allocator, wrapped, "write"));
    try testing.expect(toolNameIsDeclared(allocator, flat, "write"));
    try testing.expect(!toolNameIsDeclared(allocator, wrapped, "George Washington"));
    // Broken schema JSON → safe default TRUE (never drop on OUR parse failure).
    try testing.expect(toolNameIsDeclared(allocator, "not json", "anything"));
}

test "qwen xml: the live <function=…>/<parameter=…> capture (package.json class)" {
    // Verbatim from ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve via pi (2026-08-12).
    // Qwen 3.5/3.6's OWN template dictates this dialect ("ONLY reply in the
    // following format"), so it is the checkpoint's contract, not a mangle.
    // The JSON-only reader mined the first balanced object out of the body —
    // the file CONTENT — and promoted its "name" key to the tool name, so pi
    // answered "Tool voxel-pagoda-garden not found" and the model looped.
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n" ++
        "/Users/d/pi-lfm/package.json\n</parameter>\n<parameter=content>\n" ++
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}\n" ++
        "</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("/Users/d/pi-lfm/package.json", o.get("path").?.string);
    // The value is RAW BYTES: a JSON file stays a STRING (typing by spelling
    // is what turned this call into `{"name": "voxel-pagoda-garden", …}`).
    // Types come from the schema at the request chokepoint, never from here.
    try testing.expectEqualStrings(
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}",
        o.get("content").?.string,
    );
}

test "qwen xml: truncated mid-parameter ships NAME + the completed params only" {
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.ts\n</parameter>\n" ++
        "<parameter=content>\nexport const half = ";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a.ts", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "issue #94: a container param carried as literal TEXT coerces to a real array/object" {
    // #94 reports this as an open gap on the strength of the doc comment
    // ("only SCALARS are touched"), which was stale — `coerceValueToType` has
    // had its array/object arm since 60ba5ec (2026-07-09), two weeks before
    // the issue was filed. Pinned as a TEST rather than left as a code read,
    // so the claim is settled by evidence and cannot silently regress.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"stats","parameters":{"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}},"opts":{"type":"object"},"label":{"type":"string"}}}}}]
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "stats"),
        .arguments = try allocator.dupe(u8, "{\"nums\":\"[1, 2, 3]\",\"opts\":\"{\\\"grid\\\": true}\",\"label\":\"[not an array]\"}"),
    }};
    defer {
        allocator.free(calls[0].name);
        allocator.free(calls[0].arguments);
    }

    try coerceToolArgsToSchema(allocator, &calls, tools);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    const nums = o.get("nums").?.array;
    try testing.expectEqual(@as(usize, 3), nums.items.len);
    try testing.expectEqual(@as(i64, 2), nums.items[1].integer);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    // A STRING param whose content merely looks bracket-y is untouched — the
    // schema decides, never the spelling.
    try testing.expectEqualStrings("[not an array]", o.get("label").?.string);
}

test "appendJsonString escapes special chars" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "hello \"world\"\nnew\\line");
    try testing.expectEqualStrings("\"hello \\\"world\\\"\\nnew\\\\line\"", buf.items);
}

test "appendJsonString empty string" {
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try appendJsonString(allocator, &buf, "");
    try testing.expectEqualStrings("\"\"", buf.items);
}

test "appendJsonString escapes ALL control characters (2026-06-11 ESC-byte regression)" {
    // Live failure: a tool result carrying raw ANSI terminal codes (ESC = 0x1B
    // from `\x1b[?25l`) passed through unescaped, nlohmann::json rejected the
    // messages JSON inside jinja_render_chat, and the prompt silently fell
    // back to the wrong-format fallbackFormatChat — gemma-4-31b then
    // hallucinated entire conversations. JSON strings must escape EVERY
    // control char < 0x20, not just \n \r \t.
    const allocator = testing.allocator;

    var input: [0x20]u8 = undefined;
    for (&input, 0..) |*c, i| c.* = @intCast(i);

    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, &input);

    // No raw control byte may survive in the serialized form.
    for (buf.items) |c| {
        try testing.expect(c >= 0x20);
    }

    // And the result must round-trip through a strict JSON parser byte-exact.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf.items, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(&input, parsed.value.string);
}

test "renderChatTemplate: tool result with raw ANSI escapes still renders via Jinja" {
    // Regression for the 2026-06-11 pi/gemma-4-31b session: the third request
    // was the first whose history held a tool result with a raw ESC byte (the
    // interactive `npx sv create` output). The Jinja render failed on the
    // unescaped messages JSON and silently downgraded to fallbackFormatChat's
    // text format — not special tokens for Gemma 4, so generation never
    // stopped at turn end. The render must SUCCEED via
    // the template (template marker present, fallback marker absent).
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'tool' -%}<|tool_response>{{ message['content'] }}<turn|>
        \\{%- else -%}<|turn>{{ message['role'] }}
        \\{{ message['content'] }}{% if message.tool_calls %}{% for tc in message.tool_calls %}<|tool_call>{{ tc.function.name }}<tool_call|>{% endfor %}{% endif %}<turn|>
        \\{%- endif -%}
        \\{%- endfor -%}
        \\{# tools referenced so no synthesis kicks in: {{ tools }} #}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "bash", .arguments = "{\"command\": \"npx sv create .\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "make me a sveltekit app" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        // Verbatim shape from the live failure: hide-cursor ANSI code + prompt UI.
        .{ .role = "tool", .content = "\x1b[?25l\u{2502}\n\u{25c6}  Which template would you like?", .tool_call_id = "tc_0" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, "[]", null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "<|tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "Which template") != null);
    // Fallback format must NOT have been used.
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>") == null);
}

test "templateConsumesEffort: the real templates that READ the effort word" {
    // The families whose templates act on `reasoning_effort`. For these
    // the word is the lever and an effort-derived TOKEN budget on top is pure
    // truncation (see the fn doc).
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/qwen38_flash_next_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/dsv41_chat_template.jinja")));
    try testing.expect(templateConsumesEffort(@embedFile("fixtures/glm53_chat_template.jinja")));

    // Everything else on disk (gemma 4, LFM2.5, Qwen3.5/3.6, Ling,
    // Nemotron, llama, mistral) reads neither: thinking is a bool there.
    try testing.expect(!templateConsumesEffort(
        "{%- if enable_thinking %}<think>\n{%- endif %}"));
    try testing.expect(!templateConsumesEffort(""));
}

test "renderChatTemplate: a tool with no description/parameters still renders (no silent fallback)" {
    // OpenAI's schema makes `description` optional and a no-arg tool needs no
    // `parameters`, but a template that dumps those keys through `tojson` RAISES
    // on Undefined and fell back to ChatML with no tools preamble at all. The
    // defaults are filled for ALL templates.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    // No `description`, no `parameters` — both legal over the wire.
    const bare =
        \\[{"type":"function","function":{"name":"note"}}]
    ;
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(allocator, &messages, &config, bare, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expectEqual(before, template_fallbacks.load(.monotonic));
    try testing.expect(std.mem.indexOf(u8, rendered, "# Tools") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "\"name\": \"note\"") != null);

    // A COMPLETE declaration re-emits nothing: null means the caller keeps the
    // client's own bytes, so no request that works today is re-serialized
    // (key order, number formatting and unicode escaping all stay theirs).
    const complete =
        \\[{"type":"function","function":{"name":"note","description":"d","parameters":{"type":"object","properties":{}}}}]
    ;
    try testing.expect(try fillOptionalToolDefKeys(allocator, complete) == null);
    // Only the missing keys are added, and only to the function that lacks them.
    const partial =
        \\[{"type":"function","function":{"name":"a","description":"d"}},{"type":"function","function":{"name":"b","description":"e","parameters":{"type":"object"}}}]
    ;
    const filled = (try fillOptionalToolDefKeys(allocator, partial)).?;
    defer allocator.free(filled);
    try testing.expect(std.mem.indexOf(u8, filled, "\"parameters\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"d\"") != null);
    try testing.expect(std.mem.indexOf(u8, filled, "\"description\":\"e\"") != null);
}

test "renderChatTemplate folds a late system turn only where the template refuses it" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "system", .content = "late" },
        .{ .role = "user", .content = "again" },
    };
    // A Qwen3.8 revision the adapter does not pin raises on a system turn past index 0 (the raise was
    // the generic fallback), so the turn is folded into the leading one.
    {
        const revised = try std.mem.concat(allocator, u8, &.{ @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), "{#- revision #}" });
        defer allocator.free(revised);
        var config = ChatConfig{
            .chat_template = revised,
            .bos_token = null,
            .eos_token = "<|im_end|>",
            .add_bos_token = false,
            .allocator = allocator,
        };
        const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, rendered, "<|im_start|>system"));
        try testing.expect(std.mem.indexOf(u8, rendered, "S\n\nlate") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
    }
    // A template that renders the late turn keeps it where it was sent, byte for byte.
    {
        var config = ChatConfig{
            .chat_template = "{% for message in messages %}{{ '<|im_start|>' ~ message.role ~ '\\n' ~ message.content ~ '<|im_end|>' }}{% endfor %}",
            .bos_token = null,
            .eos_token = null,
            .add_bos_token = false,
            .allocator = allocator,
        };
        const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expectEqualStrings("<|im_start|>system\nS<|im_end|><|im_start|>user\nhi<|im_end|>" ++
            "<|im_start|>system\nlate<|im_end|><|im_start|>user\nagain<|im_end|>", rendered);
    }
}

const qwen_late_templates = [_][]const u8{
    @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
};

test "stock Qwen late system notes render in place and keep tool rounds a prefix" {
    const a = testing.allocator;
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"description\":\"Read a value\",\"parameters\":{\"type\":\"object\"}}}]";
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{\"index\":3}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "runtime note one" },
        .{ .role = "assistant", .content = "checking", .reasoning_content = "current reasoning", .tool_calls = &calls },
        .{ .role = "tool", .content = "first result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "runtime note two" },
        .{ .role = "system", .content = "runtime note three" },
        .{ .role = "assistant", .content = "answer", .reasoning_content = "more reasoning" },
    };
    for (qwen_late_templates) |tpl| {
        for ([_]?bool{ null, true, false }) |preserve| {
            const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .preserve_thinking = preserve };
            const first = try renderChatTemplate(a, messages[0..3], &config, tools, null, true, "low", false);
            defer a.free(first);
            const second = try renderChatTemplate(a, messages[0..6], &config, tools, null, true, "low", false);
            defer a.free(second);
            const third = try renderChatTemplate(a, &messages, &config, tools, null, true, "low", false);
            defer a.free(third);
            try testing.expectEqualStrings(first, second[0..@min(first.len, second.len)]);
            const stable_second = second[0 .. second.len - "<|im_start|>assistant\n<think>\n".len];
            try testing.expectEqualStrings(stable_second, third[0..@min(stable_second.len, third.len)]);
            try testing.expect(std.mem.indexOf(u8, third, "<|im_start|>system\nruntime note one<|im_end|>\n<|im_start|>assistant") != null);
            try testing.expect(std.mem.indexOf(u8, third, "</tool_response><|im_end|>\n<|im_start|>system\nruntime note two") != null);
            try testing.expect(std.mem.indexOf(u8, third, "<|im_start|>system\nruntime note three<|im_end|>") != null);
            try testing.expect(std.mem.indexOf(u8, third, "current reasoning") != null);
            try testing.expect(std.mem.indexOf(u8, third, "<function=lookup>\n<parameter=index>\n3\n</parameter>") != null);
            try testing.expectEqual(@as(usize, 1), std.mem.count(u8, third, "<tools>"));
        }
    }
}

test "stock Qwen adapter: a second leading system turn stays a turn of its own" {
    const a = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "system", .content = "second lead" },
        .{ .role = "user", .content = "question" },
    };
    for (qwen_late_templates) |tpl| {
        const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
        defer a.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "stable<|im_end|>\n<|im_start|>system\nsecond lead<|im_end|>\n<|im_start|>user\nquestion") != null);
    }
}

test "stock Qwen adapter leaves prompts without late notes byte-identical, tools, thinking and media included" {
    const a = testing.allocator;
    const parts = [_]MediaPart{ .{ .at = 0, .kind = .image }, .{ .at = 0, .kind = .video } };
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{\"index\":3}" }};
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    for (qwen_late_templates, 0..) |tpl, index| {
        const adapted = (try qwenLateSystemTemplate(a, tpl)).?;
        defer a.free(adapted);
        const messages = [_]Message{
            .{ .role = "system", .content = "stable" },
            .{ .role = "user", .content = "question", .media_parts = if (index == 0) &parts else null },
            .{ .role = "assistant", .content = "checking", .reasoning_content = "thought", .tool_calls = &calls },
            .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
            .{ .role = "assistant", .content = "answer", .reasoning_content = "more thought" },
            .{ .role = "user", .content = "next question" },
        };
        for ([_]?bool{ null, true, false }) |preserve| {
            const native: ChatConfig = .{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .preserve_thinking = preserve };
            var extended = native;
            extended.chat_template = adapted;
            for ([_]?[]const u8{ null, tools }) |tool_set| {
                for ([_][]const u8{ "low", "medium", "xhigh" }) |effort| {
                    for ([_][]const Message{ &messages, messages[1..] }) |history| {
                        const want = try renderChatTemplate(a, history, &native, tool_set, null, true, effort, false);
                        defer a.free(want);
                        const got = try renderChatTemplate(a, history, &extended, tool_set, null, true, effort, false);
                        defer a.free(got);
                        try testing.expectEqualStrings(want, got);
                    }
                }
            }
        }
    }
}

test "stock Qwen adapter: a late note between media turns renders in place" {
    const a = testing.allocator;
    const tpl = qwen_late_templates[0];
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    const parts = [_]MediaPart{.{ .at = 0, .kind = .image }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question", .media_parts = &parts },
        .{ .role = "assistant", .content = "answer", .reasoning_content = "thought" },
        .{ .role = "tool", .content = "result one" },
        .{ .role = "system", .content = "late note" },
        .{ .role = "tool", .content = "result two", .media_parts = &parts },
    };
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
    defer a.free(rendered);
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, rendered, "<|vision_start|><|image_pad|><|vision_end|>"));
    try testing.expect(std.mem.indexOf(u8, rendered, "</tool_response><|im_end|>\n<|im_start|>system\nlate note<|im_end|>\n<|im_start|>user\n<tool_response>") != null);
}

test "stock Qwen adapter engages only on the byte-pinned revisions" {
    const a = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    for (qwen_late_templates) |tpl| {
        const revised = try std.mem.concat(a, u8, &.{ tpl, "{{- 'revision marker' }}" });
        defer a.free(revised);
        const respelled = try std.mem.replaceOwned(u8, a, tpl, "message.role == \"system\"", "message.role == 'system'");
        defer a.free(respelled);
        for ([_][]const u8{ revised, respelled }) |unknown| {
            try testing.expect((try qwenLateSystemTemplate(a, unknown)) == null);
            const config = ChatConfig{ .chat_template = unknown, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
            const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
            defer a.free(rendered);
            try testing.expect(std.mem.indexOf(u8, rendered, "stable\n\nlate note<|im_end|>\n<|im_start|>user\nquestion") != null);
        }
    }
}

test "real Qwen pack loaded through ChatConfig keeps late notes in place" {
    const dir = std.c.getenv("QWEN_MID_SYSTEM_MODEL_DIR") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try loadChatConfig(io, a, std.mem.span(dir));
    defer config.deinit();
    const adapted = try qwenLateSystemTemplate(a, config.chat_template);
    defer if (adapted) |t| a.free(t);
    try testing.expect(adapted != null);
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "stable instructions" },
        .{ .role = "user", .content = "first question" },
        .{ .role = "system", .content = "runtime note one" },
        .{ .role = "assistant", .content = "checking", .reasoning_content = "thought", .tool_calls = &calls },
        .{ .role = "tool", .content = "result", .tool_call_id = "call_1" },
        .{ .role = "system", .content = "runtime note two" },
    };
    const first = try renderChatTemplate(a, messages[0..3], &config, tools, null, true, "low", false);
    defer a.free(first);
    const next = try renderChatTemplate(a, &messages, &config, tools, null, true, "low", false);
    defer a.free(next);
    try testing.expect(std.mem.indexOf(u8, first, "first question<|im_end|>\n<|im_start|>system\nruntime note one<|im_end|>") != null);
    try testing.expect(std.mem.indexOf(u8, next, "</tool_response><|im_end|>\n<|im_start|>system\nruntime note two<|im_end|>") != null);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, next, "<tools>"));
}

const forced_choice_tools =
    \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
;

test "renderChatTemplate: a forced tool_choice reaches templates that render tools themselves" {
    const allocator = testing.allocator;
    const templates = [_][]const u8{
        @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
        @embedFile("fixtures/mimo_v26_chat_template.jinja"),
    };
    const instr = "\nYou MUST call the function \"get_time\". Do not respond with text.";
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const user_last = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "What time is it?" },
    };
    const tool_last = [_]Message{
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };
    const conversations = [_][]const Message{ &user_last, &tool_last };
    for (templates) |tpl| {
        var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
        for (conversations) |msgs| {
            for ([_]bool{ true, false }) |thinking| {
                const plain = try renderChatTemplate(allocator, msgs, &config, forced_choice_tools, null, thinking, null, false);
                defer allocator.free(plain);
                const forced = try renderChatTemplate(allocator, msgs, &config, forced_choice_tools, instr, thinking, null, false);
                defer allocator.free(forced);
                try testing.expect(std.mem.indexOf(u8, plain, "You MUST call") == null);
                try testing.expectEqual(@as(usize, 1), std.mem.count(u8, forced, "You MUST call the function \"get_time\". Do not respond with text."));
                // Only the tail moves: every byte before the last message stays, so a
                // prefix cached from an auto turn still serves a forced one.
                const last = msgs[msgs.len - 1].content;
                const cut = std.mem.lastIndexOf(u8, plain, last).?;
                try testing.expect(std.mem.startsWith(u8, forced, plain[0..cut]));
                try testing.expect(std.mem.lastIndexOf(u8, forced, "You MUST call").? > std.mem.lastIndexOf(u8, forced, last).?);
                const generation_prompt = plain[std.mem.lastIndexOf(u8, plain, "<|im_start|>assistant").?..];
                try testing.expect(std.mem.endsWith(u8, forced, generation_prompt));
            }
        }
    }
}

test "forcedCallText reads each family's call markup from its own template" {
    const allocator = testing.allocator;
    const Case = struct { tpl: []const u8, sep: []const u8, required: []const u8, named: []const u8, name_tail: []const u8 };
    const cases = [_]Case{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .sep = "\n\n", .required = "<tool_call>\n<function=", .named = "<tool_call>\n<function=get_time>\n", .name_tail = ">\n" },
        .{ .tpl = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .sep = "", .required = "<tool_call><function=", .named = "<tool_call><function=get_time>", .name_tail = ">" },
    };
    for (cases) |c| {
        var config = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
        var any = (try forcedCallText(allocator, &config, null)).?;
        defer any.deinit(allocator);
        try testing.expectEqualStrings(c.sep, any.sep);
        try testing.expectEqualStrings(c.required, any.call);
        try testing.expectEqualStrings(c.name_tail, any.name_tail);
        var one = (try forcedCallText(allocator, &config, "get_time")).?;
        defer one.deinit(allocator);
        try testing.expectEqualStrings(c.sep, one.sep);
        try testing.expectEqualStrings(c.named, one.call);
    }
    // No think block to anchor on: nothing is forced.
    var plain = ChatConfig{ .chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    try testing.expect(try forcedCallText(allocator, &plain, null) == null);
}

test "renderChatTemplate: a continuation keeps the client's own prefill, not a forced tool_choice" {
    const allocator = testing.allocator;
    var config = ChatConfig{ .chat_template = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
    const msgs = [_]Message{
        .{ .role = "user", .content = "What time is it?" },
        .{ .role = "assistant", .content = "It is" },
    };
    const rendered = try renderChatTemplate(allocator, &msgs, &config, forced_choice_tools, "\nYou MUST call one of the available functions. Do not respond with text.", true, null, true);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "You MUST call") == null);
    try testing.expect(std.mem.endsWith(u8, rendered, "It is"));
}

test "renderChatTemplate: preserve_thinking keeps or drops Qwen3.8's prior-turn thinking" {
    const allocator = testing.allocator;
    const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "q1" },
        .{ .role = "assistant", .content = "a1", .reasoning_content = "PRIOR_THOUGHT" },
        .{ .role = "user", .content = "q2" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc, .reasoning_content = "CURRENT_THOUGHT" },
        .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
    };
    const templates = [_][]const u8{
        @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"),
    };
    for (templates) |tpl| {
        var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
        const absent = try renderChatTemplate(allocator, &messages, &config, forced_choice_tools, null, true, null, false);
        defer allocator.free(absent);
        try testing.expect(std.mem.indexOf(u8, absent, "<|im_start|>assistant\n<think>\nPRIOR_THOUGHT\n</think>\n\na1<|im_end|>") != null);
        try testing.expect(std.mem.indexOf(u8, absent, "<think>\nCURRENT_THOUGHT\n</think>") != null);

        config.preserve_thinking = true;
        const on = try renderChatTemplate(allocator, &messages, &config, forced_choice_tools, null, true, null, false);
        defer allocator.free(on);
        try testing.expectEqualStrings(absent, on);

        // Off renders only the latest user turn's thinking.
        config.preserve_thinking = false;
        const off = try renderChatTemplate(allocator, &messages, &config, forced_choice_tools, null, true, null, false);
        defer allocator.free(off);
        try testing.expect(std.mem.indexOf(u8, off, "PRIOR_THOUGHT") == null);
        try testing.expect(std.mem.indexOf(u8, off, "<|im_start|>assistant\na1<|im_end|>") != null);
        try testing.expect(std.mem.indexOf(u8, off, "<think>\nCURRENT_THOUGHT\n</think>") != null);
    }
}

test "renderChatTemplate: a template that does not read preserve_thinking renders the same bytes" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "q1" },
        .{ .role = "assistant", .content = "a1", .reasoning_content = "PRIOR_THOUGHT" },
        .{ .role = "user", .content = "q2" },
    };
    const templates = [_][]const u8{
        @embedFile("fixtures/mimo_v26_chat_template.jinja"),
        "{% for message in messages %}{{ '<|im_start|>' ~ message.role ~ '\\n' ~ message.content ~ '<|im_end|>' }}{% endfor %}",
    };
    for (templates) |tpl| {
        var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
        const absent = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
        defer allocator.free(absent);
        for ([_]bool{ true, false }) |p| {
            config.preserve_thinking = p;
            const got = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
            defer allocator.free(got);
            try testing.expectEqualStrings(absent, got);
        }
    }
}

test "serializeExtraContext with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, true, null);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "serializeExtraContext with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try serializeExtraContext(allocator, &config, false, null);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
}

test "serializeExtraContext: DeepSeek-V4.1 maps the client's effort onto low|high|max" {
    // The family is sniffed by `thinking_mode`, which only this template
    // reads; `thinking_mode` still carries the on/off switch.
    const allocator = testing.allocator;
    var dsv41 = ChatConfig{
        .chat_template = "{%- set mode = thinking_mode|default('chat') -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    for ([_]bool{ true, false }) |thinking| {
        const r = try serializeExtraContext(allocator, &dsv41, thinking, null);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, "\"reasoning_effort\":\"high\"") != null);
        try testing.expect(std.mem.indexOf(u8, r, if (thinking)
            "\"thinking_mode\":\"thinking\""
        else
            "\"thinking_mode\":\"chat\"") != null);
    }
    const mapped_cases = [_]struct { in: ?[]const u8, out: []const u8 }{
        .{ .in = "high", .out = "\"reasoning_effort\":\"high\"" },
        .{ .in = "max", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "xhigh", .out = "\"reasoning_effort\":\"max\"" },
        .{ .in = "low", .out = "\"reasoning_effort\":\"low\"" },
    };
    for (mapped_cases) |case| {
        const r = try serializeExtraContext(allocator, &dsv41, true, case.in);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, case.out) != null);
    }

    // Other families are untouched: high/no_think.
    var other = ChatConfig{
        .chat_template = "{{ reasoning_effort }}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const on = try serializeExtraContext(allocator, &other, true, null);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "\"reasoning_effort\":\"high\"") != null);
    const off = try serializeExtraContext(allocator, &other, false, null);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"reasoning_effort\":\"no_think\"") != null);
}

test "serializeMessagesJson basic" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Parse it back to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
    try testing.expectEqualStrings("user", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("Hello", parsed.value.array.items[0].object.get("content").?.string);
}

test "serializeMessagesJson with tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"Tokyo\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should contain tool_calls array
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "get_weather") != null);
}

test "serializeMessagesJson carries assistant reasoning_content" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const asst = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("two plus two is four", asst.get("reasoning_content").?.string);
    // A message without the field must not carry the key at all — templates
    // gate on `is string`, and an explicit null would still pass `is defined`.
    try testing.expect(parsed.value.array.items[0].object.get("reasoning_content") == null);
}

test "assistant history reasoning_content reaches a template that reads it (round-trip)" {
    // A template that persists reasoning across turns reads history
    // assistant messages render <think>{message.reasoning|reasoning_content}</think>.
    // When the parse layer drops the field, every history turn renders the
    // empty <think></think> nothink signature and the model stops thinking
    // from turn 2 of a session (live 2026-07-29, pi agent: one
    // reasoning delta on the 2-msg turn, zero across the next 13k chunks).
    // Template fragment mirrors an assistant-history branch that reads it.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "{%- for message in messages -%}" ++
            "{%- if message.role == 'assistant' -%}" ++
            "{%- set rc = '' -%}" ++
            "{%- if message.reasoning is string -%}{%- set rc = message.reasoning -%}" ++
            "{%- elif message.reasoning_content is string -%}{%- set rc = message.reasoning_content -%}" ++
            "{%- endif -%}" ++
            "<think>{{ rc }}</think>{{ message.content }}" ++
            "{%- endif -%}{%- endfor -%}",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "<think>two plus two is four</think>4") != null);
}

test "serializeMessagesJson embeds valid-JSON arguments as object (not string)" {
    // Required by Qwen 3.5/3.6 templates that do `tool_call.arguments|items`.
    // Without this, the Jinja `items` filter fails on a string and the server
    // falls back to ChatML — losing the `<think>\n` injection that primes the
    // model's reasoning + close-tag behavior on the next turn.
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Object form: ..."arguments":{"command":"date"}...
    // String form (rejected): ..."arguments":"{\"command\":\"date\"}"...
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{\"command\":\"date\"}") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"{") == null);
}

test "serializeMessagesJson embeds an empty object for malformed arguments" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "not valid json" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{}") != null);
}

test "serializeMessagesJson tool response has tool_call_id and content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Must contain tool_call_id
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"call_1\"") != null);
    // Must contain the response content
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Must NOT contain tool_responses (templates handle role:"tool" natively)
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "serializeMessagesJson full tool calling conversation" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_99_0", .name = "shell", .arguments = "{\"command\":\"echo hello\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "Run echo hello" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "hello", .tool_call_id = "call_99_0" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 4), parsed.value.array.items.len);

    // Verify roles
    try testing.expectEqualStrings("system", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("user", parsed.value.array.items[1].object.get("role").?.string);
    try testing.expectEqualStrings("assistant", parsed.value.array.items[2].object.get("role").?.string);
    try testing.expectEqualStrings("tool", parsed.value.array.items[3].object.get("role").?.string);

    // Verify assistant has tool_calls
    const assistant = parsed.value.array.items[2].object;
    try testing.expect(assistant.get("tool_calls") != null);

    // Verify tool message has tool_call_id and content, no tool_responses
    const tool_msg = parsed.value.array.items[3].object;
    try testing.expectEqualStrings("call_99_0", tool_msg.get("tool_call_id").?.string);
    try testing.expectEqualStrings("hello", tool_msg.get("content").?.string);
    try testing.expect(tool_msg.get("tool_responses") == null);
}

test "serializeMessagesJson multiple parallel tool responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "readFile", .arguments = "{\"path\":\"a.txt\"}" },
        .{ .id = "call_2", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file content", .tool_call_id = "call_1" },
        .{ .role = "tool", .content = "Fri Apr 4", .tool_call_id = "call_2" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();

    // Both tool messages should have tool_call_id and content
    const tool1 = parsed.value.array.items[1].object;
    try testing.expectEqualStrings("call_1", tool1.get("tool_call_id").?.string);
    try testing.expectEqualStrings("file content", tool1.get("content").?.string);

    const tool2 = parsed.value.array.items[2].object;
    try testing.expectEqualStrings("call_2", tool2.get("tool_call_id").?.string);
}

test "serializeMessagesJson tool response with empty content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"mkdir test\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);

    // Should have tool_call_id but no tool_responses
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "serializeMessagesJson renders each media part at its offset as a typed part" {
    const allocator = testing.allocator;
    const parts = [_]MediaPart{
        .{ .at = 0, .kind = .image },
        .{ .at = 5, .kind = .video },
        .{ .at = 11, .kind = .image },
    };
    const messages = [_]Message{
        .{ .role = "tool", .content = "shot:\nafter", .tool_call_id = "c1", .media_parts = &parts },
        .{ .role = "user", .content = "plain" },
    };
    const result = try serializeMessagesJson(allocator, &messages);
    defer allocator.free(result);
    try testing.expectEqualStrings(
        \\[{"role":"tool","content":[{"type":"image"},{"type":"text","text":"shot:"},{"type":"video"},{"type":"text","text":"\nafter"},{"type":"image"}],"tool_call_id":"c1"},{"role":"user","content":"plain"}]
    , result);
}

// ── Fallback formatter tests ──

test "fallbackFormatChat ChatML with tool calls and responses" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const result = try fallbackFormatChat(allocator, &messages, &config, tools_json, null, false);
    defer allocator.free(result);

    // Should have <tool_call> block for assistant
    try testing.expect(std.mem.indexOf(u8, result, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
    // Should have <tool_response> block for tool result
    try testing.expect(std.mem.indexOf(u8, result, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Should end with assistant prompt
    try testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n"));
}

test "fallbackFormatChat ChatML tool response uses user role" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "tool", .content = "42" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Tool responses in ChatML use <|im_start|>user role
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\n<tool_response>") != null);
    // Should NOT have <|im_start|>tool (invalid role)
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>tool") == null);
}

test "fallbackFormatChat ChatML assistant with empty content and tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"date\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Empty content should not produce a blank line before <tool_call>
    try testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant\n<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"name\": \"shell\"") != null);
}

test "fallbackFormatChat multi-round tool calling" {
    const allocator = testing.allocator;
    const tc1 = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const tc2 = [_]ToolCall{
        .{ .id = "call_2", .name = "readFile", .arguments = "{\"path\":\"main.py\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "Read main.py" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "main.py" },
        .{ .role = "assistant", .content = "Found it.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "print('hello')" },
    };
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = try allocator.dupe(u8, "<|im_end|>"),
        .add_bos_token = false,
        .allocator = allocator,
    };
    defer if (config.eos_token) |t| allocator.free(t);

    const result = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer allocator.free(result);

    // Should have two <tool_call> blocks
    var count: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_call>")) |offset| {
        count += 1;
        pos += offset + 11;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Should have two <tool_response> blocks
    count = 0;
    pos = 0;
    while (std.mem.indexOf(u8, result[pos..], "<tool_response>")) |offset| {
        count += 1;
        pos += offset + 15;
    }
    try testing.expectEqual(@as(usize, 2), count);

    // Second assistant should have content before tool_call
    try testing.expect(std.mem.indexOf(u8, result, "Found it.\n<tool_call>") != null);
}

// ── Tool fallback (DSV4 + any template lacking tools support) ──

test "templateReferencesToolRole detects role branches" {
    try testing.expect(templateReferencesToolRole("{% if message['role'] == 'tool' %}"));
    try testing.expect(templateReferencesToolRole("{% if message.role == \"tool\" %}"));
    try testing.expect(!templateReferencesToolRole("{% if message['role'] == 'user' %}{{ message['tool_calls'] }}{% endif %}"));
    try testing.expect(!templateReferencesToolRole(""));
}

test "messagesHaveToolContent" {
    const tc = [_]ToolCall{.{ .id = "x", .name = "n", .arguments = "{}" }};
    const m_tool = [_]Message{.{ .role = "tool", .content = "x" }};
    const m_tc = [_]Message{.{ .role = "assistant", .content = "", .tool_calls = &tc }};
    const m_plain = [_]Message{.{ .role = "user", .content = "hi" }};
    try testing.expect(messagesHaveToolContent(&m_tool));
    try testing.expect(messagesHaveToolContent(&m_tc));
    try testing.expect(!messagesHaveToolContent(&m_plain));
}

test "synthesizeToolFallbackMessages: inject tool prompt with no existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // System content carries the tool prompt + the tools_json blob.
    try testing.expect(std.mem.indexOf(u8, out[0].content, "tool_call") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("List files", out[1].content);
}

test "synthesizeToolFallbackMessages: merge tool prompt into existing system" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "List files" },
    };
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";

    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    // Existing system content preserved, tool prompt appended.
    try testing.expect(std.mem.startsWith(u8, out[0].content, "You are helpful."));
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "shell") != null);
}

test "synthesizeToolFallbackMessages: rewrite role:tool to role:user" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "do it" },
        .{ .role = "tool", .content = "result: 42" },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, true, false);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "result: 42") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_response>") != null);
}

test "synthesizeToolFallbackMessages: rewrite assistant tool_calls into <tool_call> blocks" {
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "c1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "ls" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
    };
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, false, false, true);
    try testing.expectEqual(@as(usize, 2), out.len);
    try testing.expectEqualStrings("assistant", out[1].role);
    try testing.expect(out[1].tool_calls == null); // moved into content
    try testing.expect(std.mem.indexOf(u8, out[1].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "\"name\": \"shell\"") != null);
    try testing.expect(std.mem.indexOf(u8, out[1].content, "</tool_call>") != null);
}

test "renderChatTemplate: DSV4-style template gets tool fallback applied" {
    const allocator = testing.allocator;
    // Minimal template that mirrors DSV4 — has system/user/assistant, no tools, no role:tool.
    const tpl =
        \\<bos>
        \\{%- for message in messages -%}
        \\{%- if message['role'] == 'system' -%}{{ message['content'] }}{%- elif message['role'] == 'user' -%}<U>{{ message['content'] }}</U><A>{%- elif message['role'] == 'assistant' -%}{{ message['content'] }}<EOS>{%- endif -%}
        \\{%- endfor -%}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\",\"description\":\"run a shell command\"}}]";
    const messages = [_]Message{
        .{ .role = "user", .content = "list my files" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
    defer allocator.free(rendered);
    // Tool prompt should have been injected as a system message.
    try testing.expect(std.mem.indexOf(u8, rendered, "Available functions") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "shell") != null);
    // User message still rendered after the synthesized system content.
    try testing.expect(std.mem.indexOf(u8, rendered, "list my files") != null);
}

test "DSV4 multi-turn agent history: every role correctly rewritten" {
    // Realistic 6-message agent transcript:
    //   1. user (initial question)
    //   2. assistant tool_call #1 (date)
    //   3. tool result #1
    //   4. assistant intermediate reasoning + tool_call #2 (ls)
    //   5. tool result #2
    //   6. user follow-up
    // The synthesizer must:
    //   - inject the tool-prompt system message at the top
    //   - rewrite each role:tool into role:user with <tool_response>
    //   - rewrite each role:assistant with tool_calls into <tool_call> blocks
    //   - preserve user messages untouched
    //   - preserve assistant intermediate text (msg #4 has BOTH content and tool_calls)
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc1 = [_]ToolCall{.{ .id = "t1", .name = "shell", .arguments = "{\"command\":\"date\"}" }};
    const tc2 = [_]ToolCall{.{ .id = "t2", .name = "shell", .arguments = "{\"command\":\"ls\"}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "what files exist as of today?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc1 },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "t1" },
        .{ .role = "assistant", .content = "Got the date. Now listing files.", .tool_calls = &tc2 },
        .{ .role = "tool", .content = "README.md\nsrc/\ntests/", .tool_call_id = "t2" },
        .{ .role = "user", .content = "summarize" },
    };

    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Expected layout: system + 6 rewritten messages.
    try testing.expectEqual(@as(usize, 7), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expect(std.mem.indexOf(u8, out[0].content, "Available functions") != null);

    // Original index 0 (user) → out[1] user, unchanged.
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("what files exist as of today?", out[1].content);

    // Original index 1 (assistant tool_call) → out[2] assistant with <tool_call> in content, no tool_calls.
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expect(out[2].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[2].content, "\"command\":\"date\"") != null);

    // Original index 2 (tool) → out[3] user with <tool_response>.
    try testing.expectEqualStrings("user", out[3].role);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);

    // Original index 3 (assistant content + tool_call) → out[4] preserves intermediate text.
    try testing.expectEqualStrings("assistant", out[4].role);
    try testing.expect(out[4].tool_calls == null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "Got the date. Now listing files.") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "<tool_call>") != null);
    try testing.expect(std.mem.indexOf(u8, out[4].content, "\"command\":\"ls\"") != null);

    // Original index 4 (tool) → out[5] user with <tool_response>.
    try testing.expectEqualStrings("user", out[5].role);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[5].content, "README.md") != null);

    // Original index 5 (user follow-up) → out[6] user, unchanged.
    try testing.expectEqualStrings("user", out[6].role);
    try testing.expectEqualStrings("summarize", out[6].content);
}

test "DSV4 fallback: null tools_json with inject_tool_prompt does not synthesize" {
    // When a request has NO tools (chat without tools param), the synthesizer
    // must NOT inject a system message even with inject_tool_prompt=true.
    // Only role:tool / assistant tool_calls rewrites still fire (in case the
    // template still lacks support and history was injected by a previous turn).
    const allocator = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const messages = [_]Message{
        .{ .role = "user", .content = "hello" },
    };
    // tools_json = null → no synthesis.
    const out = try synthesizeToolFallbackMessages(a, &messages, null, null, true, true, true);
    try testing.expectEqual(@as(usize, 1), out.len);
    try testing.expectEqualStrings("user", out[0].role);
    try testing.expectEqualStrings("hello", out[0].content);
}

// ── Schema-aware argument coercion (value-spelling type-inference class) ──

/// The Edit tool as Claude Code declares it (OpenAI shape, post
/// `buildOpenAIToolsJson`): `replace_all` is a BOOLEAN, the rest are strings.
const edit_tools_json_test =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

fn parseArgsObj(allocator: std.mem.Allocator, args: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, args, .{});
}

test "coerceToolArgsToSchema: Python-style False on a boolean param becomes JSON false" {
    // VERBATIM capture, 2026-07-09, Qwen3.6-35B-A3B via Claude Code on
    // ~/.sushi/logs/sushi-11234.log:109471. The model emits the Hermes
    // XML parameter form with Python's `False`; parseHermesToolCall's
    // isJsonLiteral only knows lowercase, so it shipped the STRING "False" and
    // Claude Code rejected every Edit with
    //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
    // — six times in a row, until the model gave up and rewrote the whole file.
    const allocator = testing.allocator;
    const raw =
        "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/Users/user/doom/index.html\n</parameter>\n" ++
        "<parameter=old_string>\n  <script src=\"game.js\"></script>\n</parameter>\n" ++
        "<parameter=new_string>\n  <script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(false, v.bool);
    // Same capture, the whitespace half of the class: `old_string` must match
    // the file "exactly, including indentation", so the two leading spaces the
    // model sent are payload. Trimming them either failed the edit or matched a
    // DIFFERENT, un-indented occurrence mid-line and rewrote it at the wrong
    // nesting; coercion leaves a declared string alone, so the parser is the
    // only layer that can preserve them.
    try testing.expectEqualStrings(
        "  <script src=\"game.js\"></script>",
        parsed.value.object.get("old_string").?.string,
    );
    try testing.expectEqualStrings(
        "  <script src=\"game.js\" type=\"module\"></script>",
        parsed.value.object.get("new_string").?.string,
    );
}

test "coerceToolArgsToSchema: unknown tool / unknown key / undecidable value pass through" {
    const allocator = testing.allocator;
    // Unknown tool name → args untouched.
    {
        const raw = "<tool_call>{\"name\":\"Nope\",\"arguments\":{\"replace_all\":\"False\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("replace_all").? == .string);
    }
    // Known tool, boolean param, value that is not a recognizable boolean →
    // left alone so the client's validation error stays honest.
    {
        const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"replace_all\":\"maybe\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const v = parsed.value.object.get("replace_all").?;
        try testing.expect(v == .string);
        try testing.expectEqualStrings("maybe", v.string);
    }
}

/// pi's `edit` tool, verbatim from its own schema
/// (@earendil-works/pi-coding-agent dist/core/tools/edit.js — typebox
/// `editSchema`). Two facts the tests below lean on: `path` is REQUIRED at the
/// TOP level, and the array's item schema declares ONLY oldText/newText — it
/// never declares `path`. That asymmetry is what makes a buried `path`
/// provably misplaced rather than a judgment call.
const pi_edit_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

const edit_array_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object"}},"dry_run":{"type":"boolean"}},"required":["path","edits"]}}}]
;

test "coerceToolArgsToSchema: an ARRAY param passed through Hermes XML is not left a string" {
    // Live capture class (15 occurrences in one session): `<parameter=edits>` holds
    // a JSON array, but parseHermesToolCall's isJsonLiteral only recognizes
    // scalars, so the whole array shipped as a STRING. Strict clients reject it
    // ("edits: want array, got str"); lenient ones silently mis-execute.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("const a = 2;", edits.array.items[0].object.get("newText").?.string);
    // sibling scalars still coerce/pass through
    try testing.expectEqualStrings("/tmp/a.js", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: identical duplicate keys inside a container param still coerce; conflicting ones do not" {
    // Issue #402: Qwen Code's ask_user_question array arrived as a string whose
    // items repeated `header` with the same value; strict parse rejects the
    // duplicate, the tolerant re-emit keeps it, and the string reached the
    // client unchanged. Identical duplicates carry no ambiguity; conflicting
    // ones do, and guessing a value is worse than leaving the string alone.
    const allocator = testing.allocator;
    const cases = [_]struct { inner: []const u8, want_array: bool }{
        .{ .inner = "[{\"header\":\"Scope\",\"question\":\"Which scope?\",\"header\":\"Scope\",\"options\":[{\"label\":\"Small\"}]}]", .want_array = true },
        .{ .inner = "[{\"header\":\"Scope\",\"header\":\"Size\"}]", .want_array = false },
    };
    for (cases) |c| {
        const raw = try std.fmt.allocPrint(allocator, "<tool_call>\n<function=edit>\n<parameter=edits>\n{s}\n</parameter>\n</function>\n</tool_call>", .{c.inner});
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const edits = parsed.value.object.get("edits").?;
        if (c.want_array) {
            try testing.expect(edits == .array);
            try testing.expectEqualStrings("Scope", edits.array.items[0].object.get("header").?.string);
            try testing.expectEqual(@as(usize, 3), edits.array.items[0].object.count());
        } else {
            try testing.expect(edits == .string);
        }
    }
}

test "coerceToolArgsToSchema: a STRING param whose content is JSON stays a string" {
    // The inverse guard for arrays/objects: `old_string` may legitimately BE the
    // text `[1,2]`. Only the schema decides.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\n[1,2]\n</parameter>\n<parameter=new_string>\n{\"k\":1}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("old_string").? == .string);
    try testing.expectEqualStrings("[1,2]", parsed.value.object.get("old_string").?.string);
    try testing.expect(parsed.value.object.get("new_string").? == .string);
}

test "coerceToolArgsToSchema: already-conforming args are byte-identical (no-op guard)" {
    // The auto-correct layer must never touch a good call. Property: when every
    // declared type already matches, `arguments` comes out byte-for-byte the same.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
        "\"new_string\":\"b\",\"replace_all\":true}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

// ── Auto-correct safety properties (deterministic fuzz) ─────────────────────
// The repair layer is only ever allowed to touch input that is actually broken.
// These generate well-formed tool calls whose values deliberately SPELL other
// JSON types ("false", "42", "[1,2]", "null") — exactly the strings the
// value-spelling inference used to mis-type — and assert round-trip identity.

fn jsonValueEql(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => a.bool == b.bool,
        .integer => a.integer == b.integer,
        .float => a.float == b.float,
        .number_string => std.mem.eql(u8, a.number_string, b.number_string),
        .string => std.mem.eql(u8, a.string, b.string),
        .array => blk: {
            if (a.array.items.len != b.array.items.len) break :blk false;
            for (a.array.items, b.array.items) |x, y| if (!jsonValueEql(x, y)) break :blk false;
            break :blk true;
        },
        .object => blk: {
            if (a.object.count() != b.object.count()) break :blk false;
            var it = a.object.iterator();
            while (it.next()) |e| {
                const other = b.object.get(e.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(e.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

/// Value strings chosen to collide with every JSON literal spelling the
/// tag-format parsers used to guess from.
const adversarial_strings = [_][]const u8{
    "false",      "true",         "False",   "True",    "null",   "None",
    "42",         "-7",           "3.14",    "0",       "1",      "",
    "[1,2]",      "{\"k\":1}",    "nan",     "inf",     "  ",     "yes",
    "a\"b",       "line\nline",   "tab\there", "back\\slash", "café ☕", "0x1f",
    "{not json",  "[unclosed",    "1e400",   "00",      "+5",     "-",
};

test "fuzz: a conforming tool call round-trips byte-identical through parse+coerce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rand = prng.random();

    const types = [_][]const u8{ "string", "integer", "number", "boolean", "array", "object" };

    var iter: usize = 0;
    while (iter < 400) : (iter += 1) {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build a random schema + a CONFORMING args object for it.
        var props: std.json.ObjectMap = .empty;
        var args: std.json.ObjectMap = .empty;
        const nprops = 1 + rand.uintLessThan(usize, 5);
        var p: usize = 0;
        while (p < nprops) : (p += 1) {
            const key = try std.fmt.allocPrint(arena, "p{d}", .{p});
            const want = types[rand.uintLessThan(usize, types.len)];

            var prop: std.json.ObjectMap = .empty;
            try prop.put(arena, "type", .{ .string = want });
            try props.put(arena, key, .{ .object = prop });

            const v: std.json.Value = if (std.mem.eql(u8, want, "string"))
                .{ .string = adversarial_strings[rand.uintLessThan(usize, adversarial_strings.len)] }
            else if (std.mem.eql(u8, want, "integer"))
                .{ .integer = @as(i64, @intCast(rand.uintLessThan(u32, 1000))) - 500 }
            else if (std.mem.eql(u8, want, "number"))
                .{ .float = 1.5 }
            else if (std.mem.eql(u8, want, "boolean"))
                .{ .bool = rand.boolean() }
            else if (std.mem.eql(u8, want, "array")) blk: {
                var arr = std.json.Array.init(arena);
                try arr.append(.{ .string = "x" });
                try arr.append(.{ .integer = 1 });
                break :blk .{ .array = arr };
            } else blk: {
                var o: std.json.ObjectMap = .empty;
                try o.put(arena, "k", .{ .string = "v" });
                break :blk .{ .object = o };
            };
            try args.put(arena, key, v);
        }

        var func: std.json.ObjectMap = .empty;
        try func.put(arena, "name", .{ .string = "t" });
        var params: std.json.ObjectMap = .empty;
        try params.put(arena, "type", .{ .string = "object" });
        try params.put(arena, "properties", .{ .object = props });
        try func.put(arena, "parameters", .{ .object = params });
        var tool: std.json.ObjectMap = .empty;
        try tool.put(arena, "type", .{ .string = "function" });
        try tool.put(arena, "function", .{ .object = func });
        var tools_arr = std.json.Array.init(arena);
        try tools_arr.append(.{ .object = tool });
        const tools_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .array = tools_arr }, .{});

        // Serialize as a canonical Hermes JSON call — always VALID input.
        const args_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = args }, .{});
        const raw = try std.fmt.allocPrint(arena,
            "<tool_call>{{\"name\":\"t\",\"arguments\":{s}}}</tool_call>", .{args_json});

        const calls = (try parseToolCalls(allocator, raw)) orelse {
            std.debug.print("\n[fuzz iter {d}] valid tool call did not parse\n  {s}\n", .{ iter, raw });
            return error.FuzzFailed;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }

        const before = try arena.dupe(u8, calls[0].arguments);
        try coerceToolArgsToSchema(allocator, calls, tools_json);

        // Property 1: a conforming call is left BYTE-identical.
        if (!std.mem.eql(u8, before, calls[0].arguments)) {
            std.debug.print("\n[fuzz iter {d}] auto-correct mutated a conforming call\n  before: {s}\n  after : {s}\n", .{ iter, before, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 2: the values survive verbatim (no spelling-based retyping).
        var got = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer got.deinit();
        if (!jsonValueEql(std.json.Value{ .object = args }, got.value)) {
            std.debug.print("\n[fuzz iter {d}] value mismatch\n  want: {s}\n  got : {s}\n", .{ iter, args_json, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 3: conformance holds, and coercion is idempotent.
        try testing.expect(toolCallConformsToSchema(allocator, calls[0], tools_json));
        try coerceToolArgsToSchema(allocator, calls, tools_json);
        try testing.expectEqualStrings(before, calls[0].arguments);
    }
}

test "coerceToolArgsToSchema: a MANGLED array-typed param is tolerantly repaired" {
    // Live soak capture (record 268): pi's `edits` array with a MISSING COMMA
    // between two object values — `"newText": "..." "path": "..."`. Strict parse
    // rejects it, so pre-fix `edits` stayed a STRING (schema contradiction). The
    // tolerant container repair recovers a real array.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n./game.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"  speed: 8,\", \"newText\": \"  vy: 0,\" \"extra\": \"x\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("  speed: 8,", edits.array.items[0].object.get("oldText").?.string);
}

// ── Misplaced required param (buried-`path` class) ──────────────────────────
//
// VERBATIM captured arguments, 2026-07-13 pi session (gemma-4-26B-A4B-it-qat-4bit,
// ~/.sushi/logs/sushi-11234.log around the us_presidents run). The model
// put `path` INSIDE each edits[] item and emitted no top-level `path`, so pi
// answered
//     Validation failed for tool "edit":
//       - path: must have required properties path
// three times in a row. Each rejection was a full multi-thousand-token
// generation; the model then abandoned `edit` and rewrote the whole file with
// `write` — the same "give up on the cheap tool, re-emit the expensive one"
// degradation as the Claude Code replace_all loop.
//
// The parse layer is NOT at fault (verified: convertGemma4Object/Array are a
// structural walk that preserves the model's own nesting and key order, and
// coerceToolArgsToSchema only rewrites values in place) — the model genuinely
// emitted it this way. The schema is what makes it provably fixable: `path` is
// required at the TOP level and the item schema declares only oldText/newText.
//
// The script bodies are the real ones, excerpted for readability — their LENGTH
// is not load-bearing (the JSON was already valid and correctly escaped; that is
// exactly why no repair path fired). The nesting and the escaping are verbatim.
const pi_buried_path_args_captured =
    \\{"edits":[{"newText":"#!/bin/bash\n\n# Get the directory where the script is located\nSCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\ncd \"$SCRIPT_DIR\"\n","oldText":"#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n","path":"us_presidents/generate_site.sh"}]}
;

test "hoistMisplacedRequiredParams: pi's buried `path` is lifted out of the edits items" {
    const allocator = testing.allocator;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, pi_buried_path_args_captured),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }

    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();

    // The required param now satisfies the schema at the top level…
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expect(path == .string);
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);

    // …and is gone from the item, whose schema never declared it.
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    const item = edits.array.items[0].object;
    try testing.expect(item.get("path") == null);

    // The payload the model worked for survives byte-exact, escaping intact.
    try testing.expectEqualStrings(
        "#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n",
        item.get("oldText").?.string,
    );
    try testing.expect(std.mem.startsWith(u8, item.get("newText").?.string, "#!/bin/bash\n"));
    try testing.expect(std.mem.indexOf(u8, item.get("newText").?.string, "${BASH_SOURCE[0]}") != null);

    // The whole call now conforms — which is the point: pi accepts it.
    try testing.expect(toolCallConformsToSchema(allocator, calls[0], pi_edit_tools_json_test));
}

test "hoistMisplacedRequiredParams: a buried `path` survives the Gemma 4 parse path end to end" {
    // The same class through the REAL parse chain (Gemma's call:name{...} form —
    // this session's model). The raw pre-parse bytes were not dumped (the server
    // was not at --log-level debug), so the tag wrapper here is reconstructed;
    // the ARGUMENT SHAPE it produces is the verbatim captured one asserted above,
    // which is what the hoist actually operates on.
    const allocator = testing.allocator;
    const raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
        "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try hoistMisplacedRequiredParams(allocator, calls, pi_edit_tools_json_test);
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits.array.items[0].object.get("path") == null);
    try testing.expectEqualStrings("old line", edits.array.items[0].object.get("oldText").?.string);
}

test "hoistMisplacedRequiredParams: a compliant call is byte-identical" {
    // The strong-model path: nothing may re-serialize when nothing was misplaced.
    const allocator = testing.allocator;
    const args =
        \\{"path":"a.js","edits":[{"oldText":"a","newText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: never touches a key the item schema DECLARES" {
    // Safety gate. A tool whose items legitimately carry their own `path` (a
    // multi-file edit) must never have it stripped out from under them — hoisting
    // there would DESTROY data, not repair it. The item schema declaring the key
    // is the proof that it belongs where it is.
    const allocator = testing.allocator;
    const multifile_schema =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"}},"required":["path","oldText"]}}},"required":["path","edits"]}}}]
    ;
    const args =
        \\{"edits":[{"path":"a.js","oldText":"a"},{"path":"b.js","oldText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, multifile_schema);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: items that DISAGREE on the value stay honest" {
    // Two items, two different paths, and the tool edits ONE file. There is no
    // correct hoist here — picking either would silently write the wrong file.
    // Leave it broken so the client's validation error reaches the model.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":"a.js"},{"oldText":"c","newText":"d","path":"b.js"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: a wrong-typed buried value is never hoisted" {
    // `path` is declared a string; an object spelled `path` inside the item is
    // not the missing param. Hoisting it would just move the schema violation.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":{"nested":"x"}}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "coerceToolArgsToSchema: quoted integer/number strings coerce to numeric" {
    // A model emitting `"limit":"5"` for an integer param → strict clients reject
    // "expected integer, got string". Coerce string→numeric per the schema.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"scale":{"type":"number"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a.txt\",\"limit\":\"5\",\"scale\":\"1.5\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    const scale = parsed.value.object.get("scale").?;
    try testing.expect(scale == .float or scale == .integer);
    // A string param is untouched.
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: nullable union type [\"string\",\"null\"] coerces to the non-null member" {
    // Real schemas declare optional params as a type UNION. declaredJsonType
    // picks the first non-null member, so a boolean-spelled value under
    // ["boolean","null"] still coerces.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"flag":{"type":["boolean","null"]},"note":{"type":["string","null"]}}}}}]
    ;
    const raw = "<tool_call>\n<function=E>\n<parameter=flag>\nFalse\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("flag").? == .bool);
    try testing.expectEqual(false, parsed.value.object.get("flag").?.bool);
    try testing.expectEqualStrings("hi", parsed.value.object.get("note").?.string);
}

test "coerceToolArgsToSchema: an explicit JSON null satisfies any typed param (not coerced away)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"limit\":null}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    // null is a valid absent-value marker for any type — left byte-identical.
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: coercion is SHALLOW — values nested inside a param are untouched" {
    // Documented boundary: the layer coerces TOP-LEVEL params against the tool's
    // declared property types. It does NOT recurse into array/object element
    // properties (the tool schema's `items`/nested `properties` are the client's
    // to validate). A nested `"port":"8080"` stays a string — byte-identical.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"W","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"W\",\"arguments\":{\"config\":{\"port\":\"8080\",\"tls\":\"false\"}}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: an unmodelled declared type leaves the value alone" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"x":{"type":"weird"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"x\":\"False\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: anyOf, oneOf, allOf and a local $ref resolve to a concrete type" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"S","parameters":{"type":"object","properties":{
        \\"limit":{"anyOf":[{"type":"integer"},{"type":"null"}]},
        \\"ratio":{"oneOf":[{"type":"null"},{"type":"number"}]},
        \\"tags":{"allOf":[{"type":"array","items":{"type":"string"}}]},
        \\"cfg":{"$ref":"#/$defs/Cfg"},
        \\"deep":{"$ref":"#/definitions/Wrap"},
        \\"flag":{"anyOf":[{"type":"boolean"},{"type":"string"}]},
        \\"plain":{"type":"integer"},
        \\"loop":{"$ref":"#/$defs/Loop"}},
        \\"$defs":{"Cfg":{"type":"object"},"Loop":{"$ref":"#/$defs/Loop"}},
        \\"definitions":{"Wrap":{"anyOf":[{"$ref":"#/$defs/Cfg"},{"type":"null"}]}}}}}]
    ;
    const raw = "<tool_call>S<arg_key>limit</arg_key><arg_value>5</arg_value>" ++
        "<arg_key>ratio</arg_key><arg_value>0.5</arg_value>" ++
        "<arg_key>tags</arg_key><arg_value>[\"a\",\"b\"]</arg_value>" ++
        "<arg_key>cfg</arg_key><arg_value>{\"k\":1}</arg_value>" ++
        "<arg_key>deep</arg_key><arg_value>{\"k\":2}</arg_value>" ++
        "<arg_key>flag</arg_key><arg_value>no</arg_value>" ++
        "<arg_key>plain</arg_key><arg_value>7</arg_value>" ++
        "<arg_key>loop</arg_key><arg_value>5</arg_value></tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(i64, 5), o.get("limit").?.integer);
    try testing.expect(o.get("ratio").? == .float);
    try testing.expectEqual(@as(usize, 2), o.get("tags").?.array.items.len);
    try testing.expectEqual(@as(i64, 1), o.get("cfg").?.object.get("k").?.integer);
    try testing.expectEqual(@as(i64, 2), o.get("deep").?.object.get("k").?.integer);
    // A value the first branch cannot take but a later one matches is kept.
    try testing.expectEqualStrings("no", o.get("flag").?.string);
    try testing.expectEqual(@as(i64, 7), o.get("plain").?.integer);
    // A self-referencing $ref terminates and leaves the value alone.
    try testing.expect(o.get("loop") != null);
}

test "toolCallConformsToSchema: a value matching any anyOf branch conforms" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"S","parameters":{"type":"object","properties":{"v":{"anyOf":[{"type":"integer"},{"type":"string"}]}}}}}]
    ;
    const ok = ParsedToolCall{ .name = "S", .arguments = "{\"v\":\"abc\"}" };
    try testing.expect(toolCallConformsToSchema(allocator, ok, tools));
    const bad = ParsedToolCall{ .name = "S", .arguments = "{\"v\":[1]}" };
    try testing.expect(!toolCallConformsToSchema(allocator, bad, tools));
}

// ── DeepSeek-V4 native DSML tool calls ─────────────────────────────────

fn freeParsedCalls(calls: []ParsedToolCall) void {
    for (calls) |tc| {
        testing.allocator.free(tc.name);
        testing.allocator.free(tc.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCalls dsml: canonical two-call block, string and JSON params" {
    const raw = "I'll check the weather.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"get_weather\">\n" ++
        "<｜DSML｜parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "<｜DSML｜invoke name=\"get_time\">\n" ++
        "<｜DSML｜parameter name=\"tz\" string=\"true\">PST</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{\"tz\":\"PST\"}", calls[1].arguments);
    try testing.expect(!calls[0].inferred);
}

test "parseToolCalls dsml: dropped invoke close before the next call still parses both" {
    // Delimiter-drop tolerance: the model omits </｜DSML｜invoke>
    // between back-to-back calls.
    const raw = "<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"first\">\n" ++
        "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n" ++
        "<｜DSML｜invoke name=\"second\">\n" ++
        "<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("first", calls[0].name);
    try testing.expectEqualStrings("{\"a\":\"x\"}", calls[0].arguments);
    try testing.expectEqualStrings("second", calls[1].name);
    try testing.expectEqualStrings("{\"b\":true}", calls[1].arguments);
}

test "parseToolCalls dsml: truncation salvages NAME + completed pairs, never fragments" {
    // Server-side cut mid-value: the unterminated pair is dropped (a half-
    // written file is worse than a re-issued write), completed pairs survive.
    const raw = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.txt</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"content\" string=\"true\">first half of a long fi";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"/tmp/a.txt\"}", calls[0].arguments);
}

test "parseToolCalls dsml: rich string value runs to the confirmed close" {
    // Newlines, quotes, braces, `<` inside a string value — value ends ONLY
    // at the closing parameter tag (the Gemma dropped-quote class).
    const raw = "<｜DSML｜invoke name=\"bash\">\n" ++
        "<｜DSML｜parameter name=\"cmd\" string=\"true\">grep -r \"a < b\" src/\necho {done}</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("grep -r \"a < b\" src/\necho {done}", parsed.value.object.get("cmd").?.string);
}

test "parseToolCalls dsml: duplicate keys dedup (last wins) and invalid JSON value degrades to string" {
    const raw = "<｜DSML｜invoke name=\"t\">\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">old</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">new</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"n\" string=\"false\">not[json</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("new", parsed.value.object.get("k").?.string);
    try testing.expectEqualStrings("not[json", parsed.value.object.get("n").?.string);
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
}

test "parseToolCalls dsml: zero-arg call and thinking prefix" {
    const raw = "thinking about it</think>Sure.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"list_files\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "renderChatTemplate: continuation appends the partial reply into the OPEN assistant turn" {
    // Continuing a reply means the prompt ends mid-assistant-turn. The trailing
    // assistant message is a PREFIX to extend, not history: rendering it as
    // history and then letting add_generation_prompt open a second assistant
    // turn is what the server did before this existed, and the model answers
    // the doubled turn by starting over.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "count to three" },
        .{ .role = "assistant", .content = "one, two," },
    };

    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    // Ends INSIDE the turn, on the partial text as the template writes the
    // turn — no close tag after it.
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistant\none, two,"));
    // Exactly one assistant turn was opened.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "<|im_start|>assistant"));
    // The partial appears once — as the prefill, not also as history.
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, cont, "one, two,"));

    // Same messages WITHOUT the flag stay history + a fresh turn, unchanged.
    const plain = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(plain);
    try testing.expect(std.mem.endsWith(u8, plain, "<|im_start|>assistant"));
    try testing.expectEqual(@as(usize, 2), std.mem.count(u8, plain, "<|im_start|>assistant"));
}

test "renderChatTemplate: continuation resumes in the CONTENT channel, not a think block" {
    // A template that does not render the partial turn as history (this one
    // drops assistant turns) falls back to the generation prompt. One that opens
    // `<think>` unconditionally (LFM2.5 class) would then put the partial reply —
    // visible prose the user already read — inside a reasoning block, and the
    // continuation would come back as reasoning. The same tail that thinking-off
    // uses closes it first.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}{%- if message['role'] != 'assistant' -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n<think>{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Hello ther" },
    };

    // Thinking ON is the hard case: nothing else would close the block.
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<think></think>Hello ther"));
}

test "renderChatTemplate: a continuation prefill is trimmed of trailing whitespace" {
    // The tokenizer merges a trailing space with the word that follows it, so
    // a prompt ending in one splits differently than the same text mid-reply —
    // the hazard Anthropic's API rejects prefills outright for. Trim it: the
    // stored reply keeps its whitespace, only the PROMPT is trimmed.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "hi" },
        .{ .role = "assistant", .content = "Once upon a time \n" },
    };
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "Once upon a time"));
}

test "renderChatTemplate: continuation is declined when the last message is not an assistant" {
    // The flag describes the SHAPE of the request; a client that sets it on a
    // user-final conversation gets an ordinary turn rather than a prompt with
    // the user's own words committed into the assistant's mouth.
    const allocator = testing.allocator;
    const tpl = "{%- for message in messages -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = "",
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};
    const cont = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, true);
    defer allocator.free(cont);
    try testing.expect(std.mem.endsWith(u8, cont, "<|im_start|>assistant"));
}

test "parseToolCalls: Claude-style bare <invoke> markup is an inferred call" {
    const allocator = testing.allocator;
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">cd /x && ls -la</parameter>\n<parameter name=\"timeout\">30</parameter>\n</invoke>\n</function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    // Not the model's own dialect: it must name a declared tool to survive.
    try testing.expect(calls[0].inferred);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("cd /x && ls -la", parsed.value.object.get("command").?.string);
    try testing.expectEqual(@as(i64, 30), parsed.value.object.get("timeout").?.integer);
    try testing.expect((try parseToolCalls(allocator, "I <invoked> it with <parameters> set.")) == null);
    // An unparsed block never reaches content; prose naming the tag stays.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls"));
    const prose = "Claude's format wraps each call in <invoke name=\"x\"> tags.";
    try testing.expectEqualStrings(prose, trimLeakedToolMarkup(prose));
}


test "parseToolCalls: <tool_call>{JSON} truncated mid-string recovers NAME + {} (never a fragment)" {
    // A 4 KB edit call that hit EOS inside a string value. The object never
    // balances, so the balanced-snap arm declines; without a name salvage the
    // whole turn was dropped at the flush (empty `stop`, nothing for the client
    // to nudge on). Same rule as every other dialect: NAME + `{}`.
    const allocator = testing.allocator;
    const text = "Fixing it.\n<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"a.ts\", \"oldText\": \"function foo() {\\n  return 1;";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
    // Reversed key order + whitespace variants still find the name.
    const text2 = "<tool_call>{ \"arguments\" : {\"name\": \"nope\", \"s\": \"a}b\"}, \"name\" : \"real\", \"extra\": \"unterm";
    const c2 = (try parseToolCalls(allocator, text2)) orelse return error.NoCalls;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("real", c2[0].name);
    // No name anywhere: still nothing (no invented call).
    try testing.expect((try parseToolCalls(allocator, "<tool_call>\n{\"arguments\": {\"path\": \"a")) == null);
}

test "parseToolCalls minicpm5: a param value keeps its own leading/trailing blank lines" {
    // Upstream #295 (stripHermesValueFraming) established the rule for the
    // Hermes dialect: the template frames a value with EXACTLY one newline per
    // side, so exactly one is what the parser may remove. Eating a RUN destroys
    // content the model meant to send - an old_string that must match a file
    // "exactly, including indentation" then edits at the wrong place (#294).
    //
    // NOTE the captured model output is value-adjacent, so this framing case is
    // NOT yet observed in the wild; it is the conservative handling for the
    // newline-framed layout rather than a claim that MiniCPM5 emits it. This pins it - the value here is
    // framed by one newline per side AND legitimately begins and ends with a
    // blank line of its own, which must survive.
    const text = "<function name=\"write\">\n<param name=\"content\">\n\nhi\n\n</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\nhi\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: character data AFTER a CDATA section is not discarded" {
    // `<![CDATA[x]]>tail` used to return just `x`: the scan took the LAST
    // `]]>` anywhere in the probe, so anything following the section was
    // silently dropped. Silent truncation of a tool argument is the worst
    // failure mode here - the call still looks well-formed to the client.
    const text = "<function name=\"w\"><param name=\"c\"><![CDATA[x]]>tail</param></function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Not a CDATA-wrapped VALUE (the section does not end it), so it is text.
    try testing.expectEqualStrings("<![CDATA[x]]>tail", parsed.value.object.get("c").?.string);
}

test "minicpm5 COMBINED: parser and scrubber agree on the same bytes" {
    // The earlier defect was two individually-green tests that CONTRADICTED
    // each other: the parser called a string prose while the scrubber cut it as
    // markup. Neither test could see it because neither ran both layers. This
    // one does, so that shape cannot recur silently.
    //
    // Fixtures marked LIVE are verbatim raw emissions of
    // mlx-community/MiniCPM5-1B-OptiQ-4bit captured via SUSHI_RAW_DUMP_FILE.
    const Case = struct {
        raw: []const u8,
        is_call: bool, // parser must produce a call
        keeps_text: bool, // scrubber must leave the text intact
        args: ?[]const u8 = null, // when set, the exact arguments JSON
    };
    const cases = [_]Case{
        // LIVE: normal parameterised call - value-adjacent, no framing newlines.
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status</param></function>", .is_call = true, .keeps_text = false },
        // LIVE: zero-argument call, closed with an empty body.
        .{ .raw = "<function name=\"get_time\"></function>", .is_call = true, .keeps_text = false },
        // LIVE: truncated at max_tokens - a real call, ZERO completed params.
        // Salvages the NAME with empty args; the unterminated value never ships
        // (a fragment argument is worse than none - the call looks complete).
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status", .is_call = true, .keeps_text = false, .args = "{}" },
        // Prose naming a function: no <param, no close. Neither layer may act.
        .{ .raw = "Document the syntax as <function name=\"shell\"> before discussing parameters.", .is_call = false, .keeps_text = true },
        .{ .raw = "Wrap it in a <functional> block - just prose.", .is_call = false, .keeps_text = true },
        // Malformed (dropped quote) WITH structural evidence: not a call, but
        // it is markup, so it must be scrubbed rather than rendered.
        .{ .raw = "<function name=\"shell>\n  <param name=\"command\">pwd</param>\n</function>", .is_call = false, .keeps_text = false },
    };
    for (cases) |c| {
        const parsed = try parseToolCalls(testing.allocator, c.raw);
        if (parsed) |calls| {
            if (c.args) |want| try testing.expectEqualStrings(want, calls[0].arguments);
            freeParsedCalls(calls);
        }
        const got_call = parsed != null;
        testing.expectEqual(c.is_call, got_call) catch |e| {
            std.debug.print("parser disagreed on: '{s}' (got call={})\n", .{ c.raw, got_call });
            return e;
        };
        const kept = std.mem.eql(u8, std.mem.trimEnd(u8, c.raw, "\n\r\t "), trimLeakedToolMarkup(c.raw));
        testing.expectEqual(c.keeps_text, kept) catch |e| {
            std.debug.print("scrubber disagreed on: '{s}' -> '{s}'\n", .{ c.raw, trimLeakedToolMarkup(c.raw) });
            return e;
        };
        // THE invariant: text the parser calls prose is text the scrubber keeps.
        if (!got_call and !kept) {
            const still_markup = std.mem.indexOf(u8, c.raw, "<param") != null or
                std.mem.indexOf(u8, c.raw, "</function>") != null;
            testing.expect(still_markup) catch |e| {
                std.debug.print("CONTRADICTION: parser says prose, scrubber cut it: '{s}'\n", .{c.raw});
                return e;
            };
        }
    }
}

test "parseToolCalls minicpm5: an UNCLOSED named opener in prose is not a call" {
    // The truncation salvage recovers NAME + complete params when EOS cut a
    // real call. It cannot, on its own, tell that apart from a model WRITING
    // ABOUT the syntax - and an unclosed opener with no params was being
    // promoted into an executable `shell({})`. For a zero-arg tool that is an
    // unintended invocation; otherwise it replaces the answer with a bogus call.
    //
    // Rule: an unclosed function must carry structural evidence that a call was
    // under way - at minimum a `<param` opener. (It was briefly a COMPLETE
    // `<param>...</param>`; a real max_tokens capture has zero completed pairs,
    // so that discarded genuine calls.) A properly CLOSED zero-arg call is
    // still a call - the close tag is the model's own commitment.
    const prose = [_][]const u8{
        "Document the syntax as <function name=\"shell\"> before discussing parameters.",
        "You write <function name=\"get_time\"> to open a call.",
    };
    for (prose) |t| {
        if (try parseToolCalls(testing.allocator, t)) |calls| {
            defer freeParsedCalls(calls);
            std.debug.print("PROSE became a call: '{s}' -> {s}({s})\n", .{ t, calls[0].name, calls[0].arguments });
            return error.ProseBecameToolCall;
        }
    }

    // A CLOSED zero-arg call still parses.
    const closed = "<function name=\"get_time\"></function>";
    const c1 = (try parseToolCalls(testing.allocator, closed)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c1);
    try testing.expectEqualStrings("get_time", c1[0].name);
    try testing.expectEqualStrings("{}", c1[0].arguments);

    // A genuinely TRUNCATED call that got one complete param through still
    // salvages - that is the case the branch exists for.
    const cut = "<function name=\"shell\">\n<param name=\"command\">ls</param>\n<param name=\"cwd\">/tm";
    const c2 = (try parseToolCalls(testing.allocator, cut)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("shell", c2[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c2[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("cwd") == null);
}

test "renderChatTemplate: a NUL byte in message content does not truncate the prompt" {
    // Bar: content after the NUL, and the assistant header, survive the render.
    const allocator = testing.allocator;
    const tpl =
        \\{%- for message in messages -%}
        \\<|im_start|>{{ message['role'] }}
        \\{{ message['content'] }}<|im_end|>
        \\{% endfor %}{%- if add_generation_prompt -%}<|im_start|>assistant
        \\{% endif %}
    ;
    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };

    const messages = [_]Message{
        .{ .role = "user", .content = "before\x00after" },
    };
    const rendered = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(rendered);

    try testing.expect(std.mem.indexOf(u8, rendered, "before") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "after") != null);
    try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>assistant") != null);
}

test "appendJsonString: invalid UTF-8 becomes U+FFFD, valid multibyte is byte-exact" {
    // Bar: the escaper's output is always well-formed UTF-8 (nlohmann rejects
    // anything else), and every valid sequence survives unchanged.
    const allocator = testing.allocator;
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try appendJsonString(allocator, &buf, "\xff\xfe\x80 binary");
    try testing.expect(std.unicode.utf8ValidateSlice(buf.items));
    try testing.expectEqualStrings("\"\u{FFFD}\u{FFFD}\u{FFFD} binary\"", buf.items);

    buf.clearRetainingCapacity();
    const valid = "héllo \u{4e2d}\u{6587} \u{1F600} ok";
    try appendJsonString(allocator, &buf, valid);
    try testing.expectEqualStrings("\"" ++ valid ++ "\"", buf.items);

    // A truncated multibyte sequence at end-of-input is one replacement.
    buf.clearRetainingCapacity();
    try appendJsonString(allocator, &buf, "a\xe4\xb8");
    try testing.expectEqualStrings("\"a\u{FFFD}\"", buf.items);
}

test "serializeMessagesJson: a string-content template gets each image as its atom where it sits, blank-line joined" {
    const a = testing.allocator;
    const parts = [_]MediaPart{ .{ .at = 0, .kind = .image }, .{ .at = 5, .kind = .image }, .{ .at = 5, .kind = .video } };
    const msgs = [_]Message{
        .{ .role = "user", .content = "AAAAABBB", .media_parts = &parts },
        .{ .role = "tool", .content = "result", .media_parts = parts[0..1] },
        .{ .role = "user", .content = "plain" },
    };
    var cfg = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = "<IMG>" };
    const json = try serializeMessagesJsonFor(a, &msgs, &cfg);
    defer a.free(json);
    try testing.expectEqualStrings(
        "[{\"role\":\"user\",\"content\":\"<IMG>\\n\\nAAAAA\\n\\n<IMG>\\n\\nBBB\"}," ++
            "{\"role\":\"tool\",\"content\":\"<IMG>\\n\\nresult\"},{\"role\":\"user\",\"content\":\"plain\"}]",
        json,
    );
    // Without the atom the same message renders as typed parts.
    cfg.image_placeholder = null;
    const typed = try serializeMessagesJsonFor(a, msgs[0..1], &cfg);
    defer a.free(typed);
    try testing.expect(std.mem.indexOf(u8, typed, "{\"type\":\"image\"}") != null);
}

test "foldSystemMessages: a system turn past index 0 joins the leading system message" {
    // Live 2026-09-15: Claude Code carries SessionStart hook output as a
    // `system`-role message INSIDE `messages`, after the top-level system
    // prompt. Qwen's template raises on a system turn that is not first, and
    // the raise is a silent generic fallback (the model loses its stop token).
    const al = std.testing.allocator;
    var msgs = std.ArrayList(Message).empty;
    defer msgs.deinit(al);
    try msgs.append(al, .{ .role = "system", .content = "You are S." });
    try msgs.append(al, .{ .role = "system", .content = "hook output" });
    try msgs.append(al, .{ .role = "user", .content = "hi" });
    try msgs.append(al, .{ .role = "system", .content = "late note" });
    const owned = try foldSystemMessages(al, &msgs);
    defer if (owned) |o| al.free(o);
    try std.testing.expectEqual(@as(usize, 2), msgs.items.len);
    try std.testing.expectEqualStrings("You are S.\n\nhook output\n\nlate note", msgs.items[0].content);
    try std.testing.expectEqualStrings("user", msgs.items[1].role);

    // No leading system: the fold creates one at index 0.
    var lone = std.ArrayList(Message).empty;
    defer lone.deinit(al);
    try lone.append(al, .{ .role = "user", .content = "hi" });
    try lone.append(al, .{ .role = "system", .content = "hook output" });
    const owned2 = try foldSystemMessages(al, &lone);
    defer if (owned2) |o| al.free(o);
    try std.testing.expectEqualStrings("system", lone.items[0].role);
    try std.testing.expectEqualStrings("hook output", lone.items[0].content);
    try std.testing.expectEqual(@as(usize, 2), lone.items.len);

    // Nothing to fold: untouched, nothing allocated.
    var plain = std.ArrayList(Message).empty;
    defer plain.deinit(al);
    try plain.append(al, .{ .role = "system", .content = "You are S." });
    try plain.append(al, .{ .role = "user", .content = "hi" });
    try std.testing.expect((try foldSystemMessages(al, &plain)) == null);
    try std.testing.expectEqual(@as(usize, 2), plain.items.len);
}

test "GLM serving template receives low high max without a derived reasoning budget" {
    const tpl = "[gMASK]<sop>{% set effective_reasoning_effort = reasoning_effort if reasoning_effort in ['low', 'high'] else 'max' %}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}<|assistant|><think>";
    var cfg = ChatConfig{ .allocator = testing.allocator, .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false };
    try testing.expect(templateConsumesEffort(tpl));
    for ([_]struct { effort: ?[]const u8, label: []const u8 }{
        .{ .effort = null, .label = "High" },
        .{ .effort = "low", .label = "Low" },
        .{ .effort = "high", .label = "High" },
        .{ .effort = "max", .label = "Max" },
    }) |c| {
        const rendered = try renderChatTemplate(testing.allocator, &.{}, &cfg, null, null, true, c.effort, false);
        defer testing.allocator.free(rendered);
        const want = try std.fmt.allocPrint(testing.allocator, "Reasoning Effort: {s}<|assistant|><think>", .{c.label});
        defer testing.allocator.free(want);
        try testing.expect(std.mem.endsWith(u8, rendered, want));
    }
}

const glm53_template = @embedFile("fixtures/glm53_chat_template.jinja");

fn glm53Config(allocator: std.mem.Allocator) ChatConfig {
    return .{ .chat_template = glm53_template, .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = allocator };
}

test "jinja: a number after a dot subscripts, as in Jinja2" {
    const tpl = "{{ x.0.y }}|{{ x.1.0 }}|{{ x.1.1 }}|{{ s.0 }}|{{ x[1].0 }}|{{ x.1.0 + 0.5 }}|{{ 'no' if x.5 is not defined else 'yes' }}|{{ 1.5 }}";
    const extra = "{\"x\":[{\"y\":\"a\"},[7,8]],\"s\":\"hey\"}";
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(tpl, "[]", null, extra, 0, &len) orelse {
        std.debug.print("\njinja error: {s}\n", .{std.mem.span(jinja_c.jinja_last_error().?)});
        return error.RenderFailed;
    };
    defer jinja_c.jinja_str_free(ptr);
    try testing.expectEqualStrings("a|7|8|h|7|7.5|no|1.5", ptr[0..len]);
}

fn renderJinjaRaw(tpl: [:0]const u8, messages: [:0]const u8, expect: []const u8) !void {
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(tpl, messages, null, null, 0, &len) orelse {
        std.debug.print("\njinja error: {s}\n", .{std.mem.span(jinja_c.jinja_last_error().?)});
        return error.RenderFailed;
    };
    defer jinja_c.jinja_str_free(ptr);
    try testing.expectEqualStrings(expect, ptr[0..len]);
}

test "jinja: a set in a loop body ends with its iteration, as in Jinja2" {
    try renderJinjaRaw(
        "{% for m in messages %}{% if m.r is string %}{% set r = m.r %}{% endif %}{% if r is defined %}[{{ r }}]{% else %}[]{% endif %}{% endfor %}",
        "[{\"r\":\"a\"},{},{\"r\":\"b\"},{}]",
        "[a][][b][]",
    );
    try renderJinjaRaw("{% for i in [1,2] %}{% set z = i %}{% endfor %}{% if z is defined %}Y{% else %}N{% endif %}", "[]", "N");
    try renderJinjaRaw("{% set a = 1 %}{% for i in [1,2] %}{% set a = a + i %}{{ a }}{% endfor %}{{ a }}", "[]", "231");
    try renderJinjaRaw("{% for i in [1,2] %}{% set z = i %}{% for j in [10,20] %}{{ z + j }}{% endfor %}{% endfor %}", "[]", "11211222");
    try renderJinjaRaw("{% for i in [] %}x{% else %}{% set q = 1 %}E{% endfor %}{% if q is defined %}Y{% else %}N{% endif %}", "[]", "EN");
}

test "jinja: a namespace mutated in a loop body accumulates across iterations" {
    try renderJinjaRaw("{% set ns = namespace(n=0) %}{% for i in [1,2,3] %}{% set ns.n = ns.n + i %}{% endfor %}{{ ns.n }}", "[]", "6");
    try renderJinjaRaw(
        "{% set ns = namespace(s='') %}{% for i in [1,2] %}{% for j in [3,4] %}{% set ns.s = ns.s ~ i ~ j ~ ',' %}{% endfor %}{% endfor %}{{ ns.s }}",
        "[]",
        "13,14,23,24,",
    );
}

test "GLM-5.3 template renders a tool round-trip itself, never the generic fallback" {
    const a = testing.allocator;
    var config = glm53Config(a);
    const tools =
        \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    ;
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" }};
    const messages = [_]Message{
        .{ .role = "system", .content = "You are helpful." },
        .{ .role = "user", .content = "What is the weather in Paris?" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "Need the weather." },
        .{ .role = "tool", .content = "Sunny, 21C", .tool_call_id = "call_1" },
    };
    const want = "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>\n# Tools\n\n" ++
        "You may call one or more functions to assist with the user query.\n\n" ++
        "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n" ++
        "{\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}\n" ++
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n" ++
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>" ++
        "<|system|>You are helpful.<|user|>What is the weather in Paris?" ++
        "<|assistant|><think>Need the weather.</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
        "<|observation|><tool_response>Sunny, 21C</tool_response><|assistant|><think>";
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(a, &messages, &config, tools, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqual(before, template_fallbacks.load(.monotonic));
    try testing.expectEqualStrings(want, rendered);
}

/// Renders `messages_json` through GLM-5.3's own template; the text from the first user turn on.
fn glm53RenderRaw(messages_json: [:0]const u8, tools_json: ?[:0]const u8, out: *std.ArrayList(u8)) !void {
    const extra = "{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}";
    var len: usize = 0;
    const ptr = jinja_c.jinja_render_chat(glm53_template, messages_json, if (tools_json) |t| t.ptr else null, extra, 1, &len) orelse {
        std.debug.print("\njinja error: {s}\n", .{std.mem.span(jinja_c.jinja_last_error().?)});
        return error.RenderFailed;
    };
    defer jinja_c.jinja_str_free(ptr);
    const text = ptr[0..len];
    try out.appendSlice(testing.allocator, text[std.mem.indexOf(u8, text, "<|user|>").?..]);
}

test "GLM-5.3 template: a tool result given as a list of outputs, text parts or tool references renders in place" {
    const user = "{\"role\":\"user\",\"content\":\"Q\"}";
    const call = "{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"R\",\"tool_calls\":[" ++
        "{\"type\":\"function\",\"id\":\"call_1\",\"function\":{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}}," ++
        "{\"type\":\"function\",\"id\":\"call_2\",\"function\":{\"name\":\"get_time\",\"arguments\":{}}}]}";
    const head = "<|user|>Q<|assistant|><think>R</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
        "<tool_call>get_time</tool_call><|observation|>";
    const tail = "<|assistant|><think>";
    const tools =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{}}}}]
    ;
    const cases = [_]struct { tool: [:0]const u8, tools: ?[:0]const u8, want: []const u8 }{
        .{
            .tool = "{\"role\":\"tool\",\"content\":[{\"tool_call_id\":\"call_2\",\"output\":\"12:00\"},{\"tool_call_id\":\"call_1\",\"output\":\"Sunny\"}]}",
            .tools = null,
            .want = head ++ "<tool_response>Sunny</tool_response><tool_response>12:00</tool_response>" ++ tail,
        },
        .{
            .tool = "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":[{\"type\":\"text\",\"text\":\"Sunny\"},{\"type\":\"image\"}]}",
            .tools = null,
            .want = head ++ "<tool_response>Sunny<|begin_of_image|><|image|><|end_of_image|></tool_response>" ++ tail,
        },
        .{
            .tool = "{\"role\":\"tool\",\"tool_call_id\":\"call_1\",\"content\":[{\"type\":\"tool_reference\",\"name\":\"get_time\"}]}",
            .tools = tools,
            .want = head ++ "<tool_response><tools>\n{\"name\": \"get_time\", \"description\": \"Get time\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}\n</tools></tool_response>" ++ tail,
        },
    };
    for (cases) |c| {
        const json = try std.fmt.allocPrintSentinel(testing.allocator, "[{s},{s},{s}]", .{ user, call, c.tool }, 0);
        defer testing.allocator.free(json);
        var out = std.ArrayList(u8).empty;
        defer out.deinit(testing.allocator);
        try glm53RenderRaw(json, c.tools, &out);
        try testing.expectEqualStrings(c.want, out.items);
    }
}

test "GLM-5.3 template: parallel results come back in the order of the calls" {
    const a = testing.allocator;
    var config = glm53Config(a);
    const calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"city\":\"Paris\"}" },
        .{ .id = "call_2", .name = "get_time", .arguments = "{}" },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "Q" },
        .{ .role = "assistant", .content = "", .tool_calls = &calls, .reasoning_content = "R" },
        .{ .role = "tool", .content = "12:00", .tool_call_id = "call_2" },
        .{ .role = "tool", .content = "Sunny", .tool_call_id = "call_1" },
    };
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(rendered);
    try testing.expectEqual(before, template_fallbacks.load(.monotonic));
    const from = std.mem.indexOf(u8, rendered, "<|user|>").?;
    try testing.expectEqualStrings(
        "<|user|>Q<|assistant|><think>R</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>" ++
            "<tool_call>get_time</tool_call><|observation|><tool_response>Sunny</tool_response><tool_response>12:00</tool_response><|assistant|><think>",
        rendered[from..],
    );
}

test "a template that raises or is missing is refused unless the model ends turns with <|im_end|>" {
    const user = [_]Message{.{ .role = "user", .content = "hi" }};
    for ([_][]const u8{ "{{ raise_exception('refused') }}", "" }) |tpl| {
        for ([_]?[]const u8{ null, "<|endoftext|>", "<｜end▁of▁sentence｜>" }) |eos| {
            var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = eos, .add_bos_token = false, .allocator = testing.allocator };
            const before = template_fallbacks.load(.monotonic);
            try testing.expectError(error.ChatTemplateUnusable, renderChatTemplate(testing.allocator, &user, &config, null, null, true, null, false));
            try testing.expectEqual(before, template_fallbacks.load(.monotonic));
        }
        var chatml = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = testing.allocator };
        const rendered = try renderChatTemplate(testing.allocator, &user, &chatml, null, null, true, null, false);
        defer testing.allocator.free(rendered);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n"));
    }
}

test "a template that raises counts one generic fallback" {
    var config = ChatConfig{
        .chat_template = "{{ raise_exception('refused') }}",
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = testing.allocator,
    };
    const before = template_fallbacks.load(.monotonic);
    const rendered = try renderChatTemplate(testing.allocator, &.{.{ .role = "user", .content = "hi" }}, &config, null, null, true, null, false);
    defer testing.allocator.free(rendered);
    try testing.expectEqual(before + 1, template_fallbacks.load(.monotonic));
}

test "request nesting limit rejects deep JSON bodies only" {
    const a = testing.allocator;
    const deep = try a.alloc(u8, 2 * (max_json_nesting + 1));
    defer a.free(deep);
    @memset(deep[0 .. max_json_nesting + 1], '[');
    @memset(deep[max_json_nesting + 1 ..], ']');
    try testing.expect(jsonNestingExceeds(deep, max_json_nesting));

    const ok = try a.alloc(u8, 2 * 64);
    defer a.free(ok);
    @memset(ok[0..64], '[');
    @memset(ok[64..], ']');
    try testing.expect(!jsonNestingExceeds(ok, max_json_nesting));

    try testing.expect(!jsonNestingExceeds("{\"s\":\"[[[[\\\"[[[[\"}", 2));
    try testing.expect(!jsonNestingExceeds("not json [[[[[[", 2));
}

test "history tool arguments nested past the limit embed an empty object" {
    const a = testing.allocator;
    var deep: std.ArrayList(u8) = .empty;
    defer deep.deinit(a);
    try deep.appendSlice(a, "{\"a\":");
    try deep.appendNTimes(a, '[', max_json_nesting);
    try deep.appendNTimes(a, ']', max_json_nesting);
    try deep.append(a, '}');
    const calls = [_]ToolCall{.{ .id = "c", .name = "f", .arguments = deep.items }};
    const msgs = [_]Message{.{ .role = "assistant", .content = "", .tool_calls = &calls }};
    const json = try serializeMessagesJson(a, &msgs);
    defer a.free(json);
    try testing.expect(std.mem.indexOf(u8, json, "\"arguments\":{}") != null);
}

test "chat: V4.1 renders images as the release's encoder does, an atom for each where it sat" {
    // Expected strings are `encoding.encode_messages` on the same blocks (image_url parts, thinking_mode chat/thinking).
    const a = testing.allocator;
    const img1 = [_]ImageData{.{ .pixels = &.{}, .width = 1, .height = 1 }};
    const lead = [_]MediaPart{.{ .at = 0, .kind = .image }};
    const tail = [_]MediaPart{.{ .at = 2, .kind = .image }};
    const cases = [_]struct { msgs: []const Message, chat: []const u8, thinking: []const u8 }{
        .{
            .msgs = &.{.{ .role = "user", .content = "What is this?", .images = &img1, .media_parts = &lead }},
            .chat = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}><\u{ff5c}deepseek_image\u{ff5c}>\n\nWhat is this?<\u{ff5c}Assistant\u{ff5c}></think>",
            .thinking = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}System\u{ff5c}>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\u{ff5c}User\u{ff5c}><\u{ff5c}deepseek_image\u{ff5c}>\n\nWhat is this?<\u{ff5c}Assistant\u{ff5c}><think>",
        },
        .{
            .msgs = &.{.{ .role = "user", .content = "Look:Describe.", .images = &img1, .media_parts = &.{.{ .at = 5, .kind = .image }} }},
            .chat = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>Look:\n\n<\u{ff5c}deepseek_image\u{ff5c}>\n\nDescribe.<\u{ff5c}Assistant\u{ff5c}></think>",
            .thinking = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}System\u{ff5c}>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\u{ff5c}User\u{ff5c}>Look:\n\n<\u{ff5c}deepseek_image\u{ff5c}>\n\nDescribe.<\u{ff5c}Assistant\u{ff5c}><think>",
        },
        .{
            .msgs = &.{
                .{ .role = "system", .content = "S" },
                .{ .role = "user", .content = "q1", .images = &img1, .media_parts = &lead },
                .{ .role = "assistant", .content = "a1" },
                .{ .role = "user", .content = "q2", .images = &img1, .media_parts = &tail },
            },
            .chat = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}System\u{ff5c}>S<\u{ff5c}User\u{ff5c}><\u{ff5c}deepseek_image\u{ff5c}>\n\nq1<\u{ff5c}Assistant\u{ff5c}></think>a1<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>q2\n\n<\u{ff5c}deepseek_image\u{ff5c}><\u{ff5c}Assistant\u{ff5c}></think>",
            .thinking = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}System\u{ff5c}>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nS<\u{ff5c}User\u{ff5c}><\u{ff5c}deepseek_image\u{ff5c}>\n\nq1<\u{ff5c}Assistant\u{ff5c}></think>a1<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>q2\n\n<\u{ff5c}deepseek_image\u{ff5c}><\u{ff5c}Assistant\u{ff5c}><think>",
        },
    };
    var cfg = ChatConfig{ .chat_template = @embedFile("fixtures/dsv41_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = dsv41_image_atom };
    const tpl_z = try a.dupeSentinel(u8, cfg.chat_template, 0);
    defer a.free(tpl_z);
    for (cases) |c| for ([_]struct { mode: []const u8, want: []const u8 }{ .{ .mode = "chat", .want = c.chat }, .{ .mode = "thinking", .want = c.thinking } }) |arm| {
        const json = try serializeMessagesJsonFor(a, c.msgs, &cfg);
        defer a.free(json);
        const json_z = try a.dupeSentinel(u8, json, 0);
        defer a.free(json_z);
        const extra = try std.fmt.allocPrintSentinel(a, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":\"high\"}}", .{arm.mode}, 0);
        defer a.free(extra);
        var len: usize = 0;
        const ptr = jinja_c.jinja_render_chat(tpl_z.ptr, json_z.ptr, null, extra.ptr, 1, &len) orelse return error.RenderFailed;
        defer jinja_c.jinja_str_free(ptr);
        try testing.expectEqualStrings(arm.want, ptr[0..len]);
    };
}

test "chat: the embedded V4.1 template renders through jinja.cpp byte-exact with DeepSeek's encoder" {
    const a = testing.allocator;
    const tpl = @embedFile("fixtures/dsv41_chat_template.jinja");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/dsv41_template_cases.json"), .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |case| {
        const o = case.object;
        const msgs = try std.json.Stringify.valueAlloc(a, o.get("messages").?, .{});
        defer a.free(msgs);
        const tools: ?[]u8 = if (o.get("tools").? == .null) null else try std.json.Stringify.valueAlloc(a, o.get("tools").?, .{});
        defer if (tools) |t| a.free(t);
        const extra = try std.fmt.allocPrint(a, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":\"{s}\"}}", .{ o.get("thinking_mode").?.string, o.get("reasoning_effort").?.string });
        defer a.free(extra);
        const tpl_z = try a.dupeSentinel(u8, tpl, 0);
        defer a.free(tpl_z);
        const msgs_z = try a.dupeSentinel(u8, msgs, 0);
        defer a.free(msgs_z);
        const extra_z = try a.dupeSentinel(u8, extra, 0);
        defer a.free(extra_z);
        const tools_z = if (tools) |t| try a.dupeSentinel(u8, t, 0) else null;
        defer if (tools_z) |t| a.free(t);
        var len: usize = 0;
        const ptr = jinja_c.jinja_render_chat(tpl_z.ptr, msgs_z.ptr, if (tools_z) |t| t.ptr else null, extra_z.ptr, 1, &len) orelse {
            std.debug.print("{s}: {s}\n", .{ o.get("label").?.string, if (jinja_c.jinja_last_error()) |e| std.mem.span(e) else "render failed" });
            return error.RenderFailed;
        };
        defer jinja_c.jinja_str_free(ptr);
        testing.expectEqualStrings(o.get("expected").?.string, ptr[0..len]) catch |e| {
            std.debug.print("case: {s}\n", .{o.get("label").?.string});
            return e;
        };
    }
}

test "chat: the V4.1 effort words stay on the encoder's low|high|max ladder" {
    try testing.expectEqualStrings("high", dsv41EffortFor(null));
    try testing.expectEqualStrings("low", dsv41EffortFor("minimal"));
    try testing.expectEqualStrings("high", dsv41EffortFor("medium"));
    try testing.expectEqualStrings("max", dsv41EffortFor("xhigh"));
}

/// One DeepSeek-V4.1 turn replayed the way a client does: the model's tokens for `raw`, the server's split and parse of
/// its text, and the next request's prompt tokens through the real template.
pub const Dsv41Replay = struct {
    gen: []u32,
    /// Generated tokens the next prompt repeats, from the first one.
    reused: usize,
    /// The next prompt starts with the first prompt.
    prefix_kept: bool,
    think_close_at: ?usize,
    markup_at: ?usize,

    pub fn deinit(self: *Dsv41Replay, a: std.mem.Allocator) void {
        a.free(self.gen);
    }
};

/// One turn of a hybrid-state arch replayed the way a client does (the pack's own template and tokenizer): the
/// generated tokens for `raw`, the server's split and parse of its text, and the next request's prompt.
const Replay = struct {
    gen: []u32,
    /// Generated tokens the next prompt repeats, from the first one.
    reused: usize,
    /// The next prompt starts with the first prompt.
    prefix_kept: bool,
    think_close_at: ?usize,

    pub fn deinit(self: *Replay, a: std.mem.Allocator) void {
        a.free(self.gen);
    }
};

pub fn replayTurn(a: std.mem.Allocator, tok: *const Tokenizer, cfg: *const ChatConfig, tools: ?[]const u8, thinking: bool, raw: []const u8) !Replay {
    const first = [_]Message{.{ .role = "user", .content = "q" }};
    const p1_text = try renderChatTemplate(a, &first, cfg, tools, null, thinking, null, false);
    defer a.free(p1_text);
    const p1 = try tok.encodeMarked(a, p1_text);
    defer a.free(p1);
    const gen = try tok.encodeMarked(a, raw);
    errdefer a.free(gen);

    const split = splitThinkBlock(raw, true, thinking);
    const parsed = if (tools != null) try parseToolCalls(a, raw) else null;
    defer if (parsed) |calls| {
        for (calls) |c| {
            a.free(c.name);
            a.free(c.arguments);
        }
        a.free(calls);
    };
    var calls: [4]ToolCall = undefined;
    if (parsed) |pc| for (pc, 0..) |c, i| {
        calls[i] = .{ .id = "call_1", .name = c.name, .arguments = c.arguments };
    };
    const assistant: Message = .{
        .role = "assistant",
        .content = split.content,
        .reasoning_content = split.reasoning_content,
        .tool_calls = if (parsed) |pc| calls[0..pc.len] else null,
    };
    const tail: Message = if (parsed != null) .{ .role = "tool", .content = "ok", .tool_call_id = "call_1" } else .{ .role = "user", .content = "next" };
    const second = [_]Message{ first[0], assistant, tail };
    const p2_text = try renderChatTemplate(a, &second, cfg, tools, null, thinking, null, false);
    defer a.free(p2_text);
    const p2 = try tok.encodeMarked(a, p2_text);
    defer a.free(p2);

    const keep = p1.len <= p2.len and std.mem.eql(u32, p1, p2[0..p1.len]);
    var reused: usize = 0;
    if (keep) {
        const tail_ids = p2[p1.len..];
        while (reused < gen.len and reused < tail_ids.len and gen[reused] == tail_ids[reused]) reused += 1;
    }
    const close_id = tok.specialTokenId("</think>");
    return .{
        .gen = gen,
        .reused = reused,
        .prefix_kept = keep,
        .think_close_at = if (close_id) |id| std.mem.indexOfScalar(u32, gen, id) else null,
    };
}

const NextReuse = enum {
    /// Every generated token comes back.
    whole,
    /// Through the thought, short of the closer: the template re-renders the thought's edges.
    thought_short,
    /// The turn is not re-rendered as it was generated.
    none,
};

pub const ReplayCase = struct { name: []const u8, tools: ?[]const u8, thinking: bool, raw: []const u8, reuse: NextReuse };
