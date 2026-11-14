const std = @import("std");
const builtin = @import("builtin");
const media_mode = @import("arch/media_mode.zig");
const effort_mod = @import("effort.zig");
const arch_chat = @import("arch/chat.zig");
const template_engine = @import("template/template.zig");
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");

const Tokenizer = tokenizer_mod.Tokenizer;
const tool_calls_mod = @import("tool_calls.zig");
const tool_schema = @import("tool_schema.zig");
const think_tags = @import("think_tags.zig");

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
    mode: media_mode.Mode = .gemma,
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
/// Every role a chat request may give a message.
pub const request_roles = [_][]const u8{ "system", "developer", "user", "assistant", "tool", "function" };

/// The role a template knows `role` by: OpenAI's `developer` is the system turn and its older `function` result a tool's.
pub fn canonicalRole(role: []const u8) []const u8 {
    if (std.mem.eql(u8, role, "developer")) return "system";
    if (std.mem.eql(u8, role, "function")) return "tool";
    return role;
}

/// A request's messages in the order its model's encoder renders them in (`arch_chat.orderMessages`): every surface
/// calls it where it has read them.
pub fn orderMessages(model_type: []const u8, messages: []Message) void {
    arch_chat.orderMessages(model_type, messages);
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

pub const JoinedText = struct { text: []const u8, owned: bool };

/// The one rule every surface joins a message's text parts (and its reasoning parts) by: an empty part drops, the
/// rest join with '\n'. One part borrows its bytes (`owned=false`); 2+ allocate (`owned=true`, the caller frees).
///
/// Multiple text parts are real traffic: opencode plan mode appends its "Plan Mode - System Reminder" as a SECOND text
/// part on the user message, and last-wins here dropped the user's actual prompt (issue #195).
pub const TextJoin = struct {
    single: []const u8 = "",
    buf: std.ArrayList(u8) = .empty,
    n: usize = 0,

    pub fn add(self: *TextJoin, allocator: std.mem.Allocator, text: []const u8) !void {
        if (text.len == 0) return;
        self.n += 1;
        if (self.n == 1) {
            self.single = text;
            return;
        }
        if (self.n == 2) try self.buf.appendSlice(allocator, self.single);
        try self.buf.append(allocator, '\n');
        try self.buf.appendSlice(allocator, text);
    }

    /// The joined length so far: where a media part between text parts renders.
    pub fn len(self: *const TextJoin) usize {
        return if (self.n <= 1) self.single.len else self.buf.items.len;
    }

    pub fn finish(self: *TextJoin, allocator: std.mem.Allocator) !JoinedText {
        if (self.n <= 1) return .{ .text = self.single, .owned = false };
        return .{ .text = try self.buf.toOwnedSlice(allocator), .owned = true };
    }

    pub fn deinit(self: *TextJoin, allocator: std.mem.Allocator) void {
        self.buf.deinit(allocator);
    }
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
    /// Who wrote each byte of `content` when the server put it together from its own text and the client's (the tool
    /// fallbacks); null when all of it is the role's.
    content_origins: ?[]const template_engine.Run = null,

    /// A message a client sent with nothing in it: no text, call, media or reasoning. Every surface's reader drops
    /// one, so a conversation renders the same prompt on each; a tool result is kept, empty or not.
    pub fn carriesNothing(self: Message) bool {
        return self.content.len == 0 and self.tool_calls == null and self.images == null and self.videos == null and
            self.audio == null and self.reasoning_content == null and !std.mem.eql(u8, self.role, "tool");
    }

    /// This message as `role`, its content `before ++ content ++ after`: every other field is carried, and its media
    /// moves with the text it sat in.
    pub fn derive(self: Message, arena: std.mem.Allocator, role: []const u8, before: []const template_engine.Str, after: []const template_engine.Str) !Message {
        var b: template_engine.Builder = .{};
        for (before) |p| try b.append(arena, p);
        const shift = b.bytes.items.len;
        try b.append(arena, contentSpans(self));
        for (after) |p| try b.append(arena, p);
        var out = self;
        out.role = role;
        out.content = b.bytes.items;
        out.content_origins = b.runs.items;
        if (self.media_parts) |parts| if (shift > 0) {
            const moved = try arena.alloc(MediaPart, parts.len);
            for (parts, moved) |p, *m| m.* = .{ .at = p.at + shift, .kind = p.kind };
            out.media_parts = moved;
        };
        return out;
    }
};

/// Only a template that reads `preserve_thinking` is handed the switch; `--preserve-thinking` and the per-model key do nothing elsewhere.
pub fn templateTakesPreserveThinking(chat_config: *const ChatConfig) bool {
    return asks.preserveThinking(chat_config.facts());
}

/// A template with a thinking span: the `think_penalty` needs its closing marker to arm.
pub fn templateHasThinkSpan(chat_config: *const ChatConfig) bool {
    return asks.thinkSpan(chat_config.facts());
}

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
    /// The checkpoint's `model_type` (config.json), owned when loaded: only that arch's face edits the template
    /// (`arch_chat.lateSystemTemplate`). Null edits nothing.
    model_type: ?[]const u8 = null,
    /// `chat_template` compiled, with the late-system form its arch edits in: set by `loadChatConfig`, so no
    /// request parses a template. Null on a config built by hand (a test), which compiles for each render.
    compiled: ?*Compiled = null,

    /// Compiles the template and its late-system form; a template that does not compile is
    /// `error.ChatTemplateInvalid`, logged with its line and column.
    pub fn compile(self: *ChatConfig) !void {
        return self.compileSaying(true);
    }

    fn compileSaying(self: *ChatConfig, say: bool) !void {
        self.dropCompiled();
        if (self.chat_template.len == 0) return;
        const c = try Compiled.init(self.allocator, self.chat_template, true, self.model_type);
        if (c.failure()) |f| {
            var buf: [1200]u8 = undefined;
            if (say) log.err("[load] chat template: {s}\n", .{f.describe(&buf)});
            c.deinit(self.allocator);
            return error.ChatTemplateInvalid;
        }
        self.compiled = c;
        c.turns_arena = std.heap.ArenaAllocator.init(self.allocator);
        c.turns = try assistantTurnRenders(c.turns_arena.?.allocator(), self);
    }

    /// What the compiled template says about itself (`template/facts.zig`); a config with no template has none.
    /// A template never compiled, or swapped in after the compile, has no answer: empty facts would answer "no" to
    /// every question.
    pub fn compiledFacts(self: *const ChatConfig) error{TemplateNotCompiled}!*const Facts {
        if (self.compiled) |c| if (c.compiledFrom(self.chat_template)) return c.facts();
        return if (self.chat_template.len == 0) &no_facts else error.TemplateNotCompiled;
    }

    /// `compiledFacts`, for a config that is compiled by contract: every load compiles, a test calls `compile`.
    pub fn facts(self: *const ChatConfig) *const Facts {
        return self.compiledFacts() catch @panic("ChatConfig: a fact asked of a template never compiled; call compile()");
    }

    /// What the template may print and where, for the reserved-token mask: null with no template, so suppression
    /// stays off.
    pub fn printedFacts(self: *const ChatConfig) ?tokenizer_mod.TemplatePrinting {
        if (self.chat_template.len == 0) return null;
        const turns = self.compiled.?.turns;
        return .{ .facts = self.facts(), .turns = turns.turns, .outside = turns.outside };
    }

    /// Frees what `compile` built, for a config built by hand whose other fields it does not own.
    pub fn dropCompiled(self: *ChatConfig) void {
        if (self.compiled) |c| c.deinit(self.allocator);
        self.compiled = null;
    }

    pub fn deinit(self: *ChatConfig) void {
        if (self.compiled) |c| c.deinit(self.allocator);
        self.allocator.free(self.chat_template);
        if (self.bos_token) |t| self.allocator.free(t);
        if (self.eos_token) |t| self.allocator.free(t);
        if (self.model_type) |t| self.allocator.free(t);
    }
};

/// A template ready to render, and the text it came from (the engine spy names a render by its text). A template
/// that does not compile keeps why, and every render of it fails with that.
const Tpl = struct {
    text: []const u8,
    tree: union(enum) { ok: template_engine.Template, bad: EngineFailure },
    at_load: bool,
    memo: *ProbeMemo,

    fn init(allocator: std.mem.Allocator, text: []const u8, at_load: bool) !Tpl {
        const memo = try allocator.create(ProbeMemo);
        errdefer allocator.destroy(memo);
        memo.* = .{ .gpa = allocator };
        var failure: EngineFailure = .{};
        const tree = template_engine.Template.compile(allocator, text, &failure.diag) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                failure.err = e;
                return .{ .text = text, .tree = .{ .bad = failure }, .at_load = at_load, .memo = memo };
            },
        };
        return .{ .text = text, .tree = .{ .ok = tree }, .at_load = at_load, .memo = memo };
    }

    fn deinit(self: *Tpl) void {
        if (self.tree == .ok) self.tree.ok.deinit();
        const gpa = self.memo.gpa;
        self.memo.deinit();
        gpa.destroy(self.memo);
    }
};

/// The probes a request may run, each a fixed conversation rendered without a generation prompt.
const Probe = enum {
    tool_content,
    late_system,
    call,
    call_empty,
    answer,

    const tool_marker = "__mlx_tool_role_probe__";
    const answer_marker = "__sushi_answer_probe__";
    const late_marker = "__sushi_late_system_probe__";
    const call_tools = [_]ToolCall{.{ .id = "call_probe", .name = probe_name, .arguments = "{\"a\": \"v\"}" }};
    const call_empty_tools = [_]ToolCall{.{ .id = "call_probe", .name = probe_name, .arguments = "{}" }};

    fn messages(self: Probe) []const Message {
        return switch (self) {
            .tool_content => &.{.{ .role = "tool", .content = tool_marker }},
            .late_system => &.{ .{ .role = "user", .content = "u" }, .{ .role = "system", .content = late_marker }, .{ .role = "user", .content = "u" } },
            .call => &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "", .tool_calls = &call_tools, .reasoning_content = "r" } },
            .call_empty => &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "", .tool_calls = &call_empty_tools, .reasoning_content = "r" } },
            .answer => &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = answer_marker, .reasoning_content = "r" } },
        };
    }
};

/// One compiled template's probe renders, by probe and settings (`Settings.key`): nothing else moves a probe, so a
/// model renders each once. Bounded: a full memo starts over, so client-chosen effort words cannot grow it.
const ProbeMemo = struct {
    const cap = 256;
    const Entry = struct { probe: Probe, extra: []u8, rendered: ?[]u8 };
    gpa: std.mem.Allocator,
    mu: std.Io.Mutex = .init,
    entries: std.ArrayList(Entry) = .empty,

    fn lock(self: *ProbeMemo) std.Io {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        return io;
    }

    /// A copy of the memoized render (null inside when the probe failed), or null when not memoized.
    fn get(self: *ProbeMemo, allocator: std.mem.Allocator, probe: Probe, extra: []const u8) !??[]u8 {
        const io = self.lock();
        defer self.mu.unlock(io);
        for (self.entries.items) |e| if (e.probe == probe and std.mem.eql(u8, e.extra, extra)) {
            const r = e.rendered orelse return @as(?[]u8, null);
            return try allocator.dupe(u8, r);
        };
        return null;
    }

    fn put(self: *ProbeMemo, probe: Probe, extra: []const u8, rendered: ?[]const u8) !void {
        const io = self.lock();
        defer self.mu.unlock(io);
        for (self.entries.items) |e| if (e.probe == probe and std.mem.eql(u8, e.extra, extra)) return;
        if (self.entries.items.len == cap) self.clear();
        const owned_extra = try self.gpa.dupe(u8, extra);
        errdefer self.gpa.free(owned_extra);
        const owned = if (rendered) |r| try self.gpa.dupe(u8, r) else null;
        errdefer if (owned) |r| self.gpa.free(r);
        try self.entries.append(self.gpa, .{ .probe = probe, .extra = owned_extra, .rendered = owned });
    }

    fn clear(self: *ProbeMemo) void {
        for (self.entries.items) |e| {
            self.gpa.free(e.extra);
            if (e.rendered) |r| self.gpa.free(r);
        }
        self.entries.clearRetainingCapacity();
    }

    fn deinit(self: *ProbeMemo) void {
        self.clear();
        self.entries.deinit(self.gpa);
    }
};

/// A chat template compiled, with the late-system form its arch edits in (`arch_chat.lateSystemTemplate`).
pub const Compiled = struct {
    main: Tpl,
    late_system: ?Tpl,
    late_text: ?[]const u8,
    /// `assistantTurnRenders` of a load's compile, in `turns_arena`; empty for a compile no config keeps.
    turns: TurnRenders = .{ .turns = &.{}, .outside = &.{} },
    turns_arena: ?std.heap.ArenaAllocator = null,

    fn init(allocator: std.mem.Allocator, text: []const u8, at_load: bool, model_type: ?[]const u8) !*Compiled {
        const c = try allocator.create(Compiled);
        errdefer allocator.destroy(c);
        c.main = try Tpl.init(allocator, text, at_load);
        errdefer c.main.deinit();
        c.late_text = if (model_type) |m| try arch_chat.lateSystemTemplate(m, allocator, text) else null;
        errdefer if (c.late_text) |t| allocator.free(t);
        c.late_system = if (c.late_text) |t| try Tpl.init(allocator, t, at_load) else null;
        return c;
    }

    /// Whether this is `text`'s compile: the same bytes, not a copy, so a template swapped in later is not.
    fn compiledFrom(self: *const Compiled, text: []const u8) bool {
        return self.main.text.ptr == text.ptr and self.main.text.len == text.len;
    }

    fn deinit(self: *Compiled, allocator: std.mem.Allocator) void {
        if (self.turns_arena) |*a| a.deinit();
        self.main.deinit();
        if (self.late_system) |*l| l.deinit();
        if (self.late_text) |t| allocator.free(t);
        allocator.destroy(self);
    }

    fn facts(self: *const Compiled) *const Facts {
        return if (self.main.tree == .ok) &self.main.tree.ok.facts else &no_facts;
    }

    /// Why the template or its late-system form does not compile; null when both do.
    fn failure(self: *const Compiled) ?*const EngineFailure {
        if (self.main.tree == .bad) return &self.main.tree.bad;
        if (self.late_system) |*l| if (l.tree == .bad) return &l.tree.bad;
        return null;
    }
};

pub const Facts = template_engine.Facts;
const no_facts: Facts = .{};

/// What the server asks of a chat template, answered from its facts, never by searching its text: a word in a
/// comment or an unrelated literal is not use.
pub const asks = struct {
    /// It reads `preserve_thinking`: only then is the switch handed over.
    pub fn preserveThinking(f: *const Facts) bool {
        return f.reads("preserve_thinking");
    }
    /// It emits a think closer: the `think_penalty` needs one to arm.
    pub fn thinkSpan(f: *const Facts) bool {
        return f.emits("</think>");
    }
    /// It reads `tools`, so it renders the tool list itself.
    pub fn tools(f: *const Facts) bool {
        return f.reads("tools");
    }
    /// It compares a role with 'tool': an explicit tool branch, so no probe is needed.
    pub fn toolRole(f: *const Facts) bool {
        return f.compares("tool");
    }
    /// It reads `reasoning_effort`: the word is a lever on this checkpoint, so the server derives no budget from it.
    pub fn effort(f: *const Facts) bool {
        return f.reads("reasoning_effort");
    }
    /// Qwen3.8's effort ladder: the literal 'xhigh' among its accepted words.
    pub fn effortLadder(f: *const Facts) bool {
        return f.hasLiteral("xhigh");
    }
    /// DeepSeek-V4.1's encoder: it reads `thinking_mode`.
    pub fn thinkingMode(f: *const Facts) bool {
        return f.reads("thinking_mode");
    }
    /// GLM's effort words: it reads `effective_reasoning_effort` and opens with `[gMASK]<sop>`.
    pub fn effortWords(f: *const Facts) bool {
        return f.reads("effective_reasoning_effort") and f.emits("[gMASK]<sop>");
    }
    /// It reads `enable_thinking`.
    pub fn enableThinking(f: *const Facts) bool {
        return f.reads("enable_thinking");
    }
    /// The one think opener its printed literals spell, each in full: a piece that starts a tag it does not finish
    /// (`'<think' ~ '>'`) or a second spelling leaves none.
    pub fn thinkOpener(f: *const Facts) ?[]const u8 {
        var found: ?[]const u8 = null;
        var it = f.pieces();
        while (it.next()) |piece| {
            var offset: usize = 0;
            while (std.mem.indexOfPos(u8, piece, offset, "<think")) |start| {
                const len = think_tags.thinkOpenTagLenAt(piece[start..]) orelse return null;
                const tag = piece[start..][0..len];
                if (found) |previous| if (!std.mem.eql(u8, previous, tag)) return null;
                found = tag;
                offset = start + len;
            }
        }
        return found;
    }
    /// It prints Gemma 4's `<|channel>`.
    pub fn channels(f: *const Facts) bool {
        return f.emits("<|channel>");
    }
    /// It reads `enable_thinking` or prints a think opener: the model can produce reasoning_content.
    pub fn reasoning(f: *const Facts) bool {
        return enableThinking(f) or f.emits("<think>") or f.emits("thought") or f.emits("<|channel>");
    }
};

/// The compiled templates of `chat_config`, or for a config built by hand, compiled now into `own`, which the caller
/// frees with `Compiled.deinit`.
fn templatesOf(allocator: std.mem.Allocator, chat_config: *const ChatConfig, own: *?*Compiled) !*const Compiled {
    if (chat_config.compiled) |c| if (c.compiledFrom(chat_config.chat_template)) return c;
    own.* = try Compiled.init(allocator, chat_config.chat_template, false, chat_config.model_type);
    return own.*.?;
}

/// Is this `chat_template` entry just an `{% include %}` POINTER at the
/// sidecar file? transformers >= 5 saves the real template to
/// `chat_template.jinja` and writes that one-liner into tokenizer_config.json
/// (mlx-community oQ4e / oQ5e builds, issue #169). The template engine has
/// no `include` statement, so the pointer taken literally would fail the
/// template's compile.
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
        // oQ4e builds, issue #169). The engine has no `include`, so the
        // pointer taken literally would fail the template's compile.
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
/// The chat template of `model_dir`, read for a look (the settings page) and never logged; `loadChatConfig` loads one.
pub fn peekChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, model_type: ?[]const u8) !ChatConfig {
    return readChatConfig(io, allocator, model_dir, model_type, false);
}

pub fn loadChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, model_type: ?[]const u8) !ChatConfig {
    return readChatConfig(io, allocator, model_dir, model_type, true);
}

/// A key of `tokenizer_config.json` in a shape the load cannot read: named in a `[load]` line when `say`.
fn badConfigKey(say: bool, key: []const u8, want: []const u8) error{InvalidTokenizerConfig} {
    if (say) log.err("[load] tokenizer_config.json: {s} is not {s}\n", .{ key, want });
    return error.InvalidTokenizerConfig;
}

/// `bos_token` or `eos_token`: a string, HF's AddedToken object (its `content`), or null for none. Caller frees.
fn specialTokenText(allocator: std.mem.Allocator, v: ?std.json.Value, key: []const u8, say: bool) !?[]const u8 {
    const t = switch (v orelse return null) {
        .null => return null,
        .string => |text| text,
        .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else return badConfigKey(say, key, "a string or an AddedToken")) else return badConfigKey(say, key, "a string or an AddedToken"),
        else => return badConfigKey(say, key, "a string or an AddedToken"),
    };
    return try allocator.dupe(u8, t);
}

/// The pack's `chat_template.jinja`, or null when it has none. Caller frees.
fn chatTemplateFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !?[]u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}/chat_template.jinja", .{model_dir});
    defer allocator.free(path);
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer f.close(io);
    var buf: [4096]u8 = undefined;
    var reader = f.reader(io, &buf);
    return try reader.interface.allocRemaining(allocator, .limited(1 * 1024 * 1024));
}

/// `model_type` is the one the load parsed from config.json: it picks the arch's pinned template and its edits.
/// `say`: a load, whose template failure and fold the log states.
fn readChatConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, model_type: ?[]const u8, say: bool) !ChatConfig {
    var config: ChatConfig = .{ .chat_template = &.{}, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    errdefer config.deinit();

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
    if (parsed.value != .object) return badConfigKey(say, "the root", "an object");
    const root = parsed.value.object;

    if (model_type) |m| config.model_type = try allocator.dupe(u8, m);
    const pin = if (model_type) |m| arch_chat.pinOf(m) else null;
    config.image_placeholder = if (pin) |p| p.image_placeholder else null;
    const own: ?[]const u8 = switch (root.get("chat_template") orelse .null) {
        .null, .string, .array => chatTemplateFromValue(root.get("chat_template")),
        else => return badConfigKey(say, "chat_template", "a string or a list of named templates"),
    };
    // As transformers loads a pack: its `chat_template.jinja` outranks the copy inside tokenizer_config.json.
    config.chat_template = if (pin) |p|
        try allocator.dupe(u8, p.template)
    else if (try chatTemplateFile(io, allocator, model_dir)) |file_text|
        file_text
    else if (own) |t|
        try allocator.dupe(u8, t)
    else
        &.{};
    config.bos_token = try specialTokenText(allocator, root.get("bos_token"), "bos_token", say);
    config.eos_token = try specialTokenText(allocator, root.get("eos_token"), "eos_token", say);
    config.add_bos_token = switch (root.get("add_bos_token") orelse .null) {
        .null => false,
        .bool => |b| b,
        else => return badConfigKey(say, "add_bos_token", "a bool"),
    };

    try config.compileSaying(say);
    if (!say) return config;
    if (model_type) |m| if (config.compiled) |c| if (c.late_system == null and arch_chat.editsLateSystem(m))
        log.info("[load] chat template: no late-system branch to edit; a system turn after the first folds into the first\n", .{});
    return config;
}

/// Format chat messages into token IDs using the model's Jinja chat template.
/// `effort` is the client's raw `reasoning_effort` string (see
/// `templateSettings`) — null when the request didn't send one.
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
    const rendered = try renderChatSpans(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    defer freeSpans(allocator, rendered);

    var ids = std.ArrayList(u32).empty;
    errdefer ids.deinit(allocator);

    if (chat_config.add_bos_token) {
        if (tok.bos_id) |bos| {
            try ids.append(allocator, bos);
        }
    }

    const body = try tok.encodeSpans(allocator, rendered);
    defer allocator.free(body);
    try ids.appendSlice(allocator, body);
    log.debug("  prompt: {d} chars -> {d} tokens\n", .{ rendered.bytes.len, ids.items.len });

    return ids.toOwnedSlice(allocator);
}

/// Renders whose shipped template raised and fell to the generic format; `/props` reports it.
pub var template_fallbacks = std.atomic.Value(u64).init(0);

/// Render the Jinja chat template with the given messages: `renderChatSpans`'s text.
pub fn renderChatTemplate(
    allocator: std.mem.Allocator,
    messages: []const Message,
    chat_config: *const ChatConfig,
    tools_json: ?[]const u8,
    tool_choice_instruction: ?[]const u8,
    enable_thinking: bool,
    effort: ?[]const u8,
    continue_final: bool,
) ![]const u8 {
    const rendered = try renderChatSpans(allocator, messages, chat_config, tools_json, tool_choice_instruction, enable_thinking, effort, continue_final);
    allocator.free(rendered.runs);
    return rendered.bytes;
}

/// A rendered prompt and who wrote each byte, every piece owned by one allocator (`freeSpans`).
pub const Spans = template_engine.Str;

pub fn freeSpans(allocator: std.mem.Allocator, s: Spans) void {
    allocator.free(s.bytes);
    allocator.free(s.runs);
}

fn dupeSpans(allocator: std.mem.Allocator, s: Spans) !Spans {
    const bytes = try allocator.dupe(u8, s.bytes);
    errdefer allocator.free(bytes);
    return .{ .bytes = bytes, .origin = s.origin, .runs = try allocator.dupe(template_engine.Run, s.runs) };
}

/// What `b` built, owned: the builder is spent.
fn ownBuilt(allocator: std.mem.Allocator, b: *template_engine.Builder) !Spans {
    const shape = b.str();
    if (shape.runs.len == 0) b.runs.clearAndFree(allocator);
    const runs = try b.runs.toOwnedSlice(allocator);
    errdefer allocator.free(runs);
    return .{ .bytes = try b.bytes.toOwnedSlice(allocator), .origin = shape.origin, .runs = runs };
}

fn joinSpans(allocator: std.mem.Allocator, parts: []const Spans) !Spans {
    var b: template_engine.Builder = .{};
    errdefer b.deinit(allocator);
    for (parts) |p| try b.append(allocator, p);
    return ownBuilt(allocator, &b);
}

/// The server's own text, read as the template's.
fn server(text: []const u8) Spans {
    return .{ .bytes = text };
}

/// Render the Jinja chat template with the given messages, each byte with who wrote it: the template (and the server's
/// own text around it), an assistant message, or other request text (`Tokenizer.encodeSpans`).
pub fn renderChatSpans(
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
) !Spans {
    // Split the partial reply off FIRST, so every transform below (tool
    // fallback synthesis) sees exactly the message list
    // an ordinary turn would have. The template must open the assistant turn
    // once, which it only does when the trailing assistant is not in `messages`.
    // The partial is joined to the render as it is, so it is mended once here, as the engine mends what it reads.
    var mend = std.heap.ArenaAllocator.init(allocator);
    defer mend.deinit();
    const prefill = if (think_tags.continuationPrefill(messages, continue_final)) |p| try utf8Text(mend.allocator(), p) else null;
    const msgs = if (prefill != null) messages[0 .. messages.len - 1] else messages;

    if (chat_config.chat_template.len == 0) {
        if (!fallbackIsChatMl(chat_config)) return refuseTemplate("the model has no chat template and no fallback format");
        const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
        if (prefill) |partial| {
            defer freeSpans(allocator, base);
            return joinSpans(allocator, &.{ base, .{ .bytes = partial, .origin = .assistant } });
        }
        return base;
    }

    // Some chat templates (e.g. DeepSeek V4) don't reference `tools` or `role == 'tool'`
    // at all — Jinja silently drops the tool definitions, and tool-result messages render
    // as empty. Detect that gap and synthesize an equivalent system-prompt + user-message
    // form so the model still sees the tool context.
    // A Qwen template whose late-system branch its chat face edits renders the note in place; the fold below would
    // rewrite every earlier byte of the prompt. Only the render and its probe see the adapted text.
    var own: ?*Compiled = null;
    defer if (own) |c| c.deinit(allocator);
    const compiled = try templatesOf(allocator, chat_config, &own);
    const main_tpl = &compiled.main;
    const render_tpl: *const Tpl = if (hasLateSystem(msgs)) if (compiled.late_system) |*l| l else main_tpl else main_tpl;
    const tpl_has_tools = asks.tools(compiled.facts());
    const messages_have_tool_content = messagesHaveToolContent(msgs);
    const settings = try templateSettings(allocator, chat_config, enable_thinking, effort);
    switch (settings.reasoning_effort) {
        .word => |w| log.debug("[chat] template kwargs: enable_thinking={} reasoning_effort={s} thinking_mode={?s} preserve_thinking={?}\n", .{ enable_thinking, w, settings.thinking_mode, settings.preserve_thinking }),
        .budget => |n| log.debug("[chat] template kwargs: enable_thinking={} reasoning_effort={d} thinking_mode={?s} preserve_thinking={?}\n", .{ enable_thinking, n, settings.thinking_mode, settings.preserve_thinking }),
    }

    // A generic role header is not proof that the template emits every role:
    // some templates guard the header to user/assistant/system. Probe only
    // tool-bearing requests; explicit tool branches remain the zero-cost path.
    const tpl_has_tool_role = !messages_have_tool_content or
        asks.toolRole(compiled.facts()) or
        try templateProbePreservesToolContent(allocator, main_tpl, &settings);
    const needs_inject_tools = tools_json != null and !tpl_has_tools;
    const needs_rewrite_tool_role = !tpl_has_tool_role and messages_have_tool_content;

    var fallback_arena: ?std.heap.ArenaAllocator = null;
    defer if (fallback_arena) |*a| a.deinit();
    var effective_messages = msgs;
    // A template that refuses a system turn past index 0 (Qwen's raises) gets it folded
    // into the leading one; a template that renders it in place (MiMo's) keeps it there.
    if (hasLateSystem(msgs) and !try templateProbeRendersLateSystem(allocator, render_tpl, &settings)) {
        fallback_arena = std.heap.ArenaAllocator.init(allocator);
        const arena_alloc = fallback_arena.?.allocator();
        var folded = try std.ArrayList(Message).initCapacity(arena_alloc, msgs.len);
        folded.appendSliceAssumeCapacity(msgs);
        _ = try foldSystemMessages(arena_alloc, &folded);
        effective_messages = folded.items;
    }
    // The render fills the optional keys a template may dump blindly (`fillToolDefKeys`).
    var effective_tools_json = tools_json;
    if (needs_inject_tools or needs_rewrite_tool_role) {
        // NEVER re-init: the fold above may already own this arena, and clobbering it orphans what it holds.
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
    // It stays only where the render keeps every byte it has without it (Qwen3.8 with preserve_thinking off drops the
    // in-flight turn's reasoning once a user turn follows); else the decode-time CallForce forces the call alone.
    var unforced: ?[]const Message = null;
    if (tool_choice_instruction) |instr| {
        if (tools_json != null and !needs_inject_tools and prefill == null) {
            if (fallback_arena == null) fallback_arena = std.heap.ArenaAllocator.init(allocator);
            unforced = effective_messages;
            effective_messages = try appendTurnInstruction(fallback_arena.?.allocator(), effective_messages, std.mem.trim(u8, instr, "\n"));
        }
    }

    if (prefill) |partial| {
        const turn = messages[messages.len - 1];
        if (try renderContinuation(allocator, render_tpl, &settings, effective_tools_json, effective_messages, turn, partial, chat_config)) |cont| return cont;
    }

    var failure: EngineFailure = .{};
    const context = RenderContext{ .messages = effective_messages, .image_atom = chat_config.image_placeholder, .tools = effective_tools_json, .settings = &settings };
    if (try renderEngine(allocator, render_tpl, &context, true, &failure)) |first_out| {
        var engine_out = first_out;
        defer freeSpans(allocator, engine_out);
        if (unforced) |plain_messages| {
            var plain_failure: EngineFailure = .{};
            const plain_context = RenderContext{ .messages = plain_messages, .image_atom = chat_config.image_placeholder, .tools = effective_tools_json, .settings = &settings };
            if (try renderEngine(allocator, render_tpl, &plain_context, true, &plain_failure)) |plain| {
                if (keepsEveryByte(plain.bytes, engine_out.bytes)) freeSpans(allocator, plain) else {
                    freeSpans(allocator, engine_out);
                    engine_out = plain;
                }
            }
        }
        const collapsed = try collapseDoubledThinkTags(allocator, engine_out);
        // A continuation commits the content channel and then hands the model
        // its own unfinished sentence. It runs INSTEAD of the thinking-off
        // tail, never beside it — both append the same channel commit, and
        // twice is a malformed turn header.
        if (prefill) |partial| {
            defer freeSpans(allocator, collapsed);
            return joinSpans(allocator, &.{
                collapsed, server(channelTail(collapsed)), .{ .bytes = partial, .origin = .assistant },
            });
        }
        if (think_tags.noThinkTailSuffix(collapsed.bytes, enable_thinking)) |suffix| if (channelTail(collapsed).len > 0) {
            defer freeSpans(allocator, collapsed);
            return joinSpans(allocator, &.{ collapsed, server(suffix) });
        };
        return collapsed;
    }

    // WARN, not debug: a failed render silently swaps a ChatML model's prompt into
    // fallbackFormatChat's format, so the downgrade must be visible at the default
    // log level. Any other model has no fallback: its turn tokens are not ChatML's.
    var reason_buf: [1200]u8 = undefined;
    const reason = failure.describe(&reason_buf);
    if (!fallbackIsChatMl(chat_config)) {
        log.warn("jinja render failed ({s}), and this model has no fallback chat format\n", .{reason});
        return refuseTemplate(reason);
    }
    // The template says the conversation has nothing to answer: a request defect, not a render fault.
    if (failure.err == error.TemplateRaised and failure.diag.messageIs(NO_USER_QUERY_RAISE)) return error.ChatNeedsUserTurn;
    _ = template_fallbacks.fetchAdd(1, .monotonic);
    log.warn("jinja render failed ({s}), falling back to generic chat format\n", .{reason});
    const base = try fallbackFormatChat(allocator, msgs, chat_config, tools_json, tool_choice_instruction, enable_thinking);
    if (prefill) |partial| {
        defer freeSpans(allocator, base);
        return joinSpans(allocator, &.{ base, .{ .bytes = partial, .origin = .assistant } });
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
    tmpl: *const Tpl,
    settings: *const Settings,
    tools: ?[]const u8,
    history: []const Message,
    turn: Message,
    partial: []const u8,
    chat_config: *const ChatConfig,
) !?Spans {
    const turns = try allocator.alloc(Message, history.len + 1);
    defer allocator.free(turns);
    @memcpy(turns[0..history.len], history);
    turns[history.len] = .{ .role = "assistant", .content = partial, .reasoning_content = turn.reasoning_content };
    var failure: EngineFailure = .{};
    const context = RenderContext{ .messages = turns, .image_atom = chat_config.image_placeholder, .tools = tools, .settings = settings };
    const engine_out = try renderEngine(allocator, tmpl, &context, false, &failure) orelse return null;
    defer freeSpans(allocator, engine_out);
    const rendered = try collapseDoubledThinkTags(allocator, engine_out);
    defer freeSpans(allocator, rendered);
    // A template may trim the text's leading whitespace, as transformers allows for. The cut is where the partial
    // turn's own bytes end, and they must end with its text: a search could land inside an earlier turn.
    const text = std.mem.trimStart(u8, partial, " \t\r\n");
    const end = rendered.lastEnd(.assistant) orelse return null;
    if (end < text.len or !std.mem.eql(u8, rendered.bytes[end - text.len .. end], text)) return null;
    if (rendered.originOf(end - text.len, end) != .assistant) return null;
    var cut = std.heap.ArenaAllocator.init(allocator);
    defer cut.deinit();
    return try dupeSpans(allocator, try rendered.slice(cut.allocator(), 0, end));
}

/// `think_tags.contentChannelTail` for a think opener the template wrote; one a request's text spells is left open.
fn channelTail(rendered: Spans) []const u8 {
    const at = think_tags.tailThinkOpener(rendered.bytes) orelse return "";
    if (rendered.originOf(at.start, at.end) != .template) return "";
    return think_tags.contentChannelTail(rendered.bytes);
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
fn collapseDoubledThinkTags(allocator: std.mem.Allocator, rendered: Spans) !Spans {
    const tag = "</think>";
    const needle = "</think></think>";
    const text = rendered.bytes;
    if (std.mem.indexOf(u8, text, needle) == null) return dupeSpans(allocator, rendered);
    var out: template_engine.Builder = .{};
    errdefer out.deinit(allocator);
    try out.bytes.ensureTotalCapacity(allocator, text.len);
    // At each `</think></think>` drop the FIRST tag and look again from the survivor, so 3x, 4x, … also collapse
    // to one; every kept byte keeps its origin.
    var keep_from: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], needle) and rendered.originOf(i, i + needle.len) == .template) {
            try out.appendPart(allocator, rendered, keep_from, i);
            i += tag.len;
            keep_from = i;
        } else i += 1;
    }
    try out.appendPart(allocator, rendered, keep_from, text.len);
    return ownBuilt(allocator, &out);
}

/// Execute a tiny role probe instead of trying to infer Jinja control flow
/// from source spelling. A missing marker means the template filtered the tool
/// message, so the caller rewrites it through the lossless tool fallback.
fn templateProbePreservesToolContent(
    allocator: std.mem.Allocator,
    tpl: *const Tpl,
    settings: *const Settings,
) !bool {
    return templateProbeRendersMarker(allocator, tpl, .tool_content, settings, Probe.tool_marker);
}

fn hasLateSystem(messages: []const Message) bool {
    if (messages.len < 2) return false;
    for (messages[1..]) |m| {
        if (std.mem.eql(u8, m.role, "system")) return true;
    }
    return false;
}

/// False when the template raises on a system turn that is not first, or drops it.
fn templateProbeRendersLateSystem(
    allocator: std.mem.Allocator,
    tpl: *const Tpl,
    settings: *const Settings,
) !bool {
    return templateProbeRendersMarker(allocator, tpl, .late_system, settings, Probe.late_marker);
}

fn templateProbeRendersMarker(
    allocator: std.mem.Allocator,
    tpl: *const Tpl,
    probe: Probe,
    settings: *const Settings,
    marker: []const u8,
) !bool {
    const rendered = (try templateProbeRender(allocator, tpl, probe, settings)) orelse return false;
    defer allocator.free(rendered);
    return std.mem.indexOf(u8, rendered, marker) != null;
}

/// The template's render of `probe`, or null when it fails, memoized on the template. Probe failures are
/// conservative: the real render either renders or logs its own failure.
fn templateProbeRender(allocator: std.mem.Allocator, tpl: *const Tpl, probe: Probe, settings: *const Settings) !?[]u8 {
    const key = try settings.key(allocator);
    defer allocator.free(key);
    if (try tpl.memo.get(allocator, probe, key)) |memoized| return memoized;
    const rendered = try renderProbeFresh(allocator, tpl, probe, settings);
    errdefer if (rendered) |r| allocator.free(r);
    try tpl.memo.put(probe, key, rendered);
    return rendered;
}

fn renderProbeFresh(allocator: std.mem.Allocator, tpl: *const Tpl, probe: Probe, settings: *const Settings) !?[]u8 {
    var failure: EngineFailure = .{};
    const context = RenderContext{ .messages = probe.messages(), .settings = settings };
    const rendered = try renderEngine(allocator, tpl, &context, false, &failure) orelse return null;
    allocator.free(rendered.runs);
    return @constCast(rendered.bytes);
}

/// Tests only: the JSON writer of a template's context (`chat_json_test.zig`), never compiled into the served binary.
const test_json = if (builtin.is_test) @import("chat_json_test.zig") else struct {};

const ContextTexts = struct { messages: []const u8, tools: ?[]const u8, extra: []const u8 };

/// A render's context as the JSON text the server once handed the engine: the oracle keys its cases by it.
fn contextTexts(arena: std.mem.Allocator, context: *const RenderContext) !ContextTexts {
    var tools = context.tools;
    if (context.tools) |t| if (t.len > 0) {
        var diag: template_engine.Diagnostic = .{};
        if (template_engine.readJson(arena, t, "tools", .input, &diag)) |v| {
            if (try fillToolDefKeys(arena, v, context.facts)) {
                var buf = std.ArrayList(u8).empty;
                try test_json.writeValue(arena, &buf, v);
                tools = buf.items;
            }
        } else |_| {}
    };
    return .{
        .messages = try test_json.alloc(arena, try templateMessages(arena, context.messages, context.image_atom)),
        .tools = tools,
        .extra = try test_json.alloc(arena, context.settings.*),
    };
}

/// Tests only: `messages` as the JSON text a template's context once held, owned by `allocator`.
pub fn messagesJsonForTest(allocator: std.mem.Allocator, messages: []const Message, image_atom: ?[]const u8) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    return allocator.dupe(u8, try test_json.alloc(arena.allocator(), try templateMessages(arena.allocator(), messages, image_atom)));
}

/// Tests only: a template's settings as the JSON text its context once held, owned by `allocator`.
fn extraJsonForTest(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool, effort: ?[]const u8) ![]u8 {
    const settings = try templateSettings(allocator, chat_config, enable_thinking, effort);
    return test_json.alloc(allocator, settings);
}

/// One call into the template engine, its inputs as the JSON text they once were (tests only, `engine_spy`).
pub const EngineCall = struct {
    template: []const u8,
    messages: []const u8,
    tools: ?[]const u8,
    extra: []const u8,
    add_generation_prompt: bool,
    /// Null when the engine raised.
    rendered: ?[]const u8,
    /// The work the render took against the engine's limits.
    usage: template_engine.Usage,
    /// The template was compiled when its model loaded, not for this render.
    compiled_at_load: bool,
};

pub const EngineSpy = struct { ctx: *anyopaque, see: *const fn (ctx: *anyopaque, call: EngineCall) void };

/// Read only by a test build: the test that sets it sees every call `renderEngine` makes.
pub var engine_spy: ?EngineSpy = null;

/// Why a render failed: the engine's error and its diagnostic (for a raise, the template's own text).
const EngineFailure = struct {
    err: template_engine.Error = error.TemplateRuntime,
    diag: template_engine.Diagnostic = .{},

    /// `line L column C: message`, or the message alone when it has no place in the template.
    fn describe(self: *const EngineFailure, buf: []u8) []const u8 {
        if (self.diag.line == 0) return std.fmt.bufPrint(buf, "{s}", .{self.diag.message()}) catch self.diag.message();
        return std.fmt.bufPrint(buf, "line {d} column {d}: {s}", .{ self.diag.line, self.diag.column, self.diag.message() }) catch self.diag.message();
    }
};

/// The only call into the template engine: the render, owned by `allocator`, or null with why in `failure`.
fn renderEngine(allocator: std.mem.Allocator, tmpl: *const Tpl, context: *const RenderContext, add_generation_prompt: bool, failure: *EngineFailure) error{OutOfMemory}!?Spans {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var usage: template_engine.Usage = .{};
    var ctx = context.*;
    if (tmpl.tree == .ok) ctx.facts = &tmpl.tree.ok.facts;
    const in: template_engine.Inputs = .{ .context = ctx.context(), .add_generation_prompt = add_generation_prompt, .usage = if (builtin.is_test) &usage else null };
    const rendered: ?Spans = switch (tmpl.tree) {
        .bad => |b| blk: {
            failure.* = b;
            break :blk null;
        },
        .ok => |*t| if (t.renderSpans(arena.allocator(), in, &failure.diag)) |out| try dupeSpans(allocator, out) else |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => blk: {
                failure.err = e;
                break :blk null;
            },
        },
    };
    if (builtin.is_test) if (engine_spy) |spy| {
        var json = std.heap.ArenaAllocator.init(allocator);
        defer json.deinit();
        const texts = contextTexts(json.allocator(), &ctx) catch |e| switch (e) {
            error.OutOfMemory => {
                if (rendered) |r| freeSpans(allocator, r);
                return error.OutOfMemory;
            },
            else => std.debug.panic("engine spy: the context is not writable as JSON ({s})", .{@errorName(e)}),
        };
        spy.see(spy.ctx, .{
            .template = tmpl.text,
            .compiled_at_load = tmpl.at_load,
            .messages = texts.messages,
            .tools = texts.tools,
            .extra = texts.extra,
            .add_generation_prompt = add_generation_prompt,
            .rendered = if (rendered) |r| r.bytes else null,
            .usage = usage,
        });
    };
    return rendered;
}

pub const TurnRenders = struct { turns: []const []const u8, outside: []const []const u8 };

/// Assistant turns of every shape a reply takes (an answer, a thought, calls, an answer after a tool result), each
/// last in its conversation.
const turn_probes = [_][]const Message{
    &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "a" } },
    &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "a", .reasoning_content = "r" } },
    &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "", .reasoning_content = "r", .tool_calls = &Probe.call_tools } },
    &.{ .{ .role = "user", .content = "u" }, .{ .role = "assistant", .content = "a", .reasoning_content = "r", .tool_calls = &Probe.call_tools } },
    &.{
        .{ .role = "system", .content = "s" },
        .{ .role = "user", .content = "u" },
        .{ .role = "assistant", .content = "", .reasoning_content = "r", .tool_calls = &Probe.call_tools },
        .{ .role = "tool", .content = "t", .tool_call_id = "call_probe" },
        .{ .role = "assistant", .content = "a", .reasoning_content = "r" },
    },
};
const turn_probe_tools =
    \\[{"type":"function","function":{"name":"__sushi_forced_call_probe__","description":"d","parameters":{"type":"object","properties":{"a":{"type":"string"}}}}}]
;

/// The template's renders around the last assistant turn of each `turn_probes` conversation, thought on and off, with
/// and without tools: the generation prompt and the rendered turn (from the first byte the conversation before it
/// does not share), and the text before them. A probe the template refuses adds nothing.
fn assistantTurnRenders(arena: std.mem.Allocator, chat_config: *const ChatConfig) !TurnRenders {
    var turns = std.ArrayList([]const u8).empty;
    var outside = std.ArrayList([]const u8).empty;
    const tpl = &chat_config.compiled.?.main;
    for ([_]bool{ true, false }) |thinking| {
        const settings = try templateSettings(arena, chat_config, thinking, null);
        for ([_]?[]const u8{ turn_probe_tools, null }) |tools| for (turn_probes) |msgs| {
            const head_ctx = RenderContext{ .messages = msgs[0 .. msgs.len - 1], .tools = tools, .settings = &settings };
            const full_ctx = RenderContext{ .messages = msgs, .tools = tools, .settings = &settings };
            var failure: EngineFailure = .{};
            const head = (try renderEngine(arena, tpl, &head_ctx, false, &failure)) orelse continue;
            const prompt = (try renderEngine(arena, tpl, &head_ctx, true, &failure)) orelse continue;
            const full = (try renderEngine(arena, tpl, &full_ctx, false, &failure)) orelse continue;
            const gen_at = std.mem.indexOfDiff(u8, head.bytes, prompt.bytes) orelse continue;
            const turn_at = std.mem.indexOfDiff(u8, head.bytes, full.bytes) orelse continue;
            try turns.append(arena, prompt.bytes[gen_at..]);
            try turns.append(arena, full.bytes[turn_at..]);
            try outside.append(arena, full.bytes[0..turn_at]);
        };
    }
    return .{ .turns = turns.items, .outside = outside.items };
}

/// What a forced tool_choice commits once the thought is closed, spelled the
/// way the template renders a call after a closed think block.
pub const ForcedCall = struct {
    /// Whitespace between the think closer and the call.
    sep: []u8,
    /// The call up to the model's first free choice: through the function name
    /// and its closing delimiter when one is named and one delimiter closes it,
    /// else up to the name.
    call: []u8,
    /// What may close a name: the delimiter and the whitespace after it, or,
    /// where the template follows the name with a tag, each tag it may write.
    name_tails: []const []const u8,

    pub fn deinit(self: *ForcedCall, allocator: std.mem.Allocator) void {
        allocator.free(self.sep);
        allocator.free(self.call);
        for (self.name_tails) |t| allocator.free(t);
        allocator.free(self.name_tails);
    }
};

/// Read the forced call from the template's own rendering of an assistant tool
/// call. Null when the template renders no think closer before the call to
/// anchor on, or nothing after the function name to close it.
const probe_name = "__sushi_forced_call_probe__";

/// The template's rendering of a turn whose assistant calls `probe_name` after a closed thought.
fn renderProbeCall(allocator: std.mem.Allocator, chat_config: *const ChatConfig, probe: Probe) !?[]u8 {
    const settings = try templateSettings(allocator, chat_config, true, null);
    var own: ?*Compiled = null;
    defer if (own) |c| c.deinit(allocator);
    return templateProbeRender(allocator, &(try templatesOf(allocator, chat_config, &own)).main, probe, &settings);
}

/// Where the probe's call markup sits: `between` runs from the think closer to the name.
const ProbeCall = struct { between: []const u8, head: []const u8, after: []const u8 };

fn locateProbeCall(rendered: []const u8) ?ProbeCall {
    const name_at = std.mem.indexOf(u8, rendered, probe_name) orelse return null;
    const closer_at = std.mem.lastIndexOf(u8, rendered[0..name_at], think_tags.BARE_THINK_CLOSER) orelse return null;
    const between = rendered[closer_at + think_tags.BARE_THINK_CLOSER.len .. name_at];
    const head = std.mem.trimStart(u8, between, " \t\r\n");
    if (head.len == 0) return null;
    return .{ .between = between, .head = head, .after = rendered[name_at + probe_name.len ..] };
}

/// What the template writes between a thought's closer and the answer after it (`\n\n` on Qwen, none on GLM), or null
/// when it writes no closer before an answer. The caller frees it.
pub fn answerSeparator(allocator: std.mem.Allocator, chat_config: *const ChatConfig) !?[]u8 {
    const rendered = (try renderProbeCall(allocator, chat_config, .answer)) orelse return null;
    defer allocator.free(rendered);
    const at = std.mem.indexOf(u8, rendered, Probe.answer_marker) orelse return null;
    const closer_at = std.mem.lastIndexOf(u8, rendered[0..at], think_tags.BARE_THINK_CLOSER) orelse return null;
    return try allocator.dupe(u8, rendered[closer_at + think_tags.BARE_THINK_CLOSER.len .. at]);
}

/// The `<tag>` the template opens every tool call with (`<tool_call>`), or null when its call
/// markup does not start with one.
pub fn callOpener(allocator: std.mem.Allocator, chat_config: *const ChatConfig) !?[]u8 {
    const rendered = (try renderProbeCall(allocator, chat_config, .call)) orelse return null;
    defer allocator.free(rendered);
    const head = (locateProbeCall(rendered) orelse return null).head;
    if (head[0] != '<') return null;
    const end = std.mem.indexOfScalar(u8, head, '>') orelse return null;
    return try allocator.dupe(u8, head[0 .. end + 1]);
}

/// The call dialect the template teaches, read from its own rendering of a call; null when it is none of the
/// parser's families.
pub fn callDialect(allocator: std.mem.Allocator, chat_config: *const ChatConfig) !?tool_calls_mod.Dialect {
    const rendered = (try renderProbeCall(allocator, chat_config, .call)) orelse return null;
    defer allocator.free(rendered);
    const name_at = std.mem.indexOf(u8, rendered, probe_name) orelse return null;
    const before = rendered[name_at -| 48 .. name_at];
    if (std.mem.indexOf(u8, before, "<function=") != null) return .function_tag;
    if (std.mem.indexOf(u8, before, "DSML") != null) return .dsml;
    if (std.mem.startsWith(u8, rendered[name_at + probe_name.len ..], "<arg_key>")) return .arg_key;
    return null;
}

/// What follows a function name in a render, up to the first byte the model chooses: a delimiter of one to four
/// bytes with the whitespace after it (`>\n`), or, where a tag follows the name, that tag (`<arg_key>`).
fn nameClose(after: []const u8) ?[]const u8 {
    var delim: usize = 0;
    while (delim < after.len and delim < 4 and !nameByte(after[delim]) and !std.ascii.isWhitespace(after[delim]) and after[delim] != '<') delim += 1;
    var at = delim;
    while (at < after.len and std.ascii.isWhitespace(after[at])) at += 1;
    if (delim > 0) return after[0..at];
    if (at >= after.len or after[at] != '<') return null;
    const end = std.mem.indexOfScalar(u8, after[at..], '>') orelse return null;
    return after[0 .. at + end + 1];
}

pub fn forcedCallText(allocator: std.mem.Allocator, chat_config: *const ChatConfig, name: ?[]const u8) !?ForcedCall {
    const rendered = (try renderProbeCall(allocator, chat_config, .call)) orelse return null;
    defer allocator.free(rendered);
    const at = locateProbeCall(rendered) orelse return null;
    const head = at.head;
    const sep = at.between[0 .. at.between.len - head.len];
    const first = nameClose(at.after) orelse return null;

    var tails = std.ArrayList([]const u8).empty;
    errdefer {
        for (tails.items) |t| allocator.free(t);
        tails.deinit(allocator);
    }
    try tails.ensureTotalCapacity(allocator, 2);
    tails.appendAssumeCapacity(try allocator.dupe(u8, first));
    // A tag after the name differs with the arguments: a call with none closes there instead.
    if (std.mem.trimStart(u8, first, " \t\r\n")[0] == '<') {
        if (try renderProbeCall(allocator, chat_config, .call_empty)) |bare| {
            defer allocator.free(bare);
            const closes = if (locateProbeCall(bare)) |b| nameClose(b.after) else null;
            if (closes) |c| if (!std.mem.eql(u8, c, first)) tails.appendAssumeCapacity(try allocator.dupe(u8, c));
        }
    }

    const owned_sep = try allocator.dupe(u8, sep);
    errdefer allocator.free(owned_sep);
    const one = tails.items.len == 1;
    const call = try std.mem.concat(allocator, u8, &.{ head, if (one) name orelse "" else "", if (one and name != null) tails.items[0] else "" });
    errdefer allocator.free(call);
    return .{ .sep = owned_sep, .call = call, .name_tails = try tails.toOwnedSlice(allocator) };
}

fn nameByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.';
}

/// `messages` with `text` closing the conversation: appended to a trailing user
/// turn, else a user turn of its own. Every byte before the last message is
/// unchanged, so the prefix cache still matches.
fn appendTurnInstruction(arena: std.mem.Allocator, messages: []const Message, text: []const u8) ![]const Message {
    // It names the client's function: the client's text.
    const instruction: template_engine.Str = .{ .bytes = text, .origin = .input };
    const out = try arena.alloc(Message, messages.len + 1);
    @memcpy(out[0..messages.len], messages);
    if (messages.len > 0 and std.mem.eql(u8, messages[messages.len - 1].role, "user")) {
        const last = &out[messages.len - 1];
        last.* = try last.derive(arena, "user", &.{}, if (last.content.len == 0) &.{instruction} else &.{ server("\n\n"), instruction });
        return out[0..messages.len];
    }
    out[messages.len] = try synthesized(arena, "user", &.{instruction});
    return out;
}

/// Whether `forced` is `plain` with bytes put in at one place: every byte of `plain` stays, in order.
fn keepsEveryByte(plain: []const u8, forced: []const u8) bool {
    if (forced.len < plain.len) return false;
    const at = std.mem.indexOfDiff(u8, plain, forced) orelse return true;
    return std.mem.endsWith(u8, forced, plain[at..]);
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

    var tool_prompt: ?template_engine.Str = null;
    if (inject_tool_prompt and tools_json != null) {
        var b: template_engine.Builder = .{};
        try appendToolPrompt(arena, &b, tools_json.?, tool_choice_instruction);
        tool_prompt = b.str();
    }

    var injected = false;
    if (tool_prompt) |tp| {
        if (messages.len == 0 or !std.mem.eql(u8, messages[0].role, "system")) {
            try out.append(arena, try synthesized(arena, "system", &.{tp}));
            injected = true;
        }
    }

    for (messages, 0..) |msg, i| {
        if (tool_prompt) |tp| {
            if (!injected and i == 0 and std.mem.eql(u8, msg.role, "system")) {
                try out.append(arena, try msg.derive(arena, "system", &.{}, if (msg.content.len > 0) &.{ server("\n\n"), tp } else &.{tp}));
                injected = true;
                continue;
            }
        }

        if (rewrite_tool_role and std.mem.eql(u8, msg.role, "tool")) {
            try out.append(arena, try msg.derive(arena, "user", &.{server("<tool_response>\n")}, &.{server("\n</tool_response>")}));
            continue;
        }

        if (rewrite_tool_calls and std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            var b: template_engine.Builder = .{};
            for (msg.tool_calls.?, 0..) |tc, n| for ([_]template_engine.Str{
                server(if (n > 0 or msg.content.len > 0) "\n<tool_call>\n{\"name\": \"" else "<tool_call>\n{\"name\": \""),
                .{ .bytes = tc.name, .origin = .assistant },
                server("\", \"arguments\": "),
                .{ .bytes = tc.arguments, .origin = .assistant },
                server("}\n</tool_call>"),
            }) |p| try b.append(arena, p);
            // The calls are inlined as text, so the turn carries them no more.
            var rewritten = try msg.derive(arena, "assistant", &.{}, &.{b.str()});
            rewritten.tool_calls = null;
            try out.append(arena, rewritten);
            continue;
        }

        try out.append(arena, msg);
    }

    return out.toOwnedSlice(arena);
}

/// A message the server writes, its content the pieces in order, each byte with who wrote it.
fn synthesized(arena: std.mem.Allocator, role: []const u8, pieces: []const template_engine.Str) !Message {
    var b: template_engine.Builder = .{};
    for (pieces) |p| try b.append(arena, p);
    return .{ .role = role, .content = b.bytes.items, .content_origins = b.runs.items };
}

/// The tool prompt of the fallbacks: the server's instructions before the client's tool list, then the tool_choice
/// instruction, which names the client's function and so is the client's text.
fn appendToolPrompt(allocator: std.mem.Allocator, b: *template_engine.Builder, tools_json: []const u8, instruction: ?[]const u8) !void {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try tool_calls_mod.appendToolSystemPrompt(allocator, &buf, tools_json, instruction);
    const tail = if (instruction) |t| t.len else 0;
    const head = buf.items.len - tools_json.len - tail;
    try b.appendBytes(allocator, buf.items[0..head], .template);
    try b.appendBytes(allocator, buf.items[head .. head + tools_json.len], .input);
    try b.appendBytes(allocator, buf.items[head + tools_json.len ..], .input);
}

/// A call's text as the model writes it: the template's markup around a function name the client chose, which is
/// text whatever it spells.
pub fn encodeCallText(allocator: std.mem.Allocator, tok: *const Tokenizer, head: []const u8, name: []const u8, tail: []const u8) ![]u32 {
    var b: template_engine.Builder = .{};
    defer b.deinit(allocator);
    try b.appendBytes(allocator, head, .template);
    try b.appendBytes(allocator, name, .input);
    try b.appendBytes(allocator, tail, .template);
    return tok.encodeSpans(allocator, b.str());
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

/// A message as a template reads it, its keys in the order a chat template's JSON has them. `content` is null when
/// there is none, never absent; an absent optional is an absent key.
const TemplateMessage = struct {
    role: []const u8,
    content: Content,
    tool_calls: ?[]const TemplateCall = null,
    tool_call_id: ?[]const u8 = null,
    /// Reasoning a client round-trips on an assistant turn; absent, not null, when there is none: templates gate on
    /// `message.reasoning_content is string`, and a null would still pass an `is defined` check.
    reasoning_content: ?[]const u8 = null,

    const Content = union(enum) { text: template_engine.Str, parts: []const Part, none };
    /// A vision template's part list: text pieces, and a placeholder where each image or video sits.
    const Part = union(enum) { text: struct { type: []const u8 = "text", text: template_engine.Str }, media: struct { type: []const u8 } };
};

const TemplateCall = struct {
    type: []const u8 = "function",
    /// A template that names a tool RESULT by matching `message.tool_call_id` against the call's `id` needs it.
    id: ?[]const u8 = null,
    function: struct { name: []const u8, arguments: Arguments },
};

/// A call's arguments: the JSON object the client sent, kept as text beside the value the engine read from it.
/// Templates iterate `arguments|items`, so anything but an object (empty, null, an array, a scalar, malformed or
/// over-nested text) is `{}`; a repeated key is read as `json.loads` reads it, its last value winning.
const Arguments = struct {
    text: []const u8,
    value: template_engine.Value,

    pub fn templateValue(self: Arguments) template_engine.Value {
        return self.value;
    }

    pub fn templateText(self: Arguments) []const u8 {
        return self.text;
    }

    fn of(arena: std.mem.Allocator, text: []const u8, origin: template_engine.Origin) error{OutOfMemory}!Arguments {
        if (!jsonNestingExceeds(text, max_json_nesting)) if (template_engine.fromJson(arena, text, origin)) |v| {
            if (v == .dict) return .{ .text = text, .value = v };
        } else |e| if (e == error.OutOfMemory) return error.OutOfMemory;
        const empty = try arena.create(template_engine.Dict);
        empty.* = .{};
        return .{ .text = "{}", .value = .{ .dict = empty } };
    }
};

/// `s`, or where it holds ill-formed UTF-8 a copy with U+FFFD for each bad sequence: the engine reads only valid
/// UTF-8 (a JSON string never held anything else).
fn utf8Text(arena: std.mem.Allocator, s: []const u8) error{OutOfMemory}![]const u8 {
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] < 0x80) {
            i += 1;
            continue;
        }
        const seq = utf8Next(s, i);
        if (!seq.valid) break;
        i = seq.end;
    } else return s;
    var out = try std.ArrayList(u8).initCapacity(arena, s.len + 2);
    out.appendSliceAssumeCapacity(s[0..i]);
    while (i < s.len) {
        if (s[i] < 0x80) {
            try out.append(arena, s[i]);
            i += 1;
            continue;
        }
        const seq = utf8Next(s, i);
        try out.appendSlice(arena, if (seq.valid) s[i..seq.end] else "\u{FFFD}");
        i = seq.end;
    }
    return out.items;
}

/// Who wrote a message's text: an assistant turn is the model's earlier output, any other the client's.
fn messageOrigin(role: []const u8) template_engine.Origin {
    return if (std.mem.eql(u8, role, "assistant")) .assistant else .input;
}

/// `messages` as a template reads them; an image becomes `image_atom` where it sits when the template takes
/// `content` as one string (DeepSeek-V4.1).
fn templateMessages(arena: std.mem.Allocator, messages: []const Message, image_atom: ?[]const u8) error{OutOfMemory}![]const TemplateMessage {
    const out = try arena.alloc(TemplateMessage, messages.len);
    for (messages, out) |msg, *t| {
        t.* = .{
            .role = try utf8Text(arena, msg.role),
            .content = try templateContent(arena, msg, image_atom),
            .tool_call_id = if (msg.tool_call_id) |id| try utf8Text(arena, id) else null,
            .reasoning_content = if (msg.reasoning_content) |rc| try utf8Text(arena, rc) else null,
        };
        if (msg.tool_calls) |tcs| {
            const calls = try arena.alloc(TemplateCall, tcs.len);
            for (tcs, calls) |tc, *c| c.* = .{
                .id = if (tc.id.len > 0) try utf8Text(arena, tc.id) else null,
                .function = .{ .name = try utf8Text(arena, tc.name), .arguments = try Arguments.of(arena, tc.arguments, messageOrigin(msg.role)) },
            };
            t.tool_calls = calls;
        }
    }
    return out;
}

fn templateContent(arena: std.mem.Allocator, msg: Message, image_atom: ?[]const u8) error{OutOfMemory}!TemplateMessage.Content {
    if (msg.media_parts) |parts| if (parts.len > 0) {
        if (image_atom) |atom| return .{ .text = try atomContent(arena, msg, parts, atom) };
        return .{ .parts = try contentParts(arena, msg, parts) };
    };
    return if (msg.content.len > 0) .{ .text = try contentText(arena, msg, 0, msg.content.len) } else .none;
}

/// `msg`'s content with who wrote each byte: the role, or as the server recorded it (`Message.content_origins`).
fn contentSpans(msg: Message) template_engine.Str {
    const runs = msg.content_origins orelse return .{ .bytes = msg.content, .origin = messageOrigin(msg.role) };
    var origin: template_engine.Origin = .template;
    for (runs) |r| origin = origin.strictest(r.origin);
    return .{ .bytes = msg.content, .origin = origin, .runs = runs };
}

/// Bytes `from..to` of `msg`'s content as a template reads them (`utf8Text`), each with who wrote it. A piece the
/// server put together is mended run by run: its own text is ASCII, so a bad sequence never crosses a run.
fn contentText(arena: std.mem.Allocator, msg: Message, from: usize, to: usize) error{OutOfMemory}!template_engine.Str {
    const piece = try contentSpans(msg).slice(arena, from, to);
    const text = try utf8Text(arena, piece.bytes);
    if (text.ptr == piece.bytes.ptr) return piece;
    if (piece.runs.len == 0) return piece.derive(text);
    var mended: template_engine.Builder = .{};
    var start: usize = 0;
    for (piece.runs) |run| {
        try mended.appendBytes(arena, try utf8Text(arena, piece.bytes[start..run.end]), run.origin);
        start = run.end;
    }
    return mended.str();
}

/// `content` as the typed part list vision templates render, each media part a `{"type":"image"|"video"}`
/// placeholder at its offset.
fn contentParts(arena: std.mem.Allocator, msg: Message, parts: []const MediaPart) error{OutOfMemory}![]const TemplateMessage.Part {
    const content = msg.content;
    var out = std.ArrayList(TemplateMessage.Part).empty;
    var from: usize = 0;
    for (parts) |p| {
        const at = @max(from, @min(p.at, content.len));
        if (at > from) {
            try out.append(arena, .{ .text = .{ .text = try contentText(arena, msg, from, at) } });
            from = at;
        }
        try out.append(arena, .{ .media = .{ .type = @tagName(p.kind) } });
    }
    if (from < content.len) try out.append(arena, .{ .text = .{ .text = try contentText(arena, msg, from, content.len) } });
    return out.items;
}

/// `content` as one string for a template that has no part list: an `atom` for each image at its offset, the
/// text around them, all joined by blank lines. Video has no atom there and is dropped. The atom and the blank lines
/// are the server's.
fn atomContent(arena: std.mem.Allocator, msg: Message, parts: []const MediaPart, atom: []const u8) error{OutOfMemory}!template_engine.Str {
    const content = msg.content;
    var text: template_engine.Builder = .{};
    var from: usize = 0;
    for (parts) |p| {
        const at = @max(from, @min(p.at, content.len));
        if (p.kind != .image) continue;
        if (at > from) try appendJoined(arena, &text, try contentText(arena, msg, from, at));
        try appendJoined(arena, &text, .{ .bytes = atom });
        from = at;
    }
    if (from < content.len) try appendJoined(arena, &text, try contentText(arena, msg, from, content.len));
    return text.str();
}

fn appendJoined(allocator: std.mem.Allocator, text: *template_engine.Builder, piece: template_engine.Str) error{OutOfMemory}!void {
    if (text.bytes.items.len > 0) try text.appendBytes(allocator, "\n\n", .template);
    try text.append(allocator, piece);
}

/// What a template reads besides its messages and tools, in the order a chat template's context has them: the turn
/// tokens, the thinking switch, and the effort key of the template's dialect (`templateSettings`).
const Settings = struct {
    bos_token: ?[]const u8 = null,
    eos_token: ?[]const u8 = null,
    enable_thinking: bool,
    reasoning_effort: Effort,
    thinking_mode: ?[]const u8 = null,
    preserve_thinking: ?bool = null,

    const Effort = union(enum) { word: []const u8, budget: i64 };

    /// Every field, unambiguously: the probe memo's key.
    fn key(self: *const Settings, allocator: std.mem.Allocator) ![]u8 {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        for ([_]?[]const u8{ self.bos_token, self.eos_token, self.thinking_mode }) |t| {
            if (t) |x| try buf.print(allocator, "{d}:{s}", .{ x.len, x }) else try buf.append(allocator, '-');
        }
        const keep: u8 = if (self.preserve_thinking) |k| @intFromBool(k) else 2;
        try buf.print(allocator, "{d}{d}", .{ @intFromBool(self.enable_thinking), keep });
        switch (self.reasoning_effort) {
            .word => |w| try buf.print(allocator, "{d}:{s}", .{ w.len, w }),
            .budget => |n| try buf.print(allocator, "#{d};", .{n}),
        }
        return buf.toOwnedSlice(allocator);
    }
};

/// A render's context (`template.Context`): built in the render's arena for each render, so a template that changes
/// a list changes only its own copy. It references `messages` and `tools`, which outlive the render.
const RenderContext = struct {
    messages: []const Message,
    image_atom: ?[]const u8 = null,
    /// The request's tool list as JSON text; null or empty, the template gets no `tools`.
    tools: ?[]const u8 = null,
    settings: *const Settings,
    /// The facts of the template rendering this context: `renderEngine` sets them.
    facts: *const Facts = &no_facts,

    fn context(self: *const RenderContext) template_engine.Context {
        return .{ .ptr = self, .build = build };
    }

    fn build(ptr: *const anyopaque, arena: std.mem.Allocator, diag: *template_engine.Diagnostic) template_engine.Error!template_engine.Vars {
        const self: *const RenderContext = @ptrCast(@alignCast(ptr));
        const typed = try templateMessages(arena, self.messages, self.image_atom);
        const list = try arena.alloc(template_engine.Value, typed.len);
        for (typed, self.messages, list) |t, m, *v| v.* = try template_engine.fromInput(arena, t, messageOrigin(m.role), "messages", diag);
        const messages = try template_engine.fromInput(arena, @as([]const template_engine.Value, list), .template, "messages", diag);
        var tools: ?template_engine.Value = null;
        if (self.tools) |t| if (t.len > 0) {
            tools = try template_engine.readJson(arena, t, "tools", .input, diag);
            _ = try fillToolDefKeys(arena, tools.?, self.facts);
        };
        // The turn tokens are the pack's; the effort word is what the client sent, or a word derived from it.
        const extra = try template_engine.fromInput(arena, self.settings.*, .template, "extra context", diag);
        for (extra.dict.entries.items) |*e| if (std.mem.eql(u8, e.key.str().?, "reasoning_effort")) {
            if (e.value == .string) e.value.string.origin = .input;
        };
        return .{ .messages = messages, .tools = tools, .extra = extra };
    }
};

/// Fill the OPTIONAL tool-definition keys a template reads by name (`facts`), in the parsed tool list.
///
/// `description` is optional in OpenAI's function schema and a no-arg tool needs no `parameters`, but a template that
/// renders `fn.description | tojson` hits `tojson` on Undefined, a jinja RAISE. A template that never reads the key
/// (one that dumps `tool | tojson` whole) gets the client's tool as sent, as transformers renders it. Whether a key
/// was added. The defaults are the empty ones: `""` says the client gave no description, and an empty-properties
/// object is what a no-argument tool means.
fn fillToolDefKeys(arena: std.mem.Allocator, tools: template_engine.Value, facts: *const Facts) error{OutOfMemory}!bool {
    if (tools != .list) return false;
    var changed = false;
    for (tools.list.items.items) |tool| {
        if (tool != .dict) continue;
        const function = dictField(tool.dict, "function") orelse continue;
        if (function != .dict) continue;
        const d = function.dict;
        if (facts.readsAttribute("description") and dictField(d, "description") == null) {
            try d.entries.append(arena, .{ .key = .plain("description"), .value = .plain("") });
            changed = true;
        }
        if (facts.readsAttribute("parameters") and dictField(d, "parameters") == null) {
            try d.entries.append(arena, .{ .key = .plain("parameters"), .value = try emptyParameters(arena) });
            changed = true;
        }
    }
    return changed;
}

/// `{"type": "object", "properties": {}}`: what a no-argument tool means.
fn emptyParameters(arena: std.mem.Allocator) error{OutOfMemory}!template_engine.Value {
    const properties = try arena.create(template_engine.Dict);
    properties.* = .{};
    const d = try arena.create(template_engine.Dict);
    d.* = .{};
    try d.entries.appendSlice(arena, &.{
        .{ .key = .plain("type"), .value = .plain("object") },
        .{ .key = .plain("properties"), .value = .{ .dict = properties } },
    });
    return .{ .dict = d };
}

fn dictField(d: *const template_engine.Dict, name: []const u8) ?template_engine.Value {
    for (d.entries.items) |e| if (e.key == .string and std.mem.eql(u8, e.key.string.bytes, name)) return e.value;
    return null;
}

/// Whether the template READS `reasoning_effort` — i.e. the word is a
/// BEHAVIORAL lever on this checkpoint (DeepSeek's preamble, Qwen3.8's and
/// GLM's effort arms) rather than a string we hand over and it ignores.
/// Where it is, the server must NOT also derive a token budget from the same
/// word: the budget only truncates what the client is shown, so the two
/// levers fight — pi asking for `medium` on Qwen3.8 got the model's default
/// (unguided, long) thinking AND a 2048-token display cut, then 25.9k tokens
/// of invisible generation (live 2026-08-14). An explicit
/// `reasoning_budget_tokens` or an effort table's cap is someone asking for a
/// cap on purpose and still applies.
///
/// Read on the KEY, so a family reading a different one is untouched.
pub fn templateConsumesEffort(chat_config: *const ChatConfig) bool {
    return asks.effort(chat_config.facts());
}

/// The model can produce reasoning_content: its template reads `enable_thinking` or prints a think opener (`<think>`,
/// Gemma 4's `<|channel>thought`).
pub fn templateSupportsThinking(chat_config: *const ChatConfig) bool {
    return asks.reasoning(chat_config.facts());
}

/// The one spelling of think opener the template prints. Null when it prints none, or a dynamic or conflicting one:
/// a generated tag needs one unambiguous literal spelling, else the unsupported-format fallback stays.
pub fn templateThinkOpener(chat_config: *const ChatConfig) ?[]const u8 {
    return asks.thinkOpener(chat_config.facts());
}

/// It prints Gemma 4's `<|channel>`: reasoning rides channels, not think tags.
pub fn templateUsesChannels(chat_config: *const ChatConfig) bool {
    return asks.channels(chat_config.facts());
}

/// OpenAI's effort vocabulary -> the scale dialect, the JSON value a `thinking_mode` encoder takes (DeepSeek-V4.1):
/// its words (low 50, high 75, max 100, default high) or a 1-100 budget for the steps between them.
fn scaleEffortFor(effort: ?[]const u8) []const u8 {
    const e = effort orelse return "\"high\"";
    // A table's value on the 1-100 scale; the encoder's own three steps go as its words.
    if (std.fmt.parseInt(u8, e, 10)) |n| {
        if (e[0] != '+' and n >= 1 and n <= 100) return switch (n) {
            50 => "\"low\"",
            75 => "\"high\"",
            100 => "\"max\"",
            else => effort_mod.scaleText(n),
        };
    } else |_| {}
    if (std.mem.eql(u8, e, "none") or std.mem.eql(u8, e, "minimal") or std.mem.eql(u8, e, "low")) return "\"low\"";
    if (std.mem.eql(u8, e, "medium")) return "62";
    if (std.mem.eql(u8, e, "xhigh")) return "87";
    if (std.mem.eql(u8, e, "max") or std.mem.eql(u8, e, "ultra")) return "\"max\"";
    return "\"high\"";
}

/// OpenAI's effort vocabulary -> the ladder dialect, xhigh|medium|low (Qwen3.8). The template
/// raise_exception's on any other string and its own default is xhigh, so an
/// absent or unrecognized effort keeps the checkpoint default. The template
/// never reads the key on its thinking-off arm, so the value is inert there.
fn ladderEffortFor(effort: ?[]const u8, enable_thinking: bool) []const u8 {
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
/// request didn't send one). The scale, ladder and word dialects each map it into their template; any other
/// template keeps a fixed vocabulary.
fn templateSettings(allocator: std.mem.Allocator, chat_config: *const ChatConfig, enable_thinking: bool, effort: ?[]const u8) !Settings {
    var own: ?*Compiled = null;
    defer if (own) |c| c.deinit(allocator);
    const facts = (try templatesOf(allocator, chat_config, &own)).facts();
    // Templates that ignore `reasoning_effort` get high/no_think. The families
    // that read it each have their own vocabulary (GLM's free word, Qwen3.8's
    // xhigh|medium|low, DeepSeek's low|high|max) and raise on any other string:
    // a raise swaps in fallbackFormatChat (wrong-family tags), so every one is
    // mapped below instead of sent the default.
    const scale_style = asks.thinkingMode(facts);
    const reasoning_effort: Settings.Effort = if (asks.effortWords(facts))
        // Sushi defaults to high; depth is a prompt instruction, never a token cap.
        .{ .word = effort orelse "high" }
    else if (scale_style)
        // V4.1 renders a numeric budget for every thinking turn, on its own ladder.
        scaleEffort(scaleEffortFor(effort))
    else if (asks.effortLadder(facts))
        // Qwen3.8's vocabulary, keyed on the literal 'xhigh': every 3.8 template raises on OpenAI's "high".
        .{ .word = ladderEffortFor(effort, enable_thinking) }
    else
        .{ .word = if (enable_thinking) "high" else "no_think" };
    return .{
        .bos_token = chat_config.bos_token,
        .eos_token = chat_config.eos_token,
        .enable_thinking = enable_thinking,
        .reasoning_effort = reasoning_effort,
        // DeepSeek-V4.1's template switches on `thinking_mode` ("chat"|"thinking", the reference encoder's own
        // vocabulary). Only set when the template reads it — other families never see the key.
        .thinking_mode = if (scale_style) (if (enable_thinking) "thinking" else "chat") else null,
        .preserve_thinking = if (asks.preserveThinking(facts)) chat_config.preserve_thinking else null,
    };
}

/// A scale value as `scaleEffortFor` spells it in JSON: a quoted word or a budget.
fn scaleEffort(text: []const u8) Settings.Effort {
    if (text[0] == '"') return .{ .word = text[1 .. text.len - 1] };
    return .{ .budget = std.fmt.parseInt(i64, text, 10) catch unreachable };
}

/// What Qwen's template raises for a conversation with no user turn to answer (a system-only or assistant-only chat).
const NO_USER_QUERY_RAISE = "No user query found in messages.";
comptime {
    // Matched whole against the diagnostic, which keeps the first `buf.len` bytes of a raise.
    std.debug.assert(NO_USER_QUERY_RAISE.len <= @typeInfo(@FieldType(template_engine.Diagnostic, "buf")).array.len);
}

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
) !Spans {
    var result: template_engine.Builder = .{};
    errdefer result.deinit(allocator);
    // It writes the request's fields itself, so it mends them as the engine does what it reads.
    var mend = std.heap.ArenaAllocator.init(allocator);
    defer mend.deinit();
    const ma = mend.allocator();
    const t = struct {
        fn text(a: std.mem.Allocator, b: *template_engine.Builder, bytes: []const u8) !void {
            try b.appendBytes(a, bytes, .template);
        }
    }.text;

    const has_system = messages.len > 0 and std.mem.eql(u8, messages[0].role, "system");

    if (tools_json != null and !has_system) {
        try t(allocator, &result, "<|im_start|>system\n");
        try appendToolPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
        try t(allocator, &result, "<|im_end|>\n");
    }

    for (messages) |msg| {
        if (std.mem.eql(u8, msg.role, "system") and tools_json != null) {
            try t(allocator, &result, "<|im_start|>system\n");
            try result.append(allocator, try contentText(ma, msg, 0, msg.content.len));
            try t(allocator, &result, "\n\n");
            try appendToolPrompt(allocator, &result, tools_json.?, tool_choice_instruction);
            try t(allocator, &result, "<|im_end|>\n");
        } else if (std.mem.eql(u8, msg.role, "assistant") and msg.tool_calls != null) {
            try t(allocator, &result, "<|im_start|>assistant\n");
            if (msg.content.len > 0) {
                try result.append(allocator, try contentText(ma, msg, 0, msg.content.len));
                try t(allocator, &result, "\n");
            }
            for (msg.tool_calls.?) |tc| {
                try t(allocator, &result, "<tool_call>\n");
                try t(allocator, &result, "{\"name\": \"");
                try result.appendBytes(allocator, try utf8Text(ma, tc.name), .assistant);
                try t(allocator, &result, "\", \"arguments\": ");
                try result.appendBytes(allocator, try utf8Text(ma, tc.arguments), .assistant);
                try t(allocator, &result, "}\n</tool_call>");
            }
            try t(allocator, &result, "<|im_end|>\n");
        } else if (std.mem.eql(u8, msg.role, "tool")) {
            try t(allocator, &result, "<|im_start|>user\n");
            try t(allocator, &result, "<tool_response>\n");
            try result.append(allocator, try contentText(ma, msg, 0, msg.content.len));
            try t(allocator, &result, "\n</tool_response>");
            try t(allocator, &result, "<|im_end|>\n");
        } else {
            try t(allocator, &result, "<|im_start|>");
            try result.appendBytes(allocator, try utf8Text(ma, msg.role), messageOrigin(msg.role));
            try t(allocator, &result, "\n");
            try result.append(allocator, try contentText(ma, msg, 0, msg.content.len));
            try t(allocator, &result, "<|im_end|>\n");
        }
    }
    try t(allocator, &result, "<|im_start|>assistant\n");
    var own: ?*Compiled = null;
    defer if (own) |c| c.deinit(allocator);
    if (asks.enableThinking((try templatesOf(allocator, chat_config, &own)).facts())) {
        if (enable_thinking) {
            try t(allocator, &result, "<think>\n");
        } else {
            try t(allocator, &result, "<think>\n\n</think>\n\n");
        }
    }

    return ownBuilt(allocator, &result);
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
            // A strict JSON parser rejects ill-formed UTF-8 like a raw control byte.
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
            // results) must be \u-escaped: a strict JSON parser rejects raw
            // control bytes.
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
const arch_fx = @import("arch_fixtures_test.zig");

test "collapseDoubledThinkTags collapses 2x → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, .{ .bytes = "<|Assistant|></think></think>Hi!" });
    defer freeSpans(testing.allocator, out);
    try testing.expectEqualStrings("<|Assistant|></think>Hi!", out.bytes);
}

test "collapseDoubledThinkTags collapses triple → 1x" {
    const out = try collapseDoubledThinkTags(testing.allocator, .{ .bytes = "</think></think></think>X" });
    defer freeSpans(testing.allocator, out);
    try testing.expectEqualStrings("</think>X", out.bytes);
}

test "collapseDoubledThinkTags leaves single </think> unchanged" {
    const out = try collapseDoubledThinkTags(testing.allocator, .{ .bytes = "<think>r</think>Hello" });
    defer freeSpans(testing.allocator, out);
    try testing.expectEqualStrings("<think>r</think>Hello", out.bytes);
}

test "collapseDoubledThinkTags handles multiple separated doublings" {
    const out = try collapseDoubledThinkTags(testing.allocator,
        .{ .bytes = "A</think></think>B</think></think>C" });
    defer freeSpans(testing.allocator, out);
    try testing.expectEqualStrings("A</think>B</think>C", out.bytes);
}

test "collapseDoubledThinkTags: the template's own doubled tags collapse, every kept byte keeping who wrote it" {
    const out = try collapseDoubledThinkTags(testing.allocator, .{ .bytes = "a</think></think>b", .origin = .assistant, .runs = &.{ .{ .end = 17, .origin = .template }, .{ .end = 18, .origin = .assistant } } });
    defer freeSpans(testing.allocator, out);
    try testing.expectEqualStrings("a</think>b", out.bytes);
    try testing.expectEqualSlices(template_engine.Run, &.{ .{ .end = 9, .origin = .template }, .{ .end = 10, .origin = .assistant } }, out.runs);
    // A pair a message wrote even half of is that message's text.
    const mixed = try collapseDoubledThinkTags(testing.allocator, .{ .bytes = "a</think></think>b", .origin = .assistant, .runs = &.{ .{ .end = 9, .origin = .template }, .{ .end = 18, .origin = .assistant } } });
    defer freeSpans(testing.allocator, mixed);
    try testing.expectEqualStrings("a</think></think>b", mixed.bytes);
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
    // from `\x1b[?25l`) passed through unescaped, the C++ template engine of the
    // time rejected the messages JSON, and the prompt silently fell
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
    // Everything else on disk (gemma 4, LFM2.5, Qwen3.5/3.6, Ling,
    // Nemotron, llama, mistral) reads neither: thinking is a bool there.
    for ([_]struct { tpl: []const u8, reads: bool }{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .reads = true },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .reads = true },
        .{ .tpl = @embedFile("fixtures/glm53_chat_template.jinja"), .reads = true },
        .{ .tpl = "{%- if enable_thinking %}<think>\n{%- endif %}", .reads = false },
        .{ .tpl = "", .reads = false },
    }) |c| {
        var cfg = ChatConfig{ .allocator = testing.allocator, .chat_template = c.tpl, .bos_token = null, .eos_token = null, .add_bos_token = false };
        try cfg.compile();
        defer cfg.dropCompiled();
        try testing.expectEqual(c.reads, templateConsumesEffort(&cfg));
    }
}

test "renderChatTemplate: a tool with no description/parameters still renders (no silent fallback)" {
    // OpenAI's schema makes `description` optional and a no-arg tool needs no
    // `parameters`, but a template that dumps those keys through `tojson` RAISES
    // on Undefined and fell back to ChatML with no tools preamble at all. The
    // defaults are filled for a template that reads those keys.
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
    var fill_arena = std.heap.ArenaAllocator.init(allocator);
    defer fill_arena.deinit();
    const fa = fill_arena.allocator();
    var diag: template_engine.Diagnostic = .{};
    var reads = try template_engine.Template.compile(allocator, "{% for t in tools %}{{ t.function.description | tojson }}{{ t.function.parameters | tojson }}{% endfor %}", &diag);
    defer reads.deinit();
    try testing.expect(!try fillToolDefKeys(fa, try template_engine.fromJson(fa, complete, .input), &reads.facts));
    // Only the missing keys are added, and only to the function that lacks them.
    const partial =
        \\[{"type":"function","function":{"name":"a","description":"d"}},{"type":"function","function":{"name":"b","description":"e","parameters":{"type":"object"}}}]
    ;
    // Qwen3.8 dumps each tool whole: it reads neither key, so nothing is added and the tool renders as sent.
    try config.compile();
    defer config.dropCompiled();
    try testing.expect(!try fillToolDefKeys(fa, try template_engine.fromJson(fa, partial, .input), config.facts()));
    const filled = try template_engine.fromJson(fa, partial, .input);
    try testing.expect(try fillToolDefKeys(fa, filled, &reads.facts));
    var text = std.ArrayList(u8).empty;
    try test_json.writeValue(fa, &text, filled);
    try testing.expectEqualStrings("[{\"type\":\"function\",\"function\":{\"name\":\"a\",\"description\":\"d\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}," ++
        "{\"type\":\"function\",\"function\":{\"name\":\"b\",\"description\":\"e\",\"parameters\":{\"type\":\"object\"}}}]", text.items);
}

test "renderChatTemplate folds a late system turn only where the template refuses it" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "S" },
        .{ .role = "user", .content = "hi" },
        .{ .role = "system", .content = "late" },
        .{ .role = "user", .content = "again" },
    };
    // A Qwen3.8 revision whose raise the late-system edit does not find raises on a system turn past index 0,
    // so the turn is folded into the leading one.
    {
        const revised = try std.mem.replaceOwned(u8, allocator, @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), "{%- if not loop.first %}", "{%- if loop.index0 > 0 %}");
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

test "stock Qwen late system notes render in place and keep tool rounds a prefix" {
    if (arch_fx.qwen_late_templates.len == 0) return error.SkipZigTest;
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
    for (arch_fx.qwen_late_templates) |tpl| {
        for ([_]?bool{ null, true, false }) |preserve| {
            const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .preserve_thinking = preserve, .model_type = arch_fx.qwen_model_type };
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
    if (arch_fx.qwen_late_templates.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "system", .content = "second lead" },
        .{ .role = "user", .content = "question" },
    };
    for (arch_fx.qwen_late_templates) |tpl| {
        const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
        const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
        defer a.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "stable<|im_end|>\n<|im_start|>system\nsecond lead<|im_end|>\n<|im_start|>user\nquestion") != null);
    }
}

test "stock Qwen adapter leaves prompts without late notes byte-identical, tools, thinking and media included" {
    if (arch_fx.qwen_late_templates.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const parts = [_]MediaPart{ .{ .at = 0, .kind = .image }, .{ .at = 0, .kind = .video } };
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "lookup", .arguments = "{\"index\":3}" }};
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\"}}}]";
    for (arch_fx.qwen_late_templates, 0..) |tpl, index| {
        const adapted = (try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, tpl)).?;
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
    if (arch_fx.qwen_late_templates.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const tpl = arch_fx.qwen_late_templates[0];
    const config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
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

test "Qwen's late-system edit engages on any revision with the branch, and one without it folds" {
    if (arch_fx.qwen_late_templates.len == 0) return error.SkipZigTest;
    const a = testing.allocator;
    const messages = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "question" },
        .{ .role = "system", .content = "late note" },
    };
    for (arch_fx.qwen_late_templates) |tpl| {
        const revised = try std.mem.concat(a, u8, &.{ tpl, "{{- 'revision marker' }}" });
        defer a.free(revised);
        const respelled = try std.mem.replaceOwned(u8, a, tpl, "message.role == \"system\"", "message.role == 'system'");
        defer a.free(respelled);
        for ([_][]const u8{ revised, respelled }) |known| {
            const adapted = (try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, known)) orelse return error.LateSystemBranchNotFound;
            a.free(adapted);
            var config = ChatConfig{ .chat_template = known, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
            const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
            defer a.free(rendered);
            try testing.expect(std.mem.indexOf(u8, rendered, "<|im_start|>system\nlate note<|im_end|>") != null);
        }
        const unguarded = try std.mem.replaceOwned(u8, a, tpl, "{%- if not loop.first %}", "{%- if loop.index0 > 0 %}");
        defer a.free(unguarded);
        try testing.expect(!std.mem.eql(u8, unguarded, tpl));
        try testing.expect((try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, unguarded)) == null);
        const config = ChatConfig{ .chat_template = unguarded, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
        const rendered = try renderChatTemplate(a, &messages, &config, null, null, true, "low", false);
        defer a.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "stable\n\nlate note<|im_end|>\n<|im_start|>user\nquestion") != null);
    }
}

test "real Qwen pack loaded through ChatConfig keeps late notes in place" {
    const dir = std.c.getenv("QWEN_MID_SYSTEM_MODEL_DIR") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try loadChatConfig(io, a, std.mem.span(dir), arch_fx.qwen_model_type);
    defer config.deinit();
    const adapted = try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, config.chat_template);
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
    const Case = struct { tpl: []const u8, sep: []const u8, required: []const u8, named: []const u8, name_tails: []const []const u8 };
    const cases = [_]Case{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .sep = "\n\n", .required = "<tool_call>\n<function=", .named = "<tool_call>\n<function=get_time>\n", .name_tails = &.{">\n"} },
        .{ .tpl = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .sep = "", .required = "<tool_call><function=", .named = "<tool_call><function=get_time>", .name_tails = &.{">"} },
        // A tag follows the name: a call with arguments writes `<arg_key>`, one without closes at once, so the name has two tails.
        .{ .tpl = @embedFile("fixtures/glm53_chat_template.jinja"), .sep = "", .required = "<tool_call>", .named = "<tool_call>", .name_tails = &.{ "<arg_key>", "</tool_call>" } },
    };
    for (cases) |c| {
        var config = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = allocator };
        var any = (try forcedCallText(allocator, &config, null)).?;
        defer any.deinit(allocator);
        try testing.expectEqualStrings(c.sep, any.sep);
        try testing.expectEqualStrings(c.required, any.call);
        try testing.expectEqual(c.name_tails.len, any.name_tails.len);
        for (c.name_tails, any.name_tails) |want, got| try testing.expectEqualStrings(want, got);
        var one = (try forcedCallText(allocator, &config, "get_time")).?;
        defer one.deinit(allocator);
        try testing.expectEqualStrings(c.sep, one.sep);
        try testing.expectEqualStrings(c.named, one.call);
    }
    // No think block to anchor on: nothing is forced.
    var plain = ChatConfig{ .chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    try testing.expect(try forcedCallText(allocator, &plain, null) == null);
}

test "answerSeparator reads what each family's template writes between a closed thought and its answer" {
    const allocator = testing.allocator;
    const Case = struct { tpl: []const u8, sep: ?[]const u8 };
    const cases = [_]Case{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .sep = "\n\n" },
        .{ .tpl = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .sep = "" },
        .{ .tpl = @embedFile("fixtures/glm53_chat_template.jinja"), .sep = "" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .sep = "" },
        .{ .tpl = "{% for message in messages %}{{ message.content }}{% endfor %}", .sep = null },
    };
    for (cases) |c| {
        var config = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
        const got = try answerSeparator(allocator, &config);
        defer if (got) |g| allocator.free(g);
        if (c.sep) |want| try testing.expectEqualStrings(want, got.?) else try testing.expect(got == null);
    }
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

test "templateSettings with thinking enabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try extraJsonForTest(allocator, &config, true, null);
    defer allocator.free(result);
    // Should contain enable_thinking:true
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":true") != null);
}

test "templateSettings with thinking disabled" {
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = "",
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    _ = &config;

    const result = try extraJsonForTest(allocator, &config, false, null);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"enable_thinking\":false") != null);
}

test "templateSettings: DeepSeek-V4.1 maps the client's effort onto the encoder's ladder" {
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
        const r = try extraJsonForTest(allocator, &dsv41, thinking, null);
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
        .{ .in = "xhigh", .out = "\"reasoning_effort\":87" },
        .{ .in = "medium", .out = "\"reasoning_effort\":62" },
        .{ .in = "low", .out = "\"reasoning_effort\":\"low\"" },
    };
    for (mapped_cases) |case| {
        const r = try extraJsonForTest(allocator, &dsv41, true, case.in);
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
    const on = try extraJsonForTest(allocator, &other, true, null);
    defer allocator.free(on);
    try testing.expect(std.mem.indexOf(u8, on, "\"reasoning_effort\":\"high\"") != null);
    const off = try extraJsonForTest(allocator, &other, false, null);
    defer allocator.free(off);
    try testing.expect(std.mem.indexOf(u8, off, "\"reasoning_effort\":\"no_think\"") != null);
}

test "templateSettings: a table's value reaches each family's template as that family spells it" {
    const allocator = testing.allocator;
    const Case = struct { tpl: []const u8, value: []const u8, want: []const u8 };
    for ([_]Case{
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .value = "medium", .want = "\"reasoning_effort\":\"medium\"" },
        .{ .tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .value = "low", .want = "\"reasoning_effort\":\"low\"" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "40", .want = "\"reasoning_effort\":40" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "62", .want = "\"reasoning_effort\":62" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "50", .want = "\"reasoning_effort\":\"low\"" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "75", .want = "\"reasoning_effort\":\"high\"" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "100", .want = "\"reasoning_effort\":\"max\"" },
        .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .value = "101", .want = "\"reasoning_effort\":\"high\"" },
        .{ .tpl = @embedFile("fixtures/glm53_chat_template.jinja"), .value = "low", .want = "\"reasoning_effort\":\"low\"" },
    }) |c| {
        var cfg = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
        const r = try extraJsonForTest(allocator, &cfg, true, c.value);
        defer allocator.free(r);
        try testing.expect(std.mem.indexOf(u8, r, c.want) != null);
    }
    var ds = ChatConfig{ .chat_template = @embedFile("fixtures/dsv41_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = allocator };
    const rendered = try renderChatTemplate(allocator, &.{.{ .role = "user", .content = "hi" }}, &ds, null, null, true, "40", false);
    defer allocator.free(rendered);
    try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning Effort: 40 (range 1-100") != null);
}

test "templateMessages basic" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "Hello" },
        .{ .role = "assistant", .content = "Hi there" },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);

    // Parse it back to verify valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.array.items.len);
    try testing.expectEqualStrings("user", parsed.value.array.items[0].object.get("role").?.string);
    try testing.expectEqualStrings("Hello", parsed.value.array.items[0].object.get("content").?.string);
}

test "templateMessages with tool_calls" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "get_weather", .arguments = "{\"location\":\"Tokyo\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);

    // Should contain tool_calls array
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_calls\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "get_weather") != null);
}

test "templateMessages carries assistant reasoning_content" {
    const allocator = testing.allocator;
    const messages = [_]Message{
        .{ .role = "user", .content = "What is 2+2?" },
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
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

test "templateMessages embeds valid-JSON arguments as object (not string)" {
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
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);

    // Object form: ..."arguments":{"command":"date"}...
    // String form (rejected): ..."arguments":"{\"command\":\"date\"}"...
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{\"command\":\"date\"}") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":\"{") == null);
}

test "templateMessages embeds an empty object for malformed arguments" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "not valid json" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);
    try testing.expect(std.mem.indexOf(u8, result, "\"arguments\":{}") != null);
}

test "templateMessages tool response has tool_call_id and content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"ls\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "file1.txt\nfile2.txt", .tool_call_id = "call_1" },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);

    // Must contain tool_call_id
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"call_1\"") != null);
    // Must contain the response content
    try testing.expect(std.mem.indexOf(u8, result, "file1.txt") != null);
    // Must NOT contain tool_responses (templates handle role:"tool" natively)
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "templateMessages full tool calling conversation" {
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
    const result = try messagesJsonForTest(allocator, &messages, null);
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

test "templateMessages multiple parallel tool responses" {
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
    const result = try messagesJsonForTest(allocator, &messages, null);
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

test "templateMessages tool response with empty content" {
    const allocator = testing.allocator;
    const tool_calls = [_]ToolCall{
        .{ .id = "call_1", .name = "shell", .arguments = "{\"command\":\"mkdir test\"}" },
    };
    const messages = [_]Message{
        .{ .role = "assistant", .content = "", .tool_calls = &tool_calls },
        .{ .role = "tool", .content = "", .tool_call_id = "call_1" },
    };
    const result = try messagesJsonForTest(allocator, &messages, null);
    defer allocator.free(result);

    // Should have tool_call_id but no tool_responses
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_call_id\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"tool_responses\"") == null);
}

test "templateMessages renders each media part at its offset as a typed part" {
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
    const result = try messagesJsonForTest(allocator, &messages, null);
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
    const spans = try fallbackFormatChat(allocator, &messages, &config, tools_json, null, false);
    defer freeSpans(allocator, spans);
    const result = spans.bytes;

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

    const spans = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer freeSpans(allocator, spans);
    const result = spans.bytes;

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

    const spans = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer freeSpans(allocator, spans);
    const result = spans.bytes;

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

    const spans = try fallbackFormatChat(allocator, &messages, &config, null, null, false);
    defer freeSpans(allocator, spans);
    const result = spans.bytes;

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
    // Bar: the escaper's output is always well-formed UTF-8 (a strict JSON
    // parser rejects anything else), and every valid sequence survives unchanged.
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

test "templateMessages: a string-content template gets each image as its atom where it sits, blank-line joined" {
    const a = testing.allocator;
    const parts = [_]MediaPart{ .{ .at = 0, .kind = .image }, .{ .at = 5, .kind = .image }, .{ .at = 5, .kind = .video } };
    const msgs = [_]Message{
        .{ .role = "user", .content = "AAAAABBB", .media_parts = &parts },
        .{ .role = "tool", .content = "result", .media_parts = parts[0..1] },
        .{ .role = "user", .content = "plain" },
    };
    var cfg = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = "<IMG>" };
    const json = try messagesJsonForTest(a, &msgs, cfg.image_placeholder);
    defer a.free(json);
    try testing.expectEqualStrings(
        "[{\"role\":\"user\",\"content\":\"<IMG>\\n\\nAAAAA\\n\\n<IMG>\\n\\nBBB\"}," ++
            "{\"role\":\"tool\",\"content\":\"<IMG>\\n\\nresult\"},{\"role\":\"user\",\"content\":\"plain\"}]",
        json,
    );
    // Without the atom the same message renders as typed parts.
    cfg.image_placeholder = null;
    const typed = try messagesJsonForTest(a, msgs[0..1], cfg.image_placeholder);
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
    try cfg.compile();
    defer cfg.dropCompiled();
    try testing.expect(templateConsumesEffort(&cfg));
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

test "an effort word with a quote or a backslash reaches the word-dialect template as that text" {
    // As GLM's: the word dialect is a template that reads `effective_reasoning_effort`.
    const tpl = "[gMASK]<sop>{% set effective_reasoning_effort = reasoning_effort %}[{{ effective_reasoning_effort }}]";
    var cfg = ChatConfig{ .allocator = testing.allocator, .chat_template = tpl, .bos_token = null, .eos_token = null, .add_bos_token = false };
    for ([_][]const u8{ "a\"b", "back\\slash", "\",\"enable_thinking\":false,\"x\":\"" }) |word| {
        const rendered = try renderChatTemplate(testing.allocator, &.{}, &cfg, null, null, true, word, false);
        defer testing.allocator.free(rendered);
        const want = try std.fmt.allocPrint(testing.allocator, "[gMASK]<sop>[{s}]", .{word});
        defer testing.allocator.free(want);
        try testing.expectEqualStrings(want, rendered);
    }
}

test "jinja: a number after a dot subscripts, as in Jinja2" {
    const tpl = "{{ x.0.y }}|{{ x.1.0 }}|{{ x.1.1 }}|{{ s.0 }}|{{ x[1].0 }}|{{ x.1.0 + 0.5 }}|{{ 'no' if x.5 is not defined else 'yes' }}|{{ 1.5 }}";
    const extra = "{\"x\":[{\"y\":\"a\"},[7,8]],\"s\":\"hey\"}";
    const out = try renderRawForTest(testing.allocator, tpl, "[]", null, extra, false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings("a|7|8|h|7|7.5|no|1.5", out);
}

/// `tpl` rendered by the served engine over raw JSON inputs; the caller frees the render.
fn renderRawForTest(a: std.mem.Allocator, tpl: []const u8, messages: []const u8, tools: ?[]const u8, extra: []const u8, add_generation_prompt: bool) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var diag: template_engine.Diagnostic = .{};
    const out = template_engine.renderJson(arena.allocator(), .{ .template = tpl, .messages_json = messages, .tools_json = tools, .extra_json = extra, .add_generation_prompt = add_generation_prompt }, &diag) catch |e| {
        std.debug.print("\ntemplate error: {s}: {s}\n", .{ @errorName(e), diag.message() });
        return error.RenderFailed;
    };
    return a.dupe(u8, out);
}

fn renderJinjaRaw(tpl: []const u8, messages: []const u8, expect: []const u8) !void {
    const out = try renderRawForTest(testing.allocator, tpl, messages, null, "{}", false);
    defer testing.allocator.free(out);
    try testing.expectEqualStrings(expect, out);
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

test "renderChatTemplate: a replayed call's numbers reach the prompt as Python's json.dumps writes them" {
    var config = ChatConfig{
        .chat_template = "{% for m in messages %}{% if m.role == 'tool' %}{{ m.content }}{% endif %}{% if m.tool_calls %}{% for c in m.tool_calls %}{{ c.function.arguments|tojson }}{% endfor %}{% endif %}{% endfor %}",
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = testing.allocator,
    };
    const call = [_]ToolCall{.{ .id = "c1", .name = "f", .arguments = "{\"one\":1.0,\"pi\":3.141592653589793,\"big\":18446744073709551616,\"tiny\":1e-05}" }};
    const msgs = [_]Message{ .{ .role = "user", .content = "q" }, .{ .role = "assistant", .content = "", .tool_calls = &call } };
    const rendered = try renderChatTemplate(testing.allocator, &msgs, &config, null, null, true, null, false);
    defer testing.allocator.free(rendered);
    try testing.expectEqualStrings("{\"one\": 1.0, \"pi\": 3.141592653589793, \"big\": 18446744073709551616, \"tiny\": 1e-05}", rendered);
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
    const json = try messagesJsonForTest(a, &msgs, null);
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
    const cfg = ChatConfig{ .chat_template = @embedFile("fixtures/dsv41_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = (arch_chat.pinOf("deepseek_v41") orelse return).image_placeholder };
    for (cases) |c| for ([_]struct { mode: []const u8, want: []const u8 }{ .{ .mode = "chat", .want = c.chat }, .{ .mode = "thinking", .want = c.thinking } }) |arm| {
        const json = try messagesJsonForTest(a, c.msgs, cfg.image_placeholder);
        defer a.free(json);
        const extra = try std.fmt.allocPrint(a, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":\"high\"}}", .{arm.mode});
        defer a.free(extra);
        const out = try renderRawForTest(a, cfg.chat_template, json, null, extra, true);
        defer a.free(out);
        try testing.expectEqualStrings(arm.want, out);
    };
}

test "chat: the embedded V4.1 template renders through the template engine byte-exact with DeepSeek's encoder" {
    const a = testing.allocator;
    const tpl = @embedFile("fixtures/dsv41_chat_template.jinja");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/dsv41_template_cases.json"), .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |case| {
        const o = case.object;
        // Messages the server orders before it renders them: the test after this one.
        if (o.get("client_order") != null) continue;
        const msgs = try std.json.Stringify.valueAlloc(a, o.get("messages").?, .{});
        defer a.free(msgs);
        const tools: ?[]u8 = if (o.get("tools").? == .null) null else try std.json.Stringify.valueAlloc(a, o.get("tools").?, .{});
        defer if (tools) |t| a.free(t);
        const effort = try std.json.Stringify.valueAlloc(a, o.get("reasoning_effort").?, .{});
        defer a.free(effort);
        const extra = try std.fmt.allocPrint(a, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":{s}}}", .{ o.get("thinking_mode").?.string, effort });
        defer a.free(extra);
        const out = renderRawForTest(a, tpl, msgs, tools, extra, true) catch |e| {
            std.debug.print("case: {s}\n", .{o.get("label").?.string});
            return e;
        };
        defer a.free(out);
        testing.expectEqualStrings(o.get("expected").?.string, out) catch |e| {
            std.debug.print("case: {s}\n", .{o.get("label").?.string});
            return e;
        };
    }
}

test "chat: tool results a client sends out of call order render byte-exact with DeepSeek's encoder once the messages are ordered" {
    const model_type = arch_chat.modelTypeOf("dsv41") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const tpl = @embedFile("fixtures/dsv41_chat_template.jinja");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures/dsv41_template_cases.json"), .{});
    defer parsed.deinit();
    var seen: usize = 0;
    for (parsed.value.array.items) |case| {
        const o = case.object;
        if (o.get("client_order") == null) continue;
        seen += 1;
        var arena_state = std.heap.ArenaAllocator.init(a);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var msgs = std.ArrayList(Message).empty;
        for (o.get("messages").?.array.items) |mv| {
            const m = mv.object;
            var calls: ?[]const ToolCall = null;
            if (m.get("tool_calls")) |tcs| {
                const list = try arena.alloc(ToolCall, tcs.array.items.len);
                for (list, tcs.array.items) |*tc, v| {
                    const f = v.object.get("function").?.object;
                    tc.* = .{ .id = v.object.get("id").?.string, .name = f.get("name").?.string, .arguments = try std.json.Stringify.valueAlloc(arena, f.get("arguments").?, .{}) };
                }
                calls = list;
            }
            try msgs.append(arena, .{
                .role = m.get("role").?.string,
                .content = m.get("content").?.string,
                .tool_calls = calls,
                .tool_call_id = if (m.get("tool_call_id")) |v| v.string else null,
                .reasoning_content = if (m.get("reasoning_content")) |v| v.string else null,
            });
        }
        const tools = try std.json.Stringify.valueAlloc(arena, o.get("tools").?, .{});
        const effort = try std.json.Stringify.valueAlloc(arena, o.get("reasoning_effort").?, .{});
        const extra = try std.fmt.allocPrint(arena, "{{\"thinking_mode\":\"{s}\",\"reasoning_effort\":{s}}}", .{ o.get("thinking_mode").?.string, effort });
        errdefer std.debug.print("case: {s}\n", .{o.get("label").?.string});
        // As sent, the render is not the encoder's: the order is what makes it so.
        const sent = try renderRawForTest(a, tpl, try messagesJsonForTest(arena, msgs.items, null), tools, extra, true);
        defer a.free(sent);
        try testing.expect(!std.mem.eql(u8, o.get("expected").?.string, sent));
        orderMessages(model_type, msgs.items);
        const out = try renderRawForTest(a, tpl, try messagesJsonForTest(arena, msgs.items, null), tools, extra, true);
        defer a.free(out);
        try testing.expectEqualStrings(o.get("expected").?.string, out);
    }
    try testing.expect(seen >= 4);
}

test "chat: the V4.1 effort words map onto the encoder's 1-100 ladder, medium at 62 and xhigh at 87" {
    try testing.expectEqualStrings("\"high\"", scaleEffortFor(null));
    try testing.expectEqualStrings("\"low\"", scaleEffortFor("minimal"));
    try testing.expectEqualStrings("62", scaleEffortFor("medium"));
    try testing.expectEqualStrings("\"high\"", scaleEffortFor("high"));
    try testing.expectEqualStrings("87", scaleEffortFor("xhigh"));
    try testing.expectEqualStrings("\"max\"", scaleEffortFor("max"));
}

test "renderChatTemplate: the largest served conversation renders on every served template within a quarter of each limit" {
    // About 8 MB of messages over 4,000 turns, as a 1M-token context holds: the engine's limits must refuse
    // hostile templates, never an honest prompt this size.
    const a = testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(a);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const turns = 4000;
    var msgs = std.ArrayList(Message).empty;
    const rep = struct {
        fn f(al: std.mem.Allocator, unit: []const u8, n: usize) ![]const u8 {
            const out = try al.alloc(u8, unit.len * n);
            for (0..n) |k| @memcpy(out[k * unit.len ..][0..unit.len], unit);
            return out;
        }
    }.f;
    const ask = try rep(arena, "The numbers moved again, and I want the reason. ", 15);
    const plan = try rep(arena, "I should read the file first, then compare. ", 12);
    const says = try rep(arena, "The figures differ in the second table. ", 8);
    const rows = try rep(arena, "| a | b | c |\n", 6);
    try msgs.append(arena, .{ .role = "system", .content = try rep(arena, "You are a careful assistant. ", 40) });
    var text_bytes: usize = 0;
    for (0..turns) |i| {
        const user = try std.fmt.allocPrint(arena, "Turn {d}: please look at the report and tell me what changed. {s}", .{ i, ask });
        const thought = try std.fmt.allocPrint(arena, "Turn {d}: the user wants the change explained. {s}", .{ i, plan });
        const answer = try std.fmt.allocPrint(arena, "Reading the report for turn {d}. {s}", .{ i, says });
        const call_id = try std.fmt.allocPrint(arena, "call_{d}", .{i});
        const calls = try arena.alloc(ToolCall, 1);
        calls[0] = .{ .id = call_id, .name = "read_file", .arguments = try std.fmt.allocPrint(arena, "{{\"path\": \"reports/turn_{d}.md\", \"range\": {{\"start\": {d}, \"end\": {d}}}, \"follow\": [\"tables\", \"notes\"]}}", .{ i, i, i + 40 }) };
        const result = try std.fmt.allocPrint(arena, "reports/turn_{d}.md: {s}", .{ i, rows });
        try msgs.append(arena, .{ .role = "user", .content = user });
        try msgs.append(arena, .{ .role = "assistant", .content = answer, .reasoning_content = thought, .tool_calls = calls });
        try msgs.append(arena, .{ .role = "tool", .content = result, .tool_call_id = call_id });
        text_bytes += user.len + thought.len + answer.len + calls[0].arguments.len + result.len;
    }
    try testing.expect(text_bytes > 7_000_000);
    const tools =
        \\[{"type": "function", "function": {"name": "read_file", "description": "Read a file of the workspace.",
        \\  "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "range": {"type": "object",
        \\  "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}}}, "follow": {"type": "array",
        \\  "items": {"type": "string", "enum": ["tables", "notes", "links"]}}}, "required": ["path"]}}},
        \\ {"type": "function", "function": {"name": "search", "description": "Search the workspace.",
        \\  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]
    ;
    const Seen = struct {
        worst: template_engine.Usage = .{},
        calls: usize = 0,
        failed: usize = 0,
        fn see(ctx: *anyopaque, call: EngineCall) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
            if (call.rendered == null) self.failed += 1;
            self.worst.steps = @max(self.worst.steps, call.usage.steps);
            self.worst.memory = @max(self.worst.memory, call.usage.memory);
            self.worst.largest_string = @max(self.worst.largest_string, call.usage.largest_string);
        }
    };
    const qwen = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    // The late-system form exists while the Qwen arch is built.
    const qwen_late = if (arch_fx.qwen_model_type) |m| try arch_chat.lateSystemTemplate(m, a, qwen) else null;
    defer if (qwen_late) |t| a.free(t);
    const templates = [_]?[]const u8{ qwen, qwen_late, @embedFile("fixtures/glm53_chat_template.jinja"), @embedFile("fixtures/dsv41_chat_template.jinja"), @embedFile("fixtures/mimo_v26_chat_template.jinja") };
    const limits = template_engine.Limits{};
    for (templates) |maybe| {
        const tpl = maybe orelse continue;
        var seen = Seen{};
        engine_spy = .{ .ctx = &seen, .see = Seen.see };
        defer engine_spy = null;
        var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const out = try renderChatTemplate(a, msgs.items, &config, tools, null, true, null, false);
        defer a.free(out);
        try testing.expect(out.len > text_bytes / 2);
        try testing.expect(seen.calls > 0);
        try testing.expectEqual(@as(usize, 0), seen.failed);
        try testing.expect(seen.worst.steps < limits.max_steps / 4);
        try testing.expect(seen.worst.memory < limits.max_memory / 4);
        try testing.expect(seen.worst.largest_string < limits.max_string / 4);
    }
}

test "renderChatTemplate: a tool schema nested 120 levels under $defs renders on every served template" {
    // Honest schemas nest a few dozen levels; each level is two JSON levels, well inside the engine's 512.
    const a = testing.allocator;
    var schema = std.ArrayList(u8).empty;
    defer schema.deinit(a);
    const levels = 120;
    try schema.appendSlice(a, "[{\"type\": \"function\", \"function\": {\"name\": \"plan\", \"description\": \"Plan.\", \"parameters\": {\"type\": \"object\", \"$defs\": {\"Node\": ");
    for (0..levels) |k| try schema.print(a, "{{\"type\": \"object\", \"properties\": {{\"level{d}\": ", .{k});
    try schema.appendSlice(a, "{\"type\": \"string\"}");
    for (0..levels) |_| try schema.appendSlice(a, "}}");
    try schema.appendSlice(a, "}, \"properties\": {\"root\": {\"$ref\": \"#/$defs/Node\"}}}}}]");
    const qwen = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    const templates = [_][]const u8{ qwen, @embedFile("fixtures/glm53_chat_template.jinja"), @embedFile("fixtures/dsv41_chat_template.jinja"), @embedFile("fixtures/mimo_v26_chat_template.jinja") };
    const msgs = [_]Message{.{ .role = "user", .content = "Plan it." }};
    for (templates) |tpl| {
        var config = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = a };
        const out = try renderChatTemplate(a, &msgs, &config, schema.items, null, true, null, false);
        defer a.free(out);
        try testing.expect(std.mem.indexOf(u8, out, "level119") != null);
    }
}

test "renderChatTemplate: a continuation the template drops follows the generation prompt" {
    const a = testing.allocator;
    const m = 50_000;
    const hist = try a.alloc(u8, 2 * m);
    defer a.free(hist);
    @memset(hist, 'a');
    const partial = try std.mem.concat(a, u8, &.{ hist[0..m], "<think>z</think>x" });
    defer a.free(partial);
    const msgs = [_]Message{ .{ .role = "user", .content = hist }, .{ .role = "assistant", .content = partial } };
    var config = ChatConfig{ .chat_template = @embedFile("fixtures/glm53_chat_template.jinja"), .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = a };
    const out = try renderChatTemplate(a, &msgs, &config, null, null, true, null, true);
    defer a.free(out);
    try testing.expect(std.mem.endsWith(u8, out, partial));
    try testing.expect(std.mem.indexOf(u8, out, hist) != null);
}

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
    const p1 = try formatChat(a, tok, &first, cfg, tools, null, thinking, null, false);
    defer a.free(p1);
    const gen = try tok.encode(a, raw);
    errdefer a.free(gen);

    const split = think_tags.splitThinkBlock(raw, true, thinking);
    const parsed = if (tools) |tj| try tool_schema.parseForRequest(a, raw, tj, .{ .allow_parallel = true, .autocorrect = true, .dialect = try callDialect(a, cfg) }) else null;
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
    const p2 = try formatChat(a, tok, &second, cfg, tools, null, thinking, null, false);
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
    /// Past the closer, short of a call value the schema typed: the template reprints it (`tojson` spaces an array).
    until_typed_value,
    /// The turn is not re-rendered as it was generated.
    none,
};

pub const ReplayCase = struct { name: []const u8, tools: ?[]const u8, thinking: bool, raw: []const u8, reuse: NextReuse };

test "templateSettings: every family's template reads the same extra context for every effort word" {
    // One line per template x thinking x effort, FNV-1a over all of them: a changed byte in any context fails this.
    const a = testing.allocator;
    var out = std.ArrayList(u8).empty;
    defer out.deinit(a);
    const tpls = [_]struct { n: []const u8, t: []const u8 }{
        .{ .n = "qwen", .t = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja") },
        .{ .n = "dsv", .t = @embedFile("fixtures/dsv41_chat_template.jinja") },
        .{ .n = "glm", .t = @embedFile("fixtures/glm53_chat_template.jinja") },
        .{ .n = "mimo", .t = @embedFile("fixtures/mimo_v26_chat_template.jinja") },
        .{ .n = "plain", .t = "{{ reasoning_effort }}" },
    };
    const efforts = [_]?[]const u8{ null, "none", "minimal", "low", "medium", "high", "xhigh", "max", "62", "100" };
    for (tpls) |tp| for ([_]bool{ true, false }) |th| for (efforts) |e| {
        var cfg = ChatConfig{ .chat_template = tp.t, .bos_token = "<b>", .eos_token = "<e>", .add_bos_token = false, .allocator = a, .preserve_thinking = true };
        const r = try extraJsonForTest(a, &cfg, th, e);
        defer a.free(r);
        const line = try std.fmt.allocPrint(a, "{s}|{}|{s}|{s}\n", .{ tp.n, th, e orelse "~", r });
        defer a.free(line);
        try out.appendSlice(a, line);
    };
    try testing.expectEqual(@as(u64, 0xb1ce8d4293138d42), std.hash.Fnv1a_64.hash(out.items));
}

test "chat: what the server asks of each served template is what searching its text answered" {
    const qwen = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");
    const qwen_late = if (arch_fx.qwen_model_type) |m| try arch_chat.lateSystemTemplate(m, testing.allocator, qwen) else null;
    defer if (qwen_late) |t| testing.allocator.free(t);
    const served = [_]?[]const u8{ qwen, qwen_late, @embedFile("fixtures/glm53_chat_template.jinja"), @embedFile("fixtures/dsv41_chat_template.jinja"), @embedFile("fixtures/mimo_v26_chat_template.jinja") };
    const added = try std.json.parseFromSlice(std.json.Value, testing.allocator, @embedFile("fixtures/chat_added_tokens.json"), .{});
    defer added.deinit();
    for (served) |maybe| {
        const text = maybe orelse continue;
        var compiled = try Tpl.init(testing.allocator, text, false);
        defer compiled.deinit();
        const f = &compiled.tree.ok.facts;
        const has = struct {
            fn search(t: []const u8, word: []const u8) bool {
                return std.mem.indexOf(u8, t, word) != null;
            }
        }.search;
        try testing.expectEqual(has(text, "preserve_thinking"), asks.preserveThinking(f));
        try testing.expectEqual(has(text, "</think>"), asks.thinkSpan(f));
        try testing.expectEqual(has(text, "tools"), asks.tools(f));
        try testing.expectEqual(has(text, "'tool'") or has(text, "\"tool\""), asks.toolRole(f));
        try testing.expectEqual(has(text, "reasoning_effort"), asks.effort(f));
        try testing.expectEqual(has(text, "'xhigh'"), asks.effortLadder(f));
        try testing.expectEqual(has(text, "thinking_mode"), asks.thinkingMode(f));
        try testing.expectEqual(has(text, "effective_reasoning_effort") and has(text, "[gMASK]<sop>"), asks.effortWords(f));
        try testing.expectEqual(has(text, "enable_thinking"), asks.enableThinking(f));
        // The reserved-token mask keeps a special the template may print: every added token of every served
        // tokenizer is kept exactly where the search found it.
        for (added.value.object.values()) |pack| for (pack.object.get("added").?.array.items) |row| {
            const special = row.array.items[0].string;
            try testing.expectEqual(has(text, special), f.emits(special));
        };
        try testing.expectEqual(has(text, "<|channel>"), asks.channels(f));
        try testing.expectEqualStrings(if (has(text, "<think")) "<think>" else "(none)", asks.thinkOpener(f) orelse "(none)");
        try testing.expectEqual(has(text, "enable_thinking") or has(text, "<think>") or has(text, "thought") or has(text, "<|channel>"), asks.reasoning(f));
    }
}

test "chat: a word the server asks about counts only where the template uses it, never in a comment" {
    const a = testing.allocator;
    // `tools` only in a comment: the template does not render the tool list, so the server injects it.
    {
        var config = ChatConfig{ .chat_template = "{# tools #}{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const msgs = [_]Message{.{ .role = "user", .content = "q" }};
        const tools = "[{\"type\": \"function\", \"function\": {\"name\": \"lookup_weather\", \"description\": \"d\", \"parameters\": {\"type\": \"object\", \"properties\": {}}}}]";
        const out = try renderChatTemplate(a, &msgs, &config, tools, null, true, null, false);
        defer a.free(out);
        try testing.expect(std.mem.indexOf(u8, out, "lookup_weather") != null);
    }
    // 'tool' only in a comment: no tool branch, so the probe finds the dropped tool turn and the server rewrites it.
    {
        var config = ChatConfig{ .chat_template = "{# 'tool' #}{% for m in messages %}{% if m.role in ['user', 'assistant'] %}[{{ m.role }}:{{ m.content }}]{% endif %}{% endfor %}", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const msgs = [_]Message{ .{ .role = "user", .content = "q" }, .{ .role = "tool", .content = "RESULT_TEXT" } };
        const out = try renderChatTemplate(a, &msgs, &config, null, null, true, null, false);
        defer a.free(out);
        try testing.expect(std.mem.indexOf(u8, out, "RESULT_TEXT") != null);
    }
    // Every effort dialect's word only in a comment: the template gets the default effort and no dialect's keys.
    {
        var config = ChatConfig{ .chat_template = "{# 'xhigh' thinking_mode effective_reasoning_effort [gMASK]<sop> preserve_thinking #}{{ reasoning_effort }}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .preserve_thinking = true };
        const extra = try extraJsonForTest(a, &config, true, "medium");
        defer a.free(extra);
        try testing.expectEqualStrings("{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}", extra);
    }
    // `enable_thinking` only in a comment: the ChatML fallback opens no think block.
    {
        var config = ChatConfig{ .chat_template = "{# enable_thinking #}{{ raise_exception('no turn format') }}", .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const msgs = [_]Message{.{ .role = "user", .content = "q" }};
        const out = try renderChatTemplate(a, &msgs, &config, null, null, true, null, false);
        defer a.free(out);
        try testing.expect(std.mem.endsWith(u8, out, "<|im_start|>assistant\n"));
    }
}

test "chat: preserve_thinking, a think closer and reasoning_effort named only in a comment are not used" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{# preserve_thinking </think> reasoning_effort #}{% for m in messages %}{{ m.content }}{% endfor %}\"}" });
    var cc = try loadChatConfig(io, testing.allocator, root, null);
    defer cc.deinit();
    try testing.expect(!templateTakesPreserveThinking(&cc));
    try testing.expect(!templateHasThinkSpan(&cc));
    try testing.expect(!templateConsumesEffort(&cc));
}

test "loadChatConfig: a template that does not compile fails the load" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    // A ChatML model too: its ChatML fallback covers a template that raises, never one that cannot be read.
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{% for m in %}\", \"eos_token\": \"<|im_end|>\"}" });
    try testing.expectError(error.ChatTemplateInvalid, loadChatConfig(io, testing.allocator, root, null));
}

test "loadChatConfig: a loaded template is compiled once, and every render of it reads the compiled tree" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}\", \"eos_token\": \"<|im_end|>\"}" });
    var cc = try loadChatConfig(io, testing.allocator, root, null);
    defer cc.deinit();
    try testing.expect(cc.compiled != null);
    const Seen = struct {
        calls: usize = 0,
        per_call: usize = 0,
        fn see(ctx: *anyopaque, call: EngineCall) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
            if (!call.compiled_at_load) self.per_call += 1;
        }
    };
    var seen = Seen{};
    engine_spy = .{ .ctx = &seen, .see = Seen.see };
    defer engine_spy = null;
    const msgs = [_]Message{ .{ .role = "user", .content = "a" }, .{ .role = "system", .content = "late" }, .{ .role = "tool", .content = "t" } };
    const out = try renderChatTemplate(testing.allocator, &msgs, &cc, null, null, true, null, false);
    defer testing.allocator.free(out);
    try testing.expect(seen.calls > 1);
    try testing.expectEqual(@as(usize, 0), seen.per_call);
}

test "loadChatConfig: a tokenizer_config.json of valid JSON but not an object fails by name, for a load and a look" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    for ([_][]const u8{ "[]", "7", "\"t\"", "null" }) |body| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = body });
        try testing.expectError(error.InvalidTokenizerConfig, loadChatConfig(io, testing.allocator, root, null));
        try testing.expectError(error.InvalidTokenizerConfig, peekChatConfig(io, testing.allocator, root, null));
    }
}

test "loadChatConfig: the model type the load parsed picks the pin, and config.json is not read again" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    // A config.json too large or broken to read once decided the template; now the caller's model type does.
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "[]" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\":\"own\"}" });
    var pinned = try loadChatConfig(io, testing.allocator, root, "deepseek_v41");
    defer pinned.deinit();
    // A build without the arch has no pin for its model type: the checkpoint's own template stands.
    const want: []const u8 = if (arch_chat.pinOf("deepseek_v41") != null) @embedFile("fixtures/dsv41_chat_template.jinja") else "own";
    try testing.expectEqualStrings(want, pinned.chat_template);
    var own = try loadChatConfig(io, testing.allocator, root, null);
    defer own.deinit();
    try testing.expectEqualStrings("own", own.chat_template);
    try testing.expect(own.image_placeholder == null);
}

test "loadChatConfig: a pack's chat_template.jinja outranks the copy in tokenizer_config.json, and a pin outranks both" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\":\"inline copy\",\"eos_token\":\"<e>\"}" });
    var inline_only = try loadChatConfig(io, testing.allocator, root, null);
    defer inline_only.deinit();
    try testing.expectEqualStrings("inline copy", inline_only.chat_template);
    try tmp.dir.writeFile(io, .{ .sub_path = "chat_template.jinja", .data = "file copy" });
    var both = try loadChatConfig(io, testing.allocator, root, null);
    defer both.deinit();
    try testing.expectEqualStrings("file copy", both.chat_template);
    var peeked = try peekChatConfig(io, testing.allocator, root, null);
    defer peeked.deinit();
    try testing.expectEqualStrings("file copy", peeked.chat_template);
    if (arch_chat.pinOf("deepseek_v41") != null) {
        var pinned = try loadChatConfig(io, testing.allocator, root, "deepseek_v41");
        defer pinned.deinit();
        try testing.expectEqualStrings(@embedFile("fixtures/dsv41_chat_template.jinja"), pinned.chat_template);
    }
}

test "loadChatConfig: bos and eos in HF's AddedToken form read their content, and null is no token" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data =
        \\{"chat_template": "x", "bos_token": null, "add_bos_token": null,
        \\ "eos_token": {"__type": "AddedToken", "content": "<|im_end|>", "lstrip": false}}
    });
    var cc = try loadChatConfig(io, testing.allocator, root, null);
    defer cc.deinit();
    try testing.expectEqualStrings("<|im_end|>", cc.eos_token.?);
    try testing.expect(cc.bos_token == null);
    try testing.expect(!cc.add_bos_token);
}

test "loadChatConfig: a key of tokenizer_config.json in a shape the load cannot read fails by name" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    for ([_][]const u8{
        "{\"chat_template\": 7}",               "{\"bos_token\": 5}",                 "{\"eos_token\": []}",
        "{\"eos_token\": {\"content\": 5}}", "{\"eos_token\": {\"__type\": \"AddedToken\"}}", "{\"add_bos_token\": \"yes\"}",
    }) |body| {
        try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = body });
        try testing.expectError(error.InvalidTokenizerConfig, loadChatConfig(io, testing.allocator, root, null));
        try testing.expectError(error.InvalidTokenizerConfig, peekChatConfig(io, testing.allocator, root, null));
    }
}

fn loadAndFree(a: std.mem.Allocator, dir: []const u8, model_type: ?[]const u8) !void {
    var cc = try loadChatConfig(testing.io, a, dir, model_type);
    cc.deinit();
}

test "loadChatConfig: a read that runs out of memory at any allocation frees all it made" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{% for m in messages %}{{ m.content }}{% endfor %}\", \"bos_token\": \"<b>\", \"eos_token\": \"<e>\"}" });
    try testing.checkAllAllocationFailures(testing.allocator, loadAndFree, .{ root, @as(?[]const u8, null) });
}

test "ChatConfig.compile: a second compile replaces the first and frees it" {
    var cc: ChatConfig = .{ .chat_template = try testing.allocator.dupe(u8, "{{ 1 }}"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    defer cc.deinit();
    try cc.compile();
    try cc.compile();
    try testing.expect(cc.compiled != null);
}

test "loadChatConfig: the checkpoint's own template is read, and a model type with a pin gets its pinned template and image atom" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    for ([_][]const u8{ "deepseek_v41", "qwen4_exp", "mimo_v2" }) |model_type| {
        const dir = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ root, model_type });
        defer testing.allocator.free(dir);
        try tmp.dir.createDirPath(io, model_type);
        const tok = try std.fmt.allocPrint(testing.allocator, "{s}/tokenizer_config.json", .{model_type});
        defer testing.allocator.free(tok);
        try tmp.dir.writeFile(io, .{ .sub_path = tok, .data = "{\"chat_template\":\"own\",\"bos_token\":\"<b>\"}" });
        var cc = try loadChatConfig(io, testing.allocator, dir, model_type);
        defer cc.deinit();
        if (arch_chat.pinOf(model_type) != null) {
            try testing.expectEqualStrings(@embedFile("fixtures/dsv41_chat_template.jinja"), cc.chat_template);
            try testing.expectEqualStrings("<\u{ff5c}deepseek_image\u{ff5c}>", cc.image_placeholder.?);
        } else {
            try testing.expectEqualStrings("own", cc.chat_template);
            try testing.expect(cc.image_placeholder == null);
        }
        try testing.expectEqualStrings("<b>", cc.bos_token.?);
    }
}

test "reasoning is advertised only when the template reads enable_thinking or prints a think opener, never from a comment" {
    const a = testing.allocator;
    const cases = [_]struct { tpl: []const u8, want: bool }{
        .{ .tpl = "{# enable_thinking <think> thought <|channel> #}{{ messages }}", .want = false },
        .{ .tpl = "{% if 'thought' == x %}{% endif %}{{ messages }}", .want = false },
        .{ .tpl = "{% if enable_thinking %}{% endif %}{{ messages }}", .want = true },
        .{ .tpl = "{{ messages }}<think>", .want = true },
        .{ .tpl = "{{ '<|channel>' ~ 'thought' }}", .want = true },
    };
    for (cases) |c| {
        var cc = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
        try cc.compile();
        defer cc.dropCompiled();
        try testing.expectEqual(c.want, templateSupportsThinking(&cc));
    }
}

test "the think opener and the channel protocol come from what the template prints, never from a comment or a comparison" {
    const a = testing.allocator;
    const cases = [_]struct { tpl: []const u8, opener: ?[]const u8, channels: bool }{
        .{ .tpl = "{{ '<think>' }} <think>", .opener = "<think>", .channels = false },
        .{ .tpl = "{{ '<think:opensource>' }}", .opener = null, .channels = false },
        .{ .tpl = "<think> <think:opensource>", .opener = null, .channels = false },
        .{ .tpl = "<think:{{ suffix }}>", .opener = null, .channels = false },
        .{ .tpl = "assistant", .opener = null, .channels = false },
        .{ .tpl = "{{ '<think' ~ '>' }}", .opener = null, .channels = false },
        .{ .tpl = "{{ '<|channel>' }}thought", .opener = null, .channels = true },
        .{ .tpl = "{# <think:opensource> <|channel> #}<think>", .opener = "<think>", .channels = false },
        .{ .tpl = "{% if c == '<think:x>' or c is sameas '<|channel>' %}{% endif %}<think>", .opener = "<think>", .channels = false },
        // A method's argument counts as printed (Facts errs toward output), so this second spelling makes two.
        .{ .tpl = "{% if c.startswith('<think:x>') %}{% endif %}<think>", .opener = null, .channels = false },
    };
    for (cases) |c| {
        var cc = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
        try cc.compile();
        defer cc.dropCompiled();
        if (c.opener) |want| try testing.expectEqualStrings(want, templateThinkOpener(&cc) orelse "(none)") else try testing.expect(templateThinkOpener(&cc) == null);
        try testing.expectEqual(c.channels, templateUsesChannels(&cc));
    }
}

test "chat: a probe renders once per template and settings, and answers as a fresh render does" {
    const a = testing.allocator;
    const Count = struct {
        calls: usize = 0,
        fn see(ctx: *anyopaque, _: EngineCall) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
        }
    };
    const served = [_][]const u8{ @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), @embedFile("fixtures/glm53_chat_template.jinja"), @embedFile("fixtures/dsv41_chat_template.jinja"), @embedFile("fixtures/mimo_v26_chat_template.jinja") };
    const efforts = [_]?[]const u8{ null, "none", "low", "medium", "high", "xhigh", "max", "a \"quoted\\ word" };
    for (served) |text| {
        const compiled = try Compiled.init(a, text, true, arch_fx.qwen_model_type);
        defer compiled.deinit(a);
        var cc = ChatConfig{ .chat_template = text, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .compiled = compiled };
        for ([_]?bool{ null, true, false }) |preserve| for ([_]bool{ true, false }) |thinking| for (efforts) |effort| {
            cc.preserve_thinking = preserve;
            const settings = try templateSettings(a, &cc, thinking, effort);
            const extra = &settings;
            for ([_]?*const Tpl{ &compiled.main, if (compiled.late_system) |*l| l else null }) |maybe| {
                const tpl = maybe orelse continue;
                for ([_]Probe{ .tool_content, .late_system, .call, .call_empty, .answer }) |probe| {
                    const fresh = try renderProbeFresh(a, tpl, probe, extra);
                    defer if (fresh) |f| a.free(f);
                    var count = Count{};
                    engine_spy = .{ .ctx = &count, .see = Count.see };
                    defer engine_spy = null;
                    for (0..2) |_| {
                        const got = try templateProbeRender(a, tpl, probe, extra);
                        defer if (got) |g| a.free(g);
                        try testing.expectEqual(fresh == null, got == null);
                        if (fresh) |f| try testing.expectEqualStrings(f, got.?);
                    }
                    try testing.expect(count.calls <= 1);
                }
            }
        };
    }
}

test "chat: a probe memo full of client words starts over and still answers right" {
    const a = testing.allocator;
    var tpl = try Tpl.init(a, "[gMASK]<sop>{% set effective_reasoning_effort = reasoning_effort %}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}[{{ effective_reasoning_effort }}]", true);
    defer tpl.deinit();
    var word: [16]u8 = undefined;
    for (0..ProbeMemo.cap * 2 + 3) |i| {
        const extra = &Settings{ .enable_thinking = true, .reasoning_effort = .{ .word = std.fmt.bufPrint(&word, "w{d}", .{i}) catch unreachable } };
        const fresh = (try renderProbeFresh(a, &tpl, .late_system, extra)).?;
        defer a.free(fresh);
        const got = (try templateProbeRender(a, &tpl, .late_system, extra)).?;
        defer a.free(got);
        try testing.expectEqualStrings(fresh, got);
        try testing.expect(tpl.memo.entries.items.len <= ProbeMemo.cap);
    }
}

test "chat: probes of one compiled template on many threads at once answer as a fresh render does" {
    const a = testing.allocator;
    const compiled = try Compiled.init(a, @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), true, arch_fx.qwen_model_type);
    defer compiled.deinit(a);
    const extras = [_]Settings{
        .{ .enable_thinking = true, .reasoning_effort = .{ .word = "medium" } },
        .{ .enable_thinking = false, .reasoning_effort = .{ .word = "low" } },
        .{ .enable_thinking = true, .reasoning_effort = .{ .word = "xhigh" }, .preserve_thinking = true },
    };
    var fresh: [extras.len * 3]?[]u8 = undefined;
    for (&fresh, 0..) |*f, i| f.* = try renderProbeFresh(a, &compiled.main, @enumFromInt(i % 3), &extras[i / 3]);
    defer for (fresh) |f| if (f) |r| a.free(r);
    const Worker = struct {
        fn run(tpl: *const Tpl, want: []const ?[]u8, ok: *bool) void {
            @setRuntimeSafety(true);
            ok.* = false;
            for (0..40) |n| {
                const i = n % want.len;
                const got = templateProbeRender(std.heap.smp_allocator, tpl, @enumFromInt(i % 3), &extras[i / 3]) catch return;
                defer if (got) |g| std.heap.smp_allocator.free(g);
                if ((got == null) != (want[i] == null)) return;
                if (got) |g| if (!std.mem.eql(u8, g, want[i].?)) return;
            }
            ok.* = true;
        }
    };
    var oks: [8]bool = @splat(false);
    var threads: [8]std.Thread = undefined;
    for (&threads, &oks) |*t, *ok| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &compiled.main, &fresh, ok });
    for (threads) |t| t.join();
    for (oks) |ok| try testing.expect(ok);
}

test "loadChatConfig: a Flash-Next template without the late-system branch loads, folds a late system turn, and says so once" {
    if (!arch_chat.editsLateSystem(arch_fx.qwen_model_type orelse return error.SkipZigTest)) return error.SkipZigTest;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{% for m in messages %}{% if m.role == 'system' and not loop.first %}{{ raise_exception('System message must be at the beginning.') }}{% endif %}<|im_start|>{{ m.role }}\\n{{ m.content }}<|im_end|>\\n{% endfor %}\", \"eos_token\": \"<|im_end|>\"}" });
    const log_path = try std.fmt.allocPrint(testing.allocator, "{s}/s.log", .{root});
    defer testing.allocator.free(log_path);
    try log.openFile(log_path, 0);
    var cc = loadChatConfig(io, testing.allocator, root, arch_fx.qwen_model_type) catch |e| {
        log.closeFile();
        return e;
    };
    defer cc.deinit();
    const msgs = [_]Message{ .{ .role = "system", .content = "rules" }, .{ .role = "user", .content = "q" }, .{ .role = "system", .content = "late note" } };
    const out = try renderChatTemplate(testing.allocator, &msgs, &cc, null, null, true, null, false);
    defer testing.allocator.free(out);
    log.closeFile();
    try testing.expect(std.mem.indexOf(u8, out, "late note") != null);
    try testing.expect(std.mem.indexOf(u8, out, "<|im_start|>system\nrules") != null);
    const logged = try tmp.dir.readFileAlloc(io, "s.log", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(logged);
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, logged, "[load] chat template: no late-system branch to edit"));
}

test "chat: Flash-Next's late-system edit finds the branch at any spacing, and renders the turn as the stock edit does" {
    const engine = template_engine;
    const a = testing.allocator;
    const stock = arch_fx.qwen_template orelse return error.SkipZigTest;
    const branch =
        \\{%- if message.role == "system" %}
        \\        {%- if not loop.first %}
        \\            {{- raise_exception('System message must be at the beginning.') }}
        \\        {%- endif %}
    ;
    const respaced_branch =
        \\{%-   if message.role=="system"   %}
        \\    {%- if not   loop.first%}{{-raise_exception( 'System message must be at the beginning.' )}}
        \\    {%- endif %}
    ;
    const respaced = try std.mem.replaceOwned(u8, a, stock, branch, respaced_branch);
    defer a.free(respaced);
    try testing.expect(!std.mem.eql(u8, respaced, stock));
    const stock_late = (try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, stock)).?;
    defer a.free(stock_late);
    const respaced_late = (try arch_chat.lateSystemTemplate(arch_fx.qwen_model_type.?, a, respaced)) orelse return error.LateSystemBranchNotFound;
    defer a.free(respaced_late);
    const late = "[{\"role\":\"system\",\"content\":\"rules\"},{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"system\",\"content\":\"late note\"},{\"role\":\"user\",\"content\":\"q2\"}]";
    const plain = "[{\"role\":\"system\",\"content\":\"rules\"},{\"role\":\"user\",\"content\":\"q\"}]";
    var outs: [4][]const u8 = undefined;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    for ([_][]const u8{ stock_late, respaced_late, stock_late, respaced_late }, 0..) |text, i| {
        var diag: engine.Diagnostic = .{};
        var t = try engine.Template.compile(a, text, &diag);
        defer t.deinit();
        const json = engine.JsonContext{ .messages = if (i < 2) late else plain, .extra = "{\"enable_thinking\":true}" };
        outs[i] = try t.render(arena.allocator(), .{ .context = json.context(), .add_generation_prompt = true }, &diag);
    }
    try testing.expect(std.mem.indexOf(u8, outs[0], "<|im_start|>system\nlate note<|im_end|>\n") != null);
    try testing.expectEqualStrings(outs[0], outs[1]);
    try testing.expectEqualStrings(outs[2], outs[3]);
}

test "chat: a Flash-Next template saved with CRLF line ends gets the same late-system edit and loads" {
    const a = testing.allocator;
    const lf = arch_fx.qwen_template orelse return error.SkipZigTest;
    const crlf = try std.mem.replaceOwned(u8, a, lf, "\n", "\r\n");
    defer a.free(crlf);
    const msgs = [_]Message{ .{ .role = "system", .content = "rules" }, .{ .role = "user", .content = "q" }, .{ .role = "system", .content = "late note" }, .{ .role = "user", .content = "q2" } };
    var lf_cfg = ChatConfig{ .chat_template = lf, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
    try lf_cfg.compile();
    defer lf_cfg.dropCompiled();
    var crlf_cfg = ChatConfig{ .chat_template = crlf, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = arch_fx.qwen_model_type };
    try crlf_cfg.compile();
    defer crlf_cfg.dropCompiled();
    try testing.expect(crlf_cfg.compiled.?.late_system != null);
    const want = try renderChatTemplate(a, &msgs, &lf_cfg, null, null, true, null, false);
    defer a.free(want);
    const got = try renderChatTemplate(a, &msgs, &crlf_cfg, null, null, true, null, false);
    defer a.free(got);
    try testing.expectEqualStrings(want, got);
}

test "settings: reading a stub's chat template logs nothing" {
    if (!arch_chat.editsLateSystem(arch_fx.qwen_model_type orelse return error.SkipZigTest)) return error.SkipZigTest;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{\"chat_template\": \"{% for m in messages %}{{ m.content }}{% endfor %}\"}" });
    const log_path = try std.fmt.allocPrint(testing.allocator, "{s}/s.log", .{root});
    defer testing.allocator.free(log_path);
    try log.openFile(log_path, 0);
    var cc = peekChatConfig(io, testing.allocator, root, arch_fx.qwen_model_type) catch |e| {
        log.closeFile();
        return e;
    };
    log.closeFile();
    defer cc.deinit();
    try testing.expect(cc.compiled != null);
    const logged = try tmp.dir.readFileAlloc(io, "s.log", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(logged);
    try testing.expectEqualStrings("", logged);
}

test "chat: a template built by hand and never compiled has no facts to give, rather than empty ones" {
    const a = testing.allocator;
    var cc = ChatConfig{ .chat_template = "{% if enable_thinking %}<think>{% endif %}", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
    try testing.expectError(error.TemplateNotCompiled, cc.compiledFacts());
    try cc.compile();
    defer cc.dropCompiled();
    try testing.expect(asks.enableThinking(try cc.compiledFacts()));
    // A template swapped in after the compile: the tree is another text's, so it answers nothing and renders nothing.
    const swapped_from = cc.chat_template;
    cc.chat_template = "[{{ messages[0].content }}]";
    try testing.expectError(error.TemplateNotCompiled, cc.compiledFacts());
    const msgs = [_]Message{.{ .role = "user", .content = "q" }};
    const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, false);
    defer a.free(out);
    try testing.expectEqualStrings("[q]", out);
    cc.chat_template = swapped_from;
    const empty = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a };
    try testing.expect(!asks.enableThinking(try empty.compiledFacts()));
    try testing.expect(empty.printedFacts() == null);
}

test "chat: Qwen's late-system edit reaches only a Qwen-typed config, never another arch's or an untyped one" {
    const a = testing.allocator;
    const qwen = arch_fx.qwen_template orelse return error.SkipZigTest;
    const msgs = [_]Message{ .{ .role = "system", .content = "rules" }, .{ .role = "user", .content = "q" }, .{ .role = "system", .content = "late note" }, .{ .role = "user", .content = "q2" } };
    for ([_]?[]const u8{ null, arch_fx.glm_model_type, arch_fx.qwen_model_type }, 0..) |model_type, i| {
        if (i > 0 and model_type == null) continue;
        var cc = ChatConfig{ .chat_template = qwen, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a, .model_type = model_type };
        try cc.compile();
        defer cc.dropCompiled();
        const edits = i == 2;
        try testing.expectEqual(edits, cc.compiled.?.late_system != null);
        const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, false);
        defer a.free(out);
        try testing.expectEqual(edits, std.mem.indexOf(u8, out, "<|im_start|>system\nlate note<|im_end|>") != null);
        try testing.expectEqual(!edits, std.mem.indexOf(u8, out, "rules\n\nlate note") != null);
    }
}

test "chat: the shared Qwen test config places a late system turn as a Qwen load does" {
    const a = testing.allocator;
    var fixture = (try arch_fx.qwenChatConfig(a)) orelse return error.SkipZigTest;
    defer fixture.dropCompiled();
    try testing.expect(fixture.compiled.?.late_system != null);
    const msgs = [_]Message{
        .{ .role = "system", .content = "stable" },
        .{ .role = "user", .content = "u1" },
        .{ .role = "assistant", .content = "a1" },
        .{ .role = "system", .content = "late note" },
        .{ .role = "user", .content = "u2" },
    };
    const got = try renderChatTemplate(a, &msgs, &fixture, null, null, true, null, false);
    defer a.free(got);
    try testing.expect(std.mem.indexOf(u8, got, "<|im_start|>system\nlate note<|im_end|>") != null);
}

/// Fails the `fail_at`-th allocation once, then serves every other; `induced` says the run reached it.
const FailOnce = struct {
    child: std.mem.Allocator,
    fail_at: usize = std.math.maxInt(usize),
    count: usize = 0,
    induced: bool = false,

    fn allocator(self: *FailOnce) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn arm(self: *FailOnce, k: usize) void {
        self.* = .{ .child = self.child, .fail_at = k };
    }
    fn disarm(self: *FailOnce) void {
        self.fail_at = std.math.maxInt(usize);
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailOnce = @ptrCast(@alignCast(ctx));
        defer self.count += 1;
        if (self.count == self.fail_at) {
            self.induced = true;
            return null;
        }
        return self.child.rawAlloc(len, al, ra);
    }
    fn resize(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailOnce = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(mem, al, new_len, ra);
    }
    fn remap(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailOnce = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(mem, al, new_len, ra);
    }
    fn free(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        const self: *FailOnce = @ptrCast(@alignCast(ctx));
        self.child.rawFree(mem, al, ra);
    }
};

const sweep_plain_template = "{% for m in messages %}{% if m.role in ['user','system','assistant'] %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

const SweepRequest = struct { msgs: []const Message, tools: ?[]const u8, instr: ?[]const u8, thinking: bool, effort: ?[]const u8, cont: bool };

/// A request with tools, a forced call, a tool turn and a late system turn; the same without tools; a continuation.
fn sweepRequests() []const SweepRequest {
    const call = [_]ToolCall{.{ .id = "c1", .name = "get_weather", .arguments = "{\"city\": \"Paris and a long tail of words to pass thirty-two bytes\"}" }};
    const S = struct {
        const msgs_a = [_]Message{
            .{ .role = "system", .content = "You are a careful assistant who answers in full sentences, always." },
            .{ .role = "user", .content = "What is the weather in Paris today, and what should I wear?" },
            .{ .role = "assistant", .content = "", .tool_calls = &call, .reasoning_content = "I should call the weather tool for Paris first." },
            .{ .role = "tool", .content = "{\"temp\": 21, \"sky\": \"clear\", \"note\": \"a long tool result body here\"}", .tool_call_id = "c1" },
            .{ .role = "system", .content = "A late note from the operator, placed after the first turn on purpose." },
            .{ .role = "user", .content = "Thanks. And tomorrow? Please answer with <|im_end|> spelled out in text." },
        };
        const msgs_b = [_]Message{
            .{ .role = "user", .content = "Write the start of a story about a lighthouse keeper and a storm." },
            .{ .role = "assistant", .content = "The keeper climbed the stairs as the storm" },
        };
    };
    const tools = "[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";
    const reqs = comptime [_]SweepRequest{
        .{ .msgs = &S.msgs_a, .tools = tools, .instr = "You must call get_weather now.", .thinking = true, .effort = "high", .cont = false },
        .{ .msgs = &S.msgs_a, .tools = null, .instr = null, .thinking = false, .effort = null, .cont = false },
        .{ .msgs = &S.msgs_b, .tools = null, .instr = null, .thinking = true, .effort = "low", .cont = true },
    };
    return &reqs;
}

fn clearMemoAndCache(cc: *ChatConfig, tok: *Tokenizer) void {
    if (cc.compiled) |c| {
        c.main.memo.clear();
        if (c.late_system) |*l| l.memo.clear();
    }
    if (tok.segment_cache) |sc| sc.clear();
}

test "formatChat: running out of memory at any allocation is OutOfMemory or the same ids, cold or warm, and the next call gives them" {
    const segment_cache_mod = @import("segment_cache.zig");
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const qwen = arch_fx.qwen_template orelse return error.SkipZigTest;
    const Case = struct { tpl: []const u8, model_type: ?[]const u8 };
    for ([_]Case{ .{ .tpl = qwen, .model_type = arch_fx.qwen_model_type }, .{ .tpl = sweep_plain_template, .model_type = null } }) |c| {
        var fa = FailOnce{ .child = testing.allocator };
        const A = fa.allocator();
        const pt = try tokenizer_mod.packTokenizerForTests(A, arena, "qwen38_flash_next");
        var tok = pt.tok;
        defer tokenizer_mod.deinitTestTokenizer(&tok);
        var cc = ChatConfig{ .chat_template = c.tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = A, .model_type = c.model_type };
        try cc.compile();
        defer cc.dropCompiled();
        try tok.bindTemplate(cc.printedFacts(), pt.stops, &.{});
        try tok.enableSegmentCache(segment_cache_mod.default_budget);
        for (sweepRequests()) |r| {
            clearMemoAndCache(&cc, &tok);
            const want = try formatChat(testing.allocator, &tok, r.msgs, &cc, r.tools, r.instr, r.thinking, r.effort, r.cont);
            defer testing.allocator.free(want);
            for ([_]bool{ true, false }) |warm| {
                var k: usize = 0;
                while (true) : (k += 1) {
                    if (!warm) clearMemoAndCache(&cc, &tok);
                    fa.arm(k);
                    const got = formatChat(A, &tok, r.msgs, &cc, r.tools, r.instr, r.thinking, r.effort, r.cont);
                    fa.disarm();
                    if (got) |ids| {
                        defer A.free(ids);
                        try testing.expectEqualSlices(u32, want, ids);
                    } else |e| try testing.expectEqual(error.OutOfMemory, e);
                    const again = try formatChat(testing.allocator, &tok, r.msgs, &cc, r.tools, r.instr, r.thinking, r.effort, r.cont);
                    defer testing.allocator.free(again);
                    try testing.expectEqualSlices(u32, want, again);
                    if (!fa.induced) break;
                }
            }
        }
    }
}

test "renderChatSpans and forcedCallText: running out of memory at any allocation leaks nothing" {
    const Case = struct { tpl: ?[]const u8, model_type: ?[]const u8 };
    for ([_]Case{ .{ .tpl = arch_fx.qwen_template, .model_type = arch_fx.qwen_model_type }, .{ .tpl = @embedFile("fixtures/dsv41_chat_template.jinja"), .model_type = "deepseek_v41" }, .{ .tpl = arch_fx.glm_template, .model_type = arch_fx.glm_model_type }, .{ .tpl = sweep_plain_template, .model_type = null } }) |c| {
        var fa = FailOnce{ .child = testing.allocator };
        const A = fa.allocator();
        var cc = ChatConfig{ .chat_template = c.tpl orelse continue, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = A, .model_type = c.model_type };
        try cc.compile();
        defer cc.dropCompiled();
        for (sweepRequests()) |r| {
            var k: usize = 0;
            while (true) : (k += 1) {
                fa.arm(k);
                if (renderChatSpans(A, r.msgs, &cc, r.tools, r.instr, r.thinking, r.effort, r.cont)) |s| freeSpans(A, s) else |e| try testing.expectEqual(error.OutOfMemory, e);
                const fc = forcedCallText(A, &cc, "get_weather") catch |e| blk: {
                    try testing.expectEqual(error.OutOfMemory, e);
                    break :blk null;
                };
                if (fc) |f| {
                    var owned = f;
                    owned.deinit(A);
                }
                fa.disarm();
                if (!fa.induced) break;
            }
        }
    }
}

test "formatChat: threads on two models, with the memo restarting and the cache evicting, give the single-thread ids" {
    const gpa = std.heap.smp_allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const qwen = arch_fx.qwen_template orelse return error.SkipZigTest;
    const Model = struct { tok: Tokenizer, cc: ChatConfig };
    var models: [2]Model = undefined;
    const specs = [_]struct { pack: []const u8, tpl: []const u8, mt: ?[]const u8, eos: ?[]const u8 }{
        .{ .pack = "qwen38_flash_next", .tpl = qwen, .mt = arch_fx.qwen_model_type, .eos = "<|im_end|>" },
        .{ .pack = "glm53", .tpl = @embedFile("fixtures/glm53_chat_template.jinja"), .mt = arch_fx.glm_model_type, .eos = null },
    };
    for (&models, specs) |*m, s| {
        const pt = try tokenizer_mod.packTokenizerForTests(gpa, arena, s.pack);
        m.tok = pt.tok;
        m.cc = .{ .chat_template = s.tpl, .bos_token = null, .eos_token = s.eos, .add_bos_token = false, .allocator = gpa, .model_type = s.mt };
        try m.cc.compile();
        try m.tok.bindTemplate(m.cc.printedFacts(), pt.stops, &.{});
    }
    defer for (&models) |*m| {
        m.cc.dropCompiled();
        tokenizer_mod.deinitTestTokenizer(&m.tok);
    };
    const Job = struct { model: usize, req: usize, msgs: []const Message, effort: ?[]const u8, thinking: bool, want: []const u32 };
    var jobs = std.ArrayList(Job).empty;
    // Many distinct effort words restart the memo; distinct last turns fill and evict the cache.
    for (0..2) |mi| for (sweepRequests(), 0..) |r, ri| for (0..120) |w| {
        const effort: ?[]const u8 = if (w == 0) r.effort else try std.fmt.allocPrint(arena, "w{d}", .{w});
        const msgs = try arena.dupe(Message, r.msgs);
        msgs[msgs.len - 1].content = try std.fmt.allocPrint(arena, "{s} Question number {d}, asked again with more words around it.", .{ msgs[msgs.len - 1].content, w });
        const ids = formatChat(arena, &models[mi].tok, msgs, &models[mi].cc, r.tools, r.instr, w % 3 != 0, effort, r.cont) catch |e| switch (e) {
            error.ChatNeedsUserTurn, error.ChatTemplateUnusable => continue,
            else => return e,
        };
        try jobs.append(arena, .{ .model = mi, .req = ri, .msgs = msgs, .effort = effort, .thinking = w % 3 != 0, .want = ids });
    };
    const Shared = struct {
        models: *[2]Model,
        jobs: []const Job,
        bad: std.atomic.Value(usize) = .init(0),
        fn run(self: *@This(), seed: u64) void {
            var prng = std.Random.DefaultPrng.init(seed);
            const rnd = prng.random();
            for (0..400) |_| {
                const j = self.jobs[rnd.uintLessThan(usize, self.jobs.len)];
                const m = &self.models[j.model];
                const r = sweepRequests()[j.req];
                const ids = formatChat(std.heap.smp_allocator, &m.tok, j.msgs, &m.cc, r.tools, r.instr, j.thinking, j.effort, r.cont) catch {
                    _ = self.bad.fetchAdd(1, .monotonic);
                    continue;
                };
                defer std.heap.smp_allocator.free(ids);
                if (!std.mem.eql(u32, ids, j.want)) _ = self.bad.fetchAdd(1, .monotonic);
            }
        }
    };
    for ([_]usize{ 12 << 10, 64 << 10 }) |budget| {
        for (&models) |*m| try m.tok.enableSegmentCache(budget);
        var shared = Shared{ .models = &models, .jobs = jobs.items };
        var threads: [8]std.Thread = undefined;
        for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Shared.run, .{ &shared, @as(u64, i) * 7919 + budget });
        for (threads) |t| t.join();
        try testing.expectEqual(@as(usize, 0), shared.bad.load(.monotonic));
        for (&models) |*m| try testing.expect(m.tok.segment_cache.?.bytes() <= budget);
    }
}

fn peekForPage(dir: []const u8) !void {
    var cc = try peekChatConfig(testing.io, std.heap.page_allocator, dir, null);
    defer cc.deinit();
    _ = templateTakesPreserveThinking(&cc);
    _ = templateHasThinkSpan(&cc);
}

/// How a forked child that ran `f(arg)` ended: a crash is the failure the shape cases look for.
fn childEnd(comptime f: anytype, arg: anytype) enum { ok, failed, crashed } {
    const pid = std.c.fork();
    if (pid == 0) {
        _ = std.c.alarm(20);
        std.c._exit(if (f(arg)) |_| 0 else |_| 3);
    }
    var status: c_int = 0;
    _ = std.c.waitpid(pid, &status, 0);
    const s: u32 = @bitCast(status);
    if (s & 0x7f != 0) return .crashed;
    return switch ((s >> 8) & 0xff) {
        0 => .ok,
        3 => .failed,
        else => .crashed,
    };
}

test "peekChatConfig: a pack folder of any shape the settings page looks at is a template or a named error, never a crash" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const F = struct { name: []const u8, body: ?[]const u8 };
    const tc = "tokenizer_config.json";
    const cases = [_][]const F{
        &.{.{ .name = tc, .body = "{\"chat_template\": 5}" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": null}" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": [1, null, {\"template\": 5}, {\"name\": 5, \"template\": \"T\"}]}" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": \"x\", \"add_bos_token\": \"true\"}" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": \"x\", \"chat_template\": \"y\"}" }},
        &.{.{ .name = tc, .body = "" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": \"x" }},
        &.{.{ .name = tc, .body = "{\"chat_template\": \"\xff\xfe\"}" }},
        &.{.{ .name = tc, .body = null }},
        &.{},
        &.{ .{ .name = tc, .body = "{}" }, .{ .name = "chat_template.jinja", .body = null } },
        &.{ .{ .name = tc, .body = "{}" }, .{ .name = "chat_template.jinja", .body = "" } },
        &.{ .{ .name = tc, .body = "{}" }, .{ .name = "chat_template.jinja", .body = "{{ '\xff' }}" } },
        &.{.{ .name = tc, .body = "{\"chat_template\": \"{% include 'chat_template.jinja' %}\"}" }},
    };
    var crashes: usize = 0;
    for (cases, 0..) |files, i| {
        var name_buf: [16]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "{d}", .{i});
        var sub = try tmp.dir.createDirPathOpen(io, name, .{});
        defer sub.close(io);
        for (files) |f| if (f.body) |b| try sub.writeFile(io, .{ .sub_path = f.name, .data = b }) else try sub.createDirPath(io, f.name);
        const dir = try sub.realPathFileAlloc(io, ".", testing.allocator);
        defer testing.allocator.free(dir);
        if (childEnd(peekForPage, @as([]const u8, dir)) == .crashed) crashes += 1;
    }
    try testing.expectEqual(@as(usize, 0), crashes);
}

test "renderChatSpans: a continuation the template does not render verbatim never cuts the prompt inside an earlier turn" {
    const a = testing.allocator;
    // Drops assistant turns, so the partial's text is found only in the user's turn.
    const tpl = "{%- for message in messages -%}{%- if message['role'] != 'assistant' -%}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";
    var cc = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    try cc.compile();
    defer cc.dropCompiled();
    const msgs = [_]Message{
        .{ .role = "user", .content = "Repeat after me: Hello there, friend. Then go on." },
        .{ .role = "assistant", .content = "Hello there" },
    };
    const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, true);
    defer a.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "Then go on.") != null);
    try testing.expect(std.mem.endsWith(u8, out, "<|im_start|>assistantHello there"));
}

test "renderChatSpans: GLM-5.3 continuing a reply that spells a think closer keeps the user's turn whole" {
    const a = testing.allocator;
    var cc = ChatConfig{ .chat_template = @embedFile("fixtures/glm53_chat_template.jinja"), .bos_token = null, .eos_token = "<|endoftext|>", .add_bos_token = false, .allocator = a };
    try cc.compile();
    defer cc.dropCompiled();
    const partial = "The tag is </think>  then the answer";
    const msgs = [_]Message{
        .{ .role = "user", .content = "Finish my sentence: " ++ partial ++ " ... and explain." },
        .{ .role = "assistant", .content = partial },
    };
    const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, true);
    defer a.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "and explain.") != null);
    try testing.expect(std.mem.endsWith(u8, out, partial));
}

test "renderChatSpans: a client's own doubled think closer reaches the prompt as sent, in any turn" {
    const a = testing.allocator;
    const tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
    var cc = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    try cc.compile();
    defer cc.dropCompiled();
    for ([_][]const u8{ "user", "tool", "assistant" }) |role| {
        const msgs = [_]Message{ .{ .role = "user", .content = "q" }, .{ .role = role, .content = "grep says: </think></think> twice" }, .{ .role = "user", .content = "ok" } };
        const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, false);
        defer a.free(out);
        try testing.expect(std.mem.indexOf(u8, out, "grep says: </think></think> twice") != null);
    }
}

test "renderChatSpans: a client's think opener at the end of a render with no generation prompt is not closed for it" {
    const a = testing.allocator;
    const tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{% endfor %}";
    var cc = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    try cc.compile();
    defer cc.dropCompiled();
    const msgs = [_]Message{.{ .role = "user", .content = "the tag is <think>" }};
    const out = try renderChatTemplate(a, &msgs, &cc, null, null, false, null, false);
    defer a.free(out);
    try testing.expectEqualStrings("<|im_start|>user\nthe tag is <think>", out);
}

test "renderChatSpans: a forced tool_choice keeps every byte of the plain render on every served template" {
    const a = testing.allocator;
    const instr = "\nYou MUST call the function \"get_time\". Do not respond with text.";
    const call = [_]ToolCall{.{ .id = "c1", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
    const msgs = [_]Message{
        .{ .role = "user", .content = "what time is it" },
        .{ .role = "assistant", .content = "", .reasoning_content = "I SHOULD CALL THE CLOCK", .tool_calls = &call },
        .{ .role = "tool", .content = "12:00", .tool_call_id = "c1" },
    };
    const served = [_]struct { text: []const u8, eos: ?[]const u8 }{
        .{ .text = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .eos = "<|im_end|>" },
        .{ .text = @embedFile("fixtures/glm53_chat_template.jinja"), .eos = "<|endoftext|>" },
        .{ .text = @embedFile("fixtures/dsv41_chat_template.jinja"), .eos = null },
        .{ .text = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .eos = "<|im_end|>" },
    };
    for (served) |s| for ([_]?bool{ null, true, false }) |preserve| for ([_]bool{ true, false }) |thinking| {
        var cc = ChatConfig{ .chat_template = s.text, .bos_token = null, .eos_token = s.eos, .add_bos_token = false, .allocator = a, .preserve_thinking = preserve };
        try cc.compile();
        defer cc.dropCompiled();
        const plain = try renderChatTemplate(a, &msgs, &cc, forced_choice_tools, null, thinking, null, false);
        defer a.free(plain);
        const forced = try renderChatTemplate(a, &msgs, &cc, forced_choice_tools, instr, thinking, null, false);
        defer a.free(forced);
        try testing.expect(keepsEveryByte(plain, forced));
    };
}

test "keepsEveryByte: the forced render is the plain one with bytes put in at one place" {
    try testing.expect(keepsEveryByte("abc", "abc"));
    try testing.expect(keepsEveryByte("abc", "abXYc"));
    try testing.expect(keepsEveryByte("abc", "abcXY"));
    try testing.expect(keepsEveryByte("abc", "XYabc"));
    try testing.expect(keepsEveryByte("aab", "aaab"));
    try testing.expect(!keepsEveryByte("abc", "ac"));
    try testing.expect(!keepsEveryByte("abc", "aXbYc"));
    try testing.expect(!keepsEveryByte("abc", "abd"));
}

test "renderChatSpans: a tool result's image keeps its placeholder when the template has no tool branch" {
    const a = testing.allocator;
    const tpl =
        \\{%- for m in messages -%}{%- if m.role in ['system','user','assistant'] -%}<|im_start|>{{ m.role }}
        \\{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}{% if p.type == 'image' %}<|image_pad|>{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}<|im_end|>
        \\{% endif -%}{%- endfor -%}
    ;
    var cc = ChatConfig{ .chat_template = tpl, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
    try cc.compile();
    defer cc.dropCompiled();
    const img = [_]ImageData{.{ .pixels = "", .width = 1, .height = 1 }};
    const call = [_]ToolCall{.{ .id = "c1", .name = "shot", .arguments = "{}" }};
    const msgs = [_]Message{
        .{ .role = "user", .content = "take one" },
        .{ .role = "assistant", .content = "", .tool_calls = &call },
        .{ .role = "tool", .content = "hereafter", .tool_call_id = "c1", .images = &img, .media_parts = &.{.{ .at = 4, .kind = .image }} },
        .{ .role = "user", .content = "describe" },
    };
    const out = try renderChatTemplate(a, &msgs, &cc, null, null, true, null, false);
    defer a.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "<tool_response>\nhere<|image_pad|>after\n</tool_response>") != null);
}

test "renderChatSpans: ill-formed UTF-8 in a continuation or the ChatML fallback reaches the prompt mended" {
    const a = testing.allocator;
    const bad = "x\xff\xc0\x80y";
    const call = [_]ToolCall{.{ .id = "c1", .name = "f" ++ bad, .arguments = "{\"a\":\"" ++ bad ++ "\"}" }};
    const msgs = [_]Message{
        .{ .role = "user", .content = "q" ++ bad },
        .{ .role = "assistant", .content = "", .tool_calls = &call },
        .{ .role = "tool", .content = "r" ++ bad, .tool_call_id = "c1" },
        .{ .role = "assistant", .content = "partial " ++ bad },
    };
    const tpl = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    for ([_][]const u8{ tpl, "" }) |text| for ([_]bool{ true, false }) |cont| {
        var cc = ChatConfig{ .chat_template = text, .bos_token = null, .eos_token = "<|im_end|>", .add_bos_token = false, .allocator = a };
        const out = try renderChatTemplate(a, &msgs, &cc, forced_choice_tools, null, true, null, cont);
        defer a.free(out);
        try testing.expect(std.unicode.utf8ValidateSlice(out));
    };
}

test "templateProbeRender: memoized probes answer as fresh renders for every effort spelling, across a restart of the memo" {
    const a = testing.allocator;
    const served = [_]struct { text: []const u8, bos: ?[]const u8, eos: ?[]const u8 }{
        .{ .text = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja"), .bos = null, .eos = "<|im_end|>" },
        .{ .text = @embedFile("fixtures/glm53_chat_template.jinja"), .bos = null, .eos = "<|endoftext|>" },
        .{ .text = @embedFile("fixtures/dsv41_chat_template.jinja"), .bos = null, .eos = null },
        .{ .text = @embedFile("fixtures/mimo_v26_chat_template.jinja"), .bos = "<s>", .eos = "<|im_end|>" },
    };
    var words = std.ArrayList(?[]const u8).empty;
    defer {
        for (words.items) |w| if (w) |x| a.free(x);
        words.deinit(a);
    }
    try words.append(a, null);
    for ([_][]const u8{ "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra", "0", "+5", "101", "-1", "1e2", "62.0", "" }) |w| try words.append(a, try a.dupe(u8, w));
    for (1..101) |n| try words.append(a, try std.fmt.allocPrint(a, "{d}", .{n}));
    var checked: usize = 0;
    for (served) |s| {
        for ([_]?[]const u8{ null, arch_fx.qwen_model_type }) |mt| {
            const compiled = try Compiled.init(a, s.text, true, mt);
            defer compiled.deinit(a);
            var cc = ChatConfig{ .chat_template = s.text, .bos_token = s.bos, .eos_token = s.eos, .add_bos_token = false, .allocator = a, .compiled = compiled };
            for (0..2) |_| for ([_]?bool{ null, true, false }) |preserve| for ([_]bool{ true, false }) |thinking| for (words.items) |effort| {
                cc.preserve_thinking = preserve;
                const settings = try templateSettings(a, &cc, thinking, effort);
                for ([_]?*const Tpl{ &compiled.main, if (compiled.late_system) |*l| l else null }) |maybe| {
                    const tpl = maybe orelse continue;
                    for ([_]Probe{ .tool_content, .late_system, .call, .call_empty, .answer }) |probe| {
                        const fresh = try renderProbeFresh(a, tpl, probe, &settings);
                        defer if (fresh) |f| a.free(f);
                        const got = try templateProbeRender(a, tpl, probe, &settings);
                        defer if (got) |g| a.free(g);
                        try testing.expectEqual(fresh == null, got == null);
                        if (fresh) |f| try testing.expectEqualStrings(f, got.?);
                        checked += 1;
                    }
                }
            };
        }
    }
    try testing.expect(checked > 0);
}
