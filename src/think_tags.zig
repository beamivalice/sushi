const std = @import("std");
const testing = std.testing;
const chat = @import("chat.zig");
const tool_calls_mod = @import("tool_calls.zig");
const BARE_INVOKE = tool_calls_mod.BARE_INVOKE;
const ChatConfig = chat.ChatConfig;
const FUNCTION_CALLS_TAG = tool_calls_mod.FUNCTION_CALLS_TAG;
const Message = chat.Message;
const ToolCall = chat.ToolCall;
const mimoTemplateConfig = chat.mimoTemplateConfig;
const miniCpm5AttrValue = tool_calls_mod.miniCpm5AttrValue;
const renderChatTemplate = chat.renderChatTemplate;

/// The FIRST unclosed think/thought opener: the earliest `<|channel>thought`
/// or `<think>` in `text` with no matching close tag anywhere after it.
/// Gemma 4 opens a NEW thought channel at the very end of a prose turn (its
/// channel-thought tail behavior) — and sometimes SEVERAL in a row, none of
/// them closed (seen live from the 26B GGUF via pi). Everything from the
/// first such opener onward is dangling thought; cutting at the LAST opener
/// instead leaks the earlier raw tags into visible content. A pos-0 opener is
/// reported too: callers (the split arms / stripTrailingThinkOpen) strip
/// their leading CLOSED block first, so an unclosed opener at the start of the
/// remainder is a genuine dangling re-open (`…<channel|>\n<|channel>thought\n`,
/// 2026-06-19 live) — excluding it leaked the bare opener into content.
const TrailingThinkOpen = struct { pos: usize, after: usize };
fn lastUnclosedThinkOpen(text: []const u8) ?TrailingThinkOpen {
    var from: usize = 0;
    while (nextThinkOpen(text, from)) |o| {
        const close = if (o.is_think_style)
            indexOfThinkCloseTag(text, o.after)
        else if (std.mem.indexOfPos(u8, text, o.after, "<channel|>")) |p|
            TagAt{ .pos = p, .len = "<channel|>".len }
        else
            null;
        if (close) |c| {
            // This block IS closed — keep scanning past its close.
            from = c.pos + c.len;
            continue;
        }
        return .{ .pos = o.pos, .after = o.after };
    }
    return null;
}

// ── Think tag matching ──

pub const TagAt = struct { pos: usize, len: usize };

/// Length of the complete tag `base>` at the START of `text` (base "<think" matches "<think>").
pub fn tagLenAt(text: []const u8, comptime base: []const u8) ?usize {
    if (text.len <= base.len or !std.mem.startsWith(u8, text, base) or text[base.len] != '>') return null;
    return base.len + 1;
}

/// True when `text` could still GROW into `full` with more streamed bytes
/// (strict prefix: "<th", "<think").
fn isPartialTag(text: []const u8, comptime full: []const u8) bool {
    return text.len > 0 and text.len < full.len and std.mem.startsWith(u8, full, text);
}

pub fn thinkOpenTagLenAt(text: []const u8) ?usize {
    return tagLenAt(text, "<think");
}

pub fn thinkCloseTagLenAt(text: []const u8) ?usize {
    return tagLenAt(text, "</think");
}

fn indexOfThinkOpenTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "<think")) |p| {
        if (thinkOpenTagLenAt(text[p..])) |l| {
            if (openCodeSpanStart(text[0..p]) == null) return .{ .pos = p, .len = l };
        }
        i = p + "<think".len;
    }
    return null;
}

pub fn indexOfThinkCloseTag(text: []const u8, from: usize) ?TagAt {
    var i = from;
    while (std.mem.indexOfPos(u8, text, i, "</think")) |p| {
        if (thinkCloseTagLenAt(text[p..])) |l| {
            if (!thinkCloseIsToolCallPayload(text, p) and openCodeSpanStart(text[0..p]) == null) return .{ .pos = p, .len = l };
        }
        i = p + "</think".len;
    }
    return null;
}

/// Start of the inline code span still open at the end of `text`, or null. A model writing about its own
/// format quotes the tag in backticks (`` `<think>plan</think>` ``): inside a span a close tag is text,
/// not the end of the thought. Judged from the bytes before the tag alone, so a stream's verdict on a
/// tag never changes as more arrive. A span ends with its line; three backticks are a fence, not a span.
fn openCodeSpanStart(text: []const u8) ?usize {
    var i = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |nl| nl + 1 else 0;
    var open: ?usize = null;
    while (i < text.len) {
        if (text[i] != '`') {
            i += 1;
            continue;
        }
        var end = i;
        while (end < text.len and text[end] == '`') end += 1;
        if (end - i < 3) open = if (open == null) i else null;
        i = end;
    }
    return open;
}

/// How much of `text` a stream may ship so the rest still carries every open code span's backtick: a
/// trailing run of backticks (its length is not yet known) and an open span wait.
fn spanSettledLen(text: []const u8) usize {
    var n = text.len;
    while (n > 0 and text[n - 1] == '`') n -= 1;
    return openCodeSpanStart(text[0..n]) orelse n;
}

/// Whether a `</think>` at `pos` is an ARGUMENT VALUE rather than a block close.
///
/// The think split runs before tool parse, and this is the shared close-tag
/// scan every surface uses — so a model writing about its own prompt format
/// (`<arg_value>…closes a thought with </think>…</arg_value>`) had the split cut
/// straight through the call, losing it entirely and leaking the fragments.
/// Agent traffic writes files about prompts.
///
/// Payload ⟺ the nearest tool opener before `pos` is still OPEN there (no
/// `</tool_call` between it and `pos`) AND the block does close afterwards.
/// Both halves are load-bearing:
///   • without the first, a call the model emitted and CLOSED inside its
///     thought would make the real `</think>` after it look like payload;
///   • without the second, an unclosed opener inside a thought (the leaked-
///     markup case, which `trimLeakedToolMarkup` handles downstream) would
///     swallow the answer that follows the close.
fn thinkCloseIsToolCallPayload(text: []const u8, pos: usize) bool {
    // `streamThinkGate2` calls the close scan once per TOKEN on a growing
    // buffer, so the common case — text with no tool markup at all — must cost
    // one substring scan, not a walk per opener spelling. Both spellings share
    // this needle.
    if (std.mem.indexOf(u8, text[0..pos], "tool_call") == null) return false;
    // Nearest `<tool_call`-family opener before `pos`. Only this family is
    // considered: it is the one whose bodies carry free-form argument text.
    var open: ?usize = null;
    for ([_][]const u8{ "<tool_call", "<|tool_call" }) |m| {
        var i: usize = 0;
        while (std.mem.indexOfPos(u8, text, i, m)) |p| {
            if (p >= pos) break;
            if (open == null or p > open.?) open = p;
            i = p + 1;
        }
    }
    const o = open orelse return false;
    var from = o;
    while (std.mem.indexOfPos(u8, text, from, "</tool_call")) |c| {
        if (c >= pos) return true;
        // A close spelled inside the argument is not the call's own end.
        if (valueStillOpen(text, o, c, "<arg_value", "</arg_value>") or
            valueStillOpen(text, o, c, "<parameter", "</parameter>"))
        {
            from = c + "</tool_call".len;
            continue;
        }
        return false; // that call already closed — a real block close
    }
    return false; // never closes — treat as leaked markup, not payload
}

fn valueStillOpen(text: []const u8, from: usize, at: usize, open_tag: []const u8, close_tag: []const u8) bool {
    const rel = std.mem.lastIndexOf(u8, text[from..at], open_tag) orelse return false;
    const open_at = from + rel;
    if (std.mem.indexOfPos(u8, text, open_at + open_tag.len, close_tag)) |end| return end > at;
    return true;
}

/// Length of the trailing bytes of `buf` that could still GROW into a think
/// close tag with more streamed bytes — either family: `</think>` or
/// Gemma's `<channel|>`. Streaming
/// reasoning flushes must hold this suffix back: flushing half a close tag
/// splits it across the flush boundary, so the retained remainder never
/// matches and the stream stays "inside thinking" forever (tag bytes leak
/// into reasoning_content). 0 = safe to flush everything.
pub fn partialThinkCloseSuffixLen(buf: []const u8) usize {
    // Any real close tag is well under 32 bytes; a longer '<'-run is prose.
    const window_start = if (buf.len > 32) buf.len - 32 else 0;
    var i = buf.len;
    while (i > window_start) {
        i -= 1;
        if (buf[i] != '<') continue;
        const tail = buf[i..];
        if (isPartialTag(tail, "</think>")) return tail.len;
        if (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail)) return tail.len;
    }
    return 0;
}

test "thinkOpenerPossible: a thought no opener can start is delivered without waiting for more bytes" {
    try std.testing.expect(!thinkOpenerPossible("The"));
    try std.testing.expect(!thinkOpenerPossible("Okay"));
    try std.testing.expect(!thinkOpenerPossible("\nWe"));
    try std.testing.expect(thinkOpenerPossible(""));
    try std.testing.expect(thinkOpenerPossible(" \n"));
    try std.testing.expect(thinkOpenerPossible("<"));
    try std.testing.expect(thinkOpenerPossible("<thi"));
    try std.testing.expect(thinkOpenerPossible("\n<|chan"));
    try std.testing.expect(!thinkOpenerPossible("too"));
}

test "partialThinkCloseSuffixLen holds back growing close tags, ignores prose" {
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("plain reasoning text"));
    try testing.expectEqual(@as(usize, 7), partialThinkCloseSuffixLen("thinking…</think"));
    try testing.expectEqual(@as(usize, 5), partialThinkCloseSuffixLen("thinking…<chan"));
    // A COMPLETE tag is not a partial — the scan-side indexOf owns it.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("done</think>"));
    // A '<' too far back is prose, not a growing tag.
    try testing.expectEqual(@as(usize, 0), partialThinkCloseSuffixLen("a < b and then lots of ordinary words follow here"));
}

/// Tag length when `text` ENDS with a complete think-close tag.
fn endsWithThinkCloseTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkCloseTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tag length when `text` ENDS with a complete think-open tag.
fn endsWithThinkOpenTag(text: []const u8) ?usize {
    if (text.len == 0 or text[text.len - 1] != '>') return null;
    const lt = std.mem.lastIndexOfScalar(u8, text, '<') orelse return null;
    const l = thinkOpenTagLenAt(text[lt..]) orelse return null;
    return if (lt + l == text.len) l else null;
}

/// Tool-call WRAPPER openers that must never survive into text a client
/// renders. Every entry is the opening marker of a call block, so whatever
/// follows it is call payload, not prose — parseToolCalls has already run by
/// the time these matter, so a marker still present means the block did not
/// parse (a mangled opener, a bare marker with no call behind it, or a call
/// the model emitted inside its think block, which the parser deliberately
/// skips). The whole tail is cut rather than excised: a wrapper we could not
/// parse has no reliable end, and shipping half of it is the same leak.
pub const tool_markup_openers = [_][]const u8{
    "<｜DSML｜", // DeepSeek-V4 native (covers tool_calls/invoke/parameter)
    "</｜DSML｜", // lone closers the model emits after prose with no opener
    "<|tool_call", // Gemma 4
    "<tool_call", // Hermes XML + the plural variants
};

/// Earliest index of Claude-style `<function_calls>` / `<invoke …>` markup, or null. Like MiniCPM5's,
/// the opener alone is prose ("wrap it in <invoke name=…>"): a `<parameter ` or `</invoke>` must be there too.
fn bareInvokeMarkupCut(text: []const u8) ?usize {
    if (std.mem.indexOf(u8, text, "<parameter ") == null and
        std.mem.indexOf(u8, text, BARE_INVOKE.invoke_close) == null) return null;
    var cut = std.mem.indexOf(u8, text, FUNCTION_CALLS_TAG);
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= text.len or std.ascii.isWhitespace(text[after])) {
            if (cut == null or p < cut.?) cut = p;
            break;
        }
        scan = after;
    }
    return cut;
}

/// True while `buf` holds, or ends in a growing prefix of, a `<function_calls>` / `<invoke …` opener.
fn invokeOpenerHoldsForTools(buf: []const u8) bool {
    if (std.mem.indexOf(u8, buf, FUNCTION_CALLS_TAG) != null) return true;
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, buf, scan, BARE_INVOKE.invoke)) |p| {
        const after = p + BARE_INVOKE.invoke.len;
        if (after >= buf.len or std.ascii.isWhitespace(buf[after])) return true;
        scan = after;
    }
    inline for (.{ FUNCTION_CALLS_TAG, BARE_INVOKE.invoke }) |marker| {
        var k: usize = 2;
        while (k < marker.len) : (k += 1) if (std.mem.endsWith(u8, buf, marker[0..k])) return true;
    }
    return false;
}

/// Earliest index of MiniCPM5 attribute-form tool markup, or null.
/// Discriminated exactly as the streaming gate does it — `<function` followed
/// by WHITESPACE, with a `name=` inside the opener region — so `<functional`,
/// `<function-like>` and `<function>` prose are never cut. A malformed opener
/// (dropped quote) still matches, which is the point: `parseToolCalls` has
/// already run by the time this matters, so markup still present is markup that
/// did NOT parse, and shipping it renders raw XML to the user.
fn miniCpm5MarkupCut(text: []const u8) ?usize {
    // The opener ALONE is not evidence of leaked markup. Prose names functions
    // ("Document the syntax as <function name=\"shell\"> before …") and cutting
    // there deletes the rest of the answer — the exact contradiction that let a
    // green parser test and a green scrubber test disagree about one string.
    // Require the same structural evidence the parser requires: a `<param`
    // element or a `</function>` close somewhere in the text.
    if (std.mem.indexOf(u8, text, "<param") == null and
        std.mem.indexOf(u8, text, "</function>") == null) return null;
    var scan: usize = 0;
    while (std.mem.indexOfPos(u8, text, scan, "<function")) |p| {
        const after = p + "<function".len;
        if (after >= text.len) return p; // truncated mid-opener
        if (!std.ascii.isWhitespace(text[after])) {
            scan = after;
            continue;
        }
        const gt_rel = std.mem.indexOfScalar(u8, text[after..], '>');
        const region_end = if (gt_rel) |g| after + g else text.len;
        if (std.mem.indexOf(u8, text[after..region_end], "name=") != null) return p;
        scan = region_end;
    }
    return null;
}

/// Cut visible text at the first unparsed tool-call marker. Returns a prefix
/// slice (no alloc); text with no marker is returned untouched.
pub fn trimLeakedToolMarkup(text: []const u8) []const u8 {
    var cut: usize = text.len;
    for (tool_markup_openers) |m| {
        if (std.mem.indexOf(u8, text, m)) |p| {
            if (p < cut) cut = p;
        }
    }
    // MiniCPM5 attribute form. Same reasoning, different discriminator: the
    // dialect is `<function` + whitespace + a `name=` attribute, so prose words
    // beginning `<function…` are left alone.
    if (miniCpm5MarkupCut(text)) |p| {
        if (p < cut) cut = p;
    }
    if (bareInvokeMarkupCut(text)) |p| {
        if (p < cut) cut = p;
    }
    if (cut == text.len) return text;
    return std.mem.trimEnd(u8, text[0..cut], "\n\r\t ");
}

// NOTE: there is deliberately NO strip-think helper anymore. Reasoning the
// model generated is ALWAYS delivered (reasoning_content / thinking blocks) —
// thinking-off is enforced on the PROMPT side (`noThinkTailSuffix`), so every
// delivery site takes `splitThinkBlock(text, true, …)` and ships both halves.

/// Trim trailing think/channel/tool CLOSE markers (and surrounding whitespace)
/// from visible content. A close marker is never valid at the tail of content —
/// but a degenerate model can spam them (live: a Gemma variant emitted 16 bare
/// `<channel|>` after its answer; the leading strip cuts the FIRST close, the
/// trailing-OPEN strip ignores closes, so they leaked). The Gemma tool CLOSE
/// `<tool_call|>` is here too: a degenerate 1-token bare close with no
/// `<|tool_call>` opener leaked as the entire content (live 2026-07-16 soak,
/// gemma-4-26B). parseToolCalls runs BEFORE this strip and extracts any real
/// call, so any residual `<tool_call|>` reaching here is orphan by construction.
/// Loops so a run of closers is fully removed. Returns a prefix slice (no alloc).
pub fn trimTrailingThinkClosers(content: []const u8) []const u8 {
    var s = content;
    while (true) {
        const t = std.mem.trimEnd(u8, s, "\n \t\r");
        if (std.mem.endsWith(u8, t, "<channel|>")) {
            s = t[0 .. t.len - "<channel|>".len];
        } else if (std.mem.endsWith(u8, t, "<tool_call|>")) {
            s = t[0 .. t.len - "<tool_call|>".len];
        } else if (endsWithThinkCloseTag(t)) |l| {
            s = t[0 .. t.len - l];
        } else {
            return t;
        }
    }
}

/// Truncate a trailing unclosed thought opener (and its dangling thought)
/// out of visible content. Used by the split arms after the leading block has
/// been handled.
fn stripTrailingThinkOpen(content: []const u8) []const u8 {
    if (lastUnclosedThinkOpen(content)) |o| {
        return std.mem.trimEnd(u8, content[0..o.pos], "\n ");
    }
    return content;
}

pub const ThinkSplit = struct {
    reasoning_content: ?[]const u8,
    content: []const u8,
};

/// True when a rendered generation prompt ends inside a think block the
/// A think block the MODEL opens when the template did not: a bare `<think>` token, or
/// Gemma 4's TWO tokens, the `<|channel>` marker (dropped from visible deltas) then the
/// word `thought`. `channel_armed` = the previous delta was that marker and nothing visible
/// has been emitted. Returns the opener to seed `think_buf` with.
pub fn modelThinkOpener(channel_armed: bool, token_text: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, std.mem.trimEnd(u8, token_text, "\n"), "<think>")) return "<think>";
    if (channel_armed and std.mem.eql(u8, token_text, "thought")) return "<|channel>thought";
    return null;
}

/// Control-channel marker TOKENS that must never appear in a visible stream
/// delta. Every entry is a single special token in its family's vocab, so
/// exact-match filtering at the delta level is complete — a marker can't be
/// split across deltas. Shared by every streaming surface's flush path.
pub fn isChannelMarkerToken(tt: []const u8) bool {
    return std.mem.eql(u8, tt, "<|channel>") or std.mem.eql(u8, tt, "<channel|>") or
        std.mem.eql(u8, tt, "<think>") or std.mem.eql(u8, tt, "</think>");
}

pub const BARE_THINK_OPENER = "<think>";
pub const BARE_THINK_CLOSER = "</think>";

/// template opened (Qwen 3.5/3.6 style: `…assistant\n<think>\n`). Callers
/// decode the last few prompt tokens and pass the tail here. A thinking-off
/// render ends with a CLOSED `</think>` block and must not match.
pub fn promptTailOpensThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return endsWithThinkOpenTag(trimmed) != null;
}

/// Narrow form used by generation-side phase switches: armed against the bare
/// `</think>` boundary only.
pub fn promptTailOpensBareThink(tail: []const u8) bool {
    const trimmed = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return std.mem.endsWith(u8, trimmed, "<think>");
}

/// Which reasoning opener a rendered generation prompt ENDS with (trailing
/// whitespace ignored): a bare `<think>`, or nothing. Generation-side protocol
/// resolution consumes this.
pub const ThinkTailClass = enum { bare, none };

pub fn promptThinkTailClass(tail: []const u8) ThinkTailClass {
    const trimmed_tail = std.mem.trimEnd(u8, tail, "\n\r\t ");
    return if (std.mem.endsWith(u8, trimmed_tail, BARE_THINK_OPENER)) .bare else .none;
}

/// Suffix that COMMITS the no-think channel in the rendered prompt, or null.
/// Thinking-off on an always-thinking template otherwise means generating the
/// whole reasoning pass and discarding it — the prompt-side commit is what
/// makes thinking-off a real skip. A think block the template opened unconditionally is closed;
/// thinking-off Qwen-style renders end CLOSED and never match.
pub fn noThinkTailSuffix(rendered: []const u8, enable_thinking: bool) ?[]const u8 {
    if (enable_thinking) return null;
    const tail = contentChannelTail(rendered);
    return if (tail.len > 0) tail else null;
}

/// What must be appended to a rendered prompt so generation resumes in the
/// assistant's VISIBLE CONTENT channel rather than inside a reasoning block.
///
/// Two callers, one rule, deliberately: thinking-off enforces the content
/// channel so no reasoning pass runs, and a CONTINUATION needs the same thing
/// for a different reason — the partial text is prose the user has already
/// read, so a prompt that lands inside a think block would return the rest of
/// their answer as reasoning. Written twice, the thinking-off and continuation
/// cases would drift apart.
pub fn contentChannelTail(rendered: []const u8) []const u8 {
    const tail = std.mem.trimEnd(u8, rendered[rendered.len -| 64 ..], "\n\r\t ");
    if (promptTailOpensThink(tail)) return "</think>";
    return "";
}

/// Whether this conversation SHAPE is a continuation: it ends with an
/// assistant message carrying prose. The one predicate every surface asks, so
/// "what counts as continuable" cannot differ between the endpoint that
/// implies it and the endpoint that takes a flag — and so the answer matches
/// what `continuationPrefill` will actually do with it.
pub fn continuationRequested(messages: []const Message) bool {
    return continuationPrefill(messages, true) != null;
}

/// The partial assistant reply a continuation request is asking to extend, or
/// null when this is an ordinary turn.
///
/// The flag describes the SHAPE of the request, so a client that sets it on a
/// conversation not ending in an assistant message gets an ordinary turn —
/// never the user's own words committed into the assistant's mouth.
///
/// The prefill is trimmed of trailing whitespace: a tokenizer merges a trailing
/// space into the word that follows, so a prompt ending in one tokenizes
/// differently than the same text does mid-reply. (Anthropic's API rejects such
/// a prefill outright; trimming keeps the request serviceable, and only the
/// PROMPT is trimmed — the stored reply keeps its whitespace.)
pub fn continuationPrefill(messages: []const Message, continue_final: bool) ?[]const u8 {
    if (!continue_final or messages.len == 0) return null;
    const last = messages[messages.len - 1];
    if (!std.mem.eql(u8, last.role, "assistant")) return null;
    // A reply that ended in a tool call has no prose to resume — splicing a
    // continuation onto it would re-run the parse chain over half a call.
    if (last.tool_calls != null) return null;
    const trimmed = std.mem.trimEnd(u8, last.content, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}

/// Split model output into reasoning_content and content.
/// Handles both `<think>...</think>` and Gemma 4's `<|channel>thought\n...<channel|>`.
/// `opened_by_template`: the generation prompt ended with a template-injected
/// think opener (see `promptTailOpensThink`), so output with no literal opener
/// and no close tag is still reasoning — generation started inside the block.
pub fn splitThinkBlock(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    const split = splitThinkBlockKeepingMarkup(text, thinking, opened_by_template);
    const reasoning: ?[]const u8 = if (split.reasoning_content) |r| blk: {
        const t = trimLeakedToolMarkup(r);
        break :blk if (t.len > 0) t else null;
    } else null;
    return .{ .reasoning_content = reasoning, .content = trimLeakedToolMarkup(split.content) };
}

/// The split WITHOUT the leaked-markup cut — for the one caller that feeds the
/// content back to `parseToolCalls` (the /v1/messages non-streaming path).
/// Every arm below returns raw slices; the cut is applied ONCE in the wrapper
/// above so a new arm cannot forget it.
pub fn splitThinkBlockKeepingMarkup(text: []const u8, thinking: bool, opened_by_template: bool) ThinkSplit {
    // Gemma 4 style: <|channel>thought\n...<channel|>\n<|channel>\ncontent
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        const think_tag = "<|channel>thought\n";
        const reasoning_start: usize = if (std.mem.startsWith(u8, text, think_tag)) think_tag.len else if (std.mem.startsWith(u8, text, "<|channel>thought")) "<|channel>thought".len else 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..end], "\n ");
        var content = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
        // Strip the content channel tag: <|channel>\n or <|channel>. A
        // re-opened THOUGHT channel (`<|channel>thought…`) is NOT a content
        // opener — stripping its `<|channel>` prefix here left a bare "thought"
        // in visible content (2026-06-19 live); leave it for stripTrailingThinkOpen.
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>") and !std.mem.startsWith(u8, content, "<|channel>thought")) {
            content = content[10..];
        }
        content = std.mem.trimStart(u8, content, "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    // Standard style: <think>...</think>
    if (indexOfThinkCloseTag(text, 0)) |close| {
        const reasoning_start: usize = thinkOpenTagLenAt(text) orelse 0;
        const reasoning = std.mem.trim(u8, text[reasoning_start..close.pos], "\n ");
        const content = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
        return .{
            .reasoning_content = if (reasoning.len > 0) reasoning else null,
            .content = trimTrailingThinkClosers(stripTrailingThinkOpen(content)),
        };
    }
    if (thinking) {
        // Unclosed think block: split policy depends on whether the model's
        // output begins with a literal opener.
        //   • Literal opener present → model definitely entered thinking but
        //     ran out of tokens / didn't close. Treat as reasoning.
        //   • No literal opener + template-injected opener → generation began
        //     INSIDE the block (Qwen 3.5/3.6 render `…assistant\n<think>\n`);
        //     an unclosed tail is truncated reasoning, never content.
        //   • No literal opener + no template opener → the model answered
        //     directly (Gemma style); keep the answer visible as content.
        if (thinkOpenTagLenAt(text) != null or std.mem.startsWith(u8, text, "<|channel>thought")) {
            const start: usize = if (thinkOpenTagLenAt(text)) |l| l else if (std.mem.startsWith(u8, text, "<|channel>thought\n")) "<|channel>thought\n".len else "<|channel>thought".len;
            const reasoning = std.mem.trimStart(u8, text[start..], "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        if (opened_by_template) {
            const reasoning = std.mem.trimStart(u8, text, "\n ");
            return .{ .reasoning_content = if (reasoning.len > 0) reasoning else null, .content = "" };
        }
        // Prose answer that ENDS by opening a new, unclosed thought block
        // (Gemma 12B tail behavior): the text before the opener is the
        // answer; the dangling thought is reasoning. The raw opener tag must
        // never leak into visible content.
        if (lastUnclosedThinkOpen(text)) |o| {
            const content = std.mem.trimEnd(u8, std.mem.trimStart(u8, text[0..o.pos], "\n "), "\n ");
            const reasoning = std.mem.trim(u8, text[o.after..], "\n ");
            return .{
                .reasoning_content = if (reasoning.len > 0) reasoning else null,
                .content = content,
            };
        }
        // No thought block, but the model may have emitted (or been truncated
        // right after) a dangling Gemma 4 *content* channel opener `<|channel>`
        // / `<|channel>\n`. Strip it so a cut-off reply never leaks the raw
        // control tag into visible content; whatever follows is the answer.
        var content = std.mem.trimStart(u8, text, "\n ");
        if (std.mem.startsWith(u8, content, "<|channel>\n")) {
            content = content[11..];
        } else if (std.mem.startsWith(u8, content, "<|channel>")) {
            content = content[10..];
        }
        return .{ .reasoning_content = null, .content = std.mem.trimStart(u8, content, "\n ") };
    }
    return .{ .reasoning_content = null, .content = text };
}

const ThinkOpen = struct { pos: usize, after: usize, is_think_style: bool };

/// Earliest think/thought opener at or after `from` (either tag family).
fn nextThinkOpen(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<|channel>thought");
    const think = indexOfThinkOpenTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<|channel>thought".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Earliest close tag at or after `from` (either tag family).
fn nextThinkClose(text: []const u8, from: usize) ?ThinkOpen {
    const chan = std.mem.indexOfPos(u8, text, from, "<channel|>");
    const think = indexOfThinkCloseTag(text, from);
    if (chan == null and think == null) return null;
    if (think == null or (chan != null and chan.? < think.?.pos)) {
        return .{ .pos = chan.?, .after = chan.? + "<channel|>".len, .is_think_style = false };
    }
    return .{ .pos = think.?.pos, .after = think.?.pos + think.?.len, .is_think_style = true };
}

/// Skip an optional Gemma 4 CONTENT channel opener (`<|channel>` not followed
/// by `thought`) plus surrounding newlines, right after a thought close.
fn skipContentChannelTag(text: []const u8, start: usize) usize {
    var pos = start;
    while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
    const tag = "<|channel>";
    if (pos + tag.len <= text.len and std.mem.eql(u8, text[pos .. pos + tag.len], tag)) {
        const rest = text[pos + tag.len ..];
        if (!std.mem.startsWith(u8, rest, "thought")) {
            pos += tag.len;
            while (pos < text.len and (text[pos] == '\n' or text[pos] == ' ')) pos += 1;
        }
    }
    return pos;
}

/// Merge ALL closed think/thought blocks in `text` into one leading block.
///
/// The split/strip layer understands exactly one leading block plus an
/// optional UNCLOSED trailing opener. Gemma 4 12B, however, re-opens a thought
/// channel mid-turn and closes it again (`…content<|channel>thought\n…
/// <channel|>more content`) — observed live via Claude Code on /v1/messages,
/// where the raw pair leaked verbatim into the visible text block. This pass
/// rewrites such output to `<|channel>thought\n{all thought text}<channel|>
/// {all content}` so every downstream consumer (splitThinkBlock,
/// parseToolCalls) handles it unchanged.
///
/// Returns null when no rewrite is needed (zero or one LEADING closed block —
/// the overwhelmingly common case, zero-cost). An unclosed trailing opener is
/// left in place for the existing trailing-strip logic.
pub fn normalizeEmbeddedThinkBlocks(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    var reasoning_parts = std.ArrayList([]const u8).empty;
    defer reasoning_parts.deinit(allocator);
    var content_parts = std.ArrayList([]const u8).empty;
    defer content_parts.deinit(allocator);

    var style_think = false;
    var style_set = false;
    var closed_blocks: usize = 0;
    var first_block_leading = false;
    var pos: usize = 0;

    // Template-opened leading block: a close tag BEFORE any opener (Qwen
    // renders `…assistant\n<think>\n` into the prompt, so output starts
    // mid-thought and contains only the close).
    if (nextThinkClose(text, 0)) |lc| {
        const open_pos = if (nextThinkOpen(text, 0)) |o| o.pos else text.len;
        if (lc.pos < open_pos) {
            try reasoning_parts.append(allocator, std.mem.trim(u8, text[0..lc.pos], "\n "));
            style_think = lc.is_think_style;
            style_set = true;
            closed_blocks += 1;
            first_block_leading = true;
            pos = skipContentChannelTag(text, lc.after);
        }
    }

    while (true) {
        const o = nextThinkOpen(text, pos) orelse {
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        const found: ?TagAt = if (o.is_think_style)
            indexOfThinkCloseTag(text, o.after)
        else if (std.mem.indexOfPos(u8, text, o.after, "<channel|>")) |cp|
            TagAt{ .pos = cp, .len = "<channel|>".len }
        else
            null;
        const close = found orelse {
            // Unclosed trailing opener — leave verbatim for the trailing-strip
            // logic in splitThinkBlock.
            try content_parts.append(allocator, text[pos..]);
            break;
        };
        try content_parts.append(allocator, text[pos..o.pos]);
        try reasoning_parts.append(allocator, std.mem.trim(u8, text[o.after..close.pos], "\n "));
        if (!style_set) {
            style_think = o.is_think_style;
            style_set = true;
        }
        closed_blocks += 1;
        if (o.pos == 0) first_block_leading = true;
        pos = skipContentChannelTag(text, close.pos + close.len);
    }

    // Rewrite only when a closed block exists beyond the single leading one.
    if (closed_blocks == 0) return null;
    if (closed_blocks == 1 and first_block_leading) return null;

    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, if (style_think) "<think>\n" else "<|channel>thought\n");
    var first = true;
    for (reasoning_parts.items) |r| {
        if (r.len == 0) continue;
        if (!first) try out.appendSlice(allocator, "\n\n");
        try out.appendSlice(allocator, r);
        first = false;
    }
    try out.appendSlice(allocator, if (style_think) "</think>\n" else "<channel|>\n");
    first = true;
    for (content_parts.items) |c| {
        const trimmed = std.mem.trim(u8, c, "\n ");
        if (trimmed.len == 0) continue;
        if (!first) try out.append(allocator, '\n');
        try out.appendSlice(allocator, trimmed);
        first = false;
    }
    return try out.toOwnedSlice(allocator);
}

/// Streaming-only: true when the buffer TAIL is a partial prefix of a think
/// opener (`<think>` / `<|channel>thought`). The buffered-stream flush must
/// hold these bytes back until the tag completes or diverges — flushing them
/// leaks tag fragments as visible content (a pi session showed prose ending
/// in a glued "thought" because `<|channel>` flushed before "thought"
/// arrived and completed the opener).
/// Can a thought that begins with `buf` still grow into an opener the streaming gates strip?
/// Every opener starts with `<`, so any other first byte is decided at once instead of waiting
/// for the gates' 7 bytes.
pub fn thinkOpenerPossible(buf: []const u8) bool {
    const rest = std.mem.trimStart(u8, buf, " \n");
    return rest.len == 0 or rest[0] == '<';
}

pub fn endsWithPartialThinkOpen(buf: []const u8) bool {
    const tags = [_][]const u8{ "<|channel>thought", "<think>" };
    for (tags) |tag| {
        // Strictly-partial prefixes only — a COMPLETE opener in the buffer is
        // the caller's contains-check's job; ours is the growing tail.
        var l = @min(buf.len, tag.len - 1);
        while (l > 0) : (l -= 1) {
            if (std.mem.endsWith(u8, buf, tag[0..l])) return true;
        }
    }
    return false;
}

/// Streaming-only: should the chat-completion SSE path defer flushing
/// because `buf` is on track to become a tool call?
///
/// True when:
///   * `buf` contains `<tool` followed by a valid tag terminator
///     (`>`, ` `, `\t`, `\n`, `_`, `|`) somewhere — that's any of the
///     accepted families: `<tool>`, `<tool …>`, `<tool_call…>`, `<tool_calls…>`,
///     `<tool_request…>`, `<tool_requests…>` (mirrors `parseToolCalls`).
///   * `buf` contains the Gemma 4 `<|tool_call` substring.
///   * `functionOpenerHoldsForTools(buf)` — a MiniCPM5 `<function name="…">`
///     opener still unresolved, or resolved into a real call. `<functional`,
///     `<function>` and `<function foo>` are not this dialect and keep flowing.
///     NOTE the bare equals-sign Hermes `<function=NAME>` form is deliberately
///     NOT held here: `parseToolCalls` accepts it wrapper-less, so the gate is
///     narrower than the parser for THAT dialect. Pre-existing on main, tracked
///     separately — two attempts at closing it inside this change were both
///     wrong, and real captures show non-adjacent `<parameter=` openers.
///   * `buf[0] == '{'` and `buf` contains `"name"` (raw JSON tool-call shape).
///   * `buf` ends with a partial prefix that could grow into one of the above
///     in the next token (`<`, `<t`, `<to`, `<too`, `<tool`, or any prefix
///     of `<|tool_call`).
///
/// Conservatively false on `<toolkit>`, `<toolbar>`, etc. — anything where
/// the char after `<tool` isn't a valid tag terminator. That keeps prose
/// that happens to mention HTML-ish tags flowing through normally.
pub fn streamShouldBufferForTools(buf: []const u8) bool {
    if (buf.len == 0) return false;

    // Raw JSON tool-call shape: starts with `{` and has `"name"` somewhere.
    if (buf[0] == '{' and std.mem.indexOf(u8, buf, "\"name\"") != null) return true;

    // Gemma 4 fully-formed canonical open.
    if (std.mem.indexOf(u8, buf, "<|tool_call") != null) return true;

    // DeepSeek-V4 DSML: everything from `<｜DSML｜` on is a tool-call block.
    if (std.mem.indexOf(u8, buf, "<｜DSML｜") != null) return true;
    if (std.mem.indexOf(u8, buf, "</｜DSML｜") != null) return true;

    // `<tool…` family: walk every `<tool` occurrence, accept the first one
    // whose terminator is valid (mirrors parseToolCalls' acceptance rule).
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<tool")) |rel| {
        const after = scan + rel + "<tool".len;
        if (after >= buf.len) return true; // truncated mid-prefix → in progress
        const c = buf[after];
        if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '_' or c == '|') return true;
        scan = after;
    }

    // `<function…` family: MiniCPM5's `<function name="…">`.
    if (functionOpenerHoldsForTools(buf)) return true;
    // Claude-style `<function_calls>` / `<invoke name="…">`.
    if (invokeOpenerHoldsForTools(buf)) return true;

    // Trailing partial prefixes — the next streamed token could complete any
    // of these into a real tool open. Order doesn't matter; first endsWith
    // hit wins. Listed shortest-first for legibility.
    const tail_prefixes = [_][]const u8{
        "<",     "<t",     "<to",     "<too",     "<tool",
        "<|",    "<|t",    "<|to",    "<|too",    "<|tool",
        "<|tool_", "<|tool_c", "<|tool_ca", "<|tool_cal",
        // MiniCPM5 V3 `<function name="…">` — COMPLETE ladder. `<funct` is a
        // real decomposition in this vocabulary (`<f` id 54303 + `unct` id
        // 14185 decode to exactly `<funct`); omitting that one rung flushed the
        // fragment and leaked the rest of the tag. The rungs are DERIVED in the
        // test, so a future gap fails there instead of shipping.
        "<f", "<fu", "<fun", "<func", "<funct", "<functi", "<functio", "<function",
        // DSML fullwidth-bar prefixes (`｜` = 3 bytes; cover mid-codepoint
        // splits too in case the tokenizer spells the marker in pieces)
        "<\xef",  "<\xef\xbd", "<｜",  "<｜D", "<｜DS", "<｜DSM",
        "<｜DSML", "<｜DSML\xef", "<｜DSML\xef\xbd",
        "</", "</\xef", "</\xef\xbd", "</｜", "</｜D", "</｜DS", "</｜DSM",
        "</｜DSML", "</｜DSML\xef", "</｜DSML\xef\xbd",
    };
    for (tail_prefixes) |p| {
        if (std.mem.endsWith(u8, buf, p)) return true;
    }
    return false;
}

/// Streaming-only: does a `<function …` opener anywhere in `buf` still hold?
/// Three arrival states:
///
///   unresolved (no `>` yet)        → HOLD; the next token may complete a call
///   resolved WITH a quoted name=   → HOLD; it is a MiniCPM5 V3 call
///   resolved WITHOUT one          → RELEASE and keep scanning; the parser
///                                    (parseMiniCpm5ToolCalls) will decline it
///
/// The `name=` test delegates to `miniCpm5AttrValue`, so the gate's acceptance
/// is DERIVED from the parser's instead of hand-mirrored. That direction
/// matters: the gate must stay a strict SUPERSET of the parser. A stricter gate
/// (e.g. one demanding `name=` immediately after the whitespace) would flush a
/// real call's opener as content — `<function foo name="x">` does parse — and
/// the end-of-stream parse would then emit the call anyway: leak AND duplicate.
///
/// Releasing the resolved-non-call case is what keeps prose usable. The hold is
/// monotonic over a buffer that never shrinks, so treating any `<function foo>`
/// as a call silences every remaining token of a tool-enabled turn.
fn functionOpenerHoldsForTools(buf: []const u8) bool {
    var scan: usize = 0;
    while (std.mem.indexOf(u8, buf[scan..], "<function")) |rel| {
        const after = scan + rel + "<function".len;
        // Truncated mid-marker — the next token decides what this is.
        if (after >= buf.len) return true;
        // `<functional`, `<function=NAME>` (the equals-sign Hermes form) and
        // `<function>` are not this dialect; skip and keep scanning so ordinary
        // prose keeps flowing. Gating the bare `<function=` dialect is a
        // SEPARATE problem with its own evidence — see the follow-up issue.
        if (!std.ascii.isWhitespace(buf[after])) {
            scan = after;
            continue;
        }
        const tag_rel = std.mem.indexOfScalar(u8, buf[after..], '>') orelse return true;
        const tag_end = after + tag_rel;
        // Empty `name=""` is rejected by the parser too, so it releases here.
        if (miniCpm5AttrValue(buf[after..tag_end], "name")) |v| {
            if (v.len > 0) return true;
        }
        scan = tag_end + 1;
    }
    return false;
}

/// Streaming-only: what should a tools-enabled SSE path do with the buffered
/// text so far, with respect to thinking? Shared by the chat-completions and
/// Anthropic /v1/messages stream handlers — the two paths drifted apart once
/// already: /v1/messages only recognized think OPENERS present in the output,
/// so Qwen-family template-opened thinking (opener injected into the PROMPT)
/// streamed as visible text and a raw `</think>` leaked into Claude Code
/// transcripts (2026-06-10 live).
///
///   .hold_thinking — inside an unclosed think block; buffer, emit nothing
///   .split_think   — close tag arrived; splitThinkBlock once, emit
///                    reasoning + visible remainder, clear the buffer, and
///                    set think_closed for the rest of the turn
///   .flush_text    — plain visible prose; stream it
///
/// `think_closed` releases the enable_thinking hold after the one split —
/// without it the visible answer sits in the buffer until end-of-stream and
/// gets misfiled as reasoning (the pi hidden-answer bug).
pub const StreamThinkGate = enum { flush_text, hold_thinking, split_think };

/// The gate WITHOUT a prompt-opened think block — every family whose template
/// gates its `<think>` on `enable_thinking` renders the closed `<think></think>`
/// signature (or no tag at all) when thinking is off, so the flag is false for
/// them and this stays their exact behavior.
pub fn streamThinkGate(buf: []const u8, enable_thinking: bool, think_closed: bool) StreamThinkGate {
    return streamThinkGate2(buf, enable_thinking, think_closed, false);
}

/// `prompt_opened_think`: the rendered generation prompt ended with an OPEN
/// think tag (`server.promptOpensThink`). That is a fact about the prompt
/// BYTES, not about our request flag — LFM2.5's template pre-opens `<think>`
/// unconditionally, so with thinking off the model's reasoning arrives as
/// tag-free prose and the 3-arg gate flushed the whole chain-of-thought as the
/// visible answer (live 2026-08-04). It can only ever ADD a hold: when
/// thinking is on `enable_thinking` already held, and once `think_closed` the
/// term is dropped so the answer after the block still flushes.
/// Memoized marker scan for the streaming think gate, owned BESIDE the stream
/// buffer and reset with it.
///
/// Without it `streamThinkGate2` re-scans the whole accumulated buffer on every
/// token for as long as thinking markup is present and no close has arrived —
/// O(buffer) per token, O(n²) over a reasoning block. Measured on a hermetic
/// 4000-token stream growing to 113 KB: 47.99 µs/token with no close in the
/// buffer (every scan runs to the end), 16.98 µs/token with a close near the
/// front (the close scan stops early). ~0.3% of a 15 ms/token model, ~1% at
/// 5 ms/token, and it GROWS with the thought.
///
/// Exactness rests on one property of the marker grammar: every recognized
/// marker contains exactly ONE `<`, at its start. A marker straddling the scanned/unscanned boundary must therefore
/// begin at the LAST `<` in the buffer — and if that `<` cannot still grow into
/// a marker, nothing straddles the boundary at all. So the resume point is
/// exact and bounded, with no overlap constant to get wrong.
pub const ThinkScan = struct {
    /// Bytes already examined. Never trusted when the buffer is SHORTER than
    /// this (the loop reset the buffer without resetting the scan).
    checked_upto: usize = 0,
    /// Index of the last `<` seen, or null. The only position a marker can
    /// straddle the boundary from.
    last_lt: ?usize = null,
    saw_channel_thought: bool = false,
    saw_open: bool = false,
    saw_channel_close: bool = false,
    /// Tool markup anywhere in the buffer. This is what makes a found close
    /// UNSTABLE and therefore un-latchable: `thinkCloseIsToolCallPayload`
    /// reclassifies a close as an argument value once a `</tool_call` arrives
    /// AFTER it, so a close that was real at token N is payload at token N+k.
    /// With no tool markup the close is stable and latching is exact.
    saw_tool_marker: bool = false,
    /// First real close tag, once found and stable.
    close: ?TagAt = null,
    /// Bytes examined by the last call, INCLUDING the exact full re-scan the
    /// tool-markup case falls back to. Test-visible: the invariant is that this
    /// stays FLAT as the buffer grows for traffic with no tool markup — which
    /// is the arm the 47.99 µs/token bench measured.
    last_scan_span: usize = 0,

    /// Longest plain substring needle minus one — the overlap a windowed
    /// substring search needs so a needle split across two arrivals is caught.
    /// `<|channel>thought` is the longest at 17.
    const substr_overlap: usize = 16;

    pub fn reset(self: *ThinkScan) void {
        self.* = .{};
    }

    /// Where the next scan must start so no marker spanning the boundary is
    /// missed: the last `<` if it can still grow into a tag (see the type's
    /// doc comment for why no earlier `<` can), and far enough back for the
    /// plain substring needles either way.
    fn resumeFrom(self: *const ThinkScan, buf: []const u8) usize {
        const upto = @min(self.checked_upto, buf.len);
        const substr_from = upto -| substr_overlap;
        const lt = self.last_lt orelse return substr_from;
        if (lt >= upto) return substr_from;
        const tail = buf[lt..upto];
        const growing = isPartialTag(tail, "<think>") or
            isPartialTag(tail, "</think>") or
            (tail.len < "<|channel>thought".len and std.mem.startsWith(u8, "<|channel>thought", tail)) or
            (tail.len < "<channel|>".len and std.mem.startsWith(u8, "<channel|>", tail));
        return if (growing) @min(lt, substr_from) else substr_from;
    }

    fn advance(self: *ThinkScan, buf: []const u8) void {
        if (buf.len < self.checked_upto) self.reset(); // buffer was cleared under us
        if (buf.len == self.checked_upto and self.checked_upto != 0) {
            self.last_scan_span = 0;
        } else {
            const from = self.resumeFrom(buf);
            self.last_scan_span = buf.len - from;
            const win = buf[from..];
            if (std.mem.lastIndexOfScalar(u8, win, '<')) |rel| self.last_lt = from + rel;
            if (!self.saw_channel_thought and std.mem.indexOf(u8, win, "<|channel>thought") != null)
                self.saw_channel_thought = true;
            if (!self.saw_channel_close and std.mem.indexOf(u8, win, "<channel|>") != null)
                self.saw_channel_close = true;
            if (!self.saw_tool_marker and std.mem.indexOf(u8, win, "tool_call") != null)
                self.saw_tool_marker = true;
            // The open scan takes the FULL buffer with a start offset: it
            // consults nothing before the match, but a windowed slice would
            // shift the reported position. Monotone — a complete opener stays
            // one — so latching is exact.
            if (!self.saw_open and indexOfThinkOpenTag(buf, from) != null) self.saw_open = true;
            if (self.close == null and !self.saw_tool_marker)
                self.close = indexOfThinkCloseTag(buf, from);
            self.checked_upto = buf.len;
        }
        if (self.saw_tool_marker) {
            // Exact, and only here: the payload classification looks FORWARD
            // past the close for a `</tool_call`, so it must be recomputed on
            // the whole buffer. The caller latches `think_closed` right after a
            // split, so this runs for a handful of tokens, not a whole block.
            self.close = indexOfThinkCloseTag(buf, 0);
            self.last_scan_span = buf.len;
        }
    }
};

/// `streamThinkGate2` with the per-token rescan memoized. Byte-identical
/// verdicts — pinned prefix-by-prefix against the fresh gate over a corpus that
/// includes every marker family, a close buried in a tool argument, and a
/// suffix longer than the partial-scan's tail window.
pub fn streamThinkGateScan(
    buf: []const u8,
    enable_thinking: bool,
    think_closed: bool,
    prompt_opened_think: bool,
    scan: *ThinkScan,
) StreamThinkGate {
    scan.advance(buf);
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        scan.saw_channel_thought or
        scan.saw_open or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    return if (scan.saw_channel_close or scan.close != null) .split_think else .hold_thinking;
}

pub fn streamThinkGate2(buf: []const u8, enable_thinking: bool, think_closed: bool, prompt_opened_think: bool) StreamThinkGate {
    const has_thinking = ((enable_thinking or prompt_opened_think) and !think_closed) or
        std.mem.indexOf(u8, buf, "<|channel>thought") != null or
        indexOfThinkOpenTag(buf, 0) != null or
        (std.mem.startsWith(u8, buf, "<|channel>") and buf.len < 18) or
        (std.mem.startsWith(u8, buf, "<think") and buf.len < 7) or
        // A partial opener at the buffer TAIL (mid-text re-opened channel
        // arriving token by token) must hold the flush — flushing leaks tag
        // fragments like a glued "thought".
        endsWithPartialThinkOpen(buf);
    if (!has_thinking) return .flush_text;
    const has_close = std.mem.indexOf(u8, buf, "<channel|>") != null or
        indexOfThinkCloseTag(buf, 0) != null;
    return if (has_close) .split_think else .hold_thinking;
}

/// End-of-stream classification for a think buffer that never saw a close tag:
/// is it reasoning, or was it content all along?
///
/// The streaming loop starts `in_think_block` from `enable_thinking`, i.e. it
/// ASSUMES the model begins inside a think block whenever thinking was
/// requested. That holds only for templates that pre-inject the opener. Gemma 4
/// renders a bare `<|turn>model\n` and lets the MODEL decide, so a turn it
/// answers directly ("391") carries no think markup at all — and the whole
/// answer was flushed as `reasoning_content` with EMPTY content, while the
/// non-streaming path (`splitThinkBlock`, no opener + no close + not
/// template-opened ⇒ content) returned it correctly (live 2026-08-04).
///
/// So require POSITIVE evidence a block was open: the PROMPT opened one, or the
/// model emitted a literal opener. This is what makes the two paths agree.
pub fn streamTailIsReasoning(in_think_block: bool, prompt_opened_think: bool, saw_think_open: bool) bool {
    return in_think_block and (prompt_opened_think or saw_think_open);
}

/// The reasoning bytes not yet streamed, or null when there is nothing new.
///
/// The tools streaming path emits reasoning INCREMENTALLY (see the
/// `.hold_thinking` arm), so every later site that emits reasoning for the same
/// Streaming counterpart to the leading `trimStart(content, "\n ")` every
/// non-streaming split already applies.
///
/// The two surfaces disagreed on the BYTES of the same answer: LFM2.5's template
/// opens `<think>` unconditionally, so its first generated token is the newline
/// pair that follows the closer, and `splitThinkBlock` drops it while a stream
/// flushing tokens verbatim shipped it — `'One, two…'` non-streaming,
/// `'\n\nOne, two…'` streaming (live 2026-08-13). Any model whose first visible
/// token is whitespace is in this class, not just LFM2.5.
///
/// Leading whitespace is the ONE thing a stream can still withhold: nothing
/// visible has been sent yet, so this is a suppression, never a retraction.
///
/// It suppresses a chunk that is ENTIRELY whitespace, never a prefix of one. A
/// partial trim would ship a logprobs entry whose `token` no longer appears in
/// `content` — the collector describes whole tokens and cannot split one — so
/// the two contracts are kept consistent by never cutting inside a token. A
/// single token mixing leading whitespace with visible text is the residual
/// case, and it keeps its whitespace on both surfaces' terms: the stream ships
/// it, and the caller flips `content_started` so nothing later is touched.
///
/// `content_started` must flip on the first NON-EMPTY emission, so a suppressed
/// whitespace token leaves the next one still eligible.
pub fn streamContentLead(chunk: []const u8, content_started: bool) []const u8 {
    if (content_started) return chunk;
    if (std.mem.trimStart(u8, chunk, "\n ").len == 0) return "";
    return chunk;
}

/// turn must send only the remainder or the client sees the thought twice.
///
/// `already >= reasoning.len` returns null rather than underflowing: the split
/// can SHRINK when a tool marker appears mid-thought and `trimLeakedToolMarkup`
/// cuts the tail. An SSE delta cannot be retracted, so the honest behavior is to
/// send nothing further — never to resend from the top.
/// The part of a still-open thought's split a stream may ship. A closed thought
/// is delivered as `trim(thought, "\n ")`, so a trailing run of those bytes, and
/// a close tag still arriving, wait for what follows.
pub fn settledReasoning(open: []const u8) []const u8 {
    return std.mem.trimEnd(u8, open[0 .. open.len - partialThinkCloseSuffixLen(open)], "\n ");
}

pub const OpenThoughtFlush = struct { skip: usize, ship: usize };

/// How a stream flushes an open thought's raw bytes (everything since its last
/// delta) so the deltas add up to the closed thought's `trim(thought, "\n ")`:
/// `skip` leading bytes drop before the first delta, `ship` bytes go out, and the
/// rest (a trailing "\n " run, a partial close tag, a split code point) waits.
pub fn openThoughtFlush(buf: []const u8, shipped: bool) OpenThoughtFlush {
    const skip = if (shipped) 0 else buf.len - std.mem.trimStart(u8, buf, "\n ").len;
    const whole = @max(buf.len - partialThinkCloseSuffixLen(buf), skip);
    var end = skip + std.mem.trimEnd(u8, buf[skip..whole], "\n ").len;
    // An open code span stays in the buffer: `indexOfThinkCloseTag` reads its backtick to know a tag inside is quoted.
    end = @max(@min(end, spanSettledLen(buf[0..whole])), skip);
    while (end > skip and end < buf.len and (buf[end] & 0xC0) == 0x80) end -= 1;
    return .{ .skip = skip, .ship = end - skip };
}

/// The last delta of a thought whose remaining raw bytes are `body`.
pub fn closedThoughtDelta(body: []const u8, shipped: bool) []const u8 {
    return if (shipped) std.mem.trimEnd(u8, body, "\n ") else std.mem.trim(u8, body, "\n ");
}

/// The last delta of a thought the generation cut while still open, from the
/// bytes the stream still holds: an opener not yet consumed is structure, and a
/// thought that shipped nothing drops its leading "\n " as the split does.
pub fn cutThoughtDelta(held: []const u8, opener_consumed: bool, shipped: bool) []const u8 {
    var body = held;
    if (!opener_consumed) {
        if (thinkOpenTagLenAt(body)) |l| {
            body = body[l..];
        } else if (std.mem.startsWith(u8, body, "<|channel>thought")) {
            body = body["<|channel>thought".len..];
        }
    }
    return if (shipped) body else std.mem.trimStart(u8, body, "\n ");
}

pub fn unstreamedReasoning(reasoning: []const u8, already: usize) ?[]const u8 {
    if (already >= reasoning.len) return null;
    return reasoning[already..];
}

test "mimo_v2 template preserves reasoning and XML tool history" {
    const a = testing.allocator;
    var config = mimoTemplateConfig(a);
    const calls = [_]ToolCall{.{ .id = "call_1", .name = "sum", .arguments = "{\"x\":2}" }};
    const messages = [_]Message{
        .{ .role = "user", .content = "Compute." },
        .{ .role = "assistant", .content = "", .reasoning_content = "Use sum.", .tool_calls = &calls },
        .{ .role = "tool", .content = "2" },
        .{ .role = "user", .content = "Continue." },
    };
    const expected = "<|im_start|>user\nCompute.<|im_end|>" ++
        "<|im_start|>assistant\n<think>Use sum.</think>" ++
        "<tool_call><function=sum><parameter=x>2</parameter></function></tool_call><|im_end|>" ++
        "<|im_start|>tool\n2<|im_end|>" ++
        "<|im_start|>user\nContinue.<|im_end|><|im_start|>assistant\n";
    const thinking = try renderChatTemplate(a, &messages, &config, null, null, true, null, false);
    defer a.free(thinking);
    try testing.expectEqualStrings(expected, thinking);
    try testing.expect(!promptTailOpensThink(thinking));
    const plain = try renderChatTemplate(a, &messages, &config, null, null, false, null, false);
    defer a.free(plain);
    try testing.expectEqualStrings(expected ++ "<think></think>", plain);
}

test "split content: think tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\nHello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<think>reasoning</think>\n\nHello", true, false).content);
}

test "split content: empty for open think tag (truncated thought is never content)" {
    try testing.expectEqualStrings("", splitThinkBlock("<think>still thinking...", true, false).content);
}

test "indexOfThinkCloseTag: a literal tool close inside the argument does not end the call" {
    const literal_close = "<tool_call>write<arg_value>see </tool_call> and </think> here</arg_value></tool_call>";
    try testing.expect(indexOfThinkCloseTag(literal_close, 0) == null);
    const hermes = "<tool_call><function=write><parameter=content>see </tool_call> and </think> here</parameter></function></tool_call>";
    try testing.expect(indexOfThinkCloseTag(hermes, 0) == null);
}

test "indexOfThinkCloseTag: a close inside an OPEN tool call is argument payload, not a block close" {
    // Agent traffic writes files about prompts, and the think split runs
    // BEFORE the tool parse — so this cut used to destroy the whole call.
    const payload = "<tool_call>write\n<arg_key>content</arg_key>\n" ++
        "<arg_value>a thought ends with </think> here</arg_value>\n</tool_call>";
    try testing.expect(indexOfThinkCloseTag(payload, 0) == null);

    // A call the model emitted and CLOSED inside its thought: the `</think>`
    // after it is a REAL close and the answer stays content.
    const in_thought = "<think>let me try <tool_call>read</tool_call> hmm</think>The answer is 4.";
    const c = indexOfThinkCloseTag(in_thought, 0) orelse return error.MissingClose;
    try testing.expectEqualStrings("The answer is 4.", in_thought[c.pos + c.len ..]);
    try testing.expectEqualStrings("The answer is 4.", splitThinkBlock(in_thought, true, false).content);

    // An opener that NEVER closes is leaked markup, not a payload container:
    // the close still ends the block, and trimLeakedToolMarkup takes the tail.
    const unclosed = "<think>starting <tool_call>partial</think>Answer.";
    try testing.expect(indexOfThinkCloseTag(unclosed, 0) != null);

    // Ordinary shapes are untouched.
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a", 0).?.pos == 8);
    try testing.expect(indexOfThinkCloseTag("<think>r</think>a<tool_call>x</tool_call>", 0).?.pos == 8);
}

test "indexOfThinkCloseTag: a close quoted in a code span does not end the thought" {
    // The model's reasoning quoting the raw string it was asked about.
    const quoted = "The user wants `<think>plan A</think>Done` parsed.\n</think>\n\nOPEN plan A CLOSE";
    const split = splitThinkBlock(quoted, true, true);
    try testing.expectEqualStrings("The user wants `<think>plan A</think>Done` parsed.", split.reasoning_content.?);
    try testing.expectEqualStrings("OPEN plan A CLOSE", split.content);
    // Double backticks are a span too; a span ends with its line.
    try testing.expect(indexOfThinkCloseTag("see ``</think>`` here", 0) == null);
    try testing.expect(indexOfThinkCloseTag("a stray ` tick\n</think>x", 0) != null);
    // A fence is not a span, so the close after it is the real one.
    try testing.expect(indexOfThinkCloseTag("```\ncode\n```</think>x", 0) != null);
    // A closed span is behind the tag.
    try testing.expect(indexOfThinkCloseTag("the `tag` is done</think>x", 0) != null);
    // Nothing after the span changes the verdict on a tag inside it.
    try testing.expect(indexOfThinkCloseTag("a `</think>", 0) == null);
    try testing.expect(indexOfThinkCloseTag("a `</think>`", 0) == null);
}

test "a visible answer quoting a think block in a code span stays one answer" {
    // The answer after a closed thought writes the tags it was asked not to; they are text, not a second thought.
    const answer = "Reading `<think>plan A</think>Done` gives plan A and Done.";
    try testing.expect(indexOfThinkOpenTag(answer, 0) == null);
    try testing.expect(try normalizeEmbeddedThinkBlocks(testing.allocator, answer) == null);
    // The same tags outside a span still open and close a thought.
    try testing.expect(indexOfThinkOpenTag("Answer.\n<think>plan A</think>Done", 0) != null);
    const merged = (try normalizeEmbeddedThinkBlocks(testing.allocator, "Answer.\n<think>plan A</think>Done")).?;
    defer testing.allocator.free(merged);
    try testing.expectEqualStrings("<think>\nplan A</think>\nAnswer.\nDone", merged);
}

test "openThoughtFlush: an open code span waits so the tag inside it stays quoted" {
    // `` ` `` + `<think>x` arrived: the span's backtick must still be in the buffer when `</think>` lands.
    const open = openThoughtFlush("the string `<think>plan", false);
    try testing.expectEqual(@as(usize, 0), open.skip);
    try testing.expectEqual(@as(usize, "the string ".len), open.ship);
    // A run of backticks may still grow; a closed span ships.
    try testing.expectEqual(@as(usize, "word ".len), openThoughtFlush("word `", false).ship);
    try testing.expectEqual(@as(usize, "word `x`y".len), openThoughtFlush("word `x`y", false).ship);
    // A line break ends the span.
    try testing.expectEqual(@as(usize, "a ` b\nc".len), openThoughtFlush("a ` b\nc", false).ship);
}

test "split content: text without think tags passes through" {
    try testing.expectEqualStrings("Hello world", splitThinkBlock("Hello world", true, false).content);
}

test "splitThinkBlock with complete think block" {
    const result = splitThinkBlock("<think>reasoning here</think>answer here", false, false);
    try testing.expectEqualStrings("reasoning here", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock with empty reasoning" {
    const result = splitThinkBlock("<think>\n\n</think>\n\nactual content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("actual content", result.content);
}

test "splitThinkBlock thinking=true no close tag, literal opener present" {
    // Model entered thinking but ran out of tokens before closing.
    const result = splitThinkBlock("<think>partial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, template-opened block" {
    // Qwen 3.6 truncated-thinking leak (regression): the chat template injects
    // `<think>\n` into the generation prompt, so the model's output starts
    // INSIDE the think block with no literal opener. If generation stops
    // (length) before `</think>`, every token so far is reasoning — it must
    // land in reasoning_content, never in content. Matches the streaming path.
    const result = splitThinkBlock("The user wants 17*23. Let me compute", true, true);
    try testing.expectEqualStrings("The user wants 17*23. Let me compute", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock thinking=true no close tag, no opener, template did NOT open" {
    // Gemma-style direct answer: thinking enabled but the template injects no
    // opener and the model answered without a thought channel. The answer must
    // stay visible as content.
    const result = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("It is currently 8:15 AM PDT.", result.content);
}

test "splitThinkBlock template-opened block with close tag still splits" {
    // Normal Qwen 3.6 round: no literal opener (template-injected), close tag
    // present — reasoning before it, content after.
    const result = splitThinkBlock("compute 340+51=391</think>\n\n391.", true, true);
    try testing.expectEqualStrings("compute 340+51=391", result.reasoning_content.?);
    try testing.expectEqualStrings("391.", result.content);
}

test "splitThinkBlock thinking=false no tags" {
    const result = splitThinkBlock("just content", false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("just content", result.content);
}

test "splitThinkBlock strips think prefix in thinking mode" {
    const result = splitThinkBlock("<think>my reasoning", true, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "promptTailOpensThink detects template-injected opener" {
    // Qwen 3.6 generation prompt tail with thinking on
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>\n"));
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
    // Thinking off renders a CLOSED empty block — must not match
    try testing.expect(!promptTailOpensThink("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // Gemma 4 prompt tail (no injected opener)
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    try testing.expect(!promptTailOpensThink(""));
}

test "promptThinkTailClass resolves bare and plain tails" {
    try testing.expectEqual(ThinkTailClass.bare, promptThinkTailClass("<|im_start|>assistant\n<think>\n"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<think:opensource>"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|im_start|>assistant\n"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<|start|>assistant"));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass(""));
    try testing.expectEqual(ThinkTailClass.none, promptThinkTailClass("<think"));
}

test "endsWithPartialThinkOpen holds back partial opener tails (pi GGUF leak repro)" {
    // The exact failure shape: `<|channel>` flushed as content before
    // "thought" arrived; the buffer tail was a partial opener at every step.
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|chan"));
    try testing.expect(endsWithPartialThinkOpen("prose ends here.<|channel>thoug"));
    try testing.expect(endsWithPartialThinkOpen("prose <"));
    try testing.expect(endsWithPartialThinkOpen("prose <think"));
    try testing.expect(endsWithPartialThinkOpen("<th"));
    // Prose and HTML-ish tags keep flowing.
    try testing.expect(!endsWithPartialThinkOpen("just prose."));
    try testing.expect(!endsWithPartialThinkOpen("prose <table"));
    try testing.expect(!endsWithPartialThinkOpen("a > b"));
    try testing.expect(!endsWithPartialThinkOpen(""));
}

test "splitThinkBlock with think block before tool call" {
    const text = "<think>I need to call the calculator</think>\n<tool_call>\n{\"name\": \"calc\", \"arguments\": {\"a\": 5}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("I need to call the calculator", result.reasoning_content.?);
    // `content` is VISIBLE text, so the call block is cut: parseToolCalls runs
    // on the raw output, and whatever it leaves behind is markup the client
    // must never render.
    try testing.expectEqualStrings("", result.content);
    // The keeping-markup variant is what the one parse-from-content caller
    // (/v1/messages non-streaming) uses, and it still hands over the block.
    const raw = splitThinkBlockKeepingMarkup(text, false, false);
    try testing.expect(std.mem.startsWith(u8, raw.content, "<tool_call>"));
}

test "splitThinkBlock with think block before regular content" {
    const text = "<think>Let me think about this</think>\n\nThe answer is 42.";
    const result = splitThinkBlock(text, false, false);
    try testing.expectEqualStrings("Let me think about this", result.reasoning_content.?);
    try testing.expectEqualStrings("The answer is 42.", result.content);
}

test "splitThinkBlock with empty think block" {
    const text = "<think>\n\n</think>\n\nJust content here.";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("Just content here.", result.content);
}

test "splitThinkBlock no think tags with tool call" {
    const text = "<tool_call>\n{\"name\": \"search\", \"arguments\": {}}\n</tool_call>";
    const result = splitThinkBlock(text, false, false);
    try testing.expect(result.reasoning_content == null);
    try testing.expectEqualStrings("", result.content);
    try testing.expectEqualStrings(text, splitThinkBlockKeepingMarkup(text, false, false).content);
}

test "split content: Gemma 4 channel tags never reach content" {
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning here<channel|>Hello", true, false).content);
    try testing.expectEqualStrings("Hello", splitThinkBlock("<|channel>thought\nreasoning<channel|>\nHello", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nstill thinking...", true, false).content);
}

test "splitThinkBlock Gemma 4 channel tags" {
    const result = splitThinkBlock("<|channel>thought\nmy reasoning<channel|>answer here", false, false);
    try testing.expectEqualStrings("my reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("answer here", result.content);
}

test "splitThinkBlock Gemma 4 thinking in progress" {
    const result = splitThinkBlock("<|channel>thought\npartial reasoning", true, false);
    try testing.expectEqualStrings("partial reasoning", result.reasoning_content.?);
    try testing.expectEqualStrings("", result.content);
}

test "splitThinkBlock truncated mid-thinking does not leak channel tag" {
    // Truncation regression: the model emitted the Gemma 4 *content* channel
    // opener (`<|channel>\n…`) directly — no thought block and no `<channel|>`
    // close — then hit the output cap. The raw `<|channel>` control tag must
    // never reach visible content (it used to leak straight through).
    {
        const r = splitThinkBlock("<|channel>\nThe answer is 42.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("The answer is 42.", r.content);
    }
    // Bare dangling opener (cut off right after the tag) → nothing visible.
    {
        const r = splitThinkBlock("<|channel>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        const r = splitThinkBlock("<|channel>\n", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    // The template-injected-but-no-tags case must still pass through untouched.
    {
        const r = splitThinkBlock("It is currently 8:15 AM PDT.", true, false);
        try testing.expect(r.reasoning_content == null);
        try testing.expectEqualStrings("It is currently 8:15 AM PDT.", r.content);
    }
}

test "splitThinkBlock trailing unclosed thought opener does not leak (Gemma 12B pi regression)" {
    // Gemma 4 12B answers in prose, then opens a NEW thought channel right
    // before the turn ends (its known channel-thought tail behavior). The raw
    // `<|channel>thought` opener — and any unclosed thought text after it —
    // must never reach visible content; pi rendered the literal tag to users.
    {
        // Bare trailing opener, nothing after.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expect(r.reasoning_content == null);
    }
    {
        // Trailing opener with unclosed thought text → thought is reasoning.
        const r = splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write the file", true, false);
        try testing.expectEqualStrings("Here is the design.", r.content);
        try testing.expectEqualStrings("I should now write the file", r.reasoning_content.?);
    }
    {
        // Same shape for the <think> family.
        const r = splitThinkBlock("Done.\n<think>wait, maybe I", true, false);
        try testing.expectEqualStrings("Done.", r.content);
        try testing.expectEqualStrings("wait, maybe I", r.reasoning_content.?);
    }
}

test "split content: trailing unclosed thought opener does not leak" {
    // The visible text must be truncated at a trailing unclosed opener (the
    // tag and dangling thought are reasoning, never content).
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought", true, false).content);
    try testing.expectEqualStrings("Here is the design.", splitThinkBlock("Here is the design.\n<|channel>thought\nI should now write", true, false).content);
    try testing.expectEqualStrings("Done.", splitThinkBlock("Done.\n<think>hmm", true, false).content);
}

test "trimLeakedToolMarkup cuts at the first unparsed tool-call opener" {
    // Untouched when there is no marker (the overwhelmingly common case —
    // the returned slice must be the input itself, not a trimmed copy).
    try testing.expectEqualStrings("plain prose", trimLeakedToolMarkup("plain prose"));
    try testing.expectEqualStrings("ends in a tag <b>", trimLeakedToolMarkup("ends in a tag <b>"));

    // Every wrapper family cuts, and the trailing whitespace before the
    // marker goes with it.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"x\">"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<|tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_call>{}"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<tool_calls>"));

    // A marker with nothing before it leaves nothing behind — a bare marker
    // is not an answer (live: DSV4 answered a question and then emitted a
    // lone `<｜DSML｜` with no call behind it).
    try testing.expectEqualStrings("", trimLeakedToolMarkup("\n\n<｜DSML｜\n"));

    // Earliest marker wins when several families appear.
    try testing.expectEqualStrings("a", trimLeakedToolMarkup("a<tool_call>b<｜DSML｜c"));
}

test "in-thought tool markup never rides out as reasoning" {
    // Live 2026-08-01 (DeepSeek-V4-Flash-0731 agent session): the model
    // emitted a complete DSML call INSIDE its think block, closed the block,
    // then issued a different call in content. parseToolCalls works on the
    // post-think text, so the in-thought call is deliberately skipped — but
    // it rode out verbatim in reasoning_content, and the client round-tripped
    // it into every later prompt.
    const raw = "Let me check the directory.<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"listFiles\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">quake</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls></think>I'll build it.";
    const split = splitThinkBlock(raw, true, true);
    try testing.expectEqualStrings("Let me check the directory.", split.reasoning_content.?);
    try testing.expectEqualStrings("I'll build it.", split.content);

    // Reasoning that is NOTHING but markup reports no reasoning at all.
    const only_markup = splitThinkBlock("<｜DSML｜tool_calls>\n</｜DSML｜tool_calls></think>answer", true, true);
    try testing.expectEqual(@as(?[]const u8, null), only_markup.reasoning_content);
    try testing.expectEqualStrings("answer", only_markup.content);
}

test "a tool marker split across tokens is only visible once concatenated" {
    // The streaming tools flush holds a tail that looked like a tool call and
    // ships it when nothing parses. DSV4 spells the DSML marker as `<` then
    // `｜DSML｜` (two tokens), so a per-token filter sees neither piece as a
    // marker and both went out as visible content (live 2026-08-01). The
    // flush concatenates first for exactly this reason.
    const pieces = [_][]const u8{ "My name is", "<", "｜DSML｜", "\n" };
    var joined: [64]u8 = undefined;
    var n: usize = 0;
    for (pieces) |p| {
        @memcpy(joined[n .. n + p.len], p);
        n += p.len;
    }
    for (pieces) |p| try testing.expectEqualStrings(p, trimLeakedToolMarkup(p));
    try testing.expectEqualStrings("My name is", trimLeakedToolMarkup(joined[0..n]));
}

test "splitThinkBlock re-opened thought opener right after close does not leak" {
    // 2026-06-19 live regression (gemma-4): the model closed its thought
    // channel and IMMEDIATELY re-opened a fresh one with nothing between, then
    // the turn ended. The leading-strip consumed the first closed block,
    // leaving the bare re-opened opener at pos 0 of the remainder — it must
    // still vanish, and `<|channel>thought` must never be mis-stripped as a
    // CONTENT channel (`<|channel>`), which leaked a glued "thought".
    {
        const r = splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false);
        try testing.expectEqualStrings("", r.content);
        try testing.expectEqualStrings("Let me plan.", r.reasoning_content.?);
    }
    {
        // <think> family equivalent.
        const r = splitThinkBlock("<think>plan</think>\n<think>", true, false);
        try testing.expectEqualStrings("", r.content);
    }
    {
        // Content BETWEEN the close and the re-open must survive the cut.
        const r = splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false);
        try testing.expectEqualStrings("Ready.", r.content);
    }
}

test "split content: re-opened thought opener right after close does not leak" {
    // Same regression, content side — this is the exact form that reached
    // chat-history.json (`<|channel>thought\n` as the whole content).
    try testing.expectEqualStrings("", splitThinkBlock("<|channel>thought\nLet me plan.<channel|>\n<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("Ready.", splitThinkBlock("<|channel>thought\nPlan.<channel|>\nReady.<|channel>thought\n", true, false).content);
    try testing.expectEqualStrings("", splitThinkBlock("<think>plan</think>\n<think>", true, false).content);
}

test "renderChatTemplate: the Flash-Next template renders its own efforts and thinking-off (hermetic)" {
    // The template answers thinking-off with a CLOSED `<think>\n\n</think>\n\n`
    // and no reasoning-instructions preamble (its effort validation lives INSIDE
    // the thinking-on branch), so thinking-off must not be forced onto the
    // thinking-on arm. The effort mapping is keyed on `'xhigh'`: the template
    // raises on OpenAI's "high".
    const allocator = testing.allocator;
    const tpl = @embedFile("fixtures/qwen38_flash_next_chat_template.jinja");

    var config = ChatConfig{
        .chat_template = tpl,
        .bos_token = null,
        .eos_token = "<|im_end|>",
        .add_bos_token = false,
        .allocator = allocator,
    };

    const tools_json =
        \\[{"type":"function","function":{"name":"get_time","description":"Get time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}]
    ;
    const messages = [_]Message{
        .{ .role = "user", .content = "What time is it?" },
    };

    // Thinking ON is unchanged: OpenAI's "high" still has to become this
    // family's "xhigh" or the render raises and silently falls back.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "high", false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to xhigh.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n"));
        try testing.expect(promptTailOpensThink(rendered));
    }
    // Absent client effort takes OUR default (low), not the template's xhigh.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to low.") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "set to xhigh") == null);
    }
    // "medium" injects no preamble.
    {
        const med = try renderChatTemplate(allocator, &messages, &config, tools_json, null, true, "medium", false);
        defer allocator.free(med);
        try testing.expect(std.mem.indexOf(u8, med, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, med, "You have access to the following functions:") != null);
    }
    // A tool round renders through the template's own call and response markup.
    {
        const tc = [_]ToolCall{.{ .id = "tc_0", .name = "get_time", .arguments = "{\"timezone\": \"UTC\"}" }};
        const round = [_]Message{
            .{ .role = "user", .content = "What time is it?" },
            .{ .role = "assistant", .content = "", .tool_calls = &tc },
            .{ .role = "tool", .content = "12:34 UTC", .tool_call_id = "tc_0" },
        };
        const rendered = try renderChatTemplate(allocator, &round, &config, tools_json, null, true, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "<function=get_time>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<parameter=timezone>\nUTC\n</parameter>") != null);
        try testing.expect(std.mem.indexOf(u8, rendered, "<tool_response>\n12:34 UTC\n</tool_response>") != null);
    }
    // Thinking OFF renders the template's OWN no-think prompt: closed block,
    // no reasoning preamble, and nothing appended after it.
    {
        const rendered = try renderChatTemplate(allocator, &messages, &config, tools_json, null, false, null, false);
        defer allocator.free(rendered);
        try testing.expect(std.mem.indexOf(u8, rendered, "Reasoning effort is set to") == null);
        try testing.expect(std.mem.indexOf(u8, rendered, "You have access to the following functions:") != null);
        try testing.expect(std.mem.endsWith(u8, rendered, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        try testing.expect(!promptTailOpensThink(rendered));
    }
}

test "streamShouldBufferForTools: empty + plain prose" {
    try testing.expect(!streamShouldBufferForTools(""));
    try testing.expect(!streamShouldBufferForTools("hello world"));
    try testing.expect(!streamShouldBufferForTools("Let me look at this."));
}

test "streamShouldBufferForTools: trailing partial prefixes of <tool" {
    // The exact regression from the screenshot: model emits `<tool` as a
    // single BPE token, server must buffer instead of streaming it as
    // content (so the next token can either complete the open or be flushed
    // together as harmless prose).
    try testing.expect(streamShouldBufferForTools("<"));
    try testing.expect(streamShouldBufferForTools("<t"));
    try testing.expect(streamShouldBufferForTools("<to"));
    try testing.expect(streamShouldBufferForTools("<too"));
    try testing.expect(streamShouldBufferForTools("<tool"));
    try testing.expect(streamShouldBufferForTools("Let me look at the CLI help:\n<tool"));
}

test "streamShouldBufferForTools: trailing partial prefixes of <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|"));
    try testing.expect(streamShouldBufferForTools("<|t"));
    try testing.expect(streamShouldBufferForTools("<|too"));
    try testing.expect(streamShouldBufferForTools("<|tool"));
    try testing.expect(streamShouldBufferForTools("<|tool_"));
    try testing.expect(streamShouldBufferForTools("<|tool_c"));
    try testing.expect(streamShouldBufferForTools("<|tool_cal"));
}

test "streamShouldBufferForTools: bare <tool> open (DSV4)" {
    try testing.expect(streamShouldBufferForTools("<tool>"));
    try testing.expect(streamShouldBufferForTools("<tool>{\"name\":\"shell\""));
    try testing.expect(streamShouldBufferForTools("<tool name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("hello <tool>{}"));
}

test "streamShouldBufferForTools: canonical <tool_call*> opens" {
    try testing.expect(streamShouldBufferForTools("<tool_call>"));
    try testing.expect(streamShouldBufferForTools("<tool_call>{\"name\":"));
    try testing.expect(streamShouldBufferForTools("<tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<tool_request>"));
    try testing.expect(streamShouldBufferForTools("prose first then <tool_call>"));
}

test "streamShouldBufferForTools: Gemma 4 <|tool_call" {
    try testing.expect(streamShouldBufferForTools("<|tool_call>"));
    try testing.expect(streamShouldBufferForTools("hello <|tool_call name=\"x\">"));
}

test "streamShouldBufferForTools: false positives — <toolkit> etc." {
    // `<toolkit>` and `<toolbar>` have `<tool` as a prefix but the char
    // after is `k` / `b` — not a tag terminator. Must NOT buffer.
    try testing.expect(!streamShouldBufferForTools("here is <toolkit>"));
    try testing.expect(!streamShouldBufferForTools("here is <toolbar>"));
    try testing.expect(!streamShouldBufferForTools("HTML: <toolset>"));
    // But a later real `<tool>` should still trip detection:
    try testing.expect(streamShouldBufferForTools("HTML: <toolset> then <tool>{"));
}

test "streamShouldBufferForTools: raw JSON shape" {
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\",\"arguments\":{}}"));
    try testing.expect(streamShouldBufferForTools("{\"name\":\"shell\""));
    // Just `{` or JSON without `"name"` shouldn't false-trigger.
    try testing.expect(!streamShouldBufferForTools("{"));
    try testing.expect(!streamShouldBufferForTools("{\"foo\":1}"));
}

test "streamThinkGate: template-opened thinking holds tag-free prose" {
    // The Claude Code leak class: with thinking on and the opener injected by
    // the template, the model's prose has NO tags — it must still be held,
    // never flushed as visible text.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The user is asking about their system specs.", true, false));
    // …and split exactly when the close tag arrives.
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("The user is asking.\n</think>\n\nI don't have direct access.", true, false));
}

test "streamThinkGate: after the split, prose flushes (hidden-answer guard)" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("The visible answer streams normally.", true, true));
}

test "streamThinkGate: explicit openers hold even with thinking off or closed" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Sure.<|channel>thought\nlet me reconsider", true, true));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("<think>hmm", false, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("<|channel>thought\nplan<channel|>done", false, false));
}

test "streamThinkGate: partial opener at the buffer tail holds" {
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("The answer is 391.\n<|channel>", false, false));
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("Done. <thi", false, false));
}

test "streamThinkGate: plain prose with thinking off flushes" {
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("17 × 23 = 391.", false, false));
}

test "streamThinkGate: a prompt-opened block is held even with thinking OFF" {
    // LFM2.5 renders `…assistant\n<think>` UNCONDITIONALLY — there is no
    // enable_thinking branch in its template, so the model always reasons and
    // the opener is never in the OUTPUT. The gate saw tag-free prose with
    // thinking off and flushed the entire chain-of-thought as the answer
    // (live 2026-08-04, streaming only — non-streaming strips it because the
    // `</think>` is present by the time the text is split).
    //
    // Whether the prompt ends inside a think block is a fact about the
    // RENDERED BYTES, not about our request flag, which is why it needs its
    // own argument rather than riding `enable_thinking`.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, true));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate2("The user wants 5 words.</think>Hi there, how are you?", false, false, true));

    // Containment, both directions. Same buffers with the prompt flag FALSE
    // behave exactly as before — every other family's template renders the
    // closed `<think></think>` signature (or no tag) when thinking is off, so
    // `promptTailOpensThink` is false for them and this term can never fire.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("The user wants me to say hi in 5 words.", false, false, false));
    // And once the block has closed, a prompt-opened flag must not re-hold the
    // visible answer — that would hide every reply after the first.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate2("Hi there, how are you?", false, true, true));
}

test "unstreamedReasoning: only the remainder, and never a resend" {
    try testing.expectEqualStrings("cde", unstreamedReasoning("abcde", 2).?);
    try testing.expectEqualStrings("abcde", unstreamedReasoning("abcde", 0).?);
    // Nothing new since the last delta.
    try testing.expect(unstreamedReasoning("abcde", 5) == null);
    // The split SHRANK (a tool marker appeared mid-thought and the leaked-markup
    // cut took the tail). Deltas cannot be retracted, so send nothing more —
    // resending from the top would duplicate the whole thought.
    try testing.expect(unstreamedReasoning("ab", 5) == null);
    try testing.expect(unstreamedReasoning("", 0) == null);
}

test "streamTailIsReasoning: an unopened block was content all along" {
    // Gemma 4 with enable_thinking:true answering "391" directly — no opener
    // in the prompt (its template renders a bare `<|turn>model\n`) and none in
    // the output. Streamed, the whole answer arrived as reasoning_content with
    // EMPTY content while non-streaming returned it as content (live
    // 2026-08-04). Content is the correct reading: nothing ever opened.
    try testing.expect(!streamTailIsReasoning(true, false, false));

    // A template-opened block that never closed IS reasoning — the truncated
    // thought must not leak into content (Qwen/LFM2.5 pre-inject the opener).
    try testing.expect(streamTailIsReasoning(true, true, false));
    // A model-emitted opener is the same evidence (Gemma's `<|channel>thought`,
    // a `<think>` the model writes itself), even with no prompt opener.
    try testing.expect(streamTailIsReasoning(true, false, true));
    // Already closed and split ⇒ the tail is the visible answer.
    try testing.expect(!streamTailIsReasoning(false, true, true));
}

test "cutThoughtDelta: a thought cut at its opener delivers nothing, as the split does" {
    // `max_tokens: 1` on a model that opens its own block: the stream holds only the seeded opener.
    try testing.expectEqualStrings("", cutThoughtDelta("<think>", false, false));
    try testing.expectEqualStrings("", cutThoughtDelta("<|channel>thought", false, false));
    try testing.expectEqualStrings("", splitThinkBlock("<think>", true, false).reasoning_content orelse "");
    // A prompt-opened thought cut before its first word, and one cut after it.
    try testing.expectEqualStrings("", cutThoughtDelta("\n", false, false));
    try testing.expectEqualStrings("Ok", cutThoughtDelta("\nOk", false, false));
    // Once the opener is consumed, the same bytes are the thought's own text.
    try testing.expectEqualStrings("<think>", cutThoughtDelta("<think>", true, false));
    // After a delta shipped, the held run (withheld whitespace, a partial close) goes out as is.
    try testing.expectEqualStrings(" \n</thi", cutThoughtDelta(" \n</thi", true, true));
}

test "promptTailOpensThink does not match Gemma's channel opener" {
    // The containment boundary for the prompt-opened-think stream arm: it
    // detects the close with `</think>` ONLY, which is sound exactly because
    // the flag can only be set by a `<think`-family opener. Gemma 4 opens with
    // `<|channel>thought` and closes with `<channel|>`, so it never enters
    // that arm and its streaming is byte-unchanged.
    //
    // If `thinkOpenTagLenAt` ever grows the channel opener, this test fails —
    // and `server.zig`'s arm then needs `<channel|>` in its close check, or a
    // Gemma turn would be held forever waiting for a tag that never arrives.
    try testing.expect(!promptTailOpensThink("<|turn>model\n<|channel>thought\n"));
    try testing.expect(!promptTailOpensThink("<|turn>model\n"));
    // The `<think>` family it DOES match.
    try testing.expect(promptTailOpensThink("<|im_start|>assistant\n<think>"));
}

test "streamThinkGate: the 3-arg form is the 4-arg form with no prompt opener" {
    // The old signature is kept as the no-prompt-opener case so every existing
    // call site and test pins the SAME behavior it always did.
    const cases = [_][]const u8{
        "The user is asking about their system specs.",
        "The user is asking.\n</think>\n\nI don't have direct access.",
        "<think>hmm",
        "17 × 23 = 391.",
        "The answer is 391.\n<|channel>",
    };
    for (cases) |c| {
        for ([_]bool{ true, false }) |think| {
            for ([_]bool{ true, false }) |closed| {
                try testing.expectEqual(streamThinkGate(c, think, closed), streamThinkGate2(c, think, closed, false));
            }
        }
    }
}

test "split content: trailing <channel|> close-marker spam never leaks (degenerate model output)" {
    // Live soak capture (record 2151, a Gemma reasoning variant): the model emitted
    // reasoning, one <channel|> close, a scrap of content, then SPAMMED 16 more bare
    // <channel|> close markers. The leading cut takes the FIRST close, and the
    // trailing-open cut only handles unclosed OPENERS — so the 16 stray CLOSE markers
    // leaked into visible content. A close marker is never valid at the tail of
    // content; trailing closers (and openers) are trimmed so <channel|> can't leak.
    const text = "Reasoning here about the file.\n<channel|>`glob` `./game.js`\n\n" ++
        "<channel|><channel|><channel|><channel|><channel|>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "<channel|>") == null);
    // The scrap of real content before the spam survives.
    try testing.expect(std.mem.indexOf(u8, content, "glob") != null);
}

test "split content: trailing </think> close spam never leaks" {
    const text = "<think>plan</think>The answer.</think></think></think>";
    const content = splitThinkBlock(text, true, false).content;
    try testing.expect(std.mem.indexOf(u8, content, "</think>") == null);
    try testing.expect(std.mem.indexOf(u8, content, "The answer.") != null);
}

test "split content: orphan Gemma <tool_call|> close never leaks (no-tag-leak)" {
    // Live 2026-07-16 soak (gemma-4-26B-A4B-it-qat-4bit, tools present, a "no
    // tools needed" probe at temp 0.7): the model degenerated into a bare
    // 1-token <tool_call|> CLOSE with NO <|tool_call> opener, so parseToolCalls
    // found no call and the orphan control token leaked as the ENTIRE visible
    // content (server response content == "<tool_call|>"). A tool CLOSE marker is
    // never valid at the tail of content — same class as the <channel|> spam.
    // parseToolCalls already extracted any real call before this split runs, so
    // any residual <tool_call|> here is orphan by construction.
    try testing.expectEqualStrings("", splitThinkBlock("<tool_call|>", true, false).content);
    try testing.expectEqualStrings("Sure.", splitThinkBlock("Sure.<tool_call|>", true, false).content);
    // Legit prose (no control token) is untouched.
    try testing.expectEqualStrings("2 + 2 = 4", splitThinkBlock("2 + 2 = 4", true, false).content);
}

test "splitThinkBlock content never keeps trailing channel close spam" {
    const text = "<|channel>thought\nplan<channel|>\n<|channel>\nThe answer.<channel|><channel|><channel|>";
    const split = splitThinkBlock(text, true, false);
    try testing.expect(std.mem.indexOf(u8, split.content, "<channel|>") == null);
    try testing.expect(std.mem.indexOf(u8, split.content, "The answer.") != null);
}

test "streamThinkGate holds a partial re-opened <think at the tail" {
    // Template-opened thinking: reasoning streams in with no literal opener.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("partial reasoning", true, false));
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGate("reasoning</think>ans", true, false));
    // Mid-text re-open arriving token by token: the partial tail must hold the
    // flush (leaking "<thin" into visible text is the tag-leak class).
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGate("visible<thin", false, true));
    // Prose that merely mentions "<think about it" is NOT a partial opener.
    try testing.expectEqual(StreamThinkGate.flush_text, streamThinkGate("let's <think about it", false, true));
}

test "streamShouldBufferForTools: lone DSML closer holds, ordinary closing tags flush" {
    try testing.expect(streamShouldBufferForTools("Done.\n\n</｜DSML｜ parameter>"));
    try testing.expect(streamShouldBufferForTools("Done. </"));
    try testing.expect(streamShouldBufferForTools("Done. </｜DS"));
    try testing.expect(streamShouldBufferForTools("Done. </｜DSML\xef\xbd"));
    try testing.expect(!streamShouldBufferForTools("Use </div> here"));
    try testing.expect(!streamShouldBufferForTools("a </b> then more"));
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n\n</｜DSML｜ calls>"));
}

test "streamShouldBufferForTools: DSML marker and partial fullwidth prefixes hold" {
    try testing.expect(streamShouldBufferForTools("Sure.\n\n<｜DSML｜tool_calls>"));
    try testing.expect(streamShouldBufferForTools("<｜DSML｜invoke name=\"x\">"));
    // partial fullwidth-bar prefix at the tail — next token may complete it
    try testing.expect(streamShouldBufferForTools("answer text <｜DSML"));
    try testing.expect(streamShouldBufferForTools("answer text <｜"));
    // plain prose with a fullwidth bar elsewhere must NOT hold
    try testing.expect(!streamShouldBufferForTools("the ｜ character is fun"));
}

test "renderChatTemplate: thinking-off closes a template-opened think block" {
    // The GLM template opens `<think>` unconditionally; thinking-off used to
    // mean generating the whole reasoning pass and discarding it. Closing the
    // block in the prompt skips it. Thinking ON keeps the opener untouched.
    const allocator = testing.allocator;
    var config = ChatConfig{
        .chat_template = @embedFile("fixtures/glm53_chat_template.jinja"),
        .bos_token = null,
        .eos_token = null,
        .add_bos_token = false,
        .allocator = allocator,
    };
    const messages = [_]Message{.{ .role = "user", .content = "hi" }};

    const off = try renderChatTemplate(allocator, &messages, &config, null, null, false, null, false);
    defer allocator.free(off);
    try testing.expect(std.mem.endsWith(u8, off, "<|assistant|><think></think>"));
    try testing.expect(!promptTailOpensThink(off));

    const on = try renderChatTemplate(allocator, &messages, &config, null, null, true, null, false);
    defer allocator.free(on);
    try testing.expect(std.mem.endsWith(u8, on, "<|assistant|><think>"));
}

test "streamShouldBufferForTools: every prefix of a bare <invoke> call holds; prose releases" {
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls</parameter>\n</invoke>\n</function_calls>";
    for ([_]usize{ 0, std.mem.indexOf(u8, text, "<invoke").? }) |start| {
        var end = start + 2;
        while (end <= text.len) : (end += 1) {
            testing.expect(streamShouldBufferForTools(text[start..end])) catch |e| {
                std.debug.print("gate flushed a prefix of an accepted call: '{s}'\n", .{text[start..end]});
                return e;
            };
        }
    }
    try testing.expect(!streamShouldBufferForTools("I <invoked> the <function_callback> helper"));
}

test "ThinkScan: incremental gate matches the fresh gate at every prefix length" {
    // The gate is O(buffer) per token today (measured 47.99 us/token at 113 KB
    // with no close in the buffer), i.e. O(n^2) over a reasoning block. A
    // cursor makes it O(new bytes) — but only if it is EXACT, and the failure
    // a naive cursor introduces is a marker split across two token arrivals.
    // Feeding prefix-by-prefix IS that case, at every possible split point.
    const corpus = [_][]const u8{
        "plain visible prose with no markup at all",
        "<think>reasoning here</think>the answer",
        // A </think> inside a tool ARGUMENT is payload, not a block close.
        "<think>plan<tool_call>f<arg_key>k</arg_key><arg_value>closes with </think> inside</arg_value></tool_call>done</think>visible",
        "<|channel>thought hidden<channel|>visible",
        "prose where a < b and nothing else happens",
        "trailing partial opener at the end <thi",
        "trailing partial close </thin",
    };
    const flags = [_][2]bool{ .{ true, false }, .{ false, false }, .{ false, true } };
    for (corpus) |full| {
        for (flags) |f| {
            const enable_thinking = f[0];
            const prompt_opened = f[1];
            var scan: ThinkScan = .{};
            var n: usize = 0;
            while (n <= full.len) : (n += 1) {
                const buf = full[0..n];
                const want = streamThinkGate2(buf, enable_thinking, false, prompt_opened);
                const got = streamThinkGateScan(buf, enable_thinking, false, prompt_opened, &scan);
                if (want != got) {
                    std.debug.print("prefix {d} of \"{s}\" (think={}, opened={}): {any} != {any}\n", .{ n, full, enable_thinking, prompt_opened, got, want });
                    return error.IncrementalGateDiverged;
                }
            }
        }
    }
}

test "ThinkScan: per-token work is FLAT in buffer size, not linear" {
    // The whole point. A smaller constant is not the fix — the re-examined
    // span must not grow with the buffer, or a 32K-token thought pays 8x what
    // the 113 KB bench measured.
    var scan: ThinkScan = .{};
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(testing.allocator);
    var max_span: usize = 0;
    var i: usize = 0;
    while (i < 400) : (i += 1) {
        // A think block that never closes — the expensive arm of the bench.
        try buf.appendSlice(testing.allocator, if (i == 0) "<think>" else "token words and more ");
        _ = streamThinkGateScan(buf.items, true, false, false, &scan);
        if (i > 0 and scan.last_scan_span > max_span) max_span = scan.last_scan_span;
    }
    try testing.expect(buf.items.len > 8_000);
    // Bounded by one token's bytes plus the trailing partial-marker candidate.
    try testing.expect(max_span <= 64);

    // A growing partial marker at the tail is re-examined, and only it.
    var scan2: ThinkScan = .{};
    var buf2 = std.ArrayList(u8).empty;
    defer buf2.deinit(testing.allocator);
    try buf2.appendSlice(testing.allocator, "<think>");
    var j: usize = 0;
    while (j < 500) : (j += 1) try buf2.appendSlice(testing.allocator, "filler ");
    _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
    for ("</think:opensou") |c| {
        try buf2.append(testing.allocator, c);
        _ = streamThinkGateScan(buf2.items, true, false, false, &scan2);
        try testing.expect(scan2.last_scan_span <= 64);
    }
}

test "ThinkScan: reset() re-arms the cursor when the stream buffer is cleared" {
    // The buffer is NOT append-only — the stream loop trims and resets it at
    // every emit. A stale cursor then points past the end or into different
    // bytes, which is the second failure mode a naive cursor introduces.
    var scan: ThinkScan = .{};
    const first = "<think>reasoning</think>";
    try testing.expectEqual(StreamThinkGate.split_think, streamThinkGateScan(first, true, false, false, &scan));
    scan.reset();
    // Same scan struct, fresh (shorter) buffer: must not carry the old close.
    try testing.expectEqual(StreamThinkGate.hold_thinking, streamThinkGateScan("<think>more", true, false, false, &scan));
}

test "streaming content agrees with the non-streaming split on leading whitespace" {
    // LFM2.5 class: the template opens <think> unconditionally, so the first
    // generated token is the newline pair after the closer. The non-streaming
    // split drops it; the stream must reach the same bytes.
    const out = "\n\nOne, two, three.";
    const expected = splitThinkBlock(out, true, false).content;
    try testing.expectEqualStrings("One, two, three.", expected);

    const tokens = [_][]const u8{ "\n\n", "One", ", two", ", three." };
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(testing.allocator);
    var content_started = false;
    for (tokens) |tt| {
        const emit = streamContentLead(tt, content_started);
        if (emit.len == 0) continue;
        content_started = true;
        try streamed.appendSlice(testing.allocator, emit);
    }
    try testing.expectEqualStrings(expected, streamed.items);
}
test "streamContentLead never touches whitespace inside the answer" {
    // Only the LEAD is suppressed: once anything visible has been sent, a
    // newline token is ordinary content (markdown paragraphs, code blocks).
    try testing.expectEqualStrings("\n\n", streamContentLead("\n\n", true));
    try testing.expectEqualStrings("  x", streamContentLead("  x", true));
    // And a lead that is not whitespace is untouched.
    try testing.expectEqualStrings("Hello", streamContentLead("Hello", false));
    // A token is never cut in half: one carrying whitespace AND text rides
    // whole, so its logprobs entry still describes bytes that reached content.
    try testing.expectEqualStrings("\nHello", streamContentLead("\nHello", false));
}

test "streamShouldBufferForTools: MiniCPM5 <function name=...> open" {
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">"));
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">\n  <param name=\"command\">pwd"));
    try testing.expect(streamShouldBufferForTools("<function name=\"shell\">\n  <param name=\"command\">pwd</param></functio"));
    try testing.expect(streamShouldBufferForTools("Sure, running that:\n<function name=\"shell\">"));
}

test "streamShouldBufferForTools: EVERY growing prefix of <function holds" {
    // DERIVED, not enumerated. The previous version of this test was a second
    // hand-written copy of `tail_prefixes` and inherited its hole: `<funct` was
    // missing from BOTH, so a token boundary landing exactly there flushed the
    // fragment and leaked the rest of the tag (`<f` id 54303 + `unct` id 14185
    // decode to exactly `<funct` in this vocabulary — a real decomposition, not
    // a hypothetical). Generating the assertions from the marker string means a
    // rung missing from the array fails here even if nobody remembers to add an
    // assertion for it.
    const marker = "<function";
    var i: usize = 2;
    while (i <= marker.len) : (i += 1) {
        try testing.expect(streamShouldBufferForTools(marker[0..i]));
    }
}

test "streamShouldBufferForTools: MiniCPM5 opener holds while unresolved, releases when resolved-and-not-a-call" {
    // HOLD — the opener is either unresolved (no `>` yet, so the next token
    // could still complete a real call) or resolved WITH a quoted name.
    const hold = [_][]const u8{
        "<f",                        "<fu",
        "<fun",                      "<func",
        "<funct",                    "<functi",
        "<functio",                  "<function",
        "<function ",                "<function n",
        "<function na",              "<function name",
        "<function name=",           "<function name=\"",
        "<function name=\"shell",    "<function name=\"shell\">",
        // Parser-ACCEPTED oddity: miniCpm5AttrValue finds `name=` anywhere in
        // the opener, so this IS a real call. The gate must therefore hold it —
        // a gate stricter than its parser flushes a genuine call's opener as
        // content and then the end-of-stream parse emits the call anyway: leak
        // AND duplicate, the exact class this work exists to fix.
        "<function foo name=\"x\">",
    };
    for (hold) |buf| {
        testing.expect(streamShouldBufferForTools(buf)) catch |e| {
            std.debug.print("expected HOLD, got flush: '{s}'\n", .{buf});
            return e;
        };
    }

    // RELEASE — resolved (`>` seen) and demonstrably not a MiniCPM5 call, or
    // never this dialect at all. Holding these buys nothing and costs the whole
    // remainder of the response: the hold is monotonic, so one `<function foo>`
    // in prose silences every later token of a tool-enabled turn.
    const release = [_][]const u8{
        "<functional",
        "<function-like>",
        "<function>",
        "<function foo>",
        "<function >",
        "<function name=unquoted>",
        // Empty name: the parser rejects it (`fn_name.len == 0` → skip), so the
        // gate releases too. The strictly-earlier prefix `<function name="` is
        // pinned in the hold set above, so the unresolved form still holds.
        "<function name=\"\">",
        "Use a <function foo> block to wrap it, then carry on writing prose.",
    };
    for (release) |buf| {
        testing.expect(!streamShouldBufferForTools(buf)) catch |e| {
            std.debug.print("expected RELEASE, got hold: '{s}'\n", .{buf});
            return e;
        };
    }
}

test "streamShouldBufferForTools: false positive — bare <function> with no attribute" {
    try testing.expect(!streamShouldBufferForTools("here is <function>"));
    try testing.expect(!streamShouldBufferForTools("Wrap it in a <functional> block"));
}

test "a model-opened think block latches from its own opener token" {
    // Both openers are dropped from visible deltas as channel markers, so without
    // this latch the whole thought streams as the answer (LFM2.5-8B-A1B, Gemma 4).
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>").?);
    try std.testing.expectEqualStrings("<think>", modelThinkOpener(false, "<think>\n").?);
    try std.testing.expectEqualStrings("<|channel>thought", modelThinkOpener(true, "thought").?);
    try std.testing.expect(modelThinkOpener(false, "thought") == null);
    // Prose that merely BEGINS with the word is not the opener token.
    try std.testing.expect(modelThinkOpener(true, "thoughts") == null);
    try std.testing.expect(modelThinkOpener(true, "\n") == null);
    try std.testing.expect(modelThinkOpener(false, "<think>x") == null);
}
