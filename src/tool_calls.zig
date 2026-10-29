const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");
const log = @import("log.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const think_tags = @import("think_tags.zig");
const tagLenAt = think_tags.tagLenAt;
const indexOfThinkCloseTag = think_tags.indexOfThinkCloseTag;
const splitThinkBlock = think_tags.splitThinkBlock;
const streamShouldBufferForTools = think_tags.streamShouldBufferForTools;
const testing = std.testing;
const chat = @import("chat.zig");
const ChatConfig = chat.ChatConfig;
const Dsv41Replay = chat.Dsv41Replay;
const Message = chat.Message;
const ParsedToolCall = chat.ParsedToolCall;
const ReplayCase = chat.ReplayCase;
const ToolCall = chat.ToolCall;
const appendJsonString = chat.appendJsonString;
const dsv41_image_atom = chat.dsv41_image_atom;
const jsonIsValidObject = chat.jsonIsValidObject;
const loadChatConfig = chat.loadChatConfig;
const renderChatTemplate = chat.renderChatTemplate;
const replayTurn = chat.replayTurn;
const synthesizeToolFallbackMessages = chat.synthesizeToolFallbackMessages;
const toolNameIsDeclared = chat.toolNameIsDeclared;

/// `<tool_call>` followed by a special token spelled where `<` belongs
/// (`<|im_start|>function=x>`): the copy with that token read as `<`, or null when none.
fn hermesSpecialOpenerRepair(allocator: std.mem.Allocator, text: []const u8) !?[]u8 {
    const wrapper = "<tool_call>";
    var out: ?std.ArrayList(u8) = null;
    errdefer if (out) |*o| o.deinit(allocator);
    var copied: usize = 0;
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, text, from, wrapper)) |at| {
        var p = at + wrapper.len;
        from = p;
        while (p < text.len and std.ascii.isWhitespace(text[p])) p += 1;
        if (!std.mem.startsWith(u8, text[p..], "<|")) continue;
        const close = std.mem.indexOfPos(u8, text, p + 2, "|>") orelse continue;
        if (close - p > 32 or std.mem.indexOfAny(u8, text[p + 2 .. close], "<>\n ") != null) continue;
        if (!std.mem.startsWith(u8, text[close + 2 ..], "function=")) continue;
        if (out == null) out = .empty;
        try out.?.appendSlice(allocator, text[copied..p]);
        try out.?.append(allocator, '<');
        copied = close + 2;
        from = copied;
    }
    var o = out orelse return null;
    try o.appendSlice(allocator, text[copied..]);
    return try o.toOwnedSlice(allocator);
}

/// Parse tool calls from model output text.
pub fn parseToolCalls(allocator: std.mem.Allocator, text: []const u8) !?[]ParsedToolCall {
    // Strip thinking blocks if present
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (indexOfThinkCloseTag(text, 0)) |close| {
        effective_text = std.mem.trimStart(u8, text[close.pos + close.len ..], "\n ");
    }

    const special_fixed = try hermesSpecialOpenerRepair(allocator, effective_text);
    defer if (special_fixed) |r| allocator.free(r);
    if (special_fixed) |r| effective_text = r;

    var calls = std.ArrayList(ParsedToolCall).empty;
    errdefer {
        for (calls.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        calls.deinit(allocator);
    }

    // Look for `<tool…>…</tool…>` patterns. DSV4-Flash hallucinates the
    // spec aggressively; we accept the entire training-bias family:
    //   • `<tool_call>JSON</tool_call>`                      (canonical Hermes)
    //   • `<tool_call name="X">JSON</tool_call>`             (attribute style)
    //   • `<tool_call …>JSON</tool_request>`                 (mismatched close)
    //   • `<tool_calls>JSON</tool_calls>`                    (plural form)
    //   • `<tool_calls name="X">{args}</tool_calls>`         (plural + attr)
    //   • `<tool name="X">{args}</tool_calls>`               (bare-tool open, plural close — nested wrappers)
    //   • `<tool>{json}</tool>`                              (bare tag, body has name+args)
    //   • `<tool name="X" arguments="{...}"/>`               (SELF-CLOSING — DSV4 emits this routinely;
    //                                                          parseSelfClosingToolTag handles broken-quote
    //                                                          variants like `…"}'/>` too)
    // Approach: search for the prefix `<tool` (5 chars). Skip the Gemma 4
    // close marker `<tool_call|>`. Try the self-closing form first (cheap
    // attribute scan + balanced-JSON extraction; tolerates the model's
    // mismatched quotes). If that doesn't match, fall through to the
    // open/close-marker path: scan to the next `>`, optionally capturing
    // `name="X"` / `name='X'`, then locate the earliest of `</tool>`,
    // `</tool_call>`, `</tool_calls>`, `</tool_request>`, `</tool_requests>`.
    // When the body fails to parse, advance only past the OPENING `>` so any
    // inner `<tool…>` blocks get a chance — this is what makes the outer
    // `<tool_calls>` wrapper case work.
    //
    // DeepSeek-V4 native DSML (`<｜DSML｜invoke …>`): distinctive fullwidth-bar
    // marker no other family emits, and the generic `<tool` scan never sees
    // it (the byte before "tool_calls" is `｜`, not `<`).
    try parseDsmlToolCalls(allocator, effective_text, &calls);
    // GLM's bare `<tool_call>NAME<arg_key>…` format is tried FIRST among the
    // tag families: the generic `<tool` scan below would misread its non-JSON
    // arg_key/arg_value body. When it matched, the generic scans are skipped
    // but the shared safety net at the bottom still runs.
    try parseArgKeyToolCalls(allocator, effective_text, &calls);
    var search_pos: usize = if (calls.items.len > 0) effective_text.len else 0;
    while (search_pos < effective_text.len) {
        const rel = std.mem.indexOf(u8, effective_text[search_pos..], "<tool") orelse break;
        const after_tool = search_pos + rel + "<tool".len;
        if (after_tool >= effective_text.len) break;

        // Reject anything that isn't actually a tool open tag — the char
        // right after `<tool` must be `>`, whitespace, `_` (suffix coming),
        // or `|` (Gemma 4 close marker). Anything else is text like
        // `<toolkit>` or `<toolbar>`; advance past `<tool` and keep scanning.
        const next = effective_text[after_tool];
        if (next != '>' and next != ' ' and next != '\t' and next != '\n' and next != '_' and next != '|') {
            search_pos = after_tool;
            continue;
        }
        // Gemma 4 close marker: `<tool_call|>`. Detected when after `<tool`
        // we see `_call|`. Let the Gemma 4 branch below pick this up.
        if (next == '_'
            and after_tool + 6 <= effective_text.len
            and std.mem.eql(u8, effective_text[after_tool .. after_tool + 6], "_call|"))
        {
            search_pos = after_tool + 6;
            continue;
        }
        // If suffix is `_`, the only accepted continuations are
        // `_call`, `_calls`, `_request`, `_requests`. Any OTHER suffix gets
        // one more chance as the XML-element-TAG form before being rejected:
        // DSV4 embeds the tool name in the tag itself —
        //   `<tool_read>\n<path>mlx.html</path>\n</tool_read>`
        // (2026-06-10 pi html-ds4 capture; both calls leaked as text and pi
        // scored 0/4). Conditions kept tight: tag must close with `>` right
        // after the name (no attributes), an EXACT `</tool_NAME>` close must
        // exist, the name must not be a result-ish marker (`<tool_output>`
        // is DSV4 hallucinating a result, not calling a tool named
        // "output"), and the body must be entirely `<key>value</key>` args.
        if (next == '_') {
            const accepted = [_][]const u8{ "_call", "_calls", "_request", "_requests" };
            var matched = false;
            for (accepted) |suf| {
                if (after_tool + suf.len > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[after_tool .. after_tool + suf.len], suf)) continue;
                // Char right after the suffix must be tag-terminating; otherwise
                // `_call` could swallow `<tool_called_X>`.
                const post = after_tool + suf.len;
                if (post >= effective_text.len) break;
                const c = effective_text[post];
                if (c == '>' or c == ' ' or c == '\t' or c == '\n' or c == '/' or c == '|') {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                const origin = after_tool - "<tool".len;
                if (try parseXmlElementTagToolCall(allocator, effective_text[origin..])) |etc| {
                    try calls.append(allocator, .{ .name = etc.call.name, .arguments = etc.call.arguments });
                    search_pos = origin + etc.consumed;
                    continue;
                }
                search_pos = after_tool;
                continue;
            }
        }

        // Self-closing `<tool …/>` form first — DSV4 emits this routinely
        // and the open/close scan below won't find a `</tool*>` close, so
        // without this branch the call gets dropped silently.
        const tag_origin = after_tool - "<tool".len;
        if (parseSelfClosingToolTag(effective_text[tag_origin..])) |sc| {
            const name_owned = try allocator.dupe(u8, sc.name);
            errdefer allocator.free(name_owned);
            const args_owned = try allocator.dupe(u8, sc.arguments);
            try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
            search_pos = tag_origin + sc.consumed;
            continue;
        }

        // Scan the opening tag to its `>`, capturing optional name attr.
        var attr_name: ?[]const u8 = null;
        var content_start: usize = 0;
        {
            var i: usize = after_tool;
            // Bound the scan so a malformed input can't run wild.
            const limit = @min(effective_text.len, after_tool + 256);
            while (i < limit and effective_text[i] != '>') : (i += 1) {
                if (effective_text[i] != 'n') continue;
                if (i + 5 > effective_text.len) continue;
                if (!std.mem.eql(u8, effective_text[i .. i + 5], "name=")) continue;
                const q_pos = i + 5;
                if (q_pos >= effective_text.len) break;
                const quote = effective_text[q_pos];
                if (quote != '"' and quote != '\'') continue;
                const val_start = q_pos + 1;
                const val_end_rel = std.mem.indexOfScalar(u8, effective_text[val_start..], quote) orelse break;
                attr_name = effective_text[val_start .. val_start + val_end_rel];
                i = val_start + val_end_rel;
            }
            if (i >= limit or effective_text[i] != '>') {
                // Unclosed opening tag — skip past `<tool` and keep going.
                search_pos = after_tool;
                continue;
            }
            content_start = i + 1;
        }
        // Find the close marker. Prefer the close matching the OPENING tag's
        // name (`<tool_calls>` closes at `</tool_calls>`, never at a
        // `</tool_name>` child element inside the body — the XML-element
        // form puts `</tool…>`-shaped children first). When no exact match
        // exists, fall back to the earliest `</tool…>` with ANY suffix of
        // word characters: DSV4 hallucinates closes freely (`</tool_action>`
        // was captured live closing a `<tool_call>` open), and pinning an
        // exact-name list drops the whole call and leaks it as visible text.
        var close_rel: ?usize = null;
        var close_len: usize = 0;
        {
            const hay = effective_text[content_start..];
            // Opening tag name: "tool" + suffix up to the attr/`>` delimiter.
            var name_end = after_tool;
            while (name_end < effective_text.len and
                (std.ascii.isAlphanumeric(effective_text[name_end]) or effective_text[name_end] == '_')) : (name_end += 1)
            {}
            const open_name = effective_text[tag_origin + 1 .. name_end];
            var exact_buf: [40]u8 = undefined;
            if (open_name.len + 3 <= exact_buf.len) {
                const exact = std.fmt.bufPrint(&exact_buf, "</{s}>", .{open_name}) catch unreachable;
                var from: usize = 0;
                while (std.mem.indexOfPos(u8, hay, from, exact)) |found| {
                    // A `<parameter=…>` value may spell the wrapper's own close.
                    if (hermesParamSpanEnclosing(hay, found)) |past| {
                        from = past;
                        continue;
                    }
                    close_rel = found;
                    close_len = exact.len;
                    break;
                }
            }
            if (close_rel == null) {
                var cpos: usize = 0;
                while (std.mem.indexOf(u8, hay[cpos..], "</tool")) |found| {
                    const open_at = cpos + found;
                    var j = open_at + "</tool".len;
                    const jlimit = @min(hay.len, j + 24);
                    var word_only = true;
                    while (j < jlimit and hay[j] != '>') : (j += 1) {
                        const c = hay[j];
                        if (!std.ascii.isAlphanumeric(c) and c != '_') {
                            word_only = false;
                            break;
                        }
                    }
                    if (word_only and j < jlimit and hay[j] == '>') {
                        if (hermesParamSpanEnclosing(hay, open_at)) |past| {
                            cpos = past;
                            continue;
                        }
                        close_rel = open_at;
                        close_len = j + 1 - open_at;
                        break;
                    }
                    cpos = open_at + "</tool".len;
                }
            }
        }
        if (close_rel == null) {
            // No `</tool…>` close in sight. Two ways this happens in the
            // wild: (1) the model emitted EOS mid-tool-call (`<tool_call>\n{
            // …well-formed args…}` then nothing); (2) max_tokens truncated.
            // The args object itself is usually intact — snap a balanced
            // JSON object from right after the open tag and try to parse it.
            // Function-tag form first: a parameter VALUE is arbitrary bytes and never names the tool.
            const truncated_fn_tag = std.mem.indexOf(u8, effective_text[content_start..], "<function=") != null;
            const truncated_json: ?[]const u8 = if (truncated_fn_tag) null else balancedJsonObject(effective_text[content_start..]);
            if (truncated_json) |json_body| {
                const json_off = @intFromPtr(json_body.ptr) - @intFromPtr(effective_text[content_start..].ptr);
                const advance_to = content_start + json_off + json_body.len;
                // Hermes shape (top-level "name" + "arguments"): use as-is.
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                    search_pos = advance_to;
                    continue;
                }
                // Attribute shape — body is JUST the args, take name from attr.
                if (attr_name) |an| {
                    const name_owned = try allocator.dupe(u8, an);
                    errdefer allocator.free(name_owned);
                    const args_owned = try allocator.dupe(u8, json_body);
                    try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                    search_pos = advance_to;
                    continue;
                }
            }
            // Truncated Hermes/XML tool call: the model emitted
            // `<tool_call><function=NAME><parameter=KEY>…` (Hermes function-tag)
            // or the XML-element `<tool_name>NAME</tool_name>…` shape and ran
            // out of tokens before ANY closing tag — so there's no balanced
            // JSON to snap. The OPENING tags still carry the tool NAME; recover
            // that (parseHermesToolCall breaks out of its parameter loop on a
            // missing `</parameter>`, yielding name + `{}` args) so the call is
            // recognized as a truncated writeFile instead of being DROPPED and
            // leaked into visible content (live JFK-novel capture, 2026-06-20:
            // a 19k-char writeFile cut off mid-content was silently lost and
            // the app fired the wrong "malformed tag" ghost nudge). Recovering
            // the name is enough for the client to fire the right chunk/append
            // nudge; we do NOT salvage the partial content (a half-written file
            // is worse than a re-issued chunked write). The text is truncated
            // to its end here, so advance past it to terminate the scan.
            if (parseHermesToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            if (parseXmlElementToolCall(allocator, effective_text[content_start..])) |tc| {
                try calls.append(allocator, tc);
                search_pos = effective_text.len;
                continue;
            }
            // JSON dialect cut off INSIDE the object (EOS mid-string on a
            // large edit): nothing balances, but a top-level `"name"` key is
            // still the tool. NAME + `{}`, never a fragment.
            if (truncatedJsonCallName(effective_text[content_start..])) |name| {
                const name_owned = try allocator.dupe(u8, name);
                errdefer allocator.free(name_owned);
                const args_owned = try allocator.dupe(u8, "{}");
                try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                search_pos = effective_text.len;
                continue;
            }
            break;
        }
        const content = std.mem.trim(u8, effective_text[content_start .. content_start + close_rel.?], " \t\n\r");

        // Pre-clean the body before the parse attempts. Two DSV4 quirks:
        //   • The args object may be wrapped in `<parameters>…</parameters>`.
        //   • The model sometimes emits an extra trailing `}` after the
        //     proper JSON close. Find the first balanced JSON object via
        //     depth tracking and use just that.
        const unwrapped = stripParametersWrapper(content);
        // Some models (often when echoing a Jinja `{{ }}` example) wrap the args
        // object in an extra brace layer: `<tool_call>{{"name":…}}</tool_call>`.
        // Strip one layer so the body parses as JSON. Only fires on the `{{…}}`
        // shape (which otherwise fails to parse), so valid single-brace bodies are
        // untouched. Mirrors the unwrap in `parseGemma4ToolCall`.
        const balanced: []const u8 = unwrapDoubleBraces(balancedJsonObject(unwrapped) orelse unwrapped);

        // Try to extract a tool call from the body. Three shapes, tried in
        // priority order:
        //   1. Attribute form (`name="X"` captured, body is JUST args).
        //   2. Canonical Hermes JSON ({"name":"X","arguments":{...}}).
        //   3. Hermes function-tag format (parseHermesToolCall).
        var parsed_ok = false;
        // 0. Function-tag form FIRST when the body actually carries a
        // `<function=` opener. The dialect qwen 3.5/3.6's own template
        // mandates rides inside the SAME `<tool_call>` wrapper the JSON form
        // uses, and a `<parameter=…>` VALUE is arbitrary bytes — a written
        // package.json made `balancedJsonObject` snap the FILE CONTENT, whose
        // "name" key then became the tool name (live 2026-08-12: pi answered
        // "Tool voxel-pagoda-garden not found" and the model looped on it).
        // The JSON shapes below can never contain that opener, so this only
        // reorders the case that was being read wrong.
        if (std.mem.indexOf(u8, content, "<function=") != null) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (attr_name) |an| {
                if (std.json.parseFromSlice(std.json.Value, allocator, balanced, .{})) |parsed| {
                    defer parsed.deinit();
                    if (parsed.value == .object) {
                        const name_owned = try allocator.dupe(u8, an);
                        errdefer allocator.free(name_owned);
                        const args_owned = try allocator.dupe(u8, balanced);
                        try calls.append(allocator, .{ .name = name_owned, .arguments = args_owned });
                        parsed_ok = true;
                    }
                } else |_| {}
            }
        }
        if (!parsed_ok) {
            if (tryParseJsonToolCall(allocator, balanced)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseHermesToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }
        if (!parsed_ok) {
            if (parseXmlElementToolCall(allocator, content)) |tc| {
                try calls.append(allocator, tc);
                parsed_ok = true;
            }
        }

        if (parsed_ok) {
            // Body consumed cleanly — advance past the close marker.
            search_pos = content_start + close_rel.? + close_len;
        } else {
            // Body wasn't a tool call. Advance only past the opening `>`
            // so inner `<tool…>` blocks (if any) still get parsed. This is
            // what makes the outer-wrapper case work: the outer
            // `<tool_calls>` has no JSON body, but its content contains
            // real `<tool name="X">` inner calls.
            search_pos = content_start;
        }
    }

    // Gemma 4 format: <|tool_call>call:name{args}<tool_call|>
    if (calls.items.len == 0) {
        search_pos = 0;
        while (search_pos < effective_text.len) {
            const tag_start = std.mem.indexOf(u8, effective_text[search_pos..], "<|tool_call>") orelse break;
            const content_start = search_pos + tag_start + "<|tool_call>".len;
            const tag_end_opt = std.mem.indexOf(u8, effective_text[content_start..], "<tool_call|>");
            const content = if (tag_end_opt) |tag_end|
                std.mem.trim(u8, effective_text[content_start .. content_start + tag_end], " \t\n\r")
            else
                // Incomplete tool call (model hit EOS before closing tag) — use rest of text
                std.mem.trim(u8, effective_text[content_start..], " \t\n\r");

            search_pos = if (tag_end_opt) |tag_end|
                content_start + tag_end + "<tool_call|>".len
            else
                effective_text.len;

            // No <tool_call|> close = the generation was CUT mid-call (EOS,
            // max_tokens, or the degenerate-tail-loop guard) — fragment values
            // are dropped inside the parse, only completed args ship.
            if (parseGemma4ToolCall(allocator, content, tag_end_opt == null)) |tc| {
                try calls.append(allocator, tc);
            } else {
                log.info("  [tool-parse] Gemma4 parse FAILED for: {s}\n", .{content[0..@min(content.len, 200)]});
            }
        }
    }

    // Bare Hermes function tag with NO <tool_call> wrapper. The outer scan
    // triggers on the substring `<tool`, which a lone `<function=…></function>`
    // (even with a trailing `</tool_call>`, since that is `</too…`, not `<tool`)
    // never provides — so a model that drops the OPENING <tool_call> would leak
    // the whole call as visible text. Only fire when a `<function=` opener AND a
    // `<parameter=` are both present, so prose that merely mentions the tag can't
    // false-fire (parseHermesToolCall needs a real parameter to recover args).
    if (calls.items.len == 0) {
        if (std.mem.indexOf(u8, effective_text, "<function=") != null and
            std.mem.indexOf(u8, effective_text, "<parameter=") != null)
        {
            if (parseHermesToolCall(allocator, effective_text)) |tc| {
                try calls.append(allocator, tc);
            }
        }
    }

    // MiniCPM5 V3 XML: `<function name="X"><param name="K">V</param>…</function>`,
    // zero or more consecutive calls, no outer wrapper. Tried only after every
    // other explicit tag format above has had its shot, and before the
    // raw-JSON guessing fallback below — an explicit recognized tag always
    // wins over heuristic inference.
    if (calls.items.len == 0) {
        if (std.mem.indexOf(u8, effective_text, "<function") != null) {
            try parseMiniCpm5ToolCalls(allocator, effective_text, &calls);
        }
    }

    // Inferred `<invoke>` runs last among the tag families: a native call can carry
    // invoke-shaped text inside a value.
    if (calls.items.len == 0) try parseInvokeToolCalls(allocator, effective_text, &calls, BARE_INVOKE);

    // If no <tool_call> tags, try to find raw JSON tool call(s).
    if (calls.items.len == 0) {
        var trimmed = std.mem.trim(u8, effective_text, " \t\n\r");
        if (std.mem.startsWith(u8, trimmed, "</tool_call>")) {
            trimmed = std.mem.trim(u8, trimmed["</tool_call>".len..], " \t\n\r");
        }
        // JSON ARRAY of {name, arguments} objects — parallel tool calls from
        // models without a trained tool format. Gemma 3 emits a ```json fence
        // around the array; skip one leading fence line before the check so
        // the array (not the first object inside it) is what we parse —
        // otherwise only the first call survives and the rest silently drop.
        var array_probe = trimmed;
        if (std.mem.startsWith(u8, array_probe, "```")) {
            if (std.mem.indexOfScalar(u8, array_probe, '\n')) |nl| {
                array_probe = std.mem.trim(u8, array_probe[nl + 1 ..], " \t\n\r");
            }
        }
        if (array_probe.len > 0 and array_probe[0] == '[') {
            if (balancedJsonArray(array_probe)) |arr_body| {
                try appendJsonToolCallArray(allocator, arr_body, &calls);
            }
        }
        if (calls.items.len == 0) {
            if (std.mem.indexOf(u8, trimmed, "{")) |brace_pos| {
                const json_start = trimmed[brace_pos..];
                // Snap the first balanced object so trailing garbage can't poison
                // the parse — models without a trained tool format (Gemma 3) emit
                // the call inside a ```json fence, leaving "\n```" after the `}`.
                const json_body = balancedJsonObject(json_start) orelse json_start;
                if (tryParseJsonToolCall(allocator, json_body)) |tc| {
                    try calls.append(allocator, tc);
                }
            }
        }
        // Everything appended in this block was inferred from BARE JSON — no
        // tag syntax anywhere. Mark provenance so the request chokepoint
        // (server.parseToolCallsForRequest → filterInferredBySchema) can
        // validate the name against the declared tools: the first balanced
        // {"name": …} object in truncated prose/code is routinely a DATA
        // record, not a call (the George Washington class).
        for (calls.items) |*tc| tc.inferred = true;
    }

    // A nameless placeholder cannot be executed or repaired by a client.
    var named: usize = 0;
    for (calls.items) |tc| {
        if (std.mem.trim(u8, tc.name, " \t\r\n").len == 0) {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
            continue;
        }
        calls.items[named] = tc;
        named += 1;
    }
    calls.items.len = named;
    if (calls.items.len == 0) {
        calls.deinit(allocator);
        return null;
    }

    // Final safety net: EVERY emitted call carries valid-JSON arguments, whatever
    // converter built them. The direct-construction converters (Gemma custom
    // format, Hermes) can still emit invalid JSON on a pathological value — e.g. a
    // JSON-style string with a bad escape (`\q`) copied verbatim. std.json is the
    // arbiter: if the args don't parse, try the tolerant re-serialize (which
    // re-escapes lone backslashes / control bytes / inner quotes), and if THAT
    // still fails, fall back to `{}` — a client can retry a named call with empty
    // args, but invalid JSON it cannot parse at all. This makes the "args are
    // always valid JSON" invariant structural, not per-converter.
    for (calls.items) |*tc| {
        if (jsonIsValidObject(allocator, tc.arguments)) continue;
        if (looseRepairToolCallJson(allocator, tc.arguments)) |repaired| {
            if (jsonIsValidObject(allocator, repaired)) {
                allocator.free(tc.arguments);
                tc.arguments = repaired;
                continue;
            }
            allocator.free(repaired);
        }
        log.info("  [tool-parse] unrepairable args for '{s}' → empty object (never ship invalid JSON)\n", .{tc.name});
        const empty = try allocator.dupe(u8, "{}");
        allocator.free(tc.arguments);
        tc.arguments = empty;
    }

    return try calls.toOwnedSlice(allocator);
}

/// Strip a leading `<parameters>` / trailing `</parameters>` pair if both
/// are present. DSV4 occasionally wraps the args object in this XML
/// container — leaving it in place would defeat the JSON parser. Trims
/// surrounding whitespace.
fn stripParametersWrapper(content: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const open = "<parameters>";
    const close = "</parameters>";
    if (trimmed.len < open.len + close.len) return trimmed;
    if (!std.mem.startsWith(u8, trimmed, open)) return trimmed;
    if (!std.mem.endsWith(u8, trimmed, close)) return trimmed;
    const inner = trimmed[open.len .. trimmed.len - close.len];
    return std.mem.trim(u8, inner, " \t\n\r");
}

/// Return the substring of `content` covering the first brace-balanced JSON
/// object, or null if no balanced object is present. Handles quoted strings
/// (so `}` inside a string doesn't close the object) and escape sequences.
/// Used to tolerate the DSV4 `{"…"}}` extra-trailing-`}` artifact.
/// Try to parse a tool tag whose body lives entirely in XML-ish
/// attributes. Two terminator shapes are accepted:
///
///   Self-closing /> :
///     <tool name="shell" arguments="{...}"/>
///     <tool name="X" arguments='{...}' />
///     <tool name="X" arguments="{...}'/>             (broken quote — DSV4)
///
///   Empty-body explicit close ></tool*> :
///     <tool_call name="cwd" arguments="{...}"></tool_call>
///     <tool name="X" arguments="{...}"></tool_calls>  (mismatched close OK)
///
/// `slice` MUST start with `<tool` (the caller has already validated the
/// terminator char after that prefix). Returns name + arguments slices into
/// `slice` plus the number of bytes consumed (up to and including the final
/// `>` or `</tool*>` close), or null if this isn't a shape we can parse.
///
/// We never trust the XML quoting around `arguments=` — the model frequently
/// uses `"` to open and `'` to close, or doesn't escape inner `"` at all.
/// Instead we expect a `{` immediately (optionally after a single opening
/// quote) and snap a balanced JSON object via `balancedJsonObject`. That
/// strategy is what makes the DSV4 fragment parse cleanly even though the
/// surrounding XML is malformed.
fn parseSelfClosingToolTag(slice: []const u8) ?struct { name: []const u8, arguments: []const u8, consumed: usize } {
    if (slice.len < "<tool".len) return null;
    if (!std.mem.startsWith(u8, slice, "<tool")) return null;
    // Bound the scan window so a runaway tag can't pin us forever.
    const limit = @min(slice.len, 8192);

    // 1. Locate `name="X"` (or `'X'`) attribute. We don't enforce ordering
    //    against `arguments=` — DSV4 sometimes interleaves them.
    var name: ?[]const u8 = null;
    var name_end: usize = "<tool".len;
    {
        var i: usize = "<tool".len;
        while (i + "name=".len < limit) : (i += 1) {
            if (slice[i] != 'n') continue;
            if (!std.mem.eql(u8, slice[i .. i + "name=".len], "name=")) continue;
            const q_pos = i + "name=".len;
            if (q_pos >= limit) break;
            const quote = slice[q_pos];
            if (quote != '"' and quote != '\'') continue;
            const val_start = q_pos + 1;
            const val_end_rel = std.mem.indexOfScalar(u8, slice[val_start..limit], quote) orelse break;
            name = slice[val_start .. val_start + val_end_rel];
            name_end = val_start + val_end_rel + 1;
            break;
        }
    }

    // 2. Locate `arguments=` and snap the next balanced JSON object as
    //    the value. We deliberately don't parse the XML quote — see
    //    function docs.
    var args: ?[]const u8 = null;
    var args_end: usize = 0;
    {
        var i: usize = "<tool".len;
        while (i + "arguments=".len < limit) : (i += 1) {
            if (slice[i] != 'a') continue;
            if (!std.mem.eql(u8, slice[i .. i + "arguments=".len], "arguments=")) continue;
            var j: usize = i + "arguments=".len;
            // Skip an opening quote if present.
            if (j < limit and (slice[j] == '"' or slice[j] == '\'')) j += 1;
            // Skip whitespace between the quote and the JSON object.
            while (j < limit and (slice[j] == ' ' or slice[j] == '\t')) j += 1;
            if (j >= limit or slice[j] != '{') break;
            const sub = slice[j..limit];
            const json = balancedJsonObject(sub) orelse break;
            args = json;
            const json_off = @intFromPtr(json.ptr) - @intFromPtr(sub.ptr);
            args_end = j + json_off + json.len;
            break;
        }
    }

    if (name == null or args == null) return null;

    // 3. Scan forward from the end of whichever attribute landed last to
    //    the terminator. Tolerate a stray closing quote (model often closes
    //    args with `'` even though the opener was `"`) and any intervening
    //    whitespace. We accept either:
    //      a) `/>` — self-closing.
    //      b) `>` followed (possibly across whitespace) by `</tool*>` —
    //         empty body, explicit close marker.
    var k: usize = @max(name_end, args_end);
    while (k < limit and (slice[k] == ' ' or slice[k] == '\t' or slice[k] == '\n' or slice[k] == '\r' or slice[k] == '"' or slice[k] == '\'')) k += 1;
    if (k >= limit) return null;

    if (slice[k] == '/') {
        k += 1;
        if (k >= limit or slice[k] != '>') return null;
        return .{ .name = name.?, .arguments = args.?, .consumed = k + 1 };
    }
    if (slice[k] == '>') {
        var w: usize = k + 1;
        while (w < slice.len and (slice[w] == ' ' or slice[w] == '\t' or slice[w] == '\n' or slice[w] == '\r')) w += 1;
        const close_markers = [_][]const u8{
            "</tool_call>",   "</tool_calls>",
            "</tool_request>", "</tool_requests>",
            "</tool>",
        };
        for (close_markers) |m| {
            if (w + m.len <= slice.len and std.mem.eql(u8, slice[w .. w + m.len], m)) {
                return .{ .name = name.?, .arguments = args.?, .consumed = w + m.len };
            }
        }
    }
    return null;
}

/// The value of a depth-1 `"name"` key in an UNBALANCED JSON object (a tool
/// call truncated mid-body). String-aware, so a `"name"` inside a nested
/// value or a string never counts. Null when the text is not an object or
/// carries no complete top-level name.
fn truncatedJsonCallName(content: []const u8) ?[]const u8 {
    const t = std.mem.trimStart(u8, content, " \t\n\r");
    if (!std.mem.startsWith(u8, t, "{")) return null;
    var depth: usize = 0;
    var i: usize = 0;
    var last_str: ?[]const u8 = null; // most recent complete string at depth 1
    var after_name_colon = false;
    while (i < t.len) : (i += 1) {
        const c = t[i];
        if (c == '"') {
            const start = i + 1;
            var j = start;
            while (j < t.len and t[j] != '"') : (j += 1) {
                if (t[j] == '\\') j += 1;
            }
            if (j >= t.len) return null; // unterminated string = the cut
            const str = t[start..j];
            if (depth == 1) {
                if (after_name_colon) return if (str.len > 0) str else null;
                last_str = str;
            }
            i = j;
            continue;
        }
        switch (c) {
            '{', '[' => {
                depth += 1;
                after_name_colon = false;
            },
            '}', ']' => {
                if (depth == 0) return null;
                depth -= 1;
            },
            ':' => if (depth == 1) {
                if (last_str) |k| after_name_colon = std.mem.eql(u8, k, "name");
            },
            ',' => {
                after_name_colon = false;
                if (depth == 1) last_str = null;
            },
            ' ', '\t', '\n', '\r' => {},
            else => after_name_colon = false,
        }
    }
    return null;
}

pub fn balancedJsonObject(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return trimmed[start .. i + 1];
            if (depth < 0) return null; // mismatched — give up
        }
    }
    return null;
}

/// DSV4-Flash XML-element tool form: the body of a `<tool_calls>`-style
/// wrapper carries the tool name and each argument as plain child elements —
/// no JSON anywhere (captured live, 2026-06-10):
///     <tool_name>shell</tool_name>
///     <command>df -h / | grep -v "Filesystem"</command>
/// The name comes from a <tool_name>, <name>, or <tool> element; every other
/// simple `<key>text</key>` pair becomes a string argument. Strict on shape:
/// non-whitespace between elements, an unclosed element, a duplicate name
/// element, or a missing name returns null so prose-ish markup can't
/// half-execute.
/// Tag suffixes after `tool_` that mark tool RESULTS or metadata, never tool
/// names. `<tool_output>…</tool_output>` is DSV4 hallucinating the result of
/// a call it never made — mapping it onto a tool named "output" would
/// fabricate a call out of thin air. `name` guards the bare XML-element-form
/// child (`<tool_name>shell</tool_name>`) appearing outside its wrapper.
const xml_tag_reserved_names = [_][]const u8{
    "output", "outputs", "result", "results", "response", "responses", "error", "errors", "name",
};

/// XML-element-TAG tool form: `<tool_NAME><key>value</key>…</tool_NAME>`,
/// the tool name embedded in the tag itself (DSV4 training-bias family;
/// captured live 2026-06-10: `<tool_read>` / `<tool_edit>`). `slice` starts
/// at `<tool`. Returns the parsed call plus bytes consumed through the close
/// tag, or null when the shape doesn't hold (no attributes allowed, exact
/// close required, reserved names rejected, body must be all elements).
fn parseXmlElementTagToolCall(
    allocator: std.mem.Allocator,
    slice: []const u8,
) !?struct { call: ParsedToolCall, consumed: usize } {
    std.debug.assert(std.mem.startsWith(u8, slice, "<tool"));
    const name_start = "<tool_".len;
    if (slice.len <= name_start) return null;
    var name_end = name_start;
    while (name_end < slice.len and
        (std.ascii.isAlphanumeric(slice[name_end]) or slice[name_end] == '_')) : (name_end += 1)
    {}
    if (name_end == name_start) return null; // bare `<tool_>`
    if (name_end >= slice.len or slice[name_end] != '>') return null; // attributes → not this form
    const tool_name = slice[name_start..name_end];
    for (xml_tag_reserved_names) |reserved| {
        if (std.ascii.eqlIgnoreCase(tool_name, reserved)) return null;
    }
    var close_buf: [72]u8 = undefined;
    if (tool_name.len + "</tool_>".len > close_buf.len) return null;
    const close_tag = std.fmt.bufPrint(&close_buf, "</tool_{s}>", .{tool_name}) catch return null;
    const body_start = name_end + 1;
    const close_rel = std.mem.indexOf(u8, slice[body_start..], close_tag) orelse return null;
    const body = slice[body_start .. body_start + close_rel];
    const args = parseXmlElementArgsJson(allocator, body) orelse blk: {
        // JSON-args body variant of the same form, also captured live:
        // `<tool_write>\n{"path": …, "content": …}\n</tool_write>`. The whole
        // object IS the args. The trimmed body must START with `{` so prose
        // that merely contains braces never produces arguments.
        const trimmed = std.mem.trim(u8, body, " \t\n\r");
        if (trimmed.len == 0 or trimmed[0] != '{') return null;
        const json_body = balancedJsonObject(trimmed) orelse return null;
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_body, .{}) catch return null;
        defer parsed.deinit();
        if (parsed.value != .object) return null;
        break :blk allocator.dupe(u8, json_body) catch return null;
    };
    const name_owned = allocator.dupe(u8, tool_name) catch {
        allocator.free(args);
        return null;
    };
    return .{
        .call = .{ .name = name_owned, .arguments = args },
        .consumed = body_start + close_rel + close_tag.len,
    };
}

/// Parse a body consisting ENTIRELY of `<key>value</key>` child elements into
/// a JSON args string. Values are kept verbatim, so nested markup (DSV4's
/// `<edits><oldText>…</oldText>…</edits>`) survives as the arg's string
/// value. Returns null unless every non-whitespace byte belongs to an element
/// and at least one element is present — a plain-text body must never
/// produce arguments.
fn parseXmlElementArgsJson(allocator: std.mem.Allocator, body: []const u8) ?[]u8 {
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);
    var count: usize = 0;
    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        args_map.put(allocator, key, .{ .string = body[val_start .. val_start + rel] }) catch return null;
        count += 1;
        i = val_start + rel + close_tag.len;
    }
    if (count == 0) return null;
    return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch null;
}

/// Earliest position at/after `from` where any of `needles` occurs, or null.
fn isBareToolName(body: []const u8) bool {
    const name = std.mem.trim(u8, body, " \t\n\r");
    if (name.len == 0) return false;
    for (name) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '_' or c == '-' or c == '.')) return false;
    }
    return true;
}

fn earliestIndexOfAny(text: []const u8, from: usize, needles: []const []const u8) ?usize {
    var best: ?usize = null;
    for (needles) |n| {
        if (std.mem.indexOfPos(u8, text, from, n)) |q| {
            if (best == null or q < best.?) best = q;
        }
    }
    return best;
}

/// A tool-name character per the identifier alphabet every template family's
/// names draw from. Anything outside this set (a channel marker's `>`, a
/// payload's `}`, prose punctuation) terminates a name run.
fn toolNameChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '-';
}

/// The trailing identifier run ending at `end` (exclusive): a name that
/// echoed a channel marker (`<|marker|>bash`) resolves to the bare NAME.
fn trailingNameRun(text: []const u8, end: usize) []const u8 {
    var start = end;
    while (start > 0 and toolNameChar(text[start - 1])) start -= 1;
    return text[start..end];
}

// DeepSeek-V4 native DSML markers. `｜` is the FULLWIDTH vertical bar
// (U+FF5C, 3 bytes) — the dsml_token "｜DSML｜" appears in no other family.
// V4.1 puts a space before every tag name (`<｜DSML｜ invoke`); both spellings parse.
const DsmlTag = struct { open: []const u8, name: []const u8 };
const DSML_INVOKE_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "invoke" };
const DSML_INVOKE_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "invoke>" };
const DSML_PARAM_TAG: DsmlTag = .{ .open = "<｜DSML｜", .name = "parameter" };
const DSML_PARAM_CLOSE: DsmlTag = .{ .open = "</｜DSML｜", .name = "parameter>" };

const DsmlHit = struct { at: usize, end: usize };

fn dsmlFind(text: []const u8, from: usize, comptime tag: DsmlTag) ?DsmlHit {
    var p = from;
    while (std.mem.indexOfPos(u8, text, p, tag.open)) |at| {
        var q = at + tag.open.len;
        if (q < text.len and text[q] == ' ') q += 1;
        if (std.mem.startsWith(u8, text[q..], tag.name)) return .{ .at = at, .end = q + tag.name.len };
        p = at + 1;
    }
    return null;
}

/// Tolerant quoted-attribute extraction from a tag-attribute segment:
/// `name="X"` / `name='X'` / mangled unquoted `name=X` (to whitespace/`>`).
fn dsmlAttr(seg: []const u8, comptime key: []const u8) ?[]const u8 {
    const pat = key ++ "=";
    var start: usize = 0;
    while (std.mem.indexOfPos(u8, seg, start, pat)) |at| {
        if (at > 0 and (std.ascii.isAlphanumeric(seg[at - 1]) or seg[at - 1] == '_')) {
            start = at + pat.len;
            continue;
        }
        var v = at + pat.len;
        if (v >= seg.len) return null;
        const q = seg[v];
        if (q == '"' or q == '\'') {
            v += 1;
            const end = std.mem.indexOfScalarPos(u8, seg, v, q) orelse seg.len;
            return seg[v..end];
        }
        const end = std.mem.indexOfAnyPos(u8, seg, v, " \t\n>") orelse seg.len;
        return seg[v..end];
    }
    return null;
}

/// DeepSeek-V4 DSML tool calls:
///   `<｜DSML｜tool_calls>` wrapping
///   `<｜DSML｜invoke name="X">` +
///   `<｜DSML｜parameter name="k" string="true|false">v</｜DSML｜parameter>`.
/// string="true" values are raw text (JSON-escaped here); string="false"
/// values are already JSON (validated; invalid degrades to an honest string).
/// House conventions: the parser keys on the INVOKE marker (a dropped
/// wrapper or invoke-close still parses — delimiter-drop class), values run
/// to the CONFIRMED closing tag (rich values with `<`/newlines survive),
/// truncation salvages NAME + completed pairs (fragments dropped), keys are
/// escaped + deduped (last wins) so emitted arguments are ALWAYS valid JSON.
fn parseDsmlToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    const Param = struct { key: []const u8, value: []const u8, is_string: bool };
    var pos: usize = 0;
    while (dsmlFind(text, pos, DSML_INVOKE_TAG)) |inv| {
        var p = inv.end;
        const tag_end = std.mem.indexOfScalarPos(u8, text, p, '>') orelse break;
        const name = dsmlAttr(text[p..tag_end], "name") orelse {
            pos = tag_end + 1;
            continue;
        };
        p = tag_end + 1;

        var params = std.ArrayList(Param).empty;
        defer params.deinit(allocator);
        while (true) {
            const next_param = dsmlFind(text, p, DSML_PARAM_TAG);
            const next_close = dsmlFind(text, p, DSML_INVOKE_CLOSE);
            const next_invoke = dsmlFind(text, p, DSML_INVOKE_TAG);
            const np = if (next_param) |h| h.at else std.math.maxInt(usize);
            const nc = if (next_close) |h| h.at else std.math.maxInt(usize);
            const ni = if (next_invoke) |h| h.at else std.math.maxInt(usize);
            if (np == std.math.maxInt(usize) and nc == std.math.maxInt(usize) and ni == std.math.maxInt(usize)) {
                // truncated invoke: ship NAME + completed pairs
                p = text.len;
                break;
            }
            if (nc <= np and nc <= ni) {
                p = next_close.?.end;
                break;
            }
            if (ni < np) {
                // dropped invoke-close — the next call starts here; the outer
                // loop picks it up from `p`.
                p = ni;
                break;
            }
            // parameter
            var q = next_param.?.end;
            const ptag_end = std.mem.indexOfScalarPos(u8, text, q, '>') orelse {
                p = text.len;
                break;
            };
            const attrs = text[q..ptag_end];
            const pkey = dsmlAttr(attrs, "name");
            const pstring = dsmlAttr(attrs, "string");
            const vstart = ptag_end + 1;
            const vclose = dsmlFind(text, vstart, DSML_PARAM_CLOSE) orelse {
                // unterminated value = server-cut fragment; drop it, end scan
                p = text.len;
                break;
            };
            q = vclose.end;
            if (pkey) |k| {
                const value = text[vstart..vclose.at];
                // string attr absent (mangled opener): treat JSON-parseable
                // values as JSON, everything else as a string.
                const is_string = if (pstring) |sf|
                    !std.mem.eql(u8, sf, "false")
                else
                    !dsmlValueIsJson(allocator, value);
                // dedup: last wins
                var replaced = false;
                for (params.items) |*prm| {
                    if (std.mem.eql(u8, prm.key, k)) {
                        prm.value = value;
                        prm.is_string = is_string;
                        replaced = true;
                        break;
                    }
                }
                if (!replaced) try params.append(allocator, .{ .key = k, .value = value, .is_string = is_string });
            }
            p = q;
        }

        // Build the arguments object: escaped keys, escaped string values,
        // raw (validated) JSON values.
        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);
        try out.append(allocator, '{');
        for (params.items, 0..) |prm, i| {
            if (i > 0) try out.append(allocator, ',');
            try appendJsonString(allocator, &out, prm.key);
            try out.append(allocator, ':');
            if (!prm.is_string and dsmlValueIsJson(allocator, prm.value)) {
                try out.appendSlice(allocator, std.mem.trim(u8, prm.value, " \t\n\r"));
            } else {
                try appendJsonString(allocator, &out, prm.value);
            }
        }
        try out.append(allocator, '}');
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = try out.toOwnedSlice(allocator) });
        pos = p;
    }
}

fn dsmlValueIsJson(allocator: std.mem.Allocator, value: []const u8) bool {
    const trimmed = std.mem.trim(u8, value, " \t\n\r");
    if (trimmed.len == 0) return false;
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, trimmed, .{}) catch return false;
    parsed.deinit();
    return true;
}

// ── Claude-style `<invoke>` tool calls ──
//
// String parameters are RAW bytes (the reference explicitly does not strip
// spaces — the Hermes trim gap stays Hermes'); lists/objects/bools/null/
// numbers are spelled as JSON and keep their spelled type (the schema
// coercion chokepoint still gets the final word). The parser keys on the
// INVOKE tag so a dropped wrapper still parses (delimiter-drop class), and
// truncation salvages NAME + completed parameters — never fragment values.
const InvokeDialect = struct {
    invoke: []const u8,
    param: []const u8,
    param_close: []const u8,
    invoke_close: []const u8,
    /// Not the model's own dialect: kept only when it names a declared tool (`filterInferredBySchema`).
    inferred: bool,
};
/// Claude's spelling of the same grammar (`<function_calls><invoke name="X"><parameter name="k">`),
/// which models trained on Claude transcripts emit in place of their own (qwen4_exp).
pub const BARE_INVOKE: InvokeDialect = .{ .invoke = "<invoke", .param = "<parameter", .param_close = "</parameter>", .invoke_close = "</invoke>", .inferred = true };
pub const FUNCTION_CALLS_TAG = "<function_calls>";

fn parseInvokeToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall), comptime d: InvokeDialect) !void {
    // An inferred dialect's opener alone is prose ("wrap it in <invoke name=\"bash\">", even naming a declared
    // tool): a call also carries a parameter or its close, as `bareInvokeMarkupCut` demands.
    if (d.inferred and std.mem.indexOf(u8, text, d.param ++ " ") == null and std.mem.indexOf(u8, text, d.invoke_close) == null) return;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, d.invoke)) |inv| {
        const after_tag = inv + d.invoke.len;
        const seg = text[after_tag..];
        // `<invoked>` is a word, not the tag.
        if (seg.len > 0 and !std.ascii.isWhitespace(seg[0])) {
            pos = after_tag;
            continue;
        }
        // Attribute scan bounded so a runaway unclosed quote can't eat the
        // rest of the output.
        const raw_name = dsmlAttr(seg[0..@min(seg.len, 512)], "name") orelse {
            pos = after_tag;
            continue;
        };
        // No real tool name carries a channel marker (corpus invariant —
        // the marker-echo class).
        var name_slice = raw_name;
        if (std.mem.indexOf(u8, name_slice, "<|") != null) {
            name_slice = trailingNameRun(name_slice, name_slice.len);
        }
        if (name_slice.len == 0) {
            pos = after_tag;
            continue;
        }

        var args_json: ?[]const u8 = null;
        if (std.mem.indexOfScalar(u8, seg, '>')) |te| {
            const body_all = seg[te + 1 ..];
            // An inferred opener must own a parameter or close before the next opener,
            // else it is prose and would borrow the next call's body.
            if (d.inferred) {
                const own = body_all[0 .. std.mem.indexOf(u8, body_all, d.invoke) orelse body_all.len];
                if (std.mem.indexOf(u8, own, d.param ++ " ") == null and std.mem.indexOf(u8, own, d.invoke_close) == null) {
                    pos = after_tag;
                    continue;
                }
            }
            const body_end = std.mem.indexOf(u8, body_all, d.invoke_close) orelse body_all.len;
            args_json = try invokeParamsToJson(allocator, body_all[0..body_end], d);
            pos = if (body_end < body_all.len)
                after_tag + te + 1 + body_end + d.invoke_close.len
            else
                text.len; // truncated body — completed params already salvaged
        } else {
            pos = text.len; // truncated inside the opening tag: NAME + {}
        }

        const name = try allocator.dupe(u8, name_slice);
        errdefer allocator.free(name);
        const arguments = args_json orelse try allocator.dupe(u8, "{}");
        try calls.append(allocator, .{ .name = name, .arguments = arguments, .inferred = d.inferred });
    }
}

/// Assemble `<parameter name="k">v</parameter>` runs into a JSON
/// object string. A value that parses as COMPLETE JSON keeps its spelled type;
/// everything else is a raw string, exact bytes. Keys are escaped + deduped
/// (last wins) via ObjectMap+Stringify, so emitted arguments are ALWAYS valid
/// JSON. Returns null when no complete parameter was found.
fn invokeParamsToJson(allocator: std.mem.Allocator, body: []const u8, comptime d: InvokeDialect) !?[]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var map: std.json.ObjectMap = .empty;
    var pos: usize = 0;
    var found = false;
    while (std.mem.indexOfPos(u8, body, pos, d.param)) |p| {
        const after = p + d.param.len;
        const seg = body[after..];
        const te = std.mem.indexOfScalar(u8, seg, '>') orelse break; // truncated open tag
        const pname = dsmlAttr(seg[0 .. te + 1], "name") orelse {
            pos = after;
            continue;
        };
        const vstart = after + te + 1;
        // The value runs to the CONFIRMED closing tag — rich values with
        // newlines/quotes/`<` survive; no close by end of text = fragment,
        // dropped.
        const vend_rel = std.mem.indexOf(u8, body[vstart..], d.param_close) orelse break;
        const raw = body[vstart .. vstart + vend_rel];
        const v: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, aa, raw, .{}) catch
            std.json.Value{ .string = raw };
        try map.put(aa, pname, v);
        found = true;
        pos = vstart + vend_rel + d.param_close.len;
    }
    if (!found) return null;
    return try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = map }, .{});
}

/// GLM's tag-format tool calls: a BARE `<tool_call>` opener, the NAME, then
/// `<arg_key>K</arg_key><arg_value>V</arg_value>` pairs:
///   <tool_call>NAME
///   <arg_key>KEY</arg_key>
///   <arg_value>VALUE</arg_value> …
///   </tool_call> …
/// Values are kept as RAW STRINGS: the template `tojson`s non-string values on
/// the way in, and the schema-driven coercion at the server chokepoint types
/// them on the way back (types come from the SCHEMA, never the value's
/// spelling). Args build through ObjectMap + Stringify — keys escaped,
/// duplicates first-wins — per the tag-format-converter class rule. Truncation
/// (max_tokens mid-call, the big-file-write class) recovers with the CLOSED
/// key/value pairs only; partial values are never salvaged.
fn parseArgKeyToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<tool_call>")) |p| {
        const after_base = p + "<tool_call".len;
        // Bare `<tool_call>` has two sub-formats sharing this opener:
        //   • Hermes JSON `<tool_call>{…}` / function-tag `<tool_call><function=…>`
        //     → the generic scan reads those; fall through.
        //   • GLM: `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…`
        //     — bare opener, NAME, then arg_key/arg_value pairs. Route HERE only
        //     when an `<arg_key` precedes this call's close (the unambiguous GLM
        //     signal a JSON/function body never has), so the Hermes path is
        //     untouched.
        const body = after_base + 1;
        const this_close = std.mem.indexOfPos(u8, text, body, "</tool_call") orelse text.len;
        const ak_probe = std.mem.indexOfPos(u8, text, body, "<arg_key") orelse text.len;
        // A parameterless GLM call has no <arg_key>: its whole body is
        // the bare NAME, a shape neither the JSON nor `<function=` arms read.
        if (!(ak_probe < this_close or isBareToolName(text[body..this_close]))) {
            pos = after_base;
            continue;
        }
        const name_start = body;

        // NAME runs to the first structural marker after the opener. A mangled
        // call (weak model) closes the name with </arg_value> instead of
        // newline/<arg_key>, so accept any of them. No marker at all → the cut
        // happened inside the name; nothing to recover.
        const name_end = earliestIndexOfAny(text, name_start, &.{
            "<arg_key", "<arg_value", "</tool_call", "</arg_key", "</arg_value",
        }) orelse {
            pos = name_start;
            continue;
        };
        const name = std.mem.trim(u8, text[name_start..name_end], " \t\n\r");
        if (name.len == 0) {
            pos = name_start;
            continue;
        }

        var args_map: std.json.ObjectMap = .empty;
        defer args_map.deinit(allocator);

        // Parse <arg_key>/<arg_value> pairs, bounded by this call's </tool_call>.
        // SCAN to the next <arg_key> (rather than requiring it right here) so a
        // stray name-close tag between the name and the first key is skipped, and
        // match the KEY block's close TOLERANTLY (</arg_key> OR </arg_value>), as
        // a weak model closes it with </arg_value>. The well-formed path is
        // unchanged: </arg_key> is the earliest close.
        var i: usize = name_end;
        const call_end = std.mem.indexOfPos(u8, text, i, "</tool_call") orelse text.len;
        while (true) {
            const ak_at = std.mem.indexOfPos(u8, text, i, "<arg_key") orelse break;
            if (ak_at >= call_end) break;
            const ak_len = tagLenAt(text[ak_at..], "<arg_key") orelse {
                i = ak_at + "<arg_key".len;
                continue;
            };
            const key_start = ak_at + ak_len;
            const ak_close = earliestIndexOfAny(text, key_start, &.{ "</arg_key", "</arg_value" }) orelse break;
            const ak_close_len = tagLenAt(text[ak_close..], "</arg_key") orelse
                tagLenAt(text[ak_close..], "</arg_value") orelse break;
            // A doubled opener (`<arg_key><arg_key>K`) keys on the last one.
            var real_start = key_start;
            while (std.mem.indexOfPos(u8, text, real_start, "<arg_key")) |n| {
                if (n >= ak_close) break;
                real_start = n + (tagLenAt(text[n..], "<arg_key") orelse break);
            }
            const key = std.mem.trim(u8, text[real_start..ak_close], " \t\n\r");
            i = ak_close + ak_close_len;
            while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
            const av_len = tagLenAt(text[i..], "<arg_value") orelse break;
            const val_start = i + av_len;
            const av_close = std.mem.indexOfPos(u8, text, val_start, "</arg_value") orelse break;
            const av_close_len = tagLenAt(text[av_close..], "</arg_value") orelse break;
            const value = text[val_start..av_close];
            i = av_close + av_close_len;
            if (key.len > 0 and args_map.getEntry(key) == null) {
                try args_map.put(allocator, key, .{ .string = value });
            }
        }

        const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
        errdefer allocator.free(args_str);
        const name_owned = try allocator.dupe(u8, name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

        if (std.mem.indexOfPos(u8, text, i, "</tool_call")) |cl| {
            const cl_len = tagLenAt(text[cl..], "</tool_call") orelse 0;
            pos = if (cl_len > 0) cl + cl_len else cl + "</tool_call".len;
        } else {
            pos = i;
        }
    }
}

fn parseXmlElementToolCall(allocator: std.mem.Allocator, body: []const u8) ?ParsedToolCall {
    var name: ?[]const u8 = null;
    var args_map: std.json.ObjectMap = .empty;
    defer args_map.deinit(allocator);

    var i: usize = 0;
    while (i < body.len) {
        while (i < body.len and std.ascii.isWhitespace(body[i])) i += 1;
        if (i >= body.len) break;
        if (body[i] != '<') return null;
        const key_start = i + 1;
        var j = key_start;
        while (j < body.len and (std.ascii.isAlphanumeric(body[j]) or body[j] == '_')) j += 1;
        if (j == key_start or j >= body.len or body[j] != '>') return null;
        const key = body[key_start..j];
        const val_start = j + 1;
        var close_buf: [64]u8 = undefined;
        if (key.len + 3 > close_buf.len) return null;
        const close_tag = std.fmt.bufPrint(&close_buf, "</{s}>", .{key}) catch return null;
        const rel = std.mem.indexOf(u8, body[val_start..], close_tag) orelse return null;
        const value = body[val_start .. val_start + rel];
        i = val_start + rel + close_tag.len;

        if (std.mem.eql(u8, key, "tool_name") or std.mem.eql(u8, key, "name") or std.mem.eql(u8, key, "tool")) {
            if (name != null) return null;
            const trimmed_name = std.mem.trim(u8, value, " \t\n\r");
            if (trimmed_name.len == 0) return null;
            name = trimmed_name;
        } else {
            args_map.put(allocator, key, .{ .string = value }) catch return null;
        }
    }

    const n = name orelse return null;
    const args_str = std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{}) catch return null;
    const name_owned = allocator.dupe(u8, n) catch {
        allocator.free(args_str);
        return null;
    };
    return .{ .name = name_owned, .arguments = args_str };
}

/// Repair a JSON object whose trailing closer(s) were dropped — e.g. the
/// model wrote `{"name":"edit","arguments":{…,"edits":[{…}]}` and went
/// straight to its close tag, one `}` short. String/escape-aware bracket
/// stack; appends exactly the missing closers in nesting order. Returns null
/// when the text doesn't start with `{`, is already balanced (trailing
/// garbage is someone else's problem), ends mid-string, or nests deeper
/// than the stack (not worth guessing).
fn completeUnbalancedJsonObject(allocator: std.mem.Allocator, content: []const u8) ?[]u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    if (trimmed.len == 0 or trimmed[0] != '{') return null;
    var stack: [16]u8 = undefined;
    var depth: usize = 0;
    var in_string = false;
    var escape = false;
    for (trimmed) |c| {
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '{' => {
                if (depth == stack.len) return null;
                stack[depth] = '}';
                depth += 1;
            },
            '[' => {
                if (depth == stack.len) return null;
                stack[depth] = ']';
                depth += 1;
            },
            '}', ']' => {
                if (depth == 0 or stack[depth - 1] != c) return null;
                depth -= 1;
                // Balanced before the end → trailing garbage, not a
                // truncated tail. balancedJsonObject owns that case.
                if (depth == 0) return null;
            },
            else => {},
        }
    }
    if (in_string or depth == 0) return null;
    var out = allocator.alloc(u8, trimmed.len + depth) catch return null;
    @memcpy(out[0..trimmed.len], trimmed);
    var i: usize = 0;
    while (i < depth) : (i += 1) {
        out[trimmed.len + i] = stack[depth - 1 - i];
    }
    return out;
}

/// Snap a balanced JSON array starting at the first `[`. Mirrors
/// balancedJsonObject: string/escape aware, tracks BOTH bracket kinds so
/// objects nested in the array can't fool the depth count.
fn balancedJsonArray(content: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, content, " \t\n\r");
    const start = std.mem.indexOfScalar(u8, trimmed, '[') orelse return null;
    var depth: i32 = 0;
    var in_string: bool = false;
    var escape: bool = false;
    var i: usize = start;
    while (i < trimmed.len) : (i += 1) {
        const c = trimmed[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escape = true;
                continue;
            }
            if (c == '"') in_string = false;
            continue;
        }
        switch (c) {
            '"' => in_string = true,
            '[', '{' => depth += 1,
            ']', '}' => {
                depth -= 1;
                if (depth == 0) return trimmed[start .. i + 1];
                if (depth < 0) return null; // mismatched — give up
            },
            else => {},
        }
    }
    return null;
}

/// Parse a JSON array of tool-call objects (`[{"name":…,"arguments":…}, …]`)
/// and append EVERY call. All-or-nothing: if any element fails to parse as a
/// tool call, nothing is appended — a prose-ish array must pass through as
/// text rather than half-execute.
fn appendJsonToolCallArray(
    allocator: std.mem.Allocator,
    arr_text: []const u8,
    calls: *std.ArrayList(ParsedToolCall),
) !void {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, arr_text, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .array) return;
    const items = parsed.value.array.items;
    if (items.len == 0) return;

    var pending = std.ArrayList(ParsedToolCall).empty;
    var ok = true;
    defer {
        for (pending.items) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        pending.deinit(allocator);
    }
    for (items) |item| {
        if (item != .object) {
            ok = false;
            break;
        }
        // Reuse tryParseJsonToolCall (leaf-name walk, flat-shape synthesis)
        // by round-tripping the element through its text form.
        const item_text = std.json.Stringify.valueAlloc(allocator, item, .{}) catch {
            ok = false;
            break;
        };
        defer allocator.free(item_text);
        const tc = tryParseJsonToolCall(allocator, item_text) orelse {
            ok = false;
            break;
        };
        try pending.append(allocator, tc);
    }
    if (!ok) return;
    try calls.appendSlice(allocator, pending.items);
    pending.clearRetainingCapacity();
}

/// Lenient recovery for tool-call argument JSON that small models mangle when
/// emitting large string values (file contents). Two failure modes, both of
/// which make strict std.json reject the WHOLE blob so the call would be dropped
/// and leak as visible text:
///   • raw control bytes (literal newlines/tabs) inside a string instead of
///     `\n`/`\t` — the dominant big-file failure;
///   • unescaped inner double-quotes (`<meta charset="UTF-8">`) and invalid
///     backslash escapes (Windows paths, regex) inside a string.
/// Re-serializes a CLEAN object by walking the input with a position-aware
/// tolerant parser (key vs value context): control bytes are re-escaped, lone/
/// invalid backslashes are escaped, and a `"` closes a string only at a
/// structural delimiter (`:` after a key; `,`/`}`/`]`/end after a value) — any
/// other `"` is an inner content quote and gets escaped. The CALLER strict-parses
/// the result, so a mis-recovery that yields invalid JSON is discarded; the only
/// residual risk is a value string closed early on pathological content (a
/// literal `"}` / `",` byte-sequence inside the file), which still beats dropping
/// the call. Does NOT tolerate truncation (no auto-close of open containers) —
/// that stays with completeUnbalancedJsonObject. Returns an allocator-owned
/// normalized JSON string, or null when no well-formed leading object recovers.
fn looseRepairToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    const obj_start = std.mem.indexOfScalar(u8, trimmed, '{') orelse return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitObject(allocator, &out, trimmed, obj_start, 0) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

/// Tolerant re-serialize of a top-level JSON CONTAINER (`[…]` or `{…}`) — the
/// array-aware sibling of looseRepairToolCallJson. Used by the schema coercion to
/// salvage a container-typed argument the model mangled (e.g. a missing comma
/// inside an `edits` array). Same discipline: the caller strict-parses the
/// result, so a mis-repair yielding invalid JSON is discarded. Returns null when
/// the text doesn't start with `[`/`{` or can't be re-serialized.
pub fn looseRepairContainer(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\r\n");
    if (trimmed.len == 0 or (trimmed[0] != '[' and trimmed[0] != '{')) return null;
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    _ = looseEmitValue(allocator, &out, trimmed, 0, 0, .object_value) catch {
        out.deinit(allocator);
        return null;
    };
    return out.toOwnedSlice(allocator) catch null;
}

const LooseStringCtx = enum { key, object_value, array_value };
const LooseError = error{ Malformed, OutOfMemory };
const loose_max_depth = 32;

fn looseSkipWs(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return i;
}

fn looseSkipWsCommas(body: []const u8, start: usize) usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return i;
}

fn looseIsHex4(s: []const u8) bool {
    if (s.len != 4) return false;
    for (s) |c| {
        const ok = (c >= '0' and c <= '9') or (c >= 'a' and c <= 'f') or (c >= 'A' and c <= 'F');
        if (!ok) return false;
    }
    return true;
}

fn looseEmitObject(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '{') return error.Malformed;
    i += 1;
    try out.append(allocator, '{');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed; // unterminated (not truncation-tolerant)
        if (body[i] == '}') {
            try out.append(allocator, '}');
            return i + 1;
        }
        if (body[i] != '"') return error.Malformed; // key must be a string
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitString(allocator, out, body, i, .key);
        i = looseSkipWs(body, i);
        if (i >= body.len or body[i] != ':') return error.Malformed;
        i += 1;
        try out.append(allocator, ':');
        i = try looseEmitValue(allocator, out, body, i, depth, .object_value);
    }
}

fn looseEmitArray(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
) LooseError!usize {
    if (depth > loose_max_depth) return error.Malformed;
    var i = looseSkipWs(body, start);
    if (i >= body.len or body[i] != '[') return error.Malformed;
    i += 1;
    try out.append(allocator, '[');
    var first = true;
    while (true) {
        i = looseSkipWsCommas(body, i);
        if (i >= body.len) return error.Malformed;
        if (body[i] == ']') {
            try out.append(allocator, ']');
            return i + 1;
        }
        if (!first) try out.append(allocator, ',');
        first = false;
        i = try looseEmitValue(allocator, out, body, i, depth, .array_value);
    }
}

fn looseEmitValue(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    str_ctx: LooseStringCtx,
) LooseError!usize {
    const i = looseSkipWs(body, start);
    if (i >= body.len) return error.Malformed;
    return switch (body[i]) {
        '"' => try looseEmitString(allocator, out, body, i, str_ctx),
        '{' => try looseEmitObject(allocator, out, body, i, depth + 1),
        '[' => try looseEmitArray(allocator, out, body, i, depth + 1),
        else => try looseEmitScalar(allocator, out, body, i),
    };
}

fn looseEmitScalar(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
) LooseError!usize {
    var i = start;
    while (i < body.len) : (i += 1) {
        switch (body[i]) {
            ',', '}', ']', ' ', '\t', '\n', '\r' => break,
            else => try out.append(allocator, body[i]),
        }
    }
    if (i == start) return error.Malformed;
    return i; // strict re-parse validates the token was a real number/bool/null
}

fn looseEmitString(
    allocator: std.mem.Allocator,
    out: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    ctx: LooseStringCtx,
) LooseError!usize {
    // body[start] == '"'
    try out.append(allocator, '"');
    var i = start + 1;
    while (i < body.len) {
        const c = body[i];
        if (c == '\\') {
            if (i + 1 >= body.len) {
                try out.appendSlice(allocator, "\\\\"); // trailing backslash → literal
                i += 1;
                continue;
            }
            const n = body[i + 1];
            switch (n) {
                '"', '\\', '/', 'b', 'f', 'n', 'r', 't' => {
                    try out.append(allocator, '\\');
                    try out.append(allocator, n);
                    i += 2;
                },
                'u' => {
                    if (i + 6 <= body.len and looseIsHex4(body[i + 2 .. i + 6])) {
                        try out.appendSlice(allocator, body[i .. i + 6]);
                        i += 6;
                    } else {
                        try out.appendSlice(allocator, "\\\\"); // invalid \u → literal backslash
                        i += 1;
                    }
                },
                else => {
                    try out.appendSlice(allocator, "\\\\"); // invalid escape → literal backslash
                    i += 1;
                },
            }
            continue;
        }
        if (c == '"') {
            const j = looseSkipWs(body, i + 1);
            const is_close = switch (ctx) {
                .key => j < body.len and body[j] == ':',
                .object_value => j >= body.len or body[j] == ',' or body[j] == '}',
                .array_value => j >= body.len or body[j] == ',' or body[j] == ']',
            };
            if (is_close) {
                try out.append(allocator, '"');
                return i + 1;
            }
            try out.appendSlice(allocator, "\\\""); // inner content quote → escape
            i += 1;
            continue;
        }
        if (c < 0x20) {
            switch (c) {
                '\n' => try out.appendSlice(allocator, "\\n"),
                '\t' => try out.appendSlice(allocator, "\\t"),
                '\r' => try out.appendSlice(allocator, "\\r"),
                0x08 => try out.appendSlice(allocator, "\\b"),
                0x0c => try out.appendSlice(allocator, "\\f"),
                else => {
                    var buf: [6]u8 = undefined;
                    const esc = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                    try out.appendSlice(allocator, esc);
                },
            }
            i += 1;
            continue;
        }
        try out.append(allocator, c);
        i += 1;
    }
    // Ran off the end with no structural close — unterminated. The enclosing
    // container won't find its close either → looseEmitObject returns Malformed
    // and the whole recovery is discarded (truncation handled elsewhere). Emit
    // a closing quote so the discarded buffer stays well-formed.
    try out.append(allocator, '"');
    return i;
}

fn tryParseJsonToolCall(allocator: std.mem.Allocator, text: []const u8) ?ParsedToolCall {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch blk: {
        // Strict parse failed. Try a chain of repairs for known shapes:
        //   1. {"name":"shell", {"command":"ls"}}           — missing `"arguments":` key entirely (Qwen MoE)
        //   2. {"name":"shell", arguments":{"command":..}}   — missing OPENING quote on `arguments` (Qwen MoE)
        //   3. {"name":"edit", "arguments": {…}]}            — truncated tail, final closer(s) dropped
        //      before the close tag (DSV4-Flash, captured live)
        // Repairs are cheap and run only on the parse-failure path.
        if (repairBrokenToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                break :blk reparsed;
            } else |_| {}
        }
        //   4. Mangled big-file content: raw control bytes (literal newlines/
        //      tabs) and unescaped inner quotes inside a string value — what
        //      small models emit when writing a large file in one shot. The
        //      re-escaped copy is re-validated by the strict parse below, so a
        //      mis-recovery that yields invalid JSON is silently discarded.
        if (looseRepairToolCallJson(allocator, text)) |repaired| {
            defer allocator.free(repaired);
            if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
                log.info("  [tool-parse] loose-repair recovered mangled tool-call JSON (raw control bytes / unescaped quotes)\n", .{});
                break :blk reparsed;
            } else |_| {}
        }
        const completed = completeUnbalancedJsonObject(allocator, text) orelse return null;
        defer allocator.free(completed);
        const reparsed = std.json.parseFromSlice(std.json.Value, allocator, completed, .{}) catch return null;
        break :blk reparsed;
    };
    defer parsed.deinit();

    if (parsed.value != .object) return null;
    var obj = parsed.value.object;

    // Qwen 3.5/3.6 MoE sometimes emits nested-name garbage like
    //   {"name":{"name":{"name":"write","arguments":{...}}}}
    // Walk down through up to a few levels of nested "name" objects to find the
    // leaf object that has a string name + arguments/parameters. Observed in the
    // wild with pi + Qwen3.6-35B in non-thinking mode.
    {
        var depth: u8 = 0;
        while (depth < 4) : (depth += 1) {
            const nv = obj.get("name") orelse return null;
            switch (nv) {
                .string => break,
                .object => |inner| obj = inner,
                else => return null,
            }
        }
    }

    const name_val = obj.get("name") orelse return null;
    if (name_val != .string) return null;

    const args_str: []const u8 = blk: {
        if (obj.get("arguments") orelse obj.get("parameters")) |args_val| {
            break :blk switch (args_val) {
                .object => std.json.Stringify.valueAlloc(allocator, args_val, .{}) catch return null,
                .string => |s| allocator.dupe(u8, s) catch return null,
                else => return null,
            };
        }
        // Flat shape (e.g. Qwen MoE): {"name":"shell","command":"ls"} —
        // parameters live at top level. Synthesize an arguments object from
        // every non-metadata key.
        var flat_map: std.json.ObjectMap = .empty;
        defer flat_map.deinit(allocator);
        var it = obj.iterator();
        while (it.next()) |entry| {
            const k = entry.key_ptr.*;
            if (std.mem.eql(u8, k, "name") or
                std.mem.eql(u8, k, "id") or
                std.mem.eql(u8, k, "type"))
            {
                continue;
            }
            flat_map.put(allocator, k, entry.value_ptr.*) catch return null;
        }
        if (flat_map.count() == 0) return null;
        const flat_value = std.json.Value{ .object = flat_map };
        break :blk std.json.Stringify.valueAlloc(allocator, flat_value, .{}) catch return null;
    };

    return .{
        .name = allocator.dupe(u8, name_val.string) catch {
            allocator.free(args_str);
            return null;
        },
        .arguments = args_str,
    };
}

/// Run known Qwen-MoE-broken-JSON repairs in sequence; return the first
/// successfully-repaired string (allocator-owned), or null if none match.
fn repairBrokenToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    if (repairFlatBraceToolCallJson(allocator, text)) |s| return s;
    if (repairUnquotedArgsKey(allocator, text)) |s| return s;
    return null;
}

/// Repair `{"name":"x", arguments":{...}}` (missing OPENING quote on
/// `arguments`/`parameters`) by injecting the quote.
fn repairUnquotedArgsKey(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    // Look for the unquoted-key pattern. Both `arguments` and `parameters`
    // have been observed; both are short fixed strings, so direct search is fine.
    const candidates = [_][]const u8{ ", arguments\":", ",arguments\":", ", parameters\":", ",parameters\":" };
    for (candidates) |needle| {
        if (std.mem.indexOf(u8, trimmed, needle)) |at| {
            const insert_at = at + 1; // right after the comma
            // Skip leading whitespace after the comma so the injected `"` lands on the identifier.
            var p = insert_at;
            while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
            return std.fmt.allocPrint(allocator, "{s}\"{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
        }
    }
    return null;
}

/// Repair the Qwen-MoE-broken shape `{"name":"x", {"k":"v"}}` by injecting
/// the missing `"arguments":` key between the comma and the inner object.
/// Returns an allocator-owned repaired JSON string, or null if the pattern
/// doesn't match.
fn repairFlatBraceToolCallJson(allocator: std.mem.Allocator, text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, text, " \t\n\r");
    if (trimmed.len < 4 or trimmed[0] != '{') return null;

    // Locate `"name"` key.
    const name_at = std.mem.indexOf(u8, trimmed, "\"name\"") orelse return null;
    var p = name_at + 6;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ':') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '"') return null;

    // Skip the string value.
    p += 1;
    while (p < trimmed.len and trimmed[p] != '"') {
        if (trimmed[p] == '\\' and p + 1 < trimmed.len) p += 2 else p += 1;
    }
    if (p >= trimmed.len) return null;
    p += 1;

    // Expect `, {` (any whitespace).
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != ',') return null;
    p += 1;
    while (p < trimmed.len and (trimmed[p] == ' ' or trimmed[p] == '\t' or trimmed[p] == '\n')) p += 1;
    if (p >= trimmed.len or trimmed[p] != '{') return null;

    return std.fmt.allocPrint(allocator, "{s}\"arguments\":{s}", .{ trimmed[0..p], trimmed[p..] }) catch null;
}

/// Strip one layer of outer braces from a `{{…}}`-wrapped JSON object. Models
/// sometimes emit args with a doubled brace layer (a literal-brace artifact from
/// Jinja `{{ }}` templates). Only touches strings that start with `{{` and end
/// with `}}`, so well-formed single-brace JSON passes through unchanged.
fn unwrapDoubleBraces(s: []const u8) []const u8 {
    if (s.len >= 4 and std.mem.startsWith(u8, s, "{{") and std.mem.endsWith(u8, s, "}}")) {
        return s[1 .. s.len - 1];
    }
    return s;
}

/// Strip artifacts models sometimes append to a tool name — surrounding
/// whitespace and a trailing ':' (Gemma 4 12B leaks one via its `call:NAME:`
/// format). A valid OpenAI tool name matches `^[A-Za-z0-9_-]{1,64}$`, so a
/// trailing colon is never legitimate and is always safe to drop.
fn sanitizeToolName(raw: []const u8) []const u8 {
    var name = std.mem.trim(u8, raw, " \t\n\r");
    while (name.len > 0 and name[name.len - 1] == ':') {
        name = std.mem.trim(u8, name[0 .. name.len - 1], " \t\n\r");
    }
    return name;
}

/// Parse Gemma 4 tool call format: "call:function_name{json_args}"
/// `input_truncated`: the call had NO `<tool_call|>` close tag — the
/// generation was cut (EOS mid-call, max_tokens, or the scheduler's
/// degenerate-tail-loop guard). Under truncation, a value whose scan runs to
/// end-of-body without its terminator is a FRAGMENT and is dropped (the
/// Hermes-truncation rule: a half-written file is worse than a re-issued
/// write); with the close tag present, behavior is byte-identical to before.
fn parseGemma4ToolCall(allocator: std.mem.Allocator, content: []const u8, input_truncated: bool) ?ParsedToolCall {
    const prefix = "call:";
    if (!std.mem.startsWith(u8, content, prefix)) return null;
    const after_prefix = content[prefix.len..];

    // Find the opening brace
    const brace_pos = std.mem.indexOf(u8, after_prefix, "{") orelse return null;
    const name = sanitizeToolName(after_prefix[0..brace_pos]);
    if (name.len == 0) return null;

    var args_str = after_prefix[brace_pos..];

    // Gemma 4 uses {{ }} (double braces) for literal braces in Jinja templates.
    // The model often generates {{"key":"value"}} — unwrap the outer braces.
    args_str = unwrapDoubleBraces(args_str);

    // Try JSON first (model sometimes outputs valid JSON arguments)
    if (std.json.parseFromSlice(std.json.Value, allocator, args_str, .{})) |parsed| {
        defer parsed.deinit();
        if (parsed.value == .object) {
            return .{
                .name = allocator.dupe(u8, name) catch return null,
                .arguments = allocator.dupe(u8, args_str) catch return null,
            };
        }
    } else |_| {}

    // Strict JSON failed — but the model may have emitted standard JSON with
    // mangled escaping (raw newlines / unescaped quotes in big content) rather
    // than the custom <|"|> format. Recover that BEFORE the custom-format
    // converter, which would garble standard JSON. looseRepair returns null for
    // the bare-key `{key:<|"|>v<|"|>}` form, so the custom path still runs.
    if (looseRepairToolCallJson(allocator, args_str)) |repaired| {
        var keep = false;
        defer if (!keep) allocator.free(repaired);
        if (std.json.parseFromSlice(std.json.Value, allocator, repaired, .{})) |reparsed| {
            defer reparsed.deinit();
            if (reparsed.value == .object) {
                const name_owned = allocator.dupe(u8, name) catch return null;
                keep = true;
                log.info("  [tool-parse] loose-repair recovered mangled Gemma 4 tool-call JSON\n", .{});
                return .{ .name = name_owned, .arguments = repaired };
            }
        } else |_| {}
    }

    // Convert Gemma 4 custom format to JSON:
    // {key:<|"|>value<|"|>,key2:<|"|>value2<|"|>} → {"key":"value","key2":"value2"}
    const json = convertGemma4ArgsToJson(allocator, args_str, input_truncated) orelse {
        log.info("  [tool-parse] convertGemma4ArgsToJson FAILED for: {s}\n", .{args_str[0..@min(args_str.len, 200)]});
        return null;
    };
    return .{
        .name = allocator.dupe(u8, name) catch return null,
        .arguments = json,
    };
}

/// Convert Gemma 4's custom key-value format to JSON.
/// Input:  {key:<|"|>value<|"|>,nested:{k:<|"|>v<|"|>},arr:[<|"|>a<|"|>]}
/// Output: {"key":"value","nested":{"k":"v"},"arr":["a"]}
///
/// Gemma 4 emits its own object/array/string syntax (bare keys, `<|"|>…<|"|>`
/// string delimiters) that can nest arbitrarily, so every value is converted
/// recursively — see `convertGemma4Value`. Nested structures that already use
/// valid JSON (regular `"…"` strings, numbers, bools, null) pass through
/// unchanged. The args are always an object, with or without a literal `{…}`.
fn convertGemma4ArgsToJson(allocator: std.mem.Allocator, input: []const u8, input_truncated: bool) ?[]const u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    // `cut` = the truncation landed INSIDE a value somewhere below; the pair
    // holding it has already been rolled back (fragment values never ship).
    var cut = false;
    _ = convertGemma4Object(allocator, &result, input, 0, 0, input_truncated, &cut) orelse return null;
    return result.toOwnedSlice(allocator) catch return null;
}

const gemma4_str_delim = "<|\"|>";
/// Recursion-depth guard against adversarial deeply-nested model output.
const gemma4_max_depth = 64;

fn gemma4SkipWs(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r' => {},
            else => break,
        }
    }
    return pos;
}

fn gemma4SkipWsCommas(body: []const u8, start: usize) usize {
    var pos = start;
    while (pos < body.len) : (pos += 1) {
        switch (body[pos]) {
            ' ', '\t', '\n', '\r', ',' => {},
            else => break,
        }
    }
    return pos;
}

/// Parse a Gemma 4 object beginning at `start` (an optional leading `{` is
/// consumed; absent braces are tolerated so a brace-less body still parses).
/// Appends the JSON object to `result`; returns the index just past it.
fn convertGemma4Object(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = gemma4SkipWs(body, start);
    if (pos < body.len and body[pos] == '{') pos += 1;
    result.append(allocator, '{') catch return null;

    // Dedup keys within this object (a repeated key would make std.json reject
    // the args with error.DuplicateField — same class as parseHermesToolCall's
    // dup-param bug). Keys are slices into the stable input `body`.
    var seen = std.ArrayList([]const u8).empty;
    defer seen.deinit(allocator);

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == '}') {
            pos += 1;
            closed = true;
            break;
        }

        // Key: everything up to the first ':'.
        const colon = std.mem.indexOfScalar(u8, body[pos..], ':') orelse break;
        const key_raw = std.mem.trim(u8, body[pos .. pos + colon], " \t\n\r");
        // Strip surrounding quotes if present (model sometimes quotes keys).
        const key = if (key_raw.len >= 2 and key_raw[0] == '"' and key_raw[key_raw.len - 1] == '"')
            key_raw[1 .. key_raw.len - 1]
        else
            key_raw;
        pos = pos + colon + 1;

        // Tentatively emit comma+key+value; roll back if this key is a dup. The
        // value is still consumed (pos advances) so the dup's payload is skipped.
        const mark = result.items.len;
        if (!first) result.append(allocator, ',') catch return null;
        // Escape the KEY — a raw `"`/control byte in the key would emit invalid
        // JSON (the Gemma converter's output is not strict-re-validated).
        appendJsonString(allocator, result, key) catch return null;
        result.append(allocator, ':') catch return null;

        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;

        // The truncation landed INSIDE this value — it is a fragment (partial
        // file content, partial url) and must never ship as a real argument
        // (the Hermes-truncation rule, extended to the Gemma arm 2026-07-14:
        // php.html loop-stop capture). Roll the whole pair back; the scan
        // consumed to end-of-body, so this object is done.
        if (cut.*) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            break;
        }

        var is_dup = false;
        for (seen.items) |s| {
            if (std.mem.eql(u8, s, key)) {
                is_dup = true;
                break;
            }
        }
        if (is_dup) {
            result.shrinkRetainingCapacity(mark); // undo comma+key+value; keep `first`
            continue;
        }
        seen.append(allocator, key) catch return null;
        first = false;
    }

    // A NESTED object the cut landed inside (it never saw its `}`) is itself a
    // fragment — even when every pair inside completed, more were coming; the
    // parent rolls the whole container back. The ROOT object (depth 0) is
    // exempt: pairs completed before the cut are intact values and survive —
    // only the value holding the cut drops.
    if (input_truncated and depth > 0 and !closed) cut.* = true;

    result.append(allocator, '}') catch return null;
    return pos;
}

/// Parse a Gemma 4 array beginning at `start` (`body[start] == '['`). Appends
/// the JSON array to `result`; returns the index just past it.
fn convertGemma4Array(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    if (depth >= gemma4_max_depth) return null;
    var pos = start + 1; // consume '['
    result.append(allocator, '[') catch return null;

    var closed = false;
    var first = true;
    while (pos < body.len) {
        pos = gemma4SkipWsCommas(body, pos);
        if (pos >= body.len) break;
        if (body[pos] == ']') {
            pos += 1;
            closed = true;
            break;
        }
        if (!first) result.append(allocator, ',') catch return null;
        first = false;
        pos = convertGemma4Value(allocator, result, body, pos, depth, input_truncated, cut) orelse return null;
        // Fragment below: stop consuming; the parent object rolls the whole
        // container pair back, so no local cleanup is needed.
        if (cut.*) break;
    }

    // Same fragment rule as nested objects: an unclosed array under truncation
    // is a partial LIST — more elements were coming (a partial edits[] applies
    // fragmentary work, the same hazard as partial file content).
    if (input_truncated and !closed) cut.* = true;

    result.append(allocator, ']') catch return null;
    return pos;
}

/// Convert one Gemma 4 value (custom string / JSON string / object / array /
/// bare literal) beginning at `start`. String branches are checked first so
/// braces inside a string stay string content. Returns the index past it.
fn convertGemma4Value(
    allocator: std.mem.Allocator,
    result: *std.ArrayList(u8),
    body: []const u8,
    start: usize,
    depth: usize,
    input_truncated: bool,
    cut: *bool,
) ?usize {
    var pos = gemma4SkipWs(body, start);
    if (pos >= body.len) {
        // `key:` then end-of-body. Under truncation the value is a fragment
        // that never arrived — drop the pair (pre-fix this shipped
        // {"path":""} and a client would act on the bogus empty value).
        if (input_truncated) {
            cut.* = true;
            return pos;
        }
        result.appendSlice(allocator, "\"\"") catch return null;
        return pos;
    }

    // Gemma custom string: <|"|>…<|"|> (closing delimiter may be missing if
    // the model output was truncated).
    if (pos + gemma4_str_delim.len <= body.len and
        std.mem.eql(u8, body[pos .. pos + gemma4_str_delim.len], gemma4_str_delim))
    {
        pos += gemma4_str_delim.len;
        const end_idx = std.mem.indexOf(u8, body[pos..], gemma4_str_delim);
        const value = if (end_idx) |e| body[pos .. pos + e] else blk: {
            // Closing <|"|> missing.
            if (input_truncated) {
                // The call itself was cut (no <tool_call|>): the truncation
                // landed INSIDE this string — a fragment (partial file
                // content, partial url) that must never ship as a real
                // argument (live 2026-07-14 php.html: the loop-stop guard cut
                // a write call mid-`content` and the 1.1 KB fragment shipped).
                cut.* = true;
                return body.len;
            }
            // Complete call (close tag present), delimiter just dropped: a
            // plain to-end-of-body scan swallows the args object's own `}`
            // and any stray fence garbage the model tacked on — a real write
            // call reached disk as "mlx_pi1.html`}". Trim one trailing `}`
            // (the enclosing object's closer) plus backtick/whitespace junk.
            var v = std.mem.trimEnd(u8, body[pos..], " \t\n\r");
            if (std.mem.endsWith(u8, v, "}")) v = v[0 .. v.len - 1];
            v = std.mem.trimEnd(u8, v, "` \t\n\r");
            break :blk v;
        };
        pos = if (end_idx) |e|
            pos + e + gemma4_str_delim.len
        else
            body.len;
        appendJsonString(allocator, result, value) catch return null;
        return pos;
    }

    // Regular JSON string: already valid, copy verbatim (respecting \" escapes).
    if (body[pos] == '"') {
        var end = pos + 1;
        var closed_quote = false;
        while (end < body.len) : (end += 1) {
            if (body[end] == '\\' and end + 1 < body.len) {
                end += 1;
                continue;
            }
            if (body[end] == '"') {
                closed_quote = true;
                end += 1;
                break;
            }
        }
        // Unclosed JSON string at end-of-body under truncation: same fragment
        // rule — and copying it verbatim would emit an UNCLOSED string, i.e.
        // invalid JSON that only the {}-fallback safety net could catch.
        if (!closed_quote and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, body[pos..end]) catch return null;
        return end;
    }

    if (body[pos] == '{') return convertGemma4Object(allocator, result, body, pos, depth + 1, input_truncated, cut);
    if (body[pos] == '[') return convertGemma4Array(allocator, result, body, pos, depth + 1, input_truncated, cut);

    // Bare value — a JSON literal (number/bool/null) terminates at the
    // enclosing separator and is emitted verbatim.
    const sep_rel = std.mem.indexOfAny(u8, body[pos..], ",}]");
    const first_sep = sep_rel orelse (body.len - pos);
    const head = std.mem.trim(u8, body[pos .. pos + first_sep], " \t\n\r");
    if (isJsonLiteral(head)) {
        // A literal that ran to end-of-body under truncation has no
        // terminator either — uniform fragment rule (a cut `5` may have been
        // `50`; the client re-asks and gets the real value).
        if (sep_rel == null and input_truncated) {
            cut.* = true;
            return body.len;
        }
        result.appendSlice(allocator, head) catch return null;
        return pos + first_sep;
    }

    // Unquoted STRING. The bare scan above stops at the first `,`/`}`/`]`, which
    // truncates rich content (HTML/markdown) that legitimately contains those
    // bytes. Observed live on gemma-4-e4b-it writing a full page: it dropped the
    // OPENING <|"|> on `content` but kept the CLOSING one
    // (`content:<!DOCTYPE…>…</html><|"|>,path:…`), so content got cut at the
    // viewport meta's comma and the rest became bogus keys → invalid args. Only
    // rich content (multi-line or markup) gets the wider scan; a plain short
    // bare token keeps the first-separator behavior so it can't swallow a
    // sibling field (e.g. `command:ls -la`).
    const rich = std.mem.indexOfScalar(u8, head, '\n') != null or
        std.mem.indexOfScalar(u8, head, '<') != null;
    if (rich) {
        // Prefer a CLOSING <|"|> as the boundary (dropped-opener case). Confirm
        // it's a closer — the byte after it (past whitespace) must be a
        // separator/closer — so a LATER field's OPENING delimiter isn't grabbed.
        if (std.mem.indexOf(u8, body[pos..], gemma4_str_delim)) |close_rel| {
            const after = gemma4SkipWs(body, pos + close_rel + gemma4_str_delim.len);
            const is_closer = after >= body.len or body[after] == ',' or body[after] == '}' or body[after] == ']';
            if (is_closer) {
                const value = std.mem.trim(u8, body[pos .. pos + close_rel], " \t\n\r");
                appendJsonString(allocator, result, value) catch return null;
                return pos + close_rel + gemma4_str_delim.len;
            }
        }
        // No usable closing delimiter at all. Under truncation the rest of the
        // body IS the fragment: consuming to the first comma would ship a
        // partial value AND shred the remainder into bogus keys, and the
        // last-`}` heuristic below would cut a truncated page at some interior
        // CSS brace — drop the whole value instead.
        if (input_truncated) {
            cut.* = true;
            return body.len;
        }
        // Complete call, both delimiters dropped. At the TOP level the value
        // runs to the object's closing `}` (the last one); nested values keep
        // the narrow scan since the outer `}` isn't theirs.
        if (depth == 0) {
            if (std.mem.lastIndexOfScalar(u8, body, '}')) |last_brace| {
                if (last_brace > pos) {
                    const value = std.mem.trim(u8, body[pos..last_brace], " \t\n\r");
                    appendJsonString(allocator, result, value) catch return null;
                    return last_brace;
                }
            }
        }
    }

    // A short bare string that ran to end-of-body under truncation is a
    // fragment too (`command:ls -la` cut mid-flags) — uniform rule.
    if (sep_rel == null and input_truncated) {
        cut.* = true;
        return body.len;
    }

    // Plain short bare string — terminate at the first separator (unchanged).
    appendJsonString(allocator, result, head) catch return null;
    return pos + first_sep;
}

/// End of a `<parameter=…>` value: the LAST `</parameter>` before the next opener (a value may
/// spell its own close tag); an unclosed parameter falls back to the first close.
fn hermesValueEnd(body: []const u8, val_start: usize) ?usize {
    const bound = std.mem.indexOfPos(u8, body, val_start, "<parameter=") orelse body.len;
    var best: ?usize = null;
    var i = val_start;
    while (std.mem.indexOfPos(u8, body, i, "</parameter>")) |at| {
        if (at >= bound) break;
        best = at;
        i = at + 1;
    }
    return best orelse std.mem.indexOfPos(u8, body, val_start, "</parameter>");
}

/// If `at` falls inside a `<parameter=…>` value, the position just past that value's close; else null.
fn hermesParamSpanEnclosing(body: []const u8, at: usize) ?usize {
    const open = std.mem.lastIndexOf(u8, body[0..at], "<parameter=") orelse return null;
    const gt = std.mem.indexOfScalarPos(u8, body, open, '>') orelse return null;
    if (gt >= at) return null;
    const end = hermesValueEnd(body, gt + 1) orelse return null;
    if (end <= at) return null;
    return end + "</parameter>".len;
}

fn parseHermesToolCall(allocator: std.mem.Allocator, block: []const u8) ?ParsedToolCall {
    const fn_start_tag = "<function=";
    const fn_start = std.mem.indexOf(u8, block, fn_start_tag) orelse return null;
    var name_start = fn_start + fn_start_tag.len;
    while (name_start < block.len and std.ascii.isWhitespace(block[name_start])) name_start += 1;
    // A name never spans a line or a tag: a newline or `<` before the `>` means the model
    // dropped the `>`, so the name ends there and the body starts there.
    const name_end = std.mem.indexOfAnyPos(u8, block, name_start, ">\n\r<") orelse return null;
    const fn_name = std.mem.trim(u8, block[name_start..name_end], " \t");
    if (!isPlausibleParamName(fn_name)) return null;

    var args_map = std.ArrayList(u8).empty;
    defer args_map.deinit(allocator);
    args_map.append(allocator, '{') catch return null;

    const fn_body_start = if (block[name_end] == '>') name_end + 1 else name_end;
    const fn_end = blk: {
        var from = fn_body_start;
        while (std.mem.indexOfPos(u8, block, from, "</function>")) |found| {
            if (hermesParamSpanEnclosing(block, found)) |past| {
                from = past;
                continue;
            }
            break :blk found - fn_body_start;
        }
        break :blk block.len - fn_body_start;
    };
    const fn_body = block[fn_body_start .. fn_body_start + fn_end];
    // Qwen3.8 frames the body and every value with newlines; MiMo writes both bare,
    // so there a value's edge newlines are its own bytes.
    const framed = std.mem.startsWith(u8, fn_body, "\n") or std.mem.startsWith(u8, fn_body, "\r\n");

    // Track emitted parameter names so a repeated `<parameter=NAME>` can't
    // produce a duplicate JSON key (std.json rejects those with DuplicateField —
    // invalid JSON to the client; live soak record 443, a 0.8B model emitting two
    // `<parameter=edits>` blocks). First occurrence wins.
    var seen_names = std.ArrayList([]const u8).empty;
    defer seen_names.deinit(allocator);

    var param_search: usize = 0;
    var first_param = true;
    while (std.mem.indexOf(u8, fn_body[param_search..], "<parameter=")) |ps| {
        const p_name_start = param_search + ps + "<parameter=".len;
        const p_name_end = std.mem.indexOf(u8, fn_body[p_name_start..], ">") orelse break;
        const p_name = std.mem.trim(u8, fn_body[p_name_start .. p_name_start + p_name_end], " \n");

        // A well-formed `<parameter=NAME>` name is a clean token. When the model
        // malforms the tag — e.g. `<parameter=limit=1` with no closing `>` — the
        // `>`-scan spills across a newline into `</parameter`, and interpolating
        // that raw would emit INVALID JSON (live soak record 317). Detect the
        // malformed name and skip just this opener, so the well-formed sibling
        // parameter after it is still recovered. Advancing to p_name_start (past
        // the `<parameter=` we matched) guarantees forward progress.
        if (!isPlausibleParamName(p_name)) {
            param_search = p_name_start;
            continue;
        }

        const p_val_start = p_name_start + p_name_end + 1;
        const p_val_end = (hermesValueEnd(fn_body, p_val_start) orelse break) - p_val_start;
        const raw_val = fn_body[p_val_start .. p_val_start + p_val_end];
        const p_val = if (framed) stripHermesValueFraming(raw_val) else raw_val;

        // Skip a duplicate name (first wins); still advance past its block.
        var dup = false;
        for (seen_names.items) |seen| {
            if (std.mem.eql(u8, seen, p_name)) {
                dup = true;
                break;
            }
        }
        if (dup) {
            param_search = p_val_start + p_val_end + "</parameter>".len;
            continue;
        }
        seen_names.append(allocator, p_name) catch return null;

        if (!first_param) args_map.append(allocator, ',') catch return null;
        first_param = false;

        // Escape the NAME too — belt-and-braces so no parameter name can ever
        // produce invalid JSON, even if a future malformed shape slips the guard.
        appendJsonString(allocator, &args_map, p_name) catch return null;
        args_map.append(allocator, ':') catch return null;

        // Whitespace around a NUMBER/BOOLEAN/null carries no meaning, so a
        // padded scalar is still typed from its spelling. A STRING's own
        // whitespace is PAYLOAD and ships verbatim: an `old_string` must match
        // the file "exactly, including indentation", and trimming it either
        // failed the edit or — worse — matched a different, un-indented
        // occurrence mid-line. Only the template framing is removed.
        const literal_val = std.mem.trim(u8, p_val, " \t\r\n");
        if (isJsonLiteral(literal_val)) {
            args_map.appendSlice(allocator, literal_val) catch return null;
        } else {
            appendJsonString(allocator, &args_map, p_val) catch return null;
        }

        param_search = p_val_start + p_val_end + "</parameter>".len;
    }

    args_map.append(allocator, '}') catch return null;

    return .{
        .name = allocator.dupe(u8, fn_name) catch return null,
        .arguments = allocator.dupe(u8, args_map.items) catch return null,
    };
}

/// The Hermes/Qwen XML tool-call template frames a parameter value with EXACTLY
/// one newline on each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
/// Qwen3.8 render test in this file), so exactly that framing is what the parser
/// may remove. Everything else between the tags is the value's own bytes: eating
/// a run of newlines, or the spaces an `old_string` needs to match a file
/// "exactly, including indentation", silently mis-edits.
fn stripHermesValueFraming(raw: []const u8) []const u8 {
    var v = raw;
    if (std.mem.startsWith(u8, v, "\r\n")) {
        v = v[2..];
    } else if (std.mem.startsWith(u8, v, "\n")) {
        v = v[1..];
    }
    if (std.mem.endsWith(u8, v, "\r\n")) {
        v = v[0 .. v.len - 2];
    } else if (std.mem.endsWith(u8, v, "\n")) {
        v = v[0 .. v.len - 1];
    }
    return v;
}

/// MiniCPM5 V3 XML tool calls — attribute-quoted, one call per `<function>`
/// tag, no outer wrapper:
///   <function name="shell">
///     <param name="command">pwd</param>
///   </function>
/// Multiple calls appear as consecutive `<function name="…">…</function>`
/// blocks. A `<param>` value may be CDATA-wrapped
/// (`<param name="x"><![CDATA[ … ]]></param>`) — the payload is used
/// verbatim. This is its own scan, not part of the generic `<tool` family or
/// the equals-sign Hermes function-tag scan just above (`<function=NAME>`):
/// the marker is `<function` followed by WHITESPACE (the attribute form),
/// which the equals-sign form never produces (its char right after
/// `<function` is always `=`), so the two can never false-fire on each
/// other. Duplicate `<param name="K">` — first occurrence wins, mirroring
/// `parseHermesToolCall`'s `<parameter=>` dedup and `parseArgKeyToolCalls`'s
/// `<arg_key>` dedup (both exist because std.json rejects duplicate object
/// keys). An unrecognized function name or an undeclared parameter is never
/// rejected here — schema validation is centralized downstream
/// (`toolCallConformsToSchema` et al., `src/server.zig`'s
/// `parseToolCallsForRequest`), exactly as for every other tag-format
/// dialect. Truncation (EOS or max_tokens mid-call): when no `</function>`
/// close is found, the param scan is bounded by end-of-text instead — any
/// COMPLETE `<param>…</param>` pairs before the cut still recover (mirrors
/// `parseArgKeyToolCalls`'s `call_end` fallback); a partial trailing value is
/// never salvaged.
fn parseMiniCpm5ToolCalls(allocator: std.mem.Allocator, text: []const u8, calls: *std.ArrayList(ParsedToolCall)) !void {
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, "<function")) |p| {
        const after = p + "<function".len;
        // The attribute form requires whitespace right after `<function` —
        // anything else is the equals-sign Hermes form or unrelated text
        // (e.g. `<functional>`), left for that scan / plain content.
        if (after >= text.len or !std.ascii.isWhitespace(text[after])) {
            pos = after;
            continue;
        }
        const tag_end = std.mem.indexOfScalarPos(u8, text, after, '>') orelse {
            pos = text.len; // Unterminated opening tag — nothing left to recover.
            break;
        };
        const open_tag = text[after..tag_end];
        const fn_name = miniCpm5AttrValue(open_tag, "name") orelse {
            // No `name="…"` attribute — not a recognizable MiniCPM5 call.
            pos = tag_end + 1;
            continue;
        };
        if (fn_name.len == 0) {
            pos = tag_end + 1;
            continue;
        }

        const body_start = tag_end + 1;
        const close_pos = miniCpm5FindCloseTag(text, body_start, "</function>");
        const body_end = if (close_pos) |cp| cp else text.len;
        const body = text[body_start..body_end];

        var args_map: std.json.ObjectMap = .empty;
        defer args_map.deinit(allocator);

        var pscan: usize = 0;
        while (std.mem.indexOfPos(u8, body, pscan, "<param")) |pp| {
            const p_after = pp + "<param".len;
            if (p_after >= body.len or !std.ascii.isWhitespace(body[p_after])) {
                pscan = p_after;
                continue;
            }
            const p_tag_end = std.mem.indexOfScalarPos(u8, body, p_after, '>') orelse break;
            const p_open_tag = body[p_after..p_tag_end];
            const p_name = miniCpm5AttrValue(p_open_tag, "name") orelse {
                pscan = p_tag_end + 1;
                continue;
            };
            const p_val_start = p_tag_end + 1;
            const p_close_pos = miniCpm5FindCloseTag(body, p_val_start, "</param>") orelse break;
            const value = miniCpm5ParamValue(body[p_val_start..p_close_pos]);

            if (p_name.len > 0 and args_map.getEntry(p_name) == null) {
                try args_map.put(allocator, p_name, .{ .string = value });
            }
            pscan = p_close_pos + "</param>".len;
        }

        // An UNCLOSED function is salvaged only with EVIDENCE that a call was
        // really under way: a completed `<param>…</param>`, or at minimum a
        // `<param` opener. Without that, a model merely WRITING ABOUT the
        // syntax ("you write <function name=\"shell\"> to open a call") is
        // promoted into an executable `shell({})` — an unintended invocation
        // for a zero-arg tool, a bogus call replacing the answer otherwise.
        //
        // The `<param` opener half is NOT belt-and-braces; it is required by a
        // real capture. MiniCPM5-1B-OptiQ-4bit at max_tokens=12 emits exactly:
        //     <function name="shell"><param name="command">git status
        // — a genuine truncated call with ZERO completed params. Demanding a
        // COMPLETE pair discarded it. Prose naming a function carries no
        // `<param` at all, so the two stay separable.
        //
        // A properly CLOSED call is untouched either way, including the
        // zero-argument form `<function name="get_time"></function>`, which the
        // same model emits verbatim: its `</function>` is the model's own
        // commitment that the call is complete.
        if (close_pos == null and args_map.count() == 0 and
            std.mem.indexOf(u8, body, "<param") == null)
        {
            pos = text.len;
            continue;
        }

        const args_str = try std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = args_map }, .{});
        errdefer allocator.free(args_str);
        const name_owned = try allocator.dupe(u8, fn_name);
        errdefer allocator.free(name_owned);
        try calls.append(allocator, .{ .name = name_owned, .arguments = args_str });

        pos = if (close_pos) |cp| cp + "</function>".len else text.len;
    }
}

/// Find `close_tag` (`</param>` or `</function>`) starting at `from`,
/// skipping over the payload of any `<![CDATA[ … ]]>` span encountered along
/// the way. A CDATA payload may legitimately contain the literal closing-tag
/// text — e.g. a `write_file` call whose content documents this very XML
/// format, or writes an unrelated `</param>`-shaped snippet — and CDATA's
/// whole purpose is to carry such content safely; a naive substring search
/// would truncate the value/call at that inner occurrence. An unterminated
/// CDATA open (no matching `]]>`) has no knowable end, so the search gives
/// up (null) rather than guess — the caller's existing truncation path
/// already treats "no close found" as EOS/max_tokens truncation.
fn miniCpm5FindCloseTag(haystack: []const u8, from: usize, close_tag: []const u8) ?usize {
    var i = from;
    while (true) {
        const cdata_at = std.mem.indexOfPos(u8, haystack, i, "<![CDATA[");
        const close_at = std.mem.indexOfPos(u8, haystack, i, close_tag);
        if (cdata_at) |cd| {
            if (close_at == null or cd < close_at.?) {
                const payload_start = cd + "<![CDATA[".len;
                const end_pos = std.mem.indexOfPos(u8, haystack, payload_start, "]]>") orelse return null;
                i = end_pos + "]]>".len;
                continue;
            }
        }
        return close_at;
    }
}

/// Extract `key="value"` or `key='value'` from an opening tag's attribute
/// span (the text strictly between the tag name and its closing `>`).
/// Requires a word boundary before `key` so `data-name=` can't match `name=`.
/// An unquoted `key=value` is malformed (MiniCPM5 always quotes) and yields
/// null, same as a missing key.
pub fn miniCpm5AttrValue(tag_attrs: []const u8, key: []const u8) ?[]const u8 {
    var buf: [40]u8 = undefined;
    if (key.len + 1 > buf.len) return null;
    const needle = std.fmt.bufPrint(&buf, "{s}=", .{key}) catch return null;
    var search_from: usize = 0;
    while (std.mem.indexOf(u8, tag_attrs[search_from..], needle)) |rel| {
        const at = search_from + rel;
        if (at > 0) {
            const before = tag_attrs[at - 1];
            if (std.ascii.isAlphanumeric(before) or before == '_' or before == '-') {
                search_from = at + needle.len;
                continue;
            }
        }
        const q_pos = at + needle.len;
        if (q_pos >= tag_attrs.len) return null;
        const quote = tag_attrs[q_pos];
        if (quote != '"' and quote != '\'') {
            search_from = q_pos;
            continue;
        }
        const val_start = q_pos + 1;
        const val_end_rel = std.mem.indexOfScalar(u8, tag_attrs[val_start..], quote) orelse return null;
        return tag_attrs[val_start .. val_start + val_end_rel];
    }
    return null;
}

/// Resolve a `<param>` body to its final value. CDATA form
/// (`<![CDATA[ … ]]>`, detected after trimming incidental surrounding
/// whitespace) yields its payload byte-exact and unfiltered — CDATA's whole
/// purpose is to carry content verbatim. Otherwise it reuses
/// `stripHermesValueFraming`, removing at most ONE newline per side.
/// This is a deliberately CONSERVATIVE choice. Captured MiniCPM5-1B-OptiQ-4bit
/// output is VALUE-ADJACENT — `<param name="command">git status</param>`, no
/// framing newlines at all — so on real emissions this strip is a no-op and the
/// question is moot. It exists for the newline-framed layout, which the fixtures
/// use but which no capture has yet shown; removing at most one newline per side
/// is the least-lossy reading if that layout is ever emitted. Trimming a RUN, which is what
/// this did first, eats blank lines the value itself carries - the
/// indentation-destroying class fixed upstream for `<parameter=>`. Interior and
/// space-only padding (shell flags, indentation) are kept either way.
fn miniCpm5ParamValue(raw: []const u8) []const u8 {
    const probe = std.mem.trim(u8, raw, " \t\n\r");
    const cdata_open = "<![CDATA[";
    const cdata_close = "]]>";
    if (std.mem.startsWith(u8, probe, cdata_open) and std.mem.endsWith(u8, probe, cdata_close)) {
        // The close must END the probe. Keying on the LAST `]]>` anywhere let
        // `<![CDATA[x]]>tail` return just `x` — silently discarding character
        // data the model sent. A section that does not end the value is not a
        // CDATA-wrapped value; fall through and treat the whole thing as text.
        const inner = probe[cdata_open.len .. probe.len - cdata_close.len];
        return inner;
    }
    return stripHermesValueFraming(raw);
}

/// A parameter name from a well-formed `<parameter=NAME>` tag is a short token
/// with no whitespace or markup. Reject names carrying a newline, angle bracket,
/// quote, or whitespace — those signal the `>`-scan spilled past a malformed tag.
fn isPlausibleParamName(name: []const u8) bool {
    if (name.len == 0 or name.len > 128) return false;
    for (name) |c| {
        switch (c) {
            '\n', '\r', '\t', ' ', '<', '>', '"' => return false,
            else => {},
        }
    }
    return true;
}

fn isJsonLiteral(s: []const u8) bool {
    if (s.len == 0) return false;
    if (std.mem.eql(u8, s, "true") or std.mem.eql(u8, s, "false") or std.mem.eql(u8, s, "null")) return true;
    // Objects/arrays are handled by brace-matching in the caller, not here
    return isJsonNumber(s);
}

/// JSON's number grammar; `parseFloat` is wider (`Infinity`, `0755`, `.5`) and the literal is spliced UNQUOTED.
fn isJsonNumber(s: []const u8) bool {
    var i: usize = 0;
    if (i < s.len and s[i] == '-') i += 1;
    if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
    if (s[i] == '0') {
        i += 1;
    } else {
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and s[i] == '.') {
        i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    if (i < s.len and (s[i] == 'e' or s[i] == 'E')) {
        i += 1;
        if (i < s.len and (s[i] == '+' or s[i] == '-')) i += 1;
        if (i >= s.len or !std.ascii.isDigit(s[i])) return false;
        while (i < s.len and std.ascii.isDigit(s[i])) i += 1;
    }
    return i == s.len;
}

/// A request's `tool_choice`, read from any surface's wire shape: chat nests the
/// name under `function`, Responses and Anthropic carry it flat, Anthropic
/// spells required `any`. Anything unrecognised is `auto`.
pub const ToolChoice = union(enum) {
    auto,
    none,
    required,
    named: []const u8,

    /// The prompt-side instruction of a forced choice; null for auto and none.
    pub fn instruction(self: ToolChoice, allocator: std.mem.Allocator) !?[]u8 {
        return switch (self) {
            .auto, .none => null,
            .required => try allocator.dupe(u8, "\nYou MUST call one of the available functions. Do not respond with text."),
            .named => |name| try std.fmt.allocPrint(allocator, "\nYou MUST call the function \"{s}\". Do not respond with text.", .{name}),
        };
    }
};

pub fn parseToolChoice(value: ?std.json.Value) ToolChoice {
    const v = value orelse return .auto;
    switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "none")) return .none;
            if (std.mem.eql(u8, s, "required") or std.mem.eql(u8, s, "any")) return .required;
            return .auto;
        },
        .object => |obj| {
            const kind = if (obj.get("type")) |t| (if (t == .string) t.string else "") else "";
            if (std.mem.eql(u8, kind, "none")) return .none;
            if (std.mem.eql(u8, kind, "any") or std.mem.eql(u8, kind, "required")) return .required;
            const holder = if (obj.get("function")) |f| (if (f == .object) f.object else obj) else obj;
            const name = holder.get("name") orelse return .auto;
            if (name != .string or name.string.len == 0) return .auto;
            return .{ .named = name.string };
        },
        else => return .auto,
    }
}

/// Every function name the OpenAI-shaped tools array declares, in order; empty
/// when it does not parse. Strings live in `arena`.
pub fn declaredToolNames(arena: std.mem.Allocator, tools_json: []const u8) ![]const []const u8 {
    const tools = std.json.parseFromSliceLeaky(std.json.Value, arena, tools_json, .{}) catch return &.{};
    if (tools != .array) return &.{};
    var names = std.ArrayList([]const u8).empty;
    for (tools.array.items) |tool| {
        if (tool != .object) continue;
        const holder = if (tool.object.get("function")) |f| (if (f == .object) f.object else tool.object) else tool.object;
        const name = holder.get("name") orelse continue;
        if (name == .string and name.string.len > 0) try names.append(arena, name.string);
    }
    return names.toOwnedSlice(arena);
}

/// A named choice for a function the request never declared cannot be honoured.
pub fn toolChoiceNamesUndeclared(allocator: std.mem.Allocator, choice: ToolChoice, tools_json: ?[]const u8) bool {
    const name = switch (choice) {
        .named => |n| n,
        else => return false,
    };
    return !toolNameIsDeclared(allocator, tools_json orelse "[]", name);
}

/// Append tool definitions as a system prompt section.
pub fn appendToolSystemPrompt(allocator: std.mem.Allocator, result_buf: *std.ArrayList(u8), tools_json: []const u8, tool_choice_instruction: ?[]const u8) !void {
    try result_buf.appendSlice(allocator,
        \\You are a helpful assistant with access to the following functions. To call a function, respond with a JSON object in the following format:
        \\<tool_call>
        \\{"name": "function_name", "arguments": {"arg1": "value1"}}
        \\</tool_call>
        \\
        \\Available functions:
        \\
    );
    try result_buf.appendSlice(allocator, tools_json);
    if (tool_choice_instruction) |instr| {
        try result_buf.appendSlice(allocator, instr);
    }
}

test "parseToolChoice reads every surface's wire shape" {
    const Case = struct { json: []const u8, want: ToolChoice };
    const cases = [_]Case{
        // OpenAI chat completions
        .{ .json = "\"auto\"", .want = .auto },
        .{ .json = "\"none\"", .want = .none },
        .{ .json = "\"required\"", .want = .required },
        .{ .json = "{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}}", .want = .{ .named = "get_time" } },
        // OpenAI Responses
        .{ .json = "{\"type\":\"function\",\"name\":\"get_time\"}", .want = .{ .named = "get_time" } },
        .{ .json = "{\"type\":\"allowed_tools\",\"mode\":\"auto\"}", .want = .auto },
        // Anthropic messages
        .{ .json = "{\"type\":\"auto\"}", .want = .auto },
        .{ .json = "{\"type\":\"none\"}", .want = .none },
        .{ .json = "{\"type\":\"any\",\"disable_parallel_tool_use\":true}", .want = .required },
        .{ .json = "{\"type\":\"tool\",\"name\":\"get_time\"}", .want = .{ .named = "get_time" } },
        // A named choice with no usable name forces nothing.
        .{ .json = "{\"type\":\"tool\"}", .want = .auto },
        .{ .json = "{\"type\":\"function\",\"function\":{\"name\":\"\"}}", .want = .auto },
        .{ .json = "7", .want = .auto },
    };
    for (cases) |c| {
        const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.json, .{});
        defer parsed.deinit();
        try testing.expectEqualDeep(c.want, parseToolChoice(parsed.value));
    }
    try testing.expectEqualDeep(ToolChoice.auto, parseToolChoice(null));
}

test "a named tool_choice must name a declared tool" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"get_time","parameters":{}}}]
    ;
    try testing.expect(!toolChoiceNamesUndeclared(allocator, .{ .named = "get_time" }, tools));
    try testing.expect(toolChoiceNamesUndeclared(allocator, .{ .named = "rm_rf" }, tools));
    try testing.expect(toolChoiceNamesUndeclared(allocator, .{ .named = "get_time" }, null));
    try testing.expect(!toolChoiceNamesUndeclared(allocator, .required, tools));
    try testing.expect(!toolChoiceNamesUndeclared(allocator, .auto, null));
}

test "declaredToolNames lists every function a tools array declares, in order" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const allocator = arena_state.allocator();
    const names = try declaredToolNames(allocator,
        \\[{"type":"function","function":{"name":"get_time"}},{"name":"flat_tool"},{"type":"function","function":{}},{"type":"function","function":{"name":"get_weather"}}]
    );
    try testing.expectEqual(@as(usize, 3), names.len);
    try testing.expectEqualStrings("get_time", names[0]);
    try testing.expectEqualStrings("flat_tool", names[1]);
    try testing.expectEqualStrings("get_weather", names[2]);
    try testing.expectEqual(@as(usize, 0), (try declaredToolNames(allocator, "not json")).len);
}

test "only a forced tool_choice carries a prompt instruction" {
    const allocator = testing.allocator;
    try testing.expect(try (ToolChoice{ .auto = {} }).instruction(allocator) == null);
    try testing.expect(try (ToolChoice{ .none = {} }).instruction(allocator) == null);

    const any = (try (ToolChoice{ .required = {} }).instruction(allocator)).?;
    defer allocator.free(any);
    try testing.expectEqualStrings("\nYou MUST call one of the available functions. Do not respond with text.", any);

    const one = (try (ToolChoice{ .named = "get_time" }).instruction(allocator)).?;
    defer allocator.free(one);
    try testing.expectEqualStrings("\nYou MUST call the function \"get_time\". Do not respond with text.", one);
}

test "parseToolCalls JSON format" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // arguments should be valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Qwen3.6 MoE nested-name garbage (real capture)" {
    // Captured verbatim from Qwen3.6-35B-A3B-6bit generating a tool call in
    // streaming mode with enable_thinking=false and tools present.
    // parseToolCalls previously returned null here because the top-level "name"
    // is an object, not a string — tokens then leaked as plain-text content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": {"name": {"name":  "write", "arguments": {"path": "/tmp/x/app.test.js", "content": "hello"}}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/x/app.test.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello", parsed.value.object.get("content").?.string);
}

test "parseToolCalls Qwen MoE double-nested name" {
    // Two layers of nesting — defensive test for a slightly different shape.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name":{"name":"shell","arguments":{"command":"ls"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls flat Qwen MoE shape (no arguments wrapper)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\", \"command\": \"mkdir -p 2ddungeon\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p 2ddungeon", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("name") == null);
}

test "parseToolCalls flat shape with multiple top-level args" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"writeFile\", \"path\": \"/tmp/a\", \"content\": \"hi\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls flat shape ignores id/type metadata" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"id\": \"call_1\", \"type\": \"function\", \"name\": \"shell\", \"command\": \"ls\"}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("id") == null);
    try testing.expect(parsed.value.object.get("type") == null);
}

test "parseToolCalls recovers writeFile content with RAW newlines (small-model big-file escaping)" {
    const allocator = testing.allocator;
    // Small models writing a big file often emit literal newlines inside the
    // JSON `content` string instead of `\n` — strict JSON rejects raw control
    // bytes, so pre-fix the whole call was dropped and the file leaked as text.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"a.js\",\"content\":\"const x = 1;\nconst y = 2;\n\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("const x = 1;\nconst y = 2;\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with RAW tab" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"m.py\",\"content\":\"def f():\n\treturn 1\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("def f():\n\treturn 1", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with UNESCAPED inner quotes (HTML)" {
    const allocator = testing.allocator;
    // HTML/code with attribute quotes — `<meta charset="UTF-8">` — is the other
    // half of the escaping class: the model forgets to backslash the inner `"`.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"i.html\",\"content\":\"<meta charset=\"UTF-8\"><a href=\"x\">go</a>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("i.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("<meta charset=\"UTF-8\"><a href=\"x\">go</a>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers writeFile content with BOTH raw newlines and unescaped quotes" {
    const allocator = testing.allocator;
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"p.html\",\"content\":\"<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<!DOCTYPE html>\n<meta charset=\"UTF-8\">\n<title>Hi</title>", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers content with lone backslash (Windows path / regex)" {
    const allocator = testing.allocator;
    // `\U` and `\d` are invalid JSON escapes — strict parse rejects them.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"C:\\Users\\app.js\",\"content\":\"re.match(\\d+)\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("C:\\Users\\app.js", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("re.match(\\d+)", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers Gemma 4 call:NAME{json} with raw newline content" {
    const allocator = testing.allocator;
    // Gemma 4's JSON-first branch hits the same strict-parse wall on raw bytes.
    const text = "<|tool_call>call:writeFile{\"path\":\"g.txt\",\"content\":\"line1\nline2\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("line1\nline2", parsed.value.object.get("content").?.string);
}

test "parseToolCalls leaves valid escaped content untouched (no regression)" {
    const allocator = testing.allocator;
    // Already-correct JSON must pass straight through — the recovery path only
    // runs after strict parse fails, so this never touches looseRepair.
    const text = "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"ok.js\",\"content\":\"a\\nb\\t\\\"q\\\"\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a\nb\t\"q\"", parsed.value.object.get("content").?.string);
}

test "parseToolCalls recovers truncated <function=writeFile> (max_tokens mid-content)" {
    const allocator = testing.allocator;
    // Live JFK-novel capture (2026-06-20): a Hermes-format writeFile dumped a
    // 19k-char novel into one <parameter=content> and hit the token cap before
    // any closing tag. Pre-fix the close_rel==null branch only tried JSON, so
    // the whole call was DROPPED and leaked as visible text. We must recover at
    // least the tool NAME (args may be empty — content is truncated, not
    // salvaged) so the client fires the chunk/append nudge instead of "use JSON".
    const text = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE LION OF MASSACHUSETTS\n\nChapter 1. The young senator rose before dawn";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    // Args are valid JSON (empty object) — the parameter never closed.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
}

test "parseToolCalls recovers a bare <function=> call with NO <tool_call> opener" {
    const allocator = testing.allocator;
    // Live capture (soak, 2026-07-09, Qwen3.6-class via Claude Code): the model
    // emitted the CLOSING </tool_call> but DROPPED the opening <tool_call>, so the
    // whole call arrived as `<function=Write>…</function></tool_call>`. The outer
    // scan triggers on the substring `<tool` — which `</tool_call>` does NOT
    // contain (it is `</too…`) — so parseHermesToolCall was never reached and the
    // entire Write (with its raw <function=/<parameter= markup) leaked into the
    // user-visible text. Same big-file-write dropped-delimiter CLASS as the
    // truncated-opener bugs; recover the call so the markup never leaks.
    const text = "\n\n<function=Write>\n<parameter=file_path>\n/tmp/index.html\n</parameter>\n" ++
        "<parameter=content>\n<!DOCTYPE html>\n<html></html>\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
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
    try testing.expectEqualStrings("/tmp/index.html", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("<!DOCTYPE html>\n<html></html>", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall ends a function name whose > is missing at the line break" {
    const allocator = testing.allocator;
    const cases = [_]struct { text: []const u8, path: ?[]const u8 }{
        .{ .text = "<tool_call>\n<function=read_file\n</parameter>\n</function>\n</tool_call>", .path = null },
        .{ .text = "<tool_call>\n<function=read_file\n<parameter=path>\n./a.txt\n</parameter>\n</function>\n</tool_call>", .path = "./a.txt" },
        .{ .text = "<tool_call>\n<function=read_file<parameter=path>./a.txt</parameter></function>\n</tool_call>", .path = "./a.txt" },
    };
    for (cases) |case| {
        const calls = (try parseToolCalls(allocator, case.text)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings("read_file", calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        if (case.path) |p| {
            try testing.expectEqualStrings(p, parsed.value.object.get("path").?.string);
        } else {
            try testing.expectEqual(@as(usize, 0), parsed.value.object.count());
        }
    }
}

test "parseHermesToolCall never emits invalid JSON on a malformed <parameter=> tag" {
    // Live soak capture (record 317): the model wrote `<parameter=limit=1` — using
    // `=1` instead of closing the tag with `>` and a value. The `>`-scan then
    // spilled the 'name' across a newline into `</parameter`, and the raw
    // (unescaped) name interpolation produced INVALID JSON args that flowed to the
    // client. Two guarantees: (1) args are ALWAYS valid JSON; (2) the well-formed
    // sibling `<parameter=path>` is still recovered.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=cat>\n<parameter=limit=1\n</parameter>\n<parameter=path>\n./parse.py\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cat", calls[0].name);
    // Hard invariant: valid JSON object, no matter how malformed the tag was.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    // The clean sibling parameter is recovered.
    try testing.expectEqualStrings("./parse.py", parsed.value.object.get("path").?.string);
}

test "parseHermesToolCall dedups a repeated <parameter=> name (no duplicate JSON key)" {
    // Live soak capture (record 443, Qwen3.5-0.8B): the model emitted TWO
    // `<parameter=edits>` blocks in one call. Pre-fix that produced
    // `{"edits":…,"edits":…,"path":…}`, which std.json rejects with
    // error.DuplicateField — invalid JSON to the client. Dedup so the args are
    // always a valid object.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=edit>\n<parameter=edits>\nfirst\n</parameter>\n" ++
        "<parameter=edits>\nsecond\n</parameter>\n<parameter=path>\n./notes.md\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expectEqualStrings("./notes.md", parsed.value.object.get("path").?.string);
    // Exactly one `edits` key survives.
    try testing.expect(parsed.value.object.get("edits") != null);
}

test "parseHermesToolCall keeps leading/trailing whitespace inside a parameter value" {
    // The Hermes/Qwen XML template frames a value with EXACTLY one newline on
    // each side (`<parameter=NAME>\nVALUE\n</parameter>` — pinned by the
    // Qwen3.8 render test above), so that framing is the only thing the parser
    // may remove. It also did `std.mem.trim(u8, p_val, " ")`, which ate an
    // `old_string`/`new_string`'s own indentation: an Edit whose needle must
    // match the file "exactly, including indentation" then either failed, or —
    // worse — matched a DIFFERENT, un-indented occurrence mid-line and wrote
    // the replacement at the wrong nesting. Trailing whitespace is payload too
    // (a `new_string` that must end in two spaces, a markdown hard line break).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=Edit>\n<parameter=file_path>\n/tmp/app.py\n</parameter>\n" ++
        "<parameter=old_string>\n        return None\n</parameter>\n" ++
        "<parameter=new_string>\n        return self.value  \n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/app.py", parsed.value.object.get("file_path").?.string);
    try testing.expectEqualStrings("        return None", parsed.value.object.get("old_string").?.string);
    try testing.expectEqualStrings("        return self.value  ", parsed.value.object.get("new_string").?.string);
}

test "parseHermesToolCall strips only the ONE framing newline on each side" {
    // A value that legitimately begins or ends with a blank line keeps it: the
    // template adds one newline, not "any run of newlines". A multi-line
    // `content` written into a file is byte-exact or the file is wrong.
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=write>\n<parameter=path>\n./a.md\n</parameter>\n" ++
        "<parameter=content>\n\n# Title\n\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\n# Title\n", parsed.value.object.get("content").?.string);
}

test "parseHermesToolCall still types a space-padded scalar from its spelling" {
    // The other half: whitespace around a NUMBER or BOOLEAN carries no meaning,
    // so a one-line `<parameter=limit> 5 </parameter>` must still ship JSON 5,
    // not the string " 5 " — fixing the string class must not break the literal
    // class (`isJsonLiteral` never matches a padded token).
    const allocator = testing.allocator;
    const text = "<tool_call>\n<function=ls>\n<parameter=path>\n./src\n</parameter>\n" ++
        "<parameter=limit> 5 </parameter>\n<parameter=recursive>\n true \n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    try testing.expectEqual(true, parsed.value.object.get("recursive").?.bool);
}

test "parseHermesToolCall strips a CRLF framing pair as one unit" {
    // A model that emits Windows line endings frames the value with
    // `>\r\nVALUE\r\n</parameter>`. Dropping the "\n" half alone would leave a
    // stray CR at both ends of every value — the framing is one unit.
    const allocator = testing.allocator;
    const text = "<tool_call>\r\n<function=write>\r\n<parameter=content>\r\n" ++
        "  hi\r\n</parameter>\r\n</function>\r\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("  hi", parsed.value.object.get("content").?.string);
}

test "parseToolCalls does NOT treat prose mentioning <function=> as a call" {
    const allocator = testing.allocator;
    // The bare-<function=> fallback must not fire on prose. Without a matching
    // </function> AND a real <parameter=, parseHermesToolCall recovers nothing
    // useful — but guard against a false positive on a lone mention.
    const text = "To define one you write `<function=name>` in the Hermes format.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        defer {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        }
        // If it parsed anything, it must at least be well-formed JSON args —
        // never leak the raw prose as a tool name.
        try testing.expect(cs.len == 0);
    }
}

test "looseRepair does not fabricate a tool call from non-JSON prose" {
    const allocator = testing.allocator;
    // A stray `{` in prose must not become a tool call via the recovery path.
    const text = "Here is the plan {step one, step two} and we proceed.";
    const calls = try parseToolCalls(allocator, text);
    if (calls) |cs| {
        for (cs) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(cs);
    }
    try testing.expect(calls == null);
}

test "parseToolCalls returns null for name-only object (no real args)" {
    const allocator = testing.allocator;
    const text = "<tool_call>\n{\"name\": \"shell\"}\n</tool_call>";
    const result = try parseToolCalls(allocator, text);
    try testing.expect(result == null);
}

test "parseToolCalls marks raw-JSON fallback calls as inferred, tag calls as explicit" {
    const allocator = testing.allocator;
    // Tagged call → explicit (never schema-name-filtered).
    {
        const calls = (try parseToolCalls(allocator, "<tool_call>{\"name\":\"shell\",\"arguments\":{\"command\":\"ls\"}}</tool_call>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(!calls[0].inferred);
    }
    // Bare raw-JSON object → inferred (flat shape included).
    {
        const calls = (try parseToolCalls(allocator, "prefix text {\"name\": \"George Washington\", \"num\": 1, \"party\": \"None\"} suffix")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expect(calls[0].inferred);
        try testing.expectEqualStrings("George Washington", calls[0].name);
    }
}

test "parseToolCalls multiple calls" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "func_a", "arguments": {"x": 1}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "func_b", "arguments": {"y": 2}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("func_a", calls[0].name);
    try testing.expectEqualStrings("func_b", calls[1].name);
}

test "parseToolCalls returns null for no tool calls" {
    const allocator = testing.allocator;
    const result = try parseToolCalls(allocator, "Hello, how can I help you?");
    try testing.expect(result == null);
}

// ── DSV4 training-bias tool-call shapes ──
// DSV4-Flash via ds4 emits tool calls in two non-standard variants the
// reference parser previously dropped on the floor:
//   1. Attribute form: `<tool_call name="X">{args}</tool_call>` — name as an
//      XML attribute, body is JUST the args object.
//   2. Mismatched closing tag: `<tool_call>{full json}</tool_request>` — open
//      tag is `tool_call`, close is `tool_request`. Hermes-style trained on a
//      mixed corpus.
// These tests are the regression guard. Without the fix they fail because
// the parser hard-matched on `<tool_call>…</tool_call>` exactly.

test "parseToolCalls DSV4 attribute form: <tool_call name=\"X\">{args}</tool_call>" {
    const allocator = testing.allocator;
    const text = "<tool_call name=\"webSearch\">{\"query\":\"top open source inference applications\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("top open source inference applications", parsed.value.object.get("query").?.string);
}

test "parseToolCalls DSV4 mismatched close: <tool_call>{json}</tool_request>" {
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "mkdir -p prisma src/lib"}}
        \\</tool_request>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mkdir -p prisma src/lib", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 attribute + mismatched close combined" {
    // Worst-case combo observed from DSV4: attribute-style open, body is
    // just the args object (no top-level "name" key), and the closing tag
    // is `</tool_request>` instead of `</tool_call>`.
    const allocator = testing.allocator;
    const text = "<tool_call name=\"shell\">{\"command\":\"ls /tmp\"}</tool_request>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls /tmp", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Hermes double-brace body: <tool_call>{{json}}</tool_call>" {
    // Some models (seen on small GGUF instruct models echoing a Jinja `{{ }}`
    // example) wrap the args object in an extra brace layer. The body must still
    // parse into a tool call rather than leaking through as raw content.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>
        \\{{"name": "get_weather", "arguments": {"city": "Paris"}}}
        \\</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Paris", parsed.value.object.get("city").?.string);
}

test "parseToolCalls DSV4 attribute single-quoted: <tool_call name='X'>" {
    // Some captures used single quotes around the attribute value. Same
    // semantics — accept both quote styles.
    const allocator = testing.allocator;
    const text = "<tool_call name='browse'>{\"action\":\"navigate\",\"url\":\"https://example.com\"}</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
}

test "parseToolCalls DSV4 plural-tag form: <tool_calls>{json}</tool_calls>" {
    // Real capture from DSV4-Flash in MLX Core ChatView:
    //   <tool_calls>
    //   {"name": "shell", "arguments": {"command": "df -h /"}}
    //   </tool_calls>
    // Plural open + plural close, single JSON object inside. Without the
    // fix the parser hard-matches `<tool_call` (the suffix `s>` makes the
    // `>`-scan walk past the tool name), the body never extracts, and the
    // raw XML leaks into the chat bubble.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\{"name": "shell", "arguments": {"command": "df -h /"}}
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("df -h /", parsed.value.object.get("command").?.string);
}

test "parseToolCalls DSV4 plural-tag with attribute: <tool_calls name=\"X\">{args}</tool_calls>" {
    const allocator = testing.allocator;
    const text = "<tool_calls name=\"webSearch\">{\"query\":\"hi\"}</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
}

test "parseToolCalls DSV4 nested <tool name=\"X\"> inside <tool_calls> wrapper" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="webSearch">{"query": "top 10 ..."}</tool_calls>
    //   <tool name="webSearch">{"query": "best ..."}</tool_calls>
    //   </tool_calls>
    // Two inner calls, each opening with `<tool name="X">` and closing with
    // `</tool_calls>` (the model conflates open and close tag names). The
    // outer `<tool_calls>` wrapper has no content of its own — it's a
    // decorative parent.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="webSearch">{"query": "top 10 open source AI inference engines 2024 2025"}</tool_calls>
        \\<tool name="webSearch">{"query": "best open source AI inference engines vllm tensorrt mlx comparison"}</tool_calls>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Both inner calls must be recovered.
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("webSearch", calls[0].name);
    try testing.expectEqualStrings("webSearch", calls[1].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expect(std.mem.indexOf(u8, a0.value.object.get("query").?.string, "top 10") != null);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expect(std.mem.indexOf(u8, a1.value.object.get("query").?.string, "best") != null);
}

test "parseToolCalls bare <tool>{json}</tool>" {
    // Hypothetical minimal form — `<tool>` open with no attribute, JSON
    // body carries the name. Belt-and-suspenders for future model output.
    const allocator = testing.allocator;
    const text = "<tool>{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}</tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls DSV4 <parameters> wrapper inside <tool name=\"X\">" {
    // Real capture from DSV4-Flash:
    //   <tool_calls>
    //   <tool name="browse"><parameters>{"action":"navigate","url":"…"}</parameters></tool>
    //   …
    //   </tool_calls>
    // The args object is wrapped in <parameters>…</parameters> before being
    // placed inside the <tool> body. Strip that wrapper before JSON-parsing.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.com"}</parameters></tool>
        \\<tool name="browse"><parameters>{"action":"navigate","url":"https://example.org"}</parameters></tool>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
    try testing.expectEqualStrings("https://example.com", a0.value.object.get("url").?.string);
    const a1 = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer a1.deinit();
    try testing.expectEqualStrings("https://example.org", a1.value.object.get("url").?.string);
}

test "parseToolCalls DSV4 trailing extra closing brace in args body" {
    // DSV4 occasionally emits an extra `}` after the args object — likely a
    // training-data artifact. Be tolerant: find the first balanced JSON
    // object and ignore trailing garbage.
    const allocator = testing.allocator;
    const text =
        "<tool name=\"browse\"><parameters>{\"action\":\"navigate\",\"url\":\"https://dev.to/agdex_ai/post\"}}</parameters></tool>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const a0 = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer a0.deinit();
    try testing.expectEqualStrings("navigate", a0.value.object.get("action").?.string);
}

test "parseToolCalls with think block" {
    const allocator = testing.allocator;
    const text = "<think>reasoning</think>\n<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"test\"}}\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("search", calls[0].name);
}

test "parseToolCalls raw JSON without tags" {
    const allocator = testing.allocator;
    const text = "{\"name\": \"get_time\", \"arguments\": {}}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_time", calls[0].name);
}

test "parseToolCalls Hermes format" {
    const allocator = testing.allocator;
    const text = "<tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    // Should have {"location":"Tokyo"}
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 format" {
    const allocator = testing.allocator;
    const text = "<|tool_call>call:get_weather{\"location\": \"Tokyo\"}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Tokyo", parsed.value.object.get("location").?.string);
}

test "parseToolCalls Gemma 4 truncated (no closing tag)" {
    const allocator = testing.allocator;
    // Model hit EOS before generating <tool_call|>
    const text = "<|tool_call>call:browse{action:<|\"|>browse<|\"|>,url:<|\"|>https://finance.yahoo.com<|\"|>}";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    // Should have parsed at least the action argument
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("browse", parsed.value.object.get("action").?.string);
}

test "parseToolCalls Gemma 4 truncated mid-value drops the partial value" {
    const allocator = testing.allocator;
    // Model stopped mid-URL: no closing <|"|>, no `}`, no <tool_call|>. POLICY
    // (2026-07-14, aligned with the Hermes truncation rule): a value whose scan
    // runs to end-of-body without its terminator is a truncation FRAGMENT —
    // partial bytes must never ship as a real argument (a client executes them:
    // a fragmentary url browses garbage, a fragmentary content writes a corrupt
    // file "successfully"). The completed pair before it survives, and the tool
    // NAME is always recovered so the client's steering fires.
    const text = "<|tool_call>call:browse{action:<|\"|>navigate<|\"|>,url:<|\"|>https://finance.";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("browse", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("navigate", parsed.value.object.get("action").?.string);
    // The truncated URL must be DROPPED, never shipped as partial bytes.
    try testing.expect(parsed.value.object.get("url") == null);
}

test "parseToolCalls Gemma 4 loop-stop truncated content dropped, name recovered (php.html capture)" {
    const allocator = testing.allocator;
    // Live 2026-07-14 (pi → gemma-4-26B-A4B-it-qat-4bit, plang/php.html): the
    // model repetition-looped INSIDE the write call's content string and the
    // scheduler's degenerate-tail-loop guard cut generation mid-word — the
    // buffered call arrived with NO closing delimiter, NO `}`, NO <tool_call|>,
    // and `path` never emitted at all. Pre-fix the salvage shipped
    // {"content":"<1.1 KB of loop garbage>"}; pi rejected the call on the
    // missing path and echoed the garbage back into the model's context.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html>\n<html lang=\"en\">\n<head>\n    <title>PHP</title>\n</head>\n<body>\n    <p>PHP is a widely-used general-purpose scripting language. It is a server-side scripting language, server-side scripting language, server-side scripting language, server-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("content") == null);
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 bare rich value truncated at end-of-body is dropped" {
    const allocator = testing.allocator;
    // Both <|"|> delimiters dropped AND the generation truncated (no `}`, no
    // close tag anywhere in the rest): the run-to-end scan must not ship the
    // fragment as content.
    const text = "<|tool_call>call:write{content:<!DOCTYPE html>\n<h1>Hi</h1>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 truncated edits container dropped, complete path kept" {
    const allocator = testing.allocator;
    // Cut INSIDE a container arg (pi-style edits array): the list is a fragment
    // — more edits were coming — and applying a partial edit list is the same
    // fragmentary-work hazard as partial content. The whole container drops;
    // the completed scalar before it survives.
    const text = "<|tool_call>call:edit{path:<|\"|>a.sh<|\"|>,edits:[{oldText:<|\"|>x<|\"|>,newText:<|\"|>y<|\"|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.sh", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("edits") == null);
}

test "parseToolCalls Gemma 4 EOS after a complete pair keeps the pair (no over-drop)" {
    const allocator = testing.allocator;
    // Truncation landing BETWEEN pairs (after a completed value + comma) must
    // not over-drop: only the value the cut landed INSIDE is a fragment.
    const text = "<|tool_call>call:write{path:<|\"|>a.html<|\"|>,";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.html", parsed.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 EOS right after the key colon drops the key" {
    const allocator = testing.allocator;
    // `path:` then nothing — pre-fix this shipped {"path":""} and a client
    // would create a file with an EMPTY name (or reject on a bogus value).
    // An absent value is a fragment; drop the key, recover the name.
    const text = "<|tool_call>call:write{path:";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("path") == null);
}

test "parseToolCalls Gemma 4 JSON-style truncated string value is dropped" {
    const allocator = testing.allocator;
    // Same truncation class through the JSON-quoted spelling: the closing `"`
    // never arrived. Pre-fix the verbatim copy emitted an UNCLOSED JSON string
    // (invalid args, caught only by the {}-fallback safety net); the complete
    // first pair must survive and the fragment must drop.
    const text = "<|tool_call>call:write{\"path\":\"plang/php.html\",\"content\":\"<!DOCTYPE html>\n<p>cut mid-";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("plang/php.html", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls Gemma 4 unterminated string must not swallow the closing brace (pi write regression)" {
    const allocator = testing.allocator;
    // Gemma 4 12B emitted a write call whose LAST string value was missing
    // its closing <|"|> delimiter and carried a stray markdown backtick
    // before the args object's `}`. The unterminated-string scan ran to end
    // of body, so the parsed path was literally "mlx_pi1.html`}" — and pi
    // created a file with that name on disk.
    const text = "<|tool_call>call:write{content:<|\"|><!DOCTYPE html><html></html><|\"|>,path:<|\"|>mlx_pi1.html`}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx_pi1.html", parsed.value.object.get("path").?.string);

    // Plain missing-delimiter variant (no backtick): same brace exclusion.
    const text2 = "<|tool_call>call:write{content:<|\"|>X<|\"|>,path:<|\"|>out.html}<tool_call|>";
    const calls2 = (try parseToolCalls(allocator, text2)).?;
    defer {
        for (calls2) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls2);
    }
    const parsed2 = try std.json.parseFromSlice(std.json.Value, allocator, calls2[0].arguments, .{});
    defer parsed2.deinit();
    try testing.expectEqualStrings("out.html", parsed2.value.object.get("path").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with custom delimiters" {
    const allocator = testing.allocator;
    // Model mixes JSON-style quoted keys with <|"|> delimiters — JSON parse fails,
    // convertGemma4ArgsToJson must strip quotes from keys to produce valid JSON.
    const text =
        \\<|tool_call>call:shell{"command":<|"|>ls -la<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 quoted keys with bare values" {
    const allocator = testing.allocator;
    // Model uses quoted keys but bare (non-JSON, non-delimited) values
    const text = "<|tool_call>call:shell{\"command\":ls -la}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

test "parseToolCalls Gemma 4 dropped OPENING delimiter on big content (E4B live)" {
    const allocator = testing.allocator;
    // Verbatim shape captured from gemma-4-e4b-it-4bit writing a full HTML page
    // (test_tool_matrix_small.sh): the model dropped the OPENING <|"|> on
    // `content` but kept the CLOSING one, so the bare-value scan cut content at
    // the FIRST comma (inside `<meta ... content="width=device-width, ...">`)
    // and shredded the rest of the markup into bogus keys → invalid args. The
    // closing <|"|> (followed by `,path`) is the real boundary.
    const text = "<|tool_call>call:write_file{content:<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html><|\"|>,path:<|\"|>mars.html<|\"|>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mars.html", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings(
        "<!DOCTYPE html>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n<style>body{margin:0}</style>\n</html>",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls Gemma 4 dropped BOTH delimiters on single content field" {
    const allocator = testing.allocator;
    // Both delimiters dropped, content is the only/last field → run to the
    // object's closing brace.
    const text = "<|tool_call>call:write_file{content:<h1>Hi</h1>\n<p>a, b, c</p>}<tool_call|>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("<h1>Hi</h1>\n<p>a, b, c</p>", parsed.value.object.get("content").?.string);
}

test "convertGemma4ArgsToJson nested braces in value" {
    const allocator = testing.allocator;
    // Content value contains JSON-like structures (e.g., JavaScript code with objects)
    const input = "{path:<|\"|>server.js<|\"|>,content:<|\"|>const x = {a: 1, b: {c: 2}};<|\"|>}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    // Verify it produces valid JSON
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("server.js", parsed.value.object.get("path").?.string);
    const content = parsed.value.object.get("content").?.string;
    try testing.expect(std.mem.indexOf(u8, content, "{a: 1") != null);
}

test "convertGemma4ArgsToJson bare array value" {
    const allocator = testing.allocator;
    // Bare array value should be preserved via brace-matching
    const input = "{stops:[\"Rome\",\"Venice\",\"Athens\"],price:1200}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.get("stops").?.array.items.len);
    try testing.expectEqual(@as(i64, 1200), parsed.value.object.get("price").?.integer);
}

test "convertGemma4ArgsToJson bare nested object value" {
    const allocator = testing.allocator;
    // Bare nested object should be preserved via brace-matching
    const input = "{name:test,config:{\"port\":3000,\"host\":\"localhost\"}}";
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const config = parsed.value.object.get("config").?.object;
    try testing.expectEqual(@as(i64, 3000), config.get("port").?.integer);
}

// Issue #16: nested objects/arrays that themselves use Gemma 4's custom
// `<|"|>`-delimited string format (or bare keys) must be converted recursively,
// not passed through verbatim.
test "convertGemma4ArgsToJson issue#16 nested array of custom strings" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array:[<|"|>foo<|"|>,<|"|>bar<|"|>]}
    ;
    const expected =
        \\{"nested_array":["foo","bar"]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object custom format" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object:{foo:<|"|>bar<|"|>}}
    ;
    const expected =
        \\{"nested_object":{"foo":"bar"}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested object inside array" {
    const allocator = testing.allocator;
    const input =
        \\{nested_object_in_array:[{foo:<|"|>bar<|"|>}]}
    ;
    const expected =
        \\{"nested_object_in_array":[{"foo":"bar"}]}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

test "convertGemma4ArgsToJson issue#16 nested array inside object" {
    const allocator = testing.allocator;
    const input =
        \\{nested_array_in_object:{foo:[<|"|>bar<|"|>]}}
    ;
    const expected =
        \\{"nested_array_in_object":{"foo":["bar"]}}
    ;
    const result = convertGemma4ArgsToJson(allocator, input, false).?;
    defer allocator.free(result);
    try testing.expectEqualStrings(expected, result);
}

// Issue #16, real capture: this is the verbatim output of gemma-4-e4b-it-8bit
// for a tool with nested-object + nested-array params. The model nests its
// custom `<|"|>`/bare-key format, so the client must still receive valid,
// fully-converted JSON arguments (no surviving <|"|> delimiters / bare keys).
test "parseToolCalls Gemma 4 nested object + array args (issue#16 real capture)" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:send_notification{message:<|"|>Deploy complete<|"|>,metadata:{priority:<|"|>high<|"|>,tags:[<|"|>ci<|"|>,<|"|>release<|"|>]},recipients:[<|"|>alice<|"|>,<|"|>bob<|"|>]}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("send_notification", calls[0].name);
    // The arguments must be valid JSON (would fail to parse with raw <|"|> in them).
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("Deploy complete", parsed.value.object.get("message").?.string);
    const metadata = parsed.value.object.get("metadata").?.object;
    try testing.expectEqualStrings("high", metadata.get("priority").?.string);
    const tags = metadata.get("tags").?.array;
    try testing.expectEqual(@as(usize, 2), tags.items.len);
    try testing.expectEqualStrings("ci", tags.items[0].string);
    try testing.expectEqualStrings("release", tags.items[1].string);
    const recipients = parsed.value.object.get("recipients").?.array;
    try testing.expectEqual(@as(usize, 2), recipients.items.len);
    try testing.expectEqualStrings("alice", recipients.items[0].string);
    try testing.expectEqualStrings("bob", recipients.items[1].string);
}

// Gemma 4 12B leaks a trailing ':' into the function name (`call:shell:{…}`),
// producing an unresolvable tool name `shell:`. The parser must strip it so the
// client resolves a real tool instead of looping on "Unknown tool 'shell:'".
test "parseToolCalls Gemma 4 strips trailing colon from tool name" {
    const allocator = testing.allocator;
    const text =
        \\<|tool_call>call:shell:{"command":<|"|>ls<|"|>}<tool_call|>
    ;
    const calls = (try parseToolCalls(allocator, text)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
}

test "sanitizeToolName strips trailing colons and whitespace" {
    try testing.expectEqualStrings("shell", sanitizeToolName("shell:"));
    try testing.expectEqualStrings("shell", sanitizeToolName("  shell  "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell : "));
    try testing.expectEqualStrings("shell", sanitizeToolName("shell::"));
    try testing.expectEqualStrings("cwd", sanitizeToolName("cwd"));
    try testing.expectEqualStrings("", sanitizeToolName(":"));
}

test "isJsonLiteral" {
    try testing.expect(isJsonLiteral("true"));
    try testing.expect(isJsonLiteral("false"));
    try testing.expect(isJsonLiteral("null"));
    // Objects/arrays are now handled by brace-matching in convertGemma4ArgsToJson, not isJsonLiteral
    try testing.expect(!isJsonLiteral("{\"key\":1}"));
    try testing.expect(!isJsonLiteral("[1,2,3]"));
    try testing.expect(isJsonLiteral("42"));
    try testing.expect(isJsonLiteral("3.14"));
    try testing.expect(!isJsonLiteral("hello"));
    try testing.expect(!isJsonLiteral(""));
}

// ── DSV4 agentic-flow compositions (round-trip + multi-turn + edge cases) ──

test "DSV4 agentic round-trip: model-generated <tool_call> parses + replays" {
    // Simulates the full agent loop step:
    //   1. Model emits a `<tool_call>` block in its generated content.
    //   2. parseToolCalls extracts the structured ToolCall(s).
    //   3. The agent builds the next prompt with role:assistant(tool_calls=...) +
    //      role:tool(content=result).
    //   4. synthesizeToolFallbackMessages rewrites both for the DSV4 template
    //      (assistant tool_calls → <tool_call> block, tool → user
    //      <tool_response>).
    //   5. The same call should round-trip: the rewritten assistant content
    //      should contain a parseable <tool_call> matching the original.
    const allocator = testing.allocator;

    // 1. Pretend the model wrote this:
    const generated =
        \\<tool_call>
        \\{"name": "shell", "arguments": {"command": "date"}}
        \\</tool_call>
    ;
    const parsed_calls = (try parseToolCalls(allocator, generated)).?;
    defer {
        for (parsed_calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(parsed_calls);
    }
    try testing.expectEqual(@as(usize, 1), parsed_calls.len);
    try testing.expectEqualStrings("shell", parsed_calls[0].name);

    // 2. Agent builds the next-turn history with the parsed call + the result.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tc = [_]ToolCall{
        .{ .id = "tc_0", .name = parsed_calls[0].name, .arguments = parsed_calls[0].arguments },
    };
    const messages = [_]Message{
        .{ .role = "user", .content = "what is the date?" },
        .{ .role = "assistant", .content = "", .tool_calls = &tc },
        .{ .role = "tool", .content = "Mon May 13 2026", .tool_call_id = "tc_0" },
    };

    // 3. Run through the DSV4 fallback. inject_tool_prompt=true mirrors how
    // renderChatTemplate calls it when the template lacks tool support.
    const tools_json = "[{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}]";
    const out = try synthesizeToolFallbackMessages(a, &messages, tools_json, null, true, true, true);

    // Should produce: system(tool-prompt), user, assistant(<tool_call>), user(<tool_response>).
    try testing.expectEqual(@as(usize, 4), out.len);
    try testing.expectEqualStrings("system", out[0].role);
    try testing.expectEqualStrings("user", out[1].role);
    try testing.expectEqualStrings("assistant", out[2].role);
    try testing.expectEqualStrings("user", out[3].role);

    // 4. The assistant's rewritten content must contain a <tool_call> block
    //    that re-parses back to the same name + arguments. This is the
    //    closed-loop test.
    try testing.expect(out[2].tool_calls == null);
    const reparsed = (try parseToolCalls(allocator, out[2].content)).?;
    defer {
        for (reparsed) |t| {
            allocator.free(t.name);
            allocator.free(t.arguments);
        }
        allocator.free(reparsed);
    }
    try testing.expectEqual(@as(usize, 1), reparsed.len);
    try testing.expectEqualStrings("shell", reparsed[0].name);
    try testing.expect(std.mem.indexOf(u8, reparsed[0].arguments, "date") != null);

    // 5. The tool-result rewrite must wrap the result in <tool_response>...
    try testing.expect(std.mem.indexOf(u8, out[3].content, "<tool_response>") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "Mon May 13 2026") != null);
    try testing.expect(std.mem.indexOf(u8, out[3].content, "</tool_response>") != null);
}

test "parseToolCalls: self-closing <tool name=... arguments=... />" {
    const allocator = testing.allocator;
    // Clean form. We use the literal byte sequence the model actually emits —
    // unescaped `"` inside the attribute value. That's XML-invalid but
    // perfectly JSON-parseable once we snap to the inner `{…}`.
    const text =
        \\<tool name="shell" arguments="{"command": "ls -la"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: DSV4 broken-quote self-closing fragment (regression repro)" {
    // Verbatim capture from the failing turn in test_agent_stop_repro.py.
    // The model opened arguments with `"`, closed with `'`, embedded
    // unescaped `"` inside the JSON, and finished with `'/>`. The parser
    // must still recover the JSON object and produce a single tool call.
    const allocator = testing.allocator;
    const text =
        "\n\n<tool_calls>\n" ++
        "<tool name=\"shell\" arguments=\"{\"command\": \"ls -la && cat package.json 2>/dev/null || echo 'no package.json yet'\"}'/>\n" ++
        "</tool_calls>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    // Args should be the balanced JSON object — starts with `{` and ends `}`.
    try testing.expect(calls[0].arguments.len > 0);
    try testing.expectEqual(@as(u8, '{'), calls[0].arguments[0]);
    try testing.expectEqual(@as(u8, '}'), calls[0].arguments[calls[0].arguments.len - 1]);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"command\"") != null);
}

test "parseToolCalls: multiple self-closing tool tags in one response" {
    const allocator = testing.allocator;
    const text =
        \\<tool name="cwd" arguments="{"path": "src"}"/>
        \\<tool name="listFiles" arguments="{"recursive": "true"}"/>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expectEqualStrings("listFiles", calls[1].name);
}

test "parseToolCalls: canonical <tool_call>JSON</tool_call> still parses" {
    // Regression guard — adding the self-closing branch must not break the
    // canonical Hermes form.
    const allocator = testing.allocator;
    const text =
        \\<tool_call>{"name": "shell", "arguments": {"command": "ls"}}</tool_call>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
}

test "parseToolCalls: empty body + attribute-args + explicit </tool_call> close" {
    // Verbatim capture from the user's failed chat-history.json. The model
    // emitted args via XML attribute and used `></tool_call>` (not `/>`)
    // as the terminator, with empty body. Without explicit handling this
    // shape parses as "empty body" → no JSON to extract → no tool call.
    const allocator = testing.allocator;
    const text =
        \\<tool_calls>
        \\<tool_call name="cwd" arguments="{"path": "/Users/user/.sushi/workspace"}"></tool_call>
        \\</tool_calls>
    ;
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("cwd", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, ".sushi/workspace") != null);
}

test "parseToolCalls: <tool_call>{JSON} truncated before </tool_call>" {
    // The model emitted `<tool_call>` + well-formed args JSON, then hit
    // EOS before completing the close marker. Output looks like:
    //   "\n\n<tool_call>\n{\"name\": \"writeFile\", \"arguments\":{...}}\n</tool_cal"
    // Previously this fell through to the orphan branch and got silently
    // dropped — that's the second failure mode the agent-stops repro
    // surfaced. We snap to a balanced JSON body and parse it.
    const allocator = testing.allocator;
    const text = "\n\n<tool_call>\n" ++
        \\{"name": "writeFile", "arguments": {"path": "vite.config.js", "content": "ok"}}
        ++ "\n</tool_cal";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("writeFile", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"path\"") != null);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "\"content\"") != null);
}

test "parseToolCalls: mismatched </tool_action> close still parses" {
    // Verbatim DSV4-Flash capture (2026-06-10 pi html-ds4 turn 2): opened
    // with <tool_call>, closed with the hallucinated </tool_action>. The
    // edit call must parse; pre-fix it leaked into visible text and the
    // agent executed nothing.
    const allocator = testing.allocator;
    const text = "<tool_call>\n" ++
        \\{"name": "edit", "arguments": {"path":"mlx.html", "edits":[{"oldText": "  </ul>\n</body>", "newText": "  </ul>\n  <button onclick=\"alert('Hello from MLX')\">Click me</button>\n</body>"}]}
        ++ "\n</tool_action>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
}

test "parseToolCalls: fenced JSON array of parallel calls parses all" {
    const allocator = testing.allocator;
    const text = "```json\n[\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris, France\"}},\n  {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo, Japan\"}}\n]\n```";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "Paris, France") != null);
    try testing.expect(std.mem.indexOf(u8, calls[1].arguments, "Tokyo, Japan") != null);
}

test "parseToolCalls: XML-element-TAG form (<tool_NAME> with arg children)" {
    // Verbatim DSV4-Flash capture (2026-06-10 pi html-ds4 turn 2): the tool
    // name rides in the tag itself — <tool_read>, <tool_edit> — with each
    // argument as an XML child element. Nested elements (the <edits> body)
    // stay verbatim as the arg's string value, matching what working models
    // send pi for the same tool.
    const allocator = testing.allocator;
    const text = "Let me read the current file first.\n\n<tool_read>\n<path>mlx.html</path>\n</tool_read>Now I'll add a button:\n\n<tool_edit>\n<path>mlx.html</path>\n<edits>\n  <oldText>    <h1>MLX</h1></oldText>\n  <newText>    <h1>MLX</h1>\n    <button onclick=\"alert('hi')\">Say Hello</button></newText>\n</edits>\n</tool_edit>";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("read", calls[0].name);
    try testing.expect(std.mem.indexOf(u8, calls[0].arguments, "mlx.html") != null);
    try testing.expectEqualStrings("edit", calls[1].name);
    // Args must be valid JSON with the nested XML preserved as a string value.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[1].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx.html", parsed.value.object.get("path").?.string);
    const edits = parsed.value.object.get("edits").?.string;
    try testing.expect(std.mem.indexOf(u8, edits, "<oldText>") != null);
    try testing.expect(std.mem.indexOf(u8, edits, "Say Hello") != null);
}

test "parseToolCalls: XML-element-TAG form with JSON args body" {
    // Verbatim-shape DSV4-Flash capture (2026-06-10 pi html-ds4, second
    // sampling): same name-in-tag form but the body is a bare JSON args
    // object instead of XML elements — `<tool_write>\n{…}\n</tool_write>`.
    const allocator = testing.allocator;
    const text = "Here's the HTML page:\n\n<tool_write>\n{\"path\": \"/tmp/ws/mlx.html\", \"content\": \"<!DOCTYPE html>\\n<html lang=\\\"en\\\">\\n<body>\\n</body>\\n</html>\"}\n</tool_write>\n\npage ready";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/ws/mlx.html", parsed.value.object.get("path").?.string);
    try testing.expect(std.mem.indexOf(u8, parsed.value.object.get("content").?.string, "<!DOCTYPE html>") != null);
}

test "parseToolCalls: <tool_NAME> with prose body containing braces is NOT a tool call" {
    // Prose with incidental {braces} inside an unknown tool-ish tag must not
    // produce arguments — the JSON-body variant requires the body to BE the
    // args object, not merely contain one.
    const allocator = testing.allocator;
    const text = "<tool_summary>The config {a: 1} was applied.</tool_summary>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <tool_output> with element children is still NOT a tool call" {
    // A hallucinated result tag can carry element-shaped content (e.g. an
    // echoed HTML fragment). Result-ish names are denylisted outright.
    const allocator = testing.allocator;
    const text = "<tool_output>\n<status>ok</status>\n</tool_output>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: XML-element form without a tool_name is NOT a tool call" {
    // Prose-ish markup inside a <tool_calls> wrapper must not half-execute.
    const allocator = testing.allocator;
    const text = "<tool_calls>\n<note>this has no tool name element</note>\n</tool_calls>";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: prose list array is NOT a tool call" {
    // All-or-nothing: an array whose elements aren't {name, arguments}
    // objects must pass through as text.
    const allocator = testing.allocator;
    const text = "[\"apples\", \"oranges\", \"pears\"]";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "parseToolCalls: <toolkit> is not a tool tag" {
    // The self-closing scan must not false-fire on HTML-ish prose. `<toolkit>`
    // has `<tool` as a prefix but the terminator (`k`) isn't valid, so the
    // outer scan rejects it before parseSelfClosingToolTag even runs.
    const allocator = testing.allocator;
    const text = "Here's a doc about <toolkit> and <toolbar>. No actual tool call.";
    const calls = try parseToolCalls(allocator, text);
    try testing.expect(calls == null);
}

test "convertGemma4 dedups a repeated key + escapes a key with special chars (valid JSON)" {
    // Same class as the parseHermesToolCall dup-key / raw-name bugs, but in the
    // Gemma `call:name{...}` converter: a repeated key produced a DUPLICATE JSON
    // key (std.json → error.DuplicateField), and a key with a `"` was appended
    // raw → invalid JSON. Both must yield a valid JSON object.
    const allocator = testing.allocator;
    // Duplicate key.
    {
        const raw = "<|tool_call>call:shell{command:<|\"|>ls<|\"|>,command:<|\"|>pwd<|\"|>}<tool_call|>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expect(parsed.value == .object);
        try testing.expect(parsed.value.object.get("command") != null);
    }
    // Key containing a double-quote must not break the JSON.
    {
        const raw = "<|tool_call>call:shell{a\"b:<|\"|>v<|\"|>}<tool_call|>";
        const calls = try parseToolCalls(allocator, raw);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, cs[0].arguments, .{});
            defer parsed.deinit();
            try testing.expect(parsed.value == .object);
        }
    }
}

test "parseXmlElementToolCall is already dup-key + escape safe (characterization)" {
    // The DSV4 XML-element path builds args via std.json.ObjectMap.put + Stringify,
    // which dedups keys (last wins) and escapes values — so it is structurally
    // immune to the dup-key/raw-interpolation class that bit parseHermesToolCall
    // and convertGemma4Object. This characterization test pins that guarantee.
    const allocator = testing.allocator;
    const raw = "<tool_calls>\n<tool_name>shell</tool_name>\n<command>echo \"a\"</command>\n<command>echo \"b\"</command>\n</tool_calls>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    // Valid JSON despite the duplicate <command> and the inner quotes.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .object);
    try testing.expect(parsed.value.object.get("command") != null);
}

test "parseToolCalls: NO path emits invalid JSON args (final safety net)" {
    // Bullet-proof invariant: whatever a converter builds, the arguments a client
    // receives are ALWAYS valid JSON. convertGemma4Value copies a JSON-style
    // string verbatim, so a bad escape (`\q`) would otherwise emit invalid JSON.
    const allocator = testing.allocator;
    const cases = [_][]const u8{
        // Gemma regular-JSON-string value with an INVALID escape.
        "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>",
        // Gemma custom-string is fine, but mix a bad JSON string sibling.
        "<|tool_call>call:write{path:<|\"|>ok<|\"|>,x:\"y\\z\"}<tool_call|>",
    };
    for (cases) |text| {
        const calls = try parseToolCalls(allocator, text);
        defer if (calls) |cs| {
            for (cs) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(cs);
        };
        if (calls) |cs| {
            for (cs) |tc| {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, tc.arguments, .{}) catch {
                    std.debug.print("\nINVALID JSON args emitted: {s}\n", .{tc.arguments});
                    return error.InvalidJsonEmitted;
                };
                parsed.deinit();
            }
        }
    }
    // The `\q` case is REPAIRABLE (looseRepair doubles the bad backslash), so the
    // path survives as the literal `a\qb` — the safety net repairs, only falling
    // back to `{}` when even tolerant repair fails.
    {
        const calls = (try parseToolCalls(allocator, "<|tool_call>call:write{path:\"a\\qb\"}<tool_call|>")).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings("a\\qb", parsed.value.object.get("path").?.string);
    }
}

// ── GLM arg_key/arg_value tag-format tool calls ──

test "parseToolCalls arg_key/arg_value format (single call, hostile value bytes)" {
    const raw = "<tool_call>write_file\n" ++
        "<arg_key>path</arg_key>\n" ++
        "<arg_value>jfk.txt</arg_value>\n" ++
        "<arg_key>content</arg_key>\n" ++
        "<arg_value>Hello \"world\"\nline2\ttabbed</arg_value>\n" ++
        "</tool_call>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expect(!calls[0].inferred);
    // Arguments must be VALID JSON with both keys, hostile bytes escaped.
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("jfk.txt", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("Hello \"world\"\nline2\ttabbed", parsed.value.object.get("content").?.string);
}

test "parseToolCalls arg_key: duplicate arg key — first wins, args stay valid JSON" {
    const raw = "<tool_call>edit\n" ++
        "<arg_key>path</arg_key>\n<arg_value>a.txt</arg_value>\n" ++
        "<arg_key>path</arg_key>\n<arg_value>b.txt</arg_value>\n" ++
        "</tool_call>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "parseToolCalls arg_key: truncated mid-value recovers name + closed args only" {
    // max_tokens cut the generation inside the second value — recover the call
    // with the CLOSED pair only; never salvage the partial content (the
    // truncated-opener class: a half-written file is worse than a retry).
    const raw = "<tool_call>write_file\n" ++
        "<arg_key>path</arg_key>\n<arg_value>x.txt</arg_value>\n" ++
        "<arg_key>content</arg_key>\n<arg_value>a very long novel that never clos";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("x.txt", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls arg_key: mangled name-close + key-close still recovers name AND args" {
    // A weak model closes the NAME with </arg_value> and closes the arg KEY
    // block with </arg_value> instead of </arg_key> — the VALUE block is
    // well-formed. A strict parser bails and the whole call LEAKS as content
    // (finish_reason stop), so the client sees bash({}) and loops. The
    // corruption is small and regular, so recover the FULL call.
    const raw = "<tool_call>bash</arg_value>\n" ++
        "<arg_key>command</arg_value>\n" ++
        "<arg_value>ls -la</arg_value>\n" ++
        "</tool_call>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls -la", parsed.value.object.get("command").?.string);
}

// ── MiniCPM5 V3 XML (`<function name="X"><param name="K">V</param></function>`) ──

test "parseToolCalls minicpm5: git log -1" {
    const raw = "<function name=\"shell\">\n  <param name=\"command\">git log -1</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("git log -1", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: multiple arguments in one call" {
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">notes.txt</param>\n" ++
        "  <param name=\"content\">hello world</param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
    try testing.expectEqualStrings("notes.txt", parsed.value.object.get("path").?.string);
    try testing.expectEqualStrings("hello world", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: CDATA payload containing a literal </param> is not truncated" {
    // A write_file call documenting this very XML format — the CDATA payload
    // legitimately contains the substring `</param>`. A naive substring
    // search for the closing tag would cut the value there; CDATA exists
    // precisely so this content is carried verbatim.
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">docs.txt</param>\n" ++
        "  <param name=\"content\"><![CDATA[Example: <param name=\"x\">y</param> then more text]]></param>\n" ++
        "</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(
        "Example: <param name=\"x\">y</param> then more text",
        parsed.value.object.get("content").?.string,
    );
}

test "parseToolCalls minicpm5: CDATA payload containing a literal </function> is not truncated, and a second real call still parses" {
    const raw = "<function name=\"write_file\">\n" ++
        "  <param name=\"path\">docs.txt</param>\n" ++
        "  <param name=\"content\"><![CDATA[Close a call with </function> like this]]></param>\n" ++
        "</function>\n" ++
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 2), calls.len);
    const first = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer first.deinit();
    try testing.expectEqualStrings("Close a call with </function> like this", first.value.object.get("content").?.string);
    try testing.expectEqualStrings("shell", calls[1].name);
    const second = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[1].arguments, .{});
    defer second.deinit();
    try testing.expectEqualStrings("pwd", second.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: missing param is never fabricated" {
    const raw = "<function name=\"write\">\n  <param name=\"path\">notes.txt</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 1), parsed.value.object.count());
    try testing.expect(parsed.value.object.get("content") == null);
}

test "parseToolCalls minicpm5: prose before and after a call, and function-like false positives" {
    const raw = "Sure, let me check the working directory.\n" ++
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>\n" ++
        "Done — see the result above.";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);

    // `<functional>`/`<function-like>` are prose, not the attribute-quoted
    // MiniCPM5 form (no whitespace right after `<function`) — no call.
    const prose = "Wrap it in a <functional> or <function-like> block — just prose, no call here.";
    try testing.expect(try parseToolCalls(testing.allocator, prose) == null);
}

test "parseToolCalls minicpm5: no tool call in plain prose" {
    const raw = "The `shell` function name attribute isn't used here at all — just chatting.";
    try testing.expect(try parseToolCalls(testing.allocator, raw) == null);
}

test "parseToolCalls minicpm5: existing bare Hermes <function=...> form is unaffected" {
    const raw = "<function=shell><parameter=command>pwd</parameter></function>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer {
        for (calls) |tc| {
            testing.allocator.free(tc.name);
            testing.allocator.free(tc.arguments);
        }
        testing.allocator.free(calls);
    }
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("shell", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("pwd", parsed.value.object.get("command").?.string);
}

test "parseToolCalls minicpm5: adversarial malformed input never hangs or crashes" {
    // 200 back-to-back unterminated openers — the scan must advance past each
    // one rather than re-finding the same offset forever. Built with a
    // comptime loop because Zig 0.17 no longer has the `**` repeat operator.
    const many_openers = comptime blk: {
        var s: []const u8 = "";
        for (0..200) |_| s = s ++ "<function name=\"a\">";
        break :blk s;
    };
    const inputs = [_][]const u8{
        "<function",
        "<function ",
        "<function name=",
        "<function name=\"",
        "<function name=\"a\">",
        "<function name=\"a\"><param",
        "<function name=\"a\"><param name=",
        "<function name=\"a\"><param name=\"\"></param></function>",
        "<function name=\"\"></function>",
        "<function><function><function>",
        many_openers,
    };
    for (inputs) |raw| {
        const calls = try parseToolCalls(testing.allocator, raw);
        if (calls) |cs| {
            for (cs) |tc| {
                testing.allocator.free(tc.name);
                testing.allocator.free(tc.arguments);
            }
            testing.allocator.free(cs);
        }
    }
}

test "parseToolCalls minicpm5: the three <function arms stay disjoint" {
    // MiniCPM5 and the two equals-sign Hermes paths all key on the SAME
    // `<function` marker and are discriminated ONLY by the byte after it
    // (`=` vs whitespace). Upstream's standing rule — "a `<tool_call>` body
    // carrying `<function=` is the XML dialect and is read FIRST (the qwen 3.5+
    // template mandates it)" — means the new arm must not be able to steal
    // those bytes. This pins the ordering against the real arm chain rather
    // than assuming it: the MiniCPM5 arm is gated on `calls.items.len == 0` and
    // sits after both Hermes paths, so it only ever sees text they declined.
    const allocator = testing.allocator;
    const Case = struct { text: []const u8, name: []const u8, key: []const u8, value: []const u8 };
    const cases = [_]Case{
        // 1. WRAPPED `<function=` — the <tool_call> arm owns it.
        .{
            .text = "<tool_call><function=shell><parameter=command>ls -la</parameter></function></tool_call>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
        // 2. BARE `<function=` + `<parameter=` — the bare-Hermes arm owns it.
        .{
            .text = "<function=shell><parameter=command>ls -la</parameter></function>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
        // 3. `<function name="…">` — only this one reaches the MiniCPM5 arm.
        .{
            .text = "<function name=\"shell\">\n  <param name=\"command\">ls -la</param>\n</function>",
            .name = "shell",
            .key = "command",
            .value = "ls -la",
        },
    };
    for (cases) |c| {
        const calls = (try parseToolCalls(allocator, c.text)) orelse {
            std.debug.print("no call parsed for: '{s}'\n", .{c.text});
            return error.ExpectedToolCall;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try testing.expectEqual(@as(usize, 1), calls.len);
        try testing.expectEqualStrings(c.name, calls[0].name);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer parsed.deinit();
        try testing.expectEqualStrings(c.value, parsed.value.object.get(c.key).?.string);
    }
}

test "streamShouldBufferForTools: the gate is a strict SUPERSET of parseMiniCpm5ToolCalls" {
    // The load-bearing invariant, checked by DERIVATION rather than by a
    // hand-listed set: for every text the parser turns into a real call, every
    // prefix from the marker onward must hold. Any prefix that flushed would be
    // content already on the wire (SSE is append-only) followed by the same call
    // arriving properly at end of stream.
    const accepted = [_][]const u8{
        "<function name=\"shell\">\n  <param name=\"command\">pwd</param>\n</function>",
        "<function name=\"get_weather\">\n  <param name=\"city\">Paris</param>\n  <param name=\"unit\">c</param>\n</function>",
        "<function foo name=\"x\"><param name=\"a\">1</param></function>",
    };
    for (accepted) |text| {
        const calls = (try parseToolCalls(testing.allocator, text)) orelse {
            std.debug.print("fixture is not parser-accepted: '{s}'\n", .{text});
            return error.ExpectedToolCall;
        };
        defer {
            for (calls) |tc| {
                testing.allocator.free(tc.name);
                testing.allocator.free(tc.arguments);
            }
            testing.allocator.free(calls);
        }
        const start = std.mem.indexOf(u8, text, "<function").?;
        var end = start + 2; // shortest rung the ladder covers
        while (end <= text.len) : (end += 1) {
            testing.expect(streamShouldBufferForTools(text[start..end])) catch |e| {
                std.debug.print("gate flushed a prefix of an ACCEPTED call: '{s}'\n", .{text[start..end]});
                return e;
            };
        }
    }
}

test "parseToolCalls qwen: a parameter value carrying </parameter> is not emptied" {
    // Bar: the value ends at the close tag that actually ends it, byte-exact.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nsee </parameter> in the docs\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("see </parameter> in the docs", parsed.value.object.get("content").?.string);
}

test "parseToolCalls qwen: a parameter value carrying </tool_call> is not dropped" {
    // Bar: a required parameter survives a value that spells the wrapper close.
    const allocator = testing.allocator;
    const raw =
        "<tool_call>\n<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n" ++
        "<parameter=content>\nemit </tool_call> here\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("emit </tool_call> here", parsed.value.object.get("content").?.string);
}

test "parseToolCalls qwen: a parameter value types from JSON's number grammar, not parseFloat" {
    // Bar: only a spelling JSON itself calls a number stays unquoted.
    const allocator = testing.allocator;
    const Case = struct { spelling: []const u8, number: bool };
    const cases = [_]Case{
        .{ .spelling = "Infinity", .number = false },
        .{ .spelling = "nan", .number = false },
        .{ .spelling = "+1", .number = false },
        .{ .spelling = "0755", .number = false },
        .{ .spelling = ".5", .number = false },
        .{ .spelling = "5.", .number = false },
        .{ .spelling = "1.5", .number = true },
        .{ .spelling = "-2", .number = true },
        .{ .spelling = "3e5", .number = true },
        .{ .spelling = "0", .number = true },
    };
    for (cases) |c| {
        const raw = try std.mem.concat(allocator, u8, &.{
            "<tool_call>\n<function=edit>\n<parameter=mode>\n", c.spelling, "\n</parameter>\n</function>\n</tool_call>",
        });
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)) orelse return error.NoToolCall;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{}) catch {
            std.debug.print("\n{s}: args not valid JSON: {s}\n", .{ c.spelling, calls[0].arguments });
            return error.ArgsNotValidJson;
        };
        defer parsed.deinit();
        const v = parsed.value.object.get("mode") orelse return error.MissingParam;
        if (c.number) {
            if (v == .string) {
                std.debug.print("\n{s}: shipped as a string\n", .{c.spelling});
                return error.NumberQuoted;
            }
        } else {
            if (v != .string) {
                std.debug.print("\n{s}: shipped unquoted as {s}\n", .{ c.spelling, @tagName(v) });
                return error.NonNumberUnquoted;
            }
            try testing.expectEqualStrings(c.spelling, v.string);
        }
    }
}

fn dsv41Replay(a: std.mem.Allocator, tok: *const Tokenizer, tools: ?[]const u8, thinking: bool, raw: []const u8) !Dsv41Replay {
    var cfg = ChatConfig{ .chat_template = @embedFile("fixtures/dsv41_chat_template.jinja"), .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = a, .image_placeholder = dsv41_image_atom };
    const first = [_]Message{.{ .role = "user", .content = "q" }};
    const p1_text = try renderChatTemplate(a, &first, &cfg, tools, null, thinking, "high", false);
    defer a.free(p1_text);
    const p1 = try tok.encode(a, p1_text);
    defer a.free(p1);
    const gen = try tok.encode(a, raw);
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
    const p2_text = try renderChatTemplate(a, &second, &cfg, tools, null, thinking, "high", false);
    defer a.free(p2_text);
    const p2 = try tok.encode(a, p2_text);
    defer a.free(p2);

    const keep = p1.len <= p2.len and std.mem.eql(u32, p1, p2[0..p1.len]);
    var reused: usize = 0;
    if (keep) {
        const tail_ids = p2[p1.len..];
        while (reused < gen.len and reused < tail_ids.len and gen[reused] == tail_ids[reused]) reused += 1;
    }
    const close_ids = try tok.encode(a, "</think>");
    defer a.free(close_ids);
    const markup_ids = try tok.encode(a, "\xef\xbd\x9cDSML\xef\xbd\x9c");
    defer a.free(markup_ids);
    return .{
        .gen = gen,
        .reused = reused,
        .prefix_kept = keep,
        .think_close_at = std.mem.indexOfScalar(u32, gen, close_ids[0]),
        .markup_at = std.mem.indexOfScalar(u32, gen, markup_ids[0]),
    };
}

const dsv41_replay_tools =
    \\[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"tags":{"type":"array"}}}}}]
;

test "dsv41 replay: what the next prompt shares with a generated turn (DeepSeek tokenizer.json)" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("test_models.zig").packPath(&buf, "DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw");
    var tok = tokenizer_mod.loadTokenizer(testing.io, testing.allocator, dir) catch return error.SkipZigTest;
    defer tok.deinit();
    const a = testing.allocator;
    const D = "\xef\xbd\x9cDSML\xef\xbd\x9c";
    const call = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"read_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">src/main.zig</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const call_json = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"read_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">src/main.zig</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"tags\" string=\"false\">[\"a\",\"b\"]</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"limit\" string=\"false\">40</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const call_code = "\n\n<" ++ D ++ " calls>\n<" ++ D ++ " invoke name=\"write_file\">\n<" ++ D ++ " parameter name=\"path\" string=\"true\">a.zig</" ++ D ++ " parameter>\n<" ++ D ++ " parameter name=\"content\" string=\"true\">const s = \"a\\nb\"; // <x> & 'y' \\ \xe2\x80\x94 \n\tfn f() void {}\n</" ++ D ++ " parameter>\n</" ++ D ++ " invoke>\n</" ++ D ++ " calls>";
    const reasoning = "The user wants the entry point. I should read src/main.zig first, then look at how the arguments are parsed.";
    const Reuse = enum {
        /// Every generated token comes back.
        whole,
        /// Through the call's first marker, then the re-render differs.
        into_call,
        /// Through the thought minus the whitespace the server trims off its end.
        thought_less_trim,
        /// The turn is not re-rendered as it was generated.
        none,
    };
    const cases = [_]struct { name: []const u8, tools: ?[]const u8, thinking: bool, raw: []const u8, reuse: Reuse }{
        .{ .name = "thinking, tools, content then a call", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>I will read the file." ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call alone", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call, .reuse = .whole },
        // jinja's `tojson` spaces the array's items, the model wrote none.
        .{ .name = "thinking, tools, a call with JSON values", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call_json, .reuse = .into_call },
        .{ .name = "thinking, tools, a call with code in a string", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call_code, .reuse = .whole },
        .{ .name = "thinking, tools, plain answer", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "</think>The entry point is `src/main.zig`.\n\nIt parses the flags first.", .reuse = .whole },
        .{ .name = "thinking, tools, reasoning ends in a newline", .tools = dsv41_replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>The entry point is `src/main.zig`.", .reuse = .thought_less_trim },
        // Without tools the template drops an earlier turn's thought and opens it `</think>`, not `<think>`.
        .{ .name = "thinking, no tools", .tools = null, .thinking = true, .raw = reasoning ++ "</think>The entry point is `src/main.zig`.", .reuse = .none },
        .{ .name = "chat, tools, content then a call", .tools = dsv41_replay_tools, .thinking = false, .raw = "I will read the file." ++ call, .reuse = .whole },
        .{ .name = "chat, no tools", .tools = null, .thinking = false, .raw = "The entry point is `src/main.zig`.", .reuse = .whole },
    };
    for (cases) |c| {
        var r = try dsv41Replay(a, &tok, c.tools, c.thinking, c.raw);
        defer r.deinit(a);
        errdefer std.debug.print("{s}: {d} tokens, prefix kept {}, reused {d}, </think> at {?d}, markup at {?d}\n", .{ c.name, r.gen.len, r.prefix_kept, r.reused, r.think_close_at, r.markup_at });
        switch (c.reuse) {
            .whole => try testing.expectEqual(r.gen.len, r.reused),
            .into_call => try testing.expect(r.reused > r.markup_at.? and r.reused < r.gen.len),
            .thought_less_trim => try testing.expectEqual(r.think_close_at.? - 1, r.reused),
            .none => try testing.expect(!r.prefix_kept),
        }
        // The point behind the closer is inside what the next prompt shares, unless the thought's edge is whitespace.
        if (c.thinking and c.tools != null and c.reuse != .thought_less_trim) try testing.expect(r.reused > r.think_close_at.?);
    }
}

const replay_tools =
    \\[{"type":"function","function":{"name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"tags":{"type":"array"}}}}}]
;

fn expectReplays(pack: []const u8, cases: []const ReplayCase) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try @import("test_models.zig").packPath(&buf, pack);
    const a = testing.allocator;
    var tok = tokenizer_mod.loadTokenizer(testing.io, a, dir) catch return error.SkipZigTest;
    defer tok.deinit();
    var cfg = loadChatConfig(testing.io, a, dir) catch return error.SkipZigTest;
    defer cfg.deinit();
    for (cases) |c| {
        var r = try replayTurn(a, &tok, &cfg, c.tools, c.thinking, c.raw);
        defer r.deinit(a);
        errdefer std.debug.print("{s}: {d} tokens, prefix kept {}, reused {d}, </think> at {?d}\n", .{ c.name, r.gen.len, r.prefix_kept, r.reused, r.think_close_at });
        switch (c.reuse) {
            .whole => try testing.expectEqual(r.gen.len, r.reused),
            .thought_short => try testing.expect(r.reused <= r.think_close_at.?),
            .none => try testing.expect(!r.prefix_kept or r.reused == 0),
        }
        // The state behind the closer is a restore point the next prompt reaches only when it repeats the closer.
        if (c.thinking and c.reuse != .thought_short and c.reuse != .none) try testing.expect(r.reused > r.think_close_at.?);
    }
}

test "qwen4 replay: what the next prompt shares with a generated turn (Flash-Next tokenizer.json, real template)" {
    const reasoning = "The user wants the entry point. I should read src/main.zig first, then look at how the arguments are parsed.";
    const call = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.zig\n</parameter>\n</function>\n</tool_call>";
    const call_json = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.zig\n</parameter>\n<parameter=tags>\n[\"a\",\"b\"]\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>";
    const call_code = "<tool_call>\n<function=write_file>\n<parameter=content>\nconst s = \"a\\nb\"; // <x> & 'y' \\ \xe2\x80\x94 \n\tfn f() void {}\n</parameter>\n</function>\n</tool_call>";
    try expectReplays("Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", &.{
        .{ .name = "thinking, tools, content then a call", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>\n\nI will read the file.\n\n" ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call alone", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>\n\n" ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call with JSON values", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>\n\n" ++ call_json, .reuse = .whole },
        .{ .name = "thinking, tools, a call with code in a string", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>\n\n" ++ call_code, .reuse = .whole },
        .{ .name = "thinking, tools, plain answer", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>\n\nThe entry point is `src/main.zig`.", .reuse = .whole },
        .{ .name = "thinking, no tools", .tools = null, .thinking = true, .raw = reasoning ++ "\n</think>\n\nThe entry point is `src/main.zig`.", .reuse = .whole },
        .{ .name = "thinking, tools, no newline before the closer", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "</think>\n\n" ++ call, .reuse = .thought_short },
        .{ .name = "chat, tools, content then a call", .tools = replay_tools, .thinking = false, .raw = "I will read the file.\n\n" ++ call, .reuse = .whole },
        .{ .name = "chat, no tools", .tools = null, .thinking = false, .raw = "The entry point is `src/main.zig`.", .reuse = .whole },
    });
}

test "glm5 replay: what the next prompt shares with a generated turn (GLM-5.3 tokenizer.json, real template)" {
    const reasoning = "The user wants the entry point. I should read src/main.zig first, then look at how the arguments are parsed.";
    const call = "<tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.zig</arg_value></tool_call>";
    const call_json = "<tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.zig</arg_value><arg_key>tags</arg_key><arg_value>[\"a\",\"b\"]</arg_value><arg_key>limit</arg_key><arg_value>40</arg_value></tool_call>";
    try expectReplays("GLM-5.3-Flash-Sushi-2.4bpw", &.{
        .{ .name = "thinking, tools, content then a call", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "</think>I will read the file." ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call alone", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call, .reuse = .whole },
        .{ .name = "thinking, tools, a call with JSON values", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "</think>" ++ call_json, .reuse = .whole },
        .{ .name = "thinking, tools, plain answer", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "</think>The entry point is `src/main.zig`.", .reuse = .whole },
        .{ .name = "thinking, tools, the thought ends in a newline", .tools = replay_tools, .thinking = true, .raw = reasoning ++ "\n</think>" ++ call, .reuse = .thought_short },
        .{ .name = "chat, tools, content then a call", .tools = replay_tools, .thinking = false, .raw = "I will read the file." ++ call, .reuse = .whole },
    });
}
