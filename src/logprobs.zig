const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");
const scheduler_mod = @import("scheduler.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const sampler = @import("sampler.zig");
const server = @import("server.zig");
const contentTokenRange = server.contentTokenRange;
const jsonEscape = server.jsonEscape;

/// Streaming-side logprobs accumulator.
///
/// The generator produces one entry per token; the SSE side emits chunks on
/// its own cadence — the think gate and tool detection buffer many tokens into
/// one delta, and some tokens produce no chunk at all. So entries are drained
/// against a high-water mark rather than paired 1:1 with chunks, and each is
/// shipped EXACTLY once: a delta cannot be retracted, so a re-send is as wrong
/// as a drop.
///
/// Entries are BORROWED from the slot, which owns each `top_logprobs`
/// allocation and frees it in `Slot.deinit`. Nothing here frees them.
pub const StreamLogprobs = struct {
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    slot: ?*scheduler_mod.Slot,
    enabled: bool,
    /// High-water mark in the slot's buffer (what we have copied out).
    cursor: usize = 0,
    /// How many entries have already ridden a chunk.
    emitted: usize = 0,
    ids: std.ArrayList(u32) = .empty,
    /// Decoded byte length of each noted token, so a byte offset in the
    /// generated text converts to a token index WITHOUT re-decoding and without
    /// depending on how far entry publication has lagged behind `ids`.
    lens: std.ArrayList(usize) = .empty,
    /// Total decoded bytes noted so far — the generation's length.
    bytes_noted: usize = 0,
    entries: std.ArrayList(sampler.LogprobResult) = .empty,
    /// The last rendered JSON, owned here and freed on the next drain — the
    /// caller passes it straight into a chunk and never sees the lifetime.
    rendered: ?[]const u8 = null,
    /// `/v1/completions` takes the LEGACY shape (four parallel arrays) rather
    /// than chat's list, and its `text_offset` indexes the whole completion —
    /// so the running base has to survive across chunks.
    legacy: bool = false,
    text_offset: usize = 0,

    pub fn deinit(self: *StreamLogprobs) void {
        if (self.rendered) |r| self.allocator.free(r);
        self.ids.deinit(self.allocator);
        self.lens.deinit(self.allocator);
        self.entries.deinit(self.allocator);
    }

    /// Record the id of a token the loop just consumed. The id is what renders
    /// the `token`/`bytes` fields; the logprob values arrive separately.
    pub fn note(self: *StreamLogprobs, id: u32) !void {
        if (!self.enabled) return;
        try self.ids.append(self.allocator, id);
        const text = self.tok.decode(self.allocator, &[_]u32{id}) catch "";
        defer if (text.len > 0) self.allocator.free(text);
        try self.lens.append(self.allocator, text.len);
        self.bytes_noted += text.len;
    }

    /// Drop the entries for tokens whose bytes never reach the client.
    ///
    /// `logprobs.content` describes `message.content`, and the think gate cuts
    /// the reasoning block out of the stream — so without this the entries
    /// shipped alongside the answer were the model's *thinking*. `tail` is the
    /// gate's buffer (always a suffix of the generation, since it is only ever
    /// cleared wholesale) and `content` the part of it that survives.
    ///
    /// Indexes `ids`/`lens`, which are complete, rather than the pending
    /// window — a token whose logprob has not been published yet still has a
    /// length, so a lagging publisher cannot skew the boundary.
    pub fn skipToContent(self: *StreamLogprobs, tail: []const u8, content: []const u8, carry: usize) void {
        if (!self.enabled or content.len == 0) return;
        const base = @intFromPtr(tail.ptr);
        const cptr = @intFromPtr(content.ptr);
        const off_in_tail = if (cptr >= base and cptr + content.len <= base + tail.len)
            cptr - base
        else
            std.mem.indexOf(u8, tail, content) orelse return;
        const noted = self.bytes_noted -| carry;
        if (noted < tail.len) return;
        const abs = noted - tail.len + off_in_tail;

        var off: usize = 0;
        for (self.lens.items, 0..) |len, i| {
            // A token straddling the boundary put bytes into content, so it is
            // the first content token rather than the last reasoning one.
            if (off + len > abs) {
                if (i > self.emitted) self.emitted = i;
                return;
            }
            off += len;
        }
        self.emitted = self.ids.items.len;
    }

    /// Retire every pending entry unshipped — the turn produced no content, so
    /// nothing pending describes text the client received.
    pub fn dropPending(self: *StreamLogprobs) void {
        if (!self.enabled) return;
        self.emitted = self.ids.items.len;
    }

    /// Render everything new, or null when logprobs weren't requested or
    /// nothing has landed since the last chunk.
    pub fn take(self: *StreamLogprobs) !?[]const u8 {
        if (!self.enabled) return null;
        if (self.rendered) |r| {
            self.allocator.free(r);
            self.rendered = null;
        }
        const slot = self.slot orelse return null;
        self.cursor = try slot.copyLogprobsFrom(self.allocator, self.cursor, &self.entries);
        // An id with no entry yet (or the reverse) is a partially published
        // token — hold it for the next chunk rather than shipping a half pair.
        const avail = @min(self.ids.items.len, self.entries.items.len);
        if (avail <= self.emitted) return null;
        const ids = self.ids.items[self.emitted..avail];
        const lps = self.entries.items[self.emitted..avail];
        const json = if (self.legacy)
            try formatCompletionsLogprobs(self.allocator, self.tok, ids, lps, &self.text_offset)
        else
            try formatLogprobsObject(self.allocator, self.tok, ids, lps);
        self.emitted = avail;
        self.rendered = json;
        return json;
    }
};

/// Build logprobs JSON for a single token (for both streaming and non-streaming).
/// Returns a string like: {"token":"hello","logprob":-1.23,"bytes":[104,101],"top_logprobs":[...]}
fn formatTokenLogprob(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_id: u32,
    logprob: f32,
    top_logprobs: []const sampler.TokenLogprob,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    const token_text = try tok.decode(allocator, &[_]u32{token_id});
    defer allocator.free(token_text);

    const escaped_token = try jsonEscape(allocator, token_text);
    defer allocator.free(escaped_token);

    // Build bytes array
    var bytes_buf = std.ArrayList(u8).empty;
    defer bytes_buf.deinit(allocator);
    try bytes_buf.appendSlice(allocator, "[");
    for (token_text, 0..) |b, i| {
        if (i > 0) try bytes_buf.appendSlice(allocator, ",");
        const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
        defer allocator.free(num);
        try bytes_buf.appendSlice(allocator, num);
    }
    try bytes_buf.appendSlice(allocator, "]");

    // Build top_logprobs array
    var top_buf = std.ArrayList(u8).empty;
    defer top_buf.deinit(allocator);
    try top_buf.appendSlice(allocator, "[");
    for (top_logprobs, 0..) |tlp, i| {
        if (i > 0) try top_buf.appendSlice(allocator, ",");

        const tlp_text = try tok.decode(allocator, &[_]u32{tlp.token_id});
        defer allocator.free(tlp_text);
        const escaped_tlp = try jsonEscape(allocator, tlp_text);
        defer allocator.free(escaped_tlp);

        // Bytes for this token
        var tlp_bytes = std.ArrayList(u8).empty;
        defer tlp_bytes.deinit(allocator);
        try tlp_bytes.appendSlice(allocator, "[");
        for (tlp_text, 0..) |b, j| {
            if (j > 0) try tlp_bytes.appendSlice(allocator, ",");
            const num = try std.fmt.allocPrint(allocator, "{d}", .{b});
            defer allocator.free(num);
            try tlp_bytes.appendSlice(allocator, num);
        }
        try tlp_bytes.appendSlice(allocator, "]");

        const entry = try std.fmt.allocPrint(allocator,
            \\{{"token":{s},"logprob":{d:.6},"bytes":{s}}}
        , .{ escaped_tlp, tlp.logprob, tlp_bytes.items });
        defer allocator.free(entry);
        try top_buf.appendSlice(allocator, entry);
    }
    try top_buf.appendSlice(allocator, "]");

    const result = try std.fmt.allocPrint(allocator,
        \\{{"token":{s},"logprob":{d:.6},"bytes":{s},"top_logprobs":{s}}}
    , .{ escaped_token, logprob, bytes_buf.items, top_buf.items });

    return result;
}

/// Build the LEGACY `/v1/completions` logprobs object — a different shape from
/// chat's: four parallel arrays (`tokens`, `token_logprobs`, `top_logprobs`,
/// `text_offset`), with `top_logprobs` a map from token text to logprob.
/// `text_offset` is the byte offset of each token within the completion text,
/// which is what a FIM client uses to align alternatives with the buffer.
/// `offset_base` is READ and ADVANCED: `text_offset` indexes into the whole
/// completion, so a streaming caller emitting one chunk per token has to carry
/// the running total across calls. Non-streaming passes a local zero.
pub fn formatCompletionsLogprobs(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const sampler.LogprobResult,
    offset_base: *usize,
) ![]const u8 {
    var toks = std.ArrayList(u8).empty;
    defer toks.deinit(allocator);
    var lps = std.ArrayList(u8).empty;
    defer lps.deinit(allocator);
    var tops = std.ArrayList(u8).empty;
    defer tops.deinit(allocator);
    var offs = std.ArrayList(u8).empty;
    defer offs.deinit(allocator);
    try toks.appendSlice(allocator, "[");
    try lps.appendSlice(allocator, "[");
    try tops.appendSlice(allocator, "[");
    try offs.appendSlice(allocator, "[");

    var offset: usize = offset_base.*;
    const count = @min(token_ids.len, logprobs.len);
    for (0..count) |i| {
        if (i > 0) {
            try toks.appendSlice(allocator, ",");
            try lps.appendSlice(allocator, ",");
            try tops.appendSlice(allocator, ",");
            try offs.appendSlice(allocator, ",");
        }
        const text = try tok.decode(allocator, &[_]u32{token_ids[i]});
        defer allocator.free(text);
        const esc = try jsonEscape(allocator, text);
        defer allocator.free(esc);
        try toks.appendSlice(allocator, esc);

        var num: [48]u8 = undefined;
        try lps.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d:.6}", .{logprobs[i].token_logprob}));
        try offs.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d}", .{offset}));
        offset += text.len;

        try tops.appendSlice(allocator, "{");
        for (logprobs[i].top_logprobs, 0..) |t, j| {
            if (j > 0) try tops.appendSlice(allocator, ",");
            const ttext = try tok.decode(allocator, &[_]u32{t.token_id});
            defer allocator.free(ttext);
            const tesc = try jsonEscape(allocator, ttext);
            defer allocator.free(tesc);
            try tops.appendSlice(allocator, tesc);
            try tops.appendSlice(allocator, ":");
            try tops.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d:.6}", .{t.logprob}));
        }
        try tops.appendSlice(allocator, "}");
    }
    try toks.appendSlice(allocator, "]");
    try lps.appendSlice(allocator, "]");
    try tops.appendSlice(allocator, "]");
    try offs.appendSlice(allocator, "]");
    offset_base.* = offset;

    return try std.fmt.allocPrint(allocator,
        \\{{"tokens":{s},"token_logprobs":{s},"top_logprobs":{s},"text_offset":{s}}}
    , .{ toks.items, lps.items, tops.items, offs.items });
}

/// Build the full logprobs object for a non-streaming response.
pub fn formatLogprobsObject(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const sampler.LogprobResult,
) ![]const u8 {
    var content_buf = std.ArrayList(u8).empty;
    defer content_buf.deinit(allocator);

    try content_buf.appendSlice(allocator, "[");
    const count = @min(token_ids.len, logprobs.len);
    for (0..count) |i| {
        if (i > 0) try content_buf.appendSlice(allocator, ",");
        const entry = try formatTokenLogprob(allocator, tok, token_ids[i], logprobs[i].token_logprob, logprobs[i].top_logprobs);
        defer allocator.free(entry);
        try content_buf.appendSlice(allocator, entry);
    }
    try content_buf.appendSlice(allocator, "]");

    return try std.fmt.allocPrint(allocator, "{{\"content\":{s}}}", .{content_buf.items});
}

test "skipToContent keeps the content token when a multi-byte character splits across tokens" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.unicode_to_byte.put('L', 0xc3);
    try tok.unicode_to_byte.put('T', 0xa9);
    try tok.id_to_token.put(1, "R");
    try tok.id_to_token.put(2, "C");
    try tok.id_to_token.put(3, "L");
    try tok.id_to_token.put(4, "T");
    const lead = try tok.decode(a, &.{3});
    defer a.free(lead);
    const cont = try tok.decode(a, &.{4});
    defer a.free(cont);
    var joined = try a.alloc(u8, lead.len + cont.len);
    defer a.free(joined);
    @memcpy(joined[0..lead.len], lead);
    @memcpy(joined[lead.len..], cont);
    try std.testing.expectEqualStrings("é", joined);

    var lps = StreamLogprobs{ .allocator = a, .tok = &tok, .slot = null, .enabled = true };
    defer lps.deinit();
    try lps.note(1);
    try lps.note(2);
    try lps.note(3);
    const tail = "RC";
    lps.skipToContent(tail, tail[1..], lead.len);
    try std.testing.expectEqual(@as(usize, 1), lps.emitted);
}

test "formatCompletionsLogprobs: legacy shape, byte-aligned offsets, escaped tokens" {
    // /v1/completions ignored its `logprobs` field entirely (hardcoded 0) while
    // still emitting the key, so a client read "no alternatives exist" rather
    // than "never asked". Its shape is NOT chat's: four parallel arrays, and
    // `top_logprobs` is a map keyed by token TEXT.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    // Ids 1..4. Id 4 carries a control byte AND a quote — the escape has to be
    // the shared sink, not a hand-rolled table (the /detokenize class).
    try tok.id_to_token.put(1, "Paris");
    try tok.id_to_token.put(2, ",");
    try tok.id_to_token.put(3, " the");
    try tok.id_to_token.put(4, "a\x01\"b");

    const token_ids = [_]u32{ 1, 2, 3 };
    var t0 = [_]sampler.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.25 },
        .{ .token_id = 4, .logprob = -3.5 },
    };
    var t1 = [_]sampler.TokenLogprob{.{ .token_id = 2, .logprob = -0.5 }};
    var t2 = [_]sampler.TokenLogprob{.{ .token_id = 3, .logprob = -1.0 }};
    const lps = [_]sampler.LogprobResult{
        .{ .token_logprob = -0.25, .top_logprobs = &t0 },
        .{ .token_logprob = -0.5, .top_logprobs = &t1 },
        .{ .token_logprob = -1.0, .top_logprobs = &t2 },
    };

    var tbase: usize = 0;
    const json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &lps, &tbase);
    defer allocator.free(json);

    // Must be parseable at all — the control byte is the reason that matters.
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    // Four arrays, all the same length as the token list.
    for ([_][]const u8{ "tokens", "token_logprobs", "top_logprobs", "text_offset" }) |k| {
        const arr = o.get(k) orelse return error.MissingKey;
        try std.testing.expectEqual(@as(usize, 3), arr.array.items.len);
    }
    try std.testing.expectEqualStrings("Paris", o.get("tokens").?.array.items[0].string);
    // text_offset is the BYTE offset of each token within the completion, which
    // is what a FIM client aligns against — not a token index.
    const offs = o.get("text_offset").?.array.items;
    try std.testing.expectEqual(@as(i64, 0), offs[0].integer); // ""
    try std.testing.expectEqual(@as(i64, 5), offs[1].integer); // "Paris"
    try std.testing.expectEqual(@as(i64, 6), offs[2].integer); // "Paris,"
    // The chosen token's own logprob rides beside the alternatives.
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), o.get("token_logprobs").?.array.items[0].float, 1e-6);
    // top_logprobs is a MAP keyed by token text (OpenAI's shape), and the
    // control-byte token survived the round trip through the escaper.
    const top0 = o.get("top_logprobs").?.array.items[0].object;
    try std.testing.expectEqual(@as(usize, 2), top0.count());
    try std.testing.expect(top0.get("Paris") != null);
    try std.testing.expect(top0.get("a\x01\"b") != null);
}

test "logprobs token strings are valid UTF-8 — a split multi-byte token can't break the body" {
    // A single token is a BPE fragment, so it can carry HALF a multi-byte
    // character; `jsonEscape` passes every byte >= 0x20 through verbatim, so
    // those raw bytes landed in the JSON string and the WHOLE response body
    // stopped being valid UTF-8 — not a degraded field, an unparseable
    // response. Live on Jundot/Qwen3.6-27B-oQ4e-mtp: a `b"\xf0\x9f"` candidate
    // (the first 2 bytes of a 4-byte emoji) in `top_logprobs`.
    //
    // `bytes` already carries the truth, so the string is the lossy view:
    // U+FFFD per invalid sequence, which is what OpenAI's own shape does.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    // Byte-level BPE stores its vocab in the byte-to-unicode alphabet, so the
    // split sequence has to come out of `decode`, not sit in the vocab: map
    // three placeholder codepoints back onto the raw bytes.
    try tok.unicode_to_byte.put('A', 0xf0);
    try tok.unicode_to_byte.put('B', 0x9f);
    try tok.unicode_to_byte.put('C', 0xe2);
    try tok.unicode_to_byte.put('D', 0x82);
    try tok.id_to_token.put(1, "ok");
    // Leading half of U+1F600, exactly as the live checkpoint emitted it.
    try tok.id_to_token.put(2, "AB");
    // A lone continuation byte, and a 3-byte lead truncated to 2.
    try tok.id_to_token.put(3, "B");
    try tok.id_to_token.put(4, "CD");

    var t0 = [_]sampler.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.1 },
        .{ .token_id = 2, .logprob = -9.0 },
        .{ .token_id = 3, .logprob = -9.5 },
        .{ .token_id = 4, .logprob = -9.9 },
    };
    const token_ids = [_]u32{1};

    // Chat shape.
    const chat_json = try formatTokenLogprob(allocator, &tok, 2, -9.0, &t0);
    defer allocator.free(chat_json);
    try std.testing.expect(std.unicode.utf8ValidateSlice(chat_json));
    const chat_parsed = try std.json.parseFromSlice(std.json.Value, allocator, chat_json, .{});
    defer chat_parsed.deinit();
    // The lossy string stands in for the split character...
    try std.testing.expectEqualStrings("\u{FFFD}", chat_parsed.value.object.get("token").?.string);
    // ...while `bytes` still reports exactly what the model emitted, so a
    // client that cares can reassemble it across tokens.
    const bytes = chat_parsed.value.object.get("bytes").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), bytes.len);
    try std.testing.expectEqual(@as(i64, 0xf0), bytes[0].integer);
    try std.testing.expectEqual(@as(i64, 0x9f), bytes[1].integer);

    // Legacy /v1/completions shape — same escaper, same guarantee. Its
    // `top_logprobs` is a map keyed by token TEXT, so it carries ONE invalid
    // candidate here: two would both render as U+FFFD and collide into a single
    // key. That collision is OpenAI's own shape reproduced faithfully, not a
    // defect this test should pin away.
    var comp_top = [_]sampler.TokenLogprob{
        .{ .token_id = 1, .logprob = -0.1 },
        .{ .token_id = 2, .logprob = -9.0 },
    };
    const comp_lps = [_]sampler.LogprobResult{.{ .token_logprob = -0.1, .top_logprobs = &comp_top }};
    var tbase: usize = 0;
    const comp_json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &comp_lps, &tbase);
    defer allocator.free(comp_json);
    try std.testing.expect(std.unicode.utf8ValidateSlice(comp_json));
    const comp_parsed = try std.json.parseFromSlice(std.json.Value, allocator, comp_json, .{});
    defer comp_parsed.deinit();
    // text_offset stays on the RAW byte length — it indexes the completion the
    // client received, which is unaffected by how the string renders.
    try std.testing.expectEqual(@as(usize, 2), tbase);

    // Valid UTF-8 is untouched, including multi-byte that arrives whole.
    try tok.id_to_token.put(5, "é€");
    const clean = try formatTokenLogprob(allocator, &tok, 5, -1.0, &[_]sampler.TokenLogprob{});
    defer allocator.free(clean);
    const clean_parsed = try std.json.parseFromSlice(std.json.Value, allocator, clean, .{});
    defer clean_parsed.deinit();
    try std.testing.expectEqualStrings("é€", clean_parsed.value.object.get("token").?.string);
}

test "contentTokenRange: logprobs cover the CONTENT, not the reasoning we stripped" {
    // `logprobs.content` is defined as the tokens of the message content, but we
    // built it from the whole generation — so on every model that thinks, entry
    // 0 was the first token of the *reasoning* and the array lined up with
    // nothing the client could see.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmptyForTests(allocator);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.id_to_token.put(1, "<think>");
    try tok.id_to_token.put(2, "\nBecause.\n");
    try tok.id_to_token.put(3, "</think>");
    try tok.id_to_token.put(4, "Can");
    try tok.id_to_token.put(5, "berra");

    const token_ids = [_]u32{ 1, 2, 3, 4, 5 };
    const full = "<think>\nBecause.\n</think>Canberra";
    const content = full[("<think>\nBecause.\n</think>").len..];
    try std.testing.expectEqualStrings("Canberra", content);

    const r = contentTokenRange(allocator, &tok, &token_ids, full, content);
    try std.testing.expectEqual(@as(usize, 3), r.start);
    try std.testing.expectEqual(@as(usize, 5), r.end);

    // Whole text as content (a model that did not think) keeps every entry.
    const all = contentTokenRange(allocator, &tok, &token_ids, full, full);
    try std.testing.expectEqual(@as(usize, 0), all.start);
    try std.testing.expectEqual(@as(usize, 5), all.end);

    // Empty content (a turn that was all reasoning) emits no entries rather
    // than the reasoning's.
    const none = contentTokenRange(allocator, &tok, &token_ids, full, full[0..0]);
    try std.testing.expectEqual(@as(usize, 0), none.start);
    try std.testing.expectEqual(@as(usize, 0), none.end);

    // A content slice we cannot locate keeps the full range: an array we can't
    // align is still better than dropping it silently.
    const foreign = contentTokenRange(allocator, &tok, &token_ids, full, "not in there");
    try std.testing.expectEqual(@as(usize, 0), foreign.start);
    try std.testing.expectEqual(@as(usize, 5), foreign.end);

    // A token straddling the boundary belongs to content — it put bytes there.
    const mid = contentTokenRange(allocator, &tok, &token_ids, full, full[("<think>\nBecause.\n</thi").len..]);
    try std.testing.expectEqual(@as(usize, 2), mid.start);
}
