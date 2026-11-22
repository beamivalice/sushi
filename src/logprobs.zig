const std = @import("std");
const tokenizer_mod = @import("tokenizer.zig");
const scheduler_mod = @import("scheduler.zig");
const Tokenizer = tokenizer_mod.Tokenizer;
const sampler = @import("sampler.zig");
const server = @import("server.zig");
const generation_stream = @import("generation_stream.zig");
const contentListing = generation_stream.contentListing;
const contentLogprobsJson = generation_stream.contentLogprobsJson;
const listed_tokens = @import("listed_tokens.zig");
const Listing = listed_tokens.Listing;
const Cut = listed_tokens.Cut;
const Gap = listed_tokens.Gap;
const Range = listed_tokens.Range;
const think_tags = @import("think_tags.zig");
const jsonEscape = server.jsonEscape;

/// Streaming-side logprobs accumulator.
///
/// The generator produces one entry per token; the SSE side emits chunks on
/// its own cadence — the think gate and tool detection buffer many tokens into
/// one delta, and some tokens produce no chunk at all. So entries are drained
/// against the bytes the reply has delivered (`Listing`) rather than paired 1:1
/// with chunks, and each is shipped EXACTLY once: a delta cannot be retracted,
/// so a re-send is as wrong as a drop.
///
/// Entries are BORROWED from the slot, which owns each `top_logprobs`
/// allocation and frees it in `Slot.deinit`. Nothing here frees them.
pub const StreamLogprobs = struct {
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    slot: ?*scheduler_mod.Slot,
    enabled: bool,
    /// Alternatives listed beside each token (`request_parse.Logprobs.top`): the slot ranks at least one, the reply lists these.
    top: u32,
    /// High-water mark in the slot's buffer (what we have copied out).
    cursor: usize = 0,
    /// Which noted tokens the reply returned text for.
    listing: Listing = .{},
    ids: std.ArrayList(u32) = .empty,
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
        self.listing.deinit(self.allocator);
        self.entries.deinit(self.allocator);
    }

    /// Record a token the loop just consumed and the bytes it adds to the reply's text (0 for one the reply hides).
    /// The id is what renders the `token`/`bytes` fields; the logprob values arrive separately.
    pub fn note(self: *StreamLogprobs, id: u32, shown_len: usize) !void {
        if (!self.enabled) return;
        try self.ids.append(self.allocator, id);
        try self.listing.note(self.allocator, shown_len);
    }

    /// The stop gate's running count of delivered bytes (`Gate.released`), after each token it took, and whether the
    /// reply takes no more text (a stop matched, or the held tail was flushed). Bytes past it (a held stop prefix,
    /// a stop's own) are never listed while held, nor once the stop fired.
    pub fn release(self: *StreamLogprobs, delivered: usize, ended: bool) void {
        self.listing.released = delivered;
        self.listing.ended = self.listing.ended or ended;
    }

    /// Skip the entries of text the client never receives as content.
    ///
    /// `logprobs.content` describes `message.content`, and the think gate cuts
    /// the reasoning block out of the stream — so without this the entries
    /// shipped alongside the answer were the model's *thinking*. `tail` is the
    /// gate's buffer (always a suffix of the delivered text, since it is only
    /// ever cleared wholesale) and `content` a slice of it that survives.
    pub fn skipToContent(self: *StreamLogprobs, tail: []const u8, content: []const u8) void {
        if (!self.enabled or content.len == 0) return;
        const base = @intFromPtr(tail.ptr);
        const cptr = @intFromPtr(content.ptr);
        std.debug.assert(cptr >= base and cptr + content.len <= base + tail.len);
        if (self.listing.released < tail.len) return;
        self.listing.skipTo(self.listing.released - tail.len + (cptr - base));
    }

    /// The content part being delivered ends (a thought opens or closes it): what it returned lists ahead of what follows,
    /// and its withheld tail never does.
    pub fn closePart(self: *StreamLogprobs) !void {
        if (self.enabled) try self.listing.endPart(self.allocator);
    }

    /// `closePart`, and nothing released so far is content any more: a thought opens.
    pub fn endPart(self: *StreamLogprobs) !void {
        if (!self.enabled) return;
        try self.listing.endPart(self.allocator);
        self.listing.retire();
    }

    /// The content arm took the piece of the generation that starts at `at` and is `len` long, and answered `step`.
    pub fn afterDeliver(self: *StreamLogprobs, step: think_tags.ContentStep, content: *const think_tags.ContentStream, at: usize, len: usize) !void {
        if (!self.enabled) return;
        if (content.trimmed > 0) self.listing.skipTo(at + content.trimmed);
        switch (step) {
            .marker => try self.listing.hide(self.allocator, at, at + len),
            .hold => if (content.mark.cut) self.listing.retire(),
            // Suppressed lead: nothing pending describes text the client received, so retire it rather than letting it
            // ride the next chunk.
            .drop => self.listing.retire(),
            .send => {},
        }
    }

    /// The reply's content is `content`, a slice of `text`, which is the last `text_len` bytes of the released generation
    /// (`runs`: a rewrite of them, as `normalizeEmbeddedThinkBlocksMapped` made it) and ships as one chunk: it lists
    /// through the same runs a non-stream reply reads, and nothing released before it does.
    pub fn contentThrough(self: *StreamLogprobs, runs: ?[]const listed_tokens.Run, text: []const u8, source_len: usize, content: []const u8) !void {
        if (!self.enabled or content.len == 0) return;
        const at = @intFromPtr(content.ptr) - @intFromPtr(text.ptr);
        try self.listing.returnedText(self.allocator, runs, self.listing.released -| source_len, at, at + content.len);
        self.listing.retire();
    }

    /// Retire every pending entry unshipped — the turn produced no content, so
    /// nothing pending describes text the client received.
    pub fn dropPending(self: *StreamLogprobs) void {
        if (!self.enabled) return;
        self.listing.retire();
    }

    /// What a delta that lists no token carries when logprobs were asked (the join of two content parts): a list with no
    /// entry, so a client reading `logprobs.content` always finds one. Null when they were not asked.
    pub fn empty(self: *const StreamLogprobs) ?[]const u8 {
        return if (self.enabled) "{\"content\":[]}" else null;
    }

    /// Render what a content chunk going out now describes, or null when logprobs weren't requested or
    /// nothing has landed since the last chunk. `withheld` is the tail of the released bytes the content arm holds
    /// back: the chunk does not carry it yet, and a trimmed reply never will.
    pub fn take(self: *StreamLogprobs, withheld: usize) !?[]const u8 {
        return self.drain(true, withheld);
    }

    /// Render the entries of content already sent that had not landed when it went out. A chunk that carries no
    /// content (the final one) takes through this: what the reply did not return is never listed there.
    pub fn takeLate(self: *StreamLogprobs) !?[]const u8 {
        self.listing.ended = true;
        return self.drain(false, 0);
    }

    fn drain(self: *StreamLogprobs, content_chunk: bool, withheld: usize) !?[]const u8 {
        if (!self.enabled) return null;
        if (self.rendered) |r| {
            self.allocator.free(r);
            self.rendered = null;
        }
        if (self.slot) |slot| self.cursor = try slot.copyLogprobsFrom(self.allocator, self.cursor, &self.entries);
        // An id with no entry yet (or the reverse) is a partially published
        // token — hold it for the next chunk rather than shipping a half pair.
        const json = if (self.legacy) blk: {
            const r = if (content_chunk) self.listing.sent(self.entries.items.len, withheld) else self.listing.late(self.entries.items.len);
            if (r.end <= r.start) return null;
            break :blk try formatCompletionsLogprobs(self.allocator, self.tok, self.ids.items[r.start..r.end], self.entries.items[r.start..r.end], self.top, r.cut(), &self.text_offset);
        } else blk: {
            const items = try self.listing.items(self.allocator, self.entries.items.len, content_chunk, withheld);
            if (items.len == 0) return null;
            break :blk try formatLogprobsObject(self.allocator, self.tok, self.ids.items, self.entries.items, self.top, items, self.listing.gaps.items);
        };
        self.rendered = json;
        return json;
    }
};

/// Build logprobs JSON for a single token (for both streaming and non-streaming). `head` and `tail` bytes of the
/// token are not text the reply returned: `token` and `bytes` carry the rest, `logprob` and the alternatives stay
/// the whole token's.
/// Returns a string like: {"token":"hello","logprob":-1.23,"bytes":[104,101],"top_logprobs":[...]}
fn formatTokenLogprob(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_id: u32,
    logprob: f32,
    top_logprobs: []const sampler.TokenLogprob,
    cut: Cut,
    gaps: []const Gap,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    const whole = try tok.decode(allocator, &[_]u32{token_id});
    defer allocator.free(whole);
    const returned = whole[@min(cut.head, whole.len)..whole.len -| cut.tail];
    // The bytes the reply cut out from inside the token are no part of what it lists.
    var kept = std.ArrayList(u8).empty;
    defer kept.deinit(allocator);
    for (returned, @min(cut.head, whole.len)..) |b, at| {
        var cut_out = false;
        for (gaps) |g| cut_out = cut_out or (at >= g.at and at < g.at + g.len);
        if (!cut_out) try kept.append(allocator, b);
    }
    const token_text = if (gaps.len == 0) returned else kept.items;

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
    top: u32,
    cut: Cut,
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
        const whole = try tok.decode(allocator, &[_]u32{token_ids[i]});
        defer allocator.free(whole);
        const text = whole[if (i == 0) @min(cut.head, whole.len) else 0 .. whole.len -| (if (i + 1 == count) cut.tail else 0)];
        const esc = try jsonEscape(allocator, text);
        defer allocator.free(esc);
        try toks.appendSlice(allocator, esc);

        var num: [48]u8 = undefined;
        try lps.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d:.6}", .{logprobs[i].token_logprob}));
        try offs.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d}", .{offset}));
        offset += text.len;

        try tops.appendSlice(allocator, "{");
        for (logprobs[i].top_logprobs[0..@min(top, logprobs[i].top_logprobs.len)], 0..) |t, j| {
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

/// Build the logprobs object of a chat reply from the runs of tokens `runs` list, in order: indexes into `token_ids` and
/// `logprobs`. The "\n" that joins content parts has no token, so no entry lists it.
pub fn formatLogprobsObject(
    allocator: std.mem.Allocator,
    tok: *const Tokenizer,
    token_ids: []const u32,
    logprobs: []const sampler.LogprobResult,
    top: u32,
    runs: []const Range,
    gaps: []const Gap,
) ![]const u8 {
    var content_buf = std.ArrayList(u8).empty;
    defer content_buf.deinit(allocator);

    try content_buf.appendSlice(allocator, "[");
    const count = @min(token_ids.len, logprobs.len);
    var first = true;
    for (runs) |r| for (r.start..@max(r.start, @min(r.end, count))) |i| {
        if (!first) try content_buf.appendSlice(allocator, ",");
        first = false;
        const own: Cut = .{ .head = if (i == r.start) r.head else 0, .tail = if (i + 1 == r.end) r.tail else 0 };
        const entry = try formatTokenLogprob(allocator, tok, token_ids[i], logprobs[i].token_logprob, logprobs[i].top_logprobs[0..@min(top, logprobs[i].top_logprobs.len)], own, Listing.gapsOf(gaps, i));
        defer allocator.free(entry);
        try content_buf.appendSlice(allocator, entry);
    };
    try content_buf.appendSlice(allocator, "]");

    return try std.fmt.allocPrint(allocator, "{{\"content\":{s}}}", .{content_buf.items});
}

test "skipToContent keeps the content token when a multi-byte character splits across tokens" {
    const a = std.testing.allocator;
    var tok = Tokenizer.initEmpty(a);
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

    var lps = StreamLogprobs{ .allocator = a, .tok = &tok, .slot = null, .enabled = true, .top = 1 };
    defer lps.deinit();
    try lps.note(1, 1);
    try lps.note(2, 1);
    try lps.note(3, lead.len);
    // The gate delivered "RC"; the lead byte of the split character is still its carry.
    lps.release(2, false);
    const tail = "RC";
    lps.skipToContent(tail, tail[1..]);
    try std.testing.expectEqual(@as(usize, 1), lps.listing.emitted);
}

test "formatCompletionsLogprobs: legacy shape, byte-aligned offsets, escaped tokens" {
    // /v1/completions ignored its `logprobs` field entirely (hardcoded 0) while
    // still emitting the key, so a client read "no alternatives exist" rather
    // than "never asked". Its shape is NOT chat's: four parallel arrays, and
    // `top_logprobs` is a map keyed by token TEXT.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmpty(allocator);
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
    const json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &lps, 20, .{}, &tbase);
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

    var tok = Tokenizer.initEmpty(allocator);
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
    const chat_json = try formatTokenLogprob(allocator, &tok, 2, -9.0, &t0, .{}, &.{});
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
    const comp_json = try formatCompletionsLogprobs(allocator, &tok, &token_ids, &comp_lps, 20, .{}, &tbase);
    defer allocator.free(comp_json);
    try std.testing.expect(std.unicode.utf8ValidateSlice(comp_json));
    const comp_parsed = try std.json.parseFromSlice(std.json.Value, allocator, comp_json, .{});
    defer comp_parsed.deinit();
    // text_offset stays on the RAW byte length — it indexes the completion the
    // client received, which is unaffected by how the string renders.
    try std.testing.expectEqual(@as(usize, 2), tbase);

    // Valid UTF-8 is untouched, including multi-byte that arrives whole.
    try tok.id_to_token.put(5, "é€");
    const clean = try formatTokenLogprob(allocator, &tok, 5, -1.0, &[_]sampler.TokenLogprob{}, .{}, &.{});
    defer allocator.free(clean);
    const clean_parsed = try std.json.parseFromSlice(std.json.Value, allocator, clean, .{});
    defer clean_parsed.deinit();
    try std.testing.expectEqualStrings("é€", clean_parsed.value.object.get("token").?.string);
}

/// The one token range a `contentListing` lists, as `(start, end)`.
fn listedRange(listing: *const Listing) struct { start: usize, end: usize } {
    const items = listing.queued();
    if (items.len == 0) return .{ .start = 0, .end = 0 };
    std.debug.assert(items.len == 1);
    return .{ .start = items[0].start, .end = items[0].end };
}

test "contentListing: logprobs cover the CONTENT, not the reasoning we stripped" {
    // `logprobs.content` is defined as the tokens of the message content, but we
    // built it from the whole generation — so on every model that thinks, entry
    // 0 was the first token of the *reasoning* and the array lined up with
    // nothing the client could see.
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmpty(allocator);
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

    var l = try contentListing(allocator, &tok, &token_ids, full, null, content);
    defer l.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 3), listedRange(&l).start);
    try std.testing.expectEqual(@as(usize, 5), listedRange(&l).end);

    // Whole text as content (a model that did not think) keeps every entry.
    var all = try contentListing(allocator, &tok, &token_ids, full, null, full);
    defer all.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), listedRange(&all).start);
    try std.testing.expectEqual(@as(usize, 5), listedRange(&all).end);

    // Empty content (a turn that was all reasoning) emits no entries rather
    // than the reasoning's.
    var none = try contentListing(allocator, &tok, &token_ids, full, null, full[0..0]);
    defer none.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), none.queued().len);

    // A content that is no slice of the text is refused: it is never found by its bytes.
    try std.testing.expectError(error.ContentIsNotTheReplyText, contentListing(allocator, &tok, &token_ids, full, null, "not in there"));

    // A token straddling the boundary belongs to content — it put bytes there.
    var mid = try contentListing(allocator, &tok, &token_ids, full, null, full[("<think>\nBecause.\n</thi").len..]);
    defer mid.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), listedRange(&mid).start);
}

test "contentLogprobsJson: a reply with a call lists the tokens of its content, never the call's, and `top` alternatives each" {
    const allocator = std.testing.allocator;

    var tok = Tokenizer.initEmpty(allocator);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.id_to_token.put(1, "Check");
    try tok.id_to_token.put(2, " it.\n");
    try tok.id_to_token.put(3, "<tool_call>");
    try tok.id_to_token.put(4, "{x}");

    const token_ids = [_]u32{ 1, 2, 3, 4 };
    const full = "Check it.\n<tool_call>{x}";
    var alt = [_]sampler.TokenLogprob{.{ .token_id = 1, .logprob = -0.5 }};
    const lps = [_]sampler.LogprobResult{
        .{ .token_logprob = -0.5, .top_logprobs = &alt },
        .{ .token_logprob = -0.5, .top_logprobs = &alt },
        .{ .token_logprob = -0.5, .top_logprobs = &alt },
        .{ .token_logprob = -0.5, .top_logprobs = &alt },
    };
    for ([_]u32{ 0, 1 }) |top| {
        const json = try contentLogprobsJson(allocator, &tok, &token_ids, &lps, full, null, full[0..10], top);
        defer allocator.free(json);
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
        defer parsed.deinit();
        const entries = parsed.value.object.get("content").?.array.items;
        try std.testing.expectEqual(@as(usize, 2), entries.len);
        try std.testing.expectEqualStrings(" it.\n", entries[1].object.get("token").?.string);
        for (entries) |e| try std.testing.expectEqual(@as(usize, top), e.object.get("top_logprobs").?.array.items.len);
    }
}

test "a token the reply returned part of lists those bytes, its logprob and alternatives whole, on both shapes" {
    const allocator = std.testing.allocator;
    var tok = Tokenizer.initEmpty(allocator);
    defer tok.vocab.deinit();
    defer tok.id_to_token.deinit();
    defer tok.merge_ranks.deinit();
    defer tok.special_tokens.deinit();
    defer tok.unicode_to_byte.deinit();
    try tok.id_to_token.put(1, " 빵");
    try tok.id_to_token.put(2, "[section");
    const ids = [_]u32{ 1, 2 };
    var alt = [_]sampler.TokenLogprob{.{ .token_id = 2, .logprob = -0.5 }};
    const lps = [_]sampler.LogprobResult{ .{ .token_logprob = -0.25, .top_logprobs = &alt }, .{ .token_logprob = -0.75, .top_logprobs = &alt } };
    // The reply returned "빵[s": the first token's space and the last token's "ection" are not its text.
    const cut: Cut = .{ .head = 1, .tail = 6 };

    const chat = try formatLogprobsObject(allocator, &tok, &ids, &lps, 1, &.{.{ .start = 0, .end = 2, .head = cut.head, .tail = cut.tail }}, &.{});
    defer allocator.free(chat);
    const cp = try std.json.parseFromSlice(std.json.Value, allocator, chat, .{});
    defer cp.deinit();
    const entries = cp.value.object.get("content").?.array.items;
    try std.testing.expectEqualStrings("빵", entries[0].object.get("token").?.string);
    try std.testing.expectEqualStrings("[s", entries[1].object.get("token").?.string);
    try std.testing.expectEqual(@as(usize, 3), entries[0].object.get("bytes").?.array.items.len);
    try std.testing.expectEqual(@as(usize, 2), entries[1].object.get("bytes").?.array.items.len);
    try std.testing.expectApproxEqAbs(@as(f64, -0.25), entries[0].object.get("logprob").?.float, 1e-6);
    try std.testing.expectEqualStrings("[section", entries[1].object.get("top_logprobs").?.array.items[0].object.get("token").?.string);

    var base: usize = 0;
    const legacy = try formatCompletionsLogprobs(allocator, &tok, &ids, &lps, 1, cut, &base);
    defer allocator.free(legacy);
    const lp = try std.json.parseFromSlice(std.json.Value, allocator, legacy, .{});
    defer lp.deinit();
    try std.testing.expectEqualStrings("빵", lp.value.object.get("tokens").?.array.items[0].string);
    try std.testing.expectEqualStrings("[s", lp.value.object.get("tokens").?.array.items[1].string);
    // text_offset indexes the returned text, and the running base ends where it ends.
    try std.testing.expectEqual(@as(i64, 3), lp.value.object.get("text_offset").?.array.items[1].integer);
    try std.testing.expectEqual(@as(usize, 5), base);
}

const reply_mod = @import("reply.zig");
const stop_seq_mod = @import("stop_sequences.zig");

/// A generation as tokens: id `i + 1` decodes to `tokens[i]`, and token `i` has logprob `-(i + 1) / 10` with itself as its
/// one alternative.
const Generation = struct {
    allocator: std.mem.Allocator,
    tok: Tokenizer,
    ids: []u32,
    lps: []sampler.LogprobResult,
    tops: []sampler.TokenLogprob,
    raw: []u8,

    fn init(allocator: std.mem.Allocator, tokens: []const []const u8) !Generation {
        var tok = Tokenizer.initEmpty(allocator);
        errdefer tok.vocab.deinit();
        const ids = try allocator.alloc(u32, tokens.len);
        const lps = try allocator.alloc(sampler.LogprobResult, tokens.len);
        const tops = try allocator.alloc(sampler.TokenLogprob, tokens.len);
        var raw = std.ArrayList(u8).empty;
        for (tokens, 0..) |t, i| {
            ids[i] = @intCast(i + 1);
            try tok.id_to_token.put(ids[i], t);
            tops[i] = .{ .token_id = ids[i], .logprob = -0.1 * @as(f32, @floatFromInt(i + 1)) };
            lps[i] = .{ .token_logprob = tops[i].logprob, .top_logprobs = tops[i .. i + 1] };
            try raw.appendSlice(allocator, t);
        }
        return .{ .allocator = allocator, .tok = tok, .ids = ids, .lps = lps, .tops = tops, .raw = try raw.toOwnedSlice(allocator) };
    }

    fn deinit(self: *Generation) void {
        self.allocator.free(self.ids);
        self.allocator.free(self.lps);
        self.allocator.free(self.tops);
        self.allocator.free(self.raw);
        self.tok.vocab.deinit();
        self.tok.id_to_token.deinit();
        self.tok.merge_ranks.deinit();
        self.tok.special_tokens.deinit();
        self.tok.unicode_to_byte.deinit();
    }
};

/// The entries of a `{"content":[...]}` object, each as its JSON text, appended to `out`.
fn addEntries(allocator: std.mem.Allocator, json: []const u8, out: *std.ArrayList([]u8)) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{});
    defer parsed.deinit();
    for (parsed.value.object.get("content").?.array.items) |e| try out.append(allocator, try std.json.Stringify.valueAlloc(allocator, e, .{}));
}

fn freeEntries(allocator: std.mem.Allocator, list: *std.ArrayList([]u8)) void {
    for (list.items) |e| allocator.free(e);
    list.deinit(allocator);
}

/// The bytes the entries list, in order.
fn entryBytes(allocator: std.mem.Allocator, entries: []const []u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (entries) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, text, .{});
        defer parsed.deinit();
        for (parsed.value.object.get("bytes").?.array.items) |b| try out.append(allocator, @intCast(b.integer));
    }
    return out.toOwnedSlice(allocator);
}

const weather_tools =
    \\[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]
;

/// A non-stream reply: its content (owned; with `tools`, the text before its call) and the entries its `logprobs` lists.
fn wholeReply(allocator: std.mem.Allocator, g: *const Generation, stops: []const []const u8, tools: ?[]const u8, entries: *std.ArrayList([]u8)) ![]u8 {
    const ask: reply_mod.Ask = .{ .stops = stops, .normalize = true, .has_tools = tools != null, .tools_json = tools, .allow_parallel = true, .dialect = null, .opens_think = false };
    const settled = try reply_mod.settle(allocator, .{ .text = g.raw, .source = .whole, .finish_reason = "length", .finish_details = null }, ask);
    defer settled.deinit(allocator);
    const content = if (settled.reply) |r| r.preamble else think_tags.splitThinkBlock(settled.text, true, false).content;
    const json = try contentLogprobsJson(allocator, &g.tok, g.ids, g.lps, settled.text, settled.provenance(), content, 1);
    defer allocator.free(json);
    try addEntries(allocator, json, entries);
    return allocator.dupe(u8, content);
}

/// What a stream ships: the bytes of its deltas and the entries of its chunks, in order, and each delta's own bytes and the bytes its entries list.
const Shipped = struct {
    delta: std.ArrayList(u8) = .empty,
    entries: std.ArrayList([]u8) = .empty,
    deltas: std.ArrayList(Delta) = .empty,

    /// `list` is whether the delta carried a `logprobs.content` list, entries or none.
    const Delta = struct { content: []u8, listed: []u8, list: bool };

    fn deinit(self: *Shipped, allocator: std.mem.Allocator) void {
        self.delta.deinit(allocator);
        freeEntries(allocator, &self.entries);
        for (self.deltas.items) |d| {
            allocator.free(d.content);
            allocator.free(d.listed);
        }
        self.deltas.deinit(allocator);
    }

    fn chunk(self: *Shipped, allocator: std.mem.Allocator, bytes: []const u8, lp: ?[]const u8) !void {
        try self.delta.appendSlice(allocator, bytes);
        var mine: std.ArrayList([]u8) = .empty;
        defer freeEntries(allocator, &mine);
        if (lp) |json| try addEntries(allocator, json, &mine);
        try self.deltas.append(allocator, .{ .content = try allocator.dupe(u8, bytes), .listed = try entryBytes(allocator, mine.items), .list = lp != null });
        for (mine.items) |e| try self.entries.append(allocator, try allocator.dupe(u8, e));
    }

    /// A content delta as `sendContentDelta` sends it: the join that opens a part first, listing nothing.
    fn contentDelta(self: *Shipped, allocator: std.mem.Allocator, content: *think_tags.ContentStream, lps: *StreamLogprobs, bytes: []const u8) !void {
        if (content.takeJoin()) |join| try self.chunk(allocator, join, lps.empty());
        try self.chunk(allocator, bytes, try lps.take(content.withheld()));
    }
};

/// The stream of a reply with no tools, token by token as `streamChatCompletion` takes it: the gate releases text, the content arm
/// delivers it, a thought the model opens is held and closed, and the final chunk lists what lagged.
fn plainStream(allocator: std.mem.Allocator, g: *const Generation, stops: []const []const u8, out: *Shipped) !void {
    var lps = StreamLogprobs{ .allocator = allocator, .tok = &g.tok, .slot = null, .enabled = true, .top = 1 };
    defer lps.deinit();
    var gate = stop_seq_mod.Gate{ .stops = stops };
    defer gate.deinit(allocator);
    var content: think_tags.ContentStream = .{ .mark = .{ .on = true }, .join_apart = true };
    defer content.deinit(allocator);
    var think_buf = std.ArrayList(u8).empty;
    defer think_buf.deinit(allocator);
    var in_think = false;
    var thought_closed = false;
    var channel_armed = false;
    var i: usize = 0;
    var tail = false;
    // The reply ended: one more turn, with no piece, reads what the arms held for bytes that decide it.
    var over = false;
    while (true) : (i += 1) {
        const arms_wait = content.waiting() or (in_think and think_buf.items.len > 0);
        if (over or (gate.matched != null and !arms_wait)) break;
        if (gate.matched != null) over = true;
        const piece: []const u8 = if (over) "" else if (i < g.ids.len) blk: {
            try lps.entries.append(allocator, g.lps[i]);
            const raw = try g.tok.decode(allocator, &[_]u32{g.ids[i]});
            defer allocator.free(raw);
            try lps.note(g.ids[i], raw.len);
            break :blk try gate.push(allocator, raw);
        } else if (gate.matched == null and gate.hasHeld() and !tail) blk: {
            tail = true;
            break :blk try gate.finish(allocator);
        } else if (arms_wait) blk: {
            over = true;
            break :blk "";
        } else break;
        defer allocator.free(piece);
        lps.release(gate.released, gate.matched != null or tail);
        if (piece.len == 0 and !over) continue;
        var think_has_token = false;
        if (!in_think) {
            if (piece.len > 0) try content.arrive(allocator, piece, lps.listing.released -| piece.len);
            while (try content.ready(allocator, channel_armed, over)) |r| switch (r) {
                .opens => |opener| {
                    try lps.endPart();
                    content.endPart();
                    channel_armed = false;
                    in_think = true;
                    try think_buf.appendSlice(allocator, opener);
                    try think_buf.appendSlice(allocator, content.takeWaiting().bytes);
                    think_has_token = true;
                    break;
                },
                .step => |st| {
                    try lps.afterDeliver(st.step, &content, st.at, st.text.len);
                    channel_armed = st.step == .marker and std.mem.eql(u8, st.text, "<|channel>");
                    switch (st.step) {
                        .marker, .hold, .drop => {},
                        .send => |bytes| try out.contentDelta(allocator, &content, &lps, bytes),
                    }
                },
            };
        }
        if (in_think) {
            if (!think_has_token) try think_buf.appendSlice(allocator, piece);
            if (think_tags.thoughtClose(think_buf.items, over)) |close| {
                const content_after = std.mem.trimStart(u8, think_buf.items[close.pos + close.len ..], "\n ");
                if (content_after.len > 0) {
                    lps.skipToContent(think_buf.items, content_after);
                    switch (try content.deliver(allocator, content_after)) {
                        .send => |bytes| try out.contentDelta(allocator, &content, &lps, bytes),
                        else => {},
                    }
                } else lps.dropPending();
                think_buf.clearRetainingCapacity();
                in_think = false;
                thought_closed = true;
            }
        }
    }
    const held = try content.finish(allocator, thought_closed);
    if (held.len > 0) try out.contentDelta(allocator, &content, &lps, held);
    try out.chunk(allocator, "", try lps.takeLate());
}

/// The stream of a reply the gate buffered to its end (a tools request): the delivered text is normalized once and its
/// content ships as one chunk.
fn flushStream(allocator: std.mem.Allocator, g: *const Generation, stops: []const []const u8, out: *Shipped) !void {
    var lps = StreamLogprobs{ .allocator = allocator, .tok = &g.tok, .slot = null, .enabled = true, .top = 1 };
    defer lps.deinit();
    var gate = stop_seq_mod.Gate{ .stops = stops };
    defer gate.deinit(allocator);
    var full_text = std.ArrayList(u8).empty;
    defer full_text.deinit(allocator);
    var i: usize = 0;
    var tail = false;
    while (true) : (i += 1) {
        const piece = if (i < g.ids.len) blk: {
            try lps.entries.append(allocator, g.lps[i]);
            const raw = try g.tok.decode(allocator, &[_]u32{g.ids[i]});
            defer allocator.free(raw);
            try lps.note(g.ids[i], raw.len);
            break :blk try gate.push(allocator, raw);
        } else if (gate.matched == null and gate.hasHeld() and !tail) blk: {
            tail = true;
            break :blk try gate.finish(allocator);
        } else break;
        defer allocator.free(piece);
        lps.release(gate.released, gate.matched != null or tail);
        try full_text.appendSlice(allocator, piece);
        if (gate.matched != null) break;
    }
    const norm = try think_tags.normalizeEmbeddedThinkBlocksMapped(allocator, full_text.items);
    defer if (norm) |n| n.deinit(allocator);
    const text: []const u8 = if (norm) |n| n.text else full_text.items;
    const content = think_tags.splitThinkBlock(text, true, false).content;
    if (content.len > 0) {
        const runs: ?[]const listed_tokens.Run = if (norm) |n| n.runs else null;
        const content_at = @intFromPtr(content.ptr) - @intFromPtr(text.ptr);
        var shipped: usize = 0;
        for (runs orelse &.{}) |run| {
            if (run.src != null or run.at < content_at + shipped or run.at + run.len > content_at + content.len) continue;
            const piece = content[shipped .. run.at - content_at];
            if (piece.len > 0) {
                try lps.contentThrough(runs, text, full_text.items.len, piece);
                try out.chunk(allocator, piece, try lps.take(0));
            }
            try out.chunk(allocator, listed_tokens.content_join, lps.empty());
            shipped = run.at - content_at + run.len;
        }
        const last = content[shipped..];
        if (last.len > 0) {
            try lps.contentThrough(runs, text, full_text.items.len, last);
            try out.chunk(allocator, last, try lps.take(0));
        }
    } else lps.dropPending();
    try out.chunk(allocator, "", try lps.takeLate());
}

const Scenario = struct {
    name: []const u8,
    tokens: []const []const u8,
    stops: []const []const u8 = &.{},
    /// The `token` of each entry the reply lists: a token cut at the reply's edge lists what it returned.
    listed: []const []const u8,
    /// Where in the content the joins no token carries lie, each one "\n" that no entry lists.
    joins: []const usize,
    /// No token of the reply holds bytes past the delta that ships them (a trimmed tail or a stop inside a token lists after
    /// them): each delta lists exactly its own bytes.
    aligned: bool = true,
    /// The think style the per-token stream follows: a channel thought opened after content is the whole reply's to read.
    plain: bool = true,
};

const scenarios = [_]Scenario{
    .{ .name = "content, a thought, content", .tokens = &.{ "Sure.", "<think>", "hmm", "</think>", "Done." }, .listed = &.{ "Sure.", "Done." }, .joins = &.{ 5 } },
    .{ .name = "whitespace around the thought, inside tokens", .tokens = &.{ "Sure.\n", "<think>", "hmm", "</think>", "\nDone.\n" }, .listed = &.{ "Sure.", "Done." }, .joins = &.{ 5 }, .aligned = false },
    .{ .name = "whitespace tokens around the thought", .tokens = &.{ "Sure.", "\n\n", "<think>", "hmm", "</think>", "\n\n", "Done." }, .listed = &.{ "Sure.", "Done." }, .joins = &.{ 5 } },
    .{ .name = "a stop inside the second part", .tokens = &.{ "Sure.", "<think>", "hmm", "</think>", "Do", "ne. ST", "OP x" }, .stops = &.{"STOP"}, .listed = &.{ "Sure.", "Do", "ne." }, .joins = &.{ 5 }, .aligned = false },
    .{ .name = "a stray closer inside a part", .tokens = &.{ "Sure.", "<think>", "hmm", "</think>", "X", "</think>", "Y" }, .listed = &.{ "Sure.", "X", "Y" }, .joins = &.{ 5 } },
    .{ .name = "a closer after a stray backtick waits for the line's end", .tokens = &.{ "<think>", "t", "</think>", "Use `", "</think>", " here" }, .listed = &.{ "Use `", " here" }, .joins = &.{} },
    .{ .name = "a closer a closing backtick follows is text", .tokens = &.{ "<think>", "t", "</think>", "Use `", "</think>", "` here" }, .listed = &.{ "Use `", "</think>", "` here" }, .joins = &.{} },
    .{ .name = "two thoughts", .tokens = &.{ "A", "<think>", "p", "</think>", "B", "<think>", "q", "</think>", "C" }, .listed = &.{ "A", "B", "C" }, .joins = &.{ 1, 3 } },
    .{ .name = "a channel thought after content, its two tokens", .tokens = &.{ "Sure.", "<|channel>", "thought", "\nhmm", "<channel|>", "Done." }, .listed = &.{ "Sure.", "Done." }, .joins = &.{5} },
    .{ .name = "a closer inside one token, bytes returned on both sides", .tokens = &.{ "<think>", "t", "</think>", "a", "b</think>c", "d" }, .listed = &.{ "a", "bc", "d" }, .joins = &.{}, .plain = false },
    .{ .name = "the channel style", .tokens = &.{ "Sure.", "<|channel>thought\n", "hmm", "<channel|>", "Done." }, .listed = &.{ "Sure.", "Done." }, .joins = &.{ 5 }, .plain = false },
};

test "a reply with a thought in the middle lists exactly the bytes its tokens returned, no entry for the join of its parts, streamed or not" {
    const allocator = std.testing.allocator;
    var bad: usize = 0;
    for (scenarios) |s| {
        var g = try Generation.init(allocator, s.tokens);
        defer g.deinit();
        var whole: std.ArrayList([]u8) = .empty;
        defer freeEntries(allocator, &whole);
        const content = try wholeReply(allocator, &g, s.stops, null, &whole);
        defer allocator.free(content);
        const listed = try entryBytes(allocator, whole.items);
        defer allocator.free(listed);
        if (whole.items.len != s.listed.len) {
            std.debug.print("{s}: the whole reply lists {d} entries, want {d}\n", .{ s.name, whole.items.len, s.listed.len });
            bad += 1;
        } else for (whole.items, s.listed) |entry, want| {
            const parsed = try std.json.parseFromSlice(std.json.Value, allocator, entry, .{});
            defer parsed.deinit();
            const token = parsed.value.object.get("token").?.string;
            // Every entry is a token's: a number logprob, whatever the client's schema reads it as.
            const lp = parsed.value.object.get("logprob").?;
            const alts = parsed.value.object.get("top_logprobs").?.array.items.len;
            if (!std.mem.eql(u8, token, want) or lp != .float or alts != 1) {
                std.debug.print("{s}: entry {f}, want token {f}\n", .{ s.name, std.zig.fmtString(entry), std.zig.fmtString(want) });
                bad += 1;
            }
        }
        // The listed bytes are the content less exactly its joins, each a "\n" at the position the cell names.
        var less_joins = std.ArrayList(u8).empty;
        defer less_joins.deinit(allocator);
        var at: usize = 0;
        for (s.joins) |p| {
            if (p >= content.len or content[p] != '\n') {
                std.debug.print("{s}: no join at {d} of {f}\n", .{ s.name, p, std.zig.fmtString(content) });
                bad += 1;
                continue;
            }
            try less_joins.appendSlice(allocator, content[at..p]);
            at = p + 1;
        }
        try less_joins.appendSlice(allocator, content[at..]);
        if (!std.mem.eql(u8, less_joins.items, listed)) {
            std.debug.print("{s}: whole reply returns {f}, lists {f}, want {f}\n", .{ s.name, std.zig.fmtString(content), std.zig.fmtString(listed), std.zig.fmtString(less_joins.items) });
            bad += 1;
        }
        const arms = [_]struct { name: []const u8, run: *const fn (std.mem.Allocator, *const Generation, []const []const u8, *Shipped) anyerror!void }{
            .{ .name = "per token", .run = plainStream },
            .{ .name = "buffered", .run = flushStream },
        };
        for (arms) |arm| {
            if (!s.plain and arm.run == plainStream) continue;
            var got: Shipped = .{};
            defer got.deinit(allocator);
            try arm.run(allocator, &g, s.stops, &got);
            // Every join ships as a delta of its own that lists nothing; the other deltas list their own bytes.
            var join_deltas: usize = 0;
            for (got.deltas.items) |d| {
                const is_join = std.mem.eql(u8, d.content, listed_tokens.content_join) and d.listed.len == 0;
                if (is_join) join_deltas += 1;
                // The join's list is an empty list, never null: a client reading `logprobs.content` finds one.
                if (is_join and !d.list) {
                    std.debug.print("{s}, {s}: the join delta carries no logprobs.content list\n", .{ s.name, arm.name });
                    bad += 1;
                }
                if (s.aligned and !is_join and !std.mem.eql(u8, d.content, d.listed)) {
                    std.debug.print("{s}, {s}: a delta returns {f}, lists {f}\n", .{ s.name, arm.name, std.zig.fmtString(d.content), std.zig.fmtString(d.listed) });
                    bad += 1;
                }
            }
            if (join_deltas != s.joins.len) {
                std.debug.print("{s}, {s}: {d} deltas are exactly a join, want {d}\n", .{ s.name, arm.name, join_deltas, s.joins.len });
                bad += 1;
            }
            if (!std.mem.eql(u8, got.delta.items, content)) {
                std.debug.print("{s}, {s}: the stream returns {f}, the whole reply {f}\n", .{ s.name, arm.name, std.zig.fmtString(got.delta.items), std.zig.fmtString(content) });
                bad += 1;
            }
            const same = got.entries.items.len == whole.items.len and for (got.entries.items, whole.items) |a, b| {
                if (!std.mem.eql(u8, a, b)) break false;
            } else true;
            if (!same) {
                std.debug.print("{s}, {s}: the stream lists {d} entries, the whole reply {d}\n", .{ s.name, arm.name, got.entries.items.len, whole.items.len });
                for (got.entries.items) |e| std.debug.print("  stream {s}\n", .{e});
                for (whole.items) |e| std.debug.print("  whole  {s}\n", .{e});
                bad += 1;
            }
        }
    }
    if (bad > 0) return error.ListedWhatWasNotReturned;
}

test "a constrained reply lists the tokens its content was copied from, never the ones a search for its bytes finds" {
    // The reasoning holds the payload's own bytes: finding the content in the generation lists the thought's token.
    const allocator = std.testing.allocator;
    const rp = @import("reasoning_protocol.zig");
    var g = try Generation.init(allocator, &.{ "<think>", "{\"a\":1}", "</think>", " ", "{\"a\":1}" });
    defer g.deinit();
    var proto = rp.Protocol{ .kind = .bare_think };
    try std.testing.expect(proto.setOpener("<think>"));
    try std.testing.expect(proto.setCloser("</think>"));
    var d = rp.Delivery.init(&proto);
    defer d.deinit(allocator);
    d.payload_byte = "<think>{\"a\":1}</think>".len;
    try d.feed(allocator, g.raw);
    try d.finish(allocator);
    try std.testing.expectEqualStrings("{\"a\":1}", d.content.items);
    const json = try generation_stream.constrainedLogprobsJson(allocator, &g.tok, g.ids, g.lps, &d, 1);
    defer allocator.free(json);
    var entries: std.ArrayList([]u8) = .empty;
    defer freeEntries(allocator, &entries);
    try addEntries(allocator, json, &entries);
    // Token 5 (index 4) holds the payload; the space before it is not content.
    try std.testing.expectEqual(@as(usize, 1), entries.items.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, entries.items[0], .{});
    defer parsed.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, -0.5), parsed.value.object.get("logprob").?.float, 1e-6);
}

test "a reply with a call lists its preamble through the same runs: the join of its parts has no entry" {
    const allocator = std.testing.allocator;
    const call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";
    var g = try Generation.init(allocator, &.{ "Sure.", "<think>", "hmm", "</think>", "Done.\n", call });
    defer g.deinit();
    var entries: std.ArrayList([]u8) = .empty;
    defer freeEntries(allocator, &entries);
    const preamble = try wholeReply(allocator, &g, &.{}, weather_tools, &entries);
    defer allocator.free(preamble);
    try std.testing.expectEqualStrings("Sure.\nDone.", preamble);
    const listed = try entryBytes(allocator, entries.items);
    defer allocator.free(listed);
    try std.testing.expectEqualStrings("Sure.Done.", listed);
    try std.testing.expectEqual(@as(usize, 2), entries.items.len);
}
