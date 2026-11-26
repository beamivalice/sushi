//! The one reading of a generation's tool-call text. Thoughts, calls and the spans of a call's values are read
//! here once, by the served dialect's own value ends (`tool_calls.closerInsideValue`), so the open-call finder
//! and every think-tag pass agree on which text is structure and which is a call's payload.
const std = @import("std");
const think_tags = @import("think_tags.zig");
const tool_calls = @import("tool_calls.zig");
const BARE_INVOKE = tool_calls.BARE_INVOKE;
const FUNCTION_CALLS_TAG = tool_calls.FUNCTION_CALLS_TAG;

/// Bytes every reading search has covered on this thread (the tool-call reading and the think-tag passes beside it): a
/// test holds the work of a whole reply to a multiple of its text, so a pass that rereads the text per call shows.
pub threadlocal var work: usize = 0;

/// Records that a reading search covered `n` bytes.
pub fn charge(n: usize) void {
    work +|= n;
}

/// One call dialect: the openers that may start a call, the closers that end it, and for a dialect whose opener
/// alone is prose, what the text from the opener on must hold for the parser to read a call there.
pub const Family = struct {
    openers: []const []const u8,
    closers: []const []const u8,
    /// Spellings that begin with an opener and are another dialect's closer.
    not_openers: []const []const u8 = &.{},
    reads_as_call: ?*const fn (text: []const u8, at: usize, scan: *Scan) bool = null,
    /// What the text after the opener must hold for the READING to open a call there, where the opener alone is markup
    /// to cut: `I could emit <tool_call> here` is prose to a reader and leaked markup to a cut.
    body_reads: ?*const fn (text: []const u8, at: usize, scan: *Scan) bool = null,
    /// Instead of `reads_as_call` when cutting leaked markup: a malformed opener (a dropped attribute quote) that the
    /// parser declined is still markup, so the cut reads looser than a call does.
    cut_reads: ?*const fn (text: []const u8, at: usize, scan: *Scan) bool = null,
    /// How the dialect writes a parameter value: a closer spelled inside one is the value's text, not the call's close.
    values: tool_calls.ValueSpans,
    /// Holds calls of other families (`<｜DSML｜tool_calls>`, `<function_calls>`): they are read where they stand,
    /// and the container is a call of its own only when none follows.
    container: bool = false,
    /// A wrapper whose body is `<function=…>` is whole at the function's own close once the model ended the
    /// generation itself: the wrapper's close is optional then.
    body_closers: []const []const u8 = &.{},
};

const hermes_closers = [_][]const u8{ "</tool_call>", "</tool_calls>", "</tool>", "</tool_request>", "</tool_requests>" };

/// Every dialect `parseToolCallsAs` reads.
pub const families = [_]Family{
    .{ .openers = &.{"<|tool_call"}, .closers = &.{"<tool_call|>"}, .body_reads = gemmaReadsAsCall, .values = .gemma },
    .{ .openers = &.{"<tool_call"}, .closers = &hermes_closers, .not_openers = &.{ "<tool_call|>", "<tool_calls:opensource>", "<tool_call:opensource>" }, .body_reads = wrapperReadsAsCall, .values = .wrapper, .body_closers = &.{"</function>"} },
    .{ .openers = &.{ "<tool>", "<tool ", "<tool\t", "<tool\n", "<tool_request" }, .closers = &hermes_closers, .reads_as_call = genericToolTagReadsAsCall, .values = .wrapper, .body_closers = &.{"</function>"} },
    .{ .openers = &.{"<｜DSML｜tool_calls"}, .closers = &.{"</｜DSML｜tool_calls>"}, .values = .dsml, .container = true },
    .{ .openers = &.{ "<｜DSML｜invoke", "<｜DSML｜ invoke" }, .closers = &.{ "</｜DSML｜invoke>", "</｜DSML｜ invoke>" }, .body_reads = dsmlInvokeReadsAsCall, .values = .dsml },
    .{ .openers = &.{FUNCTION_CALLS_TAG}, .closers = &.{"</function_calls>"}, .values = .invoke, .container = true },
    .{ .openers = &.{BARE_INVOKE.invoke}, .closers = &.{BARE_INVOKE.invoke_close}, .reads_as_call = invokeReadsAsCall, .values = .invoke },
    .{ .openers = &.{"<function"}, .closers = &.{"</function>"}, .reads_as_call = functionReadsAsCall, .cut_reads = functionCutReads, .values = .function_tag },
    // Hy4-Flash: GLM's calls in a `<tool_calls:opensource>` wrapper, a tag spelled `<tool_call:opensource>` where its pack
    // decodes no plain one. Hermes' `<tool_call` opener gives these up (`not_openers`).
    .{ .openers = &.{ "<tool_calls:opensource>", "<tool_call:opensource>" }, .closers = &.{ "</tool_call:opensource>", "</tool_call>" }, .body_reads = wrapperReadsAsCall, .values = .wrapper },
};

/// `<|tool_call>call:name{…`: the body starts with `call:`, or has not got that far.
fn gemmaReadsAsCall(text: []const u8, at: usize, _: *Scan) bool {
    const rest = text[at + "<|tool_call".len ..];
    if (rest.len == 0) return true;
    if (rest[0] != '>') return false;
    const body = std.mem.trimStart(u8, rest[1..], " \t\r\n");
    return std.mem.startsWith(u8, body, "call:") or std.mem.startsWith(u8, "call:", body);
}

/// A bare `<tool_call>` opens a call when a JSON object, a tag or a tool name followed by its first argument tag
/// leads its body, or the text stops there: `I could emit <tool_call> here` is prose, and so is a mention followed
/// by another opener. Another spelling (`<tool_call name=`, `<tool_calls>`) is a call.
fn wrapperReadsAsCall(text: []const u8, at: usize, _: *Scan) bool {
    const tail = text[at..];
    const open = "<tool_call";
    if (tail.len <= open.len or tail[open.len] != '>') return true;
    const body = std.mem.trimStart(u8, tail[open.len + 1 ..], " \t\r\n");
    if (body.len == 0 or body[0] == '{') return true;
    if (body[0] == '<') return !(std.mem.startsWith(u8, body, "<|tool_call") or std.mem.startsWith(u8, body, "<tool_call") or std.mem.startsWith(u8, body, "<\u{ff5c}DSML"));
    var name: usize = 0;
    while (name < body.len and (std.ascii.isAlphanumeric(body[name]) or body[name] == '_' or body[name] == '-' or body[name] == '.')) name += 1;
    const after = std.mem.trimStart(u8, body[name..], " \t\r\n");
    return name > 0 and (after.len == 0 or std.mem.startsWith(u8, after, "<arg_") or std.mem.startsWith(u8, after, "</"));
}

/// `<tool name="x">{…`, `<tool>{"name":…` and `<tool_request>{…`: the tag's `>` opens a body that is a JSON object or Hermes' function tag.
/// A self-closing `<tool …/>` is a whole call, never an open one.
fn genericToolTagReadsAsCall(text: []const u8, at: usize, _: *Scan) bool {
    const tail = text[at..];
    const gt = std.mem.indexOfScalar(u8, tail[0..@min(tail.len, 256)], '>') orelse return false;
    if (tail[gt - 1] == '/') return false;
    const body = std.mem.trimStart(u8, tail[gt + 1 ..], " \t\n\r");
    if (std.mem.startsWith(u8, body, "{")) return std.mem.startsWith(u8, std.mem.trimStart(u8, body[1..], " \t\n\r"), "\"");
    return std.mem.startsWith(u8, body, "<function=");
}

/// `<｜DSML｜invoke name="x">` with a parameter or its close before the next invoke, or with the text stopping inside
/// it: a mention of the opener in prose has neither.
fn dsmlInvokeReadsAsCall(text: []const u8, at: usize, scan: *Scan) bool {
    const tail = text[at..];
    const gt = std.mem.indexOfScalar(u8, tail[0..@min(tail.len, 512)], '>') orelse return tail.len < 512;
    if ((tool_calls.dsmlAttr(tail[0..gt], "name") orelse return false).len == 0) return false;
    const from = at + gt + 1;
    const next = earliestOf(scan.dsml_invoke.find(text, from), scan.dsml_invoke_spaced.find(text, from)) orelse return true;
    const own = text[from..next];
    return std.mem.indexOf(u8, own, "<｜DSML｜parameter") != null or std.mem.indexOf(u8, own, "<｜DSML｜ parameter") != null or
        std.mem.indexOf(u8, own, "</｜DSML｜invoke>") != null or std.mem.indexOf(u8, own, "</｜DSML｜ invoke>") != null;
}

/// `<invoke name="x">` with a `<parameter ` of its own before the next `<invoke`: the opener alone is prose.
fn invokeReadsAsCall(text: []const u8, at: usize, _: *Scan) bool {
    const tail = text[at..];
    const after = BARE_INVOKE.invoke.len;
    if (tail.len <= after or !std.ascii.isWhitespace(tail[after])) return false;
    const name = tool_calls.dsmlAttr(tail[after..@min(tail.len, after + 512)], "name") orelse return false;
    if (name.len == 0) return false;
    const own = tail[after .. std.mem.indexOfPos(u8, tail, after, BARE_INVOKE.invoke) orelse tail.len];
    charge(own.len);
    return std.mem.indexOf(u8, own, BARE_INVOKE.param ++ " ") != null or std.mem.indexOf(u8, own, BARE_INVOKE.invoke_close) != null;
}

/// MiniCPM5's `<function name="x">` with a `<param` after its tag, and Hermes' bare `<function=x>` with a
/// `<parameter=`: either opener alone is prose.
fn functionReadsAsCall(text: []const u8, at: usize, scan: *Scan) bool {
    const after = at + "<function".len;
    if (after >= text.len) return false;
    if (text[after] == '=') return scan.parameter_eq.find(text, after) != null;
    if (!std.ascii.isWhitespace(text[after])) return false;
    const gt = scan.gt.find(text, after) orelse return false;
    const name = tool_calls.miniCpm5AttrValue(text[after..gt], "name") orelse return false;
    return name.len > 0 and (scan.param.find(text, gt + 1) != null or scan.function_close.find(text, gt + 1) != null);
}

/// A `<function` that leaks markup: the bare `<function=` read as a call, or a MiniCPM5 `<function name=…` (a dropped
/// quote included) with a `<param` element or a `</function>` after it; text that stops inside the opener counts.
fn functionCutReads(text: []const u8, at: usize, scan: *Scan) bool {
    const after = at + "<function".len;
    if (after >= text.len) return true;
    if (text[after] == '=') return functionReadsAsCall(text, at, scan);
    if (!std.ascii.isWhitespace(text[after])) return false;
    const gt = scan.gt.find(text, after);
    if (std.mem.indexOf(u8, text[after .. gt orelse text.len], "name=") == null) return false;
    return scan.param.find(text, after) != null or scan.function_close.find(text, after) != null;
}

/// Whether the text at `at`, which starts with an opener of `f`, opens a call there: no other dialect's closer, and the
/// dialect's own reading accepts it.
fn opensCall(f: *const Family, text: []const u8, at: usize, scan: *Scan) bool {
    if (scan.strict) for (f.not_openers) |n| if (std.mem.startsWith(u8, text[at..], n)) return false;
    if (!scan.strict) if (f.cut_reads) |reads| return reads(text, at, scan);
    if (f.reads_as_call) |reads| return reads(text, at, scan);
    if (scan.strict) if (f.body_reads) |reads| return reads(text, at, scan);
    return true;
}

/// A call opener found in a text.
pub const Opener = struct { at: usize, len: usize, family: *const Family };

/// The first place a needle stands at or after a byte, asked for rising bytes: an answer stays valid for every byte
/// between the one asked and the place found (or, with none found, for every later byte up to where the search looked),
/// so a needle is searched once however many candidates ask.
pub const Next = struct {
    needle: []const u8,
    from: usize = 0,
    at: ?usize = null,
    /// With none found: the text up to here holds none from `from` on.
    to: usize = 0,
    known: bool = false,

    pub fn find(self: *Next, text: []const u8, p: usize) ?usize {
        return self.findBefore(text, p, text.len);
    }

    /// `find` that looks no further than `limit`: a caller that only wants an occurrence before it never pays for the
    /// text beyond. An occurrence at or after `limit` may come back (a cached one), never a nearer one missed.
    fn findBefore(self: *Next, text: []const u8, p: usize, limit: usize) ?usize {
        const end = @min(limit, text.len);
        var start = p;
        if (self.known and p >= self.from) {
            if (self.at) |at| {
                if (at >= p) return at;
            } else {
                if (end <= self.to) return null;
                start = @max(p, self.to -| (self.needle.len - 1));
                const found = if (start < end) std.mem.indexOfPos(u8, text[0..end], start, self.needle) else null;
                charge((found orelse end) -| start);
                if (found != null) self.at = found else self.to = end;
                return found;
            }
        }
        self.from = p;
        self.known = true;
        self.at = if (start < end) std.mem.indexOfPos(u8, text[0..end], start, self.needle) else null;
        self.to = end;
        charge((self.at orelse end) -| p);
        return self.at;
    }
};

/// Every distinct closer spelling a family ends a call with.
const closer_spellings = blk: {
    var all: [32][]const u8 = undefined;
    var n: usize = 0;
    for (families) |f| for ([_][]const []const u8{ f.closers, f.body_closers }) |list| for (list) |c| {
        var seen = false;
        for (all[0..n]) |x| seen = seen or std.mem.eql(u8, x, c);
        if (!seen) {
            all[n] = c;
            n += 1;
        }
    };
    break :blk all[0..n].*;
};

const opener_count = blk: {
    var n: usize = 0;
    for (families) |f| n += f.openers.len;
    break :blk n;
};

/// Every opener spelling with its family, in the order the families list them: the first that matches at a byte wins.
const opener_table = blk: {
    var t: [opener_count]struct { spelling: []const u8, family: *const Family } = undefined;
    var k: usize = 0;
    for (&families) |*f| for (f.openers) |o| {
        t[k] = .{ .spelling = o, .family = f };
        k += 1;
    };
    break :blk t;
};

/// What reading one text keeps forward, so the read is linear in the text: where each opener spelling and each needle
/// an opener's reading looks for next stands, and which inline code span a byte falls in.
pub const Scan = struct {
    lt: Next = .{ .needle = "<" },
    closers: [closer_spellings.len]Next,
    parameter_eq: Next = .{ .needle = "<parameter=" },
    param: Next = .{ .needle = "<param" },
    function_close: Next = .{ .needle = "</function>" },
    gt: Next = .{ .needle = ">" },
    call_tag: Next = .{ .needle = "<tool_call" },
    dsml_invoke: Next = .{ .needle = "<｜DSML｜invoke" },
    dsml_invoke_spaced: Next = .{ .needle = "<｜DSML｜ invoke" },
    spans: Spans = .{},
    /// Reading a text for calls (the default) or looking for leaked markup to cut: the cut takes an opener alone as
    /// markup, quoted in a code span or not, whatever follows it.
    strict: bool = true,

    pub fn init() Scan {
        var scan: Scan = .{ .closers = undefined };
        for (closer_spellings, 0..) |c, i| scan.closers[i] = .{ .needle = c };
        return scan;
    }

    fn closer(self: *Scan, spelling: []const u8) *Next {
        for (&self.closers) |*n| if (std.mem.eql(u8, n.needle, spelling)) return n;
        unreachable;
    }

    fn quoted(self: *Scan, text: []const u8, at: usize) bool {
        return self.spans.read(text, at, true) == .inside;
    }
};

/// Where a byte stands against the inline code spans of its line.
pub const Quote = enum {
    outside,
    inside,
    /// A backtick no run has paired yet opens before the byte and the text may still grow: the next bytes decide.
    unresolved,
};

/// The inline code spans of one text, asked for rising bytes and kept forward. Runs of one or two backticks pair up in
/// order along the line (three or more are a fence and pair with nothing); a run no later run pairs with is a literal
/// backtick and hides nothing, so a byte is inside a span only where the closing run follows on its line.
pub const Spans = struct {
    tick: Next = .{ .needle = "`" },
    newline: Next = .{ .needle = "\n" },
    /// Where the reading stands: the first byte after the last pair it settled, and the first pair at or after it.
    cursor: usize = 0,
    found: Found = .none,
    found_known: bool = false,
    started: bool = false,

    const Found = union(enum) {
        none,
        /// An opening run with no closing run on the line yet.
        stray: usize,
        pair: Pair,
    };

    /// Where the byte at `at` stands. `complete`: no bytes follow the text; otherwise they may, so a byte after an
    /// opening run that no closing run follows on the text so far is `.unresolved`, never judged on bytes to come. A
    /// text that grows needs a new `Spans` for each look.
    pub fn read(self: *Spans, text: []const u8, at: usize, complete: bool) Quote {
        if (!self.started or at < self.cursor) {
            self.cursor = if (std.mem.lastIndexOfScalar(u8, text[0..at], '\n')) |nl| nl + 1 else 0;
            charge(at - self.cursor);
            self.found_known = false;
            self.started = true;
        }
        while (true) {
            const eol = self.newline.find(text, self.cursor) orelse text.len;
            if (eol < at) {
                self.cursor = eol + 1;
                self.found_known = false;
                continue;
            }
            if (!self.found_known) {
                self.found = self.findPair(text, eol);
                self.found_known = true;
            }
            const open_line = !complete and eol == text.len;
            switch (self.found) {
                .none => return .outside,
                .stray => |open| return if (open_line and at > open) .unresolved else .outside,
                .pair => |p| {
                    if (at < p.open) return .outside;
                    // A closing run at the end of the text may still grow into a fence.
                    if (at < p.close) return if (open_line and p.end == text.len) .unresolved else .inside;
                    self.cursor = p.end;
                    self.found_known = false;
                },
            }
        }
    }

    fn findPair(self: *Spans, text: []const u8, eol: usize) Found {
        const open = self.tickRun(text, self.cursor, eol) orelse return .none;
        const close = self.tickRun(text, open.at + open.len, eol) orelse return .{ .stray = open.at };
        return .{ .pair = .{ .open = open.at, .close = close.at, .end = close.at + close.len } };
    }

    /// The first run of one or two backticks in `[from, eol)`.
    fn tickRun(self: *Spans, text: []const u8, from: usize, eol: usize) ?struct { at: usize, len: usize } {
        var p = from;
        while (self.tick.find(text, p)) |t| {
            if (t >= eol) return null;
            var len: usize = 1;
            while (t + len < text.len and text[t + len] == '`') len += 1;
            if (len < 3) return .{ .at = t, .len = len };
            p = t + len;
        }
        return null;
    }
};

const Pair = struct { open: usize, close: usize, end: usize };

/// The first call opener at or after `from`.
pub fn nextOpener(text: []const u8, from: usize, scan: *Scan) ?Opener {
    var p = from;
    while (scan.lt.find(text, p)) |at| : (p = at + 1) {
        for (opener_table) |o| {
            if (std.mem.startsWith(u8, text[at..], o.spelling) and opensCall(o.family, text, at, scan)) return .{ .at = at, .len = o.spelling.len, .family = o.family };
        }
    }
    return null;
}

/// Where a call ends: just past its closer, or, for a call that lost a close, where the next
/// call opens.
const End = struct { at: usize, lost: bool = false };

/// Where the call of `f` that opens at `at` ends: its first closer outside every parameter value, as the parser
/// ends it. A closer a value is still open at, that the next call's opener follows, belongs to a call that lost
/// its value's close: the value never runs across a call. Null when the text stops before an end. `model_ended`
/// is the model's own end of the generation: a wrapper whose function closed is whole then.
fn callEnd(f: *const Family, text: []const u8, at: usize, model_ended: bool, scan: *Scan) ?End {
    var best: ?End = null;
    for (f.closers) |c| best = earlierEnd(best, firstEnd(f, text, at, c, best, scan));
    if (model_ended and f.body_closers.len > 0 and bodyIsFunction(text, at)) {
        for (f.body_closers) |c| best = earlierEnd(best, firstEnd(f, text, at, c, best, scan));
    }
    if (f.values == .wrapper) {
        if (nextArgCallOpener(text, at, if (best) |b| b.at else text.len, scan)) |next| return .{ .at = next, .lost = true };
    }
    return best;
}

/// Where the `<arg_key>` call that opens at `at` stops because another call with arguments opens before `limit`,
/// whether a value is still open there or the call only lost its own close: a call never runs across a call. A call
/// that spells no `<arg_key>` (`<tool_call>x</tool_call>`) in its value does not end it. Hy4's wrapper ends at its
/// second call: its first stands in its body.
fn nextArgCallOpener(text: []const u8, at: usize, limit: usize, scan: *Scan) ?usize {
    const body = argKeysBody(text, at) orelse return null;
    var p = at + 1;
    while (scan.call_tag.findBefore(text, p, limit)) |q| : (p = q + 1) {
        if (q >= limit) return null;
        if (q != body and opensArgKeyCall(text, q)) return q;
    }
    return null;
}

/// Where the body of the call that opens at `at` starts when it is GLM's `<arg_key>` body or Hy4's wrapper around
/// calls, not a JSON object or Hermes' function tag.
fn argKeysBody(text: []const u8, at: usize) ?usize {
    const gt = std.mem.indexOfScalarPos(u8, text, at, '>') orelse return null;
    charge(gt - at);
    const body = skipSpace(text, gt + 1);
    const rest = text[body..];
    return if (std.mem.startsWith(u8, rest, "{") or std.mem.startsWith(u8, rest, "<function=")) null else body;
}

/// `<tool_call>NAME<arg_key` at `q`: a call that carries an argument.
fn opensArgKeyCall(text: []const u8, q: usize) bool {
    var i = q + (think_tags.callTagLenAt(text[q..], "<tool_call") orelse return false);
    i = skipSpace(text, i);
    const name_at = i;
    while (i < text.len and (std.ascii.isAlphanumeric(text[i]) or text[i] == '_' or text[i] == '-' or text[i] == '.')) i += 1;
    return i > name_at and std.mem.startsWith(u8, text[skipSpace(text, i)..], "<arg_key");
}

fn earliestOf(a: ?usize, b: ?usize) ?usize {
    const x = a orelse return b;
    const y = b orelse return a;
    return @min(x, y);
}

fn earlierEnd(a: ?End, b: ?End) ?End {
    const x = a orelse return b;
    const y = b orelse return a;
    return if (y.at < x.at) y else x;
}

/// The first end of the call at `at` that `closer` makes, unless `best` ends it sooner. A closer a value spells is
/// skipped with the rest of that value (`closerValueEnd`): the value is judged once, however many closers it spells.
fn firstEnd(f: *const Family, text: []const u8, at: usize, closer: []const u8, best: ?End, scan: *Scan) ?End {
    const limit = if (best) |b| b.at else text.len;
    const next = scan.closer(closer);
    const exact = valueQuotesAreExact(f, text, at);
    var p = at;
    var inside_until: usize = 0;
    while (next.findBefore(text, p, limit)) |c| {
        if (c >= limit) return null;
        p = c + 1;
        if (c >= inside_until) {
            charge(c - at);
            inside_until = tool_calls.closerValueEnd(f.values, text, at, c) orelse return .{ .at = c + closer.len };
            charge(@min(inside_until, text.len) -| c);
        }
        if (exact) {
            p = @max(p, inside_until);
        } else if (nextCallAfterClosers(f, text, c + closer.len, scan)) |after| {
            return .{ .at = after, .lost = true };
        }
    }
    return null;
}

/// Whether a value of the call at `at` ends where its quotes say (a JSON string, Gemma's `<|"|>`): such a value
/// is never judged by the call that follows it.
fn valueQuotesAreExact(f: *const Family, text: []const u8, at: usize) bool {
    return switch (f.values) {
        .gemma => true,
        .wrapper => blk: {
            const gt = std.mem.indexOfScalarPos(u8, text, at, '>') orelse break :blk false;
            charge(gt - at);
            break :blk std.mem.startsWith(u8, std.mem.trimStart(u8, text[gt + 1 ..], " \t\r\n"), "{");
        },
        else => false,
    };
}

/// Where the next call opens when only whitespace and closers of `f` stand between `from` and it.
fn nextCallAfterClosers(f: *const Family, text: []const u8, from: usize, scan: *Scan) ?usize {
    var p = skipSpace(text, from);
    while (closerAt(f, text, p)) |c| p = skipSpace(text, p + c.len);
    for (&families) |*g| for (g.openers) |o| {
        if (std.mem.startsWith(u8, text[p..], o) and opensCall(g, text, p, scan)) return p;
    };
    return null;
}

fn closerAt(f: *const Family, text: []const u8, p: usize) ?[]const u8 {
    for (f.closers) |c| if (std.mem.startsWith(u8, text[p..], c)) return c;
    for (f.body_closers) |c| if (std.mem.startsWith(u8, text[p..], c)) return c;
    return null;
}

fn skipSpace(text: []const u8, from: usize) usize {
    var i = from;
    while (i < text.len and std.ascii.isWhitespace(text[i])) i += 1;
    return i;
}

/// Whether the wrapper that opens at `at` carries Hermes' `<function=` body.
fn bodyIsFunction(text: []const u8, at: usize) bool {
    const gt = std.mem.indexOfScalarPos(u8, text, at, '>') orelse return false;
    charge(gt - at);
    return std.mem.startsWith(u8, std.mem.trimStart(u8, text[gt + 1 ..], " \t\r\n"), "<function=");
}

/// A call of the text and where it ends (null: the text stops inside it).
/// `lost`: the call lost the close of its last value, so it ends where the next call opens.
pub const Call = struct { start: usize, open_len: usize, end: ?usize, lost: bool = false, family: *const Family };

/// The calls of a text, in order, outside every thought that closed. A thought still open at the end of the text
/// holds calls as the rest of the text does; the one the prompt opened ends at its first close.
pub const Reader = struct {
    text: []const u8,
    pos: usize = 0,
    started: bool = false,
    model_ended: bool,
    scan: Scan = Scan.init(),
    tags: think_tags.Tags = .{},
    /// The first thought opener at or after `think_from`, asked once and kept while the reading stands between the two.
    think: ?think_tags.ThinkOpen = null,
    think_from: usize = 0,
    think_known: bool = false,

    pub fn init(text: []const u8, model_ended: bool) Reader {
        return .{ .text = text, .model_ended = model_ended };
    }

    /// The first thought opener at or after the reading's place that no call spans. A thought is found once for the calls
    /// the reading steps over on its way to it, never once per call.
    fn thinkOpen(self: *Reader) ?think_tags.ThinkOpen {
        const stale = !self.think_known or self.pos < self.think_from or (if (self.think) |t| self.pos > t.pos else false);
        if (stale) {
            self.think = think_tags.nextThinkOpenIn(&self.tags, self.text, self.pos);
            self.think_from = self.pos;
            self.think_known = true;
        }
        return self.think;
    }

    pub fn next(self: *Reader) ?Call {
        if (!self.started) {
            self.started = true;
            if (leadingThoughtEnd(self.text, self.model_ended, &self.tags)) |end| self.pos = end;
        }
        while (true) {
            const opener = nextOpener(self.text, self.pos, &self.scan);
            if (self.thinkOpen()) |t| {
                if (opener == null or t.pos < opener.?.at) {
                    const style: Style = if (t.is_think_style) .think else .channel;
                    self.pos = if (thoughtClose(self.text, t.after, self.model_ended, style, &self.tags)) |c| c.pos + c.len else t.after;
                    continue;
                }
            }
            const o = opener orelse return null;
            switch (readAt(self.text, o, self.model_ended, &self.scan)) {
                .call => |c| {
                    self.pos = c.end orelse self.text.len;
                    return c;
                },
                .enter => self.pos = o.at + o.len,
                .mention => self.pos = o.at + o.len,
            }
        }
    }
};

const Step = union(enum) {
    call: Call,
    /// A container with calls inside: they are read where they stand.
    enter,
    /// An opener quoted in an inline code span that no closer follows: a model explaining its format, not a call.
    mention,
};

/// What the opener `o` starts, read as the parser reads it. Only an unfinished call quoted in a code span is a mention:
/// a call that ends is a call wherever it is written.
fn readAt(text: []const u8, o: Opener, model_ended: bool, scan: *Scan) Step {
    const end = callEnd(o.family, text, o.at, model_ended, scan);
    if (o.family.container) {
        const inner = nextOpener(text, o.at + o.len, scan);
        const whole = end != null or model_ended;
        if (inner != null and whole and (end == null or inner.?.at < end.?.at)) return .enter;
    }
    if (end == null and scan.strict and scan.quoted(text, o.at)) return .mention;
    return .{ .call = .{ .start = o.at, .open_len = o.len, .end = if (end) |e| e.at else null, .lost = if (end) |e| e.lost else false, .family = o.family } };
}

/// The first call of the text at or after `from`, read as `Reader` reads it but with no thought skipped.
fn nextCall(text: []const u8, from: usize, model_ended: bool, scan: *Scan) ?Call {
    var pos = from;
    while (nextOpener(text, pos, scan)) |o| switch (readAt(text, o, model_ended, scan)) {
        .call => |c| return c,
        .enter, .mention => pos = o.at + o.len,
    };
    return null;
}

/// Where the call the text leaves open starts: the first call that has no end. Null when the text stops outside one.
pub fn openCallStart(text: []const u8, model_ended: bool) ?usize {
    var reader = Reader.init(text, model_ended);
    while (reader.next()) |c| if (c.end == null) return c.start;
    return null;
}

/// A call the generation did not finish: it stops inside the call, or lost the close of its last value.
pub const Defect = struct { start: usize, end: usize };

/// The unfinished calls of the text, in order: the parser reads the text without them and the reply names the first.
pub fn defects(allocator: std.mem.Allocator, text: []const u8, model_ended: bool) ![]Defect {
    var out = std.ArrayList(Defect).empty;
    errdefer out.deinit(allocator);
    var reader = Reader.init(text, model_ended);
    while (reader.next()) |c| {
        if (c.end == null) try out.append(allocator, .{ .start = c.start, .end = text.len });
        if (c.lost) try out.append(allocator, .{ .start = c.start, .end = c.end.? });
    }
    return out.toOwnedSlice(allocator);
}

const Style = enum { any, think, channel };

/// The first `tag` at or after `from` that no call's value holds: a think or channel tag inside a call is the call's
/// payload, wherever its closer stands.
pub fn outsideCalls(text: []const u8, from: usize, tag: think_tags.Tag) ?think_tags.TagAt {
    var tags: think_tags.Tags = .{};
    return firstOutsideCalls(text, from, true, &tags, tag);
}

/// A thought's close at or after `from` that no call spans: a call that opens before it and ends after it holds it
/// as payload, and so does a call the text stops inside unless a thought is open where it starts: that thought's text may
/// mention an opener.
fn thoughtClose(text: []const u8, from: usize, model_ended: bool, style: Style, tags: *think_tags.Tags) ?think_tags.TagAt {
    const think = if (style == .channel) null else firstOutsideCalls(text, from, model_ended, tags, .think_close);
    const channel = if (style == .think) null else firstOutsideCalls(text, from, model_ended, tags, .channel_close);
    const t = think orelse return channel;
    const c = channel orelse return t;
    return if (c.pos < t.pos) c else t;
}

/// The first `tag` of `tags` (the searches of this one text) at or after `from` that no call spans: a call that opens
/// before it and ends after it holds it as payload.
pub fn firstOutsideCalls(text: []const u8, from: usize, model_ended: bool, tags: *think_tags.Tags, tag: think_tags.Tag) ?think_tags.TagAt {
    var i = from;
    var q = from;
    var scan = Scan.init();
    while (tags.find(tag, text, i)) |m| {
        while (true) {
            const c = nextCall(text, q, model_ended, &scan) orelse return m;
            if (c.start >= m.pos) return m;
            const end = c.end orelse {
                // A thought's own text may mention an opener; a call cut outside any thought holds every tag after it.
                if (!thoughtOpenAt(text, c.start)) return null;
                q = c.start + c.open_len;
                continue;
            };
            q = end;
            if (end > m.pos) {
                i = end;
                break;
            }
        }
    }
    return null;
}

/// Whether a thought has opened and not closed by `at`, by the tags alone.
fn thoughtOpenAt(text: []const u8, at: usize) bool {
    const head = text[0..at];
    var tags: think_tags.Tags = .{};
    var open: ?usize = null;
    var close: ?usize = null;
    inline for (.{ .{ think_tags.Tag.think_open, think_tags.Tag.think_close }, .{ think_tags.Tag.channel_thought, think_tags.Tag.channel_close } }) |kind| {
        var p: usize = 0;
        while (tags.find(kind[0], head, p)) |t| : (p = t.pos + t.len) open = @max(open orelse 0, t.pos);
        p = 0;
        while (tags.find(kind[1], head, p)) |t| : (p = t.pos + t.len) close = @max(close orelse 0, t.pos);
    }
    const o = open orelse return false;
    return close == null or close.? < o;
}

/// Where the thought the text starts with ends (just past its close): a close before any opener is the template-opened
/// thought's.
fn leadingThoughtEnd(text: []const u8, model_ended: bool, tags: *think_tags.Tags) ?usize {
    const c = thoughtClose(text, 0, model_ended, .any, tags) orelse return null;
    if (think_tags.nextThinkOpenIn(tags, text, 0)) |o| if (o.pos < c.pos) return null;
    return c.pos + c.len;
}

/// `text` with every call the reading finds that `keep` does not accept blanked to spaces (the bytes keep their
/// places), or null when nothing is blanked: a family's parser then reads only its own calls, and a value that spells
/// another family's markup never decides the call. The caller frees the copy.
pub fn maskedExcept(allocator: std.mem.Allocator, text: []const u8, keep: *const fn (text: []const u8, call: Call) bool) !?[]u8 {
    var out: ?[]u8 = null;
    errdefer if (out) |o| allocator.free(o);
    var reader = Reader.init(text, true);
    while (reader.next()) |c| {
        if (keep(text, c)) continue;
        if (out == null) out = try allocator.dupe(u8, text);
        @memset(out.?[c.start .. c.end orelse text.len], ' ');
    }
    return out;
}

/// A call of DeepSeek's DSML dialect.
pub fn isDsmlCall(_: []const u8, call: Call) bool {
    return call.family.values == .dsml;
}

/// A call of GLM's `<tool_call>NAME<arg_key>…` dialect: a bare `<tool_call>` whose body is neither JSON nor a
/// `<function=` tag.
pub fn isArgKeyCall(text: []const u8, call: Call) bool {
    if (call.family != &families[1] and call.family != &families[families.len - 1]) return false;
    const gt = std.mem.indexOfScalarPos(u8, text, call.start, '>') orelse return true;
    const body = std.mem.trimStart(u8, text[gt + 1 ..], " \t\r\n");
    return !(std.mem.startsWith(u8, body, "{") or std.mem.startsWith(u8, body, "<function="));
}

/// Markup no call dialect owns that is never prose: DeepSeek's DSML tags and a closer of theirs with no opener.
pub const lone_markup = [_][]const u8{ "<｜DSML｜", "</｜DSML｜" };

/// Where unparsed call markup starts in `text`: the first call opener of any family (one whose opener alone is prose
/// counts where its own reading holds, a wrapper's opener always) or DSML markup; null with none.
pub fn markupStart(text: []const u8) ?usize {
    var scan = Scan.init();
    scan.strict = false;
    var cut: ?usize = if (nextOpener(text, 0, &scan)) |o| o.at else null;
    for (lone_markup) |m| if (std.mem.indexOf(u8, text, m)) |p| {
        if (cut == null or p < cut.?) cut = p;
    };
    return cut;
}

/// Whether `tail` is the start of an opener (or DSML markup) that more bytes may complete.
pub fn openerMayGrow(tail: []const u8) bool {
    for (lone_markup) |m| if (tail.len < m.len and std.mem.startsWith(u8, m, tail)) return true;
    for (families) |f| for (f.openers) |o| if (tail.len < o.len and std.mem.startsWith(u8, o, tail)) return true;
    return false;
}

/// The tool the `name="…"` attribute of the opener at `start` names (`<tool name="x">`, `<tool_call name="x">`), for a
/// call cut before its body gave the parser a name. Null when the opener has no whole tag or no plain name.
pub fn attributeName(text: []const u8, start: usize) ?[]const u8 {
    const tail = text[start..];
    const gt = std.mem.indexOfScalar(u8, tail[0..@min(tail.len, 512)], '>') orelse return null;
    const attrs_at = std.mem.indexOfAny(u8, tail[0..gt], " \t\n") orelse return null;
    const name = tool_calls.miniCpm5AttrValue(tail[attrs_at..gt], "name") orelse return null;
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "<>\r\n") != null) return null;
    return name;
}
