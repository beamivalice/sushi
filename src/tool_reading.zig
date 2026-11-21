//! The one reading of a generation's tool-call text. Thoughts, calls and the spans of a call's values are read
//! here once, by the served dialect's own value ends (`tool_calls.closerInsideValue`), so the open-call finder
//! and every think-tag pass agree on which text is structure and which is a call's payload.
const std = @import("std");
const think_tags = @import("think_tags.zig");
const tool_calls = @import("tool_calls.zig");
const BARE_INVOKE = tool_calls.BARE_INVOKE;
const FUNCTION_CALLS_TAG = tool_calls.FUNCTION_CALLS_TAG;

/// One call dialect: the openers that may start a call, the closers that end it, and for a dialect whose opener
/// alone is prose, what the text from the opener on must hold for the parser to read a call there.
pub const Family = struct {
    openers: []const []const u8,
    closers: []const []const u8,
    /// Spellings that begin with an opener and are another dialect's closer.
    not_openers: []const []const u8 = &.{},
    reads_as_call: ?*const fn (tail: []const u8) bool = null,
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
    .{ .openers = &.{"<|tool_call"}, .closers = &.{"<tool_call|>"}, .reads_as_call = gemmaReadsAsCall, .values = .gemma },
    .{ .openers = &.{"<tool_call"}, .closers = &hermes_closers, .not_openers = &.{"<tool_call|>"}, .reads_as_call = wrapperReadsAsCall, .values = .wrapper, .body_closers = &.{"</function>"} },
    .{ .openers = &.{ "<tool>", "<tool ", "<tool\t", "<tool\n", "<tool_request" }, .closers = &hermes_closers, .reads_as_call = genericToolTagReadsAsCall, .values = .wrapper, .body_closers = &.{"</function>"} },
    .{ .openers = &.{"<｜DSML｜tool_calls"}, .closers = &.{"</｜DSML｜tool_calls>"}, .values = .dsml, .container = true },
    .{ .openers = &.{ "<｜DSML｜invoke", "<｜DSML｜ invoke" }, .closers = &.{ "</｜DSML｜invoke>", "</｜DSML｜ invoke>" }, .values = .dsml },
    .{ .openers = &.{FUNCTION_CALLS_TAG}, .closers = &.{"</function_calls>"}, .values = .invoke, .container = true },
    .{ .openers = &.{BARE_INVOKE.invoke}, .closers = &.{BARE_INVOKE.invoke_close}, .reads_as_call = invokeReadsAsCall, .values = .invoke },
    .{ .openers = &.{"<function"}, .closers = &.{"</function>"}, .reads_as_call = functionReadsAsCall, .values = .function_tag },
};

/// `<|tool_call>call:name{…`: the body starts with `call:`, or has not got that far.
fn gemmaReadsAsCall(tail: []const u8) bool {
    const rest = tail["<|tool_call".len..];
    if (rest.len == 0) return true;
    if (rest[0] != '>') return false;
    const body = std.mem.trimStart(u8, rest[1..], " \t\r\n");
    return std.mem.startsWith(u8, body, "call:") or std.mem.startsWith(u8, "call:", body);
}

/// A bare `<tool_call>` opens a call when a JSON object, a tag or a tool name leads its body, or the text stops
/// there: `I could emit <tool_call> here` is prose. Another spelling (`<tool_call name=`, `<tool_calls>`) is a call.
fn wrapperReadsAsCall(tail: []const u8) bool {
    const open = "<tool_call";
    if (tail.len <= open.len or tail[open.len] != '>') return true;
    const body = std.mem.trimStart(u8, tail[open.len + 1 ..], " \t\r\n");
    if (body.len == 0 or body[0] == '{' or body[0] == '<') return true;
    var name: usize = 0;
    while (name < body.len and (std.ascii.isAlphanumeric(body[name]) or body[name] == '_' or body[name] == '-' or body[name] == '.')) name += 1;
    return name > 0 and (name == body.len or body[name] == '<');
}

/// `<tool name="x">{…`, `<tool>{"name":…` and `<tool_request>{…`: the tag's `>` opens a body that is a JSON object or Hermes' function tag.
/// A self-closing `<tool …/>` is a whole call, never an open one.
fn genericToolTagReadsAsCall(tail: []const u8) bool {
    const gt = std.mem.indexOfScalar(u8, tail[0..@min(tail.len, 256)], '>') orelse return false;
    if (tail[gt - 1] == '/') return false;
    const body = std.mem.trimStart(u8, tail[gt + 1 ..], " \t\n\r");
    if (std.mem.startsWith(u8, body, "{")) return std.mem.startsWith(u8, std.mem.trimStart(u8, body[1..], " \t\n\r"), "\"");
    return std.mem.startsWith(u8, body, "<function=");
}

/// `<invoke name="x">` with a `<parameter ` of its own before the next `<invoke`: the opener alone is prose.
fn invokeReadsAsCall(tail: []const u8) bool {
    const after = BARE_INVOKE.invoke.len;
    if (tail.len <= after or !std.ascii.isWhitespace(tail[after])) return false;
    const name = tool_calls.dsmlAttr(tail[after..@min(tail.len, after + 512)], "name") orelse return false;
    if (name.len == 0) return false;
    const own = tail[after .. std.mem.indexOfPos(u8, tail, after, BARE_INVOKE.invoke) orelse tail.len];
    return std.mem.indexOf(u8, own, BARE_INVOKE.param ++ " ") != null;
}

/// MiniCPM5's `<function name="x">` with a `<param` after its tag, and Hermes' bare `<function=x>` with a
/// `<parameter=`: either opener alone is prose.
fn functionReadsAsCall(tail: []const u8) bool {
    const after = "<function".len;
    if (tail.len <= after) return false;
    if (tail[after] == '=') return std.mem.indexOf(u8, tail, "<parameter=") != null;
    if (!std.ascii.isWhitespace(tail[after])) return false;
    const gt = std.mem.indexOfScalarPos(u8, tail, after, '>') orelse return false;
    const name = tool_calls.miniCpm5AttrValue(tail[after..gt], "name") orelse return false;
    return name.len > 0 and std.mem.indexOf(u8, tail[gt + 1 ..], "<param") != null;
}

/// Whether `tail`, which starts with an opener of `f`, opens a call there: no other dialect's closer, and the
/// dialect's own reading accepts it.
fn opensCall(f: *const Family, tail: []const u8) bool {
    for (f.not_openers) |n| if (std.mem.startsWith(u8, tail, n)) return false;
    return if (f.reads_as_call) |reads| reads(tail) else true;
}

/// A call opener found in a text.
pub const Opener = struct { at: usize, len: usize, family: *const Family };

/// The first call opener at or after `from`. An opener inside an inline code span is text: a model explaining its
/// format quotes it.
pub fn nextOpener(text: []const u8, from: usize) ?Opener {
    var best: ?Opener = null;
    for (&families) |*f| for (f.openers) |o| {
        var p = from;
        while (std.mem.indexOfPos(u8, text, p, o)) |at| : (p = at + 1) {
            if (best != null and at >= best.?.at) break;
            if (!opensCall(f, text[at..])) continue;
            if (think_tags.openCodeSpanStart(text[0..at]) != null) continue;
            best = .{ .at = at, .len = o.len, .family = f };
            break;
        }
    };
    return best;
}

/// Where a call ends: just past its closer, or, for a call that lost the close of its last value, where the next
/// call opens.
const End = struct { at: usize, lost: bool = false };

/// Where the call of `f` that opens at `at` ends: its first closer outside every parameter value, as the parser
/// ends it. A closer a value is still open at, that the next call's opener follows, belongs to a call that lost
/// its value's close: the value never runs across a call. Null when the text stops before an end. `model_ended`
/// is the model's own end of the generation: a wrapper whose function closed is whole then.
fn callEnd(f: *const Family, text: []const u8, at: usize, model_ended: bool) ?End {
    var best: ?End = null;
    for (f.closers) |c| best = earlierEnd(best, firstEnd(f, text, at, c));
    if (model_ended and f.body_closers.len > 0 and bodyIsFunction(text, at)) {
        for (f.body_closers) |c| best = earlierEnd(best, firstEnd(f, text, at, c));
    }
    return best;
}

fn earlierEnd(a: ?End, b: ?End) ?End {
    const x = a orelse return b;
    const y = b orelse return a;
    return if (y.at < x.at) y else x;
}

fn firstEnd(f: *const Family, text: []const u8, at: usize, closer: []const u8) ?End {
    var p = at;
    while (std.mem.indexOfPos(u8, text, p, closer)) |c| : (p = c + 1) {
        if (!tool_calls.closerInsideValue(f.values, text, at, c)) return .{ .at = c + closer.len };
        if (valueQuotesAreExact(f, text, at)) continue;
        if (nextCallAfterClosers(f, text, c + closer.len)) |next| return .{ .at = next, .lost = true };
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
            break :blk std.mem.startsWith(u8, std.mem.trimStart(u8, text[gt + 1 ..], " \t\r\n"), "{");
        },
        else => false,
    };
}

/// Where the next call opens when only whitespace and closers of `f` stand between `from` and it.
fn nextCallAfterClosers(f: *const Family, text: []const u8, from: usize) ?usize {
    var p = skipSpace(text, from);
    while (closerAt(f, text, p)) |c| p = skipSpace(text, p + c.len);
    for (&families) |*g| for (g.openers) |o| {
        if (std.mem.startsWith(u8, text[p..], o) and opensCall(g, text[p..])) return p;
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
    return std.mem.startsWith(u8, std.mem.trimStart(u8, text[gt + 1 ..], " \t\r\n"), "<function=");
}

/// A call of the text and where it ends (null: the text stops inside it).
/// `lost`: the call lost the close of its last value, so it ends where the next call opens.
pub const Call = struct { start: usize, end: ?usize, lost: bool = false, family: *const Family };

/// The calls of a text, in order, outside every thought that closed. A thought still open at the end of the text
/// holds calls as the rest of the text does; the one the prompt opened ends at its first close.
pub const Reader = struct {
    text: []const u8,
    pos: usize = 0,
    started: bool = false,
    model_ended: bool,

    pub fn init(text: []const u8, model_ended: bool) Reader {
        return .{ .text = text, .model_ended = model_ended };
    }

    pub fn next(self: *Reader) ?Call {
        if (!self.started) {
            self.started = true;
            if (leadingThoughtEnd(self.text, self.model_ended)) |end| self.pos = end;
        }
        while (true) {
            const opener = nextOpener(self.text, self.pos);
            if (think_tags.nextThinkOpen(self.text, self.pos)) |t| {
                if (opener == null or t.pos < opener.?.at) {
                    const style: Style = if (t.is_think_style) .think else .channel;
                    self.pos = if (thoughtClose(self.text, t.after, self.model_ended, style)) |c| c.pos + c.len else t.after;
                    continue;
                }
            }
            const o = opener orelse return null;
            const end = callEnd(o.family, self.text, o.at, self.model_ended);
            if (o.family.container) {
                const inner = nextOpener(self.text, o.at + o.len);
                const whole = end != null or self.model_ended;
                if (inner != null and whole and (end == null or inner.?.at < end.?.at)) {
                    self.pos = o.at + o.len;
                    continue;
                }
            }
            self.pos = if (end) |e| e.at else self.text.len;
            return .{ .start = o.at, .end = if (end) |e| e.at else null, .lost = if (end) |e| e.lost else false, .family = o.family };
        }
    }
};

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

/// A thought's close at or after `from` that no call spans: a call that opens before it and ends after it holds it
/// as payload. A call that never ends is leaked markup, not a span.
fn thoughtClose(text: []const u8, from: usize, model_ended: bool, style: Style) ?think_tags.TagAt {
    const think = if (style == .channel) null else firstOutsideCalls(text, from, model_ended, think_tags.rawThinkCloseAt);
    const channel = if (style == .think) null else firstOutsideCalls(text, from, model_ended, channelCloseAt);
    const t = think orelse return channel;
    const c = channel orelse return t;
    return if (c.pos < t.pos) c else t;
}

fn channelCloseAt(text: []const u8, from: usize) ?think_tags.TagAt {
    const p = std.mem.indexOfPos(u8, text, from, "<channel|>") orelse return null;
    return .{ .pos = p, .len = "<channel|>".len };
}

fn firstOutsideCalls(text: []const u8, from: usize, model_ended: bool, find: *const fn ([]const u8, usize) ?think_tags.TagAt) ?think_tags.TagAt {
    var i = from;
    outer: while (find(text, i)) |m| {
        var q = i;
        while (nextOpener(text, q)) |o| {
            if (o.at >= m.pos) break;
            const end = (callEnd(o.family, text, o.at, model_ended) orelse {
                q = o.at + o.len;
                continue;
            }).at;
            if (end > m.pos) {
                i = end;
                continue :outer;
            }
            q = end;
        }
        return m;
    }
    return null;
}

/// Where the thought the text starts with ends (just past its close): a close before any opener is the template-opened
/// thought's.
fn leadingThoughtEnd(text: []const u8, model_ended: bool) ?usize {
    const c = thoughtClose(text, 0, model_ended, .any) orelse return null;
    if (think_tags.nextThinkOpen(text, 0)) |o| if (o.pos < c.pos) return null;
    return c.pos + c.len;
}
