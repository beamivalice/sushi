//! The filters, tests, methods, globals and operators a template may use, each with the meaning Python Jinja2
//! gives it under transformers' chat-template environment. A name outside these sets is refused when the template
//! is parsed (filters and tests) or when it is looked up (methods). The set is jinja.cpp's (wangzhaode/jinja.cpp,
//! Apache-2.0, see NOTICE), and the names only it has (the `strip` filter, `str.indent`, `list.append`) follow its
//! routines in `lib/jinja_cpp/value.cpp`.
const std = @import("std");
const value = @import("value.zig");
const tojson = @import("tojson.zig");
const render = @import("render.zig");
const ast = @import("ast.zig");

const Value = value.Value;
const Str = value.Str;
const Failure = value.Failure;
const Renderer = render.Renderer;
const Kw = render.Kw;

pub const Filter = enum {
    abs,
    capitalize,
    default,
    first,
    float,
    format,
    indent,
    int,
    items,
    join,
    last,
    length,
    list,
    lower,
    map,
    max,
    min,
    reject,
    rejectattr,
    replace,
    reverse,
    safe,
    select,
    selectattr,
    sort,
    string,
    /// Not a Jinja2 filter: jinja.cpp's, kept with `str.strip`'s meaning.
    strip,
    sum,
    title,
    tojson,
    trim,
    truncate,
    unique,
    upper,
    wordcount,
};

pub const Test = enum {
    boolean,
    callable,
    defined,
    divisibleby,
    eq,
    equalto,
    escaped,
    even,
    @"false",
    filter,
    float,
    ge,
    gt,
    greaterthan,
    in,
    integer,
    iterable,
    lower,
    lt,
    lessthan,
    mapping,
    ne,
    none,
    number,
    odd,
    sameas,
    sequence,
    string,
    @"test",
    @"true",
    undefined,
    upper,
};

/// The names Jinja2's environment holds, for the `filter` and `test` tests: what Python answers, whether or not
/// this engine implements the name.
const jinja2_filters = [_][]const u8{
    "abs",     "attr",      "batch",   "capitalize", "center",     "count",   "d",       "default",  "dictsort",
    "e",       "escape",    "filesizeformat",        "first",      "float",   "forceescape", "format", "groupby",
    "indent",  "int",       "join",    "last",       "length",     "list",    "lower",   "items",    "map",
    "min",     "max",       "pprint",  "random",     "reject",     "rejectattr", "replace", "reverse", "round",
    "safe",    "select",    "selectattr", "slice",   "sort",       "string",  "striptags", "sum",    "title",
    "trim",    "truncate",  "unique",  "upper",      "urlencode",  "urlize",  "wordcount", "wordwrap", "xmlattr",
    "tojson",
};
const jinja2_tests = [_][]const u8{
    "odd",     "even",    "divisibleby", "defined", "undefined", "filter",  "test",     "none",    "boolean",
    "false",   "true",    "integer",     "float",   "lower",     "upper",   "string",   "mapping", "number",
    "sequence", "iterable", "callable",  "sameas",  "escaped",   "in",      "==",       "eq",      "equalto",
    "!=",      "ne",      ">",           "gt",      "greaterthan", "ge",    ">=",       "<",       "lt",
    "lessthan", "<=",     "le",
};

/// A test as `select` and friends name it, operator spellings included.
fn testByName(name: []const u8) ?Test {
    @setRuntimeSafety(true);
    const aliases = [_]struct { []const u8, Test }{ .{ "==", .eq }, .{ "!=", .ne }, .{ ">", .gt }, .{ ">=", .ge }, .{ "<", .lt } };
    for (aliases) |a| if (std.mem.eql(u8, a[0], name)) return a[1];
    return std.meta.stringToEnum(Test, name);
}

// ── Arguments ──

/// Python's binding of `args` and `kws` to the parameters `names`; a parameter not given is null.
fn bind(r: *Renderer, what: []const u8, comptime names: []const []const u8, args: []const Value, kws: []const Kw) Failure![names.len]?Value {
    @setRuntimeSafety(true);
    var out: [names.len]?Value = @splat(null);
    if (args.len > names.len) return r.fail("{s}() takes at most {d} argument(s) ({d} given)", .{ what, names.len, args.len });
    if (comptime names.len == 0) {
        if (kws.len > 0) return r.fail("{s}() got an unexpected keyword argument '{s}'", .{ what, kws[0].name });
        return out;
    }
    for (args, 0..) |a, i| out[i] = a;
    for (kws) |kw| {
        const i = for (names, 0..) |n, i| {
            if (std.mem.eql(u8, n, kw.name)) break i;
        } else return r.fail("{s}() got an unexpected keyword argument '{s}'", .{ what, kw.name });
        if (out[i] != null) return r.fail("{s}() got multiple values for argument '{s}'", .{ what, kw.name });
        out[i] = kw.value;
    }
    return out;
}

fn required(r: *Renderer, what: []const u8, v: ?Value) Failure!Value {
    @setRuntimeSafety(true);
    return v orelse r.fail("{s}() is missing a required argument", .{what});
}

fn intArg(r: *Renderer, v: Value, what: []const u8) Failure!i64 {
    @setRuntimeSafety(true);
    return switch (v) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        .undefined => |n| r.fail("'{s}' is undefined", .{n}),
        else => r.fail("{s}: '{s}' object cannot be interpreted as an integer", .{ what, value.typeName(v) }),
    };
}

fn strArg(r: *Renderer, v: Value, what: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    return v.str() orelse r.fail("{s}: expected a string, got '{s}'", .{ what, value.typeName(v) });
}

/// `text` with the string kind of `like`: a Markup operand gives Markup.
fn sameKind(like: Value, text: Str) Value {
    @setRuntimeSafety(true);
    return if (like == .markup) .{ .markup = text } else .{ .string = text };
}

/// `sub`, a slice of `src`'s bytes, with the origins those bytes have there.
fn within(r: *Renderer, src: Str, sub: []const u8) Failure!Str {
    @setRuntimeSafety(true);
    if (sub.len == 0) return .{ .bytes = sub, .origin = src.origin };
    const at = @intFromPtr(sub.ptr) - @intFromPtr(src.bytes.ptr);
    std.debug.assert(at + sub.len <= src.bytes.len);
    return src.slice(r.arena, at, at + sub.len);
}

fn piece(r: *Renderer, src: Str, sub: []const u8) Failure!Value {
    @setRuntimeSafety(true);
    return .{ .string = try within(r, src, sub) };
}

fn undefinedFail(r: *Renderer, v: Value) Failure!void {
    @setRuntimeSafety(true);
    if (v == .undefined) return r.fail("'{s}' is undefined", .{if (v.undefined.len > 0) v.undefined else "value"});
}

// ── Strings ──

fn decode(cs: []const u8) u21 {
    @setRuntimeSafety(true);
    return std.unicode.utf8Decode(cs) catch 0xFFFD;
}

/// ASCII case mapping; other code points are left as they are.
fn mapCase(r: *Renderer, s: []const u8, comptime upper: bool) Failure![]const u8 {
    @setRuntimeSafety(true);
    try r.charge(s.len);
    const out = try r.arena.alloc(u8, s.len);
    for (s, out) |c, *o| o.* = if (upper) std.ascii.toUpper(c) else std.ascii.toLower(c);
    return out;
}

/// Python `str.capitalize`: the first character upper case, the rest lower.
fn capitalize(r: *Renderer, s: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    const out: []u8 = @constCast(try mapCase(r, s, false));
    if (out.len > 0) out[0] = std.ascii.toUpper(out[0]);
    return out;
}

/// Python `str.title`: a letter after a letter lower case, any other letter upper case.
fn strTitle(r: *Renderer, s: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    try r.charge(s.len);
    const out = try r.arena.alloc(u8, s.len);
    var prev_cased = false;
    for (s, out) |c, *o| {
        const cased = std.ascii.isAlphabetic(c);
        o.* = if (!cased) c else if (prev_cased) std.ascii.toLower(c) else std.ascii.toUpper(c);
        prev_cased = cased;
    }
    return out;
}

/// Jinja2's `title` filter: each piece between runs of `-`, white space, `(`, `{`, `[` or `<` gets its first
/// character upper case and the rest lower.
fn filterTitle(r: *Renderer, s: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    const out: []u8 = @constCast(try mapCase(r, s, false));
    var at_start = true;
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    var i: usize = 0;
    while (it.nextCodepointSlice()) |cs| : (i += cs.len) {
        const cp = decode(cs);
        const sep = value.isSpace(cp) or cp == '-' or cp == '(' or cp == '{' or cp == '[' or cp == '<';
        if (sep) {
            at_start = true;
        } else if (at_start) {
            out[i] = std.ascii.toUpper(out[i]);
            at_start = false;
        }
    }
    return out;
}

/// Python `str.islower` / `str.isupper` over ASCII letters: at least one cased letter, all of the asked case.
fn hasCase(s: []const u8, upper: bool) bool {
    @setRuntimeSafety(true);
    var cased = false;
    for (s) |c| {
        if (!std.ascii.isAlphabetic(c)) continue;
        if (std.ascii.isUpper(c) != upper) return false;
        cased = true;
    }
    return cased;
}

/// The code points `strip(chars)` removes: white space when `chars` is none.
const CharSet = struct {
    ascii: [128]bool = @splat(false),
    wide: std.AutoHashMapUnmanaged(u21, void) = .empty,
    space: bool,

    fn init(r: *Renderer, chars: ?[]const u8) Failure!CharSet {
        @setRuntimeSafety(true);
        var set = CharSet{ .space = chars == null };
        const text = chars orelse return set;
        try r.scan(text.len);
        var it = std.unicode.Utf8View.initUnchecked(text).iterator();
        while (it.nextCodepointSlice()) |cs| {
            const cp = decode(cs);
            if (cp < 128) set.ascii[cp] = true else try set.wide.put(r.arena, cp, {});
        }
        return set;
    }

    fn has(self: *const CharSet, cp: u21) bool {
        @setRuntimeSafety(true);
        if (self.space) return value.isSpace(cp);
        return if (cp < 128) self.ascii[cp] else self.wide.contains(cp);
    }
};

/// `finder`'s index in `hay`, metered by the bytes it reads.
fn find(r: *Renderer, finder: value.Finder, hay: []const u8) Failure!?usize {
    @setRuntimeSafety(true);
    var scanned: usize = 0;
    const at = finder.find(hay, &scanned);
    try r.scan(scanned);
    return at;
}

/// Python `str.strip(chars)`, `lstrip` and `rstrip`, metered.
fn stripChars(r: *Renderer, s: []const u8, chars: ?[]const u8, left: bool, right: bool) Failure![]const u8 {
    @setRuntimeSafety(true);
    const set = try CharSet.init(r, chars);
    try r.scan(s.len);
    return strip(s, &set, left, right);
}

const white_space = CharSet{ .space = true };

fn strip(s: []const u8, set: *const CharSet, left: bool, right: bool) []const u8 {
    @setRuntimeSafety(true);
    var start: usize = 0;
    var end: usize = s.len;
    if (left) {
        var it = std.unicode.Utf8View.initUnchecked(s).iterator();
        while (it.nextCodepointSlice()) |cs| {
            if (!set.has(decode(cs))) break;
            start += cs.len;
        }
    }
    if (right) {
        while (end > start) {
            var b = end - 1;
            while (b > start and s[b] & 0xC0 == 0x80) b -= 1;
            if (!set.has(decode(s[b..end]))) break;
            end = b;
        }
    }
    return s[start..end];
}

fn stripArg(r: *Renderer, v: ?Value) Failure!?[]const u8 {
    @setRuntimeSafety(true);
    const c = v orelse return null;
    if (c == .none) return null;
    return try strArg(r, c, "strip");
}

/// Python `str.split(sep, maxsplit)` and `rsplit`.
fn split(r: *Renderer, src: Str, sep_v: ?Value, max_v: ?Value, from_right: bool) Failure!Value {
    @setRuntimeSafety(true);
    const sep: ?[]const u8 = if (sep_v) |v| (if (v == .none) null else try strArg(r, v, "split")) else null;
    // A negative maxsplit is Python's "no limit".
    var max = if (max_v) |v| try intArg(r, v, "split") else -1;
    if (max < 0) max = std.math.maxInt(i64);
    if (sep != null and sep.?.len == 0) return r.fail("empty separator", .{});
    const s = src.bytes;
    var parts = std.ArrayList(Value).empty;
    if (sep) |p| {
        const finder = try value.Finder.init(r.arena, p, from_right);
        if (!from_right) {
            var rest = s;
            while (max != 0) : (max -= 1) {
                const at = try find(r, finder, rest) orelse break;
                try parts.append(r.arena, try piece(r, src, rest[0..at]));
                rest = rest[at + p.len ..];
            }
            try parts.append(r.arena, try piece(r, src, rest));
        } else {
            var rest = s;
            while (max != 0) : (max -= 1) {
                const at = try find(r, finder, rest) orelse break;
                try parts.append(r.arena, try piece(r, src, rest[at + p.len ..]));
                rest = rest[0..at];
            }
            try parts.append(r.arena, try piece(r, src, rest));
            std.mem.reverse(Value, parts.items);
        }
        return r.newList(parts.items, .list);
    }
    // Runs of white space separate; none are kept, and the rest past `maxsplit` keeps its inner white space.
    var cps = std.ArrayList([]const u8).empty;
    var it = std.unicode.Utf8View.initUnchecked(s).iterator();
    while (it.nextCodepointSlice()) |cs| try cps.append(r.arena, cs);
    const base = @intFromPtr(s.ptr);
    const off = struct {
        fn f(b: usize, cs: []const u8) usize {
            @setRuntimeSafety(true);
            return @intFromPtr(cs.ptr) - b;
        }
    }.f;
    if (!from_right) {
        var i: usize = 0;
        while (true) {
            while (i < cps.items.len and value.isSpace(decode(cps.items[i]))) i += 1;
            if (i == cps.items.len) break;
            if (max == 0) {
                try parts.append(r.arena, try piece(r, src, s[off(base, cps.items[i])..]));
                break;
            }
            const start = i;
            while (i < cps.items.len and !value.isSpace(decode(cps.items[i]))) i += 1;
            const end = if (i < cps.items.len) off(base, cps.items[i]) else s.len;
            try parts.append(r.arena, try piece(r, src, s[off(base, cps.items[start])..end]));
            max -= 1;
        }
    } else {
        var i: usize = cps.items.len;
        while (true) {
            while (i > 0 and value.isSpace(decode(cps.items[i - 1]))) i -= 1;
            if (i == 0) break;
            const end = off(base, cps.items[i - 1]) + cps.items[i - 1].len;
            if (max == 0) {
                try parts.append(r.arena, try piece(r, src, s[0..end]));
                break;
            }
            const stop = i;
            while (i > 0 and !value.isSpace(decode(cps.items[i - 1]))) i -= 1;
            _ = stop;
            try parts.append(r.arena, try piece(r, src, s[off(base, cps.items[i])..end]));
            max -= 1;
        }
        std.mem.reverse(Value, parts.items);
    }
    return r.newList(parts.items, .list);
}

/// Python `str.replace(old, new, count)`: the kept bytes keep their origins, and each `new` brings its own.
fn replace(r: *Renderer, src: Str, old: []const u8, new: Str, count: i64) Failure!Str {
    @setRuntimeSafety(true);
    const s = src.bytes;
    var buf: value.Builder = .{};
    // A negative count is Python's "every occurrence".
    var left = if (count < 0) std.math.maxInt(i64) else count;
    if (old.len == 0) {
        var it = std.unicode.Utf8View.initUnchecked(s).iterator();
        var at: usize = 0;
        while (it.nextCodepointSlice()) |cs| : (at += cs.len) {
            if (left != 0) {
                try r.charge(buf.bytes.items.len + new.bytes.len);
                try buf.append(r.arena, new);
                left -= 1;
            }
            try buf.appendPart(r.arena, src, at, at + cs.len);
        }
        if (left != 0) try buf.append(r.arena, new);
        try r.charge(buf.bytes.items.len);
        return buf.str();
    }
    const finder = try value.Finder.init(r.arena, old, false);
    var from: usize = 0;
    while (left != 0) : (left -= 1) {
        const at = try find(r, finder, s[from..]) orelse break;
        try r.charge(buf.bytes.items.len + at + new.bytes.len);
        try buf.appendPart(r.arena, src, from, from + at);
        try buf.append(r.arena, new);
        from += at + old.len;
    }
    try buf.appendPart(r.arena, src, from, s.len);
    try r.charge(buf.bytes.items.len);
    return buf.str();
}

/// Python `str.startswith` / `endswith` with an optional `start` and `end` in code points.
fn affix(r: *Renderer, s: []const u8, args: []const Value, ends: bool) Failure!bool {
    @setRuntimeSafety(true);
    const what = if (ends) "endswith" else "startswith";
    if (args.len < 1 or args.len > 3) return r.fail("{s}() takes from 1 to 3 arguments ({d} given)", .{ what, args.len });
    const n: i64 = @intCast(render.cpCount(s));
    var lo: i64 = 0;
    var hi: i64 = n;
    if (args.len > 1 and args[1] != .none) lo = try intArg(r, args[1], what);
    if (args.len > 2 and args[2] != .none) hi = try intArg(r, args[2], what);
    if (lo < 0) lo = @max(0, lo + n);
    if (hi < 0) hi = @max(0, hi + n);
    hi = @min(hi, n);
    // A start past the end matches nothing, not even '', as in CPython's tailmatch.
    if (lo > n or lo > hi) return false;
    const sub = render.cpSlice(s, @intCast(lo), @intCast(hi));
    const probes: []const Value = switch (args[0]) {
        .list => |l| if (l.kind == .tuple) l.items.items else return r.fail("{s} first arg must be str or a tuple of str", .{what}),
        else => args[0..1],
    };
    for (probes) |p| {
        const t = p.str() orelse return r.fail("{s} first arg must be str or a tuple of str", .{what});
        if (if (ends) std.mem.endsWith(u8, sub, t) else std.mem.startsWith(u8, sub, t)) return true;
    }
    return false;
}

fn escapeHtml(r: *Renderer, s: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    var buf = std.ArrayList(u8).empty;
    for (s) |c| switch (c) {
        '&' => try buf.appendSlice(r.arena, "&amp;"),
        '<' => try buf.appendSlice(r.arena, "&lt;"),
        '>' => try buf.appendSlice(r.arena, "&gt;"),
        '\'' => try buf.appendSlice(r.arena, "&#39;"),
        '"' => try buf.appendSlice(r.arena, "&#34;"),
        else => try buf.append(r.arena, c),
    };
    try r.charge(buf.items.len);
    return buf.items;
}

/// The text of `v` as Markup adds it: Markup as it is, anything else escaped.
fn markupOperand(r: *Renderer, v: Value) Failure!Str {
    @setRuntimeSafety(true);
    return switch (v) {
        .markup => |m| m,
        .string => |s| s.derive(try escapeHtml(r, s.bytes)),
        else => r.fail("unsupported operand type for Markup: '{s}'", .{value.typeName(v)}),
    };
}

/// Python's `str % args`: `%s`, `%r`, `%d`, `%i`, `%f` with an optional precision, `%%`, and `%(name)s` keys.
/// Python `fmt % args`; `escape` is Markup's `%`, which escapes each argument that is not Markup itself.
/// Each piece keeps its origin: the format's own text, and each argument's.
fn percentFormat(r: *Renderer, fmt_s: Str, args: Value, escape: bool) Failure!Str {
    @setRuntimeSafety(true);
    const fmt = fmt_s.bytes;
    const positional: []const Value = switch (args) {
        .list => |l| if (l.kind == .tuple) l.items.items else &.{args},
        else => &.{args},
    };
    const mapping: ?*const value.Dict = if (args == .dict) args.dict else null;
    var out: value.Builder = .{};
    var buf = std.ArrayList(u8).empty;
    var next: usize = 0;
    var i: usize = 0;
    while (i < fmt.len) {
        const c = fmt[i];
        i += 1;
        if (c != '%') {
            try out.appendPart(r.arena, fmt_s, i - 1, i);
            continue;
        }
        if (i >= fmt.len) return r.fail("incomplete format", .{});
        var arg: ?Value = null;
        if (fmt[i] == '(') {
            const close = std.mem.indexOfScalarPos(u8, fmt, i, ')') orelse return r.fail("incomplete format key", .{});
            const m = mapping orelse return r.fail("format requires a mapping", .{});
            arg = try m.getStr(&r.meter, fmt[i + 1 .. close]) orelse return r.fail("format key '{s}' is missing", .{fmt[i + 1 .. close]});
            i = close + 1;
        }
        var precision: ?usize = null;
        if (i < fmt.len and fmt[i] == '.') {
            i += 1;
            const start = i;
            while (i < fmt.len and std.ascii.isDigit(fmt[i])) i += 1;
            precision = std.fmt.parseInt(usize, fmt[start..i], 10) catch 0;
        }
        if (i >= fmt.len) return r.fail("incomplete format", .{});
        const conv = fmt[i];
        i += 1;
        if (conv == '%' and arg == null and precision == null) {
            try out.appendPart(r.arena, fmt_s, i - 1, i);
            continue;
        }
        const v = arg orelse blk: {
            if (mapping != null) break :blk args;
            if (next >= positional.len) return r.fail("not enough arguments for format string", .{});
            next += 1;
            break :blk positional[next - 1];
        };
        buf.clearRetainingCapacity();
        switch (conv) {
            's' => {
                const text = try r.toText(v);
                try out.append(r.arena, if (escape and v != .markup) text.derive(try escapeHtml(r, text.bytes)) else text);
            },
            'r' => {
                var repr = std.ArrayList(u8).empty;
                value.writeRepr(&r.meter, &repr, r.arena, v, 0) catch |e| return r.wrap(e, "the value cannot be printed");
                try out.appendBytes(r.arena, if (escape) try escapeHtml(r, repr.items) else repr.items, value.originIn(v, 0));
            },
            'd', 'i' => switch (v) {
                .int, .boolean => try buf.print(r.arena, "{d}", .{try intArg(r, v, "format")}),
                .bigint => |t| try buf.appendSlice(r.arena, t),
                .float => |f| {
                    if (!std.math.isFinite(f) or @abs(f) >= 9.2e18) return r.fail("cannot convert float to integer", .{});
                    try buf.print(r.arena, "{d}", .{@as(i64, @intFromFloat(f))});
                },
                else => return r.fail("%d format: a real number is required, not {s}", .{value.typeName(v)}),
            },
            'f' => {
                const f: f64 = switch (v) {
                    .int => |n| @floatFromInt(n),
                    .boolean => |b| @floatFromInt(@intFromBool(b)),
                    .float => |f| f,
                    else => return r.fail("must be real number, not {s}", .{value.typeName(v)}),
                };
                const prec = precision orelse 6;
                try r.charge(out.bytes.items.len +| prec +| 400);
                try tojson.writeFixed(&buf, r.arena, f, prec);
            },
            else => return r.fail("unsupported format character '{c}'", .{conv}),
        }
        try out.appendBytes(r.arena, buf.items, .template);
        try r.charge(out.bytes.items.len);
    }
    if (mapping == null and next < positional.len) return r.fail("not all arguments converted during string formatting", .{});
    try r.charge(out.bytes.items.len);
    return out.str();
}

// ── Numbers ──

/// Python's `x / y` on ints (y not 0): the quotient correctly rounded, as CPython's long true division gives
/// it, never the quotient of two rounded floats.
fn trueDivide(x: i64, y: i64) f64 {
    @setRuntimeSafety(true);
    const a = @abs(x);
    const b = @abs(y);
    const exact = 1 << 53;
    var q: f64 = undefined;
    // Exact operands divide exactly in f64; a zero numerator is a signed zero, which the wide path cannot write.
    if (a == 0 or (a <= exact and b <= exact)) {
        q = @as(f64, @floatFromInt(a)) / @as(f64, @floatFromInt(b));
    } else {
        // a >= 1, so 2^128 * a / b has at least 65 bits; past 64 the dropped bits and the remainder fold into a sticky
        // low bit, so the one rounding of the u64 to f64 is the correct one.
        const n: u256 = @as(u256, a) << 128;
        const quot = n / b;
        const sticky = n % b != 0;
        const bits: u16 = 256 - @clz(quot);
        const drop: u8 = @intCast(bits - 64);
        var m: u64 = @truncate(quot >> drop);
        if (sticky or quot & ((@as(u256, 1) << drop) - 1) != 0) m |= 1;
        q = std.math.ldexp(@as(f64, @floatFromInt(m)), @as(i32, drop) - 128);
    }
    return if ((x < 0) != (y < 0)) -q else q;
}

fn pyMod(r: *Renderer, a: Value, b: Value) Failure!Value {
    @setRuntimeSafety(true);
    if (asInt(a)) |x| if (asInt(b)) |y| {
        if (y == 0) return r.fail("integer modulo by zero", .{});
        if (y == -1) return .{ .int = 0 };
        const m = @rem(x, y);
        return .{ .int = if (m != 0 and (m < 0) != (y < 0)) m + y else m };
    };
    const x = asFloat(a) orelse return r.fail("unsupported operand type(s) for %: '{s}' and '{s}'", .{ value.typeName(a), value.typeName(b) });
    const y = asFloat(b) orelse return r.fail("unsupported operand type(s) for %: '{s}' and '{s}'", .{ value.typeName(a), value.typeName(b) });
    if (y == 0) return r.fail("float modulo", .{});
    var m = @rem(x, y);
    if (m != 0 and (m < 0) != (y < 0)) m += y;
    if (m == 0) m = std.math.copysign(@as(f64, 0), y);
    return .{ .float = m };
}

fn asInt(v: Value) ?i64 {
    @setRuntimeSafety(true);
    return switch (v) {
        .int => |i| i,
        .boolean => |b| @intFromBool(b),
        else => null,
    };
}

fn asFloat(v: Value) ?f64 {
    @setRuntimeSafety(true);
    return switch (v) {
        .int => |i| @floatFromInt(i),
        .boolean => |b| @floatFromInt(@intFromBool(b)),
        .float => |f| f,
        else => null,
    };
}

/// Python's `+ - * / %` on template values.
pub fn binary(r: *Renderer, op: ast.BinOp, a: Value, b: Value) Failure!Value {
    @setRuntimeSafety(true);
    try undefinedFail(r, a);
    try undefinedFail(r, b);
    if (a == .bigint or b == .bigint) {
        if (a.isNumber() and b.isNumber()) return r.fail("arithmetic on an integer past 64 bits is not supported", .{});
    }
    if (op == .mod and a.str() != null) {
        const text = try percentFormat(r, a.strOf().?, b, a == .markup);
        return if (a == .markup) .{ .markup = text } else .{ .string = text };
    }
    if (asInt(a)) |x| if (asInt(b)) |y| {
        const res = switch (op) {
            .add => @addWithOverflow(x, y),
            .sub => @subWithOverflow(x, y),
            .mul => @mulWithOverflow(x, y),
            .div => {
                if (y == 0) return r.fail("division by zero", .{});
                return .{ .float = trueDivide(x, y) };
            },
            .mod => return pyMod(r, a, b),
        };
        if (res[1] != 0) return r.fail("integer overflow past 64 bits is not supported", .{});
        return .{ .int = res[0] };
    };
    if (asFloat(a)) |x| if (asFloat(b)) |y| return switch (op) {
        .add => .{ .float = x + y },
        .sub => .{ .float = x - y },
        .mul => .{ .float = x * y },
        .div => if (y == 0) r.fail("float division by zero", .{}) else .{ .float = x / y },
        .mod => pyMod(r, a, b),
    };
    switch (op) {
        .add => {
            if (a.str() != null and b.str() != null) {
                if (a == .markup or b == .markup) {
                    const l = try markupOperand(r, a);
                    const rt = try markupOperand(r, b);
                    return .{ .markup = try joined(r, &.{ l, rt }) };
                }
                return .{ .string = try joined(r, &.{ a.string, b.string }) };
            }
            if (a == .list and b == .list and a.list.kind == b.list.kind and (a.list.kind == .list or a.list.kind == .tuple)) {
                const items = try std.mem.concat(r.arena, Value, &.{ a.list.items.items, b.list.items.items });
                return r.newList(items, a.list.kind);
            }
        },
        .mul => {
            const seq, const n_v = if (asInt(b) != null) .{ a, b } else .{ b, a };
            if (asInt(n_v)) |n_raw| {
                const n: usize = if (n_raw < 0) 0 else @intCast(n_raw);
                if (seq.strOf()) |s| return sameKind(seq, s.derive(try repeat(r, u8, s.bytes, n)));
                if (seq == .list and (seq.list.kind == .list or seq.list.kind == .tuple))
                    return r.newList(try repeat(r, Value, seq.list.items.items, n), seq.list.kind);
            }
        },
        else => {},
    }
    const sym = switch (op) {
        .add => "+",
        .sub => "-",
        .mul => "*",
        .div => "/",
        .mod => "%",
    };
    return r.fail("unsupported operand type(s) for {s}: '{s}' and '{s}'", .{ sym, value.typeName(a), value.typeName(b) });
}

/// `parts` end to end, each keeping its origins.
fn joined(r: *Renderer, parts: []const Str) Failure!Str {
    @setRuntimeSafety(true);
    var len: usize = 0;
    for (parts) |p| len += p.bytes.len;
    try r.charge(len);
    var out: value.Builder = .{};
    for (parts) |p| try out.append(r.arena, p);
    return out.str();
}

/// Python's `item in container`.
pub fn contains(r: *Renderer, container: Value, item: Value) Failure!bool {
    @setRuntimeSafety(true);
    switch (container) {
        .undefined => return false,
        .string, .markup => |s| {
            const t = item.str() orelse return r.fail("'in <string>' requires string as left operand, not {s}", .{value.typeName(item)});
            return try find(r, try value.Finder.init(r.arena, t, false), s.bytes) != null;
        },
        .list => |l| {
            const items = l.items.items;
            for (items, 0..) |x, i| if (value.eql(&r.meter, x, item, 0) catch |e| return r.wrap(e, "the values nest too deep to compare")) {
                if (l.isIterator()) l.items = .fromOwnedSlice(@constCast(items[i + 1 ..]));
                return true;
            };
            if (l.isIterator()) l.items = .empty;
            return false;
        },
        .dict => |d| {
            try r.hashable(item);
            return (try d.indexOf(&r.meter, item)) != null;
        },
        else => return r.fail("argument of type '{s}' is not iterable", .{value.typeName(container)}),
    }
}

// ── Attribute lookup ──

const str_methods_unsupported = [_][]const u8{
    "casefold",   "center",    "count",       "encode",     "expandtabs", "find",     "format",       "format_map",
    "index",      "isalnum",   "isalpha",     "isascii",    "isdecimal",  "isdigit",  "isidentifier", "islower",
    "isnumeric",  "isprintable", "isspace",   "istitle",    "isupper",    "join",     "ljust",        "maketrans",
    "partition",  "removeprefix", "removesuffix", "rfind",  "rindex",     "rjust",    "rpartition",   "splitlines",
    "swapcase",   "translate", "zfill",
};
const dict_mutators = [_][]const u8{ "clear", "pop", "popitem", "setdefault", "update" };
const list_mutators = [_][]const u8{ "append", "clear", "extend", "insert", "pop", "remove", "reverse", "sort" };

fn among(name: []const u8, set: []const []const u8) bool {
    @setRuntimeSafety(true);
    for (set) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

/// A Python attribute of `obj` (a method), as the sandbox gives it: null when Python has none, so the lookup
/// goes on to the item; undefined for a method the immutable sandbox refuses (it fails when called).
pub fn attribute(r: *Renderer, obj: Value, name: []const u8) Failure!?Value {
    @setRuntimeSafety(true);
    if (std.mem.startsWith(u8, name, "__") and std.mem.endsWith(u8, name, "__")) return .{ .undefined = name };
    const Kind = enum { str, dict, list };
    const kind: Kind = switch (obj) {
        .string, .markup => .str,
        .dict => .dict,
        .list => |l| if (l.kind == .list or l.kind == .tuple) .list else return null,
        else => return null,
    };
    if (std.meta.stringToEnum(value.Method, name)) |m| {
        const fits = switch (m) {
            .upper, .lower, .title, .capitalize, .strip, .lstrip, .rstrip, .startswith, .endswith, .split, .rsplit, .replace, .indent => kind == .str,
            .get, .keys, .values, .items => kind == .dict,
            .append => kind == .list and obj.list.kind == .list,
        };
        if (fits) return try r.function(.{ .method = .{ .receiver = obj, .method = m } });
    }
    switch (kind) {
        .str => if (among(name, &str_methods_unsupported)) return r.fail("unsupported method 'str.{s}'", .{name}),
        .dict => {
            if (among(name, &dict_mutators)) return .{ .undefined = name };
            if (among(name, &.{ "copy", "fromkeys" })) return r.fail("unsupported method 'dict.{s}'", .{name});
        },
        .list => {
            if (obj.list.kind == .list and among(name, &list_mutators)) return .{ .undefined = name };
            if (among(name, &.{ "copy", "count", "index" })) return r.fail("unsupported method '{s}.{s}'", .{ @tagName(obj.list.kind), name });
        },
    }
    return null;
}

/// `item.a.b` or `item.0` as Jinja2's attribute getters walk a dotted name, each part through the sandbox's
/// getitem.
fn attrPath(r: *Renderer, item: Value, path: Value) Failure!Value {
    @setRuntimeSafety(true);
    if (asInt(path)) |i| return r.getItem(item, .{ .int = i });
    const text = path.str() orelse return r.getItem(item, path);
    var v = item;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |part| {
        const all_digits = part.len > 0 and for (part) |c| {
            if (!std.ascii.isDigit(c)) break false;
        } else true;
        const key: Value = if (all_digits) .{ .int = std.fmt.parseInt(i64, part, 10) catch return r.fail("index '{s}' is too large", .{part}) } else .plain(part);
        v = try r.getItem(v, key);
    }
    return v;
}

/// Jinja2's `ignore_case` key: a string lower-cased, anything else as it is.
fn caseKey(r: *Renderer, v: Value, case_sensitive: bool) Failure!Value {
    @setRuntimeSafety(true);
    if (case_sensitive) return v;
    if (v.strOf()) |s| return .{ .string = s.remap(try mapCase(r, s.bytes, false)) };
    return v;
}

fn keyOf(r: *Renderer, item: Value, attr: ?Value, case_sensitive: bool) Failure!Value {
    @setRuntimeSafety(true);
    const v = if (attr) |a| (if (a == .none) item else try attrPath(r, item, a)) else item;
    return caseKey(r, v, case_sensitive);
}

/// `unit` repeated `n` times, its work metered by the bytes it writes.
fn repeat(r: *Renderer, comptime T: type, unit: []const T, n: usize) Failure![]T {
    @setRuntimeSafety(true);
    if (unit.len == 0 or n == 0) return &.{};
    const len = std.math.mul(usize, unit.len, n) catch std.math.maxInt(usize);
    try r.charge(len);
    const out = try r.arena.alloc(T, len);
    try r.scan(len * @sizeOf(T));
    @memcpy(out[0..unit.len], unit);
    var done = unit.len;
    while (done < len) {
        const k = @min(done, len - done);
        @memcpy(out[done..][0..k], out[0..k]);
        done += k;
    }
    return out;
}

/// Python's ordering of `a` and `b`, null when unordered (a NaN): then `<`, `>` and `>=` all answer false.
fn compare(r: *Renderer, a: Value, b: Value) Failure!?std.math.Order {
    @setRuntimeSafety(true);
    return value.order(&r.meter, a, b, 0) catch |e| return switch (e) {
        error.TemplateRuntime => if (r.meter.spent()) r.wrap(e, "") else r.fail("'<' not supported between instances of '{s}' and '{s}'", .{ value.typeName(a), value.typeName(b) }),
        else => e,
    };
}

fn lessThan(r: *Renderer, a: Value, b: Value) Failure!bool {
    @setRuntimeSafety(true);
    return try compare(r, a, b) == .lt;
}

// ── Filters ──

pub fn applyFilter(r: *Renderer, f: Filter, input: Value, args: []const Value, kws: []const Kw) Failure!Value {
    @setRuntimeSafety(true);
    // A string filter passes over its text once or more.
    if (input.str()) |text| try r.scan(text.len);
    switch (f) {
        .abs => {
            _ = try bind(r, "abs", &.{}, args, kws);
            try undefinedFail(r, input);
            return switch (input) {
                .int => |i| if (i == std.math.minInt(i64)) value.integer("9223372036854775808") else .{ .int = if (i < 0) -i else i },
                .boolean => |b| .{ .int = @intFromBool(b) },
                .float => |x| .{ .float = @abs(x) },
                .bigint => |t| value.integer(if (t[0] == '-') t[1..] else t),
                else => r.fail("bad operand type for abs(): '{s}'", .{value.typeName(input)}),
            };
        },
        .capitalize, .lower, .upper, .title => {
            _ = try bind(r, @tagName(f), &.{}, args, kws);
            const s = try r.toText(input);
            return sameKind(input, s.remap(switch (f) {
                .capitalize => try capitalize(r, s.bytes),
                .lower => try mapCase(r, s.bytes, false),
                .upper => try mapCase(r, s.bytes, true),
                else => try filterTitle(r, s.bytes),
            }));
        },
        .default => {
            const p = try bind(r, "default", &.{ "default_value", "boolean" }, args, kws);
            const boolean = if (p[1]) |b| value.truthy(b) else false;
            if (input == .undefined or (boolean and !value.truthy(input))) return p[0] orelse .plain("");
            return input;
        },
        .first => {
            _ = try bind(r, "first", &.{}, args, kws);
            if (input == .list and input.list.isIterator()) {
                const l = input.list;
                if (l.items.items.len == 0) return .{ .undefined = "No first item, sequence was empty." };
                const v = l.items.items[0];
                l.items = .fromOwnedSlice(l.items.items[1..]);
                return v;
            }
            const items = try r.iterate(input);
            return if (items.len > 0) items[0] else .{ .undefined = "No first item, sequence was empty." };
        },
        .last => {
            _ = try bind(r, "last", &.{}, args, kws);
            switch (input) {
                .undefined => return .{ .undefined = "No last item, sequence was empty." },
                .list => |l| if (l.isIterator()) return r.fail("'{s}' object is not reversible", .{@tagName(l.kind)}),
                .string, .markup, .dict => {},
                else => return r.fail("'{s}' object is not reversible", .{value.typeName(input)}),
            }
            const items = try r.iterate(input);
            return if (items.len > 0) items[items.len - 1] else .{ .undefined = "No last item, sequence was empty." };
        },
        .float => {
            const p = try bind(r, "float", &.{"default"}, args, kws);
            const default = p[0] orelse Value{ .float = 0.0 };
            try undefinedFail(r, input);
            return switch (input) {
                .float => input,
                .int, .boolean => .{ .float = asFloat(input).? },
                .bigint => |t| .{ .float = value.bigToFloat(t) },
                .string, .markup => |s| if (parsePyFloat(s.bytes)) |x| .{ .float = x } else default,
                else => default,
            };
        },
        .int => {
            const p = try bind(r, "int", &.{ "default", "base" }, args, kws);
            const default = p[0] orelse Value{ .int = 0 };
            const base = if (p[1]) |b| try intArg(r, b, "int") else 10;
            try undefinedFail(r, input);
            switch (input) {
                .int, .bigint => return input,
                .boolean => |b| return .{ .int = @intFromBool(b) },
                // As Jinja2's do_int: a NaN or a text whose float is not finite falls back to `default`; an
                // infinite float raises Python's OverflowError.
                .float => |x| return if (std.math.isNan(x)) default else floatToInt(r, x),
                .string, .markup => |s| {
                    if (parsePyInt(s.bytes, base)) |i| return .{ .int = i } else |e| switch (e) {
                        error.Overflow => return r.fail("an integer past 64 bits is not supported", .{}),
                        error.Invalid => {},
                    }
                    const x = parsePyFloat(s.bytes) orelse return default;
                    return if (std.math.isFinite(x)) floatToInt(r, x) else default;
                },
                else => return default,
            }
        },
        .format => {
            const text = try r.toText(input);
            if (args.len > 0 and kws.len > 0) return r.fail("can't handle positional and keyword arguments at the same time", .{});
            const fmt_args: Value = if (kws.len > 0) blk: {
                const d = try r.arena.create(value.Dict);
                d.* = .{};
                for (kws) |kw| try d.put(&r.meter, r.arena, .plain(kw.name), kw.value);
                break :blk .{ .dict = d };
            } else try r.newList(args, .tuple);
            return sameKind(input, try percentFormat(r, text, fmt_args, input == .markup));
        },
        .indent => {
            const p = try bind(r, "indent", &.{ "width", "first", "blank" }, args, kws);
            const src = input.strOf() orelse {
                try undefinedFail(r, input);
                return r.fail("indent: expected a string, got '{s}'", .{value.typeName(input)});
            };
            const s = src.bytes;
            const pad_origin = if (p[0]) |w| value.originIn(w, 0) else .template;
            const indention: []const u8 = if (p[0]) |w| (w.str() orelse blk: {
                const n = try intArg(r, w, "indent");
                const len: usize = if (n < 0) 0 else @intCast(n);
                try r.charge(len);
                const pad = try r.arena.alloc(u8, len);
                @memset(pad, ' ');
                break :blk pad;
            }) else "    ";
            const first = if (p[1]) |b| value.truthy(b) else false;
            const blank = if (p[2]) |b| value.truthy(b) else false;
            const lines = try splitLines(r, try std.mem.concat(r.arena, u8, &.{ s, "\n" }));
            var buf = std.ArrayList(u8).empty;
            if (first) try buf.appendSlice(r.arena, indention);
            for (lines, 0..) |line, i| {
                if (i > 0) {
                    try buf.append(r.arena, '\n');
                    if (blank or line.len > 0) try buf.appendSlice(r.arena, indention);
                }
                try buf.appendSlice(r.arena, line);
                try r.charge(buf.items.len);
            }
            return sameKind(input, .{ .bytes = buf.items, .origin = src.origin.strictest(pad_origin) });
        },
        .items => {
            _ = try bind(r, "items", &.{}, args, kws);
            if (input == .undefined) return r.newList(&.{}, .generator);
            if (input != .dict) return r.fail("Can only get item pairs from a mapping.", .{});
            return dictView(r, input.dict, .generator);
        },
        .join => {
            const p = try bind(r, "join", &.{ "d", "attribute" }, args, kws);
            const sep: Str = if (p[0]) |d| try r.toText(d) else .{ .bytes = "" };
            const items = try r.iterate(input);
            var buf: value.Builder = .{};
            for (items, 0..) |item, i| {
                if (i > 0) try buf.append(r.arena, sep);
                const v = if (p[1]) |a| (if (a == .none) item else try attrPath(r, item, a)) else item;
                try buf.append(r.arena, try r.toText(v));
                try r.charge(buf.bytes.items.len);
            }
            return .{ .string = buf.str() };
        },
        .length => {
            _ = try bind(r, "length", &.{}, args, kws);
            return .{ .int = @intCast(try length(r, input)) };
        },
        .list => {
            _ = try bind(r, "list", &.{}, args, kws);
            return r.newList(try r.iterate(input), .list);
        },
        .map => {
            var out = std.ArrayList(Value).empty;
            const is_attr = args.len == 0 and for (kws) |kw| {
                if (std.mem.eql(u8, kw.name, "attribute")) break true;
            } else false;
            if (is_attr) {
                const p = try bind(r, "map", &.{ "attribute", "default" }, &.{}, kws);
                for (try r.iterate(input)) |item| {
                    var v = try attrPath(r, item, p[0].?);
                    if (v == .undefined) if (p[1]) |d| if (d != .none) {
                        v = d;
                    };
                    try out.append(r.arena, v);
                }
            } else {
                if (args.len == 0) return r.fail("map requires a filter argument", .{});
                const name = try strArg(r, args[0], "map");
                const inner = std.meta.stringToEnum(Filter, name) orelse return r.fail("unsupported filter '{s}'", .{name});
                try r.enterCall();
                defer r.call_depth -= 1;
                for (try r.iterate(input)) |item| try out.append(r.arena, try applyFilter(r, inner, item, args[1..], kws));
            }
            return r.newList(out.items, .generator);
        },
        .max, .min => {
            const p = try bind(r, @tagName(f), &.{ "case_sensitive", "attribute" }, args, kws);
            const cs = if (p[0]) |c| value.truthy(c) else false;
            const items = try r.iterate(input);
            if (items.len == 0) return .{ .undefined = "No aggregated item, sequence was empty." };
            var best = items[0];
            var best_key = try keyOf(r, best, p[1], cs);
            for (items[1..]) |item| {
                const k = try keyOf(r, item, p[1], cs);
                const better = if (f == .max) try lessThan(r, best_key, k) else try lessThan(r, k, best_key);
                if (better) {
                    best = item;
                    best_key = k;
                }
            }
            return best;
        },
        .select, .reject, .selectattr, .rejectattr => {
            const by_attr = f == .selectattr or f == .rejectattr;
            const keep_when: bool = f == .select or f == .selectattr;
            if (by_attr and args.len == 0) return r.fail("Missing parameter for attribute name", .{});
            const off: usize = @intFromBool(by_attr);
            const t: ?Test = if (args.len > off) blk: {
                const name = try strArg(r, args[off], @tagName(f));
                break :blk testByName(name) orelse return r.fail("unsupported test '{s}'", .{name});
            } else null;
            const targs = if (args.len > off + 1) args[off + 1 ..] else &[_]Value{};
            var out = std.ArrayList(Value).empty;
            if (value.truthy(input)) for (try r.iterate(input)) |item| {
                const v = if (by_attr) try attrPath(r, item, args[0]) else item;
                const ok = if (t) |tt| try applyTest(r, tt, v, targs, kws) else value.truthy(v);
                if (ok == keep_when) try out.append(r.arena, item);
            };
            return r.newList(out.items, .generator);
        },
        .replace => {
            const p = try bind(r, "replace", &.{ "old", "new", "count" }, args, kws);
            const s = try r.toText(input);
            const old = try r.toStr(try required(r, "replace", p[0]));
            const new = try r.toText(try required(r, "replace", p[1]));
            const count = if (p[2]) |c| (if (c == .none) -1 else try intArg(r, c, "replace")) else -1;
            return .{ .string = try replace(r, s, old, new, count) };
        },
        .reverse => {
            _ = try bind(r, "reverse", &.{}, args, kws);
            switch (input) {
                .string, .markup => |src| {
                    const s = src.bytes;
                    var buf = std.ArrayList(u8).empty;
                    var end = s.len;
                    while (end > 0) {
                        var b = end - 1;
                        while (b > 0 and s[b] & 0xC0 == 0x80) b -= 1;
                        try buf.appendSlice(r.arena, s[b..end]);
                        end = b;
                    }
                    return sameKind(input, src.derive(buf.items));
                },
                .undefined, .dict => {},
                .list => {},
                else => return r.fail("argument must be iterable", .{}),
            }
            const was_iterator = input == .list and input.list.isIterator();
            const items = try r.arena.dupe(Value, try r.iterate(input));
            std.mem.reverse(Value, items);
            return r.newList(items, if (was_iterator) .list else .list_reverseiterator);
        },
        .safe => {
            _ = try bind(r, "safe", &.{}, args, kws);
            return .{ .markup = try r.toText(input) };
        },
        .sort => {
            const p = try bind(r, "sort", &.{ "reverse", "case_sensitive", "attribute" }, args, kws);
            const reverse = if (p[0]) |b| value.truthy(b) else false;
            const cs = if (p[1]) |c| value.truthy(c) else false;
            const source = try r.iterate(input);
            const Keyed = struct { item: Value, key: Value };
            const keyed = try r.arena.alloc(Keyed, source.len);
            for (source, keyed) |item, *k| k.* = .{ .item = item, .key = try sortKey(r, item, p[2], cs) };
            // Stable either way, as Python's sort: equal keys keep their order under `reverse` too.
            const Ctx = struct { sort: value.SortContext, reverse: bool };
            var ctx = Ctx{ .sort = .{ .meter = &r.meter }, .reverse = reverse };
            value.stableSort(Keyed, keyed, try r.arena.alloc(Keyed, keyed.len), &ctx, struct {
                fn lt(c: *Ctx, a: Keyed, b: Keyed) Failure!bool {
                    @setRuntimeSafety(true);
                    return if (c.reverse) c.sort.less(b.key, a.key) else c.sort.less(a.key, b.key);
                }
            }.lt) catch |e| return switch (e) {
                error.TemplateRuntime => if (r.meter.spent())
                    r.wrap(e, "")
                else if (ctx.sort.unordered)
                    r.fail("sort: a NaN cannot be ordered", .{})
                else
                    r.fail("'<' not supported between the sorted values", .{}),
                else => e,
            };
            const items = try r.arena.alloc(Value, keyed.len);
            for (keyed, items) |k, *v| v.* = k.item;
            return r.newList(items, .list);
        },
        .string => {
            _ = try bind(r, "string", &.{}, args, kws);
            if (input == .markup) return input;
            return .{ .string = try r.toText(input) };
        },
        .strip, .trim => {
            const p = try bind(r, @tagName(f), &.{"chars"}, args, kws);
            const s = try r.toText(input);
            return sameKind(input, try within(r, s, try stripChars(r, s.bytes, try stripArg(r, p[0]), true, true)));
        },
        .sum => {
            const p = try bind(r, "sum", &.{ "attribute", "start" }, args, kws);
            var total = p[1] orelse Value{ .int = 0 };
            for (try r.iterate(input)) |item| {
                const v = if (p[0]) |a| (if (a == .none) item else try attrPath(r, item, a)) else item;
                if (total.str() != null) return r.fail("sum() can't sum strings", .{});
                total = try binary(r, .add, total, v);
            }
            return total;
        },
        .tojson => {
            const p = try bind(r, "tojson", &.{ "ensure_ascii", "indent", "separators", "sort_keys" }, args, kws);
            var opts = tojson.Options{ .max_len = r.limits.max_string };
            if (p[0]) |b| opts.ensure_ascii = value.truthy(b);
            if (p[3]) |b| opts.sort_keys = value.truthy(b);
            if (p[1]) |ind| switch (ind) {
                .none => {},
                .string, .markup => |s| opts.indent = s.bytes,
                else => {
                    const n = try intArg(r, ind, "tojson");
                    const len: usize = if (n < 0) 0 else @intCast(@min(n, 1 << 20));
                    const pad = try r.arena.alloc(u8, len);
                    @memset(pad, ' ');
                    opts.indent = pad;
                },
            };
            if (p[2]) |sep| if (sep != .none) {
                const pair = if (sep == .list and sep.list.isSequence() and sep.list.items.items.len == 2) sep.list.items.items else return r.fail("tojson: separators must be a pair of strings", .{});
                opts.item_separator = try strArg(r, pair[0], "tojson");
                opts.key_separator = try strArg(r, pair[1], "tojson");
            };
            var buf = std.ArrayList(u8).empty;
            tojson.dumps(&r.meter, &buf, r.arena, input, opts, 0) catch |e| switch (e) {
                error.TemplateRuntime => {
                    if (r.meter.spent()) return r.wrap(e, "");
                    if (buf.items.len > opts.max_len) return r.fail("the render builds more text than its limit", .{});
                    return r.fail("tojson: a value of type '{s}' is not JSON serializable (or nests deeper than {d})", .{ value.typeName(input), value.max_depth });
                },
                else => return e,
            };
            try r.charge(buf.items.len);
            return .{ .string = .{ .bytes = buf.items, .origin = value.originIn(input, 0) } };
        },
        .truncate => {
            const p = try bind(r, "truncate", &.{ "length", "killwords", "end", "leeway" }, args, kws);
            if (input == .undefined) return input;
            const src = input.strOf() orelse return r.fail("truncate: expected a string, got '{s}'", .{value.typeName(input)});
            const s = src.bytes;
            const len = if (p[0]) |v| try intArg(r, v, "truncate") else 255;
            const killwords = if (p[1]) |v| value.truthy(v) else false;
            const end_s: Str = if (p[2]) |v| (v.strOf() orelse return r.fail("truncate: expected a string, got '{s}'", .{value.typeName(v)})) else .{ .bytes = "..." };
            const end = end_s.bytes;
            const leeway = if (p[3]) |v| (if (v == .none) 5 else try intArg(r, v, "truncate")) else 5;
            const end_len: i64 = @intCast(render.cpCount(end));
            if (len < end_len) return r.fail("expected length >= {d}, got {d}", .{ end_len, len });
            if (leeway < 0) return r.fail("expected leeway >= 0, got {d}", .{leeway});
            const n: i64 = @intCast(render.cpCount(s));
            if (n <= @as(i128, len) + leeway) return input;
            const head = render.cpSlice(s, 0, @intCast(len - end_len));
            const kept = if (killwords) head else if (std.mem.lastIndexOfScalar(u8, head, ' ')) |at| head[0..at] else head;
            return .{ .string = try joined(r, &.{ try within(r, src, kept), end_s }) };
        },
        .unique => {
            const p = try bind(r, "unique", &.{ "case_sensitive", "attribute" }, args, kws);
            const cs = if (p[0]) |c| value.truthy(c) else false;
            var out = std.ArrayList(Value).empty;
            var seen = std.ArrayList(Value).empty;
            for (try r.iterate(input)) |item| {
                const k = try keyOf(r, item, p[1], cs);
                try r.hashable(k);
                const dup = for (seen.items) |s| {
                    if (value.eql(&r.meter, s, k, 0) catch |e| return r.wrap(e, "the values nest too deep to compare")) break true;
                } else false;
                if (dup) continue;
                try seen.append(r.arena, k);
                try out.append(r.arena, item);
            }
            return r.newList(out.items, .generator);
        },
        .wordcount => {
            _ = try bind(r, "wordcount", &.{}, args, kws);
            const s = try r.toStr(input);
            var n: i64 = 0;
            var in_word = false;
            var it = std.unicode.Utf8View.initUnchecked(s).iterator();
            while (it.nextCodepointSlice()) |cs| {
                const w = isWordChar(decode(cs));
                if (w and !in_word) n += 1;
                in_word = w;
            }
            return .{ .int = n };
        },
    }
}

/// Python's `\w` for the text a prompt carries: letters, digits and `_`; past ASCII, anything but white space and
/// the common punctuation blocks.
fn isWordChar(cp: u21) bool {
    @setRuntimeSafety(true);
    if (cp < 0x80) return std.ascii.isAlphanumeric(@intCast(cp)) or cp == '_';
    if (value.isSpace(cp)) return false;
    return switch (cp) {
        0x80...0xBF, 0xD7, 0xF7, 0x2000...0x206F, 0x3000...0x303F, 0xFF00...0xFF0F, 0xFF1A...0xFF20 => false,
        else => true,
    };
}

/// Jinja2's multi-attribute getter: always a list of parts, so equal keys compare by `==` and never reach `<`
/// (two equal dicts or two Nones sort; unequal ones fail, as in Python).
fn sortKey(r: *Renderer, item: Value, attr: ?Value, cs: bool) Failure!Value {
    @setRuntimeSafety(true);
    const a = attr orelse return r.newList(&.{try caseKey(r, item, cs)}, .list);
    const text = a.str() orelse return r.newList(&.{try keyOf(r, item, a, cs)}, .list);
    var parts = std.ArrayList(Value).empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |part| try parts.append(r.arena, try keyOf(r, item, .plain(part), cs));
    return r.newList(parts.items, .list);
}

fn floatToInt(r: *Renderer, x: f64) Failure!Value {
    @setRuntimeSafety(true);
    if (!std.math.isFinite(x)) return r.fail("cannot convert float {d} to integer", .{x});
    const t = @trunc(x);
    if (@abs(t) >= 9.2e18) return r.fail("an integer past 64 bits is not supported", .{});
    return .{ .int = @intFromFloat(t) };
}

/// Python `int(text, base)`: surrounding white space, a sign, a prefix that matches the base (base 0 reads it),
/// `_` between digits.
fn parsePyInt(text: []const u8, base_arg: i64) error{ Invalid, Overflow }!i64 {
    @setRuntimeSafety(true);
    if (base_arg < 0 or base_arg == 1 or base_arg > 36) return error.Invalid;
    var t = strip(text, &white_space, true, true);
    const neg = t.len > 0 and t[0] == '-';
    if (t.len > 0 and (t[0] == '-' or t[0] == '+')) t = t[1..];
    var base: u8 = @intCast(base_arg);
    var prefixed = false;
    if (t.len >= 2 and t[0] == '0') {
        const pb: u8 = switch (std.ascii.toLower(t[1])) {
            'x' => 16,
            'o' => 8,
            'b' => 2,
            else => 0,
        };
        if (pb != 0 and (base == 0 or base == pb)) {
            base = pb;
            t = t[2..];
            prefixed = true;
        }
    }
    if (base == 0) {
        base = 10;
        if (t.len > 0 and t[0] == '0') for (t) |c| if (c != '0' and c != '_') return error.Invalid;
    }
    if (prefixed and t.len > 0 and t[0] == '_') t = t[1..];
    if (t.len == 0 or t[0] == '_' or t[t.len - 1] == '_') return error.Invalid;
    var acc: i128 = 0;
    var prev_underscore = false;
    for (t) |c| {
        if (c == '_') {
            if (prev_underscore) return error.Invalid;
            prev_underscore = true;
            continue;
        }
        prev_underscore = false;
        const d = std.fmt.charToDigit(c, base) catch return error.Invalid;
        acc = acc * base + d;
        if (acc > @as(i128, std.math.maxInt(i64)) + 1) return error.Overflow;
    }
    if (neg) acc = -acc;
    return std.math.cast(i64, acc) orelse error.Overflow;
}

/// Python `float(text)`: surrounding white space, a sign, `_` between digits, `inf` and `nan`.
fn parsePyFloat(text: []const u8) ?f64 {
    @setRuntimeSafety(true);
    const t = strip(text, &white_space, true, true);
    if (t.len == 0) return null;
    for (t, 0..) |c, i| if (c == '_') {
        if (i == 0 or i + 1 == t.len or !std.ascii.isDigit(t[i - 1]) or !std.ascii.isDigit(t[i + 1])) return null;
    };
    const body = if (t[0] == '+' or t[0] == '-') t[1..] else t;
    if (body.len == 0) return null;
    if (std.ascii.eqlIgnoreCase(body, "inf") or std.ascii.eqlIgnoreCase(body, "infinity") or std.ascii.eqlIgnoreCase(body, "nan")) {
        const v: f64 = if (std.ascii.eqlIgnoreCase(body, "nan")) std.math.nan(f64) else std.math.inf(f64);
        return if (t[0] == '-') -v else v;
    }
    for (body) |c| if (!(std.ascii.isDigit(c) or c == '.' or c == 'e' or c == 'E' or c == '+' or c == '-' or c == '_')) return null;
    return std.fmt.parseFloat(f64, t) catch null;
}

/// Python `str.splitlines()`.
fn splitLines(r: *Renderer, s: []const u8) Failure![]const []const u8 {
    @setRuntimeSafety(true);
    var lines = std.ArrayList([]const u8).empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        const len = std.unicode.utf8ByteSequenceLength(s[i]) catch 1;
        const cp = decode(s[i..@min(s.len, i + len)]);
        const brk = switch (cp) {
            '\n', '\r', 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029 => true,
            else => false,
        };
        if (!brk) {
            i += len;
            continue;
        }
        try lines.append(r.arena, s[start..i]);
        i += len;
        if (cp == '\r' and i < s.len and s[i] == '\n') i += 1;
        start = i;
    }
    if (start < s.len) try lines.append(r.arena, s[start..]);
    return lines.items;
}

/// Python `len(v)`.
pub fn length(r: *Renderer, v: Value) Failure!usize {
    @setRuntimeSafety(true);
    return switch (v) {
        .undefined => 0,
        .string, .markup => |s| blk: {
            try r.scan(s.bytes.len);
            break :blk render.cpCount(s.bytes);
        },
        .list => |l| if (l.isIterator()) r.fail("object of type '{s}' has no len()", .{@tagName(l.kind)}) else l.items.items.len,
        .dict => |d| d.entries.items.len,
        .loop => |l| l.len(),
        else => r.fail("object of type '{s}' has no len()", .{value.typeName(v)}),
    };
}

/// `list.append(v)` that grows into a new buffer and never frees the old one: a for loop walking the list
/// reads the old buffer, and `Allocator.free` poisons what it frees in a safe build whatever the allocator does.
fn appendKeeping(r: *Renderer, l: *value.List, v: Value) Failure!void {
    @setRuntimeSafety(true);
    const items = &l.items;
    if (items.items.len == items.capacity) {
        const cap = @max(8, items.capacity *| 2);
        const buf = try r.arena.alloc(Value, cap);
        @memcpy(buf[0..items.items.len], items.items);
        items.items = buf[0..items.items.len];
        items.capacity = cap;
    }
    items.appendAssumeCapacity(v);
}

fn dictView(r: *Renderer, d: *const value.Dict, kind: value.List.Kind) Failure!Value {
    @setRuntimeSafety(true);
    const out = try r.arena.alloc(Value, d.entries.items.len);
    for (d.entries.items, out) |e, *o| o.* = switch (kind) {
        .dict_keys => e.key,
        .dict_values => e.value,
        else => try r.newList(&.{ e.key, e.value }, .tuple),
    };
    return r.newList(out, kind);
}

// ── Tests ──

pub fn applyTest(r: *Renderer, t: Test, v: Value, args: []const Value, kws: []const Kw) Failure!bool {
    @setRuntimeSafety(true);
    switch (t) {
        .divisibleby, .eq, .equalto, .ne, .ge, .gt, .greaterthan, .lt, .lessthan, .in, .sameas => {
            const p = try bind(r, @tagName(t), &.{"other"}, args, kws);
            const other = try required(r, @tagName(t), p[0]);
            return switch (t) {
                .divisibleby => value.eql(&r.meter, try pyMod(r, v, other), .{ .int = 0 }, 0) catch |e| r.wrap(e, "the values cannot be compared"),
                .eq, .equalto => value.eql(&r.meter, v, other, 0) catch |e| r.wrap(e, "the values nest too deep to compare"),
                .ne => !(value.eql(&r.meter, v, other, 0) catch |e| return r.wrap(e, "the values nest too deep to compare")),
                .lt, .lessthan => lessThan(r, v, other),
                .gt, .greaterthan => lessThan(r, other, v),
                .ge => if (try compare(r, v, other)) |o| o != .lt else false,
                .in => contains(r, other, v),
                .sameas => sameAs(v, other),
                else => unreachable,
            };
        },
        else => {},
    }
    _ = try bind(r, @tagName(t), &.{}, args, kws);
    return switch (t) {
        .boolean => v == .boolean,
        .callable => v == .macro or v == .function or v == .loop or v == .undefined,
        .defined => v != .undefined,
        .undefined => v == .undefined,
        .escaped => v == .markup,
        .even, .odd => blk: {
            try undefinedFail(r, v);
            const m = try pyMod(r, v, .{ .int = 2 });
            break :blk value.eql(&r.meter, m, .{ .int = @intFromBool(t == .odd) }, 0) catch |e| return r.wrap(e, "the values cannot be compared");
        },
        .@"false" => v == .boolean and !v.boolean,
        .@"true" => v == .boolean and v.boolean,
        .filter, .@"test" => blk: {
            const name = switch (v) {
                .string, .markup => |s| s.bytes,
                .list => |l| if (l.kind == .tuple) break :blk false else return r.fail("unhashable type: '{s}'", .{@tagName(l.kind)}),
                .dict, .namespace => return r.fail("unhashable type: '{s}'", .{value.typeName(v)}),
                else => break :blk false,
            };
            break :blk among(name, if (t == .filter) &jinja2_filters else &jinja2_tests);
        },
        .float => v == .float,
        .integer => v == .int or v == .bigint,
        .number => v.isNumber(),
        .none => v == .none,
        .string => v == .string or v == .markup,
        .mapping => v == .dict,
        .iterable => switch (v) {
            .undefined, .string, .markup, .dict, .loop => true,
            .list => true,
            else => false,
        },
        .sequence => switch (v) {
            .undefined, .string, .markup, .dict => true,
            .list => |l| l.isSequence(),
            else => false,
        },
        .lower, .upper => hasCase(try r.toStr(v), t == .upper),
        else => unreachable,
    };
}

/// Python's `is`: the same object. Small integers, None and the booleans are shared; containers by identity.
fn sameAs(a: Value, b: Value) bool {
    @setRuntimeSafety(true);
    return switch (a) {
        .none => b == .none,
        .boolean => |x| b == .boolean and b.boolean == x,
        .int => |x| b == .int and b.int == x and x >= -5 and x <= 256,
        .string, .markup => |s| if (b.str()) |t| s.bytes.ptr == t.ptr and s.bytes.len == t.len and @intFromEnum(a) == @intFromEnum(b) else false,
        .list => |l| b == .list and b.list == l,
        .dict => |d| b == .dict and b.dict == d,
        .namespace => |d| b == .namespace and b.namespace == d,
        .macro => |m| b == .macro and b.macro == m,
        .function => |f| b == .function and b.function == f,
        .loop => |l| b == .loop and b.loop == l,
        .undefined, .bigint, .float => false,
    };
}

// ── Calls: methods and globals ──

pub fn callFunction(r: *Renderer, f: value.Function, args: []const Value, kws: []const Kw) Failure!Value {
    @setRuntimeSafety(true);
    switch (f) {
        .global => |g| return callGlobal(r, g, args, kws),
        .method => |m| return callMethod(r, m.receiver, m.method, args, kws),
    }
}

fn noKeywords(r: *Renderer, what: []const u8, kws: []const Kw) Failure!void {
    @setRuntimeSafety(true);
    if (kws.len > 0) return r.fail("{s}() takes no keyword arguments", .{what});
}

fn callMethod(r: *Renderer, recv: Value, m: value.Method, args: []const Value, kws: []const Kw) Failure!Value {
    @setRuntimeSafety(true);
    if (recv.str()) |text| try r.scan(text.len);
    switch (m) {
        .split, .rsplit => {
            const p = try bind(r, @tagName(m), &.{ "sep", "maxsplit" }, args, kws);
            return split(r, recv.strOf().?, p[0], p[1], m == .rsplit);
        },
        .indent => {
            const p = try bind(r, "indent", &.{ "width", "first", "blank" }, args, kws);
            const pad_origin: value.Origin = if (p[0]) |w| value.originIn(w, 0) else .template;
            return .{ .string = .{ .bytes = try cppIndent(r, recv.str().?, p[0], p[1], p[2]), .origin = recv.strOf().?.origin.strictest(pad_origin) } };
        },
        .append => {
            try noKeywords(r, "append", kws);
            if (args.len != 1) return r.fail("append() takes exactly one argument ({d} given)", .{args.len});
            try appendKeeping(r, recv.list, args[0]);
            return recv;
        },
        .get, .keys, .values, .items => {
            try noKeywords(r, @tagName(m), kws);
            const d = recv.dict;
            switch (m) {
                .get => {
                    if (args.len < 1 or args.len > 2) return r.fail("get expected 1 or 2 arguments, got {d}", .{args.len});
                    try r.hashable(args[0]);
                    return try d.get(&r.meter, args[0]) orelse if (args.len > 1) args[1] else .none;
                },
                else => {
                    if (args.len > 0) return r.fail("{s}() takes no arguments ({d} given)", .{ @tagName(m), args.len });
                    return dictView(r, d, switch (m) {
                        .keys => .dict_keys,
                        .values => .dict_values,
                        else => .dict_items,
                    });
                },
            }
        },
        else => {},
    }
    try noKeywords(r, @tagName(m), kws);
    const src = recv.strOf().?;
    const s = src.bytes;
    switch (m) {
        .startswith, .endswith => return .{ .boolean = try affix(r, s, args, m == .endswith) },
        .replace => {
            if (args.len < 2 or args.len > 3) return r.fail("replace expected 2 or 3 arguments, got {d}", .{args.len});
            const count = if (args.len > 2) try intArg(r, args[2], "replace") else -1;
            const new = args[1].strOf() orelse return r.fail("replace: expected a string, got '{s}'", .{value.typeName(args[1])});
            return sameKind(recv, try replace(r, src, try strArg(r, args[0], "replace"), new, count));
        },
        .strip, .lstrip, .rstrip => {
            if (args.len > 1) return r.fail("{s} expected at most 1 argument, got {d}", .{ @tagName(m), args.len });
            const chars = try stripArg(r, if (args.len > 0) args[0] else null);
            return sameKind(recv, try within(r, src, try stripChars(r, s, chars, m != .rstrip, m != .lstrip)));
        },
        .upper, .lower, .title, .capitalize => {
            if (args.len > 0) return r.fail("{s}() takes no arguments ({d} given)", .{ @tagName(m), args.len });
            return sameKind(recv, src.remap(switch (m) {
                .upper => try mapCase(r, s, true),
                .lower => try mapCase(r, s, false),
                .title => try strTitle(r, s),
                else => try capitalize(r, s),
            }));
        },
        else => unreachable,
    }
}

/// jinja.cpp's `str.indent(width, first, blank)`: lines split at `\n`, a trailing newline kept (indented when
/// `blank`), and an empty first line leaves the next one counted as first.
fn cppIndent(r: *Renderer, s: []const u8, width: ?Value, first_v: ?Value, blank_v: ?Value) Failure![]const u8 {
    @setRuntimeSafety(true);
    const pad: []const u8 = if (width) |w| switch (w) {
        .int => |n| blk: {
            if (n < 0) return r.fail("indent: a negative width", .{});
            try r.charge(@intCast(n));
            const b = try r.arena.alloc(u8, @intCast(n));
            @memset(b, ' ');
            break :blk b;
        },
        .string, .markup => |t| t.bytes,
        else => "    ",
    } else "    ";
    const first = if (first_v) |v| value.truthy(v) else false;
    const blank = if (blank_v) |v| value.truthy(v) else false;
    var out = std.ArrayList(u8).empty;
    var lines = std.mem.splitScalar(u8, s, '\n');
    while (lines.next()) |line| {
        if (lines.index == null and line.len == 0) break;
        if (out.items.len > 0) try out.append(r.arena, '\n');
        if (if (out.items.len == 0) first else line.len > 0 or blank) try out.appendSlice(r.arena, pad);
        try out.appendSlice(r.arena, line);
        try r.charge(out.items.len);
    }
    if (s.len > 0 and s[s.len - 1] == '\n') {
        try out.append(r.arena, '\n');
        if (blank) try out.appendSlice(r.arena, pad);
    }
    try r.charge(out.items.len);
    return out.items;
}

/// The largest `range` the sandbox allows.
const max_range = 100_000;

pub const RangeOf = struct { start: i64, stop: i64, stride: i64, len: usize };

/// The bounds and length of Python's `range(args)`, refused past the sandbox's MAX_RANGE.
pub fn rangeOf(r: *Renderer, args: []const Value, kws: []const Kw) Failure!RangeOf {
    @setRuntimeSafety(true);
    try noKeywords(r, "range", kws);
    if (args.len < 1 or args.len > 3) return r.fail("range expected 1 to 3 arguments, got {d}", .{args.len});
    var start: i64 = 0;
    var stop: i64 = undefined;
    var stride: i64 = 1;
    if (args.len == 1) {
        stop = try intArg(r, args[0], "range");
    } else {
        start = try intArg(r, args[0], "range");
        stop = try intArg(r, args[1], "range");
        if (args.len == 3) stride = try intArg(r, args[2], "range");
    }
    if (stride == 0) return r.fail("range() arg 3 must not be zero", .{});
    const span: i128 = if (stride > 0) @as(i128, stop) - start else @as(i128, start) - stop;
    const step_abs: i128 = @intCast(@abs(@as(i128, stride)));
    const n: i128 = if (span <= 0) 0 else @divFloor(span + step_abs - 1, step_abs);
    if (n > max_range) return r.fail("Range too big. The sandbox blocks ranges larger than MAX_RANGE ({d}).", .{max_range});
    return .{ .start = start, .stop = stop, .stride = stride, .len = @intCast(n) };
}

fn callGlobal(r: *Renderer, g: value.Global, args: []const Value, kws: []const Kw) Failure!Value {
    @setRuntimeSafety(true);
    switch (g) {
        .range => {
            const b = try rangeOf(r, args, kws);
            const lazy = value.Loop{ .index0 = 0, .items = &.{}, .range = .{ .start = b.start, .step = b.stride, .len = b.len } };
            const items = try r.arena.alloc(Value, b.len);
            for (items, 0..) |*v, i| v.* = lazy.at(i);
            const range = try r.ownList(items, .range);
            const bounds = try r.arena.create([3]i128);
            bounds.* = .{ b.start, b.stop, b.stride };
            range.list.bounds = bounds;
            return range;
        },
        .namespace => {
            if (args.len > 1) return r.fail("namespace expected at most 1 argument, got {d}", .{args.len});
            const d = try r.arena.create(value.Dict);
            d.* = .{};
            if (args.len == 1) switch (args[0]) {
                .dict => |src| for (src.entries.items) |e| try d.put(&r.meter, r.arena, e.key, e.value),
                else => for (try r.iterate(args[0])) |pair| {
                    const kv = if (pair == .list and pair.list.items.items.len == 2) pair.list.items.items else return r.fail("namespace: each item must be a key and a value", .{});
                    try d.put(&r.meter, r.arena, kv[0], kv[1]);
                },
            };
            for (kws) |kw| try d.put(&r.meter, r.arena, .plain(kw.name), kw.value);
            return .{ .namespace = d };
        },
        .raise_exception => {
            const p = try bind(r, "raise_exception", &.{"message"}, args, kws);
            return r.raise(try r.toStr(try required(r, "raise_exception", p[0])));
        },
        .strftime_now => {
            const p = try bind(r, "strftime_now", &.{"format"}, args, kws);
            const given = try required(r, "strftime_now", p[0]);
            const fmt = given.strOf() orelse return r.fail("strftime_now: expected a string, got '{s}'", .{value.typeName(given)});
            return .{ .string = fmt.derive(try strftimeNow(r, fmt.bytes)) };
        },
    }
}

const Tm = extern struct {
    sec: c_int,
    min: c_int,
    hour: c_int,
    mday: c_int,
    mon: c_int,
    year: c_int,
    wday: c_int,
    yday: c_int,
    isdst: c_int,
    gmtoff: c_long,
    zone: ?[*:0]const u8,
};
extern "c" fn time(t: ?*c_long) c_long;
extern "c" fn localtime_r(t: *const c_long, out: *Tm) ?*Tm;
extern "c" fn strftime(buf: [*]u8, max: usize, fmt: [*:0]const u8, tm: *const Tm) usize;

/// `datetime.now().strftime(fmt)` through the C library Python's own `strftime` calls.
fn strftimeNow(r: *Renderer, fmt: []const u8) Failure![]const u8 {
    @setRuntimeSafety(true);
    if (fmt.len == 0) return "";
    if (std.mem.indexOfScalar(u8, fmt, 0) != null) return r.fail("strftime_now: embedded null character", .{});
    var now = time(null);
    var tm: Tm = undefined;
    if (localtime_r(&now, &tm) == null) return r.fail("strftime_now: the local time is unknown", .{});
    const fmt_z = try r.arena.dupeSentinel(u8, fmt, 0);
    var size: usize = 256 + fmt.len * 8;
    while (size <= r.limits.max_string) : (size *= 4) {
        const buf = try r.arena.alloc(u8, size);
        const n = strftime(buf.ptr, buf.len, fmt_z.ptr, &tm);
        if (n > 0) {
            try r.charge(n);
            return buf[0..n];
        }
    }
    return r.fail("strftime_now: the formatted time exceeds the render's limit", .{});
}
