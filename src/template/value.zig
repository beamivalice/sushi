//! Template values with Python's meaning (Jinja2 renders Python objects): equality, ordering, truth, `str` and
//! `repr`, and the conversion of JSON input as `json.loads` reads it.
const std = @import("std");
const builtin = @import("builtin");
const ast = @import("ast.zig");
const pyfloat = @import("tojson.zig");

pub const Allocator = std.mem.Allocator;

/// What a failed operation reports; `render.zig` turns it into the template error with its position.
pub const Failure = error{ TemplateRuntime, TemplateRaised, OutOfMemory };

/// The work a render may do, shared by everything that walks values: a statement, an iteration, a call, an item
/// visited, a pair compared or 64 bytes scanned is one step. Past `max` a step fails with `TemplateRuntime`.
/// It is also where a walk says why it stopped: a walk returns a bare `TemplateRuntime` after `fail`, and the renderer
/// names it from `why` (`Renderer.named`).
pub const Meter = struct {
    used: u64 = 0,
    max: u64 = std.math.maxInt(u64),
    why: ?Why = null,

    pub const Why = union(enum) {
        steps,
        depth,
        /// Python's TypeError for `<` between values of these two types.
        order: [2][]const u8,
        /// A NaN met by a sort, whose order Python leaves to its algorithm.
        nan,
        /// A value of this type Python prints with its address.
        print: []const u8,
        /// A value of this type JSON cannot hold.
        json: []const u8,
        /// Text past the render's string limit.
        text,
        /// A range whose bounds pass what i128 holds.
        range,
    };

    pub fn tick(self: *Meter, n: u64) Failure!void {
        @setRuntimeSafety(true);
        self.used +|= n;
        if (self.used > self.max) return self.fail(.steps);
    }

    /// Records why the walk stops, unless an earlier stop is not yet named.
    pub fn fail(self: *Meter, why: Why) Failure {
        @setRuntimeSafety(true);
        if (self.why == null) self.why = why;
        return error.TemplateRuntime;
    }

    /// Why the last walk stopped, cleared.
    pub fn take(self: *Meter) ?Why {
        @setRuntimeSafety(true);
        defer self.why = null;
        return self.why;
    }

    pub fn spent(self: *const Meter) bool {
        @setRuntimeSafety(true);
        return self.used > self.max;
    }
};

/// Test builds only: the work the walks did, in the meter's units (a value, an origin run or a code point visited, 64
/// bytes read), counted where it is done, so a test can hold it to the steps the meter charged.
pub threadlocal var work: u64 = 0;
/// Test builds only: bytes read a code point at a time, 64 of them one unit of `work`.
pub threadlocal var work_bytes: u64 = 0;

pub inline fn spend(n: u64) void {
    @setRuntimeSafety(true);
    if (builtin.is_test) work +|= n;
}

/// Test builds only: the work done since both counters were zeroed.
pub fn workDone() u64 {
    @setRuntimeSafety(true);
    return work +| work_bytes / 64;
}

/// Who wrote a string's bytes, from least to most guarded: the template's own text, an assistant message (the model's
/// earlier output), any other text a request brings. The tokenizer reads a special token's text as that token only
/// where its bytes' origin allows (`Tokenizer.encodeSpans`).
pub const Origin = enum(u8) {
    template,
    assistant,
    input,

    pub fn strictest(a: Origin, b: Origin) Origin {
        @setRuntimeSafety(true);
        return @enumFromInt(@max(@intFromEnum(a), @intFromEnum(b)));
    }
};

/// The bytes from the previous run's end up to `end` are `origin`'s.
pub const Run = struct { end: usize, origin: Origin };

/// A string's bytes and who wrote them: every byte `origin`'s when `runs` is empty, else the runs in order, the last
/// ending at `bytes.len`, and `origin` the strictest of them.
pub const Str = struct {
    bytes: []const u8,
    origin: Origin = .template,
    runs: []const Run = &.{},

    /// Bytes `a..b`, each keeping its origin.
    pub fn slice(self: Str, arena: Allocator, a: usize, b: usize) error{OutOfMemory}!Str {
        @setRuntimeSafety(true);
        if (self.runs.len == 0) return .{ .bytes = self.bytes[a..b], .origin = self.origin };
        var out: Builder = .{};
        try out.appendPart(arena, self, a, b);
        return out.str();
    }

    /// `bytes`, a map of these bytes one for one (ASCII case), keeping each byte's origin.
    pub fn remap(self: Str, bytes: []const u8) Str {
        @setRuntimeSafety(true);
        std.debug.assert(bytes.len == self.bytes.len);
        return .{ .bytes = bytes, .origin = self.origin, .runs = self.runs };
    }

    /// The one origin of bytes `a..b`, or null when more than one wrote them.
    pub fn originOf(self: Str, a: usize, b: usize) ?Origin {
        @setRuntimeSafety(true);
        if (self.runs.len == 0) return self.origin;
        var found: ?Origin = null;
        var start: usize = 0;
        for (self.runs) |r| {
            defer start = r.end;
            if (r.end <= a or start >= b) continue;
            if (found) |f| if (f != r.origin) return null;
            found = r.origin;
        }
        return found orelse self.origin;
    }

    /// The end of the last byte `origin` wrote, or null when it wrote none.
    pub fn lastEnd(self: Str, origin: Origin) ?usize {
        @setRuntimeSafety(true);
        if (self.runs.len == 0) return if (self.origin == origin and self.bytes.len > 0) self.bytes.len else null;
        var i = self.runs.len;
        while (i > 0) {
            i -= 1;
            const start = if (i == 0) 0 else self.runs[i - 1].end;
            if (self.runs[i].origin == origin and self.runs[i].end > start) return self.runs[i].end;
        }
        return null;
    }

    /// `bytes` made from these by a change with no byte-for-byte map: every byte takes their strictest origin.
    pub fn derive(self: Str, bytes: []const u8) Str {
        @setRuntimeSafety(true);
        return .{ .bytes = bytes, .origin = self.origin };
    }
};

/// Builds a string piece by piece, each piece keeping its origins.
pub const Builder = struct {
    bytes: std.ArrayListUnmanaged(u8) = .empty,
    runs: std.ArrayListUnmanaged(Run) = .empty,

    pub fn append(self: *Builder, gpa: Allocator, piece: Str) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        return self.appendPart(gpa, piece, 0, piece.bytes.len);
    }

    /// Bytes `a..b` of `piece`.
    pub fn appendPart(self: *Builder, gpa: Allocator, piece: Str, a: usize, b: usize) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        if (a >= b) return;
        spend((b - a) / 64 + 1);
        try self.bytes.appendSlice(gpa, piece.bytes[a..b]);
        if (piece.runs.len == 0) return self.mark(gpa, piece.origin);
        const base = self.bytes.items.len - (b - a);
        // The first run ending past `a`, by binary search: a string cut a character at a time must not rescan its runs.
        var lo: usize = 0;
        var hi: usize = piece.runs.len;
        while (lo < hi) {
            spend(1);
            const mid = lo + (hi - lo) / 2;
            if (piece.runs[mid].end > a) hi = mid else lo = mid + 1;
        }
        for (piece.runs[lo..]) |run| {
            spend(1);
            try self.markTo(gpa, base + @min(run.end, b) - a, run.origin);
            if (run.end >= b) break;
        }
    }

    pub fn appendBytes(self: *Builder, gpa: Allocator, bytes: []const u8, origin: Origin) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        if (bytes.len == 0) return;
        try self.bytes.appendSlice(gpa, bytes);
        try self.mark(gpa, origin);
    }

    /// The bytes not yet in a run are `origin`'s.
    fn mark(self: *Builder, gpa: Allocator, origin: Origin) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        return self.markTo(gpa, self.bytes.items.len, origin);
    }

    fn markTo(self: *Builder, gpa: Allocator, end: usize, origin: Origin) error{OutOfMemory}!void {
        @setRuntimeSafety(true);
        const runs = self.runs.items;
        if (runs.len > 0 and runs[runs.len - 1].origin == origin) {
            runs[runs.len - 1].end = end;
            return;
        }
        try self.runs.append(gpa, .{ .end = end, .origin = origin });
    }

    pub fn deinit(self: *Builder, gpa: Allocator) void {
        @setRuntimeSafety(true);
        self.bytes.deinit(gpa);
        self.runs.deinit(gpa);
    }

    /// What was built; it references the builder's memory.
    pub fn str(self: *const Builder) Str {
        @setRuntimeSafety(true);
        const runs = self.runs.items;
        if (runs.len <= 1) return .{ .bytes = self.bytes.items, .origin = if (runs.len == 1) runs[0].origin else .template };
        var o: Origin = .template;
        for (runs) |run| o = o.strictest(run.origin);
        return .{ .bytes = self.bytes.items, .origin = o, .runs = runs };
    }
};

/// The code points of `bytes`, each as its byte slice. Text the engine holds is valid UTF-8 (the template is checked
/// when it compiles, its input at the door); the walk still never reads past the end, a bad byte being a slice of one.
pub const Codepoints = struct {
    bytes: []const u8,
    i: usize = 0,

    pub fn nextCodepointSlice(self: *Codepoints) ?[]const u8 {
        @setRuntimeSafety(true);
        if (self.i >= self.bytes.len) return null;
        const want = std.unicode.utf8ByteSequenceLength(self.bytes[self.i]) catch 1;
        const n = @min(want, self.bytes.len - self.i);
        if (builtin.is_test) work_bytes +|= n;
        defer self.i += n;
        return self.bytes[self.i..][0..n];
    }
};

pub fn codepoints(bytes: []const u8) Codepoints {
    @setRuntimeSafety(true);
    return .{ .bytes = bytes };
}

pub const Value = union(enum) {
    /// Jinja2's `Undefined`; the name looked up, for the message when it is used.
    undefined: []const u8,
    none,
    boolean: bool,
    int: i64,
    /// An integer outside i64, as decimal text: printed exactly, never computed with.
    bigint: []const u8,
    float: f64,
    string: Str,
    /// Jinja2's `Markup` (the `safe` filter's result): a string that escapes what is added to it.
    markup: Str,
    list: *List,
    dict: *Dict,
    namespace: *Dict,
    macro: *const Macro,
    function: *const Function,
    loop: *Loop,

    /// The template's own text as a string.
    pub fn plain(bytes: []const u8) Value {
        @setRuntimeSafety(true);
        return .{ .string = .{ .bytes = bytes } };
    }

    pub fn str(self: Value) ?[]const u8 {
        @setRuntimeSafety(true);
        return switch (self) {
            .string, .markup => |s| s.bytes,
            else => null,
        };
    }

    /// A string's bytes with their origins.
    pub fn strOf(self: Value) ?Str {
        @setRuntimeSafety(true);
        return switch (self) {
            .string, .markup => |s| s,
            else => null,
        };
    }

    pub fn isNumber(self: Value) bool {
        @setRuntimeSafety(true);
        return switch (self) {
            .boolean, .int, .bigint, .float => true,
            else => false,
        };
    }
};

pub const List = struct {
    items: std.ArrayListUnmanaged(Value) = .empty,
    kind: Kind = .list,
    /// A range's start, stop and step, as Python prints it; null past what i128 holds (it then cannot print).
    bounds: ?*const [3]i128 = null,

    /// The Python type behind the items: an iterator has no length, no index and is consumed by a walk; a dict
    /// view has a length and no index; a range prints its bounds.
    pub const Kind = enum { list, tuple, range, generator, list_reverseiterator, dict_keys, dict_values, dict_items };

    pub fn isIterator(self: *const List) bool {
        @setRuntimeSafety(true);
        return self.kind == .generator or self.kind == .list_reverseiterator;
    }

    pub fn isSequence(self: *const List) bool {
        @setRuntimeSafety(true);
        return self.kind == .list or self.kind == .tuple or self.kind == .range;
    }
};

pub const Entry = struct { key: Value, value: Value };

/// A Python dict: insertion order kept, keys compared as Python compares them.
pub const Dict = struct {
    entries: std.ArrayListUnmanaged(Entry) = .empty,

    pub fn get(self: *const Dict, m: *Meter, key: Value) Failure!?Value {
        @setRuntimeSafety(true);
        const i = try self.indexOf(m, key) orelse return null;
        return self.entries.items[i].value;
    }

    pub fn getStr(self: *const Dict, m: *Meter, key: []const u8) Failure!?Value {
        @setRuntimeSafety(true);
        for (self.entries.items) |e| {
            try m.tick(1);
            if (e.key.str()) |k| if (try strEql(m, k, key)) return e.value;
        }
        return null;
    }

    pub fn indexOf(self: *const Dict, m: *Meter, key: Value) Failure!?usize {
        @setRuntimeSafety(true);
        for (self.entries.items, 0..) |e, i| if (try eql(m, e.key, key, 0)) return i;
        return null;
    }

    /// Python's `d[key] = value`: a key already present keeps its place.
    pub fn put(self: *Dict, m: *Meter, gpa: Allocator, key: Value, value: Value) Failure!void {
        @setRuntimeSafety(true);
        if (try self.indexOf(m, key)) |i| {
            self.entries.items[i].value = value;
            return;
        }
        try self.entries.append(gpa, .{ .key = key, .value = value });
    }
};

pub const Macro = struct {
    node: *const ast.Macro,
    /// The scope the macro was defined in (`*render.Frame`).
    closure: *anyopaque,
    /// A call block's `caller`: the macro that renders the block body.
    is_caller: bool = false,
};

pub const Global = enum { range, namespace, raise_exception, strftime_now };

pub const Method = enum {
    // str
    upper,
    lower,
    title,
    capitalize,
    strip,
    lstrip,
    rstrip,
    startswith,
    endswith,
    split,
    rsplit,
    replace,
    /// jinja.cpp's, not Python's.
    indent,
    // dict
    get,
    keys,
    values,
    items,
    // list: jinja.cpp's, not the sandbox's (which refuses a mutator).
    append,
};

pub const Function = union(enum) {
    global: Global,
    method: struct { receiver: Value, method: Method },
};

pub const Loop = struct {
    index0: usize,
    /// The items walked, or with `range` set, none: a loop over `range(...)` makes each item as it goes, as
    /// Python's lazy range does, so a loop that breaks early costs only what it walked.
    items: []const Value,
    range: ?Range = null,
    /// How many items the walk took from its iterable, as Jinja2's `LoopContext` takes them: the current one, the
    /// next when `last` or `nextitem` peeks at it, all of them for the length; `len() + 1` once it reached past the
    /// end. An iterator keeps the items after these.
    reach: usize = 0,

    pub const Range = struct { start: i64, step: i64, len: usize };

    /// The walk has taken the items up to `n`.
    pub fn took(self: *Loop, n: usize) void {
        @setRuntimeSafety(true);
        self.reach = @max(self.reach, @min(n, self.len() + 1));
    }

    pub fn len(self: *const Loop) usize {
        @setRuntimeSafety(true);
        return if (self.range) |r| r.len else self.items.len;
    }

    pub fn at(self: *const Loop, i: usize) Value {
        @setRuntimeSafety(true);
        const r = self.range orelse return self.items[i];
        // Every item lies between start and stop, so it fits i64; the product on the way may not.
        return .{ .int = @intCast(@as(i128, r.start) + @as(i128, @intCast(i)) * r.step) };
    }
};

pub const max_depth = 512;

fn depthCheck(m: *Meter, depth: usize) Failure!void {
    @setRuntimeSafety(true);
    noteStack();
    if (depth > max_depth) return m.fail(.depth);
}

/// Test builds only: the lowest frame address a render reached on this thread, for the stack-depth test.
pub threadlocal var stack_low: usize = std.math.maxInt(usize);

pub inline fn noteStack() void {
    @setRuntimeSafety(true);
    if (builtin.is_test) stack_low = @min(stack_low, @frameAddress());
}

fn numberOf(v: Value) ?union(enum) { int: i64, float: f64, big: []const u8 } {
    @setRuntimeSafety(true);
    return switch (v) {
        .boolean => |b| .{ .int = @intFromBool(b) },
        .int => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .bigint => |b| .{ .big = b },
        else => null,
    };
}

/// Python `==`.
pub fn eql(m: *Meter, a: Value, b: Value, depth: usize) Failure!bool {
    @setRuntimeSafety(true);
    try depthCheck(m, depth);
    spend(1);
    try m.tick(1);
    if (numberOf(a)) |x| {
        const y = numberOf(b) orelse return false;
        return switch (x) {
            .int => |i| switch (y) {
                .int => |j| i == j,
                .float => |g| @as(f64, @floatFromInt(i)) == g and floatIsInt(g, i),
                .big => false,
            },
            .float => |f| switch (y) {
                .int => |j| f == @as(f64, @floatFromInt(j)) and floatIsInt(f, j),
                .float => |g| f == g,
                .big => |t| orderBig(t, x) == .eq,
            },
            .big => |t| switch (y) {
                .big => |u| std.mem.eql(u8, t, u),
                .float => orderBig(t, y) == .eq,
                .int => false,
            },
        };
    }
    switch (a) {
        .string, .markup => |s| return if (b.str()) |t| strEql(m, s.bytes, t) else false,
        .none => return b == .none,
        .undefined => return b == .undefined,
        .list => |l| {
            if (b != .list) return false;
            if (l == b.list) return true;
            const other = b.list;
            if (l.kind != other.kind or l.isIterator() or l.kind == .dict_values) return false;
            if (l.items.items.len != other.items.items.len) return false;
            if (l.kind == .dict_keys or l.kind == .dict_items) {
                for (l.items.items) |x| {
                    const found = for (other.items.items) |y| {
                        if (try eql(m, x, y, depth + 1)) break true;
                    } else false;
                    if (!found) return false;
                }
                return true;
            }
            for (l.items.items, other.items.items) |x, y| if (!try eql(m, x, y, depth + 1)) return false;
            return true;
        },
        .dict => |d| {
            if (b != .dict) return false;
            if (d.entries.items.len != b.dict.entries.items.len) return false;
            for (d.entries.items) |e| {
                const other = try b.dict.get(m, e.key) orelse return false;
                if (!try eql(m, e.value, other, depth + 1)) return false;
            }
            return true;
        },
        .namespace => |n| return b == .namespace and b.namespace == n,
        .macro => |mac| return b == .macro and b.macro == mac,
        .function => |f| return b == .function and b.function == f,
        .loop => |l| return b == .loop and b.loop == l,
        else => unreachable,
    }
}

fn floatIsInt(f: f64, i: i64) bool {
    @setRuntimeSafety(true);
    if (f != f or f == std.math.inf(f64) or f == -std.math.inf(f64)) return false;
    if (@abs(f) >= 9.2e18) return false;
    return @as(i64, @intFromFloat(f)) == i;
}

pub fn bigToFloat(text: []const u8) f64 {
    @setRuntimeSafety(true);
    return std.fmt.parseFloat(f64, text) catch std.math.nan(f64);
}

/// Knuth-Morris-Pratt search for `needle`, from the start or (`last`) from the end: time linear in the text,
/// whatever it holds, and the same index `std.mem.indexOf` / `lastIndexOf` give.
pub const Finder = struct {
    needle: []const u8,
    /// `border[i]`: the longest proper border of the needle's first i + 1 bytes, read in search order.
    border: []usize,
    last: bool,

    pub fn init(gpa: Allocator, needle: []const u8, last: bool) error{OutOfMemory}!Finder {
        @setRuntimeSafety(true);
        spend(needle.len / 64 + 1);
        const f = Finder{ .needle = needle, .border = try gpa.alloc(usize, needle.len), .last = last };
        if (needle.len == 0) return f;
        f.border[0] = 0;
        var k: usize = 0;
        for (1..needle.len) |i| {
            while (k > 0 and f.at(i) != f.at(k)) k = f.border[k - 1];
            if (f.at(i) == f.at(k)) k += 1;
            f.border[i] = k;
        }
        return f;
    }

    pub fn deinit(self: Finder, gpa: Allocator) void {
        @setRuntimeSafety(true);
        gpa.free(self.border);
    }

    fn at(self: Finder, i: usize) u8 {
        @setRuntimeSafety(true);
        return if (self.last) self.needle[self.needle.len - 1 - i] else self.needle[i];
    }

    /// The first (with `last`, the last) index of the needle in `hay`; `scanned` gets the bytes read.
    pub fn find(self: Finder, hay: []const u8, scanned: *usize) ?usize {
        @setRuntimeSafety(true);
        const m = self.needle.len;
        defer spend(scanned.* / 64 + 1);
        if (m == 0) {
            scanned.* = 0;
            return if (self.last) hay.len else 0;
        }
        var k: usize = 0;
        for (0..hay.len) |n| {
            const c = hay[if (self.last) hay.len - 1 - n else n];
            while (k > 0 and c != self.at(k)) k = self.border[k - 1];
            if (c == self.at(k)) k += 1;
            if (k == m) {
                scanned.* = n + 1;
                return if (self.last) hay.len - 1 - n else n + 1 - m;
            }
        }
        scanned.* = hay.len;
        return null;
    }
};

/// Python's `s == t` on text, metered by the bytes it compares.
pub fn strEql(m: *Meter, s: []const u8, t: []const u8) Failure!bool {
    @setRuntimeSafety(true);
    if (s.len != t.len) return false;
    const diff = std.mem.indexOfDiff(u8, s, t);
    spend((diff orelse s.len) / 64);
    try m.tick((diff orelse s.len) / 64);
    return diff == null;
}

/// Python's ordering of `a` and `b`: null when they are unordered (a NaN, where Python's `<`, `>`, `<=` and `>=`
/// all answer false); a pair Python refuses to order is a runtime error.
pub fn order(m: *Meter, a: Value, b: Value, depth: usize) Failure!?std.math.Order {
    @setRuntimeSafety(true);
    try depthCheck(m, depth);
    spend(1);
    try m.tick(1);
    if (numberOf(a)) |x| {
        const y = numberOf(b) orelse return m.fail(.{ .order = .{ typeName(a), typeName(b) } });
        return orderNumbers(x, y);
    }
    if (a.str()) |s| {
        const t = b.str() orelse return m.fail(.{ .order = .{ typeName(a), typeName(b) } });
        try m.tick((std.mem.indexOfDiff(u8, s, t) orelse @min(s.len, t.len)) / 64);
        return std.mem.order(u8, s, t);
    }
    if (a == .list and b == .list and a.list.kind == b.list.kind and (a.list.kind == .list or a.list.kind == .tuple)) {
        const xs = a.list.items.items;
        const ys = b.list.items.items;
        for (0..@min(xs.len, ys.len)) |i| {
            if (!try eql(m, xs[i], ys[i], depth + 1)) return order(m, xs[i], ys[i], depth + 1);
        }
        return std.math.order(xs.len, ys.len);
    }
    return m.fail(.{ .order = .{ typeName(a), typeName(b) } });
}

const Number = @typeInfo(@typeInfo(@TypeOf(numberOf)).@"fn".return_type.?).optional.child;

/// Exact, as Python compares an int with a float: never through a rounded conversion.
fn orderNumbers(x: Number, y: Number) ?std.math.Order {
    @setRuntimeSafety(true);
    return switch (x) {
        .int => |i| switch (y) {
            .int => |j| std.math.order(i, j),
            .float => |g| orderIntFloat(i, g),
            .big => |t| if (orderBig(t, x)) |o| o.invert() else null,
        },
        .float => |f| switch (y) {
            .int => |j| if (orderIntFloat(j, f)) |o| o.invert() else null,
            .float => |g| orderFloats(f, g),
            .big => |t| if (orderBig(t, x)) |o| o.invert() else null,
        },
        .big => |t| orderBig(t, y),
    };
}

/// Python's exact ordering of an integer past 64 bits (decimal `t`, never 0) against any number.
fn orderBig(t: []const u8, y: Number) ?std.math.Order {
    @setRuntimeSafety(true);
    const neg = t[0] == '-';
    const digits = if (neg) t[1..] else t;
    const magnitude: std.math.Order = switch (y) {
        .int => |j| if (neg == (j < 0)) .gt else return if (neg) .lt else .gt,
        .big => |u| blk: {
            const u_neg = u[0] == '-';
            if (neg != u_neg) return if (neg) .lt else .gt;
            const v = if (u_neg) u[1..] else u;
            if (digits.len != v.len) break :blk std.math.order(digits.len, v.len);
            break :blk std.mem.order(u8, digits, v);
        },
        .float => |f| blk: {
            if (std.math.isNan(f)) return null;
            if (neg != (f < 0)) return if (neg) .lt else .gt;
            if (std.math.isInf(f)) break :blk .lt;
            // The integer is at least 2^63; a float below that is smaller, and one above is an integer.
            if (@abs(f) < 9223372036854775808.0) break :blk .gt;
            var buf: [320]u8 = undefined;
            const g = floatDigits(@abs(f), &buf);
            if (digits.len != g.len) break :blk std.math.order(digits.len, g.len);
            break :blk std.mem.order(u8, digits, g);
        },
    };
    return if (neg) magnitude.invert() else magnitude;
}

/// The exact decimal digits of `f`, an integral float of at least 2^63 (so below 2^1024: 309 digits at most).
fn floatDigits(f: f64, buf: *[320]u8) []const u8 {
    @setRuntimeSafety(true);
    const bits: u64 = @bitCast(f);
    var mant: u64 = (bits & ((@as(u64, 1) << 52) - 1)) | (@as(u64, 1) << 52);
    const exp: u64 = ((bits >> 52) & 0x7FF) - 1075;
    // Little-endian decimal digits of the mantissa, then doubled `exp` times.
    var n: usize = 0;
    while (mant > 0) : (mant /= 10) {
        buf[n] = @intCast(mant % 10);
        n += 1;
    }
    for (0..exp) |_| {
        var carry: u8 = 0;
        for (buf[0..n]) |*d| {
            const v = d.* * 2 + carry;
            d.* = v % 10;
            carry = v / 10;
        }
        if (carry > 0) {
            buf[n] = carry;
            n += 1;
        }
    }
    std.mem.reverse(u8, buf[0..n]);
    for (buf[0..n]) |*d| d.* += '0';
    return buf[0..n];
}

fn orderFloats(f: f64, g: f64) ?std.math.Order {
    @setRuntimeSafety(true);
    if (std.math.isNan(f) or std.math.isNan(g)) return null;
    return std.math.order(f, g);
}

fn orderIntFloat(i: i64, f: f64) ?std.math.Order {
    @setRuntimeSafety(true);
    if (std.math.isNan(f)) return null;
    if (f >= 9223372036854775808.0) return .lt;
    if (f < -9223372036854775808.0) return .gt;
    const t = @trunc(f);
    const ti: i64 = @intFromFloat(t);
    if (i != ti) return std.math.order(i, ti);
    return if (f > t) .lt else if (f < t) .gt else .eq;
}

/// Python's `<` for `stableSort`: an unordered pair (a NaN) is a failure (`Meter.Why.nan`), since the order
/// Python gives such keys depends on its sort algorithm.
pub const SortContext = struct {
    meter: *Meter,

    pub fn less(self: *SortContext, a: Value, b: Value) Failure!bool {
        @setRuntimeSafety(true);
        const o = try order(self.meter, a, b, 0) orelse return self.meter.fail(.nan);
        return o == .lt;
    }
};

/// A stable bottom-up merge sort over `items` (`scratch` at least as long) whose comparison may fail. A failure
/// ends it and is returned, `items` still holding every item; a comparison that answers inconsistently orders
/// them some way and never reads or writes outside the two slices.
pub fn stableSort(comptime T: type, items: []T, scratch: []T, ctx: anytype, comptime lessFn: anytype) Failure!void {
    @setRuntimeSafety(true);
    const n = items.len;
    const buf = scratch[0..n];
    var width: usize = 1;
    while (width < n) : (width *= 2) {
        var lo: usize = 0;
        while (lo < n) : (lo += 2 * width) {
            const mid = @min(lo + width, n);
            const hi = @min(lo + 2 * width, n);
            var i = lo;
            var j = mid;
            var k = lo;
            while (i < mid and j < hi) : (k += 1) {
                if (try lessFn(ctx, items[j], items[i])) {
                    buf[k] = items[j];
                    j += 1;
                } else {
                    buf[k] = items[i];
                    i += 1;
                }
            }
            @memcpy(buf[k..][0 .. mid - i], items[i..mid]);
            k += mid - i;
            @memcpy(buf[k..][0 .. hi - j], items[j..hi]);
            @memcpy(items[lo..hi], buf[lo..hi]);
        }
    }
}

fn toF(n: anytype) f64 {
    @setRuntimeSafety(true);
    return switch (n) {
        .int => |i| @floatFromInt(i),
        .float => |f| f,
        .big => |t| bigToFloat(t),
    };
}

/// Python truth.
pub fn truthy(v: Value) bool {
    @setRuntimeSafety(true);
    return switch (v) {
        .undefined, .none => false,
        .boolean => |b| b,
        .int => |i| i != 0,
        .bigint => true,
        .float => |f| f != 0,
        .string, .markup => |s| s.bytes.len > 0,
        .list => |l| l.isIterator() or l.items.items.len > 0,
        .dict => |d| d.entries.items.len > 0,
        .namespace, .macro, .function, .loop => true,
    };
}

pub fn typeName(v: Value) []const u8 {
    @setRuntimeSafety(true);
    return switch (v) {
        .undefined => "Undefined",
        .none => "NoneType",
        .boolean => "bool",
        .int, .bigint => "int",
        .float => "float",
        .string => "str",
        .markup => "Markup",
        .list => |l| @tagName(l.kind),
        .dict => "dict",
        .namespace => "Namespace",
        .macro => "Macro",
        .function => "builtin_function_or_method",
        .loop => "LoopContext",
    };
}

/// The type Python refuses to hash in `v` (a list, a dict, a set-like view, or one inside a tuple); null when
/// Python hashes it. A dict key, an `in` against a dict and `unique` need it.
pub fn unhashable(m: *Meter, v: Value, depth: usize) Failure!?[]const u8 {
    @setRuntimeSafety(true);
    try depthCheck(m, depth);
    spend(1);
    try m.tick(1);
    return switch (v) {
        .dict => "dict",
        .list => |l| switch (l.kind) {
            .list, .dict_keys, .dict_items => @tagName(l.kind),
            .tuple => for (l.items.items) |x| {
                if (try unhashable(m, x, depth + 1)) |t| break t;
            } else null,
            else => null,
        },
        else => null,
    };
}

/// The strictest origin of a string in `v`, keys included: what text printed from it takes.
pub fn originIn(m: *Meter, v: Value, depth: usize) Failure!Origin {
    @setRuntimeSafety(true);
    if (depth > max_depth) return .input;
    spend(1);
    try m.tick(1);
    var o: Origin = .template;
    switch (v) {
        .string, .markup => |s| return s.origin,
        .list => |l| for (l.items.items) |x| {
            o = o.strictest(try originIn(m, x, depth + 1));
        },
        .dict, .namespace => |d| for (d.entries.items) |e| {
            o = o.strictest(try originIn(m, e.key, depth + 1)).strictest(try originIn(m, e.value, depth + 1));
        },
        else => {},
    }
    return o;
}

/// Python `str(v)`, as `{{ v }}` prints it (an undefined value prints nothing).
pub fn writeStr(m: *Meter, w: *std.ArrayList(u8), gpa: Allocator, v: Value, depth: usize) Failure!void {
    @setRuntimeSafety(true);
    switch (v) {
        .undefined => {},
        .string, .markup => |s| try w.appendSlice(gpa, s.bytes),
        else => try writeRepr(m, w, gpa, v, depth),
    }
}

/// Python `repr(v)`.
pub fn writeRepr(m: *Meter, w: *std.ArrayList(u8), gpa: Allocator, v: Value, depth: usize) Failure!void {
    @setRuntimeSafety(true);
    try depthCheck(m, depth);
    spend(1);
    try m.tick(1);
    switch (v) {
        .undefined => try w.appendSlice(gpa, "Undefined"),
        .none => try w.appendSlice(gpa, "None"),
        .boolean => |b| try w.appendSlice(gpa, if (b) "True" else "False"),
        .int => |i| try w.print(gpa, "{d}", .{i}),
        .bigint => |t| try w.appendSlice(gpa, t),
        .float => |f| try pyfloat.writeFloatRepr(w, gpa, f),
        .string => |s| {
            try m.tick(s.bytes.len / 64);
            try writeStrRepr(w, gpa, s.bytes);
        },
        .markup => |s| {
            try m.tick(s.bytes.len / 64);
            try w.appendSlice(gpa, "Markup(");
            try writeStrRepr(w, gpa, s.bytes);
            try w.append(gpa, ')');
        },
        .list => |l| {
            const tuple = l.kind == .tuple;
            switch (l.kind) {
                .list, .tuple => {},
                .dict_keys, .dict_values, .dict_items => try w.print(gpa, "{s}([", .{@tagName(l.kind)}),
                .range => {
                    const b = (l.bounds orelse return m.fail(.range)).*;
                    try w.print(gpa, "range({d}, {d}", .{ b[0], b[1] });
                    if (b[2] != 1) try w.print(gpa, ", {d}", .{b[2]});
                    return w.append(gpa, ')');
                },
                // Python prints these with their address.
                .generator, .list_reverseiterator => return m.fail(.{ .print = @tagName(l.kind) }),
            }
            if (l.kind != .list and !tuple) {
                try writeItems(m, w, gpa, l.items.items, depth);
                return w.appendSlice(gpa, "])");
            }
            try w.append(gpa, if (tuple) '(' else '[');
            try writeItems(m, w, gpa, l.items.items, depth);
            if (tuple and l.items.items.len == 1) try w.append(gpa, ',');
            try w.append(gpa, if (tuple) ')' else ']');
        },
        .dict => |d| try writeDictRepr(m, w, gpa, d, depth),
        .namespace => |d| {
            try w.appendSlice(gpa, "<Namespace ");
            try writeDictRepr(m, w, gpa, d, depth);
            try w.append(gpa, '>');
        },
        .macro => |mac| if (mac.is_caller) try w.appendSlice(gpa, "<Macro anonymous>") else try w.print(gpa, "<Macro '{s}'>", .{mac.node.name}),
        .function => return m.fail(.{ .print = "function" }),
        .loop => |l| {
            l.took(l.len() + 1);
            try w.print(gpa, "<LoopContext {d}/{d}>", .{ l.index0 + 1, l.len() });
        },
    }
}

fn writeItems(m: *Meter, w: *std.ArrayList(u8), gpa: Allocator, items: []const Value, depth: usize) Failure!void {
    @setRuntimeSafety(true);
    for (items, 0..) |item, i| {
        if (i > 0) try w.appendSlice(gpa, ", ");
        try writeRepr(m, w, gpa, item, depth + 1);
    }
}

fn writeDictRepr(m: *Meter, w: *std.ArrayList(u8), gpa: Allocator, d: *const Dict, depth: usize) Failure!void {
    @setRuntimeSafety(true);
    try w.append(gpa, '{');
    for (d.entries.items, 0..) |e, i| {
        if (i > 0) try w.appendSlice(gpa, ", ");
        try writeRepr(m, w, gpa, e.key, depth + 1);
        try w.appendSlice(gpa, ": ");
        try writeRepr(m, w, gpa, e.value, depth + 1);
    }
    try w.append(gpa, '}');
}

/// Python's `str.__repr__`: single quotes unless the text holds one and no double quote.
fn writeStrRepr(w: *std.ArrayList(u8), gpa: Allocator, s: []const u8) Failure!void {
    @setRuntimeSafety(true);
    const quote: u8 = if (std.mem.indexOfScalar(u8, s, '\'') != null and std.mem.indexOfScalar(u8, s, '"') == null) '"' else '\'';
    try w.append(gpa, quote);
    var it = codepoints(s);
    while (it.nextCodepointSlice()) |cs| {
        const cp = std.unicode.utf8Decode(cs) catch 0xFFFD;
        switch (cp) {
            '\\' => try w.appendSlice(gpa, "\\\\"),
            '\n' => try w.appendSlice(gpa, "\\n"),
            '\r' => try w.appendSlice(gpa, "\\r"),
            '\t' => try w.appendSlice(gpa, "\\t"),
            else => {
                if (cp == quote) {
                    try w.append(gpa, '\\');
                    try w.append(gpa, quote);
                } else if (printable(cp)) {
                    try w.appendSlice(gpa, cs);
                } else if (cp < 0x100) {
                    try w.print(gpa, "\\x{x:0>2}", .{cp});
                } else if (cp < 0x10000) {
                    try w.print(gpa, "\\u{x:0>4}", .{cp});
                } else {
                    try w.print(gpa, "\\U{x:0>8}", .{cp});
                }
            },
        }
    }
    try w.append(gpa, quote);
}

/// Python's `str.isprintable` for the code points a prompt carries: controls, separators other than the space,
/// format characters and noncharacters are not.
fn printable(cp: u21) bool {
    @setRuntimeSafety(true);
    return switch (cp) {
        0...0x1F, 0x7F...0xA0, 0xAD => false,
        0x1680, 0x2000...0x200F, 0x2028...0x202F, 0x205F...0x2064, 0x3000, 0xFEFF => false,
        0xD800...0xDFFF, 0xFDD0...0xFDEF, 0xFFFE, 0xFFFF => false,
        else => true,
    };
}

/// Python's `str.isspace`.
pub fn isSpace(cp: u21) bool {
    @setRuntimeSafety(true);
    return switch (cp) {
        0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// The value `json.loads` reads from `text`: key order kept, a repeated key's last value at its first place, a
/// number with a fraction or exponent a float, any other an exact integer.
pub fn fromJson(arena: Allocator, text: []const u8, origin: Origin) JsonError!Value {
    @setRuntimeSafety(true);
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{
        .duplicate_field_behavior = .use_last,
        .parse_numbers = false,
    }) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.InvalidJson;
    return convert(arena, parsed, origin, 0);
}

/// Why JSON input was refused: not JSON, or nested past `max_depth`.
pub const JsonError = error{ OutOfMemory, InvalidJson, TooDeep };

fn convert(arena: Allocator, j: std.json.Value, origin: Origin, depth: usize) JsonError!Value {
    @setRuntimeSafety(true);
    noteStack();
    if (depth > max_depth) return error.TooDeep;
    return switch (j) {
        .null => .none,
        .bool => |b| .{ .boolean = b },
        .integer => |i| .{ .int = i },
        .float => |f| .{ .float = f },
        .number_string => |n| try number(n),
        .string => |s| .{ .string = .{ .bytes = s, .origin = origin } },
        .array => |a| blk: {
            const l = try arena.create(List);
            l.* = .{};
            try l.items.ensureTotalCapacity(arena, a.items.len);
            for (a.items) |item| l.items.appendAssumeCapacity(try convert(arena, item, origin, depth + 1));
            break :blk .{ .list = l };
        },
        .object => |o| blk: {
            const d = try arena.create(Dict);
            d.* = .{};
            try d.entries.ensureTotalCapacity(arena, o.count());
            var it = o.iterator();
            while (it.next()) |e| d.entries.appendAssumeCapacity(.{ .key = .{ .string = .{ .bytes = e.key_ptr.*, .origin = origin } }, .value = try convert(arena, e.value_ptr.*, origin, depth + 1) });
            break :blk .{ .dict = d };
        },
    };
}

/// `v` as the template reads it, built in `arena` without copying its strings: a struct is a dict of its fields in
/// order with a null optional left out, as an absent JSON key; a slice is a list, a `[]const u8` a string, a tagged
/// union its payload's value (a void payload `none`), an enum its tag's name, a `Value` itself, and a type with a
/// `templateValue()` what that returns (a value parsed from text the caller keeps). Its strings are `origin`'s, a
/// `Str` keeping its own, its field names the template's. A string that is not UTF-8 is `InvalidUtf8`: every walk of
/// text past this door reads code points.
pub fn fromStruct(arena: Allocator, v: anytype, origin: Origin) StructError!Value {
    @setRuntimeSafety(true);
    const T = @TypeOf(v);
    if (T == Value) return v;
    if (T == Str) return .{ .string = try utf8(v) };
    if (T == []const u8) return .{ .string = try utf8(.{ .bytes = v, .origin = origin }) };
    if (comptime (@typeInfo(T) == .@"struct" or @typeInfo(T) == .@"union") and @hasDecl(T, "templateValue")) return v.templateValue();
    switch (@typeInfo(T)) {
        .bool => return .{ .boolean = v },
        .int, .comptime_int => return .{ .int = @intCast(v) },
        .float, .comptime_float => return .{ .float = @floatCast(v) },
        .optional => return if (v) |x| fromStruct(arena, x, origin) else .none,
        .@"enum" => return .{ .string = .{ .bytes = @tagName(v), .origin = origin } },
        .void => return .none,
        .pointer => |p| switch (p.size) {
            .one => return fromStruct(arena, v.*, origin),
            .slice => {
                const l = try arena.create(List);
                l.* = .{};
                try l.items.ensureTotalCapacity(arena, v.len);
                for (v) |item| l.items.appendAssumeCapacity(try fromStruct(arena, item, origin));
                return .{ .list = l };
            },
            else => @compileError("fromStruct: no template value for " ++ @typeName(T)),
        },
        .@"union" => switch (v) {
            inline else => |payload| return fromStruct(arena, payload, origin),
        },
        .@"struct" => |st| {
            const d = try arena.create(Dict);
            d.* = .{};
            try d.entries.ensureTotalCapacity(arena, st.field_names.len);
            inline for (st.field_names, st.field_types) |name, F| {
                const field = @field(v, name);
                const absent = @typeInfo(F) == .optional and field == null;
                if (!absent) d.entries.appendAssumeCapacity(.{ .key = .plain(name), .value = try fromStruct(arena, field, origin) });
            }
            return .{ .dict = d };
        },
        else => @compileError("fromStruct: no template value for " ++ @typeName(T)),
    }
}

pub const StructError = error{ OutOfMemory, InvalidUtf8 };

fn utf8(s: Str) error{InvalidUtf8}!Str {
    @setRuntimeSafety(true);
    return if (std.unicode.utf8ValidateSlice(s.bytes)) s else error.InvalidUtf8;
}

/// The integer of decimal `text` (an optional '-', digits, no leading zero): an int when it fits 64 bits, else
/// a bigint. Every bigint is made here, so a bigint is always outside i64 and orders by its sign against an int.
pub fn integer(text: []const u8) Value {
    @setRuntimeSafety(true);
    if (std.fmt.parseInt(i64, text, 10)) |i| return .{ .int = i } else |_| return .{ .bigint = text };
}

/// A JSON number as Python reads it.
fn number(text: []const u8) JsonError!Value {
    @setRuntimeSafety(true);
    if (std.mem.indexOfAny(u8, text, ".eE") != null) return .{ .float = std.fmt.parseFloat(f64, text) catch return error.InvalidJson };
    const digits = if (text[0] == '-') text[1..] else text;
    if (digits.len == 0) return error.InvalidJson;
    for (digits) |c| if (!std.ascii.isDigit(c)) return error.InvalidJson;
    return integer(text);
}

const testing = std.testing;

test "template value: a JSON number reads as Python reads it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try fromJson(arena.allocator(), "[1, 1.0, 1e5, 18446744073709551616, -0, {\"a\":1,\"b\":2,\"a\":3}]", .template);
    const items = v.list.items.items;
    try testing.expectEqual(@as(i64, 1), items[0].int);
    try testing.expectEqual(@as(f64, 1.0), items[1].float);
    try testing.expectEqual(@as(f64, 100000.0), items[2].float);
    try testing.expectEqualStrings("18446744073709551616", items[3].bigint);
    try testing.expectEqual(@as(i64, 0), items[4].int);
    const d = items[5].dict;
    try testing.expectEqual(@as(usize, 2), d.entries.items.len);
    try testing.expectEqualStrings("a", d.entries.items[0].key.string.bytes);
    try testing.expectEqual(@as(i64, 3), d.entries.items[0].value.int);
}

test "template value: a Zig value reads as the JSON of it reads, a null optional as an absent key" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Call = struct { name: []const u8, arguments: Value };
    const Content = union(enum) { text: []const u8, parts: []const Value, none };
    const Msg = struct { role: []const u8, content: Content, calls: ?[]const Call = null, id: ?[]const u8 = null, n: i64 = 0, f: f64 = 0.5, ok: bool = true };
    const args = try fromJson(a, "{\"x\": 1.0}", .template);
    const msgs = [_]Msg{
        .{ .role = "user", .content = .{ .text = "hi" } },
        .{ .role = "assistant", .content = .none, .calls = &.{.{ .name = "f", .arguments = args }}, .id = "c1", .n = -3 },
    };
    const v = try fromStruct(a, @as([]const Msg, &msgs), .template);
    const want = try fromJson(a,
        \\[{"role": "user", "content": "hi", "n": 0, "f": 0.5, "ok": true},
        \\ {"role": "assistant", "content": null, "calls": [{"name": "f", "arguments": {"x": 1.0}}], "id": "c1", "n": -3, "f": 0.5, "ok": true}]
    , .template);
    var m: Meter = .{};
    try testing.expect(try eql(&m, want, v, 0));
    const d0 = v.list.items.items[0].dict;
    try testing.expectEqual(@as(usize, 5), d0.entries.items.len);
    for ([_][]const u8{ "role", "content", "n", "f", "ok" }, d0.entries.items) |k, e| try testing.expectEqualStrings(k, e.key.string.bytes);
    try testing.expect(d0.entries.items[0].value.string.bytes.ptr == msgs[0].role.ptr);
    // A type that keeps its text beside the value it parsed gives the template the value.
    const Raw = struct {
        text: []const u8,
        parsed: Value,
        pub fn templateValue(self: @This()) Value {
            @setRuntimeSafety(true);
            return self.parsed;
        }
    };
    const raw = try fromStruct(a, .{ .arguments = Raw{ .text = "{\"x\": 1.0}", .parsed = args } }, .template);
    try testing.expect(try eql(&m, try fromJson(a, "{\"arguments\": {\"x\": 1.0}}", .template), raw, 0));
}

test "template value: the stable sort survives a comparator that lies or fails, and loses no item" {
    var prng = std.Random.DefaultPrng.init(42);
    const rand = prng.random();
    var items: [5000]u32 = undefined;
    var scratch: [5000]u32 = undefined;
    for (&items, 0..) |*x, i| x.* = @intCast(i);
    const Liar = struct {
        rand: std.Random,
        left: usize,
        fn less(self: *@This(), _: u32, _: u32) Failure!bool {
            @setRuntimeSafety(true);
            if (self.left == 0) return error.TemplateRuntime;
            self.left -= 1;
            return self.rand.boolean();
        }
    };
    for ([_]usize{ 0, 1, 51218, std.math.maxInt(usize) }) |budget| {
        var liar = Liar{ .rand = rand, .left = budget };
        stableSort(u32, &items, &scratch, &liar, Liar.less) catch {};
        var seen: [5000]bool = @splat(false);
        for (items) |x| seen[x] = true;
        for (seen) |s| try testing.expect(s);
    }
}
