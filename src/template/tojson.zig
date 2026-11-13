//! Python's float `repr` and `json.dumps` as transformers' `tojson` calls it (ensure_ascii, indent, separators,
//! sort_keys): an integer prints exactly at any size, a float as Python's shortest repr.
const std = @import("std");
const value = @import("value.zig");

const Value = value.Value;
const Allocator = std.mem.Allocator;

/// Python `repr(float)`: the shortest digits that read back, fixed between 1e-4 and 1e16, else exponent form.
pub fn writeFloatRepr(w: *std.ArrayList(u8), gpa: Allocator, f: f64) error{OutOfMemory}!void {
    @setRuntimeSafety(true);
    if (std.math.isNan(f)) return w.appendSlice(gpa, "nan");
    if (std.math.isInf(f)) return w.appendSlice(gpa, if (f < 0) "-inf" else "inf");
    var buf: [64]u8 = undefined;
    const sci = std.fmt.bufPrint(&buf, "{e}", .{f}) catch unreachable;
    var rest = sci;
    if (rest[0] == '-') {
        try w.append(gpa, '-');
        rest = rest[1..];
    }
    const e_at = std.mem.indexOfScalar(u8, rest, 'e').?;
    const exp = std.fmt.parseInt(i32, rest[e_at + 1 ..], 10) catch unreachable;
    var digits_buf: [32]u8 = undefined;
    var n: usize = 0;
    for (rest[0..e_at]) |c| if (c != '.') {
        digits_buf[n] = c;
        n += 1;
    };
    const digits = digits_buf[0..n];
    if (exp >= -4 and exp < 16) {
        if (exp >= 0) {
            const int_len: usize = @intCast(exp + 1);
            if (digits.len <= int_len) {
                try w.appendSlice(gpa, digits);
                try w.appendNTimes(gpa, '0', int_len - digits.len);
                try w.appendSlice(gpa, ".0");
            } else {
                try w.appendSlice(gpa, digits[0..int_len]);
                try w.append(gpa, '.');
                try w.appendSlice(gpa, digits[int_len..]);
            }
        } else {
            try w.appendSlice(gpa, "0.");
            try w.appendNTimes(gpa, '0', @intCast(-exp - 1));
            try w.appendSlice(gpa, digits);
        }
        return;
    }
    try w.append(gpa, digits[0]);
    if (digits.len > 1) {
        try w.append(gpa, '.');
        try w.appendSlice(gpa, digits[1..]);
    }
    try w.print(gpa, "e{c}{d:0>2}", .{ @as(u8, if (exp < 0) '-' else '+'), @abs(exp) });
}

/// Python's `'%.{prec}f' % f`: the exact binary value rounded half-even to `prec` digits after the point.
pub fn writeFixed(w: *std.ArrayList(u8), gpa: Allocator, f: f64, prec: usize) error{OutOfMemory}!void {
    @setRuntimeSafety(true);
    if (std.math.isNan(f)) return w.appendSlice(gpa, "nan");
    if (std.math.isInf(f)) return w.appendSlice(gpa, if (f < 0) "-inf" else "inf");
    const bits: u64 = @bitCast(f);
    const biased: u64 = (bits >> 52) & 0x7FF;
    const frac: u64 = bits & ((@as(u64, 1) << 52) - 1);
    const mant: u64 = if (biased == 0) frac else frac | (@as(u64, 1) << 52);
    const exp: i64 = if (biased == 0) -1074 else @as(i64, @intCast(biased)) - 1075;
    // |f| = n / 10^k exactly: n = mant * 2^exp when exp >= 0, else mant * 5^-exp with k = -exp.
    var n = try std.math.big.int.Managed.init(gpa);
    try n.set(mant);
    var k: usize = 0;
    if (exp >= 0) {
        try n.shiftLeft(&n, @intCast(exp));
    } else {
        k = @intCast(-exp);
        var five = try std.math.big.int.Managed.init(gpa);
        try five.set(5);
        try five.pow(&five, @intCast(k));
        try n.mul(&n, &five);
    }
    const raw = n.toString(gpa, 10, .lower) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidBase => unreachable,
    };
    // At least one digit before the point.
    const digits = if (raw.len > k) raw else try std.mem.concat(gpa, u8, &.{ try zeros(gpa, k + 1 - raw.len), raw });
    var kept = std.ArrayList(u8).empty;
    const point = digits.len - k;
    if (prec >= k) {
        try kept.appendSlice(gpa, digits);
        try kept.appendNTimes(gpa, '0', prec - k);
    } else {
        const keep = point + prec;
        try kept.appendSlice(gpa, digits[0..keep]);
        const dropped = digits[keep..];
        const rest_zero = for (dropped[1..]) |c| {
            if (c != '0') break false;
        } else true;
        const last_odd = keep > 0 and (digits[keep - 1] - '0') % 2 == 1;
        const up = dropped[0] > '5' or (dropped[0] == '5' and (!rest_zero or last_odd));
        if (up) {
            var i = kept.items.len;
            while (i > 0) : (i -= 1) {
                if (kept.items[i - 1] != '9') {
                    kept.items[i - 1] += 1;
                    break;
                }
                kept.items[i - 1] = '0';
            } else try kept.insert(gpa, 0, '1');
        }
    }
    const int_len = kept.items.len - prec;
    if (std.math.signbit(f)) try w.append(gpa, '-');
    var start: usize = 0;
    while (start + 1 < int_len and kept.items[start] == '0') start += 1;
    try w.appendSlice(gpa, kept.items[start..int_len]);
    if (prec > 0) {
        try w.append(gpa, '.');
        try w.appendSlice(gpa, kept.items[int_len..]);
    }
}

fn zeros(gpa: Allocator, n: usize) error{OutOfMemory}![]u8 {
    @setRuntimeSafety(true);
    const z = try gpa.alloc(u8, n);
    @memset(z, '0');
    return z;
}

pub const Options = struct {
    ensure_ascii: bool = false,
    /// Null: one line; else the text one level indents by.
    indent: ?[]const u8 = null,
    item_separator: ?[]const u8 = null,
    key_separator: []const u8 = ": ",
    sort_keys: bool = false,
    /// Past this many bytes of output the dump stops with `TemplateRuntime`.
    max_len: usize = std.math.maxInt(usize),
};

/// `json.dumps(v, **options)`; a value JSON cannot hold is a runtime error, as Python's TypeError.
pub fn dumps(m: *value.Meter, w: *std.ArrayList(u8), gpa: Allocator, v: Value, opts: Options, depth: usize) value.Failure!void {
    @setRuntimeSafety(true);
    value.noteStack();
    if (depth > value.max_depth) return m.fail(.depth);
    if (w.items.len > opts.max_len) return m.fail(.text);
    value.spend(1);
    try m.tick(1);
    const item_sep = opts.item_separator orelse if (opts.indent != null) "," else ", ";
    switch (v) {
        .none => try w.appendSlice(gpa, "null"),
        .boolean => |b| try w.appendSlice(gpa, if (b) "true" else "false"),
        .int => |i| try w.print(gpa, "{d}", .{i}),
        .bigint => |t| try w.appendSlice(gpa, t),
        .float => |f| {
            if (std.math.isNan(f)) return w.appendSlice(gpa, "NaN");
            if (std.math.isInf(f)) return w.appendSlice(gpa, if (f < 0) "-Infinity" else "Infinity");
            try writeFloatRepr(w, gpa, f);
        },
        .string, .markup => |s| {
            try m.tick(s.bytes.len / 64);
            try writeString(w, gpa, s.bytes, opts.ensure_ascii);
        },
        .list => |l| {
            if (l.kind != .list and l.kind != .tuple) return m.fail(.{ .json = @tagName(l.kind) });
            if (l.items.items.len == 0) return w.appendSlice(gpa, "[]");
            try w.append(gpa, '[');
            for (l.items.items, 0..) |item, i| {
                if (i > 0) try w.appendSlice(gpa, item_sep);
                try newlineIndent(m, w, gpa, opts, depth + 1);
                try dumps(m, w, gpa, item, opts, depth + 1);
            }
            try newlineIndent(m, w, gpa, opts, depth);
            try w.append(gpa, ']');
        },
        .dict => |d| {
            if (d.entries.items.len == 0) return w.appendSlice(gpa, "{}");
            var entries = d.entries.items;
            if (opts.sort_keys) {
                entries = try gpa.dupe(value.Entry, entries);
                var ctx = value.SortContext{ .meter = m };
                try value.stableSort(value.Entry, entries, try gpa.alloc(value.Entry, entries.len), &ctx, struct {
                    fn lt(c: *value.SortContext, a: value.Entry, b: value.Entry) value.Failure!bool {
                        @setRuntimeSafety(true);
                        return c.less(a.key, b.key);
                    }
                }.lt);
            }
            try w.append(gpa, '{');
            for (entries, 0..) |e, i| {
                if (i > 0) try w.appendSlice(gpa, item_sep);
                try newlineIndent(m, w, gpa, opts, depth + 1);
                try writeKey(m, w, gpa, e.key, opts.ensure_ascii);
                try w.appendSlice(gpa, opts.key_separator);
                try dumps(m, w, gpa, e.value, opts, depth + 1);
            }
            try newlineIndent(m, w, gpa, opts, depth);
            try w.append(gpa, '}');
        },
        .undefined, .namespace, .macro, .function, .loop => return m.fail(.{ .json = value.typeName(v) }),
    }
}

fn newlineIndent(m: *value.Meter, w: *std.ArrayList(u8), gpa: Allocator, opts: Options, depth: usize) value.Failure!void {
    @setRuntimeSafety(true);
    const unit = opts.indent orelse return;
    // Checked before it is written: a wide indent at depth is many times the limit in one step.
    const add = std.math.mul(usize, unit.len, depth) catch return m.fail(.text);
    if (w.items.len +| 1 +| add > opts.max_len) return m.fail(.text);
    value.spend(add / 64 + 1);
    try m.tick(add / 64 + 1);
    try w.append(gpa, '\n');
    for (0..depth) |_| try w.appendSlice(gpa, unit);
}

/// A key as json.dumps writes it: strings as they are, numbers, booleans and None as their JSON text, quoted.
fn writeKey(m: *value.Meter, w: *std.ArrayList(u8), gpa: Allocator, k: Value, ensure_ascii: bool) value.Failure!void {
    @setRuntimeSafety(true);
    switch (k) {
        .string, .markup => |s| try writeString(w, gpa, s.bytes, ensure_ascii),
        .none, .boolean, .int, .bigint, .float => {
            var text = std.ArrayList(u8).empty;
            try dumps(m, &text, gpa, k, .{}, 0);
            try writeString(w, gpa, text.items, ensure_ascii);
        },
        else => return m.fail(.{ .json = value.typeName(k) }),
    }
}

fn writeString(w: *std.ArrayList(u8), gpa: Allocator, s: []const u8, ensure_ascii: bool) error{OutOfMemory}!void {
    @setRuntimeSafety(true);
    try w.append(gpa, '"');
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        if (c >= 0x80) {
            const len = std.unicode.utf8ByteSequenceLength(c) catch 1;
            const end = @min(s.len, i + len);
            if (!ensure_ascii) {
                try w.appendSlice(gpa, s[i..end]);
            } else {
                const cp = std.unicode.utf8Decode(s[i..end]) catch 0xFFFD;
                if (cp >= 0x10000) {
                    const v = cp - 0x10000;
                    try w.print(gpa, "\\u{x:0>4}\\u{x:0>4}", .{ 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF) });
                } else try w.print(gpa, "\\u{x:0>4}", .{cp});
            }
            i = end;
            continue;
        }
        i += 1;
        switch (c) {
            '"' => try w.appendSlice(gpa, "\\\""),
            '\\' => try w.appendSlice(gpa, "\\\\"),
            '\n' => try w.appendSlice(gpa, "\\n"),
            '\r' => try w.appendSlice(gpa, "\\r"),
            '\t' => try w.appendSlice(gpa, "\\t"),
            0x7F => if (ensure_ascii) try w.appendSlice(gpa, "\\u007f") else try w.append(gpa, c),
            0x08 => try w.appendSlice(gpa, "\\b"),
            0x0C => try w.appendSlice(gpa, "\\f"),
            0...0x07, 0x0B, 0x0E...0x1F => try w.print(gpa, "\\u{x:0>4}", .{c}),
            else => try w.append(gpa, c),
        }
    }
    try w.append(gpa, '"');
}

const testing = std.testing;

test "template tojson: a float prints as Python's repr" {
    const cases = [_]struct { f: f64, want: []const u8 }{
        .{ .f = 1.0, .want = "1.0" },
        .{ .f = -0.0, .want = "-0.0" },
        .{ .f = 0.1, .want = "0.1" },
        .{ .f = 1e-05, .want = "1e-05" },
        .{ .f = 0.0001, .want = "0.0001" },
        .{ .f = 100000.0, .want = "100000.0" },
        .{ .f = 1e16, .want = "1e+16" },
        .{ .f = 1e15, .want = "1000000000000000.0" },
        .{ .f = 3.141592653589793, .want = "3.141592653589793" },
        .{ .f = 5e-324, .want = "5e-324" },
        .{ .f = 1.7976931348623157e308, .want = "1.7976931348623157e+308" },
        .{ .f = 1.2345678901234568e+29, .want = "1.2345678901234568e+29" },
        .{ .f = 2.5e-8, .want = "2.5e-08" },
    };
    for (cases) |c| {
        var out = std.ArrayList(u8).empty;
        defer out.deinit(testing.allocator);
        try writeFloatRepr(&out, testing.allocator, c.f);
        try testing.expectEqualStrings(c.want, out.items);
    }
}
