//! Streaming JSON grammar state machine constrained by a `json_schema.Schema`.
//!
//! Architecture
//! ============
//! The grammar holds a single Config = a stack of Frames. Each Frame represents
//! an in-progress JSON value being parsed against a schema node. Bytes are fed in
//! one-at-a-time via `acceptByte`. The `allowedBytes` query enumerates which next
//! bytes would not be rejected — this is what the token-mask layer uses to filter
//! the LLM's vocabulary.
//!
//! Snapshots support speculative lookahead: the token-mask builder simulates each
//! candidate token's bytes through the grammar from a snapshot, accepting/rejecting
//! the token without permanently advancing real state.
//!
//! Schema construct support
//! ========================
//! Enforced strictly, with no limit on a schema's size:
//!   * JSON syntax (RFC 8259), including \uXXXX escapes and control-char rejection
//!   * `type`: object/array/string/number/integer/boolean/null
//!   * `properties` + `required` + `additionalProperties` (false, true or a schema)
//!   * `items` (homogeneous arrays)
//!   * `enum` / `const`: a string member by its decoded value, any other by its canonical JSON text
//!   * `minItems` / `maxItems`
//!   * `minLength` / `maxLength` on strings, in characters (code points), as JSON Schema counts them
//!   * `$ref` to a local pointer (resolved in json_schema.zig; others relax to any)
//! An object key is matched by its decoded value too, so `"\u0061"` names property `a`.
//! Best-effort (relaxed) — accept any valid JSON value, rely on prompt for shape:
//!   * `anyOf` / `oneOf` — treated as "any JSON value" inside the union
//!   * `pattern`, `minimum`, `maximum`, `exclusiveMinimum/Maximum` — soft prompt-only
//!
//! Why anyOf is relaxed
//! --------------------
//! Tagged-union anyOf (e.g., `{type:"text",content}` vs `{type:"cruise_cards",query}`)
//! requires multi-config parallel parsing or eager discriminator resolution. Either
//! adds significant complexity and is the most common cause of grammar bugs. We
//! accept the relaxation: the model still sees the schema in its system prompt and
//! produces structurally-valid JSON for the union; the grammar guarantees the
//! surrounding object/array is well-formed and that all *non-anyOf* parts of the
//! response strictly conform.

const std = @import("std");
const schema_mod = @import("json_schema.zig");
const utf8_seq = @import("utf8_seq.zig");

pub const Schema = schema_mod.Schema;
pub const Node = schema_mod.Node;
pub const Kind = schema_mod.Kind;

// ── ByteMask ──────────────────────────────────────────────────────────────────

pub const ByteMask = struct {
    bits: [4]u64 = @splat(0),

    pub fn empty() ByteMask {
        return .{};
    }

    pub fn full() ByteMask {
        return .{ .bits = .{ ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0), ~@as(u64, 0) } };
    }

    pub fn add(self: *ByteMask, b: u8) void {
        self.bits[b >> 6] |= (@as(u64, 1) << @intCast(b & 63));
    }

    pub fn addRange(self: *ByteMask, lo: u8, hi: u8) void {
        var b: u16 = lo;
        while (b <= hi) : (b += 1) self.add(@intCast(b));
    }

    pub fn contains(self: *const ByteMask, b: u8) bool {
        return (self.bits[b >> 6] >> @intCast(b & 63)) & 1 == 1;
    }

    pub fn isEmpty(self: *const ByteMask) bool {
        return self.bits[0] == 0 and self.bits[1] == 0 and self.bits[2] == 0 and self.bits[3] == 0;
    }
};

fn isWs(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r';
}

// ── Frame state types ─────────────────────────────────────────────────────────

const StrPhase = enum(u2) { content, after_backslash, unicode_hex };

/// Lexes a JSON string body one raw byte at a time. What a byte means (a raw byte, or a UTF-16 code unit an escape
/// spells) is put to a sink, which decides whether the value may go on that way (`stepStr`).
const StrLex = struct {
    phase: StrPhase = .content,
    hex_left: u3 = 0,
    /// The hex digits of the `\u` escape read so far.
    unit: u16 = 0,
};

/// A string value, counted in characters (code points) for `minLength` / `maxLength`.
const StringState = struct {
    lex: StrLex = .{},
    chars: u32 = 0,
    /// The last thing typed was a high-surrogate escape: a low one now completes that character.
    high: bool = false,
};

/// Which names of a sorted list a string being typed can still decode to: `[lo, hi)` share the `pos` decoded bytes
/// so far, and `high` holds a high-surrogate escape waiting for its low half (0 = none).
const Match = struct {
    lo: u32,
    hi: u32,
    pos: u32 = 0,
    high: u16 = 0,
};

const NumberPhase = enum {
    after_minus,
    leading_zero,
    int_digits,
    after_dot,
    frac_digits,
    after_exp,
    after_exp_sign,
    exp_digits,
};

const NumberState = struct {
    phase: NumberPhase,
};

const ObjectPhase = enum {
    expect_first_key_or_close,
    expect_key, // after `,`
    in_key,
    after_key,
    expect_value,
    after_value,
};

const ObjectState = struct {
    phase: ObjectPhase,
    /// Where this object's words start in `Grammar.seen`: one bit per entry of `obj_properties`.
    seen_at: u32,
    /// Properties seen, and how many of those are required.
    seen_count: u32 = 0,
    required_seen: u32 = 0,
    key: StrLex = .{},
    key_match: Match = .{ .lo = 0, .hi = 0 },
    /// Resolved when the key closes: index of the property whose value comes next,
    /// or null for an additional-property (free-form key).
    current_prop: ?u32 = null,
};

const ArrayPhase = enum {
    expect_first_value_or_close,
    expect_value,
    after_value,
};

const ArrayState = struct {
    phase: ArrayPhase,
    count: u32 = 0,
};

/// A non-string enum member being typed: `enum_literals[lo..hi]` start with the `pos` bytes so far.
const EnumState = struct {
    lo: u32,
    hi: u32,
    pos: u32 = 0,
};

/// A string enum member being typed, matched by decoded value against `enum_strings`.
const EnumStringState = struct {
    lex: StrLex = .{},
    match: Match,
};

const KeywordState = struct {
    target: []const u8,
    pos: u8,
};

const FrameSub = union(enum) {
    expect_value,
    in_string: StringState,
    in_number: NumberState,
    in_object: ObjectState,
    in_array: ArrayState,
    in_enum: EnumState,
    in_enum_string: EnumStringState,
    in_keyword: KeywordState,
    accepted,
};

const Frame = struct {
    schema: *const Node,
    sub: FrameSub,
};

// ── Grammar ───────────────────────────────────────────────────────────────────

const MAX_STACK = 64;
/// Consecutive free-whitespace bytes the grammar admits between tokens.
pub const MAX_FREE_WS = 16;

pub const Grammar = struct {
    gpa: std.mem.Allocator,
    schema: *const Schema,
    stack: std.ArrayListUnmanaged(Frame),
    /// The seen-property bits of every open object, outermost first: an object's words are pushed when it opens
    /// and dropped when it closes, so a schema's property count is no limit.
    seen: std.ArrayListUnmanaged(u64) = .empty,
    /// Set true once all configurations have rejected and recovery is impossible.
    dead: bool = false,
    ws_run: u16 = 0,

    pub fn init(gpa: std.mem.Allocator, schema: *const Schema) std.mem.Allocator.Error!Grammar {
        var g: Grammar = .{
            .gpa = gpa,
            .schema = schema,
            .stack = .empty,
        };
        // anyOf at the root is relaxed to "any JSON" (see header doc).
        const root = if (schema.root.kind == .any_of) anyNode() else schema.root;
        try g.stack.append(gpa, .{ .schema = root, .sub = .expect_value });
        return g;
    }

    pub fn deinit(self: *Grammar) void {
        self.stack.deinit(self.gpa);
        self.seen.deinit(self.gpa);
    }

    pub fn isDead(self: *const Grammar) bool {
        return self.dead;
    }

    /// True once the root value has been fully parsed. A root number has no
    /// closing byte, so it is complete as soon as it could end.
    pub fn isComplete(self: *const Grammar) bool {
        if (self.stack.items.len != 1) return false;
        const root = self.stack.items[0];
        return switch (root.sub) {
            .accepted => true,
            .in_number => |n| switch (n.phase) {
                .leading_zero, .int_digits, .frac_digits, .exp_digits => true,
                .after_minus, .after_dot, .after_exp, .after_exp_sign => false,
            },
            .in_enum => |e| enumLiteralWhole(root.schema, e),
            else => false,
        };
    }

    /// Characters still admissible in the string body the next byte lands in
    /// (content phase, no escape in flight): `maxLength` minus what was typed,
    /// or `maxInt` when unbounded. Null in every other state.
    pub fn stringBodyRoom(self: *const Grammar) ?u32 {
        if (self.dead) return null;
        const frame = self.topConst();
        const s = switch (frame.sub) {
            .in_string => |st| st,
            else => return null,
        };
        if (s.lex.phase != .content) return null;
        const max = frame.schema.str_max_len orelse return std.math.maxInt(u32);
        return max -| s.chars;
    }

    /// Bitset of bytes that would be accepted by the grammar in its current state.
    /// Implemented by speculative trial: snapshot, try each byte, restore.
    pub fn allowedBytes(self: *Grammar) std.mem.Allocator.Error!ByteMask {
        var mask = ByteMask.empty();
        if (self.dead) return mask;

        const snap = try self.snapshot();
        defer self.discardSnapshot(snap);

        var b: u16 = 0;
        while (b < 256) : (b += 1) {
            const byte: u8 = @intCast(b);
            try self.restoreFrom(snap);
            if (try self.tryAdvance(byte)) {
                mask.add(byte);
            }
        }
        try self.restoreFrom(snap);
        return mask;
    }

    /// Try to consume one byte. Returns true if accepted (state advanced),
    /// false if rejected (state unchanged).
    pub fn acceptByte(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        if (self.dead) return false;
        const snap = try self.snapshot();
        defer self.discardSnapshot(snap);
        const ok = try self.tryAdvance(byte);
        if (!ok) try self.restoreFrom(snap);
        return ok;
    }

    /// Speculative variant of `acceptByte` that does NOT take an internal snapshot.
    /// On rejection, the grammar may be left in an indeterminate state — the caller
    /// MUST hold an outer snapshot and restore it. Used by the token-mask builder
    /// to amortise away ~100k redundant per-byte allocations.
    pub fn acceptByteFast(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        if (self.dead) return false;
        return try self.tryAdvance(byte);
    }

    /// Internal: advance state by one byte. May replay (e.g., when a number ends
    /// before its actual terminator byte). Returns true on success, false on reject.
    fn tryAdvance(self: *Grammar, byte: u8) std.mem.Allocator.Error!bool {
        var b: u8 = byte;
        // Bounded replay loop: consume at most a few replays per input byte.
        var iter: u32 = 0;
        while (iter < 8) : (iter += 1) {
            const r = try advanceOne(self, b);
            switch (r) {
                .consumed, .consumed_ws => return true,
                .replay => |nb| b = nb,
                .reject => return false,
            }
        }
        return false;
    }

    // ── Snapshot / restore ────────────────────────────────────────────────────
    // A snapshot is a deep copy of the stack. We use it for both speculative
    // byte trials (in allowedBytes) and for token-mask probes by external callers.

    pub const Snapshot = struct {
        stack: []Frame, // owned
        seen: []u64, // owned
        dead: bool,
        ws_run: u16,
    };

    pub fn snapshot(self: *const Grammar) std.mem.Allocator.Error!Snapshot {
        const stack = try self.gpa.dupe(Frame, self.stack.items);
        errdefer self.gpa.free(stack);
        const seen = try self.gpa.dupe(u64, self.seen.items);
        return .{ .stack = stack, .seen = seen, .dead = self.dead, .ws_run = self.ws_run };
    }

    pub fn restoreFrom(self: *Grammar, snap: Snapshot) std.mem.Allocator.Error!void {
        self.stack.clearRetainingCapacity();
        try self.stack.appendSlice(self.gpa, snap.stack);
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.gpa, snap.seen);
        self.dead = snap.dead;
        self.ws_run = snap.ws_run;
    }

    pub fn discardSnapshot(self: *const Grammar, snap: Snapshot) void {
        self.gpa.free(snap.stack);
        self.gpa.free(snap.seen);
    }

    fn top(self: *Grammar) *Frame {
        return &self.stack.items[self.stack.items.len - 1];
    }

    fn topConst(self: *const Grammar) *const Frame {
        return &self.stack.items[self.stack.items.len - 1];
    }
};

// ── Advance result ────────────────────────────────────────────────────────────

const StepResult = union(enum) {
    consumed,
    consumed_ws,
    replay: u8,
    reject,
};

fn advanceOne(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const r = try advanceOneInner(g, byte);
    if (r == .consumed) g.ws_run = 0;
    return r;
}

/// Free whitespace is capped per run: a masked model whose real argmax is
/// off-schema otherwise idles on whitespace until the loop guard cuts it.
fn freeWhitespace(g: *Grammar) StepResult {
    if (g.ws_run >= MAX_FREE_WS) return .reject;
    g.ws_run += 1;
    return .consumed_ws;
}

fn advanceOneInner(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    // Nothing outside the root value: whitespace there is a greedy near-tie
    // the model pads with (`\r   \r   \r  {`) or idles on, and it carries no
    // formatting the model prefers. Inside the root its own layout stays.
    if (g.stack.items.len == 0) return .reject;
    if (g.stack.items.len == 1 and g.topConst().sub == .accepted) return .reject;
    const before_root = g.stack.items.len == 1 and g.topConst().sub == .expect_value;

    // Free whitespace handling for between-token positions.
    if (isWs(byte) and !before_root and canAcceptFreeWhitespace(g.topConst())) {
        return freeWhitespace(g);
    }

    if (g.stack.items.len > MAX_STACK) {
        g.dead = true;
        return .reject;
    }

    const frame = g.top();
    return switch (frame.sub) {
        .expect_value => try stepExpectValue(g, byte),
        .in_string => try stepString(g, byte),
        .in_number => try stepNumber(g, byte),
        .in_object => try stepObject(g, byte),
        .in_array => try stepArray(g, byte),
        .in_enum => try stepEnum(g, byte),
        .in_enum_string => try stepEnumString(g, byte),
        .in_keyword => try stepKeyword(g, byte),
        .accepted => .reject,
    };
}

fn canAcceptFreeWhitespace(frame: *const Frame) bool {
    return switch (frame.sub) {
        .expect_value, .accepted => true,
        .in_string, .in_keyword, .in_enum, .in_enum_string => false,
        .in_number => false,
        .in_object => |o| switch (o.phase) {
            .expect_first_key_or_close, .expect_key, .after_key, .expect_value, .after_value => true,
            .in_key => false,
        },
        .in_array => |a| switch (a.phase) {
            .expect_first_value_or_close, .expect_value, .after_value => true,
        },
    };
}

// ── expect_value dispatch ─────────────────────────────────────────────────────

fn stepExpectValue(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const schema = g.top().schema;
    return switch (schema.kind) {
        .null_ => try startKeyword(g, byte, "null"),
        .boolean => switch (byte) {
            't' => try startKeyword(g, byte, "true"),
            'f' => try startKeyword(g, byte, "false"),
            else => .reject,
        },
        .integer, .number => startNumber(g, byte),
        .string => startString(g, byte),
        .array => startArray(g, byte),
        .object => try startObject(g, byte),
        .enum_ => try startEnum(g, byte),
        // The parser keeps `false` only where it forbids a key, so no value is ever asked of it.
        .never => .reject,
        .any, .any_of => blk: {
            // Relaxed: accept any valid JSON value here. A parsed `any` node allows
            // no extra keys, so swap in the free-form one and re-dispatch.
            g.top().schema = anyNode();
            break :blk try startAnyValue(g, byte);
        },
    };
}

fn startAnyValue(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    return switch (byte) {
        '"' => startString(g, byte),
        '{' => try startObject(g, byte),
        '[' => startArray(g, byte),
        't' => try startKeyword(g, byte, "true"),
        'f' => try startKeyword(g, byte, "false"),
        'n' => try startKeyword(g, byte, "null"),
        '-', '0'...'9' => startNumber(g, byte),
        else => .reject,
    };
}

// ── keyword (true/false/null) ─────────────────────────────────────────────────

fn startKeyword(g: *Grammar, byte: u8, target: []const u8) std.mem.Allocator.Error!StepResult {
    if (target.len == 0 or target[0] != byte) return .reject;
    g.top().sub = .{ .in_keyword = .{ .target = target, .pos = 1 } };
    if (target.len == 1) try popAccepted(g);
    return .consumed;
}

fn stepKeyword(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var k = g.top().sub.in_keyword;
    if (k.pos >= k.target.len) return .reject;
    if (k.target[k.pos] != byte) return .reject;
    k.pos += 1;
    g.top().sub = .{ .in_keyword = k };
    if (k.pos == k.target.len) try popAccepted(g);
    return .consumed;
}

// ── string body: lexing, length, and matching by decoded value ────────────────

const StrStep = enum { reject, consumed, close };

fn isHighSurrogate(u: u32) bool {
    return u >= 0xD800 and u <= 0xDBFF;
}

fn isLowSurrogate(u: u32) bool {
    return u >= 0xDC00 and u <= 0xDFFF;
}

/// A UTF-8 continuation byte: it extends the character before it.
fn isContinuation(b: u8) bool {
    return b & 0xC0 == 0x80;
}

fn hexDigit(b: u8) ?u16 {
    return switch (b) {
        '0'...'9' => b - '0',
        'a'...'f' => b - 'a' + 10,
        'A'...'F' => b - 'A' + 10,
        else => null,
    };
}

/// Whether some code unit whose top `known` hex digits are `prefix` lies in `[lo, hi]`.
fn unitPrefixMeets(known: u3, prefix: u16, lo: u32, hi: u32) bool {
    const shift: u5 = @intCast(4 * (4 - @as(u32, known)));
    const first = @as(u32, prefix) << shift;
    const last = first + ((@as(u32, 1) << shift) - 1);
    return first <= hi and last >= lo;
}

/// One byte of a string body. The sink answers exactly (some continuation exists), so the grammar never enters a
/// state with no legal byte: `admitsUnit(k, p)` is whether some code unit whose first `k` hex digits are `p` may
/// come next (`k` 0: any unit at all).
fn stepStr(lex: *StrLex, byte: u8, sink: anytype) StrStep {
    switch (lex.phase) {
        .content => {
            if (byte == '"') return if (sink.admitsClose()) .close else .reject;
            if (byte == '\\') {
                if (!sink.admitsUnit(0, 0)) return .reject;
                lex.phase = .after_backslash;
                return .consumed;
            }
            if (byte < 0x20 or !sink.admitsByte(byte)) return .reject;
            sink.takeByte(byte);
            return .consumed;
        },
        .after_backslash => {
            if (byte == 'u') {
                lex.* = .{ .phase = .unicode_hex, .hex_left = 4 };
                return .consumed;
            }
            const unit: u16 = switch (byte) {
                '"', '\\', '/' => byte,
                'b' => 0x08,
                'f' => 0x0C,
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                else => return .reject,
            };
            if (!sink.admitsUnit(4, unit)) return .reject;
            sink.takeUnit(unit);
            lex.phase = .content;
            return .consumed;
        },
        .unicode_hex => {
            const unit = (lex.unit << 4) | (hexDigit(byte) orelse return .reject);
            if (!sink.admitsUnit(5 - lex.hex_left, unit)) return .reject;
            lex.unit = unit;
            lex.hex_left -= 1;
            if (lex.hex_left == 0) {
                sink.takeUnit(unit);
                lex.phase = .content;
            }
            return .consumed;
        },
    }
}

/// A string value's characters against its `minLength` / `maxLength`.
const LengthSink = struct {
    s: *StringState,
    node: *const Node,

    fn full(self: *const LengthSink) bool {
        const max = self.node.str_max_len orelse return false;
        return self.s.chars >= max;
    }

    fn admitsClose(self: *const LengthSink) bool {
        return self.s.chars >= (self.node.str_min_len orelse 0);
    }

    fn admitsByte(self: *const LengthSink, b: u8) bool {
        return isContinuation(b) or !self.full();
    }

    fn takeByte(self: *LengthSink, b: u8) void {
        if (!isContinuation(b)) self.s.chars +|= 1;
        self.s.high = false;
    }

    fn admitsUnit(self: *const LengthSink, known: u3, prefix: u16) bool {
        if (!self.full()) return true;
        return self.s.high and unitPrefixMeets(known, prefix, 0xDC00, 0xDFFF);
    }

    fn takeUnit(self: *LengthSink, unit: u16) void {
        if (self.s.high and isLowSurrogate(unit)) {
            self.s.high = false;
            return;
        }
        self.s.chars +|= 1;
        self.s.high = isHighSurrogate(unit);
    }
};

/// The next UTF-16 code unit `name` holds at byte `pos`, after the pending high surrogate `high` (0 = none); null
/// when the name ends there, or `pos` is inside a character, or `high` is not this character's high half.
fn nextUnit(name: []const u8, pos: u32, high: u16) ?u16 {
    if (pos >= name.len) return null;
    const n = std.unicode.utf8ByteSequenceLength(name[pos]) catch return null;
    if (pos + n > name.len) return null;
    const cp: u32 = utf8_seq.decode(name[pos..][0..n]) orelse return null;
    if (cp < 0x10000) return if (high == 0) @intCast(cp) else null;
    const hi_half: u16 = @intCast(0xD800 + ((cp - 0x10000) >> 10));
    if (high == 0) return hi_half;
    return if (hi_half == high) @intCast(0xDC00 + ((cp - 0x10000) & 0x3FF)) else null;
}

/// The names of a sorted list a string is matched against, and which of them it may still become.
const PropertyNames = struct {
    node: *const Node,
    seen: []const u64,

    fn name(self: PropertyNames, i: u32) []const u8 {
        return self.node.obj_properties[i].name;
    }

    /// A key may name a property once, and never one whose schema is `false`.
    fn eligible(self: PropertyNames, i: u32) bool {
        return self.node.obj_properties[i].schema.kind != .never and !isSeen(self.seen, i);
    }
};

const StringNames = struct {
    names: []const []const u8,

    fn name(self: StringNames, i: u32) []const u8 {
        return self.names[i];
    }

    fn eligible(_: StringNames, _: u32) bool {
        return true;
    }
};

/// The first index of `[lo, hi)` (names sharing `pos` bytes, so a name that ends at `pos` sorts first) whose
/// byte at `pos` is at least `b`.
fn lowerBound(names: anytype, lo: u32, hi: u32, pos: u32, b: i32) u32 {
    var l = lo;
    var h = hi;
    while (l < h) {
        const mid = l + (h - l) / 2;
        const n = names.name(mid);
        const at: i32 = if (n.len > pos) n[pos] else -1;
        if (at < b) l = mid + 1 else h = mid;
    }
    return l;
}

/// `[lo, hi)` narrowed to the names whose byte at `pos` is `b`.
fn narrow(names: anytype, lo: u32, hi: u32, pos: u32, b: u8) struct { lo: u32, hi: u32 } {
    const from = lowerBound(names, lo, hi, pos, b);
    return .{ .lo = from, .hi = lowerBound(names, from, hi, pos, @as(i32, b) + 1) };
}

/// A string matched by decoded value against sorted `names` (an object key, a string enum member). `free`: any
/// string may be typed (a key under `additionalProperties`), and the match only says which property it names.
fn MatchSink(comptime Names: type) type {
    return struct {
        m: *Match,
        names: Names,
        free: bool,

        const Self = @This();

        /// The name the string so far decodes to exactly, if it is one.
        fn exact(self: *const Self) ?u32 {
            const m = self.m;
            if (m.high != 0 or m.lo == m.hi or self.names.name(m.lo).len != m.pos) return null;
            return m.lo;
        }

        /// The eligible name the closed string named; null for a free string.
        fn matched(self: *const Self) ?u32 {
            const i = self.exact() orelse return null;
            return if (self.names.eligible(i)) i else null;
        }

        fn anyEligible(self: *const Self, lo: u32, hi: u32) bool {
            var i = lo;
            while (i < hi) : (i += 1) if (self.names.eligible(i)) return true;
            return false;
        }

        fn admitsClose(self: *const Self) bool {
            // A string that spells an ineligible name exactly is refused even when free: a key named twice, or
            // named although its schema is `false`.
            if (self.exact()) |i| return self.names.eligible(i);
            return self.free;
        }

        fn admitsByte(self: *const Self, b: u8) bool {
            if (self.free) return true;
            if (self.m.high != 0) return false;
            const r = narrow(self.names, self.m.lo, self.m.hi, self.m.pos, b);
            return self.anyEligible(r.lo, r.hi);
        }

        fn takeByte(self: *Self, b: u8) void {
            if (self.m.high != 0) return self.diverge(); // a lone high surrogate names nothing
            const r = narrow(self.names, self.m.lo, self.m.hi, self.m.pos, b);
            self.m.lo = r.lo;
            self.m.hi = r.hi;
            self.m.pos += 1;
        }

        fn admitsUnit(self: *const Self, known: u3, prefix: u16) bool {
            if (self.free) return true;
            var i = self.m.lo;
            while (i < self.m.hi) : (i += 1) {
                if (!self.names.eligible(i)) continue;
                const u = nextUnit(self.names.name(i), self.m.pos, self.m.high) orelse continue;
                if (unitPrefixMeets(known, prefix, u, u)) return true;
            }
            return false;
        }

        fn takeUnit(self: *Self, unit: u16) void {
            if (self.m.high != 0) {
                if (!isLowSurrogate(unit)) return self.diverge();
                const cp: u21 = @intCast(0x10000 + ((@as(u32, self.m.high) - 0xD800) << 10) + (unit - 0xDC00));
                self.m.high = 0;
                return self.takeCodepoint(cp);
            }
            if (isHighSurrogate(unit)) {
                self.m.high = unit;
                return;
            }
            if (isLowSurrogate(unit)) return self.diverge();
            self.takeCodepoint(unit);
        }

        fn takeCodepoint(self: *Self, cp: u21) void {
            var buf: [4]u8 = undefined;
            const n = std.unicode.utf8Encode(cp, &buf) catch unreachable; // never a surrogate here
            for (buf[0..n]) |b| self.takeByte(b);
        }

        fn diverge(self: *Self) void {
            self.m.lo = self.m.hi;
            self.m.high = 0;
        }
    };
}

fn startString(g: *Grammar, byte: u8) StepResult {
    if (byte != '"') return .reject;
    g.top().sub = .{ .in_string = .{} };
    return .consumed;
}

fn stepString(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const frame = g.top();
    var s = frame.sub.in_string;
    var sink: LengthSink = .{ .s = &s, .node = frame.schema };
    switch (stepStr(&s.lex, byte, &sink)) {
        .reject => return .reject,
        .close => try popAccepted(g),
        .consumed => frame.sub = .{ .in_string = s },
    }
    return .consumed;
}

// ── number ────────────────────────────────────────────────────────────────────

fn startNumber(g: *Grammar, byte: u8) StepResult {
    var phase: NumberPhase = undefined;
    switch (byte) {
        '-' => phase = .after_minus,
        '0' => phase = .leading_zero,
        '1'...'9' => phase = .int_digits,
        else => return .reject,
    }
    g.top().sub = .{ .in_number = .{ .phase = phase } };
    return .consumed;
}

fn stepNumber(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var n = g.top().sub.in_number;
    const integer_only = g.top().schema.kind == .integer;

    switch (n.phase) {
        .after_minus => switch (byte) {
            '0' => n.phase = .leading_zero,
            '1'...'9' => n.phase = .int_digits,
            else => return .reject,
        },
        .leading_zero => switch (byte) {
            '.' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_dot;
            },
            'e', 'E' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_exp;
            },
            else => return try numberEndAndReplay(g, byte),
        },
        .int_digits => switch (byte) {
            '0'...'9' => {},
            '.' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_dot;
            },
            'e', 'E' => {
                if (integer_only) return try numberEndAndReplay(g, byte);
                n.phase = .after_exp;
            },
            else => return try numberEndAndReplay(g, byte),
        },
        .after_dot => switch (byte) {
            '0'...'9' => n.phase = .frac_digits,
            else => return .reject,
        },
        .frac_digits => switch (byte) {
            '0'...'9' => {},
            'e', 'E' => n.phase = .after_exp,
            else => return try numberEndAndReplay(g, byte),
        },
        .after_exp => switch (byte) {
            '+', '-' => n.phase = .after_exp_sign,
            '0'...'9' => n.phase = .exp_digits,
            else => return .reject,
        },
        .after_exp_sign => switch (byte) {
            '0'...'9' => n.phase = .exp_digits,
            else => return .reject,
        },
        .exp_digits => switch (byte) {
            '0'...'9' => {},
            else => return try numberEndAndReplay(g, byte),
        },
    }
    g.top().sub = .{ .in_number = n };
    return .consumed;
}

fn numberEndAndReplay(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const phase = g.top().sub.in_number.phase;
    switch (phase) {
        .after_minus, .after_dot, .after_exp, .after_exp_sign => return .reject,
        else => {},
    }
    try popAccepted(g);
    return .{ .replay = byte };
}

// ── enum ──────────────────────────────────────────────────────────────────────

fn startEnum(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const schema = g.top().schema;
    if (byte == '"' and schema.enum_strings.len > 0) {
        g.top().sub = .{ .in_enum_string = .{ .match = .{ .lo = 0, .hi = @intCast(schema.enum_strings.len) } } };
        return .consumed;
    }
    g.top().sub = .{ .in_enum = .{ .lo = 0, .hi = @intCast(schema.enum_literals.len) } };
    return stepEnum(g, byte);
}

/// A literal typed whole, though a longer one may still follow it (`1` before `10`).
fn enumLiteralWhole(schema: *const Node, e: EnumState) bool {
    return e.lo < e.hi and schema.enum_literals[e.lo].len == e.pos;
}

fn stepEnum(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const e = g.top().sub.in_enum;
    const schema = g.top().schema;
    const r = narrow(StringNames{ .names = schema.enum_literals }, e.lo, e.hi, e.pos, byte);
    if (r.lo == r.hi) {
        // A whole literal followed by a byte that extends none ends before it: a number's terminator.
        if (!enumLiteralWhole(schema, e)) return .reject;
        try popAccepted(g);
        return .{ .replay = byte };
    }
    const next: EnumState = .{ .lo = r.lo, .hi = r.hi, .pos = e.pos + 1 };
    g.top().sub = .{ .in_enum = next };
    // Done once every literal left ends here.
    if (schema.enum_literals[next.hi - 1].len == next.pos) try popAccepted(g);
    return .consumed;
}

fn stepEnumString(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    const frame = g.top();
    var e = frame.sub.in_enum_string;
    var sink: MatchSink(StringNames) = .{ .m = &e.match, .names = .{ .names = frame.schema.enum_strings }, .free = false };
    switch (stepStr(&e.lex, byte, &sink)) {
        .reject => return .reject,
        .close => try popAccepted(g),
        .consumed => frame.sub = .{ .in_enum_string = e },
    }
    return .consumed;
}

// ── object ────────────────────────────────────────────────────────────────────

fn seenWords(n_props: usize) usize {
    return (n_props + 63) / 64;
}

fn isSeen(seen: []const u64, i: u32) bool {
    return (seen[i / 64] >> @intCast(i % 64)) & 1 != 0;
}

fn startObject(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    if (byte != '{') return .reject;
    const seen_at: u32 = @intCast(g.seen.items.len);
    try g.seen.appendNTimes(g.gpa, 0, seenWords(g.top().schema.obj_properties.len));
    g.top().sub = .{ .in_object = .{ .phase = .expect_first_key_or_close, .seen_at = seen_at } };
    return .consumed;
}

/// Some key can still be opened: a free-form one, or a property not yet seen.
fn keyRemains(schema: *const Node, o: ObjectState) bool {
    return schema.obj_additional or o.seen_count < schema.obj_open;
}

fn stepObject(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var o = g.top().sub.in_object;
    const schema = g.top().schema;

    switch (o.phase) {
        .expect_first_key_or_close, .expect_key => |phase| {
            if (phase == .expect_first_key_or_close and byte == '}') {
                if (o.required_seen < schema.obj_required) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            if (byte != '"') return .reject;
            // A key with no property left to match and no free-form keys is a dead end.
            if (!keyRemains(schema, o)) return .reject;
            o.phase = .in_key;
            o.key = .{};
            o.key_match = .{ .lo = 0, .hi = @intCast(schema.obj_properties.len) };
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .in_key => {
            const names: PropertyNames = .{ .node = schema, .seen = g.seen.items[o.seen_at..][0..seenWords(schema.obj_properties.len)] };
            var sink: MatchSink(PropertyNames) = .{ .m = &o.key_match, .names = names, .free = schema.obj_additional };
            switch (stepStr(&o.key, byte, &sink)) {
                .reject => return .reject,
                .consumed => {},
                .close => {
                    o.current_prop = sink.matched();
                    o.phase = .after_key;
                },
            }
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .after_key => {
            if (byte != ':') return .reject;
            o.phase = .expect_value;
            g.top().sub = .{ .in_object = o };
            return .consumed;
        },
        .expect_value => {
            var child = schema.obj_additional_schema orelse anyNode();
            if (o.current_prop) |idx| {
                const prop = schema.obj_properties[idx];
                child = prop.schema;
                g.seen.items[o.seen_at + idx / 64] |= @as(u64, 1) << @intCast(idx % 64);
                o.seen_count += 1;
                o.required_seen += @intFromBool(prop.required);
            }
            o.phase = .after_value;
            g.top().sub = .{ .in_object = o };
            try g.stack.append(g.gpa, .{ .schema = child, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .after_value => {
            if (byte == ',') {
                // A comma promises another key. With additionalProperties off
                // and every declared property seen there is none, and the
                // expect_key state it leads to has no legal byte at all — an
                // empty mask, which the sampler cannot express. Reject here so
                // `}` stays the only way out.
                if (!keyRemains(schema, o)) return .reject;
                o.phase = .expect_key;
                g.top().sub = .{ .in_object = o };
                return .consumed;
            }
            if (byte == '}') {
                if (o.required_seen < schema.obj_required) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            return .reject;
        },
    }
}

// ── array ─────────────────────────────────────────────────────────────────────

fn startArray(g: *Grammar, byte: u8) StepResult {
    if (byte != '[') return .reject;
    g.top().sub = .{ .in_array = .{ .phase = .expect_first_value_or_close } };
    return .consumed;
}

fn stepArray(g: *Grammar, byte: u8) std.mem.Allocator.Error!StepResult {
    var a = g.top().sub.in_array;
    const schema = g.top().schema;

    switch (a.phase) {
        .expect_first_value_or_close => {
            if (byte == ']') {
                if (schema.arr_min_items) |min| if (a.count < min) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            if (schema.arr_max_items) |max| if (a.count >= max) return .reject;
            a.phase = .after_value;
            a.count += 1;
            g.top().sub = .{ .in_array = a };
            const child = schema.arr_items orelse anyNode();
            try g.stack.append(g.gpa, .{ .schema = child, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .expect_value => {
            if (schema.arr_max_items) |max| if (a.count >= max) return .reject;
            a.phase = .after_value;
            a.count += 1;
            g.top().sub = .{ .in_array = a };
            const child = schema.arr_items orelse anyNode();
            try g.stack.append(g.gpa, .{ .schema = child, .sub = .expect_value });
            return .{ .replay = byte };
        },
        .after_value => {
            if (byte == ',') {
                if (schema.arr_max_items) |max| if (a.count >= max) return .reject;
                a.phase = .expect_value;
                g.top().sub = .{ .in_array = a };
                return .consumed;
            }
            if (byte == ']') {
                if (schema.arr_min_items) |min| if (a.count < min) return .reject;
                try popAccepted(g);
                return .consumed;
            }
            return .reject;
        },
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn popAccepted(g: *Grammar) std.mem.Allocator.Error!void {
    const frame = g.top();
    // A closed object's seen bits are the top of `seen`: objects close innermost first.
    if (frame.sub == .in_object) g.seen.shrinkRetainingCapacity(frame.sub.in_object.seen_at);
    frame.sub = .accepted;
    if (g.stack.items.len <= 1) return; // root accepted, leave on stack
    _ = g.stack.pop();
    // Parent state is already in its post-value phase (e.g., object.after_value);
    // it'll naturally handle the next byte.
}

/// A "no-constraints" schema node. Used when a sub-tree is unconstrained:
///   * inside `anyOf` (relaxed branch — see header doc)
///   * for objects with no `properties`
///   * for arrays with no `items`
///   * for additional-property values when `additionalProperties: true`
///
/// `obj_additional = true` so any property name is accepted; `arr_items = null`
/// so any items kind is accepted (recurses to `any` automatically).
fn anyNode() *const Node {
    const S = struct {
        const node: Node = .{ .kind = .any, .obj_additional = true };
    };
    return &S.node;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn parseSchema(gpa: std.mem.Allocator, src: []const u8) !Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return schema_mod.parse(gpa, v.value);
}

fn feed(g: *Grammar, input: []const u8) !void {
    for (input) |c| {
        const ok = try g.acceptByte(c);
        if (!ok) return error.GrammarRejected;
    }
}

test "grammar accepts simple object matching schema" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "{\"name\":\"alice\"}");
    try testing.expect(g.isComplete());
}

test "grammar rejects missing required" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    // Empty object is missing required `name`
    try testing.expectError(error.GrammarRejected, feed(&g, "{}"));
}

test "grammar rejects unknown property when additional=false" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    // After typing "age" through the prefix-match phase, `e"` tries to close the key — but `age`
    // isn't a known property and additional is false, so it should reject when matching.
    // (Our grammar rejects sooner — when the second char doesn't extend `name`.)
    try testing.expectError(error.GrammarRejected, feed(&g, "{\"a"));
}

test "grammar enforces enum values" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"role":{"enum":["admin","user"]}},"required":["role"]}
    );
    defer schema.deinit();

    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "{\"role\":\"admin\"}");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "{\"role\":\"guest"));
    }
}

test "grammar accepts array of integers" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "[1, 2, 30, -7]");
    try testing.expect(g.isComplete());
}

test "grammar rejects float in integer array" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try testing.expectError(error.GrammarRejected, feed(&g, "[1.5"));
}

test "grammar handles string escapes and unicode" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"string\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "\"hi\\n\\u00e9!\"");
    try testing.expect(g.isComplete());
}

test "grammar rejects unescaped control char in string" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"string\"}");
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try testing.expectError(error.GrammarRejected, feed(&g, "\"hi\nthere\""));
}

test "grammar respects minLength and maxLength" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"string","minLength":2,"maxLength":3}
    );
    defer schema.deinit();
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "\"a\""));
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "\"abc\"");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "\"abcd"));
    }
}

test "grammar respects minItems and maxItems" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3}
    );
    defer schema.deinit();
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "[1]"));
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, "[1,2,3]");
        try testing.expect(g.isComplete());
    }
    {
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try testing.expectError(error.GrammarRejected, feed(&g, "[1,2,3,4"));
    }
}

// The model keeps its own layout inside the root; outside it nothing is admitted.
test "grammar admits free whitespace inside the root value only" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    for (" \t\n\r") |ws| try testing.expect(!(try g.allowedBytes()).contains(ws));
    try testing.expect(!try g.acceptByte('\n'));
    try feed(&g, "{\n  \"x\" : 42\n}");
    try testing.expect(g.isComplete());
    for (" \t\n\r") |ws| try testing.expect(!(try g.allowedBytes()).contains(ws));
    try testing.expect(!try g.acceptByte(' '));
}

test "grammar relaxes anyOf to any-json" {
    var schema = try parseSchema(testing.allocator,
        \\{"anyOf":[{"type":"string"},{"type":"object"}]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "[1,2,3]"); // an array — also accepted under "any"
    try testing.expect(g.isComplete());
}

test "allowedBytes initial returns object opener" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('{'));
    try testing.expect(!mask.contains('['));
    try testing.expect(!mask.contains('"'));
}

test "allowedBytes after { restricts to property prefixes" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    _ = try g.acceptByte('{');
    _ = try g.acceptByte('"');
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('n')); // start of "name"
    try testing.expect(mask.contains('a')); // start of "age"
    try testing.expect(!mask.contains('z'));
}

test "no comma after the last property when additionalProperties is false" {
    // The mask is the only thing standing between the model and off-schema
    // output, so a state it can enter must always have a legal next byte. With
    // every declared property seen and `additionalProperties:false`, a comma
    // leads to expect_key where NOTHING is legal — the mask comes back empty,
    // argmax over an all-(-inf) row picks id 0, and the resulting byte kills
    // the grammar (live 2026-08-11: `"!__employee_id__"` spliced into an
    // otherwise conforming object once enforcement switched off).
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    for ("{\"name\":\"a\",\"age\":1") |b| try testing.expect(try g.acceptByte(b));

    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('}'));
    try testing.expect(!mask.contains(','));
    try testing.expect(!try g.acceptByte(','));
}

test "comma still allowed while a property remains unseen" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    for ("{\"name\":\"a\"") |b| try testing.expect(try g.acceptByte(b));

    const mask = try g.allowedBytes();
    try testing.expect(mask.contains(','));
    // `}` is not: `age` is required and still unseen.
    try testing.expect(!mask.contains('}'));
}

test "snapshot/restore roundtrip" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    _ = try g.acceptByte('{');
    const snap = try g.snapshot();
    defer g.discardSnapshot(snap);

    _ = try g.acceptByte('"');
    _ = try g.acceptByte('a');
    _ = try g.acceptByte('"');
    try g.restoreFrom(snap);

    // After restore we're back at expect_first_key_or_close. `"` is allowed
    // (start the required key); `}` is NOT allowed because required `a` is unseen.
    const mask = try g.allowedBytes();
    try testing.expect(mask.contains('"'));
    try testing.expect(!mask.contains('}'));
}

test "cruise-app-style nested schema parses sample response" {
    const src =
        \\{
        \\  "type":"object",
        \\  "properties":{
        \\    "blocks":{
        \\      "type":"array",
        \\      "items":{
        \\        "anyOf":[
        \\          {"type":"object","properties":{"type":{"enum":["text"]},"content":{"type":"string"}},"required":["type","content"],"additionalProperties":false}
        \\        ]
        \\      }
        \\    }
        \\  },
        \\  "required":["blocks"],
        \\  "additionalProperties":false
        \\}
    ;
    var schema = try parseSchema(testing.allocator, src);
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try feed(&g, "{\"blocks\":[{\"type\":\"text\",\"content\":\"hi\"}]}");
    try testing.expect(g.isComplete());
}

// A masked model whose real argmax is off-schema falls back to whitespace,
// which the grammar admitted forever; the bar is that a free-whitespace run
// ends and structure is forced.
test "free whitespace is capped so a masked model cannot idle forever" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}
    );
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();

    try testing.expect(try g.acceptByte('{'));
    var i: usize = 0;
    while (i < MAX_FREE_WS) : (i += 1) try testing.expect(try g.acceptByte('\n'));
    try testing.expect(!try g.acceptByte('\n'));
    const capped = try g.allowedBytes();
    try testing.expect(!capped.contains(' '));
    try testing.expect(capped.contains('"'));

    // A snapshot carries the run; a structural byte resets it.
    const snap = try g.snapshot();
    defer g.discardSnapshot(snap);
    try feed(&g, "\"x\":");
    try testing.expect(try g.acceptByte(' '));
    try g.restoreFrom(snap);
    try testing.expect(!try g.acceptByte(' '));

    try feed(&g, "\"x\":1}");
    try testing.expect(g.isComplete());
}

fn expectLegal(g: *Grammar, byte: u8, legal: bool) !void {
    const mask = try g.allowedBytes();
    try testing.expectEqual(legal, mask.contains(byte));
}

// A number has no closing byte: its end is the stream's end, so a terminable
// root number must accept EOS or the mask leaves only more digits.
test "a terminable root number is complete" {
    var integer = try parseSchema(testing.allocator, "{\"type\":\"integer\"}");
    defer integer.deinit();
    var g = try Grammar.init(testing.allocator, &integer);
    defer g.deinit();
    try testing.expect(!g.isComplete());
    try feed(&g, "-");
    try testing.expect(!g.isComplete());
    try feed(&g, "4");
    try testing.expect(g.isComplete());
    try feed(&g, "2");
    try testing.expect(g.isComplete());
    try expectLegal(&g, '.', false);

    var number = try parseSchema(testing.allocator, "{\"type\":\"number\"}");
    defer number.deinit();
    inline for (.{ "0", "12", "1.5", "-1.5e3", "2E+10" }) |text| {
        var n = try Grammar.init(testing.allocator, &number);
        defer n.deinit();
        try feed(&n, text);
        try testing.expect(n.isComplete());
    }
    inline for (.{ "-", "1.", "1e", "1e+" }) |text| {
        var n = try Grammar.init(testing.allocator, &number);
        defer n.deinit();
        try feed(&n, text);
        try testing.expect(!n.isComplete());
    }
}

test "a nested number is not complete before its parent closes" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"array\",\"items\":{\"type\":\"integer\"}}");
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "[42");
    try testing.expect(!g.isComplete());
    try feed(&g, "]");
    try testing.expect(g.isComplete());
}

test "root keyword and string values complete on their last byte" {
    inline for (.{
        .{ "{\"type\":\"boolean\"}", "true" },
        .{ "{\"type\":\"boolean\"}", "false" },
        .{ "{\"type\":\"null\"}", "null" },
        .{ "{\"type\":\"string\"}", "\"hi\"" },
    }) |case| {
        var schema = try parseSchema(testing.allocator, case[0]);
        defer schema.deinit();
        var g = try Grammar.init(testing.allocator, &schema);
        defer g.deinit();
        try feed(&g, case[1]);
        try testing.expect(g.isComplete());
    }
}

test "a full array offers only its closing bracket" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"array","items":{"type":"integer"},"maxItems":3}
    );
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "[1,2,3");
    try expectLegal(&g, ',', false);
    try feed(&g, "]");
    try testing.expect(g.isComplete());
}

test "an object with no admissible key offers only its closing brace" {
    var schema = try parseSchema(testing.allocator, "{\"type\":\"object\"}");
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "{");
    try expectLegal(&g, '"', false);
    try feed(&g, "}");
    try testing.expect(g.isComplete());
}

test "a string at maxLength offers only its closing quote" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"string","maxLength":2}
    );
    defer schema.deinit();
    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "\"a\\n");
    try expectLegal(&g, '\\', false);
    try expectLegal(&g, 'b', false);
    try feed(&g, "\"");
    try testing.expect(g.isComplete());
}

// Every state the grammar can reach has a legal byte or is a finished root.
test "no reachable grammar state is a dead end" {
    const schemas = [_][]const u8{
        "{\"type\":\"integer\"}",
        "{\"type\":\"number\"}",
        "{\"type\":\"object\"}",
        "{\"type\":\"object\",\"additionalProperties\":true}",
        "{\"type\":\"array\",\"maxItems\":2}",
        "{\"type\":\"array\",\"items\":{\"type\":\"object\"},\"minItems\":1,\"maxItems\":2}",
        "{\"type\":\"string\",\"minLength\":1,\"maxLength\":3}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"string\",\"maxLength\":2},\"b\":{\"type\":\"array\",\"items\":{\"type\":\"integer\"},\"maxItems\":2}},\"required\":[\"a\"]}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"object\"},\"b\":{\"type\":\"boolean\"}},\"required\":[\"a\",\"b\"]}",
        "{\"type\":\"array\",\"items\":{\"enum\":[\"x\",\"yz\",3]},\"maxItems\":3}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"},\"ab\":{\"type\":\"null\"}},\"required\":[\"a\",\"ab\"]}",
        "{\"type\":\"array\",\"items\":{\"type\":\"array\",\"items\":{\"type\":\"string\",\"maxLength\":1},\"maxItems\":1},\"minItems\":1}",
        "{\"anyOf\":[{\"type\":\"integer\"},{\"type\":\"string\"}]}",
        "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"},\"é😀\":{\"type\":\"null\"},\"ab\":{\"enum\":[\"x\",\"é\",1,10]}},\"required\":[\"a\"],\"additionalProperties\":false}",
        "{\"type\":\"string\",\"minLength\":1,\"maxLength\":2}",
        "{\"type\":\"object\",\"required\":[\"b\"],\"additionalProperties\":{\"type\":\"integer\"}}",
        "{\"type\":\"array\",\"items\":false}",
        "{\"enum\":[1,10,1.5,\"1\",\"😀\"]}",
    };
    for (schemas) |src| {
        var schema = try parseSchema(testing.allocator, src);
        defer schema.deinit();
        var seed: u64 = 0;
        while (seed < 40) : (seed += 1) {
            var prng = std.Random.DefaultPrng.init(seed);
            const rnd = prng.random();
            var g = try Grammar.init(testing.allocator, &schema);
            defer g.deinit();
            var steps: usize = 0;
            while (steps < 300 and !g.isComplete()) : (steps += 1) {
                const mask = try g.allowedBytes();
                var legal: [256]u8 = undefined;
                var n: usize = 0;
                for (0..256) |b| {
                    if (mask.contains(@intCast(b)) and !isWs(@intCast(b))) {
                        legal[n] = @intCast(b);
                        n += 1;
                    }
                }
                if (n == 0) {
                    std.debug.print("dead end: schema {s} seed {d} step {d}\n", .{ src, seed, steps });
                    return error.GrammarDeadEnd;
                }
                try testing.expect(try g.acceptByte(legal[rnd.uintLessThan(usize, n)]));
            }
        }
    }
}

test "a value whose schema names no type accepts any object" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","required":["meta"],"properties":{"meta":{"description":"free-form"}}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "{\"meta\":{\"k\":[1,{\"x\":null}]}}");
    try testing.expect(g.isComplete());
}

test "grammar enforces a $ref into $defs like the inlined schema" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","additionalProperties":false,"required":["facts"],
        \\ "properties":{"facts":{"type":"array","items":{"$ref":"#/$defs/Fact"}}},
        \\ "$defs":{"Fact":{"type":"object","additionalProperties":false,"required":["kind"],
        \\   "properties":{"kind":{"enum":["fact"]}}}}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "{\"facts\":[{\"kind\":\"fact\"}]}");
    try testing.expect(g.isComplete());

    var bad = try Grammar.init(testing.allocator, &schema);
    defer bad.deinit();
    try testing.expectError(error.GrammarRejected, feed(&bad, "{\"facts\":[false"));
}

test "grammar enforces a recursive $ref at every depth" {
    var schema = try parseSchema(testing.allocator,
        \\{"type":"object","additionalProperties":false,"required":["name"],
        \\ "properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#"}}}}
    );
    defer schema.deinit();

    var g = try Grammar.init(testing.allocator, &schema);
    defer g.deinit();
    try feed(&g, "{\"name\":\"a\",\"children\":[{\"name\":\"b\",\"children\":[{\"name\":\"c\"}]}]}");
    try testing.expect(g.isComplete());

    var bad = try Grammar.init(testing.allocator, &schema);
    defer bad.deinit();
    try testing.expectError(error.GrammarRejected, feed(&bad, "{\"name\":\"a\",\"children\":[{\"x"));
}

/// Whether the grammar takes `doc` whole as a finished value of `schema_src`.
fn conforms(gpa: std.mem.Allocator, schema_src: []const u8, doc: []const u8) !bool {
    var schema = try parseSchema(gpa, schema_src);
    defer schema.deinit();
    var g = try Grammar.init(gpa, &schema);
    defer g.deinit();
    for (doc) |b| if (!try g.acceptByte(b)) return false;
    return g.isComplete();
}

// The grammar admits a document exactly when it is valid under the schema: never a valid one masked away, never an
// invalid one let through. Lengths count characters and strings compare by decoded value, as JSON Schema does.
test "the grammar admits exactly the documents the schema accepts" {
    const Case = struct { schema: []const u8, doc: []const u8, valid: bool };
    const max2 = "{\"type\":\"string\",\"maxLength\":2}";
    const max1 = "{\"type\":\"string\",\"maxLength\":1}";
    const min2 = "{\"type\":\"string\",\"minLength\":2}";
    const req_b = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"b\"],\"additionalProperties\":true}";
    const extra_int = "{\"type\":\"object\",\"required\":[\"b\"],\"additionalProperties\":{\"type\":\"integer\"}}";
    const key_a = "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},\"required\":[\"a\"],\"additionalProperties\":false}";
    const no_x = "{\"type\":\"object\",\"properties\":{\"x\":false},\"additionalProperties\":true}";
    const nums = "{\"enum\":[1,10,1.5,\"1\"]}";
    const cases = [_]Case{
        .{ .schema = max2, .doc = "\"éé\"", .valid = true },
        .{ .schema = max2, .doc = "\"ééé\"", .valid = false },
        .{ .schema = max2, .doc = "\"日本\"", .valid = true },
        .{ .schema = min2, .doc = "\"é\"", .valid = false },
        .{ .schema = min2, .doc = "\"\\u00e9\"", .valid = false },
        .{ .schema = min2, .doc = "\"é\\n\"", .valid = true },
        .{ .schema = max1, .doc = "\"😀\"", .valid = true },
        .{ .schema = max1, .doc = "\"\\ud83d\\ude00\"", .valid = true },
        .{ .schema = max1, .doc = "\"\\u00e9\\u00e9\"", .valid = false },
        .{ .schema = max1, .doc = "\"\\ud83dx\"", .valid = false },
        .{ .schema = req_b, .doc = "{\"a\":1}", .valid = false },
        .{ .schema = req_b, .doc = "{\"a\":1,\"b\":[2]}", .valid = true },
        .{ .schema = extra_int, .doc = "{\"b\":\"x\"}", .valid = false },
        .{ .schema = extra_int, .doc = "{\"c\":1,\"b\":2}", .valid = true },
        .{ .schema = extra_int, .doc = "{\"b\":2,\"c\":true}", .valid = false },
        .{ .schema = key_a, .doc = "{\"\\u0061\":1}", .valid = true },
        .{ .schema = key_a, .doc = "{\"\\u0062\":1}", .valid = false },
        .{ .schema = "{\"type\":\"string\",\"enum\":[\"é\"]}", .doc = "\"\\u00e9\"", .valid = true },
        .{ .schema = "{\"enum\":[\"😀\",\"x\"]}", .doc = "\"\\ud83d\\ude00\"", .valid = true },
        .{ .schema = "{\"enum\":[\"😀\",\"x\"]}", .doc = "\"\\ud83d\"", .valid = false },
        .{ .schema = "{\"type\":\"array\",\"items\":false}", .doc = "[]", .valid = true },
        .{ .schema = "{\"type\":\"array\",\"items\":false}", .doc = "[1]", .valid = false },
        .{ .schema = no_x, .doc = "{\"x\":1}", .valid = false },
        .{ .schema = no_x, .doc = "{\"y\":1}", .valid = true },
        .{ .schema = nums, .doc = "10", .valid = true },
        .{ .schema = nums, .doc = "1", .valid = true },
        .{ .schema = nums, .doc = "1.5", .valid = true },
        .{ .schema = nums, .doc = "\"1\"", .valid = true },
        .{ .schema = nums, .doc = "15", .valid = false },
        .{ .schema = "{\"type\":\"array\",\"items\":{\"enum\":[1,10]}}", .doc = "[10,1]", .valid = true },
    };
    for (cases) |c| {
        const got = try conforms(testing.allocator, c.schema, c.doc);
        errdefer std.debug.print("\nschema {s}\ndoc {s}: admitted={}\n", .{ c.schema, c.doc, got });
        try testing.expectEqual(c.valid, got);
    }
}

// A schema's size is no limit: every enum member and every property stays reachable and required.
test "the grammar enforces enums and objects past 64 members and names past 256 bytes" {
    const a = testing.allocator;
    var src: std.ArrayListUnmanaged(u8) = .empty;
    defer src.deinit(a);
    try src.appendSlice(a, "{\"enum\":[");
    for (0..200) |i| try src.print(a, "{s}\"v{d}\"", .{ if (i > 0) "," else "", i });
    try src.appendSlice(a, "]}");
    try testing.expect(try conforms(a, src.items, "\"v64\""));
    try testing.expect(try conforms(a, src.items, "\"v199\""));
    try testing.expect(!try conforms(a, src.items, "\"v200\""));

    src.clearRetainingCapacity();
    try src.appendSlice(a, "{\"type\":\"object\",\"additionalProperties\":false,\"properties\":{");
    for (0..65) |i| try src.print(a, "{s}\"p{d:0>3}\":{{\"type\":\"integer\"}}", .{ if (i > 0) "," else "", i });
    try src.appendSlice(a, "},\"required\":[");
    for (0..65) |i| try src.print(a, "{s}\"p{d:0>3}\"", .{ if (i > 0) "," else "", i });
    try src.appendSlice(a, "]}");
    var doc: std.ArrayListUnmanaged(u8) = .empty;
    defer doc.deinit(a);
    try doc.append(a, '{');
    for (0..64) |i| try doc.print(a, "{s}\"p{d:0>3}\":0", .{ if (i > 0) "," else "", i });
    const missing_last = doc.items.len;
    try doc.append(a, '}');
    try testing.expect(!try conforms(a, src.items, doc.items));
    doc.shrinkRetainingCapacity(missing_last);
    try doc.appendSlice(a, ",\"p064\":0}");
    try testing.expect(try conforms(a, src.items, doc.items));

    const long: [300]u8 = @splat('k');
    const long_schema = "{\"type\":\"object\",\"properties\":{\"" ++ long ++ "\":{\"type\":\"null\"}},\"required\":[\"" ++ long ++ "\"]}";
    try testing.expect(try conforms(a, long_schema, "{\"" ++ long ++ "\":null}"));
}
