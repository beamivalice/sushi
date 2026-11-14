//! JSON Schema → internal IR for grammar-constrained decoding.
//!
//! Supports the "common" subset:
//!   * type (single string or array, incl. null/integer/number/string/boolean/object/array)
//!   * properties / required / additionalProperties (a boolean or a schema)
//!   * items (homogeneous array, `false` for none)
//!   * anyOf / oneOf (treated identically for grammar purposes)
//!   * enum / const (treated as a 1-element enum)
//!   * minLength / maxLength
//!   * minItems / maxItems
//!   * minimum / maximum / exclusiveMinimum / exclusiveMaximum
//!   * pattern (regex; see regex.zig)
//!   * $ref to a local JSON pointer (`#`, `#/$defs/Name`, `#/definitions/Name`),
//!     recursive refs included; any other ref relaxes to `any` and is named in `Schema.relaxed`
//!   * the boolean schemas `true` (any value) and `false` (no value)
//!
//! Strict OpenAI mode: when `additionalProperties` is unspecified, defaults to `false`.
//! A schema no value can meet, or a count past u32, is refused by name (`Detail`), never relaxed.
//!
//! A `Schema` owns an arena allocator; freeing the schema frees every node, string,
//! property name, and compiled regex in one shot.

const std = @import("std");
const regex = @import("regex.zig");

pub const Kind = enum {
    any,
    /// No value (the schema `false`); kept only as a property's schema, where it forbids that key.
    never,
    null_,
    boolean,
    integer,
    number,
    string,
    array,
    object,
    enum_,
    any_of,
};

pub const Property = struct {
    name: []const u8,
    schema: *const Node,
    required: bool,
};

pub const Node = struct {
    kind: Kind,

    // string
    str_min_len: ?u32 = null,
    str_max_len: ?u32 = null,
    str_pattern: ?*const regex.Nfa = null,

    // number / integer
    num_min: ?f64 = null,
    num_max: ?f64 = null,
    num_excl_min: ?f64 = null,
    num_excl_max: ?f64 = null,

    // array
    arr_items: ?*const Node = null,
    arr_min_items: ?u32 = null,
    arr_max_items: ?u32 = null,

    // object
    /// Declared properties plus every `required` name `properties` lacks, sorted by name bytes.
    obj_properties: []const Property = &.{},
    obj_additional: bool = false,
    /// The schema an undeclared key's value meets when `obj_additional`; null = any value.
    obj_additional_schema: ?*const Node = null,
    /// Properties a key may name (schema not `false`), and how many properties are required.
    obj_open: u32 = 0,
    obj_required: u32 = 0,

    // enum / const
    /// The string members, decoded, sorted by bytes.
    enum_strings: []const []const u8 = &.{},
    /// Every other member as canonical JSON text, sorted by bytes.
    enum_literals: []const []const u8 = &.{},

    // any_of (and one_of)
    any_of_options: []const *const Node = &.{},
};

pub const Schema = struct {
    arena: std.heap.ArenaAllocator,
    root: *const Node,
    /// The first construct the parser relaxed to `any` (an unresolved `$ref`), named; null when there is none.
    /// The grammar's own best-effort constructs (anyOf, pattern, numeric bounds) are not counted here.
    relaxed: ?[]const u8 = null,

    pub fn deinit(self: *Schema) void {
        self.arena.deinit();
    }
};

/// Deepest schema-within-schema nesting `parse` accepts.
pub const max_schema_depth: usize = 64;

pub const ParseError = error{
    InvalidSchema,
    UnsupportedConstruct,
    OutOfMemory,
    InvalidPattern,
};

/// What `parseExplained` could not enforce, in words a client can act on.
pub const Detail = struct {
    buf: [128]u8 = undefined,
    len: usize = 0,

    pub fn text(self: *const Detail) []const u8 {
        return self.buf[0..self.len];
    }

    fn set(self: *Detail, comptime fmt: []const u8, args: anytype) void {
        self.len = if (std.fmt.bufPrint(&self.buf, fmt, args)) |written| written.len else |_| self.buf.len;
    }
};

/// Parse a `std.json.Value` into a `Schema`. The returned schema owns its memory.
pub fn parse(gpa: std.mem.Allocator, value: std.json.Value) ParseError!Schema {
    return parseExplained(gpa, value, null);
}

/// `parse`, naming in `detail` the construct that failed it.
pub fn parseExplained(gpa: std.mem.Allocator, value: std.json.Value, detail: ?*Detail) ParseError!Schema {
    var schema: Schema = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .root = undefined,
    };
    errdefer schema.arena.deinit();

    var ctx: Ctx = .{ .arena = schema.arena.allocator(), .root = value, .detail = detail };
    // The root is the `#` ref, so a recursive ref back to it lands on this node.
    schema.root = try parseRef(&ctx, "#", 0);
    if (schema.root.kind == .never) return ctx.invalid("the schema is false: no value conforms", .{});
    schema.relaxed = ctx.relaxed;
    return schema;
}

const Ctx = struct {
    arena: std.mem.Allocator,
    root: std.json.Value,
    refs: std.StringHashMapUnmanaged(*Node) = .empty,
    detail: ?*Detail = null,
    relaxed: ?[]const u8 = null,

    fn refuse(self: *Ctx, err: ParseError, comptime fmt: []const u8, args: anytype) ParseError {
        if (self.detail) |d| d.set(fmt, args);
        return err;
    }

    fn unsupported(self: *Ctx, comptime fmt: []const u8, args: anytype) ParseError {
        return self.refuse(error.UnsupportedConstruct, fmt, args);
    }

    fn invalid(self: *Ctx, comptime fmt: []const u8, args: anytype) ParseError {
        return self.refuse(error.InvalidSchema, fmt, args);
    }

    /// A construct the grammar cannot enforce becomes `any`, and the first one is named for the strict gate.
    fn relax(self: *Ctx, comptime fmt: []const u8, args: anytype) ParseError!*const Node {
        if (self.relaxed == null) self.relaxed = try std.fmt.allocPrint(self.arena, fmt, args);
        return leaf(self.arena, .any);
    }
};

/// A client's name or value cut to what a message can carry.
fn clip(s: []const u8) []const u8 {
    return s[0..@min(s.len, 40)];
}

/// The slot is memoised before its target parses, so a recursive ref points back at it;
/// the grammar only follows a child when it descends into it.
fn parseRef(ctx: *Ctx, ref: []const u8, depth: usize) ParseError!*const Node {
    if (ctx.refs.get(ref)) |slot| return slot;
    var target = resolvePointer(ctx.root, ref) orelse return ctx.relax("$ref '{s}' does not resolve to a local schema", .{clip(ref)});
    var hops: u8 = 0;
    while (target == .object) : (hops += 1) {
        const next = target.object.get("$ref") orelse break;
        if (next != .string) break;
        if (hops == 16) return ctx.relax("$ref '{s}' never reaches a schema", .{clip(ref)});
        target = resolvePointer(ctx.root, next.string) orelse return ctx.relax("$ref '{s}' does not resolve to a local schema", .{clip(next.string)});
    }
    const slot = try ctx.arena.create(Node);
    // What a ref back into this slot reads while its target parses: a cycle is never the schema `false`.
    slot.* = .{ .kind = .any };
    try ctx.refs.put(ctx.arena, ref, slot);
    slot.* = (try parseNode(ctx, target, depth)).*;
    return slot;
}

/// A local JSON pointer into the root's objects; anything else is unresolved.
fn resolvePointer(root: std.json.Value, ref: []const u8) ?std.json.Value {
    if (ref.len == 0 or ref[0] != '#') return null;
    var node = root;
    var it = std.mem.tokenizeScalar(u8, ref[1..], '/');
    while (it.next()) |escaped| {
        var buf: [256]u8 = undefined;
        var len: usize = 0;
        var i: usize = 0;
        while (i < escaped.len) : (i += 1) {
            if (len == buf.len) return null;
            buf[len] = escaped[i];
            if (escaped[i] == '~' and i + 1 < escaped.len and (escaped[i + 1] == '0' or escaped[i + 1] == '1')) {
                buf[len] = if (escaped[i + 1] == '1') '/' else '~';
                i += 1;
            }
            len += 1;
        }
        if (node != .object) return null;
        node = node.object.get(buf[0..len]) orelse return null;
    }
    return node;
}

fn leaf(arena: std.mem.Allocator, kind: Kind) ParseError!*const Node {
    const node = try arena.create(Node);
    node.* = .{ .kind = kind };
    return node;
}

fn parseNode(ctx: *Ctx, value: std.json.Value, depth: usize) ParseError!*const Node {
    const arena = ctx.arena;
    if (depth > max_schema_depth) return ctx.invalid("schema nesting is deeper than {d}", .{max_schema_depth});
    if (value == .bool) return leaf(arena, if (value.bool) .any else .never);
    if (value != .object) return ctx.invalid("a schema must be an object or a boolean", .{});
    const obj = value.object;

    if (obj.get("$ref")) |r| {
        if (r == .string) return parseRef(ctx, r.string, depth);
    }

    // anyOf / oneOf — branching schemas. Both compile to a Kind.any_of node.
    if (obj.get("anyOf")) |v| return try parseAnyOf(ctx, "anyOf", v, depth);
    if (obj.get("oneOf")) |v| return try parseAnyOf(ctx, "oneOf", v, depth);

    // enum / const — one or more allowed literal values.
    if (obj.get("const")) |v| return makeEnum(ctx, &.{v});
    if (obj.get("enum")) |v| {
        if (v != .array) return ctx.invalid("'enum' must be an array", .{});
        return makeEnum(ctx, v.array.items);
    }

    // type — may be a string or an array of strings. Array form compiles to anyOf.
    // Unspecified type — `any`: consumers allow any JSON value.
    const type_val = obj.get("type") orelse return leaf(arena, .any);

    if (type_val == .array) {
        if (type_val.array.items.len == 0) return ctx.invalid("'type' lists no type", .{});
        var options = try arena.alloc(*const Node, type_val.array.items.len);
        for (type_val.array.items, 0..) |t, i| {
            if (t != .string) return ctx.invalid("'type' must be a string or an array of strings", .{});
            // Synthesize a single-type schema by cloning `obj` minus the `type` array.
            options[i] = try parseSingleType(ctx, obj, t.string, depth);
        }
        const node = try arena.create(Node);
        node.* = .{ .kind = .any_of, .any_of_options = options };
        return node;
    }
    if (type_val != .string) return ctx.invalid("'type' must be a string or an array of strings", .{});
    return try parseSingleType(ctx, obj, type_val.string, depth);
}

fn parseAnyOf(ctx: *Ctx, key: []const u8, v: std.json.Value, depth: usize) ParseError!*const Node {
    const arena = ctx.arena;
    if (v != .array or v.array.items.len == 0) return ctx.invalid("'{s}' must be a non-empty array", .{key});
    var options = try arena.alloc(*const Node, v.array.items.len);
    var n: usize = 0;
    for (v.array.items) |item| {
        const option = try parseNode(ctx, item, depth + 1);
        if (option.kind == .never) continue; // an option no value meets adds nothing to the union
        options[n] = option;
        n += 1;
    }
    if (n == 0) return ctx.invalid("every '{s}' option is false: no value conforms", .{key});
    const node = try arena.create(Node);
    node.* = .{ .kind = .any_of, .any_of_options = options[0..n] };
    return node;
}

fn parseSingleType(ctx: *Ctx, obj: std.json.ObjectMap, t: []const u8, depth: usize) ParseError!*const Node {
    const arena = ctx.arena;
    const node = try arena.create(Node);

    if (std.mem.eql(u8, t, "null")) {
        node.* = .{ .kind = .null_ };
    } else if (std.mem.eql(u8, t, "boolean")) {
        node.* = .{ .kind = .boolean };
    } else if (std.mem.eql(u8, t, "integer") or std.mem.eql(u8, t, "number")) {
        node.* = .{
            .kind = if (t[0] == 'i') .integer else .number,
            .num_min = readF64(obj, "minimum"),
            .num_max = readF64(obj, "maximum"),
            .num_excl_min = readF64(obj, "exclusiveMinimum"),
            .num_excl_max = readF64(obj, "exclusiveMaximum"),
        };
    } else if (std.mem.eql(u8, t, "string")) {
        var pattern_nfa: ?*const regex.Nfa = null;
        if (obj.get("pattern")) |pv| {
            if (pv != .string) return ctx.invalid("'pattern' must be a string", .{});
            pattern_nfa = regex.compile(arena, pv.string) catch
                return ctx.refuse(error.InvalidPattern, "string 'pattern' {s} does not compile", .{pv.string[0..@min(pv.string.len, 48)]});
        }
        const min = try readCount(ctx, obj, "minLength");
        const max = try readCount(ctx, obj, "maxLength");
        if (min != null and max != null and min.? > max.?) return ctx.invalid("'minLength' {d} exceeds 'maxLength' {d}", .{ min.?, max.? });
        node.* = .{ .kind = .string, .str_min_len = min, .str_max_len = max, .str_pattern = pattern_nfa };
    } else if (std.mem.eql(u8, t, "array")) {
        const min = try readCount(ctx, obj, "minItems");
        var max = try readCount(ctx, obj, "maxItems");
        var items_node: ?*const Node = null;
        if (obj.get("items")) |iv| {
            // Tuple form (`items: [a, b]`) is not in our subset.
            if (iv == .array) return ctx.unsupported("array 'items' as a list of schemas (tuple form) is not supported", .{});
            const items = try parseNode(ctx, iv, depth + 1);
            if (items.kind == .never) {
                if ((min orelse 0) > 0) return ctx.invalid("'items' is false but 'minItems' is {d}", .{min.?});
                max = 0;
            } else items_node = items;
        }
        if (min != null and max != null and min.? > max.?) return ctx.invalid("'minItems' {d} exceeds 'maxItems' {d}", .{ min.?, max.? });
        node.* = .{ .kind = .array, .arr_items = items_node, .arr_min_items = min, .arr_max_items = max };
    } else if (std.mem.eql(u8, t, "object")) {
        node.* = try parseObject(ctx, obj, depth);
    } else {
        return ctx.unsupported("type '{s}' is not supported", .{t[0..@min(t.len, 48)]});
    }

    return node;
}

/// An object node: its declared properties, plus each `required` name they lack, which takes the
/// `additionalProperties` schema, so the grammar holds one required set.
fn parseObject(ctx: *Ctx, obj: std.json.ObjectMap, depth: usize) ParseError!Node {
    const arena = ctx.arena;
    // additionalProperties defaults to `false` (OpenAI strict semantics).
    var additional = false;
    var extra: ?*const Node = null;
    if (obj.get("additionalProperties")) |ap| {
        if (ap == .bool) additional = ap.bool else {
            const s = try parseNode(ctx, ap, depth + 1);
            additional = s.kind != .never;
            if (additional) extra = s;
        }
    }

    var required: std.StringArrayHashMapUnmanaged(void) = .empty;
    if (obj.get("required")) |r| {
        if (r != .array) return ctx.invalid("'required' must be an array of strings", .{});
        for (r.array.items) |item| {
            if (item != .string) return ctx.invalid("'required' must be an array of strings", .{});
            try required.put(arena, item.string, {});
        }
    }
    const declared: std.json.ObjectMap = if (obj.get("properties")) |pv| blk: {
        if (pv != .object) return ctx.invalid("'properties' must be an object", .{});
        break :blk pv.object;
    } else .empty;

    var props: std.ArrayListUnmanaged(Property) = .empty;
    try props.ensureTotalCapacity(arena, declared.count() + required.count());
    var it = declared.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const child = try parseNode(ctx, entry.value_ptr.*, depth + 1);
        const is_required = required.contains(name);
        if (is_required and child.kind == .never) return ctx.invalid("required '{s}' has the schema false", .{clip(name)});
        props.appendAssumeCapacity(.{ .name = try arena.dupe(u8, name), .schema = child, .required = is_required });
    }
    for (required.keys()) |name| {
        if (declared.contains(name)) continue;
        if (!additional) return ctx.invalid("required '{s}' is not in 'properties' and 'additionalProperties' is false", .{clip(name)});
        props.appendAssumeCapacity(.{ .name = try arena.dupe(u8, name), .schema = extra orelse try leaf(arena, .any), .required = true });
    }
    // Sorted by bytes: the grammar narrows a key being typed to a run of this list.
    std.mem.sort(Property, props.items, {}, lessThanProperty);

    var open: u32 = 0;
    for (props.items) |p| open += @intFromBool(p.schema.kind != .never);
    return .{
        .kind = .object,
        .obj_properties = props.items,
        .obj_additional = additional,
        .obj_additional_schema = extra,
        .obj_open = open,
        .obj_required = @intCast(required.count()),
    };
}

fn lessThanProperty(_: void, a: Property, b: Property) bool {
    return std.mem.lessThan(u8, a.name, b.name);
}

fn lessThanBytes(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// String members compare by decoded value, so they stay decoded; the rest match their canonical text.
fn makeEnum(ctx: *Ctx, items: []const std.json.Value) ParseError!*const Node {
    const arena = ctx.arena;
    if (items.len == 0) return ctx.invalid("'enum' lists no value", .{});
    var n_strings: usize = 0;
    for (items) |v| n_strings += @intFromBool(v == .string);
    const strings = try arena.alloc([]const u8, n_strings);
    const literals = try arena.alloc([]const u8, items.len - n_strings);
    var s: usize = 0;
    var l: usize = 0;
    for (items) |v| {
        if (v == .string) {
            strings[s] = try arena.dupe(u8, v.string);
            s += 1;
        } else {
            literals[l] = canonicalJson(arena, v) catch |err| return switch (err) {
                error.InvalidSchema => ctx.invalid("an 'enum' value nests deeper than {d}", .{max_schema_depth}),
                else => err,
            };
            l += 1;
        }
    }
    std.mem.sort([]const u8, strings, {}, lessThanBytes);
    std.mem.sort([]const u8, literals, {}, lessThanBytes);
    const node = try arena.create(Node);
    node.* = .{ .kind = .enum_, .enum_strings = strings, .enum_literals = literals };
    return node;
}

/// Render a `std.json.Value` to canonical JSON text (no extraneous whitespace).
/// Used so the grammar can prefix-match against allowed enum/const literals byte-by-byte.
fn canonicalJson(arena: std.mem.Allocator, value: std.json.Value) ParseError![]const u8 {
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(arena);
    try writeCanonical(arena, &buf, value, 0);
    return try buf.toOwnedSlice(arena);
}

fn writeCanonical(arena: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), value: std.json.Value, depth: usize) ParseError!void {
    if (depth > max_schema_depth) return error.InvalidSchema;
    switch (value) {
        .null => try buf.appendSlice(arena, "null"),
        .bool => |b| try buf.appendSlice(arena, if (b) "true" else "false"),
        .integer => |i| {
            try buf.print(arena, "{d}", .{i});
        },
        .number_string => |s| try buf.appendSlice(arena, s),
        .float => |f| {
            try buf.print(arena, "{d}", .{f});
        },
        .string => |s| try writeJsonString(arena, buf, s),
        .array => |arr| {
            try buf.append(arena, '[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try buf.append(arena, ',');
                try writeCanonical(arena, buf, item, depth + 1);
            }
            try buf.append(arena, ']');
        },
        .object => |o| {
            try buf.append(arena, '{');
            // Sorted keys for deterministic output
            var keys = arena.alloc([]const u8, o.count()) catch return error.OutOfMemory;
            defer arena.free(keys);
            var it = o.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) keys[i] = entry.key_ptr.*;
            std.mem.sort([]const u8, keys, {}, lessThanBytes);
            for (keys, 0..) |k, j| {
                if (j > 0) try buf.append(arena, ',');
                try writeJsonString(arena, buf, k);
                try buf.append(arena, ':');
                try writeCanonical(arena, buf, o.get(k).?, depth + 1);
            }
            try buf.append(arena, '}');
        },
    }
}

fn writeJsonString(arena: std.mem.Allocator, buf: *std.ArrayListUnmanaged(u8), s: []const u8) ParseError!void {
    try buf.append(arena, '"');
    for (s) |c| {
        switch (c) {
            '"' => try buf.appendSlice(arena, "\\\""),
            '\\' => try buf.appendSlice(arena, "\\\\"),
            '\n' => try buf.appendSlice(arena, "\\n"),
            '\r' => try buf.appendSlice(arena, "\\r"),
            '\t' => try buf.appendSlice(arena, "\\t"),
            0...8, 0x0B, 0x0C, 0x0E...0x1F => {
                var esc: [6]u8 = undefined;
                const n = std.fmt.bufPrint(&esc, "\\u{x:0>4}", .{c}) catch unreachable;
                try buf.appendSlice(arena, n);
            },
            else => try buf.append(arena, c),
        }
    }
    try buf.append(arena, '"');
}

/// A count keyword (`minLength`, `maxItems`, ...): a whole number that fits u32, in any spelling the JSON reader
/// gives; anything else is refused by name rather than dropped or wrapped.
fn readCount(ctx: *Ctx, obj: std.json.ObjectMap, key: []const u8) ParseError!?u32 {
    const v = obj.get(key) orelse return null;
    const n: ?u32 = switch (v) {
        .integer => |i| std.math.cast(u32, i),
        .float => |f| wholeU32(f),
        .number_string => |s| std.fmt.parseInt(u32, s, 10) catch if (std.fmt.parseFloat(f64, s)) |f| wholeU32(f) else |_| null,
        else => null,
    };
    return n orelse ctx.invalid("'{s}' must be a whole number from 0 to {d}", .{ key, std.math.maxInt(u32) });
}

fn wholeU32(f: f64) ?u32 {
    if (!(f >= 0 and f <= std.math.maxInt(u32)) or @floor(f) != f) return null;
    return @intFromFloat(f);
}

fn readF64(obj: std.json.ObjectMap, key: []const u8) ?f64 {
    const v = obj.get(key) orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch null,
        else => null,
    };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn parseStr(gpa: std.mem.Allocator, src: []const u8) !Schema {
    const v = try std.json.parseFromSlice(std.json.Value, gpa, src, .{});
    defer v.deinit();
    return parse(gpa, v.value);
}

test "parse simple object schema with required" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.object, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.obj_properties.len);
    try testing.expectEqualStrings("name", s.root.obj_properties[0].name);
    try testing.expect(s.root.obj_properties[0].required);
    try testing.expectEqual(Kind.string, s.root.obj_properties[0].schema.kind);
    try testing.expect(!s.root.obj_additional);
}

test "additionalProperties defaults to false (OpenAI strict)" {
    var s = try parseStr(testing.allocator, "{\"type\":\"object\"}");
    defer s.deinit();
    try testing.expect(!s.root.obj_additional);
}

test "additionalProperties:true is honored" {
    var s = try parseStr(testing.allocator, "{\"type\":\"object\",\"additionalProperties\":true}");
    defer s.deinit();
    try testing.expect(s.root.obj_additional);
}

test "anyOf becomes any_of node" {
    var s = try parseStr(testing.allocator,
        \\{"anyOf":[{"type":"string"},{"type":"null"}]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.any_of, s.root.kind);
    try testing.expectEqual(@as(usize, 2), s.root.any_of_options.len);
    try testing.expectEqual(Kind.string, s.root.any_of_options[0].kind);
    try testing.expectEqual(Kind.null_, s.root.any_of_options[1].kind);
}

test "type as array compiles to any_of" {
    var s = try parseStr(testing.allocator,
        \\{"type":["string","null"]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.any_of, s.root.kind);
    try testing.expectEqual(@as(usize, 2), s.root.any_of_options.len);
}

test "enum strings stay decoded and the other values render to canonical JSON, each sorted" {
    var s = try parseStr(testing.allocator,
        \\{"enum":["text","cruise_cards",42,true,null,"\u00e9",{"b":1,"a":[2]}]}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.enum_, s.root.kind);
    const strings = [_][]const u8{ "cruise_cards", "text", "\u{e9}" };
    try testing.expectEqual(strings.len, s.root.enum_strings.len);
    for (strings, s.root.enum_strings) |want, got| try testing.expectEqualStrings(want, got);
    const literals = [_][]const u8{ "42", "null", "true", "{\"a\":[2],\"b\":1}" };
    try testing.expectEqual(literals.len, s.root.enum_literals.len);
    for (literals, s.root.enum_literals) |want, got| try testing.expectEqualStrings(want, got);
}

test "const collapses to single-element enum" {
    var s = try parseStr(testing.allocator, "{\"const\":\"hello\"}");
    defer s.deinit();
    try testing.expectEqual(Kind.enum_, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.enum_strings.len);
    try testing.expectEqualStrings("hello", s.root.enum_strings[0]);
    try testing.expectEqual(@as(usize, 0), s.root.enum_literals.len);
}

test "string constraints are captured" {
    var s = try parseStr(testing.allocator,
        \\{"type":"string","minLength":2,"maxLength":10}
    );
    defer s.deinit();
    try testing.expectEqual(@as(?u32, 2), s.root.str_min_len);
    try testing.expectEqual(@as(?u32, 10), s.root.str_max_len);
}

test "number constraints are captured" {
    var s = try parseStr(testing.allocator,
        \\{"type":"number","minimum":0,"exclusiveMaximum":1}
    );
    defer s.deinit();
    try testing.expectEqual(@as(?f64, 0), s.root.num_min);
    try testing.expectEqual(@as(?f64, 1), s.root.num_excl_max);
}

test "array with items schema" {
    var s = try parseStr(testing.allocator,
        \\{"type":"array","items":{"type":"integer"},"minItems":1,"maxItems":3}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.array, s.root.kind);
    try testing.expect(s.root.arr_items != null);
    try testing.expectEqual(Kind.integer, s.root.arr_items.?.kind);
    try testing.expectEqual(@as(?u32, 1), s.root.arr_min_items);
    try testing.expectEqual(@as(?u32, 3), s.root.arr_max_items);
}

test "properties are sorted alphabetically" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"number"},"m":{"type":"boolean"}}}
    );
    defer s.deinit();
    try testing.expectEqual(@as(usize, 3), s.root.obj_properties.len);
    try testing.expectEqualStrings("a", s.root.obj_properties[0].name);
    try testing.expectEqualStrings("m", s.root.obj_properties[1].name);
    try testing.expectEqualStrings("z", s.root.obj_properties[2].name);
}

test "nested cruise-app-style schema" {
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
    var s = try parseStr(testing.allocator, src);
    defer s.deinit();
    try testing.expectEqual(Kind.object, s.root.kind);
    try testing.expectEqual(@as(usize, 1), s.root.obj_properties.len);
    const blocks = s.root.obj_properties[0].schema;
    try testing.expectEqual(Kind.array, blocks.kind);
    const item = blocks.arr_items.?;
    try testing.expectEqual(Kind.any_of, item.kind);
    try testing.expectEqual(@as(usize, 1), item.any_of_options.len);
    const text_opt = item.any_of_options[0];
    try testing.expectEqual(Kind.object, text_opt.kind);
    // Type discriminator should be an enum
    var found_type = false;
    for (text_opt.obj_properties) |p| {
        if (std.mem.eql(u8, p.name, "type")) {
            found_type = true;
            try testing.expectEqual(Kind.enum_, p.schema.kind);
            try testing.expectEqualStrings("text", p.schema.enum_strings[0]);
        }
    }
    try testing.expect(found_type);
}

test "boolean schema true => any" {
    const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, "true", .{});
    defer v.deinit();
    var s = try parse(testing.allocator, v.value);
    defer s.deinit();
    try testing.expectEqual(Kind.any, s.root.kind);
}

test "unsupported tuple-form items returns error" {
    const result = parseStr(testing.allocator,
        \\{"type":"array","items":[{"type":"string"},{"type":"number"}]}
    );
    try testing.expectError(error.UnsupportedConstruct, result);
}

test "a schema the parser refuses names the construct" {
    const Case = struct { schema: []const u8, err: ParseError, names: []const u8 };
    const cases = [_]Case{
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"nonsense\"}}}", .err = error.UnsupportedConstruct, .names = "type 'nonsense'" },
        .{ .schema = "{\"type\":\"array\",\"items\":[{\"type\":\"string\"}]}", .err = error.UnsupportedConstruct, .names = "tuple" },
        .{ .schema = "{\"type\":\"string\",\"pattern\":\"(\"}", .err = error.InvalidPattern, .names = "pattern" },
    };
    for (cases) |c| {
        const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.schema, .{});
        defer v.deinit();
        var detail: Detail = .{};
        try testing.expectError(c.err, parseExplained(testing.allocator, v.value, &detail));
        try testing.expect(std.mem.indexOf(u8, detail.text(), c.names) != null);
    }
}

test "schema nesting beyond the depth limit is refused, ordinary depth parses" {
    const a = testing.allocator;
    const deep = max_schema_depth + 8;
    var src: std.ArrayListUnmanaged(u8) = .empty;
    defer src.deinit(a);
    for (0..deep) |_| try src.appendSlice(a, "{\"type\":\"array\",\"items\":");
    try src.appendSlice(a, "{\"type\":\"string\"}");
    for (0..deep) |_| try src.append(a, '}');
    try testing.expectError(error.InvalidSchema, parseStr(a, src.items));

    var ok: std.ArrayListUnmanaged(u8) = .empty;
    defer ok.deinit(a);
    for (0..16) |_| try ok.appendSlice(a, "{\"type\":\"array\",\"items\":");
    try ok.appendSlice(a, "{\"type\":\"string\"}");
    for (0..16) |_| try ok.append(a, '}');
    var s = try parseStr(a, ok.items);
    s.deinit();
}

test "$ref resolves through $defs and definitions; a recursive ref points back at its target" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{
        \\  "a":{"$ref":"#/$defs/A"},"b":{"$ref":"#/definitions/B"},"self":{"$ref":"#"},
        \\  "remote":{"$ref":"https://example.com/x.json"}},
        \\ "$defs":{"A":{"type":"string"}},"definitions":{"B":{"type":"integer"}}}
    );
    defer s.deinit();
    const props = s.root.obj_properties; // sorted: a, b, remote, self
    try testing.expectEqual(Kind.string, props[0].schema.kind);
    try testing.expectEqual(Kind.integer, props[1].schema.kind);
    try testing.expectEqual(Kind.any, props[2].schema.kind);
    try testing.expectEqual(s.root, props[3].schema);
    try testing.expect(std.mem.indexOf(u8, s.relaxed.?, "https://example.com/x.json") != null);
}

test "a $ref chain that never reaches a schema relaxes to any" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"loop":{"$ref":"#/$defs/A"}},"$defs":{"A":{"$ref":"#/$defs/A"}}}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.any, s.root.obj_properties[0].schema.kind);
    try testing.expect(std.mem.indexOf(u8, s.relaxed.?, "never reaches a schema") != null);
}

test "a JSON pointer keeps an unknown ~ escape and decodes ~0 and ~1" {
    var s = try parseStr(testing.allocator,
        \\{"type":"object","properties":{"a":{"$ref":"#/$defs/a~1b"},"c":{"$ref":"#/$defs/t~0"},"d":{"$ref":"#/$defs/x~2"}},
        \\ "$defs":{"a/b":{"type":"string"},"t~":{"type":"integer"},"x~2":{"type":"boolean"}}}
    );
    defer s.deinit();
    try testing.expectEqual(Kind.string, s.root.obj_properties[0].schema.kind);
    try testing.expectEqual(Kind.integer, s.root.obj_properties[1].schema.kind);
    try testing.expectEqual(Kind.boolean, s.root.obj_properties[2].schema.kind);
    try testing.expectEqual(@as(?[]const u8, null), s.relaxed);
}

// A schema no value can meet, or a bound past what the IR holds, is refused by name: the grammar would otherwise
// reach a state with no legal byte, or enforce a different number than the client sent.
test "a schema no value meets, or a bound out of range, is refused by name" {
    const Case = struct { schema: []const u8, names: []const u8 };
    const cases = [_]Case{
        .{ .schema = "false", .names = "false" },
        .{ .schema = "{\"type\":\"string\",\"maxLength\":4294967296}", .names = "'maxLength'" },
        .{ .schema = "{\"type\":\"string\",\"maxLength\":99999999999999999999}", .names = "'maxLength'" },
        .{ .schema = "{\"type\":\"string\",\"maxLength\":-1}", .names = "'maxLength'" },
        .{ .schema = "{\"type\":\"string\",\"minLength\":2.5}", .names = "'minLength'" },
        .{ .schema = "{\"type\":\"array\",\"maxItems\":\"3\"}", .names = "'maxItems'" },
        .{ .schema = "{\"type\":\"string\",\"minLength\":3,\"maxLength\":2}", .names = "'minLength'" },
        .{ .schema = "{\"type\":\"array\",\"minItems\":2,\"maxItems\":1}", .names = "'minItems'" },
        .{ .schema = "{\"type\":\"array\",\"items\":false,\"minItems\":1}", .names = "'items'" },
        .{ .schema = "{\"type\":\"object\",\"properties\":{},\"required\":[\"b\"]}", .names = "'b'" },
        .{ .schema = "{\"type\":\"object\",\"required\":[\"b\"],\"additionalProperties\":false}", .names = "'b'" },
        .{ .schema = "{\"type\":\"object\",\"properties\":{\"x\":false},\"required\":[\"x\"]}", .names = "'x'" },
        .{ .schema = "{\"enum\":[]}", .names = "'enum'" },
        .{ .schema = "{\"anyOf\":[false]}", .names = "'anyOf'" },
        .{ .schema = "{\"type\":[]}", .names = "'type'" },
    };
    for (cases) |c| {
        const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, c.schema, .{});
        defer v.deinit();
        var detail: Detail = .{};
        const got = parseExplained(testing.allocator, v.value, &detail);
        if (got) |s| {
            var owned = s;
            owned.deinit();
        } else |_| {}
        errdefer std.debug.print("\nschema {s}: detail '{s}'\n", .{ c.schema, detail.text() });
        try testing.expect(std.meta.isError(got));
        try testing.expect(std.mem.indexOf(u8, detail.text(), c.names) != null);
    }
}

test "a bound spelled as a whole float or as the client's digits is read" {
    for ([_]std.json.ParseOptions{ .{}, .{ .parse_numbers = false } }) |opts| {
        const v = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"type\":\"string\",\"minLength\":1.0,\"maxLength\":4294967295}", opts);
        defer v.deinit();
        var s = try parse(testing.allocator, v.value);
        defer s.deinit();
        try testing.expectEqual(@as(?u32, 1), s.root.str_min_len);
        try testing.expectEqual(@as(?u32, std.math.maxInt(u32)), s.root.str_max_len);
    }
}
