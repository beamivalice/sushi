const std = @import("std");
const log = @import("log.zig");
const tool_calls_mod = @import("tool_calls.zig");
const parseToolCalls = tool_calls_mod.parseToolCalls;
const balancedJsonObject = tool_calls_mod.balancedJsonObject;
const looseRepairContainer = tool_calls_mod.looseRepairContainer;
const think_tags = @import("think_tags.zig");
const trimLeakedToolMarkup = think_tags.trimLeakedToolMarkup;
const testing = std.testing;

/// True when `s` strict-parses as a JSON object. Cheap arbiter for the
/// parseToolCalls safety net.
pub fn jsonIsValidObject(allocator: std.mem.Allocator, s: []const u8) bool {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, s, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object;
}

pub const ParsedToolCall = struct {
    name: []const u8,
    arguments: []const u8, // JSON string
    /// True when the call was HEURISTICALLY inferred from bare JSON in the
    /// output (no tag syntax — the model never said "this is a tool call").
    /// The request chokepoint validates inferred names against the declared
    /// tools schema: a truncated DATA object like {"name": "George
    /// Washington", "num": 1, …} must never become a tool call (live pi
    /// capture 2026-07-13). Explicit tag-format calls keep flowing even with
    /// unknown names — "tool not found" is feedback the model corrects from.
    inferred: bool = false,
};

/// Locate a tool's `properties` map in an OpenAI-shaped tools array. Accepts
/// both the wrapped (`{"type":"function","function":{…}}`) and flat forms.
/// The tool's JSON-Schema `parameters` object (Anthropic `input_schema` too).
/// Callers that need only the property map use `toolPropertiesFor`; the hoist
/// also needs `required`, which lives beside `properties` here.
fn toolParametersFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    if (tools != .array) return null;
    for (tools.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv != .string or !std.mem.eql(u8, nv.string, name)) continue;
        const params = func.get("parameters") orelse func.get("input_schema") orelse return null;
        if (params != .object) return null;
        return params.object;
    }
    return null;
}

fn propertiesOf(params: std.json.ObjectMap) ?std.json.ObjectMap {
    const props = params.get("properties") orelse return null;
    if (props != .object) return null;
    return props.object;
}

fn toolPropertiesFor(tools: std.json.Value, name: []const u8) ?std.json.ObjectMap {
    return propertiesOf(toolParametersFor(tools, name) orelse return null);
}

/// The types a misplaced value may be hoisted on. A required OBJECT/ARRAY param
/// found inside another container is too speculative to move — only scalars.
fn isScalarJsonType(want: []const u8) bool {
    return std.mem.eql(u8, want, "string") or std.mem.eql(u8, want, "integer") or
        std.mem.eql(u8, want, "number") or std.mem.eql(u8, want, "boolean");
}

const SchemaTypes = struct {
    items: [4][]const u8 = undefined,
    len: usize = 0,

    fn add(self: *SchemaTypes, t: []const u8) void {
        if (std.mem.eql(u8, t, "null") or self.len == self.items.len) return;
        for (self.items[0..self.len]) |have| if (std.mem.eql(u8, have, t)) return;
        self.items[self.len] = t;
        self.len += 1;
    }

    fn slice(self: *const SchemaTypes) []const []const u8 {
        return self.items[0..self.len];
    }
};

const schema_ref_depth = 8;

/// The non-null JSON types a property declares, in declaration order. `"type"` may be a union
/// array; without one, `anyOf`/`oneOf` contribute every branch, `allOf` its first typed branch,
/// and a local `$ref` into `$defs`/`definitions` of `root` its target. Odd/absent → empty,
/// which means "leave the value alone".
fn collectJsonTypes(prop: std.json.Value, root: std.json.ObjectMap, out: *SchemaTypes, depth: u8) void {
    if (prop != .object or depth > schema_ref_depth) return;
    if (prop.object.get("type")) |t| {
        switch (t) {
            .string => |s| out.add(s),
            .array => |arr| for (arr.items) |item| if (item == .string) out.add(item.string),
            else => {},
        }
        return;
    }
    if (prop.object.get("$ref")) |r| {
        if (r != .string) return;
        inline for (.{ "#/$defs/", "#/definitions/" }) |prefix| {
            if (std.mem.startsWith(u8, r.string, prefix)) {
                const defs = root.get(prefix[2 .. prefix.len - 1]) orelse return;
                if (defs != .object) return;
                if (defs.object.get(r.string[prefix.len..])) |target| collectJsonTypes(target, root, out, depth + 1);
                return;
            }
        }
        return;
    }
    inline for (.{ "anyOf", "oneOf" }) |key| {
        if (prop.object.get(key)) |branches| {
            if (branches == .array) for (branches.array.items) |b| collectJsonTypes(b, root, out, depth + 1);
            return;
        }
    }
    if (prop.object.get("allOf")) |branches| {
        if (branches == .array) for (branches.array.items) |b| {
            collectJsonTypes(b, root, out, depth + 1);
            if (out.len > 0) return;
        };
    }
}

/// The first type a property declares (see `collectJsonTypes`), or null.
fn declaredJsonType(prop: std.json.Value, root: std.json.ObjectMap) ?[]const u8 {
    var types = SchemaTypes{};
    collectJsonTypes(prop, root, &types, 0);
    return if (types.len > 0) types.items[0] else null;
}

/// Tolerant boolean spelling — the union of what weak models actually emit:
/// Python's `False`, shell's `0`/`1`, prose `yes`/`no`, plus the trailing-comma
/// weld the app's `appendFlagIsTrue` already tolerates. Anything else → null
/// (never guess; an honest client validation error beats a wrong value).
fn looseBoolSpelling(raw: []const u8) ?bool {
    const s = std.mem.trim(u8, raw, " \t\r\n,");
    if (s.len == 0 or s.len > 5) return null;
    var buf: [5]u8 = undefined;
    const lower = std.ascii.lowerString(buf[0..s.len], s);
    if (std.mem.eql(u8, lower, "true") or std.mem.eql(u8, lower, "yes") or
        std.mem.eql(u8, lower, "on") or std.mem.eql(u8, lower, "1")) return true;
    if (std.mem.eql(u8, lower, "false") or std.mem.eql(u8, lower, "no") or
        std.mem.eql(u8, lower, "off") or std.mem.eql(u8, lower, "0")) return false;
    return null;
}

/// Strict parse of a container string that admits a key repeated with the SAME
/// value (issue #402: a model spelling `header` twice per item) and refuses a
/// key repeated with different values: the two are told apart by parsing
/// first-wins and last-wins and comparing the documents.
fn parseContainerAllowingRepeats(allocator: std.mem.Allocator, text: []const u8) ?std.json.Parsed(std.json.Value) {
    var first = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_first }) catch return null;
    var last = std.json.parseFromSlice(std.json.Value, allocator, text, .{ .duplicate_field_behavior = .use_last }) catch {
        first.deinit();
        return null;
    };
    defer last.deinit();
    const a = std.json.Stringify.valueAlloc(allocator, first.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(a);
    const b = std.json.Stringify.valueAlloc(allocator, last.value, .{}) catch {
        first.deinit();
        return null;
    };
    defer allocator.free(b);
    if (!std.mem.eql(u8, a, b)) {
        first.deinit();
        return null;
    }
    return first;
}

/// Coerce ONE value to the declared type. Returns null when the value already
/// matches or the spelling is undecidable. Allocated replacements are appended
/// to `owned`; parsed sub-documents to `docs` — both are freed by the caller
/// AFTER the object is re-serialized, because the returned Value borrows them.
fn coerceValueToType(
    allocator: std.mem.Allocator,
    v: std.json.Value,
    want: []const u8,
    owned: *std.ArrayList([]const u8),
    docs: *std.ArrayList(std.json.Parsed(std.json.Value)),
) !?std.json.Value {
    // A CONTAINER param carried through a tag format arrives as a STRING — no tag
    // format types its values. Live (15 hits in one session): pi's `edit.edits`,
    // an array of {oldText,newText}, shipped as `"[{\"oldText\":…}]"` on every
    // call. Strict parse only: a value that isn't well-formed JSON of the
    // DECLARED kind is left alone rather than guessed at.
    if (std.mem.eql(u8, want, "array") or std.mem.eql(u8, want, "object")) {
        if (v != .string) return null;
        const t = std.mem.trim(u8, v.string, " \t\r\n");
        const want_array = want[0] == 'a';
        if (t.len == 0 or t[0] != (if (want_array) @as(u8, '[') else @as(u8, '{'))) return null;

        // Strict first; a well-formed container is the common case.
        if (parseContainerAllowingRepeats(allocator, t)) |doc| {
            const kind_ok = if (want_array) doc.value == .array else doc.value == .object;
            if (kind_ok) {
                try docs.append(allocator, doc);
                return doc.value;
            }
            var d = doc;
            d.deinit();
            return null;
        }

        // The container string is itself MANGLED (live: pi's `edits` array with a
        // missing comma between two object values). Same big-file escaping class
        // as looseRepairToolCallJson — re-serialize tolerantly, then STRICT-parse
        // the result so a mis-repair that yields invalid JSON is discarded.
        const repaired = looseRepairContainer(allocator, t) orelse return null;
        defer allocator.free(repaired);
        var rdoc = parseContainerAllowingRepeats(allocator, repaired) orelse {
            log.debug("[tool-autocorrect] {s} param left as a string: not well-formed JSON of that kind\n", .{want});
            return null;
        };
        const rkind_ok = if (want_array) rdoc.value == .array else rdoc.value == .object;
        if (!rkind_ok) {
            rdoc.deinit();
            return null;
        }
        try docs.append(allocator, rdoc); // ownership transfers; never deinit here
        return rdoc.value;
    }
    if (std.mem.eql(u8, want, "boolean")) {
        return switch (v) {
            .bool => null,
            .string => |s| if (looseBoolSpelling(s)) |b| std.json.Value{ .bool = b } else null,
            .integer => |i| if (i == 0 or i == 1) std.json.Value{ .bool = i == 1 } else null,
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "integer")) {
        return switch (v) {
            .string => |s| blk: {
                const n = std.fmt.parseInt(i64, std.mem.trim(u8, s, " \t\r\n"), 10) catch break :blk null;
                break :blk std.json.Value{ .integer = n };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "number")) {
        return switch (v) {
            .string => |s| blk: {
                const t = std.mem.trim(u8, s, " \t\r\n");
                if (std.fmt.parseInt(i64, t, 10)) |n| break :blk std.json.Value{ .integer = n } else |_| {}
                const f = std.fmt.parseFloat(f64, t) catch break :blk null;
                if (std.math.isNan(f) or std.math.isInf(f)) break :blk null; // not representable in JSON
                break :blk std.json.Value{ .float = f };
            },
            else => null,
        };
    }
    if (std.mem.eql(u8, want, "string")) {
        // The inverse repair: the tag/custom formats infer type from the value's
        // SPELLING (isJsonLiteral), so a string param whose content happens to
        // read as `false`/`42` was shipped as a bool/number. The schema is the
        // only thing that can tell those apart.
        return switch (v) {
            .bool => |b| std.json.Value{ .string = if (b) "true" else "false" },
            .integer => |i| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{i});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .float => |f| blk: {
                const s = try std.fmt.allocPrint(allocator, "{d}", .{f});
                try owned.append(allocator, s);
                break :blk std.json.Value{ .string = s };
            },
            .number_string => |s| std.json.Value{ .string = s },
            else => null,
        };
    }
    return null; // a type we don't model — never guessed
}

fn jsonTypeMatches(v: std.json.Value, want: []const u8) bool {
    if (v == .null) return true; // an explicit null satisfies any optional field
    if (std.mem.eql(u8, want, "boolean")) return v == .bool;
    if (std.mem.eql(u8, want, "integer")) return v == .integer or v == .number_string;
    if (std.mem.eql(u8, want, "number")) return v == .integer or v == .float or v == .number_string;
    if (std.mem.eql(u8, want, "string")) return v == .string;
    if (std.mem.eql(u8, want, "array")) return v == .array;
    if (std.mem.eql(u8, want, "object")) return v == .object;
    return true; // a type we don't model constrains nothing
}

/// True when every argument whose type the tool DECLARES actually carries that
/// JSON type. This is the invariant strict clients (Claude Code, pi, opencode)
/// enforce and reject on; the format corpus asserts it universally so a new
/// entry in any family is covered without a bespoke assertion. Undeclared keys
/// and unmodelled types constrain nothing.
pub fn toolCallConformsToSchema(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return true;
    const props = propertiesOf(params) orelse return true;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .object) return true;

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const prop = props.get(entry.key_ptr.*) orelse continue;
        var types = SchemaTypes{};
        collectJsonTypes(prop, params, &types, 0);
        if (types.len == 0) continue;
        for (types.slice()) |want| {
            if (jsonTypeMatches(entry.value_ptr.*, want)) break;
        } else return false;
    }
    return true;
}

/// True when a REQUIRED scalar param is MISSING at the top level while sitting
/// buried inside one of the tool's declared container args — exactly the shape
/// `hoistMisplacedRequiredParams` exists to repair, and exactly what a strict
/// client rejects with "must have required properties X". The format corpus
/// asserts this is FALSE for every entry that declares tools, so a future family
/// that produces the buried shape is covered without a bespoke assertion.
pub fn requiredParamIsBuried(
    allocator: std.mem.Allocator,
    call: ParsedToolCall,
    tools_json: []const u8,
) bool {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return false;
    defer tools.deinit();
    const params = toolParametersFor(tools.value, call.name) orelse return false;
    const props_v = params.get("properties") orelse return false;
    const required_v = params.get("required") orelse return false;
    if (props_v != .object or required_v != .array) return false;
    const props = props_v.object;

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;

    for (required_v.array.items) |req_v| {
        if (req_v != .string) continue;
        const name = req_v.string;
        if (parsed.value.object.get(name) != null) continue;
        const want = declaredJsonType(props.get(name) orelse continue, params) orelse continue;
        if (!isScalarJsonType(want)) continue;

        var it = parsed.value.object.iterator();
        while (it.next()) |e| {
            const prop = props.get(e.key_ptr.*) orelse continue;
            if (containerItemDeclares(prop, name)) continue;
            if (buriedParamIn(e.value_ptr, name, want, false) != null) return true;
        }
    }
    return false;
}

/// Coerce every parsed call's `arguments` to the types its tool DECLARES.
///
/// Class: **type inference from value spelling.** Neither the tag formats
/// (`<parameter=x>False</parameter>`, Gemma's `key:false`) nor a model writing
/// JSON by hand carries type information a parser can trust — `isJsonLiteral`
/// has to guess from the bytes, and it guesses wrong in both directions:
///   • `False` / `"false"` on a boolean param → shipped as the STRING "false";
///   • `false` / `42` as a string param's CONTENT → shipped as a bool/number.
/// Strict clients (Claude Code, pi, opencode) reject both and the model, which
/// cannot see its own serialized request, loops forever "fixing" a value that
/// was already correct. The tool schema is the only disambiguator, and it is
/// already threaded to every parse site as `tools_json`.
///
/// Contract: only where the schema declares a type, and only when the spelling
/// is unambiguous. An undecidable value is left alone so the client's
/// validation error stays honest. Correct calls from strong models are
/// byte-unchanged (nothing re-serializes unless a coercion actually fired).
///
/// Scalars AND containers: a tag format types nothing, so a declared
/// `array`/`object` arrives as a string of JSON and `coerceValueToType` parses
/// it back (strict, then one tolerant repair re-validated strictly). This
/// comment used to say "only SCALARS are touched" — it was two weeks stale by
/// the time it was read as a spec and filed as issue #94. A contract comment is
/// load-bearing documentation: pin it with a test (`issue #94: a container
/// param carried as literal TEXT…`), don't let it drift.
pub fn coerceToolArgsToSchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props = propertiesOf(params) orelse continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var owned = std.ArrayList([]const u8).empty;
        defer {
            for (owned.items) |s| allocator.free(s);
            owned.deinit(allocator);
        }
        // Parsed container sub-documents: the rewritten object BORROWS from these,
        // so they must outlive the Stringify below.
        var docs = std.ArrayList(std.json.Parsed(std.json.Value)).empty;
        defer {
            for (docs.items) |*d| d.deinit();
            docs.deinit(allocator);
        }

        var changed = false;
        var it = parsed.value.object.iterator();
        while (it.next()) |entry| {
            const prop = props.get(entry.key_ptr.*) orelse continue;
            var types = SchemaTypes{};
            collectJsonTypes(prop, params, &types, 0);
            if (types.len == 0) continue;
            // Among several candidates a value that already fits one is kept.
            if (types.len > 1) {
                const fits = for (types.slice()) |want| {
                    if (jsonTypeMatches(entry.value_ptr.*, want)) break true;
                } else false;
                if (fits) continue;
            }
            const new_val = for (types.slice()) |want| {
                if (try coerceValueToType(allocator, entry.value_ptr.*, want, &owned, &docs)) |nv| break nv;
            } else continue;
            entry.value_ptr.* = new_val;
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        log.info("  [tool-parse] coerced {s} arguments to the declared schema types\n", .{call.name});
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does this container's ITEM schema declare `name` as one of its own
/// properties? If it does, a `name` sitting inside it BELONGS there and must
/// never be lifted out — a multi-file edit tool would lose its per-item paths.
/// This is the whole safety argument for the hoist below, so it is deliberately
/// conservative: no declared item `properties` map ⇒ nothing is proven ⇒ false.
fn containerItemDeclares(prop: std.json.Value, name: []const u8) bool {
    if (prop != .object) return false;
    // Array property → the item schema. Object property → the schema itself.
    const inner = if (prop.object.get("items")) |items| items else prop;
    if (inner != .object) return false;
    const props = inner.object.get("properties") orelse return false;
    if (props != .object) return false;
    return props.object.get(name) != null;
}

/// Scalar equality — the only shapes we ever hoist on. A container value is
/// never a candidate, so a structural compare is not needed.
fn jsonScalarEql(a: std.json.Value, b: std.json.Value) bool {
    return switch (a) {
        .string => |s| b == .string and std.mem.eql(u8, s, b.string),
        .integer => |i| b == .integer and i == b.integer,
        .bool => |x| b == .bool and x == b.bool,
        else => false,
    };
}

/// Find `name` buried one level down inside `container`, requiring UNANIMITY:
/// every object in the container carries `name`, all with the same scalar value,
/// and that value matches the type the schema declares for the top-level param.
/// Anything less is ambiguous → null. With `remove`, also deletes it from each
/// object (the returned value keeps pointing into the parse arena, which
/// outlives the removal).
fn buriedParamIn(
    container: *std.json.Value,
    name: []const u8,
    want: []const u8,
    remove: bool,
) ?std.json.Value {
    switch (container.*) {
        .array => |*arr| {
            if (arr.items.len == 0) return null;
            var found: ?std.json.Value = null;
            for (arr.items) |*item| {
                if (item.* != .object) return null;
                const v = item.object.get(name) orelse return null; // must be in EVERY item
                if (v == .null or !jsonTypeMatches(v, want)) return null;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) return null; // items disagree → ambiguous
                } else found = v;
            }
            if (remove) for (arr.items) |*item| {
                _ = item.object.orderedRemove(name);
            };
            return found;
        },
        .object => |*obj| {
            const v = obj.get(name) orelse return null;
            if (v == .null or !jsonTypeMatches(v, want)) return null;
            if (remove) _ = obj.orderedRemove(name);
            return v;
        },
        else => return null,
    }
}

/// Lift a REQUIRED top-level parameter the model BURIED inside a container arg.
///
/// Class: **misplaced required param.** A weak model that has internalized "the
/// edit object holds everything about the edit" writes the file path INSIDE each
/// `edits[]` item and never emits it at the top level:
///
///     {"edits":[{"newText":"…","oldText":"…","path":"us_presidents/generate_site.sh"}]}
///
/// The call is valid JSON and correctly escaped, so no repair path fires and no
/// type is wrong — it is purely in the wrong PLACE, which only the schema can
/// know. Strict clients reject it ("path: must have required properties path"),
/// and the model, which cannot see its own serialized request, re-emits the same
/// call until it gives up and falls back to rewriting the whole file with
/// `write` — three dead multi-thousand-token rounds in the live 2026-07-13 pi
/// session (gemma-4-26B-A4B). Same shape as the Claude Code replace_all loop.
///
/// Contract — every condition is read off the SCHEMA, never guessed:
///   • the param is declared REQUIRED at the top level and is ABSENT there;
///   • it is declared a SCALAR (hoisting a container is too speculative);
///   • it sits inside a DECLARED container arg whose item schema does NOT
///     declare it (`containerItemDeclares` — otherwise it belongs where it is);
///   • every object in that container carries it with the SAME value and the
///     declared type.
/// Ambiguity of any kind (items disagreeing, two containers offering different
/// values, a wrong-typed value) leaves the call untouched so the client's
/// validation error stays honest. A compliant call never re-serializes.
pub fn hoistMisplacedRequiredParams(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !void {
    var tools = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return;
    defer tools.deinit();
    if (tools.value != .array) return;

    for (calls) |*call| {
        const params = toolParametersFor(tools.value, call.name) orelse continue;
        const props_v = params.get("properties") orelse continue;
        const required_v = params.get("required") orelse continue;
        if (props_v != .object or required_v != .array) continue;
        const props = props_v.object;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, call.arguments, .{}) catch continue;
        defer parsed.deinit();
        if (parsed.value != .object) continue;

        var changed = false;
        for (required_v.array.items) |req_v| {
            if (req_v != .string) continue;
            const name = req_v.string; // borrows from `tools`, which outlives the Stringify
            if (parsed.value.object.get(name) != null) continue; // present — nothing misplaced

            const want = declaredJsonType(props.get(name) orelse continue, params) orelse continue;
            if (!isScalarJsonType(want)) continue;

            // Probe every declared container WITHOUT mutating: a value found in
            // two containers that disagree is ambiguous, and we must not have
            // already gutted the first one by then.
            var found: ?std.json.Value = null;
            var ambiguous = false;
            var it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                const v = buriedParamIn(e.value_ptr, name, want, false) orelse continue;
                if (found) |f| {
                    if (!jsonScalarEql(f, v)) {
                        ambiguous = true;
                        break;
                    }
                } else found = v;
            }
            if (ambiguous or found == null) continue;

            // Commit: strip it from the containers, install it at the top level.
            it = parsed.value.object.iterator();
            while (it.next()) |e| {
                const prop = props.get(e.key_ptr.*) orelse continue;
                if (containerItemDeclares(prop, name)) continue;
                _ = buriedParamIn(e.value_ptr, name, want, true);
            }
            try parsed.value.object.put(parsed.arena.allocator(), name, found.?);
            log.info("  [tool-parse] hoisted misplaced required arg '{s}' to the top level for {s}\n", .{ name, call.name });
            changed = true;
        }
        if (!changed) continue;

        const rewritten = std.json.Stringify.valueAlloc(allocator, parsed.value, .{}) catch continue;
        allocator.free(call.arguments);
        call.arguments = rewritten;
    }
}

/// Does the OpenAI-shaped tools array declare a function with this name?
/// Accepts both the wrapped ({"type":"function","function":{…}}) and flat
/// forms, mirroring `toolPropertiesFor`. An unparseable/odd `tools_json`
/// returns TRUE — never drop a call because OUR schema parse failed.
pub fn toolNameIsDeclared(allocator: std.mem.Allocator, tools_json: []const u8, name: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return true;
    defer parsed.deinit();
    if (parsed.value != .array) return true;
    for (parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func = blk: {
            if (tool_val.object.get("function")) |f| {
                if (f == .object) break :blk f.object;
            }
            break :blk tool_val.object;
        };
        const nv = func.get("name") orelse continue;
        if (nv == .string and std.mem.eql(u8, nv.string, name)) return true;
    }
    return false;
}

/// Drop HEURISTICALLY-inferred raw-JSON calls (`ParsedToolCall.inferred`)
/// whose name the request never declared. The George Washington class: a
/// generation truncated mid-data-script leaves `{"name": "George Washington",
/// "num": 1, …}` as the first balanced object, and the flat-shape synthesis
/// promoted it to a tool call the client can only answer with "tool not
/// found" — burning the turn. Explicit tag-format calls are NEVER dropped
/// here (undeclared name → honest client feedback the model corrects from).
///
/// Returns the retained slice (shrunk in place), or null when every call was
/// dropped — the caller treats that exactly like "no tool calls", so the text
/// stays visible content and finish_reason (e.g. "length") is preserved.
pub fn filterInferredBySchema(
    allocator: std.mem.Allocator,
    calls: []ParsedToolCall,
    tools_json: []const u8,
) !?[]ParsedToolCall {
    var kept: usize = 0;
    for (calls) |tc| {
        if (!tc.inferred or toolNameIsDeclared(allocator, tools_json, tc.name)) {
            calls[kept] = tc;
            kept += 1;
        } else {
            log.warn("  [tool-parse] dropped inferred raw-JSON call '{s}' — name not among the request's tools (data object, not a call)\n", .{tc.name});
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    if (kept == calls.len) return calls;
    if (kept == 0) {
        allocator.free(calls);
        return null;
    }
    return try allocator.realloc(calls, kept);
}

/// Last-resort tool-call inference for models that emit JUST the arguments
/// object — no tool name, no wrapper syntax. Observed live from Gemma 4 12B
/// via Claude Code: ```` ```json\n{"file_path": …, "content": …}\n``` ````
/// with the Write tool defined; parseToolCalls finds nothing (no "name" key)
/// and the un-executed JSON leaked into the visible text.
///
/// Conservative contract — fires ONLY when parseToolCalls returned null:
///   • the visible content (after a leading think block) must START with the
///     JSON object, optionally wrapped in a markdown fence;
///   • the object's keys must satisfy exactly ONE tool in `tools_json`
///     (OpenAI shape): required ⊆ keys AND keys ⊆ properties;
///   • zero or multiple matching tools → null (ambiguity never guesses).
pub fn inferBareJsonToolCalls(allocator: std.mem.Allocator, text: []const u8, tools_json: []const u8) !?[]ParsedToolCall {
    // Strip a leading think block, mirroring parseToolCalls.
    var effective_text = text;
    if (std.mem.indexOf(u8, text, "<channel|>")) |end| {
        effective_text = std.mem.trimStart(u8, text[end + 10 ..], "\n ");
    } else if (std.mem.indexOf(u8, text, "</think>")) |think_end| {
        effective_text = std.mem.trimStart(u8, text[think_end + 8 ..], "\n ");
    }
    var trimmed = std.mem.trimStart(u8, effective_text, "\n \t");
    // Skip one opening markdown fence line (``` or ```json etc.).
    if (std.mem.startsWith(u8, trimmed, "```")) {
        const nl = std.mem.indexOfScalar(u8, trimmed, '\n') orelse return null;
        trimmed = std.mem.trimStart(u8, trimmed[nl + 1 ..], "\n \t");
    }
    // The object must LEAD the visible content — a JSON example mid-prose is
    // never promoted to a call.
    if (!std.mem.startsWith(u8, trimmed, "{")) return null;
    const obj_slice = balancedJsonObject(trimmed) orelse return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, obj_slice, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const obj = parsed.value.object;
    if (obj.count() == 0) return null;

    const tools_parsed = std.json.parseFromSlice(std.json.Value, allocator, tools_json, .{}) catch return null;
    defer tools_parsed.deinit();
    if (tools_parsed.value != .array) return null;

    var match_name: ?[]const u8 = null;
    for (tools_parsed.value.array.items) |tool_val| {
        if (tool_val != .object) continue;
        const func_val = tool_val.object.get("function") orelse continue;
        if (func_val != .object) continue;
        const name_val = func_val.object.get("name") orelse continue;
        if (name_val != .string) continue;
        const params_val = func_val.object.get("parameters") orelse continue;
        if (params_val != .object) continue;
        const props_val = params_val.object.get("properties") orelse continue;
        if (props_val != .object) continue;
        const props = props_val.object;
        if (props.count() == 0) continue;

        // keys ⊆ properties
        var keys_ok = true;
        var it = obj.iterator();
        while (it.next()) |kv| {
            if (props.get(kv.key_ptr.*) == null) {
                keys_ok = false;
                break;
            }
        }
        if (!keys_ok) continue;

        // required ⊆ keys
        if (params_val.object.get("required")) |req_val| {
            if (req_val == .array) {
                var req_ok = true;
                for (req_val.array.items) |r| {
                    if (r != .string) continue;
                    if (obj.get(r.string) == null) {
                        req_ok = false;
                        break;
                    }
                }
                if (!req_ok) continue;
            }
        }

        if (match_name != null) return null; // ambiguous — never guess
        match_name = name_val.string;
    }

    const name = match_name orelse return null;
    const calls = try allocator.alloc(ParsedToolCall, 1);
    errdefer allocator.free(calls);
    const name_owned = try allocator.dupe(u8, name);
    errdefer allocator.free(name_owned);
    calls[0] = .{ .name = name_owned, .arguments = try allocator.dupe(u8, obj_slice) };
    log.info("  [tool-parse] inferred bare-args call to '{s}' via unique schema match\n", .{name});
    return calls;
}

const test_tools_write_bash =
    \\[{"type":"function","function":{"name":"Write","description":"Write a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"content":{"type":"string"}},"required":["file_path","content"]}}},
    \\ {"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}}]
;

test "inferBareJsonToolCalls maps bare-args fenced JSON to the unique matching tool (Claude Code capture)" {
    const allocator = testing.allocator;
    // Shape captured live from gemma-4-12b via Claude Code /v1/messages: thought
    // block, then a ```json fence holding ONLY the Write tool's arguments.
    const text = "<|channel>thought\nI will create this file using Write.<channel|>```json\n{\n  \"file_path\": \"/Users/user/mlx_info.html\",\n  \"content\": \"<h1>MLX</h1>\"\n}\n```\nI've created the file.";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
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
    try testing.expectEqualStrings("/Users/user/mlx_info.html", parsed.value.object.get("file_path").?.string);
}

test "inferBareJsonToolCalls unfenced bare object at content start" {
    const allocator = testing.allocator;
    const text = "{\"command\": \"ls -la\"}";
    const calls = (try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) orelse return error.NoCalls;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try testing.expectEqualStrings("Bash", calls[0].name);
}

test "inferBareJsonToolCalls refuses ambiguous and non-matching objects" {
    const allocator = testing.allocator;
    // Two tools share the same key → ambiguous → null.
    const dup_tools =
        \\[{"type":"function","function":{"name":"a","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}},
        \\ {"type":"function","function":{"name":"b","parameters":{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}}}]
    ;
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"x\": \"1\"}", dup_tools)) == null);
    // Keys not in any tool's properties → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"zzz\": 1}", test_tools_write_bash)) == null);
    // Missing a required key → null.
    try testing.expect((try inferBareJsonToolCalls(allocator, "{\"file_path\": \"a.txt\"}", test_tools_write_bash)) == null);
}

test "inferBareJsonToolCalls ignores JSON that does not lead the content" {
    const allocator = testing.allocator;
    // Example object mid-prose must never be promoted to a call.
    const text = "Here is an example of the payload you could send: {\"command\": \"ls\"} — adjust as needed.";
    try testing.expect((try inferBareJsonToolCalls(allocator, text, test_tools_write_bash)) == null);
}

test "filterInferredBySchema drops undeclared inferred calls, keeps declared and explicit ones" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    ;
    // Inferred + undeclared → dropped; slice becomes null.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "George Washington"),
            .arguments = try allocator.dupe(u8, "{\"num\":1}"),
            .inferred = true,
        };
        try testing.expect((try filterInferredBySchema(allocator, calls, tools)) == null);
    }
    // Inferred + declared → kept.
    {
        var calls = try allocator.alloc(ParsedToolCall, 1);
        calls[0] = .{
            .name = try allocator.dupe(u8, "write"),
            .arguments = try allocator.dupe(u8, "{\"path\":\"a\"}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
    }
    // Explicit + undeclared → kept (client feedback, not our guess); the
    // inferred sibling with an undeclared name is dropped and the slice shrinks.
    {
        var calls = try allocator.alloc(ParsedToolCall, 2);
        calls[0] = .{
            .name = try allocator.dupe(u8, "searchWeb"),
            .arguments = try allocator.dupe(u8, "{\"q\":\"zig\"}"),
            .inferred = false,
        };
        calls[1] = .{
            .name = try allocator.dupe(u8, "John Adams"),
            .arguments = try allocator.dupe(u8, "{\"num\":2}"),
            .inferred = true,
        };
        const kept = (try filterInferredBySchema(allocator, calls, tools)).?;
        defer {
            for (kept) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(kept);
        }
        try testing.expectEqual(@as(usize, 1), kept.len);
        try testing.expectEqualStrings("searchWeb", kept[0].name);
    }
}

test "toolNameIsDeclared: wrapped + flat forms; unparseable schema never drops" {
    const allocator = testing.allocator;
    const wrapped =
        \\[{"type":"function","function":{"name":"write","parameters":{"type":"object","properties":{}}}}]
    ;
    const flat =
        \\[{"name":"write","parameters":{"type":"object","properties":{}}}]
    ;
    try testing.expect(toolNameIsDeclared(allocator, wrapped, "write"));
    try testing.expect(toolNameIsDeclared(allocator, flat, "write"));
    try testing.expect(!toolNameIsDeclared(allocator, wrapped, "George Washington"));
    // Broken schema JSON → safe default TRUE (never drop on OUR parse failure).
    try testing.expect(toolNameIsDeclared(allocator, "not json", "anything"));
}

test "qwen xml: the live <function=…>/<parameter=…> capture (package.json class)" {
    // Verbatim from ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve via pi (2026-08-12).
    // Qwen 3.5/3.6's OWN template dictates this dialect ("ONLY reply in the
    // following format"), so it is the checkpoint's contract, not a mangle.
    // The JSON-only reader mined the first balanced object out of the body —
    // the file CONTENT — and promoted its "name" key to the tool name, so pi
    // answered "Tool voxel-pagoda-garden not found" and the model looped.
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n" ++
        "/Users/d/pi-lfm/package.json\n</parameter>\n<parameter=content>\n" ++
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}\n" ++
        "</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqualStrings("/Users/d/pi-lfm/package.json", o.get("path").?.string);
    // The value is RAW BYTES: a JSON file stays a STRING (typing by spelling
    // is what turned this call into `{"name": "voxel-pagoda-garden", …}`).
    // Types come from the schema at the request chokepoint, never from here.
    try testing.expectEqualStrings(
        "{\n  \"name\": \"voxel-pagoda-garden\",\n  \"version\": \"1.0.0\"\n}",
        o.get("content").?.string,
    );
}

test "qwen xml: truncated mid-parameter ships NAME + the completed params only" {
    const allocator = testing.allocator;
    const text =
        "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.ts\n</parameter>\n" ++
        "<parameter=content>\nexport const half = ";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);

    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("/tmp/a.ts", parsed.value.object.get("path").?.string);
    try testing.expect(parsed.value.object.get("content") == null);
}

test "issue #94: a container param carried as literal TEXT coerces to a real array/object" {
    // #94 reports this as an open gap on the strength of the doc comment
    // ("only SCALARS are touched"), which was stale — `coerceValueToType` has
    // had its array/object arm since 60ba5ec (2026-07-09), two weeks before
    // the issue was filed. Pinned as a TEST rather than left as a code read,
    // so the claim is settled by evidence and cannot silently regress.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"stats","parameters":{"type":"object","properties":{"nums":{"type":"array","items":{"type":"integer"}},"opts":{"type":"object"},"label":{"type":"string"}}}}}]
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "stats"),
        .arguments = try allocator.dupe(u8, "{\"nums\":\"[1, 2, 3]\",\"opts\":\"{\\\"grid\\\": true}\",\"label\":\"[not an array]\"}"),
    }};
    defer {
        allocator.free(calls[0].name);
        allocator.free(calls[0].arguments);
    }

    try coerceToolArgsToSchema(allocator, &calls, tools);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    const o = parsed.value.object;

    const nums = o.get("nums").?.array;
    try testing.expectEqual(@as(usize, 3), nums.items.len);
    try testing.expectEqual(@as(i64, 2), nums.items[1].integer);
    try testing.expectEqual(true, o.get("opts").?.object.get("grid").?.bool);
    // A STRING param whose content merely looks bracket-y is untouched — the
    // schema decides, never the spelling.
    try testing.expectEqualStrings("[not an array]", o.get("label").?.string);
}

// ── Schema-aware argument coercion (value-spelling type-inference class) ──

/// The Edit tool as Claude Code declares it (OpenAI shape, post
/// `buildOpenAIToolsJson`): `replace_all` is a BOOLEAN, the rest are strings.
const edit_tools_json_test =
    \\[{"type":"function","function":{"name":"Edit","description":"Edit a file","parameters":{"type":"object","properties":{"file_path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean","default":false}},"required":["file_path","old_string","new_string"]}}}]
;

fn parseArgsObj(allocator: std.mem.Allocator, args: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, args, .{});
}

test "coerceToolArgsToSchema: Python-style False on a boolean param becomes JSON false" {
    // VERBATIM capture, 2026-07-09, Qwen3.6-35B-A3B via Claude Code on
    // ~/.sushi/logs/sushi-11234.log:109471. The model emits the Hermes
    // XML parameter form with Python's `False`; parseHermesToolCall's
    // isJsonLiteral only knows lowercase, so it shipped the STRING "False" and
    // Claude Code rejected every Edit with
    //   "The parameter `replace_all` type is expected as `boolean` but provided as `string`"
    // — six times in a row, until the model gave up and rewrote the whole file.
    const allocator = testing.allocator;
    const raw =
        "\n<tool_call>\n<function=Edit>\n<parameter=replace_all>\nFalse\n</parameter>\n" ++
        "<parameter=file_path>\n/Users/user/doom/index.html\n</parameter>\n" ++
        "<parameter=old_string>\n  <script src=\"game.js\"></script>\n</parameter>\n" ++
        "<parameter=new_string>\n  <script src=\"game.js\" type=\"module\"></script>\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const v = parsed.value.object.get("replace_all").?;
    try testing.expect(v == .bool);
    try testing.expectEqual(false, v.bool);
    // Same capture, the whitespace half of the class: `old_string` must match
    // the file "exactly, including indentation", so the two leading spaces the
    // model sent are payload. Trimming them either failed the edit or matched a
    // DIFFERENT, un-indented occurrence mid-line and rewrote it at the wrong
    // nesting; coercion leaves a declared string alone, so the parser is the
    // only layer that can preserve them.
    try testing.expectEqualStrings(
        "  <script src=\"game.js\"></script>",
        parsed.value.object.get("old_string").?.string,
    );
    try testing.expectEqualStrings(
        "  <script src=\"game.js\" type=\"module\"></script>",
        parsed.value.object.get("new_string").?.string,
    );
}

test "coerceToolArgsToSchema: unknown tool / unknown key / undecidable value pass through" {
    const allocator = testing.allocator;
    // Unknown tool name → args untouched.
    {
        const raw = "<tool_call>{\"name\":\"Nope\",\"arguments\":{\"replace_all\":\"False\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        try testing.expect(parsed.value.object.get("replace_all").? == .string);
    }
    // Known tool, boolean param, value that is not a recognizable boolean →
    // left alone so the client's validation error stays honest.
    {
        const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"replace_all\":\"maybe\"}}</tool_call>";
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const v = parsed.value.object.get("replace_all").?;
        try testing.expect(v == .string);
        try testing.expectEqualStrings("maybe", v.string);
    }
}

/// pi's `edit` tool, verbatim from its own schema
/// (@earendil-works/pi-coding-agent dist/core/tools/edit.js — typebox
/// `editSchema`). Two facts the tests below lean on: `path` is REQUIRED at the
/// TOP level, and the array's item schema declares ONLY oldText/newText — it
/// never declares `path`. That asymmetry is what makes a buried `path`
/// provably misplaced rather than a judgment call.
const pi_edit_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to the file to edit (relative or absolute)"},"edits":{"type":"array","items":{"type":"object","properties":{"oldText":{"type":"string"},"newText":{"type":"string"}},"required":["oldText","newText"]},"description":"One or more targeted replacements."}},"required":["path","edits"]}}}]
;

const edit_array_tools_json_test =
    \\[{"type":"function","function":{"name":"edit","description":"Edit a file","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object"}},"dry_run":{"type":"boolean"}},"required":["path","edits"]}}}]
;

test "coerceToolArgsToSchema: an ARRAY param passed through Hermes XML is not left a string" {
    // Live capture class (15 occurrences in one session): `<parameter=edits>` holds
    // a JSON array, but parseHermesToolCall's isJsonLiteral only recognizes
    // scalars, so the whole array shipped as a STRING. Strict clients reject it
    // ("edits: want array, got str"); lenient ones silently mis-execute.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n/tmp/a.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"const a = 1;\", \"newText\": \"const a = 2;\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("const a = 2;", edits.array.items[0].object.get("newText").?.string);
    // sibling scalars still coerce/pass through
    try testing.expectEqualStrings("/tmp/a.js", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: identical duplicate keys inside a container param still coerce; conflicting ones do not" {
    // Issue #402: Qwen Code's ask_user_question array arrived as a string whose
    // items repeated `header` with the same value; strict parse rejects the
    // duplicate, the tolerant re-emit keeps it, and the string reached the
    // client unchanged. Identical duplicates carry no ambiguity; conflicting
    // ones do, and guessing a value is worse than leaving the string alone.
    const allocator = testing.allocator;
    const cases = [_]struct { inner: []const u8, want_array: bool }{
        .{ .inner = "[{\"header\":\"Scope\",\"question\":\"Which scope?\",\"header\":\"Scope\",\"options\":[{\"label\":\"Small\"}]}]", .want_array = true },
        .{ .inner = "[{\"header\":\"Scope\",\"header\":\"Size\"}]", .want_array = false },
    };
    for (cases) |c| {
        const raw = try std.fmt.allocPrint(allocator, "<tool_call>\n<function=edit>\n<parameter=edits>\n{s}\n</parameter>\n</function>\n</tool_call>", .{c.inner});
        defer allocator.free(raw);
        const calls = (try parseToolCalls(allocator, raw)).?;
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }
        try coerceToolArgsToSchema(allocator, calls, edit_array_tools_json_test);
        const parsed = try parseArgsObj(allocator, calls[0].arguments);
        defer parsed.deinit();
        const edits = parsed.value.object.get("edits").?;
        if (c.want_array) {
            try testing.expect(edits == .array);
            try testing.expectEqualStrings("Scope", edits.array.items[0].object.get("header").?.string);
            try testing.expectEqual(@as(usize, 3), edits.array.items[0].object.count());
        } else {
            try testing.expect(edits == .string);
        }
    }
}

test "coerceToolArgsToSchema: a STRING param whose content is JSON stays a string" {
    // The inverse guard for arrays/objects: `old_string` may legitimately BE the
    // text `[1,2]`. Only the schema decides.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=Edit>\n<parameter=file_path>\na.js\n</parameter>\n" ++
        "<parameter=old_string>\n[1,2]\n</parameter>\n<parameter=new_string>\n{\"k\":1}\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("old_string").? == .string);
    try testing.expectEqualStrings("[1,2]", parsed.value.object.get("old_string").?.string);
    try testing.expect(parsed.value.object.get("new_string").? == .string);
}

test "coerceToolArgsToSchema: already-conforming args are byte-identical (no-op guard)" {
    // The auto-correct layer must never touch a good call. Property: when every
    // declared type already matches, `arguments` comes out byte-for-byte the same.
    const allocator = testing.allocator;
    const raw = "<tool_call>{\"name\":\"Edit\",\"arguments\":{\"file_path\":\"a.js\",\"old_string\":\"a\"," ++
        "\"new_string\":\"b\",\"replace_all\":true}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, edit_tools_json_test);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

// ── Auto-correct safety properties (deterministic fuzz) ─────────────────────
// The repair layer is only ever allowed to touch input that is actually broken.
// These generate well-formed tool calls whose values deliberately SPELL other
// JSON types ("false", "42", "[1,2]", "null") — exactly the strings the
// value-spelling inference used to mis-type — and assert round-trip identity.

fn jsonValueEql(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => a.bool == b.bool,
        .integer => a.integer == b.integer,
        .float => a.float == b.float,
        .number_string => std.mem.eql(u8, a.number_string, b.number_string),
        .string => std.mem.eql(u8, a.string, b.string),
        .array => blk: {
            if (a.array.items.len != b.array.items.len) break :blk false;
            for (a.array.items, b.array.items) |x, y| if (!jsonValueEql(x, y)) break :blk false;
            break :blk true;
        },
        .object => blk: {
            if (a.object.count() != b.object.count()) break :blk false;
            var it = a.object.iterator();
            while (it.next()) |e| {
                const other = b.object.get(e.key_ptr.*) orelse break :blk false;
                if (!jsonValueEql(e.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

/// Value strings chosen to collide with every JSON literal spelling the
/// tag-format parsers used to guess from.
const adversarial_strings = [_][]const u8{
    "false",      "true",         "False",   "True",    "null",   "None",
    "42",         "-7",           "3.14",    "0",       "1",      "",
    "[1,2]",      "{\"k\":1}",    "nan",     "inf",     "  ",     "yes",
    "a\"b",       "line\nline",   "tab\there", "back\\slash", "café ☕", "0x1f",
    "{not json",  "[unclosed",    "1e400",   "00",      "+5",     "-",
};

test "fuzz: a conforming tool call round-trips byte-identical through parse+coerce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE_1234);
    const rand = prng.random();

    const types = [_][]const u8{ "string", "integer", "number", "boolean", "array", "object" };

    var iter: usize = 0;
    while (iter < 400) : (iter += 1) {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Build a random schema + a CONFORMING args object for it.
        var props: std.json.ObjectMap = .empty;
        var args: std.json.ObjectMap = .empty;
        const nprops = 1 + rand.uintLessThan(usize, 5);
        var p: usize = 0;
        while (p < nprops) : (p += 1) {
            const key = try std.fmt.allocPrint(arena, "p{d}", .{p});
            const want = types[rand.uintLessThan(usize, types.len)];

            var prop: std.json.ObjectMap = .empty;
            try prop.put(arena, "type", .{ .string = want });
            try props.put(arena, key, .{ .object = prop });

            const v: std.json.Value = if (std.mem.eql(u8, want, "string"))
                .{ .string = adversarial_strings[rand.uintLessThan(usize, adversarial_strings.len)] }
            else if (std.mem.eql(u8, want, "integer"))
                .{ .integer = @as(i64, @intCast(rand.uintLessThan(u32, 1000))) - 500 }
            else if (std.mem.eql(u8, want, "number"))
                .{ .float = 1.5 }
            else if (std.mem.eql(u8, want, "boolean"))
                .{ .bool = rand.boolean() }
            else if (std.mem.eql(u8, want, "array")) blk: {
                var arr = std.json.Array.init(arena);
                try arr.append(.{ .string = "x" });
                try arr.append(.{ .integer = 1 });
                break :blk .{ .array = arr };
            } else blk: {
                var o: std.json.ObjectMap = .empty;
                try o.put(arena, "k", .{ .string = "v" });
                break :blk .{ .object = o };
            };
            try args.put(arena, key, v);
        }

        var func: std.json.ObjectMap = .empty;
        try func.put(arena, "name", .{ .string = "t" });
        var params: std.json.ObjectMap = .empty;
        try params.put(arena, "type", .{ .string = "object" });
        try params.put(arena, "properties", .{ .object = props });
        try func.put(arena, "parameters", .{ .object = params });
        var tool: std.json.ObjectMap = .empty;
        try tool.put(arena, "type", .{ .string = "function" });
        try tool.put(arena, "function", .{ .object = func });
        var tools_arr = std.json.Array.init(arena);
        try tools_arr.append(.{ .object = tool });
        const tools_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .array = tools_arr }, .{});

        // Serialize as a canonical Hermes JSON call — always VALID input.
        const args_json = try std.json.Stringify.valueAlloc(arena, std.json.Value{ .object = args }, .{});
        const raw = try std.fmt.allocPrint(arena,
            "<tool_call>{{\"name\":\"t\",\"arguments\":{s}}}</tool_call>", .{args_json});

        const calls = (try parseToolCalls(allocator, raw)) orelse {
            std.debug.print("\n[fuzz iter {d}] valid tool call did not parse\n  {s}\n", .{ iter, raw });
            return error.FuzzFailed;
        };
        defer {
            for (calls) |tc| {
                allocator.free(tc.name);
                allocator.free(tc.arguments);
            }
            allocator.free(calls);
        }

        const before = try arena.dupe(u8, calls[0].arguments);
        try coerceToolArgsToSchema(allocator, calls, tools_json);

        // Property 1: a conforming call is left BYTE-identical.
        if (!std.mem.eql(u8, before, calls[0].arguments)) {
            std.debug.print("\n[fuzz iter {d}] auto-correct mutated a conforming call\n  before: {s}\n  after : {s}\n", .{ iter, before, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 2: the values survive verbatim (no spelling-based retyping).
        var got = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
        defer got.deinit();
        if (!jsonValueEql(std.json.Value{ .object = args }, got.value)) {
            std.debug.print("\n[fuzz iter {d}] value mismatch\n  want: {s}\n  got : {s}\n", .{ iter, args_json, calls[0].arguments });
            return error.FuzzFailed;
        }

        // Property 3: conformance holds, and coercion is idempotent.
        try testing.expect(toolCallConformsToSchema(allocator, calls[0], tools_json));
        try coerceToolArgsToSchema(allocator, calls, tools_json);
        try testing.expectEqualStrings(before, calls[0].arguments);
    }
}

test "coerceToolArgsToSchema: a MANGLED array-typed param is tolerantly repaired" {
    // Live soak capture (record 268): pi's `edits` array with a MISSING COMMA
    // between two object values — `"newText": "..." "path": "..."`. Strict parse
    // rejects it, so pre-fix `edits` stayed a STRING (schema contradiction). The
    // tolerant container repair recovers a real array.
    const allocator = testing.allocator;
    const raw = "<tool_call>\n<function=edit>\n<parameter=path>\n./game.js\n</parameter>\n" ++
        "<parameter=edits>\n[{\"oldText\": \"  speed: 8,\", \"newText\": \"  vy: 0,\" \"extra\": \"x\"}]\n</parameter>\n" ++
        "</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    try testing.expectEqualStrings("  speed: 8,", edits.array.items[0].object.get("oldText").?.string);
}

// ── Misplaced required param (buried-`path` class) ──────────────────────────
//
// VERBATIM captured arguments, 2026-07-13 pi session (gemma-4-26B-A4B-it-qat-4bit,
// ~/.sushi/logs/sushi-11234.log around the us_presidents run). The model
// put `path` INSIDE each edits[] item and emitted no top-level `path`, so pi
// answered
//     Validation failed for tool "edit":
//       - path: must have required properties path
// three times in a row. Each rejection was a full multi-thousand-token
// generation; the model then abandoned `edit` and rewrote the whole file with
// `write` — the same "give up on the cheap tool, re-emit the expensive one"
// degradation as the Claude Code replace_all loop.
//
// The parse layer is NOT at fault (verified: convertGemma4Object/Array are a
// structural walk that preserves the model's own nesting and key order, and
// coerceToolArgsToSchema only rewrites values in place) — the model genuinely
// emitted it this way. The schema is what makes it provably fixable: `path` is
// required at the TOP level and the item schema declares only oldText/newText.
//
// The script bodies are the real ones, excerpted for readability — their LENGTH
// is not load-bearing (the JSON was already valid and correctly escaped; that is
// exactly why no repair path fired). The nesting and the escaping are verbatim.
const pi_buried_path_args_captured =
    \\{"edits":[{"newText":"#!/bin/bash\n\n# Get the directory where the script is located\nSCRIPT_DIR=\"$( cd \"$( dirname \"${BASH_SOURCE[0]}\" )\" && pwd )\"\ncd \"$SCRIPT_DIR\"\n","oldText":"#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n","path":"us_presidents/generate_site.sh"}]}
;

test "hoistMisplacedRequiredParams: pi's buried `path` is lifted out of the edits items" {
    const allocator = testing.allocator;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, pi_buried_path_args_captured),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }

    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();

    // The required param now satisfies the schema at the top level…
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expect(path == .string);
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);

    // …and is gone from the item, whose schema never declared it.
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits == .array);
    try testing.expectEqual(@as(usize, 1), edits.array.items.len);
    const item = edits.array.items[0].object;
    try testing.expect(item.get("path") == null);

    // The payload the model worked for survives byte-exact, escaping intact.
    try testing.expectEqualStrings(
        "#!/bin/bash\n\n# Data for the presidents\n# Format: \"Name|Term|Bio\"\n",
        item.get("oldText").?.string,
    );
    try testing.expect(std.mem.startsWith(u8, item.get("newText").?.string, "#!/bin/bash\n"));
    try testing.expect(std.mem.indexOf(u8, item.get("newText").?.string, "${BASH_SOURCE[0]}") != null);

    // The whole call now conforms — which is the point: pi accepts it.
    try testing.expect(toolCallConformsToSchema(allocator, calls[0], pi_edit_tools_json_test));
}

test "hoistMisplacedRequiredParams: a buried `path` survives the Gemma 4 parse path end to end" {
    // The same class through the REAL parse chain (Gemma's call:name{...} form —
    // this session's model). The raw pre-parse bytes were not dumped (the server
    // was not at --log-level debug), so the tag wrapper here is reconstructed;
    // the ARGUMENT SHAPE it produces is the verbatim captured one asserted above,
    // which is what the hoist actually operates on.
    const allocator = testing.allocator;
    const raw = "<|tool_call>call:edit{edits:[{oldText:<|\"|>old line<|\"|>,newText:<|\"|>new line<|\"|>," ++
        "path:<|\"|>us_presidents/generate_site.sh<|\"|>}]}<tool_call|>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try hoistMisplacedRequiredParams(allocator, calls, pi_edit_tools_json_test);
    try coerceToolArgsToSchema(allocator, calls, pi_edit_tools_json_test);

    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const path = parsed.value.object.get("path") orelse return error.PathNotHoisted;
    try testing.expectEqualStrings("us_presidents/generate_site.sh", path.string);
    const edits = parsed.value.object.get("edits").?;
    try testing.expect(edits.array.items[0].object.get("path") == null);
    try testing.expectEqualStrings("old line", edits.array.items[0].object.get("oldText").?.string);
}

test "hoistMisplacedRequiredParams: a compliant call is byte-identical" {
    // The strong-model path: nothing may re-serialize when nothing was misplaced.
    const allocator = testing.allocator;
    const args =
        \\{"path":"a.js","edits":[{"oldText":"a","newText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: never touches a key the item schema DECLARES" {
    // Safety gate. A tool whose items legitimately carry their own `path` (a
    // multi-file edit) must never have it stripped out from under them — hoisting
    // there would DESTROY data, not repair it. The item schema declaring the key
    // is the proof that it belongs where it is.
    const allocator = testing.allocator;
    const multifile_schema =
        \\[{"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"path":{"type":"string"},"edits":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"oldText":{"type":"string"}},"required":["path","oldText"]}}},"required":["path","edits"]}}}]
    ;
    const args =
        \\{"edits":[{"path":"a.js","oldText":"a"},{"path":"b.js","oldText":"b"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, multifile_schema);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: items that DISAGREE on the value stay honest" {
    // Two items, two different paths, and the tool edits ONE file. There is no
    // correct hoist here — picking either would silently write the wrong file.
    // Leave it broken so the client's validation error reaches the model.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":"a.js"},{"oldText":"c","newText":"d","path":"b.js"}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "hoistMisplacedRequiredParams: a wrong-typed buried value is never hoisted" {
    // `path` is declared a string; an object spelled `path` inside the item is
    // not the missing param. Hoisting it would just move the schema violation.
    const allocator = testing.allocator;
    const args =
        \\{"edits":[{"oldText":"a","newText":"b","path":{"nested":"x"}}]}
    ;
    var calls = [_]ParsedToolCall{.{
        .name = try allocator.dupe(u8, "edit"),
        .arguments = try allocator.dupe(u8, args),
    }};
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
    }
    try hoistMisplacedRequiredParams(allocator, &calls, pi_edit_tools_json_test);
    try testing.expectEqualStrings(args, calls[0].arguments);
}

test "coerceToolArgsToSchema: quoted integer/number strings coerce to numeric" {
    // A model emitting `"limit":"5"` for an integer param → strict clients reject
    // "expected integer, got string". Coerce string→numeric per the schema.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"scale":{"type":"number"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a.txt\",\"limit\":\"5\",\"scale\":\"1.5\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 5), parsed.value.object.get("limit").?.integer);
    const scale = parsed.value.object.get("scale").?;
    try testing.expect(scale == .float or scale == .integer);
    // A string param is untouched.
    try testing.expectEqualStrings("a.txt", parsed.value.object.get("path").?.string);
}

test "coerceToolArgsToSchema: nullable union type [\"string\",\"null\"] coerces to the non-null member" {
    // Real schemas declare optional params as a type UNION. declaredJsonType
    // picks the first non-null member, so a boolean-spelled value under
    // ["boolean","null"] still coerces.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"flag":{"type":["boolean","null"]},"note":{"type":["string","null"]}}}}}]
    ;
    const raw = "<tool_call>\n<function=E>\n<parameter=flag>\nFalse\n</parameter>\n<parameter=note>\nhi\n</parameter>\n</function>\n</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    try testing.expect(parsed.value.object.get("flag").? == .bool);
    try testing.expectEqual(false, parsed.value.object.get("flag").?.bool);
    try testing.expectEqualStrings("hi", parsed.value.object.get("note").?.string);
}

test "coerceToolArgsToSchema: an explicit JSON null satisfies any typed param (not coerced away)" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"limit\":null}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    // null is a valid absent-value marker for any type — left byte-identical.
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: coercion is SHALLOW — values nested inside a param are untouched" {
    // Documented boundary: the layer coerces TOP-LEVEL params against the tool's
    // declared property types. It does NOT recurse into array/object element
    // properties (the tool schema's `items`/nested `properties` are the client's
    // to validate). A nested `"port":"8080"` stays a string — byte-identical.
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"W","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"W\",\"arguments\":{\"config\":{\"port\":\"8080\",\"tls\":\"false\"}}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: an unmodelled declared type leaves the value alone" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"E","parameters":{"type":"object","properties":{"x":{"type":"weird"}}}}}]
    ;
    const raw = "<tool_call>{\"name\":\"E\",\"arguments\":{\"x\":\"False\"}}</tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    const before = try allocator.dupe(u8, calls[0].arguments);
    defer allocator.free(before);
    try coerceToolArgsToSchema(allocator, calls, tools);
    try testing.expectEqualStrings(before, calls[0].arguments);
}

test "coerceToolArgsToSchema: anyOf, oneOf, allOf and a local $ref resolve to a concrete type" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"S","parameters":{"type":"object","properties":{
        \\"limit":{"anyOf":[{"type":"integer"},{"type":"null"}]},
        \\"ratio":{"oneOf":[{"type":"null"},{"type":"number"}]},
        \\"tags":{"allOf":[{"type":"array","items":{"type":"string"}}]},
        \\"cfg":{"$ref":"#/$defs/Cfg"},
        \\"deep":{"$ref":"#/definitions/Wrap"},
        \\"flag":{"anyOf":[{"type":"boolean"},{"type":"string"}]},
        \\"plain":{"type":"integer"},
        \\"loop":{"$ref":"#/$defs/Loop"}},
        \\"$defs":{"Cfg":{"type":"object"},"Loop":{"$ref":"#/$defs/Loop"}},
        \\"definitions":{"Wrap":{"anyOf":[{"$ref":"#/$defs/Cfg"},{"type":"null"}]}}}}}]
    ;
    const raw = "<tool_call>S<arg_key>limit</arg_key><arg_value>5</arg_value>" ++
        "<arg_key>ratio</arg_key><arg_value>0.5</arg_value>" ++
        "<arg_key>tags</arg_key><arg_value>[\"a\",\"b\"]</arg_value>" ++
        "<arg_key>cfg</arg_key><arg_value>{\"k\":1}</arg_value>" ++
        "<arg_key>deep</arg_key><arg_value>{\"k\":2}</arg_value>" ++
        "<arg_key>flag</arg_key><arg_value>no</arg_value>" ++
        "<arg_key>plain</arg_key><arg_value>7</arg_value>" ++
        "<arg_key>loop</arg_key><arg_value>5</arg_value></tool_call>";
    const calls = (try parseToolCalls(allocator, raw)).?;
    defer {
        for (calls) |tc| {
            allocator.free(tc.name);
            allocator.free(tc.arguments);
        }
        allocator.free(calls);
    }
    try coerceToolArgsToSchema(allocator, calls, tools);
    const parsed = try parseArgsObj(allocator, calls[0].arguments);
    defer parsed.deinit();
    const o = parsed.value.object;
    try testing.expectEqual(@as(i64, 5), o.get("limit").?.integer);
    try testing.expect(o.get("ratio").? == .float);
    try testing.expectEqual(@as(usize, 2), o.get("tags").?.array.items.len);
    try testing.expectEqual(@as(i64, 1), o.get("cfg").?.object.get("k").?.integer);
    try testing.expectEqual(@as(i64, 2), o.get("deep").?.object.get("k").?.integer);
    // A value the first branch cannot take but a later one matches is kept.
    try testing.expectEqualStrings("no", o.get("flag").?.string);
    try testing.expectEqual(@as(i64, 7), o.get("plain").?.integer);
    // A self-referencing $ref terminates and leaves the value alone.
    try testing.expect(o.get("loop") != null);
}

test "toolCallConformsToSchema: a value matching any anyOf branch conforms" {
    const allocator = testing.allocator;
    const tools =
        \\[{"type":"function","function":{"name":"S","parameters":{"type":"object","properties":{"v":{"anyOf":[{"type":"integer"},{"type":"string"}]}}}}}]
    ;
    const ok = ParsedToolCall{ .name = "S", .arguments = "{\"v\":\"abc\"}" };
    try testing.expect(toolCallConformsToSchema(allocator, ok, tools));
    const bad = ParsedToolCall{ .name = "S", .arguments = "{\"v\":[1]}" };
    try testing.expect(!toolCallConformsToSchema(allocator, bad, tools));
}

// ── DeepSeek-V4 native DSML tool calls ─────────────────────────────────

fn freeParsedCalls(calls: []ParsedToolCall) void {
    for (calls) |tc| {
        testing.allocator.free(tc.name);
        testing.allocator.free(tc.arguments);
    }
    testing.allocator.free(calls);
}

test "parseToolCalls dsml: canonical two-call block, string and JSON params" {
    const raw = "I'll check the weather.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"get_weather\">\n" ++
        "<｜DSML｜parameter name=\"city\" string=\"true\">San Francisco</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "<｜DSML｜invoke name=\"get_time\">\n" ++
        "<｜DSML｜parameter name=\"tz\" string=\"true\">PST</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n" ++
        "</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("get_weather", calls[0].name);
    try testing.expectEqualStrings("{\"city\":\"San Francisco\",\"days\":3}", calls[0].arguments);
    try testing.expectEqualStrings("get_time", calls[1].name);
    try testing.expectEqualStrings("{\"tz\":\"PST\"}", calls[1].arguments);
    try testing.expect(!calls[0].inferred);
}

test "parseToolCalls dsml: dropped invoke close before the next call still parses both" {
    // Delimiter-drop tolerance: the model omits </｜DSML｜invoke>
    // between back-to-back calls.
    const raw = "<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"first\">\n" ++
        "<｜DSML｜parameter name=\"a\" string=\"true\">x</｜DSML｜parameter>\n" ++
        "<｜DSML｜invoke name=\"second\">\n" ++
        "<｜DSML｜parameter name=\"b\" string=\"false\">true</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 2), calls.len);
    try testing.expectEqualStrings("first", calls[0].name);
    try testing.expectEqualStrings("{\"a\":\"x\"}", calls[0].arguments);
    try testing.expectEqualStrings("second", calls[1].name);
    try testing.expectEqualStrings("{\"b\":true}", calls[1].arguments);
}

test "parseToolCalls dsml: truncation salvages NAME + completed pairs, never fragments" {
    // Server-side cut mid-value: the unterminated pair is dropped (a half-
    // written file is worse than a re-issued write), completed pairs survive.
    const raw = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write_file\">\n" ++
        "<｜DSML｜parameter name=\"path\" string=\"true\">/tmp/a.txt</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"content\" string=\"true\">first half of a long fi";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("write_file", calls[0].name);
    try testing.expectEqualStrings("{\"path\":\"/tmp/a.txt\"}", calls[0].arguments);
}

test "parseToolCalls dsml: rich string value runs to the confirmed close" {
    // Newlines, quotes, braces, `<` inside a string value — value ends ONLY
    // at the closing parameter tag (the Gemma dropped-quote class).
    const raw = "<｜DSML｜invoke name=\"bash\">\n" ++
        "<｜DSML｜parameter name=\"cmd\" string=\"true\">grep -r \"a < b\" src/\necho {done}</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("grep -r \"a < b\" src/\necho {done}", parsed.value.object.get("cmd").?.string);
}

test "parseToolCalls dsml: duplicate keys dedup (last wins) and invalid JSON value degrades to string" {
    const raw = "<｜DSML｜invoke name=\"t\">\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">old</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"k\" string=\"true\">new</｜DSML｜parameter>\n" ++
        "<｜DSML｜parameter name=\"n\" string=\"false\">not[json</｜DSML｜parameter>\n" ++
        "</｜DSML｜invoke>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("new", parsed.value.object.get("k").?.string);
    try testing.expectEqualStrings("not[json", parsed.value.object.get("n").?.string);
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
}

test "parseToolCalls dsml: zero-arg call and thinking prefix" {
    const raw = "thinking about it</think>Sure.\n\n<｜DSML｜tool_calls>\n" ++
        "<｜DSML｜invoke name=\"list_files\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
    const calls = (try parseToolCalls(testing.allocator, raw)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("list_files", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
}

test "parseToolCalls: Claude-style bare <invoke> markup is an inferred call" {
    const allocator = testing.allocator;
    const text = "<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">cd /x && ls -la</parameter>\n<parameter name=\"timeout\">30</parameter>\n</invoke>\n</function_calls>";
    const calls = (try parseToolCalls(allocator, text)).?;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("bash", calls[0].name);
    // Not the model's own dialect: it must name a declared tool to survive.
    try testing.expect(calls[0].inferred);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("cd /x && ls -la", parsed.value.object.get("command").?.string);
    try testing.expectEqual(@as(i64, 30), parsed.value.object.get("timeout").?.integer);
    try testing.expect((try parseToolCalls(allocator, "I <invoked> it with <parameters> set.")) == null);
    // An unparsed block never reaches content; prose naming the tag stays.
    try testing.expectEqualStrings("Done.", trimLeakedToolMarkup("Done.\n<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"command\">ls"));
    const prose = "Claude's format wraps each call in <invoke name=\"x\"> tags.";
    try testing.expectEqualStrings(prose, trimLeakedToolMarkup(prose));
}


test "parseToolCalls: <tool_call>{JSON} truncated mid-string recovers NAME + {} (never a fragment)" {
    // A 4 KB edit call that hit EOS inside a string value. The object never
    // balances, so the balanced-snap arm declines; without a name salvage the
    // whole turn was dropped at the flush (empty `stop`, nothing for the client
    // to nudge on). Same rule as every other dialect: NAME + `{}`.
    const allocator = testing.allocator;
    const text = "Fixing it.\n<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"path\": \"a.ts\", \"oldText\": \"function foo() {\\n  return 1;";
    const calls = (try parseToolCalls(allocator, text)) orelse return error.NoCalls;
    defer freeParsedCalls(calls);
    try testing.expectEqual(@as(usize, 1), calls.len);
    try testing.expectEqualStrings("edit", calls[0].name);
    try testing.expectEqualStrings("{}", calls[0].arguments);
    // Reversed key order + whitespace variants still find the name.
    const text2 = "<tool_call>{ \"arguments\" : {\"name\": \"nope\", \"s\": \"a}b\"}, \"name\" : \"real\", \"extra\": \"unterm";
    const c2 = (try parseToolCalls(allocator, text2)) orelse return error.NoCalls;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("real", c2[0].name);
    // No name anywhere: still nothing (no invented call).
    try testing.expect((try parseToolCalls(allocator, "<tool_call>\n{\"arguments\": {\"path\": \"a")) == null);
}

test "parseToolCalls minicpm5: a param value keeps its own leading/trailing blank lines" {
    // Upstream #295 (stripHermesValueFraming) established the rule for the
    // Hermes dialect: the template frames a value with EXACTLY one newline per
    // side, so exactly one is what the parser may remove. Eating a RUN destroys
    // content the model meant to send - an old_string that must match a file
    // "exactly, including indentation" then edits at the wrong place (#294).
    //
    // NOTE the captured model output is value-adjacent, so this framing case is
    // NOT yet observed in the wild; it is the conservative handling for the
    // newline-framed layout rather than a claim that MiniCPM5 emits it. This pins it - the value here is
    // framed by one newline per side AND legitimately begins and ends with a
    // blank line of its own, which must survive.
    const text = "<function name=\"write\">\n<param name=\"content\">\n\nhi\n\n</param>\n</function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("\nhi\n", parsed.value.object.get("content").?.string);
}

test "parseToolCalls minicpm5: character data AFTER a CDATA section is not discarded" {
    // `<![CDATA[x]]>tail` used to return just `x`: the scan took the LAST
    // `]]>` anywhere in the probe, so anything following the section was
    // silently dropped. Silent truncation of a tool argument is the worst
    // failure mode here - the call still looks well-formed to the client.
    const text = "<function name=\"w\"><param name=\"c\"><![CDATA[x]]>tail</param></function>";
    const calls = (try parseToolCalls(testing.allocator, text)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(calls);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, calls[0].arguments, .{});
    defer parsed.deinit();
    // Not a CDATA-wrapped VALUE (the section does not end it), so it is text.
    try testing.expectEqualStrings("<![CDATA[x]]>tail", parsed.value.object.get("c").?.string);
}

test "minicpm5 COMBINED: parser and scrubber agree on the same bytes" {
    // The earlier defect was two individually-green tests that CONTRADICTED
    // each other: the parser called a string prose while the scrubber cut it as
    // markup. Neither test could see it because neither ran both layers. This
    // one does, so that shape cannot recur silently.
    //
    // Fixtures marked LIVE are verbatim raw emissions of
    // mlx-community/MiniCPM5-1B-OptiQ-4bit captured via SUSHI_RAW_DUMP_FILE.
    const Case = struct {
        raw: []const u8,
        is_call: bool, // parser must produce a call
        keeps_text: bool, // scrubber must leave the text intact
        args: ?[]const u8 = null, // when set, the exact arguments JSON
    };
    const cases = [_]Case{
        // LIVE: normal parameterised call - value-adjacent, no framing newlines.
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status</param></function>", .is_call = true, .keeps_text = false },
        // LIVE: zero-argument call, closed with an empty body.
        .{ .raw = "<function name=\"get_time\"></function>", .is_call = true, .keeps_text = false },
        // LIVE: truncated at max_tokens - a real call, ZERO completed params.
        // Salvages the NAME with empty args; the unterminated value never ships
        // (a fragment argument is worse than none - the call looks complete).
        .{ .raw = "<function name=\"shell\"><param name=\"command\">git status", .is_call = true, .keeps_text = false, .args = "{}" },
        // Prose naming a function: no <param, no close. Neither layer may act.
        .{ .raw = "Document the syntax as <function name=\"shell\"> before discussing parameters.", .is_call = false, .keeps_text = true },
        .{ .raw = "Wrap it in a <functional> block - just prose.", .is_call = false, .keeps_text = true },
        // Malformed (dropped quote) WITH structural evidence: not a call, but
        // it is markup, so it must be scrubbed rather than rendered.
        .{ .raw = "<function name=\"shell>\n  <param name=\"command\">pwd</param>\n</function>", .is_call = false, .keeps_text = false },
    };
    for (cases) |c| {
        const parsed = try parseToolCalls(testing.allocator, c.raw);
        if (parsed) |calls| {
            if (c.args) |want| try testing.expectEqualStrings(want, calls[0].arguments);
            freeParsedCalls(calls);
        }
        const got_call = parsed != null;
        testing.expectEqual(c.is_call, got_call) catch |e| {
            std.debug.print("parser disagreed on: '{s}' (got call={})\n", .{ c.raw, got_call });
            return e;
        };
        const kept = std.mem.eql(u8, std.mem.trimEnd(u8, c.raw, "\n\r\t "), trimLeakedToolMarkup(c.raw));
        testing.expectEqual(c.keeps_text, kept) catch |e| {
            std.debug.print("scrubber disagreed on: '{s}' -> '{s}'\n", .{ c.raw, trimLeakedToolMarkup(c.raw) });
            return e;
        };
        // THE invariant: text the parser calls prose is text the scrubber keeps.
        if (!got_call and !kept) {
            const still_markup = std.mem.indexOf(u8, c.raw, "<param") != null or
                std.mem.indexOf(u8, c.raw, "</function>") != null;
            testing.expect(still_markup) catch |e| {
                std.debug.print("CONTRADICTION: parser says prose, scrubber cut it: '{s}'\n", .{c.raw});
                return e;
            };
        }
    }
}

test "parseToolCalls minicpm5: an UNCLOSED named opener in prose is not a call" {
    // The truncation salvage recovers NAME + complete params when EOS cut a
    // real call. It cannot, on its own, tell that apart from a model WRITING
    // ABOUT the syntax - and an unclosed opener with no params was being
    // promoted into an executable `shell({})`. For a zero-arg tool that is an
    // unintended invocation; otherwise it replaces the answer with a bogus call.
    //
    // Rule: an unclosed function must carry structural evidence that a call was
    // under way - at minimum a `<param` opener. (It was briefly a COMPLETE
    // `<param>...</param>`; a real max_tokens capture has zero completed pairs,
    // so that discarded genuine calls.) A properly CLOSED zero-arg call is
    // still a call - the close tag is the model's own commitment.
    const prose = [_][]const u8{
        "Document the syntax as <function name=\"shell\"> before discussing parameters.",
        "You write <function name=\"get_time\"> to open a call.",
    };
    for (prose) |t| {
        if (try parseToolCalls(testing.allocator, t)) |calls| {
            defer freeParsedCalls(calls);
            std.debug.print("PROSE became a call: '{s}' -> {s}({s})\n", .{ t, calls[0].name, calls[0].arguments });
            return error.ProseBecameToolCall;
        }
    }

    // A CLOSED zero-arg call still parses.
    const closed = "<function name=\"get_time\"></function>";
    const c1 = (try parseToolCalls(testing.allocator, closed)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c1);
    try testing.expectEqualStrings("get_time", c1[0].name);
    try testing.expectEqualStrings("{}", c1[0].arguments);

    // A genuinely TRUNCATED call that got one complete param through still
    // salvages - that is the case the branch exists for.
    const cut = "<function name=\"shell\">\n<param name=\"command\">ls</param>\n<param name=\"cwd\">/tm";
    const c2 = (try parseToolCalls(testing.allocator, cut)) orelse return error.ExpectedToolCall;
    defer freeParsedCalls(c2);
    try testing.expectEqualStrings("shell", c2[0].name);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, c2[0].arguments, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("ls", parsed.value.object.get("command").?.string);
    try testing.expect(parsed.value.object.get("cwd") == null);
}
