//! Test-only: a chat template's context written as the JSON text the server once handed the engine. The oracle keys
//! its cases by that text and feeds it to Python and jinja.cpp, and the serializer tests pin it; the served binary
//! never compiles this file (the check below fails any build that does).
const std = @import("std");
const builtin = @import("builtin");
const chat = @import("chat.zig");
const template_engine = @import("template/template.zig");
const Value = template_engine.Value;

comptime {
    if (!builtin.is_test) @compileError("chat_json_test.zig is compiled into test binaries only");
}

/// `v` as compact JSON, its keys in field order, a null optional left out, a type with `templateText()` as that text.
pub fn write(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), v: anytype) !void {
    const T = @TypeOf(v);
    if (T == Value) return writeValue(gpa, buf, v);
    if (T == []const u8) return chat.appendJsonString(gpa, buf, v);
    if (T == template_engine.Str) return chat.appendJsonString(gpa, buf, v.bytes);
    if (comptime (@typeInfo(T) == .@"struct" or @typeInfo(T) == .@"union") and @hasDecl(T, "templateText")) return buf.appendSlice(gpa, v.templateText());
    switch (@typeInfo(T)) {
        .bool => try buf.appendSlice(gpa, if (v) "true" else "false"),
        .int, .comptime_int => try buf.print(gpa, "{d}", .{v}),
        .optional => if (v) |x| try write(gpa, buf, x) else try buf.appendSlice(gpa, "null"),
        .@"enum" => try chat.appendJsonString(gpa, buf, @tagName(v)),
        .void => try buf.appendSlice(gpa, "null"),
        .pointer => |p| switch (p.size) {
            .one => try write(gpa, buf, v.*),
            .slice => {
                try buf.append(gpa, '[');
                for (v, 0..) |item, i| {
                    if (i > 0) try buf.append(gpa, ',');
                    try write(gpa, buf, item);
                }
                try buf.append(gpa, ']');
            },
            else => @compileError("no JSON for " ++ @typeName(T)),
        },
        .@"union" => switch (v) {
            inline else => |payload| try write(gpa, buf, payload),
        },
        .@"struct" => |st| {
            try buf.append(gpa, '{');
            var first = true;
            inline for (st.field_names, st.field_types) |name, F| {
                const field = @field(v, name);
                if (!(@typeInfo(F) == .optional and field == null)) {
                    if (!first) try buf.append(gpa, ',');
                    first = false;
                    try chat.appendJsonString(gpa, buf, name);
                    try buf.append(gpa, ':');
                    try write(gpa, buf, field);
                }
            }
            try buf.append(gpa, '}');
        },
        else => @compileError("no JSON for " ++ @typeName(T)),
    }
}

/// A value read from JSON, written back: a float keeps a mark of its kind so it reads back as a float.
pub fn writeValue(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), v: Value) !void {
    switch (v) {
        .none => try buf.appendSlice(gpa, "null"),
        .boolean => |b| try buf.appendSlice(gpa, if (b) "true" else "false"),
        .int => |i| try buf.print(gpa, "{d}", .{i}),
        .bigint => |t| try buf.appendSlice(gpa, t),
        .float => |f| {
            const start = buf.items.len;
            try buf.print(gpa, "{d}", .{f});
            if (std.mem.indexOfAny(u8, buf.items[start..], ".en") == null) try buf.appendSlice(gpa, ".0");
        },
        .string, .markup => |s| try chat.appendJsonString(gpa, buf, s.bytes),
        .list => |l| {
            try buf.append(gpa, '[');
            for (l.items.items, 0..) |item, i| {
                if (i > 0) try buf.append(gpa, ',');
                try writeValue(gpa, buf, item);
            }
            try buf.append(gpa, ']');
        },
        .dict => |d| {
            try buf.append(gpa, '{');
            for (d.entries.items, 0..) |e, i| {
                if (i > 0) try buf.append(gpa, ',');
                try writeValue(gpa, buf, e.key);
                try buf.append(gpa, ':');
                try writeValue(gpa, buf, e.value);
            }
            try buf.append(gpa, '}');
        },
        else => return error.NotJson,
    }
}

/// `v` as an owned JSON text.
pub fn alloc(gpa: std.mem.Allocator, v: anytype) ![]u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(gpa);
    try write(gpa, &buf, v);
    return buf.toOwnedSlice(gpa);
}
