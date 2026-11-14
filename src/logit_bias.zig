const std = @import("std");

pub const Scope = enum { reasoning, answer, all };
pub const Bias = struct { id: u32, delta: f32, scope: Scope = .all };
pub const Loaded = struct {
    biases: []Bias,
    entries: usize = 0,
    skipped: usize = 0,
};

fn scopeValue(name: []const u8) !Scope {
    if (name.len == 0) return .all;
    return std.meta.stringToEnum(Scope, name) orelse error.LogitBiasInvalidScope;
}

fn deltaValue(v: f64) !f32 {
    if (!std.math.isFinite(v) or @abs(v) > 100) return error.LogitBiasInvalidDelta;
    return @floatCast(v);
}

fn jsonDelta(value: std.json.Value) !f32 {
    return deltaValue(switch (value) {
        .integer => |i| @as(f64, @floatFromInt(i)),
        .float => |f| f,
        else => return error.LogitBiasInvalidDelta,
    });
}

fn appendId(a: std.mem.Allocator, out: *std.ArrayList(Bias), id: u32, delta: f32, scope: Scope, vocab: usize) !void {
    if (id >= vocab) return error.LogitBiasInvalidId;
    try out.append(a, .{ .id = id, .delta = delta, .scope = scope });
}

fn add(a: std.mem.Allocator, out: *std.ArrayList(Bias), kind: []const u8, target: []const u8, delta: f32, scope: Scope, tok: anytype, vocab: usize, skipped: *usize) !void {
    if (std.mem.eql(u8, kind, "id")) {
        const id = std.fmt.parseInt(u32, target, 10) catch return error.LogitBiasInvalidId;
        return appendId(a, out, id, delta, scope, vocab);
    }
    if (std.mem.eql(u8, kind, "token")) {
        const id = tok.tokenId(target) orelse return error.LogitBiasUnknownToken;
        return appendId(a, out, id, delta, scope, vocab);
    }
    if (!std.mem.eql(u8, kind, "word")) return error.LogitBiasUnknownKind;
    if (target.len == 0) return error.LogitBiasInvalidTarget;
    const start = out.items.len;
    const buf = try a.alloc(u8, target.len + 1);
    defer a.free(buf);
    for ([_]usize{ 0, 1 }) |lead| {
        for ([_]bool{ false, true }) |cap| {
            buf[0] = ' ';
            @memcpy(buf[lead..][0..target.len], target);
            buf[lead] = if (cap) std.ascii.toUpper(buf[lead]) else std.ascii.toLower(buf[lead]);
            const ids = try tok.encode(a, buf[0 .. target.len + lead]);
            defer a.free(ids);
            if (ids.len != 1) {
                skipped.* += 1;
                continue;
            }
            var duplicate = false;
            for (out.items[start..]) |prior| if (prior.id == ids[0]) {
                duplicate = true;
                break;
            };
            if (!duplicate) try appendId(a, out, ids[0], delta, scope, vocab);
        }
    }
}

fn csvCell(a: std.mem.Allocator, body: []const u8, pos: *usize) ![]const u8 {
    var out = std.ArrayList(u8).empty;
    if (pos.* < body.len and body[pos.*] == '"') {
        pos.* += 1;
        while (pos.* < body.len) {
            const ch = body[pos.*];
            pos.* += 1;
            if (ch == '"') {
                if (pos.* < body.len and body[pos.*] == '"') {
                    pos.* += 1;
                } else {
                    if (pos.* < body.len and body[pos.*] != ',' and body[pos.*] != '\r' and body[pos.*] != '\n') return error.LogitBiasInvalidCsv;
                    return out.toOwnedSlice(a);
                }
            }
            try out.append(a, ch);
        }
        return error.LogitBiasInvalidCsv;
    }
    const begin = pos.*;
    while (pos.* < body.len and body[pos.*] != ',' and body[pos.*] != '\r' and body[pos.*] != '\n') : (pos.* += 1) {
        if (body[pos.*] == '"') return error.LogitBiasInvalidCsv;
    }
    return body[begin..pos.*];
}

pub fn parse(allocator: std.mem.Allocator, body: []const u8, extension: []const u8, tok: anytype, vocab: usize) !Loaded {
    var out = std.ArrayList(Bias).empty;
    errdefer out.deinit(allocator);
    var result: Loaded = .{ .biases = undefined };
    if (std.ascii.eqlIgnoreCase(extension, ".json")) {
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch return error.LogitBiasInvalidJson;
        defer parsed.deinit();
        if (parsed.value != .object) return error.LogitBiasInvalidJson;
        const entries = parsed.value.object.get("entries") orelse return error.LogitBiasInvalidJson;
        if (entries != .array) return error.LogitBiasInvalidJson;
        for (entries.array.items) |entry| {
            if (entry != .object) return error.LogitBiasInvalidEntry;
            const obj = entry.object;
            var kind: ?[]const u8 = null;
            var value: std.json.Value = undefined;
            for ([_][]const u8{ "id", "token", "word" }) |key| {
                if (obj.get(key)) |v| {
                    if (kind != null) return error.LogitBiasInvalidTarget;
                    kind = key;
                    value = v;
                }
            }
            const k = kind orelse return error.LogitBiasUnknownKind;
            const delta = try jsonDelta(obj.get("delta") orelse return error.LogitBiasInvalidDelta);
            const scope = if (obj.get("scope")) |v| switch (v) {
                .string => |name| if (name.len == 0) return error.LogitBiasInvalidScope else try scopeValue(name),
                else => return error.LogitBiasInvalidScope,
            } else .all;
            if (std.mem.eql(u8, k, "id")) {
                if (value != .integer or value.integer < 0 or value.integer >= vocab) return error.LogitBiasInvalidId;
                try appendId(allocator, &out, @intCast(value.integer), delta, scope, vocab);
            } else {
                if (value != .string) return error.LogitBiasInvalidTarget;
                try add(allocator, &out, k, value.string, delta, scope, tok, vocab, &result.skipped);
            }
            result.entries += 1;
        }
    } else if (std.ascii.eqlIgnoreCase(extension, ".csv")) {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        var pos: usize = 0;
        var row: usize = 0;
        while (pos < body.len) {
            var cells: [4][]const u8 = undefined;
            for (&cells, 0..) |*cell, i| {
                cell.* = try csvCell(arena.allocator(), body, &pos);
                if (i < 3) {
                    if (pos >= body.len or body[pos] != ',') return error.LogitBiasInvalidCsv;
                    pos += 1;
                }
            }
            if (pos < body.len and body[pos] == '\r') pos += 1;
            if (pos < body.len) {
                if (body[pos] != '\n') return error.LogitBiasInvalidCsv;
                pos += 1;
            }
            if (row == 0) {
                for (cells, [_][]const u8{ "kind", "target", "delta", "scope" }) |got, want| if (!std.mem.eql(u8, got, want)) return error.LogitBiasInvalidCsv;
            } else {
                const delta = try deltaValue(std.fmt.parseFloat(f64, cells[2]) catch return error.LogitBiasInvalidDelta);
                try add(allocator, &out, cells[0], cells[1], delta, try scopeValue(cells[3]), tok, vocab, &result.skipped);
                result.entries += 1;
            }
            row += 1;
        }
        if (row == 0) return error.LogitBiasInvalidCsv;
    } else return error.LogitBiasUnsupportedFormat;
    result.biases = try out.toOwnedSlice(allocator);
    return result;
}

const FakeTokenizer = struct {
    pub fn encode(_: @This(), a: std.mem.Allocator, word: []const u8) ![]u32 {
        const words = [_][]const u8{ "wait", "Wait", " wait", " Wait" };
        for (words, 0..) |w, i| if (std.mem.eql(u8, w, word)) return a.dupe(u32, &.{@intCast(i)});
        return a.dupe(u32, &.{ 0, 1 });
    }
    pub fn tokenId(_: @This(), word: []const u8) ?u32 {
        return if (std.mem.eql(u8, word, "literal")) 4 else null;
    }
};

test "logit bias CPU: JSON and CSV expand targets and scopes identically" {
    const a = std.testing.allocator;
    const json = try parse(a, "{\"entries\":[{\"word\":\"wait\",\"delta\":-1,\"scope\":\"reasoning\"},{\"token\":\"literal\",\"delta\":2,\"scope\":\"answer\"},{\"id\":5,\"delta\":3},{\"word\":\"split\",\"delta\":1}]}", ".json", FakeTokenizer{}, 6);
    defer a.free(json.biases);
    const csv = try parse(a, "kind,target,delta,scope\r\nword,wait,-1,reasoning\r\ntoken,\"literal\",2,answer\r\nid,5,3,\r\nword,split,1,all\r\n", ".csv", FakeTokenizer{}, 6);
    defer a.free(csv.biases);
    try std.testing.expectEqual(@as(usize, 4), json.entries);
    try std.testing.expectEqual(@as(usize, 6), json.biases.len);
    try std.testing.expectEqual(@as(usize, 4), json.skipped);
    try std.testing.expectEqualDeep(json, csv);
}

test "logit bias CPU: malformed entries fail by name" {
    const a = std.testing.allocator;
    const cases = .{
        .{ "{\"entries\":[{\"kind\":\"bad\",\"delta\":1}]}", error.LogitBiasUnknownKind },
        .{ "{\"entries\":[{\"id\":6,\"delta\":1}]}", error.LogitBiasInvalidId },
        .{ "{\"entries\":[{\"id\":-1,\"delta\":1}]}", error.LogitBiasInvalidId },
        .{ "{\"entries\":[{\"id\":0,\"delta\":101}]}", error.LogitBiasInvalidDelta },
        .{ "{\"entries\":[{\"id\":0,\"delta\":1,\"scope\":\"other\"}]}", error.LogitBiasInvalidScope },
        .{ "{\"entries\":[{\"token\":\"missing\",\"delta\":1}]}", error.LogitBiasUnknownToken },
    };
    inline for (cases) |c| try std.testing.expectError(c[1], parse(a, c[0], ".json", FakeTokenizer{}, 6));
    try std.testing.expectError(error.LogitBiasInvalidDelta, parse(a, "kind,target,delta,scope\nid,0,nan,all", ".csv", FakeTokenizer{}, 6));
}

pub fn request(a: std.mem.Allocator, value: ?std.json.Value, vocab: usize) ![]Bias {
    const v = value orelse return a.alloc(Bias, 0);
    // OpenAI clients send `null` (and some SDKs `[]`) for "no bias".
    if (v == .null or (v == .array and v.array.items.len == 0)) return a.alloc(Bias, 0);
    if (v != .object) return error.LogitBiasInvalidRequest;
    var out = std.ArrayList(Bias).empty;
    errdefer out.deinit(a);
    var it = v.object.iterator();
    while (it.next()) |kv| {
        const id = std.fmt.parseInt(u32, kv.key_ptr.*, 10) catch return error.LogitBiasInvalidId;
        try appendId(a, &out, id, try jsonDelta(kv.value_ptr.*), .all, vocab);
    }
    return out.toOwnedSlice(a);
}

pub fn fillDeltas(out: []f32, biases: []const Bias, inside: bool) void {
    @memset(out, 0);
    addDeltas(out, biases, inside);
}

test "logit bias CPU: request map validates ids and bias range" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"1\":-100,\"2\":100}", .{});
    defer parsed.deinit();
    const biases = try request(a, parsed.value, 3);
    defer a.free(biases);
    try std.testing.expectEqual(@as(usize, 2), biases.len);
    try std.testing.expectEqual(@as(f32, -100), biases[0].delta);
    try std.testing.expectEqual(Scope.all, biases[1].scope);
    try std.testing.expectError(error.LogitBiasInvalidId, request(a, parsed.value, 2));
    const invalid = try std.json.parseFromSlice(std.json.Value, a, "{\"1\":101}", .{});
    defer invalid.deinit();
    try std.testing.expectError(error.LogitBiasInvalidDelta, request(a, invalid.value, 3));
}

test "logit bias CPU: request treats null and an empty array as no bias" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "null", "[]", "{}" }) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        const biases = try request(a, parsed.value, 3);
        defer a.free(biases);
        try std.testing.expectEqual(@as(usize, 0), biases.len);
    }
    for ([_][]const u8{ "[1]", "\"x\"", "3" }) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer parsed.deinit();
        try std.testing.expectError(error.LogitBiasInvalidRequest, request(a, parsed.value, 3));
    }
}

test "logit bias CPU: scopes and overlapping entries add rewards and penalties" {
    const biases = [_]Bias{
        .{ .id = 0, .delta = -2, .scope = .reasoning },
        .{ .id = 0, .delta = 3, .scope = .all },
        .{ .id = 1, .delta = 5, .scope = .answer },
        .{ .id = 1, .delta = -1, .scope = .all },
    };
    var out: [3]f32 = undefined;
    fillDeltas(&out, &biases, true);
    try std.testing.expectEqualSlices(f32, &.{ 1, -1, 0 }, &out);
    fillDeltas(&out, &biases, false);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4, 0 }, &out);
}

pub fn addDeltas(out: []f32, biases: []const Bias, inside: bool) void {
    for (biases) |bias| {
        if (bias.scope == .reasoning and !inside) continue;
        if (bias.scope == .answer and inside) continue;
        if (bias.id < out.len) out[bias.id] += bias.delta;
    }
}

pub const FilePath = struct {
    bytes: [std.fs.max_path_bytes]u8 = undefined,
    len: usize,

    pub fn from(s: []const u8) ?FilePath {
        if (s.len == 0 or s.len > std.fs.max_path_bytes) return null;
        var result: FilePath = .{ .len = s.len };
        @memcpy(result.bytes[0..s.len], s);
        return result;
    }

    pub fn slice(self: *const FilePath) []const u8 {
        return self.bytes[0..self.len];
    }
};

test "logit bias CPU: capitalized words expand lowercase and uppercase spellings" {
    const a = std.testing.allocator;
    const parsed = try parse(a, "{\"entries\":[{\"word\":\"Wait\",\"delta\":-1}]}", ".json", FakeTokenizer{}, 6);
    defer a.free(parsed.biases);
    try std.testing.expectEqual(@as(usize, 4), parsed.biases.len);
}
