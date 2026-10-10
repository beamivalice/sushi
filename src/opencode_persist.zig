//! Opt-in OpenCode global configuration persistence. Invalid input fails closed.
const std = @import("std");

/// JSONC normalization keeps string literals (including URLs and escaped quotes)
/// intact. Comments and trailing commas become whitespace before strict parsing.
fn normalize(a: std.mem.Allocator, input: []const u8) ![]u8 {
    const out = try a.dupe(u8, input);
    errdefer a.free(out);
    var i: usize = 0;
    var string = false;
    while (i < out.len) : (i += 1) {
        if (string) {
            if (out[i] == '\\') {
                i += 1;
            } else if (out[i] == '"') {
                string = false;
            }
            continue;
        }
        if (out[i] == '"') {
            string = true;
            continue;
        }
        if (out[i] == '/' and i + 1 < out.len) {
            if (out[i + 1] == '/') {
                while (i < out.len and out[i] != '\n') : (i += 1) out[i] = ' ';
            } else if (out[i + 1] == '*') {
                out[i] = ' ';
                out[i + 1] = ' ';
                i += 2;
                while (i + 1 < out.len and !(out[i] == '*' and out[i + 1] == '/')) : (i += 1) out[i] = ' ';
                if (i + 1 >= out.len) return error.InvalidConfig;
                out[i] = ' ';
                out[i + 1] = ' ';
                i += 1;
            }
        }
    }
    i = 0;
    string = false;
    while (i < out.len) : (i += 1) {
        if (string) {
            if (out[i] == '\\') {
                i += 1;
            } else if (out[i] == '"') {
                string = false;
            }
            continue;
        }
        if (out[i] == '"') {
            string = true;
            continue;
        }
        if (out[i] == ',') {
            var j = i + 1;
            while (j < out.len and std.ascii.isWhitespace(out[j])) : (j += 1) {}
            if (j < out.len and (out[j] == '}' or out[j] == ']')) out[i] = ' ';
        }
    }
    return out;
}

fn mergeObjects(a: std.mem.Allocator, dst: *std.json.Value, src: std.json.Value) !void {
    if (dst.* != .object or src != .object) return error.InvalidConfig;
    var it = src.object.iterator();
    while (it.next()) |entry| {
        if (dst.object.getPtr(entry.key_ptr.*)) |old| {
            // Replace the server's supported variants, rather than retaining
            // stale unsupported effort names from an earlier model version.
            if (old.* == .object and entry.value_ptr.* == .object and !std.mem.eql(u8, entry.key_ptr.*, "variants")) {
                try mergeObjects(a, old, entry.value_ptr.*);
                continue;
            }
            // Do not erase a malformed container and silently lose user data.
            if (entry.value_ptr.* == .object and old.* != .object) return error.InvalidConfig;
        }
        try dst.object.put(a, entry.key_ptr.*, entry.value_ptr.*);
    }
}

pub fn merged(a: std.mem.Allocator, existing: []const u8, generated: []const u8, entries: anytype) ![]u8 {
    const clean = try normalize(a, existing);
    defer a.free(clean);
    var old = std.json.parseFromSlice(std.json.Value, a, clean, .{}) catch return error.InvalidConfig;
    defer old.deinit();
    var fresh = try std.json.parseFromSlice(std.json.Value, a, generated, .{});
    defer fresh.deinit();
    if (old.value != .object) return error.InvalidConfig;
    // An unloaded model advertises its architectural maximum. Keep any lower
    // configured limit until a loaded row supplies its actual serving window.
    const old_provider = if (old.value.object.get("provider")) |p| (if (p == .object) p.object.get("sushi") else null) else null;
    if (old_provider) |p| {
        if (p == .object) {
            if (p.object.get("models")) |models| {
                if (models == .object) {
                    const new_models = fresh.value.object.getPtr("provider").?.object.getPtr("sushi").?.object.getPtr("models").?;
                    for (entries) |e| {
                        if (e.loaded) continue;
                        const m = models.object.get(e.id) orelse continue;
                        if (m != .object) continue;
                        const limit = m.object.get("limit") orelse continue;
                        if (limit != .object) continue;
                        const new_limit = new_models.object.getPtr(e.id).?.object.getPtr("limit").?;
                        for ([_][]const u8{ "context", "output" }) |key| {
                            const prev = limit.object.get(key) orelse continue;
                            const next = new_limit.object.getPtr(key) orelse continue;
                            if (prev == .integer and next.* == .integer and prev.integer > 0 and prev.integer < next.integer) next.* = prev;
                        }
                    }
                }
            }
        }
    }
    try mergeObjects(old.arena.allocator(), &old.value, fresh.value);
    return std.json.Stringify.valueAlloc(a, old.value, .{ .whitespace = .indent_2 });
}

pub fn save(a: std.mem.Allocator, io: std.Io, directory: []const u8, generated: []const u8, entries: anytype) !void {
    try std.Io.Dir.cwd().createDirPath(io, directory);
    var dir = try std.Io.Dir.openDirAbsolute(io, directory, .{});
    defer dir.close(io);
    // Coordinate Sushi writers, without truncating the existing configuration.
    const lock_name = ".sushi-persist.lock";
    const lock = try dir.createFile(io, lock_name, .{ .exclusive = true, .permissions = .fromMode(0o600) });
    defer {
        lock.close(io);
        dir.deleteFile(io, lock_name) catch {};
    }
    const jsonc = dir.readFileAlloc(io, "opencode.jsonc", a, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    defer if (jsonc) |v| a.free(v);
    const json = dir.readFileAlloc(io, "opencode.json", a, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    defer if (json) |v| a.free(v);
    // Both files may participate in OpenCode's precedence. Refuse ambiguity.
    if (json != null and jsonc != null) return error.AmbiguousConfigFiles;
    const name: []const u8 = if (jsonc != null) "opencode.jsonc" else "opencode.json";
    const original = jsonc orelse json;
    const result = try merged(a, original orelse "{}", generated, entries);
    defer a.free(result);
    if (original) |raw| {
        if (std.mem.eql(u8, std.mem.trim(u8, raw, " \t\r\n"), result)) return;
        var index: usize = 0;
        while (true) : (index += 1) {
            const backup_name = try std.fmt.allocPrint(a, "{s}.sushi-backup-{d}", .{ name, index });
            defer a.free(backup_name);
            const backup = dir.createFile(io, backup_name, .{ .exclusive = true, .permissions = .fromMode(0o600) }) catch |err| switch (err) {
                error.PathAlreadyExists => continue,
                else => return err,
            };
            defer backup.close(io);
            try backup.writeStreamingAll(io, raw);
            try backup.sync(io);
            break;
        }
    }
    const tmp_name = ".sushi-persist.tmp";
    const tmp = try dir.createFile(io, tmp_name, .{ .exclusive = true, .permissions = .fromMode(0o600) });
    defer {
        tmp.close(io);
        dir.deleteFile(io, tmp_name) catch {};
    }
    try tmp.writeStreamingAll(io, result);
    try tmp.sync(io);
    // Refuse a concurrent editor's change rather than overwriting it.
    const current = dir.readFileAlloc(io, name, a, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => return err,
    };
    defer if (current) |v| a.free(v);
    if ((original == null) != (current == null)) return error.ConfigChanged;
    if (original) |raw| if (!std.mem.eql(u8, raw, current.?)) return error.ConfigChanged;
    try dir.rename(tmp_name, dir, name, io);
}

test "opencode persistence JSONC merge preserves unrelated settings and omitted effort" {
    const a = std.testing.allocator;
    const entries = [_]struct { id: []const u8, loaded: bool }{.{ .id = "qwen", .loaded = false }};
    const raw = "{ // comment\n\"plugin\":[\"https://host/a//b\",],\"provider\":{\"other\":{},\"sushi\":{\"models\":{\"qwen\":{\"options\":{\"reasoningEffort\":\"xhigh\",\"temperature\":0.2},\"limit\":{\"context\":248000},\"variants\":{\"bad\":{}}}}}},}";
    const generated = "{\"model\":\"sushi/qwen\",\"provider\":{\"sushi\":{\"models\":{\"qwen\":{\"limit\":{\"context\":1048576},\"variants\":{\"xhigh\":{}}}}}}}";
    const out = try merged(a, raw, generated, &entries);
    defer a.free(out);
    const p = try std.json.parseFromSlice(std.json.Value, a, out, .{});
    defer p.deinit();
    const providers = p.value.object.get("provider").?.object;
    try std.testing.expect(providers.contains("other"));
    const model = providers.get("sushi").?.object.get("models").?.object.get("qwen").?.object;
    try std.testing.expectEqualStrings("xhigh", model.get("options").?.object.get("reasoningEffort").?.string);
    try std.testing.expectEqual(@as(i64, 248000), model.get("limit").?.object.get("context").?.integer);
    try std.testing.expect(!model.get("variants").?.object.contains("bad"));
    try std.testing.expectEqualStrings("https://host/a//b", p.value.object.get("plugin").?.array.items[0].string);
    try std.testing.expectError(error.InvalidConfig, merged(a, "{/*", generated, &entries));
    try std.testing.expectError(error.InvalidConfig, merged(a, "[]", generated, &entries));
    try std.testing.expectError(error.InvalidConfig, merged(a, "{\"provider\":false}", generated, &entries));
}
