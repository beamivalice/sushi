//! `GET/POST /v1/settings`: the settings file (`~/.sushi/model-settings.json`) as the web UI edits it. Server-wide keys
//! live under its `"server"` object, a model's under its absolute path. A launch flag outranks the file, so a key
//! a flag pins is reported locked; the file still takes the write, for the next start without that flag.
const std = @import("std");
const log = @import("log.zig");
const model_settings = @import("model_settings.zig");
const model_registry_mod = @import("model_registry.zig");
const disk_placement = @import("disk_placement.zig");
const http_conn = @import("http_conn.zig");
const server = @import("server.zig");

const Edit = model_settings.Edit;

pub const ModelRow = struct { id: []const u8, path: []const u8, state: []const u8, bytes: ?u64 };

pub const ServerView = struct {
    default_models_dir: []const u8,
    boot_roots: []const []const u8,
    cache_budget_bytes: u64,
};

fn writeJson(w: *std.Io.Writer, v: anytype) !void {
    try std.json.Stringify.value(v, .{}, w);
}

fn writeSpec(w: *std.Io.Writer, spec: model_settings.KeySpec) !void {
    try w.print("{{\"name\":\"{s}\",\"type\":\"{s}\"", .{ spec.name, @tagName(spec.kind) });
    switch (spec.kind) {
        .int, .number => try w.print(",\"min\":{d},\"max\":{d}", .{ spec.min, spec.max }),
        .choice => if (spec.choices.len > 0) {
            try w.writeAll(",\"choices\":");
            try writeJson(w, spec.choices);
        } else try w.writeAll(",\"path\":true"),
        .bool => {},
    }
    try w.writeAll("}");
}

/// One key's state: the file's value (if the loader would read it), the flag pinning it, and which wins.
fn writeKey(w: *std.Io.Writer, spec: model_settings.KeySpec, file_value: ?std.json.Value, lock: ?*const model_settings.Lock) !void {
    const valid = if (file_value) |v| (if (model_settings.validate(spec, v)) true else |_| false) else false;
    try w.writeAll("{\"file\":");
    if (valid) try writeJson(w, file_value.?) else try w.writeAll("null");
    const pinned = if (lock) |l| l.set else false;
    try w.writeAll(",\"flag\":");
    if (pinned) try writeJson(w, lock.?.text()) else try w.writeAll("null");
    try w.print(",\"source\":\"{s}\",\"locked\":{}}}", .{ if (pinned) "flag" else if (valid) "file" else "default", pinned });
}

pub fn render(
    allocator: std.mem.Allocator,
    file_path: []const u8,
    file: *const model_settings.Settings,
    rows: []const ModelRow,
    locks: *const model_settings.LaunchLocks,
    view: ServerView,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"file\":");
    try writeJson(w, file_path);
    try w.writeAll(",\"model_keys\":[");
    for (model_settings.model_keys, 0..) |k, i| {
        if (i > 0) try w.writeAll(",");
        try writeSpec(w, k);
    }
    try w.writeAll("],\"applies\":\"next load of the model\",\"server\":{");

    const srv = model_settings.server(file);
    try w.writeAll("\"models_dir\":{\"default\":");
    try writeJson(w, view.default_models_dir);
    try w.writeAll(",\"boot_roots\":");
    try writeJson(w, view.boot_roots);
    try w.writeAll(",");
    const dir_pinned = locks.models_dir.set;
    try w.writeAll("\"file\":");
    try writeJson(w, srv.models_dir);
    try w.writeAll(",\"flag\":");
    if (dir_pinned) try writeJson(w, locks.models_dir.text()) else try w.writeAll("null");
    try w.print(",\"source\":\"{s}\",\"locked\":{},\"applies\":\"models are rescanned at once; the next start discovers from it\"}}", .{
        if (dir_pinned) "flag" else if (srv.models_dir != null) "file" else "default", dir_pinned,
    });

    const cache_pinned = locks.prefix_cache_disk_gb.set;
    try w.writeAll(",\"prefix_cache_disk_gb\":{\"file\":");
    try writeJson(w, srv.prefix_cache_disk_gb);
    try w.writeAll(",\"flag\":");
    if (cache_pinned) try writeJson(w, locks.prefix_cache_disk_gb.text()) else try w.writeAll("null");
    try w.print(",\"source\":\"{s}\",\"locked\":{},\"effective_bytes\":{d},\"applies\":\"a running tier takes a new size at once and evicts down to it; 0 (off) and removing the key apply at the next load\"}}}}", .{
        if (cache_pinned) "flag" else if (srv.prefix_cache_disk_gb != null) "file" else "default", cache_pinned, view.cache_budget_bytes,
    });

    try w.writeAll(",\"models\":[");
    for (rows, 0..) |row, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"id\":");
        try writeJson(w, row.id);
        try w.writeAll(",\"path\":");
        try writeJson(w, row.path);
        try w.print(",\"state\":\"{s}\",\"bytes_on_disk\":", .{row.state});
        try writeJson(w, row.bytes);
        try w.writeAll(",\"settings\":{");
        const obj = model_settings.modelObject(file, row.path);
        for (model_settings.model_keys, 0..) |k, ki| {
            if (ki > 0) try w.writeAll(",");
            try w.print("\"{s}\":", .{k.name});
            try writeKey(w, k, if (obj) |o| o.get(k.name) else null, &locks.models[ki]);
        }
        try w.writeAll("}}");
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}

pub const Rejection = struct { status: u16, err_type: []const u8, message: []const u8 };

pub const Plan = struct {
    server_edits: []const Edit = &.{},
    model_id: ?[]const u8 = null,
    model_edits: []const Edit = &.{},
};

pub const Parsed = union(enum) { ok: Plan, rejected: Rejection };

fn reject(arena: std.mem.Allocator, status: u16, err_type: []const u8, comptime fmt: []const u8, args: anytype) Parsed {
    const msg = std.fmt.allocPrint(arena, fmt, args) catch "invalid request";
    return .{ .rejected = .{ .status = status, .err_type = err_type, .message = msg } };
}

fn collect(arena: std.mem.Allocator, specs: []const model_settings.KeySpec, obj: std.json.ObjectMap, edits: *std.ArrayList(Edit)) ?Parsed {
    var it = obj.iterator();
    while (it.next()) |kv| {
        const spec = model_settings.findKey(specs, kv.key_ptr.*) orelse
            return reject(arena, 400, "unknown_setting", "\"{s}\" is not a setting this server has", .{kv.key_ptr.*});
        if (kv.value_ptr.* == .null) {
            edits.append(arena, .{ .key = spec.name, .value = null }) catch return reject(arena, 500, "internal_error", "out of memory", .{});
            continue;
        }
        model_settings.validate(spec, kv.value_ptr.*) catch
            return reject(arena, 400, "invalid_setting", "\"{s}\" does not take that value", .{spec.name});
        edits.append(arena, .{ .key = spec.name, .value = kv.value_ptr.* }) catch return reject(arena, 500, "internal_error", "out of memory", .{});
    }
    return null;
}

/// `{"server": {key: value|null}, "model": "<id>", "set": {key: value|null}}`; every key is checked before anything is written.
pub fn parseRequest(arena: std.mem.Allocator, body: []const u8) Parsed {
    const tree = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch
        return reject(arena, 400, "invalid_request_error", "Body must be JSON", .{});
    if (tree != .object) return reject(arena, 400, "invalid_request_error", "Body must be a JSON object", .{});
    var plan: Plan = .{};
    var it = tree.object.iterator();
    var set_obj: ?std.json.ObjectMap = null;
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        const v = kv.value_ptr.*;
        if (std.mem.eql(u8, k, "server")) {
            if (v != .object) return reject(arena, 400, "invalid_request_error", "\"server\" must be an object", .{});
            var edits = std.ArrayList(Edit).empty;
            if (collect(arena, &model_settings.server_keys, v.object, &edits)) |r| return r;
            plan.server_edits = edits.items;
        } else if (std.mem.eql(u8, k, "model")) {
            if (v != .string or v.string.len == 0) return reject(arena, 400, "invalid_request_error", "\"model\" must be a model id", .{});
            plan.model_id = v.string;
        } else if (std.mem.eql(u8, k, "set")) {
            if (v != .object) return reject(arena, 400, "invalid_request_error", "\"set\" must be an object", .{});
            set_obj = v.object;
        } else {
            return reject(arena, 400, "unknown_setting", "\"{s}\" is not a field of this request", .{k});
        }
    }
    if (set_obj) |obj| {
        if (plan.model_id == null) return reject(arena, 400, "invalid_request_error", "\"set\" needs a \"model\"", .{});
        var edits = std.ArrayList(Edit).empty;
        if (collect(arena, &model_settings.model_keys, obj, &edits)) |r| return r;
        plan.model_edits = edits.items;
    }
    if (plan.server_edits.len == 0 and plan.model_edits.len == 0) return reject(arena, 400, "invalid_request_error", "Nothing to change", .{});
    return .{ .ok = plan };
}

// ── HTTP ──

fn stateName(s: model_registry_mod.LoadState) []const u8 {
    return @tagName(s);
}

fn collectRows(allocator: std.mem.Allocator, registry: *model_registry_mod.ModelRegistry) ![]ModelRow {
    var rows = std.ArrayList(ModelRow).empty;
    errdefer rows.deinit(allocator);
    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    var it = registry.entries.valueIterator();
    while (it.next()) |p| {
        const e = p.*;
        try rows.append(allocator, .{ .id = e.id, .path = e.path, .state = stateName(e.state), .bytes = e.bytes_on_disk });
    }
    std.mem.sort(ModelRow, rows.items, {}, struct {
        fn lt(_: void, a: ModelRow, b: ModelRow) bool {
            return std.mem.lessThan(u8, a.id, b.id);
        }
    }.lt);
    return rows.toOwnedSlice(allocator);
}

pub fn handleGet(allocator: std.mem.Allocator, stream: *http_conn.Conn) !void {
    const registry = server.global_registry orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = model_settings.defaultPath(&path_buf);
    var file = model_settings.load(allocator, stream.io, path);
    defer file.deinit();
    const rows = try collectRows(allocator, registry);
    defer allocator.free(rows);
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    const default_dir = try std.fmt.allocPrint(allocator, "{s}/.sushi/models", .{home});
    defer allocator.free(default_dir);
    const roots: []const []const u8 = if (registry.discovery) |d| d.roots else &.{};
    const body = try render(allocator, path, &file, rows, &model_settings.launch_locks, .{
        .default_models_dir = default_dir,
        .boot_roots = roots,
        .cache_budget_bytes = server.prefix_cache_disk_bytes,
    });
    defer allocator.free(body);
    try server.sendResponse(stream, "200 OK", "application/json", body);
}

fn dirReadable(io: std.Io, path: []const u8) bool {
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return false;
    d.close(io);
    return true;
}

pub fn handlePost(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const sch = server.global_scheduler orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const plan = switch (parseRequest(arena, request_body)) {
        .ok => |p| p,
        .rejected => |r| {
            var status_buf: [32]u8 = undefined;
            const status = std.fmt.bufPrint(&status_buf, "{d} Bad Request", .{r.status}) catch "400 Bad Request";
            try server.sendErrorResponse(allocator, stream, status, r.err_type, r.message, r.status);
            return;
        },
    };

    for (plan.server_edits) |e| if (std.mem.eql(u8, e.key, "models_dir")) if (e.value) |v| {
        if (!dirReadable(stream.io, v.string)) {
            try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_setting", "models_dir does not exist or cannot be read", 400);
            return;
        }
    };
    var model_path: []const u8 = "";
    var model_loaded = false;
    if (plan.model_id) |id| {
        const entry = sch.registry.resolveEntry(id) catch {
            try server.sendErrorResponse(allocator, stream, "404 Not Found", "model_not_found", "Unknown model id", 404);
            return;
        };
        model_path = entry.path;
        sch.registry.mutex.lockUncancelable(sch.io);
        model_loaded = entry.state == .ready;
        sch.registry.mutex.unlock(sch.io);
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = model_settings.defaultPath(&path_buf);
    const write = struct {
        fn run(a: std.mem.Allocator, io: std.Io, p: []const u8, section: []const u8, edits: []const Edit) !void {
            if (edits.len > 0) try model_settings.writeEdits(a, io, p, section, edits);
        }
    }.run;
    write(allocator, stream.io, path, model_settings.server_section, plan.server_edits) catch |err| return writeFailed(allocator, stream, err);
    write(allocator, stream.io, path, model_path, plan.model_edits) catch |err| return writeFailed(allocator, stream, err);

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"ok\":true,\"applied\":{");
    var first = true;
    for (plan.server_edits) |e| {
        if (!first) try w.writeAll(",");
        first = false;
        try w.print("\"{s}\":", .{e.key});
        if (std.mem.eql(u8, e.key, "prefix_cache_disk_gb")) {
            try writeJson(w, applyCacheBudget(sch, e.value));
        } else {
            try writeJson(w, applyModelsDir(sch, e.value));
        }
    }
    if (plan.model_edits.len > 0) {
        if (!first) try w.writeAll(",");
        try w.writeAll("\"model\":");
        try writeJson(w, if (model_loaded) "saved; the loaded model uses it after a reload" else "saved; applies at the model's next load");
    }
    try w.print("}},\"model_loaded\":{}}}", .{model_loaded});
    try server.sendResponse(stream, "200 OK", "application/json", out.written());
}

fn writeFailed(allocator: std.mem.Allocator, stream: *http_conn.Conn, err: anyerror) !void {
    if (err == error.MalformedSettingsFile) {
        try server.sendErrorResponse(allocator, stream, "409 Conflict", "settings_file_malformed", "The settings file is not a JSON object; fix or remove it before saving", 409);
        return;
    }
    log.warn("[settings] write failed: {s}\n", .{@errorName(err)});
    try server.sendErrorResponse(allocator, stream, "500 Internal Server Error", "settings_write_failed", "Could not write the settings file", 500);
}

/// The new SSD tier size reaches the running tiers at once; the launch flag, when given, keeps the number it set.
fn applyCacheBudget(sch: *@import("scheduler.zig").Scheduler, value: ?std.json.Value) []const u8 {
    if (model_settings.launch_locks.prefix_cache_disk_gb.set) return "saved; the --prefix-cache-disk flag still sets the size until a start without it";
    const gb: ?u64 = if (value) |v| @intCast(v.integer) else null;
    server.prefix_cache_disk_bytes = if (gb) |g| g << 30 else disk_placement.DEFAULT_DISK_CAP;
    server.prefix_cache_disk_explicit = gb != null;
    const bytes = (gb orelse 0) << 30;
    if (gb == null or bytes == 0) return "saved; applies at the next model load";
    var it = sch.registry.entries.valueIterator();
    var tiers = std.ArrayList(*model_registry_mod.LoadedModel).empty;
    defer tiers.deinit(sch.allocator);
    sch.registry.mutex.lockUncancelable(sch.io);
    while (it.next()) |p| {
        if (p.*.state == .ready and p.*.disk_stats.snapshot().namespace != 0) tiers.append(sch.allocator, p.*) catch {};
    }
    sch.registry.mutex.unlock(sch.io);
    for (tiers.items) |e| _ = sch.prefixAdmin(.{ .entry = e, .action = .set_cap, .cap_bytes = bytes }) catch return "saved; applies at the next model load";
    return "applied to the running tiers";
}

fn applyModelsDir(sch: *@import("scheduler.zig").Scheduler, value: ?std.json.Value) []const u8 {
    _ = value;
    if (model_settings.launch_locks.models_dir.set) return "saved; the --model-dir flag still sets the directories until a start without it";
    _ = sch.registry.rescan() catch return "saved; the rescan failed, use /v1/models/rescan";
    return "saved; models rescanned";
}

// ── Tests ──

const testing = std.testing;

fn expectRejected(body: []const u8, status: u16, err_type: []const u8) !void {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    switch (parseRequest(arena_state.allocator(), body)) {
        .ok => return error.TestExpectedRejection,
        .rejected => |r| {
            try testing.expectEqual(status, r.status);
            try testing.expectEqualStrings(err_type, r.err_type);
        },
    }
}

test "settings api: a request names real keys with valid values, or is refused with a named 400" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const ok = parseRequest(arena_state.allocator(),
        \\{"server":{"prefix_cache_disk_gb":40,"models_dir":null},"model":"m","set":{"kv_quant":"8","mtp":null}}
    );
    try testing.expectEqual(@as(usize, 2), ok.ok.server_edits.len);
    try testing.expectEqual(@as(usize, 2), ok.ok.model_edits.len);
    try testing.expect(ok.ok.model_edits[1].value == null);
    try testing.expectEqualStrings("m", ok.ok.model_id.?);

    try expectRejected("{\"model\":\"m\",\"set\":{\"temperature\":1}}", 400, "unknown_setting");
    try expectRejected("{\"server\":{\"port\":1}}", 400, "unknown_setting");
    try expectRejected("{\"model\":\"m\",\"set\":{\"ssd_prefetch\":101}}", 400, "invalid_setting");
    try expectRejected("{\"server\":{\"models_dir\":\"relative\"}}", 400, "invalid_setting");
    try expectRejected("{\"set\":{\"mtp\":true}}", 400, "invalid_request_error");
    try expectRejected("{\"colour\":1}", 400, "unknown_setting");
    try expectRejected("{}", 400, "invalid_request_error");
    try expectRejected("[1]", 400, "invalid_request_error");
    try expectRejected("nope", 400, "invalid_request_error");
}

test "settings api: render reports the file's value, the pinning flag and the winner per key" {
    var file = try model_settings.parse(testing.allocator,
        \\{"/m/a": {"kv_quant": "4", "ssd_prefetch": 200, "mtp": false}, "server": {"prefix_cache_disk_gb": 30}}
    );
    defer file.deinit();
    var locks: model_settings.LaunchLocks = .{};
    locks.modelMut("mtp").pin("on", .{});
    const rows = [_]ModelRow{.{ .id = "a", .path = "/m/a", .state = "ready", .bytes = 5 }};
    const body = try render(testing.allocator, "/h/model-settings.json", &file, &rows, &locks, .{ .default_models_dir = "/h/.sushi/models", .boot_roots = &.{"/h/.sushi/models"}, .cache_budget_bytes = 30 << 30 });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const cache = root.get("server").?.object.get("prefix_cache_disk_gb").?.object;
    try testing.expectEqual(@as(i64, 30), cache.get("file").?.integer);
    try testing.expectEqualStrings("file", cache.get("source").?.string);
    try testing.expect(!cache.get("locked").?.bool);
    const s = root.get("models").?.array.items[0].object.get("settings").?.object;
    try testing.expectEqualStrings("file", s.get("kv_quant").?.object.get("source").?.string);
    try testing.expectEqualStrings("4", s.get("kv_quant").?.object.get("file").?.string);
    // The loader ignores 200 for ssd_prefetch, so the editor sees no file value.
    try testing.expect(s.get("ssd_prefetch").?.object.get("file").? == .null);
    try testing.expectEqualStrings("default", s.get("ssd_prefetch").?.object.get("source").?.string);
    const mtp = s.get("mtp").?.object;
    try testing.expectEqualStrings("flag", mtp.get("source").?.string);
    try testing.expect(mtp.get("locked").?.bool);
    try testing.expect(!mtp.get("file").?.bool);
    try testing.expectEqual(@as(usize, model_settings.model_keys.len), root.get("model_keys").?.array.items.len);
}
