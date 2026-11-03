//! The SSD prefix cache's admin surface: `GET /v1/prefix-cache` lists the namespaces on disk and which
//! loaded model owns each, `POST /v1/prefix-cache/clear` empties one model's tier or deletes the namespaces
//! no loaded model uses. The scan and the stale delete are filesystem-only and run on the connection thread;
//! a model's clear goes through its `DiskTier` on the inference thread, the tier's sole owner.
const std = @import("std");
const log = @import("log.zig");
const disk_placement = @import("disk_placement.zig");
const model_registry_mod = @import("model_registry.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const http_conn = @import("http_conn.zig");
const server = @import("server.zig");

pub const ns_name_len = 16;

/// A namespace directory is `<16 lowercase hex>`, the model fingerprint (`disk_placement.modelFingerprint`).
pub fn isNamespaceName(name: []const u8) bool {
    if (name.len != ns_name_len) return false;
    for (name) |c| if (!std.ascii.isDigit(c) and (c < 'a' or c > 'f')) return false;
    return true;
}

pub fn namespaceId(name: []const u8) ?u64 {
    if (!isNamespaceName(name)) return null;
    return std.fmt.parseInt(u64, name, 16) catch null;
}

/// The namespace a tier root lives in: `<base>/<fp>` or, for a private root, `<base>/<fp>/p<pid>-<n>`.
/// 0 when the root is not under a namespace directory.
pub fn namespaceOfRoot(root: []const u8, private_root: bool) u64 {
    const dir = if (private_root) std.fs.path.dirname(root) orelse return 0 else root;
    return namespaceId(std.fs.path.basename(dir)) orelse 0;
}

pub const Namespace = struct {
    id: u64,
    bytes: u64 = 0,
    /// Complete entries (a directory holding `meta.json`).
    entries: u32 = 0,
    /// Newest `meta.json` mtime, unix nanoseconds; 0 with no entry.
    newest_ns: i128 = 0,

    pub fn nameBuf(self: Namespace) [ns_name_len]u8 {
        var b: [ns_name_len]u8 = undefined;
        _ = std.fmt.bufPrint(&b, "{x:0>16}", .{self.id}) catch unreachable;
        return b;
    }
};

/// Bytes and entries under one tier root (`e<id>` directories), a hard-linked file counted once.
fn scanRoot(io: std.Io, root_abs: []const u8, seen: *disk_placement.InodeSet, out: *Namespace) void {
    var d = std.Io.Dir.openDirAbsolute(io, root_abs, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .directory or dent.name.len < 2) continue;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const sub = std.fmt.bufPrint(&buf, "{s}/{s}", .{ root_abs, dent.name }) catch continue;
        switch (dent.name[0]) {
            'e' => {
                out.bytes +|= disk_placement.dirBytes(io, sub, seen);
                var meta_buf: [std.fs.max_path_bytes]u8 = undefined;
                const meta_path = std.fmt.bufPrint(&meta_buf, "{s}/meta.json", .{sub}) catch continue;
                const meta = std.Io.Dir.cwd().statFile(io, meta_path, .{}) catch continue;
                out.entries += 1;
                out.newest_ns = @max(out.newest_ns, meta.mtime.nanoseconds);
            },
            'p' => scanRoot(io, sub, seen, out),
            else => {},
        }
    }
}

pub fn scanNamespace(allocator: std.mem.Allocator, io: std.Io, base_abs: []const u8, id: u64) Namespace {
    var ns: Namespace = .{ .id = id };
    var seen = disk_placement.InodeSet.init(allocator);
    defer seen.deinit();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{x:0>16}", .{ base_abs, id }) catch return ns;
    scanRoot(io, path, &seen, &ns);
    return ns;
}

/// Every namespace directory under `base_abs`, biggest first. A symlink is not a namespace.
pub fn scan(allocator: std.mem.Allocator, io: std.Io, base_abs: []const u8) ![]Namespace {
    var list = std.ArrayList(Namespace).empty;
    errdefer list.deinit(allocator);
    var base = std.Io.Dir.openDirAbsolute(io, base_abs, .{ .iterate = true }) catch return list.toOwnedSlice(allocator);
    defer base.close(io);
    var it = base.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .directory) continue;
        const id = namespaceId(dent.name) orelse continue;
        try list.append(allocator, scanNamespace(allocator, io, base_abs, id));
    }
    std.mem.sort(Namespace, list.items, {}, struct {
        fn gt(_: void, a: Namespace, b: Namespace) bool {
            return a.bytes > b.bytes;
        }
    }.gt);
    return list.toOwnedSlice(allocator);
}

pub const Deleted = struct {
    namespaces: u32 = 0,
    bytes: u64 = 0,
    /// Namespaces another process's live tier holds the lock of.
    skipped_live: u32 = 0,
    /// Namespace-named symlinks, left alone.
    refused_links: u32 = 0,
};

/// Deletes every namespace directory whose id is not in `keep`. Each root's lock is held across its delete, so a tier
/// that loads meanwhile takes a fresh root and one another process holds is skipped.
pub fn deleteUnused(allocator: std.mem.Allocator, io: std.Io, base_abs: []const u8, keep: []const u64) Deleted {
    return deleteNamespaces(allocator, io, base_abs, keep, null);
}

/// `deleteUnused`, narrowed to the one namespace `only` when given.
pub fn deleteNamespaces(allocator: std.mem.Allocator, io: std.Io, base_abs: []const u8, keep: []const u64, only: ?u64) Deleted {
    var res: Deleted = .{};
    var base = std.Io.Dir.openDirAbsolute(io, base_abs, .{ .iterate = true }) catch return res;
    defer base.close(io);
    var it = base.iterate();
    while (it.next(io) catch null) |dent| {
        const id = namespaceId(dent.name) orelse continue;
        if (only) |o| if (o != id) continue;
        if (dent.kind == .sym_link) {
            res.refused_links += 1;
            continue;
        }
        if (dent.kind != .directory) continue;
        if (std.mem.indexOfScalar(u64, keep, id) != null) continue;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ base_abs, dent.name }) catch continue;
        const lock = (disk_placement.lockRoot(io, path) catch continue) orelse {
            res.skipped_live += 1;
            continue;
        };
        defer lock.close(io);
        if (disk_placement.privateRootIsLive(io, path)) {
            res.skipped_live += 1;
            continue;
        }
        const bytes = scanNamespace(allocator, io, base_abs, id).bytes;
        base.deleteTree(io, dent.name) catch |err| {
            log.warn("[prefix-cache-admin] could not delete {s}: {s}\n", .{ path, @errorName(err) });
            continue;
        };
        res.namespaces += 1;
        res.bytes +|= bytes;
    }
    return res;
}

pub const ClearOutcome = enum { cleared, busy, not_loaded };

pub const ClearRequest = struct {
    entry: *model_registry_mod.LoadedModel,
    action: enum { clear, set_cap } = .clear,
    /// `set_cap`: the new operator ceiling; the tier re-derives its budget and evicts down to it.
    cap_bytes: u64 = 0,
    outcome: ClearOutcome = .not_loaded,
    freed_bytes: u64 = 0,
    entries_removed: u32 = 0,
    done: bool = false,
    done_mu: std.Io.Mutex = .init,
    done_cond: std.Io.Condition = .init,
};

/// Inference thread. A model with a request holding it is busy: the clear is refused rather than queued behind it.
pub fn runClear(io: std.Io, req: *ClearRequest) void {
    defer {
        req.done_mu.lockUncancelable(io);
        req.done = true;
        req.done_cond.broadcast(io);
        req.done_mu.unlock(io);
    }
    const entry = req.entry;
    if (entry.state != .ready) return;
    const hc = if (entry.prefix_cache) |*h| h else return;
    if (hc.disk == null) return;
    const d = &hc.disk.?;
    if (req.action == .set_cap) {
        const entries = d.entryCount();
        d.operator_cap = req.cap_bytes;
        d.refreshDiskBudget();
        req.entries_removed = @intCast(entries -| d.entryCount());
        req.outcome = .cleared;
        return;
    }
    if (entry.refcount.load(.acquire) > 0) {
        req.outcome = .busy;
        return;
    }
    const ns_dir = if (d.private_root) std.fs.path.dirname(d.root) orelse d.root else d.root;
    const base = std.fs.path.dirname(ns_dir) orelse ns_dir;
    const before = scanNamespace(entry.allocator, io, base, namespaceOfRoot(d.root, d.private_root));
    req.entries_removed = @intCast(d.entryCount());
    hc.clearDisk();
    const after = scanNamespace(entry.allocator, io, base, namespaceOfRoot(d.root, d.private_root));
    req.freed_bytes = before.bytes -| after.bytes;
    req.outcome = .cleared;
}

// ── HTTP ──

const Loaded = struct { id: []const u8, entry: *model_registry_mod.LoadedModel, snap: model_registry_mod.DiskStats.Snapshot };

/// Ready models with an SSD tier. Entry pointers stay valid while the server runs (stubs stay registered).
fn loadedTiers(allocator: std.mem.Allocator, registry: *model_registry_mod.ModelRegistry) ![]Loaded {
    var list = std.ArrayList(Loaded).empty;
    errdefer list.deinit(allocator);
    registry.mutex.lockUncancelable(registry.io);
    defer registry.mutex.unlock(registry.io);
    var it = registry.entries.valueIterator();
    while (it.next()) |p| {
        const e = p.*;
        if (e.state != .ready) continue;
        const snap = e.disk_stats.snapshot();
        if (snap.namespace == 0) continue;
        try list.append(allocator, .{ .id = e.id, .entry = e, .snap = snap });
    }
    return list.toOwnedSlice(allocator);
}

fn ownerOf(loaded: []const Loaded, id: u64) ?Loaded {
    for (loaded) |l| if (l.snap.namespace == id) return l;
    return null;
}

pub fn renderList(
    allocator: std.mem.Allocator,
    root: []const u8,
    volume: ?disk_placement.VolumeSpace,
    volume_name: ?[]const u8,
    volume_path: []const u8,
    namespaces: []const Namespace,
    loaded: []const Loaded,
    live_elsewhere: []const bool,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    const esc_root = try server.jsonEscape(allocator, root);
    defer allocator.free(esc_root);
    try w.print("{{\"root\":{s},\"volume\":", .{esc_root});
    if (volume) |v| {
        const esc_name = if (volume_name) |name| try server.jsonEscape(allocator, name) else try allocator.dupe(u8, "null");
        defer allocator.free(esc_name);
        const esc_path = try server.jsonEscape(allocator, volume_path);
        defer allocator.free(esc_path);
        try w.print("{{\"name\":{s},\"path\":{s},\"free_bytes\":{d},\"total_bytes\":{d}}}", .{ esc_name, esc_path, v.free, v.total });
    } else try w.writeAll("null");
    var total: u64 = 0;
    for (namespaces) |n| total +|= n.bytes;
    try w.print(",\"total_bytes\":{d},\"namespaces\":[", .{total});
    for (namespaces, 0..) |n, i| {
        const owner = ownerOf(loaded, n.id);
        const other = live_elsewhere[i] and owner == null;
        try w.print("{s}{{\"id\":\"{s}\",\"bytes\":{d},\"entries\":{d},\"newest_mtime\":{d},", .{
            if (i == 0) "" else ",", n.nameBuf(), n.bytes, n.entries, @divTrunc(n.newest_ns, std.time.ns_per_s),
        });
        if (owner) |o| {
            const esc_id = try server.jsonEscape(allocator, o.id);
            defer allocator.free(esc_id);
            try w.print("\"model\":{s},\"stale\":false,\"held_by_other_process\":false,\"tier\":{{\"used_bytes\":{d},\"cap_bytes\":{d},\"entries\":{d},\"written_bytes\":{d}}}}}", .{
                esc_id, o.snap.used, o.snap.budget, o.snap.entries, o.snap.written,
            });
        } else {
            try w.print("\"model\":null,\"stale\":{},\"held_by_other_process\":{},\"tier\":null}}", .{ !other, other });
        }
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}

fn cacheRoot(allocator: std.mem.Allocator) ![]u8 {
    return disk_placement.defaultBaseDir(allocator);
}

pub fn handleList(allocator: std.mem.Allocator, stream: *http_conn.Conn) !void {
    const registry = server.global_registry orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Registry not ready", 503);
        return;
    };
    const root = cacheRoot(allocator) catch {
        try server.sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "No cache directory", 500);
        return;
    };
    defer allocator.free(root);
    const loaded = try loadedTiers(allocator, registry);
    defer allocator.free(loaded);
    const namespaces = try scan(allocator, stream.io, root);
    defer allocator.free(namespaces);
    const other = try allocator.alloc(bool, namespaces.len);
    defer allocator.free(other);
    for (namespaces, other) |n, *o| {
        o.* = false;
        if (ownerOf(loaded, n.id) != null) continue;
        const nb = n.nameBuf();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = std.fmt.bufPrint(&buf, "{s}/{s}", .{ root, nb }) catch continue;
        o.* = disk_placement.namespaceIsLive(stream.io, path);
    }
    const volume_info = try disk_placement.cacheVolume(allocator);
    defer volume_info.deinit(allocator);
    const body = try renderList(allocator, root, volume_info.space, volume_info.name, volume_info.path, namespaces, loaded, other);
    defer allocator.free(body);
    try server.sendResponse(stream, "200 OK", "application/json", body);
}

const Scope = enum { model, stale, all, namespace };

/// The `id` of a namespace-scope clear: a namespace folder name.
fn clearNamespaceId(obj: std.json.ObjectMap) ?u64 {
    const v = obj.get("id") orelse return null;
    return if (v == .string) namespaceId(v.string) else null;
}

/// The `model` field of a model-scope clear. Empty and `sushi` mean the default model to the registry, so a
/// missing or non-string field must not reach it.
fn clearModelId(obj: std.json.ObjectMap) ?[]const u8 {
    const m = obj.get("model") orelse return null;
    return if (m == .string and m.string.len > 0) m.string else null;
}

pub fn parseScope(name: []const u8) ?Scope {
    return std.meta.stringToEnum(Scope, name);
}

pub fn handleClear(allocator: std.mem.Allocator, stream: *http_conn.Conn, request_body: []const u8) !void {
    const sch = server.global_scheduler orelse {
        try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Scheduler not ready", 503);
        return;
    };
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, request_body, .{}) catch {
        try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Body must be JSON", 400);
        return;
    };
    defer parsed.deinit();
    const obj = if (parsed.value == .object) parsed.value.object else {
        try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Body must be a JSON object", 400);
        return;
    };
    const scope_name = if (obj.get("scope")) |s| (if (s == .string) s.string else "") else "";
    const scope = parseScope(scope_name) orelse {
        try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "scope must be \"model\", \"namespace\", \"stale\" or \"all\"", 400);
        return;
    };
    const root = cacheRoot(allocator) catch {
        try server.sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "No cache directory", 500);
        return;
    };
    defer allocator.free(root);

    var model_freed: u64 = 0;
    var model_entries: u32 = 0;
    var busy = std.ArrayList([]const u8).empty;
    defer busy.deinit(allocator);
    var namespace_result: ?Deleted = null;

    if (scope == .namespace) {
        const ns = clearNamespaceId(obj) orelse {
            try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "id must be a 16-digit lowercase hex namespace name", 400);
            return;
        };
        const loaded = try loadedTiers(allocator, sch.registry);
        defer allocator.free(loaded);
        if (ownerOf(loaded, ns) != null) {
            try server.sendErrorResponse(allocator, stream, "409 Conflict", "namespace_in_use", "A loaded model uses this namespace; clear it with scope \"model\"", 409);
            return;
        }
        namespace_result = deleteNamespaces(allocator, stream.io, root, &.{}, ns);
        const r = namespace_result.?;
        if (r.namespaces == 0 and r.skipped_live == 0 and r.refused_links == 0) {
            try server.sendErrorResponse(allocator, stream, "404 Not Found", "namespace_not_found", "No such namespace in the cache", 404);
            return;
        }
    }

    if (scope == .model) {
        const id = clearModelId(obj) orelse {
            try server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "model must be a model id", 400);
            return;
        };
        const entry = sch.registry.resolveEntry(id) catch |err| {
            const no_default = err == error.NoDefaultModel;
            try server.sendErrorResponse(allocator, stream, if (no_default) "503 Service Unavailable" else "404 Not Found", if (no_default) "no_model" else "model_not_found", if (no_default) "No default model configured" else "Unknown model id", if (no_default) 503 else 404);
            return;
        };
        const r = sch.prefixAdmin(.{ .entry = entry }) catch {
            try server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "Server is shutting down", 503);
            return;
        };
        switch (r.outcome) {
            .not_loaded => {
                try server.sendErrorResponse(allocator, stream, "409 Conflict", "model_not_loaded", "The model is not loaded or has no SSD prefix cache; use scope \"stale\" to delete the caches of unloaded models", 409);
                return;
            },
            .busy => {
                try server.sendErrorResponse(allocator, stream, "409 Conflict", "model_busy", "A request is using the model; retry when it finishes", 409);
                return;
            },
            .cleared => {
                model_freed = r.freed_bytes;
                model_entries = r.entries_removed;
            },
        }
    }

    if (scope == .all) {
        const loaded = try loadedTiers(allocator, sch.registry);
        defer allocator.free(loaded);
        for (loaded) |l| {
            const r = sch.prefixAdmin(.{ .entry = l.entry }) catch break;
            switch (r.outcome) {
                .cleared => {
                    model_freed +|= r.freed_bytes;
                    model_entries += r.entries_removed;
                },
                .busy => try busy.append(allocator, l.id),
                .not_loaded => {},
            }
        }
    }

    var stale: Deleted = namespace_result orelse .{};
    if (scope == .stale or scope == .all) {
        const loaded = try loadedTiers(allocator, sch.registry);
        defer allocator.free(loaded);
        var keep = try allocator.alloc(u64, loaded.len);
        defer allocator.free(keep);
        for (loaded, 0..) |l, i| keep[i] = l.snap.namespace;
        stale = deleteUnused(allocator, stream.io, root, keep);
    }

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.print("{{\"scope\":\"{s}\",\"freed_bytes\":{d},\"model_freed_bytes\":{d},\"model_entries_removed\":{d},\"stale_freed_bytes\":{d},\"stale_namespaces_removed\":{d},\"skipped_live_namespaces\":{d},\"refused_symlinks\":{d},\"busy_models\":[", .{
        @tagName(scope), model_freed +| stale.bytes, model_freed, model_entries, stale.bytes, stale.namespaces, stale.skipped_live, stale.refused_links,
    });
    for (busy.items, 0..) |id, i| {
        const esc = try server.jsonEscape(allocator, id);
        defer allocator.free(esc);
        try w.print("{s}{s}", .{ if (i == 0) "" else ",", esc });
    }
    try w.writeAll("]}");
    log.info("[prefix-cache-admin] clear scope={s}: freed {d} MB\n", .{ @tagName(scope), (model_freed +| stale.bytes) >> 20 });
    try server.sendResponse(stream, "200 OK", "application/json", out.written());
}

// ── Tests ──

const testing = std.testing;

fn filler(comptime n: usize) *const [n]u8 {
    const bytes: [n]u8 = @splat('f');
    return &bytes;
}

fn put(tmp: *testing.TmpDir, io: std.Io, rel: []const u8, data: []const u8) !void {
    if (std.fs.path.dirname(rel)) |d| try tmp.dir.createDirPath(io, d);
    try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = data });
}

fn tmpRoot(tmp: *testing.TmpDir, io: std.Io, buf: []u8) ![]const u8 {
    return buf[0..try tmp.dir.realPath(io, buf)];
}

fn zpath(buf: []u8, comptime fmt: []const u8, args: anytype) ![:0]u8 {
    return std.fmt.bufPrintSentinel(buf, fmt, args, 0);
}

test "prefix cache admin: isNamespaceName: only a 16-digit lowercase hex name is a namespace" {
    try testing.expect(isNamespaceName("0123456789abcdef"));
    try testing.expect(!isNamespaceName("0123456789ABCDEF"));
    try testing.expect(!isNamespaceName("0123456789abcde"));
    try testing.expect(!isNamespaceName("0123456789abcdeg"));
    try testing.expect(!isNamespaceName("models"));
    try testing.expectEqual(@as(u64, 0xabc), namespaceId("0000000000000abc").?);
}

test "prefix cache admin: namespaceOfRoot: a private root belongs to the namespace above it" {
    try testing.expectEqual(@as(u64, 0xabc), namespaceOfRoot("/c/0000000000000abc", false));
    try testing.expectEqual(@as(u64, 0xabc), namespaceOfRoot("/c/0000000000000abc/p1-0", true));
    try testing.expectEqual(@as(u64, 0), namespaceOfRoot("/c/other", false));
}

test "prefix cache admin: scan: sizes, entries and newest mtime per namespace; a hard link counts once; other directories are not namespaces" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try put(&tmp, io, "00000000000000a1/e1/meta.json", "{}");
    try put(&tmp, io, "00000000000000a1/e1/c0.safetensors", filler(1000));
    try put(&tmp, io, "00000000000000a1/e2/meta.json", "{}");
    try put(&tmp, io, "00000000000000a1/e3/c0.safetensors", filler(50));
    try put(&tmp, io, "00000000000000a1/p9-0/e1/meta.json", "{}");
    try put(&tmp, io, "notes/e1/meta.json", "{}");
    try put(&tmp, io, "00000000000000b2/e1/meta.json", "{}");

    var sbuf: [1024]u8 = undefined;
    const src = try zpath(&sbuf, "{s}/00000000000000a1/e1/c0.safetensors", .{base});
    var dbuf: [1024]u8 = undefined;
    const dst = try zpath(&dbuf, "{s}/00000000000000a1/e2/c0.safetensors", .{base});
    try testing.expectEqual(@as(c_int, 0), std.c.link(src, dst));

    const found = try scan(testing.allocator, io, base);
    defer testing.allocator.free(found);
    try testing.expectEqual(@as(usize, 2), found.len);
    try testing.expectEqual(@as(u64, 0xa1), found[0].id);
    try testing.expectEqual(@as(u64, 1000 + 50 + 3 * 2), found[0].bytes);
    try testing.expectEqual(@as(u32, 3), found[0].entries);
    try testing.expect(found[0].newest_ns > 0);
    try testing.expectEqual(@as(u32, 1), found[1].entries);
}

test "prefix cache admin: deleteUnused: keeps the named namespaces and non-cache directories, refuses a symlink out of the root, skips a held root" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try put(&tmp, io, "root/00000000000000a1/e1/meta.json", "{}");
    try put(&tmp, io, "root/00000000000000a1/e1/c0.safetensors", filler(500));
    try put(&tmp, io, "root/00000000000000b2/e1/c0.safetensors", filler(300));
    try put(&tmp, io, "root/00000000000000c3/e1/c0.safetensors", filler(100));
    try put(&tmp, io, "root/notes/keep.txt", "mine");
    try put(&tmp, io, "outside/precious.bin", "do not delete");

    var lbuf: [1024]u8 = undefined;
    const target = try zpath(&lbuf, "{s}/outside", .{base});
    var link_buf: [1024]u8 = undefined;
    const link = try zpath(&link_buf, "{s}/root/00000000000000d4", .{base});
    try testing.expectEqual(@as(c_int, 0), std.c.symlink(target, link));

    var rbuf: [1024]u8 = undefined;
    const root = try std.fmt.bufPrint(&rbuf, "{s}/root", .{base});
    var held_buf: [1024]u8 = undefined;
    const held_path = try std.fmt.bufPrint(&held_buf, "{s}/00000000000000c3", .{root});
    const held = (try disk_placement.lockRoot(io, held_path)).?;
    defer held.close(io);

    const res = deleteUnused(testing.allocator, io, root, &.{0xa1});
    try testing.expectEqual(@as(u32, 1), res.namespaces);
    try testing.expectEqual(@as(u64, 300), res.bytes);
    try testing.expectEqual(@as(u32, 1), res.skipped_live);
    try testing.expectEqual(@as(u32, 1), res.refused_links);
    try tmp.dir.access(io, "root/00000000000000a1/e1/meta.json", .{});
    try tmp.dir.access(io, "root/00000000000000c3/e1/c0.safetensors", .{});
    try tmp.dir.access(io, "root/notes/keep.txt", .{});
    try tmp.dir.access(io, "outside/precious.bin", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "root/00000000000000b2", .{}));
}

test "prefix cache admin: clearModelId: a model-scope clear names its model, a missing or non-string one is refused" {
    const parse = struct {
        fn run(arena: std.mem.Allocator, body: []const u8) !std.json.ObjectMap {
            return (try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{})).object;
        }
    }.run;
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try testing.expectEqualStrings("m", clearModelId(try parse(a, "{\"scope\":\"model\",\"model\":\"m\"}")).?);
    try testing.expect(clearModelId(try parse(a, "{\"scope\":\"model\"}")) == null);
    try testing.expect(clearModelId(try parse(a, "{\"scope\":\"model\",\"model\":null}")) == null);
    try testing.expect(clearModelId(try parse(a, "{\"scope\":\"model\",\"model\":5}")) == null);
    try testing.expect(clearModelId(try parse(a, "{\"scope\":\"model\",\"model\":\"\"}")) == null);
}

test "prefix cache admin: deleteUnused: a namespace whose private root another process holds is skipped whole" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    try put(&tmp, io, "root/00000000000000c3/e1/c0.safetensors", filler(100));
    try put(&tmp, io, "root/00000000000000c3/p999-0/e1/c0.safetensors", filler(100));
    var rbuf: [1024]u8 = undefined;
    const root = try std.fmt.bufPrint(&rbuf, "{s}/root", .{base});
    var held_buf: [1024]u8 = undefined;
    const held_path = try std.fmt.bufPrint(&held_buf, "{s}/00000000000000c3/p999-0", .{root});
    const held = (try disk_placement.lockRoot(io, held_path)).?;
    defer held.close(io);

    const res = deleteUnused(testing.allocator, io, root, &.{});
    try testing.expectEqual(@as(u32, 0), res.namespaces);
    try testing.expectEqual(@as(u32, 1), res.skipped_live);
    try tmp.dir.access(io, "root/00000000000000c3/p999-0/e1/c0.safetensors", .{});
    try tmp.dir.access(io, "root/00000000000000c3/e1/c0.safetensors", .{});
}

test "prefix cache admin: deleteNamespaces: one namespace by id; a held, symlinked or absent one is not deleted" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    try put(&tmp, io, "root/00000000000000a1/e1/c0.safetensors", filler(100));
    try put(&tmp, io, "root/00000000000000b2/e1/c0.safetensors", filler(200));
    try put(&tmp, io, "root/00000000000000c3/p9-0/e1/c0.safetensors", filler(300));
    var rbuf: [1024]u8 = undefined;
    const root = try std.fmt.bufPrint(&rbuf, "{s}/root", .{base});

    var held_buf: [1024]u8 = undefined;
    const held_path = try std.fmt.bufPrint(&held_buf, "{s}/00000000000000c3/p9-0", .{root});
    const held = (try disk_placement.lockRoot(io, held_path)).?;
    defer held.close(io);

    const one = deleteNamespaces(testing.allocator, io, root, &.{}, 0xb2);
    try testing.expectEqual(@as(u32, 1), one.namespaces);
    try testing.expectEqual(@as(u64, 200), one.bytes);
    try tmp.dir.access(io, "root/00000000000000a1/e1/c0.safetensors", .{});
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "root/00000000000000b2", .{}));

    const live = deleteNamespaces(testing.allocator, io, root, &.{}, 0xc3);
    try testing.expectEqual(@as(u32, 0), live.namespaces);
    try testing.expectEqual(@as(u32, 1), live.skipped_live);
    try tmp.dir.access(io, "root/00000000000000c3/p9-0/e1/c0.safetensors", .{});

    const none = deleteNamespaces(testing.allocator, io, root, &.{}, 0xdd);
    try testing.expectEqual(@as(u32, 0), none.namespaces + none.skipped_live + none.refused_links);
}

test "prefix cache admin: clearNamespaceId: only a namespace folder name names a namespace" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const parse = struct {
        fn run(arena: std.mem.Allocator, body: []const u8) !std.json.ObjectMap {
            return (try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{})).object;
        }
    }.run;
    try testing.expectEqual(@as(?u64, 0xa1), clearNamespaceId(try parse(a, "{\"id\":\"00000000000000a1\"}")));
    try testing.expectEqual(@as(?u64, null), clearNamespaceId(try parse(a, "{}")));
    try testing.expectEqual(@as(?u64, null), clearNamespaceId(try parse(a, "{\"id\":5}")));
    try testing.expectEqual(@as(?u64, null), clearNamespaceId(try parse(a, "{\"id\":\"../etc\"}")));
}

test "prefix cache admin: renderList: a loaded model's namespace carries its tier, the rest are stale or held elsewhere" {
    const ns = [_]Namespace{ .{ .id = 0xa1, .bytes = 10, .entries = 1, .newest_ns = 5 * std.time.ns_per_s }, .{ .id = 0xb2, .bytes = 20 }, .{ .id = 0xc3 } };
    var entry: model_registry_mod.LoadedModel = undefined;
    const loaded = [_]Loaded{.{ .id = "m", .entry = &entry, .snap = .{ .budget = 100, .used = 10, .entries = 1, .written = 77, .namespace = 0xa1 } }};
    const body = try renderList(testing.allocator, "/cache", .{ .free = 9, .total = 99 }, "test-volume", "/cache", &ns, &loaded, &.{ false, false, true });
    defer testing.allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try testing.expectEqual(@as(i64, 30), root.get("total_bytes").?.integer);
    try testing.expectEqual(@as(i64, 9), root.get("volume").?.object.get("free_bytes").?.integer);
    try testing.expectEqualStrings("test-volume", root.get("volume").?.object.get("name").?.string);
    try testing.expectEqualStrings("/cache", root.get("volume").?.object.get("path").?.string);
    const rows = root.get("namespaces").?.array.items;
    try testing.expectEqualStrings("m", rows[0].object.get("model").?.string);
    try testing.expectEqual(@as(i64, 77), rows[0].object.get("tier").?.object.get("written_bytes").?.integer);
    try testing.expectEqual(@as(i64, 5), rows[0].object.get("newest_mtime").?.integer);
    try testing.expect(rows[1].object.get("stale").?.bool);
    try testing.expect(!rows[2].object.get("stale").?.bool);
    try testing.expect(rows[2].object.get("held_by_other_process").?.bool);
}

test "prefix cache admin: clearDisk: the index, the files and the budget agree, and the tier takes a new write afterwards" {
    try @import("mlx.zig").requireGpu();
    const io = testing.io;
    const kv_disk_cache = @import("kv_disk_cache.zig");
    const KVCache = @import("kv_cache.zig").KVCache;
    const s = @import("mlx.zig").gpuStream();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var hc = prefix_cache_mod.HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "00000000000000a1", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try prefix_cache_mod.testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const before = scanNamespace(testing.allocator, io, base, 0xa1);
    try testing.expect(before.bytes > 0);

    hc.clearDisk();
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.disk.?.total_bytes);
    try testing.expectEqual(@as(u64, 0), scanNamespace(testing.allocator, io, base, 0xa1).bytes);

    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 900);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.total_bytes > 0);
    try testing.expect(scanNamespace(testing.allocator, io, base, 0xa1).bytes > 0);
}
