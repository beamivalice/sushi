//! What `sushi pull` knows about a Hugging Face repo and the folder it fills: the endpoint, the tree listing with
//! content ids, the manifest a finished pull leaves, and the decision to skip a file. A file is kept only when
//! its content id matches, never by size alone. Links no MLX.
const std = @import("std");
const json_text = @import("json_text.zig");

pub const default_endpoint = "https://huggingface.co";

/// Name of the manifest a finished pull writes into the pack folder.
pub const manifest_name = ".sushi-pull.json";

/// Test-only (tests/test_pull_integrity.sh): replaces https://huggingface.co so a stub repo is served offline.
pub fn endpoint() []const u8 {
    const v = if (std.c.getenv("SUSHI_HF_ENDPOINT")) |e| std.mem.span(e) else return default_endpoint;
    const trimmed = std.mem.trimEnd(u8, v, "/");
    return if (trimmed.len == 0) default_endpoint else trimmed;
}

/// `base` + a formatted path, e.g. `hfUrl(a, endpoint(), "/api/models/{s}", .{repo})`.
pub fn hfUrl(allocator: std.mem.Allocator, base: []const u8, comptime tail_fmt: []const u8, args: anytype) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}" ++ tail_fmt, .{base} ++ args);
}

/// One file of a repo or of a manifest. `oid` is the sha256 for an LFS file and the git blob sha1 for a small one;
/// empty when the listing carried none.
pub const RepoFile = struct {
    path: []u8,
    size: u64,
    oid: []u8 = &.{},
    lfs: bool = false,
};

pub fn freeRepoFiles(allocator: std.mem.Allocator, files: []RepoFile) void {
    for (files) |f| {
        allocator.free(f.path);
        allocator.free(f.oid);
    }
    allocator.free(files);
}

/// Parse the HF `/api/models/<repo>/tree/main?recursive=true` JSON array.
/// LFS entries report the real artifact size and sha256 under `lfs`.
pub fn parseTreeJson(allocator: std.mem.Allocator, json: []const u8) ![]RepoFile {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidTree;
    defer parsed.deinit();
    if (parsed.value != .array) return error.InvalidTree;

    var files = std.ArrayList(RepoFile).empty;
    errdefer {
        for (files.items) |f| {
            allocator.free(f.path);
            allocator.free(f.oid);
        }
        files.deinit(allocator);
    }
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const obj = item.object;
        const t = obj.get("type") orelse continue;
        if (t != .string or !std.mem.eql(u8, t.string, "file")) continue;
        const p = obj.get("path") orelse continue;
        if (p != .string) continue;
        var size: u64 = 0;
        var oid: []const u8 = "";
        var lfs = false;
        if (obj.get("lfs")) |l| {
            if (l == .object) {
                if (l.object.get("size")) |s| {
                    if (s == .integer and s.integer > 0) size = @intCast(s.integer);
                }
                if (l.object.get("oid")) |o| if (o == .string) {
                    oid = o.string;
                    lfs = true;
                };
            }
        }
        if (!lfs) {
            if (obj.get("oid")) |o| if (o == .string) {
                oid = o.string;
            };
        }
        if (size == 0) {
            if (obj.get("size")) |s| {
                if (s == .integer and s.integer > 0) size = @intCast(s.integer);
            }
        }
        const path = try allocator.dupe(u8, p.string);
        errdefer allocator.free(path);
        try files.append(allocator, .{ .path = path, .size = size, .oid = try allocator.dupe(u8, oid), .lfs = lfs });
    }
    return files.toOwnedSlice(allocator);
}

pub const Manifest = struct {
    repo: []const u8 = "",
    commit: []const u8 = "",
    files: []const RepoFile = &.{},

    pub fn find(self: Manifest, path: []const u8) ?RepoFile {
        for (self.files) |f| if (std.mem.eql(u8, f.path, path)) return f;
        return null;
    }
};

pub fn manifestJson(allocator: std.mem.Allocator, m: Manifest) ![]u8 {
    var out = std.Io.Writer.Allocating.init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.print("{{\"version\":1,\"repo\":{f},\"commit\":{f},\"files\":[", .{ json_text.Str{ .s = m.repo }, json_text.Str{ .s = m.commit } });
    for (m.files, 0..) |f, i| {
        try w.print("{s}\n{{\"path\":{f},\"size\":{d},\"{s}\":{f}}}", .{ if (i == 0) "" else ",", json_text.Str{ .s = f.path }, f.size, if (f.lfs) "sha256" else "git_oid", json_text.Str{ .s = f.oid } });
    }
    try w.writeAll("\n]}\n");
    return out.toOwnedSlice();
}

/// A manifest read back; everything is owned by `arena`.
pub fn parseManifest(arena: std.mem.Allocator, bytes: []const u8) !Manifest {
    const Row = struct { path: []const u8, size: u64 = 0, sha256: ?[]const u8 = null, git_oid: ?[]const u8 = null };
    const Doc = struct { version: u32 = 0, repo: []const u8 = "", commit: []const u8 = "", files: []const Row = &.{} };
    const doc = std.json.parseFromSliceLeaky(Doc, arena, bytes, .{ .ignore_unknown_fields = true }) catch return error.InvalidManifest;
    if (doc.version != 1) return error.InvalidManifest;
    const files = try arena.alloc(RepoFile, doc.files.len);
    for (doc.files, files) |r, *f| f.* = .{
        .path = @constCast(r.path),
        .size = r.size,
        .oid = @constCast(r.sha256 orelse r.git_oid orelse ""),
        .lfs = r.sha256 != null,
    };
    return .{ .repo = doc.repo, .commit = doc.commit, .files = files };
}

/// The manifest in `dir_path`, or null when there is none or it does not parse.
pub fn readManifest(arena: std.mem.Allocator, io: std.Io, dir_path: []const u8) ?Manifest {
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return null;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, manifest_name, arena, .limited(64 * 1024 * 1024)) catch return null;
    return parseManifest(arena, raw) catch null;
}

/// Written beside its final name and renamed, so a crash never leaves half a manifest.
pub fn writeManifest(allocator: std.mem.Allocator, io: std.Io, dir_path: []const u8, m: Manifest) !void {
    const bytes = try manifestJson(allocator, m);
    defer allocator.free(bytes);
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = manifest_name ++ ".tmp", .data = bytes });
    try dir.rename(manifest_name ++ ".tmp", dir, manifest_name, io);
}

pub fn removeManifest(io: std.Io, dir_path: []const u8) void {
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return;
    defer dir.close(io);
    dir.deleteFile(io, manifest_name) catch {};
}

pub const FileEvent = struct { index: usize, path: []const u8, size: u64, skip: bool };

/// What a pull reports beside its lines, for a caller that draws progress (the web page's Available list).
pub const PullEvent = union(enum) {
    /// The files and bytes the pack downloads, and how many of those bytes are on disk already (`bytesPresent`),
    /// before anything is written.
    listed: struct { files: usize, bytes: u64, present: u64 },
    /// Hashing the files already on disk.
    checking,
    /// File `index` (from 1) starts; `skip` when it is already complete.
    file: FileEvent,
    /// The curl process of the file in flight, still unreaped; null once it exited.
    child: ?std.posix.pid_t,
    /// The file is down; its checksum is being checked.
    verifying,
};

/// The listed bytes a download need not fetch: files on disk at their listed size, and the partials of the others.
pub fn bytesPresent(io: std.Io, dir_path: []const u8, files: []const RepoFile) u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return 0;
    defer dir.close(io);
    var total: u64 = 0;
    for (files) |f| {
        const whole = if (dir.statFile(io, f.path, .{})) |st| st.size == f.size else |_| false;
        if (whole) {
            total += f.size;
            continue;
        }
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const partial = std.fmt.bufPrint(&buf, "{s}.partial", .{f.path}) catch continue;
        const size = (dir.statFile(io, partial, .{}) catch continue).size;
        if (partialKeyed(dir, io, partial, f)) total += @min(size, f.size);
    }
    return total;
}

/// A `.partial` resumes only for the remote file it was started for: `<file>.partial.oid` holds "<oid> <size>".
/// A partial with no key, or another one, is from a pull before keys or of another revision.
pub fn keyMatches(sidecar: []const u8, f: RepoFile) bool {
    if (f.oid.len == 0) return false;
    var buf: [128]u8 = undefined;
    const want = std.fmt.bufPrint(&buf, "{s} {d}", .{ f.oid, f.size }) catch return false;
    return std.mem.eql(u8, std.mem.trim(u8, sidecar, " \r\n"), want);
}

fn partialKeyed(dir: std.Io.Dir, io: std.Io, partial: []const u8, f: RepoFile) bool {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const key_path = std.fmt.bufPrint(&path_buf, "{s}.oid", .{partial}) catch return false;
    var buf: [128]u8 = undefined;
    const got = dir.readFile(io, key_path, &buf) catch return false;
    return keyMatches(got, f);
}

/// Before curl resumes `partial`: keeps it when it is keyed to `f`, else deletes it, and records `f`'s key.
pub fn claimPartial(io: std.Io, partial: []const u8, f: RepoFile) !void {
    if (f.oid.len == 0) return;
    const cwd = std.Io.Dir.cwd();
    if (!partialKeyed(cwd, io, partial, f)) cwd.deleteFile(io, partial) catch {};
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const key_path = try std.fmt.bufPrint(&path_buf, "{s}.oid", .{partial});
    var buf: [128]u8 = undefined;
    try cwd.writeFile(io, .{ .sub_path = key_path, .data = try std.fmt.bufPrint(&buf, "{s} {d}\n", .{ f.oid, f.size }) });
}

/// Removes `partial` and its key.
pub fn dropPartial(io: std.Io, partial: []const u8) void {
    const cwd = std.Io.Dir.cwd();
    cwd.deleteFile(io, partial) catch {};
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const key_path = std.fmt.bufPrint(&path_buf, "{s}.oid", .{partial}) catch return;
    cwd.deleteFile(io, key_path) catch {};
}

pub const Action = enum { skip, download, hash };

/// What to do with one wanted file. `old` is the previous manifest's row for it (null: no manifest, or not named),
/// `disk_size` the size on disk (null: absent). `hash` means the bytes on disk must be checked against `f.oid`.
pub fn decide(f: RepoFile, old: ?RepoFile, have_manifest: bool, disk_size: ?u64) Action {
    const have = disk_size orelse return .download;
    if (f.size == 0 or have != f.size) return .download;
    if (f.oid.len == 0) return .skip;
    if (have_manifest) if (old) |o| {
        return if (o.size == f.size and o.lfs == f.lfs and std.mem.eql(u8, o.oid, f.oid)) .skip else .download;
    };
    return .hash;
}

/// A manifest path is removable only when it stays inside the pack folder.
pub fn safeRelative(path: []const u8) bool {
    if (path.len == 0 or path[0] == '/') return false;
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |part| {
        if (part.len == 0 or std.mem.eql(u8, part, "..") or std.mem.eql(u8, part, ".")) return false;
    }
    return true;
}

/// Content id of the file at `abs_path` in the form the listing uses: sha256 (lfs) or git blob sha1, as hex.
pub fn digestFile(io: std.Io, abs_path: []const u8, lfs: bool, out: *[64]u8) ![]const u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, abs_path, .{});
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var rbuf: [4096]u8 = undefined;
    var reader = file.reader(io, &rbuf);
    const chunk = try std.heap.page_allocator.alloc(u8, 1 << 20);
    defer std.heap.page_allocator.free(chunk);
    if (lfs) {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        while (true) {
            const n = try reader.interface.readSliceShort(chunk);
            if (n == 0) break;
            h.update(chunk[0..n]);
        }
        out.* = std.fmt.bytesToHex(h.finalResult(), .lower);
        return out[0..64];
    }
    var h = std.crypto.hash.Sha1.init(.{});
    var head: [32]u8 = undefined;
    h.update(std.fmt.bufPrint(&head, "blob {d}\x00", .{size}) catch unreachable);
    while (true) {
        const n = try reader.interface.readSliceShort(chunk);
        if (n == 0) break;
        h.update(chunk[0..n]);
    }
    const hex = std.fmt.bytesToHex(h.finalResult(), .lower);
    @memcpy(out[0..40], &hex);
    return out[0..40];
}

pub fn digestMatches(io: std.Io, abs_path: []const u8, f: RepoFile) bool {
    var buf: [64]u8 = undefined;
    const got = digestFile(io, abs_path, f.lfs, &buf) catch return false;
    return std.ascii.eqlIgnoreCase(got, f.oid);
}

const HashJob = struct {
    io: std.Io,
    dir_path: []const u8,
    files: []const RepoFile,
    match: []bool,
    stop: ?*const std.atomic.Value(bool),
    next: std.atomic.Value(usize) = .init(0),

    fn run(self: *HashJob) void {
        while (true) {
            if (self.stop) |s| if (s.load(.monotonic)) return;
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= self.files.len) return;
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const path = std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ self.dir_path, self.files[i].path }) catch continue;
            self.match[i] = digestMatches(self.io, path, self.files[i]);
        }
    }
};

/// Hashes the files in parallel; `match[i]` is whether `files[i]` on disk has its listed content id. Once `stop` is
/// set no further file starts (the rest read as not matching).
pub fn hashMatches(io: std.Io, dir_path: []const u8, files: []const RepoFile, match: []bool, stop: ?*const std.atomic.Value(bool)) void {
    @memset(match, false);
    var job = HashJob{ .io = io, .dir_path = dir_path, .files = files, .match = match, .stop = stop };
    var threads: [7]std.Thread = undefined;
    var spawned: usize = 0;
    const want = @min(threads.len, files.len -| 1);
    while (spawned < want) : (spawned += 1) {
        threads[spawned] = std.Thread.spawn(.{}, HashJob.run, .{&job}) catch break;
    }
    job.run();
    for (threads[0..spawned]) |t| t.join();
}

/// Deletes the files the previous manifest named that the repo no longer lists; nothing else in the folder.
pub fn removeDropped(io: std.Io, dir_path: []const u8, old: Manifest, listing: []const RepoFile) usize {
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return 0;
    defer dir.close(io);
    var removed: usize = 0;
    for (old.files) |o| {
        if (!safeRelative(o.path)) continue;
        const listed = for (listing) |f| {
            if (std.mem.eql(u8, f.path, o.path)) break true;
        } else false;
        if (listed) continue;
        dir.deleteFile(io, o.path) catch continue;
        removed += 1;
    }
    return removed;
}

const testing = std.testing;

test "hf pull: hfUrl joins the base and the formatted path" {
    const u = try hfUrl(testing.allocator, "http://127.0.0.1:1", "/{s}/resolve/main/{s}", .{ "o/r", "a.json" });
    defer testing.allocator.free(u);
    try testing.expectEqualStrings("http://127.0.0.1:1/o/r/resolve/main/a.json", u);
}

test "hf pull: the tree listing yields the sha256 of an LFS file and the blob id of a small one" {
    const files = try parseTreeJson(testing.allocator,
        \\[{"type":"directory","path":"mtp","size":0},
        \\ {"type":"file","path":"config.json","size":1234,"oid":"aa11"},
        \\ {"type":"file","path":"m.safetensors","size":135,"oid":"ptr","lfs":{"size":5300000000,"oid":"bb22"}},
        \\ {"type":"file","path":"bare","size":5}]
    );
    defer freeRepoFiles(testing.allocator, files);
    try testing.expectEqual(@as(usize, 3), files.len);
    try testing.expectEqualStrings("aa11", files[0].oid);
    try testing.expect(!files[0].lfs);
    try testing.expectEqual(@as(u64, 5_300_000_000), files[1].size);
    try testing.expectEqualStrings("bb22", files[1].oid);
    try testing.expect(files[1].lfs);
    try testing.expectEqualStrings("", files[2].oid);
}

test "hf pull: a manifest survives a round trip" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var files = [_]RepoFile{
        .{ .path = @constCast("config.json"), .size = 12, .oid = @constCast("aa11"), .lfs = false },
        .{ .path = @constCast("d/m \"1\".safetensors"), .size = 99, .oid = @constCast("bb22"), .lfs = true },
    };
    const text = try manifestJson(testing.allocator, .{ .repo = "beamster/X", .commit = "c0ffee", .files = &files });
    defer testing.allocator.free(text);
    const back = try parseManifest(arena.allocator(), text);
    try testing.expectEqualStrings("beamster/X", back.repo);
    try testing.expectEqualStrings("c0ffee", back.commit);
    try testing.expectEqual(@as(usize, 2), back.files.len);
    try testing.expectEqualStrings("d/m \"1\".safetensors", back.files[1].path);
    try testing.expect(back.files[1].lfs and !back.files[0].lfs);
    try testing.expectEqualStrings("aa11", back.files[0].oid);
    try testing.expectError(error.InvalidManifest, parseManifest(arena.allocator(), "{\"version\":2}"));
    try testing.expectError(error.InvalidManifest, parseManifest(arena.allocator(), "nope"));
}

test "hf pull: a file is skipped only when its content id matches" {
    const listed: RepoFile = .{ .path = @constCast("a"), .size = 10, .oid = @constCast("new"), .lfs = true };
    const same: RepoFile = .{ .path = @constCast("a"), .size = 10, .oid = @constCast("new"), .lfs = true };
    const changed: RepoFile = .{ .path = @constCast("a"), .size = 10, .oid = @constCast("old"), .lfs = true };
    try testing.expectEqual(Action.skip, decide(listed, same, true, 10));
    // Same size, other content: the bug a size-only skip had.
    try testing.expectEqual(Action.download, decide(listed, changed, true, 10));
    try testing.expectEqual(Action.download, decide(listed, same, true, 9));
    try testing.expectEqual(Action.download, decide(listed, same, true, null));
    try testing.expectEqual(Action.hash, decide(listed, null, false, 10));
    try testing.expectEqual(Action.hash, decide(listed, null, true, 10));
    const no_oid: RepoFile = .{ .path = @constCast("a"), .size = 10 };
    try testing.expectEqual(Action.skip, decide(no_oid, null, false, 10));
}

test "hf pull: digests are the listing's sha256 and git blob sha1" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "f", .data = "hello\n" });
    var p: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &p);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/f", .{p[0..n]});
    defer testing.allocator.free(path);
    try testing.expect(digestMatches(testing.io, path, .{ .path = @constCast("f"), .size = 6, .oid = @constCast("5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"), .lfs = true }));
    try testing.expect(digestMatches(testing.io, path, .{ .path = @constCast("f"), .size = 6, .oid = @constCast("ce013625030ba8dba906f756967f9e9ca394464a"), .lfs = false }));
    try testing.expect(!digestMatches(testing.io, path, .{ .path = @constCast("f"), .size = 6, .oid = @constCast("ce013625030ba8dba906f756967f9e9ca394464b"), .lfs = false }));
    const files = [_]RepoFile{
        .{ .path = @constCast("f"), .size = 6, .oid = @constCast("ce013625030ba8dba906f756967f9e9ca394464a") },
        .{ .path = @constCast("f"), .size = 6, .oid = @constCast("0000") },
        .{ .path = @constCast("missing"), .size = 6, .oid = @constCast("0000") },
    };
    var match: [3]bool = undefined;
    hashMatches(testing.io, p[0..n], &files, &match, null);
    try testing.expectEqualSlices(bool, &.{ true, false, false }, &match);
    const stopped = std.atomic.Value(bool).init(true);
    hashMatches(testing.io, p[0..n], &files, &match, &stopped);
    try testing.expectEqualSlices(bool, &.{ false, false, false }, &match);
}

test "hf pull: the bytes present are whole files at their listed size and the partials of the rest" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "whole", .data = "12345" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "old", .data = "123" }); // an older revision beside its partial
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "old.partial", .data = "12" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "old.partial.oid", .data = "abc 9\n" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "legacy.partial", .data = "1234" }); // unkeyed: will not resume
    var p: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &p);
    const files = [_]RepoFile{
        .{ .path = @constCast("whole"), .size = 5 },
        .{ .path = @constCast("old"), .size = 9, .oid = @constCast("abc") },
        .{ .path = @constCast("legacy"), .size = 9, .oid = @constCast("def") },
        .{ .path = @constCast("missing"), .size = 7 },
    };
    try testing.expectEqual(@as(u64, 7), bytesPresent(testing.io, p[0..n], &files));
}

test "hf pull: a partial resumes only under the key of its own remote file" {
    const f: RepoFile = .{ .path = @constCast("m"), .size = 4096, .oid = @constCast("abc") };
    try testing.expect(keyMatches("abc 4096\n", f));
    try testing.expect(!keyMatches("abd 4096\n", f));
    try testing.expect(!keyMatches("abc 4095\n", f));
    try testing.expect(!keyMatches("", f));
    try testing.expect(!keyMatches("abc 4096", .{ .path = f.path, .size = 4096 }));
}

test "hf pull: claiming a partial discards an unkeyed or foreign one and keeps its own" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var p: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &p);
    const partial = try std.fmt.allocPrint(testing.allocator, "{s}/m.partial", .{p[0..n]});
    defer testing.allocator.free(partial);
    const f: RepoFile = .{ .path = @constCast("m"), .size = 4, .oid = @constCast("abc") };
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "m.partial", .data = "xx" });
    try claimPartial(testing.io, partial, f);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(testing.io, "m.partial", .{}));
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "m.partial", .data = "xx" });
    try claimPartial(testing.io, partial, f);
    _ = try tmp.dir.statFile(testing.io, "m.partial", .{});
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "m.partial.oid", .data = "other 4\n" });
    try claimPartial(testing.io, partial, f);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(testing.io, "m.partial", .{}));
    dropPartial(testing.io, partial);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(testing.io, "m.partial.oid", .{}));
}

test "hf pull: only files the old manifest named and the repo dropped are removed" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "kept", "dropped", "mine.txt", "outside" }) |n| try tmp.dir.writeFile(testing.io, .{ .sub_path = n, .data = "x" });
    var p: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &p);
    const old: Manifest = .{ .files = &.{
        .{ .path = @constCast("kept"), .size = 1 },
        .{ .path = @constCast("dropped"), .size = 1 },
        .{ .path = @constCast("../outside"), .size = 1 },
    } };
    const listing = [_]RepoFile{.{ .path = @constCast("kept"), .size = 1 }};
    try testing.expectEqual(@as(usize, 1), removeDropped(testing.io, p[0..n], old, &listing));
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(testing.io, "dropped", .{}));
    _ = try tmp.dir.statFile(testing.io, "kept", .{});
    _ = try tmp.dir.statFile(testing.io, "mine.txt", .{});
    try testing.expect(!safeRelative("/etc/passwd") and !safeRelative("a/../b") and safeRelative("mtp/x.safetensors"));
}
