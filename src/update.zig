//! Self-update. `sushi update` replaces a release install with the newest GitHub release; a serving process looks
//! for one at most once a day (`startDailyCheck`). A server never spawns the updater as a child: `/v1/update` and
//! the REPL's `/update` shut the server down, and `relaunchIfRequested` then replaces this process with
//! `sushi update --relaunch -- <argv>`, which updates and replaces itself with the (new or restored) server.

const std = @import("std");
const build_options = @import("build_options");
const log = @import("log.zig");

const Allocator = std.mem.Allocator;
const eql = std.mem.eql;

pub const version = build_options.version;
const repo = "beamivalice/sushi";
pub const asset = "sushi-bin-macos-arm64.tar.gz";
const asset_sha = asset ++ ".sha256";
/// The folder the release tarball unpacks to.
const staged_dir = "sushi-macos-arm64";
const day_s: i64 = 24 * 60 * 60;
const retry_s: i64 = 60 * 60;
const fetch_timeout_s = 30;

/// Set by `/v1/update` and the REPL's `/update`; read once the serve loop has shut down.
pub var relaunch_requested = std.atomic.Value(bool).init(false);

/// Test-only (tests/test_self_update.sh): replaces https://api.github.com so a fake release is served offline.
fn apiBase() []const u8 {
    return if (std.c.getenv("SUSHI_UPDATE_API")) |v| std.mem.span(v) else "https://api.github.com";
}

// ── Versions and releases ───────────────────────────────────────────────

/// A strictly higher SemVer than `current`; text that is not SemVer is never newer.
pub fn isNewer(candidate: []const u8, current: []const u8) bool {
    const c = std.SemanticVersion.parse(candidate) catch return false;
    const r = std.SemanticVersion.parse(current) catch return false;
    return c.order(r) == .gt;
}

pub const Release = struct {
    version: []const u8,
    page: []const u8,
    tarball: []const u8,
    sha256: []const u8,
};

/// The highest SemVer release in a GitHub `/releases` list that carries both assets. Drafts never count; a
/// prerelease (by flag or by version) only with `pre`.
pub fn pickRelease(arena: Allocator, body: []const u8, pre: bool) !?Release {
    const Asset = struct { name: []const u8 = "", browser_download_url: []const u8 = "" };
    const Entry = struct {
        tag_name: []const u8 = "",
        html_url: []const u8 = "",
        draft: bool = false,
        prerelease: bool = false,
        assets: []const Asset = &.{},
    };
    const list = try std.json.parseFromSliceLeaky([]const Entry, arena, body, .{ .ignore_unknown_fields = true });
    var best: ?Release = null;
    for (list) |e| {
        if (e.draft) continue;
        const v = if (std.mem.startsWith(u8, e.tag_name, "v")) e.tag_name[1..] else e.tag_name;
        const sv = std.SemanticVersion.parse(v) catch continue;
        if ((e.prerelease or sv.pre != null) and !pre) continue;
        if (best) |b| if (!isNewer(v, b.version)) continue;
        var r: Release = .{ .version = v, .page = e.html_url, .tarball = "", .sha256 = "" };
        for (e.assets) |a| {
            if (eql(u8, a.name, asset)) r.tarball = a.browser_download_url;
            if (eql(u8, a.name, asset_sha)) r.sha256 = a.browser_download_url;
        }
        if (r.tarball.len > 0 and r.sha256.len > 0) best = r;
    }
    return best;
}

/// The digest of a `shasum -a 256` line (`<hex>  <name>`), lowercased.
pub fn parseShaLine(text: []const u8) ?[64]u8 {
    const t = std.mem.trimStart(u8, text, " \t\r\n");
    if (t.len < 64) return null;
    if (t.len > 64 and !std.ascii.isWhitespace(t[64])) return null;
    var out: [64]u8 = undefined;
    for (t[0..64], &out) |c, *o| {
        if (!std.ascii.isHex(c)) return null;
        o.* = std.ascii.toLower(c);
    }
    return out;
}

pub fn fileSha256Hex(io: std.Io, path: []const u8) ![64]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var rbuf: [64 * 1024]u8 = undefined;
    var fr = f.reader(io, &rbuf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try fr.interface.readSliceShort(&chunk);
        if (n == 0) break;
        h.update(chunk[0..n]);
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

// ── Signatures ──────────────────────────────────────────────────────────

pub const Signature = struct {
    adhoc: bool = false,
    /// Slices the `codesign -dv` text it was parsed from.
    team: ?[]const u8 = null,
};

/// Reads `codesign -dv --verbose=2` output.
pub fn parseCodesign(text: []const u8) Signature {
    var s: Signature = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r ");
        if (eql(u8, line, "Signature=adhoc")) s.adhoc = true;
        if (std.mem.startsWith(u8, line, "TeamIdentifier=")) {
            const t = line["TeamIdentifier=".len..];
            if (t.len > 0 and !eql(u8, t, "not set")) s.team = t;
        }
    }
    return s;
}

/// Why the new build may not replace the running one: a Developer ID install only takes a build of its own team.
pub fn signatureRefusal(running: Signature, incoming: Signature, incoming_verifies: bool) ?[]const u8 {
    if (!incoming_verifies) return "the new build fails codesign --verify --strict";
    const team = running.team orelse return null;
    const new_team = incoming.team orelse return "the running sushi is Developer ID signed and the new build is not";
    if (!eql(u8, team, new_team)) return "the new build is signed by another team";
    return null;
}

// ── The install ─────────────────────────────────────────────────────────

/// A binary under `zig-out/bin`, or inside a checkout that holds `build.zig`, was built from source.
pub fn isSourceBuild(io: std.Io, exe_dir: []const u8) bool {
    if (std.mem.endsWith(u8, exe_dir, "/zig-out/bin")) return true;
    var d: ?[]const u8 = exe_dir;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    while (d) |dir| : (d = std.fs.path.dirname(dir)) {
        const git = std.fmt.bufPrint(&buf, "{s}/.git", .{dir}) catch return false;
        std.Io.Dir.cwd().access(io, git, .{}) catch continue;
        const zig = std.fmt.bufPrint(&buf, "{s}/build.zig", .{dir}) catch return false;
        std.Io.Dir.cwd().access(io, zig, .{}) catch continue;
        return true;
    }
    return false;
}

pub const brew_upgrade = "brew upgrade sushi";
const brew_notice = "sushi was installed with Homebrew; run: " ++ brew_upgrade;

/// A real directory inside a Homebrew keg, `<prefix>/Cellar/sushi/<version>/…`: brew owns the files.
pub fn isHomebrewKeg(real_dir: []const u8) bool {
    return std.mem.indexOf(u8, real_dir, "/Cellar/sushi/") != null;
}

/// Set once from the running binary's real path by `startDailyCheck` or `cmdUpdate`.
var homebrew = std.atomic.Value(bool).init(false);

pub fn homebrewInstall() bool {
    return homebrew.load(.acquire);
}

fn upgradeCommand() []const u8 {
    return if (homebrewInstall()) brew_upgrade else "sushi update";
}

/// Why the install at `dir` (the real directory of the running binary) cannot replace itself, or null.
pub fn installRefusal(io: std.Io, dir: []const u8, buf: []u8) ?[]const u8 {
    const cwd = std.Io.Dir.cwd();
    if (isHomebrewKeg(dir)) return brew_notice;
    if (isSourceBuild(io, dir)) return "built from source: git pull and rebuild";
    if (std.mem.indexOf(u8, dir, ".app/Contents/") != null) return "part of an app bundle: update the app";
    const lib = std.fmt.bufPrint(buf, "{s}/lib", .{dir}) catch return "install path too long";
    cwd.access(io, lib, .{}) catch
        return std.fmt.bufPrint(buf, "{s} is not a release install (no lib/ beside sushi)", .{dir}) catch "not a release install";
    cwd.access(io, dir, .{ .write = true }) catch
        return std.fmt.bufPrint(buf, "{s} is not writable", .{dir}) catch "the install is not writable";
    const parent = std.fs.path.dirname(dir) orelse return "the install has no parent folder";
    cwd.access(io, parent, .{ .write = true }) catch
        return std.fmt.bufPrint(buf, "{s} is not writable", .{parent}) catch "the install's folder is not writable";
    return null;
}

/// The release tarball's entries; `.DS_Store` is Finder's.
const release_entries = [_][]const u8{ "sushi", "sushibar", "lib", "LICENSE", "LICENSE-APACHE-2.0", "NOTICE", "guest.json", ".DS_Store" };

/// An entry of `dir` that neither the release layout nor `incoming` has: the swap would carry it away, so a folder
/// holding anything else (a shared bin/) is never swapped.
pub fn strayEntry(io: std.Io, dir: []const u8, incoming: []const u8, name_buf: []u8) !?[]const u8 {
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var in = try std.Io.Dir.cwd().openDir(io, incoming, .{});
    defer in.close(io);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        const known = for (release_entries) |r| {
            if (eql(u8, r, e.name)) break true;
        } else false;
        if (known) continue;
        in.access(io, e.name, .{ .follow_symlinks = false }) catch {
            const n = @min(e.name.len, name_buf.len);
            @memcpy(name_buf[0..n], e.name[0..n]);
            return name_buf[0..n];
        };
    }
    return null;
}

extern "c" fn renamex_np(from: [*:0]const u8, to: [*:0]const u8, flags: c_uint) c_int;
const RENAME_SWAP: c_uint = 0x2;

/// Exchanges the directories at `a` and `b` in one atomic step (APFS, HFS+); elsewhere `swapByRenames`.
pub fn swapPaths(io: std.Io, a: []const u8, b: []const u8) !void {
    var za: [std.fs.max_path_bytes:0]u8 = undefined;
    var zb: [std.fs.max_path_bytes:0]u8 = undefined;
    const pa = try std.mem.printSentinel(&za, "{s}", .{a}, 0);
    const pb = try std.mem.printSentinel(&zb, "{s}", .{b}, 0);
    const rc = renamex_np(pa.ptr, pb.ptr, RENAME_SWAP);
    switch (std.posix.errno(rc)) {
        .SUCCESS => {},
        .OPNOTSUPP, .INVAL => return swapByRenames(io, a, b),
        .NOENT => return error.FileNotFound,
        .ACCES, .PERM => return error.AccessDenied,
        else => |e| {
            log.debug("[update] renamex_np: {t}\n", .{e});
            return error.SwapFailed;
        },
    }
}

/// Three renames through `<a>.swap`, each failure undoing the ones before it.
pub fn swapByRenames(io: std.Io, a: []const u8, b: []const u8) !void {
    var tbuf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp = try std.fmt.bufPrint(&tbuf, "{s}.swap", .{a});
    try std.Io.Dir.renameAbsolute(a, tmp, io);
    std.Io.Dir.renameAbsolute(b, a, io) catch |err| {
        std.Io.Dir.renameAbsolute(tmp, a, io) catch {};
        return err;
    };
    std.Io.Dir.renameAbsolute(tmp, b, io) catch |err| {
        std.Io.Dir.renameAbsolute(a, b, io) catch {};
        std.Io.Dir.renameAbsolute(tmp, a, io) catch {};
        return err;
    };
}

fn previousPath(buf: []u8, dir: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}.previous", .{dir});
}

/// After `swapPaths(dir, incoming)` succeeded and the new build runs: the old install, now at `incoming`, becomes
/// the one `<dir>.previous`.
pub fn keepPrevious(io: std.Io, dir: []const u8, incoming: []const u8) !void {
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const prev = try previousPath(&pbuf, dir);
    std.Io.Dir.cwd().deleteTree(io, prev) catch {};
    try std.Io.Dir.renameAbsolute(incoming, prev, io);
}

// ── The daily check ─────────────────────────────────────────────────────

/// `~/.sushi/update-check.json`.
pub const Cache = struct {
    /// Unix seconds of the last answer from GitHub; 0 = never.
    checked_at: i64 = 0,
    latest: []const u8 = "",
    url: []const u8 = "",
    etag: []const u8 = "",
    /// Why the last update failed; empty after a success.
    @"error": []const u8 = "",
};

pub fn due(c: Cache, now: i64) bool {
    return now < c.checked_at or now - c.checked_at >= day_s;
}

/// Folds one answer from the releases endpoint into the cache: 200 takes the newest release and its ETag, 304 keeps
/// the cached one, both stamp the check. False (cache untouched) for anything else.
pub fn applyFetch(c: *Cache, arena: Allocator, status: u16, body: []const u8, etag: []const u8, now: i64) !bool {
    switch (status) {
        200 => {
            const r = try pickRelease(arena, body, false);
            c.latest = if (r) |x| x.version else "";
            c.url = if (r) |x| x.page else "";
            c.etag = etag;
        },
        304 => {},
        else => return false,
    }
    c.checked_at = now;
    return true;
}

fn cachePath(buf: []u8) ?[]const u8 {
    const home = std.c.getenv("HOME") orelse return null;
    return std.fmt.bufPrint(buf, "{s}/.sushi/update-check.json", .{std.mem.span(home)}) catch null;
}

pub fn readCache(arena: Allocator, io: std.Io, path: []const u8) Cache {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(64 * 1024)) catch return .{};
    return std.json.parseFromSliceLeaky(Cache, arena, text, .{ .ignore_unknown_fields = true }) catch .{};
}

pub fn writeCache(arena: Allocator, io: std.Io, path: []const u8, c: Cache) !void {
    return writeJsonAtomic(arena, io, path, c);
}

var tmp_seq = std.atomic.Value(u32).init(0);

/// Written beside under a name no other writer (thread or process) uses, then renamed over, so a reader never sees
/// half a file and the last rename wins whole.
pub fn writeJsonAtomic(arena: Allocator, io: std.Io, path: []const u8, value: anytype) !void {
    if (std.fs.path.dirname(path)) |d| try std.Io.Dir.cwd().createDirPath(io, d);
    const text = try std.json.Stringify.valueAlloc(arena, value, .{});
    const tmp = try std.fmt.allocPrint(arena, "{s}.{d}-{d}.tmp", .{ path, std.c.getpid(), tmp_seq.fetchAdd(1, .monotonic) });
    errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = text });
    try std.Io.Dir.renameAbsolute(tmp, path, io);
}

/// What `/props` reports, published by the check and read by connection threads.
var published_mu: std.c.pthread_mutex_t = .{};
var published: Cache = .{};
var published_arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);

fn publish(c: Cache) void {
    var next: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    const a = next.allocator();
    const copy: Cache = .{
        .checked_at = c.checked_at,
        .latest = a.dupe(u8, c.latest) catch "",
        .url = a.dupe(u8, c.url) catch "",
        .@"error" = a.dupe(u8, c.@"error") catch "",
    };
    _ = std.c.pthread_mutex_lock(&published_mu);
    defer _ = std.c.pthread_mutex_unlock(&published_mu);
    published_arena.deinit();
    published_arena = next;
    published = copy;
}

fn nonEmpty(s: []const u8) ?[]const u8 {
    return if (s.len == 0) null else s;
}

/// `,"update":{...}` for `/props`: null fields until a check has answered.
pub fn propsJson(allocator: Allocator) ![]u8 {
    _ = std.c.pthread_mutex_lock(&published_mu);
    defer _ = std.c.pthread_mutex_unlock(&published_mu);
    const p = published;
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try out.writer.writeAll(",\"update\":");
    try std.json.Stringify.value(.{
        .current = version,
        .latest = nonEmpty(p.latest),
        .available = isNewer(p.latest, version),
        .checked_at = if (p.checked_at > 0) @as(?i64, p.checked_at) else null,
        .url = nonEmpty(p.url),
        .@"error" = nonEmpty(p.@"error"),
        // What the chat page tells the user to run instead of offering its button.
        .command = if (homebrewInstall()) @as(?[]const u8, brew_upgrade) else null,
    }, .{}, &out.writer);
    return out.toOwnedSlice();
}

/// The newer release the last check found, or null.
pub fn availableVersion(buf: []u8) ?[]const u8 {
    _ = std.c.pthread_mutex_lock(&published_mu);
    defer _ = std.c.pthread_mutex_unlock(&published_mu);
    if (!isNewer(published.latest, version)) return null;
    const n = @min(buf.len, published.latest.len);
    @memcpy(buf[0..n], published.latest[0..n]);
    return buf[0..n];
}

pub const CheckChoice = struct { on: bool, source: []const u8 };

/// `--no-update-check` > a source build (never checks) > the settings file's `update_check` > on.
pub fn checkChoice(flag_off: bool, source_build: bool, file: ?bool) CheckChoice {
    if (flag_off) return .{ .on = false, .source = "--no-update-check" };
    if (source_build) return .{ .on = false, .source = "source build" };
    if (file) |f| return .{ .on = f, .source = "model-settings.json" };
    return .{ .on = true, .source = "default" };
}

/// The settings page flips this; the check loop reads it before each look.
var auto_on = std.atomic.Value(bool).init(true);

pub fn setAutoCheck(on: bool) void {
    auto_on.store(on, .release);
}

pub fn autoCheckOn() bool {
    return auto_on.load(.acquire);
}

/// Whether a loop wakes into a look; a `next_at` more than a day ahead is a clock set back.
fn looksNow(on: bool, now: i64, next_at: i64) bool {
    return on and (now >= next_at or next_at - now > day_s);
}

/// Logs the resolved choice once and checks on a detached thread: never the inference thread, never in the way of
/// the boot, silent on failure. `file_on` is the settings file's `update_check`; the thread idles while it is off.
pub fn startDailyCheck(io: std.Io, flag_off: bool, file_on: ?bool) void {
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    const exe_dir = selfDir(io, &ebuf) orelse "";
    homebrew.store(isHomebrewKeg(exe_dir), .release);
    const choice = checkChoice(flag_off, isSourceBuild(io, exe_dir), file_on);
    if (choice.on or !std.mem.eql(u8, choice.source, "source build")) log.info("[update] daily check {s} ({s})\n", .{ if (choice.on) "on" else "off", choice.source });
    setAutoCheck(choice.on);
    if (flag_off or std.mem.eql(u8, choice.source, "source build")) return;
    const t = std.Thread.spawn(.{}, checkLoop, .{io}) catch |err| {
        log.debug("[update] check thread: {t}\n", .{err});
        return;
    };
    t.detach();
}

/// Wakes every minute so a toggle from the settings page takes hold within it.
fn checkLoop(io: std.Io) void {
    var next_at: i64 = 0;
    while (true) {
        const now = std.Io.Timestamp.now(io, .real).toSeconds();
        if (looksNow(autoCheckOn(), now, next_at)) next_at = now + @max(checkOnce(io), 60);
        const ts = std.c.timespec{ .sec = 60, .nsec = 0 };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// One scheduled look: GitHub is asked only when the cache is a day old. Returns the seconds until the next look.
fn checkOnce(io: std.Io) i64 {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = cachePath(&pbuf) orelse return day_s;
    var c = readCache(arena, io, path);
    const now = std.Io.Timestamp.now(io, .real).toSeconds();
    var wait = c.checked_at + day_s - now;
    if (due(c, now)) {
        wait = retry_s;
        if (fetchReleases(arena, io, c.etag)) |f| {
            if (applyFetch(&c, arena, f.status, f.body, f.etag, now) catch false) {
                writeCache(arena, io, path, c) catch |err| log.debug("[update] cache write: {t}\n", .{err});
                wait = day_s;
            } else log.debug("[update] check: HTTP {d}\n", .{f.status});
        } else |err| log.debug("[update] check failed: {t}\n", .{err});
    }
    publish(c);
    if (isNewer(c.latest, version)) log.info("sushi {s} is available: run `{s}`\n", .{ c.latest, upgradeCommand() });
    return wait;
}

/// A fresh check for `sushi update --check` and the REPL's `/update`, recorded in the cache like the daily one.
pub fn checkNow(arena: Allocator, io: std.Io, pre: bool) !?Release {
    const f = fetchReleases(arena, io, "") catch |err| return fail("could not reach the release server ({s})", .{@errorName(err)});
    if (f.status != 200) return fail("GitHub answered HTTP {d} for the release list", .{f.status});
    const r = try pickRelease(arena, f.body, pre);
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (cachePath(&pbuf)) |path| {
        var c = readCache(arena, io, path);
        _ = try applyFetch(&c, arena, f.status, f.body, f.etag, std.Io.Timestamp.now(io, .real).toSeconds());
        writeCache(arena, io, path, c) catch {};
        publish(c);
    }
    return r;
}

const Fetched = struct { status: u16, body: []u8, etag: []u8 };

/// GET the release list, bounded on the wall clock so a stalled connection cannot hold an updater whose server is
/// already down.
fn fetchReleases(arena: Allocator, io: std.Io, etag: []const u8) !Fetched {
    const url = try std.fmt.allocPrint(arena, "{s}/repos/{s}/releases?per_page=100", .{ apiBase(), repo });
    const U = union(enum) { got: anyerror!Fetched, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(io, &buf);
    sel.concurrent(.got, fetchNow, .{ arena, io, url, etag }) catch return fetchNow(arena, io, url, etag);
    sel.concurrent(.timer, std.Io.sleep, .{ io, std.Io.Duration.fromSeconds(fetch_timeout_s), .awake }) catch {};
    const first = sel.await() catch |err| {
        while (sel.cancel()) |_| {}
        return err;
    };
    while (sel.cancel()) |_| {}
    return switch (first) {
        .got => |r| r,
        .timer => error.Timeout,
    };
}

fn fetchNow(arena: Allocator, io: std.Io, url: []const u8, etag: []const u8) anyerror!Fetched {
    var client: std.http.Client = .{ .allocator = arena, .io = io };
    defer client.deinit();
    var headers: [2]std.http.Header = .{
        .{ .name = "accept", .value = "application/vnd.github+json" },
        .{ .name = "if-none-match", .value = etag },
    };
    var req = try client.request(.GET, try std.Uri.parse(url), .{
        .keep_alive = false,
        .headers = .{
            .user_agent = .{ .override = "sushi/" ++ version },
            .accept_encoding = .{ .override = "identity" },
        },
        .extra_headers = headers[0 .. if (etag.len > 0) 2 else 1],
    });
    defer req.deinit();
    try req.sendBodiless();
    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    var tag: []u8 = &.{};
    var it = response.head.iterateHeaders();
    while (it.next()) |h| {
        if (std.ascii.eqlIgnoreCase(h.name, "etag")) tag = try arena.dupe(u8, h.value);
    }
    const status: u16 = @backingInt(response.head.status);
    var transfer_buf: [64]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    var body = std.ArrayList(u8).empty;
    reader.appendRemaining(arena, &body, .limited(8 * 1024 * 1024)) catch |err| switch (err) {
        error.ReadFailed => return response.bodyErr() orelse error.ReadFailed,
        else => |e| return e,
    };
    return .{ .status = status, .body = body.items, .etag = tag };
}

// ── The /v1/update guard ────────────────────────────────────────────────

pub const Ask = struct {
    bind_host: []const u8,
    port: u16,
    peer_loopback: bool,
    origin: ?[]const u8,
    key_set: bool,
    key_ok: bool,
    /// Started with `--parent-pid`: the host owns this engine's lifecycle and its updates.
    host_managed: bool,
    /// A request is decoding or waiting, or a model is loading.
    busy: bool,
    /// `installRefusal` for the running binary.
    install: ?[]const u8,
    available: bool,
};

pub const Refusal = struct { status: []const u8, code: u16, kind: []const u8, message: []const u8 };

fn forbidden(message: []const u8) Refusal {
    return .{ .status = "403 Forbidden", .code = 403, .kind = "update_forbidden", .message = message };
}

fn conflict(kind: []const u8, message: []const u8) Refusal {
    return .{ .status = "409 Conflict", .code = 409, .kind = kind, .message = message };
}

pub fn isLoopbackHost(host: []const u8) bool {
    return eql(u8, host, "localhost") or std.mem.startsWith(u8, host, "127.");
}

/// The page's own origin: `http://<bind>:<port>`, with 127.0.0.1 and localhost interchangeable.
pub fn originMatches(origin: []const u8, host: []const u8, port: u16) bool {
    var buf: [64]u8 = undefined;
    const exact = std.fmt.bufPrint(&buf, "http://{s}:{d}", .{ host, port }) catch return false;
    if (eql(u8, origin, exact)) return true;
    if (!eql(u8, host, "127.0.0.1") and !eql(u8, host, "localhost")) return false;
    for ([_][]const u8{ "127.0.0.1", "localhost" }) |h| {
        const alt = std.fmt.bufPrint(&buf, "http://{s}:{d}", .{ h, port }) catch return false;
        if (eql(u8, origin, alt)) return true;
    }
    return false;
}

/// The first reason `POST /v1/update` is refused; the key is checked from loopback too.
pub fn guard(a: Ask) ?Refusal {
    if (!isLoopbackHost(a.bind_host)) return forbidden("updates from the chat page need a loopback bind (--host 127.0.0.1); run `sushi update` on the server");
    if (!a.peer_loopback) return forbidden("updates are accepted from this Mac only");
    if (a.key_set and !a.key_ok) return .{ .status = "401 Unauthorized", .code = 401, .kind = "authentication_error", .message = "missing or invalid API key" };
    const origin = a.origin orelse return forbidden("the request carries no Origin: updates come from the chat page");
    if (!originMatches(origin, a.bind_host, a.port)) return forbidden("the request's Origin is not this server's chat page");
    if (a.host_managed) return forbidden("this server runs under a host (--parent-pid): update the host app");
    if (a.busy) return conflict("update_busy", "a request or a model load is in flight: try again when it finishes");
    if (a.install) |why| return conflict("update_unavailable", why);
    if (!a.available) return conflict("update_unavailable", "no newer release is known");
    return null;
}

// ── Processes ───────────────────────────────────────────────────────────

extern "c" fn posix_spawnattr_setsigmask(attr: *std.c.posix_spawnattr_t, mask: *const std.c.sigset_t) c_int;
extern "c" fn proc_listallpids(buffer: ?*anyopaque, buffersize: c_int) c_int;
extern "c" fn proc_pidpath(pid: c_int, buffer: [*]u8, buffersize: u32) c_int;

/// Where a tool's output goes: stdout to `path`, stderr there too or to /dev/null.
const Capture = struct { path: []const u8, stderr: bool = false };

/// posix_spawn, never fork (std.process.spawn forks on macOS, which would copy a server's whole MLX mapping).
/// `replace` runs argv IN PLACE of this process (POSIX_SPAWN_SETEXEC); every fd but 0-2 is closed, so a listening
/// socket never outlives its server.
fn spawn(arena: Allocator, argv: []const []const u8, replace: bool, capture: ?Capture) !std.c.pid_t {
    const argv_z = try arena.allocSentinel(?[*:0]const u8, argv.len, null);
    for (argv, 0..) |a, i| argv_z[i] = (try arena.dupeSentinel(u8, a, 0)).ptr;
    var attr: std.c.posix_spawnattr_t = undefined;
    if (std.c.posix_spawnattr_init(&attr) != 0) return error.SpawnFailed;
    defer _ = std.c.posix_spawnattr_destroy(&attr);
    var actions: std.c.posix_spawn_file_actions_t = undefined;
    if (std.c.posix_spawn_file_actions_init(&actions) != 0) return error.SpawnFailed;
    defer _ = std.c.posix_spawn_file_actions_destroy(&actions);
    // Dispositions pass through as with a plain exec: `nohup`'s ignored SIGHUP must outlive the update.
    const none: std.c.sigset_t = 0;
    _ = posix_spawnattr_setsigmask(&attr, &none);
    _ = std.c.posix_spawnattr_setflags(&attr, .{ .SETEXEC = replace, .CLOEXEC_DEFAULT = true, .SETSIGMASK = true });
    _ = std.c.posix_spawn_file_actions_addinherit_np(&actions, 0);
    if (capture) |c| {
        const flags: std.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true };
        _ = std.c.posix_spawn_file_actions_addopen(&actions, 1, try arena.dupeSentinel(u8, c.path, 0), @bitCast(flags), 0o644);
        if (c.stderr) {
            _ = std.c.posix_spawn_file_actions_adddup2(&actions, 1, 2);
        } else {
            _ = std.c.posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", @bitCast(flags), 0o644);
        }
    } else {
        _ = std.c.posix_spawn_file_actions_addinherit_np(&actions, 1);
        _ = std.c.posix_spawn_file_actions_addinherit_np(&actions, 2);
    }
    var pid: std.c.pid_t = undefined;
    const rc = std.c.posix_spawn(&pid, argv_z[0].?, &actions, &attr, argv_z.ptr, std.c.environ);
    if (rc != 0) {
        log.err("[update] cannot run {s}: {t}\n", .{ argv[0], @as(std.c.E, @enumFromInt(rc)) });
        return error.SpawnFailed;
    }
    return pid;
}

/// Runs a tool to completion; its exit code, 255 when a signal ended it.
fn runTool(arena: Allocator, argv: []const []const u8, capture: ?Capture) !u8 {
    const pid = try spawn(arena, argv, false, capture);
    var status: c_int = 0;
    while (true) {
        const rc = std.c.waitpid(pid, &status, 0);
        if (rc >= 0) break;
        if (std.posix.errno(rc) != .INTR) return error.WaitFailed;
    }
    const s: u32 = @bitCast(status);
    return if (std.c.W.IFEXITED(s)) std.c.W.EXITSTATUS(s) else 255;
}

/// Replaces this process; returns only when the exec could not happen, after logging why.
fn execInPlace(arena: Allocator, argv: []const []const u8) void {
    _ = spawn(arena, argv, true, null) catch {};
}

/// Whether a process running `exe` stops the swap of `dir`: anything from the install but `sushibar`, which
/// restarts itself on the new version and never holds `sushi` open.
fn blocksUpdate(dir: []const u8, exe: []const u8) bool {
    if (exe.len <= dir.len or !std.mem.startsWith(u8, exe, dir) or exe[dir.len] != '/') return false;
    return !eql(u8, std.fs.path.basename(exe), "sushibar");
}

/// Another process running a binary from `dir`, or null.
fn otherProcessIn(dir: []const u8) ?c_int {
    var pids: [8192]c_int = undefined;
    const n = proc_listallpids(&pids, @sizeOf(@TypeOf(pids)));
    if (n <= 0) return null;
    const self = std.c.getpid();
    var buf: [4096]u8 = undefined;
    for (pids[0..@min(@as(usize, @intCast(n)), pids.len)]) |pid| {
        if (pid == self or pid <= 0) continue;
        const len = proc_pidpath(pid, &buf, buf.len);
        if (len <= 0) continue;
        if (blocksUpdate(dir, buf[0..@intCast(len)])) return pid;
    }
    return null;
}

/// The real directory of the running binary.
fn selfDir(io: std.Io, buf: []u8) ?[]const u8 {
    const n = std.process.executablePath(io, buf) catch return null;
    return std.fs.path.dirname(buf[0..n]);
}

pub fn selfInstallRefusal(io: std.Io, path_buf: []u8, buf: []u8) ?[]const u8 {
    const dir = selfDir(io, path_buf) orelse return "cannot find the running binary";
    return installRefusal(io, dir, buf);
}

/// When `/v1/update` or `/update` asked for it, replaces this (already shut down) server with
/// `sushi update --relaunch -- <argv>`. An exec that cannot happen exits non-zero rather than looking like a
/// clean stop.
pub fn relaunchIfRequested(allocator: Allocator, io: std.Io, args: []const []const u8) void {
    if (!relaunch_requested.load(.acquire)) return;
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    const arena = arena_state.allocator();
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    const n = std.process.executablePath(io, &ebuf) catch {
        log.err("[update] cannot find the running binary; not relaunching\n", .{});
        std.process.exit(1);
    };
    var argv = std.ArrayList([]const u8).empty;
    argv.appendSlice(arena, &.{ ebuf[0..n], "update", "--relaunch", "--" }) catch std.process.exit(1);
    if (args.len > 1) argv.appendSlice(arena, args[1..]) catch std.process.exit(1);
    execInPlace(arena, argv.items);
    std.process.exit(1);
}

// ── `sushi update` ──────────────────────────────────────────────────────

var fail_buf: [512]u8 = undefined;
var fail_msg: []const u8 = "";

/// Records and logs why the update stopped.
fn fail(comptime fmt: []const u8, args: anytype) error{UpdateFailed} {
    fail_msg = std.fmt.bufPrint(&fail_buf, fmt, args) catch fmt;
    log.err("update: {s}\n", .{fail_msg});
    return error.UpdateFailed;
}

const Options = struct {
    check: bool = false,
    pre: bool = false,
    force: bool = false,
    rollback: bool = false,
    relaunch: ?[]const []const u8 = null,
};

const usage =
    \\Usage: sushi update [--check] [--pre] [--force] [--rollback]
    \\
    \\Replaces this install with the newest release from GitHub, after checking
    \\its SHA-256, its code signature and that it runs; the old install is kept
    \\as <install>.previous.
    \\  --check     Only report whether a newer release exists
    \\  --pre       Consider prereleases too
    \\  --force     Update even while a sushi from this install is running
    \\  --rollback  Swap <install>.previous back in
    \\
;

pub fn cmdUpdate(allocator: Allocator, io: std.Io, args: []const []const u8) !void {
    var opts: Options = .{};
    for (args, 0..) |a, i| {
        if (eql(u8, a, "--check")) {
            opts.check = true;
        } else if (eql(u8, a, "--pre")) {
            opts.pre = true;
        } else if (eql(u8, a, "--force")) {
            opts.force = true;
        } else if (eql(u8, a, "--rollback")) {
            opts.rollback = true;
        } else if (eql(u8, a, "--relaunch") and i + 1 < args.len and eql(u8, args[i + 1], "--")) {
            // Internal and absent from the usage on purpose: the updater re-executes itself with this to restart the server.
            opts.relaunch = args[i + 2 ..];
            break;
        } else if (eql(u8, a, "--help") or eql(u8, a, "-h")) {
            var out = std.Io.File.stdout().writerStreaming(io, &.{});
            out.interface.writeAll(usage) catch {};
            return;
        } else {
            log.err("unrecognized argument '{s}' — see sushi update --help\n", .{a});
            std.process.exit(1);
        }
    }
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    if (std.c.getenv("HOME")) |home| {
        const p = try std.fmt.allocPrint(arena, "{s}/.sushi/logs/update.log", .{std.mem.span(home)});
        log.openFile(p, log.default_max_bytes) catch {};
    }
    defer log.closeFile();
    logStart(io);
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    homebrew.store(isHomebrewKeg(selfDir(io, &ebuf) orelse ""), .release);
    // Exit 0, as for "up to date": the install is healthy and brew is the way to update it.
    if (homebrewInstall() and !opts.check and opts.relaunch == null) return log.info(brew_notice ++ "\n", .{});

    const result = if (opts.check) check(arena, io, opts) else if (opts.rollback) rollback(arena, io, opts) else update(arena, io, opts);
    if (opts.relaunch) |argv| relaunch(arena, io, argv, result);
    result catch std.process.exit(1);
}

fn logStart(io: std.Io) void {
    const secs: u64 = @intCast(@max(0, std.Io.Timestamp.now(io, .real).toSeconds()));
    const es: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const ds = es.getDaySeconds();
    log.info("── sushi {s} update, {d}-{d:0>2}-{d:0>2} {d:0>2}:{d:0>2} UTC\n", .{ version, yd.year, md.month.numeric(), md.day_index + 1, ds.getHoursIntoDay(), ds.getMinutesIntoHour() });
}

fn check(arena: Allocator, io: std.Io, opts: Options) !void {
    const r = try checkNow(arena, io, opts.pre) orelse return fail("no release carries {s}", .{asset});
    if (isNewer(r.version, version)) {
        log.info("sushi {s} is available (this is {s}): run `{s}`\n  {s}\n", .{ r.version, version, upgradeCommand(), r.page });
    } else {
        log.info("sushi {s} is up to date (latest release {s})\n", .{ version, r.version });
    }
}

const Install = struct {
    exe: []const u8,
    dir: []const u8,
    /// `<parent>/.<name>.update`: downloads and unpacking stay on the install's volume.
    work: []const u8,
};

fn locate(arena: Allocator, io: std.Io, opts: Options) !Install {
    const exe = std.process.executablePathAlloc(io, arena) catch return fail("cannot find the running binary", .{});
    const dir = std.fs.path.dirname(exe) orelse return fail("the binary has no folder", .{});
    var rbuf: [512]u8 = undefined;
    if (installRefusal(io, dir, &rbuf)) |why| return fail("{s}", .{why});
    if (opts.relaunch == null and !opts.force) {
        if (otherProcessIn(dir)) |pid| return fail("sushi (pid {d}) is running from {s}: stop it first, or pass --force", .{ pid, dir });
    }
    const parent = std.fs.path.dirname(dir).?;
    const work = try std.fmt.allocPrint(arena, "{s}/.{s}.update", .{ parent, std.fs.path.basename(dir) });
    std.Io.Dir.cwd().createDirPath(io, work) catch return fail("cannot create {s}", .{work});
    return .{ .exe = exe, .dir = dir, .work = work };
}

fn update(arena: Allocator, io: std.Io, opts: Options) !void {
    const inst = try locate(arena, io, opts);
    const r = try checkNow(arena, io, opts.pre) orelse return fail("no release carries {s}", .{asset});
    if (!isNewer(r.version, version)) {
        log.info("sushi {s} is up to date (latest release {s})\n", .{ version, r.version });
        return;
    }
    log.info("found sushi {s} (this is {s}): {s}\n", .{ r.version, version, r.page });

    const tarball = try std.fmt.allocPrint(arena, "{s}/sushi-{s}.tar.gz", .{ inst.work, r.version });
    const sha_file = try std.fmt.allocPrint(arena, "{s}/{s}", .{ inst.work, asset_sha });
    // The digest is always fetched whole: a leftover would resume into a wrong one.
    for ([_][]const u8{ sha_file, try std.fmt.allocPrint(arena, "{s}.partial", .{sha_file}) }) |p| std.Io.Dir.cwd().deleteFile(io, p) catch {};
    try download(arena, io, r.sha256, sha_file);
    std.Io.Dir.cwd().access(io, tarball, .{}) catch try download(arena, io, r.tarball, tarball);

    const sha_text = std.Io.Dir.cwd().readFileAlloc(io, sha_file, arena, .limited(4096)) catch return fail("cannot read {s}", .{sha_file});
    const want = parseShaLine(sha_text) orelse return fail("{s} holds no SHA-256", .{asset_sha});
    const got = fileSha256Hex(io, tarball) catch return fail("cannot read {s}", .{tarball});
    if (!eql(u8, &want, &got)) {
        std.Io.Dir.cwd().deleteFile(io, tarball) catch {};
        return fail("SHA-256 mismatch: the download is corrupt (deleted; run again)", .{});
    }
    log.info("sha256 ok\n", .{});
    // A verified tarball that fails a later step would fail the same way from the cache.
    errdefer std.Io.Dir.cwd().deleteFile(io, tarball) catch {};

    const unpacked = try std.fmt.allocPrint(arena, "{s}/unpacked", .{inst.work});
    std.Io.Dir.cwd().deleteTree(io, unpacked) catch {};
    std.Io.Dir.cwd().createDirPath(io, unpacked) catch return fail("cannot create {s}", .{unpacked});
    if (try runTool(arena, &.{ "/usr/bin/tar", "-xzf", tarball, "-C", unpacked }, null) != 0) return fail("cannot unpack {s}", .{tarball});
    const incoming = try std.fmt.allocPrint(arena, "{s}/{s}", .{ unpacked, staged_dir });
    const new_bin = try std.fmt.allocPrint(arena, "{s}/sushi", .{incoming});
    std.Io.Dir.cwd().access(io, new_bin, .{ .execute = true }) catch return fail("the release holds no {s}/sushi", .{staged_dir});
    var nbuf: [256]u8 = undefined;
    if (strayEntry(io, inst.dir, incoming, &nbuf) catch return fail("cannot list {s}", .{inst.dir})) |name|
        return fail("{s} holds '{s}', which is not part of sushi: keep the install in a folder of its own", .{ inst.dir, name });

    try verifySignature(arena, io, inst, new_bin);

    swapPaths(io, inst.dir, incoming) catch |err| return fail("cannot swap in the new install: {t}", .{err});
    if (try runsAs(arena, io, inst, r.version)) |why| {
        swapPaths(io, inst.dir, incoming) catch |err| return fail("the new build {s}, and restoring the old install failed ({t}): it is at {s}", .{ why, err, incoming });
        return fail("the new build {s}; the old install is back", .{why});
    }
    keepPrevious(io, inst.dir, incoming) catch |err| log.warn("update: could not keep the old install as .previous: {t}\n", .{err});
    std.Io.Dir.cwd().deleteTree(io, inst.work) catch {};
    log.info("installed; the previous install is at {s}.previous\n", .{inst.dir});
    log.info("updated {s} -> {s}\n", .{ version, r.version });
}

fn download(arena: Allocator, io: std.Io, url: []const u8, dest: []const u8) !void {
    const partial = try std.fmt.allocPrint(arena, "{s}.partial", .{dest});
    log.info("downloading {s}\n", .{url});
    const tty = std.Io.File.stderr().isTty(io) catch false;
    const argv = [_][]const u8{ "/usr/bin/curl", "-fL", "--retry", "3", "--retry-delay", "2", "--speed-limit", "1024", "--speed-time", "60", "-C", "-", "-o", partial, if (tty) "--progress-bar" else "-sS", url };
    if (try runTool(arena, &argv, null) != 0) return fail("download failed: {s} (a partial download resumes on the next run)", .{url});
    std.Io.Dir.renameAbsolute(partial, dest, io) catch return fail("cannot move {s} into place", .{partial});
}

fn codesignOf(arena: Allocator, inst: Install, io: std.Io, bin: []const u8) !Signature {
    const out = try std.fmt.allocPrint(arena, "{s}/codesign.txt", .{inst.work});
    _ = try runTool(arena, &.{ "/usr/bin/codesign", "-dv", "--verbose=2", bin }, .{ .path = out, .stderr = true });
    const text = std.Io.Dir.cwd().readFileAlloc(io, out, arena, .limited(64 * 1024)) catch "";
    return parseCodesign(text);
}

fn verifySignature(arena: Allocator, io: std.Io, inst: Install, new_bin: []const u8) !void {
    const running = try codesignOf(arena, inst, io, inst.exe);
    const incoming = try codesignOf(arena, inst, io, new_bin);
    const out = try std.fmt.allocPrint(arena, "{s}/verify.txt", .{inst.work});
    const verifies = try runTool(arena, &.{ "/usr/bin/codesign", "--verify", "--strict", new_bin }, .{ .path = out, .stderr = true }) == 0;
    if (signatureRefusal(running, incoming, verifies)) |why| return fail("{s}", .{why});
    if (incoming.team) |t| {
        log.info("signature ok: Developer ID team {s}{s}\n", .{ t, if (running.team == null) " (the running build is ad-hoc signed)" else "" });
    } else {
        log.info("signature ok: ad-hoc, like the running build\n", .{});
    }
}

/// Null when `<dir>/sushi` reports `want` from both `--version` and `--guest-manifest`, else what went wrong.
fn runsAs(arena: Allocator, io: std.Io, inst: Install, want: []const u8) !?[]const u8 {
    const bin = try std.fmt.allocPrint(arena, "{s}/sushi", .{inst.dir});
    const out = try std.fmt.allocPrint(arena, "{s}/run.txt", .{inst.work});
    if (try runTool(arena, &.{ bin, "--version" }, .{ .path = out }) != 0) return "does not run";
    const text = std.Io.Dir.cwd().readFileAlloc(io, out, arena, .limited(64 * 1024)) catch "";
    const first = std.mem.sliceTo(text, '\n');
    const expect = try std.fmt.allocPrint(arena, "sushi {s}", .{want});
    if (!eql(u8, first, expect)) return try std.fmt.allocPrint(arena, "reports '{s}', not '{s}'", .{ first, expect });
    if (try runTool(arena, &.{ bin, "--guest-manifest" }, .{ .path = out }) != 0) return "fails --guest-manifest";
    const manifest = std.Io.Dir.cwd().readFileAlloc(io, out, arena, .limited(64 * 1024)) catch "";
    const M = struct { version: []const u8 = "" };
    const m = std.json.parseFromSliceLeaky(M, arena, manifest, .{ .ignore_unknown_fields = true }) catch return "prints no guest manifest";
    if (!eql(u8, m.version, want)) return "reports another version in its guest manifest";
    log.info("sushi {s} runs (--version, --guest-manifest)\n", .{want});
    return null;
}

fn rollback(arena: Allocator, io: std.Io, opts: Options) !void {
    const inst = try locate(arena, io, opts);
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const prev = try previousPath(&pbuf, inst.dir);
    std.Io.Dir.cwd().access(io, prev, .{}) catch return fail("no previous install at {s}", .{prev});
    swapPaths(io, inst.dir, prev) catch |err| return fail("cannot swap {s} back: {t}", .{ prev, err });
    const out = try std.fmt.allocPrint(arena, "{s}/run.txt", .{inst.work});
    const bin = try std.fmt.allocPrint(arena, "{s}/sushi", .{inst.dir});
    const code = try runTool(arena, &.{ bin, "--version" }, .{ .path = out });
    const text = std.Io.Dir.cwd().readFileAlloc(io, out, arena, .limited(64 * 1024)) catch "";
    std.Io.Dir.cwd().deleteTree(io, inst.work) catch {};
    const first = std.mem.sliceTo(text, '\n');
    if (code != 0 or !std.mem.startsWith(u8, first, "sushi ")) {
        swapPaths(io, inst.dir, prev) catch {};
        return fail("the previous install does not run; kept this one", .{});
    }
    log.info("rolled back {s} -> {s}\n", .{ version, first["sushi ".len..] });
}

/// The argv as one log line with the `--api-key` secret replaced; exec still takes the real argv.
fn redactedLine(arena: Allocator, argv: []const []const u8) ![]u8 {
    var out = std.ArrayList(u8).empty;
    var hide_next = false;
    for (argv, 0..) |arg, i| {
        if (i > 0) try out.append(arena, ' ');
        if (hide_next) {
            hide_next = false;
            try out.appendSlice(arena, "<redacted>");
        } else if (eql(u8, arg, "--api-key")) {
            hide_next = true;
            try out.appendSlice(arena, arg);
        } else if (std.mem.startsWith(u8, arg, "--api-key=")) {
            try out.appendSlice(arena, "--api-key=<redacted>");
        } else try out.appendSlice(arena, arg);
    }
    return out.toOwnedSlice(arena);
}

/// The end of `--relaunch`: the outcome goes to the cache for `/props`, then the install's sushi (new, or the old one
/// restored) replaces this process with the server's own argv.
fn relaunch(arena: Allocator, io: std.Io, argv: []const []const u8, result: anyerror!void) noreturn {
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    if (cachePath(&pbuf)) |path| {
        var c = readCache(arena, io, path);
        c.@"error" = if (result) |_| "" else |err| if (fail_msg.len > 0) fail_msg else @errorName(err);
        writeCache(arena, io, path, c) catch {};
    }
    var ebuf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = selfDir(io, &ebuf) orelse {
        log.err("update: cannot find the install to relaunch\n", .{});
        std.process.exit(1);
    };
    var full = std.ArrayList([]const u8).empty;
    const bin = std.fmt.allocPrint(arena, "{s}/sushi", .{dir}) catch std.process.exit(1);
    full.append(arena, bin) catch std.process.exit(1);
    full.appendSlice(arena, argv) catch std.process.exit(1);
    const line = redactedLine(arena, full.items) catch bin;
    log.info("relaunching {s}\n", .{line});
    log.closeFile();
    execInPlace(arena, full.items);
    std.process.exit(1);
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "update: a process from the install blocks the swap, except the menu bar app" {
    try testing.expect(blocksUpdate("/opt/s/sushi-macos-arm64", "/opt/s/sushi-macos-arm64/sushi"));
    try testing.expect(blocksUpdate("/opt/s/sushi-macos-arm64", "/opt/s/sushi-macos-arm64/sushi-ubench"));
    try testing.expect(!blocksUpdate("/opt/s/sushi-macos-arm64", "/opt/s/sushi-macos-arm64/sushibar"));
    try testing.expect(!blocksUpdate("/opt/s/sushi-macos-arm64", "/opt/s/other/sushi"));
    try testing.expect(!blocksUpdate("/opt/s/sushi-macos-arm64", "/opt/s/sushi-macos-arm64"));
}

test "update: the relaunch log line never carries the api key" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const spaced = try redactedLine(arena, &.{ "/i/sushi", "--model", "m", "--api-key", "s3cret", "--api-key-strict", "--port", "1" });
    try testing.expectEqualStrings("/i/sushi --model m --api-key <redacted> --api-key-strict --port 1", spaced);
    const joined = try redactedLine(arena, &.{ "/i/sushi", "--api-key=s3cret", "--api-key-env", "KEYVAR" });
    try testing.expectEqualStrings("/i/sushi --api-key=<redacted> --api-key-env KEYVAR", joined);
    try testing.expectEqualStrings("/i/sushi --api-key", try redactedLine(arena, &.{ "/i/sushi", "--api-key" }));
}

test "update: SemVer order: a prerelease sorts below its release, and nothing lower is ever newer" {
    try testing.expect(isNewer("1.1.0", "1.0.4"));
    try testing.expect(isNewer("1.0.10", "1.0.9"));
    try testing.expect(isNewer("1.1.0", "1.1.0-pre-release.3"));
    try testing.expect(isNewer("1.1.0-pre-release.10", "1.1.0-pre-release.9"));
    try testing.expect(!isNewer("1.1.0-pre-release.1", "1.1.0"));
    try testing.expect(!isNewer("1.0.4", "1.0.4"));
    try testing.expect(!isNewer("1.0.3", "1.0.4"));
    try testing.expect(!isNewer("1.0.4+build.7", "1.0.4"));
    try testing.expect(!isNewer("", "1.0.4"));
    try testing.expect(!isNewer("nightly", "1.0.4"));
}

fn releaseJson(comptime tag: []const u8, comptime flags: []const u8, comptime assets: []const u8) []const u8 {
    return "{\"tag_name\":\"" ++ tag ++ "\",\"html_url\":\"https://example.test/" ++ tag ++ "\"" ++ flags ++ ",\"body\":null,\"assets\":[" ++ assets ++ "]}";
}
const both_assets =
    \\{"name":"sushi-bin-macos-arm64.tar.gz","browser_download_url":"https://dl.test/t.tar.gz","size":1},
    \\{"name":"sushi-bin-macos-arm64.tar.gz.sha256","browser_download_url":"https://dl.test/t.sha256"}
;

test "update: the release pick takes the highest SemVer carrying both assets" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const body = comptime "[" ++
        releaseJson("v1.2.0", ",\"draft\":true", both_assets) ++ "," ++
        releaseJson("v1.1.1", "", "{\"name\":\"sushi-bin-macos-arm64.tar.gz\",\"browser_download_url\":\"x\"}") ++ "," ++
        releaseJson("v1.1.0-pre-release.2", ",\"prerelease\":true", both_assets) ++ "," ++
        releaseJson("v1.1.0-pre-release.3", "", both_assets) ++ "," ++
        releaseJson("nightly", "", both_assets) ++ "," ++
        releaseJson("v1.0.5", "", both_assets) ++ "," ++
        releaseJson("v1.1.0", "", both_assets) ++ "," ++
        releaseJson("v1.0.4", "", both_assets) ++ "]";

    const stable = (try pickRelease(arena, body, false)).?;
    try testing.expectEqualStrings("1.1.0", stable.version);
    try testing.expectEqualStrings("https://example.test/v1.1.0", stable.page);
    try testing.expectEqualStrings("https://dl.test/t.tar.gz", stable.tarball);
    try testing.expectEqualStrings("https://dl.test/t.sha256", stable.sha256);

    const newest_pre = comptime "[" ++ releaseJson("v1.1.0", "", both_assets) ++ "," ++
        releaseJson("v1.2.0-pre-release.1", ",\"prerelease\":true", both_assets) ++ "]";
    try testing.expectEqualStrings("1.1.0", (try pickRelease(arena, newest_pre, false)).?.version);
    try testing.expectEqualStrings("1.2.0-pre-release.1", (try pickRelease(arena, newest_pre, true)).?.version);

    try testing.expect(try pickRelease(arena, "[]", false) == null);
    try testing.expectError(error.UnexpectedToken, pickRelease(arena, "{\"message\":\"rate limited\"", false));
}

test "update: the .sha256 asset's digest, and a file's" {
    const hex = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
    try testing.expectEqualStrings(hex, &parseShaLine(hex ++ "  sushi-bin-macos-arm64.tar.gz\n").?);
    try testing.expectEqualStrings(hex, &parseShaLine("9F86D081884C7D659A2FEAA0C55AD015A3BF4F1B2B0B822CD15D6C15B0F00A08\n").?);
    try testing.expect(parseShaLine("9f86d0818") == null);
    try testing.expect(parseShaLine(hex ++ "0  x") == null);
    try testing.expect(parseShaLine("zz" ++ hex[2..]) == null);

    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "f", .data = "test" });
    const path = try tmp.dir.realPathFileAlloc(io, "f", testing.allocator);
    defer testing.allocator.free(path);
    try testing.expectEqualStrings(hex, &try fileSha256Hex(io, path));
}

test "update: codesign output and the signature rule" {
    const team_out =
        \\Executable=/x/sushi
        \\Identifier=sushi
        \\Authority=Developer ID Application: Someone (ABCDE12345)
        \\TeamIdentifier=ABCDE12345
        \\
    ;
    const adhoc_out = "Executable=/x/sushi\nSignature=adhoc\nTeamIdentifier=not set\n";
    const team = parseCodesign(team_out);
    try testing.expectEqualStrings("ABCDE12345", team.team.?);
    try testing.expect(!team.adhoc);
    const adhoc = parseCodesign(adhoc_out);
    try testing.expect(adhoc.adhoc and adhoc.team == null);
    const other = parseCodesign("TeamIdentifier=ZZZZZ99999\n");

    try testing.expect(signatureRefusal(team, team, true) == null);
    try testing.expect(signatureRefusal(team, team, false) != null);
    try testing.expect(signatureRefusal(team, adhoc, true) != null);
    try testing.expect(signatureRefusal(team, other, true) != null);
    try testing.expect(signatureRefusal(adhoc, adhoc, true) == null);
    try testing.expect(signatureRefusal(adhoc, team, true) == null);
    try testing.expect(signatureRefusal(adhoc, adhoc, false) != null);
}

fn tmpPath(tmp: *testing.TmpDir, sub: []const u8) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(testing.io, sub, testing.allocator);
}

test "update: a Homebrew keg is known by its real path under any prefix, and leaves updates to brew" {
    try testing.expect(isHomebrewKeg("/opt/homebrew/Cellar/sushi/1.0.5/libexec"));
    try testing.expect(isHomebrewKeg("/usr/local/Cellar/sushi/1.0.5/libexec"));
    try testing.expect(isHomebrewKeg("/Users/me/.brew/Cellar/sushi/1.1.0_1/libexec"));
    try testing.expect(!isHomebrewKeg("/Users/me/sushi-macos-arm64"));
    try testing.expect(!isHomebrewKeg("/opt/homebrew/Cellar/sushi-nightly/1.0.5/libexec"));
    try testing.expect(!isHomebrewKeg("/Users/me/Cellar-sushi/1.0.5"));
    var buf: [512]u8 = undefined;
    try testing.expectEqualStrings("sushi was installed with Homebrew; run: brew upgrade sushi", installRefusal(testing.io, "/opt/homebrew/Cellar/sushi/1.0.5/libexec", &buf).?);
}

test "update: a source build refuses by name; a release install passes" {
    const io = testing.io;
    const cwd = std.Io.Dir.cwd();
    // Outside any checkout: testing.tmpDir lives in this repo's .zig-cache, under its build.zig.
    const root = try std.fmt.allocPrint(testing.allocator, "{s}/sushi-update-test-{d}", .{ std.mem.span(std.c.getenv("TMPDIR") orelse "/tmp"), std.c.getpid() });
    defer testing.allocator.free(root);
    defer cwd.deleteTree(io, root) catch {};
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    for ([_][]const u8{ "repo/.git", "repo/zig-out/bin/lib", "repo/tools/bin/lib", "dotfiles/.git", "dotfiles/sushi-macos-arm64/lib", "bare" }) |sub|
        try cwd.createDirPath(io, try std.fmt.bufPrint(&pbuf, "{s}/{s}", .{ root, sub }));
    try cwd.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&pbuf, "{s}/repo/build.zig", .{root}), .data = "" });
    var buf: [512]u8 = undefined;

    try testing.expectEqualStrings("built from source: git pull and rebuild", installRefusal(io, try std.fmt.bufPrint(&pbuf, "{s}/repo/zig-out/bin", .{root}), &buf).?);
    try testing.expect(isSourceBuild(io, try std.fmt.bufPrint(&pbuf, "{s}/repo/tools/bin", .{root})));
    // A git-managed home folder is not a sushi checkout.
    const release = try std.fmt.bufPrint(&pbuf, "{s}/dotfiles/sushi-macos-arm64", .{root});
    try testing.expect(!isSourceBuild(io, release));
    try testing.expect(installRefusal(io, release, &buf) == null);
    try testing.expect(std.mem.endsWith(u8, installRefusal(io, try std.fmt.bufPrint(&pbuf, "{s}/bare", .{root}), &buf).?, "is not a release install (no lib/ beside sushi)"));
    try testing.expectEqualStrings("part of an app bundle: update the app", installRefusal(io, "/Applications/Sushi.app/Contents/MacOS", &buf).?);
}

fn fileText(dir: std.Io.Dir, sub: []const u8) ![]u8 {
    return dir.readFileAlloc(testing.io, sub, testing.allocator, .limited(1024));
}

fn expectMarker(dir: std.Io.Dir, sub: []const u8, want: []const u8) !void {
    const got = try fileText(dir, sub);
    defer testing.allocator.free(got);
    try testing.expectEqualStrings(want, got);
}

test "update: the swap installs the new build, keeps one .previous, rolls back, and a failed swap changes nothing" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "inst/lib");
    try tmp.dir.writeFile(io, .{ .sub_path = "inst/sushi", .data = "old" });
    try tmp.dir.createDirPath(io, "inst.previous");
    try tmp.dir.writeFile(io, .{ .sub_path = "inst.previous/sushi", .data = "older" });
    try tmp.dir.createDirPath(io, "w/new/lib");
    try tmp.dir.writeFile(io, .{ .sub_path = "w/new/sushi", .data = "new" });
    const inst = try tmpPath(&tmp, "inst");
    defer testing.allocator.free(inst);
    const new = try tmpPath(&tmp, "w/new");
    defer testing.allocator.free(new);
    const prev = try std.fmt.allocPrint(testing.allocator, "{s}.previous", .{inst});
    defer testing.allocator.free(prev);

    try swapPaths(io, inst, new);
    try expectMarker(tmp.dir, "inst/sushi", "new");
    try keepPrevious(io, inst, new);
    try expectMarker(tmp.dir, "inst.previous/sushi", "old");
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "w/new", .{}));

    try swapPaths(io, inst, prev);
    try expectMarker(tmp.dir, "inst/sushi", "old");
    try expectMarker(tmp.dir, "inst.previous/sushi", "new");

    const missing = try std.fmt.allocPrint(testing.allocator, "{s}/w/gone", .{std.fs.path.dirname(inst).?});
    defer testing.allocator.free(missing);
    try testing.expectError(error.FileNotFound, swapPaths(io, inst, missing));
    try testing.expectError(error.FileNotFound, swapByRenames(io, inst, missing));
    try expectMarker(tmp.dir, "inst/sushi", "old");
    try testing.expectError(error.FileNotFound, tmp.dir.access(io, "inst.swap", .{}));

    try swapByRenames(io, inst, prev);
    try expectMarker(tmp.dir, "inst/sushi", "new");
    try expectMarker(tmp.dir, "inst.previous/sushi", "old");
}

test "update: an entry the release does not ship stops the swap" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "inst/sushi", "inst/NOTICE", "inst/guest.json", "inst/README", "new/sushi", "new/README", "new/CHANGES" }) |p| {
        try tmp.dir.createDirPath(io, std.fs.path.dirname(p).?);
        try tmp.dir.writeFile(io, .{ .sub_path = p, .data = "" });
    }
    try tmp.dir.createDirPath(io, "inst/lib");
    const inst = try tmpPath(&tmp, "inst");
    defer testing.allocator.free(inst);
    const new = try tmpPath(&tmp, "new");
    defer testing.allocator.free(new);
    var buf: [64]u8 = undefined;
    try testing.expect(try strayEntry(io, inst, new, &buf) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "inst/llama-server", .data = "" });
    try testing.expectEqualStrings("llama-server", (try strayEntry(io, inst, new, &buf)).?);
}

test "update: the daily cache is due after 24 h and folds 200, 304 and failures" {
    try testing.expect(due(.{}, 1_000_000));
    try testing.expect(!due(.{ .checked_at = 1_000_000 }, 1_000_000 + day_s - 1));
    try testing.expect(due(.{ .checked_at = 1_000_000 }, 1_000_000 + day_s));
    try testing.expect(due(.{ .checked_at = 1_000_000 }, 999_000));

    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var c: Cache = .{ .@"error" = "earlier failure" };
    const body = comptime "[" ++ releaseJson("v1.1.0", "", both_assets) ++ "]";
    try testing.expect(try applyFetch(&c, arena, 200, body, "W/\"e1\"", 100));
    try testing.expectEqualStrings("1.1.0", c.latest);
    try testing.expectEqualStrings("https://example.test/v1.1.0", c.url);
    try testing.expectEqualStrings("W/\"e1\"", c.etag);
    try testing.expectEqual(@as(i64, 100), c.checked_at);
    try testing.expect(try applyFetch(&c, arena, 304, "", "", 200));
    try testing.expectEqualStrings("1.1.0", c.latest);
    try testing.expectEqualStrings("W/\"e1\"", c.etag);
    try testing.expectEqual(@as(i64, 200), c.checked_at);
    try testing.expect(!try applyFetch(&c, arena, 403, "{}", "", 300));
    try testing.expectEqual(@as(i64, 200), c.checked_at);

    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpPath(&tmp, ".");
    defer testing.allocator.free(root);
    const path = try std.fmt.allocPrint(arena, "{s}/.sushi/update-check.json", .{root});
    try testing.expectEqual(@as(i64, 0), readCache(arena, io, path).checked_at);
    try writeCache(arena, io, path, c);
    const back = readCache(arena, io, path);
    try testing.expectEqual(@as(i64, 200), back.checked_at);
    try testing.expectEqualStrings("1.1.0", back.latest);
    try testing.expectEqualStrings("W/\"e1\"", back.etag);
    try testing.expectEqualStrings("earlier failure", back.@"error");
}

test "update: /props carries null fields until a check answers, then the newer release" {
    defer publish(.{});
    const empty = try propsJson(testing.allocator);
    defer testing.allocator.free(empty);
    try testing.expectEqualStrings(",\"update\":{\"current\":\"" ++ version ++ "\",\"latest\":null,\"available\":false,\"checked_at\":null,\"url\":null,\"error\":null,\"command\":null}", empty);

    publish(.{ .checked_at = 1_790_000_000, .latest = "999.0.0", .url = "https://example.test/v999.0.0", .etag = "e", .@"error" = "SHA-256 mismatch" });
    const found = try propsJson(testing.allocator);
    defer testing.allocator.free(found);
    try testing.expectEqualStrings(",\"update\":{\"current\":\"" ++ version ++ "\",\"latest\":\"999.0.0\",\"available\":true,\"checked_at\":1790000000,\"url\":\"https://example.test/v999.0.0\",\"error\":\"SHA-256 mismatch\",\"command\":null}", found);
    homebrew.store(true, .release);
    defer homebrew.store(false, .release);
    const brew = try propsJson(testing.allocator);
    defer testing.allocator.free(brew);
    try testing.expect(std.mem.endsWith(u8, brew, ",\"command\":\"brew upgrade sushi\"}"));
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("999.0.0", availableVersion(&buf).?);
    publish(.{ .checked_at = 1, .latest = version });
    try testing.expect(availableVersion(&buf) == null);
}

test "update: the daily check's resolved state names its source" {
    try testing.expectEqualStrings("--no-update-check", checkChoice(true, true, null).source);
    try testing.expectEqualStrings("source build", checkChoice(false, true, null).source);
    const on = checkChoice(false, false, null);
    try testing.expect(on.on);
    try testing.expectEqualStrings("default", on.source);
    try testing.expect(!checkChoice(true, false, null).on);
}

test "update: the settings file turns the check off or on, and the flag and a source build outrank it" {
    const off = checkChoice(false, false, false);
    try testing.expect(!off.on);
    try testing.expectEqualStrings("model-settings.json", off.source);
    try testing.expect(checkChoice(false, false, true).on);
    try testing.expect(!checkChoice(true, false, true).on);
    try testing.expectEqualStrings("--no-update-check", checkChoice(true, false, true).source);
    try testing.expectEqualStrings("source build", checkChoice(false, true, true).source);
}

test "update: the published toggle gates every automatic look and nothing else" {
    defer setAutoCheck(true);
    try testing.expect(autoCheckOn());
    try testing.expect(looksNow(autoCheckOn(), 100, 100));
    try testing.expect(!looksNow(autoCheckOn(), 99, 100));
    setAutoCheck(false);
    try testing.expect(!looksNow(autoCheckOn(), 1_000_000, 0));
    setAutoCheck(true);
    try testing.expect(looksNow(autoCheckOn(), 1_000_000, 0));
    // A clock set back must not strand the next look a day or more away.
    try testing.expect(looksNow(true, 100, 100 + 2 * day_s));
}

test "update: /v1/update is refused on a public bind, a remote peer, a bad key or Origin, a host, a busy server" {
    const ok: Ask = .{
        .bind_host = "127.0.0.1",
        .port = 12345,
        .peer_loopback = true,
        .origin = "http://127.0.0.1:12345",
        .key_set = false,
        .key_ok = false,
        .host_managed = false,
        .busy = false,
        .install = null,
        .available = true,
    };
    try testing.expect(guard(ok) == null);
    var a = ok;
    a.origin = "http://localhost:12345";
    try testing.expect(guard(a) == null);

    const Case = struct { ask: Ask, code: u16 };
    var cases = [_]Case{
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 401 },
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 403 },
        .{ .ask = ok, .code = 409 },
        .{ .ask = ok, .code = 409 },
        .{ .ask = ok, .code = 409 },
    };
    cases[0].ask.bind_host = "0.0.0.0";
    cases[1].ask.peer_loopback = false;
    cases[2].ask.key_set = true;
    cases[3].ask.origin = null;
    cases[4].ask.origin = "http://evil.test:12345";
    cases[5].ask.origin = "http://127.0.0.1:9999";
    cases[6].ask.host_managed = true;
    cases[7].ask.busy = true;
    cases[8].ask.install = "built from source: git pull and rebuild";
    cases[9].ask.available = false;
    for (cases) |c| try testing.expectEqual(c.code, guard(c.ask).?.code);

    a = ok;
    a.key_set = true;
    a.key_ok = true;
    try testing.expect(guard(a) == null);
    try testing.expectEqualStrings("built from source: git pull and rebuild", guard(cases[8].ask).?.message);
}

test "update: the page's origin is the bind's, with loopback names interchangeable" {
    try testing.expect(originMatches("http://127.0.0.1:8080", "127.0.0.1", 8080));
    try testing.expect(originMatches("http://localhost:8080", "127.0.0.1", 8080));
    try testing.expect(originMatches("http://127.0.0.1:8080", "localhost", 8080));
    try testing.expect(originMatches("http://127.0.0.2:8080", "127.0.0.2", 8080));
    try testing.expect(!originMatches("http://localhost:8080", "127.0.0.2", 8080));
    try testing.expect(!originMatches("http://127.0.0.1:8081", "127.0.0.1", 8080));
    try testing.expect(!originMatches("https://127.0.0.1:8080", "127.0.0.1", 8080));
    try testing.expect(!originMatches("http://127.0.0.1.evil.test:8080", "127.0.0.1", 8080));
    try testing.expect(!originMatches("null", "127.0.0.1", 8080));
}

test "update: tools run through posix_spawn with their output captured" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const root = try tmpPath(&tmp, ".");
    defer testing.allocator.free(root);
    const out = try std.fmt.allocPrint(arena, "{s}/out.txt", .{root});
    try testing.expectEqual(@as(u8, 0), try runTool(arena, &.{ "/bin/sh", "-c", "echo out; echo err >&2" }, .{ .path = out, .stderr = true }));
    try expectMarker(tmp.dir, "out.txt", "out\nerr\n");
    // A version or a manifest is read from stdout alone: a log line on stderr must not reach the parse.
    try testing.expectEqual(@as(u8, 0), try runTool(arena, &.{ "/bin/sh", "-c", "echo out; echo err >&2" }, .{ .path = out }));
    try expectMarker(tmp.dir, "out.txt", "out\n");
    try testing.expectEqual(@as(u8, 3), try runTool(arena, &.{ "/bin/sh", "-c", "exit 3" }, .{ .path = out }));
    try testing.expectError(error.SpawnFailed, runTool(arena, &.{"/nonexistent/tool"}, .{ .path = out }));

    // An ignored SIGHUP (`nohup sushi serve`) stays ignored across the spawn, as across the updater's exec.
    const ign: std.posix.Sigaction = .{ .handler = .{ .handler = std.posix.SIG.IGN }, .mask = std.posix.sigemptyset(), .flags = 0 };
    var prev: std.posix.Sigaction = undefined;
    std.posix.sigaction(std.posix.SIG.HUP, &ign, &prev);
    defer std.posix.sigaction(std.posix.SIG.HUP, &prev, null);
    try testing.expectEqual(@as(u8, 0), try runTool(arena, &.{ "/bin/sh", "-c", "kill -HUP $$; echo survived" }, .{ .path = out }));
    try expectMarker(tmp.dir, "out.txt", "survived\n");
}

test "update: concurrent writers of one JSON file never expose half a file nor fail each other" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpPath(&tmp, ".");
    defer testing.allocator.free(root);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/cache.json", .{root});
    defer testing.allocator.free(path);
    const Doc = struct { text: []const u8 };
    const Run = struct {
        path: []const u8,
        done: std.atomic.Value(u32) = .init(0),
        failed: std.atomic.Value(u32) = .init(0),
        torn: std.atomic.Value(u32) = .init(0),

        fn write(self: *@This(), fill: u8) void {
            defer _ = self.done.fetchAdd(1, .release);
            var big: [64 * 1024]u8 = undefined;
            @memset(&big, fill);
            for (0..150) |_| {
                var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
                defer arena.deinit();
                writeJsonAtomic(arena.allocator(), testing.io, self.path, Doc{ .text = &big }) catch {
                    _ = self.failed.fetchAdd(1, .monotonic);
                };
            }
        }

        fn read(self: *@This()) void {
            while (self.done.load(.acquire) < 2) {
                var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
                defer arena.deinit();
                const text = std.Io.Dir.cwd().readFileAlloc(testing.io, self.path, arena.allocator(), .limited(1 << 20)) catch continue;
                const doc = std.json.parseFromSliceLeaky(Doc, arena.allocator(), text, .{}) catch {
                    _ = self.torn.fetchAdd(1, .monotonic);
                    continue;
                };
                if (doc.text.len != 64 * 1024) _ = self.torn.fetchAdd(1, .monotonic);
            }
        }
    };
    var run: Run = .{ .path = path };
    const threads = [_]std.Thread{
        try std.Thread.spawn(.{}, Run.write, .{ &run, 'a' }),
        try std.Thread.spawn(.{}, Run.write, .{ &run, 'b' }),
        try std.Thread.spawn(.{}, Run.read, .{&run}),
    };
    for (threads) |t| t.join();
    try testing.expectEqual(@as(u32, 0), run.failed.load(.monotonic));
    try testing.expectEqual(@as(u32, 0), run.torn.load(.monotonic));
}
