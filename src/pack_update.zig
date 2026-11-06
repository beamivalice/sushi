//! A pack's revision against the one published on Hugging Face. `config.json` carries `pack_revision.id`; a config
//! without it is a legacy pack, revision 1.0. `sushi list` and the server ask and say "update available" only when
//! the remote revision is higher; an answer that never arrived claims nothing either way.

const std = @import("std");
const log = @import("log.zig");
const hf_pull = @import("hf_pull.zig");
const update = @import("update.zig");

const Allocator = std.mem.Allocator;

pub const legacy_revision = "1.0";

/// One published pack: what `sushi pull` accepts for it and the repo it comes from.
pub const Known = struct {
    name: []const u8,
    tag: []const u8,
    repo: []const u8,
    is_default: bool,
};

/// The revision a `config.json` declares: `pack_revision.id`, else the legacy 1.0. Null when it is not JSON.
pub fn revisionOf(arena: Allocator, config_json: []const u8) ?[]const u8 {
    const Doc = struct { pack_revision: ?struct { id: []const u8 = "" } = null };
    const doc = std.json.parseFromSliceLeaky(Doc, arena, config_json, .{ .ignore_unknown_fields = true }) catch return null;
    const rev = doc.pack_revision orelse return legacy_revision;
    return if (rev.id.len == 0) legacy_revision else rev.id;
}

const max_segments = 8;

fn segments(text: []const u8, out: *[max_segments]u32) ?usize {
    if (text.len == 0) return null;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, '.');
    while (it.next()) |part| {
        if (n == max_segments) return null;
        out[n] = std.fmt.parseInt(u32, part, 10) catch return null;
        n += 1;
    }
    return n;
}

/// Dotted numbers compared segment by segment, a missing segment being 0. Null when either is not dotted numbers.
pub fn compareRevisions(a: []const u8, b: []const u8) ?std.math.Order {
    var sa: [max_segments]u32 = undefined;
    var sb: [max_segments]u32 = undefined;
    const na = segments(a, &sa) orelse return null;
    const nb = segments(b, &sb) orelse return null;
    for (0..@max(na, nb)) |i| {
        const x = if (i < na) sa[i] else 0;
        const y = if (i < nb) sb[i] else 0;
        if (x != y) return std.math.order(x, y);
    }
    return .eq;
}

pub const Target = struct {
    repo: []const u8,
    /// What follows `sushi pull`.
    pull: []const u8,
};

fn repoBasename(repo: []const u8) []const u8 {
    return repo[(std.mem.lastIndexOfScalar(u8, repo, '/') orelse return repo) + 1 ..];
}

fn pullSpec(arena: Allocator, k: Known) ?[]const u8 {
    if (k.is_default) return k.name;
    return std.fmt.allocPrint(arena, "{s}:{s}", .{ k.name, k.tag }) catch null;
}

fn manifestRepo(arena: Allocator, manifest_json: []const u8) ?[]const u8 {
    const Doc = struct { repo: []const u8 = "" };
    const doc = std.json.parseFromSliceLeaky(Doc, arena, manifest_json, .{ .ignore_unknown_fields = true }) catch return null;
    return if (doc.repo.len == 0) null else doc.repo;
}

/// The repo a local pack folder came from: the pull manifest's `repo` when it names one, else the published pack
/// whose repo basename the folder carries. A folder that is neither has no update check.
pub fn resolveTarget(arena: Allocator, known: []const Known, folder: []const u8, manifest_json: ?[]const u8) ?Target {
    if (manifest_json) |m| if (manifestRepo(arena, m)) |repo| {
        for (known) |k| if (std.ascii.eqlIgnoreCase(k.repo, repo)) return .{ .repo = k.repo, .pull = pullSpec(arena, k) orelse return null };
        return .{ .repo = repo, .pull = repo };
    };
    for (known) |k| {
        if (std.ascii.eqlIgnoreCase(repoBasename(k.repo), folder)) return .{ .repo = k.repo, .pull = pullSpec(arena, k) orelse return null };
    }
    return null;
}

/// `repo` is the Hugging Face repo of the published pack, which the web page's Available list shows the update under.
pub const Available = struct { local: []const u8, latest: []const u8, command: []const u8, repo: []const u8 = "" };

pub const Status = union(enum) {
    unknown,
    current,
    available: Available,
};

pub fn decide(arena: Allocator, local: ?[]const u8, remote: ?[]const u8, target: Target) Allocator.Error!Status {
    const l = local orelse return .unknown;
    const r = remote orelse return .unknown;
    const order = compareRevisions(r, l) orelse return .unknown;
    if (order != .gt) return .current;
    return .{ .available = .{
        .local = l,
        .latest = r,
        .command = try std.fmt.allocPrint(arena, "sushi pull {s}", .{target.pull}),
        .repo = target.repo,
    } };
}

/// The line `sushi list` prints for a pack with an update.
pub fn listLine(w: *std.Io.Writer, a: Available) std.Io.Writer.Error!void {
    try w.print("update available: {s} (you have {s}) — {s}\n", .{ a.latest, a.local, a.command });
}

/// `,"pack_update":{...}` for a `/v1/models` row's `meta`; empty unless an update exists.
pub fn metaFragment(w: *std.Io.Writer, st: Status) std.Io.Writer.Error!void {
    const a = switch (st) {
        .available => |a| a,
        else => return,
    };
    try w.writeAll(",\"pack_update\":");
    try std.json.Stringify.value(.{ .current = a.local, .latest = a.latest, .command = a.command, .repo = a.repo }, .{}, w);
}

// ── Hugging Face credentials ────────────────────────────────────────────

/// The process environment a Hugging Face token is resolved from; a test passes its own.
pub const HfEnv = struct {
    token: ?[]const u8 = null,
    token_path: ?[]const u8 = null,
    hf_home: ?[]const u8 = null,
    home: []const u8,

    pub fn fromProcess() HfEnv {
        const get = struct {
            fn f(name: [:0]const u8) ?[]const u8 {
                return if (std.c.getenv(name.ptr)) |v| std.mem.span(v) else null;
            }
        }.f;
        return .{ .token = get("HF_TOKEN"), .token_path = get("HF_TOKEN_PATH"), .hf_home = get("HF_HOME"), .home = get("HOME") orelse "/tmp" };
    }
};

/// The token `sushi pull` and `sushi list` send: `HF_TOKEN`, else the file `hf auth login` wrote (`HF_TOKEN_PATH`,
/// else `$HF_HOME/token`, else `~/.cache/huggingface/token`; the first one set is the only one read). Blank is none.
pub fn resolveHfToken(allocator: Allocator, io: std.Io, env: HfEnv) !?[]u8 {
    if (env.token) |t| {
        const v = std.mem.trim(u8, t, " \t\r\n");
        if (v.len > 0) return try allocator.dupe(u8, v);
    }
    const path = try hfTokenPath(allocator, env);
    defer allocator.free(path);
    const raw = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4096)) catch return null;
    defer allocator.free(raw);
    const v = std.mem.trim(u8, raw, " \t\r\n");
    return if (v.len == 0) null else try allocator.dupe(u8, v);
}

fn hfTokenPath(allocator: Allocator, env: HfEnv) ![]u8 {
    if (env.token_path) |p| if (p.len > 0) return allocator.dupe(u8, p);
    if (env.hf_home) |h| if (h.len > 0) return std.fmt.allocPrint(allocator, "{s}/token", .{h});
    return std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/token", .{env.home});
}

// ── The check ───────────────────────────────────────────────────────────

const fetch_timeout_s = 5;

/// The remote `config.json` of `repo`, or null for anything but a 200 inside a few seconds: offline, a 404 and a
/// private repo without a token all read as "no answer".
pub fn fetchRemoteConfig(arena: Allocator, io: std.Io, repo: []const u8) ?[]const u8 {
    const url = hf_pull.hfUrl(arena, hf_pull.endpoint(), "/{s}/resolve/main/config.json", .{repo}) catch return null;
    const U = union(enum) { got: ?[]const u8, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(io, &buf);
    sel.concurrent(.got, getBody, .{ arena, io, url }) catch return getBody(arena, io, url);
    sel.concurrent(.timer, std.Io.sleep, .{ io, std.Io.Duration.fromSeconds(fetch_timeout_s), .awake }) catch {};
    const first = sel.await() catch {
        while (sel.cancel()) |_| {}
        return null;
    };
    while (sel.cancel()) |_| {}
    return switch (first) {
        .got => |r| r,
        .timer => null,
    };
}

fn getBody(arena: Allocator, io: std.Io, url: []const u8) ?[]const u8 {
    const token = resolveHfToken(arena, io, HfEnv.fromProcess()) catch null;
    const bearer = if (token) |t| std.fmt.allocPrint(arena, "Bearer {s}", .{t}) catch return null else null;
    var client: std.http.Client = .{ .allocator = arena, .io = io };
    defer client.deinit();
    var req = client.request(.GET, std.Uri.parse(url) catch return null, .{
        .keep_alive = false,
        .headers = .{
            .user_agent = .{ .override = "sushi" },
            .accept_encoding = .{ .override = "identity" },
            .authorization = if (bearer) |b| .{ .override = b } else .default,
        },
    }) catch return null;
    defer req.deinit();
    req.sendBodiless() catch return null;
    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buf) catch return null;
    if (response.head.status != .ok) return null;
    var transfer_buf: [64]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    var body = std.ArrayList(u8).empty;
    reader.appendRemaining(arena, &body, .limited(4 * 1024 * 1024)) catch return null;
    return body.items;
}

fn readSmall(arena: Allocator, io: std.Io, dir: []const u8, name: []const u8) ?[]const u8 {
    const path = std.fmt.allocPrint(arena, "{s}/{s}", .{ dir, name }) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(4 * 1024 * 1024)) catch null;
}

// ── The answer, kept for a day ──────────────────────────────────────────

const day_s: i64 = 24 * 60 * 60;
const retry_s: i64 = 60 * 60;

/// What Hugging Face last said a repo's revision is.
const CacheEntry = struct { repo: []const u8, checked_at: i64, revision: []const u8 };
const Cache = struct { packs: []const CacheEntry = &.{} };

fn cachePath(buf: []u8) ?[]const u8 {
    const home = std.c.getenv("HOME") orelse return null;
    return std.fmt.bufPrint(buf, "{s}/.sushi/pack-update-check.json", .{std.mem.span(home)}) catch null;
}

fn readCache(arena: Allocator, io: std.Io, path: []const u8) Cache {
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1024 * 1024)) catch return .{};
    return std.json.parseFromSliceLeaky(Cache, arena, text, .{ .ignore_unknown_fields = true }) catch .{};
}

var cache_mu: std.c.pthread_mutex_t = .{};

/// Replaces or adds one repo's entry; the watch threads and `sushi list` share the file.
fn remember(arena: Allocator, io: std.Io, path: []const u8, entry: CacheEntry) void {
    _ = std.c.pthread_mutex_lock(&cache_mu);
    defer _ = std.c.pthread_mutex_unlock(&cache_mu);
    var keep = std.ArrayList(CacheEntry).empty;
    for (readCache(arena, io, path).packs) |e| {
        if (!std.mem.eql(u8, e.repo, entry.repo)) keep.append(arena, e) catch return;
    }
    keep.append(arena, entry) catch return;
    update.writeJsonAtomic(arena, io, path, Cache{ .packs = keep.items }) catch |err| log.debug("[pack-update] cache write: {t}\n", .{err});
}

/// An answer is reused for a day; one stamped in the future (a clock set back) is not.
fn due(checked_at: i64, now: i64) bool {
    return now < checked_at or now - checked_at >= day_s;
}

const Fetch = *const fn (Allocator, std.Io, []const u8) ?[]const u8;

pub const Env = struct {
    cache_path: ?[]const u8,
    now: i64,
    fetch: Fetch = fetchRemoteConfig,
};

const Remote = struct { revision: ?[]const u8, wait_s: i64 };

/// The repo's published revision: the cached answer while it is under a day old, else a fresh look that is cached.
/// `wait_s` is how long until it is worth looking again.
fn remoteRevision(arena: Allocator, io: std.Io, repo: []const u8, env: Env) Remote {
    if (env.cache_path) |path| for (readCache(arena, io, path).packs) |e| {
        if (std.mem.eql(u8, e.repo, repo) and !due(e.checked_at, env.now)) return .{ .revision = e.revision, .wait_s = e.checked_at + day_s - env.now };
    };
    const rev = revisionOf(arena, env.fetch(arena, io, repo) orelse return .{ .revision = null, .wait_s = retry_s }) orelse return .{ .revision = null, .wait_s = retry_s };
    if (env.cache_path) |path| remember(arena, io, path, .{ .repo = repo, .checked_at = env.now, .revision = rev });
    return .{ .revision = rev, .wait_s = day_s };
}

pub const Look = struct { status: Status, wait_s: i64 };

pub fn lookWith(arena: Allocator, io: std.Io, known: []const Known, pack_path: []const u8, env: Env) Look {
    const folder = std.fs.path.basename(std.mem.trimEnd(u8, pack_path, "/"));
    const target = resolveTarget(arena, known, folder, readSmall(arena, io, pack_path, ".sushi-pull.json")) orelse return .{ .status = .unknown, .wait_s = day_s };
    const local = revisionOf(arena, readSmall(arena, io, pack_path, "config.json") orelse return .{ .status = .unknown, .wait_s = retry_s });
    const remote = remoteRevision(arena, io, target.repo, env);
    return .{ .status = decide(arena, local, remote.revision, target) catch .unknown, .wait_s = remote.wait_s };
}

/// Where a local pack stands against its Hugging Face repo, asking at most once a day per repo. Never throws: any
/// failure is `.unknown`.
pub fn look(arena: Allocator, io: std.Io, known: []const Known, pack_path: []const u8) Look {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    return lookWith(arena, io, known, pack_path, .{ .cache_path = cachePath(&buf), .now = std.Io.Timestamp.now(io, .real).toSeconds() });
}

pub fn checkPack(arena: Allocator, io: std.Io, known: []const Known, pack_path: []const u8) Status {
    return look(arena, io, known, pack_path).status;
}

// ── The server's look: after a load, then once a day (the answer is kept across restarts) ──

const slot_cap = 8;
const slot_meta_cap = 384;

const Slot = struct {
    path: [std.fs.max_path_bytes]u8 = undefined,
    path_len: usize = 0,
    meta: [slot_meta_cap]u8 = undefined,
    meta_len: usize = 0,
};

var mu: std.c.pthread_mutex_t = .{};
var enabled = false;
var auto_on = std.atomic.Value(bool).init(true);
var server_io: std.Io = undefined;
var server_known: []const Known = &.{};
var slots: [slot_cap]Slot = @splat(.{});

/// Called once at serve start: `on` is false under `--no-update-check`, `auto` is the settings file's choice;
/// `known` outlives the process.
pub fn configure(io: std.Io, on: bool, auto: bool, known: []const Known) void {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    server_io = io;
    server_known = known;
    enabled = on;
    setAuto(auto);
}

/// The settings page flips this; each watch thread reads it before a look.
pub fn setAuto(on: bool) void {
    auto_on.store(on, .release);
}

pub fn autoOn() bool {
    return auto_on.load(.acquire);
}

/// Whether a loop wakes into a look; a `next_at` more than a day ahead is a clock set back.
fn looksNow(on: bool, now: i64, next_at: i64) bool {
    return on and (now >= next_at or next_at - now > day_s);
}

/// False under `--no-update-check`.
pub fn checksOn() bool {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    return enabled;
}

/// Whether the server may ask Hugging Face on its own: off under `--no-update-check` and under the settings'
/// `pack_update_check: false`. A request the user makes (Check, Download) asks either way.
pub fn autoChecksOn() bool {
    return checksOn() and autoOn();
}

/// Starts watching a loaded pack's folder (once per folder) on a detached thread: never the inference thread.
pub fn watch(pack_path: []const u8) void {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    if (!enabled or pack_path.len == 0 or pack_path.len > std.fs.max_path_bytes) return;
    var free: ?usize = null;
    for (&slots, 0..) |*s, i| {
        if (s.path_len == 0) {
            if (free == null) free = i;
        } else if (std.mem.eql(u8, s.path[0..s.path_len], pack_path)) return;
    }
    const i = free orelse return;
    @memcpy(slots[i].path[0..pack_path.len], pack_path);
    slots[i].path_len = pack_path.len;
    const t = std.Thread.spawn(.{}, watchLoop, .{i}) catch |err| {
        slots[i].path_len = 0;
        log.debug("[pack-update] watch thread: {t}\n", .{err});
        return;
    };
    t.detach();
}

/// Wakes every minute so a toggle from the settings page takes hold within it.
fn watchLoop(i: usize) void {
    var next_at: i64 = 0;
    while (true) {
        _ = std.c.pthread_mutex_lock(&mu);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = path_buf[0..slots[i].path_len];
        @memcpy(path, slots[i].path[0..path.len]);
        const io = server_io;
        const known = server_known;
        _ = std.c.pthread_mutex_unlock(&mu);

        const now = std.Io.Timestamp.now(io, .real).toSeconds();
        if (looksNow(autoOn(), now, next_at)) {
            var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
            defer arena_state.deinit();
            const l = look(arena_state.allocator(), io, known, path);
            record(path, l.status);
            if (l.status == .available) log.info("[pack-update] {s}: revision {s} is available (you have {s}): run `{s}`\n", .{ std.fs.path.basename(path), l.status.available.latest, l.status.available.local, l.status.available.command });
            next_at = now + @max(l.wait_s, 60);
        }
        const ts = std.c.timespec{ .sec = 60, .nsec = 0 };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// Publishes what a look found for the pack at `pack_path`, for `rowFragment`.
pub fn record(pack_path: []const u8, st: Status) void {
    var tmp: [slot_meta_cap]u8 = undefined;
    var w = std.Io.Writer.fixed(&tmp);
    const n = if (metaFragment(&w, st)) |_| w.buffered().len else |_| 0;
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    var free: ?usize = null;
    for (&slots, 0..) |*s, i| {
        if (s.path_len == 0) {
            if (free == null) free = i;
        } else if (std.mem.eql(u8, s.path[0..s.path_len], pack_path)) {
            free = i;
            break;
        }
    }
    const i = free orelse return;
    if (slots[i].path_len == 0) {
        if (pack_path.len > std.fs.max_path_bytes) return;
        @memcpy(slots[i].path[0..pack_path.len], pack_path);
        slots[i].path_len = pack_path.len;
    }
    @memcpy(slots[i].meta[0..n], tmp[0..n]);
    slots[i].meta_len = n;
}

/// The `,"pack_update":{...}` fragment for the `/v1/models` row of the pack at `pack_path`; empty when the last
/// look found no update or none has answered. Caller frees.
pub fn rowFragment(allocator: Allocator, pack_path: []const u8) Allocator.Error![]u8 {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    for (&slots) |*s| {
        if (s.path_len == pack_path.len and std.mem.eql(u8, s.path[0..s.path_len], pack_path)) return allocator.dupe(u8, s.meta[0..s.meta_len]);
    }
    return allocator.dupe(u8, "");
}

const testing = std.testing;

const test_known = [_]Known{
    .{ .name = "alpha", .tag = "2.75bpw", .repo = "beamster/Alpha-Sushi-STQ-2.75bpw", .is_default = true },
    .{ .name = "alpha", .tag = "4bpw", .repo = "beamster/Alpha-Sushi-4bpw", .is_default = false },
    .{ .name = "beta", .tag = "1.75bpw", .repo = "beamster/Beta-1017-Sushi-1.75bpw", .is_default = true },
};

test "revision: a config without pack_revision is legacy 1.0, a stamped one carries its id" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try testing.expectEqualStrings("1.0", revisionOf(a, "{\"model_type\":\"beta_arch\"}").?);
    try testing.expectEqualStrings("2.0", revisionOf(a, "{\"model_type\":\"x\",\"pack_revision\":{\"id\":\"2.0\",\"unstamped\":[\"a\"]}}").?);
    try testing.expectEqualStrings("1.0", revisionOf(a, "{\"pack_revision\":{\"unstamped\":[]}}").?);
    try testing.expect(revisionOf(a, "<html>not json") == null);
}

test "revision: dotted numbers compare per segment, so 1.10 is above 1.9" {
    try testing.expectEqual(std.math.Order.gt, compareRevisions("2.0", "1.0").?);
    try testing.expectEqual(std.math.Order.gt, compareRevisions("1.10", "1.9").?);
    try testing.expectEqual(std.math.Order.eq, compareRevisions("2", "2.0").?);
    try testing.expectEqual(std.math.Order.lt, compareRevisions("1.0", "1.0.1").?);
    try testing.expect(compareRevisions("2.x", "1.0") == null);
    try testing.expect(compareRevisions("", "1.0") == null);
}

test "target: the manifest's repo wins, else the folder's name picks the published pack" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const by_name = resolveTarget(a, &test_known, "Beta-1017-Sushi-1.75bpw", null).?;
    try testing.expectEqualStrings("beamster/Beta-1017-Sushi-1.75bpw", by_name.repo);
    try testing.expectEqualStrings("beta", by_name.pull);
    const lower = resolveTarget(a, &test_known, "beta-1017-sushi-1.75bpw", null).?;
    try testing.expectEqualStrings("beta", lower.pull);
    const tagged = resolveTarget(a, &test_known, "my-copy", "{\"repo\":\"beamster/Alpha-Sushi-4bpw\"}").?;
    try testing.expectEqualStrings("alpha:4bpw", tagged.pull);
    const unlisted = resolveTarget(a, &test_known, "my-copy", "{\"repo\":\"beamster/Other-Pack\"}").?;
    try testing.expectEqualStrings("beamster/Other-Pack", unlisted.pull);
    try testing.expect(resolveTarget(a, &test_known, "my-copy", null) == null);
    try testing.expect(resolveTarget(a, &test_known, "my-copy", "{\"other\":1}") == null);
    const fallthrough = resolveTarget(a, &test_known, "Alpha-Sushi-4bpw", "not json").?;
    try testing.expectEqualStrings("alpha:4bpw", fallthrough.pull);
}

test "decide: only a higher remote claims an update; a missing side claims nothing" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const t: Target = .{ .repo = "beamster/X", .pull = "beta" };
    const up = try decide(a, "1.0", "2.0", t);
    try testing.expectEqualStrings("2.0", up.available.latest);
    try testing.expectEqualStrings("1.0", up.available.local);
    try testing.expectEqualStrings("sushi pull beta", up.available.command);
    try testing.expect((try decide(a, "2.0", "2.0", t)) == .current);
    try testing.expect((try decide(a, "2.0", "1.0", t)) == .current);
    try testing.expect((try decide(a, null, "2.0", t)) == .unknown);
    try testing.expect((try decide(a, "1.0", null, t)) == .unknown);
    try testing.expect((try decide(a, "1.0", "garbage", t)) == .unknown);
}

test "rows: the list line and the /v1/models fragment appear for an update only" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const up = try decide(a, "1.0", "2.0", .{ .repo = "beamster/X", .pull = "beta" });
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try listLine(&w, up.available);
    try testing.expectEqualStrings("update available: 2.0 (you have 1.0) — sushi pull beta\n", w.buffered());
    w = std.Io.Writer.fixed(&buf);
    try metaFragment(&w, up);
    try testing.expectEqualStrings(",\"pack_update\":{\"current\":\"1.0\",\"latest\":\"2.0\",\"command\":\"sushi pull beta\",\"repo\":\"beamster/X\"}", w.buffered());
    w = std.Io.Writer.fixed(&buf);
    try metaFragment(&w, .current);
    try metaFragment(&w, .unknown);
    try testing.expectEqual(@as(usize, 0), w.buffered().len);
}

var test_fetches: usize = 0;
var test_body: ?[]const u8 = null;

fn countingFetch(_: Allocator, _: std.Io, _: []const u8) ?[]const u8 {
    test_fetches += 1;
    return test_body;
}

fn tmpCachePath(tmp: *testing.TmpDir, arena: Allocator) ![]const u8 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    return std.fmt.allocPrint(arena, "{s}/.sushi/pack-update-check.json", .{dir});
}

test "cache: Hugging Face is asked once a day per repo, across restarts, and a failed look is retried within the hour" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpCachePath(&tmp, a);
    const io = testing.io;
    test_fetches = 0;
    test_body = "{\"pack_revision\":{\"id\":\"2.0\"}}";
    const at = struct {
        fn env(p: []const u8, now: i64) Env {
            return .{ .cache_path = p, .now = now, .fetch = countingFetch };
        }
    }.env;

    const first = remoteRevision(a, io, "o/a", at(path, 1_000));
    try testing.expectEqualStrings("2.0", first.revision.?);
    try testing.expectEqual(day_s, first.wait_s);
    try testing.expectEqual(@as(usize, 1), test_fetches);

    // A restart an hour later reads the cache and waits out the rest of the day.
    const again = remoteRevision(a, io, "o/a", at(path, 1_000 + 3_600));
    try testing.expectEqualStrings("2.0", again.revision.?);
    try testing.expectEqual(day_s - 3_600, again.wait_s);
    try testing.expectEqual(@as(usize, 1), test_fetches);

    // Another repo has its own entry; the first stays.
    _ = remoteRevision(a, io, "o/b", at(path, 1_000 + 3_600));
    try testing.expectEqual(@as(usize, 2), test_fetches);
    _ = remoteRevision(a, io, "o/a", at(path, 1_000 + 7_200));
    try testing.expectEqual(@as(usize, 2), test_fetches);

    // A day on, it asks again.
    test_body = "{\"pack_revision\":{\"id\":\"2.1\"}}";
    const next = remoteRevision(a, io, "o/a", at(path, 1_000 + day_s));
    try testing.expectEqualStrings("2.1", next.revision.?);
    try testing.expectEqual(@as(usize, 3), test_fetches);

    // No answer claims nothing, leaves the cache alone and is retried within the hour.
    test_body = null;
    const down = remoteRevision(a, io, "o/a", at(path, 1_000 + 2 * day_s));
    try testing.expect(down.revision == null);
    try testing.expectEqual(retry_s, down.wait_s);
    try testing.expectEqualStrings("2.1", remoteRevision(a, io, "o/a", at(path, 1_000 + day_s + 5)).revision.?);

    // A clock set back does not hold a stale answer.
    test_body = "{\"pack_revision\":{\"id\":\"3.0\"}}";
    try testing.expectEqualStrings("3.0", remoteRevision(a, io, "o/a", at(path, 10)).revision.?);
}

test "cache: a missing or unreadable file is an empty cache, and no path means no cache" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpCachePath(&tmp, a);
    try testing.expectEqual(@as(usize, 0), readCache(a, testing.io, path).packs.len);
    try tmp.dir.createDirPath(testing.io, ".sushi");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = ".sushi/pack-update-check.json", .data = "not json" });
    try testing.expectEqual(@as(usize, 0), readCache(a, testing.io, path).packs.len);
    test_fetches = 0;
    test_body = "{}";
    const r = remoteRevision(a, testing.io, "o/a", .{ .cache_path = null, .now = 5, .fetch = countingFetch });
    try testing.expectEqualStrings("1.0", r.revision.?);
    _ = remoteRevision(a, testing.io, "o/a", .{ .cache_path = null, .now = 6, .fetch = countingFetch });
    try testing.expectEqual(@as(usize, 2), test_fetches);
}

test "look: a pack's status comes from the cached remote revision" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try tmpCachePath(&tmp, a);
    try tmp.dir.createDirPath(testing.io, "beta-pack");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "beta-pack/config.json", .data = "{\"pack_revision\":{\"id\":\"1.0\"}}" });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "beta-pack/.sushi-pull.json", .data = "{\"repo\":\"beamster/Beta-1017-Sushi-1.75bpw\"}" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const pack = try std.fmt.allocPrint(a, "{s}/beta-pack", .{buf[0..try tmp.dir.realPath(testing.io, &buf)]});
    test_fetches = 0;
    test_body = "{\"pack_revision\":{\"id\":\"2.0\"}}";
    const env: Env = .{ .cache_path = path, .now = 50, .fetch = countingFetch };
    const l = lookWith(a, testing.io, &test_known, pack, env);
    try testing.expectEqualStrings("2.0", l.status.available.latest);
    try testing.expectEqualStrings("sushi pull beta", l.status.available.command);
    test_body = null;
    const cached = lookWith(a, testing.io, &test_known, pack, env);
    try testing.expectEqualStrings("2.0", cached.status.available.latest);
    try testing.expectEqual(@as(usize, 1), test_fetches);
}

test "toggle: the published switch gates the watch loop's look" {
    defer setAuto(true);
    try testing.expect(autoOn());
    try testing.expect(looksNow(autoOn(), 100, 100));
    setAuto(false);
    try testing.expect(!autoOn());
    try testing.expect(!looksNow(autoOn(), 1_000_000, 0));
    setAuto(true);
    try testing.expect(looksNow(autoOn(), 1_000_000, 0));
    try testing.expect(!looksNow(true, 99, 100));
    try testing.expect(looksNow(true, 100, 100 + 2 * day_s));
}
