//! A pack's revision against the one published on Hugging Face. `config.json` carries `pack_revision.id`; a config
//! without it is a legacy pack, revision 1.0. `sushi list` and the server ask and say "update available" only when
//! the remote revision is higher; an answer that never arrived claims nothing either way.

const std = @import("std");
const log = @import("log.zig");
const hf_pull = @import("hf_pull.zig");

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

pub const Available = struct { local: []const u8, latest: []const u8, command: []const u8 };

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
    try std.json.Stringify.value(.{ .current = a.local, .latest = a.latest, .command = a.command }, .{}, w);
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

/// Where a local pack stands against its Hugging Face repo. Never throws: any failure is `.unknown`.
pub fn checkPack(arena: Allocator, io: std.Io, known: []const Known, pack_path: []const u8) Status {
    const folder = std.fs.path.basename(std.mem.trimEnd(u8, pack_path, "/"));
    const target = resolveTarget(arena, known, folder, readSmall(arena, io, pack_path, ".sushi-pull.json")) orelse return .unknown;
    const local = revisionOf(arena, readSmall(arena, io, pack_path, "config.json") orelse return .unknown);
    const remote = revisionOf(arena, fetchRemoteConfig(arena, io, target.repo) orelse return .unknown);
    return decide(arena, local, remote, target) catch .unknown;
}

// ── The server's look: after a load, then once a day ────────────────────

const slot_cap = 8;
const slot_meta_cap = 384;
const day_s: i64 = 24 * 60 * 60;
const retry_s: i64 = 60 * 60;

const Slot = struct {
    path: [std.fs.max_path_bytes]u8 = undefined,
    path_len: usize = 0,
    meta: [slot_meta_cap]u8 = undefined,
    meta_len: usize = 0,
};

var mu: std.c.pthread_mutex_t = .{};
var enabled = false;
var server_io: std.Io = undefined;
var server_known: []const Known = &.{};
var slots: [slot_cap]Slot = @splat(.{});

/// Called once at serve start: `on` is false under `--no-update-check`; `known` outlives the process.
pub fn configure(io: std.Io, on: bool, known: []const Known) void {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    server_io = io;
    server_known = known;
    enabled = on;
}

/// False under `--no-update-check`.
pub fn checksOn() bool {
    _ = std.c.pthread_mutex_lock(&mu);
    defer _ = std.c.pthread_mutex_unlock(&mu);
    return enabled;
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

fn watchLoop(i: usize) void {
    while (true) {
        var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        _ = std.c.pthread_mutex_lock(&mu);
        const path = path_buf[0..slots[i].path_len];
        @memcpy(path, slots[i].path[0..path.len]);
        const io = server_io;
        const known = server_known;
        _ = std.c.pthread_mutex_unlock(&mu);

        const st = checkPack(arena, io, known, path);
        record(path, st);
        if (st == .available) log.info("[pack-update] {s}: revision {s} is available (you have {s}): run `{s}`\n", .{ std.fs.path.basename(path), st.available.latest, st.available.local, st.available.command });

        const ts = std.c.timespec{ .sec = if (st == .unknown) retry_s else day_s, .nsec = 0 };
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
    try testing.expectEqualStrings(",\"pack_update\":{\"current\":\"1.0\",\"latest\":\"2.0\",\"command\":\"sushi pull beta\"}", w.buffered());
    w = std.Io.Writer.fixed(&buf);
    try metaFragment(&w, .current);
    try metaFragment(&w, .unknown);
    try testing.expectEqual(@as(usize, 0), w.buffered().len);
}
