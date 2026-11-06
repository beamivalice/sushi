//! `sushibar`: the menu bar item for a sushi server. It links no MLX and no server module: the title reads
//! `/metrics.json` over HTTP, Start and Stop run the `sushi` beside this binary, updates go through sushi's own
//! updater. AppKit is called only from the main thread; a poll thread publishes a snapshot that a main-run-loop
//! timer renders.
const std = @import("std");
const logic = @import("sushibar/sushibar_logic.zig");
const proc = @import("sushibar/sushibar_proc.zig");
const objc = @import("sushibar/objc.zig");
const io_util = @import("io_util.zig");
const header_view = @import("sushibar/header_view.zig");
const port_field = @import("sushibar/port_field.zig");
const button_row = @import("sushibar/button_row.zig");
const log_view = @import("sushibar/log_view.zig");
const port_row = @import("sushibar/port_row.zig");

const usage =
    \\usage: sushibar [--port <n>]
    \\
    \\Puts sushi in the menu bar: the generation speed in tok/s as the title, and a menu to start and stop a server,
    \\switch the port, bind the started servers to the LAN or this Mac alone, update sushi, and open
    \\the dashboard, chat, models and settings pages.
    \\It detaches from the terminal.
    \\
    \\  --port <n>   Port of the server to watch and start (default: the port last chosen
    \\               in the menu, else the default port: `server.port` in
    \\               ~/.sushi/model-settings.json, else {d})
    \\  --help       This text
    \\
;

const tick_ms = 1000;
const server_exe = "sushi";
const bar_exe = "sushibar";
const bar_pid_name = "sushibar.pid";
const saved_name = "sushibar.json";
const lock_name = "sushibar.lock";
const max_listings = 8;
const props_every = 5;

fn out(io: std.Io, comptime fmt: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    std.Io.File.stdout().writeStreamingAll(io, std.fmt.bufPrint(&buf, fmt, args) catch return) catch {};
}

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("sushibar: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn Str(comptime N: usize) type {
    return struct {
        buf: [N]u8 = undefined,
        len: usize = 0,

        fn set(self: *@This(), s: []const u8) void {
            self.len = @min(s.len, N);
            while (self.len < s.len and self.len > 0 and (s[self.len] & 0xC0) == 0x80) self.len -= 1;
            @memcpy(self.buf[0..self.len], s[0..self.len]);
        }

        fn get(self: *const @This()) []const u8 {
            return self.buf[0..self.len];
        }
    };
}

// ── Test seams: SUSHIBAR_TEST_POLLS, SUSHIBAR_TEST_ACTIONS (docs/envvar.md) ──

const ActionKind = enum { start, stop, load, unload, confirm, check, update, port, typed, models, remove, lan };
const Action = struct { kind: ActionKind, at: u32, value: u32 = 0 };
var test_polls: u32 = 0;
var test_actions: [12]Action = undefined;
var test_action_n: usize = 0;
var test_confirm = false;

fn readTestSeams() void {
    test_polls = std.fmt.parseInt(u32, env("SUSHIBAR_TEST_POLLS") orelse "", 10) catch 0;
    var it = std.mem.tokenizeScalar(u8, env("SUSHIBAR_TEST_ACTIONS") orelse "", ',');
    while (it.next()) |item| {
        if (test_action_n == test_actions.len) break;
        const at_i = std.mem.indexOfScalar(u8, item, '@') orelse continue;
        const word = item[0..at_i];
        var act: Action = .{ .kind = .start, .at = std.fmt.parseInt(u32, item[at_i + 1 ..], 10) catch continue };
        if (std.mem.startsWith(u8, word, "models")) {
            act.kind = .models;
            act.value = std.fmt.parseInt(u16, word[6..], 10) catch continue;
        } else if (std.mem.startsWith(u8, word, "port")) {
            act.kind = .port;
            act.value = std.fmt.parseInt(u16, word[4..], 10) catch continue;
        } else if (std.mem.startsWith(u8, word, "remove")) {
            act.kind = .remove;
            act.value = std.fmt.parseInt(u16, word[6..], 10) catch continue;
        } else if (std.mem.startsWith(u8, word, "type")) {
            act.kind = .typed;
            act.value = std.fmt.parseInt(u32, word[4..], 10) catch continue;
        } else act.kind = std.meta.stringToEnum(ActionKind, word) orelse continue;
        if (act.kind == .confirm) test_confirm = true;
        test_actions[test_action_n] = act;
        test_action_n += 1;
    }
}

// ── State shared by the poll thread, the update worker and the main thread ──

const Listing = struct { port: u16 = 0, pid: i32 = 0, model: Str(96) = .{}, known: logic.ModelsKnown = .unknown };

/// What a foreign listener's `/v1/models` last said; fetched only when the Port submenu opens.
const ModelCache = struct { port: u16 = 0, known: logic.ModelsKnown = .unknown, busy: bool = false, model: Str(96) = .{} };

const Shared = struct {
    mu: std.Io.Mutex = .init,
    /// The port the data below belongs to; stale once the watched port changes.
    port: u16 = 0,
    view: logic.View = .{ .link = .offline },
    model: Str(logic.max_model) = .{},
    /// The byte length of the ready model's id as the server sent it, which `model` cuts to `max_model`.
    model_full_len: usize = 0,
    listings: [max_listings]Listing = undefined,
    listing_n: usize = 0,
    /// The sushi listeners of the last scan with the address each is bound to (`originFor`).
    bound: [max_listings]logic.Listener = undefined,
    bound_n: usize = 0,
    models: [16]ModelCache = @splat(.{}),
    sushi_listening: bool = false,
    /// The watched port's sushi listener is bound to a LAN address (`lanBound`).
    lan_listening: bool = false,
    port_taken: bool = false,
    /// Started servers still to be asked whether they answer (`started`), and the ports that did.
    probe_ports: [8]u16 = undefined,
    probe_n: usize = 0,
    answered_ports: [8]u16 = undefined,
    answered_n: usize = 0,
    model_loading: Str(96) = .{},
    /// The ready model's `meta.pack_update`; empty `pack_latest` means none.
    pack_current: Str(32) = .{},
    pack_latest: Str(32) = .{},
    pack_command: Str(96) = .{},
    mem_mb: ?u64 = null,
    gpu_mhz: ?u64 = null,
    gpu_pct: ?u64 = null,
    last_prefill: ?f64 = null,
    hist: [200]logic.Hist = undefined,
    hist_n: usize = 0,
    bar_version: Str(32) = .{},
    server_version: Str(32) = .{},
    /// What the running server's `/props.update` says.
    props_latest: Str(32) = .{},
    props_available: bool = false,
    props_command: Str(80) = .{},
    latest: Str(32) = .{},
    available: bool = false,
    note: Str(240) = .{},
    /// The note says something went wrong: it is drawn orange and opens the Log window.
    note_problem: bool = false,
    note_kind: logic.NoteKind = .other,
    /// The port a `.server` note is about.
    note_port: u16 = 0,
    note_at_ms: i64 = 0,
    props_url: Str(160) = .{},
    busy: logic.Busy = .none,
    check_failed: bool = false,
    /// Where the watched port's server writes its log; stale once the watched port changes (`port`).
    log_kind: logic.LogKind = .standard,
    log_path: Str(1024) = .{},
    relaunch: bool = false,
    /// How the last check the user asked for ended, and when; shown on the Update button.
    manual_result: logic.CheckResult = .none,
    manual_at_ms: i64 = 0,
    /// The key the watched server's argv gives it (`--api-key`, `--api-key-env`), sent only to `key_port`.
    api_key: Str(512) = .{},
    key_port: u16 = 0,
};

var shared: Shared = .{};
var g_io: std.Io = undefined;
var g_port = std.atomic.Value(u16).init(logic.default_port);
/// The port the menu lists first and never lets go: `server.port` of the settings file, re-read as the menu opens.
var g_default_port: u16 = logic.default_port;
/// The ports the user added; the menu and the Port field are the main thread's.
var g_saved: logic.SavedPorts = .{};
/// The "Listen on LAN" switch: Start binds `lan_host` while it is on. The main thread's.
var g_lan: bool = false;
var g_home: []const u8 = "/tmp";
var g_sushi_found = false;
var g_sibling_brew = false;
/// The path sushibar was started by, unresolved: a Homebrew link follows the keg of the day.
var g_self_exe: [:0]const u8 = "";
var g_bar_pid_path: []const u8 = "";
var g_saved_path: []const u8 = "";
var g_capture_seq = std.atomic.Value(u32).init(0);
var g_launched_version: Str(32) = .{};
/// The model the watched server last had loaded (`sushibar.json`): Load brings it back.
var g_last_model: Str(logic.max_model) = .{};
/// The ready model's id was past `max_model`, so it was not stored and the note said so.
var g_model_too_long = false;
/// The port a server is being asked to load `g_last_model` on, 0 when none is.
var g_load_port = std.atomic.Value(u16).init(0);
/// The port a server is being asked to unload `g_last_model` on, 0 when none is.
var g_unload_port = std.atomic.Value(u16).init(0);
/// The model the running load or unload names; `g_last_model` follows the watched port and may differ.
var g_op_model: Str(logic.max_model) = .{};
/// A server the user's own Stop is ending, so its going away is no news.
var g_stop_port: u16 = 0;
var g_stop_at_ms: i64 = 0;

fn lock() void {
    shared.mu.lockUncancelable(g_io);
}
fn unlock() void {
    shared.mu.unlock(g_io);
}

fn port() u16 {
    return g_port.load(.acquire);
}

// ── HTTP ──

const Reply = struct { status: u16, body: []const u8 };

const poll_timeout_s = 4;
const models_timeout_s = 1;

/// A server that accepts and never answers is offline after `timeout_s`, not a stalled poll thread.
fn request(a: std.mem.Allocator, p: u16, method: std.http.Method, url: []const u8, origin: ?[]const u8) ?Reply {
    return requestWithin(a, p, method, url, origin, poll_timeout_s, "");
}

/// `p` is the port `url` names: only the watched server's own key is sent, and only to it.
fn requestWithin(a: std.mem.Allocator, p: u16, method: std.http.Method, url: []const u8, origin: ?[]const u8, timeout_s: i64, payload: []const u8) ?Reply {
    var auth_buf: [600]u8 = undefined;
    const auth = authFor(&auth_buf, p);
    const U = union(enum) { got: ?Reply, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(g_io, &buf);
    sel.concurrent(.got, fetchNow, .{ a, method, url, origin, auth, payload }) catch return fetchNow(a, method, url, origin, auth, payload);
    sel.concurrent(.timer, std.Io.sleep, .{ g_io, std.Io.Duration.fromSeconds(timeout_s), .awake }) catch {};
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

fn get(a: std.mem.Allocator, p: u16, url: []const u8) ?Reply {
    return request(a, p, .GET, url, null);
}

/// `Bearer <key>` when `p` is the watched port and its server's argv gave it a key, else "".
fn authFor(buf: []u8, p: u16) []const u8 {
    lock();
    defer unlock();
    if (shared.key_port != p or shared.api_key.len == 0) return "";
    return std.fmt.bufPrint(buf, "Bearer {s}", .{shared.api_key.get()}) catch "";
}

fn fetchNow(a: std.mem.Allocator, method: std.http.Method, url: []const u8, origin: ?[]const u8, auth: []const u8, payload: []const u8) ?Reply {
    var client: std.http.Client = .{ .allocator = a, .io = g_io };
    defer client.deinit();
    var body: std.Io.Writer.Allocating = .init(a);
    var hdr: [2]std.http.Header = undefined;
    var hn: usize = 0;
    if (origin) |o| {
        hdr[hn] = .{ .name = "origin", .value = o };
        hn += 1;
    }
    const res = client.fetch(.{
        .location = .{ .url = url },
        .method = method,
        .payload = if (method == .POST) payload else null,
        .headers = .{
            .content_type = if (payload.len > 0) .{ .override = "application/json" } else .default,
            .authorization = if (auth.len > 0) .{ .override = auth } else .default,
        },
        .extra_headers = hdr[0..hn],
        .keep_alive = false,
        .response_writer = &body.writer,
    }) catch return null;
    return .{ .status = @intFromEnum(res.status), .body = body.written() };
}

const origin_cap = 64;
const url_cap = 128;

/// `http://host:port` of the server on `p`: the address its listener is bound to, else loopback.
fn originFor(buf: []u8, p: u16) []const u8 {
    lock();
    defer unlock();
    return logic.originOf(buf, shared.bound[0..shared.bound_n], p);
}

fn urlFor(buf: []u8, p: u16, comptime path: []const u8) []const u8 {
    var origin: [origin_cap]u8 = undefined;
    return std.fmt.bufPrint(buf, "{s}" ++ path, .{originFor(&origin, p)}) catch unreachable;
}

fn modelsOf(a: std.mem.Allocator, p: u16) logic.ModelState {
    var b: [url_cap]u8 = undefined;
    const r = get(a, p, urlFor(&b, p, "/v1/models")) orelse return .{};
    if (r.status != 200) return .{};
    return logic.parseModelState(a, r.body) catch .{};
}

fn cacheSlot(p: u16) *ModelCache {
    var free: ?*ModelCache = null;
    for (&shared.models) |*c| {
        if (c.port == p) return c;
        if (c.port == 0 and free == null) free = c;
    }
    return free orelse &shared.models[p % shared.models.len];
}

fn metricsUrl(buf: []u8, p: u16, since_ms: i64) []const u8 {
    var origin: [origin_cap]u8 = undefined;
    return std.fmt.bufPrint(buf, "{s}/metrics.json?since={d}", .{ originFor(&origin, p), since_ms }) catch unreachable;
}

/// One foreign listener's models, with a short timeout, off the poll thread.
fn modelsWorker(p: u16) void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var b: [url_cap]u8 = undefined;
    var known: logic.ModelsKnown = .failed;
    var name: []const u8 = "";
    if (requestWithin(a, p, .GET, urlFor(&b, p, "/v1/models"), null, models_timeout_s, "")) |r| {
        if (r.status == 200) {
            if (logic.parseModelState(a, r.body)) |m| {
                known = .answered;
                name = m.ready orelse "";
            } else |_| {}
        }
    }
    lock();
    defer unlock();
    const c = cacheSlot(p);
    c.port = p;
    c.known = known;
    c.busy = false;
    c.model.set(name);
}

/// The Port submenu is about to be read: ask the listed ports' listeners for their models, in parallel. This is the
/// only time sushibar talks to a server it is not watching, and never to one the menu does not list.
fn refreshModels(only: ?u16) void {
    var ports: [max_listings]u16 = undefined;
    var n: usize = 0;
    lock();
    for (shared.listings[0..shared.listing_n]) |l| {
        if (l.port == port() or (only != null and only.? != l.port)) continue;
        if (l.port != g_default_port and !g_saved.has(l.port)) continue;
        const c = cacheSlot(l.port);
        if (c.busy) continue;
        c.port = l.port;
        c.busy = true;
        ports[n] = l.port;
        n += 1;
    }
    unlock();
    for (ports[0..n]) |p| {
        const t = std.Thread.spawn(.{}, modelsWorker, .{p}) catch {
            lock();
            cacheSlot(p).busy = false;
            unlock();
            continue;
        };
        t.detach();
    }
}

// ── Version and update (poll thread and worker) ──

const path_cap = std.fs.max_path_bytes + 1;

/// `name` beside this binary, resolved now: a Homebrew upgrade replaces the keg under a running sushibar, so the
/// path found at launch may be gone. Falls back to brew's stable `opt` link.
fn siblingNow(buf: *[path_cap]u8, name: []const u8) ?[:0]const u8 {
    var real: [std.fs.max_path_bytes]u8 = undefined;
    var cand: [std.fs.max_path_bytes]u8 = undefined;
    if (proc.realPath(&real, g_self_exe)) |r| {
        if (logic.siblingPath(&cand, r, name)) |c| if (reachable(c)) return sentinel(buf, c);
        if (logic.brewOptPath(&cand, r, name)) |c| if (reachable(c)) return sentinel(buf, c);
    }
    if (logic.siblingPath(&cand, g_self_exe, name)) |c| if (reachable(c)) return sentinel(buf, c);
    return null;
}

fn reachable(path: []const u8) bool {
    std.Io.Dir.cwd().access(g_io, path, .{ .execute = true }) catch return false;
    return true;
}

fn sentinel(buf: *[path_cap]u8, s: []const u8) ?[:0]const u8 {
    if (s.len >= buf.len) return null;
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return buf[0..s.len :0];
}

const Captured = struct { code: ?u8, text: []const u8 };

/// Runs `argv` to its end; its output goes to a file of its own, so concurrent callers never share one.
fn capture(a: std.mem.Allocator, argv: []const [*:0]const u8) Captured {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrintSentinel(&pb, "{s}/.sushi/sushibar-capture-{d}-{d}.txt", .{ g_home, proc.selfPid(), g_capture_seq.fetchAdd(1, .monotonic) }, 0) catch return .{ .code = null, .text = "" };
    const code = proc.runCapture(argv, path.ptr);
    const text = std.Io.Dir.cwd().readFileAlloc(g_io, path, a, .limited(64 * 1024)) catch "";
    std.Io.Dir.cwd().deleteFile(g_io, path) catch {};
    return .{ .code = code, .text = text };
}

fn refreshBarVersion(a: std.mem.Allocator) void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return;
    const got = capture(a, &.{ sushi.ptr, "--version" });
    if (got.code != 0) return;
    const v = logic.parseVersionOutput(got.text) orelse return;
    if (g_launched_version.len == 0) g_launched_version.set(v);
    lock();
    shared.bar_version.set(v);
    // Whatever replaced the sibling (our update, a terminal `sushi update`, a server's /v1/update) restarts us.
    if (logic.relaunchNeeded(g_launched_version.get(), v)) shared.relaunch = true;
    unlock();
}

fn setNote(text: []const u8) void {
    lock();
    defer unlock();
    putNote(text, false, .other, 0);
}

fn setProblem(text: []const u8) void {
    lock();
    defer unlock();
    putNote(text, true, .other, 0);
}

/// A note about the update flow: a finished check replaces it.
fn setUpdateNote(text: []const u8, problem: bool) void {
    lock();
    defer unlock();
    putNote(text, problem, .update, 0);
}

/// A problem with the server on `p`: it stays until a server answers there again or Start is pressed.
fn setServerProblem(text: []const u8, p: u16) void {
    lock();
    defer unlock();
    putNote(text, true, .server, p);
}

/// Holding the lock.
fn putNote(text: []const u8, problem: bool, kind: logic.NoteKind, p: u16) void {
    shared.note.set(text);
    shared.note_problem = problem;
    shared.note_kind = kind;
    shared.note_port = p;
    shared.note_at_ms = io_util.nowMs(g_io);
}

/// Holding the lock.
fn clearNote() void {
    putNote("", false, .other, 0);
}

/// Folds a check's answer into what the menu shows. False when it was a failure.
fn applyCheck(result: logic.Check, manual: bool) bool {
    lock();
    defer unlock();
    switch (result) {
        .available => |v| {
            shared.latest.set(v);
            shared.available = true;
            if (logic.checkReplacesNote(shared.note_kind)) clearNote();
        },
        .up_to_date => {
            shared.available = false;
            if (logic.checkReplacesNote(shared.note_kind)) clearNote();
        },
        .failed => |line| {
            if (manual) putNote(line, true, .update, 0);
            return false;
        },
    }
    return true;
}

/// A daily check answered from sushi's own cache while it is under a day old, so a bar launch asks no one.
fn cachedCheck(a: std.mem.Allocator) ?logic.Check {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&pb, "{s}/.sushi/update-check.json", .{g_home}) catch return null;
    const text = std.Io.Dir.cwd().readFileAlloc(g_io, path, a, .limited(64 * 1024)) catch return null;
    lock();
    const bar = a.dupe(u8, shared.bar_version.get()) catch "";
    unlock();
    return logic.cachedCheck(a, text, io_util.nowSecs(g_io), bar);
}

/// Whether `model-settings.json` leaves the automatic check on.
fn autoCheckEnabled(a: std.mem.Allocator) bool {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&pb, "{s}/.sushi/model-settings.json", .{g_home}) catch return true;
    const text = std.Io.Dir.cwd().readFileAlloc(g_io, path, a, .limited(1 << 20)) catch return true;
    return logic.autoCheckOn(a, text);
}

/// `sushi update --check`, recorded in sushi's own cache like the daily check. False when it gave no answer. The
/// automatic one reuses a cache under a day old; the menu's own "Check for updates" always asks.
fn cliCheck(a: std.mem.Allocator, manual: bool) bool {
    // Both an explicit and a daily check end in the header's "up to date" or "available"; only failures are worded.
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return false;
    refreshBarVersion(a);
    if (!manual) if (cachedCheck(a)) |hit| return applyCheck(hit, false);
    const got = capture(a, &.{ sushi.ptr, "update", "--check" });
    if (got.code == null) {
        if (manual) setUpdateNote("could not run sushi", true);
        return false;
    }
    return applyCheck(logic.parseCheck(got.text), manual);
}

fn refreshProps(a: std.mem.Allocator, p: u16) void {
    var b: [url_cap]u8 = undefined;
    const r = get(a, p, urlFor(&b, p, "/props")) orelse return;
    if (r.status != 200) return;
    const u = (logic.parseProps(a, r.body) catch return) orelse return;
    lock();
    // The watch moved while this was asked: the answer is another server's.
    if (port() != p) return unlock();
    const changed = shared.server_version.len > 0 and !std.mem.eql(u8, shared.server_version.get(), u.current);
    shared.server_version.set(u.current);
    shared.props_latest.set(u.latest orelse "");
    shared.props_available = u.available;
    shared.props_command.set(u.command orelse "");
    shared.props_url.set(u.url orelse "");
    unlock();
    if (changed) refreshBarVersion(a);
}

fn finishUpdate(a: std.mem.Allocator, ok_note: []const u8) void {
    refreshBarVersion(a);
    lock();
    defer unlock();
    shared.available = false;
    putNote(ok_note, false, .update, 0);
}

/// `error.message` of a server's JSON error body, or "".
fn errorMessage(a: std.mem.Allocator, body: []const u8) []const u8 {
    const E = struct { @"error": struct { message: []const u8 = "" } = .{} };
    const e = std.json.parseFromSliceLeaky(E, a, body, .{ .ignore_unknown_fields = true }) catch return "";
    return e.@"error".message;
}

fn updateViaServer(a: std.mem.Allocator, p: u16) void {
    var url: [url_cap]u8 = undefined;
    var origin: [origin_cap]u8 = undefined;
    const origin_s = originFor(&origin, p);
    var from_buf: [32]u8 = undefined;
    lock();
    var from: []const u8 = from_buf[0..shared.server_version.len];
    @memcpy(from_buf[0..from.len], shared.server_version.get());
    unlock();
    // Asked before the first /props of a newly watched port: without the version it runs, any answer reads as updated.
    if (from.len == 0) {
        var vb: [url_cap]u8 = undefined;
        if (get(a, p, urlFor(&vb, p, "/props"))) |pr| if (logic.parseProps(a, pr.body) catch null) |u| {
            from = from_buf[0..@min(u.current.len, from_buf.len)];
            @memcpy(from_buf[0..from.len], u.current[0..from.len]);
        };
    }
    if (from.len == 0) return setUpdateNote("Update failed: the server did not say which version it runs", true);
    const r = request(a, p, .POST, urlFor(&url, p, "/v1/update"), origin_s) orelse return setUpdateNote("Update failed: the server did not answer", true);
    if (r.status != 202) {
        const why = errorMessage(a, r.body);
        var msg: [400]u8 = undefined;
        return setUpdateNote(std.fmt.bufPrint(&msg, "Update refused: {s}", .{if (why.len > 0) why else "the server did not accept it"}) catch "Update refused", true);
    }
    setUpdateNote("Updating…", false);
    var down = false;
    var i: u32 = 0;
    while (i < update_wait_polls) : (i += 1) {
        std.Io.sleep(g_io, .fromMilliseconds(500), .real) catch {};
        if (port() != p) return setUpdateNote("Update started; no longer followed after the port change", false);
        var hb: [url_cap]u8 = undefined;
        const alive = if (get(a, p, urlFor(&hb, p, "/health"))) |h| h.status == 200 else false;
        if (!alive) {
            down = true;
            continue;
        }
        if (down) break;
        var pb: [url_cap]u8 = undefined;
        const props = get(a, p, urlFor(&pb, p, "/props")) orelse continue;
        const u = (logic.parseProps(a, props.body) catch null) orelse continue;
        if (!std.mem.eql(u8, u.current, from)) break;
    }
    var pb: [url_cap]u8 = undefined;
    const props = get(a, p, urlFor(&pb, p, "/props"));
    const u = if (props) |pr| (logic.parseProps(a, pr.body) catch null) else null;
    refreshProps(a, p);
    if (u) |up| if (!std.mem.eql(u8, up.current, from)) {
        var msg: [64]u8 = undefined;
        return finishUpdate(a, std.fmt.bufPrint(&msg, "Updated to sushi {s}", .{up.current}) catch "Updated");
    };
    var msg: [400]u8 = undefined;
    const why = if (u) |up| up.err orelse "sushi restarted on the version it had" else "the server did not come back";
    setUpdateNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{why}) catch "Update failed", true);
}

/// Two minutes of half-second polls: past that the update is reported as failed instead of locking the menu.
const update_wait_polls = 240;

fn updateViaCli(a: std.mem.Allocator) void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return setUpdateNote("could not find sushi beside sushibar", true);
    setUpdateNote("Updating…", false);
    const got = capture(a, &.{ sushi.ptr, "update" });
    if (got.code == 0) {
        var msg: [64]u8 = undefined;
        refreshBarVersion(a);
        lock();
        const v = shared.bar_version.get();
        const text = std.fmt.bufPrint(&msg, "Updated to sushi {s}", .{v}) catch "Updated";
        unlock();
        return finishUpdate(a, text);
    }
    var msg: [400]u8 = undefined;
    setUpdateNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{logic.failureLine(got.text)}) catch "Update failed", true);
}

const WorkerKind = enum { auto_check, check, update };

fn worker(kind: WorkerKind) void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const p = port();
    switch (kind) {
        .auto_check, .check => {
            const ok = cliCheck(a, kind == .check);
            lock();
            shared.check_failed = !ok;
            if (kind == .check) {
                shared.manual_result = if (!ok) .failed else if (shared.available) .none else .up_to_date;
                shared.manual_at_ms = io_util.nowMs(g_io);
            }
            unlock();
        },
        .update => {
            lock();
            const ask: logic.UpdateAsk = .{
                .available = shared.available,
                .link = shared.view.link,
                .server_command = shared.props_command.len > 0,
                .sibling_brew = g_sibling_brew,
            };
            unlock();
            switch (logic.updatePath(ask)) {
                .none => setUpdateNote("Nothing to update", false),
                .brew => setUpdateNote("Homebrew install: use the Upgrade button", false),
                .server => updateViaServer(a, p),
                .cli => updateViaCli(a),
            }
        },
    }
    lock();
    shared.busy = .none;
    unlock();
}

fn startWorker(kind: WorkerKind) void {
    lock();
    if (shared.busy != .none) {
        unlock();
        return;
    }
    shared.busy = if (kind == .update) .updating else .checking;
    unlock();
    if (kind == .update) pollSoon();
    const t = std.Thread.spawn(.{}, worker, .{kind}) catch {
        lock();
        shared.busy = .none;
        unlock();
        return;
    };
    t.detach();
}

// ── Poll thread ──

/// Wakes the poll thread before its wait is over: the menu opened, or the user started something.
var g_wake: std.Io.Event = .unset;
var g_menu_open = std.atomic.Value(bool).init(false);
/// Until then the poll runs every second: a start, stop, port switch or update the user just asked for.
var g_fast_until_ms = std.atomic.Value(i64).init(0);
const fast_after_action_ms = 30_000;

/// Polls at once and then every second for a while: the user just changed something.
fn pollSoon() void {
    g_fast_until_ms.store(io_util.nowMs(g_io) + fast_after_action_ms, .release);
    g_wake.set(g_io);
}

fn pollLoop() void {
    var tracker: logic.Tracker = .{};
    var tracked_port: u16 = 0;
    var last_check_ms: i64 = 0;
    var last_ok = true;
    var iter: u32 = 0;
    var log_pid: i32 = 0;
    var log_port: u16 = 0;
    var log_buf: [1024]u8 = undefined;
    var log_src: logic.LogSource = .{ .kind = .standard };
    // The scan and the models answer are kept between the polls that skip them.
    var listings: [max_listings]Listing = undefined;
    var n: usize = 0;
    var sushi_listening = false;
    var watched_lan = false;
    var since_scan: u32 = logic.scan_every;
    var since_models: u32 = 0;
    var models_ready = false;
    var models_loading = false;
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    {
        _ = arena_state.reset(.retain_capacity);
        refreshBarVersion(arena_state.allocator());
    }
    pollSoon();
    while (true) {
        const woken = g_wake.isSet();
        g_wake.reset();
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const now = io_util.nowMs(g_io);
        const p = port();
        if (p != tracked_port) {
            tracker = .{};
            tracked_port = p;
            since_scan = logic.scan_every;
            models_ready = false;
            lock();
            shared.hist_n = 0;
            shared.last_prefill = null;
            unlock();
        }
        const menu_open = g_menu_open.load(.acquire);
        var url_buf: [url_cap + 32]u8 = undefined;
        var again_buf: [url_cap + 32]u8 = undefined;
        var url = metricsUrl(&url_buf, p, tracker.since_ms);
        var reply = get(a, p, url);
        var reached = logic.linkFor(if (reply) |r| r.status else null);

        since_scan += 1;
        if (logic.scanDue(since_scan, reached, menu_open)) {
            since_scan = 0;
            var lbuf: [16]logic.Listener = undefined;
            var bound: [max_listings]logic.Listener = undefined;
            n = 0;
            sushi_listening = false;
            watched_lan = false;
            for (proc.listeners(&lbuf, server_exe)) |l| {
                if (l.port == p) {
                    sushi_listening = true;
                    if (logic.lanBound(l.addr)) watched_lan = true;
                }
                const dup = for (listings[0..n]) |x| {
                    if (x.port == l.port) break true;
                } else false;
                if (dup or n == max_listings) continue;
                listings[n] = .{ .port = l.port, .pid = l.pid };
                bound[n] = l;
                n += 1;
            }
            lock();
            shared.bound = bound;
            shared.bound_n = n;
            unlock();
            probeStarted(a, listings[0..n], p);
            // A server bound to a non-loopback address was asked on 127.0.0.1 above: ask where it listens.
            if (reply == null) {
                const again = metricsUrl(&again_buf, p, tracker.since_ms);
                if (!std.mem.eql(u8, again, url)) {
                    url = again;
                    reply = get(a, p, url);
                    reached = logic.linkFor(if (reply) |r| r.status else null);
                }
            }
        }

        // The server's own argv says where it logs and which key it wants; read again only when its pid changes.
        var server_pid: i32 = 0;
        for (listings[0..n]) |l| if (l.port == p) {
            server_pid = l.pid;
        };
        if (server_pid != log_pid or p != log_port) {
            log_pid = server_pid;
            log_port = p;
            const args = a.alloc(u8, proc.args_cap) catch null;
            const read = if (server_pid != 0 and args != null) proc.argsOfPid(server_pid, args.?) else null;
            log_src = logic.logSource(&log_buf, g_home, p, read);
            const key = if (read) |r| logic.argsApiKey(r) orelse "" else "";
            lock();
            shared.api_key.set(key);
            shared.key_port = p;
            unlock();
            // A key found just now answers the 401 it explains in this poll, not in the next.
            if (reached == .unauthorized and key.len > 0) {
                reply = get(a, p, url);
                reached = logic.linkFor(if (reply) |r| r.status else null);
            }
        }

        var sample: logic.Sample = .{ .link = reached };
        if (reached == .up) sample = logic.parseMetrics(a, reply.?.body) catch .{};
        // A sushi that holds the port and does not answer is not a stopped one.
        if (sample.link == .offline and sushi_listening) sample.link = .silent;
        tracker.observe(now, sample);

        var models: logic.ModelState = .{};
        const answers = sample.link == .up or sample.link == .metrics_off;
        since_models += 1;
        const fetch_models = answers and logic.modelsDue(since_models, models_ready, models_loading);
        if (fetch_models) {
            models = modelsOf(a, p);
            since_models = 0;
            models_ready = models.ready != null;
            models_loading = models.loading != null;
        } else if (!answers) {
            models_ready = false;
            models_loading = false;
        }
        const taken = sample.link == .offline and !sushi_listening and proc.portInUse(p);

        if (p == port()) {
            lock();
            shared.port = p;
            shared.view = tracker.view(sample.link);
            if (fetch_models or !answers) {
                shared.model.set(models.ready orelse "");
                shared.model_full_len = if (models.ready) |m| m.len else 0;
                shared.model_loading.set(models.loading orelse "");
                const pu = models.pack_update orelse logic.PackUpdate{};
                shared.pack_current.set(pu.current);
                shared.pack_latest.set(pu.latest);
                shared.pack_command.set(pu.command);
            }
            for (listings[0..n]) |*l| {
                if (l.port != p) continue;
                l.model.set(shared.model.get());
                l.known = if (answers) .answered else if (sample.link == .silent) .failed else .unknown;
            }
            if (sample.link == .up) {
                shared.mem_mb = sample.mem_mb;
                shared.gpu_mhz = sample.gpu_mhz;
                shared.gpu_pct = sample.gpu_pct;
                shared.last_prefill = tracker.last_prefill;
                for (sample.hist) |h| {
                    if (shared.hist_n > 0 and h.t_ms <= shared.hist[shared.hist_n - 1].t_ms) continue;
                    if (shared.hist_n == shared.hist.len) {
                        std.mem.copyForwards(logic.Hist, shared.hist[0 .. shared.hist.len - 1], shared.hist[1..]);
                        shared.hist_n -= 1;
                    }
                    shared.hist[shared.hist_n] = h;
                    shared.hist_n += 1;
                }
            }
            shared.listings = listings;
            shared.listing_n = n;
            shared.sushi_listening = sushi_listening;
            shared.lan_listening = if (sushi_listening) watched_lan else false;
            shared.port_taken = taken;
            shared.log_kind = log_src.kind;
            shared.log_path.set(log_src.path);
            unlock();
        }
        // Someone is looking: the menu shows this answer now, not at its next tick.
        if (woken and test_polls == 0) objc.msg(void, menu_target, "performSelectorOnMainThread:withObject:waitUntilDone:", .{ objc.sel("refreshNow:"), @as(objc.id, null), false });

        iter += 1;
        if (iter % props_every == 1 and answers) refreshProps(a, p);

        lock();
        if (shared.check_failed) {
            last_ok = false;
            shared.check_failed = false;
        }
        const link = shared.view.link;
        const known = shared.props_latest.len > 0;
        const busy = shared.busy != .none;
        unlock();
        if (logic.checkDue(last_check_ms, now, last_ok)) {
            if (link == .offline) {
                last_check_ms = now;
                last_ok = true;
                if (autoCheckEnabled(a)) startWorker(.auto_check);
            } else if (known) {
                last_check_ms = now;
                last_ok = true;
                refreshBarVersion(a);
                lock();
                shared.available = shared.props_available;
                shared.latest.set(shared.props_latest.get());
                unlock();
            }
        }

        const activity: logic.Activity = .{
            .link = sample.link,
            .decoding = tracker.decoding or tracker.prefilling,
            .menu_open = menu_open,
            .changing = now < g_fast_until_ms.load(.acquire) or models_loading or busy or g_load_port.load(.acquire) != 0 or g_unload_port.load(.acquire) != 0,
        };
        const spent = io_util.nowMs(g_io) - now;
        const wait = logic.pollWaitMs(activity) - spent;
        if (wait > 0) g_wake.waitTimeout(g_io, .{ .duration = .{ .raw = .fromMilliseconds(wait), .clock = .awake } }) catch {};
    }
}

// ── Port ──

fn writeSaved(p: u16) void {
    var text: [logic.saved_max_bytes]u8 = undefined;
    std.Io.Dir.cwd().createDirPath(g_io, std.fs.path.dirname(g_saved_path) orelse ".") catch {};
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = g_saved_path, .data = logic.formatSaved(&text, p, g_saved.items(), g_last_model.get(), g_lan) }) catch {};
}

fn readDefaultPort() u16 {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&pb, "{s}/.sushi/model-settings.json", .{g_home}) catch return logic.default_port;
    const text = std.Io.Dir.cwd().readFileAlloc(g_io, path, arena_state.allocator(), .limited(1 << 20)) catch return logic.default_port;
    return logic.defaultPortOf(arena_state.allocator(), text);
}

fn setPort(p: u16) void {
    if (p == port()) return;
    g_port.store(p, .release);
    pollSoon();
    g_saved.add(p, g_default_port);
    writeSaved(p);
    lock();
    shared.view = .{ .link = .offline };
    shared.model.set("");
    shared.model_full_len = 0;
    shared.server_version.set("");
    shared.props_latest.set("");
    shared.props_available = false;
    shared.props_command.set("");
    shared.props_url.set("");
    unlock();
}

/// Forgets an added port; the watch moves to the default port when it was on it. A sushi still listening there
/// stays in the menu as a listener, and the server is never touched.
fn removePort(p: u16) void {
    if (p == g_default_port) return;
    g_saved.remove(p);
    const to = logic.afterRemove(port(), p, g_default_port);
    if (to != port()) setPort(to) else writeSaved(port());
}

// ── Start and Stop ──

/// One record per port, so a server started on port A stays tracked after Start on port B.
fn serverPidPath(buf: []u8, p: u16) []const u8 {
    return std.fmt.bufPrint(buf, "{s}/.sushi/{s}{d}.pid", .{ g_home, logic.server_pid_prefix, p }) catch "";
}

fn ownServer() ?logic.PidEntry {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const e = proc.verifiedPid(g_io, serverPidPath(&pb, port()), server_exe, true) orelse return null;
    return if (e.port == port()) e else null;
}

/// Servers this process spawned: each is reaped by its own pid, and one that exits before it ever answered is a
/// failed start.
var started: [8]logic.Started = @splat(.{});

fn reapStarted() void {
    for (&started) |*st| {
        if (st.pid == 0) continue;
        var status: c_int = 0;
        const got = std.c.waitpid(st.pid, &status, std.c.W.NOHANG);
        // An adopted server another sushibar started is not ours to wait for.
        if (got == -1 and std.posix.errno(got) == .CHILD) st.* = .{};
        if (got != st.pid) continue;
        if (!st.stopping) {
            var nb: [160]u8 = undefined;
            const how = logic.exitOf(@bitCast(status));
            setServerProblem(logic.startedExitNote(&nb, st.*, how), st.port);
            if (st.answered) g_watch.reaped(st.port, io_util.nowMs(g_io));
        }
        st.* = .{};
    }
}

/// After a relaunch (an `exec`, same pid) the servers started before it are still this process's children: take
/// them back from their pid files, so they are reaped, their exit reported and Quit & Stop stops them.
fn adoptStarted() void {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const dir_path = std.fmt.bufPrint(&pb, "{s}/.sushi", .{g_home}) catch return;
    var dir = std.Io.Dir.openDirAbsolute(g_io, dir_path, .{ .iterate = true }) catch return;
    defer dir.close(g_io);
    var slot: usize = 0;
    var it = dir.iterate();
    while (it.next(g_io) catch null) |entry| {
        if (slot == started.len) break;
        const p = logic.serverPidPort(entry.name) orelse continue;
        var fb: [std.fs.max_path_bytes]u8 = undefined;
        const e = proc.verifiedPid(g_io, serverPidPath(&fb, p), server_exe, true) orelse continue;
        started[slot] = .{ .pid = e.pid, .port = p, .answered = true };
        slot += 1;
    }
}

/// The poll thread asks each started server that has not answered, during its process scan, and the menu thread
/// learns the answers here: an exit then reads "exited", on whatever port the server runs.
fn syncProbes() void {
    var pend: [started.len]u16 = undefined;
    const ports = logic.unansweredPorts(&started, &pend);
    lock();
    defer unlock();
    for (shared.answered_ports[0..shared.answered_n]) |ap| logic.markAnswered(&started, ap);
    shared.answered_n = 0;
    shared.probe_n = ports.len;
    @memcpy(shared.probe_ports[0..ports.len], ports);
}

fn probeStarted(a: std.mem.Allocator, listings: []const Listing, watched: u16) void {
    var ports: [started.len]u16 = undefined;
    lock();
    const n = shared.probe_n;
    @memcpy(ports[0..n], shared.probe_ports[0..n]);
    unlock();
    for (ports[0..n]) |q| {
        if (q == watched) continue;
        for (listings) |l| if (l.port == q) {
            var b: [url_cap]u8 = undefined;
            if (requestWithin(a, q, .GET, urlFor(&b, q, "/health"), null, models_timeout_s, "") == null) break;
            lock();
            if (shared.answered_n < shared.answered_ports.len) {
                shared.answered_ports[shared.answered_n] = q;
                shared.answered_n += 1;
            }
            unlock();
            break;
        };
    }
}

fn markStopping(pid: i32) void {
    for (&started) |*st| if (st.pid == pid) {
        st.stopping = true;
    };
}

fn startServer() void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return setProblem("could not find sushi beside sushibar");
    if (ownServer() != null) return;
    var port_buf: [8:0]u8 = undefined;
    const p = port();
    const port_z = std.fmt.bufPrintSentinel(&port_buf, "{d}", .{p}, 0) catch return;
    const argv_loopback = [_][*:0]const u8{ sushi.ptr, "--serve", "--port", port_z.ptr };
    const argv_lan = [_][*:0]const u8{ sushi.ptr, "--serve", "--port", port_z.ptr, "--host", logic.lan_host };
    const argv: []const [*:0]const u8 = if (g_lan) &argv_lan else &argv_loopback;
    const pid = proc.spawnDetached(argv, false) catch return setProblem("could not start sushi");
    setNote("");
    pollSoon();
    g_watch.started(p);
    for (&started) |*st| if (st.pid == 0) {
        st.* = .{ .pid = pid, .port = p };
        break;
    };
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    proc.writePid(g_io, g_home, serverPidPath(&pb, p), .{ .pid = pid, .port = p }) catch {};
}

const ModelOp = struct { port: u16, unload: bool, model: Str(logic.max_model) };

const load_timeout_s = 1800;

/// Load or unload `g_last_model` on the watched server, the only model either button acts on. The button offers
/// neither before the server answers, so a Start never loads.
fn startModelOp(unload: bool) void {
    const p = port();
    if (g_load_port.load(.acquire) != 0 or g_unload_port.load(.acquire) != 0 or g_last_model.len == 0) return;
    const slot = if (unload) &g_unload_port else &g_load_port;
    if (slot.cmpxchgStrong(0, p, .acq_rel, .acquire) != null) return;
    g_op_model = g_last_model;
    const t = std.Thread.spawn(.{}, modelOpWorker, .{ModelOp{ .port = p, .unload = unload, .model = g_last_model }}) catch return slot.store(0, .release);
    t.detach();
    pollSoon();
}

fn modelOpWorker(job: ModelOp) void {
    defer {
        (if (job.unload) &g_unload_port else &g_load_port).store(0, .release);
        pollSoon();
    }
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var url: [url_cap]u8 = undefined;
    var origin: [origin_cap]u8 = undefined;
    const origin_s = originFor(&origin, job.port);
    var body: [logic.max_model * 6 + 40]u8 = undefined;
    const target = if (job.unload) urlFor(&url, job.port, "/v1/unload-model") else urlFor(&url, job.port, "/v1/load-model");
    const payload = if (job.unload) logic.unloadModelBody(&body, job.model.get()) else logic.loadModelBody(&body, job.model.get());
    const reply = requestWithin(a, job.port, .POST, target, origin_s, load_timeout_s, payload);
    const verb = if (job.unload) "unload" else "load";
    var msg: [240]u8 = undefined;
    if (reply) |r| if (r.status == 200) {
        lock();
        if (shared.port == job.port) {
            shared.model.set(if (job.unload) "" else job.model.get());
            shared.model_full_len = shared.model.len;
        }
        unlock();
        return;
    };
    // A server that was stopped under the request is reported by its own exit note.
    if (get(a, job.port, urlFor(&url, job.port, "/health")) == null) return;
    const why = if (reply) |r| errorMessage(a, r.body) else "";
    setProblem(std.fmt.bufPrint(&msg, "Port {d}: could not {s} {s}: {s}", .{ job.port, verb, job.model.get(), if (why.len > 0) why else if (reply == null) "no answer" else "the server refused" }) catch "Could not load or unload the model");
}

const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
extern "c" fn localtime_r(t: *const i64, out: *Tm) ?*Tm;

fn stoppedAtNote(buf: []u8, p: u16) []const u8 {
    const t = io_util.nowSecs(g_io);
    var tm: Tm = undefined;
    if (localtime_r(&t, &tm) == null) return logic.stoppedNote(buf, p, 0, 0);
    return logic.stoppedNote(buf, p, @intCast(tm.hour), @intCast(tm.min));
}

const NSRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

const alert_first = 1000;
const alert_third = 1002;

/// Runs the alert and returns which button ended it (`alert_first` is the default one).
fn runAlert(message: []const u8, info: []const u8, ok: [*:0]const u8, cancel: ?[*:0]const u8, third: ?[*:0]const u8, field: objc.id) isize {
    var m: [160:0]u8 = undefined;
    var i: [200:0]u8 = undefined;
    const mz = std.fmt.bufPrintSentinel(&m, "{s}", .{message}, 0) catch return 0;
    const iz = std.fmt.bufPrintSentinel(&i, "{s}", .{info}, 0) catch return 0;
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "activateIgnoringOtherApps:", .{true});
    const alert = objc.new(objc.class("NSAlert"));
    defer objc.release(alert);
    objc.msg(void, alert, "setMessageText:", .{objc.nsString(mz.ptr)});
    objc.msg(void, alert, "setInformativeText:", .{objc.nsString(iz.ptr)});
    _ = objc.msg(objc.id, alert, "addButtonWithTitle:", .{objc.nsString(ok)});
    if (cancel) |c| _ = objc.msg(objc.id, alert, "addButtonWithTitle:", .{objc.nsString(c)});
    if (third) |t| _ = objc.msg(objc.id, alert, "addButtonWithTitle:", .{objc.nsString(t)});
    if (field != null) {
        objc.msg(void, alert, "setAccessoryView:", .{field});
        objc.msg(void, objc.msg(objc.id, alert, "window", .{}), "setInitialFirstResponder:", .{field});
    }
    return objc.msg(isize, alert, "runModal", .{});
}

fn confirmStop(p: u16, pid: i32) bool {
    if (test_polls > 0) return test_confirm;
    var m: [96]u8 = undefined;
    const text = std.fmt.bufPrint(&m, "Stop sushi on port {d} (pid {d})?", .{ p, pid }) catch return false;
    return runAlert(text, "sushibar did not start it.", "Stop", "Cancel", null, null) == alert_first;
}

/// The Update button on a server with a model loaded: the restart is named before it happens.
fn confirmUpdate() bool {
    if (test_polls > 0) return test_confirm;
    var m: [96]u8 = undefined;
    var i: [160]u8 = undefined;
    const text = std.fmt.bufPrint(&m, "Update sushi to {s}?", .{confirm_latest.get()}) catch return false;
    const info = logic.updateConfirmInfo(&i, port(), confirm_model.get());
    const answer = runAlert(text, info, "Update", "Cancel", "Release Notes", null);
    if (answer == alert_third) openReleaseNotes(null, null, null);
    return answer == alert_first;
}

/// How long the menu says "Stopping…" before it trusts what the poll says.
const stopping_ms = 15_000;
var stopping_until_ms: i64 = 0;
/// The pid the last Stop signalled, which may outlive its port by a while, and that port.
var g_stop_pid: i32 = 0;
var g_stopping_port: u16 = 0;

/// The Stop last sent, for Force Stop when it does not take.
var g_stop_sent: logic.StopSent = .{};
var g_stuck_noted = false;

fn noteStopped(p: u16, pid: i32) void {
    pollSoon();
    g_stop_pid = pid;
    g_stopping_port = p;
    stopping_until_ms = io_util.nowMs(g_io) + stopping_ms;
    g_stop_port = p;
    g_stop_at_ms = stopping_until_ms;
    g_stop_sent = .{ .pid = pid, .port = p, .deadline_ms = stopping_until_ms };
}

/// SIGKILL, only to the pid the Stop signalled while it is still the sushi on the watched port.
fn forceStop() void {
    const p = port();
    var lbuf: [16]logic.Listener = undefined;
    const ours = ownServer();
    const pid = switch (logic.planStop(if (ours) |e| e.pid else null, p, proc.listeners(&lbuf, server_exe))) {
        .now, .confirm => |pid| pid,
        .none => return,
    };
    if (pid != g_stop_sent.pid) return;
    markStopping(pid);
    proc.kill(pid);
    noteStopped(p, pid);
}

/// A sushibar-started server stops at once; any other sushi on the port only after a confirm, re-verified.
fn stopServer() void {
    const p = port();
    var lbuf: [16]logic.Listener = undefined;
    const ours = ownServer();
    switch (logic.planStop(if (ours) |e| e.pid else null, p, proc.listeners(&lbuf, server_exe))) {
        .none => {},
        .now => |pid| {
            markStopping(pid);
            proc.terminate(pid);
            noteStopped(p, pid);
        },
        .confirm => |pid| {
            if (!confirmStop(p, pid)) return;
            var again: [16]logic.Listener = undefined;
            switch (logic.planStop(null, p, proc.listeners(&again, server_exe))) {
                .confirm => |same| if (same == pid) {
                    proc.terminate(pid);
                    noteStopped(p, pid);
                },
                else => {},
            }
        },
    }
}

/// The ports of the servers this sushibar started that still run, the watched port's included.
fn startedPorts(ports: []u16) []u16 {
    var n: usize = 0;
    var cand: [started.len + 1]u16 = undefined;
    var cn: usize = 0;
    for (started) |st| if (st.pid != 0) {
        cand[cn] = st.port;
        cn += 1;
    };
    cand[cn] = port();
    cn += 1;
    for (cand[0..cn]) |p| {
        if (n == ports.len or std.mem.indexOfScalar(u16, ports[0..n], p) != null) continue;
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        if (proc.verifiedPid(g_io, serverPidPath(&pb, p), server_exe, true) == null) continue;
        ports[n] = p;
        n += 1;
    }
    return ports[0..n];
}

/// Quit & Stop: each server sushibar started gets the Stop it would get from the menu, then sushibar quits.
fn quitStop() void {
    var ports: [started.len + 1]u16 = undefined;
    for (startedPorts(&ports)) |p| {
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        const e = proc.verifiedPid(g_io, serverPidPath(&pb, p), server_exe, true) orelse continue;
        markStopping(e.pid);
        proc.terminate(e.pid);
    }
    quit();
}

fn quit() void {
    if (proc.verifiedPid(g_io, g_bar_pid_path, bar_exe, false)) |e| {
        if (e.pid == proc.selfPid()) proc.removePid(g_io, g_bar_pid_path);
    }
    objc.msg(void, objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{}), "terminate:", .{@as(objc.id, null)});
}

// ── Menu bar ──

var status_item: objc.id = null;
var header_mi: objc.id = null;
var port_mi: objc.id = null;
var lan_mi: objc.id = null;
var port_menu: objc.id = null;
var open_mis: [4]objc.id = .{ null, null, null, null };
var button_action: logic.ButtonAction = .none;
var pack_mi: objc.id = null;
var badge_view: objc.id = null;
var shown_badge = false;
var g_watch: logic.Watch = .{};
/// What was copied to the clipboard last, and when: the button and the pack item say Copied for a few seconds.
var copied_what: enum { none, upgrade, pack } = .none;
var copied_at_ms: i64 = 0;
/// Set by `refresh` for the Update button's click: the brew command it copies, and whether an alert comes first.
var button_command: Str(80) = .{};
var confirm_needed = false;
var confirm_model: Str(96) = .{};
var confirm_latest: Str(32) = .{};
var shown_note_sig: [600]u8 = undefined;
var shown_note_sig_len: usize = std.math.maxInt(usize);
var menu_target: objc.id = null;
var top_menu: objc.id = null;
var port_field_mi: objc.id = null;
var note_mi: objc.id = null;
var icons: [2]objc.id = .{ null, null };
var shown_icon: ?logic.Icon = null;
var tick_count: u32 = 0;
var last_title: [128]u8 = undefined;
var shown_title: [128]u8 = undefined;
var shown_title_len: usize = std.math.maxInt(usize);
var port_sig: [1536]u8 = undefined;
var port_sig_len: usize = std.math.maxInt(usize);

fn setText(setter_target: objc.id, selector: [*:0]const u8, text: []const u8) void {
    var z: [256:0]u8 = undefined;
    const n = @min(text.len, z.len - 1);
    @memcpy(z[0..n], text[0..n]);
    z[n] = 0;
    objc.msg(void, setter_target, selector, .{objc.nsString(&z)});
}

fn symbol(name: []const u8) objc.id {
    var z: [48:0]u8 = undefined;
    const n = @min(name.len, z.len - 1);
    @memcpy(z[0..n], name[0..n]);
    z[n] = 0;
    return objc.msg(objc.id, objc.class("NSImage"), "imageWithSystemSymbolName:accessibilityDescription:", .{ objc.nsString(&z), @as(objc.id, null) });
}

fn showIcon(button: objc.id, icon: logic.Icon) void {
    if (shown_icon == icon) return;
    if (shown_icon != null) objc.fadeNext(button, 0.3);
    shown_icon = icon;
    objc.msg(void, button, "setImage:", .{icons[@intFromEnum(icon)]});
}

/// The button title in monospaced digits, so its width does not jitter.
fn setButtonTitle(button: objc.id, text: []const u8) void {
    if (shown_title_len == text.len and std.mem.eql(u8, shown_title[0..text.len], text)) return;
    shown_title_len = text.len;
    @memcpy(shown_title[0..text.len], text);
    var z: [192:0]u8 = undefined;
    @memcpy(z[0..text.len], text);
    z[text.len] = 0;
    const s = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsString(&z), header_view.titleAttributes() });
    defer objc.release(s);
    objc.msg(void, button, "setAttributedTitle:", .{s});
}

fn newItem(label: [*:0]const u8, action: ?[*:0]const u8, tag: isize, key: [*:0]const u8, sym: ?[]const u8) objc.id {
    const mi = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSMenuItem"), "alloc", .{}), "initWithTitle:action:keyEquivalent:", .{
        objc.nsString(label), if (action) |a| objc.sel(a) else null, objc.nsString(key),
    });
    objc.msg(void, mi, "setTarget:", .{menu_target});
    objc.msg(void, mi, "setTag:", .{tag});
    objc.msg(void, mi, "setEnabled:", .{action != null});
    if (sym) |name| objc.msg(void, mi, "setImage:", .{symbol(name)});
    return mi;
}

fn addItem(menu: objc.id, label: [*:0]const u8, action: ?[*:0]const u8, tag: isize, key: [*:0]const u8, sym: ?[]const u8) objc.id {
    const mi = newItem(label, action, tag, key, sym);
    objc.msg(void, menu, "addItem:", .{mi});
    return mi;
}

fn addSeparator(menu: objc.id) void {
    objc.msg(void, menu, "addItem:", .{objc.msg(objc.id, objc.class("NSMenuItem"), "separatorItem", .{})});
}

/// Rebuilds the Port submenu only when its entries changed, so an open menu is not disturbed.
fn syncPortMenu(entries: []const logic.PortEntry) void {
    var sig: [1536]u8 = undefined;
    var n: usize = 0;
    var lb: [96]u8 = undefined;
    for (entries) |e| {
        const l = logic.portLabel(&lb, e);
        const piece = std.fmt.bufPrint(sig[n..], "{s}{d}{d}\n", .{ l, @intFromBool(e.checked), @intFromBool(e.removable) }) catch break;
        n += piece.len;
    }
    if (n == port_sig_len and std.mem.eql(u8, sig[0..n], port_sig[0..n])) return;
    port_sig_len = n;
    @memcpy(port_sig[0..n], sig[0..n]);
    // The submenu ends with a separator, the Listen on LAN item, another separator and the port field, which
    // stay; only the port rows above them change.
    port_row.reset();
    while (objc.msg(isize, port_menu, "numberOfItems", .{}) > 4) objc.msg(void, port_menu, "removeItemAtIndex:", .{@as(isize, 0)});
    var width: f64 = port_row.min_width;
    for (entries) |e| width = @max(width, port_row.widthFor(logic.portLabel(&lb, e), e.removable));
    for (entries, 0..) |e, i| {
        var z: [112:0]u8 = undefined;
        const label = logic.portLabel(&lb, e);
        const lz = std.fmt.bufPrintSentinel(&z, "{s}", .{label}, 0) catch continue;
        const mi = newItem(lz.ptr, "pickPort:", e.port, "", null);
        defer objc.release(mi);
        const view = port_row.create(menu_target, top_menu, .{ .port = e.port, .label = label, .checked = e.checked, .removable = e.removable }, width);
        if (view) |v| {
            objc.msg(void, mi, "setView:", .{v});
            objc.release(v);
        } else if (e.checked) objc.msg(void, mi, "setState:", .{@as(isize, 1)});
        objc.msg(void, port_menu, "insertItem:atIndex:", .{ mi, @as(isize, @intCast(i)) });
    }
}

/// A server that was answering is gone: say so unless sushibar stopped it or reported its exit.
fn watchTransition(p: u16, fresh: bool, link: logic.Link, now: i64) void {
    if (!fresh) return;
    const mine = g_stop_port == p and now < g_stop_at_ms + 45_000;
    if (g_watch.observe(p, link, now, mine)) {
        var nb: [96]u8 = undefined;
        setServerProblem(stoppedAtNote(&nb, p), p);
    }
    if (link == .offline and mine) g_stop_port = 0;
}

/// `pid` is still a sushi process (a zombie is not: its children are reaped before this is asked).
fn sushiAlive(pid: i32) bool {
    if (pid <= 0) return false;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe = proc.exeOfPid(pid, &buf) orelse return false;
    return std.mem.eql(u8, std.fs.path.basename(exe), server_exe);
}

/// The orange dot on the menu bar icon, at its lower right: it sits on the colour icon where the sushi does not hide it.
fn showBadge(button: objc.id, on: bool) void {
    if (on == shown_badge) return;
    objc.fadeNext(button, 0.25);
    shown_badge = on;
    if (on) {
        const r = objc.msg(objc.NSRect, objc.msg(objc.id, button, "cell", .{}), "imageRectForBounds:", .{objc.msg(objc.NSRect, button, "bounds", .{})});
        const flipped = objc.msg(bool, button, "isFlipped", .{});
        objc.msg(void, badge_view, "setFrameOrigin:", .{objc.NSPoint{ .x = r.x + r.w - badge_size + 1, .y = if (flipped) r.y + r.h - badge_size + 2 else r.y - 2 }});
    }
    objc.msg(void, badge_view, "setHidden:", .{!on});
}

const badge_size: f64 = 7;

fn makeBadge(button: objc.id) void {
    badge_view = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSView"), "alloc", .{}), "initWithFrame:", .{objc.NSRect{ .x = 0, .y = 0, .w = badge_size, .h = badge_size }});
    objc.msg(void, badge_view, "setWantsLayer:", .{true});
    const layer = objc.msg(objc.id, badge_view, "layer", .{});
    objc.msg(void, layer, "setBackgroundColor:", .{objc.msg(objc.id, objc.color("systemOrangeColor"), "CGColor", .{})});
    objc.msg(void, layer, "setCornerRadius:", .{badge_size / 2});
    objc.msg(void, badge_view, "setHidden:", .{true});
    objc.msg(void, button, "addSubview:", .{badge_view});
}

var why_mi: objc.id = null;
var shown_why: []const u8 = "";
/// What the Load/Unload button does at the next click, from the last refresh.
var model_action: logic.ModelAction = .none;

/// The inert line under the two buttons that says why Load is off; hidden when nothing needs saying.
fn showWhy(why: []const u8) void {
    if (std.mem.eql(u8, why, shown_why)) return;
    shown_why = why;
    objc.msg(void, why_mi, "setHidden:", .{why.len == 0});
    if (why.len > 0) setText(why_mi, "setTitle:", why);
}

/// The note item: grey and inert for news, orange and a click to the Log window for a problem. The menu keeps its
/// width by drawing `title`; the tooltip has the whole `full` text.
fn showNote(title: []const u8, full: []const u8, problem: bool) void {
    objc.msg(void, note_mi, "setHidden:", .{title.len == 0});
    var sig: [600]u8 = undefined;
    const key = std.fmt.bufPrint(&sig, "{s}\x00{s}\x00{d}", .{ title, full, @intFromBool(problem) }) catch return;
    if (key.len == shown_note_sig_len and std.mem.eql(u8, key, shown_note_sig[0..key.len])) return;
    shown_note_sig_len = key.len;
    @memcpy(shown_note_sig[0..key.len], key);
    if (title.len == 0) return;
    setText(note_mi, "setTitle:", title);
    setText(note_mi, "setToolTip:", full);
    objc.msg(void, note_mi, "setEnabled:", .{problem});
    objc.msg(void, note_mi, "setAction:", .{if (problem) objc.sel("showLog:") else null});
    if (!problem) return objc.msg(void, note_mi, "setAttributedTitle:", .{@as(objc.id, null)});
    var z: [256:0]u8 = undefined;
    const n = @min(title.len, z.len - 1);
    @memcpy(z[0..n], title[0..n]);
    z[n] = 0;
    const attrs = header_view.noteAttributes();
    const text = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsString(&z), attrs });
    defer objc.release(text);
    objc.msg(void, note_mi, "setAttributedTitle:", .{text});
}

fn runAction(act: Action) void {
    switch (act.kind) {
        .start => startServer(),
        .stop => stopServer(),
        .load => if (model_action == .load) startModelOp(false),
        .unload => if (model_action == .unload) startModelOp(true),
        .confirm => {},
        .models => refreshModels(@intCast(act.value)),
        .check => {
            lock();
            shared.manual_result = .none;
            unlock();
            startWorker(.check);
        },
        .update => startWorker(.update),
        .port => setPort(@intCast(act.value)),
        .remove => removePort(@intCast(act.value)),
        .lan => toggleLan(),
        .typed => {
            var z: [16:0]u8 = undefined;
            const t = std.fmt.bufPrintSentinel(&z, "{d}", .{act.value}, 0) catch return;
            port_field.setText(t.ptr);
            applyPortText();
        },
    }
}

/// Return in the field or Switch: a valid port switches what is watched and closes the menu; anything else turns
/// the text red with a hint and keeps the menu open.
fn applyPortText() void {
    if (logic.parsePortText(port_field.text())) |p| {
        port_field.clear();
        setPort(p);
        objc.msg(void, top_menu, "cancelTracking", .{});
    } else port_field.showInvalid();
}

fn take(buf: []u8, s: anytype) []const u8 {
    @memcpy(buf[0..s.len], s.get());
    return buf[0..s.len];
}

fn tick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    tick_count += 1;
    for (test_actions[0..test_action_n]) |act| if (act.at == tick_count) runAction(act);
    refresh();
}

/// Renders the published snapshot into the menu, from the timer and straight after a click that changes it.
fn refresh() void {
    // Before the snapshot is read: an exit is reported as such before the poll can report the port gone.
    reapStarted();
    const p = port();
    const now = io_util.nowMs(g_io);
    var model_b: [logic.max_model]u8 = undefined;
    var loading_b: [96]u8 = undefined;
    var bar_b: [32]u8 = undefined;
    var srv_b: [32]u8 = undefined;
    var latest_b: [32]u8 = undefined;
    var cmd_b: [80]u8 = undefined;
    var note_b: [240]u8 = undefined;
    var listings: [max_listings]logic.Listing = undefined;
    var list_models: [max_listings][96]u8 = undefined;
    var hist: [200]logic.Hist = undefined;

    lock();
    const fresh = shared.port == p;
    const view: logic.View = if (fresh) shared.view else .{ .link = .offline };
    const model = if (fresh) take(&model_b, &shared.model) else "";
    const loading_id = if (fresh) take(&loading_b, &shared.model_loading) else "";
    var pack_cur_b: [32]u8 = undefined;
    var pack_new_b: [32]u8 = undefined;
    var pack_cmd_b: [96]u8 = undefined;
    const pack_update: ?logic.PackUpdate = if (fresh and shared.pack_latest.len > 0) .{
        .current = take(&pack_cur_b, &shared.pack_current),
        .latest = take(&pack_new_b, &shared.pack_latest),
        .command = take(&pack_cmd_b, &shared.pack_command),
    } else null;
    var ln: usize = 0;
    for (shared.listings[0..shared.listing_n]) |l| {
        var known = l.known;
        var name: []const u8 = l.model.get();
        if (l.port != p) {
            known = .unknown;
            name = "";
            for (&shared.models) |*c| if (c.port == l.port) {
                known = c.known;
                name = c.model.get();
            };
        }
        @memcpy(list_models[ln][0..name.len], name);
        const m = list_models[ln][0..name.len];
        listings[ln] = .{ .port = l.port, .pid = l.pid, .model = if (m.len > 0) m else null, .models = known };
        ln += 1;
    }
    const sushi_listening = fresh and shared.sushi_listening;
    const port_taken = fresh and shared.port_taken;
    const mem_mb = if (fresh) shared.mem_mb else null;
    const gpu_mhz = if (fresh) shared.gpu_mhz else null;
    const gpu_pct = if (fresh) shared.gpu_pct else null;
    const lan_bound = fresh and shared.lan_listening;
    const last_prefill = if (fresh) shared.last_prefill else null;
    const hist_n = if (fresh) shared.hist_n else 0;
    @memcpy(hist[0..hist_n], shared.hist[0..hist_n]);
    const bar_version = take(&bar_b, &shared.bar_version);
    const server_version = if (fresh) take(&srv_b, &shared.server_version) else "";
    const server_command = shared.props_command.len > 0;
    const cmd_len = shared.props_command.len;
    @memcpy(cmd_b[0..cmd_len], shared.props_command.get());
    const uv_base: logic.UpdateView = .{
        .bar_version = bar_version,
        .server_version = server_version,
        .latest = take(&latest_b, &shared.latest),
        .available = shared.available,
        .command = if (g_sibling_brew and cmd_len == 0) "brew upgrade sushi" else cmd_b[0..cmd_len],
        .note = take(&note_b, &shared.note),
        .busy = shared.busy,
        .checked = shared.manual_result,
    };
    const relaunch = shared.relaunch;
    const note_problem = shared.note_problem;
    const model_full_len = if (fresh) shared.model_full_len else 0;
    unlock();

    if (view.link == .up or view.link == .metrics_off) logic.markAnswered(&started, p);
    syncProbes();
    switch (logic.rememberModel(g_last_model.get(), model, model_full_len)) {
        .store => {
            g_last_model.set(model);
            g_model_too_long = false;
            writeSaved(p);
        },
        .too_long => if (!g_model_too_long) {
            g_model_too_long = true;
            var nb: [128]u8 = undefined;
            setProblem(logic.modelTooLongNote(&nb, p));
        },
        .keep => if (model.len > 0) {
            g_model_too_long = false;
        },
    }
    watchTransition(p, fresh, view.link, now);
    const recovered = g_watch.takeRecovered();
    if (recovered != 0) {
        lock();
        if (logic.recoveryClearsNote(shared.note_kind, shared.note_port, recovered)) clearNote();
        unlock();
    }
    const own = ownServer();
    const ours_alive = own != null;
    if (!logic.stillStopping(stopping_until_ms, now, view.link, stopping_until_ms > now and sushiAlive(g_stop_pid))) stopping_until_ms = 0;
    const stopping = stopping_until_ms > now and g_stopping_port == p;
    const starting = ours_alive and view.link == .offline and !stopping;
    var serving_pid: i32 = if (own) |e| e.pid else 0;
    if (own == null) for (listings[0..ln]) |l| if (l.port == p) {
        serving_pid = l.pid;
    };
    if (g_stop_sent.port != p or serving_pid == 0) g_stop_sent = .{};
    g_stop_stuck = logic.stopStuck(g_stop_sent, p, now, serving_pid);
    if (g_stop_stuck and !g_stuck_noted) {
        var sb: [96]u8 = undefined;
        setServerProblem(logic.stuckNote(&sb, p), p);
    }
    g_stuck_noted = g_stop_stuck;
    var started_b: [started.len + 1]u16 = undefined;
    g_started_count = startedPorts(&started_b).len;
    applyQuitLook();
    const state: logic.Server = .{
        .link = view.link,
        .model = if (model.len > 0) model else null,
        .ours_alive = ours_alive,
        .sushi_found = g_sushi_found,
        .port = p,
        .sushi_listening = sushi_listening,
        .port_taken = port_taken,
        .connecting = !fresh,
        .stop_stuck = g_stop_stuck,
        .stopping = stopping,
    };
    const on = logic.enabled(state);
    const tg = logic.toggle(state, stopping);
    var uv = uv_base;
    uv.path = logic.updatePath(.{ .available = uv.available, .link = view.link, .server_command = server_command, .sibling_brew = g_sibling_brew });
    uv.model = model;
    uv.copied = copied_what == .upgrade and logic.copiedNow(copied_at_ms, now);
    button_command.set(uv.command);
    confirm_needed = logic.updateNeedsConfirm(uv);
    confirm_model.set(model);
    confirm_latest.set(uv.latest);

    // The header card, then everything else it owns the numbers of.
    var arena_buf: [8192]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&arena_buf);
    // Between Load and the model's row turning to loading, the model Load asked for is already the one loading.
    const asked = g_load_port.load(.acquire) == p and view.link == .up and model.len == 0 and loading_id.len == 0;
    const loading_name = if (loading_id.len > 0) loading_id else if (asked) g_last_model.get() else "";
    const header = logic.buildHeader(fba.allocator(), .{
        .link = view.link,
        .port = p,
        .loading = loading_name.len > 0 and model.len == 0,
        .updating = uv.busy == .updating,
        .starting = starting,
        .connecting = !fresh,
        .port_taken = port_taken,
        .sushi_found = g_sushi_found,
        .last_model = g_last_model.get(),
        .model = if (model.len > 0) model else if (loading_name.len > 0) loading_name else null,
        .pack_update = pack_update,
        .mem_mb = mem_mb,
        .bar_version = bar_version,
        .server_version = server_version,
        .update_available = uv.available,
        .latest = uv.latest,
        .decoding = view.decoding,
        .live = view.live,
        .last = view.last,
        .last_prefill = last_prefill,
        .gpu_mhz = gpu_mhz,
        .gpu_pct = gpu_pct,
        .lan = lan_bound,
    }) catch unreachable;
    var spark: [160]logic.SparkPoint = undefined;
    header_view.update(header, logic.sparkPoints(&spark, hist[0..hist_n], now, header_view.spark_window_ms));

    var tv = view;
    tv.update_waiting = uv.available;
    const title = logic.title(&last_title, tv);
    const button = objc.msg(objc.id, status_item, "button", .{});
    showIcon(button, logic.iconFor(view.link));
    showBadge(button, logic.badge(view.link, port_taken, g_watch.crashed(p) and !ours_alive));
    setButtonTitle(button, title);
    var status_buf: [200]u8 = undefined;
    const status = logic.statusLine(&status_buf, state);
    setText(button, "setToolTip:", status);

    button_row.setStart(.{ .label = tg.label, .symbol = tg.symbol, .tip = tg.tip, .enabled = tg.enabled, .closes_menu = tg.action != .start });
    var model_tip: [160]u8 = undefined;
    const op_port = if (g_load_port.load(.acquire) != 0) g_load_port.load(.acquire) else g_unload_port.load(.acquire);
    const mb = logic.modelButton(&model_tip, .{
        .other = if (op_port != 0 and op_port != p) .{ .port = op_port, .unload = g_unload_port.load(.acquire) != 0, .model = g_op_model.get() } else .{},
        .link = view.link,
        .connecting = !fresh,
        .starting = starting,
        .model = if (model.len > 0) model else null,
        .last_model = g_last_model.get(),
        .loading = g_load_port.load(.acquire) == p or (loading_id.len > 0 and model.len == 0),
        .unloading = g_unload_port.load(.acquire) == p,
    });
    model_action = mb.action;
    button_row.setLoad(.{ .label = mb.label, .symbol = mb.symbol, .tip = mb.tip, .enabled = mb.enabled, .closes_menu = false, .spinner = mb.spinner });
    showWhy(mb.why);
    var lab: [96]u8 = undefined;
    setText(port_mi, "setTitle:", std.fmt.bufPrint(&lab, "Port {d}", .{p}) catch "Port");
    objc.msg(void, lan_mi, "setState:", .{@as(isize, if (g_lan) 1 else 0)});
    setText(lan_mi, "setToolTip:", if (g_lan)
        "Started servers bind 0.0.0.0, so devices on the LAN can reach them. Click to bind 127.0.0.1 again. A server already running keeps its current bind."
    else
        "Start servers on every interface (0.0.0.0) so devices on the LAN can reach them. A running server keeps its current bind.");
    for (open_mis) |mi| objc.msg(void, mi, "setEnabled:", .{logic.openEnabled(view.link)});

    var entries: [max_listings + logic.max_saved + 2]logic.PortEntry = undefined;
    const port_entries = logic.portEntries(&entries, p, g_default_port, g_saved.items(), listings[0..ln]);
    syncPortMenu(port_entries);

    var fit_b: [logic.note_title_max + 4]u8 = undefined;
    const note_shown = if (uv.busy == .none and uv.note.len > 0) logic.fitNote(&fit_b, uv.note, logic.note_title_max) else "";
    showNote(note_shown, uv.note, note_problem);
    var tip_b: [160]u8 = undefined;
    const update_button = logic.updateButton(&tip_b, uv);
    button_action = update_button.action;
    button_row.setUpdate(buttonLook(update_button));
    button_row.dropStaleHover();
    var pcb: [128]u8 = undefined;
    const pack_tip = if (pack_update) |u| logic.copyLabel(&pcb, u.command) else "";
    const pack_copied = copied_what == .pack and logic.copiedNow(copied_at_ms, now);
    objc.msg(void, pack_mi, "setHidden:", .{pack_update == null});
    setText(pack_mi, "setTitle:", if (pack_copied) "Copied Pull Command" else "Copy Pull Command");
    setText(pack_mi, "setToolTip:", pack_tip);

    if (relaunch) relaunchSelf(p);

    if (test_polls > 0) {
        var ports_buf: [256]u8 = undefined;
        var pn: usize = 0;
        for (port_entries) |e| {
            const piece = std.fmt.bufPrint(ports_buf[pn..], "{s}{s};", .{ logic.portLabel(&lab, e), if (e.checked) "*" else "" }) catch break;
            pn += piece.len;
        }
        var removable_buf: [128]u8 = undefined;
        var rn: usize = 0;
        for (port_entries) |e| if (e.removable) {
            const piece = std.fmt.bufPrint(removable_buf[rn..], "{d},", .{e.port}) catch break;
            rn += piece.len;
        };
        std.debug.print("sushibar: poll {d} title=\"{s}\" status=\"{s}\" start={d} stop={d} server_pid={d} icon={s} ports=\"{s}\" version=\"{s}\" note=\"{s}\" problem={d} pill=\"{s}\" line1=\"{s}\" line2=\"{s}\" toggle=\"{s}\" field_invalid={d} pack=\"{s}\" packcopy=\"{s}\" button=\"{s}\" tip=\"{s}\" removable=\"{s}\" badge={d} models_link={d} last_model=\"{s}\" load=\"{s}\" load_on={d} why=\"{s}\" stats=\"{s}|{s}|{s}\" a11y=\"{s}\" lan={d}\n", .{
            tick_count,                 title,                 status,                 @intFromBool(on.start),
            @intFromBool(on.stop),      if (own) |e| e.pid else 0,
            @tagName(shown_icon.?),     ports_buf[0..pn],      header.version,         note_shown,
            @intFromBool(note_problem and note_shown.len > 0), header.pill.label, header.line1, header.line2,
            tg.label,                   @intFromBool(port_field.isInvalid()), header.pack_line, pack_tip,
            update_button.label,        update_button.tip,     removable_buf[0..rn],   @intFromBool(shown_badge),
            @intFromBool(header.line1_models), g_last_model.get(), mb.label, @intFromBool(mb.enabled), mb.why,
            header.stats[0].label,      header.stats[1].label, header.stats[2].label,  header.a11y,
            @intFromBool(g_lan),
        });
        if (tick_count >= test_polls) quit();
    }
}

/// The sibling's version changed: run the new sushibar in this process, on the same port.
fn relaunchSelf(p: u16) void {
    lock();
    shared.relaunch = false;
    unlock();
    var port_z: [8:0]u8 = undefined;
    const pz = std.fmt.bufPrintSentinel(&port_z, "{d}", .{p}, 0) catch return;
    var sb: [path_cap]u8 = undefined;
    // The path sushibar was started by, so a Homebrew link lands on the new keg; brew's opt link if that is gone.
    const exe = if (reachable(g_self_exe)) g_self_exe else siblingNow(&sb, bar_exe) orelse g_self_exe;
    proc.execReplace(&.{ exe.ptr, "--port", pz.ptr });
    setProblem("Updated, but sushibar could not restart itself: quit and run it again");
}

fn openUrl(url: []const u8) void {
    var z: [256:0]u8 = undefined;
    const n = @min(url.len, z.len - 1);
    @memcpy(z[0..n], url[0..n]);
    z[n] = 0;
    const ns_url = objc.msg(objc.id, objc.class("NSURL"), "URLWithString:", .{objc.nsString(&z)});
    _ = objc.msg(bool, objc.msg(objc.id, objc.class("NSWorkspace"), "sharedWorkspace", .{}), "openURL:", .{ns_url});
}

fn openWebPage(page: logic.Page) void {
    var origin: [origin_cap]u8 = undefined;
    var url_buf: [url_cap]u8 = undefined;
    openUrl(logic.pageUrl(&url_buf, originFor(&origin, port()), page));
}

fn openPage(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    openWebPage(@enumFromInt(@as(u2, @intCast(objc.msg(isize, sender, "tag", .{})))));
}

fn openModelsAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    openWebPage(.models);
}

fn openReleaseNotes(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    lock();
    var b: [160]u8 = undefined;
    const url = take(&b, &shared.props_url);
    unlock();
    openUrl(logic.releaseNotesUrl(url));
}

/// Start or Stop, whichever the item reads.
fn toggleAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    const ours_alive = ownServer() != null;
    lock();
    const state: logic.Server = .{
        .link = shared.view.link,
        .ours_alive = ours_alive,
        .sushi_found = g_sushi_found,
        .port = port(),
        .sushi_listening = shared.sushi_listening,
        .port_taken = shared.port_taken,
        .stop_stuck = g_stop_stuck,
    };
    unlock();
    switch (logic.toggle(state, stopping_until_ms > io_util.nowMs(g_io) and g_stopping_port == port()).action) {
        .start => startServer(),
        .stop => stopServer(),
        .force_stop => forceStop(),
        .none => {},
    }
    refresh();
}
fn loadModelAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    switch (model_action) {
        .load => startModelOp(false),
        .unload => startModelOp(true),
        .none => {},
    }
    refresh();
}
fn quitAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    quit();
}
fn quitStopAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    quitStop();
}

var g_stop_stuck = false;
var g_started_count: usize = 0;
var g_option = false;
var g_flags_timer: objc.id = null;
const option_key_mask: usize = 1 << 19;

/// Quit, or Quit & Stop while Option is held and sushibar runs a server it started.
fn applyQuitLook() void {
    var tb: [128]u8 = undefined;
    const q = logic.quitButton(&tb, g_option, g_started_count);
    // "Quit & Stop" fills the button on its own; a symbol beside it would touch the edges.
    button_row.setQuit(.{ .label = q.label, .symbol = if (q.stops) "" else "power", .tip = q.tip, .selector = if (q.stops) "quitStop:" else null });
}

/// Watches the Option key while the menu is open, so Quit turns into Quit & Stop as it is pressed.
fn flagsTick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    const opt = objc.msg(usize, objc.class("NSEvent"), "modifierFlags", .{}) & option_key_mask != 0;
    if (opt == g_option) return;
    g_option = opt;
    applyQuitLook();
}
fn pickPortAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    setPort(@intCast(objc.msg(isize, sender, "tag", .{})));
}

/// Turns the LAN bind for started servers on or off. A running server keeps its current bind; the note says the
/// switch applies to the next Start.
fn toggleLan() void {
    g_lan = !g_lan;
    writeSaved(port());
    var nb: [128]u8 = undefined;
    setNote(logic.lanNote(&nb, g_lan));
    refresh();
}

fn lanToggleAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    toggleLan();
}
fn applyPortAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    applyPortText();
}
fn textChangedAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    port_field.clearInvalid();
}
fn buttonLook(b: logic.UpdateButton) button_row.Look {
    return .{ .label = b.label, .symbol = b.symbol, .tip = b.tip, .accent = b.accent, .enabled = b.enabled, .closes_menu = b.closes_menu, .spinner = b.spinner };
}
fn updateButtonAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    switch (button_action) {
        .check => runAction(.{ .kind = .check, .at = 0 }),
        .update => if (!confirm_needed or confirmUpdate()) runAction(.{ .kind = .update, .at = 0 }),
        .copy_command => copyUpgrade(),
        .none => {},
    }
    refresh();
}

fn copyUpgrade() void {
    var z: [81]u8 = undefined;
    const c = button_command.get();
    @memcpy(z[0..c.len], c);
    z[c.len] = 0;
    objc.copyText(@ptrCast(&z));
    copied_what = .upgrade;
    copied_at_ms = io_util.nowMs(g_io);
}

fn copyPackAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    var cmd: [97]u8 = undefined;
    lock();
    const n = shared.pack_command.len;
    @memcpy(cmd[0..n], shared.pack_command.get());
    unlock();
    cmd[n] = 0;
    if (n == 0) return;
    objc.copyText(@ptrCast(&cmd));
    copied_what = .pack;
    copied_at_ms = io_util.nowMs(g_io);
}

fn pickRowAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    if (port_row.portOf(sender)) |p| setPort(p);
}
fn removeRowAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    if (port_row.portOf(sender)) |p| removePort(p);
}

/// The top menu re-reads the default port and polls at once; the Port submenu asks its listeners for their models.
fn menuWillOpen(_: objc.id, _: objc.SEL, opening: objc.id) callconv(.c) void {
    if (opening != top_menu) return refreshModels(null);
    g_menu_open.store(true, .release);
    g_wake.set(g_io);
    g_default_port = readDefaultPort();
    const now = io_util.nowMs(g_io);
    lock();
    shared.manual_result = logic.shownResult(shared.manual_result, shared.manual_at_ms, now);
    if (!logic.noteKept(shared.note_problem, shared.note_at_ms, now)) shared.note.set("");
    unlock();
    g_option = false;
    flagsTick(null, null, null);
    if (g_flags_timer == null) g_flags_timer = objc.addTimer(0.1, menu_target, "flagsTick:");
    refresh();
}

fn menuDidClose(_: objc.id, _: objc.SEL, closing: objc.id) callconv(.c) void {
    if (closing != top_menu) return;
    g_menu_open.store(false, .release);
    if (g_flags_timer != null) objc.msg(void, g_flags_timer, "invalidate", .{});
    g_flags_timer = null;
}

fn refreshNow(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    refresh();
}

/// The watched port's log: what its server's argv said, or the default file while that is not known yet.
fn watchedLog(buf: *[1024]u8) logic.LogSource {
    const p = port();
    lock();
    defer unlock();
    if (shared.port != p) return logic.logSource(buf, g_home, p, null);
    const path = shared.log_path.get();
    @memcpy(buf[0..path.len], path);
    return .{ .kind = shared.log_kind, .path = buf[0..path.len] };
}

fn showLogAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    var buf: [1024]u8 = undefined;
    log_view.open(port(), watchedLog(&buf));
}

/// Follows the log while its window is open, and the watched port when it changes.
fn logTick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    if (!log_view.isVisible()) return;
    var buf: [1024]u8 = undefined;
    log_view.sync(port(), watchedLog(&buf));
}

const icon_w = 22;
const icon_h = 18;

fn loadIcons() void {
    icons[@intFromEnum(logic.Icon.outline)] = objc.loadSvg(@embedFile("sushibar/icon-offline.svg"), true, icon_w, icon_h);
    icons[@intFromEnum(logic.Icon.color)] = objc.loadSvg(@embedFile("sushibar/icon-online.svg"), false, icon_w, icon_h);
}

const find_show_interface: isize = 1;

fn addMainItem(menu: objc.id, title: [*:0]const u8, action: [*:0]const u8, key: [*:0]const u8) objc.id {
    return objc.msg(objc.id, menu, "addItemWithTitle:action:keyEquivalent:", .{ objc.nsString(title), objc.sel(action), objc.nsString(key) });
}

fn addSubmenu(bar: objc.id, title: [*:0]const u8) objc.id {
    const item = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, bar, "addItem:", .{item});
    const menu = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSMenu"), "alloc", .{}), "initWithTitle:", .{objc.nsString(title)});
    objc.msg(void, item, "setSubmenu:", .{menu});
    return menu;
}

/// An accessory app shows no menu bar, but its main menu still carries the key equivalents: without it ⌘C, ⌘A, ⌘F
/// and ⌘W do nothing in the log window and ⌘V does nothing in the Port field.
fn setMainMenu(app: objc.id) void {
    const bar = objc.new(objc.class("NSMenu"));
    const sushibar_menu = addSubmenu(bar, "sushibar");
    const quit_item = addMainItem(sushibar_menu, "Quit sushibar", "quit:", "q");
    objc.msg(void, quit_item, "setTarget:", .{menu_target});
    const edit = addSubmenu(bar, "Edit");
    _ = addMainItem(edit, "Cut", "cut:", "x");
    _ = addMainItem(edit, "Copy", "copy:", "c");
    _ = addMainItem(edit, "Paste", "paste:", "v");
    _ = addMainItem(edit, "Select All", "selectAll:", "a");
    const find = addMainItem(edit, "Find…", "performTextFinderAction:", "f");
    objc.msg(void, find, "setTag:", .{find_show_interface});
    const window = addSubmenu(bar, "Window");
    _ = addMainItem(window, "Close Window", "performClose:", "w");
    _ = addMainItem(window, "Minimize", "performMiniaturize:", "m");
    objc.msg(void, app, "setMainMenu:", .{bar});
}

fn runMenuBar() noreturn {
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "setActivationPolicy:", .{@as(isize, 1)});
    const bar = objc.msg(objc.id, objc.class("NSStatusBar"), "systemStatusBar", .{});
    status_item = objc.msg(objc.id, bar, "statusItemWithLength:", .{@as(f64, -1)});
    const button = objc.msg(objc.id, status_item, "button", .{});
    loadIcons();
    objc.msg(void, button, "setImagePosition:", .{@as(isize, 2)});
    showIcon(button, .outline);
    makeBadge(button);

    const cls = objc.defineClass("SushiBarTarget", &.{
        .{ .selector = "tick:", .imp = @ptrCast(&tick), .types = "v@:@" },
        .{ .selector = "open:", .imp = @ptrCast(&openPage), .types = "v@:@" },
        .{ .selector = "toggle:", .imp = @ptrCast(&toggleAction), .types = "v@:@" },
        .{ .selector = "loadModel:", .imp = @ptrCast(&loadModelAction), .types = "v@:@" },
        .{ .selector = "quit:", .imp = @ptrCast(&quitAction), .types = "v@:@" },
        .{ .selector = "quitStop:", .imp = @ptrCast(&quitStopAction), .types = "v@:@" },
        .{ .selector = "flagsTick:", .imp = @ptrCast(&flagsTick), .types = "v@:@" },
        .{ .selector = "pickPort:", .imp = @ptrCast(&pickPortAction), .types = "v@:@" },
        .{ .selector = "lanToggle:", .imp = @ptrCast(&lanToggleAction), .types = "v@:@" },
        .{ .selector = "pickRow:", .imp = @ptrCast(&pickRowAction), .types = "v@:@" },
        .{ .selector = "removeRow:", .imp = @ptrCast(&removeRowAction), .types = "v@:@" },
        .{ .selector = "applyPort:", .imp = @ptrCast(&applyPortAction), .types = "v@:@" },
        .{ .selector = "controlTextDidChange:", .imp = @ptrCast(&textChangedAction), .types = "v@:@" },
        .{ .selector = "updateButton:", .imp = @ptrCast(&updateButtonAction), .types = "v@:@" },
        .{ .selector = "showLog:", .imp = @ptrCast(&showLogAction), .types = "v@:@" },
        .{ .selector = "logTick:", .imp = @ptrCast(&logTick), .types = "v@:@" },
        .{ .selector = "openModels:", .imp = @ptrCast(&openModelsAction), .types = "v@:@" },
        .{ .selector = "openRelease:", .imp = @ptrCast(&openReleaseNotes), .types = "v@:@" },
        .{ .selector = "copyPack:", .imp = @ptrCast(&copyPackAction), .types = "v@:@" },
        .{ .selector = "menuWillOpen:", .imp = @ptrCast(&menuWillOpen), .types = "v@:@" },
        .{ .selector = "menuDidClose:", .imp = @ptrCast(&menuDidClose), .types = "v@:@" },
        .{ .selector = "refreshNow:", .imp = @ptrCast(&refreshNow), .types = "v@:@" },
    });
    menu_target = objc.new(cls);
    setMainMenu(app);
    const menu = objc.new(objc.class("NSMenu"));
    top_menu = menu;
    objc.msg(void, menu, "setDelegate:", .{menu_target});
    objc.msg(void, menu, "setAutoenablesItems:", .{false});

    header_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, header_mi, "setView:", .{header_view.create(
        objc.loadSvg(@embedFile("sushibar/icon-online.svg"), false, 34, 28),
        objc.loadSvg(@embedFile("sushibar/icon-offline.svg"), true, 34, 28),
        menu_target,
        menu,
    )});
    objc.msg(void, menu, "addItem:", .{header_mi});
    addSeparator(menu);

    button_row.init(menu_target, menu);
    const server_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, server_mi, "setView:", .{button_row.createServerRow()});
    objc.msg(void, menu, "addItem:", .{server_mi});
    why_mi = addItem(menu, "", null, 0, "", null);
    objc.msg(void, why_mi, "setEnabled:", .{false});
    objc.msg(void, why_mi, "setHidden:", .{true});
    port_mi = addItem(menu, "Port", null, 0, "", "network");
    objc.msg(void, port_mi, "setEnabled:", .{true});
    port_menu = objc.new(objc.class("NSMenu"));
    objc.msg(void, port_menu, "setAutoenablesItems:", .{false});
    objc.msg(void, port_mi, "setSubmenu:", .{port_menu});
    objc.msg(void, port_menu, "setDelegate:", .{menu_target});
    addSeparator(port_menu);
    lan_mi = addItem(port_menu, "Listen on LAN", "lanToggle:", 0, "", "globe");
    addSeparator(port_menu);
    port_field_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, port_field_mi, "setView:", .{port_field.create(menu_target)});
    objc.msg(void, port_menu, "addItem:", .{port_field_mi});
    addSeparator(menu);

    open_mis[0] = addItem(menu, "Dashboard", "open:", @intFromEnum(logic.Page.dashboard), "1", "gauge.with.dots.needle.67percent");
    open_mis[1] = addItem(menu, "Chat", "open:", @intFromEnum(logic.Page.chat), "2", "bubble.left.and.bubble.right");
    open_mis[2] = addItem(menu, "Models", "open:", @intFromEnum(logic.Page.models), "3", "shippingbox");
    open_mis[3] = addItem(menu, "Settings", "open:", @intFromEnum(logic.Page.settings), "4", "gearshape");
    for (open_mis) |mi| objc.msg(void, mi, "setEnabled:", .{false});
    addSeparator(menu);

    pack_mi = addItem(menu, "Copy Pull Command", "copyPack:", 0, "", "doc.on.doc");
    objc.msg(void, pack_mi, "setHidden:", .{true});
    note_mi = addItem(menu, "", null, 0, "", null);
    objc.msg(void, note_mi, "setHidden:", .{true});
    addSeparator(menu);

    const buttons_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, buttons_mi, "setView:", .{button_row.createFooter()});
    objc.msg(void, menu, "addItem:", .{buttons_mi});
    // The Quit button is the visible row; this item only keeps ⌘Q.
    const quit_mi = addItem(menu, "Quit sushibar", "quit:", 0, "q", null);
    objc.msg(void, quit_mi, "setHidden:", .{true});
    objc.msg(void, status_item, "setMenu:", .{menu});
    _ = objc.addTimer(@as(f64, tick_ms) / 1000.0, menu_target, "tick:");
    _ = objc.addTimer(0.5, menu_target, "logTick:");

    _ = std.Thread.spawn(.{}, pollLoop, .{}) catch fail("cannot start the poll thread", .{});
    objc.msg(void, app, "run", .{});
    std.process.exit(0);
}

/// `SUSHIBAR_RENDER_HEADER=<dir>`: draws the header card for fixed sample states, light and dark, at 2x into PNGs
/// in `dir`, and exits. For checking the design without a screen.
fn renderHeaders(gpa: std.mem.Allocator, dir: []const u8) noreturn {
    _ = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    _ = header_view.create(
        objc.loadSvg(@embedFile("sushibar/icon-online.svg"), false, 34, 28),
        objc.loadSvg(@embedFile("sushibar/icon-offline.svg"), true, 34, 28),
        null,
        null,
    );
    std.Io.Dir.cwd().createDirPath(g_io, dir) catch {};
    const t0: i64 = 1_700_000_000_000;
    // A busy session with a gap; one history for the card with a short outage in it, one without.
    var hist: [150]logic.Hist = undefined;
    var hist_gap: [150]logic.Hist = undefined;
    var n: usize = 0;
    var n_gap: usize = 0;
    var gen: u64 = 0;
    for (0..150) |i| {
        const f: f64 = @floatFromInt(i);
        const rate: f64 = if (i >= 25) 95 + 20 * @sin(f * 0.7) + 10 * @sin(f * 0.23) else 0;
        gen += @intFromFloat(rate * 2);
        const sample: logic.Hist = .{ .t_ms = t0 - @as(i64, @intCast(149 - i)) * 2000, .gen = gen };
        hist[n] = sample;
        n += 1;
        if (i >= 100 and i < 103) continue;
        hist_gap[n_gap] = sample;
        n_gap += 1;
    }
    const States = struct { name: []const u8, in: logic.HeaderIn, sparkline: bool = false, outage: bool = false };
    const deep = "Example-Flash-1017-Sushi-STQ-1.75bpw";
    const states = [_]States{
        .{ .name = "stopped", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "stopped-last-model", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .last_model = deep } },
        .{ .name = "port-taken", .in = .{ .link = .offline, .port = 8080, .bar_version = "1.5.0-dev1", .port_taken = true } },
        .{ .name = "no-sushi", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .sushi_found = false } },
        .{ .name = "starting", .in = .{ .link = .offline, .port = 12345, .bar_version = "1.5.0-dev1", .starting = true, .last_model = deep } },
        .{ .name = "connecting", .in = .{ .link = .offline, .port = 18803, .bar_version = "1.5.0-dev1", .connecting = true } },
        .{ .name = "silent", .in = .{ .link = .silent, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "ready-to-load", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .last_model = deep, .gpu_mhz = 0 } },
        .{ .name = "loading", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .loading = true, .model = deep, .mem_mb = 12000 } },
        .{ .name = "no-model", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .gpu_mhz = 0 } },
        .{ .name = "generating", .sparkline = true, .outage = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0-dev1",
            .decoding = true, .live = 105.27, .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
        } },
        .{ .name = "key-needed", .in = .{ .link = .unauthorized, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = "Example-Flash-Next-Sushi-STQ-2.75bpw", .mem_mb = 31200, .bar_version = "1.5.0",
            .update_available = true, .latest = "1.5.1", .last = 142.6, .last_prefill = 3120, .gpu_mhz = 1380, .gpu_pct = 8,
        } },
        .{ .name = "pack-update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0",
            .pack_update = .{ .current = "1.0", .latest = "2.0", .command = "sushi pull example-pack" },
            .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
        } },
    };
    for (states) |st| {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const h = logic.buildHeader(arena.allocator(), st.in) catch continue;
        var spark: [160]logic.SparkPoint = undefined;
        const pts = if (st.sparkline) logic.sparkPoints(&spark, if (st.outage) hist_gap[0..n_gap] else hist[0..n], t0, header_view.spark_window_ms) else spark[0..0];
        header_view.update(h, pts);
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/header-{s}-{s}.png", .{ dir, st.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!header_view.renderPng(pz.ptr, dark)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    const field = port_field.create(null);
    for ([_][]const u8{ "empty", "invalid" }) |state| {
        if (std.mem.eql(u8, state, "invalid")) {
            port_field.setText("99999");
            port_field.showInvalid();
        }
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/portfield-{s}-{s}.png", .{ dir, state, if (dark) "dark" else "light" }, 0) catch continue;
            if (!port_field.renderPng(field, pz.ptr, dark)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    renderButtonsAndLog(dir);
    renderBadge(dir);
    std.process.exit(0);
}

/// The menu bar item in its states, on a plain button of the status button's shape over a menu-bar-like fill.
fn renderBadge(dir: []const u8) void {
    loadIcons();
    const Case = struct { name: []const u8, icon: logic.Icon, title: []const u8 = "", badge: bool = false };
    const cases = [_]Case{
        .{ .name = "live", .icon = .color, .title = "105 tok/s" },
        .{ .name = "last", .icon = .color, .title = "88.0 tok/s" },
        .{ .name = "generating", .icon = .color, .title = "generating…" },
        .{ .name = "reading", .icon = .color, .title = "prompt…" },
        .{ .name = "badge", .icon = .color, .badge = true },
        .{ .name = "offline", .icon = .outline },
    };
    for (cases) |c| for ([_]bool{ false, true }) |dark| {
        const box = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSBox"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = 168, .h = 24 }});
        objc.msg(void, box, "setBoxType:", .{@as(isize, 4)});
        objc.msg(void, box, "setBorderWidth:", .{@as(f64, 0)});
        objc.msg(void, box, "setContentViewMargins:", .{objc.NSSize{ .width = 0, .height = 0 }});
        objc.msg(void, box, "setFillColor:", .{objc.msg(objc.id, objc.class("NSColor"), "colorWithWhite:alpha:", .{ @as(f64, if (dark) 0.16 else 0.92), @as(f64, 1) })});
        const button = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSButton"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 4, .y = 0, .w = 160, .h = 24 }});
        objc.msg(void, button, "setBordered:", .{false});
        objc.msg(void, button, "setImagePosition:", .{@as(isize, 2)});
        objc.msg(void, button, "setImage:", .{icons[@intFromEnum(c.icon)]});
        // The status bar draws a template image in the bar's text colour; a plain button would draw it dimmed.
        objc.msg(void, button, "setContentTintColor:", .{objc.color("labelColor")});
        shown_title_len = std.math.maxInt(usize);
        setButtonTitle(button, c.title);
        makeBadge(button);
        shown_badge = !c.badge;
        showBadge(button, c.badge);
        objc.msg(void, box, "addSubview:", .{button});
        const name = if (dark) "NSAppearanceNameDarkAqua" else "NSAppearanceNameAqua";
        objc.msg(void, box, "setAppearance:", .{objc.msg(objc.id, objc.class("NSAppearance"), "appearanceNamed:", .{objc.nsString(name)})});
        const bounds = objc.msg(NSRect, box, "bounds", .{});
        const rep = objc.msg(objc.id, box, "bitmapImageRepForCachingDisplayInRect:", .{bounds});
        if (rep == null) return;
        objc.msg(void, box, "cacheDisplayInRect:toBitmapImageRep:", .{ bounds, rep });
        const data = objc.msg(objc.id, rep, "representationUsingType:properties:", .{ @as(isize, 4), @as(objc.id, null) });
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/menubar-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch return;
        _ = objc.msg(bool, data, "writeToFile:atomically:", .{ objc.nsString(pz.ptr), true });
    };
}

fn renderButtonsAndLog(dir: []const u8) void {
    const row = button_row.createFooter();
    const Case = struct { name: []const u8, uv: logic.UpdateView = .{}, hover: ?usize = null, pressed: ?usize = null };
    const cases = [_]Case{
        .{ .name = "normal" },
        .{ .name = "hover-log", .hover = 0 },
        .{ .name = "pressed-quit", .hover = 2, .pressed = 2 },
        .{ .name = "update", .uv = .{ .available = true, .latest = "1.5.1", .path = .server } },
        .{ .name = "update-hover", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .hover = 1 },
        .{ .name = "update-pressed", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .pressed = 1 },
        .{ .name = "brew", .uv = .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi" } },
        .{ .name = "brew-copied", .uv = .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi", .copied = true } },
        .{ .name = "checking", .uv = .{ .busy = .checking } },
        .{ .name = "up-to-date", .uv = .{ .checked = .up_to_date } },
        .{ .name = "failed", .uv = .{ .checked = .failed } },
        .{ .name = "failed-hover", .uv = .{ .checked = .failed }, .hover = 1 },
    };
    var tip_b: [160]u8 = undefined;
    for (cases) |c| {
        var tip: [160]u8 = undefined;
        const b = logic.updateButton(&tip, c.uv);
        button_row.setUpdate(buttonLook(b));
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/buttons-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!button_row.renderPng(row, pz.ptr, dark, c.hover, c.pressed, "")) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
    button_row.setUpdate(buttonLook(logic.updateButton(&tip_b, .{})));
    g_option = true;
    g_started_count = 1;
    applyQuitLook();
    for ([_]bool{ false, true }) |dark| {
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/buttons-quit-stop-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch continue;
        if (!button_row.renderPng(row, pz.ptr, dark, 2, null, "")) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
    renderServerRow(dir);
    renderPortRows(dir);
    const sample =
        \\2026-11-04 09:12:01 [info] Logging to /Users/me/.sushi/logs/sushi-11250.log (rotates at 100 MB)
        \\2026-11-04 09:12:02 [info] sushi 1.5.0 serving on http://127.0.0.1:11250
        \\2026-11-04 09:12:04 [info] [kv-cache] affine8 KV, context 262144
        \\2026-11-04 09:12:09 [info] Loaded Example-Flash-Next-Sushi-STQ-2.75bpw in 4.8 s
        \\2026-11-04 09:13:20 [info] <- 412+0 tokens (chat, stream)
        \\2026-11-04 09:13:20 [warn] [admission] prefill chunk lowered to 2048 for the free memory
        \\2026-11-04 09:13:31 [info] [spec-stats] accepted 2.31 of 3 drafted
        \\2026-11-04 09:13:34 [info] -> 288 tokens, 142.6 tok/s decode
        \\2026-11-04 09:14:02 [error] jinja error: unknown filter 'tojson_pretty'
        \\2026-11-04 09:14:40 [info] [hot-cache] kept 3 entries, 1.2 GB
        \\
    ;
    const std_path = "/Users/me/.sushi/logs/sushi-11250.log";
    for ([_]bool{ false, true }) |dark| {
        var path: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&path, "{s}/log-window-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch continue;
        if (!log_view.renderPng(pz.ptr, dark, .{ .port = 11250, .src = .{ .kind = .standard, .path = std_path }, .sample = sample })) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
    var nb: [600]u8 = undefined;
    const notice = logic.logNotice(&nb, 11250, .{ .kind = .standard, .path = std_path });
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    if (std.fmt.bufPrintSentinel(&path, "{s}/log-window-missing-light.png", .{dir}, 0)) |pz| {
        if (!log_view.renderPng(pz.ptr, false, .{ .port = 11250, .src = .{ .kind = .standard, .path = std_path }, .sample = notice, .has_file = false })) std.debug.print("sushibar: could not render {s}\n", .{pz});
    } else |_| {}
    renderFollowedLog(dir);
}

/// Start/Stop beside Load/Unload for each state the pair can be in, with the line under them.
fn renderServerRow(dir: []const u8) void {
    const row = button_row.createServerRow();
    const m = "Example-Flash-1017-Sushi-STQ-1.75bpw";
    const Case = struct { name: []const u8, s: logic.Server, m: logic.ModelIn, stopping: bool = false, hover: ?usize = null, pressed: ?usize = null };
    const cases = [_]Case{
        .{ .name = "stopped", .s = .{ .link = .offline }, .m = .{ .link = .offline, .last_model = m } },
        .{ .name = "stopped-no-last", .s = .{ .link = .offline }, .m = .{ .link = .offline } },
        .{ .name = "starting", .s = .{ .link = .offline, .ours_alive = true }, .m = .{ .link = .offline, .starting = true, .last_model = m } },
        .{ .name = "ready-to-load", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m } },
        .{ .name = "ready-hover-load", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m }, .hover = 4 },
        .{ .name = "no-last-model", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up } },
        .{ .name = "loading", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .last_model = m, .loading = true } },
        .{ .name = "loaded", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m } },
        .{ .name = "loaded-hover-unload", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .hover = 4 },
        .{ .name = "loaded-pressed-stop", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .hover = 3, .pressed = 3 },
        .{ .name = "unloading", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m, .unloading = true } },
        .{ .name = "stopping", .s = .{ .link = .up, .ours_alive = true }, .m = .{ .link = .up, .model = m, .last_model = m }, .stopping = true },
    };
    for (cases) |c| {
        const tg = logic.toggle(c.s, c.stopping);
        var tip: [160]u8 = undefined;
        const mb = logic.modelButton(&tip, c.m);
        button_row.setStart(.{ .label = tg.label, .symbol = tg.symbol, .tip = tg.tip, .enabled = tg.enabled, .closes_menu = false });
        button_row.setLoad(.{ .label = mb.label, .symbol = mb.symbol, .tip = mb.tip, .enabled = mb.enabled, .closes_menu = false, .spinner = mb.spinner });
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/serverrow-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!button_row.renderPng(row, pz.ptr, dark, c.hover, c.pressed, mb.why)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
}

/// The Port submenu's rows: the default (not removable), then added ports.
fn renderPortRows(dir: []const u8) void {
    const rows = [_]port_row.Row{
        .{ .port = 12345, .label = "12345 — Example-Flash-Next-Sushi-STQ-2.75bpw", .checked = true, .removable = false },
        .{ .port = 18803, .label = "18803 — not running", .checked = false, .removable = true },
        .{ .port = 18802, .label = "18802 — Example-Flash-1017", .checked = false, .removable = true },
        .{ .port = 7000, .label = "7000 — no model", .checked = false, .removable = true },
    };
    const Case = struct { name: []const u8, hover: ?usize = null, x_hover: bool = false };
    for ([_]Case{ .{ .name = "normal" }, .{ .name = "hover-default", .hover = 0 }, .{ .name = "hover-added", .hover = 1 }, .{ .name = "hover-remove", .hover = 2, .x_hover = true } }) |c| {
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/portrows-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!port_row.renderPng(&rows, pz.ptr, dark, c.hover, c.x_hover)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
}

/// The window reading a real file: its last lines on open, then what the file grew by while the reader was scrolled
/// up, which the New lines pill announces.
fn renderFollowedLog(dir: []const u8) void {
    var path: [std.fs.max_path_bytes:0]u8 = undefined;
    const file = std.fmt.bufPrint(&path, "{s}/followed.log", .{dir}) catch return;
    var text: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer text.deinit();
    const w = &text.writer;
    for (1..3001) |i| w.print("line {d} of the first read\n", .{i}) catch return;
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = file, .data = text.written() }) catch return;
    const src: logic.LogSource = .{ .kind = .given, .path = file };
    log_view.sync(11250, src);
    log_view.scrollToTop();
    for (3001..3006) |i| w.print("line {d} appended later\n", .{i}) catch return;
    w.writeAll("a line still being writ") catch return;
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = file, .data = text.written() }) catch return;
    log_view.sync(11250, src);
    for ([_]bool{ false, true }) |dark| {
        var png: [std.fs.max_path_bytes:0]u8 = undefined;
        const pz = std.fmt.bufPrintSentinel(&png, "{s}/log-window-new-lines-{s}.png", .{ dir, if (dark) "dark" else "light" }, 0) catch return;
        if (!log_view.renderPng(pz.ptr, dark, null)) std.debug.print("sushibar: could not render {s}\n", .{pz});
    }
}

// ── Launch ──

/// The path sushibar was started by, made absolute but not resolved: its symlinks are followed at each use.
fn selfPath(buf: []u8) ?[:0]const u8 {
    var tmp: [std.fs.max_path_bytes]u8 = undefined;
    var n: u32 = tmp.len;
    if (std.c._NSGetExecutablePath(&tmp, &n) != 0) return null;
    const raw = std.mem.span(@as([*:0]const u8, @ptrCast(&tmp)));
    if (raw.len > 0 and raw[0] == '/') {
        if (raw.len >= buf.len) return null;
        @memcpy(buf[0..raw.len], raw);
        buf[raw.len] = 0;
        return buf[0..raw.len :0];
    }
    const real = proc.realPath(buf, raw) orelse return null;
    buf[real.len] = 0;
    return buf[0..real.len :0];
}

/// The terminal's copy: start the detached one, wait until it has taken the pid file, and leave.
fn detach(gpa: std.mem.Allocator, args: []const []const u8, self_exe: [:0]const u8) noreturn {
    var argv: std.ArrayList([*:0]const u8) = .empty;
    argv.append(gpa, self_exe.ptr) catch fail("out of memory", .{});
    for (args) |a| argv.append(gpa, gpa.dupeSentinel(u8, a, 0) catch fail("out of memory", .{}).ptr) catch fail("out of memory", .{});
    const child = proc.spawnDetached(argv.items, test_polls > 0) catch fail("cannot start the detached copy", .{});
    var waited: u32 = 0;
    while (waited < 100) : (waited += 1) {
        if (proc.verifiedPid(g_io, g_bar_pid_path, bar_exe, false)) |e| {
            if (e.pid == child) {
                out(g_io, "sushibar started (pid {d}), watching port {d}\n", .{ child, port() });
                std.process.exit(0);
            }
            out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
            std.process.exit(0);
        }
        if (std.c.waitpid(child, null, std.c.W.NOHANG) == child) {
            // A copy that lost the race for the single-instance lock exits at once; the winner's pid file explains it.
            if (proc.verifiedPid(g_io, g_bar_pid_path, bar_exe, false)) |e| {
                out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
                std.process.exit(0);
            }
            fail("the detached copy exited at once (no window server?)", .{});
        }
        std.Io.sleep(g_io, .fromMilliseconds(50), .real) catch {};
    }
    fail("the detached copy did not register within 5 s", .{});
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    g_io = init.io;

    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    var list: std.ArrayList([]const u8) = .empty;
    while (it.next()) |a| try list.append(gpa, try gpa.dupe(u8, a));
    const args = list.items[1..];

    var explicit: ?u16 = null;
    switch (logic.parseArgs(args)) {
        .help => {
            out(g_io, usage, .{logic.default_port});
            return;
        },
        .refused => |r| {
            const why = switch (r.reason) {
                .unknown_argument => "unknown argument",
                .missing_value => "needs a value",
                .bad_port => "needs a port from 1 to 65535",
            };
            std.debug.print("sushibar: {s}: {s}\n", .{ r.arg, why });
            std.process.exit(2);
        },
        .run => |p| explicit = p,
    }
    if (env("SUSHIBAR_RENDER_HEADER")) |dir| renderHeaders(gpa, dir);
    if (!logic.guiSession(env("SSH_CONNECTION"), env("SSH_TTY")))
        fail("no GUI session (this is an ssh login): run sushibar from the Mac's desktop session", .{});

    g_home = env("HOME") orelse "/tmp";
    var p1: [std.fs.max_path_bytes]u8 = undefined;
    var p3: [std.fs.max_path_bytes]u8 = undefined;
    g_bar_pid_path = try gpa.dupe(u8, proc.pidFilePath(&p1, g_home, bar_pid_name) orelse fail("HOME is too long", .{}));
    g_saved_path = try gpa.dupe(u8, proc.pidFilePath(&p3, g_home, saved_name) orelse fail("HOME is too long", .{}));
    readTestSeams();

    g_default_port = readDefaultPort();
    const saved_text = std.Io.Dir.cwd().readFileAlloc(g_io, g_saved_path, gpa, .limited(logic.saved_max_bytes)) catch "";
    const saved = logic.parseSaved(saved_text);
    g_saved = saved.ports;
    g_lan = saved.lan;
    g_last_model.set(saved.model());
    g_port.store(logic.choosePort(explicit, saved.port, g_default_port), .release);

    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe = selfPath(&exe_buf) orelse fail("cannot resolve its own path", .{});
    g_self_exe = try gpa.dupeSentinel(u8, self_exe, 0);
    if (proc.verifiedPid(g_io, g_bar_pid_path, bar_exe, false)) |e| {
        if (e.pid != proc.selfPid()) {
            out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
            return;
        }
    }
    if (!proc.isSessionLeader()) detach(gpa, args, g_self_exe);

    // The pid file is checked and written apart; the lock is what keeps two simultaneous launches to one.
    var lock_buf: [std.fs.max_path_bytes]u8 = undefined;
    const lock_path = std.fmt.bufPrintSentinel(&lock_buf, "{s}/.sushi/{s}", .{ g_home, lock_name }, 0) catch fail("HOME is too long", .{});
    std.Io.Dir.cwd().createDirPath(g_io, std.fs.path.dirname(lock_path) orelse ".") catch {};
    if (!proc.acquireLock(lock_path.ptr)) {
        out(g_io, "sushibar is already running\n", .{});
        return;
    }
    try proc.writePid(g_io, g_home, g_bar_pid_path, .{ .pid = proc.selfPid(), .port = port() });
    var sb: [path_cap]u8 = undefined;
    if (siblingNow(&sb, server_exe)) |sushi| {
        g_sushi_found = true;
        var real: [std.fs.max_path_bytes]u8 = undefined;
        g_sibling_brew = logic.isBrewKeg(proc.realPath(&real, sushi) orelse sushi);
    }
    adoptStarted();
    runMenuBar();
}
