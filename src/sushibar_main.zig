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
    \\Puts sushi in the menu bar: decode tok/s as the title, and a menu to start and stop a server,
    \\switch the port, update sushi, and open the dashboard, chat, models and settings pages.
    \\It detaches from the terminal.
    \\
    \\  --port <n>   Port of the server to watch and start (default: the port last chosen
    \\               in the menu, else the default port: `server.port` in
    \\               ~/.sushi/model-settings.json, else {d})
    \\  --help       This text
    \\
;

const poll_ms = 1000;
const server_exe = "sushi";
const bar_exe = "sushibar";
const bar_pid_name = "sushibar.pid";
const server_pid_prefix = "sushibar-server-";
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

const ActionKind = enum { start, stop, confirm, check, update, port, typed, models, remove };
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
    model: Str(96) = .{},
    listings: [max_listings]Listing = undefined,
    listing_n: usize = 0,
    models: [16]ModelCache = @splat(.{}),
    sushi_listening: bool = false,
    port_taken: bool = false,
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
    up_to_date: bool = false,
    bar_version: Str(32) = .{},
    server_version: Str(32) = .{},
    /// What the running server's `/props.update` says.
    props_latest: Str(32) = .{},
    props_available: bool = false,
    props_command: Str(80) = .{},
    latest: Str(32) = .{},
    available: bool = false,
    note: Str(240) = .{},
    busy: logic.Busy = .none,
    check_failed: bool = false,
    /// Where the watched port's server writes its log; stale once the watched port changes (`port`).
    log_kind: logic.LogKind = .standard,
    log_path: Str(1024) = .{},
    relaunch: bool = false,
    /// How the last check the user asked for ended, and when; shown on the Update button.
    manual_result: logic.CheckResult = .none,
    manual_at_ms: i64 = 0,
};

var shared: Shared = .{};
var g_io: std.Io = undefined;
var g_port = std.atomic.Value(u16).init(logic.default_port);
/// The port the menu lists first and never lets go: `server.port` of the settings file, re-read as the menu opens.
var g_default_port: u16 = logic.default_port;
/// The ports the user added; the menu and the Port field are the main thread's.
var g_saved: logic.SavedPorts = .{};
var g_home: []const u8 = "/tmp";
var g_sushi_found = false;
var g_sibling_brew = false;
/// The path sushibar was started by, unresolved: a Homebrew link follows the keg of the day.
var g_self_exe: [:0]const u8 = "";
var g_bar_pid_path: []const u8 = "";
var g_saved_path: []const u8 = "";
var g_capture_seq = std.atomic.Value(u32).init(0);
var g_launched_version: Str(32) = .{};

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
fn request(a: std.mem.Allocator, method: std.http.Method, url: []const u8, origin: ?[]const u8) ?Reply {
    return requestWithin(a, method, url, origin, poll_timeout_s);
}

fn requestWithin(a: std.mem.Allocator, method: std.http.Method, url: []const u8, origin: ?[]const u8, timeout_s: i64) ?Reply {
    const U = union(enum) { got: ?Reply, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(g_io, &buf);
    sel.concurrent(.got, fetchNow, .{ a, method, url, origin }) catch return fetchNow(a, method, url, origin);
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

fn get(a: std.mem.Allocator, url: []const u8) ?Reply {
    return request(a, .GET, url, null);
}

fn fetchNow(a: std.mem.Allocator, method: std.http.Method, url: []const u8, origin: ?[]const u8) ?Reply {
    var client: std.http.Client = .{ .allocator = a, .io = g_io };
    defer client.deinit();
    var body: std.Io.Writer.Allocating = .init(a);
    var hdr = [_]std.http.Header{.{ .name = "origin", .value = origin orelse "" }};
    const res = client.fetch(.{
        .location = .{ .url = url },
        .method = method,
        .payload = if (method == .POST) "" else null,
        .extra_headers = if (origin != null) hdr[0..1] else &.{},
        .keep_alive = false,
        .response_writer = &body.writer,
    }) catch return null;
    return .{ .status = @intFromEnum(res.status), .body = body.written() };
}

fn urlFor(buf: []u8, p: u16, comptime path: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "http://127.0.0.1:{d}" ++ path, .{p}) catch unreachable;
}

fn modelsOf(a: std.mem.Allocator, p: u16) logic.ModelState {
    var b: [64]u8 = undefined;
    const r = get(a, urlFor(&b, p, "/v1/models")) orelse return .{};
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

/// One foreign listener's models, with a short timeout, off the poll thread.
fn modelsWorker(p: u16) void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var b: [64]u8 = undefined;
    var known: logic.ModelsKnown = .failed;
    var name: []const u8 = "";
    if (requestWithin(a, .GET, urlFor(&b, p, "/v1/models"), null, models_timeout_s)) |r| {
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
    shared.note.set(text);
    unlock();
}

/// Folds a check's answer into what the menu shows. False when it was a failure.
fn applyCheck(result: logic.Check, manual: bool) bool {
    lock();
    defer unlock();
    switch (result) {
        .available => |v| {
            shared.latest.set(v);
            shared.available = true;
            shared.up_to_date = false;
            shared.note.set("");
        },
        .up_to_date => {
            shared.available = false;
            shared.up_to_date = true;
            shared.note.set("");
        },
        .failed => |line| {
            if (manual) shared.note.set(line);
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
        if (manual) setNote("could not run sushi");
        return false;
    }
    return applyCheck(logic.parseCheck(got.text), manual);
}

fn refreshProps(a: std.mem.Allocator, p: u16) void {
    var b: [64]u8 = undefined;
    const r = get(a, urlFor(&b, p, "/props")) orelse return;
    if (r.status != 200) return;
    const u = (logic.parseProps(a, r.body) catch return) orelse return;
    lock();
    const changed = shared.server_version.len > 0 and !std.mem.eql(u8, shared.server_version.get(), u.current);
    shared.server_version.set(u.current);
    shared.props_latest.set(u.latest orelse "");
    shared.props_available = u.available;
    shared.props_command.set(u.command orelse "");
    unlock();
    if (changed) refreshBarVersion(a);
}

fn finishUpdate(a: std.mem.Allocator, ok_note: []const u8) void {
    refreshBarVersion(a);
    lock();
    defer unlock();
    shared.available = false;
    shared.note.set(ok_note);
}

fn updateViaServer(a: std.mem.Allocator, p: u16) void {
    var url: [64]u8 = undefined;
    var origin: [48]u8 = undefined;
    const origin_s = std.fmt.bufPrint(&origin, "http://127.0.0.1:{d}", .{p}) catch return;
    var from_buf: [32]u8 = undefined;
    lock();
    const from = from_buf[0..shared.server_version.len];
    @memcpy(from, shared.server_version.get());
    unlock();
    const r = request(a, .POST, urlFor(&url, p, "/v1/update"), origin_s) orelse return setNote("Update failed: the server did not answer");
    if (r.status != 202) {
        const E = struct { @"error": struct { message: []const u8 = "" } = .{} };
        const e = std.json.parseFromSliceLeaky(E, a, r.body, .{ .ignore_unknown_fields = true }) catch null;
        var msg: [400]u8 = undefined;
        const text = std.fmt.bufPrint(&msg, "Update refused: {s}", .{if (e != null and e.?.@"error".message.len > 0) e.?.@"error".message else "the server did not accept it"}) catch "Update refused";
        return setNote(text);
    }
    setNote("Updating…");
    var down = false;
    var i: u32 = 0;
    while (i < update_wait_polls) : (i += 1) {
        std.Io.sleep(g_io, .fromMilliseconds(500), .real) catch {};
        if (port() != p) return setNote("Update started; no longer followed after the port change");
        var hb: [64]u8 = undefined;
        const alive = if (get(a, urlFor(&hb, p, "/health"))) |h| h.status == 200 else false;
        if (!alive) {
            down = true;
            continue;
        }
        if (down) break;
        var pb: [64]u8 = undefined;
        const props = get(a, urlFor(&pb, p, "/props")) orelse continue;
        const u = (logic.parseProps(a, props.body) catch null) orelse continue;
        if (!std.mem.eql(u8, u.current, from)) break;
    }
    var pb: [64]u8 = undefined;
    const props = get(a, urlFor(&pb, p, "/props"));
    const u = if (props) |pr| (logic.parseProps(a, pr.body) catch null) else null;
    refreshProps(a, p);
    if (u) |up| if (!std.mem.eql(u8, up.current, from)) {
        var msg: [64]u8 = undefined;
        return finishUpdate(a, std.fmt.bufPrint(&msg, "Updated to sushi {s}", .{up.current}) catch "Updated");
    };
    var msg: [400]u8 = undefined;
    const why = if (u) |up| up.err orelse "sushi restarted on the version it had" else "the server did not come back";
    setNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{why}) catch "Update failed");
}

/// Two minutes of half-second polls: past that the update is reported as failed instead of locking the menu.
const update_wait_polls = 240;

fn updateViaCli(a: std.mem.Allocator) void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return setNote("could not find sushi beside sushibar");
    setNote("Updating…");
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
    setNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{logic.failureLine(got.text)}) catch "Update failed");
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
                .none => setNote("Nothing to update"),
                .brew => setNote("Homebrew install: run the command in the menu"),
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
    const t = std.Thread.spawn(.{}, worker, .{kind}) catch {
        lock();
        shared.busy = .none;
        unlock();
        return;
    };
    t.detach();
}

// ── Poll thread ──

fn pollLoop() void {
    var tracker: logic.Tracker = .{};
    var tracked_port: u16 = 0;
    var last_check_ms: i64 = 0;
    var last_ok = true;
    var iter: u32 = 0;
    var loop_start_ms: i64 = 0;
    var log_pid: i32 = 0;
    var log_port: u16 = 0;
    var log_buf: [1024]u8 = undefined;
    var log_src: logic.LogSource = .{ .kind = .standard };
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    {
        _ = arena_state.reset(.retain_capacity);
        refreshBarVersion(arena_state.allocator());
    }
    while (true) : ({
        const spent = io_util.nowMs(g_io) - loop_start_ms;
        if (spent < poll_ms) std.Io.sleep(g_io, .fromMilliseconds(poll_ms - spent), .real) catch {};
    }) {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const now = io_util.nowMs(g_io);
        loop_start_ms = now;
        const p = port();
        if (p != tracked_port) {
            tracker = .{};
            tracked_port = p;
            lock();
            shared.hist_n = 0;
            shared.last_prefill = null;
            unlock();
        }
        var url_buf: [96]u8 = undefined;
        const url = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/metrics.json?since={d}", .{ p, tracker.since_ms }) catch unreachable;
        var lbuf: [16]logic.Listener = undefined;
        var listings: [max_listings]Listing = undefined;
        var n: usize = 0;
        var sushi_listening = false;
        for (proc.listeners(&lbuf, server_exe)) |l| {
            if (l.port == p) sushi_listening = true;
            const dup = for (listings[0..n]) |x| {
                if (x.port == l.port) break true;
            } else false;
            if (dup or n == max_listings) continue;
            listings[n] = .{ .port = l.port, .pid = l.pid };
            n += 1;
        }

        var sample: logic.Sample = .{};
        const reply = get(a, url);
        sample.link = logic.linkFor(if (reply) |r| r.status else null);
        if (sample.link == .up) sample = logic.parseMetrics(a, reply.?.body) catch .{};
        // A sushi that holds the port and does not answer is not a stopped one.
        if (sample.link == .offline and sushi_listening) sample.link = .silent;
        tracker.observe(now, sample);

        var models: logic.ModelState = .{};
        if (sample.link == .up or sample.link == .metrics_off) models = modelsOf(a, p);
        const model = models.ready;

        for (listings[0..n]) |*l| {
            if (l.port != p) continue;
            l.model.set(model orelse "");
            l.known = if (sample.link == .up or sample.link == .metrics_off) .answered else if (sample.link == .silent) .failed else .unknown;
        }
        const taken = sample.link == .offline and !sushi_listening and proc.portInUse(p);

        // The server's own argv says whether it was started with --log-file; read again only when its pid changes.
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
        }

        if (p == port()) {
            lock();
            shared.port = p;
            shared.view = tracker.view(sample.link);
            shared.model.set(model orelse "");
            shared.model_loading.set(models.loading orelse "");
            const pu = models.pack_update orelse logic.PackUpdate{};
            shared.pack_current.set(pu.current);
            shared.pack_latest.set(pu.latest);
            shared.pack_command.set(pu.command);
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
            shared.port_taken = taken;
            shared.log_kind = log_src.kind;
            shared.log_path.set(log_src.path);
            unlock();
        }

        iter += 1;
        if (iter % props_every == 1 and sample.link != .offline and sample.link != .silent and sample.link != .unauthorized) refreshProps(a, p);

        lock();
        if (shared.check_failed) {
            last_ok = false;
            shared.check_failed = false;
        }
        const link = shared.view.link;
        const known = shared.props_latest.len > 0;
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
                shared.up_to_date = !shared.props_available;
                shared.latest.set(shared.props_latest.get());
                unlock();
            }
        }
    }
}

// ── Port ──

fn writeSaved(p: u16) void {
    var text: [192]u8 = undefined;
    std.Io.Dir.cwd().createDirPath(g_io, std.fs.path.dirname(g_saved_path) orelse ".") catch {};
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = g_saved_path, .data = logic.formatSaved(&text, p, g_saved.items()) }) catch {};
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
    g_saved.add(p, g_default_port);
    writeSaved(p);
    lock();
    shared.view = .{ .link = .offline };
    shared.model.set("");
    shared.server_version.set("");
    shared.props_latest.set("");
    shared.props_available = false;
    shared.props_command.set("");
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
    return std.fmt.bufPrint(buf, "{s}/.sushi/{s}{d}.pid", .{ g_home, server_pid_prefix, p }) catch "";
}

fn ownServer() ?logic.PidEntry {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const e = proc.verifiedPid(g_io, serverPidPath(&pb, port()), server_exe, true) orelse return null;
    return if (e.port == port()) e else null;
}

/// Servers this process spawned: each is reaped by its own pid, and one that exits before it ever answered is a
/// failed start.
const Started = struct { pid: i32 = 0, port: u16 = 0, answered: bool = false, stopping: bool = false };
var started: [8]Started = @splat(.{});

fn reapStarted() void {
    for (&started) |*st| {
        if (st.pid == 0) continue;
        if (std.c.waitpid(st.pid, null, std.c.W.NOHANG) != st.pid) continue;
        if (!st.answered and !st.stopping) {
            var nb: [160]u8 = undefined;
            setNote(logic.startFailedNote(&nb, st.port));
        }
        st.* = .{};
    }
}

fn markAnswered(p: u16) void {
    for (&started) |*st| if (st.pid != 0 and st.port == p) {
        st.answered = true;
    };
}

fn markStopping(pid: i32) void {
    for (&started) |*st| if (st.pid == pid) {
        st.stopping = true;
    };
}

fn startServer() void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return setNote("could not find sushi beside sushibar");
    if (ownServer() != null) return;
    var port_buf: [8:0]u8 = undefined;
    const p = port();
    const port_z = std.fmt.bufPrintSentinel(&port_buf, "{d}", .{p}, 0) catch return;
    const pid = proc.spawnDetached(&.{ sushi.ptr, "--serve", "--port", port_z.ptr }, false) catch return setNote("could not start sushi");
    setNote("");
    for (&started) |*st| if (st.pid == 0) {
        st.* = .{ .pid = pid, .port = p };
        break;
    };
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    proc.writePid(g_io, g_home, serverPidPath(&pb, p), .{ .pid = pid, .port = p }) catch {};
}

const NSRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

fn runAlert(message: []const u8, info: []const u8, ok: [*:0]const u8, cancel: ?[*:0]const u8, field: objc.id) bool {
    var m: [160:0]u8 = undefined;
    var i: [200:0]u8 = undefined;
    const mz = std.fmt.bufPrintSentinel(&m, "{s}", .{message}, 0) catch return false;
    const iz = std.fmt.bufPrintSentinel(&i, "{s}", .{info}, 0) catch return false;
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "activateIgnoringOtherApps:", .{true});
    const alert = objc.new(objc.class("NSAlert"));
    defer objc.release(alert);
    objc.msg(void, alert, "setMessageText:", .{objc.nsString(mz.ptr)});
    objc.msg(void, alert, "setInformativeText:", .{objc.nsString(iz.ptr)});
    _ = objc.msg(objc.id, alert, "addButtonWithTitle:", .{objc.nsString(ok)});
    if (cancel) |c| _ = objc.msg(objc.id, alert, "addButtonWithTitle:", .{objc.nsString(c)});
    if (field != null) {
        objc.msg(void, alert, "setAccessoryView:", .{field});
        objc.msg(void, objc.msg(objc.id, alert, "window", .{}), "setInitialFirstResponder:", .{field});
    }
    return objc.msg(isize, alert, "runModal", .{}) == 1000;
}

fn confirmStop(p: u16, pid: i32) bool {
    if (test_polls > 0) return test_confirm;
    var m: [96]u8 = undefined;
    const text = std.fmt.bufPrint(&m, "Stop sushi on port {d} (pid {d})?", .{ p, pid }) catch return false;
    return runAlert(text, "sushibar did not start it.", "Stop", "Cancel", null);
}

/// How long the menu says "Stopping…" before it trusts what the poll says.
const stopping_ms = 15_000;
var stopping_until_ms: i64 = 0;

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
            stopping_until_ms = io_util.nowMs(g_io) + stopping_ms;
        },
        .confirm => |pid| {
            if (!confirmStop(p, pid)) return;
            var again: [16]logic.Listener = undefined;
            switch (logic.planStop(null, p, proc.listeners(&again, server_exe))) {
                .confirm => |same| if (same == pid) {
                    proc.terminate(pid);
                    stopping_until_ms = io_util.nowMs(g_io) + stopping_ms;
                },
                else => {},
            }
        },
    }
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
var toggle_mi: objc.id = null;
var port_mi: objc.id = null;
var port_menu: objc.id = null;
var open_mis: [4]objc.id = .{ null, null, null, null };
var button_action: logic.ButtonAction = .none;
var copy_mi: objc.id = null;
var pack_mi: objc.id = null;
var menu_target: objc.id = null;
var top_menu: objc.id = null;
var port_field_mi: objc.id = null;
var note_mi: objc.id = null;
const note_show_max = 100;
var icons: [2]objc.id = .{ null, null };
var shown_icon: ?logic.Icon = null;
var shown_symbol: ?[*]const u8 = null;
var tick_count: u32 = 0;
var last_title: [128]u8 = undefined;
var shown_title: [128]u8 = undefined;
var shown_title_len: usize = std.math.maxInt(usize);
var shown_dim = false;
var port_sig: [1536]u8 = undefined;
var port_sig_len: usize = std.math.maxInt(usize);

fn setText(setter_target: objc.id, selector: [*:0]const u8, text: []const u8) void {
    var z: [192:0]u8 = undefined;
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
    shown_icon = icon;
    objc.msg(void, button, "setImage:", .{icons[@intFromEnum(icon)]});
}

/// The button title in monospaced digits, so its width does not jitter; the last request's speed is dim.
fn setButtonTitle(button: objc.id, text: []const u8, dim: bool) void {
    if (shown_title_len == text.len and shown_dim == dim and std.mem.eql(u8, shown_title[0..text.len], text)) return;
    shown_title_len = text.len;
    shown_dim = dim;
    @memcpy(shown_title[0..text.len], text);
    var z: [192:0]u8 = undefined;
    @memcpy(z[0..text.len], text);
    z[text.len] = 0;
    const s = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsString(&z), header_view.titleAttributes(dim) });
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
    // The submenu ends with a separator and the port field, which stay; only the entries above them change.
    port_row.reset();
    while (objc.msg(isize, port_menu, "numberOfItems", .{}) > 2) objc.msg(void, port_menu, "removeItemAtIndex:", .{@as(isize, 0)});
    var width: f64 = port_row.min_width;
    for (entries) |e| width = @max(width, port_row.widthFor(logic.portLabel(&lb, e)));
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

fn runAction(act: Action) void {
    switch (act.kind) {
        .start => startServer(),
        .stop => stopServer(),
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
        .typed => {
            var z: [16:0]u8 = undefined;
            const t = std.fmt.bufPrintSentinel(&z, "{d}", .{act.value}, 0) catch return;
            port_field.setText(t.ptr);
            applyPortText();
        },
    }
}

/// Return in the field or Watch: a valid port switches what is watched and closes the menu; anything else turns
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
    reapStarted();
    refresh();
}

/// Renders the published snapshot into the menu, from the timer and straight after a click that changes it.
fn refresh() void {
    const p = port();
    const now = io_util.nowMs(g_io);
    var model_b: [96]u8 = undefined;
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
    const last_prefill = if (fresh) shared.last_prefill else null;
    const hist_n = if (fresh) shared.hist_n else 0;
    @memcpy(hist[0..hist_n], shared.hist[0..hist_n]);
    const up_to_date = shared.up_to_date;
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
    unlock();

    if (view.link == .up or view.link == .metrics_off) markAnswered(p);
    const own = ownServer();
    const ours_alive = own != null;
    if (view.link == .offline or stopping_until_ms <= now) stopping_until_ms = 0;
    const stopping = stopping_until_ms > now;
    const state: logic.Server = .{
        .link = view.link,
        .model = if (model.len > 0) model else null,
        .ours_alive = ours_alive,
        .sushi_found = g_sushi_found,
        .port = p,
        .sushi_listening = sushi_listening,
        .port_taken = port_taken,
    };
    const on = logic.enabled(state);
    const tg = logic.toggle(state, stopping);
    var uv = uv_base;
    uv.path = logic.updatePath(.{ .available = uv.available, .link = view.link, .server_command = server_command, .sibling_brew = g_sibling_brew });

    // The header card, then everything else it owns the numbers of.
    var arena_buf: [8192]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&arena_buf);
    const header = logic.buildHeader(fba.allocator(), .{
        .link = view.link,
        .port = p,
        .loading = loading_id.len > 0 and model.len == 0,
        .updating = uv.busy == .updating,
        .starting = ours_alive and view.link == .offline,
        .model = if (model.len > 0) model else if (loading_id.len > 0) loading_id else null,
        .pack_update = pack_update,
        .mem_mb = mem_mb,
        .bar_version = bar_version,
        .server_version = server_version,
        .update_available = uv.available,
        .latest = uv.latest,
        .up_to_date = up_to_date,
        .decoding = view.decoding,
        .live = view.live,
        .last = view.last,
        .last_prefill = last_prefill,
        .gpu_mhz = gpu_mhz,
        .gpu_pct = gpu_pct,
    }) catch unreachable;
    var spark: [160]logic.SparkPoint = undefined;
    header_view.update(header, logic.sparkPoints(&spark, hist[0..hist_n], now, header_view.spark_window_ms));

    var tv = view;
    tv.update_waiting = uv.available;
    const title = logic.title(&last_title, tv);
    const button = objc.msg(objc.id, status_item, "button", .{});
    showIcon(button, logic.iconFor(view.link));
    setButtonTitle(button, title, logic.titleDim(view));
    var status_buf: [200]u8 = undefined;
    const status = logic.statusLine(&status_buf, state);
    setText(button, "setToolTip:", status);

    setText(toggle_mi, "setTitle:", tg.label);
    objc.msg(void, toggle_mi, "setEnabled:", .{tg.enabled});
    if (shown_symbol != tg.symbol.ptr) {
        shown_symbol = tg.symbol.ptr;
        objc.msg(void, toggle_mi, "setImage:", .{symbol(tg.symbol)});
    }
    var lab: [96]u8 = undefined;
    setText(port_mi, "setTitle:", std.fmt.bufPrint(&lab, "Port {d}", .{p}) catch "Port");
    for (open_mis) |mi| objc.msg(void, mi, "setEnabled:", .{logic.openEnabled(view.link)});

    var entries: [max_listings + logic.max_saved + 2]logic.PortEntry = undefined;
    const port_entries = logic.portEntries(&entries, p, g_default_port, g_saved.items(), listings[0..ln]);
    syncPortMenu(port_entries);

    var cb: [128]u8 = undefined;
    const check = logic.checkLabel(&cb, uv);
    var fit_b: [note_show_max + 4]u8 = undefined;
    const note_shown = if (uv.busy == .none and uv.note.len > 0) logic.fitNote(&fit_b, uv.note, note_show_max) else "";
    setText(note_mi, "setTitle:", note_shown);
    objc.msg(void, note_mi, "setHidden:", .{note_shown.len == 0});
    var ub: [200]u8 = undefined;
    const update_label = if (uv.busy == .updating) check else logic.updateLabel(&ub, uv) orelse "";
    var tip_b: [160]u8 = undefined;
    const update_button = logic.updateButton(&tip_b, uv);
    button_action = update_button.action;
    button_row.setUpdate(buttonLook(update_button));
    button_row.dropStaleHover();
    var pb: [128]u8 = undefined;
    const copy_label = logic.copyLabel(&pb, uv.command);
    objc.msg(void, copy_mi, "setHidden:", .{!(uv.available and uv.path == .brew)});
    setText(copy_mi, "setTitle:", copy_label);
    var pcb: [128]u8 = undefined;
    const pack_copy = if (pack_update) |u| logic.copyLabel(&pcb, u.command) else "";
    objc.msg(void, pack_mi, "setHidden:", .{pack_update == null});
    setText(pack_mi, "setTitle:", pack_copy);

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
        const shown_update = if (uv.available and uv.path == .brew) copy_label else if (uv.available) update_label else "";
        const shown_check = if (uv.available) "" else check;
        std.debug.print("sushibar: poll {d} title=\"{s}\" status=\"{s}\" start={d} stop={d} server_pid={d} icon={s} ports=\"{s}\" version=\"{s}\" check=\"{s}\" update=\"{s}\" note=\"{s}\" pill=\"{s}\" line1=\"{s}\" line2=\"{s}\" toggle=\"{s}\" field_invalid={d} pack=\"{s}\" packcopy=\"{s}\" button=\"{s}\" removable=\"{s}\"\n", .{
            tick_count,                 title,                 status,                 @intFromBool(on.start),
            @intFromBool(on.stop),      if (own) |e| e.pid else 0,
            @tagName(shown_icon.?),     ports_buf[0..pn],      header.version,         shown_check,
            shown_update,               note_shown,            header.pill.label,     header.line1,           header.line2,
            tg.label,                   @intFromBool(port_field.isInvalid()), header.pack_line, pack_copy,
            update_button.label,        removable_buf[0..rn],
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
    setNote("Updated, but sushibar could not restart itself: quit and run it again");
}

fn openPage(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    const tag = objc.msg(isize, sender, "tag", .{});
    var url_buf: [96:0]u8 = undefined;
    const page: logic.Page = @enumFromInt(@as(u2, @intCast(tag)));
    const url = logic.pageUrl(&url_buf, port(), page);
    url_buf[url.len] = 0;
    const ns_url = objc.msg(objc.id, objc.class("NSURL"), "URLWithString:", .{objc.nsString(&url_buf)});
    _ = objc.msg(bool, objc.msg(objc.id, objc.class("NSWorkspace"), "sharedWorkspace", .{}), "openURL:", .{ns_url});
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
    };
    unlock();
    switch (logic.toggle(state, stopping_until_ms > io_util.nowMs(g_io)).action) {
        .start => startServer(),
        .stop => stopServer(),
        .none => {},
    }
}
fn quitAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    quit();
}
fn pickPortAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    setPort(@intCast(objc.msg(isize, sender, "tag", .{})));
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
        .update => runAction(.{ .kind = .update, .at = 0 }),
        .copy_command => copyAction(null, null, null),
        .none => {},
    }
    refresh();
}
fn copyAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    objc.copyText("brew upgrade sushi");
}
fn copyPackAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    var cmd: [97]u8 = undefined;
    lock();
    const n = shared.pack_command.len;
    @memcpy(cmd[0..n], shared.pack_command.get());
    unlock();
    cmd[n] = 0;
    if (n > 0) objc.copyText(@ptrCast(&cmd));
}

fn pickRowAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    if (port_row.portOf(sender)) |p| setPort(p);
}
fn removeRowAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    if (port_row.portOf(sender)) |p| removePort(p);
}

/// The top menu re-reads the default port; the Port submenu asks its listeners for their models.
fn menuWillOpen(_: objc.id, _: objc.SEL, opening: objc.id) callconv(.c) void {
    if (opening != top_menu) return refreshModels(null);
    g_default_port = readDefaultPort();
    lock();
    shared.manual_result = logic.shownResult(shared.manual_result, shared.manual_at_ms, io_util.nowMs(g_io));
    unlock();
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

fn runMenuBar() noreturn {
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "setActivationPolicy:", .{@as(isize, 1)});
    const bar = objc.msg(objc.id, objc.class("NSStatusBar"), "systemStatusBar", .{});
    status_item = objc.msg(objc.id, bar, "statusItemWithLength:", .{@as(f64, -1)});
    const button = objc.msg(objc.id, status_item, "button", .{});
    loadIcons();
    objc.msg(void, button, "setImagePosition:", .{@as(isize, 2)});
    showIcon(button, .outline);

    const cls = objc.defineClass("SushiBarTarget", &.{
        .{ .selector = "tick:", .imp = @ptrCast(&tick), .types = "v@:@" },
        .{ .selector = "open:", .imp = @ptrCast(&openPage), .types = "v@:@" },
        .{ .selector = "toggle:", .imp = @ptrCast(&toggleAction), .types = "v@:@" },
        .{ .selector = "quit:", .imp = @ptrCast(&quitAction), .types = "v@:@" },
        .{ .selector = "pickPort:", .imp = @ptrCast(&pickPortAction), .types = "v@:@" },
        .{ .selector = "pickRow:", .imp = @ptrCast(&pickRowAction), .types = "v@:@" },
        .{ .selector = "removeRow:", .imp = @ptrCast(&removeRowAction), .types = "v@:@" },
        .{ .selector = "applyPort:", .imp = @ptrCast(&applyPortAction), .types = "v@:@" },
        .{ .selector = "controlTextDidChange:", .imp = @ptrCast(&textChangedAction), .types = "v@:@" },
        .{ .selector = "updateButton:", .imp = @ptrCast(&updateButtonAction), .types = "v@:@" },
        .{ .selector = "showLog:", .imp = @ptrCast(&showLogAction), .types = "v@:@" },
        .{ .selector = "logTick:", .imp = @ptrCast(&logTick), .types = "v@:@" },
        .{ .selector = "copy:", .imp = @ptrCast(&copyAction), .types = "v@:@" },
        .{ .selector = "copyPack:", .imp = @ptrCast(&copyPackAction), .types = "v@:@" },
        .{ .selector = "menuWillOpen:", .imp = @ptrCast(&menuWillOpen), .types = "v@:@" },
    });
    menu_target = objc.new(cls);
    const menu = objc.new(objc.class("NSMenu"));
    top_menu = menu;
    objc.msg(void, menu, "setDelegate:", .{menu_target});
    objc.msg(void, menu, "setAutoenablesItems:", .{false});

    header_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, header_mi, "setView:", .{header_view.create(
        objc.loadSvg(@embedFile("sushibar/icon-online.svg"), false, 34, 28),
        objc.loadSvg(@embedFile("sushibar/icon-offline.svg"), true, 34, 28),
    )});
    objc.msg(void, menu, "addItem:", .{header_mi});
    addSeparator(menu);

    toggle_mi = addItem(menu, "Start Server", "toggle:", 0, "", "play.fill");
    port_mi = addItem(menu, "Port", null, 0, "", "network");
    objc.msg(void, port_mi, "setEnabled:", .{true});
    port_menu = objc.new(objc.class("NSMenu"));
    objc.msg(void, port_menu, "setAutoenablesItems:", .{false});
    objc.msg(void, port_mi, "setSubmenu:", .{port_menu});
    objc.msg(void, port_menu, "setDelegate:", .{menu_target});
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

    copy_mi = addItem(menu, "Copy", "copy:", 0, "", "doc.on.doc");
    objc.msg(void, copy_mi, "setHidden:", .{true});
    pack_mi = addItem(menu, "Copy", "copyPack:", 0, "", "doc.on.doc");
    objc.msg(void, pack_mi, "setHidden:", .{true});
    note_mi = addItem(menu, "", null, 0, "", null);
    objc.msg(void, note_mi, "setHidden:", .{true});
    addSeparator(menu);

    const buttons_mi = objc.new(objc.class("NSMenuItem"));
    objc.msg(void, buttons_mi, "setView:", .{button_row.create(menu_target, menu)});
    objc.msg(void, menu, "addItem:", .{buttons_mi});
    // The Quit button is the visible row; this item only keeps ⌘Q.
    const quit_mi = addItem(menu, "Quit sushibar", "quit:", 0, "q", null);
    objc.msg(void, quit_mi, "setHidden:", .{true});
    objc.msg(void, status_item, "setMenu:", .{menu});
    objc.addTimer(@as(f64, poll_ms) / 1000.0, menu_target, "tick:");
    objc.addTimer(0.5, menu_target, "logTick:");

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
        .{ .name = "no-model", .in = .{ .link = .up, .port = 12345, .bar_version = "1.5.0-dev1", .up_to_date = true, .gpu_mhz = 0 } },
        .{ .name = "generating", .sparkline = true, .outage = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0-dev1", .up_to_date = true,
            .decoding = true, .live = 105.27, .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
        } },
        .{ .name = "key-needed", .in = .{ .link = .unauthorized, .port = 12345, .bar_version = "1.5.0-dev1" } },
        .{ .name = "update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = "Example-Flash-Next-Sushi-STQ-2.75bpw", .mem_mb = 31200, .bar_version = "1.5.0",
            .update_available = true, .latest = "1.5.1", .last = 142.6, .last_prefill = 3120, .gpu_mhz = 1380, .gpu_pct = 8,
        } },
        .{ .name = "pack-update", .sparkline = true, .in = .{
            .link = .up, .port = 12345, .model = deep, .mem_mb = 49971, .bar_version = "1.5.0", .up_to_date = true,
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
    std.process.exit(0);
}

fn renderButtonsAndLog(dir: []const u8) void {
    const row = button_row.create(null, null);
    const Case = struct { name: []const u8, uv: logic.UpdateView = .{}, hover: ?usize = null, pressed: ?usize = null };
    const cases = [_]Case{
        .{ .name = "normal" },
        .{ .name = "hover-log", .hover = 0 },
        .{ .name = "pressed-quit", .hover = 2, .pressed = 2 },
        .{ .name = "update", .uv = .{ .available = true, .latest = "1.5.1", .path = .server } },
        .{ .name = "update-hover", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .hover = 1 },
        .{ .name = "update-pressed", .uv = .{ .available = true, .latest = "1.5.1", .path = .server }, .pressed = 1 },
        .{ .name = "brew", .uv = .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi" } },
        .{ .name = "checking", .uv = .{ .busy = .checking } },
        .{ .name = "up-to-date", .uv = .{ .checked = .up_to_date } },
        .{ .name = "failed", .uv = .{ .checked = .failed } },
        .{ .name = "failed-hover", .uv = .{ .checked = .failed }, .hover = 1 },
    };
    for (cases) |c| {
        var tip: [160]u8 = undefined;
        const b = logic.updateButton(&tip, c.uv);
        button_row.setUpdate(buttonLook(b));
        for ([_]bool{ false, true }) |dark| {
            var path: [std.fs.max_path_bytes:0]u8 = undefined;
            const pz = std.fmt.bufPrintSentinel(&path, "{s}/buttons-{s}-{s}.png", .{ dir, c.name, if (dark) "dark" else "light" }, 0) catch continue;
            if (!button_row.renderPng(row, pz.ptr, dark, c.hover, c.pressed)) std.debug.print("sushibar: could not render {s}\n", .{pz});
        }
    }
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
        if (!log_view.renderPng(pz.ptr, false, .{ .port = 11250, .src = .{ .kind = .standard, .path = std_path }, .sample = notice })) std.debug.print("sushibar: could not render {s}\n", .{pz});
    } else |_| {}
    renderFollowedLog(dir);
}

/// The Port submenu's rows: the default (not removable), an added port, a listener nobody added.
fn renderPortRows(dir: []const u8) void {
    const rows = [_]port_row.Row{
        .{ .port = 12345, .label = "12345 — Example-Flash-Next-Sushi-STQ-2.75bpw", .checked = true, .removable = false },
        .{ .port = 18803, .label = "18803 — not running", .checked = false, .removable = true },
        .{ .port = 18802, .label = "18802 — Example-Flash-1017", .checked = false, .removable = true },
        .{ .port = 7000, .label = "7000 — no model", .checked = false, .removable = false },
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

/// The window reading a real file: its last lines on open, then what the file grew by.
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
    for (3001..3006) |i| w.print("line {d} appended later\n", .{i}) catch return;
    w.writeAll("a line still being writ") catch return;
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = file, .data = text.written() }) catch return;
    log_view.sync(11250, src);
    var png: [std.fs.max_path_bytes:0]u8 = undefined;
    const pz = std.fmt.bufPrintSentinel(&png, "{s}/log-window-followed-light.png", .{dir}, 0) catch return;
    if (!log_view.renderPng(pz.ptr, false, null)) std.debug.print("sushibar: could not render {s}\n", .{pz});
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
    const saved_text = std.Io.Dir.cwd().readFileAlloc(g_io, g_saved_path, gpa, .limited(4096)) catch "";
    const saved = logic.parseSaved(saved_text);
    g_saved = saved.ports;
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
    runMenuBar();
}
