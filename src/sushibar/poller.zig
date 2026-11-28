//! Cross-thread state and background workers for `sushibar`: HTTP polling of the watched server, parallel
//! foreign-listener `/v1/models` fetches, update checks/installs, and load/unload model requests.
//! All fields inside `shared` are guarded by `lock()` / `unlock()`; AppKit is never called here except via
//! `Config.notify_main`.
const std = @import("std");
const io_util = @import("../io_util.zig");
const logic = @import("sushibar_logic.zig");
const proc = @import("sushibar_proc.zig");
const http_local = @import("http_local.zig");

pub const Str = logic.FixedStr;
pub const Reply = http_local.Reply;

pub const tick_ms = 1000;
pub const server_exe = "sushi";
pub const bar_exe = "sushibar";
pub const max_listings = 8;
pub const note_cap = 2048;
pub const origin_cap = 64;
pub const url_cap = 128;
pub const path_cap = std.fs.max_path_bytes + 1;

const props_every = 5;
const poll_timeout_s = 4;
const models_timeout_s = 1;
const load_timeout_s = 1800;
const fast_after_action_ms = 30_000;
/// Two minutes of half-second polls: past that the update is reported as failed instead of locking the menu.
const update_wait_polls = 240;

pub const Listing = struct { port: u16 = 0, pid: i32 = 0, model: Str(96) = .{}, known: logic.ModelsKnown = .unknown };

/// What a foreign listener's `/v1/models` last said; fetched only when the Port submenu opens.
pub const ModelCache = struct { port: u16 = 0, known: logic.ModelsKnown = .unknown, busy: bool = false, model: Str(96) = .{} };

pub const Shared = struct {
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
    launched_version: Str(32) = .{},
    bar_version: Str(32) = .{},
    server_version: Str(32) = .{},
    /// What the running server's `/props.update` says.
    props_latest: Str(32) = .{},
    props_available: bool = false,
    props_command: Str(80) = .{},
    latest: Str(32) = .{},
    available: bool = false,
    note: Str(note_cap) = .{},
    /// The note says something went wrong: it is drawn orange and opens the Log window.
    note_problem: bool = false,
    note_kind: logic.NoteKind = .other,
    /// The port a `.server` note is about.
    note_port: u16 = 0,
    note_at_ms: i64 = 0,
    props_url: Str(160) = .{},
    busy: logic.Busy = .none,
    /// A manual Check requested while a startup auto-check is already in flight runs immediately afterward.
    manual_check_queued: bool = false,
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

pub const Config = struct {
    io: std.Io,
    home: []const u8,
    self_exe: [:0]const u8,
    test_polls: u32,
    test_tick_ms: u32,
    notify_main: *const fn () void,
};

pub const WorkerKind = enum { auto_check, check, update };

pub var shared: Shared = .{};
/// Wakes the poll thread before its wait is over: the menu opened, or the user started something.
pub var wake: std.Io.Event = .unset;
pub var menu_open = std.atomic.Value(bool).init(false);
/// The port a server is being asked to load a model on, 0 when none is.
pub var load_port = std.atomic.Value(u16).init(0);
/// The port a server is being asked to unload a model on, 0 when none is.
pub var unload_port = std.atomic.Value(u16).init(0);

var g_cfg: Config = undefined;
var g_sibling_brew = false;
var g_port = std.atomic.Value(u16).init(logic.default_port);
var g_capture_seq = std.atomic.Value(u32).init(0);
/// Until then the poll runs every second: a start, stop, port switch or update the user just asked for.
var g_fast_until_ms = std.atomic.Value(i64).init(0);

pub fn init(cfg: Config) void {
    g_cfg = cfg;
}

pub fn setSiblingBrew(brew: bool) void {
    g_sibling_brew = brew;
}

pub fn lock() void {
    shared.mu.lockUncancelable(g_cfg.io);
}

pub fn unlock() void {
    shared.mu.unlock(g_cfg.io);
}

pub fn port() u16 {
    return g_port.load(.acquire);
}

pub fn storePort(p: u16) void {
    g_port.store(p, .release);
}

fn notifyMain() void {
    g_cfg.notify_main();
}

/// `ms` of the poll schedule at the test tick.
fn ticked(ms: i64) i64 {
    return @divTrunc(ms * g_cfg.test_tick_ms, tick_ms);
}

/// Polls at once and then every second for a while: the user just changed something.
pub fn pollSoon() void {
    g_fast_until_ms.store(io_util.nowMs(g_cfg.io) + fast_after_action_ms, .release);
    wake.set(g_cfg.io);
}

// ── HTTP ──

fn request(a: std.mem.Allocator, p: u16, method: http_local.Method, url: []const u8, origin: ?[]const u8) ?Reply {
    return requestWithin(a, p, method, url, origin, poll_timeout_s, "");
}

/// `p` is the port `url` names: only the watched server's own key is sent, and only to it.
fn requestOutcome(a: std.mem.Allocator, p: u16, method: http_local.Method, url: []const u8, origin: ?[]const u8, timeout_s: i64, payload: []const u8) http_local.Outcome {
    var auth_buf: [600]u8 = undefined;
    const auth = authFor(&auth_buf, p);
    return http_local.fetch(a, method, url, origin, auth, payload, timeout_s * 1000);
}

fn requestWithin(a: std.mem.Allocator, p: u16, method: http_local.Method, url: []const u8, origin: ?[]const u8, timeout_s: i64, payload: []const u8) ?Reply {
    return switch (requestOutcome(a, p, method, url, origin, timeout_s, payload)) {
        .reply => |r| r,
        .conn_refused, .failed => null,
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

/// `http://host:port` of the server on `p`: the address its listener is bound to, else loopback.
pub fn originFor(buf: []u8, p: u16) []const u8 {
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
    const c = cacheSlot(p);
    c.port = p;
    c.known = known;
    c.busy = false;
    c.model.set(name);
    unlock();
    notifyMain();
}

/// The Port submenu is about to be read: ask the listed ports' listeners for their models, in parallel. This is the
/// only time sushibar talks to a server it is not watching, and never to one the menu does not list.
pub fn refreshModels(only: ?u16, default_port: u16, saved: *const logic.SavedPorts) void {
    var ports: [max_listings]u16 = undefined;
    var n: usize = 0;
    lock();
    for (shared.listings[0..shared.listing_n]) |l| {
        if (l.port == port() or (only != null and only.? != l.port)) continue;
        if (l.port != default_port and !saved.has(l.port)) continue;
        const c = cacheSlot(l.port);
        if (c.busy) continue;
        if (c.port != l.port) {
            c.port = l.port;
            c.known = .unknown;
            c.model.set("");
        }
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

// ── Sibling executables & notes ──

/// `name` beside this binary, resolved now: a Homebrew upgrade replaces the keg under a running sushibar, so the
/// path found at launch may be gone. Falls back to brew's stable `opt` link.
pub fn siblingNow(buf: *[path_cap]u8, name: []const u8) ?[:0]const u8 {
    var real: [std.fs.max_path_bytes]u8 = undefined;
    var cand: [std.fs.max_path_bytes]u8 = undefined;
    if (proc.realPath(&real, g_cfg.self_exe)) |r| {
        if (logic.siblingPath(&cand, r, name)) |c| if (reachable(c)) return sentinel(buf, c);
        if (logic.brewOptPath(&cand, r, name)) |c| if (reachable(c)) return sentinel(buf, c);
    }
    if (logic.siblingPath(&cand, g_cfg.self_exe, name)) |c| if (reachable(c)) return sentinel(buf, c);
    return null;
}

pub fn reachable(path: []const u8) bool {
    std.Io.Dir.cwd().access(g_cfg.io, path, .{ .execute = true }) catch return false;
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
    const path = std.fmt.bufPrintSentinel(&pb, "{s}/.sushi/sushibar-capture-{d}-{d}.txt", .{ g_cfg.home, proc.selfPid(), g_capture_seq.fetchAdd(1, .monotonic) }, 0) catch return .{ .code = null, .text = "" };
    const code = proc.runCapture(argv, path.ptr);
    const text = std.Io.Dir.cwd().readFileAlloc(g_cfg.io, path, a, .limited(64 * 1024)) catch "";
    std.Io.Dir.cwd().deleteFile(g_cfg.io, path) catch {};
    return .{ .code = code, .text = text };
}

fn refreshBarVersion(a: std.mem.Allocator) void {
    var sb: [path_cap]u8 = undefined;
    const sushi = siblingNow(&sb, server_exe) orelse return;
    const got = capture(a, &.{ sushi.ptr, "--version" });
    if (got.code != 0) return;
    const v = logic.parseVersionOutput(got.text) orelse return;
    lock();
    if (shared.launched_version.len == 0) shared.launched_version.set(v);
    shared.bar_version.set(v);
    // Whatever replaced the sibling (our update, a terminal `sushi update`, a server's /v1/update) restarts us.
    if (logic.relaunchNeeded(shared.launched_version.get(), v)) shared.relaunch = true;
    unlock();
}

pub fn setNote(text: []const u8) void {
    lock();
    defer unlock();
    putNote(text, false, .other, 0);
}

pub fn setProblem(text: []const u8) void {
    lock();
    defer unlock();
    putNote(text, true, .other, 0);
}

/// A note about the update flow: a finished check replaces it.
fn setUpdateNote(text: []const u8, problem: bool) void {
    lock();
    putNote(text, problem, .update, 0);
    unlock();
    notifyMain();
}

/// A problem with the server on `p`: it stays until a server answers there again or Start is pressed.
pub fn setServerProblem(text: []const u8, p: u16) void {
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
    shared.note_at_ms = io_util.nowMs(g_cfg.io);
}

/// Holding the lock.
pub fn clearNote() void {
    putNote("", false, .other, 0);
}

// ── Version and update worker ──

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
    const path = std.fmt.bufPrint(&pb, "{s}/.sushi/update-check.json", .{g_cfg.home}) catch return null;
    const text = std.Io.Dir.cwd().readFileAlloc(g_cfg.io, path, a, .limited(64 * 1024)) catch return null;
    lock();
    const bar = a.dupe(u8, shared.bar_version.get()) catch "";
    unlock();
    return logic.cachedCheck(a, text, io_util.nowSecs(g_cfg.io), bar);
}

/// Whether `model-settings.json` leaves the automatic check on.
fn autoCheckEnabled(a: std.mem.Allocator) bool {
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&pb, "{s}/.sushi/model-settings.json", .{g_cfg.home}) catch return true;
    const text = std.Io.Dir.cwd().readFileAlloc(g_cfg.io, path, a, .limited(1 << 20)) catch return true;
    return logic.autoCheckOn(a, text);
}

/// `sushi update --check`, recorded in sushi's own cache like the daily check. False when it gave no answer. The
/// automatic one reuses a cache under a day old; the menu's own "Check for updates" always asks.
fn cliCheck(a: std.mem.Allocator, manual: bool) bool {
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
pub fn errorMessage(a: std.mem.Allocator, body: []const u8) []const u8 {
    const E = struct { @"error": ?struct { message: []const u8 = "" } = null };
    const e = std.json.parseFromSliceLeaky(E, a, body, .{ .ignore_unknown_fields = true }) catch return "";
    return (e.@"error" orelse return "").message;
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
        return setUpdateNote(std.fmt.allocPrint(a, "Update refused: {s}", .{if (why.len > 0) why else "the server did not accept it"}) catch "Update refused", true);
    }
    setUpdateNote("Updating…", false);
    var poll_arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer poll_arena.deinit();
    var down = false;
    var i: u32 = 0;
    while (i < update_wait_polls) : (i += 1) {
        std.Io.sleep(g_cfg.io, .fromMilliseconds(500), .real) catch {};
        if (port() != p) return setUpdateNote("Update started; no longer followed after the port change", false);
        _ = poll_arena.reset(.retain_capacity);
        const pa = poll_arena.allocator();
        var hb: [url_cap]u8 = undefined;
        const alive = if (get(pa, p, urlFor(&hb, p, "/health"))) |h| h.status == 200 else false;
        if (!alive) {
            down = true;
            continue;
        }
        if (down) break;
        var pb: [url_cap]u8 = undefined;
        const props = get(pa, p, urlFor(&pb, p, "/props")) orelse continue;
        const u = (logic.parseProps(pa, props.body) catch null) orelse continue;
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
    const why = if (u) |up| up.err orelse "sushi restarted on the version it had" else "the server did not come back";
    setUpdateNote(std.fmt.allocPrint(a, "Update failed: {s}", .{why}) catch "Update failed", true);
}

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
    setUpdateNote(std.fmt.allocPrint(a, "Update failed: {s}", .{logic.failureLine(got.text)}) catch "Update failed", true);
}

fn worker(kind: WorkerKind) void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const p = port();
    var current = kind;
    while (true) {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        switch (current) {
            .auto_check, .check => {
                const ok = cliCheck(a, current == .check);
                lock();
                shared.check_failed = !ok;
                if (current == .check) {
                    shared.manual_result = if (!ok) .failed else if (shared.available) .none else .up_to_date;
                    shared.manual_at_ms = io_util.nowMs(g_cfg.io);
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
        if (current == .auto_check and shared.manual_check_queued) {
            shared.manual_check_queued = false;
            unlock();
            current = .check;
            continue;
        }
        shared.busy = .none;
        unlock();
        break;
    }
    notifyMain();
}

pub fn startWorker(kind: WorkerKind) void {
    lock();
    if (shared.busy != .none) {
        if (kind == .check and shared.busy == .checking) shared.manual_check_queued = true;
        unlock();
        return;
    }
    shared.busy = if (kind == .update) .updating else .checking;
    shared.manual_check_queued = false;
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

// ── Model load / unload worker ──

const ModelOp = struct { port: u16, unload: bool, model: Str(logic.max_model) };

pub fn startModelOp(unload: bool, model: []const u8) bool {
    const p = port();
    if (load_port.load(.acquire) != 0 or unload_port.load(.acquire) != 0 or model.len == 0) return false;
    const slot = if (unload) &unload_port else &load_port;
    if (slot.cmpxchgStrong(0, p, .acq_rel, .acquire) != null) return false;
    var job: ModelOp = .{ .port = p, .unload = unload, .model = .{} };
    job.model.set(model);
    const t = std.Thread.spawn(.{}, modelOpWorker, .{job}) catch {
        slot.store(0, .release);
        return false;
    };
    t.detach();
    pollSoon();
    return true;
}

fn modelOpWorker(job: ModelOp) void {
    defer {
        (if (job.unload) &unload_port else &load_port).store(0, .release);
        pollSoon();
        notifyMain();
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
    setProblem(std.fmt.allocPrint(a, "Port {d}: could not {s} {s}: {s}", .{ job.port, verb, job.model.get(), if (why.len > 0) why else if (reply == null) "no answer" else "the server refused" }) catch "Could not load or unload the model");
}

// ── Poll thread ──

fn probeStarted(a: std.mem.Allocator, listings: []const Listing, watched: u16) void {
    var ports: [8]u16 = undefined;
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

/// Digest of the fields `refresh()` actually acts on while the menu is closed (bar icon, rendered readout,
/// reading-fill step, badge, update dot, tooltip, model persistence, probe completion, listener PIDs for child
/// reaping, and self-relaunch). Open menus wake unconditionally via `menu_open`, so closed-menu-only fields like
/// `hist_n`, versions, or notes never cause idle AppKit wakeups.
/// Caller holds `shared.mu`.
fn publishedSignature() u64 {
    var w = std.hash.Wyhash.init(0);
    w.update(std.mem.asBytes(&shared.port));
    w.update(std.mem.asBytes(&@intFromEnum(shared.view.link)));
    w.update(std.mem.asBytes(&.{ @intFromBool(shared.view.decoding), @intFromBool(shared.view.prefilling), @intFromBool(shared.sushi_listening), @intFromBool(shared.port_taken), @intFromBool(shared.available), @intFromBool(shared.relaunch) }));
    var rb: [2 * logic.readout_max + 8]u8 = undefined;
    const ro = logic.readout(&rb, shared.view);
    w.update(ro.caption);
    w.update("\x00");
    w.update(ro.value);
    w.update("\x00");
    if (shared.view.read) |r| {
        const step: u8 = @intCast(logic.readingStep(r.fraction));
        w.update(std.mem.asBytes(&step));
    } else w.update("R");
    w.update(shared.model.get());
    w.update("\x00");
    w.update(std.mem.asBytes(&shared.model_full_len));
    w.update(shared.model_loading.get());
    w.update("\x00");
    w.update(std.mem.asBytes(&shared.answered_n));
    // Include listener (port, pid) set so crashed unwatched child servers wake `reapStarted()`, without hashing
    // volatile fields or model strings that could cause idle wakeups.
    var pids_xor: u64 = 0;
    for (shared.listings[0..shared.listing_n]) |l| {
        var lw = std.hash.Wyhash.init(0);
        lw.update(std.mem.asBytes(&l.port));
        lw.update(std.mem.asBytes(&l.pid));
        pids_xor ^= lw.final();
    }
    w.update(std.mem.asBytes(&shared.listing_n));
    w.update(std.mem.asBytes(&pids_xor));
    return w.final();
}

pub fn startPollThread() !void {
    _ = try std.Thread.spawn(.{}, pollLoop, .{});
}

fn pollLoop() void {
    var tracker: logic.Tracker = .{};
    var tracked_port: u16 = 0;
    var last_check_ms: i64 = 0;
    var last_ok = true;
    var last_pushed_sig: ?u64 = null;
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
        const woken = wake.isSet();
        wake.reset();
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const now = io_util.nowMs(g_cfg.io);
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
        const open_now = menu_open.load(.acquire);
        var url_buf: [url_cap + 32]u8 = undefined;
        var again_buf: [url_cap + 32]u8 = undefined;
        var url = metricsUrl(&url_buf, p, tracker.since_ms);
        const initial_outcome = requestOutcome(a, p, .GET, url, null, poll_timeout_s, "");
        var reply: ?Reply = switch (initial_outcome) {
            .reply => |r| r,
            .conn_refused, .failed => null,
        };
        const loopback_refused = initial_outcome == .conn_refused and std.mem.startsWith(u8, url, "http://127.0.0.1:");
        var reached = logic.linkFor(if (reply) |r| r.status else null);

        since_scan += 1;
        if (logic.scanDue(since_scan, reached, open_now)) {
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
            log_src = logic.logSource(&log_buf, g_cfg.home, p, read);
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
        if (reached == .up) sample = logic.parseMetrics(a, reply.?.body) catch logic.unreadable(reached);
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
        const taken = sample.link == .offline and !sushi_listening and !loopback_refused and proc.portInUse(p);

        var published = false;
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
            published = true;
            unlock();
        }

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

        // Compute the digest after refreshProps / checkDue so update dots, server versions, and relaunches land
        // on the main thread in this same poll cycle rather than waiting for the next idle poll.
        if (g_cfg.test_polls == 0 and published) {
            lock();
            const pub_sig = publishedSignature();
            unlock();
            if (woken or open_now or last_pushed_sig == null or pub_sig != last_pushed_sig.?) {
                last_pushed_sig = pub_sig;
                notifyMain();
            }
        }

        const activity: logic.Activity = .{
            .link = sample.link,
            .decoding = tracker.decoding or tracker.prefilling,
            .menu_open = open_now,
            .changing = now < g_fast_until_ms.load(.acquire) or models_loading or busy or load_port.load(.acquire) != 0 or unload_port.load(.acquire) != 0,
        };
        const spent = io_util.nowMs(g_cfg.io) - now;
        const wait = ticked(logic.pollWaitMs(activity)) - spent;
        if (wait > 0) wake.waitTimeout(g_cfg.io, .{ .duration = .{ .raw = .fromMilliseconds(wait), .clock = .awake } }) catch {};
    }
}
