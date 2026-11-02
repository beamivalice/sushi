//! `sushibar`: the menu bar item for a sushi server. It links no MLX and no server module: the title reads
//! `/metrics.json` over HTTP, Start and Stop run the `sushi` beside this binary, updates go through sushi's own
//! updater. AppKit is called only from the main thread; a poll thread publishes a snapshot that a main-run-loop
//! timer renders.
const std = @import("std");
const logic = @import("sushibar/sushibar_logic.zig");
const proc = @import("sushibar/sushibar_proc.zig");
const objc = @import("sushibar/objc.zig");
const io_util = @import("io_util.zig");

const usage =
    \\usage: sushibar [--port <n>]
    \\
    \\Puts sushi in the menu bar: decode tok/s as the title, and a menu to start and stop a server,
    \\switch the port, update sushi, and open the dashboard, chat, models and settings pages.
    \\It detaches from the terminal.
    \\
    \\  --port <n>   Port of the server to watch and start (default: the port last chosen
    \\               in the menu, else {d})
    \\  --help       This text
    \\
;

const poll_ms = 1000;
const server_exe = "sushi";
const bar_exe = "sushibar";
const bar_pid_name = "sushibar.pid";
const server_pid_name = "sushibar-server.pid";
const saved_name = "sushibar.json";
const capture_name = "sushibar-capture.txt";
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
            @memcpy(self.buf[0..self.len], s[0..self.len]);
        }

        fn get(self: *const @This()) []const u8 {
            return self.buf[0..self.len];
        }
    };
}

// ── Test seams: SUSHIBAR_TEST_POLLS, SUSHIBAR_TEST_ACTIONS (docs/envvar.md) ──

const ActionKind = enum { start, stop, confirm, check, update, port };
const Action = struct { kind: ActionKind, at: u32, value: u16 = 0 };
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
        if (std.mem.startsWith(u8, word, "port")) {
            act.kind = .port;
            act.value = std.fmt.parseInt(u16, word[4..], 10) catch continue;
        } else act.kind = std.meta.stringToEnum(ActionKind, word) orelse continue;
        if (act.kind == .confirm) test_confirm = true;
        test_actions[test_action_n] = act;
        test_action_n += 1;
    }
}

// ── State shared by the poll thread, the update worker and the main thread ──

const Listing = struct { port: u16 = 0, pid: i32 = 0, model: Str(96) = .{} };

const Shared = struct {
    mu: std.Io.Mutex = .init,
    /// The port the data below belongs to; stale once the watched port changes.
    port: u16 = 0,
    view: logic.View = .{ .link = .offline },
    model: Str(96) = .{},
    listings: [max_listings]Listing = undefined,
    listing_n: usize = 0,
    sushi_listening: bool = false,
    port_taken: bool = false,
    bar_version: Str(32) = .{},
    server_version: Str(32) = .{},
    /// What the running server's `/props.update` says.
    props_latest: Str(32) = .{},
    props_available: bool = false,
    props_command: Str(80) = .{},
    latest: Str(32) = .{},
    available: bool = false,
    note: Str(120) = .{},
    busy: logic.Busy = .none,
    check_failed: bool = false,
    relaunch: bool = false,
    req_check: bool = false,
    req_update: bool = false,
};

var shared: Shared = .{};
var g_io: std.Io = undefined;
var g_port = std.atomic.Value(u16).init(logic.default_port);
var g_home: []const u8 = "/tmp";
var g_sushi: ?[:0]const u8 = null;
var g_sushi_real: []const u8 = "";
var g_sibling_brew = false;
var g_self_exe: [:0]const u8 = "";
var g_bar_pid_path: []const u8 = "";
var g_server_pid_path: []const u8 = "";
var g_saved_path: []const u8 = "";
var g_capture_path: [:0]const u8 = "";
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

/// A server that accepts and never answers is offline after `poll_timeout_s`, not a stalled poll thread.
fn request(a: std.mem.Allocator, method: std.http.Method, url: []const u8, origin: ?[]const u8) ?Reply {
    const U = union(enum) { got: ?Reply, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(g_io, &buf);
    sel.concurrent(.got, fetchNow, .{ a, method, url, origin }) catch return fetchNow(a, method, url, origin);
    sel.concurrent(.timer, std.Io.sleep, .{ g_io, std.Io.Duration.fromSeconds(poll_timeout_s), .awake }) catch {};
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

fn modelOf(a: std.mem.Allocator, p: u16) ?[]const u8 {
    var b: [64]u8 = undefined;
    const r = get(a, urlFor(&b, p, "/v1/models")) orelse return null;
    if (r.status != 200) return null;
    return logic.parseReadyModel(a, r.body) catch null;
}

// ── Version and update (poll thread and worker) ──

fn readCapture(a: std.mem.Allocator) []const u8 {
    return std.Io.Dir.cwd().readFileAlloc(g_io, g_capture_path, a, .limited(64 * 1024)) catch "";
}

fn lastLine(text: []const u8) []const u8 {
    const t = std.mem.trimEnd(u8, text, " \r\n");
    const i = std.mem.lastIndexOfScalar(u8, t, '\n');
    return if (i) |k| t[k + 1 ..] else t;
}

fn refreshBarVersion(a: std.mem.Allocator) void {
    const sushi = g_sushi orelse return;
    if (proc.runCapture(&.{ sushi.ptr, "--version" }, g_capture_path.ptr) != 0) return;
    const v = logic.parseVersionOutput(readCapture(a)) orelse return;
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

/// `sushi update --check`, recorded in sushi's own cache like the daily check. False when it gave no answer.
fn cliCheck(a: std.mem.Allocator, manual: bool) bool {
    const sushi = g_sushi orelse return false;
    refreshBarVersion(a);
    const code = proc.runCapture(&.{ sushi.ptr, "update", "--check" }, g_capture_path.ptr);
    const text = readCapture(a);
    const result = logic.parseCheck(text);
    lock();
    defer unlock();
    if (code == null) {
        if (manual) shared.note.set("could not run sushi");
        return false;
    }
    switch (result) {
        .available => |v| {
            shared.latest.set(v);
            shared.available = true;
            shared.note.set("");
        },
        .up_to_date => {
            shared.available = false;
            shared.note.set(if (manual) "Up to date" else "");
        },
        .failed => |line| {
            if (manual) shared.note.set(line);
            return false;
        },
    }
    return true;
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
        var msg: [120]u8 = undefined;
        const text = std.fmt.bufPrint(&msg, "Update refused: {s}", .{if (e != null and e.?.@"error".message.len > 0) e.?.@"error".message else "the server did not accept it"}) catch "Update refused";
        return setNote(text);
    }
    setNote("Updating…");
    var down = false;
    var i: u32 = 0;
    while (i < 1200) : (i += 1) {
        std.Io.sleep(g_io, .fromMilliseconds(500), .real) catch {};
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
    var msg: [160]u8 = undefined;
    const why = if (u) |up| up.err orelse "sushi restarted on the version it had" else "the server did not come back";
    setNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{why}) catch "Update failed");
}

fn updateViaCli(a: std.mem.Allocator) void {
    const sushi = g_sushi orelse return;
    setNote("Updating…");
    const code = proc.runCapture(&.{ sushi.ptr, "update" }, g_capture_path.ptr);
    if (code == 0) {
        var msg: [64]u8 = undefined;
        refreshBarVersion(a);
        lock();
        const v = shared.bar_version.get();
        const text = std.fmt.bufPrint(&msg, "Updated to sushi {s}", .{v}) catch "Updated";
        unlock();
        return finishUpdate(a, text);
    }
    var msg: [160]u8 = undefined;
    setNote(std.fmt.bufPrint(&msg, "Update failed: {s}", .{lastLine(readCapture(a))}) catch "Update failed");
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
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    {
        _ = arena_state.reset(.retain_capacity);
        refreshBarVersion(arena_state.allocator());
    }
    while (true) : (std.Io.sleep(g_io, .fromMilliseconds(poll_ms), .real) catch {}) {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const now = io_util.nowMs(g_io);
        const p = port();
        if (p != tracked_port) {
            tracker = .{};
            tracked_port = p;
        }
        var url_buf: [96]u8 = undefined;
        const url = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/metrics.json?since={d}", .{ p, tracker.since_ms }) catch unreachable;
        var sample: logic.Sample = .{};
        const reply = get(a, url);
        sample.link = logic.linkFor(if (reply) |r| r.status else null);
        if (sample.link == .up) sample = logic.parseMetrics(a, reply.?.body) catch .{};
        tracker.observe(now, sample);

        var model: ?[]const u8 = null;
        if (sample.link == .up or sample.link == .metrics_off) model = modelOf(a, p);

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
            const m = if (l.port == p) model else modelOf(a, l.port);
            listings[n].model.set(m orelse "");
            n += 1;
        }
        const taken = sample.link == .offline and !sushi_listening and proc.portInUse(p);

        if (p == port()) {
            lock();
            shared.port = p;
            shared.view = tracker.view(sample.link);
            shared.model.set(model orelse "");
            shared.listings = listings;
            shared.listing_n = n;
            shared.sushi_listening = sushi_listening;
            shared.port_taken = taken;
            unlock();
        }

        iter += 1;
        if (iter % props_every == 1 and sample.link != .offline and sample.link != .unauthorized) refreshProps(a, p);

        lock();
        if (shared.check_failed) {
            last_ok = false;
            shared.check_failed = false;
        }
        const link = shared.view.link;
        const known = shared.props_latest.len > 0;
        const req_check = shared.req_check;
        const req_update = shared.req_update;
        shared.req_check = false;
        shared.req_update = false;
        unlock();
        if (req_update) startWorker(.update) else if (req_check) startWorker(.check) else if (logic.checkDue(last_check_ms, now, last_ok)) {
            if (link == .offline) {
                last_check_ms = now;
                last_ok = true;
                startWorker(.auto_check);
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
    }
}

// ── Port ──

fn writeSaved(p: u16) void {
    var text: [48]u8 = undefined;
    std.Io.Dir.cwd().createDirPath(g_io, std.fs.path.dirname(g_saved_path) orelse ".") catch {};
    std.Io.Dir.cwd().writeFile(g_io, .{ .sub_path = g_saved_path, .data = logic.formatSaved(&text, p) }) catch {};
}

fn setPort(p: u16) void {
    if (p == port()) return;
    g_port.store(p, .release);
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

// ── Start and Stop ──

fn ownServer() ?logic.PidEntry {
    const e = proc.verifiedPid(g_io, g_server_pid_path, server_exe, true) orelse return null;
    return if (e.port == port()) e else null;
}

fn startServer() void {
    const sushi = g_sushi orelse return;
    if (ownServer() != null) return;
    var port_buf: [8:0]u8 = undefined;
    const p = port();
    const port_z = std.fmt.bufPrintSentinel(&port_buf, "{d}", .{p}, 0) catch return;
    const pid = proc.spawnDetached(&.{ sushi.ptr, "--serve", "--port", port_z.ptr }, false) catch return;
    proc.writePid(g_io, g_home, g_server_pid_path, .{ .pid = pid, .port = p }) catch {};
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

/// A sushibar-started server stops at once; any other sushi on the port only after a confirm, re-verified.
fn stopServer() void {
    const p = port();
    var lbuf: [16]logic.Listener = undefined;
    const ours = ownServer();
    switch (logic.planStop(if (ours) |e| e.pid else null, p, proc.listeners(&lbuf, server_exe))) {
        .none => {},
        .now => |pid| proc.terminate(pid),
        .confirm => |pid| {
            if (!confirmStop(p, pid)) return;
            var again: [16]logic.Listener = undefined;
            switch (logic.planStop(null, p, proc.listeners(&again, server_exe))) {
                .confirm => |same| if (same == pid) proc.terminate(pid),
                else => {},
            }
        },
    }
}

fn askPort() ?u16 {
    var info: []const u8 = "A port from 1 to 65535.";
    while (true) {
        const field = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSTextField"), "alloc", .{}), "initWithFrame:", .{NSRect{ .x = 0, .y = 0, .w = 220, .h = 24 }});
        var cur: [8:0]u8 = undefined;
        _ = std.fmt.bufPrintSentinel(&cur, "{d}", .{port()}, 0) catch {};
        objc.msg(void, field, "setStringValue:", .{objc.nsString(&cur)});
        if (!runAlert("Watch which port?", info, "Watch", "Cancel", field)) return null;
        const c_str = objc.msg([*:0]const u8, objc.msg(objc.id, field, "stringValue", .{}), "UTF8String", .{});
        if (logic.parsePortText(std.mem.span(c_str))) |p| return p;
        info = "That is not a port. Type a number from 1 to 65535.";
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
var status_mi: objc.id = null;
var start_mi: objc.id = null;
var stop_mi: objc.id = null;
var port_menu: objc.id = null;
var version_mi: objc.id = null;
var check_mi: objc.id = null;
var update_mi: objc.id = null;
var copy_mi: objc.id = null;
var menu_target: objc.id = null;
var icons: [2]objc.id = .{ null, null };
var shown_icon: ?logic.Icon = null;
var tick_count: u32 = 0;
var last_title: [128]u8 = undefined;
var port_sig: [1024]u8 = undefined;
var port_sig_len: usize = std.math.maxInt(usize);

fn setText(setter_target: objc.id, selector: [*:0]const u8, text: []const u8) void {
    var z: [192:0]u8 = undefined;
    const n = @min(text.len, z.len - 1);
    @memcpy(z[0..n], text[0..n]);
    z[n] = 0;
    objc.msg(void, setter_target, selector, .{objc.nsString(&z)});
}

const NSSize = extern struct { width: f64, height: f64 };

fn loadIcon(svg: []const u8, template: bool) objc.id {
    const data = objc.msg(objc.id, objc.class("NSData"), "dataWithBytes:length:", .{ svg.ptr, svg.len });
    const image = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSImage"), "alloc", .{}), "initWithData:", .{data});
    objc.msg(void, image, "setSize:", .{NSSize{ .width = 22, .height = 18 }});
    objc.msg(void, image, "setTemplate:", .{template});
    return image;
}

fn showIcon(button: objc.id, icon: logic.Icon) void {
    if (shown_icon == icon) return;
    shown_icon = icon;
    objc.msg(void, button, "setImage:", .{icons[@intFromEnum(icon)]});
}

fn addItem(menu: objc.id, label: [*:0]const u8, action: ?[*:0]const u8, tag: isize) objc.id {
    const mi = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSMenuItem"), "alloc", .{}), "initWithTitle:action:keyEquivalent:", .{
        objc.nsString(label), if (action) |a| objc.sel(a) else null, objc.nsString(""),
    });
    objc.msg(void, mi, "setTarget:", .{menu_target});
    objc.msg(void, mi, "setTag:", .{tag});
    objc.msg(void, mi, "setEnabled:", .{action != null});
    objc.msg(void, menu, "addItem:", .{mi});
    return mi;
}

fn addSeparator(menu: objc.id) void {
    objc.msg(void, menu, "addItem:", .{objc.msg(objc.id, objc.class("NSMenuItem"), "separatorItem", .{})});
}

/// Rebuilds the Port submenu only when its entries changed, so an open menu is not disturbed.
fn syncPortMenu(entries: []const logic.PortEntry) void {
    var sig: [1024]u8 = undefined;
    var n: usize = 0;
    var lb: [96]u8 = undefined;
    for (entries) |e| {
        const l = logic.portLabel(&lb, e);
        const piece = std.fmt.bufPrint(sig[n..], "{s}{d}\n", .{ l, @intFromBool(e.checked) }) catch break;
        n += piece.len;
    }
    if (n == port_sig_len and std.mem.eql(u8, sig[0..n], port_sig[0..n])) return;
    port_sig_len = n;
    @memcpy(port_sig[0..n], sig[0..n]);
    objc.msg(void, port_menu, "removeAllItems", .{});
    for (entries) |e| {
        var z: [112:0]u8 = undefined;
        const label = logic.portLabel(&lb, e);
        const lz = std.fmt.bufPrintSentinel(&z, "{s}", .{label}, 0) catch continue;
        const mi = addItem(port_menu, lz.ptr, "pickPort:", e.port);
        if (e.checked) objc.msg(void, mi, "setState:", .{@as(isize, 1)});
    }
    addSeparator(port_menu);
    _ = addItem(port_menu, "Other Port…", "otherPort:", 0);
}

fn runAction(act: Action) void {
    switch (act.kind) {
        .start => startServer(),
        .stop => stopServer(),
        .confirm => {},
        .check => {
            lock();
            shared.req_check = true;
            unlock();
        },
        .update => {
            lock();
            shared.req_update = true;
            unlock();
        },
        .port => setPort(act.value),
    }
}

fn tick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    tick_count += 1;
    for (test_actions[0..test_action_n]) |act| if (act.at == tick_count) runAction(act);
    while (std.c.waitpid(-1, null, std.c.W.NOHANG) > 0) {}

    const p = port();
    lock();
    const fresh = shared.port == p;
    var view = shared.view;
    if (!fresh) view = .{ .link = .offline };
    var model_copy: [96]u8 = undefined;
    const model_len = if (fresh) shared.model.len else 0;
    @memcpy(model_copy[0..model_len], shared.model.buf[0..model_len]);
    var listings: [max_listings]logic.Listing = undefined;
    var models: [max_listings][96]u8 = undefined;
    var ln: usize = 0;
    for (shared.listings[0..shared.listing_n]) |l| {
        @memcpy(models[ln][0..l.model.len], l.model.get());
        listings[ln] = .{ .port = l.port, .pid = l.pid, .model = if (l.model.len > 0) models[ln][0..l.model.len] else null };
        ln += 1;
    }
    const sushi_listening = fresh and shared.sushi_listening;
    const port_taken = fresh and shared.port_taken;
    var bar_v: [32]u8 = undefined;
    var srv_v: [32]u8 = undefined;
    var latest: [32]u8 = undefined;
    var cmd: [80]u8 = undefined;
    var note: [120]u8 = undefined;
    @memcpy(bar_v[0..shared.bar_version.len], shared.bar_version.get());
    @memcpy(srv_v[0..shared.server_version.len], shared.server_version.get());
    @memcpy(latest[0..shared.latest.len], shared.latest.get());
    @memcpy(cmd[0..shared.props_command.len], shared.props_command.get());
    @memcpy(note[0..shared.note.len], shared.note.get());
    const uv_base: logic.UpdateView = .{
        .bar_version = bar_v[0..shared.bar_version.len],
        .server_version = if (fresh) srv_v[0..shared.server_version.len] else "",
        .latest = latest[0..shared.latest.len],
        .available = shared.available,
        .command = if (g_sibling_brew and shared.props_command.len == 0) "brew upgrade sushi" else cmd[0..shared.props_command.len],
        .note = note[0..shared.note.len],
        .busy = shared.busy,
    };
    const relaunch = shared.relaunch;
    const server_command = shared.props_command.len > 0;
    unlock();

    const state: logic.Server = .{
        .link = view.link,
        .model = if (model_len > 0) model_copy[0..model_len] else null,
        .ours_alive = ownServer() != null,
        .sushi_found = g_sushi != null,
        .port = p,
        .sushi_listening = sushi_listening,
        .port_taken = port_taken,
    };
    const on = logic.enabled(state);
    var status_buf: [200]u8 = undefined;
    const status = logic.statusLine(&status_buf, state);
    var uv = uv_base;
    uv.path = logic.updatePath(.{
        .available = uv.available,
        .link = view.link,
        .server_command = server_command,
        .sibling_brew = g_sibling_brew,
    });
    var tv = view;
    tv.update_waiting = uv.available;
    const title = logic.title(&last_title, tv);

    const button = objc.msg(objc.id, status_item, "button", .{});
    showIcon(button, logic.iconFor(view.link));
    setText(button, "setTitle:", title);
    setText(status_mi, "setTitle:", status);
    var lab: [96]u8 = undefined;
    setText(start_mi, "setTitle:", logic.startLabel(&lab, p));
    objc.msg(void, start_mi, "setEnabled:", .{on.start});
    setText(stop_mi, "setTitle:", logic.stopLabel(&lab, p));
    objc.msg(void, stop_mi, "setEnabled:", .{on.stop});

    var entries: [max_listings + 1]logic.PortEntry = undefined;
    const port_entries = logic.portEntries(&entries, p, listings[0..ln]);
    syncPortMenu(port_entries);

    var vb: [160]u8 = undefined;
    const version = logic.versionText(&vb, uv);
    setText(version_mi, "setTitle:", version);
    var cb: [160]u8 = undefined;
    const check = logic.checkLabel(&cb, uv);
    setText(check_mi, "setTitle:", check);
    objc.msg(void, check_mi, "setEnabled:", .{uv.busy == .none});
    var ub: [200]u8 = undefined;
    const update_label = logic.updateLabel(&ub, uv);
    objc.msg(void, update_mi, "setHidden:", .{update_label == null});
    if (update_label) |l| setText(update_mi, "setTitle:", l);
    objc.msg(void, update_mi, "setEnabled:", .{logic.updateEnabled(uv)});
    objc.msg(void, copy_mi, "setHidden:", .{!(uv.available and uv.path == .brew)});

    if (relaunch) relaunchSelf(p);

    if (test_polls > 0) {
        var ports_buf: [256]u8 = undefined;
        var pn: usize = 0;
        for (port_entries) |e| {
            const piece = std.fmt.bufPrint(ports_buf[pn..], "{s}{s};", .{ logic.portLabel(&lab, e), if (e.checked) "*" else "" }) catch break;
            pn += piece.len;
        }
        std.debug.print("sushibar: poll {d} title=\"{s}\" status=\"{s}\" start={d} stop={d} server_pid={d} icon={s} ports=\"{s}\" version=\"{s}\" check=\"{s}\" update=\"{s}\"\n", .{
            tick_count,                                          title,                                status,
            @intFromBool(on.start),                              @intFromBool(on.stop),                if (ownServer()) |e| e.pid else 0,
            @tagName(shown_icon.?),                              ports_buf[0..pn],                     version,
            check,                                               update_label orelse "",
        });
        if (tick_count >= test_polls) quit();
    }
}

/// The sushi beside sushibar changed version: run the new sushibar in this process, on the same port.
fn relaunchSelf(p: u16) void {
    lock();
    shared.relaunch = false;
    unlock();
    var port_z: [8:0]u8 = undefined;
    const pz = std.fmt.bufPrintSentinel(&port_z, "{d}", .{p}, 0) catch return;
    proc.execReplace(&.{ g_self_exe.ptr, "--port", pz.ptr });
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

fn startAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    startServer();
}
fn stopAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    stopServer();
}
fn quitAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    quit();
}
fn pickPortAction(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    setPort(@intCast(objc.msg(isize, sender, "tag", .{})));
}
fn otherPortAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    if (askPort()) |p| setPort(p);
}
fn checkAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    runAction(.{ .kind = .check, .at = 0 });
}
fn updateAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    runAction(.{ .kind = .update, .at = 0 });
}
fn copyAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    objc.copyText("brew upgrade sushi");
}

fn runMenuBar() noreturn {
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "setActivationPolicy:", .{@as(isize, 1)});
    const bar = objc.msg(objc.id, objc.class("NSStatusBar"), "systemStatusBar", .{});
    status_item = objc.msg(objc.id, bar, "statusItemWithLength:", .{@as(f64, -1)});
    const button = objc.msg(objc.id, status_item, "button", .{});
    icons[@intFromEnum(logic.Icon.outline)] = loadIcon(@embedFile("sushibar/icon-offline.svg"), true);
    icons[@intFromEnum(logic.Icon.color)] = loadIcon(@embedFile("sushibar/icon-online.svg"), false);
    objc.msg(void, button, "setImagePosition:", .{@as(isize, 2)});
    showIcon(button, .outline);

    const cls = objc.defineClass("SushiBarTarget", &.{
        .{ .selector = "tick:", .imp = @ptrCast(&tick), .types = "v@:@" },
        .{ .selector = "open:", .imp = @ptrCast(&openPage), .types = "v@:@" },
        .{ .selector = "start:", .imp = @ptrCast(&startAction), .types = "v@:@" },
        .{ .selector = "stop:", .imp = @ptrCast(&stopAction), .types = "v@:@" },
        .{ .selector = "quit:", .imp = @ptrCast(&quitAction), .types = "v@:@" },
        .{ .selector = "pickPort:", .imp = @ptrCast(&pickPortAction), .types = "v@:@" },
        .{ .selector = "otherPort:", .imp = @ptrCast(&otherPortAction), .types = "v@:@" },
        .{ .selector = "check:", .imp = @ptrCast(&checkAction), .types = "v@:@" },
        .{ .selector = "update:", .imp = @ptrCast(&updateAction), .types = "v@:@" },
        .{ .selector = "copy:", .imp = @ptrCast(&copyAction), .types = "v@:@" },
    });
    menu_target = objc.new(cls);
    const menu = objc.new(objc.class("NSMenu"));
    objc.msg(void, menu, "setAutoenablesItems:", .{false});
    status_mi = addItem(menu, "Port: …", null, 0);
    start_mi = addItem(menu, "Start Server", "start:", 0);
    stop_mi = addItem(menu, "Stop Server", "stop:", 0);
    const port_item = addItem(menu, "Port", null, 0);
    objc.msg(void, port_item, "setEnabled:", .{true});
    port_menu = objc.new(objc.class("NSMenu"));
    objc.msg(void, port_menu, "setAutoenablesItems:", .{false});
    objc.msg(void, port_item, "setSubmenu:", .{port_menu});
    addSeparator(menu);
    _ = addItem(menu, "Open Dashboard", "open:", @intFromEnum(logic.Page.dashboard));
    _ = addItem(menu, "Open Chat", "open:", @intFromEnum(logic.Page.chat));
    _ = addItem(menu, "Models", "open:", @intFromEnum(logic.Page.models));
    _ = addItem(menu, "Settings", "open:", @intFromEnum(logic.Page.settings));
    addSeparator(menu);
    version_mi = addItem(menu, "sushi", null, 0);
    check_mi = addItem(menu, "Check for Updates", "check:", 0);
    update_mi = addItem(menu, "Update", "update:", 0);
    objc.msg(void, update_mi, "setHidden:", .{true});
    copy_mi = addItem(menu, "Copy Command", "copy:", 0);
    objc.msg(void, copy_mi, "setHidden:", .{true});
    addSeparator(menu);
    _ = addItem(menu, "Quit", "quit:", 0);
    objc.msg(void, status_item, "setMenu:", .{menu});
    objc.addTimer(@as(f64, poll_ms) / 1000.0, menu_target, "tick:");

    _ = std.Thread.spawn(.{}, pollLoop, .{}) catch fail("cannot start the poll thread", .{});
    objc.msg(void, app, "run", .{});
    std.process.exit(0);
}

// ── Launch ──

fn selfPath(buf: []u8) ?[:0]const u8 {
    var tmp: [std.fs.max_path_bytes]u8 = undefined;
    var n: u32 = tmp.len;
    if (std.c._NSGetExecutablePath(&tmp, &n) != 0) return null;
    const raw: [*:0]const u8 = @ptrCast(&tmp);
    const real = std.c.realpath(raw, buf.ptr) orelse return null;
    return std.mem.span(real);
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
        if (std.c.waitpid(child, null, std.c.W.NOHANG) == child) fail("the detached copy exited at once (no window server?)", .{});
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
    if (!logic.guiSession(env("SSH_CONNECTION"), env("SSH_TTY")))
        fail("no GUI session (this is an ssh login): run sushibar from the Mac's desktop session", .{});

    g_home = env("HOME") orelse "/tmp";
    var p1: [std.fs.max_path_bytes]u8 = undefined;
    var p2: [std.fs.max_path_bytes]u8 = undefined;
    var p3: [std.fs.max_path_bytes]u8 = undefined;
    var p4: [std.fs.max_path_bytes]u8 = undefined;
    g_bar_pid_path = try gpa.dupe(u8, proc.pidFilePath(&p1, g_home, bar_pid_name) orelse fail("HOME is too long", .{}));
    g_server_pid_path = try gpa.dupe(u8, proc.pidFilePath(&p2, g_home, server_pid_name) orelse fail("HOME is too long", .{}));
    g_saved_path = try gpa.dupe(u8, proc.pidFilePath(&p3, g_home, saved_name) orelse fail("HOME is too long", .{}));
    g_capture_path = try gpa.dupeSentinel(u8, proc.pidFilePath(&p4, g_home, capture_name) orelse fail("HOME is too long", .{}), 0);
    readTestSeams();

    const saved_text = std.Io.Dir.cwd().readFileAlloc(g_io, g_saved_path, gpa, .limited(4096)) catch "";
    g_port.store(logic.choosePort(explicit, logic.parseSaved(saved_text)), .release);

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

    try proc.writePid(g_io, g_home, g_bar_pid_path, .{ .pid = proc.selfPid(), .port = port() });
    var sibling: [std.fs.max_path_bytes]u8 = undefined;
    if (logic.siblingPath(&sibling, self_exe, server_exe)) |p| {
        if (std.Io.Dir.cwd().access(g_io, p, .{ .execute = true })) |_| {
            g_sushi = try gpa.dupeSentinel(u8, p, 0);
            var real: [std.fs.max_path_bytes]u8 = undefined;
            const r = std.c.realpath(g_sushi.?.ptr, &real);
            g_sushi_real = try gpa.dupe(u8, if (r) |rp| std.mem.span(rp) else p);
            g_sibling_brew = logic.isBrewKeg(g_sushi_real);
        } else |_| {}
    }
    runMenuBar();
}
