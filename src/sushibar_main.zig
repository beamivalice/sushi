//! `sushibar`: the menu bar item for a sushi server. It links no MLX and no server module: the title reads
//! `/metrics.json` over HTTP, Start and Stop run the `sushi` beside this binary, updates go through sushi's own
//! updater. This entry file owns CLI startup/locking, local server lifecycle, menu wiring, and AppKit callbacks
//! (strictly main thread); cross-thread polling/workers live in `sushibar/poller.zig`, bar visuals in
//! `sushibar/bar_button.zig`, and the PNG design harness in `sushibar/render_gallery.zig`.
const std = @import("std");
const io_util = @import("io_util.zig");
const logic = @import("sushibar/sushibar_logic.zig");
const proc = @import("sushibar/sushibar_proc.zig");
const objc = @import("sushibar/objc.zig");
const header_view = @import("sushibar/header_view.zig");
const port_field = @import("sushibar/port_field.zig");
const button_row = @import("sushibar/button_row.zig");
const log_view = @import("sushibar/log_view.zig");
const port_row = @import("sushibar/port_row.zig");
const main_bar = @import("sushibar/main_bar.zig");
const bar_button = @import("sushibar/bar_button.zig");
const poller = @import("sushibar/poller.zig");
const render_gallery = @import("sushibar/render_gallery.zig");

const Str = logic.FixedStr;

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

const bar_pid_name = "sushibar.pid";
const saved_name = "sushibar.json";
const lock_name = "sushibar.lock";

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

// ── Test seams: SUSHIBAR_TEST_POLLS, SUSHIBAR_TEST_ACTIONS, SUSHIBAR_TEST_TICK_MS (docs/envvar.md) ──

const ActionKind = enum { start, stop, load, unload, confirm, check, update, port, typed, models, remove, lan };
const Action = struct { kind: ActionKind, at: u32, value: u32 = 0 };
var test_polls: u32 = 0;
var test_actions: [12]Action = undefined;
var test_action_n: usize = 0;
var test_confirm = false;
/// The menu's tick under test: every poll interval shrinks by `test_tick_ms / tick_ms`.
var test_tick_ms: u32 = poller.tick_ms;

fn readTestSeams() void {
    if (env("SUSHIBAR_TEST_MULTIMONITOR")) |v| {
        if (std.mem.eql(u8, v, "1")) main_bar.setForceMainWindow(true);
    }
    test_polls = std.fmt.parseInt(u32, env("SUSHIBAR_TEST_POLLS") orelse "", 10) catch 0;
    if (test_polls > 0) {
        const t = std.fmt.parseInt(u32, env("SUSHIBAR_TEST_TICK_MS") orelse "", 10) catch poller.tick_ms;
        test_tick_ms = std.math.clamp(t, 10, poller.tick_ms);
    }
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

// ── Main-thread process & menu state ──

var g_io: std.Io = undefined;
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
/// The model the watched server last had loaded (`sushibar.json`): Load brings it back.
var g_last_model: Str(logic.max_model) = .{};
/// The ready model's id was past `max_model`, so it was not stored and the note said so.
var g_model_too_long = false;
/// The model the running load or unload names; `g_last_model` follows the watched port and may differ.
var g_op_model: Str(logic.max_model) = .{};
/// A server the user's own Stop is ending, so its going away is no news.
var g_stop_port: u16 = 0;
var g_stop_at_ms: i64 = 0;

/// Wakes the main thread to re-render when `test_polls == 0` (production is event-driven, not 1 Hz polled).
fn notifyMain() void {
    if (test_polls == 0 and menu_target != null) {
        objc.msg(void, menu_target, "performSelectorOnMainThread:withObject:waitUntilDone:", .{ objc.sel("refreshNow:"), @as(objc.id, null), false });
    }
}

// ── Port ──

fn writeSaved(p: u16) void {
    var text: [logic.saved_max_bytes]u8 = undefined;
    io_util.makeDirPath(g_io, std.fs.path.dirname(g_saved_path) orelse ".") catch {};
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
    if (p == poller.port()) return;
    // Clear port-scoped shared state before publishing `p` and waking `pollLoop`, so a fast loopback poll for
    // `p` can never be clobbered by this reset after `writeSaved(p)`.
    poller.lock();
    poller.shared.port = 0;
    poller.shared.view = .{ .link = .offline };
    poller.shared.model.set("");
    poller.shared.model_full_len = 0;
    poller.shared.server_version.set("");
    poller.shared.props_latest.set("");
    poller.shared.props_available = false;
    poller.shared.props_command.set("");
    poller.shared.props_url.set("");
    poller.unlock();
    poller.storePort(p);
    invalidateOwnCache();
    poller.pollSoon();
    g_saved.add(p, g_default_port);
    writeSaved(p);
}

/// Forgets an added port; the watch moves to the default port when it was on it. A sushi still listening there
/// stays in the menu as a listener, and the server is never touched.
fn removePort(p: u16) void {
    if (p == g_default_port) return;
    g_saved.remove(p);
    const cur = poller.port();
    const to = logic.afterRemove(cur, p, g_default_port);
    if (to != cur) setPort(to) else writeSaved(cur);
}

// ── Start and Stop ──

/// One record per port, so a server started on port A stays tracked after Start on port B.
fn serverPidPath(buf: []u8, p: u16) []const u8 {
    return std.fmt.bufPrint(buf, "{s}/.sushi/{s}{d}.pid", .{ g_home, logic.server_pid_prefix, p }) catch "";
}

var own_cache_valid = false;
var own_cache_port: u16 = 0;
var own_cache_entry: ?logic.PidEntry = null;

fn invalidateOwnCache() void {
    own_cache_valid = false;
}

fn ownServer() ?logic.PidEntry {
    const p = poller.port();
    if (test_polls == 0 and own_cache_valid and own_cache_port == p) {
        if (own_cache_entry) |e| {
            if (proc.pidExists(e.pid)) return e;
            own_cache_valid = false;
        } else return null;
    }
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const e = proc.verifiedPid(g_io, serverPidPath(&pb, p), poller.server_exe, true);
    own_cache_valid = true;
    own_cache_port = p;
    own_cache_entry = if (e != null and e.?.port == p) e else null;
    return own_cache_entry;
}

/// Servers this process spawned: each is reaped by its own pid, and one that exits before it ever answered is a
/// failed start.
var started: [8]logic.Started = @splat(.{});

fn hasStartingChild() bool {
    for (started) |st| if (st.pid != 0 and !st.answered) return true;
    return false;
}

fn reapStarted() void {
    for (&started) |*st| {
        if (st.pid == 0) continue;
        var status: c_int = 0;
        const got = std.c.waitpid(st.pid, &status, std.c.W.NOHANG);
        // An adopted server another sushibar started is not ours to wait for.
        if (got == -1 and std.posix.errno(got) == .CHILD) {
            st.* = .{};
            invalidateOwnCache();
        }
        if (got != st.pid) continue;
        invalidateOwnCache();
        if (!st.stopping) {
            var nb: [160]u8 = undefined;
            const how = logic.exitOf(@bitCast(status));
            poller.setServerProblem(logic.startedExitNote(&nb, st.*, how), st.port);
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
        const e = proc.verifiedPid(g_io, serverPidPath(&fb, p), poller.server_exe, true) orelse continue;
        started[slot] = .{ .pid = e.pid, .port = p, .answered = true };
        slot += 1;
    }
}

/// The poll thread asks each started server that has not answered, during its process scan, and the menu thread
/// learns the answers here: an exit then reads "exited", on whatever port the server runs.
fn syncProbes() void {
    var pend: [started.len]u16 = undefined;
    const ports = logic.unansweredPorts(&started, &pend);
    poller.lock();
    defer poller.unlock();
    for (poller.shared.answered_ports[0..poller.shared.answered_n]) |ap| logic.markAnswered(&started, ap);
    poller.shared.answered_n = 0;
    poller.shared.probe_n = ports.len;
    @memcpy(poller.shared.probe_ports[0..ports.len], ports);
}

fn markStopping(pid: i32) void {
    for (&started) |*st| if (st.pid == pid) {
        st.stopping = true;
    };
}

fn startServer() void {
    var sb: [poller.path_cap]u8 = undefined;
    const sushi = poller.siblingNow(&sb, poller.server_exe) orelse return poller.setProblem("could not find sushi beside sushibar");
    if (ownServer() != null) return;
    var port_buf: [8:0]u8 = undefined;
    const p = poller.port();
    const port_z = std.fmt.bufPrintSentinel(&port_buf, "{d}", .{p}, 0) catch return;
    const argv_loopback = [_][*:0]const u8{ sushi.ptr, "--serve", "--port", port_z.ptr };
    const argv_lan = [_][*:0]const u8{ sushi.ptr, "--serve", "--port", port_z.ptr, "--host", logic.lan_host };
    const argv: []const [*:0]const u8 = if (g_lan) &argv_lan else &argv_loopback;
    const pid = proc.spawnDetached(argv, false) catch return poller.setProblem("could not start sushi");
    poller.setNote("");
    poller.pollSoon();
    g_watch.started(p);
    for (&started) |*st| if (st.pid == 0) {
        st.* = .{ .pid = pid, .port = p };
        break;
    };
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    proc.writePid(g_io, g_home, serverPidPath(&pb, p), .{ .pid = pid, .port = p }) catch {};
    invalidateOwnCache();
}

/// Load or unload `g_last_model` on the watched server, the only model either button acts on. The button offers
/// neither before the server answers, so a Start never loads.
fn startModelOp(unload: bool) void {
    if (poller.startModelOp(unload, g_last_model.get())) {
        g_op_model = g_last_model;
    }
}

const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };
extern "c" fn localtime_r(t: *const i64, out: *Tm) ?*Tm;

fn stoppedAtNote(buf: []u8, p: u16) []const u8 {
    const t = io_util.nowSecs(g_io);
    var tm: Tm = undefined;
    if (localtime_r(&t, &tm) == null) return logic.stoppedNote(buf, p, 0, 0);
    return logic.stoppedNote(buf, p, @intCast(tm.hour), @intCast(tm.min));
}

const alert_first = 1000;
const alert_third = 1002;

/// Runs the alert and returns which button ended it (`alert_first` is the default one).
fn runAlert(message: []const u8, info: []const u8, ok: [*:0]const u8, cancel: ?[*:0]const u8, third: ?[*:0]const u8, field: objc.id) isize {
    const app = objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{});
    objc.msg(void, app, "activateIgnoringOtherApps:", .{true});
    const alert = objc.new(objc.class("NSAlert"));
    defer objc.release(alert);
    objc.msg(void, alert, "setMessageText:", .{objc.nsText(message)});
    objc.msg(void, alert, "setInformativeText:", .{objc.nsText(info)});
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
    const info = logic.updateConfirmInfo(&i, poller.port(), confirm_model.get());
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
    invalidateOwnCache();
    poller.pollSoon();
    g_stop_pid = pid;
    g_stopping_port = p;
    stopping_until_ms = io_util.nowMs(g_io) + stopping_ms;
    g_stop_port = p;
    g_stop_at_ms = stopping_until_ms;
    g_stop_sent = .{ .pid = pid, .port = p, .deadline_ms = stopping_until_ms };
}

/// SIGKILL, only to the pid the Stop signalled while it is still the sushi on the watched port.
fn forceStop() void {
    const p = poller.port();
    var lbuf: [16]logic.Listener = undefined;
    const ours = ownServer();
    const pid = switch (logic.planStop(if (ours) |e| e.pid else null, p, proc.listeners(&lbuf, poller.server_exe))) {
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
    const p = poller.port();
    var lbuf: [16]logic.Listener = undefined;
    const ours = ownServer();
    switch (logic.planStop(if (ours) |e| e.pid else null, p, proc.listeners(&lbuf, poller.server_exe))) {
        .none => {},
        .now => |pid| {
            markStopping(pid);
            proc.terminate(pid);
            noteStopped(p, pid);
        },
        .confirm => |pid| {
            if (!confirmStop(p, pid)) return;
            var again: [16]logic.Listener = undefined;
            switch (logic.planStop(null, p, proc.listeners(&again, poller.server_exe))) {
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
    cand[cn] = poller.port();
    cn += 1;
    for (cand[0..cn]) |p| {
        if (n == ports.len or std.mem.indexOfScalar(u16, ports[0..n], p) != null) continue;
        var pb: [std.fs.max_path_bytes]u8 = undefined;
        if (proc.verifiedPid(g_io, serverPidPath(&pb, p), poller.server_exe, true) == null) continue;
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
        const e = proc.verifiedPid(g_io, serverPidPath(&pb, p), poller.server_exe, true) orelse continue;
        markStopping(e.pid);
        proc.terminate(e.pid);
    }
    quit();
}

fn quit() void {
    if (proc.verifiedPid(g_io, g_bar_pid_path, poller.bar_exe, false)) |e| {
        if (e.pid == proc.selfPid()) proc.removePid(g_io, g_bar_pid_path);
    }
    objc.msg(void, objc.msg(objc.id, objc.class("NSApplication"), "sharedApplication", .{}), "terminate:", .{@as(objc.id, null)});
}

// ── Menu bar ──

var header_mi: objc.id = null;
var port_mi: objc.id = null;
var lan_mi: objc.id = null;
var port_menu: objc.id = null;
var open_mis: [4]objc.id = .{ null, null, null, null };
var button_action: logic.ButtonAction = .none;
var pack_mi: objc.id = null;
var g_watch: logic.Watch = .{};
/// What was copied to the clipboard last, and when: the button and the pack item say Copied for a few seconds.
var copied_what: enum { none, upgrade, pack } = .none;
var copied_at_ms: i64 = 0;
/// Set by `refresh` for the Update button's click: the brew command it copies, and whether an alert comes first.
var button_command: Str(80) = .{};
var confirm_needed = false;
var confirm_model: Str(96) = .{};
var confirm_latest: Str(32) = .{};
var shown_note_sig: ?u64 = null;
var shown_port_sig: ?u64 = null;
var menu_target: objc.id = null;
var top_menu: objc.id = null;
var port_field_mi: objc.id = null;
var note_mi: objc.id = null;

var ui_timer: objc.id = null;
var log_timer: objc.id = null;
var last_link: ?logic.Link = null;
var tick_count: u32 = 0;
var readout_buf: [2 * logic.readout_max + 8]u8 = undefined;

var why_mi: objc.id = null;
var shown_why: []const u8 = "";
/// What the Load/Unload button does at the next click, from the last refresh.
var model_action: logic.ModelAction = .none;

fn setText(setter_target: objc.id, comptime selector: [*:0]const u8, text: []const u8) void {
    objc.msg(void, setter_target, selector, .{objc.nsText(text)});
}

fn symbol(name: []const u8) objc.id {
    return objc.msg(objc.id, objc.class("NSImage"), "imageWithSystemSymbolName:accessibilityDescription:", .{ objc.nsText(name), @as(objc.id, null) });
}

fn newItem(label: [*:0]const u8, action: ?[*:0]const u8, tag: isize, key: [*:0]const u8, sym: ?[]const u8) objc.id {
    const mi = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSMenuItem"), "alloc", .{}), "initWithTitle:action:keyEquivalent:", .{
        objc.nsString(label), if (action) |a| objc.selDyn(a) else null, objc.nsString(key),
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
    var w = std.hash.Wyhash.init(0);
    var lb: [96]u8 = undefined;
    for (entries) |e| {
        w.update(logic.portLabel(&lb, e));
        w.update(std.mem.asBytes(&.{ @intFromBool(e.checked), @intFromBool(e.removable), @as(u8, 0) }));
    }
    const sig = w.final();
    if (shown_port_sig == sig) return;
    shown_port_sig = sig;
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
        poller.setServerProblem(stoppedAtNote(&nb, p), p);
    }
    if (link == .offline and mine) g_stop_port = 0;
}

/// `pid` is still a sushi process (a zombie is not: its children are reaped before this is asked).
fn sushiAlive(pid: i32) bool {
    if (pid <= 0) return false;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const exe = proc.exeOfPid(pid, &buf) orelse return false;
    return std.mem.eql(u8, std.fs.path.basename(exe), poller.server_exe);
}

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
    var w = std.hash.Wyhash.init(0);
    w.update(title);
    w.update("\x00");
    w.update(full);
    w.update(std.mem.asBytes(&@intFromBool(problem)));
    const sig = w.final();
    if (shown_note_sig == sig) return;
    shown_note_sig = sig;
    objc.msg(void, note_mi, "setHidden:", .{title.len == 0});
    if (title.len == 0) return;
    setText(note_mi, "setTitle:", title);
    setText(note_mi, "setToolTip:", full);
    objc.msg(void, note_mi, "setEnabled:", .{problem});
    objc.msg(void, note_mi, "setAction:", .{if (problem) objc.sel("showLog:") else null});
    if (!problem) return objc.msg(void, note_mi, "setAttributedTitle:", .{@as(objc.id, null)});
    const attrs = header_view.noteAttributes();
    const text = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSAttributedString"), "alloc", .{}), "initWithString:attributes:", .{ objc.nsText(title), attrs });
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
        .models => poller.refreshModels(@intCast(act.value), g_default_port, &g_saved),
        .check => {
            poller.lock();
            poller.shared.manual_result = .none;
            poller.unlock();
            poller.startWorker(.check);
        },
        .update => poller.startWorker(.update),
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

fn syncTimers(now: i64) void {
    const open_now = poller.menu_open.load(.acquire);
    const copied_active = copied_what != .none and logic.copiedNow(copied_at_ms, now);
    const need_ui = test_polls > 0 or open_now or stopping_until_ms > now or copied_active or hasStartingChild();
    if (need_ui and ui_timer == null) {
        const secs = @as(f64, @floatFromInt(if (test_polls > 0) test_tick_ms else poller.tick_ms)) / 1000.0;
        ui_timer = objc.addTimer(secs, menu_target, "tick:");
    } else if (!need_ui and ui_timer != null) {
        objc.msg(void, ui_timer, "invalidate", .{});
        ui_timer = null;
    }
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
    const p = poller.port();
    const now = io_util.nowMs(g_io);
    var model_b: [logic.max_model]u8 = undefined;
    var loading_b: [96]u8 = undefined;
    var bar_b: [32]u8 = undefined;
    var srv_b: [32]u8 = undefined;
    var latest_b: [32]u8 = undefined;
    var cmd_b: [80]u8 = undefined;
    var note_b: [poller.note_cap]u8 = undefined;
    var listings: [poller.max_listings]logic.Listing = undefined;
    var list_models: [poller.max_listings][96]u8 = undefined;
    var hist: [200]logic.Hist = undefined;

    poller.lock();
    const sh = &poller.shared;
    const fresh = sh.port == p;
    const view: logic.View = if (fresh) sh.view else .{ .link = .offline };
    const model = if (fresh) take(&model_b, &sh.model) else "";
    const loading_id = if (fresh) take(&loading_b, &sh.model_loading) else "";
    var pack_cur_b: [32]u8 = undefined;
    var pack_new_b: [32]u8 = undefined;
    var pack_cmd_b: [96]u8 = undefined;
    const pack_update: ?logic.PackUpdate = if (fresh and sh.pack_latest.len > 0) .{
        .current = take(&pack_cur_b, &sh.pack_current),
        .latest = take(&pack_new_b, &sh.pack_latest),
        .command = take(&pack_cmd_b, &sh.pack_command),
    } else null;
    var ln: usize = 0;
    for (sh.listings[0..sh.listing_n]) |l| {
        var known = l.known;
        var name: []const u8 = l.model.get();
        if (l.port != p) {
            known = .unknown;
            name = "";
            for (&sh.models) |*c| if (c.port == l.port) {
                known = c.known;
                name = c.model.get();
            };
        }
        @memcpy(list_models[ln][0..name.len], name);
        const m = list_models[ln][0..name.len];
        listings[ln] = .{ .port = l.port, .pid = l.pid, .model = if (m.len > 0) m else null, .models = known };
        ln += 1;
    }
    const sushi_listening = fresh and sh.sushi_listening;
    const port_taken = fresh and sh.port_taken;
    const mem_mb = if (fresh) sh.mem_mb else null;
    const gpu_mhz = if (fresh) sh.gpu_mhz else null;
    const gpu_pct = if (fresh) sh.gpu_pct else null;
    const lan_bound = fresh and sh.lan_listening;
    const last_prefill = if (fresh) sh.last_prefill else null;
    const hist_n = if (fresh) sh.hist_n else 0;
    @memcpy(hist[0..hist_n], sh.hist[0..hist_n]);
    const bar_version = take(&bar_b, &sh.bar_version);
    const server_version = if (fresh) take(&srv_b, &sh.server_version) else "";
    const server_command = sh.props_command.len > 0;
    const cmd_len = sh.props_command.len;
    @memcpy(cmd_b[0..cmd_len], sh.props_command.get());
    const uv_base: logic.UpdateView = .{
        .bar_version = bar_version,
        .server_version = server_version,
        .latest = take(&latest_b, &sh.latest),
        .available = sh.available,
        .command = if (g_sibling_brew and cmd_len == 0) "brew upgrade sushi" else cmd_b[0..cmd_len],
        .note = take(&note_b, &sh.note),
        .busy = sh.busy,
        .checked = sh.manual_result,
    };
    const relaunch = sh.relaunch;
    const note_problem = sh.note_problem;
    const model_full_len = if (fresh) sh.model_full_len else 0;
    poller.unlock();

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
            poller.setProblem(logic.modelTooLongNote(&nb, p));
        },
        .keep => if (model.len > 0) {
            g_model_too_long = false;
        },
    }
    watchTransition(p, fresh, view.link, now);
    const recovered = g_watch.takeRecovered();
    if (recovered != 0) {
        poller.lock();
        if (logic.recoveryClearsNote(poller.shared.note_kind, poller.shared.note_port, recovered)) poller.clearNote();
        poller.unlock();
    }
    if (last_link != view.link) {
        last_link = view.link;
        invalidateOwnCache();
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
        poller.setServerProblem(logic.stuckNote(&sb, p), p);
    }
    g_stuck_noted = g_stop_stuck;
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

    // Between Load and the model's row turning to loading, the model Load asked for is already the one loading.
    const asked = poller.load_port.load(.acquire) == p and view.link == .up and model.len == 0 and loading_id.len == 0;
    const loading_name = if (loading_id.len > 0) loading_id else if (asked) g_last_model.get() else "";

    // Always update the visible menu bar button (icon, readout, badge, update dot, tooltip).
    var tv = view;
    tv.update_waiting = uv_base.available;
    const shown = logic.readout(&readout_buf, tv);
    const button = main_bar.button();
    const icon = logic.iconFor(view.link, .{ .ready = model.len > 0, .loading = loading_name.len > 0 });
    if (icon == .ready and view.read != null) {
        bar_button.showReading(button, logic.readingStep(view.read.?.fraction));
    } else {
        bar_button.showIcon(button, icon);
        bar_button.setGeneratingBlink(icon == .ready and view.decoding);
    }
    bar_button.showBadge(button, logic.badge(view.link, port_taken, g_watch.crashed(p) and !ours_alive));
    bar_button.setReadout(button, shown);
    bar_button.showUpdateDot(button, uv_base.available);
    var status_buf: [200]u8 = undefined;
    const status = logic.statusLine(&status_buf, state);
    bar_button.setToolTip(button, status);

    if (relaunch) relaunchSelf(p);
    syncTimers(now);

    // Skip all menu-interior AppKit updates while the menu is closed (rebuilt right before opening in menuWillOpen:).
    const need_menu = poller.menu_open.load(.acquire) or test_polls > 0;
    if (!need_menu) return;

    var started_b: [started.len + 1]u16 = undefined;
    g_started_count = startedPorts(&started_b).len;
    applyQuitLook();
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
    const spark_now = @divFloor(now, 2000) * 2000;
    header_view.update(header, logic.sparkPoints(&spark, hist[0..hist_n], spark_now, header_view.spark_window_ms));

    button_row.setStart(.{ .label = tg.label, .symbol = tg.symbol, .tip = tg.tip, .enabled = tg.enabled, .closes_menu = tg.action != .start });
    var model_tip: [160]u8 = undefined;
    const load_p = poller.load_port.load(.acquire);
    const unload_p = poller.unload_port.load(.acquire);
    const op_port = if (load_p != 0) load_p else unload_p;
    const mb = logic.modelButton(&model_tip, .{
        .other = if (op_port != 0 and op_port != p) .{ .port = op_port, .unload = unload_p != 0, .model = g_op_model.get() } else .{},
        .link = view.link,
        .connecting = !fresh,
        .starting = starting,
        .model = if (model.len > 0) model else null,
        .last_model = g_last_model.get(),
        .loading = load_p == p or (loading_id.len > 0 and model.len == 0),
        .unloading = unload_p == p,
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

    var entries: [poller.max_listings + logic.max_saved + 2]logic.PortEntry = undefined;
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

    if (test_polls > 0) {
        var ports_buf: [256]u8 = undefined;
        var pn: usize = 0;
        for (port_entries) |e| {
            const piece = std.fmt.bufPrint(ports_buf[pn..], "{s}{s};", .{ logic.portLabel(&lab, e), if (e.checked) "*" else "" }) catch break;
            pn += piece.len;
        }
        var readout_line: [2 * logic.readout_max * 4 + 24]u8 = undefined;
        const readout_text = std.fmt.bufPrint(&readout_line, "{s}|{s}\" update_dot={d}", .{ shown.caption, shown.value, @intFromBool(uv.available) }) catch "";
        var removable_buf: [128]u8 = undefined;
        var rn: usize = 0;
        for (port_entries) |e| if (e.removable) {
            const piece = std.fmt.bufPrint(removable_buf[rn..], "{d},", .{e.port}) catch break;
            rn += piece.len;
        };
        std.debug.print("sushibar: poll {d} readout=\"{s} status=\"{s}\" start={d} stop={d} server_pid={d} icon={s} ports=\"{s}\" version=\"{s}\" note=\"{s}\" problem={d} pill=\"{s}\" line1=\"{s}\" line2=\"{s}\" toggle=\"{s}\" field_invalid={d} pack=\"{s}\" packcopy=\"{s}\" button=\"{s}\" tip=\"{s}\" removable=\"{s}\" badge={d} models_link={d} last_model=\"{s}\" load=\"{s}\" load_on={d} why=\"{s}\" stats=\"{s}|{s}|{s}\" a11y=\"{s}\" lan={d}\n", .{
            tick_count,                 readout_text,          status,                 @intFromBool(on.start),
            @intFromBool(on.stop),      if (own) |e| e.pid else 0,
            @tagName(bar_button.shownIcon().?), ports_buf[0..pn], header.version,      note_shown,
            @intFromBool(note_problem and note_shown.len > 0), header.pill.label, header.line1, header.line2,
            tg.label,                   @intFromBool(port_field.isInvalid()), header.pack_line, pack_tip,
            update_button.label,        update_button.tip,     removable_buf[0..rn],   @intFromBool(bar_button.shownBadge()),
            @intFromBool(header.line1_models), g_last_model.get(), mb.label, @intFromBool(mb.enabled), mb.why,
            header.stats[0].label,      header.stats[1].label, header.stats[2].label,  header.a11y,
            @intFromBool(g_lan),
        });
        if (tick_count >= test_polls) quit();
    }
}

/// The sibling's version changed: run the new sushibar in this process, on the same port.
fn relaunchSelf(p: u16) void {
    poller.lock();
    poller.shared.relaunch = false;
    poller.unlock();
    var port_z: [8:0]u8 = undefined;
    const pz = std.fmt.bufPrintSentinel(&port_z, "{d}", .{p}, 0) catch return;
    var sb: [poller.path_cap]u8 = undefined;
    // The path sushibar was started by, so a Homebrew link lands on the new keg; brew's opt link if that is gone.
    const exe = if (poller.reachable(g_self_exe)) g_self_exe else poller.siblingNow(&sb, poller.bar_exe) orelse g_self_exe;
    proc.execReplace(&.{ exe.ptr, "--port", pz.ptr });
    poller.setProblem("Updated, but sushibar could not restart itself: quit and run it again");
}

// ── ObjC menu actions & notifications ──

fn openUrl(url: []const u8) void {
    const ns_url = objc.msg(objc.id, objc.class("NSURL"), "URLWithString:", .{objc.nsText(url)});
    _ = objc.msg(bool, objc.msg(objc.id, objc.class("NSWorkspace"), "sharedWorkspace", .{}), "openURL:", .{ns_url});
}

fn openWebPage(page: logic.Page) void {
    var origin: [poller.origin_cap]u8 = undefined;
    var url_buf: [poller.url_cap]u8 = undefined;
    openUrl(logic.pageUrl(&url_buf, poller.originFor(&origin, poller.port()), page));
}

fn openPage(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    openWebPage(@enumFromInt(@as(u2, @intCast(objc.msg(isize, sender, "tag", .{})))));
}

fn openModelsAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    openWebPage(.models);
}

fn openReleaseNotes(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    poller.lock();
    var b: [160]u8 = undefined;
    const url = take(&b, &poller.shared.props_url);
    poller.unlock();
    openUrl(logic.releaseNotesUrl(url));
}

/// Start or Stop, whichever the item reads.
fn toggleAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    const ours_alive = ownServer() != null;
    poller.lock();
    const state: logic.Server = .{
        .link = poller.shared.view.link,
        .ours_alive = ours_alive,
        .sushi_found = g_sushi_found,
        .port = poller.port(),
        .sushi_listening = poller.shared.sushi_listening,
        .port_taken = poller.shared.port_taken,
        .stop_stuck = g_stop_stuck,
    };
    poller.unlock();
    switch (logic.toggle(state, stopping_until_ms > io_util.nowMs(g_io) and g_stopping_port == poller.port()).action) {
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
    writeSaved(poller.port());
    var nb: [128]u8 = undefined;
    poller.setNote(logic.lanNote(&nb, g_lan));
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
    poller.lock();
    const n = poller.shared.pack_command.len;
    @memcpy(cmd[0..n], poller.shared.pack_command.get());
    poller.unlock();
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
    if (opening != top_menu) return poller.refreshModels(null, g_default_port, &g_saved);
    poller.menu_open.store(true, .release);
    invalidateOwnCache();
    poller.wake.set(g_io);
    g_default_port = readDefaultPort();
    const now = io_util.nowMs(g_io);
    poller.lock();
    poller.shared.manual_result = logic.shownResult(poller.shared.manual_result, poller.shared.manual_at_ms, now);
    if (!logic.noteKept(poller.shared.note_problem, poller.shared.note_at_ms, now)) poller.shared.note.set("");
    poller.unlock();
    g_option = false;
    flagsTick(null, null, null);
    if (g_flags_timer == null) g_flags_timer = objc.addTimer(0.1, menu_target, "flagsTick:");
    refresh();
}

fn menuDidClose(_: objc.id, _: objc.SEL, closing: objc.id) callconv(.c) void {
    if (closing != top_menu) return;
    poller.menu_open.store(false, .release);
    main_bar.setHighlight(false);
    if (g_flags_timer != null) objc.msg(void, g_flags_timer, "invalidate", .{});
    g_flags_timer = null;
    syncTimers(io_util.nowMs(g_io));
}

fn popupMenuAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    main_bar.popupMenu();
}

fn screensChanged(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    if (main_bar.syncScreens()) {
        bar_button.attach(main_bar.button());
        refresh();
    } else if (main_bar.mode() == .main_window) {
        bar_button.realign(main_bar.button());
    }
}

fn spaceChanged(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    if (main_bar.reposition()) bar_button.realign(main_bar.button());
}

fn accessibilityChanged(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    bar_button.syncAnimation();
}

fn refreshNow(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    refresh();
}

/// The watched port's log: what its server's argv said, or the default file while that is not known yet.
fn watchedLog(buf: *[1024]u8) logic.LogSource {
    const p = poller.port();
    poller.lock();
    defer poller.unlock();
    if (poller.shared.port != p) return logic.logSource(buf, g_home, p, null);
    const path = poller.shared.log_path.get();
    @memcpy(buf[0..path.len], path);
    return .{ .kind = poller.shared.log_kind, .path = buf[0..path.len] };
}

fn stopLogTimer() void {
    if (log_timer != null) objc.msg(void, log_timer, "invalidate", .{});
    log_timer = null;
}

fn onLogClosed() void {
    stopLogTimer();
}

fn showLogAction(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    var buf: [1024]u8 = undefined;
    log_view.open(poller.port(), watchedLog(&buf));
    if (log_timer == null) log_timer = objc.addTimer(0.5, menu_target, "logTick:");
}

/// Follows the log while its window is open, and stops itself once the window closes.
fn logTick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    if (!log_view.isVisible()) return stopLogTimer();
    var buf: [1024]u8 = undefined;
    log_view.sync(poller.port(), watchedLog(&buf));
}

// ── App & menu construction ──

const find_show_interface: isize = 1;

fn addMainItem(menu: objc.id, title: [*:0]const u8, comptime action: [*:0]const u8, key: [*:0]const u8) objc.id {
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
    bar_button.loadIcons();

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
        .{ .selector = "popupMenu:", .imp = @ptrCast(&popupMenuAction), .types = "v@:@" },
        .{ .selector = "screensChanged:", .imp = @ptrCast(&screensChanged), .types = "v@:@" },
        .{ .selector = "spaceChanged:", .imp = @ptrCast(&spaceChanged), .types = "v@:@" },
        .{ .selector = "accessibilityChanged:", .imp = @ptrCast(&accessibilityChanged), .types = "v@:@" },
    });
    menu_target = objc.new(cls);
    setMainMenu(app);
    const menu = objc.new(objc.class("NSMenu"));
    top_menu = menu;
    objc.msg(void, menu, "setDelegate:", .{menu_target});
    objc.msg(void, menu, "setAutoenablesItems:", .{false});

    const button = main_bar.init(menu_target, menu);
    bar_button.attach(button);
    log_view.setOnClose(&onLogClosed);

    const nc = objc.msg(objc.id, objc.class("NSNotificationCenter"), "defaultCenter", .{});
    objc.msg(void, nc, "addObserver:selector:name:object:", .{
        menu_target,
        objc.sel("screensChanged:"),
        objc.nsString("NSApplicationDidChangeScreenParametersNotification"),
        @as(objc.id, null),
    });
    const ws_nc = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSWorkspace"), "sharedWorkspace", .{}), "notificationCenter", .{});
    for ([_][*:0]const u8{
        "NSWorkspaceActiveSpaceDidChangeNotification",
        "NSWorkspaceDidActivateApplicationNotification",
        "NSWorkspaceDidLaunchApplicationNotification",
        "NSWorkspaceDidTerminateApplicationNotification",
    }) |notif| {
        objc.msg(void, ws_nc, "addObserver:selector:name:object:", .{
            menu_target,
            objc.sel("spaceChanged:"),
            objc.nsString(notif),
            @as(objc.id, null),
        });
    }
    objc.msg(void, ws_nc, "addObserver:selector:name:object:", .{
        menu_target,
        objc.sel("accessibilityChanged:"),
        objc.nsString("NSWorkspaceAccessibilityDisplayOptionsDidChangeNotification"),
        @as(objc.id, null),
    });

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
    syncTimers(io_util.nowMs(g_io));

    poller.startPollThread() catch fail("cannot start the poll thread", .{});
    objc.msg(void, app, "run", .{});
    std.process.exit(0);
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
        if (proc.verifiedPid(g_io, g_bar_pid_path, poller.bar_exe, false)) |e| {
            if (e.pid == child) {
                out(g_io, "sushibar started (pid {d}), watching port {d}\n", .{ child, poller.port() });
                std.process.exit(0);
            }
            out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
            std.process.exit(0);
        }
        if (std.c.waitpid(child, null, std.c.W.NOHANG) == child) {
            // A copy that lost the race for the single-instance lock exits at once; the winner's pid file explains it.
            if (proc.verifiedPid(g_io, g_bar_pid_path, poller.bar_exe, false)) |e| {
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
    while (it.next()) |a| {
        try list.ensureUnusedCapacity(gpa, 1);
        list.appendAssumeCapacity(try gpa.dupe(u8, a));
    }
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
    if (env("SUSHIBAR_RENDER_HEADER")) |dir| render_gallery.run(g_io, gpa, dir);
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
    poller.storePort(logic.choosePort(explicit, saved.port, g_default_port));

    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe = selfPath(&exe_buf) orelse fail("cannot resolve its own path", .{});
    g_self_exe = try gpa.dupeSentinel(u8, self_exe, 0);
    if (proc.verifiedPid(g_io, g_bar_pid_path, poller.bar_exe, false)) |e| {
        if (e.pid != proc.selfPid()) {
            out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
            return;
        }
    }
    if (!proc.isSessionLeader()) detach(gpa, args, g_self_exe);

    // The pid file is checked and written apart; the lock is what keeps two simultaneous launches to one.
    var lock_buf: [std.fs.max_path_bytes]u8 = undefined;
    const lock_path = std.fmt.bufPrintSentinel(&lock_buf, "{s}/.sushi/{s}", .{ g_home, lock_name }, 0) catch fail("HOME is too long", .{});
    io_util.makeDirPath(g_io, std.fs.path.dirname(lock_path) orelse ".") catch {};
    if (!proc.acquireLock(lock_path.ptr)) {
        out(g_io, "sushibar is already running\n", .{});
        return;
    }
    try proc.writePid(g_io, g_home, g_bar_pid_path, .{ .pid = proc.selfPid(), .port = poller.port() });
    poller.init(.{
        .io = g_io,
        .home = g_home,
        .self_exe = g_self_exe,
        .test_polls = test_polls,
        .test_tick_ms = test_tick_ms,
        .notify_main = &notifyMain,
    });
    var sb: [poller.path_cap]u8 = undefined;
    if (poller.siblingNow(&sb, poller.server_exe)) |sushi| {
        g_sushi_found = true;
        var real: [std.fs.max_path_bytes]u8 = undefined;
        g_sibling_brew = logic.isBrewKeg(proc.realPath(&real, sushi) orelse sushi);
        poller.setSiblingBrew(g_sibling_brew);
    }
    adoptStarted();
    runMenuBar();
}

fn testToolTip(view: objc.id) []const u8 {
    const tip = objc.msg(objc.id, view, "toolTip", .{});
    if (tip == null) return "";
    return std.mem.span(objc.msg([*:0]const u8, tip, "UTF8String", .{}));
}

test "the note of a refused load keeps its whole text in the tooltip, a character across the old 255-byte cut included" {
    const pad: [183]u8 = @splat('x');
    var msg: [poller.note_cap]u8 = undefined;
    const full = try std.fmt.bufPrint(&msg, "Port 8080: could not load Pack-{s}: Not enough memory to load this model.\u{2014} It needs 20.0 GB, and only 12.0 GB of RAM is free right now. To load it:\n- Quit other apps or unload another model to free about 8.0 GB, then retry.\n- Use a smaller pack of this model.", .{&pad});
    try std.testing.expect(full.len > 300 and std.mem.indexOf(u8, full, "\u{2014}").? < 255 and std.mem.indexOf(u8, full, "\u{2014}").? + 3 > 255);
    note_mi = objc.new(objc.class("NSMenuItem"));
    defer {
        objc.release(note_mi);
        note_mi = null;
        shown_note_sig = null;
    }
    var fit_b: [logic.note_title_max + 4]u8 = undefined;
    showNote(logic.fitNote(&fit_b, full, logic.note_title_max), full, true);
    try std.testing.expectEqualStrings(full, testToolTip(note_mi));
}

test "the Load button tooltip keeps a long model name whole" {
    const row = button_row.createServerRow();
    defer objc.release(row);
    const name: [150]u8 = @splat('m');
    var tip_b: [200]u8 = undefined;
    const tip = try std.fmt.bufPrint(&tip_b, "Load {s}\u{2014}é", .{&name});
    button_row.setLoad(.{ .label = "Load", .symbol = "arrow.down.circle", .tip = tip, .closes_menu = false });
    const load = objc.msg(objc.id, objc.msg(objc.id, row, "subviews", .{}), "objectAtIndex:", .{@as(usize, 1)});
    try std.testing.expectEqualStrings(tip, testToolTip(load));
}

test "objc: comptime sel and class caches stay distinct across names" {
    try std.testing.expect(objc.sel("alloc") != null);
    try std.testing.expect(objc.sel("init") != null);
    try std.testing.expect(objc.sel("alloc") != objc.sel("init"));
    try std.testing.expectEqual(objc.sel("alloc"), objc.selDyn("alloc"));
    try std.testing.expect(objc.class("NSObject") != null);
    try std.testing.expect(objc.class("NSString") != null);
    try std.testing.expect(objc.class("NSObject") != objc.class("NSString"));
}

test "a server's error body gives its error.message, and anything else gives nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("no room", poller.errorMessage(a, "{\"error\":{\"message\":\"no room\",\"type\":\"x\"}}"));
    try std.testing.expectEqualStrings("", poller.errorMessage(a, "{\"error\":{\"type\":\"x\"}}"));
    try std.testing.expectEqualStrings("", poller.errorMessage(a, "{\"error\":null}"));
    try std.testing.expectEqualStrings("", poller.errorMessage(a, "{\"error\":\"no room\"}"));
    try std.testing.expectEqualStrings("", poller.errorMessage(a, "{}"));
    try std.testing.expectEqualStrings("", poller.errorMessage(a, "not json"));
}
