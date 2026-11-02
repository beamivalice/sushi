//! `sushibar`: the menu bar item for a sushi server. It links no MLX and no server module: the title reads
//! `/metrics.json` over HTTP, Start and Stop run the `sushi` beside this binary. AppKit is called only from the
//! main thread; a poll thread publishes a snapshot that a main-run-loop timer renders.
const std = @import("std");
const logic = @import("sushibar/sushibar_logic.zig");
const proc = @import("sushibar/sushibar_proc.zig");
const objc = @import("sushibar/objc.zig");
const io_util = @import("io_util.zig");

const usage =
    \\usage: sushibar [--port <n>]
    \\
    \\Puts sushi in the menu bar: decode tok/s as the title, and a menu to start and stop a server,
    \\open the dashboard, chat, models and settings pages. It detaches from the terminal.
    \\
    \\  --port <n>   Port of the server to watch and start (default: {d})
    \\  --help       This text
    \\
;

const poll_ms = 1000;
const server_exe = "sushi";
const bar_exe = "sushibar";
const bar_pid_name = "sushibar.pid";
const server_pid_name = "sushibar-server.pid";

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

// ── Test seams: SUSHIBAR_TEST_POLLS, SUSHIBAR_TEST_ACTIONS (docs/envvar.md) ──

const Action = struct { kind: enum { start, stop }, at: u32 };
var test_polls: u32 = 0;
var test_actions: [8]Action = undefined;
var test_action_n: usize = 0;

fn readTestSeams() void {
    test_polls = std.fmt.parseInt(u32, env("SUSHIBAR_TEST_POLLS") orelse "", 10) catch 0;
    var it = std.mem.tokenizeScalar(u8, env("SUSHIBAR_TEST_ACTIONS") orelse "", ',');
    while (it.next()) |item| {
        if (test_action_n == test_actions.len) break;
        const at = std.mem.indexOfScalar(u8, item, '@') orelse continue;
        const kind = std.meta.stringToEnum(@FieldType(Action, "kind"), item[0..at]) orelse continue;
        test_actions[test_action_n] = .{ .kind = kind, .at = std.fmt.parseInt(u32, item[at + 1 ..], 10) catch continue };
        test_action_n += 1;
    }
}

// ── State shared by the poll thread and the main thread ──

const Shared = struct {
    mu: std.Io.Mutex = .init,
    view: logic.View = .{ .link = .offline },
    model_buf: [160]u8 = undefined,
    model_len: usize = 0,
};

var shared: Shared = .{};
var g_io: std.Io = undefined;
var g_port: u16 = logic.default_port;
var g_home: []const u8 = "/tmp";
var g_sushi: ?[:0]const u8 = null;
var g_bar_pid_path: []const u8 = "";
var g_server_pid_path: []const u8 = "";

fn publish(view: logic.View, model: ?[]const u8) void {
    shared.mu.lockUncancelable(g_io);
    defer shared.mu.unlock(g_io);
    shared.view = view;
    shared.model_len = 0;
    if (model) |m| {
        shared.model_len = @min(m.len, shared.model_buf.len);
        @memcpy(shared.model_buf[0..shared.model_len], m[0..shared.model_len]);
    }
}

const Reply = struct { status: u16, body: []const u8 };

const poll_timeout_s = 4;

/// A server that accepts and never answers is offline after `poll_timeout_s`, not a stalled poll thread.
fn get(a: std.mem.Allocator, url: []const u8) ?Reply {
    const U = union(enum) { got: ?Reply, timer: std.Io.Cancelable!void };
    var buf: [2]U = undefined;
    var sel = std.Io.Select(U).init(g_io, &buf);
    sel.concurrent(.got, fetchNow, .{ a, url }) catch return fetchNow(a, url);
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

fn fetchNow(a: std.mem.Allocator, url: []const u8) ?Reply {
    var client: std.http.Client = .{ .allocator = a, .io = g_io };
    defer client.deinit();
    var body: std.Io.Writer.Allocating = .init(a);
    const res = client.fetch(.{ .location = .{ .url = url }, .keep_alive = false, .response_writer = &body.writer }) catch return null;
    return .{ .status = @intFromEnum(res.status), .body = body.written() };
}

fn pollLoop() void {
    var tracker: logic.Tracker = .{};
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    while (true) {
        _ = arena_state.reset(.retain_capacity);
        const a = arena_state.allocator();
        const now = io_util.nowMs(g_io);
        var url_buf: [96]u8 = undefined;
        const url = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/metrics.json?since={d}", .{ g_port, tracker.since_ms }) catch unreachable;
        var sample: logic.Sample = .{};
        const reply = get(a, url);
        sample.link = logic.linkFor(if (reply) |r| r.status else null);
        if (sample.link == .up) sample = logic.parseMetrics(a, reply.?.body) catch .{};
        tracker.observe(now, sample);
        var model: ?[]const u8 = null;
        if (sample.link == .up or sample.link == .metrics_off) {
            const murl = std.fmt.bufPrint(&url_buf, "http://127.0.0.1:{d}/v1/models", .{g_port}) catch unreachable;
            if (get(a, murl)) |r| if (r.status == 200) {
                model = logic.parseReadyModel(a, r.body) catch null;
            };
        }
        publish(tracker.view(sample.link), model);
        std.Io.sleep(g_io, .fromMilliseconds(poll_ms), .real) catch {};
    }
}

// ── Start and Stop ──

fn ownServer() ?logic.PidEntry {
    const e = proc.verifiedPid(g_io, g_server_pid_path, server_exe, true) orelse return null;
    return if (e.port == g_port) e else null;
}

fn startServer() void {
    const sushi = g_sushi orelse return;
    if (ownServer() != null) return;
    var port_buf: [8:0]u8 = undefined;
    const port = std.fmt.bufPrintSentinel(&port_buf, "{d}", .{g_port}, 0) catch return;
    const pid = proc.spawnDetached(&.{ sushi.ptr, "--serve", "--port", port.ptr }, false) catch return;
    proc.writePid(g_io, g_home, g_server_pid_path, .{ .pid = pid, .port = g_port }) catch {};
}

fn stopServer() void {
    if (ownServer()) |e| proc.terminate(e.pid);
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
var icons: [2]objc.id = .{ null, null };
var shown_icon: ?logic.Icon = null;
var tick_count: u32 = 0;
var last_title: [128]u8 = undefined;
var last_title_len: usize = 0;

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

fn tick(_: objc.id, _: objc.SEL, _: objc.id) callconv(.c) void {
    tick_count += 1;
    for (test_actions[0..test_action_n]) |act| if (act.at == tick_count) switch (act.kind) {
        .start => startServer(),
        .stop => stopServer(),
    };
    while (std.c.waitpid(-1, null, std.c.W.NOHANG) > 0) {}

    shared.mu.lockUncancelable(g_io);
    const view = shared.view;
    var model_copy: [160]u8 = undefined;
    const model_len = shared.model_len;
    @memcpy(model_copy[0..model_len], shared.model_buf[0..model_len]);
    shared.mu.unlock(g_io);

    const state: logic.Server = .{
        .link = view.link,
        .model = if (model_len > 0) model_copy[0..model_len] else null,
        .ours_alive = ownServer() != null,
        .sushi_found = g_sushi != null,
    };
    const on = logic.enabled(state);
    var status_buf: [200]u8 = undefined;
    const status = logic.statusLine(&status_buf, state);
    const title = logic.title(&last_title, view);
    last_title_len = title.len;

    const button = objc.msg(objc.id, status_item, "button", .{});
    showIcon(button, logic.iconFor(view.link));
    setText(button, "setTitle:", title);
    setText(status_mi, "setTitle:", status);
    objc.msg(void, start_mi, "setEnabled:", .{on.start});
    objc.msg(void, stop_mi, "setEnabled:", .{on.stop});

    if (test_polls > 0) {
        std.debug.print("sushibar: poll {d} title=\"{s}\" status=\"{s}\" start={d} stop={d} server_pid={d} icon={s}\n", .{
            tick_count, title, status, @intFromBool(on.start), @intFromBool(on.stop), if (ownServer()) |e| e.pid else 0, @tagName(shown_icon.?),
        });
        if (tick_count >= test_polls) quit();
    }
}

fn openPage(_: objc.id, _: objc.SEL, sender: objc.id) callconv(.c) void {
    const tag = objc.msg(isize, sender, "tag", .{});
    var url_buf: [96:0]u8 = undefined;
    const page: logic.Page = @enumFromInt(@as(u2, @intCast(tag)));
    const url = logic.pageUrl(&url_buf, g_port, page);
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

fn addItem(menu: objc.id, target: objc.id, label: [*:0]const u8, action: ?[*:0]const u8, tag: isize) objc.id {
    const mi = objc.msg(objc.id, objc.msg(objc.id, objc.class("NSMenuItem"), "alloc", .{}), "initWithTitle:action:keyEquivalent:", .{
        objc.nsString(label), if (action) |a| objc.sel(a) else null, objc.nsString(""),
    });
    objc.msg(void, mi, "setTarget:", .{target});
    objc.msg(void, mi, "setTag:", .{tag});
    objc.msg(void, mi, "setEnabled:", .{action != null});
    objc.msg(void, menu, "addItem:", .{mi});
    return mi;
}

fn addSeparator(menu: objc.id) void {
    objc.msg(void, menu, "addItem:", .{objc.msg(objc.id, objc.class("NSMenuItem"), "separatorItem", .{})});
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
    });
    const target = objc.new(cls);
    const menu = objc.new(objc.class("NSMenu"));
    objc.msg(void, menu, "setAutoenablesItems:", .{false});
    status_mi = addItem(menu, target, "Server: …", null, 0);
    start_mi = addItem(menu, target, "Start Server", "start:", 0);
    stop_mi = addItem(menu, target, "Stop Server", "stop:", 0);
    addSeparator(menu);
    _ = addItem(menu, target, "Open Dashboard", "open:", @intFromEnum(logic.Page.dashboard));
    _ = addItem(menu, target, "Open Chat", "open:", @intFromEnum(logic.Page.chat));
    _ = addItem(menu, target, "Models", "open:", @intFromEnum(logic.Page.models));
    _ = addItem(menu, target, "Settings", "open:", @intFromEnum(logic.Page.settings));
    addSeparator(menu);
    _ = addItem(menu, target, "Quit", "quit:", 0);
    objc.msg(void, status_item, "setMenu:", .{menu});
    objc.addTimer(@as(f64, poll_ms) / 1000.0, target, "tick:");

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
                out(g_io, "sushibar started (pid {d}), watching port {d}\n", .{ child, g_port });
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
        .run => |port| g_port = port,
    }
    if (!logic.guiSession(env("SSH_CONNECTION"), env("SSH_TTY")))
        fail("no GUI session (this is an ssh login): run sushibar from the Mac's desktop session", .{});

    g_home = env("HOME") orelse "/tmp";
    var bar_path: [std.fs.max_path_bytes]u8 = undefined;
    var server_path: [std.fs.max_path_bytes]u8 = undefined;
    g_bar_pid_path = try gpa.dupe(u8, proc.pidFilePath(&bar_path, g_home, bar_pid_name) orelse fail("HOME is too long", .{}));
    g_server_pid_path = try gpa.dupe(u8, proc.pidFilePath(&server_path, g_home, server_pid_name) orelse fail("HOME is too long", .{}));
    readTestSeams();

    var exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe = selfPath(&exe_buf) orelse fail("cannot resolve its own path", .{});
    if (proc.verifiedPid(g_io, g_bar_pid_path, bar_exe, false)) |e| {
        if (e.pid != proc.selfPid()) {
            out(g_io, "sushibar is already running (pid {d})\n", .{e.pid});
            return;
        }
    }
    if (!proc.isSessionLeader()) detach(gpa, args, self_exe);

    try proc.writePid(g_io, g_home, g_bar_pid_path, .{ .pid = proc.selfPid(), .port = g_port });
    var sibling: [std.fs.max_path_bytes]u8 = undefined;
    if (logic.siblingPath(&sibling, self_exe, server_exe)) |p| {
        if (std.Io.Dir.cwd().access(g_io, p, .{ .execute = true })) |_| g_sushi = try gpa.dupeSentinel(u8, p, 0) else |_| {}
    }
    runMenuBar();
}
