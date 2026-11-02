//! The decisions of `sushibar` that need no AppKit: arguments, the title and menu a server state earns, the
//! metrics reading, pid-file verdicts. MLX-free, so `zig build test-host` covers it.
const std = @import("std");
const builtin = @import("builtin");

pub const default_port = @import("../default_port.zig").port;

pub const Reason = enum { unknown_argument, missing_value, bad_port };
pub const Command = union(enum) {
    run: u16,
    help,
    refused: struct { arg: []const u8, reason: Reason },
};

pub fn parseArgs(args: []const []const u8) Command {
    var port: u16 = default_port;
    var refusal: ?Command = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) return .help;
        if (refusal != null) continue;
        if (std.mem.eql(u8, a, "--port")) {
            if (i + 1 >= args.len) {
                refusal = .{ .refused = .{ .arg = a, .reason = .missing_value } };
                continue;
            }
            i += 1;
            port = parsePort(args[i]) orelse {
                refusal = .{ .refused = .{ .arg = a, .reason = .bad_port } };
                continue;
            };
        } else if (std.mem.startsWith(u8, a, "--port=")) {
            port = parsePort(a["--port=".len..]) orelse {
                refusal = .{ .refused = .{ .arg = "--port", .reason = .bad_port } };
                continue;
            };
        } else refusal = .{ .refused = .{ .arg = a, .reason = .unknown_argument } };
    }
    return refusal orelse .{ .run = port };
}

fn parsePort(text: []const u8) ?u16 {
    const n = std.fmt.parseInt(u16, text, 10) catch return null;
    return if (n == 0) null else n;
}

/// An ssh login has no window server to draw a menu bar item in.
pub fn guiSession(ssh_connection: ?[]const u8, ssh_tty: ?[]const u8) bool {
    const none: []const u8 = "";
    return (ssh_connection orelse none).len == 0 and (ssh_tty orelse none).len == 0;
}

// ── What the server says ─────────────────────────────────────────────────

pub const Link = enum {
    offline,
    /// 401 or 403: `--api-key --api-key-strict` keeps the feeds from us.
    unauthorized,
    /// 503: the server runs with `--no-metrics`.
    metrics_off,
    up,
};

pub const Row = struct { request_id: u64, generated: u64 };

/// One `/metrics.json?since=` answer, reduced to what the title needs.
pub const Sample = struct {
    link: Link = .offline,
    decoding: []const Row = &.{},
    prefilling: bool = false,
    /// The newest finished request that generated tokens: `generated_tokens / decode_seconds`.
    finished_tps: ?f64 = null,
    newest_finished_ms: i64 = 0,
};

const Wire = struct {
    gauges: struct { requests_prefilling: u64 = 0 } = .{},
    sessions: []const struct { request_id: u64 = 0, phase: []const u8 = "", generated_tokens: u64 = 0 } = &.{},
    requests: []const struct { t_ms: i64 = 0, generated_tokens: u64 = 0, decode_seconds: f64 = 0 } = &.{},
};

pub fn parseMetrics(arena: std.mem.Allocator, body: []const u8) !Sample {
    const wire = try std.json.parseFromSliceLeaky(Wire, arena, body, .{ .ignore_unknown_fields = true });
    var out: Sample = .{ .link = .up, .prefilling = wire.gauges.requests_prefilling > 0 };
    var rows: std.ArrayList(Row) = .empty;
    for (wire.sessions) |s| {
        if (std.mem.eql(u8, s.phase, "decode")) try rows.append(arena, .{ .request_id = s.request_id, .generated = s.generated_tokens });
        if (std.mem.eql(u8, s.phase, "prefill")) out.prefilling = true;
    }
    out.decoding = rows.items;
    var newest_speed_ms: i64 = std.math.minInt(i64);
    for (wire.requests) |r| {
        out.newest_finished_ms = @max(out.newest_finished_ms, r.t_ms);
        if (r.generated_tokens == 0 or r.decode_seconds <= 0 or r.t_ms < newest_speed_ms) continue;
        newest_speed_ms = r.t_ms;
        out.finished_tps = @as(f64, @floatFromInt(r.generated_tokens)) / r.decode_seconds;
    }
    return out;
}

/// The id of the first model `/v1/models` lists as `ready`.
pub fn parseReadyModel(arena: std.mem.Allocator, body: []const u8) !?[]const u8 {
    const List = struct { data: []const struct { id: []const u8 = "", state: []const u8 = "" } = &.{} };
    const list = try std.json.parseFromSliceLeaky(List, arena, body, .{ .ignore_unknown_fields = true });
    for (list.data) |m| if (std.mem.eql(u8, m.state, "ready")) return m.id;
    return null;
}

pub const View = struct {
    link: Link,
    decoding: bool = false,
    prefilling: bool = false,
    live: ?f64 = null,
    last: ?f64 = null,
};

/// Turns polls into the live decode speed (tokens added to the decoding rows over the wall time between two
/// polls) and remembers the last finished request's speed.
pub const Tracker = struct {
    prev: [32]Row = undefined,
    prev_n: usize = 0,
    prev_ms: i64 = 0,
    since_ms: i64 = 0,
    decoding: bool = false,
    prefilling: bool = false,
    live: ?f64 = null,
    last: ?f64 = null,

    /// `now_ms` is read before the request went out, so the next `since` cannot skip a request.
    pub fn observe(self: *Tracker, now_ms: i64, sample: Sample) void {
        self.live = null;
        self.decoding = sample.decoding.len > 0;
        self.prefilling = sample.prefilling;
        if (sample.link == .up) {
            self.since_ms = @max(sample.newest_finished_ms, now_ms);
            if (sample.finished_tps) |t| self.last = t;
            const dt_ms = now_ms - self.prev_ms;
            if (self.prev_n > 0 and dt_ms > 0) {
                var added: u64 = 0;
                var matched = false;
                for (sample.decoding) |row| for (self.prev[0..self.prev_n]) |p| {
                    if (p.request_id != row.request_id or row.generated < p.generated) continue;
                    added += row.generated - p.generated;
                    matched = true;
                };
                if (matched) self.live = @as(f64, @floatFromInt(added)) * 1000.0 / @as(f64, @floatFromInt(dt_ms));
            }
        }
        self.prev_n = @min(sample.decoding.len, self.prev.len);
        @memcpy(self.prev[0..self.prev_n], sample.decoding[0..self.prev_n]);
        self.prev_ms = now_ms;
    }

    pub fn view(self: *const Tracker, link: Link) View {
        return .{ .link = link, .decoding = self.decoding, .prefilling = self.prefilling, .live = self.live, .last = self.last };
    }
};

fn speedText(buf: []u8, tps: f64) []const u8 {
    return (if (tps >= 100)
        std.fmt.bufPrint(buf, "{d:.0}", .{tps})
    else
        std.fmt.bufPrint(buf, "{d:.1}", .{tps})) catch "?";
}

/// The menu bar title.
pub fn title(buf: []u8, v: View) []const u8 {
    var num: [32]u8 = undefined;
    const text = switch (v.link) {
        .offline => std.fmt.bufPrint(buf, "", .{}),
        .unauthorized => std.fmt.bufPrint(buf, "key needed", .{}),
        .metrics_off => std.fmt.bufPrint(buf, "metrics off", .{}),
        .up => if (v.decoding)
            std.fmt.bufPrint(buf, "{s} tok/s", .{if (v.live) |t| speedText(&num, t) else "…"})
        else if (v.prefilling)
            std.fmt.bufPrint(buf, "prefill…", .{})
        else if (v.last) |t|
            std.fmt.bufPrint(buf, "last {s} tok/s", .{speedText(&num, t)})
        else
            std.fmt.bufPrint(buf, "", .{}),
    };
    return text catch "";
}

pub const Icon = enum { outline, color };

/// Outline while no server answers; colour once one does, whatever it says.
pub fn iconFor(link: Link) Icon {
    return if (link == .offline) .outline else .color;
}

/// The HTTP status of a poll, or null when nothing answered in time.
pub fn linkFor(status: ?u16) Link {
    return switch (status orelse return .offline) {
        200 => .up,
        401, 403 => .unauthorized,
        503 => .metrics_off,
        else => .offline,
    };
}

// ── Menu ─────────────────────────────────────────────────────────────────

pub const Page = enum {
    dashboard,
    chat,
    models,
    settings,

    fn path(p: Page) []const u8 {
        return switch (p) {
            .dashboard => "/",
            .chat => "/chat",
            .models => "/models",
            .settings => "/settings",
        };
    }
};

pub fn pageUrl(buf: []u8, port: u16, page: Page) []const u8 {
    return std.fmt.bufPrint(buf, "http://127.0.0.1:{d}{s}", .{ port, page.path() }) catch "";
}

pub const Server = struct {
    link: Link,
    model: ?[]const u8 = null,
    /// A server sushibar started is alive, verified by its pid file.
    ours_alive: bool = false,
    sushi_found: bool = true,
};

pub fn statusLine(buf: []u8, s: Server) []const u8 {
    const text = switch (s.link) {
        .offline => if (s.ours_alive)
            std.fmt.bufPrint(buf, "Server: starting", .{})
        else if (!s.sushi_found)
            std.fmt.bufPrint(buf, "Server: not running (sushi not found beside sushibar)", .{})
        else
            std.fmt.bufPrint(buf, "Server: not running", .{}),
        .unauthorized => std.fmt.bufPrint(buf, "Server: running (API key required)", .{}),
        .metrics_off => std.fmt.bufPrint(buf, "Server: running, metrics off", .{}),
        .up => if (s.model) |m|
            std.fmt.bufPrint(buf, "Server: {s}", .{m})
        else
            std.fmt.bufPrint(buf, "Server: running, no model loaded", .{}),
    };
    return text catch "Server";
}

pub const Enabled = struct { start: bool, stop: bool };

pub fn enabled(s: Server) Enabled {
    return .{ .start = s.sushi_found and s.link == .offline and !s.ours_alive, .stop = s.ours_alive };
}

// ── Pid files and processes ──────────────────────────────────────────────

pub const PidEntry = struct { pid: i32, port: u16 };

/// A pid file holds `<pid> <port>`.
pub fn parsePidFile(text: []const u8) ?PidEntry {
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    const pid = std.fmt.parseInt(i32, it.next() orelse return null, 10) catch return null;
    const port = std.fmt.parseInt(u16, it.next() orelse return null, 10) catch return null;
    if (pid <= 0 or it.next() != null) return null;
    return .{ .pid = pid, .port = port };
}

pub fn formatPidFile(buf: []u8, e: PidEntry) []const u8 {
    return std.fmt.bufPrint(buf, "{d} {d}\n", .{ e.pid, e.port }) catch "";
}

/// What the process table says about a pid. `exe` is null when the pid is gone.
pub const Probe = struct { exe: ?[]const u8, args: ?[]const u8 = null };

/// A pid file counts only while its pid is still the program that wrote it (pids are reused): `want_exe` names the
/// executable, and a server must still carry `--serve --port <entry.port>`.
pub fn pidAlive(e: PidEntry, probe: Probe, want_exe: []const u8, server: bool) bool {
    const exe = probe.exe orelse return false;
    if (!std.mem.eql(u8, std.fs.path.basename(exe), want_exe)) return false;
    return !server or argsServe(probe.args orelse return false, e.port);
}

/// `args` is a KERN_PROCARGS2 buffer: argc, the executable path, NUL padding, then argc NUL-separated strings.
pub fn argsServe(args: []const u8, port: u16) bool {
    if (args.len < 4) return false;
    const argc: usize = @intCast(@max(0, std.mem.readInt(i32, args[0..4], builtin.cpu.arch.endian())));
    var it = std.mem.splitScalar(u8, args[4..], 0);
    _ = it.next();
    while (it.peek()) |s| : (_ = it.next()) if (s.len > 0) break;
    var want: [8]u8 = undefined;
    const port_text = std.fmt.bufPrint(&want, "{d}", .{port}) catch return false;
    var serving = false;
    var port_ok = false;
    var prev: []const u8 = "";
    var seen: usize = 0;
    while (it.next()) |a| : (seen += 1) {
        if (seen >= argc) break;
        if (std.mem.eql(u8, a, "--serve")) serving = true;
        if (std.mem.eql(u8, prev, "--port") and std.mem.eql(u8, a, port_text)) port_ok = true;
        prev = a;
    }
    return serving and port_ok;
}

/// `name` in the directory of the running executable.
pub fn siblingPath(buf: []u8, exe_path: []const u8, name: []const u8) ?[]const u8 {
    const dir = std.fs.path.dirname(exe_path) orelse return null;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name }) catch null;
}

test "args: no argument runs on the server's default port" {
    try std.testing.expectEqual(Command{ .run = default_port }, parseArgs(&.{}));
}

test "args: --port takes a value or an =form, --help wins" {
    try std.testing.expectEqual(Command{ .run = 18801 }, parseArgs(&.{ "--port", "18801" }));
    try std.testing.expectEqual(Command{ .run = 18802 }, parseArgs(&.{"--port=18802"}));
    try std.testing.expectEqual(Command.help, parseArgs(&.{"--help"}));
    try std.testing.expectEqual(Command.help, parseArgs(&.{ "--port", "1", "-h" }));
}

test "args: a bad value or an unknown argument is refused by name" {
    const bad = parseArgs(&.{ "--port", "abc" }).refused;
    try std.testing.expectEqualStrings("--port", bad.arg);
    try std.testing.expectEqual(Reason.bad_port, bad.reason);
    try std.testing.expectEqual(Reason.bad_port, parseArgs(&.{ "--port", "0" }).refused.reason);
    try std.testing.expectEqual(Reason.bad_port, parseArgs(&.{ "--port", "70000" }).refused.reason);
    try std.testing.expectEqual(Reason.missing_value, parseArgs(&.{"--port"}).refused.reason);
    const unknown = parseArgs(&.{"--model"}).refused;
    try std.testing.expectEqualStrings("--model", unknown.arg);
    try std.testing.expectEqual(Reason.unknown_argument, unknown.reason);
    try std.testing.expectEqual(Reason.unknown_argument, parseArgs(&.{"serve"}).refused.reason);
}

test "gui session: an ssh login has none" {
    try std.testing.expect(guiSession(null, null));
    try std.testing.expect(guiSession("", ""));
    try std.testing.expect(!guiSession("10.0.0.2 5000 10.0.0.3 22", null));
    try std.testing.expect(!guiSession(null, "/dev/ttys003"));
}

test "metrics: a decoding row, a prefilling gauge and the newest finished speed" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body =
        \\{"counters":{},"gauges":{"requests_prefilling":0,"x":1},"sessions":[
        \\{"model":"m","request_id":7,"phase":"decode","generated_tokens":40},
        \\{"model":"m","request_id":0,"phase":"cached","generated_tokens":0}],
        \\"history":{"samples":[[1,2]]},
        \\"requests":[{"request_id":5,"t_ms":100,"generated_tokens":10,"decode_seconds":2.0,"outcome":"ok"},
        \\{"request_id":6,"t_ms":200,"generated_tokens":90,"decode_seconds":1,"outcome":"ok"},
        \\{"request_id":8,"t_ms":300,"generated_tokens":0,"decode_seconds":0,"outcome":"rejected"}]}
    ;
    const s = try parseMetrics(arena.allocator(), body);
    try std.testing.expectEqual(Link.up, s.link);
    try std.testing.expectEqual(@as(usize, 1), s.decoding.len);
    try std.testing.expectEqual(@as(u64, 40), s.decoding[0].generated);
    try std.testing.expect(!s.prefilling);
    try std.testing.expectApproxEqAbs(@as(f64, 90), s.finished_tps.?, 1e-9);
    try std.testing.expectEqual(@as(i64, 300), s.newest_finished_ms);
    try std.testing.expect((try parseMetrics(arena.allocator(), "{\"gauges\":{\"requests_prefilling\":1}}")).prefilling);
    try std.testing.expect(std.meta.isError(parseMetrics(arena.allocator(), "not json")));
}

test "models: the first ready model names the server" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body = "{\"object\":\"list\",\"data\":[{\"id\":\"a\",\"state\":\"unloaded\"},{\"id\":\"b\",\"state\":\"ready\"}]}";
    try std.testing.expectEqualStrings("b", (try parseReadyModel(arena.allocator(), body)).?);
    try std.testing.expect((try parseReadyModel(arena.allocator(), "{\"data\":[]}")) == null);
}

test "tracker: live speed is the tokens added to the same requests over the wall time between polls" {
    var t: Tracker = .{};
    t.observe(1000, .{ .link = .up, .decoding = &.{.{ .request_id = 1, .generated = 10 }} });
    try std.testing.expect(t.live == null and t.decoding);
    t.observe(2000, .{ .link = .up, .decoding = &.{.{ .request_id = 1, .generated = 110 }} });
    try std.testing.expectApproxEqAbs(@as(f64, 100), t.live.?, 1e-9);
    t.observe(3500, .{ .link = .up, .decoding = &.{ .{ .request_id = 1, .generated = 140 }, .{ .request_id = 2, .generated = 0 } } });
    try std.testing.expectApproxEqAbs(@as(f64, 20), t.live.?, 1e-9);
    t.observe(4000, .{ .link = .up, .decoding = &.{.{ .request_id = 9, .generated = 5 }} });
    try std.testing.expect(t.live == null);
}

test "tracker: the last speed outlives the request, and since follows the newest row or the poll" {
    var t: Tracker = .{};
    try std.testing.expectEqual(@as(i64, 0), t.since_ms);
    t.observe(5000, .{ .link = .up, .finished_tps = 42, .newest_finished_ms = 4000 });
    try std.testing.expectEqual(@as(i64, 5000), t.since_ms);
    t.observe(6000, .{ .link = .up, .newest_finished_ms = 7000 });
    try std.testing.expectEqual(@as(i64, 7000), t.since_ms);
    try std.testing.expectApproxEqAbs(@as(f64, 42), t.last.?, 1e-9);
    t.observe(8000, .{ .link = .offline });
    try std.testing.expectEqual(@as(i64, 7000), t.since_ms);
    try std.testing.expectApproxEqAbs(@as(f64, 42), t.last.?, 1e-9);
    try std.testing.expect(t.live == null and !t.decoding);
}

test "icon and link: outline only when nothing answers, and a silent server is offline" {
    try std.testing.expectEqual(Icon.outline, iconFor(.offline));
    for ([_]Link{ .up, .unauthorized, .metrics_off }) |l| try std.testing.expectEqual(Icon.color, iconFor(l));
    try std.testing.expectEqual(Link.offline, linkFor(null));
    try std.testing.expectEqual(Link.up, linkFor(200));
    try std.testing.expectEqual(Link.unauthorized, linkFor(401));
    try std.testing.expectEqual(Link.metrics_off, linkFor(503));
    try std.testing.expectEqual(Link.offline, linkFor(404));
}

test "title: live, last, prefill, idle, offline, unauthorized, metrics off" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("103 tok/s", title(&b, .{ .link = .up, .decoding = true, .live = 102.7 }));
    try std.testing.expectEqualStrings("42.7 tok/s", title(&b, .{ .link = .up, .decoding = true, .live = 42.66, .last = 9 }));
    try std.testing.expectEqualStrings("… tok/s", title(&b, .{ .link = .up, .decoding = true }));
    try std.testing.expectEqualStrings("prefill…", title(&b, .{ .link = .up, .prefilling = true, .last = 9 }));
    try std.testing.expectEqualStrings("last 88.0 tok/s", title(&b, .{ .link = .up, .last = 88 }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .up }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .offline, .last = 5, .decoding = true }));
    try std.testing.expectEqualStrings("key needed", title(&b, .{ .link = .unauthorized, .last = 5 }));
    try std.testing.expectEqualStrings("metrics off", title(&b, .{ .link = .metrics_off }));
}

test "menu: Start and Stop follow the server state" {
    var b: [128]u8 = undefined;
    try std.testing.expectEqual(Enabled{ .start = true, .stop = false }, enabled(.{ .link = .offline }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .offline, .sushi_found = false }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .up, .ours_alive = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .offline, .ours_alive = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .up }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .unauthorized }));
    try std.testing.expectEqualStrings("Server: not running", statusLine(&b, .{ .link = .offline }));
    try std.testing.expect(std.mem.indexOf(u8, statusLine(&b, .{ .link = .offline, .sushi_found = false }), "not found") != null);
    try std.testing.expectEqualStrings("Server: Qwen-X", statusLine(&b, .{ .link = .up, .model = "Qwen-X" }));
    try std.testing.expectEqualStrings("Server: running, no model loaded", statusLine(&b, .{ .link = .up }));
    try std.testing.expectEqualStrings("Server: running (API key required)", statusLine(&b, .{ .link = .unauthorized }));
}

test "menu: page urls carry the port" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("http://127.0.0.1:18801/", pageUrl(&b, 18801, .dashboard));
    try std.testing.expectEqualStrings("http://127.0.0.1:18801/chat", pageUrl(&b, 18801, .chat));
    try std.testing.expectEqualStrings("http://127.0.0.1:18801/models", pageUrl(&b, 18801, .models));
    try std.testing.expectEqualStrings("http://127.0.0.1:18801/settings", pageUrl(&b, 18801, .settings));
}

test "pid file: parse, format, and refuse anything else" {
    try std.testing.expectEqual(PidEntry{ .pid = 4242, .port = 18801 }, parsePidFile("4242 18801\n").?);
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("4242 18801\n", formatPidFile(&b, .{ .pid = 4242, .port = 18801 }));
    for ([_][]const u8{ "", "12", "x 5", "0 5", "-3 5", "1 2 3", "1 99999" }) |bad| try std.testing.expect(parsePidFile(bad) == null);
}

fn procargs(comptime argv: []const []const u8) []const u8 {
    comptime var out: []const u8 = &std.mem.toBytes(@as(i32, argv.len)) ++ "/bin/" ++ argv[0] ++ "\x00\x00\x00";
    inline for (argv) |a| out = out ++ a ++ "\x00";
    return out ++ "HOME=/x\x00";
}

test "pid file: a stale, reused or foreign pid is not alive" {
    const e: PidEntry = .{ .pid = 7, .port = 18801 };
    const serve = procargs(&.{ "sushi", "--serve", "--port", "18801" });
    try std.testing.expect(pidAlive(e, .{ .exe = "/opt/s/sushi", .args = serve }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = null }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/usr/bin/vim", .args = serve }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/opt/s/sushi-ubench", .args = serve }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/opt/s/sushi", .args = procargs(&.{ "sushi", "--serve", "--port", "12345" }) }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/opt/s/sushi", .args = procargs(&.{ "sushi", "run", "--port", "18801" }) }, "sushi", true));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/opt/s/sushi" }, "sushi", true));
    try std.testing.expect(pidAlive(e, .{ .exe = "/opt/s/sushibar" }, "sushibar", false));
    try std.testing.expect(!pidAlive(e, .{ .exe = "/opt/s/sushi" }, "sushibar", false));
}

test "sushi is looked up beside the running executable" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("/opt/s/sushi", siblingPath(&b, "/opt/s/sushibar", "sushi").?);
    try std.testing.expect(siblingPath(&b, "sushibar", "sushi") == null);
}
