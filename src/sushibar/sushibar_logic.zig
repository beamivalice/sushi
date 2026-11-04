//! The decisions of `sushibar` that need no AppKit: arguments, the title and menu a server state earns, the
//! metrics reading, pid-file verdicts. MLX-free, so `zig build test-host` covers it.
const std = @import("std");
const builtin = @import("builtin");

const default_port_mod = @import("../default_port.zig");
const json_text = @import("../json_text.zig");
pub const default_port = default_port_mod.port;

pub const Reason = enum { unknown_argument, missing_value, bad_port };
pub const Command = union(enum) {
    run: ?u16,
    help,
    refused: struct { arg: []const u8, reason: Reason },
};

pub fn parseArgs(args: []const []const u8) Command {
    var port: ?u16 = null;
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
    /// A sushi listens on the port and does not answer in time.
    silent,
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
    /// The same request's prompt speed: `(prompt_tokens - cached_tokens) / prefill_seconds`.
    finished_prefill_tps: ?f64 = null,
    newest_finished_ms: i64 = 0,
    mem_mb: ?u64 = null,
    gpu_mhz: ?u64 = null,
    gpu_pct: ?u64 = null,
    /// `history` samples newer than the request's `since`.
    hist: []const Hist = &.{},
};

pub const Hist = struct { t_ms: i64, gen: u64 };

const Wire = struct {
    gauges: struct {
        requests_prefilling: u64 = 0,
        memory_mb: u64 = 0,
        gpu_clock_mhz: u64 = 0,
        gpu_utilization_pct: u64 = 0,
    } = .{},
    sessions: []const struct { request_id: u64 = 0, phase: []const u8 = "", generated_tokens: u64 = 0 } = &.{},
    requests: []const struct {
        t_ms: i64 = 0,
        prompt_tokens: u64 = 0,
        cached_tokens: u64 = 0,
        outcome: []const u8 = "",
        prefill_seconds: f64 = 0,
        generated_tokens: u64 = 0,
        decode_seconds: f64 = 0,
    } = &.{},
    history: ?struct { fields: []const []const u8 = &.{}, samples: []const []const f64 = &.{} } = null,
};

fn nonZero(v: u64) ?u64 {
    return if (v == 0) null else v;
}

pub fn parseMetrics(arena: std.mem.Allocator, body: []const u8) !Sample {
    const wire = try std.json.parseFromSliceLeaky(Wire, arena, body, .{ .ignore_unknown_fields = true });
    var out: Sample = .{
        .link = .up,
        .prefilling = wire.gauges.requests_prefilling > 0,
        .mem_mb = nonZero(wire.gauges.memory_mb),
        .gpu_mhz = nonZero(wire.gauges.gpu_clock_mhz),
        .gpu_pct = if (wire.gauges.gpu_clock_mhz > 0 or wire.gauges.gpu_utilization_pct > 0) wire.gauges.gpu_utilization_pct else null,
    };
    var rows: std.ArrayList(Row) = .empty;
    for (wire.sessions) |s| {
        if (std.mem.eql(u8, s.phase, "decode")) try rows.append(arena, .{ .request_id = s.request_id, .generated = s.generated_tokens });
        if (std.mem.eql(u8, s.phase, "prefill")) out.prefilling = true;
    }
    out.decoding = rows.items;
    var newest_speed_ms: i64 = std.math.minInt(i64);
    var newest_prefill_ms: i64 = std.math.minInt(i64);
    for (wire.requests) |r| {
        out.newest_finished_ms = @max(out.newest_finished_ms, r.t_ms);
        const forwarded = r.prompt_tokens -| r.cached_tokens;
        if (std.mem.eql(u8, r.outcome, "ok") and r.t_ms >= newest_prefill_ms and forwarded > 0 and r.prefill_seconds > 0) {
            newest_prefill_ms = r.t_ms;
            out.finished_prefill_tps = @as(f64, @floatFromInt(forwarded)) / r.prefill_seconds;
        }
        if (r.generated_tokens == 0 or r.decode_seconds <= 0 or r.t_ms < newest_speed_ms) continue;
        newest_speed_ms = r.t_ms;
        out.finished_tps = @as(f64, @floatFromInt(r.generated_tokens)) / r.decode_seconds;
    }
    if (wire.history) |h| {
        const ti = fieldIndex(h.fields, "t_ms");
        const gi = fieldIndex(h.fields, "gen_tokens");
        if (ti != null and gi != null) {
            var list: std.ArrayList(Hist) = .empty;
            for (h.samples) |row| {
                if (row.len <= @max(ti.?, gi.?)) continue;
                try list.append(arena, .{ .t_ms = @intFromFloat(row[ti.?]), .gen = @intFromFloat(@max(0, row[gi.?])) });
            }
            out.hist = list.items;
        }
    }
    return out;
}

fn fieldIndex(fields: []const []const u8, name: []const u8) ?usize {
    for (fields, 0..) |f, i| if (std.mem.eql(u8, f, name)) return i;
    return null;
}

/// `meta.pack_update` of a `/v1/models` row: a newer pack revision is published.
pub const PackUpdate = struct { current: []const u8 = "", latest: []const u8 = "", command: []const u8 = "" };

pub const ModelState = struct { ready: ?[]const u8 = null, loading: ?[]const u8 = null, pack_update: ?PackUpdate = null };

/// Which model `/v1/models` lists as ready, and which as loading.
pub fn parseModelState(arena: std.mem.Allocator, body: []const u8) !ModelState {
    const List = struct { data: []const struct {
        id: []const u8 = "",
        state: []const u8 = "",
        meta: struct { pack_update: ?PackUpdate = null } = .{},
    } = &.{} };
    const list = try std.json.parseFromSliceLeaky(List, arena, body, .{ .ignore_unknown_fields = true });
    var out: ModelState = .{};
    for (list.data) |m| {
        if (out.ready == null and std.mem.eql(u8, m.state, "ready")) {
            out.ready = m.id;
            if (m.meta.pack_update) |u| if (u.latest.len > 0 and u.command.len > 0) {
                out.pack_update = u;
            };
        }
        if (out.loading == null and std.mem.eql(u8, m.state, "loading")) out.loading = m.id;
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
    /// The last request ended `title_last_ms` or more ago: the title drops its speed.
    last_old: bool = false,
    update_waiting: bool = false,
};

/// How long the menu bar keeps showing the last request's speed once the server went idle.
pub const title_last_ms: i64 = 10 * 60_000;

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
    last_prefill: ?f64 = null,
    /// When a request last decoded or finished.
    last_at_ms: i64 = 0,

    /// `now_ms` is read before the request went out, so the next `since` cannot skip a request.
    pub fn observe(self: *Tracker, now_ms: i64, sample: Sample) void {
        self.live = null;
        self.decoding = sample.decoding.len > 0;
        self.prefilling = sample.prefilling;
        if (sample.link == .up) {
            self.since_ms = @max(sample.newest_finished_ms, now_ms);
            if (sample.finished_tps) |t| self.last = t;
            if (sample.finished_prefill_tps) |t| self.last_prefill = t;
            if (sample.finished_tps != null or self.decoding) self.last_at_ms = now_ms;
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

    pub fn view(self: *const Tracker, link: Link, now_ms: i64) View {
        return .{
            .link = link,
            .decoding = self.decoding,
            .prefilling = self.prefilling,
            .live = self.live,
            .last = self.last,
            .last_old = now_ms - self.last_at_ms >= title_last_ms,
        };
    }
};

fn speedText(buf: []u8, tps: f64) []const u8 {
    return (if (tps >= 100)
        std.fmt.bufPrint(buf, "{d:.0}", .{tps})
    else
        std.fmt.bufPrint(buf, "{d:.1}", .{tps})) catch "?";
}

/// The menu bar title; a dot follows it while an update waits.
pub fn title(buf: []u8, v: View) []const u8 {
    const base = titleBase(buf, v);
    if (!v.update_waiting) return base;
    const tail = std.fmt.bufPrint(buf[base.len..], "{s}•", .{if (base.len == 0) "" else " "}) catch return base;
    return buf[0 .. base.len + tail.len];
}

/// The title shows the last request's speed, not a live one: drawn dim.
pub fn titleDim(v: View) bool {
    return v.link == .up and !v.decoding and !v.prefilling and v.last != null and !v.last_old;
}

fn titleBase(buf: []u8, v: View) []const u8 {
    var num: [32]u8 = undefined;
    const text = switch (v.link) {
        .offline, .unauthorized, .metrics_off, .silent => std.fmt.bufPrint(buf, "", .{}),
        .up => if (v.decoding)
            std.fmt.bufPrint(buf, "{s} tok/s", .{if (v.live) |t| speedText(&num, t) else "…"})
        else if (v.prefilling)
            std.fmt.bufPrint(buf, "prefill…", .{})
        else if (v.last != null and !v.last_old)
            std.fmt.bufPrint(buf, "{s} tok/s", .{speedText(&num, v.last.?)})
        else
            std.fmt.bufPrint(buf, "", .{}),
    };
    return text catch "";
}

pub const Icon = enum { outline, filled };

/// Outline while no server answers; filled once one does, whatever it says. Both are template images, so the menu
/// bar tints them like its neighbours; the colour icon is the card's.
pub fn iconFor(link: Link) Icon {
    return if (link == .offline or link == .silent) .outline else .filled;
}

/// A server replied, whatever it said.
pub fn answering(link: Link) bool {
    return link != .offline and link != .silent;
}

/// The orange mark on the menu bar icon: a state that wants a look, in place of words in the title.
pub fn badge(link: Link, port_taken: bool, crashed: bool) bool {
    return switch (link) {
        .silent, .unauthorized => true,
        .offline => port_taken or crashed,
        .up, .metrics_off => false,
    };
}

/// A server that was answering is gone and nothing sushibar did explains it.
pub fn stoppedUnexpectedly(was_answering: bool, link: Link, we_stopped: bool, reaped: bool) bool {
    return was_answering and link == .offline and !we_stopped and !reaped;
}

pub const Activity = struct { link: Link, decoding: bool = false, menu_open: bool = false, changing: bool = false };

pub const poll_fast_ms: i64 = 1000;
pub const poll_idle_ms: i64 = 5000;
pub const poll_offline_ms: i64 = 10_000;

/// The wait before the next poll: a second while a request runs, the menu is open or a start, stop, load or update
/// is under way; longer while the server idles; longest while nothing answers.
pub fn pollWaitMs(a: Activity) i64 {
    if (a.decoding or a.menu_open or a.changing) return poll_fast_ms;
    return if (answering(a.link)) poll_idle_ms else poll_offline_ms;
}

pub const models_every = 5;

/// `/v1/models` is asked each poll until a model is ready and while one loads, else every `models_every`-th poll.
pub fn modelsDue(polls_since: u32, ready: bool, loading: bool) bool {
    return !ready or loading or polls_since >= models_every;
}

pub const scan_every = 5;

/// The process scan runs each poll while the server does not answer in full (a silent sushi and a port taken are
/// told apart by it; a 401 needs the pid whose argv holds the key) or the menu is open, else every `scan_every`-th.
pub fn scanDue(polls_since: u32, link: Link, menu_open: bool) bool {
    return (link != .up and link != .metrics_off) or menu_open or polls_since >= scan_every;
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

// ── Version and update ───────────────────────────────────────────────────

/// `sushi --version` prints `sushi <version>` after any `[mem]` lines the engine logs to stderr.
pub fn parseVersionOutput(text: []const u8) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        if (!std.mem.eql(u8, words.next() orelse continue, "sushi")) continue;
        return words.next();
    }
    return null;
}

pub const Check = union(enum) { available: []const u8, up_to_date, failed: []const u8 };

/// What `sushi update --check` says: `sushi X is available (this is Y): …` or `sushi X is up to date (…)`,
/// after whatever the engine logs first.
pub fn parseCheck(text: []const u8) Check {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r");
        if (line.len == 0) continue;
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        if (!std.mem.eql(u8, words.next() orelse "", "sushi")) continue;
        const v = words.next() orelse continue;
        const rest = words.rest();
        if (std.mem.startsWith(u8, rest, "is available")) return .{ .available = v };
        if (std.mem.startsWith(u8, rest, "is up to date")) return .up_to_date;
    }
    return .{ .failed = failureLine(text) };
}

/// Why a sushi command failed: its last `update:` or `error:` line, else its last line that is not a log header.
pub fn failureLine(text: []const u8) []const u8 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var tagged: []const u8 = "";
    var last: []const u8 = "";
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \r");
        if (line.len == 0 or std.mem.startsWith(u8, line, "──") or std.mem.startsWith(u8, line, "[mem]")) continue;
        last = line;
        if (std.mem.startsWith(u8, line, "update:") or std.mem.startsWith(u8, line, "error:")) tagged = line;
    }
    if (tagged.len > 0) return tagged;
    return if (last.len == 0) "no answer" else last;
}

/// `text` cut to at most `max` bytes on a character boundary, with an ellipsis when it was cut.
pub fn fitNote(buf: []u8, text: []const u8, max: usize) []const u8 {
    const cap = @min(max, buf.len);
    if (text.len <= cap) {
        @memcpy(buf[0..text.len], text);
        return buf[0..text.len];
    }
    const ellipsis = "…";
    var n = if (cap >= ellipsis.len) cap - ellipsis.len else 0;
    while (n > 0 and (text[n] & 0xC0) == 0x80) n -= 1;
    @memcpy(buf[0..n], text[0..n]);
    @memcpy(buf[n..][0..ellipsis.len], ellipsis);
    return buf[0 .. n + ellipsis.len];
}

/// The `update` object of `/props`.
pub const PropsUpdate = struct {
    current: []const u8 = "",
    latest: ?[]const u8 = null,
    available: bool = false,
    command: ?[]const u8 = null,
    err: ?[]const u8 = null,
    url: ?[]const u8 = null,
};

pub fn parseProps(arena: std.mem.Allocator, body: []const u8) !?PropsUpdate {
    const PropsWire = struct {
        update: ?struct {
            current: []const u8 = "",
            latest: ?[]const u8 = null,
            available: bool = false,
            command: ?[]const u8 = null,
            @"error": ?[]const u8 = null,
            url: ?[]const u8 = null,
        } = null,
    };
    const w = try std.json.parseFromSliceLeaky(PropsWire, arena, body, .{ .ignore_unknown_fields = true });
    const u = w.update orelse return null;
    return .{ .current = u.current, .latest = u.latest, .available = u.available, .command = u.command, .err = u.@"error", .url = u.url };
}

pub const releases_url = "https://github.com/beamivalice/sushi/releases/latest";

/// Where "X available" leads: the release page the server's check found, else the project's latest release.
pub fn releaseNotesUrl(props_url: []const u8) []const u8 {
    return if (std.mem.startsWith(u8, props_url, "https://")) props_url else releases_url;
}

pub fn isBrewKeg(real_path: []const u8) bool {
    return std.mem.indexOf(u8, real_path, "/Cellar/sushi/") != null;
}

/// `name` under brew's stable `opt/sushi/libexec` link of the keg `real_path` lives in; null outside a keg.
pub fn brewOptPath(buf: []u8, real_path: []const u8, name: []const u8) ?[]const u8 {
    const marker = "/Cellar/sushi/";
    const i = std.mem.indexOf(u8, real_path, marker) orelse return null;
    return std.fmt.bufPrint(buf, "{s}/opt/sushi/libexec/{s}", .{ real_path[0..i], name }) catch null;
}

pub const UpdatePath = enum { none, server, cli, brew };

pub const UpdateAsk = struct {
    available: bool,
    link: Link,
    /// `/props.update.command` names a brew upgrade.
    server_command: bool = false,
    /// The sushi beside sushibar lives in a Homebrew keg.
    sibling_brew: bool = false,
};

/// A running server updates itself (the page's `POST /v1/update`); with none, `sushi update` does; brew's is left to brew.
pub fn updatePath(a: UpdateAsk) UpdatePath {
    if (!a.available) return .none;
    if (a.server_command or a.sibling_brew) return .brew;
    return if (a.link == .offline) .cli else .server;
}

const day_ms: i64 = 24 * 3600 * 1000;
const retry_ms: i64 = 3600 * 1000;

/// Once a day, an hour after a failed check; `last_ms` 0 means never checked.
pub fn checkDue(last_ms: i64, now_ms: i64, last_ok: bool) bool {
    if (last_ms == 0) return true;
    return now_ms - last_ms >= (if (last_ok) day_ms else retry_ms);
}

/// Whether `model-settings.json` leaves the automatic sushi check on: only `server.update_check: false` turns it off,
/// and a file that is missing or not JSON leaves it on.
pub fn autoCheckOn(arena: std.mem.Allocator, settings_json: []const u8) bool {
    const Doc = struct { server: ?struct { update_check: ?bool = null } = null };
    const doc = std.json.parseFromSliceLeaky(Doc, arena, settings_json, .{ .ignore_unknown_fields = true }) catch return true;
    return (doc.server orelse return true).update_check orelse true;
}

/// The answer sushi's own cache (`update-check.json`) already holds, while it is under a day old; null when a
/// fresh look is due. Reading it is what keeps a bar launch from asking GitHub again.
pub fn cachedCheck(arena: std.mem.Allocator, cache_json: []const u8, now_s: i64, bar_version: []const u8) ?Check {
    const Doc = struct { checked_at: i64 = 0, latest: []const u8 = "" };
    const doc = std.json.parseFromSliceLeaky(Doc, arena, cache_json, .{ .ignore_unknown_fields = true }) catch return null;
    if (doc.checked_at <= 0 or now_s < doc.checked_at or now_s - doc.checked_at >= day_ms / 1000) return null;
    const latest = std.SemanticVersion.parse(doc.latest) catch return .up_to_date;
    const current = std.SemanticVersion.parse(bar_version) catch return .up_to_date;
    return if (latest.order(current) == .gt) .{ .available = doc.latest } else .up_to_date;
}

pub fn relaunchNeeded(launched: []const u8, installed: []const u8) bool {
    return launched.len > 0 and installed.len > 0 and !std.mem.eql(u8, launched, installed);
}

pub const Busy = enum { none, checking, updating };

pub const UpdateView = struct {
    bar_version: []const u8 = "",
    server_version: []const u8 = "",
    latest: []const u8 = "",
    available: bool = false,
    command: []const u8 = "",
    note: []const u8 = "",
    busy: Busy = .none,
    path: UpdatePath = .none,
    checked: CheckResult = .none,
    /// The model the watched server has loaded: an update through the server unloads it.
    model: []const u8 = "",
    /// The Homebrew command was copied a moment ago.
    copied: bool = false,
};

/// The update item's label, or null while there is nothing newer.
pub fn updateLabel(buf: []u8, v: UpdateView) ?[]const u8 {
    if (!v.available) return null;
    return std.fmt.bufPrint(buf, "Update to sushi {s}", .{v.latest}) catch "Update";
}

pub fn copyLabel(buf: []u8, command: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "Copy “{s}”", .{command}) catch "Copy command";
}

/// How long "Copied" stays on the button and the menu item.
pub const copied_ms: i64 = 10_000;

pub fn copiedNow(at_ms: i64, now_ms: i64) bool {
    return at_ms > 0 and now_ms - at_ms < copied_ms;
}

pub fn updateEnabled(v: UpdateView) bool {
    return v.available and v.busy == .none and (v.path == .server or v.path == .cli);
}

/// An update through a server that has a model loaded restarts it mid-request: the user is asked first.
pub fn updateNeedsConfirm(v: UpdateView) bool {
    return v.available and v.path == .server and v.model.len > 0;
}

pub fn updateConfirmInfo(buf: []u8, port: u16, model: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "The server on port {d} restarts and unloads {s}; a request in progress is lost.", .{ port, model }) catch "The server restarts and unloads its model.";
}

pub const ButtonAction = enum { none, check, update, copy_command };

/// How the last check the user asked for ended; the quiet daily checks do not set it.
pub const CheckResult = enum { none, up_to_date, failed };

/// How long an answer stays on the button once the menu is closed and opened again.
pub const result_shown_ms: i64 = 60_000;

/// `last`, unless the menu was opened `result_shown_ms` or more after it was given.
pub fn shownResult(last: CheckResult, at_ms: i64, open_ms: i64) CheckResult {
    return if (open_ms - at_ms >= result_shown_ms) .none else last;
}

pub const UpdateState = enum { idle, checking, up_to_date, available, failed, updating, brew_copy, copied };

pub fn updateState(v: UpdateView) UpdateState {
    switch (v.busy) {
        .checking => return .checking,
        .updating => return .updating,
        .none => {},
    }
    if (v.available and v.path == .brew) return if (v.copied) .copied else .brew_copy;
    if (v.available and updateEnabled(v)) return .available;
    return switch (v.checked) {
        .up_to_date => .up_to_date,
        .failed => .failed,
        .none => .idle,
    };
}

/// The menu's one update button. `accent` is the brand orange. The menu stays open for the answer of a check, the
/// progress of an update and the "Copied" of a Homebrew command; it closes before an update that asks first.
pub const UpdateButton = struct { state: UpdateState, label: []const u8, symbol: []const u8, tip: []const u8, accent: bool, enabled: bool, action: ButtonAction, closes_menu: bool = false, spinner: bool = false };

pub fn updateButton(buf: []u8, v: UpdateView) UpdateButton {
    const check_symbol = "arrow.triangle.2.circlepath";
    const st = updateState(v);
    return switch (st) {
        .checking => .{ .state = st, .label = "Checking…", .symbol = check_symbol, .tip = "Checking for updates", .accent = false, .enabled = false, .action = .none, .spinner = true },
        .updating => .{ .state = st, .label = "Updating…", .symbol = "arrow.down.circle", .tip = "Updating sushi", .accent = false, .enabled = false, .action = .none, .spinner = true },
        .up_to_date => .{ .state = st, .label = "Up to date", .symbol = "checkmark", .tip = "sushi is up to date", .accent = false, .enabled = false, .action = .none },
        .failed => .{ .state = st, .label = "Retry", .symbol = check_symbol, .tip = "The check failed: click to try again", .accent = false, .enabled = true, .action = .check },
        .brew_copy => .{ .state = st, .label = "Upgrade", .symbol = "doc.on.doc", .tip = copyLabel(buf, v.command), .accent = true, .enabled = true, .action = .copy_command },
        .copied => .{ .state = st, .label = "Copied", .symbol = "checkmark", .tip = "Paste it in a terminal to upgrade", .accent = false, .enabled = true, .action = .copy_command },
        .available => .{ .state = st, .label = "Update", .symbol = "arrow.down.circle", .tip = updateLabel(buf, v) orelse "Update", .accent = true, .enabled = true, .action = .update, .closes_menu = updateNeedsConfirm(v) },
        .idle => .{ .state = st, .label = "Check", .symbol = check_symbol, .tip = "Check for Updates", .accent = false, .enabled = true, .action = .check },
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
    port: u16 = default_port,
    /// A sushi that sushibar did not necessarily start listens on the watched port.
    sushi_listening: bool = false,
    /// Something that is not a sushi answers connections on the watched port.
    port_taken: bool = false,
    /// The watched port was just chosen and the first poll has not come back.
    connecting: bool = false,
    /// A Stop sent a while ago has not taken: the same pid still listens (`stopStuck`).
    stop_stuck: bool = false,
};

pub fn statusLine(buf: []u8, s: Server) []const u8 {
    if (s.connecting) return std.fmt.bufPrint(buf, "Port {d}: connecting…", .{s.port}) catch "Connecting…";
    const text = switch (s.link) {
        .offline => if (s.ours_alive)
            std.fmt.bufPrint(buf, "Port {d}: starting", .{s.port})
        else if (s.port_taken)
            std.fmt.bufPrint(buf, "Port {d}: in use by another program", .{s.port})
        else if (!s.sushi_found)
            std.fmt.bufPrint(buf, "Port {d}: not running (sushi not found beside sushibar)", .{s.port})
        else
            std.fmt.bufPrint(buf, "Port {d}: not running", .{s.port}),
        .unauthorized => std.fmt.bufPrint(buf, "Port {d}: running (API key required)", .{s.port}),
        .metrics_off => std.fmt.bufPrint(buf, "Port {d}: running, metrics off", .{s.port}),
        .silent => std.fmt.bufPrint(buf, "Port {d}: not answering", .{s.port}),
        .up => if (s.model) |m|
            std.fmt.bufPrint(buf, "Port {d}: {s}", .{ s.port, m })
        else
            std.fmt.bufPrint(buf, "Port {d}: running, no model loaded", .{s.port}),
    };
    return text catch "Server";
}

pub const Enabled = struct { start: bool, stop: bool };

pub fn enabled(s: Server) Enabled {
    return .{
        .start = s.sushi_found and s.link == .offline and !s.ours_alive and !s.port_taken and !s.connecting,
        .stop = s.ours_alive or s.sushi_listening,
    };
}

pub const ToggleAction = enum { none, start, stop, force_stop };
pub const Toggle = struct { label: []const u8, enabled: bool, action: ToggleAction, symbol: []const u8 };

/// The one Start/Stop item: Stop (with an ellipsis when it asks first) wherever a sushi listens on the port.
pub fn toggle(s: Server, stopping: bool) Toggle {
    if (stopping) return .{ .label = "Stopping…", .enabled = false, .action = .none, .symbol = "hourglass" };
    if (s.stop_stuck and (s.ours_alive or s.sushi_listening)) return .{ .label = "Force Stop", .enabled = true, .action = .force_stop, .symbol = "xmark.octagon" };
    if (s.ours_alive and s.link == .offline) return .{ .label = "Starting…", .enabled = false, .action = .none, .symbol = "hourglass" };
    if (s.ours_alive or s.sushi_listening)
        return .{ .label = if (s.ours_alive) "Stop Server" else "Stop Server…", .enabled = true, .action = .stop, .symbol = "stop.fill" };
    const can = enabled(s).start;
    return .{ .label = "Start Server", .enabled = can, .action = if (can) .start else .none, .symbol = "play.fill" };
}

/// The Stop last sent: the pid signalled, its port, and when the menu stops saying Stopping….
pub const StopSent = struct { pid: i32 = 0, port: u16 = 0, deadline_ms: i64 = 0 };

/// The Stop has not taken: past its deadline the pid it signalled still serves the watched port (`serving_pid`: the
/// started server's verified pid, else the sushi listening there).
pub fn stopStuck(sent: StopSent, watched: u16, now_ms: i64, serving_pid: i32) bool {
    return sent.pid > 0 and sent.port == watched and now_ms >= sent.deadline_ms and serving_pid == sent.pid;
}

pub fn stuckNote(buf: []u8, port: u16) []const u8 {
    return std.fmt.bufPrint(buf, "Port {d}: the server did not stop", .{port}) catch "The server did not stop";
}

pub const QuitButton = struct { label: []const u8, tip: []const u8, stops: bool };

/// Quit leaves the servers sushibar started running; with Option held while there are any, it stops them too.
pub fn quitButton(buf: []u8, option: bool, started: usize) QuitButton {
    if (started == 0) return .{ .label = "Quit", .tip = "Quit sushibar (⌘Q)", .stops = false };
    if (!option) return .{ .label = "Quit", .tip = if (started == 1) "Quit sushibar (⌘Q); the server it started keeps running. Hold ⌥ to stop it too" else "Quit sushibar (⌘Q); the servers it started keep running. Hold ⌥ to stop them too", .stops = false };
    const tip = if (started == 1)
        "Quit sushibar and stop the server it started"
    else
        std.fmt.bufPrint(buf, "Quit sushibar and stop the {d} servers it started", .{started}) catch "Quit sushibar and stop the servers it started";
    return .{ .label = "Quit & Stop", .tip = tip, .stops = true };
}

/// The pages need a server that answers.
pub fn openEnabled(link: Link) bool {
    return link != .offline and link != .silent;
}

// ── The header card ──────────────────────────────────────────────────────

pub const Tone = enum { green, orange, grey };
pub const Pill = struct { label: []const u8, tone: Tone };

pub fn pillFor(in: HeaderIn) Pill {
    if (in.updating) return .{ .label = "Updating…", .tone = .orange };
    if (in.connecting) return .{ .label = "Connecting…", .tone = .grey };
    return switch (in.link) {
        .offline => if (in.starting)
            .{ .label = "Starting", .tone = .orange }
        else if (in.port_taken)
            .{ .label = "Port in use", .tone = .orange }
        else if (!in.sushi_found)
            .{ .label = "No sushi", .tone = .orange }
        else
            .{ .label = "Stopped", .tone = .grey },
        .unauthorized => .{ .label = "Key needed", .tone = .orange },
        .metrics_off => .{ .label = "Metrics off", .tone = .grey },
        .silent => .{ .label = "Not answering", .tone = .orange },
        .up => if (in.loading) .{ .label = "Loading", .tone = .orange } else if (in.model == null) .{ .label = "No model", .tone = .grey } else .{ .label = "Running", .tone = .green },
    };
}

pub fn thousands(buf: []u8, n: u64) []const u8 {
    var digits: [20]u8 = undefined;
    const d = std.fmt.bufPrint(&digits, "{d}", .{n}) catch return "?";
    var i: usize = 0;
    for (d, 0..) |c, k| {
        if (k > 0 and (d.len - k) % 3 == 0 and i < buf.len) {
            buf[i] = ',';
            i += 1;
        }
        if (i < buf.len) {
            buf[i] = c;
            i += 1;
        }
    }
    return buf[0..i];
}

pub const HeaderIn = struct {
    link: Link,
    port: u16,
    loading: bool = false,
    updating: bool = false,
    starting: bool = false,
    connecting: bool = false,
    /// Why Start is off while nothing answers.
    port_taken: bool = false,
    sushi_found: bool = true,
    /// The model the next Start loads.
    last_model: []const u8 = "",
    model: ?[]const u8 = null,
    pack_update: ?PackUpdate = null,
    mem_mb: ?u64 = null,
    bar_version: []const u8 = "",
    /// The running server's version; shown only when it differs from `bar_version`.
    server_version: []const u8 = "",
    update_available: bool = false,
    latest: []const u8 = "",
    decoding: bool = false,
    live: ?f64 = null,
    last: ?f64 = null,
    last_prefill: ?f64 = null,
    gpu_mhz: ?u64 = null,
    gpu_pct: ?u64 = null,
};

pub const Stat = struct { value: []const u8, label: []const u8 };

pub const Header = struct {
    color_icon: bool,
    version: []const u8,
    /// The end of `version` that the card draws in the accent colour.
    version_tail: []const u8,
    pill: Pill,
    line1: []const u8,
    /// Accent line under the model: a newer pack revision is published; empty when not.
    pack_line: []const u8,
    line2: []const u8,
    /// `line1` is a link to the web Models page.
    line1_models: bool,
    show_stats: bool,
    stats: [3]Stat,
    a11y: []const u8,
};

const dash = "–";

fn fmt(a: std.mem.Allocator, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(a, f, args) catch "";
}

/// Everything the menu's header card shows, as strings; the view only draws them.
pub fn buildHeader(a: std.mem.Allocator, in: HeaderIn) !Header {
    var h: Header = undefined;
    h.color_icon = answering(in.link) and !in.connecting;
    h.line1_models = false;
    h.pill = pillFor(in);
    h.version_tail = "";
    const base = if (in.server_version.len > 0 and in.bar_version.len > 0 and !std.mem.eql(u8, in.server_version, in.bar_version))
        fmt(a, "{s} · server {s}", .{ in.bar_version, in.server_version })
    else
        in.bar_version;
    h.version = if (in.update_available and in.latest.len > 0)
        fmt(a, "{s}{s}", .{ base, blk: {
            h.version_tail = fmt(a, " · {s} available", .{in.latest});
            break :blk h.version_tail;
        } })
    else
        base;
    if (in.connecting) {
        h.line1 = "Looking for a server";
        h.line2 = fmt(a, "Port {d}", .{in.port});
    } else if (in.link == .offline) {
        h.line2 = fmt(a, "Port {d}", .{in.port});
        if (in.starting) {
            h.line1 = "Waiting for the server to answer";
        } else if (in.port_taken) {
            h.line1 = "Another program holds this port";
            h.line2 = fmt(a, "Port {d} · pick another in the Port menu", .{in.port});
        } else if (!in.sushi_found) {
            h.line1 = "sushi is not beside sushibar";
            h.line2 = fmt(a, "Port {d} · reinstall sushi", .{in.port});
        } else {
            h.line1 = "Ready to start";
            if (in.last_model.len > 0) h.line2 = fmt(a, "Port {d} · loads {s}", .{ in.port, in.last_model });
        }
    } else if (in.link == .silent) {
        h.line1 = "Holds the port, does not reply";
        h.line2 = fmt(a, "Port {d} · stop it from the menu", .{in.port});
    } else if (in.link == .unauthorized) {
        h.line1 = "The server requires an API key";
        h.line2 = fmt(a, "Port {d} · its stats are hidden", .{in.port});
    } else if (in.model) |m| {
        h.line1 = m;
        h.line2 = if (in.mem_mb) |mb|
            fmt(a, "Port {d} · {d:.1} GB in use", .{ in.port, @as(f64, @floatFromInt(mb)) / 1024.0 })
        else
            fmt(a, "Port {d}", .{in.port});
    } else {
        h.line1 = "Open Models to load one";
        h.line1_models = true;
        h.line2 = fmt(a, "Port {d}", .{in.port});
    }
    h.show_stats = in.link == .up and in.model != null;
    h.pack_line = "";
    if (in.link == .up and !in.loading and in.model != null) if (in.pack_update) |u| {
        h.pack_line = fmt(a, "Pack {s} available (you have {s})", .{ u.latest, u.current });
    };

    var tmp: [32]u8 = undefined;
    const speed: ?f64 = if (in.decoding) in.live else in.last;
    h.stats[0] = .{
        .value = if (speed) |t| try a.dupe(u8, speedText(&tmp, t)) else dash,
        .label = if (!in.decoding and in.last != null) "LAST TOK/S" else "DECODE TOK/S",
    };
    h.stats[1] = .{ .value = if (in.last_prefill) |t| try a.dupe(u8, thousands(&tmp, @intFromFloat(@round(t)))) else dash, .label = "PREFILL TOK/S" };
    const mhz_ok = (in.gpu_mhz orelse 0) > 0;
    h.stats[2] = if (in.gpu_pct) |p| .{
        .value = fmt(a, "{d}%", .{p}),
        .label = if (mhz_ok) fmt(a, "GPU · {s} MHZ", .{thousands(&tmp, in.gpu_mhz.?)}) else "GPU",
    } else .{ .value = if (mhz_ok) try a.dupe(u8, thousands(&tmp, in.gpu_mhz.?)) else dash, .label = "GPU MHZ" };

    var state_word: [24]u8 = undefined;
    const word = std.ascii.lowerString(&state_word, if (in.link == .offline and !in.starting) "stopped" else h.pill.label);
    var out = fmt(a, "sushi {s} on port {d}", .{ word, in.port });
    if (in.link == .offline and !in.starting) out = fmt(a, "sushi stopped, port {d}", .{in.port});
    if (in.model) |m| out = fmt(a, "{s}, {s}", .{ out, m });
    if (speed) |t| out = fmt(a, "{s}, decode {d:.0} tokens per second", .{ out, t });
    h.a11y = out;
    return h;
}

// ── The sparkline ────────────────────────────────────────────────────────

/// `x` is 0 (window start) to 1 (now), `v` the decode tok/s over the interval ending there; `gap` means no line
/// joins it to the point before (server down, a restart, the first point).
pub const SparkPoint = struct { x: f32, v: f32, gap: bool };

pub const PlotPoint = struct { x: f64, y: f64, gap: bool };

/// Sparkline points in view coordinates: the window spans `left` to `left + inner_w`, and the series' own peak (plus
/// a tenth of headroom) sits at `top`, zero at `bottom`.
pub fn plotSpark(out: []PlotPoint, pts: []const SparkPoint, left: f64, inner_w: f64, top: f64, bottom: f64) []PlotPoint {
    var peak: f64 = 1;
    for (pts) |p| peak = @max(peak, p.v);
    peak *= 1.1;
    const n = @min(out.len, pts.len);
    for (pts[0..n], 0..) |p, i| {
        const x01 = std.math.clamp(@as(f64, p.x), 0, 1);
        out[i] = .{ .x = left + x01 * inner_w, .y = bottom - std.math.clamp(p.v / peak, 0, 1) * (bottom - top), .gap = p.gap };
    }
    return out[0..n];
}

/// The scale the sparkline is drawn to: its peak and the span it covers.
pub fn sparkCaption(buf: []u8, pts: []const SparkPoint, window_ms: i64) []const u8 {
    if (pts.len < 2) return "";
    var peak: f32 = 0;
    for (pts) |p| peak = @max(peak, p.v);
    return std.fmt.bufPrint(buf, "peak {d:.0} tok/s · last {d} min", .{ peak, @divTrunc(window_ms, 60_000) }) catch "";
}

const spark_max_dt_ms: i64 = 5000;

pub fn sparkPoints(out: []SparkPoint, hist: []const Hist, now_ms: i64, window_ms: i64) []SparkPoint {
    var n: usize = 0;
    const start = now_ms - window_ms;
    var prev_emitted = false;
    var i: usize = 1;
    while (i < hist.len and n < out.len) : (i += 1) {
        const a = hist[i - 1];
        const b = hist[i];
        if (b.t_ms < start) continue;
        const dt = b.t_ms - a.t_ms;
        const x: f32 = @floatCast(@min(1.0, @as(f64, @floatFromInt(b.t_ms - start)) / @as(f64, @floatFromInt(window_ms))));
        if (b.gen < a.gen or dt <= 0) {
            out[n] = .{ .x = x, .v = 0, .gap = true };
            prev_emitted = true;
            n += 1;
            continue;
        }
        const rate = @as(f64, @floatFromInt(b.gen - a.gen)) * 1000.0 / @as(f64, @floatFromInt(dt));
        out[n] = .{ .x = x, .v = @floatCast(rate), .gap = !prev_emitted or dt > spark_max_dt_ms };
        prev_emitted = true;
        n += 1;
    }
    return out[0..n];
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

/// The arguments of a KERN_PROCARGS2 buffer (argc, the executable path, NUL padding, then argc NUL-separated
/// strings): the environment that follows them is never yielded.
const ProcArgs = struct {
    it: std.mem.SplitIterator(u8, .scalar),
    left: usize,

    fn init(args: []const u8) ?ProcArgs {
        if (args.len < 4) return null;
        const argc: usize = @intCast(@max(0, std.mem.readInt(i32, args[0..4], builtin.cpu.arch.endian())));
        var it = std.mem.splitScalar(u8, args[4..], 0);
        _ = it.next();
        while (it.peek()) |s| : (_ = it.next()) if (s.len > 0) break;
        return .{ .it = it, .left = argc };
    }

    fn next(self: *ProcArgs) ?[]const u8 {
        if (self.left == 0) return null;
        self.left -= 1;
        return self.it.next();
    }

    /// `name`'s value in the environment that follows the arguments; call once `next` is done.
    fn env(self: *ProcArgs, name: []const u8) ?[]const u8 {
        while (self.it.next()) |e| {
            if (e.len > name.len and e[name.len] == '=' and std.mem.startsWith(u8, e, name)) return e[name.len + 1 ..];
        }
        return null;
    }
};

/// The key the server was started with: `--api-key <key>`, or `--api-key-env <VAR>` read from the server's own
/// environment; the last of them wins, as the server reads them. Null with none or an empty one.
pub fn argsApiKey(args: []const u8) ?[]const u8 {
    var it = ProcArgs.init(args) orelse return null;
    var prev: []const u8 = "";
    var key: ?[]const u8 = null;
    var var_name: ?[]const u8 = null;
    while (it.next()) |a| : (prev = a) {
        if (std.mem.eql(u8, prev, "--api-key")) {
            key = a;
            var_name = null;
        } else if (std.mem.eql(u8, prev, "--api-key-env")) {
            var_name = a;
            key = null;
        }
    }
    if (var_name) |n| key = it.env(n);
    const k = key orelse return null;
    return if (k.len > 0) k else null;
}

pub fn argsServe(args: []const u8, port: u16) bool {
    var it = ProcArgs.init(args) orelse return false;
    var want: [8]u8 = undefined;
    const port_text = std.fmt.bufPrint(&want, "{d}", .{port}) catch return false;
    var serving = false;
    var port_ok = false;
    var prev: []const u8 = "";
    while (it.next()) |a| {
        if (std.mem.eql(u8, a, "--serve")) serving = true;
        if (std.mem.eql(u8, prev, "--port") and std.mem.eql(u8, a, port_text)) port_ok = true;
        prev = a;
    }
    return serving and port_ok;
}

/// The value of `--log-file` in a process's arguments.
pub fn argsLogFile(args: []const u8) ?[]const u8 {
    var it = ProcArgs.init(args) orelse return null;
    var prev: []const u8 = "";
    while (it.next()) |a| : (prev = a) if (std.mem.eql(u8, prev, "--log-file")) return a;
    return null;
}

/// `name` in the directory of the running executable.
pub fn siblingPath(buf: []u8, exe_path: []const u8, name: []const u8) ?[]const u8 {
    const dir = std.fs.path.dirname(exe_path) orelse return null;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ dir, name }) catch null;
}


// ── Ports ────────────────────────────────────────────────────────────────

/// A process listening on a TCP port.
pub const Listener = struct { pid: i32, port: u16 };

pub const StopPlan = union(enum) { none, now: i32, confirm: i32 };

pub fn planStop(ours: ?i32, watched: u16, listeners: []const Listener) StopPlan {
    if (ours) |pid| return .{ .now = pid };
    for (listeners) |l| if (l.port == watched and l.pid > 1) return .{ .confirm = l.pid };
    return .none;
}

/// A port typed into the Other Port dialog.
pub fn parsePortText(text: []const u8) ?u16 {
    return parsePort(std.mem.trim(u8, text, " \t\r\n"));
}

pub const port_placeholder: [*:0]const u8 = "e.g. 8080";
pub const port_hint: [*:0]const u8 = "1–65535";

/// An explicit `--port` beats the saved one, which beats the default port.
pub fn choosePort(explicit: ?u16, saved: ?u16, default: u16) u16 {
    return explicit orelse saved orelse default;
}

pub const max_saved = 16;

/// The ports the user added, beside the default port that is always listed. Oldest first.
pub const SavedPorts = struct {
    list: [max_saved]u16 = undefined,
    n: usize = 0,

    pub fn items(self: *const SavedPorts) []const u16 {
        return self.list[0..self.n];
    }

    pub fn has(self: *const SavedPorts, p: u16) bool {
        return std.mem.indexOfScalar(u16, self.items(), p) != null;
    }

    /// The default port is listed on its own and never saved; a full list drops its oldest.
    pub fn add(self: *SavedPorts, p: u16, default: u16) void {
        if (p == default or self.has(p)) return;
        if (self.n == max_saved) self.remove(self.list[0]);
        self.list[self.n] = p;
        self.n += 1;
    }

    pub fn remove(self: *SavedPorts, p: u16) void {
        const i = std.mem.indexOfScalar(u16, self.items(), p) orelse return;
        std.mem.copyForwards(u16, self.list[i .. self.n - 1], self.list[i + 1 .. self.n]);
        self.n -= 1;
    }
};

pub const max_model = 96;

/// `sushibar.json`: the watched port, the ports added to the menu and the model the watched server last had loaded.
pub const Saved = struct {
    port: ?u16 = null,
    ports: SavedPorts = .{},
    model_buf: [max_model]u8 = undefined,
    model_len: usize = 0,

    pub fn model(self: *const Saved) []const u8 {
        return self.model_buf[0..self.model_len];
    }
};

/// A file from before added ports holds the one key `port`; one from before the model has no `model`.
pub fn parseSaved(text: []const u8) Saved {
    var arena: std.heap.FixedBufferAllocator = .init(&saved_scratch);
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch return .{};
    const obj = switch (v) {
        .object => |o| o,
        else => return .{},
    };
    var out: Saved = .{};
    if (obj.get("port")) |p| out.port = default_port_mod.fromValue(p);
    if (obj.get("ports")) |list| if (list == .array) for (list.array.items) |item| {
        const p = default_port_mod.fromValue(item) orelse continue;
        out.ports.add(p, 0);
    };
    if (obj.get("model")) |m| if (m == .string and m.string.len <= max_model) {
        @memcpy(out.model_buf[0..m.string.len], m.string);
        out.model_len = m.string.len;
    };
    return out;
}
var saved_scratch: [4096]u8 = undefined;

pub fn formatSaved(buf: []u8, watched: u16, ports: []const u16, model: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("{{\"port\":{d},\"ports\":[", .{watched}) catch return "";
    for (ports, 0..) |p, i| w.print("{s}{d}", .{ if (i > 0) "," else "", p }) catch return "";
    w.writeAll("]") catch return "";
    if (model.len > 0) w.print(",\"model\":{f}", .{json_text.Str{ .s = model }}) catch return "";
    w.writeAll("}\n") catch return "";
    return w.buffered();
}

/// The body of `POST /v1/load-model` for the model Start brings back; `default` is what the web page's switch sends.
pub fn loadModelBody(buf: []u8, model: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{{\"model\":{f},\"default\":true}}", .{json_text.Str{ .s = model }}) catch "";
}

/// The port the Port menu lists first and never lets go: `server.port` of `model-settings.json`, else the built-in
/// one. The same rule as the server's, through `default_port.fromValue`.
pub fn defaultPortOf(arena: std.mem.Allocator, settings_json: []const u8) u16 {
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena, settings_json, .{}) catch return default_port;
    const server = switch (v) {
        .object => |o| o.get("server") orelse return default_port,
        else => return default_port,
    };
    if (server != .object) return default_port;
    return default_port_mod.fromValue(server.object.get("port") orelse return default_port) orelse default_port;
}

/// Where the watch goes when `removed` is forgotten: the default port if it was the watched one.
pub fn afterRemove(watched: u16, removed: u16, default: u16) u16 {
    return if (watched == removed) default else watched;
}

/// A child's wait status: its exit code, or the signal that ended it.
pub const Exit = union(enum) { code: u8, signal: u8 };

pub fn exitOf(status: u32) Exit {
    const sig: u8 = @intCast(status & 0x7f);
    return if (sig == 0) .{ .code = @intCast((status >> 8) & 0xff) } else .{ .signal = sig };
}

pub const note_kept_ms: i64 = 60_000;

/// A note with news, not a problem, is dropped when the menu opens a minute or more after it was given.
pub fn noteKept(problem: bool, at_ms: i64, open_ms: i64) bool {
    return problem or open_ms - at_ms < note_kept_ms;
}

/// The longest a note is drawn in the menu; the item's tooltip has the rest, so the menu keeps its width.
pub const note_title_max = 46;

fn exitText(buf: []u8, e: Exit) []const u8 {
    return (switch (e) {
        .code => |c| std.fmt.bufPrint(buf, "exit {d}", .{c}),
        .signal => |g| std.fmt.bufPrint(buf, "signal {d}", .{g}),
    }) catch "";
}

/// A server sushibar started exited before it ever answered.
pub fn startFailedNote(buf: []u8, port: u16, e: Exit) []const u8 {
    var eb: [16]u8 = undefined;
    return std.fmt.bufPrint(buf, "Port {d}: server did not start ({s})", .{ port, exitText(&eb, e) }) catch "Server did not start";
}

/// A server sushibar started answered and then exited on its own.
pub fn exitedNote(buf: []u8, port: u16, e: Exit) []const u8 {
    var eb: [16]u8 = undefined;
    return std.fmt.bufPrint(buf, "Port {d}: server exited ({s})", .{ port, exitText(&eb, e) }) catch "Server exited";
}

/// A server that sushibar did not start went away.
pub fn stoppedNote(buf: []u8, port: u16, hour: u8, minute: u8) []const u8 {
    return std.fmt.bufPrint(buf, "Port {d}: server stopped at {d:0>2}:{d:0>2}", .{ port, hour, minute }) catch "Server stopped";
}

/// A sushi listener and what its `/v1/models` said.
pub const Listing = struct { port: u16, pid: i32, model: ?[]const u8 = null, models: ModelsKnown = .unknown };

/// What the last `/v1/models` fetch of a listener said: not asked yet, answered, or did not answer.
pub const ModelsKnown = enum { unknown, answered, failed };

pub const PortEntry = struct { port: u16, checked: bool, listening: bool, model: ?[]const u8, models: ModelsKnown = .answered, removable: bool = false };

/// The Port submenu: the default port first, then the added ports, then the watched port when it is none of those. A
/// sushi on any other port is not listed. Every port but the default can be forgotten, the watched one too (even when
/// a file from before added ports or `--port` named it and it was never saved); the listener scan only gives a row its state.
pub fn portEntries(out: []PortEntry, watched: u16, default: u16, saved: []const u16, listings: []const Listing) []PortEntry {
    var n: usize = 0;
    const lead = [_][]const u16{ &.{default}, saved };
    for (lead) |ports| for (ports) |p| {
        if (n < out.len and !listed(out[0..n], p)) {
            out[n] = portEntry(p, watched, default, listings);
            n += 1;
        }
    };
    if (n < out.len and !listed(out[0..n], watched)) {
        out[n] = portEntry(watched, watched, default, listings);
        n += 1;
    }
    return out[0..n];
}

fn portEntry(p: u16, watched: u16, default: u16, listings: []const Listing) PortEntry {
    const listing = for (listings) |l| {
        if (l.port == p) break l;
    } else null;
    return .{
        .port = p,
        .checked = p == watched,
        .listening = listing != null,
        .model = if (listing) |l| l.model else null,
        .models = if (listing) |l| l.models else .answered,
        .removable = p != default,
    };
}

fn listed(entries: []const PortEntry, p: u16) bool {
    for (entries) |e| if (e.port == p) return true;
    return false;
}

pub fn portLabel(buf: []u8, e: PortEntry) []const u8 {
    const what = if (!e.listening) "not running" else if (e.model) |m| m else switch (e.models) {
        .answered => "no model",
        .failed => "not answering",
        .unknown => "…",
    };
    return std.fmt.bufPrint(buf, "{d} — {s}", .{ e.port, what }) catch "";
}

// ── Log window ───────────────────────────────────────────────────────────

/// `standard`: the per-port default, the file when the server was not started with `--log-file`.
pub const LogKind = enum { standard, given, off, relative };
pub const LogSource = struct { kind: LogKind, path: []const u8 = "" };

/// Where the server on `port` writes its log. `server_args` are its KERN_PROCARGS2 bytes when a sushi listens
/// there; without them the default file is the answer.
pub fn logSource(buf: []u8, home: []const u8, port: u16, server_args: ?[]const u8) LogSource {
    if (server_args) |args| if (argsLogFile(args)) |v| {
        if (std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "none")) return .{ .kind = .off };
        const path = std.fmt.bufPrint(buf, "{s}", .{v}) catch return standardLog(buf, home, port);
        return .{ .kind = if (v.len > 0 and v[0] == '/') .given else .relative, .path = path };
    };
    return standardLog(buf, home, port);
}

fn standardLog(buf: []u8, home: []const u8, port: u16) LogSource {
    const path = std.fmt.bufPrint(buf, "{s}/.sushi/logs/sushi-{d}.log", .{ home, port }) catch "";
    return .{ .kind = .standard, .path = path };
}

/// What the log window says in place of text: the file it looked for, or why there is none.
pub fn logNotice(buf: []u8, port: u16, src: LogSource) []const u8 {
    const t = switch (src.kind) {
        .standard => std.fmt.bufPrint(buf, "No log file yet.\n\nLooked for {s}\n\nsushi writes it when it starts. A server started with --log-file logs to that file instead.", .{src.path}),
        .given => std.fmt.bufPrint(buf, "No log file yet.\n\nThe server on port {d} was started with --log-file {s}.", .{ port, src.path }),
        .off => std.fmt.bufPrint(buf, "The server on port {d} was started with --log-file off, so it keeps no log file.", .{port}),
        .relative => std.fmt.bufPrint(buf, "The server on port {d} was started with --log-file {s}, a relative path: sushibar cannot tell which folder it is relative to.", .{ port, src.path }),
    };
    return t catch "No log file.";
}

/// Where the last `max` lines of `text` begin.
pub fn tailStart(text: []const u8, max: usize) usize {
    if (max == 0) return text.len;
    var end = text.len;
    if (end > 0 and text[end - 1] == '\n') end -= 1;
    var seen: usize = 0;
    var i = end;
    while (i > 0) {
        i -= 1;
        if (text[i] == '\n') {
            seen += 1;
            if (seen == max) return i + 1;
        }
    }
    return 0;
}

/// Bytes of `chunk` up to its last newline: a line the server is still writing waits for the rest.
pub fn completeLen(chunk: []const u8) usize {
    return if (std.mem.lastIndexOfScalar(u8, chunk, '\n')) |i| i + 1 else 0;
}

pub fn countLines(text: []const u8) usize {
    return std.mem.count(u8, text, "\n");
}

/// How far into which file (device and inode) the window has read.
pub const Follow = struct { offset: u64 = 0, file: u64 = 0 };
pub const FollowStep = enum { idle, append, reload };

/// A shorter file was truncated and a different one rotated in: both start over.
pub fn followStep(f: Follow, size: u64, file: u64) FollowStep {
    if (file != f.file or size < f.offset) return .reload;
    return if (size > f.offset) .append else .idle;
}

/// How many of the oldest lines to drop once the window holds more than twice `keep`.
pub fn excessLines(shown: usize, keep: usize) usize {
    return if (shown > keep * 2) shown - keep else 0;
}

/// The view is following the log while its visible bottom is within a few points of the text's end.
pub fn atBottom(visible_max_y: f64, doc_height: f64) bool {
    return visible_max_y >= doc_height - 24;
}

test "ports: an explicit --port beats the saved one, which beats the default" {
    try std.testing.expectEqual(@as(u16, 18801), choosePort(18801, 18802, 9000));
    try std.testing.expectEqual(@as(u16, 18802), choosePort(null, 18802, 9000));
    try std.testing.expectEqual(@as(u16, 9000), choosePort(null, null, 9000));
}

test "ports: the saved file round-trips the watched port and the added ones, and refuses anything else" {
    var b: [320]u8 = undefined;
    const got = parseSaved(formatSaved(&b, 18802, &.{ 18803, 18804 }, ""));
    try std.testing.expectEqual(@as(?u16, 18802), got.port);
    try std.testing.expectEqualSlices(u16, &.{ 18803, 18804 }, got.ports.items());
    try std.testing.expectEqualStrings("", got.model());
    for ([_][]const u8{ "", "{}", "not json", "{\"port\":0}", "{\"port\":70000}", "{\"port\":\"x\"}", "{\"port\":-1}" }) |bad| {
        const r = parseSaved(bad);
        try std.testing.expect(r.port == null and r.ports.n == 0);
    }
    try std.testing.expectEqual(@as(?u16, 5), parseSaved("{\"other\":1,\"port\":5}").port);
}

test "last model: the saved file keeps the model the watched server had loaded, escaped, and old files have none" {
    var b: [320]u8 = undefined;
    const got = parseSaved(formatSaved(&b, 18802, &.{18803}, "Example-Flash-Next-Sushi-STQ-2.75bpw"));
    try std.testing.expectEqualStrings("Example-Flash-Next-Sushi-STQ-2.75bpw", got.model());
    try std.testing.expectEqual(@as(?u16, 18802), got.port);
    try std.testing.expectEqualSlices(u16, &.{18803}, got.ports.items());
    try std.testing.expectEqualStrings("a\"b\\c", parseSaved(formatSaved(&b, 1, &.{}, "a\"b\\c")).model());
    try std.testing.expectEqualStrings("", parseSaved("{\"port\":5,\"ports\":[6]}").model());
    try std.testing.expectEqualStrings("", parseSaved("{\"port\":5}").model());
    try std.testing.expectEqualStrings("", parseSaved("{\"model\":5}").model());
    const long: [max_model + 1]u8 = @splat('m');
    var jb: [160]u8 = undefined;
    try std.testing.expectEqualStrings("", parseSaved(try std.fmt.bufPrint(&jb, "{{\"model\":\"{s}\"}}", .{&long})).model());
    try std.testing.expect(std.mem.indexOf(u8, formatSaved(&b, 1, &.{}, ""), "model") == null);
}

test "last model: Start asks the server to load it by the web page's request" {
    var b: [160]u8 = undefined;
    try std.testing.expectEqualStrings("{\"model\":\"Qwen-X\",\"default\":true}", loadModelBody(&b, "Qwen-X"));
    try std.testing.expectEqualStrings("{\"model\":\"a\\\"b\",\"default\":true}", loadModelBody(&b, "a\"b"));
}

test "ports: a file from before added ports reads as its one watched port" {
    const old = parseSaved("{\"port\":18802}\n");
    try std.testing.expectEqual(@as(?u16, 18802), old.port);
    try std.testing.expectEqual(@as(usize, 0), old.ports.n);
}

test "ports: the added list drops what is not a port, repeats and what overflows" {
    const got = parseSaved("{\"port\":1,\"ports\":[8080,\"x\",0,70000,8080,9090,1.5,null]}");
    try std.testing.expectEqualSlices(u16, &.{ 8080, 9090 }, got.ports.items());
    var many: SavedPorts = .{};
    for (0..max_saved + 4) |i| many.add(@intCast(2000 + i), 12345);
    try std.testing.expectEqual(max_saved, many.n);
    try std.testing.expect(many.has(@intCast(2000 + max_saved + 3)) and !many.has(2000));
    try std.testing.expect(parseSaved("{\"ports\":8080}").ports.n == 0);
}

test "ports: adding dedupes and never saves the default; removing forgets one port" {
    var s: SavedPorts = .{};
    s.add(18802, 12345);
    s.add(18803, 12345);
    s.add(18802, 12345);
    s.add(12345, 12345);
    try std.testing.expectEqualSlices(u16, &.{ 18802, 18803 }, s.items());
    s.remove(18802);
    try std.testing.expectEqualSlices(u16, &.{18803}, s.items());
    s.remove(9);
    try std.testing.expectEqualSlices(u16, &.{18803}, s.items());
}

test "ports: removing the watched port falls back to the default, any other leaves the watch alone" {
    try std.testing.expectEqual(@as(u16, 12345), afterRemove(18802, 18802, 12345));
    try std.testing.expectEqual(@as(u16, 18803), afterRemove(18803, 18802, 12345));
}

test "ports: the default port is the settings file's server.port, else 12345" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try std.testing.expectEqual(@as(u16, 8080), defaultPortOf(a, "{\"server\":{\"port\":8080}}"));
    for ([_][]const u8{ "", "{}", "not json", "{\"server\":{}}", "{\"server\":{\"port\":0}}", "{\"server\":{\"port\":\"80\"}}", "{\"server\":{\"port\":65536}}", "{\"server\":5}" }) |bad|
        try std.testing.expectEqual(default_port, defaultPortOf(a, bad));
}

test "ports: typed text is a port from 1 to 65535" {
    try std.testing.expectEqual(@as(u16, 12345), parsePortText(" 12345\n").?);
    for ([_][]const u8{ "", "0", "65536", "-1", "abc", "12 3", "1.5" }) |bad| try std.testing.expect(parsePortText(bad) == null);
}

test "stop: a server sushibar started goes at once, any other sushi on the port after a confirm" {
    const l = [_]Listener{ .{ .pid = 50, .port = 18802 }, .{ .pid = 60, .port = 18801 } };
    try std.testing.expectEqual(StopPlan{ .now = 77 }, planStop(77, 18801, &l));
    try std.testing.expectEqual(StopPlan{ .confirm = 60 }, planStop(null, 18801, &l));
    try std.testing.expectEqual(StopPlan{ .confirm = 50 }, planStop(null, 18802, &l));
    try std.testing.expectEqual(StopPlan.none, planStop(null, 12345, &l));
    try std.testing.expectEqual(StopPlan.none, planStop(null, 18801, &.{}));
    try std.testing.expectEqual(StopPlan.none, planStop(null, 18801, &.{.{ .pid = 0, .port = 18801 }}));
}

test "exit notes: a wait status reads as an exit code or a signal, and the notes name the port and the cause" {
    try std.testing.expectEqual(Exit{ .code = 1 }, exitOf(1 << 8));
    try std.testing.expectEqual(Exit{ .code = 0 }, exitOf(0));
    try std.testing.expectEqual(Exit{ .signal = 11 }, exitOf(11));
    try std.testing.expectEqual(Exit{ .signal = 9 }, exitOf(9 | 0x80));
    var b: [96]u8 = undefined;
    try std.testing.expectEqualStrings("Port 18801: server did not start (exit 1)", startFailedNote(&b, 18801, .{ .code = 1 }));
    try std.testing.expectEqualStrings("Port 18801: server exited (signal 11)", exitedNote(&b, 18801, .{ .signal = 11 }));
    try std.testing.expectEqualStrings("Port 18801: server stopped at 09:05", stoppedNote(&b, 18801, 9, 5));
    var b2: [96]u8 = undefined;
    var b3: [96]u8 = undefined;
    for ([_][]const u8{ startFailedNote(&b, 65535, .{ .signal = 31 }), exitedNote(&b2, 65535, .{ .signal = 31 }), stoppedNote(&b3, 65535, 23, 59) }) |n|
        try std.testing.expect(n.len <= note_title_max);
}

test "unexpected stop: only an answering server that went away without sushibar stopping it" {
    try std.testing.expect(stoppedUnexpectedly(true, .offline, false, false));
    try std.testing.expect(!stoppedUnexpectedly(false, .offline, false, false));
    try std.testing.expect(!stoppedUnexpectedly(true, .up, false, false));
    try std.testing.expect(!stoppedUnexpectedly(true, .offline, true, false));
    try std.testing.expect(!stoppedUnexpectedly(true, .offline, false, true));
    try std.testing.expect(answering(.up) and answering(.unauthorized) and answering(.metrics_off) and !answering(.offline) and !answering(.silent));
}

test "badge: a state that wants a look marks the icon, a healthy or stopped server does not" {
    try std.testing.expect(badge(.silent, false, false));
    try std.testing.expect(badge(.unauthorized, false, false));
    try std.testing.expect(badge(.offline, true, false));
    try std.testing.expect(badge(.offline, false, true));
    try std.testing.expect(!badge(.offline, false, false));
    try std.testing.expect(!badge(.up, false, false));
    try std.testing.expect(!badge(.metrics_off, false, false));
}

test "release notes: the page the server's check found, else the project's latest release" {
    try std.testing.expectEqualStrings("https://example.test/v9", releaseNotesUrl("https://example.test/v9"));
    try std.testing.expectEqualStrings(releases_url, releaseNotesUrl(""));
    try std.testing.expectEqualStrings(releases_url, releaseNotesUrl("file:///etc/passwd"));
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const u = (try parseProps(arena.allocator(), "{\"update\":{\"current\":\"1.5.0\",\"url\":\"https://example.test/v1.5.1\"}}")).?;
    try std.testing.expectEqualStrings("https://example.test/v1.5.1", u.url.?);
}

test "port menu: the default first, then the added ports, each with its state; a listener nobody added is not listed" {
    var out: [12]PortEntry = undefined;
    var b: [96]u8 = undefined;
    const l = [_]Listing{ .{ .port = 18802, .pid = 5, .model = "Qwen", .models = .answered }, .{ .port = 12345, .pid = 6, .models = .answered }, .{ .port = 7000, .pid = 7, .models = .answered } };
    const e = portEntries(&out, 12345, 12345, &.{ 18803, 18802 }, &l);
    try std.testing.expectEqual(@as(usize, 3), e.len);
    try std.testing.expectEqual(@as(u16, 12345), e[0].port);
    try std.testing.expectEqualStrings("12345 — no model", portLabel(&b, e[0]));
    try std.testing.expect(e[0].checked and !e[1].checked);
    try std.testing.expectEqualStrings("18803 — not running", portLabel(&b, e[1]));
    try std.testing.expectEqualStrings("18802 — Qwen", portLabel(&b, e[2]));
}

test "port menu: the default port is always listed and never removable, the others are" {
    var out: [12]PortEntry = undefined;
    const none = portEntries(&out, 12345, 12345, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 1), none.len);
    try std.testing.expect(none[0].checked and !none[0].listening and !none[0].removable);
    const l = [_]Listing{.{ .port = 18802, .pid = 5, .models = .answered }};
    const e = portEntries(&out, 12345, 12345, &.{ 18803, 18802 }, &l);
    try std.testing.expect(!e[0].removable and e[1].removable and e[2].removable);
}

test "port menu: a listener nobody added is not listed, an absent watched port is, and can be dropped" {
    var out: [12]PortEntry = undefined;
    const l = [_]Listing{.{ .port = 7000, .pid = 5, .models = .answered }};
    const e = portEntries(&out, 18801, 12345, &.{}, &l);
    try std.testing.expectEqual(@as(usize, 2), e.len);
    try std.testing.expectEqual(@as(u16, 18801), e[1].port);
    try std.testing.expect(e[1].checked and !e[1].listening and e[1].removable);
}

test "port menu: the watched port can be forgotten even when it was never saved and a sushi listens on it" {
    var out: [12]PortEntry = undefined;
    const l = [_]Listing{.{ .port = 18802, .pid = 5, .models = .answered }};
    const e = portEntries(&out, 18802, 12345, &.{}, &l);
    try std.testing.expectEqual(@as(usize, 2), e.len);
    try std.testing.expect(e[1].port == 18802 and e[1].checked and e[1].listening and e[1].removable);
    const after = portEntries(&out, afterRemove(18802, 18802, 12345), 12345, &.{}, &l);
    try std.testing.expectEqual(@as(usize, 1), after.len);
    try std.testing.expect(after[0].checked and !after[0].removable);
}

test "port menu: forgetting an added port drops it even while a sushi still listens there" {
    var out: [12]PortEntry = undefined;
    const l = [_]Listing{.{ .port = 18802, .pid = 5, .models = .answered }};
    const before = portEntries(&out, 12345, 12345, &.{18802}, &l);
    try std.testing.expectEqual(@as(usize, 2), before.len);
    try std.testing.expect(before[1].listening and before[1].removable);
    const after = portEntries(&out, 12345, 12345, &.{}, &l);
    try std.testing.expectEqual(@as(usize, 1), after.len);
}

test "port menu: a default changed in settings is not listed twice when it was added before" {
    var out: [12]PortEntry = undefined;
    const e = portEntries(&out, 9000, 9000, &.{ 9000, 18803 }, &.{});
    try std.testing.expectEqual(@as(usize, 2), e.len);
    try std.testing.expectEqual(@as(u16, 9000), e[0].port);
    try std.testing.expectEqual(@as(u16, 18803), e[1].port);
}

test "port menu: a models fetch that failed or has not run never reads as no model" {
    var out: [12]PortEntry = undefined;
    var b: [96]u8 = undefined;
    const l = [_]Listing{ .{ .port = 18802, .pid = 5, .models = .failed }, .{ .port = 18803, .pid = 6, .models = .unknown } };
    const e = portEntries(&out, 18801, 12345, &.{ 18802, 18803 }, &l);
    try std.testing.expectEqualStrings("18802 — not answering", portLabel(&b, e[1]));
    try std.testing.expectEqualStrings("18803 — …", portLabel(&b, e[2]));
}

test "note: a failure line is the last update/error line, and a long note is cut with an ellipsis, never dropped" {
    const text = "── sushi 1.5.0 update, 2026-11-03 14:57 UTC\nupdate: disk full\nlater noise\n";
    try std.testing.expectEqualStrings("update: disk full", failureLine(text));
    try std.testing.expectEqualStrings("later noise", failureLine("── header\nlater noise\n"));
    try std.testing.expectEqualStrings("no answer", failureLine("── header\n"));
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("short", fitNote(&b, "short", 20));
    try std.testing.expectEqualStrings("aaaaaaaaaaaaaaaaa…", fitNote(&b, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 20));
    try std.testing.expectEqualStrings("aaaaaaaaaaaaaaaaé…", fitNote(&b, "aaaaaaaaaaaaaaaaéééé", 21));
}

test "header: the server's version shows only when it differs, before the update tail" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("1.5.0 · server 1.4.1", (try buildHeader(a, .{ .link = .up, .port = 1, .bar_version = "1.5.0", .server_version = "1.4.1" })).version);
    try std.testing.expectEqualStrings("1.5.0", (try buildHeader(a, .{ .link = .up, .port = 1, .bar_version = "1.5.0", .server_version = "1.5.0" })).version);
    const h = try buildHeader(a, .{ .link = .up, .port = 1, .bar_version = "1.5.0", .server_version = "1.4.1", .update_available = true, .latest = "1.5.1" });
    try std.testing.expectEqualStrings("1.5.0 · server 1.4.1 · 1.5.1 available", h.version);
    try std.testing.expectEqualStrings(" · 1.5.1 available", h.version_tail);
}

test "silent: a sushi that listens and does not answer is Not answering, with Stop offered" {
    try std.testing.expectEqualStrings("Not answering", pillFor(.{ .link = .silent, .port = 1 }).label);
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("Port 18801: not answering", statusLine(&b, .{ .link = .silent, .port = 18801, .sushi_listening = true }));
    const t = toggle(.{ .link = .silent, .sushi_listening = true }, false);
    try std.testing.expect(t.action == .stop and t.enabled);
    try std.testing.expect(!enabled(.{ .link = .silent, .sushi_listening = true }).start);
    try std.testing.expect(!openEnabled(.silent));
    try std.testing.expectEqual(Icon.outline, iconFor(.silent));
}

test "status: the line names the port, and a port held by another program blocks Start" {
    var b: [128]u8 = undefined;
    try std.testing.expectEqualStrings("Port 18801: not running", statusLine(&b, .{ .link = .offline, .port = 18801 }));
    try std.testing.expectEqualStrings("Port 18801: Qwen-X", statusLine(&b, .{ .link = .up, .port = 18801, .model = "Qwen-X" }));
    try std.testing.expectEqualStrings("Port 18801: in use by another program", statusLine(&b, .{ .link = .offline, .port = 18801, .port_taken = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .offline, .port_taken = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .up, .sushi_listening = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .unauthorized, .sushi_listening = true }));
}

test "update: the version comes from the second word of the first --version line" {
    try std.testing.expectEqualStrings("1.5.0", parseVersionOutput("sushi 1.5.0\nmlx 0.32.3\n").?);
    try std.testing.expectEqualStrings("1.5.0-dev1", parseVersionOutput("sushi 1.5.0-dev1\n").?);
    for ([_][]const u8{ "", "sushi", "\n" }) |bad| try std.testing.expect(parseVersionOutput(bad) == null);
    try std.testing.expect(parseVersionOutput("dyld: Library not loaded: x") == null);
    try std.testing.expectEqualStrings("1.5.0", parseVersionOutput("[mem] MLX buffer-pool cap 8192 MB\n[mem] GPU floor 1 MB\nsushi 1.5.0\nmlx 0.32.3\n").?);
}

test "update: sushi update --check output is available, up to date or a failure" {
    const newer = parseCheck("sushi 1.6.0 is available (this is 1.5.0): run `sushi update`\n  https://example/releases/v1.6.0\n");
    try std.testing.expectEqualStrings("1.6.0", newer.available);
    try std.testing.expectEqualStrings("1.6.0", parseCheck("sushi 1.6.0 is available (this is 1.5.0): run `brew upgrade sushi`\n").available);
    try std.testing.expect(parseCheck("sushi 1.5.0 is up to date (latest release 1.5.0)\n") == .up_to_date);
    const logged = parseCheck("[mem] MLX buffer-pool cap 8192 MB\n── sushi 1.5.0 update, 2026-11-02 UTC\nsushi 1.6.0 is available (this is 1.5.0): run `sushi update`\n");
    try std.testing.expectEqualStrings("1.6.0", logged.available);
    const bad = parseCheck("[mem] x\nsushi update: cannot reach GitHub\n");
    try std.testing.expectEqualStrings("sushi update: cannot reach GitHub", bad.failed);
    try std.testing.expectEqualStrings("no answer", parseCheck("").failed);
}

test "update: /props carries the server's version and the update object" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body =
        \\{"status":"ok","update":{"current":"1.4.0","latest":"1.6.0","available":true,"checked_at":1,"url":"u","error":null,"command":"brew upgrade sushi"}}
    ;
    const u = (try parseProps(arena.allocator(), body)).?;
    try std.testing.expectEqualStrings("1.4.0", u.current);
    try std.testing.expectEqualStrings("1.6.0", u.latest.?);
    try std.testing.expect(u.available);
    try std.testing.expectEqualStrings("brew upgrade sushi", u.command.?);
    try std.testing.expect(u.err == null);
    const none = (try parseProps(arena.allocator(), "{\"update\":{\"current\":\"1\",\"latest\":null,\"available\":false,\"error\":\"disk full\"}}")).?;
    try std.testing.expect(none.latest == null and !none.available);
    try std.testing.expectEqualStrings("disk full", none.err.?);
    try std.testing.expect((try parseProps(arena.allocator(), "{\"status\":\"ok\"}")) == null);
}

test "update: a Homebrew keg is brew's" {
    try std.testing.expect(isBrewKeg("/opt/homebrew/Cellar/sushi/1.5.0/libexec/sushi"));
    try std.testing.expect(!isBrewKeg("/opt/homebrew/Cellar/sushi-nightly/1.5.0/libexec/sushi"));
    try std.testing.expect(!isBrewKeg("/Users/me/sushi-macos-arm64/sushi"));
}

test "update: a keg path maps to brew's opt link, which survives an upgrade" {
    var b: [128]u8 = undefined;
    try std.testing.expectEqualStrings("/opt/homebrew/opt/sushi/libexec/sushi", brewOptPath(&b, "/opt/homebrew/Cellar/sushi/1.4.1/libexec/sushibar", "sushi").?);
    try std.testing.expect(brewOptPath(&b, "/usr/local/bin/sushibar", "sushi") == null);
}

test "update: which path an update takes" {
    const up: UpdateAsk = .{ .available = true, .link = .up };
    try std.testing.expectEqual(UpdatePath.none, updatePath(.{ .available = false, .link = .up }));
    try std.testing.expectEqual(UpdatePath.server, updatePath(up));
    try std.testing.expectEqual(UpdatePath.server, updatePath(.{ .available = true, .link = .unauthorized }));
    try std.testing.expectEqual(UpdatePath.server, updatePath(.{ .available = true, .link = .metrics_off }));
    try std.testing.expectEqual(UpdatePath.cli, updatePath(.{ .available = true, .link = .offline }));
    try std.testing.expectEqual(UpdatePath.brew, updatePath(.{ .available = true, .link = .up, .server_command = true }));
    try std.testing.expectEqual(UpdatePath.brew, updatePath(.{ .available = true, .link = .offline, .sibling_brew = true }));
    try std.testing.expectEqual(UpdatePath.none, updatePath(.{ .available = false, .link = .offline, .sibling_brew = true }));
}

test "update: the daily check is due once a day, an hour after a failure, and not twice" {
    const h: i64 = 3600 * 1000;
    try std.testing.expect(checkDue(0, 5 * h, true));
    try std.testing.expect(!checkDue(10 * h, 11 * h, true));
    try std.testing.expect(!checkDue(10 * h, 33 * h, true));
    try std.testing.expect(checkDue(10 * h, 34 * h, true));
    try std.testing.expect(!checkDue(10 * h, 10 * h + h / 2, false));
    try std.testing.expect(checkDue(10 * h, 11 * h, false));
}

test "update: the bar's automatic check follows server.update_check, on unless the file says false" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try std.testing.expect(autoCheckOn(a, "{\"server\":{\"update_check\":true}}"));
    try std.testing.expect(!autoCheckOn(a, "{\"server\":{\"update_check\":false,\"pack_update_check\":true}}"));
    try std.testing.expect(autoCheckOn(a, "{\"server\":{\"pack_update_check\":false}}"));
    try std.testing.expect(autoCheckOn(a, "{\"/m/a\":{\"mtp\":false}}"));
    try std.testing.expect(autoCheckOn(a, ""));
    try std.testing.expect(autoCheckOn(a, "[1,"));
    try std.testing.expect(autoCheckOn(a, "{\"server\":{\"update_check\":\"no\"}}"));
}

test "update: a cache under a day old answers the bar's check, an older or missing one leaves it to sushi" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const newer = "{\"checked_at\":1000,\"latest\":\"1.6.0\",\"url\":\"u\",\"etag\":\"e\",\"error\":\"\"}";
    try std.testing.expectEqualStrings("1.6.0", cachedCheck(a, newer, 1000 + 3600, "1.5.0").?.available);
    try std.testing.expect(cachedCheck(a, newer, 1000 + 3600, "1.6.0").? == .up_to_date);
    try std.testing.expect(cachedCheck(a, newer, 1000 + 3600, "1.7.0").? == .up_to_date);
    try std.testing.expect(cachedCheck(a, newer, 1000 + 24 * 3600, "1.5.0") == null);
    try std.testing.expect(cachedCheck(a, newer, 999, "1.5.0") == null);
    try std.testing.expect(cachedCheck(a, "{}", 5000, "1.5.0") == null);
    try std.testing.expect(cachedCheck(a, "", 5000, "1.5.0") == null);
    try std.testing.expect(cachedCheck(a, "{\"checked_at\":1000,\"latest\":\"\"}", 2000, "1.5.0").? == .up_to_date);
}

test "update: sushibar relaunches only when the sushi beside it changed version" {
    try std.testing.expect(relaunchNeeded("1.5.0", "1.6.0"));
    try std.testing.expect(!relaunchNeeded("1.5.0", "1.5.0"));
    try std.testing.expect(!relaunchNeeded("1.5.0", ""));
    try std.testing.expect(!relaunchNeeded("", "1.6.0"));
}

test "update menu: a dot on the button while an update waits" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("•", title(&b, .{ .link = .up, .update_waiting = true }));
    try std.testing.expectEqualStrings("88.0 tok/s •", title(&b, .{ .link = .up, .last = 88, .update_waiting = true }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .offline, .update_waiting = false }));
}

test "header: the pill names the server's state" {
    const pill = struct {
        fn of(in: HeaderIn) Pill {
            return pillFor(in);
        }
    }.of;
    try std.testing.expectEqualStrings("Running", pill(.{ .link = .up, .port = 1, .model = "m" }).label);
    try std.testing.expectEqual(Tone.green, pill(.{ .link = .up, .port = 1, .model = "m" }).tone);
    try std.testing.expectEqualStrings("Loading", pill(.{ .link = .up, .port = 1, .model = "m", .loading = true }).label);
    try std.testing.expectEqualStrings("Starting", pill(.{ .link = .offline, .port = 1, .starting = true }).label);
    try std.testing.expectEqualStrings("Stopped", pill(.{ .link = .offline, .port = 1 }).label);
    try std.testing.expectEqual(Tone.grey, pill(.{ .link = .offline, .port = 1 }).tone);
    try std.testing.expectEqualStrings("Key needed", pill(.{ .link = .unauthorized, .port = 1 }).label);
    try std.testing.expectEqual(Tone.orange, pill(.{ .link = .unauthorized, .port = 1 }).tone);
    try std.testing.expectEqualStrings("Metrics off", pill(.{ .link = .metrics_off, .port = 1 }).label);
    try std.testing.expectEqualStrings("Updating…", pill(.{ .link = .up, .port = 1, .model = "m", .updating = true }).label);
    try std.testing.expectEqual(Tone.orange, pill(.{ .link = .up, .port = 1, .updating = true }).tone);
    try std.testing.expectEqualStrings("No model", pill(.{ .link = .up, .port = 1 }).label);
    try std.testing.expectEqual(Tone.grey, pill(.{ .link = .up, .port = 1 }).tone);
    try std.testing.expectEqualStrings("Connecting…", pill(.{ .link = .offline, .port = 1, .connecting = true }).label);
    try std.testing.expectEqual(Tone.grey, pill(.{ .link = .offline, .port = 1, .connecting = true }).tone);
}

test "header: Start being off says why, in an orange pill and a line of its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const taken = try buildHeader(a, .{ .link = .offline, .port = 8080, .port_taken = true });
    try std.testing.expectEqualStrings("Port in use", taken.pill.label);
    try std.testing.expectEqual(Tone.orange, taken.pill.tone);
    try std.testing.expectEqualStrings("Another program holds this port", taken.line1);
    try std.testing.expectEqualStrings("Port 8080 · pick another in the Port menu", taken.line2);
    const nobin = try buildHeader(a, .{ .link = .offline, .port = 8080, .sushi_found = false });
    try std.testing.expectEqualStrings("No sushi", nobin.pill.label);
    try std.testing.expectEqual(Tone.orange, nobin.pill.tone);
    try std.testing.expectEqualStrings("sushi is not beside sushibar", nobin.line1);
    const both = try buildHeader(a, .{ .link = .offline, .port = 8080, .port_taken = true, .sushi_found = false });
    try std.testing.expectEqualStrings("Port in use", both.pill.label);
    const ours = try buildHeader(a, .{ .link = .offline, .port = 8080, .starting = true, .port_taken = true });
    try std.testing.expectEqualStrings("Starting", ours.pill.label);
}

test "header: the lines say the cause or the next step, never the pill again" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const states = [_]HeaderIn{
        .{ .link = .offline, .port = 1 },
        .{ .link = .offline, .port = 1, .starting = true },
        .{ .link = .silent, .port = 1 },
        .{ .link = .unauthorized, .port = 1 },
        .{ .link = .up, .port = 1 },
        .{ .link = .up, .port = 1, .loading = true, .model = "m" },
        .{ .link = .offline, .port = 1, .connecting = true },
    };
    for (states) |in| {
        const h = try buildHeader(a, in);
        try std.testing.expect(!std.mem.eql(u8, h.pill.label, h.line1));
        try std.testing.expect(std.mem.indexOf(u8, h.line2, h.pill.label) == null);
    }
    const ready = try buildHeader(a, .{ .link = .offline, .port = 18801 });
    try std.testing.expectEqualStrings("Ready to start", ready.line1);
    try std.testing.expectEqualStrings("Port 18801", ready.line2);
    const again = try buildHeader(a, .{ .link = .offline, .port = 18801, .last_model = "Qwen-X" });
    try std.testing.expectEqualStrings("Port 18801 · loads Qwen-X", again.line2);
    const loading = try buildHeader(a, .{ .link = .up, .port = 18801, .loading = true, .model = "Qwen-X", .mem_mb = 2048 });
    try std.testing.expectEqualStrings("Port 18801 · 2.0 GB in use", loading.line2);
    const connecting = try buildHeader(a, .{ .link = .offline, .port = 18801, .connecting = true });
    try std.testing.expectEqualStrings("Looking for a server", connecting.line1);
    try std.testing.expect(!connecting.color_icon);
}

test "header: a server with no model points at the Models page" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const none = try buildHeader(a, .{ .link = .up, .port = 18801, .last_model = "Qwen-X" });
    try std.testing.expectEqualStrings("No model", none.pill.label);
    try std.testing.expectEqualStrings("Open Models to load one", none.line1);
    try std.testing.expect(none.line1_models and !none.show_stats);
    const model = try buildHeader(a, .{ .link = .up, .port = 18801, .model = "Qwen-X" });
    try std.testing.expect(!model.line1_models);
    const off = try buildHeader(a, .{ .link = .offline, .port = 18801 });
    try std.testing.expect(!off.line1_models);
}

test "connecting: the status names it and Start waits for the first poll" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("Port 18801: connecting…", statusLine(&b, .{ .link = .offline, .port = 18801, .connecting = true }));
    try std.testing.expect(!enabled(.{ .link = .offline, .connecting = true }).start);
    try std.testing.expect(!toggle(.{ .link = .offline, .connecting = true }, false).enabled);
    try std.testing.expect(toggle(.{ .link = .offline }, false).enabled);
}

test "header: thousands separators" {
    var b: [16]u8 = undefined;
    try std.testing.expectEqualStrings("0", thousands(&b, 0));
    try std.testing.expectEqualStrings("999", thousands(&b, 999));
    try std.testing.expectEqualStrings("5,844", thousands(&b, 5844));
    try std.testing.expectEqualStrings("1,195", thousands(&b, 1195));
    try std.testing.expectEqualStrings("1,234,567", thousands(&b, 1234567));
}

test "header: a running model with throughput" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const h = try buildHeader(arena.allocator(), .{
        .link = .up, .port = 12345, .model = "DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", .mem_mb = 49971,
        .bar_version = "1.5.0-dev1",
        .decoding = true, .live = 105.27, .last = 90, .last_prefill = 5843.6, .gpu_mhz = 1195, .gpu_pct = 62,
    });
    try std.testing.expect(h.color_icon);
    try std.testing.expectEqualStrings("1.5.0-dev1", h.version);
    try std.testing.expectEqualStrings("Running", h.pill.label);
    try std.testing.expectEqualStrings("DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", h.line1);
    try std.testing.expectEqualStrings("Port 12345 · 48.8 GB in use", h.line2);
    try std.testing.expect(h.show_stats);
    try std.testing.expectEqualStrings("105", h.stats[0].value);
    try std.testing.expectEqualStrings("DECODE TOK/S", h.stats[0].label);
    try std.testing.expectEqualStrings("5,844", h.stats[1].value);
    try std.testing.expectEqualStrings("PREFILL TOK/S", h.stats[1].label);
    try std.testing.expectEqualStrings("62%", h.stats[2].value);
    try std.testing.expectEqualStrings("GPU · 1,195 MHZ", h.stats[2].label);
    try std.testing.expectEqualStrings("sushi running on port 12345, DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw, decode 105 tokens per second", h.a11y);
}

test "header: idle shows the last request, gaps show a dash, an update is named" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const h = try buildHeader(arena.allocator(), .{
        .link = .up, .port = 12345, .model = "m", .bar_version = "1.5.0", .update_available = true, .latest = "1.5.1",
        .last = 88.04,
    });
    try std.testing.expectEqualStrings("1.5.0 · 1.5.1 available", h.version);
    try std.testing.expectEqualStrings("88.0", h.stats[0].value);
    try std.testing.expectEqualStrings("LAST TOK/S", h.stats[0].label);
    try std.testing.expectEqualStrings("–", h.stats[1].value);
    try std.testing.expectEqualStrings("–", h.stats[2].value);
    try std.testing.expectEqualStrings("GPU MHZ", h.stats[2].label);
    try std.testing.expectEqualStrings("Port 12345", h.line2);
    const mhz = try buildHeader(arena.allocator(), .{ .link = .up, .port = 1, .model = "m", .gpu_mhz = 1380 });
    try std.testing.expectEqualStrings("1,380", mhz.stats[2].value);
    const pct = try buildHeader(arena.allocator(), .{ .link = .up, .port = 1, .model = "m", .gpu_mhz = 0, .gpu_pct = 8 });
    try std.testing.expectEqualStrings("8%", pct.stats[2].value);
    try std.testing.expectEqualStrings("GPU", pct.stats[2].label);
}

test "sparkline: the caption states the scale it is drawn to" {
    var b: [64]u8 = undefined;
    const pts = [_]SparkPoint{ .{ .x = 0.1, .v = 90.4, .gap = true }, .{ .x = 0.5, .v = 124.6, .gap = false }, .{ .x = 0.9, .v = 100, .gap = false } };
    try std.testing.expectEqualStrings("peak 125 tok/s · last 5 min", sparkCaption(&b, &pts, 300_000));
    try std.testing.expectEqualStrings("", sparkCaption(&b, pts[0..1], 300_000));
    try std.testing.expectEqualStrings("", sparkCaption(&b, &.{}, 300_000));
}

test "header: no model, stopped, key needed, loading" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const none = try buildHeader(a, .{ .link = .up, .port = 12345, .bar_version = "1.5.0" });
    try std.testing.expectEqualStrings("Port 12345", none.line2);
    try std.testing.expect(!none.show_stats);
    const stale = try buildHeader(a, .{ .link = .metrics_off, .port = 12345 });
    try std.testing.expect(!stale.show_stats);
    const off = try buildHeader(a, .{ .link = .offline, .port = 18801, .bar_version = "1.5.0" });
    try std.testing.expect(!off.color_icon and !off.show_stats);
    try std.testing.expectEqualStrings("Ready to start", off.line1);
    try std.testing.expectEqualStrings("Port 18801", off.line2);
    try std.testing.expectEqualStrings("sushi stopped, port 18801", off.a11y);
    const key = try buildHeader(a, .{ .link = .unauthorized, .port = 12345 });
    try std.testing.expectEqualStrings("The server requires an API key", key.line1);
    try std.testing.expect(key.color_icon and !key.show_stats);
    const load = try buildHeader(a, .{ .link = .up, .loading = true, .model = "Qwen", .port = 12345 });
    try std.testing.expectEqualStrings("Qwen", load.line1);
    try std.testing.expectEqualStrings("Port 12345", load.line2);
}

test "sparkline: rates between samples, gaps where the server was down or restarted" {
    const win: i64 = 300_000;
    const now: i64 = 1_000_000;
    const h = [_]Hist{
        .{ .t_ms = now - 10_000, .gen = 100 }, .{ .t_ms = now - 8_000, .gen = 300 }, .{ .t_ms = now - 6_000, .gen = 300 },
        .{ .t_ms = now - 2_000, .gen = 50 }, // counter went back: a restart
        .{ .t_ms = now, .gen = 150 },
    };
    var out: [8]SparkPoint = undefined;
    const pts = sparkPoints(&out, &h, now, win);
    try std.testing.expectEqual(@as(usize, 4), pts.len);
    try std.testing.expectApproxEqAbs(@as(f32, 100), pts[0].v, 1e-3);
    try std.testing.expect(pts[0].gap);
    try std.testing.expectApproxEqAbs(@as(f32, 0), pts[1].v, 1e-3);
    try std.testing.expect(!pts[1].gap);
    try std.testing.expect(pts[2].gap);
    try std.testing.expectApproxEqAbs(@as(f32, 50), pts[3].v, 1e-3);
    try std.testing.expect(!pts[3].gap);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pts[3].x, 1e-6);
    try std.testing.expect(pts[0].x > 0.97 and pts[0].x < 0.98);
    const old = [_]Hist{ .{ .t_ms = now - 400_000, .gen = 1 }, .{ .t_ms = now - 398_000, .gen = 9 } };
    try std.testing.expectEqual(@as(usize, 0), sparkPoints(&out, &old, now, win).len);
    const down = [_]Hist{ .{ .t_ms = now - 60_000, .gen = 10 }, .{ .t_ms = now - 58_000, .gen = 30 }, .{ .t_ms = now - 20_000, .gen = 90 } };
    const g = sparkPoints(&out, &down, now, win);
    try std.testing.expect(g.len == 2 and g[1].gap);
}

test "sparkline plot: the window maps onto the inner width, nothing leaves it, a gap splits the path" {
    const win: i64 = 300_000;
    const now: i64 = 1_000_000;
    const h = [_]Hist{
        .{ .t_ms = now - win - 2_000, .gen = 0 }, .{ .t_ms = now - win, .gen = 200 }, .{ .t_ms = now - 150_000, .gen = 20_000 },
        .{ .t_ms = now - 149_000, .gen = 20_100 }, .{ .t_ms = now - 100_000, .gen = 30_000 }, .{ .t_ms = now, .gen = 30_200 },
    };
    var sp: [8]SparkPoint = undefined;
    const pts = sparkPoints(&sp, &h, now, win);
    var pl: [8]PlotPoint = undefined;
    const plot = plotSpark(&pl, pts, 12, 276, 4, 28);
    try std.testing.expectEqual(pts.len, plot.len);
    try std.testing.expectApproxEqAbs(@as(f64, 12), plot[0].x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 288), plot[plot.len - 1].x, 1e-6);
    for (plot) |p| {
        try std.testing.expect(p.x >= 12 and p.x <= 288);
        try std.testing.expect(p.y >= 4 and p.y <= 28);
    }
    try std.testing.expect(plot[1].gap and !plot[2].gap and plot[3].gap);
    var peak: f64 = 28;
    for (plot) |p| peak = @min(peak, p.y);
    try std.testing.expect(peak > 4 and peak < 8);
}

test "header: an available update names its version in a tail of its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const up = try buildHeader(a, .{ .link = .up, .port = 1, .bar_version = "1.5.0", .update_available = true, .latest = "1.5.1" });
    try std.testing.expectEqualStrings("1.5.0 · 1.5.1 available", up.version);
    try std.testing.expectEqualStrings(" · 1.5.1 available", up.version_tail);
    const same = try buildHeader(a, .{ .link = .up, .port = 1, .bar_version = "1.5.0" });
    try std.testing.expectEqualStrings("", same.version_tail);
}

test "sparkline: /metrics.json history is read by field name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const body =
        \\{"gauges":{"memory_mb":50000,"gpu_clock_mhz":1195,"gpu_utilization_pct":62,"requests_prefilling":0},"sessions":[],
        \\"history":{"interval_ms":2000,"capacity":1800,"started_ms":1,"fields":["t_ms","gen_tokens","prefill_tokens","gpu_mhz"],
        \\"samples":[[1700000000000,10,0,900],[1700000002000,50,0,1000]]},"requests":[
        \\{"t_ms":5,"prompt_tokens":600,"cached_tokens":0,"prefill_seconds":0.5,"generated_tokens":10,"decode_seconds":1.0,"outcome":"ok"}]}
    ;
    const s = try parseMetrics(arena.allocator(), body);
    try std.testing.expectEqual(@as(?u64, 50000), s.mem_mb);
    try std.testing.expectEqual(@as(?u64, 1195), s.gpu_mhz);
    try std.testing.expectEqual(@as(?u64, 62), s.gpu_pct);
    try std.testing.expectEqual(@as(usize, 2), s.hist.len);
    try std.testing.expectEqual(@as(i64, 1700000002000), s.hist[1].t_ms);
    try std.testing.expectEqual(@as(u64, 50), s.hist[1].gen);
    try std.testing.expectApproxEqAbs(@as(f64, 1200), s.finished_prefill_tps.?, 1e-6);
    const bare = try parseMetrics(arena.allocator(), "{\"gauges\":{}}");
    try std.testing.expect(bare.mem_mb == null and bare.gpu_mhz == null and bare.hist.len == 0);
}

test "models: ready and loading are told apart" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ready = try parseModelState(a, "{\"data\":[{\"id\":\"a\",\"state\":\"unloaded\"},{\"id\":\"b\",\"state\":\"ready\"}]}");
    try std.testing.expectEqualStrings("b", ready.ready.?);
    try std.testing.expect(ready.loading == null);
    const loading = try parseModelState(a, "{\"data\":[{\"id\":\"c\",\"state\":\"loading\"}]}");
    try std.testing.expect(loading.ready == null);
    try std.testing.expectEqualStrings("c", loading.loading.?);
}

test "pack update: the ready model's meta.pack_update is read, a missing or other model's is not" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const up = "{\"current\":\"1.0\",\"latest\":\"2.0\",\"command\":\"sushi pull example-pack\"}";
    const has = try parseModelState(a, "{\"data\":[{\"id\":\"a\",\"state\":\"unloaded\",\"meta\":{\"pack_update\":" ++ up ++ "}},{\"id\":\"b\",\"state\":\"ready\",\"meta\":{\"pack_update\":" ++ up ++ "}}]}");
    try std.testing.expectEqualStrings("2.0", has.pack_update.?.latest);
    try std.testing.expectEqualStrings("1.0", has.pack_update.?.current);
    try std.testing.expectEqualStrings("sushi pull example-pack", has.pack_update.?.command);
    const none = try parseModelState(a, "{\"data\":[{\"id\":\"b\",\"state\":\"ready\",\"meta\":{\"quantization\":\"x\"}}]}");
    try std.testing.expect(none.pack_update == null);
    const other = try parseModelState(a, "{\"data\":[{\"id\":\"a\",\"state\":\"unloaded\",\"meta\":{\"pack_update\":" ++ up ++ "}},{\"id\":\"b\",\"state\":\"ready\"}]}");
    try std.testing.expect(other.pack_update == null);
    const empty = try parseModelState(a, "{\"data\":[{\"id\":\"b\",\"state\":\"ready\",\"meta\":{\"pack_update\":{\"latest\":\"\"}}}]}");
    try std.testing.expect(empty.pack_update == null);
}

test "pack update: the card names it under a loaded model, the menu offers its command" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pu: PackUpdate = .{ .current = "1.0", .latest = "2.0", .command = "sushi pull example-pack" };
    const h = try buildHeader(a, .{ .link = .up, .port = 1, .model = "m", .pack_update = pu });
    try std.testing.expectEqualStrings("Pack 2.0 available (you have 1.0)", h.pack_line);
    try std.testing.expectEqualStrings("", (try buildHeader(a, .{ .link = .up, .port = 1, .model = "m" })).pack_line);
    try std.testing.expectEqualStrings("", (try buildHeader(a, .{ .link = .offline, .port = 1, .model = "m", .pack_update = pu })).pack_line);
    try std.testing.expectEqualStrings("", (try buildHeader(a, .{ .link = .up, .port = 1, .model = "m", .loading = true, .pack_update = pu })).pack_line);
    try std.testing.expectEqualStrings("", (try buildHeader(a, .{ .link = .up, .port = 1, .pack_update = pu })).pack_line);
    var b: [96]u8 = undefined;
    try std.testing.expectEqualStrings("Copy “sushi pull example-pack”", copyLabel(&b, pu.command));
}

test "menu: the one server toggle" {
    const t0 = toggle(.{ .link = .offline }, false);
    try std.testing.expectEqualStrings("Start Server", t0.label);
    try std.testing.expect(t0.enabled and t0.action == .start);
    try std.testing.expectEqualStrings("play.fill", t0.symbol);
    const ours = toggle(.{ .link = .up, .ours_alive = true }, false);
    try std.testing.expectEqualStrings("Stop Server", ours.label);
    try std.testing.expect(ours.enabled and ours.action == .stop);
    try std.testing.expectEqualStrings("stop.fill", ours.symbol);
    const foreign = toggle(.{ .link = .up, .sushi_listening = true }, false);
    try std.testing.expectEqualStrings("Stop Server…", foreign.label);
    const starting = toggle(.{ .link = .offline, .ours_alive = true }, false);
    try std.testing.expectEqualStrings("Starting…", starting.label);
    try std.testing.expect(!starting.enabled and starting.action == .none);
    const stopping = toggle(.{ .link = .up, .ours_alive = true }, true);
    try std.testing.expectEqualStrings("Stopping…", stopping.label);
    try std.testing.expect(!stopping.enabled);
    const taken = toggle(.{ .link = .offline, .port_taken = true }, false);
    try std.testing.expect(!taken.enabled and taken.action == .none);
    const nobin = toggle(.{ .link = .offline, .sushi_found = false }, false);
    try std.testing.expect(!nobin.enabled);
    try std.testing.expect(openEnabled(.up) and openEnabled(.unauthorized) and openEnabled(.metrics_off) and !openEnabled(.offline));
}

test "menu: update group labels" {
    var b: [96]u8 = undefined;
    try std.testing.expectEqualStrings("Update to sushi 1.5.1", updateLabel(&b, .{ .available = true, .latest = "1.5.1", .path = .server }).?);
    try std.testing.expect(updateLabel(&b, .{}) == null);
    try std.testing.expectEqualStrings("Copy “brew upgrade sushi”", copyLabel(&b, "brew upgrade sushi"));
}

test "title: the last request's speed is dim and drops the word last" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("88.0 tok/s", title(&b, .{ .link = .up, .last = 88 }));
    try std.testing.expect(titleDim(.{ .link = .up, .last = 88 }));
    try std.testing.expect(!titleDim(.{ .link = .up, .decoding = true, .live = 5, .last = 88 }));
    try std.testing.expect(!titleDim(.{ .link = .up }));
    try std.testing.expect(!titleDim(.{ .link = .offline, .last = 4 }));
}

test "args: no argument runs on the server's default port" {
    try std.testing.expectEqual(Command{ .run = null }, parseArgs(&.{}));
}

test "args: --port takes a value or an =form, --help wins" {
    try std.testing.expectEqual(Command{ .run = @as(?u16, 18801) }, parseArgs(&.{ "--port", "18801" }));
    try std.testing.expectEqual(Command{ .run = @as(?u16, 18802) }, parseArgs(&.{"--port=18802"}));
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

test "metrics: prefill speed counts only the tokens forwarded, of finished requests" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const warm =
        \\{"requests":[
        \\{"t_ms":100,"prompt_tokens":1000,"cached_tokens":900,"prefill_seconds":1.0,"generated_tokens":5,"decode_seconds":1,"outcome":"ok"},
        \\{"t_ms":200,"prompt_tokens":5000,"cached_tokens":0,"prefill_seconds":0.1,"outcome":"cancelled"}]}
    ;
    const s = try parseMetrics(arena.allocator(), warm);
    try std.testing.expectApproxEqAbs(@as(f64, 100), s.finished_prefill_tps.?, 1e-9);
    const all_cached =
        \\{"requests":[{"t_ms":100,"prompt_tokens":1000,"cached_tokens":1000,"prefill_seconds":0.01,"outcome":"ok"}]}
    ;
    try std.testing.expect((try parseMetrics(arena.allocator(), all_cached)).finished_prefill_tps == null);
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
    for ([_]Link{ .up, .unauthorized, .metrics_off }) |l| try std.testing.expectEqual(Icon.filled, iconFor(l));
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
    try std.testing.expectEqualStrings("88.0 tok/s", title(&b, .{ .link = .up, .last = 88 }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .up }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .offline, .last = 5, .decoding = true }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .unauthorized, .last = 5 }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .metrics_off }));
    try std.testing.expectEqualStrings("", title(&b, .{ .link = .silent }));
}

test "menu: Start and Stop follow the server state" {
    var b: [128]u8 = undefined;
    try std.testing.expectEqual(Enabled{ .start = true, .stop = false }, enabled(.{ .link = .offline }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .offline, .sushi_found = false }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .up, .ours_alive = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = true }, enabled(.{ .link = .offline, .ours_alive = true }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .up }));
    try std.testing.expectEqual(Enabled{ .start = false, .stop = false }, enabled(.{ .link = .unauthorized }));
    try std.testing.expectEqualStrings("Port 12345: not running", statusLine(&b, .{ .link = .offline }));
    try std.testing.expect(std.mem.indexOf(u8, statusLine(&b, .{ .link = .offline, .sushi_found = false }), "not found") != null);
    try std.testing.expectEqualStrings("Port 12345: Qwen-X", statusLine(&b, .{ .link = .up, .model = "Qwen-X" }));
    try std.testing.expectEqualStrings("Port 12345: running, no model loaded", statusLine(&b, .{ .link = .up }));
    try std.testing.expectEqualStrings("Port 12345: running (API key required)", statusLine(&b, .{ .link = .unauthorized }));
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
    return procargsEnv(argv, &.{"HOME=/x"});
}

fn procargsEnv(comptime argv: []const []const u8, comptime envp: []const []const u8) []const u8 {
    comptime var out: []const u8 = &std.mem.toBytes(@as(i32, argv.len)) ++ "/bin/" ++ argv[0] ++ "\x00\x00\x00";
    inline for (argv) |a| out = out ++ a ++ "\x00";
    inline for (envp) |e| out = out ++ e ++ "\x00";
    return out;
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

test "update button: Check when idle, Update (orange) when a release waits, and what each runs" {
    var b: [96]u8 = undefined;
    const idle = updateButton(&b, .{});
    try std.testing.expectEqual(UpdateState.idle, idle.state);
    try std.testing.expectEqualStrings("Check", idle.label);
    try std.testing.expectEqual(ButtonAction.check, idle.action);
    try std.testing.expect(!idle.accent and idle.enabled and !idle.closes_menu and !idle.spinner);
    try std.testing.expectEqualStrings("Check for Updates", idle.tip);

    const waiting = updateButton(&b, .{ .available = true, .latest = "1.5.1", .path = .server });
    try std.testing.expectEqual(UpdateState.available, waiting.state);
    try std.testing.expectEqualStrings("Update", waiting.label);
    try std.testing.expectEqual(ButtonAction.update, waiting.action);
    try std.testing.expect(waiting.accent and waiting.enabled and !waiting.closes_menu);
    try std.testing.expectEqualStrings("Update to sushi 1.5.1", waiting.tip);

    const cli = updateButton(&b, .{ .available = true, .latest = "1.5.1", .path = .cli });
    try std.testing.expectEqual(ButtonAction.update, cli.action);
}

test "update button: a click on Check waits for the answer, which is Up to date, Update or Retry" {
    var b: [96]u8 = undefined;
    const checking = updateButton(&b, .{ .busy = .checking });
    try std.testing.expectEqual(UpdateState.checking, checking.state);
    try std.testing.expectEqualStrings("Checking…", checking.label);
    try std.testing.expect(!checking.enabled and checking.spinner and !checking.accent and checking.action == .none);

    const current = updateButton(&b, .{ .checked = .up_to_date });
    try std.testing.expectEqual(UpdateState.up_to_date, current.state);
    try std.testing.expectEqualStrings("Up to date", current.label);
    try std.testing.expect(!current.enabled and !current.accent and !current.spinner and current.action == .none);

    const newer = updateButton(&b, .{ .checked = .none, .available = true, .latest = "1.5.1", .path = .cli });
    try std.testing.expect(newer.state == .available and newer.accent and newer.enabled);

    const failed = updateButton(&b, .{ .checked = .failed });
    try std.testing.expectEqual(UpdateState.failed, failed.state);
    try std.testing.expectEqualStrings("Retry", failed.label);
    try std.testing.expect(failed.enabled and !failed.accent and failed.action == .check and !failed.closes_menu);
}

test "update button: a release found beats an older answer, a busy check beats everything but an update" {
    var b: [96]u8 = undefined;
    const stale = updateButton(&b, .{ .checked = .up_to_date, .available = true, .latest = "1.5.1", .path = .server });
    try std.testing.expectEqual(UpdateState.available, stale.state);
    const again = updateButton(&b, .{ .checked = .failed, .busy = .checking });
    try std.testing.expectEqual(UpdateState.checking, again.state);
}

test "update button: a Homebrew update is one Upgrade button that copies its command and says Copied, an update in progress waits" {
    var b: [96]u8 = undefined;
    const brew = updateButton(&b, .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi" });
    try std.testing.expectEqual(UpdateState.brew_copy, brew.state);
    try std.testing.expectEqualStrings("Upgrade", brew.label);
    try std.testing.expectEqual(ButtonAction.copy_command, brew.action);
    try std.testing.expect(brew.accent and brew.enabled and !brew.closes_menu);
    try std.testing.expectEqualStrings("Copy “brew upgrade sushi”", brew.tip);

    const copied = updateButton(&b, .{ .available = true, .latest = "1.5.1", .path = .brew, .command = "brew upgrade sushi", .copied = true });
    try std.testing.expectEqual(UpdateState.copied, copied.state);
    try std.testing.expectEqualStrings("Copied", copied.label);
    try std.testing.expect(!copied.accent and copied.enabled and !copied.closes_menu);

    const updating = updateButton(&b, .{ .available = true, .path = .server, .busy = .updating });
    try std.testing.expectEqual(UpdateState.updating, updating.state);
    try std.testing.expectEqualStrings("Updating…", updating.label);
    try std.testing.expect(!updating.enabled and updating.spinner and updating.action == .none);
}

test "copied: Copied stays ten seconds, then the command is offered again" {
    try std.testing.expect(copiedNow(1_000, 5_000));
    try std.testing.expect(!copiedNow(1_000, 11_000));
    try std.testing.expect(!copiedNow(0, 1_000));
}

test "update confirm: an update through a server with a model loaded asks first and closes the menu to do it" {
    var b: [96]u8 = undefined;
    const loaded: UpdateView = .{ .available = true, .latest = "1.5.1", .path = .server, .model = "Qwen-X" };
    try std.testing.expect(updateNeedsConfirm(loaded));
    try std.testing.expect(updateButton(&b, loaded).closes_menu);
    var empty = loaded;
    empty.model = "";
    try std.testing.expect(!updateNeedsConfirm(empty) and !updateButton(&b, empty).closes_menu);
    var cli = loaded;
    cli.path = .cli;
    try std.testing.expect(!updateNeedsConfirm(cli));
    var none = loaded;
    none.available = false;
    try std.testing.expect(!updateNeedsConfirm(none));
    try std.testing.expectEqualStrings("The server on port 18801 restarts and unloads Qwen-X; a request in progress is lost.", updateConfirmInfo(&b, 18801, "Qwen-X"));
}

test "update button: an answer older than a minute is dropped when the menu opens, and Check returns" {
    try std.testing.expectEqual(CheckResult.up_to_date, shownResult(.up_to_date, 1_000, 30_000));
    try std.testing.expectEqual(CheckResult.failed, shownResult(.failed, 1_000, 30_000));
    try std.testing.expectEqual(CheckResult.none, shownResult(.up_to_date, 1_000, 61_000));
    try std.testing.expectEqual(CheckResult.none, shownResult(.none, 1_000, 2_000));
}

test "log source: the default file is per port, --log-file moves or turns it off" {
    var b: [128]u8 = undefined;
    const home = "/Users/x";
    const plain = logSource(&b, home, 11250, null);
    try std.testing.expectEqual(LogKind.standard, plain.kind);
    try std.testing.expectEqualStrings("/Users/x/.sushi/logs/sushi-11250.log", plain.path);

    const own = procargs(&.{ "/opt/sushi", "--serve", "--port", "11250" });
    try std.testing.expectEqualStrings("/Users/x/.sushi/logs/sushi-11250.log", logSource(&b, home, 11250, own).path);

    const moved = procargs(&.{ "/opt/sushi", "--serve", "--log-file", "/var/log/s.log", "--port", "9" });
    const m = logSource(&b, home, 9, moved);
    try std.testing.expectEqual(LogKind.given, m.kind);
    try std.testing.expectEqualStrings("/var/log/s.log", m.path);

    try std.testing.expectEqual(LogKind.off, logSource(&b, home, 1, procargs(&.{ "/opt/sushi", "--serve", "--log-file", "off" })).kind);
    try std.testing.expectEqual(LogKind.off, logSource(&b, home, 1, procargs(&.{ "/opt/sushi", "--log-file", "none" })).kind);

    const r = logSource(&b, home, 1, procargs(&.{ "/opt/sushi", "--log-file", "run/s.log" }));
    try std.testing.expectEqual(LogKind.relative, r.kind);
    try std.testing.expectEqualStrings("run/s.log", r.path);
}

test "log source: a --log-file with no value is not one, and the environment after argv is not read" {
    try std.testing.expect(argsLogFile(procargs(&.{ "/opt/sushi", "--serve", "--log-file" })) == null);
}

test "log notice: says which file was looked for and why there is none" {
    var b: [400]u8 = undefined;
    const std_missing = logNotice(&b, 11250, .{ .kind = .standard, .path = "/h/.sushi/logs/sushi-11250.log" });
    try std.testing.expect(std.mem.indexOf(u8, std_missing, "/h/.sushi/logs/sushi-11250.log") != null);
    try std.testing.expect(std.mem.indexOf(u8, std_missing, "--log-file") != null);
    const off = logNotice(&b, 7, .{ .kind = .off });
    try std.testing.expect(std.mem.indexOf(u8, off, "port 7") != null and std.mem.indexOf(u8, off, "--log-file off") != null);
    const rel = logNotice(&b, 7, .{ .kind = .relative, .path = "run/s.log" });
    try std.testing.expect(std.mem.indexOf(u8, rel, "run/s.log") != null and std.mem.indexOf(u8, rel, "relative") != null);
    const given = logNotice(&b, 7, .{ .kind = .given, .path = "/v/s.log" });
    try std.testing.expect(std.mem.indexOf(u8, given, "/v/s.log") != null);
}

test "log tail: the last N complete lines, whatever the file ends with" {
    const text = "a\nb\nc\nd\n";
    try std.testing.expectEqual(@as(usize, 0), tailStart(text, 10));
    try std.testing.expectEqualStrings("c\nd\n", text[tailStart(text, 2)..]);
    try std.testing.expectEqualStrings("d\n", text[tailStart(text, 1)..]);
    try std.testing.expectEqual(text.len, tailStart(text, 0));
    try std.testing.expectEqual(@as(usize, 0), tailStart("", 5));
    try std.testing.expectEqualStrings("\n\n", "x\n\n\n"[tailStart("x\n\n\n", 2)..]);
}

test "log tail: only complete lines are taken, the rest waits for its newline" {
    try std.testing.expectEqual(@as(usize, 3), completeLen("ab\nc"));
    try std.testing.expectEqual(@as(usize, 3), completeLen("ab\n"));
    try std.testing.expectEqual(@as(usize, 0), completeLen("abc"));
    try std.testing.expectEqual(@as(usize, 0), completeLen(""));
    try std.testing.expectEqual(@as(usize, 3), countLines("a\nb\nc\n"));
}

test "log follow: growth appends, a smaller or different file reloads, nothing new is idle" {
    const f: Follow = .{ .offset = 100, .file = 7 };
    try std.testing.expectEqual(FollowStep.idle, followStep(f, 100, 7));
    try std.testing.expectEqual(FollowStep.append, followStep(f, 180, 7));
    try std.testing.expectEqual(FollowStep.reload, followStep(f, 40, 7));
    try std.testing.expectEqual(FollowStep.reload, followStep(f, 400, 8));
    try std.testing.expectEqual(FollowStep.reload, followStep(.{}, 0, 3));
}

test "log trim: the oldest lines go in place once the window holds more than twice what it keeps" {
    try std.testing.expectEqual(@as(usize, 0), excessLines(2000, 2000));
    try std.testing.expectEqual(@as(usize, 0), excessLines(4000, 2000));
    try std.testing.expectEqual(@as(usize, 2001), excessLines(4001, 2000));
    try std.testing.expectEqual(@as(usize, 0), excessLines(0, 2000));
}

test "log scroll: following resumes only at the bottom" {
    try std.testing.expect(atBottom(1000, 1000));
    try std.testing.expect(atBottom(990, 1000));
    try std.testing.expect(!atBottom(600, 1000));
    try std.testing.expect(atBottom(1200, 1000));
}

test "title: the last request's speed leaves the menu bar after ten idle minutes, the card keeps it" {
    var t: Tracker = .{};
    t.observe(1_000, .{ .link = .up, .finished_tps = 42 });
    var b: [64]u8 = undefined;
    const fresh = t.view(.up, 1_000 + title_last_ms - 1);
    try std.testing.expectEqualStrings("42.0 tok/s", title(&b, fresh));
    try std.testing.expect(titleDim(fresh));
    const old = t.view(.up, 1_000 + title_last_ms);
    try std.testing.expectEqualStrings("", title(&b, old));
    try std.testing.expect(!titleDim(old) and old.last.? == 42);
    t.observe(2_000_000, .{ .link = .up, .decoding = &.{.{ .request_id = 1, .generated = 1 }} });
    t.observe(2_001_000, .{ .link = .up });
    try std.testing.expectEqualStrings("42.0 tok/s", title(&b, t.view(.up, 2_000_000 + title_last_ms - 1)));
}

test "poll cadence: a second while something moves or the menu is open, slower idle, slowest offline" {
    try std.testing.expectEqual(poll_fast_ms, pollWaitMs(.{ .link = .up, .decoding = true }));
    try std.testing.expectEqual(poll_fast_ms, pollWaitMs(.{ .link = .offline, .menu_open = true }));
    try std.testing.expectEqual(poll_fast_ms, pollWaitMs(.{ .link = .offline, .changing = true }));
    try std.testing.expectEqual(poll_idle_ms, pollWaitMs(.{ .link = .up }));
    try std.testing.expectEqual(poll_idle_ms, pollWaitMs(.{ .link = .unauthorized }));
    try std.testing.expectEqual(poll_offline_ms, pollWaitMs(.{ .link = .offline }));
    try std.testing.expectEqual(poll_offline_ms, pollWaitMs(.{ .link = .silent }));
    try std.testing.expect(poll_fast_ms < poll_idle_ms and poll_idle_ms < poll_offline_ms);
}

test "poll cadence: models are asked each poll until one is ready and while one loads, then every fifth" {
    try std.testing.expect(modelsDue(0, false, false));
    try std.testing.expect(modelsDue(1, true, true));
    try std.testing.expect(!modelsDue(1, true, false));
    try std.testing.expect(!modelsDue(models_every - 1, true, false));
    try std.testing.expect(modelsDue(models_every, true, false));
}

test "poll cadence: the process scan runs each poll until the server answers in full or while the menu is open, else every fifth" {
    try std.testing.expect(scanDue(1, .offline, false));
    try std.testing.expect(scanDue(1, .silent, false));
    try std.testing.expect(scanDue(1, .unauthorized, false));
    try std.testing.expect(scanDue(1, .up, true));
    try std.testing.expect(!scanDue(1, .up, false));
    try std.testing.expect(!scanDue(scan_every - 1, .metrics_off, false));
    try std.testing.expect(scanDue(scan_every, .up, false));
}

test "api key: the server's own --api-key, or the variable --api-key-env names in its environment, last flag wins" {
    try std.testing.expectEqualStrings("s3cret", argsApiKey(procargs(&.{ "sushi", "--serve", "--api-key", "s3cret", "--api-key-strict" })).?);
    const env_key = procargsEnv(&.{ "sushi", "--serve", "--api-key-env", "SUSHI_KEY" }, &.{ "HOME=/x", "SUSHI_KEY=from-env", "SUSHI_KEY_2=no" });
    try std.testing.expectEqualStrings("from-env", argsApiKey(env_key).?);
    try std.testing.expectEqualStrings("b", argsApiKey(procargsEnv(&.{ "sushi", "--api-key-env", "K", "--api-key", "b" }, &.{"K=a"})).?);
    try std.testing.expectEqualStrings("a", argsApiKey(procargsEnv(&.{ "sushi", "--api-key", "b", "--api-key-env", "K" }, &.{"K=a"})).?);
    try std.testing.expect(argsApiKey(procargs(&.{ "sushi", "--serve" })) == null);
    try std.testing.expect(argsApiKey(procargs(&.{ "sushi", "--api-key" })) == null);
    try std.testing.expect(argsApiKey(procargs(&.{ "sushi", "--api-key", "" })) == null);
    try std.testing.expect(argsApiKey(procargsEnv(&.{ "sushi", "--api-key-env", "MISSING" }, &.{"MISSING_NOT=x"})) == null);
    try std.testing.expect(argsApiKey(procargsEnv(&.{ "sushi", "--api-key-env", "K" }, &.{"K="})) == null);
    try std.testing.expect(argsApiKey(procargs(&.{ "sushi", "--model", "--api-key-env" })) == null);
}

test "force stop: a Stop that has not taken after its wait offers Force Stop on the pid it signalled" {
    const sent: StopSent = .{ .pid = 50, .port = 18801, .deadline_ms = 15_000 };
    try std.testing.expect(stopStuck(sent, 18801, 15_000, 50));
    try std.testing.expect(!stopStuck(sent, 18801, 14_999, 50));
    try std.testing.expect(!stopStuck(sent, 18801, 20_000, 0));
    try std.testing.expect(!stopStuck(sent, 18801, 20_000, 51));
    try std.testing.expect(!stopStuck(sent, 18802, 20_000, 50));
    try std.testing.expect(!stopStuck(.{}, 0, 20_000, 0));
    const ours = toggle(.{ .link = .up, .ours_alive = true, .stop_stuck = true }, false);
    try std.testing.expectEqualStrings("Force Stop", ours.label);
    try std.testing.expect(ours.enabled and ours.action == .force_stop);
    const silent = toggle(.{ .link = .silent, .sushi_listening = true, .stop_stuck = true }, false);
    try std.testing.expect(silent.action == .force_stop);
    try std.testing.expect(toggle(.{ .link = .offline, .ours_alive = true, .stop_stuck = true }, false).action == .force_stop);
    try std.testing.expect(toggle(.{ .link = .offline, .stop_stuck = true }, false).action == .start);
    var b: [96]u8 = undefined;
    try std.testing.expectEqualStrings("Port 18801: the server did not stop", stuckNote(&b, 18801));
    try std.testing.expect(stuckNote(&b, 65535).len <= note_title_max);
}

test "note: news goes when the menu opens a minute after it was given, a problem stays" {
    try std.testing.expect(noteKept(false, 1_000, 1_000 + note_kept_ms - 1));
    try std.testing.expect(!noteKept(false, 1_000, 1_000 + note_kept_ms));
    try std.testing.expect(noteKept(true, 1_000, 1_000 + 10 * note_kept_ms));
}

test "quit: plain Quit leaves a started server running and says so; with Option it is Quit & Stop" {
    var b: [160]u8 = undefined;
    const plain = quitButton(&b, false, 0);
    try std.testing.expectEqualStrings("Quit", plain.label);
    try std.testing.expect(!plain.stops);
    try std.testing.expect(!quitButton(&b, true, 0).stops);
    const kept = quitButton(&b, false, 1);
    try std.testing.expectEqualStrings("Quit", kept.label);
    try std.testing.expect(std.mem.indexOf(u8, kept.tip, "keeps running") != null and std.mem.indexOf(u8, kept.tip, "⌥") != null);
    const both = quitButton(&b, true, 2);
    try std.testing.expectEqualStrings("Quit & Stop", both.label);
    try std.testing.expect(both.stops and std.mem.indexOf(u8, both.tip, "2 servers") != null);
    try std.testing.expect(std.mem.indexOf(u8, quitButton(&b, true, 1).tip, "the server it started") != null);
}

test "speeds: one rounding rule in the title and the card, whole numbers from 100 up" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var b: [64]u8 = undefined;
    for ([_]struct { tps: f64, text: []const u8 }{ .{ .tps = 105.27, .text = "105" }, .{ .tps = 99.94, .text = "99.9" }, .{ .tps = 7.25, .text = "7.3" } }) |c| {
        const h = try buildHeader(arena.allocator(), .{ .link = .up, .port = 1, .model = "m", .decoding = true, .live = c.tps });
        try std.testing.expectEqualStrings(c.text, h.stats[0].value);
        const t = title(&b, .{ .link = .up, .decoding = true, .live = c.tps });
        try std.testing.expectEqualStrings(c.text, t[0..std.mem.indexOfScalar(u8, t, ' ').?]);
    }
}
