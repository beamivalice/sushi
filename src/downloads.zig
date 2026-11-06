//! The web page's Available list. `GET /v1/downloads` lists the published Sushi packs (the `cli.aliases` table plus
//! the org's public repos Hugging Face lists) with size, fit for this Mac's RAM and whether each is on disk, beside
//! the download queue; `POST /v1/downloads` queues a pack, `POST /v1/downloads/cancel` stops or unqueues one.
//! Downloads run one at a time through `cli.pullRepo` on a thread this module owns, never the inference thread and
//! never a connection thread. Partials and the pull manifest stay on disk, so a stopped download resumes when it is
//! started again, after a restart too. The byte accounting, speed and space rules are in `download_progress.zig`.
const std = @import("std");
const cli = @import("cli.zig");
const dp = @import("download_progress.zig");
const pack_update = @import("pack_update.zig");
const disk_placement = @import("disk_placement.zig");
const model_settings = @import("model_settings.zig");
const status = @import("status.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const http_conn = @import("http_conn.zig");
const server = @import("server.zig");

const Allocator = std.mem.Allocator;
const Phase = dp.Phase;

const Job = struct {
    allocator: Allocator,
    repo: []u8,
    /// The pack folder it fills: an installed pack's own folder for an update.
    dest: []u8,
    update: bool = false,
    phase: Phase = .queued,
    progress: dp.Progress = .{},
    message: []u8 = &.{},
    cancel: std.atomic.Value(bool) = .init(false),
    /// The curl of the file in flight, unreaped while set (`cli.waitExitedUnreaped`), so a SIGTERM never hits another
    /// process.
    pid: ?std.posix.pid_t = null,
    meter: dp.Meter = .{},

    fn create(allocator: Allocator, repo: []const u8, dest: []const u8, update: bool) !*Job {
        const job = try allocator.create(Job);
        job.* = .{ .allocator = allocator, .repo = &.{}, .dest = &.{}, .update = update };
        errdefer job.destroy();
        job.repo = try allocator.dupe(u8, repo);
        job.dest = try allocator.dupe(u8, dest);
        return job;
    }

    fn destroy(self: *Job) void {
        const a = self.allocator;
        a.free(self.repo);
        a.free(self.dest);
        a.free(self.message);
        self.progress.deinit(a);
        a.destroy(self);
    }

    fn setMessage(self: *Job, text: []const u8) void {
        const copy = self.allocator.dupe(u8, text) catch return;
        self.allocator.free(self.message);
        self.message = copy;
    }

    /// Bytes of the file in flight on disk now, from its partial.
    fn partialNow(self: *const Job, io: std.Io) u64 {
        if (self.progress.file.len == 0) return 0;
        if (self.phase == .verifying) return self.progress.file_size;
        return partialSize(io, self.dest, self.progress.file);
    }
};

fn partialSize(io: std.Io, dir: []const u8, file: []const u8) u64 {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/{s}.partial", .{ dir, file }) catch return 0;
    return (std.Io.Dir.cwd().statFile(io, path, .{}) catch return 0).size;
}

const gpa = std.heap.smp_allocator;

var mu: std.c.pthread_mutex_t = .{};
var jobs: std.ArrayList(*Job) = .empty;
var worker_running = false;
var stopping = false;

fn lock() void {
    _ = std.c.pthread_mutex_lock(&mu);
}

fn unlock() void {
    _ = std.c.pthread_mutex_unlock(&mu);
}

// ── The catalog ─────────────────────────────────────────────────────────

const Entry = struct { repo: []const u8, size: ?u64, update: ?pack_update.Available = null };

const CatalogState = enum { loading, ready, offline };

var cat_arena: ?std.heap.ArenaAllocator = null;
var cat_entries: []Entry = &.{};
var cat_state: CatalogState = .loading;
var cat_at_ms: i64 = 0;
var cat_busy = false;

const fresh_ms: i64 = 10 * 60 * 1000;
const offline_retry_ms: i64 = 60 * 1000;

/// Known packs first, in table order, then the org's other public Sushi repos; a pack Hugging Face says is not
/// there (unpublished, private) is left out, and with no answer at all the known packs stand without sizes. With
/// `updates`, each installed pack's update is looked up too.
fn buildCatalog(arena: Allocator, io: std.Io, rs: []const []const u8, updates: bool) struct { entries: []Entry, state: CatalogState } {
    var repos = std.ArrayList([]const u8).empty;
    for (cli.aliases) |a| repos.append(arena, a.repo) catch {};
    const listed = cli.fetchOrgPacks(arena, io);
    if (listed) |ids| for (ids) |id| {
        if (cli.aliasForRepo(id) == null) repos.append(arena, id) catch {};
    };
    var entries = std.ArrayList(Entry).empty;
    var offline = false;
    for (repos.items, 0..) |repo, i| {
        const outcome: cli.FetchOutcome = if (offline) .offline else cli.fetchRepoSize(arena, io, repo);
        if (i == 0 and outcome == .offline and listed == null) offline = true;
        var e: Entry = .{ .repo = repo, .size = null };
        switch (outcome) {
            .size => |s| e.size = s,
            .not_found => continue,
            .offline => {},
        }
        if (!offline and updates) if (installedPath(arena, io, rs, repo)) |path| {
            switch (pack_update.checkPack(arena, io, &cli.known_packs, path)) {
                .available => |u| e.update = u,
                else => {},
            }
        };
        entries.append(arena, e) catch {};
    }
    return .{ .entries = entries.items, .state = if (offline) .offline else .ready };
}

fn refreshThread(io: std.Io, updates: bool) void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    var roots_buf: [9][]const u8 = undefined;
    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const built = buildCatalog(arena.allocator(), io, roots(&roots_buf, destRoot(&dest_buf, io).path), updates);
    lock();
    defer unlock();
    if (cat_arena) |*old| old.deinit();
    cat_arena = arena;
    cat_entries = built.entries;
    cat_state = built.state;
    cat_at_ms = io_util.nowMsMonotonic(io);
    cat_busy = false;
}

/// Starts a catalog refresh, at most one at a time: on Check (`force`), on the first look, and while automatic checks
/// are on when the last one is stale. Installed packs' updates are looked up only on Check or with the checks on.
/// Holds `mu`.
fn refreshIfStaleLocked(io: std.Io, force: bool) void {
    if (cat_busy) return;
    const auto = pack_update.autoChecksOn();
    const age = io_util.nowMsMonotonic(io) - cat_at_ms;
    const stale = cat_at_ms == 0 or force or (auto and age > (if (cat_state == .offline) offline_retry_ms else fresh_ms));
    if (!stale) return;
    const t = std.Thread.spawn(.{}, refreshThread, .{ io, auto or force }) catch return;
    t.detach();
    cat_busy = true;
    if (cat_entries.len == 0) cat_state = .loading;
}

/// The catalog's own spelling of `repo` (an alias or a fetched repo), or null for a repo it does not list. Holds `mu`.
fn catalogRepoLocked(repo: []const u8) ?[]const u8 {
    if (cli.aliasForRepo(repo)) |a| return a.repo;
    for (cat_entries) |e| if (std.ascii.eqlIgnoreCase(e.repo, repo)) return e.repo;
    return null;
}

// ── Where packs go ──────────────────────────────────────────────────────

const Dest = struct { path: []const u8, source: []const u8, locked: bool };

/// The models folder the Settings tab edits: the `--model-dir` flag, else `server.models_dir`, else the default.
fn destRoot(buf: []u8, io: std.Io) Dest {
    const locks = &model_settings.launch_locks;
    if (locks.models_dir.set) return .{ .path = locks.models_dir.text(), .source = "flag", .locked = true };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const reg = server.global_registry;
    const settings_path = if (reg) |r| r.settings_path orelse model_settings.defaultPath(&path_buf) else model_settings.defaultPath(&path_buf);
    var file = model_settings.load(gpa, io, settings_path);
    defer file.deinit();
    if (model_settings.server(&file).models_dir) |d| if (d.len <= buf.len) {
        @memcpy(buf[0..d.len], d);
        return .{ .path = buf[0..d.len], .source = "file", .locked = false };
    };
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    return .{ .path = std.fmt.bufPrint(buf, "{s}/.sushi/models", .{home}) catch "/tmp", .source = "default", .locked = false };
}

/// The destination first, then the folders the server discovered from.
fn roots(buf: *[9][]const u8, dest: []const u8) []const []const u8 {
    buf[0] = dest;
    var n: usize = 1;
    const boot = if (server.global_registry) |r| if (r.discovery) |d| d.roots else &.{} else &.{};
    for (boot) |r| {
        if (n == buf.len) break;
        if (std.mem.eql(u8, r, buf[0])) continue;
        buf[n] = r;
        n += 1;
    }
    return buf[0..n];
}

fn installedPath(allocator: Allocator, io: std.Io, rs: []const []const u8, repo: []const u8) ?[]u8 {
    for (rs) |r| if (cli.catalogPackPath(io, allocator, r, repo)) |p| return p;
    return null;
}

/// `<root>/<org>/<repo>`, where a download of a pack not yet installed goes.
fn ownFolder(allocator: Allocator, root: []const u8, repo: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ std.mem.trimEnd(u8, root, "/"), repo });
}

/// Bytes on disk under `path`, two levels deep (a pack's layout), partials included; null when it is not there.
fn dirBytes(io: std.Io, path: []const u8) ?u64 {
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    return dirBytesIn(io, dir, 2);
}

fn dirBytesIn(io: std.Io, dir: std.Io.Dir, depth: u8) u64 {
    var total: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |e| switch (e.kind) {
        .file => total += (dir.statFile(io, e.name, .{}) catch continue).size,
        .directory => if (depth > 1) {
            var sub = dir.openDir(io, e.name, .{ .iterate = true }) catch continue;
            defer sub.close(io);
            total += dirBytesIn(io, sub, depth - 1);
        },
        else => {},
    };
    return total;
}

/// The id of the loaded (or loading) model served from `path`, or null. Caller holds the registry's mutex.
fn loadedModelAtLocked(allocator: Allocator, path: []const u8) ?[]u8 {
    const reg = server.global_registry orelse return null;
    const e = reg.peekByPathLocked(path) orelse return null;
    if (e.state != .ready and e.state != .loading) return null;
    return allocator.dupe(u8, e.id) catch null;
}

/// The registry's load gate: a pack folder a queued or running download or update fills cannot load, since its
/// files are replaced one by one and a load would read some of each revision; nor can one a stopped or failed pull
/// (this server's or `sushi pull`'s) left behind with its `.pull-incomplete` marker.
pub fn loadGate(buf: []u8, path: []const u8) ?[]const u8 {
    const want = std.mem.trimEnd(u8, path, "/");
    lock();
    defer unlock();
    for (jobs.items) |j| {
        if (!j.phase.active() or !std.mem.eql(u8, std.mem.trimEnd(u8, j.dest, "/"), want)) continue;
        return std.fmt.bufPrint(buf, "{s} is being {s} on Downloads: wait for it to finish, or stop it there, then load it", .{ j.repo, if (j.update) "updated" else "downloaded" }) catch
            "This model is being downloaded on Downloads: wait for it to finish, or stop it there, then load it";
    }
    // A stopped or failed pull leaves its marker and some files of each revision.
    var marker: [std.fs.max_path_bytes]u8 = undefined;
    const at = std.fmt.bufPrintSentinel(&marker, "{s}/{s}", .{ want, cli.pull_marker }, 0) catch return null;
    if (std.c.access(at, 0) != 0) return null;
    return std.fmt.bufPrint(buf, "The last download or update of {s} did not finish, so its files are mixed: resume it on Downloads (or `sushi pull` it again), then load it", .{std.fs.path.basename(want)}) catch
        "The last download or update of this model did not finish: resume it on Downloads, then load it";
}

// ── The download thread ─────────────────────────────────────────────────

fn onLine(impl: *anyopaque, line: []const u8) void {
    const job: *Job = @ptrCast(@alignCast(impl));
    log.info("[download] {s}: {s}\n", .{ job.repo, line });
    if (!std.mem.startsWith(u8, line, "error: ")) return;
    lock();
    defer unlock();
    job.setMessage(line["error: ".len..]);
}

var worker_io: std.Io = undefined;

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

fn onEvent(impl: *anyopaque, ev: cli.PullEvent) bool {
    const job: *Job = @ptrCast(@alignCast(impl));
    const space = if (ev == .listed) disk_placement.volumeSpaceNear(job.dest) else null;
    const partial = if (ev == .file and !ev.file.skip) partialSize(worker_io, job.dest, ev.file.path) else 0;
    lock();
    defer unlock();
    switch (ev) {
        .listed => |l| {
            job.progress.files = l.files;
            job.progress.total = l.bytes;
            const need = l.bytes -| l.present;
            if (space) |v| if (dp.spaceVerdict(need, v.free, v.total) == .short) {
                var buf: [256]u8 = undefined;
                job.setMessage(std.fmt.bufPrint(&buf, "needs {d:.1} GiB more; the destination has {d:.1} GiB free and keeps {d:.1} GiB in reserve", .{ gib(need), gib(v.free), gib(dp.reserve(v.total)) }) catch "not enough free space");
                return false;
            };
        },
        .checking => job.phase = .checking,
        .file => |f| {
            job.progress.startFile(job.allocator, f, partial) catch {};
            if (!f.skip) job.phase = .downloading;
        },
        .child => |pid| job.pid = pid,
        .verifying => job.phase = .verifying,
    }
    return !job.cancel.load(.monotonic);
}

fn nextQueuedLocked() ?*Job {
    for (jobs.items) |j| if (j.phase == .queued) return j;
    return null;
}

fn workerLoop(io: std.Io) void {
    while (true) {
        lock();
        const job = (if (stopping) null else nextQueuedLocked()) orelse {
            worker_running = false;
            unlock();
            return;
        };
        job.phase = .listing;
        job.setMessage("");
        job.meter = .{};
        unlock();

        const reporter: cli.Reporter = .{ .impl = job, .reportFn = &onLine, .eventFn = &onEvent, .stop = &job.cancel };
        const ok = if (cli.pullRepo(gpa, io, .{ .repo = job.repo }, job.dest, reporter, false)) |_| true else |_| false;

        var repo_buf: [256]u8 = undefined;
        var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
        lock();
        const repo = repo_buf[0..@min(job.repo.len, repo_buf.len)];
        @memcpy(repo, job.repo[0..repo.len]);
        const dest = dest_buf[0..@min(job.dest.len, dest_buf.len)];
        @memcpy(dest, job.dest[0..dest.len]);
        job.progress.endFile(job.allocator, ok);
        job.pid = null;
        job.phase = if (ok) .done else if (job.cancel.load(.monotonic)) .cancelled else .failed;
        if (job.phase == .failed and job.message.len == 0) job.setMessage("the download failed");
        if (job.phase != .failed) job.setMessage("");
        // The pack is now the published revision: its row drops the update without asking Hugging Face again.
        if (ok) for (cat_entries) |*e| if (std.ascii.eqlIgnoreCase(e.repo, repo)) {
            e.update = null;
        };
        const rescan = ok and !stopping;
        unlock();
        // `job` may be dropped from here on: a new start of the same pack replaces a finished one.
        if (rescan) if (server.global_registry) |r| {
            const added = r.absorbDownload(dest) catch 0;
            log.info("[download] {s} is ready; models rescanned, {d} added\n", .{ repo, added });
        };
    }
}

/// Queues `repo` into the pack folder `dest` and starts the download thread when it is idle. False when it is
/// already queued or running.
fn enqueue(io: std.Io, repo: []const u8, dest: []const u8, update: bool) !bool {
    lock();
    defer unlock();
    if (stopping) return error.ShuttingDown;
    var i: usize = 0;
    while (i < jobs.items.len) {
        const j = jobs.items[i];
        if (std.ascii.eqlIgnoreCase(j.repo, repo)) {
            if (j.phase.active()) return false;
            _ = jobs.orderedRemove(i);
            j.destroy();
        } else i += 1;
    }
    const job = try Job.create(gpa, repo, dest, update);
    errdefer job.destroy();
    try jobs.ensureUnusedCapacity(gpa, 1);
    // The thread waits on `mu` until this returns, so it finds the job; it is spawned before the append, so a failed
    // spawn leaves no pointer to the freed job.
    if (!worker_running) {
        worker_io = io;
        const t = try std.Thread.spawn(.{}, workerLoop, .{io});
        t.detach();
        worker_running = true;
    }
    jobs.appendAssumeCapacity(job);
    return true;
}

/// Stops a running download (its partials stay) or drops a queued one. False when `repo` has neither.
fn cancel(repo: []const u8) bool {
    lock();
    defer unlock();
    for (jobs.items) |j| {
        if (!std.ascii.eqlIgnoreCase(j.repo, repo) or !j.phase.active()) continue;
        if (j.phase == .queued) {
            j.phase = .cancelled;
        } else {
            j.cancel.store(true, .monotonic);
            if (j.pid) |pid| _ = std.c.kill(pid, .TERM);
        }
        return true;
    }
    return false;
}

/// Server shutdown: no curl outlives the server, nothing queued starts and nothing new is queued.
pub fn stop() void {
    lock();
    defer unlock();
    stopping = true;
    for (jobs.items) |j| {
        if (j.phase == .queued) j.phase = .cancelled;
        j.cancel.store(true, .monotonic);
        if (j.pid) |pid| _ = std.c.kill(pid, .TERM);
    }
}

// ── The answer ──────────────────────────────────────────────────────────

pub const PackView = struct {
    repo: []const u8,
    size: ?u64,
    installed: ?[]const u8 = null,
    /// Bytes on disk of a pack folder that is there but not complete; null while that pack downloads.
    partial_bytes: ?u64 = null,
    update: ?pack_update.Available = null,
};

/// `'path'` for a shell, a `'` written as `'\''`.
fn shellQuote(w: *std.Io.Writer, path: []const u8) !void {
    try w.writeByte('\'');
    for (path) |c| if (c == '\'') try w.writeAll("'\\''") else try w.writeByte(c);
    try w.writeByte('\'');
}

/// One pack row: what it is, whether this Mac's RAM meets its tier (null for a pack not in the table), the pack
/// `sushi list` recommends, what is on disk, and for a downloaded known pack its recommended launch.
pub fn writePack(w: *std.Io.Writer, p: PackView, ram_gb: u64) !void {
    const a = cli.aliasForRepo(p.repo);
    const base = p.repo[(std.mem.indexOfScalar(u8, p.repo, '/') orelse 0) + 1 ..];
    const recommended = if (a) |al| if (cli.recommendedAlias(al.name, ram_gb)) |r| std.mem.eql(u8, r.repo, al.repo) else false else false;
    var launch_buf: [2 * std.fs.max_path_bytes]u8 = undefined;
    var lw = std.Io.Writer.fixed(&launch_buf);
    const launch: ?[]const u8 = if (p.installed != null and a != null) blk: {
        lw.writeAll("sushi serve --model ") catch break :blk null;
        shellQuote(&lw, p.installed.?) catch break :blk null;
        if (a.?.serve_flags.len > 0) lw.print(" {s}", .{a.?.serve_flags}) catch break :blk null;
        break :blk lw.buffered();
    } else null;
    try std.json.Stringify.value(.{
        .repo = p.repo,
        .name = if (a) |al| al.name else base,
        .tag = if (a) |al| al.tag else "",
        .size_bytes = p.size,
        .tier_gb = if (a) |al| @as(?u32, al.tier_gb) else null,
        .fits = if (a) |al| @as(?bool, cli.verdictFor(al, ram_gb) == .fits) else null,
        .streams_experts = if (a) |al| al.streams_experts else false,
        .recommended = recommended,
        .installed = p.installed != null,
        .path = p.installed,
        .partial_bytes = p.partial_bytes,
        .update = if (p.update) |u| @as(?struct { current: []const u8, latest: []const u8 }, .{ .current = u.local, .latest = u.latest }) else null,
        .launch = launch,
    }, .{}, w);
}

fn writeJob(w: *std.Io.Writer, j: *Job, io: std.Io, now_ms: i64) !void {
    const live = j.phase.active();
    const at = j.progress.now(if (live) j.partialNow(io) else 0);
    if (j.phase == .downloading) j.meter.add(now_ms, at.fetched);
    const speed = if (j.phase == .downloading or j.phase == .verifying) j.meter.rate() else null;
    const eta: ?u64 = if (speed) |s| if (s >= 1) @intFromFloat(@as(f64, @floatFromInt(j.progress.total -| at.done)) / s) else null else null;
    const file = j.progress.file;
    try std.json.Stringify.value(.{
        .repo = j.repo,
        .dest = j.dest,
        .phase = @tagName(j.phase),
        .stopping = live and j.cancel.load(.monotonic),
        .files = j.progress.files,
        .file_index = j.progress.file_index,
        .total_bytes = j.progress.total,
        .done_bytes = at.done,
        .file = if (file.len > 0 and live) @as(?struct { path: []const u8, size: u64, done: u64 }, .{ .path = file, .size = j.progress.file_size, .done = at.done - j.progress.done }) else null,
        .speed_bps = if (speed) |s| @as(?u64, @intFromFloat(s)) else null,
        .eta_s = eta,
        .message = if (j.message.len > 0) @as(?[]const u8, j.message) else null,
    }, .{}, w);
}

fn render(allocator: Allocator, io: std.Io) ![]u8 {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The catalog and the downloading packs, copied under the lock; the disk is read without it.
    var entries: []Entry = &.{};
    var state: CatalogState = undefined;
    var refreshing = false;
    var active = std.ArrayList([]const u8).empty;
    {
        lock();
        defer unlock();
        state = if (cat_busy and cat_entries.len == 0) .loading else cat_state;
        refreshing = cat_busy;
        if (cat_entries.len > 0 or cat_state == .ready) {
            entries = try arena.alloc(Entry, cat_entries.len);
            for (cat_entries, entries) |e, *c| c.* = .{
                .repo = try arena.dupe(u8, e.repo),
                .size = e.size,
                .update = if (e.update) |u| @as(?pack_update.Available, .{ .local = try arena.dupe(u8, u.local), .latest = try arena.dupe(u8, u.latest), .command = "" }) else null,
            };
        } else {
            entries = try arena.alloc(Entry, cli.aliases.len);
            for (cli.aliases, entries) |a, *c| c.* = .{ .repo = a.repo, .size = null };
        }
        for (jobs.items) |j| if (j.phase.active()) try active.append(arena, try arena.dupe(u8, j.repo));
    }

    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dest = destRoot(&dest_buf, io);
    var roots_buf: [9][]const u8 = undefined;
    const rs = roots(&roots_buf, dest.path);
    const volume = try disk_placement.cacheVolumeAt(allocator, dest.path, false);
    defer volume.deinit(allocator);
    const ram = status.getTotalMemBytes();
    const ram_gb = cli.installedGb(ram);

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.print("{{\"ram_bytes\":{d},\"ram_gb\":{d},\"update_check\":{},\"dest\":", .{ ram, ram_gb, pack_update.autoChecksOn() });
    try std.json.Stringify.value(.{
        .path = dest.path,
        .source = dest.source,
        .locked = dest.locked,
        .volume = volume.name,
        .free_bytes = if (volume.space) |s| @as(?u64, s.free) else null,
        .total_bytes = if (volume.space) |s| @as(?u64, s.total) else null,
    }, .{}, w);
    try w.print(",\"catalog\":\"{s}\",\"refreshing\":{},\"packs\":[", .{ @tagName(state), refreshing });
    for (entries, 0..) |e, i| {
        if (i > 0) try w.writeAll(",");
        const installed = installedPath(arena, io, rs, e.repo);
        const downloading = for (active.items) |r| {
            if (std.ascii.eqlIgnoreCase(r, e.repo)) break true;
        } else false;
        const partial = if (installed == null and !downloading) dirBytes(io, try ownFolder(arena, dest.path, e.repo)) else null;
        try writePack(w, .{ .repo = e.repo, .size = e.size, .installed = installed, .partial_bytes = partial, .update = if (installed != null) e.update else null }, ram_gb);
    }
    try w.writeAll("],\"jobs\":[");
    {
        lock();
        defer unlock();
        const now = io_util.nowMsMonotonic(io);
        for (jobs.items, 0..) |j, i| {
            if (i > 0) try w.writeAll(",");
            try writeJob(w, j, io, now);
        }
    }
    try w.writeAll("]}");
    return out.toOwnedSlice();
}

// ── Routes ──────────────────────────────────────────────────────────────

/// `GET /v1/downloads[?refresh=1|?background=1]`: the catalog, this Mac and the queue. A refresh runs in the
/// background; the answer never waits on Hugging Face. The page's look from another tab (`background=1`) asks
/// Hugging Face nothing while automatic checks are off (`--no-update-check`, `pack_update_check: false`).
pub fn handleGet(allocator: Allocator, stream: *http_conn.Conn, raw_path: []const u8) !void {
    const background = std.mem.indexOf(u8, raw_path, "background=1") != null;
    if (!background or pack_update.autoChecksOn()) {
        lock();
        defer unlock();
        refreshIfStaleLocked(stream.io, std.mem.indexOf(u8, raw_path, "refresh=1") != null);
    }
    const body = try render(allocator, stream.io);
    defer allocator.free(body);
    try server.sendResponseQuiet(stream, "200 OK", "application/json", body);
}

fn bodyRepo(arena: Allocator, body: []const u8) ?[]const u8 {
    const Req = struct { repo: []const u8 = "" };
    const req = std.json.parseFromSliceLeaky(Req, arena, body, .{ .ignore_unknown_fields = true }) catch return null;
    return if (req.repo.len == 0) null else req.repo;
}

/// `POST /v1/downloads {"repo":"beamster/<repo>"}`: only a pack the catalog lists, under its own spelling. A pack
/// already downloaded is updated in its own folder, and refused while it is loaded. A new pack that would eat into the
/// destination's reserve is refused with 507 and one that leaves under a tenth of the volume free carries a `warning`;
/// the download checks again against the exact listing before it writes.
pub fn handleStart(allocator: Allocator, stream: *http_conn.Conn, body: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const asked = bodyRepo(arena, body) orelse
        return server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Send {\"repo\":\"beamster/<repo>\"}", 400);
    if (!dp.validRepo(cli.HF_ORG, asked))
        return server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Not a Sushi pack: downloads come from the Hugging Face org beamster only", 400);
    var size: ?u64 = null;
    const repo = blk: {
        lock();
        defer unlock();
        const r = catalogRepoLocked(asked) orelse break :blk null;
        for (cat_entries) |e| if (std.mem.eql(u8, e.repo, r)) {
            size = e.size;
        };
        break :blk try arena.dupe(u8, r);
    } orelse return server.sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Not a published Sushi pack; Check again refreshes the list", 404);

    var dest_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = destRoot(&dest_buf, stream.io).path;
    var roots_buf: [9][]const u8 = undefined;
    const installed = installedPath(arena, stream.io, roots(&roots_buf, root), repo);
    const dest = installed orelse try ownFolder(arena, root, repo);
    var warning: ?[]const u8 = null;
    if (installed == null) if (size) |s| if (disk_placement.volumeSpaceNear(dest)) |v| {
        const need = s -| (dirBytes(stream.io, dest) orelse 0);
        switch (dp.spaceVerdict(need, v.free, v.total)) {
            .short => {
                const msg = try std.fmt.allocPrint(arena, "{s} needs {d:.1} GiB; the destination has {d:.1} GiB free and keeps {d:.1} GiB in reserve", .{ repo, gib(need), gib(v.free), gib(dp.reserve(v.total)) });
                return server.sendErrorResponse(allocator, stream, "507 Insufficient Storage", "insufficient_storage", msg, 507);
            },
            .low => warning = try std.fmt.allocPrint(arena, "leaves {d:.1} GiB free on the destination", .{gib(v.free - need)}),
            .fits => {},
        }
    };
    // The registry's mutex spans the loaded check and the enqueue: a load checks the queue under it (`load_gate`).
    const reg = server.global_registry;
    if (reg) |r| r.mutex.lockUncancelable(r.io);
    const loaded_as = if (installed) |path| loadedModelAtLocked(arena, path) else null;
    const queued = if (loaded_as != null) false else enqueue(stream.io, repo, dest, installed != null);
    if (reg) |r| r.mutex.unlock(r.io);
    if (loaded_as) |id| {
        const msg = try std.fmt.allocPrint(arena, "{s} is loaded as {s}: unload {s} first, then update it", .{ repo, id, id });
        return server.sendErrorResponse(allocator, stream, "409 Conflict", "model_loaded", msg, 409);
    }
    const queued_now = queued catch |err| switch (err) {
        error.ShuttingDown => return server.sendErrorResponse(allocator, stream, "503 Service Unavailable", "shutting_down", "The server is shutting down", 503),
        else => return err,
    };
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try std.json.Stringify.value(.{ .ok = true, .queued = queued_now, .repo = repo, .dest = dest, .warning = warning }, .{}, &out.writer);
    try server.sendResponse(stream, "202 Accepted", "application/json", out.written());
}

/// `POST /v1/downloads/cancel {"repo":...}`: the partials stay, so starting it again resumes.
pub fn handleCancel(allocator: Allocator, stream: *http_conn.Conn, body: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const repo = bodyRepo(arena_state.allocator(), body) orelse
        return server.sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Send {\"repo\":\"beamster/<repo>\"}", 400);
    if (!cancel(repo)) return server.sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "No download of that pack is queued or running", 404);
    try server.sendResponse(stream, "200 OK", "application/json", "{\"ok\":true}");
}

const testing = std.testing;

test "downloads: a pack row carries the tier verdict, the recommended pack and its recommended launch" {
    const Row = struct { repo: []const u8, tier_gb: ?u32 = null, fits: ?bool = null, recommended: bool, installed: bool, streams_experts: bool, name: []const u8, launch: ?[]const u8 = null };
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const row = struct {
        fn of(a: Allocator, p: PackView, ram_gb: u64) !Row {
            var buf: [4096]u8 = undefined;
            var w = std.Io.Writer.fixed(&buf);
            try writePack(&w, p, ram_gb);
            return std.json.parseFromSliceLeaky(Row, a, w.buffered(), .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        }
    }.of;
    const q275 = try row(arena, .{ .repo = "beamster/Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .size = 40 << 30 }, 64);
    try testing.expect(q275.recommended and q275.fits.? and q275.tier_gb.? == 64 and q275.launch == null);
    try testing.expectEqualStrings("qwen3.8-flash-next", q275.name);
    const q4 = try row(arena, .{ .repo = "beamster/Qwen3.8-Flash-Next-Sushi-4bpw", .size = null, .installed = "/m/it's/Qwen3.8-Flash-Next-Sushi-4bpw" }, 64);
    try testing.expect(!q4.recommended and !q4.fits.? and q4.installed);
    try testing.expect(std.mem.startsWith(u8, q4.launch.?, "sushi serve --model '/m/it'\\''s/Qwen3.8-Flash-Next-Sushi-4bpw' --mtp --kv-quant 8"));
    const ds = try row(arena, .{ .repo = "beamster/DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw", .size = 1, .installed = "/m/d" }, 128);
    try testing.expect(ds.streams_experts and ds.fits.?);
    try testing.expectEqualStrings("sushi serve --model '/m/d'", ds.launch.?);
    // A pack published after this build has no tier and no recipe: no verdict, never recommended, no launch line.
    const new = try row(arena, .{ .repo = "beamster/New-Sushi-3bpw", .size = 1, .installed = "/m/n" }, 128);
    try testing.expect(new.fits == null and new.tier_gb == null and !new.recommended and new.launch == null);
    try testing.expectEqualStrings("New-Sushi-3bpw", new.name);
}

test "downloads: progress follows the pull's events, and a stop is answered at the next one" {
    worker_io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var p: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &p);
    const job = try Job.create(testing.allocator, "beamster/X", p[0..n], false);
    defer job.destroy();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.safetensors.partial", .data = "1234" });
    try testing.expect(onEvent(job, .{ .listed = .{ .files = 3, .bytes = 30, .present = 0 } }));
    try testing.expect(onEvent(job, .{ .file = .{ .index = 1, .path = "a.json", .size = 10, .skip = true } }));
    try testing.expect(onEvent(job, .{ .file = .{ .index = 2, .path = "b.safetensors", .size = 10, .skip = false } }));
    try testing.expectEqual(Phase.downloading, job.phase);
    try testing.expectEqual(@as(u64, 4), job.progress.resume_from);
    try testing.expectEqual(@as(u64, 4), job.partialNow(testing.io));
    try testing.expect(onEvent(job, .verifying));
    try testing.expectEqual(@as(u64, 10), job.partialNow(testing.io));
    try testing.expect(onEvent(job, .{ .file = .{ .index = 3, .path = "c.safetensors", .size = 10, .skip = false } }));
    try testing.expectEqual(@as(u64, 20), job.progress.done);
    try testing.expectEqual(@as(u64, 6), job.progress.fetched);
    job.cancel.store(true, .monotonic);
    try testing.expect(!onEvent(job, .{ .child = 42 }));
}

test "downloads: a pack that cannot fit is stopped once its listing is known, naming the space" {
    worker_io = testing.io;
    const job = try Job.create(testing.allocator, "beamster/X", "/", false);
    defer job.destroy();
    const v = disk_placement.volumeSpaceNear("/").?;
    try testing.expect(!onEvent(job, .{ .listed = .{ .files = 1, .bytes = v.free + (1 << 40), .present = 0 } }));
    try testing.expect(std.mem.indexOf(u8, job.message, "GiB free") != null);
    // What is on disk already is not needed again.
    try testing.expect(onEvent(job, .{ .listed = .{ .files = 1, .bytes = v.free + (1 << 40), .present = v.free + (1 << 40) } }));
}
