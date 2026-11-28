//! The engine's pipeline manifest for a Cooler load (`src/pipeline_manifest.zig`), for every arch: which manifest a load
//! starts from, what it makes ahead of the warm, and the manifest it leaves behind, rewritten after a request that made a
//! pipeline the warm did not. An arch names its scope (`coolerPipelineScope`: the pack and what selects its kernels).
//! Every case is one `[pipelines]` line.
const std = @import("std");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const device = @import("sushi_exl3").device;
const load_clock = @import("../load_clock.zig");
const manifest = @import("../pipeline_manifest.zig");

const cache = mlx.engine.pipeline_cache;
const registry = mlx.engine.pipeline_registry;

/// Variables of the shared kernels that select which ones the engine makes or what their source says.
const recipe_env = [_][*:0]const u8{
    "SUSHI_EXL3_GEMM_WIN",
    "SUSHI_EXL3_WIN_ALIGN",
    "SUSHI_EXL3_SWIGLU_MAXABS",
    "SUSHI_FORCE_GPU_FAMILY_FALLBACK",
    "MLX_METAL_GPU_ARCH",
    "MLX_ENABLE_TF32",
};

pub fn addRecipeEnv(d: *manifest.Digest) void {
    for (recipe_env) |name| if (std.c.getenv(name)) |v| d.add(std.mem.span(name), std.mem.span(v));
}

var cached_build_key: ?[16]u8 = null;

pub fn buildKey(a: std.mem.Allocator, io: std.Io) ![16]u8 {
    if (!@import("builtin").is_test) {
        if (cached_build_key) |k| return k;
    }
    var d: manifest.Digest = .{};
    var arch_buf: [64]u8 = undefined;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    try manifest.addBuild(&d, a, io, device.archName(&arch_buf), try mlx.engine.metallibPath(&path_buf));
    const key = d.hex();
    if (!@import("builtin").is_test) cached_build_key = key;
    return key;
}

pub const Case = enum { off, none, unreadable, other_build, same };

/// A test's directory, in place of the process's root.
pub var root_hook: ?std.Io.Dir = null;

/// The root directory the process was given (`manifest.setRootFromHome`), made if it is missing. `reason` names why
/// there is none: nothing derives a default, so a process that was not given one keeps no manifest.
pub fn openRoot(io: std.Io, reason: *[]const u8) ?std.Io.Dir {
    if (root_hook) |dir| return dir;
    const path = manifest.configuredRoot() orelse {
        reason.* = "no manifest root was given";
        return null;
    };
    return std.Io.Dir.cwd().createDirPathOpen(io, path, .{}) catch |e| {
        reason.* = @errorName(e);
        return null;
    };
}

pub fn threadCount() usize {
    return @min(std.Thread.getCpuCount() catch 4, 8);
}

pub const Plan = struct {
    a: std.mem.Allocator,
    io: std.Io,
    root: ?std.Io.Dir,
    key: manifest.Key,
    key_ms: u64,
    found: manifest.Lookup,
    case: Case,
    why: []const u8 = "",
    threads: usize,
    /// Whether `root` came from the process's root path (`openRoot`), which a later write opens again.
    from_path: bool = false,
    /// Whether `root` is a handle this plan opened, which it closes.
    owns_root: bool = false,
    /// What the last `settle` spent making pipelines ahead, and how many it made.
    made_ns: u64 = 0,
    made_count: u64 = 0,
    /// Whether `archive.bin` beside this key's manifest loaded cleanly and how many recipes it covered.
    archive_loaded: bool = false,
    archive_recipes: usize = 0,
    /// `pipeline_registry.generation` captured right after `await` settled, so warm loads with 0 new recipes skip disk rewrite.
    settled_gen: u64 = 0,
    /// In-flight async preload state between `startAsync` and `await`.
    async_job: ?*cache.Job = null,
    async_archive: ?cache.Archive = null,
    async_t0: u64 = 0,
    async_started: bool = false,

    pub fn deinit(p: *Plan) void {
        if (p.async_job) |job| {
            cache.abort(job);
            p.async_job = null;
        }
        if (p.async_archive) |arch| {
            arch.release();
            p.async_archive = null;
        }
        p.found.deinit();
        // A path-made root is opened again by every later write and shape: this handle is the load's own.
        if (p.owns_root) p.root.?.close(p.io);
    }

    /// Launches pipeline creation on worker threads without blocking, so weight I/O can overlap with Metal pipeline loading.
    /// Must be paired with `await()` (or `deinit()` on error).
    pub fn startAsync(p: *Plan) void {
        if (p.async_started) return;
        p.async_started = true;
        var dir_buf: [manifest.Key.dir_len]u8 = undefined;
        const dir = p.key.dirName(&dir_buf);
        const recipes: []const manifest.Recipe = switch (p.found) {
            .same, .other => |l| l.recipes,
            else => &.{},
        };
        if (recipes.len == 0) return;
        if (p.case == .same and p.from_path) {
            if (manifest.configuredRoot()) |root_path| {
                var path_buf: [std.fs.max_path_bytes]u8 = undefined;
                if (manifest.archivePath(&path_buf, root_path, p.key)) |apath| {
                    if (std.c.access(apath.ptr, std.c.F_OK) == 0) {
                        if (cache.openArchive(apath)) |a| {
                            p.async_archive = a;
                            p.archive_loaded = true;
                            p.archive_recipes = recipes.len;
                        } else |_| {
                            log.warn("[pipelines] binary archive {s} unreadable ({s}); building from sushi.metallib\n", .{ dir, cache.lastMetalError() });
                        }
                    }
                } else |_| {}
            }
        }
        p.async_t0 = load_clock.now();
        p.async_job = cache.startWithArchive(recipes, p.threads, p.async_archive) catch |e| {
            log.warn("[pipelines] manifest {s}: the workers did not start ({s}); the warm forwards run and rebuild it\n", .{ dir, @errorName(e) });
            if (p.async_archive) |a| {
                a.release();
                p.async_archive = null;
            }
            p.case = .unreadable;
            return;
        };
    }

    /// Waits for `startAsync()` workers (calling `startAsync()` first if not yet started), installs preloaded
    /// pipelines into `stock`, and settles markers. True when every pipeline of this build's manifest succeeded.
    pub fn await(p: *Plan) bool {
        if (!p.async_started) p.startAsync();
        // A later warm adds to a manifest only where a complete one stands: the one found, or the one `store` writes.
        last = null;
        // Every way out settles the markers: this manifest's if it is trusted, and none of an earlier load's.
        var trusted = false;
        defer {
            registry.settleMarkers(trusted);
            p.settled_gen = registry.generation;
            if (last) |*l| {
                l.written_gen = registry.generation;
            }
        }
        settled_scope = if (p.root != null) p.key.scope else null;
        var dir_buf: [manifest.Key.dir_len]u8 = undefined;
        const dir = p.key.dirName(&dir_buf);
        const recipes: []const manifest.Recipe = switch (p.found) {
            .same, .other => |l| l.recipes,
            else => &.{},
        };
        if (p.async_job) |job| {
            p.async_job = null;
            const archive_used = p.async_archive != null;
            defer if (p.async_archive) |a| {
                a.release();
                p.async_archive = null;
            };
            const report = cache.finish(job);
            const wall = load_clock.now() - p.async_t0;
            p.made_ns = wall;
            p.made_count = report.made;
            const wall_ms = wall / std.time.ns_per_ms;
            if (report.failed > 0) {
                log.warn("[pipelines] manifest {s}: {d} of {d} recipes did not make ({s}); not trusted, the warm forwards run and rebuild it\n", .{ dir, report.failed, recipes.len, report.firstError() });
                p.case = .unreadable;
                return false;
            }
            trusted = p.case == .same;
            if (p.case == .same) {
                // Every recipe of the file made, so every one is a pipeline of this process: a manifest rewritten
                // after a late warm keeps the ones this load never launched.
                for (recipes) |r| registry.record(r);
                remember(p);
                log.info("[pipelines] made {d} pipelines from the manifest {s}{s} on {d} threads in {d} ms; an arch may skip its warm forwards (key {d} ms)\n", .{ report.made, dir, if (archive_used) " (binary archive)" else "", p.threads, wall_ms, p.key_ms });
                return true;
            }
            log.info("[pipelines] manifest of another build, recompiled {d} on {d} threads in {d} ms; the warm forwards run and the manifest {s} is rewritten (key {d} ms)\n", .{ report.made, p.threads, wall_ms, dir, p.key_ms });
            return false;
        }
        if (recipes.len > 0) return false;
        switch (p.case) {
            .none => log.info("[pipelines] no manifest; the warm forwards run and write {s} (key {d} ms)\n", .{ dir, p.key_ms }),
            .unreadable => log.warn("[pipelines] manifest unreadable ({s}); the warm forwards run and rebuild it (key {d} ms)\n", .{ p.why, p.key_ms }),
            else => log.info("[pipelines] manifest off ({s}); the warm forwards run\n", .{p.why}),
        }
        return false;
    }

    /// Makes the pipelines of the manifest found, on worker threads. True when it made every pipeline of this build's
    /// manifest: an arch may then skip the warm forwards it runs only to make them.
    pub fn settle(p: *Plan) bool {
        p.startAsync();
        return p.await();
    }

    /// After a complete warm: the recipes the process made, written for this key, and the older builds of the scope
    /// removed. A failure is a warning, not a failed load.
    pub fn store(p: *Plan) void {
        p.storeWithArchive(true);
    }

    /// Writes the manifest and prunes older builds, deferring `archive.bin` until `finalizeLate()` (used by archs
    /// whose vision tower and served warm run after their initial arch warm so `archive.bin` is written once with all recipes).
    pub fn storeManifestOnly(p: *Plan) void {
        p.storeWithArchive(false);
    }

    fn storeWithArchive(p: *Plan, write_archive: bool) void {
        const root = p.root orelse return;
        const total = registry.recipes().len;
        const archive_needed = write_archive and p.from_path and (!p.archive_loaded or total > p.archive_recipes);
        remember(p);
        // A manifest found whole and left as it was still has the older builds of its scope removed.
        if (p.case == .same and registry.generation == p.settled_gen and !archive_needed) return pruneOlder(p, root);
        if (!write(p.a, p.io, root, p.key)) return;
        if (last) |*l| l.written_gen = registry.generation;
        var dir_buf: [manifest.Key.dir_len]u8 = undefined;
        const dir = p.key.dirName(&dir_buf);
        log.info("[pipelines] manifest {s} written: {d} recipes\n", .{ dir, total });
        pruneOlder(p, root);
        if (archive_needed) {
            if (writeArchiveForKey(p.io, p.key, dir)) |n| {
                p.archive_loaded = true;
                p.archive_recipes = total;
                if (last) |*l| {
                    l.archive_loaded = true;
                    l.archive_recipes = total;
                }
                _ = n;
            }
        }
    }
};

fn pruneOlder(p: *const Plan, root: std.Io.Dir) void {
    var removed: std.ArrayList([manifest.Key.dir_len]u8) = .empty;
    defer removed.deinit(p.a);
    manifest.prune(p.a, p.io, root, p.key, &removed) catch |e| log.warn("[pipelines] older manifests not removed: {s}\n", .{@errorName(e)});
    for (removed.items) |name| log.info("[pipelines] removed the manifest {s} of an older build\n", .{name});
}

fn writeArchiveForKey(io: std.Io, key: manifest.Key, dir: []const u8) ?usize {
    const root_path = manifest.configuredRoot() orelse return null;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const apath = manifest.archivePath(&path_buf, root_path, key) catch return null;
    const t0 = load_clock.now();
    if (cache.writeArchive(io, registry.recipes(), apath)) |n| {
        log.info("[pipelines] binary archive {s} written: {d} pipelines in {d} ms\n", .{ dir, n, (load_clock.now() - t0) / std.time.ns_per_ms });
        return n;
    } else |e| {
        log.warn("[pipelines] binary archive {s} not written ({s}: {s})\n", .{ dir, @errorName(e), cache.lastMetalError() });
        return null;
    }
}

const OpenedRoot = struct { dir: std.Io.Dir, owned: bool };

fn reopenRoot(io: std.Io) ?OpenedRoot {
    var reason: []const u8 = "";
    const dir = openRoot(io, &reason) orelse return null;
    return .{ .dir = dir, .owned = root_hook == null };
}

/// Where this load's manifest lives once it holds a complete warm, for a warm that runs after the load's own (the
/// vision tower's). A root made from the process's path is opened again at each write, so a root removed and made
/// again is never a stale handle.
var last: ?struct {
    a: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    from_path: bool,
    key: manifest.Key,
    written_gen: u64 = 0,
    archive_loaded: bool = false,
    archive_recipes: usize = 0,
} = null;

/// The scope of the manifest the load in progress settled, which its warm keys its markers by; null when no manifest
/// can hold them (no root, or no scope).
var settled_scope: ?[16]u8 = null;

pub fn settledScope() ?[16]u8 {
    return settled_scope;
}

/// The key the load's manifest is written under, once it holds a complete warm.
pub fn writtenKey() ?manifest.Key {
    return if (last) |l| l.key else null;
}

fn remember(p: *const Plan) void {
    last = if (p.root) |root| .{
        .a = p.a,
        .io = p.io,
        .root = root,
        .from_path = p.from_path,
        .key = p.key,
        .written_gen = registry.generation,
        .archive_loaded = p.archive_loaded,
        .archive_recipes = p.archive_recipes,
    } else null;
}

fn write(a: std.mem.Allocator, io: std.Io, root: std.Io.Dir, key: manifest.Key) bool {
    if (registry.lost > 0) {
        log.warn("[pipelines] manifest not written: {d} recipes could not be kept\n", .{registry.lost});
        return false;
    }
    manifest.write(a, io, root, key, registry.recipes()) catch |e| {
        log.warn("[pipelines] manifest not written: {s}\n", .{@errorName(e)});
        return false;
    };
    return true;
}

/// The model whose manifest this is unloaded: no manifest is rewritten until a load settles the next, and a marker
/// recorded meanwhile (a vision tower's) is that load's.
pub fn modelUnloaded() void {
    last = null;
    registry.manifestClosed();
}

/// A test's directory is gone with the test: nothing is rewritten after it.
pub fn forget() void {
    last = null;
}

/// Pipelines made after the load's manifest was settled or written: the manifest of the load's key, written again with
/// them. A load with no manifest root writes nothing.
pub fn rewriteLate() void {
    rewriteLateImpl(false);
}

/// Called at the end of `warmLoadWith` when an arch owns its own manifest (DeepSeek-V4.1): skips disk I/O when nothing
/// changed and `archive.bin` already covers every recipe; otherwise writes the complete manifest and binary archive once.
pub fn finalizeLate() void {
    rewriteLateImpl(true);
}

fn rewriteLateImpl(with_archive: bool) void {
    const l = &(last orelse return);
    const total = registry.recipes().len;
    const archive_needed = with_archive and l.from_path and (!l.archive_loaded or total > l.archive_recipes);
    const manifest_needed = registry.generation != l.written_gen;
    if (with_archive and !manifest_needed and !archive_needed) return;
    const opened: ?OpenedRoot = if (l.from_path) reopenRoot(l.io) else .{ .dir = l.root, .owned = false };
    const root = opened orelse {
        log.warn("[pipelines] manifest not written: the manifest root did not open\n", .{});
        return;
    };
    defer if (root.owned) root.dir.close(l.io);
    var dir_buf: [manifest.Key.dir_len]u8 = undefined;
    const dir = l.key.dirName(&dir_buf);
    if (!with_archive or manifest_needed) {
        if (!write(l.a, l.io, root.dir, l.key)) return;
        l.written_gen = registry.generation;
        log.info("[pipelines] manifest {s} written again with a later warm: {d} recipes\n", .{ dir, total });
    }
    if (archive_needed) {
        if (writeArchiveForKey(l.io, l.key, dir) != null) {
            l.archive_loaded = true;
            l.archive_recipes = total;
        }
    }
}

/// Looks up this key's manifest. A problem with the directory or the
/// file is a case, never an error.
pub fn begin(a: std.mem.Allocator, io: std.Io, root: ?std.Io.Dir, why_no_root: []const u8, key: manifest.Key, key_ms: u64) Plan {
    var plan: Plan = .{ .a = a, .io = io, .root = root, .key = key, .key_ms = key_ms, .found = .none, .case = .off, .why = why_no_root, .threads = threadCount() };
    const dir = root orelse return plan;
    plan.found = manifest.lookup(a, io, dir, key) catch |e| {
        plan.why = @errorName(e);
        plan.found = .none;
        return plan;
    };
    switch (plan.found) {
        .none => plan.case = .none,
        .unreadable => |u| {
            plan.case = .unreadable;
            plan.why = u.reason;
        },
        .same => plan.case = .same,
        .other => plan.case = .other_build,
    }
    return plan;
}

/// The plan of a load whose scope is `scope` (an arch's `coolerPipelineScope`), or none with the reason.
pub fn beginScope(a: std.mem.Allocator, io: std.Io, scope: anyerror![16]u8) Plan {
    const none: manifest.Key = .{ .scope = @splat('0'), .build = @splat('0') };
    const t0 = load_clock.now();
    const s = scope catch |e| return begin(a, io, null, @errorName(e), none, 0);
    const build = buildKey(a, io) catch |e| return begin(a, io, null, @errorName(e), none, 0);
    const key_ms = (load_clock.now() - t0) / std.time.ns_per_ms;
    var reason: []const u8 = "";
    const root = openRoot(io, &reason);
    var plan = begin(a, io, root, reason, .{ .scope = s, .build = build }, key_ms);
    plan.from_path = root != null;
    plan.owns_root = root != null and root_hook == null;
    return plan;
}

/// The plan of a load of `config` whose arch names its scope (`coolerPipelineScope`); null for an arch that settles its
/// manifest in its own load (DeepSeek's decides there which warm forwards run).
pub fn beginFor(a: std.mem.Allocator, io: std.Io, config: *const @import("../model.zig").ModelConfig) ?Plan {
    const tag = @import("config.zig").tagOf(config) orelse return null;
    return switch (tag) {
        inline else => |t| if (comptime @hasDecl(@import("registry.zig").Mod(t), "coolerPipelineScope"))
            beginScope(a, io, @import("registry.zig").Mod(t).coolerPipelineScope(config))
        else
            null,
    };
}

/// Late pipelines a manifest has: a request that made more rewrites it (`rewriteLate`), so the next load makes them.
var late_written: u64 = 0;

/// A request ended: the manifest takes what it compiled late.
pub fn requestEnded() void {
    const late = mlx.engine.latePipelines();
    if (late == late_written) return;
    late_written = late;
    rewriteLate();
}

test "pipelines: a load that leaves its manifest as it found it still removes the older builds of its scope but the newest" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const scope: [16]u8 = @splat('a');
    const mine: manifest.Key = .{ .scope = scope, .build = @splat('3') };
    const older: manifest.Key = .{ .scope = scope, .build = @splat('1') };
    const old: manifest.Key = .{ .scope = scope, .build = @splat('2') };
    for ([_]manifest.Key{ older, old, mine }, 1..) |k, second| {
        try manifest.write(a, io, tmp.dir, k, &.{});
        try manifest.setWrittenForTest(io, tmp.dir, k, @intCast(second * 1000));
    }
    var p = begin(a, io, tmp.dir, "", mine, 0);
    defer p.deinit();
    defer forget();
    try std.testing.expectEqual(Case.same, p.case);
    p.settled_gen = registry.generation;
    p.store();
    var buf: [manifest.Key.dir_len]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, tmp.dir.access(io, older.dirName(&buf), .{}));
    for ([_]manifest.Key{ old, mine }) |k| try tmp.dir.access(io, k.dirName(&buf), .{});
}
