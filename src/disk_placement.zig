const transformer_mod = @import("transformer.zig");
const KVCache = transformer_mod.KVCache;
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const std = @import("std");
const testing = std.testing;
const kv_disk_cache = @import("kv_disk_cache.zig");
const model_discovery = @import("model_discovery.zig");

/// One piece of a writer-armed commit: the bytes it stages before its manifest commits. The
/// commit runs pieces until the entry is whole, so a crash keeps every finished piece.
pub const FLUSH_PIECE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Guard against a commit that never completes: 64 pieces is 128 GiB, past any tier.
pub const MAX_FLUSH_PIECES: usize = 64;

/// Free space the tier never uses, for resolution, every store and the report alike. Sizes are binary
/// (1 GB = 1 GiB), as the flag parser reads them.
pub const DISK_FREE_RESERVE: u64 = 4 * 1024 * 1024 * 1024;

/// What the volume leaves the tier: free space less the reserve.
pub fn freeDiskRoom(free_bytes: u64) u64 {
    return free_bytes -| DISK_FREE_RESERVE;
}

/// Bytes this tier may occupy given the operator cap (0 = none) and the volume's free space; null when
/// nothing is left ("do not store"), never 0.
pub fn diskBudgetFromFreeSpace(operator_cap: u64, free_bytes: u64) ?u64 {
    const avail = freeDiskRoom(free_bytes);
    const budget = if (operator_cap == 0) avail else @min(operator_cap, avail);
    if (budget == 0) return null;
    return budget;
}

/// The unnamed SSD budget's ceiling, the slack added to the per-token formula, and the free space it leaves.
pub const DEFAULT_DISK_CAP: u64 = 20 * 1024 * 1024 * 1024;

pub const DEFAULT_DISK_SLACK: u64 = 2 * 1024 * 1024 * 1024;

/// Which term set the SSD budget.
pub const DiskBound = enum {
    formula,
    cap,
    free_disk,
    flag,

    pub fn label(self: DiskBound) []const u8 {
        return switch (self) {
            .formula => "the formula",
            .cap => "20 GB cap",
            .free_disk => "free disk - 4 GB",
            .flag => "the flag",
        };
    }
};

/// `bytes` is the budget now; `cap` is the operator's ceiling (the flag, or the formula / 20 GB cap), never
/// derived from free space, which the tier re-reads before every store.
pub const DefaultDiskBudget = struct { bytes: u64, bound: DiskBound, cap: u64 };

/// An operator's `--prefix-cache-disk` is the budget as given; unnamed it is
/// `min(entries x context x bytes_per_token + 2 GB, 20 GB, free + held - 4 GB)`, where `held` is what the tier
/// already occupies on the volume. Null when that is not positive.
pub fn resolveDiskBudget(explicit: bool, ask: u64, entries: u32, context: u64, bytes_per_token: u64, free_bytes: u64, held_bytes: u64) ?DefaultDiskBudget {
    if (explicit) return if (ask == 0) null else .{ .bytes = ask, .bound = .flag, .cap = ask };
    const formula = @as(u64, entries) *| context *| bytes_per_token +| DEFAULT_DISK_SLACK;
    const room = freeDiskRoom(free_bytes +| held_bytes);
    if (room == 0) return null;
    const cap = @min(formula, DEFAULT_DISK_CAP);
    var out: DefaultDiskBudget = .{ .bytes = cap, .bound = if (DEFAULT_DISK_CAP < formula) .cap else .formula, .cap = cap };
    if (room < cap) out = .{ .bytes = room, .bound = .free_disk, .cap = cap };
    return out;
}

fn toGb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

/// The startup line for the SSD tier: the sizing terms and the bound that set the number.
pub fn allocatingDiskLine(buf: []u8, plan: DefaultDiskBudget, entries: u32, context: u64, bytes_per_token: u64, dir: []const u8) []const u8 {
    const text = if (plan.bound == .flag)
        std.fmt.bufPrint(buf, "Allocating {d:.1} GB SSD for the prefix cache (--prefix-cache-disk) at {s}", .{ toGb(plan.bytes), dir })
    else
        std.fmt.bufPrint(buf, "Allocating {d:.1} GB SSD for the prefix cache ({d} entries x {d} tokens x {d:.1} KB + 2 GB; bound: {s}) at {s}", .{ toGb(plan.bytes), entries, context, @as(f64, @floatFromInt(bytes_per_token)) / 1024.0, plan.bound.label(), dir });
    return text catch buf[0..0];
}

/// The per-turn usage line, also what the web page shows.
pub fn diskUsageLine(buf: []u8, used: u64, budget: u64, entries: u64) []const u8 {
    return std.fmt.bufPrint(buf, "[disk-cache] usage {d:.2} / {d:.1} GB, {d} entries", .{ toGb(used), toGb(budget), entries }) catch buf[0..0];
}

/// The startup line when no SSD budget remains.
pub fn diskOffLine(buf: []u8, free_bytes: u64, dir: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "SSD prefix cache OFF: {d:.1} GB free at {s} leaves nothing after the 4 GB reserve", .{ toGb(free_bytes), dir }) catch buf[0..0];
}

/// `volumeSpace` of the nearest existing ancestor: the cache directory is created on first store.
pub fn volumeSpaceNear(path: []const u8) ?VolumeSpace {
    var p = path;
    while (true) {
        if (volumeSpace(p)) |v| return v;
        p = std.fs.path.dirname(p) orelse return null;
    }
}

/// macOS `struct statfs`, leading fields only; the rest is slack. std has no binding for it.
const DarwinStatfs = extern struct {
    f_bsize: u32,
    f_iosize: i32,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    tail: [4096]u8,
};
extern "c" fn statfs(path: [*:0]const u8, buf: *DarwinStatfs) c_int;

extern fn sushi_volume_free_for_use(path: [*:0]const u8) u64;

pub const VolumeSpace = struct { free: u64, total: u64 };

/// Free and total bytes of the volume holding `path`, or null when the query fails or returns
/// implausible numbers (the plausibility check is the ABI guard).
pub fn volumeSpace(path: []const u8) ?VolumeSpace {
    var buf: [std.fs.max_path_bytes + 1]u8 = undefined;
    if (path.len >= buf.len) return null;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    var st: DarwinStatfs = undefined;
    if (statfs(buf[0..path.len :0].ptr, &st) != 0) return null;
    const bsize: u64 = st.f_bsize;
    if (bsize < 512 or bsize > (1 << 20) or !std.math.isPowerOfTwo(bsize)) return null;
    if (st.f_blocks == 0 or st.f_bavail > st.f_blocks) return null;
    const total = bsize *| st.f_blocks;
    // statfs excludes purgeable space the OS releases on demand; ask what a write really gets.
    const granted = sushi_volume_free_for_use(buf[0..path.len :0].ptr);
    return .{ .free = freeForUse(granted, total, bsize *| st.f_bavail), .total = total };
}

/// Which of the two answers a write really gets: the OS grant when it is one, else statfs.
fn freeForUse(granted: u64, total: u64, statfs_free: u64) u64 {
    if (granted > 0 and granted <= total) return granted;
    return statfs_free;
}

/// How a `DiskTier` asks what the volume has left. Injectable: with the live probe hard-wired
/// every SSD-first test asserted a property of the tester's free disk space.
pub const SpaceProbeFn = *const fn (path: []const u8) ?VolumeSpace;

/// Test hook, armed through `DiskTier.armTestSpace`.
pub var test_space: ?VolumeSpace = null;

pub var test_qsa_overlay_mismatch = false;

pub var test_ssm_write_qsa_aux = false;

/// How many free-space probes the test hook answered.
var test_space_probes: usize = 0;

pub fn testSpaceProbe(path: []const u8) ?VolumeSpace {
    _ = path;
    test_space_probes += 1;
    return test_space;
}

/// Identity of the weights the persisted KV was computed against: absolute
/// model dir + config.json size/mtime. A re-downloaded or re-quantized
/// checkpoint rewrites config.json, which rolls the fingerprint and orphans
/// the stale KV (GC'd by the disk budget eventually; different fingerprint
/// dirs never mix). 16 hex chars of XxHash64.
/// Root-wide LRU sweep across sibling model fingerprints (never `keep_root`, which the tier's
/// own `scan` owns), oldest first until they fit `sibling_budget`, plus index-less strays that
/// are old enough to be crash leftovers. Best effort.
pub fn sweepBase(
    allocator: std.mem.Allocator,
    io: std.Io,
    base_dir: []const u8,
    keep_root: []const u8,
    sibling_budget: u64,
) void {
    var base = std.Io.Dir.openDirAbsolute(io, base_dir, .{ .iterate = true }) catch return;
    defer base.close(io);

    const Victim = struct { path: []u8, bytes: u64, mtime: i128 };
    var victims = std.ArrayList(Victim).empty;
    defer {
        for (victims.items) |v| allocator.free(v.path);
        victims.deinit(allocator);
    }
    var total: u64 = 0;
    var strays: usize = 0;
    var seen = InodeSet.init(allocator);
    defer seen.deinit();
    // Each swept root's lock is held to the end: a root a live tier owns is skipped, and no tier
    // can take one while its entries are being deleted.
    var held = std.ArrayList(std.Io.File).empty;
    defer {
        for (held.items) |f| f.close(io);
        held.deinit(allocator);
    }

    var fps = base.iterate();
    while (fps.next(io) catch null) |fp| {
        if (fp.kind != .directory) continue;
        const fp_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_dir, fp.name }) catch continue;
        defer allocator.free(fp_abs);
        if (std.mem.eql(u8, fp_abs, keep_root)) continue; // the live tier owns its own
        const lock = (lockRoot(io, fp_abs) catch continue) orelse continue;
        held.append(allocator, lock) catch {
            lock.close(io);
            continue;
        };
        reapPrivateRoots(allocator, io, fp_abs);

        var fpd = std.Io.Dir.openDirAbsolute(io, fp_abs, .{ .iterate = true }) catch continue;
        defer fpd.close(io);
        var es = fpd.iterate();
        while (es.next(io) catch null) |dent| {
            if (dent.kind != .directory) continue;
            if (dent.name.len < 2 or dent.name[0] != 'e') continue;
            const e_abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ fp_abs, dent.name }) catch continue;
            const meta = std.fmt.allocPrint(allocator, "{s}/meta.json", .{e_abs}) catch {
                allocator.free(e_abs);
                continue;
            };
            defer allocator.free(meta);
            const st = std.Io.Dir.cwd().statFile(io, meta, .{}) catch {
                // No index: a crash leftover, OR another server's flush in progress (meta lands
                // last). Age is the only signal.
                if (dirYoungerThan(io, e_abs, STRAY_MIN_AGE_NS)) {
                    allocator.free(e_abs);
                    continue;
                }
                deleteTreeAbsolute(io, e_abs);
                allocator.free(e_abs);
                strays += 1;
                continue;
            };
            // A `.tmp` older than the same bar is a crash leftover of the writer's tmp+rename.
            reapStaleTmp(io, e_abs);
            const bytes = dirBytes(io, e_abs, &seen);
            total += bytes;
            victims.append(allocator, .{ .path = e_abs, .bytes = bytes, .mtime = st.mtime.nanoseconds }) catch {
                allocator.free(e_abs);
            };
        }
    }

    if (strays > 0) log.info("  [disk-cache] swept {d} stray entry directories under {s}\n", .{ strays, base_dir });
    if (total <= sibling_budget) return;

    std.mem.sort(Victim, victims.items, {}, struct {
        fn lt(_: void, a: Victim, b: Victim) bool {
            return a.mtime < b.mtime;
        }
    }.lt);
    var freed: u64 = 0;
    for (victims.items) |v| {
        if (total -| freed <= sibling_budget) break;
        deleteTreeAbsolute(io, v.path);
        freed += v.bytes;
    }
    if (freed > 0) {
        log.info("  [disk-cache] root-wide LRU freed {d} MB across other models ({d} MB held, {d} MB budget)\n", .{
            freed >> 20,
            total >> 20,
            sibling_budget >> 20,
        });
    }
}

/// The flock that marks a tier root as owned by a live tier (`DiskTier.root_lock`).
const ROOT_LOCK_NAME = ".lock";

/// Distinguishes the private roots of several tiers in one process.
var private_root_seq = std.atomic.Value(u32).init(0);

/// Take `<root_abs>/.lock` without waiting; the returned file holds the lock until closed.
/// Null when a live tier holds it. The lock file is never unlinked (a racing open would lock a
/// second inode).
pub fn lockRoot(io: std.Io, root_abs: []const u8) !?std.Io.File {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/" ++ ROOT_LOCK_NAME, .{root_abs});
    return std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = false, .lock = .exclusive, .lock_nonblocking = true }) catch |err| switch (err) {
        error.WouldBlock => null,
        else => err,
    };
}

/// Does a live tier own `root_abs`? A root without a lock file has none.
pub fn rootIsLive(io: std.Io, root_abs: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/" ++ ROOT_LOCK_NAME, .{root_abs}) catch return true;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{ .lock = .exclusive, .lock_nonblocking = true }) catch |err|
        return err != error.FileNotFound;
    f.close(io);
    return false;
}

/// Create and lock a private root `<shared>/p<pid>-<n>` for a tier whose shared root is owned.
pub fn ownPrivateRoot(allocator: std.mem.Allocator, io: std.Io, shared: []const u8) !struct { root: []u8, lock: std.Io.File } {
    for (0..8) |_| {
        const root = try std.fmt.allocPrint(allocator, "{s}/p{d}-{d}", .{ shared, std.c.getpid(), private_root_seq.fetchAdd(1, .monotonic) });
        errdefer allocator.free(root);
        try std.Io.Dir.cwd().createDirPath(io, root);
        if (try lockRoot(io, root)) |lock| return .{ .root = root, .lock = lock };
        allocator.free(root);
    }
    return error.DiskCacheRootBusy;
}

/// Remove the private roots under `root_abs` whose tier is gone (a crash skips `deinit`). A young
/// one may belong to a tier between its mkdir and its lock.
pub fn reapPrivateRoots(allocator: std.mem.Allocator, io: std.Io, root_abs: []const u8) void {
    var d = std.Io.Dir.openDirAbsolute(io, root_abs, .{ .iterate = true }) catch return;
    defer d.close(io);
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .directory or dent.name.len < 2 or dent.name[0] != 'p') continue;
        const sub = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root_abs, dent.name }) catch continue;
        defer allocator.free(sub);
        const lock = (lockRoot(io, sub) catch continue) orelse continue;
        defer lock.close(io);
        if (dirYoungerThan(io, sub, STRAY_MIN_AGE_NS)) continue;
        log.info("  [disk-cache] removing {s}, the private SSD tier of a process that is gone\n", .{sub});
        d.deleteTree(io, dent.name) catch {};
    }
}

/// How old an index-less entry directory must be before a sweep may treat it as a crash leftover.
const STRAY_MIN_AGE_NS: i128 = 10 * 60 * @as(i128, std.time.ns_per_s);

/// True when any regular file directly inside `dir_abs` was modified within `age_ns`. Unreadable = young.
fn dirYoungerThan(io: std.Io, dir_abs: []const u8, age_ns: i128) bool {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return true;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch return true;
        if (now -| st.mtime.nanoseconds < age_ns) return true;
    }
    return false;
}

/// Delete `.tmp` files in `dir_abs` older than `STRAY_MIN_AGE_NS`.
fn reapStaleTmp(io: std.Io, dir_abs: []const u8) void {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return;
    defer d.close(io);
    const now = std.Io.Timestamp.now(io, .real).nanoseconds;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        if (!std.mem.endsWith(u8, dent.name, ".tmp")) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        if (now -| st.mtime.nanoseconds < STRAY_MIN_AGE_NS) continue;
        d.deleteFile(io, dent.name) catch {};
    }
}

/// Bytes the tier `<base>/<fingerprint>` already holds on the volume: the files of every `e<id>` directory.
pub fn tierBytes(io: std.Io, base_dir: []const u8, fingerprint: []const u8) u64 {
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = std.fmt.bufPrint(&root_buf, "{s}/{s}", .{ base_dir, fingerprint }) catch return 0;
    // Another live tier's bytes: a tier loading beside it keeps a private root (`DiskTier.init`).
    if (rootIsLive(io, root)) return 0;
    var d = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var seen = InodeSet.init(std.heap.page_allocator);
    defer seen.deinit();
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .directory or dent.name.len < 2 or dent.name[0] != 'e') continue;
        var e_buf: [std.fs.max_path_bytes]u8 = undefined;
        const e_abs = std.fmt.bufPrint(&e_buf, "{s}/{s}", .{ root, dent.name }) catch continue;
        total +|= dirBytes(io, e_abs, &seen);
    }
    return total;
}

/// Inodes already counted: chunk sharing hard-links one file into several entries.
pub const InodeSet = std.AutoHashMap(std.Io.File.INode, void);

/// Bytes of the regular files directly inside `dir_abs` whose inode `seen` has not counted yet.
pub fn dirBytes(io: std.Io, dir_abs: []const u8, seen: *InodeSet) u64 {
    var d = std.Io.Dir.openDirAbsolute(io, dir_abs, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |dent| {
        if (dent.kind != .file) continue;
        const st = d.statFile(io, dent.name, .{}) catch continue;
        if (st.nlink > 1 and (seen.fetchPut(st.inode, {}) catch null) != null) continue;
        total += st.size;
    }
    return total;
}

pub fn deleteTreeAbsolute(io: std.Io, dir_abs: []const u8) void {
    const parent = std.fs.path.dirname(dir_abs) orelse return;
    const name = std.fs.path.basename(dir_abs);
    var pd = std.Io.Dir.openDirAbsolute(io, parent, .{ .iterate = true }) catch return;
    defer pd.close(io);
    pd.deleteTree(io, name) catch {};
}

fn fingerprintFile(h: *std.hash.XxHash64, io: std.Io, dir: std.Io.Dir, name: []const u8) void {
    h.update(name);
    h.update(&[_]u8{0});
    const st = dir.statFile(io, name, .{}) catch {
        h.update(&[_]u8{0});
        return;
    };
    h.update(&[_]u8{1});
    h.update(std.mem.asBytes(&st.size));
    const mt: i128 = st.mtime.nanoseconds;
    h.update(std.mem.asBytes(&mt));
}

pub fn modelFingerprint(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8) ![]u8 {
    return modelFingerprintWithLayout(allocator, io, model_dir, null);
}

/// `layout` names a stored-state layout the model files do not tell apart; null keeps the historical fingerprint.
pub fn modelFingerprintWithLayout(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, layout: ?[]const u8) ![]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return error.BadModelDir;
    var h = std.hash.XxHash64.init(0x6b76_6361_6368_6531);
    h.update(model_dir);
    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(cfg_path);
    if (kv_disk_cache.statFile(io, cfg_path)) |st| {
        h.update(std.mem.asBytes(&st.size));
        const mt: i128 = st.mtime.nanoseconds;
        h.update(std.mem.asBytes(&mt));
    }
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var referenced = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    var names = std.ArrayList([]const u8).empty;
    defer {
        for (names.items) |name| allocator.free(name);
        names.deinit(allocator);
    }
    if (referenced) |r| {
        var it = r.keyIterator();
        while (it.next()) |name| {
            const owned = try allocator.dupe(u8, name.*);
            errdefer allocator.free(owned);
            try names.append(allocator, owned);
        }
    } else {
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file and entry.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            const owned = try allocator.dupe(u8, entry.name);
            errdefer allocator.free(owned);
            try names.append(allocator, owned);
        }
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    for (names.items) |name| fingerprintFile(&h, io, dir, name);
    fingerprintFile(&h, io, dir, "ngram_table.bin");
    if (model.getConfigOverrides()) |raw| h.update(raw);
    if (layout) |tag| h.update(tag);
    return std.fmt.allocPrint(allocator, "{x:0>16}", .{h.final()});
}

/// Persistence root; an explicit directory also isolates live cache tests.
pub fn defaultBaseDir(allocator: std.mem.Allocator) ![]u8 {
    return cacheBaseDir(allocator,
        if (std.c.getenv("HOME")) |h| std.mem.span(h) else null,
        if (std.c.getenv("SUSHI_PREFIX_CACHE_DIR")) |p| std.mem.span(p) else null);
}

fn cacheBaseDir(allocator: std.mem.Allocator, home: ?[]const u8, override: ?[]const u8) ![]u8 {
    if (override) |path| {
        if (path.len > 0) {
            if (!std.fs.path.isAbsolute(path)) return error.InvalidCacheDirectory;
            return allocator.dupe(u8, path);
        }
    }
    return std.fmt.allocPrint(allocator, "{s}/.sushi/kv-cache", .{home orelse return error.NoHome});
}

fn evictionDuringStoreRound(selected: usize, evicted: usize, ssm_only: bool) !void {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-evict", 0, 128);
    tier.ssd_first = true;
    const roomy: u64 = 1024 * 1024 * 1024 * 1024;
    tier.armTestSpace(roomy, 2 * roomy);

    const seeds = [3]f64{ 0.0, 100_000.0, 200_000.0 };
    var toks: [3][640]u32 = undefined;
    for (&toks, 0..) |*t, k| for (t, 0..) |*v, i| {
        v.* = @intCast(i + 7 + k * 100_000);
    };
    for (0..3) |k| {
        var c = try KVCache.init(testing.allocator, 2);
        defer c.deinit();
        try kv_disk_cache.fillCache(&c, s, 2, 600, 8, seeds[k], .float32);
        _ = try tier.appendCommit(c.entries, c.step, c.config, toks[k][0..600], false, null, s);
        tier.drainWriter();
    }
    try testing.expectEqual(@as(usize, 3), tier.entryCount());
    for (0..3) |k| try testing.expectEqualSlices(u32, toks[k][0..600], tier.entries.items[k].tokens);

    // Commit entry `selected`, extended to 640 tokens or with a spec sidecar, while the probe evicts `evicted`.
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 2, 640, 8, seeds[selected], .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try kv_disk_cache.fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    const snap: kv_disk_cache.SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    kv_disk_cache.test_evict_tier = &tier;
    kv_disk_cache.test_evict_id = tier.entries.items[evicted].id;
    tier.space_probe = kv_disk_cache.evictingSpaceProbe;
    const len: usize = if (ssm_only) 600 else 640;
    _ = try tier.appendCommitWithSpec(cache.entries, len, cache.config, toks[selected][0..len], false, null, null, if (ssm_only) snap else null, s);
    tier.drainWriter();
    try testing.expectEqual(@as(?u64, null), kv_disk_cache.test_evict_id);

    tier.deinit();
    var tier2 = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-evict", 0, 128);
    defer tier2.deinit();
    for (0..3) |k| {
        if (k == evicted and k != selected) continue;
        const m = tier2.bestMatch(&toks[k], false, kv_quant.KVQuantConfig.dense, .{}).?;
        const want: u32 = if (k == selected and !ssm_only) 640 else 600;
        try testing.expectEqual(want, m.usable);
        var restored = try KVCache.init(testing.allocator, 2);
        defer restored.deinit();
        _ = try tier2.restoreInto(&restored, m.idx, s);
        try testing.expectEqual(seeds[k], try kv_disk_cache.cacheValueAt(&restored, 0, 0, 0, s));
        try testing.expectEqual(seeds[k] + 8.0 * @as(f64, @floatFromInt(want - 1)), try kv_disk_cache.cacheValueAt(&restored, 0, want - 1, 0, s));
        // The sidecar belongs to the committed entry alone.
        const sidecar = tier2.entries.items[m.idx].spec_bytes > 0;
        try testing.expectEqual(ssm_only and k == selected, sidecar);
    }
}

test "prefix cache root: explicit absolute directory and default home path" {
    const a = std.testing.allocator;
    const explicit = try cacheBaseDir(a, null, "/tmp/sushi-cache-test");
    defer a.free(explicit);
    try std.testing.expectEqualStrings("/tmp/sushi-cache-test", explicit);
    const fallback = try cacheBaseDir(a, "/tmp/sushi-test-user", "");
    defer a.free(fallback);
    try std.testing.expectEqualStrings("/tmp/sushi-test-user/.sushi/kv-cache", fallback);
    try std.testing.expectError(error.InvalidCacheDirectory, cacheBaseDir(a, null, "relative"));
    try std.testing.expectError(error.NoHome, cacheBaseDir(a, null, null));
}

test "DiskTier: the root-wide sweep never deletes a root a live tier owns" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    var held = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-held", 0, 128);
    defer held.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try held.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const other = try std.fmt.allocPrint(testing.allocator, "{s}/fp-other", .{base});
    defer testing.allocator.free(other);

    sweepBase(testing.allocator, io, base, other, 0);
    try testing.expect(tmp.dir.statFile(io, "fp-held/e1/meta.json", .{}) catch null != null);
    try testing.expect(rootIsLive(io, held.root));
    // A load beside the live tier keeps a private root: none of those bytes are its own.
    try testing.expectEqual(@as(u64, 0), tierBytes(io, base, "fp-held"));
}

test "DiskTier: hard-linked chunks are counted once by the root-wide sweep and tierBytes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    const chunk: [1000]u8 = @splat(0);
    try tmp.dir.createDirPath(io, "fp-shared/e1");
    try tmp.dir.createDirPath(io, "fp-shared/e2");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-shared/e1/c000000.safetensors", .data = &chunk });
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-shared/e1/meta.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-shared/e2/meta.json", .data = "{}" });
    try std.Io.Dir.hardLink(tmp.dir, "fp-shared/e1/c000000.safetensors", tmp.dir, "fp-shared/e2/c000000.safetensors", io, .{});

    try testing.expectEqual(@as(u64, 1004), tierBytes(io, base, "fp-shared"));

    // 1004 B held, so a 1500 B budget deletes nothing.
    const other = try std.fmt.allocPrint(testing.allocator, "{s}/fp-other", .{base});
    defer testing.allocator.free(other);
    sweepBase(testing.allocator, io, base, other, 1500);
    try testing.expect(tmp.dir.statFile(io, "fp-shared/e1/meta.json", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-shared/e2/meta.json", .{}) catch null != null);
}

test "modelFingerprint: stable per path, rolls with config.json changes" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    const fp1 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp1);
    const fp2 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp2);
    try testing.expectEqualStrings(fp1, fp2);
    try testing.expectEqual(@as(usize, 16), fp1.len);

    // Rewriting config.json (re-download / re-quant) rolls the fingerprint.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"y\",\"pad\":1}" });
    const fp3 = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp3);
    try testing.expect(!std.mem.eql(u8, fp1, fp3));

    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, ""));
    try testing.expectError(error.BadModelDir, modelFingerprint(testing.allocator, io, "rel/path"));
}

test "modelFingerprint: rolls with --config-overrides" {
    defer model.setConfigOverrides(null);
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    try tmp.dir.createDirPath(io, "model-a");
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a/config.json", .data = "{\"model_type\":\"x\"}" });
    const dir_a = try std.fmt.allocPrint(testing.allocator, "{s}/model-a", .{base});
    defer testing.allocator.free(dir_a);

    model.setConfigOverrides(null);
    const fp_none = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_none);

    const yarn = "{\"text_config\":{\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0}}}";
    model.setConfigOverrides(yarn);
    const fp_over = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_over);
    try testing.expectEqual(@as(usize, 16), fp_over.len);
    try testing.expect(!std.mem.eql(u8, fp_none, fp_over));

    model.setConfigOverrides(yarn);
    const fp_again = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_again);
    try testing.expectEqualStrings(fp_over, fp_again);

    // Raw-bytes pin: a whitespace-different spelling of the same JSON is a
    // different fingerprint (canonicalizing would hide a YaRN boot restoring
    // an unscaled SSD prefix if the override was re-spelled).
    const yarn_ws = "{\"text_config\": {\"rope_parameters\": {\"rope_type\": \"yarn\", \"factor\": 4.0}}}";
    model.setConfigOverrides(yarn_ws);
    const fp_ws = try modelFingerprint(testing.allocator, io, dir_a);
    defer testing.allocator.free(fp_ws);
    try testing.expect(!std.mem.eql(u8, fp_over, fp_ws));
}

test "DiskTier: SSD-first stages the flush off-thread and indexes LAST" {
    // The commit returns with the whole entry still staged, and meta.json is submitted after
    // every chunk so the FIFO lands it last.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    try testing.expectEqual(FLUSH_PIECE_BYTES, tier.max_flush_bytes);
    tier.writer.?.setPaused(true);
    defer if (tier.writer) |w| w.setPaused(false);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 3, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    const complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(kv_disk_cache.PersistOutcome.persisted, complete);
    try testing.expect(tier.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(u64, 0), tier.writer.?.filesWritten());
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-ssd-writer/e1/meta.json", .{}));

    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |p| testing.allocator.free(p);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    try testing.expectEqual(@as(usize, 6), paths.items.len); // 5 chunks + meta
    for (paths.items[0 .. paths.items.len - 1]) |p| {
        try testing.expect(std.mem.indexOf(u8, p, "/c0000") != null);
    }
    try testing.expect(std.mem.endsWith(u8, paths.items[paths.items.len - 1], "/meta.json"));

    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 6), tier.writer.?.filesWritten());

    // The staged bytes are a real safetensors image: a fresh tier restores it.
    tier.deinit();
    var tier2 = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ssd-writer", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    try testing.expectEqual(@as(u32, 640), try tier2.restoreInto(&cache2, 0, s));
    for ([_]u32{ 0, 1, 2 }) |li| {
        for ([_]u32{ 0, 127, 128, 511, 639 }) |pos| {
            try testing.expectEqual(
                try kv_disk_cache.cacheValueAt(&cache, li, pos, 3, s),
                try kv_disk_cache.cacheValueAt(&cache2, li, pos, 3, s),
            );
        }
    }
}

test "resolveDiskBudget: the formula, the 20 GB cap and free disk - 4 GB bind in turn; no room is null; a flag wins" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    // 2 entries x 1000 tokens x 1 MB + 2 GB: the formula binds.
    const small = resolveDiskBudget(false, DEFAULT_DISK_CAP, 2, 1000, MB, 500 * GB, 0).?;
    try testing.expectEqual(@as(u64, 2000 * MB + 2 * GB), small.bytes);
    try testing.expectEqual(DiskBound.formula, small.bound);
    // 32 entries x 262144 tokens x 18.4 kB is far past 20 GB.
    const big = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, 500 * GB, 0).?;
    try testing.expectEqual(DEFAULT_DISK_CAP, big.bytes);
    try testing.expectEqual(DiskBound.cap, big.bound);
    // 10 GB free leaves 6.
    const tight = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, 10 * GB, 0).?;
    try testing.expectEqual(6 * GB, tight.bytes);
    try testing.expectEqual(DiskBound.free_disk, tight.bound);
    try testing.expectEqual(@as(?DefaultDiskBudget, null), resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, 4 * GB, 0));
    try testing.expectEqual(@as(?DefaultDiskBudget, null), resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, 1 * GB, 0));
    // Overflow saturates to the cap, never wraps.
    try testing.expectEqual(DEFAULT_DISK_CAP, resolveDiskBudget(false, 0, std.math.maxInt(u32), std.math.maxInt(u64), 1 << 40, 500 * GB, 0).?.bytes);
    // An explicit number is the budget, whatever the formula or the free space say; 0 is off.
    const flag = resolveDiskBudget(true, 50 * GB, 32, 262144, 18_400, 10 * GB, 0).?;
    try testing.expectEqual(@as(u64, 50 * GB), flag.bytes);
    try testing.expectEqual(DiskBound.flag, flag.bound);
    try testing.expectEqual(@as(?DefaultDiskBudget, null), resolveDiskBudget(true, 0, 32, 262144, 18_400, 500 * GB, 0));
}

test "diskUsageLine: used / budget in GB and the entry count" {
    var buf: [128]u8 = undefined;
    const GB: u64 = 1024 * 1024 * 1024;
    try testing.expectEqualStrings("[disk-cache] usage 3.50 / 20.0 GB, 5 entries", diskUsageLine(&buf, 3 * GB + GB / 2, 20 * GB, 5));
}

test "allocatingDiskLine: starts with Allocating and names the terms, the bound and the directory" {
    var buf: [256]u8 = undefined;
    const GB: u64 = 1024 * 1024 * 1024;
    const line = allocatingDiskLine(&buf, .{ .bytes = 20 * GB, .bound = .cap, .cap = 20 * GB }, 32, 262144, 18_400, "/c");
    try testing.expectEqualStrings("Allocating 20.0 GB SSD for the prefix cache (32 entries x 262144 tokens x 18.0 KB + 2 GB; bound: 20 GB cap) at /c", line);
    const flag = allocatingDiskLine(&buf, .{ .bytes = 12 * GB, .bound = .flag, .cap = 12 * GB }, 32, 0, 0, "/c");
    try testing.expectEqualStrings("Allocating 12.0 GB SSD for the prefix cache (--prefix-cache-disk) at /c", flag);
    try testing.expectEqualStrings("SSD prefix cache OFF: 3.0 GB free at /c leaves nothing after the 4 GB reserve", diskOffLine(&buf, 3 * GB, "/c"));
}

test "diskBudgetFromFreeSpace: the one reserve is 4 GiB, whatever the volume; nothing left stores nothing" {
    const GB: u64 = 1 << 30;
    try testing.expectEqual(@as(?u64, 1020 * GB), diskBudgetFromFreeSpace(0, 1024 * GB));
    try testing.expectEqual(@as(?u64, 100 * GB), diskBudgetFromFreeSpace(100 * GB, 1024 * GB));
    // A small budget is still a budget: free less 4 GiB, announced and stored alike.
    try testing.expectEqual(@as(?u64, 6 * GB), diskBudgetFromFreeSpace(0, 10 * GB));
    try testing.expectEqual(@as(?u64, GB / 2), diskBudgetFromFreeSpace(0, 4 * GB + GB / 2));
    // The operator's cap is a cap, not a promise the volume must keep.
    try testing.expectEqual(@as(?u64, 6 * GB), diskBudgetFromFreeSpace(500 * GB, 10 * GB));
    // Refuse, never a silent 0.
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(0, 4 * GB));
    try testing.expectEqual(@as(?u64, null), diskBudgetFromFreeSpace(500 * GB, 2 * GB));
    // Load-time resolution and the store-time bound read the same free space the same way.
    for ([_]u64{ 10 * GB, 5 * GB, 4 * GB + GB / 2, 3 * GB }) |free| {
        const announced = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 1 << 20, 1 << 20, free, 0);
        const stored = diskBudgetFromFreeSpace(if (announced) |a| a.cap else 0, free);
        try testing.expectEqual(if (announced) |a| @as(?u64, a.bytes) else null, stored);
    }
}

test "volumeSpace: the live probe is plausible or null (statfs ABI guard)" {
    // On this machine the probe must succeed: a null means the struct layout broke.
    const vs = volumeSpace("/") orelse return error.VolumeSpaceProbeFailed;
    try testing.expect(vs.total > 0);
    try testing.expect(vs.free <= vs.total);
    try testing.expect(vs.total > 1024 * 1024 * 1024); // a macOS root volume
}

test "volumeSpace: free is what the OS grants, never statfs' f_bavail" {
    // The choice, not the live numbers: two samples of free disk space are never equal under a suite that writes.
    const GB: u64 = 1024 * 1024 * 1024;
    try testing.expectEqual(@as(u64, 117 * GB), freeForUse(117 * GB, 500 * GB, 36 * GB));
    try testing.expectEqual(@as(u64, 36 * GB), freeForUse(0, 500 * GB, 36 * GB));
    try testing.expectEqual(@as(u64, 36 * GB), freeForUse(500 * GB + 1, 500 * GB, 36 * GB));
    // The grant probe is wired: absent it, every volume would fall back to f_bavail.
    try testing.expect(sushi_volume_free_for_use("/") > 0);
}

test "DiskTier: a budget the load line announced is a budget the store honours (10 GiB free on a 1 TiB volume)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);
    const free: u64 = 10 * 1024 * 1024 * 1024;
    const total: u64 = 1024 * 1024 * 1024 * 1024;
    const plan = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, free, 0).?;
    try testing.expectEqual(DiskBound.free_disk, plan.bound);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-announced", plan.bytes, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(free, total);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(!tier.store_declined);
    try testing.expectEqual(plan.bytes, tier.max_bytes);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.bestMatch(&tokens, false, cache.config, .{}).?.usable);
}

test "DiskTier: a restart on a nearly full volume keeps what the tier already holds" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);
    const total: u64 = 1024 * 1024 * 1024 * 1024;

    {
        var first = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-restart", 0, 128);
        defer first.deinit();
        first.ssd_first = true;
        first.armTestSpace(total / 2, total);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try kv_disk_cache.fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
        for ([_]u32{ 7, 90_000 }) |off| {
            var tokens: [640]u32 = undefined;
            for (&tokens, 0..) |*t, i| t.* = @intCast(i + off);
            _ = try first.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
        }
        first.drainWriter();
        try testing.expectEqual(@as(usize, 2), first.entryCount());
    }
    const held = tierBytes(io, base, "fp-restart");
    try testing.expect(held > 0);

    // Free space is the reserve plus half of what the tier holds: the entries alone are not "free".
    const free = DISK_FREE_RESERVE + held / 2;
    const blind = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, free, 0).?;
    try testing.expectEqual(held / 2, blind.bytes);
    const plan = resolveDiskBudget(false, DEFAULT_DISK_CAP, 32, 262144, 18_400, free, held).?;
    try testing.expectEqual(held + held / 2, plan.bytes);
    try testing.expectEqual(DiskBound.free_disk, plan.bound);
    try testing.expectEqual(DEFAULT_DISK_CAP, plan.cap);

    var second = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-restart", plan.bytes, 128);
    defer second.deinit();
    second.operator_cap = plan.cap;
    try testing.expectEqual(@as(usize, 2), second.entryCount());
    // The live bound holds the entries it was given and follows the free space back up toward the cap.
    second.armTestSpace(free, total);
    second.refreshDiskBudget();
    try testing.expectEqual(free - DISK_FREE_RESERVE + second.total_bytes, second.max_bytes);
    try testing.expectEqual(@as(usize, 2), second.entryCount());
    second.armTestSpace(free + 8 * 1024 * 1024 * 1024, total);
    second.refreshDiskBudget();
    try testing.expectEqual(free - DISK_FREE_RESERVE + second.total_bytes + 8 * 1024 * 1024 * 1024, second.max_bytes);

    // An explicit flag is the budget as given, whatever the volume holds or leaves.
    const flag = resolveDiskBudget(true, 3 * held, 32, 262144, 18_400, free, held).?;
    try testing.expectEqual(3 * held, flag.bytes);
    try testing.expectEqual(DiskBound.flag, flag.bound);
    try testing.expectEqual(3 * held, flag.cap);
}

test "DiskTier: the free-space probe runs only before a store, never for a copy already on disk" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-probe", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    const roomy: u64 = 1024 * 1024 * 1024 * 1024;
    tier.armTestSpace(roomy, 2 * roomy);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try kv_disk_cache.fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var p = test_space_probes;
    _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, null, s);
    tier.drainWriter();
    try testing.expect(test_space_probes > p);

    // Unchanged recommit: the tier already holds it.
    p = test_space_probes;
    try testing.expectEqual(kv_disk_cache.PersistOutcome.persisted, try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, null, s));
    try testing.expectEqual(p, test_space_probes);

    // Pending spec work is a store.
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try kv_disk_cache.fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    const snap: kv_disk_cache.SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    p = test_space_probes;
    _ = try tier.appendCommitWithSpec(cache.entries, 600, cache.config, tokens[0..600], false, null, null, snap, s);
    tier.drainWriter();
    try testing.expect(test_space_probes > p);

    // An extension is a store.
    p = test_space_probes;
    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(test_space_probes > p);

    // With nothing left after the reserve a copy already on disk still counts; a new entry declines.
    tier.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);
    try testing.expectEqual(kv_disk_cache.PersistOutcome.persisted, try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s));
    var other: [640]u32 = undefined;
    for (&other, 0..) |*t, i| t.* = @intCast(i + 90_000);
    try testing.expectEqual(kv_disk_cache.PersistOutcome.skipped, try tier.appendCommit(cache.entries, 640, cache.config, &other, false, null, s));
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
}

test "DiskTier: a budget refresh that evicts mid-commit cannot redirect the store to another entry" {
    // swapRemove moves the LAST entry into the hole: a saved index then names another entry or none.
    try evictionDuringStoreRound(0, 0, false);
    try evictionDuringStoreRound(2, 0, false);
    try evictionDuringStoreRound(0, 0, true);
    try evictionDuringStoreRound(2, 1, true);
}

test "DiskTier: the root-wide sweep drops strays and never touches the live tier's own root" {
    // A sibling's index-less entry is a crash leftover only once it is older than
    // `STRAY_MIN_AGE_NS` (another server's flush writes meta.json last); the live tier's own
    // root is skipped entirely.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try kv_disk_cache.tmpRoot(&tmp, io, &buf);

    {
        var sib = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-sibling", 0, 128);
        defer sib.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try kv_disk_cache.fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try sib.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }
    try tmp.dir.createDirPath(io, "fp-stray/e9");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-stray/e9/c000000.safetensors", .data = "orphan" });
    const aged: std.Io.Timestamp = .{
        .nanoseconds = std.Io.Timestamp.now(io, .real).nanoseconds - 2 * @as(i96, @intCast(STRAY_MIN_AGE_NS)),
    };
    try tmp.dir.setTimestamps(io, "fp-stray/e9/c000000.safetensors", .{ .modify_timestamp = .{ .new = aged } });
    // Its young twin: index-less but written just now, so it must survive.
    try tmp.dir.createDirPath(io, "fp-inflight/e3");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-inflight/e3/c000000.safetensors", .data = "another server" });
    var live = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-live", 0, 128);
    defer live.deinit();
    live.ssd_first = true;
    live.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    // The live tier's own root, mid-write-through: chunks, no index yet. Staged after init
    // (init's own `scan` rightly drops an index-less entry).
    try tmp.dir.createDirPath(io, "fp-live/e1");
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-live/e1/c000000.safetensors", .data = "inflight" });
    live.max_bytes = 1 << 40;
    sweepBase(testing.allocator, io, base, live.root, live.max_bytes);

    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-stray/e9/c000000.safetensors", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);

    // With a budget of zero the siblings' real entries go too, and the live root is still untouched.
    sweepBase(testing.allocator, io, base, live.root, 0);
    try testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "fp-sibling/e1/meta.json", .{}));
    try testing.expect(tmp.dir.statFile(io, "fp-live/e1/c000000.safetensors", .{}) catch null != null);
    try testing.expect(tmp.dir.statFile(io, "fp-inflight/e3/c000000.safetensors", .{}) catch null != null);
}
