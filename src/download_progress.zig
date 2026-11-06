//! The MLX-free half of the downloads (`downloads.zig`): which repo names a download may take, the org listing,
//! the free-space verdict, a download's byte accounting from the pull's events, and its speed.
const std = @import("std");
const hf_pull = @import("hf_pull.zig");

pub const Phase = enum {
    queued,
    listing,
    checking,
    downloading,
    verifying,
    done,
    failed,
    cancelled,

    pub fn active(p: Phase) bool {
        return p != .done and p != .failed and p != .cancelled;
    }
};

/// `<org>/<name>` with a name of `[A-Za-z0-9._-]` that is not `.` or `..`: it becomes a folder under the models root.
pub fn validRepo(org: []const u8, repo: []const u8) bool {
    if (repo.len <= org.len + 1 or !std.ascii.startsWithIgnoreCase(repo, org) or repo[org.len] != '/') return false;
    const name = repo[org.len + 1 ..];
    if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '.' and c != '_' and c != '-') return false;
    return true;
}

/// The org's public Sushi repos in a `/api/models?author=<org>` answer: a pack published after this build is listed
/// without being in the alias table.
pub fn parseOrgPacks(allocator: std.mem.Allocator, json: []const u8, org: []const u8) ![][]u8 {
    const Row = struct { id: []const u8 = "", private: bool = false };
    const parsed = std.json.parseFromSlice([]const Row, allocator, json, .{ .ignore_unknown_fields = true }) catch return error.InvalidListing;
    defer parsed.deinit();
    var ids = std.ArrayList([]u8).empty;
    errdefer {
        for (ids.items) |id| allocator.free(id);
        ids.deinit(allocator);
    }
    for (parsed.value) |r| {
        if (r.private or !validRepo(org, r.id) or std.ascii.findIgnoreCase(r.id, "-Sushi-") == null) continue;
        try ids.append(allocator, try allocator.dupe(u8, r.id));
    }
    return ids.toOwnedSlice(allocator);
}

pub fn freeOrgPacks(allocator: std.mem.Allocator, ids: []const []u8) void {
    for (ids) |id| allocator.free(id);
    allocator.free(ids);
}

/// A download from another machine takes the API key: without `--api-key` only this Mac may start one.
pub fn startRefused(loopback: bool, has_key: bool) bool {
    return !loopback and !has_key;
}

pub const Space = enum { fits, low, short };

/// The free space a download must leave: a twentieth of the volume, at least the 4 GiB the SSD cache keeps too.
pub fn reserve(total: u64) u64 {
    return @max(total / 20, 4 << 30);
}

/// `short`: the download would eat into the reserve. `low`: it leaves under a tenth of the volume free.
pub fn spaceVerdict(need: u64, free: u64, total: u64) Space {
    if (need +| reserve(total) > free) return .short;
    return if (free - need < total / 10) .low else .fits;
}

/// Download speed from the bytes fetched, sampled when the page polls: the average over the last few seconds.
pub const Meter = struct {
    const cap = 16;
    const window_ms = 5000;
    t: [cap]i64 = undefined,
    b: [cap]u64 = undefined,
    n: usize = 0,

    pub fn add(self: *Meter, now_ms: i64, bytes: u64) void {
        if (self.n > 0 and now_ms - self.t[self.n - 1] < 250) return;
        if (self.n == cap) {
            std.mem.copyForwards(i64, self.t[0 .. cap - 1], self.t[1..]);
            std.mem.copyForwards(u64, self.b[0 .. cap - 1], self.b[1..]);
            self.n -= 1;
        }
        self.t[self.n] = now_ms;
        self.b[self.n] = bytes;
        self.n += 1;
    }

    /// Bytes per second from the oldest sample inside the window (or the newest one before it) to the newest.
    pub fn rate(self: *const Meter) ?f64 {
        if (self.n < 2) return null;
        const last = self.n - 1;
        var first: usize = last - 1;
        while (first > 0 and self.t[last] - self.t[first - 1] <= window_ms) first -= 1;
        const ms = self.t[last] - self.t[first];
        if (ms <= 0 or ms > 6 * window_ms) return null;
        return @as(f64, @floatFromInt(self.b[last] -| self.b[first])) * 1000.0 / @as(f64, @floatFromInt(ms));
    }
};

/// A download's bytes from the pull's events. `done` counts the files finished or found complete; the file in flight
/// adds its partial, of which only what this run fetched counts toward the speed.
pub const Progress = struct {
    files: usize = 0,
    file_index: usize = 0,
    total: u64 = 0,
    done: u64 = 0,
    file: []u8 = &.{},
    file_size: u64 = 0,
    /// The partial's size when its file started: a resumed file counts from there.
    resume_from: u64 = 0,
    /// Bytes this run fetched in finished files.
    fetched: u64 = 0,

    pub fn deinit(self: *Progress, allocator: std.mem.Allocator) void {
        allocator.free(self.file);
        self.file = &.{};
    }

    pub fn finishFile(self: *Progress, allocator: std.mem.Allocator) void {
        if (self.file.len == 0) return;
        self.done += self.file_size;
        self.fetched += self.file_size -| self.resume_from;
        self.deinit(allocator);
    }

    /// The pull ended; `ok` false (a failure or a stop) leaves the file in flight unfinished.
    pub fn endFile(self: *Progress, allocator: std.mem.Allocator, ok: bool) void {
        if (ok) return self.finishFile(allocator);
        self.deinit(allocator);
    }

    /// File `f` starts, with `partial` bytes of it already on disk.
    pub fn startFile(self: *Progress, allocator: std.mem.Allocator, f: hf_pull.FileEvent, partial: u64) !void {
        self.finishFile(allocator);
        self.file_index = f.index;
        if (f.skip) {
            self.done += f.size;
            return;
        }
        self.file = try allocator.dupe(u8, f.path);
        self.file_size = f.size;
        self.resume_from = @min(partial, f.size);
    }

    /// Done and fetched bytes with `partial` bytes of the file in flight on disk.
    pub fn now(self: *const Progress, partial: u64) struct { done: u64, fetched: u64 } {
        const cur = if (self.file.len == 0) 0 else @min(partial, self.file_size);
        return .{ .done = self.done + cur, .fetched = self.fetched + (cur -| self.resume_from) };
    }
};

const testing = std.testing;

test "downloads: a repo name is one plain folder under the org" {
    for ([_][]const u8{ "beamster/Qwen3.8-Flash-Next-Sushi-2bpw", "Beamster/a_b.c-d" }) |r| try testing.expect(validRepo("beamster", r));
    for ([_][]const u8{ "beamster/..", "beamster/.", "beamster/", "beamster/a/b", "beamster/a b", "other/x", "beamsterx/y", "beamster/a\x00" }) |r|
        try testing.expect(!validRepo("beamster", r));
}

test "org listing: only the org's public Sushi repos, in the API's order" {
    const ids = try parseOrgPacks(testing.allocator,
        \\[{"id":"beamster/New-Pack-Sushi-3bpw","private":false},
        \\ {"id":"beamster/Secret-Sushi-1bpw","private":true},
        \\ {"id":"beamster/some-dataset-tool"},
        \\ {"id":"other/Fake-Sushi-2bpw","private":false},
        \\ {"id":"beamster/a/b-Sushi-x"},
        \\ {"id":"beamster/..-Sushi-/.."},
        \\ {"id":"beamster/Qwen3.8-Flash-Next-Sushi-2bpw"}]
    , "beamster");
    defer freeOrgPacks(testing.allocator, ids);
    try testing.expectEqual(@as(usize, 2), ids.len);
    try testing.expectEqualStrings("beamster/New-Pack-Sushi-3bpw", ids[0]);
    try testing.expectEqualStrings("beamster/Qwen3.8-Flash-Next-Sushi-2bpw", ids[1]);
    try testing.expectError(error.InvalidListing, parseOrgPacks(testing.allocator, "{}", "beamster"));
}

test "downloads: a start from another machine needs the API key" {
    try testing.expect(!startRefused(true, false));
    try testing.expect(!startRefused(false, true));
    try testing.expect(startRefused(false, false));
}

test "downloads: a pack that eats into the reserve is refused, one that leaves under a tenth of the disk warns" {
    const free: u64 = 300 << 30;
    const total: u64 = 1000 << 30;
    try testing.expectEqual(Space.fits, spaceVerdict(100 << 30, free, total));
    try testing.expectEqual(Space.low, spaceVerdict(240 << 30, free, total));
    try testing.expectEqual(Space.short, spaceVerdict(251 << 30, free, total)); // would leave under 50 GiB, a twentieth
    try testing.expectEqual(Space.short, spaceVerdict(2 << 30, 5 << 30, 40 << 30)); // a small disk keeps 4 GiB
    try testing.expectEqual(Space.short, spaceVerdict(std.math.maxInt(u64), free, total));
}

test "downloads: speed is the average over the last few seconds of samples" {
    var m: Meter = .{};
    try testing.expect(m.rate() == null);
    m.add(0, 0);
    try testing.expect(m.rate() == null);
    m.add(100, 999); // under 250 ms after the last sample: dropped
    m.add(1000, 1000);
    try testing.expectApproxEqAbs(@as(f64, 1000), m.rate().?, 0.01);
    var t: i64 = 2000;
    while (t <= 20_000) : (t += 1000) m.add(t, @intCast(t * 3 - 2000));
    // Only the last five seconds count: 3000 B/s, not the slow start.
    try testing.expectApproxEqAbs(@as(f64, 3000), m.rate().?, 0.01);
    m.add(200_000, 1 << 40); // a gap of minutes is no speed
    try testing.expect(m.rate() == null);
}

test "downloads: progress counts skipped and finished files, and a resumed partial as done but not fetched" {
    var p: Progress = .{ .files = 3, .total = 30 };
    defer p.deinit(testing.allocator);
    try p.startFile(testing.allocator, .{ .index = 1, .path = "a.json", .size = 10, .skip = true }, 0);
    try p.startFile(testing.allocator, .{ .index = 2, .path = "b.safetensors", .size = 10, .skip = false }, 4);
    const mid = p.now(7);
    try testing.expectEqual(@as(u64, 17), mid.done);
    try testing.expectEqual(@as(u64, 3), mid.fetched);
    try p.startFile(testing.allocator, .{ .index = 3, .path = "c.safetensors", .size = 10, .skip = false }, 0);
    try testing.expectEqual(@as(u64, 20), p.done);
    try testing.expectEqual(@as(u64, 6), p.fetched);
    try testing.expectEqual(@as(u64, 30), p.now(99).done);
    p.finishFile(testing.allocator);
    try testing.expectEqual(@as(u64, 30), p.done);
}

test "downloads: a pull that fails or is stopped mid-file does not count that file as done" {
    var p: Progress = .{ .files = 2, .total = 20 };
    defer p.deinit(testing.allocator);
    try p.startFile(testing.allocator, .{ .index = 1, .path = "a.safetensors", .size = 10, .skip = false }, 0);
    try p.startFile(testing.allocator, .{ .index = 2, .path = "b.safetensors", .size = 10, .skip = false }, 4);
    p.endFile(testing.allocator, false);
    try testing.expectEqual(@as(u64, 10), p.done);
    try testing.expectEqual(@as(u64, 10), p.fetched);
    try testing.expectEqual(@as(u64, 10), p.now(0).done);
    try p.startFile(testing.allocator, .{ .index = 2, .path = "b.safetensors", .size = 10, .skip = false }, 4);
    p.endFile(testing.allocator, true);
    try testing.expectEqual(@as(u64, 20), p.done);
    try testing.expectEqual(@as(u64, 16), p.fetched);
}
