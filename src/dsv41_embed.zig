//! DeepSeek-V4.1's token embedding as MLX's affine triple (`embed.weight` u32 codes, `embed.scales`, `embed.biases`),
//! read by row from the shard instead of held: a forward `pread`s the rows of its token ids, and the caller dequantizes
//! them exactly as the resident table would. The page cache keeps the common tokens' rows hot.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const expert_io = @import("expert_io.zig");
const engram = @import("dsv41_engram.zig");

pub const Ssd = struct {
    fd: c_int,
    rows: u64,
    /// Per tensor (codes, scales, biases): the bytes of one row and where row 0 sits in the shard.
    len: [3]u32,
    off: [3]u64,
    /// The dtype scales and biases are stored in.
    sdtype: mlx.mlx_dtype,
    bits: c_int,
    group: c_int,
    /// What the three tensors hold in the shard (the bytes a resident load would bill).
    stored: u64,

    pub fn close(self: *Ssd) void {
        _ = std.c.close(self.fd);
    }

    /// The triple in the checkpoint at `dir_path`, for a model of width `dim`.
    pub fn open(gpa: std.mem.Allocator, dir_path: []const u8, dim: u32) !Ssd {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const stored_types = [_][]const u8{ "BF16", "F16", "F32" };
        const w = try located(a, dir_path, "embed.weight", &.{"U32"});
        const s = try located(a, dir_path, "embed.scales", &stored_types);
        const b = try located(a, dir_path, "embed.biases", &stored_types);
        if (!std.mem.eql(u8, w.file, s.file) or !std.mem.eql(u8, w.file, b.file)) return error.Dsv41EmbedSplit;
        const rows = w.shape[0];
        const groups = s.shape[1];
        const words = w.shape[1];
        const ok = s.dtype == b.dtype and s.shape[0] == rows and b.shape[0] == rows and b.shape[1] == groups and
            groups > 0 and dim % groups == 0 and (words * 32) % dim == 0;
        if (!ok) return error.Dsv41WeightLayout;
        const esize: u32 = if (s.dtype == 2) 4 else 2;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, w.file }, 0);
        // Rows are read at random: read-ahead would pull their neighbours in.
        const fd = try expert_io.openHinted(path, .{ .nocache = false, .readahead_off = true });
        const srow: u32 = @as(u32, @intCast(groups)) * esize;
        const len = [3]u32{ @intCast(words * 4), srow, srow };
        return .{
            .fd = fd,
            .rows = rows,
            .len = len,
            .off = .{ w.start, s.start, b.start },
            .sdtype = switch (s.dtype) {
                0 => .bfloat16,
                1 => .float16,
                else => .float32,
            },
            .bits = @intCast(words * 32 / dim),
            .group = @intCast(dim / groups),
            .stored = rows * (len[0] + len[1] + len[2]),
        };
    }

    fn located(a: std.mem.Allocator, dir_path: []const u8, name: []const u8, dtypes: []const []const u8) !engram.Table.TensorAt {
        return engram.Table.locate(a, dir_path, name, dtypes) catch |e| switch (e) {
            error.EngramTableMissing, error.EngramTableHeader => error.Dsv41WeightLayout,
            else => e,
        };
    }

    /// Rows as they sit in the shard, one `[n][len]` buffer per tensor.
    pub const Raw = struct {
        w: []align(4) u8,
        s: []align(4) u8,
        b: []align(4) u8,

        fn alloc(gpa: std.mem.Allocator, n: usize, len: [3]u32) !Raw {
            const w = try gpa.alignedAlloc(u8, .@"4", n * len[0]);
            errdefer gpa.free(w);
            const s = try gpa.alignedAlloc(u8, .@"4", n * len[1]);
            errdefer gpa.free(s);
            return .{ .w = w, .s = s, .b = try gpa.alignedAlloc(u8, .@"4", n * len[2]) };
        }

        pub fn deinit(self: Raw, gpa: std.mem.Allocator) void {
            gpa.free(self.w);
            gpa.free(self.s);
            gpa.free(self.b);
        }

        fn part(self: Raw, k: usize) []u8 {
            return switch (k) {
                0 => self.w,
                1 => self.s,
                else => self.b,
            };
        }
    };

    /// Row `ids[i]` into position i, each distinct id read once (host only, any thread). An id past the table is an error.
    pub fn read(self: *const Ssd, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        const n = ids.len;
        const raw = try Raw.alloc(gpa, n, self.len);
        errdefer raw.deinit(gpa);
        const first = try gpa.alloc(u32, n);
        defer gpa.free(first);
        // Out of memory for the map reads every position, as a batch without repeats does.
        const repeats = engram.firstOccurrences(gpa, ids, first) catch 0;
        if (repeats == 0) {
            try self.readInto(ids, null, raw);
            return raw;
        }
        const only = try gpa.alloc(u32, n - repeats);
        defer gpa.free(only);
        var k: usize = 0;
        for (first, 0..) |f, i| if (f == i) {
            only[k] = @intCast(i);
            k += 1;
        };
        try self.readInto(ids, only, raw);
        for (0..3) |p| {
            const l = self.len[p];
            const dst = raw.part(p);
            for (first, 0..) |f, i| if (f != i) @memcpy(dst[i * l ..][0..l], dst[f * l ..][0..l]);
        }
        return raw;
    }

    const rows_per_thread = 16;
    const max_threads = 16;

    fn readInto(self: *const Ssd, ids: []const u32, only: ?[]const u32, raw: Raw) !void {
        const n = if (only) |o| o.len else ids.len;
        const job: Job = .{ .t = self, .ids = ids, .only = only, .raw = raw };
        const parts = std.math.clamp(n / rows_per_thread, 1, max_threads);
        if (parts == 1) {
            if (n > 0 and !job.run(0, 1)) return error.Dsv41EmbedRead;
            return;
        }
        var threads: [max_threads]?std.Thread = @splat(null);
        for (threads[1..parts], 1..) |*t, k| t.* = std.Thread.spawn(.{}, Job.runVoid, .{ &job, k, parts }) catch null;
        var failed = !job.run(0, parts);
        for (threads[1..parts], 1..) |t, k| {
            if (t) |th| th.join() else failed = !job.run(k, parts) or failed;
        }
        if (failed or job.bad.load(.acquire)) return error.Dsv41EmbedRead;
    }

    const Job = struct {
        t: *const Ssd,
        ids: []const u32,
        only: ?[]const u32,
        raw: Raw,
        bad: std.atomic.Value(bool) = .init(false),

        fn runVoid(self: *const Job, first: usize, stride: usize) void {
            _ = self.run(first, stride);
        }

        /// Every `stride`-th row from `first`, asked of the kernel first so a batch costs about one SSD latency.
        fn run(self: *const Job, first: usize, stride: usize) bool {
            const t = self.t;
            const n = if (self.only) |o| o.len else self.ids.len;
            if (comptime builtin.os.tag.isDarwin()) {
                var j = first;
                while (j < n) : (j += stride) {
                    const r: u64 = self.ids[if (self.only) |o| o[j] else j];
                    if (r < t.rows) for (0..3) |p| engram.rdadvise(t.fd, t.off[p] + r * t.len[p], t.len[p]);
                }
            }
            var j = first;
            while (j < n) : (j += stride) {
                const i = if (self.only) |o| o[j] else j;
                const r: u64 = self.ids[i];
                if (r >= t.rows) return self.fail();
                for (0..3) |p| {
                    const l = t.len[p];
                    expert_io.readExact(t.fd, self.raw.part(p)[i * l ..][0..l], t.off[p] + r * l) catch return self.fail();
                }
            }
            return true;
        }

        fn fail(self: *const Job) bool {
            @constCast(&self.bad).store(true, .release);
            return false;
        }
    };
};

/// The rows of the next forward, read on a thread while this one runs; the forward that asks for the same ids takes
/// them, any other reads its own.
pub const Ahead = struct {
    gpa: std.mem.Allocator,
    ssd: *const Ssd,
    ids: []u32,
    raw: ?Ssd.Raw = null,
    thread: ?std.Thread = null,

    /// The returned value must not move (the thread holds it).
    pub fn start(gpa: std.mem.Allocator, ssd: *const Ssd, ids: []const u32) !*Ahead {
        const self = try gpa.create(Ahead);
        errdefer gpa.destroy(self);
        const own = try gpa.dupe(u32, ids);
        errdefer gpa.free(own);
        self.* = .{ .gpa = gpa, .ssd = ssd, .ids = own };
        self.thread = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    fn run(self: *Ahead) void {
        self.raw = self.ssd.read(self.gpa, self.ids) catch null;
    }

    /// The rows of `ids` when they are the ones this read was started for and it succeeded (the caller frees them).
    pub fn take(self: *Ahead, ids: []const u32) ?Ssd.Raw {
        if (self.thread) |t| t.join();
        self.thread = null;
        if (!std.mem.eql(u32, self.ids, ids)) return null;
        const raw = self.raw;
        self.raw = null;
        return raw;
    }

    pub fn finish(self: *Ahead) void {
        if (self.thread) |t| t.join();
        if (self.raw) |r| r.deinit(self.gpa);
        self.gpa.free(self.ids);
        self.gpa.destroy(self);
    }
};

const testing = std.testing;

pub const test_dim = 64;
pub const test_rows = 40;
pub const test_groups = 2;
pub const test_words = 8;

/// A shard holding a 4-bit g32 triple of `test_rows` rows over `test_dim` columns, every byte a function of its row and
/// column, laid out as the pack's globals shard (biases first).
pub fn writeTriple(io: std.Io, dir: std.Io.Dir, sdtype: []const u8, esize: usize) !void {
    const wb = test_rows * test_words * 4;
    const sb = test_rows * test_groups * esize;
    var hdr_buf: [512]u8 = undefined;
    const hdr = try std.fmt.bufPrint(&hdr_buf, "{{\"embed.biases\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"embed.scales\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}," ++
        "\"embed.weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{
        sdtype, test_rows, test_groups, sb, sdtype, test_rows, test_groups, sb, 2 * sb, test_rows, test_words, 2 * sb, 2 * sb + wb,
    });
    const file = try testing.allocator.alloc(u8, 8 + hdr.len + 2 * sb + wb);
    defer testing.allocator.free(file);
    std.mem.writeInt(u64, file[0..8], hdr.len, .little);
    @memcpy(file[8..][0..hdr.len], hdr);
    const data = file[8 + hdr.len ..];
    for (data, 0..) |*byte, i| byte.* = @truncate(i *% 2654435761 >> 7);
    // A scale or bias is a finite number of about 1 (the top byte of a little-endian float).
    for (data[0 .. 2 * sb], 0..) |*byte, i| if (i % esize == esize - 1) {
        byte.* = 0x30 | (byte.* & 0x0f);
    };
    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = file });
}

fn tmpRoot(io: std.Io, td: *testing.TmpDir, buf: []u8) ![]const u8 {
    return buf[0..try td.dir.realPath(io, buf)];
}

test "dsv41 embed: rows read from the shard are the stored bytes, repeats and the first and last rows included" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeTriple(io, td.dir, "BF16", 2);
    var buf: [512]u8 = undefined;
    var ssd = try Ssd.open(testing.allocator, try tmpRoot(io, &td, &buf), test_dim);
    defer ssd.close();
    try testing.expectEqual(@as(c_int, 4), ssd.bits);
    try testing.expectEqual(@as(c_int, 32), ssd.group);
    try testing.expectEqual(@as(u64, test_rows * (32 + 4 + 4)), ssd.stored);

    const file = try td.dir.readFileAlloc(io, "model-00001-of-00001.safetensors", testing.allocator, .limited(1 << 20));
    defer testing.allocator.free(file);
    // Every row twice and in a scrambled order, so the threaded split and the repeat copies both run.
    var ids: [2 * test_rows + 3]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 17 + 3) % test_rows);
    ids[0] = 0;
    ids[1] = test_rows - 1;
    ids[2] = 0;
    for ([_][]const u32{ ids[0..1], ids[0..3], &ids }) |set| {
        const raw = try ssd.read(testing.allocator, set);
        defer raw.deinit(testing.allocator);
        for (set, 0..) |id, i| for (0..3) |p| {
            const l = ssd.len[p];
            const want = file[ssd.off[p] + id * l ..][0..l];
            try testing.expectEqualSlices(u8, want, raw.part(p)[i * l ..][0..l]);
        };
    }
    try testing.expectError(error.Dsv41EmbedRead, ssd.read(testing.allocator, &.{ 1, test_rows }));
}

test "dsv41 embed: a triple that does not fit the model width, or is missing, is refused by name" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeTriple(io, td.dir, "BF16", 2);
    var buf: [512]u8 = undefined;
    const root = try tmpRoot(io, &td, &buf);
    try testing.expectError(error.Dsv41WeightLayout, Ssd.open(testing.allocator, root, 48));
    try testing.expectError(error.Dsv41WeightLayout, Ssd.open(testing.allocator, root, 7));
    var other = testing.tmpDir(.{});
    defer other.cleanup();
    try other.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    var buf2: [512]u8 = undefined;
    try testing.expectError(error.Dsv41WeightLayout, Ssd.open(testing.allocator, try tmpRoot(io, &other, &buf2), test_dim));
}

test "dsv41 embed: rows read ahead equal a cold read, and only for the ids they were started for" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeTriple(io, td.dir, "F16", 2);
    var buf: [512]u8 = undefined;
    var ssd = try Ssd.open(testing.allocator, try tmpRoot(io, &td, &buf), test_dim);
    defer ssd.close();
    const ids = [_]u32{ 5, 9, 5, 39, 0 };
    const cold = try ssd.read(testing.allocator, &ids);
    defer cold.deinit(testing.allocator);
    const ahead = try Ahead.start(testing.allocator, &ssd, &ids);
    defer ahead.finish();
    try testing.expect(ahead.take(&.{ 5, 9, 5, 39, 1 }) == null);
    const ahead2 = try Ahead.start(testing.allocator, &ssd, &ids);
    defer ahead2.finish();
    const warm = ahead2.take(&ids).?;
    defer warm.deinit(testing.allocator);
    for (0..3) |p| try testing.expectEqualSlices(u8, cold.part(p), warm.part(p));
    try testing.expect(ahead2.take(&ids) == null);
}
