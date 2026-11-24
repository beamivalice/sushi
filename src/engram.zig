//! Engram: the n-gram hash of the reference `engram.py` (DeepSeek's, which Hy4-Flash shares), the on-disk tables it
//! indexes, and their read-ahead. A table (tens of GB a module) is read from the checkpoint's shards per lookup and
//! never loaded. An arch gives the geometry and the tables' names; the tables are fp8 rows with one e8m0 scale per 32
//! columns. Ported from mlx-serve's `dsv41_engram.zig` (MIT).
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const expert_io = @import("expert_io.zig");

/// The most Engram modules a model carries, the longest n-gram and the most hash heads an n-gram size takes.
pub const max_modules = 4;
pub const max_ngram = 8;
pub const max_heads = 16;

/// A model's Engram modules: the layer each sits at, the id its hash multipliers are drawn with, its table's rows, and
/// the hash's shape.
pub const Geometry = struct {
    n: u32 = 0,
    layers: [max_modules]u32 = @splat(0),
    /// The multipliers of module i are numpy `default_rng(seed + 10007 * hash_ids[i])` draws.
    hash_ids: [max_modules]u32 = @splat(0),
    seed: u32 = 0,
    rows: [max_modules]u64 = @splat(0),
    max_ngram: u32 = 0,
    vocab_size: u64 = 0,
    heads: u32 = 0,
    head_dim: u32 = 0,
    pad_id: u32 = 0,
    compressed_vocab: u32 = 0,

    pub fn cols(self: Geometry) u32 {
        return (self.max_ngram - 1) * self.heads;
    }

    pub fn layerAt(self: Geometry, layer: usize) ?usize {
        for (self.layers[0..self.n], 0..) |l, i| if (l == layer) return i;
        return null;
    }
};

pub const MAX_COLS = (max_ngram - 1) * max_heads;

/// numpy's `default_rng(seed)`: SeedSequence entropy mixing feeding PCG64 (XSL-RR), and `integers()`'s
/// Lemire draw. The hash multipliers come from it; any other generator rehashes the whole table.
pub const NumpyRng = struct {
    state: u128,
    inc: u128,

    const MULT: u128 = (@as(u128, 2549297995355413924) << 64) | 4865540595714422341;

    pub fn init(seed: u32) NumpyRng {
        const INIT_A: u32 = 0x43b0d7e5;
        const MULT_A: u32 = 0x931e8875;
        const INIT_B: u32 = 0x8b51f9dd;
        const MULT_B: u32 = 0x58f38ded;
        const hashmix = struct {
            fn f(value: u32, hash_const: *u32) u32 {
                var v = value ^ hash_const.*;
                hash_const.* *%= MULT_A;
                v *%= hash_const.*;
                return v ^ (v >> 16);
            }
        }.f;
        const mix = struct {
            fn f(x: u32, y: u32) u32 {
                const r = 0xca01f9dd *% x -% 0x4973f715 *% y;
                return r ^ (r >> 16);
            }
        }.f;
        var pool: [4]u32 = undefined;
        var hc = INIT_A;
        for (&pool, 0..) |*p, i| p.* = hashmix(if (i == 0) seed else 0, &hc);
        for (0..4) |src| for (0..4) |dst| {
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], &hc));
        };
        var words: [8]u32 = undefined;
        var hb = INIT_B;
        for (&words, 0..) |*w, i| {
            var v = pool[i % 4] ^ hb;
            hb *%= MULT_B;
            v *%= hb;
            w.* = v ^ (v >> 16);
        }
        var u: [4]u64 = undefined;
        for (&u, 0..) |*x, k| x.* = @as(u64, words[2 * k]) | (@as(u64, words[2 * k + 1]) << 32);
        var r: NumpyRng = .{ .state = 0, .inc = ((@as(u128, u[2]) << 64 | u[3]) << 1) | 1 };
        r.step();
        r.state +%= @as(u128, u[0]) << 64 | u[1];
        r.step();
        return r;
    }

    fn step(self: *NumpyRng) void {
        self.state = self.state *% MULT +% self.inc;
    }

    fn next64(self: *NumpyRng) u64 {
        self.step();
        const x: u64 = @truncate((self.state >> 64) ^ self.state);
        return std.math.rotr(u64, x, @as(u32, @truncate(self.state >> 122)));
    }

    /// A draw in [0, n): numpy's 64-bit Lemire arm, the one `integers` takes for a range wider than 32 bits.
    pub fn below(self: *NumpyRng, n: u64) u64 {
        std.debug.assert(n > 1 << 32);
        var m: u128 = @as(u128, self.next64()) * n;
        var leftover: u64 = @truncate(m);
        if (leftover < n) {
            const threshold = (std.math.maxInt(u64) - (n - 1)) % n;
            while (leftover < threshold) {
                m = @as(u128, self.next64()) * n;
                leftover = @truncate(m);
            }
        }
        return @intCast(m >> 64);
    }
};

fn isPrime(n: u64) bool {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    var i: u64 = 3;
    while (i * i <= n) : (i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/// The per-layer hash constants: one odd multiplier per lookback, one prime bucket per
/// (n-gram size, head), and each bucket's first row.
pub const Hash = struct {
    /// The compressed id of a token that takes no part in an n-gram: every n-gram reaching back to or
    /// past it hashes as all-pad, as the release's `engram_hash` blocks them.
    pub const DEAD: u32 = std.math.maxInt(u32);

    n_layers: u32,
    max_ngram: u32,
    heads: u32,
    pad: u32,
    mult: [max_modules][max_ngram]u64 = @splat(@splat(0)),
    primes: [max_modules][MAX_COLS]u64 = @splat(@splat(0)),
    offsets: [max_modules][MAX_COLS]u64 = @splat(@splat(0)),

    /// `pad` is the compressed id of the config's pad token. Refuses a config whose prime buckets do
    /// not add up to its declared table rows.
    pub fn init(e: Geometry, pad: u32) !Hash {
        var h: Hash = .{ .n_layers = e.n, .max_ngram = e.max_ngram, .heads = e.heads, .pad = pad };
        if (e.n > max_modules or e.max_ngram < 2 or e.max_ngram > max_ngram or e.heads == 0 or e.heads > max_heads) return error.EngramGeometry;
        const bound = (@as(u64, std.math.maxInt(i64)) / e.compressed_vocab) / 2;
        if (bound <= 1 << 32) return error.EngramGeometry;
        var p = e.vocab_size - 1;
        for (0..e.n) |l| {
            var rng = NumpyRng.init(e.seed + 10007 * e.hash_ids[l]);
            for (0..h.max_ngram) |k| h.mult[l][k] = rng.below(bound) * 2 + 1;
            var off: u64 = 0;
            for (0..h.cols()) |c| {
                p += 1;
                while (!isPrime(p)) p += 1;
                h.primes[l][c] = p;
                h.offsets[l][c] = off;
                off += p;
            }
            if (off != e.rows[l]) return error.EngramTableRows;
        }
        return h;
    }

    pub fn cols(self: *const Hash) u32 {
        return (self.max_ngram - 1) * self.heads;
    }

    /// Row ids of the n-grams ending at each of `ids`, which follow `hist` (the request's earlier
    /// compressed ids). `out` is `[ids.len][n_layers][cols]`. Look-back stops at the sequence start,
    /// where the slots read `pad`.
    pub fn rows(self: *const Hash, hist: []const u32, ids: []const u32, out: []u32) void {
        const nc = self.cols();
        std.debug.assert(out.len == ids.len * self.n_layers * nc);
        for (ids, 0..) |_, i| {
            const pos = hist.len + i;
            var toks: [max_ngram]u64 = undefined;
            var blocked = false;
            for (0..self.max_ngram) |s| {
                const t: u32 = if (pos < s) self.pad else if (pos - s >= hist.len) ids[pos - s - hist.len] else hist[pos - s];
                blocked = blocked or t == DEAD;
                toks[s] = if (blocked) self.pad else t;
            }
            for (0..self.n_layers) |l| {
                const dst = out[(i * self.n_layers + l) * nc ..][0..nc];
                var rolling = toks[0] *% self.mult[l][0];
                for (1..self.max_ngram) |k| {
                    rolling ^= toks[k] *% self.mult[l][k];
                    for (0..self.heads) |hd| {
                        const c = (k - 1) * self.heads + hd;
                        dst[c] = @intCast(rolling % self.primes[l][c] + self.offsets[l][c]);
                    }
                }
            }
        }
    }
};

/// One module's table, read per lookup from the shard that holds it: e4m3 code bytes `[rows, head_dim]`
/// and one e8m0 scale byte per 32 columns `[rows, head_dim / 32]`.
pub const Table = struct {
    fd: c_int,
    rows: u64,
    head_dim: u32,
    w_off: u64,
    s_off: u64,
    /// Scale bytes a row; 0 for a BF16 table, which stores its rows as values.
    sdim: u32,
    /// Bytes a row of `w`: `head_dim` FP8 codes, or twice that as BF16.
    w_row: u32,
    /// Hot rows held in RAM; null reads every row from the shard.
    cache: ?*RowCache = null,
    /// A counter of the rows `pread` from the shard, for a test to point at its own cell.
    rows_read: ?*std.atomic.Value(u64) = null,
    /// A counter of the `F_RDADVISE` calls, for a test to point at its own cell.
    advised: ?*std.atomic.Value(u64) = null,

    pub fn close(self: *Table) void {
        _ = std.c.close(self.fd);
    }

    /// `dtype` is the position of the stored dtype among the ones `locate` was offered.
    pub const TensorAt = struct { file: []const u8, start: u64, shape: [2]u64, dtype: usize = 0 };

    /// Where `name` sits: the shard the index names for it, or the first shard whose header lists it. Its stored dtype
    /// must be one of `dtypes`.
    pub fn locate(a: std.mem.Allocator, dir_path: []const u8, name: []const u8, dtypes: []const []const u8) !TensorAt {
        const io = std.Io.Threaded.global_single_threaded.io();
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var named: ?[]const u8 = null;
        if (dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(256 << 20))) |raw| {
            const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return error.EngramTableMissing;
            if (idx == .object) if (idx.object.get("weight_map")) |wm| if (wm == .object) if (wm.object.get(name)) |f| {
                if (f != .string) return error.EngramTableMissing;
                named = f.string;
            };
        } else |_| {}
        if (named) |file| return try inShard(a, dir_path, file, name, dtypes) orelse error.EngramTableMissing;
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            if (try inShard(a, dir_path, try a.dupe(u8, entry.name), name, dtypes)) |t| return t;
        }
        return error.EngramTableMissing;
    }

    fn inShard(a: std.mem.Allocator, dir_path: []const u8, file: []const u8, name: []const u8, dtypes: []const []const u8) !?TensorAt {
        if (std.mem.indexOfAny(u8, file, "/\\") != null) return error.EngramTableMissing;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, file }, 0);
        const fd = try expert_io.openHinted(path, .{ .nocache = false, .readahead_off = false });
        defer _ = std.c.close(fd);
        var header = try expert_io.SafetensorsHeader.read(a, fd);
        defer header.deinit();
        const meta = header.parsed.value.object.get(name) orelse return null;
        if (meta != .object) return error.EngramTableHeader;
        const dt = meta.object.get("dtype") orelse return error.EngramTableHeader;
        const shape = meta.object.get("shape") orelse return error.EngramTableHeader;
        const offs = meta.object.get("data_offsets") orelse return error.EngramTableHeader;
        if (dt != .string or shape != .array or shape.array.items.len != 2 or offs != .array or offs.array.items.len != 2)
            return error.EngramTableHeader;
        const r = shape.array.items[0];
        const c = shape.array.items[1];
        const lo = offs.array.items[0];
        if (r != .integer or c != .integer or lo != .integer or r.integer < 0 or c.integer < 0 or lo.integer < 0) return error.EngramTableHeader;
        const which = for (dtypes, 0..) |d, i| {
            if (std.mem.eql(u8, d, dt.string)) break i;
        } else return error.EngramTableHeader;
        return .{ .file = file, .start = header.data_start + @as(u64, @intCast(lo.integer)), .shape = .{ @intCast(r.integer), @intCast(c.integer) }, .dtype = which };
    }

    /// The checkpoint's names of one table's code and scale tensors.
    pub const Names = struct { weight: []const u8, scale: []const u8 };

    /// The table `names` in the checkpoint at `dir_path`.
    pub fn open(gpa: std.mem.Allocator, dir_path: []const u8, names: Names, rows: u64, head_dim: u32) !Table {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try locate(a, dir_path, names.weight, &.{"F8_E4M3"});
        const s = try locate(a, dir_path, names.scale, &.{"F8_E8M0"});
        if (!std.mem.eql(u8, w.file, s.file)) return error.EngramTableSplit;
        if (w.shape[0] != rows or s.shape[0] != rows) return error.EngramTableSize;
        if (w.shape[1] != head_dim or s.shape[1] * 32 != head_dim) return error.EngramTableGeometry;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, w.file }, 0);
        // Rows are read at random: read-ahead would pull their neighbours in.
        const fd = try expert_io.openHinted(path, .{ .nocache = false, .readahead_off = true });
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .w_off = w.start, .s_off = s.start, .sdim = head_dim / 32, .w_row = head_dim };
    }

    /// The BF16 table `name` (`[rows, head_dim]`, no scales) in the checkpoint at `dir_path`.
    pub fn openBf16(gpa: std.mem.Allocator, dir_path: []const u8, name: []const u8, rows: u64, head_dim: u32) !Table {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const w = try locate(a, dir_path, name, &.{"BF16"});
        if (w.shape[0] != rows) return error.EngramTableSize;
        if (w.shape[1] != head_dim) return error.EngramTableGeometry;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, w.file }, 0);
        const fd = try expert_io.openHinted(path, .{ .nocache = false, .readahead_off = true });
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .w_off = w.start, .s_off = 0, .sdim = 0, .w_row = 2 * head_dim };
    }

    /// Rows `ids` as they sit on disk (host only, any thread).
    pub const Raw = struct {
        w: []align(4) u8,
        s: []align(4) u8,

        pub fn deinit(self: Raw, gpa: std.mem.Allocator) void {
            gpa.free(self.w);
            gpa.free(self.s);
        }
    };

    /// Each distinct id is read once, from the cache or the shard, and copied to every position that holds it.
    pub fn read(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.w_row);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.sdim);
        errdefer gpa.free(sc);
        const first = try gpa.alloc(u32, n);
        defer gpa.free(first);
        // Out of memory for the map reads every position, as a batch without repeats does.
        const repeats = firstOccurrences(gpa, ids, first) catch 0;
        const cache = self.cache orelse {
            if (repeats == 0) {
                try self.readInto(ids, null, w, sc);
                return .{ .w = w, .s = sc };
            }
            const only = try gpa.alloc(u32, n - repeats);
            defer gpa.free(only);
            var k: usize = 0;
            for (first, 0..) |f, i| if (f == i) {
                only[k] = @intCast(i);
                k += 1;
            };
            try self.readInto(ids, only, w, sc);
            self.copyRepeats(first, w, sc);
            return .{ .w = w, .s = sc };
        };
        const missed = try gpa.alloc(u32, n);
        defer gpa.free(missed);
        var k = cache.lookup(ids, w, sc, missed);
        if (repeats > 0) {
            var kept: usize = 0;
            for (missed[0..k]) |i| if (first[i] == i) {
                missed[kept] = i;
                kept += 1;
            };
            k = kept;
        }
        try self.readInto(ids, missed[0..k], w, sc);
        cache.insert(ids, missed[0..k], w, sc);
        if (repeats > 0) self.copyRepeats(first, w, sc);
        return .{ .w = w, .s = sc };
    }

    fn copyRepeats(self: *const Table, first: []const u32, w: []u8, sc: []u8) void {
        const wl = self.w_row;
        const sl = self.sdim;
        for (first, 0..) |f, i| if (f != i) {
            @memcpy(w[i * wl ..][0..wl], w[f * wl ..][0..wl]);
            @memcpy(sc[i * sl ..][0..sl], sc[f * sl ..][0..sl]);
        };
    }

    /// Reads `ids[i]` into row i of `w` and `sc` for every i of `only` (all of them when null).
    fn readInto(self: *const Table, ids: []const u32, only: ?[]const u32, w: []u8, sc: []u8) !void {
        const n = if (only) |o| o.len else ids.len;
        if (self.rows_read) |c| _ = c.fetchAdd(n, .monotonic);
        var job: Job = .{ .t = self, .ids = ids, .only = only, .w = w, .s = sc };
        if (n == 0) return;
        if (n >= WIDE_READ_ROWS and builtin.os.tag.isDarwin()) return job.readWide(n, expert_io.processDiskBytesRead);
        return job.readRange(0, n, readParts(n));
    }

    /// The first `n` rows of `raw` dequantized: f32 `[n, head_dim]` (the caller frees).
    pub fn dequant(self: *const Table, raw: Raw, n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const rows: c_int = @intCast(n);
        if (self.sdim == 0) {
            const va = try mlx.newData(raw.w.ptr, &[_]c_int{ rows, @intCast(self.head_dim) }, .bfloat16);
            defer _ = mlx.mlx_array_free(va);
            var wide = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(wide);
            try mlx.check(mlx.mlx_astype(&wide, va, .float32, s));
            return wide;
        }
        const wa = try mlx.newData(raw.w.ptr, &[_]c_int{ rows, @intCast(self.head_dim / 4) }, .uint32);
        defer _ = mlx.mlx_array_free(wa);
        const sa = try mlx.newData(raw.s.ptr, &[_]c_int{ rows, @intCast(self.sdim) }, .uint8);
        defer _ = mlx.mlx_array_free(sa);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_dequantize(&out, wa, sa, .{ .ctx = null }, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        return out;
    }

    /// Rows `ids`, dequantized: f32 `[ids.len, head_dim]` (the caller frees).
    pub fn gather(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
        const raw = try self.read(gpa, ids);
        defer raw.deinit(gpa);
        return self.dequant(raw, ids.len, s);
    }

    /// Asks the kernel to read every `stride`-th of `ids` from `first` into the page cache now
    /// (`F_RDADVISE`: asynchronous, no copy), so the preads behind it find the rows cached or in
    /// flight and a batch of random rows costs about one SSD latency, not one per row. An
    /// out-of-range id is left for `read` to refuse. A batch wide enough to keep every thread busy is not advised:
    /// its preads queue deep enough, and the advise calls cost several times the CPU of the preads.
    fn advise(self: *const Table, ids: []const u32, only: ?[]const u32, first: usize, stride: usize) void {
        if (comptime !builtin.os.tag.isDarwin()) return;
        const n = if (only) |o| o.len else ids.len;
        if (n >= ADVISE_BELOW) return;
        var j = first;
        while (j < n) : (j += stride) {
            const r: u64 = ids[if (only) |o| o[j] else j];
            if (r >= self.rows) continue;
            if (self.advised) |c| _ = c.fetchAdd(2, .monotonic);
            rdadvise(self.fd, self.w_off + r * self.w_row, self.w_row);
            if (self.sdim > 0) rdadvise(self.fd, self.s_off + r * self.sdim, self.sdim);
        }
    }

    const Job = struct {
        t: *const Table,
        ids: []const u32,
        only: ?[]const u32 = null,
        w: []u8,
        s: []u8,
        bad: std.atomic.Value(bool) = .init(false),

        /// Page-cached rows read fastest on a few threads (64 preads spend ~25x the CPU for the same rows); a batch that
        /// reaches the SSD needs the queue depth, so the first sign of a physical read widens the rest to `readParts`.
        fn readWide(self: *Job, n: usize, comptime diskBytes: fn () u64) !void {
            const cached_readers = 4;
            var start: usize = 0;
            var end: usize = 256;
            var previous = diskBytes();
            while (true) {
                try self.readRange(start, end, @min(end - start, cached_readers));
                if (end == n) return;
                const current = diskBytes();
                if (current > previous) return self.readRange(end, n, readParts(n - end));
                previous = current;
                start = end;
                end = @min(n, start + WIDE_READ_ROWS);
            }
        }

        fn readRange(self: *Job, start: usize, end: usize, parts: usize) !void {
            if (comptime builtin.is_test) peak_readers_for_test = @max(peak_readers_for_test, parts);
            if (parts == 1) {
                if (!self.run(start, 1, end)) return error.EngramRead;
                return;
            }
            var threads: [PAR_THREADS]?std.Thread = @splat(null);
            for (threads[1..parts], 1..) |*t, k| t.* = std.Thread.spawn(.{}, runVoid, .{ self, start + k, parts, end }) catch null;
            var failed = !self.run(start, parts, end);
            for (threads[1..parts], 1..) |t, k| {
                if (t) |th| th.join() else failed = !self.run(start + k, parts, end) or failed;
            }
            if (failed or self.bad.load(.acquire)) return error.EngramRead;
        }

        fn runVoid(self: *Job, first: usize, stride: usize, end: usize) void {
            _ = self.run(first, stride, end);
        }

        /// Every `stride`-th row from `first`, asked of the kernel first.
        fn run(self: *Job, first: usize, stride: usize, end: usize) bool {
            const t = self.t;
            const wl = t.w_row;
            const sl = t.sdim;
            t.advise(self.ids, self.only, first, stride);
            var j = first;
            while (j < end) : (j += stride) {
                const i = if (self.only) |o| o[j] else j;
                const r: u64 = self.ids[i];
                if (r >= t.rows or
                    !preadAll(t.fd, self.w[i * wl ..][0..wl], t.w_off + r * wl) or
                    (sl > 0 and !preadAll(t.fd, self.s[i * sl ..][0..sl], t.s_off + r * sl)))
                {
                    self.bad.store(true, .release);
                    return false;
                }
            }
            return true;
        }
    };
};

/// Sets `first[i]` to the earliest position holding `ids[i]` and returns how many positions repeat an earlier one.
pub fn firstOccurrences(gpa: std.mem.Allocator, ids: []const u32, first: []u32) !usize {
    var seen: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer seen.deinit(gpa);
    try seen.ensureTotalCapacity(gpa, @intCast(ids.len));
    var repeats: usize = 0;
    for (ids, first, 0..) |id, *f, i| {
        const slot = seen.getOrPutAssumeCapacity(id);
        if (slot.found_existing) repeats += 1 else slot.value_ptr.* = @intCast(i);
        f.* = slot.value_ptr.*;
    }
    return repeats;
}

/// Rows each table's cache holds. On 1M tokens of documents in turn the hit rate is 31% at
/// 64k rows, 42% at 1M and 47% at 4M: past 1M a row buys little.
pub const cache_rows: u64 = 1 << 20;

/// Host bytes the row caches of `e`'s tables hold, which a load bills beside the trunk.
pub fn cacheBytes(e: Geometry) u64 {
    var total: u64 = 0;
    for (e.rows[0..e.n]) |rows| total += RowCache.bytesFor(@min(cache_rows, rows), e.head_dim, e.head_dim / 32);
    return total;
}

/// Engram rows held in RAM by row id: 8-way set-associative, least recent out of a set. A frequent n-gram's row is
/// read from the shard once, not as a fresh SSD page each token. Shared by a table's readers (the forward and the
/// read-ahead thread). Row ids are already hashes, so a set is the id's residue.
pub const RowCache = struct {
    pub const WAYS = 8;
    const EMPTY = std.math.maxInt(u32);

    gpa: std.mem.Allocator,
    mu: std.Io.Mutex = .init,
    keys: []u32,
    ages: []u32,
    w: []u8,
    s: []u8,
    wl: u32,
    sl: u32,
    tick: u32 = 0,
    hits: u64 = 0,
    misses: u64 = 0,

    /// Host bytes a cache of `rows` rows holds.
    pub fn bytesFor(rows: u64, wl: u32, sl: u32) u64 {
        return rows / WAYS * WAYS * (wl + sl + 8);
    }

    pub fn init(gpa: std.mem.Allocator, rows: u64, wl: u32, sl: u32) !RowCache {
        const n: usize = @intCast(rows / WAYS * WAYS);
        if (n == 0) return error.EngramRowCacheEmpty;
        var self: RowCache = .{ .gpa = gpa, .keys = &.{}, .ages = &.{}, .w = &.{}, .s = &.{}, .wl = wl, .sl = sl };
        errdefer self.deinit();
        self.keys = try gpa.alloc(u32, n);
        @memset(self.keys, EMPTY);
        self.ages = try gpa.alloc(u32, n);
        @memset(self.ages, 0);
        self.w = try gpa.alloc(u8, n * wl);
        self.s = try gpa.alloc(u8, n * sl);
        return self;
    }

    pub fn create(gpa: std.mem.Allocator, rows: u64, wl: u32, sl: u32) !*RowCache {
        const self = try gpa.create(RowCache);
        errdefer gpa.destroy(self);
        self.* = try init(gpa, rows, wl, sl);
        return self;
    }

    pub fn destroy(self: *RowCache) void {
        const gpa = self.gpa;
        self.deinit();
        gpa.destroy(self);
    }

    pub fn deinit(self: *RowCache) void {
        self.gpa.free(self.keys);
        self.gpa.free(self.ages);
        self.gpa.free(self.w);
        self.gpa.free(self.s);
    }

    fn set(self: *const RowCache, id: u32) usize {
        return (id % (self.keys.len / WAYS)) * WAYS;
    }

    fn find(self: *const RowCache, id: u32) ?usize {
        const base = self.set(id);
        for (self.keys[base..][0..WAYS], base..) |k, slot| if (k == id) return slot;
        return null;
    }

    fn bump(self: *RowCache, slot: usize) void {
        self.tick +%= 1;
        self.ages[slot] = self.tick;
    }

    /// Copies each cached row of `ids` into its row of `w` and `s`; the positions of the others go to `missed`, whose
    /// count it returns.
    pub fn lookup(self: *RowCache, ids: []const u32, w: []u8, s: []u8, missed: []u32) usize {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        var k: usize = 0;
        for (ids, 0..) |id, i| {
            const slot = self.find(id) orelse {
                missed[k] = @intCast(i);
                k += 1;
                continue;
            };
            self.bump(slot);
            @memcpy(w[i * self.wl ..][0..self.wl], self.w[slot * self.wl ..][0..self.wl]);
            @memcpy(s[i * self.sl ..][0..self.sl], self.s[slot * self.sl ..][0..self.sl]);
        }
        self.hits += ids.len - k;
        self.misses += k;
        return k;
    }

    /// Keeps the rows just read (positions `missed` of `ids`), each in place of its set's least recent row.
    pub fn insert(self: *RowCache, ids: []const u32, missed: []const u32, w: []const u8, s: []const u8) void {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        for (missed) |i| {
            const id = ids[i];
            if (self.find(id) != null) continue;
            const base = self.set(id);
            var slot = base;
            // Ages wrap with the tick: the oldest is the one furthest behind it.
            for (base..base + WAYS) |j| {
                if (self.keys[j] == EMPTY) {
                    slot = j;
                    break;
                }
                if (self.tick -% self.ages[j] > self.tick -% self.ages[slot]) slot = j;
            }
            self.keys[slot] = id;
            self.bump(slot);
            @memcpy(self.w[slot * self.wl ..][0..self.wl], w[i * self.wl ..][0..self.wl]);
            @memcpy(self.s[slot * self.sl ..][0..self.sl], s[i * self.sl ..][0..self.sl]);
        }
    }
};

/// Engram rows read ahead on a thread of its own: per grid (a run of `len` tokens from position `at`),
/// layer by layer and span by span, the order the forward asks for them, so the SSD reads hide behind
/// the GPU work of the spans, layers and chunks before.
pub const Prefetch = struct {
    gpa: std.mem.Allocator,
    tables: []const *const Table,
    grids: []Grid,
    span: usize,
    nc: usize,
    stop: std.atomic.Value(bool) = .init(false),
    /// Set once the thread has read every grid (or was stopped).
    ended: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    /// `ids` is `[len][tables.len][nc]` row ids (`Hash.rows`).
    pub const Spec = struct { at: usize, len: usize, ids: []const u32 };

    const Grid = struct {
        at: usize,
        len: usize,
        ids: []u32,
        items: []Item,
    };
    const Item = struct {
        raw: ?Table.Raw = null,
        state: std.atomic.Value(u8) = .init(pending),
    };
    const pending = 0;
    const done = 1;
    const failed = 2;

    /// Reads `specs` in order. Copies what it is given; the returned value must not move (the thread holds it).
    pub fn start(gpa: std.mem.Allocator, tables: []const *const Table, specs: []const Spec, nc: usize, span: usize) !*Prefetch {
        const self = try gpa.create(Prefetch);
        errdefer gpa.destroy(self);
        const own_tables = try gpa.dupe(*const Table, tables);
        errdefer gpa.free(own_tables);
        const grids = try gpa.alloc(Grid, specs.len);
        var made: usize = 0;
        errdefer {
            for (grids[0..made]) |g| freeGrid(gpa, g);
            gpa.free(grids);
        }
        for (specs, grids) |sp, *g| {
            const ids = try gpa.dupe(u32, sp.ids);
            errdefer gpa.free(ids);
            const items = try gpa.alloc(Item, tables.len * ((sp.len + span - 1) / span));
            @memset(items, .{});
            g.* = .{ .at = sp.at, .len = sp.len, .ids = ids, .items = items };
            made += 1;
        }
        self.* = .{ .gpa = gpa, .tables = own_tables, .grids = grids, .span = span, .nc = nc };
        self.thread = try std.Thread.spawn(.{}, run, .{self});
        return self;
    }

    fn freeGrid(gpa: std.mem.Allocator, g: Grid) void {
        for (g.items) |*item| if (item.raw) |r| r.deinit(gpa);
        gpa.free(g.items);
        gpa.free(g.ids);
    }

    fn run(self: *Prefetch) void {
        defer self.ended.store(true, .release);
        for (self.grids) |*g| {
            const n_spans = g.items.len / self.tables.len;
            for (self.tables, 0..) |table, layer| for (0..n_spans) |k| {
                if (self.stop.load(.acquire)) return;
                const item = &g.items[layer * n_spans + k];
                item.raw = self.readSpan(g, table, layer, k) catch null;
                item.state.store(if (item.raw != null) done else failed, .release);
            };
        }
    }

    fn readSpan(self: *Prefetch, g: *const Grid, table: *const Table, layer: usize, k: usize) !Table.Raw {
        const lo = k * self.span;
        const hi = @min(g.len, lo + self.span);
        const mine = try self.gpa.alloc(u32, (hi - lo) * self.nc);
        defer self.gpa.free(mine);
        for (lo..hi) |t| @memcpy(mine[(t - lo) * self.nc ..][0..self.nc], g.ids[(t * self.tables.len + layer) * self.nc ..][0..self.nc]);
        return table.read(self.gpa, mine);
    }

    /// Does a grid start at `at` and cover `len` tokens?
    pub fn covers(self: *const Prefetch, at: usize, len: usize) bool {
        for (self.grids) |g| if (g.at == at and g.len >= len) return true;
        return false;
    }

    /// Does a grid start at `at` with `ids` (row ids, `Spec.ids`' layout) as its first tokens' rows? Row ids decide what
    /// the reads hold, so a grid that matches serves exactly the rows a fresh read of `ids` would.
    pub fn matches(self: *const Prefetch, at: usize, ids: []const u32) bool {
        for (self.grids) |g| if (g.at == at and g.ids.len >= ids.len and std.mem.eql(u32, g.ids[0..ids.len], ids)) return true;
        return false;
    }

    /// Has the thread read everything it was given?
    pub fn finished(self: *const Prefetch) bool {
        return self.ended.load(.acquire);
    }

    /// The rows of `layer`'s span `k` of the grid at `at`, once read (the caller frees them).
    pub fn take(self: *Prefetch, at: usize, layer: usize, k: usize) !Table.Raw {
        for (self.grids) |*g| if (g.at == at) {
            const item = &g.items[layer * (g.items.len / self.tables.len) + k];
            while (item.state.load(.acquire) == pending) std.Thread.yield() catch {};
            const raw = item.raw orelse return error.EngramRead;
            item.raw = null;
            return raw;
        };
        return error.EngramRead;
    }

    /// Stops the thread, frees what was never taken and the prefetch itself.
    pub fn finish(self: *Prefetch) void {
        self.stop.store(true, .release);
        if (self.thread) |t| t.join();
        for (self.grids) |g| freeGrid(self.gpa, g);
        self.gpa.free(self.grids);
        self.gpa.free(self.tables);
        self.gpa.destroy(self);
    }
};

const WIDE_READ_ROWS = 4096;
const PAR_ROWS_PER_THREAD = 16;
const PAR_THREADS = 64;
threadlocal var peak_readers_for_test: usize = 0;
/// Rows of one read from which it is not advised: its threads' preads already queue deep enough.
const ADVISE_BELOW = 256;

/// Threads a read of `n` rows takes, this one included: one per 16 rows (a verify's rows wait at the first Engram
/// layer), up to the queue depth past which the SSD answers random rows no sooner.
fn readParts(n: usize) usize {
    return std.math.clamp(n / PAR_ROWS_PER_THREAD, 1, PAR_THREADS);
}

fn preadAll(fd: c_int, dst: []u8, off: u64) bool {
    expert_io.readExact(fd, dst, off) catch return false;
    return true;
}

/// macOS `struct radvisory`.
const Radvisory = extern struct { ra_offset: i64, ra_count: c_int };

pub fn rdadvise(fd: c_int, off: u64, len: usize) void {
    var ra: Radvisory = .{ .ra_offset = @intCast(off), .ra_count = @intCast(len) };
    _ = std.c.fcntl(fd, std.c.F.RDADVISE, &ra);
}

const testing = std.testing;

const layer1: Table.Names = .{ .weight = "layers.1.engram.embed.weight", .scale = "layers.1.engram.embed.scale" };
const layer3: Table.Names = .{ .weight = "layers.3.engram.embed.weight", .scale = "layers.3.engram.embed.scale" };

/// A shard holding layer 1's table: 4 rows of head_dim 64 as raw e4m3 codes (1.0) with row r's first
/// scale 2^r and its second 0.5.
fn writeFp8Fixture(io: std.Io, dir: std.Io.Dir, indexed: bool) !void {
    const hdr = "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4,64],\"data_offsets\":[0,256]}," ++
        "\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[4,2],\"data_offsets\":[256,264]}}";
    var shard: [8 + hdr.len + 264]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    const data = shard[8 + hdr.len ..];
    @memset(data[0..256], 0x38); // e4m3 1.0
    for (0..4) |r| {
        data[256 + 2 * r] = @intCast(127 + r); // 2^r for the first 32 columns
        data[256 + 2 * r + 1] = 126; // 0.5 for the rest
    }
    try dir.writeFile(io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = &shard });
    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
    if (indexed) try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{" ++
        "\"layers.1.engram.embed.weight\":\"model-00002-of-00002.safetensors\"," ++
        "\"layers.1.engram.embed.scale\":\"model-00002-of-00002.safetensors\"}}" });
}

test "engram: the table is found by the index, or by the headers when there is none, and dequantizes as mxfp8" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]bool{ true, false }) |indexed| {
        var td = testing.tmpDir(.{});
        defer td.cleanup();
        try writeFp8Fixture(io, td.dir, indexed);
        var buf: [512]u8 = undefined;
        var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, 4, 64);
        defer t.close();
        const rows = try t.gather(testing.allocator, &.{ 3, 0 }, s);
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_array_eval(rows));
        const v = mlx.mlx_array_data_float32(rows).?;
        try testing.expectEqual(@as(f32, 8), v[0]);
        try testing.expectEqual(@as(f32, 0.5), v[63]);
        try testing.expectEqual(@as(f32, 1), v[64]);
        try testing.expectEqual(@as(f32, 0.5), v[64 + 32]);
    }
}

test "engram: a table of the wrong size or geometry is refused by name" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    try testing.expectError(error.EngramTableSize, Table.open(testing.allocator, dir, layer1, 5, 64));
    try testing.expectError(error.EngramTableGeometry, Table.open(testing.allocator, dir, layer1, 4, 128));
    try testing.expectError(error.EngramTableMissing, Table.open(testing.allocator, dir, layer3, 4, 64));
}

test "engram: an id past the table is refused by a plain read and by one split over threads" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, 4, 64);
    defer t.close();
    t.advise(&.{ 3, 7 }, null, 0, 1);
    try testing.expectError(error.EngramRead, t.read(testing.allocator, &.{ 3, 1, 7 }));
    var ids: [144]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 7 + i / 5) % 4);
    const raw = try t.read(testing.allocator, &ids);
    defer raw.deinit(testing.allocator);
    for (ids, 0..) |id, i| {
        try testing.expectEqual(@as(u8, @intCast(127 + id)), raw.s[2 * i]);
        try testing.expectEqual(@as(u8, 126), raw.s[2 * i + 1]);
        try testing.expect(std.mem.allEqual(u8, raw.w[64 * i ..][0..64], 0x38));
    }
    ids[100] = 7;
    try testing.expectError(error.EngramRead, t.read(testing.allocator, &ids));
}

/// Layer 1's table of `rows` rows of head_dim 64, every byte a function of its row and column.
fn patternByte(plane: u8, row: u64, col: u64) u8 {
    return @truncate(row *% 131 +% col *% 7 +% plane *% 91 +% (row >> 8));
}

fn writePatternShard(io: std.Io, dir: std.Io.Dir, rows: usize) !void {
    var hdr_buf: [512]u8 = undefined;
    const hdr = try std.fmt.bufPrint(&hdr_buf, "{{\"layers.1.engram.embed.weight\":{{\"dtype\":\"F8_E4M3\",\"shape\":[{d},64],\"data_offsets\":[0,{d}]}}," ++
        "\"layers.1.engram.embed.scale\":{{\"dtype\":\"F8_E8M0\",\"shape\":[{d},2],\"data_offsets\":[{d},{d}]}}}}", .{ rows, rows * 64, rows, rows * 64, rows * 66 });
    const shard = try testing.allocator.alloc(u8, 8 + hdr.len + rows * 66);
    defer testing.allocator.free(shard);
    std.mem.writeInt(u64, shard[0..8], hdr.len, .little);
    @memcpy(shard[8..][0..hdr.len], hdr);
    const data = shard[8 + hdr.len ..];
    for (0..rows) |r| {
        for (0..64) |c| data[r * 64 + c] = patternByte(0, r, c);
        for (0..2) |c| data[rows * 64 + r * 2 + c] = patternByte(1, r, c);
    }
    try dir.writeFile(io, .{ .sub_path = "model-00002-of-00002.safetensors", .data = shard });
    try dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = "\x02\x00\x00\x00\x00\x00\x00\x00{}" });
}

fn expectPatternRows(raw: Table.Raw, ids: []const u32) !void {
    for (ids, 0..) |id, i| {
        for (0..64) |c| try testing.expectEqual(patternByte(0, id, c), raw.w[i * 64 + c]);
        for (0..2) |c| try testing.expectEqual(patternByte(1, id, c), raw.s[i * 2 + c]);
    }
}

test "engram: rows read together equal rows read one at a time, in any order, with repeats and at both ends" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    const rows = 1000;
    try writePatternShard(io, td.dir, rows);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, rows, 64);
    defer t.close();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    var ids: [900]u32 = undefined;
    for (&ids) |*id| id.* = rnd.uintLessThan(u32, rows);
    ids[0] = rows - 1;
    ids[1] = 0;
    ids[2] = rows - 1;
    ids[899] = 0;
    // A run of neighbours, in order and reversed, among the random ones.
    for (300..340) |i| ids[i] = @intCast(500 + (i - 300));
    for (340..380) |i| ids[i] = @intCast(539 - (i - 340));
    for ([_][]const u32{ ids[0..1], ids[0..20], ids[290..390], &ids, &.{ 0, 1, 2, 3, 999, 998, 997 } }) |part| {
        const together = try t.read(testing.allocator, part);
        defer together.deinit(testing.allocator);
        try expectPatternRows(together, part);
        for (part, 0..) |id, i| {
            const one = try t.read(testing.allocator, &.{id});
            defer one.deinit(testing.allocator);
            try testing.expectEqualSlices(u8, one.w, together.w[i * 64 ..][0..64]);
            try testing.expectEqualSlices(u8, one.s, together.s[i * 2 ..][0..2]);
        }
    }
    var bad = ids;
    for ([_]u32{ rows, rows + 1, std.math.maxInt(u32) }) |id| {
        bad[450] = id;
        try testing.expectError(error.EngramRead, t.read(testing.allocator, &bad));
        try testing.expectError(error.EngramRead, t.read(testing.allocator, &.{ 5, id }));
    }
}

test "engram: a read takes one thread per 16 rows, up to 64" {
    for ([_][2]usize{ .{ 0, 1 }, .{ 24, 1 }, .{ 32, 2 }, .{ 192, 12 }, .{ 512, 32 }, .{ 1024, 64 }, .{ 49152, 64 } }) |c|
        try testing.expectEqual(c[1], readParts(c[0]));
}

test "engram: a batch of 256 rows or more is read without advise calls, a narrower one with two a row" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    const rows = 1000;
    try writePatternShard(io, td.dir, rows);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, rows, 64);
    defer t.close();
    var advised: std.atomic.Value(u64) = .init(0);
    t.advised = &advised;
    var ids: [ADVISE_BELOW + 40]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 389) % rows);
    for ([_]struct { n: usize, advises: u64 }{
        .{ .n = 24, .advises = 48 },
        .{ .n = ADVISE_BELOW - 1, .advises = 2 * (ADVISE_BELOW - 1) },
        .{ .n = ADVISE_BELOW, .advises = 0 },
        .{ .n = ids.len, .advises = 0 },
    }) |c| {
        const before = advised.load(.monotonic);
        const raw = try t.read(testing.allocator, ids[0..c.n]);
        defer raw.deinit(testing.allocator);
        try testing.expectEqual(c.advises, advised.load(.monotonic) - before);
        try expectPatternRows(raw, ids[0..c.n]);
    }
}

test "engram: a row cache serves repeats from RAM with the bytes a pread gives, through evictions" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var plain = try Table.open(testing.allocator, dir, layer1, 4, 64);
    defer plain.close();
    var cached = try Table.open(testing.allocator, dir, layer1, 4, 64);
    defer cached.close();
    var cache = try RowCache.init(testing.allocator, RowCache.WAYS, 64, 2);
    defer cache.deinit();
    cached.cache = &cache;
    for ([_][]const u32{ &.{ 3, 0, 3 }, &.{3}, &.{ 0, 1, 2 }, &.{ 2, 1, 0, 3, 3 } }) |ids| {
        const want = try plain.read(testing.allocator, ids);
        defer want.deinit(testing.allocator);
        const got = try cached.read(testing.allocator, ids);
        defer got.deinit(testing.allocator);
        try testing.expectEqualSlices(u8, want.w, got.w);
        try testing.expectEqualSlices(u8, want.s, got.s);
    }
    // {3, 0, 3}: row 3 is read once; {3}: a hit.
    try testing.expect(cache.hits >= 2);
    try testing.expectEqual(@as(u64, 12), cache.hits + cache.misses);
    try testing.expectError(error.EngramRead, cached.read(testing.allocator, &.{ 1, 7 }));
}

test "engram: a full row-cache set gives up its least recent row" {
    var c = try RowCache.init(testing.allocator, RowCache.WAYS, 1, 1);
    defer c.deinit();
    var w: [1]u8 = undefined;
    var sc: [1]u8 = undefined;
    var missed: [1]u32 = undefined;
    for (0..RowCache.WAYS + 1) |id| {
        const ids = [_]u32{@intCast(id)};
        w[0] = @intCast(id);
        sc[0] = @intCast(id);
        try testing.expectEqual(@as(usize, 1), c.lookup(&ids, &w, &sc, &missed));
        c.insert(&ids, &missed, &w, &sc);
        // Row 0 stays the most recent, so row 1 is the one the ninth row replaces.
        if (id > 0) try testing.expectEqual(@as(usize, 0), c.lookup(&.{0}, &w, &sc, &missed));
    }
    try testing.expectEqual(@as(usize, 1), c.lookup(&.{1}, &w, &sc, &missed));
    try testing.expectEqual(@as(usize, 0), c.lookup(&.{8}, &w, &sc, &missed));
    try testing.expectEqual(@as(u8, 8), w[0]);
}

test "engram: a prefetch hands each span's rows as a gather of the same ids would" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, 4, 64);
    defer t.close();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // 5 tokens, 2 layers, 2 columns: ids [(token * 2 + layer) * 2 + column]; spans of 2 leave a ragged last one.
    var ids: [5 * 2 * 2]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 3 + 1) % 4);
    const tables = [_]*const Table{ &t, &t };
    var pf = try Prefetch.start(testing.allocator, &tables, &.{ .{ .at = 0, .len = 5, .ids = &ids }, .{ .at = 5, .len = 2, .ids = ids[0..8] } }, 2, 2);
    defer pf.finish();
    try testing.expect(pf.covers(5, 2) and pf.covers(0, 3) and !pf.covers(0, 6) and !pf.covers(4, 1));
    for (0..2) |layer| for (0..3) |k| {
        const lo = k * 2;
        const hi = @min(5, lo + 2);
        var mine: [4]u32 = undefined;
        for (lo..hi) |tok| @memcpy(mine[(tok - lo) * 2 ..][0..2], ids[(tok * 2 + layer) * 2 ..][0..2]);
        const want = try t.gather(testing.allocator, mine[0 .. (hi - lo) * 2], s);
        defer _ = mlx.mlx_array_free(want);
        const raw = try pf.take(0, layer, k);
        defer raw.deinit(testing.allocator);
        const got = try t.dequant(raw, (hi - lo) * 2, s);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(got));
        const n = (hi - lo) * 2 * 64;
        try testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(want).?[0..n], mlx.mlx_array_data_float32(got).?[0..n]);
    };
}

test "engram: an id repeated within a batch is read from the shard once and fills every place it sits" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var t = try Table.open(testing.allocator, dir, layer1, 4, 64);
    defer t.close();
    var rows_read: std.atomic.Value(u64) = .init(0);
    t.rows_read = &rows_read;
    var wide: [144]u32 = undefined;
    for (&wide, 0..) |*id, i| id.* = @intCast((i * 7 + i / 5) % 4);
    for ([_][]const u32{ &.{ 3, 1, 3, 3, 0, 1, 2, 3 }, &.{ 2, 2 }, &.{1}, &wide }) |ids| {
        const before = rows_read.load(.monotonic);
        const got = try t.read(testing.allocator, ids);
        defer got.deinit(testing.allocator);
        var distinct: u64 = 0;
        for (0..4) |r| distinct += @intFromBool(std.mem.indexOfScalar(u32, ids, @intCast(r)) != null);
        try testing.expectEqual(distinct, rows_read.load(.monotonic) - before);
        for (ids, 0..) |id, i| {
            try testing.expectEqual(@as(u8, @intCast(127 + id)), got.s[2 * i]);
            try testing.expectEqual(@as(u8, 126), got.s[2 * i + 1]);
            try testing.expect(std.mem.allEqual(u8, got.w[64 * i ..][0..64], 0x38));
        }
    }
    try testing.expectError(error.EngramRead, t.read(testing.allocator, &.{ 7, 7 }));
}

test "engram: with a row cache a repeated missed id is read once, and a cached one not at all" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var t = try Table.open(testing.allocator, dir, layer1, 4, 64);
    defer t.close();
    var cache = try RowCache.init(testing.allocator, 4 * RowCache.WAYS, 64, 2);
    defer cache.deinit();
    t.cache = &cache;
    var rows_read: std.atomic.Value(u64) = .init(0);
    t.rows_read = &rows_read;
    for ([_]struct { ids: []const u32, reads: u64 }{
        .{ .ids = &.{ 2, 1, 2, 2, 1 }, .reads = 2 },
        .{ .ids = &.{ 1, 2, 3, 3 }, .reads = 1 },
        .{ .ids = &.{ 3, 2, 1, 2 }, .reads = 0 },
    }) |c| {
        const before = rows_read.load(.monotonic);
        const got = try t.read(testing.allocator, c.ids);
        defer got.deinit(testing.allocator);
        try testing.expectEqual(c.reads, rows_read.load(.monotonic) - before);
        for (c.ids, 0..) |id, i| try testing.expectEqual(@as(u8, @intCast(127 + id)), got.s[2 * i]);
    }
}

test "engram: a bad id in a helper thread's rows fails the read" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, 4, 64);
    defer t.close();
    // 64 rows make four parts; part k takes the rows j with j % 4 == k, so row 17 is a helper's.
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i % 4);
    var w: [64 * 64]u8 = undefined;
    var sc: [64 * 2]u8 = undefined;
    try t.readInto(&ids, null, &w, &sc);
    ids[17] = 7;
    try testing.expectError(error.EngramRead, t.readInto(&ids, null, &w, &sc));
}

test "engram: a prefetch is the next chunk's only when its grid starts there and holds the same row ids" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, 4, 64);
    defer t.close();
    // 5 tokens, 2 tables, 2 columns.
    var ids: [5 * 2 * 2]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 3 + 1) % 4);
    const tables = [_]*const Table{ &t, &t };
    var pf = try Prefetch.start(testing.allocator, &tables, &.{.{ .at = 7, .len = 5, .ids = &ids }}, 2, 5);
    defer pf.finish();
    try testing.expect(pf.matches(7, &ids) and pf.matches(7, ids[0..8]) and pf.matches(7, &.{}));
    try testing.expect(!pf.matches(6, &ids) and !pf.matches(8, ids[0..8]));
    var longer: [5 * 2 * 2 + 4]u32 = undefined;
    @memcpy(longer[0..ids.len], &ids);
    @memset(longer[ids.len..], 0);
    try testing.expect(!pf.matches(7, &longer));
    var other = ids;
    other[9] ^= 1;
    try testing.expect(!pf.matches(7, &other) and pf.matches(7, other[0..8]));
    for (0..2) |layer| (try pf.take(7, layer, 0)).deinit(testing.allocator);
    while (!pf.finished()) std.Thread.yield() catch {};
}

test "engram: a module drawn with its own hash id and a seed, as Hy4-Flash's four are" {
    // Hy4-Flash: modules at layers 1, 11, 21, 31 hash with the 1-based ids 2, 12, 22, 32; its config lists the
    // 16 primes of each, which are the derived ones. Multipliers: numpy over an identity map (120818 ids).
    var g: Geometry = .{ .n = 4, .layers = .{ 1, 11, 21, 31 }, .hash_ids = .{ 2, 12, 22, 32 }, .rows = .{ 164994380, 164999008, 165004086, 165008942 }, .max_ngram = 3, .vocab_size = 10312000, .heads = 8, .head_dim = 96, .pad_id = 120002, .compressed_vocab = 120818 };
    const h = try Hash.init(g, 120002);
    const want = [4][3]u64{
        .{ 62749575316035, 30426722171175, 18574837598557 },
        .{ 14916611332703, 34290019069965, 64992498366675 },
        .{ 55942601167937, 16505523750331, 42928537650479 },
        .{ 22330137526737, 12836963598171, 17229190754683 },
    };
    for (want, 0..) |row, l| try testing.expectEqualSlices(u64, &row, h.mult[l][0..3]);
    try testing.expectEqualSlices(u64, &.{ 10312007, 10312021, 10312039 }, h.primes[0][0..3]);
    try testing.expectEqualSlices(u64, &.{ 10313179, 10313189, 10313201 }, h.primes[3][13..16]);
    g.seed = 5;
    try testing.expectEqualSlices(u64, &.{ 8224917011737, 28807692472155, 5317873185359 }, (try Hash.init(g, 120002)).mult[0][0..3]);
}

test "engram reads a page-cached wide shard without fanning out to the SSD queue depth" {
    if (comptime !builtin.os.tag.isDarwin()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    const rows = 8193;
    try writePatternShard(io, td.dir, rows);
    var buf: [512]u8 = undefined;
    var table = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], layer1, rows, 64);
    defer table.close();
    var ids: [rows]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(rows - 1 - i);
    const primed = try table.read(testing.allocator, &ids);
    primed.deinit(testing.allocator);
    peak_readers_for_test = 0;
    const raw = try table.read(testing.allocator, &ids);
    defer raw.deinit(testing.allocator);
    try expectPatternRows(raw, &ids);
    try testing.expectEqual(@as(usize, 4), peak_readers_for_test);

    const MixedPages = struct {
        threadlocal var samples: usize = 0;

        fn bytes() u64 {
            samples += 1;
            return if (samples >= 3) 16384 else 0;
        }
    };
    @memset(raw.w, 0);
    @memset(raw.s, 0);
    MixedPages.samples = 0;
    var job: Table.Job = .{ .t = &table, .ids = &ids, .w = raw.w, .s = raw.s };
    try job.readWide(ids.len, MixedPages.bytes);
    try testing.expectEqual(@as(usize, 64), peak_readers_for_test);
    try testing.expectEqual(@as(usize, 3), MixedPages.samples);
    try expectPatternRows(raw, &ids);
}
