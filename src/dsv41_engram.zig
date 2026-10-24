//! DeepSeek-V4.1 Engram: the n-gram hash of DeepSeek's `inference/engram.py`, the on-disk tables it
//! indexes, and their read-ahead. The tables (about 100 GB a layer, fp8 rows with one e8m0 scale per
//! 32 columns) are read from the checkpoint's shards per lookup and never loaded.
//! Ported from mlx-serve's `dsv41_engram.zig` (MIT), reading the release's own layout only.
const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx.zig");
const cfg_mod = @import("dsv41_config.zig");
const expert_io = @import("expert_io.zig");

pub const MAX_COLS = (cfg_mod.max_ngram - 1) * cfg_mod.max_heads;

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
    mult: [cfg_mod.max_engram][cfg_mod.max_ngram]u64 = @splat(@splat(0)),
    primes: [cfg_mod.max_engram][MAX_COLS]u64 = @splat(@splat(0)),
    offsets: [cfg_mod.max_engram][MAX_COLS]u64 = @splat(@splat(0)),

    /// `pad` is the compressed id of the config's pad token. Refuses a config whose prime buckets do
    /// not add up to its declared table rows.
    pub fn init(e: cfg_mod.Engram, pad: u32) !Hash {
        var h: Hash = .{ .n_layers = e.n, .max_ngram = e.max_ngram, .heads = e.heads, .pad = pad };
        if (e.n > cfg_mod.max_engram or e.max_ngram < 2 or e.max_ngram > cfg_mod.max_ngram or e.heads == 0 or e.heads > cfg_mod.max_heads) return error.EngramGeometry;
        const bound = (@as(u64, std.math.maxInt(i64)) / e.compressed_vocab) / 2;
        if (bound <= 1 << 32) return error.EngramGeometry;
        var p = e.vocab_size - 1;
        for (0..e.n) |l| {
            var rng = NumpyRng.init(10007 * e.layers[l]);
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
            var toks: [cfg_mod.max_ngram]u64 = undefined;
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

/// V4.1's token map (`tests/dump_dsv41_token_map.py`): one bit per token id, set where the token opens
/// the next compressed id, then every other token's id as a u32.
const v41_token_map = @embedFile("fixtures/dsv41_engram_token_map.bin");

fn unpackTokenMap(map: []u32, raw: []const u8) !void {
    const bits = (map.len + 7) / 8;
    if (raw.len < bits) return error.EngramTokenMap;
    var next: u32 = 0;
    var at = bits;
    for (map, 0..) |*v, t| {
        if ((raw[t / 8] >> @intCast(t % 8)) & 1 != 0) {
            v.* = next;
            next += 1;
        } else {
            if (at + 4 > raw.len) return error.EngramTokenMap;
            v.* = std.mem.readInt(u32, raw[at..][0..4], .little);
            at += 4;
        }
    }
    if (at != raw.len) return error.EngramTokenMap;
}

/// Token id to compressed id: a pack's own `engram_token_map.json` (a list) when it ships one, else the
/// map DeepSeek's `engram.py` derives from V4.1's tokenizer. Every hash multiplier derives from the
/// compressed vocabulary's size, so a map built for another tokenizer is refused.
pub fn loadTokenMap(gpa: std.mem.Allocator, dir_path: []const u8, vocab: u32, compressed: u32) ![]u32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const map = try gpa.alloc(u32, vocab);
    errdefer gpa.free(map);
    var own = false;
    if (dir.readFileAlloc(io, "engram_token_map.json", gpa, .limited(64 << 20))) |raw| {
        defer gpa.free(raw);
        const parsed = std.json.parseFromSlice([]const u32, gpa, raw, .{}) catch return error.EngramTokenMap;
        defer parsed.deinit();
        if (parsed.value.len != vocab) return error.EngramTokenMap;
        @memcpy(map, parsed.value);
        own = true;
    } else |_| {
        unpackTokenMap(map, v41_token_map) catch return error.EngramTokenMapMissing;
    }
    var hi: u32 = 0;
    for (map) |v| hi = @max(hi, v);
    if (hi + 1 != compressed) return if (own) error.EngramTokenMap else error.EngramTokenMapMissing;
    return map;
}

/// One layer's table, read per lookup from the shard that holds it: e4m3 code bytes `[rows, head_dim]`
/// and one e8m0 scale byte per 32 columns `[rows, head_dim / 32]`.
pub const Table = struct {
    fd: c_int,
    rows: u64,
    head_dim: u32,
    w_off: u64,
    s_off: u64,
    sdim: u32,
    /// Hot rows held in RAM; null reads every row from the shard.
    cache: ?*RowCache = null,

    pub fn close(self: *Table) void {
        _ = std.c.close(self.fd);
    }

    pub const TensorAt = struct { file: []const u8, start: u64, shape: [2]u64 };

    /// Where `name` sits: the shard the index names for it, or the first shard whose header lists it.
    fn locate(a: std.mem.Allocator, dir_path: []const u8, name: []const u8, dtype: []const u8) !TensorAt {
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
        if (named) |file| return try inShard(a, dir_path, file, name, dtype) orelse error.EngramTableMissing;
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            if (try inShard(a, dir_path, try a.dupe(u8, entry.name), name, dtype)) |t| return t;
        }
        return error.EngramTableMissing;
    }

    fn inShard(a: std.mem.Allocator, dir_path: []const u8, file: []const u8, name: []const u8, dtype: []const u8) !?TensorAt {
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
        if (dt != .string or !std.mem.eql(u8, dt.string, dtype) or shape != .array or shape.array.items.len != 2 or offs != .array or offs.array.items.len != 2)
            return error.EngramTableHeader;
        const r = shape.array.items[0];
        const c = shape.array.items[1];
        const lo = offs.array.items[0];
        if (r != .integer or c != .integer or lo != .integer or r.integer < 0 or c.integer < 0 or lo.integer < 0) return error.EngramTableHeader;
        return .{ .file = file, .start = header.data_start + @as(u64, @intCast(lo.integer)), .shape = .{ @intCast(r.integer), @intCast(c.integer) } };
    }

    /// Layer `layer`'s table in the checkpoint at `dir_path`.
    pub fn open(gpa: std.mem.Allocator, dir_path: []const u8, layer: usize, rows: u64, head_dim: u32) !Table {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var name_buf: [96]u8 = undefined;
        const wname = try std.fmt.bufPrint(&name_buf, "layers.{d}.engram.embed.weight", .{layer});
        const w = try locate(a, dir_path, wname, "F8_E4M3");
        const sname = try std.fmt.bufPrint(&name_buf, "layers.{d}.engram.embed.scale", .{layer});
        const s = try locate(a, dir_path, sname, "F8_E8M0");
        if (!std.mem.eql(u8, w.file, s.file)) return error.EngramTableSplit;
        if (w.shape[0] != rows or s.shape[0] != rows) return error.EngramTableSize;
        if (w.shape[1] != head_dim or s.shape[1] * 32 != head_dim) return error.EngramTableGeometry;
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, w.file }, 0);
        // Rows are read at random: read-ahead would pull their neighbours in.
        const fd = try expert_io.openHinted(path, .{ .nocache = false, .readahead_off = true });
        return .{ .fd = fd, .rows = rows, .head_dim = head_dim, .w_off = w.start, .s_off = s.start, .sdim = head_dim / 32 };
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

    pub fn read(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.head_dim);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.sdim);
        errdefer gpa.free(sc);
        const cache = self.cache orelse {
            try self.readInto(ids, null, w, sc);
            return .{ .w = w, .s = sc };
        };
        const missed = try gpa.alloc(u32, n);
        defer gpa.free(missed);
        const k = cache.lookup(ids, w, sc, missed);
        try self.readInto(ids, missed[0..k], w, sc);
        cache.insert(ids, missed[0..k], w, sc);
        return .{ .w = w, .s = sc };
    }

    /// Reads `ids[i]` into row i of `w` and `sc` for every i of `only` (all of them when null).
    fn readInto(self: *const Table, ids: []const u32, only: ?[]const u32, w: []u8, sc: []u8) !void {
        const n = if (only) |o| o.len else ids.len;
        const job: Job = .{ .t = self, .ids = ids, .only = only, .w = w, .s = sc };
        // One thread per 16 rows (a verify's rows wait at the first Engram layer), this one included.
        const parts = std.math.clamp(n / PAR_ROWS_PER_THREAD, 1, PAR_THREADS);
        if (parts > 1) {
            var threads: [PAR_THREADS]?std.Thread = @splat(null);
            for (threads[1..parts], 1..) |*t, k| t.* = std.Thread.spawn(.{}, Job.runVoid, .{ &job, k, parts }) catch null;
            var failed = !job.run(0, parts);
            for (threads[1..parts], 1..) |t, k| {
                if (t) |th| th.join() else failed = !job.run(k, parts) or failed;
            }
            if (failed or job.bad.load(.acquire)) return error.EngramRead;
        } else if (n > 0 and !job.run(0, 1)) return error.EngramRead;
    }

    /// The first `n` rows of `raw` dequantized: f32 `[n, head_dim]` (the caller frees).
    pub fn dequant(self: *const Table, raw: Raw, n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const rows: c_int = @intCast(n);
        const wa = mlx.mlx_array_new_data(raw.w.ptr, &[_]c_int{ rows, @intCast(self.head_dim / 4) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(wa);
        const sa = mlx.mlx_array_new_data(raw.s.ptr, &[_]c_int{ rows, @intCast(self.sdim) }, 2, .uint8);
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
    /// out-of-range id is left for `read` to refuse.
    fn advise(self: *const Table, ids: []const u32, only: ?[]const u32, first: usize, stride: usize) void {
        if (comptime !builtin.os.tag.isDarwin()) return;
        const n = if (only) |o| o.len else ids.len;
        var j = first;
        while (j < n) : (j += stride) {
            const r: u64 = ids[if (only) |o| o[j] else j];
            if (r >= self.rows) continue;
            rdadvise(self.fd, self.w_off + r * self.head_dim, self.head_dim);
            rdadvise(self.fd, self.s_off + r * self.sdim, self.sdim);
        }
    }

    const Job = struct {
        t: *const Table,
        ids: []const u32,
        only: ?[]const u32 = null,
        w: []u8,
        s: []u8,
        bad: std.atomic.Value(bool) = .init(false),

        fn runVoid(self: *const Job, first: usize, stride: usize) void {
            _ = self.run(first, stride);
        }

        /// Every `stride`-th row from `first`, asked of the kernel first.
        fn run(self: *const Job, first: usize, stride: usize) bool {
            const t = self.t;
            const wl = t.head_dim;
            const sl = t.sdim;
            t.advise(self.ids, self.only, first, stride);
            const n = if (self.only) |o| o.len else self.ids.len;
            var j = first;
            while (j < n) : (j += stride) {
                const i = if (self.only) |o| o[j] else j;
                const r: u64 = self.ids[i];
                if (r >= t.rows or
                    !preadAll(t.fd, self.w[i * wl ..][0..wl], t.w_off + r * wl) or
                    !preadAll(t.fd, self.s[i * sl ..][0..sl], t.s_off + r * sl))
                {
                    @constCast(&self.bad).store(true, .release);
                    return false;
                }
            }
            return true;
        }
    };
};

/// Rows each table's cache holds. On 1M tokens of documents in turn the hit rate is 31% at
/// 64k rows, 42% at 1M and 47% at 4M: past 1M a row buys little.
pub const cache_rows: u64 = 1 << 20;

/// Host bytes the row caches of `e`'s tables hold, which a load bills beside the trunk.
pub fn cacheBytes(e: cfg_mod.Engram) u64 {
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

const PAR_ROWS_PER_THREAD = 16;
const PAR_THREADS = 16;

fn preadAll(fd: c_int, dst: []u8, off: u64) bool {
    expert_io.readExact(fd, dst, off) catch return false;
    return true;
}

/// macOS `struct radvisory`.
const Radvisory = extern struct { ra_offset: i64, ra_count: c_int };

fn rdadvise(fd: c_int, off: u64, len: usize) void {
    var ra: Radvisory = .{ .ra_offset = @intCast(off), .ra_count = @intCast(len) };
    _ = std.c.fcntl(fd, std.c.F.RDADVISE, &ra);
}

const testing = std.testing;

fn releaseEngram() cfg_mod.Engram {
    return .{ .n = 2, .layers = .{ 1, 14, 0, 0 }, .rows = .{ 384006168, 384016682, 0, 0 }, .max_ngram = 4, .vocab_size = 16000000, .heads = 8, .head_dim = 256, .pad_id = 2, .compressed_vocab = 99092 };
}

test "dsv41 engram: hash multipliers are numpy default_rng(10007 * layer) draws" {
    // The integers DeepSeek's engram.py derives for the release.
    const h = try Hash.init(releaseEngram(), 2);
    const want = [2][4]u64{
        .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 },
        .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 },
    };
    for (want, 0..) |row, l| try testing.expectEqualSlices(u64, &row, h.mult[l][0..4]);
}

test "dsv41 engram: prime buckets continue across layers and fill the declared tables" {
    const h = try Hash.init(releaseEngram(), 2);
    try testing.expectEqualSlices(u64, &.{ 16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183 }, h.primes[0][0..8]);
    try testing.expectEqualSlices(u64, &.{ 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889 }, h.primes[1][16..24]);
    try testing.expectEqualSlices(u64, &.{ 0, 16000477, 32000964, 48001463, 64001970 }, h.offsets[1][0..5]);
    var bad = releaseEngram();
    bad.rows[1] += 1;
    try testing.expectError(error.EngramTableRows, Hash.init(bad, 2));
}

test "dsv41 engram: row ids match DeepSeek's NgramHashState across prefill and decode" {
    // inference/engram.py over compressed ids [3, 99091, 2, 77, 77, 1024], then one decode step [5]
    // at position 6 (identity token map, pad = 2).
    const h = try Hash.init(releaseEngram(), 2);
    const prompt = [_]u32{ 3, 99091, 2, 77, 77, 1024 };
    var pre: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &prompt, &pre);
    try testing.expectEqualSlices(u32, &.{ 7010988, 29651738, 47710115, 48176674, 72877570, 81111450, 101228567, 120693201, 143450283, 146576661, 168605081, 191902092, 196072724, 214158605, 233287660, 246388310, 261549600, 286523983, 293659617, 313227845, 320939872, 340508126, 357771274, 372619255 }, pre[0..24]);
    try testing.expectEqualSlices(u32, &.{ 11344931, 28156158, 45201617, 61942444, 78326219, 85383927, 106445668, 125242217, 129707946, 158452422, 167536917, 177080926, 202551179, 221735906, 238090062, 254314270, 259798390, 277434364, 301936998, 307898224, 335252679, 339837629, 352557585, 368186488 }, pre[(5 * 2 + 1) * 24 ..][0..24]);
    var dec: [2 * 24]u32 = undefined;
    h.rows(&prompt, &.{5}, &dec);
    try testing.expectEqualSlices(u32, &.{ 14359625, 17064925, 34766379, 48382691, 68824895, 91643820, 111054239, 121732398, 132594636, 146910113, 174982973, 181797503, 206306954, 210573457, 224988242, 243323351, 267856496, 279543941, 298892988, 316569280, 333687525, 351365567, 362014820, 382511691 }, dec[0..24]);
    // A chunk boundary anywhere hashes the same rows.
    var split: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, prompt[0..2], split[0 .. 2 * 48]);
    h.rows(prompt[0..2], prompt[2..], split[2 * 48 ..]);
    try testing.expectEqualSlices(u32, &pre, &split);
}

test "dsv41 engram: a dead token makes every n-gram across it the all-pad one" {
    const h = try Hash.init(releaseEngram(), 2);
    const ids = [_]u32{ 3, Hash.DEAD, 77 };
    var got: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &ids, &got);
    var want: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &.{2}, want[48..96]); // the dead token's own row: every slot pad
    try testing.expectEqualSlices(u32, want[48..96], got[48..96]);
    h.rows(&.{ 2, 2 }, &.{77}, want[96..]); // 77 with only pad behind it
    try testing.expectEqualSlices(u32, want[96..], got[96..]);
    h.rows(&.{}, &.{3}, want[0..48]); // no look-ahead
    try testing.expectEqualSlices(u32, want[0..48], got[0..48]);
}

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

test "dsv41 engram: the table is found by the index, or by the headers when there is none, and dequantizes as mxfp8" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]bool{ true, false }) |indexed| {
        var td = testing.tmpDir(.{});
        defer td.cleanup();
        try writeFp8Fixture(io, td.dir, indexed);
        var buf: [512]u8 = undefined;
        var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 1, 4, 64);
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

test "dsv41 engram: a table of the wrong size or geometry is refused by name" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    try testing.expectError(error.EngramTableSize, Table.open(testing.allocator, dir, 1, 5, 64));
    try testing.expectError(error.EngramTableGeometry, Table.open(testing.allocator, dir, 1, 4, 128));
    try testing.expectError(error.EngramTableMissing, Table.open(testing.allocator, dir, 3, 4, 64));
}

test "dsv41 engram: an id past the table is refused by a plain read and by one split over threads" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 1, 4, 64);
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

test "dsv41 engram: a row cache serves repeats from RAM with the bytes a pread gives, through evictions" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    var plain = try Table.open(testing.allocator, dir, 1, 4, 64);
    defer plain.close();
    var cached = try Table.open(testing.allocator, dir, 1, 4, 64);
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

test "dsv41 engram: a full row-cache set gives up its least recent row" {
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

test "dsv41 engram: a prefetch hands each span's rows as a gather of the same ids would" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try writeFp8Fixture(io, td.dir, true);
    var buf: [512]u8 = undefined;
    var t = try Table.open(testing.allocator, buf[0..try td.dir.realPath(io, &buf)], 1, 4, 64);
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

test "dsv41 engram: the embedded V4.1 token map collapses case variants and ends where the config says" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const map = try loadTokenMap(testing.allocator, dir, 129280, 99092);
    defer testing.allocator.free(map);
    // the, The, " the", " The", THE, " THE" normalize alike (engram.py's ids for the release tokenizer).
    for ([_]u32{ 1805, 671, 270, 455, 20852, 6367 }) |id| try testing.expectEqual(@as(u32, 237), map[id]);
    try testing.expectEqual(@as(u32, 2), map[2]);
    try testing.expectEqual(@as(u32, 99091), map[129279]);
    // Another tokenizer's sizes: the pack must ship its own map.
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 129280, 99000));
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 151936, 99092));
}

test "dsv41 engram: a pack's own token map is read as a list and held to the config's size" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(io, .{ .sub_path = "engram_token_map.json", .data = "[0, 1, 1, 2]" });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const map = try loadTokenMap(testing.allocator, dir, 4, 3);
    defer testing.allocator.free(map);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2 }, map);
    try testing.expectError(error.EngramTokenMap, loadTokenMap(testing.allocator, dir, 4, 4));
    try testing.expectError(error.EngramTokenMap, loadTokenMap(testing.allocator, dir, 5, 3));
}

// Row reuse of the release's Engram hash over a token stream: `SUSHI_DSV41_ENGRAM_REUSE=<u32 tokens file>`, read as
// documents of 2048 tokens in file order (a file of u32 token ids). Per table, `RowCache`'s hit rate at
// several capacities. Prints only under the switch.
test "dsv41 engram: row reuse by cache capacity over a token file (SUSHI_DSV41_ENGRAM_REUSE=<tokens>)" {
    const path = std.c.getenv("SUSHI_DSV41_ENGRAM_REUSE") orelse return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.sliceTo(path, 0), a, .limited(1 << 30));
    const toks = std.mem.bytesAsSlice(u32, raw[0 .. raw.len / 4 * 4]);
    const e = releaseEngram();
    const map = try a.alloc(u32, 129280);
    try unpackTokenMap(map, v41_token_map);
    const h = try Hash.init(e, map[e.pad_id]);
    const nc = h.cols();
    const doc: usize = 2048;
    const caps = [_]u64{ 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24 };
    var caches: [caps.len][2]RowCache = undefined;
    for (&caches, caps) |*pair, cap| for (pair) |*c| {
        c.* = try RowCache.init(a, cap, 0, 0);
    };
    const comp = try a.alloc(u32, doc);
    const rows = try a.alloc(u32, doc * h.n_layers * nc);
    var missed: [MAX_COLS]u32 = undefined;
    var d: usize = 0;
    while ((d + 1) * doc <= toks.len) : (d += 1) {
        for (comp, toks[d * doc ..][0..doc]) |*o, t| o.* = if (t < map.len) map[t] else map[e.pad_id];
        h.rows(&.{}, comp, rows);
        for (0..doc) |t| for (0..h.n_layers) |l| {
            const ids = rows[(t * h.n_layers + l) * nc ..][0..nc];
            for (&caches) |*pair| {
                const k = pair[l].lookup(ids, &.{}, &.{}, &missed);
                pair[l].insert(ids, missed[0..k], &.{}, &.{});
            }
        };
    }
    for (caches, caps) |pair, cap| {
        const rate = [2]f64{ @as(f64, @floatFromInt(pair[0].hits)) / @as(f64, @floatFromInt(pair[0].hits + pair[0].misses)), @as(f64, @floatFromInt(pair[1].hits)) / @as(f64, @floatFromInt(pair[1].hits + pair[1].misses)) };
        std.debug.print("[engram-reuse] {d} docs, {d} rows a table ({d:.0} MB a table at 272 B a row): hit layer 1 {d:.3}, layer 14 {d:.3}\n", .{ d, cap, @as(f64, @floatFromInt(RowCache.bytesFor(cap, 256, 8))) / 1e6, rate[0], rate[1] });
    }
}
