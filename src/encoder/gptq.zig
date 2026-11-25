//! GPTQ into the mlx affine layout a served trunk stores: per output row and input group, a bf16 scale and bias,
//! codes in [0, 2^bits). Columns are rounded left to right, and each column's error is fed to the columns after it
//! through the upper Cholesky factor U of the damped inverse input Gram (H^-1 = U'U), 128 columns at a time.
const std = @import("std");
const quantize = @import("quantize.zig");
const pool_mod = @import("pool.zig");

pub const damp: f32 = 0.01;
const block = 128;
/// Output rows rounded together: they share each streamed row of U.
const row_tile = 8;

pub const Grid = struct { bits: u8, group: usize };

/// U of one input Gram, shared by every linear that reads that input.
pub const Factor = struct {
    n: usize,
    /// Row-major upper triangle, float32 as the rounding loop reads it.
    u: []f32,
    /// Input columns with no energy: their weights are zeroed before rounding.
    dead: []bool,

    /// `h` is X'X / rows ([n, n]). Dead diagonal entries become 1, the diagonal gains `damp` times its mean, and
    /// U = R^-1 for the flipped Cholesky H = R R' (R upper), which is chol(H^-1)' computed without the inverse.
    pub fn init(allocator: std.mem.Allocator, pool: ?*pool_mod.Pool, h: []const f32, n: usize) !Factor {
        std.debug.assert(h.len == n * n);
        const dead = try allocator.alloc(bool, n);
        errdefer allocator.free(dead);
        var hd = try allocator.alloc(f32, n);
        defer allocator.free(hd);
        var sum: f64 = 0;
        for (0..n) |i| {
            dead[i] = h[i * n + i] == 0;
            hd[i] = if (dead[i]) 1 else h[i * n + i];
            sum += hd[i];
        }
        const ridge: f32 = damp * @as(f32, @floatCast(sum / @as(f64, @floatFromInt(n))));
        // Flipped: m[i][j] = H[n-1-i][n-1-j], so its lower factor M gives R = J M J.
        const m = try allocator.alloc(f64, n * n);
        defer allocator.free(m);
        for (0..n) |i| for (0..n) |j| {
            const src = (n - 1 - i) * n + (n - 1 - j);
            m[i * n + j] = if (i == j) @as(f64, hd[n - 1 - i] + ridge) else h[src];
        };
        if (quantize.choleskyLowerOn(pool, m, n) != 0) return error.GptqNotPositiveDefinite;
        // U = R^-1 = J M^-1 J; column c of M^-1 by forward substitution, kept as row c of `lt`.
        const lt = try allocator.alloc(f64, n * n);
        defer allocator.free(lt);
        var inv = Inverse{ .m = m, .lt = lt, .n = n };
        if (pool) |p| p.run(n, &inv, Inverse.column) else for (0..n) |c| inv.column(c);
        const u = try allocator.alloc(f32, n * n);
        errdefer allocator.free(u);
        @memset(u, 0);
        // U[i][j] = M^-1[n-1-i][n-1-j] = lt[n-1-j][n-1-i], for j >= i.
        const tile = 32;
        var ia: usize = 0;
        while (ia < n) : (ia += tile) {
            var ja: usize = ia;
            while (ja < n) : (ja += tile) {
                for (ia..@min(ia + tile, n)) |i| for (@max(ja, i)..@min(ja + tile, n)) |j| {
                    u[i * n + j] = @floatCast(lt[(n - 1 - j) * n + (n - 1 - i)]);
                };
            }
        }
        return .{ .n = n, .u = u, .dead = dead };
    }

    pub fn deinit(self: *Factor, allocator: std.mem.Allocator) void {
        allocator.free(self.u);
        allocator.free(self.dead);
    }
};

const Inverse = struct {
    m: []const f64,
    lt: []f64,
    n: usize,

    fn column(self: *Inverse, c: usize) void {
        const n = self.n;
        const v = self.lt[c * n ..][0..n];
        @memset(v[0..c], 0);
        for (c..n) |i| {
            const row = self.m[i * n ..][0..n];
            const s: f64 = if (i == c) 1 else 0;
            v[i] = (s - quantize.dotRows(row[c..], v[c..], i - c)) / row[i];
        }
    }
};

pub const Rounded = struct {
    /// [out, in], one code per weight.
    codes: []u8,
    /// [out, in / group] bf16 bit patterns.
    scales: []u16,
    biases: []u16,
    /// [out, in]: the weight the codes reconstruct.
    deq: []f32,

    pub fn deinit(self: *Rounded, allocator: std.mem.Allocator) void {
        allocator.free(self.codes);
        allocator.free(self.scales);
        allocator.free(self.biases);
        allocator.free(self.deq);
    }

    /// The codes packed as mlx stores them: 32 / bits consecutive codes per u32, little end first.
    pub fn packed32(self: *const Rounded, allocator: std.mem.Allocator, bits: u8) ![]u32 {
        const per: usize = 32 / @as(usize, bits);
        if (self.codes.len % per != 0) return error.GptqShape;
        const out = try allocator.alloc(u32, self.codes.len / per);
        for (out, 0..) |*o, wi| {
            var word: u32 = 0;
            for (0..per) |s| word |= @as(u32, self.codes[wi * per + s]) << @intCast(s * bits);
            o.* = word;
        }
        return out;
    }
};

/// The widths `packed32` can pack: whole codes per u32, which is not MLX's layout for 3, 5 and 6 bits.
pub fn bitsRounded(bits: u32) bool {
    return bits >= 1 and bits <= 8 and 32 % bits == 0;
}

/// Rounds `w` ([out, in] row-major) under `f`. Rows are independent, so row tiles spread over `pool`.
pub fn quantizeRows(allocator: std.mem.Allocator, pool: ?*pool_mod.Pool, w: []const f32, out: usize, in: usize, f: *const Factor, grid: Grid) !Rounded {
    if (w.len != out * in or f.n != in or in % grid.group != 0 or !bitsRounded(grid.bits)) return error.GptqShape;
    const groups = in / grid.group;
    var r = Rounded{
        .codes = try allocator.alloc(u8, out * in),
        .scales = undefined,
        .biases = undefined,
        .deq = undefined,
    };
    errdefer allocator.free(r.codes);
    r.scales = try allocator.alloc(u16, out * groups);
    errdefer allocator.free(r.scales);
    r.biases = try allocator.alloc(u16, out * groups);
    errdefer allocator.free(r.biases);
    r.deq = try allocator.alloc(f32, out * in);
    errdefer allocator.free(r.deq);
    var job = Rows{ .allocator = allocator, .w = w, .out = out, .in = in, .f = f, .grid = grid, .r = &r };
    const tiles = (out + row_tile - 1) / row_tile;
    if (pool) |p| p.run(tiles, &job, Rows.tile) else for (0..tiles) |t| job.tile(t);
    if (job.failed.load(.monotonic)) return error.OutOfMemory;
    return r;
}

const Rows = struct {
    allocator: std.mem.Allocator,
    w: []const f32,
    out: usize,
    in: usize,
    f: *const Factor,
    grid: Grid,
    r: *Rounded,
    failed: std.atomic.Value(bool) = .init(false),

    fn tile(self: *Rows, t: usize) void {
        self.tileOrFail(t) catch {
            self.failed.store(true, .monotonic);
        };
    }

    fn tileOrFail(self: *Rows, t: usize) !void {
        const in = self.in;
        const n_in: usize = in;
        const r0 = t * row_tile;
        const rows: usize = @min(row_tile, self.out - r0);
        const u = self.f.u;
        const maxq: f32 = @floatFromInt((@as(u32, 1) << @intCast(self.grid.bits)) - 1);
        const g = self.grid.group;
        const groups = in / g;
        // Working weights, row by row, and each row's errors of the current block.
        const ws = try self.allocator.alloc(f32, rows * n_in);
        defer self.allocator.free(ws);
        const errs = try self.allocator.alloc(f32, rows * block);
        defer self.allocator.free(errs);
        const acc = try self.allocator.alloc(f32, rows * n_in);
        defer self.allocator.free(acc);
        for (0..rows) |i| {
            @memcpy(ws[i * n_in ..][0..n_in], self.w[(r0 + i) * n_in ..][0..n_in]);
            for (self.f.dead, 0..) |d, c| {
                if (d) ws[i * n_in + c] = 0;
            }
        }
        var b0: usize = 0;
        while (b0 < in) : (b0 += block) {
            const b1: usize = @min(b0 + block, in);
            for (0..rows) |i| {
                const row = ws[i * n_in ..][0..n_in];
                const row_err = errs[i * block ..][0..block];
                const gi = r0 + i;
                for (b0..b1) |j| {
                    const grp = j / g;
                    if (j % g == 0) {
                        var lo = row[j];
                        var hi = row[j];
                        for (row[j..][0..g]) |v| {
                            lo = @min(lo, v);
                            hi = @max(hi, v);
                        }
                        var scale = bf16Round((hi - lo) / maxq);
                        if (scale == 0) scale = 1;
                        self.r.scales[gi * groups + grp] = bf16Bits(scale);
                        self.r.biases[gi * groups + grp] = bf16Bits(bf16Round(lo));
                    }
                    const scale = bf16Value(self.r.scales[gi * groups + grp]);
                    const bias = bf16Value(self.r.biases[gi * groups + grp]);
                    const q = std.math.clamp(roundEven((row[j] - bias) / scale), 0, maxq);
                    const deq = q * scale + bias;
                    self.r.codes[gi * n_in + j] = @intFromFloat(q);
                    const e = (row[j] - deq) / u[j * in + j];
                    const urow = u[j * in ..][0..in];
                    for (j + 1..b1) |c| row[c] -= e * urow[c];
                    row_err[j - b0] = e;
                    row[j] = deq;
                }
            }
            if (b1 < in) {
                // The block's errors reach every later column: row -= errs @ U[b0..b1, b1..], each product summed
                // in k order before the one subtraction; a row of U serves the whole tile.
                const width = in - b1;
                for (0..rows) |i| @memset(acc[i * n_in ..][0..width], 0);
                for (b0..b1) |k| {
                    const urow = u[k * in + b1 ..][0..width];
                    for (0..rows) |i| {
                        const e = errs[i * block + (k - b0)];
                        for (acc[i * n_in ..][0..width], urow) |*x, uv| x.* += e * uv;
                    }
                }
                for (0..rows) |i| {
                    for (ws[i * n_in + b1 ..][0..width], acc[i * n_in ..][0..width]) |*x, d| x.* -= d;
                }
            }
        }
        for (0..rows) |i| {
            const gi = r0 + i;
            for (0..in) |c| {
                const grp = c / g;
                self.r.deq[gi * in + c] = @as(f32, @floatFromInt(self.r.codes[gi * in + c])) * bf16Value(self.r.scales[gi * groups + grp]) + bf16Value(self.r.biases[gi * groups + grp]);
            }
        }
    }
};

fn roundEven(x: f32) f32 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return 2 * @round(x / 2);
    return r;
}

/// Round to nearest even bf16, as an f32.
fn bf16Round(x: f32) f32 {
    return bf16Value(bf16Bits(x));
}

fn bf16Bits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u +% 0x7FFF +% ((u >> 16) & 1)) >> 16);
}

fn bf16Value(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// tr(E H E') / tr(W H W') with E = deq - w: the error the input Gram `h` weighs, relative to the weight's.
pub fn relErr(deq: []const f32, w: []const f32, h: []const f32, out: usize, in: usize) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    const e = std.heap.page_allocator.alloc(f64, in) catch return std.math.nan(f64);
    defer std.heap.page_allocator.free(e);
    for (0..out) |r| {
        for (e, deq[r * in ..][0..in], w[r * in ..][0..in]) |*o, d, x| o.* = @as(f64, d) - x;
        const wr = w[r * in ..][0..in];
        for (0..in) |i| {
            var se: f64 = 0;
            var sw: f64 = 0;
            const hr = h[i * in ..][0..in];
            for (hr, e, wr) |hv, ev, wv| {
                se += hv * ev;
                sw += hv * wv;
            }
            num += e[i] * se;
            den += @as(f64, wr[i]) * sw;
        }
    }
    return num / den;
}

const st = @import("safetensors.zig");
const fixture = @import("safetensors.zig").gptq;

fn gramOf(allocator: std.mem.Allocator, x: []const f32, rows: usize, n: usize) ![]f32 {
    const h = try allocator.alloc(f32, n * n);
    for (0..n) |a| for (0..n) |b| {
        var s: f64 = 0;
        for (0..rows) |i| s += @as(f64, x[i * n + a]) * x[i * n + b];
        h[a * n + b] = @floatCast(s / @as(f64, @floatFromInt(rows)));
    };
    return h;
}

// The reference's GPTQ on the same deterministic (X, W), with a dead input column, at group 64 and 32: it solves in
// float32 LAPACK and BLAS and this factor in float64, and every code, scale and bias still agrees.
test "gptq: rounds a deterministic case as the reference GPTQ does" {
    const t = std.testing;
    var file = try st.parse(t.allocator, fixture);
    defer file.deinit(t.allocator);
    const x = st.slice(f32, file.tensors.get("x").?);
    const w = st.slice(f32, file.tensors.get("w").?);
    const rows = 500;
    const in = 192;
    const out = 12;
    const h = try gramOf(t.allocator, x, rows, in);
    defer t.allocator.free(h);
    var f = try Factor.init(t.allocator, null, h, in);
    defer f.deinit(t.allocator);
    try t.expect(f.dead[70]);
    for ([_]usize{ 64, 32 }) |group| {
        var name: [16]u8 = undefined;
        var r = try quantizeRows(t.allocator, null, w, out, in, &f, .{ .bits = 8, .group = group });
        defer r.deinit(t.allocator);
        const want_q = st.slice(u8, file.tensors.get(try std.fmt.bufPrint(&name, "q{d}", .{group})).?);
        const want_s = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&name, "s{d}", .{group})).?);
        const want_b = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&name, "b{d}", .{group})).?);
        const want_d = st.slice(f32, file.tensors.get(try std.fmt.bufPrint(&name, "deq{d}", .{group})).?);
        const want_c = st.slice(u32, file.tensors.get(try std.fmt.bufPrint(&name, "codes{d}", .{group})).?);
        try t.expectEqualSlices(u8, want_q, r.codes);
        try t.expectEqualSlices(u16, want_s, r.scales);
        try t.expectEqualSlices(u16, want_b, r.biases);
        try t.expectEqualSlices(f32, want_d, r.deq);
        const words = try r.packed32(t.allocator, 8);
        defer t.allocator.free(words);
        try t.expectEqualSlices(u32, want_c, words);
    }
}

test "gptq: with an identity Gram every column rounds to its nearest grid point" {
    const t = std.testing;
    const in = 128;
    const out = 5;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const w = try t.allocator.alloc(f32, out * in);
    defer t.allocator.free(w);
    for (w) |*v| v.* = rnd.floatNorm(f32);
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    @memset(h, 0);
    for (0..in) |i| h[i * in + i] = 1;
    var f = try Factor.init(t.allocator, null, h, in);
    defer f.deinit(t.allocator);
    var r = try quantizeRows(t.allocator, null, w, out, in, &f, .{ .bits = 4, .group = 32 });
    defer r.deinit(t.allocator);
    for (0..out) |row| for (0..in) |c| {
        const grp = row * (in / 32) + c / 32;
        const s = bf16Value(r.scales[grp]);
        const b = bf16Value(r.biases[grp]);
        const want = std.math.clamp(roundEven((w[row * in + c] - b) / s), 0, 15);
        try t.expectEqual(want, @as(f32, @floatFromInt(r.codes[row * in + c])));
    };
}

test "gptq: a pool rounds and factors to the same bytes as one thread" {
    const t = std.testing;
    const in = 320;
    const out = 21;
    var prng = std.Random.DefaultPrng.init(4);
    const rnd = prng.random();
    const x = try t.allocator.alloc(f32, 700 * in);
    defer t.allocator.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);
    const w = try t.allocator.alloc(f32, out * in);
    defer t.allocator.free(w);
    for (w) |*v| v.* = rnd.floatNorm(f32) * 0.05;
    const h = try gramOf(t.allocator, x, 700, in);
    defer t.allocator.free(h);
    const pool = try pool_mod.Pool.init(t.allocator, 5);
    defer pool.deinit();
    var f1 = try Factor.init(t.allocator, null, h, in);
    defer f1.deinit(t.allocator);
    var f2 = try Factor.init(t.allocator, pool, h, in);
    defer f2.deinit(t.allocator);
    try t.expectEqualSlices(f32, f1.u, f2.u);
    var r1 = try quantizeRows(t.allocator, null, w, out, in, &f1, .{ .bits = 8, .group = 64 });
    defer r1.deinit(t.allocator);
    var r2 = try quantizeRows(t.allocator, pool, w, out, in, &f2, .{ .bits = 8, .group = 64 });
    defer r2.deinit(t.allocator);
    try t.expectEqualSlices(u8, r1.codes, r2.codes);
    try t.expectEqualSlices(f32, r1.deq, r2.deq);
    try t.expect(relErr(r1.deq, w, h, out, in) < 1e-3);
}
