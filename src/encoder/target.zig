//! The compensated rounding target of a sequential conversion. X~ is a linear's input on the quantized path (the
//! layers below already rounded) and X its input on the BF16 path, over the same routed tokens. The quantizer then
//! rounds W* = W + alpha (M W - W) under X~'X~, M = (X~'X~ + damp mean diag)^-1 X~'X: it moves the weight part of the
//! way toward the one that maps the drifted inputs onto the outputs the BF16 model computes.
const std = @import("std");
const quantize = @import("quantize.zig");

pub const alpha: f64 = 0.2;
pub const damp: f64 = 0.01;
/// At or below this many routed rows an expert keeps W: its X~'X~ is too noisy for M.
pub const thin_rows: u64 = 2560;

/// Full compensation from four times the input width (never below `thin_rows`), none at or below `thin_rows`,
/// linear in between.
pub fn alphaForRows(rows: u64, n_in: usize) f64 {
    const full: u64 = @max(4 * @as(u64, n_in), thin_rows);
    if (rows <= thin_rows) return 0;
    if (rows >= full) return alpha;
    return alpha * @as(f64, @floatFromInt(rows - thin_rows)) / @as(f64, @floatFromInt(full - thin_rows));
}

/// Replaces `w` ([n, k] row-major, the public weight or several sharing one input side by side) with its target.
/// `h` is X~'X~ ([n, n]) and `cw` is X~'X W ([n, k]); both are sums over the same rows, so their scale cancels.
pub fn compensate(allocator: std.mem.Allocator, w: []f32, h: []const f32, cw: []const f32, n: usize, k: usize, a: f64) !void {
    std.debug.assert(w.len == n * k and h.len == n * n and cw.len == n * k);
    if (a == 0) return;
    var f = try Factor.init(allocator, h, n);
    defer f.deinit(allocator);
    try f.apply(allocator, w, cw, k, 0, k, a);
}

/// The Cholesky factor of X~'X~ + damp mean diag: L in the lower triangle and L' mirrored into the upper, so both
/// solves walk contiguous rows. One factor serves any split of the right-hand sides.
pub const Factor = struct {
    m: []f64,
    n: usize,

    pub fn init(allocator: std.mem.Allocator, h: []const f32, n: usize) !Factor {
        const m = try allocator.alloc(f64, n * n);
        errdefer allocator.free(m);
        var diag: f64 = 0;
        for (0..n) |i| diag += h[i * n + i];
        const ridge = damp * diag / @as(f64, @floatFromInt(n));
        for (m, h) |*o, v| o.* = v;
        for (0..n) |i| m[i * n + i] += ridge;
        if (quantize.choleskyLower(m, n) != 0) return error.TargetNotPositiveDefinite;
        const tile = 32;
        var r0: usize = 0;
        while (r0 < n) : (r0 += tile) {
            var c0: usize = r0;
            while (c0 < n) : (c0 += tile) {
                for (r0..@min(r0 + tile, n)) |r| for (@max(c0, r + 1)..@min(c0 + tile, n)) |c| {
                    m[r * n + c] = m[c * n + r];
                };
            }
        }
        return .{ .m = m, .n = n };
    }

    pub fn deinit(self: *Factor, allocator: std.mem.Allocator) void {
        allocator.free(self.m);
    }

    /// Columns [c0, c1) of `w` ([n, k] row-major) become the target's, from the same columns of `cw`.
    pub fn apply(self: *const Factor, allocator: std.mem.Allocator, w: []f32, cw: []const f32, k: usize, c0: usize, c1: usize, a: f64) !void {
        const n = self.n;
        const cols = c1 - c0;
        const yt = try allocator.alloc(f64, cols * n);
        defer allocator.free(yt);
        for (0..n) |r| for (0..cols) |c| {
            yt[c * n + r] = cw[r * k + c0 + c];
        };
        solve(self.m, yt, n, cols, false);
        solve(self.m, yt, n, cols, true);
        for (0..n) |r| for (0..cols) |c| {
            const w64: f64 = w[r * k + c0 + c];
            w[r * k + c0 + c] = @floatCast(w64 + a * (yt[c * n + r] - w64));
        };
    }
};

const Lanes = @Vector(4, f64);
const blk = 64;
/// The update walks the reduction in spans of this many terms, keeping each output's four lane sums between spans, so
/// a tile's rows of `a` and columns of `yt` stay in L1; every output still adds its terms in the same order.
const span = 256;
const tile_rows = 8;
const tile_cols = 16;

/// yt[c][r] -= sum over m in [m0, m1) of t(r, m) yt[c][m] for r in [r0, r1) and every column c, t row-major in `a`.
fn update(a: []const f64, yt: []f64, n: usize, k: usize, r0: usize, r1: usize, m0: usize, m1: usize) void {
    if (m1 <= m0) return;
    const len = m1 - m0;
    const vec_len = len - len % 4;
    var acc: [tile_rows][tile_cols]Lanes = undefined;
    var rt: usize = r0;
    while (rt < r1) : (rt += tile_rows) {
        const rows = @min(tile_rows, r1 - rt);
        var ct: usize = 0;
        while (ct < k) : (ct += tile_cols) {
            const cols = @min(tile_cols, k - ct);
            for (0..rows) |i| @memset(acc[i][0..cols], @splat(0));
            var s0: usize = 0;
            while (s0 < vec_len) : (s0 += span) {
                const s1 = @min(s0 + span, vec_len);
                var i: usize = 0;
                while (i < rows) : (i += 2) {
                    const pair = i + 1 < rows;
                    const a0 = a[(rt + i) * n + m0 ..][0..len];
                    const a1 = if (pair) a[(rt + i + 1) * n + m0 ..][0..len] else a0;
                    var c: usize = 0;
                    while (c < cols) : (c += 4) {
                        const q_n = @min(4, cols - c);
                        var acc0: [4]Lanes = undefined;
                        var acc1: [4]Lanes = undefined;
                        inline for (0..4) |q| if (q < q_n) {
                            acc0[q] = acc[i][c + q];
                            acc1[q] = if (pair) acc[i + 1][c + q] else @splat(0);
                        };
                        var mi = s0;
                        while (mi < s1) : (mi += 4) {
                            const x0: Lanes = a0[mi..][0..4].*;
                            const x1: Lanes = a1[mi..][0..4].*;
                            inline for (0..4) |q| {
                                if (q < q_n) {
                                    const y: Lanes = yt[(ct + c + q) * n + m0 + mi ..][0..4].*;
                                    acc0[q] = @mulAdd(Lanes, x0, y, acc0[q]);
                                    acc1[q] = @mulAdd(Lanes, x1, y, acc1[q]);
                                }
                            }
                        }
                        inline for (0..4) |q| if (q < q_n) {
                            acc[i][c + q] = acc0[q];
                            if (pair) acc[i + 1][c + q] = acc1[q];
                        };
                    }
                }
            }
            for (0..rows) |i| {
                const ar = a[(rt + i) * n + m0 ..][0..len];
                for (0..cols) |c| {
                    const yc = yt[(ct + c) * n + m0 ..][0..len];
                    var sum = @reduce(.Add, acc[i][c]);
                    for (vec_len..len) |tt| sum = @mulAdd(f64, ar[tt], yc[tt], sum);
                    yt[(ct + c) * n + rt + i] -= sum;
                }
            }
        }
    }
}

/// Forward substitution with the lower factor in `a` (`upper` false), or back substitution with the upper one
/// (true), on every column of `yt` in place.
fn solve(a: []const f64, yt: []f64, n: usize, k: usize, upper: bool) void {
    var done: usize = 0;
    while (done < n) {
        const kb = @min(blk, n - done);
        const b0 = if (upper) n - done - kb else done;
        const b1 = b0 + kb;
        if (upper) update(a, yt, n, k, b0, b1, b1, n) else update(a, yt, n, k, b0, b1, 0, b0);
        var step: usize = 0;
        while (step < kb) : (step += 1) {
            const r = if (upper) b1 - 1 - step else b0 + step;
            const d = a[r * n + r];
            for (0..k) |c| {
                var s = yt[c * n + r];
                if (upper) {
                    for (r + 1..b1) |mi| s = @mulAdd(f64, -a[r * n + mi], yt[c * n + mi], s);
                } else {
                    for (b0..r) |mi| s = @mulAdd(f64, -a[r * n + mi], yt[c * n + mi], s);
                }
                yt[c * n + r] = s / d;
            }
        }
        done += kb;
    }
}

test "target: the ramp from thin to full compensation is the reference's" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 0), alphaForRows(2560, 2560));
    try t.expectApproxEqRel(@as(f64, 2.604166666666667e-05), alphaForRows(2561, 2560), 1e-12);
    try t.expectApproxEqRel(@as(f64, 0.1), alphaForRows(6400, 2560), 1e-12);
    try t.expectEqual(alpha, alphaForRows(10240, 2560));
    try t.expectEqual(alpha, alphaForRows(20000, 2560));
    try t.expectEqual(alpha, alphaForRows(2561, 640));
    try t.expectEqual(@as(f64, 0), alphaForRows(100, 640));
}

// The same deterministic (X~, X, W) as the reference fixture script; the pinned values are its output.
test "target: the compensated target matches the reference solve on the same inputs" {
    const t = std.testing;
    const n = 96;
    const k = 8;
    const rows = 400;
    const xq = try t.allocator.alloc(f32, rows * n);
    defer t.allocator.free(xq);
    const xf = try t.allocator.alloc(f32, rows * n);
    defer t.allocator.free(xf);
    for (0..rows) |i| for (0..n) |j| {
        const fi: f64 = @floatFromInt(i);
        const fj: f64 = @floatFromInt(j);
        const q: f32 = @floatCast(@sin(fi * 0.37 + fj * 1.3) + 0.5 * @cos(fi * fj * 0.011));
        xq[i * n + j] = q;
        xf[i * n + j] = @floatCast(@as(f64, q) + 0.05 * @sin(fi * 0.11 + fj * 0.7));
    };
    var w: [n * k]f32 = undefined;
    for (0..n) |a| for (0..k) |b| {
        w[a * k + b] = @floatCast(0.1 * @sin(@as(f64, @floatFromInt(a)) * 0.23 + @as(f64, @floatFromInt(b)) * 0.9));
    };
    var h: [n * n]f32 = undefined;
    var cross: [n * n]f64 = undefined;
    for (0..n) |a| for (0..n) |b| {
        var s: f64 = 0;
        var c: f64 = 0;
        for (0..rows) |i| {
            s += @as(f64, xq[i * n + a]) * xq[i * n + b];
            c += @as(f64, xq[i * n + a]) * xf[i * n + b];
        }
        h[a * n + b] = @floatCast(s);
        cross[a * n + b] = @as(f64, @as(f32, @floatCast(c)));
    };
    var cw: [n * k]f32 = undefined;
    for (0..n) |a| for (0..k) |b| {
        var s: f64 = 0;
        for (0..n) |m| s += cross[a * n + m] * w[m * k + b];
        cw[a * k + b] = @floatCast(s);
    };
    var got = w;
    try compensate(t.allocator, &got, &h, &cw, n, k, alpha);
    var sum: f64 = 0;
    var sumsq: f64 = 0;
    var dmax: f64 = 0;
    for (got, w) |g, w0| {
        sum += g;
        sumsq += @as(f64, g) * g;
        dmax = @max(dmax, @abs(@as(f64, g) - w0));
    }
    try t.expectApproxEqRel(@as(f64, 0.8729233094200026), sum, 1e-5);
    try t.expectApproxEqRel(@as(f64, 3.7295506378419043), sumsq, 1e-6);
    try t.expectApproxEqRel(@as(f64, 0.012214142829179764), dmax, 1e-4);
    const want0 = [_]f32{ 7.624845602549613e-05, 0.07773004472255707, 0.09655929356813431, 0.04231439158320427 };
    const want95 = [_]f32{ 0.030858663842082024, 0.09258158504962921, 0.0842406153678894, 0.012148021720349789 };
    for (want0, got[0..4]) |a, b| try t.expectApproxEqAbs(a, b, 2e-6);
    for (want95, got[95 * k + 4 ..][0..4]) |a, b| try t.expectApproxEqAbs(a, b, 2e-6);
}

test "target: a factor applied in column chunks gives the bytes of one whole solve" {
    const t = std.testing;
    const n = 80;
    const k = 11;
    var prng = std.Random.DefaultPrng.init(6);
    const r = prng.random();
    var h: [n * n]f32 = undefined;
    for (0..n) |i| for (0..n) |j| {
        h[i * n + j] = if (i == j) 3 else @as(f32, @floatFromInt((i * 7 + j * 7) % 5)) * 0.01;
    };
    var w: [n * k]f32 = undefined;
    var cw: [n * k]f32 = undefined;
    for (&w, &cw) |*a, *b| {
        a.* = r.floatNorm(f32);
        b.* = r.floatNorm(f32);
    }
    var whole = w;
    try compensate(t.allocator, &whole, &h, &cw, n, k, alpha);
    var parts = w;
    var f = try Factor.init(t.allocator, &h, n);
    defer f.deinit(t.allocator);
    try f.apply(t.allocator, &parts, &cw, k, 0, 3, alpha);
    try f.apply(t.allocator, &parts, &cw, k, 3, k, alpha);
    try t.expectEqualSlices(f32, &whole, &parts);
}

test "target: the blocked triangular solves invert a Cholesky factor larger than one block" {
    const t = std.testing;
    const n = 150;
    const k = 13;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    const x = try t.allocator.alloc(f64, n * 3 * n);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f64);
    const a = try t.allocator.alloc(f64, n * n);
    defer t.allocator.free(a);
    for (0..n) |i| for (0..n) |j| {
        var s: f64 = 0;
        for (0..3 * n) |m| s += x[m * n + i] * x[m * n + j];
        a[i * n + j] = s;
    };
    const l = try t.allocator.dupe(f64, a);
    defer t.allocator.free(l);
    try t.expectEqual(@as(usize, 0), quantize.choleskyLower(l, n));
    for (0..n) |i| for (i + 1..n) |j| {
        l[i * n + j] = l[j * n + i];
    };
    const b = try t.allocator.alloc(f64, n * k);
    defer t.allocator.free(b);
    for (b) |*v| v.* = r.floatNorm(f64);
    const yt = try t.allocator.alloc(f64, k * n);
    defer t.allocator.free(yt);
    for (0..n) |i| for (0..k) |c| {
        yt[c * n + i] = b[i * k + c];
    };
    solve(l, yt, n, k, false);
    solve(l, yt, n, k, true);
    for (0..n) |i| for (0..k) |c| {
        var s: f64 = 0;
        for (0..n) |m| s += a[i * n + m] * yt[c * n + m];
        try t.expectApproxEqAbs(b[i * k + c], s, 1e-8 * @max(1, @abs(b[i * k + c])));
    };
}

// (1 + e)(1 - e) = 1 - e^2 with e = 2^-27 needs 55 bits: a product rounded on its own is 1 and the sum loses e^2.
test "target: the update and the solves join each product to their sum with one rounding" {
    const t = std.testing;
    const e = std.math.ldexp(@as(f64, 1), -27);
    const n = 16;
    var a: [n * n]f64 = @splat(0);
    var yt: [n]f64 = @splat(0);
    a[8 * n + 0] = 1;
    a[8 * n + 4] = 1 + e;
    yt[0] = -1;
    yt[4] = 1 - e;
    update(&a, &yt, n, 1, 8, 9, 0, 8);
    try t.expectEqual(e * e, yt[8]);
    a[9 * n + 0] = 1;
    a[9 * n + 8] = 1 + e;
    yt[8] = 1 - e;
    update(&a, &yt, n, 1, 9, 10, 0, 9);
    try t.expectEqual(e * e, yt[9]);

    var l = [_]f64{ 1, 1 + e, 1 + e, 1 };
    var lo = [_]f64{ 1 - e, 1 };
    solve(&l, &lo, 2, 1, false);
    try t.expectEqual(e * e, lo[1]);
    var up = [_]f64{ 1, 1 - e };
    solve(&l, &up, 2, 1, true);
    try t.expectEqual(e * e, up[0]);
}
