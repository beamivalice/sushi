const std = @import("std");
const format = @import("sushi_exl3").format;
const encode = @import("encode.zig");
const st = @import("safetensors.zig");

pub const had_dim = format.HAD_DIM;
pub const sigma_reg: f32 = 0.025;
pub const codebook_scale: f32 = 1.24371088;
const out_scale_skew_cutoff = 0.15;
const g_scale_tol = 0.03;
const g_scale_max_evals = 10;
const g_scale_tiles = 128;
const g_scale_width = 3;
const ldl_block = 16;
const ldlq_buffer_rows = 128;
const ldlq_feedback_rows = 16;
pub const max_cholesky_retries = 10;

pub const OutScales = enum { auto, always, never };

pub const Regularized = struct {
    inner: []f32,
    suh: []f32,
    svh: []f32,

    pub fn deinit(self: *Regularized, allocator: std.mem.Allocator) void {
        allocator.free(self.inner);
        allocator.free(self.suh);
        allocator.free(self.svh);
    }
};

fn fwht(comptime T: type, v: *[had_dim]T) void {
    var h: usize = 1;
    while (h < had_dim) : (h *= 2) {
        var i: usize = 0;
        while (i < had_dim) : (i += 2 * h) {
            for (i..i + h) |j| {
                const a = v[j];
                const b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    const scale: T = 1.0 / @sqrt(@as(T, had_dim));
    for (v) |*x| x.* *= scale;
}

/// `fwht` down every column of each 128-row block, run as butterflies on whole rows so the inner loops are
/// contiguous; every element sees the same adds, subtracts and final scale as the column-at-a-time transform.
fn hadamardRows(comptime T: type, m: []T, rows: usize, cols: usize) void {
    const scale: T = 1.0 / @sqrt(@as(T, had_dim));
    var b: usize = 0;
    while (b < rows) : (b += had_dim) {
        const blk = m[b * cols ..][0 .. had_dim * cols];
        var h: usize = 1;
        while (h < had_dim) : (h *= 2) {
            var i: usize = 0;
            while (i < had_dim) : (i += 2 * h) {
                for (i..i + h) |j| {
                    for (blk[j * cols ..][0..cols], blk[(j + h) * cols ..][0..cols]) |*x, *y| {
                        const a = x.*;
                        const c = y.*;
                        x.* = a + c;
                        y.* = a - c;
                    }
                }
            }
        }
        for (blk) |*x| x.* *= scale;
    }
}

fn hadamardCols(comptime T: type, m: []T, rows: usize, cols: usize) void {
    var v: [had_dim]T = undefined;
    for (0..rows) |r| {
        var b: usize = 0;
        while (b < cols) : (b += had_dim) {
            @memcpy(&v, m[r * cols + b ..][0..had_dim]);
            fwht(T, &v);
            @memcpy(m[r * cols + b ..][0..had_dim], &v);
        }
    }
}

const rms_block = 32;

fn blockRms(sq: f32, n: usize) f32 {
    return @sqrt(sq / @as(f32, @floatFromInt(n)));
}

fn chooseApplyOutScales(allocator: std.mem.Allocator, mode: OutScales, hdiag: ?[]const f32) !bool {
    const forced: ?bool = switch (mode) {
        .auto => null,
        .always => true,
        .never => false,
    };
    const d = hdiag orelse return forced orelse true;
    var sum: f64 = 0;
    for (d) |v| sum += @max(v, 0);
    const mean = sum / @as(f64, @floatFromInt(d.len));
    if (!std.math.isFinite(mean) or mean < 1e-20) return forced orelse false;
    if (forced) |f| return f;
    const diag = try allocator.alloc(f32, d.len);
    defer allocator.free(diag);
    const damp: f32 = @floatCast(@as(f64, sigma_reg) * mean);
    for (diag, d) |*o, v| o.* = @sqrt(@max(v, 0) + damp);
    std.mem.sort(f32, diag, {}, std.sort.desc(f32));
    const cutoff = @max(1, diag.len / 50);
    var top: f64 = 0;
    var all: f64 = 0;
    for (diag, 0..) |v, i| {
        if (i < cutoff) top += v;
        all += v;
    }
    return top / all < out_scale_skew_cutoff;
}

/// Public weight [in, out] row-major to the inner matrix the trellis quantizes, with the scale
/// vectors the decoder multiplies back in: public = diag(suh) H_L inner H_R diag(svh).
fn admitGeometry(in: usize, out: usize) !void {
    if (in == 0 or out == 0) return error.EmptyTensor;
    if (in % 16 != 0 or out % 16 != 0 or in % had_dim != 0 or out % had_dim != 0) return error.GeometryUnsupported;
}

fn admitRate(rate: format.Rate, window: format.Window) !void {
    if (rate.n % 2 != 0 or rate.n < format.Rate.min_n or rate.n > format.Rate.max_n) return error.RateUnsupported;
    if (window.bits() < 12 or window.bits() > 16) return error.WindowUnsupported;
}

pub fn regularize(allocator: std.mem.Allocator, w: []const f32, in: usize, out: usize, sign_in: []const f32, sign_out: []const f32, mode: OutScales, hdiag: ?[]const f32) !Regularized {
    try admitGeometry(in, out);
    if (w.len != in * out or sign_in.len != in or sign_out.len != out) return error.ShapeMismatch;
    if (hdiag) |d| if (d.len != in) return error.ShapeMismatch;
    const weight = try allocator.dupe(f32, w);
    errdefer allocator.free(weight);
    const suh = try allocator.alloc(f32, in);
    errdefer allocator.free(suh);
    const svh = try allocator.dupe(f32, sign_out);
    errdefer allocator.free(svh);
    const out_scales = try allocator.alloc(f32, out);
    defer allocator.free(out_scales);

    var apply = try chooseApplyOutScales(allocator, mode, hdiag);
    @memset(out_scales, 0);
    for (0..out) |c| {
        var total: f32 = 0;
        var r0: usize = 0;
        while (r0 < in) : (r0 += rms_block) {
            var block: f32 = 0;
            for (r0..@min(r0 + rms_block, in)) |r| block += weight[r * out + c] * weight[r * out + c];
            total += block;
        }
        out_scales[c] = blockRms(total, in);
    }
    var mean_sum: f64 = 0;
    for (out_scales) |v| mean_sum += v;
    const out_mean = mean_sum / @as(f64, @floatFromInt(out));
    if (out_mean <= 1e-30) {
        @memset(out_scales, 0);
        if (mode != .auto) apply = true;
    } else {
        const m: f32 = @floatCast(out_mean);
        for (out_scales) |*v| v.* /= m;
    }
    if (apply) {
        for (svh, out_scales) |*s, o| s.* = s.* * (if (@abs(o) < 1e-30) 0.1 else o) + 1e-10;
    }
    for (0..in) |r| for (0..out) |c| {
        weight[r * out + c] /= svh[c];
    };
    for (svh, out_scales) |*s, o| if (@abs(o) < 1e-30) {
        s.* = 0;
    };
    hadamardCols(f32, weight, in, out);

    for (0..in) |r| {
        var total: f32 = 0;
        var c0: usize = 0;
        while (c0 < out) : (c0 += rms_block) {
            var block: f32 = 0;
            for (c0..@min(c0 + rms_block, out)) |c| block += weight[r * out + c] * weight[r * out + c];
            total += block;
        }
        var in_scale = blockRms(total, out);
        if (@abs(in_scale) < 1e-30) in_scale = 0.1;
        suh[r] = sign_in[r] * in_scale / -codebook_scale + 1e-10;
    }
    for (0..in) |r| for (weight[r * out ..][0..out]) |*v| {
        v.* /= suh[r];
    };
    hadamardRows(f32, weight, in, out);
    return .{ .inner = weight, .suh = suh, .svh = svh };
}

/// The public input Hessian in the inner basis: H_L diag(suh) H diag(suh) H_L, in float64.
pub fn rotatedHessian(allocator: std.mem.Allocator, h: []const f32, suh: []const f32) ![]f32 {
    const n = suh.len;
    try admitGeometry(n, n);
    if (h.len != n * n) return error.ShapeMismatch;
    const m = try allocator.alloc(f64, n * n);
    defer allocator.free(m);
    for (0..n) |i| for (0..n) |j| {
        m[i * n + j] = @as(f64, h[i * n + j]) * suh[i] * suh[j];
    };
    hadamardRows(f64, m, n, n);
    hadamardCols(f64, m, n, n);
    const out = try allocator.alloc(f32, n * n);
    for (out, m) |*o, v| o.* = @floatCast(v);
    return out;
}

const chol_block = 64;
const Lanes = @Vector(4, f64);

fn dotRows(a: []const f64, b: []const f64, len: usize) f64 {
    var acc: Lanes = @splat(0);
    var m: usize = 0;
    while (m + 4 <= len) : (m += 4) {
        const x: Lanes = a[m..][0..4].*;
        const y: Lanes = b[m..][0..4].*;
        acc += x * y;
    }
    var sum = @reduce(.Add, acc);
    while (m < len) : (m += 1) sum += a[m] * b[m];
    return sum;
}

/// In-place lower Cholesky factor of the row-major n x n matrix `a` (upper triangle untouched).
/// Returns 0, or the 1-based index of the first pivot that is not positive.
fn choleskyLower(a: []f64, n: usize) usize {
    var k: usize = 0;
    while (k < n) : (k += chol_block) {
        const kb = @min(chol_block, n - k);
        for (0..kb) |j| {
            const jj = k + j;
            const d = a[jj * n + jj] - dotRows(a[jj * n + k ..], a[jj * n + k ..], j);
            if (!(d > 0)) return jj + 1;
            const root = @sqrt(d);
            a[jj * n + jj] = root;
            for (j + 1..kb) |i| {
                const ii = k + i;
                a[ii * n + jj] = (a[ii * n + jj] - dotRows(a[ii * n + k ..], a[jj * n + k ..], j)) / root;
            }
        }
        for (k + kb..n) |i| {
            for (0..kb) |j| {
                const jj = k + j;
                a[i * n + jj] = (a[i * n + jj] - dotRows(a[i * n + k ..], a[jj * n + k ..], j)) / a[jj * n + jj];
            }
        }
        trailingUpdate(a, n, k, kb);
    }
    return 0;
}

/// a[i][j] -= dotRows(a[i][k..k+kb], a[j][k..k+kb]) for k+kb <= j <= i. Tiles of two rows by four columns
/// share their loads; each element keeps dotRows' own lanes, order and reduction, so the result is bit-identical.
fn trailingUpdate(a: []f64, n: usize, k: usize, kb: usize) void {
    const lo = k + kb;
    const vec_len = kb - kb % 4;
    var i: usize = lo;
    while (i < n) : (i += 2) {
        const pair = i + 1 < n;
        const xi0 = a[i * n + k ..][0..kb];
        const xi1 = if (pair) a[(i + 1) * n + k ..][0..kb] else xi0;
        var j: usize = lo;
        while (j + 4 <= i + 1) : (j += 4) {
            var acc: [2][4]Lanes = @splat(@splat(@splat(0)));
            var m: usize = 0;
            while (m < vec_len) : (m += 4) {
                const x0: Lanes = xi0[m..][0..4].*;
                const x1: Lanes = xi1[m..][0..4].*;
                inline for (0..4) |c| {
                    const y: Lanes = a[(j + c) * n + k + m ..][0..4].*;
                    acc[0][c] += x0 * y;
                    acc[1][c] += x1 * y;
                }
            }
            inline for (0..4) |c| {
                var s0 = @reduce(.Add, acc[0][c]);
                var s1 = @reduce(.Add, acc[1][c]);
                const yj = a[(j + c) * n + k ..][0..kb];
                for (vec_len..kb) |t| {
                    s0 += xi0[t] * yj[t];
                    s1 += xi1[t] * yj[t];
                }
                a[i * n + j + c] -= s0;
                if (pair) a[(i + 1) * n + j + c] -= s1;
            }
        }
        for (j..i + 1) |jj| a[i * n + jj] -= dotRows(xi0, a[jj * n + k ..][0..kb], kb);
        if (pair) for (j..i + 2) |jj| {
            a[(i + 1) * n + jj] -= dotRows(xi1, a[jj * n + k ..][0..kb], kb);
        };
    }
}

/// Inverts the lower-triangular `blk` (row-major, ldl_block square) in place; false when a diagonal is zero or not finite.
fn invertLowerBlock(blk: *[ldl_block * ldl_block]f64) bool {
    var inv: [ldl_block * ldl_block]f64 = @splat(0);
    for (0..ldl_block) |c| {
        const d = blk[c * ldl_block + c];
        if (d == 0 or !std.math.isFinite(d)) return false;
        inv[c * ldl_block + c] = 1.0 / d;
        for (c + 1..ldl_block) |r| {
            var sum: f64 = 0;
            for (c..r) |m| sum += blk[r * ldl_block + m] * inv[m * ldl_block + c];
            inv[r * ldl_block + c] = -sum / blk[r * ldl_block + r];
        }
    }
    blk.* = inv;
    return true;
}

/// Fixes an empty diagonal entry and adds the damping `sigma_reg * mean(diag)` in place; returns the damped
/// diagonal mean.
pub fn dampDiagonal(h: []f32, n: usize) f64 {
    var diag_sum: f64 = 0;
    for (0..n) |i| {
        if (h[i * n + i] == 0) h[i * n + i] = 1;
        diag_sum += h[i * n + i];
    }
    const diag_mean = diag_sum / @as(f64, @floatFromInt(n));
    if (diag_mean > 0) {
        const damp: f32 = @floatCast(@as(f64, sigma_reg) * diag_mean);
        for (0..n) |i| h[i * n + i] += damp;
    }
    var damped_sum: f64 = 0;
    for (0..n) |i| damped_sum += h[i * n + i];
    return damped_sum / @as(f64, @floatFromInt(n));
}

/// The diagonal added to a Hessian whose Cholesky pivot failed.
pub fn retryBump(damped_mean: f64) f32 {
    return @floatCast(2.0 * @as(f64, sigma_reg) * damped_mean);
}

/// Strictly block-lower factor L of the damped Hessian, H = (I + L) D (I + L)^T with 16x16 blocks:
/// the Cholesky factor with each block column divided by its diagonal block, diagonal blocks zeroed.
pub fn ldlFactor(allocator: std.mem.Allocator, h_inner: []const f32, n: usize) ![]f32 {
    if (n == 0) return error.EmptyTensor;
    if (n % 16 != 0) return error.GeometryUnsupported;
    if (h_inner.len != n * n) return error.ShapeMismatch;
    const h = try allocator.dupe(f32, h_inner);
    defer allocator.free(h);
    const damped_mean = dampDiagonal(h, n);
    return ldlFactorDamped(allocator, h, n, damped_mean);
}

/// `ldlFactor` from a Hessian that `dampDiagonal` already treated; `h` takes the retry bumps.
pub fn ldlFactorDamped(allocator: std.mem.Allocator, h: []f32, n: usize, damped_mean: f64) ![]f32 {
    const chol64 = try allocator.alloc(f64, n * n);
    defer allocator.free(chol64);
    var retries: usize = 0;
    while (true) {
        for (chol64, h) |*o, v| o.* = v;
        if (choleskyLower(chol64, n) == 0) break;
        retries += 1;
        if (retries > max_cholesky_retries or damped_mean <= 0) return error.HessianNotPositiveDefinite;
        const bump = retryBump(damped_mean);
        for (0..n) |i| h[i * n + i] += bump;
    }
    const chol = try allocator.alloc(f32, n * n);
    defer allocator.free(chol);
    for (0..n) |i| for (0..n) |j| {
        chol[i * n + j] = if (j <= i) @floatCast(chol64[i * n + j]) else 0;
    };

    const l = try allocator.dupe(f32, chol);
    errdefer allocator.free(l);
    var blk: [ldl_block * ldl_block]f64 = undefined;
    var inv: [ldl_block * ldl_block]f32 = undefined;
    var c0: usize = 0;
    while (c0 < n) : (c0 += ldl_block) {
        for (0..ldl_block) |i| for (0..ldl_block) |j| {
            blk[i * ldl_block + j] = if (j <= i) chol[(c0 + i) * n + c0 + j] else 0;
        };
        if (!invertLowerBlock(&blk)) return error.HessianNotPositiveDefinite;
        for (0..ldl_block) |i| for (0..ldl_block) |j| {
            inv[i * ldl_block + j] = if (j <= i) @floatCast(blk[i * ldl_block + j]) else 0;
        };
        // Rows above the block's lower edge stay zero: chol is zero there and the diagonal block is cleared.
        for (c0 + ldl_block..n) |r| {
            var row: [ldl_block]f32 = undefined;
            for (0..ldl_block) |x| {
                var acc: f32 = 0;
                for (0..ldl_block) |m| acc += chol[r * n + c0 + m] * inv[m * ldl_block + x];
                row[x] = acc;
            }
            @memcpy(l[r * n + c0 ..][0..ldl_block], &row);
        }
        for (0..ldl_block) |i| @memset(l[(c0 + i) * n + c0 ..][0..ldl_block], 0);
    }
    return l;
}

pub fn gScaleBracket(codebook: format.Codebook) [2]f32 {
    return switch (codebook) {
        .mul1 => .{ 0.4, 1.2 },
        .mcg => .{ 0.5, 1.5 },
        .spec1, .pair1, .std1 => unreachable,
    };
}

fn gScaleSteps(low: f64, high: f64, resphi: f64) usize {
    var width = high - low;
    var steps: usize = 0;
    while (width > g_scale_tol and steps < g_scale_max_evals - 2) {
        width *= 1.0 - resphi;
        steps += 1;
    }
    return steps;
}

/// One global scale for a tensor: golden section over the codebook's bracket, scoring the squared
/// error of a wrapped diagonal of sample tiles searched at inner * g and read back at 1/g.
pub fn globalScale(allocator: std.mem.Allocator, enc: *encode.Encoder, codebook: format.Codebook, inner: []const f32, rows: usize, cols: usize) !f32 {
    try admitGeometry(rows, cols);
    if (inner.len != rows * cols) return error.ShapeMismatch;
    if (enc.rate.n % 2 != 0 or enc.rate.n < format.Rate.min_n or enc.rate.n > format.Rate.max_n) return error.RateUnsupported;
    if (enc.w < 12 or enc.w > 16) return error.WindowUnsupported;
    const tk = rows / 16;
    const tn = cols / 16;
    const pair_count = @max(tk, tn) * g_scale_width;
    const stride = @max(1, pair_count / g_scale_tiles);
    var tiles: std.ArrayList([encode.tile_len]f32) = .empty;
    defer tiles.deinit(allocator);
    var p: usize = 0;
    while (p < pair_count and tiles.items.len < g_scale_tiles) : (p += stride) {
        const i = p / g_scale_width;
        const wv = p % g_scale_width;
        const r0 = (i % tk) * 16;
        const c0 = ((i + wv) % tn) * 16;
        var tile: [encode.tile_len]f32 = undefined;
        for (0..16) |r| @memcpy(tile[r * 16 ..][0..16], inner[(r0 + r) * cols + c0 ..][0..16]);
        try tiles.append(allocator, tile);
    }
    const packed_scratch = try allocator.alloc(u16, enc.rate.n);
    defer allocator.free(packed_scratch);

    const score_ctx = struct {
        enc: *encode.Encoder,
        tiles: []const [encode.tile_len]f32,
        scratch: []u16,

        fn at(self: @This(), g: f32) !f64 {
            var sse: f64 = 0;
            var scaled: [encode.tile_len]f32 = undefined;
            var recon: [encode.tile_len]f32 = undefined;
            for (self.tiles) |tile| {
                for (&scaled, tile) |*o, v| o.* = v * g;
                _ = try self.enc.encodeTile(&scaled, self.scratch, &recon);
                for (recon, tile) |r, v| {
                    const d = r / g - v;
                    sse += @as(f64, d * d);
                }
            }
            return sse / @as(f64, @floatFromInt(self.tiles.len * encode.tile_len));
        }
    }{ .enc = enc, .tiles = tiles.items, .scratch = packed_scratch };

    const resphi64: f64 = 2.0 - (1.0 + @sqrt(5.0)) / 2.0;
    const resphi: f32 = @floatCast(resphi64);
    const bracket = gScaleBracket(codebook);
    var a: f32 = bracket[0];
    var b: f32 = bracket[1];
    var x1 = a + resphi * (b - a);
    var x2 = b - resphi * (b - a);
    var f1 = try score_ctx.at(x1);
    var f2 = try score_ctx.at(x2);
    for (0..gScaleSteps(bracket[0], bracket[1], resphi64)) |_| {
        const left = f1 < f2;
        const a_n = if (left) a else x1;
        const b_n = if (left) x2 else b;
        const probe = if (left) a_n + resphi * (b_n - a_n) else b_n - resphi * (b_n - a_n);
        const fp = try score_ctx.at(probe);
        a = a_n;
        b = b_n;
        if (left) {
            x2 = x1;
            f2 = f1;
            x1 = probe;
            f1 = fp;
        } else {
            x1 = x2;
            f1 = f2;
            x2 = probe;
            f2 = fp;
        }
    }
    return (a + b) / 2.0;
}

/// Reverse LDLQ: 16-row groups from the bottom, each searched after the error of the rows below
/// it is fed back through `l`. Fills `packed_out` in loader order and, if asked, `recon` with the
/// values the decoder will produce.
pub fn ldlq(allocator: std.mem.Allocator, enc: *encode.Encoder, inner: []const f32, l: []const f32, rows: usize, cols: usize, packed_out: []u16, recon: ?[]f32) !void {
    const n = enc.rate.n;
    const out_tiles = cols / 16;
    const rec = try allocator.alloc(f32, rows * cols);
    defer allocator.free(rec);
    @memset(rec, 0);
    const prod = try allocator.alloc(f32, rows * cols);
    defer allocator.free(prod);
    @memset(prod, 0);
    const comp = try allocator.alloc(f32, 16 * cols);
    defer allocator.free(comp);
    const err_row = try allocator.alloc(f32, cols);
    defer allocator.free(err_row);

    var j = rows;
    while (j > 0) {
        const i = j -| ldlq_buffer_rows;
        const chunk = j - i;
        var bj = chunk;
        while (bj > 0) : (bj -= ldlq_feedback_rows) {
            const bi = bj -| ldlq_feedback_rows;
            const height = bj - bi;
            @memcpy(comp[0 .. height * cols], prod[(i + bi) * cols ..][0 .. height * cols]);
            for (bj..chunk) |m| {
                for (err_row, inner[(i + m) * cols ..][0..cols], rec[(i + m) * cols ..][0..cols]) |*e, wv, rv| e.* = wv - rv;
                for (0..height) |r| {
                    const lv = l[(i + m) * rows + i + bi + r];
                    for (comp[r * cols ..][0..cols], err_row) |*c, e| c.* += lv * e;
                }
            }
            for (0..height) |r| for (comp[r * cols ..][0..cols], inner[(i + bi + r) * cols ..][0..cols]) |*c, wv| {
                c.* = wv + c.*;
            };
            const tk = (i + bi) / 16;
            for (0..out_tiles) |tn| {
                var tile: [encode.tile_len]f32 = undefined;
                for (0..16) |r| @memcpy(tile[r * 16 ..][0..16], comp[r * cols + tn * 16 ..][0..16]);
                var tile_rec: [encode.tile_len]f32 = undefined;
                _ = try enc.encodeTile(&tile, packed_out[(tk * out_tiles + tn) * n ..][0..n], &tile_rec);
                for (0..16) |r| @memcpy(rec[(i + bi + r) * cols + tn * 16 ..][0..16], tile_rec[r * 16 ..][0..16]);
            }
        }
        if (i > 0) {
            for (0..chunk) |m| {
                for (err_row, inner[(i + m) * cols ..][0..cols], rec[(i + m) * cols ..][0..cols]) |*e, wv, rv| e.* = wv - rv;
                for (0..i) |r| {
                    const lv = l[(i + m) * rows + r];
                    for (prod[r * cols ..][0..cols], err_row) |*c, e| c.* += lv * e;
                }
            }
        }
        j = i;
    }
    if (recon) |out| @memcpy(out, rec);
}

pub const Quantized = struct {
    packed_out: []u16,
    suh: []u16,
    svh: []u16,
    g: f32,

    pub fn deinit(self: *Quantized, allocator: std.mem.Allocator) void {
        allocator.free(self.packed_out);
        allocator.free(self.suh);
        allocator.free(self.svh);
    }
};

/// {-1, +1} vector for one tensor. The stream is splitmix64, seeded by mixing `seed` with every
/// byte of `name` (each byte is xored in and the state is mixed). Element i is -1 when bit 0 of
/// the i-th output is set, else +1. The same (seed, name, len) is byte-identical. The name must
/// include the expert index: two experts passed the same name share one sign vector. `quantizeNamed`
/// takes the first `in` values as `sign_in` and the next `out` as `sign_out`.
pub fn signs(allocator: std.mem.Allocator, seed: u64, name: []const u8, len: usize) ![]f32 {
    var state = seed;
    for (name) |b| state = splitmix(state ^ b);
    state = splitmix(state);
    const out = try allocator.alloc(f32, len);
    for (out) |*v| {
        state = splitmix(state);
        v.* = if (state & 1 == 0) 1 else -1;
    }
    return out;
}

fn splitmix(x: u64) u64 {
    var z = x +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// `quantizeTensor` with the sign vectors of `(seed, name)`.
pub fn quantizeNamed(allocator: std.mem.Allocator, w: []const f32, h: []const f32, in: usize, out: usize, seed: u64, name: []const u8, rate: format.Rate, dec: format.Decode, mode: OutScales) !Quantized {
    try admitGeometry(in, out);
    if (w.len != in * out or h.len != in * in) return error.ShapeMismatch;
    try admitRate(rate, dec.window);
    const both = try signs(allocator, seed, name, in + out);
    defer allocator.free(both);
    return quantizeTensor(allocator, w, h, in, out, both[0..in], both[in..], rate, dec, mode);
}

/// A public weight [in, out] and its input Hessian [in, in] to the three tensors of a pack:
/// `packed_out` [in/16, out/16, rate.n] u16, `suh` [in] and `svh` [out] as F16 bits.
pub fn quantizeTensor(allocator: std.mem.Allocator, w: []const f32, h: []const f32, in: usize, out: usize, sign_in: []const f32, sign_out: []const f32, rate: format.Rate, dec: format.Decode, mode: OutScales) !Quantized {
    try admitGeometry(in, out);
    if (w.len != in * out or h.len != in * in or sign_in.len != in or sign_out.len != out) return error.ShapeMismatch;
    try admitRate(rate, dec.window);
    var enc = try encode.Encoder.init(allocator, rate, dec);
    defer enc.deinit();
    const hdiag = try allocator.alloc(f32, in);
    defer allocator.free(hdiag);
    for (hdiag, 0..) |*d, i| d.* = h[i * in + i];
    var reg = try regularize(allocator, w, in, out, sign_in, sign_out, mode, hdiag);
    defer reg.deinit(allocator);
    const rotated = try rotatedHessian(allocator, h, reg.suh);
    defer allocator.free(rotated);
    const l = try ldlFactor(allocator, rotated, in);
    defer allocator.free(l);
    const g = try globalScale(allocator, &enc, dec.codebook, reg.inner, in, out);
    for (reg.inner) |*v| v.* *= g;
    const packed_out = try allocator.alloc(u16, in / 16 * (out / 16) * rate.n);
    errdefer allocator.free(packed_out);
    try ldlq(allocator, &enc, reg.inner, l, in, out, packed_out, null);
    const suh = try allocator.alloc(u16, in);
    errdefer allocator.free(suh);
    for (suh, reg.suh) |*o, v| o.* = format.f32ToF16Bits(v / g);
    const svh = try allocator.alloc(u16, out);
    for (svh, reg.svh) |*o, v| o.* = format.f32ToF16Bits(v);
    return .{ .packed_out = packed_out, .suh = suh, .svh = svh, .g = g };
}

const Fixture = struct {
    file: st.File,
    allocator: std.mem.Allocator,

    fn load(allocator: std.mem.Allocator) !Fixture {
        return .{ .file = try st.parse(allocator, st.tensor), .allocator = allocator };
    }

    fn deinit(self: *Fixture) void {
        self.file.deinit(self.allocator);
    }

    fn get(self: *const Fixture, comptime T: type, comptime fmt: []const u8, args: anytype) ![]const T {
        var key: [96]u8 = undefined;
        const view = self.file.tensors.get(try std.fmt.bufPrint(&key, fmt, args)) orelse return error.FixtureTensorMissing;
        return st.slice(T, view);
    }

    fn f32Of(self: *const Fixture, comptime T: type, comptime fmt: []const u8, args: anytype, rows: usize) ![]f32 {
        const raw = try self.get(T, fmt, args);
        const out = try self.allocator.alloc(f32, rows);
        for (out, raw[0..rows]) |*o, v| o.* = if (T == f16) @floatCast(v) else v;
        return out;
    }
};

const TensorConfig = struct { name: []const u8, dec: format.Decode, n: u32, rows: usize, mode: OutScales };
const tensor_configs = [_]TensorConfig{
    .{ .name = "mcg_n32_w12_r256_auto", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 32, .rows = 256, .mode = .auto },
    .{ .name = "mul1_n40_w12_r256_always", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 40, .rows = 256, .mode = .always },
    .{ .name = "mcg_n24_w12_r256_never", .dec = .{ .codebook = .mcg, .window = .w12 }, .n = 24, .rows = 256, .mode = .never },
    .{ .name = "mcg_n40_w15_r128_auto", .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 40, .rows = 128, .mode = .auto },
    .{ .name = "mul1_n64_w12_r128_always", .dec = .{ .codebook = .mul1, .window = .w12 }, .n = 64, .rows = 128, .mode = .always },
};
const fx_in = 256;
const fx_out = 128;

fn modeKey(mode: OutScales) []const u8 {
    return @tagName(mode);
}

fn expectCloseRel(want: []const f32, got: []const f32, rel: f32, floor: f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |a, b, i| {
        if (@abs(a - b) > rel * @abs(a) + floor) {
            std.debug.print("index {d}: want {d} got {d}\n", .{ i, a, b });
            return error.TestExpectedApproxEqual;
        }
    }
}

test "quantize: regularize reproduces the reference scales and inner matrix" {
    const t = std.testing;
    var fx = try Fixture.load(t.allocator);
    defer fx.deinit();
    const w = try fx.f32Of(f16, "W", .{}, fx_in * fx_out);
    defer t.allocator.free(w);
    const h = try fx.f32Of(f16, "H", .{}, fx_in * fx_in);
    defer t.allocator.free(h);
    const sign_out = try fx.get(f32, "sign_out", .{});
    const Case = struct { rows: usize, mode: OutScales };
    for ([_]Case{ .{ .rows = 256, .mode = .always }, .{ .rows = 256, .mode = .never }, .{ .rows = 256, .mode = .auto }, .{ .rows = 128, .mode = .auto }, .{ .rows = 128, .mode = .always } }) |c| {
        const sign_in = if (c.rows == 256) try fx.get(f32, "sign_in_r256", .{}) else try fx.get(f32, "sign_in_r128", .{});
        var hdiag: [fx_in]f32 = undefined;
        for (0..c.rows) |i| hdiag[i] = h[i * fx_in + i];
        var reg = try regularize(t.allocator, w[0 .. c.rows * fx_out], c.rows, fx_out, sign_in, sign_out, c.mode, hdiag[0..c.rows]);
        defer reg.deinit(t.allocator);
        try expectCloseRel(try fx.get(f32, "reg_r{d}_{s}.suh", .{ c.rows, modeKey(c.mode) }), reg.suh, 2e-6, 1e-9);
        try expectCloseRel(try fx.get(f32, "reg_r{d}_{s}.svh", .{ c.rows, modeKey(c.mode) }), reg.svh, 2e-6, 1e-9);
        if (c.rows == 256 and c.mode == .always) {
            const want = try fx.get(f16, "reg_r256_always.inner", .{});
            for (want, reg.inner, 0..) |a, b, i| {
                const aw: f32 = @floatCast(a);
                if (@abs(aw - b) > 1.5e-3 * @abs(aw) + 1e-5) {
                    std.debug.print("inner {d}: want {d} got {d}\n", .{ i, aw, b });
                    return error.TestExpectedApproxEqual;
                }
            }
        }
    }
}

test "quantize: the decoder's transform chain turns the regularized inner matrix back into the weight" {
    const t = std.testing;
    var fx = try Fixture.load(t.allocator);
    defer fx.deinit();
    const w = try fx.f32Of(f16, "W", .{}, fx_in * fx_out);
    defer t.allocator.free(w);
    var reg = try regularize(t.allocator, w, fx_in, fx_out, try fx.get(f32, "sign_in_r256", .{}), try fx.get(f32, "sign_out", .{}), .always, null);
    defer reg.deinit(t.allocator);
    const pub_w = try t.allocator.dupe(f32, reg.inner);
    defer t.allocator.free(pub_w);
    var vec: [had_dim]f32 = undefined;
    for (0..fx_in / had_dim) |b| for (0..fx_out) |c| {
        for (0..had_dim) |r| vec[r] = pub_w[(b * had_dim + r) * fx_out + c];
        format.hadamard128(&vec);
        for (0..had_dim) |r| pub_w[(b * had_dim + r) * fx_out + c] = vec[r];
    };
    for (0..fx_in) |r| for (pub_w[r * fx_out ..][0..fx_out]) |*v| {
        v.* *= reg.suh[r];
    };
    for (0..fx_in) |r| for (0..fx_out / had_dim) |b| {
        @memcpy(&vec, pub_w[r * fx_out + b * had_dim ..][0..had_dim]);
        format.hadamard128(&vec);
        @memcpy(pub_w[r * fx_out + b * had_dim ..][0..had_dim], &vec);
    };
    for (0..fx_in) |r| for (0..fx_out) |c| {
        pub_w[r * fx_out + c] *= reg.svh[c];
    };
    try expectCloseRel(w, pub_w, 1e-3, 1e-6);
}

test "quantize: the block LDL factor of a damped rotated Hessian carries H10 = L10 * H00" {
    const t = std.testing;
    const n = 32;
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    var a: [n * n]f32 = undefined;
    for (&a) |*v| v.* = r.floatNorm(f32);
    var h: [n * n]f32 = undefined;
    for (0..n) |i| for (0..n) |j| {
        var s: f32 = 0;
        for (0..n) |m| s += a[i * n + m] * a[j * n + m];
        h[i * n + j] = s / n;
    };
    const l = try ldlFactor(t.allocator, &h, n);
    defer t.allocator.free(l);
    var mean: f32 = 0;
    for (0..n) |i| mean += h[i * n + i];
    mean /= n;
    for (0..n) |i| for (0..n) |j| {
        if (i / 16 == j / 16 or j > i) try t.expectEqual(@as(f32, 0), l[i * n + j]);
    };
    for (16..n) |i| for (0..16) |j| {
        var s: f32 = 0;
        for (0..16) |m| {
            const h00 = h[m * n + j] + if (m == j) sigma_reg * mean else 0;
            s += l[i * n + m] * h00;
        }
        try t.expectApproxEqAbs(h[i * n + j], s, 2e-4);
    };
}

test "quantize: the whole tensor reproduces the reference trellis, scales and global scale in its bracket" {
    const t = std.testing;
    var fx = try Fixture.load(t.allocator);
    defer fx.deinit();
    const w = try fx.f32Of(f16, "W", .{}, fx_in * fx_out);
    defer t.allocator.free(w);
    const h_full = try fx.f32Of(f16, "H", .{}, fx_in * fx_in);
    defer t.allocator.free(h_full);
    for (tensor_configs) |c| {
        const sign_in = if (c.rows == 256) try fx.get(f32, "sign_in_r256", .{}) else try fx.get(f32, "sign_in_r128", .{});
        const h = try t.allocator.alloc(f32, c.rows * c.rows);
        defer t.allocator.free(h);
        for (0..c.rows) |i| @memcpy(h[i * c.rows ..][0..c.rows], h_full[i * fx_in ..][0..c.rows]);
        var q = try quantizeTensor(t.allocator, w[0 .. c.rows * fx_out], h, c.rows, fx_out, sign_in, try fx.get(f32, "sign_out", .{}), .{ .n = c.n }, c.dec, c.mode);
        defer q.deinit(t.allocator);
        const want = try fx.get(u16, "{s}.packed", .{c.name});
        try t.expectEqual(want.len, q.packed_out.len);
        try t.expectEqualSlices(u16, want, q.packed_out);
        const bracket = gScaleBracket(c.dec.codebook);
        try t.expect(q.g > bracket[0] and q.g < bracket[1]);
        try t.expectApproxEqAbs((try fx.get(f32, "{s}.g", .{c.name}))[0], q.g, 1e-5);
        try t.expectEqualSlices(u16, try fx.get(u16, "{s}.suh", .{c.name}), q.suh);
        try t.expectEqualSlices(u16, try fx.get(u16, "{s}.svh", .{c.name}), q.svh);
    }
}

test "quantize: sign vectors are a pure function of seed and tensor name" {
    const t = std.testing;
    const a = try signs(t.allocator, 7, "language_model.model.layers.0.mlp.switch_mlp.gate_proj", 128);
    defer t.allocator.free(a);
    const b = try signs(t.allocator, 7, "language_model.model.layers.0.mlp.switch_mlp.gate_proj", 128);
    defer t.allocator.free(b);
    const other = try signs(t.allocator, 7, "language_model.model.layers.0.mlp.switch_mlp.up_proj", 128);
    defer t.allocator.free(other);
    const reseeded = try signs(t.allocator, 8, "language_model.model.layers.0.mlp.switch_mlp.gate_proj", 128);
    defer t.allocator.free(reseeded);
    try t.expectEqualSlices(f32, a, b);
    try t.expect(a.len == 128);
    for (a) |v| try t.expect(v == 1 or v == -1);
    try t.expect(!std.mem.eql(f32, a, other));
    try t.expect(!std.mem.eql(f32, a, reseeded));
}

test "quantize: the same seed and tensor name quantize to the same bytes" {
    const t = std.testing;
    const in: usize = 128;
    const out: usize = 128;
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    const w = try t.allocator.alloc(f32, in * out);
    defer t.allocator.free(w);
    const h = try t.allocator.alloc(f32, in * in);
    defer t.allocator.free(h);
    for (w) |*v| v.* = r.floatNorm(f32) * 0.02;
    for (0..in) |i| {
        for (0..in) |j| {
            var s: f32 = 0;
            for (0..8) |m| s += w[i * out + m] * w[j * out + m];
            h[i * in + j] = s;
        }
        h[i * in + i] += 0.1;
    }
    const name = "language_model.model.layers.3.mlp.switch_mlp.down_proj";
    var first = try quantizeNamed(t.allocator, w, h, in, out, 11, name, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w12 }, .never);
    defer first.deinit(t.allocator);
    var second = try quantizeNamed(t.allocator, w, h, in, out, 11, name, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w12 }, .never);
    defer second.deinit(t.allocator);
    try t.expectEqualSlices(u16, first.packed_out, second.packed_out);
    try t.expectEqualSlices(u16, first.suh, second.suh);
    try t.expectEqualSlices(u16, first.svh, second.svh);
    try t.expectEqual(first.g, second.g);
}

test "quantize: in, out, rate, window and slice length are refused before any buffer is walked" {
    const t = std.testing;
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    const w = try t.allocator.alloc(f32, 128 * 128);
    defer t.allocator.free(w);
    const wide = try t.allocator.alloc(f32, 192 * 128);
    defer t.allocator.free(wide);
    const h192 = try t.allocator.alloc(f32, 192 * 192);
    defer t.allocator.free(h192);
    const h128 = try t.allocator.alloc(f32, 128 * 128);
    defer t.allocator.free(h128);
    const si = try t.allocator.alloc(f32, 192);
    defer t.allocator.free(si);
    const so = try t.allocator.alloc(f32, 128);
    defer t.allocator.free(so);
    @memset(si, 1);
    @memset(so, 1);
    try t.expectError(error.GeometryUnsupported, quantizeTensor(t.allocator, wide, h192, 192, 128, si, so, .{ .n = 32 }, dec, .never));
    try t.expectError(error.EmptyTensor, quantizeTensor(t.allocator, w[0..0], h128[0..0], 0, 0, si[0..0], so[0..0], .{ .n = 32 }, dec, .never));
    try t.expectError(error.ShapeMismatch, quantizeTensor(t.allocator, w[0..10], h128, 128, 128, si[0..128], so, .{ .n = 32 }, dec, .never));
    try t.expectError(error.RateUnsupported, quantizeTensor(t.allocator, w, h128, 128, 128, si[0..128], so, .{ .n = 15 }, dec, .never));
    try t.expectError(error.RateUnsupported, quantizeTensor(t.allocator, w, h128, 128, 128, si[0..128], so, .{ .n = 130 }, dec, .never));
    try t.expectError(error.WindowUnsupported, quantizeTensor(t.allocator, w, h128, 128, 128, si[0..128], so, .{ .n = 32 }, .{ .codebook = .mcg, .window = .w11 }, .never));
    try t.expectError(error.WindowUnsupported, quantizeNamed(t.allocator, w, h128, 128, 128, 1, "gate", .{ .n = 32 }, .{ .codebook = .mcg, .window = .w8 }, .never));
    try t.expectError(error.GeometryUnsupported, ldlFactor(t.allocator, h128[0 .. 20 * 20], 20));
    var enc = try encode.Encoder.init(t.allocator, .{ .n = 32 }, dec);
    defer enc.deinit();
    try t.expectError(error.GeometryUnsupported, globalScale(t.allocator, &enc, .mcg, wide, 192, 128));
    try t.expectError(error.ShapeMismatch, globalScale(t.allocator, &enc, .mcg, w[0..10], 128, 128));
    try t.expectError(error.EmptyTensor, globalScale(t.allocator, &enc, .mcg, w[0..0], 0, 0));
}

test "quantize: the lower Cholesky factor reproduces a symmetric positive definite matrix and names a failing pivot" {
    const t = std.testing;
    const n = 200;
    var prng = std.Random.DefaultPrng.init(31);
    const r = prng.random();
    const x = try t.allocator.alloc(f64, n * 96);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f64);
    const a = try t.allocator.alloc(f64, n * n);
    defer t.allocator.free(a);
    for (0..n) |i| for (0..n) |j| {
        var s: f64 = if (i == j) 0.5 else 0;
        for (0..96) |m| s += x[i * 96 + m] * x[j * 96 + m];
        a[i * n + j] = s;
    };
    const l = try t.allocator.dupe(f64, a);
    defer t.allocator.free(l);
    try t.expectEqual(@as(usize, 0), choleskyLower(l, n));
    for (0..n) |i| for (0..n) |j| {
        var s: f64 = 0;
        for (0..@min(i, j) + 1) |m| s += l[i * n + m] * l[j * n + m];
        try t.expectApproxEqAbs(a[i * n + j], s, 1e-9);
    };

    const bad = try t.allocator.dupe(f64, a);
    defer t.allocator.free(bad);
    bad[70 * n + 70] = -1;
    try t.expectEqual(@as(usize, 71), choleskyLower(bad, n));
    bad[70 * n + 70] = std.math.nan(f64);
    try t.expect(choleskyLower(bad, n) != 0);
}

test "quantize: the tiled trailing update equals the element-at-a-time Cholesky bit for bit" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(33);
    const r = prng.random();
    for ([_]usize{ 130, 256 }) |n| {
        const x = try t.allocator.alloc(f64, n * 40);
        defer t.allocator.free(x);
        for (x) |*v| v.* = r.floatNorm(f64);
        const a = try t.allocator.alloc(f64, n * n);
        defer t.allocator.free(a);
        for (0..n) |i| for (0..n) |j| {
            var s: f64 = if (i == j) 0.3 else 0;
            for (0..40) |m| s += x[i * 40 + m] * x[j * 40 + m];
            a[i * n + j] = s;
        };
        const want = try t.allocator.dupe(f64, a);
        defer t.allocator.free(want);
        var k: usize = 0;
        while (k < n) : (k += chol_block) {
            const kb = @min(chol_block, n - k);
            for (0..kb) |j| {
                const jj = k + j;
                const d = want[jj * n + jj] - dotRows(want[jj * n + k ..], want[jj * n + k ..], j);
                const root = @sqrt(d);
                want[jj * n + jj] = root;
                for (j + 1..kb) |i| want[(k + i) * n + jj] = (want[(k + i) * n + jj] - dotRows(want[(k + i) * n + k ..], want[jj * n + k ..], j)) / root;
            }
            for (k + kb..n) |i| for (0..kb) |j| {
                want[i * n + k + j] = (want[i * n + k + j] - dotRows(want[i * n + k ..], want[(k + j) * n + k ..], j)) / want[(k + j) * n + k + j];
            };
            for (k + kb..n) |i| for (k + kb..i + 1) |j| {
                want[i * n + j] -= dotRows(want[i * n + k ..][0..kb], want[j * n + k ..][0..kb], kb);
            };
        }
        try t.expectEqual(@as(usize, 0), choleskyLower(a, n));
        for (want, a) |w, g| try t.expectEqual(@as(u64, @bitCast(w)), @as(u64, @bitCast(g)));
    }
}

test "quantize: the row-butterfly Hadamard equals the column-at-a-time transform bit for bit" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(34);
    const r = prng.random();
    const rows = 256;
    const cols = 40;
    inline for (.{ f32, f64 }) |T| {
        const m = try t.allocator.alloc(T, rows * cols);
        defer t.allocator.free(m);
        for (m) |*v| v.* = r.floatNorm(T);
        const want = try t.allocator.dupe(T, m);
        defer t.allocator.free(want);
        var v: [had_dim]T = undefined;
        var b: usize = 0;
        while (b < rows) : (b += had_dim) for (0..cols) |c| {
            for (0..had_dim) |k| v[k] = want[(b + k) * cols + c];
            fwht(T, &v);
            for (0..had_dim) |k| want[(b + k) * cols + c] = v[k];
        };
        hadamardRows(T, m, rows, cols);
        try t.expectEqualSlices(T, want, m);
    }
}

test "quantize: the lower-triangular block inverse multiplies back to the identity" {
    const t = std.testing;
    var prng = std.Random.DefaultPrng.init(32);
    const r = prng.random();
    var blk: [ldl_block * ldl_block]f64 = undefined;
    for (0..ldl_block) |i| for (0..ldl_block) |j| {
        blk[i * ldl_block + j] = if (j < i) r.floatNorm(f64) * 0.3 else if (j == i) 1.0 + r.float(f64) else 0;
    };
    var inv = blk;
    try t.expect(invertLowerBlock(&inv));
    for (0..ldl_block) |i| for (0..ldl_block) |j| {
        var s: f64 = 0;
        for (0..ldl_block) |m| s += blk[i * ldl_block + m] * inv[m * ldl_block + j];
        try t.expectApproxEqAbs(if (i == j) @as(f64, 1) else 0, s, 1e-12);
    };
    inv = blk;
    inv[5 * ldl_block + 5] = 0;
    try t.expect(!invertLowerBlock(&inv));
}

test "quantize: factor throughput (SUSHI_ENCODER_FACTOR_UBENCH)" {
    if (std.c.getenv("SUSHI_ENCODER_FACTOR_UBENCH") == null) return error.SkipZigTest;
    const t = std.testing;
    const n = 2560;
    var prng = std.Random.DefaultPrng.init(2560);
    const r = prng.random();
    const x = try t.allocator.alloc(f32, n * 64);
    defer t.allocator.free(x);
    for (x) |*v| v.* = r.floatNorm(f32);
    const h = try t.allocator.alloc(f32, n * n);
    defer t.allocator.free(h);
    for (0..n) |i| for (0..n) |j| {
        var acc: f32 = 0;
        for (0..64) |m| acc += x[i * 64 + m] * x[j * 64 + m];
        h[i * n + j] = acc;
    };
    const suh = try t.allocator.alloc(f32, n);
    defer t.allocator.free(suh);
    for (suh) |*v| v.* = 0.5 + r.float(f32);
    for (0..3) |_| {
        const t0 = @import("stages.zig").nowNs();
        const rot = try rotatedHessian(t.allocator, h, suh);
        defer t.allocator.free(rot);
        const t1 = @import("stages.zig").nowNs();
        const a = try t.allocator.alloc(f64, n * n);
        defer t.allocator.free(a);
        for (a, rot) |*o, v| o.* = v;
        for (0..n) |i| a[i * n + i] += 1;
        const t2 = @import("stages.zig").nowNs();
        _ = choleskyLower(a, n);
        const t3 = @import("stages.zig").nowNs();
        const l = try ldlFactor(t.allocator, rot, n);
        defer t.allocator.free(l);
        const t4 = @import("stages.zig").nowNs();
        std.debug.print("rotate {d:.0} ms, cholesky {d:.0} ms, ldlFactor {d:.0} ms\n", .{ @as(f64, @floatFromInt(t1 - t0)) / 1e6, @as(f64, @floatFromInt(t3 - t2)) / 1e6, @as(f64, @floatFromInt(t4 - t3)) / 1e6 });
    }
}
