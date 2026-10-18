//! `--factor-precision 32`: the block-LDL factor of `quantize.ldlFactor` computed in float32 on the GPU for
//! a bank of Hessians. The float64 CPU factor stays the default and the oracle.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const quantize = @import("quantize.zig");
const stages_mod = @import("stages.zig");

pub const Error = error{
    ShapeMismatch,
    GeometryUnsupported,
    HessianNotPositiveDefinite,
    MetalKernelCompileFailed,
    MetalKernelBadOutputCount,
    MlxError,
    OutOfMemory,
};

/// 64 factors on the CPU in float64 (the default and the oracle); 32 factors on the GPU in float32.
pub const Precision = enum { f64, f32 };

pub const Stats = struct {
    retries: usize = 0,
    fallbacks: usize = 0,
    ns: u64 = 0,
};

pub const Options = struct {
    gpu_retries: usize = quantize.max_cholesky_retries,
    chunk_elems: usize = 40 * 2560 * 2560,
};

const block = 16;
const threads = 1024;
const k_chunk = 256;

/// Float32 helpers of the kernel. The Cholesky factor of a 16x16 tile and its inverse run on one thread; a
/// pivot that is not positive clears the matrix's flag and is replaced by 1 so the later steps stay finite.
const kernel_header =
    \\inline bool chol_inverse16(threadgroup float* t, threadgroup float* inv) {
    \\    bool ok = true;
    \\    for (int j = 0; j < 16; ++j) {
    \\        float d = t[j * 16 + j];
    \\        for (int k = 0; k < j; ++k) d -= t[j * 16 + k] * t[j * 16 + k];
    \\        if (!(d > 0.0f)) { ok = false; d = 1.0f; }
    \\        float r = metal::precise::sqrt(d);
    \\        t[j * 16 + j] = r;
    \\        for (int i = j + 1; i < 16; ++i) {
    \\            float v = t[i * 16 + j];
    \\            for (int k = 0; k < j; ++k) v -= t[i * 16 + k] * t[j * 16 + k];
    \\            t[i * 16 + j] = metal::precise::divide(v, r);
    \\        }
    \\    }
    \\    for (int i = 0; i < 16; ++i) {
    \\        for (int j = 0; j < 16; ++j) inv[i * 16 + j] = 0.0f;
    \\    }
    \\    for (int col = 0; col < 16; ++col) {
    \\        inv[col * 16 + col] = metal::precise::divide(1.0f, t[col * 16 + col]);
    \\        for (int row = col + 1; row < 16; ++row) {
    \\            float sum = 0.0f;
    \\            for (int m = col; m < row; ++m) sum += t[row * 16 + m] * inv[m * 16 + col];
    \\            inv[row * 16 + col] = metal::precise::divide(-sum, t[row * 16 + row]);
    \\        }
    \\    }
    \\    return ok;
    \\}
    \\
    \\inline void finish_row(const device float* a_row, thread const float* acc, threadgroup const float* inv, device float* w_row, device float* l_row) {
    \\    float col[16];
    \\    float w[16];
    \\    for (int c = 0; c < 16; ++c) col[c] = a_row[c] - acc[c];
    \\    for (int i = 0; i < 16; ++i) {
    \\        float sum = 0.0f;
    \\        for (int c = 0; c <= i; ++c) sum += col[c] * inv[i * 16 + c];
    \\        w[i] = sum;
    \\        w_row[i] = sum;
    \\    }
    \\    for (int i = 0; i < 16; ++i) {
    \\        float sum = 0.0f;
    \\        for (int c = i; c < 16; ++c) sum += w[c] * inv[c * 16 + i];
    \\        l_row[i] = sum;
    \\    }
    \\}
;

/// One threadgroup per matrix runs the whole left-looking block factorization: a step takes 16 columns, updates
/// them from the earlier Cholesky columns `w`, factors the 16x16 diagonal tile (c), and writes W = A21 c^-T into
/// `w` and the output block column L21 = W c^-1 into `l`. MLX's float32 matmul runs below float32 accuracy on
/// this GPU, so the products are plain FMA loops over threadgroup-staged chunks.
const kernel_body =
    \\uint tid = thread_position_in_threadgroup.x;
    \\size_t mat = threadgroup_position_in_grid.x;
    \\const device float* ab = a + mat * N * N;
    \\device float* wb = w + mat * N * N;
    \\device float* lb = l + mat * N * N;
    \\threadgroup float4 top[16 * (KC / 4)];
    \\threadgroup float tile[256];
    \\threadgroup float inv[256];
    \\threadgroup uint good_tg;
    \\if (tid == 0) good_tg = 1;
    \\const int rr = tid >> 4;
    \\const int cc = tid & 15;
    \\for (int j = 0; j < N / 16; ++j) {
    \\    const int c0 = j * 16;
    \\    const int rows_below = N - c0 - 16;
    \\    for (int idx = tid; idx < (c0 + 16) * 16; idx += T) lb[(size_t)(idx >> 4) * N + c0 + (idx & 15)] = 0.0f;
    \\    const int waves = rows_below > 0 ? (rows_below + 2 * T - 1) / (2 * T) : 1;
    \\    for (int wave = 0; wave < waves; ++wave) {
    \\        const int r_a = c0 + 16 + wave * 2 * T + (int)tid;
    \\        const int r_b = r_a + T;
    \\        const int s_a = min(r_a, N - 1);
    \\        const int s_b = min(r_b, N - 1);
    \\        float acc_a[16];
    \\        float acc_b[16];
    \\        for (int c = 0; c < 16; ++c) { acc_a[c] = 0.0f; acc_b[c] = 0.0f; }
    \\        float dacc = 0.0f;
    \\        for (int k0 = 0; k0 < c0; k0 += KC) {
    \\            const int kc = min(KC, c0 - k0);
    \\            const int kv = kc / 4;
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            for (int idx = tid; idx < 16 * kv; idx += T) {
    \\                const int row = idx / kv;
    \\                const int col4 = idx - row * kv;
    \\                top[row * (KC / 4) + col4] = ((const device float4*)(wb + (size_t)(c0 + row) * N + k0))[col4];
    \\            }
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            const device float4* pa = (const device float4*)(wb + (size_t)s_a * N + k0);
    \\            const device float4* pb = (const device float4*)(wb + (size_t)s_b * N + k0);
    \\            for (int kk = 0; kk < kv; ++kk) {
    \\                const float4 va = pa[kk];
    \\                const float4 vb = pb[kk];
    \\                for (int c = 0; c < 16; ++c) {
    \\                    const float4 t = top[c * (KC / 4) + kk];
    \\                    acc_a[c] += dot(va, t);
    \\                    acc_b[c] += dot(vb, t);
    \\                }
    \\                if (wave == 0 && tid < 256) dacc += dot(top[rr * (KC / 4) + kk], top[cc * (KC / 4) + kk]);
    \\            }
    \\        }
    \\        if (wave == 0) {
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            if (tid < 256) tile[tid] = ab[(size_t)(c0 + rr) * N + c0 + cc] - dacc;
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\            if (tid == 0 && !chol_inverse16(tile, inv)) good_tg = 0;
    \\            threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        }
    \\        if (r_a < N) finish_row(ab + (size_t)r_a * N + c0, acc_a, inv, wb + (size_t)r_a * N + c0, lb + (size_t)r_a * N + c0);
    \\        if (r_b < N) finish_row(ab + (size_t)r_b * N + c0, acc_b, inv, wb + (size_t)r_b * N + c0, lb + (size_t)r_b * N + c0);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    \\}
    \\if (tid == 0) good_out[mat] = good_tg;
;

const max_kernels = 4;
var kernel_ns: [max_kernels]usize = @splat(0);
var kernels: [max_kernels]mlx.mlx_fast_metal_kernel = undefined;
var kernel_count: usize = 0;

fn getKernel(allocator: std.mem.Allocator, n: usize) Error!mlx.mlx_fast_metal_kernel {
    for (kernel_ns[0..kernel_count], kernels[0..kernel_count]) |kn, k| if (kn == n) return k;
    const body = try std.fmt.allocPrintSentinel(allocator, "const int N = {d};\nconst int KC = {d};\nconst int T = {d};\n{s}", .{ n, k_chunk, threads, kernel_body }, 0);
    defer allocator.free(body);
    var name_buf: [64]u8 = undefined;
    const name = std.fmt.bufPrintSentinel(&name_buf, "sushi_encoder_ldl_factor_n{d}", .{n}, 0) catch unreachable;
    const ins = [_][*:0]const u8{"a"};
    const outs = [_][*:0]const u8{ "l", "w", "good_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, body.ptr, kernel_header, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    if (kernel_count == max_kernels) {
        _ = mlx.mlx_fast_metal_kernel_free(kernels[0]);
        kernel_ns[0] = n;
        kernels[0] = kernel;
        return kernel;
    }
    kernel_ns[kernel_count] = n;
    kernels[kernel_count] = kernel;
    kernel_count += 1;
    return kernel;
}

const Arr = mlx.mlx_array;

fn drop(a: Arr) void {
    _ = mlx.mlx_array_free(a);
}

const Chunk = struct {
    l: Arr,
    good: Arr,
    sum: Arr,

    fn deinit(self: *Chunk) void {
        drop(self.l);
        drop(self.good);
        drop(self.sum);
    }
};

fn factorChunk(allocator: std.mem.Allocator, s: mlx.mlx_stream, host: []const f32, count: usize, n: usize) Error!Chunk {
    const kernel = try getKernel(allocator, n);
    const b: c_int = @intCast(count);
    const nn: c_int = @intCast(n);
    const a = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ b, nn, nn }, 3, .float32);
    defer drop(a);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, nn, nn }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ b, nn, nn }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{b}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, b * threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    const inputs = [_]Arr{a};
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs) != 3) return error.MetalKernelBadOutputCount;
    var l = mlx.mlx_array_new();
    errdefer drop(l);
    var good = mlx.mlx_array_new();
    errdefer drop(good);
    try mlx.check(mlx.mlx_vector_array_get(&l, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&good, outputs, 2));
    var rows = mlx.mlx_array_new();
    defer drop(rows);
    try mlx.check(mlx.mlx_sum_axis(&rows, l, 2, false, s));
    var total = mlx.mlx_array_new();
    errdefer drop(total);
    try mlx.check(mlx.mlx_sum_axis(&total, rows, 1, false, s));
    const evals = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_vector_array_append_value(evals, l));
    try mlx.check(mlx.mlx_vector_array_append_value(evals, good));
    try mlx.check(mlx.mlx_vector_array_append_value(evals, total));
    try mlx.check(mlx.mlx_eval(evals));
    return .{ .l = l, .good = good, .sum = total };
}

/// Factors every Hessian of `damped` (each `n` x `n`, already through `quantize.dampDiagonal`, whose mean is in
/// `means`) into `ls`, which the caller frees. A matrix whose float32 pivot fails is retried with the CPU
/// path's diagonal bump, then falls back to the float64 factor.
pub fn factorBatch(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    damped: []const []const f32,
    means: []const f64,
    n: usize,
    opts: Options,
    stats: *Stats,
    ls: [][]f32,
) Error!void {
    if (n == 0 or n % block != 0) return error.GeometryUnsupported;
    if (damped.len != means.len or damped.len != ls.len) return error.ShapeMismatch;
    for (damped) |h| if (h.len != n * n) return error.ShapeMismatch;
    for (ls) |*l| l.* = &.{};
    errdefer for (ls) |*l| {
        allocator.free(l.*);
        l.* = &.{};
    };
    const t0 = stages_mod.nowNs();
    defer stats.ns += stages_mod.nowNs() - t0;

    const per_chunk = @max(1, opts.chunk_elems / (n * n));
    const staging = try allocator.alloc(f32, @min(damped.len, per_chunk) * n * n);
    defer allocator.free(staging);
    const pending = try allocator.alloc(usize, damped.len);
    defer allocator.free(pending);
    for (pending, 0..) |*p, i| p.* = i;
    var live: usize = pending.len;
    var attempt: usize = 0;
    while (live > 0) : (attempt += 1) {
        var failed: usize = 0;
        var start: usize = 0;
        while (start < live) : (start += per_chunk) {
            const batch = pending[start..@min(live, start + per_chunk)];
            for (batch, 0..) |idx, slot| {
                const dst = staging[slot * n * n ..][0 .. n * n];
                @memcpy(dst, damped[idx]);
                const bump = quantize.retryBump(means[idx]);
                for (0..n) |i| for (0..attempt) |_| {
                    dst[i * n + i] += bump;
                };
            }
            var chunk = try factorChunk(allocator, s, staging[0 .. batch.len * n * n], batch.len, n);
            defer chunk.deinit();
            const good = mlx.mlx_array_data_uint32(chunk.good).?;
            const sums = mlx.mlx_array_data_float32(chunk.sum).?;
            const l_ptr = mlx.mlx_array_data_float32(chunk.l).?;
            for (batch, 0..) |idx, slot| {
                if (good[slot] == 0 or !std.math.isFinite(sums[slot])) {
                    pending[failed] = idx;
                    failed += 1;
                    continue;
                }
                ls[idx] = try allocator.dupe(f32, l_ptr[slot * n * n ..][0 .. n * n]);
            }
        }
        live = 0;
        for (pending[0..failed]) |idx| {
            if (attempt < opts.gpu_retries and means[idx] > 0) {
                pending[live] = idx;
                live += 1;
                stats.retries += 1;
                continue;
            }
            const copy = try allocator.dupe(f32, damped[idx]);
            defer allocator.free(copy);
            ls[idx] = quantize.ldlFactorDamped(allocator, copy, n, means[idx]) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.HessianNotPositiveDefinite,
            };
            stats.fallbacks += 1;
        }
    }
}

/// A low-rank Hessian with log-normal channel scales; `spike` adds a rank-one outlier of that many times
/// mean(diag) * n eigenvalue, the shape of a massive-activation channel.
fn spdHessian(allocator: std.mem.Allocator, r: std.Random, n: usize, rank: usize, spike: f32) ![]f32 {
    const g = try allocator.alloc(f32, n * rank);
    defer allocator.free(g);
    for (g) |*v| v.* = r.floatNorm(f32);
    const scale = try allocator.alloc(f32, n);
    defer allocator.free(scale);
    for (scale) |*v| v.* = @exp(0.5 * r.floatNorm(f32));
    const h = try allocator.alloc(f32, n * n);
    for (0..n) |i| for (0..i + 1) |j| {
        var acc: f32 = 0;
        for (g[i * rank ..][0..rank], g[j * rank ..][0..rank]) |x, y| acc += x * y;
        const v = acc * scale[i] * scale[j];
        h[i * n + j] = v;
        h[j * n + i] = v;
    };
    if (spike > 0) {
        const dir = try allocator.alloc(f32, n);
        defer allocator.free(dir);
        for (dir) |*v| v.* = r.floatNorm(f32);
        var mean: f32 = 0;
        for (0..n) |i| mean += h[i * n + i];
        mean /= @floatFromInt(n);
        for (0..n) |i| for (0..n) |j| {
            h[i * n + j] += spike * mean * dir[i] * dir[j];
        };
    }
    return h;
}

// Measured on this test's spectra (log-normal channel scales, a rank-one outlier of 0.5 * n * mean(diag)):
// 4e-6 / 8e-5 relative to the float64 factor at n = 256 / 2560 and 2e-7 for the reconstruction. The bounds keep
// a factor of five.
const bound_frob_small = 2e-5;
const bound_max_small = 3e-5;
const bound_frob_large = 4e-4;
const bound_max_large = 3e-4;
const bound_recon = 1e-6;

const Compared = struct { max_abs: f64, max_ref: f64, frob_rel: f64 };

fn compare(got: []const f32, want: []const f32) Compared {
    var max_abs: f64 = 0;
    var max_ref: f64 = 0;
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, want) |a, b| {
        const d = @as(f64, a) - @as(f64, b);
        max_abs = @max(max_abs, @abs(d));
        max_ref = @max(max_ref, @abs(@as(f64, b)));
        num += d * d;
        den += @as(f64, b) * @as(f64, b);
    }
    return .{ .max_abs = max_abs, .max_ref = max_ref, .frob_rel = @sqrt(num / den) };
}

const Bank = struct {
    allocator: std.mem.Allocator,
    raw: [][]f32,
    damped: [][]f32,
    means: []f64,
    ls: [][]f32,

    fn init(allocator: std.mem.Allocator, seed: u64, count: usize, n: usize, rank: usize, spike: f32) !Bank {
        var prng = std.Random.DefaultPrng.init(seed);
        var self = Bank{
            .allocator = allocator,
            .raw = try allocator.alloc([]f32, count),
            .damped = try allocator.alloc([]f32, count),
            .means = try allocator.alloc(f64, count),
            .ls = try allocator.alloc([]f32, count),
        };
        for (0..count) |b| {
            self.raw[b] = try spdHessian(allocator, prng.random(), n, rank, spike);
            self.damped[b] = try allocator.dupe(f32, self.raw[b]);
            self.means[b] = quantize.dampDiagonal(self.damped[b], n);
            self.ls[b] = &.{};
        }
        return self;
    }

    fn deinit(self: *Bank) void {
        for (self.raw, self.damped, self.ls) |r, d, l| {
            self.allocator.free(r);
            self.allocator.free(d);
            self.allocator.free(l);
        }
        self.allocator.free(self.raw);
        self.allocator.free(self.damped);
        self.allocator.free(self.means);
        self.allocator.free(self.ls);
    }

    fn view(self: *const Bank) []const []const f32 {
        return @ptrCast(self.damped);
    }
};

test "gpu_factor: float32 matches the float64 factor on a bank of SPD Hessians" {
    const t = std.testing;
    const n = 256;
    var bank = try Bank.init(t.allocator, 11, 3, n, 24, 0);
    defer bank.deinit();
    var stats = Stats{};
    try factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{}, &stats, bank.ls);
    try t.expectEqual(@as(usize, 0), stats.fallbacks);
    for (bank.raw, bank.ls) |raw, got| {
        const want = try quantize.ldlFactor(t.allocator, raw, n);
        defer t.allocator.free(want);
        try t.expectEqual(want.len, got.len);
        const c = compare(got, want);
        try t.expect(c.frob_rel < bound_frob_small);
        try t.expect(c.max_abs < bound_max_small * c.max_ref);
    }
}

/// max |H - (I+L) D (I+L)^T| / max |H|, D the block diagonal that makes the factorization exact on the diagonal
/// blocks: D = T^-1 H T^-T read at the 16x16 diagonal blocks, in float64.
fn reconstructionResidual(allocator: std.mem.Allocator, h: []const f32, l: []const f32, n: usize) !f64 {
    const t_mat = try allocator.alloc(f64, n * n);
    defer allocator.free(t_mat);
    for (0..n) |r| for (0..n) |c| {
        t_mat[r * n + c] = @as(f64, l[r * n + c]) + if (r == c) @as(f64, 1) else 0;
    };
    const x = try allocator.alloc(f64, n * n);
    defer allocator.free(x);
    const z = try allocator.alloc(f64, n * n);
    defer allocator.free(z);
    for (0..n) |r| {
        for (0..n) |c| x[r * n + c] = h[r * n + c];
        for (0..(r / block) * block) |k| {
            const lv = t_mat[r * n + k];
            for (0..n) |c| x[r * n + c] -= lv * x[k * n + c];
        }
    }
    for (0..n) |r| {
        for (0..n) |c| z[r * n + c] = x[c * n + r];
        for (0..(r / block) * block) |k| {
            const lv = t_mat[r * n + k];
            for (0..n) |c| z[r * n + c] -= lv * z[k * n + c];
        }
    }
    const td = try allocator.alloc(f64, n * n);
    defer allocator.free(td);
    @memset(td, 0);
    for (0..n) |r| for (0..n / block) |blk| for (0..block) |m| {
        const tv = t_mat[r * n + blk * block + m];
        if (tv == 0) continue;
        for (0..block) |c| td[r * n + blk * block + c] += tv * z[(blk * block + m) * n + blk * block + c];
    };
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (0..n) |r| for (0..n) |c| {
        var acc: f64 = 0;
        for (0..n) |m| acc += td[r * n + m] * t_mat[c * n + m];
        worst = @max(worst, @abs(acc - h[r * n + c]));
        scale = @max(scale, @abs(@as(f64, h[r * n + c])));
    };
    return worst / scale;
}

test "gpu_factor: the float32 factor reconstructs H as (I+L) D (I+L)^T" {
    const t = std.testing;
    const n = 512;
    var bank = try Bank.init(t.allocator, 12, 2, n, 24, 0.5);
    defer bank.deinit();
    var stats = Stats{};
    try factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{}, &stats, bank.ls);
    for (bank.damped, bank.ls) |h, l| {
        const r = try reconstructionResidual(t.allocator, h, l, n);
        try t.expect(r < bound_recon);
    }
}

/// `h` with its first row and column cleared and a diagonal entry of `diag_scale` times the mean diagonal.
fn withBadPivot(h: []f32, n: usize, diag_scale: f32) void {
    var mean: f32 = 0;
    for (0..n) |i| mean += h[i * n + i];
    mean /= @floatFromInt(n);
    for (0..n) |i| {
        h[i] = 0;
        h[i * n] = 0;
    }
    h[0] = diag_scale * mean;
}

fn bankWithBadPivot(allocator: std.mem.Allocator, scale: f32) !Bank {
    const n = 64;
    var bank = try Bank.init(allocator, 21, 3, n, 8, 0);
    errdefer bank.deinit();
    withBadPivot(bank.raw[1], n, scale);
    @memcpy(bank.damped[1], bank.raw[1]);
    bank.means[1] = quantize.dampDiagonal(bank.damped[1], n);
    return bank;
}

test "gpu_factor: a failed float32 pivot is retried with the CPU path's bump and lands on its factor" {
    const t = std.testing;
    const n = 64;
    var bank = try bankWithBadPivot(t.allocator, -0.1);
    defer bank.deinit();
    var stats = Stats{};
    try factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{}, &stats, bank.ls);
    try t.expectEqual(@as(usize, 2), stats.retries);
    try t.expectEqual(@as(usize, 0), stats.fallbacks);
    for (bank.raw, bank.ls) |raw, got| {
        const want = try quantize.ldlFactor(t.allocator, raw, n);
        defer t.allocator.free(want);
        const c = compare(got, want);
        try t.expect(c.frob_rel < bound_frob_small);
    }
}

test "gpu_factor: past the GPU retries a matrix falls back to the float64 factor, byte for byte" {
    const t = std.testing;
    const n = 64;
    var bank = try bankWithBadPivot(t.allocator, -0.1);
    defer bank.deinit();
    var stats = Stats{};
    try factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{ .gpu_retries = 1 }, &stats, bank.ls);
    try t.expectEqual(@as(usize, 1), stats.retries);
    try t.expectEqual(@as(usize, 1), stats.fallbacks);
    const want = try quantize.ldlFactor(t.allocator, bank.raw[1], n);
    defer t.allocator.free(want);
    try t.expectEqualSlices(f32, want, bank.ls[1]);
}

test "gpu_factor: a Hessian no bump repairs fails like the float64 path" {
    const t = std.testing;
    const n = 64;
    var bank = try bankWithBadPivot(t.allocator, -1000);
    defer bank.deinit();
    var stats = Stats{};
    try t.expectError(error.HessianNotPositiveDefinite, factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{}, &stats, bank.ls));
    try t.expectError(error.HessianNotPositiveDefinite, quantize.ldlFactor(t.allocator, bank.raw[1], n));
    for (bank.ls) |l| try t.expectEqual(@as(usize, 0), l.len);
}

test "gpu_factor: the chunk size does not change a byte" {
    const t = std.testing;
    const n = 128;
    var bank = try Bank.init(t.allocator, 31, 5, n, 16, 0);
    defer bank.deinit();
    var stats = Stats{};
    const s = mlx.gpuStream();
    try factorBatch(t.allocator, s, bank.view(), bank.means, n, .{}, &stats, bank.ls);
    var small = try Bank.init(t.allocator, 31, 5, n, 16, 0);
    defer small.deinit();
    try factorBatch(t.allocator, s, small.view(), small.means, n, .{ .chunk_elems = 2 * n * n }, &stats, small.ls);
    for (bank.ls, small.ls) |a, b| try t.expectEqualSlices(f32, a, b);
}

test "gpu_factor: shapes and geometry are refused by name" {
    const t = std.testing;
    var stats = Stats{};
    var one = [_][]f32{&.{}};
    const h: [400]f32 = @splat(0);
    const hs = [_][]const f32{&h};
    try t.expectError(error.GeometryUnsupported, factorBatch(t.allocator, mlx.gpuStream(), &hs, &.{1}, 20, .{}, &stats, &one));
    try t.expectError(error.ShapeMismatch, factorBatch(t.allocator, mlx.gpuStream(), &hs, &.{ 1, 1 }, 16, .{}, &stats, &one));
    try t.expectError(error.ShapeMismatch, factorBatch(t.allocator, mlx.gpuStream(), &hs, &.{1}, 16, .{}, &stats, &one));
}

test "gpu_factor: a Flash-Next sized bank stays within the float32 bound of the float64 factor" {
    const t = std.testing;
    const n = 2560;
    var bank = try Bank.init(t.allocator, 41, 2, n, 64, 0.5);
    defer bank.deinit();
    var stats = Stats{};
    try factorBatch(t.allocator, mlx.gpuStream(), bank.view(), bank.means, n, .{}, &stats, bank.ls);
    try t.expectEqual(@as(usize, 0), stats.fallbacks);
    for (bank.raw, bank.ls) |raw, got| {
        const want = try quantize.ldlFactor(t.allocator, raw, n);
        defer t.allocator.free(want);
        const c = compare(got, want);
        try t.expect(c.frob_rel < bound_frob_large);
        try t.expect(c.max_abs < bound_max_large * c.max_ref);
    }
}

test "gpu_factor: factor throughput (SUSHI_ENCODER_FACTOR_UBENCH)" {
    if (std.c.getenv("SUSHI_ENCODER_FACTOR_UBENCH") == null) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    for ([_][2]usize{ .{ 2560, 32 }, .{ 512, 16 } }) |cfg| {
        const n = cfg[0];
        var bank = try Bank.init(t.allocator, 77, 1, n, 64, 0.5);
        defer bank.deinit();
        const many = try t.allocator.alloc([]const f32, cfg[1]);
        defer t.allocator.free(many);
        const means = try t.allocator.alloc(f64, cfg[1]);
        defer t.allocator.free(means);
        const ls = try t.allocator.alloc([]f32, cfg[1]);
        defer t.allocator.free(ls);
        for (many, means) |*m, *mean| {
            m.* = bank.damped[0];
            mean.* = bank.means[0];
        }
        for (0..2) |_| {
            var stats = Stats{};
            const t0 = stages_mod.nowNs();
            try factorBatch(t.allocator, s, many, means, n, .{}, &stats, ls);
            const dt = stages_mod.nowNs() - t0;
            for (ls) |l| t.allocator.free(l);
            std.debug.print("factor n{d} x{d}: {d:.1} ms ({d:.2} ms per matrix)\n", .{ n, cfg[1], @as(f64, @floatFromInt(dt)) / 1e6, @as(f64, @floatFromInt(dt)) / 1e6 / @as(f64, @floatFromInt(cfg[1])) });
        }
        const c0 = stages_mod.nowNs();
        const cpu = try quantize.ldlFactor(t.allocator, bank.raw[0], n);
        t.allocator.free(cpu);
        std.debug.print("cpu factor n{d}: {d:.1} ms for one matrix\n", .{ n, @as(f64, @floatFromInt(stages_mod.nowNs() - c0)) / 1e6 });
    }
}
