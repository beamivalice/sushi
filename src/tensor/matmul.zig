//! MLX's matrix product on the engine: `mlx::core::matmul` and `Matmul::eval_gpu` (mlx/ops.cpp,
//! mlx/backend/metal/matmul.cpp), choice for choice, over the compiled kernels of `mlx.metallib`, so a product here
//! writes the bytes it writes under MLX. Ported from MLX's Metal backend (MIT, see NOTICE). An empty operand is not
//! ported.

const std = @import("std");
const builtin = @import("builtin");
const metal = @import("../metal.zig");
const mlx = @import("../mlx.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;
const max_batch = core.max_dims;

/// The GPU as MLX's dispatch reads it: the last letter of its architecture name, its generation, and whether the
/// NAX kernels run (macOS 26.2 is this engine's floor) and take f32.
const Gpu = struct { class: u8, gen: u32, nax: bool, tf32: bool };

var gpu_known: ?Gpu = null;

fn atoi(text: []const u8) i64 {
    var end: usize = 0;
    while (end < text.len and (std.ascii.isDigit(text[end]) or (end == 0 and (text[end] == '-' or text[end] == '+')))) end += 1;
    return std.fmt.parseInt(i64, text[0..end], 10) catch 0;
}

fn gpu() Error!Gpu {
    if (gpu_known) |g| return g;
    var buf: [64]u8 = undefined;
    var name: []const u8 = (try metal.device()).architecture(&buf);
    // MLX's own overrides, read as MLX reads them.
    if (std.c.getenv("MLX_METAL_GPU_ARCH")) |forced| {
        if (std.mem.span(forced).len > 0) name = std.mem.span(forced);
    }
    const digit = struct {
        fn of(c: u8) u32 {
            return if (std.ascii.isDigit(c)) c - '0' else 0;
        }
    }.of;
    const gen: u32 = if (name.len >= 3) digit(name[name.len - 3]) * 10 + digit(name[name.len - 2]) else 0;
    const class: u8 = if (name.len > 0) name[name.len - 1] else 0;
    var tf32 = if (std.c.getenv("MLX_ENABLE_TF32")) |v| atoi(std.mem.span(v)) != 0 else true;
    const nax = gen >= @as(u32, if (class == 'p') 18 else 17);
    if (comptime builtin.is_test) {
        if (nax) tf32 = mlxTf32() catch tf32;
    }
    gpu_known = .{ .class = class, .gen = gen, .nax = nax, .tf32 = tf32 };
    return gpu_known.?;
}

/// Whether the GPU has the matrix units MLX's NAX kernels run on.
pub fn matrixUnits() bool {
    return (gpu() catch return false).nax;
}

/// Whether MLX takes an f32 product on the NAX tiles, by what a product of its own gives: those keep 10 mantissa bits
/// of an operand, so 1 + 2^-12 multiplies as 1. Tests only: MLX keeps its first reading of `MLX_ENABLE_TF32` for the
/// process, so a test that changes the variable around an MLX product leaves MLX on a value the environment no
/// longer shows, and every test after it that holds the engine to MLX's bytes would fail by the order they ran in.
fn mlxTf32() !bool {
    const c = struct {
        extern "c" fn mlx_matmul(res: *mlx.mlx_array, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) c_int;
    };
    const n = 64;
    const values: [n * n]f32 = @splat(1.0 + 1.0 / 4096.0);
    const x = mlx.raw.mlx_array_new_data(&values, &[_]c_int{ n, n }, 2, .float32);
    defer _ = mlx.raw.mlx_array_free(x);
    const gpu_stream = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu_stream);
    var squared = mlx.raw.mlx_array_new();
    defer _ = mlx.raw.mlx_array_free(squared);
    try mlx.check(c.mlx_matmul(&squared, x, x, gpu_stream));
    try mlx.check(mlx.raw.mlx_array_eval(squared));
    const got = mlx.mlx_array_data_float32(squared) orelse return error.NotEvaluated;
    return got[0] == n;
}

const Operand = struct { t: *Tensor, transposed: bool, ld: i64 };

/// MLX's `check_transpose`: the operand as it is when its last two axes are a row-major or a column-major matrix,
/// else a row-contiguous copy.
fn matrixOperand(t: *Tensor, is_vector: bool) Error!Operand {
    const stx: i64 = @intCast(t.strides[t.ndim - 2]);
    const sty: i64 = @intCast(t.strides[t.ndim - 1]);
    if (sty == 1 and (!is_vector or stx == t.shape[t.ndim - 1])) return .{ .t = try core.share(t), .transposed = false, .ld = stx };
    if (stx == 1 and (!is_vector or sty == t.shape[t.ndim - 2])) return .{ .t = try core.share(t), .transposed = true, .ld = sty };
    return .{ .t = try ops.contiguousCopy(t), .transposed = false, .ld = t.shape[t.ndim - 1] };
}

/// The batch axes of two operands as the kernels walk them: runs of axes that are one run of memory in both become
/// one axis.
const Batches = struct {
    n: usize = 0,
    shape: [max_batch]c_int = undefined,
    a: [max_batch]i64 = undefined,
    b: [max_batch]i64 = undefined,

    const one: Batches = blk: {
        var x: Batches = .{};
        x.push(1, 0, 0);
        break :blk x;
    };

    fn push(self: *Batches, len: c_int, sa: i64, sb: i64) void {
        self.shape[self.n] = len;
        self.a[self.n] = sa;
        self.b[self.n] = sb;
        self.n += 1;
    }
};

/// MLX's `collapse_batches` over `collapse_contiguous_dims`.
fn collapseBatches(a: *const Tensor, b: *const Tensor) Batches {
    if (a.ndim == 2) return .one;
    const nb = a.ndim - 2;
    const shape = a.shape[0..nb];
    const sa = a.strides[0..nb];
    const sb = b.strides[0..nb];
    const cut: i32 = -1;
    var plan: [2 * max_batch + 2]i32 = undefined;
    var len: usize = 0;
    if (shape[0] != 1) {
        plan[len] = 0;
        len += 1;
    }
    var size: i64 = shape[0];
    for (1..nb) |i| {
        size *= shape[i];
        const run = for ([_][]const usize{ sa, sb }) |st| {
            if (st[i] * @as(usize, @intCast(shape[i])) != st[i - 1] or size > std.math.maxInt(i32)) break false;
        } else true;
        if (!run) {
            size = shape[i];
            plan[len] = cut;
            len += 1;
        }
        if (shape[i] != 1) {
            plan[len] = @intCast(i);
            len += 1;
        }
    }
    plan[len] = cut;
    len += 1;

    var out: Batches = .{};
    var i: usize = 0;
    while (true) {
        while (i < len and plan[i] == cut) i += 1;
        if (i == len) break;
        var extent: c_int = shape[@intCast(plan[i])];
        var k = i + 1;
        while (plan[k] != cut) : (k += 1) extent *= shape[@intCast(plan[k])];
        const last: usize = @intCast(plan[k - 1]);
        out.push(extent, @intCast(sa[last]), @intCast(sb[last]));
        i = k + 1;
    }
    return if (out.n == 0) .one else out;
}

fn letter(yes: bool) u8 {
    return if (yes) 't' else 'n';
}

/// `a @ b` as MLX's `matmul` gives it: a 1-D operand is a row or a column for the product, batch axes broadcast,
/// and a batched `a` against one matrix is flattened into rows.
pub fn matmul(a_in: *Tensor, b_in: *Tensor) Error!*Tensor {
    if (a_in.ndim == 0 or b_in.ndim == 0) return error.TensorShape;
    if (a_in.dtype != b_in.dtype) return error.TensorDtype;
    switch (a_in.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    if (a_in.size() == 0 or b_in.size() == 0) return error.TensorUnsupported;
    var a = if (a_in.ndim == 1) try ops.expandDims(a_in, 0) else try core.share(a_in);
    defer core.release(a);
    var b = if (b_in.ndim == 1) try ops.expandDims(b_in, 1) else try core.share(b_in);
    defer core.release(b);
    const k = a.shape[a.ndim - 1];
    if (k != b.shape[b.ndim - 2]) return error.TensorShape;
    const n = b.shape[b.ndim - 1];

    const flattened = a_in.ndim > 2 and b_in.ndim <= 2;
    if (flattened) {
        const flat = try ops.reshape(a, &.{ -1, k });
        core.release(a);
        a = flat;
    } else if (a_in.ndim > 2 or b_in.ndim > 2) {
        const an = a.ndim - 2;
        const bn = b.ndim - 2;
        const batch = @max(an, bn);
        if (batch + 2 > core.max_dims) return error.TensorShape;
        var sa: [core.max_dims]c_int = undefined;
        var sb: [core.max_dims]c_int = undefined;
        for (0..batch) |i| {
            const da: c_int = if (i + an >= batch) a.shape[i + an - batch] else 1;
            const db: c_int = if (i + bn >= batch) b.shape[i + bn - batch] else 1;
            const d = if (da == db or db == 1) da else if (da == 1) db else return error.TensorShape;
            sa[i] = d;
            sb[i] = d;
        }
        @memcpy(sa[batch..][0..2], a.shape[an..][0..2]);
        @memcpy(sb[batch..][0..2], b.shape[bn..][0..2]);
        const wide_a = try ops.broadcastTo(a, sa[0 .. batch + 2]);
        core.release(a);
        a = wide_a;
        const wide_b = try ops.broadcastTo(b, sb[0 .. batch + 2]);
        core.release(b);
        b = wide_b;
    }

    var out = try product(a, b);
    errdefer core.release(out);
    if (flattened) {
        var shape: [core.max_dims]c_int = undefined;
        const lead = a_in.ndim - 1;
        @memcpy(shape[0..lead], a_in.shape[0..lead]);
        shape[lead] = n;
        const shaped = try ops.reshape(out, shape[0 .. lead + 1]);
        core.release(out);
        out = shaped;
    }
    var axes: [2]c_int = undefined;
    var dropped: usize = 0;
    if (a_in.ndim == 1) {
        axes[dropped] = @as(c_int, out.ndim) - 2;
        dropped += 1;
    }
    if (b_in.ndim == 1) {
        axes[dropped] = @as(c_int, out.ndim) - 1;
        dropped += 1;
    }
    if (dropped == 0) return out;
    const squeezed = try ops.squeeze(out, axes[0..dropped]);
    core.release(out);
    return squeezed;
}

/// `Matmul::eval_gpu`: operands of one rank whose batch axes agree.
fn product(a_pre: *Tensor, b_pre: *Tensor) Error!*Tensor {
    var m: usize = @intCast(a_pre.shape[a_pre.ndim - 2]);
    const n: usize = @intCast(b_pre.shape[b_pre.ndim - 1]);
    const k: usize = @intCast(a_pre.shape[a_pre.ndim - 1]);
    var shape: [core.max_dims]c_int = undefined;
    @memcpy(shape[0..a_pre.ndim], a_pre.dims());
    shape[a_pre.ndim - 1] = @intCast(n);

    const a = try matrixOperand(a_pre, m == 1);
    defer core.release(a.t);
    const b = try matrixOperand(b_pre, n == 1);
    defer core.release(b.t);
    var batches = collapseBatches(a.t, b.t);
    var batch_out: usize = 1;
    for (shape[0 .. a_pre.ndim - 2]) |d| batch_out *= @intCast(d);

    // A batch of `a` that is one run of rows against one `b` is one taller product.
    if (batch_out > 1 and !a.transposed and batches.n == 1 and a.t.strides[a.t.ndim - 2] == k and
        batches.a[0] == @as(i64, @intCast(m * k)) and batches.b[0] == 0)
    {
        m *= @intCast(batches.shape[0]);
        batch_out = 1;
        batches = .one;
    }
    if (m == 1 and n == 1 and batch_out == 1 and a.t.rowContiguous() and b.t.rowContiguous()) return dot(a.t, b.t, k, shape[0..a_pre.ndim]);

    const out = try core.fresh(a_pre.dtype, shape[0..a_pre.ndim]);
    errdefer core.release(out);
    const g = try gpu();
    const call: Call = .{ .g = g, .a = a, .b = b, .out = out, .m = m, .n = n, .k = k, .batch_out = batch_out, .batches = batches };
    if (!a.transposed and b.transposed and try gemvWide(call)) return out;
    if (@min(m, n) == 1) {
        try gemv(call);
        return out;
    }
    try steel(call);
    return out;
}

/// MLX's `dot_product`: partial sums in f32 per block of 512 threads, summed by MLX's all-reduce, then cast.
fn dot(a: *Tensor, b: *Tensor, k: usize, shape: []const c_int) Error!*Tensor {
    const group = 512;
    const blocks = ((k + 31) / 32 + group - 1) / group;
    const partials = try core.fresh(.float32, &.{@intCast(blocks)});
    defer core.release(partials);
    var buf: [64]u8 = undefined;
    const p = try core.pass();
    p.pipeline(try stock.kernel(try stock.named(&buf, "dot_product_{s}_it32_tg512_sg16", .{try stock.typeName(a.dtype)})));
    core.bind(p, a, 0);
    core.bind(p, b, 1);
    core.bindOut(p, partials, 2);
    p.constant(@as(c_int, @intCast(k)), 3);
    core.dispatch(p, .{ .x = blocks * group }, .{ .x = group });

    const sum = try core.fresh(.float32, shape);
    errdefer core.release(sum);
    try @import("reduce.zig").all(.sum, partials, sum);
    if (a.dtype == .float32) return sum;
    defer core.release(sum);
    return ops.astype(sum, a.dtype);
}

const Call = struct {
    g: Gpu,
    a: Operand,
    b: Operand,
    out: *Tensor,
    m: usize,
    n: usize,
    k: usize,
    batch_out: usize,
    batches: Batches,

    fn int(v: usize) c_int {
        return @intCast(v);
    }
};

fn byteOffset(t: *const Tensor) usize {
    return t.store.at + t.offset;
}

/// MLX's `gemv_wide`: up to fifteen rows against a transposed half-precision weight, the weight streamed once per
/// five rows. False when the plan does not apply.
fn gemvWide(c: Call) Error!bool {
    const mat = c.b;
    const vec = c.a;
    // Generations before M3 keep the other kernels.
    if (c.g.gen < 15) return false;
    var aligned = byteOffset(mat.t) % 8 == 0 and byteOffset(vec.t) % 8 == 0 and @rem(mat.ld, 4) == 0 and @rem(vec.ld, 4) == 0;
    for (c.batches.b[0..c.batches.n], c.batches.a[0..c.batches.n]) |sm, sv| aligned = aligned and @rem(sm, 4) == 0 and @rem(sv, 4) == 0;
    if (c.m <= 1 or c.n <= 1 or c.k % 4 != 0 or (c.out.dtype != .float16 and c.out.dtype != .bfloat16) or !aligned) return false;
    const passes = (c.m + 4) / 5;
    if (passes > 3) return false;
    const full_simd = passes == 1 or c.n <= 64;
    const grid_x: usize = if (c.n >= 65536) 1 else passes;
    const vecs = (c.m + passes - 1) / passes;
    const lanes: usize = if (full_simd) 32 else 16;
    const has_batch = c.batches.n != 1;

    var name_buf: [96]u8 = undefined;
    const name = try stock.named(&name_buf, "gemv_wide_{s}_nv{d}_kl{d}", .{ try stock.typeName(c.out.dtype), vecs, lanes });
    var hash_buf: [128]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_nc{d}_axpby0", .{ name, @intFromBool(has_batch) });
    const kern = try stock.kernelFlags(name, hash, &.{ .{ .index = 0, .value = has_batch }, .{ .index = 1, .value = false } });
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, mat.t, 0);
    core.bind(p, vec.t, 1);
    core.bindOut(p, c.out, 3);
    p.constant(Call.int(c.k), 4);
    p.constant(Call.int(c.n), 5);
    p.constant(Call.int(c.m), 6);
    p.constant(@as(c_int, @intCast(mat.ld)), 7);
    p.constant(@as(c_int, @intCast(vec.ld)), 8);
    p.constant(Call.int(c.batches.n), 11);
    p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 12);
    p.bytes(std.mem.sliceAsBytes(c.batches.a[0..c.batches.n]), 13);
    p.bytes(std.mem.sliceAsBytes(c.batches.b[0..c.batches.n]), 14);
    core.dispatchGroups(p, .{ .x = grid_x, .y = (c.n + 3) / 4, .z = c.batch_out }, .{ .x = 32, .y = lanes / 8, .z = 1 });
    return true;
}

/// MLX's `gemv`: one side is a single row or column per batch.
fn gemv(c: Call) Error!void {
    const b_is_matrix = c.n != 1;
    const mat = if (b_is_matrix) c.b else c.a;
    const vec = if (b_is_matrix) c.a else c.b;
    const transpose_mat = if (b_is_matrix) !c.b.transposed else c.a.transposed;
    const out_len = if (b_is_matrix) c.n else c.m;
    const strides_mat = if (b_is_matrix) &c.batches.b else &c.batches.a;
    const strides_vec = if (b_is_matrix) &c.batches.a else &c.batches.b;
    const contiguous = c.batches.n == 1;

    var tm: usize = 4;
    var tn: usize = 4;
    var sm: usize = 1;
    var sn: usize = 32;
    var bm: usize = 1;
    var bn: usize = 1;
    var per_group: usize = undefined;
    if (transpose_mat) {
        if (c.k >= 8192 and out_len >= 2048) {
            sm = 4;
            sn = 8;
        } else {
            sm = 8;
            sn = 4;
        }
        bn = if (out_len >= 2048) 16 else if (out_len >= 512) 4 else 2;
        if (out_len < tn) tn = 1;
        per_group = bn * sn * tn;
    } else {
        bm = if (out_len >= 4096) 8 else 4;
        if (c.k <= 64) {
            bm = 1;
            sm = 8;
            sn = 4;
        } else if (c.k >= 16 * out_len) {
            bm = 1;
            bn = 8;
        }
        if (out_len < tm) tm = 1;
        per_group = bm * sm * tm;
    }
    var buf: [96]u8 = undefined;
    const kern = try stock.kernel(try stock.named(&buf, "gemv_{s}{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}_nc{d}_axpby0", .{ if (transpose_mat) "t_" else "", try stock.typeName(c.out.dtype), bm, bn, sm, sn, tm, tn, @intFromBool(!contiguous) }));
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, mat.t, 0);
    core.bind(p, vec.t, 1);
    core.bindOut(p, c.out, 3);
    p.constant(Call.int(c.k), 4);
    p.constant(Call.int(out_len), 5);
    p.constant(@as(c_int, @intCast(mat.ld)), 6);
    p.constant(Call.int(c.batches.n), 9);
    p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 10);
    p.bytes(std.mem.sliceAsBytes(strides_vec[0..c.batches.n]), 11);
    p.bytes(std.mem.sliceAsBytes(strides_mat[0..c.batches.n]), 12);
    core.dispatchGroups(p, .{ .x = (out_len + per_group - 1) / per_group, .y = 1, .z = c.batch_out }, .{ .x = 32, .y = bn, .z = bm });
}

/// MLX's `steel_matmul`: two matrices, by the tiled kernels.
fn steel(c: Call) Error!void {
    const use_nax = c.g.nax and (c.g.tf32 or c.out.dtype != .float32);
    const threshold: usize = if (c.g.class == 's' or c.g.class == 'd') 2048 else 1024;
    const big = @max(c.m, c.n);
    if (!use_nax and c.batch_out == 1 and ((c.m + 15) / 16) * ((c.n + 15) / 16) <= threshold and c.k / 16 >= 8 and c.k >= big) return splitK(c);
    if (use_nax and c.batch_out == 1 and (c.k >= 3 * big or (big <= 1024 and c.k > 2 * big))) return splitKNax(c);
    return if (use_nax) fusedNax(c) else fused(c);
}

const Tile = struct { bm: usize, bn: usize, bk: usize, wm: usize, wn: usize };

/// `GEMMParams` of mlx/backend/metal/kernels/steel/gemm/params.h.
const GemmParams = extern struct {
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldd: c_int,
    tiles_n: c_int,
    tiles_m: c_int,
    batch_stride_a: i64,
    batch_stride_b: i64,
    batch_stride_d: i64,
    swizzle_log: c_int,
    gemm_k_iterations_aligned: c_int,
    batch_ndim: c_int,
};

/// `GEMMSpiltKParams` of the same header.
const SplitKParams = extern struct {
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
    tiles_n: c_int,
    tiles_m: c_int,
    split_k_partitions: c_int,
    split_k_partition_stride: c_int,
    split_k_partition_size: c_int,
    swizzle_log: c_int,
    gemm_k_iterations_aligned: c_int,
};

/// `steel_matmul_regular_axpby`'s tile, MLX's `GEMM_TPARAM_MACRO`.
fn fused(c: Call) Error!void {
    var t: Tile = .{ .bm = 64, .bn = 64, .bk = 16, .wm = 2, .wn = 2 };
    const nt = !c.a.transposed and c.b.transposed;
    const half = c.out.dtype != .float32;
    const nt_tile: Tile = .{ .bm = 64, .bn = 32, .bk = 32, .wm = 2, .wn = 2 };
    const half_tile: Tile = .{ .bm = 64, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
    if (c.g.class == 'g' or c.g.class == 'p') {
        if (nt) t = nt_tile else if (half) t = half_tile;
    } else if (c.g.class == 'd') {
        if (c.batch_out * c.m * c.n >= 1 << 20) {
            if (half) {
                t = if (2 * @max(c.m, c.n) > c.k) half_tile else if (nt) nt_tile else .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 };
            }
        } else if (half) {
            t = if (nt) nt_tile else half_tile;
        } else {
            t = if (nt) .{ .bm = 32, .bn = 64, .bk = 16, .wm = 1, .wn = 2 } else nt_tile;
        }
    }
    return fusedLaunch(c, "steel_gemm_fused", t, 0);
}

/// `steel_matmul_regular_axpby_nax`'s tile and swizzle.
fn fusedNax(c: Call) Error!void {
    var t: Tile = .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
    const class = c.g.class;
    if (class == 's' or class == 'c') {
        t.bk = if (c.k >= 8192 and c.k > c.m + c.n) 64 else 256;
        t.bm = 64;
        t.wm = 2;
    } else if (class == 'd') {
        t = if (c.n > c.m) .{ .bm = 128, .bn = 64, .bk = 512, .wm = 4, .wn = 2 } else .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4 };
    }
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    var swizzle: u5 = if (tiles_m <= 3) 0 else 1;
    if (class == 's' or class == 'c') swizzle = 2 else if (class == 'd') swizzle = if (c.m > c.n) 0 else 2;
    return fusedLaunch(c, "steel_gemm_fused_nax", t, swizzle);
}

fn fusedLaunch(c: Call, comptime family: []const u8, t: Tile, swizzle: u5) Error!void {
    const type_name = try stock.typeName(c.out.dtype);
    var name_buf: [128]u8 = undefined;
    const name = try stock.named(&name_buf, family ++ "_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ letter(c.a.transposed), letter(c.b.transposed), type_name, type_name, t.bm, t.bn, t.bk, t.wm, t.wn });
    const has_batch = c.batches.n > 1;
    const align_m = c.m % t.bm == 0;
    const align_n = c.n % t.bn == 0;
    const align_k = c.k % t.bk == 0;
    var hash_buf: [256]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_has_batch_{c}_use_out_source_n_do_axpby_n_align_M_{c}_align_N_{c}_align_K_{c}", .{ name, letter(has_batch), letter(align_m), letter(align_n), letter(align_k) });
    const kern = try stock.kernelFlags(name, hash, &.{
        .{ .index = 10, .value = has_batch },
        .{ .index = 100, .value = false },
        .{ .index = 110, .value = false },
        .{ .index = 200, .value = align_m },
        .{ .index = 201, .value = align_n },
        .{ .index = 202, .value = align_k },
    });
    var tiles_n = (c.n + t.bn - 1) / t.bn;
    var tiles_m = (c.m + t.bm - 1) / t.bm;
    const params: GemmParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldd = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .batch_stride_a = c.batches.a[c.batches.n - 1],
        .batch_stride_b = c.batches.b[c.batches.n - 1],
        .batch_stride_d = @intCast(c.m * c.n),
        .swizzle_log = swizzle,
        .gemm_k_iterations_aligned = Call.int(c.k / t.bk),
        .batch_ndim = Call.int(c.batches.n),
    };
    const tile = @as(usize, 1) << swizzle;
    tiles_m = (tiles_m + tile - 1) / tile;
    tiles_n *= tile;

    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, c.out, 3);
    p.constant(params, 4);
    if (has_batch) {
        var strides: [2 * max_batch]i64 = undefined;
        @memcpy(strides[0..c.batches.n], c.batches.a[0..c.batches.n]);
        @memcpy(strides[c.batches.n..][0..c.batches.n], c.batches.b[0..c.batches.n]);
        p.bytes(std.mem.sliceAsBytes(c.batches.shape[0..c.batches.n]), 6);
        p.bytes(std.mem.sliceAsBytes(strides[0 .. 2 * c.batches.n]), 7);
    }
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = c.batch_out }, .{ .x = 32, .y = t.wn, .z = t.wm });
}

/// MLX's `get_block_dims`: a threadgroup of up to 1024 threads, grown one axis at a time.
fn blockDims(dims: [3]usize) metal.Size {
    var pows: [3]u6 = .{ 0, 0, 0 };
    var sum: usize = 0;
    outer: while (true) {
        const before = sum;
        for (dims, &pows, 0..) |d, *p, axis| {
            if (d >= @as(usize, 1) << (p.* + 1)) {
                p.* += 1;
                sum += 1;
            }
            if (sum == 10 and axis < 2) break :outer;
        }
        if (sum == before or sum == 10) break;
    }
    return .{ .x = @as(usize, 1) << pows[0], .y = @as(usize, 1) << pows[1], .z = @as(usize, 1) << pows[2] };
}

/// The f32 parts of a split product summed into `out`.
fn accumulate(c: Call, parts: *Tensor, count: usize) Error!void {
    var buf: [96]u8 = undefined;
    const kern = try stock.kernel(try stock.named(&buf, "steel_gemm_splitk_accum_{s}_float32", .{try stock.typeName(c.out.dtype)}));
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, parts, 0);
    core.bindOut(p, c.out, 1);
    p.constant(Call.int(count), 2);
    p.constant(Call.int(c.m * c.n), 3);
    p.constant(Call.int(c.n), 4);
    core.dispatch(p, .{ .x = c.n, .y = c.m, .z = 1 }, blockDims(.{ c.n, c.m, 1 }));
}

fn nextPowerOfTwo(v: usize) usize {
    return if (v <= 1) v else std.math.ceilPowerOfTwo(usize, v) catch v;
}

/// `steel_gemm_splitk_axpby`: a long inner axis cut into parts, each a product into f32, then summed.
fn splitK(c: Call) Error!void {
    const tm = (c.m + 31) / 32;
    const tn = (c.n + 31) / 32;
    const tk = c.k / 16;
    const t: Tile = .{ .bm = if (c.m < 40) 16 else 32, .bn = if (c.n < 40) 16 else 32, .bk = 16, .wm = 2, .wn = 2 };
    const parts_n = @min(@max(2, nextPowerOfTwo(tk / (tm * tn))), 32);
    const iterations = (c.k / t.bk) / parts_n;
    const parts = try core.fresh(.float32, &.{ Call.int(parts_n), Call.int(c.m), Call.int(c.n) });
    defer core.release(parts);
    const aligned_mn = c.m % t.bm == 0 and c.n % t.bn == 0;
    const aligned_k = c.k % t.bk == 0;
    var buf: [160]u8 = undefined;
    const kern = try stock.kernel(try stock.named(&buf, "steel_gemm_splitk_{c}{c}_{s}_float32_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}_MN_{c}aligned_K_{c}aligned", .{ letter(c.a.transposed), letter(c.b.transposed), try stock.typeName(c.out.dtype), t.bm, t.bn, t.bk, t.wm, t.wn, letter(aligned_mn), letter(aligned_k) }));
    const tiles_n = (c.n + t.bn - 1) / t.bn;
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    const params: SplitKParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldc = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .split_k_partitions = Call.int(parts_n),
        .split_k_partition_stride = Call.int(c.m * c.n),
        .split_k_partition_size = Call.int(iterations * t.bk),
        .swizzle_log = 0,
        .gemm_k_iterations_aligned = Call.int(iterations),
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, parts, 2);
    p.constant(params, 3);
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = parts_n }, .{ .x = 32, .y = t.wn, .z = t.wm });
    try accumulate(c, parts, parts_n);
}

/// `steel_gemm_splitk_axpby_nax`.
fn splitKNax(c: Call) Error!void {
    var t: Tile = .{ .bm = 128, .bn = 128, .bk = 512, .wm = 4, .wn = 4 };
    var part_size: usize = 4096;
    const ultra = c.g.class == 'd';
    if (ultra) {
        if (c.m <= 2048 and c.n <= 2048) {
            t = .{ .bm = 64, .bn = 64, .bk = 256, .wm = 2, .wn = 2 };
            part_size = 2048;
        } else if (c.n > c.m) {
            t = .{ .bm = 128, .bn = 64, .bk = 512, .wm = 4, .wn = 2 };
        } else {
            t = .{ .bm = 64, .bn = 128, .bk = 512, .wm = 2, .wn = 4 };
        }
    }
    if ((c.m + c.n) / 2 < 512 or c.k <= 4096) t = .{ .bm = 64, .bn = 64, .bk = 256, .wm = 2, .wn = 2 };
    if (c.k <= 1024) part_size = c.k / 2 else if (c.k <= 2048) part_size = 1024 else if (c.k <= 4096) part_size = 2048;
    const parts_n = (c.k + part_size - 1) / part_size;
    const parts = try core.fresh(.float32, &.{ Call.int(parts_n), Call.int(c.m), Call.int(c.n) });
    defer core.release(parts);
    const align_m = c.m % t.bm == 0;
    const align_n = c.n % t.bn == 0;
    const align_k = c.k % t.bk == 0;
    var name_buf: [128]u8 = undefined;
    const name = try stock.named(&name_buf, "steel_gemm_splitk_nax_{c}{c}_{s}_float32_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ letter(c.a.transposed), letter(c.b.transposed), try stock.typeName(c.out.dtype), t.bm, t.bn, t.bk, t.wm, t.wn });
    var hash_buf: [192]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_align_M_{c}_align_N_{c}_align_K_{c}", .{ name, letter(align_m), letter(align_n), letter(align_k) });
    const kern = try stock.kernelFlags(name, hash, &.{ .{ .index = 200, .value = align_m }, .{ .index = 201, .value = align_n } });
    const tiles_n = (c.n + t.bn - 1) / t.bn;
    const tiles_m = (c.m + t.bm - 1) / t.bm;
    var swizzle: u5 = if (tiles_m <= 3) 0 else 1;
    if (ultra) swizzle = if (c.m > c.n) 0 else 1;
    const tile = @as(usize, 1) << swizzle;
    const params: SplitKParams = .{
        .m = Call.int(c.m),
        .n = Call.int(c.n),
        .k = Call.int(c.k),
        .lda = @intCast(c.a.ld),
        .ldb = @intCast(c.b.ld),
        .ldc = Call.int(c.n),
        .tiles_n = Call.int(tiles_n),
        .tiles_m = Call.int(tiles_m),
        .split_k_partitions = Call.int(parts_n),
        .split_k_partition_stride = Call.int(c.m * c.n),
        .split_k_partition_size = Call.int(part_size),
        .swizzle_log = swizzle,
        .gemm_k_iterations_aligned = Call.int(part_size / t.bk),
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, c.a.t, 0);
    core.bind(p, c.b.t, 1);
    core.bindOut(p, parts, 2);
    p.constant(params, 3);
    core.dispatchGroups(p, .{ .x = (tiles_n * tile) * ((tiles_m + tile - 1) / tile) * parts_n, .y = 1, .z = 1 }, .{ .x = 32, .y = t.wn, .z = t.wm });
    try accumulate(c, parts, parts_n);
}
