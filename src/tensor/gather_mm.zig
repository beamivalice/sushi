//! MLX's dense gathered product on the engine: `gather_mm` (mlx/ops.cpp) and `GatherMM::eval_gpu` with `gather_mv`,
//! `gather_mm_wide` and `gather_mm` (mlx/backend/metal/matmul.cpp) over the compiled kernels of `mlx.metallib`, so a
//! product here writes the bytes it writes under MLX. Ported from MLX's Metal backend (MIT, see NOTICE). Refused by name,
//! each with its reason (`core.refusing`), and reached by no caller: the sorted kernels of a one-row product
//! (`gather_mm_rhs`), operands of two dtypes (MLX's promotion), any but a float dtype, a 1-D operand, an empty one, and
//! indices with no axes.

const std = @import("std");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");
const matmul = @import("matmul.zig");
const quantized = @import("quantized.zig");

const Tensor = core.Tensor;
const Error = core.Error;

/// `a @ b[rhs]` per index pair: `lhs` picks the batch of `a` and `rhs` the matrix of `b`, each the default `arange`
/// over its batch when not given; `sorted` (the indices given are in order) lets MLX take the sorted kernels.
pub fn gatherMm(a_in: *Tensor, b_in: *Tensor, lhs_in: ?*Tensor, rhs_in: ?*Tensor, sorted: bool) Error!*Tensor {
    if (lhs_in == null and rhs_in == null) return matmul.matmul(a_in, b_in);
    if (a_in.ndim < 2 or b_in.ndim < 2) return core.refusing("a 1-D operand is not ported", error.TensorUnsupported);
    if (a_in.dtype != b_in.dtype) return core.refusing("operands of two dtypes (MLX's promotion) are not ported", error.TensorUnsupported);
    if (a_in.dtype != .float32 and a_in.dtype != .float16 and a_in.dtype != .bfloat16) return core.refusing("a dtype other than float32, float16 and bfloat16 is not ported", error.TensorDtype);
    if (a_in.shape[a_in.ndim - 1] != b_in.shape[b_in.ndim - 2]) return error.TensorShape;
    if (a_in.size() == 0 or b_in.size() == 0) return core.refusing("an empty operand is not ported", error.TensorUnsupported);
    const m: usize = @intCast(a_in.shape[a_in.ndim - 2]);
    const n: usize = @intCast(b_in.shape[b_in.ndim - 1]);
    const k: usize = @intCast(a_in.shape[a_in.ndim - 1]);
    if (m == 1 and sorted and lhs_in == null) return core.refusing("the sorted kernels of a one-row product (gather_mm_rhs) are not ported", error.TensorUnsupported);

    const lhs0 = try quantized.indicesOrDefault(lhs_in, a_in);
    defer core.release(lhs0);
    const rhs0 = try quantized.indicesOrDefault(rhs_in, b_in);
    defer core.release(rhs0);
    var idx_shape: [core.max_dims]c_int = undefined;
    const idx_dims = try quantized.broadcastShape(lhs0, rhs0, &idx_shape);
    if (idx_dims.len == 0) return core.refusing("indices with no axes are not ported", error.TensorUnsupported);
    const lhs = try ops.broadcastTo(lhs0, idx_dims);
    defer core.release(lhs);
    const rhs = try ops.broadcastTo(rhs0, idx_dims);
    defer core.release(rhs);

    var out_shape = try core.Dims.of(idx_dims);
    try out_shape.appendSlice(&.{ @intCast(m), @intCast(n) });
    const out = try core.fresh(a_in.dtype, out_shape.slice());
    errdefer core.release(out);
    const g = try matmul.gpu();
    if (m == 1) {
        try gatherMv(b_in, a_in, rhs, lhs, out, n, k, false);
    } else if (n == 1) {
        try gatherMv(a_in, b_in, lhs, rhs, out, m, k, true);
    } else if (!try wide(a_in, b_in, lhs, rhs, out, m, n, k, g)) {
        try steel(a_in, b_in, lhs, rhs, out, m, n, k, g);
    }
    return out;
}

/// A tensor's strides as the kernels read them.
const Strides = struct {
    v: [2 * core.max_dims]i64 = undefined,
    len: usize = 0,

    fn of(ts: []const *const Tensor) Strides {
        var s: Strides = .{};
        for (ts) |t| for (t.strides[0..t.ndim]) |x| {
            s.v[s.len] = @intCast(x);
            s.len += 1;
        };
        return s;
    }

    fn slice(self: *const Strides) []const i64 {
        return self.v[0..self.len];
    }
};

fn shapeBytes(t: *const Tensor) []const u8 {
    return std.mem.sliceAsBytes(t.shape[0..t.ndim]);
}

/// `gather_mv`: one side a single row (`is_mv` false: `vec @ mat`) or column per pair.
fn gatherMv(mat_in: *Tensor, vec_in: *Tensor, mat_idx: *const Tensor, vec_idx: *const Tensor, out: *Tensor, out_len: usize, k: usize, is_mv: bool) Error!void {
    const mat = try matmul.matrixOperand(mat_in, out_len == 1);
    defer core.release(mat.t);
    const vec = try matmul.matrixOperand(vec_in, true);
    defer core.release(vec.t);
    const transpose_mat = (!is_mv) != mat.transposed;

    var tm: usize = 4;
    var tn: usize = 4;
    var sm: usize = 1;
    var sn: usize = 32;
    var bm: usize = 1;
    var bn: usize = 1;
    var per_group: usize = undefined;
    if (transpose_mat) {
        if (k >= 8192 and out_len >= 2048) {
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
        if (out_len < tm) tm = 1;
        per_group = bm * sm * tm;
    }
    var buf: [96]u8 = undefined;
    const kern = try stock.kernel(try stock.named(&buf, "gemv_{s}gather_{s}_bm{d}_bn{d}_sm{d}_sn{d}_tm{d}_tn{d}", .{ if (transpose_mat) "t_" else "", try stock.typeName(out.dtype), bm, bn, sm, sn, tm, tn }));

    const index_strides = Strides.of(&.{ vec_idx, mat_idx });
    const vec_strides = Strides.of(&.{vec.t});
    const mat_strides = Strides.of(&.{mat.t});
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, mat.t, 0);
    core.bind(p, vec.t, 1);
    core.bindOut(p, out, 3);
    p.constant(int(k), 4);
    p.constant(int(out_len), 5);
    p.constant(@as(c_int, @intCast(mat.ld)), 6);
    p.constant(int(out.ndim - 2), 9);
    p.bytes(shapeBytes(out), 10);
    p.bytes(std.mem.sliceAsBytes(index_strides.slice()), 11);
    p.constant(int(vec.t.ndim - 2), 12);
    p.bytes(shapeBytes(vec.t), 13);
    p.bytes(std.mem.sliceAsBytes(vec_strides.slice()), 14);
    p.constant(int(mat.t.ndim - 2), 15);
    p.bytes(shapeBytes(mat.t), 16);
    p.bytes(std.mem.sliceAsBytes(mat_strides.slice()), 17);
    core.bind(p, vec_idx, 18);
    core.bind(p, mat_idx, 19);
    core.dispatchGroups(p, .{ .x = (out_len + per_group - 1) / per_group, .y = 1, .z = out.size() / out_len }, .{ .x = 32, .y = bn, .z = bm });
}

fn byteOffset(t: *const Tensor) usize {
    return t.store.at + t.offset;
}

/// `gather_mm_wide`: up to fifteen rows a pair against a transposed view of `b`, each matrix streamed once per five
/// rows. False when the plan does not apply.
fn wide(a_in: *Tensor, b: *Tensor, vec_idx: *const Tensor, mat_idx: *const Tensor, out: *Tensor, m: usize, n: usize, k: usize, g: matmul.Gpu) Error!bool {
    if (b.strides[b.ndim - 1] == 1 or b.strides[b.ndim - 2] != 1) return false;
    const a_direct = a_in.strides[a_in.ndim - 1] == 1;
    if (!a_direct and a_in.strides[a_in.ndim - 2] == 1) return false;
    // Generations before M3 keep the other kernels.
    if (g.gen < 15) return false;
    const ldb = b.strides[b.ndim - 1];
    const lda: usize = if (a_direct) a_in.strides[a_in.ndim - 2] else @intCast(a_in.shape[a_in.ndim - 1]);
    var aligned = (if (a_direct) byteOffset(a_in) % 8 == 0 else true) and byteOffset(b) % 8 == 0 and ldb % 4 == 0 and lda % 4 == 0;
    for (b.strides[0 .. b.ndim - 2]) |st| aligned = aligned and st % 4 == 0;
    if (a_direct) {
        for (a_in.strides[0 .. a_in.ndim - 2]) |st| aligned = aligned and st % 4 == 0;
    } else {
        var stride: usize = @as(usize, @intCast(a_in.shape[a_in.ndim - 2])) * @as(usize, @intCast(a_in.shape[a_in.ndim - 1]));
        var i = a_in.ndim - 2;
        while (i > 0) {
            i -= 1;
            aligned = aligned and stride % 4 == 0;
            stride *= @intCast(a_in.shape[i]);
        }
    }
    if (m <= 1 or n <= 1 or k % 4 != 0 or (out.dtype != .float16 and out.dtype != .bfloat16) or !aligned) return false;
    const passes = (m + 4) / 5;
    if (passes > 3) return false;
    const full_simd = passes == 1 or n <= 64;
    const grid_x: usize = if (n >= 65536) 1 else passes;
    const vecs = (m + passes - 1) / passes;
    const lanes: usize = if (full_simd) 32 else 16;

    // Until the load seals the pipelines, a row count makes every kernel the wide plan takes.
    if (!core.sealed) for (2..6) |v| for ([_]usize{ 16, 32 }) |l| {
        _ = wideKernel(out.dtype, v, l) catch {};
    };
    const kern = try wideKernel(out.dtype, vecs, lanes);
    const a = if (a_direct) try core.share(a_in) else try ops.contiguousCopy(a_in);
    defer core.release(a);
    var index_strides = Strides.of(&.{ vec_idx, mat_idx });
    if (index_strides.len == 0) {
        index_strides.v[0..2].* = .{ 0, 0 };
        index_strides.len = 2;
    }
    const a_strides = Strides.of(&.{a});
    const b_strides = Strides.of(&.{b});
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, b, 0);
    core.bind(p, a, 1);
    core.bindOut(p, out, 3);
    p.constant(int(k), 4);
    p.constant(int(n), 5);
    p.constant(int(m), 6);
    p.constant(int(ldb), 7);
    p.constant(int(lda), 8);
    p.constant(int(out.ndim - 2), 11);
    p.bytes(shapeBytes(out), 12);
    p.bytes(std.mem.sliceAsBytes(index_strides.slice()), 13);
    p.constant(int(a.ndim - 2), 14);
    p.bytes(shapeBytes(a), 15);
    p.bytes(std.mem.sliceAsBytes(a_strides.slice()), 16);
    p.constant(int(b.ndim - 2), 17);
    p.bytes(shapeBytes(b), 18);
    p.bytes(std.mem.sliceAsBytes(b_strides.slice()), 19);
    core.bind(p, vec_idx, 20);
    core.bind(p, mat_idx, 21);
    core.dispatchGroups(p, .{ .x = grid_x, .y = (n + 3) / 4, .z = out.size() / m / n }, .{ .x = 32, .y = lanes / 8, .z = 1 });
    return true;
}

fn wideKernel(dtype: core.Dtype, vecs: usize, lanes: usize) Error!@import("../metal.zig").Pipeline {
    var name_buf: [96]u8 = undefined;
    const name = try stock.named(&name_buf, "gemv_wide_gather_{s}_nv{d}_kl{d}", .{ try stock.typeName(dtype), vecs, lanes });
    var hash_buf: [128]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_axpby0", .{name});
    return stock.kernelFlags(name, hash, &.{.{ .index = 1, .value = false }});
}

/// `gather_mm`: the steel tiles, one tile grid per pair.
fn steel(a_in: *Tensor, b_in: *Tensor, lhs: *const Tensor, rhs: *const Tensor, out: *Tensor, m: usize, n: usize, k: usize, g: matmul.Gpu) Error!void {
    const a = try matmul.matrixOperand(a_in, false);
    defer core.release(a.t);
    const b = try matmul.matrixOperand(b_in, false);
    defer core.release(b.t);
    const batch_out = out.size() / m / n;
    const batch_ndim = out.ndim - 2;
    const t = matmul.gemmTile(g, out.dtype, a.transposed, b.transposed, batch_out, m, n, k);
    const has_batch = batch_ndim > 1;
    // Until the load seals the pipelines, a row count makes the kernels every other one aligns to.
    if (!core.sealed) for (0..2) |am| {
        _ = steelKernel(a, b, out.dtype, t, has_batch, am == 1, n % t.bn == 0, k % t.bk == 0) catch {};
    };
    const kern = try steelKernel(a, b, out.dtype, t, has_batch, m % t.bm == 0, n % t.bn == 0, k % t.bk == 0);
    const tiles_n = (n + t.bn - 1) / t.bn;
    const tiles_m = (m + t.bm - 1) / t.bm;
    const params: matmul.GemmParams = .{
        .m = int(m),
        .n = int(n),
        .k = int(k),
        .lda = @intCast(a.ld),
        .ldb = @intCast(b.ld),
        .ldd = int(n),
        .tiles_n = int(tiles_n),
        .tiles_m = int(tiles_m),
        .batch_stride_a = if (batch_ndim > 0) @intCast(lhs.strides[0]) else 0,
        .batch_stride_b = if (batch_ndim > 0) @intCast(rhs.strides[0]) else 0,
        .batch_stride_d = @intCast(m * n),
        .swizzle_log = 0,
        .gemm_k_iterations_aligned = int(k / t.bk),
        .batch_ndim = int(batch_ndim),
    };
    const lhs_strides = Strides.of(&.{lhs});
    const rhs_strides = Strides.of(&.{rhs});
    const a_strides = Strides.of(&.{a.t});
    const b_strides = Strides.of(&.{b.t});
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, a.t, 0);
    core.bind(p, b.t, 1);
    core.bind(p, lhs, 2);
    core.bind(p, rhs, 3);
    core.bindOut(p, out, 4);
    p.constant(params, 5);
    p.bytes(shapeBytes(lhs), 6);
    p.bytes(std.mem.sliceAsBytes(lhs_strides.slice()), 7);
    p.bytes(std.mem.sliceAsBytes(rhs_strides.slice()), 8);
    p.constant(int(a.t.ndim - 2), 9);
    p.bytes(shapeBytes(a.t), 10);
    p.bytes(std.mem.sliceAsBytes(a_strides.slice()), 11);
    p.constant(int(b.t.ndim - 2), 12);
    p.bytes(shapeBytes(b.t), 13);
    p.bytes(std.mem.sliceAsBytes(b_strides.slice()), 14);
    core.dispatchGroups(p, .{ .x = tiles_n, .y = tiles_m, .z = batch_out }, .{ .x = 32, .y = t.wn, .z = t.wm });
}

fn steelKernel(a: matmul.Operand, b: matmul.Operand, dtype: core.Dtype, t: matmul.Tile, has_batch: bool, align_m: bool, align_n: bool, align_k: bool) Error!@import("../metal.zig").Pipeline {
    const type_name = try stock.typeName(dtype);
    var name_buf: [128]u8 = undefined;
    const name = try stock.named(&name_buf, "steel_gather_mm_{c}{c}_{s}_{s}_bm{d}_bn{d}_bk{d}_wm{d}_wn{d}", .{ letter(a.transposed), letter(b.transposed), type_name, type_name, t.bm, t.bn, t.bk, t.wm, t.wn });
    var hash_buf: [256]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_has_batch_{c}_align_M_{c}_align_N_{c}_align_K_{c}", .{ name, letter(has_batch), letter(align_m), letter(align_n), letter(align_k) });
    return stock.kernelFlags(name, hash, &.{
        .{ .index = 10, .value = has_batch },
        .{ .index = 200, .value = align_m },
        .{ .index = 201, .value = align_n },
        .{ .index = 202, .value = align_k },
    });
}

fn letter(yes: bool) u8 {
    return if (yes) 't' else 'n';
}

fn int(v: usize) c_int {
    return @intCast(v);
}
