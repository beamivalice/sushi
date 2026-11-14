//! MLX's affine quantized matrix product on the engine: `quantized_matmul` (mlx/ops.cpp) and
//! `QuantizedMatmul::eval_gpu`, `qmm_splitk`, `qmm` and `qmm_nax` (mlx/backend/metal/quantized.cpp) over the compiled
//! kernels of `mlx.metallib`, so a product here writes the bytes it writes under MLX. Ported from MLX's Metal backend
//! (MIT, see NOTICE). Ported: a transposed affine weight against a 2-D `x`, on the tiles MLX takes for
//! `M >= get_qmv_batch_limit`: the NAX tiles of a GPU with matrix units, the SIMD tiles of any other (and of an
//! unaligned K). Refused by name, each one because no request of the one caller reaches it:
//!   - the vector kernels and the split-K tiles: the vision tower gives every linear 164 rows or more (`Dsv41Vision`,
//!     `engineGap` checks it at the load), and MLX takes them below 257 rows at most;
//!   - a batch, a weight that is not transposed, another mode: the tower's linears are 2-D, transposed and affine.

const core = @import("core.zig");
const matmul = @import("matmul.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;

/// `get_qmv_batch_limit` (mlx/backend/metal/quantized.cpp): the rows from which MLX leaves the vector kernels, by the
/// generation and the class letter.
fn vectorLimit(g: matmul.Gpu, k: usize, n: usize) usize {
    const small = k <= 2048 and n <= 2048;
    const mid = k <= 4096 and n <= 4096;
    const ultra = g.class == .d;
    if (g.gen >= 17 and !ultra) return if (small) 33 else if (mid) 25 else 13;
    if (g.gen >= 15 and !ultra) return if (small) 13 else if (mid) 15 else 13;
    if (g.gen >= 13) {
        if (ultra) return if (small) 32 else if (mid) 18 else 12;
        return if (small) 14 else if (mid) 10 else 6;
    }
    if (ultra) return if (small) 32 else if (mid) 18 else 12;
    return if (small) 18 else if (mid) 12 else 10;
}

/// `x @ w.T` for the affine codes `w` (uint32, `[n, k * bits / 32]`) with a scale and a bias per group of `group`
/// inputs, as `mlx::core::quantized_matmul` gives it.
pub fn quantizedMatmul(x_in: *Tensor, w_in: *Tensor, scales_in: *Tensor, biases_in: *Tensor, group: usize, bits: usize) Error!*Tensor {
    if (w_in.dtype != .uint32 or x_in.ndim != 2 or w_in.ndim != 2 or scales_in.ndim != 2 or biases_in.ndim != 2) return error.TensorShape;
    switch (x_in.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    if (scales_in.dtype != x_in.dtype or biases_in.dtype != x_in.dtype) return error.TensorDtype;
    if (group == 0 or bits == 0 or 32 % bits != 0) return error.TensorUnsupported;
    const m: usize = @intCast(x_in.shape[0]);
    const k: usize = @intCast(x_in.shape[1]);
    const n: usize = @intCast(w_in.shape[0]);
    if (k != @as(usize, @intCast(w_in.shape[1])) * 32 / bits or scales_in.shape[0] != w_in.shape[0] or scales_in.shape[1] != biases_in.shape[1] or
        k != @as(usize, @intCast(scales_in.shape[1])) * group or biases_in.shape[0] != w_in.shape[0]) return error.TensorShape;
    if (m == 0 or n == 0 or k == 0) return error.TensorUnsupported;

    // `ensure_row_contiguous_matrix`: rows of `k` laid out, whatever the offset.
    const x = try matrixRows(x_in);
    defer core.release(x);
    const w = try matrixRows(w_in);
    defer core.release(w);
    const scales = try matrixRows(scales_in);
    defer core.release(scales);
    const biases = try matrixRows(biases_in);
    defer core.release(biases);

    const g = try matmul.gpu();
    if (unported(g, m, n, k, group) != null) return error.TensorUnsupported;

    const nax = useNax(g, k, x.dtype);
    const bm: usize = if (!nax) 32 else if (m <= 32) 32 else 64;
    const bn: usize = if (nax) 64 else 32;
    const aligned = n % bn == 0;
    var name_buf: [160]u8 = undefined;
    const name = if (nax)
        try stock.named(&name_buf, "affine_qmm_t_nax_{s}_gs_{d}_b_{d}_bm{d}_bn64_bk64_wm2_wn2_alN_{s}_batch_0", .{ try stock.cType(x.dtype), group, bits, bm, if (aligned) "true" else "false" })
    else
        try stock.named(&name_buf, "affine_qmm_t_{s}_gs_{d}_b_{d}_alN_{s}_batch_0", .{ try stock.cType(x.dtype), group, bits, if (aligned) "true" else "false" });
    const kern = try stock.kernel(name);
    const out = try core.fresh(x.dtype, &.{ @intCast(m), @intCast(n) });
    errdefer core.release(out);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, w, 0);
    core.bind(p, scales, 1);
    core.bind(p, biases, 2);
    core.bind(p, x, 3);
    core.bindOut(p, out, 4);
    p.constant(@as(c_int, @intCast(k)), 5);
    p.constant(@as(c_int, @intCast(n)), 6);
    p.constant(@as(c_int, @intCast(m)), 7);
    core.dispatchGroups(p, .{ .x = (n + bn - 1) / bn, .y = (m + bm - 1) / bm, .z = 1 }, .{ .x = 32, .y = 2, .z = 2 });
    return out;
}

/// `qmm`'s choice of the NAX tiles: matrix units, K a whole 64 (the weight is transposed), and f32 only with TF32.
fn useNax(g: matmul.Gpu, k: usize, dtype: core.Dtype) bool {
    return g.nax and k % 64 == 0 and (dtype != .float32 or g.tf32);
}

/// The MLX plans this port leaves out, and the reason a product takes one: fewer rows than MLX's vector limit, or a
/// split along K.
pub const Unported = enum { vector_kernels, split_k };

/// Which plan MLX picks for `m` rows against an `n` by `k` weight, when it is one this port lacks.
pub fn unported(g: matmul.Gpu, m: usize, n: usize, k: usize, group: usize) ?Unported {
    if (m < vectorLimit(g, k, n)) return .vector_kernels;
    if (splitsK(m, n, k, group)) return .split_k;
    return null;
}

/// `qmm_splitk`'s choice of `split_k`: more than one means MLX takes the split-K tiles.
fn splitsK(m: usize, n: usize, k: usize, group: usize) bool {
    const tiles = ((n + 31) / 32) * ((m + 31) / 32);
    var split = @max(1, 512 / tiles);
    const align_k = if (group > 32) group else 32;
    split = @min(split, k / align_k);
    while (split > 1 and k % (split * align_k) != 0) split -= 1;
    return split > 1;
}

/// The matrix as MLX's `ensure_row_contiguous_matrix` leaves it.
fn matrixRows(t: *Tensor) Error!*Tensor {
    const rows_laid_out = t.strides[0] == @as(usize, @intCast(t.shape[1])) and t.strides[1] == 1;
    return if (rows_laid_out or t.rowContiguous()) core.share(t) else ops.contiguousCopy(t);
}
