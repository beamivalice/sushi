//! MLX's quantized matrix product on the engine: `quantized_matmul` (mlx/ops.cpp) and `QuantizedMatmul::eval_gpu`
//! with `dispatch_qmv`, `qmv_quad`, `qmv_wide`, `qmv`, `qmm_splitk`, `qmm` and `qmm_nax`
//! (mlx/backend/metal/quantized.cpp) over the compiled kernels of `mlx.metallib`, so a product here writes the bytes it
//! writes under MLX; `gather_qmm` and `GatherQMM::eval_gpu` the same way, with `gather_qmv`, `gather_qmm{,_nax}` and
//! `gather_qmm_rhs{,_nax}`. Ported from MLX's Metal backend (MIT, see NOTICE). Ported: a transposed weight in the
//! affine, `mxfp4` or `mxfp8` mode, every plan MLX picks for it, one weight or a batch of them. Refused by name, each
//! with its reason (`core.refusing`), and reached by no caller: a weight that is not transposed (`qvm`, `qmm_n`,
//! `gather_qvm`), `nvfp4` (its global scale), x batch axes MLX would broadcast to w's, and
//! indices of an integer type other than int32 and uint32.

const std = @import("std");
const core = @import("core.zig");
const matmul = @import("matmul.zig");
const ops = @import("ops.zig");
const reduce = @import("reduce.zig");
const stock = @import("stock.zig");
const metal = @import("../metal.zig");

const Tensor = core.Tensor;
const Error = core.Error;

pub const Mode = enum {
    affine,
    mxfp4,
    mxfp8,

    fn fp(m: Mode) bool {
        return m != .affine;
    }
};

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

/// The vector limit on this GPU, for a test to probe both sides of it.
pub fn vectorLimitHere(k: usize, n: usize) !usize {
    return vectorLimit(try matmul.gpu(), k, n);
}

/// `get_pack_factor(bits, 32)`: codes a u32 holds.
fn packFactor(bits: usize) usize {
    return if (bits == 3 or bits == 5) 8 else if (bits == 6) 4 else 32 / bits;
}

/// `qmv_fast_k_alignment`: the K step of `qmv_fast`.
fn fastKAlignment(bits: usize) usize {
    return packFactor(bits) * @as(usize, if (bits == 2) 1 else 2) * 32;
}

/// The operands of one product once MLX's checks and `ensure_row_contiguous_matrix` have run.
const Product = struct {
    x: *Tensor,
    w: *Tensor,
    scales: *Tensor,
    biases: ?*Tensor,
    out: *Tensor,
    mode: Mode,
    group: usize,
    bits: usize,
    m: usize,
    n: usize,
    k: usize,
    /// Weights in the batch: `out.size() / M / N`.
    b: usize,

    fn tname(self: *const Product) Error![]const u8 {
        return stock.cType(self.x.dtype);
    }

    /// The weight, scales and (affine) biases at 0, 1 and 2.
    fn bindWeights(self: *const Product, p: metal.Pass) usize {
        core.bind(p, self.w, 0);
        core.bind(p, self.scales, 1);
        if (self.biases) |b| {
            core.bind(p, b, 2);
            return 3;
        }
        return 2;
    }

    /// `add_strides_and_shapes`, skipped for a single weight.
    fn bindBatch(self: *const Product, p: metal.Pass, at: usize) void {
        if (self.b > 1) _ = self.bindStrides(p, at);
    }

    /// `add_strides_and_shapes` from slot `at`; the next free slot.
    fn bindStrides(self: *const Product, p: metal.Pass, at: usize) usize {
        p.constant(@as(c_int, @intCast(self.x.ndim - 2)), at);
        p.bytes(std.mem.sliceAsBytes(self.x.dims()), at + 1);
        p.bytes(std.mem.sliceAsBytes(self.x.strides[0..self.x.ndim]), at + 2);
        p.constant(@as(c_int, @intCast(self.w.ndim - 2)), at + 3);
        p.bytes(std.mem.sliceAsBytes(self.w.dims()), at + 4);
        p.bytes(std.mem.sliceAsBytes(self.w.strides[0..self.w.ndim]), at + 5);
        p.bytes(std.mem.sliceAsBytes(self.scales.strides[0..self.scales.ndim]), at + 6);
        const b = self.biases orelse return at + 7;
        p.bytes(std.mem.sliceAsBytes(b.strides[0..b.ndim]), at + 7);
        return at + 8;
    }

    fn batchTag(self: *const Product) []const u8 {
        return if (self.b > 1) "_batch_1" else "_batch_0";
    }
};

/// `x @ w.T` for the codes `w` (uint32, `[..., n, k * bits / 32]`) with a scale per group of `group` inputs, and in the
/// affine mode a bias per group, as `mlx::core::quantized_matmul` gives it.
pub fn quantizedMatmul(x_in: *Tensor, w_in: *Tensor, scales_in: *Tensor, biases_in: ?*Tensor, mode: Mode, group: usize, bits: usize) Error!*Tensor {
    const k, const n = try validate(x_in, w_in, scales_in, biases_in, mode, group, bits);
    // `broadcast_arrays` over the batch axes when both carry them: this port takes equal ones only.
    const batched_w = w_in.ndim > 2;
    if (batched_w and (x_in.ndim != w_in.ndim or !std.mem.eql(c_int, x_in.shape[0 .. x_in.ndim - 2], w_in.shape[0 .. w_in.ndim - 2])))
        return core.refusing("x's batch axes must equal w's: MLX's broadcast of them is not ported", error.TensorUnsupported);
    if (batched_w and !std.mem.eql(c_int, scales_in.shape[0 .. scales_in.ndim - 2], w_in.shape[0 .. w_in.ndim - 2])) return error.TensorShape;

    var out_shape: [core.max_dims]c_int = undefined;
    @memcpy(out_shape[0..x_in.ndim], x_in.dims());
    out_shape[x_in.ndim - 1] = @intCast(n);

    const x = try matrixRows(x_in);
    defer core.release(x);
    const w = try matrixRows(w_in);
    defer core.release(w);
    const scales = try matrixRows(scales_in);
    defer core.release(scales);
    const biases: ?*Tensor = if (biases_in) |b| try matrixRows(b) else null;
    defer if (biases) |b| core.release(b);

    const non_batched = w.ndim == 2 and x.rowContiguous();
    const m: usize = if (non_batched) x.size() / k else @intCast(x.shape[x.ndim - 2]);
    const out = try core.fresh(x.dtype, out_shape[0..x.ndim]);
    errdefer core.release(out);
    if (m == 0) return out;
    var pr: Product = .{ .x = x, .w = w, .scales = scales, .biases = biases, .out = out, .mode = mode, .group = group, .bits = bits, .m = m, .n = n, .k = k, .b = out.size() / m / n };

    const g = try matmul.gpu();
    if (m >= vectorLimit(g, k, n)) {
        if (pr.b == 1) try qmmSplitK(&pr, g) else try qmm(&pr, g);
    } else if ((k == 128 or k == 64) and std.math.isPowerOfTwo(bits)) {
        try qmvQuad(&pr);
    } else if (m >= 2 and (mode.fp() or g.gen >= 15)) {
        try qmvWide(&pr);
    } else {
        try qmv(&pr);
    }
    return out;
}

/// `x @ w[rhs].T` per index pair, MLX's `gather_qmm` (mlx/ops.cpp) and `GatherQMM::eval_gpu` for a transposed weight
/// `[E, n, k * bits / 32]`: `lhs` picks the batch of `x` and `rhs` the weight, each one the default `arange` over its
/// batch when not given; `sorted` (the indices given are in order) lets MLX take the sorted kernels.
pub fn gatherQmm(x_in: *Tensor, w_in: *Tensor, scales_in: *Tensor, biases_in: ?*Tensor, lhs_in: ?*Tensor, rhs_in: ?*Tensor, mode: Mode, group: usize, bits: usize, sorted: bool) Error!*Tensor {
    if (lhs_in == null and rhs_in == null) return quantizedMatmul(x_in, w_in, scales_in, biases_in, mode, group, bits);
    if (x_in.ndim < 2 or w_in.ndim < 3) return error.TensorShape;
    const k, const n = try validate(x_in, w_in, scales_in, biases_in, mode, group, bits);

    const lhs0 = try indicesOrDefault(lhs_in, x_in);
    defer core.release(lhs0);
    const rhs0 = try indicesOrDefault(rhs_in, w_in);
    defer core.release(rhs0);
    var idx_shape: [core.max_dims]c_int = undefined;
    const idx_dims = try broadcastShape(lhs0, rhs0, &idx_shape);
    // A 0-d pair sends MLX's kernels an empty vector; no caller gathers without a batch.
    if (idx_dims.len == 0) return core.refusing("indices with no axes are not ported", error.TensorUnsupported);
    const lhs = try ops.broadcastTo(lhs0, idx_dims);
    defer core.release(lhs);
    const rhs = try ops.broadcastTo(rhs0, idx_dims);
    defer core.release(rhs);
    const right_sorted = sorted and lhs_in == null;

    const m: usize = @intCast(x_in.shape[x_in.ndim - 2]);
    var out_shape = try core.Dims.of(idx_dims);
    try out_shape.appendSlice(&.{ @intCast(m), @intCast(n) });
    const out = try core.fresh(x_in.dtype, out_shape.slice());
    errdefer core.release(out);
    if (out.size() == 0) return out;

    const x = try matrixRows(x_in);
    defer core.release(x);
    const w = try matrixRows(w_in);
    defer core.release(w);
    const scales = try matrixRows(scales_in);
    defer core.release(scales);
    const biases: ?*Tensor = if (biases_in) |b| try matrixRows(b) else null;
    defer if (biases) |b| core.release(b);

    const b_count = out.size() / m / n;
    const e = w.size() / @as(usize, @intCast(w.shape[w.ndim - 1])) / @as(usize, @intCast(w.shape[w.ndim - 2]));
    const pr: Product = .{ .x = x, .w = w, .scales = scales, .biases = biases, .out = out, .mode = mode, .group = group, .bits = bits, .m = m, .n = n, .k = k, .b = b_count };
    const g = try matmul.gpu();
    if (m == 1 and b_count >= 16 and right_sorted and b_count / e >= 4) {
        try gatherQmmRhs(&pr, x_in, w_in, scales_in, biases_in, rhs, e, g);
    } else if (m >= vectorLimit(g, k, n)) {
        try gatherQmmTiles(&pr, lhs, rhs, g);
    } else {
        try gatherQmv(&pr, lhs, rhs);
    }
    return out;
}

/// `indices_or_default`: the indices given as uint32, or `arange` over the batch of `of`.
fn indicesOrDefault(given: ?*Tensor, of: *const Tensor) Error!*Tensor {
    if (given) |t| {
        switch (t.dtype) {
            .uint32 => return core.share(t),
            .int32 => return ops.astype(t, .uint32),
            else => return core.refusing("indices must be int32 or uint32: MLX's cast of other integers is not ported", error.TensorDtype),
        }
    }
    const batch = of.shape[0 .. of.ndim - 2];
    var total: usize = 1;
    for (batch) |d| total *= @intCast(d);
    const flat = ops.arange(0, @floatFromInt(total), 1, .uint32) catch |err| return switch (err) {
        error.ArangeLengthUndefined, error.ArangeStepZero, error.ArangeTooLong, error.ArangeOutOfRange => error.TensorShape,
        else => |e2| e2,
    };
    defer core.release(flat);
    return ops.reshape(flat, batch);
}

/// The broadcast shape of two index arrays, aligned at their last axes.
fn broadcastShape(a: *const Tensor, b: *const Tensor, out: *[core.max_dims]c_int) Error![]const c_int {
    const nd = @max(a.ndim, b.ndim);
    for (0..nd) |i| {
        const da: c_int = if (i < nd - a.ndim) 1 else a.shape[i - (nd - a.ndim)];
        const db: c_int = if (i < nd - b.ndim) 1 else b.shape[i - (nd - b.ndim)];
        if (da != db and da != 1 and db != 1) return error.TensorShape;
        out[i] = if (da == 1) db else da;
    }
    return out[0..nd];
}

/// `add_gather_strides_and_shapes`: the pair's shape collapsed over runs both walk alike, then both strides.
fn bindGatherStrides(p: metal.Pass, lhs: *const Tensor, rhs: *const Tensor, at: usize) void {
    const nd = lhs.ndim;
    const cut: i32 = -1;
    var plan: [2 * core.max_dims + 2]i32 = undefined;
    var len: usize = 0;
    if (lhs.shape[0] != 1) {
        plan[len] = 0;
        len += 1;
    }
    var size: i64 = lhs.shape[0];
    for (1..nd) |i| {
        size *= lhs.shape[i];
        const run = for ([_]*const Tensor{ lhs, rhs }) |t| {
            if (t.strides[i] * @as(usize, @intCast(lhs.shape[i])) != t.strides[i - 1] or size > std.math.maxInt(i32)) break false;
        } else true;
        if (!run) {
            size = lhs.shape[i];
            plan[len] = cut;
            len += 1;
        }
        if (lhs.shape[i] != 1) {
            plan[len] = @intCast(i);
            len += 1;
        }
    }
    plan[len] = cut;
    len += 1;
    var shape: [core.max_dims]c_int = undefined;
    var sa: [core.max_dims]i64 = undefined;
    var sb: [core.max_dims]i64 = undefined;
    var n: usize = 0;
    var i: usize = 0;
    while (true) {
        while (i < len and plan[i] == cut) i += 1;
        if (i == len) break;
        var extent: c_int = lhs.shape[@intCast(plan[i])];
        var j = i + 1;
        while (plan[j] != cut) : (j += 1) extent *= lhs.shape[@intCast(plan[j])];
        const last: usize = @intCast(plan[j - 1]);
        shape[n] = extent;
        sa[n] = @intCast(lhs.strides[last]);
        sb[n] = @intCast(rhs.strides[last]);
        n += 1;
        i = j + 1;
    }
    if (n == 0) {
        shape[0] = 1;
        sa[0] = 0;
        sb[0] = 0;
        n = 1;
    }
    p.constant(@as(c_int, @intCast(n)), at);
    p.bytes(std.mem.sliceAsBytes(shape[0..n]), at + 1);
    p.bytes(std.mem.sliceAsBytes(sa[0..n]), at + 2);
    p.bytes(std.mem.sliceAsBytes(sb[0..n]), at + 3);
}

fn gatherQmv(pr: *const Product, lhs: *const Tensor, rhs: *const Tensor) Error!void {
    const bn = 8;
    const fast = pr.n % bn == 0 and pr.k % fastKAlignment(pr.bits) == 0;
    var buf: [160]u8 = undefined;
    const name = try stock.named(&buf, "{s}_{s}_{s}_gs_{d}_b_{d}", .{ @tagName(pr.mode), if (fast) "gather_qmv_fast" else "gather_qmv", try pr.tname(), pr.group, pr.bits });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    _ = pr.bindWeights(p);
    core.bind(p, pr.x, 3);
    core.bind(p, lhs, 4);
    core.bind(p, rhs, 5);
    core.bindOut(p, pr.out, 6);
    p.constant(@as(c_int, @intCast(pr.k)), 7);
    p.constant(@as(c_int, @intCast(pr.n)), 8);
    bindGatherStrides(p, lhs, rhs, pr.bindStrides(p, 9));
    core.dispatchGroups(p, .{ .x = pr.m, .y = (pr.n + bn - 1) / bn, .z = pr.b }, .{ .x = 32, .y = 2, .z = 1 });
}

/// `gather_qmm`, and `gather_qmm_nax` where it hands over: matrix units, K a whole 64, f32 only with TF32.
fn gatherQmmTiles(pr: *const Product, lhs: *const Tensor, rhs: *const Tensor, g: matmul.Gpu) Error!void {
    const nax = g.nax and pr.k % 64 == 0 and (pr.x.dtype != .float32 or g.tf32);
    const bm: usize = if (nax) 64 else 32;
    const bn: usize = if (nax) 64 else 32;
    const aligned = pr.n % bn == 0;
    var buf: [192]u8 = undefined;
    const name = if (nax)
        try stock.named(&buf, "{s}_gather_qmm_t_nax_{s}_gs_{d}_b_{d}_bm64_bn64_bk64_wm2_wn2_alN_{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, if (aligned) "true" else "false" })
    else
        try stock.named(&buf, "{s}_gather_qmm_t_{s}_gs_{d}_b_{d}_alN_{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, if (aligned) "true" else "false" });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    _ = pr.bindWeights(p);
    core.bind(p, pr.x, 3);
    core.bind(p, lhs, 4);
    core.bind(p, rhs, 5);
    core.bindOut(p, pr.out, 6);
    p.constant(@as(c_int, @intCast(pr.k)), 7);
    p.constant(@as(c_int, @intCast(pr.n)), 8);
    p.constant(@as(c_int, @intCast(pr.m)), 9);
    bindGatherStrides(p, lhs, rhs, pr.bindStrides(p, 10));
    core.dispatchGroups(p, .{ .x = (pr.n + bn - 1) / bn, .y = (pr.m + bm - 1) / bm, .z = pr.b }, .{ .x = 32, .y = 2, .z = 2 });
}

/// `gather_qmm_rhs` (and `_nax`): one row a pair with the weights in order, the rows of each weight run together after
/// `gather_mm_offsets` finds where each weight's run starts. Its operands are laid out whole, not only by matrix.
fn gatherQmmRhs(pr: *const Product, x_in: *Tensor, w_in: *Tensor, scales_in: *Tensor, biases_in: ?*Tensor, rhs_view: *Tensor, e: usize, g: matmul.Gpu) Error!void {
    const indices = try ops.contiguous(rhs_view);
    defer core.release(indices);
    // `broadcast_with_indices`: x as one matrix a pair.
    const x = blk: {
        const per = x_in.size() / @as(usize, @intCast(x_in.shape[x_in.ndim - 2])) / @as(usize, @intCast(x_in.shape[x_in.ndim - 1]));
        if (per == indices.size()) break :blk try ops.contiguous(x_in);
        var shape = try core.Dims.of(indices.dims());
        try shape.appendSlice(x_in.dims()[x_in.ndim - 2 ..]);
        const view = try ops.broadcastTo(x_in, shape.slice());
        defer core.release(view);
        break :blk try ops.contiguousCopy(view);
    };
    defer core.release(x);
    const w = try ops.contiguous(w_in);
    defer core.release(w);
    const scales = try ops.contiguous(scales_in);
    defer core.release(scales);
    const biases: ?*Tensor = if (biases_in) |b| try ops.contiguous(b) else null;
    defer if (biases) |b| core.release(b);

    const rows = x.size() / pr.k;
    const nax = g.nax and (x.dtype != .float32 or g.tf32);
    const bm: usize = if (!nax) 16 else if (rows / e < 64) 32 else 64;
    const bn: usize = if (nax) 64 else 32;
    const bk: usize = if (nax) 64 else 32;
    const wm: usize = if (nax) 2 else 1;
    const wn: usize = 2;
    const align_n = pr.n % bn == 0;
    const align_k = pr.k % bk == 0;
    var name_buf: [192]u8 = undefined;
    const name = try stock.named(&name_buf, "{s}_gather_qmm_rhs{s}_nt_{s}_gs_{d}_b_{d}_bm_{d}_bn_{d}_bk_{d}_wm_{d}_wn_{d}", .{ @tagName(pr.mode), if (nax) "_nax" else "", try pr.tname(), pr.group, pr.bits, bm, bn, bk, wm, wn });
    var hash_buf: [224]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_align_N_{c}_align_K_{c}", .{ name, @as(u8, if (align_n) 't' else 'n'), @as(u8, if (align_k) 't' else 'n') });
    const kern = try stock.kernelFlags(name, hash, &.{ .{ .index = 201, .value = align_n }, .{ .index = 202, .value = align_k } });

    const offsets = try gatherMmOffsets(indices, e, rows);
    defer core.release(offsets);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, x, 0);
    core.bind(p, w, 1);
    core.bind(p, scales, 2);
    if (biases) |b| core.bind(p, b, 3);
    core.bind(p, offsets, 4);
    core.bindOut(p, pr.out, 5);
    p.constant(@as(c_int, @intCast(rows)), 6);
    p.constant(@as(c_int, @intCast(pr.n)), 7);
    p.constant(@as(c_int, @intCast(pr.k)), 8);
    p.constant(@as(c_int, @intCast(e)), 9);
    core.dispatchGroups(p, .{ .x = (pr.n + bn - 1) / bn, .y = @min(rows, (rows + bm - 1) / bm + e - 1), .z = 1 }, .{ .x = 32, .y = wn, .z = wm });
}

/// MLX's `gather_mm_offsets` (mlx/backend/metal/matmul.cpp): where each of `groups` weights' run of rows starts in
/// the sorted `indices`, int32.
fn gatherMmOffsets(indices: *const Tensor, groups: usize, rows: usize) Error!*Tensor {
    const out = try core.fresh(.int32, &.{@intCast(groups)});
    errdefer core.release(out);
    const kern = try stock.kernel("gather_mm_offsets");
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, indices, 0);
    core.bindOut(p, out, 1);
    p.constant(@as(c_int, @intCast(rows)), 2);
    core.dispatch(p, .{ .x = groups }, .{ .x = @min(groups, kern.max_threads) });
    return out;
}

/// MLX's `validate_mode_with_type` and `extract_quantized_matmul_dims` for a transposed weight: the inner and outer
/// width.
fn validate(x: *const Tensor, w: *const Tensor, scales: *const Tensor, biases: ?*const Tensor, mode: Mode, group: usize, bits: usize) Error!struct { usize, usize } {
    switch (x.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return core.refusing("a quantized product with an x that is not a float", error.TensorDtype),
    }
    if (w.dtype != .uint32 or x.ndim < 1 or w.ndim < 2 or scales.ndim != w.ndim) return error.TensorShape;
    switch (mode) {
        .affine => {
            const b = biases orelse return core.refusing("an affine weight needs its biases", error.TensorUnsupported);
            if (scales.dtype != x.dtype or b.dtype != x.dtype) return core.refusing("an affine product whose scales or biases are not x's dtype: the seam promotes them first", error.TensorDtype);
            if (group == 0 or !(bits >= 2 and bits <= 6 or bits == 8)) return core.refusing("affine bits are 2, 3, 4, 5, 6 or 8", error.TensorUnsupported);
        },
        .mxfp4, .mxfp8 => {
            if (biases != null) return core.refusing("an mx weight has no biases", error.TensorUnsupported);
            if (scales.dtype != .uint8) return core.refusing("an mx product whose scales are not uint8", error.TensorDtype);
            if (group != 32 or bits != @as(usize, if (mode == .mxfp4) 4 else 8)) return core.refusing("an mx mode takes group 32 and its own bits", error.TensorUnsupported);
        },
    }
    const k: usize = @intCast(x.shape[x.ndim - 1]);
    const n: usize = @intCast(w.shape[w.ndim - 2]);
    if (k == 0 or n == 0 or k != @as(usize, @intCast(w.shape[w.ndim - 1])) * 32 / bits or
        k != @as(usize, @intCast(scales.shape[scales.ndim - 1])) * group or scales.shape[scales.ndim - 2] != w.shape[w.ndim - 2]) return error.TensorShape;
    if (biases) |b| if (b.ndim != scales.ndim or !std.mem.eql(c_int, b.dims(), scales.dims())) return error.TensorShape;
    return .{ k, n };
}

fn qmvQuad(pr: *const Product) Error!void {
    const bn = 8 * 8;
    var buf: [160]u8 = undefined;
    const name = try stock.named(&buf, "{s}_qmv_quad_{s}_gs_{d}_b_{d}_d_{d}{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, pr.k, pr.batchTag() });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    var c = pr.bindWeights(p);
    core.bind(p, pr.x, c);
    core.bindOut(p, pr.out, c + 1);
    p.constant(@as(c_int, @intCast(pr.k)), c + 2);
    p.constant(@as(c_int, @intCast(pr.n)), c + 3);
    c += 4;
    pr.bindBatch(p, c);
    core.dispatchGroups(p, .{ .x = pr.m, .y = (pr.n + bn - 1) / bn, .z = pr.b }, .{ .x = 32, .y = 1, .z = 1 });
}

fn qmv(pr: *const Product) Error!void {
    const fast = pr.n % 8 == 0 and pr.k % fastKAlignment(pr.bits) == 0;
    // The narrow tile MLX keeps for `nvfp4` only; no other mode takes it.
    const bn = 2 * 4;
    var buf: [160]u8 = undefined;
    const name = try stock.named(&buf, "{s}_{s}_{s}_gs_{d}_b_{d}{s}", .{ @tagName(pr.mode), if (fast) "qmv_fast" else "qmv", try pr.tname(), pr.group, pr.bits, pr.batchTag() });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    _ = pr.bindWeights(p);
    core.bind(p, pr.x, 3);
    core.bindOut(p, pr.out, 4);
    p.constant(@as(c_int, @intCast(pr.k)), 5);
    p.constant(@as(c_int, @intCast(pr.n)), 6);
    pr.bindBatch(p, 7);
    core.dispatchGroups(p, .{ .x = pr.m, .y = (pr.n + bn - 1) / bn, .z = pr.b }, .{ .x = 32, .y = 2, .z = 1 });
}

fn qmvWide(pr: *const Product) Error!void {
    const n_tiles = (pr.m + 4) / 5;
    const vecs = (pr.m + n_tiles - 1) / n_tiles;
    const k_lanes: usize = if (pr.mode == .affine) 8 else 16;
    const simdgroups = 2;
    const rows_per_tg = (32 / k_lanes) * simdgroups;
    var buf: [160]u8 = undefined;
    const name = try stock.named(&buf, "{s}_qmv_wide_{s}_gs_{d}_b_{d}_nv_{d}_kl_{d}{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, vecs, k_lanes, pr.batchTag() });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    var c = pr.bindWeights(p);
    core.bind(p, pr.x, c);
    core.bindOut(p, pr.out, c + 1);
    p.constant(@as(c_int, @intCast(pr.k)), c + 2);
    p.constant(@as(c_int, @intCast(pr.n)), c + 3);
    p.constant(@as(c_int, @intCast(pr.m)), c + 4);
    c += 5;
    pr.bindBatch(p, c);
    core.dispatchGroups(p, .{ .x = (pr.m + vecs - 1) / vecs, .y = (pr.n + rows_per_tg - 1) / rows_per_tg, .z = pr.b }, .{ .x = 32, .y = simdgroups, .z = 1 });
}

/// `qmm_splitk`: K cut into parts that write a `[split, M, N]` buffer, summed over the parts; one part is `qmm`. A
/// split-K product has one batch, so `out` is those `M x N` rows whatever axes the caller gave them.
fn qmmSplitK(pr: *const Product, g: matmul.Gpu) Error!void {
    const n_tiles = (pr.n + 31) / 32;
    const m_tiles = (pr.m + 31) / 32;
    var split = @max(1, 512 / (n_tiles * m_tiles));
    const k_align = if (pr.group > 32) pr.group else 32;
    split = @min(split, pr.k / k_align);
    while (split > 1 and pr.k % (split * k_align) != 0) split -= 1;
    if (split <= 1) return qmm(pr, g);

    std.debug.assert(pr.b == 1);
    const parts = try core.fresh(pr.x.dtype, &.{ @intCast(split), @intCast(pr.m), @intCast(pr.n) });
    defer core.release(parts);

    const aligned = pr.n % 32 == 0;
    var buf: [160]u8 = undefined;
    const name = try stock.named(&buf, "{s}_qmm_t_splitk_{s}_gs_{d}_b_{d}_alN_{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, if (aligned) "true" else "false" });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    const c = pr.bindWeights(p);
    core.bind(p, pr.x, c);
    core.bindOut(p, parts, c + 1);
    p.constant(@as(c_int, @intCast(pr.k)), c + 2);
    p.constant(@as(c_int, @intCast(pr.n)), c + 3);
    p.constant(@as(c_int, @intCast(pr.m)), c + 4);
    p.constant(@as(c_int, @intCast(pr.k / split)), c + 5);
    p.constant(@as(c_int, @intCast(pr.m * pr.n)), c + 6);
    core.dispatchGroups(p, .{ .x = n_tiles, .y = m_tiles, .z = split }, .{ .x = 32, .y = 2, .z = 2 });
    try reduce.strided(.sum, parts, pr.out, 0);
}

/// `qmm`, and `qmm_nax` where it hands over: matrix units, K a whole 64, f32 only with TF32.
fn qmm(pr: *const Product, g: matmul.Gpu) Error!void {
    const nax = g.nax and pr.k % 64 == 0 and (pr.x.dtype != .float32 or g.tf32);
    const bm: usize = if (!nax) 32 else if (pr.m <= 32) 32 else 64;
    const bn: usize = if (nax) 64 else 32;
    const aligned = pr.n % bn == 0;
    var buf: [192]u8 = undefined;
    const name = if (nax)
        try stock.named(&buf, "{s}_qmm_t_nax_{s}_gs_{d}_b_{d}_bm{d}_bn64_bk64_wm2_wn2_alN_{s}{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, bm, if (aligned) "true" else "false", pr.batchTag() })
    else
        try stock.named(&buf, "{s}_qmm_t_{s}_gs_{d}_b_{d}_alN_{s}{s}", .{ @tagName(pr.mode), try pr.tname(), pr.group, pr.bits, if (aligned) "true" else "false", pr.batchTag() });
    const p = try core.pass();
    p.pipeline(try stock.kernel(name));
    _ = pr.bindWeights(p);
    // A transposed product leaves slot 2 to the biases or the global scale, bound or not.
    core.bind(p, pr.x, 3);
    core.bindOut(p, pr.out, 4);
    p.constant(@as(c_int, @intCast(pr.k)), 5);
    p.constant(@as(c_int, @intCast(pr.n)), 6);
    p.constant(@as(c_int, @intCast(pr.m)), 7);
    pr.bindBatch(p, 8);
    core.dispatchGroups(p, .{ .x = (pr.n + bn - 1) / bn, .y = (pr.m + bm - 1) / bm, .z = pr.b }, .{ .x = 32, .y = 2, .z = 2 });
}

/// The operand as MLX's `ensure_row_contiguous_matrix` leaves it: its last two axes laid out as rows.
fn matrixRows(t: *Tensor) Error!*Tensor {
    if (t.rowContiguous()) return core.share(t);
    const last_rows = if (t.ndim < 2)
        t.strides[0] == 1
    else
        t.strides[t.ndim - 2] == @as(usize, @intCast(t.shape[t.ndim - 1])) and t.strides[t.ndim - 1] == 1;
    return if (last_rows) core.share(t) else ops.contiguousCopy(t);
}
