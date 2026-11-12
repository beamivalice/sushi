//! MLX's cumulative sum on the engine: `Scan::eval_gpu` and `scan_gpu_inplace` (mlx/backend/metal/scan.cpp) over the
//! compiled kernels of `mlx.metallib`. A float scan taken in parts is other bytes than one in order, so this is MLX's
//! choice of kernel and its grid. Ported from MLX's Metal backend (MIT, see NOTICE).

const std = @import("std");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;

/// MLX's `get_2d_grid_dims` with a divisor: a shape as a grid of two axes, less the `divisor` elements one
/// threadgroup scans.
fn rowGrid(shape: []const c_int, divisor_in: usize) Error![2]usize {
    var divided: [core.max_dims]usize = undefined;
    var remaining = divisor_in;
    for (shape, 0..) |d, i| {
        const extent: usize = @intCast(d);
        const factor = std.math.gcd(extent, remaining);
        divided[i] = extent / factor;
        remaining /= factor;
    }
    var x: usize = 1;
    var y: usize = 1;
    if (remaining == 1) {
        for (divided[0..shape.len]) |extent| {
            if (x * extent < std.math.maxInt(u32)) x *= extent else y *= extent;
        }
        if (x > std.math.maxInt(u32) or y > std.math.maxInt(u32)) return error.TensorShape;
        return .{ @max(x, y), @min(x, y) };
    }
    return error.TensorUnsupported;
}

/// The axes of `t` from the slowest in memory to the fastest, when `t` reads every element of one block once: what
/// MLX's `contiguous` flag says of an array. Null for a view with gaps or a broadcast.
fn memoryOrder(t: *const Tensor) ?[core.max_dims]u8 {
    var order: [core.max_dims]u8 = undefined;
    for (order[0..t.ndim], 0..) |*o, i| o.* = @intCast(i);
    const Ctx = struct {
        fn slower(strides: *const [core.max_dims]usize, a: u8, b: u8) bool {
            return strides[a] > strides[b];
        }
    };
    std.sort.insertion(u8, order[0..t.ndim], &t.strides, Ctx.slower);
    var want: usize = 1;
    var i: usize = t.ndim;
    while (i > 0) {
        i -= 1;
        const extent: usize = @intCast(t.shape[order[i]]);
        if (extent == 1) continue;
        if (t.strides[order[i]] != want) return null;
        want *= extent;
    }
    return order;
}

/// `t` summed cumulatively along `axis_in`, `inclusive` of each element or not, from the end with `reverse`. As MLX
/// does, a view that fills its memory is scanned where it lies and any other as a row-major copy; a scanned axis whose
/// stride is one element takes the kernel of contiguous rows, any other stride the strided one.
pub fn cumsum(t_in: *Tensor, axis_in: c_int, reverse: bool, inclusive: bool) Error!*Tensor {
    if (t_in.ndim == 0) return error.TensorShape;
    switch (t_in.dtype) {
        .float16, .bfloat16, .float32, .int32, .uint32 => {},
        else => return error.TensorDtype,
    }
    const ndim: usize = t_in.ndim;
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t_in.ndim) else axis_in);
    if (axis >= ndim) return error.TensorShape;
    if (t_in.size() == 0) return core.fresh(t_in.dtype, t_in.dims());

    // One element along the axis has no stride to go by: MLX's own is then that of a row-major array.
    const lies = if (t_in.shape[axis] == 1) null else memoryOrder(t_in);
    const t = if (lies == null) try ops.contiguous(t_in) else try core.share(t_in);
    defer core.release(t);
    var order: [core.max_dims]u8 = undefined;
    for (order[0..ndim], 0..) |*o, i| o.* = @intCast(i);
    if (lies) |found| order = found;

    // The output has the input's layout: row-major over the axes in their order in memory, then turned back.
    var laid: [core.max_dims]c_int = undefined;
    var back: [core.max_dims]c_int = undefined;
    for (order[0..ndim], 0..) |from, i| {
        laid[i] = t.shape[from];
        back[from] = @intCast(i);
    }
    const out = try core.fresh(t.dtype, laid[0..ndim]);
    defer core.release(out);

    const size: usize = @intCast(t.shape[axis]);
    var stride: usize = 1;
    for (order[0..ndim], 0..) |from, i| {
        if (from == axis) for (laid[i + 1 .. ndim]) |d| {
            stride *= @intCast(d);
        };
    }
    const n_reads: usize = if (core.itemSize(t.dtype) <= 4) 4 else 2;
    var buf: [96]u8 = undefined;
    const name = try stock.typeName(t.dtype);
    const k = try stock.kernel(try stock.named(&buf, "{s}_scan_{s}{s}_sum_{s}_{s}", .{ if (stride == 1) "contig" else "strided", if (reverse) "reverse_" else "", if (inclusive) "inclusive" else "exclusive", name, name }));
    var group: usize = undefined;
    var rows: [2]usize = undefined;
    const blocks = (stride + 31) / 32;
    if (stride == 1) {
        const per_simd = n_reads * 32;
        group = k.max_threads;
        if (size <= n_reads * 1024) {
            group = ((size + per_simd - 1) / per_simd) * 32;
        } else if (size <= n_reads * 2048) {
            group = ((size / 2 + per_simd - 1) / per_simd) * 32;
        }
        group = @min(group, k.max_threads);
        rows = try rowGrid(t.dims(), size);
    } else {
        group = (32 / n_reads) * 32;
        if (group > k.max_threads) return error.TensorUnsupported;
        rows = try rowGrid(t.dims(), size * stride);
        if (rows[0] * blocks <= std.math.maxInt(u32)) rows[0] *= blocks else rows[1] *= blocks;
    }
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, t, 0);
    core.bindOut(p, out, 1);
    p.constant(size, 2);
    if (stride != 1) {
        p.constant(stride, 3);
        p.constant(blocks, 4);
    }
    core.dispatch(p, .{ .x = group, .y = rows[0], .z = rows[1] }, .{ .x = group });
    return ops.transpose(out, back[0..ndim]);
}
