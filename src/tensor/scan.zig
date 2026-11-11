//! MLX's cumulative sum on the engine: `scan_gpu_inplace` for a scan along the last axis
//! (mlx/backend/metal/scan.cpp) over the compiled kernels of `mlx.metallib`. A float scan taken in parts is other
//! bytes than one in order, so this is MLX's kernel and grid. Ported from MLX's Metal backend (MIT, see NOTICE).

const std = @import("std");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;

/// MLX's `get_2d_grid_dims` with a divisor: the rows of a shape as a grid of two axes, `divisor` being the extent
/// of the scanned axis.
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

/// `t` summed cumulatively along its last axis, `inclusive` of each element or not, from the end with `reverse`.
pub fn cumsum(t_in: *Tensor, axis_in: c_int, reverse: bool, inclusive: bool) Error!*Tensor {
    if (t_in.ndim == 0) return error.TensorShape;
    switch (t_in.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t_in.ndim) else axis_in);
    if (axis != t_in.ndim - 1) return error.TensorUnsupported;
    const t = try ops.contiguous(t_in);
    defer core.release(t);
    const out = try core.fresh(t.dtype, t.dims());
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const size: usize = @intCast(t.shape[axis]);
    var buf: [96]u8 = undefined;
    const name = try stock.typeName(t.dtype);
    const k = try stock.kernel(try stock.named(&buf, "contig_scan_{s}{s}_sum_{s}_{s}", .{ if (reverse) "reverse_" else "", if (inclusive) "inclusive" else "exclusive", name, name }));
    const n_reads: usize = if (core.itemSize(t.dtype) <= 4) 4 else 2;
    const per_simd = n_reads * 32;
    var group: usize = k.max_threads;
    if (size <= n_reads * 1024) {
        group = ((size + per_simd - 1) / per_simd) * 32;
    } else if (size <= n_reads * 2048) {
        group = ((size / 2 + per_simd - 1) / per_simd) * 32;
    }
    group = @min(group, k.max_threads);
    // MLX divides by the scanned axis's extent over the strides of the array, which are row-major here.
    const rows = try rowGrid(t.dims(), size);
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, t, 0);
    core.bindOut(p, out, 1);
    p.constant(size, 2);
    core.dispatch(p, .{ .x = group, .y = rows[0], .z = rows[1] }, .{ .x = group });
    return out;
}
