//! MLX's sum, max and argmax over one axis on the engine: `Reduce::eval_gpu` and its plan (mlx/backend/metal/reduce.cpp,
//! mlx/backend/common/reduce.cpp) over the compiled kernels of `mlx.metallib`. A float sum depends on the order it is
//! taken in, and MLX takes a long row or column in parallel parts, so a reduction here is MLX's kernel under MLX's
//! plan; argmax is `ArgReduce::eval_gpu` (mlx/backend/metal/primitives.cpp).
//! Ported from MLX's Metal backend (MIT, see NOTICE). An input past 2^31 elements is not ported.

const std = @import("std");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;
const max_dims = core.max_dims;

const n_reads = 4;
const n_writes = 4;

/// MLX's reduction plans for one axis.
const Plan = enum { all, row, row_general, strided, strided_general, general };

fn planOf(x: *const Tensor, axis: usize) Plan {
    const stride = x.strides[axis];
    const row_contiguous = x.rowContiguous();
    if (x.ndim == 1 and x.size() * core.itemSize(x.dtype) == x.store.len and stride == 1) return .all;
    if (row_contiguous) return if (stride == 1) .row else .strided;
    if (stride == 1) return .row_general;
    if (stride > 1) {
        // The axes after the reduced one in memory are one run, so the column kernels walk them.
        var size: usize = 1;
        var have_expand = false;
        var i = x.ndim;
        while (i > 0) {
            i -= 1;
            if (i == axis) continue;
            const extent: usize = @intCast(x.shape[i]);
            if (x.strides[i] == 0) {
                if (extent == 1) continue;
                have_expand = true;
                break;
            }
            if (x.strides[i] != size and extent != 1) break;
            size *= extent;
        }
        if (size > stride or (size == stride and !have_expand)) return .strided_general;
    }
    return .general;
}

/// A shape and its strides as the kernels walk them, MLX's `collapse_contiguous_dims` of one array.
const Walk = struct {
    n: usize = 0,
    shape: [max_dims]c_int = @splat(0),
    strides: [max_dims]i64 = @splat(0),

    fn without(x: *const Tensor, axis: usize) Walk {
        var w: Walk = .{};
        for (0..x.ndim) |i| {
            if (i == axis) continue;
            w.shape[w.n] = x.shape[i];
            w.strides[w.n] = @intCast(x.strides[i]);
            w.n += 1;
        }
        return w;
    }

    fn pop(w: *Walk) void {
        w.n -= 1;
        w.shape[w.n] = 0;
        w.strides[w.n] = 0;
    }

    fn collapsed(w: Walk) Walk {
        var out: Walk = .{};
        if (w.n == 0) return out;
        out.shape[0] = w.shape[0];
        out.strides[0] = w.strides[0];
        out.n = 1;
        for (1..w.n) |i| {
            if (w.shape[i] == 1) continue;
            const last = out.n - 1;
            if (w.strides[i] * w.shape[i] != out.strides[last] or @as(i64, out.shape[last]) * w.shape[i] > std.math.maxInt(i32)) {
                out.shape[out.n] = w.shape[i];
                out.strides[out.n] = w.strides[i];
                out.n += 1;
            } else {
                out.shape[last] *= w.shape[i];
                out.strides[last] = w.strides[i];
            }
        }
        return out;
    }

    /// The bytes of a vector the kernel indexes: one zero when the walk is empty, as MLX sends.
    fn count(w: *const Walk) usize {
        return @max(w.n, 1);
    }
};

/// MLX's `get_2d_grid_dims`: the elements of a shape as a grid of two axes, broadcast axes left out.
pub fn grid2d(shape: []const c_int, strides: []const usize) Error!metal.Size {
    var x: usize = 1;
    var y: usize = 1;
    for (shape, strides) |d, st| {
        if (st == 0) continue;
        const extent: usize = @intCast(d);
        if (x * extent < std.math.maxInt(u32)) x *= extent else y *= extent;
    }
    if (x > std.math.maxInt(u32) or y > std.math.maxInt(u32)) return error.TensorShape;
    return .{ .x = @max(x, y), .y = @min(x, y), .z = 1 };
}

fn threadsForRow(len: usize) usize {
    if (len <= 512) return 32;
    if (len <= 1024) return 128;
    return @min(1024, (((len + n_reads - 1) / n_reads + 31) / 32) * 32);
}

pub const Op = enum { sum, max };

fn kernelOf(comptime fmt: []const u8, op: Op, t: *const Tensor) Error!metal.Pipeline {
    var buf: [96]u8 = undefined;
    return stock.kernel(try stock.named(&buf, fmt ++ "{t}{s}", .{ op, try stock.typeName(t.dtype) }));
}

/// `t` reduced over `axis`, the axis kept with one element.
pub fn reduce(op: Op, t: *Tensor, axis: usize) Error!*Tensor {
    switch (t.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    if (t.size() == 0 or t.size() > std.math.maxInt(i32)) return error.TensorUnsupported;
    // One element along the axis: MLX copies.
    if (t.shape[axis] == 1) return ops.contiguousCopy(t);
    var kept: [max_dims]c_int = undefined;
    @memcpy(kept[0..t.ndim], t.dims());
    kept[axis] = 1;

    var x = try core.share(t);
    defer core.release(x);
    var plan = planOf(x, axis);
    if (plan == .general) {
        const copy = try ops.contiguousCopy(x);
        core.release(x);
        x = copy;
        plan = planOf(x, axis);
    }
    const out = try core.fresh(t.dtype, kept[0..t.ndim]);
    errdefer core.release(out);
    switch (plan) {
        .all => try all(op, x, out),
        .row, .row_general => try row(op, x, out, axis, plan == .row),
        .strided, .strided_general => try strided(op, x, out, axis),
        .general => unreachable,
    }
    return out;
}

/// MLX's `all_reduce_dispatch`: every element of the 1-D `in` into the one element of `out`.
pub fn all(op: Op, in: *Tensor, out: *Tensor) Error!void {
    const kern = try kernelOf("all_reduce_", op, in);
    const size = in.size();
    const p = try core.pass();
    p.pipeline(kern);
    if (size <= n_reads * 1024) {
        const threads = (((size + n_reads - 1) / n_reads + 31) / 32) * 32;
        core.bind(p, in, 0);
        core.bindOut(p, out, 1);
        p.constant(size, 2);
        p.constant(size, 3);
        core.dispatch(p, .{ .x = threads }, .{ .x = threads });
        return;
    }
    const small = size * core.itemSize(in.dtype) <= 1 << 26;
    const rows: usize = if (small) 32 * n_reads else 1024 * n_reads;
    const second: usize = if (small) 32 else 1024;
    const parts = try core.fresh(in.dtype, &.{@intCast(rows)});
    defer core.release(parts);
    const row_size = (size + rows - 1) / rows;
    const threads = ((@min((row_size + n_reads - 1) / n_reads, 1024) + 31) / 32) * 32;
    core.bind(p, in, 0);
    core.bindOut(p, parts, 1);
    p.constant(size, 2);
    p.constant(row_size, 3);
    core.dispatch(p, .{ .x = threads, .y = rows }, .{ .x = threads });
    p.pipeline(kern);
    core.bind(p, parts, 0);
    core.bindOut(p, out, 1);
    p.constant(rows, 2);
    p.constant(rows, 3);
    core.dispatch(p, .{ .x = second }, .{ .x = second });
}

/// MLX's `row_reduce_general_dispatch`: the axis is the one memory runs along.
fn row(op: Op, in: *Tensor, out: *Tensor, axis: usize, contiguous: bool) Error!void {
    const row_size: usize = @intCast(in.shape[axis]);
    const walk = Walk.without(in, axis).collapsed();
    const none: Walk = .{};
    const out_grid = try grid2d(out.dims(), out.strides[0..out.ndim]);
    const p = try core.pass();
    if (row_size > 64 and contiguous and in.size() / row_size >= 32) {
        p.pipeline(try kernelOf("row_reduce_simple_", op, in));
        const threads = threadsForRow(row_size);
        core.bind(p, in, 0);
        core.bindOut(p, out, 1);
        p.constant(row_size, 2);
        p.constant(out.size(), 3);
        core.dispatch(p, .{ .x = threads, .y = (out_grid.x + n_writes - 1) / n_writes, .z = out_grid.y }, .{ .x = threads });
        return;
    }
    const small = row_size <= 64;
    p.pipeline(if (small) try kernelOf("row_reduce_small_1_reduce_", op, in) else try kernelOf("row_reduce_looped_1_reduce_", op, in));
    core.bind(p, in, 0);
    core.bindOut(p, out, 1);
    p.constant(row_size, 2);
    p.constant(@as(usize, 1), 3);
    p.bytes(std.mem.sliceAsBytes(walk.shape[0..walk.count()]), 4);
    p.bytes(std.mem.sliceAsBytes(walk.strides[0..walk.count()]), 5);
    p.constant(@as(c_int, @intCast(walk.n)), 6);
    p.bytes(std.mem.sliceAsBytes(none.shape[0..1]), 7);
    p.bytes(std.mem.sliceAsBytes(none.strides[0..1]), 8);
    p.constant(@as(c_int, 0), 9);
    if (small) {
        // One reduction per output and no other reduced axis: MLX's first grid.
        core.dispatch(p, out_grid, .{ .x = @min(out_grid.x, 1024) });
    } else {
        const threads = threadsForRow(row_size);
        core.dispatch(p, .{ .x = threads, .y = out_grid.x, .z = out_grid.y }, .{ .x = threads });
    }
}

/// What the column kernels read at buffers 2 to 10, MLX's `ColReduceArgs::encode`.
const Column = struct {
    size: usize,
    stride: i64,
    walk: Walk,
    reduce_shape: [1]c_int = .{0},
    reduce_strides: [1]i64 = .{0},
    reduce_ndim: c_int = 0,

    fn encode(c: *const Column, p: metal.Pass) void {
        p.constant(c.size, 2);
        p.constant(c.stride, 3);
        p.bytes(std.mem.sliceAsBytes(c.walk.shape[0..c.walk.count()]), 4);
        p.bytes(std.mem.sliceAsBytes(c.walk.strides[0..c.walk.count()]), 5);
        p.constant(@as(c_int, @intCast(c.walk.n)), 6);
        p.bytes(std.mem.sliceAsBytes(&c.reduce_shape), 7);
        p.bytes(std.mem.sliceAsBytes(&c.reduce_strides), 8);
        p.constant(c.reduce_ndim, 9);
        p.constant(@as(usize, 1), 10);
    }
};

/// The parts of a long column summed into `out`, MLX's second pass over an intermediate of `blocks` rows.
fn foldParts(op: Op, p: metal.Pass, parts: *Tensor, out: *Tensor, blocks: usize, threads: usize) Error!void {
    const second: Column = .{ .size = blocks, .stride = @intCast(out.size()), .walk = .{}, .reduce_shape = .{@intCast(blocks)}, .reduce_strides = .{@intCast(out.size())}, .reduce_ndim = 1 };
    p.pipeline(try kernelOf("col_reduce_looped_1_32_32_reduce_", op, parts));
    core.bind(p, parts, 0);
    core.bindOut(p, out, 1);
    second.encode(p);
    core.dispatch(p, .{ .x = threads * ((out.size() + 31) / 32) }, .{ .x = threads });
}

/// MLX's `strided_reduce_general_dispatch`: the axis steps over the memory that follows it.
fn strided(op: Op, in: *Tensor, out: *Tensor, axis: usize) Error!void {
    const size: usize = @intCast(in.shape[axis]);
    const stride: usize = in.strides[axis];
    var walk = Walk.without(in, axis);
    var back: usize = 1;
    while (walk.n > 0 and back < stride) {
        back *= @intCast(walk.shape[walk.n - 1]);
        walk.pop();
    }
    var col: Column = .{ .size = size, .stride = @intCast(stride), .walk = walk.collapsed() };
    // Every variant walks the reduced axis itself as its one reduction axis.
    col.reduce_shape = .{@intCast(size)};
    col.reduce_strides = .{@intCast(stride)};
    col.reduce_ndim = 1;

    var out_n: usize = out.ndim;
    while (out_n > 0 and out.strides[out_n - 1] < stride) out_n -= 1;
    const out_grid = try grid2d(out.shape[0..out_n], out.strides[0..out_n]);
    const p = try core.pass();
    const group = 8 * 32;
    const bn = 32;

    if (size < 32) {
        const kern = try kernelOf("col_reduce_small_1_reduce_", op, in);
        const blocks = (stride + n_reads - 1) / n_reads;
        const gx = @min(blocks, 32);
        const gy = @min(8, @min(kern.max_threads / gx, size));
        p.pipeline(kern);
        core.bind(p, in, 0);
        core.bindOut(p, out, 1);
        col.encode(p);
        core.dispatchGroups(p, .{ .x = (blocks + gx - 1) / gx, .y = out_grid.x, .z = out_grid.y }, .{ .x = gx, .y = gy, .z = 1 });
        return;
    }

    var parts_shape: [max_dims]c_int = undefined;
    if (out.ndim + 1 > max_dims) return error.TensorShape;
    @memcpy(parts_shape[1..][0..out.ndim], out.dims());

    if (stride < 32 and size >= 1024) {
        const blocks: usize = if (size >= 32768) 128 else 32;
        parts_shape[0] = @intCast(blocks);
        const parts = try core.fresh(in.dtype, parts_shape[0 .. out.ndim + 1]);
        defer core.release(parts);
        p.pipeline(try kernelOf("col_reduce_longcolumn_1_reduce_", op, in));
        core.bind(p, in, 0);
        core.bindOut(p, parts, 1);
        col.encode(p);
        p.constant(out.size(), 11);
        core.dispatchGroups(p, .{ .x = out_grid.x, .y = out_grid.y, .z = blocks }, .{ .x = stride, .y = @min(32, (size + blocks - 1) / blocks), .z = 1 });
        return foldParts(op, p, parts, out, blocks, 256);
    }

    if (size > 256 and out.size() / 32 < 1024) {
        const blocks = 32;
        parts_shape[0] = blocks;
        const parts = try core.fresh(in.dtype, parts_shape[0 .. out.ndim + 1]);
        defer core.release(parts);
        p.pipeline(try kernelOf("col_reduce_2pass_1_32_32_reduce_", op, in));
        core.bind(p, in, 0);
        core.bindOut(p, parts, 1);
        col.encode(p);
        p.constant(out.size() / stride, 11);
        core.dispatch(p, .{ .x = group * ((stride + bn - 1) / bn), .y = out_grid.x * blocks, .z = out_grid.y }, .{ .x = group });
        return foldParts(op, p, parts, out, blocks, group);
    }

    p.pipeline(try kernelOf("col_reduce_looped_1_32_32_reduce_", op, in));
    core.bind(p, in, 0);
    core.bindOut(p, out, 1);
    col.encode(p);
    core.dispatch(p, .{ .x = group * ((stride + bn - 1) / bn), .y = out_grid.x, .z = out_grid.y }, .{ .x = group });
}

/// MLX's `ArgReduce::eval_gpu`: the index of the largest element along `axis` (u32), the axis kept with one element.
pub fn argmax(t: *Tensor, axis: usize) Error!*Tensor {
    if (t.size() == 0) return error.TensorUnsupported;
    var kept: [max_dims]c_int = undefined;
    @memcpy(kept[0..t.ndim], t.dims());
    kept[axis] = 1;
    const out = try core.fresh(.uint32, kept[0..t.ndim]);
    errdefer core.release(out);
    const in_walk = Walk.without(t, axis);
    const out_walk = Walk.without(out, axis);
    var buf: [48]u8 = undefined;
    const kern = try stock.kernel(try stock.named(&buf, "argmax_{s}", .{try stock.typeName(t.dtype)}));
    const size: usize = @intCast(t.shape[axis]);
    const threads = ((@min((size + n_reads - 1) / n_reads, kern.max_threads) + 31) / 32) * 32;
    const grid = try grid2d(out.dims(), out.strides[0..out.ndim]);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, t, 0);
    core.bindOut(p, out, 1);
    p.bytes(std.mem.sliceAsBytes(in_walk.shape[0..in_walk.count()]), 2);
    p.bytes(std.mem.sliceAsBytes(in_walk.strides[0..in_walk.count()]), 3);
    p.bytes(std.mem.sliceAsBytes(out_walk.strides[0..out_walk.count()]), 4);
    p.constant(in_walk.n, 5);
    p.constant(@as(i64, @intCast(t.strides[axis])), 6);
    p.constant(size, 7);
    core.dispatch(p, .{ .x = threads, .y = grid.x, .z = grid.y }, .{ .x = threads });
    return out;
}
