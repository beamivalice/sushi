//! MLX's sort, argsort and searchsorted on the engine: `gpu_merge_sort` with its single-block and multi-block plans
//! and `SearchSorted::eval_gpu` (mlx/backend/metal/sort.cpp) over the compiled kernels of `mlx.metallib`. A merge sort
//! is stable, so the order of equal values is part of the bytes; this is MLX's kernel under MLX's plan.
//! Ported from MLX's Metal backend (MIT, see NOTICE). Only the last axis is sorted.

const std = @import("std");
const core = @import("core.zig");
const ops = @import("ops.zig");
const reduce = @import("reduce.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;
const max_dims = core.max_dims;

const tn = 4;

/// MLX's block width for a sorted axis of `n`, and the blocks it takes.
fn planOf(n: usize) struct { bn: usize, blocks: usize } {
    const potential = (n + tn - 1) / tn;
    const bn: usize = if (potential > 256) 512 else if (potential > 128) 256 else if (potential > 64) 128 else if (potential > 32) 64 else 32;
    return .{ .bn = bn, .blocks = (n + bn * tn - 1) / (bn * tn) };
}

fn sortable(d: core.Dtype) bool {
    return switch (d) {
        .uint8, .uint16, .uint32, .int8, .int16, .int32, .float16, .float32, .bfloat16 => true,
        else => false,
    };
}

/// `t` sorted along its last axis, or with `arg` the uint32 indices that sort it.
pub fn sort(t: *Tensor, axis_in: c_int, comptime arg: bool) Error!*Tensor {
    if (t.ndim == 0 or !sortable(t.dtype)) return error.TensorDtype;
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis != t.ndim - 1) return error.TensorUnsupported;
    const out_type: core.Dtype = if (arg) .uint32 else t.dtype;
    if (t.size() == 0) return core.fresh(out_type, t.dims());
    const n: usize = @intCast(t.shape[axis]);
    const plan = planOf(n);
    if (plan.blocks > 1) return multiBlock(t, n, plan.bn, plan.blocks, arg);
    const out = try core.fresh(out_type, t.dims());
    errdefer core.release(out);
    try singleBlock(t, out, n, plan.bn, arg);
    return out;
}

fn singleBlock(t: *Tensor, out: *Tensor, n: usize, bn: usize, comptime arg: bool) Error!void {
    const n_rows = t.size() / n;
    const flat = t.rowContiguous();
    var buf: [96]u8 = undefined;
    const k = try stock.kernel(try stock.named(&buf, "{s}{s}_block_sort_{s}_{s}_bn{d}_tn{d}", .{
        if (flat) "c" else "nc",
        if (arg) "arg" else "",
        try stock.typeName(t.dtype),
        try stock.typeName(out.dtype),
        bn,
        tn,
    }));
    if (bn > k.max_threads) return error.TensorShape;
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, t, 0);
    core.bindOut(p, out, 1);
    p.constant(@as(c_int, @intCast(n)), 2);
    p.constant(@as(c_int, @intCast(t.strides[t.ndim - 1])), 3);
    p.constant(@as(c_int, @intCast(out.strides[out.ndim - 1])), 4);
    const lead = t.ndim - 1;
    if (flat) {
        // The smallest stride of a leading axis with more than one element: MLX's segment stride.
        var segment: c_int = std.math.maxInt(i32);
        for (0..lead) |i| {
            if (t.shape[i] != 1) segment = @min(segment, @as(c_int, @intCast(t.strides[i])));
        }
        p.constant(segment, 5);
        p.constant(segment, 6);
    } else {
        var shape: [max_dims]c_int = @splat(0);
        var in_str: [max_dims]i64 = @splat(0);
        var out_str: [max_dims]i64 = @splat(0);
        for (0..lead) |i| {
            shape[i] = t.shape[i];
            in_str[i] = @intCast(t.strides[i]);
            out_str[i] = @intCast(out.strides[i]);
        }
        p.constant(@as(c_int, @intCast(lead)), 5);
        p.bytes(std.mem.sliceAsBytes(shape[0..@max(lead, 1)]), 6);
        p.bytes(std.mem.sliceAsBytes(in_str[0..@max(lead, 1)]), 7);
        p.bytes(std.mem.sliceAsBytes(out_str[0..@max(lead, 1)]), 8);
    }
    core.dispatchGroups(p, .{ .x = 1, .y = n_rows }, .{ .x = bn });
}

fn multiBlock(t: *Tensor, n: usize, bn: usize, n_blocks: usize, comptime arg: bool) Error!*Tensor {
    const n_rows = t.size() / n;
    const lead = t.ndim - 1;
    var shape: [max_dims]c_int = @splat(0);
    var strides: [max_dims]i64 = @splat(0);
    // With no other axis MLX sends one shape of 0 and one stride of 1.
    strides[0] = 1;
    for (0..lead) |i| {
        shape[i] = t.shape[i];
        strides[i] = @intCast(t.strides[i]);
    }
    const rows: c_int = @intCast(n_rows);
    var vals: [2]*Tensor = undefined;
    var idxs: [2]*Tensor = undefined;
    var made: usize = 0;
    defer for (0..made) |i| {
        core.release(vals[i]);
        core.release(idxs[i]);
    };
    while (made < 2) : (made += 1) {
        vals[made] = try core.fresh(t.dtype, &.{ rows, @intCast(n) });
        errdefer core.release(vals[made]);
        idxs[made] = try core.fresh(.uint32, &.{ rows, @intCast(n) });
    }
    const partitions = try core.fresh(.uint32, &.{ rows, @intCast(n_blocks + 1) });
    defer core.release(partitions);

    var buf: [96]u8 = undefined;
    const value_type = try stock.typeName(t.dtype);
    const p = try core.pass();
    p.pipeline(try stock.kernel(try stock.named(&buf, "sort_mbsort_{s}_uint32_bn{d}_tn{d}", .{ value_type, bn, tn })));
    core.bind(p, t, 0);
    core.bindOut(p, vals[0], 1);
    core.bindOut(p, idxs[0], 2);
    p.constant(@as(c_int, @intCast(n)), 3);
    p.constant(@as(c_int, @intCast(t.strides[lead])), 4);
    p.constant(@as(c_int, @intCast(lead)), 5);
    p.bytes(std.mem.sliceAsBytes(shape[0..@max(lead, 1)]), 6);
    p.bytes(std.mem.sliceAsBytes(strides[0..@max(lead, 1)]), 7);
    core.dispatchGroups(p, .{ .x = n_blocks, .y = n_rows }, .{ .x = bn });

    var from: usize = 1;
    var merge_tiles: c_int = 2;
    const threads = @min(n_blocks + 1, 1024);
    while (@divTrunc(merge_tiles, 2) < n_blocks) : (merge_tiles *= 2) {
        from = 1 - from;
        const to = 1 - from;
        const part = try core.pass();
        part.pipeline(try stock.kernel(try stock.named(&buf, "partition_mbsort_{s}_uint32_bn{d}_tn{d}", .{ value_type, bn, tn })));
        core.bindOut(part, partitions, 0);
        core.bind(part, vals[from], 1);
        core.bind(part, idxs[from], 2);
        part.constant(@as(c_int, @intCast(n)), 3);
        part.constant(merge_tiles, 4);
        part.constant(@as(c_int, @intCast(n_blocks)), 5);
        core.dispatchGroups(part, .{ .x = 1, .y = n_rows }, .{ .x = threads });

        const merge = try core.pass();
        merge.pipeline(try stock.kernel(try stock.named(&buf, "merge_mbsort_{s}_uint32_bn{d}_tn{d}", .{ value_type, bn, tn })));
        core.bind(merge, partitions, 0);
        core.bind(merge, vals[from], 1);
        core.bind(merge, idxs[from], 2);
        core.bindOut(merge, vals[to], 3);
        core.bindOut(merge, idxs[to], 4);
        merge.constant(@as(c_int, @intCast(n)), 5);
        merge.constant(merge_tiles, 6);
        merge.constant(@as(c_int, @intCast(n_blocks)), 7);
        core.dispatchGroups(merge, .{ .x = n_blocks, .y = n_rows }, .{ .x = bn });
    }
    // The last merge wrote the buffers the next one would have read.
    const last = 1 - from;
    return ops.reshape(if (arg) idxs[last] else vals[last], t.dims());
}

/// `SearchSorted`: for each of `values`, where it goes in the sorted 1-D `seq`, after (`right`) or before the
/// elements equal to it.
pub fn searchSorted(seq: *Tensor, values_in: *Tensor, right: bool) Error!*Tensor {
    if (seq.ndim != 1 or seq.dtype != values_in.dtype or !sortable(seq.dtype) or seq.size() == 0 or seq.size() > std.math.maxInt(u32)) return error.TensorUnsupported;
    const values = try ops.contiguous(values_in);
    defer core.release(values);
    const out = try core.fresh(.uint32, values.dims());
    errdefer core.release(out);
    if (out.size() == 0) return out;
    var buf: [96]u8 = undefined;
    const k = try stock.kernel(try stock.named(&buf, "searchsorted_{s}_{s}", .{ try stock.typeName(seq.dtype), if (right) "right" else "left" }));
    const p = try core.pass();
    p.pipeline(k);
    core.bind(p, seq, 0);
    core.bind(p, values, 1);
    core.bindOut(p, out, 2);
    p.constant(@as(u32, @intCast(seq.size())), 3);
    p.constant(@as(i64, @intCast(seq.strides[0])), 4);
    core.dispatch(p, try reduce.grid2d(out.dims(), out.strides[0..out.ndim]), .{ .x = @min(k.max_threads, out.size()) });
    return out;
}
