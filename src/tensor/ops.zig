//! The stock ops a forward calls, on tensors of ours. A view is host arithmetic; the rest is one dispatch of a small
//! kernel of ours. Every function returns a tensor the caller releases.

const std = @import("std");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const stock_reduce = @import("reduce.zig");
const kernel = @import("kernel.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const Error = core.Error;
const max_dims = core.max_dims;
const gpa = std.heap.c_allocator;

// ── Views ──

pub fn reshape(t: *Tensor, shape_in: []const c_int) Error!*Tensor {
    if (shape_in.len > max_dims) return error.TensorShape;
    var shape: [max_dims]c_int = undefined;
    var known: usize = 1;
    var infer: ?usize = null;
    for (shape_in, 0..) |d, i| {
        shape[i] = d;
        if (d == -1) {
            if (infer != null) return error.TensorShape;
            infer = i;
        } else known *= @intCast(d);
    }
    const total = t.size();
    if (infer) |i| {
        if (known == 0 or total % known != 0) return error.TensorShape;
        shape[i] = @intCast(total / known);
    } else if (known != total) return error.TensorShape;

    const base = if (t.rowContiguous()) try core.share(t) else try contiguousCopy(t);
    base.ndim = @intCast(shape_in.len);
    var stride: usize = 1;
    var axis = shape_in.len;
    while (axis > 0) {
        axis -= 1;
        base.shape[axis] = shape[axis];
        base.strides[axis] = stride;
        stride *= @intCast(shape[axis]);
    }
    return base;
}

pub fn transpose(t: *Tensor, axes: []const c_int) Error!*Tensor {
    if (axes.len != t.ndim) return error.TensorShape;
    const v = try core.share(t);
    for (axes, 0..) |a, i| {
        const from: usize = @intCast(if (a < 0) a + @as(c_int, t.ndim) else a);
        if (from >= t.ndim) {
            core.release(v);
            return error.TensorShape;
        }
        v.shape[i] = t.shape[from];
        v.strides[i] = t.strides[from];
    }
    return v;
}

/// MLX's slice: per axis `start`, `stop` (a negative index counts from the end, both clamp to the axis) and a
/// positive step.
pub fn slice(t: *Tensor, start: []const c_int, stop: []const c_int, step: []const c_int) Error!*Tensor {
    if (start.len != t.ndim or stop.len != t.ndim or step.len != t.ndim) return error.TensorShape;
    const v = try core.share(t);
    errdefer core.release(v);
    var skip: usize = 0;
    for (0..t.ndim) |i| {
        const n = t.shape[i];
        if (step[i] <= 0) return error.TensorUnsupported;
        const lo = std.math.clamp(if (start[i] < 0) start[i] + n else start[i], 0, n);
        const hi = std.math.clamp(if (stop[i] < 0) stop[i] + n else stop[i], lo, n);
        skip += @as(usize, @intCast(lo)) * t.strides[i];
        v.shape[i] = @divFloor(hi - lo + step[i] - 1, step[i]);
        v.strides[i] = t.strides[i] * @as(usize, @intCast(step[i]));
    }
    v.offset += skip * core.itemSize(t.dtype);
    return v;
}

fn dropAxes(t: *Tensor, drop: []const bool) Error!*Tensor {
    const v = try core.share(t);
    var n: u8 = 0;
    for (0..t.ndim) |i| {
        if (drop[i]) continue;
        v.shape[n] = t.shape[i];
        v.strides[n] = t.strides[i];
        n += 1;
    }
    v.ndim = n;
    return v;
}

pub fn squeeze(t: *Tensor, axes: ?[]const c_int) Error!*Tensor {
    var drop: [max_dims]bool = @splat(false);
    if (axes) |list| {
        for (list) |a| {
            const axis: usize = @intCast(if (a < 0) a + @as(c_int, t.ndim) else a);
            if (axis >= t.ndim or t.shape[axis] != 1) return error.TensorShape;
            drop[axis] = true;
        }
    } else for (0..t.ndim) |i| {
        drop[i] = t.shape[i] == 1;
    }
    return dropAxes(t, &drop);
}

pub fn expandDims(t: *Tensor, axis_in: c_int) Error!*Tensor {
    if (t.ndim == max_dims) return error.TensorShape;
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) + 1 else axis_in);
    if (axis > t.ndim) return error.TensorShape;
    const v = try core.share(t);
    var i: usize = t.ndim;
    while (i > axis) : (i -= 1) {
        v.shape[i] = t.shape[i - 1];
        v.strides[i] = t.strides[i - 1];
    }
    v.shape[axis] = 1;
    v.strides[axis] = if (axis < t.ndim) t.strides[axis] * @as(usize, @intCast(t.shape[axis])) else 1;
    v.ndim = t.ndim + 1;
    return v;
}

/// A view at `shape` whose broadcast axes repeat the one element they have (stride 0).
pub fn broadcastTo(t: *Tensor, shape: []const c_int) Error!*Tensor {
    if (shape.len > max_dims or shape.len < t.ndim) return error.TensorShape;
    const v = try core.share(t);
    errdefer core.release(v);
    const lead = shape.len - t.ndim;
    for (shape, 0..) |d, i| {
        v.shape[i] = d;
        if (i < lead) {
            v.strides[i] = 0;
            continue;
        }
        const have = t.shape[i - lead];
        if (have == d) {
            v.strides[i] = t.strides[i - lead];
        } else if (have == 1) {
            v.strides[i] = 0;
        } else return error.TensorShape;
    }
    v.ndim = @intCast(shape.len);
    return v;
}

fn sameBits(a: Dtype, b: Dtype) bool {
    const ints = [_][2]Dtype{ .{ .int8, .uint8 }, .{ .int16, .uint16 }, .{ .int32, .uint32 }, .{ .int64, .uint64 } };
    for (ints) |pair| {
        if ((a == pair[0] and b == pair[1]) or (a == pair[1] and b == pair[0])) return true;
    }
    return a == b;
}

/// A cast. Between a signed integer and its unsigned twin the bits are the value's, so that is a view.
pub fn astype(t: *Tensor, dtype: Dtype) Error!*Tensor {
    if (sameBits(t.dtype, dtype)) {
        const v = try core.share(t);
        v.dtype = dtype;
        return v;
    }
    return elementwise("cast", "$O(x)", &.{t}, dtype, t.dims());
}

/// The same bytes read as another dtype (`mlx_view`): the last axis takes up the change in item size.
pub fn reinterpret(t: *Tensor, dtype: Dtype) Error!*Tensor {
    const from = core.itemSize(t.dtype);
    const to = core.itemSize(dtype);
    const base = if (t.rowContiguous()) try core.share(t) else try contiguousCopy(t);
    errdefer core.release(base);
    base.dtype = dtype;
    if (from == to or base.ndim == 0) return base;
    const last = base.ndim - 1;
    const bytes = @as(usize, @intCast(base.shape[last])) * from;
    if (bytes % to != 0) return error.TensorShape;
    var shape: [max_dims]c_int = base.shape;
    shape[last] = @intCast(bytes / to);
    base.ndim = t.ndim;
    var stride: usize = 1;
    var axis: usize = base.ndim;
    while (axis > 0) {
        axis -= 1;
        base.shape[axis] = shape[axis];
        base.strides[axis] = stride;
        stride *= @intCast(shape[axis]);
    }
    return base;
}

// ── Kernels of the engine's own ──

const helpers =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\typedef bfloat bfloat16_t;
    \\typedef half float16_t;
    \\inline long sushi_loc(uint3 at, const constant int* shape, const constant long* strides, int ndim) {
    \\  long loc = long(at.x) * strides[ndim - 1] + long(at.y) * strides[ndim - 2];
    \\  uint z = at.z;
    \\  for (int d = ndim - 3; d >= 0; --d) {
    \\    loc += long(z % uint(shape[d])) * strides[d];
    \\    z /= uint(shape[d]);
    \\  }
    \\  return loc;
    \\}
    \\inline long sushi_flat(uint3 at, uint3 grid) {
    \\  return long(at.x) + long(grid.x) * (long(at.y) + long(grid.y) * long(at.z));
    \\}
    \\// MLX's Sigmoid, Maximum and Minimum (mlx/backend/metal/kernels/{unary,binary}_ops.h), statement for statement.
    \\template <typename T> T sushi_sigmoid(T x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
    \\template <typename T> metal::enable_if_t<metal::is_integral_v<T>, T> sushi_maximum(T x, T y) { return metal::max(x, y); }
    \\template <typename T> metal::enable_if_t<!metal::is_integral_v<T>, T> sushi_maximum(T x, T y) {
    \\  if (metal::isnan(x)) return x;
    \\  return x > y ? x : y;
    \\}
    \\template <typename T> metal::enable_if_t<metal::is_integral_v<T>, T> sushi_minimum(T x, T y) { return metal::min(x, y); }
    \\template <typename T> metal::enable_if_t<!metal::is_integral_v<T>, T> sushi_minimum(T x, T y) {
    \\  if (metal::isnan(x)) return x;
    \\  return x < y ? x : y;
    \\}
    \\
;

var pipelines: std.StringHashMapUnmanaged(metal.Pipeline) = .empty;

/// The pipeline of `source` with `$0`, `$1`, .. and `$O` standing for the input and output type names.
fn pipelineFor(comptime tag: []const u8, source: []const u8, in_types: []const Dtype, out_type: Dtype) Error!metal.Pipeline {
    var key_buf: [96]u8 = undefined;
    var key: std.ArrayList(u8) = .initBuffer(&key_buf);
    key.appendSliceBounded(tag) catch return error.TensorUnsupported;
    for (in_types) |d| key.appendBounded(@intCast(@intFromEnum(d) + 'a')) catch return error.TensorUnsupported;
    key.appendBounded(@intCast(@intFromEnum(out_type) + 'A')) catch return error.TensorUnsupported;
    if (pipelines.get(key.items)) |p| return p;

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    text.appendSlice(gpa, helpers) catch return error.OutOfMemory;
    var at: usize = 0;
    while (at < source.len) {
        if (source[at] == '$' and at + 1 < source.len) {
            const c = source[at + 1];
            const name = if (c == 'O') try kernel.typeName(out_type) else if (c >= '0' and c <= '9' and c - '0' < in_types.len) try kernel.typeName(in_types[c - '0']) else null;
            if (name) |n| {
                text.appendSlice(gpa, n) catch return error.OutOfMemory;
                at += 2;
                continue;
            }
        }
        text.append(gpa, source[at]) catch return error.OutOfMemory;
        at += 1;
    }
    text.append(gpa, 0) catch return error.OutOfMemory;
    const t0 = core.nowNs();
    const pipeline = try (try metal.device()).compile(text.items[0 .. text.items.len - 1 :0], "sushi_" ++ tag);
    core.madePipeline("sushi_" ++ tag, t0, pipeline);
    pipelines.put(gpa, gpa.dupe(u8, key.items) catch return error.OutOfMemory, pipeline) catch return error.OutOfMemory;
    return pipeline;
}

/// A shape as the grid kernels index it: at least three axes, the last two the grid's x and y.
const Grid = struct {
    shape: [max_dims]c_int,
    ndim: usize,
    size: metal.Size,
    group: metal.Size,

    fn of(shape: []const c_int) Grid {
        var g: Grid = undefined;
        g.ndim = @max(shape.len, 3);
        const lead = g.ndim - shape.len;
        for (0..g.ndim) |i| g.shape[i] = if (i < lead) 1 else shape[i - lead];
        var rest: usize = 1;
        for (g.shape[0 .. g.ndim - 2]) |d| rest *= @intCast(d);
        g.size = .{ .x = @intCast(g.shape[g.ndim - 1]), .y = @intCast(g.shape[g.ndim - 2]), .z = rest };
        g.group = block(g.size, 1024);
        return g;
    }

    /// `t`'s strides against this shape: an axis `t` lacks or has one element of does not move.
    fn strides(g: *const Grid, t: *const Tensor) [max_dims]i64 {
        var out: [max_dims]i64 = @splat(0);
        const lead = g.ndim - t.ndim;
        for (0..t.ndim) |i| {
            if (t.shape[i] != 1) out[i + lead] = @intCast(t.strides[i]);
        }
        return out;
    }
};

fn block(size: metal.Size, max_threads: usize) metal.Size {
    var b: metal.Size = .{ .x = 1 };
    var total: usize = 1;
    var grew = true;
    while (grew) {
        grew = false;
        inline for (.{ "x", "y", "z" }) |f| {
            if (@field(b, f) < @field(size, f) and total * 2 <= max_threads) {
                @field(b, f) *= 2;
                total *= 2;
                grew = true;
            }
        }
    }
    return b;
}

/// `out[i] = expr` over the broadcast of up to three inputs, each read as `x`, `y`, `z` in its own type.
fn elementwise(comptime tag: []const u8, comptime expr: []const u8, inputs: []const *Tensor, out_dtype: Dtype, out_shape: []const c_int) Error!*Tensor {
    const names = [_][]const u8{ "x", "y", "z" };
    comptime var source: []const u8 = "[[kernel]] void sushi_" ++ tag ++ "(\n";
    comptime var n_in = 0;
    inline for (names, 0..) |name, i| {
        if (comptime std.mem.indexOf(u8, expr, name) != null or i == 0) {
            source = source ++ std.fmt.comptimePrint("  const device ${d}* in{d} [[buffer({d})]],\n  const constant long* s{d} [[buffer({d})]],\n", .{ i, i, 2 * i, i, 2 * i + 1 });
            n_in = i + 1;
        }
    }
    const base = 2 * n_in;
    source = source ++ std.fmt.comptimePrint(
        \\  device $O* out [[buffer({d})]],
        \\  const constant int* shape [[buffer({d})]],
        \\  const constant int& ndim [[buffer({d})]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {{
        \\
    , .{ base, base + 1, base + 2 });
    inline for (0..n_in) |i| {
        source = source ++ std.fmt.comptimePrint("  const ${d} {s} = in{d}[sushi_loc(at, shape, s{d}, ndim)];\n", .{ i, names[i], i, i });
    }
    source = source ++ "  out[sushi_flat(at, grid)] = " ++ expr ++ ";\n}\n";
    if (inputs.len != n_in) return error.TensorShape;

    var types: [3]Dtype = undefined;
    for (inputs, 0..) |in, i| types[i] = in.dtype;
    const pipeline = try pipelineFor(tag, source, types[0..n_in], out_dtype);
    const g = Grid.of(out_shape);
    const out = try core.fresh(out_dtype, out_shape);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const p = try core.pass();
    p.pipeline(pipeline);
    for (inputs, 0..) |in, i| {
        core.bind(p, in, 2 * i);
        const s = g.strides(in);
        p.bytes(std.mem.sliceAsBytes(s[0..g.ndim]), 2 * i + 1);
    }
    core.bindOut(p, out, base);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), base + 1);
    p.constant(@as(c_int, @intCast(g.ndim)), base + 2);
    core.dispatch(p, g.size, g.group);
    return out;
}

/// A row-contiguous copy of `t` in a buffer of its own.
pub fn contiguousCopy(t: *Tensor) Error!*Tensor {
    return elementwise("copy", "x", &.{t}, t.dtype, t.dims());
}

pub fn contiguous(t: *Tensor) Error!*Tensor {
    return if (t.rowContiguous()) core.share(t) else contiguousCopy(t);
}

fn broadcastShape(a: *const Tensor, b: *const Tensor, out: *[max_dims]c_int) Error![]const c_int {
    const n = @max(a.ndim, b.ndim);
    for (0..n) |i| {
        const da: c_int = if (i + a.ndim >= n) a.shape[i + a.ndim - n] else 1;
        const db: c_int = if (i + b.ndim >= n) b.shape[i + b.ndim - n] else 1;
        if (da != db and da != 1 and db != 1) return error.TensorShape;
        out[i] = if (da == 1) db else da;
    }
    return out[0..n];
}

pub const Binary = enum { add, subtract, multiply, divide, maximum, minimum, less, less_equal, greater, greater_equal, equal, logical_and, logical_or };

/// An elementwise op of two tensors of one dtype, broadcast together. A comparison gives bool.
pub fn binary(comptime op: Binary, a: *Tensor, b: *Tensor) Error!*Tensor {
    if (a.dtype != b.dtype) return error.TensorDtype;
    var shape_buf: [max_dims]c_int = undefined;
    const shape = try broadcastShape(a, b, &shape_buf);
    const compare = switch (op) {
        .less, .less_equal, .greater, .greater_equal, .equal => true,
        else => false,
    };
    const expr = switch (op) {
        .add => "x + y",
        .subtract => "x - y",
        .multiply => "x * y",
        .divide => "x / y",
        .maximum => "sushi_maximum(x, y)",
        .minimum => "sushi_minimum(x, y)",
        .less => "x < y",
        .less_equal => "x <= y",
        .greater => "x > y",
        .greater_equal => "x >= y",
        .equal => "x == y",
        .logical_and => "x && y",
        .logical_or => "x || y",
    };
    return elementwise("bin_" ++ @tagName(op), expr, &.{ a, b }, if (compare) .bool_ else a.dtype, shape);
}

pub const Unary = enum { exp, log, sigmoid, sqrt, rsqrt, square, floor, negative };

pub fn unary(comptime op: Unary, a: *Tensor) Error!*Tensor {
    if (op == .log and a.dtype != .float32 and a.dtype != .float16 and a.dtype != .bfloat16) return error.TensorDtype;
    const expr = switch (op) {
        .exp => "$O(metal::precise::exp(x))",
        .log => "$O(metal::precise::log(x))",
        .sigmoid => "sushi_sigmoid(x)",
        .sqrt => "$O(metal::precise::sqrt(x))",
        .rsqrt => "$O(metal::precise::rsqrt(x))",
        .square => "x * x",
        .floor => "floor(x)",
        .negative => "-x",
    };
    return elementwise("un_" ++ @tagName(op), expr, &.{a}, a.dtype, a.dims());
}

/// `cond ? a : b`, the three broadcast together.
pub fn where(cond: *Tensor, a: *Tensor, b: *Tensor) Error!*Tensor {
    if (cond.dtype != .bool_ or a.dtype != b.dtype) return error.TensorDtype;
    var buf_ab: [max_dims]c_int = undefined;
    const ab = try broadcastShape(a, b, &buf_ab);
    var probe: Tensor = cond.*;
    probe.ndim = @intCast(ab.len);
    @memcpy(probe.shape[0..ab.len], ab);
    var buf: [max_dims]c_int = undefined;
    const shape = try broadcastShape(cond, &probe, &buf);
    return elementwise("where", "x ? y : z", &.{ cond, a, b }, a.dtype, shape);
}

pub const Reduce = enum { sum, max };

/// A reduction along one axis, read in index order.
pub fn reduce(comptime op: Reduce, t: *Tensor, axis_in: c_int, keepdims: bool) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim) return error.TensorShape;
    switch (t.dtype) {
        // A float reduction is MLX's kernel under MLX's plan: the order a sum is taken in is part of its bytes.
        .float16, .bfloat16, .float32 => {
            const out = try stock_reduce.reduce(if (op == .sum) .sum else .max, t, axis);
            if (keepdims) return out;
            defer core.release(out);
            return squeeze(out, &.{@intCast(axis)});
        },
        else => {},
    }
    const source =
        "[[kernel]] void sushi_red_" ++ @tagName(op) ++ "(\n" ++
        \\  const device $0* in [[buffer(0)]],
        \\  const constant long* s0 [[buffer(1)]],
        \\  device $O* out [[buffer(2)]],
        \\  const constant int* shape [[buffer(3)]],
        \\  const constant int& ndim [[buffer(4)]],
        \\  const constant long& step [[buffer(5)]],
        \\  const constant int& count [[buffer(6)]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {
        \\  long loc = sushi_loc(at, shape, s0, ndim);
        \\  $O acc = in[loc];
        \\  for (int k = 1; k < count; ++k) {
        \\    loc += step;
        \\
    ++ (if (op == .sum) "    acc += in[loc];\n" else "    acc = max(acc, $O(in[loc]));\n") ++
        \\  }
        \\  out[sushi_flat(at, grid)] = acc;
        \\}
        \\
    ;
    var kept: [max_dims]c_int = undefined;
    @memcpy(kept[0..t.ndim], t.dims());
    kept[axis] = 1;
    const pipeline = try pipelineFor("red_" ++ @tagName(op), source, &.{t.dtype}, t.dtype);
    const g = Grid.of(kept[0..t.ndim]);
    const out = try core.fresh(t.dtype, kept[0..t.ndim]);
    errdefer core.release(out);
    // The reduced axis has one element in the grid, so its stride there is never taken.
    var strides: [max_dims]i64 = @splat(0);
    const lead = g.ndim - t.ndim;
    for (0..t.ndim) |i| {
        if (i != axis) strides[i + lead] = @intCast(t.strides[i]);
    }
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, t, 0);
    p.bytes(std.mem.sliceAsBytes(strides[0..g.ndim]), 1);
    core.bindOut(p, out, 2);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 3);
    p.constant(@as(c_int, @intCast(g.ndim)), 4);
    p.constant(@as(i64, @intCast(t.strides[axis])), 5);
    p.constant(t.shape[axis], 6);
    core.dispatch(p, g.size, g.group);
    if (keepdims) return out;
    defer core.release(out);
    return squeeze(out, &.{@intCast(axis)});
}

/// `base` with `update` written over the slice `start..stop` (step 1). The write waits until the result's bytes are
/// first used (`settle`): if by then the result is the last holder of `base`'s bytes, as it is once the caller has
/// dropped `base`, the update goes into them and nothing else is copied.
pub fn sliceUpdate(base: *Tensor, update: *Tensor, start: []const c_int, stop: []const c_int) Error!*Tensor {
    if (base.dtype != update.dtype) return error.TensorDtype;
    if (start.len != base.ndim or stop.len != base.ndim) return error.TensorShape;
    var ones: [max_dims]c_int = @splat(1);
    const window = try slice(base, start, stop, ones[0..base.ndim]);
    defer core.release(window);
    const src = try broadcastTo(update, window.dims());
    errdefer core.release(src);
    const out = try core.share(base);
    errdefer core.release(out);
    out.pending = try core.newPending(src, start, stop);
    return out;
}

/// Writes `t`'s pending slice update.
pub fn settle(t: *Tensor) Error!void {
    const p = t.pending orelse return;
    t.pending = null;
    defer core.freePending(p);
    // Through a broadcast axis one write would land on every element that axis reads.
    const in_place = !t.repeats() and try core.sole(t.store);
    if (!in_place) core.adopt(t, try contiguousCopy(t));
    core.countUpdate(in_place);
    var ones: [max_dims]c_int = @splat(1);
    const window = try slice(t, p.start[0..t.ndim], p.stop[0..t.ndim], ones[0..t.ndim]);
    defer core.release(window);
    try copyInto(window, p.update);
}

/// Writes `src` over the bytes `dst` views; the two have one shape.
fn copyInto(dst: *Tensor, src: *Tensor) Error!void {
    const source =
        \\[[kernel]] void sushi_copy_into(
        \\  const device $0* in [[buffer(0)]],
        \\  const constant long* s0 [[buffer(1)]],
        \\  device $O* out [[buffer(2)]],
        \\  const constant long* so [[buffer(3)]],
        \\  const constant int* shape [[buffer(4)]],
        \\  const constant int& ndim [[buffer(5)]],
        \\  uint3 at [[thread_position_in_grid]]) {
        \\  out[sushi_loc(at, shape, so, ndim)] = in[sushi_loc(at, shape, s0, ndim)];
        \\}
        \\
    ;
    const pipeline = try pipelineFor("copy_into", source, &.{src.dtype}, dst.dtype);
    const g = Grid.of(dst.dims());
    if (dst.size() == 0) return;
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, src, 0);
    const s_in = g.strides(src);
    p.bytes(std.mem.sliceAsBytes(s_in[0..g.ndim]), 1);
    core.bindOut(p, dst, 2);
    var s_out: [max_dims]i64 = @splat(0);
    const lead = g.ndim - dst.ndim;
    for (0..dst.ndim) |i| s_out[i + lead] = @intCast(dst.strides[i]);
    p.bytes(std.mem.sliceAsBytes(s_out[0..g.ndim]), 3);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 4);
    p.constant(@as(c_int, @intCast(g.ndim)), 5);
    core.dispatch(p, g.size, g.group);
}

pub fn concatenate(parts: []const *Tensor, axis_in: c_int) Error!*Tensor {
    if (parts.len == 0) return error.TensorShape;
    const first = parts[0];
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, first.ndim) else axis_in);
    if (axis >= first.ndim) return error.TensorShape;
    var shape: [max_dims]c_int = undefined;
    @memcpy(shape[0..first.ndim], first.dims());
    shape[axis] = 0;
    for (parts) |part| {
        if (part.ndim != first.ndim or part.dtype != first.dtype) return error.TensorShape;
        for (0..first.ndim) |i| {
            if (i != axis and part.shape[i] != first.shape[i]) return error.TensorShape;
        }
        shape[axis] += part.shape[axis];
    }
    const out = try core.fresh(first.dtype, shape[0..first.ndim]);
    errdefer core.release(out);
    var start: [max_dims]c_int = @splat(0);
    var stop: [max_dims]c_int = shape;
    var ones: [max_dims]c_int = @splat(1);
    var at: c_int = 0;
    for (parts) |part| {
        start[axis] = at;
        stop[axis] = at + part.shape[axis];
        const window = try slice(out, start[0..first.ndim], stop[0..first.ndim], ones[0..first.ndim]);
        defer core.release(window);
        try copyInto(window, part);
        at = stop[axis];
    }
    return out;
}

/// `t[indices]` along `axis`: the result has `indices`' shape in that axis's place.
pub fn take(t: *Tensor, indices: *Tensor, axis_in: c_int) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim) return error.TensorShape;
    if (indices.dtype != .int32 and indices.dtype != .uint32) return error.TensorDtype;
    if (t.ndim + indices.ndim - 1 > max_dims) return error.TensorShape;
    // The kernel walks the result as [before.., index.., after..]: an index axis moves the table's `axis` stride
    // by the index it reads, every other axis moves its own.
    var shape: [max_dims]c_int = undefined;
    var n: usize = 0;
    for (0..axis) |i| {
        shape[n] = t.shape[i];
        n += 1;
    }
    for (indices.dims()) |d| {
        shape[n] = d;
        n += 1;
    }
    for (axis + 1..t.ndim) |i| {
        shape[n] = t.shape[i];
        n += 1;
    }
    const source =
        \\[[kernel]] void sushi_take(
        \\  const device $0* table [[buffer(0)]],
        \\  const constant long* st [[buffer(1)]],
        \\  const device $1* index [[buffer(2)]],
        \\  const constant long* si [[buffer(3)]],
        \\  device $O* out [[buffer(4)]],
        \\  const constant int* shape [[buffer(5)]],
        \\  const constant int& ndim [[buffer(6)]],
        \\  const constant long& axis_stride [[buffer(7)]],
        \\  const constant int& axis_len [[buffer(8)]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {
        \\  long pick = long(index[sushi_loc(at, shape, si, ndim)]);
        \\  if (pick < 0) pick += axis_len;
        \\  out[sushi_flat(at, grid)] = table[sushi_loc(at, shape, st, ndim) + pick * axis_stride];
        \\}
        \\
    ;
    const pipeline = try pipelineFor("take", source, &.{ t.dtype, indices.dtype }, t.dtype);
    const g = Grid.of(shape[0..n]);
    const out = try core.fresh(t.dtype, shape[0..n]);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const lead = g.ndim - n;
    var st: [max_dims]i64 = @splat(0);
    var si: [max_dims]i64 = @splat(0);
    for (0..axis) |i| st[lead + i] = @intCast(t.strides[i]);
    for (0..indices.ndim) |i| si[lead + axis + i] = @intCast(indices.strides[i]);
    for (axis + 1..t.ndim) |i| st[lead + indices.ndim + i - 1] = @intCast(t.strides[i]);
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, t, 0);
    p.bytes(std.mem.sliceAsBytes(st[0..g.ndim]), 1);
    core.bind(p, indices, 2);
    p.bytes(std.mem.sliceAsBytes(si[0..g.ndim]), 3);
    core.bindOut(p, out, 4);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 5);
    p.constant(@as(c_int, @intCast(g.ndim)), 6);
    p.constant(@as(i64, @intCast(t.strides[axis])), 7);
    p.constant(t.shape[axis], 8);
    core.dispatch(p, g.size, g.group);
    return out;
}

/// `start, start + step, ..` below `stop`.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype) Error!*Tensor {
    const count: usize = @intFromFloat(@max(0, @ceil((stop - start) / step)));
    const out = try core.freshForHost(dtype, &.{@intCast(count)});
    errdefer core.release(out);
    const bytes = try core.hostBytes(out);
    switch (dtype) {
        .int32 => for (@as([]i32, @alignCast(std.mem.bytesAsSlice(i32, bytes))), 0..) |*v, i| {
            v.* = @intFromFloat(start + step * @as(f64, @floatFromInt(i)));
        },
        .float32 => for (@as([]f32, @alignCast(std.mem.bytesAsSlice(f32, bytes))), 0..) |*v, i| {
            v.* = @floatCast(start + step * @as(f64, @floatFromInt(i)));
        },
        else => return error.TensorDtype,
    }
    return out;
}

pub fn zeros(shape: []const c_int, dtype: Dtype) Error!*Tensor {
    const out = try core.freshForHost(dtype, shape);
    errdefer core.release(out);
    @memset(try core.hostBytes(out), 0);
    return out;
}

/// `t` with each element along `axis` written `count` times in a row.
pub fn repeat(t: *Tensor, count: c_int, axis_in: c_int) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim or count < 0 or t.ndim + 1 > max_dims) return error.TensorShape;
    const wide = try expandDims(t, @intCast(axis + 1));
    defer core.release(wide);
    var shape: [max_dims]c_int = undefined;
    @memcpy(shape[0..wide.ndim], wide.dims());
    shape[axis + 1] = count;
    const spread = try broadcastTo(wide, shape[0..wide.ndim]);
    defer core.release(spread);
    @memcpy(shape[0..t.ndim], t.dims());
    shape[axis] *= count;
    return reshape(spread, shape[0..t.ndim]);
}

/// `parts` joined along a new axis.
pub fn stack(parts: []const *Tensor, axis: c_int) Error!*Tensor {
    var wide: [64]*Tensor = undefined;
    if (parts.len == 0 or parts.len > wide.len) return error.TensorShape;
    var n: usize = 0;
    defer for (wide[0..n]) |w| core.release(w);
    const at = if (axis < 0) axis + @as(c_int, parts[0].ndim) + 1 else axis;
    for (parts) |part| {
        wide[n] = try expandDims(part, at);
        n += 1;
    }
    return concatenate(wide[0..n], at);
}

/// `t[.., indices[..], ..]` along `axis`: the result has `indices`' shape, which matches `t`'s off the axis.
pub fn takeAlong(t: *Tensor, indices: *Tensor, axis_in: c_int) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim or indices.ndim != t.ndim) return error.TensorShape;
    if (indices.dtype != .int32 and indices.dtype != .uint32) return error.TensorDtype;
    for (0..t.ndim) |i| {
        if (i != axis and t.shape[i] != indices.shape[i]) return error.TensorShape;
    }
    const source =
        \\[[kernel]] void sushi_take_along(
        \\  const device $0* table [[buffer(0)]],
        \\  const constant long* st [[buffer(1)]],
        \\  const device $1* index [[buffer(2)]],
        \\  const constant long* si [[buffer(3)]],
        \\  device $O* out [[buffer(4)]],
        \\  const constant int* shape [[buffer(5)]],
        \\  const constant int& ndim [[buffer(6)]],
        \\  const constant long& axis_stride [[buffer(7)]],
        \\  const constant int& axis_len [[buffer(8)]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {
        \\  long pick = long(index[sushi_loc(at, shape, si, ndim)]);
        \\  if (pick < 0) pick += axis_len;
        \\  out[sushi_flat(at, grid)] = table[sushi_loc(at, shape, st, ndim) + pick * axis_stride];
        \\}
        \\
    ;
    const pipeline = try pipelineFor("take_along", source, &.{ t.dtype, indices.dtype }, t.dtype);
    const g = Grid.of(indices.dims());
    const out = try core.fresh(t.dtype, indices.dims());
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const lead = g.ndim - t.ndim;
    var st: [max_dims]i64 = @splat(0);
    for (0..t.ndim) |i| {
        if (i != axis) st[lead + i] = @intCast(t.strides[i]);
    }
    const si = g.strides(indices);
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, t, 0);
    p.bytes(std.mem.sliceAsBytes(st[0..g.ndim]), 1);
    core.bind(p, indices, 2);
    p.bytes(std.mem.sliceAsBytes(si[0..g.ndim]), 3);
    core.bindOut(p, out, 4);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 5);
    p.constant(@as(c_int, @intCast(g.ndim)), 6);
    p.constant(@as(i64, @intCast(t.strides[axis])), 7);
    p.constant(t.shape[axis], 8);
    core.dispatch(p, g.size, g.group);
    return out;
}

/// MLX's `put_along_axis`: a copy of `t` with `values` written at `indices` along `axis`. Two indices that meet
/// write in no set order, as under MLX.
pub fn putAlong(t: *Tensor, indices: *Tensor, values_in: *Tensor, axis_in: c_int) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim or indices.ndim != t.ndim) return error.TensorShape;
    if (indices.dtype != .int32 and indices.dtype != .uint32) return error.TensorDtype;
    for (0..t.ndim) |i| {
        if (i != axis and t.shape[i] != indices.shape[i]) return error.TensorShape;
    }
    const typed = try astype(values_in, t.dtype);
    defer core.release(typed);
    const values = try broadcastTo(typed, indices.dims());
    defer core.release(values);
    const out = try contiguousCopy(t);
    errdefer core.release(out);
    if (indices.size() == 0) return out;
    const source =
        \\[[kernel]] void sushi_put_along(
        \\  const device $0* upd [[buffer(0)]],
        \\  const constant long* su [[buffer(1)]],
        \\  const device $1* index [[buffer(2)]],
        \\  const constant long* si [[buffer(3)]],
        \\  device $O* out [[buffer(4)]],
        \\  const constant long* so [[buffer(5)]],
        \\  const constant int* shape [[buffer(6)]],
        \\  const constant int& ndim [[buffer(7)]],
        \\  const constant long& axis_stride [[buffer(8)]],
        \\  const constant int& axis_len [[buffer(9)]],
        \\  uint3 at [[thread_position_in_grid]]) {
        \\  long pick = long(index[sushi_loc(at, shape, si, ndim)]);
        \\  if (pick < 0) pick += axis_len;
        \\  out[sushi_loc(at, shape, so, ndim) + pick * axis_stride] = upd[sushi_loc(at, shape, su, ndim)];
        \\}
        \\
    ;
    const pipeline = try pipelineFor("put_along", source, &.{ t.dtype, indices.dtype }, t.dtype);
    const g = Grid.of(indices.dims());
    const lead = g.ndim - t.ndim;
    var so: [max_dims]i64 = @splat(0);
    for (0..t.ndim) |i| {
        if (i != axis) so[lead + i] = @intCast(out.strides[i]);
    }
    const su = g.strides(values);
    const si = g.strides(indices);
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, values, 0);
    p.bytes(std.mem.sliceAsBytes(su[0..g.ndim]), 1);
    core.bind(p, indices, 2);
    p.bytes(std.mem.sliceAsBytes(si[0..g.ndim]), 3);
    core.bindOut(p, out, 4);
    p.bytes(std.mem.sliceAsBytes(so[0..g.ndim]), 5);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 6);
    p.constant(@as(c_int, @intCast(g.ndim)), 7);
    p.constant(@as(i64, @intCast(out.strides[axis])), 8);
    p.constant(t.shape[axis], 9);
    core.dispatch(p, g.size, g.group);
    return out;
}

/// MLX's constant `pad`: `t` with `low[i]` and `high[i]` elements of `value` added around `axes[i]`.
pub fn pad(t: *Tensor, axes: []const c_int, low: []const c_int, high: []const c_int, value_in: *Tensor) Error!*Tensor {
    if (axes.len != low.len or axes.len != high.len or value_in.size() != 1) return error.TensorShape;
    var shape: [max_dims]c_int = undefined;
    @memcpy(shape[0..t.ndim], t.dims());
    var before: [max_dims]c_int = @splat(0);
    for (axes, low, high) |a, lo, hi| {
        const axis: usize = @intCast(if (a < 0) a + @as(c_int, t.ndim) else a);
        if (axis >= t.ndim or lo < 0 or hi < 0) return error.TensorShape;
        before[axis] += lo;
        shape[axis] += lo + hi;
    }
    const value = try astype(value_in, t.dtype);
    defer core.release(value);
    const source =
        \\[[kernel]] void sushi_pad(
        \\  const device $0* in [[buffer(0)]],
        \\  const constant long* ist [[buffer(1)]],
        \\  const constant int* lo [[buffer(2)]],
        \\  const constant int* extent [[buffer(3)]],
        \\  const device $0* value [[buffer(4)]],
        \\  device $O* out [[buffer(5)]],
        \\  const constant int* shape [[buffer(6)]],
        \\  const constant int& ndim [[buffer(7)]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {
        \\  long loc = 0;
        \\  bool inside = true;
        \\  int c = int(at.x) - lo[ndim - 1];
        \\  inside = inside && c >= 0 && c < extent[ndim - 1];
        \\  loc += long(c) * ist[ndim - 1];
        \\  c = int(at.y) - lo[ndim - 2];
        \\  inside = inside && c >= 0 && c < extent[ndim - 2];
        \\  loc += long(c) * ist[ndim - 2];
        \\  uint z = at.z;
        \\  for (int d = ndim - 3; d >= 0; --d) {
        \\    c = int(z % uint(shape[d])) - lo[d];
        \\    z /= uint(shape[d]);
        \\    inside = inside && c >= 0 && c < extent[d];
        \\    loc += long(c) * ist[d];
        \\  }
        \\  out[sushi_flat(at, grid)] = inside ? in[loc] : value[0];
        \\}
        \\
    ;
    const pipeline = try pipelineFor("pad", source, &.{t.dtype}, t.dtype);
    const g = Grid.of(shape[0..t.ndim]);
    const out = try core.fresh(t.dtype, shape[0..t.ndim]);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const lead = g.ndim - t.ndim;
    var lo: [max_dims]c_int = @splat(0);
    var extent: [max_dims]c_int = @splat(1);
    var strides: [max_dims]i64 = @splat(0);
    for (0..t.ndim) |i| {
        lo[lead + i] = before[i];
        extent[lead + i] = t.shape[i];
        strides[lead + i] = @intCast(t.strides[i]);
    }
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, t, 0);
    p.bytes(std.mem.sliceAsBytes(strides[0..g.ndim]), 1);
    p.bytes(std.mem.sliceAsBytes(lo[0..g.ndim]), 2);
    p.bytes(std.mem.sliceAsBytes(extent[0..g.ndim]), 3);
    core.bind(p, value, 4);
    core.bindOut(p, out, 5);
    p.bytes(std.mem.sliceAsBytes(g.shape[0..g.ndim]), 6);
    p.constant(@as(c_int, @intCast(g.ndim)), 7);
    core.dispatch(p, g.size, g.group);
    return out;
}

/// The index of the first largest element along `axis`, uint32.
pub fn argmax(t: *Tensor, axis_in: c_int, keepdims: bool) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim) return error.TensorShape;
    const out = try stock_reduce.argmax(t, axis);
    if (keepdims) return out;
    defer core.release(out);
    return squeeze(out, &.{@intCast(axis)});
}
