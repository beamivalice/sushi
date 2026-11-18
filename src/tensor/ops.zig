//! The stock ops a forward calls, on tensors of ours. A view is host arithmetic; the rest is one dispatch of a small
//! kernel of ours. Every function returns a tensor the caller releases.

const std = @import("std");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const stock_reduce = @import("reduce.zig");
const kernel = @import("kernel.zig");
const stock = @import("stock.zig");

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

    if (!t.rowContiguous() and total != 0) if (viewStrides(t, shape[0..shape_in.len])) |strides| {
        const view = try core.share(t);
        view.ndim = @intCast(shape_in.len);
        for (0..shape_in.len) |i| {
            view.shape[i] = shape[i];
            view.strides[i] = strides[i];
        }
        return view;
    };
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

/// MLX's `Reshape::prepare_reshape` (mlx/primitives.cpp): `t`'s strides as `shape` when its runs of joined axes
/// (`collapse_contiguous_dims`) split into it, or null where MLX copies.
fn viewStrides(t: *const Tensor, shape: []const c_int) ?[max_dims]usize {
    // The runs: axes whose stride times extent is the stride before them, a run's size kept under 2^31.
    var runs: [max_dims]usize = undefined;
    var run_strides: [max_dims]usize = undefined;
    var n: usize = 0;
    var open = false;
    var size: usize = 0;
    for (0..t.ndim) |i| {
        const d: usize = @intCast(t.shape[i]);
        const joined = i > 0 and t.strides[i] * d == t.strides[i - 1] and size * d <= std.math.maxInt(i32);
        size = if (joined) size * d else d;
        if (!joined) open = false;
        if (d == 1) continue;
        if (open) {
            runs[n - 1] *= d;
            run_strides[n - 1] = t.strides[i];
        } else {
            runs[n] = d;
            run_strides[n] = t.strides[i];
            n += 1;
            open = true;
        }
    }
    var out: [max_dims]usize = undefined;
    var j: usize = 0;
    for (shape, 0..) |d_in, i| {
        const d: usize = @intCast(d_in);
        if (j < n and runs[j] % d == 0) {
            runs[j] /= d;
            out[i] = runs[j] * run_strides[j];
            if (runs[j] == 1) j += 1;
        } else if (d == 1) {
            out[i] = if (i > 0) out[i - 1] else 1;
        } else return null;
    }
    return out;
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
        if (step[i] <= 0) return core.refusing("a slice step of zero or below", error.TensorUnsupported);
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
    if (dtype == .bool_ and (t.dtype == .float32 or t.dtype == .bfloat16)) return elementwise("cast", "sushi_true(x)", &.{t}, dtype, t.dims());
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
    \\inline int sushi_loc32(uint3 at, const constant int* shape, const constant int* strides, int ndim) {
    \\  int loc = int(at.x) * strides[ndim - 1] + int(at.y) * strides[ndim - 2];
    \\  uint z = at.z;
    \\  for (int d = ndim - 3; d >= 0; --d) {
    \\    loc += int(z % uint(shape[d])) * strides[d];
    \\    z /= uint(shape[d]);
    \\  }
    \\  return loc;
    \\}
    \\inline long sushi_flat(uint3 at, uint3 grid) {
    \\  return long(at.x) + long(grid.x) * (long(at.y) + long(grid.y) * long(at.z));
    \\}
    \\// MLX's `cast_to<bool, T>` (mlx/backend/metal/kernels/utils.h): any bit under the sign, so a denormal is true.
    \\inline bool sushi_true(float v) { return (as_type<uint>(v) & 0x7FFFFFFF) != 0; }
    \\inline bool sushi_true(bfloat v) { return (as_type<ushort>(v) & 0x7FFF) != 0; }
    \\// MLX's Sigmoid, Maximum and Minimum (mlx/backend/metal/kernels/{unary,binary}_ops.h), statement for statement.
    \\template <typename T> T sushi_sigmoid(T x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return T((x < 0) ? y : 1 - y);
    \\}
    \\// On bfloat16 MLX's math functions work in float and round their result (kernels/bf16_math.h); the rest is bfloat's.
    \\inline bfloat sushi_sigmoid(bfloat x) {
    \\  bfloat y = 1 / (1 + bfloat(metal::precise::exp(metal::abs(float(x)))));
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
    key.appendSliceBounded(tag) catch return core.refusing("a pipeline key past 96 bytes", error.TensorUnsupported);
    for (in_types) |d| key.appendBounded(@intCast(@intFromEnum(d) + 'a')) catch return core.refusing("a pipeline key past 96 bytes", error.TensorUnsupported);
    key.appendBounded(@intCast(@intFromEnum(out_type) + 'A')) catch return core.refusing("a pipeline key past 96 bytes", error.TensorUnsupported);
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
    const pipeline = try core.pipelineFromSource("sushi_" ++ tag, text.items[0 .. text.items.len - 1 :0]);
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

/// Elements one thread of the flat kernel takes.
const flat_per_thread = 4;

/// `elementwise` over inputs each row-major at the output's size or a single element: flat 32-bit indices, several
/// elements a thread, MLX's contiguous kernels' shape. `scalars` bit `i` reads input `i` at element 0.
fn flatSource(comptime tag: []const u8, comptime expr: []const u8, comptime n_in: usize, comptime scalars: u3) []const u8 {
    const names = [_][]const u8{ "x", "y", "z" };
    comptime var src: []const u8 = "[[kernel]] void sushi_" ++ tag ++ std.fmt.comptimePrint("_v{d}(\n", .{scalars});
    inline for (0..n_in) |i| src = src ++ std.fmt.comptimePrint("  const device ${d}* in{d} [[buffer({d})]],\n", .{ i, i, i });
    src = src ++ std.fmt.comptimePrint(
        \\  device $O* out [[buffer({d})]],
        \\  const constant uint& n [[buffer({d})]],
        \\  uint at [[thread_position_in_grid]]) {{
        \\  for (uint e = at * {d}u; e < min(n, at * {d}u + {d}u); ++e) {{
        \\
    , .{ n_in, n_in + 1, flat_per_thread, flat_per_thread, flat_per_thread });
    inline for (0..n_in) |i| src = src ++ std.fmt.comptimePrint("    const ${d} {s} = in{d}[{s}];\n", .{ i, names[i], i, if ((scalars >> i) & 1 == 1) "0" else "e" });
    return src ++ "    out[e] = " ++ expr ++ ";\n  }\n}\n";
}

/// The flat kernel's mask of single-element inputs, or null when an input is strided or broadcast along an axis.
fn flatMask(inputs: []const *Tensor, size: usize) ?u3 {
    if (size == 0 or size > std.math.maxInt(u32) - flat_per_thread) return null;
    var mask: u3 = 0;
    for (inputs, 0..) |in, i| {
        if (in.size() == 1 and size > 1) {
            mask |= @as(u3, 1) << @intCast(i);
        } else if (!in.rowContiguous() or in.size() != size) return null;
    }
    return mask;
}

/// `out[i] = expr` over the broadcast of up to three inputs, each read as `x`, `y`, `z` in its own type.
/// Inputs `expr` reads: `x` always, then `y` and `z` where it names them.
fn exprInputs(comptime expr: []const u8) usize {
    const names = [_][]const u8{ "x", "y", "z" };
    var n = 1;
    for (names[1..], 1..) |name, i| if (std.mem.indexOf(u8, expr, name) != null) {
        n = i + 1;
    };
    return n;
}

/// The strided kernel: each input read at its location in the output's grid, on 64-bit locations (`wide`) or 32-bit;
/// `three`: a grid of three axes, each location read straight off the thread's position, as MLX's `g3` kernels.
fn generalSource(comptime tag: []const u8, comptime expr: []const u8, comptime wide: bool, comptime three: bool) []const u8 {
    const names = [_][]const u8{ "x", "y", "z" };
    const n_in = exprInputs(expr);
    const idx = if (wide) "long" else "int";
    const loc = if (wide) "sushi_loc" else "sushi_loc32";
    comptime var source: []const u8 = "[[kernel]] void sushi_" ++ tag ++ (if (wide) "" else if (three) "_g3" else "_i32") ++ "(\n";
    inline for (0..n_in) |i| source = source ++ std.fmt.comptimePrint("  const device ${d}* in{d} [[buffer({d})]],\n  const constant {s}* s{d} [[buffer({d})]],\n", .{ i, i, 2 * i, idx, i, 2 * i + 1 });
    const base = 2 * n_in;
    source = source ++ std.fmt.comptimePrint(
        \\  device $O* out [[buffer({d})]],
        \\  const constant int* shape [[buffer({d})]],
        \\  const constant int& ndim [[buffer({d})]],
        \\  uint3 at [[thread_position_in_grid]],
        \\  uint3 grid [[threads_per_grid]]) {{
        \\
    , .{ base, base + 1, base + 2 });
    inline for (0..n_in) |i| source = source ++ if (three)
        std.fmt.comptimePrint("  const ${d} {s} = in{d}[int(at.x) * s{d}[2] + int(at.y) * s{d}[1] + int(at.z) * s{d}[0]];\n", .{ i, names[i], i, i, i, i })
    else
        std.fmt.comptimePrint("  const ${d} {s} = in{d}[{s}(at, shape, s{d}, ndim)];\n", .{ i, names[i], i, loc, i });
    const flat = if (wide) "sushi_flat(at, grid)" else "at.x + grid.x * (at.y + grid.y * at.z)";
    return source ++ "  out[" ++ flat ++ "] = " ++ expr ++ ";\n}\n";
}

/// `out[i] = expr` over the broadcast of up to three inputs, each read as `x`, `y`, `z` in its own type.
/// Before the seal: the contiguous copy and every concatenate part count at each dtype a model's state holds. A restore
/// that rolls a live state back copies whole tensors, and a sampled draft joins rows of a dtype no warm width reaches.
pub fn makeCopiesAhead() Error!void {
    inline for (.{ Dtype.bfloat16, Dtype.float16, Dtype.float32, Dtype.int32, Dtype.uint32, Dtype.uint8 }) |d| {
        _ = try pipelineFor("copy_v0", comptime flatSource("copy", "x", 1, 0), &.{d}, d);
        inline for (2..5) |k| _ = try pipelineFor(std.fmt.comptimePrint("concat{d}", .{k}), comptime concatSource(k), &.{ d, d }, d);
    }
}

fn elementwise(comptime tag: []const u8, comptime expr: []const u8, inputs: []const *Tensor, out_dtype: Dtype, out_shape: []const c_int) Error!*Tensor {
    const n_in = comptime exprInputs(expr);
    const base = 2 * n_in;
    if (inputs.len != n_in) return error.TensorShape;

    var types: [3]Dtype = undefined;
    for (inputs, 0..) |in, i| types[i] = in.dtype;
    const out = try core.fresh(out_dtype, out_shape);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    if (flatMask(inputs, out.size())) |mask| {
        const pipeline = switch (mask) {
            inline else => |m| try pipelineFor(tag ++ std.fmt.comptimePrint("_v{d}", .{m}), comptime flatSource(tag, expr, n_in, m), types[0..n_in], out_dtype),
        };
        const p = try core.pass();
        p.pipeline(pipeline);
        for (inputs, 0..) |in, i| core.bind(p, in, i);
        core.bindOut(p, out, n_in);
        const n: u32 = @intCast(out.size());
        p.constant(n, n_in + 1);
        const threads = (out.size() + flat_per_thread - 1) / flat_per_thread;
        core.dispatch(p, .{ .x = threads }, .{ .x = @min(threads, 256) });
        return out;
    }
    var narrow = out.size() <= std.math.maxInt(i32);
    for (inputs) |in| narrow = narrow and in.reach() <= std.math.maxInt(i32);
    const g = Grid.of(out_shape);
    const pipeline = if (narrow and g.ndim == 3)
        try pipelineFor(tag ++ "_g3", comptime generalSource(tag, expr, false, true), types[0..n_in], out_dtype)
    else if (narrow)
        try pipelineFor(tag ++ "_i32", comptime generalSource(tag, expr, false, false), types[0..n_in], out_dtype)
    else
        try pipelineFor(tag, comptime generalSource(tag, expr, true, false), types[0..n_in], out_dtype);
    const p = try core.pass();
    p.pipeline(pipeline);
    for (inputs, 0..) |in, i| {
        core.bind(p, in, 2 * i);
        const s = g.strides(in);
        if (narrow) {
            var s32: [max_dims]i32 = undefined;
            for (0..g.ndim) |d| s32[d] = @intCast(s[d]);
            p.bytes(std.mem.sliceAsBytes(s32[0..g.ndim]), 2 * i + 1);
        } else p.bytes(std.mem.sliceAsBytes(s[0..g.ndim]), 2 * i + 1);
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
    return broadcastDims(a.dims(), b.dims(), out);
}

fn broadcastDims(a: []const c_int, b: []const c_int, out: *[max_dims]c_int) Error![]const c_int {
    const n = @max(a.len, b.len);
    var r: [max_dims]c_int = undefined;
    for (0..n) |i| {
        const da: c_int = if (i + a.len >= n) a[i + a.len - n] else 1;
        const db: c_int = if (i + b.len >= n) b[i + b.len - n] else 1;
        if (da != db and da != 1 and db != 1) return error.TensorShape;
        r[i] = if (da == 1) db else da;
    }
    @memcpy(out[0..n], r[0..n]);
    return out[0..n];
}

/// An elementwise chain as one kernel: `expr` over one to three inputs of one dtype, broadcast together, in that dtype.
pub fn chain(comptime tag: []const u8, comptime expr: []const u8, inputs: []const *Tensor) Error!*Tensor {
    var buf: [max_dims]c_int = undefined;
    var shape: []const c_int = inputs[0].dims();
    for (inputs[1..]) |t| shape = try broadcastDims(shape, t.dims(), &buf);
    return elementwise("chain_" ++ tag, expr, inputs, inputs[0].dtype, shape);
}

pub const Binary = enum { add, subtract, multiply, divide, floor_divide, remainder, maximum, minimum, less, less_equal, greater, greater_equal, equal, logical_and, logical_or, logaddexp };

/// MLX's `promote_types` (mlx/dtype.cpp `type_rules`), indexed by `mlx_dtype`.
const promote_table = blk: {
    const D = Dtype;
    const r = [14][14]D{
        .{ .bool_, .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .uint8, .uint8, .uint16, .uint32, .uint64, .int16, .int16, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .uint16, .uint16, .uint16, .uint32, .uint64, .int32, .int32, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .uint32, .uint32, .uint32, .uint32, .uint64, .int64, .int64, .int64, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .uint64, .uint64, .uint64, .uint64, .uint64, .float32, .float32, .float32, .float32, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .int8, .int16, .int32, .int64, .float32, .int8, .int16, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .int16, .int16, .int32, .int64, .float32, .int16, .int16, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .int32, .int32, .int32, .int64, .float32, .int32, .int32, .int32, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .int64, .int64, .int64, .int64, .float32, .int64, .int64, .int64, .int64, .float16, .float32, .float64, .bfloat16, .complex64 },
        .{ .float16, .float16, .float16, .float16, .float16, .float16, .float16, .float16, .float16, .float16, .float32, .float64, .float32, .complex64 },
        .{ .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float32, .float64, .float32, .complex64 },
        .{ .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .float64, .complex64 },
        .{ .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .bfloat16, .float32, .float32, .float64, .bfloat16, .complex64 },
        .{ .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64, .complex64 },
    };
    break :blk r;
};

pub fn promoteTypes(a: Dtype, b: Dtype) Dtype {
    return promote_table[@intCast(@intFromEnum(a))][@intCast(@intFromEnum(b))];
}

/// An elementwise op of two tensors broadcast together, each cast first to the type they promote to, as MLX does. A
/// comparison gives bool.
pub fn binary(comptime op: Binary, a: *Tensor, b: *Tensor) Error!*Tensor {
    if (a.dtype != b.dtype) {
        const to = promoteTypes(a.dtype, b.dtype);
        const ca = try astype(a, to);
        defer core.release(ca);
        const cb = try astype(b, to);
        defer core.release(cb);
        return binary(op, ca, cb);
    }
    // Its arithmetic is MLX's compiled kernel's: a source of ours rounded differently in the last bits.
    if (op == .logaddexp) return stock.binaryOf("LogAddExp", .float, a, b);
    // The sign of a divisor's rule, a float's fmod and an integer's in one of MLX's kernels.
    if (op == .remainder) return stock.binaryOf("Remainder", .all, a, b);
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
        // MLX's `FloorDivide` for signed integers: truncate, then one less when the remainder and divisor differ in sign.
        .floor_divide => "$O(x / y - $O((x % y) != 0 && ((x < 0) != (y < 0))))",
        .maximum => "sushi_maximum(x, y)",
        .minimum => "sushi_minimum(x, y)",
        .less => "x < y",
        .less_equal => "x <= y",
        .greater => "x > y",
        .greater_equal => "x >= y",
        .equal => "x == y",
        .logical_and => "x && y",
        .logical_or => "x || y",
        .logaddexp, .remainder => unreachable,
    };
    return elementwise("bin_" ++ @tagName(op), expr, &.{ a, b }, if (compare) .bool_ else a.dtype, shape);
}

pub const Unary = enum { exp, log, abs, sigmoid, sqrt, rsqrt, square, floor, negative };

pub fn unary(comptime op: Unary, a: *Tensor) Error!*Tensor {
    if ((op == .log or op == .abs) and a.dtype != .float32 and a.dtype != .float16 and a.dtype != .bfloat16) return core.refusing("log or abs of a non-float here: the seam casts it or takes MLX's integer kernel first", error.TensorDtype);
    const expr = switch (op) {
        .exp => "$O(metal::precise::exp(x))",
        .log => "$O(metal::precise::log(x))",
        .abs => "$O(metal::abs(float(x)))",
        .sigmoid => "sushi_sigmoid(x)",
        .sqrt => "$O(metal::precise::sqrt(x))",
        .rsqrt => "$O(metal::precise::rsqrt(x))",
        .square => "x * x",
        .floor => "$O(floor(x))",
        .negative => "-x",
    };
    return elementwise("un_" ++ @tagName(op), expr, &.{a}, a.dtype, a.dims());
}

/// `cond ? a : b`, the three broadcast together.
pub fn where(cond_in: *Tensor, a_in: *Tensor, b_in: *Tensor) Error!*Tensor {
    // As MLX: the condition is read as bool, and values of two dtypes are cast to the type they promote to.
    const dtype = promoteTypes(a_in.dtype, b_in.dtype);
    const cond = if (cond_in.dtype == .bool_) try core.share(cond_in) else try astype(cond_in, .bool_);
    defer core.release(cond);
    const a = if (a_in.dtype == dtype) try core.share(a_in) else try astype(a_in, dtype);
    defer core.release(a);
    const b = if (b_in.dtype == dtype) try core.share(b_in) else try astype(b_in, dtype);
    defer core.release(b);
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
    if (t.size() == 0) {
        // Nothing to reduce into nothing. Over an axis of no elements into a result that has some, MLX fills in the
        // op's identity or refuses: not served.
        var shape: [max_dims]c_int = undefined;
        var n: usize = 0;
        var holds: usize = 1;
        for (t.dims(), 0..) |d, i| {
            if (i != axis) holds *= @intCast(d);
            if (i == axis and !keepdims) continue;
            shape[n] = if (i == axis) 1 else d;
            n += 1;
        }
        if (holds != 0) return core.refusing("a reduction over an empty axis into a result with elements", error.TensorUnsupported);
        return core.fresh(t.dtype, shape[0..n]);
    }
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
    const squeezed = try squeeze(out, &.{@intCast(axis)});
    core.release(out);
    return squeezed;
}

/// `base` with `update` written over the slice `start..stop` (step 1). The write waits until the result's bytes are
/// first used (`settle`): if by then the result is the last holder of `base`'s bytes, as it is once the caller has
/// dropped `base`, the update goes into them and nothing else is copied.
pub fn sliceUpdate(base: *Tensor, update: *Tensor, start: []const c_int, stop: []const c_int) Error!*Tensor {
    if (base.dtype != update.dtype) return core.refusing("a slice update whose dtype is not its destination's: the seam casts it first", error.TensorDtype);
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
    // Off while the ops below read `t`; a write that fails leaves the update owed, for the next read to make.
    t.pending = null;
    errdefer t.pending = p;
    // Through a broadcast axis one write would land on every element that axis reads.
    const in_place = !t.repeats() and try core.sole(t.store);
    if (!in_place) core.adopt(t, try contiguousCopy(t));
    var ones: [max_dims]c_int = @splat(1);
    const window = try slice(t, p.start[0..t.ndim], p.stop[0..t.ndim], ones[0..t.ndim]);
    defer core.release(window);
    try copyInto(window, p.update);
    core.countUpdate(in_place);
    core.freePending(p);
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
    // As MLX: the parts of two dtypes are cast to the type they promote to.
    var dtype = first.dtype;
    for (parts) |part| {
        if (part.ndim != first.ndim) return error.TensorShape;
        for (0..first.ndim) |i| {
            if (i != axis and part.shape[i] != first.shape[i]) return error.TensorShape;
        }
        shape[axis] += part.shape[axis];
        dtype = promoteTypes(dtype, part.dtype);
    }
    const out = try core.fresh(dtype, shape[0..first.ndim]);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    var joinable = parts.len >= 2 and parts.len <= 4 and out.size() <= std.math.maxInt(u32);
    for (parts) |part| joinable = joinable and part.dtype == dtype and part.rowContiguous();
    if (joinable) {
        // Two to four row-contiguous parts in one dispatch: each element copied from the part its axis position falls in.
        var inner: u32 = 1;
        for (first.dims()[axis + 1 ..]) |d| inner *= @intCast(d);
        // Until the seal, a dtype's first concatenate makes every part count's kernel: a verify's joins take the count
        // its row groups have, which the warm's widths do not all reach.
        if (!core.sealed) inline for (2..5) |k| {
            _ = try pipelineFor(std.fmt.comptimePrint("concat{d}", .{k}), comptime concatSource(k), &.{ dtype, dtype }, dtype);
        };
        const pipeline = switch (parts.len) {
            inline 2, 3, 4 => |k| try pipelineFor(std.fmt.comptimePrint("concat{d}", .{k}), comptime concatSource(k), &.{ dtype, dtype }, dtype),
            else => unreachable,
        };
        const p = try core.pass();
        p.pipeline(pipeline);
        for (parts, 0..) |part, i| core.bind(p, part, i);
        core.bindOut(p, out, parts.len);
        var extents: [4]u32 = @splat(0);
        for (parts, 0..) |part, i| extents[i] = @intCast(part.shape[axis]);
        const n: u32 = @intCast(out.size());
        p.constant(inner, parts.len + 1);
        p.bytes(std.mem.sliceAsBytes(&extents), parts.len + 2);
        p.constant(n, parts.len + 3);
        core.dispatch(p, .{ .x = n }, .{ .x = @min(n, 256) });
        return out;
    }
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

/// The kernel joining `k` row-contiguous parts along one axis: `inner` elements after the axis, each part's extent
/// along it in `ext`.
fn concatSource(comptime k: usize) []const u8 {
    const names = [_][]const u8{ "a", "b", "c", "d" };
    comptime var src: []const u8 = std.fmt.comptimePrint("[[kernel]] void sushi_concat{d}(\n", .{k});
    inline for (0..k) |i| src = src ++ std.fmt.comptimePrint("  const device $0* {s} [[buffer({d})]],\n", .{ names[i], i });
    src = src ++ std.fmt.comptimePrint(
        \\  device $O* out [[buffer({d})]],
        \\  const constant uint& inner [[buffer({d})]],
        \\  const constant uint* ext [[buffer({d})]],
        \\  const constant uint& n [[buffer({d})]],
        \\  uint i [[thread_position_in_grid]]) {{
        \\  if (i >= n) return;
        \\  uint total = 0;
        \\  for (uint j = 0; j < {d}u; ++j) total += ext[j];
        \\  uint within = i % inner;
        \\  uint row = i / inner;
        \\  uint pos = row % total;
        \\  uint outer = row / total;
        \\
    , .{ k, k + 1, k + 2, k + 3, k });
    comptime var start: []const u8 = "0u";
    inline for (0..k) |i| {
        const at = std.fmt.comptimePrint("ext[{d}]", .{i});
        const read = std.fmt.comptimePrint("{s}[(outer * {s} + pos - ({s})) * inner + within]", .{ names[i], at, start });
        src = src ++ if (i + 1 < k)
            std.fmt.comptimePrint("  if (pos < {s} + {s}) {{ out[i] = {s}; return; }}\n", .{ start, at, read })
        else
            std.fmt.comptimePrint("  out[i] = {s};\n", .{read});
        start = start ++ " + " ++ at;
    }
    return src ++ "}\n";
}

/// `t[indices]` along `axis`: the result has `indices`' shape in that axis's place.
pub fn take(t: *Tensor, indices: *Tensor, axis_in: c_int) Error!*Tensor {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim) return error.TensorShape;
    switch (indices.dtype) {
        .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => {},
        else => return core.refusing("take with indices that are not integers", error.TensorDtype),
    }
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

/// `start, start + step, ..` below `stop`, as MLX makes it: the length from the f64 arguments, each element
/// `start + i * step` in the dtype with `start` and `step = dtype(start + step) - start` rounded to it first. What MLX
/// refuses, and an int32 start or first step with no int32 value, is refused by name.
pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype) (Error || error{ ArangeLengthUndefined, ArangeStepZero, ArangeTooLong, ArangeOutOfRange })!*Tensor {
    if (std.math.isNan(start) or std.math.isNan(stop) or std.math.isNan(step) or std.math.isInf(start) or std.math.isInf(stop)) return error.ArangeLengthUndefined;
    if (step == 0) return error.ArangeStepZero;
    // An infinite step toward the stop holds the start alone.
    const lone = std.math.isInf(step) and ((step > 0 and start < stop) or (step < 0 and start > stop));
    const real = if (lone) 1 else @ceil((stop - start) / step);
    if (real > std.math.maxInt(i32)) return error.ArangeTooLong;
    const count: usize = @intFromFloat(@max(0, real));
    const out = try core.freshForHost(dtype, &.{@intCast(count)});
    errdefer core.release(out);
    const bytes = try core.hostBytes(out);
    switch (dtype) {
        .int32 => {
            const lo: f64 = std.math.minInt(i32);
            const hi: f64 = std.math.maxInt(i32);
            if (!(start >= lo and start <= hi) or (!lone and !(start + step >= lo and start + step <= hi))) return error.ArangeOutOfRange;
            const first: i32 = @intFromFloat(start);
            const delta: u32 = if (lone) 0 else @bitCast(@as(i32, @intFromFloat(start + step)) -% first);
            for (@as([]i32, @alignCast(std.mem.bytesAsSlice(i32, bytes))), 0..) |*v, i| {
                v.* = @bitCast(@as(u32, @bitCast(first)) +% @as(u32, @intCast(i)) *% delta);
            }
        },
        .uint32 => {
            const hi: f64 = std.math.maxInt(u32);
            if (!(start >= 0 and start <= hi) or (!lone and !(start + step >= 0 and start + step <= hi))) return error.ArangeOutOfRange;
            const first: u32 = @intFromFloat(start);
            const delta: u32 = if (lone) 0 else @as(u32, @intFromFloat(start + step)) -% first;
            for (@as([]u32, @alignCast(std.mem.bytesAsSlice(u32, bytes))), 0..) |*v, i| v.* = first +% @as(u32, @intCast(i)) *% delta;
        },
        .float32 => {
            const first: f32 = @floatCast(start);
            const delta: f32 = if (lone) 0 else @as(f32, @floatCast(start + step)) - first;
            for (@as([]f32, @alignCast(std.mem.bytesAsSlice(f32, bytes))), 0..) |*v, i| {
                // MLX's kernel computes `start + index * step`, which Metal contracts into one fused multiply-add.
                v.* = @mulAdd(f32, @floatFromInt(@as(u32, @intCast(i))), delta, first);
            }
        },
        else => return core.refusing("arange in a dtype other than int32, uint32 and f32", error.TensorDtype),
    }
    return out;
}

/// MLX's `full`: `vals` cast to `dtype` and broadcast to `shape`, written out (the `Full` primitive's copy).
pub fn full(shape: []const c_int, vals: *Tensor, dtype: Dtype) Error!*Tensor {
    for (shape) |d| if (d < 0) return error.TensorShape;
    const cast = try astype(vals, dtype);
    defer core.release(cast);
    const wide = try broadcastTo(cast, shape);
    defer core.release(wide);
    return contiguousCopy(wide);
}

/// MLX's `ones`: `full` of a one of `dtype`.
pub fn onesOf(shape: []const c_int, dtype: Dtype) Error!*Tensor {
    const one: i32 = 1;
    const v = try core.fromHost(std.mem.asBytes(&one), .int32, &.{});
    defer core.release(v);
    return full(shape, v, dtype);
}

/// MLX's `tile`: `t` reshaped with a unit axis before each repeated one, broadcast over the repeats, and flattened back.
pub fn tile(t: *Tensor, reps_in: []const c_int) Error!*Tensor {
    const rank = @max(reps_in.len, t.ndim);
    var reps: core.Dims = .{};
    var shape: core.Dims = .{};
    for (0..rank - reps_in.len) |_| try reps.append(1);
    try reps.appendSlice(reps_in);
    for (0..rank - t.ndim) |_| try shape.append(1);
    try shape.appendSlice(t.dims());
    var expand: core.Dims = .{};
    var broad: core.Dims = .{};
    var final: core.Dims = .{};
    for (reps.slice(), shape.slice()) |r, d| {
        if (r < 0) return error.TensorShape;
        if (r != 1) {
            try expand.append(1);
            try broad.append(r);
        }
        try expand.append(d);
        try broad.append(d);
        try final.append(std.math.mul(c_int, r, d) catch return error.TensorShape);
    }
    const x = try reshape(t, expand.slice());
    defer core.release(x);
    const wide = try broadcastTo(x, broad.slice());
    defer core.release(wide);
    return reshape(wide, final.slice());
}

/// MLX's `split` into `parts` equal pieces along `axis`: views of `t`.
pub fn split(t: *Tensor, parts: c_int, axis_in: c_int, out: []*Tensor) Error!void {
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, t.ndim) else axis_in);
    if (axis >= t.ndim or parts <= 0 or out.len != parts) return error.TensorShape;
    const width = t.shape[axis];
    if (@rem(width, parts) != 0) return error.TensorShape;
    const each = @divExact(width, parts);
    var start: [max_dims]c_int = @splat(0);
    var stop: [max_dims]c_int = undefined;
    const step: [max_dims]c_int = @splat(1);
    @memcpy(stop[0..t.ndim], t.dims());
    var made: usize = 0;
    errdefer for (out[0..made]) |o| core.release(o);
    for (out, 0..) |*o, i| {
        start[axis] = each * @as(c_int, @intCast(i));
        stop[axis] = start[axis] + each;
        o.* = try slice(t, start[0..t.ndim], stop[0..t.ndim], step[0..t.ndim]);
        made += 1;
    }
}

/// MLX's `isfinite`: neither infinite nor NaN; every integer and bool is.
pub fn isFinite(t: *Tensor) Error!*Tensor {
    switch (t.dtype) {
        .float32, .float16, .bfloat16 => return elementwise("isfinite", "bool(metal::isfinite(float(x)))", &.{t}, .bool_, t.dims()),
        .bool_, .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => {
            const yes: u8 = 1;
            const v = try core.fromHost(std.mem.asBytes(&yes), .bool_, &.{});
            defer core.release(v);
            return full(t.dims(), v, .bool_);
        },
        else => return core.refusing("isfinite of a dtype other than a float, an integer or bool", error.TensorDtype),
    }
}

/// A reduction over every axis: a result of no axes, or of `t`'s axes each one wide.
fn wholeOf(r: *Tensor, t: *const Tensor, keepdims: bool) Error!*Tensor {
    if (!keepdims) return r;
    defer core.release(r);
    const ones_shape: [max_dims]c_int = @splat(1);
    return reshape(r, ones_shape[0..t.ndim]);
}

/// MLX's `max` over every axis.
pub fn maxAll(t: *Tensor, keepdims: bool) Error!*Tensor {
    if (t.size() == 0) return core.refusing("the max of no elements", error.TensorShape);
    const flat = try reshape(t, &.{@intCast(t.size())});
    defer core.release(flat);
    return wholeOf(try reduce(.max, flat, 0, false), t, keepdims);
}

/// MLX's `all` over every axis, of a bool array: no element false.
pub fn allOf(t: *Tensor, keepdims: bool) Error!*Tensor {
    if (t.dtype != .bool_) return core.refusing("all of an array that is not bool", error.TensorDtype);
    if (t.size() == 0) {
        const yes: u8 = 1;
        return wholeOf(try core.fromHost(std.mem.asBytes(&yes), .bool_, &.{}), t, keepdims);
    }
    const flat = try reshape(t, &.{@intCast(t.size())});
    defer core.release(flat);
    const miss = try elementwise("all_miss", "$O(!x)", &.{flat}, .uint8, flat.dims());
    defer core.release(miss);
    const any = try reduce(.max, miss, 0, false);
    defer core.release(any);
    return wholeOf(try elementwise("all_none", "x == 0", &.{any}, .bool_, any.dims()), t, keepdims);
}

/// MLX's `any` over `axes` of a bool array (`Reduce::Or`): some element true. A bool OR is exact, so the reduction of
/// its 0/1 bytes gives MLX's bytes whatever kernel runs it.
pub fn anyAxes(t: *Tensor, axes: []const c_int, keepdims: bool) Error!*Tensor {
    if (t.dtype != .bool_) return core.refusing("any of an array that is not bool", error.TensorDtype);
    if (t.size() == 0) return core.refusing("any over an array of no elements", error.TensorUnsupported);
    var norm: [max_dims]c_int = undefined;
    for (axes, 0..) |ax, i| {
        const at = if (ax < 0) ax + @as(c_int, t.ndim) else ax;
        if (at < 0 or at >= t.ndim) return error.TensorShape;
        norm[i] = at;
    }
    var x = try elementwise("any_bytes", "$O(x)", &.{t}, .uint8, t.dims());
    defer core.release(x);
    for (norm[0..axes.len]) |at| {
        const r = try reduce(.max, x, at, true);
        core.release(x);
        x = r;
    }
    const kept = try elementwise("any_true", "x != 0", &.{x}, .bool_, x.dims());
    if (keepdims) return kept;
    defer core.release(kept);
    return squeeze(kept, norm[0..axes.len]);
}

/// MLX's `array_equal`: false for two shapes, else every element equal in the promoted type (NaN equal to NaN under
/// `equal_nan`, for floats).
pub fn arrayEqual(a: *Tensor, b: *Tensor, equal_nan: bool) Error!*Tensor {
    if (!std.mem.eql(c_int, a.dims(), b.dims())) {
        const no: u8 = 0;
        return core.fromHost(std.mem.asBytes(&no), .bool_, &.{});
    }
    const floats = [_]Dtype{ .float16, .bfloat16, .float32 };
    const dtype = promoteTypes(a.dtype, b.dtype);
    const af = std.mem.indexOfScalar(Dtype, &floats, dtype) != null;
    const ca = try astype(a, dtype);
    defer core.release(ca);
    const cb = try astype(b, dtype);
    defer core.release(cb);
    const eq = if (equal_nan and af)
        try elementwise("eq_nan", "x == y || (metal::isnan(float(x)) && metal::isnan(float(y)))", &.{ ca, cb }, .bool_, ca.dims())
    else
        try elementwise("eq_all", "x == y", &.{ ca, cb }, .bool_, ca.dims());
    defer core.release(eq);
    return allOf(eq, false);
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
    if (parts.len == 0) return error.TensorShape;
    const wide = try gpa.alloc(*Tensor, parts.len);
    defer gpa.free(wide);
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
    switch (indices.dtype) {
        .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => {},
        else => return core.refusing("take_along with indices that are not integers", error.TensorDtype),
    }
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
    switch (indices.dtype) {
        .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64 => {},
        else => return core.refusing("put_along with indices that are not integers", error.TensorDtype),
    }
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
    const squeezed = try squeeze(out, &.{@intCast(axis)});
    core.release(out);
    return squeezed;
}
