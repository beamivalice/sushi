//! An op scope for the deepseek_v41 forward: every intermediate an op chain makes is freed together,
//! so a layer's graph is built with `try sc.op(...)` and no per-array bookkeeping.
const std = @import("std");
const mlx = @import("../mlx.zig");
const exl3 = @import("sushi_exl3");

pub const A = mlx.mlx_array;
pub const S = mlx.mlx_stream;
pub const none: A = .{ .ctx = null };

pub fn dim(x: A, axis: isize) usize {
    const sh = mlx.getShape(x);
    const ax: usize = if (axis < 0) @intCast(@as(isize, @intCast(sh.len)) + axis) else @intCast(axis);
    return @intCast(sh[ax]);
}

pub fn ci(v: usize) c_int {
    return @intCast(v);
}

pub const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,
    /// How the trellis linears built through this scope launch.
    rows: exl3.project_rows.Mode = .{},

    /// A scope with the default launch mode, for tests, probes and code outside a pass. A scope inside a pass comes
    /// from `Model.scope()` or `child()`, or it drops the pass's exact rows.
    pub fn plain(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }

    /// A scope of its own over the same allocator, stream and launch mode.
    pub fn child(self: *const Scope) Scope {
        return .{ .a = self.a, .s = self.s, .rows = self.rows };
    }

    pub fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }

    pub fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    pub fn res(self: *Scope, rc: c_int, o: *const A) !A {
        mlx.check(rc) catch |e| {
            _ = mlx.mlx_array_free(o.*);
            return e;
        };
        return self.keep(o.*);
    }

    /// The scope's own handle on `x`, for an op that makes nothing new: like every op's result it stays valid
    /// after the caller frees `x`.
    pub fn share(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_array_set(&o, x), &o);
    }

    /// A handle on `x` that outlives the scope (the caller frees it).
    pub fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }

    pub fn op1(sc: *Scope, comptime func: anytype, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, sc.s), &o);
    }

    pub fn op2(sc: *Scope, comptime func: anytype, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, y, sc.s), &o);
    }

    pub fn add(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_add, x, y);
    }
    pub fn sub(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_subtract, x, y);
    }
    pub fn mul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_multiply, x, y);
    }
    pub fn div(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_divide, x, y);
    }
    pub fn matmul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_matmul, x, y);
    }
    pub fn min(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_minimum, x, y);
    }
    pub fn maximum(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_maximum, x, y);
    }
    pub fn lt(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_less, x, y);
    }
    pub fn ge(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_greater_equal, x, y);
    }

    // A scalar or a host array has no stream of its own: it is made on the side of the scope's.

    /// An f32 scalar: it promotes bf16 operands, so use `fl` where the operand's dtype must survive.
    pub fn f(sc: *Scope, v: f32) !A {
        const host = mlx.engine.enter(sc.s);
        defer host.end();
        return sc.keep(try mlx.newFloat(v));
    }
    /// A scalar of `dt`.
    pub fn fl(sc: *Scope, v: f32, dt: mlx.mlx_dtype) !A {
        return sc.astype(try sc.f(v), dt);
    }
    pub fn i(sc: *Scope, v: i32) !A {
        const host = mlx.engine.enter(sc.s);
        defer host.end();
        return sc.keep(try mlx.newInt(v));
    }
    pub fn ints(sc: *Scope, data: []const i32, shape: []const c_int) !A {
        const host = mlx.engine.enter(sc.s);
        defer host.end();
        return sc.keep(try mlx.newData(data.ptr, shape, .int32));
    }
    pub fn floats(sc: *Scope, data: []const f32, shape: []const c_int) !A {
        const host = mlx.engine.enter(sc.s);
        defer host.end();
        return sc.keep(try mlx.newData(data.ptr, shape, .float32));
    }

    pub fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        if (mlx.mlx_array_dtype(x) == dt) return sc.share(x);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    pub fn view(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_view(&o, x, dt, sc.s), &o);
    }
    pub fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    pub fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    pub fn t2(sc: *Scope, x: A) !A {
        return sc.transpose(x, &.{ 1, 0 });
    }
    pub fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    pub fn expand(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_expand_dims(&o, x, axis, sc.s), &o);
    }
    pub fn contiguous(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_contiguous(&o, x, false, sc.s), &o);
    }

    /// x[..., lo:hi, ...] on one axis (a negative axis counts from the end).
    pub fn slice(sc: *Scope, x: A, axis_in: isize, lo: usize, hi: usize) !A {
        const sh = mlx.getShape(x);
        const axis: usize = if (axis_in < 0) @intCast(@as(isize, @intCast(sh.len)) + axis_in) else @intCast(axis_in);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[axis] = @intCast(lo);
        stop[axis] = @intCast(hi);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, sc.s), &o);
    }
    pub fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        if (xs.len == 1) return sc.share(xs[0]);
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    pub fn stack(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_stack_axis(&o, vec, axis, sc.s), &o);
    }
    pub fn take(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, x, idx, axis, sc.s), &o);
    }
    pub fn takeAlong(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_along_axis(&o, x, idx, axis, sc.s), &o);
    }

    fn reduce(sc: *Scope, comptime f_: anytype, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(f_(&o, x, axis, keep_, sc.s), &o);
    }
    pub fn sum(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_sum_axis, x, axis, keep_);
    }
    pub fn mean(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_mean_axis, x, axis, keep_);
    }
    pub fn max(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_max_axis, x, axis, keep_);
    }
    pub fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, true, sc.s), &o);
    }
    /// Indices of the `k` largest along the last axis (unordered).
    pub fn topkIdx(sc: *Scope, x: A, k: usize) !A {
        const neg = try sc.op1(mlx.mlx_negative, x);
        var o = mlx.mlx_array_new();
        const part = try sc.res(mlx.mlx_argpartition_axis(&o, neg, @intCast(k - 1), -1, sc.s), &o);
        // Row-major: the routed experts' lease reads these ids on the host, which refuses a strided view.
        return sc.contiguous(try sc.slice(part, -1, 0, k));
    }
    /// Indices that sort `x` along axis 0.
    pub fn argsort(sc: *Scope, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_argsort_axis(&o, x, 0, sc.s), &o);
    }
    pub fn where(sc: *Scope, c: A, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_where(&o, c, x, y, sc.s), &o);
    }
    pub fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    pub fn arange(sc: *Scope, lo: usize, hi: usize) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_arange(&o, @floatFromInt(lo), @floatFromInt(hi), 1, .int32, sc.s), &o);
    }
    pub fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    pub fn clip(sc: *Scope, x: A, lo: f32, hi: f32) !A {
        return sc.min(try sc.maximum(x, try sc.f(lo)), try sc.f(hi));
    }
    pub fn sigmoid(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sigmoid, x);
    }
};

/// Evaluates `xs` together.
pub fn evalAll(xs: []const A) !void {
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

pub fn hostF32(a: A) ![]const f32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_float32(a) orelse return error.HostRead;
    return p[0..mlx.mlx_array_size(a)];
}

pub fn hostI32(a: A) ![]const i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_int32(a) orelse return error.HostRead;
    return p[0..mlx.mlx_array_size(a)];
}

/// Evaluates `a` and reads via `mlx.hostResult` so dry recordings receive fill-patterned slices without reading GPU memory.
pub fn hostResultF32(a: A) ![]const f32 {
    try mlx.check(mlx.mlx_array_eval(a));
    return mlx.hostResult(f32, a, null);
}

/// Evaluates `a` and reads via `mlx.hostResult`, bounding `.ramp` fill values below `below` when non-null.
pub fn hostResultI32(a: A, below: ?u32) ![]const i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    return mlx.hostResult(i32, a, below);
}

test "a scope's concat of one part and astype to the same dtype give a handle of the scope's own: the input freed and its box reused, the result keeps its bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    const want = [_]f32{ 1, 2, 3, 4 };
    for (0..2) |case| {
        var sc = Scope.plain(std.testing.allocator, mlx.engine.stream());
        defer sc.deinit();
        const part = sc.out(try sc.floats(&want, &.{4}));
        const got = if (case == 0) try sc.concat(&.{part}, 0) else try sc.astype(part, .float32);
        _ = mlx.mlx_array_free(part);
        _ = try sc.floats(&.{ 9, 9, 9, 9 }, &.{4});
        try mlx.check(mlx.mlx_array_eval(got));
        try std.testing.expectEqualSlices(f32, &want, mlx.mlx_array_data_float32(got).?[0..4]);
    }
}

test "a scope's top-k indices read back on the host: the op chain's slice of the partition is made row-major" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    var sc = Scope.plain(std.testing.allocator, mlx.engine.stream());
    defer sc.deinit();
    const x = try sc.floats(&.{ 0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4 }, &.{ 2, 4 });
    const ids = try hostI32(try sc.astype(try sc.topkIdx(x, 2), .int32));
    try std.testing.expectEqual(@as(usize, 4), ids.len);
    for ([_][2]i32{ .{ 1, 2 }, .{ 0, 2 } }, 0..) |want, r| {
        const row = ids[r * 2 ..][0..2];
        try std.testing.expect((row[0] == want[0] and row[1] == want[1]) or (row[0] == want[1] and row[1] == want[0]));
    }
}
