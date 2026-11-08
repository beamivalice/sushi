//! A bounded list of MLX arrays freed together, and the affine-width reader the eager graphs dequantize by.
//! Shared by every arch that builds a short eager graph, so it is named by capability.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;

pub const Ops = struct {
    s: mlx.mlx_stream,
    values: [768]Arr = undefined,
    count: usize = 0,

    pub fn deinit(self: *Ops) void {
        for (self.values[0..self.count]) |value| _ = mlx.mlx_array_free(value);
    }

    pub fn slot(self: *Ops) !*Arr {
        if (self.count == self.values.len) return error.GraphTooLarge;
        self.values[self.count] = mlx.mlx_array_new();
        self.count += 1;
        return &self.values[self.count - 1];
    }

    pub fn own(self: *Ops, value: Arr) !Arr {
        if (self.count == self.values.len) {
            _ = mlx.mlx_array_free(value);
            return error.GraphTooLarge;
        }
        self.values[self.count] = value;
        self.count += 1;
        return value;
    }

    pub fn cast(self: *Ops, x: Arr, dtype: mlx.mlx_dtype) !Arr {
        if (mlx.mlx_array_dtype(x) == dtype) return x;
        const out = try self.slot();
        try mlx.check(mlx.mlx_astype(out, x, dtype, self.s));
        return out.*;
    }

    pub fn reshape(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_reshape(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    pub fn transpose(self: *Ops, x: Arr, axes: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_transpose_axes(out, x, axes.ptr, axes.len, self.s));
        return out.*;
    }

    pub fn contiguous(self: *Ops, x: Arr) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_contiguous(out, x, false, self.s));
        return out.*;
    }

    pub fn slice(self: *Ops, x: Arr, axis: usize, start: c_int, end: c_int) !Arr {
        const shape = mlx.getShape(x);
        if (shape.len > 4 or axis >= shape.len or start < 0 or end < start or end > shape[axis]) return error.InvalidShape;
        var lo: [4]c_int = @splat(0);
        var hi: [4]c_int = @splat(0);
        const step: [4]c_int = @splat(1);
        @memcpy(hi[0..shape.len], shape);
        lo[axis] = start;
        hi[axis] = end;
        const out = try self.slot();
        try mlx.check(mlx.mlx_slice(out, x, &lo, shape.len, &hi, shape.len, &step, shape.len, self.s));
        return out.*;
    }

    pub fn concat(self: *Ops, parts: []const Arr, axis: c_int) !Arr {
        const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        const out = try self.slot();
        try mlx.check(mlx.mlx_concatenate_axis(out, vec, axis, self.s));
        return out.*;
    }

    pub fn scalar(self: *Ops, value: f32, dtype: mlx.mlx_dtype) !Arr {
        return self.cast(try self.own(mlx.mlx_array_new_float(value)), dtype);
    }

    pub fn ones(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_ones(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    pub fn zeros(self: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_zeros(out, shape.ptr, shape.len, dtype, self.s));
        return out.*;
    }

    const Binary = enum { add, sub, mul, div, min, max, mm, less, le, land, lor, floor_div };
    pub fn binary(self: *Ops, comptime operation: Binary, a: Arr, b: Arr) !Arr {
        const function = switch (operation) {
            .add => mlx.mlx_add,
            .sub => mlx.mlx_subtract,
            .mul => mlx.mlx_multiply,
            .div => mlx.mlx_divide,
            .min => mlx.mlx_minimum,
            .max => mlx.mlx_maximum,
            .mm => mlx.mlx_matmul,
            .less => mlx.mlx_less,
            .le => mlx.mlx_less_equal,
            .land => mlx.mlx_logical_and,
            .lor => mlx.mlx_logical_or,
            .floor_div => mlx.mlx_floor_divide,
        };
        const out = try self.slot();
        try mlx.check(function(out, a, b, self.s));
        return out.*;
    }

    const Unary = enum { exp, sigmoid, rsqrt, negative };
    pub fn unary(self: *Ops, comptime operation: Unary, x: Arr) !Arr {
        const function = switch (operation) {
            .exp => mlx.mlx_exp,
            .sigmoid => mlx.mlx_sigmoid,
            .rsqrt => mlx.mlx_rsqrt,
            .negative => mlx.mlx_negative,
        };
        const out = try self.slot();
        try mlx.check(function(out, x, self.s));
        return out.*;
    }

    pub fn reduce(self: *Ops, x: Arr, axis: c_int, mean: bool, keep: bool) !Arr {
        const out = try self.slot();
        if (mean) try mlx.check(mlx.mlx_mean_axis(out, x, axis, keep, self.s)) else try mlx.check(mlx.mlx_sum_axis(out, x, axis, keep, self.s));
        return out.*;
    }

    pub fn rms(self: *Ops, x: Arr, weight: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_rms_norm(out, x, weight, eps, self.s));
        return out.*;
    }

    pub fn layerNorm(self: *Ops, x: Arr, weight: Arr, bias: Arr, eps: f32) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_fast_layer_norm(out, x, weight, bias, eps, self.s));
        return out.*;
    }

    pub fn softmax(self: *Ops, x: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_softmax_axis(out, x, axis, true, self.s));
        return out.*;
    }

    pub fn take(self: *Ops, x: Arr, indices: Arr, axis: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_take_axis(out, x, indices, axis, self.s));
        return out.*;
    }

    pub fn broadcast(self: *Ops, x: Arr, shape: []const c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_broadcast_to(out, x, shape.ptr, shape.len, self.s));
        return out.*;
    }

    /// `qmmBits` for a tensor whose bits are not bound: they are read from the stored grids.
    pub fn qmm(self: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr, transposed: bool) !Arr {
        return self.qmmBits(x, w, scales, biases, transposed, try storedAffineBits(w, scales, biases));
    }

    pub fn qmmBits(self: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr, transposed: bool, bits: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_quantized_matmul(out, x, w, scales, biases, transposed, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(bits), "affine", self.s));
        return out.*;
    }

    pub fn dequant(self: *Ops, w: Arr, scales: Arr, biases: Arr) !Arr {
        return self.dequantBits(w, scales, biases, try storedAffineBits(w, scales, biases));
    }

    pub fn dequantBits(self: *Ops, w: Arr, scales: Arr, biases: Arr, bits: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_dequantize(out, w, scales, biases, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(bits), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, self.s));
        return out.*;
    }

    pub fn silu(self: *Ops, x: Arr) !Arr {
        // The source rounds sigmoid to the activation dtype before multiplying.
        return self.binary(.mul, x, try self.unary(.sigmoid, x));
    }

    pub fn conv(self: *Ops, x: Arr, weight: Arr, groups: c_int) !Arr {
        const out = try self.slot();
        try mlx.check(mlx.mlx_conv1d(out, x, weight, 1, 0, 1, groups, self.s));
        return out.*;
    }

    pub fn result(_: *Ops, x: Arr) !Arr {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_set(&out, x));
        return out;
    }
};

/// Read affine precision from stored grids, including sliced MLA/embedding tensors.
/// Supported grids are BF16 scales and biases at group size 128.
pub fn storedAffineBits(w: Arr, scales: Arr, biases: Arr) !c_int {
    if (w.ctx == null or scales.ctx == null or biases.ctx == null or mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(scales) != .bfloat16 or mlx.mlx_array_dtype(biases) != .bfloat16) return error.InvalidAffine;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(scales);
    if (ws.len < 2 or ws.len != ss.len or !std.mem.eql(c_int, ss, mlx.getShape(biases)) or
        !std.mem.eql(c_int, ws[0 .. ws.len - 1], ss[0 .. ss.len - 1])) return error.InvalidAffine;
    for (ws) |d| if (d <= 0) return error.InvalidAffine;
    const groups = ss[ss.len - 1];
    if (groups <= 0) return error.InvalidAffine;
    const packed_bits = @as(u64, @intCast(ws[ws.len - 1])) * 32;
    const width = @as(u64, @intCast(groups)) * 128;
    if (packed_bits % width != 0) return error.InvalidAffine;
    const bits = packed_bits / width;
    if (bits != 5 and bits != 6 and bits != 8) return error.InvalidAffine;
    return @intCast(bits);
}

/// Two arrays equal bit for bit (compared as f32), for tests.
pub fn expectArrayBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    // MLX gives an empty array no data pointer.
    const count = mlx.mlx_array_size(a);
    if (count == 0) return;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const av = try ops.cast(a, .float32);
    const bv = try ops.cast(b, .float32);
    try mlx.check(mlx.mlx_array_eval(av));
    try mlx.check(mlx.mlx_array_eval(bv));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(av).?[0..count]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(bv).?[0..count]));
}
