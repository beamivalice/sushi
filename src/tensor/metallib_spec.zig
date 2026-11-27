//! Single source of truth for Sushi Cooler's precompiled Metal kernel catalog (`sushi.metallib`).
//! Imported by:
//! - `gen_metallib.zig` (build-time codegen emitting `.metal` units compiled into `sushi.metallib`),
//! - `ops.zig` and `kernel.zig` (runtime lookup of precompiled function names + function constants),
//! - `stock.zig` (completeness test verifying every catalogued function exists in `sushi.metallib`).
//!
//! Pure Zig — imports no MLX, Metal, or Apple framework symbols so host-only build tools compile it directly.

const std = @import("std");
const chains = @import("chains.zig");

/// Global `[[function_constant(idx)]]` slot layout shared by every Sushi-generated `.metal` unit.
/// Metal only binds constants ODR-used by a specific kernel function, so declaring one shared block
/// at file scope lets multiple kernels coexist in a single `.metal` unit without duplicate indices.
pub const fc_int_base: u32 = 0;
pub const fc_int_count: u32 = 48;
pub const fc_bool_base: u32 = 48;
pub const fc_bool_count: u32 = 16;

/// Mirror of `mlx.mlx_dtype` values supported on Sushi Cooler (kept dependency-free for host codegen).
pub const Dtype = enum(u8) {
    bool_ = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    bfloat16 = 12,

    pub const all = [_]Dtype{
        .bool_,   .uint8, .uint16, .uint32, .uint64, .int8,
        .int16,   .int32, .int64,  .float16, .bfloat16, .float32,
    };
    pub const numeric = [_]Dtype{
        .uint8, .uint16, .uint32, .uint64, .int8, .int16,
        .int32, .int64,  .float16, .bfloat16, .float32,
    };
    pub const ints = [_]Dtype{
        .uint8, .uint16, .uint32, .uint64, .int8, .int16, .int32, .int64,
    };
    pub const floats = [_]Dtype{ .float16, .bfloat16, .float32 };

    pub fn fromInt(raw: anytype) ?Dtype {
        const v: u32 = @intCast(raw);
        return switch (v) {
            0 => .bool_,
            1 => .uint8,
            2 => .uint16,
            3 => .uint32,
            4 => .uint64,
            5 => .int8,
            6 => .int16,
            7 => .int32,
            8 => .int64,
            9 => .float16,
            10 => .float32,
            12 => .bfloat16,
            else => null,
        };
    }

    pub fn metalType(d: Dtype) []const u8 {
        return switch (d) {
            .bool_ => "bool",
            .uint8 => "uint8_t",
            .uint16 => "uint16_t",
            .uint32 => "uint32_t",
            .uint64 => "uint64_t",
            .int8 => "int8_t",
            .int16 => "int16_t",
            .int32 => "int32_t",
            .int64 => "int64_t",
            .float16 => "float16_t",
            .float32 => "float",
            .bfloat16 => "bfloat16_t",
        };
    }

    pub fn shortName(d: Dtype) []const u8 {
        return switch (d) {
            .bool_ => "b1",
            .uint8 => "u8",
            .uint16 => "u16",
            .uint32 => "u32",
            .uint64 => "u64",
            .int8 => "i8",
            .int16 => "i16",
            .int32 => "i32",
            .int64 => "i64",
            .float16 => "f16",
            .float32 => "f32",
            .bfloat16 => "bf16",
        };
    }

    pub fn isFloat(d: Dtype) bool {
        return d == .float16 or d == .bfloat16 or d == .float32;
    }

    pub fn isSignedInt(d: Dtype) bool {
        return d == .int8 or d == .int16 or d == .int32 or d == .int64;
    }

    pub fn isUnsignedInt(d: Dtype) bool {
        return d == .uint8 or d == .uint16 or d == .uint32 or d == .uint64;
    }

    pub fn sameBits(a: Dtype, b: Dtype) bool {
        const pairs = [_][2]Dtype{ .{ .int8, .uint8 }, .{ .int16, .uint16 }, .{ .int32, .uint32 }, .{ .int64, .uint64 } };
        for (pairs) |pair| {
            if ((a == pair[0] and b == pair[1]) or (a == pair[1] and b == pair[0])) return true;
        }
        return a == b;
    }
};

// ── Ops & Chains Catalog ──

pub const OpsUnit = enum {
    ops_copy_cast,
    ops_elem,
    ops_gather,
    ops_misc,

    pub fn filename(u: OpsUnit) []const u8 {
        return switch (u) {
            .ops_copy_cast => "sushi_ops_copy_cast.metal",
            .ops_elem => "sushi_ops_elem.metal",
            .ops_gather => "sushi_ops_gather.metal",
            .ops_misc => "sushi_ops_misc.metal",
        };
    }
};

pub const ElemVariant = enum {
    flat,
    g3,
    i32,
    wide,

    pub fn suffix(v: ElemVariant) []const u8 {
        return switch (v) {
            .flat => "_v",
            .g3 => "_g3",
            .i32 => "_i32",
            .wide => "_w",
        };
    }
};

pub const ElemDtypeRule = enum {
    /// T -> T over all 12 dtypes.
    all_same,
    /// T -> T over 11 numeric dtypes.
    numeric_same,
    /// T -> T over 8 integer dtypes.
    int_same,
    /// T -> T over 3 float dtypes.
    float_same,
    /// (T, T) -> bool over all 12 dtypes.
    all_cmp,
    /// (T, T) -> bool over 3 float dtypes.
    float_cmp,
    /// (bool, T, T) -> T over all 12 dtypes.
    where_all,
    /// float -> bool over 3 float dtypes.
    float_to_bool,
    /// bool -> uint8.
    bool_to_u8,
    /// uint8 -> bool.
    u8_to_bool,
    /// (in, out) over all 12x12 pairs where !sameBits(in, out).
    cast_pairs,
};

pub const ElemOp = struct {
    unit: OpsUnit,
    tag: []const u8,
    expr: []const u8,
    n_in: u8,
    rule: ElemDtypeRule,
};

pub const elem_ops = [_]ElemOp{
    // Copy & cast
    .{ .unit = .ops_copy_cast, .tag = "copy", .expr = "x", .n_in = 1, .rule = .all_same },
    .{ .unit = .ops_copy_cast, .tag = "cast", .expr = "$O(x)", .n_in = 1, .rule = .cast_pairs },

    // Binary ops
    .{ .unit = .ops_elem, .tag = "bin_add", .expr = "x + y", .n_in = 2, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "bin_subtract", .expr = "x - y", .n_in = 2, .rule = .numeric_same },
    .{ .unit = .ops_elem, .tag = "bin_multiply", .expr = "x * y", .n_in = 2, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "bin_divide", .expr = "x / y", .n_in = 2, .rule = .numeric_same },
    .{ .unit = .ops_elem, .tag = "bin_floor_divide", .expr = "$O(x / y - $O((x % y) != 0 && ((x < 0) != (y < 0))))", .n_in = 2, .rule = .int_same },
    .{ .unit = .ops_elem, .tag = "bin_maximum", .expr = "sushi_maximum(x, y)", .n_in = 2, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "bin_minimum", .expr = "sushi_minimum(x, y)", .n_in = 2, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "bin_less", .expr = "x < y", .n_in = 2, .rule = .all_cmp },
    .{ .unit = .ops_elem, .tag = "bin_less_equal", .expr = "x <= y", .n_in = 2, .rule = .all_cmp },
    .{ .unit = .ops_elem, .tag = "bin_greater", .expr = "x > y", .n_in = 2, .rule = .all_cmp },
    .{ .unit = .ops_elem, .tag = "bin_greater_equal", .expr = "x >= y", .n_in = 2, .rule = .all_cmp },
    .{ .unit = .ops_elem, .tag = "bin_equal", .expr = "x == y", .n_in = 2, .rule = .all_cmp },
    .{ .unit = .ops_elem, .tag = "bin_logical_and", .expr = "x && y", .n_in = 2, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "bin_logical_or", .expr = "x || y", .n_in = 2, .rule = .all_same },

    // Unary ops
    .{ .unit = .ops_elem, .tag = "un_exp", .expr = "$O(metal::precise::exp(x))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_log", .expr = "$O(metal::precise::log(x))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_abs", .expr = "$O(metal::abs(float(x)))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_sigmoid", .expr = "sushi_sigmoid(x)", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_sqrt", .expr = "$O(metal::precise::sqrt(x))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_rsqrt", .expr = "$O(metal::precise::rsqrt(x))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_square", .expr = "x * x", .n_in = 1, .rule = .all_same },
    .{ .unit = .ops_elem, .tag = "un_floor", .expr = "$O(floor(x))", .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_elem, .tag = "un_negative", .expr = "-x", .n_in = 1, .rule = .numeric_same },

    // Where & predicate/equality helpers
    .{ .unit = .ops_misc, .tag = "where", .expr = "x ? y : z", .n_in = 3, .rule = .where_all },
    .{ .unit = .ops_misc, .tag = "isfinite", .expr = "bool(metal::isfinite(float(x)))", .n_in = 1, .rule = .float_to_bool },
    .{ .unit = .ops_misc, .tag = "all_miss", .expr = "$O(!x)", .n_in = 1, .rule = .bool_to_u8 },
    .{ .unit = .ops_misc, .tag = "all_none", .expr = "x == 0", .n_in = 1, .rule = .u8_to_bool },
    .{ .unit = .ops_misc, .tag = "any_bytes", .expr = "$O(x)", .n_in = 1, .rule = .bool_to_u8 },
    .{ .unit = .ops_misc, .tag = "any_true", .expr = "x != 0", .n_in = 1, .rule = .u8_to_bool },
    .{ .unit = .ops_misc, .tag = "eq_nan", .expr = "x == y || (metal::isnan(float(x)) && metal::isnan(float(y)))", .n_in = 2, .rule = .float_cmp },
    .{ .unit = .ops_misc, .tag = "eq_all", .expr = "x == y", .n_in = 2, .rule = .all_cmp },

    // Fused chains (imported directly from src/tensor/chains.zig — single source of truth)
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.sigmoid_gated_add.tag, .expr = chains.sigmoid_gated_add.expr, .n_in = 3, .rule = .float_same },
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.twice_sigmoid.tag, .expr = chains.twice_sigmoid.expr, .n_in = 1, .rule = .float_same },
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.multiply_add.tag, .expr = chains.multiply_add.expr, .n_in = 3, .rule = .float_same },
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.sigmoid_gate.tag, .expr = chains.sigmoid_gate.expr, .n_in = 2, .rule = .float_same },
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.ple_gate.tag, .expr = chains.ple_gate.expr, .n_in = 2, .rule = .float_same },
    .{ .unit = .ops_misc, .tag = "chain_" ++ chains.silu.tag, .expr = chains.silu.expr, .n_in = 1, .rule = .float_same },
};

pub const concat_widths = [_]usize{ 2, 3, 4, 8, 16 };
pub const flat_per_thread: u32 = 4;

/// Canonical metallib function name for any `ops.zig` kernel.
/// Used identically by `gen_metallib.zig`, runtime `ops.zig`, and `stock.zig`'s completeness test.
pub fn opsFnName(buf: []u8, tag: []const u8, variant_suffix: []const u8, in_types: []const Dtype, out_type: Dtype) error{OutOfMemory}![]const u8 {
    var w: std.ArrayList(u8) = .initBuffer(buf);
    try w.appendSliceBounded("sushi_ops_");
    try w.appendSliceBounded(tag);
    try w.appendSliceBounded(variant_suffix);
    for (in_types) |d| {
        try w.appendBounded('_');
        try w.appendSliceBounded(d.shortName());
    }
    try w.appendBounded('_');
    try w.appendSliceBounded(out_type.shortName());
    return w.items;
}

/// Calls `cb(ctx, unit, fn_name, tag, expr, variant, in_types, out_type)` for EVERY elementwise op instance
/// in `elem_ops`, plus calls `special_cb` for non-elementwise `ops.zig` kernels.
/// Both `gen_metallib.zig` and `stock.zig` iterate this exact function so the metallib and completeness
/// test cannot drift apart by construction.
pub fn visitOps(
    ctx: anytype,
    comptime onElem: fn (@TypeOf(ctx), OpsUnit, []const u8, ElemOp, ElemVariant, []const Dtype, Dtype) anyerror!void,
    comptime onSpecial: fn (@TypeOf(ctx), OpsUnit, []const u8, SpecialOp, []const Dtype, Dtype) anyerror!void,
) !void {
    var name_buf: [128]u8 = undefined;
    const variants = [_]ElemVariant{ .flat, .g3, .i32, .wide };

    for (elem_ops) |op| {
        switch (op.rule) {
            .all_same => for (Dtype.all) |d| {
                const ins = [3]Dtype{ d, d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), ins[0..op.n_in], d) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, ins[0..op.n_in], d);
                }
            },
            .numeric_same => for (Dtype.numeric) |d| {
                const ins = [3]Dtype{ d, d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), ins[0..op.n_in], d) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, ins[0..op.n_in], d);
                }
            },
            .int_same => for (Dtype.ints) |d| {
                const ins = [3]Dtype{ d, d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), ins[0..op.n_in], d) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, ins[0..op.n_in], d);
                }
            },
            .float_same => for (Dtype.floats) |d| {
                const ins = [3]Dtype{ d, d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), ins[0..op.n_in], d) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, ins[0..op.n_in], d);
                }
            },
            .all_cmp => for (Dtype.all) |d| {
                const ins = [2]Dtype{ d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, .bool_) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, .bool_);
                }
            },
            .float_cmp => for (Dtype.floats) |d| {
                const ins = [2]Dtype{ d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, .bool_) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, .bool_);
                }
            },
            .where_all => for (Dtype.all) |d| {
                const ins = [3]Dtype{ .bool_, d, d };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, d) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, d);
                }
            },
            .float_to_bool => for (Dtype.floats) |d| {
                const ins = [1]Dtype{d};
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, .bool_) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, .bool_);
                }
            },
            .bool_to_u8 => {
                const ins = [1]Dtype{.bool_};
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, .uint8) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, .uint8);
                }
            },
            .u8_to_bool => {
                const ins = [1]Dtype{.uint8};
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, .bool_) catch unreachable;
                    try onElem(ctx, op.unit, name, op, v, &ins, .bool_);
                }
            },
            .cast_pairs => for (Dtype.all) |from| for (Dtype.all) |to| {
                if (Dtype.sameBits(from, to)) continue;
                const ins = [1]Dtype{from};
                const eff: ElemOp = .{
                    .unit = op.unit,
                    .tag = op.tag,
                    .expr = if (to == .bool_ and (from == .float32 or from == .bfloat16)) "sushi_true(x)" else op.expr,
                    .n_in = 1,
                    .rule = op.rule,
                };
                for (variants) |v| {
                    const name = opsFnName(&name_buf, op.tag, v.suffix(), &ins, to) catch unreachable;
                    try onElem(ctx, op.unit, name, eff, v, &ins, to);
                }
            },
        }
    }

    // Special non-elementwise ops
    for (Dtype.all) |d| {
        const name = opsFnName(&name_buf, "zeros", "_v", &.{}, d) catch unreachable;
        try onSpecial(ctx, .ops_copy_cast, name, .zeros, &.{}, d);
    }
    for (Dtype.all) |from| for (Dtype.all) |to| {
        const ins = [1]Dtype{from};
        const name = opsFnName(&name_buf, "copy_into", "", &ins, to) catch unreachable;
        try onSpecial(ctx, .ops_copy_cast, name, .copy_into, &ins, to);
    };
    for (Dtype.ints) |d| {
        const ins = [1]Dtype{d};
        try onSpecial(ctx, .ops_misc, opsFnName(&name_buf, "red_sum", "", &ins, d) catch unreachable, .red_sum, &ins, d);
        try onSpecial(ctx, .ops_misc, opsFnName(&name_buf, "red_max", "", &ins, d) catch unreachable, .red_max, &ins, d);
    }
    for (concat_widths) |w| {
        var tag_buf: [16]u8 = undefined;
        const tag = std.fmt.bufPrint(&tag_buf, "concat{d}", .{w}) catch unreachable;
        for (Dtype.all) |d| {
            const ins = [1]Dtype{d};
            const name = opsFnName(&name_buf, tag, "", &ins, d) catch unreachable;
            try onSpecial(ctx, .ops_misc, name, .{ .concat = w }, &ins, d);
        }
    }
    for (Dtype.all) |val_t| for (Dtype.ints) |idx_t| {
        const ins = [2]Dtype{ val_t, idx_t };
        try onSpecial(ctx, .ops_gather, opsFnName(&name_buf, "take", "", &ins, val_t) catch unreachable, .take, &ins, val_t);
        try onSpecial(ctx, .ops_gather, opsFnName(&name_buf, "take_along", "", &ins, val_t) catch unreachable, .take_along, &ins, val_t);
        try onSpecial(ctx, .ops_gather, opsFnName(&name_buf, "put_along", "", &ins, val_t) catch unreachable, .put_along, &ins, val_t);
    };
    for (Dtype.all) |d| {
        const ins = [1]Dtype{d};
        try onSpecial(ctx, .ops_gather, opsFnName(&name_buf, "pad", "", &ins, d) catch unreachable, .pad, &ins, d);
    }
}

pub const SpecialOp = union(enum) {
    zeros,
    copy_into,
    red_sum,
    red_max,
    concat: usize,
    take,
    take_along,
    put_along,
    pad,
};

/// Metal helper header shared by all `sushi_ops_*.metal` units.
/// Matches `ops.zig`'s exact arithmetic (`sushi_sigmoid`, `sushi_maximum`, `sushi_minimum`, `sushi_true`).
pub const ops_metal_helpers =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\typedef bfloat bfloat16_t;
    \\typedef half float16_t;
    \\
    \\constant bool FC_B0 [[function_constant(48)]];
    \\constant bool FC_B1 [[function_constant(49)]];
    \\constant bool FC_B2 [[function_constant(50)]];
    \\
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
