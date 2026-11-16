//! MLX's dtype rules (mlx/ops.cpp, MIT, see NOTICE), applied by the seam before an op of the engine runs: the dtype
//! MLX casts an input to and the dtype an op gives back, in one place, so no entry derives them by hand.
const core = @import("core.zig");
const ops = @import("ops.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;

pub fn isFloat(d: Dtype) bool {
    return switch (d) {
        .float16, .bfloat16, .float32, .float64 => true,
        else => false,
    };
}

/// `at_least_float`: a float keeps its dtype, anything else is float32.
pub fn atLeastFloat(d: Dtype) Dtype {
    return if (isFloat(d)) d else .float32;
}

/// `sum`'s output: a signed integer of four bytes or less sums as int32, an unsigned one as uint32, bool as int32.
pub fn sumOf(d: Dtype) Dtype {
    return switch (d) {
        .bool_, .int8, .int16, .int32 => .int32,
        .uint8, .uint16, .uint32 => .uint32,
        else => d,
    };
}

/// `cumsum`'s output: bool scans as int32, every other dtype as itself.
pub fn cumsumOf(d: Dtype) Dtype {
    return if (d == .bool_) .int32 else d;
}

/// `promote_types` when both are floats, else null.
pub fn promoteFloats(a: Dtype, b: Dtype) ?Dtype {
    if (!isFloat(a) or !isFloat(b)) return null;
    return ops.promoteTypes(a, b);
}

/// `t` as `d`, shared when it already is.
pub fn as(t: *Tensor, d: Dtype) core.Error!*Tensor {
    return if (t.dtype == d) core.share(t) else ops.astype(t, d);
}
