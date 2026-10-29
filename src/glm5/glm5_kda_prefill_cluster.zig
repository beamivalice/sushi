//! One prefill GEMM over the concatenated KDA f_a/g_a/beta banks; the stored weights stay.
const std = @import("std");
const mlx = @import("../mlx.zig");
const native = @import("glm5_model.zig");
pub const widths = [_]c_int{ 128, 128, 64 };
pub const weight_bytes: usize = 320 * 4096 * 2;
pub const transient_bytes: usize = 2048 * 320 * 2;
var calls: usize = 0;
pub fn enabled() bool {
    return !native.reference_numerics and native.naxArms();
}
pub fn resetDispatchCount() void {
    calls = 0;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!enabled() or chunk != 2048) return 0;
    return std.math.mul(usize, transient_bytes, pending_layers);
}
pub fn eligible(bank: [3]native.Linear) bool {
    for (bank, widths) |linear, n| {
        if (linear.w.ctx == null or linear.scales.ctx != null or linear.biases.ctx != null or
            linear.input != 4096 or linear.output != n or mlx.mlx_array_dtype(linear.w) != .bfloat16 or
            !std.mem.eql(c_int, mlx.getShape(linear.w), &.{ n, 4096 })) return false;
    }
    return true;
}

pub fn prepare(ops: *native.Ops, bank: [3]native.Linear) !mlx.mlx_array {
    if (!eligible(bank)) return error.InvalidGlmPrefillCluster;
    var weights: [3]mlx.mlx_array = undefined;
    for (bank, 0..) |linear, i| weights[i] = linear.w;
    return ops.contiguous(try ops.concat(&weights, 0));
}

pub fn tryProject(ops: *native.Ops, weight: mlx.mlx_array, x: mlx.mlx_array) !?[3]mlx.mlx_array {
    if (weight.ctx == null or x.ctx == null or !enabled() or !mlx.streamIsGpu(ops.s) or
        mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 2048, 4096 })) return null;
    const result = try project(ops, weight, x);
    calls += 1;
    return result;
}

pub fn project(ops: *native.Ops, weight: mlx.mlx_array, x: mlx.mlx_array) ![3]mlx.mlx_array {
    if (weight.ctx == null or x.ctx == null or mlx.mlx_array_dtype(weight) != .bfloat16 or
        mlx.mlx_array_dtype(x) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(weight), &.{ 320, 4096 }) or
        !std.mem.eql(c_int, mlx.getShape(x), &.{ 1, 2048, 4096 })) return error.InvalidGlmPrefillCluster;
    const joined = try ops.binary(.mm, x, try ops.transpose(weight, &.{ 1, 0 }));
    var out: [3]mlx.mlx_array = undefined;
    var offset: c_int = 0;
    for (widths, 0..) |n, i| {
        // Existing fused prework directly indexes beta's compact rows. Return
        // compact banks so callers keep their current layout contract.
        out[i] = try ops.contiguous(try ops.slice(joined, 2, offset, offset + n));
        offset += n;
    }
    return out;
}
