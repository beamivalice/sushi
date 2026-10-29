//! Prefill KDA recurrence: one SIMD group carries independent value rows.
//! FP32 state and each member's original 32-lane reductions remain.
const std = @import("std");
const mlx = @import("../mlx.zig");
const primitive = @import("glm5_next.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
const SOURCE =
    \\#define GLM_VALUE_UNROLL _Pragma("clang loop unroll(full)")
    \\const uint n=thread_position_in_grid.z;
    \\const uint batch=n/uint(H),head=n%uint(H);
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint dv0=thread_position_in_grid.y*uint(R);
    \\const device InT* qp=q+(size_t(batch)*uint(T)*uint(H)+head)*128u;
    \\const device InT* kp=k+(size_t(batch)*uint(T)*uint(H)+head)*128u;
    \\const device InT* vp=v+(size_t(batch)*uint(T)*uint(H)+head)*128u;
    \\const device float* gp=g+(size_t(batch)*uint(T)*uint(H)+head)*128u;
    \\const auto bp=beta+size_t(batch)*uint(T)*uint(H);
    \\device OutT* yp=y+(size_t(batch)*uint(T)*uint(H)+head)*128u;
    \\float state[R][4];
    \\GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) GLM_VALUE_UNROLL for(uint i=0;i<4u;++i)
    \\ state[r][i]=state_in[(size_t(n)*128u+dv0+r)*128u+4u*lane+i];
    \\for(int t=0;t<T;++t) {
    \\ float keys[4],memory[R]={};
    \\ GLM_VALUE_UNROLL for(uint i=0;i<4u;++i) {
    \\  const uint key=4u*lane+i;
    \\  keys[i]=float(kp[key]);
    \\  const float decay=gp[key];
    \\  GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) {
    \\   state[r][i]=state[r][i]*decay;
    \\   memory[r]+=state[r][i]*keys[i];
    \\  }
    \\ }
    \\ GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) memory[r]=simd_sum(memory[r]);
    \\ const auto bet=bp[size_t(t)*uint(H)+head];
    \\ float delta[R];
    \\ GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) delta[r]=(vp[dv0+r]-memory[r])*bet;
    \\ float output[R]={};
    \\ GLM_VALUE_UNROLL for(uint i=0;i<4u;++i) {
    \\  const float query=float(qp[4u*lane+i]);
    \\  GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) {
    \\   state[r][i]=state[r][i]+keys[i]*delta[r];
    \\   output[r]+=state[r][i]*query;
    \\  }
    \\ }
    \\ GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) {
    \\  output[r]=simd_sum(output[r]);
    \\  if(thread_index_in_simdgroup==0u) yp[dv0+r]=OutT(output[r]);
    \\ }
    \\ qp+=uint(H)*128u;kp+=uint(H)*128u;gp+=uint(H)*128u;
    \\ vp+=uint(H)*128u;yp+=uint(H)*128u;
    \\}
    \\GLM_VALUE_UNROLL for(uint r=0;r<uint(R);++r) GLM_VALUE_UNROLL for(uint i=0;i<4u;++i)
    \\ state_out[(size_t(n)*128u+dv0+r)*128u+4u*lane+i]=state[r][i];
;
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}
pub fn resetDispatchCount() void {
    dispatches = 0;
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn spanFits(shape: []const c_int) bool {
    var count: u64 = 1;
    for (shape) |d| {
        if (d <= 0) return false;
        count = std.math.mul(u64, count, @intCast(d)) catch return false;
    }
    return count <= std.math.maxInt(c_int);
}

pub fn run(input: primitive.KdaInputs, rows_per_simd: u32, s: mlx.mlx_stream) !?primitive.KdaResult {
    if (rows_per_simd != 1 and rows_per_simd != 2 and rows_per_simd != 4) return error.InvalidKdaValueRows;
    if (!mlx.streamIsGpu(s)) return null;
    for ([_]Arr{ input.q, input.k, input.v, input.decay, input.beta, input.state }) |a| if (a.ctx == null) return null;
    const shape = mlx.getShape(input.q);
    if (shape.len != 4 or shape[0] <= 0 or shape[1] < 2 or shape[2] <= 0 or shape[3] != 128 or !spanFits(shape)) return null;
    for ([_]Arr{ input.k, input.v, input.decay }) |a| if (!std.mem.eql(c_int, shape, mlx.getShape(a))) return null;
    const state_shape = [_]c_int{ shape[0], shape[2], 128, 128 };
    if (!spanFits(&state_shape) or !std.mem.eql(c_int, &state_shape, mlx.getShape(input.state)) or !std.mem.eql(c_int, shape[0..3], mlx.getShape(input.beta))) return null;
    const dtype = mlx.mlx_array_dtype(input.q);
    if ((dtype != .bfloat16 and dtype != .float32) or mlx.mlx_array_dtype(input.k) != dtype or mlx.mlx_array_dtype(input.v) != dtype or mlx.mlx_array_dtype(input.decay) != .float32 or mlx.mlx_array_dtype(input.state) != .float32 or (mlx.mlx_array_dtype(input.beta) != .float32 and mlx.mlx_array_dtype(input.beta) != .bfloat16)) return null;
    if (kernel == null) {
        const iv = mlx.mlx_vector_string_new_data(&.{ "q", "k", "v", "g", "beta", "state_in", "T" }, 7);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&.{ "y", "state_out" }, 2);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_kda_value_rows", iv, ov, SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &state_shape, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @intCast(128 / rows_per_simd), try std.math.mul(c_int, shape[0], shape[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", @intCast(rows_per_simd)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", shape[2]));
    const length = mlx.mlx_array_new_int(shape[1]);
    defer _ = mlx.mlx_array_free(length);
    const iv = mlx.mlx_vector_array_new_data(&.{ input.q, input.k, input.v, input.decay, input.beta, input.state, length }, 7);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg, s));
    var result = primitive.KdaResult{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.state, ov, 1));
    dispatches += 1;
    return result;
}

fn random(ops: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, scale: f32) !Arr {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const a = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(a, shape.ptr, shape.len, dtype, 0, scale, key.*, ops.s));
    return a.*;
}
fn fixture(ops: *Ops, b: c_int, t: c_int, h: c_int, dtype: mlx.mlx_dtype, beta_dtype: mlx.mlx_dtype, unit_decay: bool) !primitive.KdaInputs {
    const shape = [_]c_int{ b, t, h, 128 };
    return .{
        .q = try random(ops, &shape, dtype, 227, 0.0078125),
        .k = try random(ops, &shape, dtype, 229, 0.0625),
        .v = try random(ops, &shape, dtype, 233, 0.2),
        .decay = if (unit_decay) try ops.ones(&shape, .float32) else try ops.unary(.exp, try ops.binary(.mul, try ops.unary(.sigmoid, try random(ops, &shape, .float32, 241, 1)), try ops.scalar(-0.1, .float32))),
        .beta = try ops.unary(.sigmoid, try random(ops, &.{ b, t, h }, beta_dtype, 251, 1)),
        .state = try random(ops, &.{ b, h, 128, 128 }, .float32, 239, 0.03),
    };
}
fn exact(a: Arr, b: Arr, s: mlx.mlx_stream) !void {
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    try mlx.check(mlx.mlx_array_eval(x));
    try mlx.check(mlx.mlx_array_eval(y));
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32) try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(x).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(y).?[0..n])) else try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(x).?[0..n], mlx.mlx_array_data_bfloat16(y).?[0..n]);
}

test "GLM KDA value-row candidate keeps native bits with nonzero state" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |beta_dtype| {
        for ([_]c_int{ 2, 3, 17, 128 }) |tokens| {
            var ops = Ops{ .s = s };
            defer ops.deinit();
            const in = try fixture(&ops, 2, tokens, 3, dtype, beta_dtype, tokens == 17);
            const expected = try primitive.kda(in, s);
            defer expected.deinit();
            for ([_]u32{ 1, 2, 4 }) |r| {
                const got = (try run(in, r, s)) orelse return error.TestExpectedValueRows;
                defer got.deinit();
                try exact(expected.y, got.y, s);
                try exact(expected.state, got.state, s);
            }
        }
    };
}

test "GLM KDA value-row candidate keeps production state and irregular continuation" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]c_int{ 512, 2048 }) |tokens| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const in = try fixture(&ops, 1, tokens, 64, dtype, .bfloat16, true);
        const expected = try primitive.kda(in, s);
        defer expected.deinit();
        for ([_]u32{ 1, 2, 4 }) |r| {
            const got = (try run(in, r, s)) orelse return error.TestExpectedValueRows;
            defer got.deinit();
            try exact(expected.y, got.y, s);
            try exact(expected.state, got.state, s);
            var state = try ops.result(in.state);
            defer _ = mlx.mlx_array_free(state);
            const ends = [_]c_int{ 0, 17, 80, tokens };
            var parts: [3]Arr = undefined;
            var made: usize = 0;
            defer for (parts[0..made]) |part| {
                _ = mlx.mlx_array_free(part);
            };
            for (0..3) |i| {
                var scope = Ops{ .s = s };
                defer scope.deinit();
                const lo = ends[i];
                const hi = ends[i + 1];
                const out = (try run(.{ .q = try scope.slice(in.q, 1, lo, hi), .k = try scope.slice(in.k, 1, lo, hi), .v = try scope.slice(in.v, 1, lo, hi), .decay = try scope.slice(in.decay, 1, lo, hi), .beta = try scope.slice(in.beta, 1, lo, hi), .state = state }, r, s)) orelse return error.TestExpectedValueRows;
                defer out.deinit();
                parts[i] = try scope.result(out.y);
                made += 1;
                try mlx.check(mlx.mlx_array_set(&state, out.state));
            }
            try exact(expected.y, try ops.concat(parts[0..made], 1), s);
            try exact(expected.state, state, s);
        }
    };
}

test "GLM KDA value-row admission rejects unchecked spans and decode" {
    try std.testing.expect(!spanFits(&.{ 1, std.math.maxInt(c_int), 64, 128 }));
    try std.testing.expect(!spanFits(&.{ 0, 1, 128, 128 }));
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const in = try fixture(&ops, 1, 1, 1, .bfloat16, .bfloat16, false);
    try std.testing.expect((try run(in, 2, s)) == null);
    try std.testing.expectError(error.InvalidKdaValueRows, run(in, 3, s));
}
