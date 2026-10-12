//! Exact mHC coefficients for 1-4 rows: one uniform SIMD32 subgroup per row.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const Result = @import("glm5_next.zig").HcResult;
const Ops = @import("glm5_model.zig").Ops;
var calls: usize = 0;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}
pub fn geometry(x: []const c_int, mixes: []const c_int, scale: []const c_int, base: []const c_int, iters: c_int, epsilon: f32) bool {
    return x.len == 4 and x[0] == 1 and x[1] >= 1 and x[1] <= 4 and std.mem.eql(c_int, x[2..], &.{ 4, 4096 }) and std.mem.eql(c_int, mixes, &.{ 1, x[1], 24 }) and
        std.mem.eql(c_int, scale, &.{3}) and std.mem.eql(c_int, base, &.{24}) and iters == 20 and std.math.isFinite(epsilon) and epsilon > 0;
}
const COEFFICIENTS: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row=threadgroup_position_in_grid.x;
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    \\threadgroup float pre[4];
    \\threadgroup float matrix[16];
    \\const float epsilon=float(eps);
    \\if(sg==0u) {
    \\  if(lane<4u) {
    \\    float z=mixes[row*24u+lane]*scale[0]+base[lane];
    \\    pre[lane]=1.0f/(1.0f+precise::exp(-z))+epsilon;
    \\    z=mixes[row*24u+4u+lane]*scale[1]+base[4u+lane];
    \\    post[row*4u+lane]=2.0f/(1.0f+precise::exp(-z));
    \\  }
    \\  float value=0.0f;
    \\  if(lane<16u) value=mixes[row*24u+8u+lane]*scale[2]+base[8u+lane];
    \\  const uint j=(lane&15u)/4u, k=lane&3u;
    \\  float maximum=-INFINITY;
    \\  for(uint c=0u;c<4u;++c) maximum=max(maximum,simd_shuffle(value,j*4u+c));
    \\  value=precise::exp(value-maximum);
    \\  float sum=0.0f;
    \\  for(uint c=0u;c<4u;++c) sum+=simd_shuffle(value,j*4u+c);
    \\  value=value/sum+epsilon;
    \\  for(uint iter=0u;iter<20u;++iter) {
    \\    if(iter!=0u) {
    \\      sum=0.0f;
    \\      for(uint c=0u;c<4u;++c) sum+=simd_shuffle(value,j*4u+c);
    \\      value/=sum+epsilon;
    \\    }
    \\    sum=0.0f;
    \\    for(uint r=0u;r<4u;++r) sum+=simd_shuffle(value,r*4u+k);
    \\    value/=sum+epsilon;
    \\  }
    \\  if(lane<16u) {matrix[lane]=value;comb[row*16u+lane]=value;}
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;
const SOURCE = COEFFICIENTS ++
    \\for(uint d=tid;d<4096u;d+=256u) {
    \\  float value=0.0f;
    \\  for(uint j=0u;j<4u;++j) value+=pre[j]*float(x[(row*4u+j)*4096u+d]);
    \\  mixed[row*4096u+d]=OutT(value);
    \\}
;
// Match MLX rms_single_row: 1024 threads, four consecutive values per thread.
const NORMALIZED_SOURCE = COEFFICIENTS ++
    \\float vals[4];
    \\for(uint i=0;i<4;++i) {
    \\ uint d=tid*4+i;float value=0.0f;
    \\ for(uint j=0;j<4;++j)value+=pre[j]*float(x[(row*4u+j)*4096u+d]);
    \\ vals[i]=float(bfloat(value));
    \\}
    \\{
    \\#pragma clang fp contract(on)
    \\#pragma clang fp reassociate(on)
    \\ threadgroup float sums[32];
    \\ threadgroup float inv[1];
    \\ float acc=0.0f;
    \\ for(uint i=0;i<4;++i)acc+=vals[i]*vals[i];
    \\ acc=simd_sum(acc);
    \\ if(sg==0)sums[lane]=0;
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ if(lane==0)sums[sg]=acc;
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ if(sg==0){acc=simd_sum(sums[lane]);if(lane==0)inv[0]=metal::precise::rsqrt(acc/4096.0f+float(norm_eps));}
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ for(uint i=0;i<4;++i)mixed[row*4096+tid*4+i]=norm_w[tid*4+i]*bfloat(vals[i]*inv[0]);
    \\}
;
var kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);
pub const Norm = struct { weight: Arr, epsilon: f32 };
pub fn collapse(x: Arr, mixes: Arr, scale: Arr, base: Arr, iters: c_int, epsilon: f32, s: mlx.mlx_stream) !?Result {
    return run(x, mixes, scale, base, iters, epsilon, s, null);
}
pub fn collapseNormalized(x: Arr, mixes: Arr, scale: Arr, base: Arr, iters: c_int, epsilon: f32, s: mlx.mlx_stream, norm: Norm) !?Result {
    return run(x, mixes, scale, base, iters, epsilon, s, norm);
}
fn run(x: Arr, mixes: Arr, scale: Arr, base: Arr, iters: c_int, epsilon: f32, s: mlx.mlx_stream, norm: ?Norm) !?Result {
    if (!mlx.streamIsGpu(s)) return null;
    for ([_]Arr{ x, mixes, scale, base }) |a| if (a.ctx == null) return null;
    if (!geometry(mlx.getShape(x), mlx.getShape(mixes), mlx.getShape(scale), mlx.getShape(base), iters, epsilon) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    for ([_]Arr{ mixes, scale, base }) |a| if (mlx.mlx_array_dtype(a) != .float32) return null;
    if (norm) |n| {
        if (!enabled() or !@import("glm5_model.zig").naxArms() or (mlx.getShape(x)[1] != 1 and mlx.getShape(x)[1] != 3) or n.weight.ctx == null or
            mlx.mlx_array_dtype(n.weight) != .bfloat16 or !std.mem.eql(c_int, mlx.getShape(n.weight), &.{4096}) or
            !std.math.isFinite(n.epsilon) or n.epsilon <= 0) return null;
    }
    const kernel = &kernels[@intFromBool(norm != null)];
    if (kernel.* == null) {
        const names = [_][*:0]const u8{ "x", "mixes", "scale", "base", "eps", "norm_w", "norm_eps" };
        const ins = mlx.mlx_vector_string_new_data(&names, if (norm != null) 7 else 5);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "mixed", "post", "comb" }, 3);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new(if (norm != null) "sushi_glm_hc_collapse_norm" else "sushi_glm_hc_collapse_simd32", ins, outs, if (norm != null) NORMALIZED_SOURCE else SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel.* = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const rows = mlx.getShape(x)[1];
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, 4096 }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, 4 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, 4, 4 }, 4, .float32));
    const threads: c_int = if (norm != null) 1024 else 256;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", .bfloat16));
    const eps = mlx.mlx_array_new_float(epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const norm_eps = if (norm) |n| mlx.mlx_array_new_float(n.epsilon) else Arr{ .ctx = null };
    defer if (norm_eps.ctx != null) {
        _ = mlx.mlx_array_free(norm_eps);
    };
    const arrays = [_]Arr{ x, mixes, scale, base, eps, if (norm) |n| n.weight else x, norm_eps };
    const inputs = mlx.mlx_vector_array_new_data(&arrays, if (norm != null) 7 else 5);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.*.?, inputs, cfg, s));
    if (mlx.mlx_vector_array_size(outputs) != 3) return error.MetalKernelBadOutputCount;
    var result = Result{ .mixed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.mixed, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.post, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.comb, outputs, 2));
    calls += 1;
    return result;
}

fn exact(ops: *Ops, a: Arr, b: Arr) !void {
    const x = try ops.contiguous(a);
    const y = try ops.contiguous(b);
    const ev = mlx.mlx_vector_array_new_data(&.{ x, y }, 2);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(x), mlx.getShape(y));
    try std.testing.expectEqual(mlx.mlx_array_dtype(x), mlx.mlx_array_dtype(y));
    const n = mlx.mlx_array_size(x);
    if (mlx.mlx_array_dtype(x) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(x).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(y).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(x).?[0..n], mlx.mlx_array_data_bfloat16(y).?[0..n]);
}
fn check(ops: *Ops, x: Arr, mix: Arr, scale: Arr, base: Arr, expected_calls: usize) !void {
    const hc = @import("glm5_next.zig");
    const model = @import("glm5_model.zig");
    resetDispatchCount();
    const reference = blk: {
        model.reference_numerics = true;
        defer model.reference_numerics = false;
        break :blk try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    };
    defer reference.deinit();
    try std.testing.expectEqual(@as(usize, 0), dispatchCount());
    const new = try hc.hcCollapse(x, mix, scale, base, 20, 1e-6, ops.s);
    defer new.deinit();
    try std.testing.expectEqual(expected_calls, dispatchCount());
    try exact(ops, reference.mixed, new.mixed);
    try exact(ops, reference.post, new.post);
    try exact(ops, reference.comb, new.comb);
    if (mlx.mlx_array_dtype(x) == .bfloat16 and (mlx.getShape(x)[1] == 1 or mlx.getShape(x)[1] == 3) and @import("glm5_model.zig").naxArms()) {
        const weight = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{4096}, 997, ops.s));
        for ([_]f32{ 1e-6, 1e-5 }) |eps| {
            const normalized = (try collapseNormalized(x, mix, scale, base, 20, 1e-6, ops.s, .{ .weight = weight, .epsilon = eps })) orelse return error.TestExpectedHcNorm;
            defer normalized.deinit();
            try exact(ops, try ops.rms(new.mixed, weight, eps), normalized.mixed);
            try exact(ops, new.post, normalized.post);
            try exact(ops, new.comb, normalized.comb);
        }
    }
}
fn fixtureTensor(comptime name: []const u8) ![]const u8 {
    const bytes = @embedFile("fixtures/glm5_layers.safetensors");
    const header_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    const header = bytes[8 .. 8 + header_len];
    const at = std.mem.indexOf(u8, header, "\"" ++ name ++ "\"") orelse return error.MissingFixture;
    const key = "\"data_offsets\":[";
    const o = std.mem.indexOfPos(u8, header, at, key) orelse return error.MissingFixture;
    const rest = header[o + key.len ..];
    const comma = std.mem.indexOfScalar(u8, rest, ',') orelse return error.MissingFixture;
    const close = std.mem.indexOfScalar(u8, rest, ']') orelse return error.MissingFixture;
    const lo = try std.fmt.parseInt(usize, rest[0..comma], 10);
    const hi = try std.fmt.parseInt(usize, rest[comma + 1 .. close], 10);
    return bytes[8 + header_len + lo .. 8 + header_len + hi];
}
fn bf16At(raw: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16);
}
fn f32At(raw: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
}

test "GLM HC SIMD32 primitive hook exact T1-T4 specials and original fallback" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var values: [96]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) / 8 - 1;
    const scale = try ops.own(mlx.mlx_array_new_data(&[_]f32{ 0.125, 0.25, 0.0625 }, &.{3}, 1, .float32));
    var bases: [24]f32 = undefined;
    for (&bases, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 16 - 0.25;
    const base = try ops.own(mlx.mlx_array_new_data(&bases, &.{24}, 1, .float32));
    const wide = try ops.own(try @import("dflash.zig").TinyFix.bf16ArrShaped(&.{ 1, 4, 4, 4096 }, 731, ops.s));
    // Hostile rows: signed zeros, NaN/Inf, saturated logits and opposite huge comb logits.
    var special = values;
    special[0] = -0.0;
    special[1] = 0.0;
    special[8] = std.math.nan(f32);
    special[9] = std.math.inf(f32);
    special[10] = -std.math.inf(f32);
    special[24 + 8] = 1000;
    special[24 + 9] = -1000;
    for (0..16) |i| special[72 + 8 + i] = if (i % 2 == 0) 1e30 else -1e30;
    for (1..5) |t| {
        const x = try ops.slice(wide, 1, 0, @intCast(t));
        for ([_]*const [96]f32{ &values, &special }) |set| {
            const mix = try ops.own(mlx.mlx_array_new_data(set, &.{ 1, @intCast(t), 24 }, 3, .float32));
            try check(&ops, x, mix, scale, base, 1);
        }
    }
    const mix3 = try ops.own(mlx.mlx_array_new_data(&values, &.{ 1, 3, 24 }, 3, .float32));
    try check(&ops, try ops.cast(try ops.slice(wide, 1, 0, 3), .float32), mix3, scale, base, 0);
}

test "GLM HC SIMD32 exact on the captured layer's coefficients at T1-T4" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const input = try fixtureTensor("hc.input");
    const fn_w = try fixtureTensor("h.hc_attn_fn");
    var scale_v: [3]f32 = undefined;
    var base_v: [24]f32 = undefined;
    for (&scale_v, 0..) |*v, i| v.* = f32At(try fixtureTensor("h.hc_attn_scale"), i);
    for (&base_v, 0..) |*v, i| v.* = f32At(try fixtureTensor("h.hc_attn_base"), i);
    const scale = try ops.own(mlx.mlx_array_new_data(&scale_v, &.{3}, 1, .float32));
    const base = try ops.own(mlx.mlx_array_new_data(&base_v, &.{24}, 1, .float32));
    // Captured rows are 4x128; tiling a stream 32x keeps its RMS, so the captured mixes stay valid at D=4096.
    var mix_v: [4 * 24]f32 = undefined;
    const x_v = try std.testing.allocator.alloc(u16, 4 * 4 * 4096);
    defer std.testing.allocator.free(x_v);
    for (0..4) |r| {
        var ss: f32 = 0;
        for (0..512) |k| ss += bf16At(input, r * 512 + k) * bf16At(input, r * 512 + k);
        const inv = 1.0 / @sqrt(ss / 512 + 1e-6);
        for (0..24) |m| {
            var acc: f32 = 0;
            for (0..512) |k| acc += bf16At(input, r * 512 + k) * inv * bf16At(fn_w, m * 512 + k);
            mix_v[r * 24 + m] = acc;
        }
        for (0..4) |j| for (0..4096) |d| {
            x_v[(r * 4 + j) * 4096 + d] = std.mem.readInt(u16, input[(r * 512 + j * 128 + d % 128) * 2 ..][0..2], .little);
        };
    }
    const x = try ops.own(mlx.mlx_array_new_data(x_v.ptr, &.{ 1, 4, 4, 4096 }, 4, .bfloat16));
    const mix = try ops.own(mlx.mlx_array_new_data(&mix_v, &.{ 1, 4, 24 }, 3, .float32));
    for (1..5) |t| try check(&ops, try ops.slice(x, 1, 0, @intCast(t)), try ops.slice(mix, 1, 0, @intCast(t)), scale, base, 1);
}
