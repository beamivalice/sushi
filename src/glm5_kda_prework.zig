//! Parallel GLM KDA prework; recurrence and output normalization remain separate.
const std = @import("std");
const mlx = @import("mlx.zig");
const unary = @import("glm5_kda_fused.zig");
const Arr = mlx.mlx_array;
pub const Inputs = struct {
    qkv: Arr,
    /// The q, k and v projections as separate `[1, rows, width]` arrays, read in place of `qkv`.
    parts: ?[3]Arr = null,
    a: Arr,
    beta: Arr,
    conv_weight: Arr,
    exp_a: Arr,
    dt_bias: Arr,
    conv_state: ?Arr = null,
    heads: c_int,
    lower: f32 = -5,
};
pub const Result = struct {
    q: Arr,
    k: Arr,
    v: Arr,
    decay: Arr,
    beta: Arr,
    conv: Arr,
    pub fn arrays(self: Result) [6]Arr {
        return .{ self.q, self.k, self.v, self.decay, self.beta, self.conv };
    }
    pub fn deinit(self: Result) void {
        for (self.arrays()) |a| {
            _ = mlx.mlx_array_free(a);
        }
    }
};
pub const TreeResult = struct {
    q: Arr,
    k: Arr,
    v: Arr,
    decay: Arr,
    beta: Arr,
    pub fn arrays(self: TreeResult) [5]Arr {
        return .{ self.q, self.k, self.v, self.decay, self.beta };
    }
    pub fn deinit(self: TreeResult) void {
        for (self.arrays()) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn applyTree(s: mlx.mlx_stream, in: Inputs, parents: []const i32) !?TreeResult {
    if (parents.len == 0 or parents.len > 16 or parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, row| if (parent < 0 or parent >= row) return error.InvalidGlmDraftTree;
    if (@import("builtin").is_test and force_reference_for_tests) return null;
    if (!mlx.streamIsGpu(s) or in.heads < 1 or in.heads > @divTrunc(std.math.maxInt(c_int), 384) or in.lower != -5) return null;
    for ([_]Arr{ in.qkv, in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias }) |a| if (a.ctx == null) return null;
    const shape = mlx.getShape(in.qkv);
    const width = in.heads * 128;
    if (shape.len != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != 3 * width or
        @as(i64, shape[1]) * shape[2] > std.math.maxInt(c_int) or mlx.mlx_array_dtype(in.qkv) != .bfloat16) return null;
    const rows = shape[1];
    if (rows != parents.len) return error.InvalidGlmDraftTree;
    if (!geometryFits(rows, in.heads)) return null;
    for ([_]Arr{ in.a, in.beta, in.conv_weight }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (!std.mem.eql(c_int, &.{ 1, rows, width }, mlx.getShape(in.a)) or !std.mem.eql(c_int, &.{ 1, rows, in.heads }, mlx.getShape(in.beta)) or
        !std.mem.eql(c_int, &.{ 3 * width, 4, 1 }, mlx.getShape(in.conv_weight)) or
        !std.mem.eql(c_int, &.{in.heads}, mlx.getShape(in.exp_a)) or mlx.mlx_array_dtype(in.exp_a) != .float32 or
        !std.mem.eql(c_int, &.{width}, mlx.getShape(in.dt_bias)) or mlx.mlx_array_dtype(in.dt_bias) != .float32) return null;
    if (in.conv_state) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16 or !std.mem.eql(c_int, &.{ 1, 3, 3 * width }, mlx.getShape(a))) return null;
    var windows: [16 * 4]i32 = undefined;
    for (0..parents.len) |row| for (0..4) |tap| {
        var back = 3 - tap;
        var at: i32 = @intCast(row);
        while (back > 0 and at >= 0) {
            at = parents[@intCast(at)];
            back -= 1;
        }
        windows[row * 4 + tap] = if (at >= 0) 3 + at else 2 - @as(i32, @intCast(back));
    };
    const window_ids = mlx.mlx_array_new_data(&windows, &[_]c_int{rows * 4}, 1, .int32);
    defer _ = mlx.mlx_array_free(window_ids);
    const mode = (try unary.unaryModes(s)) orelse return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for ([_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .bfloat16, .float32 }) |dtype| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads, 128 }, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * in.heads, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", in.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HAS_CONV", @intFromBool(in.conv_state != null)));
    inline for (.{ "SIG_B", "SIG_F", "EXP_F", "RSQ_F" }, .{ mode.sig_b, mode.sig_f, mode.exp_f, mode.rsq_f }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, @intFromBool(value)));
    const length = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(length);
    const constants = mlx.mlx_array_new_data(&[_]f32{ 1 / @sqrt(@as(f32, 128)), 1e-6, in.lower, 0 }, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(constants);
    const iv = mlx.mlx_vector_array_new_data(&.{ in.qkv, in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias, in.conv_state orelse in.qkv, length, constants, window_ids }, 10);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getTreeKernel(), iv, cfg, s));
    var arrays: [5]Arr = undefined;
    var made: usize = 0;
    errdefer for (arrays[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&arrays, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    }
    tree_count += 1;
    return .{ .q = arrays[0], .k = arrays[1], .v = arrays[2], .decay = arrays[3], .beta = arrays[4] };
}
var count: usize = 0;
var force_reference_for_tests = false;
pub fn forceReferenceForTest(on: bool) void {
    if (@import("builtin").is_test) force_reference_for_tests = on;
}

pub fn dispatchCount() usize {
    return count;
}
pub fn resetDispatchCount() void {
    count = 0;
}
// Arithmetic follows the one-token body and oMLX's prework mapping;
// retained BF16 convolution/SiLU stores and FP32 normalization order are explicit.
const HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\using namespace metal;
    \\template<typename T,bool P> inline T sigmoid(T x) {
    \\ auto y=1/(1+(P?metal::precise::exp(metal::abs(x)):metal::exp(metal::abs(x))));
    \\ return x<0?y:1-y;
    \\}
;
const SOURCE: [:0]const u8 =
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    \\const uint head=threadgroup_position_in_grid.x,row=threadgroup_position_in_grid.y;
    \\const uint width=uint(H)*128u,channels=3u*width;
    \\const uint rows=uint(length);
    \\threadgroup T qs[128],ks[128];
    \\const uint base=(row*uint(H)+head)*128u;
    \\for(uint part=0;part<3u;++part) {
    \\ const uint channel=part*width+head*128u+tid;
    \\ float acc=0.0f;
    \\ for(uint tap=0;tap<4u;++tap) {
    \\   const uint pos=row+tap;
    \\   T x=pos<3u?(HAS_CONV?previous[pos*channels+channel]:T(0)):qkv[(pos-3u)*channels+channel];
    \\   acc+=float(x)*float(conv_w[channel*4u+tap]);
    \\ }
    \\ const T co=T(acc),sig=sigmoid<T,SIG_B>(co);
    \\ const T act=co*sig;
    \\ if(part==0u) qs[tid]=act; else if(part==1u) ks[tid]=act; else v_out[base+tid]=act;
    \\ if(row==0u) {
    \\   for(uint j=0;j<3u;++j) {
    \\     const uint pos=rows+j;
    \\     const ushort raw=pos<3u?(HAS_CONV?((const device ushort*)previous)[pos*channels+channel]:ushort(0)):((const device ushort*)qkv)[(pos-3u)*channels+channel];
    \\     // Single-token views preserve subnormal bits without a float conversion.
    \\     if(rows==1u) ((device ushort*)conv_out)[j*channels+channel]=raw;
    \\     else conv_out[j*channels+channel]=T(float(as_type<T>(raw))+constants[3]);
    \\   }
    \\ }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if(sg<2u) {
    \\ threadgroup T* values=sg==0u?qs:ks;
    \\ float x[4],total=0.0f;
    \\ for(uint i=0;i<4u;++i) {x[i]=float(values[lane*4u+i]);float sq=x[i]*x[i];total=sq+total;}
    \\ total=simd_sum(total);
    \\ const float arg=total+constants[1];
    \\ const float inv=RSQ_F?metal::precise::rsqrt(arg):metal::rsqrt(arg);
    \\ for(uint i=0;i<4u;++i) {
    \\   const float normalized=x[i]*inv;
    \\   if(sg==0u) q_out[base+lane*4u+i]=T(normalized*constants[0]);
    \\   else k_out[base+lane*4u+i]=T(normalized);
    \\ }
    \\}
    \\const float shift=float(a[base+tid])+dt_bias[head*128u+tid];
    \\const float scaled=exp_a[head]*shift;
    \\const float forget=sigmoid<float,SIG_F>(scaled);
    \\const float gate=constants[2]*forget;
    \\decay_out[base+tid]=EXP_F?metal::precise::exp(gate):metal::exp(gate);
    \\if(tid==0u) beta_out[row*uint(H)+head]=sigmoid<T,SIG_B>(beta[row*uint(H)+head]);
;
// Reuse that arithmetic verbatim, changing only history lookup and
// omitting the linear-prefill tail materialization. Tape replay owns raw tails.
const TREE_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(100000);
    const begin = std.mem.indexOf(u8, SOURCE, " if(row==0u) {").?;
    const end = std.mem.indexOfPos(u8, SOURCE, begin, "\n }\n}\nthreadgroup_barrier").?;
    const stripped = SOURCE[0..begin] ++ SOURCE[end + 3 ..];
    const find = "const uint pos=row+tap;";
    const at = std.mem.indexOf(u8, stripped, find).?;
    break :blk stripped[0..at] ++ "const uint pos=uint(windows[row*4u+tap]);" ++ stripped[at + find.len ..];
};
var tree_kernel: ?mlx.mlx_fast_metal_kernel = null;
var tree_count: usize = 0;
pub fn treeDispatchCount() usize {
    return tree_count;
}
fn getTreeKernel() !mlx.mlx_fast_metal_kernel {
    if (tree_kernel) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(&.{ "qkv", "a", "beta", "conv_w", "exp_a", "dt_bias", "previous", "length", "constants", "windows" }, 10);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&.{ "q_out", "k_out", "v_out", "decay_out", "beta_out" }, 5);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_kda_tree_prework", iv, ov, TREE_SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    tree_kernel = k;
    return k;
}

// The joined kernel with each part's rows addressed at `width` instead of `channels`.
const PARTS_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(100000);
    const edits = [_][2][]const u8{
        .{ "const uint channel=part*width+head*128u+tid;", "const uint channel=part*width+head*128u+tid,inner=head*128u+tid;\n const device T* src=part==0u?qkv0:(part==1u?qkv1:qkv2);" },
        .{ "qkv[(pos-3u)*channels+channel]", "src[(pos-3u)*width+inner]" },
        .{ "((const device ushort*)qkv)[(pos-3u)*channels+channel]", "((const device ushort*)src)[(pos-3u)*width+inner]" },
    };
    var text: []const u8 = SOURCE;
    for (edits) |edit| {
        const at = std.mem.indexOf(u8, text, edit[0]).?;
        text = text[0..at] ++ edit[1] ++ text[at + edit[0].len ..];
    }
    break :blk text ++ "";
};
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var parts_kernel: ?mlx.mlx_fast_metal_kernel = null;
var parts_count: usize = 0;
pub fn partsDispatchCount() usize {
    return parts_count;
}
fn getKernel(split: bool) !mlx.mlx_fast_metal_kernel {
    if (if (split) parts_kernel else kernel) |k| return k;
    const iv = if (split)
        mlx.mlx_vector_string_new_data(&.{ "qkv0", "qkv1", "qkv2", "a", "beta", "conv_w", "exp_a", "dt_bias", "previous", "length", "constants" }, 11)
    else
        mlx.mlx_vector_string_new_data(&.{ "qkv", "a", "beta", "conv_w", "exp_a", "dt_bias", "previous", "length", "constants" }, 9);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&.{ "q_out", "k_out", "v_out", "decay_out", "beta_out", "conv_out" }, 6);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(if (split) "sushi_glm_kda_prework_parts" else "sushi_glm_kda_prework", iv, ov, if (split) PARTS_SOURCE else SOURCE, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    if (split) parts_kernel = k else kernel = k;
    return k;
}
fn geometryFits(rows: c_int, heads: c_int) bool {
    if (rows < 1 or heads < 1) return false;
    const channels = std.math.mul(u64, @intCast(heads), 384) catch return false;
    const input_count = std.math.mul(u64, @intCast(rows), channels) catch return false;
    const weights = std.math.mul(u64, channels, 4) catch return false;
    const tail = std.math.mul(u64, channels, 3) catch return false;
    const grid = std.math.mul(u64, @intCast(heads), 128) catch return false;
    // Shader offsets are uint32; a span of 2^32 elements has last index UINT_MAX.
    return channels <= std.math.maxInt(c_int) and input_count <= std.math.maxInt(c_int) and
        weights - 1 <= std.math.maxInt(u32) and tail - 1 <= std.math.maxInt(u32) and grid <= std.math.maxInt(c_int);
}

pub fn apply(s: mlx.mlx_stream, in: Inputs) !?Result {
    if (@import("builtin").is_test and force_reference_for_tests) return null;
    if (!mlx.streamIsGpu(s) or in.heads < 1 or in.heads > @divTrunc(std.math.maxInt(c_int), 384) or in.lower != -5) return null;
    const split = in.parts != null;
    const joined = [1]Arr{in.qkv};
    const sources: []const Arr = if (in.parts) |*p| p else &joined;
    for (sources) |a| if (a.ctx == null) return null;
    for ([_]Arr{ in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias }) |a| if (a.ctx == null) return null;
    const shape = mlx.getShape(sources[0]);
    const width = in.heads * 128;
    if (shape.len != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != (if (split) width else 3 * width) or
        @as(i64, shape[1]) * 3 * width > std.math.maxInt(c_int)) return null;
    for (sources) |a| if (mlx.mlx_array_dtype(a) != .bfloat16 or !std.mem.eql(c_int, shape, mlx.getShape(a))) return null;
    const rows = shape[1];
    if (!geometryFits(rows, in.heads)) return null;
    for ([_]Arr{ in.a, in.beta, in.conv_weight }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (!std.mem.eql(c_int, &.{ 1, rows, width }, mlx.getShape(in.a)) or !std.mem.eql(c_int, &.{ 1, rows, in.heads }, mlx.getShape(in.beta)) or
        !std.mem.eql(c_int, &.{ 3 * width, 4, 1 }, mlx.getShape(in.conv_weight)) or
        !std.mem.eql(c_int, &.{in.heads}, mlx.getShape(in.exp_a)) or mlx.mlx_array_dtype(in.exp_a) != .float32 or
        !std.mem.eql(c_int, &.{width}, mlx.getShape(in.dt_bias)) or mlx.mlx_array_dtype(in.dt_bias) != .float32) return null;
    if (in.conv_state) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16 or !std.mem.eql(c_int, &.{ 1, 3, 3 * width }, mlx.getShape(a))) return null;
    const mode = (try unary.unaryModes(s)) orelse return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for ([_]mlx.mlx_dtype{ .bfloat16, .bfloat16, .bfloat16, .float32 }) |dtype| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads, 128 }, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, rows, in.heads }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, 3 * width }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * in.heads, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", in.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HAS_CONV", @intFromBool(in.conv_state != null)));
    inline for (.{ "SIG_B", "SIG_F", "EXP_F", "RSQ_F" }, .{ mode.sig_b, mode.sig_f, mode.exp_f, mode.rsq_f }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, @intFromBool(value)));
    const length = mlx.mlx_array_new_int(rows);
    defer _ = mlx.mlx_array_free(length);
    // constants[3] is added to a copied tail row, and -0.0 keeps the sign of a zero as `materializedOwnedCopy` does.
    const constants = mlx.mlx_array_new_data(&[_]f32{ 1 / @sqrt(@as(f32, 128)), 1e-6, in.lower, -0.0 }, &[_]c_int{4}, 1, .float32);
    defer _ = mlx.mlx_array_free(constants);
    const rest = [_]Arr{ in.a, in.beta, in.conv_weight, in.exp_a, in.dt_bias, in.conv_state orelse sources[0], length, constants };
    var args: [3 + rest.len]Arr = undefined;
    @memcpy(args[0..sources.len], sources);
    @memcpy(args[sources.len..][0..rest.len], &rest);
    const iv = mlx.mlx_vector_array_new_data(&args, sources.len + rest.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(split), iv, cfg, s));
    var arrays: [6]Arr = undefined;
    var made: usize = 0;
    errdefer for (arrays[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&arrays, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    }
    count += 1;
    parts_count += @intFromBool(split);
    return .{ .q = arrays[0], .k = arrays[1], .v = arrays[2], .decay = arrays[3], .beta = arrays[4], .conv = arrays[5] };
}

fn reference(ops: *@import("glm5_model.zig").Ops, in: Inputs) ![6]Arr {
    const rows = mlx.getShape(in.qkv)[1];
    const width = in.heads * 128;
    const previous = in.conv_state orelse try ops.zeros(&.{ 1, 3, 3 * width }, .bfloat16);
    const joined = try ops.concat(&.{ previous, in.qkv }, 1);
    const conv = try ops.silu(try ops.conv(joined, in.conv_weight, 3 * width));
    const shape = [_]c_int{ 1, rows, in.heads, 128 };
    const q = try ops.cast(try ops.reshape(try ops.slice(conv, 2, 0, width), &shape), .float32);
    const k = try ops.cast(try ops.reshape(try ops.slice(conv, 2, width, 2 * width), &shape), .float32);
    const v = try ops.reshape(try ops.slice(conv, 2, 2 * width, 3 * width), &shape);
    const eps = try ops.scalar(1e-6, .float32);
    const qnorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, q, q), -1, false, true), eps));
    const knorm = try ops.unary(.rsqrt, try ops.binary(.add, try ops.reduce(try ops.binary(.mul, k, k), -1, false, true), eps));
    const qr = try ops.cast(try ops.binary(.mul, try ops.binary(.mul, q, qnorm), try ops.scalar(1 / @sqrt(@as(f32, 128)), .float32)), .bfloat16);
    const kr = try ops.cast(try ops.binary(.mul, k, knorm), .bfloat16);
    const a = try ops.reshape(try ops.cast(in.a, .float32), &shape);
    const shift = try ops.reshape(in.dt_bias, &.{ 1, 1, in.heads, 128 });
    const magnitude = try ops.reshape(in.exp_a, &.{ 1, 1, in.heads, 1 });
    const forget = try ops.unary(.sigmoid, try ops.binary(.mul, magnitude, try ops.binary(.add, a, shift)));
    const decay = try ops.unary(.exp, try ops.binary(.mul, forget, try ops.scalar(in.lower, .float32)));
    const beta = try ops.unary(.sigmoid, in.beta);
    const tail = try ops.slice(joined, 1, rows, rows + 3);
    const compact = if (rows > 1) try ops.own(try @import("transformer.zig").materializedOwnedCopy(ops.s, tail)) else try ops.contiguous(tail);
    return .{ qr, kr, v, decay, beta, compact };
}
fn randomArray(ops: *@import("glm5_model.zig").Ops, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64, scale: f32) !Arr {
    var size: usize = 1;
    for (shape) |d| size *= @intCast(d);
    var random = std.Random.DefaultPrng.init(seed);
    const rng = random.random();
    const allocator = std.testing.allocator;
    if (dtype == .bfloat16) {
        const data = try allocator.alloc(u16, size);
        defer allocator.free(data);
        for (data, 0..) |*v, i| {
            const f = (rng.float(f32) - 0.5) * scale;
            v.* = if (i % 257 == 0) 1 else if (i % 17 == 0) 0x8000 else @truncate(@as(u32, @bitCast(f)) >> 16);
        }
        return ops.own(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dtype));
    }
    const data = try allocator.alloc(f32, size);
    defer allocator.free(data);
    for (data) |*v| v.* = (rng.float(f32) - 0.5) * scale;
    return ops.own(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dtype));
}
fn expectBits(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    var cb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cb);
    try mlx.check(mlx.mlx_contiguous(&ca, a, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_contiguous(&cb, b, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(ca));
    try mlx.check(mlx.mlx_array_eval(cb));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .float32)
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(ca).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(cb).?[0..n]))
    else
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ca).?[0..n], mlx.mlx_array_data_bfloat16(cb).?[0..n]);
}

test "GLM KDA prework exact raw outputs and compact tails across token boundaries" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    for ([_]c_int{ 1, 3, 64 }) |heads| {
        for ([_]c_int{ 1, 2, 3, 4, 17, 128, 512 }) |rows| {
            for ([_]bool{ false, true }) |hot| {
                var ops = Ops{ .s = s };
                defer ops.deinit();
                const width = heads * 128;
                const exp_a = try ops.unary(.exp, try randomArray(&ops, &.{heads}, .float32, 55, 1));
                const in = Inputs{
                    .qkv = try randomArray(&ops, &.{ 1, rows, 3 * width }, .bfloat16, 3, 3),
                    .a = try randomArray(&ops, &.{ 1, rows, width }, .bfloat16, 15, 12),
                    .beta = try randomArray(&ops, &.{ 1, rows, heads }, .bfloat16, 33, 8),
                    .conv_weight = try randomArray(&ops, &.{ 3 * width, 4, 1 }, .bfloat16, 45, 0.5),
                    .exp_a = exp_a,
                    .dt_bias = try randomArray(&ops, &.{width}, .float32, 67, 2),
                    .conv_state = if (hot) try randomArray(&ops, &.{ 1, 3, 3 * width }, .bfloat16, 89, 2) else null,
                    .heads = heads,
                };
                const want = try reference(&ops, in);
                const got = (try apply(s, in)) orelse return error.TestExpectedPrework;
                defer got.deinit();
                for (want, got.arrays(), 0..) |expected, actual, field| expectBits(expected, actual) catch |err| {
                    std.debug.print("prework mismatch H={d} rows={d} hot={} field={d}\n", .{ heads, rows, hot, field });
                    return err;
                };
            }
        }
    }
}

test "GLM KDA prework reads the q/k/v projections in place with the joined kernel's bits" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    for ([_]c_int{ 1, 3, 64 }) |heads| {
        for ([_]c_int{ 128, 133, 512 }) |rows| {
            for ([_]bool{ false, true }) |hot| {
                var ops = Ops{ .s = s };
                defer ops.deinit();
                const width = heads * 128;
                var parts: [3]Arr = undefined;
                for (&parts, 0..) |*part, i| part.* = try randomArray(&ops, &.{ 1, rows, width }, .bfloat16, 3 + i, 3);
                var in = Inputs{
                    .qkv = try ops.concat(&parts, -1),
                    .a = try randomArray(&ops, &.{ 1, rows, width }, .bfloat16, 15, 12),
                    .beta = try randomArray(&ops, &.{ 1, rows, heads }, .bfloat16, 33, 8),
                    .conv_weight = try randomArray(&ops, &.{ 3 * width, 4, 1 }, .bfloat16, 45, 0.5),
                    .exp_a = try ops.unary(.exp, try randomArray(&ops, &.{heads}, .float32, 55, 1)),
                    .dt_bias = try randomArray(&ops, &.{width}, .float32, 67, 2),
                    .conv_state = if (hot) try randomArray(&ops, &.{ 1, 3, 3 * width }, .bfloat16, 89, 2) else null,
                    .heads = heads,
                };
                const want = (try apply(s, in)) orelse return error.TestExpectedPrework;
                defer want.deinit();
                const before = parts_count;
                in.qkv = .{ .ctx = null };
                in.parts = parts;
                const got = (try apply(s, in)) orelse return error.TestExpectedPrework;
                defer got.deinit();
                try std.testing.expectEqual(before + 1, parts_count);
                for (want.arrays(), got.arrays(), 0..) |expected, actual, field| expectBits(expected, actual) catch |err| {
                    std.debug.print("parts mismatch H={d} rows={d} hot={} field={d}\n", .{ heads, rows, hot, field });
                    return err;
                };
            }
        }
    }
}

test "GLM KDA prework refuses malformed projection parts before dispatch" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const part = try ops.zeros(&.{ 1, 2, 128 }, .bfloat16);
    const valid = Inputs{
        .qkv = .{ .ctx = null },
        .parts = .{ part, part, part },
        .a = try ops.zeros(&.{ 1, 2, 128 }, .bfloat16),
        .beta = try ops.zeros(&.{ 1, 2, 1 }, .bfloat16),
        .conv_weight = try ops.zeros(&.{ 384, 4, 1 }, .bfloat16),
        .exp_a = try ops.ones(&.{1}, .float32),
        .dt_bias = try ops.zeros(&.{128}, .float32),
        .heads = 1,
    };
    const before = parts_count;
    var bad = valid;
    bad.parts = .{ part, part, try ops.zeros(&.{ 1, 2, 256 }, .bfloat16) };
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad.parts = .{ part, part, try ops.zeros(&.{ 1, 2, 128 }, .float32) };
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad.parts = .{ part, .{ .ctx = null }, part };
    try std.testing.expect((try apply(ops.s, bad)) == null);
    try std.testing.expectEqual(before, parts_count);
}

test "GLM KDA prework refuses unsupported inputs before dispatch" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const count_before = dispatchCount();
    const valid = Inputs{
        .qkv = try ops.zeros(&.{ 1, 2, 384 }, .bfloat16),
        .a = try ops.zeros(&.{ 1, 2, 128 }, .bfloat16),
        .beta = try ops.zeros(&.{ 1, 2, 1 }, .bfloat16),
        .conv_weight = try ops.zeros(&.{ 384, 4, 1 }, .bfloat16),
        .exp_a = try ops.ones(&.{1}, .float32),
        .dt_bias = try ops.zeros(&.{128}, .float32),
        .heads = 1,
    };
    var bad = valid;
    bad.lower = -6;
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.heads = 0;
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.qkv = .{ .ctx = null };
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.a = try ops.cast(valid.a, .float32);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.conv_state = try ops.zeros(&.{ 1, 2, 384 }, .bfloat16);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    bad = valid;
    bad.exp_a = try ops.cast(valid.exp_a, .bfloat16);
    try std.testing.expect((try apply(ops.s, bad)) == null);
    try std.testing.expectEqual(count_before, dispatchCount());
}

test "GLM KDA prework geometry guards every uint32 shader offset" {
    try std.testing.expect(geometryFits(512, 64));
    const largest_heads: c_int = @intCast((@as(u64, std.math.maxInt(u32)) + 1) / (384 * 4));
    try std.testing.expect(geometryFits(1, largest_heads));
    try std.testing.expect(geometryFits(2, largest_heads));
    try std.testing.expect(!geometryFits(1, largest_heads + 1));
    try std.testing.expect(!geometryFits(1, 3_000_000));
    const longest: c_int = @divTrunc(std.math.maxInt(c_int), 64 * 384);
    try std.testing.expect(geometryFits(longest, 64));
    try std.testing.expect(!geometryFits(longest + 1, 64));
    try std.testing.expect(!geometryFits(0, 64));
    try std.testing.expect(!geometryFits(1, 0));
    try std.testing.expect(!geometryFits(std.math.maxInt(c_int), std.math.maxInt(c_int)));
}

test "GLM KDA tree prework follows raw-bit serial ancestor windows" {
    const Ops = @import("glm5_model.zig").Ops;
    const s = mlx.gpuStream();
    for ([_]c_int{ 1, 3, 64 }) |heads| {
        for ([_]usize{ 1, 3, 16 }) |rows| {
            for (0..3) |topology| {
                for ([_]bool{ false, true }) |hot| {
                    var ops = Ops{ .s = s };
                    defer ops.deinit();
                    const width = heads * 128;
                    const r: c_int = @intCast(rows);
                    const in = Inputs{
                        .qkv = try randomArray(&ops, &.{ 1, r, 3 * width }, .bfloat16, 353, 3),
                        .a = try randomArray(&ops, &.{ 1, r, width }, .bfloat16, 375, 12),
                        .beta = try randomArray(&ops, &.{ 1, r, heads }, .bfloat16, 393, 8),
                        .conv_weight = try randomArray(&ops, &.{ 3 * width, 4, 1 }, .bfloat16, 445, 0.5),
                        .exp_a = try ops.unary(.exp, try randomArray(&ops, &.{heads}, .float32, 355, 1)),
                        .dt_bias = try randomArray(&ops, &.{width}, .float32, 467, 2),
                        .conv_state = if (hot) try randomArray(&ops, &.{ 1, 3, 3 * width }, .bfloat16, 489, 2) else null,
                        .heads = heads,
                    };
                    var parents: [16]i32 = @splat(-1);
                    for (1..rows) |i| parents[i] = @intCast(switch (topology) {
                        0 => i - 1,
                        1 => (i - 1) / 2,
                        else => 0,
                    });
                    const got = (try applyTree(s, in, parents[0..rows])) orelse return error.TestExpectedTreePrework;
                    defer got.deinit();
                    var tails: [16]Arr = undefined;
                    for (0..rows) |i| {
                        var one_ops = Ops{ .s = s };
                        defer one_ops.deinit();
                        var one = in;
                        const row: c_int = @intCast(i);
                        one.qkv = try one_ops.slice(in.qkv, 1, row, row + 1);
                        one.a = try one_ops.slice(in.a, 1, row, row + 1);
                        one.beta = try one_ops.slice(in.beta, 1, row, row + 1);
                        one.conv_state = if (parents[i] < 0) in.conv_state else tails[@intCast(parents[i])];
                        const want = try reference(&one_ops, one);
                        tails[i] = try ops.own(try one_ops.result(want[5]));
                        for (got.arrays(), want[0..5], 0..) |actual, expected, field| expectBits(expected, try one_ops.slice(actual, 1, row, row + 1)) catch |err| {
                            std.debug.print("tree prework mismatch H={d} rows={d} topology={d} hot={} node={d} field={d}\n", .{ heads, rows, topology, hot, i, field });
                            return err;
                        };
                    }
                }
            }
        }
    }
}

test "GLM KDA tree prework rejects malformed parents and checked geometry" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const in = Inputs{
        .qkv = try ops.zeros(&.{ 1, 3, 384 }, .bfloat16),
        .a = try ops.zeros(&.{ 1, 3, 128 }, .bfloat16),
        .beta = try ops.zeros(&.{ 1, 3, 1 }, .bfloat16),
        .conv_weight = try ops.zeros(&.{ 384, 4, 1 }, .bfloat16),
        .exp_a = try ops.ones(&.{1}, .float32),
        .dt_bias = try ops.zeros(&.{128}, .float32),
        .heads = 1,
    };
    const before = treeDispatchCount();
    for ([_][]const i32{ &.{}, &.{0}, &.{ -1, 1, 0 }, &.{ -1, -1, 0 }, &.{ -1, 2, 0 }, &.{-1} }) |parents|
        try std.testing.expectError(error.InvalidGlmDraftTree, applyTree(ops.s, in, parents));
    var bad = in;
    bad.heads = std.math.maxInt(c_int);
    try std.testing.expect((try applyTree(ops.s, bad, &.{ -1, 0, 1 })) == null);
    bad = in;
    bad.lower = -4;
    try std.testing.expect((try applyTree(ops.s, bad, &.{ -1, 0, 1 })) == null);
    try std.testing.expectEqual(before, treeDispatchCount());
}
