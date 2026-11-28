//! Hy4's small f32 elementwise chains, each one launch that repeats the ops' arithmetic in their order with no fma
//! contraction, so every element is the chain's bytes: the engine's `fusedChain` takes BF16 chains only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// MLX's `Sigmoid` functor (`unary_ops.h`, the engine's `sushi_sigmoid`) on a float.
pub const header: [:0]const u8 =
    \\inline float hy4_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
;

/// `(g · σ(g)) · u`: the SwiGLU's `silu` (`Ops.silu`, the Sigmoid then the Multiply) times the up rows.
pub const swiglu_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\float x = static_cast<float>(g[i]);
    \\float a = x * hy4_sigmoid(x);
    \\out[i] = static_cast<OT>(a * static_cast<float>(u[i]));
;

var swiglu_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Which launch a config is for, and its shape.
const Kind = enum { swiglu, swiglu_finish_prepare, groups, rows, gated, gated_into, gated_prepared_into, sink_sm_causal, sink_sm_mask, ring_keys };
const Key = struct { kind: Kind, a: c_int, b: c_int, c: c_int = 0, d: c_int = 0, e: c_int = 0 };
var cfgs: kcfg.Cache(Key) = .{};

fn sinkSoftmaxGroup(n: c_int) c_int {
    const axis = n + 1;
    return 32 * @divTrunc(@divTrunc(axis + 3, 4) + 31, 32);
}

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    // (outputs, grid x and y, threads a group)
    switch (k.kind) {
        .swiglu => {
            const out_dt: mlx.mlx_dtype = @enumFromInt(k.d);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.a * k.b, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.a * k.b, 256), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", out_dt));
        },
        .swiglu_finish_prepare => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, .float16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.b, 128) * 32, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "DIM", k.b));
        },
        .groups => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b + 1 }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b + 1, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b + 1, 256), 1, 1));
        },
        .rows => {
            const width = k.b * k.c;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, width }, 2, .uint32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, width }, 2, .bool_));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(width, 256), 1, 1));
        },
        .gated => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b, 256), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", k.c));
        },
        .gated_into => {
            const out_dt: mlx.mlx_dtype = @enumFromInt(k.d);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.e, k.b }, 2, out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b, 256), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", k.c));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", out_dt));
        },
        .gated_prepared_into => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.e, k.b }, 2, .float16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.b, 128) * 32, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", k.c));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HV", k.b));
        },
        .sink_sm_causal, .sink_sm_mask => {
            const heads_first = (k.d & 1) != 0;
            const out_shape = if (heads_first) [_]c_int{ k.a, k.b, k.c } else [_]c_int{ k.b, k.a, k.c };
            const group = sinkSoftmaxGroup(k.c);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.a * k.b * group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "HEADS_FIRST", heads_first));
            if (k.kind == .sink_sm_mask) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "MASK_BOOL", (k.d & 2) != 0));
            }
        },
        .ring_keys => {
            const width = k.c + k.d;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b, k.c }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b, k.d }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, k.b, k.a));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(width, 256), 1, 1));
        },
    }
    return cfg;
}

/// Whether `swiglu` takes `g` and `u`: f32 of one shape on a GPU stream.
pub fn swigluApplies(s: mlx.mlx_stream, g: Arr, u: Arr) bool {
    const gd = mlx.mlx_array_dtype(g);
    const ud = mlx.mlx_array_dtype(u);
    return mlx.streamIsGpu(s) and (gd == .float32 or gd == .bfloat16) and (ud == .float32 or ud == .bfloat16) and
        std.mem.eql(c_int, mlx.getShape(g), mlx.getShape(u)) and mlx.mlx_array_size(g) > 0;
}

/// `silu(g) · u` in f32, `g` and `u` of one shape. The caller frees the result.
pub fn swiglu(s: mlx.mlx_stream, g: Arr, u: Arr) !Arr {
    return swigluAs(s, g, u, .float32);
}

/// `swiglu`'s launch with its output in `out_dt` (BF16 or f32): the gate and up rows as stored, no casts around it.
pub fn swigluAs(s: mlx.mlx_stream, g: Arr, u: Arr, out_dt: mlx.mlx_dtype) !Arr {
    const k = try getKernel(&swiglu_kernel, "sushi_hy4_swiglu", &.{ "g", "u" }, swiglu_source);
    const sh = mlx.getShape(g);
    const rows: c_int = if (sh.len == 2) sh[0] else 1;
    const cfg = try cfgs.get(.{
        .kind = .swiglu,
        .a = rows,
        .b = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(g))), rows),
        .c = @intFromEnum(mlx.mlx_array_dtype(g)),
        .d = @intFromEnum(out_dt),
        .e = @intFromEnum(mlx.mlx_array_dtype(u)),
    }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ g, u }, cfg, &out);
    if (sh.len == 2) return out[0];
    defer _ = mlx.mlx_array_free(out[0]);
    var shaped = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, out[0], sh.ptr, sh.len, s));
    return shaped;
}

pub const swiglu_finish_prepare_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(DIM) + base;
    \\const float sc = 0.08838834764831845f;
    \\float4 vg = float4(
    \\  float(g_inner[xb + lane]),
    \\  float(g_inner[xb + lane + 32u]),
    \\  float(g_inner[xb + lane + 64u]),
    \\  float(g_inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(vg.x, bit);
    \\  const float p1 = simd_shuffle_xor(vg.y, bit);
    \\  const float p2 = simd_shuffle_xor(vg.z, bit);
    \\  const float p3 = simd_shuffle_xor(vg.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  vg.x = lower ? vg.x + p0 : p0 - vg.x;
    \\  vg.y = lower ? vg.y + p1 : p1 - vg.y;
    \\  vg.z = lower ? vg.z + p2 : p2 - vg.z;
    \\  vg.w = lower ? vg.w + p3 : p3 - vg.w;
    \\}
    \\const float gs0 = vg.x + vg.y;
    \\const float gs1 = vg.x - vg.y;
    \\const float gs2 = vg.z + vg.w;
    \\const float gs3 = vg.z - vg.w;
    \\const float g0 = float(bfloat16_t((gs0 + gs2) * sc * float(g_svh[base + lane])));
    \\const float g1 = float(bfloat16_t((gs1 + gs3) * sc * float(g_svh[base + lane + 32u])));
    \\const float g2 = float(bfloat16_t((gs0 - gs2) * sc * float(g_svh[base + lane + 64u])));
    \\const float g3 = float(bfloat16_t((gs1 - gs3) * sc * float(g_svh[base + lane + 96u])));
    \\float4 vu = float4(
    \\  float(u_inner[xb + lane]),
    \\  float(u_inner[xb + lane + 32u]),
    \\  float(u_inner[xb + lane + 64u]),
    \\  float(u_inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(vu.x, bit);
    \\  const float p1 = simd_shuffle_xor(vu.y, bit);
    \\  const float p2 = simd_shuffle_xor(vu.z, bit);
    \\  const float p3 = simd_shuffle_xor(vu.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  vu.x = lower ? vu.x + p0 : p0 - vu.x;
    \\  vu.y = lower ? vu.y + p1 : p1 - vu.y;
    \\  vu.z = lower ? vu.z + p2 : p2 - vu.z;
    \\  vu.w = lower ? vu.w + p3 : p3 - vu.w;
    \\}
    \\const float us0 = vu.x + vu.y;
    \\const float us1 = vu.x - vu.y;
    \\const float us2 = vu.z + vu.w;
    \\const float us3 = vu.z - vu.w;
    \\const float u0 = float(bfloat16_t((us0 + us2) * sc * float(u_svh[base + lane])));
    \\const float u1 = float(bfloat16_t((us1 + us3) * sc * float(u_svh[base + lane + 32u])));
    \\const float u2 = float(bfloat16_t((us0 - us2) * sc * float(u_svh[base + lane + 64u])));
    \\const float u3 = float(bfloat16_t((us1 - us3) * sc * float(u_svh[base + lane + 96u])));
    \\const float a0 = float(bfloat16_t((g0 * hy4_sigmoid(g0)) * u0));
    \\const float a1 = float(bfloat16_t((g1 * hy4_sigmoid(g1)) * u1));
    \\const float a2 = float(bfloat16_t((g2 * hy4_sigmoid(g2)) * u2));
    \\const float a3 = float(bfloat16_t((g3 * hy4_sigmoid(g3)) * u3));
    \\float4 v = float4(
    \\  a0 * float(d_suh[base + lane]),
    \\  a1 * float(d_suh[base + lane + 32u]),
    \\  a2 * float(d_suh[base + lane + 64u]),
    \\  a3 * float(d_suh[base + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\out[xb + lane] = half((s0 + s2) * sc);
    \\out[xb + lane + 32u] = half((s1 + s3) * sc);
    \\out[xb + lane + 64u] = half((s0 - s2) * sc);
    \\out[xb + lane + 96u] = half((s1 - s3) * sc);
;

var swiglu_finish_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `silu(gate) · up` from the two banks' unfinished GEMM rows (`hy4_stq.applyInner`), finished with their `svh`, then
/// prepared for the down bank (`suh`, Hadamard, f16): the finishes, the SwiGLU and the prepare in one launch, their
/// bytes.
pub fn swigluFinishPrepare(s: mlx.mlx_stream, g_inner: Arr, g_svh: Arr, u_inner: Arr, u_svh: Arr, d_suh: Arr) !Arr {
    const k = try getKernel(&swiglu_finish_prepare_kernel, "sushi_hy4_swiglu_finish_prepare", &.{ "g_inner", "g_svh", "u_inner", "u_svh", "d_suh" }, swiglu_finish_prepare_source);
    const sh = mlx.getShape(g_inner);
    const cfg = try cfgs.get(.{ .kind = .swiglu_finish_prepare, .a = sh[0], .b = sh[1] }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ g_inner, g_svh, u_inner, u_svh, d_suh }, cfg, &out);
    return out[0];
}

/// `recallAt`'s groups [T, take + 1] f32: the picks `order` [T, take] (int32) widened, then each row's own group
/// `floor((p0 + t) / ratio)` (the arange, the Divide and the Floor).
pub const groups_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint c = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint k = threads_per_grid.x;
    \\out[t * k + c] = c + 1 < k ? static_cast<float>(order[t * (k - 1) + c]) : metal::floor(static_cast<float>(meta[0] + int(t)) / static_cast<float>(meta[1]));
;

/// `recalledRows` over the groups [T, k] f32 (own group last): each row's `k · ratio` key rows `pos = g · ratio + r`
/// clamped to `n - 1` (uint32), and whether it reads each (`pos` at or before its position, and from an earlier group
/// or the own group's column), in the ops' f32 order.
pub const rows_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint s = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint width = threads_per_grid.x;
    \\int ratio = meta[1];
    \\uint k = width / uint(ratio);
    \\uint j = s / uint(ratio);
    \\uint r = s % uint(ratio);
    \\float g = groups[t * k + j];
    \\float own_group = groups[t * k + k - 1];
    \\bool ok = (g < own_group) || (static_cast<float>(k - 1) <= static_cast<float>(j));
    \\float base = g * static_cast<float>(ratio);
    \\float pos = base + static_cast<float>(r);
    \\bool reads = (pos <= static_cast<float>(meta[0] + int(t))) && ok;
    \\float last = static_cast<float>(meta[2] - 1);
    \\float clamped = metal::isnan(pos) ? pos : (pos < last ? pos : last);
    \\rows[t * width + s] = static_cast<uint>(clamped);
    \\keep[t * width + s] = reads;
;

var groups_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn intsArr(v: []const i32) !Arr {
    return mlx.newData(v.ptr, &.{@intCast(v.len)}, .int32);
}

/// `recallAt`'s groups on a GPU stream from the top-k picks `order` [T, take] int32 (take ≥ 1). The caller frees.
pub fn recallGroups(s: mlx.mlx_stream, order: Arr, p0: c_int, ratio: c_int) !Arr {
    const k = try getKernel(&groups_kernel, "sushi_hy4_recall_groups", &.{ "order", "meta" }, groups_source);
    const sh = mlx.getShape(order);
    const mv = [_]i32{ p0, ratio };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .groups, .a = sh[0], .b = sh[1] }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ order, meta }, cfg, &out);
    return out[0];
}

/// `fwd.recalledRows` on a GPU stream over `groups` [T, k] f32: the key rows [T, k·ratio] uint32 and the keep mask
/// (bool). The caller frees both.
pub fn recalledRows(s: mlx.mlx_stream, groups: Arr, p0: c_int, n: c_int, ratio: c_int) !struct { rows: Arr, keep: Arr } {
    if (rows_kernel == null) {
        const ins = [_][*:0]const u8{ "groups", "meta" };
        const outs = [_][*:0]const u8{ "rows", "keep" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_recalled_rows", iv, ov, rows_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernel = made;
    }
    const sh = mlx.getShape(groups);
    const mv = [_]i32{ p0, ratio, n };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .rows, .a = sh[0], .b = sh[1], .c = ratio }, build);
    var out: [2]Arr = undefined;
    try applyOuts(s, rows_kernel.?, &.{ groups, meta }, cfg, &out);
    return .{ .rows = out[0], .keep = out[1] };
}

fn applyOuts(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, outs: []Arr) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, ov, i)) catch |e| {
            for (outs[0 .. i + 1]) |prev| _ = mlx.mlx_array_free(prev);
            return e;
        };
    }
}

/// The attention output gate: `u ⊙ σ(gate)` with `u` [H, T, v] (any strides) and the gate's rows [T, H·v] as stored,
/// widened, written [T, H·v] as `W_o` reads them (the Cast, the Sigmoid, the Multiply and the transpose's copy).
pub const gate_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint hv = threads_per_grid.x;
    \\uint h = i / uint(V);
    \\uint c = i % uint(V);
    \\float g = static_cast<float>(gate[(size_t)t * hv + i]);
    \\float uv = u[(size_t)h * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c * (size_t)u_strides[2]];
    \\out[(size_t)t * hv + i] = uv * hy4_sigmoid(g);
;

var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `gated` takes `u` [H, T, v] f32 and `gate` [T, H·v] (BF16 or f32) on a GPU stream.
pub fn gatedApplies(s: mlx.mlx_stream, u: Arr, gate: Arr) bool {
    const us = mlx.getShape(u);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    return mlx.streamIsGpu(s) and us.len == 3 and gs.len == 2 and mlx.mlx_array_dtype(u) == .float32 and
        (gd == .bfloat16 or gd == .float32) and gs[0] == us[1] and gs[1] == us[0] * us[2];
}

/// `u ⊙ σ(gate)` as [T, H·v] f32. The caller frees the result.
pub fn gated(s: mlx.mlx_stream, u: Arr, gate: Arr) !Arr {
    if (gate_kernel == null) {
        const ins = [_][*:0]const u8{ "u", "gate" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        // `u` is read through its strides: the value product's [H, T, v] as it stands.
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_gated", iv, ov, gate_source, header, false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        gate_kernel = made;
    }
    const gs = mlx.getShape(gate);
    const cfg = try cfgs.get(.{ .kind = .gated, .a = gs[0], .b = gs[1], .c = mlx.getShape(u)[2] }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, gate_kernel.?, &.{ u, gate }, cfg, &out);
    return out[0];
}

pub const gate_into_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint hv = threads_per_grid.x;
    \\uint h = i / uint(V);
    \\uint c = i % uint(V);
    \\float g = static_cast<float>(gate[(size_t)t * (size_t)gate_strides[0] + (size_t)i * (size_t)gate_strides[1]]);
    \\float uv = u[(size_t)h * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c * (size_t)u_strides[2]];
    \\out[(size_t)(uint(row0[0]) + t) * hv + i] = static_cast<OT>(uv * hy4_sigmoid(g));
;

var gate_into_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `gatedInto` takes `u` and `gate` into `dst`: an engine stream, `gated`'s shapes, `dst` as wide as the gate
/// in BF16 or f32.
pub fn gatedIntoApplies(s: mlx.mlx_stream, u: Arr, gate: Arr, dst: Arr) bool {
    if (!mlx.engine.isStream(s) or !gatedApplies(s, u, gate)) return false;
    const ds = mlx.getShape(dst);
    const dd = mlx.mlx_array_dtype(dst);
    return ds.len == 2 and ds[1] == mlx.getShape(gate)[1] and (dd == .bfloat16 or dd == .float32);
}

/// `gated`'s rows written into `dst` from row `r0`, in `dst`'s dtype: a span's gated rows land in the chunk's block
/// that `W_o` reads, with no join after the spans.
pub fn gatedInto(s: mlx.mlx_stream, u: Arr, gate: Arr, dst: Arr, r0: c_int) !void {
    if (gate_into_kernel == null) {
        const ins = [_][*:0]const u8{ "u", "gate", "row0" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_gated_into", iv, ov, gate_into_source, header, false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        gate_into_kernel = made;
    }
    const gs = mlx.getShape(gate);
    const ds = mlx.getShape(dst);
    const cfg = try cfgs.get(.{ .kind = .gated_into, .a = gs[0], .b = gs[1], .c = mlx.getShape(u)[2], .d = @intFromEnum(mlx.mlx_array_dtype(dst)), .e = ds[0] }, build);
    const rv = [_]i32{r0};
    const row0 = try intsArr(&rv);
    defer _ = mlx.mlx_array_free(row0);
    const inputs = [_]Arr{ u, gate, row0 };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.kernelApplyInto(&ov, gate_into_kernel.?, iv, cfg, s, dst) orelse return error.MlxError);
}

pub const gate_prepared_into_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint t = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\uint hv = uint(HV);
    \\uint base = block * 128u;
    \\float4 v;
    \\{
    \\  uint i0 = base + lane;
    \\  uint h0 = i0 / uint(V); uint c0 = i0 % uint(V);
    \\  float g0 = static_cast<float>(gate[(size_t)t * (size_t)gate_strides[0] + (size_t)i0 * (size_t)gate_strides[1]]);
    \\  float u0 = u[(size_t)h0 * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c0 * (size_t)u_strides[2]];
    \\  v.x = float(bfloat16_t(u0 * hy4_sigmoid(g0))) * float(suh[i0]);
    \\  uint i1 = i0 + 32u;
    \\  uint h1 = i1 / uint(V); uint c1 = i1 % uint(V);
    \\  float g1 = static_cast<float>(gate[(size_t)t * (size_t)gate_strides[0] + (size_t)i1 * (size_t)gate_strides[1]]);
    \\  float u1 = u[(size_t)h1 * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c1 * (size_t)u_strides[2]];
    \\  v.y = float(bfloat16_t(u1 * hy4_sigmoid(g1))) * float(suh[i1]);
    \\  uint i2 = i0 + 64u;
    \\  uint h2 = i2 / uint(V); uint c2 = i2 % uint(V);
    \\  float g2 = static_cast<float>(gate[(size_t)t * (size_t)gate_strides[0] + (size_t)i2 * (size_t)gate_strides[1]]);
    \\  float u2 = u[(size_t)h2 * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c2 * (size_t)u_strides[2]];
    \\  v.z = float(bfloat16_t(u2 * hy4_sigmoid(g2))) * float(suh[i2]);
    \\  uint i3 = i0 + 96u;
    \\  uint h3 = i3 / uint(V); uint c3 = i3 % uint(V);
    \\  float g3 = static_cast<float>(gate[(size_t)t * (size_t)gate_strides[0] + (size_t)i3 * (size_t)gate_strides[1]]);
    \\  float u3 = u[(size_t)h3 * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c3 * (size_t)u_strides[2]];
    \\  v.w = float(bfloat16_t(u3 * hy4_sigmoid(g3))) * float(suh[i3]);
    \\}
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\const size_t yb = (size_t)(uint(row0[0]) + t) * (size_t)hv + base;
    \\out[yb + lane] = half((s0 + s2) * sc);
    \\out[yb + lane + 32u] = half((s1 + s3) * sc);
    \\out[yb + lane + 64u] = half((s0 - s2) * sc);
    \\out[yb + lane + 96u] = half((s1 - s3) * sc);
;

var gate_prepared_into_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `gatedPreparedInto` serves: `gatedIntoApplies`' shapes, an f16 `dst` and `suh` as wide as the gate, in
/// whole 128-column blocks.
pub fn gatedPreparedIntoApplies(s: mlx.mlx_stream, u: Arr, gate: Arr, suh: Arr, dst: Arr) bool {
    if (!mlx.engine.isStream(s) or !gatedApplies(s, u, gate)) return false;
    const gs = mlx.getShape(gate);
    const ds = mlx.getShape(dst);
    return ds.len == 2 and ds[1] == gs[1] and @rem(gs[1], 128) == 0 and mlx.mlx_array_dtype(dst) == .float16 and
        mlx.mlx_array_dtype(suh) == .float16 and mlx.mlx_array_size(suh) == @as(usize, @intCast(gs[1]));
}

/// `gatedInto` with `W_o`'s input preparation in the launch: the span's gated rows written to `dst` from row `r0` as
/// `exl3.kernels.prepareIndexed` makes them from their BF16 rounding (scaled by `suh`, Hadamard-rotated, f16).
pub fn gatedPreparedInto(s: mlx.mlx_stream, u: Arr, gate: Arr, suh: Arr, dst: Arr, r0: c_int) !void {
    if (gate_prepared_into_kernel == null) {
        const ins = [_][*:0]const u8{ "u", "gate", "suh", "row0" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_gated_prepared_into", iv, ov, gate_prepared_into_source, header, false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        gate_prepared_into_kernel = made;
    }
    const gs = mlx.getShape(gate);
    const ds = mlx.getShape(dst);
    const cfg = try cfgs.get(.{ .kind = .gated_prepared_into, .a = gs[0], .b = gs[1], .c = mlx.getShape(u)[2], .e = ds[0] }, build);
    const rv = [_]i32{r0};
    const row0 = try intsArr(&rv);
    defer _ = mlx.mlx_array_free(row0);
    const inputs = [_]Arr{ u, gate, suh, row0 };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.kernelApplyInto(&ov, gate_prepared_into_kernel.?, iv, cfg, s, dst) orelse return error.MlxError);
}

const sink_softmax_tail: [:0]const u8 =
    \\threadgroup float local_max[32];
    \\threadgroup float local_normalizer[32];
    \\if (simd_group_id == 0) {
    \\  local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
    \\  local_normalizer[simd_lane_id] = 0.0f;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float maxval = -metal::numeric_limits<float>::max();
    \\for (int i = 0; i < 4; i++) {
    \\  maxval = (maxval < ld[i]) ? ld[i] : maxval;
    \\}
    \\maxval = simd_max(maxval);
    \\if (simd_lane_id == 0) {
    \\  local_max[simd_group_id] = maxval;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  maxval = simd_max(local_max[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_max[0] = maxval;
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\maxval = local_max[0];
    \\float normalizer = 0.0f;
    \\for (int i = 0; i < 4; i++) {
    \\  float exp_x = fast::exp(ld[i] - maxval);
    \\  ld[i] = exp_x;
    \\  normalizer += exp_x;
    \\}
    \\normalizer = simd_sum(normalizer);
    \\if (simd_lane_id == 0) {
    \\  local_normalizer[simd_group_id] = normalizer;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  normalizer = simd_sum(local_normalizer[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_normalizer[0] = normalizer;
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\normalizer = 1.0f / local_normalizer[0];
    \\device float* dst = out + (size_t)gid * (size_t)N + (size_t)lid * 4;
    \\if (lid * 4 + 4 <= N) {
    \\  for (int i = 0; i < 4; i++) {
    \\    dst[i] = ld[i] * normalizer;
    \\  }
    \\} else {
    \\  for (int i = 0; i < 4; i++) {
    \\    if (lid * 4 + i < N) {
    \\      dst[i] = ld[i] * normalizer;
    \\    }
    \\  }
    \\}
;

pub const sink_softmax_causal_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint gid = threadgroup_position_in_grid.x;
    \\int lid = int(thread_index_in_threadgroup);
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\int H = meta[0];
    \\int T = meta[1];
    \\int N = meta[2];
    \\int p0 = meta[3];
    \\int window = meta[4];
    \\float scale = as_type<float>(meta[5]);
    \\int h = HEADS_FIRST ? (int(gid) / T) : (int(gid) % H);
    \\int t = HEADS_FIRST ? (int(gid) % T) : (int(gid) / H);
    \\float rf = static_cast<float>(p0 + t);
    \\float wf = static_cast<float>(window);
    \\float ld[4];
    \\for (int i = 0; i < 4; i++) {
    \\  int col = lid * 4 + i;
    \\  if (col < N) {
    \\    size_t idx = (size_t)gid * (size_t)N + (size_t)col;
    \\    float s = a[idx] + b[idx];
    \\    float sc = s * scale;
    \\    float cf = static_cast<float>(col);
    \\    bool keep = (cf <= rf) && (window <= 0 || (cf > (rf - wf)));
    \\    float m = keep ? 0.0f : -metal::numeric_limits<float>::infinity();
    \\    ld[i] = sc + m;
    \\  } else if (col == N) {
    \\    ld[i] = static_cast<float>(sinks[h]);
    \\  } else {
    \\    ld[i] = -metal::numeric_limits<float>::infinity();
    \\  }
    \\}
++ sink_softmax_tail;

pub const sink_softmax_mask_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint gid = threadgroup_position_in_grid.x;
    \\int lid = int(thread_index_in_threadgroup);
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\int H = meta[0];
    \\int T = meta[1];
    \\int N = meta[2];
    \\float scale = as_type<float>(meta[3]);
    \\int h = HEADS_FIRST ? (int(gid) / T) : (int(gid) % H);
    \\int t = HEADS_FIRST ? (int(gid) % T) : (int(gid) / H);
    \\float ld[4];
    \\for (int i = 0; i < 4; i++) {
    \\  int col = lid * 4 + i;
    \\  if (col < N) {
    \\    size_t idx = (size_t)gid * (size_t)N + (size_t)col;
    \\    float s = a[idx] + b[idx];
    \\    float sc = s * scale;
    \\    size_t midx = (size_t)t * (size_t)N + (size_t)col;
    \\    float m = MASK_BOOL ? (mask[midx] ? 0.0f : -metal::numeric_limits<float>::infinity()) : static_cast<float>(mask[midx]);
    \\    ld[i] = sc + m;
    \\  } else if (col == N) {
    \\    ld[i] = static_cast<float>(sinks[h]);
    \\  } else {
    \\    ld[i] = -metal::numeric_limits<float>::infinity();
    \\  }
    \\}
++ sink_softmax_tail;

var sink_sm_causal_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sink_sm_mask_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn sinkSoftmaxCausalApplies(s: mlx.mlx_stream, a: Arr, b: Arr, sinks: Arr, heads_first: bool) bool {
    const sa = mlx.getShape(a);
    const sb = mlx.getShape(b);
    const ss = mlx.getShape(sinks);
    const sdt = mlx.mlx_array_dtype(sinks);
    if (!mlx.streamIsGpu(s) or sa.len != 3 or !std.mem.eql(c_int, sa, sb) or ss.len != 1) return false;
    if (mlx.mlx_array_dtype(a) != .float32 or mlx.mlx_array_dtype(b) != .float32) return false;
    if (sdt != .float32 and sdt != .bfloat16) return false;
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    if (h <= 0 or t <= 0 or n <= 0 or n >= 4096 or ss[0] != h) return false;
    return @as(i64, h) * @as(i64, t) * @as(i64, sinkSoftmaxGroup(n)) <= std.math.maxInt(c_int);
}

pub fn sinkSoftmaxMaskApplies(s: mlx.mlx_stream, a: Arr, b: Arr, mask: Arr, sinks: Arr, heads_first: bool) bool {
    if (!sinkSoftmaxCausalApplies(s, a, b, sinks, heads_first)) return false;
    const sa = mlx.getShape(a);
    const sm = mlx.getShape(mask);
    const mdt = mlx.mlx_array_dtype(mask);
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    return sm.len == 2 and sm[0] == t and sm[1] == n and (mdt == .bool_ or mdt == .float32);
}

pub fn sinkSoftmaxCausal(s: mlx.mlx_stream, a: Arr, b: Arr, sinks: Arr, scale: f32, p0: c_int, window: c_int, heads_first: bool) !Arr {
    const k = try getKernel(&sink_sm_causal_kernel, "sushi_hy4_sink_softmax_causal", &.{ "a", "b", "sinks", "meta" }, sink_softmax_causal_source);
    const sa = mlx.getShape(a);
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    const mv = [_]i32{ h, t, n, p0, window, kcfg.bits(scale) };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .sink_sm_causal, .a = h, .b = t, .c = n, .d = @intFromBool(heads_first) }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ a, b, sinks, meta }, cfg, &out);
    return out[0];
}

pub fn sinkSoftmaxMask(s: mlx.mlx_stream, a: Arr, b: Arr, mask: Arr, sinks: Arr, scale: f32, heads_first: bool) !Arr {
    const k = try getKernel(&sink_sm_mask_kernel, "sushi_hy4_sink_softmax_mask", &.{ "a", "b", "mask", "sinks", "meta" }, sink_softmax_mask_source);
    const sa = mlx.getShape(a);
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    const mask_bool = mlx.mlx_array_dtype(mask) == .bool_;
    const mv = [_]i32{ h, t, n, kcfg.bits(scale) };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const flags: c_int = @as(c_int, @intFromBool(heads_first)) | (@as(c_int, @intFromBool(mask_bool)) << 1);
    const cfg = try cfgs.get(.{ .kind = .sink_sm_mask, .a = h, .b = t, .c = n, .d = flags }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ a, b, mask, sinks, meta }, cfg, &out);
    return out[0];
}

pub const ring_keys_source: [:0]const u8 =
    \\uint col = thread_position_in_grid.x;
    \\uint i = thread_position_in_grid.y;
    \\uint r = thread_position_in_grid.z;
    \\int w = meta[1];
    \\int p0 = meta[2];
    \\int rp0 = meta[3];
    \\int lat = meta[4];
    \\int rope = meta[5];
    \\int lo = p0 + int(r) + 1 - w;
    \\int slot0 = (rp0 + int(r) + 1 - w) % w;
    \\if (slot0 < 0) slot0 += w;
    \\int src = lo + (int(i) < slot0 ? (w - slot0 + int(i)) : (int(i) - slot0));
    \\if (int(col) < lat) {
    \\  c_out[((size_t)r * (size_t)w + (size_t)i) * (size_t)lat + col] = static_cast<float>(c[(size_t)src * (size_t)lat + col]);
    \\} else {
    \\  uint pc = col - uint(lat);
    \\  pe_out[((size_t)r * (size_t)w + (size_t)i) * (size_t)rope + pc] = static_cast<float>(k_pe[(size_t)src * (size_t)rope + pc]);
    \\}
;

var ring_keys_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `ringKeys` builds a sliding block's windows: a GPU stream, f32 or bf16 latent and rope, the ring full
/// from the block's first row on (the caller decides the block is a verify block).
pub fn ringKeysApplies(s: mlx.mlx_stream, c: Arr, k_pe: Arr, t: c_int, p0: c_int, window: c_int, ring_p0: c_int) bool {
    const cs = mlx.getShape(c);
    const ps = mlx.getShape(k_pe);
    const cd = mlx.mlx_array_dtype(c);
    const pd = mlx.mlx_array_dtype(k_pe);
    return mlx.streamIsGpu(s) and cs.len == 2 and ps.len == 2 and cs[0] == ps[0] and cs[1] > 0 and ps[1] > 0 and
        (cd == .float32 or cd == .bfloat16) and (pd == .float32 or pd == .bfloat16) and
        t > 0 and window > 0 and ring_p0 + 1 > window and p0 + 1 >= window and p0 + t <= cs[0];
}

/// Each of a verify block's `t` rows' sliding window, widened to f32 in the order its one-row step reads the ring
/// (rotated at its own slot): [t, W, latent] and [t, W, rope], one launch for the block. The caller frees both.
pub fn ringKeys(s: mlx.mlx_stream, c: Arr, k_pe: Arr, t: c_int, p0: c_int, window: c_int, ring_p0: c_int) !struct { c: Arr, pe: Arr } {
    if (ring_keys_kernel == null) {
        const ins = [_][*:0]const u8{ "c", "k_pe", "meta" };
        const outs = [_][*:0]const u8{ "c_out", "pe_out" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_ring_keys", iv, ov, ring_keys_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        ring_keys_kernel = made;
    }
    const lat = mlx.getShape(c)[1];
    const rope = mlx.getShape(k_pe)[1];
    const mv = [_]i32{ t, window, p0, ring_p0, lat, rope };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .ring_keys, .a = t, .b = window, .c = lat, .d = rope }, build);
    var out: [2]Arr = undefined;
    try applyOuts(s, ring_keys_kernel.?, &.{ c, k_pe, meta }, cfg, &out);
    return .{ .c = out[0], .pe = out[1] };
}

/// A decode row's two score products: over the latent keys (`s1`) and over the rope keys (`s2`), f32.
pub const Scores = struct { s1: Arr, s2: Arr };

const SteelPlan = struct { parts: c_int, pk: c_int };

fn ceilDiv(a: c_int, b: c_int) c_int {
    return @divTrunc(a + b - 1, b);
}

/// MLX's choice for an f32 `[m, k] x [k, n]` product at these shapes (`steel_matmul` in the pinned mlx's
/// `matmul.cpp`): split-k into `parts` partitions of `pk` each, added in order, or one fused chain (`parts` 1); null where
/// MLX takes another kernel. The steel launches' bytes are MLX's only while this mirrors it: an mlx bump re-proves it.
fn steelPlan(m: c_int, n: c_int, k: c_int) ?SteelPlan {
    if (m < 32 or @rem(m, 32) != 0 or n < 32 or @rem(n, 8) != 0 or @rem(k, 16) != 0) return null;
    const tm16 = ceilDiv(m, 16);
    const tn16 = ceilDiv(n, 16);
    const tk = @divTrunc(k, 16);
    if (tm16 * tn16 > 1024) return null;
    if (!(tk >= 8 and k >= @max(m, n))) return .{ .parts = 1, .pk = k };
    const per = @divTrunc(tk, ceilDiv(m, 32) * ceilDiv(n, 32));
    var parts: c_int = 2;
    while (parts < per and parts < 32) parts *= 2;
    if (@rem(tk, parts) != 0) return null;
    return .{ .parts = parts, .pk = @divTrunc(tk, parts) * 16 };
}

pub const steel_scores_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int key0 = int(threadgroup_position_in_grid.x) * 8;
    \\const ulong qs = ulong(ql_strides[QA]);
    \\const ulong ps = ulong(qp_strides[PA]);
    \\const ulong cs = ulong(c_strides[CA]);
    \\const device float* qrow = ql + size_t(sg * 8) * size_t(qs);
    \\const device float* prow = qp + size_t(sg * 8) * size_t(ps);
    \\const device float* crow = c + size_t(key0) * size_t(cs);
    \\threadgroup float pes[8 * R];
    \\for (int e = int(thread_position_in_threadgroup.x); e < 8 * R; e += H * 4)
    \\  pes[e] = static_cast<float>(pe[size_t(key0 + e / R) * size_t(pe_strides[EA]) + size_t(e % R)]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\simdgroup_float8x8 acc[NPART];
    \\simdgroup_float8x8 rope = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\#pragma clang loop unroll(full)
    \\for (int p = 0; p < NPART; ++p) acc[p] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int kb = 0; kb < PK / 8; kb += U) {
    \\  simdgroup_float8x8 a[NPART][U], b[NPART][U];
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) {
    \\    #pragma clang loop unroll(full)
    \\    for (int u = 0; u < U; ++u) {
    \\      const int k0 = p * PK + (kb + u) * 8;
    \\      simdgroup_load(b[p][u], crow + k0, cs, ulong2(0, 0), true);
    \\      simdgroup_load(a[p][u], qrow + k0, qs);
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < U; ++u) {
    \\    #pragma clang loop unroll(full)
    \\    for (int p = 0; p < NPART; ++p) simdgroup_multiply_accumulate(acc[p], a[p][u], b[p][u], acc[p]);
    \\  }
    \\}
    \\{
    \\  simdgroup_float8x8 a[R / 8], b[R / 8];
    \\  #pragma clang loop unroll(full)
    \\  for (int kk = 0; kk < R / 8; ++kk) {
    \\    simdgroup_load(b[kk], pes + kk * 8, R, ulong2(0, 0), true);
    \\    simdgroup_load(a[kk], prow + kk * 8, ps);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int kk = 0; kk < R / 8; ++kk) simdgroup_multiply_accumulate(rope, a[kk], b[kk], rope);
    \\}
    \\simdgroup_float8x8 res;
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < 2; ++e) {
    \\  float v = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) v += acc[p].thread_elements()[e];
    \\  res.thread_elements()[e] = v;
    \\}
    \\simdgroup_store(res, s1 + size_t(sg * 8) * N + key0, N);
    \\simdgroup_store(rope, s2 + size_t(sg * 8) * N + key0, N);
;

pub const steel_values_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int d0 = int(threadgroup_position_in_grid.x) * 8;
    \\const ulong ps = ulong(pp_strides[PA]);
    \\const ulong cs = ulong(c_strides[CA]);
    \\const device float* prow = pp + size_t(sg * 8) * size_t(ps);
    \\const device float* ccol = c + d0;
    \\simdgroup_float8x8 acc[NPART];
    \\#pragma clang loop unroll(full)
    \\for (int p = 0; p < NPART; ++p) acc[p] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int kb = 0; kb < PK / 8; kb += U) {
    \\  simdgroup_float8x8 a[NPART][U], b[NPART][U];
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) {
    \\    #pragma clang loop unroll(full)
    \\    for (int u = 0; u < U; ++u) {
    \\      const int k0 = p * PK + (kb + u) * 8;
    \\      simdgroup_load(b[p][u], ccol + size_t(k0) * size_t(cs), cs);
    \\      simdgroup_load(a[p][u], prow + k0, ps);
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < U; ++u) {
    \\    #pragma clang loop unroll(full)
    \\    for (int p = 0; p < NPART; ++p) simdgroup_multiply_accumulate(acc[p], a[p][u], b[p][u], acc[p]);
    \\  }
    \\}
    \\simdgroup_float8x8 res;
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < 2; ++e) {
    \\  float v = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) v += acc[p].thread_elements()[e];
    \\  res.thread_elements()[e] = v;
    \\}
    \\simdgroup_store(res, u + size_t(sg * 8) * W + d0, W);
;

var steel_scores_kernel: ?mlx.mlx_fast_metal_kernel = null;
var steel_values_kernel: ?mlx.mlx_fast_metal_kernel = null;

const SteelKey = struct { h: c_int, n: c_int, w: c_int, r: c_int = 0, parts: c_int, pk: c_int, a0: c_int, a1: c_int, a2: c_int, a3: c_int = 0, scores: bool };
var steel_cfgs: kcfg.Cache(SteelKey) = .{};

/// The k-steps a simdgroup loads before its multiplies: 16 over the partitions, a divisor of the partition's steps.
fn batchOf(parts: c_int, pk: c_int) c_int {
    var u: c_int = @max(1, @divTrunc(16, parts));
    while (@rem(@divTrunc(pk, 8), u) != 0) u -= 1;
    return u;
}

fn steelBuild(k: SteelKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (k.scores) {
        const shape = if (k.a0 == 0) [3]c_int{ k.h, 1, k.n } else [3]c_int{ 1, k.h, k.n };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.n, 8) * k.h * 4, 1, 1));
        for ([_]struct { [:0]const u8, c_int }{ .{ "N", k.n }, .{ "R", k.r }, .{ "QA", k.a0 }, .{ "PA", k.a1 }, .{ "CA", k.a2 }, .{ "EA", k.a3 } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0].ptr, kv[1]));
    } else {
        const shape = if (k.a0 == 0) [3]c_int{ k.h, 1, k.w } else [3]c_int{ 1, k.h, k.w };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.w, 8) * k.h * 4, 1, 1));
        for ([_]struct { [:0]const u8, c_int }{ .{ "W", k.w }, .{ "PA", k.a0 }, .{ "CA", k.a1 } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0].ptr, kv[1]));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, k.h * 4, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", k.h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "U", batchOf(k.parts, k.pk)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NPART", k.parts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PK", k.pk));
    return cfg;
}

fn rowAxis(a: Arr) ?c_int {
    const sh = mlx.getShape(a);
    return switch (sh.len) {
        2 => 0,
        3 => if (sh[0] == 1 and sh[1] != 1) 1 else if (sh[1] == 1) 0 else null,
        else => null,
    };
}

fn rowsOf(a: Arr) c_int {
    const sh = mlx.getShape(a);
    return sh[@intCast(rowAxis(a).?)];
}

fn colsOf(a: Arr) c_int {
    const sh = mlx.getShape(a);
    return sh[sh.len - 1];
}

/// Whether `steelScores` takes these operands: an engine stream, one query row a head, and products `steelPlan` mirrors.
pub fn steelScoresApplies(s: mlx.mlx_stream, q_lat: Arr, q_pe: Arr, c: Arr, k_pe: Arr) bool {
    if (!mlx.streamIsGpu(s) or !mlx.engine.isStream(s)) return false;
    const qa = rowAxis(q_lat) orelse return false;
    const pa = rowAxis(q_pe) orelse return false;
    const ca = rowAxis(c) orelse return false;
    const ea = rowAxis(k_pe) orelse return false;
    _ = .{ qa, pa, ca, ea };
    if (mlx.mlx_array_dtype(q_lat) != .float32 or mlx.mlx_array_dtype(q_pe) != .float32 or mlx.mlx_array_dtype(c) != .float32 or mlx.mlx_array_dtype(k_pe) != .bfloat16) return false;
    const h = rowsOf(q_lat);
    const n = rowsOf(c);
    const lat = colsOf(c);
    const rope = colsOf(k_pe);
    if (rowsOf(q_pe) != h or rowsOf(k_pe) != n or colsOf(q_lat) != lat or colsOf(q_pe) != rope) return false;
    if (h != 32 and h != 64) return false;
    if (@rem(rope, 8) != 0 or rope > 128 or @rem(lat, 16) != 0) return false;
    if (steelPlan(h, n, rope) == null or steelPlan(h, n, rope).?.parts != 1) return false;
    return steelPlan(h, n, lat) != null and @rem(n, 8) == 0;
}

/// A decode row's score products over its keys in one launch, each element summed in MLX's order (`steelPlan`), so the
/// bytes are the op chain's `q_lat · cᵀ` and `q_pe · k_peᵀ`.
pub fn steelScores(s: mlx.mlx_stream, q_lat: Arr, q_pe: Arr, c: Arr, k_pe: Arr) !Scores {
    if (steel_scores_kernel == null) {
        const ins = [_][*:0]const u8{ "ql", "qp", "c", "pe" };
        const outs = [_][*:0]const u8{ "s1", "s2" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_steel_scores", iv, ov, steel_scores_source, "", false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        steel_scores_kernel = made;
    }
    const h = rowsOf(q_lat);
    const n = rowsOf(c);
    const plan = steelPlan(h, n, colsOf(c)).?;
    const cfg = try steel_cfgs.get(.{ .h = h, .n = n, .w = colsOf(c), .r = colsOf(k_pe), .parts = plan.parts, .pk = plan.pk, .a0 = rowAxis(q_lat).?, .a1 = rowAxis(q_pe).?, .a2 = rowAxis(c).?, .a3 = rowAxis(k_pe).?, .scores = true }, steelBuild);
    var out: [2]Arr = undefined;
    try applyOuts(s, steel_scores_kernel.?, &.{ q_lat, q_pe, c, k_pe }, cfg, &out);
    return .{ .s1 = out[0], .s2 = out[1] };
}

/// Whether `steelValues` takes `p` [H, 1, n] or [1, H, n] against `c`: an engine stream and a product `steelPlan` mirrors.
pub fn steelValuesApplies(s: mlx.mlx_stream, p: Arr, c: Arr) bool {
    if (!mlx.streamIsGpu(s) or !mlx.engine.isStream(s)) return false;
    _ = rowAxis(p) orelse return false;
    _ = rowAxis(c) orelse return false;
    if (mlx.mlx_array_dtype(p) != .float32 or mlx.mlx_array_dtype(c) != .float32) return false;
    const h = rowsOf(p);
    const n = rowsOf(c);
    const lat = colsOf(c);
    if (colsOf(p) != n or (h != 32 and h != 64) or @rem(lat, 8) != 0) return false;
    return steelPlan(h, lat, n) != null and @rem(n, 16) == 0;
}

/// A decode row's value product `p · c` in one launch, summed in MLX's order, so the bytes are the op chain's.
pub fn steelValues(s: mlx.mlx_stream, p: Arr, c: Arr) !Arr {
    if (steel_values_kernel == null) {
        const ins = [_][*:0]const u8{ "pp", "c" };
        const outs = [_][*:0]const u8{"u"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_steel_values", iv, ov, steel_values_source, "", false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        steel_values_kernel = made;
    }
    const h = rowsOf(p);
    const n = rowsOf(c);
    const lat = colsOf(c);
    const plan = steelPlan(h, lat, n).?;
    const cfg = try steel_cfgs.get(.{ .h = h, .n = n, .w = lat, .parts = plan.parts, .pk = plan.pk, .a0 = rowAxis(p).?, .a1 = rowAxis(c).?, .a2 = 0, .scores = false }, steelBuild);
    var out: [1]Arr = undefined;
    try applyOuts(s, steel_values_kernel.?, &.{ p, c }, cfg, &out);
    return out[0];
}

pub const steel_scores_groups_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int key0 = int(threadgroup_position_in_grid.x) * 8;
    \\const int grp = int(threadgroup_position_in_grid.y);
    \\const ulong qs = ulong(ql_strides[1]);
    \\const ulong ps = ulong(qp_strides[1]);
    \\const ulong cs = ulong(c_strides[1]);
    \\const device float* qrow = ql + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(qs);
    \\const device float* prow = qp + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(ps);
    \\const device float* crow = c + (size_t(grp) * size_t(N) + size_t(key0)) * size_t(cs);
    \\threadgroup float pes[8 * R];
    \\for (int e = int(thread_position_in_threadgroup.x); e < 8 * R; e += H * 4)
    \\  pes[e] = static_cast<float>(pe[(size_t(grp) * size_t(N) + size_t(key0 + e / R)) * size_t(R) + size_t(e % R)]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\simdgroup_float8x8 acc[NPART];
    \\simdgroup_float8x8 rope = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\#pragma clang loop unroll(full)
    \\for (int p = 0; p < NPART; ++p) acc[p] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int kb = 0; kb < PK / 8; kb += U) {
    \\  simdgroup_float8x8 a[NPART][U], b[NPART][U];
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) {
    \\    #pragma clang loop unroll(full)
    \\    for (int u = 0; u < U; ++u) {
    \\      const int k0 = p * PK + (kb + u) * 8;
    \\      simdgroup_load(b[p][u], crow + k0, cs, ulong2(0, 0), true);
    \\      simdgroup_load(a[p][u], qrow + k0, qs);
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < U; ++u) {
    \\    #pragma clang loop unroll(full)
    \\    for (int p = 0; p < NPART; ++p) simdgroup_multiply_accumulate(acc[p], a[p][u], b[p][u], acc[p]);
    \\  }
    \\}
    \\{
    \\  simdgroup_float8x8 a[R / 8], b[R / 8];
    \\  #pragma clang loop unroll(full)
    \\  for (int kk = 0; kk < R / 8; ++kk) {
    \\    simdgroup_load(b[kk], pes + kk * 8, R, ulong2(0, 0), true);
    \\    simdgroup_load(a[kk], prow + kk * 8, ps);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int kk = 0; kk < R / 8; ++kk) simdgroup_multiply_accumulate(rope, a[kk], b[kk], rope);
    \\}
    \\simdgroup_float8x8 res;
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < 2; ++e) {
    \\  float v = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) v += acc[p].thread_elements()[e];
    \\  res.thread_elements()[e] = v;
    \\}
    \\const device float* s1g = s1 + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(N) + size_t(key0);
    \\const device float* s2g = s2 + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(N) + size_t(key0);
    \\simdgroup_store(res, const_cast<device float*>(s1g), N);
    \\simdgroup_store(rope, const_cast<device float*>(s2g), N);
;

pub const steel_values_groups_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const int d0 = int(threadgroup_position_in_grid.x) * 8;
    \\const int grp = int(threadgroup_position_in_grid.y);
    \\const ulong ps = ulong(pp_strides[1]);
    \\const ulong cs = ulong(c_strides[1]);
    \\const device float* prow = pp + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(ps);
    \\const device float* ccol = c + size_t(grp) * size_t(N) * size_t(W) + size_t(d0);
    \\simdgroup_float8x8 acc[NPART];
    \\#pragma clang loop unroll(full)
    \\for (int p = 0; p < NPART; ++p) acc[p] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int kb = 0; kb < PK / 8; kb += U) {
    \\  simdgroup_float8x8 a[NPART][U], b[NPART][U];
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) {
    \\    #pragma clang loop unroll(full)
    \\    for (int u = 0; u < U; ++u) {
    \\      const int k0 = p * PK + (kb + u) * 8;
    \\      simdgroup_load(b[p][u], ccol + size_t(k0) * size_t(cs), cs);
    \\      simdgroup_load(a[p][u], prow + k0, ps);
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < U; ++u) {
    \\    #pragma clang loop unroll(full)
    \\    for (int p = 0; p < NPART; ++p) simdgroup_multiply_accumulate(acc[p], a[p][u], b[p][u], acc[p]);
    \\  }
    \\}
    \\simdgroup_float8x8 res;
    \\#pragma clang loop unroll(full)
    \\for (int e = 0; e < 2; ++e) {
    \\  float v = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int p = 0; p < NPART; ++p) v += acc[p].thread_elements()[e];
    \\  res.thread_elements()[e] = v;
    \\}
    \\simdgroup_store(res, u + (size_t(grp) * size_t(H) + size_t(sg * 8)) * size_t(W) + size_t(d0), W);
;

var steel_scores_groups_kernel: ?mlx.mlx_fast_metal_kernel = null;
var steel_values_groups_kernel: ?mlx.mlx_fast_metal_kernel = null;

const SteelGroupsKey = struct { h: c_int, n: c_int, w: c_int, r: c_int = 0, parts: c_int, pk: c_int, groups: c_int, scores: bool };
var steel_groups_cfgs: kcfg.Cache(SteelGroupsKey) = .{};

fn steelGroupsBuild(k: SteelGroupsKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (k.scores) {
        const shape = [_]c_int{ k.groups, k.h, k.n };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.n, 8) * k.h * 4, k.groups, 1));
        for ([_]struct { [:0]const u8, c_int }{ .{ "N", k.n }, .{ "R", k.r } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0].ptr, kv[1]));
    } else {
        const shape = [_]c_int{ k.groups, k.h, k.w };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(k.w, 8) * k.h * 4, k.groups, 1));
        for ([_]struct { [:0]const u8, c_int }{ .{ "W", k.w }, .{ "N", k.n } }) |kv|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0].ptr, kv[1]));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, k.h * 4, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", k.h));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "U", batchOf(k.parts, k.pk)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NPART", k.parts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "PK", k.pk));
    return cfg;
}

pub fn steelScoresGroupsFits(h: c_int, n: c_int, lat: c_int, rope: c_int) bool {
    if (h != 32 and h != 64) return false;
    if (@rem(rope, 8) != 0 or rope > 128 or @rem(lat, 16) != 0) return false;
    const rp = steelPlan(h, n, rope) orelse return false;
    if (rp.parts != 1) return false;
    return steelPlan(h, n, lat) != null and @rem(n, 8) == 0;
}

pub fn steelValuesGroupsFits(h: c_int, n: c_int, lat: c_int) bool {
    if ((h != 32 and h != 64) or @rem(lat, 8) != 0) return false;
    return steelPlan(h, lat, n) != null and @rem(n, 16) == 0;
}

pub fn steelScoresGroupsApplies(s: mlx.mlx_stream, groups: c_int, ql: Arr, qp: Arr, c: Arr, pe: Arr) bool {
    if (!mlx.streamIsGpu(s) or !mlx.engine.isStream(s) or groups < 2) return false;
    const qs = mlx.getShape(ql);
    const ps = mlx.getShape(qp);
    const cs = mlx.getShape(c);
    const es = mlx.getShape(pe);
    if (qs.len != 4 or ps.len != 4 or cs.len != 3 or es.len != 3) return false;
    if (qs[0] != groups or ps[0] != groups or cs[0] != groups or es[0] != groups) return false;
    if (qs[1] != ps[1] or qs[2] != 1 or ps[2] != 1) return false;
    if (cs[1] != es[1] or cs[2] != qs[3] or ps[3] != es[2]) return false;
    if (mlx.mlx_array_dtype(ql) != .float32 or mlx.mlx_array_dtype(qp) != .float32 or mlx.mlx_array_dtype(c) != .float32 or mlx.mlx_array_dtype(pe) != .bfloat16) return false;
    return steelScoresGroupsFits(qs[1], cs[1], cs[2], es[2]);
}

pub fn steelScoresGroups(s: mlx.mlx_stream, groups: c_int, ql: Arr, qp: Arr, c: Arr, pe: Arr) !Scores {
    if (steel_scores_groups_kernel == null) {
        const ins = [_][*:0]const u8{ "ql", "qp", "c", "pe" };
        const outs = [_][*:0]const u8{ "s1", "s2" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_steel_scores_groups", iv, ov, steel_scores_groups_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        steel_scores_groups_kernel = made;
    }
    const h = mlx.getShape(ql)[1];
    const n = mlx.getShape(c)[1];
    const lat = mlx.getShape(c)[2];
    const rope = mlx.getShape(pe)[2];
    const plan = steelPlan(h, n, lat).?;
    const cfg = try steel_groups_cfgs.get(.{ .h = h, .n = n, .w = lat, .r = rope, .parts = plan.parts, .pk = plan.pk, .groups = groups, .scores = true }, steelGroupsBuild);
    var out: [2]Arr = undefined;
    try applyOuts(s, steel_scores_groups_kernel.?, &.{ ql, qp, c, pe }, cfg, &out);
    return .{ .s1 = out[0], .s2 = out[1] };
}

pub fn steelValuesGroupsApplies(s: mlx.mlx_stream, groups: c_int, p: Arr, c: Arr) bool {
    if (!mlx.streamIsGpu(s) or !mlx.engine.isStream(s) or groups < 2) return false;
    const ps = mlx.getShape(p);
    const cs = mlx.getShape(c);
    if (ps.len != 4 or cs.len != 3) return false;
    if (ps[0] != groups or cs[0] != groups or ps[2] != 1) return false;
    if (mlx.mlx_array_dtype(p) != .float32 or mlx.mlx_array_dtype(c) != .float32) return false;
    if (cs[1] != ps[3]) return false;
    return steelValuesGroupsFits(ps[1], ps[3], cs[2]);
}

pub fn steelValuesGroups(s: mlx.mlx_stream, groups: c_int, p: Arr, c: Arr) !Arr {
    if (steel_values_groups_kernel == null) {
        const ins = [_][*:0]const u8{ "pp", "c" };
        const outs = [_][*:0]const u8{"u"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_steel_values_groups", iv, ov, steel_values_groups_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        steel_values_groups_kernel = made;
    }
    const h = mlx.getShape(p)[1];
    const n = mlx.getShape(p)[3];
    const lat = mlx.getShape(c)[2];
    const plan = steelPlan(h, lat, n).?;
    const cfg = try steel_groups_cfgs.get(.{ .h = h, .n = n, .w = lat, .parts = plan.parts, .pk = plan.pk, .groups = groups, .scores = false }, steelGroupsBuild);
    var out: [1]Arr = undefined;
    try applyOuts(s, steel_values_groups_kernel.?, &.{ p, c }, cfg, &out);
    return out[0];
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────
test "hy4 chain: N decode rows' score and value products in one launch each keep their solo launches' bytes (release geometry, 2 to 16 rows)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const h: c_int = 64;
    const n: c_int = 512;
    const lat: c_int = 512;
    const rope: c_int = 64;
    for (2..17) |g| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        var ql: [16]Arr = undefined;
        var qp: [16]Arr = undefined;
        var ck: [16]Arr = undefined;
        var pe: [16]Arr = undefined;
        var pr: [16]Arr = undefined;
        var qw: [16]Arr = undefined;
        var pw: [16]Arr = undefined;
        var cw: [16]Arr = undefined;
        var ew: [16]Arr = undefined;
        var vw: [16]Arr = undefined;
        for (0..g) |i| {
            ql[i] = try randomF32(&o, &.{ h, 1, lat }, 3000 + g * 10 + i, 1.5);
            qp[i] = try randomF32(&o, &.{ h, 1, rope }, 3500 + g * 10 + i, 1.0);
            ck[i] = try randomF32(&o, &.{ n, lat }, 4000 + g * 10 + i, 2.0);
            pe[i] = try o.cast(try randomF32(&o, &.{ n, rope }, 4500 + g * 10 + i, 1.0), .bfloat16);
            pr[i] = try randomF32(&o, &.{ h, 1, n }, 5000 + g * 10 + i, 1.0);
            qw[i] = try o.reshape(ql[i], &.{ 1, h, 1, lat });
            pw[i] = try o.reshape(qp[i], &.{ 1, h, 1, rope });
            cw[i] = try o.reshape(ck[i], &.{ 1, n, lat });
            ew[i] = try o.reshape(pe[i], &.{ 1, n, rope });
            vw[i] = try o.reshape(pr[i], &.{ 1, h, 1, n });
            for ([_]Arr{ ql[i], qp[i], ck[i], pe[i], pr[i] }) |a| try mlx.check(mlx.mlx_array_eval(a));
        }
        var want_s1: [16][]const u8 = undefined;
        var want_s2: [16][]const u8 = undefined;
        var want_u: [16][]const u8 = undefined;
        var solo_l: u64 = 0;
        for (0..g) |i| {
            const d0 = mlx.engine.stats().dispatches;
            const sc = try steelScores(es, ql[i], qp[i], ck[i], pe[i]);
            _ = try o.own(sc.s1);
            _ = try o.own(sc.s2);
            want_s1[i] = try hostF32(&o, sc.s1);
            want_s2[i] = try hostF32(&o, sc.s2);
            const u = try steelValues(es, pr[i], ck[i]);
            _ = try o.own(u);
            want_u[i] = try hostF32(&o, u);
            solo_l += mlx.engine.stats().dispatches - d0;
        }
        const s1l = try o.concat(qw[0..g], 0);
        const s2l = try o.concat(pw[0..g], 0);
        const c_l = try o.concat(cw[0..g], 0);
        const e_l = try o.concat(ew[0..g], 0);
        const v_l = try o.concat(vw[0..g], 0);
        for ([_]Arr{ s1l, s2l, c_l, e_l, v_l }) |a| try mlx.check(mlx.mlx_array_eval(a));
        try testing.expect(steelScoresGroupsApplies(es, @intCast(g), s1l, s2l, c_l, e_l));
        try testing.expect(steelValuesGroupsApplies(es, @intCast(g), v_l, c_l));
        const d1 = mlx.engine.stats().dispatches;
        const sc = try steelScoresGroups(es, @intCast(g), s1l, s2l, c_l, e_l);
        _ = try o.own(sc.s1);
        _ = try o.own(sc.s2);
        const u = try steelValuesGroups(es, @intCast(g), v_l, c_l);
        _ = try o.own(u);
        const joined_l = mlx.engine.stats().dispatches - d1;
        for (0..g) |i| {
            const sl = @as(usize, @intCast(i));
            try testing.expectEqualSlices(u8, want_s1[i], try hostF32(&o, try o.slice(sc.s1, 0, @intCast(sl), @intCast(sl + 1))));
            try testing.expectEqualSlices(u8, want_s2[i], try hostF32(&o, try o.slice(sc.s2, 0, @intCast(sl), @intCast(sl + 1))));
            try testing.expectEqualSlices(u8, want_u[i], try hostF32(&o, try o.slice(u, 0, @intCast(sl), @intCast(sl + 1))));
        }
        try testing.expect(joined_l <= 2);
        try testing.expect(joined_l < solo_l);
    }
}


const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn hostF32(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

fn randomF32(o: *Ops, shape: []const c_int, seed: u64, scale: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * scale);
    // Exact zeros, both signs, and values past the sigmoid's saturation.
    if (n > 4) {
        v[0] = 0;
        v[1] = -0.0;
        v[2] = 120;
        v[3] = -120;
    }
    return o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
}

test "hy4 chain: the SwiGLU in one launch gives the bytes of Ops.silu then the multiply, f32, at the shared expert's and the dense FFN's widths" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_][2]c_int{ .{ 1, 1536 }, .{ 1, 13824 }, .{ 7, 1536 }, .{ 1, 256 }, .{ 3, 128 } }, 0..) |sh, i| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const g = try randomF32(&o, &sh, 2 * i + 1, 4);
        const u = try randomF32(&o, &sh, 2 * i + 2, 3);
        const want = try hostF32(&o, try o.binary(.mul, try o.silu(g), u));
        try testing.expect(swigluApplies(es, g, u));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try swiglu(es, g, u));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostF32(&o, got));
    }
}

fn hostAny(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..n]),
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(c) orelse return error.HostRead)[0..n]),
        .bool_ => std.mem.sliceAsBytes((mlx.mlx_array_data_bool(c) orelse return error.HostRead)[0..n]),
        else => error.UnexpectedDtype,
    };
}

fn dispatchesOf(comptime f: anytype, args: anytype) !struct { u64, @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.payload } {
    const before = mlx.engine.stats().dispatches;
    const r = try @call(.auto, f, args);
    return .{ mlx.engine.stats().dispatches - before, r };
}

test "hy4 chain: the indexer's groups and recalled rows in one launch each give the bytes of the ops, at the release's 511 picks of groups of 4 and the tiny's, decode and verify rows, near the context's start and far out" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { t: c_int, take: c_int, ratio: c_int, p0: c_int };
    for ([_]Case{
        .{ .t = 1, .take = 511, .ratio = 4, .p0 = 4096 },
        .{ .t = 6, .take = 511, .ratio = 4, .p0 = 1_000_003 },
        .{ .t = 1, .take = 3, .ratio = 4, .p0 = 600 },
        .{ .t = 3, .take = 3, .ratio = 4, .p0 = 17 },
    }) |c| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const n = c.p0 + c.t;
        const groups_seen = @divTrunc(n, c.ratio);
        const v = try testing.allocator.alloc(i32, @intCast(c.t * c.take));
        defer testing.allocator.free(v);
        var r = std.Random.DefaultPrng.init(@intCast(c.p0));
        // Picks anywhere in the pooled groups, the row's own and later ones among them (a row with few earlier groups).
        for (v) |*d| d.* = r.random().intRangeLessThan(i32, 0, @max(1, groups_seen + 1));
        const order = try o.own(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ c.t, c.take }, 2, .int32));
        const own_v = try testing.allocator.alloc(f32, @intCast(c.t));
        defer testing.allocator.free(own_v);
        for (own_v, 0..) |*d, i| d.* = @floatFromInt(c.p0 + @as(c_int, @intCast(i)));
        const pos = try o.own(mlx.mlx_array_new_data(own_v.ptr, &[_]c_int{c.t}, 1, .float32));
        const fd = try o.slot();
        try mlx.check(mlx.mlx_floor_divide(fd, pos, try o.scalar(@floatFromInt(c.ratio), .float32), es));
        const want_groups = try o.concat(&.{ try o.cast(order, .float32), try o.reshape(fd.*, &.{ c.t, 1 }) }, 1);
        const want_g = try hostAny(&o, want_groups);
        try mlx.check(mlx.mlx_array_eval(order));
        const ng, const groups = try dispatchesOf(recallGroups, .{ es, order, c.p0, c.ratio });
        _ = try o.own(groups);
        try mlx.check(mlx.mlx_array_eval(groups));
        try testing.expectEqual(@as(u64, 1), ng);
        try testing.expectEqualSlices(u8, want_g, try hostAny(&o, groups));

        fwd.chain_kernels_on = false;
        const want = try fwd.recalledRows(&o, c.t, n, c.p0, groups, c.ratio);
        const want_rows = try hostAny(&o, want.rows);
        const want_keep = try hostAny(&o, want.keep);
        fwd.chain_kernels_on = true;
        const nr, const got = try dispatchesOf(recalledRows, .{ es, groups, c.p0, n, c.ratio });
        _ = try o.own(got.rows);
        _ = try o.own(got.keep);
        try mlx.check(mlx.mlx_array_eval(got.keep));
        try testing.expectEqual(@as(u64, 1), nr);
        try testing.expectEqualSlices(u8, want_rows, try hostAny(&o, got.rows));
        try testing.expectEqualSlices(u8, want_keep, try hostAny(&o, got.keep));
    }
}

test "hy4 chain: the attention output gate in one launch gives the bytes of the cast, the Sigmoid, the Multiply and the transpose, BF16 and f32 gate rows, the release's 64 and 32 heads of 256, a decode row and a block" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_][3]c_int{ .{ 64, 1, 256 }, .{ 32, 1, 256 }, .{ 64, 6, 256 }, .{ 8, 3, 32 } }, 0..) |sh, i| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |gdt| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const h = sh[0];
        const t = sh[1];
        const v = sh[2];
        // `u` as the value product leaves it: [T, H, v] seen as [H, T, v].
        const u = try o.transpose(try randomF32(&o, &.{ t, h, v }, 3 * i + 1, 2), &.{ 1, 0, 2 });
        const gate = try o.cast(try randomF32(&o, &.{ t, h * v }, 3 * i + 2, 3), gdt);
        const g32 = try o.unary(.sigmoid, try o.transpose(try o.reshape(try o.cast(gate, .float32), &.{ t, h, v }), &.{ 1, 0, 2 }));
        const want = try hostF32(&o, try o.reshape(try o.transpose(try o.binary(.mul, u, g32), &.{ 1, 0, 2 }), &.{ t, h * v }));
        try testing.expect(gatedApplies(es, u, gate));
        try mlx.check(mlx.mlx_array_eval(u));
        try mlx.check(mlx.mlx_array_eval(gate));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try gated(es, u, gate));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostF32(&o, got));
    };
}

test "hy4 chain: attention score add, scale, mask, sink concat and softmax in one launch gives the op chain's bytes at the release's sliding and gathered shapes, decode and verify blocks, causal, bool keep and additive masks" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const Case = struct { h: c_int, t: c_int, s: c_int, heads_first: bool, scale: f32 };
    const cases = [_]Case{
        .{ .h = 64, .t = 1, .s = 128, .heads_first = true, .scale = 1.0 / @sqrt(@as(f32, 192.0)) },
        .{ .h = 64, .t = 1, .s = 512, .heads_first = true, .scale = 0.0625 },
        .{ .h = 64, .t = 6, .s = 512, .heads_first = true, .scale = 0.0625 },
        .{ .h = 32, .t = 1, .s = 128, .heads_first = false, .scale = 1.0 / @sqrt(@as(f32, 192.0)) },
        .{ .h = 32, .t = 1, .s = 512, .heads_first = false, .scale = 0.0625 },
        .{ .h = 32, .t = 1, .s = 2048, .heads_first = false, .scale = 0.0625 },
        .{ .h = 32, .t = 6, .s = 2048, .heads_first = false, .scale = 0.0625 },
    };

    for (cases, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const sh = if (c.heads_first) [3]c_int{ c.h, c.t, c.s } else [3]c_int{ c.t, c.h, c.s };
        const a = try randomF32(&o, &sh, 100 * ci + 1, 5.0);
        const b = try randomF32(&o, &sh, 100 * ci + 2, 5.0);
        const sinks = try randomF32(&o, &.{c.h}, 100 * ci + 3, 2.0);
        const sink_sh: [3]c_int = if (c.heads_first) .{ c.h, 1, 1 } else .{ 1, c.h, 1 };
        const sink_full: [3]c_int = if (c.heads_first) .{ c.h, c.t, 1 } else .{ c.t, c.h, 1 };
        const sink_col = try o.broadcast(try o.reshape(sinks, &sink_sh), &sink_full);

        const kn: usize = @intCast(c.t * c.s);
        const kv = try testing.allocator.alloc(bool, kn);
        defer testing.allocator.free(kv);
        var prng = std.Random.DefaultPrng.init(100 * ci + 4);
        for (0..@intCast(c.t)) |row| {
            for (0..@intCast(c.s)) |col| {
                kv[row * @as(usize, @intCast(c.s)) + col] = if (c.t > 1 and row == 0)
                    false
                else if (c.t > 1 and row == 1)
                    true
                else
                    prng.random().boolean();
            }
        }
        const keep = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ c.t, c.s }, 2, .bool_));
        const add_m = try o.slot();
        try mlx.check(mlx.mlx_where(add_m, keep, try o.scalar(0, .float32), try o.scalar(-std.math.inf(f32), .float32), es));
        const m_bcast = if (c.heads_first) add_m.* else try o.reshape(add_m.*, &.{ c.t, 1, c.s });
        const want_m_arr = try o.slice(try o.softmax(try o.concat(&.{ try o.binary(.add, try o.binary(.mul, try o.binary(.add, a, b), try o.scalar(c.scale, .float32)), m_bcast), sink_col }, 2), 2), 2, 0, c.s);
        const want_m = try hostF32(&o, want_m_arr);

        for ([_]Arr{ a, b, sinks, keep, add_m.* }) |arr| try mlx.check(mlx.mlx_array_eval(arr));

        try testing.expect(sinkSoftmaxMaskApplies(es, a, b, keep, sinks, c.heads_first));
        const d_bool, const got_bool = try dispatchesOf(sinkSoftmaxMask, .{ es, a, b, keep, sinks, c.scale, c.heads_first });
        _ = try o.own(got_bool);
        try mlx.check(mlx.mlx_array_eval(got_bool));
        try testing.expectEqual(@as(u64, 1), d_bool);
        try testing.expectEqualSlices(u8, want_m, try hostF32(&o, got_bool));

        try testing.expect(sinkSoftmaxMaskApplies(es, a, b, add_m.*, sinks, c.heads_first));
        const d_f32, const got_f32 = try dispatchesOf(sinkSoftmaxMask, .{ es, a, b, add_m.*, sinks, c.scale, c.heads_first });
        _ = try o.own(got_f32);
        try mlx.check(mlx.mlx_array_eval(got_f32));
        try testing.expectEqual(@as(u64, 1), d_f32);
        try testing.expectEqualSlices(u8, want_m, try hostF32(&o, got_f32));

        if (c.heads_first) {
            const p0: c_int = if (c.t == 1) c.s - 1 else c.s - c.t;
            const win: c_int = if (c.t == 1) 0 else @divTrunc(c.s, 2);
            const cm = try fwd.causalMask(&o, c.t, c.s, p0, win);
            const want_c_arr = try o.slice(try o.softmax(try o.concat(&.{ try o.binary(.add, try o.binary(.mul, try o.binary(.add, a, b), try o.scalar(c.scale, .float32)), cm), sink_col }, 2), 2), 2, 0, c.s);
            const want_c = try hostF32(&o, want_c_arr);
            try testing.expect(sinkSoftmaxCausalApplies(es, a, b, sinks, true));
            const d_c, const got_c = try dispatchesOf(sinkSoftmaxCausal, .{ es, a, b, sinks, c.scale, p0, win, true });
            _ = try o.own(got_c);
            try mlx.check(mlx.mlx_array_eval(got_c));
            try testing.expectEqual(@as(u64, 1), d_c);
            try testing.expectEqualSlices(u8, want_c, try hostF32(&o, got_c));
        }
    }
}

test "hy4 chain: gatedInto writes spans directly into a bf16 or f32 destination block in one launch per span, byte for byte" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |out_dt| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const h: c_int = 32;
        const t: c_int = 12;
        const v: c_int = 256;
        const cut: c_int = 5;
        const u = try o.transpose(try randomF32(&o, &.{ t, h, v }, 41, 2), &.{ 1, 0, 2 });
        const gate = try o.cast(try randomF32(&o, &.{ t, h * v }, 42, 3), .bfloat16);
        const want = try o.cast(try o.own(try gated(es, u, gate)), out_dt);
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(u));
        try mlx.check(mlx.mlx_array_eval(gate));
        var dst = mlx.mlx_array_new();
        try mlx.check(mlx.engine.freshArray(&dst, &.{ t, h * v }, out_dt));
        _ = try o.own(dst);
        const ua = try o.slice(u, 1, 0, cut);
        const g0 = try o.slice(gate, 0, 0, cut);
        const ub = try o.slice(u, 1, cut, t);
        const g1 = try o.slice(gate, 0, cut, t);
        try testing.expect(gatedIntoApplies(es, ua, g0, dst));
        const before = mlx.engine.stats().dispatches;
        try gatedInto(es, ua, g0, dst, 0);
        try gatedInto(es, ub, g1, dst, cut);
        try mlx.check(mlx.mlx_array_eval(dst));
        try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
        const c_want = try o.contiguous(want);
        const c_got = try o.contiguous(dst);
        try mlx.check(mlx.mlx_array_eval(c_want));
        try mlx.check(mlx.mlx_array_eval(c_got));
        const n = mlx.mlx_array_size(c_want);
        if (out_dt == .bfloat16) {
            const wb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(c_want) orelse return error.HostRead);
            const gb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(c_got) orelse return error.HostRead);
            try testing.expectEqualSlices(u16, wb[0..n], gb[0..n]);
        } else {
            const wf = (mlx.mlx_array_data_float32(c_want) orelse return error.HostRead)[0..n];
            const gf = (mlx.mlx_array_data_float32(c_got) orelse return error.HostRead)[0..n];
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(wf), std.mem.sliceAsBytes(gf));
        }
    }
}

test "hy4 chain: swigluAs accepts bf16 inputs and writes bf16 output in one launch, byte for byte" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const g_bf = try o.cast(try randomF32(&o, &.{ 16, 1536 }, 81, 4), .bfloat16);
    const u_bf = try o.cast(try randomF32(&o, &.{ 16, 1536 }, 82, 3), .bfloat16);
    const want = try o.cast(try o.own(try swiglu(es, try o.cast(g_bf, .float32), try o.cast(u_bf, .float32))), .bfloat16);
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(g_bf));
    try mlx.check(mlx.mlx_array_eval(u_bf));
    try testing.expect(swigluApplies(es, g_bf, u_bf));
    const before = mlx.engine.stats().dispatches;
    const got = try o.own(try swigluAs(es, g_bf, u_bf, .bfloat16));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    const cw = try o.contiguous(want);
    const cg = try o.contiguous(got);
    try mlx.check(mlx.mlx_array_eval(cw));
    try mlx.check(mlx.mlx_array_eval(cg));
    const n = mlx.mlx_array_size(cw);
    const wb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(cw) orelse return error.HostRead);
    const gb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(cg) orelse return error.HostRead);
    try testing.expectEqualSlices(u16, wb[0..n], gb[0..n]);
}

test "hy4 chain: a decode row's score and value products as one launch each give the bytes of MLX's split-k and fused GEMMs, at the release's sliding window and selected rows" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const lat: c_int = 512;
    const rope: c_int = 64;
    const Case = struct { h: c_int, n: c_int, heads_lead: bool };
    const cases = [_]Case{
        .{ .h = 64, .n = 512, .heads_lead = true },
        .{ .h = 64, .n = 256, .heads_lead = true },
        .{ .h = 64, .n = 128, .heads_lead = true },
        .{ .h = 64, .n = 64, .heads_lead = true },
        .{ .h = 32, .n = 2048, .heads_lead = false },
        .{ .h = 32, .n = 512, .heads_lead = false },
        .{ .h = 32, .n = 128, .heads_lead = false },
    };
    for (cases, 0..) |cs, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed = 1000 * ci;
        const q_full = try randomF32(&o, &.{ cs.h, 1, 256 }, seed + 1, 1.5);
        const q_pe_h = try o.slice(q_full, 2, 192, 256);
        const q_lat_h = try randomF32(&o, &.{ cs.h, 1, lat }, seed + 2, 1.5);
        const keys = try randomF32(&o, &.{ cs.n + 7, lat }, seed + 3, 2.0);
        const c = try o.slice(keys, 0, 3, 3 + cs.n);
        const pe_f = try randomF32(&o, &.{ cs.n + 7, rope }, seed + 4, 1.0);
        const pe_all = try o.cast(pe_f, .bfloat16);
        const pe = try o.slice(pe_all, 0, 3, 3 + cs.n);
        const probs = try randomF32(&o, &.{ cs.h, 1, cs.n }, seed + 5, 0.01);

        const q_lat = if (cs.heads_lead) q_lat_h else try o.transpose(q_lat_h, &.{ 1, 0, 2 });
        const q_pe = if (cs.heads_lead) q_pe_h else try o.transpose(q_pe_h, &.{ 1, 0, 2 });
        const p = if (cs.heads_lead) probs else try o.transpose(probs, &.{ 1, 0, 2 });
        const c_in = if (cs.heads_lead) c else try o.reshape(c, &.{ 1, cs.n, lat });
        const pe_in = if (cs.heads_lead) pe else try o.reshape(pe, &.{ 1, cs.n, rope });
        const pe32 = try o.cast(pe, .float32);
        const want1 = if (cs.heads_lead) try o.binary(.mm, q_lat, try o.transpose(c, &.{ 1, 0 })) else try o.binary(.mm, q_lat, try o.transpose(c_in, &.{ 0, 2, 1 }));
        const want2 = if (cs.heads_lead) try o.binary(.mm, q_pe, try o.transpose(pe32, &.{ 1, 0 })) else try o.binary(.mm, q_pe, try o.transpose(try o.reshape(pe32, &.{ 1, cs.n, rope }), &.{ 0, 2, 1 }));
        const wantu = if (cs.heads_lead) try o.binary(.mm, p, c) else try o.binary(.mm, p, c_in);
        const w1 = try hostF32(&o, want1);
        const w2 = try hostF32(&o, want2);
        const wu = try hostF32(&o, wantu);
        for ([_]Arr{ q_lat, q_pe, c_in, pe_in, p }) |a| try mlx.check(mlx.mlx_array_eval(a));

        try testing.expect(steelScoresApplies(es, q_lat, q_pe, c_in, pe_in));
        const d_s, const got = try dispatchesOf(steelScores, .{ es, q_lat, q_pe, c_in, pe_in });
        _ = try o.own(got.s1);
        _ = try o.own(got.s2);
        try mlx.check(mlx.mlx_array_eval(got.s1));
        try mlx.check(mlx.mlx_array_eval(got.s2));
        try testing.expectEqual(@as(u64, 1), d_s);
        try testing.expectEqualSlices(u8, w1, try hostF32(&o, got.s1));
        try testing.expectEqualSlices(u8, w2, try hostF32(&o, got.s2));

        try testing.expect(steelValuesApplies(es, p, c_in));
        const d_v, const gu = try dispatchesOf(steelValues, .{ es, p, c_in });
        _ = try o.own(gu);
        try mlx.check(mlx.mlx_array_eval(gu));
        try testing.expectEqual(@as(u64, 1), d_v);
        try testing.expectEqualSlices(u8, wu, try hostF32(&o, gu));
    }
}

test "hy4 chain: gatedPreparedInto fuses gatedInto and prepare_h128 in one launch per span, byte for byte" {
    const exl3 = @import("sushi_exl3");
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const h: c_int = 32;
    const t: c_int = 12;
    const v: c_int = 256;
    const cut: c_int = 5;
    const u = try o.transpose(try randomF32(&o, &.{ t, h, v }, 61, 2), &.{ 1, 0, 2 });
    const gate = try o.cast(try randomF32(&o, &.{ t, h * v }, 62, 3), .bfloat16);
    const suh = try o.cast(try randomF32(&o, &.{ 1, h * v }, 63, 0.05), .float16);
    const plain = try o.cast(try o.own(try gated(es, u, gate)), .bfloat16);
    const inputs = try exl3.linear.gemmInputs(1, @intCast(t), exl3.kernels.gemmWindowRows());
    const want = try o.own(try exl3.kernels.prepareIndexed(es, plain, suh, inputs.slots));
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(u));
    try mlx.check(mlx.mlx_array_eval(gate));
    try mlx.check(mlx.mlx_array_eval(suh));
    var dst = mlx.mlx_array_new();
    try mlx.check(mlx.engine.freshArray(&dst, &.{ t, h * v }, .float16));
    _ = try o.own(dst);
    const ua = try o.slice(u, 1, 0, cut);
    const g0 = try o.slice(gate, 0, 0, cut);
    const ub = try o.slice(u, 1, cut, t);
    const g1 = try o.slice(gate, 0, cut, t);
    try testing.expect(gatedPreparedIntoApplies(es, ua, g0, suh, dst));
    const before = mlx.engine.stats().dispatches;
    try gatedPreparedInto(es, ua, g0, suh, dst, 0);
    try gatedPreparedInto(es, ub, g1, suh, dst, cut);
    try mlx.check(mlx.mlx_array_eval(dst));
    try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
    const cw = try o.contiguous(want);
    const cg = try o.contiguous(dst);
    try mlx.check(mlx.mlx_array_eval(cw));
    try mlx.check(mlx.mlx_array_eval(cg));
    const n = mlx.mlx_array_size(cw);
    const wb: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(cw) orelse return error.HostRead);
    const gb: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(cg) orelse return error.HostRead);
    try testing.expectEqualSlices(u16, wb[0..n], gb[0..n]);
}
