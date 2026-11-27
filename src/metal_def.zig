//! A custom Metal kernel's definition, built into its MLX kernel on first use, and its launch on a stream. Shared by
//! every arch's fused kernels, so it is named by capability. The launch's outputs belong to the caller.
const mlx = @import("mlx.zig");

pub const A = mlx.mlx_array;
pub const Tmpl = struct { name: [*:0]const u8, v: c_int };
pub const TDtype = struct { name: [*:0]const u8, dt: mlx.mlx_dtype };
/// `into`: an array the first output is written over in place (`mlx.kernelApplyInto`, the engine's stream only).
pub const OutSpec = struct { shape: []const c_int, dt: mlx.mlx_dtype, into: ?A = null };
pub const Grid = [3]usize;

/// A kernel's source and signature; the pipeline is built on first use.
pub const Def = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8 = "",
    built: ?mlx.mlx_fast_metal_kernel = null,
};

fn ci(v: usize) c_int {
    return @intCast(v);
}

pub fn kernelNew(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Applies `k` on `s`, or on the GPU stream when `s` is the CPU's (a Metal kernel runs on the GPU even inside an exact
/// CPU-stream test), and fills `outs` with arrays the caller frees; with `into`, the first output is written over it.
pub fn applyKernel(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, inputs: []const A, cfg: mlx.mlx_fast_metal_kernel_config, outs: []A, into: ?A) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    if (into) |dst| {
        try mlx.check(mlx.kernelApplyInto(&ov, k, iv, cfg, s, dst) orelse return error.TensorUnsupported);
    } else if (mlx.streamIsGpu(s)) {
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    } else {
        const gpu = mlx.mlx_default_gpu_stream_new();
        defer _ = mlx.mlx_stream_free(gpu);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, gpu));
    }
    for (outs, 0..) |*o, i| {
        var made = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&made, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(made);
            for (outs[0..i]) |prev| _ = mlx.mlx_array_free(prev);
            return e;
        };
        o.* = made;
    }
}

/// Launches `def` on `s` with `outs`' shapes, the grid and threadgroup, and the template args; `result` gets arrays the
/// caller frees.
pub fn launch(s: mlx.mlx_stream, def: *Def, inputs: []const A, outs: []const OutSpec, grid: Grid, tg: Grid, ints: []const Tmpl, dts: []const TDtype, result: []A) !void {
    const k = try kernelNew(&def.built, def.name, def.ins, def.outs, def.source, def.header);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(grid[0]), ci(grid[1]), ci(grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg[0]), ci(tg[1]), ci(tg[2])));
    for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.v));
    for (dts) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, t.dt));
    try applyKernel(s, k, inputs, cfg, result, outs[0].into);
}

/// Launches `def` on `s` writing its outputs directly into `dsts` (`mlx.kernelApplyInPlace`, the engine's stream only).
/// Returns false when `s` is not the engine's stream or any destination is not an engine tensor, so the caller can fall
/// back to fresh outputs.
pub fn launchInPlace(s: mlx.mlx_stream, def: *Def, inputs: []const A, outs: []const OutSpec, grid: Grid, tg: Grid, ints: []const Tmpl, dts: []const TDtype, dsts: []const A) !bool {
    if (!mlx.engine.isStream(s)) return false;
    for (dsts) |d| if (!mlx.engine.isOurs(d)) return false;
    const k = try kernelNew(&def.built, def.name, def.ins, def.outs, def.source, def.header);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(grid[0]), ci(grid[1]), ci(grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg[0]), ci(tg[1]), ci(tg[2])));
    for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.v));
    for (dts) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, t.dt));
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    try mlx.check(mlx.kernelApplyInPlace(k, iv, cfg, s, dsts) orelse return false);
    return true;
}

/// The fragment layout and the 16-row matrix-unit product the NAX kernels share.
pub const NAX_HEADER =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\inline short dsv41_nax_r(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 4) | ((short(lane) >> 1) & 3)) + (i >> 2) * 8; }
    \\inline short dsv41_nax_c(ushort lane, int i) { const short qid = short(lane) >> 2; return ((qid & 2) | (short(lane) & 1)) * 4 + (i & 3); }
    \\template <int N, int KK, typename AT, typename BT>
    \\inline void dsv41_nax_mma(thread float* c, thread const AT* a, thread const BT* b) {
    \\  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, N, KK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    \\  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\  auto ca = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    \\  auto cb = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    \\  auto cc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ca)>, metal::remove_addrspace_t<decltype(cb)>, float>();
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < KK / 2; ++i) ca[i] = a[i];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < KK * N / 32; ++i) cb[i] = b[i];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < N / 2; ++i) cc[i] = c[i];
    \\  op.run(ca, cb, cc);
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < N / 2; ++i) c[i] = cc[i];
    \\}
    \\
;
