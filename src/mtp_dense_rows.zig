const std = @import("std");
const mlx = @import("mlx.zig");

/// Test hook: false takes MLX's own matmul.
pub var override: ?bool = null;

pub fn enabled() bool {
    return override orelse true;
}

// MLX v0.32.2 GEMV (BM4 BN1 SM1 SN32 TM4 TN4), with independent activation rows.
const ROUTER_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const uint out_row = threadgroup_position_in_grid.x * 16 + sg * 4;
    \\float result[4] = {};
    \\int bn = int(lane) * 4;
    \\for (int block = 0; block < 20; ++block) {
    \\  float v[4];
    \\  for (int t = 0; t < 4; ++t) v[t] = float(x[row * x_strides[0] + (bn + t) * x_strides[1]]);
    \\  for (int m = 0; m < 4; ++m) {
    \\    for (int t = 0; t < 4; ++t) result[m] += float(w[(out_row + m) * 2560 + bn + t]) * v[t];
    \\  }
    \\  bn += 128;
    \\}
    \\for (int m = 0; m < 4; ++m) {
    \\  for (ushort offset = 16; offset >= 1; offset >>= 1) result[m] += simd_shuffle_down(result[m], offset);
    \\  if (lane == 0) y[row * 512 + out_row + m] = bfloat(result[m]);
    \\}
;

// MLX v0.32.2 dot_product BF16 (it32 tg512 sg16); the one-block sum is unchanged.
const GATE_SOURCE =
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint row = threadgroup_position_in_grid.y;
    \\const int start = int(sg * 32) * 32 + int(lane) * 8;
    \\float4 c = 0.0f;
    \\for (int i = 0; i < 32; i += 8) {
    \\  const int idx = start + i * 32;
    \\  if (idx + 8 <= 2560) {
    \\    for (int j = 0; j < 8; j += 4) {
    \\      float4 a, b;
    \\      for (int t = 0; t < 4; ++t) {
    \\        a[t] = float(x[row * x_strides[0] + (idx + j + t) * x_strides[1]]);
    \\        b[t] = float(w[idx + j + t]);
    \\      }
    \\      c += a * b;
    \\    }
    \\  }
    \\}
    \\threadgroup float smem[16];
    \\float value = c[0] + c[1] + c[2] + c[3];
    \\value = simd_sum(value);
    \\if (lane == 0) smem[sg] = value;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (tid < 16) {
    \\  value = simd_sum(smem[tid]);
    \\  if (tid == 0) y[row] = bfloat(value);
    \\}
;

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var configurations: [2][32]?mlx.mlx_fast_metal_kernel_config = @splat(@splat(null));
pub var calls: [2]usize = .{ 0, 0 };
var logged: [2]bool = .{ false, false };

pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or x.ctx == null or w.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len < 2 or xs.len > 8 or xs[xs.len - 1] != 2560 or ws.len != 2 or ws[0] != 2560) return null;
    const n = ws[1];
    if (n != 512 and n != 1) return null;
    const strides = mlx.mlx_array_strides(w);
    if (strides[0] != 1 or strides[1] != 2560) return null;
    const rows = mlx.mlx_array_size(x) / 2560;
    if (rows < 1 or rows > 32) return null;
    const which: usize = if (n == 512) 0 else 1;
    if (kernels[which] == null) {
        const names = [_][*:0]const u8{ "x", "w" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&names, names.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        // Preserve the weight transpose view; making every input contiguous copies the router.
        const kernel = mlx.mlx_fast_metal_kernel_new(
            if (which == 0) "mtp_dense_router_rows" else "mtp_dense_gate_rows",
            iv,
            ov,
            if (which == 0) ROUTER_SOURCE else GATE_SOURCE,
            "",
            false,
            false,
        );
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        kernels[which] = kernel;
    }
    if (configurations[which][rows - 1] == null) {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ @intCast(rows), n }, 2, .bfloat16));
        const threads: c_int = if (which == 0) 128 else 512;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, if (which == 0) 32 * threads else threads, @intCast(rows), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
        configurations[which][rows - 1] = cfg;
    }
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &.{ @intCast(rows), 2560 }, 2, s));
    const iv = mlx.mlx_vector_array_new_data(&.{ flat, w }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernels[which].?, iv, configurations[which][rows - 1].?, s));
    calls[which] += 1;
    if (!@import("builtin").is_test and !logged[which]) {
        logged[which] = true;
        @import("log.zig").info("[mtp-dense-rows] {s} engaged K=2560 N={d} rows={d}\n", .{ if (which == 0) "router" else "gate", n, rows });
    }
    var result = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..xs.len], xs);
    shape[xs.len - 1] = n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, result, &shape, xs.len, s));
    return out;
}

test "dense rows, serial decode included, preserve MLX's bits on the transposed router and singleton gate" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd3e5e);
    const rng = prng.random();
    const k: c_int = 2560;
    for ([_]c_int{ 512, 1 }) |n| {
        const wh = try a.alloc(u16, @intCast(n * k));
        defer a.free(wh);
        for (wh) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * 0.2)) >> 16);
        const raw = mlx.mlx_array_new_data(wh.ptr, &.{ n, k }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(raw);
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_transpose_axes(&w, raw, &.{ 1, 0 }, 2, s));
        try mlx.check(mlx.mlx_array_eval(w));
        try std.testing.expectEqual(@as(usize, 1), mlx.mlx_array_strides(w)[0]);
        try std.testing.expectEqual(@as(usize, @intCast(k)), mlx.mlx_array_strides(w)[1]);
        for ([_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32 }) |rows| {
            const xh = try a.alloc(u16, @intCast(rows * k));
            defer a.free(xh);
            for (xh, 0..) |*v, i| {
                const scale: f32 = if (i % 3 == 0) 32.0 else 0.04;
                v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * scale)) >> 16);
            }
            const x = mlx.mlx_array_new_data(xh.ptr, &.{ 1, rows, k }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const candidate = (try matmul(s, x, w)) orelse return error.DenseRowsDeclined;
            defer _ = mlx.mlx_array_free(candidate);
            try mlx.check(mlx.mlx_array_eval(candidate));
            const cp = mlx.mlx_array_data_bfloat16(candidate) orelse return error.Unreadable;
            for (0..@intCast(rows)) |row| {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                var ref = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ref);
                try mlx.check(mlx.mlx_slice(&xr, x, &.{ 0, @intCast(row), 0 }, 3, &.{ 1, @intCast(row + 1), k }, 3, &.{ 1, 1, 1 }, 3, s));
                try mlx.check(mlx.mlx_matmul(&ref, xr, w, s));
                try mlx.check(mlx.mlx_array_eval(ref));
                const rp = mlx.mlx_array_data_bfloat16(ref) orelse return error.Unreadable;
                for (0..@intCast(n)) |col| {
                    const actual = cp[row * @as(usize, @intCast(n)) + col];
                    if (rp[col] != actual) {
                        std.debug.print("dense rows N={d} S={d} row={d} col={d}: serial={x} candidate={x}\n", .{ n, rows, row, col, rp[col], actual });
                        return error.DenseRowsNotIdentical;
                    }
                }
            }
        }
    }
}
