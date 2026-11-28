//! One f32 row through a BF16 weight as `fwd.linear`'s BF16 product gives it, in one launch: the row rounded to BF16 in
//! the register, MLX's `gemv` over the BF16 operands (`gemv.h`, `GEMVKernel` bm4 bn1 sm1 sn32 tm4 tn4: 4 simdgroups of
//! 4 rows, 4 columns a lane, f32 sums, `simd_shuffle_down` over 32 lanes), the sum rounded to BF16 and widened. So a
//! row is the bytes of the cast in, MLX's BF16 `gemv` and the cast out (the latent projection `W_kva`, stored BF16).
//! A verify block's rows share one launch: the weight is read once, and each row keeps its one-row sums.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

pub const max_rows: c_int = 16;

pub const source: [:0]const u8 =
    \\using B4 = metal::vec<bfloat16_t, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\using X4 = metal::vec<decltype(x[0] + x[0]), 4>;
    \\constexpr int TM = 2;
    \\constexpr int NSG = 2;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_N = 32 * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\int out_row = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * TM;
    \\const device bfloat16_t* mat = w + (size_t)out_row * (size_t)K;
    \\#ifndef SUSHI_METALLIB
    \\static_assert(K % BLOCK_N == 0, "a row off the gemv tile");
    \\#endif
    \\float result[R][TM];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < R; ++r) {
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) result[r][tm] = 0.0f;
    \\}
    \\int bn = int(lane) * TN;
    \\const int n_iter = K / BLOCK_N;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  F4 cv[R];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < R; ++r) {
    \\    const X4 xv = *(const device X4*)(x + (size_t)r * (size_t)K + (size_t)bn);
    \\    cv[r] = F4(
    \\      static_cast<float>(static_cast<bfloat16_t>(xv[0])),
    \\      static_cast<float>(static_cast<bfloat16_t>(xv[1])),
    \\      static_cast<float>(static_cast<bfloat16_t>(xv[2])),
    \\      static_cast<float>(static_cast<bfloat16_t>(xv[3])));
    \\  }
    \\  const B4 m0 = *(const device B4*)(mat + bn);
    \\  const B4 m1 = *(const device B4*)(mat + (size_t)K + bn);
    \\  {
    \\    const float w0 = static_cast<float>(m0[0]), w1 = static_cast<float>(m0[1]);
    \\    const float w2 = static_cast<float>(m0[2]), w3 = static_cast<float>(m0[3]);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < R; ++r) {
    \\      result[r][0] += w0 * cv[r][0];
    \\      result[r][0] += w1 * cv[r][1];
    \\      result[r][0] += w2 * cv[r][2];
    \\      result[r][0] += w3 * cv[r][3];
    \\    }
    \\  }
    \\  {
    \\    const float w0 = static_cast<float>(m1[0]), w1 = static_cast<float>(m1[1]);
    \\    const float w2 = static_cast<float>(m1[2]), w3 = static_cast<float>(m1[3]);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < R; ++r) {
    \\      result[r][1] += w0 * cv[r][0];
    \\      result[r][1] += w1 * cv[r][1];
    \\      result[r][1] += w2 * cv[r][2];
    \\      result[r][1] += w3 * cv[r][3];
    \\    }
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < R; ++r) {
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    #pragma clang loop unroll(full)
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    \\  }
    \\}
    \\if (lane == 0u) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < R; ++r) {
    \\    #pragma clang loop unroll(full)
    \\    for (int tm = 0; tm < TM; ++tm) out[(size_t)r * (size_t)N + out_row + tm] = static_cast<OT>(static_cast<bfloat16_t>(result[r][tm]));
    \\  }
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `rowsThru` takes `x` [R, K] f32 (1 to `max_rows` rows) against `w` [N, K] BF16: a GPU stream and the shapes
/// MLX's `gemv` runs with this tile (as `hy4_row.applies`).
pub fn applies(s: mlx.mlx_stream, x: Arr, w: Arr) bool {
    return mlx.mlx_array_dtype(x) == .float32 and appliesStored(s, x, w);
}

/// Whether `rowsStored` takes `x` [R, K] f32 or BF16 (1 to `max_rows` rows) against `w` [N, K] BF16.
pub fn appliesStored(s: mlx.mlx_stream, x: Arr, w: Arr) bool {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[0] < 1 or xs[0] > max_rows or xs[1] != ws[1]) return false;
    const xdt = mlx.mlx_array_dtype(x);
    const n = ws[0];
    const k = ws[1];
    return mlx.streamIsGpu(s) and (xdt == .float32 or xdt == .bfloat16) and mlx.mlx_array_dtype(w) == .bfloat16 and
        n < 4096 and @rem(n, 16) == 0 and k > 64 and @rem(k, 128) == 0 and k < 16 * n;
}

const Key = struct { rows: c_int, n: c_int, k: c_int, out_dt: mlx.mlx_dtype };
var cfgs: kcfg.Cache(Key) = .{};

fn build(key: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, key.n }, 2, key.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(key.n, 4), 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", key.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", key.out_dt));
    return cfg;
}

fn rowsAs(s: mlx.mlx_stream, x: Arr, w: Arr, out_dt: mlx.mlx_dtype) !Arr {
    if (kernel == null) {
        const ins = [_][*:0]const u8{ "w", "x" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_gemv_thru", iv, ov, source, "", true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        kernel = made;
    }
    const ws = mlx.getShape(w);
    const cfg = try cfgs.get(.{ .rows = mlx.getShape(x)[0], .n = ws[0], .k = ws[1], .out_dt = out_dt }, build);
    const ins = [_]Arr{ w, x };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel.?, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `cast(cast(x, bf16) · Wᵀ, f32)` for each of `x`'s rows, [R, N]. The caller frees the result.
pub fn rowsThru(s: mlx.mlx_stream, x: Arr, w: Arr) !Arr {
    return rowsAs(s, x, w, .float32);
}

/// `cast(x, bf16) · Wᵀ` as BF16 `[R, N]` for `x` f32 or BF16. The caller frees the result.
pub fn rowsStored(s: mlx.mlx_stream, x: Arr, w: Arr) !Arr {
    return rowsAs(s, x, w, .bfloat16);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

test "hy4 gemv: one row through a BF16 weight, rounded in and out in the launch, is the bytes of the casts around MLX's BF16 gemv, at the latent projection's 576 x 3,072, the release's other row shapes and the tiny's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_][2]c_int{ .{ 576, 3072 }, .{ 512, 3072 }, .{ 3072, 1536 }, .{ 144, 128 } }, 0..) |sh, i| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const n: usize = @intCast(sh[0] * sh[1]);
        const wv = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(wv);
        const xv = try testing.allocator.alloc(f32, @intCast(sh[1]));
        defer testing.allocator.free(xv);
        var r = std.Random.DefaultPrng.init(i + 7);
        for (wv) |*d| d.* = @floatCast(r.random().floatNorm(f64) * 0.05);
        for (xv) |*d| d.* = @floatCast(r.random().floatNorm(f64) * 2);
        const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &sh, 2, .float32)), .bfloat16);
        const x = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, sh[1] }, 2, .float32));
        const want_a = try o.contiguous(try o.cast(try o.binary(.mm, try o.cast(x, .bfloat16), try o.transpose(w, &.{ 1, 0 })), .float32));
        try mlx.check(mlx.mlx_array_eval(want_a));
        const want = std.mem.sliceAsBytes(mlx.mlx_array_data_float32(want_a).?[0..@intCast(sh[0])]);
        try testing.expect(applies(es, x, w));
        try mlx.check(mlx.mlx_array_eval(w));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try rowsThru(es, x, w));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(got).?[0..@intCast(sh[0])]));
    }
}

test "hy4 gemv: rowsStored matches per-row BF16 gemv byte for byte for f32 and bf16 inputs at R=1 and R=6" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const sh = [2]c_int{ 576, 3072 };
    for ([_]c_int{ 1, 6 }) |rows| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const wv = try testing.allocator.alloc(f32, @intCast(sh[0] * sh[1]));
        defer testing.allocator.free(wv);
        const xv = try testing.allocator.alloc(f32, @intCast(rows * sh[1]));
        defer testing.allocator.free(xv);
        var r = std.Random.DefaultPrng.init(@intCast(100 + rows));
        for (wv) |*d| d.* = @floatCast(r.random().floatNorm(f64) * 0.05);
        for (xv) |*d| d.* = @floatCast(r.random().floatNorm(f64) * 2);
        const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &sh, 2, .float32)), .bfloat16);
        const x_f32 = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ rows, sh[1] }, 2, .float32));
        const x_bf16 = try o.cast(x_f32, .bfloat16);
        const wt = try o.transpose(w, &.{ 1, 0 });
        var want_rows: [6]Arr = undefined;
        for (0..@as(usize, @intCast(rows))) |ri| {
            const r_i: c_int = @intCast(ri);
            want_rows[ri] = try o.binary(.mm, try o.slice(x_bf16, 0, r_i, r_i + 1), wt);
        }
        const want_a = try o.contiguous(try o.cast(try o.concat(want_rows[0..@intCast(rows)], 0), .float32));
        try mlx.check(mlx.mlx_array_eval(want_a));
        const n_out: usize = @intCast(rows * sh[0]);
        const want = std.mem.sliceAsBytes(mlx.mlx_array_data_float32(want_a).?[0..n_out]);
        try mlx.check(mlx.mlx_array_eval(w));
        try mlx.check(mlx.mlx_array_eval(x_f32));
        try mlx.check(mlx.mlx_array_eval(x_bf16));
        for ([_]Arr{ x_f32, x_bf16 }) |x_in| {
            try testing.expect(appliesStored(es, x_in, w));
            const before = mlx.engine.stats().dispatches;
            const got_bf = try o.own(try rowsStored(es, x_in, w));
            try mlx.check(mlx.mlx_array_eval(got_bf));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got_bf));
            const got_f32 = try o.contiguous(try o.cast(got_bf, .float32));
            try mlx.check(mlx.mlx_array_eval(got_f32));
            try testing.expectEqualSlices(u8, want, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(got_f32).?[0..n_out]));
        }
    }
}
