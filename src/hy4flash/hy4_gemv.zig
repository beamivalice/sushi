//! One f32 row through a BF16 weight as `fwd.linear`'s BF16 product gives it, in one launch: the row rounded to BF16 in
//! the register, MLX's `gemv` over the BF16 operands (`gemv.h`, `GEMVKernel` bm4 bn1 sm1 sn32 tm4 tn4: 4 simdgroups of
//! 4 rows, 4 columns a lane, f32 sums, `simd_shuffle_down` over 32 lanes), the sum rounded to BF16 and widened. So a
//! row is the bytes of the cast in, MLX's BF16 `gemv` and the cast out (the latent projection `W_kva`, stored BF16).
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

const source: [:0]const u8 =
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_N = 32 * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\int out_row = int(threadgroup_position_in_grid.x) * 4 * TM + int(sg) * TM;
    \\const device bfloat16_t* mat = w + (size_t)out_row * (size_t)K;
    \\static_assert(K % BLOCK_N == 0, "a row off the gemv tile");
    \\float result[TM] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\int bn = int(lane) * TN;
    \\constexpr int n_iter = K / BLOCK_N;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  float coeff[TN];
    \\  for (int tn = 0; tn < TN; ++tn) coeff[tn] = static_cast<float>(static_cast<bfloat16_t>(x[bn + tn]));
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    float inter[TN];
    \\    for (int tn = 0; tn < TN; ++tn) inter[tn] = static_cast<float>(mat[(size_t)tm * (size_t)K + bn + tn]);
    \\    for (int tn = 0; tn < TN; ++tn) result[tm] += inter[tn] * coeff[tn];
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\for (int tm = 0; tm < TM; ++tm) {
    \\  for (ushort sn = 16; sn >= 1; sn >>= 1) result[tm] += simd_shuffle_down(result[tm], sn);
    \\}
    \\if (lane == 0u) {
    \\  for (int tm = 0; tm < TM; ++tm) out[out_row + tm] = static_cast<float>(static_cast<bfloat16_t>(result[tm]));
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `rowThru` takes `x` [1, K] f32 against `w` [N, K] BF16: a GPU stream and the shapes MLX's `gemv` runs with
/// this tile (as `hy4_row.applies`).
pub fn applies(s: mlx.mlx_stream, x: Arr, w: Arr) bool {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[0] != 1 or xs[1] != ws[1]) return false;
    const n = ws[0];
    const k = ws[1];
    return mlx.streamIsGpu(s) and mlx.mlx_array_dtype(x) == .float32 and mlx.mlx_array_dtype(w) == .bfloat16 and
        n < 4096 and @rem(n, 16) == 0 and k > 64 and @rem(k, 128) == 0 and k < 16 * n;
}

const Key = struct { n: c_int, k: c_int };
var cfgs: kcfg.Cache(Key) = .{};

fn build(key: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, key.n }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(key.n, 16), 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
    return cfg;
}

/// `cast(cast(x, bf16) · Wᵀ, f32)` for one row, [1, N]. The caller frees the result.
pub fn rowThru(s: mlx.mlx_stream, x: Arr, w: Arr) !Arr {
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
    const cfg = try cfgs.get(.{ .n = ws[0], .k = ws[1] }, build);
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
        const got = try o.own(try rowThru(es, x, w));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(got).?[0..@intCast(sh[0])]));
    }
}
