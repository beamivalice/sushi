//! One to eight rows through a BF16 weight widened in the register: `x · Wᵀ`, `x` [T, K] f32, `W` [N, K] as stored,
//! [T, N] f32, each row the bytes it has alone, the weight read once; for the f32 linears a decode row or verify block runs (router, Engram, the indexer's query). A simdgroup sums `TM` output rows over
//! the inputs as MLX's `gemv` (bm4 bn1 sm1 sn32 tm4 tn4) sums each of its rows: 4 inputs a lane a step, then the lanes
//! folded by `simd_shuffle_down`. A row's sum is its own whatever the rows beside it, so `TM` and `NSG` spread the rows
//! over more simdgroups than MLX's tile with the bytes of the f32 product over the widened weight. Where MLX takes its
//! bm1 bn8 tile (`K ≥ 16·N`: the indexer's token key and head weights), the launch is that tile as MLX runs it, the
//! weight read as stored in place of its widened copy.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// Output rows a simdgroup sums, and simdgroups a threadgroup: a router's 512 rows are 256 simdgroups.
const tm = 2;
const nsg = 2;
pub const max_rows: c_int = 16;

const source: [:0]const u8 =
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_N = 32 * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\int out_row = (int(threadgroup_position_in_grid.x) * NSG + int(sg)) * TM;
    \\const device bfloat16_t* mat = w + (size_t)out_row * (size_t)K;
    \\static_assert(K % BLOCK_N == 0, "a row off the gemv tile");
    \\float result[R][TM];
    \\for (int r = 0; r < R; ++r) {
    \\  for (int t = 0; t < TM; ++t) result[r][t] = 0.0f;
    \\}
    \\int bn = int(lane) * TN;
    \\constexpr int n_iter = K / BLOCK_N;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  float inter[TM][TN];
    \\  for (int t = 0; t < TM; ++t) {
    \\    for (int tn = 0; tn < TN; ++tn) inter[t][tn] = static_cast<float>(mat[(size_t)t * (size_t)K + bn + tn]);
    \\  }
    \\  for (int r = 0; r < R; ++r) {
    \\    float coeff[TN];
    \\    const device float* xr = x + (size_t)r * (size_t)K;
    \\    for (int tn = 0; tn < TN; ++tn) coeff[tn] = xr[bn + tn];
    \\    for (int t = 0; t < TM; ++t) {
    \\      for (int tn = 0; tn < TN; ++tn) result[r][t] += inter[t][tn] * coeff[tn];
    \\    }
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\for (int r = 0; r < R; ++r) {
    \\  for (int t = 0; t < TM; ++t) {
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r][t] += simd_shuffle_down(result[r][t], sn);
    \\  }
    \\}
    \\if (lane == 0u) {
    \\  for (int r = 0; r < R; ++r) {
    \\    for (int t = 0; t < TM; ++t) out[(size_t)r * (size_t)N + out_row + t] = result[r][t];
    \\  }
    \\}
;

/// The arm for `K ≥ 16·N` (the indexer's token key and head weights): MLX's `gemv` at bm1 bn8 sm1 sn32 tm4 tn4. A
/// threadgroup's 8 simdgroups each take every eighth block of 128 inputs for its 4 rows (the last partial block read
/// safe, zeros past K, as MLX's `load_safe`), the lanes fold by `simd_shuffle_down`, and simdgroup 0 adds the others'
/// sums in order.
const wide_source: [:0]const u8 =
    \\constexpr int TM = 4, TN = 4, BN = 8, SN = 32;
    \\constexpr int BLOCK_N = BN * SN * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\threadgroup float tgp[BN * R * TM];
    \\float result[R][TM];
    \\for (int r = 0; r < R; r++) {
    \\  for (int tm = 0; tm < TM; tm++) result[r][tm] = 0.0f;
    \\}
    \\float inter[TM][TN];
    \\float v_coeff[TN];
    \\const int out_row = int(threadgroup_position_in_grid.x) * TM;
    \\const device bfloat16_t* mat = w + (size_t)out_row * (size_t)K;
    \\int bn = (SN * int(sg) + int(lane)) * TN;
    \\constexpr int n_iter = K / BLOCK_N;
    \\constexpr int leftover = K - BLOCK_N * n_iter;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  int mat_offset = 0;
    \\  for (int tm = 0; tm < TM; tm++) {
    \\    for (int tn = 0; tn < TN; tn++) inter[tm][tn] = static_cast<float>(mat[mat_offset + bn + tn]);
    \\    mat_offset += K;
    \\  }
    \\  for (int r = 0; r < R; r++) {
    \\    const device float* xr = x + (size_t)r * (size_t)K;
    \\    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = xr[bn + tn];
    \\    for (int tm = 0; tm < TM; tm++) {
    \\      for (int tn = 0; tn < TN; tn++) result[r][tm] += inter[tm][tn] * v_coeff[tn];
    \\    }
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\if (leftover > 0) {
    \\  for (int tm = 0; tm < TM; tm++) {
    \\    for (int tn = 0; tn < TN; tn++) inter[tm][tn] = bn + tn < K ? static_cast<float>(mat[tm * K + bn + tn]) : 0.0f;
    \\  }
    \\  for (int r = 0; r < R; r++) {
    \\    const device float* xr = x + (size_t)r * (size_t)K;
    \\    for (int tn = 0; tn < TN; tn++) v_coeff[tn] = bn + tn < K ? xr[bn + tn] : 0.0f;
    \\    for (int tm = 0; tm < TM; tm++) {
    \\      for (int tn = 0; tn < TN; tn++) result[r][tm] += inter[tm][tn] * v_coeff[tn];
    \\    }
    \\  }
    \\}
    \\for (int r = 0; r < R; r++) {
    \\  for (int tm = 0; tm < TM; tm++) {
    \\    for (ushort sn = SN / 2; sn >= 1; sn >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    \\  }
    \\}
    \\threadgroup float* mine = tgp + sg * (R * TM);
    \\if (lane == 0) {
    \\  for (int r = 0; r < R; r++) {
    \\    for (int tm = 0; tm < TM; tm++) mine[r * TM + tm] = result[r][tm];
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0 && lane == 0) {
    \\  for (int sgn = 1; sgn < BN; sgn++) {
    \\    for (int r = 0; r < R; r++) {
    \\      for (int tm = 0; tm < TM; tm++) result[r][tm] += tgp[sgn * (R * TM) + r * TM + tm];
    \\    }
    \\  }
    \\  for (int r = 0; r < R; r++) {
    \\    for (int tm = 0; tm < TM; tm++) out[(size_t)r * (size_t)N + out_row + tm] = result[r][tm];
    \\  }
    \\}
;

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

fn getKernel(wide: bool) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[@intFromBool(wide)];
    if (slot.*) |k| return k;
    const ins = [_][*:0]const u8{ "w", "x" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(if (wide) "sushi_hy4_row_f32_wide" else "sushi_hy4_row_f32", iv, ov, if (wide) wide_source else source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Whether MLX's `gemv` takes `w` [n, k] at its bm1 bn8 tile (`k >= 16·n`) rather than the four-rows tile.
fn wideTile(n: c_int, k: c_int) bool {
    return k >= 16 * n;
}

/// Whether `rowF32` takes `x` [T ≤ 8, K] f32 against `w` [N, K] BF16 on a GPU stream, at the shapes MLX's `gemv` runs
/// with either tile: more than 64 inputs; under `16·N` inputs a multiple of 128 and under 4096 rows, a multiple of a
/// threadgroup's; from `16·N` on (bm1 bn8) rows a multiple of 4.
pub fn applies(s: mlx.mlx_stream, x: Arr, w: Arr) bool {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[0] < 1 or xs[0] > max_rows or xs[1] != ws[1]) return false;
    const n = ws[0];
    const k = ws[1];
    if (!mlx.streamIsGpu(s) or mlx.mlx_array_dtype(w) != .bfloat16 or mlx.mlx_array_dtype(x) != .float32 or k <= 64) return false;
    if (wideTile(n, k)) return n >= 4 and @rem(n, 4) == 0;
    return n < 4096 and @rem(n, 16) == 0 and @rem(n, tm * nsg) == 0 and @rem(k, 128) == 0;
}

const Key = struct { rows: c_int, n: c_int, k: c_int };
var cfgs: kcfg.Cache(Key) = .{};

fn build(key: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, key.n }, 2, .float32));
    if (wideTile(key.n, key.k)) {
        // MLX's grid: a threadgroup of 8 simdgroups a block of 4 rows.
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(key.n, 4), 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 8, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", key.n));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", key.rows));
        return cfg;
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(key.n, tm * nsg), nsg, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, nsg, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TM", tm));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", nsg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", key.rows));
    return cfg;
}

/// `x · Wᵀ` [T, N] f32 for `applies`' shapes, each row its one-row launch's bytes; `x` contiguous. The caller frees it.
pub fn rowF32(s: mlx.mlx_stream, x: Arr, w: Arr) !Arr {
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const kern = try getKernel(wideTile(ws[0], ws[1]));
    const cfg = try cfgs.get(.{ .rows = xs[0], .n = ws[0], .k = ws[1] }, build);
    const ins = [_]Arr{ w, x };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

/// One row against a random BF16 weight [n, k], against the f32 product over the widened weight, byte for byte.
fn rowCase(s: mlx.mlx_stream, n: c_int, k: c_int, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const w_bits = try gpa.alloc(u16, @intCast(n * k));
    defer gpa.free(w_bits);
    for (w_bits) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64) * 0.05)))) >> 16);
    const x_host = try gpa.alloc(f32, @intCast(k));
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = @floatCast(rnd.floatNorm(f64));
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const w = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ n, k }, 2, .bfloat16));
    const x = try o.own(mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ 1, k }, 2, .float32));
    try testing.expect(applies(s, x, w));
    const want = try o.binary(.mm, x, try o.transpose(try o.cast(w, .float32), &.{ 1, 0 }));
    const got = try o.own(try rowF32(s, x, w));
    try mlx.check(mlx.mlx_array_eval(got));
    try mlx.check(mlx.mlx_array_eval(want));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const len = mlx.mlx_array_size(want);
    const a = (mlx.mlx_array_data_float32(want) orelse return error.MlxArrayDataNull)[0..len];
    const b = (mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull)[0..len];
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

test "hy4 row: one row through a BF16 weight widened in the register is the f32 product's bytes, at the router's, Engram's, the indexer query's, token key's and head weights' shapes and the tiny's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    try rowCase(es, 512, 3072, 1);
    try rowCase(es, 3072, 1536, 2);
    try rowCase(es, 2048, 1024, 3);
    try rowCase(es, 16, 128, 4);
    // MLX's bm1 bn8 tile: the indexer's token key and head weights, and the tiny's head weights (a leftover block).
    try rowCase(es, 128, 3072, 5);
    try rowCase(es, 16, 3072, 6);
    try rowCase(es, 4, 128, 7);
}

fn blockCase(s: mlx.mlx_stream, t: c_int, n: c_int, k: c_int, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const w_bits = try gpa.alloc(u16, @intCast(n * k));
    defer gpa.free(w_bits);
    for (w_bits) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64) * 0.05)))) >> 16);
    const x_host = try gpa.alloc(f32, @intCast(t * k));
    defer gpa.free(x_host);
    for (x_host) |*v| v.* = @floatCast(rnd.floatNorm(f64));
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const w = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ n, k }, 2, .bfloat16));
    const x = try o.own(mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ t, k }, 2, .float32));
    try testing.expect(applies(s, x, w));
    try mlx.check(mlx.mlx_array_eval(w));
    try mlx.check(mlx.mlx_array_eval(x));
    const before = mlx.engine.stats().dispatches;
    const got = try o.own(try rowF32(s, x, w));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(c_int, &[_]c_int{ t, n }, mlx.getShape(got));
    const got_data = (mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull)[0..@intCast(t * n)];
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);
    for (0..@intCast(t)) |r| {
        const xr = try o.own(mlx.mlx_array_new_data(x_host[r * ku ..].ptr, &[_]c_int{ 1, k }, 2, .float32));
        const want = try o.own(try rowF32(s, xr, w));
        try mlx.check(mlx.mlx_array_eval(want));
        const want_data = (mlx.mlx_array_data_float32(want) orelse return error.MlxArrayDataNull)[0..nu];
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want_data), std.mem.sliceAsBytes(got_data[r * nu ..][0..nu]));
    }
}

test "hy4 row: 2, 3, 6 and 8 rows through a BF16 weight in one launch equal each row alone through the one-row kernel, at the router's, Engram's, the indexer's and the tiny's shapes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const shapes = [_][2]c_int{
        .{ 512, 3072 },
        .{ 3072, 1536 },
        .{ 2048, 1024 },
        .{ 128, 3072 },
        .{ 16, 3072 },
        .{ 16, 128 },
        .{ 4, 128 },
    };
    const widest: c_int = @import("hy4_forward.zig").joinedRowsMax();
    for ([_]c_int{ 2, 3, 6, 8, 9, 16 }, 0..) |t, ti| {
        if (t > widest) continue;
        for (shapes, 0..) |sh, si| {
            try blockCase(es, t, sh[0], sh[1], 100 + ti * 16 + si);
        }
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const x17 = try o.zeros(&.{ 17, 3072 }, .float32);
    const w17 = try o.zeros(&.{ 512, 3072 }, .bfloat16);
    try testing.expect(!applies(es, x17, w17));
}

test "hy4 rows: a shared f32 row linear at every width 2..8 gives each row its one-row bytes, rows from different positions" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const widest: usize = @intCast(@import("hy4_forward.zig").joinedRowsMax());
    for (2..widest + 1) |t| try blockCase(es, @intCast(t), 512, 3072, 0x726f + t);
}
