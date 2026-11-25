//! A decode row's (or a verify block's, up to eight rows, each row its own bytes) absorbed MLA products over `kv_b` as stored (BF16), each value widened in the register: the query
//! into the latent (`q_nope · W_k`, MLX's `gemv_t` over the widened weight) and the attended latent out (`u_lat · W_vᵀ`,
//! MLX's `gemv`). Each keeps the thread layout and f32 accumulation order of the MLX kernel the engine picks for the
//! f32 product at these shapes (`gemv.h`: `GEMVTKernel` bm1 bn4 sm8 sn4 tm4 tn4, `GEMVKernel` bm4 bn1 sm1 sn32 tm4
//! tn4), so each output is the bytes of the product over the f32 copy, which the weight then never needs.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// `out[h, n] = Σ_k q[h, k] · W[h·ROWS + k, n]`: `q` [H, NOPE] f32, `W` = kv_b [H·ROWS, LAT] BF16. `gemv_t`'s tile: NSG
/// simdgroups of 8 x 4 lanes, 4 inputs by 4 outputs a thread, NSG · 16 outputs a threadgroup.
const q_source: [:0]const u8 =
    \\constexpr int SM = 8;
    \\constexpr int SN = 4;
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_M = SM * TM;
    \\constexpr int BLOCK_N = NSG * SN * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint h = threadgroup_position_in_grid.z;
    \\const size_t h_off = (size_t)h * (size_t)R;
    \\const device TW* mat = kvb + (size_t)h * (size_t)ROWS * (size_t)LAT;
    \\int thr_m = int(lane) / SN;
    \\int thr_n = int(lane) % SN;
    \\int bm = thr_m * TM;
    \\int out_col = int(threadgroup_position_in_grid.x) * BLOCK_N + (SN * int(sg) + thr_n) * TN;
    \\float result[R][TN];
    \\for (int r = 0; r < R; ++r) {
    \\  for (int tn = 0; tn < TN; ++tn) result[r][tn] = 0.0f;
    \\}
    \\constexpr int n_iter = NOPE / BLOCK_M;
    \\static_assert(NOPE % BLOCK_M == 0 && LAT % BLOCK_N == 0, "an absorbed shape off the gemv_t tile");
    \\for (int i = 0; i < n_iter; ++i) {
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    float inter[TN];
    \\    for (int tn = 0; tn < TN; ++tn) inter[tn] = static_cast<float>(mat[(size_t)(bm + tm) * (size_t)LAT + out_col + tn]);
    \\    for (int r = 0; r < R; ++r) {
    \\      float vc = q[(h_off + (size_t)r) * (size_t)QS + bm + tm];
    \\      for (int tn = 0; tn < TN; ++tn) result[r][tn] += vc * inter[tn];
    \\    }
    \\  }
    \\  bm += BLOCK_M;
    \\}
    \\for (int r = 0; r < R; ++r) {
    \\  for (int tn = 0; tn < TN; ++tn) {
    \\    for (ushort sm = SM / 2; sm >= 1; sm >>= 1) result[r][tn] += simd_shuffle_down(result[r][tn], ushort(SN) * sm);
    \\  }
    \\}
    \\if (thr_m == 0) {
    \\  for (int r = 0; r < R; ++r) {
    \\    device float* o = out + (h_off + (size_t)r) * (size_t)LAT + out_col;
    \\    for (int tn = 0; tn < TN; ++tn) o[tn] = result[r][tn];
    \\  }
    \\}
;

/// `out[h, r] = Σ_k W[h·ROWS + NOPE + r, k] · u[h, k]`: `u` [H, LAT] f32. `gemv`'s tile: 4 simdgroups of 4 rows, 4
/// columns a lane, 16 rows a threadgroup.
const v_main: [:0]const u8 =
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_N = 32 * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint h = threadgroup_position_in_grid.z;
    \\const size_t h_off = (size_t)h * (size_t)R;
    \\int out_row = int(threadgroup_position_in_grid.x) * 4 * TM + int(sg) * TM;
    \\const device TW* mat = kvb + ((size_t)h * (size_t)ROWS + (size_t)NOPE + (size_t)out_row) * (size_t)LAT;
    \\static_assert(LAT % BLOCK_N == 0 && VDIM % (4 * TM) == 0, "an absorbed shape off the gemv tile");
    \\float result[R][TM];
    \\for (int r = 0; r < R; ++r) {
    \\  for (int tm = 0; tm < TM; ++tm) result[r][tm] = 0.0f;
    \\}
    \\int bn = int(lane) * TN;
    \\constexpr int n_iter = LAT / BLOCK_N;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  float inter[TM][TN];
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    for (int tn = 0; tn < TN; ++tn) inter[tm][tn] = static_cast<float>(mat[(size_t)tm * (size_t)LAT + bn + tn]);
    \\  }
    \\  for (int r = 0; r < R; ++r) {
    \\    const device float* vec = u + (size_t)h * (size_t)u_strides[0] + (size_t)r * (size_t)u_strides[1];
    \\    float coeff[TN];
    \\    for (int tn = 0; tn < TN; ++tn) coeff[tn] = vec[bn + tn];
    \\    for (int tm = 0; tm < TM; ++tm) {
    \\      for (int tn = 0; tn < TN; ++tn) result[r][tm] += inter[tm][tn] * coeff[tn];
    \\    }
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\for (int r = 0; r < R; ++r) {
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    \\  }
    \\}
;

const v_out_plain: [:0]const u8 =
    \\if (lane == 0u) {
    \\  for (int r = 0; r < R; ++r) {
    \\    device float* o = out + (h_off + (size_t)r) * (size_t)VDIM + out_row;
    \\    for (int tm = 0; tm < TM; ++tm) o[tm] = result[r][tm];
    \\  }
    \\}
;

const v_out_gated: [:0]const u8 =
    \\if (lane == 0u) {
    \\  device float* o = out + h_off * (size_t)VDIM + out_row;
    \\  for (int tm = 0; tm < TM; ++tm) o[tm] = result[0][tm] * hy4_sigmoid(static_cast<float>(gate[h_off * (size_t)VDIM + out_row + tm]));
    \\}
;

const v_source = v_main ++ v_out_plain;
const v_source_gated = v_main ++ v_out_gated;

const sigmoid_header: [:0]const u8 =
    \\inline float hy4_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
;

var q_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_gated_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8, ensure_row_contiguous: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, ensure_row_contiguous, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The geometry of one layer's `kv_b` [heads · (nope + v), latent].
pub const Geometry = struct { heads: c_int, nope: c_int, v: c_int, latent: c_int };
pub const max_rows: c_int = 8;

/// Whether the products run here: one to `max_rows` rows, a BF16 `kv_b`, a GPU stream, and shapes on the tiles MLX picks for them
/// (`gemv_t` with 8 x 4 lanes, under 8192 inputs; `gemv` with 4 x 4 rows a simdgroup, more than 64 inputs and under 4096
/// rows, fewer inputs than 16 times its rows).
pub fn applies(s: mlx.mlx_stream, kv_b: Arr, g: Geometry, rows: c_int) bool {
    return rows >= 1 and rows <= max_rows and mlx.streamIsGpu(s) and mlx.mlx_array_dtype(kv_b) == .bfloat16 and
        g.nope < 8192 and @rem(g.nope, 32) == 0 and @rem(g.latent, gemvTSimdgroups(g.latent) * 16) == 0 and
        g.latent > 64 and @rem(g.latent, 128) == 0 and g.v < 4096 and @rem(g.v, 16) == 0 and g.latent < 16 * g.v;
}

/// The simdgroups a `gemv_t` threadgroup spans for `out` outputs (MLX's `bn`).
fn gemvTSimdgroups(out: c_int) c_int {
    return if (out >= 2048) 16 else if (out >= 512) 4 else 2;
}

const Key = struct { g: Geometry, rows: c_int, qs: c_int, out_cols: c_int, groups_x: c_int, simdgroups: c_int, gated: bool = false };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const g = k.g;
    if (k.gated) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.heads * k.out_cols }, 2, .float32));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g.heads, k.rows, k.out_cols }, 3, .float32));
    }
    // A block's rows ride the grid's y (one threadgroup a row), so a row's bytes are its one-row launch's.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * k.groups_x, k.simdgroups, g.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, k.simdgroups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", k.simdgroups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TW", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VDIM", g.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "QS", k.qs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", k.rows));
    return cfg;
}

fn run(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const Arr, g: Geometry, out_cols: c_int, groups_x: c_int, simdgroups: c_int, gated: bool) !Arr {
    // The vector's [H, T, width] (a plain row [1, K] is one head's one row).
    const sh = mlx.getShape(inputs[1]);
    const rows: c_int = if (sh.len == 3) sh[1] else 1;
    const cfg = try cfgs.get(.{ .g = g, .rows = rows, .qs = sh[sh.len - 1], .out_cols = out_cols, .groups_x = groups_x, .simdgroups = simdgroups, .gated = gated }, build);
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, in_vec, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    return out;
}

/// The query's first `nope` dims of each of `q`'s [H, T, ≥ nope] f32 rows (contiguous; a rope part past them is not
/// read) into the latent through `kv_b`'s key rows: [H, T, latent] f32, each row its one-row launch's bytes.
pub fn queryLatent(s: mlx.mlx_stream, q_nope: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&q_kernel, "sushi_hy4_absorb_q", &.{ "kvb", "q" }, q_source, "", true);
    const bn = gemvTSimdgroups(g.latent);
    return run(s, k, &.{ kv_b, q_nope }, g, g.latent, @divExact(g.latent, bn * 16), bn, false);
}

/// `u_lat` [H, T, latent] f32 (contiguous) out through `kv_b`'s value rows: [H, T, v] f32.
pub fn valueOut(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&v_kernel, "sushi_hy4_absorb_v", &.{ "kvb", "u" }, v_source, "", false);
    return run(s, k, &.{ kv_b, u_lat }, g, g.v, @divExact(g.v, 16), 4, false);
}

/// Whether `valueOutGated` takes a decode row: `valueOut`'s shapes and a BF16 or f32 gate as wide as the heads' values.
pub fn valueOutGatedApplies(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) bool {
    const us = mlx.getShape(u_lat);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    return applies(s, kv_b, g, 1) and us.len == 3 and us[1] == 1 and gs.len == 2 and gs[0] == 1 and gs[1] == g.heads * g.v and
        (gd == .bfloat16 or gd == .float32);
}

/// `valueOut` with the attention output gate in the same launch: each value times `σ(gate)`, written [1, H·v] as `W_o`
/// reads it, the bytes of `valueOut` then `hy4_chain.gated`.
pub fn valueOutGated(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) !Arr {
    const k = try getKernel(&v_gated_kernel, "sushi_hy4_absorb_v_gated", &.{ "kvb", "u", "gate" }, v_source_gated, sigmoid_header, false);
    return run(s, k, &.{ kv_b, u_lat, gate }, g, g.v, @divExact(g.v, 16), 4, true);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn randomBf16(gpa: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64) * 0.05)))) >> 16);
    return out;
}

fn randomF32(gpa: std.mem.Allocator, rnd: std.Random, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    for (out) |*v| v.* = @floatCast(rnd.floatNorm(f64));
    return out;
}

fn expectSameBytes(got: Arr, want: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(got));
    try mlx.check(mlx.mlx_array_eval(want));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const n = mlx.mlx_array_size(want);
    const a = (mlx.mlx_array_data_float32(want) orelse return error.MlxArrayDataNull)[0..n];
    const b = (mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull)[0..n];
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

/// Both products on random bytes, against the forward's composed f32 products over the widened `kv_b`, byte for byte.
fn parityCase(s: mlx.mlx_stream, g: Geometry, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const rows: usize = @intCast(g.heads * (g.nope + g.v));
    const lat: usize = @intCast(g.latent);
    const h: usize = @intCast(g.heads);
    const w_bits = try randomBf16(gpa, rnd, rows * lat);
    defer gpa.free(w_bits);
    const q_host = try randomF32(gpa, rnd, h * @as(usize, @intCast(g.nope + 64)));
    defer gpa.free(q_host);
    const u_host = try randomF32(gpa, rnd, h * lat);
    defer gpa.free(u_host);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ @intCast(rows), g.latent }, 2, .bfloat16));
    // The query as the forward holds it: [H, 1, nope + rope], its nope part a view.
    const q = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, 1, g.nope + 64 }, 3, .float32));
    const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, 1, g.latent }, 3, .float32));
    const kvb = try o.reshape(try o.cast(kv_b, .float32), &.{ g.heads, g.nope + g.v, g.latent });
    const q_nope = try o.slice(q, 2, 0, g.nope);
    const want_q = try o.binary(.mm, q_nope, try o.slice(kvb, 1, 0, g.nope));
    const want_v = try o.binary(.mm, u, try o.transpose(try o.slice(kvb, 1, g.nope, g.nope + g.v), &.{ 0, 2, 1 }));
    try expectSameBytes(try o.own(try queryLatent(s, try o.contiguous(q_nope), kv_b, g)), want_q);
    try expectSameBytes(try o.own(try valueOut(s, u, kv_b, g)), want_v);
}


test "hy4 absorb: a decode row's query and value products over kv_b as stored are the f32 products' bytes, at the release's sliding and sparse heads and at the tiny's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    try parityCase(es, .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 }, 1);
    try parityCase(es, .{ .heads = 32, .nope = 192, .v = 256, .latent = 512 }, 2);
    try parityCase(es, .{ .heads = 8, .nope = 32, .v = 32, .latent = 128 }, 3);
}

test "hy4 absorb: a block's rows through the products in one launch each are the bytes of each row launched alone, the query read in place past its rope part" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(31);
    const rnd = prng.random();
    const g: Geometry = .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 };
    const t = 6;
    const qk = g.nope + 64;
    const w_bits = try randomBf16(gpa, rnd, @intCast(g.heads * (g.nope + g.v) * g.latent));
    defer gpa.free(w_bits);
    const q_host = try randomF32(gpa, rnd, @intCast(g.heads * t * qk));
    defer gpa.free(q_host);
    const u_host = try randomF32(gpa, rnd, @intCast(g.heads * t * g.latent));
    defer gpa.free(u_host);
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ g.heads * (g.nope + g.v), g.latent }, 2, .bfloat16));
    const q = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, t, qk }, 3, .float32));
    const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, t, g.latent }, 3, .float32));
    const before = mlx.engine.stats().dispatches;
    const q_lat = try o.own(try queryLatent(es, q, kv_b, g));
    const v_out = try o.own(try valueOut(es, u, kv_b, g));
    try mlx.check(mlx.mlx_array_eval(q_lat));
    try mlx.check(mlx.mlx_array_eval(v_out));
    try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
    for (0..t) |r| {
        const lo: c_int = @intCast(r);
        const q_nope = try o.contiguous(try o.slice(try o.slice(q, 1, lo, lo + 1), 2, 0, g.nope));
        try expectSameBytes(try o.contiguous(try o.slice(q_lat, 1, lo, lo + 1)), try o.own(try queryLatent(es, q_nope, kv_b, g)));
        try expectSameBytes(try o.contiguous(try o.slice(v_out, 1, lo, lo + 1)), try o.own(try valueOut(es, try o.contiguous(try o.slice(u, 1, lo, lo + 1)), kv_b, g)));
    }
}

fn blockCase(s: mlx.mlx_stream, g: Geometry, t: c_int, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const qk = g.nope + 64;
    const w_bits = try randomBf16(gpa, rnd, @intCast(g.heads * (g.nope + g.v) * g.latent));
    defer gpa.free(w_bits);
    const q_host = try randomF32(gpa, rnd, @intCast(g.heads * t * qk));
    defer gpa.free(q_host);
    const u_host = try randomF32(gpa, rnd, @intCast(g.heads * t * g.latent));
    defer gpa.free(u_host);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ g.heads * (g.nope + g.v), g.latent }, 2, .bfloat16));
    const q = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, t, qk }, 3, .float32));
    const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, t, g.latent }, 3, .float32));
    try testing.expect(applies(s, kv_b, g, t));
    try mlx.check(mlx.mlx_array_eval(kv_b));
    try mlx.check(mlx.mlx_array_eval(q));
    try mlx.check(mlx.mlx_array_eval(u));
    const before = mlx.engine.stats().dispatches;
    const q_lat = try o.own(try queryLatent(s, q, kv_b, g));
    const v_out = try o.own(try valueOut(s, u, kv_b, g));
    try mlx.check(mlx.mlx_array_eval(q_lat));
    try mlx.check(mlx.mlx_array_eval(v_out));
    try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
    for (0..@intCast(t)) |r| {
        const lo: c_int = @intCast(r);
        const q_nope = try o.contiguous(try o.slice(try o.slice(q, 1, lo, lo + 1), 2, 0, g.nope));
        try expectSameBytes(try o.contiguous(try o.slice(q_lat, 1, lo, lo + 1)), try o.own(try queryLatent(s, q_nope, kv_b, g)));
        try expectSameBytes(try o.contiguous(try o.slice(v_out, 1, lo, lo + 1)), try o.own(try valueOut(s, try o.contiguous(try o.slice(u, 1, lo, lo + 1)), kv_b, g)));
    }
}

test "hy4 absorb: 2, 3, 6 and 8 rows through queryLatent and valueOut in one launch equal each row alone through the one-row kernel, at H 64, 32 and the tiny's shapes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const geometries = [_]Geometry{
        .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 },
        .{ .heads = 32, .nope = 192, .v = 256, .latent = 512 },
        .{ .heads = 8, .nope = 32, .v = 32, .latent = 128 },
    };
    for ([_]c_int{ 2, 3, 6, 8 }, 0..) |t, ti| {
        for (geometries, 0..) |g, gi| {
            try blockCase(es, g, t, 200 + ti * 8 + gi);
        }
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const g = geometries[0];
    const kv_b = try o.zeros(&.{ g.heads * (g.nope + g.v), g.latent }, .bfloat16);
    try testing.expect(!applies(es, kv_b, g, 9));
}

test "hy4 absorb: a decode row's value product gated in the same launch is the bytes of valueOut then the gate's launch, BF16 and f32 gate rows, at the release's 64 and 32 heads" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const chain = @import("hy4_chain.zig");
    const gpa = testing.allocator;
    for ([_]c_int{ 64, 32 }, 0..) |heads, ci| {
        const g: Geometry = .{ .heads = heads, .nope = 192, .v = 256, .latent = 512 };
        var prng = std.Random.DefaultPrng.init(0x9a7e + ci);
        const rnd = prng.random();
        const rows: usize = @intCast(g.heads * (g.nope + g.v));
        const w_bits = try randomBf16(gpa, rnd, rows * 512);
        defer gpa.free(w_bits);
        const u_host = try randomF32(gpa, rnd, @as(usize, @intCast(heads)) * 512);
        defer gpa.free(u_host);
        const gate_host = try randomF32(gpa, rnd, @as(usize, @intCast(heads)) * 256);
        defer gpa.free(gate_host);
        for (gate_host[0..4], [_]f32{ 0, -0.0, 120, -120 }) |*d, v| d.* = v;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ @intCast(rows), g.latent }, 2, .bfloat16));
        const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, 1, g.latent }, 3, .float32));
        const gate32 = try o.own(mlx.mlx_array_new_data(gate_host.ptr, &[_]c_int{ 1, g.heads * g.v }, 2, .float32));
        for ([_]Arr{ gate32, try o.cast(gate32, .bfloat16) }) |gate| {
            try mlx.check(mlx.mlx_array_eval(gate));
            const value = try o.own(try valueOut(es, u, kv_b, g));
            const want = try o.own(try chain.gated(es, value, gate));
            try testing.expect(valueOutGatedApplies(es, u, kv_b, g, gate));
            const before = mlx.engine.stats().dispatches;
            const got = try o.own(try valueOutGated(es, u, kv_b, g, gate));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            try expectSameBytes(got, want);
        }
    }
}
