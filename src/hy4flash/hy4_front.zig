//! The `kv_a` rows' tail in one launch, a threadgroup a row: the latent `c` rms-normed (`hy4_norm.rows`' arithmetic, MLX's
//! `rms_single_row` over the row's latent part), then `k_pe` rotated (`hy4_rope.tail`'s arithmetic, MLX's rope), in
//! place of `fwd.projectAs`' norm and rope launches. Both read `kv_a` alone, so the launch keeps their dependency
//! level; the query's rotation, which waits for `q_b`, stays a launch of its own.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

// No contract pragma: the norm and the rotations are written as `hy4_norm` and `hy4_rope` write them, under the same
// default.
const source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\{
    \\  constexpr uint axis_size = uint(LAT);
    \\  float acc = 0;
    \\  float thread_x[N_READS];
    \\  const device T* xr = ckv + (size_t)gid * (size_t)CKW;
    \\  for (int i = 0; i < N_READS; i++) {
    \\    uint c = lid * N_READS + i;
    \\    float v = 0;
    \\    if (c < axis_size) {
    \\      v = static_cast<float>(xr[c]);
    \\    }
    \\    thread_x[i] = v;
    \\    acc += thread_x[i] * thread_x[i];
    \\  }
    \\  acc = simd_sum(acc);
    \\  if (simd_group_id == 0) {
    \\    local_sums[simd_lane_id] = 0;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (simd_lane_id == 0) {
    \\    local_sums[simd_group_id] = acc;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (simd_group_id == 0) {
    \\    acc = simd_sum(local_sums[simd_lane_id]);
    \\    if (simd_lane_id == 0) {
    \\      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  device OT* o = c_out + (size_t)gid * (size_t)LAT;
    \\  for (int i = 0; i < N_READS; i++) {
    \\    uint c = lid * N_READS + i;
    \\    if (c < axis_size) {
    \\      float v = static_cast<float>(w[c]) * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    \\      o[c] = static_cast<OT>(v);
    \\    }
    \\  }
    \\  if (lid < uint(HALF)) {
    \\    uint i = lid;
    \\    uint t = gid;
    \\    const device T* row = xr + LAT;
    \\    device bfloat16_t* dst = k_out + (size_t)t * (size_t)(2 * HALF);
    \\    float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\    float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\    float L = as_type<float>(SCALE) * static_cast<float>(PER_ROW ? pos[t] : (pos[0] + int(t)));
    \\    float theta = L * inv_freq;
    \\    float costheta = metal::fast::cos(theta);
    \\    float sintheta = metal::fast::sin(theta);
    \\    float x1 = static_cast<float>(row[2 * i]);
    \\    float x2 = static_cast<float>(row[2 * i + 1]);
    \\    float rx1 = x1 * costheta - x2 * sintheta;
    \\    float rx2 = x1 * sintheta + x2 * costheta;
    \\    dst[2 * i] = static_cast<bfloat16_t>(rx1);
    \\    dst[2 * i + 1] = static_cast<bfloat16_t>(rx2);
    \\  }
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "ckv", "w", "pos" };
    const outs = [_][*:0]const u8{ "c_out", "k_out" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_front", iv, ov, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The norm's threads (`hy4_norm`'s: four columns a thread, whole simdgroups) for a latent of `lat`.
fn normGroup(lat: c_int) c_int {
    return 32 * @divFloor(@divFloor(lat + 3, 4) + 31, 32);
}

/// Whether `run` takes the `kv_a` rows `ckv` [T, latent + rope] f32 with the norm weight `w` [latent]:
/// `hy4_norm.rows`' and `hy4_rope.tail`'s shapes, a rope pair a norm thread.
pub fn applies(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int) bool {
    const cs = mlx.getShape(ckv);
    const cd = mlx.mlx_array_dtype(ckv);
    const wd = mlx.mlx_array_dtype(w);
    if (cs.len != 2 or mlx.getShape(w).len != 1) return false;
    return mlx.streamIsGpu(s) and cs[1] == latent + rope and latent > 0 and latent <= 4096 and
        mlx.getShape(w)[0] == latent and @rem(rope, 2) == 0 and rope > 0 and @divExact(rope, 2) <= normGroup(latent) and
        (cd == .float32 or cd == .bfloat16) and (wd == .float32 or wd == .bfloat16);
}

const Key = struct { t: c_int, lat: c_int, half: c_int, in_dt: mlx.mlx_dtype, c_dt: mlx.mlx_dtype, w_dt: mlx.mlx_dtype, base: c_int, eps: c_int, per_row: bool = false };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const tg = normGroup(k.lat);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, k.lat }, 2, k.c_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, 2 * k.half }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.t * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    inline for (.{ "LAT", "CKW", "HALF", "BASE", "EPS", "SCALE" }, .{ k.lat, k.lat + 2 * k.half, k.half, k.base, k.eps, kcfg.bits(1.0) }) |n, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", k.in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", k.c_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "PER_ROW", k.per_row));
    return cfg;
}

pub const Front = struct { c: Arr, k_pe: Arr };

/// `applies`' operands at positions `pos0..`: `c` [T, latent] = `RMSNorm_w(ckv[:, :latent])` (ε `eps`) as `c_dt`,
/// and `k_pe` [T, rope] BF16 = `ckv[:, latent:]` rotated (θ `theta`). The caller frees both.
pub fn run(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int, theta: f32, eps: f32, pos0: c_int, c_dt: mlx.mlx_dtype) !Front {
    const pv = [_]i32{pos0};
    return runWith(s, ckv, w, latent, rope, theta, eps, &pv, c_dt, false);
}

/// `run` with row `t` rotated at its own `positions[t]`.
pub fn runPerRow(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int, theta: f32, eps: f32, positions: []const c_int, c_dt: mlx.mlx_dtype) !Front {
    return runWith(s, ckv, w, latent, rope, theta, eps, positions, c_dt, true);
}

fn runWith(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int, theta: f32, eps: f32, positions: []const c_int, c_dt: mlx.mlx_dtype, per_row: bool) !Front {
    const t = mlx.getShape(ckv)[0];
    if (per_row and positions.len != @as(usize, @intCast(t))) return error.Hy4RowsPositionsLen;
    const cfg = try cfgs.get(.{ .t = t, .lat = latent, .half = @divExact(rope, 2), .in_dt = mlx.mlx_array_dtype(ckv), .c_dt = c_dt, .w_dt = mlx.mlx_array_dtype(w), .base = kcfg.bits(std.math.log2(theta)), .eps = kcfg.bits(eps), .per_row = per_row }, build);
    const pos = try mlx.newData(positions.ptr, &.{@intCast(positions.len)}, .int32);
    defer _ = mlx.mlx_array_free(pos);
    const ins = [_]Arr{ ckv, w, pos };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(), iv, cfg, s));
    var out: [2]Arr = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for (&out, 0..) |*x, i| try mlx.check(mlx.mlx_vector_array_get(x, ov, i));
    return .{ .c = out[0], .k_pe = out[1] };
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn randomArr(o: *Ops, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, sc: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * sc);
    return o.cast(try o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32)), dt);
}

fn bytesOf(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

test "hy4 front: the latent's norm and the key's rotation in one launch give the bytes of the norm and the rope launch, at the release's sliding and sparse shapes and the tiny's, one row and three, the latent BF16 and f32" {
    try mlx.requireGpu();
    const hy4_norm = @import("hy4_norm.zig");
    const hy4_rope = @import("hy4_rope.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { lat: c_int, rope: c_int, theta: f32 };
    const cases = [_]Case{
        .{ .lat = 512, .rope = 64, .theta = 1e4 },
        .{ .lat = 512, .rope = 64, .theta = 1e7 },
        .{ .lat = 128, .rope = 16, .theta = 1e4 },
    };
    for (cases, 0..) |c, ci| for ([_]c_int{ 1, 3 }) |t| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |c_dt| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 13 * ci + @as(u64, @intCast(t));
        const ckv = try o.contiguous(try randomArr(&o, &.{ t, c.lat + c.rope }, .float32, seed, 1.5));
        const w = try o.contiguous(try randomArr(&o, &.{c.lat}, .bfloat16, seed + 1, 0.3));
        const pos0: c_int = 4093 + @as(c_int, @intCast(ci));
        // The chain projectAs runs.
        const wc = try o.own(try hy4_norm.rows(es, try o.slice(ckv, 1, 0, c.lat), null, w, 1e-5, c_dt));
        const k_pe = try o.reshape(try o.slice(ckv, 1, c.lat, c.lat + c.rope), &.{ 1, t, c.rope });
        const wk = try o.own(try hy4_rope.tail(es, k_pe, c.rope, c.theta, pos0, 1, .bfloat16));
        const want = [_][]const u8{ try bytesOf(&o, wc), try bytesOf(&o, wk) };
        try testing.expect(applies(es, ckv, w, c.lat, c.rope));
        const before = mlx.engine.stats().dispatches;
        const got = try run(es, ckv, w, c.lat, c.rope, c.theta, 1e-5, pos0, c_dt);
        const g = [_]Arr{ try o.own(got.c), try o.own(got.k_pe) };
        try mlx.check(mlx.mlx_array_eval(g[1]));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqual(c_dt, mlx.mlx_array_dtype(g[0]));
        for (want, g) |wb, ga| try testing.expectEqualSlices(u8, wb, try bytesOf(&o, ga));
    };
}

test "hy4 front: bf16 ckv widens in register and matches f32 ckv byte for byte in one launch" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const ckv_bf = try o.contiguous(try randomArr(&o, &.{ 16, 576 }, .bfloat16, 91, 1.5));
    const ckv_f32 = try o.contiguous(try o.cast(ckv_bf, .float32));
    const w = try o.contiguous(try randomArr(&o, &.{512}, .bfloat16, 92, 0.3));
    const want = try run(es, ckv_f32, w, 512, 64, 1e4, 1e-5, 4096, .bfloat16);
    const wc = try bytesOf(&o, try o.own(want.c));
    const wk = try bytesOf(&o, try o.own(want.k_pe));
    try testing.expect(applies(es, ckv_bf, w, 512, 64));
    try mlx.check(mlx.mlx_array_eval(ckv_bf));
    const before = mlx.engine.stats().dispatches;
    const got = try run(es, ckv_bf, w, 512, 64, 1e4, 1e-5, 4096, .bfloat16);
    const gc = try o.own(got.c);
    const gk = try o.own(got.k_pe);
    try mlx.check(mlx.mlx_array_eval(gk));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(u8, wc, try bytesOf(&o, gc));
    try testing.expectEqualSlices(u8, wk, try bytesOf(&o, gk));
}

test "hy4 front: per-row positions match per-row single launches byte for byte" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const positions = [_]c_int{ 0, 511, 512, 4097, 2048, 100_003, 7, 65_535 };
    const t: c_int = positions.len;
    const ckv = try o.contiguous(try randomArr(&o, &.{ t, 576 }, .float32, 314, 1.5));
    const w = try o.contiguous(try randomArr(&o, &.{512}, .bfloat16, 315, 0.3));
    try mlx.check(mlx.mlx_array_eval(ckv));
    try mlx.check(mlx.mlx_array_eval(w));
    const before = mlx.engine.stats().dispatches;
    const got = try runPerRow(es, ckv, w, 512, 64, 1e4, 1e-5, &positions, .bfloat16);
    const gc = try o.own(got.c);
    const gk = try o.own(got.k_pe);
    try mlx.check(mlx.mlx_array_eval(gk));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    for (positions, 0..) |pos, r| {
        const ri: c_int = @intCast(r);
        const want = try run(es, try o.contiguous(try o.slice(ckv, 0, ri, ri + 1)), w, 512, 64, 1e4, 1e-5, pos, .bfloat16);
        try testing.expectEqualSlices(u8, try bytesOf(&o, try o.own(want.c)), try bytesOf(&o, try o.slice(gc, 0, ri, ri + 1)));
        try testing.expectEqualSlices(u8, try bytesOf(&o, try o.own(want.k_pe)), try bytesOf(&o, try o.slice(gk, 0, ri, ri + 1)));
    }
}
