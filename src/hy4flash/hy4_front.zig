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
pub const source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\{
    \\  const uint axis_size = uint(LAT);
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

/// Direct in-place variant of `source`: writes BF16 `c` and rotated BF16 `k_pe` directly into the ring (`ring_len > 0`)
/// or linear store (`ring_len == 0`) starting at `start_row`, eliminating the separate `copy_into_bf16` dispatches.
pub const into_source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\{
    \\  const uint axis_size = uint(LAT);
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
    \\  const uint start_row = uint(meta[1]);
    \\  const uint ring_len = uint(meta[2]);
    \\  const uint dst_row = ring_len > 0u ? ((start_row + gid) % ring_len) : (start_row + gid);
    \\  device bfloat16_t* o = c_dst + (size_t)dst_row * (size_t)LAT;
    \\  for (int i = 0; i < N_READS; i++) {
    \\    uint c = lid * N_READS + i;
    \\    if (c < axis_size) {
    \\      float v = static_cast<float>(w[c]) * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    \\      o[c] = static_cast<bfloat16_t>(v);
    \\    }
    \\  }
    \\  if (lid < uint(HALF)) {
    \\    uint i = lid;
    \\    const device T* row = xr + LAT;
    \\    device bfloat16_t* dst = k_dst + (size_t)dst_row * (size_t)(2 * HALF);
    \\    float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\    float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\    float L = as_type<float>(SCALE) * static_cast<float>(meta[0] + int(gid));
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

/// Direct in-place KV8 (affine 8-bit, group 64) variant of `into_source`: fuses RMSNorm, affine-8 group-64
/// quantization, and RoPE into one launch writing directly into `(q_dst, s_dst, b_dst, k_dst)` of the ring
/// (`ring_len > 0`) or linear store (`ring_len == 0`) starting at `start_row`. Each 16-lane half-simdgroup holds one
/// 64-element quant group (4 consecutive columns per thread = one packed `uint32_t` word), matching `hy4_front` +
/// `hy4_latent.append` byte for byte with zero extra barriers.
pub const into_kv8_source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\{
    \\  const uint axis_size = uint(LAT);
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
    \\  const uint start_row = uint(meta[1]);
    \\  const uint ring_len = uint(meta[2]);
    \\  const uint dst_row = ring_len > 0u ? ((start_row + gid) % ring_len) : (start_row + gid);
    \\  const uint words = axis_size >> 2;
    \\  if (lid < words) {
    \\    float qv[4];
    \\    const float inv_m = local_inv_mean[0];
    \\    for (int i = 0; i < 4; i++) {
    \\      uint c = lid * 4 + i;
    \\      float v = static_cast<float>(w[c]) * static_cast<float>(thread_x[i] * inv_m);
    \\      qv[i] = static_cast<float>(static_cast<bfloat16_t>(v));
    \\    }
    \\    float w_min = min(min(metal::numeric_limits<float>::infinity(), min(qv[0], qv[1])), min(qv[2], qv[3]));
    \\    float w_max = max(max(0.0f, max(qv[0], qv[1])), max(qv[2], qv[3]));
    \\    w_min = min(w_min, simd_shuffle_xor(w_min, ushort(1)));
    \\    w_min = min(w_min, simd_shuffle_xor(w_min, ushort(2)));
    \\    w_min = min(w_min, simd_shuffle_xor(w_min, ushort(4)));
    \\    w_min = min(w_min, simd_shuffle_xor(w_min, ushort(8)));
    \\    w_max = max(w_max, simd_shuffle_xor(w_max, ushort(1)));
    \\    w_max = max(w_max, simd_shuffle_xor(w_max, ushort(2)));
    \\    w_max = max(w_max, simd_shuffle_xor(w_max, ushort(4)));
    \\    w_max = max(w_max, simd_shuffle_xor(w_max, ushort(8)));
    \\    constexpr float eps_q = 1e-7f;
    \\    constexpr float n_bins = 255.0f;
    \\    float scale = max((w_max - w_min) / n_bins, eps_q);
    \\    bool side = abs(w_min) > abs(w_max);
    \\    scale = side ? scale : -scale;
    \\    float edge = side ? w_min : w_max;
    \\    float q0 = round(edge / scale);
    \\    bool at_zero = q0 == 0.0f;
    \\    scale = at_zero ? scale : edge / q0;
    \\    float bias = at_zero ? 0.0f : edge;
    \\    const uint groups = axis_size >> 6;
    \\    const uint g = lid >> 4;
    \\    if ((simd_lane_id & 15u) == 0u) {
    \\      size_t g_out = size_t(dst_row) * size_t(groups) + size_t(g);
    \\      s_dst[g_out] = static_cast<bfloat16_t>(scale);
    \\      b_dst[g_out] = static_cast<bfloat16_t>(bias);
    \\    }
    \\    uint8_t b0 = uint8_t(min(round((qv[0] - bias) / scale), n_bins));
    \\    uint8_t b1 = uint8_t(min(round((qv[1] - bias) / scale), n_bins));
    \\    uint8_t b2 = uint8_t(min(round((qv[2] - bias) / scale), n_bins));
    \\    uint8_t b3 = uint8_t(min(round((qv[3] - bias) / scale), n_bins));
    \\    q_dst[size_t(dst_row) * size_t(words) + size_t(lid)] =
    \\        uint32_t(b0) | (uint32_t(b1) << 8) | (uint32_t(b2) << 16) | (uint32_t(b3) << 24);
    \\  }
    \\  if (lid < uint(HALF)) {
    \\    uint i = lid;
    \\    const device T* row = xr + LAT;
    \\    device bfloat16_t* dst = k_dst + (size_t)dst_row * (size_t)(2 * HALF);
    \\    float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\    float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\    float L = as_type<float>(SCALE) * static_cast<float>(meta[0] + int(gid));
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

var into_kernel: ?mlx.mlx_fast_metal_kernel = null;
var into_kv8_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getIntoKernel() !mlx.mlx_fast_metal_kernel {
    if (into_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "ckv", "w", "meta" };
    const outs = [_][*:0]const u8{ "c_dst", "k_dst" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_front_into", iv, ov, into_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    into_kernel = k;
    return k;
}

fn getIntoKv8Kernel() !mlx.mlx_fast_metal_kernel {
    if (into_kv8_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "ckv", "w", "meta" };
    const outs = [_][*:0]const u8{ "q_dst", "s_dst", "b_dst", "k_dst" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_front_into_kv8", iv, ov, into_kv8_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    into_kv8_kernel = k;
    return k;
}

/// Destination buffers for `runInto`: ring (`ring_len > 0`) or linear store (`ring_len == 0`) starting at `start_row`.
/// When `s_dst` and `b_dst` are non-null, writes affine-8 (group 64) quantized `c` directly into `(c_dst, s_dst, b_dst)`.
pub const Into = struct {
    c_dst: Arr,
    k_dst: Arr,
    s_dst: ?Arr = null,
    b_dst: ?Arr = null,
    start_row: c_int,
    ring_len: c_int,

    pub fn isKv8(self: Into) bool {
        return self.s_dst != null and self.b_dst != null;
    }
};

const IntoKey = struct { t: c_int, lat: c_int, half: c_int, in_dt: mlx.mlx_dtype, w_dt: mlx.mlx_dtype, base: c_int, eps: c_int, kv8: bool = false };
var into_cfgs: kcfg.Cache(IntoKey) = .{};

fn buildInto(k: IntoKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const tg = normGroup(k.lat);
    if (k.kv8) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, @divExact(k.lat, 4) }, 2, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, @divExact(k.lat, 64) }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, @divExact(k.lat, 64) }, 2, .bfloat16));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k.lat }, 2, .bfloat16));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 2 * k.half }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.t * tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    inline for (.{ "LAT", "CKW", "HALF", "BASE", "EPS", "SCALE" }, .{ k.lat, k.lat + 2 * k.half, k.half, k.base, k.eps, kcfg.bits(1.0) }) |n, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", k.in_dt));
    return cfg;
}

/// Whether `runInto` can write BF16 or KV8 `c` and BF16 `k_pe` directly into `into` on the engine stream.
pub fn intoApplies(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int, into: Into) bool {
    if (!mlx.engine.isStream(s) or !applies(s, ckv, w, latent, rope)) return false;
    if (into.start_row < 0 or into.ring_len < 0) return false;
    if (mlx.mlx_array_dtype(into.k_dst) != .bfloat16) return false;
    const cs = mlx.getShape(into.c_dst);
    const ks = mlx.getShape(into.k_dst);
    if (cs.len != 2 or ks.len != 2 or ks[1] != rope or cs[0] != ks[0]) return false;
    if (into.isKv8()) {
        if (@rem(latent, 64) != 0) return false;
        if (mlx.mlx_array_dtype(into.c_dst) != .uint32 or cs[1] != @divExact(latent, 4)) return false;
        const sc = into.s_dst.?;
        const bi = into.b_dst.?;
        if (mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
        const ss = mlx.getShape(sc);
        const bs = mlx.getShape(bi);
        const groups = @divExact(latent, 64);
        return ss.len == 2 and bs.len == 2 and ss[0] == cs[0] and bs[0] == cs[0] and ss[1] == groups and bs[1] == groups;
    }
    return mlx.mlx_array_dtype(into.c_dst) == .bfloat16 and cs[1] == latent;
}

/// Runs `hy4_front` writing BF16 or KV8 `c` and rotated BF16 `k_pe` directly into `into` via `kernelApplyInPlace`.
pub fn runInto(s: mlx.mlx_stream, ckv: Arr, w: Arr, latent: c_int, rope: c_int, theta: f32, eps: f32, pos0: c_int, into: Into) !void {
    const t = mlx.getShape(ckv)[0];
    const kv8 = into.isKv8();
    const cfg = try into_cfgs.get(.{ .t = t, .lat = latent, .half = @divExact(rope, 2), .in_dt = mlx.mlx_array_dtype(ckv), .w_dt = mlx.mlx_array_dtype(w), .base = kcfg.bits(std.math.log2(theta)), .eps = kcfg.bits(eps), .kv8 = kv8 }, buildInto);
    const mv = [_]i32{ pos0, into.start_row, into.ring_len };
    const meta = try mlx.newData(&mv, &.{3}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const ins = [_]Arr{ ckv, w, meta };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    if (kv8) {
        const dsts = [_]Arr{ into.c_dst, into.s_dst.?, into.b_dst.?, into.k_dst };
        try mlx.check(mlx.kernelApplyInPlace(try getIntoKv8Kernel(), iv, cfg, s, &dsts) orelse return error.MlxError);
    } else {
        const dsts = [_]Arr{ into.c_dst, into.k_dst };
        try mlx.check(mlx.kernelApplyInPlace(try getIntoKernel(), iv, cfg, s, &dsts) orelse return error.MlxError);
    }
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

test "hy4 front: runInto writes ring (with wrap) and linear stores byte for byte identically to run in one dispatch" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |in_dt| {
        // 1) Ring write with wrap (ring_len = 512, start_row = 510, t = 6 -> wraps to 510, 511, 0, 1, 2, 3)
        {
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const t: c_int = 6;
            const ring_len: c_int = 512;
            const pos0: c_int = 1022; // 1022 % 512 == 510
            const ckv = try o.contiguous(try randomArr(&o, &.{ t, 576 }, in_dt, 701, 1.5));
            const w = try o.contiguous(try randomArr(&o, &.{512}, .bfloat16, 702, 0.3));
            const c_dst = try o.zeros(&.{ ring_len, 512 }, .bfloat16);
            const k_dst = try o.zeros(&.{ ring_len, 64 }, .bfloat16);
            for ([_]Arr{ ckv, w, c_dst, k_dst }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
            const want = try run(es, ckv, w, 512, 64, 1e4, 1e-5, pos0, .bfloat16);
            const wc = try o.own(want.c);
            const wk = try o.own(want.k_pe);
            const into: Into = .{ .c_dst = c_dst, .k_dst = k_dst, .start_row = pos0, .ring_len = ring_len };
            try testing.expect(intoApplies(es, ckv, w, 512, 64, into));
            const before = mlx.engine.stats().dispatches;
            try runInto(es, ckv, w, 512, 64, 1e4, 1e-5, pos0, into);
            try mlx.check(mlx.mlx_array_eval(c_dst));
            try mlx.check(mlx.mlx_array_eval(k_dst));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            for (0..@intCast(t)) |r| {
                const ri: c_int = @intCast(r);
                const slot: c_int = @mod(pos0 + ri, ring_len);
                try testing.expectEqualSlices(u8, try bytesOf(&o, try o.slice(wc, 0, ri, ri + 1)), try bytesOf(&o, try o.slice(c_dst, 0, slot, slot + 1)));
                try testing.expectEqualSlices(u8, try bytesOf(&o, try o.slice(wk, 0, ri, ri + 1)), try bytesOf(&o, try o.slice(k_dst, 0, slot, slot + 1)));
            }
        }
        // 2) Linear sparse append (ring_len = 0, start_row = 17, t = 12)
        {
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const t: c_int = 12;
            const start_row: c_int = 17;
            const ckv = try o.contiguous(try randomArr(&o, &.{ t, 576 }, in_dt, 711, 1.5));
            const w = try o.contiguous(try randomArr(&o, &.{512}, .bfloat16, 712, 0.3));
            const c_dst = try o.zeros(&.{ 64, 512 }, .bfloat16);
            const k_dst = try o.zeros(&.{ 64, 64 }, .bfloat16);
            for ([_]Arr{ ckv, w, c_dst, k_dst }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
            const want = try run(es, ckv, w, 512, 64, 1e4, 1e-5, start_row, .bfloat16);
            const wc = try o.own(want.c);
            const wk = try o.own(want.k_pe);
            const into: Into = .{ .c_dst = c_dst, .k_dst = k_dst, .start_row = start_row, .ring_len = 0 };
            try testing.expect(intoApplies(es, ckv, w, 512, 64, into));
            const before = mlx.engine.stats().dispatches;
            try runInto(es, ckv, w, 512, 64, 1e4, 1e-5, start_row, into);
            try mlx.check(mlx.mlx_array_eval(c_dst));
            try mlx.check(mlx.mlx_array_eval(k_dst));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            try testing.expectEqualSlices(u8, try bytesOf(&o, wc), try bytesOf(&o, try o.slice(c_dst, 0, start_row, start_row + t)));
            try testing.expectEqualSlices(u8, try bytesOf(&o, wk), try bytesOf(&o, try o.slice(k_dst, 0, start_row, start_row + t)));
        }
    }
}

test "hy4 front: runInto KV8 writes ring (with wrap) and linear stores byte for byte identically to run + latent.append in one dispatch" {
    try mlx.requireGpu();
    const hy4_latent = @import("hy4_latent.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const u32Bytes = struct {
        fn of(o: *Ops, a: Arr) ![]const u8 {
            const c = try o.contiguous(a);
            try mlx.check(mlx.mlx_array_eval(c));
            return std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
        }
    }.of;

    const Case = struct { lat: c_int, rope: c_int, theta: f32 };
    for ([_]Case{
        .{ .lat = 512, .rope = 64, .theta = 1e4 },
        .{ .lat = 512, .rope = 64, .theta = 1e7 },
        .{ .lat = 128, .rope = 16, .theta = 1e4 },
    }, 0..) |c, ci| {
        const words = @divExact(c.lat, 4);
        const groups = @divExact(c.lat, 64);
        for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |in_dt| {
            // 1) Ring write with wrap
            {
                var o: Ops = .{ .s = es };
                defer o.deinit();
                const t: c_int = 6;
                const ring_len: c_int = 512;
                const pos0: c_int = 1022;
                const ckv = try o.contiguous(try randomArr(&o, &.{ t, c.lat + c.rope }, in_dt, 801 + ci, 1.5));
                const w = try o.contiguous(try randomArr(&o, &.{c.lat}, .bfloat16, 802 + ci, 0.3));
                const wq = try o.zeros(&.{ ring_len, words }, .uint32);
                const ws = try o.zeros(&.{ ring_len, groups }, .bfloat16);
                const wb = try o.zeros(&.{ ring_len, groups }, .bfloat16);
                const wk = try o.zeros(&.{ ring_len, c.rope }, .bfloat16);
                const gq = try o.zeros(&.{ ring_len, words }, .uint32);
                const gs = try o.zeros(&.{ ring_len, groups }, .bfloat16);
                const gb = try o.zeros(&.{ ring_len, groups }, .bfloat16);
                const gk = try o.zeros(&.{ ring_len, c.rope }, .bfloat16);
                for ([_]Arr{ ckv, w, wq, ws, wb, wk, gq, gs, gb, gk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                const f = try run(es, ckv, w, c.lat, c.rope, c.theta, 1e-5, pos0, .bfloat16);
                const fc = try o.own(f.c);
                const fk = try o.own(f.k_pe);
                try hy4_latent.append(es, fc, fk, &.{ wq, ws, wb, wk }, pos0, ring_len);
                for ([_]Arr{ wq, ws, wb, wk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                const into: Into = .{ .c_dst = gq, .s_dst = gs, .b_dst = gb, .k_dst = gk, .start_row = pos0, .ring_len = ring_len };
                try testing.expect(intoApplies(es, ckv, w, c.lat, c.rope, into));
                const before = mlx.engine.stats().dispatches;
                try runInto(es, ckv, w, c.lat, c.rope, c.theta, 1e-5, pos0, into);
                for ([_]Arr{ gq, gs, gb, gk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
                try testing.expectEqualSlices(u8, try u32Bytes(&o, wq), try u32Bytes(&o, gq));
                try testing.expectEqualSlices(u8, try bytesOf(&o, ws), try bytesOf(&o, gs));
                try testing.expectEqualSlices(u8, try bytesOf(&o, wb), try bytesOf(&o, gb));
                try testing.expectEqualSlices(u8, try bytesOf(&o, wk), try bytesOf(&o, gk));
            }
            // 2) Linear sparse append
            {
                var o: Ops = .{ .s = es };
                defer o.deinit();
                const t: c_int = 12;
                const start_row: c_int = 17;
                const ckv = try o.contiguous(try randomArr(&o, &.{ t, c.lat + c.rope }, in_dt, 821 + ci, 1.5));
                const w = try o.contiguous(try randomArr(&o, &.{c.lat}, .bfloat16, 822 + ci, 0.3));
                const wq = try o.zeros(&.{ 64, words }, .uint32);
                const ws = try o.zeros(&.{ 64, groups }, .bfloat16);
                const wb = try o.zeros(&.{ 64, groups }, .bfloat16);
                const wk = try o.zeros(&.{ 64, c.rope }, .bfloat16);
                const gq = try o.zeros(&.{ 64, words }, .uint32);
                const gs = try o.zeros(&.{ 64, groups }, .bfloat16);
                const gb = try o.zeros(&.{ 64, groups }, .bfloat16);
                const gk = try o.zeros(&.{ 64, c.rope }, .bfloat16);
                for ([_]Arr{ ckv, w, wq, ws, wb, wk, gq, gs, gb, gk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                const f = try run(es, ckv, w, c.lat, c.rope, c.theta, 1e-5, start_row, .bfloat16);
                const fc = try o.own(f.c);
                const fk = try o.own(f.k_pe);
                try hy4_latent.append(es, fc, fk, &.{ wq, ws, wb, wk }, start_row, 0);
                for ([_]Arr{ wq, ws, wb, wk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                const into: Into = .{ .c_dst = gq, .s_dst = gs, .b_dst = gb, .k_dst = gk, .start_row = start_row, .ring_len = 0 };
                try testing.expect(intoApplies(es, ckv, w, c.lat, c.rope, into));
                const before = mlx.engine.stats().dispatches;
                try runInto(es, ckv, w, c.lat, c.rope, c.theta, 1e-5, start_row, into);
                for ([_]Arr{ gq, gs, gb, gk }) |arr| try mlx.check(mlx.mlx_array_eval(arr));
                try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
                try testing.expectEqualSlices(u8, try u32Bytes(&o, wq), try u32Bytes(&o, gq));
                try testing.expectEqualSlices(u8, try bytesOf(&o, ws), try bytesOf(&o, gs));
                try testing.expectEqualSlices(u8, try bytesOf(&o, wb), try bytesOf(&o, gb));
                try testing.expectEqualSlices(u8, try bytesOf(&o, wk), try bytesOf(&o, gk));
            }
        }
    }
}
