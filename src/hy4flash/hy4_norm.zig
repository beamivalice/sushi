//! Hy4's f32 RMS norm (`fwd.norm`) in one launch: MLX's `rms_single_row` (`rms_norm.metal`: four values a thread, the
//! simd sums, `precise::rsqrt(sum / D + ε)`, the weight times the normed value) over f32 rows, its weight read as stored
//! (BF16) and widened in the register where the chain cast it to f32 first. The rows are read through their strides (a
//! slice of the latent projection is no copy), may be the sum of two row sets (the combine norm's `R + shared`, the
//! add's f32 sum), and are written at the dtype the caller casts them to next (the latent's BF16). So a row is the
//! bytes of the casts, the add, MLX's norm and the cast after.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// MLX's single-row arm: four reads a thread, rows up to 4,096 (`stock.rms_reads`, `rms_looped_limit`).
const reads = 4;
pub const max_cols: c_int = 4096;

pub const source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\const uint axis_size = uint(D);
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\float acc = 0;
    \\float thread_x[N_READS];
    \\const device T* xr = x + (size_t)gid * (size_t)x_strides[0];
    \\const device TY* yr = y + (YROW ? 0 : (size_t)gid * (size_t)y_strides[0]);
    \\const device TZ* zr = z + (size_t)gid * (size_t)z_strides[0];
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  float v = 0;
    \\  if (c < axis_size) {
    \\    v = static_cast<float>(xr[(size_t)c * (size_t)x_strides[1]]);
    \\    if (ADD) v = v + static_cast<float>(yr[(size_t)c * (size_t)y_strides[1]]);
    \\  }
    \\  thread_x[i] = v;
    \\  acc += thread_x[i] * thread_x[i];
    \\}
    \\acc = simd_sum(acc);
    \\if (simd_group_id == 0) {
    \\  local_sums[simd_lane_id] = 0;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_lane_id == 0) {
    \\  local_sums[simd_group_id] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  acc = simd_sum(local_sums[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\device OT* o = out + (size_t)gid * (size_t)(HC > 0 ? HC : 1) * (size_t)D;
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  if (c < axis_size) {
    \\    float v = static_cast<float>(w[c]) * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    \\    if (MUL) v = static_cast<float>(zr[(size_t)c * (size_t)z_strides[1]]) * v;
    \\    if (HC > 0) {
    \\      // The streams' write-back `post_k · y + h_k` (`hc_ident_expand`), one rounding a step.
    \\      for (int k = 0; k < HC; ++k) {
    \\        float py = static_cast<float>(post[(size_t)gid * HC + k]) * v;
    \\        size_t at = (size_t)k * (size_t)D + c;
    \\        o[at] = static_cast<OT>(py + static_cast<float>(hs[(size_t)gid * HC * (size_t)D + at]));
    \\      }
    \\    } else {
    \\      o[c] = static_cast<OT>(v);
    \\    }
    \\  }
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "x", "y", "z", "w", "hs", "post" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_rms", iv, ov, source, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

fn floatRows(a: Arr) bool {
    const dt = mlx.mlx_array_dtype(a);
    return mlx.getShape(a).len == 2 and (dt == .float32 or dt == .bfloat16);
}

/// Whether `rows` takes `x` [R, D] (and `y`, its shape or one row added to every row) against `w` [D]: a GPU stream, f32 or BF16 rows and weight, and
/// D within MLX's single-row arm.
pub fn applies(s: mlx.mlx_stream, x: Arr, y: ?Arr, w: Arr) bool {
    return appliesTimes(s, x, y, null, w);
}

/// `applies` for `rowsTimes`: `z` too has `x`'s shape.
pub fn appliesTimes(s: mlx.mlx_stream, x: Arr, y: ?Arr, z: ?Arr, w: Arr) bool {
    if (!mlx.streamIsGpu(s) or !floatRows(x)) return false;
    const d = mlx.getShape(x)[1];
    if (y) |b| if (!floatRows(b) or mlx.getShape(b)[1] != d or (mlx.getShape(b)[0] != 1 and mlx.getShape(b)[0] != mlx.getShape(x)[0])) return false;
    if (z) |b| if (!floatRows(b) or !std.mem.eql(c_int, mlx.getShape(b), mlx.getShape(x))) return false;
    const wd = mlx.mlx_array_dtype(w);
    return d > 0 and d <= max_cols and mlx.getShape(w).len == 1 and mlx.getShape(w)[0] == d and (wd == .bfloat16 or wd == .float32);
}

/// `RMSNorm_w(x + y)` in f32 (`y` absent: of `x`), times `z` after (`z` absent: not), as `out_dt` [R, D]. The caller
/// frees the result.
pub fn rows(s: mlx.mlx_stream, x: Arr, y: ?Arr, w: Arr, eps: f32, out_dt: mlx.mlx_dtype) !Arr {
    return rowsTimes(s, x, y, null, w, eps, out_dt);
}

const Key = struct { rows: c_int, d: c_int, x_dt: mlx.mlx_dtype, y_dt: mlx.mlx_dtype, z_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, add: bool, yrow: bool, mul: bool, eps: c_int, hc: c_int = 0 };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const group = 32 * @divFloor(@divFloor(k.d + reads - 1, reads) + 31, 32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (k.hc > 0)
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.hc, k.d }, 3, k.out_dt))
    else
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.d }, 2, k.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", k.hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.rows * group, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", k.x_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TY", k.y_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", k.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", k.d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TZ", k.z_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ADD", k.add));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "YROW", k.yrow));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "MUL", k.mul));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS", k.eps));
    return cfg;
}

pub fn rowsTimes(s: mlx.mlx_stream, x: Arr, y: ?Arr, z: ?Arr, w: Arr, eps: f32, out_dt: mlx.mlx_dtype) !Arr {
    return launch(s, x, y, z, w, eps, out_dt, null);
}

/// The streams after a sublayer whose output is `RMSNorm_w(x + y)`: `post_k · y + h_k` for each of `h`'s [R, HC, D]
/// streams (f32, row-contiguous: the caller's `contiguous`) with `post` [R, HC] f32, in the norm's launch; [R, HC, D] f32.
pub fn rowsExpand(s: mlx.mlx_stream, x: Arr, y: ?Arr, w: Arr, eps: f32, h: Arr, post: Arr) !Arr {
    return launch(s, x, y, null, w, eps, .float32, .{ .h = h, .post = post });
}

/// Whether `rowsExpand` takes `h` [R, HC, D] and `post` [R, HC] beside `rows`' operands.
pub fn expandApplies(s: mlx.mlx_stream, x: Arr, y: ?Arr, w: Arr, h: Arr, post: Arr) bool {
    const hs = mlx.getShape(h);
    const ps = mlx.getShape(post);
    return applies(s, x, y, w) and hs.len == 3 and ps.len == 2 and hs[0] == mlx.getShape(x)[0] and hs[2] == mlx.getShape(x)[1] and
        ps[0] == hs[0] and ps[1] == hs[1] and hs[1] <= 8 and mlx.mlx_array_dtype(h) == .float32 and mlx.mlx_array_dtype(post) == .float32;
}

const Expand = struct { h: Arr, post: Arr };

fn launch(s: mlx.mlx_stream, x: Arr, y: ?Arr, z: ?Arr, w: Arr, eps: f32, out_dt: mlx.mlx_dtype, ex: ?Expand) !Arr {
    const sh = mlx.getShape(x);
    const k = try getKernel();
    const cfg = try cfgs.get(.{
        .rows = sh[0],
        .d = sh[1],
        .x_dt = mlx.mlx_array_dtype(x),
        .y_dt = mlx.mlx_array_dtype(y orelse x),
        .z_dt = mlx.mlx_array_dtype(z orelse x),
        .out_dt = out_dt,
        .add = y != null,
        // A one-row `y` is added to every row (a bias).
        .yrow = if (y) |b| mlx.getShape(b)[0] == 1 and sh[0] != 1 else false,
        .mul = z != null,
        .eps = kcfg.bits(eps),
        .hc = if (ex) |e| mlx.getShape(e.h)[1] else 0,
    }, build);
    const ins = [_]Arr{ x, y orelse x, z orelse x, w, if (ex) |e| e.h else x, if (ex) |e| e.post else x };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

pub const expand_inner_source: [:0]const u8 =
    \\constexpr int N_READS = 4;
    \\constexpr int SIMD_SIZE = 32;
    \\uint gid = threadgroup_position_in_grid.x;
    \\uint lid = thread_position_in_threadgroup.x;
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\const uint axis_size = uint(D);
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float t_sum[4096];
    \\#else
    \\threadgroup float t_sum[D];
    \\#endif
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\{
    \\#pragma clang fp contract(off)
    \\  const uint NB = uint(D) / 128u;
    \\  const float psc = 0.08838834764831845f;
    \\  for (uint blk = simd_group_id; blk < NB; blk += uint(TG) / 32u) {
    \\    const uint col = blk * 128u + simd_lane_id;
    \\    const size_t xb = (size_t)gid * uint(D) + col;
    \\    float4 va = float4(float(a[xb]), float(a[xb + 32u]), float(a[xb + 64u]), float(a[xb + 96u]));
    \\    for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\      const float4 p = simd_shuffle_xor(va, bit);
    \\      va = (simd_lane_id & bit) == 0u ? va + p : p - va;
    \\    }
    \\    const float as0 = va.x + va.y, as1 = va.x - va.y, as2 = va.z + va.w, as3 = va.z - va.w;
    \\    float4 vb = float4(float(b[xb]), float(b[xb + 32u]), float(b[xb + 64u]), float(b[xb + 96u]));
    \\    for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\      const float4 p = simd_shuffle_xor(vb, bit);
    \\      vb = (simd_lane_id & bit) == 0u ? vb + p : p - vb;
    \\    }
    \\    const float bs0 = vb.x + vb.y, bs1 = vb.x - vb.y, bs2 = vb.z + vb.w, bs3 = vb.z - vb.w;
    \\    t_sum[col] = float(bfloat16_t((as0 + as2) * psc * float(asvh[col]))) + float(bfloat16_t((bs0 + bs2) * psc * float(bsvh[col])));
    \\    t_sum[col + 32u] = float(bfloat16_t((as1 + as3) * psc * float(asvh[col + 32u]))) + float(bfloat16_t((bs1 + bs3) * psc * float(bsvh[col + 32u])));
    \\    t_sum[col + 64u] = float(bfloat16_t((as0 - as2) * psc * float(asvh[col + 64u]))) + float(bfloat16_t((bs0 - bs2) * psc * float(bsvh[col + 64u])));
    \\    t_sum[col + 96u] = float(bfloat16_t((as1 - as3) * psc * float(asvh[col + 96u]))) + float(bfloat16_t((bs1 - bs3) * psc * float(bsvh[col + 96u])));
    \\  }
    \\}
    \\if (simd_group_id == 0) {
    \\  local_sums[simd_lane_id] = 0;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float acc = 0;
    \\float thread_x[N_READS];
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  thread_x[i] = c < axis_size ? t_sum[c] : 0;
    \\  acc += thread_x[i] * thread_x[i];
    \\}
    \\acc = simd_sum(acc);
    \\if (simd_lane_id == 0) {
    \\  local_sums[simd_group_id] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  acc = simd_sum(local_sums[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\device float* o = out + (size_t)gid * (size_t)HC * (size_t)D;
    \\for (int i = 0; i < N_READS; i++) {
    \\  uint c = lid * N_READS + i;
    \\  if (c < axis_size) {
    \\    float v = static_cast<float>(w[c]) * static_cast<float>(thread_x[i] * local_inv_mean[0]);
    \\    for (int k = 0; k < HC; ++k) {
    \\      float py = static_cast<float>(post[(size_t)gid * HC + k]) * v;
    \\      size_t at = (size_t)k * (size_t)D + c;
    \\      o[at] = py + static_cast<float>(hs[(size_t)gid * HC * (size_t)D + at]);
    \\    }
    \\  }
    \\}
;

var expand_inner_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ExpandInnerKey = struct { rows: c_int, d: c_int, hc: c_int, eps: c_int };
var expand_inner_cfgs: kcfg.Cache(ExpandInnerKey) = .{};

fn buildExpandInner(k: ExpandInnerKey) !kcfg.Config {
    const group = 32 * @divFloor(@divFloor(k.d + reads - 1, reads) + 31, 32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.hc, k.d }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.rows * group, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", k.d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", group));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", k.hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS", k.eps));
    return cfg;
}

/// Whether `rowsExpandInner` serves: a GPU stream, up to 8 streams of f32 rows, `a` and `b` as [rows, D] unfinished
/// rows with a D-wide `svh`.
pub fn expandInnerApplies(s: mlx.mlx_stream, a: Arr, a_svh: Arr, b: Arr, b_svh: Arr, w: Arr, h: Arr, post: Arr) bool {
    if (!mlx.streamIsGpu(s)) return false;
    const hs = mlx.getShape(h);
    const ps = mlx.getShape(post);
    if (hs.len != 3 or ps.len != 2 or hs[1] > 8) return false;
    const d = hs[2];
    const rsh = [_]c_int{ hs[0], d };
    const wd = mlx.mlx_array_dtype(w);
    return d > 0 and d <= max_cols and @rem(d, 128) == 0 and
        std.mem.eql(c_int, mlx.getShape(a), &rsh) and mlx.mlx_array_dtype(a) == .float16 and
        std.mem.eql(c_int, mlx.getShape(b), &rsh) and mlx.mlx_array_dtype(b) == .float16 and
        mlx.mlx_array_dtype(a_svh) == .float16 and mlx.mlx_array_size(a_svh) == @as(usize, @intCast(d)) and
        mlx.mlx_array_dtype(b_svh) == .float16 and mlx.mlx_array_size(b_svh) == @as(usize, @intCast(d)) and
        mlx.getShape(w).len == 1 and mlx.getShape(w)[0] == d and (wd == .bfloat16 or wd == .float32) and
        ps[0] == hs[0] and ps[1] == hs[1] and mlx.mlx_array_dtype(h) == .float32 and mlx.mlx_array_dtype(post) == .float32;
}

/// `rowsExpand` of the sum of two banks' unfinished GEMM rows `a` and `b`, finished with their `svh` in the launch: the
/// finishes and the expand's norm in one dispatch, their bytes.
pub fn rowsExpandInner(s: mlx.mlx_stream, a: Arr, a_svh: Arr, b: Arr, b_svh: Arr, w: Arr, eps: f32, h: Arr, post: Arr) !Arr {
    if (expand_inner_kernel == null) {
        const ins = [_][*:0]const u8{ "a", "asvh", "b", "bsvh", "w", "hs", "post" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_rms_expand_inner", iv, ov, expand_inner_source, "", true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        expand_inner_kernel = made;
    }
    const hs = mlx.getShape(h);
    const cfg = try expand_inner_cfgs.get(.{ .rows = hs[0], .d = hs[2], .hc = hs[1], .eps = kcfg.bits(eps) }, buildExpandInner);
    const ins = [_]Arr{ a, a_svh, b, b_svh, w, h, post };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, expand_inner_kernel.?, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;
const fwd = @import("hy4_forward.zig");

fn hostBytes(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..n]),
        else => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(c) orelse return error.HostRead)[0..n]),
    };
}

fn randomArr(o: *Ops, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, scale: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * scale);
    const a = try o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
    return o.cast(a, dt);
}

/// The kernel against the chain it replaces (the casts, the add, MLX's norm, the cast after), `x` as given (a view).
fn normCase(o: *Ops, x: Arr, y: ?Arr, w: Arr, eps: f32, out_dt: mlx.mlx_dtype) !void {
    const sum = if (y) |b| try o.binary(.add, try o.cast(x, .float32), try o.cast(b, .float32)) else x;
    const want = try o.cast(try o.rms(try o.cast(sum, .float32), try o.cast(w, .float32), eps), out_dt);
    const want_bytes = try hostBytes(o, want);
    try testing.expect(applies(o.s, x, y, w));
    try mlx.check(mlx.mlx_array_eval(x));
    if (y) |b| try mlx.check(mlx.mlx_array_eval(b));
    try mlx.check(mlx.mlx_array_eval(w));
    const before = mlx.engine.stats().dispatches;
    const got = try o.own(try rows(o.s, x, y, w, eps, out_dt));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqual(out_dt, mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(u8, want_bytes, try hostBytes(o, got));
}

test "hy4 norm: one launch gives the bytes of the weight's cast, the add, MLX's f32 RMS norm and the cast after, at the release's widths (3,072, the latent's 512 as a slice, 1,024, 128, Engram's 96), BF16 and f32 weights and rows, one row and many" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var seed: u64 = 1;
    for ([_]c_int{ 1, 3, 37 }) |t| {
        for ([_]c_int{ 3072, 1024, 128, 96, 4096, 200 }) |d| {
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |wdt| {
                var o: Ops = .{ .s = es };
                defer o.deinit();
                seed += 1;
                const w = try randomArr(&o, &.{d}, wdt, seed, 0.3);
                const x = try randomArr(&o, &.{ t, d }, .float32, seed + 100, 2);
                try normCase(&o, x, null, w, 1e-5, .float32);
                try normCase(&o, x, null, w, fwd.engram_eps, .bfloat16);
                try normCase(&o, x, try randomArr(&o, &.{ t, d }, .float32, seed + 200, 5), w, 1e-5, .float32);
                try normCase(&o, try randomArr(&o, &.{ t, d }, .bfloat16, seed + 300, 2), null, w, 1e-5, .float32);
            }
        }
        var o: Ops = .{ .s = es };
        defer o.deinit();
        // The latent: the first 512 of the 576-wide projection, read in place.
        const ckv = try randomArr(&o, &.{ t, 576 }, .float32, seed + 400, 1);
        const w = try randomArr(&o, &.{512}, .bfloat16, seed + 401, 0.3);
        try normCase(&o, try o.slice(ckv, 1, 0, 512), null, w, 1e-5, .bfloat16);
        // One hash head's rows of Engram's [T, heads, 96] gather.
        const heads = try randomArr(&o, &.{ t, 16, 96 }, .float32, seed + 402, 1);
        try normCase(&o, try o.reshape(try o.slice(heads, 1, 5, 6), &.{ t, 96 }), null, try randomArr(&o, &.{96}, .bfloat16, seed + 403, 0.3), fwd.engram_eps, .float32);
    }
}

test "hy4 norm: the launch with a broadcast bias added before and a row set multiplied after gives the bytes of Engram's key norm and its query norm times the key" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_]c_int{ 1, 6 }) |t| for ([_]c_int{ 3072, 128 }) |d| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = @intCast(t * d);
        const lin = try randomArr(&o, &.{ t, d }, .float32, seed, 2);
        const bias = try randomArr(&o, &.{d}, .bfloat16, seed + 1, 1);
        const wk = try randomArr(&o, &.{d}, .bfloat16, seed + 2, 0.3);
        const wq = try randomArr(&o, &.{d}, .bfloat16, seed + 3, 0.3);
        const h = try randomArr(&o, &.{ t, 4, d }, .float32, seed + 4, 1);
        const hc = try o.reshape(try o.slice(h, 1, 2, 3), &.{ t, d });
        // The chain: k = norm(lin + bias), q = norm(h_c), k · q.
        const k_want = try o.rms(try o.binary(.add, lin, try o.cast(bias, .float32)), try o.cast(wk, .float32), fwd.engram_eps);
        const kq_want = try o.binary(.mul, k_want, try o.rms(hc, try o.cast(wq, .float32), fwd.engram_eps));
        const k_bytes = try hostBytes(&o, k_want);
        const kq_bytes = try hostBytes(&o, kq_want);
        const bias_rows = try o.reshape(bias, &.{ 1, d });
        try testing.expect(applies(es, lin, bias_rows, wk));
        const k = try o.own(try rows(es, lin, bias_rows, wk, fwd.engram_eps, .float32));
        try testing.expectEqualSlices(u8, k_bytes, try hostBytes(&o, k));
        try testing.expect(appliesTimes(es, hc, null, k, wq));
        try mlx.check(mlx.mlx_array_eval(k));
        const before = mlx.engine.stats().dispatches;
        const kq = try o.own(try rowsTimes(es, hc, null, k, wq, fwd.engram_eps, .float32));
        try mlx.check(mlx.mlx_array_eval(kq));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, kq_bytes, try hostBytes(&o, kq));
    };
}

test "hy4 norm: the combine norm with the streams' write-back in its launch gives the bytes of the norm and hc_ident_expand after it, at the release's 4 streams of 3,072 and the tiny's, one row and six" {
    try mlx.requireGpu();
    const hck = @import("../hc_kernels.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_]c_int{ 1, 6 }) |t| for ([_]c_int{ 3072, 128 }) |d| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = @intCast(7 * t + d);
        const up = try randomArr(&o, &.{ t, d }, .float32, seed, 2);
        const shared = try randomArr(&o, &.{ t, d }, .float32, seed + 1, 2);
        const w = try randomArr(&o, &.{d}, .bfloat16, seed + 2, 0.3);
        const h = try randomArr(&o, &.{ t, 4, d }, .float32, seed + 3, 3);
        const post = try randomArr(&o, &.{ t, 4 }, .float32, seed + 4, 1);
        const y = try o.own(try rows(es, up, shared, w, 1e-5, .float32));
        const want = try hostBytes(&o, try o.own(try hck.expandIdentity(es, h, y, post)));
        try testing.expect(expandApplies(es, up, shared, w, h, post));
        try mlx.check(mlx.mlx_array_eval(h));
        try mlx.check(mlx.mlx_array_eval(post));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try rowsExpand(es, up, shared, w, 1e-5, h, post));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostBytes(&o, got));
    };
}
