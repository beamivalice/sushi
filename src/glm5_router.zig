//! One-token GLM sigmoid router; source-derived FP32 GEMV and stable selection.
//! Adapted from oMLX 6745c39c (Apache-2.0); see NOTICE.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\using namespace metal;
    \\template<typename T,bool precise> inline T sigmoid(T x) {
    \\ auto y=1/(1+(precise?metal::precise::exp(metal::abs(x)):metal::exp(metal::abs(x))));
    \\ return x<0?y:1-y;
    \\}
;
const LOGITS: [:0]const u8 =
    \\
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const uint sg = simdgroup_index_in_threadgroup;
    \\  const int tok = int(threadgroup_position_in_grid.y);
    \\  constexpr int RPS = ROWS_PER_SIMD;
    \\  const int out_row = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RPS;
    \\  if (out_row >= E) {
    \\    return;
    \\  }
    \\  auto mat = w + size_t(out_row) * K;
    \\  const device T* xv = x + size_t(tok) * K;
    \\  float result[RPS];
    \\  for (int tm = 0; tm < RPS; tm++) {
    \\    result[tm] = 0.0f;
    \\  }
    \\  int bn = int(lane) * 4;
    \\  for (int i = 0; i < K / 128; ++i) {
    \\    float v_coeff[4];
    \\    for (int tn = 0; tn < 4; tn++) {
    \\      v_coeff[tn] = static_cast<float>(xv[bn + tn]);
    \\    }
    \\    int mat_offset = 0;
    \\    for (int tm = 0; tm < RPS; tm++) {
    \\      float inter[4];
    \\      for (int tn = 0; tn < 4; tn++) {
    \\        inter[tn] = mat[mat_offset + bn + tn];
    \\      }
    \\      for (int tn = 0; tn < 4; tn++) {
    \\        result[tm] += inter[tn] * v_coeff[tn];
    \\      }
    \\      mat_offset += K;
    \\    }
    \\    bn += 128;
    \\  }
    \\  for (int tm = 0; tm < RPS; tm++) {
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) {
    \\      result[tm] += simd_shuffle_down(result[tm], sn);
    \\    }
    \\  }
    \\  if (lane == 0) {
    \\    for (int tm = 0; tm < RPS; tm++) {
    \\      const int e = out_row + tm;
    \\      float sgm = sigmoid<float,PRECISE>(result[tm]);
    \\      float biased = sgm + bias[e];
    \\      sig[size_t(tok) * E + e] = sgm;
    \\      biased_out[size_t(tok) * E + e] = biased;
    \\    }
    \\  }
;
const SELECT: [:0]const u8 =
    \\
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int tok = int(threadgroup_position_in_grid.x);
    \\  constexpr int PER = (E + 31) / 32;
    \\  const device float* bz = biased + size_t(tok) * E;
    \\  const device float* sz = sig + size_t(tok) * E;
    \\  float vals[PER];
    \\  bool taken[PER];
    \\  for (int j = 0; j < PER; j++) {
    \\    const int e = j * 32 + int(lane);
    \\    vals[j] = e < E ? bz[e] : -INFINITY;
    \\    taken[j] = e >= E;
    \\  }
    \\  int picked[TOPK];
    \\  for (int r = 0; r < TOPK; r++) {
    \\    float best = -INFINITY;
    \\    int best_e = 0x7fffffff;
    \\    for (int j = 0; j < PER; j++) {
    \\      const int e = j * 32 + int(lane);
    \\      if (!taken[j] && !isnan(vals[j]) &&
    \\          (best_e == 0x7fffffff || vals[j] > best || (vals[j] == best && e < best_e))) {
    \\        best = vals[j];
    \\        best_e = e;
    \\      }
    \\    }
    \\    for (ushort off = 16; off >= 1; off >>= 1) {
    \\      float ob = simd_shuffle_xor(best, off);
    \\      int oe = simd_shuffle_xor(best_e, off);
    \\      const bool other_better = oe != 0x7fffffff &&
    \\          (best_e == 0x7fffffff || ob > best || (ob == best && oe < best_e));
    \\      if (other_better) {
    \\        best = ob;
    \\        best_e = oe;
    \\      }
    \\    }
    \\    if (best_e == 0x7fffffff) {
    \\      for (int j = 0; j < PER; j++) {
    \\        const int e = j * 32 + int(lane);
    \\        if (!taken[j] && e < best_e) {
    \\          best_e = e;
    \\        }
    \\      }
    \\      for (ushort off = 16; off >= 1; off >>= 1) {
    \\        best_e = min(best_e, simd_shuffle_xor(best_e, off));
    \\      }
    \\    }
    \\    picked[r] = best_e;
    \\    if ((best_e % 32) == int(lane)) {
    \\      taken[best_e / 32] = true;
    \\    }
    \\  }
    \\  if (lane == 0) {
    \\    float total = 0.0f;
    \\    float gathered[TOPK];
    \\    for (int r = 0; r < TOPK; r++) {
    \\      gathered[r] = sz[picked[r]];
    \\      total = gathered[r] + total;
    \\    }
    \\    for (int r = 0; r < TOPK; r++) {
    \\      float q = NORM ? gathered[r] / total : gathered[r];
    \\      float s = q * scaling[0];
    \\      indices[tok * TOPK + r] = uint(picked[r]);
    \\      scores[tok * TOPK + r] = s;
    \\    }
    \\  }
;

pub const Result = struct { indices: Arr, scores: Arr };
var logits_kernel: ?mlx.mlx_fast_metal_kernel = null;
var select_kernel: ?mlx.mlx_fast_metal_kernel = null;
var calls: usize = 0;
var batch_calls: usize = 0;
pub fn batchCallCount() usize {
    return batch_calls;
}
pub fn resetBatchCallCount() void {
    batch_calls = 0;
}
pub fn callCount() usize {
    return calls;
}
pub fn resetCallCount() void {
    calls = 0;
}
fn kernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}
const Configs = struct { logits: mlx.mlx_fast_metal_kernel_config, select: mlx.mlx_fast_metal_kernel_config, cached: bool };
const Key = struct { rows: c_int, width: c_int, experts: c_int, top: c_int, norm: bool, dtype: mlx.mlx_dtype, precise: bool };
const Entry = struct { key: Key, cfg: Configs };
var cache: [8]?Entry = @splat(null);
fn configs(key: Key) !Configs {
    for (cache) |entry| if (entry) |e| {
        if (std.meta.eql(e.key, key)) return e.cfg;
    };
    const lc = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(lc);
    const sc = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(sc);
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(lc, &.{ key.rows, key.experts }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(lc, 128 * @divExact(key.experts, 4), key.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(lc, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(lc, "T", key.dtype));
    inline for (.{ "K", "E", "ROWS_PER_SIMD", "PRECISE" }, .{ key.width, key.experts, @as(c_int, 1), @as(c_int, @intFromBool(key.precise)) }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(lc, name, value));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(sc, &.{ 1, key.rows, key.top }, 3, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(sc, &.{ 1, key.rows, key.top }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(sc, 32 * key.rows, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(sc, 32, 1, 1));
    inline for (.{ "E", "TOPK", "NORM" }, .{ key.experts, key.top, @as(c_int, @intFromBool(key.norm)) }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(sc, name, value));
    for (&cache) |*entry| if (entry.* == null) {
        const result = Configs{ .logits = lc, .select = sc, .cached = true };
        entry.* = .{ .key = key, .cfg = result };
        return result;
    };
    return .{ .logits = lc, .select = sc, .cached = false };
}
fn apply(k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) ![2]Arr {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = [_]Arr{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (&out, 0..) |*v, i| try mlx.check(mlx.mlx_vector_array_get(v, ov, i));
    return out;
}
pub fn route(s: mlx.mlx_stream, x: Arr, weight: Arr, bias: Arr, top: c_int, scale: f32, norm: bool) !?Result {
    return routeImpl(s, x, weight, bias, top, scale, norm, 1);
}
fn routeImpl(s: mlx.mlx_stream, x: Arr, weight: Arr, bias: Arr, top: c_int, scale: f32, norm: bool, max_rows: c_int) !?Result {
    if (!mlx.streamIsGpu(s) or x.ctx == null or weight.ctx == null or bias.ctx == null or !std.math.isFinite(scale) or scale <= 0) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(weight);
    if (xs.len != 3 or xs[0] != 1 or xs[1] < 1 or xs[1] > max_rows or ws.len != 2 or xs[2] != ws[1]) return null;
    const width = ws[1];
    const experts = ws[0];
    const dtype = mlx.mlx_array_dtype(x);
    if ((dtype != .bfloat16 and dtype != .float32) or (mlx.mlx_array_dtype(weight) != .float32 and mlx.mlx_array_dtype(weight) != .bfloat16) or mlx.mlx_array_dtype(bias) != .float32 or !std.mem.eql(c_int, &.{experts}, mlx.getShape(bias)) or experts < 16 or experts > 1024 or @mod(experts, 16) != 0 or width <= 64 or width >= 16 * experts or @mod(width, 128) != 0 or top < 1 or top > experts or top > 32) return null;
    var ready = false;
    try mlx.check(mlx._mlx_array_is_available(&ready, weight));
    if (!ready) return null;
    const strides = mlx.mlx_array_strides(weight);
    if (strides[1] != 1 or strides[0] != @as(usize, @intCast(width))) return null;
    const precise = (try @import("glm5_kda_fused.zig").sigmoidFloatMode(s)) orelse return null;
    const cfg = try configs(.{ .rows = xs[1], .width = width, .experts = experts, .top = top, .norm = norm, .dtype = dtype, .precise = precise });
    defer if (!cfg.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.logits);
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg.select);
    };
    const lk = try kernel(&logits_kernel, "sushi_glm_router_logits", &.{ "x", "w", "bias" }, &.{ "sig", "biased_out" }, LOGITS);
    const values = try apply(lk, &.{ x, weight, bias }, cfg.logits, s);
    defer for (values) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const scaling = mlx.mlx_array_new_data(&scale, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(scaling);
    const sk = try kernel(&select_kernel, "sushi_glm_router_select", &.{ "sig", "biased", "scaling" }, &.{ "indices", "scores" }, SELECT);
    const chosen = try apply(sk, &.{ values[0], values[1], scaling }, cfg.select, s);
    calls += 1;
    if (xs[1] > 1) batch_calls += 1;
    return .{ .indices = chosen[0], .scores = chosen[1] };
}

pub fn routeBatch(s: mlx.mlx_stream, x: Arr, weight: Arr, bias: Arr, top: c_int, scale: f32, norm: bool) !?Result {
    return routeImpl(s, x, weight, bias, top, scale, norm, 16);
}

test "GLM batched router preserves every serial index and score bit" {
    try mlx.requireGpu();
    const Ops = @import("glm5_model.zig").Ops;
    const stream = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| {
        var ops = Ops{ .s = stream };
        defer ops.deinit();
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 842));
        const weight = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(weight, &.{ 288, 4096 }, 2, dtype, 0, 0.01, key.*, stream));
        const correction = try ops.zeros(&.{288}, .float32);
        const input = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(input, &.{ 1, 16, 4096 }, 3, dtype, 0, 1, key.*, stream));
        try mlx.check(mlx.mlx_array_eval(weight.*));
        for ([_]c_int{ 2, 3, 4, 8, 16 }) |rows| {
            for ([_]bool{ false, true }) |normalize| {
                var scope = Ops{ .s = stream };
                defer scope.deinit();
                const x = try scope.slice(input.*, 1, 0, rows);
                const got = (try routeBatch(stream, x, weight.*, correction, 8, 2.5, normalize)) orelse return error.TestExpectedBatchedRouter;
                defer _ = mlx.mlx_array_free(got.indices);
                defer _ = mlx.mlx_array_free(got.scores);
                try mlx.check(mlx.mlx_array_eval(got.indices));
                try mlx.check(mlx.mlx_array_eval(got.scores));
                const gi = mlx.mlx_array_data_uint32(got.indices).?;
                const gs = mlx.mlx_array_data_float32(got.scores).?;
                for (0..@intCast(rows)) |row| {
                    var one = Ops{ .s = stream };
                    defer one.deinit();
                    const from: c_int = @intCast(row);
                    const expected = (try route(stream, try one.slice(x, 1, from, from + 1), weight.*, correction, 8, 2.5, normalize)).?;
                    defer _ = mlx.mlx_array_free(expected.indices);
                    defer _ = mlx.mlx_array_free(expected.scores);
                    try mlx.check(mlx.mlx_array_eval(expected.indices));
                    try mlx.check(mlx.mlx_array_eval(expected.scores));
                    try std.testing.expectEqualSlices(u32, mlx.mlx_array_data_uint32(expected.indices).?[0..8], gi[row * 8 ..][0..8]);
                    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(expected.scores).?[0..8]), std.mem.sliceAsBytes(gs[row * 8 ..][0..8]));
                }
            }
        }
    }
}

test "GLM batched router stable ties and single-row contract" {
    try mlx.requireGpu();
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const weight = try ops.zeros(&.{ 288, 4096 }, .bfloat16);
    const x = try ops.zeros(&.{ 1, 4, 4096 }, .bfloat16);
    const bias = try ops.zeros(&.{288}, .float32);
    try mlx.check(mlx.mlx_array_eval(weight));
    try std.testing.expect((try route(ops.s, x, weight, bias, 8, 2.5, true)) == null);
    const got = (try routeBatch(ops.s, x, weight, bias, 8, 2.5, true)).?;
    defer _ = mlx.mlx_array_free(got.indices);
    defer _ = mlx.mlx_array_free(got.scores);
    try mlx.check(mlx.mlx_array_eval(got.indices));
    try mlx.check(mlx.mlx_array_eval(got.scores));
    for (0..32) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i % 8)), mlx.mlx_array_data_uint32(got.indices).?[i]);
        try std.testing.expectEqual(@as(f32, 0.3125), mlx.mlx_array_data_float32(got.scores).?[i]);
    }
    const too_many = try ops.zeros(&.{ 1, 17, 4096 }, .bfloat16);
    try std.testing.expect((try routeBatch(ops.s, too_many, weight, bias, 8, 2.5, true)) == null);
}

