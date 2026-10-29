const trf = @import("transformer.zig");
const moe_gqmv = @import("moe_gqmv.zig");
const std = @import("std");
const mtp_mod = @import("mtp.zig");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const test_support = @import("test_support.zig");
const attn256RandBf16 = test_support.attn256RandBf16;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const QuantMode = model_mod.QuantMode;
const testing = std.testing;
const attn_kernels = @import("attn_kernels.zig");
const QsaCfgCache = attn_kernels.QsaCfgCache;
const sliceAttentionSeq = attn_kernels.sliceAttentionSeq;
const nax_gate = @import("nax_gate.zig");
const gpuArchitecture = nax_gate.gpuArchitecture;
const naxMinMForBits = nax_gate.naxMinMForBits;
const verifyQmmNaxAvailable = nax_gate.verifyQmmNaxAvailable;
const verify_qmm = @import("verify_qmm.zig");
const cachedScalarInt = verify_qmm.cachedScalarInt;
const mixedNaxShapeEnabled = verify_qmm.mixedNaxShapeEnabled;
const mixedPlainEnabled = verify_qmm.mixedPlainEnabled;
const verifyQmm = verify_qmm.verifyQmm;
const verifyQmmEnabled = verify_qmm.verifyQmmEnabled;
const gatherExpertMm = trf.gatherExpertMm;
const lastDim = trf.lastDim;
const msvQmvRows = trf.msvQmvRows;
const transposeBf16Weight = trf.transposeBf16Weight;

const bf16Trunc = test_support.bf16Trunc;
const evalToF32 = test_support.evalToF32;

pub const PREFILL_DQ_GEMM_MIN_M: usize = 2048;

/// Test seam: forces the route on/off without the environment.
pub var prefill_dq_gemm_override: ?bool = null;

pub fn prefillDqGemmEnabled() bool {
    return prefill_dq_gemm_override orelse true;
}

/// Test seam: engagement is counted, never inferred from output equality.
pub var prefill_dq_gemm_engaged: u64 = 0;

var verify_shared_arch: ?bool = null;

// Preserve MLX's affine qmv K reduction while sharing each packed word across a pair.
const VERIFY_EXPERT_REUSE_SOURCE =
    \\auto lane = thread_index_in_simdgroup;
    \\auto sg = simdgroup_index_in_threadgroup;
    \\auto tg = threadgroup_position_in_grid;
    \\uint row = tg.z;
    \\uint eid = ids[row];
    \\uint first = row;
    \\while (first > 0 && ids[first - 1] == eid) --first;
    \\if ((row - first) % 2 != 0) return;
    \\bool pair = row + 1 < uint(ids_shape[0]) && ids[row + 1] == eid;
    \\uint out_row = tg.y * 8 + sg * 4;
    \\float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
    \\constexpr uint VPT = 8;
    \\for (uint k = 0; k < K; k += VPT * 32) {
    \\  uint offset = k + lane * VPT;
    \\  if (offset >= K) continue;
    \\  float2 xt[VPT];
    \\  float2 sum = float2(0);
    \\  for (uint i = 0; i < VPT; i += 4) {
    \\    size_t p = size_t(row) * K + offset + i;
    \\    T a0 = x[p], a1 = x[p+1], a2 = x[p+2], a3 = x[p+3];
    \\    T b0 = T(0), b1 = T(0), b2 = T(0), b3 = T(0);
    \\    if (pair) { b0 = x[p+K]; b1 = x[p+K+1]; b2 = x[p+K+2]; b3 = x[p+K+3]; }
    \\    sum.x += a0 + a1 + a2 + a3;
    \\    sum.y += b0 + b1 + b2 + b3;
    \\    xt[i] = float2(float(a0), float(b0));
    \\    xt[i+1] = float2(float(a1), float(b1)) / 16.0f;
    \\    xt[i+2] = float2(float(a2), float(b2)) / 256.0f;
    \\    xt[i+3] = float2(float(a3), float(b3)) / 4096.0f;
    \\  }
    \\  for (uint r = 0; r < 4; ++r) {
    \\    size_t wr = size_t(eid) * N + out_row + r;
    \\    uint packed[VPT/8];
    \\    for (uint i = 0; i < VPT/8; ++i) packed[i] = w[wr * (K/8) + offset/8 + i];
    \\    float scale = float(sc[wr * (K/64) + offset/64]);
    \\    float bias = float(bi[wr * (K/64) + offset/64]);
    \\    float2 accum = float2(0);
    \\    for (uint i = 0; i < VPT/4; ++i) {
    \\      uint ws = (packed[i/2] >> (16*(i%2))) & 65535;
    \\      accum += (xt[4*i] * float(ws & 0x000f) + xt[4*i+1] * float(ws & 0x00f0) + xt[4*i+2] * float(ws & 0x0f00) + xt[4*i+3] * float(ws & 0xf000));
    \\    }
    \\    result[r] += scale * accum + sum * bias;
    \\  }
    \\}
    \\for (uint r = 0; r < 4; ++r) {
    \\  float a = simd_sum(result[r].x);
    \\  float b = simd_sum(result[r].y);
    \\  if (lane == 0) {
    \\    y[size_t(row) * N + out_row + r] = T(a);
    \\    if (pair) y[size_t(row+1) * N + out_row + r] = T(b);
    \\  }
    \\}
;

var verify_expert_reuse_kernel: ?mlx.mlx_fast_metal_kernel = null;

const ExpertReuseKey = struct { rows: c_int, n: c_int, k: c_int };
var verify_expert_reuse_cfgs = QsaCfgCache(ExpertReuseKey, 1){};

pub fn verifyExpertReuse(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null or ids.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    const is = mlx.getShape(ids);
    if (xs.len != 3 or xs[1] != 1 or ws.len != 3 or ss.len != 3 or !std.mem.eql(c_int, ss, bs) or is.len != 1 or is[0] != xs[0]) return null;
    const count = xs[0];
    const k = xs[2];
    const n = ws[1];
    if (count < 2 or count > 320 or ws[0] < 1 or @divTrunc(count, ws[0]) >= 4) return null;
    if (k != 640 or n != 2560) return null;
    if (ws[2] != @divExact(k, 8) or ss[0] != ws[0] or ss[1] != n or ss[2] != @divExact(k, 64)) return null;
    const kernel = blk: {
        if (verify_expert_reuse_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "ids" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reuse", iv, ov, VERIFY_EXPERT_REUSE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reuse_kernel = value;
        break :blk value;
    };
    const key = ExpertReuseKey{ .rows = count, .n = n, .k = k };
    const cfg = blk: {
        if (verify_expert_reuse_cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ count, 1, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @divExact(n, 4), count));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "N", n));
        verify_expert_reuse_cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, ids };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// Match the eight bf16 partial sums used by MLX col_reduce_small for top-10.
const VERIFY_EXPERT_REDUCE_SOURCE =
    \\uint p = thread_position_in_grid.x;
    \\uint row = p / 2560;
    \\uint col = p % 2560;
    \\T partial[8];
    \\#pragma unroll
    \\for (uint i = 0; i < 8; ++i) partial[i] = T(0);
    \\#pragma unroll
    \\for (uint e = 0; e < 10; ++e) {
    \\  uint at = row * 10 + e;
    \\  T product = down[size_t(inverse[at]) * 2560 + col] * scores[at];
    \\  partial[e % 8] = product + partial[e % 8];
    \\}
    \\T total = partial[0];
    \\#pragma unroll
    \\for (uint i = 1; i < 8; ++i) total = partial[i] + total;
    \\y[p] = total;
;
var verify_expert_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_expert_reduce_cfgs = QsaCfgCache(c_int, 1){};

pub fn verifyExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (down.ctx == null or inverse.ctx == null or scores.ctx == null) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const ix = mlx.getShape(inverse);
    if (ds.len != 2 or ds[1] != 2560 or ss.len != 3 or ss[0] != 1 or ss[2] != 10 or ix.len != 1) return null;
    const tokens = ss[1];
    if (tokens < 2 or tokens > 32 or ds[0] != tokens * 10 or ix[0] != ds[0]) return null;
    const kernel = blk: {
        if (verify_expert_reduce_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reduce", iv, ov, VERIFY_EXPERT_REDUCE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reduce_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (verify_expert_reduce_cfgs.get(tokens)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, tokens, 2560 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, tokens * 2560, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        verify_expert_reduce_cfgs.put(tokens, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ down, inverse, scores };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

// MLX affine qmv-fast reduction, with one 3-bit weight read serving two rows.
const MTP_COARSE_PAIRS_SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\auto tg = threadgroup_position_in_grid;
    \\int K = int(K_size), N = int(N_size);
    \\uint row = tg.z * 2;
    \\bool pair = row + 1 < uint(x_shape[0]);
    \\uint out_row = tg.y * 8 + sg * 4;
    \\const device uint8_t* w8 = (const device uint8_t*)w;
    \\float2 result[4] = {float2(0), float2(0), float2(0), float2(0)};
    \\for (int k = 0; k < K; k += 512) {
    \\  float2 xt[16];
    \\  float2 sum = float2(0);
    \\  for (uint i = 0; i < 16; i += 8) {
    \\    size_t at = size_t(row) * K + k + lane * 16 + i;
    \\    const device T* a = x + at;
    \\    sum.x += a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    \\    float b[8] = {0,0,0,0,0,0,0,0};
    \\    if (pair) {
    \\      const device T* v = a + K;
    \\      sum.y += v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
    \\      for (uint j = 0; j < 8; ++j) b[j] = float(v[j]);
    \\    }
    \\    xt[i] = float2(float(a[0]), b[0]);
    \\    xt[i+1] = float2(float(a[1]), b[1]) / 8.0f;
    \\    xt[i+2] = float2(float(a[2]), b[2]) / 64.0f;
    \\    xt[i+3] = float2(float(a[3]), b[3]) / 2.0f;
    \\    xt[i+4] = float2(float(a[4]), b[4]) / 16.0f;
    \\    xt[i+5] = float2(float(a[5]), b[5]) / 128.0f;
    \\    xt[i+6] = float2(float(a[6]), b[6]) / 4.0f;
    \\    xt[i+7] = float2(float(a[7]), b[7]) / 32.0f;
    \\  }
    \\  for (uint r = 0; r < 4; ++r) {
    \\    const device uint8_t* packed = w8 + size_t(out_row+r) * (K*3/8) + k*3/8 + lane*6;
    \\    float scale = float(sc[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    \\    float bias = float(bi[size_t(out_row+r) * (K/64) + k/64 + lane/4]);
    \\    float2 accum = float2(0);
    \\    for (uint i = 0; i < 2; ++i) {
    \\      uint a = packed[3*i], b = packed[3*i+1], c = packed[3*i+2];
    \\      accum += float(a & 0x07) * xt[8*i];
    \\      accum += float(a & 0x38) * xt[8*i+1];
    \\      accum += float(a & 0xc0) * xt[8*i+2];
    \\      accum += float(b & 0x01) * (xt[8*i+2] * 256.0f);
    \\      accum += float(b & 0x0e) * xt[8*i+3];
    \\      accum += float(b & 0x70) * xt[8*i+4];
    \\      accum += float(b & 0x80) * xt[8*i+5];
    \\      accum += float(c & 0x03) * (xt[8*i+5] * 256.0f);
    \\      accum += float(c & 0x1c) * xt[8*i+6];
    \\      accum += float(c & 0xe0) * xt[8*i+7];
    \\    }
    \\    result[r] += scale * accum + sum * bias;
    \\  }
    \\}
    \\for (uint r = 0; r < 4; ++r) {
    \\  float a = simd_sum(result[r].x), b = simd_sum(result[r].y);
    \\  if (lane == 0) {
    \\    y[size_t(row)*N + out_row+r] = T(a);
    \\    if (pair) y[size_t(row+1)*N + out_row+r] = T(b);
    \\  }
    \\}
;
var mtp_coarse_pairs_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mtp_coarse_pairs_cfgs = QsaCfgCache(c_int, 1){};

pub fn mtpCoarsePairs(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, gs: u32) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or bits != 3 or gs != 64) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] < 2 or xs[0] > 8 or xs[1] != 1 or xs[2] != 2560) return null;
    if (!std.mem.eql(c_int, ws, &.{ 248320, 240 }) or !std.mem.eql(c_int, ss, &.{ 248320, 40 }) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (mtp_coarse_pairs_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "K_size", "N_size" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("mtp_coarse_pairs", iv, ov, MTP_COARSE_PAIRS_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        mtp_coarse_pairs_kernel = value;
        break :blk value;
    };
    const cfg = blk: {
        if (mtp_coarse_pairs_cfgs.get(xs[0])) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ xs[0], 1, 248320 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, 248320 / 4, @divTrunc(xs[0] + 1, 2)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        mtp_coarse_pairs_cfgs.put(xs[0], .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, cachedScalarInt(2560), cachedScalarInt(248320) };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    trf.mtp_coarse_pair_dispatches +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] paired coarse head engaged (rows={d})\n", .{xs[0]});
    }
    return result;
}

// Preserve input order on ties so routing and the inverse permutation stay stable.
const VERIFY_ROUTE_PACK_SOURCE =
    \\threadgroup uint values[320];
    \\const uint n = uint(ids_shape[0]);
    \\const uint lane = thread_index_in_threadgroup;
    \\for (uint i = lane; i < n; i += 128) values[i] = ids[i];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint i = lane; i < n; i += 128) {
    \\  const uint key = values[i];
    \\  uint rank = 0;
    \\  for (uint j = 0; j < n; ++j) {
    \\    const uint other = values[j];
    \\    rank += uint(other < key || (other == key && j < i));
    \\  }
    \\  inverse[i] = rank;
    \\  sorted[rank] = key;
    \\  lhs[rank] = i / 10;
    \\}
;
var verify_route_pack_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_route_pack_configs = QsaCfgCache(c_int, 1){};
const VerifyRoutePack = struct {
    inverse: mlx.mlx_array = .{},
    sorted: mlx.mlx_array = .{},
    lhs: mlx.mlx_array = .{},
    pub fn deinit(self: *VerifyRoutePack) void {
        inline for (.{ "inverse", "sorted", "lhs" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }
};

pub fn verifyRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or top_k != 10 or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const shape = mlx.getShape(ids);
    if (shape.len != 1 or shape[0] < 20 or shape[0] > 320 or @mod(shape[0], 10) != 0) return null;
    const kernel = blk: {
        if (verify_route_pack_kernel) |kernel| break :blk kernel;
        const ins = [_][*:0]const u8{"ids"};
        const outs = [_][*:0]const u8{ "inverse", "sorted", "lhs" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kernel = mlx.mlx_fast_metal_kernel_new("verify_route_pack", iv, ov, VERIFY_ROUTE_PACK_SOURCE, "", true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        verify_route_pack_kernel = kernel;
        break :blk kernel;
    };
    const config = blk: {
        if (verify_route_pack_configs.get(shape[0])) |hit| break :blk hit[0];
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        verify_route_pack_configs.put(shape[0], .{config});
        break :blk config;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ids}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result: VerifyRoutePack = .{};
    errdefer result.deinit();
    inline for (.{ "inverse", "sorted", "lhs" }, 0..) |name, i| {
        @field(result, name) = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&@field(result, name), ov, i));
    }
    return result;
}

pub fn verifyIndexedExpertInput(s: mlx.mlx_stream, x: mlx.mlx_array, lhs: mlx.mlx_array, experts: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or experts <= 0) return null;
    if (x.ctx == null or lhs.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(lhs) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(lhs);
    if (xs.len != 2 or xs[1] != 2560 or xs[0] < 2 or xs[0] > 32 or ix.len != 1 or ix[0] != xs[0] * 10) return null;
    // Explicit LHS indices disable MLX's sorted matrix path; keep both arms below it.
    if (@divTrunc(ix[0], experts) >= 4) return null;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, x, &.{ xs[0], 1, xs[1] }, 3, s));
    return view;
}

pub fn verifySharedHardware() bool {
    if (!verifyQmmNaxAvailable()) return false;
    if (verify_shared_arch) |value| return value;
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse return false;
    const value = std.mem.startsWith(u8, arch, "applegpu_g17") and !std.mem.endsWith(u8, arch, "d");
    verify_shared_arch = value;
    return value;
}

pub const VerifyProjectionKind = enum { shared, gdn, lm_head, attention };

pub fn verifySharedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .shared);
}

pub fn verifyJoinedProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int, comptime kind: VerifyProjectionKind) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or
        mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    if (xs.len != 3 or xs[0] != 1 or ws.len != 2 or ss.len != 2 or bs.len != 2) return null;
    const k = xs[2];
    const n = ws[0];
    const wide = kind != .shared;
    const shape_ok = switch (kind) {
        .shared => (k == 2560 and n == 640) or (k == 640 and n == 2560),
        .gdn => (k == 2560 and (n == 10240 or n == 6144 or n == 48)) or (k == 6144 and n == 2560),
        .lm_head => k == 2560 and n == 248320,
        .attention => (k == 2560 and (n == 12288 or n == 512)) or (k == 6144 and n == 2560),
    };
    if (!shape_ok) return null;
    if (ws[1] != @divExact(k, 4) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, bs)) return null;
    // The joined path requires the reference qmv_wide K reduction.
    if (mixedPlainEnabled(8, 64, n)) return null;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 2 or width > (if (wide) @as(c_int, 6) else 7)) return null;
        if (wide and verifyQmmEnabled() and
            mixedNaxShapeEnabled(8, 64, n) and width >= naxMinMForBits(8) and @mod(k, 256) == 0) return null;
        total += width;
    }
    if (total != xs[1] or total > (if (wide) @as(c_int, 12) else 24)) return null;
    const value = (try @import("mtp_qmv.zig").matmul(s, x, w, sc, bi)) orelse return error.Affine8SerialProjectionUnsupported;
    trf.mtp_verify_kernel_calls[0] +%= 1;
    return value;
}

/// One row per slot's decode tick, any affine8 group size: each row is computed as the single-row qmv computes it.
pub fn decodeRowsProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or widths.len < 2 or widths.len > 8) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] != @as(c_int, @intCast(widths.len)) or ws.len != 2 or ss.len != 2 or ss[0] != ws[0]) return null;
    for (widths) |width| if (width != 1) return null;
    const geom = @import("expert_quant.zig").affineGeomFromShapes(@intCast(ws[1]), @intCast(ss[1]), @intCast(xs[2])) orelse return null;
    if (geom.bits != 8 or mixedPlainEnabled(8, geom.group_size, ws[0])) return null;
    return @import("mtp_qmv.zig").matmul(s, x, w, sc, bi);
}

pub fn qmatmulBits(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, s: mlx.mlx_stream) !mlx.mlx_array {
    // Unquantized weights are pre-transposed to [in, out] at load.
    // Short forwards retain each serial row's matmul arithmetic.
    if (sc.ctx == null) {
        if (try @import("mtp_qmv.zig").denseMatmul(s, x, w)) |out| return out;
        var fp_result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&fp_result, x, w, s));
        return fp_result;
    }

    // Non-affine model-wide mode (mxfp4 / mxfp8 from config.json):
    // bias-less by construction; bits/group_size come from the config like
    // affine. Checked BEFORE the legacy null-bias heuristic below, which
    // would misroute e.g. an mxfp4 weight to mxfp8.
    if (mode != .affine) {
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            mlx.mlx_array{ .ctx = null },
            true,
            mlx.mlx_optional_int.some(@intCast(group_size)),
            mlx.mlx_optional_int.some(@intCast(bits)),
            mode.cstr(),
            s,
        ));
        return result;
    }

    // Legacy heuristic for mixed checkpoints whose config declares affine but
    // whose individual tensors lack biases (NVIDIA mxfp8 layers): biases array
    // has null ctx (created with mlx_array_new()).
    const is_mxfp = bi.ctx == null;

    if (is_mxfp) {
        // mxfp8: bits is always 8; infer group_size from scales/weight shape ratio
        const mxfp_bits: u32 = 8;
        const s_shape = mlx.getShape(sc);
        const w_shape = mlx.getShape(w);
        const mxfp_gs: u32 = if (s_shape.len >= 2 and w_shape.len >= 2) blk: {
            const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
            const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
            if (s_cols > 0) break :blk (w_cols * 32) / (s_cols * mxfp_bits);
            break :blk 32;
        } else 32;

        const null_bi = mlx.mlx_array{ .ctx = null };
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(
            &result,
            x,
            w,
            sc,
            null_bi,
            true,
            mlx.mlx_optional_int.some(@intCast(mxfp_gs)),
            mlx.mlx_optional_int.some(@intCast(mxfp_bits)),
            "mxfp8",
            s,
        ));
        return result;
    }

    if (bits == 8 and mode == .affine) {
        const shape = mlx.getShape(x);
        if (shape.len >= 2 and shape[shape.len - 1] > 0) {
            const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(shape[shape.len - 1]));
            if (rows >= 2 and rows <= 32)
                return (try @import("mtp_qmv.zig").matmul(s, x, w, sc, bi)) orelse error.Affine8SerialProjectionUnsupported;
        }
    }

    // Spec-verify fast path: M=2..6-row activations (verify forwards, small
    // decode batches) route through the split-K verify kernel when eligible
    // (see VERIFY_QMM_SOURCE) — stock qmm's qmv/steel dead zone.
    if (try verifyQmm(s, x, w, sc, bi, bits, group_size)) |vy| return vy;

    // Prefill-width fast path: M >= 2048 dequantizes + runs the steel bf16
    // GEMM (see prefillDqGemm) — stock qmm_t's 32x32x32 tile dead zone.
    if (try prefillDqGemm(x, w, sc, bi, bits, group_size, s)) |py| return py;

    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_quantized_matmul(
        &result,
        x,
        w,
        sc,
        bi,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));
    return result;
}

/// The dequant+GEMM prefill route (see the block comment above
/// prefillDqGemmEnabled). Returns null when the call must stay on stock qmm.
fn prefillDqGemm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!prefillDqGemmEnabled()) return null;
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return null,
    }
    const last = lastDim(x) orelse return null;
    if (last == 0) return null;
    const rows = mlx.mlx_array_size(x) / @as(usize, @intCast(last));
    if (rows < PREFILL_DQ_GEMM_MIN_M) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;

    var dq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq);
    try mlx.check(mlx.mlx_dequantize(&dq, w, sc, bi, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = xd, .has_value = true }, s));
    var dq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dq_t);
    try mlx.check(mlx.mlx_transpose(&dq_t, dq, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, dq_t, s));
    prefill_dq_gemm_engaged += 1;
    return out;
}

fn expertReduceReference(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !mlx.mlx_array {
    const tokens = mlx.getShape(scores)[1];
    var unsorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unsorted);
    try mlx.check(mlx.mlx_take_axis(&unsorted, down, inverse, 0, s));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_reshape(&rows, unsorted, &.{ 1, tokens, 10, 2560 }, 4, s));
    var expanded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expanded);
    try mlx.check(mlx.mlx_expand_dims(&expanded, scores, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, rows, expanded, s));
    var expected = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(expected);
    try mlx.check(mlx.mlx_sum_axis(&expected, weighted, -2, false, s));
    return expected;
}

fn testAffine8ProjectionRows(gs: u32) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x8dec0de);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| {
        for ([_][2]c_int{ .{ 12288, 2560 }, .{ 512, 2560 }, .{ 2560, 6144 }, .{ 2560, 640 }, .{ 10240, 384 }, .{ 7, 256 }, .{ 8, 128 }, .{ 1, 128 }, .{ 13, 512 }, .{ 9, @intCast(gs) }, .{ 8, @intCast(gs * 3) }, .{ 8, @intCast(gs * 9) } }) |shape| {
            const n = shape[0];
            const k = shape[1];
            const bf = try testRandWeightBf16(prng.random(), &shape, s);
            defer _ = mlx.mlx_array_free(bf);
            var dense = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dense);
            try mlx.check(mlx.mlx_astype(&dense, bf, dtype, s));
            var weights = try kv_quant.quantizeAffine(s, dense, gs, 8);
            defer weights.deinit();
            for ([_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32 }) |rows| {
                const x = try qkvIdentityArray(s, 1, @intCast(rows), k, 0.039 * @as(f32, @floatFromInt(rows)), dtype);
                defer _ = mlx.mlx_array_free(x);
                const together = try qmatmulBits(x, weights.q, weights.scales, weights.biases, 8, gs, .affine, s);
                defer _ = mlx.mlx_array_free(together);
                for (0..rows) |row| {
                    const begin: c_int = @intCast(row);
                    const xr = try sliceAttentionSeq(s, x, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(xr);
                    const single = try qmatmulBits(xr, weights.q, weights.scales, weights.biases, 8, gs, .affine, s);
                    defer _ = mlx.mlx_array_free(single);
                    const candidate = try sliceAttentionSeq(s, together, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(candidate);
                    qkvExpectBitsEqual(s, single, candidate) catch |err| {
                        std.debug.print("affine8 gs={d} N={d} K={d} rows={d} row={d} dtype={s}\n", .{ gs, n, k, rows, row, @tagName(dtype) });
                        return err;
                    };
                }
            }
        }
    }
}

test "qmatmulBits dispatches to plain matmul when scales has null ctx (Unsloth Dynamic bf16)" {
    const s = mlx.gpuStream();

    // x: [1, 1, in=4], w: [out=2, in=4] (PyTorch convention).
    // Expected: x @ w.T = [1, 1, 2]
    var x_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_flat);
    try mlx.check(mlx.mlx_arange(&x_flat, 0.0, 4.0, 1.0, .float32, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    {
        const sh = [_]c_int{ 1, 1, 4 };
        try mlx.check(mlx.mlx_reshape(&x, x_flat, &sh, 3, s));
    }

    var w_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_flat);
    try mlx.check(mlx.mlx_arange(&w_flat, 0.0, 8.0, 1.0, .float32, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    {
        const sh = [_]c_int{ 2, 4 };
        try mlx.check(mlx.mlx_reshape(&w, w_flat, &sh, 2, s));
    }

    // Pre-transpose like initMoeLayers does for null-scales weights: [out, in] → [in, out]
    const w_t = try transposeBf16Weight(w, s);
    defer _ = mlx.mlx_array_free(w_t);

    // qmatmulBits with null sc/bi must plain-matmul x @ w_t.
    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const got = try qmatmulBits(x, w_t, null_sc, null_bi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(got);

    // Reduce to host floats for comparison.
    var got_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_f32);
    try mlx.check(mlx.mlx_astype(&got_f32, got, .float32, s));
    var got_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got_flat);
    {
        const sh = [_]c_int{2};
        try mlx.check(mlx.mlx_reshape(&got_flat, got_f32, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, got_flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const data = mlx.mlx_array_data_float32(got_flat) orelse return error.TestUnexpectedNullData;
    // [0,1,2,3] @ [[0,4],[1,5],[2,6],[3,7]] = [0+1*1+2*2+3*3, 0+1*5+2*6+3*7] = [14, 38]
    // (w transposed from [[0,1,2,3],[4,5,6,7]] gives w_t = [[0,4],[1,5],[2,6],[3,7]])
    try testing.expectApproxEqAbs(@as(f32, 14.0), data[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 38.0), data[1], 1e-3);
}

test "verify shared projection preserves solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 24) {
                try testing.expect((try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) == null);
                continue;
            }
            const grouped = (try verifySharedProjection(s, x, w, sc, bi, lengths[0..count])) orelse return error.SharedProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifySharedProjection(s, x, w, sc, bi, widths)) == null);
        }
        try testing.expect((try verifySharedProjection(s, x, w, sc, .{}, &.{ 4, 4 })) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifySharedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 })) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifySharedProjection(s, x, w, sc, bi, &.{ 4, 4 })) == null);
    }
}

test "verify GDN projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 10240 }, .{ 2560, 6144 }, .{ 2560, 48 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .gdn)) == null);
                continue;
            }
            const grouped = (try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .gdn)) orelse return error.GdnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, widths, .gdn)) == null);
        }
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, .{}, &.{ 4, 4 }, .gdn)) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyJoinedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 }, .gdn)) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, &.{ 4, 4 }, .gdn)) == null);
    }
}

test "verify attention projections preserve solo qmv values across request widths" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5A4E);
    for ([_][2]c_int{ .{ 2560, 12288 }, .{ 2560, 512 }, .{ 6144, 2560 } }) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]usize{ 2, 3, 4, 8 }) |count| for ([_][8]c_int{ @splat(2), @splat(3), @splat(4), @splat(5), @splat(6), @splat(7), .{ 5, 3, 2, 4, 2, 2, 3, 3 } }) |lengths| {
            var total: c_int = 0;
            for (lengths[0..count]) |width| total += width;
            const x = try attn256RandBf16(prng.random(), &.{ 1, total, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (total > 12 or lengths[0] > 6) {
                try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .attention)) == null);
                continue;
            }
            const grouped = (try verifyJoinedProjection(s, x, w, sc, bi, lengths[0..count], .attention)) orelse return error.AttnProjectionDeclined;
            defer _ = mlx.mlx_array_free(grouped);
            var offset: c_int = 0;
            for (lengths[0..count]) |width| {
                var row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(row);
                try mlx.check(mlx.mlx_slice(&row, x, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, k }, 3, &.{ 1, 1, 1 }, 3, s));
                const expected = try qmatmulBits(row, w, sc, bi, 8, 64, .affine, s);
                defer _ = mlx.mlx_array_free(expected);
                var actual = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(actual);
                try mlx.check(mlx.mlx_slice(&actual, grouped, &.{ 0, offset, 0 }, 3, &.{ 1, offset + width, n }, 3, &.{ 1, 1, 1 }, 3, s));
                try testing.expect(try qsaArraysAllEqual(expected, actual, s));
                offset += width;
            }
        };
        const x = try attn256RandBf16(prng.random(), &.{ 1, 8, k }, s);
        defer _ = mlx.mlx_array_free(x);
        for ([_][]const c_int{ &.{ 1, 7 }, &.{8}, &.{ 2, 2 }, &.{ 0, 8 } }) |widths| {
            try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, widths, .attention)) == null);
        }
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, .{}, &.{ 4, 4 }, .attention)) == null);
        const wide = try attn256RandBf16(prng.random(), &.{ 1, 25, k }, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expect((try verifyJoinedProjection(s, wide, w, sc, bi, &.{ 5, 5, 5, 5, 5 }, .attention)) == null);
        const old = verify_qmm.vqmm_mixed_plain_test_force;
        verify_qmm.vqmm_mixed_plain_test_force = true;
        defer verify_qmm.vqmm_mixed_plain_test_force = old;
        try testing.expect((try verifyJoinedProjection(s, x, w, sc, bi, &.{ 4, 4 }, .attention)) == null);
    }
}

test "qmatmulBits uses canonical arithmetic for plain batched projections" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const saved_kernel = moe_gqmv.sushi_qmv_rows_kernel;
    moe_gqmv.sushi_qmv_rows_kernel = null;
    defer {
        if (moe_gqmv.sushi_qmv_rows_kernel) |kernel| _ = mlx.mlx_fast_metal_kernel_free(kernel);
        moe_gqmv.sushi_qmv_rows_kernel = saved_kernel;
    }
    var prng = std.Random.DefaultPrng.init(0xB47C4);
    const w_bf = try attn256RandBf16(prng.random(), &.{ 512, 2560 }, s);
    defer _ = mlx.mlx_array_free(w_bf);
    for ([_]u32{ 4, 8 }) |bits| {
        var qvec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(qvec);
        try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
        for ([_]c_int{ 2, 4, 5, 9 }) |rows| {
            const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
            defer _ = mlx.mlx_array_free(x);
            const y = try qmatmulBits(x, w, sc, bi, bits, 64, .affine, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
            if (moe_gqmv.sushi_qmv_rows_kernel != null) std.debug.print("generic projection invoked MTP row-axis kernel: rows={d} bits={d}\n", .{ rows, bits });
            try testing.expect(moe_gqmv.sushi_qmv_rows_kernel == null);
            const expected = if (bits == 8) blk: {
                var parts: [9]mlx.mlx_array = @splat(.{});
                var count: usize = 0;
                defer for (parts[0..count]) |array| {
                    _ = mlx.mlx_array_free(array);
                };
                var row: c_int = 0;
                while (row < rows) : (row += 1) {
                    var xr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr);
                    try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ row, 0 }, 2, &[_]c_int{ row + 1, 2560 }, 2, &[_]c_int{ 1, 1 }, 2, s));
                    parts[count] = mlx.mlx_array_new();
                    count += 1;
                    try mlx.check(mlx.mlx_quantized_matmul(&parts[count - 1], xr, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
                }
                const vec = mlx.mlx_vector_array_new_data(&parts, count);
                defer _ = mlx.mlx_vector_array_free(vec);
                var result = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(result);
                try mlx.check(mlx.mlx_concatenate_axis(&result, vec, 0, s));
                break :blk result;
            } else (try verifyQmm(s, x, w, sc, bi, bits, 64)) orelse blk: {
                var stock = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&stock, x, w, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
                break :blk stock;
            };
            defer _ = mlx.mlx_array_free(expected);
            try testing.expect(try qsaArraysAllEqual(y, expected, s));
        }
    }
}

test "qmatmulBits: prefill-width affine calls take the dequant+GEMM route (engaged + no worse than stock)" {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xD0DE);
    const rnd = prng.random();
    const K: c_int = 256;
    const N: c_int = 384;
    const M: c_int = @intCast(PREFILL_DQ_GEMM_MIN_M);

    // Random bf16 weight quantized to affine q4 gs64 (the 27B trunk class).
    const wn: usize = @intCast(N * K);
    const wbuf = try al.alloc(f32, wn);
    for (wbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const wshape = [_]c_int{ N, K };
    const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 2, .float32);
    al.free(wbuf);
    defer _ = mlx.mlx_array_free(w32);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    const xn: usize = @intCast(M * K);
    const xbuf = try al.alloc(f32, xn);
    for (xbuf) |*v| v.* = bf16Trunc(rnd.float(f32) - 0.5);
    const xshape = [_]c_int{ 1, M, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
    al.free(xbuf);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    // f32 ground truth: fp32 dequant + fp32 GEMM.
    var wdq32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32);
    try mlx.check(mlx.mlx_dequantize(&wdq32, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .value = .float32, .has_value = true }, s));
    var wdq32_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wdq32_t);
    try mlx.check(mlx.mlx_transpose(&wdq32_t, wdq32, s));
    var gt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gt);
    try mlx.check(mlx.mlx_matmul(&gt, x32, wdq32_t, s));
    try mlx.check(mlx.mlx_array_eval(gt));
    const gtd = mlx.mlx_array_data_float32(gt) orelse return error.InvalidDtype;
    const on: usize = @intCast(M * N);

    // Stock qmm (route forced off).
    prefill_dq_gemm_override = false;
    const stock = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(stock);
    const stock_f = try evalToF32(al, stock, on, s);
    defer al.free(stock_f);

    // Routed (default on): engagement must be COUNTED.
    prefill_dq_gemm_override = true;
    defer prefill_dq_gemm_override = null;
    const before = prefill_dq_gemm_engaged;
    const routed = try qmatmulBits(x, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(routed);
    try testing.expectEqual(before + 1, prefill_dq_gemm_engaged);
    const routed_f = try evalToF32(al, routed, on, s);
    defer al.free(routed_f);

    var stock_err: f32 = 0;
    var routed_err: f32 = 0;
    for (0..on) |i| {
        stock_err = @max(stock_err, @abs(stock_f[i] - gtd[i]));
        routed_err = @max(routed_err, @abs(routed_f[i] - gtd[i]));
    }
    // bf16-weight-rounding is the only extra error source — same magnitude
    // class as the stock kernel's own bf16 output rounding.
    try testing.expect(routed_err <= 1.5 * stock_err + 5e-3);

    // Decode/verify widths never route (M below the floor).
    var x8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x8);
    {
        var x8_32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x8_32);
        const s8 = [_]c_int{ 0, 0, 0 };
        const e8 = [_]c_int{ 1, 8, K };
        const st8 = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&x8_32, x32, &s8, 3, &e8, 3, &st8, 3, s));
        try mlx.check(mlx.mlx_astype(&x8, x8_32, .bfloat16, s));
    }
    const small_before = prefill_dq_gemm_engaged;
    const small = try qmatmulBits(x8, wq, wsc, wbi, 4, 64, .affine, s);
    defer _ = mlx.mlx_array_free(small);
    try mlx.check(mlx.mlx_array_eval(small));
    try testing.expectEqual(small_before, prefill_dq_gemm_engaged);
}

test "verify expert reuse preserves stock gather reductions" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE4E4);
    for ([_][2]c_int{.{ 640, 2560 }}) |shape| {
        const k = shape[0];
        const n = shape[1];
        const weight = try attn256RandBf16(prng.random(), &.{ 64, n, k }, s);
        defer _ = mlx.mlx_array_free(weight);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        for ([_]c_int{ 20, 30, 40, 50, 60, 80, 200 }) |count| for (0..3) |pattern| {
            var host_ids: [200]u32 = undefined;
            var expert: u32 = 0;
            var remain: usize = 0;
            for (host_ids[0..@intCast(count)], 0..) |*id, i| {
                if (remain == 0) {
                    if (i != 0) expert += 1;
                    remain = switch (pattern) {
                        0 => @intCast(@divTrunc(count + 63, 64)),
                        1 => 4,
                        else => ([_]usize{ 1, 3, 6 })[expert % 3],
                    };
                }
                id.* = expert;
                remain -= 1;
            }
            const ids = mlx.mlx_array_new_data(&host_ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const x = try attn256RandBf16(prng.random(), &.{ count, 1, k }, s);
            defer _ = mlx.mlx_array_free(x);
            if (count == 20 and pattern == 0) {
                try testing.expect((try verifyExpertReuse(s, x, w, sc, .{}, ids)) == null);
                const cpu = mlx.mlx_default_cpu_stream_new();
                defer _ = mlx.mlx_stream_free(cpu);
                try testing.expect((try verifyExpertReuse(cpu, x, w, sc, bi, ids)) == null);
                var x32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32);
                try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
                try testing.expect((try verifyExpertReuse(s, x32, w, sc, bi, ids)) == null);
                var signed_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(signed_ids);
                try mlx.check(mlx.mlx_astype(&signed_ids, ids, .int32, s));
                try testing.expect((try verifyExpertReuse(s, x, w, sc, bi, signed_ids)) == null);
                var large_x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_x);
                var large_ids = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(large_ids);
                try mlx.check(mlx.mlx_zeros(&large_x, &.{ 256, 1, k }, 3, .bfloat16, s));
                try mlx.check(mlx.mlx_zeros(&large_ids, &.{256}, 1, .uint32, s));
                try testing.expect((try verifyExpertReuse(s, large_x, w, sc, bi, large_ids)) == null);
            }
            var expected = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(expected);
            try gatherExpertMm(&expected, x, w, sc, bi, .{}, ids, 4, 64, .affine, true, s);
            const actual = (try verifyExpertReuse(s, x, w, sc, bi, ids)) orelse return error.ExpertReuseDeclined;
            defer _ = mlx.mlx_array_free(actual);
            const same = try qsaArraysAllEqual(expected, actual, s);
            if (!same) std.debug.print("[expert-reuse-mismatch] K={d} N={d} count={d} pattern={d}\n", .{ k, n, count, pattern });
            try testing.expect(same);
        };
    }
}

test "verify expert reduction preserves unsort multiply and sum" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xE55E);
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 10, 12, 20, 32 }) |tokens| {
        const count = tokens * 10;
        const down = try attn256RandBf16(prng.random(), &.{ count, 2560 }, s);
        defer _ = mlx.mlx_array_free(down);
        const scores = try attn256RandBf16(prng.random(), &.{ 1, tokens, 10 }, s);
        defer _ = mlx.mlx_array_free(scores);
        var perm: [320]u32 = undefined;
        for (perm[0..@intCast(count)], 0..) |*id, i| id.* = @intCast(i);
        prng.random().shuffle(u32, perm[0..@intCast(count)]);
        const inverse = mlx.mlx_array_new_data(&perm, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(inverse);
        if (tokens == 4) {
            try testing.expect((try verifyExpertReduce(s, down, inverse, .{})) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyExpertReduce(cpu, down, inverse, scores)) == null);
            var down32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(down32);
            try mlx.check(mlx.mlx_astype(&down32, down, .float32, s));
            try testing.expect((try verifyExpertReduce(s, down32, inverse, scores)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, inverse, .int32, s));
            try testing.expect((try verifyExpertReduce(s, down, signed, scores)) == null);
            var batched = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(batched);
            try mlx.check(mlx.mlx_reshape(&batched, scores, &.{ 2, 2, 10 }, 3, s));
            try testing.expect((try verifyExpertReduce(s, down, inverse, batched)) == null);
        }
        const expected = try expertReduceReference(s, down, inverse, scores);
        defer _ = mlx.mlx_array_free(expected);
        const actual = (try verifyExpertReduce(s, down, inverse, scores)) orelse return error.ExpertReduceDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
    }
}

test "verify route packing preserves stable order and row indices" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var ids: [320]u32 = undefined;
    for ([_]c_int{ 20, 30, 40, 50, 60, 80, 120, 200, 320 }) |count| {
        for (0..3) |pattern| {
            for (ids[0..@intCast(count)], 0..) |*id, i| id.* = switch (pattern) {
                0 => @intCast((i * 37 + 11) % 17),
                1 => 7,
                else => @intCast(@as(usize, @intCast(count)) - i),
            };
            const input = mlx.mlx_array_new_data(&ids, &.{count}, 1, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyRoutePack(cpu, input, 10)) == null);
            try testing.expect((try verifyRoutePack(s, input, 8)) == null);
            try testing.expect((try verifyRoutePack(s, .{}, 10)) == null);
            var got = (try verifyRoutePack(s, input, 10)) orelse return error.RoutePackDeclined;
            defer got.deinit();
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            var inverse = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(inverse);
            var sorted = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted);
            var lhs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(lhs);
            try mlx.check(mlx.mlx_argsort_axis(&order, input, 0, s));
            try mlx.check(mlx.mlx_argsort_axis(&inverse, order, 0, s));
            try mlx.check(mlx.mlx_take_axis(&sorted, input, order, 0, s));
            try mlx.check(mlx.mlx_floor_divide(&lhs, order, cachedScalarInt(10), s));
            try testing.expect(try qsaArraysAllEqual(got.inverse, inverse, s));
            try testing.expect(try qsaArraysAllEqual(got.sorted, sorted, s));
            try testing.expect(try qsaArraysAllEqual(got.lhs, lhs, s));
        }
    }
}

test "verify indexed expert inputs preserve projections without activation copies" {
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x1D1E);
    const weight = try attn256RandBf16(prng.random(), &.{ 64, 640, 2560 }, s);
    defer _ = mlx.mlx_array_free(weight);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, weight, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    for ([_]c_int{ 2, 3, 4, 5, 6, 8, 20 }) |tokens| {
        const count = tokens * 10;
        const x = try attn256RandBf16(prng.random(), &.{ tokens, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        var left: [200]u32 = undefined;
        var right: [200]u32 = undefined;
        for (left[0..@intCast(count)], right[0..@intCast(count)], 0..) |*l, *r, i| {
            l.* = @intCast(i / 10);
            r.* = @intCast(i / @as(usize, @intCast(@divTrunc(count + 63, 64))));
        }
        prng.random().shuffle(u32, left[0..@intCast(count)]);
        const lhs = mlx.mlx_array_new_data(&left, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(lhs);
        const rhs = mlx.mlx_array_new_data(&right, &.{count}, 1, .uint32);
        defer _ = mlx.mlx_array_free(rhs);
        const start_copy = mlx.op_count.load(.monotonic);
        var copied = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(copied);
        try mlx.check(mlx.mlx_take_axis(&copied, x, lhs, 0, s));
        var expanded = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expanded);
        try mlx.check(mlx.mlx_reshape(&expanded, copied, &.{ count, 1, 2560 }, 3, s));
        const copy_ops = mlx.op_count.load(.monotonic) - start_copy;
        const start_view = mlx.op_count.load(.monotonic);
        const view = (try verifyIndexedExpertInput(s, x, lhs, 64)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(view);
        const view_ops = mlx.op_count.load(.monotonic) - start_view;
        try testing.expectEqualSlices(c_int, &.{ tokens, 1, 2560 }, mlx.getShape(view));
        try testing.expect(view_ops < copy_ops);
        var expected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expected);
        try gatherExpertMm(&expected, expanded, w, sc, bi, .{}, rhs, 4, 64, .affine, true, s);
        var actual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(actual);
        try gatherExpertMm(&actual, view, w, sc, bi, lhs, rhs, 4, 64, .affine, true, s);
        try testing.expect(try qsaArraysAllEqual(expected, actual, s));
        // The sorted-matrix boundary is four assignments per expert. Ten
        // experts crosses it at width 4, but remains eligible at widths 2–3.
        const boundary_experts = @divTrunc(count, 4);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, boundary_experts)) == null);
        const sparse_view = (try verifyIndexedExpertInput(s, x, lhs, boundary_experts + 1)) orelse return error.IndexedExpertInputDeclined;
        defer _ = mlx.mlx_array_free(sparse_view);
        try testing.expect((try verifyIndexedExpertInput(s, x, lhs, 0)) == null);
        if (tokens == 4) {
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try verifyIndexedExpertInput(cpu, x, lhs, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x32, lhs, 64)) == null);
            var signed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(signed);
            try mlx.check(mlx.mlx_astype(&signed, lhs, .int32, s));
            try testing.expect((try verifyIndexedExpertInput(s, x, signed, 64)) == null);
            try testing.expect((try verifyIndexedExpertInput(s, view, lhs, 64)) == null);
        }
    }
}

test "MTP coarse pairs preserve row-kernel logits (MTP_LMHEAD_TEST_FILE)" {
    const path = std.c.getenv("MTP_LMHEAD_TEST_FILE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !verifySharedHardware()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var weights = try model_mod.loadWeightsSingleFile(testing.allocator, std.mem.span(path));
    defer weights.deinit();
    const w = weights.get("language_model.lm_head.weight") orelse return error.MissingLmHead;
    const sc = weights.get("language_model.lm_head.scales") orelse return error.MissingLmHead;
    const bi = weights.get("language_model.lm_head.biases") orelse return error.MissingLmHead;
    var coarse = try mtp_mod.requantizeRows(s, w, sc, bi, 64, 8, "affine", 64, 3, 32768);
    defer coarse.deinit();
    var prng = std.Random.DefaultPrng.init(0xC0A45E);
    for ([_]c_int{ 2, 3, 4, 8 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 1, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const reference = (try msvQmvRows(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(reference);
        const actual = (try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 3, 64)) orelse return error.CoarsePairsDeclined;
        defer _ = mlx.mlx_array_free(actual);
        try testing.expect(try qsaArraysAllEqual(reference, actual, s));
        if (rows == 4) {
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, .{}, 3, 64)) == null);
            try testing.expect((try mtpCoarsePairs(s, x, coarse.w, coarse.s, coarse.b, 2, 64)) == null);
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try testing.expect((try mtpCoarsePairs(cpu, x, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var wide = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wide);
            try mlx.check(mlx.mlx_reshape(&wide, x, &.{ 2, 2, 2560 }, 3, s));
            try testing.expect((try mtpCoarsePairs(s, wide, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
            var x32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32);
            try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
            try testing.expect((try mtpCoarsePairs(s, x32, coarse.w, coarse.s, coarse.b, 3, 64)) == null);
        }
    }
}

test "affine8 projections are row-identical gs32" {
    try testAffine8ProjectionRows(32);
}

test "affine8 projections are row-identical gs64" {
    try testAffine8ProjectionRows(64);
}

test "affine8 projections are row-identical gs128" {
    try testAffine8ProjectionRows(128);
}

test "dense projections are row-identical to serial matmul" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xded8);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| {
        for ([_][2]c_int{ .{ 512, 2560 }, .{ 1, 2560 }, .{ 4, 10240 }, .{ 2560, 640 } }) |shape| {
            const bf = try testRandWeightBf16(prng.random(), &shape, s);
            defer _ = mlx.mlx_array_free(bf);
            var dense = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(dense);
            try mlx.check(mlx.mlx_astype(&dense, bf, dtype, s));
            const w = try transposeBf16Weight(dense, s);
            defer _ = mlx.mlx_array_free(w);
            const k = shape[1];
            for ([_]usize{ 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 32 }) |rows| {
                const x = try qkvIdentityArray(s, 1, @intCast(rows), k, 0.039 * @as(f32, @floatFromInt(rows)), dtype);
                defer _ = mlx.mlx_array_free(x);
                const together = try qmatmulBits(x, w, .{}, .{}, 0, 64, .affine, s);
                defer _ = mlx.mlx_array_free(together);
                for (0..rows) |row| {
                    const begin: c_int = @intCast(row);
                    const xr = try sliceAttentionSeq(s, x, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(xr);
                    const single = try qmatmulBits(xr, w, .{}, .{}, 0, 64, .affine, s);
                    defer _ = mlx.mlx_array_free(single);
                    const candidate = try sliceAttentionSeq(s, together, begin, begin + 1);
                    defer _ = mlx.mlx_array_free(candidate);
                    qkvExpectBitsEqual(s, single, candidate) catch |err| {
                        std.debug.print("dense N={d} K={d} rows={d} row={d} dtype={s}\n", .{ shape[0], k, rows, row, @tagName(dtype) });
                        return err;
                    };
                }
            }
        }
    }
}
