const trf = @import("../transformer.zig");
const moe_gqmv = @import("../moe_gqmv.zig");
const nax_gate = @import("../nax_gate.zig");
const std = @import("std");
const mlx = @import("../mlx.zig");
const mrope = @import("../mrope.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const QSA_SELECT_SPLIT_MAX_ROWS = qwen4_qsa.QSA_SELECT_SPLIT_MAX_ROWS;
const qsaAllBlocksVisible = qwen4_qsa.qsaAllBlocksVisible;
const qsaDumpQsaBlocks = qwen4_qsa.qsaDumpQsaBlocks;
const qsaScoreFused = qwen4_qsa.qsaScoreFused;
const qsaScoreFusedEligible = qwen4_qsa.qsaScoreFusedEligible;
const qsaSelectComposedOps = qwen4_qsa.qsaSelectComposedOps;
const qsaSelectDigitBits = qwen4_qsa.qsaSelectDigitBits;
const qsaSelectSplitServes = qwen4_qsa.qsaSelectSplitServes;
const qsaSelectTgFor = qwen4_qsa.qsaSelectTgFor;
const qsaSelectTopBlocks = qwen4_qsa.qsaSelectTopBlocks;
const qsaSelectTopBlocksSplit = qwen4_qsa.qsaSelectTopBlocksSplit;
const qsaVisibleBoundsHost = qwen4_qsa.qsaVisibleBoundsHost;
const model_mod = @import("../model.zig");
const log = @import("../log.zig");
const testing = std.testing;
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const OneShotBits = qwen4_qsa_attn.OneShotBits;
const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT = qwen4_qsa_attn.QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
const qsaPrefillTransientBytes = qwen4_qsa_attn.qsaPrefillTransientBytes;
const qsaScoreRowsPerChunkFor = qwen4_qsa_attn.qsaScoreRowsPerChunkFor;
const qsaScoreRowsPerChunkFused = qwen4_qsa_attn.qsaScoreRowsPerChunkFused;
const qsaWidthBucket = qwen4_qsa_attn.qsaWidthBucket;
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const qsaProbeBitEqual = qwen4_qsa_gather.qsaProbeBitEqual;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const qwen4ReadF32 = test_support.qwen4ReadF32;
const testRandUniformBf16 = test_support.testRandUniformBf16;
const ForwardCtx = trf.ForwardCtx;
const FullAttnWeights = trf.FullAttnWeights;
const KVCache = trf.KVCache;
const MoeDecodeDispatchArm = trf.MoeDecodeDispatchArm;
const QSA_SCORE_KERNEL_HEADER = trf.QSA_SCORE_KERNEL_HEADER;
const QsaCfgCache = trf.QsaCfgCache;
const SSMCacheEntry = trf.SSMCacheEntry;
const Transformer = trf.Transformer;
const VqmmLane = trf.VqmmLane;
const moeDecodeDispatchArm = trf.moeDecodeDispatchArm;
const qsaProbeAllFinite = trf.qsaProbeAllFinite;
const qsaProbeCorruptFirst = trf.qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = trf.qsaProbeLcgBf16;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const verifyQmmNaxAvailable = trf.verifyQmmNaxAvailable;
const vqmmLaneFor = trf.vqmmLaneFor;
const yarnSpec = trf.yarnSpec;

/// Keyed on what the config reads: output `[1, rows, kb]`, grid `(tg, rows, 1)`, templates
/// TGS and K. `nb` is a runtime shape (it advances every `ratio` decode tokens).
pub const QsaSelectCfgKey = struct { rows: c_int, k: c_int, tg: c_int, digits: c_int };
pub var qsa_select_cfgs = QsaCfgCache(QsaSelectCfgKey, 1){};
pub fn qsaSelectCfgBuilds() usize {
    return qsa_select_cfgs.builds;
}

pub var qsa_select_kernel_override: ?bool = null;

/// Fused QSA block select; the override restores the argpartition chain.
pub fn qsaSelectKernelEnabled() bool {
    return qsa_select_kernel_override orelse true;
}






pub var qsa_select_split_override: ?bool = null;

pub fn qsaSelectSplitEnabled() bool {
    return qsa_select_split_override orelse true;
}





pub const QsaSelectSplitCfgKey = struct { rows: c_int, k: c_int, slice_tg: c_int, merge_tg: c_int, digits: c_int };
pub var qsa_select_split_cfgs = QsaCfgCache(QsaSelectSplitCfgKey, 2){};
pub fn qsaSelectSplitCfgBuilds() usize {
    return qsa_select_split_cfgs.builds;
}


pub const QsaScoreLayout = enum(u8) { base = 0, h4 = 1 };

pub var qsa_score_fused_override: ?bool = null;
pub var qsa_score_probe_flip_override: bool = false;
pub var qsa_score_layout_override: ?QsaScoreLayout = null;
pub var qsa_score_tf32_override: ?bool = null;
var qsa_score_tf32_env: ?bool = null;
var qsa_score_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_score_kernel_builds: usize = 0;

pub fn qsaScoreKernelBuilt() bool {
    return qsa_score_kernel_cached != null;
}

pub fn qsaScoreKernelBuilds() usize {
    return qsa_score_kernel_builds;
}

pub fn qsaScoreFusedEnabled() bool {
    return qsa_score_fused_override orelse true;
}

pub fn qsaScoreFusedActive() bool {
    if (!qsaScoreFusedEnabled()) return false;
    if (!verifyQmmNaxAvailable()) return false;
    if (!qsaScoreTf32Enabled()) return false;
    return qsaScoreFusedProbeOk();
}

var qsa_score_probe_state: std.atomic.Value(u8) = .init(0);

fn qsaScoreFusedProbeOk() bool {
    return qsa_score_probe_state.load(.acquire) == 2;
}

pub fn qsaScoreFusedProbeFailed() bool {
    return qsa_score_probe_state.load(.acquire) == 1;
}

pub fn qsaScoreFusedProbeReset() void {
    qsa_score_probe_state.store(0, .release);
}

pub fn qsaScoreFusedArm() void {
    if (qsa_score_probe_state.load(.acquire) != 0) return;
    if (!qsaScoreFusedEnabled() or !verifyQmmNaxAvailable() or !qsaScoreTf32Enabled()) return;
    const ok = qsaScoreFusedRunProbe();
    qsa_score_probe_state.store(if (ok) 2 else 1, .release);
}

pub fn qsaScoreFusedActiveFor(batch: c_int, n_idx: c_int, idx_hd: c_int) bool {
    return qsaScoreFusedActive() and batch == 1 and n_idx == 4 and idx_hd == 128;
}

fn qsaScoreTf32ValueOn(raw: []const u8) bool {
    var i: usize = 0;
    while (i < raw.len and std.ascii.isWhitespace(raw[i])) : (i += 1) {}
    if (i >= raw.len) return false;
    var sign: i32 = 1;
    if (raw[i] == '-') {
        sign = -1;
        i += 1;
    } else if (raw[i] == '+') {
        i += 1;
    }
    var val: i64 = 0;
    var any = false;
    while (i < raw.len) : (i += 1) {
        const c = raw[i];
        if (c < '0' or c > '9') break;
        any = true;
        val = val * 10 + @as(i64, c - '0');
    }
    if (!any) return false;
    return (if (sign < 0) -val else val) != 0;
}

fn qsaScoreTf32Enabled() bool {
    if (qsa_score_tf32_override) |v| return v;
    if (qsa_score_tf32_env) |v| return v;
    const v = blk: {
        const raw = std.c.getenv("MLX_ENABLE_TF32") orelse break :blk true;
        break :blk qsaScoreTf32ValueOn(std.mem.sliceTo(raw, 0));
    };
    qsa_score_tf32_env = v;
    return v;
}




fn qsaScoreLayoutFor(rows: c_int) QsaScoreLayout {
    if (qsa_score_layout_override) |v| return v;
    return if (rows >= 128) .h4 else .base;
}

fn qsaScoreNsg(layout: QsaScoreLayout) c_int {
    return if (layout == .h4) 8 else 4;
}

fn qsaScoreRowsPerTg(layout: QsaScoreLayout, nsg: c_int) c_int {
    return nsg * (if (layout == .h4) @as(c_int, 16) else 4);
}

fn qsaScoreNsh(nslab: u32, ngrp: u32, layout: QsaScoreLayout) u32 {
    const slabs_per: u32 = if (layout == .h4) 32 else 1;
    var nsh: u32 = @max(@as(u32, 1), nslab / slabs_per);
    const min_nsh: u32 = @max(@as(u32, 1), (256 + ngrp - 1) / ngrp);
    if (nsh < min_nsh) nsh = min_nsh;
    if (nsh > nslab) nsh = nslab;
    return nsh;
}

const QSA_SCORE_KERNEL_SOURCE =
    \\constexpr uint NSG = (uint)NSGVAL;
    \\constexpr uint TGN = NSG * 32u;
    \\static_assert(4096u % TGN == 0);
    \\constexpr uint USEH4 = (uint)USEH4VAL;
    \\constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
    \\    16, 32, 16, false, false, true,
    \\    mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    \\mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    \\threadgroup bfloat tile[128*32];
    \\const uint nb   = (uint)POOL_shape[1];
    \\const uint rows = (uint)Q_shape[2];
    \\const uint tid  = thread_position_in_threadgroup.x;
    \\const uint sg   = tid / 32u;
    \\const uint lane = tid % 32u;
    \\const uint rg   = threadgroup_position_in_grid.y;
    \\const uint gsh  = threadgroup_position_in_grid.x;
    \\const uint NSH  = (uint)threadgroups_per_grid.x;
    \\const short qid = lane >> 2;
    \\const short fm  = ((qid & 4) | ((lane >> 1) & 3));
    \\const short fn  = ((qid & 2) | (lane & 1)) * 4;
    \\const uint nslab = (nb + 31u) / 32u;
    \\auto ct_a = op.get_left_input_cooperative_tensor<bfloat, bfloat, float>();
    \\auto ct_b = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    \\auto ct_c = op.get_destination_cooperative_tensor<
    \\    metal::remove_addrspace_t<decltype(ct_a)>,
    \\    metal::remove_addrspace_t<decltype(ct_b)>, float>();
    \\if (USEH4 != 0u) {
    \\  const uint qrow0 = rg * (NSG*16u) + sg * 16u;
    \\  bfloat A[4][8][8];
    \\  #pragma unroll
    \\  for (short h = 0; h < 4; h++) {
    \\    #pragma unroll
    \\    for (short ks = 0; ks < 8; ks++) {
    \\      #pragma unroll
    \\      for (short i = 0; i < 2; i++) {
    \\        #pragma unroll
    \\        for (short j = 0; j < 4; j++) {
    \\          const uint qr = qrow0 + (uint)(fm + i*8), c = (uint)ks*16u + (uint)(fn + j);
    \\          A[h][ks][i*4+j] = (qr < rows) ? Q[((ulong)h*(ulong)rows + qr)*128ul + c] : (bfloat)0;
    \\        }
    \\      }
    \\    }
    \\  }
    \\  float acc[16];
    \\  for (uint sl = gsh; sl < nslab; sl += NSH) {
    \\    const uint n0 = sl * 32u;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma unroll
    \\    for (uint i = 0; i < 4096u/TGN; i++) {
    \\      const uint idx = tid + i*TGN;
    \\      const uint n = idx / 128u, k = idx % 128u;
    \\      tile[k*32u + n] = ((n0 + n) < nb) ? POOL[(ulong)(n0 + n)*128ul + k] : (bfloat)0;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma unroll
    \\    for (short e = 0; e < 16; e++) acc[e] = 0.0f;
    \\    #pragma unroll
    \\    for (short h = 0; h < 4; h++) {
    \\      #pragma unroll
    \\      for (short e = 0; e < 16; e++) ct_c[e] = 0.0f;
    \\      #pragma unroll
    \\      for (short ks = 0; ks < 8; ks++) {
    \\        #pragma unroll
    \\        for (short i = 0; i < 8; i++) ct_a[i] = A[h][ks][i];
    \\        #pragma unroll
    \\        for (short i = 0; i < 2; i++) {
    \\          #pragma unroll
    \\          for (short j = 0; j < 4; j++) {
    \\            const uint kk = (uint)ks*16u + (uint)(fm + i*8);
    \\            ct_b[i*4+j]     = tile[kk*32u + (uint)(fn + j)];
    \\            ct_b[8 + i*4+j] = tile[kk*32u + (uint)(fn + j + 16)];
    \\          }
    \\        }
    \\        op.run(ct_a, ct_b, ct_c);
    \\      }
    \\      #pragma unroll
    \\      // heads accumulate ascending in f32: mlx_sum_axis reduces a length-4 axis sequentially
    \\      for (short e = 0; e < 16; e++) acc[e] += metal::max(ct_c[e], 0.0f);
    \\    }
    \\    #pragma unroll
    \\    for (short i = 0; i < 2; i++) {
    \\      #pragma unroll
    \\      for (short j = 0; j < 4; j++) {
    \\        const uint qr = qrow0 + (uint)(fm + i*8);
    \\        const uint c0 = n0 + (uint)(fn + j), c1 = c0 + 16u;
    \\        if (qr < rows) {
    \\          if (c0 < nb) OUT[(ulong)qr*(ulong)nb + c0] = acc[i*4+j];
    \\          if (c1 < nb) OUT[(ulong)qr*(ulong)nb + c1] = acc[8 + i*4+j];
    \\        }
    \\      }
    \\    }
    \\  }
    \\} else {
    \\  const uint arow0 = rg * (NSG*16u) + sg * 16u;
    \\  const uint qrow0 = arow0 / 4u;
    \\  bfloat a0[8],a1[8],a2[8],a3[8],a4[8],a5[8],a6[8],a7[8];
    \\  #define LOADA(KS, DST) { _Pragma("unroll") for (short i = 0; i < 2; i++) { _Pragma("unroll") for (short j = 0; j < 4; j++) { const uint ar = arow0 + fm + i*8, c = (KS)*16u + fn + j; const uint qr = ar >> 2, hh = ar & 3u; DST[i*4+j] = (qr < rows) ? Q[((ulong)hh*(ulong)rows + qr)*128ul + c] : (bfloat)0; } } }
    \\  LOADA(0u,a0) LOADA(1u,a1) LOADA(2u,a2) LOADA(3u,a3)
    \\  LOADA(4u,a4) LOADA(5u,a5) LOADA(6u,a6) LOADA(7u,a7)
    \\  for (uint sl = gsh; sl < nslab; sl += NSH) {
    \\    const uint n0 = sl * 32u;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma unroll
    \\    for (uint i = 0; i < 4096u/TGN; i++) {
    \\      const uint idx = tid + i*TGN;
    \\      const uint n = idx / 128u, k = idx % 128u;
    \\      tile[k*32u + n] = ((n0 + n) < nb) ? POOL[(ulong)(n0 + n)*128ul + k] : (bfloat)0;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma unroll
    \\    for (short i = 0; i < 16; i++) ct_c[i] = 0.0f;
    \\    #define STEP(KS, SRCA) { _Pragma("unroll") for (short i = 0; i < 8; i++) ct_a[i] = SRCA[i]; _Pragma("unroll") for (short i = 0; i < 2; i++) { _Pragma("unroll") for (short j = 0; j < 4; j++) { const uint kk = (KS)*16u + fm + i*8; ct_b[i*4+j] = tile[kk*32u + (fn + j)]; ct_b[8 + i*4+j] = tile[kk*32u + (fn + j + 16)]; } } op.run(ct_a, ct_b, ct_c); }
    \\    STEP(0u,a0) STEP(1u,a1) STEP(2u,a2) STEP(3u,a3) STEP(4u,a4) STEP(5u,a5) STEP(6u,a6) STEP(7u,a7)
    \\    float sv[16];
    \\    #pragma unroll
    \\    for (short e = 0; e < 16; e++) {
    \\      float v = metal::max(ct_c[e], 0.0f);
    \\      // lanes l^2, l^4, l^6 hold heads 1..3; the sum stays ascending, matching mlx_sum_axis
    \\      v += metal::max(simd_shuffle_xor(ct_c[e], 2u), 0.0f);
    \\      v += metal::max(simd_shuffle_xor(ct_c[e], 4u), 0.0f);
    \\      v += metal::max(simd_shuffle_xor(ct_c[e], 6u), 0.0f);
    \\      sv[e] = v;
    \\    }
    \\    if ((fm & 3) == 0) {
    \\      #pragma unroll
    \\      for (short i = 0; i < 2; i++) {
    \\        #pragma unroll
    \\        for (short j = 0; j < 4; j++) {
    \\          const uint qr = qrow0 + (uint)(fm + i*8) / 4u;
    \\          const uint c0 = n0 + fn + j, c1 = c0 + 16u;
    \\          if (qr < rows) {
    \\            if (c0 < nb) OUT[(ulong)qr*(ulong)nb + c0] = sv[i*4+j];
    \\            if (c1 < nb) OUT[(ulong)qr*(ulong)nb + c1] = sv[8 + i*4+j];
    \\          }
    \\        }
    \\      }
    \\    }
    \\  }
    \\}
    \\
;

const QsaScoreCfgKey = struct { nsg: c_int, rows: c_int, nb: c_int, layout: u8 };
var qsa_score_cfgs = QsaCfgCache(QsaScoreCfgKey, 1){};
var qsa_score_engaged_bits: OneShotBits = .{};

fn getQsaScoreKernel() !mlx.mlx_fast_metal_kernel {
    if (!verifyQmmNaxAvailable()) return error.NaxUnavailable;
    if (qsa_score_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "Q", "POOL" };
    const output_names = [_][*:0]const u8{"OUT"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qsa_score",
        in_vec,
        out_vec,
        QSA_SCORE_KERNEL_SOURCE,
        QSA_SCORE_KERNEL_HEADER,
        // row-contiguous is load-bearing: q is an axis-2 slice whose parent head stride is
        // S*128; the copy is what makes Q_shape[2] the physical row stride the kernel indexes by
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_score_kernel_cached = kernel;
    qsa_score_kernel_builds += 1;
    return kernel;
}

pub fn qsaScoreSheetComposed(s: mlx.mlx_stream, q: mlx.mlx_array, k32t: mlx.mlx_array) !mlx.mlx_array {
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, q32, k32t, s));
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var relu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(relu);
    try mlx.check(mlx.mlx_maximum(&relu, sc, zero, s));
    var scores = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&scores, relu, 1, false, s));
    return scores;
}

pub fn qsaScoreK32tFromPooled(s: mlx.mlx_stream, pooled: mlx.mlx_array) !mlx.mlx_array {
    const ps = mlx.getShape(pooled);
    var k4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k4);
    try mlx.check(mlx.mlx_reshape(&k4, pooled, &[_]c_int{ ps[0], 1, ps[1], ps[2] }, 4, s));
    const perm = [_]c_int{ 0, 1, 3, 2 };
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k4, &perm, 4, s));
    var k32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
    return k32;
}

fn qsaScoreFusedFillConfig(config: mlx.mlx_fast_metal_kernel_config, rows: c_int, nb: c_int, nsg: c_int, ngrp: c_int, nsh: c_int, layout: QsaScoreLayout) !void {
    const o_shape = [_]c_int{ 1, rows, nb };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nsh * nsg * 32, ngrp, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, nsg * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGVAL", nsg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "USEH4VAL", if (layout == .h4) 1 else 0));
}

pub fn qsaScoreFusedDispatch(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    const rows = qs[2];
    const nb = ps[1];
    const layout = qsaScoreLayoutFor(rows);
    const nsg = qsaScoreNsg(layout);
    const r_tg = qsaScoreRowsPerTg(layout, nsg);
    const ngrp: c_int = @divTrunc(rows + r_tg - 1, r_tg);
    const nslab: u32 = @intCast(@divTrunc(nb + 31, 32));
    const nsh: c_int = @intCast(qsaScoreNsh(nslab, @intCast(ngrp), layout));
    const kernel = try getQsaScoreKernel();
    const key = QsaScoreCfgKey{
        .nsg = nsg,
        .rows = rows,
        .nb = nb,
        .layout = @backingInt(layout),
    };
    const score_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_score_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try qsaScoreFusedFillConfig(config, rows, nb, nsg, ngrp, nsh, layout);
        qsa_score_cfgs.put(key, .{config});
        break :blk .{config};
    };
    const inputs_arr = [_]mlx.mlx_array{ q, pooled };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, score_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (qsa_score_engaged_bits.take(qsaWidthBucket(rows))) {
        log.debug(
            "[qsa-score] engaged (S={d} nb={d} layout={s} nsg={d} nsh={d})\n",
            .{ rows, nb, if (layout == .h4) "h4" else "base", nsg, nsh },
        );
    }
    return out;
}

fn qsaScoreFusedProbeLayout(s: mlx.mlx_stream, rows: c_int, nb: c_int) bool {
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 4, rows, 128 }, 0x51C04E11) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const pooled = qsaProbeLcgBf16(s, &[_]c_int{ 1, nb, 128 }, 0xA5A55A5A) orelse return false;
    defer _ = mlx.mlx_array_free(pooled);
    const k32t = qsaScoreK32tFromPooled(s, pooled) catch return false;
    defer _ = mlx.mlx_array_free(k32t);
    const want = qsaScoreSheetComposed(s, q, k32t) catch return false;
    defer _ = mlx.mlx_array_free(want);
    var got = qsaScoreFusedDispatch(s, q, pooled) catch return false;
    defer _ = mlx.mlx_array_free(got);
    if (qsa_score_probe_flip_override) {
        const flipped = qsaProbeCorruptFirst(s, got) orelse return false;
        _ = mlx.mlx_array_free(got);
        got = flipped;
    }
    if (mlx.mlx_array_eval(got) != 0 or mlx.mlx_array_eval(want) != 0) return false;
    if (!qsaProbeBitEqual(s, got, want)) {
        log.info("[qsa-score] disabled: probe mismatch ({s})\n", .{if (rows >= 128) "h4" else "base"});
        return false;
    }
    return true;
}

fn qsaScoreFusedRunProbe() bool {
    if (mlx.noGpuBackend()) {
        log.info("[qsa-score] disabled: no gpu\n", .{});
        return false;
    }
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) {
        log.info("[qsa-score] disabled: not gpu stream\n", .{});
        return false;
    }
    _ = getQsaScoreKernel() catch |e| {
        log.info("[qsa-score] disabled: {s}\n", .{@errorName(e)});
        return false;
    };
    if (!qsaScoreFusedProbeLayout(s, 16, 700)) return false;
    if (!qsaScoreFusedProbeLayout(s, 128, 700)) return false;
    return true;
}

pub fn qsaScoreRowsPerChunk(n_idx: u64, nb: u64, rows: u64, idx_hd: u64) u64 {
    return qsaScoreRowsPerChunkFused(n_idx, nb, rows, qsaScoreFusedActiveFor(1, @intCast(n_idx), @intCast(idx_hd)));
}

/// The QSA pooled-block RoPE tables, shared by every full-attention layer and across
/// forwards while the key holds. The key is the block position (`base`, `step`, `n`, dtype)
/// plus, on the M-RoPE arm, the position table compared by pointer and length; the pointer
/// is never dereferenced. The owner slot frees that table only after
/// `markQsaPooledRopeStale`, so a same-address table cannot hit. Connection threads write
/// only the atomic `stale`; the inference thread frees.
pub const QsaPooledRope = struct {
    base: c_int = 0,
    step: c_int = 0,
    n: c_int = 0,
    dtype: mlx.mlx_dtype = .float32,
    mrope: bool = false,
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    cos: mlx.mlx_array = .{ .ctx = null },
    sin: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,
    stale: std.atomic.Value(bool) = .init(false),

    pub fn deinit(self: *QsaPooledRope) void {
        if (self.cos.ctx != null) _ = mlx.mlx_array_free(self.cos);
        if (self.sin.ctx != null) _ = mlx.mlx_array_free(self.sin);
        self.cos = .{ .ctx = null };
        self.sin = .{ .ctx = null };
        self.mrope_pos = null;
        self.stale.store(false, .monotonic);
    }
};



/// The nb-keyed constants of the QSA block selection, built once per (nb, ratio) instead of
/// once per full-attention layer. A key mismatch rebuilds.
pub const QsaBlockConsts = struct {
    nb: c_int = 0,
    ratio: c_int = 0,
    /// int32 `[1, nb]`: the last cache position of each complete block.
    blk_end2: mlx.mlx_array = .{ .ctx = null },
    /// f32 `[nb]`: `b * 1e-7`, the index-descending bias reproducing torch.topk's lower-index-wins.
    tie_bias: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,

    pub fn deinit(self: *QsaBlockConsts) void {
        if (self.blk_end2.ctx != null) _ = mlx.mlx_array_free(self.blk_end2);
        if (self.tie_bias.ctx != null) _ = mlx.mlx_array_free(self.tie_bias);
        self.blk_end2 = .{ .ctx = null };
        self.tie_bias = .{ .ctx = null };
        self.nb = 0;
        self.ratio = 0;
    }

    /// After this returns, `blk_end2`/`tie_bias` are borrowed handles valid until the next `ensure` with a different key.
    pub fn ensure(self: *QsaBlockConsts, s: mlx.mlx_stream, nb: c_int, ratio: c_int) !void {
        if (self.blk_end2.ctx != null and self.nb == nb and self.ratio == ratio) return;
        self.deinit();
        errdefer self.deinit();
        var blk_end = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_end);
        try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
        self.blk_end2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&self.blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
        var blk_idx_f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_idx_f);
        try mlx.check(mlx.mlx_arange(&blk_idx_f, 0, @floatFromInt(nb), 1.0, .float32, s));
        const tie_eps = mlx.mlx_array_new_float(1e-7);
        defer _ = mlx.mlx_array_free(tie_eps);
        self.tie_bias = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&self.tie_bias, blk_idx_f, tie_eps, s));
        self.nb = nb;
        self.ratio = ratio;
        self.builds += 1;
    }
};

/// Fused launches since the fwd-ubench last zeroed it: an A/B arm's proof of engagement.
pub var qsa_pool_rope_dispatches: usize = 0;
/// Config builds whose template args changed; each new tuple is a pipeline compile.
var qsa_pool_rope_tpl_switches: usize = 0;


// Ported from upstream mlx-serve #556 (Samuel Reed). Mirrors each rounding of the composed chain:
//   mean = f32 sum in row order r = 0..R-1 times f32(1/R), then bf16 (MLX's order only while
//          R <= QSA_POOL_SERIAL_MAX_R);
//   norm = `rms_single_row` at D = 128: 32 lanes x 4 reads, simd_sum, precise rsqrt, w * T(x * inv);
//   rope = the bf16 cos/sin table, bf16(bf16(x*cos) + bf16(rot*sin)) as MLX's binary kernels round.
// One simdgroup per pooled row; lane l owns dims 4l..4l+3, so rotate_half's partner (d +- HALF) is
// HALF/4 lanes away. The block count `nblk` varies per call and rides an INPUT.
const QSA_POOL_ROPE_SOURCE =
    \\uint lane = thread_position_in_threadgroup.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\uint blk = row % uint(nblk);
    \\uint kbase = row * uint(R * 128) + lane * 4;
    \\float xs[4];
    \\float sumsq = 0.0f;
    \\for (uint j = 0; j < 4; ++j) {
    \\    float acc = 0.0f;
    \\    for (uint r = 0; r < uint(R); ++r) acc = float(kb[kbase + r * 128 + j]) + acc;
    \\    const T pooled = T(acc * inv_r);
    \\    xs[j] = float(pooled);
    \\    sumsq += xs[j] * xs[j];
    \\}
    \\sumsq = simd_sum(sumsq);
    \\float inv = metal::precise::rsqrt(sumsq / 128.0f + eps);
    \\for (uint j = 0; j < 4; ++j) {
    \\    const T nv = norm_w[lane * 4 + j] * T(xs[j] * inv);
    \\    const float nf = float(nv);
    \\    const float up = simd_shuffle_down(nf, ushort(HALF / 4));
    \\    const float dn = simd_shuffle_up(nf, ushort(HALF / 4));
    \\    uint d = lane * 4 + j;
    \\    T o = nv;
    \\    if (d < uint(RD)) {
    \\        const T rot = d < uint(HALF) ? T(-up) : T(dn);
    \\        const T a = T(nf * float(cosv[blk * RD + d]));
    \\        const T b = T(float(rot) * float(sinv[blk * RD + d]));
    \\        o = T(float(a) + float(b));
    \\    }
    \\    out[row * 128 + d] = o;
    \\}
;

/// Largest block ratio whose f32 block sum MLX folds in row order 0..R-1. The mean over axis 2 of
/// `[B, nb, R, 128]` is `col_reduce_small` with threadgroup_y = min(8, R): up to 8 rows each y-thread
/// owns one row and lid.y 0 folds them in order; past 8 threads pre-sum rows y, y+8, ... or trees.
const QSA_POOL_SERIAL_MAX_R: c_int = 8;

var qsa_pool_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_pool_rope_engaged: bool = false;
const QsaPoolRopeCfgKey = struct { tpl: [3]c_int, b: c_int, nb: c_int };
var qsa_pool_rope_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qsa_pool_rope_cfg_key: QsaPoolRopeCfgKey = std.mem.zeroes(QsaPoolRopeCfgKey);

fn getQsaPoolRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_pool_rope_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "kb", "norm_w", "cosv", "sinv", "eps", "inv_r", "nblk" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("sushi_qsa_pool_rope", in_vec, out_vec, QSA_POOL_ROPE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_pool_rope_kernel = kernel;
    return kernel;
}

/// Pooled block keys in one launch: f32 block mean, bf16, rms_norm(w), then partial RoPE with the
/// bf16 cos/sin table `[nb, rope_dims]`. Null → the caller runs `qsaPoolNorm` + `qsaPooledRopeComposed`.
pub fn qsaPoolNormRopeFused(
    s: mlx.mlx_stream,
    kb4: mlx.mlx_array, // [B, nb, R, D] bf16
    norm_w: mlx.mlx_array, // [D] bf16
    eps: f32,
    cosv: mlx.mlx_array, // [nb, rope_dims] bf16
    sinv: mlx.mlx_array,
    rope_dims: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ksh = mlx.getShape(kb4);
    const wsh = mlx.getShape(norm_w);
    const csh = mlx.getShape(cosv);
    if (ksh.len != 4 or ksh[3] != 128 or ksh[1] < 1 or ksh[2] < 1 or ksh[2] > QSA_POOL_SERIAL_MAX_R) return null;
    if (wsh.len != 1 or wsh[0] != 128) return null;
    // rotate_half's partner must sit whole lanes away: HALF a multiple of 4.
    if (rope_dims <= 0 or rope_dims > 128 or @mod(rope_dims, 8) != 0) return null;
    if (csh.len != 2 or csh[0] != ksh[1] or csh[1] != rope_dims) return null;
    if (!std.mem.eql(c_int, csh, mlx.getShape(sinv))) return null;
    inline for (.{ kb4, norm_w, cosv, sinv }) |arr| {
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return null;
    }
    const key = QsaPoolRopeCfgKey{ .tpl = .{ ksh[2], rope_dims, @divExact(rope_dims, 2) }, .b = ksh[0], .nb = ksh[1] };
    if (qsa_pool_rope_cfg == null or !std.meta.eql(qsa_pool_rope_cfg_key, key)) {
        if (qsa_pool_rope_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        qsa_pool_rope_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const out_shape = [_]c_int{ ksh[0], ksh[1], 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, ksh[0] * ksh[1], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        for ([_][*:0]const u8{ "R", "RD", "HALF" }, key.tpl) |n, v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, n, v));
        if (!std.meta.eql(qsa_pool_rope_cfg_key.tpl, key.tpl)) qsa_pool_rope_tpl_switches += 1;
        qsa_pool_rope_cfg = config;
        qsa_pool_rope_cfg_key = key;
    }
    const kernel = try getQsaPoolRopeKernel();
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    // MLX's mean multiplies by NumberOfElements: f32(1.0 / n) computed in f64.
    const inv_r = mlx.mlx_array_new_float(@floatCast(1.0 / @as(f64, @floatFromInt(ksh[2]))));
    defer _ = mlx.mlx_array_free(inv_r);
    const nblk = mlx.mlx_array_new_int(ksh[1]);
    defer _ = mlx.mlx_array_free(nblk);
    const inputs_arr = [_]mlx.mlx_array{ kb4, norm_w, cosv, sinv, eps_a, inv_r, nblk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qsa_pool_rope_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    qsa_pool_rope_dispatches += 1;
    if (!qsa_pool_rope_engaged) {
        qsa_pool_rope_engaged = true;
        log.info("[qsa] fused pooled-key upkeep engaged (blocks={d} ratio={d} rope={d})\n", .{ ksh[1], ksh[2], rope_dims });
    }
    return out;
}

/// Elements whose bits differ; bf16 and f16 widen to f32 exactly, so ±0 and NaN payloads count.
fn qsaPoolBitMismatches(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !usize {
    try testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    const ah = try qwen4ReadF32(testing.allocator, ac, s);
    defer testing.allocator.free(ah);
    const bh = try qwen4ReadF32(testing.allocator, bc, s);
    defer testing.allocator.free(bh);
    var bad: usize = 0;
    for (ah, bh) |x, y| bad += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
    return bad;
}

/// Raw index keys `[B, nb, ratio, 128]` with magnitudes spread over 10^lo..10^hi and random signs.
fn qsaPoolTestKeys(rnd: std.Random, b: c_int, nb: c_int, ratio: c_int, lo_exp: f32, hi_exp: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ b, nb, ratio, 128 };
    const data = try testing.allocator.alloc(f32, @intCast(b * nb * ratio * 128));
    defer testing.allocator.free(data);
    for (data) |*x| {
        const mag = std.math.pow(f32, 10.0, lo_exp + (hi_exp - lo_exp) * rnd.float(f32));
        x.* = if (rnd.boolean()) mag else -mag;
    }
    const f = mlx.mlx_array_new_data(data.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dt, s));
    return out;
}

/// Keys `[1, nb, ratio, 128]` whose f32 block sums depend on the add order: even columns hold
/// 2^24, -2^24 half a block later and 1s, rotated per column and block; odd columns are noise.
fn qsaPoolCancelKeys(rnd: std.Random, nb: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const r: usize = @intCast(ratio);
    const data = try testing.allocator.alloc(f32, @as(usize, @intCast(nb)) * r * 128);
    defer testing.allocator.free(data);
    for (0..@intCast(nb)) |b| for (0..128) |c| for (0..r) |row| {
        const at = (b * r + row) * 128 + c;
        const p = (row + c / 2 + b) % r;
        data[at] = if (c % 2 == 1) rnd.float(f32) - 0.5 else if (r > 1 and p == 0) 16777216.0 else if (r > 1 and p == r / 2) -16777216.0 else 1.0;
    };
    const shape = [_]c_int{ 1, nb, ratio, 128 };
    const f = mlx.mlx_array_new_data(data.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

/// The served Flash-Next indexer: 4 x 128 index heads, ratio 4, rope theta 1e7 on 64 of 256 dims,
/// and (`yarn`) the packs' YaRN x4 over 262144 with its bf16 mscale on the cos/sin table.
const QsaPoolTestTrunk = struct {
    t: Transformer,
    inv: [32]f64,
    freqs: mlx.mlx_array,

    pub fn init(self: *QsaPoolTestTrunk, s: mlx.mlx_stream, yarn: bool) void {
        const t = &self.t;
        t.s = s;
        t.allocator = testing.allocator;
        t.config = .{};
        // qwen4_exp's YaRN spans its partial rotary slice; the default model_type spans the whole head.
        t.config.model_type = "qwen4_exp";
        t.config.head_dim = 256;
        t.config.partial_rotary_factor = 0.25;
        t.config.rope_theta = 10_000_000.0;
        t.config.rms_norm_eps = 1e-6;
        t.config.indexer_n_heads = 4;
        t.config.indexer_head_dim = 128;
        t.config.indexer_compress_ratio = 4;
        t.config.indexer_budget = 1 << 30; // pool only: never reach the selection
        t.rope_freqs_yarn = null;
        t.yarn_inv_freq = null;
        t.fwd_gen = 0;
        t.qsa_pooled_rope = .{};
        self.freqs = .{ .ctx = null };
        if (!yarn) return;
        t.config.rope_yarn = true;
        t.config.yarn_factor = 4.0;
        t.config.yarn_orig_max_pos = 262_144;
        t.config.yarn_attention_factor = 0.1 * @log(@as(f32, 4.0)) + 1.0;
        yarnSpec(&t.config).invFreq(&self.inv);
        var den: [32]f32 = undefined;
        for (self.inv, 0..) |v, i| den[i] = @floatCast(1.0 / v);
        self.freqs = mlx.mlx_array_new_data(&den, &[_]c_int{32}, 1, .float32);
        t.rope_freqs_yarn = self.freqs;
        t.yarn_inv_freq = &self.inv;
    }

    pub fn deinit(self: *QsaPoolTestTrunk) void {
        self.t.qsa_pooled_rope.deinit();
        if (self.freqs.ctx != null) _ = mlx.mlx_array_free(self.freqs);
    }

    /// The indexer's table for `nb` blocks from `base`, as `qsaPooledCosSin` builds it on a text turn.
    fn cosSin(self: *QsaPoolTestTrunk, base: c_int, ratio: c_int, nb: c_int) !Transformer.MropeCosSin {
        const t = &self.t;
        return t.ropeCosSinFromFreqs(64, try t.ropeInvFreq(64, t.config.rope_theta), @floatFromInt(base), @floatFromInt(ratio), nb, .bfloat16);
    }

    /// Fused vs composed pooled keys for one block batch at `base`: the bit-mismatch count.
    fn case(self: *QsaPoolTestTrunk, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, base: c_int) !usize {
        const t = &self.t;
        const sh = mlx.getShape(kb4);
        const cs = try self.cosSin(base, sh[2], sh[1]);
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        const pn = try t.qsaPoolNorm(kb4, norm_w);
        defer _ = mlx.mlx_array_free(pn);
        const want = try t.qsaPooledRopeComposed(pn, 64, cs.cos, cs.sin);
        defer _ = mlx.mlx_array_free(want);
        const got = (try qsaPoolNormRopeFused(t.s, kb4, norm_w, t.config.rms_norm_eps, cs.cos, cs.sin, 64)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(got);
        try testing.expect(qsaProbeAllFinite(t.s, want));
        return qsaPoolBitMismatches(want, got, t.s);
    }
};

/// Random f32 score sheet (the reference's own dtype).
fn qsaRandScoresF32(rnd: std.Random, shape: []const c_int) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    // relu output: many exact zeros.
    for (data) |*x| {
        const v = rnd.float(f32) - 0.35;
        x.* = if (v > 0) v else 0.0;
    }
    return mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
}

/// The `[1, rows, nb]` bool visibility sheet the composed arm masks with, free of a Transformer.
fn qsaVisSheetForTest(s: mlx.mlx_stream, row0: c_int, rows: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    var blk_end = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end);
    try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
    var blk_end2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end2);
    try mlx.check(mlx.mlx_reshape(&blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(row0), @floatFromInt(row0 + rows), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ rows, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, blk_end2, pos2, s));
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s));
    return vis3;
}

/// `QsaBlockConsts.tie_bias` without the cache: `b * 1e-7`.
fn qsaTieBiasForTest(s: mlx.mlx_stream, nb: c_int) !mlx.mlx_array {
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0, @floatFromInt(nb), 1.0, .float32, s));
    const eps = mlx.mlx_array_new_float(1e-7);
    defer _ = mlx.mlx_array_free(eps);
    var bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&bias, idx, eps, s));
    return bias;
}

/// The reference both arms are measured against: per row, the top-`k` visible blocks by
/// (score descending, index ascending), emitted ascending with INT_MAX past the row's count.
fn qsaExactTopHost(alloc: std.mem.Allocator, scores: []const f32, rows: c_int, nb: c_int, bounds: []const i32, k: c_int) ![]i32 {
    const nbu: usize = @intCast(nb);
    const ku: usize = @intCast(k);
    const out = try alloc.alloc(i32, @as(usize, @intCast(rows)) * ku);
    errdefer alloc.free(out);
    const Cand = struct { v: f32, i: i32 };
    const cand = try alloc.alloc(Cand, nbu);
    defer alloc.free(cand);
    const pick = try alloc.alloc(i32, ku);
    defer alloc.free(pick);
    const Lt = struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            if (a.v != b.v) return a.v > b.v;
            return a.i < b.i;
        }
    };
    for (0..@intCast(rows)) |r| {
        const vb: usize = @intCast(@max(bounds[r], 0));
        for (0..vb) |i| cand[i] = .{ .v = scores[r * nbu + i], .i = @intCast(i) };
        std.sort.pdq(Cand, cand[0..vb], {}, Lt.lt);
        const take = @min(vb, ku);
        for (0..take) |i| pick[i] = cand[i].i;
        std.sort.pdq(i32, pick[0..take], {}, std.sort.asc(i32));
        for (0..take) |i| out[r * ku + i] = pick[i];
        for (take..ku) |i| out[r * ku + i] = std.math.maxInt(i32);
    }
    return out;
}

fn qsaExpectMatchesHost(got: mlx.mlx_array, want: []const i32, rows: c_int, k: c_int, who: []const u8) !void {
    const gs = mlx.getShape(got);
    try testing.expectEqual(@as(usize, 3), gs.len);
    try testing.expectEqual(@as(c_int, 1), gs[0]);
    try testing.expectEqual(rows, gs[1]);
    try testing.expectEqual(k, gs[2]);
    try testing.expectEqual(mlx.mlx_dtype.int32, mlx.mlx_array_dtype(got));
    try mlx.check(mlx.mlx_array_eval(got));
    const b = mlx.mlx_array_data_int32(got).?;
    const ku: usize = @intCast(k);
    for (want, 0..) |w, i| {
        if (w != b[i]) {
            std.debug.print(
                "[qsa-select] {s} differs from the exact reference at row {d} slot {d}: reference {d} got {d}\n",
                .{ who, i / ku, i % ku, w, b[i] },
            );
            return error.QsaSelectMismatch;
        }
    }
}

fn qsaIdsBitEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_array_equal(&res, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(res));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, res));
    return v;
}

fn qsaSelectMakeSheet(kind: u8, buf: []f32, rnd: std.Random) void {
    switch (kind) {
        0 => {
            for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
        },
        1 => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 5) == 0) rnd.float(f32) else 0.0;
        },
        2 => {
            @memset(buf, 0.0);
            const n = buf.len;
            const span = @min(n, 1024);
            for (0..span) |i| buf[i] = 5.0;
        },
        3 => {
            for (buf) |*v| v.* = rnd.float(f32);
            buf[0] = std.math.nan(f32);
            if (buf.len > 1) buf[1] = -0.0;
            if (buf.len > 2) buf[2] = 0.0;
        },
        else => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0) 0.0 else @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        },
    }
}

test "MTP head row widths pick the verify lane, the MoE arm and the QSA score kernel" {
    const t = std.testing;
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.msg, vqmmLaneFor(2, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 248320, false, 8));
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, false, 8));
    try t.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, false, 8));

    moe_gqmv.moe_rows_fused_override = true;
    defer moe_gqmv.moe_rows_fused_override = null;
    try t.expectEqual(MoeDecodeDispatchArm.gather_qmv, moeDecodeDispatchArm(1, 1, 10));
    try t.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(2, 1, 10));
    try t.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(4, 1, 10));
    moe_gqmv.moe_rows_fused_override = false;
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 1, 10));
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 2, 10));
    try t.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(4, 2, 10));

    try t.expect(!qsaScoreFusedActiveFor(2, 4, 128));
    try t.expect(!qsaScoreFusedActiveFor(4, 4, 128));
    try t.expectEqual(qsaScoreFusedActive(), qsaScoreFusedActiveFor(1, 4, 128));
}

test "qsa pooled keys: the fused kernel is bit-identical to the composed chain at the served geometry" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x1D8_0BE5);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    // { batch, blocks, base position, lo exp, hi exp }
    const cases = [_][5]f32{
        .{ 1, 1, 0, -1, 0.5 }, // first block: cos 1, sin 0
        .{ 1, 1, 2052, -1, 0.5 }, // decode just past the budget
        .{ 1, 2, 16000, -1, 0.5 }, // a 5-row verify completing two blocks
        .{ 1, 1, 262_148, -1, 0.5 }, // past YaRN's original window
        .{ 1, 1, 1_000_000, -1, 0.5 }, // 1M context
        .{ 1, 512, 0, -1, 0.5 }, // a 2048-token prefill chunk
        .{ 1, 37, 8196, -3, 4 }, // wide range: the f32 block sums round
        .{ 2, 5, 4096, -1, 0.5 }, // two rows
    };
    for ([_]bool{ true, false }) |yarn| {
        var tr: QsaPoolTestTrunk = undefined;
        tr.init(s, yarn);
        defer tr.deinit();
        for (cases) |c| {
            const kb4 = try qsaPoolTestKeys(rnd, @intFromFloat(c[0]), @intFromFloat(c[1]), 4, c[3], c[4], .bfloat16, s);
            defer _ = mlx.mlx_array_free(kb4);
            try testing.expectEqual(@as(usize, 0), try tr.case(kb4, norm_w, @intFromFloat(c[2])));
        }
    }
}

// Bar: a strided view of the indexer projection (the live k_raw slice) reads the same bits as a packed array.
test "qsa pooled keys: the fused kernel reads a strided raw-key view bit-identically" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var tr: QsaPoolTestTrunk = undefined;
    tr.init(s, true);
    defer tr.deinit();
    var prng = std.Random.DefaultPrng.init(0x5171_DE);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const qk = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, 5 * 128 }, s);
    defer _ = mlx.mlx_array_free(qk);
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    try mlx.check(mlx.mlx_slice(&k_raw, qk, &[_]c_int{ 0, 0, 4 * 128 }, 3, &[_]c_int{ 1, 12, 5 * 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var kb4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb4);
    try mlx.check(mlx.mlx_reshape(&kb4, k_raw, &[_]c_int{ 1, 3, 4, 128 }, 4, s));
    try testing.expectEqual(@as(usize, 0), try tr.case(kb4, norm_w, 2048));
}

// Bar: the kernel sums rows 0..R-1 in order, MLX's order while R <= 8; past 8 it declines.
test "qsa pooled keys: order-sensitive block sums match at ratios 1..8, and larger ratios decline" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var tr: QsaPoolTestTrunk = undefined;
    tr.init(s, true);
    defer tr.deinit();
    var prng = std.Random.DefaultPrng.init(0xCA2CE1);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    for ([_]c_int{ 1, 2, 3, 4, 8 }) |ratio| {
        const kb4 = try qsaPoolCancelKeys(rnd, 9, ratio, s);
        defer _ = mlx.mlx_array_free(kb4);
        try testing.expectEqual(@as(usize, 0), try tr.case(kb4, norm_w, 4096));
        const wide = try qsaPoolTestKeys(rnd, 1, 5, ratio, -3, 4, .bfloat16, s);
        defer _ = mlx.mlx_array_free(wide);
        try testing.expectEqual(@as(usize, 0), try tr.case(wide, norm_w, 8192));
    }
    for ([_]c_int{ 9, 16, 32 }) |ratio| {
        const kb4 = try qsaPoolCancelKeys(rnd, 3, ratio, s);
        defer _ = mlx.mlx_array_free(kb4);
        const cs = try tr.cosSin(4096, ratio, 3);
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        try testing.expect((try qsaPoolNormRopeFused(s, kb4, norm_w, 1e-6, cs.cos, cs.sin, 64)) == null);
    }
}

// Bar: only the shapes and dtypes the kernel mirrors engage; everything else keeps the chain.
test "qsa pooled keys: f16 and f32 operands, other head and rope widths, and a CPU stream decline" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var tr: QsaPoolTestTrunk = undefined;
    tr.init(s, true);
    defer tr.deinit();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();
    const cs = try tr.cosSin(0, 4, 1);
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    const kb4 = try qsaPoolTestKeys(rnd, 1, 1, 4, -1, 0.5, .bfloat16, s);
    defer _ = mlx.mlx_array_free(kb4);
    const w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(w);
    const got = (try qsaPoolNormRopeFused(s, kb4, w, 1e-6, cs.cos, cs.sin, 64)) orelse return error.FusedDeclined;
    _ = mlx.mlx_array_free(got);

    inline for (.{ .float16, .float32 }) |dt| {
        const kb_dt = try qsaPoolTestKeys(rnd, 1, 1, 4, -1, 0.5, dt, s);
        defer _ = mlx.mlx_array_free(kb_dt);
        var w_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_dt);
        try mlx.check(mlx.mlx_astype(&w_dt, w, dt, s));
        var cos_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cos_dt);
        try mlx.check(mlx.mlx_astype(&cos_dt, cs.cos, dt, s));
        var sin_dt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sin_dt);
        try mlx.check(mlx.mlx_astype(&sin_dt, cs.sin, dt, s));
        try testing.expect((try qsaPoolNormRopeFused(s, kb_dt, w, 1e-6, cs.cos, cs.sin, 64)) == null);
        try testing.expect((try qsaPoolNormRopeFused(s, kb4, w_dt, 1e-6, cs.cos, cs.sin, 64)) == null);
        try testing.expect((try qsaPoolNormRopeFused(s, kb4, w, 1e-6, cos_dt, sin_dt, 64)) == null);
        try testing.expect((try qsaPoolNormRopeFused(s, kb_dt, w_dt, 1e-6, cos_dt, sin_dt, 64)) == null);
    }

    const kb64 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 4, 64 }, s);
    defer _ = mlx.mlx_array_free(kb64);
    const w64 = try testRandUniformBf16(rnd, &[_]c_int{64}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(w64);
    try testing.expect((try qsaPoolNormRopeFused(s, kb64, w64, 1e-6, cs.cos, cs.sin, 64)) == null);
    var cos60 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cos60);
    try mlx.check(mlx.mlx_slice(&cos60, cs.cos, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 60 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try testing.expect((try qsaPoolNormRopeFused(s, kb4, w, 1e-6, cos60, cos60, 60)) == null);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const switches = qsa_pool_rope_tpl_switches;
    try testing.expect((try qsaPoolNormRopeFused(cpu, kb4, w, 1e-6, cs.cos, cs.sin, 64)) == null);
    try testing.expectEqual(switches, qsa_pool_rope_tpl_switches);
}

// Bar: the block count is a runtime input, so decode (1), verify (2) and prefill (512) share ONE pipeline.
test "qsa pooled keys: block counts 1, 2, 7 and 512 share one fused pipeline" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var tr: QsaPoolTestTrunk = undefined;
    tr.init(s, true);
    defer tr.deinit();
    var prng = std.Random.DefaultPrng.init(0x919E);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    var after_first: ?usize = null;
    for ([_]c_int{ 1, 2, 7, 512, 1 }) |nb| {
        const kb4 = try qsaPoolTestKeys(rnd, 1, nb, 4, -1, 0.5, .bfloat16, s);
        defer _ = mlx.mlx_array_free(kb4);
        try testing.expectEqual(@as(usize, 0), try tr.case(kb4, norm_w, 4096));
        if (after_first == null) after_first = qsa_pool_rope_tpl_switches;
        try testing.expectEqual(after_first.?, qsa_pool_rope_tpl_switches);
    }
}

// Bar: the indexer's own upkeep (ring tails joined to the new rows, block-aligned strided views,
// rope bases at `pos_base + nb_cached*ratio`) leaves the same pooled bank with the kernel on or off.
test "qsa pooled keys: the pooled bank after prefill, decode and verify is bit-identical with the kernel on and off" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    defer qwen4_qsa.qsa_pool_rope_fused_override = null;
    var tr: QsaPoolTestTrunk = undefined;
    tr.init(s, true);
    defer tr.deinit();
    var prng = std.Random.DefaultPrng.init(0xB4_2CE);
    const rnd = prng.random();
    const none: mlx.mlx_array = .{ .ctx = null };
    var fa: FullAttnWeights = .{ .q_w = none, .q_s = none, .q_b = none, .k_w = none, .k_s = none, .k_b = none, .v_w = none, .v_s = none, .v_b = none, .o_w = none, .o_s = none, .o_b = none, .q_norm = none, .k_norm = none };
    fa.idx_k_norm = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(fa.idx_k_norm);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    var seq_off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &seq_off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    // A 37-row prefill leaves a 1-row tail; 36 is block-aligned, so its later 4-row verifies read strided views.
    const widths = [_][]const c_int{ &.{ 37, 1, 1, 1, 3, 3, 3, 5, 5, 5, 5, 1, 2 }, &.{ 36, 4, 4, 1, 1, 1, 1, 5, 3, 3, 5, 4 } };
    for ([_]c_int{ 0, 262_101, 999_937 }) |pos_base| for (widths) |seq| {
        var arms: [2]SSMCacheEntry = @splat(.{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true });
        defer for (&arms) |*e| ssmFreeQsaState(e);
        const launched = qsa_pool_rope_dispatches;
        var offset: c_int = 0;
        for (seq) |w| {
            const qk = try qsaPoolTestKeys(rnd, 1, 1, w, -1, 0.5, .bfloat16, s);
            defer _ = mlx.mlx_array_free(qk);
            var qk3 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(qk3);
            const wide = try attn256RandBf16(rnd, &[_]c_int{ 1, w, 4 * 128 }, s);
            defer _ = mlx.mlx_array_free(wide);
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try mlx.check(mlx.mlx_reshape(&raw, qk, &[_]c_int{ 1, w, 128 }, 3, s));
            const parts = [_]mlx.mlx_array{ wide, raw };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&qk3, vec, 2, s));
            for (&arms, [_]bool{ true, false }) |*e, fused| {
                qwen4_qsa.qsa_pool_rope_fused_override = fused;
                const m = try tr.t.qsaMaskFromQk(&ctx, qk3, &fa, e, offset, pos_base, 1, w, 0);
                if (m.ctx != null) _ = mlx.mlx_array_free(m);
            }
            offset += w;
        }
        try testing.expect(qsa_pool_rope_dispatches > launched);
        try testing.expectEqual(@as(c_int, @divTrunc(offset, 4)), mlx.getShape(arms[0].qsa_pooled)[1]);
        try testing.expectEqual(@as(usize, 0), try qsaPoolBitMismatches(arms[0].qsa_pooled, arms[1].qsa_pooled, s));
    };
}

test "qsa mask: the all-visible skip must not hand a NULL sheet to an mlx op (kv just past the budget)" {
    // `mlx_logical_and(&blk_sel, picked, vis3)` in the nb > block_topk arm passed the null
    // all-visible sheet straight to mlx-c and killed the process. At decode width all-visible
    // is always true, so every generated token past kv = (block_topk + 1) * ratio crashed.
    // Bar: where the predicate claims all-visible, the real sheet is all-true and the null-sheet
    // arm produces the same mask.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ratio: c_int = 4;
    const batch: c_int = 1;
    var prng = std.Random.DefaultPrng.init(0x2068_B0DE);
    const rnd = prng.random();

    var checked_skip: usize = 0;
    // 2049..2051 sit just below the selection boundary; 2052 is the first kv that can crash.
    for ([_]c_int{ 2049, 2052, 2068, 2100, 4100 }) |kv| {
        // `nb` is the complete-block count, floor(kv/ratio).
        const nb = @divTrunc(kv, ratio);
        for ([_]c_int{ 1, 31, 1024 }) |seq_len| {
            if (seq_len > kv) continue;
            const offset = kv - seq_len;
            var topk_buf: [11]c_int = undefined;
            topk_buf[0] = nb;
            topk_buf[1] = nb + 4;
            for (1..9) |d| topk_buf[1 + d] = nb - @as(c_int, @intCast(d));
            topk_buf[10] = 512;

            for (topk_buf) |block_topk| {
                if (block_topk <= 0) continue;
                const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
                defer _ = mlx.mlx_array_free(scores);
                const tie = try qsaTieBiasForTest(s, nb);
                defer _ = mlx.mlx_array_free(tie);
                const sheet = try qsaVisSheetForTest(s, offset, seq_len, nb, ratio);
                defer _ = mlx.mlx_array_free(sheet);

                const ref = try Transformer.qsaSelectMaskOps(std.testing.allocator, s, scores, sheet, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(ref);
                try mlx.check(mlx.mlx_array_eval(ref));

                if (!qsaAllBlocksVisible(offset, nb, ratio)) continue;
                var ones = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ones);
                try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1, seq_len, nb }, 3, .bool_, s));
                try std.testing.expect(try qsaArraysAllEqual(sheet, ones, s));
                const null_vis = mlx.mlx_array{ .ctx = null };
                const got = try Transformer.qsaSelectMaskOps(std.testing.allocator, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
                defer _ = mlx.mlx_array_free(got);
                try std.testing.expect(try qsaArraysAllEqual(got, ref, s));
                checked_skip += 1;
            }
        }
    }
    errdefer std.debug.print("[qsa-boundary] all-visible skip exercised in {d} cases\n", .{checked_skip});
    try std.testing.expect(checked_skip >= 8);
}

test "qsa selection: the mask arm and the gather path use ONE tie rule" {
    // Two selection paths with two different tie rules would let two slots with identical
    // scores attend different blocks. The signal: the mask arm now dispatches the select kernel.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const batch: c_int = 1;
    // All-visible at width S needs S <= (kv % ratio) + 1, so kv must not be a multiple of ratio.
    const seq_len: c_int = 4;
    const kv: c_int = 4099;
    const nb = @divTrunc(kv, ratio);
    // 511 so this test owns a config key no other test builds.
    const block_topk: c_int = 511;
    var prng = std.Random.DefaultPrng.init(0x71E5_10FF);
    const rnd = prng.random();

    const scores = try qsaRandScoresF32(rnd, &[_]c_int{ batch, seq_len, nb });
    defer _ = mlx.mlx_array_free(scores);
    const tie = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(tie);
    const offset = kv - seq_len;
    try std.testing.expect(qsaAllBlocksVisible(offset, nb, ratio));
    const null_vis = mlx.mlx_array{ .ctx = null };

    const before = qsaSelectCfgBuilds();
    const got = try Transformer.qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    // A build, not a hit: this key is unique to this test, so the counter moving IS the dispatch.
    try std.testing.expect(qsaSelectCfgBuilds() > before);

    try std.testing.expectEqual(@as(usize, 4), @as(usize, @intCast(mlx.mlx_array_ndim(got))));
    const gsh = mlx.getShape(got);
    try std.testing.expectEqual(batch, gsh[0]);
    try std.testing.expectEqual(seq_len, gsh[2]);
    try std.testing.expectEqual(kv, gsh[3]);

    // With the kernel off both arms fall to the chain: still one rule.
    qsa_select_kernel_override = false;
    defer qsa_select_kernel_override = null;
    const chain = try Transformer.qsaSelectMaskOps(ta, s, scores, null_vis, tie, offset, seq_len, kv, nb, ratio, block_topk, batch);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    try std.testing.expectEqual(kv, mlx.getShape(chain)[3]);
}

test "qsa boundary: past (block_topk+1)*ratio EVERY decode step reaches the all-visible skip" {
    // At decode width offset = kv-1 and nb = floor(kv/ratio), so the all-visible claim holds at
    // every kv and the skip is gated by `nb > block_topk` alone.
    const ratio: c_int = 4;
    const block_topk: c_int = 512; // budget 2048 / ratio
    const first: c_int = (block_topk + 1) * ratio; // 2052
    var kv: c_int = 1;
    while (kv <= 6000) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try std.testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
        try std.testing.expectEqual(kv >= first, nb > block_topk);
    }
    var kv2: c_int = 1;
    while (kv2 < first) : (kv2 += 1) {
        try std.testing.expect(!(@divTrunc(kv2, ratio) > block_topk));
    }
    // A prefill chunk's first row can sit far below the last complete block.
    try std.testing.expect(!qsaAllBlocksVisible(1028, @divTrunc(@as(c_int, 2052), ratio), ratio));
}

test "qsa visibility: decode width is ALWAYS all-visible; a prefill chunk's FIRST row decides" {
    const ratio: c_int = 4;
    // Decode (S == 1): all-visible for every ragged tail.
    var kv: c_int = 2049;
    while (kv <= 2400) : (kv += 1) {
        const nb = @divTrunc(kv, ratio);
        try testing.expect(qsaAllBlocksVisible(kv - 1, nb, ratio));
    }
    try testing.expect(qsaAllBlocksVisible(2051, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(2050, 513, ratio));
    try testing.expect(!qsaAllBlocksVisible(0, 513, ratio));
    try testing.expect(qsaAllBlocksVisible(0, 0, ratio));
}

test "qsa visibility skip: bit-identical selection where it is an identity, and a REAL difference where it is not" {
    // Where every block is visible the two arms must agree bit for bit; where they are not,
    // the arms must differ (else a blanket "always skip" would pass).
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.qsa_consts = .{};
    defer t.qsa_consts.deinit();

    const ratio: c_int = 4;
    const nb: c_int = 24;
    const block_topk: c_int = 8;
    const rows: c_int = 6;
    var prng = std.Random.DefaultPrng.init(0x5EED_9E11);
    const rnd = prng.random();
    var sc_buf: [@as(usize, @intCast(rows)) * @as(usize, @intCast(nb))]f32 = undefined;
    for (&sc_buf) |*v| v.* = if (rnd.boolean()) 0.0 else rnd.float(f32) * 4.0;
    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(&sc_buf, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const none = mlx.mlx_array{ .ctx = null };

    const row0_all: c_int = nb * ratio - 1;
    try testing.expect(qsaAllBlocksVisible(row0_all, nb, ratio));
    const a = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(a);
    const b = try t.qsaChunkSelect(scores, none, row0_all, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(b);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(a, b, s));

    // One row earlier is not all-visible: the two arms must disagree.
    const row0_partial: c_int = 4;
    try testing.expect(!qsaAllBlocksVisible(row0_partial, nb, ratio));
    const c = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, true);
    defer _ = mlx.mlx_array_free(c);
    const d = try t.qsaChunkSelect(scores, none, row0_partial, rows, nb, ratio, block_topk, block_topk, false);
    defer _ = mlx.mlx_array_free(d);
    try testing.expect(try attn256MaxDiff(c, d, s) > 0.0);

    // The `nb <= block_topk` arm takes the same skip.
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nb), 1.0, .int32, s));
    var canonical = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(canonical);
    try mlx.check(mlx.mlx_reshape(&canonical, ar, &[_]c_int{ 1, 1, nb }, 3, s));
    const e = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, true);
    defer _ = mlx.mlx_array_free(e);
    const f = try t.qsaChunkSelect(scores, canonical, row0_all, rows, nb, ratio, nb, nb, false);
    defer _ = mlx.mlx_array_free(f);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(e, f, s));
}

test "qsa block constants: one build per nb, not one per full-attention layer" {
    // Twelve full-attention layers asked for the same nb-keyed aranges; `builds` is the meter.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var qc: QsaBlockConsts = .{};
    defer qc.deinit();

    try qc.ensure(s, 6, 4);
    try testing.expectEqual(@as(usize, 1), qc.builds);
    const first_end = qc.blk_end2.ctx;
    const first_bias = qc.tie_bias.ctx;
    for (0..11) |_| try qc.ensure(s, 6, 4); // layers 2..12 of the same forward
    try testing.expectEqual(@as(usize, 1), qc.builds);
    try testing.expect(qc.blk_end2.ctx == first_end);
    try testing.expect(qc.tie_bias.ctx == first_bias);

    var ends = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ends);
    try mlx.check(mlx.mlx_astype(&ends, qc.blk_end2, .float32, s));
    try mlx.check(mlx.mlx_array_eval(ends));
    const ev = mlx.mlx_array_data_float32(ends).?;
    for (0..6) |b| try testing.expectEqual(@as(f32, @floatFromInt(4 * b + 3)), ev[b]);
    try mlx.check(mlx.mlx_array_eval(qc.tie_bias));
    const bv = mlx.mlx_array_data_float32(qc.tie_bias).?;
    for (0..6) |b| try testing.expectEqual(@as(f32, @floatFromInt(b)) * 1e-7, bv[b]);

    try qc.ensure(s, 7, 4);
    try testing.expectEqual(@as(usize, 2), qc.builds);
    try testing.expectEqual(@as(c_int, 7), mlx.getShape(qc.blk_end2)[1]);
    try qc.ensure(s, 7, 8);
    try testing.expectEqual(@as(usize, 3), qc.builds);
}

test "qsaVisibleBoundsHost: the first NOT-fully-visible block per row" {
    var out: [6]i32 = undefined;
    qsaVisibleBoundsHost(&out, 0, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 0, 0, 0, 1, 1, 1 }, &out);
    qsaVisibleBoundsHost(&out, 2799, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 700, 700, 700, 700, 700, 700 }, &out);
    qsaVisibleBoundsHost(&out, 2044, 6, 700, 4, false);
    try testing.expectEqualSlices(i32, &[_]i32{ 511, 511, 511, 512, 512, 512 }, &out);
    qsaVisibleBoundsHost(&out, 4, 6, 24, 4, true);
    try testing.expectEqualSlices(i32, &[_]i32{ 24, 24, 24, 24, 24, 24 }, &out);
}

test "qsa select kernel: the fused radix-select IS the exact top-k with the split on and off, and so is the chain where its bias can resolve" {
    // Bit equality against `qsaExactTopHost` across the nb the arch reaches (700..262144), the
    // query widths it runs at and three visibility regimes. Scores are exact binary fractions
    // k/16 in [0, 0.5] plus forced zeros: the regime where the chain's bias is also exact, so
    // both arms can be held to the same bar.
    try mlx.requireGpu();
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const k: c_int = 512;

    const cases = [_]struct { nb: c_int, rows: c_int }{
        .{ .nb = 700, .rows = 1 },
        .{ .nb = 700, .rows = 6 },
        .{ .nb = 700, .rows = 15 },
        .{ .nb = 700, .rows = 256 },
        .{ .nb = 2048, .rows = 1 },
        .{ .nb = 2048, .rows = 15 },
        .{ .nb = 2048, .rows = 256 },
        .{ .nb = 15675, .rows = 1 },
        .{ .nb = 15675, .rows = 6 },
        .{ .nb = 15675, .rows = 15 },
        .{ .nb = 15675, .rows = 256 },
        .{ .nb = 65536, .rows = 1 },
        .{ .nb = 65536, .rows = 15 },
        .{ .nb = 65536, .rows = 64 },
        .{ .nb = 262144, .rows = 1 },
        .{ .nb = 262144, .rows = 6 },
    };
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();

    for (cases) |c| {
        const nbu: usize = @intCast(c.nb);
        const rowsu: usize = @intCast(c.rows);
        const buf = try ta.alloc(f32, rowsu * nbu);
        defer ta.free(buf);
        for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
            0.0
        else
            @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        const sc_shape = [_]c_int{ 1, c.rows, c.nb };
        const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);

        const row0s = [_]c_int{ 0, 2044, 2000, c.nb * ratio };
        for (row0s) |row0| {
            const all_vis = qsaAllBlocksVisible(row0, c.nb, ratio);
            const bounds_host = try ta.alloc(i32, rowsu);
            defer ta.free(bounds_host);
            qsaVisibleBoundsHost(bounds_host, row0, c.rows, c.nb, ratio, all_vis);
            const want = try qsaExactTopHost(ta, buf, c.rows, c.nb, bounds_host, k);
            defer ta.free(want);

            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{c.rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            var vis3 = mlx.mlx_array{ .ctx = null };
            defer if (vis3.ctx != null) {
                _ = mlx.mlx_array_free(vis3);
            };
            if (!all_vis) vis3 = try qsaVisSheetForTest(s, row0, c.rows, c.nb, ratio);
            const bias = try qsaTieBiasForTest(s, c.nb);
            defer _ = mlx.mlx_array_free(bias);
            const chain = try qsaSelectComposedOps(s, scores, vis3, bias, c.rows, c.nb, k, k, all_vis);
            defer _ = mlx.mlx_array_free(chain);

            for ([_]bool{ false, true }) |split| {
                qsa_select_split_override = split;
                const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                defer _ = mlx.mlx_array_free(got);
                qsaExpectMatchesHost(got, want, c.rows, k, if (split) "the split-forced fused kernel" else "the fused kernel") catch |e| {
                    std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                    return e;
                };
            }
            qsaExpectMatchesHost(chain, want, c.rows, k, "the composed chain") catch |e| {
                std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
        }
    }
}

test "qsa select kernel: past the bias's f32 resolution the chain's tie rule is UNSPECIFIED and the kernel's is not" {
    // The chain's `b * 1e-7` bias only survives rounding below |v| ~= 0.84 (one f32 ulp is
    // v * 2^-23); above it ties fall back to argpartition's arbitrary order. A relu sum over
    // four heads sits far above that, so this is the production regime.
    const eps: f32 = 1e-7;
    // Distinct biased values the chain can see across a 2048-block row: 4 at production
    // magnitude, every block at |v| <= 0.5. Pure f32 arithmetic.
    const Distinct = struct {
        fn count(v: f32, n: u32) usize {
            var seen: usize = 0;
            var prev: f32 = 0;
            var b: u32 = 0;
            while (b < n) : (b += 1) {
                const cur = v - @as(f32, @floatFromInt(b)) * eps;
                if (b == 0 or cur != prev) seen += 1;
                prev = cur;
            }
            return seen;
        }
    };
    try testing.expectEqual(@as(usize, 4), Distinct.count(625.0, 2048));
    try testing.expectEqual(@as(usize, 2048), Distinct.count(0.5, 2048));
    const big: f32 = 625.0;
    try testing.expectEqual(big - 1369 * eps, big - 1370 * eps);

    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const nb: c_int = 2048;
    const k: c_int = 512;
    const rows: c_int = 16;

    const nbu: usize = @intCast(nb);
    const buf = try ta.alloc(f32, @as(usize, @intCast(rows)) * nbu);
    defer ta.free(buf);
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();
    for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
        0.0
    else
        @floatFromInt(rnd.uintLessThan(u32, 1000));

    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const row0: c_int = nb * ratio;
    try testing.expect(qsaAllBlocksVisible(row0, nb, ratio));

    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    @memset(bounds_host, nb);
    const want = try qsaExactTopHost(ta, buf, rows, nb, bounds_host, k);
    defer ta.free(want);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);

    // The kernel is the reference at production magnitude.
    const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(got);
    try qsaExpectMatchesHost(got, want, rows, k, "the fused kernel");

    // The chain may only ever be wrong about a tie: the bar is the selected score multiset per row.
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, rows, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const ku: usize = @intCast(k);
    const sw = try ta.alloc(f32, ku);
    defer ta.free(sw);
    const sc = try ta.alloc(f32, ku);
    defer ta.free(sc);
    var diffs: usize = 0;
    for (0..@intCast(rows)) |r| {
        for (0..ku) |i| {
            sw[i] = buf[r * nbu + @as(usize, @intCast(want[r * ku + i]))];
            sc[i] = buf[r * nbu + @as(usize, @intCast(ch[r * ku + i]))];
            if (want[r * ku + i] != ch[r * ku + i]) diffs += 1;
        }
        std.sort.pdq(f32, sw, {}, std.sort.asc(f32));
        std.sort.pdq(f32, sc, {}, std.sort.asc(f32));
        try testing.expectEqualSlices(f32, sw, sc);
    }
}

test "qsa select kernel: on a near-tie the exact rule and the bias trick DISAGREE, and the kernel takes the exact one" {
    // The bias is exact only on exact ties: at nb = 262144 it spans 0.026, so a genuinely higher
    // score loses to a lower-indexed rival. torch.topk and the kernel take the higher score.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const nb: c_int = 262144;
    const k: c_int = 512;
    const ratio: c_int = 4;
    const winner: usize = 200000;

    const buf = try ta.alloc(f32, @intCast(nb));
    defer ta.free(buf);
    @memset(buf, 0.0);
    for (1..512) |i| buf[i] = 1000.0;
    buf[0] = 10.0; //          biased: 10.000  (bias 0)
    buf[winner] = 10.005; //   biased:  9.985  (bias 0.02) — the bias flips it
    const sc_shape = [_]c_int{ 1, 1, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);

    const row0: c_int = nb * ratio; // all-visible
    try testing.expect(qsaAllBlocksVisible(row0, nb, ratio));
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, 1, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);

    var bounds_host = [_]i32{nb};
    const bounds = mlx.mlx_array_new_data(&bounds_host, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const fused = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);

    try mlx.check(mlx.mlx_array_eval(chain));
    try mlx.check(mlx.mlx_array_eval(fused));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const fu = mlx.mlx_array_data_int32(fused).?;
    const H = struct {
        fn has(row: [*]const i32, n: usize, v: i32) bool {
            for (0..n) |i| if (row[i] == v) return true;
            return false;
        }
    };
    const ku: usize = @intCast(k);
    try testing.expect(H.has(fu, ku, @intCast(winner)));
    try testing.expect(!H.has(fu, ku, 0));
    try testing.expect(H.has(ch, ku, 0));
    try testing.expect(!H.has(ch, ku, @intCast(winner)));
    for (1..ku) |i| try testing.expect(fu[i] > fu[i - 1]);
}

test "qsa select kernel: the config is cached per FULL shape and rebuilt when rows or nb change" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 8;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    // Normalize first: one global cache shared with every test in the file.
    try Run.once(s, ta, 4, 64, k);
    const base = qsaSelectCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    // `nb` is a runtime shape, not a config input.
    try Run.once(s, ta, 4, 128, k);
    try Run.once(s, ta, 4, 517, k);
    try Run.once(s, ta, 4, 1027, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    try Run.once(s, ta, 9, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try testing.expect(@TypeOf(qsa_select_cfgs).SLOTS == 4);
}

test "qsa select split: ids are identical to the single-threadgroup kernel" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 512;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = false;

    const nbs = [_]c_int{ 2048, 15675, 65536, 215000, 262144 };
    const row_counts = [_]c_int{ 1, 2, 4, 7, 15, 16 };
    var prng = std.Random.DefaultPrng.init(0x51E7_0A11);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            const kind: u8 = 4;
            qsaSelectMakeSheet(kind, buf, rnd);
            const sc_shape = [_]c_int{ 1, rows, nb };
            const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
            defer _ = mlx.mlx_array_free(scores);
            const bound_kinds = [_]i32{ nb, 0 };
            for (bound_kinds) |bv| {
                const bounds_host = try ta.alloc(i32, rowsu);
                defer ta.free(bounds_host);
                var r: usize = 0;
                while (r < rowsu) : (r += 1) {
                    const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                    bounds_host[r] = @max(0, @min(nb, adj));
                }
                const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                defer _ = mlx.mlx_array_free(bounds);
                const single = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                defer _ = mlx.mlx_array_free(single);
                const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                defer _ = mlx.mlx_array_free(split);
                if (!try qsaIdsBitEqual(s, single, split)) {
                    std.debug.print("[qsa-select-split] mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                    return error.QsaSelectSplitMismatch;
                }
            }
        }
    }

    {
        const nb: c_int = 65536;
        const rows: c_int = 1;
        const kk: c_int = 511;
        const buf = try ta.alloc(f32, @intCast(nb));
        defer ta.free(buf);
        qsaSelectMakeSheet(1, buf, rnd);
        const scores = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, rows, nb }, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var bh = [_]i32{nb};
        const bounds = mlx.mlx_array_new_data(&bh, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const single = (try qsaSelectTopBlocks(s, scores, bounds, kk)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(single);
        const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, kk, rows, nb)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(split);
        if (!try qsaIdsBitEqual(s, single, split)) return error.QsaSelectSplitMismatch;
    }
}

test "qsa select split: declines above the width floor and below the nb floor" {
    const k: c_int = 512;
    try testing.expect(!qsaSelectSplitServes(16, 215000, k));
    try testing.expect(!qsaSelectSplitServes(32, 262144, k));
    try testing.expect(!qsaSelectSplitServes(1, 24575, k));
    try testing.expect(qsaSelectSplitServes(1, 24576, k));
    try testing.expect(qsaSelectSplitServes(1, 32768, k));
    try testing.expect(!qsaSelectSplitServes(15, 2048, k));
    try testing.expect(!qsaSelectSplitServes(0, 65536, k));
    try testing.expect(!qsaSelectSplitServes(-1, 65536, k));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 0));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 4096));
    try testing.expect(qsaSelectSplitServes(1, 65536, k));
    try testing.expect(qsaSelectSplitServes(15, 65536, k));
    try testing.expect(qsaSelectSplitServes(7, 215000, k));

    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 8)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 16, 65536);
    try testing.expect(!qwen4_qsa.qsa_select_used_split);
    try Run.once(s, ta, 1, 2048);
    try testing.expect(!qwen4_qsa.qsa_select_used_split);
    try Run.once(s, ta, 1, 24576);
    try testing.expect(qwen4_qsa.qsa_select_used_split);
}

test "qsa select split: kill switch off uses the single-threadgroup kernel" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 512)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qsa_select_split_override = false;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(!qwen4_qsa.qsa_select_used_split);

    qsa_select_split_override = true;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(qwen4_qsa.qsa_select_used_split);
}

test "qsa select split: the config is cached per FULL shape and rebuilt when rows change" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;
    const k: c_int = 512;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 1, 65536, k);
    try testing.expect(qwen4_qsa.qsa_select_used_split);
    const base = qsaSelectSplitCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 131072, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 2, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
}

test "qsa select: the digit scheme is a pure function of rows" {
    const saved_digits = qwen4_qsa.qsa_select_digits_override;
    defer qwen4_qsa.qsa_select_digits_override = saved_digits;
    qwen4_qsa.qsa_select_digits_override = null;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(1));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(7));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(15));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(16));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(256));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(4096));
    qwen4_qsa.qsa_select_digits_override = 8;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(256));
    qwen4_qsa.qsa_select_digits_override = 11;
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(1));
    qwen4_qsa.qsa_select_digits_override = null;

    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const kk: c_int = 13;
    const rows: c_int = 5;
    const nb: c_int = 96;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, r: c_int, n: c_int, k: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(r * n));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, r, n };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(r));
            defer alloc.free(bh);
            @memset(bh, n);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{r}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qwen4_qsa.qsa_select_digits_override = 11;
    try Run.once(s, ta, rows, nb, kk);
    const base = qsaSelectCfgBuilds();
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    qwen4_qsa.qsa_select_digits_override = 8;
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
}

test "qsa select: 8-bit digits are the exact top-k" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const saved_digits = qwen4_qsa.qsa_select_digits_override;
    const saved_split = qsa_select_split_override;
    defer {
        qwen4_qsa.qsa_select_digits_override = saved_digits;
        qsa_select_split_override = saved_split;
    }
    const k: c_int = 512;
    const nbs = [_]c_int{ 2048, 15675, 24576, 65536, 215000 };
    const row_counts = [_]c_int{ 1, 7, 15, 16, 256 };
    var prng = std.Random.DefaultPrng.init(0x8B17_D161);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const ku: usize = @intCast(k);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            var kind: u8 = 0;
            while (kind < 5) : (kind += 1) {
                switch (kind) {
                    0 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
                    },
                    1 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.60) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    2 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.95) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    3 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 0.5;
                        var r: usize = 0;
                        while (r < rowsu) : (r += 1) {
                            const row = buf[r * nbu ..][0..nbu];
                            var t: usize = 0;
                            while (t < ku + 400) : (t += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 3.0;
                            }
                            var h: usize = 0;
                            while (h < 50) : (h += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 9.0;
                            }
                        }
                    },
                    else => {
                        for (buf) |*v| v.* = rnd.float(f32);
                        var i: usize = 0;
                        while (i < buf.len) : (i += 1) {
                            const u = rnd.uintLessThan(u8, 100);
                            if (u < 2) buf[i] = std.math.nan(f32) else if (u < 32) buf[i] = -0.0 else if (u < 52) buf[i] = 0.0;
                        }
                    },
                }
                const sc_shape = [_]c_int{ 1, rows, nb };
                const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const bound_kinds = [_]i32{ nb, @divTrunc(k, 2), @min(nb, k + 17) };
                for (bound_kinds) |bv| {
                    const bounds_host = try ta.alloc(i32, rowsu);
                    defer ta.free(bounds_host);
                    var r: usize = 0;
                    while (r < rowsu) : (r += 1) {
                        const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                        bounds_host[r] = @max(0, @min(nb, adj));
                    }
                    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                    defer _ = mlx.mlx_array_free(bounds);

                    qsa_select_split_override = false;
                    qwen4_qsa.qsa_select_digits_override = 11;
                    const ref11 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(ref11);

                    qwen4_qsa.qsa_select_digits_override = 8;
                    const single8 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(single8);
                    if (!try qsaIdsBitEqual(s, ref11, single8)) {
                        std.debug.print("[qsa-select-8bit] single mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                        return error.QsaSelect8BitMismatch;
                    }

                    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) {
                        const split8 = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                        defer _ = mlx.mlx_array_free(split8);
                        if (!try qsaIdsBitEqual(s, ref11, split8)) {
                            std.debug.print("[qsa-select-8bit] split mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                            return error.QsaSelect8BitMismatch;
                        }
                    }
                }
            }
        }
    }
}

test "qsa score kernel: fused score sheet is bit-identical to the composed chain" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0x51C0_4E11);
    const rnd = prng.random();

    const nbs = [_]c_int{ 700, 2048, 15675, 65536 };
    const rowss = [_]c_int{ 1, 2, 3, 8, 15, 16, 17, 31, 64, 128, 256 };
    const layouts = [_]QsaScoreLayout{ .base, .h4 };
    for (nbs) |nb| {
        for (rowss) |rows| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
            defer _ = mlx.mlx_array_free(q);
            const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
            defer _ = mlx.mlx_array_free(pooled);
            try mlx.check(mlx.mlx_array_eval(pooled));
            const k32t = try qsaScoreK32tFromPooled(s, pooled);
            defer _ = mlx.mlx_array_free(k32t);
            const want = try qsaScoreSheetComposed(s, q, k32t);
            defer _ = mlx.mlx_array_free(want);
            for (layouts) |layout| {
                qsa_score_layout_override = layout;
                const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
            }
        }
    }

    {
        const extras = [_]struct { nb: c_int, rows: c_int }{
            .{ .nb = 262144, .rows = 1 },
            .{ .nb = 262144, .rows = 16 },
            .{ .nb = 262144, .rows = 64 },
            .{ .nb = 2048, .rows = 4096 },
        };
        for (extras) |c| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, c.rows, 128 }, s);
            defer _ = mlx.mlx_array_free(q);
            const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, c.nb, 128 }, s);
            defer _ = mlx.mlx_array_free(pooled);
            try mlx.check(mlx.mlx_array_eval(pooled));
            const k32t = try qsaScoreK32tFromPooled(s, pooled);
            defer _ = mlx.mlx_array_free(k32t);
            const want = try qsaScoreSheetComposed(s, q, k32t);
            defer _ = mlx.mlx_array_free(want);
            for (layouts) |layout| {
                qsa_score_layout_override = layout;
                const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
                defer _ = mlx.mlx_array_free(got);
                try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
            }
        }
    }

    {
        const parent_rows: c_int = 96;
        const rows: c_int = 40;
        const row0: c_int = 24;
        const nb: c_int = 2048;
        const parent = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, parent_rows, 128 }, s);
        defer _ = mlx.mlx_array_free(parent);
        var q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        try mlx.check(mlx.mlx_slice(&q, parent, &[_]c_int{ 0, 0, row0, 0 }, 4, &[_]c_int{ 1, 4, row0 + rows, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_array_eval(pooled));
        const k32t = try qsaScoreK32tFromPooled(s, pooled);
        defer _ = mlx.mlx_array_free(k32t);
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }

    {
        const nb: c_int = 2048;
        const cap: c_int = 3072;
        const rows: c_int = 16;
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const buf = try attn256RandBf16(rnd, &[_]c_int{ 1, cap, 128 }, s);
        defer _ = mlx.mlx_array_free(buf);
        var pooled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_slice(&pooled, buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, nb, 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
        const k32t = try qsaScoreK32tFromPooled(s, pooled);
        defer _ = mlx.mlx_array_free(k32t);
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }

    {
        const nb: c_int = 2048;
        const cap: c_int = 3072;
        const rows: c_int = 16;
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const buf = try attn256RandBf16(rnd, &[_]c_int{ 1, cap, 128 }, s);
        defer _ = mlx.mlx_array_free(buf);
        var pooled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pooled);
        try mlx.check(mlx.mlx_slice(&pooled, buf, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, nb, 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        try mlx.check(mlx.mlx_array_eval(pooled));
        var k4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k4);
        try mlx.check(mlx.mlx_reshape(&k4, buf, &[_]c_int{ 1, 1, cap, 128 }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k4, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32full = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32full);
        try mlx.check(mlx.mlx_astype(&k32full, kt, .float32, s));
        var k32t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32t);
        try mlx.check(mlx.mlx_slice(&k32t, k32full, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 1, 128, nb }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        const want = try qsaScoreSheetComposed(s, q, k32t);
        defer _ = mlx.mlx_array_free(want);
        for (layouts) |layout| {
            qsa_score_layout_override = layout;
            const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, want, s));
        }
    }
}

test "qsa score kernel: block ids through qsaSelectTopBlocks are identical on both arms" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_layout_override = .h4;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0xB10C_1D5);
    const rnd = prng.random();
    const nb: c_int = 2048;
    const rows: c_int = 16;
    const ratio: c_int = 4;
    const k: c_int = 512;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const k32t = try qsaScoreK32tFromPooled(s, pooled);
    defer _ = mlx.mlx_array_free(k32t);
    const composed = try qsaScoreSheetComposed(s, q, k32t);
    defer _ = mlx.mlx_array_free(composed);
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);

    const row0s = [_]c_int{ 0, 2044, nb * ratio };
    for (row0s) |row0| {
        const all_vis = qsaAllBlocksVisible(row0, nb, ratio);
        const bounds_host = try ta.alloc(i32, @intCast(rows));
        defer ta.free(bounds_host);
        qsaVisibleBoundsHost(bounds_host, row0, rows, nb, ratio, all_vis);
        const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const ids_c = (try qsaSelectTopBlocks(s, composed, bounds, k)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(ids_c);
        const ids_f = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(ids_f);
        try testing.expect(try qsaArraysAllEqual(ids_c, ids_f, s));
    }
}

test "qsa score kernel: never built without NAX" {
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    nax_gate.vqmm_nax_probe_override = false;
    defer nax_gate.vqmm_nax_probe_override = null;
    const saved = qsa_score_kernel_cached;
    qsa_score_kernel_cached = null;
    defer qsa_score_kernel_cached = saved;
    try testing.expect(!qsaScoreFusedEligible(1, 4, 128, .bfloat16, .bfloat16));
    try testing.expect(!qsaScoreKernelBuilt());
    const builds = qsaScoreKernelBuilds();
    try testing.expectError(error.NaxUnavailable, getQsaScoreKernel());
    try testing.expectEqual(builds, qsaScoreKernelBuilds());
    try testing.expect(!qsaScoreKernelBuilt());
}

test "qsa score fused: a probe mismatch latches the composed arm" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaScoreTf32Enabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const saved_override = qsa_score_fused_override;
    const saved_flip = qsa_score_probe_flip_override;
    defer {
        qsa_score_fused_override = saved_override;
        qsa_score_probe_flip_override = saved_flip;
        qsaScoreFusedProbeReset();
    }
    qsa_score_fused_override = true;
    qsa_score_probe_flip_override = true;
    qsaScoreFusedProbeReset();
    qsaScoreFusedArm();
    try testing.expect(qsaScoreFusedProbeFailed());
    try testing.expect(!qsaScoreFusedActive());
    var prng = std.Random.DefaultPrng.init(0xF11);
    const q = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 2, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    try testing.expect((try qsaScoreFused(s, q, pooled)) == null);
    const k32t = try qsaScoreK32tFromPooled(s, pooled);
    defer _ = mlx.mlx_array_free(k32t);
    const composed = try qsaScoreSheetComposed(s, q, k32t);
    defer _ = mlx.mlx_array_free(composed);
    try testing.expect(composed.ctx != null);
}

test "qsa score fused: probe passes on this machine" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaScoreTf32Enabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const saved_override = qsa_score_fused_override;
    const saved_flip = qsa_score_probe_flip_override;
    defer {
        qsa_score_fused_override = saved_override;
        qsa_score_probe_flip_override = saved_flip;
        qsaScoreFusedProbeReset();
    }
    qsa_score_fused_override = true;
    qsa_score_probe_flip_override = false;
    qsaScoreFusedProbeReset();
    qsaScoreFusedArm();
    try testing.expect(!qsaScoreFusedProbeFailed());
    try testing.expect(qsaScoreFusedActive());
}

test "qsa score fused: default on engages, the override declines" {
    const saved_override = qsa_score_fused_override;
    defer qsa_score_fused_override = saved_override;
    qsa_score_fused_override = null;
    try testing.expect(qsaScoreFusedEnabled());
    qsa_score_fused_override = false;
    try testing.expect(!qsaScoreFusedEnabled());
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEF0);
    const q = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 4, 2, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    try testing.expect((try qsaScoreFused(s, q, pooled)) == null);
    qsa_score_fused_override = true;
    if (verifyQmmNaxAvailable() and qsaScoreTf32Enabled()) {
        const got = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
        defer _ = mlx.mlx_array_free(got);
    }
}

test "qsa score kernel: TF32 off declines" {
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_tf32_override = false;
    defer qsa_score_tf32_override = null;
    try testing.expect(!qsaScoreFusedEligible(1, 4, 128, .bfloat16, .bfloat16));
}

test "qsa score fused: the bill follows the predicate" {
    const saved_fused = qsa_score_fused_override;
    const saved_nax = nax_gate.vqmm_nax_probe_override;
    const saved_sheet = qwen4_qsa_attn.qsa_score_sheet_mb_override;
    defer {
        qsa_score_fused_override = saved_fused;
        nax_gate.vqmm_nax_probe_override = saved_nax;
        qwen4_qsa_attn.qsa_score_sheet_mb_override = saved_sheet;
    }
    qwen4_qsa_attn.qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    const n_idx: u64 = 4;
    const fwd: u64 = 4096;
    const nb: u64 = 262144;
    const kv = nb * 4;
    const ratio: u64 = 4;

    nax_gate.vqmm_nax_probe_override = false;
    qsa_score_fused_override = true;
    const composed_rows = qsaScoreRowsPerChunk(n_idx, nb, fwd, 128);
    const composed_bill = qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128);
    var cfg = model_mod.ModelConfig{};
    cfg.num_hidden_layers = 48;
    cfg.full_attention_interval = 4;
    cfg.indexer_budget = 2048;
    cfg.indexer_n_heads = 4;
    cfg.indexer_head_dim = 128;
    cfg.indexer_compress_ratio = 4;
    const composed_bank = cfg.qsaScoreBankBytesPerToken();
    try testing.expectEqual(@as(u64, 64), composed_rows);
    try testing.expectEqual(@as(u64, 1_536), composed_bank);

    qsa_score_fused_override = false;
    try testing.expectEqual(composed_rows, qsaScoreRowsPerChunk(n_idx, nb, fwd, 128));
    try testing.expectEqual(composed_bill, qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128));
    try testing.expectEqual(composed_bank, cfg.qsaScoreBankBytesPerToken());

    qsa_score_fused_override = true;
    nax_gate.vqmm_nax_probe_override = null;
    qsaScoreFusedArm();
    if (qsaScoreFusedActive()) {
        const fused_rows = qsaScoreRowsPerChunk(n_idx, nb, fwd, 128);
        try testing.expectEqual(composed_rows * 4, fused_rows);
        try testing.expect(fused_rows >= 128);
        const fused_bill = qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128);
        try testing.expectEqual(fused_rows * nb * 4 + fwd * @min(nb, 512) * 4, fused_bill);
        try testing.expectEqual(@as(u64, 0), cfg.qsaScoreBankBytesPerToken());
        const saved_sel = qsa_select_kernel_override;
        defer qsa_select_kernel_override = saved_sel;
        qsa_select_kernel_override = false;
        try testing.expectEqual(fused_rows * nb * 4 * 4 + fwd * @min(nb, 512) * 4, qsaPrefillTransientBytes(n_idx, fwd, kv, ratio, 128));
    }
}

test "qsa score fused: sizing policy" {
    try testing.expect(qsaScoreTf32ValueOn("1"));
    try testing.expect(qsaScoreTf32ValueOn(" 1"));
    try testing.expect(!qsaScoreTf32ValueOn("0"));
    try testing.expect(!qsaScoreTf32ValueOn("00"));
    try testing.expect(!qsaScoreTf32ValueOn("false"));
    try testing.expect(!qsaScoreTf32ValueOn(" 0"));
    try testing.expect(!qsaScoreTf32ValueOn("0x0"));
    try testing.expectEqual(QsaScoreLayout.base, qsaScoreLayoutFor(1));
    try testing.expectEqual(QsaScoreLayout.base, qsaScoreLayoutFor(127));
    try testing.expectEqual(QsaScoreLayout.h4, qsaScoreLayoutFor(128));
    try testing.expectEqual(@as(c_int, 4), qsaScoreNsg(.base));
    try testing.expectEqual(@as(c_int, 8), qsaScoreNsg(.h4));
    try testing.expectEqual(@as(c_int, 16), qsaScoreRowsPerTg(.base, 4));
    try testing.expectEqual(@as(c_int, 128), qsaScoreRowsPerTg(.h4, 8));
    try testing.expectEqual(@as(u32, 64), qsaScoreNsh(64, 1, .base));
    try testing.expectEqual(@as(u32, 32), qsaScoreNsh(1024, 8, .h4));
    const saved_fused = qsa_score_fused_override;
    const saved_nax = nax_gate.vqmm_nax_probe_override;
    const saved_sheet = qwen4_qsa_attn.qsa_score_sheet_mb_override;
    defer {
        qsa_score_fused_override = saved_fused;
        nax_gate.vqmm_nax_probe_override = saved_nax;
        qwen4_qsa_attn.qsa_score_sheet_mb_override = saved_sheet;
    }
    qwen4_qsa_attn.qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    const composed = qsaScoreRowsPerChunkFor(4 * 262144 * 4, 4096);
    const fused = qsaScoreRowsPerChunkFor(262144 * 4, 4096);
    try testing.expectEqual(composed * 4, fused);
    try testing.expect(fused >= 128);
}

test "qsa score fused: the argpartition decline arm consumes the fused sheet" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    qsa_score_layout_override = .h4;
    defer qsa_score_layout_override = null;
    var prng = std.Random.DefaultPrng.init(0xA2B0);
    const rnd = prng.random();
    const nb: c_int = 2048;
    const rows: c_int = 16;
    const k: c_int = 512;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    qsaVisibleBoundsHost(bounds_host, nb * 4, rows, nb, 4, true);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const ids_k = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(ids_k);
    const saved_sel = qsa_select_kernel_override;
    defer qsa_select_kernel_override = saved_sel;
    qsa_select_kernel_override = false;
    try testing.expect((try qsaSelectTopBlocks(s, fused, bounds, k)) == null);
    const bias = try qsaTieBiasForTest(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const vis = mlx.mlx_array{ .ctx = null };
    const ids_a = try qsaSelectComposedOps(s, fused, vis, bias, rows, nb, k, k, true);
    defer _ = mlx.mlx_array_free(ids_a);
    try testing.expect(try qsaArraysAllEqual(ids_k, ids_a, s));
}

test "qsa score fused: no bank is built under the fused arm" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    xfm.qsa_score_bank_builds = 0;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);
    entry.qsa_ratio = 4;
    entry.qsa_reserve_rows = 256;
    var prng = std.Random.DefaultPrng.init(0xBA11);
    const hd: c_int = 128;
    const nb: c_int = 32;
    const pooled = try attn256RandBf16(prng.random(), &[_]c_int{ 1, nb, hd }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try xfm.qsaAppendPooled(&entry, pooled, 0);
    const saved_fused = qsa_score_fused_override;
    const saved_nax = nax_gate.vqmm_nax_probe_override;
    defer {
        qsa_score_fused_override = saved_fused;
        nax_gate.vqmm_nax_probe_override = saved_nax;
    }
    qsa_score_fused_override = false;
    nax_gate.vqmm_nax_probe_override = false;
    _ = try xfm.qsaScoreOperand(&entry, 1, nb, hd, false);
    try testing.expect(entry.qsa_score_bank.ctx != null);
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    entry.qsa_score_bank = .{ .ctx = null };
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    qsa_score_fused_override = true;
    nax_gate.vqmm_nax_probe_override = null;
    qsaScoreFusedArm();
    if (qsaScoreFusedActive()) {
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, hd, true);
        try testing.expect(entry.qsa_score_bank.ctx == null);
    }
}

test "qsa score fused: dump hook writes a parseable line" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    var file_buf: [1024]u8 = undefined;
    const path = try std.fmt.bufPrint(&file_buf, "{s}/qsa-blocks.txt", .{path_buf[0..root_len]});
    qwen4_qsa.qsa_dump_blocks_path_override = path;
    defer qwen4_qsa.qsa_dump_blocks_path_override = null;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    qsa_score_fused_override = true;
    defer qsa_score_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0xD00B);
    const rnd = prng.random();
    const nb: c_int = 64;
    const rows: c_int = 2;
    const k: c_int = 8;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
    defer _ = mlx.mlx_array_free(q);
    const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
    defer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_array_eval(pooled));
    const fused = (try qsaScoreFused(s, q, pooled)) orelse return error.ScoreKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    qsaVisibleBoundsHost(bounds_host, nb * 4, rows, nb, 4, true);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const ids = (try qsaSelectTopBlocks(s, fused, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(ids);
    qsaDumpQsaBlocks(3, 12, rows, k, ids);
    const body = try tmp.dir.readFileAlloc(std.testing.io, "qsa-blocks.txt", ta, .limited(1 << 20));
    defer ta.free(body);
    try testing.expect(std.mem.startsWith(u8, body, "layer=3 row0=12 rows=2 kb=8\n"));
    try testing.expect(std.mem.indexOfScalar(u8, body, ' ') != null);
}

test "qsa select tg policy: a pure function of rows" {
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(4096));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(1024));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(256));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(255));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(64));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(63));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(16));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(1));
}
