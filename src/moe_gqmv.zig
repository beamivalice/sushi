const trf = @import("transformer.zig");
const std = @import("std");
const expert_stream_mod = @import("expert_stream.zig");
const mlx = @import("mlx.zig");
const test_support = @import("test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const testReadF32 = test_support.testReadF32;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const QuantMode = model_mod.QuantMode;
const testing = std.testing;
const attn_kernels = @import("attn_kernels.zig");
const QsaCfgCache = attn_kernels.QsaCfgCache;
const verify_qmm = @import("verify_qmm.zig");
const cachedScalarInt = verify_qmm.cachedScalarInt;
const fusedSwiGLU = trf.fusedSwiGLU;
const gatherExpertMm = trf.gatherExpertMm;
const moeRowsAffineBank = trf.moeRowsAffineBank;
const moeRowsSliceRow = trf.moeRowsSliceRow;
const seHash = trf.seHash;
const swigluSigTable = trf.swigluSigTable;

fn gatherQmvSource(comptime x_per_expert: bool) [:0]const u8 {
    // Everything is indexed off the raw buffer names rather than through
    // explicitly address-spaced pointers: MLX generates the signature from each
    // input's ACTUAL dtype and puts arrays with fewer than 8 elements in
    // `constant` instead of `device` (backend/common/metal_kernel.cpp), so a
    // hand-declared `const device T*` would mismatch on both counts. `x` in
    // particular can arrive as float32 while scales/biases are bf16.
    //
    // A uint4 (128-bit) load variant was measured and is NOT kept: 38.9 us vs
    // 37.3 for the scalar form. At ~5 ALU ops per dequantized value this loop
    // is ALU-bound, not load-width bound, so widening the loads only adds code.
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;      // output row within the expert
        \\uint e = thread_position_in_grid.z;      // top-K slot
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;  // values per pack: a word, or 3 bytes at 3-bit
        \\int K_by_p = K / VPW;                    // packs per row
        \\int K_by_w = K * BITS / 32;              // words per row
        \\int K_by_gs = K / GS;                    // quant groups per row
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[e];                      // bank row for this slot
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xoff = {s};
        \\
        \\// Four independent accumulators so the per-lane FMA chain is not one
        \\// serial dependency; VPW is 16/8/4 for bits 2/4/8, always a multiple of 4.
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
        \\  uint32_t packed = sushi_qpack<BITS>(w_q, wbase, pack);
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;                  // GS >= VPW, so one group per word
        \\{s}
        \\}}
        \\float acc = simd_sum((a0 + a1) + (a2 + a3));
        \\if (lane == 0) {{
        \\  y[(size_t)e * (size_t)N + (size_t)n] = T(acc);
        \\}}
    , .{
        if (x_per_expert) "(size_t)e * (size_t)K" else "0",
        GQMV_BODY_AFFINE,
    });
}

const GQMV_BODY_AFFINE =
    \\  float sj = float(scales[gbase + (size_t)gi]);
    \\  float bj = float(biases[gbase + (size_t)gi]);
    \\  for (int ki = 0; ki < VPW; ki += 4) {
    \\    size_t xi = xoff + (size_t)(k_base + ki);
    \\    uint32_t q = packed >> (ki * BITS);
    \\    a0 += float(x[xi + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
    \\    a1 += float(x[xi + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
    \\    a2 += float(x[xi + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
    \\    a3 += float(x[xi + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
    \\  }
;

// One packed unit of quantized values as a uint32: a word for 2/4/8-bit; at
// 3-bit, mx.quantize packs 8 values into 3 bytes (value i at bit 3i), so the
// unit is a byte triple and the body's `>> (i * BITS)` walk is unchanged.
const GQMV_AFFINE_HEADER =
    \\template <int B>
    \\inline uint32_t sushi_qpack(const device uint32_t* w, size_t wbase_words, int pack) {
    \\  if (B == 3) {
    \\    const device uchar* wb = (const device uchar*)w + wbase_words * 4 + (size_t)pack * 3;
    \\    return uint32_t(wb[0]) | (uint32_t(wb[1]) << 8) | (uint32_t(wb[2]) << 16);
    \\  }
    \\  return w[wbase_words + (size_t)pack];
    \\}
;

// ── Fused routed gate+up+SwiGLU (one kernel for what was three) ──
//
// The decode expert path ran gate gather -> up gather -> activation as three
// dispatches, all on the critical chain into down_proj. This computes BOTH dot
// products in the SAME simdgroup and applies the activation before it writes,
// so the x element loaded for the gate FMA is reused for the up FMA and the two
// dequant chains interleave instead of running as separate launches.
//
// It halves the simdgroup count rather than the work (each does two rows'
// worth), which is the right trade here: 4096 simdgroups over 40 cores was ~100
// per core, so there is occupancy to spend on instruction-level parallelism.
//
// Bit-exactness comes from rounding each accumulator to T FIRST — exactly where
// `gatherQmv` writes `T(acc)` — and then running the same table-lookup SwiGLU
// `fusedSwiGLU` does. No f32 shortcut through the activation.
fn gatherQmvGateUpSource() [:0]const u8 {
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;      // output row within the expert
        \\uint e = thread_position_in_grid.z;      // top-K slot
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_p = K / VPW;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[e];
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xrow = 0;
        \\
        \\// Every lane's packed words are issued up front (KP = K_by_p is a
        \\// template constant): −0.7% per forward in idle same-boot pairs vs
        \\// the runtime-count loop. Same per-lane accumulation order.
        \\constexpr int IT = (KP + 31) / 32;
        \\uint32_t pg[IT];
        \\uint32_t pu[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  pg[i] = (pack < KP) ? sushi_qpack<BITS>(wg_q, wbase, pack) : 0u;
        \\  pu[i] = (pack < KP) ? sushi_qpack<BITS>(wu_q, wbase, pack) : 0u;
        \\}}
        \\// Two independent 4-way accumulator sets: the gate chain and the up
        \\// chain never wait on each other.
        \\float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        \\float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed_g = pg[i];
        \\  uint32_t packed_u = pu[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc_g = simd_sum((g0 + g1) + (g2 + g3));
        \\float acc_u = simd_sum((u0 + u1) + (u2 + u3));
        \\if (lane == 0) {{
        \\  // Round exactly where the unfused path's two kernels wrote T(acc),
        \\  // then the same table-lookup SwiGLU.
        \\  T gt = T(acc_g);
        \\  T ut = T(acc_u);
        \\  T sig = sigtab[as_type<ushort>(gt)];
        \\  y[(size_t)e * (size_t)N + (size_t)n] = (gt * sig) * ut;
        \\}}
    , .{
        GQMV_GATEUP_BODY_AFFINE,
    });
}

const GQMV_GATEUP_BODY_AFFINE =
    \\  float sjg = float(g_scales[gbase + (size_t)gi]);
    \\  float bjg = float(g_biases[gbase + (size_t)gi]);
    \\  float sju = float(u_scales[gbase + (size_t)gi]);
    \\  float bju = float(u_biases[gbase + (size_t)gi]);
    \\  for (int ki = 0; ki < VPW; ki += 4) {
    \\    size_t xi = xrow + (size_t)(k_base + ki);
    \\    uint32_t qg = packed_g >> (ki * BITS);
    \\    uint32_t qu = packed_u >> (ki * BITS);
    \\    float x0 = float(x[xi + 0]);
    \\    float x1 = float(x[xi + 1]);
    \\    float x2 = float(x[xi + 2]);
    \\    float x3 = float(x[xi + 3]);
    \\    g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
    \\    g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
    \\    g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
    \\    g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
    \\    u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
    \\    u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
    \\    u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
    \\    u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
    \\  }
;

const GQMV_GATEUP_SOURCE = gatherQmvGateUpSource();
var gqmv_gateup_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvGateUpKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_gateup_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_gateup",
        in_vec,
        out_vec,
        GQMV_GATEUP_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_kernel = kernel;
    return kernel;
}














/// One quant group per pack: a word for 2/4/8-bit, a byte triple (8 values) for 3-bit.
fn affineQmvUnitOk(bits: u32, group_size: u32) bool {
    return switch (bits) {
        2, 4, 8 => group_size % (32 / bits) == 0,
        3 => group_size % 8 == 0,
        else => false,
    };
}

fn affineQmvPackUnits(K: c_int, bits: u32) c_int {
    return if (bits == 3) @divExact(K, 8) else @divExact(K * @as(c_int, @intCast(bits)), 32);
}

const GateUpCfgKey = struct { topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var gateup_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_cfg_key: GateUpCfgKey = std.mem.zeroes(GateUpCfgKey);
var gqmv_gateup_engaged: bool = false;
pub var gqmv_gateup_override: ?bool = null;

pub fn gatherQmvGateUpEnabled() bool {
    return gqmv_gateup_override orelse true;
}

/// One kernel for `silu(gather(gate) @ x) * (gather(up) @ x)` at decode width.
/// Returns [TOPK, N] in x's dtype, or null when anything is outside the
/// supported set (caller falls back to the two separate gathers + activation).
pub fn gatherQmvGateUp(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (gb.ctx == null or ub.ctx == null) return null;
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    // The activation is a 16-bit table lookup, exactly as `fusedSwiGLU` gates
    // it; f32 activations keep the unfused path so the two can never disagree.
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    // gate and up must be the SAME geometry — the kernel walks one index set
    // through both banks.
    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpCfgKey{ .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    if (gateup_cfg == null or !std.meta.eql(gateup_cfg_key, key)) {
        if (gateup_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        gateup_cfg = config;
        gateup_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_engaged) {
        gqmv_gateup_engaged = true;
        log.info("[moe] fused gate+up+SwiGLU kernel engaged: E_topk={d} inter={d} bits={d} gs={d}\n", .{ topk, N, bits, group_size });
    }
    return y;
}

fn gatherQmvGateUpRowsSource() [:0]const u8 {
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint n = thread_position_in_grid.y;
        \\uint combo = thread_position_in_grid.z;
        \\uint e = combo % uint(TOPK);
        \\uint row = combo / uint(TOPK);
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_p = K / VPW;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[(size_t)row * (size_t)TOPK + e];
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\size_t xrow = (size_t)row * (size_t)K;
        \\
        \\constexpr int IT = (KP + 31) / 32;
        \\uint32_t pg[IT];
        \\uint32_t pu[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  pg[i] = (pack < KP) ? sushi_qpack<BITS>(wg_q, wbase, pack) : 0u;
        \\  pu[i] = (pack < KP) ? sushi_qpack<BITS>(wu_q, wbase, pack) : 0u;
        \\}}
        \\float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        \\float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(lane) + 32 * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed_g = pg[i];
        \\  uint32_t packed_u = pu[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc_g = simd_sum((g0 + g1) + (g2 + g3));
        \\float acc_u = simd_sum((u0 + u1) + (u2 + u3));
        \\if (lane == 0) {{
        \\  T gt = T(acc_g);
        \\  T ut = T(acc_u);
        \\  T sig = sigtab[as_type<ushort>(gt)];
        \\  y[(((size_t)row * (size_t)TOPK) + e) * (size_t)N + (size_t)n] = (gt * sig) * ut;
        \\}}
    , .{
        GQMV_GATEUP_BODY_AFFINE,
    });
}

const GQMV_GATEUP_ROWS_SOURCE = gatherQmvGateUpRowsSource();
var gqmv_gateup_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvGateUpRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_gateup_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_gateup_rows",
        in_vec,
        out_vec,
        GQMV_GATEUP_ROWS_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_rows_kernel = kernel;
    return kernel;
}

const GateUpRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var gateup_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_rows_cfg_key: GateUpRowsCfgKey = std.mem.zeroes(GateUpRowsCfgKey);
var gqmv_gateup_rows_engaged: bool = false;
pub var moe_rows_fused_override: ?bool = null;
pub var moe_rows_fused_layers: u32 = 0;
pub var moe_sorted_layers: u32 = 0;

pub fn moeRowsFusedEnabled() bool {
    return moe_rows_fused_override orelse true;
}

pub const MoeDecodeDispatchArm = enum { rows, sorted, gather_qmv };

pub fn moeDecodeDispatchArm(B: c_int, S: c_int, K: c_int) MoeDecodeDispatchArm {
    const total_inds: c_int = B * S * K;
    const do_sort = B * S > 1 or total_inds >= 64;
    if (B >= 1 and B <= 8 and S >= 1 and B * S >= 2 and B * S <= 16 and moeRowsFusedEnabled() and
        !trf.qwen4Standin().moe_gateup and !trf.qwen4Standin().moe_down)
        return .rows;
    if (do_sort) return .sorted;
    return .gather_qmv;
}

/// Row-axis twin of `gatherQmvGateUp` for the batched decode tick (S = 1,
/// 2 <= B <= 8): the grid runs over output tiles x (row, expert choice) and
/// each row is the single-row arithmetic unchanged, so the result is
/// bit-identical per row to B solo calls. Shares the single-row Metal body.
pub fn gatherQmvGateUpRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (gb.ctx == null or ub.ctx == null) return null;
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 128 or topk < 1 or topk > 16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2 or xsh[0] != nrows or xsh[1] != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    if (gateup_rows_cfg == null or !std.meta.eql(gateup_rows_cfg_key, key)) {
        if (gateup_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 3, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, nrows * topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPK", topk));
        gateup_rows_cfg = config;
        gateup_rows_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_rows_engaged) {
        gqmv_gateup_rows_engaged = true;
        log.info("[moe] fused rows gate+up+SwiGLU kernel engaged: B={d} E_topk={d} inter={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

const GQMV_SOURCES = [2][:0]const u8{ gatherQmvSource(false), gatherQmvSource(true) };
const GQMV_NAMES = [2][*:0]const u8{ "sushi_moe_gather_qmv", "sushi_moe_gather_qmv_px" };
var gqmv_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvKernel(x_per_expert: bool) !mlx.mlx_fast_metal_kernel {
    const xi: usize = @intFromBool(x_per_expert);
    if (gqmv_kernels[xi]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_NAMES[xi],
        in_vec,
        out_vec,
        GQMV_SOURCES[xi],
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_kernels[xi] = kernel;
    return kernel;
}

/// Everything a `gatherQmv` config bakes in; one slot per input layout, since
/// the gate/up (shared-x) and down (per-expert-x) calls alternate every layer.
const GqmvCfgKey = struct {
    topk: c_int,
    n: c_int,
    bits: u32,
    gs: u32,
    dtype: mlx.mlx_dtype,
    x_per_expert: bool,
};
const GqmvCfgSlot = struct {
    cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    key: GqmvCfgKey = std.mem.zeroes(GqmvCfgKey),
};
var gqmv_cfg_cache: [2]GqmvCfgSlot = .{ .{}, .{} };

pub var gqmv_engaged: bool = false;
pub var decode_ladder_engaged: bool = false;

/// Decode-width gathered qmv over an expert bank, read in place.
/// `x`: [K] shared (x_per_expert=false) or [TOPK, K] (true), bf16/fp16.
/// `w`/`sc`/`bi`: the affine bank, [E, N, K*BITS/32] / [E, N, K/GS].
/// `inds`: [TOPK] uint32 bank rows. Returns [TOPK, N], or null when the shape
/// or quant geometry is outside the supported set (caller falls back).
pub fn gatherQmv(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
) !?mlx.mlx_array {
    // Any other mode (mxfp4/mxfp8, whose scales are e8m0 over different group
    // sizes) falls back — running a bank through the wrong dequant is silently
    // wrong, not a crash.
    if (mode != .affine) return null;
    // Only widths where one uint32 holds a whole number of values (5/6-bit
    // affine is byte-packed differently — those fall back).
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    const want: i64 = if (x_per_expert) @as(i64, topk) * K else K;
    if (xelems != want) return null;

    // One simdgroup per (slot, row); 8 simdgroups to a threadgroup.
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    // CACHED, not rebuilt per call. This runs three times per MoE layer, and a
    // fresh config is ~10 FFI calls: measured `ops/forward` 3226 vs 2329 and
    // +0.4 ms of CPU graph build per token on a 40-layer MoE, which is roughly 60%
    // of the margin this path was being charged against stock `gather_qmm`.
    const key = GqmvCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd, .x_per_expert = x_per_expert };
    const slot = &gqmv_cfg_cache[@intFromBool(x_per_expert)];
    if (slot.cfg == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        slot.* = .{ .cfg = cfg, .key = key };
    }
    const config = slot.cfg.?;

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvKernel(x_per_expert);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

// ── Fused MoE down + router weighting + K-reduction (mlxfast down-path mega-fusion class) ──
//
// The decode expert tail ran down-gather -> score multiply -> sum over K as
// three dispatches, materializing a [K, hidden] intermediate between them.
// This kernel keeps the down projection's per-row arithmetic TEXTUALLY
// identical to `gatherQmv` (same GQMV_BODY_*, same lane-strided pack loop,
// same simd_sum tree, same T(acc) rounding — so each (slot, row) value is
// bit-identical to the unfused kernel's), parks the K per-row values in
// threadgroup memory, and finishes with the composed pair's exact semantics:
// per-slot product rounded to T (the elementwise `mlx_multiply`), then an
// ascending T accumulate over slots 0..K-1 (MLX's bf16 sum reduction
// accumulates in the INPUT dtype — the moeRouterTopK lesson). One dispatch
// replaces three, and the [K, hidden] round-trip never happens.
//
// Each simdgroup owns one top-K slot and computes ROWS consecutive output
// rows for it (mlxfast tuned 1-vs-4 rows per simd; 4 won). Bit-identity is
// pinned by the "down+reduce" parity tests.
fn gatherQmvDownReduceSource() [:0]const u8 {
    // LPR lanes share one output row and ROWS = 32 / LPR rows ride one simdgroup
    // in parallel: K is short (the MoE intermediate), so one 32-lane reduction
    // per row cost more than the dot product it reduced, and K/8 packs over 32
    // lanes left a third of them idle. Every row's packs split exactly over LPR
    // lanes, and one log2(LPR)-step shuffle reduces all ROWS rows at once.
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint slot = simdgroup_index_in_threadgroup;   // top-K slot
        \\uint tile = threadgroup_position_in_grid.x;   // block of ROWS output rows
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[slot];
        \\size_t xoff = (size_t)slot * (size_t)K;
        \\threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
        \\uint r = lane / uint(LPR);
        \\uint sub = lane % uint(LPR);
        \\uint n = tile * uint(ROWS) + r;
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\constexpr int IT = (KP + LPR - 1) / LPR;
        \\uint32_t pw[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  pw[i] = (pack < KP) ? sushi_qpack<BITS>(w_q, wbase, pack) : 0u;
        \\}}
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed = pw[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc = (a0 + a1) + (a2 + a3);
        \\for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
        \\if (sub == 0u) {{
        \\  slot_vals[slot * uint(ROWS) + r] = T(acc);
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\// Composed-pair replica: T-rounded per-slot product, ascending T sum.
        \\if (slot == 0 && lane < uint(ROWS)) {{
        \\  T total = T(0.0f);
        \\  for (uint s2 = 0; s2 < uint(TOPK); ++s2) {{
        \\    T p = slot_vals[s2 * uint(ROWS) + lane] * scores[s2];
        \\    total = total + p;
        \\  }}
        \\  y[(size_t)tile * (size_t)ROWS + (size_t)lane] = total;
        \\}}
    , .{
        GQMV_BODY_AFFINE,
    });
}

const GQMV_DOWNRED_SOURCE = gatherQmvDownReduceSource();
var gqmv_downred_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvDownReduceKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_downred_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_downred",
        in_vec,
        out_vec,
        GQMV_DOWNRED_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_kernel = kernel;
    return kernel;
}

/// Lanes per output row in the down+reduce kernels; 32/LPR rows per simdgroup.
/// 4 spills its hoisted packs, 16 measured the same as 8 on the shipped shapes.
const DOWNRED_LPR: c_int = 8;
const DownRedCfgKey = struct { topk: c_int, n: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var downred_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_cfg_key: DownRedCfgKey = std.mem.zeroes(DownRedCfgKey);
pub var downred_engaged: bool = false;
pub var downred_override: ?bool = null;

fn downReduceFusedEnabled() bool {
    return downred_override orelse true;
}

/// Fused down-projection + router weighting + K-reduction at decode width.
/// `x`: [TOPK, K] per-expert activations; `scores`: [TOPK] router weights in
/// the SAME dtype as x (the composed multiply is elementwise in that dtype —
/// a differing dtype would promote, so it declines). Returns the weighted
/// expert sum [N], or null when outside the kernel's envelope.
pub fn gatherQmvDownReduce(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    if (topk <= 0 or topk > 32) return null; // one simdgroup per slot, <= 1024 threads
    const ssh = mlx.getShape(scores);
    if (ssh.len != 1 or ssh[0] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, topk) * K) return null;
    const rows: c_int = @divExact(32, DOWNRED_LPR);
    if (@rem(N, rows) != 0) return null;

    const key = DownRedCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd };
    if (downred_cfg == null or !std.meta.eql(downred_cfg_key, key)) {
        if (downred_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{N};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 1, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", DOWNRED_LPR));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_cfg = cfg;
        downred_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

fn gatherQmvDownReduceRowsSource() [:0]const u8 {
    // The solo kernel's layout with a batch row on grid.y; per-row output must
    // stay bit-identical to the solo call (the rows-vs-solo test pins it).
    return std.fmt.comptimePrint(
        \\auto lane = thread_index_in_simdgroup;
        \\uint slot = simdgroup_index_in_threadgroup;
        \\uint tile = threadgroup_position_in_grid.x;
        \\uint row = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\constexpr int VPW = (BITS == 3) ? 8 : 32 / BITS;
        \\int K_by_w = K * BITS / 32;
        \\int K_by_gs = K / GS;
        \\uint mask = (1u << BITS) - 1u;
        \\
        \\uint eid = inds[(size_t)row * (size_t)TOPK + slot];
        \\size_t xoff = ((size_t)row * (size_t)TOPK + (size_t)slot) * (size_t)K;
        \\threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
        \\uint r = lane / uint(LPR);
        \\uint sub = lane % uint(LPR);
        \\uint n = tile * uint(ROWS) + r;
        \\size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_w + (size_t)n * (size_t)K_by_w;
        \\size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
        \\float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        \\constexpr int IT = (KP + LPR - 1) / LPR;
        \\uint32_t pw[IT];
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  pw[i] = (pack < KP) ? sushi_qpack<BITS>(w_q, wbase, pack) : 0u;
        \\}}
        \\for (int i = 0; i < IT; ++i) {{
        \\  int pack = int(sub) + LPR * i;
        \\  if (pack >= KP) break;
        \\  uint32_t packed = pw[i];
        \\  int k_base = pack * VPW;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\float acc = (a0 + a1) + (a2 + a3);
        \\for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
        \\if (sub == 0u) {{
        \\  slot_vals[slot * uint(ROWS) + r] = T(acc);
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (slot == 0 && lane < uint(ROWS)) {{
        \\  T total = T(0.0f);
        \\  for (uint s2 = 0; s2 < uint(TOPK); ++s2) {{
        \\    T p = slot_vals[s2 * uint(ROWS) + lane] * scores[(size_t)row * (size_t)TOPK + s2];
        \\    total = total + p;
        \\  }}
        \\  y[(size_t)row * (size_t)N + (size_t)tile * (size_t)ROWS + (size_t)lane] = total;
        \\}}
    , .{
        GQMV_BODY_AFFINE,
    });
}

const GQMV_DOWNRED_ROWS_SOURCE = gatherQmvDownReduceRowsSource();
var gqmv_downred_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getGatherQmvDownReduceRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (gqmv_downred_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_moe_gather_qmv_downred_rows",
        in_vec,
        out_vec,
        GQMV_DOWNRED_ROWS_SOURCE,
        GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_downred_rows_kernel = kernel;
    return kernel;
}

const DownRedRowsCfgKey = struct { nrows: c_int, topk: c_int, n: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var downred_rows_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var downred_rows_cfg_key: DownRedRowsCfgKey = std.mem.zeroes(DownRedRowsCfgKey);
var downred_rows_engaged: bool = false;

/// Asked BEFORE the gate/up dispatch: a decline is static per geometry and
/// would otherwise cost a wasted gate/up launch on every MoE layer of every tick.
pub fn gatherQmvDownReduceRowsEligible(hidden_act: anytype, gate_qp: anytype, up_qp: anytype, down_qp: anytype, N: c_int) bool {
    if (!downReduceFusedEnabled()) return false;
    if (!moeRowsFusedEnabled()) return false;
    if (hidden_act != .silu) return false;
    if (gate_qp.bits != up_qp.bits or gate_qp.group_size != up_qp.group_size or gate_qp.mode != up_qp.mode) return false;
    if (down_qp.mode != .affine) return false;
    if (down_qp.bits != 2 and down_qp.bits != 4 and down_qp.bits != 8) return false;
    if (down_qp.group_size % (32 / down_qp.bits) != 0) return false;
    if (@rem(N, 4) != 0) return false;
    return true;
}

/// Row-axis twin of `gatherQmvDownReduce`: the K-expert threadgroup reduction
/// runs per row (grid.y = B) in the single-row order, never redistributed
/// across rows, so each row is bit-identical to its solo call.
pub fn gatherQmvDownReduceRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    scores: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
) !?mlx.mlx_array {
    if (!downReduceFusedEnabled()) return null;
    if (!moeRowsFusedEnabled()) return null;
    if (mode != .affine) return null;
    if (!affineQmvUnitOk(bits, group_size)) return null;
    if (bi.ctx == null) return null;
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(scores) != xd) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 2) return null;
    const nrows: c_int = ish[0];
    const topk: c_int = ish[1];
    if (nrows < 1 or nrows > 128 or topk <= 0 or topk > 32) return null;
    const ssh = mlx.getShape(scores);
    if (ssh.len != 2 or ssh[0] != nrows or ssh[1] != topk) return null;
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != @as(i64, nrows) * topk * K) return null;
    const rows: c_int = @divExact(32, DOWNRED_LPR);
    if (@rem(N, rows) != 0) return null;

    const key = DownRedRowsCfgKey{ .nrows = nrows, .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd };
    if (downred_rows_cfg == null or !std.meta.eql(downred_rows_cfg_key, key)) {
        if (downred_rows_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ nrows, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(N, rows) * topk * 32, nrows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", DOWNRED_LPR));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KP", affineQmvPackUnits(K, bits)));
        downred_rows_cfg = cfg;
        downred_rows_cfg_key = key;
    }

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, inds, scores, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvDownReduceRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, downred_rows_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!downred_rows_engaged) {
        downred_rows_engaged = true;
        log.info("[moe] fused rows down+reduce kernel engaged: B={d} E_topk={d} hidden={d} bits={d} gs={d}\n", .{ nrows, topk, N, bits, group_size });
    }
    return y;
}

const SUSHI_QMV_ROWS_SOURCE =
    \\using U = float;
    \\auto simd_lid = thread_position_in_threadgroup.x;
    \\auto simd_gid = thread_position_in_threadgroup.y;
    \\auto tg = threadgroup_position_in_grid;
    \\const int in_vec_size = int(K_size);
    \\const int out_vec_size = int(N_size);
    \\constexpr int packs_per_thread = 2;
    \\constexpr int pack_factor = BITS == 3 ? 8 : 32 / BITS;
    \\constexpr int bytes_per_pack = BITS == 3 ? 3 : 4;
    \\constexpr int values_per_thread = pack_factor * packs_per_thread;
    \\constexpr int block_size = values_per_thread * 32;
    \\constexpr int scale_step_per_thread = GS / values_per_thread;
    \\const device uint8_t* ws = (const device uint8_t*)w_q;
    \\const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    \\const int in_vec_size_g = in_vec_size / GS;
    \\const int out_row = int(tg.y) * 8 + int(simd_gid) * 4;
    \\ws += out_row * in_vec_size_w + int(simd_lid) * packs_per_thread * bytes_per_pack;
    \\const device T* scales_p = scales + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
    \\const device T* biases_p = biases + out_row * in_vec_size_g + int(simd_lid) / scale_step_per_thread;
    \\const device T* xp = x + int(tg.z) * in_vec_size + int(simd_lid) * values_per_thread;
    \\device T* yp = y + int(tg.z) * out_vec_size + out_row;
    \\thread U x_thread[values_per_thread];
    \\thread U result[4] = {0, 0, 0, 0};
    \\for (int k = 0; k < in_vec_size; k += block_size) {
    \\  U sum = 0;
    \\  if (BITS == 3) {
    \\    for (int i = 0; i < values_per_thread; i += 8) {
    \\      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3] + xp[i + 4] + xp[i + 5] + xp[i + 6] + xp[i + 7];
    \\      x_thread[i] = xp[i];
    \\      x_thread[i + 1] = xp[i + 1] / 8.0f;
    \\      x_thread[i + 2] = xp[i + 2] / 64.0f;
    \\      x_thread[i + 3] = xp[i + 3] / 2.0f;
    \\      x_thread[i + 4] = xp[i + 4] / 16.0f;
    \\      x_thread[i + 5] = xp[i + 5] / 128.0f;
    \\      x_thread[i + 6] = xp[i + 6] / 4.0f;
    \\      x_thread[i + 7] = xp[i + 7] / 32.0f;
    \\    }
    \\  } else if (BITS == 4) {
    \\    for (int i = 0; i < values_per_thread; i += 4) {
    \\      sum += xp[i] + xp[i + 1] + xp[i + 2] + xp[i + 3];
    \\      x_thread[i] = xp[i];
    \\      x_thread[i + 1] = xp[i + 1] / 16.0f;
    \\      x_thread[i + 2] = xp[i + 2] / 256.0f;
    \\      x_thread[i + 3] = xp[i + 3] / 4096.0f;
    \\    }
    \\  } else {
    \\    for (int i = 0; i < values_per_thread; i++) {
    \\      sum += xp[i];
    \\      x_thread[i] = xp[i];
    \\    }
    \\  }
    \\  for (int rr = 0; rr < 4; rr++) {
    \\    const device uint8_t* wl = ws + rr * in_vec_size_w;
    \\    U scv = scales_p[rr * in_vec_size_g];
    \\    U biv = biases_p[rr * in_vec_size_g];
    \\    U accum = 0;
    \\    if (BITS == 3) {
    \\      for (int i = 0; i < values_per_thread / 8; i++) {
    \\        const device uint8_t* w3 = wl + 3 * i;
    \\        thread U* x3 = x_thread + 8 * i;
    \\        accum += (w3[0] & 0x07) * x3[0];
    \\        accum += (w3[0] & 0x38) * x3[1];
    \\        accum += (w3[0] & 0xc0) * x3[2];
    \\        accum += (w3[1] & 0x01) * (x3[2] * 256.0f);
    \\        accum += (w3[1] & 0x0e) * x3[3];
    \\        accum += (w3[1] & 0x70) * x3[4];
    \\        accum += (w3[1] & 0x80) * x3[5];
    \\        accum += (w3[2] & 0x03) * (x3[5] * 256.0f);
    \\        accum += (w3[2] & 0x1c) * x3[6];
    \\        accum += (w3[2] & 0xe0) * x3[7];
    \\      }
    \\    } else if (BITS == 4) {
    \\      const device uint16_t* w16 = (const device uint16_t*)wl;
    \\      for (int i = 0; i < (values_per_thread / 4); i++) {
    \\        accum += (x_thread[4 * i] * (w16[i] & 0x000f) + x_thread[4 * i + 1] * (w16[i] & 0x00f0) + x_thread[4 * i + 2] * (w16[i] & 0x0f00) + x_thread[4 * i + 3] * (w16[i] & 0xf000));
    \\      }
    \\    } else {
    \\      for (int i = 0; i < values_per_thread; i++) {
    \\        accum += x_thread[i] * wl[i];
    \\      }
    \\    }
    \\    result[rr] += scv * accum + sum * biv;
    \\  }
    \\  ws += block_size * bytes_per_pack / pack_factor;
    \\  scales_p += block_size / GS;
    \\  biases_p += block_size / GS;
    \\  xp += block_size;
    \\}
    \\for (int rr = 0; rr < 4; rr++) {
    \\  result[rr] = simd_sum(result[rr]);
    \\  if (simd_lid == 0) {
    \\    yp[rr] = static_cast<T>(result[rr]);
    \\  }
    \\}
;

pub var sushi_qmv_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;
const MsvQmvRowsCfgKey = struct { nrows: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype };
var sushi_qmv_rows_cfgs = QsaCfgCache(MsvQmvRowsCfgKey, 1){};
var sushi_qmv_rows_logged = false;
pub var mtp_head_row_dispatches: usize = 0;

fn getMsvQmvRowsKernel() !mlx.mlx_fast_metal_kernel {
    if (sushi_qmv_rows_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qmv_rows",
        in_vec,
        out_vec,
        SUSHI_QMV_ROWS_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    sushi_qmv_rows_kernel = kernel;
    return kernel;
}

pub fn msvQmvRows(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (bits != 3 and bits != 4 and bits != 8) return null;
    if (group_size != 64) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(sc) != xd or mlx.mlx_array_dtype(bi) != xd) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 1 or xsh.len > 8) return null;
    const K = xsh[xsh.len - 1];
    var nrows: c_int = 1;
    var i: usize = 0;
    while (i + 1 < xsh.len) : (i += 1) nrows *= xsh[i];
    if (nrows < 2 or nrows > 16) return null;
    const vpt: c_int = if (bits == 3) 16 else @intCast((32 / bits) * 2);
    const block: c_int = vpt * 32;
    if (@rem(K, block) != 0) return null;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const Nout = wsh[0];
    if (@rem(Nout, 8) != 0) return null;
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * bits) return null;
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    if (xsh.len == 2 and xsh[0] == nrows) {
        try mlx.check(mlx.mlx_array_set(&x2, x));
    } else {
        try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ nrows, K }, 2, s));
    }

    const key = MsvQmvRowsCfgKey{ .nrows = nrows, .n = Nout, .k = K, .bits = bits, .gs = group_size, .dtype = xd };
    const config = blk: {
        if (sushi_qmv_rows_cfgs.get(key)) |hit| break :blk hit[0];
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const y_shape = [_]c_int{ nrows, Nout };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, @divExact(Nout, 4), nrows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        sushi_qmv_rows_cfgs.put(key, .{cfg});
        break :blk cfg;
    };

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(Nout);
    const inputs = [_]mlx.mlx_array{ x2, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getMsvQmvRowsKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    mtp_head_row_dispatches += 1;
    if (!sushi_qmv_rows_logged) {
        sushi_qmv_rows_logged = true;
        log.info("[mtp] row-axis projection engaged: rows={d} K={d} N={d} bits={d} gs={d}\n", .{ nrows, K, Nout, bits, group_size });
    }
    var y2 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));
    if (xsh.len == 2) return y2;
    var ysh_buf: [8]c_int = undefined;
    for (xsh[0 .. xsh.len - 1], 0..) |d, di| ysh_buf[di] = d;
    ysh_buf[xsh.len - 1] = Nout;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &ysh_buf, @intCast(xsh.len), s));
    _ = mlx.mlx_array_free(y2);
    return y;
}

fn moeRowsFusedCase(s: mlx.mlx_stream, rnd: std.Random, E: c_int, hid: c_int, inter: c_int, topk: c_int, nrows: c_int, bits: u32, gs: u32, dbits: u32, dgs: u32) !void {
    const gate = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(gate.w);
    defer _ = mlx.mlx_array_free(gate.sc);
    defer _ = mlx.mlx_array_free(gate.bi);
    const up = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(up.w);
    defer _ = mlx.mlx_array_free(up.sc);
    defer _ = mlx.mlx_array_free(up.bi);
    const down = try moeRowsAffineBank(s, rnd, E, hid, inter, dbits, dgs);
    defer _ = mlx.mlx_array_free(down.w);
    defer _ = mlx.mlx_array_free(down.sc);
    defer _ = mlx.mlx_array_free(down.bi);
    const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, hid }, s);
    defer _ = mlx.mlx_array_free(x);
    const idata = try testing.allocator.alloc(u32, @intCast(nrows * topk));
    defer testing.allocator.free(idata);
    const perm = [_]u32{ 0, 3, 0, 7, 2, 3, 5, 1 };
    for (0..@intCast(nrows)) |r| {
        const base = perm[r % perm.len] * 17;
        for (0..10) |k| idata[r * 10 + k] = (base + @as(u32, @intCast(k)) * 13) % @as(u32, @intCast(E));
    }
    const inds = mlx.mlx_array_new_data(idata.ptr, &[_]c_int{ nrows, topk }, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const scores = try attn256RandBf16(rnd, &[_]c_int{ nrows, topk }, s);
    defer _ = mlx.mlx_array_free(scores);
    const fused_gu = (try gatherQmvGateUpRows(s, x, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused_gu);
    const fused_dn = (try gatherQmvDownReduceRows(s, fused_gu, down.w, down.sc, down.bi, inds, scores, dbits, dgs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused_dn);
    var r: c_int = 0;
    while (r < nrows) : (r += 1) {
        const xr = try moeRowsSliceRow(s, x, r);
        defer _ = mlx.mlx_array_free(xr);
        const ir = try moeRowsSliceRow(s, inds, r);
        defer _ = mlx.mlx_array_free(ir);
        const sr = try moeRowsSliceRow(s, scores, r);
        defer _ = mlx.mlx_array_free(sr);
        const solo_gu = (try gatherQmvGateUp(s, xr, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, ir, bits, gs, .affine)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(solo_gu);
        const fused_r = try moeRowsSliceRow(s, fused_gu, r);
        defer _ = mlx.mlx_array_free(fused_r);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused_r, solo_gu, s));
        const solo_dn = (try gatherQmvDownReduce(s, solo_gu, down.w, down.sc, down.bi, ir, sr, dbits, dgs, .affine)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(solo_dn);
        const fused_dr = try moeRowsSliceRow(s, fused_dn, r);
        defer _ = mlx.mlx_array_free(fused_dr);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused_dr, solo_dn, s));
    }
}

/// Max |a - ref| as f32, for the gatherQmv no-worse-than-reference parity tests.
fn gqmvMaxAbsErr(a: mlx.mlx_array, ref: mlx.mlx_array, str: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, str));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a32, ref, str));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, str));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, str));
    try mlx.check(mlx.mlx_array_eval(mx));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, mx));
    return out;
}

fn moeDownReduceParityCase(s: mlx.mlx_stream, rnd: std.Random, t: anytype, E: usize, TOPK: c_int, KIN: c_int, N: c_int) !void {
    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 4, .gs = 32 },
        .{ .bits = 3, .gs = 64 },
    }) |cfg| {
        const bits = cfg.bits;
        const gs = cfg.gs;
        const words_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)) * bits, 32));
        const groups_per_row: usize = @intCast(@divExact(@as(u32, @intCast(KIN)), gs));

        // Packed weight bank: any bit pattern is a valid quantized row.
        const wn = E * @as(usize, @intCast(N)) * words_per_row;
        const wdata = try t.allocator.alloc(u32, wn);
        defer t.allocator.free(wdata);
        for (wdata) |*v| v.* = rnd.int(u32);
        const w_shape = [_]c_int{ @intCast(E), N, @intCast(words_per_row) };
        const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
        defer _ = mlx.mlx_array_free(w);

        const sn = E * @as(usize, @intCast(N)) * groups_per_row;
        const s_shape = [_]c_int{ @intCast(E), N, @intCast(groups_per_row) };
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        const sf = try t.allocator.alloc(f32, sn);
        defer t.allocator.free(sf);
        for (sf) |*v| v.* = 0.01 + 0.05 * rnd.float(f32);
        const bf = try t.allocator.alloc(f32, sn);
        defer t.allocator.free(bf);
        for (bf) |*v| v.* = rnd.float(f32) - 0.5;
        const sc_src = mlx.mlx_array_new_data(sf.ptr, &s_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(sc_src);
        const bi_src = mlx.mlx_array_new_data(bf.ptr, &s_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(bi_src);
        try mlx.check(mlx.mlx_astype(&sc, sc_src, .bfloat16, s));
        try mlx.check(mlx.mlx_astype(&bi, bi_src, .bfloat16, s));

        const x_shape = [_]c_int{ TOPK, KIN };
        const x = try attn256RandBf16(rnd, &x_shape, s);
        defer _ = mlx.mlx_array_free(x);
        const sc_shape = [_]c_int{TOPK};
        const scores = try attn256RandBf16(rnd, &sc_shape, s);
        defer _ = mlx.mlx_array_free(scores);
        var idata: [8]u32 = undefined;
        for (&idata) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        const mode: QuantMode = .affine;

        // Composed chain: gatherQmv down → [TOPK,N] → ·scores → sum over K.
        const down_2d = (try gatherQmv(s, x, w, sc, bi, inds, bits, gs, mode, true)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(down_2d);
        var down_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_4d);
        const d4 = [_]c_int{ 1, 1, TOPK, N };
        try mlx.check(mlx.mlx_reshape(&down_4d, down_2d, &d4, 4, s));
        var scores_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_exp);
        const se = [_]c_int{ 1, 1, TOPK, 1 };
        try mlx.check(mlx.mlx_reshape(&scores_exp, scores, &se, 4, s));
        var weighted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(weighted);
        try mlx.check(mlx.mlx_multiply(&weighted, down_4d, scores_exp, s));
        var composed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(composed);
        try mlx.check(mlx.mlx_sum_axis(&composed, weighted, -2, false, s));

        // Fused: one dispatch.
        const fused_flat = (try gatherQmvDownReduce(s, x, w, sc, bi, inds, scores, bits, gs, mode)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused_flat);
        var fused = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fused);
        const f3 = [_]c_int{ 1, 1, N };
        try mlx.check(mlx.mlx_reshape(&fused, fused_flat, &f3, 3, s));

        // f32 truth: dequantized bank, the TOPK experts, f32 matmul, f32 weighted sum.
        var deq32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deq32);
        try mlx.check(mlx.mlx_dequantize(&deq32, w, sc, bi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var sel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sel);
        try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
        var x32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x32);
        try mlx.check(mlx.mlx_astype(&x32, x, .float32, s));
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        const x3s = [_]c_int{ TOPK, KIN, 1 };
        try mlx.check(mlx.mlx_reshape(&x3, x32, &x3s, 3, s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_matmul(&prod, sel, x3, s));
        var sc32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc32);
        try mlx.check(mlx.mlx_astype(&sc32, scores, .float32, s));
        var sc3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc3);
        const sc3s = [_]c_int{ TOPK, 1, 1 };
        try mlx.check(mlx.mlx_reshape(&sc3, sc32, &sc3s, 3, s));
        var wprod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wprod);
        try mlx.check(mlx.mlx_multiply(&wprod, prod, sc3, s));
        var truth_n = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth_n);
        try mlx.check(mlx.mlx_sum_axis(&truth_n, wprod, 0, false, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_reshape(&truth, truth_n, &f3, 3, s));

        const cnt: usize = @intCast(N);
        const hf = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hf);
        const hc = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(hc);
        const ht = try t.allocator.alloc(f32, cnt);
        defer t.allocator.free(ht);
        try testReadF32(fused, hf, s);
        try testReadF32(composed, hc, s);
        try testReadF32(truth, ht, s);
        var se_f: f64 = 0;
        var se_c: f64 = 0;
        for (hf, hc, ht) |a, b, c| {
            try t.expect(std.math.isFinite(a));
            se_f += (a - c) * (a - c);
            se_c += (b - c) * (b - c);
        }
        const rms_f = @sqrt(se_f / @as(f64, @floatFromInt(cnt)));
        const rms_c = @sqrt(se_c / @as(f64, @floatFromInt(cnt)));
        t.expect(rms_f <= rms_c * 1.15 + 1e-6) catch |e| {
            std.debug.print("\n[down-reduce] bits={d} gs={d} K={d} N={d} fused_rms={e:.4} composed_rms={e:.4}\n", .{ bits, gs, KIN, N, rms_f, rms_c });
            return e;
        };
    }
}

fn sqIds(host: []const u32, len: c_int) mlx.mlx_array {
    return mlx.mlx_array_new_data(host.ptr, &[_]c_int{len}, 1, .uint32);
}

fn sqRandomBf16(alloc: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const host = try alloc.alloc(u16, n);
    for (host) |*v| {
        const value: f32 = -1.0 + rnd.float(f32) * 2.0;
        v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    return host;
}

const SqArm = struct {
    gate_w: mlx.mlx_array,
    gate_s: mlx.mlx_array,
    gate_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
};

test "fused gate+up+SwiGLU expert kernel is bit-identical to the split gatherQmv path" {
    try mlx.requireGpu();
    // The split path is `gatherQmv(gate)` + `gatherQmv(up)` + `fusedSwiGLU`.
    // Fusing them may not change a single rounding: each accumulator is still
    // rounded to T before the activation, and the activation is still the
    // table. Covers the live affine widths.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E0FF5);
    const rnd = prng.random();
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    trf.swiglu_fused_override = true;
    defer trf.swiglu_fused_override = null;

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 3, .gs = 128 },
    }) |cfg| {
        const bits = cfg.bits;
        const gs = cfg.gs;
        const mode: QuantMode = .affine;

        // Two independent banks, so a kernel that read the gate bank twice
        // could not pass.
        var banks: [2]struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } = undefined;
        var made: usize = 0;
        defer for (banks[0..made]) |b| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.sc);
            _ = mlx.mlx_array_free(b.bi);
        };
        for (0..2) |bi_idx| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
            defer _ = mlx.mlx_array_free(wb);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            var sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
            var bias = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&bias, triple, 2));
            banks[bi_idx] = .{ .w = wq, .sc = sc, .bi = bias };
            made += 1;
        }

        const xbuf = try allocator.alloc(f32, @intCast(K));
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        const xsh = [_]c_int{K};
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 1, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        var idx: [3]u32 = .{ 5, 0, 3 };
        const ish = [_]c_int{TOPK};
        const inds = mlx.mlx_array_new_data(&idx, &ish, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        // Reference: the two separate gathers, then the fused activation.
        const g2 = (try gatherQmv(s, x, banks[0].w, banks[0].sc, banks[0].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(g2);
        const up2 = (try gatherQmv(s, x, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(up2);
        const ref = (try fusedSwiGLU(s, g2, up2)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(ref);

        const got = (try gatherQmvGateUp(s, x, banks[0].w, banks[0].sc, banks[0].bi, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode)) orelse
            return error.FusedGateUpDeclined;
        defer _ = mlx.mlx_array_free(got);

        const cnt: usize = @intCast(TOPK * N);
        const ref_host = try allocator.alloc(f32, cnt);
        defer allocator.free(ref_host);
        const got_host = try allocator.alloc(f32, cnt);
        defer allocator.free(got_host);
        try testReadF32(ref, ref_host, s);
        try testReadF32(got, got_host, s);
        try testing.expectEqualSlices(f32, ref_host, got_host);
    }
}

test "gatherQmvGateUpRows accepts N=8 K=10" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    const nrows: c_int = 8;
    const topk: c_int = 10;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const E: c_int = 8;
    const bits: u32 = 4;
    const gs: u32 = 64;
    const xsh = [_]c_int{ nrows, hid };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &xsh, 2, .bfloat16, s));
    const ish = [_]c_int{ nrows, topk };
    var idata: [80]u32 = @splat(0);
    const inds = mlx.mlx_array_new_data(&idata, &ish, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const wsh = [_]c_int{ E, inter, @intCast(@divExact(hid * @as(c_int, @intCast(bits)), 32)) };
    var gw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gw);
    try mlx.check(mlx.mlx_zeros(&gw, &wsh, 3, .uint32, s));
    var uw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uw);
    try mlx.check(mlx.mlx_zeros(&uw, &wsh, 3, .uint32, s));
    const ssh = [_]c_int{ E, inter, @divExact(hid, @as(c_int, @intCast(gs))) };
    var gscl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gscl);
    try mlx.check(mlx.mlx_zeros(&gscl, &ssh, 3, .bfloat16, s));
    var uscl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uscl);
    try mlx.check(mlx.mlx_zeros(&uscl, &ssh, 3, .bfloat16, s));
    var gb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gb);
    try mlx.check(mlx.mlx_zeros(&gb, &ssh, 3, .bfloat16, s));
    var ub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ub);
    try mlx.check(mlx.mlx_zeros(&ub, &ssh, 3, .bfloat16, s));
    const got = (try gatherQmvGateUpRows(s, x, gw, gscl, gb, uw, uscl, ub, inds, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got);
}

test "moe rows fused: N=1 is byte-identical to the single-row kernel" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x51F0);
    const rnd = prng.random();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    downred_override = true;
    defer downred_override = null;
    const E: c_int = 32;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    const bits: u32 = 4;
    const gs: u32 = 64;
    const gate = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(gate.w);
    defer _ = mlx.mlx_array_free(gate.sc);
    defer _ = mlx.mlx_array_free(gate.bi);
    const up = try moeRowsAffineBank(s, rnd, E, inter, hid, bits, gs);
    defer _ = mlx.mlx_array_free(up.w);
    defer _ = mlx.mlx_array_free(up.sc);
    defer _ = mlx.mlx_array_free(up.bi);
    const down = try moeRowsAffineBank(s, rnd, E, hid, inter, 8, 32);
    defer _ = mlx.mlx_array_free(down.w);
    defer _ = mlx.mlx_array_free(down.sc);
    defer _ = mlx.mlx_array_free(down.bi);
    const x1 = try attn256RandBf16(rnd, &[_]c_int{hid}, s);
    defer _ = mlx.mlx_array_free(x1);
    var xrow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrow);
    const xrsh = [_]c_int{ 1, hid };
    try mlx.check(mlx.mlx_reshape(&xrow, x1, &xrsh, 2, s));
    var idx: [10]u32 = .{ 0, 3, 7, 2, 11, 5, 9, 1, 4, 8 };
    const inds1 = mlx.mlx_array_new_data(&idx, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds1);
    const inds2 = mlx.mlx_array_new_data(&idx, &[_]c_int{ 1, topk }, 2, .uint32);
    defer _ = mlx.mlx_array_free(inds2);
    const solo_gu = (try gatherQmvGateUp(s, x1, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds1, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(solo_gu);
    const rows_gu = (try gatherQmvGateUpRows(s, xrow, gate.w, gate.sc, gate.bi, up.w, up.sc, up.bi, inds2, bits, gs, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rows_gu);
    var rows_gu_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows_gu_2d);
    try mlx.check(mlx.mlx_squeeze(&rows_gu_2d, rows_gu, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(solo_gu, rows_gu_2d, s));
    const scores1 = try attn256RandBf16(rnd, &[_]c_int{topk}, s);
    defer _ = mlx.mlx_array_free(scores1);
    var scores2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores2);
    const scsh = [_]c_int{ 1, topk };
    try mlx.check(mlx.mlx_reshape(&scores2, scores1, &scsh, 2, s));
    const solo_dn = (try gatherQmvDownReduce(s, solo_gu, down.w, down.sc, down.bi, inds1, scores1, 8, 32, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(solo_dn);
    const rows_dn = (try gatherQmvDownReduceRows(s, rows_gu, down.w, down.sc, down.bi, inds2, scores2, 8, 32, .affine)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(rows_dn);
    var rows_dn_1d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows_dn_1d);
    try mlx.check(mlx.mlx_squeeze(&rows_dn_1d, rows_dn, s));
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(solo_dn, rows_dn_1d, s));
}

test "moe rows fused: N=8 K=10 gate/up and down/reduce match N solo calls" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xC0DE);
    const rnd = prng.random();
    moe_rows_fused_override = true;
    defer moe_rows_fused_override = null;
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    downred_override = true;
    defer downred_override = null;
    const E: c_int = 512;
    const hid: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    const nrows: c_int = 8;
    for ([_]struct { bits: u32, gs: u32, dbits: u32, dgs: u32 }{
        .{ .bits = 4, .gs = 64, .dbits = 8, .dgs = 32 },
        .{ .bits = 3, .gs = 128, .dbits = 3, .dgs = 64 },
    }) |cfg| try moeRowsFusedCase(s, rnd, E, hid, inter, topk, nrows, cfg.bits, cfg.gs, cfg.dbits, cfg.dgs);
}

test "moe verify rows retain serial expert arithmetic through grouped widths" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x165e71a1);
    const saved_rows = moe_rows_fused_override;
    const saved_gu = gqmv_gateup_override;
    const saved_down = downred_override;
    moe_rows_fused_override = true;
    gqmv_gateup_override = true;
    downred_override = true;
    defer moe_rows_fused_override = saved_rows;
    defer gqmv_gateup_override = saved_gu;
    defer downred_override = saved_down;
    for ([_]c_int{ 2, 3, 8, 9, 16, 18, 32 }) |rows| {
        try moeRowsFusedCase(s, prng.random(), 16, 2560, 640, 10, rows, 4, 64, 4, 64);
    }
}

test "sushi_qmv_rows is mlx_equal per row to stock M=1 qmv at every head shape" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x51500A01);
    const rnd = prng.random();
    const shapes = [_][2]c_int{
        .{ 2560, 2560 },
        .{ 2560, 12288 },
        .{ 2560, 512 },
        .{ 12288, 2560 },
        .{ 2560, 640 },
        .{ 2560, 248320 },
    };
    const bitss = [_]u32{ 3, 4, 8 };
    const ns = [_]c_int{ 2, 4, 8, 16 };
    const gs: u32 = 64;
    for (bitss) |bits| {
        for (shapes) |kn| {
            const K = kn[0];
            const Nout = kn[1];
            const w_bf = try testRandWeightBf16(rnd, &[_]c_int{ Nout, K }, s);
            defer _ = mlx.mlx_array_free(w_bf);
            var qvec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qvec);
            try mlx.check(mlx.mlx_quantize(&qvec, w_bf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, s));
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            var sc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc);
            var bi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi);
            try mlx.check(mlx.mlx_vector_array_get(&w, qvec, 0));
            try mlx.check(mlx.mlx_vector_array_get(&sc, qvec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&bi, qvec, 2));
            for (ns) |nrows| {
                const x = try attn256RandBf16(rnd, &[_]c_int{ nrows, K }, s);
                defer _ = mlx.mlx_array_free(x);
                const got = (try msvQmvRows(s, x, w, sc, bi, bits, gs)) orelse {
                    std.debug.print("[sushi_qmv_rows] declined bits={d} K={d} N={d} rows={d}\n", .{ bits, K, Nout, nrows });
                    return error.QmvRowsDeclined;
                };
                defer _ = mlx.mlx_array_free(got);
                var r: c_int = 0;
                while (r < nrows) : (r += 1) {
                    const xr = try moeRowsSliceRow(s, x, r);
                    defer _ = mlx.mlx_array_free(xr);
                    var xr2 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr2);
                    try mlx.check(mlx.mlx_reshape(&xr2, xr, &[_]c_int{ 1, K }, 2, s));
                    var want = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(want);
                    try mlx.check(mlx.mlx_quantized_matmul(
                        &want,
                        xr2,
                        w,
                        sc,
                        bi,
                        true,
                        mlx.mlx_optional_int.some(@intCast(gs)),
                        mlx.mlx_optional_int.some(@intCast(bits)),
                        "affine",
                        s,
                    ));
                    const gr = try moeRowsSliceRow(s, got, r);
                    defer _ = mlx.mlx_array_free(gr);
                    const diff = try attn256MaxDiff(gr, want, s);
                    if (diff != 0) {
                        std.debug.print("[sushi_qmv_rows] NE bits={d} K={d} N={d} rows={d} row={d} maxdiff={d}\n", .{ bits, K, Nout, nrows, r, diff });
                    }
                    try testing.expectEqual(@as(f32, 0.0), diff);
                }
            }
        }
    }
}

test "gatherQmv is no worse than stock gather_qmm vs fp32 dequant ground truth" {
    try mlx.requireGpu();
    // Per the kernel-testing rule: assert NO-WORSE-THAN-REFERENCE against fp32
    // dequant ground truth, never kernel-vs-kernel agreement (both round to
    // bf16 in different orders, so exact agreement is not a real invariant).
    // Covers BOTH input layouts: the shared-x gate/up projection and the
    // per-expert-x down projection.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rnd = prng.random();

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 2, .gs = 64 },
        .{ .bits = 3, .gs = 64 },
        .{ .bits = 3, .gs = 128 },
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 8, .gs = 32 },
    }) |cfg| {
        // fp32 is the LIVE case: expert_x arrives as float32 while the
        // bank's scales/biases are bf16. A bf16-only test missed it entirely.
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
            for ([_]bool{ false, true }) |x_per_expert| {
                // ── quantized bank [E,N,K] ──
                const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
                defer _ = mlx.mlx_array_free(wb);
                var triple = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(triple);
                try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, s));
                var wq = mlx.mlx_array_new();
                var wsc = mlx.mlx_array_new();
                var wbi = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wq);
                defer _ = mlx.mlx_array_free(wsc);
                defer _ = mlx.mlx_array_free(wbi);
                try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
                try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
                try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

                // ── x: [K] shared, or [TOPK,K] per expert ──
                const xrows: c_int = if (x_per_expert) TOPK else 1;
                const xcnt: usize = @intCast(xrows * K);
                const xbuf = try allocator.alloc(f32, xcnt);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xsh = [_]c_int{ xrows, K };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, xdt, s));

                const idxvals = [_]u32{ 5, 0, 3 };
                const idxsh = [_]c_int{TOPK};
                const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 1, .uint32);
                defer _ = mlx.mlx_array_free(inds);

                // ── fp32 ground truth: dequantize the bank, take the 3 experts,
                // and do the matmul entirely in float32.
                var deq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq);
                try mlx.check(mlx.mlx_dequantize(&deq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, .{}, s));
                var deq32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq32);
                try mlx.check(mlx.mlx_astype(&deq32, deq, .float32, s));
                var sel = mlx.mlx_array_new(); // [TOPK,N,K]
                defer _ = mlx.mlx_array_free(sel);
                try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
                var selT = mlx.mlx_array_new(); // [TOPK,K,N]
                defer _ = mlx.mlx_array_free(selT);
                const tax = [_]c_int{ 0, 2, 1 };
                try mlx.check(mlx.mlx_transpose_axes(&selT, sel, &tax, 3, s));
                // x as [TOPK,1,K] (broadcast the shared row when needed)
                var x32r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32r);
                const x3 = [_]c_int{ xrows, 1, K };
                try mlx.check(mlx.mlx_reshape(&x32r, x32, &x3, 3, s));
                var x32b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32b);
                const xb3 = [_]c_int{ TOPK, 1, K };
                try mlx.check(mlx.mlx_broadcast_to(&x32b, x32r, &xb3, 3, s));
                var truth = mlx.mlx_array_new(); // [TOPK,1,N]
                defer _ = mlx.mlx_array_free(truth);
                try mlx.check(mlx.mlx_matmul(&truth, x32b, selT, s));
                const tflat = [_]c_int{ TOPK, N };
                var truth2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(truth2);
                try mlx.check(mlx.mlx_reshape(&truth2, truth, &tflat, 2, s));

                // ── stock gather_qmm, same layouts ──
                var xg = mlx.mlx_array_new(); // [TOPK,1,K]
                defer _ = mlx.mlx_array_free(xg);
                {
                    var xr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr);
                    try mlx.check(mlx.mlx_reshape(&xr, x, &x3, 3, s));
                    try mlx.check(mlx.mlx_broadcast_to(&xg, xr, &xb3, 3, s));
                }
                var stock3 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock3);
                const lhs3 = [_]u32{ 0, 1, 2 };
                const lhs_used = if (x_per_expert) blk: {
                    break :blk mlx.mlx_array_new_data(&lhs3, &idxsh, 1, .uint32);
                } else blk: {
                    const zeros = [_]u32{ 0, 0, 0 };
                    break :blk mlx.mlx_array_new_data(&zeros, &idxsh, 1, .uint32);
                };
                defer _ = mlx.mlx_array_free(lhs_used);
                try mlx.check(mlx.mlx_gather_qmm(&stock3, xg, wq, wsc, wbi, lhs_used, inds, true, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", false, s));
                var stock2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock2);
                try mlx.check(mlx.mlx_reshape(&stock2, stock3, &tflat, 2, s));

                // ── ours ──
                const maybe = try gatherQmv(s, x, wq, wsc, wbi, inds, cfg.bits, cfg.gs, .affine, x_per_expert);
                try testing.expect(maybe != null);
                const ours = maybe.?;
                defer _ = mlx.mlx_array_free(ours);

                const err_ours = try gqmvMaxAbsErr(ours, truth2, s);
                const err_stock = try gqmvMaxAbsErr(stock2, truth2, s);
                // No worse than stock against the same fp32 ground truth (small
                // slack for bf16 accumulation-order differences).
                testing.expect(err_ours <= err_stock * 1.05 + 1e-3) catch |e| {
                    std.debug.print("\n[gather-qmv] bits={d} gs={d} xdt={any} per_expert_x={} ours_err={d:.6} stock_err={d:.6}\n", .{ cfg.bits, cfg.gs, xdt, x_per_expert, err_ours, err_stock });
                    return e;
                };
            }
        }
    }
}

test "moe down+reduce fused: no worse than gatherQmv + multiply + sum vs fp32 truth" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09E);
    const rnd = prng.random();
    const t = std.testing;
    downred_override = true;
    defer downred_override = null;

    // Two geometries: a small one, and a real decode shape. The fused
    // kernel reduces each row over 8 lanes, the composed chain over 32, so the
    // bar is error against the f32 truth, never bit identity.
    const geoms = [_][4]c_int{
        .{ 16, 8, 64, 32 }, // E, TOPK, KIN, N
        .{ 64, 8, 512, 2048 },
    };
    for (geoms) |geom| {
        const E: usize = @intCast(geom[0]);
        const TOPK: c_int = geom[1];
        const KIN: c_int = geom[2];
        const N: c_int = geom[3];
        try moeDownReduceParityCase(s, rnd, t, E, TOPK, KIN, N);
    }
}

test "moe down+reduce fused: declines outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xD09F);
    const rnd = prng.random();
    downred_override = true;
    defer downred_override = null;

    const TOPK: c_int = 8;
    const KIN: c_int = 64;
    const x_shape = [_]c_int{ TOPK, KIN };
    const x = try attn256RandBf16(rnd, &x_shape, s);
    defer _ = mlx.mlx_array_free(x);
    const sc_shape = [_]c_int{TOPK};
    const scores = try attn256RandBf16(rnd, &sc_shape, s);
    defer _ = mlx.mlx_array_free(scores);
    var idata: [8]u32 = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const inds = mlx.mlx_array_new_data(&idata, &sc_shape, 1, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const no_arr = mlx.mlx_array{ .ctx = null };

    // mxfp8 mode: wrong dequant arithmetic — must decline, not compute.
    try std.testing.expect((try gatherQmvDownReduce(s, x, no_arr, no_arr, no_arr, inds, scores, 8, 32, .mxfp8)) == null);
    // f32 scores against bf16 x: the composed multiply would promote — decline.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, scores, .float32, s));
    const wn: usize = 8 * 32 * 8;
    const wdata = try std.testing.allocator.alloc(u32, wn);
    defer std.testing.allocator.free(wdata);
    for (wdata) |*v| v.* = rnd.int(u32);
    const w_shape = [_]c_int{ 8, 32, 8 };
    const w = mlx.mlx_array_new_data(wdata.ptr, &w_shape, 3, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const s_shape = [_]c_int{ 8, 32, 2 };
    const scq = try attn256RandBf16(rnd, &s_shape, s);
    defer _ = mlx.mlx_array_free(scq);
    try std.testing.expect((try gatherQmvDownReduce(s, x, w, scq, scq, inds, scores_f32, 4, 32, .affine)) == null);
}

test "row kernel declines a CPU stream" {
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const got = try msvQmvRows(cpu, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 4, 64);
    defer if (got) |y| {
        _ = mlx.mlx_array_free(y);
    };
    try testing.expect(got == null);
}

test "row kernel declines scales and biases with another dtype" {
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 4, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const x = try attn256RandBf16(prng.random(), &.{ 2, 2560 }, s);
    defer _ = mlx.mlx_array_free(x);
    var sc32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc32);
    var bi32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi32);
    try mlx.check(mlx.mlx_astype(&sc32, fx.xfm.lm_head_s, .float32, s));
    try mlx.check(mlx.mlx_astype(&bi32, fx.xfm.lm_head_b, .float32, s));
    for ([_][2]mlx.mlx_array{ .{ sc32, fx.xfm.lm_head_b }, .{ fx.xfm.lm_head_s, bi32 } }) |pair| {
        const got = try msvQmvRows(s, x, fx.xfm.lm_head_w, pair[0], pair[1], 4, 64);
        defer if (got) |y| {
            _ = mlx.mlx_array_free(y);
        };
        try testing.expect(got == null);
    }
}

test "row kernel cache reuses alternating full-shape configurations" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var fx = try @import("mtp.zig").RerankFixture.init(s, 512, 2560, 8, 64, 11);
    defer fx.deinit();
    var prng = std.Random.DefaultPrng.init(12);
    const before = sushi_qmv_rows_cfgs.builds;
    for ([_]c_int{ 3, 5, 3 }) |rows| {
        const x = try attn256RandBf16(prng.random(), &.{ rows, 2560 }, s);
        defer _ = mlx.mlx_array_free(x);
        const y = (try msvQmvRows(s, x, fx.xfm.lm_head_w, fx.xfm.lm_head_s, fx.xfm.lm_head_b, 8, 64)) orelse return error.HeadRowsDeclined;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try testing.expectEqual(@as(usize, 2), sushi_qmv_rows_cfgs.builds - before);
}

test "a streamed quantized slab matches the resident bank through the same kernels" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const experts: u16 = 6;
    const hidden: u32 = 128;
    const inter: u32 = 64;
    const gs: u32 = 64;

    for ([_]u32{ 4, 8 }) |bits| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        var pack = try expert_stream_mod.writeTinyQuantCheckpoint(alloc, tmp.dir, 0x9e37 + bits, experts, hidden, inter, gs, .{ bits, bits, bits });
        defer pack.deinit();
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try tmp.dir.realPath(std.testing.io, &path_buf);
        const path = path_buf[0..path_len];
        const geom = expert_stream_mod.Geometry{ .layers = 1, .experts = experts, .hidden = hidden, .intermediate = inter };
        const per_expert = try expert_stream_mod.expertBytesFor(alloc, path, geom, .quantized_split);

        var bank: [expert_stream_mod.quant.component_count]mlx.mlx_array = undefined;
        for (&bank, 0..) |*value, ci| value.* = pack.residentBank(experts, ci);
        defer for (bank) |value| {
            _ = mlx.mlx_array_free(value);
        };

        var engine = try expert_stream_mod.Engine.initWithOptions(alloc, path, geom, 3 * per_expert, s, .{
            .bounce_size = 1 << 20,
            .layout = .quantized_split,
        });
        defer engine.deinit();
        var prepared = try engine.prepareHost(0, &.{ 5, 1, 4 });
        defer prepared.deinit();
        try std.testing.expect(prepared.quantized);
        var slab: [expert_stream_mod.quant.component_count]mlx.mlx_array = undefined;
        for (&slab, 0..) |*value, ci| value.* = prepared.quantOperand(@fromBackingInt(@intCast(ci)));

        const arms = [_]SqArm{
            .{ .gate_w = bank[0], .gate_s = bank[1], .gate_b = bank[2], .up_w = bank[3], .up_s = bank[4], .up_b = bank[5], .down_w = bank[6], .down_s = bank[7], .down_b = bank[8] },
            .{ .gate_w = slab[0], .gate_s = slab[1], .gate_b = slab[2], .up_w = slab[3], .up_s = slab[4], .up_b = slab[5], .down_w = slab[6], .down_s = slab[7], .down_b = slab[8] },
        };
        const bank_decode = [_]u32{ 5, 4 };
        const slab_decode = [_]u32{ prepared.remapped[0], prepared.remapped[2] };
        const topk: c_int = 2;

        var prng = std.Random.DefaultPrng.init(0xc0ffee + bits);
        const rnd = prng.random();

        {
            const x_host = try sqRandomBf16(alloc, rnd, hidden);
            defer alloc.free(x_host);
            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{@intCast(hidden)}, 1, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const score_host = [_]u16{ @truncate(@as(u32, @bitCast(@as(f32, 0.7))) >> 16), @truncate(@as(u32, @bitCast(@as(f32, 0.3))) >> 16) };
            const scores = mlx.mlx_array_new_data(&score_host, &[_]c_int{topk}, 1, .bfloat16);
            defer _ = mlx.mlx_array_free(scores);
            var hashes: [2]u64 = undefined;
            for (arms, [_][]const u32{ &bank_decode, &slab_decode }, 0..) |arm, ids_host, i| {
                const ids = sqIds(ids_host, topk);
                defer _ = mlx.mlx_array_free(ids);
                const act = (try gatherQmvGateUp(s, x, arm.gate_w, arm.gate_s, arm.gate_b, arm.up_w, arm.up_s, arm.up_b, ids, bits, gs, .affine)) orelse return error.SkipZigTest;
                defer _ = mlx.mlx_array_free(act);
                const out = (try gatherQmvDownReduce(s, act, arm.down_w, arm.down_s, arm.down_b, ids, scores, bits, gs, .affine)) orelse return error.SkipZigTest;
                defer _ = mlx.mlx_array_free(out);
                try std.testing.expectEqualSlices(c_int, &[_]c_int{@intCast(hidden)}, mlx.getShape(out));
                hashes[i] = try seHash(out);
            }
            try std.testing.expectEqual(hashes[0], hashes[1]);
        }

        for ([_]c_int{ 4, 64 }) |rows| {
            const n: c_int = rows * topk;
            const order = try alloc.alloc(usize, @intCast(n));
            defer alloc.free(order);
            const bank_ids = try alloc.alloc(u32, @intCast(n));
            defer alloc.free(bank_ids);
            const slab_ids = try alloc.alloc(u32, @intCast(n));
            defer alloc.free(slab_ids);
            const members = [_]u16{ 5, 1, 4 };
            for (order, 0..) |*slot, i| slot.* = (i * members.len) / @as(usize, @intCast(n));
            std.mem.sort(usize, order, prepared.remapped, struct {
                fn less(remap: []const u16, a: usize, b: usize) bool {
                    return remap[a] < remap[b];
                }
            }.less);
            for (bank_ids, slab_ids, order) |*b, *l, which| {
                b.* = members[which];
                l.* = prepared.remapped[which];
            }
            const x_host = try sqRandomBf16(alloc, rnd, @intCast(n * @as(c_int, @intCast(hidden))));
            defer alloc.free(x_host);
            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ n, 1, @intCast(hidden) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const act_host = try sqRandomBf16(alloc, rnd, @intCast(n * @as(c_int, @intCast(inter))));
            defer alloc.free(act_host);
            const act = mlx.mlx_array_new_data(act_host.ptr, &[_]c_int{ n, 1, @intCast(inter) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(act);

            var gate_hashes: [2]u64 = undefined;
            var down_hashes: [2]u64 = undefined;
            for (arms, [_][]const u32{ bank_ids, slab_ids }, 0..) |arm, ids_host, i| {
                // Repacking changes expert IDs; each sorted arm needs its own row order.
                const permutation = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(permutation);
                const inverse = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(inverse);
                for (permutation, 0..) |*slot, j| slot.* = @intCast(j);
                std.mem.sort(u32, permutation, ids_host, struct {
                    fn less(ids_: []const u32, a: u32, b: u32) bool {
                        return ids_[a] < ids_[b];
                    }
                }.less);
                const sorted_ids = try alloc.alloc(u32, @intCast(n));
                defer alloc.free(sorted_ids);
                for (permutation, 0..) |slot, j| {
                    sorted_ids[j] = ids_host[slot];
                    inverse[slot] = @intCast(j);
                }
                const perm = sqIds(permutation, n);
                defer _ = mlx.mlx_array_free(perm);
                const inv = sqIds(inverse, n);
                defer _ = mlx.mlx_array_free(inv);
                const ids = sqIds(sorted_ids, n);
                defer _ = mlx.mlx_array_free(ids);
                var sorted_x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sorted_x);
                var sorted_act = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(sorted_act);
                try mlx.check(mlx.mlx_take_axis(&sorted_x, x, perm, 0, s));
                try mlx.check(mlx.mlx_take_axis(&sorted_act, act, perm, 0, s));
                var gate_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_out);
                var gate_restored = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(gate_restored);
                try gatherExpertMm(&gate_out, sorted_x, arm.gate_w, arm.gate_s, arm.gate_b, .{ .ctx = null }, ids, bits, gs, .affine, true, s);
                try mlx.check(mlx.mlx_take_axis(&gate_restored, gate_out, inv, 0, s));
                gate_hashes[i] = try seHash(gate_restored);
                var down_out = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_out);
                var down_restored = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(down_restored);
                try gatherExpertMm(&down_out, sorted_act, arm.down_w, arm.down_s, arm.down_b, .{ .ctx = null }, ids, bits, gs, .affine, true, s);
                try mlx.check(mlx.mlx_take_axis(&down_restored, down_out, inv, 0, s));
                down_hashes[i] = try seHash(down_restored);
            }
            try std.testing.expectEqual(gate_hashes[0], gate_hashes[1]);
            try std.testing.expectEqual(down_hashes[0], down_hashes[1]);
        }
    }
}
