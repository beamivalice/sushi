//! MiMo-V2 forward: sliding/global attention arms, routed-MoE dispatch, batched decode and verify rows.

const trf = @import("../transformer.zig");
const Transformer = trf.Transformer;
const std = @import("std");
const fp8_block = @import("../fp8_block.zig");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
pub const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("../model.zig");
const io_util_mod = @import("../io_util.zig");
const Weights = model_mod.Weights;
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const test_support = @import("../test_support.zig");
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const testKVWide = test_support.testKVWide;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const testReadF32 = test_support.testReadF32;
const routerRefRow = test_support.routerRefRow;

const DenseKVView = trf.DenseKVView;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const FullAttnWeights = trf.FullAttnWeights;
const KVCache = trf.KVCache;
const KV_UNPACK_HEADER = trf.KV_UNPACK_HEADER;
const KvPrefixView = trf.KvPrefixView;
const ModelConfig = trf.ModelConfig;
const MoeMlpWeights = trf.MoeMlpWeights;
const ProfClock = trf.ProfClock;
const QSA_ATTN_BK = trf.QSA_ATTN_BK;
const QSA_SCORE_KERNEL_HEADER = trf.QSA_SCORE_KERNEL_HEADER;
const QsaAttnCfgKey = trf.QsaAttnCfgKey;
const SLIDING_TRIM_UNBOUNDED = trf.SLIDING_TRIM_UNBOUNDED;
const ShapeKey = trf.ShapeKey;
const bf16Scalar = trf.bf16Scalar;
const capturePrefillHidden = trf.capturePrefillHidden;
const capturePrefillHiddenLast = trf.capturePrefillHiddenLast;
const decodeProfReport = trf.decodeProfReport;
const decodeProfileEnabled = trf.decodeProfileEnabled;
const decodeProfileRows = trf.decodeProfileRows;
const fusedSdpaPrefillKv = trf.fusedSdpaPrefillKv;
const lastDim = trf.lastDim;
const layerCap = trf.layerCap;
const log = trf.log;
const moeDumpBeginForward = trf.moeDumpBeginForward;
const moeDumpForwardDone = trf.moeDumpForwardDone;
const moeDumpTensor = trf.moeDumpTensor;
const moeRouterTopK = trf.moeRouterTopK;
const qmatmulBits = trf.qmatmulBits;
const ropeAngleRows = trf.ropeAngleRows;
const scalarOf = trf.scalarOf;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const slidingDecodeMask = trf.slidingDecodeMask;
const slidingPrefillAttn = trf.slidingPrefillAttn;
const slidingTailSpan = trf.slidingTailSpan;
const slidingViewFor = trf.slidingViewFor;
const splitKAttnDispatch = trf.splitKAttnDispatch;
const verifyQmmNaxAvailable = trf.verifyQmmNaxAvailable;

pub var mimo_batched_logged: bool = false;

/// Build the template sets a MiMo global decode row and a verify's row groups (`mimoGlobalRowsGroup`)
/// dispatch past the packed floor, before a request needs them (each probe is its JIT).
pub fn mimoWarmPackedDecode(config: *const ModelConfig, kv_config: KVQuantConfig) void {
    const kernel = qkvMppKernel() orelse return;
    var li: u32 = 0;
    while (li < config.num_hidden_layers and !config.isGlobalLayer(li)) : (li += 1) {}
    if (li == config.num_hidden_layers) return;
    const h_kv: c_int = @intCast(config.layerKVHeads(li));
    if (h_kv == 0) return;
    var key = QkvMppKey{
        .dk = @intCast(config.layerHeadDim(li)),
        .dv = @intCast(config.layerVHeadDim(li)),
        .bits = kv_config.bits,
        .gs = kv_config.group_size,
        .gqa = @divTrunc(@as(c_int, @intCast(config.num_attention_heads)), h_kv),
        .tq = 1,
        .dtype = .bfloat16,
    };
    _ = qkvMppProbe(kernel, key);
    const rows_kernel = qkvMppRowsKernel() orelse return;
    key.rows = true;
    for ([_]c_int{ 2, 3 }) |tq| {
        key.tq = tq;
        _ = qkvMppProbe(rows_kernel, key);
    }
}

pub const MimoDecodeArm = enum { mpp, split_k, dense };

pub fn mimoDecodeUsesNax() bool {
    return verifyQmmNaxAvailable();
}

/// MiMo global-layer decode on a packed cache: matmul2d where matrix units exist, else split-K.
pub fn mimoGlobalDecodeArm(view: *const DenseKVView, t_q: c_int, nax: bool) MimoDecodeArm {
    if (nax) return if (qkvMppDecodeServes(view, t_q)) .mpp else .dense;
    return if (packedDecodeServesFrom(view, t_q, QKV_SPLITK_DECODE_MIN_TK)) .split_k else .dense;
}

/// The most keys a MiMo global layer's decode still dequantizes whole: below the packed arms'
/// floor. `server.kvDequantScratchBytes` bills it.
pub fn mimoGlobalDecodeRebuildMaxKeys() u64 {
    return @intCast(@max(QKV_MPP_DECODE_MIN_TK, QKV_SPLITK_DECODE_MIN_TK) - 1);
}

/// One MiMo query row attending `total_kv` keys with a decode tick's arithmetic. `view`
/// is what that tick's cache update returned; `decode_mask` is the forward's sliding
/// decode mask (built whenever the forward reaches past the `window`).
pub fn mimoDecodeAttn(
    s: mlx.mlx_stream,
    window: c_int,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    sinks: mlx.mlx_array,
    is_global: bool,
    total_kv: c_int,
    attn_scale: f32,
    decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    // The packed arm follows the cache's length every step.
    if (is_global and sinks.ctx == null) {
        const packed_read = switch (mimoGlobalDecodeArm(view, 1, mimoDecodeUsesNax())) {
            .mpp => try qkvAttnMppKernel(s, q, view, attn_scale, ""),
            .split_k => try qkvAttnSplitKKernel(s, q, view, attn_scale, ""),
            .dense => null,
        };
        if (packed_read) |out| return out;
    }
    const windowed = !is_global and total_kv > window;
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, view.k, view.v, attn_scale, if (windowed) "array" else "", if (windowed) decode_mask else none_mask, sinks, false, s));
    return out;
}

/// MLX v0.32.2 `sdpa_vector` (float mask, sinks) with one threadgroup per (head, verify row);
/// row r reads the `W` keys ending at its own key, the window its decode tick saw.
pub const MIMO_SWA_ROWS_SOURCE =
    \\constexpr int BN = 32;
    \\constexpr int BD = 32;
    \\constexpr int qk_per_thread = D / BD;
    \\constexpr int v_per_thread = V / BD;
    \\typedef float U;
    \\const uint simd_gid = simdgroup_index_in_threadgroup;
    \\const uint simd_lid = thread_index_in_simdgroup;
    \\const int head = int(threadgroup_position_in_grid.x);
    \\const int row = int(threadgroup_position_in_grid.y);
    \\const int rows = int(threadgroups_per_grid.y);
    \\const int kv_head = head / (int(threadgroups_per_grid.x) / int(k_shape[1]));
    \\const int first = int(k_shape[2]) - rows + 1 - W + row;
    \\const int inner_k_stride = BN * int(k_strides[2]);
    \\const int inner_v_stride = BN * int(v_strides[2]);
    \\thread U qv[qk_per_thread];
    \\thread U kv[qk_per_thread];
    \\thread U o[v_per_thread];
    \\threadgroup U outputs[BN * BD];
    \\threadgroup U max_scores[BN];
    \\threadgroup U sum_exp_scores[BN];
    \\const device T* queries = q + head * q_strides[1] + row * q_strides[2] + simd_lid * qk_per_thread;
    \\const device T* keys = k + kv_head * k_strides[1] + (first + int(simd_gid)) * k_strides[2] + simd_lid * qk_per_thread;
    \\const device T* values = v + kv_head * v_strides[1] + (first + int(simd_gid)) * v_strides[2] + simd_lid * v_per_thread;
    \\const device T* fmask = mask + simd_gid;
    \\device T* outp = out + (head * rows + row) * V + simd_gid * v_per_thread;
    \\for (int i = 0; i < qk_per_thread; i++) qv[i] = static_cast<U>(scale[0]) * queries[i];
    \\for (int i = 0; i < v_per_thread; i++) o[i] = 0;
    \\U max_score = Limits<U>::finite_min;
    \\U sum_exp_score = 0;
    \\if (simd_gid == 0) {
    \\  max_score = static_cast<U>(sinks[head]);
    \\  sum_exp_score = 1;
    \\}
    \\for (int i = simd_gid; i < W; i += BN) {
    \\  if (fmask[0] >= Limits<T>::finite_min) {
    \\    for (int j = 0; j < qk_per_thread; j++) kv[j] = keys[j];
    \\    U score = 0;
    \\    for (int j = 0; j < qk_per_thread; j++) score += qv[j] * kv[j];
    \\    score = simd_sum(score);
    \\    score += static_cast<U>(fmask[0]);
    \\    U new_max = max(max_score, score);
    \\    U factor = fast::exp(max_score - new_max);
    \\    U exp_score = fast::exp(score - new_max);
    \\    max_score = new_max;
    \\    sum_exp_score = sum_exp_score * factor + exp_score;
    \\    for (int j = 0; j < v_per_thread; j++) o[j] = o[j] * factor + exp_score * values[j];
    \\  }
    \\  keys += inner_k_stride;
    \\  values += inner_v_stride;
    \\  fmask += BN;
    \\}
    \\if (simd_lid == 0) {
    \\  max_scores[simd_gid] = max_score;
    \\  sum_exp_scores[simd_gid] = sum_exp_score;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\max_score = max_scores[simd_lid];
    \\U new_max = simd_max(max_score);
    \\U factor = fast::exp(max_score - new_max);
    \\sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
    \\for (int i = 0; i < v_per_thread; i++) {
    \\  outputs[simd_lid * BD + simd_gid] = o[i];
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    \\  o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\if (simd_lid == 0) {
    \\  for (int i = 0; i < v_per_thread; i++) outp[i] = static_cast<T>(o[i]);
    \\}
;

pub var mimo_swa_rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub var mimo_swa_rows_engaged = false;

pub const MimoSwaRowsKey = struct { heads: c_int, dtype: mlx.mlx_dtype, window: c_int };

pub var mimo_swa_rows_cfgs: [MIMO_VERIFY_ROWS_MAX + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);

pub var mimo_swa_rows_keys: [MIMO_VERIFY_ROWS_MAX + 1]MimoSwaRowsKey = undefined;

/// Every sliding row of a verify in one dispatch, each row that row's decode sdpa bit for bit.
/// Null while the first row still reads fewer than `window` keys (its tick took no mask).
pub fn mimoSlidingRowsAttn(
    s: mlx.mlx_stream,
    window: c_int,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    sinks: mlx.mlx_array,
    offset: c_int,
    rows: c_int,
    attn_scale: f32,
    decode_mask: mlx.mlx_array,
) !?mlx.mlx_array {
    // MLX's one-pass vector kernel serves (qk 192, v 128) only below 1024 keys.
    if (offset + 1 <= window or window >= 1024 or rows < 2 or rows > MIMO_VERIFY_ROWS_MAX) return null;
    if (sinks.ctx == null or decode_mask.ctx == null or view.k.ctx == null or view.v.ctx == null) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(view.k);
    const vs = mlx.getShape(view.v);
    if (qs.len != 4 or ks.len != 4 or vs.len != 4 or qs[0] != 1 or ks[0] != 1 or qs[2] != rows) return null;
    if (qs[3] != 192 or ks[3] != 192 or vs[3] != 128) return null;
    const heads = qs[1];
    const kv_heads = ks[1];
    if (kv_heads <= 0 or vs[1] != kv_heads or @rem(heads, kv_heads) != 0 or @divExact(heads, kv_heads) > 32) return null;
    if (ks[2] != vs[2] or ks[2] < window + rows - 1) return null;
    const dt = mlx.mlx_array_dtype(q);
    if ((dt != .bfloat16 and dt != .float16 and dt != .float32) or mlx.mlx_array_dtype(view.k) != dt or mlx.mlx_array_dtype(view.v) != dt) return null;
    if (mlx.mlx_array_strides(q)[3] != 1 or mlx.mlx_array_strides(view.k)[3] != 1 or mlx.mlx_array_strides(view.v)[3] != 1) return null;
    const ms = mlx.getShape(decode_mask);
    if (mlx.mlx_array_size(decode_mask) != @as(usize, @intCast(window)) or ms[ms.len - 1] != window) return null;
    if (mlx.mlx_array_size(sinks) != @as(usize, @intCast(heads))) return null;

    if (mimo_swa_rows_kernel == null) {
        const input_names = [_][*:0]const u8{ "q", "k", "v", "sinks", "mask", "scale" };
        const output_names = [_][*:0]const u8{"out"};
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kernel = mlx.mlx_fast_metal_kernel_new("sushi_mimo_swa_rows", in_vec, out_vec, MIMO_SWA_ROWS_SOURCE, "", false, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        mimo_swa_rows_kernel = kernel;
    }
    const slot: usize = @intCast(rows);
    const key = MimoSwaRowsKey{ .heads = heads, .dtype = dt, .window = window };
    if (mimo_swa_rows_cfgs[slot] == null or !std.meta.eql(mimo_swa_rows_keys[slot], key)) {
        if (mimo_swa_rows_cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        mimo_swa_rows_cfgs[slot] = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, heads, rows, 128 }, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, heads * 1024, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 1024, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "D", 192));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "V", 128));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "W", window));
        mimo_swa_rows_cfgs[slot] = config;
        mimo_swa_rows_keys[slot] = key;
    }

    // MLX's sdpa casts the mask and the sinks to the output type before its kernel reads them. The
    // kernel reads both unstrided: one `[window]` mask row shared by every head and row (each row here
    // sees a full window) and one sink per head.
    var sinks_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks_c);
    try mlx.check(mlx.mlx_astype(&sinks_c, sinks, dt, s));
    var sinks_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks_t);
    try mlx.check(mlx.mlx_contiguous(&sinks_t, sinks_c, false, s));
    var mask_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_c);
    try mlx.check(mlx.mlx_astype(&mask_c, decode_mask, dt, s));
    var mask_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_t);
    try mlx.check(mlx.mlx_contiguous(&mask_t, mask_c, false, s));
    const one = [_]c_int{1};
    const scale_data = [_]f32{attn_scale};
    const scale = mlx.mlx_array_new_data(&scale_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scale);
    const inputs = [_]mlx.mlx_array{ q, view.k, view.v, sinks_t, mask_t, scale };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, mimo_swa_rows_kernel.?, in_vec, mimo_swa_rows_cfgs[slot].?, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, out_vec, 0));
    if (!mimo_swa_rows_engaged) {
        mimo_swa_rows_engaged = true;
        log.info("[mimo-verify] sliding rows in one dispatch engaged: rows={d} heads={d} kv_heads={d} window={d}\n", .{ rows, heads, kv_heads, window });
    }
    return out;
}

/// Test seam: global verify forwards whose rows went through row-group passes.
pub var mimo_global_rows_mpp_count: u32 = 0;

/// A spec verify's rows, or a global prefill too short for the fused kernel, each attending
/// exactly the keys its own decode tick would have seen through `mimoDecodeAttn`, so an
/// accepted row is the serial row bit for bit. `kv_view` is the forward's cache update: it
/// ends at the last row's key.
pub fn mimoVerifyRowsAttn(
    s: mlx.mlx_stream,
    window: c_int,
    q_rope: mlx.mlx_array,
    kv_view: *const DenseKVView,
    sinks: mlx.mlx_array,
    is_global: bool,
    offset: c_int,
    seq_len: c_int,
    attn_scale: f32,
    decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    if (!is_global and seq_len <= MIMO_VERIFY_ROWS_MAX) {
        if (try mimoSlidingRowsAttn(s, window, q_rope, kv_view, sinks, offset, seq_len, attn_scale, decode_mask)) |out| return out;
    }
    if (is_global and sinks.ctx == null) {
        if (try mimoGlobalRowsMpp(s, q_rope, kv_view, seq_len, attn_scale)) |out| return out;
    }
    return mimoVerifyRowsAttnPerRow(s, window, q_rope, kv_view, sinks, is_global, offset, seq_len, attn_scale, decode_mask);
}

pub var mimo_global_rows_mpp_logged = false;

/// Rows per rows-kernel pass with `left` rows still to place: pairs, since each row's running output
/// costs the kernel registers, and a last three in one pass, which beats a pair and a lone row.
pub fn mimoGlobalRowsGroup(left: c_int) c_int {
    return if (left == 3) 3 else 2;
}

/// A global layer's verify rows through the packed cache in groups (`mimoGlobalRowsGroup`), one
/// rows-kernel pass each, when every row's decode tick takes the matmul2d arm and each group shares
/// its last row's split partition: a shorter row then reads at most one more page, fully masked,
/// which adds exact zeros. Null = the per-row path serves.
pub fn mimoGlobalRowsMpp(s: mlx.mlx_stream, q_rope: mlx.mlx_array, kv_view: *const DenseKVView, seq_len: c_int, attn_scale: f32) !?mlx.mlx_array {
    if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN or !kv_view.has_quant_triple or !mimoDecodeUsesNax()) return null;
    const t_k = mlx.getShape(kv_view.k_triple_q)[2];
    const first = t_k - seq_len + 1;
    if (first < QKV_MPP_DECODE_MIN_TK) return null;
    // A serial tick whose decode template set failed its probe reads through dense SDPA instead.
    if (!qkvMppDecodeProbeOk(q_rope, kv_view)) return null;
    var r: c_int = 0;
    while (r < seq_len) : (r += mimoGlobalRowsGroup(seq_len - r)) {
        const last = first + r + mimoGlobalRowsGroup(seq_len - r) - 1;
        if (!std.meta.eql(qkvMppPartition(first + r), qkvMppPartition(last))) return null;
    }
    var parts: [@divTrunc(FUSED256_MIN_Q_LEN, 2)]mlx.mlx_array = @splat(.{});
    var n: usize = 0;
    defer for (parts[0..n]) |part| {
        _ = mlx.mlx_array_free(part);
    };
    r = 0;
    while (r < seq_len) : (r += mimoGlobalRowsGroup(seq_len - r)) {
        const g = mimoGlobalRowsGroup(seq_len - r);
        var rows = try KvPrefixView.initRange(s, kv_view.*, 0, first + r + g - 1);
        defer rows.deinit();
        const q_g = try sliceAttentionSeq(s, q_rope, r, r + g);
        defer _ = mlx.mlx_array_free(q_g);
        parts[n] = (try qkvAttnMpp(s, q_g, &rows.view, attn_scale, "causal", true)) orelse return null;
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    mimo_global_rows_mpp_count +%= 1;
    if (!mimo_global_rows_mpp_logged) {
        mimo_global_rows_mpp_logged = true;
        log.info("[mimo-verify] global rows in matmul2d row groups engaged: rows={d} keys={d}\n", .{ seq_len, t_k });
    }
    return out;
}

pub fn mimoVerifyRowsAttnPerRow(
    s: mlx.mlx_stream,
    window: c_int,
    q_rope: mlx.mlx_array,
    kv_view: *const DenseKVView,
    sinks: mlx.mlx_array,
    is_global: bool,
    offset: c_int,
    seq_len: c_int,
    attn_scale: f32,
    decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    var parts: [FUSED256_MIN_Q_LEN - 1]mlx.mlx_array = @splat(.{});
    if (seq_len > parts.len) return error.MimoVerifyRowsTooWide;
    const view_len = mlx.getShape(kv_view.k)[2];
    defer for (parts) |part| {
        if (part.ctx != null) _ = mlx.mlx_array_free(part);
    };
    for (0..@intCast(seq_len)) |ri| {
        const r: c_int = @intCast(ri);
        const total_r = offset + r + 1;
        const end = view_len - (seq_len - 1 - r);
        const len_r = if (is_global) end else @min(total_r, window);
        if (len_r > end) return error.MimoVerifyViewTooShort;
        var rows = try KvPrefixView.initRange(s, kv_view.*, end - len_r, end);
        defer rows.deinit();
        const q_r = try sliceAttentionSeq(s, q_rope, r, r + 1);
        defer _ = mlx.mlx_array_free(q_r);
        parts[ri] = try mimoDecodeAttn(s, window, q_r, &rows.view, sinks, is_global, total_r, attn_scale, decode_mask);
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, @intCast(seq_len));
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

/// Widest MiMo spec verify: the FP8 trunk GEMV keeps a decode row's arithmetic only up
/// to `fp8_block.gemv_direct_max_rows` rows.
pub const MIMO_VERIFY_ROWS_MAX: c_int = 8;

pub const MimoAttnArm = enum { verify_rows, decode, prefill_global, prefill_sliding, prefill_rows };

pub var mimo_prefill_rows_logged: bool = false; // one-shot log guard

/// The widest verify the trunk still serves row for row: the FP8 GEMV keeps a
/// single decode row's arithmetic only to `fp8_block.gemv_direct_max_rows` rows,
/// so the budget is read from the kernel rather than restated beside it. Two
/// equal literals drift apart silently, and a verify past the kernel's arm loses
/// byte identity without losing the decode-shaped attention.
pub fn mimoVerifyRowsBudget() c_int {
    return @min(MIMO_VERIFY_ROWS_MAX, fp8_block.gemv_direct_max_rows);
}

/// Which attention arm a MiMo forward runs. A verify is DECODE-shaped at every
/// width: past the row budget the row arithmetic is gone, so the width is
/// refused by name instead of served by the causal prefill arm.
pub fn mimoAttnArm(verify_rows: bool, is_prefill: bool, is_global: bool, seq_len: c_int) error{MimoVerifyRowsTooWide}!MimoAttnArm {
    if (verify_rows) {
        if (seq_len > mimoVerifyRowsBudget()) return error.MimoVerifyRowsTooWide;
        return if (seq_len > 1) .verify_rows else .decode;
    }
    if (!is_prefill) return .decode;
    if (!is_global) return .prefill_sliding;
    // Under the fused kernel's floor the composed arm would rebuild the whole packed cache
    // dense beside a [heads, rows, keys] score sheet; a warm tail reaches it at any context.
    return if (seq_len < FUSED256_MIN_Q_LEN) .prefill_rows else .prefill_global;
}

pub fn computeMimoRouting(self: *const Transformer, router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array) !Transformer.MoeRouting {
    const k: c_int = @intCast(self.config.num_experts_per_tok);
    if (self.config.moe_n_group > 1) {
        return groupLimitedRouting(
            router_logits,
            expert_bias,
            k,
            @intCast(self.config.moe_n_group),
            @intCast(self.config.moe_topk_group),
            self.config.moe_route_norm,
            self.config.router_scaling_factor,
            self.s,
        );
    }
    if (try moeRouterTopK(
        self.s,
        router_logits,
        expert_bias,
        k,
        .sigmoid_bias,
        self.config.moe_route_norm,
        self.config.router_scaling_factor,
        .float32,
        0,
        0,
    )) |fused| return fused;
    return mimoRoutingChain(
        router_logits,
        expert_bias,
        k,
        self.config.moe_route_norm,
        self.config.router_scaling_factor,
        self.s,
    );
}

/// Bit `r` set = a MiMo verify of `r` rows ran.
pub const MimoWarmRows = u32;

// A MiMo verify row keeps its decode tick's arithmetic, so each row count JITs its own FP8,
// EXL3 and qmv pipelines, and the packed global attention its matmul2d set past the key
// floor; left to the first request, each compile stalled a round 450-630 ms on a new binary.
pub fn warmupMimoVerify(self: *Transformer, kv_config: KVQuantConfig) !MimoWarmRows {
    trf.moe_dump_warmup_depth += 1;
    defer trf.moe_dump_warmup_depth -= 1;
    if (!self.config.isMimo()) return 0;
    const sl = try Transformer.SpecWarmSlot.init(self, kv_config);
    defer {
        sl.deinit(self.allocator);
        _ = mlx.mlx_clear_cache();
    }
    if (self.config.swaRingTokens() > 0) sl.cache.setSwaRing(self.config.sliding_window);
    var prompt_ids: [8]i32 = undefined;
    const prompt = Transformer.specWarmIds(&prompt_ids);
    defer _ = mlx.mlx_array_free(prompt);
    Transformer.specWarmEvalFree(&[_]mlx.mlx_array{try self.forwardWith(&sl.ctx, prompt)});
    var warmed: MimoWarmRows = 0;
    sl.ctx.verify_rows = true;
    var rows: usize = 2;
    while (rows <= MIMO_VERIFY_ROWS_MAX) : (rows += 1) {
        var ids: [@intCast(MIMO_VERIFY_ROWS_MAX)]i32 = undefined;
        const block = Transformer.specWarmIds(ids[0..rows]);
        defer _ = mlx.mlx_array_free(block);
        var last = mlx.mlx_array_new();
        var all = mlx.mlx_array_new();
        const logits = try self.forwardWithCaptureAll(&sl.ctx, block, &last, &all);
        Transformer.specWarmEvalFree(&[_]mlx.mlx_array{ logits, last, all });
        warmed |= @as(MimoWarmRows, 1) << @intCast(rows);
    }
    if (kv_config.isQuant() and mimoDecodeUsesNax()) mimoWarmPackedDecode(&self.config, kv_config);
    return warmed;
}

/// MiMo batched decode: N slots' next tokens as ONE `[1, N]` forward on the verify-row
/// arithmetic, whose every op but attention already computes a row as its decode tick does;
/// row i attends and appends on slot i's own cache (`ForwardCtx.batch_rows`). Each row is
/// read out as its slot's tick reads it (the shortlist under `argmax_only`). Returns N
/// logits `[1, 1, V]`; caller owns each and the slice. `hidden_rows`, when given, receives
/// every row's final-normed hidden `[1, 1, H]` (what a solo tick captures for the MTP heads).
pub fn forwardMimoBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    const n = next_tokens.len;
    std.debug.assert(n == ctxs.len and n == rope_offsets.len and n >= 1 and n <= MIMO_VERIFY_ROWS_MAX);
    if (!mimo_batched_logged) {
        mimo_batched_logged = true;
        log.info("[batched] mimo batched decode engaged (slots={d})\n", .{n});
    }
    var token_buf: [MIMO_VERIFY_ROWS_MAX]i32 = undefined;
    for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
    const token_arr = mlx.mlx_array_new_data(&token_buf, &[_]c_int{ 1, @intCast(n) }, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);
    // The scratch position only sizes the sliding decode mask, built when any row reads past the window.
    var scratch: usize = 0;
    for (rope_offsets) |o| scratch = @max(scratch, o);
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch,
        .ssm_entries = null,
        .capture_hidden = null,
        .vision_embeddings = null,
        .verify_rows = true,
        .skip_lm_head = true,
        .batch_rows = ctxs,
        .batch_row_offsets = rope_offsets,
    };
    const normed = try self.forwardMoeWith(&bctx, token_arr);
    defer _ = mlx.mlx_array_free(normed);
    const hidden = mlx.getShape(normed)[2];
    const out = try self.allocator.alloc(mlx.mlx_array, n);
    var filled: usize = 0;
    errdefer {
        for (out[0..filled]) |a| _ = mlx.mlx_array_free(a);
        self.allocator.free(out);
    }
    for (ctxs, 0..) |c, i| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, normed, &[_]c_int{ 0, @intCast(i), 0 }, 3, &[_]c_int{ 1, @as(c_int, @intCast(i)) + 1, hidden }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        out[i] = try self.lmHeadProject(row, c.argmax_only);
        filled += 1;
    }
    if (hidden_rows) |dst| {
        var as_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(as_rows);
        try mlx.check(mlx.mlx_reshape(&as_rows, normed, &[_]c_int{ @intCast(n), 1, hidden }, 3, self.s));
        dst.* = try Transformer.sliceBatchRows(self.allocator, self.s, as_rows, n);
    }
    return out;
}

pub fn supportsBatchedMimoDecode(self: *const Transformer) bool {
    return self.config.supportsBatchedMimoDecode() and self.moe_layers != null and self.expert_stream == null;
}

pub fn forwardMoeWith(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const dumping = moeDumpBeginForward();
    defer if (dumping) moeDumpForwardDone();
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const ml = self.moe_layers.?;
    const offset = ctx.moe_seq_offset.*;
    const cfg = &self.config;
    const is_mimo = std.mem.eql(u8, cfg.model_type, "mimo_v2");

    // PLD spec-decode: thread the per-position SSM capture flag down to the
    // GatedDeltaNet layers (which don't take the ctx). Reset on exit so it
    // never leaks into a non-capturing forward.
    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    // Decode sub-block profiler: start the clock before embedding.
    const prof_on = decodeProfileEnabled();
    var pclk: ProfClock = if (prof_on) ProfClock.init() else undefined;

    var h = try self.embedding(token_ids);
    {
        errdefer _ = mlx.mlx_array_free(h);

        // Splice vision embeddings at image_token_id positions (prefill only)
        h = try self.applyVisionEmbeddingsWith(ctx, h, token_ids);
    }
    if (ctx.capture_layers) |cl| if (cl.input) |slot| {
        _ = mlx.mlx_array_set(slot, h);
    };

    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;
    if (is_mimo) {
        if (self.expert_stream != null and ctx.capture_ssm_seq) return error.StreamingSpecCaptureUnsupported;
        if (self.expert_stream) |engine| engine.beginForward();
    }
    defer if (is_mimo) {
        if (self.expert_stream) |engine| engine.finishForward(@intCast(batch * seq_len));
    };
    const prof = prof_on and seq_len <= decodeProfileRows();
    if (prof) {
        try mlx.check(mlx.mlx_array_eval(h));
        trf.decode_prof.embed_ns += pclk.lap();
    }
    // One-shot dtype trace. A residual stream that is wider than the
    // weights promotes EVERY projection's weight on read, which inflates
    // each stage uniformly rather than showing up as one slow op.
    var dt = mlx.DtypeTrace.begin("moe", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        else => null,
    });

    // Qwen3-VL interleaved M-RoPE: the per-prefill-chunk cos/sin, shared
    // by every full-attn layer this forward.
    try self.beginMropeChunk(ctx, @intCast(offset), @intCast(seq_len), .bfloat16);
    defer Transformer.endMropeChunk(ctx);

    // Precompute the sliding-window masks MiMo's sliding layers read. An arch
    // whose attention arm reads these masks MUST appear in this gate — a
    // missing arm here hands that arm an empty mask handle, and its decode
    // branch then attends over the whole history with no error anywhere.
    var local_prefill_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_prefill_mask);
    var local_decode_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_decode_mask);

    if (is_mimo and cfg.has_sliding_window) {
        const sw: c_int = @intCast(cfg.sliding_window);
        const total_kv: c_int = @as(c_int, @intCast(offset)) + seq_len;
        const sliding = slidingViewFor(cfg, total_kv, seq_len);
        // A MiMo spec verify attends row by row with the decode arithmetic.
        const verify_rows = is_mimo and ctx.verify_rows;
        if (is_prefill and !verify_rows) {
            // Built at the TRIMMED length, which is what the attn fns hand SDPA.
            local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
        }
        if ((!is_prefill or verify_rows) and total_kv > sw) {
            const local_kv_len: c_int = @min(total_kv, sw);
            local_decode_mask = try self.createSlidingWindowDecodeMask(local_kv_len, sw);
        }
    }

    // Eval cadence: drop to per-layer when this chunk's score/dequant
    // transients are large (unfused head_dim > 128 at long ctx, or a
    // quantized cache's dense rebuild) — see prefillEvalCadence.
    const moe_eval_cadence = Transformer.prefillEvalCadence(
        Transformer.PREFILL_EVAL_CADENCE_DEFAULT,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    // MiMo: each residual add runs fused with the norm that reads its sum
    // next, so the next layer's input norm arrives with the add.
    const fuse_norms = is_mimo;
    var carried_normed: mlx.mlx_array = .{ .ctx = null };
    defer if (carried_normed.ctx != null) {
        _ = mlx.mlx_array_free(carried_normed);
    };

    // DIAGNOSTIC (SUSHI_LAYER_CAP=N): run only the first N layers, so a
    // ms-vs-N sweep separates the forward's per-layer slope from its fixed
    // cost. Every layer that runs does its complete real work, so the slope
    // is a marginal cost, not an ablation artifact.
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];

        if (dumping) trf.moe_dump_layer = li;
        const normed = if (carried_normed.ctx != null) blk: {
            const c = carried_normed;
            carried_normed = .{ .ctx = null };
            break :blk c;
        } else try self.rmsNorm(h, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);

        const attn_out = switch (lw.attn) {
            .linear => return error.UnsupportedLinearAttention,
            .full => |fa| try self.mimoAttnWith(ctx, normed, &fa, li, @intCast(offset), batch, seq_len, is_prefill, &local_prefill_mask, local_decode_mask),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(attn_out));
            trf.decode_prof.attn_ns += pclk.lap();
        }

        {
            // Simple residual + post_attn_norm before MLP. The two
            // are strictly serial (the norm waits on the add, the MLP waits
            // on the norm), so one kernel does both — see fusedAddRmsNorm.
            var ff_normed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff_normed);
            // MiMo's router reads the same normed row widened to f32.
            var router_x32: mlx.mlx_array = .{ .ctx = null };
            defer if (router_x32.ctx != null) {
                _ = mlx.mlx_array_free(router_x32);
            };
            const post_fused = if (fuse_norms)
                try fusedAddRmsNormRouted(self.s, h, attn_out, lw.post_attn_norm, self.rms_eps_arr)
            else
                null;
            if (post_fused) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.sum;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = fused.normed;
                router_x32 = fused.normed_f32;
            } else {
                var h_new = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_new, h, attn_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_new;
                _ = mlx.mlx_array_free(ff_normed);
                ff_normed = try self.rmsNorm(h, lw.post_attn_norm);
            }
            if (lw.mlp == .moe) moeDumpTensor(self.s, "hin", trf.moe_dump_layer, h);
            const mlp_out = switch (lw.mlp) {
                .moe => |*mw| if (is_mimo and self.expert_stream != null)
                    try self.moeMLPStreamed(ctx, ff_normed, mw, @intCast(layer_idx))
                else if (router_x32.ctx != null)
                    try self.moeMLP2(router_x32, ff_normed, mw)
                else
                    try self.moeMLP(ff_normed, mw),
                .dense => |*dw| try self.denseMLP(ff_normed, dw),
            };
            defer _ = mlx.mlx_array_free(mlp_out);

            const next_norm = if (layer_idx + 1 < layerCap(cfg.num_hidden_layers)) ml[layer_idx + 1].input_norm else self.final_norm;
            if (if (fuse_norms) try fusedAddRmsNormUngated(self.s, h, mlp_out, next_norm, self.rms_eps_arr) else null) |fused| {
                _ = mlx.mlx_array_free(h);
                h = fused.sum;
                carried_normed = fused.normed;
            } else {
                var h_next = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&h_next, h, mlp_out, self.s));
                _ = mlx.mlx_array_free(h);
                h = h_next;
            }
        }

        if (prof) {
            try mlx.check(mlx.mlx_array_eval(h));
            trf.decode_prof.mlp_ns += pclk.lap();
        }

        // DFlash capture: this h IS `hidden_states[li+1]` — the layer's
        // final output (mirrors forwardStandardWith's capture site).
        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) _ = mlx.mlx_array_set(slot, h);
            }
        }

        if (is_prefill and Transformer.prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % moe_eval_cadence == 0 or layer_idx + 1 == layerCap(cfg.num_hidden_layers))) {
            try Transformer.evalCadencePoint(h, ctx.ssm_entries);
        }
        self.ladderStep(h, layer_idx, seq_len);
        dt.layer(h, layer_idx);
    }

    ctx.moe_seq_offset.* += @intCast(seq_len);
    dt.end(h);

    const final_normed = if (carried_normed.ctx != null) blk: {
        const c = carried_normed;
        carried_normed = .{ .ctx = null };
        break :blk c;
    } else try self.rmsNorm(h, self.final_norm);
    _ = mlx.mlx_array_free(h);
    // Every row, even when this chunk skips the projection: each is a row
    // some position's logits are read from.
    if (self.imatrix) |c| try c.observeLinear(.lm_head, final_normed);

    // Speculative-decoding capture: slice the post-final-norm hidden
    // at the LAST position only. Used by PLD verify-fusion as `h_prev`.
    // Caller frees the captured
    // array.
    if (ctx.capture_hidden) |target| {
        try capturePrefillHiddenLast(self.s, target, final_normed, !ctx.verify_rows);
    }
    if (ctx.capture_hidden_all) |target_all| {
        try capturePrefillHidden(self.s, target_all, final_normed, !ctx.verify_rows);
    }

    if (ctx.skip_lm_head) return final_normed;
    const logits = try self.lmHeadProject(final_normed, ctx.argmax_only);
    _ = mlx.mlx_array_free(final_normed);

    if (prof) {
        try mlx.check(mlx.mlx_array_eval(logits));
        trf.decode_prof.lmhead_ns += pclk.lap();
        trf.decode_prof.calls += 1;
        if (trf.decode_prof.calls % 64 == 0) decodeProfReport();
    }

    return logits;
}

/// MiMo decode or verify rows whose FP8 QKV left V scaled, on a kv8 or kv6 g64 cache: Q/K rope and
/// the K/V quantize run as one dispatch and K/V append already quantized. Null = composed path.
pub fn mimoDecodePrep(
    self: *Transformer,
    ctx: *ForwardCtx,
    proj: [3]mlx.mlx_array,
    is_global: bool,
    rope_dims: c_int,
    rope_base: mlx.mlx_optional_float,
    offset: c_int,
    rows: c_int,
    h_count: c_int,
    kv_h: c_int,
    layer: u32,
    max_kv: u32,
    q_out: *mlx.mlx_array,
) !?DenseKVView {
    const kvc = ctx.cache.config;
    if (kvc.scheme != .affine or (kvc.bits != 8 and kvc.bits != 6) or kvc.group_size != 64) return null;
    const angles = try self.qkAngleRowsFor(if (is_global) 0 else 1, rope_dims, rope_base, .{ .ctx = null }, offset, rows, 1.0);
    var prep = (try mimoDecodeQkvPrep(self.s, proj[0], proj[1], proj[2], angles, h_count, kv_h, rope_dims, rows, kvc.bits)) orelse return null;
    errdefer _ = mlx.mlx_array_free(prep.q);
    defer {
        prep.k.deinit();
        prep.v.deinit();
    }
    const view = try ctx.cache.appendQuantized(layer, prep.k, prep.v, self.s, max_kv);
    _ = mlx.mlx_array_free(q_out.*);
    q_out.* = prep.q;
    return view;
}

/// MiMo V2 attention (XiaomiMiMo/MiMo-V2.6-Flash-RL, Apache-2.0):
/// per-layer GQA geometry, asymmetric K/V widths, and
/// optional sink columns. Prefill takes `sushi_attn_pd` at qk 192 / v 128 (sliding
/// layers with their sinks, sink-free global layers), a packed global decode
/// `sushi_qkv_mpp`; the rest is native SDPA.
pub fn mimoAttnWith(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    local_prefill_mask: *mlx.mlx_array,
    local_decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.layerNumHeads(layer));
    const kv_h: c_int = @intCast(cfg.layerKVHeads(layer));
    const hd: c_int = @intCast(cfg.layerHeadDim(layer));
    const vhd: c_int = @intCast(cfg.layerVHeadDim(layer));
    const flat_shape = [_]c_int{ batch, seq_len, h_count * vhd };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    var proj: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    defer for (proj) |a| {
        _ = mlx.mlx_array_free(a);
    };
    if (mlx.mlx_array_dtype(fa.q_w) == .uint8) {
        // The source FP8 QKV: q_w holds all three, rank-local; V leaves it already scaled.
        var split = try fp8_block.RowSplit.qkv(@intCast(h_count * hd), @intCast(kv_h * hd), @intCast(kv_h * vhd), @intCast(mlx.getShape(fa.q_s)[0]));
        split.v_scale = cfg.attention_value_scale;
        try fp8_block.projectServing(self.s, x, fa.q_w, fa.q_s, split, &proj);
    } else {
        proj[0] = try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
        proj[1] = try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
        proj[2] = try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
    }
    const attn_out = if (ctx.batch_rows) |rows|
        try self.mimoBatchRowsAttn(rows, ctx.batch_row_offsets, proj, fa, layer, local_decode_mask)
    else
        try self.mimoAttnCore(ctx, proj, fa, layer, offset, batch, seq_len, is_prefill, local_prefill_mask, local_decode_mask);
    defer _ = mlx.mlx_array_free(attn_out);

    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, self.s));
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));
    if (self.imatrix) |c| try c.observeLinear(.{ .o_proj = layer }, attn_flat);
    return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
}

/// Rope, KV append and attention of one MiMo layer over the QKV projection `proj`, on
/// `ctx`'s cache at `offset`: `[batch, heads, seq_len, v_dim]`.
pub fn mimoAttnCore(
    self: *Transformer,
    ctx: *ForwardCtx,
    proj: [3]mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    local_prefill_mask: *mlx.mlx_array,
    local_decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    const cfg = &self.config;
    const is_global = cfg.isGlobalLayer(layer);
    const h_count: c_int = @intCast(cfg.layerNumHeads(layer));
    const kv_h: c_int = @intCast(cfg.layerKVHeads(layer));
    const hd: c_int = @intCast(cfg.layerHeadDim(layer));
    const vhd: c_int = @intCast(cfg.layerVHeadDim(layer));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(hd)) * cfg.partial_rotary_factor);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
    const k_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    const v_shape = [_]c_int{ batch, seq_len, kv_h, vhd };
    const perm = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    const rope_base = mlx.mlx_optional_float{
        .value = if (is_global) cfg.rope_theta else cfg.rope_local_base_freq,
        .has_value = true,
    };
    const sliding = slidingViewFor(cfg, offset + seq_len, seq_len);
    const max_kv: u32 = if (is_global) 0 else sliding.span;
    const fp8_qkv = mlx.mlx_array_dtype(fa.q_w) == .uint8;
    const prepped: ?DenseKVView = if (fp8_qkv and batch == 1 and seq_len <= MIMO_VERIFY_ROWS_MAX and hd == 192 and vhd == 128)
        try self.mimoDecodePrep(ctx, proj, is_global, rope_dims, rope_base, offset, seq_len, h_count, kv_h, layer, max_kv, &q_rope)
    else
        null;
    var kv_view = prepped orelse blk: {
        var q_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_r);
        try mlx.check(mlx.mlx_reshape(&q_r, proj[0], &q_shape, 4, self.s));
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_r, &perm, 4, self.s));
        try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, offset, .{ .ctx = null }, self.s));

        var k_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_r);
        try mlx.check(mlx.mlx_reshape(&k_r, proj[1], &k_shape, 4, self.s));
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_r, &perm, 4, self.s));
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, offset, .{ .ctx = null }, self.s));

        var v_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_r);
        try mlx.check(mlx.mlx_reshape(&v_r, proj[2], &v_shape, 4, self.s));
        var v_t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));
        var v_scaled = v_t;
        var v_scale: mlx.mlx_array = .{ .ctx = null };
        const value_scale: f32 = if (fp8_qkv) 1.0 else cfg.attention_value_scale;
        if (value_scale != 1.0) {
            v_scale = try scalarOf(value_scale, mlx.mlx_array_dtype(v_t), self.s);
            var scaled = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&scaled, v_t, v_scale, self.s));
            _ = mlx.mlx_array_free(v_t);
            v_scaled = scaled;
        }
        defer {
            _ = mlx.mlx_array_free(v_scaled);
            if (v_scale.ctx != null) _ = mlx.mlx_array_free(v_scale);
        }
        break :blk try ctx.cache.update(layer, k_rope, v_scaled, self.s, max_kv);
    };
    defer kv_view.deinit();

    var attn_out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(attn_out);
    switch (try mimoAttnArm(ctx.verify_rows, is_prefill, is_global, seq_len)) {
        .verify_rows, .prefill_rows => {
            if (!ctx.verify_rows and !mimo_prefill_rows_logged) {
                mimo_prefill_rows_logged = true;
                log.info("[mimo] short global prefill engaged: {d} rows read row by row over {d} keys\n", .{ seq_len, offset + seq_len });
            }
            _ = mlx.mlx_array_free(attn_out);
            attn_out = try mimoVerifyRowsAttn(self.s, @intCast(cfg.sliding_window), q_rope, &kv_view, fa.sinks, is_global, offset, seq_len, attn_scale, local_decode_mask);
        },
        .decode => {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = try mimoDecodeAttn(self.s, @intCast(cfg.sliding_window), q_rope, &kv_view, fa.sinks, is_global, offset + seq_len, attn_scale, local_decode_mask);
        },
        .prefill_global => {
            // MLX has no fused prefill kernel at qk 192 (steel_attention ships
            // bd 64/96/128 and a 256 dsplit), so the composed arm materializes
            // [heads, chunk, total_kv] — 32 GiB at a 512k prompt. A global layer
            // that carries sinks keeps it: only the band arm's sink is proven.
            const pd: ?mlx.mlx_array = if (fa.sinks.ctx == null)
                try fusedSdpaPrefillKv(self.s, q_rope, &kv_view, attn_scale, 0, .{ .ctx = null })
            else
                null;
            if (pd) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, "causal", none_mask, fa.sinks, false, self.s));
            }
        },
        .prefill_sliding => {
            const sw: c_int = @intCast(cfg.sliding_window);
            if (try slidingPrefillAttn(self.s, cfg, q_rope, &kv_view, attn_scale, fa.sinks)) |out| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = out;
            } else if (offset + seq_len <= sw) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, "causal", none_mask, fa.sinks, false, self.s));
            } else {
                if (local_prefill_mask.ctx == null) {
                    local_prefill_mask.* = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
                }
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, "array", local_prefill_mask.*, fa.sinks, false, self.s));
            }
        },
    }

    return attn_out;
}

/// Row i of a batched decode attends as slot i's own decode tick: row i of the QKV
/// projection through `mimoAttnCore` on that slot's cache at its offset. `[1, heads, N, v_dim]`.
pub fn mimoBatchRowsAttn(
    self: *Transformer,
    rows: []const *ForwardCtx,
    offsets: []const u32,
    proj: [3]mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    local_decode_mask: mlx.mlx_array,
) !mlx.mlx_array {
    std.debug.assert(rows.len == offsets.len and rows.len <= MIMO_VERIFY_ROWS_MAX);
    var outs: [MIMO_VERIFY_ROWS_MAX]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (outs[0..n]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var no_prefill_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(no_prefill_mask);
    for (rows, offsets, 0..) |row_ctx, off, i| {
        var row_proj: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
        defer for (row_proj) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (proj, &row_proj) |src, *dst| {
            const sh = mlx.getShape(src);
            dst.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(dst, src, &[_]c_int{ 0, @intCast(i), 0 }, 3, &[_]c_int{ 1, @as(c_int, @intCast(i)) + 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        }
        outs[n] = try self.mimoAttnCore(row_ctx, row_proj, fa, layer, @intCast(off), 1, 1, false, &no_prefill_mask, local_decode_mask);
        n += 1;
    }
    const vec = mlx.mlx_vector_array_new_data(&outs, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, self.s));
    return out;
}

pub fn mimoRouterLogits(self: *Transformer, router_x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    // MiMo routing is defined over f32 activations and f32 router weights,
    // independently of the residual/expert storage dtype.
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, router_x, .float32, self.s));
    if (mw.router_s.ctx != null) {
        const qp = self.quantParamsHinted(mw.router_w, mw.router_s, lastDim(x32));
        return qmatmulBits(x32, mw.router_w, mw.router_s, mw.router_b, qp.bits, qp.group_size, qp.mode, self.s);
    }
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, mw.router_w, .float32, self.s));
    return qmatmulBits(x32, w32, .{ .ctx = null }, .{ .ctx = null }, 0, 0, .affine, self.s);
}

/// MiMo keeps the normalized sigmoid weights in f32 until the expert sum. The
/// reference only narrows the final residual, so bf16 routing weights change
/// near-tied expert contributions even when selection is unchanged.
pub fn mimoRoutingChain(router_logits: mlx.mlx_array, expert_bias: mlx.mlx_array, k: c_int, route_norm: bool, route_scale: f32, s: mlx.mlx_stream) !Transformer.MoeRouting {
    return sigmoidBiasRoutingChain(router_logits, expert_bias, k, route_norm, route_scale, .float32, s);
}

// ── MiMo decode QKV prep: partial rope of Q and K + kv8 quantize of K and V, one dispatch ──
//
// One dispatch per layer and forward does the work of two partial ropes (each a whole-input copy
// plus the rotation) and two `affine_quantize` dispatches. The rotation mirrors `rope_single_impl`
// against the `ropeAngleRow` probe's floats; each 64-wide group is one simdgroup running
// `affine_quantize<T, 64, BITS>` line for line (BITS 8 or 6; 6 packs four values into three bytes, two lanes
// per three bytes). Bit-identical to the composed ops.
pub const MIMO_QKV_PREP_HEADER =
    \\template <typename T, int RD>
    \\inline T sushi_rope_elem(const device T* src, uint j, const device float* angles) {
    \\    constexpr uint half_rd = uint(RD) / 2;
    \\    if (j < half_rd) {
    \\        float x1 = float(src[j]);
    \\        float x2 = float(src[j + half_rd]);
    \\        return static_cast<T>(x1 * angles[j] - x2 * angles[half_rd + j]);
    \\    } else if (j < uint(RD)) {
    \\        uint p = j - half_rd;
    \\        float x1 = float(src[p]);
    \\        float x2 = float(src[j]);
    \\        return static_cast<T>(x1 * angles[j] + x2 * angles[p]);
    \\    }
    \\    return src[j];
    \\}
    \\template <typename T, int BITS>
    \\inline void sushi_q_group(thread float* w, uint lane, device uint8_t* out, device T* scale_out, device T* bias_out) {
    \\    constexpr float eps = 1e-7;
    \\    constexpr float n_bins = float((1u << BITS) - 1u);
    \\    float w_min = Limits<T>::max;
    \\    float w_max = 0;
    \\    for (int i = 0; i < 2; i++) {
    \\        w_min = min(w_min, w[i]);
    \\        w_max = max(w_max, w[i]);
    \\    }
    \\    w_min = simd_min(w_min);
    \\    w_max = simd_max(w_max);
    \\    float scale = max((w_max - w_min) / n_bins, eps);
    \\    bool side = abs(w_min) > abs(w_max);
    \\    scale = side ? scale : -scale;
    \\    float edge = side ? w_min : w_max;
    \\    float q0 = round(edge / scale);
    \\    bool at_zero = q0 == 0.0f;
    \\    scale = at_zero ? scale : edge / q0;
    \\    float bias = at_zero ? 0 : edge;
    \\    if (lane == 0) {
    \\        *scale_out = static_cast<T>(scale);
    \\        *bias_out = static_cast<T>(bias);
    \\    }
    \\    uint val[2];
    \\    for (int i = 0; i < 2; i++) {
    \\        val[i] = uint(min(round((w[i] - bias) / scale), n_bins));
    \\    }
    \\    if (BITS == 8) {
    \\        out[lane * 2] = uint8_t(val[0]);
    \\        out[lane * 2 + 1] = uint8_t(val[1]);
    \\    } else {
    \\        // Four 6-bit values make three bytes (a little-endian bitstream): an even lane writes its pair and the next lane's.
    \\        uint pair = val[0] | (val[1] << 6);
    \\        uint next = simd_shuffle_down(pair, 1);
    \\        if ((lane & 1u) == 0) {
    \\            uint x = pair | (next << 12);
    \\            device uint8_t* o = out + (lane >> 1) * 3;
    \\            o[0] = uint8_t(x & 255u);
    \\            o[1] = uint8_t((x >> 8) & 255u);
    \\            o[2] = uint8_t((x >> 16) & 255u);
    \\        }
    \\    }
    \\}
;

pub const MIMO_QKV_PREP_SOURCE =
    \\constexpr uint HD = 192;
    \\constexpr uint VD = 128;
    \\constexpr uint QG = uint(HQ) * (HD / 64);
    \\constexpr uint KG = uint(HK) * (HD / 64);
    \\uint g = threadgroup_position_in_grid.x;
    \\uint r = threadgroup_position_in_grid.y;
    \\uint lane = thread_position_in_threadgroup.x;
    \\const device float* row_angles = angles + r * uint(RD);
    \\if (g < QG) {
    \\    uint head = g / (HD / 64);
    \\    uint j = (g % (HD / 64)) * 64 + lane * 2;
    \\    for (uint i = 0; i < 2; ++i) {
    \\        oq[(head * NR + r) * HD + j + i] = sushi_rope_elem<T, RD>(q + (r * uint(HQ) + head) * HD, j + i, row_angles);
    \\    }
    \\    return;
    \\}
    \\float w[2];
    \\device uint8_t* qout;
    \\device T* sc;
    \\device T* bi;
    \\if (g < QG + KG) {
    \\    uint kg = g - QG;
    \\    uint head = kg / (HD / 64);
    \\    uint j = (kg % (HD / 64)) * 64 + lane * 2;
    \\    for (uint i = 0; i < 2; ++i) {
    \\        w[i] = float(sushi_rope_elem<T, RD>(k + (r * uint(HK) + head) * HD, j + i, row_angles));
    \\    }
    \\    uint slot = (head * NR + r) * (HD / 64) + kg % (HD / 64);
    \\    qout = (device uint8_t*)kq + (head * NR + r) * (HD * BITS / 8) + (kg % (HD / 64)) * (8 * BITS);
    \\    sc = ks + slot;
    \\    bi = kb + slot;
    \\} else {
    \\    uint vg = g - QG - KG;
    \\    uint head = vg / (VD / 64);
    \\    uint j = (vg % (VD / 64)) * 64 + lane * 2;
    \\    for (uint i = 0; i < 2; ++i) {
    \\        w[i] = float(v[(r * uint(HK) + head) * VD + j + i]);
    \\    }
    \\    uint slot = (head * NR + r) * (VD / 64) + vg % (VD / 64);
    \\    qout = (device uint8_t*)vq + (head * NR + r) * (VD * BITS / 8) + (vg % (VD / 64)) * (8 * BITS);
    \\    sc = vs + slot;
    \\    bi = vb + slot;
    \\}
    \\sushi_q_group<T, BITS>(w, lane, qout, sc, bi);
;

pub var mimo_qkv_prep_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub var mimo_qkv_prep_engaged: bool = false;

pub const MimoQkvPrepKey = struct { hq: c_int, hk: c_int, rd: c_int, rows: c_int, bits: u8 };

/// Two layer geometries (global, sliding) at every verify width, for each of the two cache widths.
pub var mimo_qkv_prep_cfgs: [4 * MIMO_VERIFY_ROWS_MAX]?struct { key: MimoQkvPrepKey, cfg: mlx.mlx_fast_metal_kernel_config } = @splat(null);

pub const MimoQkvPrep = struct {
    q: mlx.mlx_array, // [1, HQ, rows, 192], roped
    k: kv_quant.QuantizedKV, // roped K, affine-8 or -6 g64, [1, HK, rows, *]
    v: kv_quant.QuantizedKV,

    pub fn deinit(self: *MimoQkvPrep) void {
        _ = mlx.mlx_array_free(self.q);
        self.k.deinit();
        self.v.deinit();
    }
};

pub fn mimoQkvPrepConfig(key: MimoQkvPrepKey, dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    for (&mimo_qkv_prep_cfgs) |*slot| {
        if (slot.*) |c| {
            if (std.meta.eql(c.key, key)) return c.cfg;
        } else {
            const config = mlx.mlx_fast_metal_kernel_config_new();
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
            const hq = key.hq;
            const hk = key.hk;
            const rows = key.rows;
            const shapes = [_][4]c_int{
                .{ 1, hq, rows, 192 },
                .{ 1, hk, rows, @divTrunc(192 * @as(c_int, key.bits), 32) },
                .{ 1, hk, rows, 3 },
                .{ 1, hk, rows, 3 },
                .{ 1, hk, rows, @divTrunc(128 * @as(c_int, key.bits), 32) },
                .{ 1, hk, rows, 2 },
                .{ 1, hk, rows, 2 },
            };
            const dtypes = [_]mlx.mlx_dtype{ dt, .uint32, dt, dt, .uint32, dt, dt };
            for (shapes, dtypes) |sh, odt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &sh, 4, odt));
            const groups = hq * 3 + hk * 3 + hk * 2;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, groups * 32, rows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HK", hk));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", key.rd));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NR", rows));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", key.bits));
            slot.* = .{ .key = key, .cfg = config };
            return config;
        }
    }
    return error.MimoQkvPrepConfigsFull;
}

/// Up to `MIMO_VERIFY_ROWS_MAX` rows of a MiMo layer's Q/K/V on a kv8 or kv6 (g64) cache: `q_flat`
/// [1,R,HQ*192], `k_flat` [1,R,HK*192], `v_flat` [1,R,HK*128] (already value-scaled), `angles`
/// the [R, rd] `ropeAngleRows` from the first row's position. Null = geometry the kernel does
/// not cover.
pub fn mimoDecodeQkvPrep(
    s: mlx.mlx_stream,
    q_flat: mlx.mlx_array,
    k_flat: mlx.mlx_array,
    v_flat: mlx.mlx_array,
    angles: mlx.mlx_array,
    hq: c_int,
    hk: c_int,
    rd: c_int,
    rows: c_int,
    bits: u8,
) !?MimoQkvPrep {
    if (bits != 8 and bits != 6) return null;
    const dt = mlx.mlx_array_dtype(q_flat);
    if (dt != .bfloat16 or mlx.mlx_array_dtype(k_flat) != .bfloat16 or mlx.mlx_array_dtype(v_flat) != .bfloat16) return null;
    if (rd != 64 or rows < 1 or rows > MIMO_VERIFY_ROWS_MAX) return null;
    if (mlx.mlx_array_size(q_flat) != @as(usize, @intCast(rows * hq * 192)) or
        mlx.mlx_array_size(k_flat) != @as(usize, @intCast(rows * hk * 192)) or
        mlx.mlx_array_size(v_flat) != @as(usize, @intCast(rows * hk * 128))) return null;

    const config = mimoQkvPrepConfig(.{ .hq = hq, .hk = hk, .rd = rd, .rows = rows, .bits = bits }, dt) catch |err| {
        if (err == error.MimoQkvPrepConfigsFull) return null;
        return err;
    };
    const kernel = mimo_qkv_prep_kernel orelse blk: {
        const input_names = [_][*:0]const u8{ "q", "k", "v", "angles" };
        const output_names = [_][*:0]const u8{ "oq", "kq", "ks", "kb", "vq", "vs", "vb" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const kn = mlx.mlx_fast_metal_kernel_new("sushi_mimo_qkv_prep", in_vec, out_vec, MIMO_QKV_PREP_SOURCE, MIMO_QKV_PREP_HEADER, true, false);
        if (kn.ctx == null) return error.MetalKernelCompileFailed;
        mimo_qkv_prep_kernel = kn;
        break :blk kn;
    };

    const inputs_arr = [_]mlx.mlx_array{ q_flat, k_flat, v_flat, angles };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 7) return error.MetalKernelBadOutputCount;
    var outs: [7]mlx.mlx_array = undefined;
    for (&outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, outputs_vec, i));
    }
    if (!mimo_qkv_prep_engaged) {
        mimo_qkv_prep_engaged = true;
        log.info("[attn] MiMo decode QKV prep engaged: hq={d} hk={d} rd={d} rows={d} (partial rope + kv{d} quantize, one dispatch)\n", .{ hq, hk, rd, rows, bits });
    }
    return .{
        .q = outs[0],
        .k = .{ .q = outs[1], .scales = outs[2], .biases = outs[3] },
        .v = .{ .q = outs[4], .scales = outs[5], .biases = outs[6] },
    };
}

/// Split geometry of the range arm, measured at MiMo's global decode shape (M5 Max): the split
/// count is a threadgroup-count decision, flat past 256k keys.
pub const QKV_SPLITK_KEYS_PER_SPLIT: c_int = 512;
pub const QKV_SPLITK_MIN_SPLITS: c_int = 64;
pub const QKV_SPLITK_MAX_SPLITS: c_int = 128;
/// µbench seam: split geometry (keys per split, split floor, split cap).
pub var qkv_splitk_split_override: ?[3]c_int = null;

pub fn qkvSplitKSplits(t_k: c_int) c_int {
    const g = qkv_splitk_split_override orelse .{ QKV_SPLITK_KEYS_PER_SPLIT, QKV_SPLITK_MIN_SPLITS, QKV_SPLITK_MAX_SPLITS };
    return std.math.clamp(@divTrunc(t_k + g[0] - 1, g[0]), g[1], g[2]);
}

var qkv_splitk_engaged: bool = false;

/// Packed-KV attention through the QSA split-K body over each row's whole causal range: no
/// matrix units, so it serves every Apple GPU. t_q 1..8, no mask or end-aligned causal; any
/// (DK, DV) up to 256 in multiples of the quant group. Null = declined.
pub fn qkvAttnSplitKKernel(s: mlx.mlx_stream, q_in: mlx.mlx_array, view: *const DenseKVView, scale: f32, mask_mode: []const u8) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (!view.has_quant_triple or !KVQuantConfig.supportedBits(view.bits)) return null;
    const gs: c_int = @intCast(view.group_size);
    if (gs == 0 or @rem(gs, 8) != 0) return null;
    if (mlx.mlx_array_ndim(q_in) != 4 or mlx.mlx_array_dtype(q_in) != .bfloat16) return null;
    const qs = mlx.getShape(q_in);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 1 or t_q > QKV_VERIFY_MAX_TQ or mlx.mlx_array_strides(q_in)[3] != 1) return null;
    if (!std.mem.eql(u8, mask_mode, "causal") and !(t_q == 1 and mask_mode.len == 0)) return null;
    const arrays = [_]mlx.mlx_array{ view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.v_triple_q, view.v_triple_scales, view.v_triple_biases };
    for (arrays) |a| {
        if (a.ctx == null or mlx.mlx_array_ndim(a) != 4 or mlx.mlx_array_strides(a)[3] != 1) return null;
    }
    const ks = mlx.getShape(arrays[0]);
    const vs = mlx.getShape(arrays[3]);
    const dk: c_int = kv_quant.unpackedDim(ks[3], view.bits);
    const dv: c_int = kv_quant.unpackedDim(vs[3], view.bits);
    // Registers hold DK/8 + DV/8 fragments; the staging holds a DK x (BK + 8) tile.
    if (qs[3] != dk or dk > 256 or dv > 256 or @rem(dk, gs) != 0 or @rem(dv, gs) != 0) return null;
    const h_kv = ks[1];
    const t_k = ks[2];
    if (h_kv <= 0 or ks[0] != 1 or vs[0] != 1 or vs[1] != h_kv or vs[2] != t_k or @rem(qs[1], h_kv) != 0 or t_k < t_q) return null;
    const gqa = @divExact(qs[1], h_kv);
    if (gqa > 64) return null;
    for ([_]usize{ 1, 2, 4, 5 }, [_]c_int{ dk, dk, dv, dv }) |i, d| {
        const shape = mlx.getShape(arrays[i]);
        if (shape[0] != 1 or shape[1] != h_kv or shape[2] != t_k or shape[3] != @divExact(d, gs)) return null;
    }

    const no_blocks_data = [_]i32{0};
    const no_blocks = mlx.mlx_array_new_data(&no_blocks_data, &[_]c_int{ 1, 1, 1 }, 3, .int32);
    defer _ = mlx.mlx_array_free(no_blocks);
    const key = QsaAttnCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = h_kv,
        .dv = dv,
        .bits = view.bits,
        .gs = view.group_size,
        .quant = true,
        .sparse = false,
        .nsg = @divTrunc(gqa + 7, 8),
        .nsgt = @divTrunc(gqa + 7, 8),
        .nsplit = qkvSplitKSplits(t_k),
        .bk = QSA_ATTN_BK,
        .ratio = 1,
        .dtype = .bfloat16,
    };
    const out = (try splitKAttnDispatch(s, q_in, no_blocks, arrays, scale, key)) orelse return null;
    if (!qkv_splitk_engaged) {
        qkv_splitk_engaged = true;
        log.info("[kv-attn] split-K packed attention engaged: bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} Tq={d} Tk={d} nsplit={d}\n", .{ view.bits, view.group_size, dk, dv, qs[1], h_kv, t_q, t_k, key.nsplit });
    }
    return out;
}

/// Merge per-block split-KV partials (m, l, unnormalized O) into a typed
/// [ROWS, DV] tensor — the standard flash-attention carry:
///   M  = max_b m_b            [ROWS, 1]
///   w  = exp(m_b - M)         [ROWS, NBLK]
///   L  = Σ_b l_b · w          [ROWS, 1]
///   O  = Σ_b o_b · w          [ROWS, DV]
///   out = (O / L) as T.
/// KB-scale tensors — a handful of elementwise/reduce ops per layer. The op
/// sequence is load-bearing (decode bit-identity across refactors is a bar).
fn qkvMergePartials(s: mlx.mlx_stream, outputs_vec: mlx.mlx_vector_array, qdt: mlx.mlx_dtype) !mlx.mlx_array {
    if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
    var m_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_b);
    var l_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_b);
    var o_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_b);
    try mlx.check(mlx.mlx_vector_array_get(&m_b, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&l_b, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&o_b, outputs_vec, 2));
    var m_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_max);
    try mlx.check(mlx.mlx_max_axis(&m_max, m_b, 1, true, s));
    var m_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m_shift);
    try mlx.check(mlx.mlx_subtract(&m_shift, m_b, m_max, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_exp(&w, m_shift, s));
    var lw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lw);
    try mlx.check(mlx.mlx_multiply(&lw, l_b, w, s));
    var l_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_tot);
    try mlx.check(mlx.mlx_sum_axis(&l_tot, lw, 1, true, s)); // [ROWS, 1]
    var w3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w3);
    try mlx.check(mlx.mlx_expand_dims(&w3, w, 2, s)); // [ROWS, NBLK, 1]
    var ow = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ow);
    try mlx.check(mlx.mlx_multiply(&ow, o_b, w3, s));
    var o_tot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_tot);
    try mlx.check(mlx.mlx_sum_axis(&o_tot, ow, 1, false, s)); // [ROWS, DV]
    var o_norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o_norm);
    try mlx.check(mlx.mlx_divide(&o_norm, o_tot, l_tot, s)); // [ROWS,DV] / [ROWS,1]
    var o_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o_t);
    try mlx.check(mlx.mlx_astype(&o_t, o_norm, qdt, s));
    return o_t;
}

/// Widest verify block the packed matmul2d and split-K kernels serve.
pub const QKV_VERIFY_MAX_TQ: c_int = 8;

// ── matmul2d packed attention (t_q 1..8) ──
//
// One threadgroup (128 threads, 4 simdgroups) per (kv head, kv split). Per 32-token page:
// dequantize K into a threadgroup tile, QK through the matmul2d tensor op, an online softmax
// per q row, dequantize V into the same tile, PV accumulated into a cooperative running
// output rescaled when a row's max grows. Packed words are read into registers one phase
// before their tile needs them. Each split emits the (m, l, unnormalized O) partials
// `qkvMergePartials` reads. Ported from upstream mlx-serve, structure after Inco Splash's q8
// attention tile (Apache-2.0, see NOTICE).
const QKV_MPP_PAGE: c_int = 32;
const QKV_MPP_MAX_SPLITS: c_int = 128;
const QKV_MPP_PAGES_PER_SPLIT: c_int = 2;

const QKV_MPP_KERNEL_SOURCE =
    \\using namespace mpp::tensor_ops;
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int MR = GQA * TQ;
    \\constexpr int M = MPAD; // matmul2d wants rows in multiples of 8; pad rows carry zero queries
    \\constexpr int N = 32;
    \\constexpr int NT = 128; // the page loop is barrier- and matmul-latency bound: 8 simdgroups wait on each other
    \\constexpr int UW = BITS == 6 ? 3 : 2; // uint32 words per load unit
    \\constexpr int UV = UW * 32 / BITS; // values per unit: 16 at 4 and 6 bits, 8 at 8
    \\constexpr int KC = DK / UV; // units per K row
    \\constexpr int VC = DV / UV;
    \\constexpr int CPG = GS / UV; // units per quant group
    \\constexpr int KIT = (N * KC + NT - 1) / NT;
    \\constexpr int VIT = (N * VC + NT - 1) / NT;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint hkv = threadgroup_position_in_grid.y;
    \\const uint split = threadgroup_position_in_grid.z;
    \\const uint nsplit = threadgroups_per_grid.z;
    \\const int Tk = tk[0];
    \\const float scale = scl[0];
    \\const uint mask_bits = (1u << BITS) - 1u;
    \\const int pages = (Tk + N - 1) / N;
    \\const int per = (pages + int(nsplit) - 1) / int(nsplit);
    \\const int pb = int(split) * per;
    \\const int pe = min(pages, pb + per);
    \\
    \\threadgroup T tile[N * (DK > DV ? DK : DV)];
    \\threadgroup float sp[M * N]; // page scores, overwritten in place by the row's probabilities
    \\threadgroup float rmax[M];
    \\threadgroup float rsum[M];
    \\threadgroup float pscale[M];
    \\
    \\if (tid < uint(M)) { rmax[tid] = -INFINITY; rsum[tid] = 0.0f; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\constexpr auto qk_desc = matmul2d_descriptor(M, N, DK, false, true, false, matmul2d_descriptor::mode::multiply);
    \\constexpr auto pv_desc = matmul2d_descriptor(M, DV, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<qk_desc, execution_simdgroups<NT / 32>> qk;
    \\matmul2d<pv_desc, execution_simdgroups<NT / 32>> pv;
    \\auto qt = tensor((device T*)q + (long)hkv * M * DK, dextents<int, 2>{DK, M}, array<int, 2>{1, DK});
    \\auto kt = tensor(tile, dextents<int, 2>{DK, N}, array<int, 2>{1, DK});
    \\auto vt = tensor(tile, dextents<int, 2>{DV, N}, array<int, 2>{1, DV});
    \\auto st = tensor(sp, dextents<int, 2>{N, M}, array<int, 2>{1, N});
    \\auto q0 = qt.template slice<DK, M>(0, 0);
    \\auto k0 = kt.template slice<DK, N>(0, 0);
    \\auto v0 = vt.template slice<DV, N>(0, 0);
    \\auto p0 = st.template slice<N, M>(0, 0);
    \\auto running = pv.template get_destination_cooperative_tensor<
    \\    metal::remove_addrspace_t<decltype(p0)>, metal::remove_addrspace_t<decltype(v0)>, float>();
    \\for (ushort i = 0; i < running.get_capacity(); ++i)
    \\  if (running.is_valid_element(i)) running[i] = 0.0f;
    \\
    \\const long kq0 = (long)hkv * kq_strides[1], ks0 = (long)hkv * ksc_strides[1], kb0 = (long)hkv * kbi_strides[1];
    \\const long vq0 = (long)hkv * vq_strides[1], vs0 = (long)hkv * vsc_strides[1], vb0 = (long)hkv * vbi_strides[1];
    \\uint2 kr[KIT]; uint kr3[KIT]; float ksr[KIT], kbr[KIT];
    \\uint2 vr[VIT]; uint vr3[VIT]; float vsr[VIT], vbr[VIT];
    \\// Rows past Tk re-read row Tk-1 and are masked in the softmax.
    \\#define QKV_LOAD(C, IT, REG, SREG, BREG, QP, Q0, QSTR, SP, S0, SSTR, BP, B0, BSTR, TOK0) \
    \\  for (int i = 0; i < IT; ++i) { \
    \\    const int c = int(tid) + i * NT; \
    \\    if (c < N * C) { \
    \\      const int n = c / C, ci = c % C, g = ci / CPG; \
    \\      const long r = (long)min((TOK0) + n, Tk - 1); \
    \\      if constexpr (BITS == 6) { \
    \\        const device uint* wp = (const device uint*)(QP + Q0 + r * QSTR[2]) + ci * 3; \
    \\        REG[i] = uint2(wp[0], wp[1]); \
    \\        REG##3[i] = wp[2]; \
    \\      } else { \
    \\        REG[i] = ((const device uint2*)(QP + Q0 + r * QSTR[2]))[ci]; \
    \\      } \
    \\      SREG[i] = float(SP[S0 + r * SSTR[2] + g]); \
    \\      BREG[i] = float(BP[B0 + r * BSTR[2] + g]); \
    \\    } \
    \\  }
    \\#define QKV_STORE(C, D, IT, REG, SREG, BREG) \
    \\  for (int i = 0; i < IT; ++i) { \
    \\    const int c = int(tid) + i * NT; \
    \\    if (c < N * C) { \
    \\      threadgroup T* dst = tile + (c / C) * D + (c % C) * UV; \
    \\      if constexpr (BITS == 6) { \
    \\        sushi_kv_store16_b6<T>(REG[i].x, REG[i].y, REG##3[i], SREG[i], BREG[i], dst); \
    \\      } else { \
    \\        for (int h = 0; h < 2; ++h) \
    \\          for (int u = 0; u < VPW; ++u) \
    \\            dst[h * VPW + u] = T(float((REG[i][h] >> (u * BITS)) & mask_bits) * SREG[i] + BREG[i]); \
    \\      } \
    \\    } \
    \\  }
    \\if (pb < pe) { QKV_LOAD(KC, KIT, kr, ksr, kbr, kq, kq0, kq_strides, ksc, ks0, ksc_strides, kbi, kb0, kbi_strides, pb * N) }
    \\for (int page = pb; page < pe; ++page) {
    \\  const int tok0 = page * N;
    \\  QKV_STORE(KC, DK, KIT, kr, ksr, kbr)
    \\  // In flight while QK and the softmax run: this page's V and the next page's K.
    \\  QKV_LOAD(VC, VIT, vr, vsr, vbr, vq, vq0, vq_strides, vsc, vs0, vsc_strides, vbi, vb0, vbi_strides, tok0)
    \\  if (page + 1 < pe) { QKV_LOAD(KC, KIT, kr, ksr, kbr, kq, kq0, kq_strides, ksc, ks0, ksc_strides, kbi, kb0, kbi_strides, tok0 + N) }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  auto page_scores = qk.template get_destination_cooperative_tensor<
    \\      metal::remove_addrspace_t<decltype(q0)>, metal::remove_addrspace_t<decltype(k0)>, float>();
    \\  qk.run(q0, k0, page_scores);
    \\  page_scores.store(p0);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  // Online softmax: simdgroup sg owns rows sg, sg+4, ...; lane n owns page column n (N = 32).
    \\  for (int m = int(tid / 32u); m < M; m += NT / 32) {
    \\    const int n = int(tid % 32u);
    \\    const int lim = m < MR ? Tk - TQ + 1 + (m % TQ) : Tk;
    \\    const bool live = tok0 + n < lim;
    \\    const float x = live ? sp[m * N + n] * scale : -INFINITY;
    \\    const float prev = rmax[m];
    \\    const float next = max(prev, simd_max(x));
    \\    const float pr = live ? metal::exp(x - next) : 0.0f;
    \\    sp[m * N + n] = pr;
    \\    const float lsum = simd_sum(pr);
    \\    if (n == 0) {
    \\      const float sc = (next == -INFINITY || next == prev) ? 1.0f : metal::exp(prev - next);
    \\      pscale[m] = sc;
    \\      rsum[m] = rsum[m] * sc + lsum;
    \\      rmax[m] = next;
    \\    }
    \\  }
    \\  QKV_STORE(VC, DV, VIT, vr, vsr, vbr)
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (ushort i = 0; i < running.get_capacity(); ++i) {
    \\    if (!running.is_valid_element(i)) continue;
    \\    const auto c = running.get_multidimensional_index(i);
    \\    running[i] *= pscale[c[1]];
    \\  }
    \\  pv.run(p0, v0, running);
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\#undef QKV_LOAD
    \\#undef QKV_STORE
    \\// Partials land in the merge layout [Hkv * MR, NSPLIT(, DV)]; pad rows are never stored.
    \\for (ushort i = 0; i < running.get_capacity(); ++i) {
    \\  if (!running.is_valid_element(i)) continue;
    \\  const auto c = running.get_multidimensional_index(i);
    \\  if (c[1] < MR) o_out[(((long)hkv * MR + c[1]) * nsplit + split) * DV + c[0]] = running[i];
    \\}
    \\if (tid < uint(MR)) {
    \\  const long row = (long)hkv * MR + tid;
    \\  m_out[row * nsplit + split] = max(rmax[tid], -1e30f);
    \\  l_out[row * nsplit + split] = rsum[tid];
    \\}
;

// The same page walk for TQ verify rows at once: each page's K/V is staged once, and every row runs
// the decode kernel's own GQA-row matmuls, softmax and rescale on it, so a row's partials are its
// decode tick's when the rows share its page partition. q is (kv head, row, group head)-major.
const QKV_MPP_ROWS_KERNEL_SOURCE =
    \\using namespace mpp::tensor_ops;
    \\constexpr int VPW = 32 / BITS;
    \\constexpr int M = GQA;
    \\constexpr int MR = GQA * TQ;
    \\constexpr int N = 32;
    \\constexpr int NT = 128;
    \\constexpr int UW = BITS == 6 ? 3 : 2;
    \\constexpr int UV = UW * 32 / BITS;
    \\constexpr int KC = DK / UV;
    \\constexpr int VC = DV / UV;
    \\constexpr int CPG = GS / UV;
    \\constexpr int KIT = (N * KC + NT - 1) / NT;
    \\constexpr int VIT = (N * VC + NT - 1) / NT;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\const uint hkv = threadgroup_position_in_grid.y;
    \\const uint split = threadgroup_position_in_grid.z;
    \\const uint nsplit = threadgroups_per_grid.z;
    \\const int Tk = tk[0];
    \\const float scale = scl[0];
    \\const uint mask_bits = (1u << BITS) - 1u;
    \\const int pages = (Tk + N - 1) / N;
    \\const int per = (pages + int(nsplit) - 1) / int(nsplit);
    \\const int pb = int(split) * per;
    \\const int pe = min(pages, pb + per);
    \\
    \\threadgroup T tile[N * (DK > DV ? DK : DV)];
    \\threadgroup float sp[MR * N];
    \\threadgroup float rmax[MR];
    \\threadgroup float rsum[MR];
    \\threadgroup float pscale[MR];
    \\
    \\if (tid < uint(MR)) { rmax[tid] = -INFINITY; rsum[tid] = 0.0f; }
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\constexpr auto qk_desc = matmul2d_descriptor(M, N, DK, false, true, false, matmul2d_descriptor::mode::multiply);
    \\constexpr auto pv_desc = matmul2d_descriptor(M, DV, N, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<qk_desc, execution_simdgroups<NT / 32>> qk;
    \\matmul2d<pv_desc, execution_simdgroups<NT / 32>> pv;
    \\auto qt = tensor((device T*)q + (long)hkv * MR * DK, dextents<int, 2>{DK, MR}, array<int, 2>{1, DK});
    \\auto kt = tensor(tile, dextents<int, 2>{DK, N}, array<int, 2>{1, DK});
    \\auto vt = tensor(tile, dextents<int, 2>{DV, N}, array<int, 2>{1, DV});
    \\auto st = tensor(sp, dextents<int, 2>{N, MR}, array<int, 2>{1, N});
    \\auto k0 = kt.template slice<DK, N>(0, 0);
    \\auto v0 = vt.template slice<DV, N>(0, 0);
    \\// One running output per verify row (MSL takes no arrays of cooperative tensors); TQ <= 4.
    \\#define ROWS(X) X(0) if (TQ > 1) { X(1) } if (TQ > 2) { X(2) } if (TQ > 3) { X(3) }
    \\#define RUN_DECL(b) auto running##b = pv.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(st.template slice<N, M>(0, 0))>, metal::remove_addrspace_t<decltype(v0)>, float>();
    \\RUN_DECL(0) RUN_DECL(1) RUN_DECL(2) RUN_DECL(3)
    \\#define RUN_ZERO(b) for (ushort i = 0; i < running##b.get_capacity(); ++i) if (running##b.is_valid_element(i)) running##b[i] = 0.0f;
    \\ROWS(RUN_ZERO)
    \\
    \\const long kq0 = (long)hkv * kq_strides[1], ks0 = (long)hkv * ksc_strides[1], kb0 = (long)hkv * kbi_strides[1];
    \\const long vq0 = (long)hkv * vq_strides[1], vs0 = (long)hkv * vsc_strides[1], vb0 = (long)hkv * vbi_strides[1];
    \\uint2 kr[KIT]; uint kr3[KIT]; float ksr[KIT], kbr[KIT];
    \\uint2 vr[VIT]; uint vr3[VIT]; float vsr[VIT], vbr[VIT];
    \\#define QKV_LOAD(C, IT, REG, SREG, BREG, QP, Q0, QSTR, SP, S0, SSTR, BP, B0, BSTR, TOK0) \
    \\  for (int i = 0; i < IT; ++i) { \
    \\    const int c = int(tid) + i * NT; \
    \\    if (c < N * C) { \
    \\      const int n = c / C, ci = c % C, g = ci / CPG; \
    \\      const long r = (long)min((TOK0) + n, Tk - 1); \
    \\      if constexpr (BITS == 6) { \
    \\        const device uint* wp = (const device uint*)(QP + Q0 + r * QSTR[2]) + ci * 3; \
    \\        REG[i] = uint2(wp[0], wp[1]); \
    \\        REG##3[i] = wp[2]; \
    \\      } else { \
    \\        REG[i] = ((const device uint2*)(QP + Q0 + r * QSTR[2]))[ci]; \
    \\      } \
    \\      SREG[i] = float(SP[S0 + r * SSTR[2] + g]); \
    \\      BREG[i] = float(BP[B0 + r * BSTR[2] + g]); \
    \\    } \
    \\  }
    \\#define QKV_STORE(C, D, IT, REG, SREG, BREG) \
    \\  for (int i = 0; i < IT; ++i) { \
    \\    const int c = int(tid) + i * NT; \
    \\    if (c < N * C) { \
    \\      threadgroup T* dst = tile + (c / C) * D + (c % C) * UV; \
    \\      if constexpr (BITS == 6) { \
    \\        sushi_kv_store16_b6<T>(REG[i].x, REG[i].y, REG##3[i], SREG[i], BREG[i], dst); \
    \\      } else { \
    \\        for (int h = 0; h < 2; ++h) \
    \\          for (int u = 0; u < VPW; ++u) \
    \\            dst[h * VPW + u] = T(float((REG[i][h] >> (u * BITS)) & mask_bits) * SREG[i] + BREG[i]); \
    \\      } \
    \\    } \
    \\  }
    \\if (pb < pe) { QKV_LOAD(KC, KIT, kr, ksr, kbr, kq, kq0, kq_strides, ksc, ks0, ksc_strides, kbi, kb0, kbi_strides, pb * N) }
    \\for (int page = pb; page < pe; ++page) {
    \\  const int tok0 = page * N;
    \\  QKV_STORE(KC, DK, KIT, kr, ksr, kbr)
    \\  QKV_LOAD(VC, VIT, vr, vsr, vbr, vq, vq0, vq_strides, vsc, vs0, vsc_strides, vbi, vb0, vbi_strides, tok0)
    \\  if (page + 1 < pe) { QKV_LOAD(KC, KIT, kr, ksr, kbr, kq, kq0, kq_strides, ksc, ks0, ksc_strides, kbi, kb0, kbi_strides, tok0 + N) }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\#define ROW_QK(b) { auto qb = qt.template slice<DK, M>(0, b * M); auto page_scores = qk.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(qb)>, metal::remove_addrspace_t<decltype(k0)>, float>(); qk.run(qb, k0, page_scores); auto sb = st.template slice<N, M>(0, b * M); page_scores.store(sb); }
    \\  ROWS(ROW_QK)
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (int m = int(tid / 32u); m < MR; m += NT / 32) {
    \\    const int n = int(tid % 32u);
    \\    const int lim = Tk - TQ + 1 + m / M;
    \\    const bool live = tok0 + n < lim;
    \\    const float x = live ? sp[m * N + n] * scale : -INFINITY;
    \\    const float prev = rmax[m];
    \\    const float next = max(prev, simd_max(x));
    \\    const float pr = live ? metal::exp(x - next) : 0.0f;
    \\    sp[m * N + n] = pr;
    \\    const float lsum = simd_sum(pr);
    \\    if (n == 0) {
    \\      const float sc = (next == -INFINITY || next == prev) ? 1.0f : metal::exp(prev - next);
    \\      pscale[m] = sc;
    \\      rsum[m] = rsum[m] * sc + lsum;
    \\      rmax[m] = next;
    \\    }
    \\  }
    \\  QKV_STORE(VC, DV, VIT, vr, vsr, vbr)
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\#define ROW_PV(b) { for (ushort i = 0; i < running##b.get_capacity(); ++i) { if (!running##b.is_valid_element(i)) continue; const auto c = running##b.get_multidimensional_index(i); running##b[i] *= pscale[b * M + c[1]]; } auto pb_ = st.template slice<N, M>(0, b * M); pv.run(pb_, v0, running##b); }
    \\  ROWS(ROW_PV)
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\#undef QKV_LOAD
    \\#undef QKV_STORE
    \\#define ROW_OUT(b) for (ushort i = 0; i < running##b.get_capacity(); ++i) { if (!running##b.is_valid_element(i)) continue; const auto c = running##b.get_multidimensional_index(i); o_out[(((long)hkv * MR + b * M + c[1]) * nsplit + split) * DV + c[0]] = running##b[i]; }
    \\ROWS(ROW_OUT)
    \\#undef ROWS
    \\#undef RUN_DECL
    \\#undef RUN_ZERO
    \\#undef ROW_QK
    \\#undef ROW_PV
    \\#undef ROW_OUT
    \\if (tid < uint(MR)) {
    \\  const long row = (long)hkv * MR + tid;
    \\  m_out[row * nsplit + split] = max(rmax[tid], -1e30f);
    \\  l_out[row * nsplit + split] = rsum[tid];
    \\}
;

var qkv_mpp_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qkv_mpp_rows_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
pub const QkvMppKey = struct { dk: c_int, dv: c_int, bits: u8, gs: u32, gqa: c_int, tq: c_int, dtype: mlx.mlx_dtype, rows: bool = false };
const QkvMppCfgKey = struct { t: QkvMppKey, h_kv: c_int, nsplit: c_int };
/// [decode kernel, rows kernel][t_q]
var qkv_mpp_cfgs: [2][@intCast(QKV_VERIFY_MAX_TQ + 1)]?mlx.mlx_fast_metal_kernel_config = @splat(@splat(null));
var qkv_mpp_cfg_keys: [2][@intCast(QKV_VERIFY_MAX_TQ + 1)]QkvMppCfgKey = @splat(@splat(std.mem.zeroes(QkvMppCfgKey)));
/// Template sets whose JIT build was proven on a one-page problem, and the ones that failed.
var qkv_mpp_probed: [16]?struct { key: QkvMppKey, ok: bool } = @splat(null);
var qkv_mpp_engaged: bool = false;

/// µbench seam: split geometry (pages per split, split cap).
pub var qkv_mpp_split_override: ?[2]c_int = null;

pub fn qkvMppSplits(t_k: c_int) c_int {
    const g = qkv_mpp_split_override orelse .{ QKV_MPP_PAGES_PER_SPLIT, QKV_MPP_MAX_SPLITS };
    const pages = @divTrunc(t_k + QKV_MPP_PAGE - 1, QKV_MPP_PAGE);
    return std.math.clamp(@divTrunc(pages, g[0]), 1, g[1]);
}

fn qkvMppMRows(key: QkvMppKey) c_int {
    return @divTrunc(key.gqa * key.tq + 7, 8) * 8;
}

fn qkvMppConfig(key: QkvMppKey, h_kv: c_int, nsplit: c_int) !mlx.mlx_fast_metal_kernel_config {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const rows: c_int = h_kv * key.gqa * key.tq;
    const ml_shape = [_]c_int{ rows, nsplit };
    const o_shape = [_]c_int{ rows, nsplit, key.dv };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, h_kv, nsplit));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", key.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(key.bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", key.dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", key.dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GQA", key.gqa));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TQ", key.tq));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "MPAD", qkvMppMRows(key)));
    return config;
}

/// Metal JIT-compiles at the first EVAL, so each template set runs once on a one-page,
/// one-kv-head problem of zeros before a forward depends on it. False = the set declines.
pub fn qkvMppProbe(kernel: mlx.mlx_fast_metal_kernel, key: QkvMppKey) bool {
    for (qkv_mpp_probed) |p| if (p) |e| if (std.meta.eql(e.key, key)) return e.ok;
    const ok = qkvMppRunProbe(kernel, key);
    for (&qkv_mpp_probed) |*p| if (p.* == null) {
        p.* = .{ .key = key, .ok = ok };
        break;
    };
    if (!ok) log.info("[kv-attn] matmul2d packed attention declined: probe failed (DK={d} DV={d} bits={d} gqa={d} Tq={d})\n", .{ key.dk, key.dv, key.bits, key.gqa, key.tq });
    return ok;
}

fn qkvMppRunProbe(kernel: mlx.mlx_fast_metal_kernel, key: QkvMppKey) bool {
    const s = mlx.gpuStream();
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const cfg = qkvMppConfig(key, 1, 1) catch return false;
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const gs: c_int = @intCast(key.gs);
    const shapes = [_][4]c_int{
        .{ 1, 1, QKV_MPP_PAGE, kv_quant.packedWords(key.dk, key.bits) }, .{ 1, 1, QKV_MPP_PAGE, @divExact(key.dk, gs) },
        .{ 1, 1, QKV_MPP_PAGE, @divExact(key.dk, gs) },                  .{ 1, 1, QKV_MPP_PAGE, kv_quant.packedWords(key.dv, key.bits) },
        .{ 1, 1, QKV_MPP_PAGE, @divExact(key.dv, gs) },  .{ 1, 1, QKV_MPP_PAGE, @divExact(key.dv, gs) },
    };
    var arrs: [9]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (arrs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    arrs[0] = mlx.mlx_array_new();
    if (mlx.mlx_zeros(&arrs[0], &[_]c_int{ qkvMppMRows(key), key.dk }, 2, key.dtype, s) != 0) return false;
    for (shapes, 0..) |sh, i| {
        arrs[i + 1] = mlx.mlx_array_new();
        const dt: mlx.mlx_dtype = if (i == 0 or i == 3) .uint32 else key.dtype;
        if (mlx.mlx_zeros(&arrs[i + 1], &sh, 4, dt, s) != 0) return false;
    }
    const one = [_]c_int{1};
    const scl_data = [_]f32{1.0};
    arrs[7] = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    const tk_data = [_]i32{QKV_MPP_PAGE};
    arrs[8] = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    const in_vec = mlx.mlx_vector_array_new_data(&arrs, arrs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, in_vec, cfg, s) != 0) return false;
    if (mlx.mlx_vector_array_size(outputs) != 3) return false;
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    if (mlx.mlx_vector_array_get(&o, outputs, 2) != 0) return false;
    if (mlx.mlx_array_eval(o) != 0) return false;
    return !mlx.errorPending();
}

pub fn qkvMppKernel() ?mlx.mlx_fast_metal_kernel {
    return qkvMppKernelFor(&qkv_mpp_kernel_cached, "sushi_qkv_mpp", QKV_MPP_KERNEL_SOURCE);
}

pub fn qkvMppRowsKernel() ?mlx.mlx_fast_metal_kernel {
    return qkvMppKernelFor(&qkv_mpp_rows_kernel_cached, "sushi_qkv_mpp_rows", QKV_MPP_ROWS_KERNEL_SOURCE);
}

fn qkvMppKernelFor(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, source: [*:0]const u8) ?mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "kq", "ksc", "kbi", "vq", "vsc", "vbi", "scl", "tk" };
    const output_names = [_][*:0]const u8{ "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kk = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, KV_UNPACK_HEADER ++ QSA_SCORE_KERNEL_HEADER, false, false);
    if (kk.ctx == null) return null;
    slot.* = kk;
    return kk;
}


/// Packed-KV attention through matmul2d, t_q 1..8, no mask or end-aligned causal. Null = declined.
pub fn qkvAttnMppKernel(s: mlx.mlx_stream, q_in: mlx.mlx_array, view: *const DenseKVView, scale: f32, mask_mode: []const u8) !?mlx.mlx_array {
    return qkvAttnMpp(s, q_in, view, scale, mask_mode, false);
}

/// `rows`: causal verify rows through `QKV_MPP_ROWS_KERNEL_SOURCE`, each row its own GQA block.
pub fn qkvAttnMpp(s: mlx.mlx_stream, q_in: mlx.mlx_array, view: *const DenseKVView, scale: f32, mask_mode: []const u8, rows: bool) !?mlx.mlx_array {
    if (!verifyQmmNaxAvailable()) return null;
    if (!view.has_quant_triple or view.group_size == 0) return null;
    if (!KVQuantConfig.supportedBits(view.bits)) return null;
    if (mlx.mlx_array_ndim(q_in) != 4) return null;
    const qs = mlx.getShape(q_in);
    const t_q: c_int = qs[2];
    if (qs[0] != 1 or t_q < 1 or t_q > QKV_VERIFY_MAX_TQ) return null;
    const causal = std.mem.eql(u8, mask_mode, "causal");
    if (!causal and !(t_q == 1 and mask_mode.len == 0)) return null;
    const qdt = mlx.mlx_array_dtype(q_in);
    if (qdt != .bfloat16 and qdt != .float16) return null;
    if (mlx.mlx_array_ndim(view.k_triple_q) != 4 or mlx.mlx_array_ndim(view.v_triple_q) != 4) return null;
    const ks = mlx.getShape(view.k_triple_q);
    const vs = mlx.getShape(view.v_triple_q);
    const dk: c_int = kv_quant.unpackedDim(ks[3], view.bits);
    const dv: c_int = kv_quant.unpackedDim(vs[3], view.bits);
    // Measured shapes only: the tile holds 32 rows of max(DK, DV), 16 KiB at 256.
    if (qs[3] != dk or !((dk == 256 and dv == 256) or (dk == 192 and dv == 128))) return null;
    const gs: c_int = @intCast(view.group_size);
    if (@rem(dk, gs) != 0 or @rem(dv, gs) != 0) return null;
    // The kernel loads packed words in units that never straddle a quant group: aligned pairs, or
    // word triples at 6 bits (sixteen values, read as plain words).
    const unit_words: c_int = if (view.bits == 6) 3 else 2;
    if (@rem(kv_quant.packedWords(gs, view.bits), unit_words) != 0) return null;
    for ([_]mlx.mlx_array{ view.k_triple_q, view.v_triple_q }) |a| {
        const st = mlx.mlx_array_strides(a);
        if (unit_words == 2 and (@rem(st[1], 2) != 0 or @rem(st[2], 2) != 0)) return null;
    }
    const h_q: c_int = qs[1];
    const h_kv: c_int = ks[1];
    if (h_kv <= 0 or @rem(h_q, h_kv) != 0 or vs[1] != h_kv or vs[2] != ks[2] or ks[0] != 1) return null;
    const gqa: c_int = @divExact(h_q, h_kv);
    if (gqa * t_q > 64) return null;
    const t_k: c_int = ks[2];
    if (t_k < t_q) return null;
    // The rows kernel's matmuls take one GQA block per row, so the block must be whole rows of 8, and it
    // names one running output per row, four at most.
    if (rows and !(causal and t_q >= 2 and t_q <= 4 and @rem(gqa, 8) == 0)) return null;

    const kernel = (if (rows) qkvMppRowsKernel() else qkvMppKernel()) orelse return null;
    const tkey = QkvMppKey{ .dk = dk, .dv = dv, .bits = view.bits, .gs = view.group_size, .gqa = gqa, .tq = t_q, .dtype = qdt, .rows = rows };
    if (!qkvMppProbe(kernel, tkey)) return null;
    const nsplit = qkvMppSplits(t_k);
    const real_rows: c_int = gqa * t_q;
    const m_rows = qkvMppMRows(tkey);
    const cfgs = &qkv_mpp_cfgs[@intFromBool(rows)];
    const cfg_keys = &qkv_mpp_cfg_keys[@intFromBool(rows)];
    const slot: usize = @intCast(t_q);
    const key = QkvMppCfgKey{ .t = tkey, .h_kv = h_kv, .nsplit = nsplit };
    if (cfgs[slot] == null or !std.meta.eql(cfg_keys[slot], key)) {
        const config = try qkvMppConfig(tkey, h_kv, nsplit);
        if (cfgs[slot]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        cfgs[slot] = config;
        cfg_keys[slot] = key;
    }

    // [1, Hq, TQ, D] is already (kv head, group head, row)-major: [Hkv, real_rows, D], zero-padded.
    // The rows kernel takes (kv head, row, group head) instead.
    var q3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q3);
    if (rows) {
        var q4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q4);
        try mlx.check(mlx.mlx_reshape(&q4, q_in, &[_]c_int{ h_kv, gqa, t_q, dk }, 4, s));
        var qt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, q4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        var qc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qc);
        try mlx.check(mlx.mlx_contiguous(&qc, qt, false, s));
        try mlx.check(mlx.mlx_reshape(&q3, qc, &[_]c_int{ h_kv, real_rows, dk }, 3, s));
    } else {
        try mlx.check(mlx.mlx_reshape(&q3, q_in, &[_]c_int{ h_kv, real_rows, dk }, 3, s));
    }
    var qp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qp);
    if (m_rows > real_rows) {
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        try mlx.check(mlx.mlx_pad(&qp, q3, &[_]c_int{1}, 1, &[_]c_int{0}, 1, &[_]c_int{m_rows - real_rows}, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&qp, q3));
    }
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_contiguous(&q, qp, false, s));
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const tk_data = [_]i32{t_k};
    const tk = mlx.mlx_array_new_data(&tk_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(tk);
    const inputs_arr = [_]mlx.mlx_array{ q, view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.v_triple_q, view.v_triple_scales, view.v_triple_biases, scl, tk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfgs[slot].?, s));
    const o_t = try qkvMergePartials(s, outputs_vec, qdt);
    defer _ = mlx.mlx_array_free(o_t);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (rows) {
        var o4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o4);
        try mlx.check(mlx.mlx_reshape(&o4, o_t, &[_]c_int{ h_kv, t_q, gqa, dv }, 4, s));
        var ot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ot);
        try mlx.check(mlx.mlx_transpose_axes(&ot, o4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try mlx.check(mlx.mlx_reshape(&out, ot, &[_]c_int{ 1, h_q, t_q, dv }, 4, s));
    } else {
        try mlx.check(mlx.mlx_reshape(&out, o_t, &[_]c_int{ 1, h_q, t_q, dv }, 4, s));
    }
    if (!qkv_mpp_engaged) {
        qkv_mpp_engaged = true;
        log.info("[kv-attn] matmul2d packed attention engaged: bits={d} gs={d} DK={d} DV={d} Hq={d} Hkv={d} Tq={d} Tk={d}\n", .{ view.bits, view.group_size, dk, dv, h_q, h_kv, t_q, t_k });
    }
    return out;
}

// ── Packed decode floors ──

/// Decode width for the matmul2d arm at MiMo's global shape: the floor measured on M5 Max (dense
/// dequant + SDPA still wins at 2k keys).
pub const QKV_MPP_DECODE_MIN_TK: c_int = 4096;

pub fn qkvMppDecodeServes(view: *const DenseKVView, t_q: c_int) bool {
    return packedDecodeServesFrom(view, t_q, QKV_MPP_DECODE_MIN_TK);
}

/// A packed cache read at decode width (t_q == 1) from `floor` keys; every other width reads the dense view.
pub fn packedDecodeServesFrom(view: *const DenseKVView, t_q: c_int, floor: c_int) bool {
    if (!(view.has_quant_triple and t_q == 1)) return false;
    const ks = mlx.getShape(view.k_triple_q);
    return ks.len >= 4 and ks[2] >= floor;
}

/// Decode floor for the split-K arm at MiMo's global shape where matrix units are absent.
pub const QKV_SPLITK_DECODE_MIN_TK: c_int = 4096;

/// The (splits, pages per split) a matmul2d read over `t_k` keys cuts its pages into.
pub fn qkvMppPartition(t_k: c_int) [2]c_int {
    const pages = @divTrunc(t_k + QKV_MPP_PAGE - 1, QKV_MPP_PAGE);
    const nsplit = qkvMppSplits(t_k);
    return .{ nsplit, @divTrunc(pages + nsplit - 1, nsplit) };
}


/// Whether a decode row (t_q 1) over `view` gets the matmul2d kernel, its template set probed.
pub fn qkvMppDecodeProbeOk(q: mlx.mlx_array, view: *const DenseKVView) bool {
    if (!KVQuantConfig.supportedBits(view.bits)) return false;
    const kernel = qkvMppKernel() orelse return false;
    const ks = mlx.getShape(view.k_triple_q);
    return qkvMppProbe(kernel, .{
        .dk = kv_quant.unpackedDim(ks[3], view.bits),
        .dv = kv_quant.unpackedDim(mlx.getShape(view.v_triple_q)[3], view.bits),
        .bits = view.bits,
        .gs = view.group_size,
        .gqa = @divTrunc(mlx.getShape(q)[1], ks[1]),
        .tq = 1,
        .dtype = mlx.mlx_array_dtype(q),
    });
}

/// fp32 ground truth over a packed cache: dequantize the triples in f32, then causal
/// (end-aligned) attention in f32. `[1, Hq, TQ, DV]` f32.
fn qkvTruthF32(s: mlx.mlx_stream, q: mlx.mlx_array, view: *const DenseKVView, scale: f32) !mlx.mlx_array {
    const deq = struct {
        fn f(st: mlx.mlx_stream, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, gs: u32, bits: u8) !mlx.mlx_array {
            var sc32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sc32);
            var bi32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bi32);
            try mlx.check(mlx.mlx_astype(&sc32, sc, .float32, st));
            try mlx.check(mlx.mlx_astype(&bi32, bi, .float32, st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_dequantize(&out, w, sc32, bi32, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, .{ .value = .float32, .has_value = true }, st));
            return out;
        }
    }.f;
    const k32 = try deq(s, view.k_triple_q, view.k_triple_scales, view.k_triple_biases, view.group_size, view.bits);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = try deq(s, view.v_triple_q, view.v_triple_scales, view.v_triple_biases, view.group_size, view.bits);
    defer _ = mlx.mlx_array_free(v32);
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k32);
    const h_kv = ks[1];
    const t_k = ks[2];
    const rows: c_int = @divExact(qs[1], h_kv) * qs[2];
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    try mlx.check(mlx.mlx_reshape(&q4, q32, &[_]c_int{ 1, h_kv, rows, qs[3] }, 4, s));
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k32, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_matmul(&sc, q4, kt, s));
    // Row m = g * TQ + tq sees keys j <= t_k - TQ + tq.
    const mask_host = try testing.allocator.alloc(f32, @intCast(rows * t_k));
    defer testing.allocator.free(mask_host);
    for (0..@intCast(rows)) |m| for (0..@intCast(t_k)) |j| {
        const lim: usize = @intCast(t_k - qs[2] + @as(c_int, @intCast(m % @as(usize, @intCast(qs[2])))));
        mask_host[m * @as(usize, @intCast(t_k)) + j] = if (j <= lim) @as(f32, 0) else -std.math.inf(f32);
    };
    const mask = mlx.mlx_array_new_data(mask_host.ptr, &[_]c_int{ rows, t_k }, 2, .float32);
    defer _ = mlx.mlx_array_free(mask);
    const scl = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scl);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, sc, scl, s));
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_add(&masked, scaled, mask, s));
    var p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p);
    try mlx.check(mlx.mlx_softmax_axis(&p, masked, -1, true, s));
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_matmul(&o, p, v32, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, o, &[_]c_int{ 1, qs[1], qs[2], mlx.getShape(v32)[3] }, 4, s));
    return out;
}

const TruthErr = struct { max: f32, rms: f32, scale: f32 };

/// Error of `cand` against the f32 truth; asserts finiteness before it diffs.
fn truthErr(s: mlx.mlx_stream, cand: mlx.mlx_array, truth: mlx.mlx_array) !TruthErr {
    var c_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_c);
    try mlx.check(mlx.mlx_contiguous(&c_c, cand, false, s));
    var c32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c32);
    try mlx.check(mlx.mlx_astype(&c32, c_c, .float32, s));
    var t_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_c);
    try mlx.check(mlx.mlx_contiguous(&t_c, truth, false, s));
    try mlx.check(mlx.mlx_array_eval(c32));
    try mlx.check(mlx.mlx_array_eval(t_c));
    const n = mlx.mlx_array_size(c32);
    if (n != mlx.mlx_array_size(t_c)) return error.ShapeMismatch;
    const cd = mlx.mlx_array_data_float32(c32) orelse return error.InvalidDtype;
    const td = mlx.mlx_array_data_float32(t_c) orelse return error.InvalidDtype;
    var e: TruthErr = .{ .max = 0, .rms = 0, .scale = 0 };
    var ss: f64 = 0;
    for (0..n) |i| {
        try testing.expect(std.math.isFinite(cd[i]));
        const d = @abs(cd[i] - td[i]);
        e.max = @max(e.max, d);
        e.scale = @max(e.scale, @abs(td[i]));
        ss += @as(f64, d) * d;
    }
    e.rms = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(n))));
    return e;
}

/// A packed-cache kernel (matmul2d or split-K) vs dense SDPA over the bf16 dequantized view, both
/// against the fp32 truth. The dense arm rounds K/V to bf16 exactly as both kernels' tiles do.
fn qkvPackedParityCase(comptime cand_fn: anytype, rnd: std.Random, h_q: c_int, h_kv: c_int, dk: c_int, dv: c_int, bits: u8, t_prefill: c_int, t_q: c_int) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
    defer cache.deinit();
    {
        const k = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, t_prefill, dk }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, t_prefill, dv }, s);
        defer _ = mlx.mlx_array_free(v);
        var dv0 = try cache.update(0, k, v, s, 0);
        dv0.deinit();
    }
    // A second append leaves capacity slack: the views the kernel reads are strided.
    const k1 = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, t_q, dk }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, t_q, dv }, s);
    defer _ = mlx.mlx_array_free(v1);
    var view = try cache.update(0, k1, v1, s, 0);
    defer view.deinit();
    try testing.expect(view.has_quant_triple);
    const q = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, h_q, t_q, dk }, 8.0, s);
    defer _ = mlx.mlx_array_free(q);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(dk)));
    const mode: [:0]const u8 = if (t_q == 1) "" else "causal";
    const none = mlx.mlx_array{ .ctx = null };

    const truth = try qkvTruthF32(s, q, &view, scale);
    defer _ = mlx.mlx_array_free(truth);
    const cand = (try cand_fn(s, q, &view, scale, mode)) orelse return error.KernelDeclined;
    defer _ = mlx.mlx_array_free(cand);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, h_q, t_q, dv }, mlx.getShape(cand));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&dense, q, view.k, view.v, scale, mode, none, .{ .ctx = null }, false, s));

    const ce = try truthErr(s, cand, truth);
    const de = try truthErr(s, dense, truth);
    // One bf16 ulp at the output's scale covers the final rounding landing on the other side.
    const ulp = ce.scale * 0.00390625;
    const pass = ce.max <= de.max + ulp and ce.rms <= de.rms * 1.25 + 1e-7;
    if (!pass or diagEnvOn("SUSHI_KVQ_PARITY_TRACE")) {
        std.debug.print("[qkv-packed parity] Hq={d} Hkv={d} DK={d} DV={d} bits={d} Tk={d} Tq={d}: kernel max {e:.3} rms {e:.3} | dense max {e:.3} rms {e:.3}\n", .{ h_q, h_kv, dk, dv, bits, t_prefill + t_q, t_q, ce.max, ce.rms, de.max, de.rms });
    }
    try testing.expect(pass);
}

const KvqUbArm = union(enum) {
    empty,
    /// SDPA over the view's dense arrays, which the view builds once.
    dense,
    /// Dequantize the triples on every call, then SDPA: what a decode step does.
    dense_rebuild,
    /// matmul2d at (pages per split, split cap).
    mpp: [2]c_int,
    /// split-K at (keys per split, split floor, split cap).
    split_k: [3]c_int,
};

/// MiMo global decode (64/4 heads, qk 192 / v 128) over nine dependent layers, each over its OWN
/// cache so small contexts cannot hide in the SLC; arms interleave per rep, `arms[0]` is the empty
/// chain. us/call = (chain - empty chain) / 9.
fn kvqDecodeUbench(comptime tag: []const u8, arms: []const KvqUbArm) !void {
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var prng = std.Random.DefaultPrng.init(0x3D2D_BE7C);
    const rnd = prng.random();
    const h_q: c_int = 64;
    const h_kv: c_int = 4;
    const layers: usize = 9;
    const reps: usize = 7;
    const max_arms = 8;
    const scale: f32 = 1.0 / @sqrt(192.0);
    const none = mlx.mlx_array{ .ctx = null };
    const bits: u8 = if (diagEnvOn("SUSHI_KVQ_MPP_UBENCH_KV4")) 4 else 8;
    defer qkv_mpp_split_override = null;
    defer qkv_splitk_split_override = null;

    const Next = struct {
        // [1,64,1,128] -> [1,64,1,192]: the next layer's query depends on this layer's output.
        fn f(st: mlx.mlx_stream, out: mlx.mlx_array) !mlx.mlx_array {
            var head = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(head);
            try mlx.check(mlx.mlx_slice(&head, out, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 64, 1, 64 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, st));
            const parts = [_]mlx.mlx_array{ out, head };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&q, vec, 3, st));
            return q;
        }
    };

    std.debug.print("\n[" ++ tag ++ "] MiMo global decode 64/4 qk192/v128 kv{d} gs64, {d} dependent layers, median of {d}\n", .{ bits, layers, reps });
    std.debug.print("[" ++ tag ++ "]      kv ", .{});
    for (arms[1..]) |arm| switch (arm) {
        .empty => {},
        .dense => std.debug.print("|   dense_us  GB/s ", .{}),
        .dense_rebuild => std.debug.print("| rebuild_us  GB/s ", .{}),
        .mpp => |g| std.debug.print("| mpp{d}/{d}_us  GB/s ", .{ g[0], g[1] }),
        .split_k => |g| std.debug.print("| sk{d}/{d}/{d}_us  GB/s ", .{ g[0], g[1], g[2] }),
    };
    std.debug.print("\n", .{});
    for ([_]c_int{ 2048, 4096, 8192, 16384, 65536, 262144, 524288 }) |kv| {
        var caches: [layers]KVCache = undefined;
        var views: [layers]DenseKVView = undefined;
        var built: usize = 0;
        defer for (0..built) |i| {
            views[i].deinit();
            caches[i].deinit();
        };
        for (0..layers) |i| {
            caches[i] = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(bits));
            {
                const k = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, kv - 1, 192 }, s);
                defer _ = mlx.mlx_array_free(k);
                const v = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, kv - 1, 128 }, s);
                defer _ = mlx.mlx_array_free(v);
                var d0 = try caches[i].update(0, k, v, s, 0);
                d0.deinit();
            }
            const k1 = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, 1, 192 }, s);
            defer _ = mlx.mlx_array_free(k1);
            const v1 = try testRandWeightBf16(rnd, &[_]c_int{ 1, h_kv, 1, 128 }, s);
            defer _ = mlx.mlx_array_free(v1);
            views[i] = try caches[i].update(0, k1, v1, s, 0);
            built += 1;
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for ([_]mlx.mlx_array{ views[i].k_triple_q, views[i].k_triple_scales, views[i].k_triple_biases, views[i].v_triple_q, views[i].v_triple_scales, views[i].v_triple_biases }) |a| _ = mlx.mlx_vector_array_append_value(ev, a);
            try mlx.check(mlx.mlx_eval(ev));
        }
        var bytes: f64 = 0;
        for ([_]mlx.mlx_array{ views[0].k_triple_q, views[0].k_triple_scales, views[0].k_triple_biases, views[0].v_triple_q, views[0].v_triple_scales, views[0].v_triple_biases }) |a|
            bytes += @floatFromInt(mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a));
        const q0 = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, h_q, 1, 192 }, 8.0, s);
        defer _ = mlx.mlx_array_free(q0);

        var samples: [max_arms][reps]f64 = undefined;
        for (0..reps + 2) |r| {
            for (arms, 0..) |arm, ai| {
                const t0 = std.Io.Timestamp.now(io, .boot);
                var q = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&q, q0));
                for (0..layers) |l| {
                    var out = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(out);
                    switch (arm) {
                        .empty => try mlx.check(mlx.mlx_slice(&out, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, 64, 1, 128 }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s)),
                        .dense => try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, views[l].k, views[l].v, scale, "", none, .{ .ctx = null }, false, s)),
                        .dense_rebuild => {
                            const kd = try kv_quant.dequantizeAffine(s, views[l].k_triple_q, views[l].k_triple_scales, views[l].k_triple_biases, views[l].group_size, bits);
                            defer _ = mlx.mlx_array_free(kd);
                            const vd = try kv_quant.dequantizeAffine(s, views[l].v_triple_q, views[l].v_triple_scales, views[l].v_triple_biases, views[l].group_size, bits);
                            defer _ = mlx.mlx_array_free(vd);
                            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kd, vd, scale, "", none, .{ .ctx = null }, false, s));
                        },
                        .mpp => |g| {
                            qkv_mpp_split_override = g;
                            _ = mlx.mlx_array_free(out);
                            out = (try qkvAttnMppKernel(s, q, &views[l], scale, "")) orelse return error.KernelDeclined;
                        },
                        .split_k => |g| {
                            qkv_splitk_split_override = g;
                            _ = mlx.mlx_array_free(out);
                            out = (try qkvAttnSplitKKernel(s, q, &views[l], scale, "")) orelse return error.KernelDeclined;
                        },
                    }
                    const nq = try Next.f(s, out);
                    _ = mlx.mlx_array_free(q);
                    q = nq;
                }
                try mlx.check(mlx.mlx_array_eval(q));
                _ = mlx.mlx_array_free(q);
                if (r >= 2) samples[ai][r - 2] = @as(f64, @floatFromInt(t0.untilNow(io, .boot).nanoseconds)) / 1e3;
            }
        }
        var med: [max_arms]f64 = undefined;
        for (0..arms.len) |a| {
            std.mem.sort(f64, &samples[a], {}, std.sort.asc(f64));
            med[a] = samples[a][reps / 2];
        }
        std.debug.print("[" ++ tag ++ "] {d:>7} ", .{kv});
        for (1..arms.len) |a| {
            const us = @max(0.001, (med[a] - med[0]) / @as(f64, @floatFromInt(layers)));
            std.debug.print("| {d:>10.1}  {d:>5.0} ", .{ us, bytes / us / 1e3 });
        }
        std.debug.print("\n", .{});
        _ = mlx.mlx_clear_cache();
    }
}

/// Serial decode ticks against one verify forward over the same rows, on twin caches.
fn mimoVerifyRowsIdentityCase(config: KVQuantConfig, is_global: bool, prefix: c_int, width: c_int) !void {
    return mimoVerifyRowsIdentityCaseHeads(config, is_global, prefix, width, 16, if (is_global) 1 else 2);
}

fn mimoVerifyRowsIdentityCaseHeads(config: KVQuantConfig, is_global: bool, prefix: c_int, width: c_int, hq: c_int, hkv: c_int) !void {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const window: c_int = 128;
    const scale: f32 = 1.0 / @sqrt(192.0);
    var caches: [2]KVCache = undefined;
    for (&caches) |*c| {
        c.* = try KVCache.initWithConfig(a, 1, config);
        if (!is_global) c.setSwaRing(@intCast(window));
    }
    defer for (&caches) |*c| c.deinit();
    const span = struct {
        fn of(global: bool, w: c_int, rows: c_int) u32 {
            return if (global) 0 else slidingTailSpan(@intCast(w), @intCast(rows), SLIDING_TRIM_UNBOUNDED);
        }
    };
    const k0 = try qkvIdentityArray(s, hkv, prefix + width, 192, 0.7, .bfloat16);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try qkvIdentityArray(s, hkv, prefix + width, 128, 2.1, .bfloat16);
    defer _ = mlx.mlx_array_free(v0);
    const q = try qkvIdentityArray(s, hq, width, 192, 1.9, .bfloat16);
    defer _ = mlx.mlx_array_free(q);
    const sinks = if (is_global) mlx.mlx_array{ .ctx = null } else try qkvSinks(s, hq);
    defer if (sinks.ctx != null) {
        _ = mlx.mlx_array_free(sinks);
    };
    const mask = try slidingDecodeMask(s, window, window);
    defer _ = mlx.mlx_array_free(mask);
    for (&caches) |*c| {
        const kp = try sliceAttentionSeq(s, k0, 0, prefix);
        defer _ = mlx.mlx_array_free(kp);
        const vp = try sliceAttentionSeq(s, v0, 0, prefix);
        defer _ = mlx.mlx_array_free(vp);
        var view = try c.update(0, kp, vp, s, span.of(is_global, window, prefix));
        view.deinit();
    }
    const kw = try sliceAttentionSeq(s, k0, prefix, prefix + width);
    defer _ = mlx.mlx_array_free(kw);
    const vw = try sliceAttentionSeq(s, v0, prefix, prefix + width);
    defer _ = mlx.mlx_array_free(vw);
    var verify_view = try caches[1].update(0, kw, vw, s, span.of(is_global, window, width));
    defer verify_view.deinit();
    const verify = try mimoVerifyRowsAttn(s, window, q, &verify_view, sinks, is_global, prefix, width, scale, mask);
    defer _ = mlx.mlx_array_free(verify);
    for (0..@intCast(width)) |ri| {
        const r: c_int = @intCast(ri);
        const kr = try sliceAttentionSeq(s, k0, prefix + r, prefix + r + 1);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try sliceAttentionSeq(s, v0, prefix + r, prefix + r + 1);
        defer _ = mlx.mlx_array_free(vr);
        var tick_view = try caches[0].update(0, kr, vr, s, span.of(is_global, window, 1));
        defer tick_view.deinit();
        const qr = try sliceAttentionSeq(s, q, r, r + 1);
        defer _ = mlx.mlx_array_free(qr);
        const tick = try mimoDecodeAttn(s, window, qr, &tick_view, sinks, is_global, prefix + r + 1, scale, mask);
        defer _ = mlx.mlx_array_free(tick);
        const row = try sliceAttentionSeq(s, verify, r, r + 1);
        defer _ = mlx.mlx_array_free(row);
        qkvExpectBitsEqual(s, tick, row) catch |err| {
            std.debug.print("mimo verify row {d}: global={} prefix={d} width={d} kv={s}\n", .{ r, is_global, prefix, width, @tagName(config.scheme) });
            return err;
        };
    }
}

fn qkvSinks(s: mlx.mlx_stream, heads: c_int) !mlx.mlx_array {
    var buf: [64]f32 = undefined;
    for (buf[0..@intCast(heads)], 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.37) * 2.0;
    const raw = mlx.mlx_array_new_data(&buf, &[_]c_int{heads}, 1, .float32);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    return out;
}

/// One dispatch for a verify's sliding rows against the per-row sdpa it replaces, on the served
/// head geometry and a ring cache.
fn mimoSlidingRowsCase(config: KVQuantConfig, prefix: c_int, width: c_int, gain: f32) !void {
    const s = mlx.gpuStream();
    const window: c_int = 128;
    const hq: c_int = 64;
    const hkv: c_int = 8;
    const scale: f32 = 1.0 / @sqrt(192.0);
    var cache = try KVCache.initWithConfig(testing.allocator, 1, config);
    defer cache.deinit();
    cache.setSwaRing(@intCast(window));
    const g = bf16Scalar(gain, s);
    defer _ = mlx.mlx_array_free(g);
    var arrays: [3]mlx.mlx_array = @splat(.{});
    defer for (arrays) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&arrays, [_]c_int{ hkv, hkv, hq }, [_]c_int{ prefix + width, prefix + width, width }, [_]c_int{ 192, 128, 192 }, [_]f32{ 0.7, 2.1, 1.9 }) |*out, heads, rows, dim, seed| {
        const raw = try qkvIdentityArray(s, heads, rows, dim, seed, .bfloat16);
        defer _ = mlx.mlx_array_free(raw);
        out.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(out, raw, g, s));
    }
    const sinks = try qkvSinks(s, hq);
    defer _ = mlx.mlx_array_free(sinks);
    const mask = try slidingDecodeMask(s, window, window);
    defer _ = mlx.mlx_array_free(mask);
    {
        const kp = try sliceAttentionSeq(s, arrays[0], 0, prefix);
        defer _ = mlx.mlx_array_free(kp);
        const vp = try sliceAttentionSeq(s, arrays[1], 0, prefix);
        defer _ = mlx.mlx_array_free(vp);
        var view = try cache.update(0, kp, vp, s, slidingTailSpan(@intCast(window), @intCast(prefix), SLIDING_TRIM_UNBOUNDED));
        view.deinit();
    }
    const kw = try sliceAttentionSeq(s, arrays[0], prefix, prefix + width);
    defer _ = mlx.mlx_array_free(kw);
    const vw = try sliceAttentionSeq(s, arrays[1], prefix, prefix + width);
    defer _ = mlx.mlx_array_free(vw);
    var view = try cache.update(0, kw, vw, s, slidingTailSpan(@intCast(window), @intCast(width), SLIDING_TRIM_UNBOUNDED));
    defer view.deinit();
    const one = (try mimoSlidingRowsAttn(s, window, arrays[2], &view, sinks, prefix, width, scale, mask)) orelse return error.SlidingRowsDeclined;
    defer _ = mlx.mlx_array_free(one);
    const per_row = try mimoVerifyRowsAttnPerRow(s, window, arrays[2], &view, sinks, false, prefix, width, scale, mask);
    defer _ = mlx.mlx_array_free(per_row);
    qkvExpectBitsEqual(s, one, per_row) catch |err| {
        std.debug.print("mimo sliding rows: prefix={d} width={d} gain={d} kv={s}\n", .{ prefix, width, gain, @tagName(config.scheme) });
        return err;
    };
}

fn testExpectSameBits(want: mlx.mlx_array, got: mlx.mlx_array, s: mlx.mlx_stream) !void {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    const n = mlx.mlx_array_size(want);
    const a = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(a);
    const b = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(b);
    for ([_]mlx.mlx_array{ want, got }, [_][]u32{ a, b }) |arr, out| {
        if (mlx.mlx_array_dtype(arr) == .uint32) {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            _ = mlx.mlx_vector_array_append_value(ev, arr);
            try mlx.check(mlx.mlx_eval(ev));
            const data = mlx.mlx_array_data_uint32(arr) orelse return error.TestUnexpectedNullData;
            @memcpy(out, data[0..n]);
        } else {
            // bf16 -> f32 is exact; comparing the f32 bits also tells -0 from +0.
            try testReadF32(arr, @ptrCast(out), s);
        }
    }
    try testing.expectEqualSlices(u32, a, b);
}

/// The HF oracle keeps separate experts; the serving contract uses leading-index banks.
pub fn stackMimoFixtureExperts(weights: *Weights, config: ModelConfig, s: mlx.mlx_stream) !void {
    // A served pack's experts are EXL3 banks the loader already serves; only the fixture's per-expert rows stack.
    if (config.expert_layout == .exl3_k4) return;
    const a = weights.allocator;
    for (config.first_k_dense_replace..config.num_hidden_layers) |li| {
        for ([_][]const u8{ "gate", "up", "down" }) |projection| {
            const key = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.{s}_proj.weight", .{ li, projection });
            errdefer a.free(key);
            if (weights.get(key) != null) {
                a.free(key);
                continue;
            }
            const arrays = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(arrays);
            for (0..config.num_experts) |ei| {
                var buf: [256]u8 = undefined;
                const source = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.experts.{d}.{s}_proj.weight", .{ li, ei, projection });
                try mlx.check(mlx.mlx_vector_array_append_value(arrays, weights.get(source) orelse return error.MissingFixtureTensor));
            }
            var bank = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(bank);
            try mlx.check(mlx.mlx_stack_axis(&bank, arrays, 0, s));
            try weights.map.put(key, bank);
        }
    }
}

/// Group-limited routing mask (DeepSeek-style `group_limited_topk`): split the BIASED scores
/// into `n_group` equal groups, score each group by the SUM OF ITS TOP TWO
/// experts, keep the best `topk_group` groups, and drop every expert outside
/// them before the final top-k.
///
/// Returns the biased scores with the losing groups pushed far below any
/// reachable value. The reference writes -inf; we subtract a large FINITE
/// constant instead, which selects identically (biased scores are
/// sigmoid+bias, so bounded well inside +/-2) and keeps a non-finite value
/// out of a tensor that later feeds argpartition. Non-finite sentinels are
/// how the composed causal-mask arm shipped an all-NaN path that scored a
/// perfect parity max_err of 0.
fn groupLimitMask(
    biased: mlx.mlx_array,
    n_group: c_int,
    topk_group: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const bsh = mlx.getShape(biased);
    const num_experts = bsh[bsh.len - 1];
    if (n_group <= 0 or topk_group <= 0 or topk_group > n_group) return error.BadShape;
    if (@rem(num_experts, n_group) != 0) return error.BadShape;
    const group_size = @divExact(num_experts, n_group);
    if (group_size < 2) return error.BadShape; // the group score is a top-2 sum

    var rows: c_int = 1;
    for (bsh[0 .. bsh.len - 1]) |d| rows *= d;

    // [rows, n_group, group_size]
    const grouped_shape = [_]c_int{ rows, n_group, group_size };
    var grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grouped);
    try mlx.check(mlx.mlx_reshape(&grouped, biased, &grouped_shape, 3, s));

    // Group score = sum of the top 2 within the group. mlx_topk_axis, NOT
    // mlx_topk: the latter is the FLATTENED reduction and silently collapses
    // every group into one, which is an uncatchable shape error downstream.
    var top2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top2);
    try mlx.check(mlx.mlx_topk_axis(&top2, grouped, 2, -1, s));
    var gscore = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gscore);
    try mlx.check(mlx.mlx_sum_axis(&gscore, top2, -1, false, s)); // [rows, n_group]

    // Indices of the best `topk_group` groups.
    var gneg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gneg);
    try mlx.check(mlx.mlx_negative(&gneg, gscore, s));
    var gpart = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gpart);
    try mlx.check(mlx.mlx_argpartition_axis(&gpart, gneg, topk_group - 1, -1, s));
    var gidx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gidx);
    {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ rows, topk_group };
        const stride = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&gidx, gpart, &start, 2, &stop, 2, &stride, 2, s));
    }

    // keep[rows, n_group] = 1 for a surviving group.
    const keep_shape = [_]c_int{ rows, n_group };
    var keep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep);
    try mlx.check(mlx.mlx_zeros(&keep, &keep_shape, 2, .float32, s));
    const ones_shape = [_]c_int{ rows, topk_group };
    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &ones_shape, 2, .float32, s));
    var keep_set = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_set);
    try mlx.check(mlx.mlx_put_along_axis(&keep_set, keep, gidx, ones, -1, s));

    // Broadcast a group's decision across its experts: [rows, G, 1] -> [rows, G, S] -> [rows, E].
    const keep3_shape = [_]c_int{ rows, n_group, 1 };
    var keep3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep3);
    try mlx.check(mlx.mlx_reshape(&keep3, keep_set, &keep3_shape, 3, s));
    var keep_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_b);
    try mlx.check(mlx.mlx_broadcast_to(&keep_b, keep3, &grouped_shape, 3, s));
    const flat_shape = [_]c_int{ rows, num_experts };
    var keep_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keep_flat);
    try mlx.check(mlx.mlx_reshape(&keep_flat, keep_b, &flat_shape, 2, s));

    var biased_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased_flat);
    try mlx.check(mlx.mlx_reshape(&biased_flat, biased, &flat_shape, 2, s));

    const penalty = mlx.mlx_array_new_float(-1.0e4);
    defer _ = mlx.mlx_array_free(penalty);
    var pen_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pen_full);
    try mlx.check(mlx.mlx_full(&pen_full, &flat_shape, 2, penalty, .float32, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var cond = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cond);
    try mlx.check(mlx.mlx_greater(&cond, keep_flat, zero, s));

    var masked_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_flat);
    try mlx.check(mlx.mlx_where(&masked_flat, cond, biased_flat, pen_full, s));

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, masked_flat, bsh.ptr, bsh.len, s));
    return out;
}

/// Group-limited routing = the sigmoid+bias chain with the group limit inserted
/// between the bias and the top-k. Weights still come from the UNBIASED
/// sigmoid scores, sum-normalized with the reference's 1e-20 and scaled by
/// `routed_scaling_factor`.
pub fn groupLimitedRouting(
    router_logits: mlx.mlx_array,
    expert_bias: mlx.mlx_array,
    k: c_int,
    n_group: c_int,
    topk_group: c_int,
    route_norm: bool,
    route_scale: f32,
    s: mlx.mlx_stream,
) !Transformer.MoeRouting {
    // One dispatch instead of ~28. Weights stay f32 for the reason the comment
    // at the tail of this function gives, so the fused kernel is asked for f32
    // too. A decline (geometry outside the kernel's set, kill switch) falls
    // through to the chain below, which stays the definition of correct.
    if (try moeRouterTopK(
        s,
        router_logits,
        expert_bias,
        k,
        .sigmoid_bias_grouped,
        route_norm,
        route_scale,
        .float32,
        n_group,
        topk_group,
    )) |fused| return fused;

    // Router runs in fp32 (`router_dtype: "fp32"`).
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits_f32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, scores, expert_bias, s));

    const limited = try groupLimitMask(biased, n_group, topk_group, s);
    defer _ = mlx.mlx_array_free(limited);

    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, limited, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_take_along_axis(&top, scores, inds, -1, s));

    var weights_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights_f32);
    if (route_norm and k > 1) {
        var sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&sum_raw, top, -1, true, s));
        const eps = mlx.mlx_array_new_float(1e-20);
        defer _ = mlx.mlx_array_free(eps);
        var sum_eps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_eps);
        try mlx.check(mlx.mlx_add(&sum_eps, sum_raw, eps, s));
        try mlx.check(mlx.mlx_divide(&weights_f32, top, sum_eps, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&weights_f32, top));
    }

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (route_scale != 1.0) {
        const scale_arr = mlx.mlx_array_new_float(route_scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        try mlx.check(mlx.mlx_multiply(&scaled, weights_f32, scale_arr, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&scaled, weights_f32));
    }

    // Weights stay f32: a group-limited arch multiplies these into the
    // routed expert sum AFTER `routed_scaling_factor` (2.5), so a bf16 round trip lands ~0.4%
    // of 2.5 on every token -- measured 2.6e-3 max error on the reference MoE
    // block, which is 500x the tolerance the rest of the port holds. The caller
    // casts to the activation dtype.
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_array_set(&norm_scores, scaled));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

pub fn sigmoidBiasRoutingChain(
    router_logits: mlx.mlx_array,
    expert_bias: mlx.mlx_array,
    k: c_int,
    route_norm: bool,
    route_scale: f32,
    output_dtype: mlx.mlx_dtype,
    s: mlx.mlx_stream,
) !Transformer.MoeRouting {
    var logits_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits_f32);
    try mlx.check(mlx.mlx_astype(&logits_f32, router_logits, .float32, s));
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_sigmoid(&scores, logits_f32, s));

    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_add(&biased, scores, expert_bias, s));

    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, biased, s));
    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg, k - 1, -1, s));

    const p_shape = mlx.getShape(partitioned);
    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    {
        var start_arr: [4]c_int = undefined;
        var stop_arr: [4]c_int = undefined;
        var strides_arr: [4]c_int = undefined;
        for (0..p_shape.len) |d| {
            start_arr[d] = 0;
            stop_arr[d] = if (d == p_shape.len - 1) k else p_shape[d];
            strides_arr[d] = 1;
        }
        try mlx.check(mlx.mlx_slice(&inds, partitioned, &start_arr, p_shape.len, &stop_arr, p_shape.len, &strides_arr, p_shape.len, s));
    }

    // Weights = ORIGINAL (unbiased) sigmoid scores at the selected indices.
    var top = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top);
    try mlx.check(mlx.mlx_take_along_axis(&top, scores, inds, -1, s));

    var weights_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weights_f32);
    if (route_norm and k > 1) {
        var sum_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_raw);
        try mlx.check(mlx.mlx_sum_axis(&sum_raw, top, -1, true, s));
        const eps = mlx.mlx_array_new_float(1e-20);
        defer _ = mlx.mlx_array_free(eps);
        var sum_eps = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum_eps);
        try mlx.check(mlx.mlx_add(&sum_eps, sum_raw, eps, s));
        try mlx.check(mlx.mlx_divide(&weights_f32, top, sum_eps, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&weights_f32, top));
    }

    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    if (route_scale != 1.0) {
        const scale_arr = mlx.mlx_array_new_float(route_scale);
        defer _ = mlx.mlx_array_free(scale_arr);
        try mlx.check(mlx.mlx_multiply(&scaled, weights_f32, scale_arr, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&scaled, weights_f32));
    }

    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_astype(&norm_scores, scaled, output_dtype, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}

const ADD_RMSNORM_SOURCE =
    \\constexpr int SIMD_SIZE = 32;
    \\constexpr int NR = 4;              // RMS_N_READS
    \\uint gid = thread_position_in_grid.y;
    \\uint lid = thread_position_in_threadgroup.x;
    \\
    \\threadgroup float local_inv_mean[1];
    \\threadgroup float local_sums[SIMD_SIZE];
    \\
    \\size_t base = (size_t)gid * (size_t)AXIS + (size_t)lid * NR;
    \\float acc = 0;
    \\float thread_x[NR];
    \\if (int(lid) * NR + NR <= AXIS) {
    \\  for (int i = 0; i < NR; i++) {
    \\    T s = a[base + i] + b[base + i];
    \\    sum[base + i] = s;
    \\    thread_x[i] = float(s);
    \\    acc += thread_x[i] * thread_x[i];
    \\  }
    \\} else {
    \\  for (int i = 0; i < NR; i++) {
    \\    if (int(lid) * NR + i < AXIS) {
    \\      T s = a[base + i] + b[base + i];
    \\      sum[base + i] = s;
    \\      thread_x[i] = float(s);
    \\    } else {
    \\      thread_x[i] = 0;
    \\    }
    \\    acc += thread_x[i] * thread_x[i];
    \\  }
    \\}
    \\acc = simd_sum(acc);
    \\if (simdgroup_index_in_threadgroup == 0) {
    \\  local_sums[thread_index_in_simdgroup] = 0;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (thread_index_in_simdgroup == 0) {
    \\  local_sums[simdgroup_index_in_threadgroup] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simdgroup_index_in_threadgroup == 0) {
    \\  acc = simd_sum(local_sums[thread_index_in_simdgroup]);
    \\  if (thread_index_in_simdgroup == 0) {
    \\    local_inv_mean[0] = metal::precise::rsqrt(acc / AXIS + eps);
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\if (int(lid) * NR + NR <= AXIS) {
    \\  for (int i = 0; i < NR; i++) {
    \\    SUSHI_NORM_STORE(base + i, w[lid * NR + i] * T(thread_x[i] * local_inv_mean[0]));
    \\  }
    \\} else {
    \\  for (int i = 0; i < NR; i++) {
    \\    if (int(lid) * NR + i < AXIS) {
    \\      SUSHI_NORM_STORE(base + i, w[lid * NR + i] * T(thread_x[i] * local_inv_mean[0]));
    \\    }
    \\  }
    \\}
;

/// Index 1 also stores the normed row widened to f32 (`out32`), exact from any T.
const ADD_RMSNORM_STORE = [2][:0]const u8{
    "#define SUSHI_NORM_STORE(j, v) out[j] = v;\n",
    "#define SUSHI_NORM_STORE(j, v) { T o_ = v; out[j] = o_; out32[j] = float(o_); }\n",
};
var add_rmsnorm_kernel: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var add_rmsnorm_engaged: bool = false;

fn getAddRmsNormKernel(f32_copy: bool) !mlx.mlx_fast_metal_kernel {
    const v = @intFromBool(f32_copy);
    if (add_rmsnorm_kernel[v]) |k| return k;
    const input_names = [_][*:0]const u8{ "a", "b", "w", "eps" };
    const output_names = [_][*:0]const u8{ "sum", "out", "out32" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, 2 + @as(usize, v));
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        if (f32_copy) "sushi_add_rmsnorm_f32" else "sushi_add_rmsnorm",
        in_vec,
        out_vec,
        ADD_RMSNORM_SOURCE,
        ADD_RMSNORM_STORE[v],
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    add_rmsnorm_kernel[v] = kernel;
    return kernel;
}

const AddNormCfgKey = struct { shape: ShapeKey, dtype: mlx.mlx_dtype };
var add_rmsnorm_cfg: [2]?mlx.mlx_fast_metal_kernel_config = .{ null, null };
var add_rmsnorm_cfg_key: [2]AddNormCfgKey = @splat(std.mem.zeroes(AddNormCfgKey));

pub const AddNormResult = struct { sum: mlx.mlx_array, normed: mlx.mlx_array, normed_f32: mlx.mlx_array = .{ .ctx = null } };

/// `sum = a + b` and `normed = rms_norm(sum, w, eps)` in one dispatch, or null
/// where the shape is one the kernel cannot reproduce (the caller keeps the two
/// ops). Bit-identical to the composed pair; `eps_arr` must be a 0-dim f32 scalar.
pub fn fusedAddRmsNormUngated(
    s: mlx.mlx_stream,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    w: mlx.mlx_array,
    eps_arr: mlx.mlx_array,
) !?AddNormResult {
    return addRmsNormKernel(s, a, b, w, eps_arr, false);
}

/// `fusedAddRmsNormUngated` plus `normed_f32`, the normed row as `astype(f32)`
/// would widen it: MiMo's f32 router input without its own dispatch.
pub fn fusedAddRmsNormRouted(
    s: mlx.mlx_stream,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    w: mlx.mlx_array,
    eps_arr: mlx.mlx_array,
) !?AddNormResult {
    return addRmsNormKernel(s, a, b, w, eps_arr, true);
}

fn addRmsNormKernel(
    s: mlx.mlx_stream,
    a: mlx.mlx_array,
    b: mlx.mlx_array,
    w: mlx.mlx_array,
    eps_arr: mlx.mlx_array,
    f32_copy: bool,
) !?AddNormResult {
    // A Transformer built by a construction site that predates `rms_eps_arr`
    // would hand us a null handle; decline rather than pass it to mlx.
    if (w.ctx == null or eps_arr.ctx == null) return null;
    const dt = mlx.mlx_array_dtype(a);
    if (dt != mlx.mlx_array_dtype(b) or dt != mlx.mlx_array_dtype(w)) return null;
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return null;
    const ash = mlx.getShape(a);
    const bsh = mlx.getShape(b);
    if (ash.len != bsh.len or ash.len == 0 or ash.len > 4) return null;
    for (ash, bsh) |x, y| if (x != y) return null; // no broadcasting
    const axis = ash[ash.len - 1];
    // Above RMS_LOOPED_LIMIT mlx switches to a different kernel with a
    // different reduction shape; decline rather than guess.
    if (axis <= 0 or axis > 4096) return null;
    const wsh = mlx.getShape(w);
    if (wsh.len != 1 or wsh[0] != axis) return null;

    var rows: c_int = 1;
    for (ash[0 .. ash.len - 1]) |d| rows *= d;
    if (rows < 1) return null;

    // MLX's own launch: ceil(axis / N_READS) threads, rounded up to simdgroups.
    const tg: c_int = @divTrunc(@divTrunc(axis + 3, 4) + 31, 32) * 32;
    if (tg > 1024) return null;

    const v = @intFromBool(f32_copy);
    const key = AddNormCfgKey{ .shape = ShapeKey.from(ash), .dtype = dt };
    if (add_rmsnorm_cfg[v] == null or !std.meta.eql(add_rmsnorm_cfg_key[v], key)) {
        if (add_rmsnorm_cfg[v]) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        add_rmsnorm_cfg[v] = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ash.ptr, ash.len, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ash.ptr, ash.len, dt));
        if (f32_copy) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, ash.ptr, ash.len, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "AXIS", axis));
        add_rmsnorm_cfg[v] = config;
        add_rmsnorm_cfg_key[v] = key;
    }

    const inputs_arr = [_]mlx.mlx_array{ a, b, w, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getAddRmsNormKernel(f32_copy);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, add_rmsnorm_cfg[v].?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2 + @as(usize, v)) return error.MetalKernelBadOutputCount;
    var sum = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_vector_array_get(&sum, outputs_vec, 0));
    var normed = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_vector_array_get(&normed, outputs_vec, 1));
    var normed_f32: mlx.mlx_array = .{ .ctx = null };
    if (f32_copy) {
        normed_f32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&normed_f32, outputs_vec, 2));
    }
    if (!add_rmsnorm_engaged) {
        add_rmsnorm_engaged = true;
        log.info("[layer] fused residual+RMSNorm kernel engaged: axis={d} tg={d}\n", .{ axis, tg });
    }
    return .{ .sum = sum, .normed = normed, .normed_f32 = normed_f32 };
}

test "mimoAttnArm: the verify budget follows the FP8 GEMV's direct-row arm" {
    // Byte identity for an accepted row needs the trunk GEMV to keep a single
    // row's arithmetic, so the budget is that row count and not a second literal.
    const design = MIMO_VERIFY_ROWS_MAX;
    const armed = fp8_block.gemv_direct_max_rows;
    defer fp8_block.gemv_direct_max_rows = armed;
    fp8_block.gemv_direct_max_rows = design - 1;
    try testing.expectEqual(MimoAttnArm.verify_rows, try mimoAttnArm(true, true, false, design - 1));
    // A width the trunk would serve by reassociating is refused, not re-armed.
    try testing.expectError(error.MimoVerifyRowsTooWide, mimoAttnArm(true, true, false, design));
}

test "mimoAttnArm: a verify wider than the decode row budget is refused, not re-armed" {
    // A verify is decode-shaped at every width it serves...
    try testing.expectEqual(MimoAttnArm.decode, try mimoAttnArm(true, false, false, 1));
    var rows: c_int = 2;
    while (rows <= MIMO_VERIFY_ROWS_MAX) : (rows += 1) {
        try testing.expectEqual(MimoAttnArm.verify_rows, try mimoAttnArm(true, true, false, rows));
        try testing.expectEqual(MimoAttnArm.verify_rows, try mimoAttnArm(true, true, true, rows));
    }
    // ...and past it there is no other correct arm.
    while (rows < 2 * MIMO_VERIFY_ROWS_MAX) : (rows += 1) {
        try testing.expectError(error.MimoVerifyRowsTooWide, mimoAttnArm(true, true, false, rows));
        try testing.expectError(error.MimoVerifyRowsTooWide, mimoAttnArm(true, true, true, rows));
    }
    // A plain forward keeps the arms it had.
    try testing.expectEqual(MimoAttnArm.decode, try mimoAttnArm(false, false, false, 1));
    try testing.expectEqual(MimoAttnArm.prefill_global, try mimoAttnArm(false, true, true, 512));
    try testing.expectEqual(MimoAttnArm.prefill_sliding, try mimoAttnArm(false, true, false, 512));
}

test "mimoAttnArm: a global prefill under the fused kernel's floor runs row by row" {
    // Below FUSED256_MIN_Q_LEN the fused kernel declines, and the composed arm would rebuild the
    // whole packed cache dense beside a [heads, rows, keys] score sheet, neither billed.
    try testing.expectEqual(MimoAttnArm.prefill_rows, try mimoAttnArm(false, true, true, 2));
    try testing.expectEqual(MimoAttnArm.prefill_rows, try mimoAttnArm(false, true, true, FUSED256_MIN_Q_LEN - 1));
    try testing.expectEqual(MimoAttnArm.prefill_global, try mimoAttnArm(false, true, true, FUSED256_MIN_Q_LEN));
    // A sliding layer reads a window-bounded view at any width.
    try testing.expectEqual(MimoAttnArm.prefill_sliding, try mimoAttnArm(false, true, false, 2));
}

test "qkv matmul2d kernel parity: MiMo global shape (qk 192 / v 128, gqa 16) vs the fp32 truth" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x3D2D_0192);
    const rnd = prng.random();
    for ([_]u8{ 8, 6, 4 }) |bits| {
        // 1000 + Tq keys: a partial last page.
        for (1..5) |tq| try qkvPackedParityCase(qkvAttnMppKernel, rnd, 64, 4, 192, 128, bits, 1000, @intCast(tq));
    }
}

test "qkv matmul2d kernel parity: many splits, and the narrower gqa 4 and 24/4 hd 256 shapes" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(0x3D2D_5117);
    const rnd = prng.random();
    // 20000 keys = 625 pages over 78 splits.
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 64, 4, 192, 128, 8, 20000, 1);
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 64, 4, 192, 128, 8, 20000, 4);
    // Narrower shapes: gqa 4, and upstream's 24/4 hd 256.
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 16, 4, 192, 128, 8, 3000, 1);
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 24, 4, 256, 256, 8, 3000, 1);
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 24, 4, 256, 256, 4, 3000, 2);
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 24, 4, 256, 256, 6, 3000, 2);
    try qkvPackedParityCase(qkvAttnMppKernel, rnd, 64, 4, 192, 128, 6, 20000, 4);
}

test "qkv split-K kernel parity: MiMo global shape (qk 192 / v 128, gqa 16) vs the fp32 truth, no matrix units" {
    try mlx.requireGpu();
    const saved_nax = trf.vqmm_nax_probe_override;
    trf.vqmm_nax_probe_override = false;
    defer trf.vqmm_nax_probe_override = saved_nax;
    defer qkv_splitk_split_override = null;
    var prng = std.Random.DefaultPrng.init(0x3D2D_5A11);
    const rnd = prng.random();
    for ([_]u8{ 8, 6, 4 }) |bits| {
        // 1001 keys: a partial last tile. Tq 3 = the causal verify rows.
        qkv_splitk_split_override = null;
        try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 64, 4, 192, 128, bits, 1000, 1);
        try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 64, 4, 192, 128, bits, 1000, 3);
        try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 64, 4, 192, 128, bits, 20000, 1);
        // More splits than tiles: some splits are empty, and the rest hold one tile each.
        qkv_splitk_split_override = .{ 15, 1, 1024 };
        try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 64, 4, 192, 128, bits, 1000, 1);
        // 1251 tiles over 7 splits: long splits, the last one short.
        qkv_splitk_split_override = .{ 3000, 1, 1024 };
        try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 64, 4, 192, 128, bits, 20000, 1);
    }
    // Flash-Next's own width through the same range body.
    qkv_splitk_split_override = null;
    try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 24, 2, 256, 256, 8, 3000, 1);
    try qkvPackedParityCase(qkvAttnSplitKKernel, rnd, 24, 2, 256, 256, 6, 3000, 1);
}

test "qkv matmul2d µbench: MiMo global decode, dense dequant+SDPA vs matmul2d split sweep (SUSHI_KVQ_MPP_UBENCH=1)" {
    if (!diagEnvOn("SUSHI_KVQ_MPP_UBENCH")) return error.SkipZigTest;
    try kvqDecodeUbench("kvq-mpp", &.{ .empty, .dense, .{ .mpp = .{ QKV_MPP_PAGES_PER_SPLIT, QKV_MPP_MAX_SPLITS } }, .{ .mpp = .{ 8, 128 } }, .{ .mpp = .{ 2, 512 } } });
}

test "qkv split-K µbench: MiMo global decode, dequant rebuild+SDPA vs split-K geometry sweep (SUSHI_KVQ_SPLITK_UBENCH=1)" {
    if (!diagEnvOn("SUSHI_KVQ_SPLITK_UBENCH")) return error.SkipZigTest;
    try kvqDecodeUbench("kvq-splitk", &.{
        .empty,
        .dense_rebuild,
        .{ .split_k = .{ QKV_SPLITK_KEYS_PER_SPLIT, QKV_SPLITK_MIN_SPLITS, QKV_SPLITK_MAX_SPLITS } },
        .{ .split_k = .{ 512, 32, 128 } },
        .{ .split_k = .{ 256, 64, 128 } },
        .{ .split_k = .{ 512, 64, 256 } },
    });
}

test "qkvMppDecodeServes: decode width, packed cache, and the measured kv floor" {
    const s = mlx.gpuStream();
    const View = struct {
        fn at(st: mlx.mlx_stream, t_k: c_int, q: *mlx.mlx_array) !DenseKVView {
            try mlx.check(mlx.mlx_zeros(q, &[_]c_int{ 1, 4, t_k, 48 }, 4, .uint32, st));
            return .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = q.* };
        }
    };
    var below = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(below);
    var floor = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(floor);
    const v_below = try View.at(s, QKV_MPP_DECODE_MIN_TK - 1, &below);
    const v_floor = try View.at(s, QKV_MPP_DECODE_MIN_TK, &floor);
    try testing.expect(packedDecodeServesFrom(&v_below, 1, QKV_MPP_DECODE_MIN_TK - 1)); // the floor is the caller's
    try testing.expect(!qkvMppDecodeServes(&v_below, 1));
    try testing.expect(qkvMppDecodeServes(&v_floor, 1));
    try testing.expect(!qkvMppDecodeServes(&v_floor, 2));
    var dense = v_floor;
    dense.has_quant_triple = false;
    try testing.expect(!qkvMppDecodeServes(&dense, 1));
}

test "mimoGlobalDecodeArm: matmul2d with matrix units, otherwise split-K from its floor, else dense" {
    const s = mlx.gpuStream();
    const View = struct {
        fn at(st: mlx.mlx_stream, t_k: c_int, q: *mlx.mlx_array) !DenseKVView {
            try mlx.check(mlx.mlx_zeros(q, &[_]c_int{ 1, 4, t_k, 48 }, 4, .uint32, st));
            return .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false, .has_quant_triple = true, .k_triple_q = q.* };
        }
    };
    var arrs: [4]mlx.mlx_array = undefined;
    for (&arrs) |*a| a.* = mlx.mlx_array_new();
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const mpp_below = try View.at(s, QKV_MPP_DECODE_MIN_TK - 1, &arrs[0]);
    const mpp_floor = try View.at(s, QKV_MPP_DECODE_MIN_TK, &arrs[1]);
    const sk_below = try View.at(s, QKV_SPLITK_DECODE_MIN_TK - 1, &arrs[2]);
    const sk_floor = try View.at(s, QKV_SPLITK_DECODE_MIN_TK, &arrs[3]);
    try testing.expectEqual(MimoDecodeArm.dense, mimoGlobalDecodeArm(&mpp_below, 1, true));
    try testing.expectEqual(MimoDecodeArm.mpp, mimoGlobalDecodeArm(&mpp_floor, 1, true));
    try testing.expectEqual(MimoDecodeArm.dense, mimoGlobalDecodeArm(&sk_below, 1, false));
    try testing.expectEqual(MimoDecodeArm.split_k, mimoGlobalDecodeArm(&sk_floor, 1, false));
    try testing.expectEqual(MimoDecodeArm.dense, mimoGlobalDecodeArm(&sk_floor, 2, false));
    var dense = sk_floor;
    dense.has_quant_triple = false;
    try testing.expectEqual(MimoDecodeArm.dense, mimoGlobalDecodeArm(&dense, 1, false));
}

test "mimoGlobalDecodeArm: a packed cache grown one token at a time switches to the packed arm at the floor" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try cache.reinit(1, KVQuantConfig.affine(8));
    const floor: usize = @intCast(QKV_MPP_DECODE_MIN_TK);
    {
        const k = testKVWide(floor - 2, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(floor - 2, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        dv.deinit();
    }
    for (floor - 1..floor + 2) |keys| {
        const k = testKVWide(1, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(1, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        const packed_arm = keys >= floor;
        for ([_]bool{ true, false }) |nax| {
            const arm = mimoGlobalDecodeArm(&dv, 1, nax);
            try testing.expectEqual(packed_arm, arm != .dense);
            try testing.expectEqual(!packed_arm, keys <= mimoGlobalDecodeRebuildMaxKeys());
        }
    }
}

test "mimo sliding verify rows run in one dispatch, each row bit-identical to its own sdpa" {
    try mlx.requireGpu();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        for ([_]c_int{ 128, 129, 700 }) |prefix| {
            for ([_]c_int{ 2, 3, 4, 5, 6, 7, 8 }) |width| {
                for ([_]f32{ 0.05, 1.0, 6.0 }) |gain| try mimoSlidingRowsCase(config, prefix, width, gain);
            }
        }
    }
}

test "mimo sliding verify rows keep the per-row path while a row still sees the whole history" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const q = try qkvIdentityArray(s, 64, 3, 192, 0.3, .bfloat16);
    defer _ = mlx.mlx_array_free(q);
    const k = try qkvIdentityArray(s, 8, 130, 192, 0.5, .bfloat16);
    defer _ = mlx.mlx_array_free(k);
    const v = try qkvIdentityArray(s, 8, 130, 128, 0.9, .bfloat16);
    defer _ = mlx.mlx_array_free(v);
    const sinks = try qkvSinks(s, 64);
    defer _ = mlx.mlx_array_free(sinks);
    const mask = try slidingDecodeMask(s, 128, 128);
    defer _ = mlx.mlx_array_free(mask);
    const view: DenseKVView = .{ .k = k, .v = v, .owned = false };
    try testing.expect((try mimoSlidingRowsAttn(s, 128, q, &view, sinks, 127, 3, 0.07, mask)) == null);
}

test "mimo verify rows attend with each serial decode tick's arithmetic (dense, kv8, window edge, ring, packed global)" {
    try mlx.requireGpu();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        for ([_]bool{ false, true }) |is_global| {
            for ([_]c_int{ 1, 60, 125, 126, 700 }) |prefix| {
                for ([_]c_int{ 2, 3, 4, 5, 6, 7, 8 }) |width| try mimoVerifyRowsIdentityCase(config, is_global, prefix, width);
            }
        }
    }
    // The packed global reads (matmul2d / split-K) serve from their key floor; the second block straddles it.
    try mimoVerifyRowsIdentityCase(KVQuantConfig.affine(8), true, QKV_MPP_DECODE_MIN_TK + 3, 4);
    try mimoVerifyRowsIdentityCase(KVQuantConfig.affine(8), true, QKV_MPP_DECODE_MIN_TK - 2, 4);
    // A short global prefill (a warm tail) takes the same rows, up to the fused kernel's floor.
    for ([_]c_int{ 5, FUSED256_MIN_Q_LEN - 1 }) |width| {
        try mimoVerifyRowsIdentityCase(KVQuantConfig.affine(8), true, QKV_MPP_DECODE_MIN_TK + 3, width);
        try mimoVerifyRowsIdentityCase(KVQuantConfig.dense, true, 700, width);
    }
}

test "mimo global verify rows on a packed cache share matmul2d passes in groups, each row its decode tick's bytes" {
    if (mlx.noGpuBackend() or !mimoDecodeUsesNax()) return error.SkipZigTest;
    const Case = struct { prefix: c_int, width: c_int, shared: bool, kv_heads: c_int = 1, bits: u8 = 8 };
    // A row's split partition follows its own key count: a group of rows shares a pass only where every
    // partition is the group's last row's, and a shorter row then reads one fully masked page more.
    const cases = [_]Case{
        .{ .prefix = QKV_MPP_DECODE_MIN_TK, .width = 4, .shared = true },
        .{ .prefix = QKV_MPP_DECODE_MIN_TK, .width = 4, .shared = true, .kv_heads = 4 },
        .{ .prefix = QKV_MPP_DECODE_MIN_TK + 61, .width = 4, .shared = false }, // pages 130 -> 131: 2 -> 3 per split
        .{ .prefix = 65566, .width = 4, .shared = true }, // pages 2049 -> 2050, 17 per split
        .{ .prefix = 65566, .width = 4, .shared = true, .kv_heads = 4 },
        .{ .prefix = 65566, .width = 4, .shared = true, .bits = 4 },
        .{ .prefix = 65566, .width = 4, .shared = true, .bits = 6, .kv_heads = 4 },
        .{ .prefix = 65566, .width = 3, .shared = true },
        .{ .prefix = 65566, .width = 8, .shared = true },
        .{ .prefix = 262175, .width = 4, .shared = true }, // pages 8193 -> 8194 inside the first pair, 65 per split
        .{ .prefix = 262175, .width = 3, .shared = true },
        .{ .prefix = QKV_MPP_DECODE_MIN_TK, .width = FUSED256_MIN_Q_LEN - 1, .shared = true },
    };
    for (cases) |c| {
        mimo_global_rows_mpp_count = 0;
        try mimoVerifyRowsIdentityCaseHeads(KVQuantConfig.affine(c.bits), true, c.prefix, c.width, 16 * c.kv_heads, c.kv_heads);
        try testing.expectEqual(c.shared, mimo_global_rows_mpp_count > 0);
    }
}

test "mimo decode qkv prep: bit-identical to partial rope + kv8 and kv6 quantize at every verify width" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4D69);
    const rnd = prng.random();
    const hq: c_int = 64;
    const rd: c_int = 64;
    const perm = [_]c_int{ 0, 2, 1, 3 };
    // Global (base 1e7, 4 KV heads) and sliding (base 1e4, 8 KV heads) layers.
    const layers = [_]struct { hk: c_int, base: f32 }{ .{ .hk = 4, .base = 1.0e7 }, .{ .hk = 8, .base = 1.0e4 } };
    // Heads span tiny to large activations with outliers; one K group is constant and one
    // V group is zero, the quantizer's eps and at-zero corners.
    const head_scales = [_]f32{ 0.02, 0.5, 3.0, 40.0, 300.0 };
    for (layers) |layer| for (1..@as(usize, MIMO_VERIFY_ROWS_MAX) + 1) |rows_u| {
        const rows: c_int = @intCast(rows_u);
        for ([_]c_int{ 0, 1, 4095, 65537, 500000 }) |offset| {
            var parts: [3]mlx.mlx_array = undefined;
            const widths = [_]c_int{ hq * 192, layer.hk * 192, layer.hk * 128 };
            for (&parts, widths, 0..) |*p, w, which| {
                const buf = try allocator.alloc(f32, @intCast(w * rows));
                defer allocator.free(buf);
                const head_w: usize = if (which == 2) 128 else 192;
                for (buf, 0..) |*v, i| {
                    const sc = head_scales[(i / head_w) % head_scales.len];
                    const outlier: f32 = if (rnd.uintLessThan(u32, 97) == 0) 20.0 else 1.0;
                    v.* = rnd.floatNorm(f32) * sc * outlier;
                }
                if (which == 1) @memset(buf[192 + 64 .. 192 + 128], 7.25);
                if (which == 2) @memset(buf[64..128], 0.0);
                const raw = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, rows, w }, 3, .float32);
                defer _ = mlx.mlx_array_free(raw);
                p.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(p, raw, .bfloat16, s));
            }
            defer for (parts) |p| {
                _ = mlx.mlx_array_free(p);
            };
            const base = mlx.mlx_optional_float{ .value = layer.base, .has_value = true };

            var want: [3]mlx.mlx_array = undefined;
            for (&want, [_]c_int{ hq, layer.hk, layer.hk }, [_]c_int{ 192, 192, 128 }, 0..) |*o, heads, hd, i| {
                var r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(r);
                try mlx.check(mlx.mlx_reshape(&r, parts[i], &[_]c_int{ 1, rows, heads, hd }, 4, s));
                var t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(t);
                try mlx.check(mlx.mlx_transpose_axes(&t, r, &perm, 4, s));
                o.* = mlx.mlx_array_new();
                if (i == 2) {
                    try mlx.check(mlx.mlx_array_set(o, t));
                } else {
                    try mlx.check(mlx.mlx_fast_rope(o, t, rd, false, base, 1.0, offset, .{ .ctx = null }, s));
                }
            }
            defer for (want) |a| {
                _ = mlx.mlx_array_free(a);
            };
            const angles = try ropeAngleRows(s, rd, base, .{ .ctx = null }, offset, rows, allocator, 1.0);
            defer _ = mlx.mlx_array_free(angles);
            for ([_]u8{ 8, 6 }) |bits| {
                var want_k = try kv_quant.quantizeAffine(s, want[1], 64, bits);
                defer want_k.deinit();
                var want_v = try kv_quant.quantizeAffine(s, want[2], 64, bits);
                defer want_v.deinit();
                var got = (try mimoDecodeQkvPrep(s, parts[0], parts[1], parts[2], angles, hq, layer.hk, rd, rows, bits)) orelse return error.TestExpectedKernel;
                defer got.deinit();

                try testExpectSameBits(want[0], got.q, s);
                try testExpectSameBits(want_k.q, got.k.q, s);
                try testExpectSameBits(want_k.scales, got.k.scales, s);
                try testExpectSameBits(want_k.biases, got.k.biases, s);
                try testExpectSameBits(want_v.q, got.v.q, s);
                try testExpectSameBits(want_v.scales, got.v.scales, s);
                try testExpectSameBits(want_v.biases, got.v.biases, s);
            }
        }
    };
}

test "fused GROUPED MoE router reproduces groupLimitedRouting at the SHIPPED 512/8/4/8 geometry" {
    try mlx.requireGpu();
    // Bit equality with the composed chain is the bar, not "close enough": the
    // first live A/B of the plain fused router computed a cleaner softmax and
    // diverged at token ~80 (the INT4 near-tie class,
    // triggered by an avoidable numerical change). The geometry is the SHIPPED
    // one on purpose — a group limit tuned at a toy 4x4 cannot see a mask that
    // is right for 4 groups and wrong for 8.
    //
    // ROWS is a SAMPLE SIZE, and a bit-equality bar is worth exactly its
    // sample: at 4 rows this compared 32 weights, which dodged a live
    // 6.1%-of-weights divergence on the dev GPU and hit it on CI's (the
    // JIT-vs-metallib `exp` above). 256 rows is 2048 weights.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x11_6C_0DE);
    const rnd = prng.random();
    trf.moe_router_fused_override = true;
    defer trf.moe_router_fused_override = null;

    const ROWS: c_int = 256;
    const E: c_int = 512;
    const K: c_int = 8;
    const NG: c_int = 8;
    const TOPKG: c_int = 4;
    const GSZ: usize = @intCast(@divExact(E, NG));
    const ROUTE_SCALE: f32 = 2.5;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 6.0 + @as(f32, @floatFromInt(i % 7)) * 0.011;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);

    const bbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const fused = (try moeRouterTopK(s, l32, bias, K, .sigmoid_bias_grouped, true, ROUTE_SCALE, .float32, NG, TOPKG)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);

    trf.moe_router_fused_override = false;
    const chain = try groupLimitedRouting(l32, bias, K, NG, TOPKG, true, ROUTE_SCALE, s);
    trf.moe_router_fused_override = true;
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    // Independent f64 reference for the WHOLE gate, group limit included.
    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    const sig = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(sig);
    var ref_ids: [8]usize = undefined;
    var dropped_seen = false;
    for (0..@intCast(ROWS)) |r| {
        const row = buf[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| {
            sig[i] = 1.0 / (1.0 + @exp(-@as(f64, v)));
            keys[i] = sig[i] + @as(f64, bbuf[i]);
        }
        // Group score = sum of the top TWO biased experts in the group.
        var gscore: [8]f64 = undefined;
        for (0..@intCast(NG)) |g| {
            var m1: f64 = -std.math.inf(f64);
            var m2: f64 = -std.math.inf(f64);
            for (keys[g * GSZ ..][0..GSZ]) |v| {
                if (v > m1) {
                    m2 = m1;
                    m1 = v;
                } else if (v > m2) m2 = v;
            }
            gscore[g] = m1 + m2;
        }
        var kept: [8]bool = @splat(false);
        var gsel: [8]usize = undefined;
        routerRefRow(gscore[0..@intCast(NG)], @intCast(TOPKG), &gsel);
        for (gsel[0..@intCast(TOPKG)]) |g| kept[g] = true;
        for (0..@intCast(E)) |i| {
            if (!kept[i / GSZ]) {
                keys[i] = -1.0e4;
                dropped_seen = true;
            }
        }
        routerRefRow(keys, @intCast(K), &ref_ids);

        // Every selected expert must live in a surviving group — the failure
        // mode of a silently skipped group step is that low groups stay
        // reachable, and it is invisible in the weights.
        for (0..@intCast(K)) |j| {
            try testing.expect(kept[ref_ids[j] / GSZ]);
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        // The chain does not order its top-k, so compare index SETS with it.
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        for (0..@intCast(K)) |j| {
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            try testing.expectEqual(c_w[r * @as(usize, @intCast(K)) + slot], f_w[r * @as(usize, @intCast(K)) + j]);
        }
    }
    // The test data has to actually exercise the limit, or it proves nothing.
    try testing.expect(dropped_seen);
}

test "fused MoE router reproduces the sigmoid+bias routing chain" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED2026);
    const rnd = prng.random();
    trf.moe_router_fused_override = true;
    defer trf.moe_router_fused_override = null;

    const ROWS: c_int = 4;
    const E: c_int = 256;
    const K: c_int = 8;
    const ROUTE_SCALE: f32 = 2.5;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 6.0 + @as(f32, @floatFromInt(i % 5)) * 0.013;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);
    var lbf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lbf);
    try mlx.check(mlx.mlx_astype(&lbf, l32, .bfloat16, s));

    const bbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const host = try allocator.alloc(f32, n);
    defer allocator.free(host);
    try testReadF32(lbf, host, s);

    const fused = (try moeRouterTopK(s, lbf, bias, K, .sigmoid_bias, true, ROUTE_SCALE, .bfloat16, 0, 0)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try sigmoidBiasRoutingChain(lbf, bias, K, true, ROUTE_SCALE, .bfloat16, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    const sig = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(sig);
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = host[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| {
            sig[i] = 1.0 / (1.0 + @exp(-@as(f64, v)));
            keys[i] = sig[i] + @as(f64, bbuf[i]);
        }
        routerRefRow(keys, @intCast(K), &ref_ids);
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        var denom: f64 = 1e-20;
        for (0..@intCast(K)) |j| denom += sig[ref_ids[j]];
        for (0..@intCast(K)) |j| {
            const want = sig[ref_ids[j]] / denom * @as(f64, ROUTE_SCALE);
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            try testing.expectEqual(chain_got, got);
        }
    }
    // Both round to bf16 at the end, so the bar is "no worse", not "better".
    try testing.expect(fused_err <= chain_err * 1.0001 + 1e-6);
}

test "fused MoE router reproduces the mimo_v2 sigmoid+bias routing chain in f32" {
    try mlx.requireGpu();
    // The pairing mimo_v2 SHIPS: ungrouped `.sigmoid_bias` with an f32 output and
    // f32 router inputs (the router is widened once at load). The arm above
    // proves the same kernel at a bf16 output, which absorbs a last-ulp
    // difference between the kernel's `w / (tot + 1e-20f)` and MLX's divide
    // chain; MiMo keeps the weights in f32 into the expert sum, so nothing
    // absorbs it and the bar stays BIT equality.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x2113_0F32);
    const rnd = prng.random();
    trf.moe_router_fused_override = true;
    defer trf.moe_router_fused_override = null;

    const ROWS: c_int = 256;
    const E: c_int = 256;
    const K: c_int = 8;
    const ROUTE_SCALE: f32 = 2.5;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 6.0 + @as(f32, @floatFromInt(i % 5)) * 0.013;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);

    const bbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(bbuf);
    for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.2;
    const bsh = [_]c_int{E};
    const bias = mlx.mlx_array_new_data(bbuf.ptr, &bsh, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const fused = (try moeRouterTopK(s, l32, bias, K, .sigmoid_bias, true, ROUTE_SCALE, .float32, 0, 0)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try mimoRoutingChain(l32, bias, K, true, ROUTE_SCALE, s);
    defer _ = mlx.mlx_array_free(chain.inds);
    defer _ = mlx.mlx_array_free(chain.norm_scores);

    const kn: usize = @intCast(ROWS * K);
    const f_ids = try allocator.alloc(f32, kn);
    defer allocator.free(f_ids);
    const c_ids = try allocator.alloc(f32, kn);
    defer allocator.free(c_ids);
    const f_w = try allocator.alloc(f32, kn);
    defer allocator.free(f_w);
    const c_w = try allocator.alloc(f32, kn);
    defer allocator.free(c_w);
    try testReadF32(fused.inds, f_ids, s);
    try testReadF32(chain.inds, c_ids, s);
    try testReadF32(fused.norm_scores, f_w, s);
    try testReadF32(chain.norm_scores, c_w, s);

    const keys = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(keys);
    const sig = try allocator.alloc(f64, @intCast(E));
    defer allocator.free(sig);
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = buf[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| {
            sig[i] = 1.0 / (1.0 + @exp(-@as(f64, v)));
            keys[i] = sig[i] + @as(f64, bbuf[i]);
        }
        routerRefRow(keys, @intCast(K), &ref_ids);
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        var denom: f64 = 1e-20;
        for (0..@intCast(K)) |j| denom += sig[ref_ids[j]];
        for (0..@intCast(K)) |j| {
            const want = sig[ref_ids[j]] / denom * @as(f64, ROUTE_SCALE);
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            try testing.expectEqual(@as(u32, @bitCast(chain_got)), @as(u32, @bitCast(got)));
        }
    }
    // f32 weights, so the fused arm is the chain's arithmetic, not a cleaner one.
    try testing.expect(fused_err <= chain_err * 1.0001 + 1e-6);
}

test "fused residual+RMSNorm is bit-identical to mlx_add + mlx_fast_rms_norm" {
    try mlx.requireGpu();
    // Same bar as the other fusions, and the same reason it is reachable: the
    // kernel keeps MLX's launch geometry, so the reduction is the same TREE and
    // not merely the same sum. Widths chosen to hit the aligned path (2048,
    // 512), the ragged tail (2050 — not a multiple of 4), and a non-power-of-2
    // axis (5120) where `acc / axis_size` could differ if a compiler turned the
    // divide into a reciprocal multiply.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x0DDBA11);
    const rnd = prng.random();

    const EPS: f32 = 1e-6;
    const eps_arr = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps_arr);

    for ([_]c_int{ 512, 1536, 2048, 2050, 4096 }) |axis| {
        for ([_]c_int{ 1, 3 }) |rows| {
            for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
                const cnt: usize = @intCast(rows * axis);
                const abuf = try allocator.alloc(f32, cnt);
                defer allocator.free(abuf);
                const bbuf = try allocator.alloc(f32, cnt);
                defer allocator.free(bbuf);
                const wbuf = try allocator.alloc(f32, @intCast(axis));
                defer allocator.free(wbuf);
                for (abuf) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;
                for (bbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 6.0;
                for (wbuf) |*v| v.* = 0.5 + rnd.float(f32);

                const sh = [_]c_int{ rows, axis };
                const wsh = [_]c_int{axis};
                const a32 = mlx.mlx_array_new_data(abuf.ptr, &sh, 2, .float32);
                defer _ = mlx.mlx_array_free(a32);
                const b32 = mlx.mlx_array_new_data(bbuf.ptr, &sh, 2, .float32);
                defer _ = mlx.mlx_array_free(b32);
                const w32 = mlx.mlx_array_new_data(wbuf.ptr, &wsh, 1, .float32);
                defer _ = mlx.mlx_array_free(w32);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                try mlx.check(mlx.mlx_astype(&a, a32, dt, s));
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_astype(&b, b32, dt, s));
                var w = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(w);
                try mlx.check(mlx.mlx_astype(&w, w32, dt, s));

                var ref_sum = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ref_sum);
                try mlx.check(mlx.mlx_add(&ref_sum, a, b, s));
                var ref_norm = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(ref_norm);
                try mlx.check(mlx.mlx_fast_rms_norm(&ref_norm, ref_sum, w, EPS, s));

                const got = (try fusedAddRmsNormUngated(s, a, b, w, eps_arr)) orelse return error.FusedAddNormDeclined;
                defer _ = mlx.mlx_array_free(got.sum);
                defer _ = mlx.mlx_array_free(got.normed);

                const h1 = try allocator.alloc(f32, cnt);
                defer allocator.free(h1);
                const h2 = try allocator.alloc(f32, cnt);
                defer allocator.free(h2);
                try testReadF32(ref_sum, h1, s);
                try testReadF32(got.sum, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
                try testReadF32(ref_norm, h1, s);
                try testReadF32(got.normed, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);

                // The routed variant adds the norm's exact f32 widening, a router's input.
                const routed = (try fusedAddRmsNormRouted(s, a, b, w, eps_arr)) orelse return error.FusedAddNormDeclined;
                defer _ = mlx.mlx_array_free(routed.sum);
                defer _ = mlx.mlx_array_free(routed.normed);
                defer _ = mlx.mlx_array_free(routed.normed_f32);
                try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(routed.normed_f32));
                try testReadF32(routed.normed, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
                try testReadF32(routed.normed_f32, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
                try testReadF32(routed.sum, h2, s);
                try testReadF32(ref_sum, h1, s);
                try testing.expectEqualSlices(f32, h1, h2);
            }
        }
    }
}

test "fused residual+RMSNorm declines what it cannot reproduce" {
    const s = mlx.gpuStream();
    // mlx_array_new_data COPIES shape-worth of bytes — the buffer must really
    // hold 8192 floats (a 16-float buffer here "passed" by overreading the
    // stack until a layout shift put the guard page in range).
    var buf: [8192]f32 = @splat(0.25);
    // Above RMS_LOOPED_LIMIT mlx switches to a different kernel with a
    // different reduction shape.
    const big = [_]c_int{ 1, 8192 };
    const wbig = [_]c_int{8192};
    const a = mlx.mlx_array_new_data(&buf, &big, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    const w = mlx.mlx_array_new_data(&buf, &wbig, 1, .float32);
    defer _ = mlx.mlx_array_free(w);
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    try testing.expect((try fusedAddRmsNormUngated(s, a, a, w, eps)) == null);
    // Mismatched shapes must decline rather than read out of bounds.
    const s2 = [_]c_int{ 1, 4 };
    const small = mlx.mlx_array_new_data(&buf, &s2, 2, .float32);
    defer _ = mlx.mlx_array_free(small);
    try testing.expect((try fusedAddRmsNormUngated(s, a, small, w, eps)) == null);
}

test "sigmoidBiasRoutingChain selects on biased scores but weights by original sigmoid scores" {
    const s = mlx.gpuStream();

    // 1 row over 6 experts, k=2. sigmoid(logits):
    //   [0.8808, 0.5, 0.1192, 0.7311, 0.2689, 0.6225]
    // With NO bias the top-2 would be {0 (0.8808), 3 (0.7311)}. The expert bias
    // +0.5 on expert 1 lifts it to 1.0, so selection must pick {1, 0} — but the
    // WEIGHTS must come from the ORIGINAL sigmoid scores {0.5, 0.8808}
    // (sum 1.3808), renormalized to 1 then scaled by router_scaling_factor.
    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const bias_data = [_]f32{ 0.0, 0.5, 0.0, 0.0, 0.0, 0.0 };
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const scale: f32 = 2.0;
    const routed = try sigmoidBiasRoutingChain(logits, bias, 2, true, scale, .bfloat16, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Gather the ORIGINAL sigmoid scores at the selected indices. If selection
    // ignored the bias the gathered sum would be 0.8808+0.7311 = 1.6119; with
    // bias-driven selection it must be 0.5+0.8808 = 1.3808.
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, logits, s));
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, sig, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // Renorm + scale: scores sum to exactly router_scaling_factor.
    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // norm_scores are bf16 (cast for the expert-combine multiply) → ~0.4% rel.
    try testing.expect(@abs(gs[0] - 1.3808) < 1e-3);
    try testing.expect(@abs(ss[0] - scale) < 0.02);

    // Per-element weights pin "weighted by ORIGINAL scores": {0.5, 0.8808}
    // renormed ×2 → {0.7242, 1.2758} (any order). Had the chain weighted by
    // the BIASED scores {1.0, 0.8808} the pair would be {1.0634, 0.9366} —
    // same sum, different elements, so the sum check alone can't see it.
    try mlx.check(mlx.mlx_array_eval(scores_f32));
    const sv = mlx.mlx_array_data_float32(scores_f32) orelse return error.InvalidDtype;
    const lo = @min(sv[0], sv[1]);
    const hi = @max(sv[0], sv[1]);
    try testing.expect(@abs(lo - 0.7242) < 0.02);
    try testing.expect(@abs(hi - 1.2758) < 0.02);
}

test "sigmoidBiasRoutingChain route_norm=false keeps raw sigmoid weights (scaled only)" {
    const s = mlx.gpuStream();

    const logits_data = [_]f32{ 2.0, 0.0, -2.0, 1.0, -1.0, 0.5 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&logits_data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Zero bias: selection = top-2 of raw sigmoid = {0.8808, 0.7311}.
    const bias_data: [6]f32 = @splat(0.0);
    const bias_shape = [_]c_int{6};
    const bias = mlx.mlx_array_new_data(&bias_data, &bias_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(bias);

    const routed = try sigmoidBiasRoutingChain(logits, bias, 2, false, 2.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    var scores_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_f32);
    try mlx.check(mlx.mlx_astype(&scores_f32, routed.norm_scores, .float32, s));
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, scores_f32, -1, false, s));
    try mlx.check(mlx.mlx_array_eval(scores_sum));

    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    // (0.880797 + 0.731059) × 2.0 = 3.2237 — un-normalized, scaled.
    try testing.expect(@abs(ss[0] - 3.2237) < 0.02);
}
