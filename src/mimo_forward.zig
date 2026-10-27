//! MiMo-V2 forward: sliding/global attention arms, routed-MoE dispatch, batched decode and verify rows.

const trf = @import("transformer.zig");
const Transformer = trf.Transformer;
const std = @import("std");
const fp8_block = @import("fp8_block.zig");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
pub const KVQuantConfig = kv_quant.KVQuantConfig;

const DenseKVView = trf.DenseKVView;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const FullAttnWeights = trf.FullAttnWeights;
const KvPrefixView = trf.KvPrefixView;
const ModelConfig = trf.ModelConfig;
const MoeMlpWeights = trf.MoeMlpWeights;
const ProfClock = trf.ProfClock;
const QKV_MPP_DECODE_MIN_TK = trf.QKV_MPP_DECODE_MIN_TK;
const QKV_SPLITK_DECODE_MIN_TK = trf.QKV_SPLITK_DECODE_MIN_TK;
const QkvMppKey = trf.QkvMppKey;
const capturePrefillHidden = trf.capturePrefillHidden;
const capturePrefillHiddenLast = trf.capturePrefillHiddenLast;
const decodeProfReport = trf.decodeProfReport;
const decodeProfileEnabled = trf.decodeProfileEnabled;
const decodeProfileRows = trf.decodeProfileRows;
const fusedAddRmsNormRouted = trf.fusedAddRmsNormRouted;
const fusedAddRmsNormUngated = trf.fusedAddRmsNormUngated;
const fusedSdpaPrefillKv = trf.fusedSdpaPrefillKv;
const groupLimitedRouting = trf.groupLimitedRouting;
const lastDim = trf.lastDim;
const layerCap = trf.layerCap;
const log = trf.log;
const moeDumpBeginForward = trf.moeDumpBeginForward;
const moeDumpForwardDone = trf.moeDumpForwardDone;
const moeDumpTensor = trf.moeDumpTensor;
const moeRouterTopK = trf.moeRouterTopK;
const packedDecodeServesFrom = trf.packedDecodeServesFrom;
const qkvAttnMpp = trf.qkvAttnMpp;
const qkvAttnMppKernel = trf.qkvAttnMppKernel;
const qkvAttnSplitKKernel = trf.qkvAttnSplitKKernel;
const qkvMppDecodeProbeOk = trf.qkvMppDecodeProbeOk;
const qkvMppDecodeServes = trf.qkvMppDecodeServes;
const qkvMppKernel = trf.qkvMppKernel;
const qkvMppPartition = trf.qkvMppPartition;
const qkvMppProbe = trf.qkvMppProbe;
const qkvMppRowsKernel = trf.qkvMppRowsKernel;
const qmatmulBits = trf.qmatmulBits;
const scalarOf = trf.scalarOf;
const sigmoidBiasRoutingChain = trf.sigmoidBiasRoutingChain;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const slidingPrefillAttn = trf.slidingPrefillAttn;
const slidingViewFor = trf.slidingViewFor;
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

/// Meter seam: false keeps every global verify row on its own dispatch.
pub var mimo_global_rows_override: ?bool = null;

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
    if (mimo_global_rows_override == false) return null;
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
            // Skipped when the fused hd-256 kernel band-masks in-kernel
            // (the mask itself is chunk x kv_len — GBs at long ctx).
            // Built at the TRIMMED length, which is what the attn fns hand SDPA.
            if (!sliding.band_in_kernel) {
                local_prefill_mask = try self.createSlidingWindowMask(seq_len, sliding.kv_len, sw);
            }
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
    // at the LAST position only. Used by PLD verify-fusion and the
    // Gemma 4 assistant drafter as `h_prev`. Caller frees the captured
    // array.
    if (ctx.capture_hidden) |target| {
        try capturePrefillHiddenLast(self.s, target, final_normed, !ctx.verify_rows);
    }
    if (ctx.capture_hidden_all) |target_all| {
        try capturePrefillHidden(self.s, target_all, final_normed, !ctx.verify_rows);
    }

    if (self.embedding_mode) return final_normed;
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

// ── DiffusionGemma bidirectional canvas decoder ──

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
    if (mimo_qkv_prep_override == false) return null;
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

/// Test and fwd-meter seam: false keeps the composed rope + quantize.
pub var mimo_qkv_prep_override: ?bool = null;

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
