const trf = @import("../transformer.zig");
const std = @import("std");
const qwen4_mod = @import("qwen4_exp.zig");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const qsaBatchedGatherAttn = qwen4_qsa.qsaBatchedGatherAttn;
const qsaPackedGatherServes = qwen4_qsa.qsaPackedGatherServes;
const qwen4Standin = qwen4_forward.qwen4Standin;
const KVQuantConfig = kv_quant.KVQuantConfig;
const log = @import("../log.zig");
const io_util_mod = @import("../io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const OneShotBits = qwen4_qsa_attn.OneShotBits;
const attn256Cosine = qwen4_qsa_attn.attn256Cosine;
const attn256UlpBar = qwen4_qsa_attn.attn256UlpBar;
const qsaAttnCacheFixture = qwen4_qsa_attn.qsaAttnCacheFixture;
const qsaAttnKernelEnabled = qwen4_qsa_attn.qsaAttnKernelEnabled;
const qsaBoostLastTileK = qwen4_qsa_attn.qsaBoostLastTileK;
const qsaDecodeGatherAttn = qwen4_qsa_attn.qsaDecodeGatherAttn;
const qsaEngagedBit = qwen4_qsa_attn.qsaEngagedBit;
const qsaGatherF32Reference = qwen4_qsa_attn.qsaGatherF32Reference;
const qsaMaskFromBlocks = qwen4_qsa_attn.qsaMaskFromBlocks;
const qsaSparseAttn = qwen4_qsa_attn.qsaSparseAttn;
const qsaVerifyBlocksHost = qwen4_qsa_attn.qsaVerifyBlocksHost;
const qsaVerifyGatherEnabled = qwen4_qsa_attn.qsaVerifyGatherEnabled;
const qsaVerifyGatherMinKvFor = qwen4_qsa_attn.qsaVerifyGatherMinKvFor;
const qwen4_test_support = @import("qwen4_test_support.zig");
const attn256QsaMask = qwen4_test_support.attn256QsaMask;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256Reference = test_support.attn256Reference;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const ATTN_QSA256_KERNEL_HEADER = trf.ATTN_QSA256_KERNEL_HEADER;
const DenseKVView = trf.DenseKVView;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const KV_UNPACK_HEADER = trf.KV_UNPACK_HEADER;
const QsaCfgCache = trf.QsaCfgCache;
const ShapeKey = trf.ShapeKey;
const Transformer = trf.Transformer;
const attn256Base16Aligned = trf.attn256Base16Aligned;
const fused256Enabled = trf.fused256Enabled;
const fusedSdpaPrefillImpl = trf.fusedSdpaPrefillImpl;
const qsaNaxOsOk = trf.qsaNaxOsOk;
const qsaNaxOsSupported = trf.qsaNaxOsSupported;
const qsaProbeAllFinite = trf.qsaProbeAllFinite;
const qsaProbeCorruptFirst = trf.qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = trf.qsaProbeLcgBf16;
const qsaProbeMaxAbsDiff = trf.qsaProbeMaxAbsDiff;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const splitMaskedSdpa256 = trf.splitMaskedSdpa256;
const verifyQmmNaxAvailable = trf.verifyQmmNaxAvailable;

var qsa_fused_logged = false;

/// Array-mask arm of sushi_attn_p256 (qwen4_exp QSA prefill): `mask` is the
/// [B,1,qL,kL] bool visibility with causal folded in. Key blocks no row of a
/// 64-row q tile sees are skipped in-kernel via a per-tile any-visible table
/// built here. Same envelope as the causal arm (q >= 16, hd 256, bf16).
pub fn fusedSdpa256Masked(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!fused256Enabled()) return null;
    if (mask.ctx == null or mlx.mlx_array_ndim(mask) != 4 or mlx.mlx_array_dtype(mask) != .bool_) return null;
    const ms = mlx.getShape(mask);
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    if (ms[0] != qs[0] or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const out = try fusedSdpaPrefillImpl(s, q, k, v, scale, 0, mask, null, .{ .ctx = null });
    if (out != null and !qsa_fused_logged) {
        qsa_fused_logged = true;
        log.info("[qsa-fused] engaged: sushi_attn_p256 mask arm qL={d} kL={d} Hq={d} Hkv={d}\n", .{ qs[2], ks[2], qs[1], ks[1] });
    }
    return out;
}

pub var qsa_gather_override: ?bool = null;

/// The block-gather kernel serves the qwen4_exp QSA prefill; the override restores the dense
/// [S, kv] mask arm.
pub fn qsaGatherEnabled() bool {
    return qsa_gather_override orelse true;
}

/// Below this many keys the mask arm wins: it reads at most kv rows per
/// 64-row tile, the gather reads 2051 rows per token unshared (ladder: 8k
/// -3%, 16k +4%, 32k +19%, 64k +42%).
pub const QSA_GATHER_MIN_KV_DEFAULT: c_int = 8192;
pub var qsa_gather_min_kv_override: ?c_int = null;

pub fn qsaGatherMinKv() c_int {
    return qsa_gather_min_kv_override orelse QSA_GATHER_MIN_KV_DEFAULT;
}

pub const QSA_GATHER_MIN_KV_NAX: c_int = 2052;

pub fn qsaPrefillGatherMinKvFrom(seq_len: c_int, nax_serves: bool, override: ?c_int) c_int {
    if (override) |v| return v;
    return if (seq_len >= FUSED256_MIN_Q_LEN and nax_serves) QSA_GATHER_MIN_KV_NAX else QSA_GATHER_MIN_KV_DEFAULT;
}


/// Key tile of the gather kernel.
const QSA_GATHER_BK: c_int = 32;

pub var qsa_decode_gather_override: ?bool = null;

/// Decode-width QSA gather (S==1): reuse the sorted block selection and read
/// only the selected blocks + incomplete tail instead of the full KV — O(topk)
/// per step, never O(kv). The override restores the dense bool-mask arm (same selection,
/// full-KV SDPA).
pub fn qsaDecodeGatherEnabled() bool {
    return qsa_decode_gather_override orelse true;
}

pub var qsa_batched_gather_override: ?bool = null;

pub fn qsaBatchedGatherEnabled() bool {
    return qsa_batched_gather_override orelse true;
}

pub fn qsaBatchedGatherOn(seq_len: c_int) bool {
    const st = qwen4Standin();
    if (st.attn_qsa or st.attn_sdpa) return false;
    if (!qsaBatchedGatherEnabled()) return false;
    if (!qsaGatherEnabled()) return false;
    if (seq_len <= 1) return qsaDecodeGatherEnabled();
    return qsaVerifyGatherEnabled();
}

pub fn qsaBatchedGatherFloor(seq_len: c_int, quantized: bool) c_int {
    if (seq_len >= 2) return @max(qsaGatherMinKv(), qsaVerifyGatherMinKvFor(quantized));
    return qsaGatherMinKv();
}

/// sushi_attn_qsa256: one threadgroup per (query token, kv head, batch). The
/// direct-index approach (gather each query's selected blocks, never a dense
/// [S, kv] mask) follows oMLX's Qwen4 exact QSA prefill (jundot/omlx PR
/// #3244, Apache-2.0); no code was ported — the kernel body is derived from
/// our own sushi_attn_p256 (whose K/V staging their kernel credits in turn).
/// token's `gqa` query heads are the tile rows (NSG simdgroups x 8); its keys
/// are gathered by the token's sorted selected blocks (`blocks[b, s, :]`,
/// RATIO tokens each, INT_MAX past the row's count) followed by its own
/// incomplete tail — O(budget) per token, never O(kv). Fragment math,
/// staging and online softmax mirror sushi_attn_p256.
const ATTN_QSA256_KERNEL_SOURCE =
    \\constexpr int BD = 256;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = BD + 8;
    \\constexpr int NT = 32 * NSG;
    \\constexpr int KT = BK / 8;
    \\
    \\const int qL = q_shape[2];
    \\const int kL = k_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = k_shape[1];
    \\const int gqa = Hq / Hk;
    \\const int KB = blocks_shape[2];
    \\
    \\const int s = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\const int bb = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\// Query s sits at absolute key position p (bottom-right aligned). Its
    \\// visible keys: `count` complete blocks, then the tail [tail_start, p].
    \\const int p = (kL - qL) + s;
    \\const int complete = (p + 1) / RATIO;
    \\const int count = metal::min(complete, KB);
    \\const int sel_len = count * RATIO;
    \\const int tail_start = complete * RATIO;
    \\const int L = sel_len + (p + 1 - tail_start);
    \\
    \\const device int* blk = blocks + (long)bb * blocks_strides[0] + (long)s * blocks_strides[1];
    \\#if QSA_PACKED
    \\const device uint32_t* Kp = k + bb * k_strides[0] + hk * k_strides[1];
    \\const device uint32_t* Vp = v + bb * v_strides[0] + hk * v_strides[1];
    \\const device T* Kscp = ksc + bb * ksc_strides[0] + hk * ksc_strides[1];
    \\const device T* Kbip = kbi + bb * kbi_strides[0] + hk * kbi_strides[1];
    \\const device T* Vscp = vsc + bb * vsc_strides[0] + hk * vsc_strides[1];
    \\const device T* Vbip = vbi + bb * vbi_strides[0] + hk * vbi_strides[1];
    \\#else
    \\const device T* Kp = k + bb * k_strides[0] + hk * k_strides[1];
    \\const device T* Vp = v + bb * v_strides[0] + hk * v_strides[1];
    \\#endif
    \\
    \\threadgroup T KVs[LDK * BD];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const short2 sc = sushi_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\const int row = tm + sm;
    \\const bool row_ok = row < gqa;
    \\
    \\float2 Qfrag[BD / 8];
    \\if (row_ok) {
    \\  const device T* Qrow = q + bb * q_strides[0] + (long)(hk * gqa + row) * q_strides[1] + (long)s * q_strides[2];
    \\  for (int dd = 0; dd < BD / 8; ++dd) {
    \\    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\    Qfrag[dd] = float2(float(pr.x), float(pr.y));
    \\  }
    \\} else {
    \\  for (int dd = 0; dd < BD / 8; ++dd) Qfrag[dd] = float2(0.0f);
    \\}
    \\float2 Ofrag[BD / 8];
    \\for (int i = 0; i < BD / 8; ++i) Ofrag[i] = float2(0.0f);
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\for (int t0 = 0; t0 < L; t0 += BK) {
    \\  const int rows_k = metal::min(BK, L - t0);
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      const int pos = sushi_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\#if QSA_PACKED
    \\      w = sushi_qsa_unpack8<T, BITS, GS>(Kp + (long)pos * k_strides[2], Kscp + (long)pos * ksc_strides[2], Kbip + (long)pos * kbi_strides[2], c8);
    \\#else
    \\      w = *((const device uint4*)(Kp + (long)pos * k_strides[2]) + c8);
    \\#endif
    \\    }
    \\    thread T* e = (thread T*)&w;
    \\    const int cb = c8 * 8;
    \\    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = e[j];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  float2 Sfrag[KT];
    \\  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
    \\  for (int dd = 0; dd < BD / 8; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
    \\      sushi_mma(Sfrag[kt], qf, kf);
    \\    }
    \\  }
    \\  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
    \\  if (rows_k < BK) {
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
    \\      if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
    \\    }
    \\  }
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * (BD / 8); i += NT) {
    \\    const int r = i >> 5;
    \\    const int c8 = i & 31;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      const int pos = sushi_qsa_pos(blk, t0 + r, sel_len, tail_start, RATIO);
    \\#if QSA_PACKED
    \\      w = sushi_qsa_unpack8<T, BITS, GS>(Vp + (long)pos * v_strides[2], Vscp + (long)pos * vsc_strides[2], Vbip + (long)pos * vbi_strides[2], c8);
    \\#else
    \\      w = *((const device uint4*)(Vp + (long)pos * v_strides[2]) + c8);
    \\#endif
    \\    }
    \\    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
    \\  }
    \\
    \\  float new_max = max_score;
    \\  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, sushi_row_max(Sfrag[kt]));
    \\  float rowsum = 0.0f;
    \\  for (int kt = 0; kt < KT; ++kt) {
    \\    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    \\    rowsum += sushi_row_sum(Sfrag[kt]);
    \\  }
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < BD / 8; ++i) Ofrag[i] *= factor;
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int id = 0; id < BD / 8; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
    \\      sushi_mma(Ofrag[id], Sfrag[kt], vf);
    \\    }
    \\  }
    \\}
    \\
    \\if (row_ok) {
    \\  const float inv = 1.0f / sum_score;
    \\  device T* Optr = out + (((long)bb * Hq + (hk * gqa + row)) * (long)qL + (long)s) * BD + sn;
    \\  for (int id = 0; id < BD / 8; ++id) {
    \\    Optr[id * 8] = T(Ofrag[id].x * inv);
    \\    Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
    \\  }
    \\}
;

var attn_qsa256_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getAttnQsa256Kernel() !mlx.mlx_fast_metal_kernel {
    if (attn_qsa256_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_attn_qsa256",
        in_vec,
        out_vec,
        ATTN_QSA256_KERNEL_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // K/V are cache views (see sushi_attn_p256)
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    attn_qsa256_kernel_cached = kernel;
    return kernel;
}

var qsa_nax_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
pub var qsa_nax_override: ?bool = null;
var qsa_nax_env: ?bool = null;

/// Default on: the hardware gate (`qsaNaxEligible`) decides. `SUSHI_QSA_NAX=0` forces the stock
/// gather.
pub fn qsaNaxEnabledFrom(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

pub fn qsaNaxEnabled() bool {
    if (qsaNaxProbeFailed()) return false;
    if (qsa_nax_override) |v| return v;
    if (qsa_nax_env) |v| return v;
    const raw = std.c.getenv("SUSHI_QSA_NAX");
    const v = qsaNaxEnabledFrom(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    qsa_nax_env = v;
    return v;
}

pub fn qsaNaxEligibleFrom(
    nax_available: bool,
    os_ver: []const u8,
    q_dtype: mlx.mlx_dtype,
    k_dtype: mlx.mlx_dtype,
    v_dtype: mlx.mlx_dtype,
    hd: c_int,
    gqa: c_int,
    q_len: c_int,
) bool {
    if (!nax_available) return false;
    if (!qsaNaxOsSupported(os_ver)) return false;
    if (q_dtype != .bfloat16 or k_dtype != .bfloat16 or v_dtype != .bfloat16) return false;
    if (hd != 256) return false;
    if (gqa != 12) return false;
    if (q_len < 16) return false;
    return true;
}

pub fn qsaNaxEligible(
    q_dtype: mlx.mlx_dtype,
    k_dtype: mlx.mlx_dtype,
    v_dtype: mlx.mlx_dtype,
    hd: c_int,
    gqa: c_int,
    q_len: c_int,
) bool {
    if (qsaNaxProbeFailed()) return false;
    return qsaNaxEligibleFrom(
        verifyQmmNaxAvailable(),
        if (qsaNaxOsOk()) "26.3" else "26.2",
        q_dtype,
        k_dtype,
        v_dtype,
        hd,
        gqa,
        q_len,
    );
}

var qsa_nax_probe_state: std.atomic.Value(u8) = .init(0);
pub var qsa_nax_probe_fail_override: bool = false;
pub var qsa_nax_probe_flip_override: bool = false;
pub var qsa_nax_probe_flip_packed_bits: ?u8 = null;
var qsa_nax_kernel_builds: usize = 0;
pub var qsa_gather_used_nax: bool = false;
pub var qsa_gather_engaged_stock: u32 = 0;
pub var qsa_gather_engaged_nax: u32 = 0;

pub fn qsaNaxProbeFailed() bool {
    return qsa_nax_probe_state.load(.acquire) == 1;
}

pub fn qsaNaxProbeReset() void {
    qsa_nax_probe_state.store(0, .release);
}

pub fn qsaNaxKernelBuilds() usize {
    return qsa_nax_kernel_builds;
}

pub fn qsaNaxArm() void {
    if (qsa_nax_probe_fail_override) {
        qsa_nax_probe_state.store(1, .release);
        return;
    }
    if (qsa_nax_probe_state.load(.acquire) != 0) return;
    if (qsa_nax_override) |v| {
        if (!v) return;
    } else {
        const raw = std.c.getenv("SUSHI_QSA_NAX");
        if (!qsaNaxEnabledFrom(if (raw) |r| std.mem.sliceTo(r, 0) else null)) return;
    }
    if (!verifyQmmNaxAvailable()) return;
    if (!qsaNaxOsOk()) return;
    const ok = qsaNaxRunProbe();
    qsa_nax_probe_state.store(if (ok) 2 else 1, .release);
}

const QSA_NAX_PROBE_FLOOR: f32 = 4.9e-4;

pub fn qsaProbeBitEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    if (mlx.mlx_array_equal(&eq, a, b, false, s) != 0) return false;
    if (mlx.mlx_array_eval(eq) != 0) return false;
    var ok: bool = false;
    if (mlx.mlx_array_item_bool(&ok, eq) != 0) return false;
    return ok;
}

fn qsaGatherProbeDispatch(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    nax: bool,
    packed_kv: ?*const [2]kv_quant.QuantizedKV,
    bits: u8,
) ?mlx.mlx_array {
    const kernel = (if (packed_kv != null) getQsaGatherPackedKernel(nax) else if (nax) getQsaNaxKernel() else getAttnQsa256Kernel()) catch return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const bk: c_int = QSA_GATHER_BK;
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_set_grid(config, qs[2] * 32, ks[1] * nsg, qs[0]) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio) != 0) return null;
    var inputs_arr = [_]mlx.mlx_array{ q, k, v, scl, blocks, .{}, .{}, .{}, .{} };
    var n_inputs: usize = 5;
    if (packed_kv) |p| {
        if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", bits) != 0) return null;
        if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", 64) != 0) return null;
        inputs_arr = .{ q, p[0].q, p[1].q, scl, blocks, p[0].scales, p[0].biases, p[1].scales, p[1].biases };
        n_inputs = inputs_arr.len;
    }
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, n_inputs);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s) != 0) return null;
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return null;
    var out = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&out, outputs_vec, 0) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    if (mlx.mlx_array_eval(out) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

fn qsaNaxRunProbe() bool {
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return false;
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 24, 16, 256 }, 0x9A7E016) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const k = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, 16, 256 }, 0x51A2002) orelse return false;
    defer _ = mlx.mlx_array_free(k);
    const v = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, 16, 256 }, 0xC0FFEE3) orelse return false;
    defer _ = mlx.mlx_array_free(v);
    var blk: [16 * 512]i32 = undefined;
    for (0..16) |r| {
        const complete = (r + 1) / 4;
        for (0..512) |b| blk[r * 512 + b] = if (b < complete) @intCast(b) else std.math.maxInt(i32);
    }
    const bshape = [_]c_int{ 1, 16, 512 };
    const blocks = mlx.mlx_array_new_data(&blk, &bshape, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;
    const ratio: c_int = 4;
    const stock = qsaGatherProbeDispatch(s, q, k, v, scale, blocks, ratio, false, null, 0) orelse return false;
    defer _ = mlx.mlx_array_free(stock);
    var nax = qsaGatherProbeDispatch(s, q, k, v, scale, blocks, ratio, true, null, 0) orelse return false;
    defer _ = mlx.mlx_array_free(nax);
    if (qsa_nax_probe_flip_override) {
        const flipped = qsaProbeCorruptFirst(s, nax) orelse return false;
        _ = mlx.mlx_array_free(nax);
        nax = flipped;
    }
    const ref = qsaGatherF32Reference(q, k, v, scale, blocks, ratio, s) catch return false;
    defer _ = mlx.mlx_array_free(ref);
    if (!qsaProbeAllFinite(s, nax) or !qsaProbeAllFinite(s, stock) or !qsaProbeAllFinite(s, ref)) {
        log.info("[qsa-gather] disabled: probe mismatch (non-finite)\n", .{});
        return false;
    }
    const max_stock = qsaProbeMaxAbsDiff(s, stock, ref) orelse return false;
    const max_nax = qsaProbeMaxAbsDiff(s, nax, ref) orelse return false;
    if (!(max_nax <= @max(1.5 * max_stock, QSA_NAX_PROBE_FLOOR))) {
        log.info("[qsa-gather] disabled: probe mismatch (nax vs stock)\n", .{});
        return false;
    }
    return qsaNaxRunPackedProbe(s, q, k, v, scale, blocks, ratio);
}

/// The packed variant is a JIT of its own per width: the same tile, read from an N-bit cache.
fn qsaNaxRunPackedProbe(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, blocks: mlx.mlx_array, ratio: c_int) bool {
    for ([_]u8{ 8, 6 }) |bits| {
        if (!qsaNaxRunPackedProbeAt(s, q, k, v, scale, blocks, ratio, bits)) return false;
    }
    return true;
}

fn qsaNaxRunPackedProbeAt(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, blocks: mlx.mlx_array, ratio: c_int, bits: u8) bool {
    var packed_kv: [2]kv_quant.QuantizedKV = undefined;
    packed_kv[0] = kv_quant.quantizeAffine(s, k, 64, bits) catch return false;
    defer packed_kv[0].deinit();
    packed_kv[1] = kv_quant.quantizeAffine(s, v, 64, bits) catch return false;
    defer packed_kv[1].deinit();
    const kd = kv_quant.dequantizeAffine(s, packed_kv[0].q, packed_kv[0].scales, packed_kv[0].biases, 64, bits) catch return false;
    defer _ = mlx.mlx_array_free(kd);
    const vd = kv_quant.dequantizeAffine(s, packed_kv[1].q, packed_kv[1].scales, packed_kv[1].biases, 64, bits) catch return false;
    defer _ = mlx.mlx_array_free(vd);
    const stock = qsaGatherProbeDispatch(s, q, packed_kv[0].q, packed_kv[1].q, scale, blocks, ratio, false, &packed_kv, bits) orelse return false;
    defer _ = mlx.mlx_array_free(stock);
    var nax = qsaGatherProbeDispatch(s, q, packed_kv[0].q, packed_kv[1].q, scale, blocks, ratio, true, &packed_kv, bits) orelse return false;
    defer _ = mlx.mlx_array_free(nax);
    if (qsa_nax_probe_flip_packed_bits == bits) {
        const flipped = qsaProbeCorruptFirst(s, nax) orelse return false;
        _ = mlx.mlx_array_free(nax);
        nax = flipped;
    }
    const ref = qsaGatherF32Reference(q, kd, vd, scale, blocks, ratio, s) catch return false;
    defer _ = mlx.mlx_array_free(ref);
    if (!qsaProbeAllFinite(s, nax) or !qsaProbeAllFinite(s, stock) or !qsaProbeAllFinite(s, ref)) {
        log.info("[qsa-gather] disabled: packed probe mismatch at {d} bits (non-finite)\n", .{bits});
        return false;
    }
    const max_stock = qsaProbeMaxAbsDiff(s, stock, ref) orelse return false;
    const max_nax = qsaProbeMaxAbsDiff(s, nax, ref) orelse return false;
    if (!(max_nax <= @max(1.5 * max_stock, QSA_NAX_PROBE_FLOOR))) {
        log.info("[qsa-gather] disabled: packed probe mismatch at {d} bits (nax vs stock)\n", .{bits});
        return false;
    }
    return true;
}

fn getQsaNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (!verifyQmmNaxAvailable()) return error.NaxUnavailable;
    if (qsa_nax_kernel_cached) |kernel| return kernel;
    const names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks" };
    const outputs = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qsa_nax_precise",
        in_vec,
        out_vec,
        @embedFile("../kernels/qsa_nax.metal"),
        @embedFile("../kernels/qsa_nax_header.metal"),
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_nax_kernel_cached = kernel;
    qsa_nax_kernel_builds += 1;
    return kernel;
}

/// Prepended to a gather kernel's header to build its packed-cache variant: each staged row
/// chunk of 8 is dequantized from the cache's affine words (the qsaSparseAttn arithmetic).
const QSA_PACKED_HEADER = "#define QSA_PACKED 1\n" ++ KV_UNPACK_HEADER ++
    \\template <typename T, int BITS, int GS>
    \\inline uint4 sushi_qsa_unpack8(const device uint32_t* wq, const device T* sc, const device T* bi, int c8) {
    \\  const int g = (c8 * 8) / GS;
    \\  uint4 out = uint4(0);
    \\  sushi_kv_unpack8<T, BITS>(wq, c8, float(sc[g]), float(bi[g]), (thread T*)&out);
    \\  return out;
    \\}
    \\
;

/// [stock, nax]
var qsa_gather_packed_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

fn getQsaGatherPackedKernel(nax: bool) !mlx.mlx_fast_metal_kernel {
    if (nax and !verifyQmmNaxAvailable()) return error.NaxUnavailable;
    const slot = &qsa_gather_packed_kernels[@intFromBool(nax)];
    if (slot.*) |kernel| return kernel;
    const names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks", "ksc", "kbi", "vsc", "vbi" };
    const outputs = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const name: [*:0]const u8 = if (nax) "sushi_qsa_nax_packed" else "sushi_attn_qsa256_packed";
    const source: [*:0]const u8 = if (nax) @embedFile("../kernels/qsa_nax.metal") else ATTN_QSA256_KERNEL_SOURCE;
    const header: [*:0]const u8 = if (nax) QSA_PACKED_HEADER ++ @embedFile("../kernels/qsa_nax_header.metal") else QSA_PACKED_HEADER ++ ATTN_QSA256_KERNEL_HEADER;
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, false, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = kernel;
    return kernel;
}

const QsaGatherCfgKey = struct {
    q_shape: ShapeKey,
    h_kv: c_int,
    nsg: c_int,
    bk: c_int,
    ratio: c_int,
    nax: bool,
    dtype: mlx.mlx_dtype,
    /// 0 = dense bf16 K/V.
    packed_bits: u8 = 0,
    packed_gs: u32 = 0,
};
var qsa_gather_cfgs = QsaCfgCache(QsaGatherCfgKey, 1){};
var qsa_gather_packed_engaged_bits: OneShotBits = .{};

/// Block-gathered QSA attention (qwen4_exp prefill): `blocks` is the
/// [B, S, KB] int32 selection, each row ascending with INT_MAX past its
/// count; `ratio` tokens per block. Same envelope as the p256 kernel (hd
/// 256, bf16, q <= k). Null = declined (kill switch, dtype, geometry) — the
/// caller expands the selection to the dense mask arm.
pub fn gatherQsa256(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
) !?mlx.mlx_array {
    return gatherQsa256Impl(s, q, .{ .k = k, .v = v, .owned = false }, scale, blocks, ratio);
}

/// `gatherQsa256` straight off a PACKED cache (`view.has_quant_triple`): the kernel stages only
/// the rows each query gathers, so no layer attends a dense rebuild of the whole cache.
pub fn gatherQsa256Packed(s: mlx.mlx_stream, q: mlx.mlx_array, view: *const DenseKVView, scale: f32, blocks: mlx.mlx_array, ratio: c_int) !?mlx.mlx_array {
    if (!view.has_quant_triple or !qsaAttnKernelEnabled()) return null;
    return gatherQsa256Impl(s, q, view.*, scale, blocks, ratio);
}

fn gatherQsa256Impl(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: DenseKVView,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
) !?mlx.mlx_array {
    if (!qsaGatherEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const is_packed = view.has_quant_triple;
    const k = if (is_packed) view.k_triple_q else view.k;
    const v = if (is_packed) view.v_triple_q else view.v;
    if (blocks.ctx == null or mlx.mlx_array_ndim(blocks) != 3 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const bs = mlx.getShape(blocks);
    if (is_packed) {
        if (!KVQuantConfig.supportedBits(view.bits)) return null;
        if (view.group_size == 0 or 256 % view.group_size != 0 or view.group_size % 8 != 0) return null;
        const g: c_int = @intCast(256 / view.group_size);
        for ([_]mlx.mlx_array{ view.k_triple_scales, view.k_triple_biases, view.v_triple_scales, view.v_triple_biases }) |a| {
            if (a.ctx == null or mlx.mlx_array_ndim(a) != 4 or mlx.mlx_array_dtype(a) != .bfloat16) return null;
            const sh = mlx.getShape(a);
            if (sh[0] != ks[0] or sh[1] != ks[1] or sh[2] != ks[2] or sh[3] != g or mlx.mlx_array_strides(a)[3] != 1) return null;
        }
        if (mlx.mlx_array_dtype(k) != .uint32 or mlx.mlx_array_dtype(v) != .uint32) return null;
        if (mlx.mlx_array_strides(k)[3] != 1 or mlx.mlx_array_strides(v)[3] != 1) return null;
    } else if (mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16 or !attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;
    const row_cols: c_int = if (is_packed) kv_quant.packedWords(256, view.bits) else 256;
    if (qs[3] != 256 or ks[3] != row_cols or vs[3] != row_cols) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    if (gqa > 64) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (bs[0] != qs[0] or bs[1] != qs[2] or bs[2] <= 0) return null;
    if (ratio <= 0) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16) return null;

    qsaNaxArm();
    const qst = mlx.mlx_array_strides(q);
    const kst = mlx.mlx_array_strides(k);
    const vst = mlx.mlx_array_strides(v);
    const bst = mlx.mlx_array_strides(blocks);
    const nax_aligned = qst[3] == 1 and kst[3] == 1 and vst[3] == 1 and bst[2] == 1 and
        qst[1] % 8 == 0 and qst[2] % 8 == 0 and
        kst[1] % 8 == 0 and kst[2] % 8 == 0 and
        vst[1] % 8 == 0 and vst[2] % 8 == 0;
    const nax_geometry = is_packed or (nax_aligned and ratio == 4 and qs[0] == 1 and qs[1] == 24 and ks[1] == 2 and bs[2] == 512);
    var use_nax = nax_geometry and qsaNaxEnabled() and qsaNaxEligible(.bfloat16, .bfloat16, .bfloat16, qs[3], gqa, qs[2]);
    const kernel = blk: {
        if (use_nax) {
            break :blk (if (is_packed) getQsaGatherPackedKernel(true) else getQsaNaxKernel()) catch {
                qsa_nax_probe_state.store(1, .release);
                use_nax = false;
                break :blk (if (is_packed) getQsaGatherPackedKernel(false) else getAttnQsa256Kernel()) catch return null;
            };
        }
        break :blk (if (is_packed) getQsaGatherPackedKernel(false) else getAttnQsa256Kernel()) catch return null;
    };
    qsa_gather_used_nax = use_nax;
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const bk: c_int = QSA_GATHER_BK;
    const key = QsaGatherCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = ks[1],
        .nsg = nsg,
        .bk = bk,
        .ratio = ratio,
        .nax = use_nax,
        .dtype = .bfloat16,
        .packed_bits = if (is_packed) view.bits else 0,
        .packed_gs = if (is_packed) view.group_size else 0,
    };
    const cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_gather_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, qs[2] * 32, ks[1] * nsg, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio));
        if (is_packed) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(view.bits)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(view.group_size)));
        }
        qsa_gather_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const dense_in = [_]mlx.mlx_array{ q, k, v, scl, blocks };
    const packed_in = [_]mlx.mlx_array{ q, k, v, scl, blocks, view.k_triple_scales, view.k_triple_biases, view.v_triple_scales, view.v_triple_biases };
    const inputs: []const mlx.mlx_array = if (is_packed) &packed_in else &dense_in;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (is_packed) {
        if (qsa_gather_packed_engaged_bits.take(@intFromBool(use_nax))) {
            log.info("[qsa-gather] engaged: {s} S={d} kv={d} blocks={d} kv{d}/gs{d}\n", .{ if (use_nax) "sushi_qsa_nax_packed" else "sushi_attn_qsa256_packed", qs[2], ks[2], bs[2], view.bits, view.group_size });
        }
        return out;
    }
    const engaged_bit: u5 = qsaEngagedBit(.prefill_gather, qs[2]) + @as(u5, if (use_nax) 16 else 0);
    if (qwen4_qsa_attn.qsa_engaged_bits.take(engaged_bit)) {
        const tgmem: usize = if (use_nax)
            2 * 8 * 32 * 4
        else
            @as(usize, @intCast(bk + 8)) * 256 * 2;
        if (use_nax) {
            qsa_gather_engaged_nax += 1;
            log.info("[qsa-gather] engaged: sushi_qsa_nax_precise S={d} kv={d} blocks={d} bk={d} tgmem={d}\n", .{ qs[2], ks[2], bs[2], bk, tgmem });
        } else {
            qsa_gather_engaged_stock += 1;
            log.info("[qsa-gather] engaged: sushi_attn_qsa256 S={d} kv={d} blocks={d} bk={d} tgmem={d}\n", .{ qs[2], ks[2], bs[2], bk, tgmem });
        }
    }
    return out;
}

pub fn qsaDecodeGatherIdsHost(
    alloc: std.mem.Allocator,
    blocks: []const i32,
    kb: c_int,
    kv: c_int,
    ratio: c_int,
) ![]i32 {
    const kbu: usize = @intCast(kb);
    if (blocks.len != kbu) return error.QsaDecodeBadSelection;
    if (ratio <= 0 or kb <= 0 or kv <= 0) return error.QsaDecodeGeometry;
    const nb: c_int = @divTrunc(kv, ratio);
    const tail_start: c_int = nb * ratio;
    const tail_len: c_int = kv - tail_start;
    const sel_len: c_int = kb * ratio;
    const gathered_len: c_int = sel_len + tail_len;
    const rows = try alloc.alloc(i32, @intCast(gathered_len));
    var r: usize = 0;
    for (0..kbu) |c| {
        const b = blocks[c];
        const use = b >= 0 and b < nb;
        const base: i32 = if (use) b else 0;
        var j: c_int = 0;
        while (j < ratio) : (j += 1) {
            rows[r] = base * ratio + j;
            r += 1;
        }
    }
    var t: c_int = tail_start;
    while (t < kv) : (t += 1) {
        rows[r] = t;
        r += 1;
    }
    return rows;
}

/// A materialized `[B,H,T,264]` padded cache sliced on the last axis to `[lo, lo+256)`: a 256-wide view whose
/// row stride passes the 16-byte stride gates while its base sits `lo*2` bytes into the buffer.
fn attn256SliceLastDim(parent: mlx.mlx_array, lo: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const ps = mlx.getShape(parent);
    const one = [_]c_int{ 1, 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const stop = [_]c_int{ ps[0], ps[1], ps[2], lo + 256 };
    try mlx.check(mlx.mlx_slice(&out, parent, &[_]c_int{ 0, 0, 0, lo }, 4, &stop, 4, &one, 4, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn attn256Contiguous(a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, a, false, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A sorted qwen4 block selection [1,qL,kb] int32 for rows straddling the
/// "every block fits" boundary (INT_MAX past each row's count), plus the
/// equivalent dense [1,1,qL,kL] mask the composed reference consumes.
pub const QsaBlockFixture = struct {
    blocks: mlx.mlx_array,
    mask: mlx.mlx_array,
    pub fn build(rnd: std.Random, qL: c_int, kL: c_int, kb: c_int, ratio: c_int) !QsaBlockFixture {
        const ta = std.testing.allocator;
        const nq: usize = @intCast(qL);
        const nk: usize = @intCast(kL);
        const nkb: usize = @intCast(kb);
        const blocks = try ta.alloc(i32, nq * nkb);
        defer ta.free(blocks);
        const mask = try ta.alloc(bool, nq * nk);
        defer ta.free(mask);
        const picked = try ta.alloc(bool, @intCast(@divTrunc(kL, ratio) + 1));
        defer ta.free(picked);
        const q_off = kL - qL;
        for (0..nq) |r| {
            const p: c_int = q_off + @as(c_int, @intCast(r));
            const complete: c_int = @divTrunc(p + 1, ratio);
            const count: c_int = @min(complete, kb);
            @memset(picked, false);
            var n: c_int = 0;
            while (n < count) {
                const b: usize = rnd.uintLessThan(usize, @intCast(complete));
                if (picked[b]) continue;
                picked[b] = true;
                n += 1;
            }
            var w: usize = 0;
            for (0..@intCast(complete)) |b| if (picked[b]) {
                blocks[r * nkb + w] = @intCast(b);
                w += 1;
            };
            while (w < nkb) : (w += 1) blocks[r * nkb + w] = std.math.maxInt(i32);
            for (0..nk) |c| {
                const ci: c_int = @intCast(c);
                mask[r * nk + c] = ci <= p and (ci >= complete * ratio or picked[@intCast(@divTrunc(ci, ratio))]);
            }
        }
        const bshape = [_]c_int{ 1, qL, kb };
        const mshape = [_]c_int{ 1, 1, qL, kL };
        return .{
            .blocks = mlx.mlx_array_new_data(blocks.ptr, &bshape, 3, .int32),
            .mask = mlx.mlx_array_new_data(mask.ptr, &mshape, 4, .bool_),
        };
    }
    pub fn deinit(self: *QsaBlockFixture) void {
        _ = mlx.mlx_array_free(self.blocks);
        _ = mlx.mlx_array_free(self.mask);
    }
};

fn qsaNaxAssertNoWorseThanStock(
    stream: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    floor: f32,
) !void {
    const ref = try qsaGatherF32Reference(q, k, v, scale, blocks, ratio, stream);
    defer _ = mlx.mlx_array_free(ref);
    qsa_nax_override = false;
    const stock = (try gatherQsa256(stream, q, k, v, scale, blocks, ratio)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_override = true;
    const nax = (try gatherQsa256(stream, q, k, v, scale, blocks, ratio)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(nax);
    try std.testing.expect(qsa_gather_used_nax);
    try std.testing.expect(qsaProbeAllFinite(stream, nax));
    try std.testing.expect(qsaProbeAllFinite(stream, stock));
    try std.testing.expect(qsaProbeAllFinite(stream, ref));
    const max_stock = try attn256MaxDiff(stock, ref, stream);
    const max_nax = try attn256MaxDiff(nax, ref, stream);
    errdefer std.debug.print("max_stock={e} max_nax={e} floor={e} bar={e}\n", .{ max_stock, max_nax, floor, max_stock });
    try std.testing.expect(max_nax <= max_stock);
}

/// `attn256Reference` a slab of query rows at a time: an hd-256 array mask has no fused sdpa
/// arm, and a prefill-width score sheet would not fit a unit test.
fn attn256ReferenceRows(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: mlx.mlx_array, rows: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const kl = mlx.getShape(k)[2];
    var parts: [64]mlx.mlx_array = undefined;
    var n: usize = 0;
    defer for (parts[0..n]) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var r0: c_int = 0;
    while (r0 < qs[2]) : (r0 += rows) {
        const r1 = @min(qs[2], r0 + rows);
        const qr = try sliceAttentionSeq(s, q, r0, r1);
        defer _ = mlx.mlx_array_free(qr);
        var mr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mr);
        try mlx.check(mlx.mlx_slice(&mr, mask, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, 1, r1, kl }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        parts[n] = try attn256Reference(qr, k, v, scale, "array", mr, s);
        n += 1;
        try mlx.check(mlx.mlx_array_eval(parts[n - 1]));
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    return out;
}

const QsaUbenchArm = enum { empty, old, sparse16, packed_nax, packed_stock, gather_rebuild, mask_rebuild };

/// 12 dependent layers of kv8 QSA attention at Flash-Next shapes: each layer feeds its output to
/// the next layer's query and rebuilds its own dequant graph, as a forward does. Prints ms per
/// forward minus the empty chain. `old` = the pre-packed route: split-K below 16 rows, else the
/// whole-cache rebuild into the gather (kv past its floor) or the mask arm.
fn qsaWideUbench(kvs: []const c_int, widths: []const c_int, arms_run: []const QsaUbenchArm, comptime reps: usize) !void {
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var prng = std.Random.DefaultPrng.init(0x51DE_0001);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 512;
    const layers: usize = 12;
    const scale: f32 = 1.0 / 16.0;
    qwen4_qsa_attn.qsa_attn_packed_max_s_override = 1 << 20;
    defer qwen4_qsa_attn.qsa_attn_packed_max_s_override = null;
    defer qsa_nax_override = null;

    const Arm = QsaUbenchArm;
    const Chain = struct {
        view: *const DenseKVView,
        blocks: mlx.mlx_array,

        fn rebuilt(c: @This(), st: mlx.mlx_stream, q: mlx.mlx_array, gather: bool) !mlx.mlx_array {
            const v = c.view;
            const k = try kv_quant.dequantizeAffine(st, v.k_triple_q, v.k_triple_scales, v.k_triple_biases, v.group_size, v.bits);
            defer _ = mlx.mlx_array_free(k);
            const vv = try kv_quant.dequantizeAffine(st, v.v_triple_q, v.v_triple_scales, v.v_triple_biases, v.group_size, v.bits);
            defer _ = mlx.mlx_array_free(vv);
            const kv = mlx.getShape(k)[2];
            if (gather) return (try gatherQsa256(st, q, k, vv, scale, c.blocks, 4)) orelse error.GatherDeclined;
            const mask = try qsaMaskFromBlocks(st, c.blocks, kv, 4);
            defer _ = mlx.mlx_array_free(mask);
            if (try fusedSdpa256Masked(st, q, k, vv, scale, mask)) |o| return o;
            if (try splitMaskedSdpa256(st, q, k, vv, scale, mask)) |o| return o;
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k, vv, scale, "array", mask, .{ .ctx = null }, false, st));
            return out;
        }

        fn layer(c: @This(), st: mlx.mlx_stream, q: mlx.mlx_array, arm: Arm) !mlx.mlx_array {
            qsa_nax_override = null;
            switch (arm) {
                .empty => {
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&out, q, q, st));
                    return out;
                },
                .old => {
                    if (mlx.getShape(q)[2] < FUSED256_MIN_Q_LEN)
                        return (try qsaSparseAttn(st, q, c.view, c.blocks, 4, scale)) orelse error.SparseAttnDeclined;
                    return c.rebuilt(st, q, mlx.getShape(c.view.k_triple_q)[2] > qsaGatherMinKv());
                },
                .gather_rebuild => return c.rebuilt(st, q, true),
                .mask_rebuild => return c.rebuilt(st, q, false),
                .sparse16 => return (try qsaSparseAttn(st, q, c.view, c.blocks, 4, scale)) orelse error.SparseAttnDeclined,
                .packed_nax, .packed_stock => {
                    qsa_nax_override = arm == .packed_nax;
                    return (try gatherQsa256Packed(st, q, c.view, scale, c.blocks, 4)) orelse error.PackedGatherDeclined;
                },
            }
        }

        fn run(c: @This(), st: mlx.mlx_stream, q0: mlx.mlx_array, n: usize, arm: Arm) !mlx.mlx_array {
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&q, q0));
            for (0..n) |_| {
                const next = c.layer(st, q, arm) catch |e| {
                    _ = mlx.mlx_array_free(q);
                    return e;
                };
                _ = mlx.mlx_array_free(q);
                q = next;
            }
            return q;
        }
    };

    std.debug.print("\n[qsa-wide] kv8 gs64, 24/2 hd256, ratio 4, kb 512, {d} dependent layers; median of {d}, ms per forward minus the empty chain\n[qsa-wide]      kv     S", .{ layers, reps });
    for (arms_run) |arm| std.debug.print(" {s:>14}", .{@tagName(arm)});
    std.debug.print("\n", .{});
    for (kvs) |kv| {
        const kd = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
        defer _ = mlx.mlx_array_free(kd);
        const vd = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
        defer _ = mlx.mlx_array_free(vd);
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
        defer view.deinit();
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for ([_]mlx.mlx_array{ view.k_triple_q, view.v_triple_q, view.k_triple_scales, view.v_triple_scales, view.k_triple_biases, view.v_triple_biases }) |a| _ = mlx.mlx_vector_array_append_value(ev, a);
            try mlx.check(mlx.mlx_eval(ev));
        }
        for (widths) |sq| {
            if (sq > kv) continue;
            const bh = try qsaVerifyBlocksHost(ta, rnd, sq, kv, kb, ratio);
            defer ta.free(bh);
            const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, sq, kb }, 3, .int32);
            defer _ = mlx.mlx_array_free(blocks);
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, sq, 256 }, s);
            defer _ = mlx.mlx_array_free(q);
            const chain = Chain{ .view = &view, .blocks = blocks };
            var samples: [8][reps]f64 = undefined;
            for (0..reps + 2) |r| {
                for (0..arms_run.len + 1) |ai| {
                    const arm: Arm = if (ai == 0) .empty else arms_run[ai - 1];
                    const t0 = std.Io.Timestamp.now(io, .boot);
                    const out = try chain.run(s, q, layers, arm);
                    defer _ = mlx.mlx_array_free(out);
                    try mlx.check(mlx.mlx_array_eval(out));
                    const ns: f64 = @floatFromInt(t0.untilNow(io, .boot).nanoseconds);
                    if (r >= 2) samples[ai][r - 2] = ns / 1e6;
                }
            }
            var med: [8]f64 = undefined;
            for (0..arms_run.len + 1) |ai| {
                std.mem.sort(f64, &samples[ai], {}, std.sort.asc(f64));
                med[ai] = samples[ai][reps / 2];
            }
            std.debug.print("[qsa-wide] {d:>7} {d:>5}", .{ kv, sq });
            for (1..arms_run.len + 1) |ai| std.debug.print(" {d:>14.2}", .{med[ai] - med[0]});
            std.debug.print("\n", .{});
        }
        _ = mlx.mlx_clear_cache();
    }
}

test "fusedSdpa256Masked: QSA bool-mask parity vs composed 'array' SDPA (GQA, ragged tiles, kv chunking)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    trf.fused256_override = true;
    defer trf.fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x45a45a);
    const rnd = prng.random();

    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpa256Masked(s, q, k, v, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(fused, ref, s) < 0.005);

    // Wrong mask shape declines.
    const bad_shape = [_]c_int{ 1, 1, 70, 192 };
    const bad = try attn256RandBf16(rnd, &bad_shape, s);
    defer _ = mlx.mlx_array_free(bad);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, bad)) == null);
}

test "fusedSdpa256Masked: an unaligned last-axis slice declines and the fallback reads exactly; aligned is bit-identical to a contiguous copy" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    trf.fused256_override = true;
    defer trf.fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x5E7_5E7);
    const rnd = prng.random();

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 6, 70, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 193, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    const mask = try attn256QsaMask(rnd, 70, 193);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    const k = try attn256SliceLastDim(k_pad, 3, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256SliceLastDim(v_pad, 3, s);
    defer _ = mlx.mlx_array_free(v);
    try std.testing.expect((try fusedSdpa256Masked(s, q, k, v, scale, mask)) == null);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const kc = try attn256Contiguous(k, s);
    defer _ = mlx.mlx_array_free(kc);
    const vc = try attn256Contiguous(v, s);
    defer _ = mlx.mlx_array_free(vc);
    const on_copy = (try fusedSdpa256Masked(s, q, kc, vc, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(on_copy);
    try std.testing.expect(try attn256MaxDiff(on_copy, ref, s) < 0.005);

    const k_al = try attn256SliceLastDim(k_pad, 0, s);
    defer _ = mlx.mlx_array_free(k_al);
    const v_al = try attn256SliceLastDim(v_pad, 0, s);
    defer _ = mlx.mlx_array_free(v_al);
    const fast = (try fusedSdpa256Masked(s, q, k_al, v_al, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fast);
    const kac = try attn256Contiguous(k_al, s);
    defer _ = mlx.mlx_array_free(kac);
    const vac = try attn256Contiguous(v_al, s);
    defer _ = mlx.mlx_array_free(vac);
    const twin = (try fusedSdpa256Masked(s, q, kac, vac, scale, mask)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(twin);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(fast, twin, s));
}

test "gatherQsa256: an unaligned last-axis slice declines and the fallback reads exactly; aligned is bit-identical to a contiguous copy" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    qsa_nax_override = false;
    defer qsa_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x0FF_5E7);
    const rnd = prng.random();

    const qL: c_int = 40;
    const kL: c_int = 101;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, qL, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kL, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kL, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    var fx = try QsaBlockFixture.build(rnd, qL, kL, 512, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const k = try attn256SliceLastDim(k_pad, 3, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256SliceLastDim(v_pad, 3, s);
    defer _ = mlx.mlx_array_free(v);
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) == null);
    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const kc = try attn256Contiguous(k, s);
    defer _ = mlx.mlx_array_free(kc);
    const vc = try attn256Contiguous(v, s);
    defer _ = mlx.mlx_array_free(vc);
    const on_copy = (try gatherQsa256(s, q, kc, vc, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(on_copy);
    try std.testing.expect(try attn256MaxDiff(on_copy, ref, s) < 0.005);

    const k_al = try attn256SliceLastDim(k_pad, 0, s);
    defer _ = mlx.mlx_array_free(k_al);
    const v_al = try attn256SliceLastDim(v_pad, 0, s);
    defer _ = mlx.mlx_array_free(v_al);
    const fast = (try gatherQsa256(s, q, k_al, v_al, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fast);
    const kac = try attn256Contiguous(k_al, s);
    defer _ = mlx.mlx_array_free(kac);
    const vac = try attn256Contiguous(v_al, s);
    defer _ = mlx.mlx_array_free(vac);
    const twin = (try gatherQsa256(s, q, kac, vac, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(twin);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(fast, twin, s));
}

test "qsa prefill floor: NAX serving, widths, unavailable hardware and explicit overrides" {
    const expect = std.testing.expectEqual;
    try expect(@as(c_int, 2052), qsaPrefillGatherMinKvFrom(16, true, null));
    try expect(@as(c_int, 2052), qsaPrefillGatherMinKvFrom(8192, true, null));
    for ([_]c_int{ 1, 2, 4, 15 }) |width| {
        try expect(@as(c_int, 8192), qsaPrefillGatherMinKvFrom(width, true, null));
    }
    try expect(@as(c_int, 8192), qsaPrefillGatherMinKvFrom(16, false, null));
    try expect(@as(c_int, 4096), qsaPrefillGatherMinKvFrom(16, true, 4096));
}

test "qsaNaxEnabledFrom: on by default, SUSHI_QSA_NAX=0 restores the stock gather" {
    try std.testing.expect(qsaNaxEnabledFrom(null));
    try std.testing.expect(qsaNaxEnabledFrom("1"));
    try std.testing.expect(!qsaNaxEnabledFrom("0"));
}

test "qsaNaxEligible: kernel predicate names gqa 12, hd 256, q_len>=16, bf16, NAX, macOS 26.3" {
    const dt_bf16 = mlx.mlx_dtype.bfloat16;
    const dt_f32 = mlx.mlx_dtype.float32;
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.5", dt_bf16, dt_bf16, dt_bf16, 256, 12, 40));
    try std.testing.expect(!qsaNaxEligibleFrom(false, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.2", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_f32, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_f32, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_f32, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 128, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 8, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 16, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 15));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 1));
}

test "gatherQsa256 NAX: precise sparse attention and stock fallback" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x9a7e);
    const rnd = prng.random();

    // qwen4_exp geometry (24/2 heads, hd 256, ratio 4) with a tiny block
    // budget so the chunk's rows cross from "every block fits" to top-k.
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 6;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    for ([_]bool{ false, true }) |enabled| {
        qsa_nax_override = enabled;
        const out = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expect(try attn256MaxDiff(out, ref, s) < 0.005);
        try std.testing.expect(!qsa_gather_used_nax);
    }
}

test "gatherQsa256 NAX: max error vs f32 gather-softmax is no worse than stock" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.3", .bfloat16, .bfloat16, .bfloat16, 256, 12, 40));
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x51a2);
    const rnd = prng.random();
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    try qsaNaxAssertNoWorseThanStock(s, q, k, v, scale, fx.blocks, 4, 4.9e-4);
}

test "gatherQsa256 NAX: online softmax rescale across 32-key tiles vs f32 gather-softmax" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0ff);
    const rnd = prng.random();
    const qL: c_int = 40;
    const kL: c_int = 203;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k0 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k0);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const k = try qsaBoostLastTileK(q, k0, fx.blocks, 4, s);
    defer _ = mlx.mlx_array_free(k);
    const scale: f32 = 1.0 / 16.0;
    try qsaNaxAssertNoWorseThanStock(s, q, k, v, scale, fx.blocks, 4, 4.9e-4);
}

test "gatherQsa256 NAX: top-512 selection and partial causal tail vs f32 gather-softmax" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const saved_nax = qsa_nax_override;
    qsa_nax_override = true;
    defer qsa_nax_override = saved_nax;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x4020);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kb: c_int = 512;
    const qshape = [_]c_int{ 1, 24, qL, 256 };
    const q = try attn256RandBf16(rnd, &qshape, s);
    defer _ = mlx.mlx_array_free(q);
    for ([_]c_int{ 2053, 2153 }) |kL| {
        const kvshape = [_]c_int{ 1, 2, kL, 256 };
        const k = try attn256RandBf16(rnd, &kvshape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &kvshape, s);
        defer _ = mlx.mlx_array_free(v);
        var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
        defer fx.deinit();
        try qsaNaxAssertNoWorseThanStock(s, q, k, v, 1.0 / 16.0, fx.blocks, 4, 4.9e-4);
    }
}

test "gatherQsa256 NAX: probe failure latches stock gather bit-identical to the stock arm" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsaNaxProbeReset();
    }
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kL: c_int = 16;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const stock = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_probe_fail_override = true;
    qsaNaxProbeReset();
    qsa_nax_override = true;
    const fallback = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fallback);
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(stock, fallback, s));
}

test "gatherQsa256 NAX: a probe mismatch latches the stock gather" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_flip = qsa_nax_probe_flip_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_probe_flip_override = saved_flip;
        qsaNaxProbeReset();
    }
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kL: c_int = 16;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const stock = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_probe_fail_override = false;
    qsa_nax_probe_flip_override = true;
    qsaNaxProbeReset();
    qsa_nax_override = true;
    const fallback = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fallback);
    try std.testing.expect(qsaNaxProbeFailed());
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(stock, fallback, s));
}

test "gatherQsa256 NAX: a packed probe mismatch at 8 or at 6 bits latches the stock gather" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_flip = qsa_nax_probe_flip_packed_bits;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_probe_flip_packed_bits = saved_flip;
        qsaNaxProbeReset();
    }
    qsa_nax_probe_fail_override = false;
    qsa_nax_override = true;
    for ([_]?u8{ 8, 6, null }) |flipped| {
        qsa_nax_probe_flip_packed_bits = flipped;
        qsaNaxProbeReset();
        qsaNaxArm();
        try std.testing.expectEqual(flipped != null, qsaNaxProbeFailed());
    }
}

test "gatherQsa256 NAX: probe passes on this machine" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_flip = qsa_nax_probe_flip_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_probe_flip_override = saved_flip;
        qsaNaxProbeReset();
    }
    qsa_nax_probe_fail_override = false;
    qsa_nax_probe_flip_override = false;
    qsa_nax_override = true;
    qsaNaxProbeReset();
    qsaNaxArm();
    try std.testing.expect(!qsaNaxProbeFailed());
    try std.testing.expect(qsaNaxEnabled());
}

test "gatherQsa256 NAX: latch is consulted before NAX kernel construction" {
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_nax = qsa_nax_override;
    defer {
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_override = saved_nax;
        qsaNaxProbeReset();
    }
    qsa_nax_probe_fail_override = true;
    qsaNaxProbeReset();
    qsaNaxArm();
    try std.testing.expect(qsaNaxProbeFailed());
    try std.testing.expect(!qsaNaxEnabled());
    const builds = qsaNaxKernelBuilds();
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    qsa_nax_override = true;
    var prng = std.Random.DefaultPrng.init(0xb0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 512, 4);
    defer fx.deinit();
    const out = (try gatherQsa256(s, q, k, v, 1.0 / 16.0, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(builds, qsaNaxKernelBuilds());
}

test "gatherQsa256: declines a non-GPU stream before kernel selection" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expect(!mlx.streamIsGpu(cpu));
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, cpu);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, cpu);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, cpu);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 4, 4);
    defer fx.deinit();
    try std.testing.expect((try gatherQsa256(cpu, q, k, v, 1.0 / 16.0, fx.blocks, 4)) == null);
}

test "gatherQsa256: stock then NAX each fire one engaged line" {
    try mlx.requireGpu();
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    defer qsa_nax_override = saved_nax;
    qwen4_qsa_attn.qsa_engaged_bits.reset();
    qsa_gather_engaged_stock = 0;
    qsa_gather_engaged_nax = 0;
    var prng = std.Random.DefaultPrng.init(0xd0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 512, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const a = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(a);
    qsa_nax_override = true;
    qsa_nax_probe_fail_override = false;
    qsaNaxProbeReset();
    const b = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(b);
    try std.testing.expectEqual(@as(u32, 1), qsa_gather_engaged_stock);
    try std.testing.expectEqual(@as(u32, 1), qsa_gather_engaged_nax);
}

test "gatherQsa256: block-gathered QSA parity vs composed 'array' SDPA over the expanded mask (gqa 12, sentinel rows)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    qsa_nax_override = false;
    defer qsa_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x9a7e);
    const rnd = prng.random();

    // qwen4_exp geometry (24/2 heads, hd 256, ratio 4) with a tiny block
    // budget so the chunk's rows cross from "every block fits" to top-k.
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 6;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const out = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expect(try attn256MaxDiff(out, ref, s) < 0.005);

    // The production expansion (fallback arm + fixture trace) is the same mask.
    const expanded = try qsaMaskFromBlocks(s, fx.blocks, kL, 4);
    defer _ = mlx.mlx_array_free(expanded);
    var ours32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ours32);
    try mlx.check(mlx.mlx_astype(&ours32, expanded, .float32, s));
    var ref32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref32);
    try mlx.check(mlx.mlx_astype(&ref32, fx.mask, .float32, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(ours32, ref32, s));

    // Kill switch declines; so does a non-int32 selection.
    qsa_gather_override = false;
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) == null);
    qsa_gather_override = true;
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, ours32, 4)) == null);
}

test "qsa batched gather S=1: N=1 is byte-identical to solo, N=2 matches per-slot solos" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    qsa_decode_gather_override = true;
    defer qsa_decode_gather_override = null;
    qsa_batched_gather_override = true;
    defer qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xb07a);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const scale: f32 = 1.0 / 8.0;

    const kL0: c_int = 50;
    const kL1: c_int = 36;
    const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q1);
    const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(v0);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(v1);
    var fx0 = try QsaBlockFixture.build(rnd, 1, kL0, kb, ratio);
    defer fx0.deinit();
    var fx1 = try QsaBlockFixture.build(rnd, 1, kL1, kb, ratio);
    defer fx1.deinit();
    const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
    const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };

    const solo0 = (try qsaDecodeGatherAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo0);
    const bat1 = try qsaBatchedGatherAttn(ta, s, q0, &.{view0}, &.{fx0.blocks}, &.{}, ratio, scale, &.{}, 0);
    defer _ = mlx.mlx_array_free(bat1);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat1, solo0, s));

    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qparts = [_]mlx.mlx_array{ q0, q1 };
    const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
    const views = [_]DenseKVView{ view0, view1 };
    const blks = [_]mlx.mlx_array{ fx0.blocks, fx1.blocks };
    const bat2 = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &.{}, ratio, scale, &.{}, 0);
    defer _ = mlx.mlx_array_free(bat2);
    const solo1 = (try qsaDecodeGatherAttn(s, q1, &view1, fx1.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo1);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    const wparts = [_]mlx.mlx_array{ solo0, solo1 };
    const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
    defer _ = mlx.mlx_vector_array_free(wvec);
    try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat2, want, s));
}

test "qsa batched gather mixed group: a mask-only slot matches its solo masked SDPA" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    qsa_decode_gather_override = true;
    defer qsa_decode_gather_override = null;
    qsa_batched_gather_override = true;
    defer qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x51a0);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const scale: f32 = 1.0 / 8.0;
    const kL0: c_int = 50;
    const kL1: c_int = 36;
    const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    defer _ = mlx.mlx_array_free(q1);
    const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(k0);
    const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
    defer _ = mlx.mlx_array_free(v0);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
    defer _ = mlx.mlx_array_free(v1);
    var fx0 = try QsaBlockFixture.build(rnd, 1, kL0, kb, ratio);
    defer fx0.deinit();
    var fx1 = try QsaBlockFixture.build(rnd, 1, kL1, kb, ratio);
    defer fx1.deinit();
    const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
    const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };
    const solo0 = (try qsaDecodeGatherAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(solo0);
    const solo1 = try attn256Reference(q1, k1, v1, scale, "array", fx1.mask, s);
    defer _ = mlx.mlx_array_free(solo1);
    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qparts = [_]mlx.mlx_array{ q0, q1 };
    const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
    const views = [_]DenseKVView{ view0, view1 };
    const blks = [_]mlx.mlx_array{ fx0.blocks, .{ .ctx = null } };
    const masks = [_]mlx.mlx_array{ .{ .ctx = null }, fx1.mask };
    const bat = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &masks, ratio, scale, &.{}, 0);
    defer _ = mlx.mlx_array_free(bat);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    const wparts = [_]mlx.mlx_array{ solo0, solo1 };
    const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
    defer _ = mlx.mlx_vector_array_free(wvec);
    try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat, want, s));
}

test "qsaBatchedAttn: the kill switch refuses the gather arm" {
    try mlx.requireGpu();
    var xfm_bytes: [@sizeOf(Transformer)]u8 align(@alignOf(Transformer)) = @splat(0);
    const xfm: *Transformer = @ptrCast(&xfm_bytes);
    xfm.allocator = std.testing.allocator;
    var q4_bytes: [@sizeOf(qwen4_mod.Qwen4State)]u8 align(@alignOf(qwen4_mod.Qwen4State)) = @splat(0);
    xfm.qwen4 = @ptrCast(&q4_bytes);
    var cache_bytes: [@sizeOf(KVCache)]u8 align(@alignOf(KVCache)) = @splat(0);
    const cache: *KVCache = @ptrCast(&cache_bytes);
    var off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = cache, .moe_seq_offset = &off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var slot: ForwardCtx = .{ .cache = cache, .moe_seq_offset = &off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    const slots = [_]*ForwardCtx{&slot};
    const dummy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dummy);

    const prev_g = qsa_batched_gather_override;
    defer qsa_batched_gather_override = prev_g;
    qsa_batched_gather_override = false;
    try std.testing.expect((try xfm.qsaBatchedAttn(&ctx, &slots, dummy, 0, 4, 1.0)) == null);
}

test "qsaBatchedAttn: an M-RoPE slot takes the gather arm, byte-identical to the same slot without positions" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const prev_d = qsa_decode_gather_override;
    defer qsa_decode_gather_override = prev_d;
    qsa_decode_gather_override = true;
    const prev_b = qsa_batched_gather_override;
    defer qsa_batched_gather_override = prev_b;
    qsa_batched_gather_override = true;
    var prng = std.Random.DefaultPrng.init(0x3b0e);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 64;
    const kls = [_]c_int{ 50, 36 };

    var xfm_bytes: [@sizeOf(Transformer)]u8 align(@alignOf(Transformer)) = @splat(0);
    const xfm: *Transformer = @ptrCast(&xfm_bytes);
    xfm.allocator = ta;
    xfm.s = s;
    xfm.config.indexer_compress_ratio = @intCast(ratio);
    var q4_bytes: [@sizeOf(qwen4_mod.Qwen4State)]u8 align(@alignOf(qwen4_mod.Qwen4State)) = @splat(0);
    xfm.qwen4 = @ptrCast(&q4_bytes);

    var caches: [2]KVCache = undefined;
    var fxs: [2]QsaBlockFixture = undefined;
    var offs: [2]usize = undefined;
    var ctxs: [2]ForwardCtx = undefined;
    var qs: [2]mlx.mlx_array = undefined;
    for (0..2) |i| {
        caches[i] = try KVCache.init(ta, 1);
        const k = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kls[i], hd }, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kls[i], hd }, s);
        defer _ = mlx.mlx_array_free(v);
        var view = try caches[i].update(0, k, v, s, 128);
        view.deinit();
        fxs[i] = try QsaBlockFixture.build(rnd, 1, kls[i], kb, ratio);
        offs[i] = @intCast(kls[i]);
        ctxs[i] = .{ .cache = &caches[i], .moe_seq_offset = &offs[i], .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
        ctxs[i].qsa_blocks = fxs[i].blocks;
        qs[i] = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, 1, hd }, s);
    }
    defer for (0..2) |i| {
        caches[i].deinit();
        fxs[i].deinit();
        _ = mlx.mlx_array_free(qs[i]);
    };
    var q_n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_n2);
    const qvec = mlx.mlx_vector_array_new_data(&qs, 2);
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));

    const slots = [_]*ForwardCtx{ &ctxs[0], &ctxs[1] };
    const plain = (try xfm.qsaBatchedAttn(&ctxs[0], &slots, q_n2, 0, 1, 0.125)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(plain);
    const pos = [_]i32{ 0, 1, 2 };
    ctxs[1].mrope_pos = &pos;
    const with_pos = (try xfm.qsaBatchedAttn(&ctxs[0], &slots, q_n2, 0, 1, 0.125)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(with_pos);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(with_pos, plain, s));
}

test "qsaBatchedGatherOn matches the arm's switches" {
    const prev_b = qsa_batched_gather_override;
    const prev_g = qsa_gather_override;
    const prev_d = qsa_decode_gather_override;
    const prev_v = qwen4_qsa_attn.qsa_verify_gather_override;
    defer {
        qsa_batched_gather_override = prev_b;
        qsa_gather_override = prev_g;
        qsa_decode_gather_override = prev_d;
        qwen4_qsa_attn.qsa_verify_gather_override = prev_v;
    }
    qsa_batched_gather_override = true;
    qsa_gather_override = true;
    qsa_decode_gather_override = true;
    qwen4_qsa_attn.qsa_verify_gather_override = true;
    try std.testing.expect(qsaBatchedGatherOn(1));
    try std.testing.expect(qsaBatchedGatherOn(4));
    qsa_batched_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    qsa_batched_gather_override = true;
    qsa_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    qsa_gather_override = true;
    qsa_decode_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherOn(1));
    try std.testing.expect(qsaBatchedGatherOn(4));
    qsa_decode_gather_override = true;
    qwen4_qsa_attn.qsa_verify_gather_override = false;
    try std.testing.expect(qsaBatchedGatherOn(1));
    try std.testing.expect(!qsaBatchedGatherOn(4));
    qwen4_qsa_attn.qsa_verify_gather_override = true;
    const prev_st = qwen4_forward.qwen4_standin_override;
    defer qwen4_forward.qwen4_standin_override = prev_st;
    qwen4_forward.qwen4_standin_override = .{ .attn_qsa = true };
    try std.testing.expect(!qsaBatchedGatherOn(1));
    qwen4_forward.qwen4_standin_override = .{ .attn_sdpa = true };
    try std.testing.expect(!qsaBatchedGatherOn(1));
    qwen4_forward.qwen4_standin_override = .{};
    try std.testing.expect(qsaBatchedGatherOn(1));
}

test "qsaBatchedGatherEnabled: default on, override restores the dense padded mask" {
    const prev = qsa_batched_gather_override;
    defer qsa_batched_gather_override = prev;
    qsa_batched_gather_override = true;
    try std.testing.expect(qsaBatchedGatherEnabled());
    qsa_batched_gather_override = false;
    try std.testing.expect(!qsaBatchedGatherEnabled());
}

test "qsa packed gather: prefill widths read the packed cache as precisely as the gather over its dense rebuild" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x9A7E_D0A1);
    const rnd = prng.random();
    qsa_gather_override = true;
    defer qsa_gather_override = null;
    defer qsa_nax_override = null;

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, bits: u8 }{
        .{ .s = 16, .kv = 70001, .kb = 512, .bits = 8 },
        .{ .s = 64, .kv = 33333, .kb = 512, .bits = 4 },
        .{ .s = 256, .kv = 20002, .kb = 512, .bits = 8 },
        .{ .s = 1024, .kv = 12345, .kb = 512, .bits = 8 },
        .{ .s = 128, .kv = 20003, .kb = 512, .bits = 6 },
        .{ .s = 1024, .kv = 12347, .kb = 512, .bits = 6 },
        // Fewer visible blocks than the budget: the INT_MAX pads must never be read.
        .{ .s = 512, .kv = 4097, .kb = 2048, .bits = 4 },
        .{ .s = 512, .kv = 4099, .kb = 2048, .bits = 6 },
        // A first chunk: the early rows see every block they have.
        .{ .s = 3001, .kv = 3001, .kb = 512, .bits = 8 },
    };
    for (cases) |c| {
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, c.s, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(v_dense);
        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, c.s, c.kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(c.bits));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.affine(c.bits), &cache);
        defer view.deinit();
        const scale: f32 = 1.0 / 16.0;
        const ref = try attn256ReferenceRows(q, view.k, view.v, scale, mask, 64, s);
        defer _ = mlx.mlx_array_free(ref);
        const bar = try attn256UlpBar(ref, s);

        for ([_]bool{ false, true }) |nax| {
            if (nax and !(verifyQmmNaxAvailable() and qsaNaxOsOk())) continue;
            qsa_nax_override = nax;
            const old = (try gatherQsa256(s, q, view.k, view.v, scale, blocks, ratio)) orelse return error.GatherDeclined;
            defer _ = mlx.mlx_array_free(old);
            try std.testing.expectEqual(nax and c.kb == 512, qsa_gather_used_nax);
            const got = (try gatherQsa256Packed(s, q, &view, scale, blocks, ratio)) orelse return error.PackedGatherDeclined;
            defer _ = mlx.mlx_array_free(got);
            try std.testing.expectEqual(nax, qsa_gather_used_nax);
            const d_old = try attn256MaxDiff(old, ref, s);
            const d = try attn256MaxDiff(got, ref, s);
            const cs = try attn256Cosine(got, ref, s);
            errdefer std.debug.print("[qsa-packed-gather] nax={} S={d} kv={d} kb={d} bits={d}: max diff {d:.6} (dense rebuild {d:.6}) cos {d:.7} (bar {d:.6})\n", .{ nax, c.s, c.kv, c.kb, c.bits, d, d_old, cs, bar });
            try std.testing.expect(cs > 0.99999);
            try std.testing.expect(d <= bar);
            try std.testing.expect(d <= 1.5 * d_old + 1.0e-6);
        }
    }
}

test "qsa packed gather: serves while each packed row is staged at most 4 times (the measured crossover)" {
    // kb 512 blocks of 4: every query row stages 2048 selected keys.
    try std.testing.expect(qsaPackedGatherServes(64, 32768, 512, 4));
    try std.testing.expect(qsaPackedGatherServes(256, 131072, 512, 4));
    try std.testing.expect(!qsaPackedGatherServes(128, 32768, 512, 4));
    try std.testing.expect(!qsaPackedGatherServes(64, 8192, 512, 4));
    try std.testing.expect(!qsaPackedGatherServes(8192, 65536, 512, 4));
}

test "qsa wide µbench: verify and short-prefill widths (SUSHI_QSA_WIDE_UBENCH=1)" {
    if (!diagEnvOn("SUSHI_QSA_WIDE_UBENCH")) return error.SkipZigTest;
    try qsaWideUbench(&.{ 8192, 65536, 262144 }, &.{ 8, 12, 15, 16, 24, 32, 40, 48, 56, 64, 96, 128, 256, 512 }, &.{ .old, .sparse16, .packed_nax, .packed_stock }, 7);
}

test "qsa wide µbench: prefill chunk widths (SUSHI_QSA_PREFILL_UBENCH=1)" {
    if (!diagEnvOn("SUSHI_QSA_PREFILL_UBENCH")) return error.SkipZigTest;
    try qsaWideUbench(&.{ 4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072 }, &.{ 64, 128, 256, 512, 1024, 2048, 4096, 8192 }, &.{ .mask_rebuild, .gather_rebuild, .packed_nax }, 5);
}
